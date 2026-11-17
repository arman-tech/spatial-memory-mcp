//! `ambient_recall` (CONTRACTS 7; SPEC 6, ambient recall; B-15, B-34,
//! B-37): the daemon's half of the `user-prompt-submit` hook.
//!
//! 1. The four-word minimum (`short_prompt`). The hook checks it first;
//!    the daemon checks again so a direct caller gets the same answer.
//! 2. The project: the current one, unless the request asks for `all`.
//!    No resolvable project gives `no_project`.
//! 3. The model: `loading` while it loads, `error` after it failed (B-36).
//! 4. The prompt cut to `ambient::QUERY_MAX_TOKENS`, embedded with the
//!    query session, then exact search over active memories
//!    (`MemoryFilter::default()`).
//! 5. `ambient::apply_gates` with the request's `top_k` and the
//!    similarity floor: empty, floor, repeat, the cap, the 800-token
//!    ceiling. The block is exactly what it builds (AT-17).
//! 6. A reply built after the deadline is `timeout` and records nothing.
//!    Otherwise the injected ids are recorded after the reply is written
//!    (`After::RecordInjections`), still only before the deadline.
//!
//! The ambient switches are the hook's: `config::resolve_ambient` puts the
//! plugin option and version 2's variables above `[ambient]`, so the
//! daemon does not overrule them with its own configuration, and never
//! answers `disabled` itself. The hook journals `hook_gate_total`,
//! `hook_top_score` and `hook_latency_ms`; the daemon does not (B-34).
//!
//! The search asks for `top_k` memories plus the session's injected ones,
//! so memories already shown are replaced by the next best rather than
//! turning a fresh result into `repeat`. A memory longer than one chunk is
//! cut to its best chunk (SPEC 6): the content is chunked again, which
//! gives the stored pieces because the index stamp pins the chunk
//! settings, and the piece the search matched is used.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use sm_core::ambient::{self, Candidate};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::ids::MemoryId;
use sm_core::journal::HookGate;
use sm_core::protocol::AmbientParams;
use sm_core::protocol::AmbientReply;
use sm_core::traits::{MemoryFilter, MemoryHit, QueryScope, VectorQuery};
use sm_core::types::{Memory, Scope};

use crate::dispatch::Ctx;
use crate::state::Ready;

/// What to record after the reply: the session, the ids, the deadline.
pub(crate) type Record = (String, Vec<MemoryId>, Instant);

fn reply(gate: HookGate, project: Option<String>) -> AmbientReply {
    AmbientReply {
        gate,
        text: None,
        injected: Vec::new(),
        top_scores: Vec::new(),
        project,
    }
}

pub(crate) fn ambient(ctx: &Ctx<'_>, p: &AmbientParams) -> (AmbientReply, Option<Record>) {
    let sh = ctx.sh;
    if !ambient::prompt_passes(&p.prompt) {
        return (reply(HookGate::ShortPrompt, None), None);
    }
    let resolved = ctx.project();
    let project_name = resolved.as_ref().ok().map(|e| e.project.name.clone());
    let scope = match (p.scope, &resolved) {
        (Scope::All, _) => QueryScope::All,
        (Scope::Project, Ok(e)) => QueryScope::Project(e.project.id),
        (Scope::Project, Err(_)) => return (reply(HookGate::NoProject, None), None),
    };
    let ready = match sh.ready() {
        Ok(r) => r,
        Err(e) if e.code == ErrorCode::DaemonLoading => {
            return (reply(HookGate::Loading, project_name), None);
        }
        Err(e) => {
            log_error(ctx, &e);
            return (reply(HookGate::Error, project_name), None);
        }
    };
    match search(ctx, &ready, p, scope) {
        Ok(Found {
            candidates,
            outcome,
        }) => {
            let mut scores: Vec<f32> = candidates
                .iter()
                .map(|c| c.score)
                .filter(|s| s.is_finite())
                .collect();
            scores.sort_by(|a, b| b.total_cmp(a));
            scores.truncate(p.top_k as usize);
            let deadline = ctx
                .deadline
                .unwrap_or_else(|| Instant::now() + Duration::from_millis(ambient::DEADLINE_MS));
            if Instant::now() > deadline {
                let mut r = reply(HookGate::Timeout, project_name);
                r.top_scores = scores;
                return (r, None);
            }
            let mut r = reply(outcome.gate, project_name);
            r.top_scores = scores;
            let mut record = None;
            if let Some(block) = outcome.block {
                if let Some(session) = ctx.session() {
                    record = Some((session.to_owned(), block.injected.clone(), deadline));
                }
                r.text = Some(block.text);
                r.injected = block.injected;
            }
            (r, record)
        }
        Err(e) if e.code == ErrorCode::Timeout => (reply(HookGate::Timeout, project_name), None),
        Err(e) => {
            log_error(ctx, &e);
            (reply(HookGate::Error, project_name), None)
        }
    }
}

fn log_error(ctx: &Ctx<'_>, e: &CoreError) {
    ctx.sh.log.line(format!(
        "ambient_recall {}: {}: {}",
        ctx.req.id,
        e.code.as_str(),
        e.message
    ));
}

struct Found {
    candidates: Vec<Candidate>,
    outcome: ambient::GateOutcome,
}

fn search(
    ctx: &Ctx<'_>,
    ready: &Ready,
    p: &AmbientParams,
    scope: QueryScope,
) -> sm_core::error::Result<Found> {
    let sh = ctx.sh;
    let cap = p.top_k as usize;
    let floor = sh.settings.recall.similarity_floor as f32;
    let vector = ready
        .query
        .embed_query(&p.prompt, ambient::QUERY_MAX_TOKENS)?;
    ctx.check_deadline("embedding the prompt")?;
    let already: HashSet<MemoryId> = match ctx.session() {
        Some(s) => sh.store.injected(s)?.into_iter().collect(),
        None => HashSet::new(),
    };
    let hits = sh.store.nearest(&VectorQuery {
        vector: &vector,
        scope,
        filter: MemoryFilter::default(),
        limit: cap + already.len(),
    })?;
    ctx.check_deadline("the vector search")?;

    // Only the memories that can reach the block need their text: at or
    // above the floor, not yet shown, the best `cap` of them.
    let shown: HashSet<MemoryId> = {
        let mut order: Vec<&MemoryHit> = hits.iter().filter(|h| h.score.is_finite()).collect();
        order.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        order
            .into_iter()
            .filter(|h| h.score >= floor && !already.contains(&h.memory_id))
            .take(cap)
            .map(|h| h.memory_id)
            .collect()
    };
    let ids: Vec<MemoryId> = shown.iter().copied().collect();
    let memories: std::collections::HashMap<MemoryId, Memory> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let candidates: Vec<Candidate> = hits
        .iter()
        .map(|h| Candidate {
            id: h.memory_id,
            score: h.score,
            text: memories
                .get(&h.memory_id)
                .map(|m| best_chunk(ctx, ready, m, h.chunk_seq))
                .unwrap_or_default(),
        })
        .collect();
    let outcome = ambient::apply_gates(&candidates, floor, cap, &already);
    Ok(Found {
        candidates,
        outcome,
    })
}

/// The memory's text for its line: the whole content when it is one chunk,
/// else the chunk the search matched (SPEC 6, "Longer memories are cut to
/// their best chunk").
fn best_chunk(ctx: &Ctx<'_>, ready: &Ready, m: &Memory, seq: u32) -> String {
    let sh = ctx.sh;
    if seq == 0 && ambient::estimate_tokens(&m.content) <= sh.settings.chunk.max_tokens as usize {
        return m.content.clone();
    }
    match ready.chunker.chunk(&m.content, &sh.chunk_params(None)) {
        Ok(plan) if plan.pieces.len() > 1 => plan
            .pieces
            .into_iter()
            .find(|piece| piece.seq == seq)
            .map_or_else(|| m.content.clone(), |piece| piece.text),
        _ => m.content.clone(),
    }
}
