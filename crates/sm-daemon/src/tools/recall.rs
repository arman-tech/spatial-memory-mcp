//! `recall` and `reinforce` (SPEC 6 recall pipeline, SPEC 8).
//!
//! - `vector`: embed the query cut to the chunk limit, exact search, top
//!   `k`; results under the similarity floor are marked weak and returned.
//! - `keyword`: FTS5 only (the store quotes the query as one phrase). It
//!   needs no model, so it is served while the model is loading or failed.
//! - `auto` (default): vector, plus keyword hits appended when the query
//!   looks like an identifier, each marked `matched_by: keyword`, never
//!   score-fused.
//!
//! Each recall's result ids are kept per session (bounded) so `reinforce`
//! can find the rank a memory was recalled at.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::Value;
use sm_core::error::Result;
use sm_core::ids::MemoryId;
use sm_core::journal::metrics;
use sm_core::protocol::{
    MemoryView, RankSource, RecallParams, RecallReply, ReinforceParams, ReinforceReply,
};
use sm_core::traits::{MemoryFilter, VectorQuery};
use sm_core::types::{MatchKind, RecallMode};

use crate::budget::{fit, fixed_cost};
use crate::dispatch::Ctx;
use crate::state::lock;
use crate::view::Names;

/// Whether a query looks like an identifier (SPEC 6, step 6): a token with
/// `_`, `::` or a `.` between word characters (a symbol, a table or file
/// name), or an error code such as `E0425` or `ORA-00942`.
pub(crate) fn looks_like_identifier(query: &str) -> bool {
    query.split_whitespace().any(|raw| {
        let t = raw.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'));
        if t.is_empty() {
            return false;
        }
        if t.contains("::") {
            return true;
        }
        let word = |c: char| c.is_alphanumeric();
        if t.contains('_') && t.chars().any(word) {
            return true;
        }
        let chars: Vec<char> = t.chars().collect();
        if chars
            .windows(3)
            .any(|w| w[1] == '.' && word(w[0]) && word(w[2]))
        {
            return true;
        }
        error_code(t)
    })
}

/// Letters, an optional `-`, then at least two digits, with no lower-case
/// word around it (`E0425`, `ORA-00942`, `HTTP404`).
fn error_code(t: &str) -> bool {
    let letters: String = t.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() || letters.len() > 10 {
        return false;
    }
    let rest = &t[letters.len()..];
    let digits = rest.strip_prefix('-').unwrap_or(rest);
    digits.len() >= 2 && digits.chars().all(|c| c.is_ascii_digit())
}

pub(crate) fn recall(ctx: &Ctx<'_>, p: &RecallParams) -> Result<RecallReply> {
    let sh = ctx.sh;
    let mode = p.mode.unwrap_or_default();
    let k = p.k.unwrap_or(sh.settings.recall.k) as usize;
    let floor = sh.settings.recall.similarity_floor as f32;
    let (scope, _) = ctx.scope(p.scope)?;
    let filter = MemoryFilter {
        types: p.memory_type.clone(),
        statuses: p.status.clone(),
        anchor: p.anchor.clone(),
        min_importance: p.min_importance,
        exclude: Vec::new(),
    };
    let mut found: Vec<(MemoryId, Option<f32>, MatchKind)> = Vec::new();
    if mode != RecallMode::Keyword {
        let ready = sh.ready()?;
        let vector = ready
            .query
            .embed_query(&p.query, sh.settings.chunk.max_tokens as usize)?;
        ctx.check_deadline("embedding the query")?;
        let hits = sh.store.nearest(&VectorQuery {
            vector: &vector,
            scope,
            filter: filter.clone(),
            limit: k,
        })?;
        ctx.check_deadline("the vector search")?;
        found.extend(
            hits.into_iter()
                .map(|h| (h.memory_id, Some(h.score), MatchKind::Vector)),
        );
    }
    if mode == RecallMode::Keyword || (mode == RecallMode::Auto && looks_like_identifier(&p.query))
    {
        let present: HashSet<MemoryId> = found.iter().map(|f| f.0).collect();
        let hits = sh.store.keyword_search(&p.query, scope, &filter, k)?;
        ctx.check_deadline("the keyword search")?;
        found.extend(
            hits.into_iter()
                .filter(|h| !present.contains(&h.memory_id))
                .map(|h| (h.memory_id, None, MatchKind::Keyword)),
        );
    }

    let ids: Vec<MemoryId> = found.iter().map(|f| f.0).collect();
    let mut memories: HashMap<MemoryId, _> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let names = Names::load(&sh.store)?;
    // Ranks are numbered after a memory deleted since the search is
    // dropped, so each rank is the memory's position in `returned`, the
    // one `reinforce` records.
    let mut views: Vec<MemoryView> = Vec::with_capacity(found.len());
    for (id, score, kind) in found {
        let Some(m) = memories.remove(&id) else {
            continue;
        };
        let mut v = names.view(m);
        v.score = score;
        v.weak = score.is_some_and(|s| s < floor);
        v.rank = Some(views.len() as u32 + 1);
        v.matched_by = Some(kind);
        views.push(v);
    }
    let weak_only = !views.is_empty() && views.iter().all(|v| v.weak);
    let returned: Vec<MemoryId> = views.iter().map(|v| v.id).collect();
    let budget = ctx.budget(p.limit);
    let skeleton = RecallReply {
        results: Vec::new(),
        weak_only: true,
        truncated: true,
        omitted: u32::MAX,
    };
    let fitted = fit(views, budget, fixed_cost("recall", &skeleton, budget)?);

    let mut labels = BTreeMap::new();
    labels.insert("mode".to_owned(), Value::from(mode.as_str()));
    sh.journal.record(
        &ctx.req.id,
        metrics::RECALL_RESULT_COUNT,
        returned.len() as f64,
        labels.clone(),
    );
    if weak_only {
        sh.journal
            .record(&ctx.req.id, metrics::RECALL_WEAK_RESULT_TOTAL, 1.0, labels);
    }
    if let Some(session) = ctx.session() {
        lock(&sh.ranks).insert(session.to_owned(), returned);
    }
    Ok(RecallReply {
        results: fitted.items,
        weak_only,
        truncated: fitted.truncated,
        omitted: fitted.omitted,
    })
}

/// Records the rank a memory helped at (SPEC 8: a signal, not a score).
/// The rank is the one given, else the memory's position in this
/// session's last recall, else unknown. Nothing in the store changes.
pub(crate) fn reinforce(ctx: &Ctx<'_>, p: &ReinforceParams) -> Result<ReinforceReply> {
    let sh = ctx.sh;
    sh.store.get_memory(&p.id)?;
    let (rank, rank_source) = match p.rank {
        Some(r) => (Some(r), RankSource::Given),
        None => {
            let last = ctx.session().and_then(|s| {
                lock(&sh.ranks)
                    .get(&s.to_owned())
                    .and_then(|ids| ids.iter().position(|i| *i == p.id))
            });
            match last {
                Some(pos) => (Some(pos as u32 + 1), RankSource::LastRecall),
                None => (None, RankSource::Unknown),
            }
        }
    };
    if let Some(r) = rank {
        let mut labels = BTreeMap::new();
        labels.insert("rank_source".to_owned(), Value::from(rank_source.as_str()));
        labels.insert("memory_id".to_owned(), Value::from(p.id.to_string()));
        sh.journal.record(
            &ctx.req.id,
            metrics::REINFORCED_MEMORY_RANK,
            f64::from(r),
            labels,
        );
    }
    Ok(ReinforceReply {
        id: p.id,
        rank,
        rank_source,
    })
}

#[cfg(test)]
mod tests {
    use super::looks_like_identifier;

    #[test]
    fn s6_identifier_queries_run_keyword_search() {
        for q in [
            "user_sessions table",
            "sm_core::error",
            "config.toml",
            "E0425",
            "error ORA-00942 on insert",
            "why does read_frame fail",
        ] {
            assert!(looks_like_identifier(q), "{q}");
        }
        for q in [
            "how do we deploy the service",
            "what did we decide about caching.",
            "v2",
            "",
        ] {
            assert!(!looks_like_identifier(q), "{q}");
        }
    }
}
