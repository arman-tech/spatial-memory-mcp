//! `capture` (CONTRACTS 7; SPEC 16; g-report section 2): the reply says
//! `accepted` once the request is queued, and one worker thread runs
//! `sm_capture::capture` off the reply path.
//!
//! - **Never loses** (SPEC 16, AT-15). A capture that cannot be processed
//!   now is written back to the spool for a later daemon: when the queue
//!   holds [`MAX_QUEUED`] jobs, when the model failed to load (the
//!   tokenizer chunks every memory), and when the daemon stops with jobs
//!   still queued. A job that fails for another reason is written to
//!   `spool/failed/`, which is never retried automatically (B-33), and the
//!   failure is logged.
//! - **Sessions.** One `SessionState` and one transcript offset per
//!   session, at most `capture_sessions` (256) sessions, the least recently used
//!   forgotten first. A forgotten session starts again at offset 0 with
//!   no hashes; the store's dedupe by content hash (B-29) keeps a text
//!   read twice from being stored twice. A capture with no session id
//!   (neither the request's nor the payload's) keeps no state: it reads
//!   its transcript from offset 0 with fresh hashes, as version 1 did for
//!   an empty session id (`transcript_cases.json`, `load_empty_session_id`
//!   and `save_empty_session_id_noop`), so two unrelated sessionless
//!   transcripts never share an offset.
//! - **Storing.** Each candidate becomes a memory with
//!   `Candidate::to_new_memory`, gets the derived title `remember` gives,
//!   and is chunked with `capture.max_index_tokens` as the index cap, then
//!   stored through `remember`'s path (`tools::remember::store`).
//! - **Metrics.** Per stored candidate, `capture_size_tokens` is the
//!   chunker's token count of the whole content (`ChunkPlan.total_tokens`,
//!   the same tokenizer as `count_tokens`), with the indexed tokens and the
//!   tool as labels; `capture_cap_hit_total` is 1 when the index cap cut it
//!   (`ChunkPlan.capped`). The 200,000-character extraction cap is the
//!   candidate's `provenance.extra.truncated` and is not counted (B-65).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::Value;
use sm_capture::{CaptureInput, SessionState};
use sm_core::error::{CoreError, Result};
use sm_core::journal::metrics;
use sm_core::protocol::{CaptureParams, CaptureReply, HookEvent, Op, PROTOCOL_VERSION, Request};
use sm_core::spool::write_spool;
use sm_core::types::derive_title;
use sm_store::MemoryInput;

use crate::dispatch::Ctx;
use crate::project;
use crate::state::{CaptureJob, Lru, ModelState, Ready, Shared, lock};
use crate::tools::remember::{pending_chunks, store};

/// Captures held in memory at most; more are spooled.
pub(crate) const MAX_QUEUED: usize = 1024;

fn is_capture_event(e: HookEvent) -> bool {
    matches!(
        e,
        HookEvent::PostToolUse | HookEvent::PreCompact | HookEvent::Stop
    )
}

pub(crate) fn enqueue(ctx: &Ctx<'_>, p: &CaptureParams) -> Result<CaptureReply> {
    let sh = ctx.sh;
    if !sh.settings.capture.enabled || !is_capture_event(p.event) {
        return Ok(CaptureReply { accepted: false });
    }
    let job = CaptureJob {
        request_id: ctx.req.id.clone(),
        cwd: ctx.req.cwd.clone(),
        session_id: ctx.req.session_id.clone(),
        client: ctx.req.client,
        params: p.clone(),
    };
    let failed = matches!(
        &*sh.model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        ModelState::Failed(_)
    );
    let mut q = lock(&sh.captures);
    if failed || sh.stopping() || q.len() >= MAX_QUEUED {
        drop(q);
        spool(sh, &job, false)?;
        return Ok(CaptureReply { accepted: true });
    }
    q.push_back(job);
    sh.captures_pending.fetch_add(1, Ordering::SeqCst);
    drop(q);
    sh.capture_wake.notify();
    Ok(CaptureReply { accepted: true })
}

fn request_of(sh: &Shared, job: &CaptureJob) -> Request {
    Request {
        v: PROTOCOL_VERSION,
        build: sm_core::BUILD.to_owned(),
        id: job.request_id.clone(),
        cwd: job.cwd.clone(),
        session_id: job.session_id.clone(),
        client: job.client,
        profile: sh.paths.profile.clone(),
        deadline_ms: None,
        call: Op::Capture(job.params.clone()),
    }
}

/// Writes the job to `spool/new/`, or to `spool/failed/` when it failed.
fn spool(sh: &Shared, job: &CaptureJob, failed: bool) -> Result<()> {
    let dir = if failed {
        sh.paths.spool_failed()
    } else {
        sh.paths.spool_new()
    };
    write_spool(&sh.paths.spool_tmp(), &dir, &request_of(sh, job)).map(|_| ())
}

/// Spools every queued job (shutdown, or the model failed).
fn spool_queued(sh: &Shared) {
    loop {
        let Some(job) = lock(&sh.captures).pop_front() else {
            return;
        };
        if let Err(e) = spool(sh, &job, false) {
            sh.log.line(format!(
                "capture {}: cannot spool: {}: {}",
                job.request_id,
                e.code.as_str(),
                e.message
            ));
        }
        sh.captures_pending.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One session's capture state.
struct Session {
    state: SessionState,
    offset: u64,
}

/// The capture worker: runs until the daemon stops.
pub(crate) fn run(sh: &Arc<Shared>) {
    let mut sessions: Lru<String, Session> = Lru::new(sh.opts.capture_sessions);
    loop {
        if sh.stopping() {
            spool_queued(sh);
            return;
        }
        let ready = match &*sh
            .model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            ModelState::Ready(r) => Some(Arc::clone(r)),
            ModelState::Loading | ModelState::Failed(_) => None,
        };
        let Some(ready) = ready else {
            if !sh.is_loading() {
                spool_queued(sh);
            }
            sh.capture_wake.wait(Duration::from_millis(200));
            continue;
        };
        let job = lock(&sh.captures).pop_front();
        let Some(job) = job else {
            sh.capture_wake.wait(Duration::from_millis(500));
            continue;
        };
        if let Err(e) = process(sh, &ready, &mut sessions, &job) {
            sh.log.line(format!(
                "capture {}: {}: {}; kept in spool/failed",
                job.request_id,
                e.code.as_str(),
                e.message
            ));
            if let Err(e) = spool(sh, &job, true) {
                sh.log.line(format!(
                    "capture {}: cannot spool: {}: {}",
                    job.request_id,
                    e.code.as_str(),
                    e.message
                ));
            }
        }
        sh.captures_pending.fetch_sub(1, Ordering::SeqCst);
        sh.touch();
    }
}

fn session_key(job: &CaptureJob) -> String {
    job.session_id
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            job.params
                .payload
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default()
}

/// Runs the pipeline for one job and stores its candidates.
fn process(
    sh: &Arc<Shared>,
    ready: &Ready,
    sessions: &mut Lru<String, Session>,
    job: &CaptureJob,
) -> Result<()> {
    let entry = project::resolve(sh, &job.cwd)?;
    let commit = entry.identity.repo.as_ref().and_then(|r| r.head_commit());
    let key = session_key(job);
    let mut stateless = Session {
        state: SessionState::new(),
        offset: 0,
    };
    let session = if key.is_empty() {
        &mut stateless
    } else {
        if sessions.get(&key).is_none() {
            sessions.insert(
                key.clone(),
                Session {
                    state: SessionState::new(),
                    offset: 0,
                },
            );
        }
        sessions
            .get_mut(&key)
            .ok_or_else(|| CoreError::internal("the capture session was not kept"))?
    };
    let outcome = sm_capture::capture(
        &CaptureInput {
            event: job.params.event,
            payload: &job.params.payload,
            settings: &sh.settings.capture,
            project: &entry.identity,
            commit_sha: commit.as_deref(),
            transcript_offset: session.offset,
        },
        &mut session.state,
    );
    if let Some(o) = outcome.new_offset {
        session.offset = o;
    }
    if outcome.candidates.is_empty() {
        return Ok(());
    }
    let tool = job
        .params
        .payload
        .get("tool_name")
        .and_then(Value::as_str)
        .unwrap_or(job.params.event.as_str())
        .to_owned();
    let cap = Some(sh.settings.capture.max_index_tokens as usize);
    let mut inputs = Vec::with_capacity(outcome.candidates.len());
    let mut sizes = Vec::with_capacity(outcome.candidates.len());
    for c in &outcome.candidates {
        let mut memory = c.to_new_memory(entry.project.id);
        if memory.title.is_none() {
            memory.title = derive_title(&memory.content);
        }
        let (chunks, plan) = pending_chunks(sh, ready, &memory.content, cap)?;
        sizes.push((plan.total_tokens, plan.indexed_tokens, plan.capped));
        inputs.push(MemoryInput { memory, chunks });
    }
    store(sh, &job.request_id, &entry.project.name, inputs)?;
    for (total, indexed, capped) in sizes {
        let mut labels = BTreeMap::new();
        labels.insert(
            "project".to_owned(),
            Value::from(entry.project.name.as_str()),
        );
        labels.insert("tool".to_owned(), Value::from(tool.as_str()));
        let mut size_labels = labels.clone();
        size_labels.insert("indexed_tokens".to_owned(), Value::from(indexed));
        sh.journal.record(
            &job.request_id,
            metrics::CAPTURE_SIZE_TOKENS,
            total as f64,
            size_labels,
        );
        if capped {
            sh.journal
                .record(&job.request_id, metrics::CAPTURE_CAP_HIT_TOTAL, 1.0, labels);
        }
    }
    Ok(())
}
