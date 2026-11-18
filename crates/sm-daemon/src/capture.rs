//! `capture` (CONTRACTS 7; SPEC 16; g-report section 2): the reply says
//! `accepted` once the request is queued, and one worker thread runs
//! `sm_capture::capture` off the reply path.
//!
//! - **On disk until stored** (SPEC 16, never loses; AT-15; e-report
//!   decision 28). From the moment the worker takes a capture until it is
//!   stored, the capture is a file in `spool/new/`: a drained file stays
//!   where it is, and a socket capture is written there before it is
//!   processed. The file is removed only after the store succeeds. Each
//!   file a job holds is named in `Shared::held`, and the drain skips those
//!   names, so a file is never queued twice by one daemon. A capture that
//!   is not processed now stays in, or is written to, `spool/new/` for a
//!   later pass or a later daemon: when the queue holds [`MAX_QUEUED`] jobs,
//!   when the model failed to load (the tokenizer chunks every memory), and
//!   when the daemon stops with jobs still queued. After a crash the file
//!   is processed again; the store's duplicate rule (B-29) returns the
//!   memory already stored, so processing is at least once and a capture
//!   is never stored twice.
//! - **Cannot spool** (stream H 7). When the socket capture cannot be
//!   written to `spool/new/` (a full disk, a broken spool folder), the
//!   worker logs it and processes the capture from memory: it is in memory
//!   either way, and processing gives it the one durable place left, the
//!   store. Only a crash before the store succeeds loses it, as it would
//!   lose a capture still queued. The write is tried again before each
//!   retry, and a capture that fails for good is written to
//!   `spool/failed/` as usual (that write may fail as well, and is logged).
//! - **Retries** (e-report decision 27). The pipeline runs on a copy of the
//!   session's state and offset, and the copy replaces the session only
//!   after the attempt ends with every candidate stored or dropped. A
//!   failure that may pass later (any code but `invalid_input` and
//!   `secret_detected`, and a panic) is tried again after
//!   `capture_retry_backoff`, doubled after each failure, up to
//!   [`MAX_ATTEMPTS`] attempts in this daemon; the worker takes other jobs
//!   meanwhile. `project_unresolved` is one of them (stream H 5): the
//!   working folder may be created, or come back, within the bound. At the
//!   bound, or at once for those two codes, the file moves to
//!   `spool/failed/` (B-33), with a log line naming the request id and the
//!   cause.
//! - **Drops** (stream H 1). A failure is per candidate where it can be:
//!   one candidate that can never be stored (those two codes), or that
//!   fails at the last attempt, is dropped, the others are stored, and the
//!   session moves on, so one bad candidate never blocks the captures
//!   after it. One log line names the request id, the cause and how many
//!   candidates were dropped, and the request moves to `spool/failed/` as
//!   the record. A candidate that may still be stored is never dropped
//!   before the last attempt: the attempt fails and the whole job is tried
//!   again, and the store's dedupe (B-29) keeps what was stored already
//!   from being stored twice. A failure of the job itself (the project,
//!   or a panic in the pipeline) has no new session state to move on to,
//!   so the session stays where it was.
//! - **Unremovable files** (stream H 6). When a stored capture's file
//!   cannot be removed, its name goes to `Shared::unremovable`, under the
//!   `held` lock before the name is released, and one line is logged. The
//!   drain never claims such a file again; it tries the removal again,
//!   quietly, at each pass, and forgets the name once the file is gone.
//! - **Sessions** (B-162, B-163). One `SessionState` and one transcript
//!   read position per session, at most `capture_sessions` (256) sessions
//!   in memory, the least recently used dropped first. The read position is
//!   kept in the store too (`capture_sessions`): after every attempt that
//!   ends with each candidate stored or dropped, it is written with the
//!   hashes of the attempt's candidates in one transaction, so a session
//!   dropped from memory, or read by a later daemon, goes on from where the
//!   last read stopped. A session the store has no position for starts at
//!   `None`, and the pipeline reads only what follows its transcript's last
//!   compaction boundary (B-162). A capture with no session id (neither the
//!   request's nor the payload's) keeps no position and has fresh hashes,
//!   as version 1 did for an empty session id (`transcript_cases.json`,
//!   `load_empty_session_id` and `save_empty_session_id_noop`), so two
//!   unrelated sessionless transcripts never share an offset.
//! - **Seen** (B-163). The hash of every candidate is recorded for its
//!   project (`capture_seen`), whatever becomes of it, and a candidate whose
//!   hash is there is dropped before the store: a text read again is never
//!   captured twice, even once its memory has been forgotten, rejected or
//!   pruned.
//! - **Near duplicates** (B-164, RC9). Before the store, the candidates are
//!   chunked and embedded on the bulk session, and a candidate whose vector
//!   has a cosine of at least [`CAPTURE_NEAR_DUPLICATE_COSINE`] with an active or
//!   pending memory of the project, or with an earlier candidate of the
//!   same attempt, is dropped. When the embedder fails, the candidates are
//!   stored with their chunks waiting for the embedding queue and without
//!   this check, and one line is logged: a capture is never lost for it.
//! - **Storing.** Each candidate becomes a memory with
//!   `Candidate::to_new_memory`, gets the derived title `remember` gives,
//!   and is chunked with `capture.max_index_tokens` as the index cap, then
//!   stored `pending` with its context (B-166) and queued for the judge
//!   (`Store::insert_captures`, B-167), and the judge's worker is woken.
//!   Nothing stored here is served until the judge keeps it (B-157,
//!   B-168).
//! - **Metrics.** Per stored candidate, `capture_size_tokens` is the
//!   chunker's token count of the whole content (`ChunkPlan.total_tokens`,
//!   the same tokenizer as `count_tokens`), with the indexed tokens and the
//!   tool as labels; `capture_cap_hit_total` is 1 when the index cap cut it
//!   (`ChunkPlan.capped`). The 200,000-character extraction cap is the
//!   candidate's `provenance.extra.truncated` and is not counted (B-65).
//!   Each candidate dropped as seen or as a near duplicate is one
//!   `capture_dropped_total` line, with the reason and, for a near
//!   duplicate, the cosine.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sm_capture::{CaptureInput, REPLAY_TRIGGER, SessionState};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::journal::metrics;
use sm_core::protocol::{
    CaptureParams, CaptureReply, HookEvent, Op, PROTOCOL_VERSION, RememberOutcome, ReplayParams,
    Request,
};
use sm_core::spool::{quarantine, write_spool};
use sm_core::time::Timestamp;
use sm_core::traits::DocInput;
use sm_core::types::derive_title;
use sm_store::{CaptureProgress, MemoryInput, NewChunk, unit_mean};

use crate::dispatch::Ctx;
use crate::project;
use crate::queue;
use crate::state::{CaptureJob, Lru, ModelState, ProjectEntry, Ready, Shared, lock, panic_text};
use crate::tools::remember::{journal_chunks, pending_chunks};

/// Captures held in memory at most; more are spooled.
pub(crate) const MAX_QUEUED: usize = 1024;

/// Attempts at one capture in one daemon before its file moves to
/// `spool/failed/` (e-report decision 27).
pub(crate) const MAX_ATTEMPTS: u32 = 5;

/// The cosine at or above which a capture candidate is a near duplicate of
/// an active or pending memory of its project, or of an earlier candidate
/// of the same attempt, and is not stored (B-164, RC9). This drop happens
/// before anything has judged the candidate, against pending captures the
/// judge may yet reject, so it is set where it cost no "worth keeping"
/// item of the labelled export: the lowest two-place cosine with none
/// dropped, simulated in capture order (B-169; version 1's 0.85 dropped 15
/// of 34). The meaning check is the judge stage's
/// [`crate::judge::NEAR_DUPLICATE_COSINE`]. Each drop is journalled as
/// `capture_dropped_total` with its cosine.
pub(crate) const CAPTURE_NEAR_DUPLICATE_COSINE: f32 = 0.93;

// The capture stage never drops what the judge stage would keep.
const _: () = assert!(CAPTURE_NEAR_DUPLICATE_COSINE >= crate::judge::NEAR_DUPLICATE_COSINE);

fn is_capture_event(e: HookEvent) -> bool {
    matches!(
        e,
        HookEvent::PostToolUse | HookEvent::PreCompact | HookEvent::Stop
    )
}

/// Queues a capture. `ctx.spool_file` is the drained file it came from,
/// claimed by the drain: an accepted capture takes the claim over (the job
/// holds the file, or the file is released where it is); otherwise the
/// drain keeps it.
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
        replay: None,
        file: ctx.spool_file.map(Path::to_path_buf),
        attempts: 0,
        not_before: None,
    };
    queue_job(sh, job)
}

/// Queues a replay (SPEC 16, Replay; B-172): the session's stored
/// transcript, read from its start, through the capture pipeline. It is
/// queued, spooled and retried as a capture is, under the session's own
/// working folder; the reply comes as soon as it is queued. With capture
/// off it is not accepted.
pub(crate) fn enqueue_replay(ctx: &Ctx<'_>, p: &ReplayParams) -> Result<CaptureReply> {
    let sh = ctx.sh;
    if !sh.settings.capture.enabled {
        return Ok(CaptureReply { accepted: false });
    }
    let job = CaptureJob {
        request_id: ctx.req.id.clone(),
        cwd: p.cwd.clone(),
        session_id: Some(p.session_id.clone()),
        client: ctx.req.client,
        params: CaptureParams {
            event: HookEvent::PreCompact,
            payload: json!({
                "session_id": p.session_id,
                "transcript_path": p.transcript_path,
                "cwd": p.cwd,
            }),
            received_at: p.received_at,
        },
        replay: Some(Box::new(p.clone())),
        file: ctx.spool_file.map(Path::to_path_buf),
        attempts: 0,
        not_before: None,
    };
    queue_job(sh, job)
}

/// Puts an accepted job on the queue, or on disk when the model failed,
/// the daemon is stopping or the queue is full.
fn queue_job(sh: &Shared, job: CaptureJob) -> Result<CaptureReply> {
    let failed = matches!(
        &*sh.model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        ModelState::Failed(_)
    );
    let mut q = lock(&sh.captures);
    if failed || sh.stopping() || q.len() >= MAX_QUEUED {
        drop(q);
        to_disk(sh, job)?;
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
        call: match &job.replay {
            Some(r) => Op::Replay(ReplayParams::clone(r)),
            None => Op::Capture(job.params.clone()),
        },
    }
}

/// Claims a file of `spool/new/` for the drain. False when a job of this
/// daemon holds it already, or when it is the file of a stored capture that
/// could not be removed. `held` is locked before `unremovable`, here and in
/// [`settle`], so a name moves from one to the other with no gap between.
pub(crate) fn claim(sh: &Shared, file: &Path) -> bool {
    file.file_name().is_some_and(|n| {
        let mut held = lock(&sh.held);
        !lock(&sh.unremovable).contains(n) && held.insert(n.to_owned())
    })
}

/// For the drain: when `file` is the file of a stored capture that could not
/// be removed, tries the removal again, quietly, forgets the name once the
/// file is gone, and returns true; the file is not processed again.
pub(crate) fn retry_removal(sh: &Shared, file: &Path) -> bool {
    let Some(n) = file.file_name() else {
        return false;
    };
    let mut unremovable = lock(&sh.unremovable);
    if !unremovable.contains(n) {
        return false;
    }
    match std::fs::remove_file(file) {
        Ok(()) => {
            unremovable.remove(n);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            unremovable.remove(n);
        }
        Err(_) => {}
    }
    true
}

/// Releases a held file; the file itself is left where it is.
pub(crate) fn release(sh: &Shared, file: &Path) {
    if let Some(n) = file.file_name() {
        lock(&sh.held).remove(n);
    }
}

/// Leaves a capture that is not processed now in `spool/new/` for a later
/// pass or a later daemon: a held file is released where it is, and a
/// socket capture is written there.
fn to_disk(sh: &Shared, job: CaptureJob) -> Result<()> {
    match &job.file {
        Some(f) => {
            release(sh, f);
            Ok(())
        }
        None => write_spool(
            &sh.paths.spool_tmp(),
            &sh.paths.spool_new(),
            &request_of(sh, &job),
        )
        .map(|_| ()),
    }
}

/// Moves a capture that will never be stored to `spool/failed/`.
fn to_failed(sh: &Shared, job: &CaptureJob) {
    let failed = sh.paths.spool_failed();
    let moved = match &job.file {
        Some(f) => {
            let r = quarantine(f, &failed).map(|_| ());
            release(sh, f);
            r
        }
        None => write_spool(&sh.paths.spool_tmp(), &failed, &request_of(sh, job)).map(|_| ()),
    };
    if let Err(e) = moved {
        sh.log.line(format!(
            "capture {}: cannot move it to spool/failed: {}: {}",
            job.request_id,
            e.code.as_str(),
            e.message
        ));
    }
}

fn done(sh: &Shared) {
    sh.captures_pending.fetch_sub(1, Ordering::SeqCst);
}

/// Leaves every queued job on disk (shutdown, or the model failed). Holds
/// `taking`, so no job is between the queue and the worker meanwhile.
pub(crate) fn spool_queued(sh: &Shared) {
    let _taking = lock(&sh.taking);
    loop {
        let Some(job) = lock(&sh.captures).pop_front() else {
            return;
        };
        let id = job.request_id.clone();
        if let Err(e) = to_disk(sh, job) {
            sh.log.line(format!(
                "capture {id}: cannot spool: {}: {}",
                e.code.as_str(),
                e.message
            ));
        }
        done(sh);
    }
}

enum Taken {
    Job(CaptureJob),
    /// Nothing is due; the earliest retry is due after this long.
    Wait(Option<Duration>),
}

/// The first job that is due, moved to the worker. A socket capture is
/// written to `spool/new/` first.
fn take(sh: &Shared) -> Taken {
    let _taking = lock(&sh.taking);
    let now = Instant::now();
    let mut q = lock(&sh.captures);
    let Some(pos) = q.iter().position(|j| j.not_before.is_none_or(|t| t <= now)) else {
        let next = q.iter().filter_map(|j| j.not_before).min();
        return Taken::Wait(next.map(|t| t.saturating_duration_since(now)));
    };
    let Some(mut job) = q.remove(pos) else {
        return Taken::Wait(None);
    };
    drop(q);
    if job.file.is_none() {
        match hold_new(sh, &job) {
            Ok(path) => job.file = Some(path),
            Err(e) => sh.log.line(format!(
                "capture {}: cannot spool before processing: {}: {}; processed from memory",
                job.request_id,
                e.code.as_str(),
                e.message
            )),
        }
    }
    Taken::Job(job)
}

/// Writes a socket capture to `spool/new/` and holds its file. The `held`
/// lock is kept across the write, so the drain cannot list the file before
/// its name is held.
fn hold_new(sh: &Shared, job: &CaptureJob) -> Result<PathBuf> {
    let mut held = lock(&sh.held);
    let path = write_spool(
        &sh.paths.spool_tmp(),
        &sh.paths.spool_new(),
        &request_of(sh, job),
    )?;
    if let Some(n) = path.file_name() {
        held.insert(n.to_owned());
    }
    Ok(path)
}

/// A failure that can never pass: the candidate is dropped, or the file goes
/// to `spool/failed/`, at once. `project_unresolved` (a working folder that
/// is the home folder, a filesystem root or gone) is not one: a folder that
/// is gone may be created again within the bound (stream H 5).
fn permanent(e: &CoreError) -> bool {
    matches!(e.code, ErrorCode::InvalidInput | ErrorCode::SecretDetected)
}

/// Runs `f`, turning a panic into an `internal` error.
fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        Err(CoreError::internal(format!(
            "processing panicked: {}",
            panic_text(p.as_ref())
        )))
    })
}

fn cause_of(e: &CoreError) -> String {
    format!("{}: {}", e.code.as_str(), e.message)
}

/// How an attempt that did not fail ended.
enum Attempt {
    /// Every candidate was stored, or there were none.
    Stored,
    /// `dropped` of the `of` candidates can never be stored, or failed at
    /// the last attempt, and were dropped; the others were stored. `cause`
    /// and `why` are the first dropped candidate's.
    Dropped {
        cause: String,
        why: String,
        dropped: usize,
        of: usize,
    },
}

/// Removes a stored capture's file. A file that cannot be removed is named
/// in `unremovable` before it is released, under the `held` lock, so the
/// drain never processes it again (stream H 6).
fn remove_stored(sh: &Shared, job: &CaptureJob) {
    let Some(f) = &job.file else {
        return;
    };
    let failed = match std::fs::remove_file(f) {
        Ok(()) => None,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => Some(e),
    };
    let Some(n) = f.file_name() else {
        return;
    };
    let mut held = lock(&sh.held);
    if failed.is_some() {
        lock(&sh.unremovable).insert(n.to_owned());
    }
    held.remove(n);
    drop(held);
    if let Some(e) = failed {
        sh.log.line(format!(
            "capture {}: stored, but {} cannot be removed: {e}; it is not processed again, \
             and its removal is tried again at each drain",
            job.request_id,
            f.display()
        ));
    }
}

/// The wait before attempt `failed + 1`.
fn backoff(sh: &Shared, failed: u32) -> Duration {
    sh.opts
        .capture_retry_backoff
        .saturating_mul(1u32 << failed.saturating_sub(1).min(16))
}

/// After one attempt: the file is removed once stored; a capture with
/// candidates dropped, one that can never be stored, or one that failed
/// [`MAX_ATTEMPTS`] times, moves to `spool/failed/`; any other failure is
/// queued again after its backoff.
fn settle(sh: &Shared, mut job: CaptureJob, result: Result<Attempt>) {
    let e = match result {
        Ok(Attempt::Stored) => {
            remove_stored(sh, &job);
            done(sh);
            return;
        }
        Ok(Attempt::Dropped {
            cause,
            why,
            dropped,
            of,
        }) => {
            sh.log.line(format!(
                "capture {}: {cause}; {why}, moved to spool/failed; {dropped} of {of} \
                 candidates dropped, and its session moves on",
                job.request_id
            ));
            to_failed(sh, &job);
            done(sh);
            return;
        }
        Err(e) => e,
    };
    job.attempts += 1;
    let cause = cause_of(&e);
    if permanent(&e) || job.attempts >= MAX_ATTEMPTS {
        let why = if permanent(&e) {
            "it can never be stored".to_owned()
        } else {
            format!("failed {} times", job.attempts)
        };
        sh.log.line(format!(
            "capture {}: {cause}; {why}, moved to spool/failed",
            job.request_id
        ));
        to_failed(sh, &job);
        done(sh);
        return;
    }
    let wait = backoff(sh, job.attempts);
    sh.log.line(format!(
        "capture {}: {cause}; attempt {} of {MAX_ATTEMPTS} failed, tried again in {wait:?}",
        job.request_id, job.attempts
    ));
    let mut q = lock(&sh.captures);
    // Checked under the queue lock: once stopping, shutdown may have
    // written the queue to disk already.
    if !sh.stopping() {
        job.not_before = Some(Instant::now() + wait);
        q.push_back(job);
        return;
    }
    drop(q);
    let id = job.request_id.clone();
    if let Err(e) = to_disk(sh, job) {
        sh.log.line(format!(
            "capture {id}: cannot spool: {}: {}",
            e.code.as_str(),
            e.message
        ));
    }
    done(sh);
}

/// One session's capture state.
#[derive(Clone, Default)]
struct Session {
    state: SessionState,
    /// Where the next transcript read starts; `None` for a session with no
    /// recorded read (B-162).
    offset: Option<u64>,
}

/// The capture worker: runs until the daemon stops.
pub(crate) fn run(sh: &Arc<Shared>) {
    let mut sessions: Lru<String, Session> = Lru::new(sh.opts.capture_sessions);
    loop {
        if sh.stopping() {
            spool_queued(sh);
            return;
        }
        let Some(ready) = sh.ready_opt() else {
            if !sh.is_loading() {
                spool_queued(sh);
            }
            sh.capture_wake.wait(Duration::from_millis(200));
            continue;
        };
        let job = match take(sh) {
            Taken::Job(j) => j,
            Taken::Wait(next) => {
                let idle = Duration::from_millis(500);
                sh.capture_wake.wait(
                    next.map_or(idle, |d| d.min(idle))
                        .max(Duration::from_millis(1)),
                );
                continue;
            }
        };
        let result = guarded(|| process(sh, &ready, &mut sessions, &job));
        settle(sh, job, result);
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

/// Runs the pipeline for one job and stores its candidates. The session
/// moves on only once each is stored or dropped; then its read position and
/// the candidates' hashes are recorded in the store (B-163). A replay
/// (B-172) reads the whole transcript with fresh session state, records
/// only the hashes, and leaves the session's read position and its state in
/// memory as they were.
fn process(
    sh: &Arc<Shared>,
    ready: &Ready,
    sessions: &mut Lru<String, Session>,
    job: &CaptureJob,
) -> Result<Attempt> {
    let entry = project::resolve(sh, &job.cwd)?;
    let commit = entry.identity.repo.as_ref().and_then(|r| r.head_commit());
    let key = if job.replay.is_some() {
        String::new()
    } else {
        session_key(job)
    };
    let mut next = if key.is_empty() {
        Session::default()
    } else if let Some(s) = sessions.get(&key) {
        s.clone()
    } else {
        Session {
            state: SessionState::default(),
            offset: sh.store.capture_offset(&key)?,
        }
    };
    let read_from = next.offset;
    let input = CaptureInput {
        event: job.params.event,
        payload: &job.params.payload,
        settings: &sh.settings.capture,
        project: &entry.identity,
        commit_sha: commit.as_deref(),
        transcript_offset: next.offset,
    };
    let outcome = if job.replay.is_some() {
        sm_capture::replay(&input, &mut next.state)
    } else {
        sm_capture::capture(&input, &mut next.state)
    };
    if let Some(o) = outcome.new_offset {
        next.offset = Some(o);
    }
    let attempt = if outcome.candidates.is_empty() {
        Attempt::Stored
    } else {
        store_candidates(sh, ready, job, &entry, &outcome.candidates)?
    };
    let hashes: Vec<String> = outcome
        .candidates
        .iter()
        .map(|c| c.content_hash.clone())
        .collect();
    let moved = match next.offset {
        Some(o) if !key.is_empty() && next.offset != read_from => Some((key.as_str(), o)),
        _ => None,
    };
    if moved.is_some() || !hashes.is_empty() {
        sh.store.record_capture_progress(&CaptureProgress {
            session: moved,
            project: entry.project.id,
            hashes: &hashes,
            at: Timestamp::now(),
        })?;
    }
    if !key.is_empty() {
        sessions.insert(key, next);
    }
    Ok(attempt)
}

/// One candidate ready to store, with its chunk plan's sizes.
struct Prepared {
    input: MemoryInput,
    /// What the judge reads besides the text (B-166).
    context: String,
    total_tokens: usize,
    indexed_tokens: usize,
    capped: bool,
}

fn prepare(
    sh: &Shared,
    ready: &Ready,
    entry: &ProjectEntry,
    candidate: &sm_capture::Candidate,
) -> Result<Prepared> {
    let mut memory = candidate.to_new_memory(entry.project.id);
    if memory.title.is_none() {
        memory.title = derive_title(&memory.content);
    }
    let cap = Some(sh.settings.capture.max_index_tokens as usize);
    let (chunks, plan) = pending_chunks(sh, ready, &memory.content, cap)?;
    Ok(Prepared {
        input: MemoryInput { memory, chunks },
        context: candidate.context.clone(),
        total_tokens: plan.total_tokens,
        indexed_tokens: plan.indexed_tokens,
        capped: plan.capped,
    })
}

/// Stores the candidates, then journals the sizes of those the store kept:
/// a candidate the store found to be a duplicate (B-29) is not a capture,
/// so it is not sized but journalled as dropped (M1, B-159).
fn store_batch(
    sh: &Shared,
    job: &CaptureJob,
    entry: &ProjectEntry,
    tool: &str,
    batch: &[Prepared],
) -> Result<()> {
    let inputs = batch
        .iter()
        .map(|p| (p.input.clone(), p.context.clone()))
        .collect();
    let inserted = sh.store.insert_captures(inputs)?;
    sh.after_write();
    journal_chunks(sh, &job.request_id, &entry.project.name, &inserted);
    sh.judge_wake.notify();
    for (p, i) in batch.iter().zip(&inserted) {
        if i.outcome != RememberOutcome::Created {
            journal_drop(sh, job, entry, tool, "duplicate", None);
            continue;
        }
        let mut labels = BTreeMap::new();
        labels.insert(
            "project".to_owned(),
            Value::from(entry.project.name.as_str()),
        );
        labels.insert("tool".to_owned(), Value::from(tool));
        let mut size_labels = labels.clone();
        size_labels.insert("indexed_tokens".to_owned(), Value::from(p.indexed_tokens));
        sh.journal.record(
            &job.request_id,
            metrics::CAPTURE_SIZE_TOKENS,
            p.total_tokens as f64,
            size_labels,
        );
        if p.capped {
            sh.journal
                .record(&job.request_id, metrics::CAPTURE_CAP_HIT_TOTAL, 1.0, labels);
        }
    }
    Ok(())
}

/// One line of `capture_dropped_total` (B-163, B-164).
fn journal_drop(
    sh: &Shared,
    job: &CaptureJob,
    entry: &ProjectEntry,
    tool: &str,
    reason: &str,
    cosine: Option<f32>,
) {
    let mut labels = BTreeMap::new();
    labels.insert(
        "project".to_owned(),
        Value::from(entry.project.name.as_str()),
    );
    labels.insert("tool".to_owned(), Value::from(tool));
    labels.insert("reason".to_owned(), Value::from(reason));
    if let Some(c) = cosine {
        labels.insert("cosine".to_owned(), Value::from(f64::from(c)));
    }
    sh.journal
        .record(&job.request_id, metrics::CAPTURE_DROPPED_TOTAL, 1.0, labels);
}

/// The candidates whose hash the project's capture has not handled before
/// (B-163); each of the others is journalled and dropped.
fn unseen<'a>(
    sh: &Shared,
    job: &CaptureJob,
    entry: &ProjectEntry,
    tool: &str,
    candidates: &'a [sm_capture::Candidate],
) -> Result<Vec<&'a sm_capture::Candidate>> {
    let hashes: Vec<String> = candidates.iter().map(|c| c.content_hash.clone()).collect();
    let seen = sh.store.captures_seen(entry.project.id, &hashes)?;
    let mut fresh = Vec::with_capacity(candidates.len());
    for c in candidates {
        if seen.contains(&c.content_hash) {
            journal_drop(sh, job, entry, tool, "seen", None);
        } else {
            fresh.push(c);
        }
    }
    Ok(fresh)
}

/// The cosine of two unit vectors.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Embeds every chunk of `prepared` in one call on the bulk session and
/// returns one vector per candidate: the unit mean of its chunks, as the
/// store computes a memory's vector.
fn embed(sh: &Shared, ready: &Ready, prepared: &mut [Prepared]) -> Result<Vec<Vec<f32>>> {
    let mut docs: Vec<(Option<&str>, &mut [NewChunk])> = prepared
        .iter_mut()
        .map(|p| {
            (
                p.input.memory.title.as_deref(),
                p.input.chunks.as_mut_slice(),
            )
        })
        .collect();
    embed_docs(sh, ready, &mut docs)
}

/// Embeds every chunk of `docs`, each a title and its chunks, in one call
/// on the bulk session, sets each chunk's embedding, and returns one
/// vector per document: the unit mean of its chunks, as the store computes
/// a memory's vector. The capture worker and the judge share it.
pub(crate) fn embed_docs(
    sh: &Shared,
    ready: &Ready,
    docs: &mut [(Option<&str>, &mut [NewChunk])],
) -> Result<Vec<Vec<f32>>> {
    let vectors = queue::with_bulk(sh, ready, |session| {
        let inputs: Vec<DocInput<'_>> = docs
            .iter()
            .flat_map(|(title, chunks)| {
                let title = *title;
                chunks.iter().map(move |c| DocInput {
                    title,
                    text: &c.text,
                })
            })
            .collect();
        let vectors = session.embed_documents(&inputs)?;
        if vectors.len() != inputs.len() {
            return Err(CoreError::internal(format!(
                "the embedder returned {} vectors for {} chunks",
                vectors.len(),
                inputs.len()
            )));
        }
        Ok(vectors)
    })?;
    let mut vectors = vectors.into_iter();
    let mut means = Vec::with_capacity(docs.len());
    for (_, chunks) in docs.iter_mut() {
        let mut own = Vec::with_capacity(chunks.len());
        for c in chunks.iter_mut() {
            let v = vectors
                .next()
                .ok_or_else(|| CoreError::internal("the embedder returned too few vectors"))?;
            c.embedding = Some(v.clone());
            own.push(v);
        }
        means.push(unit_mean(&own));
    }
    Ok(means)
}

/// Drops each candidate that is a near duplicate of an active or pending
/// memory of the project, or of an earlier candidate kept from this attempt
/// (B-164, RC9). When the embedder or the lookup fails, every candidate is
/// kept, its chunks left for the embedding queue, and one line is logged.
fn distinct(
    sh: &Shared,
    ready: &Ready,
    job: &CaptureJob,
    entry: &ProjectEntry,
    tool: &str,
    mut prepared: Vec<Prepared>,
) -> Vec<Prepared> {
    let looked_up = guarded(|| {
        let means = embed(sh, ready, &mut prepared)?;
        let nearest = sh.store.capture_nearest(entry.project.id, &means)?;
        Ok((means, nearest))
    });
    let (means, nearest) = match looked_up {
        Ok(found) => found,
        Err(e) => {
            sh.log.line(format!(
                "capture {}: cannot check for near duplicates: {}; stored without the check, \
                 embedded later by the queue",
                job.request_id,
                cause_of(&e)
            ));
            for p in &mut prepared {
                for c in &mut p.input.chunks {
                    c.embedding = None;
                }
            }
            return prepared;
        }
    };
    let mut kept: Vec<Prepared> = Vec::with_capacity(prepared.len());
    let mut kept_means: Vec<&[f32]> = Vec::with_capacity(prepared.len());
    for ((p, mean), near) in prepared.into_iter().zip(&means).zip(nearest) {
        let closest = near
            .map(|n| n.cosine)
            .into_iter()
            .chain(kept_means.iter().map(|k| cosine(k, mean)))
            .reduce(f32::max);
        match closest {
            Some(c) if c >= CAPTURE_NEAR_DUPLICATE_COSINE => {
                journal_drop(sh, job, entry, tool, "near_duplicate", Some(c));
            }
            _ => {
                kept.push(p);
                kept_means.push(mean);
            }
        }
    }
    kept
}

/// Stores a job's candidates in one batch, after dropping those the
/// project has seen and the near duplicates. A candidate that can never be
/// stored, or that fails at the last attempt, is dropped and the others
/// are stored (one at a time when the batch fails); any other failure
/// fails the attempt, so the whole job is tried again (stream H 1).
fn store_candidates(
    sh: &Arc<Shared>,
    ready: &Ready,
    job: &CaptureJob,
    entry: &ProjectEntry,
    candidates: &[sm_capture::Candidate],
) -> Result<Attempt> {
    let tool = if job.replay.is_some() {
        REPLAY_TRIGGER
    } else {
        job.params
            .payload
            .get("tool_name")
            .and_then(Value::as_str)
            .unwrap_or(job.params.event.as_str())
    }
    .to_owned();
    let last = job.attempts + 1 >= MAX_ATTEMPTS;
    let droppable = |e: &CoreError| permanent(e) || last;
    let mut dropped: Vec<CoreError> = Vec::new();
    let fresh = unseen(sh, job, entry, &tool, candidates)?;
    let mut prepared = Vec::with_capacity(fresh.len());
    for c in fresh {
        match guarded(|| prepare(sh, ready, entry, c)) {
            Ok(p) => prepared.push(p),
            Err(e) if droppable(&e) => dropped.push(e),
            Err(e) => return Err(e),
        }
    }
    if !prepared.is_empty() {
        prepared = distinct(sh, ready, job, entry, &tool, prepared);
    }
    if !prepared.is_empty() {
        match guarded(|| store_batch(sh, job, entry, &tool, &prepared)) {
            Ok(()) => {}
            Err(e) if !droppable(&e) => return Err(e),
            Err(e) if prepared.len() == 1 => dropped.push(e),
            Err(_) => {
                for p in &prepared {
                    match guarded(|| store_batch(sh, job, entry, &tool, std::slice::from_ref(p))) {
                        Ok(()) => {}
                        Err(e) if droppable(&e) => dropped.push(e),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }
    let Some(first) = dropped.first() else {
        return Ok(Attempt::Stored);
    };
    let why = if permanent(first) {
        "it can never be stored".to_owned()
    } else {
        format!("failed {} times", job.attempts + 1)
    };
    Ok(Attempt::Dropped {
        cause: cause_of(first),
        why,
        dropped: dropped.len(),
        of: candidates.len(),
    })
}
