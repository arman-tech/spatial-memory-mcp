//! The bulk embedding queue (SPEC 5, Runtime; B-27): one worker thread
//! that embeds pending chunks with the daemon's bulk session.
//!
//! - It waits for the model, and does nothing while the stored index stamp
//!   differs from the daemon's (the store refuses vectors then; a reindex
//!   sets `index_ok`).
//! - It takes `embed.batch_size` pending chunks at a time, embeds them in
//!   one call, and stores them with one `set_embeddings` call.
//! - **One bulk session** (SPEC 3 rule 5; AT-3; e-report decision 35). The
//!   daemon has one, in `Shared::bulk`, opened on first use with
//!   `Settings::bulk_threads` threads (`cpu.budget` of the physical cores)
//!   and dropped once it has been idle for `bulk_grace`. The queue and
//!   `doctor_eval` both embed through [`with_bulk`], one batch at a time,
//!   so the bulk threads alive never exceed the budget.
//! - Every write that stores chunks wakes it; so does the model becoming
//!   ready, which is how chunks left pending by an earlier daemon are
//!   queued at start.
//! - Stored vectors change the maps, so each batch drops the cached runs.
//! - **Failures** (e-report decision 33). A storage failure (the database
//!   busy past its timeout, a full disk) or a refused index is retried
//!   after `bulk_error_backoff`. Any other failure of a batch, a panic in
//!   the embedder included (it is caught; the thread keeps running), is
//!   retried one chunk at a time, so the chunks that embed alone are
//!   stored. A chunk that fails alone [`SKIP_AFTER`] times is skipped with
//!   a log line naming it: it stays pending in the store (nothing is
//!   lost, `status` counts it), this daemon no longer offers it to the
//!   embedder, and a later daemon tries it again. A revise or a reindex
//!   gives the memory new chunks, which are not skipped. When no chunk
//!   embeds alone, the session is dropped (the next batch opens a fresh
//!   one) and the queue waits `bulk_error_backoff`.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::ids::ChunkId;
use sm_core::traits::{DocInput, Embedder, SessionKind};
use sm_store::{PendingChunk, Store};

use crate::state::{Ready, Shared, lock, panic_text};

/// Times a chunk may fail alone before this daemon skips it.
pub(crate) const SKIP_AFTER: u32 = 3;

/// The daemon's one bulk session.
pub(crate) struct BulkSession {
    session: Box<dyn Embedder>,
    idle_since: Instant,
}

fn log_error(sh: &Shared, what: &str, e: &CoreError) {
    sh.log.line(format!(
        "bulk queue: {what}: {}: {}",
        e.code.as_str(),
        e.message
    ));
}

fn open(sh: &Shared, ready: &Ready) -> Result<BulkSession> {
    let threads = sh.settings.bulk_threads(sh.physical_cores);
    let session = ready.provider.open(SessionKind::Bulk, threads)?;
    sh.bulk_open.fetch_add(1, Ordering::SeqCst);
    Ok(BulkSession {
        session,
        idle_since: Instant::now(),
    })
}

/// Opens the bulk session if it is not open.
fn ensure_open(sh: &Shared, ready: &Ready) -> Result<()> {
    let mut slot = lock(&sh.bulk);
    if slot.is_none() {
        *slot = Some(open(sh, ready)?);
    }
    Ok(())
}

/// Runs `f` on the daemon's bulk session, opening it if needed; one caller
/// at a time. A panic in `f` is an `internal` error.
pub(crate) fn with_bulk<T>(
    sh: &Shared,
    ready: &Ready,
    f: impl FnOnce(&dyn Embedder) -> Result<T>,
) -> Result<T> {
    let mut slot = lock(&sh.bulk);
    let mut b = match slot.take() {
        Some(b) => b,
        None => open(sh, ready)?,
    };
    let result = catch_unwind(AssertUnwindSafe(|| f(b.session.as_ref()))).unwrap_or_else(|p| {
        Err(CoreError::internal(format!(
            "the embedder panicked: {}",
            panic_text(p.as_ref())
        )))
    });
    b.idle_since = Instant::now();
    *slot = Some(b);
    result
}

/// Drops the bulk session.
pub(crate) fn release(sh: &Shared) {
    if lock(&sh.bulk).take().is_some() {
        sh.bulk_open.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Drops the session once idle for `bulk_grace`; the wait until the next
/// check.
fn release_idle(sh: &Shared) -> Duration {
    let mut slot = lock(&sh.bulk);
    let Some(idle) = slot.as_ref().map(|b| b.idle_since.elapsed()) else {
        return Duration::from_secs(5);
    };
    if idle < sh.opts.bulk_grace {
        return sh.opts.bulk_grace - idle;
    }
    *slot = None;
    sh.bulk_open.fetch_sub(1, Ordering::SeqCst);
    Duration::from_secs(5)
}

/// Embeds `batch` in one call on the bulk session and stores the vectors
/// in `store`.
pub(crate) fn embed_chunks(
    sh: &Shared,
    ready: &Ready,
    store: &Store,
    batch: &[PendingChunk],
) -> Result<()> {
    with_bulk(sh, ready, |session| {
        let docs: Vec<DocInput<'_>> = batch
            .iter()
            .map(|c| DocInput {
                title: c.title.as_deref(),
                text: &c.text,
            })
            .collect();
        let vectors = session.embed_documents(&docs)?;
        if vectors.len() != batch.len() {
            return Err(CoreError::internal(format!(
                "the embedder returned {} vectors for {} chunks",
                vectors.len(),
                batch.len()
            )));
        }
        let items: Vec<_> = batch.iter().map(|c| c.id).zip(vectors).collect();
        store.set_embeddings(&session.info().model_id, &items)?;
        Ok(())
    })
}

/// A failure of the store or the index, not of the chunks: waited out.
fn systemic(e: &CoreError) -> bool {
    !matches!(e.code, ErrorCode::Internal | ErrorCode::InvalidInput)
}

/// Chunks that failed alone, and those this daemon no longer offers.
#[derive(Default)]
struct Failures {
    strikes: HashMap<ChunkId, u32>,
    skipped: HashSet<ChunkId>,
}

impl Failures {
    fn failed_alone(&mut self, sh: &Shared, c: &PendingChunk, e: &CoreError) {
        let n = self.strikes.entry(c.id).or_insert(0);
        *n += 1;
        if *n < SKIP_AFTER {
            log_error(
                sh,
                &format!(
                    "chunk {} of memory {} failed alone ({n} of {SKIP_AFTER})",
                    c.id, c.memory_id
                ),
                e,
            );
            return;
        }
        self.strikes.remove(&c.id);
        self.skipped.insert(c.id);
        log_error(
            sh,
            &format!(
                "chunk {} of memory {} skipped after failing alone {SKIP_AFTER} times; \
                 it stays pending and a later daemon tries it again",
                c.id, c.memory_id
            ),
            e,
        );
    }
}

pub(crate) fn run(sh: &Arc<Shared>) {
    let mut failures = Failures::default();
    loop {
        if sh.stopping() {
            release(sh);
            return;
        }
        let Some(ready) = sh.ready_opt() else {
            sh.bulk_wake.wait(Duration::from_millis(250));
            continue;
        };
        if !sh.index_ok.load(Ordering::SeqCst) {
            release(sh);
            sh.bulk_wake.wait(Duration::from_secs(1));
            continue;
        }
        let batch_size = sh.settings.embed.batch_size.max(1) as usize;
        let batch: Vec<PendingChunk> =
            match sh.store.pending_chunks(batch_size + failures.skipped.len()) {
                Ok(b) => b
                    .into_iter()
                    .filter(|c| !failures.skipped.contains(&c.id))
                    .take(batch_size)
                    .collect(),
                Err(e) => {
                    log_error(sh, "reading pending chunks", &e);
                    sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
                    continue;
                }
            };
        if batch.is_empty() {
            sh.bulk_wake.wait(release_idle(sh));
            continue;
        }
        if let Err(e) = ensure_open(sh, &ready) {
            log_error(sh, "opening the bulk session", &e);
            sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
            continue;
        }
        let e = match embed_chunks(sh, &ready, &sh.store, &batch) {
            Ok(()) => {
                for c in &batch {
                    failures.strikes.remove(&c.id);
                }
                sh.drop_maps();
                continue;
            }
            Err(e) => e,
        };
        if systemic(&e) {
            log_error(sh, "storing a batch", &e);
            sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
            continue;
        }
        if !one_at_a_time(sh, &ready, &batch, &e, &mut failures) {
            release(sh);
            sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
        }
    }
}

/// After a batch failed on its chunks or the embedder: each chunk alone.
/// Returns whether any chunk was stored.
fn one_at_a_time(
    sh: &Shared,
    ready: &Ready,
    batch: &[PendingChunk],
    batch_error: &CoreError,
    failures: &mut Failures,
) -> bool {
    if let [c] = batch {
        failures.failed_alone(sh, c, batch_error);
        return false;
    }
    log_error(
        sh,
        &format!(
            "a batch of {} chunks failed; embedding them one at a time",
            batch.len()
        ),
        batch_error,
    );
    let mut stored = false;
    for c in batch {
        match embed_chunks(sh, ready, &sh.store, std::slice::from_ref(c)) {
            Ok(()) => {
                failures.strikes.remove(&c.id);
                stored = true;
            }
            Err(e) if systemic(&e) => {
                log_error(sh, "storing a chunk", &e);
                break;
            }
            Err(e) => failures.failed_alone(sh, c, &e),
        }
    }
    if stored {
        sh.drop_maps();
    }
    stored
}
