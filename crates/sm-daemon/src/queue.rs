//! The bulk embedding queue (SPEC 5, Runtime; B-27): one worker thread
//! that embeds pending chunks with a bulk session.
//!
//! - It waits for the model, and does nothing while the stored index stamp
//!   differs from the daemon's (the store refuses vectors then; a reindex
//!   sets `index_ok`).
//! - It takes `embed.batch_size` pending chunks at a time, embeds them in
//!   one call, and stores them with one `set_embeddings` call.
//! - The bulk session is opened on the first batch with
//!   `Settings::bulk_threads` threads (`cpu.budget` of the physical cores)
//!   and dropped once the queue has been empty for `bulk_grace`.
//! - Every write that stores chunks wakes it; so does the model becoming
//!   ready, which is how chunks left pending by an earlier daemon are
//!   queued at start.
//! - Stored vectors change the maps, so each batch drops the cached runs.
//! - A failed batch is logged and retried after `bulk_error_backoff`.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use sm_core::error::CoreError;
use sm_core::traits::{DocInput, Embedder, SessionKind};

use crate::state::{Ready, Shared};

struct Bulk {
    session: Box<dyn Embedder>,
    idle_since: Instant,
}

fn release(sh: &Shared, bulk: &mut Option<Bulk>) {
    if bulk.take().is_some() {
        sh.bulk_open.fetch_sub(1, Ordering::SeqCst);
    }
}

fn log_error(sh: &Shared, what: &str, e: &CoreError) {
    sh.log.line(format!(
        "bulk queue: {what}: {}: {}",
        e.code.as_str(),
        e.message
    ));
}

pub(crate) fn run(sh: &Arc<Shared>) {
    let mut bulk: Option<Bulk> = None;
    loop {
        if sh.stopping() {
            release(sh, &mut bulk);
            return;
        }
        let Some(ready) = sh.ready_opt() else {
            sh.bulk_wake.wait(Duration::from_millis(250));
            continue;
        };
        if !sh.index_ok.load(Ordering::SeqCst) {
            release(sh, &mut bulk);
            sh.bulk_wake.wait(Duration::from_secs(1));
            continue;
        }
        let batch_size = sh.settings.embed.batch_size.max(1) as usize;
        let batch = match sh.store.pending_chunks(batch_size) {
            Ok(b) => b,
            Err(e) => {
                log_error(sh, "reading pending chunks", &e);
                sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
                continue;
            }
        };
        if batch.is_empty() {
            let wait = match &bulk {
                Some(b) => {
                    let idle = b.idle_since.elapsed();
                    if idle >= sh.opts.bulk_grace {
                        release(sh, &mut bulk);
                        Duration::from_secs(5)
                    } else {
                        sh.opts.bulk_grace - idle
                    }
                }
                None => Duration::from_secs(5),
            };
            sh.bulk_wake.wait(wait);
            continue;
        }
        if bulk.is_none() {
            match open(sh, &ready) {
                Ok(session) => {
                    sh.bulk_open.fetch_add(1, Ordering::SeqCst);
                    bulk = Some(Bulk {
                        session,
                        idle_since: Instant::now(),
                    });
                }
                Err(e) => {
                    log_error(sh, "opening the bulk session", &e);
                    sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
                    continue;
                }
            }
        }
        let Some(b) = bulk.as_mut() else {
            continue;
        };
        let docs: Vec<DocInput<'_>> = batch
            .iter()
            .map(|c| DocInput {
                title: c.title.as_deref(),
                text: &c.text,
            })
            .collect();
        let stored = b.session.embed_documents(&docs).and_then(|vectors| {
            if vectors.len() != batch.len() {
                return Err(CoreError::internal(format!(
                    "the embedder returned {} vectors for {} chunks",
                    vectors.len(),
                    batch.len()
                )));
            }
            let items: Vec<_> = batch.iter().map(|c| c.id).zip(vectors).collect();
            sh.store.set_embeddings(&b.session.info().model_id, &items)
        });
        b.idle_since = Instant::now();
        match stored {
            Ok(_) => sh.drop_maps(),
            Err(e) => {
                log_error(sh, "embedding a batch", &e);
                sh.bulk_wake.wait(sh.opts.bulk_error_backoff);
            }
        }
    }
}

fn open(sh: &Shared, ready: &Ready) -> sm_core::error::Result<Box<dyn Embedder>> {
    let threads = sh.settings.bulk_threads(sh.physical_cores);
    ready.provider.open(SessionKind::Bulk, threads)
}
