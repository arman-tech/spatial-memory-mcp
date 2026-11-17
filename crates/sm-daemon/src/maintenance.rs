//! The maintenance thread: the spool drain, the daily pruning and the idle
//! exit (SPEC 16, Spool; SPEC 6, Housekeeping; CONTRACTS 4, 8; B-25, B-33,
//! B-66).
//!
//! **Spool drain**, at start and every `spool_interval` (60 s):
//! `remove_stale_tmp`, then `list_spool` in name order. Each file is read
//! with `read_spool` and dispatched with `dispatch::handle_request`, the
//! path a socket request takes after its envelope is decoded. Then:
//!
//! - an accepted capture: the file stays in `spool/new/`, held by its job
//!   in `Shared::held`, and the capture worker removes it only after the
//!   capture is stored (e-report decision 28; CONTRACTS 7 said "after its
//!   capture is queued"). A file already held is skipped, so one daemon
//!   never queues a file twice;
//! - a file that cannot be parsed, a request for another profile or one
//!   that fails validation, or a capture the daemon does not accept: the
//!   file is moved to `spool/failed/` with `spool::quarantine` and the
//!   reason is logged; nothing there is retried (B-33);
//! - a file that cannot be read, or any other error: the file stays for
//!   the next drain and is logged;
//! - the file of a capture already stored that could not be removed
//!   (`Shared::unremovable`): it is never processed again. Each drain tries
//!   the removal again, quietly, and forgets the name once the file is
//!   gone; the one log line was written when the removal first failed.
//!
//! The drain stops early when the daemon stops or the capture queue is
//! full, and is skipped while the model is in the `failed` state: those
//! captures cannot be chunked, so the files are left for a later daemon
//! rather than read and spooled again every minute.
//!
//! **Pruning**, at start and every `maintenance_interval` (a day):
//! injection rows past their retention (SPEC 6) and journal files past
//! theirs (`journal::prune`).
//!
//! **Idle exit**: when `daemon.idle_exit_minutes` is above 0 and no request
//! has been served for that long, with nothing in flight, no capture
//! queued and no bulk session open, the daemon stops itself.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};

use sm_core::error::{CoreError, ErrorCode};
use sm_core::journal;
use sm_core::protocol::{Outcome, Reply};
use sm_core::spool::{STALE_TMP_AGE, list_spool, quarantine, read_spool, remove_stale_tmp};
use sm_core::time::Timestamp;

use crate::capture::{MAX_QUEUED, claim, release, retry_removal};
use crate::dispatch::handle_request;
use crate::server::request_stop;
use crate::state::{ModelState, Shared, lock};

/// Prunes injection rows and journal files past their retention.
pub(crate) fn prune(sh: &Shared) {
    let now = Timestamp::now();
    match sh.store.prune_injections(now) {
        Ok(0) => {}
        Ok(n) => sh.log.line(format!("pruned {n} injection rows")),
        Err(e) => log_error(sh, "pruning injections", &e),
    }
    journal::prune(&sh.paths.metrics_dir(), now);
}

fn log_error(sh: &Shared, what: &str, e: &CoreError) {
    sh.log.line(format!(
        "maintenance: {what}: {}: {}",
        e.code.as_str(),
        e.message
    ));
}

/// The maintenance thread: runs until the daemon stops.
pub(crate) fn run(sh: &Arc<Shared>) {
    let mut next_drain = Instant::now();
    let mut next_prune = Instant::now() + sh.opts.maintenance_interval;
    loop {
        if sh.stopping() {
            return;
        }
        if Instant::now() >= next_drain {
            drain(sh);
            next_drain = Instant::now() + sh.opts.spool_interval;
        }
        if Instant::now() >= next_prune {
            prune(sh);
            next_prune = Instant::now() + sh.opts.maintenance_interval;
        }
        if idle(sh) {
            sh.log
                .line(idle_exit_line(sh.settings.daemon.idle_exit_minutes));
            request_stop(sh, None);
            return;
        }
        let until = next_drain
            .min(next_prune)
            .saturating_duration_since(Instant::now());
        // Short enough for the idle check; the stop flag also wakes it.
        sh.maintenance_wake
            .wait(until.clamp(Duration::from_millis(10), Duration::from_millis(250)));
    }
}

/// The log line written when the daemon stops for being idle: "1 minute"
/// for one, "N minutes" otherwise.
fn idle_exit_line(minutes: u32) -> String {
    let unit = if minutes == 1 { "minute" } else { "minutes" };
    format!("no request for {minutes} {unit}; exiting")
}

fn idle(sh: &Shared) -> bool {
    let minutes = sh.settings.daemon.idle_exit_minutes;
    if minutes == 0 {
        return false;
    }
    let limit = sh.opts.idle_unit.saturating_mul(minutes);
    lock(&sh.last_activity).elapsed() >= limit
        && sh.in_flight.load(Ordering::SeqCst) == 0
        && sh.captures_pending.load(Ordering::SeqCst) == 0
        && sh.bulk_open.load(Ordering::SeqCst) == 0
}

/// One pass over `spool/new/`.
pub(crate) fn drain(sh: &Arc<Shared>) {
    if matches!(
        &*sh.model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        ModelState::Failed(_)
    ) {
        return;
    }
    if let Err(e) = remove_stale_tmp(&sh.paths.spool_tmp(), SystemTime::now(), STALE_TMP_AGE) {
        log_error(sh, "removing stale spool writes", &e);
    }
    let files = match list_spool(&sh.paths.spool_new()) {
        Ok(f) => f,
        Err(e) => {
            log_error(sh, "listing the spool", &e);
            return;
        }
    };
    for file in files {
        if sh.stopping() || lock(&sh.captures).len() >= MAX_QUEUED {
            return;
        }
        drain_one(sh, &file);
    }
}

fn drain_one(sh: &Arc<Shared>, file: &Path) {
    if retry_removal(sh, file) || !claim(sh, file) {
        return;
    }
    let req = match read_spool(file) {
        Ok(r) => r,
        Err(e) if e.code == ErrorCode::InvalidInput => {
            to_failed(sh, file, &e.message);
            release(sh, file);
            return;
        }
        Err(e) => {
            // Gone meanwhile (stored by a job that held it), or unreadable.
            if file.exists() {
                kept(sh, file, &e);
            }
            release(sh, file);
            return;
        }
    };
    let (response, _) = handle_request(sh, &req, Instant::now(), Some(file));
    match response.result {
        // The job holds the file now, or released it where it is.
        Outcome::Ok(reply) if matches!(*reply, Reply::Captured(ref c) if c.accepted) => {}
        Outcome::Ok(_) => {
            to_failed(sh, file, "the daemon did not accept the capture");
            release(sh, file);
        }
        Outcome::Err(e) if e.code == ErrorCode::InvalidInput => {
            to_failed(sh, file, &e.message);
            release(sh, file);
        }
        Outcome::Err(e) => {
            kept(sh, file, &e);
            release(sh, file);
        }
    }
}

fn kept(sh: &Shared, file: &Path, e: &CoreError) {
    sh.log.line(format!(
        "spool: {} kept for the next drain: {}: {}",
        file.display(),
        e.code.as_str(),
        e.message
    ));
}

fn to_failed(sh: &Shared, file: &Path, why: &str) {
    match quarantine(file, &sh.paths.spool_failed()) {
        Ok(dst) => sh.log.line(format!(
            "spool: {} moved to {}: {why}",
            file.display(),
            dst.display()
        )),
        Err(e) => sh.log.line(format!(
            "spool: {} cannot be moved to spool/failed ({}); {why}",
            file.display(),
            e.message
        )),
    }
}
