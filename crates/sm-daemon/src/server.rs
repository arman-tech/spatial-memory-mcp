//! Start-up, the accept loop, connections and shutdown (CONTRACTS 3.1, 4;
//! SPEC 14, stream E).
//!
//! Order at start: the lock, the store and its index stamp, the endpoint,
//! then the loader. `hello` answers `loading` as soon as the endpoint is
//! bound. The loader creates the query session before the state becomes
//! `ready`; a load error leaves the daemon serving with `failed` and the
//! error (B-36).
//!
//! Order at shutdown (CONTRACTS 3.1; e-report decision 29): the
//! `shutting_down` reply is written and the stop flag is set. Then
//! `closed` is set, so a request still arriving on an open connection gets
//! `daemon_unavailable`; the accept loop is woken by a connection to itself
//! and stops, which closes the endpoint; the workers and the requests in
//! flight get `shutdown_grace` to finish; every capture still queued is
//! written to `spool/new/`, while a capture being processed is already a
//! file there (decision 28); and the lock is released last. A thread still
//! running after the grace holds nothing that is not on disk: a capture
//! request in flight finds the stop flag and spools itself, and the
//! worker's job is a file in `spool/new/` that the next daemon processes.

use std::io::{ErrorKind, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use interprocess::local_socket::Stream;
use sm_capture::RegexSecretScanner;
use sm_core::config::Settings;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::journal::{self, Journal};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{Envelope, FRAME_PREFIX_BYTES, Response, frame_len, write_frame};
use sm_core::spool::create_spool_dir;
use sm_core::time::Timestamp;
use sm_core::traits::SessionKind;
use sm_core::types::IndexStamp;
use sm_store::Store;

use crate::dispatch::{self, After};
use crate::endpoint;
use crate::loader::ModelLoader;
use crate::lock::{DaemonLock, read_hold};
use crate::log::DaemonLog;
use crate::options::DaemonOptions;
use crate::sharing::SharingOps;
use crate::state::{Lru, ModelState, Ready, Shared, Signal, lock};
use crate::{capture, maintenance, queue};

/// Why the daemon stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit {
    /// A `shutdown` request, or [`DaemonHandle::shutdown`].
    Shutdown { reason: String },
    /// No request for `daemon.idle_exit_minutes`.
    IdleExit,
    /// Another process held `daemon.lock` for the whole wait; this one did
    /// nothing and exits 0 (CONTRACTS 4, Auto-start).
    LockHeld,
    /// The profile is held stopped for an upgrade (B-180): this one took
    /// the lock, found `daemon.hold`, did nothing and exits 0.
    Held { hold: String },
}

/// The result of [`start`].
pub enum Started {
    LockHeld,
    /// `daemon.hold` exists (B-180); the lock is released again.
    Held {
        hold: String,
    },
    Running(DaemonHandle),
}

/// A running daemon.
pub struct DaemonHandle {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    lock: Option<DaemonLock>,
    finished: bool,
}

impl Drop for DaemonHandle {
    /// A handle dropped without [`DaemonHandle::wait`] stops the daemon in
    /// the same order as `wait`, so the lock is released last.
    fn drop(&mut self) {
        if !self.finished {
            request_stop(&self.shared, Some("handle dropped".to_owned()));
            self.finish();
        }
    }
}

impl DaemonHandle {
    /// The endpoint clients connect to.
    pub fn endpoint(&self) -> &Endpoint {
        &self.shared.endpoint
    }

    /// The daemon process's log.
    pub fn log_path(&self) -> &std::path::Path {
        self.shared.log.path()
    }

    /// Whether the daemon has been asked to stop.
    pub fn stopping(&self) -> bool {
        self.shared.stopping()
    }

    /// Asks the daemon to stop and waits for it.
    pub fn shutdown(self, reason: &str) -> Exit {
        request_stop(&self.shared, Some(reason.to_owned()));
        self.wait()
    }

    /// Blocks until the daemon stops (a `shutdown` request or idle exit),
    /// then closes the endpoint, waits for the workers and releases the
    /// lock last.
    ///
    /// **Threads can outlive this call.** The workers, the requests in
    /// flight and the connection threads get `shutdown_grace` to finish;
    /// a thread still running after it (a long embedding batch, a request
    /// blocked on a slow client) is not joined and keeps running in this
    /// process after `wait` returns, possibly still writing to the store
    /// or the log. It holds nothing that is not on disk (see the module
    /// documentation), but it is not stopped. The caller must therefore
    /// exit the process once `wait` returns, as the `daemon` subcommand
    /// does; a caller that keeps the process alive (a test, an embedding
    /// program) must not start another daemon on the same profile from it
    /// and expect exclusive access. The same holds for
    /// [`DaemonHandle::shutdown`] and for dropping the handle.
    pub fn wait(mut self) -> Exit {
        while !self.shared.stopping() {
            self.shared
                .maintenance_wake
                .wait(Duration::from_millis(200));
        }
        self.finish()
    }

    /// The shutdown order of the module documentation, once stop is set.
    fn finish(&mut self) -> Exit {
        let sh = Arc::clone(&self.shared);
        // New requests on open connections are refused from here on.
        sh.closed.store(true, Ordering::SeqCst);
        // Wake every waiter.
        sh.bulk_wake.notify();
        sh.capture_wake.notify();
        sh.maintenance_wake.notify();
        sh.judge_wake.notify();
        wake_accept(&sh);
        let deadline = Instant::now() + sh.opts.shutdown_grace;
        for t in self.threads.drain(..) {
            while !t.is_finished() && Instant::now() < deadline {
                sh.bulk_wake.notify();
                sh.capture_wake.notify();
                sh.judge_wake.notify();
                std::thread::sleep(Duration::from_millis(5));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
        // Read after `closed` was set: once it is 0, no request can start.
        while sh.in_flight.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        // The worker spools the queue when it stops; this covers a worker
        // still busy after the grace.
        capture::spool_queued(&sh);
        let reason = lock(&sh.stop_reason).clone();
        let exit = match reason {
            Some(r) => Exit::Shutdown { reason: r },
            None => Exit::IdleExit,
        };
        sh.log.line(format!("stopped: {exit:?}"));
        self.finished = true;
        if let Some(l) = self.lock.take() {
            l.release();
        }
        exit
    }
}

/// Sets the stop flag once. `reason` is `None` for an idle exit.
pub(crate) fn request_stop(sh: &Shared, reason: Option<String>) {
    {
        let mut r = lock(&sh.stop_reason);
        if sh.stop.load(Ordering::SeqCst) {
            return;
        }
        *r = reason;
        sh.stop.store(true, Ordering::SeqCst);
    }
    sh.maintenance_wake.notify();
    sh.bulk_wake.notify();
    sh.capture_wake.notify();
    sh.judge_wake.notify();
}

/// A connection to our own endpoint unblocks `accept` once stop is set.
fn wake_accept(sh: &Shared) {
    if let Ok(s) = endpoint::connect(&sh.endpoint, Duration::from_millis(250)) {
        drop(s);
    }
}

/// Starts a daemon for `paths`. Returns once the endpoint is bound and the
/// loader has started.
pub fn start(
    paths: ProfilePaths,
    settings: Settings,
    loader: Arc<dyn ModelLoader>,
    sharing: Arc<dyn SharingOps>,
    opts: DaemonOptions,
) -> Result<Started> {
    let log = DaemonLog::new(paths.daemon_log(), opts.log_max_bytes);
    // Before the lock file is created in it.
    #[cfg(unix)]
    {
        let me = endpoint::current_user().map_err(|e| {
            CoreError::new(
                ErrorCode::DaemonUnavailable,
                format!("cannot read the current user: {e}"),
            )
        })?;
        if let Some(line) = private_profile(&paths.dir, me)? {
            log.line(line);
        }
    }
    let Some(daemon_lock) =
        DaemonLock::acquire(&paths.lock_file(), opts.lock_wait, opts.retry_every)?
    else {
        log.line(format!(
            "pid {}: another daemon holds {}; exiting",
            std::process::id(),
            paths.lock_file().display()
        ));
        return Ok(Started::LockHeld);
    };
    // B-180: read with the lock held, so no daemon starts once the hold is
    // written, however long it waited for the lock.
    if let Some(hold) = read_hold(&paths) {
        log.line(format!(
            "pid {}: the profile is held stopped ({hold}); exiting until `spatial-memory release`",
            std::process::id(),
        ));
        daemon_lock.release();
        return Ok(Started::Held { hold });
    }
    // On an error the lock is dropped, which releases it.
    start_locked(paths, settings, loader, sharing, opts, log, daemon_lock).map(Started::Running)
}

/// Makes the profile folder user-only on Unix (SPEC 3 rule 1, SPEC 9
/// "Security"): it holds `memory.db`, `daemon.log`, `metrics/` and
/// `backups/`. A missing folder is created with mode 0700 (its missing
/// parents with the default mode). An existing folder of this user that
/// group or others can reach is set to 0700, and the returned line, which
/// names it, is logged. A folder another user owns is refused with
/// `daemon_unavailable`, before anything is written in it. The folder is
/// read through a link, so a profile folder the user moved elsewhere and
/// linked keeps working.
///
/// Windows is left as it is: the profile folder is under `%LOCALAPPDATA%`,
/// whose access list admits only the user, SYSTEM and administrators, and
/// is inherited by every folder created in it.
#[cfg(unix)]
fn private_profile(dir: &std::path::Path, me: u32) -> Result<Option<String>> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    let refuse = |why: String| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("refusing the profile folder {}: {why}", dir.display()),
        )
    };
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| refuse(format!("cannot create the folder that holds it: {e}")))?;
    }
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(refuse(format!("cannot create it: {e}"))),
    }
    let m = std::fs::metadata(dir).map_err(|e| refuse(format!("cannot read it: {e}")))?;
    if !m.is_dir() {
        return Err(refuse("it is not a folder".to_owned()));
    }
    if m.uid() != me {
        return Err(refuse(format!(
            "it is owned by another user (uid {}), not by this user (uid {me})",
            m.uid()
        ))
        .with_next_action(format!(
            "Use a profile folder of your own: set SPATIAL_MEMORY_HOME to a folder you own, \
             or have that user or an administrator remove {} or give it to you, then start \
             the daemon again.",
            dir.display()
        )));
    }
    let mode = m.mode() & 0o777;
    if mode & 0o077 == 0 {
        return Ok(None);
    }
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .map_err(|e| refuse(format!("cannot set it to mode 700 (it is {mode:o}): {e}")))?;
    Ok(Some(format!(
        "set the profile folder {} to mode 700: it was {mode:o}, open to group or others",
        dir.display()
    )))
}

/// Starts the daemon and blocks until it stops.
pub fn run(
    paths: ProfilePaths,
    settings: Settings,
    loader: Arc<dyn ModelLoader>,
    sharing: Arc<dyn SharingOps>,
    opts: DaemonOptions,
) -> Result<Exit> {
    match start(paths, settings, loader, sharing, opts)? {
        Started::LockHeld => Ok(Exit::LockHeld),
        Started::Held { hold } => Ok(Exit::Held { hold }),
        Started::Running(h) => Ok(h.wait()),
    }
}

fn start_locked(
    paths: ProfilePaths,
    settings: Settings,
    loader: Arc<dyn ModelLoader>,
    sharing: Arc<dyn SharingOps>,
    opts: DaemonOptions,
    log: DaemonLog,
    daemon_lock: DaemonLock,
) -> Result<DaemonHandle> {
    let store = Store::open(&paths)?;
    let info = loader.info();
    let status = store.configure_index(IndexStamp {
        model_id: info.model_id.clone(),
        revision: info.revision.clone(),
        dim: info.dim,
        chunk_max_tokens: settings.chunk.max_tokens,
        chunk_overlap: settings.chunk.overlap,
    })?;
    if !status.matches {
        log.line(format!(
            "index stamp differs ({}); searches give index_mismatch until `spatial-memory reindex`",
            status.differences.join(", ")
        ));
    }
    for dir in [paths.spool_tmp(), paths.spool_new(), paths.spool_failed()] {
        create_spool_dir(&dir)?;
    }
    let endpoint = opts.endpoint.clone().unwrap_or_else(|| paths.endpoint());
    let listener = endpoint::bind(&endpoint, opts.bind_wait, opts.retry_every)?;
    let physical = opts
        .physical_cores
        .unwrap_or_else(sm_embed::physical_cores)
        .max(1);
    let journal = Journal::new(paths.metrics_dir(), settings.journal.include_prompts);
    let now = Instant::now();
    let shared = Arc::new(Shared {
        journal,
        sharing,
        scanner: RegexSecretScanner::new(),
        info,
        physical_cores: physical,
        started_at: Timestamp::now(),
        model: RwLock::new(ModelState::Loading),
        ready_seconds: Mutex::new(None),
        index_ok: AtomicBool::new(status.matches),
        projects: Mutex::new(Lru::new(256)),
        ranks: Mutex::new(Lru::new(opts.rank_sessions)),
        maps: Mutex::new(crate::state::MapIndex::default()),
        bulk_wake: Signal::default(),
        bulk: Mutex::new(None),
        bulk_open: AtomicUsize::new(0),
        captures: Mutex::new(std::collections::VecDeque::new()),
        capture_wake: Signal::default(),
        captures_pending: AtomicUsize::new(0),
        held: Mutex::new(std::collections::HashSet::new()),
        unremovable: Mutex::new(std::collections::HashSet::new()),
        taking: Mutex::new(()),
        revising: Mutex::new(()),
        stop: AtomicBool::new(false),
        closed: AtomicBool::new(false),
        stop_reason: Mutex::new(None),
        last_activity: Mutex::new(now),
        in_flight: AtomicUsize::new(0),
        maintenance_wake: Signal::default(),
        judge_wake: Signal::default(),
        judge_state: Mutex::new(crate::judge::JudgeRuntime::default()),
        endpoint,
        store,
        log,
        settings,
        opts,
        paths,
    });
    // The build as `spatial-memory --version` prints it (B-231).
    let commit = shared
        .opts
        .commit
        .as_deref()
        .map(|c| format!("+{c}"))
        .unwrap_or_default();
    shared.log.line(format!(
        "pid {} listening on {} (protocol {}, build {}{commit})",
        std::process::id(),
        shared.endpoint.display_path(),
        sm_core::protocol::PROTOCOL_VERSION,
        sm_core::BUILD
    ));
    // Injections older than the retention are dropped at start and daily.
    maintenance::prune(&shared);
    let mut threads = Vec::new();
    threads.push(spawn("sm-accept", {
        let sh = Arc::clone(&shared);
        move || accept_loop(&sh, listener)
    })?);
    threads.push(spawn("sm-loader", {
        let sh = Arc::clone(&shared);
        move || load_model(&sh, loader.as_ref())
    })?);
    threads.push(spawn("sm-bulk", {
        let sh = Arc::clone(&shared);
        move || queue::run(&sh)
    })?);
    threads.push(spawn("sm-capture", {
        let sh = Arc::clone(&shared);
        move || capture::run(&sh)
    })?);
    // The judge runs only when automatic capture and the judge are on, and
    // the daemon was given one (B-168).
    let judging = shared.settings.capture.enabled && shared.settings.capture.judge.enabled;
    if let Some(judge) = shared.opts.judge.clone().filter(|_| judging) {
        threads.push(spawn("sm-judge", {
            let sh = Arc::clone(&shared);
            move || crate::judge::run(&sh, judge.as_ref())
        })?);
    }
    threads.push(spawn("sm-maintenance", {
        let sh = Arc::clone(&shared);
        move || maintenance::run(&sh)
    })?);
    Ok(DaemonHandle {
        shared,
        threads,
        lock: Some(daemon_lock),
        finished: false,
    })
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(f)
        .map_err(|e| CoreError::internal(format!("cannot start thread {name}: {e}")))
}

/// Loads the model, then opens the query session, then goes ready.
fn load_model(sh: &Shared, loader: &dyn ModelLoader) {
    let t0 = Instant::now();
    let loaded = loader.load().and_then(|m| {
        let threads = sh.settings.query_threads(sh.physical_cores);
        let query = m.provider.open(SessionKind::Query, threads)?;
        Ok(Ready {
            provider: m.provider,
            chunker: m.chunker,
            query,
        })
    });
    let seconds = t0.elapsed().as_secs_f64();
    match loaded {
        Ok(ready) => {
            *sh.model
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                ModelState::Ready(Arc::new(ready));
            *lock(&sh.ready_seconds) = Some(seconds);
            sh.journal.record(
                journal::STARTUP_REQUEST_ID,
                journal::metrics::EMBEDDER_COLD_START_SECONDS,
                seconds,
                Default::default(),
            );
            sh.log.line(format!("model ready in {seconds:.2} s"));
            sh.bulk_wake.notify();
            sh.capture_wake.notify();
            sh.judge_wake.notify();
        }
        Err(e) => {
            sh.log
                .line(format!("model failed to load: {:?}: {}", e.code, e.message));
            *sh.model
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = ModelState::Failed(e);
            // Captures queued while loading are spooled for a later daemon.
            sh.capture_wake.notify();
        }
    }
}

fn accept_loop(sh: &Arc<Shared>, listener: endpoint::Bound) {
    loop {
        if sh.stopping() {
            break;
        }
        match listener.accept() {
            Ok(endpoint::Accepted::Refused(why)) => {
                // Dropped before anything was read (B-95); the next
                // connection is taken at once.
                sh.log.line(why);
            }
            Ok(endpoint::Accepted::Client(stream)) => {
                if sh.stopping() {
                    break;
                }
                let conn = Arc::clone(sh);
                let spawned = std::thread::Builder::new()
                    .name("sm-conn".to_owned())
                    .spawn(move || serve(&conn, stream));
                if let Err(e) = spawned {
                    sh.log
                        .line(format!("cannot start a connection thread: {e}"));
                }
            }
            Err(e) => {
                if sh.stopping() {
                    break;
                }
                sh.log.line(format!("accept failed: {e}"));
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    }
    // Dropping the listener closes the endpoint and removes the socket
    // file (interprocess's reclaim_name, or `Bound` for a moved socket).
    drop(listener);
}

enum Frame {
    Body(Vec<u8>),
    Closed,
    /// A zero-length frame: refused, the connection stays usable.
    Empty,
    /// An oversized frame or a broken read: refused, the connection closes.
    Fatal(Option<CoreError>),
}

fn read_body(s: &mut Stream) -> Frame {
    let mut prefix = [0u8; FRAME_PREFIX_BYTES];
    let mut got = 0;
    while got < FRAME_PREFIX_BYTES {
        match s.read(&mut prefix[got..]) {
            Ok(0) if got == 0 => return Frame::Closed,
            Ok(0) => return Frame::Fatal(None),
            Ok(n) => got += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e)
                if got == 0
                    && matches!(
                        e.kind(),
                        ErrorKind::BrokenPipe
                            | ErrorKind::ConnectionReset
                            | ErrorKind::ConnectionAborted
                            | ErrorKind::UnexpectedEof
                    ) =>
            {
                return Frame::Closed;
            }
            Err(_) => return Frame::Fatal(None),
        }
    }
    match frame_len(prefix) {
        Ok(len) => {
            let mut body = vec![0u8; len];
            match s.read_exact(&mut body) {
                Ok(()) => Frame::Body(body),
                Err(_) => Frame::Fatal(None),
            }
        }
        Err(_) if u32::from_le_bytes(prefix) == 0 => Frame::Empty,
        Err(e) => Frame::Fatal(Some(e)),
    }
}

/// The reply to a request that arrives after shutdown began.
fn stopped(body: &[u8]) -> Response {
    let id = Envelope::peek(body).map(|e| e.id).unwrap_or_default();
    Response::err(
        id,
        CoreError::new(ErrorCode::DaemonUnavailable, "the daemon has stopped")
            .with_next_action("Connect again; a new daemon starts on demand."),
    )
}

/// Serves one connection until the peer closes it.
fn serve(sh: &Arc<Shared>, mut stream: Stream) {
    loop {
        let body = match read_body(&mut stream) {
            Frame::Body(b) => b,
            Frame::Closed => return,
            Frame::Empty => {
                let r = Response::err(String::new(), CoreError::invalid_input("empty frame"));
                if write_frame(&mut stream, &r).is_err() {
                    return;
                }
                continue;
            }
            Frame::Fatal(e) => {
                if let Some(e) = e {
                    let _ = write_frame(&mut stream, &Response::err(String::new(), e));
                }
                return;
            }
        };
        sh.in_flight.fetch_add(1, Ordering::SeqCst);
        // Read after the increment: shutdown sets `closed`, then waits for
        // `in_flight` to reach 0, so no request starts after that.
        if sh.closed.load(Ordering::SeqCst) {
            sh.in_flight.fetch_sub(1, Ordering::SeqCst);
            let _ = write_frame(&mut stream, &stopped(&body));
            return;
        }
        sh.touch();
        let (response, after) = dispatch::handle(sh, &body);
        let written = write_frame(&mut stream, &response);
        sh.touch();
        sh.in_flight.fetch_sub(1, Ordering::SeqCst);
        match after {
            After::Nothing => {}
            After::Shutdown(reason) => {
                sh.log.line(format!("shutdown requested: {reason}"));
                request_stop(sh, Some(reason));
                return;
            }
            After::RecordInjections {
                session,
                ids,
                deadline,
            } => {
                // Off the reply path, and only if the reply left in time
                // (SPEC 6, step 6).
                if written.is_ok()
                    && Instant::now() <= deadline
                    && let Err(e) = sh.store.record_injections(&session, &ids, Timestamp::now())
                {
                    // Not recorded: these memories may be injected again
                    // in this session (SPEC 6, never repeat).
                    sh.log.line(format!(
                        "session {session}: {} injected memories not recorded: {}: {}",
                        ids.len(),
                        e.code.as_str(),
                        e.message
                    ));
                }
            }
        }
        if written.is_err() {
            return;
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    use super::*;

    fn me() -> u32 {
        endpoint::current_user().expect("the current user")
    }

    fn mode(dir: &std::path::Path) -> u32 {
        std::fs::metadata(dir).expect("the folder").mode() & 0o777
    }

    /// Stream H item 17: a new profile folder is created user-only, and
    /// an existing one open to group or others is set to 0700 and logged.
    #[test]
    fn s3_the_profile_folder_is_user_only() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let new = tmp.path().join("root").join("test");
        assert_eq!(private_profile(&new, me()).expect("created"), None);
        assert_eq!(mode(&new), 0o700);
        assert_eq!(std::fs::metadata(&new).expect("the folder").uid(), me());

        let open = tmp.path().join("open");
        std::fs::create_dir(&open).expect("the folder");
        for before in [0o755, 0o750, 0o705, 0o701, 0o777] {
            std::fs::set_permissions(&open, std::fs::Permissions::from_mode(before))
                .expect("chmod");
            let line = private_profile(&open, me())
                .expect("made private")
                .expect("a line for the log");
            assert!(line.contains(&open.display().to_string()), "{line}");
            assert!(line.contains(&format!("{before:o}")), "{line}");
            assert_eq!(mode(&open), 0o700, "from {before:o}");
        }
        // Already private: nothing to do, nothing to log.
        assert_eq!(private_profile(&open, me()).expect("private"), None);
        assert_eq!(mode(&open), 0o700);
    }

    /// Stream H item 17: a profile folder another user owns is refused,
    /// with a next action, and left as it is.
    #[test]
    fn s3_a_profile_folder_another_user_owns_is_refused() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let dir = tmp.path().join("theirs");
        std::fs::create_dir(&dir).expect("the folder");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let e = private_profile(&dir, me().wrapping_add(1)).expect_err("another user's folder");
        assert_eq!(e.code, ErrorCode::DaemonUnavailable);
        assert!(e.message.contains("owned by another user"), "{}", e.message);
        assert!(
            e.message.contains(&dir.display().to_string()),
            "{}",
            e.message
        );
        assert!(
            e.next_action.contains("SPATIAL_MEMORY_HOME"),
            "{}",
            e.next_action
        );
        assert_eq!(mode(&dir), 0o755, "another user's folder is not changed");
        // A file where the folder should be.
        let file = tmp.path().join("file");
        std::fs::write(&file, b"not a folder").expect("a file");
        let e = private_profile(&file, me()).expect_err("not a folder");
        assert!(e.message.contains("not a folder"), "{}", e.message);
    }
}
