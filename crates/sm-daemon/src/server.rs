//! Start-up, the accept loop, connections and shutdown (CONTRACTS 3.1, 4;
//! SPEC 14, stream E).
//!
//! Order at start: the lock, the store and its index stamp, the endpoint,
//! then the loader. `hello` answers `loading` as soon as the endpoint is
//! bound. The loader creates the query session before the state becomes
//! `ready`; a load error leaves the daemon serving with `failed` and the
//! error (B-36).
//!
//! Order at shutdown: the `shutting_down` reply is written, the stop flag is
//! set, the accept loop is woken by a connection to itself and stops, the
//! endpoint is closed, in-flight requests and the workers get
//! `shutdown_grace` to finish, and the lock is released last.

use std::io::{ErrorKind, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use interprocess::local_socket::{Listener, Stream};
use sm_capture::RegexSecretScanner;
use sm_core::config::Settings;
use sm_core::error::{CoreError, Result};
use sm_core::journal::{self, Journal};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{FRAME_PREFIX_BYTES, Response, frame_len, write_frame};
use sm_core::spool::create_spool_dir;
use sm_core::time::Timestamp;
use sm_core::traits::SessionKind;
use sm_core::types::IndexStamp;
use sm_store::Store;

use crate::dispatch::{self, After};
use crate::endpoint;
use crate::loader::ModelLoader;
use crate::lock::DaemonLock;
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
}

/// The result of [`start`].
pub enum Started {
    LockHeld,
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
    /// A handle dropped without [`DaemonHandle::wait`] stops the daemon so
    /// no thread outlives it; the lock is released when the field drops.
    fn drop(&mut self) {
        if !self.finished {
            request_stop(&self.shared, Some("handle dropped".to_owned()));
            wake_accept(&self.shared);
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
    pub fn wait(mut self) -> Exit {
        let sh = Arc::clone(&self.shared);
        while !sh.stopping() {
            sh.maintenance_wake.wait(Duration::from_millis(200));
        }
        // Wake every waiter.
        sh.bulk_wake.notify();
        sh.capture_wake.notify();
        sh.maintenance_wake.notify();
        wake_accept(&sh);
        let deadline = Instant::now() + sh.opts.shutdown_grace;
        for t in self.threads.drain(..) {
            while !t.is_finished() && Instant::now() < deadline {
                sh.bulk_wake.notify();
                sh.capture_wake.notify();
                std::thread::sleep(Duration::from_millis(5));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
        while sh.in_flight.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
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
    // On an error the lock is dropped, which releases it.
    start_locked(paths, settings, loader, sharing, opts, log, daemon_lock).map(Started::Running)
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
        maps: Mutex::new(std::collections::HashMap::new()),
        bulk_wake: Signal::default(),
        bulk_open: AtomicUsize::new(0),
        captures: Mutex::new(std::collections::VecDeque::new()),
        capture_wake: Signal::default(),
        captures_pending: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        stop_reason: Mutex::new(None),
        last_activity: Mutex::new(now),
        in_flight: AtomicUsize::new(0),
        maintenance_wake: Signal::default(),
        endpoint,
        store,
        log,
        settings,
        opts,
        paths,
    });
    shared.log.line(format!(
        "pid {} listening on {} (protocol {}, build {})",
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

fn accept_loop(sh: &Arc<Shared>, listener: Listener) {
    loop {
        if sh.stopping() {
            break;
        }
        match endpoint::accept(&listener) {
            Ok(stream) => {
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
    // Dropping the listener closes the endpoint; interprocess removes the
    // socket file it created (reclaim_name).
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
                if written.is_ok() && Instant::now() <= deadline {
                    let _ = sh.store.record_injections(&session, &ids, Timestamp::now());
                }
            }
        }
        if written.is_err() {
            return;
        }
    }
}
