//! Test support shared by the integration tests: a mock daemon that speaks
//! the frozen message schema on a profile's endpoint, an isolated runner for
//! the `spatial-memory` binary, and the capture fixtures.
//!
//! Every test gets its own [`Sandbox`]: a temporary `SPATIAL_MEMORY_HOME`
//! (so its own endpoint), a temporary user home, and a `{TMP}` folder for
//! fixture files. Nothing reads or writes the real profile, `~/.claude` or
//! `~/.cursor`, and the binary runs with a cleared environment.
//!
//! Two things a sandboxed binary run can no longer do since the binary
//! carries the real daemon and model step (phase 3, package J):
//!
//! - **Download the model.** `SPATIAL_MEMORY_MODEL_URL` points at a local
//!   server that answers 404 to everything ([`Sandbox::model_requests`]),
//!   and `SPATIAL_MEMORY_MODEL_DIR` at an empty folder.
//! - **Leave a daemon running.** In a [`Sandbox::new`] no daemon can start:
//!   `memory.db` is a folder, so a daemon a hook spawns takes the lock,
//!   fails to open the store and exits 1, as the unparsed `daemon`
//!   subcommand did before. Such a daemon can start after the test ends,
//!   so when a start was tried the sandbox keeps the block at drop and
//!   deletes the rest; a later test binary sweeps the kept folders. A
//!   [`Sandbox::with_daemon`] lets the daemon start, and dropping the
//!   sandbox shuts it down. Every sandbox is a `sb-*` folder under
//!   `spatial-memory-cli-tests` in the system's temporary folder.

#![allow(dead_code)]

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use interprocess::ConnectWaitMode;
use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{ConnectOptions, ListenerOptions, Stream};
use serde_json::Value;
use sm_cli::client::Spawn;
use sm_core::config::ProfileName;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::journal::HookGate;
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    self, AmbientReply, CaptureReply, DaemonInfo, DaemonState, Envelope, Op, Reply, Request,
    Response, ShutdownParams,
};
use sm_core::time::Timestamp;

/// The build a mock daemon of another version reports.
pub const OTHER_BUILD: &str = "0.0.0-other-build";

/// The binary under test.
pub const BIN: &str = env!("CARGO_BIN_EXE_spatial-memory");

// ---------------------------------------------------------------------------
// The mock daemon
// ---------------------------------------------------------------------------

/// How the mock answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Answers at once.
    Answer,
    /// Answers after the delay.
    Late(Duration),
    /// Reads the request and never answers; the connection is held open
    /// until the mock stops.
    Hang,
    /// `hello` says `loading`; every other op fails with `daemon_loading`.
    Loading,
    /// A daemon of another build: every reply carries [`OTHER_BUILD`], so
    /// the client sees `version_mismatch`. It obeys `shutdown`.
    VersionMismatch,
}

/// A custom answer, tried before the built-in ones.
pub type Responder = Arc<dyn Fn(&Request) -> Option<Result<Reply, CoreError>> + Send + Sync>;

struct Shared {
    endpoint: Endpoint,
    mode: Mutex<Mode>,
    ambient: Mutex<Result<AmbientReply, CoreError>>,
    responder: Mutex<Option<Responder>>,
    requests: Mutex<Vec<Request>>,
    connections: AtomicUsize,
    stop: AtomicBool,
    closed: AtomicBool,
    held: Mutex<Vec<Stream>>,
    lock: Mutex<Option<File>>,
}

/// A daemon stand-in listening on a profile's endpoint. It records every
/// request and every accepted connection (including probes that send
/// nothing). Dropping it stops it.
pub struct MockDaemon {
    pub paths: ProfilePaths,
    shared: Arc<Shared>,
    accept: Mutex<Option<JoinHandle<()>>>,
}

/// An `ambient_recall` answer with nothing to print.
pub fn empty_ambient() -> AmbientReply {
    ambient_with(HookGate::Empty, None)
}

/// An `ambient_recall` answer with this gate and text.
pub fn ambient_with(gate: HookGate, text: Option<&str>) -> AmbientReply {
    AmbientReply {
        gate,
        text: text.map(str::to_owned),
        injected: Vec::new(),
        top_scores: Vec::new(),
        project: None,
    }
}

/// The `hello` answer of a mock.
pub fn daemon_info(request: &Request, state: DaemonState, build: &str) -> DaemonInfo {
    DaemonInfo {
        protocol: protocol::PROTOCOL_VERSION,
        build: build.to_owned(),
        commit: None,
        pid: std::process::id(),
        profile: request.profile.as_str().to_owned(),
        state,
        state_error: None,
        started_at: Timestamp::now(),
        ready_seconds: None,
        model_id: "mock-model".to_owned(),
        endpoint: "mock".to_owned(),
    }
}

fn bind(endpoint: &Endpoint) -> std::io::Result<interprocess::local_socket::Listener> {
    let name = sm_cli::client::endpoint_name(endpoint)?;
    ListenerOptions::new().name(name).create_sync()
}

/// What the real daemon does first (CONTRACTS 3.1): open `daemon.lock`
/// and retry `try_lock` every 25 ms for up to 2 s. `None` means another
/// daemon holds it, and this one exits.
pub fn take_daemon_lock(paths: &ProfilePaths) -> Option<File> {
    std::fs::create_dir_all(&paths.dir).ok()?;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(paths.lock_file())
        .ok()?;
    let until = Instant::now() + Duration::from_secs(2);
    loop {
        match file.try_lock() {
            Ok(()) => return Some(file),
            Err(TryLockError::WouldBlock) if Instant::now() < until => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return None,
        }
    }
}

/// Whether some process holds the profile's lock now.
pub fn lock_is_held(paths: &ProfilePaths) -> bool {
    let Ok(file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(paths.lock_file())
    else {
        return false;
    };
    matches!(file.try_lock(), Err(TryLockError::WouldBlock))
}

impl MockDaemon {
    /// Binds the profile's endpoint now, without the lock.
    pub fn start(paths: &ProfilePaths, mode: Mode) -> MockDaemon {
        Self::start_inner(paths, mode, None).expect("the mock daemon binds its endpoint")
    }

    /// Starts like the real daemon: the lock first (retrying for 2 s), then
    /// the endpoint; the lock is held until the endpoint is closed. `None`
    /// is the daemon that found the lock held and exited.
    pub fn start_like_daemon(paths: &ProfilePaths, mode: Mode) -> Option<MockDaemon> {
        let lock = take_daemon_lock(paths)?;
        Self::start_inner(paths, mode, Some(lock)).ok()
    }

    fn start_inner(
        paths: &ProfilePaths,
        mode: Mode,
        lock: Option<File>,
    ) -> std::io::Result<MockDaemon> {
        std::fs::create_dir_all(&paths.dir)?;
        let endpoint = paths.endpoint();
        let listener = bind(&endpoint)?;
        let shared = Arc::new(Shared {
            endpoint,
            mode: Mutex::new(mode),
            ambient: Mutex::new(Ok(empty_ambient())),
            responder: Mutex::new(None),
            requests: Mutex::new(Vec::new()),
            connections: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            held: Mutex::new(Vec::new()),
            lock: Mutex::new(lock),
        });
        let s = Arc::clone(&shared);
        let accept = std::thread::Builder::new()
            .name("mock-daemon-accept".into())
            .spawn(move || {
                loop {
                    let conn = listener.accept();
                    if s.stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = conn else { continue };
                    s.connections.fetch_add(1, Ordering::SeqCst);
                    let s2 = Arc::clone(&s);
                    let _ = std::thread::Builder::new()
                        .name("mock-daemon-conn".into())
                        .spawn(move || handle(stream, &s2));
                }
                // CONTRACTS 3.1 shutdown order: stop accepting, close the
                // endpoint, release the lock last.
                drop(listener);
                s.held.lock().unwrap().clear();
                s.closed.store(true, Ordering::SeqCst);
                s.lock.lock().unwrap().take();
            })?;
        Ok(MockDaemon {
            paths: paths.clone(),
            shared,
            accept: Mutex::new(Some(accept)),
        })
    }

    pub fn set_mode(&self, mode: Mode) {
        *self.shared.mode.lock().unwrap() = mode;
    }

    /// The `ambient_recall` answer (an error is sent as the daemon's error).
    pub fn set_ambient(&self, reply: Result<AmbientReply, CoreError>) {
        *self.shared.ambient.lock().unwrap() = reply;
    }

    pub fn set_responder(&self, responder: Responder) {
        *self.shared.responder.lock().unwrap() = Some(responder);
    }

    /// Every request received, in arrival order.
    pub fn requests(&self) -> Vec<Request> {
        self.shared.requests.lock().unwrap().clone()
    }

    /// Every connection accepted, including ones that sent nothing.
    pub fn connections(&self) -> usize {
        self.shared.connections.load(Ordering::SeqCst)
    }

    /// Whether the endpoint has been closed.
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }

    /// Waits for the endpoint to close.
    pub fn wait_closed(&self, timeout: Duration) -> bool {
        let until = Instant::now() + timeout;
        while !self.is_closed() {
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// Whether this mock holds the profile lock.
    pub fn holds_lock(&self) -> bool {
        self.shared.lock.lock().unwrap().is_some()
    }

    /// Stops accepting, closes the endpoint and releases the lock.
    pub fn stop(&self) {
        begin_stop(&self.shared);
        let handle = self.accept.lock().unwrap().take();
        if let Some(handle) = handle {
            let until = Instant::now() + Duration::from_secs(5);
            while !handle.is_finished() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for MockDaemon {
    fn drop(&mut self) {
        self.stop();
    }
}

fn begin_stop(shared: &Shared) {
    if shared.stop.swap(true, Ordering::SeqCst) {
        return;
    }
    // Wake the blocking accept with a connection of our own.
    if let Ok(name) = sm_cli::client::endpoint_name(&shared.endpoint) {
        let _ = ConnectOptions::new()
            .name(name)
            .wait_mode(ConnectWaitMode::Timeout(Duration::from_secs(2)))
            .connect_sync();
    }
}

fn handle(stream: Stream, shared: &Arc<Shared>) {
    let Ok(Some(body)) = protocol::read_frame(&mut &stream) else {
        // A probe that connected and closed, or garbage.
        return;
    };
    let Ok(request) = protocol::decode_body::<Request>(&body) else {
        return;
    };
    shared.requests.lock().unwrap().push(request.clone());
    let mode = *shared.mode.lock().unwrap();
    match mode {
        Mode::Hang => {
            shared.held.lock().unwrap().push(stream);
            return;
        }
        Mode::Late(delay) => std::thread::sleep(delay),
        _ => {}
    }
    let response = respond(shared, &request, mode, &body);
    let _ = protocol::write_frame(&mut &stream, &response);
    let _ = (&stream).flush();
    let shutdown = matches!(request.call, Op::Shutdown(_));
    drop(stream);
    if shutdown {
        begin_stop(shared);
    }
}

fn respond(shared: &Shared, request: &Request, mode: Mode, body: &[u8]) -> Response {
    let mismatch = mode == Mode::VersionMismatch;
    let mut response = if mismatch && !matches!(request.call, Op::Shutdown(_)) {
        // What a daemon of another build answers to anything but shutdown.
        let envelope = Envelope::peek(body).expect("a request has an envelope");
        Response::version_mismatch(&envelope)
    } else {
        let custom = shared
            .responder
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|f| f(request));
        let result = custom.unwrap_or_else(|| default_reply(shared, request, mode));
        match result {
            Ok(reply) => Response::ok(request.id.clone(), reply),
            Err(e) => Response::err(request.id.clone(), e),
        }
    };
    if mismatch {
        OTHER_BUILD.clone_into(&mut response.build);
    }
    response
}

/// The `daemon_loading` error the mock sends while "loading".
pub fn loading_error() -> CoreError {
    CoreError::new(ErrorCode::DaemonLoading, "the embedding model is loading")
        .with_next_action("Retry in a few seconds.")
}

fn default_reply(shared: &Shared, request: &Request, mode: Mode) -> Result<Reply, CoreError> {
    match &request.call {
        Op::Hello => {
            let state = if mode == Mode::Loading {
                DaemonState::Loading
            } else {
                DaemonState::Ready
            };
            Ok(Reply::Hello(daemon_info(request, state, sm_core::BUILD)))
        }
        Op::Shutdown(_) => Ok(Reply::ShuttingDown),
        _ if mode == Mode::Loading => Err(loading_error()),
        Op::AmbientRecall(_) => shared.ambient.lock().unwrap().clone().map(Reply::Ambient),
        Op::Capture(_) => Ok(Reply::Captured(CaptureReply { accepted: true })),
        other => Err(CoreError::internal(format!(
            "the mock daemon has no answer for {}",
            other.name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Spawners
// ---------------------------------------------------------------------------

/// A spawner that only counts.
#[derive(Default)]
pub struct CountingSpawn {
    pub count: AtomicUsize,
}

impl Spawn for CountingSpawn {
    fn spawn_daemon(&self, _profile: &ProfileName) -> std::io::Result<()> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl CountingSpawn {
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

/// A spawner that starts a lock-taking mock daemon on a thread, as the real
/// one would start in its own process. The mocks that came up are kept.
pub struct MockSpawn {
    pub paths: ProfilePaths,
    pub mode: Mode,
    pub ambient: Result<AmbientReply, CoreError>,
    pub spawned: AtomicUsize,
    /// Mocks that found the lock held and exited, as a second daemon does.
    pub exited: Arc<AtomicUsize>,
    pub started: Arc<Mutex<Vec<MockDaemon>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl MockSpawn {
    pub fn new(paths: &ProfilePaths, mode: Mode, ambient: Result<AmbientReply, CoreError>) -> Self {
        MockSpawn {
            paths: paths.clone(),
            mode,
            ambient,
            spawned: AtomicUsize::new(0),
            exited: Arc::new(AtomicUsize::new(0)),
            started: Arc::new(Mutex::new(Vec::new())),
            threads: Mutex::new(Vec::new()),
        }
    }

    pub fn spawned(&self) -> usize {
        self.spawned.load(Ordering::SeqCst)
    }

    pub fn exited(&self) -> usize {
        self.exited.load(Ordering::SeqCst)
    }

    /// Waits until every spawned mock has either bound or exited.
    pub fn settle(&self) {
        let threads: Vec<_> = self.threads.lock().unwrap().drain(..).collect();
        for t in threads {
            let _ = t.join();
        }
    }

    /// How many mocks bound and run now.
    pub fn running(&self) -> usize {
        self.started
            .lock()
            .unwrap()
            .iter()
            .filter(|m| !m.is_closed())
            .count()
    }
}

impl Spawn for MockSpawn {
    fn spawn_daemon(&self, _profile: &ProfileName) -> std::io::Result<()> {
        self.spawned.fetch_add(1, Ordering::SeqCst);
        let paths = self.paths.clone();
        let mode = self.mode;
        let ambient = self.ambient.clone();
        let started = Arc::clone(&self.started);
        let exited = Arc::clone(&self.exited);
        let t = std::thread::spawn(move || {
            if let Some(mock) = MockDaemon::start_like_daemon(&paths, mode) {
                mock.set_ambient(ambient);
                started.lock().unwrap().push(mock);
            } else {
                exited.fetch_add(1, Ordering::SeqCst);
            }
        });
        self.threads.lock().unwrap().push(t);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The isolated binary runner
// ---------------------------------------------------------------------------

/// What one run of the binary produced.
#[derive(Debug)]
pub struct Run {
    pub stdout: String,
    pub stderr: String,
    pub code: Option<i32>,
    /// From just before spawn to exit with both pipes closed.
    pub elapsed: Duration,
}

/// A private home, user home and `{TMP}` folder.
pub struct Sandbox {
    /// Taken in `drop`: deleted, or kept with only the store block in it.
    dir: Option<tempfile::TempDir>,
    /// `SPATIAL_MEMORY_HOME`.
    pub home: PathBuf,
    /// `HOME`, `USERPROFILE` and the platform data folders.
    pub user_home: PathBuf,
    /// The fixtures' `{TMP}`.
    pub tmp: PathBuf,
    /// `SPATIAL_MEMORY_MODEL_URL`: this sandbox's folder on the local 404
    /// server.
    pub model_url: String,
    model_prefix: String,
    /// Whether a real daemon can start here.
    daemon_allowed: bool,
}

/// The model address every sandboxed run gets: an HTTP server on 127.0.0.1
/// that reads each request and answers 404, recording the path asked for.
/// Nothing leaves the machine.
struct NoModelServer {
    base: String,
    paths: Mutex<Vec<String>>,
}

fn no_model_server() -> &'static NoModelServer {
    static SERVER: OnceLock<&'static NoModelServer> = OnceLock::new();
    SERVER.get_or_init(|| {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the local model server");
        let server: &'static NoModelServer = Box::leak(Box::new(NoModelServer {
            base: format!("http://{}", listener.local_addr().unwrap()),
            paths: Mutex::new(Vec::new()),
        }));
        std::thread::Builder::new()
            .name("no-model-server".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() {
                        continue;
                    }
                    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                    loop {
                        let mut h = String::new();
                        if reader.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
                            break;
                        }
                    }
                    server.paths.lock().unwrap().push(path);
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    let _ = stream.flush();
                }
            })
            .expect("start the local model server");
        server
    })
}

/// Windows: the test process's own standard handles (cargo's pipes) are
/// inheritable, and `Command::spawn` passes every inheritable handle to the
/// binary, which would pass them on to a daemon it spawns. A daemon holding
/// cargo's pipe keeps `cargo test` waiting for it. Cleared once per test
/// binary; `Stdio::inherit` still works, as `client::disinherit` explains.
fn disinherit_own_std_handles() {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            sm_cli::client::disinherit(&[
                std::io::stdin().as_raw_handle(),
                std::io::stdout().as_raw_handle(),
                std::io::stderr().as_raw_handle(),
            ]);
        });
    }
}

/// The folder every sandbox is made in, under the system's temporary
/// folder, so what a sandbox leaves behind is easy to tell from anything
/// else there.
fn sandbox_parent() -> PathBuf {
    std::env::temp_dir().join("spatial-memory-cli-tests")
}

/// How old a leftover sandbox must be before a later test binary deletes
/// it. No test lives this long.
const LEFTOVER_AGE: Duration = Duration::from_secs(60 * 60);

/// Once per test binary: deletes sandboxes kept by earlier runs (see
/// [`Sandbox`]'s `drop`) that are older than [`LEFTOVER_AGE`]. Errors are
/// ignored; a folder still in use is tried again next time.
fn sweep_leftovers() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Ok(entries) = std::fs::read_dir(sandbox_parent()) else {
            return;
        };
        for entry in entries.flatten() {
            let old = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > LEFTOVER_AGE);
            let ours = entry.file_name().to_string_lossy().starts_with("sb-");
            if old && ours {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    });
}

/// Removes everything under `root` except `keep` and the folders above it.
/// Errors are ignored.
fn remove_all_but(root: &Path, keep: &Path) {
    let Ok(rel) = keep.strip_prefix(root) else {
        return;
    };
    let mut dir = root.to_path_buf();
    for part in rel.components() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            if entry.file_name() == part.as_os_str() {
                continue;
            }
            let path = entry.path();
            let _ = if entry.file_type().is_ok_and(|t| t.is_dir()) {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
        }
        dir.push(part);
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let paths = self.paths();
        if self.daemon_allowed {
            shut_down_daemon(&paths, "test sandbox dropped");
        }
        // A daemon a hook spawned may still be on its way out; on Windows
        // its folder cannot be deleted while it runs.
        let start_tried = paths.lock_file().exists();
        if start_tried {
            wait_for_lock_free(&paths, Duration::from_secs(10));
        }
        let Some(dir) = self.dir.take() else {
            return;
        };
        if self.daemon_allowed || !start_tried {
            return; // the TempDir deletes the folder
        }
        // A daemon spawned here may not have started yet: the lock is free
        // in the moment between the client dropping it and the daemon
        // taking it. Deleting the whole folder would let that daemon make a
        // fresh profile, open a real store and run until killed. So the
        // store block stays, and a late daemon still exits 1; the folder is
        // swept by a later test binary.
        remove_all_but(dir.path(), &paths.database());
        let _ = dir.keep();
    }
}

/// Sends `shutdown` to whatever listens on the profile's endpoint and waits
/// for the lock to come free. True when nothing holds it afterwards.
pub fn shut_down_daemon(paths: &ProfilePaths, reason: &str) -> bool {
    let endpoint = paths.endpoint();
    if sm_cli::client::is_listening(&endpoint) {
        let request = Request::new(
            paths.dir.to_string_lossy().into_owned(),
            paths.profile.clone(),
            Op::Shutdown(ShutdownParams {
                reason: reason.to_owned(),
            }),
        );
        let _ = sm_cli::client::exchange(&endpoint, &request, Some(Duration::from_secs(5)));
    }
    !paths.lock_file().exists() || wait_for_lock_free(paths, Duration::from_secs(10))
}

/// Waits until no process holds the profile's lock.
pub fn wait_for_lock_free(paths: &ProfilePaths, timeout: Duration) -> bool {
    let until = Instant::now() + timeout;
    while lock_is_held(paths) {
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

/// Variables passed through from the test process: what a process needs to
/// start on each platform, nothing that names user data.
const PASS_THROUGH: &[&str] = &[
    "PATH",
    "SystemRoot",
    "SYSTEMROOT",
    "windir",
    "ComSpec",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
];

impl Default for Sandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox {
    /// A sandbox where no daemon can start: a daemon the binary spawns here
    /// exits 1 at once (the store cannot be opened), so "no daemon" means
    /// no daemon answers.
    pub fn new() -> Self {
        let sb = Self::make(false);
        // SQLite cannot open a folder as its database.
        std::fs::create_dir_all(sb.paths().database()).unwrap();
        sb
    }

    /// A sandbox where the binary's real daemon can start. Dropping it
    /// shuts the daemon down and waits for its lock.
    pub fn with_daemon() -> Self {
        Self::make(true)
    }

    fn make(daemon_allowed: bool) -> Self {
        disinherit_own_std_handles();
        sweep_leftovers();
        let parent = sandbox_parent();
        std::fs::create_dir_all(&parent).expect("the sandbox parent folder");
        let dir = tempfile::Builder::new()
            .prefix("sb-")
            .tempdir_in(&parent)
            .expect("temporary folder");
        let home = dir.path().join("h");
        let user_home = dir.path().join("u");
        let tmp = dir.path().join("t");
        for d in [&home, &user_home, &tmp] {
            std::fs::create_dir_all(d).unwrap();
        }
        let name = dir
            .path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let model_prefix = format!("/{}/", name.trim_start_matches('.'));
        let model_url = format!(
            "{}{}",
            no_model_server().base,
            model_prefix.trim_end_matches('/')
        );
        Sandbox {
            dir: Some(dir),
            home,
            user_home,
            tmp,
            model_url,
            model_prefix,
            daemon_allowed,
        }
    }

    /// The paths asked of the local model server by runs in this sandbox.
    pub fn model_requests(&self) -> Vec<String> {
        no_model_server()
            .paths
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.starts_with(&self.model_prefix))
            .cloned()
            .collect()
    }

    /// The default profile's paths under this home.
    pub fn paths(&self) -> ProfilePaths {
        ProfilePaths::new(self.home.clone(), ProfileName::default_profile())
    }

    /// `{TMP}` as the fixtures write it: absolute, forward slashes.
    pub fn tmp_token(&self) -> String {
        self.tmp.to_string_lossy().replace('\\', "/")
    }

    /// The binary with a cleared, private environment.
    pub fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new(BIN);
        cmd.args(args).env_clear();
        for key in PASS_THROUGH {
            if let Some(v) = std::env::var_os(key) {
                cmd.env(key, v);
            }
        }
        let u = &self.user_home;
        cmd.env("HOME", u)
            .env("USERPROFILE", u)
            .env("APPDATA", u.join("AppData").join("Roaming"))
            .env("LOCALAPPDATA", u.join("AppData").join("Local"))
            .env("XDG_CONFIG_HOME", u.join(".config"))
            .env("XDG_DATA_HOME", u.join(".local").join("share"))
            .env("XDG_CACHE_HOME", u.join(".cache"))
            .env(sm_core::paths::MODEL_DIR_ENV, u.join("no-model"))
            .env(sm_cli::model_step::MODEL_URL_ENV, &self.model_url)
            .env(sm_core::paths::HOME_ENV, &self.home)
            .current_dir(&self.tmp);
        cmd
    }

    /// Runs the binary with extra variables and this stdin. The home
    /// variable always points at the sandbox, whatever `env` says.
    pub fn run(&self, args: &[&str], env: &[(String, String)], stdin: &[u8]) -> Run {
        let mut cmd = self.command(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.env(sm_core::paths::HOME_ENV, &self.home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let input = stdin.to_vec();
        let start = Instant::now();
        let mut child = cmd.spawn().expect("spawn spatial-memory");
        let mut pipe = child.stdin.take().expect("stdin pipe");
        // The hook may stop reading early (oversized input); a broken pipe
        // is expected then.
        let writer = std::thread::spawn(move || {
            let _ = pipe.write_all(&input);
        });
        let out = child.wait_with_output().expect("wait for spatial-memory");
        let elapsed = start.elapsed();
        let _ = writer.join();
        Run {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            code: out.status.code(),
            elapsed,
        }
    }

    /// `spatial-memory hook <args>`.
    pub fn hook(&self, args: &[&str], env: &[(String, String)], stdin: &[u8]) -> Run {
        let mut all = vec!["hook"];
        all.extend_from_slice(args);
        self.run(&all, env, stdin)
    }

    /// Spool files waiting in `spool/new/`.
    pub fn spooled(&self) -> Vec<PathBuf> {
        let dir = self.paths().spool_new();
        if !dir.exists() {
            return Vec::new();
        }
        sm_core::spool::list_spool(&dir).expect("list the spool")
    }

    /// Whether the profile lock file was ever created.
    pub fn lock_file_exists(&self) -> bool {
        self.paths().lock_file().exists()
    }

    /// The `gate` label of every `hook_gate_total` journal line, sorted
    /// (each hook process writes its own file, so file order is not time
    /// order).
    pub fn gates(&self) -> Vec<String> {
        let dir = self.paths().metrics_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut gates = Vec::new();
        for entry in entries.filter_map(Result::ok) {
            let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
            for line in text.lines() {
                let e: sm_core::journal::Entry = serde_json::from_str(line).expect("journal line");
                if e.metric == sm_core::journal::metrics::HOOK_GATE_TOTAL {
                    gates.push(e.labels["gate"].as_str().unwrap_or_default().to_owned());
                }
            }
        }
        gates.sort();
        gates
    }
}

/// Environment pairs from string slices.
pub fn env_of(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

/// The median and 95th percentile (nearest rank) and the worst of a set of
/// times, in milliseconds.
pub fn stats_ms(times: &[Duration]) -> (f64, f64, f64) {
    let mut ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let n = ms.len();
    let median = if n % 2 == 1 {
        ms[n / 2]
    } else {
        (ms[n / 2 - 1] + ms[n / 2]) / 2.0
    };
    let rank = ((0.95 * n as f64).ceil() as usize).clamp(1, n);
    (median, ms[rank - 1], ms[n - 1])
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// `fixtures/capture`.
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/capture"
    ))
}

pub fn load_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// `event_aliases.json`.
pub fn event_aliases() -> Value {
    load_json(&fixture_dir().join("event_aliases.json"))
}

/// One payload of `hook_payloads/` by its file stem.
pub fn payload(name: &str) -> Value {
    load_json(
        &fixture_dir()
            .join("hook_payloads")
            .join(format!("{name}.json")),
    )
}

/// Every payload in `hook_payloads/`, by file name.
pub fn hook_payloads() -> Vec<Value> {
    let dir = fixture_dir().join("hook_payloads");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && p.file_name().is_some_and(|n| n != "INDEX.json")
        })
        .collect();
    names.sort();
    names.iter().map(|p| load_json(p)).collect()
}

/// Replaces `{TMP}` in every string.
pub fn expand_tmp(value: &Value, tmp: &str) -> Value {
    match value {
        Value::String(s) => Value::String(s.replace("{TMP}", tmp)),
        Value::Array(a) => Value::Array(a.iter().map(|v| expand_tmp(v, tmp)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), expand_tmp(v, tmp)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `stdin_text`: a string, or `{"$concat": [...]}` whose parts are strings
/// or `{"$repeat": s, "count": n}`.
pub fn build_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(m) if m.contains_key("$concat") => m["$concat"]
            .as_array()
            .expect("$concat is a list")
            .iter()
            .map(build_text)
            .collect(),
        Value::Object(m) if m.contains_key("$repeat") => {
            let unit = m["$repeat"].as_str().expect("$repeat is a string");
            let count = m["count"].as_u64().expect("count") as usize;
            unit.repeat(count)
        }
        other => panic!("unexpected stdin_text form: {other}"),
    }
}

/// Lower- or upper-case hex to bytes.
pub fn hex_decode(s: &str) -> Vec<u8> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// A case's stdin bytes, as the fixture generator fed them.
pub fn stdin_bytes(case: &Value, tmp: &str) -> Vec<u8> {
    if let Some(v) = case.get("stdin_json") {
        serde_json::to_vec(&expand_tmp(v, tmp)).unwrap()
    } else if let Some(v) = case.get("stdin_text") {
        build_text(&expand_tmp(v, tmp)).into_bytes()
    } else if let Some(v) = case.get("stdin_hex") {
        hex_decode(v.as_str().expect("stdin_hex is a string"))
    } else {
        Vec::new()
    }
}

/// A case's environment with `{TMP}` expanded.
pub fn case_env(case: &Value, tmp: &str) -> Vec<(String, String)> {
    case.get("env")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str().unwrap_or_default().replace("{TMP}", tmp),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Writes a case's `files`.
pub fn create_files(case: &Value, tmp: &str) {
    for f in case
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let path = PathBuf::from(f["path"].as_str().unwrap().replace("{TMP}", tmp));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, f["text"].as_str().unwrap_or_default()).unwrap();
    }
}

/// A case's argv as `&str`.
pub fn case_argv(case: &Value) -> Vec<String> {
    case["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

/// Reads all of a reader (for tests that capture output in-process).
pub fn read_all(mut r: impl Read) -> String {
    let mut s = String::new();
    let _ = r.read_to_string(&mut s);
    s
}
