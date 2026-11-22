//! The daemon client (CONTRACTS sections 3, 3.1 and 4).
//!
//! One request per connection: connect, write one frame, read one frame,
//! peek the envelope's version, decode. A request with a deadline runs on
//! its own thread and the caller waits on a channel, so the deadline holds
//! even when the daemon accepts and never answers; the abandoned thread ends
//! with the process.
//!
//! Auto-start follows CONTRACTS 3.1 exactly: one `try_lock` on the profile's
//! `daemon.lock`; if it is acquired, spawn `<this binary> daemon --profile
//! <name>` detached and drop the lock at once. The spawn is a parameter
//! ([`Spawn`]) so tests can count spawns and start a mock daemon instead.
//!
//! A profile held stopped for an upgrade (B-180: `daemon.hold`, written by
//! [`hold`], removed by [`release`]) starts no daemon: [`try_start`] spawns
//! nothing and a waiting client fails at once with the hold's next action.
//!
//! Every connection goes through [`sm_daemon::connect`], the daemon crate's
//! owner-checked connect (B-95, B-112): the hook, the MCP proxy and the
//! command line all refuse an endpoint the daemon would refuse. The
//! deadlines stay this module's (B-80).

use std::fs::{OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{GenericFilePath, GenericNamespaced, Name};
use sm_core::config::ProfileName;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    Envelope, Op, Outcome, Reply, Request, Response, ShutdownParams, decode_body, encode_frame,
    read_frame,
};
use sm_core::time::Timestamp;

/// Interval between connection attempts while waiting for a daemon
/// (CONTRACTS 3.1: every 50 ms).
pub const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long the MCP proxy and the command line wait for a daemon to answer
/// `hello` (CONTRACTS 3.1: up to 10 s).
pub const CONNECT_WINDOW: Duration = Duration::from_secs(10);
/// The bound on one `hello` or `shutdown` exchange inside the window.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
/// How long a connect may wait for a busy pipe instance when the request
/// itself has no deadline.
const CONNECT_WAIT: Duration = Duration::from_secs(2);

/// `ShutdownParams::reason` sent to a daemon of another build (CONTRACTS 4).
pub const REASON_VERSION_MISMATCH: &str = "version_mismatch";

// ---------------------------------------------------------------------------
// Spawning the daemon
// ---------------------------------------------------------------------------

/// Starts a daemon for a profile. The real implementation runs this binary
/// detached; tests substitute a counter or an in-process mock.
pub trait Spawn: Send + Sync {
    fn spawn_daemon(&self, profile: &ProfileName) -> io::Result<()>;
}

/// The Windows creation flags of the daemon process: detached from the
/// console, in its own process group, with no window. Never
/// `CREATE_BREAKAWAY_FROM_JOB` (CONTRACTS 3.1).
#[cfg(windows)]
pub const WINDOWS_CREATION_FLAGS: u32 = windows_sys::Win32::System::Threading::DETACHED_PROCESS
    | windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP
    | windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

/// Spawns `<exe> daemon --profile <name>` with null standard streams,
/// detached from the caller's console or session.
#[derive(Debug, Clone)]
pub struct DetachedSpawn {
    pub exe: PathBuf,
}

impl DetachedSpawn {
    /// This binary.
    pub fn current() -> io::Result<Self> {
        Ok(DetachedSpawn {
            exe: std::env::current_exe()?,
        })
    }
}

/// The daemon command line, ready to spawn. On Windows [`DetachedSpawn`]
/// takes only its program and arguments and starts it with
/// [`spawn_detached`]; the null streams and flags here are what `Command`
/// itself would use.
pub fn daemon_command(exe: &std::path::Path, profile: &ProfileName) -> Command {
    let mut cmd = Command::new(exe);
    cmd.arg("daemon")
        .arg("--profile")
        .arg(profile.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(WINDOWS_CREATION_FLAGS);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: `setsid` is async-signal-safe and touches no memory of the
        // parent; it is the only call made between fork and exec.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    cmd
}

/// Windows: clears `HANDLE_FLAG_INHERIT` on each handle. `Command::spawn`
/// passes `bInheritHandles = TRUE`, so a child inherits every inheritable
/// handle of its parent, including handles the parent itself inherited.
/// The daemon spawn does not rely on this: [`spawn_detached`] passes
/// `bInheritHandles = FALSE`. The test harness uses it on its own standard
/// handles. Errors are ignored: a handle that cannot be changed is a
/// console or is absent. `Stdio::inherit` in a later spawn is unaffected,
/// because std duplicates the handle as inheritable for that child.
#[cfg(windows)]
pub fn disinherit(handles: &[std::os::windows::io::RawHandle]) {
    use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
    for &handle in handles {
        if handle.is_null() {
            continue;
        }
        // SAFETY: `SetHandleInformation` only changes the flags of a handle
        // value this process owns; an invalid value makes it fail, which is
        // ignored, and no memory is read or written.
        unsafe {
            SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
        }
    }
}

/// Windows: starts `cmd` (program and arguments only) with `CreateProcessW`,
/// `bInheritHandles = FALSE` and [`WINDOWS_CREATION_FLAGS`]. The daemon
/// inherits no handle at all: not the hook's or the proxy's stdin and
/// stdout, and not a handle that process inherited from its own parent.
/// Clearing the inherit flag on the caller's standard handles is not
/// enough, because a process can hold further inheritable copies of the
/// same pipe; the client then sees no end of file until the daemon exits,
/// which is hours. The child's standard handles are null, which Rust's
/// standard streams treat as a sink. The environment and the working folder
/// are the caller's, as with `Command`. An argument that would need quoting
/// is refused: the daemon's are `daemon`, `--profile` and a profile name
/// of `a-z`, `0-9` and `-`.
#[cfg(windows)]
pub fn spawn_detached(cmd: &Command) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        CreateProcessW, PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOW,
    };
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidInput, what.to_owned());
    let program: Vec<u16> = cmd.get_program().encode_wide().collect();
    if program.is_empty() || program.contains(&0) || program.contains(&u16::from(b'"')) {
        return Err(invalid("the daemon's program path cannot be quoted"));
    }
    let mut line: Vec<u16> = Vec::with_capacity(program.len() + 64);
    line.push(u16::from(b'"'));
    line.extend_from_slice(&program);
    line.push(u16::from(b'"'));
    for arg in cmd.get_args() {
        let arg: Vec<u16> = arg.encode_wide().collect();
        let plain = !arg.is_empty()
            && arg.iter().all(|&c| {
                c != 0 && c != u16::from(b' ') && c != u16::from(b'\t') && c != u16::from(b'"')
            });
        if !plain {
            return Err(invalid("a daemon argument would need quoting"));
        }
        line.push(u16::from(b' '));
        line.extend_from_slice(&arg);
    }
    line.push(0);
    // An absolute path is run as it is; anything else is searched for the
    // way `Command` would (`CreateProcessW` parses the quoted first token).
    let app: Option<Vec<u16>> = std::path::Path::new(cmd.get_program())
        .is_absolute()
        .then(|| program.iter().copied().chain(std::iter::once(0)).collect());

    // SAFETY: both are plain C structs for which all-zero is a valid value
    // (null handles, null pointers, zero sizes and flags).
    let mut startup: STARTUPINFOW = unsafe { std::mem::zeroed() };
    startup.cb = u32::try_from(std::mem::size_of::<STARTUPINFOW>()).unwrap_or(u32::MAX);
    // Null standard handles, stated rather than left to the console rules.
    startup.dwFlags = STARTF_USESTDHANDLES;
    // SAFETY: as above.
    let mut info: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `app` (when present) and `line` are NUL-terminated UTF-16
    // buffers that outlive the call, and `line` is mutable as the API
    // requires; the security attributes, environment and folder are null,
    // which means the defaults and the caller's own; `startup` and `info`
    // are valid for reads and writes. No handle is inherited.
    let ok = unsafe {
        CreateProcessW(
            app.as_ref().map_or(std::ptr::null(), |a| a.as_ptr()),
            line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            WINDOWS_CREATION_FLAGS,
            std::ptr::null(),
            std::ptr::null(),
            &startup,
            &mut info,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // The daemon outlives its starter; there is nothing to reap on Windows.
    // SAFETY: both handles were just returned by `CreateProcessW` and are
    // closed once each.
    unsafe {
        CloseHandle(info.hThread);
        CloseHandle(info.hProcess);
    }
    Ok(())
}

impl Spawn for DetachedSpawn {
    #[cfg(windows)]
    fn spawn_daemon(&self, profile: &ProfileName) -> io::Result<()> {
        spawn_detached(&daemon_command(&self.exe, profile))
    }

    #[cfg(not(windows))]
    fn spawn_daemon(&self, profile: &ProfileName) -> io::Result<()> {
        let mut child = daemon_command(&self.exe, profile).spawn()?;
        // A long-lived parent (the MCP proxy) reaps the daemon when it exits;
        // a short-lived one (the hook) exits first and the thread goes with
        // it.
        let _ = std::thread::Builder::new()
            .name("sm-daemon-reaper".into())
            .spawn(move || {
                let _ = child.wait();
            });
        Ok(())
    }
}

/// Creates the profile folder if it is missing (B-117). On Unix it is
/// user-only: the folder itself is made with mode 0700 and its missing
/// parents with the default mode, as the daemon's start makes it, so a
/// client never leaves a folder the daemon then has to fix. An existing
/// folder is left as it is: the daemon's start sets one of this user to
/// 0700, with a log line, and refuses one another user owns. Windows is
/// unchanged: the folder inherits the access list of `%LOCALAPPDATA%`.
pub fn create_profile_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        // `DirBuilder::recursive` would give every folder it creates the
        // mode, parents included, so the parents are made apart.
        if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        match std::fs::DirBuilder::new().mode(0o700).create(dir) {
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && dir.is_dir() => Ok(()),
            made => made,
        }
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// CONTRACTS 3.1: one `try_lock` on the profile's lock file, no retry. When
/// it is acquired, spawn the daemon and drop the lock at once (the daemon
/// takes it for its lifetime). Returns whether a daemon was spawned. A held
/// lock means another client is starting one, or one is running. A profile
/// held stopped (B-180) spawns nothing and creates nothing.
pub fn try_start(paths: &ProfilePaths, spawner: &dyn Spawn) -> io::Result<bool> {
    if sm_daemon::read_hold(paths).is_some() {
        return Ok(false);
    }
    create_profile_dir(&paths.dir)?;
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(paths.lock_file())?;
    match lock.try_lock() {
        Ok(()) => {
            let spawned = spawner.spawn_daemon(&paths.profile);
            drop(lock);
            spawned.map(|()| true)
        }
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// One exchange
// ---------------------------------------------------------------------------

/// Why an exchange produced no usable response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExchangeError {
    /// Nothing is listening at the endpoint; the request was not sent.
    Absent,
    /// No complete reply before the deadline. `connected` says whether the
    /// connection was made (a hung daemon) or not (a busy one).
    Timeout { connected: bool },
    /// The daemon is another protocol or build: its reply's envelope
    /// differs, or it answered `version_mismatch` (CONTRACTS 4).
    Mismatch,
    /// Any other failure: the connection closed early, a bad frame.
    Io(String),
    /// [`sm_daemon::connect`] refused the endpoint before sending anything,
    /// for a reason other than reachability: on Unix, a socket another user
    /// owns. The error is the daemon crate's, with its next action.
    Refused(CoreError),
}

impl ExchangeError {
    /// The error a caller reports for this outcome.
    pub fn into_core(self, endpoint: &Endpoint) -> CoreError {
        let at = endpoint.display_path();
        match self {
            ExchangeError::Absent => CoreError::new(
                ErrorCode::DaemonUnavailable,
                format!("no daemon is listening at {at}"),
            ),
            ExchangeError::Timeout { connected: true } => CoreError::new(
                ErrorCode::Timeout,
                format!("the daemon at {at} did not answer in time"),
            ),
            ExchangeError::Timeout { connected: false } => CoreError::new(
                ErrorCode::DaemonUnavailable,
                format!("could not connect to the daemon at {at} in time"),
            ),
            ExchangeError::Mismatch => CoreError::new(
                ErrorCode::VersionMismatch,
                format!(
                    "the daemon at {at} is another build than this client ({})",
                    sm_core::BUILD
                ),
            ),
            ExchangeError::Io(e) => CoreError::new(
                ErrorCode::DaemonUnavailable,
                format!("the connection to the daemon at {at} failed: {e}"),
            ),
            ExchangeError::Refused(e) => e,
        }
    }
}

/// The interprocess name of an endpoint.
pub fn endpoint_name(endpoint: &Endpoint) -> io::Result<Name<'static>> {
    match endpoint {
        Endpoint::NamedPipe(n) => n
            .clone()
            .to_ns_name::<GenericNamespaced>()
            .map(Name::into_owned),
        Endpoint::UnixSocket(p) => p
            .clone()
            .to_fs_name::<GenericFilePath>()
            .map(Name::into_owned),
    }
}

/// Connects through the daemon crate's owner-checked [`sm_daemon::connect`]
/// (B-95, B-112), so every client refuses an endpoint the daemon would.
pub(crate) fn connect(
    endpoint: &Endpoint,
    wait: Duration,
) -> Result<interprocess::local_socket::Stream, ExchangeError> {
    sm_daemon::connect(endpoint, wait).map_err(|e| classify_connect_error(endpoint, e))
}

/// Sorts a failed [`sm_daemon::connect`] into an [`ExchangeError`].
///
/// `sm_daemon::connect` reports every failure as `daemon_unavailable`. A
/// failure to reach the endpoint reads `cannot reach <endpoint>: <io
/// error>`, and an operating-system error's text ends with `(os error N)`,
/// which gives the error kind back: not found or refused is
/// [`ExchangeError::Absent`] (nothing listens, so the request was not sent
/// and a daemon may be started); timed out is a busy endpoint.
///
/// A refusal that this user's daemon causes while it exits
/// ([`sm_daemon::refusal_may_pass`]: its socket removed, or its pipe server
/// gone, between connecting and the check) is [`ExchangeError::Io`], so
/// [`Client::ensure_daemon`] polls again and starts a daemon once nothing
/// listens (B-136). Any other failure, the owner checks above all, is
/// [`ExchangeError::Refused`]: the endpoint must not be used, so the client
/// neither starts a daemon nor retries, and reports the daemon crate's error
/// with its next action unchanged.
pub(crate) fn classify_connect_error(endpoint: &Endpoint, e: CoreError) -> ExchangeError {
    let prefix = format!("cannot reach {}: ", endpoint.display_path());
    let Some(io_text) = e.message.strip_prefix(&prefix) else {
        if sm_daemon::refusal_may_pass(&e) {
            return ExchangeError::Io(e.message);
        }
        return ExchangeError::Refused(e);
    };
    let kind = io_text
        .strip_suffix(')')
        .and_then(|t| t.rsplit_once("(os error "))
        .and_then(|(_, code)| code.parse::<i32>().ok())
        .map(|code| io::Error::from_raw_os_error(code).kind());
    match kind {
        Some(io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused) => ExchangeError::Absent,
        Some(io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock) => {
            ExchangeError::Timeout { connected: false }
        }
        Some(_) => ExchangeError::Io(io_text.to_owned()),
        None if io_text.contains("timed out") => ExchangeError::Timeout { connected: false },
        None => ExchangeError::Io(io_text.to_owned()),
    }
}

/// Whether anything accepts a connection at the endpoint now.
pub fn is_listening(endpoint: &Endpoint) -> bool {
    !matches!(connect(endpoint, POLL_INTERVAL), Err(ExchangeError::Absent))
}

fn run_exchange(
    endpoint: &Endpoint,
    frame: &[u8],
    wait: Duration,
    connected: &AtomicBool,
) -> Result<Response, ExchangeError> {
    let stream = connect(endpoint, wait)?;
    connected.store(true, Ordering::SeqCst);
    let mut s = &stream;
    s.write_all(frame)
        .and_then(|()| s.flush())
        .map_err(|e| ExchangeError::Io(e.to_string()))?;
    let body = read_frame(&mut s)
        .map_err(|e| ExchangeError::Io(e.message))?
        .ok_or_else(|| ExchangeError::Io("the daemon closed the connection".into()))?;
    // CONTRACTS 4: peek the envelope before relying on the body.
    let envelope = Envelope::peek(&body).map_err(|e| ExchangeError::Io(e.message))?;
    if envelope.check_version().is_err() {
        return Err(ExchangeError::Mismatch);
    }
    let response: Response = decode_body(&body).map_err(|e| ExchangeError::Io(e.message))?;
    if let Outcome::Err(e) = &response.result
        && e.code == ErrorCode::VersionMismatch
    {
        return Err(ExchangeError::Mismatch);
    }
    Ok(response)
}

/// Sends one request and reads its response. With a `timeout`, the whole
/// exchange (connect, write, read) runs on a separate thread and the caller
/// returns when the timeout passes, whatever the daemon does. Without one it
/// waits for the reply or for the connection to close.
pub fn exchange(
    endpoint: &Endpoint,
    request: &Request,
    timeout: Option<Duration>,
) -> Result<Response, ExchangeError> {
    let frame = encode_frame(request).map_err(|e| ExchangeError::Io(e.message))?;
    let Some(timeout) = timeout else {
        return run_exchange(endpoint, &frame, CONNECT_WAIT, &AtomicBool::new(false));
    };
    let connected = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let flag = Arc::clone(&connected);
    let ep = endpoint.clone();
    let spawned = std::thread::Builder::new()
        .name("sm-exchange".into())
        .spawn(move || {
            let _ = tx.send(run_exchange(&ep, &frame, timeout, &flag));
        });
    if let Err(e) = spawned {
        return Err(ExchangeError::Io(e.to_string()));
    }
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => Err(ExchangeError::Timeout {
            connected: connected.load(Ordering::SeqCst),
        }),
    }
}

// ---------------------------------------------------------------------------
// Version mismatch (CONTRACTS 4)
// ---------------------------------------------------------------------------

/// Asks a daemon of another build to exit and waits, up to `budget`, for
/// its endpoint to close. Returns whether it closed. The caller then starts
/// a daemon of its own build under the lock.
pub fn retire_other_build(
    endpoint: &Endpoint,
    cwd: &str,
    profile: &ProfileName,
    budget: Duration,
) -> bool {
    let until = Instant::now() + budget;
    let shutdown = Request::new(
        cwd,
        profile.clone(),
        Op::Shutdown(ShutdownParams {
            reason: REASON_VERSION_MISMATCH.to_owned(),
        }),
    );
    // The reply comes from the other build, so it reads as a mismatch; the
    // only thing that matters is that the endpoint closes afterwards.
    let _ = exchange(endpoint, &shutdown, Some(budget.min(PROBE_TIMEOUT)));
    loop {
        if !is_listening(endpoint) {
            return true;
        }
        if Instant::now() + POLL_INTERVAL > until {
            return false;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

// ---------------------------------------------------------------------------
// Stopping the daemon (B-155)
// ---------------------------------------------------------------------------

/// How long `stop` waits for the daemon to go: its 5 s shutdown grace, the
/// spooling of queued captures and the process exit, with room to spare.
pub const STOP_BUDGET: Duration = Duration::from_secs(15);

/// What [`stop_daemon`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// Nothing listened at the endpoint and nothing held the lock.
    NotRunning,
    /// The daemon closed its endpoint and released its lock. `pid` is the
    /// one its `hello` reported, when it answered `hello`.
    Stopped { pid: Option<u32> },
}

/// Whether a process holds the profile's lock. The lock file is opened only
/// if it exists, so the check creates nothing; when the lock is free, the
/// check takes it and releases it at once.
fn lock_held(paths: &ProfilePaths) -> bool {
    let Ok(file) = OpenOptions::new().write(true).open(paths.lock_file()) else {
        return false;
    };
    matches!(file.try_lock(), Err(TryLockError::WouldBlock))
}

/// The error when the daemon is still there after `budget`.
fn not_stopped(paths: &ProfilePaths, pid: Option<u32>, budget: Duration) -> CoreError {
    let which = pid.map_or_else(
        || String::from("the spatial-memory daemon process"),
        |p| format!("process {p}"),
    );
    CoreError::new(
        ErrorCode::Timeout,
        format!(
            "the daemon of profile {} did not stop within {} s",
            paths.profile.as_str(),
            budget.as_secs()
        ),
    )
    .with_next_action(format!(
        "See {} for what the daemon was doing, then end {which} (Task Manager on Windows, \
         kill on Linux and macOS).",
        paths.daemon_log().display()
    ))
}

/// Asks the profile's daemon to shut down with `reason` and waits, up to
/// `budget`, until its endpoint is closed and its lock is free (B-155). It
/// never starts a daemon and never creates the profile's folder. A daemon
/// of another build is asked too: shutdown is the one request every build
/// accepts (CONTRACTS 4). A daemon that holds the lock but does not listen
/// yet is still opening its store; it is waited for, then asked.
pub fn stop_daemon(
    paths: &ProfilePaths,
    cwd: &str,
    reason: &str,
    budget: Duration,
) -> Result<Stop, CoreError> {
    let until = Instant::now() + budget;
    let endpoint = paths.endpoint();
    while !is_listening(&endpoint) {
        if !lock_held(paths) {
            return Ok(Stop::NotRunning);
        }
        if Instant::now() + POLL_INTERVAL > until {
            return Err(not_stopped(paths, None, budget));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    let hello = Request::new(cwd, paths.profile.clone(), Op::Hello);
    let pid = match exchange(&endpoint, &hello, Some(PROBE_TIMEOUT)) {
        Ok(response) => match response.into_result() {
            Ok(Reply::Hello(info)) => Some(info.pid),
            _ => None,
        },
        Err(ExchangeError::Refused(e)) => return Err(e),
        Err(_) => None,
    };
    let shutdown = Request::new(
        cwd,
        paths.profile.clone(),
        Op::Shutdown(ShutdownParams {
            reason: reason.to_owned(),
        }),
    );
    match exchange(&endpoint, &shutdown, Some(PROBE_TIMEOUT)) {
        Ok(response) => {
            response.into_result()?;
        }
        Err(ExchangeError::Refused(e)) => return Err(e),
        // Another build's reply reads as a mismatch, a daemon that already
        // went reads as absent, and a slow one is caught by the wait below.
        Err(_) => {}
    }
    loop {
        if !is_listening(&endpoint) && !lock_held(paths) {
            return Ok(Stop::Stopped { pid });
        }
        if Instant::now() + POLL_INTERVAL > until {
            return Err(not_stopped(paths, pid, budget));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

// ---------------------------------------------------------------------------
// Holding the daemon stopped (B-180)
// ---------------------------------------------------------------------------

/// The `--profile` words a command needs to name this profile.
fn profile_words(paths: &ProfilePaths) -> String {
    let name = paths.profile.as_str();
    if name == sm_core::config::DEFAULT_PROFILE {
        String::new()
    } else {
        format!(" --profile {name}")
    }
}

/// Puts the hold on the profile (B-180): `daemon.hold`, one line written in
/// one call, naming the time, this process and `reason`. While it exists no
/// client starts a daemon and a daemon started anyway exits at once. It
/// creates the profile's folder if there is none.
pub fn hold(paths: &ProfilePaths, reason: &str) -> Result<(), CoreError> {
    let path = paths.hold_file();
    let line = format!(
        "since {} by process {}: {reason}\n",
        Timestamp::now().to_rfc3339(),
        std::process::id()
    );
    create_profile_dir(&paths.dir)
        .and_then(|()| std::fs::write(&path, line))
        .map_err(|e| {
            CoreError::storage(format!("cannot write the hold {}: {e}", path.display()))
                .with_next_action(
                    "Check that the profile folder can be written, then run `spatial-memory \
                     stop --hold` again.",
                )
        })
}

/// Lifts the hold. Returns its line, or `None` when the profile was not
/// held. It starts no daemon: the next request that needs one does.
pub fn release(paths: &ProfilePaths) -> Result<Option<String>, CoreError> {
    let Some(line) = sm_daemon::read_hold(paths) else {
        return Ok(None);
    };
    let path = paths.hold_file();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(Some(line)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CoreError::storage(format!(
            "cannot remove the hold {}: {e}",
            path.display()
        ))
        .with_next_action(format!(
            "Close what holds the file open, or delete {} yourself.",
            path.display()
        ))),
    }
}

/// The error a waiting client gives while the profile is held (B-180).
pub fn held_error(paths: &ProfilePaths, hold: &str) -> CoreError {
    CoreError::new(
        ErrorCode::DaemonUnavailable,
        format!(
            "the daemon of profile {} is held stopped for an upgrade or an uninstall ({hold})",
            paths.profile.as_str()
        ),
    )
    .with_next_action(format!(
        "Run `spatial-memory release{}` when the upgrade is done; the next request starts \
         the daemon again.",
        profile_words(paths)
    ))
}

// ---------------------------------------------------------------------------
// The waiting client: the MCP proxy and the command line
// ---------------------------------------------------------------------------

/// A client that may wait for the daemon (CONTRACTS 3.1): it starts one
/// when none answers, then polls `hello` every 50 ms for up to 10 s. A
/// daemon that is still loading counts as connected.
pub struct Connector<'a> {
    pub paths: ProfilePaths,
    pub spawner: &'a dyn Spawn,
    pub cwd: String,
    pub poll: Duration,
    pub window: Duration,
}

impl<'a> Connector<'a> {
    pub fn new(paths: ProfilePaths, spawner: &'a dyn Spawn, cwd: String) -> Self {
        Connector {
            paths,
            spawner,
            cwd,
            poll: POLL_INTERVAL,
            window: CONNECT_WINDOW,
        }
    }

    pub fn endpoint(&self) -> Endpoint {
        self.paths.endpoint()
    }

    fn request(&self, call: Op) -> Request {
        Request::new(self.cwd.clone(), self.paths.profile.clone(), call)
    }

    /// A daemon the client started reports why it stopped only in its log,
    /// so the next action names the log before `doctor` (B-138).
    fn unavailable(&self, why: &str) -> CoreError {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "no daemon answered at {} within {} s{why}",
                self.endpoint().display_path(),
                self.window.as_secs()
            ),
        )
        .with_next_action(format!(
            "Read {} for why the daemon stopped or did not answer, then run \
             `spatial-memory doctor`.",
            self.paths.daemon_log().display()
        ))
    }

    /// Waits until a daemon of this build answers `hello`, starting one if
    /// none does and replacing one of another build. Returns the daemon's
    /// `hello` reply, or `None` when it answered with an error such as
    /// `daemon_loading` (still counted as connected).
    ///
    /// After retiring a daemon of another build, the lock can still be held
    /// for a moment: that daemon closes its endpoint first and its exit
    /// releases the lock last (CONTRACTS 3.1, daemon rule 5). So once a
    /// mismatch was handled, each poll that finds nothing listening tries
    /// the lock once more, until this client has spawned a daemon.
    pub fn ensure_daemon(&self) -> Result<Option<Reply>, CoreError> {
        let endpoint = self.endpoint();
        let until = Instant::now() + self.window;
        let mut tried = false;
        let mut spawned = false;
        let mut retired = false;
        let mut last = String::new();
        let start = || {
            try_start(&self.paths, self.spawner).map_err(|e| {
                CoreError::new(
                    ErrorCode::DaemonUnavailable,
                    format!("could not start the daemon: {e}"),
                )
            })
        };
        loop {
            let hello = self.request(Op::Hello);
            match exchange(&endpoint, &hello, Some(PROBE_TIMEOUT)) {
                Ok(response) => return Ok(response.into_result().ok()),
                Err(ExchangeError::Absent) => {
                    if let Some(hold) = sm_daemon::read_hold(&self.paths) {
                        return Err(held_error(&self.paths, &hold));
                    }
                    if !spawned && (!tried || retired) {
                        tried = true;
                        spawned = start()?;
                    }
                    last = String::from("; nothing is listening");
                }
                Err(ExchangeError::Mismatch) => {
                    if retired {
                        return Err(ExchangeError::Mismatch.into_core(&endpoint));
                    }
                    retired = true;
                    let left = until.saturating_duration_since(Instant::now());
                    retire_other_build(&endpoint, &self.cwd, &self.paths.profile, left);
                    tried = true;
                    spawned = start().unwrap_or(false);
                }
                Err(ExchangeError::Timeout { .. }) => {
                    last = String::from("; the daemon did not answer hello");
                }
                Err(ExchangeError::Io(e)) => last = format!("; {e}"),
                // Waiting cannot help: the endpoint is not to be used.
                Err(ExchangeError::Refused(e)) => return Err(e),
            }
            if Instant::now() + self.poll > until {
                return Err(self.unavailable(&last));
            }
            std::thread::sleep(self.poll);
        }
    }

    /// Sends one op and waits for its reply with no client-side deadline
    /// (`deadline_ms` absent: CONTRACTS 5, wait for completion). A request
    /// is re-sent only when it was certainly not executed: nothing was
    /// listening, or the daemon was another build.
    ///
    /// A `status` reply comes back with this program's stamp in
    /// `client_build` (B-231), so the command line and the MCP server both
    /// say which build asked.
    pub fn call(&self, call: Op) -> Result<Reply, CoreError> {
        self.send(call).map(with_client_build)
    }

    fn send(&self, call: Op) -> Result<Reply, CoreError> {
        self.ensure_daemon()?;
        let endpoint = self.endpoint();
        let request = self.request(call);
        match exchange(&endpoint, &request, None) {
            Ok(response) => response.into_result(),
            Err(ExchangeError::Absent | ExchangeError::Mismatch) => {
                self.ensure_daemon()?;
                match exchange(&endpoint, &request, None) {
                    Ok(response) => response.into_result(),
                    Err(e) => Err(e.into_core(&endpoint)),
                }
            }
            Err(e) => Err(e.into_core(&endpoint)),
        }
    }
}

/// Puts this program's stamp in a `status` reply (B-231); other replies
/// pass unchanged.
fn with_client_build(mut reply: Reply) -> Reply {
    if let Reply::Status(s) = &mut reply {
        s.client_build = Some(crate::version::STAMP.to_owned());
    }
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn s3_windows_flags_detach_without_breakaway() {
        use windows_sys::Win32::System::Threading::{
            CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, DETACHED_PROCESS,
        };
        assert_eq!(WINDOWS_CREATION_FLAGS & CREATE_BREAKAWAY_FROM_JOB, 0);
        assert_eq!(CREATE_BREAKAWAY_FROM_JOB, 0x0100_0000);
        assert_ne!(WINDOWS_CREATION_FLAGS & DETACHED_PROCESS, 0);
        assert_ne!(WINDOWS_CREATION_FLAGS & CREATE_NEW_PROCESS_GROUP, 0);
        assert_ne!(WINDOWS_CREATION_FLAGS & CREATE_NO_WINDOW, 0);
    }

    #[cfg(windows)]
    #[test]
    fn spike2_detached_daemon_does_not_inherit_the_callers_pipes() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{
            GetHandleInformation, HANDLE_FLAG_INHERIT, SetHandleInformation,
        };
        let (reader, writer) = io::pipe().unwrap();
        let handles = [reader.as_raw_handle(), writer.as_raw_handle()];
        let flags = |h| {
            let mut f = 0u32;
            // SAFETY: `h` is a live pipe handle owned by this test.
            assert_ne!(unsafe { GetHandleInformation(h, &mut f) }, 0);
            f
        };
        for h in handles {
            // SAFETY: as above; this makes the handle inheritable, as a
            // client's pipe given to the hook is.
            assert_ne!(
                unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) },
                0
            );
            assert_ne!(flags(h) & HANDLE_FLAG_INHERIT, 0);
        }
        disinherit(&handles);
        for h in handles {
            assert_eq!(flags(h) & HANDLE_FLAG_INHERIT, 0);
        }
        // A null handle (no console) is skipped, not an error.
        disinherit(&[std::ptr::null_mut()]);
    }

    #[cfg(windows)]
    #[test]
    fn s3_the_windows_spawn_refuses_an_argument_it_would_have_to_quote() {
        let exe = std::path::Path::new(r"C:\no\such\spatial-memory.exe");
        for bad in ["two words", "", "tab\there", "quote\"d"] {
            let mut cmd = Command::new(exe);
            cmd.arg("daemon").arg(bad);
            let err = spawn_detached(&cmd).expect_err(bad);
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{bad:?}");
        }
        // A plain command line reaches `CreateProcessW`, which reports the
        // missing file.
        let cmd = daemon_command(exe, &ProfileName::parse("work").unwrap());
        let err = spawn_detached(&cmd).expect_err("no such file");
        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");
    }

    #[test]
    fn s3_daemon_command_is_daemon_profile_with_null_stdio() {
        let profile = ProfileName::parse("work").unwrap();
        let cmd = daemon_command(std::path::Path::new("spatial-memory"), &profile);
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["daemon", "--profile", "work"]);
    }

    #[test]
    fn s3_exchange_with_nothing_listening_is_absent_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let paths = ProfilePaths::new(dir.path().to_path_buf(), ProfileName::default_profile());
        let req = Request::new("/", ProfileName::default_profile(), Op::Hello);
        let started = Instant::now();
        let r = exchange(&paths.endpoint(), &req, Some(Duration::from_millis(500)));
        assert_eq!(r, Err(ExchangeError::Absent));
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    /// The text `sm_daemon::connect` gives for an io error at `endpoint`.
    fn reach_error(endpoint: &Endpoint, e: &io::Error) -> CoreError {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("cannot reach {}: {e}", endpoint.display_path()),
        )
    }

    #[test]
    fn s3_connect_errors_from_the_daemon_crate_sort_into_exchange_errors() {
        let endpoint = Endpoint::UnixSocket(PathBuf::from("/run/user/1000/sm/timed out/d.sock"));
        let os = |kind: io::ErrorKind| {
            // An operating-system error of `kind` on this platform.
            let code = (1..20_000)
                .find(|c| io::Error::from_raw_os_error(*c).kind() == kind)
                .expect("an os error of the kind");
            io::Error::from_raw_os_error(code)
        };
        assert_eq!(
            classify_connect_error(
                &endpoint,
                reach_error(&endpoint, &os(io::ErrorKind::NotFound))
            ),
            ExchangeError::Absent
        );
        assert_eq!(
            classify_connect_error(
                &endpoint,
                reach_error(&endpoint, &os(io::ErrorKind::ConnectionRefused))
            ),
            ExchangeError::Absent
        );
        assert_eq!(
            classify_connect_error(
                &endpoint,
                reach_error(&endpoint, &os(io::ErrorKind::TimedOut))
            ),
            ExchangeError::Timeout { connected: false }
        );
        // interprocess's own timeout is not an operating-system error.
        let custom = io::Error::new(
            io::ErrorKind::TimedOut,
            "timed out while connecting to local socket server",
        );
        assert_eq!(
            classify_connect_error(&endpoint, reach_error(&endpoint, &custom)),
            ExchangeError::Timeout { connected: false }
        );
        // The endpoint's own text is not read: this path holds "timed out".
        let other = io::Error::other("bad name");
        assert_eq!(
            classify_connect_error(&endpoint, reach_error(&endpoint, &other)),
            ExchangeError::Io("bad name".to_owned())
        );
        let denied = os(io::ErrorKind::PermissionDenied);
        assert_eq!(
            classify_connect_error(&endpoint, reach_error(&endpoint, &denied)),
            ExchangeError::Io(denied.to_string())
        );
    }

    #[test]
    fn s3_an_owner_check_refusal_is_reported_unchanged_and_never_retried() {
        let endpoint = Endpoint::UnixSocket(PathBuf::from("/tmp/sm/d.sock"));
        let refusal = CoreError::new(
            ErrorCode::DaemonUnavailable,
            "refusing /tmp/sm/d.sock: it is owned by another user (uid 1001), not by this user \
             (uid 1000)",
        )
        .with_next_action("Do not use this socket: another user may be posing as the daemon.");
        let sorted = classify_connect_error(&endpoint, refusal.clone());
        assert_eq!(sorted, ExchangeError::Refused(refusal.clone()));
        assert_eq!(sorted.into_core(&endpoint), refusal);
    }

    /// B-136: a daemon that exits when idle removes its socket between a
    /// client's connect and its check. That refusal is polled again, not
    /// reported at once, so the client starts a daemon once nothing listens.
    #[test]
    fn s3_a_refusal_from_an_exiting_daemon_is_retried() {
        let endpoint = Endpoint::UnixSocket(PathBuf::from("/tmp/sm/d.sock"));
        let at = endpoint.display_path();
        let gone = CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "refusing {at}: the socket is gone after connecting, so the peer cannot be \
                 trusted"
            ),
        )
        .with_next_action("Do not use this socket: another user may be posing as the daemon.");
        assert!(sm_daemon::refusal_may_pass(&gone));
        assert_eq!(
            classify_connect_error(&endpoint, gone.clone()),
            ExchangeError::Io(gone.message.clone())
        );
        let process_gone = CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "refusing {at}: cannot check the user of process 4242, which serves it: The \
                 parameter is incorrect. (os error 87)"
            ),
        );
        assert!(matches!(
            classify_connect_error(&endpoint, process_gone),
            ExchangeError::Io(_)
        ));
    }

    /// Counts the daemons it was asked to start and starts none.
    struct Counting(std::sync::atomic::AtomicUsize);

    impl Spawn for Counting {
        fn spawn_daemon(&self, _: &ProfileName) -> io::Result<()> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn s3_b180_a_held_profile_spawns_no_daemon_and_a_waiting_client_fails_at_once() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = ProfilePaths::new(tmp.path().join("h"), ProfileName::default_profile());
        let spawner = Counting(Default::default());
        hold(&paths, "stop --hold").unwrap();
        let line = std::fs::read_to_string(paths.hold_file()).unwrap();
        assert!(
            line.starts_with("since ") && line.ends_with(": stop --hold\n"),
            "{line}"
        );
        assert_eq!(line.lines().count(), 1, "{line}");

        // The hook's start (and the waiting client's) spawns nothing.
        assert!(!try_start(&paths, &spawner).unwrap());
        assert_eq!(spawner.0.load(Ordering::SeqCst), 0);
        assert!(!paths.lock_file().exists(), "a held start creates nothing");

        // The MCP proxy and the command line fail at once, naming release.
        let connector = Connector::new(paths.clone(), &spawner, String::new());
        let began = Instant::now();
        let e = connector.ensure_daemon().unwrap_err();
        assert!(
            began.elapsed() < Duration::from_secs(5),
            "{:?}",
            began.elapsed()
        );
        assert_eq!(e.code, ErrorCode::DaemonUnavailable, "{e:?}");
        assert!(e.message.contains("held stopped"), "{e:?}");
        assert!(e.message.contains(line.trim()), "{e:?}");
        assert!(e.next_action.contains("`spatial-memory release`"), "{e:?}");
        assert_eq!(spawner.0.load(Ordering::SeqCst), 0);

        // Release lifts it, once; then a start spawns again.
        assert_eq!(release(&paths).unwrap().as_deref(), Some(line.trim()));
        assert!(!paths.hold_file().exists());
        assert_eq!(release(&paths).unwrap(), None);
        assert!(try_start(&paths, &spawner).unwrap());
        assert_eq!(spawner.0.load(Ordering::SeqCst), 1);

        // Another profile's next action names it.
        let work = ProfilePaths::new(tmp.path().join("h"), ProfileName::parse("work").unwrap());
        let e = held_error(&work, "since then");
        assert!(
            e.next_action
                .contains("`spatial-memory release --profile work`"),
            "{e:?}"
        );
    }

    /// B-117: the folder a client creates before it starts the daemon is
    /// user-only, so the daemon's start finds nothing to fix.
    #[cfg(unix)]
    #[test]
    fn s3_the_client_creates_the_profile_folder_user_only() {
        use std::os::unix::fs::PermissionsExt;
        struct NoSpawn;
        impl Spawn for NoSpawn {
            fn spawn_daemon(&self, _: &ProfileName) -> io::Result<()> {
                Ok(())
            }
        }
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let tmp = tempfile::tempdir().unwrap();
        // A folder made the usual way shows the default mode here.
        let usual = tmp.path().join("usual");
        std::fs::create_dir(&usual).unwrap();

        let root = tmp.path().join("data").join("spatial-memory");
        let paths = ProfilePaths::new(root.clone(), ProfileName::default_profile());
        assert!(try_start(&paths, &NoSpawn).unwrap(), "the lock was free");
        assert_eq!(mode(&paths.dir), 0o700, "the profile folder");
        assert!(paths.lock_file().is_file());
        // Its missing parents get the default mode, as the daemon makes them.
        assert_eq!(mode(&root), mode(&usual), "the data root");
        assert_eq!(mode(root.parent().unwrap()), mode(&usual));
        // A second start finds the folder and keeps it user-only.
        assert!(try_start(&paths, &NoSpawn).unwrap());
        assert_eq!(mode(&paths.dir), 0o700);

        // A folder that already exists is the daemon's to fix (it sets it
        // to 0700 and logs it), not the client's.
        let old = tmp.path().join("old");
        std::fs::create_dir(&old).unwrap();
        std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o755)).unwrap();
        create_profile_dir(&old).unwrap();
        assert_eq!(mode(&old), 0o755);
        // A file where the folder goes is an error, as with create_dir_all.
        let file = tmp.path().join("file");
        std::fs::write(&file, b"").unwrap();
        assert!(create_profile_dir(&file).is_err());
    }
}
