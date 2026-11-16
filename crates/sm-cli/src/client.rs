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

use std::fs::{OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use interprocess::ConnectWaitMode;
use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{ConnectOptions, GenericFilePath, GenericNamespaced, Name};
use sm_core::config::ProfileName;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    Envelope, Op, Outcome, Reply, Request, Response, ShutdownParams, decode_body, encode_frame,
    read_frame,
};

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

/// The daemon command line, ready to spawn.
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
/// passes `bInheritHandles = TRUE`, so without this the daemon inherits
/// every inheritable handle of its parent, including the pipes the client
/// gave the hook or the proxy as stdin and stdout. The client then sees no
/// end of file on stdout until the daemon exits, which is hours. Errors
/// are ignored: a handle that cannot be changed is a console or is absent.
/// `Stdio::inherit` in a later spawn is unaffected, because std duplicates
/// the handle as inheritable for that child.
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

/// This process's standard handles.
#[cfg(windows)]
fn std_handles() -> [std::os::windows::io::RawHandle; 3] {
    use std::os::windows::io::AsRawHandle;
    [
        io::stdin().as_raw_handle(),
        io::stdout().as_raw_handle(),
        io::stderr().as_raw_handle(),
    ]
}

impl Spawn for DetachedSpawn {
    fn spawn_daemon(&self, profile: &ProfileName) -> io::Result<()> {
        #[cfg(windows)]
        disinherit(&std_handles());
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

/// CONTRACTS 3.1: one `try_lock` on the profile's lock file, no retry. When
/// it is acquired, spawn the daemon and drop the lock at once (the daemon
/// takes it for its lifetime). Returns whether a daemon was spawned. A held
/// lock means another client is starting one, or one is running.
pub fn try_start(paths: &ProfilePaths, spawner: &dyn Spawn) -> io::Result<bool> {
    std::fs::create_dir_all(&paths.dir)?;
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

fn is_absent(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

fn connect(
    endpoint: &Endpoint,
    wait: Duration,
) -> Result<interprocess::local_socket::Stream, ExchangeError> {
    let name = endpoint_name(endpoint).map_err(|e| ExchangeError::Io(e.to_string()))?;
    ConnectOptions::new()
        .name(name)
        .wait_mode(ConnectWaitMode::Timeout(wait))
        .connect_sync()
        .map_err(|e| {
            if is_absent(&e) {
                ExchangeError::Absent
            } else if matches!(
                e.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ) {
                ExchangeError::Timeout { connected: false }
            } else {
                ExchangeError::Io(e.to_string())
            }
        })
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

    fn unavailable(&self, why: &str) -> CoreError {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "no daemon answered at {} within {} s{why}",
                self.endpoint().display_path(),
                self.window.as_secs()
            ),
        )
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
    pub fn call(&self, call: Op) -> Result<Reply, CoreError> {
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
}
