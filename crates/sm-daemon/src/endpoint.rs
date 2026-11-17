//! The local endpoint (CONTRACTS 3.1 and 4): a Unix domain socket, or a
//! named pipe on Windows, carrying length-prefixed JSON frames.
//!
//! Unix (SPEC 3 rule 1; B-95, B-112; e-report decisions 31 and 32):
//! - **Folder.** `bind` creates a missing socket folder with mode 0700.
//!   The fallback folder `sm_core::paths::unix_socket_path` gives when the
//!   profile path is too long, `<XDG_RUNTIME_DIR or /tmp>/spatial-memory-<hash>/`,
//!   must be a real folder (not a link) that this user owns and that
//!   neither group nor others can write to; `bind` and `connect` both refuse
//!   any other, with an error that names it. A socket folder another user
//!   owns is refused wherever it is. A profile folder that its group can
//!   write to is accepted: under the user-private-group umask 002 (Debian,
//!   Ubuntu) every folder is created that way, and the peer check below
//!   still holds.
//! - **Owner.** The socket must belong to the current (effective) user,
//!   checked with `lstat`, which does not follow a link. `connect` refuses
//!   a socket another user owns before connecting. Once connected, before
//!   anything is sent, it refuses a socket that is gone or is another
//!   user's, and a peer whose effective user id is not this user's. The
//!   peer's id is read from the connected socket by interprocess's
//!   `peer_creds`: `SO_PEERCRED` on Linux, `LOCAL_PEERCRED` (the call
//!   `getpeereid` makes) on macOS and the BSDs. The daemon checks each
//!   client the same way and drops one of another user. `bind` refuses to
//!   replace another user's socket. Each refusal has a next action.
//! - **Mode.** A leftover socket of this user is removed before binding
//!   (the caller holds `daemon.lock`, so no live daemon owns it) and the
//!   socket is created with mode 0600: interprocess runs `fchmod` on the
//!   socket before `bind`. Where that is unsupported (interprocess 2.4.8
//!   documents it for Linux, FreeBSD 14.3 and later and OpenBSD, and maps
//!   `EINVAL` to `Unsupported`; macOS is expected to be one), the socket is
//!   bound in a new folder that only this user can enter, with a name
//!   nobody can predict, set to 0600 and then renamed into place, so no
//!   other user can connect before its mode is set. That path is
//!   unverified on macOS (no machine); it runs in the Linux tests by
//!   calling it directly.
//!
//! Windows: the pipe is created with `FILE_FLAG_FIRST_PIPE_INSTANCE`, which
//! interprocess adds on the first instance itself, so a second daemon cannot
//! create an instance of a pipe another process owns; remote clients are
//! rejected (interprocess's default `PIPE_REJECT_REMOTE_CLIENTS`). The
//! descriptor grants `GENERIC_ALL` to the current user only and is
//! protected from inheritance: SDDL `D:P(A;;GA;;;<user SID>)`, built with
//! `ConvertStringSecurityDescriptorToSecurityDescriptorW`, copied into an
//! owned interprocess `SecurityDescriptor` and passed through
//! `ListenerOptionsExt::security_descriptor`. Once connected, the client
//! reads the server's process id (`GetNamedPipeServerProcessId`, through
//! interprocess's `peer_creds`), opens that process's token and refuses the
//! pipe unless the token's user SID equals the current user's (`EqualSid`).
//! The daemon does not open its clients' tokens: the descriptor already
//! admits no other user.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use interprocess::ConnectWaitMode;
use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{ConnectOptions, ListenerNonblockingMode, ListenerOptions, Name};
use interprocess::local_socket::{Listener, Stream};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::paths::Endpoint;

/// The user the endpoint's checks are made for: the effective user id on
/// Unix, the user SID in string form (`S-1-5-21-...`) on Windows.
#[cfg(unix)]
pub(crate) type User = u32;
/// The user the endpoint's checks are made for: the effective user id on
/// Unix, the user SID in string form (`S-1-5-21-...`) on Windows.
#[cfg(windows)]
pub(crate) type User = String;

/// The user this process runs as.
pub(crate) fn current_user() -> io::Result<User> {
    #[cfg(unix)]
    {
        Ok(security::current_uid())
    }
    #[cfg(windows)]
    {
        security::current_user_sid()
    }
}

fn name_of(endpoint: &Endpoint) -> io::Result<Name<'_>> {
    match endpoint {
        Endpoint::NamedPipe(n) => {
            #[cfg(windows)]
            {
                n.as_str()
                    .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
            }
            #[cfg(not(windows))]
            {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("named pipe {n} on a platform without named pipes"),
                ))
            }
        }
        Endpoint::UnixSocket(p) => file_name(p),
    }
}

fn file_name(p: &Path) -> io::Result<Name<'_>> {
    p.to_fs_name::<interprocess::local_socket::GenericFilePath>()
}

fn unavailable(endpoint: &Endpoint, e: &io::Error) -> CoreError {
    CoreError::new(
        ErrorCode::DaemonUnavailable,
        format!("cannot reach {}: {e}", endpoint.display_path()),
    )
}

/// A bound endpoint. A socket bound in a private folder and moved into
/// place is removed when this is dropped; interprocess removes a socket it
/// bound in place itself (`reclaim_name`).
pub(crate) struct Bound {
    listener: Listener,
    moved: Option<PathBuf>,
    /// On Unix, the only user whose clients are served.
    #[cfg(unix)]
    serves: Option<User>,
}

/// One connection taken from the endpoint.
pub(crate) enum Accepted {
    /// A client of this user.
    Client(Stream),
    /// A client that failed the peer check, already dropped; the text says
    /// why, for the log. Unix only: on Windows the descriptor refuses
    /// another user's client before it connects.
    #[cfg_attr(windows, allow(dead_code))]
    Refused(String),
}

impl Bound {
    fn new(listener: Listener, moved: Option<PathBuf>) -> Self {
        Bound {
            listener,
            moved,
            #[cfg(unix)]
            serves: None,
        }
    }

    /// Accepts one connection. On Unix a client whose effective user id is
    /// not this user's is refused and dropped before anything is read.
    pub(crate) fn accept(&self) -> io::Result<Accepted> {
        let stream = self.listener.accept()?;
        #[cfg(unix)]
        if let Some(me) = self.serves
            && let Err(why) = peer_is(&stream, me)
        {
            return Ok(Accepted::Refused(format!("refused a client: {why}")));
        }
        Ok(Accepted::Client(stream))
    }
}

impl Drop for Bound {
    fn drop(&mut self) {
        if let Some(path) = self.moved.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Binds the endpoint, retrying every `every` for up to `wait`.
pub(crate) fn bind(endpoint: &Endpoint, wait: Duration, every: Duration) -> Result<Bound> {
    let me = current_user().map_err(|e| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "cannot bind {}: cannot read the current user: {e}",
                endpoint.display_path()
            ),
        )
    })?;
    bind_as(endpoint, wait, every, Some(me))
}

/// [`bind`] for the user `me`. The folder and owner checks and the check of
/// each client run on Unix only; on Windows the pipe's descriptor admits
/// the current user only.
fn bind_as(
    endpoint: &Endpoint,
    wait: Duration,
    every: Duration,
    me: Option<User>,
) -> Result<Bound> {
    #[cfg(windows)]
    let _ = &me;
    if let Endpoint::UnixSocket(path) = endpoint {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            make_folder(dir)?;
            #[cfg(unix)]
            if let Some(me) = me {
                check_folder(dir, me).map_err(|e| {
                    e.with_next_action(format!(
                        "The socket's folder is not private to you. Remove {} if it is \
                         yours, or have that user or an administrator remove it; or set \
                         XDG_RUNTIME_DIR to a folder only you can enter, or use a shorter \
                         profile path so the socket lives in the profile folder. Then \
                         start the daemon again.",
                        dir.display()
                    ))
                })?;
            }
        }
        #[cfg(unix)]
        if let Some(me) = me {
            check_owner(path, me).map_err(|e| {
                e.with_next_action(format!(
                    "Another user's socket is in the way. Set XDG_RUNTIME_DIR to a folder \
                     only you can enter, or use a shorter profile path so the socket lives \
                     in the profile folder, then start the daemon again; or have that user \
                     or an administrator remove {}.",
                    path.display()
                ))
            })?;
        }
        // Leftover from a daemon of this user that did not exit cleanly.
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(CoreError::storage(format!(
                    "cannot remove the leftover socket {}: {e}",
                    path.display()
                )));
            }
        }
    }
    let start = Instant::now();
    loop {
        match try_bind(endpoint) {
            Ok(mut bound) => {
                #[cfg(unix)]
                {
                    bound.serves = me;
                }
                #[cfg(windows)]
                let _ = &mut bound;
                return Ok(bound);
            }
            Err(e) if start.elapsed() >= wait => {
                return Err(CoreError::new(
                    ErrorCode::DaemonUnavailable,
                    format!("cannot bind {}: {e}", endpoint.display_path()),
                ));
            }
            Err(_) => std::thread::sleep(every),
        }
    }
}

/// Creates the socket's folder, and any missing parent, with mode 0700 on
/// Unix (B-112). An existing folder is left as it is and checked by the
/// caller.
fn make_folder(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    let made = {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
    };
    #[cfg(windows)]
    let made = std::fs::create_dir_all(dir);
    made.map_err(|e| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("cannot create the socket folder {}: {e}", dir.display()),
        )
    })
}

fn options(name: Name<'_>) -> ListenerOptions<'_> {
    ListenerOptions::new()
        .name(name)
        .nonblocking(ListenerNonblockingMode::Neither)
        .reclaim_name(true)
        .try_overwrite(false)
}

#[cfg(windows)]
fn try_bind(endpoint: &Endpoint) -> io::Result<Bound> {
    use interprocess::os::windows::local_socket::ListenerOptionsExt;
    let listener = options(name_of(endpoint)?)
        .security_descriptor(security::user_only_descriptor()?)
        .create_sync()?;
    Ok(Bound::new(listener, None))
}

#[cfg(unix)]
fn try_bind(endpoint: &Endpoint) -> io::Result<Bound> {
    use interprocess::os::unix::local_socket::ListenerOptionsExt;
    match options(name_of(endpoint)?).mode(0o600).create_sync() {
        Ok(listener) => Ok(Bound::new(listener, None)),
        Err(e) if e.kind() == io::ErrorKind::Unsupported => match endpoint {
            Endpoint::UnixSocket(path) => bind_private(path),
            Endpoint::NamedPipe(_) => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// Tries for a free private folder name before giving up.
#[cfg(unix)]
const PRIVATE_TRIES: u32 = 16;

/// A folder name for [`bind_private`] that another user cannot predict:
/// `.` and 8 hex characters from std's randomly keyed SipHash (each
/// `RandomState` takes fresh keys seeded from the operating system) over
/// the process id, the time and the attempt. Eight characters keep
/// `/.<name>/s` no longer than `/daemon.sock`.
#[cfg(unix)]
fn private_name(attempt: u32) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.write_u32(attempt);
    if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(t.as_nanos());
    }
    format!(".{:08x}", h.finish() as u32)
}

/// Binds `path` where `fchmod` before `bind` is unsupported: in a new
/// folder only this user can enter (mode 0700, created here with
/// `create`, which fails on any existing entry, never one that existed;
/// a collision takes another name), sets the socket to 0600, then renames
/// it to `path`. Connecting needs search permission on every folder of the
/// path, so nobody else can reach the socket before its mode is set. The
/// folder name is unpredictable and short (`.<8 hex>/s` is no longer than
/// `/daemon.sock`), so the path stays within the `sun_path` limit the
/// final path meets.
#[cfg(unix)]
fn bind_private(path: &Path) -> io::Result<Bound> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut dir = None;
    for attempt in 0..PRIVATE_TRIES {
        let candidate = parent.join(private_name(attempt));
        match std::fs::DirBuilder::new().mode(0o700).create(&candidate) {
            Ok(()) => {
                dir = Some(candidate);
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    let Some(dir) = dir else {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "no free private folder to bind in under {} after {PRIVATE_TRIES} tries",
                parent.display()
            ),
        ));
    };
    let temp = dir.join("s");
    let bound = (|| {
        let listener = options(file_name(&temp)?)
            .reclaim_name(false)
            .create_sync()?;
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600))?;
        std::fs::rename(&temp, path)?;
        Ok(Bound::new(listener, Some(path.to_path_buf())))
    })();
    // Gone after the rename; left only when a step failed.
    let _ = std::fs::remove_file(&temp);
    let _ = std::fs::remove_dir(&dir);
    bound
}

/// Whether `dir` is the fallback socket folder of
/// `sm_core::paths::unix_socket_path`: `spatial-memory-` and 16 hex
/// characters.
#[cfg(unix)]
fn is_fallback_folder(dir: &Path) -> bool {
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("spatial-memory-"))
        .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Refuses a socket folder another user owns, and a fallback folder that
/// is a link or that group or others can write to (B-112). The folder is
/// read with `lstat`. A missing folder passes; `connect` then finds no
/// socket.
#[cfg(unix)]
fn check_folder(dir: &Path, me: u32) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let refuse = |why: String| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("refusing the socket folder {}: {why}", dir.display()),
        )
    };
    let m = match std::fs::symlink_metadata(dir) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(refuse(format!("cannot read it: {e}"))),
    };
    if m.uid() != me {
        return Err(refuse(format!(
            "it is owned by another user (uid {}), not by this user (uid {me})",
            m.uid()
        )));
    }
    if is_fallback_folder(dir) {
        if !m.file_type().is_dir() {
            return Err(refuse("it is not a folder (a link or a file)".to_owned()));
        }
        let mode = m.mode() & 0o777;
        if mode & 0o022 != 0 {
            return Err(refuse(format!(
                "its group or others can write to it (mode {mode:o}; it must be 700)"
            )));
        }
    }
    Ok(())
}

/// Refuses an entry at `path` that another user owns, not following a
/// link. A missing entry passes.
#[cfg(unix)]
fn check_owner(path: &Path, me: u32) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.uid() == me => Ok(()),
        Ok(m) => Err(CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!(
                "refusing {}: it is owned by another user (uid {}), not by this user \
                 (uid {me})",
                path.display(),
                m.uid()
            ),
        )),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("cannot check the owner of {}: {e}", path.display()),
        )),
    }
}

/// Checks the peer of a connected Unix socket: its effective user id must
/// be `me`. An unreadable id is a refusal.
#[cfg(unix)]
fn peer_is(stream: &Stream, me: u32) -> std::result::Result<(), String> {
    match stream.peer_creds().map(|c| c.euid()) {
        Ok(Some(uid)) if uid == me => Ok(()),
        Ok(Some(uid)) => Err(format!(
            "the peer runs as another user (uid {uid}), not as this user (uid {me})"
        )),
        Ok(None) => Err("the peer's user id is not available on this platform".to_owned()),
        Err(e) => Err(format!("cannot read the peer's user id: {e}")),
    }
}

/// Connects to a daemon, waiting up to `timeout` for a free pipe instance.
///
/// On Unix, a socket another user owns, or a socket folder that is not
/// private, is refused before connecting. Once connected, before anything
/// is sent, the socket must still be there and be this user's, and the
/// peer's effective user id must be this user's. On Windows the pipe
/// server's process must run as this user. Every refusal is
/// `daemon_unavailable` with a next action; the stream is dropped unused.
pub fn connect(endpoint: &Endpoint, timeout: Duration) -> Result<Stream> {
    let me = current_user().map_err(|e| unavailable(endpoint, &e))?;
    connect_as(endpoint, timeout, Some(me))
}

/// [`connect`] for the user `me`. `None` skips the checks; only the tests
/// of this module pass it.
fn connect_as(endpoint: &Endpoint, timeout: Duration, me: Option<User>) -> Result<Stream> {
    if let Some(me) = &me {
        before_connect(endpoint, me)?;
    }
    let name = name_of(endpoint).map_err(|e| unavailable(endpoint, &e))?;
    let stream = ConnectOptions::new()
        .name(name)
        .wait_mode(ConnectWaitMode::Timeout(timeout))
        .connect_sync()
        .map_err(|e| unavailable(endpoint, &e))?;
    if let Some(me) = &me {
        check_connected(endpoint, &stream, me)?;
    }
    Ok(stream)
}

#[cfg(unix)]
const CONNECT_ACTION: &str = "Do not use this socket: another user may be posing as the \
     daemon. Set XDG_RUNTIME_DIR to a folder only you can enter, or use a shorter profile \
     path so the socket lives in the profile folder, then retry.";

/// The checks before connecting: the socket's folder, then the socket.
fn before_connect(endpoint: &Endpoint, me: &User) -> Result<()> {
    #[cfg(unix)]
    if let Endpoint::UnixSocket(path) = endpoint {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            check_folder(dir, *me).map_err(|e| e.with_next_action(CONNECT_ACTION))?;
        }
        check_owner(path, *me).map_err(|e| e.with_next_action(CONNECT_ACTION))?;
    }
    #[cfg(windows)]
    let _ = (endpoint, me);
    Ok(())
}

/// The checks on a connected stream, before anything is sent.
///
/// Unix: the folder again, then the socket entry, which must still exist
/// and be this user's (another user who bound the fallback path and
/// removed the entry after `accept` is refused here), then the peer's
/// effective user id, which no file operation can change.
#[cfg(unix)]
fn check_connected(endpoint: &Endpoint, stream: &Stream, me: &User) -> Result<()> {
    let Endpoint::UnixSocket(path) = endpoint else {
        return Ok(());
    };
    let refuse = |why: String| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("refusing {}: {why}", path.display()),
        )
        .with_next_action(CONNECT_ACTION)
    };
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        check_folder(dir, *me).map_err(|e| e.with_next_action(CONNECT_ACTION))?;
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => check_owner(path, *me).map_err(|e| e.with_next_action(CONNECT_ACTION))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(refuse(
                "the socket is gone after connecting, so the peer cannot be trusted".to_owned(),
            ));
        }
        Err(e) => return Err(refuse(format!("cannot check the socket: {e}"))),
    }
    peer_is(stream, *me).map_err(refuse)
}

/// The check on a connected pipe, before anything is sent: the process
/// that serves it must run as `me` (spike decision 1, B-95).
#[cfg(windows)]
fn check_connected(endpoint: &Endpoint, stream: &Stream, me: &User) -> Result<()> {
    let refuse = |why: String| {
        CoreError::new(
            ErrorCode::DaemonUnavailable,
            format!("refusing {}: {why}", endpoint.display_path()),
        )
        .with_next_action(
            "Do not use this pipe: a process of another user may be posing as the daemon. \
             Stop the process that serves it, or sign out and in again, then retry.",
        )
    };
    let pid = match stream.peer_creds().map(|c| c.pid()) {
        Ok(Some(pid)) => pid,
        Ok(None) => return Err(refuse("the pipe server's process id is unknown".to_owned())),
        Err(e) => {
            return Err(refuse(format!(
                "cannot read the pipe server's process id: {e}"
            )));
        }
    };
    match security::process_runs_as(pid, me) {
        Ok(true) => Ok(()),
        Ok(false) => Err(refuse(format!(
            "it is served by process {pid}, which runs as another user"
        ))),
        Err(e) => Err(refuse(format!(
            "cannot check the user of process {pid}, which serves it: {e}"
        ))),
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
mod security {
    //! The one libc call the Unix endpoint needs.

    /// This process's effective user id, the owner of every file it
    /// creates.
    pub(super) fn current_uid() -> u32 {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }
}

#[cfg(windows)]
pub use security::{current_user_sid, pipe_dacl_sddl};

#[cfg(windows)]
#[allow(unsafe_code)]
mod security {
    //! Win32 calls for the pipe descriptor and the pipe server's user.
    //! Every pointer the system allocates is freed with `LocalFree`, and
    //! every handle opened is closed, on every path.

    use std::ffi::c_void;
    use std::io;
    use std::iter::once;

    use interprocess::os::windows::security_descriptor::{
        AsSecurityDescriptorExt, BorrowedSecurityDescriptor, SecurityDescriptor,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, ConvertSidToStringSidW,
        ConvertStringSecurityDescriptorToSecurityDescriptorW, ConvertStringSidToSidW,
        GetSecurityInfo, SDDL_REVISION_1, SE_KERNEL_OBJECT,
    };
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, EqualSid, GetTokenInformation, PSECURITY_DESCRIPTOR, PSID,
        TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, READ_CONTROL,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(once(0)).collect()
    }

    /// Reads a NUL-terminated UTF-16 string the system allocated.
    ///
    /// # Safety
    /// `p` must point to a NUL-terminated UTF-16 string.
    unsafe fn from_wide(p: *const u16) -> String {
        let mut len = 0usize;
        // SAFETY: the caller guarantees a terminator.
        unsafe {
            while *p.add(len) != 0 {
                len += 1;
            }
            String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
        }
    }

    /// The `TOKEN_USER` of an open token, in u64 storage that keeps it
    /// aligned. Its SID points into the returned buffer.
    fn token_user(token: HANDLE) -> io::Result<Vec<u64>> {
        let mut len = 0u32;
        // SAFETY: a size query with a null buffer.
        unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len) };
        if len == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        // SAFETY: `buf` holds at least `len` bytes.
        let ok = unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buf.as_mut_ptr().cast::<c_void>(),
                len,
                &mut len,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(buf)
    }

    /// The user SID inside a buffer [`token_user`] filled.
    fn user_sid(buf: &[u64]) -> PSID {
        // SAFETY: GetTokenInformation filled a TOKEN_USER at the start of
        // `buf`, which is aligned for it.
        unsafe { &*buf.as_ptr().cast::<TOKEN_USER>() }.User.Sid
    }

    /// The current user's SID in string form (`S-1-5-21-...`).
    pub fn current_user_sid() -> io::Result<String> {
        let mut token: HANDLE = std::ptr::null_mut();
        // SAFETY: GetCurrentProcess returns a pseudo-handle; `token` is a
        // valid out pointer.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let result = (|| {
            let buf = token_user(token)?;
            let mut text: *mut u16 = std::ptr::null_mut();
            // SAFETY: the SID points into `buf`, alive for this call.
            if unsafe { ConvertSidToStringSidW(user_sid(&buf), &mut text) } == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: the system wrote a NUL-terminated string; freed below.
            let sid = unsafe { from_wide(text) };
            // SAFETY: allocated by ConvertSidToStringSidW with LocalAlloc.
            unsafe { LocalFree(text.cast::<c_void>()) };
            Ok(sid)
        })();
        // SAFETY: `token` was opened above.
        unsafe { CloseHandle(token) };
        result
    }

    /// Whether the process `pid` runs as the user whose SID, in string
    /// form, is `sid`: the user SID of its token and `sid` are equal
    /// (`EqualSid`). The process is opened for
    /// `PROCESS_QUERY_LIMITED_INFORMATION` and its token for `TOKEN_QUERY`.
    pub(super) fn process_runs_as(pid: u32, sid: &str) -> io::Result<bool> {
        let text = wide(sid);
        let mut expected: PSID = std::ptr::null_mut();
        // SAFETY: `text` is NUL-terminated; `expected` is a valid out pointer.
        if unsafe { ConvertStringSidToSidW(text.as_ptr(), &mut expected) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let result = (|| {
            // SAFETY: plain call; a null handle is checked below.
            let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if process.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut token: HANDLE = std::ptr::null_mut();
            // SAFETY: `process` is open; `token` is a valid out pointer.
            let opened = unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } != 0;
            let error = io::Error::last_os_error();
            // SAFETY: `process` was opened above and is not used again.
            unsafe { CloseHandle(process) };
            if !opened {
                return Err(error);
            }
            let user = token_user(token);
            // SAFETY: `token` was opened above and is not used again.
            unsafe { CloseHandle(token) };
            let buf = user?;
            // SAFETY: both SIDs are valid: one points into `buf`, the other
            // was allocated by the conversion above and is freed below.
            Ok(unsafe { EqualSid(user_sid(&buf), expected) } != 0)
        })();
        // SAFETY: allocated by ConvertStringSidToSidW with LocalAlloc.
        unsafe { LocalFree(expected) };
        result
    }

    /// A descriptor whose DACL grants `GENERIC_ALL` to the current user only.
    pub(super) fn user_only_descriptor() -> io::Result<SecurityDescriptor> {
        let sddl = wide(&format!("D:P(A;;GA;;;{})", current_user_sid()?));
        let mut raw: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: `sddl` is NUL-terminated; `raw` is a valid out pointer.
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut raw,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 || raw.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `raw` is a valid self-relative descriptor until LocalFree.
        let owned =
            unsafe { BorrowedSecurityDescriptor::from_ptr(raw as *const c_void) }.to_owned_sd();
        // SAFETY: allocated by the conversion with LocalAlloc.
        unsafe { LocalFree(raw) };
        owned
    }

    /// Opens the pipe `\\.\pipe\<name>` for `READ_CONTROL` and returns its
    /// DACL in SDDL form. Opening it is a client connection, which the
    /// daemon accepts and then sees closed.
    pub fn pipe_dacl_sddl(name: &str) -> io::Result<String> {
        let path = wide(&format!(r"\\.\pipe\{name}"));
        // SAFETY: `path` is NUL-terminated; no security attributes or template.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                READ_CONTROL,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let result = (|| {
            let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            // SAFETY: valid handle; only the descriptor out pointer is used.
            let err = unsafe {
                GetSecurityInfo(
                    handle,
                    SE_KERNEL_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut sd,
                )
            };
            if err != 0 {
                return Err(io::Error::from_raw_os_error(err as i32));
            }
            let mut text: *mut u16 = std::ptr::null_mut();
            // SAFETY: `sd` came from GetSecurityInfo.
            let ok = unsafe {
                ConvertSecurityDescriptorToStringSecurityDescriptorW(
                    sd,
                    SDDL_REVISION_1,
                    DACL_SECURITY_INFORMATION,
                    &mut text,
                    std::ptr::null_mut(),
                )
            };
            let out = if ok == 0 {
                Err(io::Error::last_os_error())
            } else {
                // SAFETY: NUL-terminated string from the conversion.
                let s = unsafe { from_wide(text) };
                // SAFETY: allocated with LocalAlloc.
                unsafe { LocalFree(text.cast::<c_void>()) };
                Ok(s)
            };
            // SAFETY: allocated by GetSecurityInfo with LocalAlloc.
            unsafe { LocalFree(sd) };
            out
        })();
        // SAFETY: opened above.
        unsafe { CloseHandle(handle) };
        result
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};

    use super::*;

    fn me() -> u32 {
        security::current_uid()
    }

    /// e-report decision 31: the bind for platforms without `fchmod` on a
    /// socket (macOS). It runs here by calling it directly.
    #[test]
    fn s3_a_socket_bound_in_a_private_folder_is_user_only_and_moved_into_place() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let path = tmp.path().join("daemon.sock");
        let bound = bind_private(&path).expect("bound");
        let meta = std::fs::symlink_metadata(&path).expect("the socket");
        assert!(meta.file_type().is_socket());
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        let names: Vec<_> = std::fs::read_dir(tmp.path())
            .expect("the folder")
            .map(|e| e.expect("an entry").file_name())
            .collect();
        assert_eq!(
            names,
            vec![std::ffi::OsString::from("daemon.sock")],
            "the private folder is gone"
        );
        let endpoint = Endpoint::UnixSocket(path.clone());
        let _client = connect(&endpoint, Duration::from_secs(2)).expect("connected");
        bound.accept().expect("a connection at the final path");
        drop(bound);
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "the moved socket is removed with the endpoint"
        );
    }

    /// e-report decision 32: a socket another user owns is refused by
    /// `connect` before any connection, and `bind` does not replace one.
    #[test]
    fn s3_a_socket_another_user_owns_is_refused_by_connect_and_bind() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let path = tmp.path().join("daemon.sock");
        let endpoint = Endpoint::UnixSocket(path.clone());
        let wait = Duration::from_millis(100);
        let every = Duration::from_millis(10);
        let other = me().wrapping_add(1);

        let bound = bind_as(&endpoint, wait, every, Some(me())).expect("bound by this user");
        let _ok = connect_as(&endpoint, Duration::from_secs(1), Some(me())).expect("connected");
        let e = connect_as(&endpoint, Duration::from_secs(1), Some(other))
            .expect_err("as another user, this socket is not the daemon's");
        assert_eq!(e.code, ErrorCode::DaemonUnavailable);
        assert!(e.message.contains("owned by another user"), "{}", e.message);
        assert!(
            e.next_action.contains("XDG_RUNTIME_DIR"),
            "{}",
            e.next_action
        );
        drop(bound);

        std::fs::write(&path, b"left over by another user").expect("plant a file");
        let e = match bind_as(&endpoint, wait, every, Some(other)) {
            Err(e) => e,
            Ok(_) => panic!("bound over another user's socket"),
        };
        assert_eq!(e.code, ErrorCode::DaemonUnavailable);
        assert!(e.message.contains("owned by another user"), "{}", e.message);
        assert!(
            e.next_action.contains("XDG_RUNTIME_DIR"),
            "{}",
            e.next_action
        );
        assert_eq!(
            std::fs::read(&path).expect("the file"),
            b"left over by another user",
            "the other user's file is left alone"
        );
        // This user's own leftover is replaced.
        let bound = bind_as(&endpoint, wait, every, Some(me())).expect("bound over a leftover");
        drop(bound);
    }

    /// The owner check reads the entry itself (`lstat`), never what a link
    /// points to.
    #[test]
    fn s3_the_owner_check_does_not_follow_a_link() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let link = tmp.path().join("daemon.sock");
        std::os::unix::fs::symlink(tmp.path().join("nowhere"), &link).expect("a link");
        assert!(check_owner(&link, me()).is_ok());
        let e = check_owner(&link, me().wrapping_add(1))
            .expect_err("the link is checked, not its missing target");
        assert!(e.message.contains("owned by another user"), "{}", e.message);
        assert!(check_owner(&tmp.path().join("absent"), me().wrapping_add(1)).is_ok());
    }

    /// B-95, stream H item 2: another user who binds the path and removes
    /// the entry after `accept` is not trusted. The connected stream is
    /// refused once its socket is gone.
    #[test]
    fn s3_a_socket_gone_after_connecting_is_refused() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let path = tmp.path().join("daemon.sock");
        let endpoint = Endpoint::UnixSocket(path.clone());
        let wait = Duration::from_millis(100);
        let every = Duration::from_millis(10);
        let bound = bind_as(&endpoint, wait, every, Some(me())).expect("bound");
        let stream = connect_as(&endpoint, Duration::from_secs(1), None).expect("connected");
        assert!(check_connected(&endpoint, &stream, &me()).is_ok());
        std::fs::remove_file(&path).expect("remove the entry");
        let e = check_connected(&endpoint, &stream, &me()).expect_err("the socket is gone");
        assert_eq!(e.code, ErrorCode::DaemonUnavailable);
        assert!(e.message.contains("gone after connecting"), "{}", e.message);
        assert!(
            e.next_action.contains("XDG_RUNTIME_DIR"),
            "{}",
            e.next_action
        );
        drop(bound);
    }

    /// B-95, stream H item 2: the peer's effective user id on the connected
    /// stream (`SO_PEERCRED` on Linux) is checked by the client and by the
    /// daemon. Both are run here against a user the peer is not.
    #[test]
    fn s3_a_peer_of_another_user_is_refused_by_client_and_daemon() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let path = tmp.path().join("daemon.sock");
        let endpoint = Endpoint::UnixSocket(path.clone());
        let wait = Duration::from_millis(100);
        let every = Duration::from_millis(10);
        let other = me().wrapping_add(1);
        let mut bound = bind_as(&endpoint, wait, every, Some(me())).expect("bound");

        // This user's client and daemon accept each other.
        let client = connect_as(&endpoint, Duration::from_secs(1), Some(me())).expect("connected");
        assert!(matches!(bound.accept(), Ok(Accepted::Client(_))));
        assert!(peer_is(&client, me()).is_ok());
        let why = peer_is(&client, other).expect_err("the daemon runs as this user");
        assert!(why.contains("another user"), "{why}");

        // A daemon serving only `other` refuses this user's client.
        bound.serves = Some(other);
        let _client = connect_as(&endpoint, Duration::from_secs(1), None).expect("connected");
        match bound.accept() {
            Ok(Accepted::Refused(why)) => assert!(why.contains("another user"), "{why}"),
            Ok(Accepted::Client(_)) => panic!("a client of another user was served"),
            Err(e) => panic!("accept failed: {e}"),
        }
    }

    /// B-112, stream H item 4: the fallback folder is created with mode
    /// 0700, even under a permissive umask's default.
    #[test]
    fn s3_the_fallback_socket_folder_is_created_user_only() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let dir = tmp.path().join("spatial-memory-0123456789abcdef");
        let endpoint = Endpoint::UnixSocket(dir.join("daemon.sock"));
        let wait = Duration::from_millis(100);
        let every = Duration::from_millis(10);
        let bound = bind_as(&endpoint, wait, every, Some(me())).expect("bound");
        let m = std::fs::symlink_metadata(&dir).expect("the folder");
        assert!(m.is_dir());
        assert_eq!(m.mode() & 0o777, 0o700, "mode {:o}", m.mode());
        assert_eq!(m.uid(), me());
        let _client = connect_as(&endpoint, Duration::from_secs(1), Some(me())).expect("connected");
        drop(bound);
    }

    /// B-112, stream H item 4: a fallback folder that group or others can
    /// write to, that another user owns or that is a link is refused by
    /// `bind` and by `connect`, naming the folder, with a next action.
    #[test]
    fn s3_a_fallback_folder_that_is_not_private_is_refused_by_bind_and_connect() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let dir = tmp.path().join("spatial-memory-fedcba9876543210");
        let endpoint = Endpoint::UnixSocket(dir.join("daemon.sock"));
        let wait = Duration::from_millis(100);
        let every = Duration::from_millis(10);
        let other = me().wrapping_add(1);
        let check = |e: CoreError, what: &str| {
            assert_eq!(e.code, ErrorCode::DaemonUnavailable);
            assert!(
                e.message.contains(&dir.display().to_string()),
                "{}",
                e.message
            );
            assert!(e.message.contains(what), "{}", e.message);
            assert!(
                e.next_action.contains("XDG_RUNTIME_DIR"),
                "{}",
                e.next_action
            );
        };

        // Writable by group and others.
        std::fs::create_dir(&dir).expect("the folder");
        for mode in [0o777, 0o720, 0o702] {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).expect("chmod");
            match bind_as(&endpoint, wait, every, Some(me())) {
                Err(e) => check(e, "can write"),
                Ok(_) => panic!("bound in a folder of mode {mode:o}"),
            }
            match connect_as(&endpoint, Duration::from_millis(100), Some(me())) {
                Err(e) => check(e, "can write"),
                Ok(_) => panic!("connected through a folder of mode {mode:o}"),
            }
        }

        // Private, but owned by another user (seen from `other`).
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        match bind_as(&endpoint, wait, every, Some(other)) {
            Err(e) => check(e, "owned by another user"),
            Ok(_) => panic!("bound in another user's folder"),
        }
        let bound = bind_as(&endpoint, wait, every, Some(me())).expect("this user's folder");
        match connect_as(&endpoint, Duration::from_millis(100), Some(other)) {
            Err(e) => check(e, "owned by another user"),
            Ok(_) => panic!("connected through another user's folder"),
        }
        drop(bound);

        // A link where the folder should be, even to a private folder.
        let _ = std::fs::remove_file(dir.join("daemon.sock"));
        std::fs::remove_dir(&dir).expect("remove the folder");
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).expect("the target");
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        std::os::unix::fs::symlink(&real, &dir).expect("a link");
        match bind_as(&endpoint, wait, every, Some(me())) {
            Err(e) => check(e, "not a folder"),
            Ok(_) => panic!("bound through a link"),
        }
        match connect_as(&endpoint, Duration::from_millis(100), Some(me())) {
            Err(e) => check(e, "not a folder"),
            Ok(_) => panic!("connected through a link"),
        }
    }

    /// The mode rule is for the fallback folder only: a profile folder its
    /// group can write to (umask 002) still binds, and the peer check holds.
    #[test]
    fn s3_a_group_writable_profile_folder_still_binds() {
        let tmp = tempfile::tempdir().expect("a temporary folder");
        let dir = tmp.path().join("profile");
        std::fs::create_dir(&dir).expect("the folder");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o775)).expect("chmod");
        let endpoint = Endpoint::UnixSocket(dir.join("daemon.sock"));
        let bound = bind_as(
            &endpoint,
            Duration::from_millis(100),
            Duration::from_millis(10),
            Some(me()),
        )
        .expect("bound");
        let _client = connect_as(&endpoint, Duration::from_secs(1), Some(me())).expect("connected");
        drop(bound);
    }

    /// Stream H item 11: the private folder's name is unpredictable, and a
    /// taken name is skipped, not reused.
    #[test]
    fn s3_the_private_bind_folder_name_is_unpredictable() {
        let names: std::collections::HashSet<String> = (0..64).map(|_| private_name(0)).collect();
        assert!(names.len() > 60, "{} distinct of 64", names.len());
        for n in &names {
            assert_eq!(n.len(), ".".len() + 8, "{n}");
            assert!(n.starts_with('.'), "{n}");
            assert!(n[1..].bytes().all(|b| b.is_ascii_hexdigit()), "{n}");
        }
        assert!(".".len() + 8 + "//s".len() <= "/daemon.sock".len());
    }

    #[test]
    fn s3_only_the_hashed_folder_is_the_fallback_folder() {
        assert!(is_fallback_folder(Path::new(
            "/tmp/spatial-memory-0123456789abcdef"
        )));
        assert!(!is_fallback_folder(Path::new("/tmp/spatial-memory-0123")));
        assert!(!is_fallback_folder(Path::new(
            "/tmp/spatial-memory-0123456789abcdeg"
        )));
        assert!(!is_fallback_folder(Path::new("/home/u/.spatial-memory")));
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn unique_pipe() -> Endpoint {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        Endpoint::NamedPipe(format!(
            "sm-endpoint-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn bound(endpoint: &Endpoint) -> Bound {
        bind_as(
            endpoint,
            Duration::from_millis(500),
            Duration::from_millis(10),
            None,
        )
        .expect("bound")
    }

    /// Stream H item 3, the Windows counterpart of the Unix peer checks: the
    /// client checks the user of the process that serves the pipe. This
    /// user's daemon passes.
    #[test]
    fn spike1_connect_accepts_a_pipe_this_user_serves() {
        let endpoint = unique_pipe();
        let bound = bound(&endpoint);
        let me = current_user_sid().expect("the current SID");
        let stream = connect_as(&endpoint, Duration::from_secs(2), Some(me.clone()))
            .expect("served by this user");
        assert!(check_connected(&endpoint, &stream, &me).is_ok());
        assert!(security::process_runs_as(std::process::id(), &me).expect("this process"));
        drop(bound);
    }

    /// Stream H item 3: a pipe served by a process of another user is
    /// refused, before anything is sent. Seen here as a client that expects
    /// the daemon to run as `SYSTEM` (S-1-5-18).
    #[test]
    fn spike1_connect_refuses_a_pipe_another_user_serves() {
        let endpoint = unique_pipe();
        let bound = bound(&endpoint);
        let system = "S-1-5-18".to_owned();
        let e = connect_as(&endpoint, Duration::from_secs(2), Some(system.clone()))
            .expect_err("this process does not run as SYSTEM");
        assert_eq!(e.code, ErrorCode::DaemonUnavailable);
        assert!(e.message.contains("another user"), "{}", e.message);
        assert!(e.next_action.contains("pipe"), "{}", e.next_action);
        assert!(!security::process_runs_as(std::process::id(), &system).expect("this process"));
        // A malformed SID is an error, not a match.
        assert!(security::process_runs_as(std::process::id(), "not a SID").is_err());
        drop(bound);
    }
}
