//! The local endpoint (CONTRACTS 3.1 and 4): a Unix domain socket, or a
//! named pipe on Windows, carrying length-prefixed JSON frames.
//!
//! Unix (SPEC 3 rule 1; e-report decisions 31 and 32):
//! - **Owner.** The socket must belong to the current (effective) user,
//!   checked with `lstat`, which does not follow a link. `connect` refuses
//!   a socket another user owns, before connecting and again once
//!   connected (an entry missing at the first check may have been created
//!   since), and `bind` refuses to replace one, each with a next action. This matters for the fallback socket in `/tmp`
//!   (`sm_core::paths::unix_socket_path`, used when the profile path is
//!   too long and `XDG_RUNTIME_DIR` is unset): its name is predictable, but
//!   `/tmp` is sticky, so once the check passes nobody else can swap the
//!   entry.
//! - **Mode.** A leftover socket of this user is removed before binding
//!   (the caller holds `daemon.lock`, so no live daemon owns it) and the
//!   socket is created with mode 0600: interprocess runs `fchmod` on the
//!   socket before `bind`. Where that is unsupported (interprocess 2.4.8
//!   documents it for Linux, FreeBSD 14.3 and later and OpenBSD, and maps
//!   `EINVAL` to `Unsupported`; macOS is expected to be one), the socket is
//!   bound in a new folder that only this user can enter, set to 0600 and
//!   then renamed into place, so no other user can connect before its mode
//!   is set. That path is unverified on macOS (no machine); it runs in the
//!   Linux tests by calling it directly.
//!
//! Windows: the pipe is created with `FILE_FLAG_FIRST_PIPE_INSTANCE`, which
//! interprocess adds on the first instance itself, so a second daemon cannot
//! create an instance of a pipe another process owns; remote clients are
//! rejected (interprocess's default `PIPE_REJECT_REMOTE_CLIENTS`). The
//! descriptor grants `GENERIC_ALL` to the current user only and is
//! protected from inheritance: SDDL `D:P(A;;GA;;;<user SID>)`, built with
//! `ConvertStringSecurityDescriptorToSecurityDescriptorW`, copied into an
//! owned interprocess `SecurityDescriptor` and passed through
//! `ListenerOptionsExt::security_descriptor`.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use interprocess::ConnectWaitMode;
use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{ConnectOptions, ListenerNonblockingMode, ListenerOptions, Name};
use interprocess::local_socket::{Listener, Stream};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::paths::Endpoint;

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
}

impl Bound {
    /// Accepts one connection.
    pub(crate) fn accept(&self) -> io::Result<Stream> {
        self.listener.accept()
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
    #[cfg(unix)]
    let me = Some(security::current_uid());
    #[cfg(not(unix))]
    let me = None;
    bind_as(endpoint, wait, every, me)
}

/// [`bind`] for the user `me` (the owner check runs on Unix only).
fn bind_as(endpoint: &Endpoint, wait: Duration, every: Duration, me: Option<u32>) -> Result<Bound> {
    if let Endpoint::UnixSocket(path) = endpoint {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
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
            Ok(l) => return Ok(l),
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
    Ok(Bound {
        listener,
        moved: None,
    })
}

#[cfg(unix)]
fn try_bind(endpoint: &Endpoint) -> io::Result<Bound> {
    use interprocess::os::unix::local_socket::ListenerOptionsExt;
    match options(name_of(endpoint)?).mode(0o600).create_sync() {
        Ok(listener) => Ok(Bound {
            listener,
            moved: None,
        }),
        Err(e) if e.kind() == io::ErrorKind::Unsupported => match endpoint {
            Endpoint::UnixSocket(path) => bind_private(path),
            Endpoint::NamedPipe(_) => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// Binds `path` where `fchmod` before `bind` is unsupported: in a new
/// folder only this user can enter (mode 0700, created here, never one
/// that existed), sets the socket to 0600, then renames it to `path`.
/// Connecting needs search permission on every folder of the path, so
/// nobody else can reach the socket before its mode is set. The folder
/// name is short (`.<pid in hex><n>/s` is no longer than `/daemon.sock`),
/// so the path stays within the `sun_path` limit the final path meets.
#[cfg(unix)]
fn bind_private(path: &Path) -> io::Result<Bound> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let pid = std::process::id();
    let mut dir = None;
    for n in 0..10u32 {
        let candidate = parent.join(format!(".{pid:x}{n}"));
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
                "no free private folder to bind in under {}",
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
        Ok(Bound {
            listener,
            moved: Some(path.to_path_buf()),
        })
    })();
    // Gone after the rename; left only when a step failed.
    let _ = std::fs::remove_file(&temp);
    let _ = std::fs::remove_dir(&dir);
    bound
}

/// Refuses an entry at `path` that another user owns, not following a
/// link. A missing entry passes.
fn check_owner(path: &Path, me: u32) -> Result<()> {
    #[cfg(unix)]
    {
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
    #[cfg(not(unix))]
    {
        let _ = (path, me);
        Ok(())
    }
}

/// Connects to a daemon, waiting up to `timeout` for a free pipe instance.
/// On Unix, a socket another user owns is refused before connecting, and
/// checked again once connected, before anything is sent.
pub fn connect(endpoint: &Endpoint, timeout: Duration) -> Result<Stream> {
    #[cfg(unix)]
    let me = Some(security::current_uid());
    #[cfg(not(unix))]
    let me = None;
    connect_as(endpoint, timeout, me)
}

/// [`connect`] for the user `me` (the owner check runs on Unix only).
fn connect_as(endpoint: &Endpoint, timeout: Duration, me: Option<u32>) -> Result<Stream> {
    let owned = || match (endpoint, me) {
        (Endpoint::UnixSocket(path), Some(me)) => check_owner(path, me).map_err(|e| {
            e.with_next_action(
                "Do not use this socket: another user may be posing as the daemon. Set \
                 XDG_RUNTIME_DIR to a folder only you can enter, or use a shorter profile \
                 path so the socket lives in the profile folder, then retry.",
            )
        }),
        _ => Ok(()),
    };
    owned()?;
    let name = name_of(endpoint).map_err(|e| unavailable(endpoint, &e))?;
    let stream = ConnectOptions::new()
        .name(name)
        .wait_mode(ConnectWaitMode::Timeout(timeout))
        .connect_sync()
        .map_err(|e| unavailable(endpoint, &e))?;
    // Again, before anything is sent: a socket missing at the first check
    // may have been created by another user since. In a sticky folder only
    // its owner can remove it, so it is still theirs here (e-report
    // decision 32). The stream is dropped unused on a refusal.
    owned()?;
    Ok(stream)
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
    //! Win32 calls for the pipe descriptor. Every pointer the system
    //! allocates is freed with `LocalFree` on every path.

    use std::ffi::c_void;
    use std::io;
    use std::iter::once;

    use interprocess::os::windows::security_descriptor::{
        AsSecurityDescriptorExt, BorrowedSecurityDescriptor, SecurityDescriptor,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, ConvertSidToStringSidW,
        ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo, SDDL_REVISION_1,
        SE_KERNEL_OBJECT,
    };
    use windows_sys::Win32::Security::{
        DACL_SECURITY_INFORMATION, GetTokenInformation, PSECURITY_DESCRIPTOR, TOKEN_QUERY,
        TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, READ_CONTROL,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

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

    /// The current user's SID in string form (`S-1-5-21-...`).
    pub fn current_user_sid() -> io::Result<String> {
        let mut token: HANDLE = std::ptr::null_mut();
        // SAFETY: GetCurrentProcess returns a pseudo-handle; `token` is a
        // valid out pointer.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let result = (|| {
            let mut len = 0u32;
            // SAFETY: a size query with a null buffer.
            unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len) };
            if len == 0 {
                return Err(io::Error::last_os_error());
            }
            // u64 storage keeps TOKEN_USER aligned.
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
            // SAFETY: the call filled a TOKEN_USER at the start of `buf`.
            let user = unsafe { &*buf.as_ptr().cast::<TOKEN_USER>() };
            let mut text: *mut u16 = std::ptr::null_mut();
            // SAFETY: the SID points into `buf`, alive for this call.
            if unsafe { ConvertSidToStringSidW(user.User.Sid, &mut text) } == 0 {
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
}
