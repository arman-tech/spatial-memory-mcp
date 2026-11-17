//! The local endpoint (CONTRACTS 3.1 and 4): a Unix domain socket, or a
//! named pipe on Windows, carrying length-prefixed JSON frames.
//!
//! Unix: a leftover socket file is removed before binding (the caller holds
//! `daemon.lock`, so no live daemon owns it) and the socket is created with
//! mode 0600.
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
        Endpoint::UnixSocket(p) => p
            .as_path()
            .to_fs_name::<interprocess::local_socket::GenericFilePath>(),
    }
}

fn unavailable(endpoint: &Endpoint, e: &io::Error) -> CoreError {
    CoreError::new(
        ErrorCode::DaemonUnavailable,
        format!("cannot reach {}: {e}", endpoint.display_path()),
    )
}

/// Binds the endpoint, retrying every `every` for up to `wait`.
pub(crate) fn bind(endpoint: &Endpoint, wait: Duration, every: Duration) -> Result<Listener> {
    if let Endpoint::UnixSocket(path) = endpoint {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Leftover from a daemon that did not exit cleanly.
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

fn try_bind(endpoint: &Endpoint) -> io::Result<Listener> {
    let opts = ListenerOptions::new()
        .name(name_of(endpoint)?)
        .nonblocking(ListenerNonblockingMode::Neither)
        .reclaim_name(true)
        .try_overwrite(false);
    #[cfg(windows)]
    let opts = {
        use interprocess::os::windows::local_socket::ListenerOptionsExt;
        opts.security_descriptor(security::user_only_descriptor()?)
    };
    #[cfg(unix)]
    let opts = {
        use interprocess::os::unix::local_socket::ListenerOptionsExt;
        opts.mode(0o600)
    };
    opts.create_sync()
}

/// Connects to a daemon, waiting up to `timeout` for a free pipe instance.
pub fn connect(endpoint: &Endpoint, timeout: Duration) -> Result<Stream> {
    let name = name_of(endpoint).map_err(|e| unavailable(endpoint, &e))?;
    let stream = ConnectOptions::new()
        .name(name)
        .wait_mode(ConnectWaitMode::Timeout(timeout))
        .connect_sync()
        .map_err(|e| unavailable(endpoint, &e))?;
    Ok(stream)
}

/// Accepts one connection.
pub(crate) fn accept(listener: &Listener) -> io::Result<Stream> {
    listener.accept()
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
