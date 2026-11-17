//! The daemon side of the auto-start lock (CONTRACTS 4, Auto-start).
//!
//! `daemon.lock` is opened create + write without truncation and never
//! written: its content means nothing, only the lock does. The daemon tries
//! it every `retry_every` for up to `lock_wait`; if another process holds it
//! that process is the daemon (or is about to be), and this one exits 0.
//! The lock is held for the daemon's life and released last, after the
//! endpoint is closed.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::time::{Duration, Instant};

use sm_core::error::{CoreError, Result};

/// An exclusive hold on `daemon.lock`. Dropping it releases the lock.
#[derive(Debug)]
pub struct DaemonLock {
    file: File,
}

impl DaemonLock {
    /// `Ok(Some)` when acquired, `Ok(None)` when another process holds it
    /// for the whole wait.
    pub fn acquire(path: &Path, wait: Duration, every: Duration) -> Result<Option<DaemonLock>> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|e| {
                CoreError::storage(format!("cannot open the lock file {}: {e}", path.display()))
            })?;
        let start = Instant::now();
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Some(DaemonLock { file })),
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Error(e)) => {
                    return Err(CoreError::storage(format!(
                        "cannot lock {}: {e}",
                        path.display()
                    )));
                }
            }
            if start.elapsed() >= wait {
                return Ok(None);
            }
            std::thread::sleep(every);
        }
    }

    /// Releases the lock explicitly (it is also released on drop).
    pub fn release(self) {
        let _ = self.file.unlock();
    }
}
