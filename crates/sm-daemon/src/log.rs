//! `daemon.log`: one line per event, each written with one write call
//! (CLAUDE.md, Windows is first-class), bounded by rotation to
//! `daemon.log.1` once it reaches `log_max_bytes`.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use sm_core::time::Timestamp;

/// Longest message kept on one line; longer ones are cut.
const MAX_MESSAGE_CHARS: usize = 2000;

#[derive(Debug)]
pub struct DaemonLog {
    path: PathBuf,
    max_bytes: u64,
    lock: Mutex<()>,
}

impl DaemonLog {
    pub fn new(path: PathBuf, max_bytes: u64) -> Self {
        DaemonLog {
            path,
            max_bytes: max_bytes.max(1024),
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Appends one line. Failures to log are ignored: logging never fails a
    /// request.
    pub fn line(&self, message: impl AsRef<str>) {
        let mut text: String = message
            .as_ref()
            .chars()
            .take(MAX_MESSAGE_CHARS)
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .collect();
        text.insert(0, ' ');
        text.insert_str(0, &Timestamp::now().to_rfc3339());
        text.push('\n');
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(meta) = std::fs::metadata(&self.path)
            && meta.len().saturating_add(text.len() as u64) > self.max_bytes
        {
            let rotated = self.path.with_extension("log.1");
            let _ = std::fs::remove_file(&rotated);
            let _ = std::fs::rename(&self.path, &rotated);
        }
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = f.write_all(text.as_bytes());
        }
    }
}
