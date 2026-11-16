//! The capture spool (SPEC section 16, Hand-off; AT-15).
//!
//! When the daemon cannot be reached, a capture hook writes its request to
//! the spool instead and exits 0. The file is written whole into
//! `spool/tmp/` and then renamed into `spool/new/`, so the daemon never sees
//! a partial file. At start the daemon drains `spool/new/` in name order,
//! which is arrival order, deleting each file once the work is queued in the
//! store. A file it cannot read is moved to `spool/failed/` and logged.
//!
//! Each file holds one [`Request`] whose call is [`Op::Capture`], as JSON
//! without a frame prefix. There is no count or size cap (decision B-33):
//! the spool only grows while the daemon is down, and every file is a
//! hook payload the agent already produced.
//!
//! Files are not synced to disk before the rename (decision B-33): a power
//! loss can lose the last captures, which costs less than a sync inside a
//! hook that must exit within 200 ms.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime};

use crate::error::{CoreError, Result};
use crate::protocol::{Op, Request, decode_body};
use crate::time::Timestamp;

/// Extension of a finished spool file.
pub const SPOOL_EXT: &str = "json";

/// A file left in `tmp/` this long is from a writer that died before its
/// rename; the daemon removes it at start.
pub const STALE_TMP_AGE: Duration = Duration::from_secs(60 * 60);

static SEQ: AtomicU32 = AtomicU32::new(0);

/// `{microseconds:020}-{pid}-{seq:08x}.json`: sorts by time, and is unique
/// across processes (pid) and within one (sequence).
pub fn spool_file_name(at: Timestamp) -> String {
    let micros = at.as_micros().max(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{micros:020}-{}-{seq:08x}.{SPOOL_EXT}", std::process::id())
}

/// Creates a spool folder, user-only on Unix.
pub fn create_spool_dir(dir: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(dir)
        .map_err(|e| CoreError::storage(format!("cannot create {}: {e}", dir.display())))
}

/// Writes a capture request to the spool: whole into `tmp_dir` in one write
/// call (spike result 3), then renamed into `new_dir`. Returns the final path.
pub fn write_spool(tmp_dir: &Path, new_dir: &Path, request: &Request) -> Result<PathBuf> {
    if !matches!(request.call, Op::Capture(_)) {
        return Err(CoreError::internal(format!(
            "only capture requests are spooled, not {}",
            request.call.name()
        )));
    }
    let body = serde_json::to_vec(request)
        .map_err(|e| CoreError::internal(format!("cannot encode spool file: {e}")))?;
    create_spool_dir(tmp_dir)?;
    create_spool_dir(new_dir)?;
    let name = spool_file_name(Timestamp::now());
    let tmp = tmp_dir.join(&name);
    let dst = new_dir.join(&name);
    let write = || -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(&body)?;
        drop(f);
        fs::rename(&tmp, &dst)
    };
    write().map_err(|e| {
        let _ = fs::remove_file(&tmp);
        CoreError::storage(format!("cannot write spool file {}: {e}", dst.display()))
    })?;
    Ok(dst)
}

/// Finished spool files in `new_dir`, oldest first. A missing folder is an
/// empty spool.
pub fn list_spool(new_dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(new_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(CoreError::storage(format!(
                "cannot list {}: {e}",
                new_dir.display()
            )));
        }
    };
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| CoreError::storage(format!("cannot list {}: {e}", new_dir.display())))?;
        let path = entry.path();
        let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if is_file && path.extension().is_some_and(|x| x == SPOOL_EXT) {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(files)
}

/// Reads one spool file. A file that is not a capture request is
/// `invalid_input`; the daemon moves it to `failed/` with [`quarantine`].
pub fn read_spool(path: &Path) -> Result<Request> {
    let body = fs::read(path)
        .map_err(|e| CoreError::storage(format!("cannot read {}: {e}", path.display())))?;
    let request: Request = decode_body(&body)
        .map_err(|e| CoreError::invalid_input(format!("{}: {}", path.display(), e.message)))?;
    if !matches!(request.call, Op::Capture(_)) {
        return Err(CoreError::invalid_input(format!(
            "{}: spool files hold capture requests, not {}",
            path.display(),
            request.call.name()
        )));
    }
    Ok(request)
}

/// Moves an unreadable spool file into `failed_dir`, keeping its name.
pub fn quarantine(path: &Path, failed_dir: &Path) -> Result<PathBuf> {
    create_spool_dir(failed_dir)?;
    let name = path
        .file_name()
        .ok_or_else(|| CoreError::internal(format!("{} has no file name", path.display())))?;
    let dst = failed_dir.join(name);
    fs::rename(path, &dst).map_err(|e| {
        CoreError::storage(format!(
            "cannot move {} to {}: {e}",
            path.display(),
            dst.display()
        ))
    })?;
    Ok(dst)
}

/// Removes files in `tmp_dir` last modified before `now - max_age`: writes
/// that died before their rename. Returns how many were removed.
pub fn remove_stale_tmp(tmp_dir: &Path, now: SystemTime, max_age: Duration) -> Result<usize> {
    let entries = match fs::read_dir(tmp_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(CoreError::storage(format!(
                "cannot list {}: {e}",
                tmp_dir.display()
            )));
        }
    };
    let cutoff = now.checked_sub(max_age).unwrap_or(SystemTime::UNIX_EPOCH);
    let mut removed = 0;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        let old = meta.modified().is_ok_and(|m| m < cutoff);
        if meta.is_file() && old && fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProfileName;
    use crate::paths::ProfilePaths;
    use crate::protocol::{CaptureParams, HookEvent};

    fn capture(n: u64) -> Request {
        Request::new(
            "/work/repo",
            ProfileName::default_profile(),
            Op::Capture(CaptureParams {
                event: HookEvent::PostToolUse,
                payload: serde_json::json!({"tool_name": "Edit", "n": n}),
                received_at: Timestamp::now(),
            }),
        )
        .with_session(Some("s-1".into()))
    }

    #[test]
    fn s16_spool_write_is_tmp_then_rename() {
        let dir = tempfile::tempdir().unwrap();
        let paths = ProfilePaths::new(dir.path().to_path_buf(), ProfileName::default_profile());
        let req = capture(1);
        let path = write_spool(&paths.spool_tmp(), &paths.spool_new(), &req).unwrap();
        assert_eq!(path.parent().unwrap(), paths.spool_new());
        assert!(
            fs::read_dir(paths.spool_tmp()).unwrap().next().is_none(),
            "tmp must be empty"
        );
        assert_eq!(read_spool(&path).unwrap(), req);

        // Only capture requests are spooled or read back.
        let hello = Request::new("/x", ProfileName::default_profile(), Op::Hello);
        assert!(write_spool(&paths.spool_tmp(), &paths.spool_new(), &hello).is_err());
        let bad = paths
            .spool_new()
            .join("00000000000000000000-1-00000000.json");
        fs::write(&bad, serde_json::to_vec(&hello).unwrap()).unwrap();
        assert!(read_spool(&bad).is_err());
        let moved = quarantine(&bad, &paths.spool_failed()).unwrap();
        assert!(moved.exists() && !bad.exists());
    }

    #[test]
    fn s16_spool_lists_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("tmp");
        let new = dir.path().join("new");
        assert!(
            list_spool(&new).unwrap().is_empty(),
            "missing folder is an empty spool"
        );
        let mut written = Vec::new();
        for n in 0..5 {
            written.push(write_spool(&tmp, &new, &capture(n)).unwrap());
        }
        fs::write(new.join("notes.txt"), b"ignored").unwrap();
        let listed = list_spool(&new).unwrap();
        assert_eq!(listed, written);
        let ns: Vec<u64> = listed
            .iter()
            .map(|p| match read_spool(p).unwrap().call {
                Op::Capture(c) => c.payload["n"].as_u64().unwrap(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(ns, [0, 1, 2, 3, 4]);

        // Names sort by time, then sequence.
        let a = spool_file_name(Timestamp::from_micros(5));
        let b = spool_file_name(Timestamp::from_micros(5));
        let c = spool_file_name(Timestamp::from_micros(40));
        assert!(a < b && b < c, "{a} {b} {c}");
    }

    #[test]
    fn s16_spool_removes_only_stale_tmp_files() {
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("tmp");
        create_spool_dir(&tmp).unwrap();
        fs::write(tmp.join("a.json"), b"{").unwrap();
        let now = SystemTime::now();
        assert_eq!(remove_stale_tmp(&tmp, now, STALE_TMP_AGE).unwrap(), 0);
        let later = now + STALE_TMP_AGE + Duration::from_secs(1);
        assert_eq!(remove_stale_tmp(&tmp, later, STALE_TMP_AGE).unwrap(), 1);
        assert_eq!(
            remove_stale_tmp(&dir.path().join("missing"), now, STALE_TMP_AGE).unwrap(),
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn s16_spool_dirs_are_user_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("spool").join("tmp");
        create_spool_dir(&tmp).unwrap();
        let mode = fs::metadata(&tmp).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }
}
