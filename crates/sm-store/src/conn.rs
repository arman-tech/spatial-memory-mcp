//! Opening a connection with the store's settings.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};
use sm_core::error::{CoreError, Result};

use crate::error::Ctx;

/// How long a connection waits for a lock before SQLite gives up. Long
/// enough that a bulk write, a checkpoint or another process's migration
/// never surfaces as a lock error (AT-5).
pub(crate) const BUSY_TIMEOUT: Duration = Duration::from_secs(30);

/// How much of the database file a reader connection reads through a memory
/// map (decision B-75): SQLite's ceiling, `SQLITE_MAX_MMAP_SIZE`
/// (0x7fff0000 bytes, just under 2 GiB), which covers a store of about
/// 250,000 chunks. Pages past it are read as before.
///
/// Project scope reads every chunk row of the project; at 100,000 chunks
/// that is one 4 KiB page per chunk, and through SQLite's page cache each is
/// a `ReadFile` or `read` call and a copy. Mapped, a page is a pointer into
/// the operating system's file cache, shared by every reader.
///
/// Only readers map the file; the writer writes through SQLite's page cache
/// as before. A checkpoint sets the main file's length to the database's
/// size, which shrinks it only after `VACUUM` or auto-vacuum, neither of
/// which the store runs; on Windows, SQLite treats a length change refused
/// because another connection has the file mapped (`ERROR_USER_MAPPED_FILE`)
/// as done.
pub(crate) const READER_MMAP_BYTES: i64 = 0x7fff_0000;

/// Opens (creating if needed) the database at `path` and applies the
/// per-connection settings. `vec1` must already be registered
/// (`crate::vec1::register`), so the connection has the module.
pub(crate) fn open(path: &Path) -> Result<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)
        .ctx(&format!("opening the database {}", path.display()))?;
    configure(&conn)?;
    Ok(conn)
}

/// [`open`] for a reader connection: the same settings, and the file read
/// through a memory map of up to [`READER_MMAP_BYTES`].
pub(crate) fn open_reader(path: &Path) -> Result<Connection> {
    let conn = open(path)?;
    conn.pragma_update_and_check(None, "mmap_size", READER_MMAP_BYTES, |r| r.get::<_, i64>(0))
        .ctx("mapping the database for reads")?;
    Ok(conn)
}

fn configure(conn: &Connection) -> Result<()> {
    // First, so that switching to WAL below can wait for another process.
    conn.busy_timeout(BUSY_TIMEOUT)
        .ctx("setting the busy timeout")?;
    let mode: String = conn
        .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
        .ctx("switching to WAL")?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(CoreError::storage(format!(
            "the database refused WAL mode (journal_mode is {mode})"
        )));
    }
    conn.pragma_update(None, "foreign_keys", true)
        .ctx("turning on foreign keys")?;
    // FULL syncs the WAL on every commit: a committed memory survives a power
    // cut. The extra fsync per write is small next to embedding.
    conn.pragma_update(None, "synchronous", "FULL")
        .ctx("setting synchronous")?;
    let fk: i64 = conn
        .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
        .ctx("reading foreign_keys")?;
    if fk != 1 {
        return Err(CoreError::storage("foreign keys could not be enabled"));
    }
    // The extension is registered as an automatic extension; confirm it.
    conn.query_row("SELECT vec1_info()", [], |r| r.get::<_, String>(0))
        .ctx("checking the vec1 extension")?;
    Ok(())
}
