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
