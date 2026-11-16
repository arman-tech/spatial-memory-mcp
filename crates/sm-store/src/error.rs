//! Turning SQLite errors into `sm_core` errors.

use rusqlite::ErrorCode as SqliteCode;
use sm_core::error::{CoreError, ErrorCode};

use crate::conn::BUSY_TIMEOUT;

/// Adds what the store was doing to a `rusqlite` error. The orphan rule rules
/// out a `From` impl, so every call site names its step instead.
pub(crate) trait Ctx<T> {
    fn ctx(self, what: &str) -> Result<T, CoreError>;
}

impl<T> Ctx<T> for rusqlite::Result<T> {
    fn ctx(self, what: &str) -> Result<T, CoreError> {
        self.map_err(|e| sqlite_error(what, &e))
    }
}

/// A lock that outlived the busy timeout is `timeout`; everything else from
/// SQLite is `storage`.
pub(crate) fn sqlite_error(what: &str, e: &rusqlite::Error) -> CoreError {
    let busy = matches!(
        e.sqlite_error_code(),
        Some(SqliteCode::DatabaseBusy | SqliteCode::DatabaseLocked)
    );
    if busy {
        CoreError::new(
            ErrorCode::Timeout,
            format!(
                "{what}: the database stayed locked for {} seconds: {e}",
                BUSY_TIMEOUT.as_secs()
            ),
        )
        .with_next_action(
            "Retry; if it keeps happening, another process is holding the database \
             (run `spatial-memory doctor`).",
        )
    } else {
        CoreError::storage(format!("{what}: {e}"))
    }
}

/// A value read back from the database that this build cannot interpret.
pub(crate) fn corrupt(what: &str, detail: impl std::fmt::Display) -> CoreError {
    CoreError::storage(format!("{what}: unreadable stored value: {detail}"))
}

/// Wraps a conversion error so it can leave a `rusqlite` row closure.
pub(crate) fn conversion(idx: usize, e: CoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(idx, rusqlite::types::Type::Text, Box::new(e))
}
