//! Injection records: which memories each session has been shown, for the
//! no-repeat rule (SPEC section 4, `injections`; section 6, No repeats).

use sm_core::error::{CoreError, Result};
use sm_core::ids::MemoryId;
use sm_core::time::Timestamp;

use crate::Store;
use crate::codec::{memory_id_col, ts};
use crate::error::Ctx;

/// Injection rows older than this are pruned (SPEC section 4).
pub const INJECTION_RETENTION_DAYS: i64 = 30;

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

impl Store {
    /// Records that `session` was shown `ids` at `at` and returns how many
    /// rows were written. A memory shown again keeps the later time;
    /// memories that no longer exist are skipped.
    pub fn record_injections(&self, session: &str, ids: &[MemoryId], at: Timestamp) -> Result<u64> {
        if session.trim().is_empty() {
            return Err(CoreError::invalid_input("an injection needs a session id"));
        }
        let at = ts(at);
        self.write("recording injections", |tx| {
            let mut s = tx
                .prepare_cached(
                    "INSERT INTO injections(session_id, memory_id, at)
                     SELECT ?1, ?2, ?3 WHERE EXISTS (SELECT 1 FROM memories WHERE id = ?2)
                     ON CONFLICT(session_id, memory_id) DO UPDATE SET at = max(at, excluded.at)",
                )
                .ctx("recording an injection")?;
            let mut n = 0u64;
            for id in ids {
                n += u64::from(
                    s.execute((session, id.to_string(), at))
                        .ctx("recording an injection")?
                        > 0,
                );
            }
            Ok(n)
        })
    }

    /// The memories `session` has been shown, by id.
    pub fn injected(&self, session: &str) -> Result<Vec<MemoryId>> {
        self.read("reading injections", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT memory_id FROM injections WHERE session_id = ?1 ORDER BY memory_id",
                )
                .ctx("reading injections")?;
            let rows = s
                .query_map([session], |r| memory_id_col(r, 0))
                .ctx("reading injections")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .ctx("reading injections")
        })
    }

    /// Deletes injection rows older than [`INJECTION_RETENTION_DAYS`] before
    /// `now` and returns how many were deleted.
    pub fn prune_injections(&self, now: Timestamp) -> Result<u64> {
        let cutoff = ts(now).saturating_sub(INJECTION_RETENTION_DAYS * MICROS_PER_DAY);
        self.write("pruning injections", |tx| {
            let n = tx
                .execute("DELETE FROM injections WHERE at < ?1", [cutoff])
                .ctx("pruning injections")?;
            Ok(n as u64)
        })
    }
}
