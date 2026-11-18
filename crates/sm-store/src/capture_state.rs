//! The capture state that outlives the daemon (decision B-163, RC2 in
//! docs/v3/CAPTURE-RCA.md) and the near-duplicate lookup capture runs
//! before it stores a candidate (decision B-164, RC9).
//!
//! * `capture_sessions`: each session's transcript read position.
//! * `capture_seen`: the hash of every candidate a project's pipeline has
//!   handled. A text whose hash is here is never captured again for the
//!   project, whatever became of its memory: forgotten, rejected or pruned.
//!
//! Rows unused for [`CAPTURE_STATE_RETENTION_DAYS`] are pruned. Claude Code
//! removes a transcript 30 days after its last use by default
//! (`cleanupPeriodDays`), so the state outlives the transcripts it covers.

use std::collections::BTreeSet;

use rusqlite::OptionalExtension;
use rusqlite::types::Value;
use sm_core::error::{CoreError, Result};
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::time::Timestamp;

use crate::Store;
use crate::codec::{check_vector, ts};
use crate::error::Ctx;
use crate::search::mean_vectors;

/// Capture state rows unused for this long are pruned (B-163).
pub const CAPTURE_STATE_RETENTION_DAYS: i64 = 90;

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// What one capture attempt handled, recorded in one transaction by
/// [`Store::record_capture_progress`].
#[derive(Debug, Clone, Copy)]
pub struct CaptureProgress<'a> {
    /// The session and the transcript offset its next read starts from;
    /// `None` for a capture with no session id, or one whose read position
    /// did not move.
    pub session: Option<(&'a str, u64)>,
    pub project: ProjectId,
    /// The content hashes of every candidate the attempt handled.
    pub hashes: &'a [String],
    pub at: Timestamp,
}

/// The closest active or pending memory to one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    pub id: MemoryId,
    pub cosine: f32,
}

impl Store {
    /// Where the next read of `session`'s transcript starts; `None` for a
    /// session this store has not recorded.
    pub fn capture_offset(&self, session: &str) -> Result<Option<u64>> {
        self.read("reading a capture offset", |tx| {
            let at: Option<i64> = tx
                .prepare_cached(
                    "SELECT transcript_offset FROM capture_sessions WHERE session_id = ?1",
                )
                .and_then(|mut s| s.query_row([session], |r| r.get(0)).optional())
                .ctx("reading a capture offset")?;
            at.map(|o| {
                u64::try_from(o)
                    .map_err(|_| crate::error::corrupt("capture offset", format!("{o}")))
            })
            .transpose()
        })
    }

    /// The hashes among `hashes` that `project`'s capture has handled.
    pub fn captures_seen(&self, project: ProjectId, hashes: &[String]) -> Result<BTreeSet<String>> {
        if hashes.is_empty() {
            return Ok(BTreeSet::new());
        }
        self.read("reading seen captures", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT 1 FROM capture_seen WHERE project_id = ?1 AND content_hash = ?2",
                )
                .ctx("reading seen captures")?;
            let mut seen = BTreeSet::new();
            for h in hashes {
                if s.exists((project.0, h)).ctx("reading seen captures")? {
                    seen.insert(h.clone());
                }
            }
            Ok(seen)
        })
    }

    /// Records, in one transaction, the session's new read position and the
    /// hashes the attempt handled. A hash seen again keeps the later time.
    pub fn record_capture_progress(&self, p: &CaptureProgress<'_>) -> Result<()> {
        let at = ts(p.at);
        let offset = match p.session {
            Some((session, offset)) => {
                if session.trim().is_empty() {
                    return Err(CoreError::invalid_input(
                        "a capture read position needs a session id",
                    ));
                }
                let o = i64::try_from(offset).map_err(|_| {
                    CoreError::invalid_input(format!("capture offset {offset} is out of range"))
                })?;
                Some((session, o))
            }
            None => None,
        };
        self.write("recording capture progress", |tx| {
            if let Some((session, o)) = offset {
                tx.prepare_cached(
                    "INSERT INTO capture_sessions(session_id, transcript_offset, updated_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(session_id) DO UPDATE SET
                         transcript_offset = excluded.transcript_offset,
                         updated_at = excluded.updated_at",
                )
                .and_then(|mut s| s.execute((session, o, at)))
                .ctx("recording a capture offset")?;
            }
            let mut s = tx
                .prepare_cached(
                    "INSERT INTO capture_seen(project_id, content_hash, seen_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(project_id, content_hash) DO UPDATE SET
                         seen_at = max(seen_at, excluded.seen_at)",
                )
                .ctx("recording a seen capture")?;
            for h in p.hashes {
                s.execute((p.project.0, h, at))
                    .ctx("recording a seen capture")?;
            }
            Ok(())
        })
    }

    /// Deletes capture state unused for [`CAPTURE_STATE_RETENTION_DAYS`]
    /// before `now` and returns how many rows were deleted.
    pub fn prune_capture_state(&self, now: Timestamp) -> Result<u64> {
        let cutoff = ts(now).saturating_sub(CAPTURE_STATE_RETENTION_DAYS * MICROS_PER_DAY);
        self.write("pruning capture state", |tx| {
            let sessions = tx
                .execute(
                    "DELETE FROM capture_sessions WHERE updated_at < ?1",
                    [cutoff],
                )
                .ctx("pruning capture offsets")?;
            let seen = tx
                .execute("DELETE FROM capture_seen WHERE seen_at < ?1", [cutoff])
                .ctx("pruning seen captures")?;
            Ok((sessions + seen) as u64)
        })
    }

    /// For each vector in `vectors`, the closest active or pending memory of
    /// `project`, by the cosine of the memory's vector (the mean of its
    /// embedded chunks, as [`Store::memory_vector`] computes it); `None`
    /// when the project has no such memory with an embedded chunk. Each
    /// vector must be unit length. Pending memories count, though they are
    /// never served, so a capture that waits for the judge still keeps a
    /// restatement of it from being stored (B-164).
    pub fn capture_nearest(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Option<Nearest>>> {
        self.nearest_among(project, vectors, "('active', 'pending')")
    }

    /// [`Store::capture_nearest`] among the project's active memories only:
    /// the judge's check before a kept capture becomes active (SPEC 16, The
    /// judge; B-169). The capture itself is pending, so it is never its own
    /// neighbour.
    pub fn active_nearest(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Option<Nearest>>> {
        self.nearest_among(project, vectors, "('active')")
    }

    fn nearest_among(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
        statuses: &'static str,
    ) -> Result<Vec<Option<Nearest>>> {
        for v in vectors {
            check_vector("a capture vector", v)?;
        }
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        self.read("finding near duplicates", |tx| {
            self.check_index(tx)?;
            let params = [Value::Integer(project.0)];
            let clause = format!(" AND c.project_id = ? AND m.status IN {statuses}");
            let stored = mean_vectors(tx, &clause, &params)?;
            Ok(vectors
                .iter()
                .map(|q| {
                    let mut best: Option<Nearest> = None;
                    for (id, v) in &stored {
                        let cosine: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                        if best.is_none_or(|b| cosine > b.cosine) {
                            best = Some(Nearest { id: *id, cosine });
                        }
                    }
                    best
                })
                .collect())
        })
    }
}
