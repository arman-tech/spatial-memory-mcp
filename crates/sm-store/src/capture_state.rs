//! The capture state that outlives the daemon (decision B-163, RC2 in
//! docs/v3/CAPTURE-RCA.md) and the near-duplicate lookup capture runs
//! before it stores a candidate (decision B-164, RC9).
//!
//! * `capture_sessions`: each session's transcript read position, and
//!   whether capture read the session from its first line (B-184).
//! * `capture_seen`: the hash of every candidate a project's pipeline has
//!   handled. A text whose hash is here is never captured again for the
//!   project, whatever became of its memory: forgotten, rejected or pruned.
//!   The one exception is a memory still pending with no row in the judge's
//!   queue, made by the rules before the judge: a capture of its text
//!   replaces it (B-176).
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

/// Where the next read of a session's transcript starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapturePosition {
    /// The transcript byte offset the next read starts from.
    pub offset: u64,
    /// Whether capture has read every text before `offset`, from the
    /// session's first line; only then are its compaction summaries offered
    /// (B-184).
    pub from_start: bool,
}

/// What one capture attempt handled, recorded in one transaction by
/// [`Store::record_capture_progress`].
#[derive(Debug, Clone, Copy)]
pub struct CaptureProgress<'a> {
    /// The session and the position its next read starts from; `None` for a
    /// capture with no session id, or one whose read position did not move.
    pub session: Option<(&'a str, CapturePosition)>,
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
    pub fn capture_position(&self, session: &str) -> Result<Option<CapturePosition>> {
        self.read("reading a capture offset", |tx| {
            let at: Option<(i64, bool)> = tx
                .prepare_cached(
                    "SELECT transcript_offset, from_start FROM capture_sessions
                     WHERE session_id = ?1",
                )
                .and_then(|mut s| {
                    s.query_row([session], |r| Ok((r.get(0)?, r.get(1)?)))
                        .optional()
                })
                .ctx("reading a capture offset")?;
            at.map(|(o, from_start)| {
                u64::try_from(o)
                    .map(|offset| CapturePosition { offset, from_start })
                    .map_err(|_| crate::error::corrupt("capture offset", format!("{o}")))
            })
            .transpose()
        })
    }

    /// The transcript offset of [`Store::capture_position`].
    pub fn capture_offset(&self, session: &str) -> Result<Option<u64>> {
        Ok(self.capture_position(session)?.map(|p| p.offset))
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

    /// The hashes among `hashes` held by a pending memory of `project` with
    /// no row in the judge's queue: a capture the rules made before the
    /// judge (B-167), never judged. Capture lets such a text through though
    /// its hash was seen, so [`Store::insert_captures`] replaces the old
    /// capture with one the judge settles (B-176).
    pub fn unqueued_capture_hashes(
        &self,
        project: ProjectId,
        hashes: &[String],
    ) -> Result<BTreeSet<String>> {
        if hashes.is_empty() {
            return Ok(BTreeSet::new());
        }
        self.read("reading unqueued captures", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT 1 FROM memories
                     WHERE project_id = ?1 AND content_hash = ?2 AND status = 'pending'
                       AND id NOT IN (SELECT memory_id FROM capture_judge)",
                )
                .ctx("reading unqueued captures")?;
            let mut held = BTreeSet::new();
            for h in hashes {
                if s.exists((project.0, h)).ctx("reading unqueued captures")? {
                    held.insert(h.clone());
                }
            }
            Ok(held)
        })
    }

    /// Records, in one transaction, the session's new read position and the
    /// hashes the attempt handled. A hash seen again keeps the later time.
    pub fn record_capture_progress(&self, p: &CaptureProgress<'_>) -> Result<()> {
        let at = ts(p.at);
        let offset = match p.session {
            Some((session, position)) => {
                if session.trim().is_empty() {
                    return Err(CoreError::invalid_input(
                        "a capture read position needs a session id",
                    ));
                }
                let offset = position.offset;
                let o = i64::try_from(offset).map_err(|_| {
                    CoreError::invalid_input(format!("capture offset {offset} is out of range"))
                })?;
                Some((session, o, position.from_start))
            }
            None => None,
        };
        self.write("recording capture progress", |tx| {
            if let Some((session, o, from_start)) = offset {
                tx.prepare_cached(
                    "INSERT INTO capture_sessions(session_id, transcript_offset, from_start,
                                                  updated_at)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(session_id) DO UPDATE SET
                         transcript_offset = excluded.transcript_offset,
                         from_start = excluded.from_start,
                         updated_at = excluded.updated_at",
                )
                .and_then(|mut s| s.execute((session, o, from_start, at)))
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
    /// restatement of it from being stored (B-164). A pending memory with no
    /// row in the judge's queue does not count: the rules before the judge
    /// made it, it is never judged, and it must not keep a capture of the
    /// same knowledge under the current rules from being stored (B-176).
    /// Nor does a rejected capture: it is kept only to be looked at, and
    /// must not keep a later capture of better text out (B-190).
    pub fn capture_nearest(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Option<Nearest>>> {
        Ok(self
            .near_among(
                project,
                vectors,
                "(m.status = 'active' OR (m.status = 'pending' \
                 AND m.id IN (SELECT memory_id FROM capture_judge \
                              WHERE rejected_at IS NULL)))",
                f32::NEG_INFINITY,
                1,
            )?
            .into_iter()
            .map(|n| n.into_iter().next())
            .collect())
    }

    /// For each vector in `vectors`, the project's active memories whose
    /// cosine with it is at least `min_cosine`, closest first, at most `k`
    /// of them: the judge's checks before a kept capture becomes active
    /// (SPEC 16, The judge; B-169, B-192). The capture itself is pending,
    /// so it is never its own neighbour.
    pub fn active_near(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
        min_cosine: f32,
        k: usize,
    ) -> Result<Vec<Vec<Nearest>>> {
        self.near_among(project, vectors, "m.status = 'active'", min_cosine, k)
    }

    /// For each vector, the memories of `project` that `among`, an SQL
    /// condition on the memory `m`, holds for and whose cosine with it is
    /// at least `min_cosine`: closest first, at most `k`, an equal cosine
    /// keeping the lower id first.
    fn near_among(
        &self,
        project: ProjectId,
        vectors: &[Vec<f32>],
        among: &'static str,
        min_cosine: f32,
        k: usize,
    ) -> Result<Vec<Vec<Nearest>>> {
        for v in vectors {
            check_vector("a capture vector", v)?;
        }
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        self.read("finding near memories", |tx| {
            self.check_index(tx)?;
            let params = [Value::Integer(project.0)];
            let clause = format!(" AND c.project_id = ? AND {among}");
            let stored = mean_vectors(tx, &clause, &params)?;
            Ok(vectors
                .iter()
                .map(|q| {
                    let mut near: Vec<Nearest> = stored
                        .iter()
                        .map(|(id, v)| Nearest {
                            id: *id,
                            cosine: v.iter().zip(q).map(|(a, b)| a * b).sum(),
                        })
                        .filter(|n| n.cosine >= min_cosine)
                        .collect();
                    // Stable: equal cosines keep the stored order, by id.
                    near.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
                    near.truncate(k);
                    near
                })
                .collect())
        })
    }
}
