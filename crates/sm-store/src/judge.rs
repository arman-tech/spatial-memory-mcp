//! The capture judge's queue and its daily usage (SPEC section 16, The
//! judge; decisions B-166 to B-168, RC4, RC6 and RC7 in
//! docs/v3/CAPTURE-RCA.md).
//!
//! Every automatic capture is stored `pending` (B-157) with a row in
//! `capture_judge`: the context the judge reads with it and how often a
//! call about it failed. The daemon's judge worker takes a batch from
//! [`Store::judge_queue`], asks the judge, and settles each verdict with
//! [`Store::settle_captures`]: a kept capture becomes `active` with the
//! judge's rewrite as its content, and a rejected one is deleted. A capture
//! never reaches `active` any other way, so nothing unjudged is served.
//!
//! `judge_usage` counts the judge's calls, the captures it judged, its
//! failures and its cost per UTC day, for the daily cap and for `status`.
//! Days older than [`JUDGE_USAGE_RETENTION_DAYS`] are pruned.

use rusqlite::{Connection, OptionalExtension};
use sm_core::error::{CoreError, Result};
use sm_core::hash::content_hash;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::RememberOutcome;
use sm_core::time::Timestamp;
use sm_core::types::{Memory, MemoryType, Status};

use crate::Store;
use crate::codec::{from_ts, memory_id_col, parse_col, ts};
use crate::error::Ctx;
use crate::memories::{Access, InsertOptions, Inserted, MemoryInput, NewChunk};
use crate::memories::{check_chunks, find, insert_chunks, insert_one};

/// The most captures one judge call is given (B-158: a batch of 25 took
/// 10 to 11.5 seconds and cost about $0.0019).
pub const JUDGE_BATCH_MAX: usize = 25;

/// Calls about one capture that may fail with a bad answer before it is
/// left pending, unjudged, until it is pruned `pending_max_days` after the
/// last one (B-178).
pub const JUDGE_MAX_ATTEMPTS: u32 = 3;

/// Days of judge usage kept.
pub const JUDGE_USAGE_RETENTION_DAYS: i64 = 90;

/// The `meta` key holding when schema version 4 was applied, in
/// microseconds since the Unix epoch. A capture pending from before then
/// has no queue row, and its wait for [`Store::prune_pending`] starts here,
/// not when it was captured (B-167). A queued capture is never pruned for
/// waiting (B-178).
pub const JUDGE_SINCE_KEY: &str = "capture_judge_since";

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// One capture waiting for the judge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgeItem {
    pub id: MemoryId,
    pub project: ProjectId,
    /// The capture's content, redacted when it was captured.
    pub content: String,
    /// The text's turn prompt, redacted when it was captured; may be empty.
    pub context: String,
    /// The type the rules gave it.
    pub memory_type: MemoryType,
    /// Calls about it that failed with a bad answer.
    pub attempts: u32,
    pub queued_at: Timestamp,
}

/// How many captures wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JudgeQueueStats {
    /// Every pending memory, judged or not: those the judge cannot take
    /// (captured before the judge, or past [`JUDGE_MAX_ATTEMPTS`]) wait
    /// until they are pruned.
    pub pending: u64,
    /// Pending memories the judge can still take.
    pub queued: u64,
    /// When the oldest of those was queued.
    pub oldest: Option<Timestamp>,
}

/// One verdict to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum Settle {
    /// Make the capture `active` with the judge's rewrite. `chunks` are the
    /// rewrite's, numbered from 0; `judge_model` is recorded in the
    /// provenance's `extra.judge`.
    Keep {
        id: MemoryId,
        content: String,
        title: Option<String>,
        memory_type: MemoryType,
        chunks: Vec<NewChunk>,
        judge_model: String,
        reason: String,
    },
    /// Delete the capture: the judge rejected it, or its rewrite is a near
    /// duplicate of an active memory.
    Delete { id: MemoryId },
}

/// What [`Store::settle_captures`] did with one verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum Settled {
    /// The capture is active, as stored.
    Kept(Box<Memory>),
    /// The rewrite's content hash is held by `of`, a memory of the project
    /// that is not pending; the capture was deleted.
    Duplicate { of: MemoryId },
    /// The capture was deleted.
    Deleted,
    /// The capture was no longer pending (forgotten, replaced by a save of
    /// the same text, or pruned); nothing was changed.
    Gone,
}

/// One day of judge use.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JudgeUsage {
    pub calls: u32,
    pub judged: u64,
    pub failures: u32,
    pub cost_usd: f64,
}

fn status_of(c: &Connection, id: &MemoryId) -> Result<Option<Status>> {
    c.prepare_cached("SELECT status FROM memories WHERE id = ?1")
        .and_then(|mut s| {
            s.query_row([id.to_string()], |r| parse_col(r, 0))
                .optional()
        })
        .ctx("reading a capture's status")
}

fn delete_pending(c: &Connection, id: &MemoryId) -> Result<bool> {
    let n = c
        .prepare_cached("DELETE FROM memories WHERE id = ?1 AND status = 'pending'")
        .and_then(|mut s| s.execute([id.to_string()]))
        .ctx("deleting a capture")?;
    Ok(n > 0)
}

fn drop_judge_row(c: &Connection, id: &MemoryId) -> Result<()> {
    c.prepare_cached("DELETE FROM capture_judge WHERE memory_id = ?1")
        .and_then(|mut s| s.execute([id.to_string()]))
        .ctx("removing a judged capture from the queue")?;
    Ok(())
}

/// The first memory of `project` other than `id`, not pending, whose
/// content hash is `hash`.
fn held_by(
    c: &Connection,
    id: &MemoryId,
    project: ProjectId,
    hash: &str,
) -> Result<Option<MemoryId>> {
    c.prepare_cached(
        "SELECT id FROM memories
         WHERE project_id = ?1 AND content_hash = ?2 AND id <> ?3 AND status <> 'pending'
         ORDER BY created_at, id LIMIT 1",
    )
    .and_then(|mut s| {
        s.query_row((project.0, hash, id.to_string()), |r| memory_id_col(r, 0))
            .optional()
    })
    .ctx("checking a rewrite for a duplicate")
}

/// Adds `"judge": {"model": ..., "reason": ...}` to the provenance's
/// `extra`.
fn mark_judged(c: &Connection, id: &MemoryId, model: &str, reason: &str) -> Result<()> {
    let ids = id.to_string();
    let extra: Option<String> = c
        .prepare_cached("SELECT extra FROM provenance WHERE memory_id = ?1")
        .and_then(|mut s| s.query_row([&ids], |r| r.get(0)))
        .ctx("reading a capture's provenance")?;
    let mut extra = match extra {
        Some(text) => serde_json::from_str::<serde_json::Value>(&text)?,
        None => serde_json::Value::Object(serde_json::Map::new()),
    };
    let Some(map) = extra.as_object_mut() else {
        return Err(crate::error::corrupt(
            "provenance extra",
            "not a JSON object",
        ));
    };
    map.insert(
        "judge".to_owned(),
        serde_json::json!({"model": model, "reason": reason}),
    );
    let extra = serde_json::to_string(&extra)?;
    c.prepare_cached("UPDATE provenance SET extra = ?2 WHERE memory_id = ?1")
        .and_then(|mut s| s.execute((&ids, &extra)))
        .ctx("recording the judge in the provenance")?;
    Ok(())
}

fn settle_one(
    c: &Connection,
    s: Settle,
    model_id: Option<&str>,
    now: Timestamp,
) -> Result<Settled> {
    match s {
        Settle::Delete { id } => Ok(if delete_pending(c, &id)? {
            Settled::Deleted
        } else {
            drop_judge_row(c, &id)?;
            Settled::Gone
        }),
        Settle::Keep {
            id,
            content,
            title,
            memory_type,
            chunks,
            judge_model,
            reason,
        } => {
            if status_of(c, &id)? != Some(Status::Pending) {
                drop_judge_row(c, &id)?;
                return Ok(Settled::Gone);
            }
            if content.trim().is_empty() {
                return Err(CoreError::invalid_input(
                    "a kept capture's rewrite is empty",
                ));
            }
            check_chunks(&chunks, model_id)?;
            let project: ProjectId = c
                .prepare_cached("SELECT project_id FROM memories WHERE id = ?1")
                .and_then(|mut st| st.query_row([id.to_string()], |r| r.get(0).map(ProjectId)))
                .ctx("reading a capture's project")?;
            let hash = content_hash(&content);
            if let Some(of) = held_by(c, &id, project, &hash)? {
                delete_pending(c, &id)?;
                return Ok(Settled::Duplicate { of });
            }
            let ids = id.to_string();
            // The title first, so its trigger refreshes the keyword rows of
            // the old chunks, which are then replaced.
            c.prepare_cached(
                "UPDATE memories SET title = ?2, content = ?3, content_hash = ?4, type = ?5,
                     status = 'active', updated_at = ?6
                 WHERE id = ?1",
            )
            .and_then(|mut st| {
                st.execute(rusqlite::params![
                    &ids,
                    &title,
                    &content,
                    &hash,
                    memory_type.as_str(),
                    ts(now),
                ])
            })
            .ctx("making a capture active")?;
            c.prepare_cached("DELETE FROM chunks WHERE memory_id = ?1")
                .and_then(|mut st| st.execute([&ids]))
                .ctx("removing a capture's chunks")?;
            insert_chunks(c, &id, project, &chunks, model_id)?;
            mark_judged(c, &id, &judge_model, &reason)?;
            drop_judge_row(c, &id)?;
            find(c, &id)?
                .map(|m| Settled::Kept(Box::new(m)))
                .ok_or_else(|| CoreError::internal("a kept capture vanished"))
        }
    }
}

impl Store {
    /// Stores automatic captures, each `pending` with its judge context, in
    /// one transaction: [`Store::insert_memories`] for captures (B-157,
    /// B-166). A capture that is created gets a row in the judge's queue; a
    /// duplicate (B-29) does not, so the memory it matched is left as it
    /// is. Every input must be `pending`.
    pub fn insert_captures(&self, inputs: Vec<(MemoryInput, String)>) -> Result<Vec<Inserted>> {
        if let Some((i, _)) = inputs
            .iter()
            .find(|(i, _)| i.memory.status != Status::Pending)
        {
            return Err(CoreError::internal(format!(
                "an automatic capture must be stored pending, not {}",
                i.memory.status.as_str()
            )));
        }
        let model = self.configured_model_id();
        let now = Timestamp::now();
        self.write("storing captures", |tx| {
            let mut out = Vec::with_capacity(inputs.len());
            for (input, context) in inputs {
                // A capture made by the rules before the judge (no queue
                // row; B-167) is never judged, so a capture of the same text
                // under the current rules replaces it, and a replay can
                // capture what it held again (B-176).
                let hash = input
                    .memory
                    .content_hash
                    .clone()
                    .unwrap_or_else(|| content_hash(&input.memory.content));
                tx.prepare_cached(
                    "DELETE FROM memories
                     WHERE project_id = ?1 AND content_hash = ?2 AND status = 'pending'
                       AND id NOT IN (SELECT memory_id FROM capture_judge)",
                )
                .and_then(|mut s| s.execute((input.memory.project_id.0, &hash)))
                .ctx("replacing an unqueued pending capture")?;
                let inserted = insert_one(
                    tx,
                    input,
                    Access::Local,
                    model.as_deref(),
                    now,
                    InsertOptions::default(),
                )?;
                if inserted.outcome == RememberOutcome::Created {
                    tx.prepare_cached(
                        "INSERT INTO capture_judge(memory_id, context, queued_at, attempts)
                         VALUES (?1, ?2, ?3, 0)",
                    )
                    .and_then(|mut s| s.execute((inserted.id.to_string(), &context, ts(now))))
                    .ctx("queueing a capture for the judge")?;
                }
                out.push(inserted);
            }
            Ok(out)
        })
    }

    /// The next batch for the judge. Captures no call has failed on come
    /// first, oldest first, up to `max_items` and, after the first,
    /// `max_chars` characters of content and context together; when there
    /// are none, the oldest capture a call failed on, alone, so a bad
    /// answer is traced to one capture. Only pending memories with fewer
    /// than [`JUDGE_MAX_ATTEMPTS`] failed calls are taken.
    pub fn judge_queue(&self, max_items: usize, max_chars: usize) -> Result<Vec<JudgeItem>> {
        if max_items == 0 {
            return Ok(Vec::new());
        }
        self.read("reading the judge queue", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT m.id, m.project_id, m.content, j.context, m.type, j.attempts, j.queued_at
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     WHERE m.status = 'pending' AND j.attempts < ?1
                     ORDER BY j.attempts > 0, j.queued_at, m.id",
                )
                .ctx("reading the judge queue")?;
            let mut rows = s
                .query([i64::from(JUDGE_MAX_ATTEMPTS)])
                .ctx("reading the judge queue")?;
            let mut out: Vec<JudgeItem> = Vec::new();
            let mut chars = 0usize;
            while let Some(r) = rows.next().ctx("reading the judge queue")? {
                let attempts: i64 = r.get(5).ctx("reading the judge queue")?;
                let item = JudgeItem {
                    id: memory_id_col(r, 0).ctx("reading the judge queue")?,
                    project: ProjectId(r.get(1).ctx("reading the judge queue")?),
                    content: r.get(2).ctx("reading the judge queue")?,
                    context: r.get(3).ctx("reading the judge queue")?,
                    memory_type: parse_col(r, 4).ctx("reading the judge queue")?,
                    attempts: u32::try_from(attempts).unwrap_or(u32::MAX),
                    queued_at: from_ts(r.get(6).ctx("reading the judge queue")?),
                };
                let size = item.content.chars().count() + item.context.chars().count();
                if item.attempts > 0 {
                    if out.is_empty() {
                        out.push(item);
                    }
                    break;
                }
                if !out.is_empty() && (out.len() >= max_items || chars + size > max_chars) {
                    break;
                }
                chars += size;
                out.push(item);
            }
            Ok(out)
        })
    }

    /// How many captures wait, and since when.
    pub fn judge_queue_stats(&self) -> Result<JudgeQueueStats> {
        self.read("counting pending captures", |tx| {
            let pending: i64 = tx
                .query_row(
                    "SELECT count(*) FROM memories WHERE status = 'pending'",
                    [],
                    |r| r.get(0),
                )
                .ctx("counting pending captures")?;
            let (queued, oldest): (i64, Option<i64>) = tx
                .query_row(
                    "SELECT count(*), min(j.queued_at)
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     WHERE m.status = 'pending' AND j.attempts < ?1",
                    [i64::from(JUDGE_MAX_ATTEMPTS)],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ctx("counting queued captures")?;
            Ok(JudgeQueueStats {
                pending: u64::try_from(pending).unwrap_or(0),
                queued: u64::try_from(queued).unwrap_or(0),
                oldest: oldest.map(from_ts),
            })
        })
    }

    /// Counts a failed call (a bad answer) against each capture in `ids`,
    /// and records when, for [`Store::prune_pending`] (B-178).
    pub fn judge_failed(&self, ids: &[MemoryId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = ts(Timestamp::now());
        self.write("counting a failed judge call", |tx| {
            let mut s = tx
                .prepare_cached(
                    "UPDATE capture_judge SET attempts = attempts + 1, failed_at = ?2
                     WHERE memory_id = ?1",
                )
                .ctx("counting a failed judge call")?;
            for id in ids {
                s.execute((id.to_string(), now))
                    .ctx("counting a failed judge call")?;
            }
            Ok(())
        })
    }

    /// Applies the judge's verdicts in one transaction. A capture that is no
    /// longer pending is left as it is ([`Settled::Gone`]). A kept capture
    /// keeps its id, provenance, tags, importance and anchors, and takes the
    /// rewrite's content, hash, title, type and chunks; when another memory
    /// of the project that is not pending holds the rewrite's content hash,
    /// the capture is deleted instead ([`Settled::Duplicate`]).
    pub fn settle_captures(&self, verdicts: Vec<Settle>) -> Result<Vec<Settled>> {
        let model = self.configured_model_id();
        let now = Timestamp::now();
        self.write("settling judged captures", |tx| {
            verdicts
                .into_iter()
                .map(|v| settle_one(tx, v, model.as_deref(), now))
                .collect()
        })
    }

    /// Records one judge call for `day` (`YYYY-MM-DD`, UTC): the captures it
    /// judged, whether it failed and its cost.
    pub fn record_judge_call(
        &self,
        day: &str,
        judged: u64,
        failed: bool,
        cost_usd: f64,
    ) -> Result<()> {
        if !cost_usd.is_finite() || cost_usd < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "a judge call's cost must be a finite, non-negative number, not {cost_usd}"
            )));
        }
        let judged = i64::try_from(judged)
            .map_err(|_| CoreError::invalid_input("too many judged captures"))?;
        self.write("recording a judge call", |tx| {
            tx.prepare_cached(
                "INSERT INTO judge_usage(day, calls, judged, failures, cost_usd)
                 VALUES (?1, 1, ?2, ?3, ?4)
                 ON CONFLICT(day) DO UPDATE SET
                     calls = calls + 1,
                     judged = judged + excluded.judged,
                     failures = failures + excluded.failures,
                     cost_usd = cost_usd + excluded.cost_usd",
            )
            .and_then(|mut s| s.execute((day, judged, i64::from(failed), cost_usd)))
            .ctx("recording a judge call")?;
            Ok(())
        })
    }

    /// The judge's use on `day` (`YYYY-MM-DD`, UTC); zeros for a day with
    /// no call.
    pub fn judge_usage(&self, day: &str) -> Result<JudgeUsage> {
        self.read("reading judge usage", |tx| {
            let row: Option<(i64, i64, i64, f64)> = tx
                .prepare_cached(
                    "SELECT calls, judged, failures, cost_usd FROM judge_usage WHERE day = ?1",
                )
                .and_then(|mut s| {
                    s.query_row([day], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
                        .optional()
                })
                .ctx("reading judge usage")?;
            Ok(
                row.map_or_else(JudgeUsage::default, |(calls, judged, failures, cost)| {
                    JudgeUsage {
                        calls: u32::try_from(calls).unwrap_or(u32::MAX),
                        judged: u64::try_from(judged).unwrap_or(0),
                        failures: u32::try_from(failures).unwrap_or(u32::MAX),
                        cost_usd: cost,
                    }
                }),
            )
        })
    }

    /// Deletes the pending memories the judge will never take that have
    /// waited more than `max_days` days before `now` (a queue row goes with
    /// its memory), and judge usage older than
    /// [`JUDGE_USAGE_RETENTION_DAYS`] (B-178).
    ///
    /// A capture in the queue that the judge has not given up on is never
    /// pruned, however long the judge was off, unavailable or out of calls:
    /// only a capture with [`JUDGE_MAX_ATTEMPTS`] bad answers is, and its
    /// wait starts at its last bad answer. A capture pending from before
    /// schema version 4 has no queue row and the judge never takes it; it
    /// waits from when it was captured or from the upgrade
    /// ([`JUDGE_SINCE_KEY`]), whichever is later, so the upgrade itself
    /// deletes nothing. Returns the captures deleted.
    pub fn prune_pending(&self, now: Timestamp, max_days: u32) -> Result<u64> {
        let cutoff = ts(now).saturating_sub(i64::from(max_days) * MICROS_PER_DAY);
        let usage_cutoff = now
            .add_micros(-JUDGE_USAGE_RETENTION_DAYS * MICROS_PER_DAY)
            .date_string();
        self.write("pruning pending captures", |tx| {
            let n = tx
                .execute(
                    "DELETE FROM memories
                     WHERE status = 'pending'
                       AND (
                         (id NOT IN (SELECT memory_id FROM capture_judge)
                          AND max(created_at, coalesce(
                                (SELECT CAST(value AS INTEGER) FROM meta WHERE key = ?2),
                                0)) < ?1)
                         OR id IN (SELECT memory_id FROM capture_judge
                                   WHERE attempts >= ?3
                                     AND coalesce(failed_at, queued_at) < ?1))",
                    (cutoff, JUDGE_SINCE_KEY, i64::from(JUDGE_MAX_ATTEMPTS)),
                )
                .ctx("pruning pending captures")?;
            tx.execute("DELETE FROM judge_usage WHERE day < ?1", [&usage_cutoff])
                .ctx("pruning judge usage")?;
            Ok(n as u64)
        })
    }
}
