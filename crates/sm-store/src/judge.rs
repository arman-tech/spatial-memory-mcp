//! The capture judge's queue and its daily usage (SPEC section 16, The
//! judge; decisions B-166 to B-168, RC4, RC6 and RC7 in
//! docs/v3/CAPTURE-RCA.md).
//!
//! Every automatic capture is stored `pending` (B-157) with a row in
//! `capture_judge`: the context the judge reads with it and how often a
//! call about it failed. The daemon's judge worker takes a batch from
//! [`Store::judge_queue`], asks the judge, and settles each verdict with
//! [`Store::settle_captures`]: a kept capture becomes `active` with the
//! judge's rewrite as its content, and a rejected one stays `pending`,
//! marked rejected, for [`JUDGE_REJECTED_KEEP_DAYS`] days and is then
//! deleted (B-190). A capture never reaches `active` any other way, so
//! nothing unjudged or rejected is served.
//!
//! `judge_usage` counts the judge's calls, the captures it judged, its
//! failures and its cost per UTC day, for the daily cap and for `status`.
//! Days older than [`JUDGE_USAGE_RETENTION_DAYS`] are pruned.

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension, Row};
use sm_core::error::{CoreError, Result};
use sm_core::hash::content_hash;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::RememberOutcome;
use sm_core::time::Timestamp;
use sm_core::types::{Memory, MemoryType, Provenance, ProvenanceSource, SourceTag, Status};

use crate::Store;
use crate::codec::{from_ts, memory_id_col, opt_json_col, parse_col, parse_opt_col, ts};
use crate::error::Ctx;
use crate::memories::{Access, InsertOptions, Inserted, MemoryInput, NewChunk};
use crate::memories::{check_chunks, find, insert_chunks, insert_one};

/// The most captures one judge call is given (B-158: a batch of 25 took
/// 10 to 11.5 seconds and cost about $0.0019).
pub const JUDGE_BATCH_MAX: usize = 25;

/// Calls about one capture whose verdict could not be applied (its spans
/// found nothing, or it had to be judged again) before it is left pending,
/// unjudged, until it is pruned `pending_max_days` after the last one
/// (B-178). A capture marked failed judgement after two bad answers alone
/// is raised to this count at once (B-206).
pub const JUDGE_MAX_ATTEMPTS: u32 = 3;

/// Days of judge usage kept.
pub const JUDGE_USAGE_RETENTION_DAYS: i64 = 90;

/// Days a rejected capture is kept, pending and never served, before
/// [`Store::prune_pending`] deletes it (B-190).
pub const JUDGE_REJECTED_KEEP_DAYS: i64 = 30;

/// The longest reason a rejection records (schema version 7).
pub const JUDGE_REASON_MAX_CHARS: usize = 64;

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
    /// Calls about it whose verdict could not be applied (B-178, B-206).
    pub attempts: u32,
    pub queued_at: Timestamp,
    /// The session it was captured in, when the hook recorded one.
    pub session: Option<String>,
    /// The kind of text it holds (B-179, B-220): a capture whose kind is
    /// switched off is settled without a call (B-245).
    pub source: SourceTag,
}

/// The columns [`item_from_row`] reads, from the queue joined to its
/// memory and provenance.
const QUEUE_SELECT: &str =
    "SELECT m.id, m.project_id, m.content, j.context, m.type, j.attempts, j.queued_at,
            p.session_id, p.source, p.tool, p.extra
     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
     LEFT JOIN provenance p ON p.memory_id = m.id";

fn item_from_row(r: &Row<'_>) -> rusqlite::Result<JudgeItem> {
    let attempts: i64 = r.get(5)?;
    let source: Option<ProvenanceSource> = parse_opt_col(r, 8)?;
    let provenance = Provenance {
        tool: r.get(9)?,
        extra: opt_json_col(r, 10)?,
        ..Provenance::new(source.unwrap_or(ProvenanceSource::Capture))
    };
    Ok(JudgeItem {
        id: memory_id_col(r, 0)?,
        project: ProjectId(r.get(1)?),
        content: r.get(2)?,
        context: r.get(3)?,
        memory_type: parse_col(r, 4)?,
        attempts: u32::try_from(attempts).unwrap_or(u32::MAX),
        queued_at: from_ts(r.get(6)?),
        session: r.get(7)?,
        source: SourceTag::of(&provenance),
    })
}

/// How many captures wait, and how many were rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JudgeQueueStats {
    /// Every pending memory not rejected: those the judge cannot take
    /// (captured before the judge, or past [`JUDGE_MAX_ATTEMPTS`]) wait
    /// until they are pruned.
    pub pending: u64,
    /// Pending memories the judge can still take.
    pub queued: u64,
    /// When the oldest of those was queued.
    pub oldest: Option<Timestamp>,
    /// Rejected captures, kept for [`JUDGE_REJECTED_KEEP_DAYS`] days (B-190).
    pub rejected: u64,
    /// Pending captures the judge gave up on: past [`JUDGE_MAX_ATTEMPTS`],
    /// among them those marked failed judgement (B-206). They are counted
    /// in `pending`, never served, and wait only to be pruned.
    pub failed: u64,
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
    /// Reject the capture: the judge rejected it, its rewrite is a near
    /// duplicate of an active memory, or the duplicate check found nothing
    /// new in it (B-192). It stays pending with its own content, never
    /// served, and is deleted [`JUDGE_REJECTED_KEEP_DAYS`] days later
    /// (B-190). `reason` is the judge's reason code, `near_duplicate` or
    /// `repeats`, at most [`JUDGE_REASON_MAX_CHARS`] characters.
    Reject { id: MemoryId, reason: String },
    /// The statements task's answer for one capture (B-242): each
    /// statement it keeps, in the answer's order, with its own words.
    /// `judge_model` is recorded in each memory's `extra.judge`.
    Statements {
        id: MemoryId,
        judge_model: String,
        statements: Vec<StatementSettle>,
    },
}

/// One statement of a capture, as the statements task settles it
/// (B-242). Its content is the capture's own words, quoted by the judge's
/// span and redacted; its title is derived from them (owner's change A6:
/// no text a model wrote is stored).
#[derive(Debug, Clone, PartialEq)]
pub struct StatementSettle {
    /// Its place in the judge's answer for the capture, from 0.
    pub index: u32,
    pub content: String,
    pub title: Option<String>,
    pub memory_type: MemoryType,
    /// The statement's chunks, numbered from 0.
    pub chunks: Vec<NewChunk>,
    /// The judge's grade: `must_keep` or `good_to_have`.
    pub grade: String,
    /// Set when the statement is not stored but kept unserved (B-243).
    pub hold: Option<Hold>,
}

/// Why a statement is kept unserved, not stored (B-243). Either way it is
/// its own pending memory, rejected and kept for
/// [`JUDGE_REJECTED_KEEP_DAYS`] days, so the drop can be audited and
/// undone by revising it to active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    /// The duplicate step said it repeats this (rejected `repeat`).
    Repeat(RepeatOf),
    /// Its duplicate answer still failed after the call was halved to it
    /// alone and tried [`JUDGE_MAX_ATTEMPTS`] times (rejected `unchecked`;
    /// owner's change A2).
    Unchecked,
}

/// What a held statement repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatOf {
    /// A memory already in the store.
    Memory(MemoryId),
    /// An earlier statement of the same batch, settled in the same call:
    /// the memory it became, or the one it was found to duplicate.
    Statement { capture: MemoryId, index: u32 },
}

/// What [`Store::settle_captures`] did with one statement.
#[derive(Debug, Clone, PartialEq)]
pub enum StatementSettled {
    /// Stored as an active memory.
    Stored(Box<Memory>),
    /// Kept unserved as the pending memory `id` (B-243); `repeat_of` is
    /// the memory it was said to repeat, `None` when it was unchecked.
    Held {
        id: MemoryId,
        repeat_of: Option<MemoryId>,
    },
    /// Its content hash is held by `of`, a memory of the project that is
    /// not pending; nothing was stored.
    Duplicate { of: MemoryId },
}

/// Statements of the statements task since a time (B-242, B-243), for
/// `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatementCounts {
    /// Stored, and not pending.
    pub stored: u64,
    /// Kept unserved as repeats, and still held.
    pub repeats: u64,
    /// Kept unserved as unchecked, and still held.
    pub unchecked: u64,
}

/// What [`Store::settle_captures`] did with one verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum Settled {
    /// The capture is active, as stored.
    Kept(Box<Memory>),
    /// The rewrite's content hash is held by `of`, a memory of the project
    /// that is not pending; the capture was rejected with the reason
    /// `duplicate`.
    Duplicate { of: MemoryId },
    /// The capture was rejected.
    Rejected,
    /// The capture was no longer pending (forgotten, replaced by a save of
    /// the same text, or pruned); nothing was changed.
    Gone,
    /// The statements task's capture, one entry per statement in order.
    /// When none was stored, the capture was rejected (B-242).
    Statements(Vec<StatementSettled>),
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

/// Whether the capture was rejected (B-190).
fn is_rejected(c: &Connection, id: &MemoryId) -> Result<bool> {
    c.prepare_cached("SELECT 1 FROM capture_judge WHERE memory_id = ?1 AND rejected_at IS NOT NULL")
        .and_then(|mut s| s.exists([id.to_string()]))
        .ctx("reading whether a capture was rejected")
}

/// Marks a pending capture rejected, keeping it and its queue row; `false`
/// when it is not pending. A capture with no queue row gets one.
fn reject_pending(c: &Connection, id: &MemoryId, reason: &str, now: Timestamp) -> Result<bool> {
    let reason = reason.trim();
    if reason.is_empty() || reason.chars().count() > JUDGE_REASON_MAX_CHARS {
        return Err(CoreError::invalid_input(format!(
            "a rejection's reason must be 1 to {JUDGE_REASON_MAX_CHARS} characters"
        )));
    }
    if status_of(c, id)? != Some(Status::Pending) {
        return Ok(false);
    }
    c.prepare_cached(
        "INSERT INTO capture_judge(memory_id, context, queued_at, attempts, rejected_at,
                                   rejected_reason)
         VALUES (?1, '', ?2, 0, ?2, ?3)
         ON CONFLICT(memory_id) DO UPDATE SET
             rejected_at = excluded.rejected_at,
             rejected_reason = excluded.rejected_reason",
    )
    .and_then(|mut s| s.execute((id.to_string(), ts(now), reason)))
    .ctx("rejecting a capture")?;
    Ok(true)
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
    placed: &mut HashMap<(MemoryId, u32), MemoryId>,
) -> Result<Settled> {
    match s {
        Settle::Statements {
            id,
            judge_model,
            statements,
        } => settle_statements(c, id, &judge_model, statements, model_id, now, placed),
        Settle::Reject { id, reason } => {
            if is_rejected(c, &id)? {
                return Ok(Settled::Gone);
            }
            Ok(if reject_pending(c, &id, &reason, now)? {
                Settled::Rejected
            } else {
                drop_judge_row(c, &id)?;
                Settled::Gone
            })
        }
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
            if is_rejected(c, &id)? {
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
                reject_pending(c, &id, "duplicate", now)?;
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

/// The first memory of `project`, not pending, whose content hash is
/// `hash`: a statement the store already holds word for word.
fn hash_held(c: &Connection, project: ProjectId, hash: &str) -> Result<Option<MemoryId>> {
    c.prepare_cached(
        "SELECT id FROM memories
         WHERE project_id = ?1 AND content_hash = ?2 AND status <> 'pending'
         ORDER BY created_at, id LIMIT 1",
    )
    .and_then(|mut s| {
        s.query_row((project.0, hash), |r| memory_id_col(r, 0))
            .optional()
    })
    .ctx("checking a statement for a duplicate")
}

/// What a statement's memory carries from its capture: the project, the
/// provenance's `extra` as captured, and the judge's context.
struct CaptureRow {
    project: ProjectId,
    extra: Option<String>,
    context: String,
}

fn capture_row(c: &Connection, id: &MemoryId) -> Result<CaptureRow> {
    c.prepare_cached(
        "SELECT m.project_id, p.extra, coalesce(j.context, '')
         FROM memories m LEFT JOIN provenance p ON p.memory_id = m.id
         LEFT JOIN capture_judge j ON j.memory_id = m.id
         WHERE m.id = ?1",
    )
    .and_then(|mut s| {
        s.query_row([id.to_string()], |r| {
            Ok(CaptureRow {
                project: ProjectId(r.get(0)?),
                extra: r.get(1)?,
                context: r.get(2)?,
            })
        })
    })
    .ctx("reading a judged capture")
}

/// The capture's `extra` with `"judge": judge` added, as JSON text.
fn with_judge(extra: Option<&str>, judge: serde_json::Value) -> Result<String> {
    let mut extra = match extra {
        Some(text) => serde_json::from_str::<serde_json::Value>(text)?,
        None => serde_json::Value::Object(serde_json::Map::new()),
    };
    let Some(map) = extra.as_object_mut() else {
        return Err(crate::error::corrupt(
            "provenance extra",
            "not a JSON object",
        ));
    };
    map.insert("judge".to_owned(), judge);
    Ok(serde_json::to_string(&extra)?)
}

/// A new memory for one statement of `capture`, with `status`: the
/// capture's project, origin, importance, tags, imported fields and time,
/// its provenance with `extra`, and its anchors; the statement's own
/// words, title, type and chunks. Written by SQL, not by `insert_one`, so
/// that an active statement never deletes a pending memory of the same
/// text (B-157) and a held one never counts as a duplicate of one.
#[allow(clippy::too_many_arguments)]
fn copy_capture(
    c: &Connection,
    capture: &MemoryId,
    project: ProjectId,
    st: &StatementSettle,
    hash: &str,
    status: Status,
    extra: &str,
    model_id: Option<&str>,
    now: Timestamp,
) -> Result<MemoryId> {
    let id = MemoryId::new_v7();
    let ids = id.to_string();
    let from = capture.to_string();
    c.prepare_cached(
        "INSERT INTO memories(id, project_id, type, status, title, content, content_hash,
             supersedes_id, origin, importance, tags, imported, created_at, updated_at)
         SELECT ?1, project_id, ?3, ?4, ?5, ?6, ?7, NULL, origin, importance, tags, imported,
             created_at, ?8
         FROM memories WHERE id = ?2",
    )
    .and_then(|mut s| {
        s.execute(rusqlite::params![
            &ids,
            &from,
            st.memory_type.as_str(),
            status.as_str(),
            &st.title,
            &st.content,
            hash,
            ts(now),
        ])
    })
    .ctx("storing a statement")?;
    c.prepare_cached(
        "INSERT INTO provenance(memory_id, source, author, session_id, commit_sha, file_path,
             tool, extra)
         SELECT ?1, source, author, session_id, commit_sha, file_path, tool, ?3
         FROM provenance WHERE memory_id = ?2",
    )
    .and_then(|mut s| s.execute((&ids, &from, extra)))
    .ctx("recording a statement's provenance")?;
    c.prepare_cached(
        "INSERT INTO memory_anchors(memory_id, anchor_id, position)
         SELECT ?1, anchor_id, position FROM memory_anchors WHERE memory_id = ?2",
    )
    .and_then(|mut s| s.execute((&ids, &from)))
    .ctx("attaching a statement's anchors")?;
    // After the row, so the keyword trigger reads the statement's title.
    insert_chunks(c, &id, project, &st.chunks, model_id)?;
    Ok(id)
}

/// Settles one capture's statements (B-242, B-243), in the answer's
/// order, inside [`Store::settle_captures`]' transaction. `placed` maps
/// each statement of the batch settled so far to the memory it became or
/// duplicates, for a later statement said to repeat it.
fn settle_statements(
    c: &Connection,
    id: MemoryId,
    judge_model: &str,
    statements: Vec<StatementSettle>,
    model_id: Option<&str>,
    now: Timestamp,
    placed: &mut HashMap<(MemoryId, u32), MemoryId>,
) -> Result<Settled> {
    if status_of(c, &id)? != Some(Status::Pending) {
        drop_judge_row(c, &id)?;
        return Ok(Settled::Gone);
    }
    if is_rejected(c, &id)? {
        return Ok(Settled::Gone);
    }
    let row = capture_row(c, &id)?;
    let ids = id.to_string();
    let mut out = Vec::with_capacity(statements.len());
    let mut taken = false;
    let (mut unchecked, mut repeats, mut duplicates) = (false, false, false);
    for st in statements {
        if st.content.trim().is_empty() {
            return Err(CoreError::invalid_input("a kept statement is empty"));
        }
        if st.grade.trim().is_empty() {
            return Err(CoreError::invalid_input("a kept statement has no grade"));
        }
        check_chunks(&st.chunks, model_id)?;
        let hash = content_hash(&st.content);
        let mut judge = serde_json::json!({
            "model": judge_model,
            "task": "statements",
            "capture": ids,
            "statement": st.index,
            "grade": st.grade,
            "at": ts(now),
        });
        if let Some(hold) = st.hold {
            let (reason, repeat_of) = match hold {
                Hold::Unchecked => {
                    unchecked = true;
                    ("unchecked", None)
                }
                Hold::Repeat(RepeatOf::Memory(of)) => {
                    repeats = true;
                    ("repeat", Some(of))
                }
                Hold::Repeat(RepeatOf::Statement { capture, index }) => {
                    repeats = true;
                    // A statement of a capture that was gone is named by
                    // its capture.
                    let of = placed.get(&(capture, index)).copied().unwrap_or(capture);
                    ("repeat", Some(of))
                }
            };
            judge["repeat_of"] = repeat_of.map_or(serde_json::Value::Null, |o: MemoryId| {
                serde_json::Value::String(o.to_string())
            });
            let extra = with_judge(row.extra.as_deref(), judge)?;
            let held = copy_capture(
                c,
                &id,
                row.project,
                &st,
                &hash,
                Status::Pending,
                &extra,
                model_id,
                now,
            )?;
            c.prepare_cached(
                "INSERT INTO capture_judge(memory_id, context, queued_at, attempts, rejected_at,
                                           rejected_reason)
                 VALUES (?1, ?2, ?3, 0, ?3, ?4)",
            )
            .and_then(|mut s| s.execute((held.to_string(), &row.context, ts(now), reason)))
            .ctx("holding a statement")?;
            placed.insert((id, st.index), held);
            out.push(StatementSettled::Held {
                id: held,
                repeat_of,
            });
            continue;
        }
        if let Some(of) = hash_held(c, row.project, &hash)? {
            duplicates = true;
            placed.insert((id, st.index), of);
            out.push(StatementSettled::Duplicate { of });
            continue;
        }
        let extra = with_judge(row.extra.as_deref(), judge)?;
        let stored = if taken {
            copy_capture(
                c,
                &id,
                row.project,
                &st,
                &hash,
                Status::Active,
                &extra,
                model_id,
                now,
            )?
        } else {
            // The first statement stored takes the capture's row. The
            // title first, so its trigger refreshes the keyword rows of the
            // old chunks, which are then replaced.
            c.prepare_cached(
                "UPDATE memories SET title = ?2, content = ?3, content_hash = ?4, type = ?5,
                     status = 'active', updated_at = ?6
                 WHERE id = ?1",
            )
            .and_then(|mut s| {
                s.execute(rusqlite::params![
                    &ids,
                    &st.title,
                    &st.content,
                    &hash,
                    st.memory_type.as_str(),
                    ts(now),
                ])
            })
            .ctx("storing a capture's first statement")?;
            c.prepare_cached("DELETE FROM chunks WHERE memory_id = ?1")
                .and_then(|mut s| s.execute([&ids]))
                .ctx("removing a capture's chunks")?;
            insert_chunks(c, &id, row.project, &st.chunks, model_id)?;
            c.prepare_cached("UPDATE provenance SET extra = ?2 WHERE memory_id = ?1")
                .and_then(|mut s| s.execute((&ids, &extra)))
                .ctx("recording the judge in the provenance")?;
            drop_judge_row(c, &id)?;
            taken = true;
            id
        };
        placed.insert((id, st.index), stored);
        let memory =
            find(c, &stored)?.ok_or_else(|| CoreError::internal("a stored statement vanished"))?;
        out.push(StatementSettled::Stored(Box::new(memory)));
    }
    if !taken {
        let reason = if unchecked {
            "unchecked"
        } else if repeats {
            "repeats"
        } else if duplicates {
            "duplicate"
        } else {
            "nothing_kept"
        };
        reject_pending(c, &id, reason, now)?;
    }
    Ok(Settled::Statements(out))
}

impl Store {
    /// The next batch for the statements task (B-241): one session's
    /// waiting captures in transcript order. The oldest waiting capture
    /// with no attempt names the project and session; that session's
    /// captures with no attempt follow in the order they were captured, up
    /// to `max_items` and, after the first, `max_chars` characters of
    /// content and context together. When no capture is fresh, the oldest
    /// capture with an attempt goes alone, as in [`Store::judge_queue`].
    /// A rejected capture is never taken.
    pub fn judge_queue_session(
        &self,
        max_items: usize,
        max_chars: usize,
    ) -> Result<Vec<JudgeItem>> {
        if max_items == 0 {
            return Ok(Vec::new());
        }
        self.read("reading the judge queue by session", |tx| {
            let head: Option<(i64, Option<String>)> = tx
                .prepare_cached(
                    "SELECT m.project_id, p.session_id
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     LEFT JOIN provenance p ON p.memory_id = m.id
                     WHERE m.status = 'pending' AND j.attempts = 0 AND j.rejected_at IS NULL
                     ORDER BY j.queued_at, m.id LIMIT 1",
                )
                .and_then(|mut s| s.query_row([], |r| Ok((r.get(0)?, r.get(1)?))).optional())
                .ctx("reading the judge queue by session")?;
            let Some((project, session)) = head else {
                let sql = format!(
                    "{QUEUE_SELECT}
                     WHERE m.status = 'pending' AND j.attempts > 0 AND j.attempts < ?1
                       AND j.rejected_at IS NULL
                     ORDER BY j.queued_at, m.id LIMIT 1"
                );
                let mut s = tx.prepare_cached(&sql).ctx("reading the judge queue")?;
                let retry = s
                    .query_row([i64::from(JUDGE_MAX_ATTEMPTS)], item_from_row)
                    .optional()
                    .ctx("reading the judge queue")?;
                return Ok(retry.into_iter().collect());
            };
            let sql = format!(
                "{QUEUE_SELECT}
                 WHERE m.status = 'pending' AND j.attempts = 0 AND j.rejected_at IS NULL
                   AND m.project_id = ?1 AND p.session_id IS ?2
                 ORDER BY m.created_at, m.id"
            );
            let mut s = tx
                .prepare_cached(&sql)
                .ctx("reading the judge queue by session")?;
            let mut rows = s
                .query((project, &session))
                .ctx("reading the judge queue by session")?;
            let mut out: Vec<JudgeItem> = Vec::new();
            let mut chars = 0usize;
            while let Some(r) = rows.next().ctx("reading the judge queue by session")? {
                let item = item_from_row(r).ctx("reading the judge queue by session")?;
                let size = item.content.chars().count() + item.context.chars().count();
                if !out.is_empty() && (out.len() >= max_items || chars + size > max_chars) {
                    break;
                }
                chars += size;
                out.push(item);
            }
            Ok(out)
        })
    }

    /// The statements stored, and those held as repeats or unchecked,
    /// since `since` (B-242, B-243): for `status`.
    pub fn judge_statements_since(&self, since: Timestamp) -> Result<StatementCounts> {
        self.read("counting judged statements", |tx| {
            let stored: i64 = tx
                .prepare_cached(
                    "SELECT count(*) FROM memories m JOIN provenance p ON p.memory_id = m.id
                     WHERE m.status <> 'pending' AND p.source = 'capture'
                       AND json_extract(p.extra, '$.judge.task') = 'statements'
                       AND json_extract(p.extra, '$.judge.at') >= ?1",
                )
                .and_then(|mut s| s.query_row([ts(since)], |r| r.get(0)))
                .ctx("counting stored statements")?;
            let (repeats, unchecked): (i64, i64) = tx
                .prepare_cached(
                    "SELECT coalesce(sum(j.rejected_reason = 'repeat'), 0),
                            coalesce(sum(j.rejected_reason = 'unchecked'), 0)
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     JOIN provenance p ON p.memory_id = m.id
                     WHERE m.status = 'pending' AND j.rejected_at >= ?1
                       AND json_extract(p.extra, '$.judge.task') = 'statements'",
                )
                .and_then(|mut s| s.query_row([ts(since)], |r| Ok((r.get(0)?, r.get(1)?))))
                .ctx("counting held statements")?;
            Ok(StatementCounts {
                stored: u64::try_from(stored).unwrap_or(0),
                repeats: u64::try_from(repeats).unwrap_or(0),
                unchecked: u64::try_from(unchecked).unwrap_or(0),
            })
        })
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

    /// The next batch for the judge. Captures with no attempt come first,
    /// oldest first, up to `max_items` and, after the first, `max_chars`
    /// characters of content and context together; when there are none,
    /// the oldest capture with an attempt, alone, so one whose verdict
    /// could not be applied cannot hold others back. Only pending memories
    /// with fewer than [`JUDGE_MAX_ATTEMPTS`] attempts are taken, and never
    /// a rejected one (B-190).
    pub fn judge_queue(&self, max_items: usize, max_chars: usize) -> Result<Vec<JudgeItem>> {
        if max_items == 0 {
            return Ok(Vec::new());
        }
        self.read("reading the judge queue", |tx| {
            let sql = format!(
                "{QUEUE_SELECT}
                 WHERE m.status = 'pending' AND j.attempts < ?1 AND j.rejected_at IS NULL
                 ORDER BY j.attempts > 0, j.queued_at, m.id"
            );
            let mut s = tx.prepare_cached(&sql).ctx("reading the judge queue")?;
            let mut rows = s
                .query([i64::from(JUDGE_MAX_ATTEMPTS)])
                .ctx("reading the judge queue")?;
            let mut out: Vec<JudgeItem> = Vec::new();
            let mut chars = 0usize;
            while let Some(r) = rows.next().ctx("reading the judge queue")? {
                let item = item_from_row(r).ctx("reading the judge queue")?;
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

    /// How many captures wait, since when, and how many were rejected.
    pub fn judge_queue_stats(&self) -> Result<JudgeQueueStats> {
        self.read("counting pending captures", |tx| {
            let (pending, rejected): (i64, i64) = tx
                .query_row(
                    "SELECT count(*) - count(j.rejected_at), count(j.rejected_at)
                     FROM memories m LEFT JOIN capture_judge j ON j.memory_id = m.id
                     WHERE m.status = 'pending'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ctx("counting pending captures")?;
            let (queued, oldest): (i64, Option<i64>) = tx
                .query_row(
                    "SELECT count(*), min(j.queued_at)
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     WHERE m.status = 'pending' AND j.attempts < ?1 AND j.rejected_at IS NULL",
                    [i64::from(JUDGE_MAX_ATTEMPTS)],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ctx("counting queued captures")?;
            let failed: i64 = tx
                .query_row(
                    "SELECT count(*)
                     FROM capture_judge j JOIN memories m ON m.id = j.memory_id
                     WHERE m.status = 'pending' AND j.attempts >= ?1 AND j.rejected_at IS NULL",
                    [i64::from(JUDGE_MAX_ATTEMPTS)],
                    |r| r.get(0),
                )
                .ctx("counting failed captures")?;
            Ok(JudgeQueueStats {
                pending: u64::try_from(pending).unwrap_or(0),
                queued: u64::try_from(queued).unwrap_or(0),
                oldest: oldest.map(from_ts),
                rejected: u64::try_from(rejected).unwrap_or(0),
                failed: u64::try_from(failed).unwrap_or(0),
            })
        })
    }

    /// Marks each capture in `ids` failed judgement (B-206): the judge gave
    /// two bad answers about it alone. Its attempts are raised to
    /// [`JUDGE_MAX_ATTEMPTS`], so it leaves the queue and stays pending,
    /// never served, until [`Store::prune_pending`] deletes it
    /// `pending_max_days` after now. It is counted in
    /// [`JudgeQueueStats::failed`].
    pub fn mark_failed_judgement(&self, ids: &[MemoryId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = ts(Timestamp::now());
        self.write("marking a capture failed judgement", |tx| {
            let mut s = tx
                .prepare_cached(
                    "UPDATE capture_judge SET attempts = max(attempts, ?3), failed_at = ?2
                     WHERE memory_id = ?1",
                )
                .ctx("marking a capture failed judgement")?;
            for id in ids {
                s.execute((id.to_string(), now, i64::from(JUDGE_MAX_ATTEMPTS)))
                    .ctx("marking a capture failed judgement")?;
            }
            Ok(())
        })
    }

    /// Counts an attempt against each capture in `ids` whose verdict could
    /// not be applied (unquoted, too long, unchunked or to be judged again),
    /// and records when, for [`Store::prune_pending`] (B-178). A bad answer
    /// no longer counts here: its batch is halved (B-206).
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
    /// the capture is rejected instead ([`Settled::Duplicate`]). A rejected
    /// capture keeps its own content and stays pending, and a verdict on a
    /// capture already rejected changes nothing (B-190).
    pub fn settle_captures(&self, verdicts: Vec<Settle>) -> Result<Vec<Settled>> {
        let model = self.configured_model_id();
        let now = Timestamp::now();
        self.write("settling judged captures", |tx| {
            let mut placed = HashMap::new();
            verdicts
                .into_iter()
                .map(|v| settle_one(tx, v, model.as_deref(), now, &mut placed))
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
    /// its memory), the captures rejected more than
    /// [`JUDGE_REJECTED_KEEP_DAYS`] days before `now` (B-190), and judge
    /// usage older than [`JUDGE_USAGE_RETENTION_DAYS`] (B-178).
    ///
    /// A capture in the queue that the judge has not given up on is never
    /// pruned, however long the judge was off, unavailable or out of calls:
    /// only a capture the judge gave up on is, one with [`JUDGE_MAX_ATTEMPTS`]
    /// attempts (marked failed judgement, B-206, or with that many verdicts
    /// that could not be applied), and its wait starts at its last failure. A capture pending from before
    /// schema version 4 has no queue row and the judge never takes it; it
    /// waits from when it was captured or from the upgrade
    /// ([`JUDGE_SINCE_KEY`]), whichever is later, so the upgrade itself
    /// deletes nothing. Returns the captures deleted.
    pub fn prune_pending(&self, now: Timestamp, max_days: u32) -> Result<u64> {
        let cutoff = ts(now).saturating_sub(i64::from(max_days) * MICROS_PER_DAY);
        let rejected_cutoff = ts(now).saturating_sub(JUDGE_REJECTED_KEEP_DAYS * MICROS_PER_DAY);
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
                                   WHERE rejected_at IS NULL AND attempts >= ?3
                                     AND coalesce(failed_at, queued_at) < ?1)
                         OR id IN (SELECT memory_id FROM capture_judge
                                   WHERE rejected_at < ?4))",
                    (
                        cutoff,
                        JUDGE_SINCE_KEY,
                        i64::from(JUDGE_MAX_ATTEMPTS),
                        rejected_cutoff,
                    ),
                )
                .ctx("pruning pending captures")?;
            tx.execute("DELETE FROM judge_usage WHERE day < ?1", [&usage_cutoff])
                .ctx("pruning judge usage")?;
            Ok(n as u64)
        })
    }
}
