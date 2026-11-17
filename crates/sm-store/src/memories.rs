//! Memories: insert with deduplication, revise, supersede, merge, forget,
//! get and list (SPEC section 4; decisions B-29, B-30, B-31).

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, Row, params_from_iter};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::hash::content_hash;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{ForgetFilter, RememberOutcome};
use sm_core::time::Timestamp;
use sm_core::traits::ChunkPiece;
use sm_core::types::{
    AnchorRef, Memory, MemoryType, NewMemory, Origin, Provenance, Scope, Status, check_importance,
    normalize_tags,
};

use crate::Store;
use crate::codec::{
    check_vector, count, from_ts, memory_id_col, opt_json_col, parse_col, parse_opt_col, sql_limit,
    tags_col, tags_json, ts, vector_to_blob,
};
use crate::error::Ctx;
use crate::projects::require_project;

/// Who is changing memories. Shared memories (origin `shared`) belong to the
/// repository's shared folder: only `sync` may change or delete them (SPEC
/// section 4, Forgetting cascades; section 9, Team sharing rule 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// A tool call or CLI command. Shared memories are read-only.
    Local,
    /// `sm-share`'s `sync`, which applies the shared folder's contents.
    Sync,
}

/// One chunk to store with a memory.
#[derive(Debug, Clone, PartialEq)]
pub struct NewChunk {
    /// 0-based position; the chunks of a memory are numbered 0, 1, 2, …
    pub seq: u32,
    pub text: String,
    pub token_count: u32,
    /// The unit-length embedding, or `None` while it waits on the bulk queue
    /// (decision B-27). Embedded chunks are stamped with the configured
    /// model ([`Store::configure_index`]).
    pub embedding: Option<Vec<f32>>,
}

impl NewChunk {
    /// A chunk whose embedding is still to come.
    pub fn pending(piece: ChunkPiece) -> Self {
        NewChunk {
            seq: piece.seq,
            text: piece.text,
            token_count: piece.token_count,
            embedding: None,
        }
    }

    /// A chunk with its embedding.
    pub fn embedded(piece: ChunkPiece, embedding: Vec<f32>) -> Self {
        NewChunk {
            seq: piece.seq,
            text: piece.text,
            token_count: piece.token_count,
            embedding: Some(embedding),
        }
    }
}

/// A memory and its chunks, for [`Store::insert_memories`].
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryInput {
    pub memory: NewMemory,
    pub chunks: Vec<NewChunk>,
}

/// What happened to one [`MemoryInput`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inserted {
    /// The new memory, or the existing one for a duplicate.
    pub id: MemoryId,
    /// `duplicate` when the id already existed, or the project already held
    /// the same content hash (decision B-29). Nothing was stored then.
    pub outcome: RememberOutcome,
    /// Chunks stored (0 for a duplicate).
    pub chunks: u32,
    /// The memory this one superseded, now `superseded`.
    pub superseded: Option<MemoryId>,
}

/// New content for [`Revision`]; every chunk is replaced.
#[derive(Debug, Clone, PartialEq)]
pub struct ContentChange {
    pub content: String,
    pub chunks: Vec<NewChunk>,
}

/// Changes to one memory, applied in one transaction by
/// [`Store::revise_memory`]. `None` leaves a field as it is.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Revision {
    pub content: Option<ContentChange>,
    pub title: Option<String>,
    pub memory_type: Option<MemoryType>,
    pub status: Option<Status>,
    /// Replaces the tag list.
    pub tags: Option<Vec<String>>,
    /// Replaces the anchor list.
    pub anchors: Option<Vec<AnchorRef>>,
    pub importance: Option<f64>,
}

/// What `sync` last read from a memory's shared file (SPEC 9, "Change
/// detection before reading"; decision B-78). Stored in the memory's
/// provenance: `file_path`, `commit_sha`, and
/// `extra.sync = {"blob": …, "text_hash": …}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncMark {
    /// The file's path from the repository root, with `/` separators.
    pub file_path: String,
    /// The commit the file was read at.
    pub commit_sha: Option<String>,
    /// Git's blob id of the file as read.
    pub blob_id: String,
    /// The hash of the title and body that decides re-embedding.
    pub text_hash: String,
}

impl SyncMark {
    /// The mark `sync` recorded on a memory, or `None` if it has none.
    pub fn of(memory: &Memory) -> Option<SyncMark> {
        let p = &memory.provenance;
        mark_from(
            p.file_path.as_deref(),
            p.commit_sha.as_deref(),
            p.extra.as_ref(),
        )
    }
}

/// The mark in a provenance row's parts ([`SyncMark::of`]).
fn mark_from(
    file_path: Option<&str>,
    commit_sha: Option<&str>,
    extra: Option<&serde_json::Value>,
) -> Option<SyncMark> {
    let sync = extra?.get("sync")?;
    Some(SyncMark {
        file_path: file_path?.to_owned(),
        commit_sha: commit_sha.map(str::to_owned),
        blob_id: sync.get("blob")?.as_str()?.to_owned(),
        text_hash: sync.get("text_hash")?.as_str()?.to_owned(),
    })
}

/// Provenance `extra.unresolved_supersedes`, when it is a memory id.
fn unresolved_in(extra: Option<&serde_json::Value>) -> Option<MemoryId> {
    MemoryId::parse(extra?.get(UNRESOLVED_SUPERSEDES)?.as_str()?).ok()
}

/// Provenance `extra.merged_into` is present and not null (B-73).
fn merged_in(extra: Option<&serde_json::Value>) -> bool {
    extra
        .and_then(|e| e.get("merged_into"))
        .is_some_and(|v| !v.is_null())
}

/// Provenance `extra` key under which `sm-share`'s `sync` keeps a
/// `supersedes` target that was not in the store when its file was added
/// (SPEC 9, "Supersedes"). It counts as a supersession for the liveness
/// rule (B-87) as `supersedes_id` does.
pub const UNRESOLVED_SUPERSEDES: &str = "unresolved_supersedes";

/// Options for [`Store::insert_memories_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InsertOptions {
    /// Leave `updated_at` as it is on a memory that an inserted memory
    /// supersedes: only its status changes, not its content. For an import,
    /// so that an export, import and export again gives the same bytes
    /// (B-92, B-93). Off by default: the target is stamped with the time of
    /// the insert, as [`Store::insert_memories`] does.
    pub keep_target_updated_at: bool,
}

/// Options for [`Store::revise_memory_with`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReviseOptions {
    /// Record this sync mark in the same transaction, as
    /// [`Store::record_sync`] does (B-90, B-93): the memory becomes shared
    /// and its provenance names the file. Allowed only under
    /// [`Access::Sync`].
    pub sync_mark: Option<SyncMark>,
    /// Leave `updated_at` as it is. For an import's status restore (B-92).
    pub keep_updated_at: bool,
}

/// What `sm-share`'s `sync` needs of one memory before it reads any file
/// (B-93): [`Store::list_marks`] returns these without content, chunks or
/// anchors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkRow {
    pub id: MemoryId,
    pub status: Status,
    pub origin: Origin,
    pub content_hash: String,
    pub supersedes_id: Option<MemoryId>,
    /// Provenance `extra.unresolved_supersedes`, when it is a string that
    /// parses as a memory id ([`UNRESOLVED_SUPERSEDES`]).
    pub unresolved_supersedes: Option<MemoryId>,
    /// Consolidation merged it into another memory: provenance
    /// `extra.merged_into` is present and not null (B-73).
    pub merged: bool,
    /// Provenance `file_path`.
    pub file_path: Option<String>,
    /// The mark `sync` recorded, as [`SyncMark::of`] reads it from the
    /// whole memory.
    pub mark: Option<SyncMark>,
}

/// How many memories a filtered `forget` matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ForgetCount {
    /// Memories that are (or would be) deleted.
    pub matched: u64,
    /// Shared memories that match but are left alone (local access only).
    pub shared_skipped: u64,
}

/// Which memories [`Store::list_memories`] returns, in id order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MemoryListFilter {
    /// `None` lists every project.
    pub project: Option<ProjectId>,
    /// Empty means every type.
    pub types: Vec<MemoryType>,
    /// Empty means every status.
    pub statuses: Vec<Status>,
    /// Memories carrying every one of these tags.
    pub tags: Vec<String>,
    pub anchor: Option<AnchorRef>,
    pub origin: Option<Origin>,
    /// Keyset pagination: only ids after this one.
    pub after: Option<MemoryId>,
    /// Most memories returned; 0 means no limit.
    pub limit: usize,
}

pub(crate) fn shared_read_only(id: &MemoryId) -> CoreError {
    CoreError::new(
        ErrorCode::SharedReadOnly,
        format!("memory {id} is shared and cannot be changed or forgotten locally"),
    )
    .with_next_action(
        "Edit or delete its file in the repository's .spatial-memory/shared folder, \
         then run sync.",
    )
}

pub(crate) fn memory_not_found(id: &MemoryId) -> CoreError {
    CoreError::not_found(format!("memory {id} does not exist"))
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

const MEMORY_SELECT: &str = "SELECT m.id, m.project_id, m.type, m.status, m.title, m.content, \
     m.content_hash, m.supersedes_id, m.origin, m.importance, m.tags, m.imported, m.created_at, \
     m.updated_at, p.source, p.author, p.session_id, p.commit_sha, p.file_path, p.tool, p.extra \
     FROM memories m JOIN provenance p ON p.memory_id = m.id";

fn memory_from_row(r: &Row<'_>) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: memory_id_col(r, 0)?,
        project_id: ProjectId(r.get(1)?),
        memory_type: parse_col(r, 2)?,
        status: parse_col(r, 3)?,
        title: r.get(4)?,
        content: r.get(5)?,
        content_hash: r.get(6)?,
        supersedes_id: parse_opt_col(r, 7)?,
        origin: parse_col(r, 8)?,
        importance: r.get(9)?,
        tags: tags_col(r, 10)?,
        imported: opt_json_col(r, 11)?,
        created_at: from_ts(r.get(12)?),
        updated_at: from_ts(r.get(13)?),
        provenance: Provenance {
            source: parse_col(r, 14)?,
            author: r.get(15)?,
            session_id: r.get(16)?,
            commit_sha: r.get(17)?,
            file_path: r.get(18)?,
            tool: r.get(19)?,
            extra: opt_json_col(r, 20)?,
        },
        anchors: Vec::new(),
    })
}

pub(crate) fn anchors_of(c: &Connection, id: &MemoryId) -> Result<Vec<AnchorRef>> {
    let mut s = c
        .prepare_cached(
            "SELECT a.kind, a.name FROM memory_anchors ma JOIN anchors a ON a.id = ma.anchor_id
             WHERE ma.memory_id = ?1 ORDER BY ma.position",
        )
        .ctx("reading anchors")?;
    let rows = s
        .query_map([id.to_string()], |r| {
            Ok(AnchorRef {
                kind: parse_col(r, 0)?,
                name: r.get(1)?,
            })
        })
        .ctx("reading anchors")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading anchors")
}

/// One memory with its provenance and anchors.
pub(crate) fn find(c: &Connection, id: &MemoryId) -> Result<Option<Memory>> {
    let m = c
        .prepare_cached(&format!("{MEMORY_SELECT} WHERE m.id = ?1"))
        .and_then(|mut s| s.query_row([id.to_string()], memory_from_row).optional())
        .ctx("reading a memory")?;
    match m {
        Some(mut m) => {
            m.anchors = anchors_of(c, &m.id)?;
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

fn query_memories(c: &Connection, sql: &str, params: Vec<Value>) -> Result<Vec<Memory>> {
    let mut s = c.prepare_cached(sql).ctx("listing memories")?;
    let rows = s
        .query_map(params_from_iter(params), memory_from_row)
        .ctx("listing memories")?;
    let mut out = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .ctx("listing memories")?;
    for m in &mut out {
        m.anchors = anchors_of(c, &m.id)?;
    }
    Ok(out)
}

/// Up to `limit` memories with ids after `after`, in id order.
pub(crate) fn page_after(
    c: &Connection,
    after: Option<&MemoryId>,
    limit: usize,
) -> Result<Vec<Memory>> {
    let after = after.map_or_else(String::new, MemoryId::to_string);
    query_memories(
        c,
        &format!("{MEMORY_SELECT} WHERE m.id > ?1 ORDER BY m.id LIMIT ?2"),
        vec![Value::Text(after), Value::Integer(sql_limit(limit))],
    )
}

pub(crate) fn in_list(column: &str, values: &[&str], params: &mut Vec<Value>) -> String {
    let marks: Vec<&str> = values.iter().map(|_| "?").collect();
    params.extend(values.iter().map(|v| Value::Text((*v).to_owned())));
    format!("{column} IN ({})", marks.join(", "))
}

/// `m.id IN (memories attached to the anchor)`, in the memory's project.
pub(crate) fn anchor_clause(anchor: &AnchorRef, params: &mut Vec<Value>) -> String {
    params.push(Value::Text(anchor.kind.as_str().to_owned()));
    params.push(Value::Text(anchor.name.clone()));
    "m.id IN (SELECT ma.memory_id FROM anchors a JOIN memory_anchors ma ON ma.anchor_id = a.id \
     WHERE a.kind = ? AND a.name = ? AND a.project_id = m.project_id)"
        .to_owned()
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub(crate) fn check_chunks(chunks: &[NewChunk], model_id: Option<&str>) -> Result<()> {
    if chunks.is_empty() {
        return Err(CoreError::invalid_input(
            "a memory needs at least one chunk to be searchable",
        ));
    }
    for (i, c) in chunks.iter().enumerate() {
        if usize::try_from(c.seq).ok() != Some(i) {
            return Err(CoreError::invalid_input(format!(
                "chunk {i} has seq {}; chunks must be numbered 0, 1, 2, … in order",
                c.seq
            )));
        }
        if let Some(v) = &c.embedding {
            check_vector(&format!("chunk {i}'s embedding"), v)?;
            if model_id.is_none() {
                return Err(CoreError::internal(
                    "an embedded chunk was stored before the index was configured \
                     (call Store::configure_index first)",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn insert_chunks(
    c: &Connection,
    id: &MemoryId,
    project: ProjectId,
    chunks: &[NewChunk],
    model_id: Option<&str>,
) -> Result<()> {
    let mut s = c
        .prepare_cached(
            "INSERT INTO chunks(memory_id, project_id, seq, token_count, model_id, embedding, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .ctx("storing chunks")?;
    let id = id.to_string();
    for ch in chunks {
        let blob = ch.embedding.as_deref().map(vector_to_blob);
        let model = blob.as_ref().and(model_id);
        s.execute((
            &id,
            project.0,
            i64::from(ch.seq),
            i64::from(ch.token_count),
            model,
            blob,
            &ch.text,
        ))
        .ctx("storing a chunk")?;
    }
    Ok(())
}

fn check_anchors(anchors: &[AnchorRef]) -> Result<Vec<AnchorRef>> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(anchors.len());
    for a in anchors {
        let a = AnchorRef::new(a.kind, a.name.clone())?;
        if seen.insert(a.clone()) {
            out.push(a);
        }
    }
    Ok(out)
}

/// The anchor's row id in the project, creating the row if needed.
pub(crate) fn anchor_id(c: &Connection, project: ProjectId, a: &AnchorRef) -> Result<i64> {
    c.prepare_cached("INSERT OR IGNORE INTO anchors(project_id, kind, name) VALUES (?1, ?2, ?3)")
        .and_then(|mut s| s.execute((project.0, a.kind.as_str(), &a.name)))
        .ctx("recording an anchor")?;
    c.prepare_cached("SELECT id FROM anchors WHERE project_id = ?1 AND kind = ?2 AND name = ?3")
        .and_then(|mut s| s.query_row((project.0, a.kind.as_str(), &a.name), |r| r.get(0)))
        .ctx("reading an anchor")
}

fn set_anchors(
    c: &Connection,
    id: &MemoryId,
    project: ProjectId,
    anchors: &[AnchorRef],
) -> Result<()> {
    let ids = id.to_string();
    c.prepare_cached("DELETE FROM memory_anchors WHERE memory_id = ?1")
        .and_then(|mut s| s.execute([&ids]))
        .ctx("replacing anchors")?;
    for (pos, a) in anchors.iter().enumerate() {
        let aid = anchor_id(c, project, a)?;
        c.prepare_cached(
            "INSERT INTO memory_anchors(memory_id, anchor_id, position) VALUES (?1, ?2, ?3)",
        )
        .and_then(|mut s| s.execute((&ids, aid, sql_limit(pos))))
        .ctx("attaching an anchor")?;
    }
    Ok(())
}

/// The origin of a memory, or `not_found`.
fn origin_of(c: &Connection, id: &MemoryId) -> Result<Origin> {
    let o: Option<Origin> = c
        .prepare_cached("SELECT origin FROM memories WHERE id = ?1")
        .and_then(|mut s| {
            s.query_row([id.to_string()], |r| parse_col(r, 0))
                .optional()
        })
        .ctx("reading a memory")?;
    o.ok_or_else(|| memory_not_found(id))
}

/// Refuses a shared memory under local access.
fn check_writable(c: &Connection, id: &MemoryId, access: Access) -> Result<()> {
    if origin_of(c, id)? == Origin::Shared && access == Access::Local {
        return Err(shared_read_only(id));
    }
    Ok(())
}

/// The memory that makes a new one a duplicate, with its status: the same
/// id, or else the same content hash in the project. A memory that is not
/// `pending` is preferred, so a `pending` one is returned only when every
/// memory with that hash is pending.
fn existing_duplicate(
    c: &Connection,
    id: Option<&MemoryId>,
    project: ProjectId,
    hash: &str,
) -> Result<Option<(MemoryId, Status)>> {
    if let Some(id) = id {
        let found: Option<Status> = c
            .prepare_cached("SELECT status FROM memories WHERE id = ?1")
            .and_then(|mut s| {
                s.query_row([id.to_string()], |r| parse_col(r, 0))
                    .optional()
            })
            .ctx("checking for an existing memory")?;
        if let Some(status) = found {
            return Ok(Some((*id, status)));
        }
    }
    c.prepare_cached(
        "SELECT id, status FROM memories WHERE project_id = ?1 AND content_hash = ?2
         ORDER BY status = 'pending', created_at, id LIMIT 1",
    )
    .and_then(|mut s| {
        s.query_row((project.0, hash), |r| {
            Ok((memory_id_col(r, 0)?, parse_col(r, 1)?))
        })
        .optional()
    })
    .ctx("checking for a duplicate")
}

fn insert_one(
    c: &Connection,
    input: MemoryInput,
    access: Access,
    model_id: Option<&str>,
    now: Timestamp,
    opts: InsertOptions,
) -> Result<Inserted> {
    let MemoryInput { memory: m, chunks } = input;
    if m.content.trim().is_empty() {
        return Err(CoreError::invalid_input("memory content is empty"));
    }
    check_importance(m.importance)?;
    let tags = normalize_tags(&m.tags)?;
    let anchors = check_anchors(&m.anchors)?;
    check_chunks(&chunks, model_id)?;
    require_project(c, m.project_id)?;
    let hash = m
        .content_hash
        .clone()
        .unwrap_or_else(|| content_hash(&m.content));
    if let Some((existing, status)) = existing_duplicate(c, m.id.as_ref(), m.project_id, &hash)? {
        let replaces_pending = status == Status::Pending
            && m.status != Status::Pending
            && m.id.as_ref() != Some(&existing);
        if !replaces_pending {
            return Ok(Inserted {
                id: existing,
                outcome: RememberOutcome::Duplicate,
                chunks: 0,
                superseded: None,
            });
        }
        // A save of what an unjudged capture holds: the save is kept and
        // the capture deleted, so the text is stored once (B-157). Pending
        // captures never supersede, so no liveness is restored.
        c.prepare_cached(
            "DELETE FROM memories
             WHERE project_id = ?1 AND content_hash = ?2 AND status = 'pending'",
        )
        .and_then(|mut s| s.execute((m.project_id.0, &hash)))
        .ctx("replacing a pending capture")?;
    }
    if let Some(old) = &m.supersedes_id {
        check_writable(c, old, access)?;
    }
    let id = m.id.unwrap_or_else(MemoryId::new_v7);
    let ids = id.to_string();
    let created = m.created_at.unwrap_or(now);
    let updated = m.updated_at.unwrap_or(created);
    let imported = m.imported.as_ref().map(serde_json::to_string).transpose()?;
    let tags = tags_json(&tags)?;
    c.prepare_cached(
        "INSERT INTO memories(id, project_id, type, status, title, content, content_hash,
             supersedes_id, origin, importance, tags, imported, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
    )
    .and_then(|mut s| {
        s.execute(rusqlite::params![
            &ids,
            m.project_id.0,
            m.memory_type.as_str(),
            m.status.as_str(),
            &m.title,
            &m.content,
            &hash,
            m.supersedes_id.map(|s| s.to_string()),
            m.origin.as_str(),
            m.importance,
            &tags,
            imported,
            ts(created),
            ts(updated),
        ])
    })
    .ctx("storing a memory")?;
    insert_provenance(c, &ids, &m.provenance)?;
    set_anchors(c, &id, m.project_id, &anchors)?;
    insert_chunks(c, &id, m.project_id, &chunks, model_id)?;
    if let Some(old) = &m.supersedes_id {
        let stamp = (!opts.keep_target_updated_at).then_some(now);
        mark_superseded(c, old, stamp)?;
    }
    Ok(Inserted {
        id,
        outcome: RememberOutcome::Created,
        chunks: u32::try_from(chunks.len()).unwrap_or(u32::MAX),
        superseded: m.supersedes_id,
    })
}

fn insert_provenance(c: &Connection, id: &str, p: &Provenance) -> Result<()> {
    let extra = p.extra.as_ref().map(serde_json::to_string).transpose()?;
    c.prepare_cached(
        "INSERT INTO provenance(memory_id, source, author, session_id, commit_sha, file_path, tool, extra)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )
    .and_then(|mut s| {
        s.execute((
            id,
            p.source.as_str(),
            &p.author,
            &p.session_id,
            &p.commit_sha,
            &p.file_path,
            &p.tool,
            extra,
        ))
    })
    .ctx("storing provenance")?;
    Ok(())
}

/// Sets `status = superseded`. The memory's own `supersedes_id` is left as it
/// is: it always means "the memory this one replaces" (SPEC 4, B-73).
/// `stamp` is the new `updated_at`, or `None` to keep it (B-92).
fn mark_superseded(c: &Connection, id: &MemoryId, stamp: Option<Timestamp>) -> Result<()> {
    let n = match stamp {
        Some(now) => c
            .prepare_cached(
                "UPDATE memories SET status = 'superseded', updated_at = ?2 WHERE id = ?1",
            )
            .and_then(|mut s| s.execute((id.to_string(), ts(now)))),
        None => c
            .prepare_cached("UPDATE memories SET status = 'superseded' WHERE id = ?1")
            .and_then(|mut s| s.execute([id.to_string()])),
    }
    .ctx("marking a memory superseded")?;
    if n == 0 {
        return Err(memory_not_found(id));
    }
    Ok(())
}

/// Records in a merged memory's provenance which memory absorbed it, as
/// `extra.merged_into` (B-73). An `extra` that is not a JSON object is kept
/// under `extra.value`.
fn record_merged_into(c: &Connection, id: &MemoryId, keep: &MemoryId) -> Result<()> {
    let n = c
        .prepare_cached(
            "UPDATE provenance SET extra = json_set(
                 CASE WHEN extra IS NULL THEN '{}'
                      WHEN json_type(extra) = 'object' THEN extra
                      ELSE json_object('value', json(extra)) END,
                 '$.merged_into', ?2)
             WHERE memory_id = ?1",
        )
        .and_then(|mut s| s.execute((id.to_string(), keep.to_string())))
        .ctx("recording a merge in provenance")?;
    if n == 0 {
        return Err(CoreError::storage(format!(
            "memory {id} has no provenance row"
        )));
    }
    Ok(())
}

/// Makes a memory shared and records the file `sync` read it from
/// ([`Store::record_sync`]).
fn apply_sync_mark(c: &Connection, id: &MemoryId, mark: &SyncMark) -> Result<()> {
    let ids = id.to_string();
    c.prepare_cached("UPDATE memories SET origin = 'shared' WHERE id = ?1")
        .and_then(|mut s| s.execute([&ids]))
        .ctx("marking a memory shared")?;
    let n = c
        .prepare_cached(
            "UPDATE provenance SET file_path = ?2, commit_sha = ?3, extra = json_set(
                 CASE WHEN extra IS NULL THEN '{}'
                      WHEN json_type(extra) = 'object' THEN extra
                      ELSE json_object('value', json(extra)) END,
                 '$.sync', json_object('blob', ?4, 'text_hash', ?5))
             WHERE memory_id = ?1",
        )
        .and_then(|mut s| {
            s.execute((
                &ids,
                &mark.file_path,
                &mark.commit_sha,
                &mark.blob_id,
                &mark.text_hash,
            ))
        })
        .ctx("recording a shared file in provenance")?;
    if n == 0 {
        return Err(CoreError::storage(format!(
            "memory {id} has no provenance row"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The liveness rule (B-87): a memory is superseded while at least one memory
// of its project that is not retracted supersedes it, through
// `supersedes_id` or provenance `extra.unresolved_supersedes`.
// ---------------------------------------------------------------------------

/// A memory, its `supersedes_id` and its provenance `extra`.
type SupersessionRow = (MemoryId, Option<MemoryId>, Option<serde_json::Value>);

/// Reads [`SupersessionRow`]s; `sql` selects those three columns.
fn supersessions(c: &Connection, sql: &str, params: &[Value]) -> Result<Vec<SupersessionRow>> {
    let mut s = c.prepare_cached(sql).ctx("reading supersessions")?;
    let rows = s
        .query_map(params_from_iter(params), |r| {
            Ok((
                memory_id_col(r, 0)?,
                parse_opt_col(r, 1)?,
                opt_json_col(r, 2)?,
            ))
        })
        .ctx("reading supersessions")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading supersessions")
}

/// Every memory the memories matching `where_` (over `memories m` and
/// `provenance p`) supersede, whether or not it exists.
fn targets_where(c: &Connection, where_: &str, params: &[Value]) -> Result<Vec<MemoryId>> {
    let rows = supersessions(
        c,
        &format!(
            "SELECT m.id, m.supersedes_id, p.extra FROM memories m
             JOIN provenance p ON p.memory_id = m.id
             WHERE ({where_}) AND (m.supersedes_id IS NOT NULL
                   OR instr(p.extra, '{UNRESOLVED_SUPERSEDES}') > 0)"
        ),
        params,
    )?;
    Ok(rows
        .into_iter()
        .flat_map(|(_, sid, extra)| sid.into_iter().chain(unresolved_in(extra.as_ref())))
        .collect())
}

/// The memories one memory supersedes.
fn targets_of(c: &Connection, id: &MemoryId) -> Result<Vec<MemoryId>> {
    targets_where(c, "m.id = ?", &[Value::Text(id.to_string())])
}

/// The first memory of `project`, in id order, that is not retracted and
/// supersedes `target`.
fn live_superseder(
    c: &Connection,
    target: &MemoryId,
    project: ProjectId,
) -> Result<Option<MemoryId>> {
    let direct: Option<MemoryId> = c
        .prepare_cached(
            "SELECT id FROM memories
             WHERE supersedes_id = ?1 AND project_id = ?2 AND status <> 'retracted'
             ORDER BY id LIMIT 1",
        )
        .and_then(|mut s| {
            s.query_row((target.to_string(), project.0), |r| memory_id_col(r, 0))
                .optional()
        })
        .ctx("reading a memory's superseders")?;
    if direct.is_some() {
        return Ok(direct);
    }
    Ok(unresolved_live(c, project)?
        .into_iter()
        .find(|(s, t)| t == target && s != target)
        .map(|(s, _)| s))
}

/// `(superseder, target)` for every memory of `project` that is not
/// retracted and records an unresolved supersedes target, in superseder id
/// order.
fn unresolved_live(c: &Connection, project: ProjectId) -> Result<Vec<(MemoryId, MemoryId)>> {
    let rows = supersessions(
        c,
        &format!(
            "SELECT m.id, NULL, p.extra FROM memories m JOIN provenance p ON p.memory_id = m.id
             WHERE m.project_id = ? AND m.status <> 'retracted'
               AND instr(p.extra, '{UNRESOLVED_SUPERSEDES}') > 0
             ORDER BY m.id"
        ),
        &[Value::Integer(project.0)],
    )?;
    Ok(rows
        .into_iter()
        .filter_map(|(s, _, extra)| unresolved_in(extra.as_ref()).map(|t| (s, t)))
        .collect())
}

/// After a superseder is forgotten or retracted (B-87): restores to
/// `active` each of `targets` that is a local memory, `superseded`, not
/// merged into another (B-73) and left with no live superseder in its
/// project. A shared target is left alone: its status comes from its file,
/// which only `sync` reads, and `sync` settles it (SPEC 9, "Supersedes").
/// A retracted target is never touched. Returns the memories restored.
fn restore_unheld(c: &Connection, targets: Vec<MemoryId>, now: Timestamp) -> Result<Vec<MemoryId>> {
    let mut seen = HashSet::new();
    let mut unresolved: HashMap<i64, HashSet<MemoryId>> = HashMap::new();
    let mut restored = Vec::new();
    for t in targets {
        if !seen.insert(t) {
            continue;
        }
        let row: Option<(i64, Option<serde_json::Value>)> = c
            .prepare_cached(
                "SELECT m.project_id, p.extra FROM memories m
                 JOIN provenance p ON p.memory_id = m.id
                 WHERE m.id = ?1 AND m.origin = 'local' AND m.status = 'superseded'",
            )
            .and_then(|mut s| {
                s.query_row([t.to_string()], |r| Ok((r.get(0)?, opt_json_col(r, 1)?)))
                    .optional()
            })
            .ctx("reading a superseded memory")?;
        let Some((project, extra)) = row else {
            continue;
        };
        if merged_in(extra.as_ref()) {
            continue;
        }
        let project = ProjectId(project);
        let direct: bool = c
            .prepare_cached(
                "SELECT EXISTS (SELECT 1 FROM memories
                 WHERE supersedes_id = ?1 AND project_id = ?2 AND status <> 'retracted')",
            )
            .and_then(|mut s| s.query_row((t.to_string(), project.0), |r| r.get(0)))
            .ctx("reading a memory's superseders")?;
        if direct {
            continue;
        }
        if let Entry::Vacant(e) = unresolved.entry(project.0) {
            e.insert(
                unresolved_live(c, project)?
                    .into_iter()
                    .map(|(_, t)| t)
                    .collect(),
            );
        }
        if unresolved.get(&project.0).is_some_and(|h| h.contains(&t)) {
            continue;
        }
        c.prepare_cached("UPDATE memories SET status = 'active', updated_at = ?2 WHERE id = ?1")
            .and_then(|mut s| s.execute((t.to_string(), ts(now))))
            .ctx("restoring a memory that nothing supersedes")?;
        restored.push(t);
    }
    Ok(restored)
}

/// After a retracted superseder of `project` becomes live again (B-87):
/// marks `superseded` each of `targets` that is a local, active memory of
/// the same project. Returns the memories changed.
fn resupersede(
    c: &Connection,
    targets: Vec<MemoryId>,
    project: ProjectId,
    now: Timestamp,
) -> Result<Vec<MemoryId>> {
    let mut changed = Vec::new();
    for t in targets {
        let n = c
            .prepare_cached(
                "UPDATE memories SET status = 'superseded', updated_at = ?3
                 WHERE id = ?1 AND project_id = ?2 AND origin = 'local' AND status = 'active'",
            )
            .and_then(|mut s| s.execute((t.to_string(), project.0, ts(now))))
            .ctx("marking a memory superseded")?;
        if n > 0 {
            changed.push(t);
        }
    }
    Ok(changed)
}

/// Builds `WHERE …` for a forget filter. `project` is ignored for scope
/// `all`.
fn forget_where(project: ProjectId, f: &ForgetFilter, params: &mut Vec<Value>) -> Result<String> {
    let any = !f.memory_type.is_empty()
        || !f.status.is_empty()
        || f.anchor.is_some()
        || f.tag.is_some()
        || f.created_before.is_some();
    if !any {
        return Err(CoreError::invalid_input(
            "a forget filter needs a criterion besides scope (type, status, anchor, tag or \
             created_before)",
        ));
    }
    let mut clauses = Vec::new();
    if f.scope.unwrap_or_default() == Scope::Project {
        clauses.push("m.project_id = ?".to_owned());
        params.push(Value::Integer(project.0));
    }
    if !f.memory_type.is_empty() {
        let v: Vec<&str> = f.memory_type.iter().map(|t| t.as_str()).collect();
        clauses.push(in_list("m.type", &v, params));
    }
    if !f.status.is_empty() {
        let v: Vec<&str> = f.status.iter().map(|t| t.as_str()).collect();
        clauses.push(in_list("m.status", &v, params));
    }
    if let Some(a) = &f.anchor {
        clauses.push(anchor_clause(a, params));
    }
    if let Some(t) = &f.tag {
        clauses.push("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE json_each.value = ?)".into());
        params.push(Value::Text(t.clone()));
    }
    if let Some(t) = f.created_before {
        clauses.push("m.created_at < ?".into());
        params.push(Value::Integer(ts(t)));
    }
    Ok(clauses.join(" AND "))
}

fn count_where(c: &Connection, where_: &str, params: &[Value]) -> Result<ForgetCount> {
    let (local, shared): (i64, i64) = c
        .prepare(&format!(
            "SELECT coalesce(sum(m.origin = 'local'), 0), coalesce(sum(m.origin = 'shared'), 0)
             FROM memories m WHERE {where_}"
        ))
        .and_then(|mut s| s.query_row(params_from_iter(params), |r| Ok((r.get(0)?, r.get(1)?))))
        .ctx("counting memories to forget")?;
    Ok(ForgetCount {
        matched: count(local),
        shared_skipped: count(shared),
    })
}

impl Store {
    /// Inserts memories with their provenance, anchors and chunks (and the
    /// chunks' keyword and vector index rows) in one transaction. An item
    /// whose id already exists, or whose content hash the project already
    /// holds, is not stored; its result is `duplicate` with the existing id
    /// (decision B-29). An item with `supersedes_id` marks that memory
    /// superseded in the same transaction. Any invalid item fails the whole
    /// batch and nothing is stored.
    pub fn insert_memories(
        &self,
        inputs: Vec<MemoryInput>,
        access: Access,
    ) -> Result<Vec<Inserted>> {
        self.insert_memories_with(inputs, access, InsertOptions::default())
    }

    /// [`Store::insert_memories`] with options: an import keeps the
    /// `updated_at` of each memory an inserted memory supersedes (B-92,
    /// B-93).
    pub fn insert_memories_with(
        &self,
        inputs: Vec<MemoryInput>,
        access: Access,
        opts: InsertOptions,
    ) -> Result<Vec<Inserted>> {
        let model = self.configured_model_id();
        let now = Timestamp::now();
        self.write("storing memories", |tx| {
            inputs
                .into_iter()
                .map(|i| insert_one(tx, i, access, model.as_deref(), now, opts))
                .collect()
        })
    }

    /// [`Store::insert_memories`] for one memory.
    pub fn insert_memory(&self, input: MemoryInput, access: Access) -> Result<Inserted> {
        let mut out = self.insert_memories(vec![input], access)?;
        out.pop()
            .ok_or_else(|| CoreError::internal("insert returned no result"))
    }

    /// Applies a revision in one transaction and returns the memory as
    /// stored. New content replaces every chunk (and their index rows) and
    /// the content hash. A shared memory is refused under local access.
    ///
    /// Under local access the liveness rule holds (B-87): a memory is
    /// superseded while at least one memory of its project that is not
    /// retracted supersedes it.
    /// - Making a memory `active` while such a memory supersedes it is
    ///   `invalid_input`, naming the superseder.
    /// - Retracting a superseder restores to `active`, in the same
    ///   transaction, each local target it leaves with no live superseder,
    ///   unless the target is retracted or merged (`extra.merged_into`).
    /// - Making a retracted superseder live again marks its local `active`
    ///   targets `superseded`.
    ///
    /// Under sync access none of this applies: `sync` settles supersessions
    /// itself (SPEC 9, "Supersedes").
    pub fn revise_memory(&self, id: &MemoryId, rev: Revision, access: Access) -> Result<Memory> {
        self.revise_memory_with(id, rev, access, &ReviseOptions::default())
    }

    /// [`Store::revise_memory`] with options: record a sync mark in the same
    /// transaction (sync access only; B-90, B-93), or keep `updated_at`
    /// (an import's status restore; B-92).
    pub fn revise_memory_with(
        &self,
        id: &MemoryId,
        rev: Revision,
        access: Access,
        opts: &ReviseOptions,
    ) -> Result<Memory> {
        if opts.sync_mark.is_some() && access != Access::Sync {
            return Err(CoreError::invalid_input(format!(
                "a sync mark for memory {id} can only be recorded under sync access"
            ))
            .with_next_action("Revise the memory without a sync mark, or run sync."));
        }
        let model = self.configured_model_id();
        let now = Timestamp::now();
        if let Some(i) = rev.importance {
            check_importance(i)?;
        }
        let tags = rev.tags.as_ref().map(normalize_tags).transpose()?;
        let anchors = rev.anchors.as_deref().map(check_anchors).transpose()?;
        if let Some(c) = &rev.content {
            if c.content.trim().is_empty() {
                return Err(CoreError::invalid_input("memory content is empty"));
            }
            check_chunks(&c.chunks, model.as_deref())?;
        }
        self.write("revising a memory", |tx| {
            check_writable(tx, id, access)?;
            let (project, before): (i64, Status) = tx
                .query_row(
                    "SELECT project_id, status FROM memories WHERE id = ?1",
                    [id.to_string()],
                    |r| Ok((r.get(0)?, parse_col(r, 1)?)),
                )
                .ctx("reading a memory")?;
            let project = ProjectId(project);
            let local = access == Access::Local;
            if local
                && rev.status == Some(Status::Active)
                && before != Status::Active
                && let Some(by) = live_superseder(tx, id, project)?
            {
                return Err(CoreError::invalid_input(format!(
                    "memory {id} cannot be made active: memory {by} supersedes it and is not \
                     retracted, and a memory stays superseded while a live memory supersedes it"
                ))
                .with_next_action(format!(
                    "Retract or forget memory {by} first, or revise memory {by} instead."
                )));
            }
            let ids = id.to_string();
            let mut changed = false;
            if let Some(t) = &rev.title {
                tx.execute("UPDATE memories SET title = ?2 WHERE id = ?1", (&ids, t))
                    .ctx("changing the title")?;
                changed = true;
            }
            if let Some(c) = &rev.content {
                tx.execute(
                    "UPDATE memories SET content = ?2, content_hash = ?3 WHERE id = ?1",
                    (&ids, &c.content, content_hash(&c.content)),
                )
                .ctx("changing the content")?;
                tx.execute("DELETE FROM chunks WHERE memory_id = ?1", [&ids])
                    .ctx("removing old chunks")?;
                insert_chunks(tx, id, project, &c.chunks, model.as_deref())?;
                changed = true;
            }
            if let Some(t) = rev.memory_type {
                tx.execute(
                    "UPDATE memories SET type = ?2 WHERE id = ?1",
                    (&ids, t.as_str()),
                )
                .ctx("changing the type")?;
                changed = true;
            }
            if let Some(s) = rev.status {
                tx.execute(
                    "UPDATE memories SET status = ?2 WHERE id = ?1",
                    (&ids, s.as_str()),
                )
                .ctx("changing the status")?;
                changed = true;
                let was_live = before != Status::Retracted;
                let is_live = s != Status::Retracted;
                if local && was_live && !is_live {
                    restore_unheld(tx, targets_of(tx, id)?, now)?;
                } else if local && !was_live && is_live {
                    resupersede(tx, targets_of(tx, id)?, project, now)?;
                }
            }
            if let Some(t) = &tags {
                tx.execute(
                    "UPDATE memories SET tags = ?2 WHERE id = ?1",
                    (&ids, tags_json(t)?),
                )
                .ctx("changing the tags")?;
                changed = true;
            }
            if let Some(i) = rev.importance {
                tx.execute(
                    "UPDATE memories SET importance = ?2 WHERE id = ?1",
                    (&ids, i),
                )
                .ctx("changing the importance")?;
                changed = true;
            }
            if let Some(a) = &anchors {
                set_anchors(tx, id, project, a)?;
                changed = true;
            }
            if changed && !opts.keep_updated_at {
                tx.execute(
                    "UPDATE memories SET updated_at = ?2 WHERE id = ?1",
                    (&ids, ts(now)),
                )
                .ctx("stamping the revision")?;
            }
            if let Some(mark) = &opts.sync_mark {
                apply_sync_mark(tx, id, mark)?;
            }
            find(tx, id)?.ok_or_else(|| memory_not_found(id))
        })
    }

    /// Applies one consolidation group (decisions B-31 and B-73): every
    /// `merge` memory becomes `superseded` and its provenance records
    /// `extra.merged_into = keep`, in one transaction. Its `supersedes_id`
    /// keeps SPEC 4's meaning and is not changed. Nothing is deleted.
    /// Returns the memories changed.
    pub fn merge_memories(
        &self,
        keep: &MemoryId,
        merge: &[MemoryId],
        access: Access,
    ) -> Result<Vec<MemoryId>> {
        let now = Timestamp::now();
        let mut unique = Vec::new();
        for m in merge {
            if m == keep {
                return Err(CoreError::invalid_input(format!(
                    "memory {m} is both kept and merged"
                )));
            }
            if !unique.contains(m) {
                unique.push(*m);
            }
        }
        self.write("merging memories", |tx| {
            origin_of(tx, keep)?;
            for m in &unique {
                check_writable(tx, m, access)?;
                mark_superseded(tx, m, Some(now))?;
                record_merged_into(tx, m, keep)?;
            }
            Ok(unique)
        })
    }

    /// For `sm-share`'s `sync` only (SPEC 9, Team sharing rules 3 and 5;
    /// decision B-78): makes a memory shared and records the file it was
    /// read from, in one transaction. Sets `origin = shared`, provenance
    /// `file_path` and `commit_sha`, and `extra.sync`; other `extra` keys
    /// are kept, and an `extra` that is not a JSON object is kept under
    /// `extra.value`. Provenance `source` is not changed. Returns the
    /// memory as stored; `not_found` if it does not exist.
    pub fn record_sync(&self, id: &MemoryId, mark: &SyncMark) -> Result<Memory> {
        self.write("recording a shared file", |tx| {
            origin_of(tx, id)?;
            apply_sync_mark(tx, id, mark)?;
            find(tx, id)?.ok_or_else(|| memory_not_found(id))
        })
    }

    /// For `sm-share`'s `sync` (B-93): every memory of `project`, in id
    /// order, with only what sync needs before it reads a file: status,
    /// origin, content hash, what it supersedes, whether it was merged, and
    /// its sync mark. No content, chunks or anchors are read.
    pub fn list_marks(&self, project: ProjectId) -> Result<Vec<MarkRow>> {
        self.read("listing sync marks", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT m.id, m.status, m.origin, m.content_hash, m.supersedes_id,
                            p.file_path, p.commit_sha, p.extra
                     FROM memories m JOIN provenance p ON p.memory_id = m.id
                     WHERE m.project_id = ?1 ORDER BY m.id",
                )
                .ctx("listing sync marks")?;
            let rows = s
                .query_map([project.0], |r| {
                    let file_path: Option<String> = r.get(5)?;
                    let commit_sha: Option<String> = r.get(6)?;
                    let extra = opt_json_col(r, 7)?;
                    Ok(MarkRow {
                        id: memory_id_col(r, 0)?,
                        status: parse_col(r, 1)?,
                        origin: parse_col(r, 2)?,
                        content_hash: r.get(3)?,
                        supersedes_id: parse_opt_col(r, 4)?,
                        unresolved_supersedes: unresolved_in(extra.as_ref()),
                        merged: merged_in(extra.as_ref()),
                        mark: mark_from(
                            file_path.as_deref(),
                            commit_sha.as_deref(),
                            extra.as_ref(),
                        ),
                        file_path,
                    })
                })
                .ctx("listing sync marks")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .ctx("listing sync marks")
        })
    }

    /// Deletes memories by id in one transaction, with everything that
    /// depends on them: provenance, chunks and their keyword and vector
    /// rows, anchor mappings, links, map points and injections. All or
    /// nothing: an unknown id is `not_found` and a shared one (under local
    /// access) is `shared_read_only`, and then nothing is deleted. Returns
    /// the number deleted.
    ///
    /// The liveness rule holds (B-87): each local memory that a forgotten
    /// memory superseded and that is left with no live superseder in its
    /// project is restored to `active` in the same transaction, unless it
    /// is retracted, merged (`extra.merged_into`) or forgotten too.
    pub fn forget(&self, ids: &[MemoryId], access: Access) -> Result<u64> {
        let now = Timestamp::now();
        self.write("forgetting memories", |tx| {
            let mut n = 0;
            let mut seen = HashSet::new();
            let mut targets = Vec::new();
            for id in ids {
                if !seen.insert(*id) {
                    continue;
                }
                check_writable(tx, id, access)?;
                targets.extend(targets_of(tx, id)?);
                n += tx
                    .prepare_cached("DELETE FROM memories WHERE id = ?1")
                    .and_then(|mut s| s.execute([id.to_string()]))
                    .ctx("forgetting a memory")?;
            }
            restore_unheld(tx, targets, now)?;
            Ok(n as u64)
        })
    }

    /// How many memories a filtered forget would delete in `project` (any
    /// project for scope `all`). Under local access shared memories are
    /// counted apart and never deleted.
    pub fn count_forget(
        &self,
        project: ProjectId,
        filter: &ForgetFilter,
        access: Access,
    ) -> Result<ForgetCount> {
        self.read("counting memories to forget", |tx| {
            let mut params = Vec::new();
            let where_ = forget_where(project, filter, &mut params)?;
            let c = count_where(tx, &where_, &params)?;
            Ok(apply_access(c, access))
        })
    }

    /// Deletes the memories a filter matches, in one transaction, and
    /// returns how many matched (counted before deleting, in the same
    /// transaction). Under local access shared memories are skipped and
    /// counted in `shared_skipped`. Targets left with no live superseder
    /// are restored as [`Store::forget`] restores them (B-87).
    pub fn forget_by_filter(
        &self,
        project: ProjectId,
        filter: &ForgetFilter,
        access: Access,
    ) -> Result<ForgetCount> {
        let now = Timestamp::now();
        self.write("forgetting memories", |tx| {
            let mut params = Vec::new();
            let mut where_ = forget_where(project, filter, &mut params)?;
            let c = apply_access(count_where(tx, &where_, &params)?, access);
            if access == Access::Local {
                where_.push_str(" AND m.origin = 'local'");
            }
            let targets = targets_where(tx, &where_, &params)?;
            let deleted = tx
                .prepare(&format!(
                    "DELETE FROM memories WHERE id IN (SELECT m.id FROM memories m WHERE {where_})"
                ))
                .and_then(|mut s| s.execute(params_from_iter(&params)))
                .ctx("forgetting memories")?;
            if deleted as u64 != c.matched {
                return Err(CoreError::internal(format!(
                    "forget counted {} memories but deleted {deleted}",
                    c.matched
                )));
            }
            restore_unheld(tx, targets, now)?;
            Ok(c)
        })
    }

    /// One memory with its provenance and anchors; `not_found` if missing.
    pub fn get_memory(&self, id: &MemoryId) -> Result<Memory> {
        self.find_memory(id)?.ok_or_else(|| memory_not_found(id))
    }

    /// One memory, or `None`.
    pub fn find_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        self.read("reading a memory", |tx| find(tx, id))
    }

    /// The memories that exist among `ids`, in the order given (duplicates
    /// and missing ids dropped).
    pub fn get_memories(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        self.read("reading memories", |tx| {
            let mut seen = HashSet::new();
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                if seen.insert(*id)
                    && let Some(m) = find(tx, id)?
                {
                    out.push(m);
                }
            }
            Ok(out)
        })
    }

    /// Memories matching a filter, in id order.
    pub fn list_memories(&self, f: &MemoryListFilter) -> Result<Vec<Memory>> {
        let mut params = Vec::new();
        let mut clauses = vec!["1".to_owned()];
        if let Some(p) = f.project {
            clauses.push("m.project_id = ?".into());
            params.push(Value::Integer(p.0));
        }
        if !f.types.is_empty() {
            let v: Vec<&str> = f.types.iter().map(|t| t.as_str()).collect();
            clauses.push(in_list("m.type", &v, &mut params));
        }
        if !f.statuses.is_empty() {
            let v: Vec<&str> = f.statuses.iter().map(|t| t.as_str()).collect();
            clauses.push(in_list("m.status", &v, &mut params));
        }
        for t in &f.tags {
            clauses
                .push("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE json_each.value = ?)".into());
            params.push(Value::Text(t.clone()));
        }
        if let Some(a) = &f.anchor {
            clauses.push(anchor_clause(a, &mut params));
        }
        if let Some(o) = f.origin {
            clauses.push("m.origin = ?".into());
            params.push(Value::Text(o.as_str().to_owned()));
        }
        if let Some(a) = &f.after {
            clauses.push("m.id > ?".into());
            params.push(Value::Text(a.to_string()));
        }
        let limit = if f.limit == 0 { -1 } else { sql_limit(f.limit) };
        params.push(Value::Integer(limit));
        let sql = format!(
            "{MEMORY_SELECT} WHERE {} ORDER BY m.id LIMIT ?",
            clauses.join(" AND ")
        );
        self.read("listing memories", |tx| query_memories(tx, &sql, params))
    }

    /// Ids of the memories attached to an anchor in a project, in id order.
    pub fn memories_for_anchor(
        &self,
        project: ProjectId,
        anchor: &AnchorRef,
    ) -> Result<Vec<MemoryId>> {
        self.read("reading an anchor's memories", |tx| {
            memories_for_anchor(tx, project, anchor)
        })
    }

    /// The number of memories across every project.
    pub fn memory_count(&self) -> Result<u64> {
        self.read("counting memories", |tx| {
            let n: i64 = tx
                .query_row("SELECT count(*) FROM memories", [], |r| r.get(0))
                .ctx("counting memories")?;
            Ok(count(n))
        })
    }

    pub(crate) fn configured_model_id(&self) -> Option<String> {
        self.configured
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|s| s.model_id.clone())
    }
}

fn apply_access(c: ForgetCount, access: Access) -> ForgetCount {
    match access {
        Access::Local => c,
        Access::Sync => ForgetCount {
            matched: c.matched + c.shared_skipped,
            shared_skipped: 0,
        },
    }
}

pub(crate) fn memories_for_anchor(
    c: &Connection,
    project: ProjectId,
    anchor: &AnchorRef,
) -> Result<Vec<MemoryId>> {
    let mut s = c
        .prepare_cached(
            "SELECT ma.memory_id FROM anchors a JOIN memory_anchors ma ON ma.anchor_id = a.id
             WHERE a.project_id = ?1 AND a.kind = ?2 AND a.name = ?3 ORDER BY ma.memory_id",
        )
        .ctx("reading an anchor's memories")?;
    let rows = s
        .query_map((project.0, anchor.kind.as_str(), &anchor.name), |r| {
            memory_id_col(r, 0)
        })
        .ctx("reading an anchor's memories")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading an anchor's memories")
}
