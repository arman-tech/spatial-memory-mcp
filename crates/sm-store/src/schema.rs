//! The schema, its forward-only migrations and database backups (SPEC
//! section 4).
//!
//! The schema version lives in `meta`. Each migration runs in its own
//! `BEGIN IMMEDIATE` transaction and writes its version in the same
//! transaction, so a failed migration leaves the previous version intact. A
//! database that already holds a schema is copied to the backups folder
//! (`VACUUM INTO`) before the first migration runs; a new, empty database is
//! not.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior};
use sm_core::error::{CoreError, Result};
use sm_core::time::Timestamp;
use sm_core::types::meta_keys;

use crate::error::Ctx;

/// The schema version this build writes.
pub const SCHEMA_VERSION: u32 = 1;

/// One forward step.
pub(crate) struct Migration {
    pub(crate) version: u32,
    pub(crate) apply: fn(&Transaction<'_>) -> Result<()>,
}

/// Every migration, in order. The last one's version is
/// [`SCHEMA_VERSION`].
pub(crate) const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    apply: v1,
}];

/// What [`migrate`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Migrated {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) backup: Option<PathBuf>,
}

/// Brings the database up to the last migration's version.
pub(crate) fn migrate(
    conn: &mut Connection,
    migrations: &[Migration],
    backups_dir: &Path,
) -> Result<Migrated> {
    let latest = migrations.last().map_or(0, |m| m.version);
    let from = stored_version(conn)?;
    if from > latest {
        return Err(CoreError::storage(format!(
            "the database has schema version {from}; this build understands up to {latest}"
        ))
        .with_next_action("Upgrade spatial-memory, or restore a backup written by this version."));
    }
    if from == latest {
        return Ok(Migrated {
            from,
            to: from,
            backup: None,
        });
    }
    let backup = if from >= 1 {
        Some(backup_to(conn, backups_dir, &format!("schema-v{from}"))?)
    } else {
        None
    };
    let mut to = from;
    for m in migrations.iter().filter(|m| m.version > from) {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .ctx("starting a migration")?;
        // Another process may have migrated while this one waited.
        let now = stored_version(&tx)?;
        if now < m.version {
            (m.apply)(&tx)?;
            tx.execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (meta_keys::SCHEMA_VERSION, m.version.to_string()),
            )
            .ctx("recording the schema version")?;
        }
        tx.commit()
            .ctx(&format!("committing migration {}", m.version))?;
        to = m.version;
    }
    Ok(Migrated { from, to, backup })
}

/// The schema version recorded in `meta`; 0 for a database with no schema.
pub(crate) fn stored_version(conn: &Connection) -> Result<u32> {
    let has_meta: bool = conn
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'meta')",
            [],
            |r| r.get(0),
        )
        .ctx("reading the schema")?;
    if !has_meta {
        return Ok(0);
    }
    let v: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            [meta_keys::SCHEMA_VERSION],
            |r| r.get(0),
        )
        .optional()
        .ctx("reading the schema version")?;
    let v = v.ok_or_else(|| CoreError::storage("meta has no schema_version row"))?;
    v.parse()
        .map_err(|_| crate::error::corrupt("meta schema_version", &v))
}

/// Copies the database to `dir` with `VACUUM INTO`, which writes a
/// consistent, compacted snapshot without blocking readers or writers for
/// long. The file is `memory-<UTC time>-<reason>.db`; an existing file is
/// never overwritten. Must not be called inside a transaction.
pub(crate) fn backup_to(conn: &Connection, dir: &Path, reason: &str) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let stamp: String = Timestamp::now()
        .to_rfc3339()
        .chars()
        .filter(|c| *c != '-' && *c != ':')
        .collect();
    let reason: String = reason
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let base = format!("memory-{stamp}-{reason}");
    let mut path = dir.join(format!("{base}.db"));
    let mut n = 1;
    while path.exists() {
        path = dir.join(format!("{base}-{n}.db"));
        n += 1;
    }
    let target = path.to_str().ok_or_else(|| {
        CoreError::storage(format!(
            "the backup path {} is not valid UTF-8",
            path.display()
        ))
    })?;
    conn.execute("VACUUM INTO ?1", [target])
        .ctx(&format!("backing up the database to {}", path.display()))?;
    Ok(path)
}

/// Schema version 1 (SPEC section 4). Timestamps are microseconds since the
/// Unix epoch (UTC, decision B-18's precision). Tables are STRICT.
const V1_SQL: &str = r#"
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE projects (
    id         INTEGER PRIMARY KEY,
    identity   TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    git_remote TEXT,
    root_path  TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
) STRICT;
CREATE INDEX projects_name ON projects(name);

CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    type          TEXT NOT NULL CHECK (type IN ('decision', 'pitfall', 'convention', 'incident', 'runbook', 'note')),
    status        TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'retracted')),
    title         TEXT,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    supersedes_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('local', 'shared')),
    importance    REAL NOT NULL CHECK (importance >= 0.0 AND importance <= 1.0),
    tags          TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tags) AND json_type(tags) = 'array'),
    imported      TEXT CHECK (imported IS NULL OR json_valid(imported)),
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
) STRICT;
CREATE INDEX memories_project_status ON memories(project_id, status);
CREATE INDEX memories_project_hash ON memories(project_id, content_hash);
CREATE INDEX memories_supersedes ON memories(supersedes_id) WHERE supersedes_id IS NOT NULL;

CREATE TABLE provenance (
    memory_id  TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    source     TEXT NOT NULL CHECK (source IN ('agent', 'human', 'capture', 'import', 'sync')),
    author     TEXT,
    session_id TEXT,
    commit_sha TEXT,
    file_path  TEXT,
    tool       TEXT,
    extra      TEXT CHECK (extra IS NULL OR json_valid(extra))
) STRICT, WITHOUT ROWID;

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL CHECK (seq >= 0),
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    model_id    TEXT,
    embedding   BLOB CHECK (embedding IS NULL OR length(embedding) = 3072),
    text        TEXT NOT NULL,
    UNIQUE (memory_id, seq),
    CHECK ((embedding IS NULL) = (model_id IS NULL))
) STRICT;
CREATE INDEX chunks_project ON chunks(project_id);
CREATE INDEX chunks_pending ON chunks(id) WHERE embedding IS NULL;

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    title, text,
    content = '', contentless_delete = 1,
    tokenize = 'porter unicode61'
);

CREATE VIRTUAL TABLE chunk_vectors USING vec1(vector);

CREATE TABLE anchors (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL CHECK (kind IN ('project', 'service', 'database', 'table', 'endpoint', 'file', 'symbol', 'commit')),
    name       TEXT NOT NULL,
    UNIQUE (project_id, kind, name)
) STRICT;

CREATE TABLE memory_anchors (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    PRIMARY KEY (memory_id, anchor_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX memory_anchors_anchor ON memory_anchors(anchor_id);

CREATE TABLE links (
    id              INTEGER PRIMARY KEY,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    src_kind        TEXT NOT NULL CHECK (src_kind IN ('memory', 'project', 'service', 'database', 'table', 'endpoint', 'file', 'symbol', 'commit')),
    src_id          TEXT NOT NULL,
    dst_kind        TEXT NOT NULL CHECK (dst_kind IN ('memory', 'project', 'service', 'database', 'table', 'endpoint', 'file', 'symbol', 'commit')),
    dst_id          TEXT NOT NULL,
    relation        TEXT NOT NULL CHECK (relation IN ('calls', 'reads', 'writes', 'depends_on', 'relates_to', 'caused_by')),
    source          TEXT NOT NULL CHECK (source IN ('manual', 'extracted')),
    extractor       TEXT,
    verified_commit TEXT,
    verified_at     INTEGER,
    created_at      INTEGER NOT NULL,
    UNIQUE (project_id, src_kind, src_id, dst_kind, dst_id, relation)
) STRICT;
CREATE INDEX links_src ON links(src_kind, src_id);
CREATE INDEX links_dst ON links(dst_kind, dst_id);

CREATE TABLE map_runs (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    input_hash TEXT NOT NULL UNIQUE,
    params     TEXT NOT NULL CHECK (json_valid(params)),
    memories   INTEGER NOT NULL CHECK (memories >= 0),
    clustered  INTEGER NOT NULL CHECK (clustered IN (0, 1)),
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX map_runs_project ON map_runs(project_id, created_at);

CREATE TABLE map_clusters (
    run_id INTEGER NOT NULL REFERENCES map_runs(id) ON DELETE CASCADE,
    region INTEGER NOT NULL CHECK (region >= 0),
    size   INTEGER NOT NULL CHECK (size >= 0),
    label  TEXT,
    PRIMARY KEY (run_id, region)
) STRICT, WITHOUT ROWID;

CREATE TABLE map_points (
    run_id       INTEGER NOT NULL REFERENCES map_runs(id) ON DELETE CASCADE,
    memory_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    region       INTEGER,
    x            REAL NOT NULL,
    y            REAL NOT NULL,
    central_rank INTEGER CHECK (central_rank IS NULL OR central_rank >= 0),
    PRIMARY KEY (run_id, memory_id),
    FOREIGN KEY (run_id, region) REFERENCES map_clusters(run_id, region) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;
CREATE INDEX map_points_memory ON map_points(memory_id);

CREATE TABLE injections (
    session_id TEXT NOT NULL,
    memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    at         INTEGER NOT NULL,
    PRIMARY KEY (session_id, memory_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX injections_at ON injections(at);
CREATE INDEX injections_memory ON injections(memory_id);

-- The keyword and vector indexes follow chunks in the same transaction.
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, text)
        VALUES (new.id, coalesce((SELECT title FROM memories WHERE id = new.memory_id), ''), new.text);
    INSERT INTO chunk_vectors(rowid, vector)
        SELECT new.id, new.embedding WHERE new.embedding IS NOT NULL;
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
    DELETE FROM chunk_vectors WHERE rowid = old.id;
END;

CREATE TRIGGER chunks_au_embedding AFTER UPDATE OF embedding ON chunks BEGIN
    DELETE FROM chunk_vectors WHERE rowid = old.id;
    INSERT INTO chunk_vectors(rowid, vector)
        SELECT new.id, new.embedding WHERE new.embedding IS NOT NULL;
END;

CREATE TRIGGER chunks_au_text AFTER UPDATE OF text ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
    INSERT INTO chunks_fts(rowid, title, text)
        VALUES (new.id, coalesce((SELECT title FROM memories WHERE id = new.memory_id), ''), new.text);
END;

CREATE TRIGGER memories_au_title AFTER UPDATE OF title ON memories
WHEN old.title IS NOT new.title BEGIN
    DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE memory_id = new.id);
    INSERT INTO chunks_fts(rowid, title, text)
        SELECT id, coalesce(new.title, ''), text FROM chunks WHERE memory_id = new.id;
END;

-- Links name their memory ends by id, so they are removed with the memory.
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM links WHERE src_kind = 'memory' AND src_id = old.id;
    DELETE FROM links WHERE dst_kind = 'memory' AND dst_id = old.id;
END;
"#;

fn v1(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(V1_SQL).ctx("creating schema version 1")?;
    // Exact search: a flat index (every vector compared) with cosine
    // distance (SPEC section 4, Cross-project search).
    tx.execute(
        r#"INSERT INTO chunk_vectors(cmd, arg) VALUES ('rebuild', '{index:"flat", distance:"cos"}')"#,
        [],
    )
    .ctx("configuring the vec1 table")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::types::{
        AnchorKind, LinkSource, MemoryType, Origin, ProvenanceSource, Relation, Status,
    };

    fn in_list<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
        let quoted: Vec<String> = items.into_iter().map(|s| format!("'{s}'")).collect();
        format!("IN ({})", quoted.join(", "))
    }

    #[test]
    fn s4_check_constraints_list_every_enum_value() {
        assert!(V1_SQL.contains(&in_list(MemoryType::ALL.iter().map(|t| t.as_str()))));
        assert!(V1_SQL.contains(&in_list(Status::ALL.iter().map(|t| t.as_str()))));
        assert!(V1_SQL.contains(&in_list(Origin::ALL.iter().map(|t| t.as_str()))));
        assert!(V1_SQL.contains(&in_list(ProvenanceSource::ALL.iter().map(|t| t.as_str()))));
        assert!(V1_SQL.contains(&in_list(AnchorKind::ALL.iter().map(|t| t.as_str()))));
        let ends = std::iter::once("memory").chain(AnchorKind::ALL.iter().map(|t| t.as_str()));
        assert!(V1_SQL.contains(&in_list(ends)));
        assert!(V1_SQL.contains(&in_list(Relation::ALL.iter().map(|t| t.as_str()))));
        assert!(V1_SQL.contains(&in_list(LinkSource::ALL.iter().map(|t| t.as_str()))));
        assert_eq!(MIGRATIONS.last().map(|m| m.version), Some(SCHEMA_VERSION));
    }
}
