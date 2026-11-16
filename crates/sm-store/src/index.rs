//! The index stamp, pending embeddings and `reindex` (SPEC section 4, Model
//! stamp and Housekeeping; AT-11; decision B-27).
//!
//! `meta` records the embedding model, its revision, the dimension and the
//! chunk settings the stored chunks were built with. The daemon passes the
//! configured stamp to [`Store::configure_index`]. A new store takes it as
//! its own; an existing store whose stamp differs refuses every search,
//! vector and keyword, with `index_mismatch` until [`Store::reindex`] has
//! rebuilt the chunks and written the configured stamp.

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension};
use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::ids::{ChunkId, MemoryId};
use sm_core::protocol::{IndexStatus, ReindexReport};
use sm_core::traits::ChunkPiece;
use sm_core::types::{IndexStamp, Memory};

use crate::Store;
use crate::codec::{check_vector, count, memory_id_col, vector_to_blob};
use crate::error::Ctx;
use crate::memories::{self, NewChunk};

/// Memories re-chunked per write transaction by [`Store::reindex`].
const REINDEX_BATCH: usize = 256;

/// A chunk stored without its embedding, waiting for the bulk queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChunk {
    pub id: ChunkId,
    pub memory_id: MemoryId,
    pub seq: u32,
    pub text: String,
    pub title: Option<String>,
}

fn not_configured() -> CoreError {
    CoreError::internal("the store's index is not configured (call Store::configure_index first)")
}

/// The stamp stored in `meta`, if any.
pub(crate) fn read_stamp(c: &Connection) -> Result<Option<IndexStamp>> {
    let mut s = c
        .prepare_cached("SELECT key, value FROM meta")
        .ctx("reading the index stamp")?;
    let rows = s
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .ctx("reading the index stamp")?;
    let meta = rows
        .collect::<rusqlite::Result<HashMap<_, _>>>()
        .ctx("reading the index stamp")?;
    IndexStamp::from_meta(|k| meta.get(k).cloned())
}

fn write_stamp(c: &Connection, stamp: &IndexStamp) -> Result<()> {
    let mut s = c
        .prepare_cached(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .ctx("writing the index stamp")?;
    for (k, v) in stamp.to_meta() {
        s.execute((k, v)).ctx("writing the index stamp")?;
    }
    Ok(())
}

fn chunk_counts(c: &Connection) -> Result<(u64, u64)> {
    let total: i64 = c
        .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .ctx("counting chunks")?;
    let pending: i64 = c
        .query_row(
            "SELECT count(*) FROM chunks WHERE embedding IS NULL",
            [],
            |r| r.get(0),
        )
        .ctx("counting pending chunks")?;
    Ok((count(total), count(pending)))
}

fn status(
    c: &Connection,
    stored: Option<&IndexStamp>,
    configured: Option<&IndexStamp>,
) -> Result<IndexStatus> {
    let differences = match (stored, configured) {
        (_, None) => vec!["no embedding model is configured".to_owned()],
        (None, Some(_)) => vec!["the index has no stamp".to_owned()],
        (Some(s), Some(c)) => s.differences(c),
    };
    let (chunks, pending_embeddings) = chunk_counts(c)?;
    Ok(IndexStatus {
        matches: differences.is_empty(),
        differences,
        chunks,
        pending_embeddings,
    })
}

impl Store {
    pub(crate) fn configured_stamp(&self) -> Option<IndexStamp> {
        self.configured
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Fails with `index_mismatch` unless the stored stamp matches the
    /// configured one. Called inside every search's read transaction.
    pub(crate) fn check_index(&self, c: &Connection) -> Result<()> {
        let configured = self.configured_stamp().ok_or_else(not_configured)?;
        match read_stamp(c)? {
            Some(stored) => stored.check_matches(&configured),
            None => Err(CoreError::new(
                ErrorCode::IndexMismatch,
                "the index has no stamp",
            )),
        }
    }

    /// Sets the embedder and chunk settings this process uses. A store with
    /// no stamp yet records this one. Returns whether the stored stamp
    /// matches; when it does not, searches fail with `index_mismatch` until
    /// [`Store::reindex`] runs.
    pub fn configure_index(&self, stamp: IndexStamp) -> Result<IndexStatus> {
        if stamp.dim != EMBED_DIM {
            return Err(CoreError::config_invalid(format!(
                "the embedder has {} dimensions; the store holds {EMBED_DIM}",
                stamp.dim
            )));
        }
        if stamp.model_id.trim().is_empty() {
            return Err(CoreError::config_invalid("the embedder has no model id"));
        }
        let out = self.write("configuring the index", |tx| {
            let stored = match read_stamp(tx)? {
                Some(s) => s,
                None => {
                    write_stamp(tx, &stamp)?;
                    stamp.clone()
                }
            };
            status(tx, Some(&stored), Some(&stamp))
        })?;
        *self
            .configured
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stamp);
        Ok(out)
    }

    /// The stamp stored in the database, if one has been written.
    pub fn stored_stamp(&self) -> Result<Option<IndexStamp>> {
        self.read("reading the index stamp", |tx| read_stamp(tx))
    }

    /// Whether the index matches the configured stamp, with chunk counts.
    pub fn index_status(&self) -> Result<IndexStatus> {
        let configured = self.configured_stamp();
        self.read("reading the index status", |tx| {
            let stored = read_stamp(tx)?;
            status(tx, stored.as_ref(), configured.as_ref())
        })
    }

    /// Up to `limit` chunks waiting for an embedding, oldest first.
    pub fn pending_chunks(&self, limit: usize) -> Result<Vec<PendingChunk>> {
        self.read("reading pending chunks", |tx| {
            let mut s = tx
                .prepare_cached(
                    "SELECT c.id, c.memory_id, c.seq, c.text, m.title
                     FROM chunks c JOIN memories m ON m.id = c.memory_id
                     WHERE c.embedding IS NULL ORDER BY c.id LIMIT ?1",
                )
                .ctx("reading pending chunks")?;
            let rows = s
                .query_map([crate::codec::sql_limit(limit)], |r| {
                    Ok(PendingChunk {
                        id: ChunkId(r.get(0)?),
                        memory_id: memory_id_col(r, 1)?,
                        seq: r.get(2)?,
                        text: r.get(3)?,
                        title: r.get(4)?,
                    })
                })
                .ctx("reading pending chunks")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .ctx("reading pending chunks")
        })
    }

    /// Stores embeddings for pending chunks in one transaction and returns
    /// how many chunks were updated. Chunks that no longer exist (their
    /// memory was revised or forgotten meanwhile) are skipped. `model_id`
    /// must be the configured model, and the stored stamp must match it.
    pub fn set_embeddings(&self, model_id: &str, items: &[(ChunkId, Vec<f32>)]) -> Result<u64> {
        let configured = self.configured_stamp().ok_or_else(not_configured)?;
        if configured.model_id != model_id {
            return Err(CoreError::internal(format!(
                "embeddings from model {model_id} offered to an index configured for {}",
                configured.model_id
            )));
        }
        for (id, v) in items {
            check_vector(&format!("the embedding for chunk {id}"), v)?;
        }
        self.write("storing embeddings", |tx| {
            self.check_index(tx)?;
            let mut s = tx
                .prepare_cached(
                    "UPDATE chunks SET embedding = ?2, model_id = ?3
                     WHERE id = ?1 AND embedding IS NULL",
                )
                .ctx("storing an embedding")?;
            let mut n = 0u64;
            for (id, v) in items {
                n += s
                    .execute((id.0, vector_to_blob(v), model_id))
                    .ctx("storing an embedding")? as u64;
            }
            Ok(n)
        })
    }

    /// Rebuilds every memory's chunks with the configured settings, as
    /// chunks waiting for embedding, then writes the configured stamp
    /// (SPEC section 4, Model stamp). The database is backed up first.
    /// `rechunk` splits one memory's content; it runs outside any
    /// transaction. Searches return `index_mismatch` until the last step
    /// writes the stamp; the daemon's bulk queue then embeds the chunks.
    pub fn reindex(
        &self,
        mut rechunk: impl FnMut(&Memory) -> Result<Vec<ChunkPiece>>,
    ) -> Result<ReindexReport> {
        let stamp = self.configured_stamp().ok_or_else(not_configured)?;
        let backup = self.backup("reindex")?;
        let mut memories_done = 0u64;
        let mut chunks_queued = 0u64;
        let mut after: Option<MemoryId> = None;
        loop {
            let page = self.memories_page(after.as_ref(), REINDEX_BATCH)?;
            let Some(last) = page.last() else { break };
            after = Some(last.id);
            let mut batch = Vec::with_capacity(page.len());
            for m in &page {
                let pieces = rechunk(m)?;
                let chunks: Vec<NewChunk> = pieces.into_iter().map(NewChunk::pending).collect();
                memories::check_chunks(&chunks, None)?;
                batch.push((m.id, m.content_hash.clone(), chunks));
            }
            let (m, c) = self.write("rebuilding chunks", |tx| {
                let mut done = (0u64, 0u64);
                for (id, hash, chunks) in &batch {
                    // Revised or forgotten since the page was read: a
                    // revision already stored chunks built with the
                    // configured settings.
                    let current: Option<(String, i64)> = tx
                        .query_row(
                            "SELECT content_hash, project_id FROM memories WHERE id = ?1",
                            [id.to_string()],
                            |r| Ok((r.get(0)?, r.get(1)?)),
                        )
                        .optional()
                        .ctx("reading a memory")?;
                    let Some((current_hash, project)) = current else {
                        continue;
                    };
                    if &current_hash != hash {
                        continue;
                    }
                    tx.execute("DELETE FROM chunks WHERE memory_id = ?1", [id.to_string()])
                        .ctx("removing old chunks")?;
                    memories::insert_chunks(
                        tx,
                        id,
                        sm_core::ids::ProjectId(project),
                        chunks,
                        None,
                    )?;
                    done.0 += 1;
                    done.1 += chunks.len() as u64;
                }
                Ok(done)
            })?;
            memories_done += m;
            chunks_queued += c;
            if page.len() < REINDEX_BATCH {
                break;
            }
        }
        self.write("writing the index stamp", |tx| write_stamp(tx, &stamp))?;
        Ok(ReindexReport {
            backup: backup.display().to_string(),
            memories: memories_done,
            chunks_queued,
        })
    }
}
