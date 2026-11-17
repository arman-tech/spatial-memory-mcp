//! Exact vector search and FTS5 keyword search (SPEC section 4, Project-scoped
//! and Cross-project search; section 6 steps 2 to 5; decisions D-2, D-3).
//!
//! Both vector scopes return exactly what a brute-force scan would: every
//! embedded chunk that passes the filters is scored with the dot product
//! (cosine, for the unit vectors the `Embedder` contract guarantees), each
//! memory keeps its best chunk, and memories are sorted by score, highest
//! first, ties broken by id.
//!
//! * **Project scope** reads `chunks WHERE project_id = ?` through its index,
//!   keeps the chunks whose memory passes the filters, and scores in Rust.
//! * **Scope `all`** asks `vec1` (flat index, exact) for the nearest chunks
//!   across every project, then re-scores and filters them in Rust. When the
//!   filters leave fewer than `limit` memories that are provably ahead of
//!   every chunk `vec1` did not return, it falls back to scoring every
//!   embedded chunk in Rust, so the answer stays exact. An anchor filter goes
//!   straight to the anchor's memories.

use std::collections::{HashMap, HashSet};

use rusqlite::types::{Value, ValueRef};
use rusqlite::{Connection, params_from_iter};
use sm_core::error::{CoreError, Result};
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::traits::{MemoryFilter, MemoryHit, QueryScope, VectorIndex, VectorQuery};

use crate::codec::{UNIT_TOLERANCE, blob_to_vector, check_vector, dot_blob, vector_to_blob};
use crate::error::{Ctx, corrupt};
use crate::memories::{anchor_clause, in_list};
use crate::{EMBEDDING_BYTES, Store};

/// `vec1` is first asked for `limit` × this many chunks …
const FIRST_K_PER_RESULT: usize = 8;
/// … and never fewer than this.
const MIN_FIRST_K: usize = 256;
/// How far a re-scored dot product may sit from `vec1`'s cosine for the
/// same chunk. Stored and query vectors are unit length within
/// `UNIT_TOLERANCE` of squared length, so the two differ by at most about
/// that much; twice it, plus float error, is a safe bound.
const SCORE_MARGIN: f32 = 2.0 * UNIT_TOLERANCE + 1e-5;

/// One memory found by keyword search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeywordHit {
    pub memory_id: MemoryId,
    /// `seq` of the memory's best-ranked matching chunk.
    pub chunk_seq: u32,
    /// FTS5 `bm25` rank of that chunk: lower is better.
    pub rank: f64,
}

#[derive(Debug, Clone, Copy)]
struct Best {
    score: f32,
    seq: u32,
}

/// Keeps each memory's best chunk: the highest score, then the lowest seq.
fn collapse(best: &mut HashMap<String, Best>, memory_id: &str, seq: u32, score: f32) {
    if let Some(b) = best.get_mut(memory_id) {
        if score > b.score || (score == b.score && seq < b.seq) {
            *b = Best { score, seq };
        }
    } else {
        best.insert(memory_id.to_owned(), Best { score, seq });
    }
}

/// Highest score first, then id: a total order, so the first `limit` hits
/// are the same however they are found.
fn by_rank(a: &MemoryHit, b: &MemoryHit) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.memory_id.cmp(&b.memory_id))
}

/// The first `limit` memories in rank order ([`by_rank`]), excluded ones
/// dropped. Only those `limit` are sorted (decision B-75): a project of
/// 50,000 memories is partitioned once rather than sorted whole.
fn ranked(
    best: HashMap<String, Best>,
    exclude: &HashSet<MemoryId>,
    limit: usize,
) -> Result<Vec<MemoryHit>> {
    let mut hits = Vec::with_capacity(best.len());
    for (id, b) in best {
        let memory_id: MemoryId = id
            .parse()
            .map_err(|e: CoreError| corrupt("chunk memory id", e.message))?;
        if !exclude.contains(&memory_id) {
            hits.push(MemoryHit {
                memory_id,
                score: b.score,
                chunk_seq: b.seq,
            });
        }
    }
    if limit == 0 {
        return Ok(Vec::new());
    }
    if hits.len() > limit {
        hits.select_nth_unstable_by(limit - 1, by_rank);
        hits.truncate(limit);
    }
    hits.sort_by(by_rank);
    Ok(hits)
}

/// ` AND …` clauses on `m` (memories) for a filter; `exclude` is applied in
/// Rust after collapsing.
fn filter_sql(f: &MemoryFilter, params: &mut Vec<Value>) -> String {
    let mut sql = String::new();
    // `pending` is dropped here as well as in the tools, so no search can
    // serve an unjudged capture (B-157).
    let served = f.served_statuses();
    let statuses: Vec<&str> = served.iter().map(|s| s.as_str()).collect();
    sql.push_str(" AND ");
    if statuses.is_empty() {
        sql.push('0');
    } else {
        sql.push_str(&in_list("m.status", &statuses, params));
    }
    if !f.types.is_empty() {
        let types: Vec<&str> = f.types.iter().map(|t| t.as_str()).collect();
        sql.push_str(" AND ");
        sql.push_str(&in_list("m.type", &types, params));
    }
    if let Some(min) = f.min_importance {
        sql.push_str(" AND m.importance >= ?");
        params.push(Value::Real(min));
    }
    if let Some(a) = &f.anchor {
        sql.push_str(" AND ");
        sql.push_str(&anchor_clause(a, params));
    }
    sql
}

fn chunk_seq(seq: i64) -> Result<u32> {
    u32::try_from(seq).map_err(|_| corrupt("chunk seq", seq))
}

/// Scores every embedded chunk matching `where_` (clauses on `c` and `m`)
/// and keeps each memory's best chunk.
fn scan(
    c: &Connection,
    where_: &str,
    params: &[Value],
    q: &[f32],
) -> Result<HashMap<String, Best>> {
    let sql = format!(
        "SELECT c.memory_id, c.seq, c.embedding FROM chunks c JOIN memories m ON m.id = c.memory_id
         WHERE c.embedding IS NOT NULL{where_}"
    );
    score_rows(c, &sql, params, q)
}

/// Scores the rows of `sql`, which selects `memory_id, seq, embedding` of
/// chunks, and keeps each memory's best chunk.
fn score_rows(
    c: &Connection,
    sql: &str,
    params: &[Value],
    q: &[f32],
) -> Result<HashMap<String, Best>> {
    let mut s = c.prepare_cached(sql).ctx("searching vectors")?;
    let mut rows = s.query(params_from_iter(params)).ctx("searching vectors")?;
    let mut best = HashMap::new();
    while let Some(r) = rows.next().ctx("searching vectors")? {
        let blob = match r.get_ref(2).ctx("reading a vector")? {
            ValueRef::Blob(b) if b.len() == EMBEDDING_BYTES => b,
            other => {
                return Err(corrupt(
                    "chunk embedding",
                    format!("{:?} value", other.data_type()),
                ));
            }
        };
        let score = dot_blob(blob, q);
        let seq = chunk_seq(r.get(1).ctx("reading a chunk")?)?;
        let id = match r.get_ref(0).ctx("reading a chunk")? {
            ValueRef::Text(t) => {
                std::str::from_utf8(t).map_err(|e| corrupt("chunk memory id", e))?
            }
            _ => return Err(corrupt("chunk memory id", "not text")),
        };
        collapse(&mut best, id, seq, score);
    }
    Ok(best)
}

/// The `k` nearest chunks by `vec1`'s cosine distance, nearest first.
fn knn(c: &Connection, q_blob: &[u8], k: usize) -> Result<Vec<(i64, f64)>> {
    let mut s = c
        .prepare_cached("SELECT rowid, distance FROM chunk_vectors(?1, ?2)")
        .ctx("searching all projects")?;
    let rows = s
        .query_map((q_blob, format!("{{K: {k}}}")), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })
        .ctx("searching all projects")?;
    let mut out = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .ctx("searching all projects")?;
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    Ok(out)
}

/// Project scope: every embedded chunk of the project whose memory passes
/// the filters.
///
/// The memories that pass are listed once, from `memories_project_status`,
/// and each chunk read through `chunks_project` is tested against that list,
/// rather than joined to its memory row one lookup at a time (decision B-75:
/// at 100,000 chunks in one project, readers mapping the file, a search takes
/// 67.9 ms at the median against 72.9 ms with the join). This relies on a
/// chunk's `project_id` being its memory's: `insert_chunks` copies it from
/// the memory, and nothing in the store changes either afterwards.
fn search_project(
    c: &Connection,
    project: ProjectId,
    q: &[f32],
    f: &MemoryFilter,
    exclude: &HashSet<MemoryId>,
    limit: usize,
) -> Result<Vec<MemoryHit>> {
    let mut params = vec![Value::Integer(project.0), Value::Integer(project.0)];
    let filters = filter_sql(f, &mut params);
    let sql = format!(
        "SELECT c.memory_id, c.seq, c.embedding FROM chunks c
         WHERE c.embedding IS NOT NULL AND c.project_id = ?
           AND c.memory_id IN (SELECT m.id FROM memories m WHERE m.project_id = ?{filters})"
    );
    let best = score_rows(c, &sql, &params, q)?;
    ranked(best, exclude, limit)
}

fn search_all(
    c: &Connection,
    q: &[f32],
    f: &MemoryFilter,
    exclude: &HashSet<MemoryId>,
    limit: usize,
) -> Result<Vec<MemoryHit>> {
    let mut params = Vec::new();
    let filters = filter_sql(f, &mut params);
    if f.anchor.is_some() {
        // The anchor's memories are few: score their chunks directly.
        let best = scan(c, &filters, &params, q)?;
        return ranked(best, exclude, limit);
    }
    let k = limit
        .saturating_add(exclude.len())
        .saturating_mul(FIRST_K_PER_RESULT)
        .max(MIN_FIRST_K);
    let near = knn(c, &vector_to_blob(q), k)?;
    let ids: Vec<i64> = near.iter().map(|(id, _)| *id).collect();
    let mut cand_params = vec![Value::Text(
        serde_json::to_string(&ids).map_err(CoreError::from)?,
    )];
    cand_params.extend(params.iter().cloned());
    let best = scan(
        c,
        &format!(" AND c.id IN (SELECT value FROM json_each(?)){filters}"),
        &cand_params,
        q,
    )?;
    let hits = ranked(best, exclude, limit)?;
    if near.len() < k {
        // vec1 returned every stored vector: the candidates are complete.
        return Ok(hits);
    }
    // Every chunk vec1 left out is at least as far as the farthest it
    // returned, so its score is at most `boundary`.
    let boundary = near.last().map_or(f32::INFINITY, |(_, d)| 1.0 - *d as f32) + SCORE_MARGIN;
    if hits.len() >= limit && hits[limit - 1].score > boundary {
        return Ok(hits);
    }
    // The filters removed too many candidates: score everything.
    let best = scan(c, &filters, &params, q)?;
    ranked(best, exclude, limit)
}

/// One vector per memory matching `where_`: the mean of its embedded chunk
/// vectors, re-normalized (SPEC section 7, Pipeline step 1), by memory id.
/// Memories with no embedded chunk yet are left out.
fn mean_vectors(
    c: &Connection,
    where_: &str,
    params: &[Value],
) -> Result<Vec<(MemoryId, Vec<f32>)>> {
    fn finish(id: MemoryId, sum: &[f64], first: Vec<f32>) -> (MemoryId, Vec<f32>) {
        let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-6 {
            // Chunks that cancel out have no mean direction; the first
            // chunk stands for the memory.
            return (id, first);
        }
        // Each component of a unit vector fits f32.
        #[allow(clippy::cast_possible_truncation)]
        let v = sum.iter().map(|x| (x / norm) as f32).collect();
        (id, v)
    }

    let sql = format!(
        "SELECT c.memory_id, c.embedding FROM chunks c JOIN memories m ON m.id = c.memory_id
         WHERE c.embedding IS NOT NULL{where_} ORDER BY c.memory_id, c.seq"
    );
    let mut s = c.prepare_cached(&sql).ctx("reading memory vectors")?;
    let mut rows = s
        .query(params_from_iter(params))
        .ctx("reading memory vectors")?;
    let mut out = Vec::new();
    let mut current: Option<(MemoryId, Vec<f64>, Vec<f32>)> = None;
    while let Some(r) = rows.next().ctx("reading memory vectors")? {
        let id: MemoryId = r
            .get::<_, String>(0)
            .ctx("reading a chunk")?
            .parse()
            .map_err(|e: CoreError| corrupt("chunk memory id", e.message))?;
        let v = match r.get_ref(1).ctx("reading a vector")? {
            ValueRef::Blob(b) => blob_to_vector(b)?,
            other => {
                return Err(corrupt(
                    "chunk embedding",
                    format!("{:?} value", other.data_type()),
                ));
            }
        };
        match &mut current {
            Some((cur, sum, _)) if *cur == id => {
                for (s, x) in sum.iter_mut().zip(&v) {
                    *s += f64::from(*x);
                }
            }
            _ => {
                if let Some((done, sum, first)) = current.take() {
                    out.push(finish(done, &sum, first));
                }
                let sum = v.iter().map(|x| f64::from(*x)).collect();
                current = Some((id, sum, v));
            }
        }
    }
    if let Some((done, sum, first)) = current {
        out.push(finish(done, &sum, first));
    }
    Ok(out)
}

/// The FTS5 query for `text`: one phrase, with any `"` doubled, so
/// identifiers keep their tokens adjacent and nothing in the text is read
/// as query syntax. `None` for blank text.
pub(crate) fn fts_phrase(text: &str) -> Option<String> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    Some(format!("\"{}\"", t.replace('"', "\"\"")))
}

impl Store {
    /// Exact nearest-memory search; see the module documentation.
    /// `index_mismatch` until a stored index matches the configured one.
    pub fn nearest(&self, q: &VectorQuery<'_>) -> Result<Vec<MemoryHit>> {
        check_vector("the query vector", q.vector)?;
        let exclude: HashSet<MemoryId> = q.filter.exclude.iter().copied().collect();
        self.read("searching vectors", |tx| {
            self.check_index(tx)?;
            if q.limit == 0 {
                return Ok(Vec::new());
            }
            match q.scope {
                QueryScope::Project(p) => {
                    search_project(tx, p, q.vector, &q.filter, &exclude, q.limit)
                }
                QueryScope::All => search_all(tx, q.vector, &q.filter, &exclude, q.limit),
            }
        })
    }

    /// One vector per memory for the map (SPEC section 7, Pipeline step 1):
    /// the mean of the memory's chunk vectors, re-normalized, by memory id.
    /// Memories whose chunks all wait for an embedding are left out. The
    /// filters are those of vector search. `index_mismatch` until a stored
    /// index matches the configured one.
    pub fn memory_vectors(
        &self,
        scope: QueryScope,
        filter: &MemoryFilter,
    ) -> Result<Vec<(MemoryId, Vec<f32>)>> {
        let exclude: HashSet<MemoryId> = filter.exclude.iter().copied().collect();
        self.read("reading memory vectors", |tx| {
            self.check_index(tx)?;
            let mut params = Vec::new();
            let mut where_ = String::new();
            if let QueryScope::Project(p) = scope {
                where_.push_str(" AND c.project_id = ?");
                params.push(Value::Integer(p.0));
            }
            where_.push_str(&filter_sql(filter, &mut params));
            let mut out = mean_vectors(tx, &where_, &params)?;
            out.retain(|(id, _)| !exclude.contains(id));
            Ok(out)
        })
    }

    /// One memory's vector, as [`Store::memory_vectors`] computes it, whatever
    /// its status; `None` while all its chunks wait for an embedding.
    /// `not_found` when the memory does not exist.
    pub fn memory_vector(&self, id: &MemoryId) -> Result<Option<Vec<f32>>> {
        self.read("reading a memory vector", |tx| {
            self.check_index(tx)?;
            if crate::memories::find(tx, id)?.is_none() {
                return Err(crate::memories::memory_not_found(id));
            }
            let params = [Value::Text(id.to_string())];
            Ok(mean_vectors(tx, " AND c.memory_id = ?", &params)?
                .pop()
                .map(|(_, v)| v))
        })
    }

    /// Keyword search over chunk text and memory titles (FTS5, `porter
    /// unicode61`). `text` is searched as one quoted phrase. Each memory
    /// appears once, with its best-ranked chunk; memories are ordered by
    /// rank, then id. The filters are those of vector search. Chunks still
    /// waiting for their embedding are found too (decision B-27).
    /// `index_mismatch` until a stored index matches the configured one.
    pub fn keyword_search(
        &self,
        text: &str,
        scope: QueryScope,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<KeywordHit>> {
        let exclude: HashSet<MemoryId> = filter.exclude.iter().copied().collect();
        self.read("searching keywords", |tx| {
            self.check_index(tx)?;
            let Some(phrase) = fts_phrase(text) else {
                return Ok(Vec::new());
            };
            if limit == 0 {
                return Ok(Vec::new());
            }
            let mut params = vec![Value::Text(phrase)];
            let mut where_ = String::new();
            if let QueryScope::Project(p) = scope {
                where_.push_str(" AND m.project_id = ?");
                params.push(Value::Integer(p.0));
            }
            where_.push_str(&filter_sql(filter, &mut params));
            let sql = format!(
                "SELECT c.memory_id, c.seq, chunks_fts.rank FROM chunks_fts
                 JOIN chunks c ON c.id = chunks_fts.rowid
                 JOIN memories m ON m.id = c.memory_id
                 WHERE chunks_fts MATCH ?{where_}
                 ORDER BY chunks_fts.rank, c.memory_id, c.seq"
            );
            let mut s = tx.prepare_cached(&sql).ctx("searching keywords")?;
            let mut rows = s
                .query(params_from_iter(&params))
                .ctx("searching keywords")?;
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            while let Some(r) = rows.next().ctx("searching keywords")? {
                let id: String = r.get(0).ctx("reading a keyword hit")?;
                if seen.contains(&id) {
                    continue;
                }
                let memory_id: MemoryId = id
                    .parse()
                    .map_err(|e: CoreError| corrupt("chunk memory id", e.message))?;
                seen.insert(id);
                if exclude.contains(&memory_id) {
                    continue;
                }
                out.push(KeywordHit {
                    memory_id,
                    chunk_seq: chunk_seq(r.get(1).ctx("reading a keyword hit")?)?,
                    rank: r.get(2).ctx("reading a keyword hit")?,
                });
                if out.len() == limit {
                    break;
                }
            }
            Ok(out)
        })
    }
}

impl VectorIndex for Store {
    fn nearest_memories(&self, query: &VectorQuery<'_>) -> Result<Vec<MemoryHit>> {
        self.nearest(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3_keyword_text_is_one_quoted_phrase() {
        assert_eq!(fts_phrase("  "), None);
        assert_eq!(fts_phrase("Foo::bar"), Some("\"Foo::bar\"".to_owned()));
        assert_eq!(
            fts_phrase(r#"say "hi" OR x"#),
            Some(r#""say ""hi"" OR x""#.to_owned())
        );
    }
}
