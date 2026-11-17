//! Bulk loading into the vector index (decision B-74): what storing 2,000,
//! 10,000 and 50,000 chunks costs the store, release build.
//!
//! ```text
//! cargo run --release -p sm-store --example bulk_insert
//! cargo run --release -p sm-store --example bulk_insert -- 10000
//! ```
//!
//! The optional argument is the largest size measured (default 50,000).
//! Each shape runs in a fresh store in a temporary folder, one project,
//! random unit vectors, two chunks per memory:
//!
//! * **embed**: chunks stored pending, then given their vectors by
//!   `set_embeddings`, 4 per call. That is the daemon's bulk queue at the
//!   default `embed.batch_size`, and the way every chunk from `import`,
//!   `migrate`, `sync` and `reindex` reaches the vector index. Only the
//!   `set_embeddings` calls are timed.
//! * **insert**: `insert_memories` with embedded chunks, 500 memories (1,000
//!   chunks) per transaction: the shape of stream A's probe (a-report
//!   section 9, item 1).
//! * **single**: one memory of one chunk per transaction, the shape of
//!   `remember`, 20 at each size of the insert store.
//!
//! After each shape the vector index's blocks are counted (rows of the
//! `chunk_vectors_idx` table vec1 keeps) and the all-projects search is
//! timed, so a change to how blocks are written shows its effect on reads.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, Result};
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::traits::{ChunkPiece, MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_store::{Access, MemoryInput, NewChunk, Store};

const SIZES: [usize; 3] = [2_000, 10_000, 50_000];
const CHUNKS_PER_MEMORY: usize = 2;
/// `insert_memories` batch of the pending load (import and migrate use 256).
const PENDING_BATCH: usize = 256;
/// Memories per transaction in the insert shape (1,000 chunks).
const INSERT_BATCH: usize = 500;
/// The default `embed.batch_size` (SPEC 15).
const EMBED_BATCH: usize = 4;
const SINGLES: usize = 20;
const SEGMENT: usize = 5_000;
const SEARCH_WARMUP: usize = 10;
const SEARCHES: usize = 100;
const TOP_K: usize = 5;

/// SplitMix64: seedable, so every run measures the same data.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// A uniformly random direction: normalized Gaussian components
    /// (Box-Muller).
    fn unit_vector(&mut self) -> Vec<f32> {
        let mut v: Vec<f64> = (0..EMBED_DIM)
            .map(|_| {
                let (u1, u2) = (self.unit_interval(), self.unit_interval());
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v.into_iter().map(|x| x as f32).collect()
    }
}

fn identity(name: &str) -> ProjectIdentity {
    ProjectIdentity {
        root: PathBuf::from(format!("/bench/{name}")),
        root_path: format!("/bench/{name}"),
        root_key: format!("/bench/{name}"),
        git_remote: None,
        repo: None,
        name: name.to_owned(),
        capture_enabled: true,
        warnings: Vec::new(),
    }
}

fn memory(
    rng: &mut Rng,
    project: ProjectId,
    n: usize,
    parts: usize,
    embedded: bool,
) -> MemoryInput {
    let chunks: Vec<NewChunk> = (0..parts)
        .map(|c| {
            let text = format!("memory {n} part {c} token{}", rng.next_u64() % 10_000);
            let piece = ChunkPiece {
                seq: c as u32,
                token_count: text.split_whitespace().count() as u32,
                text,
            };
            if embedded {
                NewChunk::embedded(piece, rng.unit_vector())
            } else {
                NewChunk::pending(piece)
            }
        })
        .collect();
    let content = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    MemoryInput {
        memory: NewMemory::new(project, content, Provenance::new(ProvenanceSource::Agent)),
        chunks,
    }
}

struct Bench {
    store: Store,
    project: ProjectId,
    // Dropped after the store, so the folder is removed once the file is
    // closed (Windows refuses to delete an open file).
    _dir: tempfile::TempDir,
}

fn open(name: &str) -> Result<Bench> {
    let dir = tempfile::tempdir()
        .map_err(|e| CoreError::storage(format!("creating a temporary folder: {e}")))?;
    let store = Store::open_at(&dir.path().join("memory.db"), &dir.path().join("backups"))?;
    store.configure_index(IndexStamp {
        model_id: "random-unit-vectors".to_owned(),
        revision: "bulk-insert".to_owned(),
        dim: EMBED_DIM,
        chunk_max_tokens: 256,
        chunk_overlap: 0.125,
    })?;
    let project = store.upsert_project(&identity(name))?.id;
    Ok(Bench {
        store,
        project,
        _dir: dir,
    })
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

/// Prints the mean cost per chunk of each [`SEGMENT`] chunks, and the
/// running total at each of [`SIZES`].
struct Progress {
    done: usize,
    total: Duration,
    segment: Duration,
    segment_start: usize,
    totals: Vec<(usize, Duration)>,
}

impl Progress {
    fn new() -> Self {
        Progress {
            done: 0,
            total: Duration::ZERO,
            segment: Duration::ZERO,
            segment_start: 0,
            totals: Vec::new(),
        }
    }

    /// Records `chunks` stored in `took`; returns the size reached, if it
    /// is one of [`SIZES`].
    fn add(&mut self, chunks: usize, took: Duration) -> Option<usize> {
        self.done += chunks;
        self.total += took;
        self.segment += took;
        if self.done - self.segment_start >= SEGMENT {
            let n = self.done - self.segment_start;
            println!(
                "    chunks {:>6} to {:>6}: {:>7.3} ms per chunk",
                self.segment_start,
                self.done,
                millis(self.segment) / n as f64
            );
            self.segment = Duration::ZERO;
            self.segment_start = self.done;
        }
        if SIZES.contains(&self.done) {
            self.totals.push((self.done, self.total));
            return Some(self.done);
        }
        None
    }

    fn summary(&self, what: &str) {
        for (n, t) in &self.totals {
            println!(
                "  {what}: {n:>6} chunks in {:>8.2} s ({:.3} ms per chunk)",
                t.as_secs_f64(),
                millis(*t) / *n as f64
            );
        }
    }
}

/// Rows in vec1's block table and their mean number of vectors.
fn blocks(store: &Store) -> Result<String> {
    let c = rusqlite::Connection::open_with_flags(
        store.path(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| CoreError::storage(format!("opening the database to count blocks: {e}")))?;
    let (rows, bytes): (i64, i64) = c
        .query_row(
            "SELECT count(*), coalesce(sum(length(val)), 0) FROM chunk_vectors_idx",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| CoreError::storage(format!("counting vec1 blocks: {e}")))?;
    let per_vector = (EMBED_DIM * 4 + 4) as f64;
    let vectors = if rows == 0 {
        0.0
    } else {
        (bytes as f64 / rows as f64 - 12.0) / per_vector
    };
    Ok(format!(
        "{rows} blocks, {vectors:.1} vectors per block on average"
    ))
}

fn search(store: &Store, seed: u64) -> Result<String> {
    let mut rng = Rng(seed);
    let mut times = Vec::with_capacity(SEARCHES);
    for i in 0..SEARCH_WARMUP + SEARCHES {
        let q = rng.unit_vector();
        let started = Instant::now();
        let hits = store.nearest(&VectorQuery {
            vector: &q,
            scope: QueryScope::All,
            filter: MemoryFilter::default(),
            limit: TOP_K,
        })?;
        let took = started.elapsed();
        assert_eq!(hits.len(), TOP_K, "every search fills the top {TOP_K}");
        if i >= SEARCH_WARMUP {
            times.push(took);
        }
    }
    times.sort();
    let at = |p: f64| millis(times[((p * times.len() as f64).ceil() as usize).max(1) - 1]);
    Ok(format!(
        "all-projects search p50 {:.1} ms, p95 {:.1} ms ({SEARCHES} queries, top {TOP_K})",
        at(0.50),
        at(0.95)
    ))
}

fn embed_shape(max: usize) -> Result<()> {
    println!("embed: pending chunks, set_embeddings {EMBED_BATCH} per call");
    let b = open("embed")?;
    let mut rng = Rng(1);
    let memories = max / CHUNKS_PER_MEMORY;
    let mut n = 0;
    while n < memories {
        let end = (n + PENDING_BATCH).min(memories);
        let batch = (n..end)
            .map(|m| memory(&mut rng, b.project, m, CHUNKS_PER_MEMORY, false))
            .collect();
        b.store.insert_memories(batch, Access::Local)?;
        n = end;
    }
    let model = "random-unit-vectors";
    let mut progress = Progress::new();
    loop {
        let pending = b.store.pending_chunks(EMBED_BATCH)?;
        if pending.is_empty() {
            break;
        }
        let items: Vec<_> = pending.iter().map(|c| (c.id, rng.unit_vector())).collect();
        let started = Instant::now();
        let stored = b.store.set_embeddings(model, &items)?;
        let took = started.elapsed();
        assert_eq!(
            stored as usize,
            items.len(),
            "every pending chunk is stored"
        );
        if progress.add(items.len(), took).is_some() {
            println!("    at {}: {}", progress.done, blocks(&b.store)?);
        }
    }
    progress.summary("embed");
    println!("  {}", search(&b.store, 2)?);
    Ok(())
}

fn insert_shape(max: usize) -> Result<()> {
    println!(
        "insert: insert_memories, {INSERT_BATCH} memories ({} chunks) per transaction",
        INSERT_BATCH * CHUNKS_PER_MEMORY
    );
    let b = open("insert")?;
    let mut rng = Rng(3);
    let memories = max / CHUNKS_PER_MEMORY;
    let mut progress = Progress::new();
    let mut singles = Vec::new();
    let mut n = 0;
    while n < memories {
        let end = (n + INSERT_BATCH).min(memories);
        let batch: Vec<_> = (n..end)
            .map(|m| memory(&mut rng, b.project, m, CHUNKS_PER_MEMORY, true))
            .collect();
        let started = Instant::now();
        b.store.insert_memories(batch, Access::Local)?;
        let took = started.elapsed();
        let reached = progress.add((end - n) * CHUNKS_PER_MEMORY, took);
        n = end;
        if let Some(size) = reached {
            println!("    at {size}: {}", blocks(&b.store)?);
            singles.push((size, single(&b, &mut rng, memories + size)?));
        }
    }
    progress.summary("insert");
    for (size, (median, max)) in singles {
        println!(
            "  single: one-chunk memory at {size:>6} chunks, median {median:.2} ms, max {max:.2} ms ({SINGLES} inserts)"
        );
    }
    println!("  {}", search(&b.store, 4)?);
    Ok(())
}

/// [`SINGLES`] one-chunk memories, one transaction each: the median and
/// the maximum in milliseconds.
fn single(b: &Bench, rng: &mut Rng, first: usize) -> Result<(f64, f64)> {
    let mut times = Vec::with_capacity(SINGLES);
    for i in 0..SINGLES {
        let input = memory(rng, b.project, first + i, 1, true);
        let started = Instant::now();
        b.store.insert_memories(vec![input], Access::Local)?;
        times.push(millis(started.elapsed()));
    }
    times.sort_by(f64::total_cmp);
    Ok((times[SINGLES / 2], times[SINGLES - 1]))
}

fn main() -> Result<()> {
    let max: usize = match std::env::args().nth(1) {
        Some(a) => a
            .parse()
            .map_err(|_| CoreError::invalid_input(format!("{a} is not a chunk count")))?,
        None => SIZES[SIZES.len() - 1],
    };
    println!(
        "bulk_insert: {} {}, {} logical cores, CPU {}, {} profile",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        cpu(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("up to {max} chunks x {EMBED_DIM} dimensions, {CHUNKS_PER_MEMORY} chunks per memory");
    embed_shape(max)?;
    insert_shape(max)?;
    Ok(())
}

fn cpu() -> String {
    if let Ok(id) = std::env::var("PROCESSOR_IDENTIFIER") {
        return id;
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}
