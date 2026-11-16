//! SPEC section 11 search budgets, the search part only: `recall` over
//! 100,000 chunks of 768 dimensions, 95th percentile, project scope against
//! 250 ms and all projects against 350 ms. The budgets also cover embedding
//! the query, which is not the store's; this measures `Store::nearest` alone.
//!
//! ```text
//! cargo run --release -p sm-store --example search_budget
//! ```
//!
//! Two stores are built in a temporary folder from random unit vectors, two
//! chunks per memory:
//!
//! * **spread**: 100,000 chunks in 60 projects, rows interleaved (the layout
//!   of SPIKE s7b), searched in one project (about 1,667 chunks) and in all;
//! * **single**: 100,000 chunks in one project, the worst case for project
//!   scope, searched in that project and in all.
//!
//! Each search asks for the default `recall` top 5 (SPEC section 6).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use sm_core::EMBED_DIM;
use sm_core::error::Result;
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::traits::{ChunkPiece, MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_store::{Access, MemoryInput, NewChunk, Store};

const CHUNKS: usize = 100_000;
const CHUNKS_PER_MEMORY: usize = 2;
const MEMORIES_PER_BATCH: usize = 1_000;
const WARMUP: usize = 10;
const QUERIES: usize = 300;
const TOP_K: usize = 5;
const PROJECT_BUDGET_MS: f64 = 250.0;
const ALL_BUDGET_MS: f64 = 350.0;

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

fn memory(rng: &mut Rng, project: ProjectId, n: usize) -> MemoryInput {
    let chunks: Vec<NewChunk> = (0..CHUNKS_PER_MEMORY)
        .map(|c| {
            let text = format!("memory {n} part {c} token{}", rng.next_u64() % 10_000);
            NewChunk::embedded(
                ChunkPiece {
                    seq: c as u32,
                    token_count: text.split_whitespace().count() as u32,
                    text,
                },
                rng.unit_vector(),
            )
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

/// A store of [`CHUNKS`] chunks in `projects` projects, memories assigned
/// round-robin so each project's rows are scattered through the file.
fn build(dir: &std::path::Path, projects: usize, seed: u64) -> Result<(Store, Vec<ProjectId>)> {
    let store = Store::open_at(&dir.join("memory.db"), &dir.join("backups"))?;
    store.configure_index(IndexStamp {
        model_id: "random-unit-vectors".to_owned(),
        revision: "search-budget".to_owned(),
        dim: EMBED_DIM,
        chunk_max_tokens: 256,
        chunk_overlap: 0.125,
    })?;
    let ids: Vec<ProjectId> = (0..projects)
        .map(|i| {
            store
                .upsert_project(&identity(&format!("project-{i}")))
                .map(|p| p.id)
        })
        .collect::<Result<_>>()?;
    let mut rng = Rng(seed);
    let memories = CHUNKS / CHUNKS_PER_MEMORY;
    let started = Instant::now();
    let mut n = 0;
    while n < memories {
        let end = (n + MEMORIES_PER_BATCH).min(memories);
        let batch = (n..end)
            .map(|m| memory(&mut rng, ids[m % projects], m))
            .collect();
        store.insert_memories(batch, Access::Local)?;
        n = end;
    }
    println!(
        "  built {} memories, {CHUNKS} chunks, {projects} project(s) in {:.1} s",
        memories,
        started.elapsed().as_secs_f64()
    );
    Ok((store, ids))
}

struct Summary {
    p50: f64,
    p95: f64,
    max: f64,
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

fn measure(store: &Store, scope: QueryScope, seed: u64) -> Result<Summary> {
    let mut rng = Rng(seed);
    let queries: Vec<Vec<f32>> = (0..WARMUP + QUERIES).map(|_| rng.unit_vector()).collect();
    let mut times = Vec::with_capacity(QUERIES);
    for (i, q) in queries.iter().enumerate() {
        let started = Instant::now();
        let hits = store.nearest(&VectorQuery {
            vector: q,
            scope,
            filter: MemoryFilter::default(),
            limit: TOP_K,
        })?;
        let took = started.elapsed();
        assert_eq!(hits.len(), TOP_K, "every search fills the top {TOP_K}");
        if i >= WARMUP {
            times.push(took);
        }
    }
    times.sort();
    let at = |p: f64| millis(times[((p * times.len() as f64).ceil() as usize).max(1) - 1]);
    Ok(Summary {
        p50: at(0.50),
        p95: at(0.95),
        max: millis(times[times.len() - 1]),
    })
}

fn report(what: &str, s: &Summary, budget_ms: f64) {
    println!(
        "  {what:<44} p50 {:>7.1} ms  p95 {:>7.1} ms  max {:>7.1} ms  budget {budget_ms:.0} ms: {}",
        s.p50,
        s.p95,
        s.max,
        if s.p95 <= budget_ms { "within" } else { "OVER" }
    );
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

fn main() -> Result<()> {
    println!(
        "search_budget: {} {}, {} logical cores, CPU {}, {} profile",
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
    println!(
        "{CHUNKS} chunks x {EMBED_DIM} dimensions, top {TOP_K}, {QUERIES} timed queries after {WARMUP} warm-up"
    );

    let spread_dir = tempfile::tempdir().map_err(|e| {
        sm_core::error::CoreError::storage(format!("creating a temporary folder: {e}"))
    })?;
    println!("spread: 60 projects");
    let (spread, projects) = build(spread_dir.path(), 60, 1)?;
    report(
        "project scope (one project, ~1,667 chunks)",
        &measure(&spread, QueryScope::Project(projects[17]), 2)?,
        PROJECT_BUDGET_MS,
    );
    report(
        "all projects",
        &measure(&spread, QueryScope::All, 3)?,
        ALL_BUDGET_MS,
    );
    drop(spread);
    drop(spread_dir);

    let single_dir = tempfile::tempdir().map_err(|e| {
        sm_core::error::CoreError::storage(format!("creating a temporary folder: {e}"))
    })?;
    println!("single: 1 project");
    let (single, projects) = build(single_dir.path(), 1, 4)?;
    report(
        "project scope (the project holds all chunks)",
        &measure(&single, QueryScope::Project(projects[0]), 5)?,
        PROJECT_BUDGET_MS,
    );
    report(
        "all projects",
        &measure(&single, QueryScope::All, 6)?,
        ALL_BUDGET_MS,
    );
    Ok(())
}
