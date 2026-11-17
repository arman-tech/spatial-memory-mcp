//! Storage tests (SPEC section 4; AT-5, AT-11, AT-12; D-2, D-3).
//!
//! Every test opens its own database in a temporary folder and embeds with
//! the fake embedder (`sm_core::testing`) or with exact random unit vectors;
//! none needs the real model.

mod concurrency;
mod index;
mod injections;
mod keyword;
mod links;
mod liveness;
mod maps;
mod memories;
mod migrations;
mod projects;
mod scope;
mod search_exact;
mod vec1_blocks;

use std::path::{Path, PathBuf};

use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::project::ProjectIdentity;
use sm_core::testing::{FAKE_MODEL_ID, FAKE_REVISION, fake_vector};
use sm_core::traits::ChunkPiece;
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use tempfile::TempDir;

use crate::error::Ctx;
use crate::{Access, MemoryInput, NewChunk, Store};

/// A store in its own temporary folder, configured with [`stamp`].
///
/// Fields drop in declaration order: the store closes its connections
/// before the folder is removed. Windows refuses to delete an open file, and
/// `TempDir` ignores that failure, so the other order leaks the folder.
pub(crate) struct Fixture {
    pub(crate) store: Store,
    pub(crate) dir: TempDir,
}

impl Fixture {
    pub(crate) fn db_path(&self) -> PathBuf {
        self.dir.path().join("memory.db")
    }

    pub(crate) fn backups(&self) -> PathBuf {
        self.dir.path().join("backups")
    }

    /// A second, independent `Store` on the same file (as a second process
    /// would open it), configured with `stamp`.
    pub(crate) fn reopen(&self, stamp: IndexStamp) -> Store {
        let s = Store::open_at(&self.db_path(), &self.backups()).expect("reopen the store");
        s.configure_index(stamp).expect("configure the index");
        s
    }
}

/// The fake embedder's stamp with the SPEC 15 chunk defaults.
pub(crate) fn stamp() -> IndexStamp {
    IndexStamp {
        model_id: FAKE_MODEL_ID.to_owned(),
        revision: FAKE_REVISION.to_owned(),
        dim: EMBED_DIM,
        chunk_max_tokens: 256,
        chunk_overlap: 0.125,
    }
}

pub(crate) fn open_in(dir: &Path) -> Store {
    let s = Store::open_at(&dir.join("memory.db"), &dir.join("backups")).expect("open the store");
    s.configure_index(stamp()).expect("configure the index");
    s
}

pub(crate) fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temporary folder");
    let store = open_in(dir.path());
    Fixture { dir, store }
}

/// A synthetic project identity (no repository, no remote).
pub(crate) fn identity(name: &str) -> ProjectIdentity {
    ProjectIdentity {
        root: PathBuf::from(format!("/work/{name}")),
        root_path: format!("/work/{name}"),
        root_key: format!("/work/{name}"),
        git_remote: None,
        repo: None,
        name: name.to_owned(),
        capture_enabled: true,
        warnings: Vec::new(),
    }
}

pub(crate) fn project(store: &Store, name: &str) -> ProjectId {
    store.upsert_project(&identity(name)).expect("project").id
}

pub(crate) fn piece(seq: u32, text: &str) -> ChunkPiece {
    ChunkPiece {
        seq,
        text: text.to_owned(),
        token_count: u32::try_from(text.split_whitespace().count()).expect("token count"),
    }
}

/// A local note whose content is one chunk embedded with the fake embedder.
pub(crate) fn note(project: ProjectId, content: &str) -> MemoryInput {
    MemoryInput {
        memory: NewMemory::new(project, content, Provenance::new(ProvenanceSource::Agent)),
        chunks: vec![NewChunk::embedded(piece(0, content), fake_vector(content))],
    }
}

/// A note with one fake-embedded chunk per part; the content is the parts
/// joined with spaces.
pub(crate) fn note_in_parts(project: ProjectId, parts: &[&str]) -> MemoryInput {
    let content = parts.join(" ");
    let chunks = parts
        .iter()
        .enumerate()
        .map(|(i, p)| NewChunk::embedded(piece(u32::try_from(i).expect("seq"), p), fake_vector(p)))
        .collect();
    MemoryInput {
        memory: NewMemory::new(project, content, Provenance::new(ProvenanceSource::Agent)),
        chunks,
    }
}

pub(crate) fn add(store: &Store, project: ProjectId, content: &str) -> MemoryId {
    store
        .insert_memory(note(project, content), Access::Local)
        .expect("insert a memory")
        .id
}

/// Rows in a table, virtual tables included.
pub(crate) fn table_count(store: &Store, table: &str) -> i64 {
    store
        .read("counting rows", |tx| {
            tx.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                .ctx("counting rows")
        })
        .expect("count rows")
}

/// Rows of `table` whose `column` equals `value`.
pub(crate) fn rows_where(store: &Store, table: &str, column: &str, value: &str) -> i64 {
    store
        .read("counting rows", |tx| {
            tx.query_row(
                &format!("SELECT count(*) FROM {table} WHERE {column} = ?1"),
                [value],
                |r| r.get(0),
            )
            .ctx("counting rows")
        })
        .expect("count rows")
}

/// Asserts that `r` failed with `code`.
#[track_caller]
pub(crate) fn assert_code<T: std::fmt::Debug>(r: Result<T, CoreError>, code: ErrorCode) {
    match r {
        Ok(v) => panic!("expected {code:?}, got Ok({v:?})"),
        Err(e) => assert_eq!(e.code, code, "unexpected error: {e:?}"),
    }
}

/// SplitMix64: a small, seedable generator, so every run of a test sees
/// the same data.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (the modulo bias is irrelevant here).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        let n64 = u64::try_from(n).expect("range");
        usize::try_from(self.next_u64() % n64).expect("index")
    }

    /// Uniform in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = u64::try_from(hi - lo + 1).expect("range");
        lo + i64::try_from(self.next_u64() % span).expect("offset")
    }

    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// A random unit vector (Gaussian-ish via the sum of uniforms).
    pub(crate) fn unit_vector(&mut self) -> Vec<f32> {
        let mut v: Vec<f64> = (0..EMBED_DIM)
            .map(|_| {
                (0..4)
                    .map(|_| self.next_u64() as f64 / u64::MAX as f64 - 0.5)
                    .sum::<f64>()
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        #[allow(clippy::cast_possible_truncation)]
        v.into_iter().map(|x| x as f32).collect()
    }
}

/// The scale of [`ExactVector`] components: each is an integer over 1024.
pub(crate) const EXACT_SCALE: f32 = 1024.0;
/// The squared length of an [`ExactVector`]'s integers: 1024².
pub(crate) const EXACT_NORM: i64 = 1 << 20;

/// A unit vector whose components are integers in -64..=64 over 1024, with
/// squared integers summing to exactly 1024². Its squared length is exactly
/// 1.0 in `f32`, and every partial sum of a dot product of two such vectors
/// is a multiple of 2⁻²⁰ no larger than 1 in magnitude, so `f32` computes it
/// exactly in any order: the store's score of a chunk is exactly
/// `Σ nᵢmᵢ / 2²⁰`, which the brute-force reference computes in integers.
#[derive(Debug, Clone)]
pub(crate) struct ExactVector {
    pub(crate) ints: Vec<i64>,
    pub(crate) vector: Vec<f32>,
}

impl ExactVector {
    /// `Σ nᵢmᵢ`: the dot product times 2²⁰.
    pub(crate) fn dot(&self, other: &ExactVector) -> i64 {
        self.ints.iter().zip(&other.ints).map(|(a, b)| a * b).sum()
    }
}

/// The score the store reports for an integer dot product.
pub(crate) fn exact_score(k: i64) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let k = k as f32;
    k / EXACT_NORM as f32
}

/// Writes `d` (at most 4096) as a sum of four squares, each at most 64.
fn four_squares(d: i64) -> Option<[i64; 4]> {
    let top = d.isqrt();
    for a in (0..=top).rev() {
        let ra = d - a * a;
        for b in (0..=a.min(ra.isqrt())).rev() {
            let rb = ra - b * b;
            for c in (0..=b.min(rb.isqrt())).rev() {
                let rc = rb - c * c;
                let e = rc.isqrt();
                if e <= c && e * e == rc {
                    return Some([a, b, c, e]);
                }
            }
        }
    }
    None
}

pub(crate) fn exact_unit(rng: &mut Rng) -> ExactVector {
    loop {
        let mut n = vec![0i64; EMBED_DIM];
        let mut order: Vec<usize> = (0..EMBED_DIM).collect();
        rng.shuffle(&mut order);
        let (reserved, free) = order.split_at(4);
        for &i in free {
            n[i] = rng.range(-40, 40);
        }
        let mut d = EXACT_NORM - n.iter().map(|x| x * x).sum::<i64>();
        if d < 0 {
            continue;
        }
        let mut raise: Vec<usize> = free.to_vec();
        rng.shuffle(&mut raise);
        for &i in &raise {
            if d <= 4096 {
                break;
            }
            let old = n[i];
            let new = (old * old + d).isqrt().min(64);
            let sign = if old < 0 || (old == 0 && rng.below(2) == 0) {
                -1
            } else {
                1
            };
            d -= new * new - old * old;
            n[i] = sign * new;
        }
        if d > 4096 {
            continue;
        }
        let Some(squares) = four_squares(d) else {
            continue;
        };
        for (&i, s) in reserved.iter().zip(squares) {
            n[i] = if rng.below(2) == 0 { s } else { -s };
        }
        assert_eq!(n.iter().map(|x| x * x).sum::<i64>(), EXACT_NORM);
        #[allow(clippy::cast_precision_loss)]
        let vector = n.iter().map(|&x| x as f32 / EXACT_SCALE).collect();
        return ExactVector { ints: n, vector };
    }
}

#[test]
fn d2_exact_vectors_are_unit_length_in_f32() {
    let mut rng = Rng::new(7);
    for _ in 0..20 {
        let v = exact_unit(&mut rng);
        let sq: f32 = v.vector.iter().map(|x| x * x).sum();
        assert_eq!(sq, 1.0);
        assert!(v.ints.iter().all(|x| x.abs() <= 64));
        crate::codec::check_vector("an exact vector", &v.vector).expect("unit length");
        let w = exact_unit(&mut rng);
        let f: f32 = v.vector.iter().zip(&w.vector).map(|(a, b)| a * b).sum();
        assert_eq!(f, exact_score(v.dot(&w)));
        assert_eq!(
            crate::codec::dot_blob(&crate::codec::vector_to_blob(&v.vector), &w.vector),
            exact_score(v.dot(&w))
        );
    }
}
