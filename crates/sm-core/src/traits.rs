//! The traits behind every external engine (SPEC section 3, Crates; section
//! 14). The daemon is written against these so its tests run on fakes
//! (`crate::testing`) and so an engine can be replaced without touching the
//! request handlers.
//!
//! Every trait is `Send + Sync`: the daemon shares one instance between its
//! request tasks. Implementations that hold a non-`Sync` handle (an SQLite
//! connection, an ONNX Runtime session) guard it internally.

use crate::error::Result;
use crate::ids::{MemoryId, ProjectId};
use crate::types::{AnchorRef, MemoryType, Status};

// ---------------------------------------------------------------------------
// Embedding (SPEC section 5)
// ---------------------------------------------------------------------------

/// What an embedder is. Stamped into the index (`meta`) so a model change
/// blocks search until `reindex` (AT-11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderInfo {
    /// `embeddinggemma-2` or `arctic-embed-m` (`EmbedModel::as_str`).
    pub model_id: String,
    /// The pinned revision of the model files.
    pub revision: String,
    /// Vector length; 768 for both supported models.
    pub dim: usize,
}

/// One chunk to embed with the document prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocInput<'a> {
    /// The memory's title; `None` is written as `none` in the prompt. The
    /// embedder cuts it to [`MAX_TITLE_TOKENS`].
    pub title: Option<&'a str>,
    /// The chunk text, already within the chunk limit.
    pub text: &'a str,
}

/// Tokens of title the document prompt carries (SPEC section 5, rule 1).
pub const MAX_TITLE_TOKENS: usize = 32;

/// Turns text into unit-length vectors.
pub trait Embedder: Send + Sync {
    fn info(&self) -> &EmbedderInfo;

    /// Embeds one query with the query prompt, after cutting the query to
    /// `max_tokens` tokens of its own text (128 for the hook, the chunk
    /// limit for `recall`; SPEC section 6, step 1).
    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>>;

    /// Embeds chunks with the document prompt. Returns one unit-length
    /// vector per input, in input order. A chunk embedded in a batch scores
    /// a cosine of at least 0.9999 against the same chunk embedded alone
    /// (AT-1).
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>>;
}

/// Which ONNX Runtime session an embedder runs on (SPEC section 5, Using the
/// machine's cores; spike result 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    /// Created first, kept for the daemon's life, normal priority,
    /// `embed.query_threads` threads.
    Query,
    /// Created for a bulk job and dropped when the queue is empty, below
    /// normal priority, `cpu.budget` × physical cores threads.
    Bulk,
}

/// Opens embedding sessions. The real one loads and verifies the pinned
/// model files; a model whose hash does not match is `model_integrity`
/// (AT-10).
pub trait EmbedderProvider: Send + Sync {
    /// What the sessions will report, without opening one.
    fn info(&self) -> &EmbedderInfo;

    /// Opens a session with exactly `threads` intra-op threads.
    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>>;
}

// ---------------------------------------------------------------------------
// Chunking (SPEC section 5, Chunking rules)
// ---------------------------------------------------------------------------

/// How to split one memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkParams {
    /// `chunk.max_tokens`.
    pub max_tokens: usize,
    /// `chunk.overlap`, for windows over an oversized piece.
    pub overlap: f64,
    /// Tokens indexed at most (`capture.max_index_tokens` for captures,
    /// `None` for explicit `remember`, which is never capped). Chunks past
    /// the cap are not produced; the text is still stored whole.
    pub index_cap: Option<usize>,
}

/// One chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPiece {
    /// 0-based position within the memory.
    pub seq: u32,
    /// The chunk's text: a contiguous part of the memory's content (windows
    /// are cut on token boundaries using the tokenizer's offsets).
    pub text: String,
    /// Tokens in `text`, at most `max_tokens`.
    pub token_count: u32,
}

/// The result of chunking one memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    /// At least one piece for non-empty content.
    pub pieces: Vec<ChunkPiece>,
    /// Tokens in the whole content (`capture_size_tokens`).
    pub total_tokens: usize,
    /// Tokens covered by `pieces` without counting overlap twice.
    pub indexed_tokens: usize,
    /// Whether `index_cap` cut the content (`capture_cap_hit_total`).
    pub capped: bool,
}

/// Splits text along its structure: headings, then paragraphs, then
/// sentences; code fences whole; pieces packed up to the limit but never
/// across a heading; an oversized piece cut into windows with
/// `overlap` overlap. Content at or under the limit is one chunk.
pub trait Chunker: Send + Sync {
    /// Tokens in `text` with the model's tokenizer, without special tokens.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    fn chunk(&self, content: &str, params: &ChunkParams) -> Result<ChunkPlan>;
}

// ---------------------------------------------------------------------------
// Vector search (SPEC section 4, Search; section 6)
// ---------------------------------------------------------------------------

/// Which projects a search reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryScope {
    /// `WHERE project_id = ?`, scored exactly in Rust.
    Project(ProjectId),
    /// Every project, through `vec1` in exact mode.
    All,
}

/// Filters applied after collapsing chunks to memories (SPEC section 6,
/// step 4).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MemoryFilter {
    /// Empty means every type.
    pub types: Vec<MemoryType>,
    /// Empty means `active` only.
    pub statuses: Vec<Status>,
    /// Only memories attached to this anchor.
    pub anchor: Option<AnchorRef>,
    /// Only memories with at least this importance (SPEC section 16).
    pub min_importance: Option<f64>,
    /// Memories never returned (the hook's already-injected set, the query
    /// memory for `related`).
    pub exclude: Vec<MemoryId>,
}

impl MemoryFilter {
    /// The statuses a search keeps.
    pub fn effective_statuses(&self) -> &[Status] {
        if self.statuses.is_empty() {
            &[Status::Active]
        } else {
            &self.statuses
        }
    }
}

/// A nearest-memory query.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery<'a> {
    /// Unit-length query vector.
    pub vector: &'a [f32],
    pub scope: QueryScope,
    pub filter: MemoryFilter,
    /// Memories returned at most.
    pub limit: usize,
}

/// One memory found by vector search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryHit {
    pub memory_id: MemoryId,
    /// Cosine similarity of the memory's best chunk.
    pub score: f32,
    /// `seq` of that chunk.
    pub chunk_seq: u32,
}

/// Exact nearest-memory search over stored chunk vectors. Chunks whose
/// embedding is still pending (NULL) are skipped. Results are sorted by
/// score, highest first, ties broken by memory id so the order is stable.
pub trait VectorIndex: Send + Sync {
    fn nearest_memories(&self, query: &VectorQuery<'_>) -> Result<Vec<MemoryHit>>;
}

// ---------------------------------------------------------------------------
// Map (SPEC section 7)
// ---------------------------------------------------------------------------

/// UMAP parameters. The defaults are SPEC section 7, step 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectParams {
    pub dims: usize,
    pub neighbors: usize,
    pub min_dist: f32,
    pub seed: u64,
}

impl Default for ProjectParams {
    fn default() -> Self {
        ProjectParams {
            dims: 5,
            neighbors: 15,
            min_dist: 0.0,
            seed: 42,
        }
    }
}

/// Reduces unit vectors (cosine metric) to `dims` dimensions. Must be
/// deterministic: the same input and parameters give the same output
/// (AT-6).
pub trait Projector: Send + Sync {
    fn project(&self, vectors: &[Vec<f32>], params: &ProjectParams) -> Result<Vec<Vec<f32>>>;
}

/// Projection input is capped at this many memories (SPEC section 7, Rust
/// libraries); the rest are assigned to the nearest cluster centre.
pub const MAX_PROJECTION_POINTS: usize = 2000;
/// Fewer memories than this form one group, unclustered (SPEC section 7,
/// step 1).
pub const MIN_CLUSTER_MEMORIES: usize = 30;

/// HDBSCAN parameters, in the Python reference's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterParams {
    pub min_cluster_size: usize,
    /// `min_samples` as the Python library counts it (not counting the
    /// point itself).
    pub min_samples_python: usize,
    pub max_cluster_size: usize,
}

impl ClusterParams {
    /// SPEC section 7, step 3, for `n` points: `min_cluster_size =
    /// max(5, round(sqrt(n) / 2))`, `min_samples = 5`, `max_cluster_size =
    /// n / 4` (integer division, at least `min_cluster_size`).
    pub fn for_n(n: usize) -> Self {
        let half_root = ((n as f64).sqrt() / 2.0).round() as usize;
        let min_cluster_size = half_root.max(5);
        ClusterParams {
            min_cluster_size,
            min_samples_python: 5,
            max_cluster_size: (n / 4).max(min_cluster_size),
        }
    }

    /// The value to pass to the `hdbscan` crate, whose `min_samples` counts
    /// the point itself (SPEC section 7: "pass the Python value plus one").
    pub fn rust_min_samples(&self) -> usize {
        self.min_samples_python + 1
    }
}

/// Clusters points; `None` marks an unclustered point. Cluster numbers are
/// meaningful only within one call (SPEC section 4: never keys).
pub trait Clusterer: Send + Sync {
    fn cluster(&self, points: &[Vec<f32>], params: &ClusterParams) -> Result<Vec<Option<usize>>>;
}

// ---------------------------------------------------------------------------
// Secrets (SPEC section 9, Security)
// ---------------------------------------------------------------------------

/// One credential found in text, as byte offsets into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretMatch {
    /// A short stable name such as `aws_access_key` or `private_key`,
    /// written into `[REDACTED:<kind>]`.
    pub kind: String,
    pub start: usize,
    pub end: usize,
}

/// The result of redacting text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redaction {
    pub text: String,
    /// Kinds replaced, in order of appearance (repeats kept).
    pub kinds: Vec<String>,
}

/// Finds credentials: private keys, cloud access keys, tokens, connection
/// strings with passwords.
pub trait SecretScanner: Send + Sync {
    /// Every match, in any order. Offsets lie on `char` boundaries.
    fn scan(&self, text: &str) -> Vec<SecretMatch>;

    /// Replaces each match with `[REDACTED:<kind>]`. Overlapping matches are
    /// merged and take the kind of the one that starts first (the longer on
    /// a tie). `share` re-scans the result and refuses if anything is left.
    fn redact(&self, text: &str) -> Redaction {
        redact_matches(text, self.scan(text))
    }
}

/// The default redaction over a list of matches; see [`SecretScanner::redact`].
pub fn redact_matches(text: &str, mut matches: Vec<SecretMatch>) -> Redaction {
    matches.retain(|m| {
        m.start < m.end
            && m.end <= text.len()
            && text.is_char_boundary(m.start)
            && text.is_char_boundary(m.end)
    });
    matches.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut out = String::with_capacity(text.len());
    let mut kinds = Vec::new();
    let mut pos = 0;
    let mut i = 0;
    while i < matches.len() {
        let first = &matches[i];
        let start = first.start.max(pos);
        let mut end = first.end;
        let kind = first.kind.clone();
        i += 1;
        while i < matches.len() && matches[i].start < end {
            end = end.max(matches[i].end);
            i += 1;
        }
        if end <= pos {
            continue;
        }
        out.push_str(&text[pos..start]);
        out.push_str("[REDACTED:");
        out.push_str(&kind);
        out.push(']');
        kinds.push(kind);
        pos = end;
    }
    out.push_str(&text[pos..]);
    Redaction { text: out, kinds }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s7_cluster_params_follow_the_spec_formula() {
        let p = ClusterParams::for_n(500);
        assert_eq!((p.min_cluster_size, p.max_cluster_size), (11, 125));
        assert_eq!(ClusterParams::for_n(1000).min_cluster_size, 16);
        assert_eq!(ClusterParams::for_n(5000).min_cluster_size, 35);
        assert_eq!(ClusterParams::for_n(30).min_cluster_size, 5);
        assert_eq!(ClusterParams::for_n(30).max_cluster_size, 7);
        assert_eq!(ClusterParams::for_n(12).max_cluster_size, 5);
        assert_eq!(p.rust_min_samples(), 6);
        assert_eq!(
            ProjectParams::default(),
            ProjectParams {
                dims: 5,
                neighbors: 15,
                min_dist: 0.0,
                seed: 42
            }
        );
    }

    #[test]
    fn s6_default_filter_keeps_active_only() {
        assert_eq!(
            MemoryFilter::default().effective_statuses(),
            &[Status::Active]
        );
        let f = MemoryFilter {
            statuses: vec![Status::Superseded],
            ..MemoryFilter::default()
        };
        assert_eq!(f.effective_statuses(), &[Status::Superseded]);
    }

    fn m(kind: &str, start: usize, end: usize) -> SecretMatch {
        SecretMatch {
            kind: kind.into(),
            start,
            end,
        }
    }

    #[test]
    fn at9_redaction_replaces_and_merges_matches() {
        let text = "key=AKIAABCDEFGHIJKLMNOP and token=ghp_x";
        let a = text.find("AKIA").unwrap();
        let g = text.find("ghp_").unwrap();
        let r = redact_matches(
            text,
            vec![
                m("github_token", g, text.len()),
                m("aws_access_key", a, a + 20),
            ],
        );
        assert_eq!(
            r.text,
            "key=[REDACTED:aws_access_key] and token=[REDACTED:github_token]"
        );
        assert_eq!(r.kinds, ["aws_access_key", "github_token"]);
        // Overlaps merge; the earlier start names the kind.
        let r = redact_matches("abcdefgh", vec![m("b", 3, 8), m("a", 1, 5)]);
        assert_eq!(r.text, "a[REDACTED:a]");
        // Same start: the longer wins.
        let r = redact_matches("abcdefgh", vec![m("short", 0, 2), m("long", 0, 6)]);
        assert_eq!(r.text, "[REDACTED:long]gh");
        // Contained matches vanish into the outer one.
        let r = redact_matches("abcdefgh", vec![m("outer", 1, 7), m("inner", 2, 3)]);
        assert_eq!((r.text.as_str(), r.kinds.len()), ("a[REDACTED:outer]h", 1));
        // Invalid offsets are ignored rather than panicking.
        let r = redact_matches("é", vec![m("bad", 1, 2), m("oob", 0, 9)]);
        assert_eq!(r.text, "é");
        assert!(redact_matches("plain", vec![]).kinds.is_empty());
    }
}
