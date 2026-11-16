//! Deterministic stand-ins for unit tests (build prompt: unit tests use a
//! fake embedder; only one integration test binary loads the real model).
//!
//! Enabled with the `test-support` feature; never part of the binary.
//!
//! [`FakeEmbedder`] hashes lower-cased alphanumeric words into 768 signed
//! buckets with FNV-1a and normalises the result, so texts that share words
//! score high, unrelated texts score near zero, and the same text always
//! gives the same vector on every platform. [`SimpleChunker`] counts
//! whitespace-separated words as tokens.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::EMBED_DIM;
use crate::error::Result;
use crate::hash::fnv1a64;
use crate::traits::{
    ChunkParams, ChunkPiece, ChunkPlan, Chunker, DocInput, Embedder, EmbedderInfo,
    EmbedderProvider, SessionKind,
};

/// Model id the fake reports.
pub const FAKE_MODEL_ID: &str = "fake-hash";
/// Revision the fake reports.
pub const FAKE_REVISION: &str = "fnv1a64-v1";

/// What the fake reports as its model.
pub fn fake_info() -> EmbedderInfo {
    EmbedderInfo {
        model_id: FAKE_MODEL_ID.to_owned(),
        revision: FAKE_REVISION.to_owned(),
        dim: EMBED_DIM,
    }
}

/// Lower-cased runs of alphanumeric characters.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
}

/// The fake vector for a text: each word adds ±1 to the bucket its hash
/// picks; the result is scaled to unit length. Text with no words gets a
/// fixed unit vector (bucket 0).
pub fn fake_vector(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; EMBED_DIM];
    for w in words(text) {
        let h = fnv1a64(w.as_bytes());
        let bucket = (h % EMBED_DIM as u64) as usize;
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        v[bucket] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v[0] = 1.0;
    } else {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Cosine similarity of two unit vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A deterministic embedder. Counts its calls so tests can assert batching.
#[derive(Debug)]
pub struct FakeEmbedder {
    info: EmbedderInfo,
    kind: SessionKind,
    threads: usize,
    query_calls: AtomicUsize,
    document_calls: AtomicUsize,
    documents_embedded: AtomicUsize,
}

impl FakeEmbedder {
    pub fn new() -> Self {
        Self::with_session(SessionKind::Query, 1)
    }

    pub fn with_session(kind: SessionKind, threads: usize) -> Self {
        FakeEmbedder {
            info: fake_info(),
            kind,
            threads,
            query_calls: AtomicUsize::new(0),
            document_calls: AtomicUsize::new(0),
            documents_embedded: AtomicUsize::new(0),
        }
    }

    pub fn kind(&self) -> SessionKind {
        self.kind
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn query_calls(&self) -> usize {
        self.query_calls.load(Ordering::Relaxed)
    }

    /// Calls to `embed_documents` (batches).
    pub fn document_calls(&self) -> usize {
        self.document_calls.load(Ordering::Relaxed)
    }

    /// Chunks embedded across all batches.
    pub fn documents_embedded(&self) -> usize {
        self.documents_embedded.load(Ordering::Relaxed)
    }
}

impl Default for FakeEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for FakeEmbedder {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        self.query_calls.fetch_add(1, Ordering::Relaxed);
        let cut: Vec<&str> = text.split_whitespace().take(max_tokens).collect();
        Ok(fake_vector(&cut.join(" ")))
    }

    /// The title does not change the vector, so a chunk scores the same
    /// whatever its memory's title; batching never changes a vector.
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        self.document_calls.fetch_add(1, Ordering::Relaxed);
        self.documents_embedded
            .fetch_add(docs.len(), Ordering::Relaxed);
        Ok(docs.iter().map(|d| fake_vector(d.text)).collect())
    }
}

/// Opens [`FakeEmbedder`] sessions and counts them.
#[derive(Debug)]
pub struct FakeProvider {
    info: EmbedderInfo,
    opened: AtomicUsize,
}

impl FakeProvider {
    pub fn new() -> Self {
        FakeProvider {
            info: fake_info(),
            opened: AtomicUsize::new(0),
        }
    }

    /// Sessions opened so far.
    pub fn opened(&self) -> usize {
        self.opened.load(Ordering::Relaxed)
    }
}

impl Default for FakeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbedderProvider for FakeProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        self.opened.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(FakeEmbedder::with_session(kind, threads)))
    }
}

/// Splits on whitespace: a token is a word. Content at or under the limit is
/// one chunk; longer content is cut into windows of `max_tokens` words that
/// overlap by `overlap × max_tokens` words, stopping at `index_cap` words.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleChunker;

impl Chunker for SimpleChunker {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn chunk(&self, content: &str, params: &ChunkParams) -> Result<ChunkPlan> {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        let total = tokens.len();
        let max = params.max_tokens.max(1);
        let limit = params.index_cap.map_or(total, |cap| cap.min(total));
        let capped = limit < total;
        let overlap = ((max as f64) * params.overlap).floor() as usize;
        let step = max.saturating_sub(overlap).max(1);
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < limit {
            let end = (start + max).min(limit);
            pieces.push(ChunkPiece {
                seq: u32::try_from(pieces.len()).unwrap_or(u32::MAX),
                text: tokens[start..end].join(" "),
                token_count: u32::try_from(end - start).unwrap_or(u32::MAX),
            });
            if end == limit {
                break;
            }
            start += step;
        }
        Ok(ChunkPlan {
            pieces,
            total_tokens: total,
            indexed_tokens: limit,
            capped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s14_fake_embedder_is_deterministic_and_unit_length() {
        let e = FakeEmbedder::new();
        let a = e.embed_query("Orders table uses soft delete", 128).unwrap();
        let b = e.embed_query("orders TABLE uses soft-delete", 128).unwrap();
        assert_eq!(a.len(), EMBED_DIM);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5);
        assert!(dot(&a, &b) > 0.99, "case and punctuation do not matter");
        let unrelated = e
            .embed_query("kubernetes ingress certificate rotation", 128)
            .unwrap();
        assert!(dot(&a, &unrelated) < 0.5);
        let empty = e.embed_query("", 128).unwrap();
        assert!((dot(&empty, &empty) - 1.0).abs() < 1e-5);

        // Batching never changes a vector (AT-1's property holds trivially).
        let docs = [
            DocInput {
                title: Some("t"),
                text: "alpha beta",
            },
            DocInput {
                title: None,
                text: "gamma",
            },
        ];
        let batch = e.embed_documents(&docs).unwrap();
        let alone = e.embed_documents(&docs[1..]).unwrap();
        assert!(dot(&batch[1], &alone[0]) > 0.9999);
        assert_eq!(e.document_calls(), 2);
        assert_eq!(e.documents_embedded(), 3);

        // The query is cut to max_tokens words.
        let cut = e.embed_query("alpha beta gamma", 2).unwrap();
        assert!(dot(&cut, &fake_vector("alpha beta")) > 0.9999);
    }

    #[test]
    fn s14_fake_provider_opens_sessions() {
        let p = FakeProvider::new();
        let s = p.open(SessionKind::Bulk, 12).unwrap();
        assert_eq!(s.info().dim, EMBED_DIM);
        assert_eq!(p.opened(), 1);
    }

    #[test]
    fn s14_simple_chunker_windows_and_cap() {
        let c = SimpleChunker;
        let text = (0..10)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let one = c
            .chunk(
                &text,
                &ChunkParams {
                    max_tokens: 16,
                    overlap: 0.125,
                    index_cap: None,
                },
            )
            .unwrap();
        assert_eq!(one.pieces.len(), 1);
        assert_eq!(one.total_tokens, 10);
        let win = c
            .chunk(
                &text,
                &ChunkParams {
                    max_tokens: 4,
                    overlap: 0.25,
                    index_cap: None,
                },
            )
            .unwrap();
        let texts: Vec<&str> = win.pieces.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]);
        let capped = c
            .chunk(
                &text,
                &ChunkParams {
                    max_tokens: 4,
                    overlap: 0.0,
                    index_cap: Some(6),
                },
            )
            .unwrap();
        assert!(capped.capped);
        assert_eq!(capped.indexed_tokens, 6);
        assert_eq!(capped.pieces.len(), 2);
        assert!(
            c.chunk(
                "",
                &ChunkParams {
                    max_tokens: 4,
                    overlap: 0.0,
                    index_cap: None
                }
            )
            .unwrap()
            .pieces
            .is_empty()
        );
    }
}
