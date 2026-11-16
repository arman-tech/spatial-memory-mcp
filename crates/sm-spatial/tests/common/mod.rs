//! Shared helpers for sm-spatial's integration tests: the AT-7 fixture and
//! fake topic memories from the deterministic fake embedder.

#![allow(dead_code)]

use sm_core::EMBED_DIM;
use sm_core::hash::fnv1a64;
use sm_core::ids::MemoryId;
use sm_core::testing::fake_vector;
use sm_spatial::MemoryVectors;

/// SHA-256 of `fixtures/spatial/vectors.f32le`, as `fixtures/spatial/README.md`
/// records it.
pub const FIXTURE_SHA256: &str = "d7845752b6dafaeb4f9976b3bf8238766bc18571618aa2a0421c8815b859ead8";
pub const FIXTURE_TEXTS: usize = 320;
pub const FIXTURE_TOPICS: usize = 10;

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/spatial/vectors.f32le"
);
const TEXTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/spatial/texts.json"
);
const README: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/spatial/README.md"
);

pub fn fixture_bytes() -> Vec<u8> {
    std::fs::read(VECTORS).expect("fixtures/spatial/vectors.f32le is readable")
}

pub fn fixture_readme() -> String {
    std::fs::read_to_string(README).expect("fixtures/spatial/README.md is readable")
}

/// The fixture's 320 vectors, in `texts.json` order.
pub fn fixture_vectors() -> Vec<Vec<f32>> {
    let bytes = fixture_bytes();
    assert_eq!(bytes.len(), FIXTURE_TEXTS * EMBED_DIM * 4);
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    floats
        .chunks_exact(EMBED_DIM)
        .map(<[f32]>::to_vec)
        .collect()
}

/// Each text's topic number, in order of first appearance in `texts.json`.
pub fn fixture_topics() -> Vec<usize> {
    let raw = std::fs::read_to_string(TEXTS).expect("fixtures/spatial/texts.json is readable");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("texts.json parses");
    let mut names: Vec<String> = Vec::new();
    entries
        .iter()
        .map(|e| {
            let topic = e["topic"]
                .as_str()
                .expect("every entry has a topic")
                .to_owned();
            match names.iter().position(|n| *n == topic) {
                Some(i) => i,
                None => {
                    names.push(topic);
                    names.len() - 1
                }
            }
        })
        .collect()
}

pub fn fixture_id(i: usize) -> MemoryId {
    MemoryId::from_v1(&format!("spatial-fixture-{i:03}")).0
}

/// The fixture as one memory per text, one chunk each.
pub fn fixture_memories() -> Vec<MemoryVectors> {
    fixture_vectors()
        .into_iter()
        .enumerate()
        .map(|(i, v)| MemoryVectors {
            id: fixture_id(i),
            chunks: vec![v],
        })
        .collect()
}

pub fn fake_id(i: usize) -> MemoryId {
    MemoryId::from_v1(&format!("fake-{i}")).0
}

/// A fake memory text of topic `i % topics`: six words from the topic's
/// 16-word vocabulary and two from a 50-word vocabulary shared by every
/// topic, chosen by hash of `i` (no random source).
pub fn fake_text(i: usize, topics: usize) -> String {
    let t = i % topics;
    let mut words = Vec::with_capacity(8);
    for k in 0..6 {
        let w = fnv1a64(format!("{i}:{k}").as_bytes()) % 16;
        words.push(format!("t{t}w{w}"));
    }
    for k in 0..2 {
        let w = fnv1a64(format!("{i}:common:{k}").as_bytes()) % 50;
        words.push(format!("common{w}"));
    }
    words.join(" ")
}

/// `n` fake topic memories, one chunk each, from the fake embedder.
pub fn fake_memories(n: usize, topics: usize) -> Vec<MemoryVectors> {
    (0..n)
        .map(|i| MemoryVectors {
            id: fake_id(i),
            chunks: vec![fake_vector(&fake_text(i, topics))],
        })
        .collect()
}
