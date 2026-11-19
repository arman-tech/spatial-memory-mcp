//! Near-identical results take one place (SPEC 6, near-identical results;
//! B-197). Ambient recall and the `recall` tool walk their results best
//! first, filling their places. A result whose memory vector has a cosine
//! of at least [`RECALL_COLLAPSE_COSINE`] with a result kept before it is
//! hidden and the walk moves on, so the best of each set of near-identical
//! memories keeps its place and the next distinct memory takes the others.
//! Nothing in the store changes: a hidden memory stays as it is and is shown
//! by a search it leads.
//!
//! Ambient recall keeps a memory already shown in the session as the best
//! of its set: it is not shown again (the repeat gate), and its
//! near-identical memories are hidden behind it, so the same thing is not
//! shown twice in a session. A memory with no vector yet (its chunks wait
//! for an embedding) is never hidden and hides nothing. Each read that hides
//! a result journals `recall_collapsed_total` with the number hidden and the
//! `path` (`ambient` or `recall`), the threshold's metric.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::Value;
use sm_core::error::Result;
use sm_core::ids::MemoryId;
use sm_core::journal::metrics;

use crate::state::Shared;

/// The cosine at or above which two results say the same thing and only
/// the better one is shown (B-197). Chosen on the stored design data (the
/// design and hold-out runs of the shipped configuration: 245 active
/// memories, 8,541 pairs within a project) by a rule fixed before the
/// figures were read: start at the judge's near-duplicate threshold
/// ([`crate::judge::NEAR_DUPLICATE_COSINE`], 0.90) and take the lowest
/// two-place cosine at which no pair of memories holding different
/// labelled knowledge is at or above it. One such pair is at 0.9029, so
/// the threshold is 0.91. Provisional; its metric is
/// `recall_collapsed_total`.
pub const RECALL_COLLAPSE_COSINE: f32 = 0.91;

const _: () = assert!(RECALL_COLLAPSE_COSINE >= crate::judge::NEAR_DUPLICATE_COSINE);

/// How many times its places a search asks for, so that a hidden result is
/// replaced by the next distinct one. When more are hidden than that, fewer
/// results are returned.
pub(crate) const FETCH_FACTOR: usize = 2;

/// One read's walk over its results, best first.
pub(crate) struct Collapse {
    vectors: HashMap<MemoryId, Vec<f32>>,
    kept: Vec<MemoryId>,
    hidden: HashSet<MemoryId>,
}

impl Collapse {
    /// A walk over results whose vectors are `vectors`.
    pub(crate) fn new(vectors: HashMap<MemoryId, Vec<f32>>) -> Self {
        Collapse {
            vectors,
            kept: Vec::new(),
            hidden: HashSet::new(),
        }
    }

    /// A walk over `ids`, their vectors read from the store in one call.
    pub(crate) fn read(sh: &Shared, ids: &[MemoryId]) -> Result<Self> {
        let vectors = if ids.len() < 2 {
            HashMap::new()
        } else {
            sh.store.memory_vectors_of(ids)?
        };
        Ok(Collapse::new(vectors))
    }

    /// Whether the next result, `id`, keeps its place: `false`, and it is
    /// counted as hidden, when its cosine with a result kept before it is
    /// at least [`RECALL_COLLAPSE_COSINE`]; `true`, and it is kept, else.
    /// A result met again gets the same answer and is counted once.
    pub(crate) fn admit(&mut self, id: MemoryId) -> bool {
        if self.hidden.contains(&id) {
            return false;
        }
        if self.kept.contains(&id) {
            return true;
        }
        let Some(v) = self.vectors.get(&id) else {
            return true;
        };
        let same = self.kept.iter().any(|k| {
            self.vectors
                .get(k)
                .is_some_and(|kv| crate::capture::cosine(kv, v) >= RECALL_COLLAPSE_COSINE)
        });
        if same {
            self.hidden.insert(id);
            false
        } else {
            self.kept.push(id);
            true
        }
    }

    /// How many results the walk hid.
    pub(crate) fn hidden(&self) -> usize {
        self.hidden.len()
    }

    /// Journals `recall_collapsed_total` when the walk hid a result.
    pub(crate) fn record(&self, sh: &Shared, request_id: &str, path: &str) {
        if self.hidden() == 0 {
            return;
        }
        let mut labels = BTreeMap::new();
        labels.insert("path".to_owned(), Value::from(path));
        sh.journal.record(
            request_id,
            metrics::RECALL_COLLAPSED_TOTAL,
            self.hidden() as f64,
            labels,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// A vector whose cosine with `[1, 0, 0]` is `c`, in the plane of the
    /// first axis and `axis`.
    fn at(c: f32, axis: usize) -> Vec<f32> {
        let mut v = vec![c, 0.0, 0.0];
        v[axis] = (1.0 - c * c).sqrt();
        unit(&v)
    }

    fn walk(order: &[MemoryId], vectors: &HashMap<MemoryId, Vec<f32>>) -> (Vec<bool>, usize) {
        let mut c = Collapse::new(vectors.clone());
        let kept = order.iter().map(|id| c.admit(*id)).collect();
        (kept, c.hidden())
    }

    #[test]
    fn s6_b197_hides_later_near_identical_results_only() {
        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId::new_v7()).collect();
        let mut vectors = HashMap::new();
        vectors.insert(ids[0], vec![1.0, 0.0, 0.0]);
        // At or above the threshold: hidden behind the first.
        vectors.insert(ids[1], at(RECALL_COLLAPSE_COSINE + 0.001, 1));
        // Under it: kept.
        vectors.insert(ids[2], at(RECALL_COLLAPSE_COSINE - 0.01, 2));
        // ids[3] has no vector: kept, hides nothing.
        vectors.insert(ids[4], vec![0.0, 0.0, 1.0]);
        assert_eq!(
            walk(&ids, &vectors),
            (vec![true, false, true, true, true], 1)
        );

        // Order decides which one of a set is kept: the earlier, better one.
        let reversed: Vec<MemoryId> = ids.iter().rev().copied().collect();
        assert_eq!(
            walk(&reversed, &vectors),
            (vec![true, true, true, true, false], 1)
        );

        // A hidden result hides nothing after it: only kept ones do.
        let mut chain = HashMap::new();
        chain.insert(ids[0], vec![1.0, 0.0, 0.0]);
        chain.insert(ids[1], at(0.95, 1));
        // 0.95 with ids[1], under the threshold with ids[0].
        let c1 = at(0.95, 1);
        let rot = |c: f32, s: f32| vec![c1[0] * c - c1[1] * s, c1[0] * s + c1[1] * c, 0.0];
        let third = rot(0.95, (1.0f32 - 0.95 * 0.95).sqrt());
        assert!(crate::capture::cosine(&third, &chain[&ids[0]]) < RECALL_COLLAPSE_COSINE);
        assert!(crate::capture::cosine(&third, &chain[&ids[1]]) >= RECALL_COLLAPSE_COSINE);
        chain.insert(ids[2], third);
        assert_eq!(walk(&ids[..3], &chain), (vec![true, false, true], 1));

        // A result met twice gets the same answer and is counted once.
        let twice = [ids[0], ids[1], ids[0], ids[1]];
        assert_eq!(walk(&twice, &chain), (vec![true, false, true, false], 1));
    }
}
