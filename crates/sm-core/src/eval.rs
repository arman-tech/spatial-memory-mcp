//! Retrieval evaluation: the AT-2 regression gate and `doctor --eval`
//! (SPEC section 11, Acceptance tests 2, and Evaluation on real memories).
//!
//! The formulas are a direct port of version 2's
//! `tests/integration/test_retrieval_quality.py`:
//!
//! - DCG is the sum of `gain / log2(rank + 1)` with binary gains, and the
//!   ideal ranking has `min(|ground truth|, k)` hits at the top;
//! - reciprocal rank is `1 / rank` of the first relevant result anywhere in
//!   the returned list, or 0;
//! - hit-at-k is whether any of the first `k` results is relevant;
//! - every query asks recall for 10 results and the metrics are averaged over
//!   queries.
//!
//! A measured value fails the gate only when it is below the baseline by more
//! than the fixture's own tolerance (decision B-13).

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::journal::percentile;

/// How many results each evaluation query asks for, and the nDCG cut-off.
pub const EVAL_LIMIT: usize = 10;
/// The cut-off for hit-at-k.
pub const HIT_K: usize = 5;
/// Tolerance on nDCG@10 (version 2's `EPS_NDCG`).
pub const EPS_NDCG: f64 = 0.02;
/// Tolerance on mean reciprocal rank (version 2's `EPS_MRR`).
pub const EPS_MRR: f64 = 0.02;
/// Tolerance on hit-at-5 (version 2's `EPS_HIT_AT_K`).
pub const EPS_HIT_AT_K: f64 = 0.07;

/// One memory of a self-contained evaluation set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvalMemory {
    pub id: String,
    pub content: String,
}

/// One query and the memories it should find.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvalQuery {
    pub text: String,
    pub ground_truth_ids: Vec<String>,
}

/// The `retrieval_eval_set.json` format. `memories` is present in the ported
/// fixture, which is indexed into an empty store before the queries run. A
/// set written against a live store (`doctor --eval`) leaves it out, and its
/// ground-truth ids are stored memory ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvalSet {
    #[serde(default)]
    pub memories: Vec<EvalMemory>,
    pub queries: Vec<EvalQuery>,
}

impl EvalSet {
    /// Parses and checks a set.
    pub fn parse(json: &str) -> Result<Self> {
        let set: EvalSet = serde_json::from_str(json)
            .map_err(|e| CoreError::invalid_input(format!("evaluation set is not valid: {e}")))?;
        set.validate()?;
        Ok(set)
    }

    /// Reads, parses and checks a set file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            CoreError::invalid_input(format!(
                "cannot read evaluation set {}: {e}",
                path.display()
            ))
        })?;
        EvalSet::parse(&text)
    }

    /// At least one query; every query has text and at least one ground-truth
    /// id; memory ids are unique; when memories are given, every ground-truth
    /// id names one of them.
    pub fn validate(&self) -> Result<()> {
        if self.queries.is_empty() {
            return Err(CoreError::invalid_input("evaluation set has no queries")
                .with_next_action("Add queries as {\"text\": ..., \"ground_truth_ids\": [...]}."));
        }
        let mut ids = HashSet::new();
        for m in &self.memories {
            if !ids.insert(m.id.as_str()) {
                return Err(CoreError::invalid_input(format!(
                    "memory id {:?} appears twice",
                    m.id
                )));
            }
        }
        for (i, q) in self.queries.iter().enumerate() {
            if q.text.trim().is_empty() {
                return Err(CoreError::invalid_input(format!(
                    "query {} has no text",
                    i + 1
                )));
            }
            if q.ground_truth_ids.is_empty() {
                return Err(CoreError::invalid_input(format!(
                    "query {} has no ground_truth_ids",
                    i + 1
                )));
            }
            if !self.memories.is_empty()
                && let Some(missing) = q
                    .ground_truth_ids
                    .iter()
                    .find(|g| !ids.contains(g.as_str()))
            {
                return Err(CoreError::invalid_input(format!(
                    "query {} names memory {missing:?}, which is not in the set",
                    i + 1
                )));
            }
        }
        Ok(())
    }
}

/// The stored baseline (`retrieval_quality_baseline.json`). Other keys
/// (`_comment`, `eval_set_size`) are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Baseline {
    #[serde(rename = "nDCG@10")]
    pub ndcg_at_10: f64,
    #[serde(rename = "MRR")]
    pub mrr: f64,
    #[serde(rename = "hit@5")]
    pub hit_at_5: f64,
    #[serde(default)]
    pub model: Option<String>,
}

impl Baseline {
    pub fn parse(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| CoreError::invalid_input(format!("baseline is not valid: {e}")))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            CoreError::invalid_input(format!("cannot read baseline {}: {e}", path.display()))
        })?;
        Baseline::parse(&text)
    }
}

/// Discounted cumulative gain of a ranked list of gains.
pub fn dcg(gains: &[f64]) -> f64 {
    gains
        .iter()
        .enumerate()
        .map(|(i, g)| g / ((i + 2) as f64).log2())
        .sum()
}

/// nDCG at `k` with binary relevance.
pub fn ndcg_at_k<S: AsRef<str>>(retrieved: &[S], ground_truth: &BTreeSet<String>, k: usize) -> f64 {
    let gains: Vec<f64> = retrieved
        .iter()
        .take(k)
        .map(|r| {
            if ground_truth.contains(r.as_ref()) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let ideal = vec![1.0; ground_truth.len().min(k)];
    let idcg = dcg(&ideal);
    if idcg > 0.0 { dcg(&gains) / idcg } else { 0.0 }
}

/// Reciprocal rank of the first relevant result in the whole list.
pub fn reciprocal_rank<S: AsRef<str>>(retrieved: &[S], ground_truth: &BTreeSet<String>) -> f64 {
    retrieved
        .iter()
        .position(|r| ground_truth.contains(r.as_ref()))
        .map_or(0.0, |i| 1.0 / (i + 1) as f64)
}

/// Whether any of the first `k` results is relevant.
pub fn hit_at_k<S: AsRef<str>>(retrieved: &[S], ground_truth: &BTreeSet<String>, k: usize) -> bool {
    retrieved
        .iter()
        .take(k)
        .any(|r| ground_truth.contains(r.as_ref()))
}

/// One returned result, in rank order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Retrieved {
    pub id: String,
    pub score: f32,
}

/// The three averaged metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvalMetrics {
    #[serde(rename = "nDCG@10")]
    pub ndcg_at_10: f64,
    #[serde(rename = "MRR")]
    pub mrr: f64,
    #[serde(rename = "hit@5")]
    pub hit_at_5: f64,
}

impl EvalMetrics {
    /// The gate: each metric may fall below the baseline by at most its
    /// tolerance. Returns one message per failing metric; empty means pass.
    pub fn regressions(&self, baseline: &Baseline) -> Vec<String> {
        let mut out = Vec::new();
        let mut check = |name: &str, measured: f64, base: f64, eps: f64| {
            if measured < base - eps {
                out.push(format!(
                    "{name} {measured:.4} is below the baseline {base:.4} by more than {eps}"
                ));
            }
        };
        check("nDCG@10", self.ndcg_at_10, baseline.ndcg_at_10, EPS_NDCG);
        check("MRR", self.mrr, baseline.mrr, EPS_MRR);
        check("hit@5", self.hit_at_5, baseline.hit_at_5, EPS_HIT_AT_K);
        out
    }
}

/// Summary of a set of similarity scores, for calibrating the floor
/// (SPEC section 6, Similarity floor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScoreDistribution {
    pub count: usize,
    pub min: Option<f64>,
    pub p5: Option<f64>,
    pub p25: Option<f64>,
    pub median: Option<f64>,
    pub p75: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
}

impl ScoreDistribution {
    /// Percentiles use the same method as the metrics journal.
    pub fn from_scores(scores: &[f64]) -> Self {
        let finite: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
        let min = finite.iter().copied().reduce(f64::min);
        let max = finite.iter().copied().reduce(f64::max);
        ScoreDistribution {
            count: finite.len(),
            min,
            p5: percentile(&finite, 5),
            p25: percentile(&finite, 25),
            median: percentile(&finite, 50),
            p75: percentile(&finite, 75),
            p95: percentile(&finite, 95),
            max,
        }
    }
}

/// The outcome for one query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QueryOutcome {
    pub text: String,
    #[serde(rename = "nDCG@10")]
    pub ndcg_at_10: f64,
    pub reciprocal_rank: f64,
    #[serde(rename = "hit@5")]
    pub hit_at_5: bool,
    /// 1-based rank of the first relevant result, if any was returned.
    pub first_relevant_rank: Option<usize>,
    /// Ground-truth ids that were not returned in the first 10.
    pub missing: Vec<String>,
}

/// What `doctor --eval` prints and the AT-2 test checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvalReport {
    pub queries: usize,
    pub metrics: EvalMetrics,
    /// Scores of returned results that were in the ground truth.
    pub relevant_scores: ScoreDistribution,
    /// Scores of returned results that were not.
    pub irrelevant_scores: ScoreDistribution,
    pub per_query: Vec<QueryOutcome>,
    /// Present when a baseline was supplied.
    pub baseline: Option<Baseline>,
    /// Messages for metrics that fell below the baseline beyond tolerance.
    pub regressions: Vec<String>,
}

/// Scores a run. `retrieved[i]` is the ranked result list for
/// `set.queries[i]`, at most [`EVAL_LIMIT`] long; longer lists are cut.
pub fn evaluate(
    set: &EvalSet,
    retrieved: &[Vec<Retrieved>],
    baseline: Option<&Baseline>,
) -> Result<EvalReport> {
    if retrieved.len() != set.queries.len() {
        return Err(CoreError::internal(format!(
            "{} result lists for {} queries",
            retrieved.len(),
            set.queries.len()
        )));
    }
    let n = set.queries.len();
    let (mut ndcg, mut mrr, mut hits) = (0.0, 0.0, 0.0);
    let mut relevant = Vec::new();
    let mut irrelevant = Vec::new();
    let mut per_query = Vec::with_capacity(n);
    for (q, results) in set.queries.iter().zip(retrieved) {
        let results = &results[..results.len().min(EVAL_LIMIT)];
        let truth: BTreeSet<String> = q.ground_truth_ids.iter().cloned().collect();
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let n_q = ndcg_at_k(&ids, &truth, EVAL_LIMIT);
        let rr = reciprocal_rank(&ids, &truth);
        let hit = hit_at_k(&ids, &truth, HIT_K);
        for r in results {
            if truth.contains(&r.id) {
                relevant.push(f64::from(r.score));
            } else {
                irrelevant.push(f64::from(r.score));
            }
        }
        ndcg += n_q;
        mrr += rr;
        hits += if hit { 1.0 } else { 0.0 };
        per_query.push(QueryOutcome {
            text: q.text.clone(),
            ndcg_at_10: n_q,
            reciprocal_rank: rr,
            hit_at_5: hit,
            first_relevant_rank: ids.iter().position(|id| truth.contains(*id)).map(|i| i + 1),
            missing: truth
                .iter()
                .filter(|t| !ids.contains(&t.as_str()))
                .cloned()
                .collect(),
        });
    }
    let nf = n as f64;
    let metrics = EvalMetrics {
        ndcg_at_10: ndcg / nf,
        mrr: mrr / nf,
        hit_at_5: hits / nf,
    };
    let regressions = baseline.map(|b| metrics.regressions(b)).unwrap_or_default();
    Ok(EvalReport {
        queries: n,
        metrics,
        relevant_scores: ScoreDistribution::from_scores(&relevant),
        irrelevant_scores: ScoreDistribution::from_scores(&irrelevant),
        per_query,
        baseline: baseline.cloned(),
        regressions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|s| (*s).to_owned()).collect()
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/retrieval")
            .join(name)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn at2_metrics_match_version_2_formulas() {
        let gt = truth(&["a", "b"]);
        // Values computed with version 2's Python helpers.
        assert!(close(ndcg_at_k(&["a", "b", "x"], &gt, 10), 1.0));
        let expected = (1.0 / 3f64.log2()) / (1.0 + 1.0 / 3f64.log2());
        assert!(close(ndcg_at_k(&["x", "a"], &gt, 10), expected));
        assert!(close(ndcg_at_k::<&str>(&[], &gt, 10), 0.0));
        assert!(close(ndcg_at_k(&["a"], &BTreeSet::new(), 10), 0.0));
        // The cut-off applies to gains and to the ideal list.
        assert!(close(ndcg_at_k(&["x", "a"], &gt, 1), 0.0));
        assert!(close(reciprocal_rank(&["x", "y", "b"], &gt), 1.0 / 3.0));
        assert!(close(reciprocal_rank(&["x"], &gt), 0.0));
        assert!(hit_at_k(&["1", "2", "3", "4", "a"], &gt, 5));
        assert!(!hit_at_k(&["1", "2", "3", "4", "5", "a"], &gt, 5));
        assert!(close(dcg(&[1.0, 1.0]), 1.0 + 1.0 / 3f64.log2()));
    }

    #[test]
    fn at2_gate_fails_only_beyond_the_fixture_tolerances() {
        let base = Baseline {
            ndcg_at_10: 0.9742,
            mrr: 0.9667,
            hit_at_5: 1.0,
            model: None,
        };
        let ok = EvalMetrics {
            ndcg_at_10: 0.9542,
            mrr: 0.9467,
            hit_at_5: 0.94,
        };
        assert!(
            ok.regressions(&base).is_empty(),
            "{:?}",
            ok.regressions(&base)
        );
        let bad = EvalMetrics {
            ndcg_at_10: 0.95,
            mrr: 0.94,
            hit_at_5: 0.92,
        };
        let r = bad.regressions(&base);
        assert_eq!(r.len(), 3, "{r:?}");
        assert!(r[0].starts_with("nDCG@10"));
        assert_eq!((EPS_NDCG, EPS_MRR, EPS_HIT_AT_K), (0.02, 0.02, 0.07));
    }

    #[test]
    fn at2_eval_fixtures_load() {
        let set = EvalSet::load(&fixture("retrieval_eval_set.json")).unwrap();
        assert_eq!((set.memories.len(), set.queries.len()), (30, 15));
        let base = Baseline::load(&fixture("retrieval_quality_baseline.json")).unwrap();
        assert_eq!(
            (base.ndcg_at_10, base.mrr, base.hit_at_5),
            (0.9742, 0.9667, 1.0)
        );
        assert_eq!(base.model.as_deref(), Some("arctic-m-long"));
    }

    #[test]
    fn at2_perfect_run_scores_one_and_reports_distributions() {
        let set = EvalSet::load(&fixture("retrieval_eval_set.json")).unwrap();
        let runs: Vec<Vec<Retrieved>> = set
            .queries
            .iter()
            .map(|q| {
                let mut v: Vec<Retrieved> = q
                    .ground_truth_ids
                    .iter()
                    .map(|id| Retrieved {
                        id: id.clone(),
                        score: 0.8,
                    })
                    .collect();
                v.push(Retrieved {
                    id: "noise".into(),
                    score: 0.6,
                });
                v
            })
            .collect();
        let base = Baseline::load(&fixture("retrieval_quality_baseline.json")).unwrap();
        let report = evaluate(&set, &runs, Some(&base)).unwrap();
        assert_eq!(report.queries, 15);
        assert!(close(report.metrics.ndcg_at_10, 1.0));
        assert!(close(report.metrics.mrr, 1.0));
        assert!(close(report.metrics.hit_at_5, 1.0));
        assert!(report.regressions.is_empty());
        assert_eq!(report.irrelevant_scores.count, 15);
        assert_eq!(report.irrelevant_scores.max, Some(f64::from(0.6f32)));
        assert_eq!(report.relevant_scores.min, Some(f64::from(0.8f32)));
        assert!(
            report
                .per_query
                .iter()
                .all(|q| q.missing.is_empty() && q.first_relevant_rank == Some(1))
        );
        // An empty run regresses on every metric.
        let empty = vec![Vec::new(); 15];
        let report = evaluate(&set, &empty, Some(&base)).unwrap();
        assert_eq!(report.regressions.len(), 3);
        assert_eq!(report.relevant_scores.count, 0);
        assert_eq!(report.relevant_scores.median, None);
        assert!(evaluate(&set, &empty[..3], None).is_err());
    }

    #[test]
    fn s11_doctor_eval_sets_may_omit_memories_but_not_queries() {
        let live = r#"{"queries":[{"text":"how do we deploy","ground_truth_ids":["0192f0c4-0000-7000-8000-000000000001"]}]}"#;
        let set = EvalSet::parse(live).unwrap();
        assert!(set.memories.is_empty());
        assert!(EvalSet::parse(r#"{"queries":[]}"#).is_err());
        assert!(EvalSet::parse(r#"{"queries":[{"text":" ","ground_truth_ids":["a"]}]}"#).is_err());
        assert!(EvalSet::parse(r#"{"queries":[{"text":"q","ground_truth_ids":[]}]}"#).is_err());
        let dangling = r#"{"memories":[{"id":"m1","content":"x"}],"queries":[{"text":"q","ground_truth_ids":["m2"]}]}"#;
        assert!(EvalSet::parse(dangling).is_err());
        let dup = r#"{"memories":[{"id":"m1","content":"x"},{"id":"m1","content":"y"}],"queries":[{"text":"q","ground_truth_ids":["m1"]}]}"#;
        assert!(EvalSet::parse(dup).is_err());
    }
}
