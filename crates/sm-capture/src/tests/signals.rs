//! `signal_cases.json` and `signal_patterns.json`: version 1's classifier
//! (SPEC section 16, Pipeline in the daemon, step 2).

use std::time::Instant;

use serde_json::{Value, json};

use super::checker::{json_eq, text};
use super::{Suite, case_id, fixture, list};
use crate::signal::{
    MEDIUM_THRESHOLD, SIGNAL_PATTERNS, STRONG_THRESHOLD, classify, matching_pattern_ids,
};

/// `classify` as version 1's `SignalResult`.
pub fn observed(text: &str) -> Value {
    let s = classify(text);
    json!({"tier": s.tier.number(), "score": s.score, "patterns_matched": s.families})
}

/// The dict-form `expect` of `signal_cases.json` (its `expect_keys`).
fn check_expect(obs: &Value, expect: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let matched: Vec<&str> = obs["patterns_matched"]
        .as_array()
        .map(|l| l.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let score = obs["score"].as_f64().unwrap_or(f64::NAN);
    let Some(expect) = expect.as_object() else {
        return vec![format!("expect is not an object: {expect}")];
    };
    for (key, value) in expect {
        let ok = match key.as_str() {
            "tier" | "score" | "patterns_matched" => json_eq(&obs[key.as_str()], value),
            "patterns_include" => value.as_array().is_some_and(|l| {
                l.iter()
                    .all(|f| f.as_str().is_some_and(|f| matched.contains(&f)))
            }),
            "patterns_exclude" => value.as_array().is_some_and(|l| {
                l.iter()
                    .all(|f| f.as_str().is_some_and(|f| !matched.contains(&f)))
            }),
            "patterns_matched_min_len" => value.as_u64().is_some_and(|n| matched.len() as u64 >= n),
            "score_min" => value.as_f64().is_some_and(|m| score >= m),
            "score_max" => value.as_f64().is_some_and(|m| score <= m),
            other => {
                problems.push(format!("unknown expect key {other}"));
                continue;
            }
        };
        if !ok {
            problems.push(format!("{key} {value}: got {obs}"));
        }
    }
    problems
}

/// Every classifier case, the two performance budgets and the six `$`
/// rewrite probes.
#[test]
fn s16_signal_cases() {
    let f = fixture("signal_cases.json");
    let mut suite = Suite::new("signal_cases.json");
    for case in list(&f, "cases") {
        let id = case_id(case);
        let obs = observed(&text(&case["text"]));
        suite.case(&id, check_expect(&obs, &case["expect"]));
        suite.computed(&id, "computed_result", &obs, &case["computed_result"]);
    }

    for perf in list(&f, "performance") {
        let id = case_id(perf);
        let budget = &perf["budget"];
        let mut problems = Vec::new();
        if let Some(texts) = perf.get("texts").and_then(Value::as_array) {
            let texts: Vec<(String, &Value)> = texts
                .iter()
                .map(|t| (text(&t["text"]), &t["computed_result"]))
                .collect();
            for (i, (t, computed)) in texts.iter().enumerate() {
                let obs = observed(t);
                suite.computed(&format!("{id}[{i}]"), "computed_result", &obs, computed);
            }
            let warmup = budget["warmup_calls_per_text"].as_u64().unwrap_or(0);
            let iterations = budget["iterations"].as_u64().expect("iterations");
            let limit = budget["avg_ms_per_call_lt"]
                .as_f64()
                .expect("avg_ms_per_call_lt");
            for (t, _) in &texts {
                for _ in 0..warmup {
                    std::hint::black_box(classify(t));
                }
            }
            let start = Instant::now();
            for _ in 0..iterations {
                for (t, _) in &texts {
                    std::hint::black_box(classify(t));
                }
            }
            let calls = iterations as f64 * texts.len() as f64;
            let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / calls;
            println!("BUDGET {id}: {avg_ms:.4} ms per call (budget < {limit} ms)");
            if avg_ms >= limit {
                problems.push(format!(
                    "average {avg_ms:.4} ms per call, budget {limit} ms"
                ));
            }
        } else {
            let t = text(&perf["text"]);
            let limit = budget["elapsed_ms_lt"].as_f64().expect("elapsed_ms_lt");
            let start = Instant::now();
            let obs = observed(&t);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            println!("BUDGET {id}: {ms:.3} ms (budget < {limit} ms)");
            if ms >= limit {
                problems.push(format!("{ms:.3} ms, budget {limit} ms"));
            }
            suite.computed(&id, "computed_result", &obs, &perf["computed_result"]);
        }
        suite.case(&id, problems);
    }

    let probes = &f["rust_dollar_rewrite_probes"];
    for (i, probe) in list(probes, "cases").iter().enumerate() {
        let id = format!("rust_dollar_rewrite_probes[{i}]");
        let t = text(&probe["text"]);
        let obs = observed(&t);
        let ids: Vec<&str> = matching_pattern_ids(&t);
        suite.computed(&id, "computed_result", &obs, &probe["computed_result"]);
        suite.computed(
            &id,
            "regex_ids_matching",
            &json!(ids),
            &probe["regex_ids_matching"],
        );
        suite.case(&id, Vec::new());
    }

    suite.finish(&[]);
}

/// The 22 patterns, in order, against `signal_patterns.json`: identifier,
/// family, confidence, keywords, and the expression (`rust_regex` where the
/// pattern needed rewriting, otherwise `(?i)` and the Python expression).
#[test]
fn s16_signal_patterns_table() {
    let f = fixture("signal_patterns.json");
    let mut suite = Suite::new("signal_patterns.json");
    let patterns = list(&f, "patterns");
    let mut count_problems = Vec::new();
    if patterns.len() != SIGNAL_PATTERNS.len() {
        count_problems.push(format!(
            "fixture has {} patterns, crate {}",
            patterns.len(),
            SIGNAL_PATTERNS.len()
        ));
    }
    for (i, p) in patterns.iter().enumerate() {
        let id = case_id(p);
        let mut problems = Vec::new();
        let Some(ours) = SIGNAL_PATTERNS.get(i) else {
            suite.case(&id, vec![format!("no pattern at index {i}")]);
            continue;
        };
        if ours.id != id {
            problems.push(format!("id {} != {id}", ours.id));
        }
        if p["family"].as_str() != Some(ours.family) {
            problems.push(format!("family {} != {}", ours.family, p["family"]));
        }
        if p["confidence"].as_f64() != Some(ours.confidence) {
            problems.push(format!(
                "confidence {} != {}",
                ours.confidence, p["confidence"]
            ));
        }
        if !json_eq(&json!(ours.keywords), &p["keywords"]) {
            problems.push(format!("keywords {:?} != {}", ours.keywords, p["keywords"]));
        }
        let expected = match p.get("rust_regex").and_then(Value::as_str) {
            Some(r) => r.to_owned(),
            None => format!("(?i){}", p["regex"].as_str().unwrap_or_default()),
        };
        if ours.regex != expected {
            problems.push(format!("regex {:?} != {expected:?}", ours.regex));
        }
        let flags = &p["flags"];
        if !json_eq(flags, &json!(["IGNORECASE"])) {
            problems.push(format!("flags {flags} are not IGNORECASE alone"));
        }
        suite.case(&id, problems);
    }
    let thresholds = &f["tier_thresholds"];
    let mut problems = count_problems;
    if thresholds["tier1_min_score"].as_f64() != Some(STRONG_THRESHOLD) {
        problems.push(format!(
            "tier 1 threshold {STRONG_THRESHOLD} != {}",
            thresholds["tier1_min_score"]
        ));
    }
    if thresholds["tier2_min_score"].as_f64() != Some(MEDIUM_THRESHOLD) {
        problems.push(format!(
            "tier 2 threshold {MEDIUM_THRESHOLD} != {}",
            thresholds["tier2_min_score"]
        ));
    }
    suite.case("tier_thresholds", problems);
    suite.finish(&[]);
}
