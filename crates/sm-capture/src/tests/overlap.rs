//! `overlap_cases.json`: the per-session dedupe between PostToolUse's
//! captures and the transcript pass (SPEC section 16, Pipeline in the
//! daemon, step 4; AT-14; B-14).
//!
//! Version 1 rebuilt the set of hashes from the queue files on every
//! transcript pass (`get_queued_hashes`). Version 3 has no queue files (the
//! spool replaces them, B-25 and B-38): the session's hashes live in
//! [`SessionState`]. The `is_duplicate` cases that set up queue files are
//! run by recording each file's `content` in a `SessionState`.

use serde_json::{Value, json};
use sm_core::hash::{content_hash, python_strip};

use super::checker::check_all;
use super::{Suite, case_id, fixture, list, short};
use crate::session::{SESSION_HASH_CAP, SessionState};

/// The state a case sets up: each queue file's `content`, recorded in
/// file order, or the literal `queued_hashes`.
fn state_for(case: &Value) -> SessionState {
    let mut state = SessionState::new();
    for file in case["setup"]["files_in_new"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let text = file["text"].as_str().expect("queue file text");
        let doc: Value = serde_json::from_str(text).expect("queue file JSON");
        if let Some(content) = doc.get("content").and_then(Value::as_str) {
            state.record_content(content);
        }
    }
    for hash in case["args"]["queued_hashes"]
        .as_array()
        .into_iter()
        .flatten()
    {
        state.record(hash.as_str().expect("a hash"));
    }
    state
}

fn hashes(state: &SessionState) -> Value {
    json!(state.hashes().collect::<Vec<_>>())
}

/// Every `is_duplicate` and `_content_hash` case, the probes, and the
/// 23-row hash table; `get_queued_hashes` does not apply.
#[test]
fn at14_overlap_cases() {
    let f = fixture("overlap_cases.json");
    let mut suite = Suite::new("overlap_cases.json");

    for case in list(&f, "is_duplicate_cases") {
        let id = case_id(case);
        let state = state_for(case);
        let content = case["args"]["content"].as_str().expect("content");
        let observed = json!({"result": state.is_duplicate(content)});
        suite.case(&id, check_all(&observed, &case["expect"]));
        suite.computed(
            &id,
            "computed_result",
            &observed["result"],
            &case["computed_result"],
        );
        if let Some(queued) = case.get("computed_queued_hashes") {
            suite.computed(&id, "computed_queued_hashes", &hashes(&state), queued);
        }
    }

    for case in list(&f, "content_hash_cases") {
        let id = case_id(case);
        let a = content_hash(case["args"]["a"].as_str().expect("a"));
        let b = content_hash(case["args"]["b"].as_str().expect("b"));
        let relation = if a == b { "equal" } else { "not_equal" };
        let observed = json!({"relation": relation});
        suite.case(&id, check_all(&observed, &case["expect"]));
        suite.computed(&id, "computed_hash_a", &json!(a), &case["computed_hash_a"]);
        suite.computed(&id, "computed_hash_b", &json!(b), &case["computed_hash_b"]);
    }

    let probes = &f["probes"];
    for probe in list(probes, "is_duplicate") {
        let id = case_id(probe);
        let state = state_for(probe);
        let content = probe["args"]["content"].as_str().expect("content");
        suite.computed(
            &id,
            "computed_result",
            &json!(state.is_duplicate(content)),
            &probe["computed_result"],
        );
        suite.case(&id, Vec::new());
    }

    for (i, row) in list(&f, "content_hash_table").iter().enumerate() {
        let id = format!("content_hash_table[{i}]");
        let input = row["input"].as_str().expect("input");
        let normalized = python_strip(input).to_lowercase();
        let hex: String = normalized.bytes().map(|b| format!("{b:02x}")).collect();
        let mut problems = Vec::new();
        for (what, ours, python) in [
            (
                "computed_normalized",
                json!(normalized),
                &row["computed_normalized"],
            ),
            (
                "computed_normalized_utf8_hex",
                json!(hex),
                &row["computed_normalized_utf8_hex"],
            ),
            (
                "computed_hash",
                json!(content_hash(input)),
                &row["computed_hash"],
            ),
        ] {
            if !suite.computed(&id, what, &ours, python) {
                problems.push(format!("{what} {} != {}", short(&ours), short(python)));
            }
        }
        // The table is version 1's definition of `content_hash` (B-14), so
        // a difference is a failure, not a decision.
        suite.case(&id, problems);
    }

    let reason = "version 1's queue files are replaced by the spool (B-25, B-38); the session's \
                  hashes are held in SessionState, fed by the candidates it produced";
    for case in list(&f, "get_queued_hashes_cases") {
        suite.not_applicable(&format!("get_queued_hashes.{}", case_id(case)), reason);
    }
    for probe in list(probes, "get_queued_hashes") {
        suite.not_applicable(&format!("get_queued_hashes.{}", case_id(probe)), reason);
    }

    suite.finish(&["probe.whitespace_content_matches_empty_hash"]);
}

/// The session state's bound: at most [`SESSION_HASH_CAP`] hashes, the
/// oldest forgotten first, and a repeated hash neither re-recorded nor
/// moved to the back.
#[test]
fn at14_session_state_bound_is_fifo() {
    let mut state = SessionState::new();
    let text = |i: usize| format!("captured text number {i}");
    for i in 0..SESSION_HASH_CAP {
        assert!(state.record_content(&text(i)), "text {i} recorded once");
    }
    assert_eq!(state.len(), SESSION_HASH_CAP);
    assert!(
        !state.record_content(&text(0)),
        "a repeat is not recorded again"
    );
    assert!(state.is_duplicate(&text(0)));

    assert!(state.record_content(&text(SESSION_HASH_CAP)));
    assert_eq!(state.len(), SESSION_HASH_CAP, "the bound holds");
    assert!(
        !state.is_duplicate(&text(0)),
        "the oldest is forgotten first"
    );
    assert!(state.is_duplicate(&text(1)));
    assert!(state.is_duplicate(&text(SESSION_HASH_CAP)));
    assert_eq!(state.hashes().next(), Some(content_hash(&text(1)).as_str()));

    // Blank text is never recorded and never a duplicate, even when the
    // empty string's hash is in the set.
    assert!(!state.record_content(" \t\n"));
    let mut with_empty = SessionState::new();
    with_empty.record(content_hash(""));
    assert!(!with_empty.is_duplicate("   "));
    assert!(with_empty.is_duplicate("") == with_empty.is_duplicate("   "));
}
