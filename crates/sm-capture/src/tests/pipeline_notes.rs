//! `queue_and_pipeline_notes.json`: version 1's pipeline orchestration for
//! PostToolUse and for the transcript pass (PreCompact and Stop), with its
//! helpers (SPEC section 16, Pipeline in the daemon, steps 1 to 5).
//!
//! Version 1's tests replaced every step with a fake and checked the order
//! of the calls, the gating and the queue file's fields. This test does the
//! same with [`capture_with`]: [`Fakes`] returns the fixture's fake values
//! and records each call as version 1's call log did, and the candidate
//! stands in for the queue file (`write`).
//!
//! What does not carry over: the queue writer, its rate limit and its file
//! names (replaced by the spool, B-25 and B-38), the per-session transcript
//! offset (the daemon's, stream E), and the client and project root (stream
//! D normalises the payload and resolves the project). Each case or check
//! that needs them is listed as not applicable with its reason.
//!
//! Version 3 changes the transcript pass's order. Version 1 checked a text
//! for a duplicate on the raw text before redacting it; version 3 redacts
//! first and checks the redacted text, because the redacted text is what is
//! stored and what the daemon's content hash dedupe sees (B-14, B-29). The
//! cases where the call log differs for this reason are announced.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use regex::Regex;
use serde_json::{Map, Value, json};
use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;
use sm_core::traits::{Redaction, SecretMatch};

use super::checker::{check_all, float, has_py_literal, json_eq, resolve};
use super::{
    Suite, case_id, claude_name, fixture, list, project, queue_fields, settings_with_tools,
    without_unowned,
};
use crate::candidate::{derive_tags, derive_transcript_tags};
use crate::extract::{Extracted, SKIP_TOOLS, SPATIAL_MEMORY_PREFIX, V1_TOOL_NAMES, skip_rule};
use crate::pipeline::{CaptureInput, CaptureOutcome, Skip, Stages, TextDrop, capture_with};
use crate::secrets::SecretReport;
use crate::session::SessionState;
use crate::signal::{NAMESPACE_MAP, Signal, Tier, importance, v1_namespace};
use crate::transcript::{AssistantTexts, TranscriptEntry, TranscriptRead};

/// The project root a case runs with when its `kwargs` give none.
const DEFAULT_ROOT: &str = "/project";

/// Checks whose path names something this crate does not own, with the
/// reason. Matched as a substring of the check's `path`.
const NOT_OURS: [(&str, &str); 6] = [
    (
        "kwargs.client",
        "this crate records no client: stream D normalises a Cursor payload to Claude Code's \
         field names before the daemon runs capture",
    ),
    (
        "kwargs.project_root_dir",
        "the candidate belongs to CaptureInput.project; the daemon stores it under that \
         project's ProjectId (CONTRACTS 7)",
    ),
    (
        "calls.load_state",
        "the per-session transcript offset is the daemon's (stream E); this crate takes \
         transcript_offset and returns new_offset",
    ),
    (
        "calls.save_state",
        "the per-session transcript offset is the daemon's (stream E); this crate takes \
         transcript_offset and returns new_offset",
    ),
    (
        "calls.get_queued_hashes",
        "queue files are replaced by the spool (B-25, B-38); the session's hashes are held in \
         SessionState",
    ),
    (
        "result.queue_path",
        "queue files are replaced by the spool (B-25, B-38)",
    ),
];

/// Whole cases that do not apply, with the reason.
const WHOLE_CASES: [(&str, &str); 9] = [
    (
        "queue_path_recorded",
        "the queue file path: queue files are replaced by the spool (B-25, B-38)",
    ),
    (
        "rate_limited_skipped",
        "the queue writer's rate limit is replaced by the spool (B-25, B-38)",
    ),
    (
        "rate_limited_stops_loop",
        "the queue writer's rate limit is replaced by the spool (B-25, B-38)",
    ),
    (
        "probe.rate_limit_after_first_success",
        "the queue writer's rate limit is replaced by the spool (B-25, B-38)",
    ),
    (
        "probe.tier_0_uses_fallback",
        "version 3's Tier has no tier 0: a classifier only gives strong, medium or weak",
    ),
    (
        "probe.empty_event_name_source_hook_transcript",
        "the event comes from the CLI subcommand (stream D) as CaptureInput.event, never from \
         hook_event_name, so it cannot be empty",
    ),
    (
        "probe.state_offset_string_coerced",
        "the stored offset is the daemon's (stream E); this crate takes transcript_offset as a \
         number",
    ),
    (
        "probe.state_missing_offset_defaults_zero",
        "the stored offset is the daemon's (stream E); this crate takes transcript_offset as a \
         number",
    ),
    (
        "empty_fields_excluded",
        "the payload has no hook_event_name, and the event comes from the CLI subcommand \
         (stream D) as CaptureInput.event; the empty session and trigger are checked by \
         v3.transcript_context_empty_fields_excluded",
    ),
];

fn whole_case_reason(id: &str) -> Option<&'static str> {
    WHOLE_CASES
        .iter()
        .find(|(case, _)| *case == id)
        .map(|(_, reason)| *reason)
}

/// Version 1's fakes as one set of pipeline steps.
///
/// A fake has `returns` (the same value every call) or `returns_sequence`
/// (one value per call). Version 3 calls `redact` and `is_duplicate` for a
/// different set of texts than version 1 did, so a sequence for those two is
/// keyed by the text version 1 called them with (from the fixture's call
/// log): `redact` by its argument, `is_duplicate` by the raw text being
/// processed. A text version 1 never passed gets no redaction and is not a
/// duplicate.
struct Fakes {
    fakes: Value,
    classify_sequence: VecDeque<Value>,
    redact_by_text: HashMap<String, VecDeque<Value>>,
    duplicate_by_text: HashMap<String, VecDeque<Value>>,
    /// The raw text being processed: the last one classified.
    current: String,
    calls: Map<String, Value>,
    classified: Vec<Value>,
    redacted: Vec<Value>,
    entries_read: usize,
}

impl Fakes {
    fn new(fakes: &Value, python_calls: &Value) -> Self {
        let keyed = |name: &str| -> HashMap<String, VecDeque<Value>> {
            let mut map: HashMap<String, VecDeque<Value>> = HashMap::new();
            if let Some(sequence) = fakes[name]["returns_sequence"].as_array() {
                let calls = python_calls[name]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                for (call, value) in calls.iter().zip(sequence) {
                    let text = call["args"][0]
                        .as_str()
                        .expect("a text argument")
                        .to_owned();
                    map.entry(text).or_default().push_back(value.clone());
                }
            }
            map
        };
        Fakes {
            classify_sequence: fakes["classify"]["returns_sequence"]
                .as_array()
                .map(|s| s.iter().cloned().collect())
                .unwrap_or_default(),
            redact_by_text: keyed("redact"),
            duplicate_by_text: keyed("is_duplicate"),
            fakes: fakes.clone(),
            current: String::new(),
            calls: Map::new(),
            classified: Vec::new(),
            redacted: Vec::new(),
            entries_read: 0,
        }
    }

    fn record(&mut self, name: &str, args: Value) {
        let log = self
            .calls
            .entry(name.to_owned())
            .or_insert_with(|| json!([]));
        log.as_array_mut()
            .expect("a call list")
            .push(json!({"args": args, "kwargs": {}}));
    }

    fn calls(&self, name: &str) -> Value {
        self.calls.get(name).cloned().unwrap_or_else(|| json!([]))
    }

    /// The fixed value of fake `name`.
    fn returns(&self, name: &str) -> Option<&Value> {
        self.fakes[name].get("returns")
    }
}

/// An entry as version 1's `TranscriptEntry` dict.
fn entry_json(e: &TranscriptEntry) -> Value {
    json!({"role": e.role, "text": e.text, "timestamp": e.timestamp, "uuid": e.uuid, "entry_type": e.entry_type})
}

fn entry_from(v: &Value) -> TranscriptEntry {
    let field = |k: &str| v[k].as_str().unwrap_or_default().to_owned();
    TranscriptEntry {
        role: field("role"),
        text: field("text"),
        timestamp: field("timestamp"),
        uuid: field("uuid"),
        entry_type: field("entry_type"),
        prompt: String::new(),
    }
}

/// Version 1's `ClassificationResult` as a [`Signal`].
fn signal_from(v: &Value) -> Signal {
    let number = v["tier"].as_i64().expect("tier");
    Signal {
        tier: Tier::from_number(number).unwrap_or_else(|| {
            panic!("tier {number} has no version 3 tier; list the case as not applicable")
        }),
        score: float(&v["score"]).expect("score"),
        families: v["patterns_matched"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
    }
}

/// Version 1's `RedactionResult` as a [`SecretReport`]: one match of kind
/// `fake` per redaction counted.
fn report_from(v: &Value) -> SecretReport {
    let count = v["redaction_count"].as_u64().unwrap_or(0) as usize;
    SecretReport {
        redaction: Redaction {
            text: v["redacted_text"]
                .as_str()
                .expect("redacted_text")
                .to_owned(),
            kinds: vec!["fake".to_owned(); count],
        },
        matches: vec![
            SecretMatch {
                kind: "fake".to_owned(),
                start: 0,
                end: 0,
            };
            count
        ],
        has_private_key: false,
        should_skip: v["should_skip"].as_bool().unwrap_or(false),
    }
}

impl Stages for Fakes {
    fn extract(&mut self, tool_name: &str, tool_input: &Value, response: &str) -> Extracted {
        self.record("extract", json!([tool_name, tool_input, response]));
        let text = self
            .returns("extract")
            .and_then(Value::as_str)
            .expect("extract returns a string")
            .to_owned();
        let original_chars = text.chars().count();
        Extracted {
            text,
            cap_hit: false,
            original_chars,
        }
    }

    fn read(&mut self, path: &str, offset: Option<u64>) -> Option<TranscriptRead> {
        // Python's offset 0 is a session never read, which is `None` here.
        self.record("read", json!([path, offset.unwrap_or(0)]));
        let returned = self.returns("read").expect("a read fake").clone();
        let entries: Vec<TranscriptEntry> = returned["entries"]
            .as_array()
            .into_iter()
            .flatten()
            .map(entry_from)
            .collect();
        self.entries_read = entries.len();
        Some(TranscriptRead {
            entries,
            new_offset: returned["new_offset"].as_u64().expect("new_offset"),
        })
    }

    fn assistant_texts(&mut self, entries: &[TranscriptEntry]) -> AssistantTexts {
        let listed: Vec<Value> = entries.iter().map(entry_json).collect();
        self.record("extract", json!([listed]));
        AssistantTexts {
            texts: self
                .returns("extract")
                .and_then(Value::as_array)
                .expect("extract returns a list")
                .iter()
                .map(|t| t.as_str().expect("a text").to_owned())
                .collect(),
            prompts: Vec::new(),
            cap_hit: false,
        }
    }

    fn classify(&mut self, text: &str) -> Signal {
        self.record("classify", json!([text]));
        self.current = text.to_owned();
        let value = match self.returns("classify") {
            Some(v) => v.clone(),
            None => self
                .classify_sequence
                .pop_front()
                .expect("the classify sequence ran out"),
        };
        self.classified.push(value.clone());
        signal_from(&value)
    }

    fn redact(&mut self, text: &str) -> SecretReport {
        self.record("redact", json!([text]));
        let value = match self.returns("redact") {
            Some(v) => v.clone(),
            None => self
                .redact_by_text
                .get_mut(text)
                .and_then(VecDeque::pop_front)
                .unwrap_or_else(
                    || json!({"redacted_text": text, "redaction_count": 0, "should_skip": false}),
                ),
        };
        self.redacted.push(value.clone());
        report_from(&value)
    }

    fn is_duplicate(&mut self, session: &SessionState, content: &str) -> bool {
        self.record("is_duplicate", json!([content]));
        if self.fakes.get("is_duplicate").is_none() {
            return session.is_duplicate(content);
        }
        match self.returns("is_duplicate") {
            Some(v) => v.as_bool().expect("is_duplicate returns a bool"),
            None => self
                .duplicate_by_text
                .get_mut(&self.current)
                .and_then(VecDeque::pop_front)
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        }
    }
}

/// The checks of `expect` this crate owns; the others are listed as not
/// applicable. `None` when every check was someone else's.
fn split_expect(suite: &mut Suite, id: &str, expect: &Value) -> Option<Value> {
    let checks = expect.as_array().map(Vec::as_slice).unwrap_or_default();
    let mut kept = Vec::new();
    for (i, check) in checks.iter().enumerate() {
        let path = check["path"].as_str().unwrap_or_default();
        match NOT_OURS.iter().find(|(p, _)| path.contains(p)) {
            Some((_, reason)) => suite.not_applicable(&format!("{id}.expect[{i}] {path}"), reason),
            None => kept.push(check.clone()),
        }
    }
    (checks.is_empty() || !kept.is_empty()).then_some(Value::Array(kept))
}

/// A case with `expect` (a ported test) or without (a probe).
fn ported(suite: &mut Suite, id: &str, case: &Value, observed: &Value) {
    match case.get("expect") {
        Some(expect) => match split_expect(suite, id, &resolve(expect)) {
            Some(checks) => suite.case(id, check_all(observed, &checks)),
            None => suite.not_applicable(
                id,
                "every expectation is the spool's, the daemon's or stream D's",
            ),
        },
        None => suite.case(id, Vec::new()),
    }
}

/// The comparable part of version 1's `write` calls.
fn python_writes(python_calls: &Value) -> Value {
    let writes = python_calls["write"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    Value::Array(
        writes
            .iter()
            .map(|w| without_unowned(&w["kwargs"]))
            .collect(),
    )
}

/// The candidates as version 1's `write` call log (`write_queue_file`'s
/// keyword arguments, without the project root and the client).
fn v3_writes(outcome: &CaptureOutcome) -> Vec<Value> {
    outcome
        .candidates
        .iter()
        .map(|c| json!({"args": [], "kwargs": queue_fields(c)}))
        .collect()
}

fn kwargs_only(writes: &[Value]) -> Value {
    Value::Array(writes.iter().map(|w| w["kwargs"].clone()).collect())
}

/// Version 1's `ProcessingResult` for a PostToolUse outcome.
fn pipeline_result(outcome: &CaptureOutcome, fakes: &Fakes) -> Value {
    let skip_reason = match (&outcome.skipped, outcome.drops.first()) {
        (Some(Skip::ToolSkipped(rule)), _) => rule.reason(),
        (Some(Skip::NoToolName), _) => "empty_tool_name".to_owned(),
        (Some(other), _) => format!("{other:?}"),
        (None, Some(TextDrop::Empty)) => "empty_content".to_owned(),
        (None, Some(TextDrop::Weak)) => "tier_3_no_signal".to_owned(),
        (None, Some(TextDrop::Secret)) => "redaction_skip".to_owned(),
        (None, Some(TextDrop::NonFiniteScore)) => "non_finite_score".to_owned(),
        (None, Some(TextDrop::Duplicate)) => "duplicate".to_owned(),
        (None, None) => String::new(),
    };
    let (tier, score, patterns) = match (outcome.candidates.first(), fakes.classified.last()) {
        (Some(c), _) => (json!(c.tier.number()), json!(c.score), json!(c.patterns)),
        (None, Some(v)) => (
            v["tier"].clone(),
            v["score"].clone(),
            v["patterns_matched"].clone(),
        ),
        (None, None) => (json!(3), json!(0.0), json!([])),
    };
    let redaction_count = fakes
        .redacted
        .last()
        .map_or(json!(0), |v| v["redaction_count"].clone());
    let empty = outcome.skipped.is_some() || outcome.drops.first() == Some(&TextDrop::Empty);
    json!({
        "skipped": outcome.candidates.is_empty(),
        "skip_reason": skip_reason,
        "queued": !outcome.candidates.is_empty(),
        "signal_tier": tier,
        "signal_score": score,
        "patterns_matched": patterns,
        "redaction_count": redaction_count,
        "content_length": if empty { 0 } else { outcome.extracted_chars },
    })
}

/// Version 1's computed result without the fields that are not ours.
fn without(v: &Value, keys: &[&str]) -> Value {
    let mut map = v.as_object().cloned().unwrap_or_default();
    for k in keys {
        map.remove(*k);
    }
    Value::Object(map)
}

/// A strong fake for the context cases.
fn strong_fakes() -> Value {
    json!({
        "extract": {"returns": "Decided to use PostgreSQL"},
        "classify": {"returns": {"tier": 1, "score": 0.9, "patterns_matched": ["decision"]}},
        "redact": {"returns": {"redacted_text": "Decided to use PostgreSQL", "redaction_count": 0, "should_skip": false}},
    })
}

/// Runs PostToolUse with the fakes.
fn run_post_tool_use(
    hook_input: &Value,
    fakes: &Value,
    python_calls: &Value,
    root: &str,
) -> (CaptureOutcome, Fakes) {
    let tool = hook_input["tool_name"].as_str().unwrap_or_default();
    let settings = settings_with_tools(&[tool]);
    let project = project(Path::new(root));
    let input = CaptureInput {
        event: HookEvent::PostToolUse,
        payload: hook_input,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: None,
    };
    let mut stages = Fakes::new(fakes, python_calls);
    let outcome = capture_with(&input, &mut SessionState::new(), &mut stages);
    (outcome, stages)
}

/// Runs the transcript pass with the fakes; `None` when the payload's event
/// is not one version 3 can be given.
fn run_transcript(
    hook_input: &Value,
    fakes: &Value,
    python_calls: &Value,
    root: &str,
    offset: u64,
) -> Option<(CaptureOutcome, Fakes, HookEvent)> {
    let event = match hook_input["hook_event_name"].as_str() {
        Some("PreCompact") => HookEvent::PreCompact,
        Some("Stop") => HookEvent::Stop,
        _ => return None,
    };
    let settings = CaptureSettings::default();
    let project = project(Path::new(root));
    let input = CaptureInput {
        event,
        payload: hook_input,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: (offset > 0).then_some(offset),
    };
    let mut stages = Fakes::new(fakes, python_calls);
    let outcome = capture_with(&input, &mut SessionState::new(), &mut stages);
    Some((outcome, stages, event))
}

fn run_pipeline_case(suite: &mut Suite, case: &Value) {
    let id = case_id(case);
    if let Some(reason) = whole_case_reason(&id) {
        suite.not_applicable(&id, reason);
        return;
    }
    let hook_input = resolve(&case["hook_input"]);
    let fakes = resolve(&case["fakes"]);
    let python_calls = resolve(&case["computed_calls"]);
    let root = case["kwargs"]["project_root"]
        .as_str()
        .unwrap_or(DEFAULT_ROOT);
    let (outcome, stages) = run_post_tool_use(&hook_input, &fakes, &python_calls, root);
    let writes = v3_writes(&outcome);
    let observed = json!({
        "result": pipeline_result(&outcome, &stages),
        "calls": {
            "extract": stages.calls("extract"),
            "classify": stages.calls("classify"),
            "redact": stages.calls("redact"),
            "write": writes,
        },
    });
    ported(suite, &id, case, &observed);
    let python_result = without(&resolve(&case["computed_result"]), &["queue_path"]);
    suite.computed(&id, "computed_result", &observed["result"], &python_result);
    for name in ["extract", "classify", "redact"] {
        suite.computed(
            &id,
            &format!("computed_calls.{name}"),
            &observed["calls"][name],
            &python_calls[name],
        );
    }
    suite.computed(
        &id,
        "computed_calls.write",
        &kwargs_only(&writes),
        &python_writes(&python_calls),
    );
}

fn transcript_case(suite: &mut Suite, case: &Value) {
    let id = case_id(case);
    if let Some(reason) = whole_case_reason(&id) {
        suite.not_applicable(&id, reason);
        return;
    }
    let hook_input = resolve(&case["hook_input"]);
    let fakes = resolve(&case["fakes"]);
    let python_calls = resolve(&case["computed_calls"]);
    let root = case["kwargs"]["project_root"]
        .as_str()
        .unwrap_or(DEFAULT_ROOT);
    let offset = fakes["load_state"]["returns"]["last_offset"]
        .as_u64()
        .expect("a numeric stored offset");
    let Some((outcome, stages, event)) =
        run_transcript(&hook_input, &fakes, &python_calls, root, offset)
    else {
        suite.not_applicable(
            &id,
            "the payload names no capture event; the event is stream D's",
        );
        return;
    };
    let count = |d: TextDrop| outcome.drops.iter().filter(|x| **x == d).count();
    let texts = outcome.candidates.len() + outcome.drops.len();
    let with_signal =
        texts - count(TextDrop::Empty) - count(TextDrop::Weak) - count(TextDrop::NonFiniteScore);
    let writes = v3_writes(&outcome);
    let observed = json!({
        "result": {
            "entries_scanned": stages.entries_read,
            "texts_extracted": texts,
            "entries_with_signal": with_signal,
            "entries_queued": outcome.candidates.len(),
            "entries_skipped_dup": count(TextDrop::Duplicate),
            "entries_skipped_redaction": count(TextDrop::Secret),
            "new_offset": outcome.new_offset.unwrap_or(offset),
            "source_hook": claude_name(event),
        },
        "calls": {
            "read": stages.calls("read"),
            "extract": stages.calls("extract"),
            "classify": stages.calls("classify"),
            "redact": stages.calls("redact"),
            "is_duplicate": stages.calls("is_duplicate"),
            "write": writes,
        },
    });
    ported(suite, &id, case, &observed);
    suite.computed(
        &id,
        "computed_result",
        &observed["result"],
        &resolve(&case["computed_result"]),
    );
    for name in ["read", "extract", "classify", "redact"] {
        suite.computed(
            &id,
            &format!("computed_calls.{name}"),
            &observed["calls"][name],
            &python_calls[name],
        );
    }
    // The second argument of version 1's is_duplicate was the queued hashes.
    let first_args = |calls: &Value| -> Value {
        Value::Array(
            calls
                .as_array()
                .into_iter()
                .flatten()
                .map(|c| c["args"][0].clone())
                .collect(),
        )
    };
    suite.computed(
        &id,
        "computed_calls.is_duplicate",
        &first_args(&observed["calls"]["is_duplicate"]),
        &first_args(&python_calls["is_duplicate"]),
    );
    suite.computed(
        &id,
        "computed_calls.write",
        &kwargs_only(&writes),
        &python_writes(&python_calls),
    );
}

/// Version 1's `_build_context` for a PostToolUse payload: the candidate's
/// provenance, or nothing when there is no candidate.
fn build_context(hook_input: &Value) -> Value {
    let (outcome, _) = run_post_tool_use(hook_input, &strong_fakes(), &json!({}), DEFAULT_ROOT);
    outcome
        .candidates
        .first()
        .map_or_else(|| json!({}), |c| queue_fields(c)["context"].clone())
}

/// Version 1's `_build_transcript_context`: the candidate's provenance.
fn build_transcript_context(hook_input: &Value) -> Option<Value> {
    let fakes = json!({
        "read": {"returns": {"entries": [{"role": "assistant", "text": "Decided to use Redis.", "timestamp": "", "uuid": "", "entry_type": "assistant"}], "new_offset": 10}},
        "extract": {"returns": ["Decided to use Redis."]},
        "classify": {"returns": {"tier": 1, "score": 0.9, "patterns_matched": ["decision"]}},
        "redact": {"returns": {"redacted_text": "Decided to use Redis.", "redaction_count": 0, "should_skip": false}},
    });
    let (outcome, _, _) = run_transcript(hook_input, &fakes, &json!({}), DEFAULT_ROOT, 0)?;
    let c = outcome
        .candidates
        .first()
        .expect("a candidate from a strong text");
    Some(queue_fields(c)["context"].clone())
}

fn patterns(v: &Value) -> Vec<String> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn skip_observed(tool_name: &str) -> Value {
    match skip_rule(tool_name) {
        Some(skip) => json!({"skip": true, "reason": skip.reason()}),
        None => json!({"skip": false, "reason": ""}),
    }
}

fn importance_observed(args: &Value) -> Value {
    let score = float(&args["score"]).expect("score");
    let tier = args["tier"].as_i64().expect("tier");
    json!(Tier::from_number(tier).and_then(|t| importance(t, score)))
}

fn sorted(v: &[&str]) -> Value {
    let mut v = v.to_vec();
    v.sort_unstable();
    json!(v)
}

/// The cases whose result satisfies `expect` but differs from version 1's
/// computed value, each one a recorded decision.
fn expected_differences() -> Vec<&'static str> {
    vec![
        // The transcript pass redacts first and checks the redacted text for
        // a duplicate (B-14, B-29), so the `redact` and `is_duplicate` call
        // logs differ: a duplicate is redacted before it is dropped, a text
        // that must be skipped for a secret never reaches `is_duplicate`,
        // and `is_duplicate` is given the fake's redacted text, not the raw
        // one. The counters and the written candidates are version 1's.
        "duplicate_skipped",
        "should_skip_stops_processing",
        "clean_content_proceeds",
        "multiple_texts_processed",
        "empty_text_in_list_skipped",
        "client_default_is_claude_code",
        "client_custom_passed",
        "probe.counters_mixed",
        "probe.dedup_and_redact_get_raw_text",
        "probe.python_strip_control_chars",
        "probe.hashes_loaded_once_with_limit_50",
        // Importance is clamped to 0..1 (stream G step 1), so 1.5 and 1.2
        // become 1.0.
        "probe.tier_1_above_one_not_clamped",
        "probe.tier_2_above_one",
        "probe.score_above_one_passed_unclamped",
        // A weak text, a tier version 3 does not have (0) and a non-finite
        // score give no importance and no candidate (stream G step 1).
        "probe.tier_3_fallback",
        "probe.tier_0_fallback",
        "probe.tier_1_nan",
        "probe.tier_2_nan",
        // A NotebookEdit candidate records its notebook path in provenance
        // (it gets no anchor); version 1's context had no file for it.
        "probe.notebook_path_not_used",
        // The provenance file path is sm_core::project::normalize_file_anchor
        // of the path: root-relative under the root (`/p/a.py` under `/p` is
        // `a.py`), and a path that leaves the root is resolved lexically and
        // kept absolute (`../x` under `/project` is `/x`).
        "probe.metadata_full",
        "probe.file_path_not_validated",
    ]
}

/// Every case and probe of `queue_and_pipeline_notes.json`.
#[test]
fn s16_queue_and_pipeline_notes() {
    let f = fixture("queue_and_pipeline_notes.json");
    let mut suite = Suite::new("queue_and_pipeline_notes.json");
    let probes = &f["probes"];

    for case in list(&f, "model_cases") {
        suite.not_applicable(
            &case_id(case),
            "version 3 reads the payload as JSON and CaptureOutcome replaces ProcessingResult; \
             their mapping is exercised by run_pipeline_cases",
        );
    }

    for case in list(&f, "constants_cases") {
        let id = case_id(case);
        let value = match case["constant"].as_str() {
            Some("SKIP_TOOLS") => json!(SKIP_TOOLS),
            Some("SPATIAL_MEMORY_PREFIX") => json!(SPATIAL_MEMORY_PREFIX),
            Some("SPATIAL_MEMORY_TOOL_NAMES") => json!(V1_TOOL_NAMES),
            other => panic!("unknown constant {other:?}"),
        };
        suite.case(&id, check_all(&json!({ "value": value }), &case["expect"]));
        let ours = match value.as_array() {
            Some(names) => sorted(&names.iter().filter_map(Value::as_str).collect::<Vec<_>>()),
            None => value,
        };
        suite.computed(&id, "computed_value", &ours, &case["computed_value"]);
    }

    for case in list(&f, "should_skip_tool_cases") {
        let id = case_id(case);
        let observed = skip_observed(case["args"]["tool_name"].as_str().expect("tool_name"));
        suite.case(&id, check_all(&observed, &case["expect"]));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }
    for probe in list(probes, "should_skip_tool") {
        let id = case_id(probe);
        let observed = skip_observed(probe["args"]["tool_name"].as_str().expect("tool_name"));
        suite.computed(&id, "computed_result", &observed, &probe["computed_result"]);
        suite.case(&id, Vec::new());
    }

    for case in list(&f, "score_to_importance_cases")
        .iter()
        .chain(list(probes, "score_to_importance"))
    {
        let id = case_id(case);
        let observed = importance_observed(&case["args"]);
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(
            &id,
            "computed_result",
            &observed,
            &resolve(&case["computed_result"]),
        );
    }

    for case in list(&f, "derive_namespace_cases")
        .iter()
        .chain(list(probes, "derive_namespace"))
    {
        let id = case_id(case);
        let observed = json!(v1_namespace(&patterns(&case["args"]["patterns"])));
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }

    let namespaces: Vec<&str> = NAMESPACE_MAP
        .iter()
        .map(|(_, ns)| *ns)
        .chain(["notes"])
        .collect();
    for case in list(&f, "namespace_format_cases") {
        let id = case_id(case);
        let value = case["value"].as_str().expect("value");
        let regex = Regex::new(case["regex"].as_str().expect("regex")).expect("a valid expression");
        let mut problems = check_all(&json!({"matches": regex.is_match(value)}), &case["expect"]);
        if !namespaces.contains(&value) {
            problems.push(format!("{value} is not a namespace version 3 derives"));
        }
        suite.case(&id, problems);
    }

    for case in list(&f, "build_context_cases")
        .iter()
        .chain(list(probes, "build_context"))
    {
        let id = case_id(case);
        let observed = build_context(&resolve(&case["hook_input"]));
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }

    for case in list(&f, "derive_tags_cases")
        .iter()
        .chain(list(probes, "derive_tags"))
    {
        let id = case_id(case);
        let tool = case["hook_input"]["tool_name"].as_str().unwrap_or_default();
        let observed = json!(derive_tags(tool, &patterns(&case["patterns"])));
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }

    for case in list(&f, "build_transcript_context_cases")
        .iter()
        .chain(list(probes, "build_transcript_context"))
    {
        let id = case_id(case);
        if let Some(reason) = whole_case_reason(&id) {
            suite.not_applicable(&id, reason);
            continue;
        }
        let mut hook_input = resolve(&case["hook_input"]);
        if hook_input.get("transcript_path").is_none() {
            hook_input["transcript_path"] = json!("/tmp/transcript.jsonl");
        }
        let Some(observed) = build_transcript_context(&hook_input) else {
            suite.not_applicable(
                &id,
                "the payload names no capture event; the event is stream D's",
            );
            continue;
        };
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }
    // Version 3's form of `empty_fields_excluded`: no session and no
    // trigger in the payload give neither in the provenance.
    let observed = build_transcript_context(
        &json!({"hook_event_name": "PreCompact", "transcript_path": "/t"}),
    )
    .expect("a PreCompact context");
    let mut problems = Vec::new();
    if !json_eq(&observed, &json!({"hook_event": "PreCompact"})) {
        problems.push(format!("context {observed}"));
    }
    suite.case("v3.transcript_context_empty_fields_excluded", problems);

    for case in list(&f, "derive_transcript_tags_cases")
        .iter()
        .chain(list(probes, "derive_transcript_tags"))
    {
        let id = case_id(case);
        let observed = json!(derive_transcript_tags(&patterns(&case["patterns"])));
        ported(&mut suite, &id, case, &json!({ "result": observed }));
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }

    for case in list(&f, "run_pipeline_cases")
        .iter()
        .chain(list(probes, "run_pipeline"))
    {
        run_pipeline_case(&mut suite, case);
    }
    for case in list(&f, "run_transcript_pipeline_cases")
        .iter()
        .chain(list(probes, "run_transcript_pipeline"))
    {
        transcript_case(&mut suite, case);
    }
    suite.not_applicable(
        "run_pipeline.computed_queue_path_posix_simulated",
        "the queue file path: queue files are replaced by the spool (B-25, B-38)",
    );
    suite.not_applicable(
        "run_transcript_pipeline.computed_calls (load_state, save_state, get_queued_hashes, \
         is_duplicate's second argument)",
        "the stored offset is the daemon's (stream E) and the queued hashes are replaced by \
         SessionState (B-25, B-38)",
    );

    let spool = "queue files are replaced by the spool (B-25, B-38)";
    for case in list(&f["queue_constants"], "cases") {
        suite.not_applicable(&format!("queue_constants.{}", case_id(case)), spool);
    }
    for section in [
        "get_queue_dir_cases",
        "filename_cases",
        "write_queue_file_cases",
    ] {
        for case in list(&f, section) {
            suite.not_applicable(&format!("{section}.{}", case_id(case)), spool);
        }
    }
    for section in ["get_queue_dir", "write_queue_file"] {
        for probe in list(probes, section) {
            let reason = if has_py_literal(probe) {
                "Python only ($py_literal): a lone surrogate cannot be held in a Rust String, \
                 and the queue writer is replaced by the spool (B-25, B-38)"
            } else {
                spool
            };
            suite.not_applicable(&format!("probes.{section}.{}", case_id(probe)), reason);
        }
    }

    suite.finish(&expected_differences());
}
