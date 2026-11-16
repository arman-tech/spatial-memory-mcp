//! `transcript/transcript_cases.json`: the transcript reader PreCompact and
//! Stop use, the entry parser, and version 2's assistant text extractor
//! with its one 200,000-character cap (SPEC section 16, Pipeline in the
//! daemon, step 1; AT-14, AT-17).

use std::path::Path;

use serde_json::{Value, json};

use super::checker::{json_eq, resolve};
use super::{Suite, case_id, fixture, fixture_path, list, short, temp_dir};
use crate::extract::{MAX_CAPTURE_CHARS, TRUNCATION_MARKER};
use crate::transcript::{
    MAX_LINE_BYTES, TranscriptEntry, TranscriptRead, extract_assistant_texts, parse_entry,
    read_transcript,
};

fn entry_json(e: &TranscriptEntry) -> Value {
    json!({
        "role": e.role,
        "text": e.text,
        "timestamp": e.timestamp,
        "uuid": e.uuid,
        "entry_type": e.entry_type,
    })
}

fn read_json(r: &TranscriptRead) -> Value {
    json!({
        "entries": r.entries.iter().map(entry_json).collect::<Vec<_>>(),
        "new_offset": r.new_offset,
    })
}

fn path_str(p: &Path) -> String {
    p.to_str().expect("a UTF-8 temporary path").to_owned()
}

/// A fixture entry (`{role, text, timestamp, uuid, entry_type}`, the text
/// possibly a generator) as the reader's entry.
fn entry_from(v: &Value) -> TranscriptEntry {
    let v = resolve(v);
    let field = |k: &str| v[k].as_str().unwrap_or_default().to_owned();
    TranscriptEntry {
        role: field("role"),
        text: field("text"),
        timestamp: field("timestamp"),
        uuid: field("uuid"),
        entry_type: field("entry_type"),
    }
}

/// The `expect` of one reader step. `offsets` holds the new offsets of the
/// steps already run.
fn check_step(read: &TranscriptRead, expect: &Value, offsets: &[u64]) -> Vec<String> {
    let mut problems = Vec::new();
    let Some(expect) = expect.as_object() else {
        return vec![format!("expect is not an object: {expect}")];
    };
    let observed = read_json(read);
    let step_offset = |v: &Value| v.as_u64().and_then(|i| offsets.get(i as usize).copied());
    for (key, value) in expect {
        let value = resolve(value);
        let ok = match key.as_str() {
            "entries_len" => value.as_u64() == Some(read.entries.len() as u64),
            "entries" => json_eq(&observed["entries"], &value),
            "new_offset" => value.as_u64() == Some(read.new_offset),
            "new_offset_gt" => value.as_u64().is_some_and(|n| read.new_offset > n),
            "new_offset_equals_step" => step_offset(&value) == Some(read.new_offset),
            "new_offset_gt_step" => step_offset(&value).is_some_and(|n| read.new_offset > n),
            other => match parse_indexed(other) {
                Some((i, field)) => observed["entries"]
                    .get(i)
                    .and_then(|e| e.get(field))
                    .is_some_and(|got| json_eq(got, &value)),
                None => {
                    problems.push(format!("unknown expect key {other}"));
                    continue;
                }
            },
        };
        if !ok {
            problems.push(format!("{key} {}: got {}", short(&value), short(&observed)));
        }
    }
    problems
}

/// `entries[N].field` as `(N, field)`.
fn parse_indexed(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("entries[")?;
    let (index, field) = rest.split_once("].")?;
    Some((index.parse().ok()?, field))
}

/// The `expect` of the extractor cases: `equals`, `len`, `total_length_max`,
/// `[i]` (an exact text; `-1` is the last), `[i].length`, `[i].ends_with`,
/// `[i].not_ends_with` and `each.not_ends_with`.
fn check_extract(texts: &[String], expect: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let Some(expect) = expect.as_object() else {
        return vec![format!("expect is not an object: {expect}")];
    };
    let lengths: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();
    let at = |i: i64| -> Option<&String> {
        let i = if i < 0 { texts.len() as i64 + i } else { i };
        usize::try_from(i).ok().and_then(|i| texts.get(i))
    };
    for (key, value) in expect {
        let value = resolve(value);
        let ok = match key.as_str() {
            "equals" => json_eq(&json!(texts), &value),
            "len" => value.as_u64() == Some(texts.len() as u64),
            "total_length_max" => value
                .as_u64()
                .is_some_and(|m| lengths.iter().sum::<usize>() as u64 <= m),
            "each.not_ends_with" => value
                .as_str()
                .is_some_and(|s| texts.iter().all(|t| !t.ends_with(s))),
            other => {
                let parsed = other.strip_prefix('[').and_then(|r| r.split_once(']'));
                let Some((index, rest)) =
                    parsed.and_then(|(i, r)| Some((i.parse::<i64>().ok()?, r)))
                else {
                    problems.push(format!("unknown expect key {other}"));
                    continue;
                };
                let text = at(index);
                match rest {
                    "" => text.is_some_and(|t| value.as_str() == Some(t.as_str())),
                    ".length" => {
                        text.is_some_and(|t| value.as_u64() == Some(t.chars().count() as u64))
                    }
                    ".ends_with" => {
                        text.is_some_and(|t| value.as_str().is_some_and(|s| t.ends_with(s)))
                    }
                    ".not_ends_with" => {
                        text.is_some_and(|t| value.as_str().is_some_and(|s| !t.ends_with(s)))
                    }
                    _ => {
                        problems.push(format!("unknown expect key {other}"));
                        continue;
                    }
                }
            }
        };
        if !ok {
            problems.push(format!("{key} {}: got lengths {lengths:?}", short(&value)));
        }
    }
    problems
}

/// The assistant texts of a fixture's `entries`, with their lengths.
fn extract(entries: &Value) -> (Vec<String>, Value, Value) {
    let entries: Vec<TranscriptEntry> = entries
        .as_array()
        .expect("entries")
        .iter()
        .map(entry_from)
        .collect();
    let texts = extract_assistant_texts(&entries).texts;
    let lengths = json!(texts.iter().map(|t| t.chars().count()).collect::<Vec<_>>());
    let observed = json!(texts);
    (texts, observed, lengths)
}

/// Every reader case and probe, the entry parser's cases and probes, and the
/// extractor's cases, probes and the master cases version 2 superseded.
#[test]
fn s16_transcript_cases() {
    let f = fixture("transcript/transcript_cases.json");
    let mut suite = Suite::new("transcript/transcript_cases.json");

    // The reader. Each case gets its own directory and one transcript path,
    // which a later step may overwrite (the append case).
    for case in list(&f, "reader_cases") {
        let id = case_id(case);
        let dir = temp_dir();
        let target = dir.path().join("transcript.jsonl");
        if let Some(spec) = case.get("file_spec") {
            let bytes = match resolve(spec) {
                Value::String(s) => s.into_bytes(),
                other => panic!("{id}: file_spec is not a string: {other}"),
            };
            std::fs::write(&target, bytes).expect("write the file_spec transcript");
        }
        let mut offsets: Vec<u64> = Vec::new();
        let mut problems = Vec::new();
        for (n, step) in list(case, "steps").iter().enumerate() {
            let file = step
                .get("file")
                .or_else(|| case.get("file"))
                .and_then(Value::as_str);
            if let Some(rel) = file {
                std::fs::copy(fixture_path(rel), &target)
                    .unwrap_or_else(|e| panic!("{id}: cannot copy {rel}: {e}"));
            }
            let path = match case.get("path").and_then(Value::as_str) {
                Some(literal) => literal.to_owned(),
                None => path_str(&target),
            };
            let last_offset = match step.get("last_offset_from_step").and_then(Value::as_u64) {
                Some(from) => offsets[from as usize],
                None => step["last_offset"].as_u64().expect("last_offset"),
            };
            if step["last_offset"].as_u64() != Some(last_offset) {
                problems.push(format!(
                    "step {n}: the fixture's last_offset {} is not step {}'s new offset {last_offset}",
                    step["last_offset"], step["last_offset_from_step"]
                ));
            }
            let read = read_transcript(&path, last_offset);
            for p in check_step(&read, &step["expect"], &offsets) {
                problems.push(format!("step {n}: {p}"));
            }
            suite.computed(
                &format!("{id}.step{n}"),
                "computed_result",
                &read_json(&read),
                &resolve(&step["computed_result"]),
            );
            offsets.push(read.new_offset);
        }
        suite.case(&id, problems);
    }

    for probe in list(&f, "reader_probes") {
        let id = format!("reader_probes.{}", case_id(probe));
        let dir = temp_dir();
        let target = dir.path().join("transcript.jsonl");
        let rel = probe["file"].as_str().expect("file");
        std::fs::copy(fixture_path(rel), &target).expect("copy the probe transcript");
        let mut problems = Vec::new();
        let size = std::fs::metadata(&target).expect("metadata").len();
        if probe["file_size"].as_u64() != Some(size) {
            problems.push(format!("file size {size} != {}", probe["file_size"]));
        }
        let read = read_transcript(
            &path_str(&target),
            probe["last_offset"].as_u64().unwrap_or(0),
        );
        suite.computed(
            &id,
            "computed_result",
            &read_json(&read),
            &probe["computed_result"],
        );
        suite.case(&id, problems);
    }

    let mut problems = Vec::new();
    if MAX_LINE_BYTES != 1_048_576 {
        problems.push(format!("MAX_LINE_BYTES {MAX_LINE_BYTES} is not 1,048,576"));
    }
    suite.case("reader_rules.max_line_bytes", problems);

    // The entry parser.
    for case in list(&f, "parse_entry_cases") {
        let id = format!("parse_entry.{}", case_id(case));
        let observed = parse_entry(&case["input"])
            .as_ref()
            .map_or(Value::Null, entry_json);
        let expect = &case["expect"];
        let mut problems = Vec::new();
        if let Some(returns) = expect.get("returns") {
            if !json_eq(&observed, returns) {
                problems.push(format!("returns {returns}: got {}", short(&observed)));
            }
        } else if let Some(parts) = expect.get("text_contains").and_then(Value::as_array) {
            let text = observed["text"].as_str().unwrap_or_default();
            for part in parts.iter().filter_map(Value::as_str) {
                if !text.contains(part) {
                    problems.push(format!(
                        "text does not contain {part:?}: {}",
                        short(&observed)
                    ));
                }
            }
        } else if let Some(fields) = expect.as_object() {
            for (k, v) in fields {
                if !observed.get(k).is_some_and(|got| json_eq(got, v)) {
                    problems.push(format!("{k} {v}: got {}", short(&observed)));
                }
            }
        }
        suite.case(&id, problems);
        suite.computed(&id, "computed_result", &observed, &case["computed_result"]);
    }
    for probe in list(&f, "parse_entry_probes") {
        let id = format!("parse_entry_probes.{}", case_id(probe));
        let observed = parse_entry(&probe["input"])
            .as_ref()
            .map_or(Value::Null, entry_json);
        suite.computed(&id, "computed_result", &observed, &probe["computed_result"]);
        suite.case(&id, Vec::new());
    }

    // Version 2's extractor.
    for case in list(&f, "extractor_cases") {
        let id = format!("extractor.{}", case_id(case));
        let (texts, observed, lengths) = extract(&case["entries"]);
        suite.case(&id, check_extract(&texts, &case["expect"]));
        suite.computed(
            &id,
            "computed_result",
            &observed,
            &resolve(&case["computed_result"]),
        );
        suite.computed(&id, "computed_lengths", &lengths, &case["computed_lengths"]);
    }
    for probe in list(&f, "extractor_probes") {
        let id = format!("extractor_probes.{}", case_id(probe));
        let (_, observed, lengths) = extract(&probe["entries"]);
        suite.computed(
            &id,
            "computed_result",
            &observed,
            &resolve(&probe["computed_result"]),
        );
        suite.computed(
            &id,
            "computed_lengths",
            &lengths,
            &probe["computed_lengths"],
        );
        suite.case(&id, Vec::new());
    }
    // Master's per-entry (5,000) and combined (10,000) caps, which version 2
    // removed: the result must be version 2's, so master's expectation and
    // computed values are expected to differ.
    for case in list(&f, "extractor_superseded_by_v2") {
        let id = format!("extractor_superseded.{}", case_id(case));
        let (_, observed, lengths) = extract(&case["entries"]);
        let mut problems = Vec::new();
        let v2 = resolve(&case["computed_result_v2"]);
        if !json_eq(&observed, &v2) {
            problems.push(format!("v2 result {} != {}", short(&observed), short(&v2)));
        }
        if !json_eq(&lengths, &case["computed_lengths_v2"]) {
            problems.push(format!(
                "v2 lengths {lengths} != {}",
                case["computed_lengths_v2"]
            ));
        }
        suite.case(&id, problems);
        suite.computed(
            &id,
            "computed_lengths_master",
            &lengths,
            &case["computed_lengths_master"],
        );
    }

    // The invariants: one cap of 200,000 with the `...` marker, and no
    // smaller per-entry or combined cap (a 150,000 and a 49,999 character
    // text both pass whole).
    let mut problems = Vec::new();
    if f["extractor_rules_v2"]["safety_cap"].as_u64() != Some(MAX_CAPTURE_CHARS as u64) {
        problems.push(format!(
            "safety cap {MAX_CAPTURE_CHARS} != {}",
            f["extractor_rules_v2"]["safety_cap"]
        ));
    }
    if MAX_CAPTURE_CHARS != 200_000 || TRUNCATION_MARKER != "..." {
        problems.push(format!(
            "cap {MAX_CAPTURE_CHARS}, marker {TRUNCATION_MARKER:?}"
        ));
    }
    suite.case("invariant_assertions[0]", problems);
    let entries = vec![
        TranscriptEntry {
            role: "assistant".to_owned(),
            text: "a".repeat(150_000),
            timestamp: String::new(),
            uuid: String::new(),
            entry_type: "assistant".to_owned(),
        },
        TranscriptEntry {
            role: "assistant".to_owned(),
            text: "b".repeat(49_999),
            timestamp: String::new(),
            uuid: String::new(),
            entry_type: "assistant".to_owned(),
        },
    ];
    let out = extract_assistant_texts(&entries);
    let lens: Vec<usize> = out.texts.iter().map(|t| t.chars().count()).collect();
    let mut problems = Vec::new();
    if lens != [150_000, 49_999] || out.cap_hit {
        problems.push(format!("lengths {lens:?}, cap_hit {}", out.cap_hit));
    }
    suite.case("invariant_assertions[1]", problems);

    // The offset state is the daemon's: this crate's reader keeps none.
    for section in ["state_cases", "state_probes"] {
        for case in list(&f, section) {
            suite.not_applicable(
                &format!("{section}.{}", case_id(case)),
                "per-session offset state is kept by the daemon (stream E); this crate's reader is \
                 stateless and B-29's content-hash dedupe catches a re-read",
            );
        }
    }

    suite.finish(&[
        "extractor_superseded.master_individual_entry_truncated",
        "extractor_superseded.master_combined_budget_enforced",
        "extractor_superseded.master_partial_entry_at_budget_boundary",
    ]);
}
