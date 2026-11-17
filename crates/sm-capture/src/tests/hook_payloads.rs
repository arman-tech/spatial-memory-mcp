//! `hook_payloads/`: version 1's hook test cases, ported as fixtures (SPEC
//! section 14, phase 1, stream G's done-when; AT-13).
//!
//! Each PostToolUse, PreCompact and Stop payload runs through [`capture`]
//! as the daemon will run it, and the candidates are written as version 1's
//! queue files ([`queue_fields`]) so the payload's `checks` apply to them.
//! A Cursor payload runs in the form version 2's `_normalize_stdin` gave it
//! (`fixtures/capture/rust/cursor_normalized_payloads.json`): stream D
//! normalises Cursor input before the daemon sees it.
//!
//! Version 1 captured every tool but its skip rules; version 3 captures only
//! the tools in `capture.tools`, and none by default (B-160). To port
//! version 1's behaviour, each payload runs with the payload's own tool
//! added to version 1's list ([`V1_TOOLS`]), so version 1's skip rules
//! decide. [`at13_hook_payloads_default_settings`] checks that the default
//! list captures none of the AT-13 payloads, then runs them with version
//! 1's list named, as a user who wants tool capture names it.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Value, json};
use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;
use sm_core::types::{AnchorKind, MemoryType, Status};

use super::checker::check;
use super::{
    Suite, V1_TOOLS, claude_name, fixture, list, project, python_form, queue_fields, temp_dir,
    v1_tool_settings, with_tmp, without_unowned,
};
use crate::pipeline::{CaptureInput, CaptureOutcome, Skip, TextDrop, TextSource, capture};
use crate::session::SessionState;
use crate::signal::Tier;

const NOT_A_CAPTURE_EVENT: &str = "SessionStart and UserPromptSubmit are ambient recall: the \
    hook's stdout and its recall calls are stream D's and the daemon's (capture() returns \
    Skip::NotACaptureEvent; see v3.not_a_capture_event)";
const RAW_STDIN: &str = "the payload is raw stdin text or bytes, not a JSON object: reading and \
    parsing stdin is stream D's";

/// Why a check outside the queue files does not apply.
fn check_reason(path: &str) -> &'static str {
    if path.starts_with("stdout") {
        "the hook's stdout is stream D's"
    } else if path.starts_with("recall_calls") {
        "ambient recall is stream D's and the daemon's"
    } else if path.ends_with(".client") {
        "the client is stream D's: it normalises Cursor input before the daemon sees it"
    } else {
        "the project root of a queue file is CaptureInput.project, which the daemon resolves"
    }
}

/// Whether a check is on what this crate produces: the queue files, apart
/// from their `client` and `project_root_dir`.
fn ours(path: &str) -> bool {
    path.starts_with("queue_files")
        && !path.ends_with(".client")
        && !path.ends_with(".project_root_dir")
}

fn event_named(name: &str) -> HookEvent {
    *HookEvent::ALL
        .iter()
        .find(|e| claude_name(**e) == name)
        .unwrap_or_else(|| panic!("unknown event {name}"))
}

/// The Cursor payloads after version 2's `_normalize_stdin`, by id.
fn cursor_normalized() -> BTreeMap<String, Value> {
    let f = fixture("rust/cursor_normalized_payloads.json");
    list(&f, "payloads")
        .iter()
        .map(|p| {
            (
                p["id"].as_str().expect("an id").to_owned(),
                p["normalized"].clone(),
            )
        })
        .collect()
}

/// A payload ready to run: its files written to a fresh `{TMP}`.
struct Prepared {
    _dir: tempfile::TempDir,
    tmp: String,
    event: HookEvent,
    payload: Value,
    root: String,
}

fn prepare(doc: &Value, cursor: &BTreeMap<String, Value>) -> Prepared {
    let dir = temp_dir();
    let tmp = dir
        .path()
        .to_str()
        .expect("a UTF-8 temporary path")
        .replace('\\', "/");
    for file in doc["files"].as_array().into_iter().flatten() {
        let path = with_tmp(&file["path"], &tmp);
        let text = with_tmp(&file["text"], &tmp);
        std::fs::write(
            path.as_str().expect("a path"),
            text.as_str().expect("a text"),
        )
        .expect("write a fixture file");
    }
    let id = doc["id"].as_str().expect("an id");
    let raw = cursor.get(id).unwrap_or(&doc["stdin_json"]);
    let payload = with_tmp(raw, &tmp);
    // Version 1's project root: CLAUDE_PROJECT_DIR, else the working
    // directory, else the current directory.
    let root = doc["env"]["CLAUDE_PROJECT_DIR"]
        .as_str()
        .map(|r| r.replace("{TMP}", &tmp))
        .or_else(|| {
            payload["cwd"]
                .as_str()
                .filter(|c| !c.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| tmp.clone());
    Prepared {
        _dir: dir,
        tmp,
        event: event_named(doc["event"].as_str().expect("an event")),
        payload,
        root,
    }
}

/// Version 1's effective tool list for `payload`: version 1's default list
/// plus the payload's tool.
fn v1_settings(payload: &Value) -> CaptureSettings {
    let mut settings = v1_tool_settings();
    if let Some(tool) = payload["tool_name"].as_str().filter(|t| !t.is_empty())
        && !settings.captures_tool(tool)
    {
        settings.tools.push(tool.to_owned());
    }
    settings
}

fn run(p: &Prepared, settings: &CaptureSettings) -> CaptureOutcome {
    let project = project(Path::new(&p.root));
    let input = CaptureInput {
        event: p.event,
        payload: &p.payload,
        settings,
        project: &project,
        commit_sha: None,
        transcript_offset: None,
    };
    capture(&input, &mut SessionState::new())
}

/// The candidates as version 1's queue files, with version 1's placeholders.
fn observed(outcome: &CaptureOutcome) -> Value {
    let files: Vec<Value> = outcome
        .candidates
        .iter()
        .map(|c| {
            let mut file = queue_fields(c);
            file["content"] = json!(python_form(&c.content));
            file
        })
        .collect();
    json!({ "queue_files": files })
}

fn payload_file(entry: &Value) -> Value {
    fixture(&format!(
        "hook_payloads/{}",
        entry["file"].as_str().expect("a file")
    ))
}

/// Every payload of `hook_payloads/INDEX.json`.
#[test]
fn at13_hook_payloads() {
    let index = fixture("hook_payloads/INDEX.json");
    let cursor = cursor_normalized();
    let mut suite = Suite::new("hook_payloads/");
    let payloads = list(&index, "payloads");
    assert_eq!(
        index["count"].as_u64(),
        Some(payloads.len() as u64),
        "INDEX count"
    );

    let mut cursor_used = 0;
    for entry in payloads {
        let doc = payload_file(entry);
        let id = doc["id"].as_str().expect("an id").to_owned();
        let event = event_named(doc["event"].as_str().expect("an event"));
        if !event.is_capture() {
            suite.not_applicable(&id, NOT_A_CAPTURE_EVENT);
            continue;
        }
        if doc.get("stdin_json").is_none() {
            suite.not_applicable(&id, RAW_STDIN);
            continue;
        }
        let mut problems = Vec::new();
        if doc["client"] == "cursor" {
            if cursor.contains_key(&id) {
                cursor_used += 1;
            } else {
                problems
                    .push("no normalised form in rust/cursor_normalized_payloads.json".to_owned());
            }
        }
        let p = prepare(&doc, &cursor);
        let outcome = run(&p, &v1_settings(&p.payload));
        let obs = observed(&outcome);

        for (i, c) in doc["checks"].as_array().into_iter().flatten().enumerate() {
            let path = c["path"].as_str().expect("a check path");
            if ours(path) {
                if let Err(e) = check(&obs, c) {
                    problems.push(e);
                }
            } else {
                suite.not_applicable(&format!("{id}.checks[{i}] {path}"), check_reason(path));
            }
        }

        let expected = with_tmp(&doc["expected"], &p.tmp);
        let python: Vec<Value> = list(&expected, "queue_files")
            .iter()
            .map(without_unowned)
            .collect();
        suite.computed(
            &id,
            "expected.queue_files",
            &obs["queue_files"],
            &json!(python),
        );
        let python_offset = expected["state_files"]
            .as_object()
            .and_then(|m| m.values().next())
            .map_or(Value::Null, |state| state["last_offset"].clone());
        suite.computed(
            &id,
            "expected.state_files last_offset",
            &json!(outcome.new_offset),
            &python_offset,
        );
        if python_offset != Value::Null {
            suite.not_applicable(
                &format!("{id}.state_files.last_timestamp"),
                "the session's transcript state is the daemon's (stream E); this crate returns \
                 the new offset only",
            );
        }
        suite.case(&id, problems);
    }
    assert_eq!(
        cursor_used,
        cursor.len(),
        "every normalised Cursor payload is used"
    );

    for (what, reason) in [
        ("expected.stdout", "the hook's stdout is stream D's"),
        (
            "expected.queue_tmp_leftovers",
            "queue files are replaced by the spool (B-25, B-38)",
        ),
        (
            "expected.hook_errors_log",
            "the hook error log is replaced by the journal",
        ),
        (
            "expected.other_files",
            "files a hook run creates are stream D's and the spool's",
        ),
        (
            "variants",
            "the legacy per-event scripts and master's dispatcher; `expected` is version 2's \
             dispatcher, which version 3 ports",
        ),
    ] {
        suite.not_applicable(&format!("*.{what} (every payload)"), reason);
    }

    suite.finish(&[
        // Version 3 records the notebook as the provenance file path.
        "cc_post_tool_use_notebook_edit_decision",
        // serde_json objects keep keys sorted, so a JSON tool_response is
        // written with its keys in order.
        "cc_post_tool_use_tool_response_object",
    ]);
}

/// AT-13, library part. With the default settings no tool call is
/// captured (B-160). With version 1's tools named in `capture.tools` and
/// the other settings at their defaults, the Edit, Write and Bash payloads
/// give a candidate when the text carries a strong signal and nothing when
/// it is weak, and the fake credential is redacted with its kind reported.
#[test]
fn at13_hook_payloads_default_settings() {
    let cursor = BTreeMap::new();
    let defaults = CaptureSettings::default();
    for (name, tool) in [
        ("cc_post_tool_use_edit_decision", "Edit"),
        ("cc_post_tool_use_write_solution", "Write"),
        ("cc_post_tool_use_bash_aws_key_redacted", "Bash"),
        ("cc_post_tool_use_mcp_tool_fix", "mcp__db__query"),
    ] {
        let doc = fixture(&format!("hook_payloads/{name}.json"));
        let outcome = run(&prepare(&doc, &cursor), &defaults);
        assert_eq!(
            outcome.skipped,
            Some(Skip::ToolNotCaptured(tool.to_owned())),
            "{name} by default"
        );
        assert!(outcome.candidates.is_empty(), "{name} by default");
    }

    assert_eq!(V1_TOOLS, ["Edit", "Write", "Bash", "NotebookEdit"]);
    let settings = v1_tool_settings();
    let run_named = |name: &str| {
        let doc = fixture(&format!("hook_payloads/{name}.json"));
        let p = prepare(&doc, &cursor);
        run(&p, &settings)
    };

    for (name, anchor) in [
        ("cc_post_tool_use_edit_decision", Some("/src/db.py")),
        ("cc_post_tool_use_write_solution", Some("/docs/fix.md")),
        ("cc_post_tool_use_bash_aws_key_redacted", None),
    ] {
        let outcome = run_named(name);
        assert_eq!(outcome.skipped, None, "{name}");
        assert_eq!(outcome.source, Some(TextSource::ToolCall), "{name}");
        assert_eq!(outcome.candidates.len(), 1, "{name}: one candidate");
        let c = &outcome.candidates[0];
        assert_eq!(c.tier, Tier::Strong, "{name}");
        assert!(c.importance >= 0.7, "{name}: importance {}", c.importance);
        assert_eq!(c.status, Status::Pending, "{name}");
        let anchors: Vec<(AnchorKind, &str)> = c
            .anchors
            .iter()
            .map(|a| (a.kind, a.name.as_str()))
            .collect();
        assert_eq!(
            anchors,
            anchor
                .map(|a| (AnchorKind::File, a))
                .into_iter()
                .collect::<Vec<_>>(),
            "{name}"
        );
    }

    let aws = run_named("cc_post_tool_use_bash_aws_key_redacted");
    let c = &aws.candidates[0];
    assert!(c.content.contains("[REDACTED:aws_key]"), "{}", c.content);
    assert!(!c.content.contains("AKIA1234567890ABCDEF"), "{}", c.content);
    assert_eq!(c.redaction_kinds, ["aws_key"]);
    assert_eq!(
        c.memory_type,
        MemoryType::Pitfall,
        "the solution family is a pitfall"
    );

    let trivial = run_named("cc_post_tool_use_bash_trivial");
    assert!(trivial.candidates.is_empty());
    assert_eq!(trivial.drops, [TextDrop::Weak]);

    // The payload's text ("The fix is to use this key:" and a newline) is
    // weak in both engines: `.` stops at the newline and there is no full
    // stop, so the classifier stops it before the secret check does.
    let private_key = run_named("cc_post_tool_use_bash_private_key_skipped");
    assert!(private_key.candidates.is_empty());
    assert_eq!(private_key.drops, [TextDrop::Weak]);
    // With a strong signal, the private key stops it.
    let doc = fixture("hook_payloads/cc_post_tool_use_bash_private_key_skipped.json");
    let mut p = prepare(&doc, &cursor);
    let key = p.payload["tool_response"]
        .as_str()
        .expect("a response")
        .replacen(
            "The fix is to use this key:",
            "The fix is to rotate this key.",
            1,
        );
    p.payload["tool_response"] = json!(key);
    let strong_key = run(&p, &settings);
    assert!(strong_key.candidates.is_empty());
    assert_eq!(strong_key.drops, [TextDrop::Secret]);

    // A tool outside version 1's list is not captured, even with a signal.
    let mcp = run_named("cc_post_tool_use_mcp_tool_fix");
    assert_eq!(
        mcp.skipped,
        Some(Skip::ToolNotCaptured("mcp__db__query".to_owned()))
    );
    assert!(mcp.candidates.is_empty());
}

/// The ambient-recall payloads are not capture events.
#[test]
fn s16_hook_payloads_not_a_capture_event() {
    let index = fixture("hook_payloads/INDEX.json");
    let cursor = cursor_normalized();
    let mut seen = 0;
    for entry in list(&index, "payloads") {
        let doc = payload_file(entry);
        let event = event_named(doc["event"].as_str().expect("an event"));
        if event.is_capture() || doc.get("stdin_json").is_none() {
            continue;
        }
        seen += 1;
        let p = prepare(&doc, &cursor);
        let outcome = run(&p, &v1_settings(&p.payload));
        assert_eq!(
            outcome.skipped,
            Some(Skip::NotACaptureEvent(event)),
            "{}",
            doc["id"]
        );
        assert!(outcome.candidates.is_empty());
    }
    assert!(seen > 0, "the index holds ambient-recall payloads");
}

/// `event_aliases.json` `normalize_stdin_cases`: the hook client's
/// normalised payloads (stream D's) taken as capture input, as stream G's
/// file asks. Each case's `expected` is laid over a PostToolUse payload and
/// a PreCompact payload that carry a strong signal. Capture must read
/// Claude Code's field names from it, and only those: `session_id`,
/// `tool_response` and `trigger`; Cursor's `conversation_id`, `tool_output`,
/// `result_json` and `status` are never read here. The other fields are not
/// capture's: the daemon resolves the project (`cwd`, `workspace_roots`),
/// and `source` is SessionStart's.
#[test]
fn s16_cursor_normalised_payloads_are_read_by_claude_code_names() {
    const STRONG: &str = "Decided to use Redis for the session cache.";
    let f = fixture("event_aliases.json");
    let mut suite = Suite::new("event_aliases.json normalize_stdin_cases (as capture input)");
    let dir = temp_dir();
    let transcript = dir.path().join("session.jsonl");
    let entry = json!({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": STRONG}]},
        "isSidechain": false,
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "a1",
    });
    std::fs::write(&transcript, format!("{entry}\n")).expect("write the transcript");
    let transcript = transcript.to_str().expect("a UTF-8 path").to_owned();
    let project = project(dir.path());
    // The PostToolUse half needs its tool named (B-160).
    let settings = v1_tool_settings();
    let run_one = |event: HookEvent, payload: &Value| {
        let input = CaptureInput {
            event,
            payload,
            settings: &settings,
            project: &project,
            commit_sha: None,
            transcript_offset: None,
        };
        capture(&input, &mut SessionState::new())
    };

    for (i, case) in list(&f, "normalize_stdin_cases").iter().enumerate() {
        let id = format!(
            "normalize_stdin_cases[{i}] ({})",
            case["client"].as_str().unwrap_or_default()
        );
        let normalized = case["expected"].as_object().expect("an object");
        let field = |k: &str| normalized.get(k).and_then(Value::as_str);
        let mut problems = Vec::new();

        let mut post = json!({
            "tool_name": "Bash",
            "tool_input": {"command": STRONG},
            "tool_response": "base output",
        });
        let mut pre = json!({ "transcript_path": transcript });
        for (k, v) in normalized {
            post[k] = v.clone();
            pre[k] = v.clone();
        }

        let outcome = run_one(HookEvent::PostToolUse, &post);
        match outcome.candidates.as_slice() {
            [c] => {
                let context = &queue_fields(c)["context"];
                if context["session_id"].as_str() != field("session_id") {
                    problems.push(format!("PostToolUse session_id {}", context["session_id"]));
                }
                let response = match normalized.get("tool_response") {
                    None => Some("base output"),
                    Some(v) => v.as_str(),
                };
                if let Some(r) = response
                    && !c.content.contains(&format!("Output: {r}"))
                {
                    problems.push(format!(
                        "content {:?} lacks the tool_response {r:?}",
                        c.content
                    ));
                }
                for k in ["tool_output", "result_json"] {
                    if let Some(v) = field(k)
                        && Some(v) != response
                        && v.chars().count() >= 4
                        && c.content.contains(v)
                    {
                        problems.push(format!("content holds Cursor's {k} {v:?}"));
                    }
                }
            }
            other => problems.push(format!(
                "PostToolUse gave {} candidates ({:?}, {:?})",
                other.len(),
                outcome.skipped,
                outcome.drops
            )),
        }

        let outcome = run_one(HookEvent::PreCompact, &pre);
        match outcome.candidates.as_slice() {
            [c] => {
                let context = &queue_fields(c)["context"];
                if context["session_id"].as_str() != field("session_id") {
                    problems.push(format!("PreCompact session_id {}", context["session_id"]));
                }
                if context["trigger"].as_str() != field("trigger") {
                    problems.push(format!("PreCompact trigger {}", context["trigger"]));
                }
            }
            other => problems.push(format!(
                "PreCompact gave {} candidates ({:?}, {:?})",
                other.len(),
                outcome.skipped,
                outcome.drops
            )),
        }
        suite.case(&id, problems);
    }
    suite.finish(&[]);
}
