//! Every hook event, daemon side, over the real socket with the fake
//! embedder (SPEC 14, stream E, Tests; SPEC 6, ambient recall; SPEC 16;
//! AT-13 to AT-17).
//!
//! - `ambient_recall` runs the applicable cases and probes of
//!   `fixtures/capture/ambient_recall_cases.json`.
//! - `capture` runs every applicable payload of
//!   `fixtures/capture/hook_payloads/` for `post-tool-use`, `pre-compact`
//!   and `stop`, and the daemon stores what `sm_capture::capture` returns.
//! - `session-start` needs no daemon: the hook prints
//!   `sm_core::ambient::session_start_output` itself (stream D), and the
//!   daemon has no operation for it. A `session-start` or
//!   `user-prompt-submit` capture is refused with `accepted: false`.
//!
//! Captures are processed by one worker in the order they are queued, so a
//! sentinel capture queued last and seen stored ([`barrier`]) means every
//! earlier one has been processed.

mod common;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use common::*;
use serde_json::{Value, json};
use sm_capture::{CaptureInput, SessionState, capture};
use sm_core::ambient;
use sm_core::config::CaptureSettings;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::journal::{HookGate, metrics};
use sm_core::project::resolve_project;
use sm_core::protocol::{
    AmbientParams, AmbientReply, CaptureParams, HookEvent, Op, Reply, Request,
};
use sm_core::spool::{STALE_TMP_AGE, read_spool, write_spool};
use sm_core::testing::{FakeEmbedder, FakeProvider, SimpleChunker, dot, fake_info, fake_vector};
use sm_core::time::Timestamp;
use sm_core::traits::{
    ChunkParams, ChunkPlan, Chunker, DocInput, Embedder, EmbedderInfo, EmbedderProvider,
    SessionKind,
};
use sm_core::types::{
    Client as HookClient, Memory, Origin, ProvenanceSource, Scope, Status, derive_title,
};
use sm_daemon::{Exit, LoadedModel, UnwiredSharing};
use sm_store::{MemoryListFilter, Store};

const WAIT: Duration = Duration::from_secs(20);

// --- helpers ---------------------------------------------------------------

/// The memories of the project `cwd` resolves to; none when the daemon has
/// not seen the project yet.
fn project_memories(store: &Store, cwd: &str) -> Vec<Memory> {
    let identity = resolve_project(Path::new(cwd)).expect("a project");
    match store
        .project_by_identity(&identity.key())
        .expect("find the project")
    {
        Some(p) => store
            .list_memories(&MemoryListFilter {
                project: Some(p.id),
                ..Default::default()
            })
            .expect("list memories"),
        None => Vec::new(),
    }
}

fn contents(memories: &[Memory]) -> Vec<String> {
    let mut v: Vec<String> = memories.iter().map(|m| m.content.clone()).collect();
    v.sort();
    v
}

fn capture_req(
    d: &Daemon,
    cwd: &str,
    session: Option<&str>,
    event: HookEvent,
    payload: Value,
) -> Request {
    d.req(
        cwd,
        Op::Capture(CaptureParams {
            event,
            payload,
            received_at: Timestamp::now(),
        }),
    )
    .with_session(session.map(str::to_owned))
}

/// Sends a capture and returns `accepted`.
fn send_capture(d: &Daemon, req: &Request) -> bool {
    match d.send(req).into_result() {
        Ok(Reply::Captured(r)) => r.accepted,
        Ok(other) => panic!("capture replied {other:?}"),
        Err(e) => panic!("capture failed: {}: {}", e.code.as_str(), e.message),
    }
}

/// Queues a capture and returns its request id.
fn queue(d: &Daemon, cwd: &str, session: Option<&str>, event: HookEvent, payload: Value) -> String {
    let req = capture_req(d, cwd, session, event, payload);
    assert!(send_capture(d, &req), "the capture was accepted");
    req.id
}

fn edit(text: &str) -> Value {
    json!({"tool_name": "Edit", "tool_input": {"new_string": text}, "tool_response": "ok"})
}

/// Waits until every capture queued before it has been processed (one
/// worker, first in first out).
fn barrier(d: &Daemon, rig: &Rig, store: &Store) {
    let cwd = rig.project(&unique("barrier"));
    let text = format!("Decided to use {} for the barrier.", unique("sentinel"));
    queue(d, &cwd, None, HookEvent::PostToolUse, edit(&text));
    assert!(
        wait_until(WAIT, || !project_memories(store, &cwd).is_empty()),
        "the barrier capture was stored"
    );
}

/// Replaces the fixtures' `{TMP}` placeholder in every string.
fn with_tmp(v: &Value, tmp: &str) -> Value {
    match v {
        Value::String(s) => Value::String(s.replace("{TMP}", tmp)),
        Value::Array(a) => Value::Array(a.iter().map(|x| with_tmp(x, tmp)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), with_tmp(x, tmp)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// One assistant entry of a Claude Code transcript, with its newline.
fn entry(text: &str, n: usize) -> String {
    let v = json!({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "isSidechain": false,
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": format!("u{n}"),
    });
    format!("{v}\n")
}

/// Every journal line for `metric`.
fn journal_lines(rig: &Rig, metric: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(rig.paths.metrics_dir()) else {
        return out;
    };
    for e in entries.flatten() {
        let text = std::fs::read_to_string(e.path()).unwrap_or_default();
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line)
                && v["metric"] == metric
            {
                out.push(v);
            }
        }
    }
    out
}

/// The journal lines for `metric` written for one request.
fn journal_for(rig: &Rig, metric: &str, request_id: &str) -> Vec<Value> {
    journal_lines(rig, metric)
        .into_iter()
        .filter(|v| v["request_id"] == request_id)
        .collect()
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// What `sm_capture::capture` gives for the payload, as the daemon runs it
/// for a new session: the project `cwd` resolves to, its head commit,
/// offset 0 and fresh hashes; deduplicated by content hash as the store
/// does.
fn expected_shapes(
    event: HookEvent,
    payload: &Value,
    settings: &CaptureSettings,
    cwd: &str,
) -> Vec<Value> {
    let identity = resolve_project(Path::new(cwd)).expect("a project");
    let commit = identity.repo.as_ref().and_then(|r| r.head_commit());
    let outcome = capture(
        &CaptureInput {
            event,
            payload,
            settings,
            project: &identity,
            commit_sha: commit.as_deref(),
            transcript_offset: 0,
        },
        &mut SessionState::new(),
    );
    let mut seen = HashSet::new();
    let mut shapes: Vec<Value> = outcome
        .candidates
        .iter()
        .filter(|c| seen.insert(c.content_hash.clone()))
        .map(|c| {
            json!({
                "content": c.content,
                "content_hash": c.content_hash,
                "type": c.memory_type,
                "status": c.status,
                "importance": c.importance,
                "tags": c.tags,
                "anchors": c.anchors,
                "provenance": c.provenance,
                "origin": Origin::Local,
                "title": derive_title(&c.content),
            })
        })
        .collect();
    shapes.sort_by(|a, b| a["content_hash"].as_str().cmp(&b["content_hash"].as_str()));
    shapes
}

fn stored_shapes(memories: &[Memory]) -> Vec<Value> {
    let mut shapes: Vec<Value> = memories
        .iter()
        .map(|m| {
            json!({
                "content": m.content,
                "content_hash": m.content_hash,
                "type": m.memory_type,
                "status": m.status,
                "importance": m.importance,
                "tags": m.tags,
                "anchors": m.anchors,
                "provenance": m.provenance,
                "origin": m.origin,
                "title": m.title,
            })
        })
        .collect();
    shapes.sort_by(|a, b| a["content_hash"].as_str().cmp(&b["content_hash"].as_str()));
    shapes
}

// --- capture: hook_payloads/ -------------------------------------------------

const NOT_A_CAPTURE_EVENT: &str = "SessionStart and UserPromptSubmit are not captures: the \
     daemon refuses them with accepted:false; their output is the hook's (stream D) and \
     UserPromptSubmit's recall is ambient_recall, tested below";
const RAW_STDIN: &str = "the payload is raw stdin text or bytes, not a JSON object: reading and \
     parsing stdin is stream D's";

/// The index names events as Claude Code does (`PostToolUse`); Cursor's
/// `argv` uses its own (`postToolUse`), so the name is matched, not parsed.
fn event_of(entry: &Value) -> HookEvent {
    let name = entry["event"].as_str().expect("an event");
    HookEvent::ALL
        .iter()
        .copied()
        .find(|e| format!("{e:?}") == name)
        .unwrap_or_else(|| panic!("unknown event in {entry}"))
}

/// Version 1's placeholders `[REDACTED_<KIND>]` written as version 3's
/// markers `[REDACTED:<kind>]` (sm-capture's secret scanner), so the
/// fixtures' checks apply to what version 3 stores.
fn v3_markers(s: &str) -> String {
    const OPEN: &str = "[REDACTED_";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(OPEN) {
        out.push_str(&rest[..i]);
        let after = &rest[i + OPEN.len()..];
        let kind_len = after
            .find(|c: char| !(c.is_ascii_uppercase() || c == '_'))
            .unwrap_or(after.len());
        if kind_len > 0 && after[kind_len..].starts_with(']') {
            out.push_str(&format!(
                "[REDACTED:{}]",
                after[..kind_len].to_ascii_lowercase()
            ));
            rest = &after[kind_len + 1..];
        } else {
            out.push_str(OPEN);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// The Cursor payloads after stream D's normalisation, by id.
fn cursor_normalized() -> BTreeMap<String, Value> {
    fixture("capture/rust/cursor_normalized_payloads.json")["payloads"]
        .as_array()
        .expect("payloads")
        .iter()
        .map(|p| {
            (
                p["id"].as_str().expect("an id").to_owned(),
                p["normalized"].clone(),
            )
        })
        .collect()
}

/// Applies a payload's `queue_files` length and content checks to what the
/// daemon stored. The other checks are on candidate fields the stored
/// memory does not keep (tier, patterns, version 1's namespace, the
/// source hook) and are covered by comparing with `sm_capture::capture`.
fn apply_checks(id: &str, doc: &Value, stored: &[Memory], problems: &mut Vec<String>) {
    for c in doc["checks"].as_array().into_iter().flatten() {
        let path = c["path"].as_str().unwrap_or_default();
        let op = c["op"].as_str().unwrap_or_default();
        let ok = match (path, op) {
            ("queue_files.length", "eq") => Some(c["value"].as_u64() == Some(stored.len() as u64)),
            ("queue_files.0.content", _) => {
                let content = stored.first().map(|m| m.content.as_str());
                let value = v3_markers(c["value"].as_str().unwrap_or_default());
                let value = value.as_str();
                match op {
                    "contains" => Some(content.is_some_and(|t| t.contains(value))),
                    "not_contains" => Some(content.is_some_and(|t| !t.contains(value))),
                    "truthy" => Some(content.is_some_and(|t| !t.is_empty())),
                    _ => None,
                }
            }
            _ => None,
        };
        if ok == Some(false) {
            problems.push(format!("{id}: check {c} failed on {stored:?}"));
        }
    }
}

/// AT-13 (daemon side): every PostToolUse, PreCompact and Stop payload of
/// `hook_payloads/INDEX.json` is captured over the socket, and the daemon
/// stores exactly what `sm_capture::capture` returns for it, with the
/// derived title. As in stream G's port, each payload's own tool is added
/// to the default list so version 1's skip rules decide.
#[test]
fn at13_hook_payloads_over_the_socket() {
    let index = fixture("capture/hook_payloads/INDEX.json");
    let payloads = index["payloads"].as_array().expect("payloads");
    assert_eq!(index["count"].as_u64(), Some(payloads.len() as u64));
    let cursor = cursor_normalized();

    let mut rig = Rig::new();
    for e in payloads {
        let doc = fixture(&format!(
            "capture/hook_payloads/{}",
            e["file"].as_str().expect("a file")
        ));
        if let Some(tool) = doc["stdin_json"]["tool_name"]
            .as_str()
            .filter(|t| !t.is_empty())
            && !rig.settings.capture.captures_tool(tool)
        {
            rig.settings.capture.tools.push(tool.to_owned());
        }
    }
    let settings = rig.settings.capture.clone();
    let d = rig.start_fake();
    let store = rig.store();

    struct Sent {
        id: String,
        doc: Value,
        cwd: String,
        expected: Vec<Value>,
    }
    let mut sent = Vec::new();
    let mut not_applicable = Vec::new();
    for (i, e) in payloads.iter().enumerate() {
        let doc = fixture(&format!(
            "capture/hook_payloads/{}",
            e["file"].as_str().expect("a file")
        ));
        let id = doc["id"].as_str().expect("an id").to_owned();
        let event = event_of(e);
        let cwd = rig.project(&format!("p{i}"));
        if !event.is_capture() {
            // The daemon refuses a non-capture event whatever the payload.
            let req = capture_req(
                &d,
                &cwd,
                None,
                event,
                doc.get("stdin_json").cloned().unwrap_or(Value::Null),
            );
            assert!(!send_capture(&d, &req), "{id}: not a capture event");
            not_applicable.push((id, NOT_A_CAPTURE_EVENT));
            continue;
        }
        if doc.get("stdin_json").is_none() {
            not_applicable.push((id, RAW_STDIN));
            continue;
        }
        let tmp = cwd.replace('\\', "/");
        for f in doc["files"].as_array().into_iter().flatten() {
            let path = with_tmp(&f["path"], &tmp);
            let path = Path::new(path.as_str().expect("a path"));
            std::fs::create_dir_all(path.parent().expect("a parent")).expect("a folder");
            std::fs::write(path, with_tmp(&f["text"], &tmp).as_str().expect("a text"))
                .expect("write a fixture file");
        }
        let is_cursor = doc["client"] == "cursor";
        let raw = if is_cursor {
            cursor
                .get(&id)
                .unwrap_or_else(|| panic!("{id}: no normalised Cursor form"))
        } else {
            &doc["stdin_json"]
        };
        let payload = with_tmp(raw, &tmp);
        let expected = expected_shapes(event, &payload, &settings, &cwd);
        // The fixtures share one session id across unrelated transcripts;
        // each payload is its own session here, as in its own hook run.
        let session = unique("payload-session");
        let mut req = capture_req(&d, &cwd, Some(&session), event, payload);
        if is_cursor {
            req = req.with_client(HookClient::Cursor);
        }
        assert!(send_capture(&d, &req), "{id}: accepted");
        sent.push(Sent {
            id,
            doc,
            cwd,
            expected,
        });
    }
    barrier(&d, &rig, &store);

    let mut problems = Vec::new();
    let mut stored_any = 0;
    for s in &sent {
        let stored = project_memories(&store, &s.cwd);
        stored_any += usize::from(!stored.is_empty());
        let shapes = stored_shapes(&stored);
        if shapes != s.expected {
            problems.push(format!(
                "{}: stored {shapes:#?}\nexpected {:#?}",
                s.id, s.expected
            ));
        }
        apply_checks(&s.id, &s.doc, &stored, &mut problems);
    }
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
    assert!(stored_any >= 3, "some payloads store memories");
    assert!(
        files_in(&rig.paths.spool_failed()).is_empty(),
        "no capture failed"
    );
    eprintln!(
        "hook_payloads: {} captured and compared, {} not applicable:",
        sent.len(),
        not_applicable.len()
    );
    for (id, why) in &not_applicable {
        eprintln!("  {id}: {why}");
    }
}

/// AT-13 with the default settings: the Edit, Write and Bash payloads store
/// strong-signal text as an active capture with importance at least 0.7,
/// the fake AWS key redacted, and the trivial Bash payload stores nothing.
#[test]
fn at13_default_settings_store_strong_text_and_nothing_for_weak() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let mut cases = Vec::new();
    for (name, n) in [
        ("cc_post_tool_use_edit_decision", 1),
        ("cc_post_tool_use_write_solution", 1),
        ("cc_post_tool_use_bash_aws_key_redacted", 1),
        ("cc_post_tool_use_bash_trivial", 0),
    ] {
        let doc = fixture(&format!("capture/hook_payloads/{name}.json"));
        let cwd = rig.project(name);
        let payload = with_tmp(&doc["stdin_json"], &cwd.replace('\\', "/"));
        queue(
            &d,
            &cwd,
            Some(&unique("s")),
            HookEvent::PostToolUse,
            payload,
        );
        cases.push((name, n, cwd));
    }
    barrier(&d, &rig, &store);
    for (name, n, cwd) in cases {
        let stored = project_memories(&store, &cwd);
        assert_eq!(stored.len(), n, "{name}");
        for m in &stored {
            assert_eq!(m.status, Status::Active, "{name}");
            assert!(m.importance >= 0.7, "{name}: importance {}", m.importance);
            assert_eq!(m.provenance.source, ProvenanceSource::Capture, "{name}");
            assert_eq!(m.origin, Origin::Local, "{name}");
        }
        if name.ends_with("aws_key_redacted") {
            let c = &stored[0].content;
            assert!(c.contains("[REDACTED:aws_key]"), "{c}");
            assert!(!c.contains("AKIA1234567890ABCDEF"), "{c}");
        }
    }
}

// --- capture: transcript offsets ---------------------------------------------

/// SPEC 16 (transcript offsets, `transcript/transcript_cases.json`): the
/// daemon keeps one offset per session. A session reads what was appended
/// since its last read (`reads_appended_entries`, `save_and_load`,
/// `save_overwrites_previous`) and nothing before it (`reads_from_offset`:
/// a line rewritten in place before the offset is not read again); a new
/// session starts at 0 (`load_missing_returns_defaults`); a file shorter
/// than the offset is read from 0 (`offset_beyond_file_size_resets`); a
/// capture with no session id keeps no state (`load_empty_session_id`,
/// `save_empty_session_id_noop`), so two sessionless transcripts never
/// share an offset.
///
/// Not applicable: `load_corrupted_json_returns_defaults` and the
/// `state_probes` (string, float and non-numeric offsets, an array
/// document, an unsafe session id). Version 3 keeps the offsets in the
/// daemon's memory: there is no state file to corrupt and no path built
/// from a session id.
#[test]
fn s16_transcript_offsets_per_session() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("offsets");
    let transcript = Path::new(&cwd).join("transcript.jsonl");
    let path = transcript.to_str().expect("UTF-8").to_owned();
    let pre_compact = |session: &str| {
        queue(
            &d,
            &cwd,
            Some(session),
            HookEvent::PreCompact,
            json!({"session_id": session, "transcript_path": path, "trigger": "auto"}),
        )
    };
    let has = |text: &str| contents(&project_memories(&store, &cwd)).contains(&text.to_owned());

    let x = "Decided to use SQLite for the local cache.";
    let y = "Decided to use DuckDB for the local cache.";
    let b = "The fix is to add a retry around the upload call.";
    let z = "The bug was a stale connection pool in the worker.";
    let w = "The fix was adding a null check in the handler.";
    assert_eq!(entry(x, 1).len(), entry(y, 1).len(), "same line length");

    let s = unique("s");
    std::fs::write(&transcript, entry(x, 1)).expect("write");
    pre_compact(&s);
    barrier(&d, &rig, &store);
    assert!(has(x), "the first read stores X");

    std::fs::write(&transcript, entry(x, 1) + &entry(b, 2)).expect("append");
    let second = pre_compact(&s);
    barrier(&d, &rig, &store);
    assert!(has(b), "the appended entry is stored");
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &second).len(),
        1,
        "the second read gives only the appended entry"
    );

    // X rewritten in place as Y (same length), then Z appended: the session
    // reads from its offset, so Y is never seen.
    std::fs::write(&transcript, entry(y, 1) + &entry(b, 2) + &entry(z, 3)).expect("rewrite");
    let third = pre_compact(&s);
    barrier(&d, &rig, &store);
    assert!(has(z), "the entry after the offset is stored");
    assert!(!has(y), "the entry before the offset is not read again");
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &third).len(),
        1
    );

    // A new session reads the same file from 0.
    let s2 = unique("s");
    let fresh = pre_compact(&s2);
    barrier(&d, &rig, &store);
    assert!(has(y), "a new session reads from offset 0");
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &fresh).len(),
        3,
        "Y, B and Z are candidates for the new session"
    );

    // The file is now shorter than session S's offset: read from 0.
    std::fs::write(&transcript, entry(w, 1)).expect("shorten");
    assert!(
        std::fs::metadata(&transcript).expect("meta").len()
            < (entry(y, 1).len() + entry(b, 2).len() + entry(z, 3).len()) as u64
    );
    pre_compact(&s);
    barrier(&d, &rig, &store);
    assert!(has(w), "a file shorter than the offset is read from 0");

    // Sessionless captures keep no state between transcripts.
    let t1 = Path::new(&cwd).join("t1.jsonl");
    let t2 = Path::new(&cwd).join("t2.jsonl");
    let k = "Decided to use Kafka for the event bus.";
    let f1 = "Decided to use NATS for the internal message bus because it is light and easy \
              to run in tests.";
    let f2 = "The bug was a missing index on the events table.";
    std::fs::write(&t1, entry(k, 1)).expect("write t1");
    std::fs::write(&t2, entry(f1, 1) + &entry(f2, 2)).expect("write t2");
    assert!(
        entry(f1, 1).len() > entry(k, 1).len(),
        "F1's line is longer than T1"
    );
    for t in [&t1, &t2] {
        queue(
            &d,
            &cwd,
            None,
            HookEvent::PreCompact,
            json!({"transcript_path": t.to_str().expect("UTF-8"), "trigger": "auto"}),
        );
    }
    barrier(&d, &rig, &store);
    assert!(has(k), "the first sessionless transcript is stored");
    assert!(
        has(f1),
        "the second sessionless transcript is read from 0, not from the first one's offset"
    );
    assert!(has(f2));
}

// --- capture: AT-14 -----------------------------------------------------------

const CAPTURED_EDIT: &str = "Decided to use PostgreSQL because it handles JSONB well.";
const CAPTURED_WRITE: &str = "The fix is to add the missing import statement.";
const MISSED: &str = "Decided to use Redis for the session cache.";

/// A project holding a copy of `rust/at14_session.jsonl`, and PostToolUse
/// captures of two of its texts in `session`. Returns the folder and the
/// transcript path.
fn at14_session(d: &Daemon, rig: &Rig, session: &str) -> (String, String) {
    let cwd = rig.project(&unique("at14"));
    let transcript = Path::new(&cwd).join("at14_session.jsonl");
    std::fs::copy(fixture_path("capture/rust/at14_session.jsonl"), &transcript)
        .expect("copy the transcript");
    for (tool, input) in [
        ("Edit", json!({"new_string": CAPTURED_EDIT})),
        ("Write", json!({"content": CAPTURED_WRITE})),
    ] {
        queue(
            d,
            &cwd,
            Some(session),
            HookEvent::PostToolUse,
            json!({"session_id": "s1", "tool_name": tool, "tool_input": input, "tool_response": "ok"}),
        );
    }
    (cwd, transcript.to_str().expect("UTF-8").to_owned())
}

/// AT-14 (daemon side): PreCompact and Stop store what PostToolUse missed
/// and not what it already captured. The store's dedupe by content hash
/// would hide a second copy, so the journal shows it: the PreCompact
/// request stores one candidate, not three.
#[test]
fn at14_transcript_stores_only_what_post_tool_use_missed() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();

    // PreCompact.
    let session = unique("s1");
    let (cwd, path) = at14_session(&d, &rig, &session);
    let pre = queue(
        &d,
        &cwd,
        Some(&session),
        HookEvent::PreCompact,
        json!({"session_id": "s1", "transcript_path": path, "trigger": "auto"}),
    );
    barrier(&d, &rig, &store);
    let stored = project_memories(&store, &cwd);
    let mut want = vec![
        CAPTURED_EDIT.to_owned(),
        CAPTURED_WRITE.to_owned(),
        MISSED.to_owned(),
    ];
    want.sort();
    assert_eq!(contents(&stored), want);
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &pre).len(),
        1,
        "PreCompact gave only the missed text"
    );
    let missed = stored
        .iter()
        .find(|m| m.content == MISSED)
        .expect("the missed text");
    assert_eq!(missed.tags, ["transcript", "decision"]);
    assert!(
        missed.anchors.is_empty(),
        "a transcript capture has no anchor"
    );
    assert_eq!(missed.provenance.source, ProvenanceSource::Capture);
    assert_eq!(
        missed
            .provenance
            .extra
            .as_ref()
            .map(|e| e["trigger"].clone()),
        Some(json!("auto"))
    );

    // Stop without a final message reads the transcript.
    let session = unique("s1");
    let (cwd, path) = at14_session(&d, &rig, &session);
    let stop = queue(
        &d,
        &cwd,
        Some(&session),
        HookEvent::Stop,
        json!({"session_id": "s1", "transcript_path": path}),
    );
    barrier(&d, &rig, &store);
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &stop).len(),
        1
    );
    let stored = project_memories(&store, &cwd);
    let missed = stored
        .iter()
        .find(|m| m.content == MISSED)
        .expect("Stop stores the missed text");
    assert_eq!(
        missed
            .provenance
            .extra
            .as_ref()
            .map(|e| e["trigger"].clone()),
        Some(json!("session_end"))
    );

    // Stop with a final message stores it and does not read the transcript.
    let session = unique("s1");
    let (cwd, path) = at14_session(&d, &rig, &session);
    let message = "The workaround is to pin the driver to version 2.";
    queue(
        &d,
        &cwd,
        Some(&session),
        HookEvent::Stop,
        json!({"session_id": "s1", "transcript_path": path, "last_assistant_message": message}),
    );
    barrier(&d, &rig, &store);
    let mut want = vec![
        CAPTURED_EDIT.to_owned(),
        CAPTURED_WRITE.to_owned(),
        message.to_owned(),
    ];
    want.sort();
    assert_eq!(contents(&project_memories(&store, &cwd)), want);
}

// --- capture: AT-16, and AT-17's index cap ----------------------------------

/// AT-16 (daemon side): a medium signal is stored active with importance at
/// least 0.4 and a strong one at least 0.7, from each capture source.
#[test]
fn at16_tiers_are_stored_active() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let mut cases = Vec::new();
    for (text, floor) in [
        (
            "The convention here is to use snake_case for module names.",
            0.4,
        ),
        ("You need to set RUST_LOG=debug for verbose output.", 0.4),
        ("Decided to use Redis for the session cache.", 0.7),
        ("The bug was a stale connection pool.", 0.7),
    ] {
        for event in [
            HookEvent::PostToolUse,
            HookEvent::PreCompact,
            HookEvent::Stop,
        ] {
            let cwd = rig.project(&unique("at16"));
            let transcript = Path::new(&cwd).join("t.jsonl");
            std::fs::write(&transcript, entry(text, 1)).expect("write a transcript");
            let path = transcript.to_str().expect("UTF-8").to_owned();
            let payload = match event {
                HookEvent::PostToolUse => edit(text),
                HookEvent::PreCompact => json!({"transcript_path": path}),
                _ => json!({"last_assistant_message": text}),
            };
            queue(&d, &cwd, Some(&unique("s")), event, payload);
            cases.push((text, floor, event, cwd));
        }
    }
    barrier(&d, &rig, &store);
    for (text, floor, event, cwd) in cases {
        let stored = project_memories(&store, &cwd);
        assert_eq!(contents(&stored), [text], "{event:?}");
        let m = &stored[0];
        assert_eq!(m.status, Status::Active, "{event:?} {text}");
        assert!(
            m.importance >= floor && m.importance <= 1.0,
            "{event:?} {text}: importance {}",
            m.importance
        );
        assert_eq!(m.provenance.source, ProvenanceSource::Capture);
    }
}

/// AT-17 (daemon side, SPEC 11 lists the index cap there): a capture over
/// `capture.max_index_tokens` is stored whole, indexed up to the cap, and
/// journalled as a cap hit with its full size and the tokens indexed.
#[test]
fn at17_capture_over_the_index_cap_is_stored_whole() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("cap");
    let cap = rig.settings.capture.max_index_tokens as usize;
    let mut text = String::from(CAPTURED_EDIT);
    for i in 0..cap + 800 {
        text.push_str(&format!(" filler{i}"));
    }
    let id = queue(&d, &cwd, None, HookEvent::PostToolUse, edit(&text));
    assert!(
        wait_until(WAIT, || !project_memories(&store, &cwd).is_empty()),
        "the capture was stored"
    );
    let stored = project_memories(&store, &cwd);
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].content, text, "stored whole");

    let params = |index_cap| ChunkParams {
        max_tokens: rig.settings.chunk.max_tokens as usize,
        overlap: rig.settings.chunk.overlap,
        index_cap,
    };
    let capped = SimpleChunker
        .chunk(&text, &params(Some(cap)))
        .expect("chunk");
    let whole = SimpleChunker.chunk(&text, &params(None)).expect("chunk");
    assert!(capped.capped && capped.indexed_tokens <= cap);
    assert!(capped.pieces.len() < whole.pieces.len());
    assert_eq!(
        d.status(&cwd).index.chunks,
        capped.pieces.len() as u64,
        "indexed up to the cap"
    );

    assert!(wait_until(WAIT, || {
        !journal_for(&rig, metrics::CAPTURE_CAP_HIT_TOTAL, &id).is_empty()
    }));
    let size = journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &id);
    assert_eq!(size.len(), 1);
    assert_eq!(size[0]["value"], json!(whole.total_tokens as f64));
    assert_eq!(
        size[0]["labels"]["indexed_tokens"],
        json!(capped.indexed_tokens)
    );
}

// --- capture: AT-15, the spool ------------------------------------------------

/// AT-15 (daemon side): a capture spooled while no daemon runs is processed
/// when the daemon starts and its file is gone; a corrupt file moves to
/// `spool/failed/`; a stale `tmp/` file is removed and a fresh one kept; a
/// file written while the daemon runs is taken within one drain interval.
#[test]
fn at15_spool_is_drained() {
    let mut rig = Rig::new();
    let interval = Duration::from_millis(500);
    rig.opts.spool_interval = interval;
    let cwd = rig.project("spooled");
    let spooled_req = |text: &str| {
        Request::new(
            cwd.as_str(),
            rig.profile(),
            Op::Capture(CaptureParams {
                event: HookEvent::PostToolUse,
                payload: edit(text),
                received_at: Timestamp::now(),
            }),
        )
        .with_session(Some(unique("spool")))
    };
    let first = "Decided to use Redis for the spool test.";
    let second = "Decided to use Kafka for the spool test.";
    let (tmp, new, failed) = (
        rig.paths.spool_tmp(),
        rig.paths.spool_new(),
        rig.paths.spool_failed(),
    );
    let spooled = write_spool(&tmp, &new, &spooled_req(first)).expect("spool");
    let corrupt = new.join("corrupt.json");
    std::fs::write(&corrupt, "not json").expect("write a corrupt file");
    let stale = tmp.join("stale.json");
    let fresh = tmp.join("fresh.json");
    std::fs::write(&stale, "{").expect("write");
    std::fs::write(&fresh, "{").expect("write");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&stale)
        .expect("open")
        .set_modified(SystemTime::now() - STALE_TMP_AGE - Duration::from_secs(60))
        .expect("age the file");

    let _daemon = rig.start_fake();
    let store = rig.store();
    assert!(
        wait_until(WAIT, || contents(&project_memories(&store, &cwd))
            == [first.to_owned()]
            && !spooled.exists()),
        "the spooled capture was stored and its file removed"
    );
    assert!(wait_until(WAIT, || !corrupt.exists()));
    assert!(failed.join("corrupt.json").exists(), "quarantined");
    assert!(!stale.exists(), "the stale tmp file was removed");
    assert!(fresh.exists(), "a fresh tmp file is kept");

    let start = Instant::now();
    let later = write_spool(&tmp, &new, &spooled_req(second)).expect("spool");
    assert!(
        wait_until(interval + Duration::from_secs(1), || !later.exists()),
        "taken within one drain interval"
    );
    let taken = start.elapsed();
    assert!(wait_until(WAIT, || {
        contents(&project_memories(&store, &cwd)).contains(&second.to_owned())
    }));
    eprintln!(
        "a file written while the daemon runs was taken in {taken:?} (interval {interval:?})"
    );
    assert_eq!(files_in(&failed), ["corrupt.json"]);
}

// --- capture: failures, retries and shutdown (e-report decisions 27, 28) -------

/// Content holding this word fails to chunk once, with a storage error.
const FAIL_ONCE: &str = "FAILONCE";
/// Content holding this word always fails to chunk, with a storage error.
const ALWAYS_FAIL: &str = "ALWAYSFAIL";
/// Content holding this word panics the chunker.
const PANICS: &str = "PANICS";
/// Content holding this word blocks the chunker until [`FaultChunker::open`].
const HOLD: &str = "HOLDHERE";

/// The attempts the daemon makes at one capture (capture.rs `MAX_ATTEMPTS`).
const MAX_ATTEMPTS: usize = 5;

/// [`SimpleChunker`], with faults for content holding a marker word. The
/// capture worker chunks each candidate between `sm_capture::capture` and
/// the store's insert, so a fault here is a failure after the pipeline has
/// moved the session's offset and hashes and before anything is stored.
#[derive(Default)]
struct FaultChunker {
    /// Every content chunked, in order.
    seen: Mutex<Vec<String>>,
    failed_once: Mutex<HashSet<String>>,
    gate: Mutex<bool>,
    cv: Condvar,
    /// Calls that reached the [`HOLD`] gate.
    entered: AtomicUsize,
}

fn guard<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FaultChunker {
    fn calls_with(&self, marker: &str) -> usize {
        guard(&self.seen)
            .iter()
            .filter(|t| t.contains(marker))
            .count()
    }

    fn entered(&self) -> usize {
        self.entered.load(Ordering::SeqCst)
    }

    fn open(&self) {
        *guard(&self.gate) = true;
        self.cv.notify_all();
    }
}

impl Chunker for FaultChunker {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        SimpleChunker.count_tokens(text)
    }

    fn chunk(&self, content: &str, params: &ChunkParams) -> Result<ChunkPlan> {
        guard(&self.seen).push(content.to_owned());
        if content.contains(FAIL_ONCE) && guard(&self.failed_once).insert(content.to_owned()) {
            return Err(CoreError::storage("database is locked (injected once)"));
        }
        if content.contains(ALWAYS_FAIL) {
            return Err(CoreError::storage("database is locked (injected)"));
        }
        if content.contains(PANICS) {
            panic!("injected chunker panic");
        }
        if content.contains(HOLD) {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut open = guard(&self.gate);
            while !*open && Instant::now() < deadline {
                open = self
                    .cv
                    .wait_timeout(open, Duration::from_millis(50))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
        SimpleChunker.chunk(content, params)
    }
}

/// A daemon with the fake embedder and `chunker`, ready.
fn start_faulty(rig: &Rig, chunker: &Arc<FaultChunker>) -> Daemon {
    let c = Arc::clone(chunker);
    let loader = FnLoader(move || {
        Ok(LoadedModel {
            provider: Arc::new(FakeProvider::new()),
            chunker: Arc::clone(&c) as Arc<dyn Chunker>,
        })
    });
    let d = rig.start(Arc::new(loader), Arc::new(UnwiredSharing));
    d.wait_ready();
    d
}

fn log_of(d: &Daemon) -> String {
    std::fs::read_to_string(d.log_path()).unwrap_or_default()
}

/// A capture request for `cwd` as the hook spools it. It has no session,
/// so no session's hashes hide a second processing of it.
fn spooled_capture(rig: &Rig, cwd: &str, text: &str) -> Request {
    Request::new(
        cwd,
        rig.profile(),
        Op::Capture(CaptureParams {
            event: HookEvent::PostToolUse,
            payload: edit(text),
            received_at: Timestamp::now(),
        }),
    )
}

/// SPEC 16 (never loses; e-report decision 27): a capture whose store fails
/// once is tried again and stored once, and its session moves on only once
/// it is stored, so the next capture of the session reads from the right
/// offset.
#[test]
fn s16_a_failed_capture_is_retried_and_its_session_moves_on_only_once_stored() {
    let mut rig = Rig::new();
    rig.opts.capture_retry_backoff = Duration::from_millis(20);
    let chunker = Arc::new(FaultChunker::default());
    let d = start_faulty(&rig, &chunker);
    let store = rig.store();
    let cwd = rig.project("retried");
    let transcript = Path::new(&cwd).join("transcript.jsonl");
    let path = transcript.to_str().expect("UTF-8").to_owned();
    let s = unique("s");
    let pre_compact = || {
        queue(
            &d,
            &cwd,
            Some(&s),
            HookEvent::PreCompact,
            json!({"session_id": s, "transcript_path": path, "trigger": "auto"}),
        )
    };
    let has = |text: &str| contents(&project_memories(&store, &cwd)).contains(&text.to_owned());

    let x = "Decided to use SQLite FAILONCE for the local cache.";
    let b = "The fix is to add a retry around the upload call.";
    std::fs::write(&transcript, entry(x, 1)).expect("write");
    let first = pre_compact();
    assert!(
        wait_until(WAIT, || has(x)),
        "the capture was stored on its second attempt"
    );
    // The capture metrics are journalled after the store commits (B-99), so
    // wait for the worker to finish this job before reading the journal.
    barrier(&d, &rig, &store);
    assert_eq!(
        chunker.calls_with(FAIL_ONCE),
        2,
        "one attempt failed, the next stored it"
    );
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &first).len(),
        1,
        "stored once"
    );
    let log = log_of(&d);
    assert!(
        log.contains(&format!("capture {first}: storage: ")),
        "the failure names the request and the cause: {log}"
    );
    assert!(log.contains("attempt 1 of 5 failed"), "{log}");

    std::fs::write(&transcript, entry(x, 1) + &entry(b, 2)).expect("append");
    let second = pre_compact();
    barrier(&d, &rig, &store);
    assert!(has(b), "the appended entry is stored");
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &second).len(),
        1,
        "the second read starts after X: the offset moved once X was stored"
    );
    assert_eq!(chunker.calls_with(FAIL_ONCE), 2, "X is not read again");
    assert_eq!(project_memories(&store, &cwd).len(), 2);
    assert!(files_in(&rig.paths.spool_new()).is_empty(), "nothing left");
    assert!(
        files_in(&rig.paths.spool_failed()).is_empty(),
        "nothing lost"
    );
}

/// E-report decision 27: a capture that keeps failing, or panics, is tried
/// `MAX_ATTEMPTS` times and then moves to `spool/failed/` with a log line
/// naming the request and the cause; the worker keeps running.
#[test]
fn s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound() {
    let mut rig = Rig::new();
    rig.opts.capture_retry_backoff = Duration::from_millis(5);
    let chunker = Arc::new(FaultChunker::default());
    let d = start_faulty(&rig, &chunker);
    let store = rig.store();
    let cwd = rig.project("bound");
    let failing = queue(
        &d,
        &cwd,
        None,
        HookEvent::PostToolUse,
        edit("Decided to use ALWAYSFAIL for the local cache."),
    );
    let panicking = queue(
        &d,
        &cwd,
        None,
        HookEvent::PostToolUse,
        edit("Decided to use PANICS for the message queue."),
    );
    let failed = rig.paths.spool_failed();
    assert!(
        wait_until(WAIT, || files_in(&failed).len() == 2),
        "both moved to spool/failed: {:?}",
        files_in(&failed)
    );
    assert_eq!(chunker.calls_with(ALWAYS_FAIL), MAX_ATTEMPTS);
    assert_eq!(chunker.calls_with(PANICS), MAX_ATTEMPTS);
    let log = log_of(&d);
    assert!(
        log.contains(&format!("capture {failing}: storage: "))
            && log.contains("failed 5 times, moved to spool/failed"),
        "{log}"
    );
    assert!(
        log.contains(&format!(
            "capture {panicking}: internal: processing panicked: injected chunker panic"
        )),
        "{log}"
    );
    let ids: HashSet<String> = files_in(&failed)
        .iter()
        .map(|n| read_spool(&failed.join(n)).expect("a capture request").id)
        .collect();
    assert_eq!(ids, HashSet::from([failing, panicking]));
    assert!(project_memories(&store, &cwd).is_empty(), "nothing stored");
    assert!(files_in(&rig.paths.spool_new()).is_empty());
    // The worker survived the panics.
    barrier(&d, &rig, &store);
}

/// B-33, e-report decision 27: a payload that cannot be parsed, and a
/// capture whose working folder is gone, move to `spool/failed/` at once.
#[test]
fn s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once() {
    let mut rig = Rig::new();
    rig.opts.spool_interval = Duration::from_millis(100);
    let (tmp, new, failed) = (
        rig.paths.spool_tmp(),
        rig.paths.spool_new(),
        rig.paths.spool_failed(),
    );
    let gone = rig.tmp.path().join("work").join("never-created");
    let gone = gone.to_str().expect("UTF-8").to_owned();
    let orphan_req = spooled_capture(
        &rig,
        &gone,
        "Decided to use FAILONCE in a folder that is gone.",
    );
    let orphan = write_spool(&tmp, &new, &orphan_req).expect("spool");
    let corrupt = new.join("corrupt.json");
    std::fs::write(&corrupt, "{\"v\": 1, \"call\":").expect("write a corrupt file");

    let chunker = Arc::new(FaultChunker::default());
    let d = start_faulty(&rig, &chunker);
    let name = |p: &Path| {
        p.file_name()
            .expect("a name")
            .to_string_lossy()
            .into_owned()
    };
    let mut expected = vec![name(&orphan), name(&corrupt)];
    expected.sort();
    assert!(
        wait_until(WAIT, || files_in(&failed) == expected),
        "both moved to spool/failed: {:?}",
        files_in(&failed)
    );
    assert!(files_in(&new).is_empty());
    assert_eq!(
        chunker.calls_with(FAIL_ONCE),
        0,
        "the capture failed before chunking, once"
    );
    let log = log_of(&d);
    assert!(
        log.contains(&format!("capture {}: project_unresolved: ", orphan_req.id))
            && log.contains("it can never be stored, moved to spool/failed"),
        "{log}"
    );
    assert!(
        !log.contains("attempt 1 of 5 failed"),
        "not tried again: {log}"
    );
}

/// AT-15, e-report decision 28: a drained file stays in `spool/new/` until
/// its capture is stored, and the drain does not queue it a second time
/// meanwhile.
#[test]
fn at15_a_drained_file_stays_in_new_until_its_capture_is_stored() {
    let mut rig = Rig::new();
    let interval = Duration::from_millis(100);
    rig.opts.spool_interval = interval;
    let cwd = rig.project("held-file");
    let text = "Decided to use HOLDHERE for the drained capture.";
    let req = spooled_capture(&rig, &cwd, text);
    let file = write_spool(&rig.paths.spool_tmp(), &rig.paths.spool_new(), &req).expect("spool");

    let chunker = Arc::new(FaultChunker::default());
    let d = start_faulty(&rig, &chunker);
    let store = rig.store();
    assert!(
        wait_until(WAIT, || chunker.entered() == 1),
        "the worker holds the capture"
    );
    std::thread::sleep(interval * 6);
    assert!(
        file.exists(),
        "the file stays in spool/new while its capture is processed"
    );
    assert!(project_memories(&store, &cwd).is_empty());

    chunker.open();
    assert!(
        wait_until(WAIT, || contents(&project_memories(&store, &cwd))
            == [text.to_owned()]
            && !file.exists()),
        "stored, then its file removed"
    );
    std::thread::sleep(interval * 3);
    barrier(&d, &rig, &store);
    assert_eq!(
        chunker.calls_with(HOLD),
        1,
        "the drain never queued the held file a second time"
    );
    assert_eq!(
        journal_for(&rig, metrics::CAPTURE_SIZE_TOKENS, &req.id).len(),
        1,
        "processed once"
    );
    assert!(files_in(&rig.paths.spool_new()).is_empty());
    assert!(files_in(&rig.paths.spool_failed()).is_empty());
}

/// SPEC 16 (never loses), CONTRACTS 3.1, e-report decisions 28 and 29: a
/// capture being processed when the daemon stops is a file in `spool/new/`
/// (a socket capture is spooled before it is processed); the next daemon
/// stores it. When the first worker finishes later, the store's duplicate
/// rule (B-29) returns the memory already stored, so the capture is stored
/// once.
#[test]
fn at15_a_capture_held_at_shutdown_is_stored_after_a_restart() {
    let mut rig = Rig::new();
    rig.opts.shutdown_grace = Duration::from_millis(300);
    let chunker = Arc::new(FaultChunker::default());
    let d = start_faulty(&rig, &chunker);
    let cwd = rig.project("held-at-shutdown");
    let text = "Decided to use HOLDHERE for the capture held at shutdown.";
    let id = queue(
        &d,
        &cwd,
        Some(&unique("s")),
        HookEvent::PostToolUse,
        edit(text),
    );
    assert!(
        wait_until(WAIT, || chunker.entered() == 1),
        "the worker holds the capture"
    );
    let new = rig.paths.spool_new();
    let held = files_in(&new);
    assert_eq!(
        held.len(),
        1,
        "a socket capture is written to spool/new before it is processed"
    );
    assert_eq!(read_spool(&new.join(&held[0])).expect("a capture").id, id);

    let exit = d.shutdown();
    assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
    assert_eq!(
        files_in(&new),
        held,
        "the capture is still on disk after the daemon stopped"
    );
    assert!(project_memories(&rig.store(), &cwd).is_empty());

    let _d2 = rig.start_fake();
    let store = rig.store();
    assert!(
        wait_until(WAIT, || contents(&project_memories(&store, &cwd))
            == [text.to_owned()]
            && files_in(&new).is_empty()),
        "the next daemon stored the capture and removed its file"
    );

    // The first daemon's worker now finishes the same capture.
    chunker.open();
    assert!(
        wait_until(WAIT, || journal_for(
            &rig,
            metrics::CAPTURE_SIZE_TOKENS,
            &id
        )
        .len()
            == 2),
        "both workers stored the capture"
    );
    assert_eq!(
        project_memories(&store, &cwd).len(),
        1,
        "the duplicate rule returned the memory already stored"
    );
    assert!(files_in(&rig.paths.spool_failed()).is_empty());
}

// --- capture: the project is read afresh (e-report decision 30) ----------------

/// SPEC 9, SPEC 16: `capture.enabled = false` added to a project's
/// `project.toml` counts from the next capture, with no restart.
#[test]
fn s16_capture_disabled_in_project_toml_counts_without_a_restart() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("opted-out");
    let first = "Decided to use Redis for the opt-out test.";
    queue(&d, &cwd, None, HookEvent::PostToolUse, edit(first));
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        [first.to_owned()]
    );

    let dir = Path::new(&cwd).join(".spatial-memory");
    std::fs::create_dir_all(&dir).expect("create .spatial-memory");
    std::fs::write(dir.join("project.toml"), "[capture]\nenabled = false\n")
        .expect("write project.toml");
    let second = "Decided to use Kafka for the opt-out test.";
    queue(&d, &cwd, None, HookEvent::PostToolUse, edit(second));
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        [first.to_owned()],
        "the project's opt-out is read on the next capture"
    );
}

// --- capture: refusals ----------------------------------------------------------

/// SPEC 16: a non-capture event is refused, and with capture disabled
/// nothing is queued.
#[test]
fn s16_capture_refusals() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("refused");
    for event in [HookEvent::SessionStart, HookEvent::UserPromptSubmit] {
        let req = capture_req(&d, &cwd, None, event, json!({"prompt": "x"}));
        assert!(!send_capture(&d, &req), "{event:?}");
    }
    drop(d);

    let mut rig = Rig::new();
    rig.settings.capture.enabled = false;
    let d = rig.start_fake();
    let cwd = rig.project("disabled");
    let req = capture_req(&d, &cwd, None, HookEvent::PostToolUse, edit(MISSED));
    assert!(!send_capture(&d, &req), "capture disabled");
}

// --- ambient recall -----------------------------------------------------------

/// The fake provider, with a query session that waits `delay` and then
/// fails when `fail` is set. Bulk sessions are the plain fake.
struct QueryProvider {
    info: EmbedderInfo,
    delay: Duration,
    fail: bool,
}

impl QueryProvider {
    fn new(delay: Duration, fail: bool) -> Arc<QueryProvider> {
        Arc::new(QueryProvider {
            info: fake_info(),
            delay,
            fail,
        })
    }
}

struct QueryEmbedder {
    inner: FakeEmbedder,
    kind: SessionKind,
    delay: Duration,
    fail: bool,
}

impl EmbedderProvider for QueryProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }
    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        Ok(Box::new(QueryEmbedder {
            inner: FakeEmbedder::with_session(kind, threads),
            kind,
            delay: self.delay,
            fail: self.fail,
        }))
    }
}

impl Embedder for QueryEmbedder {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }
    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        if self.kind == SessionKind::Query {
            std::thread::sleep(self.delay);
            if self.fail {
                return Err(CoreError::internal("search failed"));
            }
        }
        self.inner.embed_query(text, max_tokens)
    }
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(docs)
    }
}

fn ambient_req(d: &Daemon, cwd: &str, prompt: &str, top_k: u32, scope: Scope) -> Request {
    d.req(
        cwd,
        Op::AmbientRecall(AmbientParams {
            prompt: prompt.to_owned(),
            top_k,
            scope,
        }),
    )
}

fn ambient_reply(d: &Daemon, req: &Request) -> AmbientReply {
    match d.send(req).into_result() {
        Ok(Reply::Ambient(r)) => r,
        Ok(other) => panic!("ambient_recall replied {other:?}"),
        Err(e) => panic!("ambient_recall failed: {}: {}", e.code.as_str(), e.message),
    }
}

fn ambient_in(d: &Daemon, cwd: &str, session: Option<&str>, prompt: &str) -> AmbientReply {
    let req =
        ambient_req(d, cwd, prompt, 3, Scope::Project).with_session(session.map(str::to_owned));
    ambient_reply(d, &req)
}

const PROMPT: &str = "release checklist steps for the billing service deploy";

/// Seeds four memories close to [`PROMPT`] and one far from it; returns
/// (content, id, fake score), best first, and the far one.
fn seed(d: &Daemon, cwd: &str) -> (Vec<(String, sm_core::ids::MemoryId, f32)>, String) {
    let q = fake_vector(PROMPT);
    let score = |t: &str| dot(&q, &fake_vector(t));
    let near = [
        PROMPT.to_owned(),
        format!("{PROMPT} today"),
        format!("{PROMPT} today quickly"),
        format!("{PROMPT} today quickly please"),
    ];
    let far = "unrelated note about garden tomatoes and summer rain".to_owned();
    let mut seeded: Vec<(String, sm_core::ids::MemoryId, f32)> = near
        .iter()
        .map(|t| (t.clone(), remember(d, cwd, t), score(t)))
        .collect();
    remember(d, cwd, &far);
    d.wait_embedded(cwd);
    seeded.sort_by(|a, b| b.2.total_cmp(&a.2));
    for w in seeded.windows(2) {
        assert!(w[0].2 > w[1].2 + 1e-3, "distinct fake scores: {seeded:?}");
    }
    for s in &seeded {
        assert!(s.2 >= 0.75, "above the 0.70 floor: {s:?}");
    }
    assert!(score(&far) < 0.5, "the far memory is under the floor");
    (seeded, far)
}

/// The block `ambient::format_block` builds for these memories with the
/// scores the daemon reported.
fn block_for(memories: &[(String, sm_core::ids::MemoryId, f32)], scores: &[f32]) -> String {
    let cands: Vec<ambient::Candidate> = memories
        .iter()
        .zip(scores)
        .map(|(m, s)| ambient::Candidate {
            id: m.1,
            score: *s,
            text: m.0.clone(),
        })
        .collect();
    let refs: Vec<&ambient::Candidate> = cands.iter().collect();
    ambient::format_block(&refs).expect("a block").text
}

/// AT-17 and SPEC 6: the block is exactly `AmbientReply.text`, built by
/// `ambient::format_block` from the best memories over the floor, at most
/// `top_k` of them; no memory repeats within a session, and memories
/// already shown are replaced by the next best; another session, and the
/// Cursor client, get the same block (`recall_returns_formatted_block`,
/// `probe.client_cursor_same_output`).
#[test]
fn at17_ambient_block_is_the_reply_text_and_never_repeats() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("ambient");
    let (seeded, _) = seed(&d, &cwd);

    let s = unique("session");
    let first = ambient_in(&d, &cwd, Some(&s), PROMPT);
    assert_eq!(first.gate, HookGate::Injected);
    let ids: Vec<_> = seeded.iter().map(|m| m.1).collect();
    assert_eq!(first.injected, ids[..3]);
    assert_eq!(first.top_scores.len(), 3);
    for (got, want) in first.top_scores.iter().zip(&seeded) {
        assert!((got - want.2).abs() < 1e-4, "score {got} vs {}", want.2);
    }
    let text = first.text.clone().expect("a block");
    assert_eq!(text, block_for(&seeded[..3], &first.top_scores));
    assert!(text.starts_with(ambient::HEADER) && text.ends_with(ambient::FOOTER));
    assert_eq!(first.project.as_deref(), Some("ambient"));
    assert!(wait_until(WAIT, || store
        .injected(&s)
        .expect("injected")
        .len()
        == 3));

    let second = ambient_in(&d, &cwd, Some(&s), PROMPT);
    assert_eq!(second.gate, HookGate::Injected);
    assert_eq!(second.injected, ids[3..]);
    // `top_scores` are the best scores found, shown or not; the block holds
    // only the memory not yet shown.
    for (got, want) in second.top_scores.iter().zip(&seeded) {
        assert!((got - want.2).abs() < 1e-4, "score {got} vs {}", want.2);
    }
    let text2 = second.text.clone().expect("a block");
    assert!(text2.starts_with(ambient::HEADER) && text2.ends_with(ambient::FOOTER));
    assert!(text2.contains(&format!("[{}]", ids[3])) && text2.contains(&seeded[3].0));
    for id in &ids[..3] {
        assert!(
            !text2.contains(&format!("[{id}]")),
            "{id} repeated: {text2}"
        );
    }
    assert!(wait_until(WAIT, || store
        .injected(&s)
        .expect("injected")
        .len()
        == 4));

    let third = ambient_in(&d, &cwd, Some(&s), PROMPT);
    assert_eq!(third.gate, HookGate::Repeat);
    assert!(third.text.is_none() && third.injected.is_empty());

    let other = ambient_in(&d, &cwd, Some(&unique("session")), PROMPT);
    assert_eq!(other.injected, ids[..3]);
    assert_eq!(other.text.as_deref(), Some(text.as_str()));

    let req = ambient_req(&d, &cwd, PROMPT, 3, Scope::Project).with_client(HookClient::Cursor);
    assert_eq!(ambient_reply(&d, &req).text.as_deref(), Some(text.as_str()));

    // top_k caps the block.
    let one = ambient_reply(&d, &ambient_req(&d, &cwd, PROMPT, 1, Scope::Project));
    assert_eq!(one.injected, ids[..1]);
    assert_eq!(one.top_scores.len(), 1);

    // `scope: all` needs no project.
    let all = ambient_reply(&d, &ambient_req(&d, "", PROMPT, 3, Scope::All));
    assert_eq!(all.gate, HookGate::Injected);
    assert_eq!(all.injected, ids[..3]);
}

/// SPEC 6 gates: `empty` with no memories, `floor` when none is close
/// enough, `no_project` when the folder resolves to no project
/// (`probe.payload_without_session_or_cwd` with a four-word prompt), and
/// `short_prompt` for fewer than four words, before anything else.
#[test]
fn s6_ambient_gates() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let empty = rig.project("empty");
    assert_eq!(ambient_in(&d, &empty, None, PROMPT).gate, HookGate::Empty);

    let far = rig.project("far");
    remember(
        &d,
        &far,
        "unrelated note about garden tomatoes and summer rain",
    );
    d.wait_embedded(&far);
    let r = ambient_in(&d, &far, None, PROMPT);
    assert_eq!(r.gate, HookGate::Floor);
    assert_eq!(r.top_scores.len(), 1, "the score is reported");
    assert!(r.text.is_none() && r.injected.is_empty());

    assert_eq!(
        ambient_in(&d, "", None, "How do I configure it").gate,
        HookGate::NoProject
    );
    assert_eq!(
        ambient_in(&d, "", None, "Just a prompt").gate,
        HookGate::ShortPrompt
    );
}

/// `ambient_recall_cases.json` prompt cases and probes: an empty, blank or
/// missing prompt, or one that is not a string, is a malformed request
/// (`invalid_input`); a string with fewer than four words (Python's
/// whitespace) is `short_prompt`. Version 2 printed a block for
/// `padded`, `control_ws` and the others, having no four-word minimum;
/// SPEC 6's minimum supersedes that.
#[test]
fn s6_ambient_prompt_cases() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("prompts");
    remember(&d, &cwd, "Remembered fact.");
    d.wait_embedded(&cwd);
    let f = fixture("capture/ambient_recall_cases.json");

    let base = ambient_req(&d, &cwd, PROMPT, 5, Scope::Project);
    let with_prompt = |prompt: Option<&Value>| {
        let mut v = serde_json::to_value(&base).expect("encode");
        let args = v["call"]["args"].as_object_mut().expect("args");
        match prompt {
            Some(p) => {
                args.insert("prompt".to_owned(), p.clone());
            }
            None => {
                args.remove("prompt");
            }
        }
        v
    };

    let mut seen = 0;
    for c in f["cases"].as_array().expect("cases") {
        let name = c["name"].as_str().expect("a name");
        let payload = &c["args"]["payload"];
        match name {
            "empty_prompt_returns_none" | "blank_prompt_returns_none" => {
                let p = payload["prompt"].as_str().expect("a prompt");
                let e = err_of(d.send(&ambient_req(&d, &cwd, p, 5, Scope::Project)));
                assert_code(&e, ErrorCode::InvalidInput);
                seen += 1;
            }
            "missing_prompt_field_returns_none" => {
                assert!(payload.get("prompt").is_none());
                let e = err_of(d.client().raw_value(&with_prompt(None)));
                assert_code(&e, ErrorCode::InvalidInput);
                seen += 1;
            }
            _ => {}
        }
    }
    assert_eq!(seen, 3);

    let probes = f["probes"]["prompt"].as_array().expect("prompt probes");
    for p in probes {
        let name = p["name"].as_str().expect("a name");
        let prompt = &p["args"]["payload"]["prompt"];
        match prompt.as_str() {
            Some(text) => {
                assert!(ambient::prompt_word_count(text) < ambient::MIN_PROMPT_WORDS);
                let r = ambient_in(&d, &cwd, None, text);
                assert_eq!(r.gate, HookGate::ShortPrompt, "{name}");
            }
            None => {
                let e = err_of(d.client().raw_value(&with_prompt(Some(prompt))));
                assert_code(&e, ErrorCode::InvalidInput);
            }
        }
    }
    assert_eq!(probes.len(), 9);
}

/// `recall_raises_returns_none` and `probe.hybrid_recall_raises`: a search
/// that fails gives `error` and records nothing;
/// `probe.build_factory_raises`: a model that failed to load gives
/// `error`; a model still loading gives `loading`.
#[test]
fn s6_ambient_errors_and_loading() {
    let rig = Rig::new();
    let d = rig.start(
        FakeLoader::with(QueryProvider::new(Duration::ZERO, true)),
        Arc::new(UnwiredSharing),
    );
    d.wait_ready();
    let cwd = rig.project("fails");
    remember(&d, &cwd, PROMPT);
    d.wait_embedded(&cwd);
    let s = unique("session");
    let r = ambient_in(&d, &cwd, Some(&s), PROMPT);
    assert_eq!(r.gate, HookGate::Error);
    assert!(r.text.is_none());
    assert!(rig.store().injected(&s).expect("injected").is_empty());
    drop(d);

    let rig = Rig::new();
    let d = rig.start(
        Arc::new(FnLoader(|| -> Result<sm_daemon::LoadedModel> {
            Err(CoreError::internal("bad settings"))
        })),
        Arc::new(UnwiredSharing),
    );
    assert!(wait_until(WAIT, || {
        d.hello().state == sm_core::protocol::DaemonState::Failed
    }));
    let cwd = rig.project("failed");
    assert_eq!(ambient_in(&d, &cwd, None, PROMPT).gate, HookGate::Error);
    drop(d);

    let rig = Rig::new();
    let gate = GateLoader::new();
    let d = rig.start(gate.clone(), Arc::new(UnwiredSharing));
    let cwd = rig.project("loading");
    assert_eq!(ambient_in(&d, &cwd, None, PROMPT).gate, HookGate::Loading);
    gate.open();
    d.wait_ready();
}

/// SPEC 6: a reply built after the request's deadline is `timeout`, keeps
/// the scores it has, and records no injection; the same call without a
/// deadline injects and records.
#[test]
fn s6_ambient_deadline_records_nothing() {
    let rig = Rig::new();
    let d = rig.start(
        FakeLoader::with(QueryProvider::new(Duration::from_millis(100), false)),
        Arc::new(UnwiredSharing),
    );
    d.wait_ready();
    let store = rig.store();
    let cwd = rig.project("deadline");
    remember(&d, &cwd, PROMPT);
    d.wait_embedded(&cwd);

    let s = unique("session");
    let req = ambient_req(&d, &cwd, PROMPT, 3, Scope::Project)
        .with_session(Some(s.clone()))
        .with_deadline_ms(20);
    let r = ambient_reply(&d, &req);
    assert_eq!(r.gate, HookGate::Timeout);
    assert!(r.text.is_none() && r.injected.is_empty());
    std::thread::sleep(Duration::from_millis(300));
    assert!(store.injected(&s).expect("injected").is_empty());

    let r = ambient_in(&d, &cwd, Some(&s), PROMPT);
    assert_eq!(r.gate, HookGate::Injected);
    assert!(wait_until(WAIT, || store.injected(&s).expect("injected") == r.injected));
}

/// `integration_test_data` of `ambient_recall_cases.json`: the five seeds
/// and the prompt give a block that holds the header and the start of at
/// least one seed. The similarity floor is model-specific (SPEC 15,
/// provisional): the fake embedder's bag-of-words cosine for this prompt
/// is about 0.43, so this test alone lowers the floor to 0.30. The default
/// is not changed.
#[test]
fn s6_ambient_fixture_integration_data() {
    let mut rig = Rig::new();
    rig.settings.recall.similarity_floor = 0.30;
    let d = rig.start_fake();
    let cwd = rig.project("seeded");
    let data = &fixture("capture/ambient_recall_cases.json")["integration_test_data"];
    let seeds: Vec<String> = data["seeds"]
        .as_array()
        .expect("seeds")
        .iter()
        .map(|s| s["content"].as_str().expect("content").to_owned())
        .collect();
    assert_eq!(seeds.len(), 5);
    for s in &seeds {
        remember(&d, &cwd, s);
    }
    d.wait_embedded(&cwd);
    let prompt = data["args"]["payload"]["prompt"]
        .as_str()
        .expect("a prompt");
    let top_k = data["env"]["SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K"]
        .as_str()
        .and_then(|k| k.parse().ok())
        .expect("top_k");
    let r = ambient_reply(&d, &ambient_req(&d, &cwd, prompt, top_k, Scope::Project));
    assert_eq!(r.gate, HookGate::Injected);
    let text = r.text.expect("a block");
    assert!(text.contains(ambient::HEADER));
    assert!(
        seeds
            .iter()
            .any(|s| text.contains(&s.chars().take(30).collect::<String>())),
        "{text}"
    );
}
