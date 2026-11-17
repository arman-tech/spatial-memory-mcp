//! SPEC section 16 rules that no fixture file covers on its own:
//! `capture.enabled` off, the project opt-out, a tool not in
//! `capture.tools`, each default ignore pattern of B-21 against Edit and
//! Write paths, the gate order, this server's own tools, anchors for Edit
//! and Write, provenance with capture as the source, types from pattern
//! families, and what [`Candidate::to_new_memory`] leaves for the store.

use std::collections::BTreeSet;

use serde_json::{Value, json};
use sm_core::config::{CaptureSettings, DEFAULT_CAPTURE_IGNORE, DEFAULT_CAPTURE_TOOLS};
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::{HookEvent, Tool};
use sm_core::types::{AnchorKind, AnchorRef, MemoryType, Origin, ProvenanceSource, Status};

use super::{project, settings_with_tools, temp_dir};
use crate::candidate::Candidate;
use crate::extract::{ToolSkip, V1_TOOL_NAMES};
use crate::pipeline::{CaptureInput, CaptureOutcome, Skip, TextSource, capture};
use crate::session::SessionState;
use crate::signal::{SIGNAL_PATTERNS, Tier};

/// A strong decision: every gate test uses it, so only the gate can stop it.
const STRONG: &str = "Decided to use Redis for the session cache.";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const SESSION: &str = "rules-session";

/// A project in a temporary directory, with a transcript holding one
/// assistant text.
struct Fixture {
    _dir: tempfile::TempDir,
    project: ProjectIdentity,
    transcript: String,
}

impl Fixture {
    fn new() -> Self {
        let dir = temp_dir();
        let transcript = dir.path().join("session.jsonl");
        let line = json!({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": STRONG}]},
            "uuid": "u1",
        });
        std::fs::write(&transcript, format!("{line}\n")).expect("write a transcript");
        Fixture {
            project: project(dir.path()),
            transcript: transcript.to_str().expect("a UTF-8 path").to_owned(),
            _dir: dir,
        }
    }

    fn run(&self, event: HookEvent, payload: &Value, settings: &CaptureSettings) -> CaptureOutcome {
        run_in(&self.project, event, payload, settings)
    }

    /// One payload for each capture event, each carrying [`STRONG`].
    fn capture_payloads(&self) -> [(HookEvent, Value); 4] {
        [
            (HookEvent::PostToolUse, edit("src/cache.rs", STRONG)),
            (
                HookEvent::PreCompact,
                json!({"session_id": SESSION, "transcript_path": self.transcript, "trigger": "auto"}),
            ),
            (
                HookEvent::Stop,
                json!({"session_id": SESSION, "transcript_path": self.transcript}),
            ),
            (
                HookEvent::Stop,
                json!({"session_id": SESSION, "last_assistant_message": STRONG}),
            ),
        ]
    }
}

fn run_in(
    project: &ProjectIdentity,
    event: HookEvent,
    payload: &Value,
    settings: &CaptureSettings,
) -> CaptureOutcome {
    let mut state = SessionState::new();
    capture(
        &CaptureInput {
            event,
            payload,
            settings,
            project,
            commit_sha: Some(COMMIT),
            transcript_offset: 0,
        },
        &mut state,
    )
}

fn edit(path: &str, text: &str) -> Value {
    json!({
        "session_id": SESSION,
        "tool_name": "Edit",
        "tool_input": {"file_path": path, "old_string": "x", "new_string": text},
        "tool_response": "ok",
    })
}

fn write(path: &str, text: &str) -> Value {
    json!({
        "session_id": SESSION,
        "tool_name": "Write",
        "tool_input": {"file_path": path, "content": text},
        "tool_response": "ok",
    })
}

/// A PostToolUse payload for any tool, the text in its response.
fn tool(name: &str, tool_input: Value, response: &str) -> Value {
    json!({
        "session_id": SESSION,
        "tool_name": name,
        "tool_input": tool_input,
        "tool_response": response,
    })
}

/// The only candidate of an outcome.
fn only(outcome: &CaptureOutcome, what: &str) -> Candidate {
    assert_eq!(outcome.skipped, None, "{what}: skipped");
    assert_eq!(outcome.candidates.len(), 1, "{what}: {outcome:?}");
    outcome.candidates[0].clone()
}

#[test]
fn s16_capture_disabled_gives_nothing() {
    let f = Fixture::new();
    let settings = CaptureSettings {
        enabled: false,
        ..CaptureSettings::default()
    };
    for (event, payload) in f.capture_payloads() {
        let outcome = f.run(event, &payload, &settings);
        assert_eq!(
            outcome,
            CaptureOutcome {
                skipped: Some(Skip::CaptureDisabled),
                ..CaptureOutcome::default()
            },
            "{event:?}"
        );
        // The same payload gives a candidate when capture is on.
        only(
            &f.run(event, &payload, &CaptureSettings::default()),
            "enabled",
        );
    }
}

#[test]
fn s16_project_opt_out_gives_nothing() {
    let mut f = Fixture::new();
    f.project.capture_enabled = false;
    for (event, payload) in f.capture_payloads() {
        let outcome = f.run(event, &payload, &CaptureSettings::default());
        assert_eq!(
            outcome,
            CaptureOutcome {
                skipped: Some(Skip::ProjectOptedOut),
                ..CaptureOutcome::default()
            },
            "{event:?}"
        );
    }
    // `capture.enabled` off is checked first.
    let off = CaptureSettings {
        enabled: false,
        ..CaptureSettings::default()
    };
    let outcome = f.run(HookEvent::PostToolUse, &edit("a.rs", STRONG), &off);
    assert_eq!(outcome.skipped, Some(Skip::CaptureDisabled));
}

#[test]
fn s16_a_tool_not_in_capture_tools_gives_nothing() {
    let f = Fixture::new();
    // B-21: the default list is Edit, Write, Bash and NotebookEdit.
    assert_eq!(
        CaptureSettings::default().tools,
        DEFAULT_CAPTURE_TOOLS.map(str::to_owned).to_vec()
    );
    let defaults = CaptureSettings::default();
    for (name, payload) in [
        ("Read", tool("Read", json!({"file_path": "a.rs"}), STRONG)),
        ("mcp__db__query", tool("mcp__db__query", json!({}), STRONG)),
        (
            "edit",
            tool(
                "edit",
                json!({"file_path": "a.rs", "new_string": STRONG}),
                "ok",
            ),
        ),
        ("Task", tool("Task", json!({"prompt": "x"}), STRONG)),
    ] {
        let outcome = f.run(HookEvent::PostToolUse, &payload, &defaults);
        assert_eq!(
            outcome.skipped,
            Some(Skip::ToolNotCaptured(name.to_owned())),
            "{name}"
        );
        assert!(outcome.candidates.is_empty(), "{name}");
    }
    // Bash is captured by default; with a list that leaves it out it is not.
    let bash = tool("Bash", json!({"command": "make"}), STRONG);
    only(
        &f.run(HookEvent::PostToolUse, &bash, &defaults),
        "Bash by default",
    );
    let outcome = f.run(
        HookEvent::PostToolUse,
        &bash,
        &settings_with_tools(&["Edit"]),
    );
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolNotCaptured("Bash".to_owned()))
    );
    // An empty list captures no tool; transcript events do not use the list.
    let none = settings_with_tools(&[]);
    let outcome = f.run(HookEvent::PostToolUse, &edit("a.rs", STRONG), &none);
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolNotCaptured("Edit".to_owned()))
    );
    let pre_compact = json!({"session_id": SESSION, "transcript_path": f.transcript});
    only(
        &f.run(HookEvent::PreCompact, &pre_compact, &none),
        "PreCompact with no tools",
    );
}

/// Each B-21 default ignore pattern and a path it matches.
const IGNORED: [(&str, &str); 13] = [
    (".env", ".env"),
    (".env.*", "config/.env.local"),
    ("*.pem", "certs/server.pem"),
    ("*.key", "tls/server.key"),
    ("*.p12", "certs/client.p12"),
    ("*.pfx", "certs/client.pfx"),
    ("*.keystore", "android/release.keystore"),
    ("*.jks", "config/app.jks"),
    ("id_rsa*", "keys/id_rsa"),
    ("id_dsa*", "keys/id_dsa.pub"),
    ("id_ecdsa*", "id_ecdsa"),
    ("id_ed25519*", ".ssh/id_ed25519"),
    (".netrc", "home/.netrc"),
];

#[test]
fn s16_each_default_ignore_pattern_skips_edit_and_write() {
    let patterns: BTreeSet<&str> = IGNORED.iter().map(|(p, _)| *p).collect();
    let defaults: BTreeSet<&str> = DEFAULT_CAPTURE_IGNORE.iter().copied().collect();
    assert_eq!(patterns, defaults, "the table covers B-21's defaults");

    let f = Fixture::new();
    let settings = CaptureSettings::default();
    let root = &f.project.root;
    let mut failures = Vec::new();
    for (pattern, rel) in IGNORED {
        let native = root.join(rel).to_str().expect("a UTF-8 path").to_owned();
        let forms = [
            rel.to_owned(),
            format!("./{rel}"),
            native,
            format!("{}/{rel}", f.project.root_path),
        ];
        for path in &forms {
            for payload in [edit(path, STRONG), write(path, STRONG)] {
                let outcome = f.run(HookEvent::PostToolUse, &payload, &settings);
                if outcome.skipped != Some(Skip::PathIgnored(rel.to_owned()))
                    || !outcome.candidates.is_empty()
                {
                    failures.push(format!(
                        "{pattern} {} {path}: {:?}",
                        payload["tool_name"], outcome.skipped
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "not ignored:\n{}", failures.join("\n"));

    // Near misses are captured.
    for rel in [
        "src/environment.rs",
        "docs/keys.md",
        "src/pem.rs",
        "netrc.md",
    ] {
        for payload in [edit(rel, STRONG), write(rel, STRONG)] {
            only(&f.run(HookEvent::PostToolUse, &payload, &settings), rel);
        }
    }
    // Bash has no path, so the patterns do not apply to it.
    let bash = tool("Bash", json!({"command": "cat .env"}), STRONG);
    only(&f.run(HookEvent::PostToolUse, &bash, &settings), "Bash");
    // NotebookEdit's path is checked too.
    let notebook = tool(
        "NotebookEdit",
        json!({"notebook_path": "secrets/.env", "new_source": STRONG}),
        "ok",
    );
    let outcome = f.run(HookEvent::PostToolUse, &notebook, &settings);
    assert_eq!(
        outcome.skipped,
        Some(Skip::PathIgnored("secrets/.env".to_owned()))
    );
}

/// On Windows a path that differs only in case is the same file.
#[cfg(windows)]
#[test]
fn s16_ignore_patterns_fold_case_windows() {
    let f = Fixture::new();
    let outcome = f.run(
        HookEvent::PostToolUse,
        &edit("CERTS/SERVER.PEM", STRONG),
        &CaptureSettings::default(),
    );
    assert_eq!(
        outcome.skipped,
        Some(Skip::PathIgnored("CERTS/SERVER.PEM".to_owned()))
    );
    let outcome = f.run(
        HookEvent::PostToolUse,
        &write(".ENV", STRONG),
        &CaptureSettings::default(),
    );
    assert_eq!(outcome.skipped, Some(Skip::PathIgnored(".ENV".to_owned())));
}

/// On Unix case matters: `SERVER.PEM` is not `*.pem`.
#[cfg(unix)]
#[test]
fn s16_ignore_patterns_fold_case_unix() {
    let f = Fixture::new();
    let outcome = f.run(
        HookEvent::PostToolUse,
        &edit("CERTS/SERVER.PEM", STRONG),
        &CaptureSettings::default(),
    );
    only(&outcome, "SERVER.PEM");
    let outcome = f.run(
        HookEvent::PostToolUse,
        &write(".ENV", STRONG),
        &CaptureSettings::default(),
    );
    only(&outcome, ".ENV");
}

/// `capture.tools` is checked first, then version 1's skip rules, then
/// `capture.ignore`.
#[test]
fn s16_gate_order() {
    let f = Fixture::new();
    let read = tool("Read", json!({"file_path": ".env"}), STRONG);
    let outcome = f.run(HookEvent::PostToolUse, &read, &CaptureSettings::default());
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolNotCaptured("Read".to_owned()))
    );
    let outcome = f.run(
        HookEvent::PostToolUse,
        &read,
        &settings_with_tools(&["Read"]),
    );
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolSkipped(ToolSkip::SkipTool("Read".to_owned())))
    );

    let own = "mcp__spatial-memory__remember";
    let payload = tool(own, json!({"file_path": ".env", "content": STRONG}), STRONG);
    let outcome = f.run(
        HookEvent::PostToolUse,
        &payload,
        &CaptureSettings::default(),
    );
    assert_eq!(outcome.skipped, Some(Skip::ToolNotCaptured(own.to_owned())));
    let outcome = f.run(
        HookEvent::PostToolUse,
        &payload,
        &settings_with_tools(&[own]),
    );
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolSkipped(ToolSkip::SpatialMemoryTool))
    );

    let outcome = f.run(
        HookEvent::PostToolUse,
        &edit(".env", STRONG),
        &settings_with_tools(&["Write"]),
    );
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolNotCaptured("Edit".to_owned()))
    );
    let outcome = f.run(
        HookEvent::PostToolUse,
        &edit(".env", STRONG),
        &settings_with_tools(&["Edit"]),
    );
    assert_eq!(outcome.skipped, Some(Skip::PathIgnored(".env".to_owned())));

    let unnamed = json!({"session_id": SESSION, "tool_input": {"new_string": STRONG}});
    let outcome = f.run(
        HookEvent::PostToolUse,
        &unnamed,
        &CaptureSettings::default(),
    );
    assert_eq!(outcome.skipped, Some(Skip::NoToolName));

    for event in [HookEvent::SessionStart, HookEvent::UserPromptSubmit] {
        let outcome = f.run(event, &edit("a.rs", STRONG), &CaptureSettings::default());
        assert_eq!(outcome.skipped, Some(Skip::NotACaptureEvent(event)));
    }
}

/// This server's own tools are never captured, under version 3's 13 names
/// and version 1's 23, in Claude Code's form, its plugin form and Cursor's
/// `MCP:` form, even when `capture.tools` lists them.
#[test]
fn s16_own_tools_are_never_captured() {
    let f = Fixture::new();
    let mut names: Vec<String> = Vec::new();
    for t in Tool::ALL {
        let t = t.as_str();
        names.push(format!("mcp__spatial-memory__{t}"));
        names.push(format!("mcp__plugin_spatial-memory_spatial-memory__{t}"));
        names.push(format!("MCP:{t}"));
    }
    for t in V1_TOOL_NAMES {
        names.push(format!("mcp__spatial-memory__{t}"));
        names.push(format!("MCP:{t}"));
    }
    assert_eq!(Tool::ALL.len(), 13);
    let mut failures = Vec::new();
    for name in &names {
        let payload = tool(name, json!({"content": STRONG}), STRONG);
        let outcome = f.run(
            HookEvent::PostToolUse,
            &payload,
            &settings_with_tools(&[name]),
        );
        if outcome.skipped != Some(Skip::ToolSkipped(ToolSkip::SpatialMemoryTool)) {
            failures.push(format!("{name}: {:?}", outcome.skipped));
        }
    }
    assert!(failures.is_empty(), "captured:\n{}", failures.join("\n"));

    // Another server's tool with the same name is captured when listed.
    for name in [
        "mcp__other-server__recall",
        "mcp__plugin_tools_other__remember",
    ] {
        let payload = tool(name, json!({}), STRONG);
        let c = only(
            &f.run(
                HookEvent::PostToolUse,
                &payload,
                &settings_with_tools(&[name]),
            ),
            name,
        );
        assert_eq!(c.content, format!("Tool: {name}\nResponse: {STRONG}"));
    }
}

#[test]
fn s16_edit_and_write_are_anchored_to_their_file() {
    let f = Fixture::new();
    let settings = CaptureSettings::default();
    let absolute = f.project.root.join("src").join("db.rs");
    let absolute = absolute.to_str().expect("a UTF-8 path");
    let file = |name: &str| vec![AnchorRef::new(AnchorKind::File, name).expect("a valid anchor")];
    for (payload, anchor) in [
        (edit("src/db.rs", STRONG), "src/db.rs"),
        (edit(absolute, STRONG), "src/db.rs"),
        (edit("./src/db.rs", STRONG), "src/db.rs"),
        (write("docs/fix.md", STRONG), "docs/fix.md"),
        (write(absolute, STRONG), "src/db.rs"),
    ] {
        let c = only(&f.run(HookEvent::PostToolUse, &payload, &settings), anchor);
        assert_eq!(c.anchors, file(anchor), "{payload}");
        assert_eq!(c.dropped_anchor, None);
        assert_eq!(c.provenance.file_path.as_deref(), Some(anchor));
    }

    // NotebookEdit records the notebook as the file path but has no anchor;
    // Bash has neither.
    let notebook = tool(
        "NotebookEdit",
        json!({"notebook_path": "nb/analysis.ipynb", "new_source": STRONG}),
        "ok",
    );
    let c = only(
        &f.run(HookEvent::PostToolUse, &notebook, &settings),
        "NotebookEdit",
    );
    assert!(c.anchors.is_empty());
    assert_eq!(c.provenance.file_path.as_deref(), Some("nb/analysis.ipynb"));
    let bash = tool("Bash", json!({"command": "make"}), STRONG);
    let c = only(&f.run(HookEvent::PostToolUse, &bash, &settings), "Bash");
    assert!(c.anchors.is_empty());
    assert_eq!(c.provenance.file_path, None);

    // Transcript captures have no file and no anchor.
    for (event, payload) in f.capture_payloads().into_iter().skip(1) {
        let c = only(&f.run(event, &payload, &settings), "transcript");
        assert!(c.anchors.is_empty(), "{event:?}");
        assert_eq!(c.provenance.file_path, None);
    }
}

#[test]
fn s16_provenance_records_capture_with_session_tool_file_and_commit() {
    let f = Fixture::new();
    let settings = CaptureSettings::default();
    let c = only(
        &f.run(
            HookEvent::PostToolUse,
            &edit("src/db.rs", STRONG),
            &settings,
        ),
        "Edit",
    );
    let p = &c.provenance;
    assert_eq!(p.source, ProvenanceSource::Capture);
    assert_eq!(p.session_id.as_deref(), Some(SESSION));
    assert_eq!(p.tool.as_deref(), Some("Edit"));
    assert_eq!(p.file_path.as_deref(), Some("src/db.rs"));
    assert_eq!(p.commit_sha.as_deref(), Some(COMMIT));
    assert_eq!(p.author, None);

    let expected = [
        (TextSource::ToolCall, None, "post-tool-use"),
        (TextSource::Transcript, Some("auto"), "pre-compact"),
        (TextSource::Transcript, Some("session_end"), "stop"),
        (
            TextSource::LastAssistantMessage,
            Some("session_end"),
            "stop",
        ),
    ];
    for ((event, payload), (source, trigger, hook)) in
        f.capture_payloads().into_iter().zip(expected)
    {
        let outcome = f.run(event, &payload, &settings);
        assert_eq!(outcome.source, Some(source), "{event:?}");
        let c = only(&outcome, hook);
        let p = &c.provenance;
        assert_eq!(p.source, ProvenanceSource::Capture, "{hook}");
        assert_eq!(p.session_id.as_deref(), Some(SESSION), "{hook}");
        assert_eq!(p.commit_sha.as_deref(), Some(COMMIT), "{hook}");
        let extra = p.extra.as_ref().expect("the classification in extra");
        assert_eq!(extra["hook"], json!(hook), "{hook}: {extra}");
        assert_eq!(
            extra.get("trigger").and_then(Value::as_str),
            trigger,
            "{hook}: {extra}"
        );
        if event == HookEvent::PostToolUse {
            assert_eq!(p.tool.as_deref(), Some("Edit"));
        } else {
            assert_eq!(p.tool, None, "{hook}");
        }
    }

    // Without a commit from the caller, none is recorded.
    let mut state = SessionState::new();
    let outcome = capture(
        &CaptureInput {
            event: HookEvent::PostToolUse,
            payload: &edit("a.rs", STRONG),
            settings: &settings,
            project: &f.project,
            commit_sha: None,
            transcript_offset: 0,
        },
        &mut state,
    );
    assert_eq!(only(&outcome, "no commit").provenance.commit_sha, None);
}

/// SPEC section 16's type for each family: decision, pitfall, convention,
/// or a note. Written out here rather than taken from
/// [`crate::signal::memory_type_for`], which this checks.
fn spec_type(family: &str) -> MemoryType {
    match family {
        "decision" => MemoryType::Decision,
        "error" | "solution" | "testing" => MemoryType::Pitfall,
        "pattern" | "convention" | "workaround" | "configuration" | "performance" => {
            MemoryType::Convention
        }
        _ => MemoryType::Note,
    }
}

/// A text for each of the 22 patterns that matches that pattern alone.
const PATTERN_SAMPLES: [(&str, &str); 22] = [
    ("decision", "Decided to use Redis for caching."),
    ("important", "Important: keep the cache warm."),
    ("solution_fix", "The fix was to retry the call."),
    ("error_diagnosis", "The bug was a stale connection pool."),
    (
        "explicit_save",
        "Save that the cache key changes each release.",
    ),
    ("pattern_learning", "The trick is to batch the writes."),
    ("resolved_by", "Resolved by pinning the driver."),
    (
        "broke_failed_because",
        "The build broke because the lockfile was stale.",
    ),
    ("error_due_to", "The error was due to a missing index."),
    ("architecture", "The architecture will be event driven."),
    ("definition", "A shard is a slice of the index."),
    ("convention", "The convention here is snake_case names."),
    ("watch_out_for", "Watch out for trailing slashes."),
    ("workaround_is", "The workaround is to clear the cache."),
    (
        "configuration",
        "You need to set RUST_LOG for verbose output.",
    ),
    ("testing", "The tests passed on the second run."),
    ("performance", "Latency doubled under load."),
    ("environment", "This requires version 3.11 of the runtime."),
    ("dependency", "The crate depends on serde."),
    ("api", "The endpoint rejects empty names."),
    ("procedure", "Steps to rebuild the index are in the wiki."),
    ("workflow", "Always run before merging the branch."),
];

#[test]
fn s16_types_come_from_pattern_families() {
    let f = Fixture::new();
    let settings = CaptureSettings::default();
    let mut failures = Vec::new();
    for (pattern, (id, text)) in SIGNAL_PATTERNS.iter().zip(PATTERN_SAMPLES) {
        assert_eq!(pattern.id, id, "the samples follow the pattern order");
        let c = only(
            &f.run(HookEvent::PostToolUse, &edit("", text), &settings),
            id,
        );
        let expected = spec_type(pattern.family);
        if c.patterns != [pattern.family] || c.memory_type != expected {
            failures.push(format!(
                "{id}: families {:?}, type {:?}, expected [{}] and {expected:?}",
                c.patterns, c.memory_type, pattern.family
            ));
        }
        // Every pattern is at least medium, so each gives a candidate, stored
        // pending until the judge decides (B-157).
        if c.tier == Tier::Weak || c.status != Status::Pending {
            failures.push(format!("{id}: tier {:?}, status {:?}", c.tier, c.status));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // With several families, version 1's namespace order decides: a
    // decision before a pitfall, a pitfall before a convention.
    for (text, families, expected) in [
        (
            "The bug was a stale pool. Decided to use Redis.",
            vec!["error", "decision"],
            MemoryType::Decision,
        ),
        (
            "Watch out for stale pools. The fix was to retry.",
            vec!["solution", "workaround"],
            MemoryType::Pitfall,
        ),
        (
            "The endpoint is slow; latency doubled.",
            vec!["performance", "api"],
            MemoryType::Convention,
        ),
    ] {
        let c = only(
            &f.run(HookEvent::PostToolUse, &edit("", text), &settings),
            text,
        );
        let got: BTreeSet<&str> = c.patterns.iter().map(String::as_str).collect();
        let want: BTreeSet<&str> = families.into_iter().collect();
        assert_eq!(got, want, "{text}");
        assert_eq!(c.memory_type, expected, "{text}");
    }
}

#[test]
fn s16_to_new_memory_leaves_the_store_fields_empty() {
    let f = Fixture::new();
    let c = only(
        &f.run(
            HookEvent::PostToolUse,
            &edit("src/db.rs", STRONG),
            &CaptureSettings::default(),
        ),
        "Edit",
    );
    let memory = c.to_new_memory(ProjectId(7));
    assert_eq!(memory.id, None);
    assert_eq!(memory.title, None);
    assert_eq!(memory.created_at, None);
    assert_eq!(memory.updated_at, None);
    assert_eq!(memory.supersedes_id, None);
    assert_eq!(memory.imported, None);
    assert_eq!(memory.project_id, ProjectId(7));
    assert_eq!(memory.origin, Origin::Local);
    assert_eq!(memory.content, c.content);
    assert_eq!(
        memory.content_hash.as_deref(),
        Some(c.content_hash.as_str())
    );
    assert_eq!(memory.memory_type, MemoryType::Decision);
    assert_eq!(memory.status, Status::Pending);
    assert_eq!(memory.importance, c.importance);
    assert_eq!(memory.tags, c.tags);
    assert_eq!(memory.anchors, c.anchors);
    assert_eq!(memory.provenance, c.provenance);
}

/// Stop with `stop_hook_active` set gives nothing (the loop guard), and a
/// transcript event without a usable path says so.
#[test]
fn s16_stop_loop_guard_and_missing_transcript() {
    let f = Fixture::new();
    let settings = CaptureSettings::default();
    let looping =
        json!({"session_id": SESSION, "stop_hook_active": true, "last_assistant_message": STRONG});
    assert_eq!(
        f.run(HookEvent::Stop, &looping, &settings).skipped,
        Some(Skip::StopHookActive)
    );
    for event in [HookEvent::PreCompact, HookEvent::Stop] {
        for payload in [
            json!({"session_id": SESSION}),
            json!({"session_id": SESSION, "transcript_path": "relative/session.jsonl"}),
            json!({"session_id": SESSION, "transcript_path": "../session.jsonl"}),
        ] {
            let outcome = f.run(event, &payload, &settings);
            assert_eq!(
                outcome.skipped,
                Some(Skip::NoTranscript),
                "{event:?} {payload}"
            );
        }
    }
}
