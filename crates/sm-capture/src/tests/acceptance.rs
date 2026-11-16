//! The library parts of AT-14, AT-16 and AT-17 (SPEC section 11). AT-9 is
//! in `redaction.rs` and AT-13 in `hook_payloads.rs`.

use std::path::PathBuf;

use serde_json::{Value, json};
use sm_core::config::CaptureSettings;
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::HookEvent;
use sm_core::types::{ProvenanceSource, Status};

use super::{copy_fixtures, project, settings_with_tools, temp_dir};
use crate::extract::{MAX_CAPTURE_CHARS, TRUNCATION_MARKER};
use crate::pipeline::{CaptureInput, CaptureOutcome, TextDrop, TextSource, capture};
use crate::session::SessionState;
use crate::signal::Tier;

/// The AT-14 transcript: two assistant texts PostToolUse also captured,
/// one it missed, a weak one, and a user turn.
const AT14_TRANSCRIPT: &str = "rust/at14_session.jsonl";
const CAPTURED_EDIT: &str = "Decided to use PostgreSQL because it handles JSONB well.";
const CAPTURED_WRITE: &str = "The fix is to add the missing import statement.";
const MISSED: &str = "Decided to use Redis for the session cache.";

struct Session {
    _dir: tempfile::TempDir,
    transcript: PathBuf,
    project: ProjectIdentity,
    settings: CaptureSettings,
    state: SessionState,
}

impl Session {
    fn new() -> Self {
        let dir = temp_dir();
        copy_fixtures(dir.path(), &[AT14_TRANSCRIPT]);
        let transcript = dir.path().join("at14_session.jsonl");
        let project = project(dir.path());
        Session {
            _dir: dir,
            transcript,
            project,
            settings: CaptureSettings::default(),
            state: SessionState::new(),
        }
    }

    fn run(&mut self, event: HookEvent, payload: &Value, offset: u64) -> CaptureOutcome {
        let input = CaptureInput {
            event,
            payload,
            settings: &self.settings,
            project: &self.project,
            commit_sha: None,
            transcript_offset: offset,
        };
        capture(&input, &mut self.state)
    }

    fn transcript_path(&self) -> String {
        self.transcript.to_str().expect("a UTF-8 path").to_owned()
    }

    fn transcript_len(&self) -> u64 {
        std::fs::metadata(&self.transcript)
            .expect("the transcript")
            .len()
    }

    /// PostToolUse captures two of the transcript's texts. An Edit or Write
    /// without a file path gives its text alone, so the capture's content
    /// is the assistant's text.
    fn post_tool_use_captures(&mut self) {
        for (tool, input) in [
            ("Edit", json!({"new_string": CAPTURED_EDIT})),
            ("Write", json!({"content": CAPTURED_WRITE})),
        ] {
            let payload = json!({"session_id": "s1", "tool_name": tool, "tool_input": input, "tool_response": "ok"});
            let outcome = self.run(HookEvent::PostToolUse, &payload, 0);
            assert_eq!(
                contents(&outcome),
                [if tool == "Edit" {
                    CAPTURED_EDIT
                } else {
                    CAPTURED_WRITE
                }]
            );
        }
        assert_eq!(self.state.len(), 2);
    }
}

fn contents(outcome: &CaptureOutcome) -> Vec<&str> {
    outcome
        .candidates
        .iter()
        .map(|c| c.content.as_str())
        .collect()
}

/// AT-14: PreCompact returns the assistant text PostToolUse missed and not
/// what it already captured. A re-read from an older offset returns nothing
/// new: the reader is stateless, and the session's hashes catch it.
#[test]
fn at14_pre_compact_returns_only_what_post_tool_use_missed() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let payload =
        json!({"session_id": "s1", "transcript_path": s.transcript_path(), "trigger": "auto"});

    let first = s.run(HookEvent::PreCompact, &payload, 0);
    assert_eq!(first.skipped, None);
    assert_eq!(first.source, Some(TextSource::Transcript));
    assert_eq!(contents(&first), [MISSED]);
    assert_eq!(
        first.drops,
        [TextDrop::Duplicate, TextDrop::Duplicate, TextDrop::Weak]
    );
    assert_eq!(first.new_offset, Some(s.transcript_len()));
    let c = &first.candidates[0];
    assert_eq!(c.source_hook, HookEvent::PreCompact);
    assert_eq!(c.tags, ["transcript", "decision"]);
    assert!(c.anchors.is_empty(), "a transcript capture has no anchor");
    assert_eq!(
        c.provenance.extra.as_ref().map(|e| e["trigger"].clone()),
        Some(json!("auto"))
    );

    let next = s.run(HookEvent::PreCompact, &payload, s.transcript_len());
    assert!(next.candidates.is_empty() && next.drops.is_empty());
    assert_eq!(next.new_offset, Some(s.transcript_len()));

    let reread = s.run(HookEvent::PreCompact, &payload, 0);
    assert!(reread.candidates.is_empty());
    assert_eq!(
        reread.drops,
        [
            TextDrop::Duplicate,
            TextDrop::Duplicate,
            TextDrop::Duplicate,
            TextDrop::Weak
        ]
    );
}

/// AT-14: Stop without `last_assistant_message` reads the transcript and
/// returns only the missed text.
#[test]
fn at14_stop_reads_the_transcript_when_there_is_no_final_message() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let payload = json!({"session_id": "s1", "transcript_path": s.transcript_path()});
    let stop = s.run(HookEvent::Stop, &payload, 0);
    assert_eq!(stop.source, Some(TextSource::Transcript));
    assert_eq!(contents(&stop), [MISSED]);
    assert_eq!(stop.new_offset, Some(s.transcript_len()));
    let extra = stop.candidates[0]
        .provenance
        .extra
        .clone()
        .expect("provenance extra");
    assert_eq!(extra["trigger"], json!("session_end"));
}

/// AT-14: Stop prefers `last_assistant_message` and then does not read the
/// transcript; a final message PostToolUse already captured gives nothing.
#[test]
fn at14_stop_prefers_last_assistant_message() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let message = "The workaround is to pin the driver to version 2.";
    let payload = json!({
        "session_id": "s1",
        "transcript_path": s.transcript_path(),
        "last_assistant_message": message,
    });
    let stop = s.run(HookEvent::Stop, &payload, 0);
    assert_eq!(stop.source, Some(TextSource::LastAssistantMessage));
    assert_eq!(contents(&stop), [message]);
    assert_eq!(stop.new_offset, None, "the transcript is not read");

    let repeat = json!({
        "session_id": "s1",
        "transcript_path": s.transcript_path(),
        "last_assistant_message": format!("  {CAPTURED_WRITE}\n"),
    });
    let stop = s.run(HookEvent::Stop, &repeat, 0);
    assert!(stop.candidates.is_empty());
    assert_eq!(stop.drops, [TextDrop::Duplicate]);

    // The missed transcript text is still there for PreCompact.
    let payload = json!({"session_id": "s1", "transcript_path": s.transcript_path()});
    assert_eq!(
        contents(&s.run(HookEvent::PreCompact, &payload, 0)),
        [MISSED]
    );
}

/// AT-16: a medium signal gives an active candidate with importance at least
/// 0.4, a strong one at least 0.7, from every capture source; the
/// candidate's provenance source is `capture`, and `to_new_memory` keeps
/// the status and importance.
#[test]
fn at16_medium_and_strong_signals_give_active_candidates() {
    let dir = temp_dir();
    let project = project(dir.path());
    let settings = CaptureSettings::default();
    let transcript = dir.path().join("t.jsonl");
    let project_id = ProjectId(7);

    for (text, tier, floor) in [
        (
            "The convention here is to use snake_case for module names.",
            Tier::Medium,
            0.4,
        ),
        (
            "You need to set RUST_LOG=debug for verbose output.",
            Tier::Medium,
            0.4,
        ),
        (
            "Decided to use Redis for the session cache.",
            Tier::Strong,
            0.7,
        ),
        ("The bug was a stale connection pool.", Tier::Strong, 0.7),
    ] {
        let entry = json!({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}, "uuid": "u1"});
        std::fs::write(&transcript, format!("{entry}\n")).expect("write a transcript");
        let path = transcript.to_str().expect("UTF-8");
        let cases = [
            (
                HookEvent::PostToolUse,
                json!({"tool_name": "Edit", "tool_input": {"new_string": text}}),
            ),
            (HookEvent::PreCompact, json!({"transcript_path": path})),
            (HookEvent::Stop, json!({"last_assistant_message": text})),
        ];
        for (event, payload) in cases {
            let input = CaptureInput {
                event,
                payload: &payload,
                settings: &settings,
                project: &project,
                commit_sha: Some("0123abcd"),
                transcript_offset: 0,
            };
            let outcome = capture(&input, &mut SessionState::new());
            assert_eq!(outcome.candidates.len(), 1, "{event:?} {text}");
            let c = &outcome.candidates[0];
            assert_eq!(c.tier, tier, "{text}");
            assert!(
                c.importance >= floor,
                "{event:?} {text}: importance {}",
                c.importance
            );
            assert!(c.importance <= 1.0);
            assert_eq!(c.status, Status::Active);
            assert_eq!(c.provenance.source, ProvenanceSource::Capture);
            assert_eq!(c.provenance.commit_sha.as_deref(), Some("0123abcd"));
            let memory = c.to_new_memory(project_id);
            assert_eq!(memory.status, Status::Active);
            assert_eq!(memory.importance, c.importance);
            assert_eq!(memory.content, c.content);
        }
    }
}

fn long_text(chars: usize) -> String {
    let head = "The fix is to raise the limit. ";
    let mut text = String::with_capacity(chars);
    text.push_str(head);
    text.extend(std::iter::repeat_n('é', chars - head.chars().count()));
    text
}

fn assert_cut(outcome: &CaptureOutcome, what: &str) {
    assert!(outcome.cap_hit, "{what}: the cap was hit");
    assert_eq!(outcome.candidates.len(), 1, "{what}");
    let c = &outcome.candidates[0];
    assert!(c.cap_hit, "{what}: the candidate records the cut");
    assert!(c.content.ends_with(TRUNCATION_MARKER), "{what}: the marker");
    assert_eq!(
        c.content.chars().count(),
        MAX_CAPTURE_CHARS + TRUNCATION_MARKER.chars().count(),
        "{what}: cut at the cap"
    );
}

/// AT-17, library part: a capture longer than 200,000 characters is cut at
/// extraction, with the marker, from every source. The index cap is the
/// chunker's (stream B) and is not repeated here.
#[test]
fn at17_long_capture_is_cut_at_extraction_with_the_marker() {
    let dir = temp_dir();
    let project = project(dir.path());
    let settings = settings_with_tools(&["Bash"]);
    let long = long_text(250_000);

    let bash =
        json!({"tool_name": "Bash", "tool_input": {"command": "make"}, "tool_response": long});
    let stop = json!({"last_assistant_message": long});
    let transcript = dir.path().join("long.jsonl");
    let entry = json!({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": long}]}, "uuid": "u1"});
    std::fs::write(&transcript, format!("{entry}\n")).expect("write a transcript");
    let pre_compact = json!({"transcript_path": transcript.to_str().expect("UTF-8")});

    for (event, payload, before_cap) in [
        (
            HookEvent::PostToolUse,
            &bash,
            "Command: make\nOutput: ".chars().count() + 250_000,
        ),
        (HookEvent::Stop, &stop, 250_000),
        (HookEvent::PreCompact, &pre_compact, 250_000),
    ] {
        let input = CaptureInput {
            event,
            payload,
            settings: &settings,
            project: &project,
            commit_sha: None,
            transcript_offset: 0,
        };
        let outcome = capture(&input, &mut SessionState::new());
        assert_cut(&outcome, event.as_str());
        assert_eq!(
            outcome.extracted_chars, before_cap,
            "{event:?}: the size before the cap"
        );
    }

    // At the cap exactly, nothing is cut.
    let exact = json!({"last_assistant_message": long_text(MAX_CAPTURE_CHARS)});
    let input = CaptureInput {
        event: HookEvent::Stop,
        payload: &exact,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: 0,
    };
    let outcome = capture(&input, &mut SessionState::new());
    assert!(!outcome.cap_hit);
    assert_eq!(
        outcome.candidates[0].content.chars().count(),
        MAX_CAPTURE_CHARS
    );
}
