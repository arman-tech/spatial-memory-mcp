//! The library parts of AT-14, AT-16 and AT-17 (SPEC section 11). AT-9 is
//! in `redaction.rs` and AT-13 in `hook_payloads.rs`.

use std::path::PathBuf;

use serde_json::{Value, json};
use sm_core::config::CaptureSettings;
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::HookEvent;
use sm_core::types::{ProvenanceSource, Status};

use super::{copy_fixtures, project, settings_with_tools, source_of, temp_dir, v1_tool_settings};
use crate::extract::{MAX_CAPTURE_CHARS, TRUNCATION_MARKER};
use crate::pipeline::{
    CaptureInput, CaptureOutcome, TextDrop, TextSource, WEAK_IMPORTANCE, capture,
};
use crate::select::UNIT_MAX;
use crate::session::SessionState;
use crate::signal::Tier;
use crate::transcript::PROMPT_SOURCE;

/// The AT-14 transcript: two assistant texts PostToolUse also captured,
/// one it missed and a weak one, each the final text of its own turn
/// (B-162 reads a turn's final text as one), and the user prompts that
/// start the turns. Since B-179 a prompt of at least 20 characters is
/// offered too, and the weak reply is pending like any other text.
const AT14_TRANSCRIPT: &str = "rust/at14_session.jsonl";
const CAPTURED_EDIT: &str = "Decided to use PostgreSQL because it handles JSONB well.";
const CAPTURED_WRITE: &str = "The fix is to add the missing import statement.";
const MISSED: &str = "Decided to use Redis for the session cache.";
const ASK_IMPORT: &str = "Why did the import fail?";
const ASK_CACHE: &str = "Which cache did you pick?";
/// The last turn's reply, which the pattern rules call weak. Its prompt,
/// "Run the tests.", is shorter than a unit and is not offered.
const WEAK: &str = "Running the tests now.";

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
            // AT-14 needs PostToolUse to capture, which it does only for
            // the tools `capture.tools` names (B-160).
            settings: v1_tool_settings(),
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
            transcript_offset: (offset > 0).then_some(offset),
            from_start: true,
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

    /// The offset of the transcript's second line: a session the daemon
    /// has seen up to the end of the first reply.
    fn after_first_line(&self) -> u64 {
        let bytes = std::fs::read(&self.transcript).expect("the transcript");
        let end = bytes
            .iter()
            .position(|b| *b == b'\n')
            .expect("a line break");
        u64::try_from(end + 1).expect("a small offset")
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

fn sources(outcome: &CaptureOutcome) -> Vec<&str> {
    outcome.candidates.iter().map(source_of).collect()
}

/// AT-14: PreCompact returns what PostToolUse missed and not what it
/// already captured. The read of a session the daemon has not seen is its
/// live segment, here the whole file. The two prompts, the missed reply and
/// the weak last reply are new: since B-179 the pattern rules no longer
/// drop a transcript text, and a weak one is pending at importance 0.4 for
/// the judge. The two replies PostToolUse captured are duplicates. A
/// re-read from an older offset returns nothing new: the reader is
/// stateless, and the session's hashes catch it.
#[test]
fn at14_pre_compact_returns_only_what_post_tool_use_missed() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let payload =
        json!({"session_id": "s1", "transcript_path": s.transcript_path(), "trigger": "auto"});

    let first = s.run(HookEvent::PreCompact, &payload, 0);
    assert_eq!(first.skipped, None);
    assert_eq!(first.source, Some(TextSource::Transcript));
    assert_eq!(contents(&first), [ASK_IMPORT, ASK_CACHE, MISSED, WEAK]);
    assert_eq!(sources(&first), ["prompt", "prompt", "reply", "reply"]);
    assert_eq!(first.drops, [TextDrop::Duplicate, TextDrop::Duplicate]);
    assert_eq!(first.new_offset, Some(s.transcript_len()));
    let c = &first.candidates[2];
    assert_eq!(c.source_hook, HookEvent::PreCompact);
    assert_eq!(c.tags, ["transcript", "decision"]);
    assert_eq!(c.context, ASK_CACHE, "a reply carries its turn's prompt");
    assert!(c.anchors.is_empty(), "a transcript capture has no anchor");
    assert_eq!(
        c.provenance.extra.as_ref().map(|e| e["trigger"].clone()),
        Some(json!("auto"))
    );
    for prompt in &first.candidates[..2] {
        assert_eq!(prompt.context, PROMPT_SOURCE);
    }
    let weak = &first.candidates[3];
    assert_eq!(weak.tier, Tier::Weak, "the tier stays weak in provenance");
    assert_eq!(weak.importance, WEAK_IMPORTANCE);
    for c in &first.candidates {
        assert_eq!(c.status, Status::Pending);
    }

    let next = s.run(HookEvent::PreCompact, &payload, s.transcript_len());
    assert!(next.candidates.is_empty() && next.drops.is_empty());
    assert_eq!(next.new_offset, Some(s.transcript_len()));

    let reread = s.run(HookEvent::PreCompact, &payload, 0);
    assert!(reread.candidates.is_empty());
    assert_eq!(reread.drops, [TextDrop::Duplicate; 6]);
}

/// AT-14: Stop without `last_assistant_message` reads the transcript. A
/// session the daemon has not seen gives only its current turn (B-179,
/// as B-162 gives such a session no history): here the weak final reply.
/// From a seen session's offset it gives every text since, and of the
/// replies only what PostToolUse and the first Stop did not take.
#[test]
fn at14_stop_reads_the_transcript_when_there_is_no_final_message() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let payload = json!({"session_id": "s1", "transcript_path": s.transcript_path()});
    let stop = s.run(HookEvent::Stop, &payload, 0);
    assert_eq!(stop.source, Some(TextSource::Transcript));
    assert_eq!(contents(&stop), [WEAK], "only the current turn");
    assert!(stop.drops.is_empty(), "{:?}", stop.drops);
    assert_eq!(stop.new_offset, Some(s.transcript_len()));
    let extra = stop.candidates[0]
        .provenance
        .extra
        .clone()
        .expect("provenance extra");
    assert_eq!(extra["trigger"], json!("session_end"));

    let seen = s.run(HookEvent::Stop, &payload, s.after_first_line());
    assert_eq!(contents(&seen), [ASK_IMPORT, ASK_CACHE, MISSED]);
    assert_eq!(seen.drops, [TextDrop::Duplicate, TextDrop::Duplicate]);
    assert_eq!(seen.new_offset, Some(s.transcript_len()));
}

/// AT-14: Stop reads its turn from the transcript and takes
/// `last_assistant_message` as the turn's reply too (B-179: the turn's
/// prompt and tool texts are only in the transcript, and the transcript
/// may not hold the final message yet). When it does, the session's hashes
/// drop the repeat. A final message PostToolUse already captured gives
/// nothing. Without a transcript the message is taken alone.
#[test]
fn at14_stop_reads_its_turn_and_takes_last_assistant_message() {
    let mut s = Session::new();
    s.post_tool_use_captures();
    let same = json!({
        "session_id": "s1",
        "transcript_path": s.transcript_path(),
        "last_assistant_message": WEAK,
    });
    let stop = s.run(HookEvent::Stop, &same, 0);
    assert_eq!(stop.source, Some(TextSource::Transcript));
    assert_eq!(contents(&stop), [WEAK], "the final reply is taken once");
    assert_eq!(stop.drops, [TextDrop::Duplicate]);
    assert_eq!(stop.new_offset, Some(s.transcript_len()));

    let mut s = Session::new();
    s.post_tool_use_captures();
    let message = "The workaround is to pin the driver to version 2.";
    let payload = json!({
        "session_id": "s1",
        "transcript_path": s.transcript_path(),
        "last_assistant_message": message,
    });
    let stop = s.run(HookEvent::Stop, &payload, 0);
    assert_eq!(stop.source, Some(TextSource::Transcript));
    assert_eq!(contents(&stop), [WEAK, message]);
    assert_eq!(sources(&stop), ["reply", "reply"]);
    // B-162: the session's offset moves to the transcript's end, so the
    // next read does not take the final message again.
    assert_eq!(stop.new_offset, Some(s.transcript_len()));

    let repeat = json!({
        "session_id": "s1",
        "transcript_path": s.transcript_path(),
        "last_assistant_message": format!("  {CAPTURED_WRITE}\n"),
    });
    let stop = s.run(HookEvent::Stop, &repeat, 0);
    assert!(stop.candidates.is_empty());
    assert_eq!(stop.drops, [TextDrop::Duplicate, TextDrop::Duplicate]);

    // The missed transcript text is still there for PreCompact.
    let payload = json!({"session_id": "s1", "transcript_path": s.transcript_path()});
    assert_eq!(
        contents(&s.run(HookEvent::PreCompact, &payload, 0)),
        [ASK_IMPORT, ASK_CACHE, MISSED]
    );

    let alone = "The cache key is the session id and the user id together.";
    let stop = s.run(
        HookEvent::Stop,
        &json!({"session_id": "s1", "last_assistant_message": alone}),
        0,
    );
    assert_eq!(stop.source, Some(TextSource::LastAssistantMessage));
    assert_eq!(contents(&stop), [alone]);
    assert_eq!(sources(&stop), ["reply"]);
}

/// AT-16: a medium signal gives a pending candidate (B-157) with importance
/// at least 0.4, a strong one at least 0.7, from every capture source; the
/// candidate's provenance source is `capture`, and `to_new_memory` keeps
/// the status and importance.
#[test]
fn at16_medium_and_strong_signals_give_pending_candidates() {
    let dir = temp_dir();
    let project = project(dir.path());
    // PostToolUse is a capture source only for a tool `capture.tools`
    // names (B-160).
    let settings = v1_tool_settings();
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
                transcript_offset: None,
                from_start: false,
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
            assert_eq!(c.status, Status::Pending);
            assert_eq!(c.provenance.source, ProvenanceSource::Capture);
            assert_eq!(c.provenance.commit_sha.as_deref(), Some("0123abcd"));
            let memory = c.to_new_memory(project_id);
            assert_eq!(memory.status, Status::Pending);
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

/// A text of exactly `chars` characters made of distinct numbered lines,
/// each holding a non-ASCII character, so that lengths are counted in
/// characters and no unit repeats another.
fn long_lines(chars: usize) -> String {
    let mut text = String::with_capacity(chars * 2);
    let mut count = 0;
    let mut i = 0;
    while count < chars {
        i += 1;
        let line = format!("Line {i:06} é: the limit is raised once.\n");
        count += line.chars().count();
        text.push_str(&line);
    }
    let mut text: String = text.chars().take(chars - 1).collect();
    text.push('é');
    text
}

/// PostToolUse: the one candidate is cut at the cap, with the marker.
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

/// A transcript text (B-179): the text is cut at the cap with the marker,
/// then split into units of at most [`UNIT_MAX`] characters, each recording
/// the cut. Joined again, the units are the cut text.
fn assert_cut_units(outcome: &CaptureOutcome, what: &str, text: &str) {
    assert!(outcome.cap_hit, "{what}: the cap was hit");
    assert!(outcome.candidates.len() > 1, "{what}: split into units");
    for c in &outcome.candidates {
        assert!(c.cap_hit, "{what}: each unit records the cut");
        assert!(c.content.chars().count() <= UNIT_MAX, "{what}: a unit");
    }
    let cut: String = text.chars().take(MAX_CAPTURE_CHARS).collect();
    assert!(
        contents(outcome).join("\n") == format!("{cut}{TRUNCATION_MARKER}"),
        "{what}: the units are the text cut at the cap, with the marker"
    );
}

/// AT-17, library part: a capture longer than 200,000 characters is cut at
/// extraction, with the marker, from every source. A transcript text is
/// then split into the units offered to the judge (B-179). The index cap
/// is the chunker's (stream B) and is not repeated here.
#[test]
fn at17_long_capture_is_cut_at_extraction_with_the_marker() {
    let dir = temp_dir();
    let project = project(dir.path());
    let settings = settings_with_tools(&["Bash"]);
    let input = |event, payload| CaptureInput {
        event,
        payload,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: None,
        from_start: false,
    };

    let long = long_text(250_000);
    let bash =
        json!({"tool_name": "Bash", "tool_input": {"command": "make"}, "tool_response": long});
    let outcome = capture(
        &input(HookEvent::PostToolUse, &bash),
        &mut SessionState::new(),
    );
    assert_cut(&outcome, "PostToolUse");
    assert_eq!(
        outcome.extracted_chars,
        "Command: make\nOutput: ".chars().count() + 250_000,
        "PostToolUse: the size before the cap"
    );

    let lines = long_lines(250_000);
    let stop = json!({"last_assistant_message": lines});
    let transcript = dir.path().join("long.jsonl");
    let entry = json!({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": lines}]}, "uuid": "u1"});
    std::fs::write(&transcript, format!("{entry}\n")).expect("write a transcript");
    let pre_compact = json!({"transcript_path": transcript.to_str().expect("UTF-8")});
    for (event, payload) in [
        (HookEvent::Stop, &stop),
        (HookEvent::PreCompact, &pre_compact),
    ] {
        let outcome = capture(&input(event, payload), &mut SessionState::new());
        assert_cut_units(&outcome, event.as_str(), &lines);
        assert_eq!(
            outcome.extracted_chars, 250_000,
            "{event:?}: the size before the cap"
        );
    }

    // At the cap exactly, nothing is cut.
    let exact_text = long_lines(MAX_CAPTURE_CHARS);
    let exact = json!({"last_assistant_message": exact_text});
    let outcome = capture(&input(HookEvent::Stop, &exact), &mut SessionState::new());
    assert!(!outcome.cap_hit);
    assert!(outcome.candidates.iter().all(|c| !c.cap_hit));
    assert!(contents(&outcome).join("\n") == exact_text);
}
