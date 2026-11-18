//! B-172: `replay` runs a session's stored transcript through the pipeline
//! again, on request only (SPEC section 16, Replay). It reads the whole
//! transcript, before and after each compaction boundary, and never moves
//! the session's read position. The transcripts are synthetic.

use std::path::Path;

use serde_json::{Value, json};

use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;

use super::{project, temp_dir};
use crate::extract::{MAX_CAPTURE_CHARS, TRUNCATION_MARKER};
use crate::pipeline::{
    CaptureInput, CaptureOutcome, REPLAY_TRIGGER, Skip, TextSource, capture, replay,
};
use crate::session::SessionState;

fn entry(kind: &str, uuid: &str, content: Value) -> Value {
    json!({
        "type": kind,
        "uuid": uuid,
        "timestamp": format!("t-{uuid}"),
        "message": {"role": kind, "content": content},
    })
}

fn user(uuid: &str, text: &str) -> Value {
    entry("user", uuid, json!(text))
}

fn says(uuid: &str, text: &str) -> Value {
    entry("assistant", uuid, json!([{"type": "text", "text": text}]))
}

fn says_then_calls(uuid: &str, text: &str) -> Value {
    entry(
        "assistant",
        uuid,
        json!([
            {"type": "text", "text": text},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]),
    )
}

fn tool_result(uuid: &str) -> Value {
    entry(
        "user",
        uuid,
        json!([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]),
    )
}

fn compact() -> Value {
    json!({
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": "cb",
        "timestamp": "t-cb",
        "content": "Conversation compacted",
    })
}

fn write(dir: &Path, entries: &[Value]) -> String {
    let p = dir.join("session.jsonl");
    let text: String = entries.iter().map(|e| format!("{e}\n")).collect();
    std::fs::write(&p, text).expect("write a transcript");
    p.to_str().expect("a UTF-8 temporary path").to_owned()
}

/// Runs `replay`, or with `live` a PreCompact capture, on `path` with the
/// read position `offset` and fresh session state.
fn run(
    live: bool,
    path: &str,
    offset: Option<u64>,
    settings: &CaptureSettings,
    root: &Path,
) -> CaptureOutcome {
    let payload = json!({"session_id": "s-1", "transcript_path": path, "cwd": root});
    let project = project(root);
    let input = CaptureInput {
        event: HookEvent::PreCompact,
        payload: &payload,
        settings,
        project: &project,
        commit_sha: None,
        transcript_offset: offset,
    };
    if live {
        capture(&input, &mut SessionState::new())
    } else {
        replay(&input, &mut SessionState::new())
    }
}

fn contents(o: &CaptureOutcome) -> Vec<&str> {
    o.candidates.iter().map(|c| c.content.as_str()).collect()
}

#[test]
fn s16_b172_a_replay_reads_the_whole_transcript_with_each_turns_prompt() {
    let dir = temp_dir();
    let path = write(
        dir.path(),
        &[
            user("u1", "Which cache should the sessions use?"),
            says_then_calls("a1", "Let me read the cache code first."),
            tool_result("r1"),
            says("a2", "Decided to use Redis for the session cache."),
            compact(),
            user("u2", "And how long do entries live?"),
            says("a3", "Decided that cache entries expire after one day."),
        ],
    );
    let settings = CaptureSettings::default();

    // A live capture of an unseen session reads only the live segment.
    let live = run(true, &path, None, &settings, dir.path());
    assert_eq!(
        contents(&live),
        ["Decided that cache entries expire after one day."]
    );
    // A read position inside the file, at the first line after the
    // boundary, is honoured by a live capture, which reads from there.
    let text = std::fs::read_to_string(&path).expect("read the transcript");
    let mid = u64::try_from(text.lines().take(5).map(|l| l.len() + 1).sum::<usize>())
        .expect("a small offset");
    let live = run(true, &path, Some(mid), &settings, dir.path());
    assert_eq!(
        contents(&live),
        ["Decided that cache entries expire after one day."]
    );

    // A replay reads from the start, whatever read position the session
    // has (none, one inside the file, one past its end): each turn's final
    // text, with the prompt that started its turn.
    for offset in [None, Some(mid), Some(10_000)] {
        let o = run(false, &path, offset, &settings, dir.path());
        assert_eq!(o.skipped, None, "{o:?}");
        assert_eq!(o.source, Some(TextSource::Transcript));
        assert_eq!(
            contents(&o),
            [
                "Decided to use Redis for the session cache.",
                "Decided that cache entries expire after one day.",
            ],
            "{o:?}"
        );
        let contexts: Vec<&str> = o.candidates.iter().map(|c| c.context.as_str()).collect();
        assert_eq!(
            contexts,
            [
                "Which cache should the sessions use?",
                "And how long do entries live?"
            ]
        );
        for c in &o.candidates {
            assert_eq!(c.source_hook, HookEvent::PreCompact);
            assert_eq!(c.provenance.session_id.as_deref(), Some("s-1"));
            let trigger = c.provenance.extra.as_ref().map(|e| e["trigger"].clone());
            assert_eq!(trigger, Some(json!(REPLAY_TRIGGER)), "{:?}", c.provenance);
        }
        // The read position is never moved.
        assert_eq!(o.new_offset, None);
    }
}

#[test]
fn s16_b172_a_replay_caps_each_text_but_not_all_of_them_together() {
    let dir = temp_dir();
    let long = |what: &str| {
        let mut t = format!("Decided to keep the {what} queue bounded. ");
        while t.chars().count() < 110_000 {
            t.push_str("The queue keeps its order under load. ");
        }
        t
    };
    let (first, second) = (long("importer"), long("indexer"));
    let path = write(
        dir.path(),
        &[
            user("u1", "Bound the importer queue."),
            says("a1", &first),
            user("u2", "Bound the indexer queue."),
            says("a2", &second),
        ],
    );
    let settings = CaptureSettings::default();

    // A transcript read from the hook has one cap over all its texts.
    let live = run(true, &path, Some(0), &settings, dir.path());
    assert!(live.cap_hit, "{:?}", live.drops);

    // A replay has none: both texts are whole, though together they pass
    // the cap.
    let o = run(false, &path, None, &settings, dir.path());
    assert!(!o.cap_hit);
    assert_eq!(contents(&o), [first.trim_end(), second.trim_end()]);
    assert_eq!(
        o.extracted_chars,
        first.trim_end().chars().count() + second.trim_end().chars().count()
    );

    // One text longer than the cap is cut to it, with the marker; the
    // others stay whole.
    let mut huge = long("exporter");
    while huge.chars().count() <= MAX_CAPTURE_CHARS + 10_000 {
        huge.push_str("The queue keeps its order under load. ");
    }
    let path = write(
        dir.path(),
        &[
            user("u1", "Bound the importer queue."),
            says("a1", &first),
            user("u3", "Bound the exporter queue."),
            says("a3", &huge),
        ],
    );
    let o = run(false, &path, None, &settings, dir.path());
    assert!(o.cap_hit);
    let got = contents(&o);
    assert_eq!(got.len(), 2, "{:?}", o.drops);
    assert_eq!(got[0], first.trim_end());
    let cut: String = huge.chars().take(MAX_CAPTURE_CHARS).collect();
    assert_eq!(got[1], format!("{cut}{TRUNCATION_MARKER}"));
    assert_eq!(
        o.extracted_chars,
        first.trim_end().chars().count() + huge.trim_end().chars().count()
    );
}

#[test]
fn s16_b172_a_replay_follows_the_capture_settings() {
    let dir = temp_dir();
    let path = write(
        dir.path(),
        &[
            user("u1", "Pick a cache."),
            says("a1", "Decided to use Redis for the session cache."),
        ],
    );
    let off = CaptureSettings {
        enabled: false,
        ..CaptureSettings::default()
    };
    let o = run(false, &path, None, &off, dir.path());
    assert_eq!(o.skipped, Some(Skip::CaptureDisabled));
    assert!(o.candidates.is_empty());

    let mut opted_out = project(dir.path());
    opted_out.capture_enabled = false;
    let payload = json!({"session_id": "s-1", "transcript_path": path});
    let defaults = CaptureSettings::default();
    let input = CaptureInput {
        event: HookEvent::PreCompact,
        payload: &payload,
        settings: &defaults,
        project: &opted_out,
        commit_sha: None,
        transcript_offset: None,
    };
    let o = replay(&input, &mut SessionState::new());
    assert_eq!(o.skipped, Some(Skip::ProjectOptedOut));

    // A transcript that is not there gives nothing.
    let missing = dir.path().join("missing.jsonl");
    let o = run(
        false,
        missing.to_str().expect("a UTF-8 temporary path"),
        None,
        &defaults,
        dir.path(),
    );
    assert_eq!(o.skipped, Some(Skip::NoTranscript));
}
