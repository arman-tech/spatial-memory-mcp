//! B-172: `replay` runs a session's stored transcript through the pipeline
//! again, on request only (SPEC section 16, Replay). It reads the whole
//! transcript, before and after each compaction boundary, and never moves
//! the session's read position. The transcripts are synthetic.

use std::path::Path;

use serde_json::{Value, json};

use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;

use super::{project, source_of, temp_dir};
use crate::candidate::Candidate;
use crate::extract::{MAX_CAPTURE_CHARS, TRUNCATION_MARKER};
use crate::pipeline::{
    CaptureInput, CaptureOutcome, REPLAY_TRIGGER, Skip, TextSource, capture, replay,
};
use crate::select::split_units;
use crate::session::SessionState;
use crate::transcript::PROMPT_SOURCE;

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

/// The candidates from texts of one kind (B-179).
fn of_kind<'a>(o: &'a CaptureOutcome, kind: &str) -> Vec<&'a Candidate> {
    o.candidates
        .iter()
        .filter(|c| source_of(c) == kind)
        .collect()
}

fn texts<'a>(cs: &[&'a Candidate]) -> Vec<&'a str> {
    cs.iter().map(|c| c.content.as_str()).collect()
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

    // A live capture of an unseen session reads only the live segment: its
    // prompt and its reply (B-179).
    let live_segment = [
        "And how long do entries live?",
        "Decided that cache entries expire after one day.",
    ];
    let live = run(true, &path, None, &settings, dir.path());
    assert_eq!(contents(&live), live_segment);
    // A read position inside the file, at the first line after the
    // boundary, is honoured by a live capture, which reads from there.
    let text = std::fs::read_to_string(&path).expect("read the transcript");
    let mid = u64::try_from(text.lines().take(5).map(|l| l.len() + 1).sum::<usize>())
        .expect("a small offset");
    let live = run(true, &path, Some(mid), &settings, dir.path());
    assert_eq!(contents(&live), live_segment);

    // A replay reads from the start, whatever read position the session
    // has (none, one inside the file, one past its end): each prompt, and
    // each turn's final text with the prompt that started its turn.
    for offset in [None, Some(mid), Some(10_000)] {
        let o = run(false, &path, offset, &settings, dir.path());
        assert_eq!(o.skipped, None, "{o:?}");
        assert_eq!(o.source, Some(TextSource::Transcript));
        assert_eq!(
            contents(&o),
            [
                "Which cache should the sessions use?",
                "Decided to use Redis for the session cache.",
                "And how long do entries live?",
                "Decided that cache entries expire after one day.",
            ],
            "{o:?}"
        );
        let kinds: Vec<&str> = o.candidates.iter().map(source_of).collect();
        assert_eq!(kinds, ["prompt", "reply", "prompt", "reply"]);
        let contexts: Vec<&str> = o.candidates.iter().map(|c| c.context.as_str()).collect();
        assert_eq!(
            contexts,
            [
                PROMPT_SOURCE,
                "Which cache should the sessions use?",
                PROMPT_SOURCE,
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
    // Each sentence is distinct, so no unit is a duplicate of another.
    let sentences = |what: &str, chars: usize| {
        let mut t = format!("Decided to keep the {what} queue bounded. ");
        let mut count = t.chars().count();
        let mut i = 0;
        while count < chars {
            i += 1;
            let s = format!("Entry {i} of the {what} queue keeps its order under load. ");
            count += s.chars().count();
            t.push_str(&s);
        }
        t
    };
    let long = |what: &str| sentences(what, 110_000);
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
    // the cap. Each is split into units (B-179), which joined again are the
    // text.
    let o = run(false, &path, None, &settings, dir.path());
    assert!(!o.cap_hit);
    assert_eq!(
        texts(&of_kind(&o, "prompt")),
        ["Bound the importer queue.", "Bound the indexer queue."]
    );
    let expected: Vec<String> = split_units(first.trim_end())
        .into_iter()
        .chain(split_units(second.trim_end()))
        .collect();
    let replies = texts(&of_kind(&o, "reply"));
    assert_eq!(replies, expected);
    assert!(
        replies.join(" ") == format!("{} {}", first.trim_end(), second.trim_end()),
        "the units are the two texts"
    );
    assert!(o.candidates.iter().all(|c| !c.cap_hit));
    let prompts = "Bound the importer queue.".len() + "Bound the indexer queue.".len();
    assert_eq!(
        o.extracted_chars,
        prompts + first.trim_end().chars().count() + second.trim_end().chars().count()
    );

    // One text longer than the cap is cut to it, with the marker; the
    // others stay whole.
    let huge = sentences("exporter", MAX_CAPTURE_CHARS + 10_001);
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
    assert!(o.drops.is_empty(), "{:?}", o.drops);
    let cut: String = huge.chars().take(MAX_CAPTURE_CHARS).collect();
    let cut = format!("{cut}{TRUNCATION_MARKER}");
    let (whole, was_cut) = (split_units(first.trim_end()), split_units(&cut));
    let replies = of_kind(&o, "reply");
    assert_eq!(replies.len(), whole.len() + was_cut.len());
    let (a, b) = replies.split_at(whole.len());
    assert_eq!(texts(a), whole);
    assert!(a.iter().all(|c| !c.cap_hit), "the first text stays whole");
    assert_eq!(texts(b), was_cut);
    assert!(
        b.iter().all(|c| c.cap_hit),
        "each unit of the cut text records the cut"
    );
    assert!(
        was_cut.join(" ") == cut,
        "the units of the cut text are the text cut at the cap, with the marker"
    );
    let prompts = "Bound the importer queue.".len() + "Bound the exporter queue.".len();
    assert_eq!(
        o.extracted_chars,
        prompts + first.trim_end().chars().count() + huge.trim_end().chars().count()
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
