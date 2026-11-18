//! B-162 (RC1 and RC5 in docs/v3/CAPTURE-RCA.md): capture reads the final
//! text of each turn, and an unseen session only its live segment, after
//! its last compaction (SPEC section 16, Pipeline in the daemon, step 1).
//! The transcripts are synthetic.
//!
//! B-166: each text carries the user prompt of its turn, redacted and
//! bounded, as the context the judge reads with it.

use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};

use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;

use super::{project, temp_dir, v1_tool_settings};
use crate::extract::TRUNCATION_MARKER;
use crate::pipeline::{CaptureInput, capture};
use crate::session::SessionState;
use crate::transcript::{CONTEXT_CHARS, TranscriptRead, last_prompt, read_turns};

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

fn tool_result(uuid: &str) -> Value {
    entry(
        "user",
        uuid,
        json!([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]),
    )
}

fn says(uuid: &str, text: &str) -> Value {
    entry("assistant", uuid, json!([{"type": "text", "text": text}]))
}

fn tool_use() -> Value {
    json!({"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "cargo test"}})
}

/// Text and a tool call in one message.
fn says_then_calls(uuid: &str, text: &str) -> Value {
    entry(
        "assistant",
        uuid,
        json!([{"type": "text", "text": text}, tool_use()]),
    )
}

/// A tool call alone.
fn calls(uuid: &str) -> Value {
    entry("assistant", uuid, json!([tool_use()]))
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

fn sidechain(mut v: Value) -> Value {
    v["isSidechain"] = json!(true);
    v
}

fn lines(entries: &[Value]) -> String {
    entries.iter().map(|e| format!("{e}\n")).collect()
}

fn write(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).expect("write a transcript");
    p.to_str().expect("a UTF-8 temporary path").to_owned()
}

fn append(path: &str, text: &str) {
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open the transcript");
    f.write_all(text.as_bytes())
        .expect("append to the transcript");
}

fn texts(r: &TranscriptRead) -> Vec<&str> {
    r.entries.iter().map(|e| e.text.as_str()).collect()
}

#[test]
fn s16_b162_an_unseen_session_reads_only_its_live_segment() {
    let dir = temp_dir();
    let text = lines(&[
        user("u1", "Pick a cache."),
        says("a1", "Compacted away: the cache is keyed by branch."),
        compact(),
        user("u2", "This session continues an earlier conversation."),
        says("a2", "Live: the importer runs before the indexer."),
    ]);
    let path = write(dir.path(), "session.jsonl", &text);
    let size = text.len() as u64;

    let unseen = read_turns(&path, None);
    assert_eq!(
        texts(&unseen),
        ["Live: the importer runs before the indexer."]
    );
    assert_eq!(unseen.new_offset, size);
    // For a seen session the boundary only ends a turn: all after the
    // offset is new.
    assert_eq!(
        texts(&read_turns(&path, Some(0))),
        [
            "Compacted away: the cache is keyed by branch.",
            "Live: the importer runs before the indexer.",
        ]
    );
    // An offset past the end (the transcript was replaced) reads it as unseen.
    assert_eq!(texts(&read_turns(&path, Some(size + 1))), texts(&unseen));
    // Nothing is new at the end.
    let after = read_turns(&path, Some(size));
    assert!(after.entries.is_empty(), "{after:?}");
    assert_eq!(after.new_offset, size);
}

#[test]
fn s16_b162_a_turn_gives_its_final_text_once() {
    let dir = temp_dir();
    let text = lines(&[
        user("u1", "Why do the tests fail?"),
        says_then_calls("a1", "I will run the tests first."),
        tool_result("r1"),
        says("a2", "They fail because the fixture is stale."),
        says("a3", "Regenerating it fixes them."),
        user("u2", "Run them again."),
        says("a4", "Running them again now."),
        calls("a5"),
        tool_result("r2"),
    ]);
    let path = write(dir.path(), "session.jsonl", &text);
    let read = read_turns(&path, None);
    // The text before each tool call is dropped; the second turn ends in a
    // tool call and gives nothing.
    assert_eq!(
        texts(&read),
        ["They fail because the fixture is stale.\n\nRegenerating it fixes them."]
    );
    let e = &read.entries[0];
    assert_eq!(
        (e.uuid.as_str(), e.timestamp.as_str(), e.role.as_str()),
        ("a3", "t-a3", "assistant")
    );
}

#[test]
fn s16_b162_a_line_still_being_written_is_read_next_time() {
    let dir = temp_dir();
    let done = lines(&[
        user("u1", "Pick a cache."),
        says("a1", "The cache is keyed by branch."),
    ]);
    let next = says("a2", "Entries expire after a day.").to_string();
    let (head, tail) = next.split_at(next.len() / 2);
    let path = write(dir.path(), "session.jsonl", &format!("{done}{head}"));

    let first = read_turns(&path, None);
    assert_eq!(texts(&first), ["The cache is keyed by branch."]);
    assert_eq!(first.new_offset, done.len() as u64);
    append(&path, &format!("{tail}\n"));
    let second = read_turns(&path, Some(first.new_offset));
    assert_eq!(texts(&second), ["Entries expire after a day."]);
    assert_eq!(second.new_offset, (done.len() + next.len() + 1) as u64);

    // A last line that is whole JSON is read without its newline.
    let whole = format!("{done}{next}");
    let path = write(dir.path(), "whole.jsonl", &whole);
    let read = read_turns(&path, None);
    assert_eq!(
        texts(&read),
        ["The cache is keyed by branch.\n\nEntries expire after a day."]
    );
    assert_eq!(read.new_offset, whole.len() as u64);
}

#[test]
fn s16_b162_sidechain_entries_are_skipped() {
    let dir = temp_dir();
    let text = lines(&[
        user("u1", "Review the change."),
        says("a1", "The change keeps the old reader."),
        sidechain(user("s1", "A subagent's prompt.")),
        sidechain(says("s2", "A subagent's finding, not the main thread's.")),
        says("a2", "It also adds the new one."),
    ]);
    let path = write(dir.path(), "session.jsonl", &text);
    // The sidechain's prompt does not end the main turn either.
    assert_eq!(
        texts(&read_turns(&path, None)),
        ["The change keeps the old reader.\n\nIt also adds the new one."]
    );
}

#[test]
fn s16_b162_no_transcript_gives_nothing() {
    let dir = temp_dir();
    assert_eq!(
        read_turns("relative.jsonl", None),
        TranscriptRead::default()
    );
    let missing = dir.path().join("missing.jsonl");
    let missing = missing.to_str().expect("a UTF-8 temporary path");
    assert_eq!(read_turns(missing, Some(10)), TranscriptRead::default());
    let empty = write(dir.path(), "empty.jsonl", "");
    assert_eq!(read_turns(&empty, None), TranscriptRead::default());
}

fn prompts(r: &TranscriptRead) -> Vec<&str> {
    r.entries.iter().map(|e| e.prompt.as_str()).collect()
}

#[test]
fn s16_b166_a_turn_carries_the_prompt_that_started_it() {
    let dir = temp_dir();
    let mut meta = user("m1", "Caveat: the lines below came from a local command.");
    meta["isMeta"] = json!(true);
    let first = lines(&[user("u1", "Which queue should the importer use?")]);
    let text = first.clone()
        + &lines(&[
            says_then_calls("a1", "Let me look at the queue code."),
            tool_result("r1"),
            says(
                "a2",
                "The importer uses the bulk queue, so it never blocks a hook.",
            ),
            entry(
                "user",
                "u2",
                json!([{"type": "text", "text": "  And the indexer?  "}, {"type": "image", "source": {}}]),
            ),
            says("a3", "The indexer uses the same queue."),
            meta,
            says("a4", "Noted after the local command."),
        ]);
    let path = write(dir.path(), "session.jsonl", &text);

    let read = read_turns(&path, None);
    assert_eq!(
        texts(&read),
        [
            "The importer uses the bulk queue, so it never blocks a hook.",
            "The indexer uses the same queue.",
            "Noted after the local command.",
        ]
    );
    // A meta entry ends a turn but is not a prompt: the next turn keeps the
    // last real one.
    assert_eq!(
        prompts(&read),
        [
            "Which queue should the importer use?",
            "And the indexer?",
            "And the indexer?",
        ]
    );
    // A turn whose prompt came before the offset has none.
    let later = read_turns(&path, Some(first.len() as u64));
    assert_eq!(
        prompts(&later),
        ["", "And the indexer?", "And the indexer?"]
    );
}

#[test]
fn s16_b166_the_last_prompt_is_read_from_the_end_and_bounded() {
    let dir = temp_dir();
    let mut summary = user("s1", "This session continues an earlier conversation.");
    summary["isCompactSummary"] = json!(true);
    let long = "p".repeat(CONTEXT_CHARS + 500);
    let path = write(
        dir.path(),
        "session.jsonl",
        &lines(&[
            user("u1", "An earlier prompt."),
            says("a1", "An earlier answer."),
            user("u2", &long),
            says_then_calls("a2", "Working on it."),
            tool_result("r1"),
            sidechain(user("x1", "A subagent's prompt.")),
            summary,
            says("a3", "Done."),
        ]),
    );
    let prompt = last_prompt(&path);
    assert_eq!(
        prompt.chars().count(),
        CONTEXT_CHARS + TRUNCATION_MARKER.len()
    );
    assert!(prompt.starts_with("ppp") && prompt.ends_with(TRUNCATION_MARKER));
    // Bounded the same way per turn. The long prompt's own turn gives no
    // text (its text comes before a tool call, B-162); the compact summary
    // is no prompt, so the turn after it, "Done.", keeps the long prompt.
    let read = read_turns(&path, None);
    let per_turn = prompts(&read);
    assert_eq!(per_turn, ["An earlier prompt.", prompt.as_str()]);
    assert_eq!(read.entries[1].text, "Done.");
    // No transcript, or no prompt in it, gives nothing.
    assert_eq!(last_prompt("relative.jsonl"), "");
    let none = write(dir.path(), "none.jsonl", &lines(&[says("a1", "Hello.")]));
    assert_eq!(last_prompt(&none), "");
}

#[test]
fn s16_b166_a_candidate_carries_its_prompt_redacted() {
    let dir = temp_dir();
    let token = format!("ghp_{}", "A1b2C3d4".repeat(5));
    let ask = format!("Pick the session cache. The deploy token is {token}.");
    let path = write(
        dir.path(),
        "session.jsonl",
        &lines(&[
            user("u1", &ask),
            says("a1", "Decided to use Redis for the session cache."),
        ]),
    );
    let project = project(dir.path());
    let defaults = CaptureSettings::default();
    let run = |event: HookEvent, payload: Value, settings: &CaptureSettings| {
        capture(
            &CaptureInput {
                event,
                payload: &payload,
                settings,
                project: &project,
                commit_sha: None,
                transcript_offset: None,
            },
            &mut SessionState::new(),
        )
    };
    let from_transcript = run(
        HookEvent::PreCompact,
        json!({"session_id": "s", "transcript_path": path}),
        &defaults,
    );
    let from_message = run(
        HookEvent::Stop,
        json!({
            "session_id": "s",
            "transcript_path": path,
            "last_assistant_message": "Decided to use Redis for the session cache.",
        }),
        &defaults,
    );
    for outcome in [&from_transcript, &from_message] {
        assert_eq!(outcome.candidates.len(), 1, "{outcome:?}");
        let context = &outcome.candidates[0].context;
        assert!(context.starts_with("Pick the session cache."), "{context}");
        assert!(!context.contains(&token), "the token reached the context");
        assert!(!context.contains("ghp_"), "{context}");
    }
    // PostToolUse has no context.
    let edit = run(
        HookEvent::PostToolUse,
        json!({
            "session_id": "s",
            "tool_name": "Edit",
            "tool_input": {"file_path": "src/cache.rs", "old_string": "x",
                "new_string": "Decided to use Redis for the session cache."},
            "tool_response": "ok",
        }),
        &v1_tool_settings(),
    );
    assert_eq!(edit.candidates.len(), 1, "{edit:?}");
    assert_eq!(edit.candidates[0].context, "");
}
