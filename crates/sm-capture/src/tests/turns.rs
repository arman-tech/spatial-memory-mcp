//! B-162 (RC1 and RC5 in docs/v3/CAPTURE-RCA.md): capture reads the final
//! text of each turn, and an unseen session only its live segment, after
//! its last compaction (SPEC section 16, Pipeline in the daemon, step 1).
//! The transcripts are synthetic.

use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};

use super::temp_dir;
use crate::transcript::{TranscriptRead, read_turns};

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
