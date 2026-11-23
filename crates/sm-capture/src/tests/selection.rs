//! B-179 (SPEC section 16, Candidate selection): what the transcript offers
//! the judge. Every text is synthetic.
//!
//! - the units a text is split into ([`split_units`]);
//! - the prose filter for tool text ([`prose`]);
//! - each kind of text [`read_texts`] gives, with its `Source:` line and
//!   the request of its turn;
//! - how an unseen session is read for a Stop, a PreCompact and a replay
//!   ([`ReadMode`]): never its history, never its summary;
//! - that a plainly stated decision without the old decision words reaches
//!   the judge (the owner's step 21).

use std::path::Path;

use serde_json::{Value, json};
use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;
use sm_core::types::Status;

use super::{project, source_of, temp_dir, tool_results_on, v1_tool_settings};
use crate::pipeline::{
    CaptureInput, CaptureOutcome, Skip, TextDrop, TextSource, WEAK_IMPORTANCE, capture,
};
use crate::select::{UNIT_MAX, UNIT_MIN, is_prose_line, prose, split_units};
use crate::session::SessionState;
use crate::signal::{Tier, classify, importance};
use crate::transcript::{
    PROMPT_SOURCE, ReadMode, SUMMARY_SOURCE, TextKind, TranscriptRead, read_texts, text_context,
};

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

fn call(id: &str, name: &str, input: Value) -> Value {
    json!({"type": "tool_use", "id": id, "name": name, "input": input})
}

fn result(id: &str, content: &str) -> Value {
    json!({"type": "tool_result", "tool_use_id": id, "content": content})
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

fn summary(text: &str) -> Value {
    let mut v = user("sum", text);
    v["isCompactSummary"] = json!(true);
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

/// The byte offset just after the first `n` lines of `text`.
fn after_lines(text: &str, n: usize) -> u64 {
    let end: usize = text.split_inclusive('\n').take(n).map(str::len).sum();
    u64::try_from(end).expect("a small offset")
}

fn texts(r: &TranscriptRead) -> Vec<&str> {
    r.entries.iter().map(|e| e.text.as_str()).collect()
}

fn kinds(r: &TranscriptRead) -> Vec<TextKind> {
    r.entries.iter().map(|e| e.kind).collect()
}

fn run(
    event: HookEvent,
    payload: &Value,
    settings: &CaptureSettings,
    root: &Path,
) -> CaptureOutcome {
    let project = project(root);
    capture(
        &CaptureInput {
            event,
            payload,
            settings,
            project: &project,
            commit_sha: None,
            transcript_offset: None,
            from_start: false,
        },
        &mut SessionState::new(),
    )
}

#[test]
fn s16_b179_a_text_is_split_into_units() {
    // A unit shorter than UNIT_MIN is not offered.
    assert!("Run the tests.".chars().count() < UNIT_MIN);
    assert!(split_units("Run the tests.").is_empty());
    assert!(split_units("  \n\n  ").is_empty());

    // Short paragraphs are packed into one unit, a blank line between them.
    let short =
        "The importer queue is bounded at one thousand.\n\nThe exporter queue has no bound yet.";
    assert_eq!(split_units(short), [short]);

    // A heading starts a new unit: a Markdown heading, or a short numbered
    // title ending in a colon.
    let headed = "The queues were measured on Tuesday.\n## Decisions\nThe importer queue is bounded.\n2. Root causes:\nThe exporter never drained its queue.";
    assert_eq!(
        split_units(headed),
        [
            "The queues were measured on Tuesday.",
            "## Decisions\nThe importer queue is bounded.",
            "2. Root causes:\nThe exporter never drained its queue.",
        ]
    );

    // Lines are packed up to UNIT_MAX, a newline between them; a unit ends
    // only when the next line would pass the limit.
    let many: Vec<String> = (0..60)
        .map(|i| format!("Line {i:03} of the log says the importer queue kept its order."))
        .collect();
    let text = many.join("\n");
    let units = split_units(&text);
    assert!(units.len() > 1, "{units:?}");
    assert_eq!(units.join("\n"), text);
    for pair in units.windows(2) {
        assert!(pair[0].chars().count() <= UNIT_MAX);
        let next = pair[1].split('\n').next().expect("a line");
        assert!(
            pair[0].chars().count() + 1 + next.chars().count() > UNIT_MAX,
            "a unit ended early"
        );
    }

    // A line longer than UNIT_MAX is cut at sentence ends.
    let line: String = (0..80)
        .map(|i| format!("Sentence {i:02} of the long line is here."))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(line.chars().count() > UNIT_MAX);
    let units = split_units(&line);
    assert!(units.len() > 1);
    assert!(units.iter().all(|u| u.chars().count() <= UNIT_MAX));
    assert!(units.iter().all(|u| u.ends_with("is here.")), "{units:?}");
    assert_eq!(units.join(" "), line);

    // A sentence longer than UNIT_MAX is cut at UNIT_MAX characters,
    // counted in characters, not bytes.
    let run_on = "é".repeat(2 * UNIT_MAX + 100);
    let units = split_units(&run_on);
    let lens: Vec<usize> = units.iter().map(|u| u.chars().count()).collect();
    assert_eq!(lens, [UNIT_MAX, UNIT_MAX, 100]);
    assert_eq!(units.concat(), run_on);
}

/// Two prose lines that together pass the paragraph minimum.
const P1: &str = "The session cache keeps each entry for one day because the upstream service limits our request rate.\nEntries older than a day are dropped on the next read and are never served to any caller.";

#[test]
fn s16_b179_only_the_prose_of_a_tool_text_is_offered() {
    assert!(is_prose_line(
        "The session cache keeps each entry for one day because upstream limits us."
    ));
    // A line number or a comment marker does not stop a line being prose.
    assert!(is_prose_line(
        "    12\u{2192}/// The session cache keeps each entry for one day because upstream limits us."
    ));
    assert!(is_prose_line(
        "src/cache.rs:12:// The session cache keeps each entry for one day because upstream limits us."
    ));
    // Code, tables, paths and short lines are not.
    for line in [
        "let entry = cache.get(&key).ok_or(Error::Missing)?;",
        "| session | 86400 | 3 | eu-west-1 | true | 2026-11-17 |",
        "C:/Users/someone/AppData/Local/spatial-memory/default/memories.db",
        "The cache keeps entries.",
        "1700000000 1700086400 86400 3 17 0.25 99.9 1024 2048",
    ] {
        assert!(!is_prose_line(line), "{line}");
    }

    let tool_text = format!(
        "# Notes\nfn main() {{ let x = 1; }}\n{P1}\nlet y = x + 1;\nA single prose line on its own is too short to be offered at all.\n"
    );
    assert_eq!(prose(&tool_text, 1_500), P1);
    // Raw command output has no prose.
    let raw = "running 12 tests\ntest cache::expiry ... ok\ntest cache::keys ... ok\ntest result: ok. 12 passed; 0 failed";
    assert_eq!(prose(raw, 1_500), "");

    // Paragraphs are taken in order until the next would pass the cap.
    let second = "A second paragraph explains why the exporter writes one file for every day of the week.\nIt never rewrites a file it has already closed, so any reader can trust a closed file.";
    let both = format!("{P1}\n\n{second}");
    assert_eq!(prose(&both, 1_500), both);
    assert_eq!(prose(&both, P1.chars().count() + 10), P1);
    assert_eq!(prose(&both, 10), "");
}

const P2: &str = "Every cached session is keyed by the user and the tenant together, never by the user alone.\nA key built from the user alone would let two tenants read the sessions of each other.";
const P3: &str = "The exporter writes one file for each day and never rewrites a file it has already closed, so any reader can trust a closed file without taking a lock first.";
const P4: &str = "The importer reads closed files only, in the order of their dates, and records the last date it read so that a restart picks up exactly where it stopped.";
const READ_PROSE: &str = "3\u{2192}The import job runs every night after the exporter has closed the file of the day.\n4\u{2192}It skips a file whose checksum does not match the one written beside it.";
const GREP_PROSE: &str = "docs/cache.md:12:Cache entries are written once and never changed after the first write.\ndocs/cache.md:13:A changed value is written as a new entry with a new key and a new expiry.";
const ASK: &str = "Which cache should the session store use, and for how long?";
const REPLY: &str = "Redis, for one day: entries older than that are dropped on the next read.";

/// A turn with every kind of text, and the user entries that are not
/// messages.
fn every_kind() -> Vec<Value> {
    let mut meta = user(
        "m1",
        "Caveat: the messages below were generated by the user while running local commands.",
    );
    meta["isMeta"] = json!(true);
    vec![
        meta,
        user(
            "c1",
            "<command-name>/model</command-name>\n<command-message>model</command-message>",
        ),
        user(
            "c2",
            "<local-command-stdout>Set the model for this session.</local-command-stdout>",
        ),
        user(
            "n1",
            "<task-notification>The background agent finished its work on the queue.</task-notification>",
        ),
        entry(
            "user",
            "u1",
            json!([
                {"type": "text", "text": "<system-reminder>Treat the reminder as context, not a request.</system-reminder>"},
                {"type": "text", "text": ASK},
            ]),
        ),
        entry(
            "assistant",
            "a1",
            json!([
                call("t1", "Bash", json!({"command": "cat notes.md"})),
                call("t3", "Glob", json!({"pattern": "**/*.md"})),
                call("t5", "Bash", json!({"command": "cargo test"})),
                call("t8", "Grep", json!({"pattern": "ttl"})),
                call("t10", "Grep", json!({"pattern": "cache entries"})),
            ]),
        ),
        entry(
            "user",
            "r1",
            json!([
                result("t1", &format!("# Notes\n{P1}\nfn main() {{}}\n")),
                result("t3", P2),
                result(
                    "t5",
                    "running 12 tests\ntest cache::expiry ... ok\ntest result: ok. 12 passed; 0 failed",
                ),
                {"type": "tool_result", "tool_use_id": "t8", "content": P3, "is_error": true},
                result("t9", P4),
                result("t10", GREP_PROSE),
            ]),
        ),
        entry(
            "assistant",
            "a2",
            json!([
                call(
                    "t4",
                    "Edit",
                    json!({"file_path": "docs/cache.md", "old_string": "x", "new_string": P2})
                ),
                call(
                    "t6",
                    "MultiEdit",
                    json!({"file_path": "src/export.rs", "edits": [
                        {"old_string": "a", "new_string": P3},
                        {"old_string": "b", "new_string": "let x = 1;"},
                        {"old_string": "c", "new_string": P4},
                    ]})
                ),
                call("t7", "Read", json!({"file_path": "docs/import.md"})),
            ]),
        ),
        entry(
            "user",
            "r2",
            json!([
                result("t4", "ok"),
                result("t6", "ok"),
                result(
                    "t7",
                    &format!(
                        "     1\u{2192}# Import\n     2\u{2192}\n     {}",
                        READ_PROSE.replace('\n', "\n     ")
                    )
                ),
            ]),
        ),
        says("a3", REPLY),
    ]
}

const BASH_SOURCE: &str = "Source: prose from the output of the command `cat notes.md`.";
const GREP_SOURCE: &str = "Source: prose from the results of a search for `cache entries`.";
const EDIT_SOURCE: &str = "Source: prose from the new text written to docs/cache.md.";
const MULTI_SOURCE: &str = "Source: prose from the new text written to src/export.rs.";
const READ_SOURCE: &str = "Source: prose from the file docs/import.md, as read.";

#[test]
fn s16_b179_each_kind_of_text_is_offered_with_its_source() {
    let dir = temp_dir();
    let body = lines(&every_kind());
    let path = write(dir.path(), "kinds.jsonl", &body);
    let read = read_texts(&path, None, false, ReadMode::Segment);
    use TextKind::{Prompt, Reply, ToolInput, ToolResult};
    assert_eq!(
        kinds(&read),
        [
            Prompt, ToolResult, ToolResult, ToolInput, ToolInput, ToolResult, Reply
        ],
        "{read:?}"
    );
    // The user's message without its reminder; a meta entry, a command, its
    // output and a task notification are not messages. Of the results, only
    // a Bash, Read or Grep call's that is not an error and has prose; a raw
    // command output, another tool's result and a result whose call is not
    // in the transcript are not offered.
    let multi = format!("{P3}\n\n{P4}");
    assert_eq!(
        texts(&read),
        [ASK, P1, GREP_PROSE, P2, multi.as_str(), READ_PROSE, REPLY]
    );
    let sources: Vec<&str> = read.entries.iter().map(|e| e.source.as_str()).collect();
    assert_eq!(
        sources,
        [
            PROMPT_SOURCE,
            BASH_SOURCE,
            GREP_SOURCE,
            EDIT_SOURCE,
            MULTI_SOURCE,
            READ_SOURCE,
            ""
        ]
    );
    // Every text but the prompt carries the request of its turn. The judge
    // reads a reply with its request, a prompt with its source, and any
    // other text with its source, then the request.
    for e in &read.entries[1..] {
        assert_eq!(e.prompt, ASK);
    }
    let contexts: Vec<String> = read.entries.iter().map(text_context).collect();
    assert_eq!(contexts[0], PROMPT_SOURCE);
    assert_eq!(contexts[1], format!("{BASH_SOURCE}\nRequest: {ASK}"));
    assert_eq!(contexts[6], ASK);
    let roles: Vec<&str> = read.entries.iter().map(|e| e.role.as_str()).collect();
    assert_eq!(
        roles,
        [
            "user",
            "user",
            "user",
            "assistant",
            "assistant",
            "user",
            "assistant"
        ]
    );
    assert_eq!(read.new_offset, body.len() as u64);

    // A seen session read from just after the first tool calls: their
    // results are not offered, and with the prompt before the offset the
    // other texts carry their source alone.
    let read = read_texts(&path, Some(after_lines(&body, 6)), true, ReadMode::Segment);
    assert_eq!(
        texts(&read),
        [P2, multi.as_str(), READ_PROSE, REPLY],
        "{read:?}"
    );
    assert_eq!(text_context(&read.entries[0]), EDIT_SOURCE);
    assert_eq!(text_context(&read.entries[3]), "");
}

#[test]
fn s16_b179_a_candidate_records_the_kind_of_its_text() {
    let dir = temp_dir();
    let path = write(dir.path(), "kinds.jsonl", &lines(&every_kind()));
    // Every kind, `tool_result` too, which is offered only with the switch
    // on (B-236).
    let o = run(
        HookEvent::PreCompact,
        &json!({"session_id": "s", "transcript_path": path}),
        &tool_results_on(),
        dir.path(),
    );
    let got: Vec<&str> = o.candidates.iter().map(source_of).collect();
    assert_eq!(
        got,
        [
            "prompt",
            "tool_result",
            "tool_result",
            "tool_input",
            "tool_input",
            "tool_result",
            "reply"
        ],
        "{o:?}"
    );
    assert!(o.drops.is_empty(), "{:?}", o.drops);
    let c = &o.candidates[1];
    assert_eq!(c.content, P1);
    assert_eq!(c.context, format!("{BASH_SOURCE}\nRequest: {ASK}"));
    for c in &o.candidates {
        assert_eq!(
            c.status,
            Status::Pending,
            "every transcript text is pending"
        );
    }
}

const Q1: &str = "What bounds the importer queue today?";
const R1: &str = "The importer queue is bounded at one thousand entries.";
const SUMMARY: &str = "This session is being continued from an earlier conversation. The importer queue is bounded at one thousand entries; the exporter queue is not bounded yet.";
const Q2: &str = "And what bounds the exporter queue now?";
const R2: &str = "The exporter queue is bounded at five hundred entries.";
const Q3: &str = "What about the indexer queue, then?";
const R3: &str = "The indexer queue has no bound yet, by design.";

fn compacted() -> Vec<Value> {
    vec![
        user("u1", Q1),
        says("a1", R1),
        compact(),
        summary(SUMMARY),
        user("u2", Q2),
        says("a2", R2),
        user("u3", Q3),
        says("a3", R3),
    ]
}

#[test]
fn s16_b179_an_unseen_session_never_gives_its_history_or_its_summary() {
    let dir = temp_dir();
    let body = lines(&compacted());
    let path = write(dir.path(), "compacted.jsonl", &body);
    let past_end = Some(body.len() as u64 + 100);

    // A Stop of an unseen session: its current turn only.
    for offset in [None, past_end] {
        let read = read_texts(&path, offset, false, ReadMode::Turn);
        assert_eq!(texts(&read), [Q3, R3], "{offset:?}");
        assert_eq!(read.new_offset, body.len() as u64);
        assert!(!read.from_start, "it skipped history (B-184)");
    }
    // A PreCompact of an unseen session: its live segment, without the
    // summary that opens it.
    for offset in [None, past_end] {
        let read = read_texts(&path, offset, false, ReadMode::Segment);
        assert_eq!(texts(&read), [Q2, R2, Q3, R3], "{offset:?}");
        assert!(!read.from_start, "it skipped history (B-184)");
    }
    // A replay: the whole transcript, the summary included, whatever the
    // offset.
    for offset in [None, Some(after_lines(&body, 5)), past_end] {
        let read = read_texts(&path, offset, false, ReadMode::Whole);
        assert_eq!(
            texts(&read),
            [Q1, R1, SUMMARY, Q2, R2, Q3, R3],
            "{offset:?}"
        );
        assert!(read.from_start, "a replay reads from the first line");
        let summary = &read.entries[2];
        assert_eq!(summary.kind, TextKind::Summary);
        assert_eq!(text_context(summary), SUMMARY_SOURCE);
        // The reply after the summary keeps the request of its own turn.
        assert_eq!(read.entries[4].prompt, Q2);
    }
    // A session seen from its first line gives every text after its
    // offset, the summary included, in every mode.
    let seen = Some(after_lines(&body, 2));
    for mode in [ReadMode::Turn, ReadMode::Segment] {
        let read = read_texts(&path, seen, true, mode);
        assert_eq!(texts(&read), [SUMMARY, Q2, R2, Q3, R3], "{mode:?}");
        assert!(read.from_start, "{mode:?}");
    }
    // A session capture joined part-way gives the same texts without the
    // summary, which restates history capture never read (B-184).
    for mode in [ReadMode::Turn, ReadMode::Segment] {
        let read = read_texts(&path, seen, false, mode);
        assert_eq!(texts(&read), [Q2, R2, Q3, R3], "{mode:?}");
        assert!(!read.from_start, "{mode:?}");
    }
}

#[test]
fn s16_b184_a_read_says_whether_it_began_at_the_first_line() {
    let dir = temp_dir();
    // One turn, read unseen: nothing was skipped.
    let one = lines(&[user("u1", Q1), says("a1", R1)]);
    let path = write(dir.path(), "one.jsonl", &one);
    for mode in [ReadMode::Turn, ReadMode::Segment, ReadMode::Whole] {
        let read = read_texts(&path, None, false, mode);
        assert_eq!(texts(&read), [Q1, R1], "{mode:?}");
        assert!(read.from_start, "{mode:?}");
    }
    // Two turns, read unseen: a Stop reads the last turn only and has
    // skipped the first; a PreCompact reads both.
    let two = lines(&[
        user("u1", Q1),
        says("a1", R1),
        user("u2", Q2),
        says("a2", R2),
    ]);
    let path = write(dir.path(), "two.jsonl", &two);
    let read = read_texts(&path, None, false, ReadMode::Turn);
    assert_eq!(texts(&read), [Q2, R2]);
    assert!(!read.from_start);
    let read = read_texts(&path, None, false, ReadMode::Segment);
    assert_eq!(texts(&read), [Q1, R1, Q2, R2]);
    assert!(read.from_start);
    // A seen read keeps what it was given.
    let seen = Some(after_lines(&two, 2));
    for from_start in [true, false] {
        let read = read_texts(&path, seen, from_start, ReadMode::Turn);
        assert_eq!(texts(&read), [Q2, R2]);
        assert_eq!(read.from_start, from_start);
    }
}

/// The owner's step 21: a decision stated plainly, with none of the old
/// decision words, and one that begins `Decision:`, which the old rules
/// missed.
const PLAIN: &str =
    "From now on the session cache lives in Redis, and the in-process map goes away.";
const LABELLED: &str = "Decision: the session cache lives in Redis for every region.";

#[test]
fn s16_b179_a_plainly_stated_decision_reaches_the_judge() {
    for text in [PLAIN, LABELLED] {
        let signal = classify(text);
        assert!(
            !signal.families.iter().any(|f| f == "decision"),
            "{text}: {signal:?}"
        );
        assert_eq!(signal.tier, Tier::Weak, "{text}");
        assert_eq!(
            importance(signal.tier, signal.score),
            None,
            "the pattern gate drops it"
        );
    }

    let dir = temp_dir();
    let defaults = CaptureSettings::default();

    // A Stop with the decision as its final message.
    let stop = run(
        HookEvent::Stop,
        &json!({"session_id": "s", "last_assistant_message": PLAIN}),
        &defaults,
        dir.path(),
    );
    assert!(stop.drops.is_empty(), "{:?}", stop.drops);
    assert_eq!(stop.candidates.len(), 1);
    let c = &stop.candidates[0];
    assert_eq!(c.content, PLAIN);
    assert_eq!(c.status, Status::Pending);
    assert_eq!(c.tier, Tier::Weak);
    assert_eq!(c.importance, WEAK_IMPORTANCE);

    // A Stop as Claude Code sends it, with the transcript: the decision is
    // read from the transcript's last turn (B-185).
    let turn = write(
        dir.path(),
        "turn.jsonl",
        &lines(&[
            user("u1", "Where should the session cache live from now on?"),
            says("a1", PLAIN),
        ]),
    );
    let stop = run(
        HookEvent::Stop,
        &json!({"session_id": "s", "transcript_path": turn, "last_assistant_message": PLAIN}),
        &defaults,
        dir.path(),
    );
    assert_eq!(stop.source, Some(TextSource::Transcript), "{stop:?}");
    let decision: Vec<_> = stop
        .candidates
        .iter()
        .filter(|c| c.content == PLAIN)
        .collect();
    assert_eq!(decision.len(), 1, "{stop:?}");
    assert_eq!(decision[0].status, Status::Pending);
    assert_eq!(
        decision[0].context,
        "Where should the session cache live from now on?"
    );

    // A PreCompact with the decision in the transcript, and a prompt that
    // states one too.
    let path = write(
        dir.path(),
        "decision.jsonl",
        &lines(&[
            user("u1", "Where should the session cache live from now on?"),
            says("a1", LABELLED),
            user("u2", PLAIN),
            says(
                "a2",
                "Understood, the in-process map is gone in the next change.",
            ),
        ]),
    );
    let pre = run(
        HookEvent::PreCompact,
        &json!({"session_id": "s", "transcript_path": path}),
        &defaults,
        dir.path(),
    );
    let contents: Vec<&str> = pre.candidates.iter().map(|c| c.content.as_str()).collect();
    assert!(contents.contains(&LABELLED), "{contents:?}");
    assert!(contents.contains(&PLAIN), "{contents:?}");
    assert!(pre.candidates.iter().all(|c| c.status == Status::Pending));

    // PostToolUse, opt-in only, keeps the pattern gate (B-160, B-179).
    let edit = run(
        HookEvent::PostToolUse,
        &json!({
            "session_id": "s",
            "tool_name": "Edit",
            "tool_input": {"old_string": "x", "new_string": PLAIN},
            "tool_response": "ok",
        }),
        &v1_tool_settings(),
        dir.path(),
    );
    assert!(edit.candidates.is_empty(), "{edit:?}");
    assert_eq!(edit.drops, [TextDrop::Weak]);
}

/// B-221: with `capture.tool_results` off, a read leaves out the text read
/// from files and command output, one drop each, and offers every other
/// text as before. The read position still moves past the whole read.
#[test]
fn s16_b221_tool_results_off_leaves_out_file_and_command_output() {
    let dir = temp_dir();
    let body = lines(&every_kind());
    let path = write(dir.path(), "kinds.jsonl", &body);
    let payload = json!({"session_id": "s", "transcript_path": path});
    let on = run(
        HookEvent::PreCompact,
        &payload,
        &tool_results_on(),
        dir.path(),
    );
    let settings = CaptureSettings {
        tool_results: false,
        ..CaptureSettings::default()
    };
    let off = run(HookEvent::PreCompact, &payload, &settings, dir.path());
    let got: Vec<&str> = off.candidates.iter().map(source_of).collect();
    assert_eq!(
        got,
        ["prompt", "tool_input", "tool_input", "reply"],
        "{off:?}"
    );
    assert_eq!(off.drops, [TextDrop::ToolResultsOff; 3]);
    let rest: Vec<_> = on
        .candidates
        .iter()
        .filter(|c| source_of(c) != "tool_result")
        .cloned()
        .collect();
    assert_eq!(off.candidates, rest);
    assert_eq!(off.new_offset, Some(body.len() as u64));
    assert_eq!(off.new_offset, on.new_offset);

    // A read that holds only tool calls and their output (the first tool
    // calls of the turn, which write nothing) gives no candidate, and one
    // drop for each output text.
    let only_output = lines(&every_kind()[5..7]);
    let path = write(dir.path(), "output.jsonl", &only_output);
    let payload = json!({"session_id": "s", "transcript_path": path});
    let off = run(HookEvent::PreCompact, &payload, &settings, dir.path());
    assert!(off.candidates.is_empty(), "{off:?}");
    assert_eq!(off.drops, [TextDrop::ToolResultsOff; 2], "{off:?}");
    assert_eq!(off.new_offset, Some(only_output.len() as u64));
}

/// B-236: `capture.tool_results` is off by default. With the default
/// settings a read leaves out the text read from files and command output,
/// one drop each, exactly as the switch turned off does, and PostToolUse
/// skips a tool that writes no file even when `capture.tools` names it.
#[test]
fn s16_b236_by_default_no_file_or_command_output_is_offered() {
    let dir = temp_dir();
    let body = lines(&every_kind());
    let path = write(dir.path(), "kinds.jsonl", &body);
    let payload = json!({"session_id": "s", "transcript_path": path});
    let defaults = run(
        HookEvent::PreCompact,
        &payload,
        &CaptureSettings::default(),
        dir.path(),
    );
    let off = CaptureSettings {
        tool_results: false,
        ..CaptureSettings::default()
    };
    assert_eq!(CaptureSettings::default(), off);
    assert_eq!(
        defaults,
        run(HookEvent::PreCompact, &payload, &off, dir.path())
    );
    assert_eq!(
        defaults.drops,
        [TextDrop::ToolResultsOff; 3],
        "{defaults:?}"
    );
    assert!(
        defaults
            .candidates
            .iter()
            .all(|c| source_of(c) != "tool_result"),
        "{defaults:?}"
    );
    let on = run(
        HookEvent::PreCompact,
        &payload,
        &tool_results_on(),
        dir.path(),
    );
    assert!(
        on.candidates.iter().any(|c| source_of(c) == "tool_result"),
        "{on:?}"
    );

    let bash_named = CaptureSettings {
        tools: vec!["Bash".to_owned()],
        ..CaptureSettings::default()
    };
    let bash = json!({
        "session_id": "s",
        "tool_name": "Bash",
        "tool_input": {"command": "make"},
        "tool_response": "Decided to use Redis for the session cache.",
    });
    let outcome = run(HookEvent::PostToolUse, &bash, &bash_named, dir.path());
    assert_eq!(
        outcome.skipped,
        Some(Skip::ToolResultsOff("Bash".to_owned())),
        "{outcome:?}"
    );
    assert!(outcome.candidates.is_empty(), "{outcome:?}");
}
