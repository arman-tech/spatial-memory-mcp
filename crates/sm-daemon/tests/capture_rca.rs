//! Reproductions of the root causes in `docs/v3/CAPTURE-RCA.md` (SPEC 16;
//! AT-13, AT-14, AT-16). Each test builds a synthetic Claude Code
//! transcript or hook payload, sends it to a daemon with the fake embedder
//! and the default settings, and asserts what automatic capture should
//! store. Every text here was written for these tests; none comes from a
//! real session.
//!
//! Captures are processed by one worker in the order they are queued, so a
//! sentinel capture queued last and seen stored ([`barrier`]) means every
//! earlier one has been processed.
//!
//! Without a judge every capture waits as pending (B-157), so the tests of
//! the mechanical fixes (RC1, RC2, RC3, RC5, RC8, RC9, M1) see what capture
//! hands the judge. RC4, RC6 and RC7 are decided by the judge (B-168): those
//! tests run the daemon with [`RecordedJudge`], which replays the real
//! judge's verdicts recorded in the spike, and wait with [`judged_barrier`]
//! until nothing is pending. They show what the daemon does with the model's
//! answers on these texts; the model itself is measured on the labelled
//! export (`CAPTURE-RCA.md` section 8.5).

mod common;

use std::path::Path;
use std::time::Duration;

use common::*;
use serde_json::{Value, json};
use sm_core::error::ErrorCode;
use sm_core::journal::metrics;
use sm_core::project::resolve_project;
use sm_core::protocol::{CaptureParams, ForgetParams, HookEvent, Op, ReplayParams, Reply, Request};
use sm_core::time::Timestamp;
use sm_core::types::{Memory, MemoryType, Status};
use sm_daemon::Exit;
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

fn sorted(texts: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = texts.iter().map(|t| (*t).to_owned()).collect();
    v.sort();
    v
}

fn capture_req(d: &Daemon, cwd: &str, session: &str, event: HookEvent, payload: Value) -> Request {
    d.req(
        cwd,
        Op::Capture(CaptureParams {
            event,
            payload,
            received_at: Timestamp::now(),
        }),
    )
    .with_session(Some(session.to_owned()))
}

/// Queues a capture and returns its request id.
fn queue(d: &Daemon, cwd: &str, session: &str, event: HookEvent, payload: Value) -> String {
    let req = capture_req(d, cwd, session, event, payload);
    match d.send(&req).into_result() {
        Ok(Reply::Captured(r)) => assert!(r.accepted, "the capture was accepted"),
        Ok(other) => panic!("capture replied {other:?}"),
        Err(e) => panic!("capture failed: {}: {}", e.code.as_str(), e.message),
    }
    req.id
}

/// A Stop capture of `message`, the final assistant text of a turn.
fn stop(d: &Daemon, cwd: &str, session: &str, message: &str) -> String {
    queue(
        d,
        cwd,
        session,
        HookEvent::Stop,
        json!({"session_id": session, "stop_hook_active": false, "last_assistant_message": message}),
    )
}

/// A PreCompact capture of the transcript at `path`.
fn pre_compact(d: &Daemon, cwd: &str, session: &str, path: &Path) -> String {
    queue(
        d,
        cwd,
        session,
        HookEvent::PreCompact,
        json!({
            "session_id": session,
            "transcript_path": path.to_str().expect("UTF-8"),
            "trigger": "auto",
        }),
    )
}

/// Waits until every capture queued before it has been processed (one
/// worker, first in first out). The sentinel is a decision with its reason
/// in a Stop message, which capture must store.
fn barrier(d: &Daemon, rig: &Rig, store: &Store) {
    let cwd = rig.project(&unique("barrier"));
    let session = unique("barrier");
    let text = format!(
        "We decided to use {} as the barrier marker, because the capture queue is first in first out.",
        unique("sentinel")
    );
    stop(d, &cwd, &session, &text);
    assert!(
        wait_until(WAIT, || !project_memories(store, &cwd).is_empty()),
        "the barrier capture was stored"
    );
}

/// The barrier for a daemon with a judge: a sentinel capture in its own
/// project is judged and made active, and no capture is still pending, so
/// every capture queued before it has been judged.
fn judged_barrier(d: &Daemon, rig: &Rig, store: &Store, judge: &RecordedJudge) {
    let cwd = rig.project(&unique("barrier"));
    stop(d, &cwd, &unique("barrier"), RECORDED_SENTINEL);
    let judged = wait_until(WAIT, || {
        project_memories(store, &cwd)
            .iter()
            .any(|m| m.status == Status::Active)
            && store.judge_queue_stats().is_ok_and(|s| s.pending == 0)
    });
    assert!(
        judged,
        "every capture was judged (not in the recording: {:?})",
        judge.misses()
    );
}

/// One transcript line with its newline.
fn line(v: Value) -> String {
    format!("{v}\n")
}

/// A user prompt: the start of a turn.
fn prompt(text: &str, ts: &str, n: usize) -> String {
    line(json!({
        "type": "user",
        "message": {"role": "user", "content": text},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("p{n}"),
    }))
}

/// An assistant text entry.
fn say(text: &str, ts: &str, n: usize) -> String {
    line(json!({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("a{n}"),
    }))
}

/// An assistant tool call and its result, as two entries.
fn tool_round_trip(name: &str, input: Value, ts: &str, n: usize) -> String {
    let call = line(json!({
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": format!("t{n}"), "name": name, "input": input}
        ]},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("c{n}"),
    }));
    let result = line(json!({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": format!("t{n}"), "content": "ok"}
        ]},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("r{n}"),
    }));
    call + &result
}

/// What Claude Code writes when it compacts: the boundary, then the summary
/// as a user entry.
fn compaction(ts: &str, n: usize) -> String {
    let boundary = line(json!({
        "type": "system",
        "subtype": "compact_boundary",
        "content": "Conversation compacted",
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("b{n}"),
        "compactMetadata": {"trigger": "auto", "preTokens": 150_000},
    }));
    let summary = line(json!({
        "type": "user",
        "isCompactSummary": true,
        "message": {"role": "user", "content": "This session is being continued from a previous conversation. Summary: the import and the cache were discussed."},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("s{n}"),
    }));
    boundary + &summary
}

fn append(path: &Path, text: &str) {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open the transcript");
    f.write_all(text.as_bytes())
        .expect("append to the transcript");
}

// Knowledge statements that capture must store, each in the form the
// labelling guide calls worth keeping.
const FIX_WITH_CAUSE: &str = "The import failed because the parser rejected byte order marks; the fix was to strip the mark before parsing.";
const DECISION_WITH_REASON: &str = "We decided to keep the cache per branch rather than per repository, because two branches share one store.";

// --- RC1 ------------------------------------------------------------------

/// RC1: a session the daemon has never seen (one open before setup, or a
/// resumed one under a new id) is read from its first byte, so text
/// compacted out of context long before the hook fired is captured. Only
/// the live segment, after the last compaction boundary, may be read.
#[test]
fn s16_rc1_an_unseen_session_does_not_capture_its_compacted_history() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc1");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    t += &compaction("2026-01-01T00:10:00Z", 3);
    t += &prompt("Where should the cache live?", "2026-01-01T00:11:00Z", 4);
    t += &say(DECISION_WITH_REASON, "2026-01-01T00:12:00Z", 5);
    append(&path, &t);

    pre_compact(&d, &cwd, &unique("rc1"), &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON]),
        "only the live segment is captured"
    );
}

// --- RC2 ------------------------------------------------------------------

/// RC2: the offset of a session lives only in the daemon's memory. After a
/// restart the next capture reads the session from the start again, and a
/// capture the user deleted in between is stored again.
#[test]
fn s16_rc2_a_restart_does_not_reread_a_session_or_restore_a_forgotten_capture() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc2");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("rc2");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    append(&path, &t);
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    let first = project_memories(&store, &cwd);
    assert_eq!(contents(&first), sorted(&[FIX_WITH_CAUSE]));

    match d.ok(
        &cwd,
        Op::Forget(ForgetParams {
            ids: vec![first[0].id],
            filter: None,
            confirm: false,
        }),
    ) {
        Reply::Forget(f) => assert_eq!(f.deleted, vec![first[0].id]),
        other => panic!("forget replied {other:?}"),
    }

    let exit = d.shutdown();
    assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
    let d = rig.start_fake();
    let mut t = prompt("Where should the cache live?", "2026-01-01T00:02:00Z", 3);
    t += &say(DECISION_WITH_REASON, "2026-01-01T00:03:00Z", 4);
    append(&path, &t);
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON]),
        "only the text appended since the last capture is read; the forgotten one stays forgotten"
    );
}

// --- RC3 ------------------------------------------------------------------

/// RC3: the extraction cap keeps the oldest text and drops the rest, while
/// the offset moves to the end of the file, so in a long history the newest
/// text is never read.
#[test]
fn s16_rc3_the_newest_text_is_read_when_the_history_exceeds_the_cap() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc3");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let filler = "alpha beta gamma delta epsilon ".repeat(200);
    let mut t = String::new();
    // 40 turns of about 6,200 characters each: more than the 200,000 cap.
    for n in 0..40 {
        t += &prompt(
            &format!("Continue with part {n}."),
            "2026-01-01T00:00:00Z",
            n,
        );
        t += &say(&format!("{filler}part {n}"), "2026-01-01T00:00:01Z", n);
    }
    let newest = "The build failed on Windows because the linker in the Git folder shadowed the Microsoft linker; the fix is to run cargo from PowerShell.";
    t += &prompt("Why does the build fail?", "2026-01-01T00:01:00Z", 100);
    t += &say(newest, "2026-01-01T00:02:00Z", 100);
    append(&path, &t);

    pre_compact(&d, &cwd, &unique("rc3"), &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[newest]),
        "the newest text is read and stored"
    );
}

// --- RC4 ------------------------------------------------------------------

/// RC4: a topic word anywhere in the text is the signal. Each text below is
/// progress narration, the final text of its turn, and matches a family
/// with a strong or medium score (testing and explicit save, important,
/// explicit save, definition). The rules are only a first filter; the
/// judge rejects these, and a rejected capture is deleted (B-168).
#[test]
fn s16_rc4_narration_with_topic_words_is_not_stored() {
    let judge = RecordedJudge::new();
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc4");
    let session = unique("rc4");
    for text in [
        "All 42 tests pass. Next I will wire the store into the daemon.",
        "Important: next I am going to run the full suite again.",
        "Done. I will note the change in the progress file and save the log.",
        "Now I will commit this. The next step is the replay command.",
    ] {
        stop(&d, &cwd, &session, text);
    }
    judged_barrier(&d, &rig, &store, &judge);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        Vec::<String>::new(),
        "narration is not stored"
    );
}

// --- RC5 ------------------------------------------------------------------

/// RC5: every assistant text is a candidate, including the text written
/// before a tool call, which announces the next action. The text before the
/// call here matches the decision family; the final text states the fix
/// and its cause.
#[test]
fn s16_rc5_text_before_a_tool_call_is_not_stored() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc5");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let before_call = "I decided to read the config loader first, because the error names it.";
    let conclusion = "The config was ignored because the loader read the user folder after the project folder; the fix was to read the project folder last so it wins.";
    let mut t = prompt("Why is the config ignored?", "2026-01-01T00:00:00Z", 1);
    t += &say(before_call, "2026-01-01T00:00:01Z", 2);
    t += &tool_round_trip(
        "Read",
        json!({"file_path": "src/config.rs"}),
        "2026-01-01T00:00:02Z",
        3,
    );
    t += &say(conclusion, "2026-01-01T00:00:03Z", 4);
    append(&path, &t);

    pre_compact(&d, &cwd, &unique("rc5"), &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[conclusion]),
        "only the turn's final text is a source"
    );
}

// --- RC6 ------------------------------------------------------------------

/// RC6: nothing gates a capture; a medium-tier topic match is stored as an
/// active memory and served at once. These match the api and dependency
/// families (0.6) and state nothing a later session can act on. The judge
/// is the gate: it rejects them, and they are deleted (B-168).
#[test]
fn s16_rc6_a_medium_topic_match_alone_is_not_stored() {
    let judge = RecordedJudge::new();
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc6");
    let session = unique("rc6");
    for text in [
        "The webhook endpoint is ready.",
        "The dependency is installed.",
    ] {
        stop(&d, &cwd, &session, text);
    }
    judged_barrier(&d, &rig, &store, &judge);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        Vec::<String>::new(),
        "a topic match alone is not stored"
    );
}

// --- RC7 ------------------------------------------------------------------

/// RC7: the type comes from the order of version 1's namespaces, not from
/// what the text states: anything that mentions tests passing or failing is
/// a pitfall, and a decision stated without "decided" or "chose" is a note.
/// The judge gives the type and quotes the text (B-168, B-173), so each
/// memory is found by the text its sentences were copied from.
#[test]
fn s16_rc7_the_type_follows_what_the_text_states() {
    let judge = RecordedJudge::new();
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc7");
    let session = unique("rc7");
    let incident = "The flaky clock test failed because the clock was not frozen; the fix was to inject a fixed clock.";
    let decision = "Note: we keep the cache per branch rather than per repository, because two branches share one store.";
    let test_status = "All 12 tests pass after the change.";
    for text in [incident, decision, test_status] {
        stop(&d, &cwd, &session, text);
    }
    judged_barrier(&d, &rig, &store, &judge);
    let memories = project_memories(&store, &cwd);
    assert!(
        memories.iter().all(|m| m.status == Status::Active),
        "nothing is left pending"
    );
    let type_of = |text: &str| {
        memories
            .iter()
            .find(|m| m.content.lines().all(|l| text.contains(l.trim())))
            .map(|m| m.memory_type)
    };
    assert_eq!(
        type_of(incident),
        Some(MemoryType::Incident),
        "a failure with its cause and fix is an incident"
    );
    assert_eq!(
        type_of(decision),
        Some(MemoryType::Decision),
        "a choice with its reason is a decision"
    );
    assert_ne!(
        type_of(test_status),
        Some(MemoryType::Pitfall),
        "a statement that tests pass is never a pitfall"
    );
}

// --- RC8 ------------------------------------------------------------------

/// RC8: with the default settings PostToolUse stores raw tool material: a
/// command's output, or the code just written. Both payloads match strong
/// or medium families (testing, important) and state no conclusion.
#[test]
fn s16_rc8_raw_tool_output_is_not_stored_with_the_default_settings() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc8");
    let session = unique("rc8");
    let file = Path::new(&cwd).join("src").join("reader.rs");
    queue(
        &d,
        &cwd,
        &session,
        HookEvent::PostToolUse,
        json!({
            "session_id": session,
            "tool_name": "Bash",
            "tool_input": {"command": "cargo test"},
            "tool_response": "running 42 tests\n42 tests passed, 0 failed\nnote: run with RUST_BACKTRACE=1 for a backtrace",
        }),
    );
    queue(
        &d,
        &cwd,
        &session,
        HookEvent::PostToolUse,
        json!({
            "session_id": session,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": file.to_str().expect("UTF-8"),
                "old_string": "let offset = 0;",
                "new_string": "// Note the offset before reading the next entry.\nlet offset = state.offset;",
            },
            "tool_response": "ok",
        }),
    );
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        Vec::<String>::new(),
        "raw tool material is not stored"
    );
}

// --- RC9 ------------------------------------------------------------------

/// RC9: a duplicate is found only by an exact hash of the text, so the same
/// statement in other words is stored twice: once across two captures, and
/// once within one transcript read.
#[test]
fn s16_rc9_a_restatement_in_other_words_is_not_stored_twice() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();

    let cwd = rig.project("rc9-later");
    let session = unique("rc9");
    stop(
        &d,
        &cwd,
        &session,
        "We decided to keep the cache per branch, because two branches share one store.",
    );
    barrier(&d, &rig, &store);
    d.wait_embedded(&cwd);
    stop(
        &d,
        &cwd,
        &session,
        "Because two branches share one store, we decided to keep the cache per branch.",
    );
    barrier(&d, &rig, &store);
    assert_eq!(
        project_memories(&store, &cwd).len(),
        1,
        "a later restatement is not stored"
    );

    let cwd = rig.project("rc9-same-read");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let mut t = prompt("Which runs first?", "2026-01-01T00:00:00Z", 1);
    t += &say(
        "We decided that the importer runs before the indexer, because the indexer reads the imported rows.",
        "2026-01-01T00:00:01Z",
        2,
    );
    t += &prompt("Say that again.", "2026-01-01T00:00:02Z", 3);
    t += &say(
        "Because the indexer reads the imported rows, we decided that the importer runs before the indexer.",
        "2026-01-01T00:00:03Z",
        4,
    );
    append(&path, &t);
    pre_compact(&d, &cwd, &unique("rc9"), &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        project_memories(&store, &cwd).len(),
        1,
        "a restatement in the same read is not stored"
    );
}

// --- M1 -------------------------------------------------------------------

/// M1: the capture size metric is journalled for every candidate, including
/// one the store drops as a duplicate, so it counts captures that were
/// never stored.
#[test]
fn s16_m1_the_size_metric_counts_only_stored_captures() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("m1");
    let text =
        "We decided to pin the driver to version 2, because version 3 drops the old protocol.";
    let first = stop(&d, &cwd, &unique("m1-a"), text);
    barrier(&d, &rig, &store);
    let second = stop(&d, &cwd, &unique("m1-b"), text);
    barrier(&d, &rig, &store);
    assert_eq!(project_memories(&store, &cwd).len(), 1, "stored once");
    let sized = |id: &str| {
        let mut n = 0;
        for e in std::fs::read_dir(rig.paths.metrics_dir())
            .expect("the metrics folder")
            .flatten()
        {
            let text = std::fs::read_to_string(e.path()).unwrap_or_default();
            n += text
                .lines()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .filter(|v| v["metric"] == metrics::CAPTURE_SIZE_TOKENS && v["request_id"] == id)
                .count();
        }
        n
    };
    assert_eq!(sized(&first), 1, "the stored capture is sized");
    assert_eq!(
        sized(&second),
        0,
        "the duplicate the store dropped is not counted as a capture"
    );
}

// --- Replay (B-172) ---------------------------------------------------------

fn replay_op(path: &Path, session: &str, cwd: &str) -> Op {
    Op::Replay(ReplayParams {
        transcript_path: path.to_str().expect("UTF-8").to_owned(),
        session_id: session.to_owned(),
        cwd: cwd.to_owned(),
        received_at: Timestamp::now(),
    })
}

/// A replay reads the session's whole transcript, history before the
/// compaction included, into the session's own project, whatever folder
/// asked. What it stores waits pending, queued for the judge; what capture
/// handled before is not stored again; and live capture of the session goes
/// on from its own read position.
#[test]
fn s16_b172_a_replay_stores_the_history_pending_and_leaves_live_capture_alone() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("replay");
    let elsewhere = rig.plain_dir("replay-asked-from");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("replay");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    t += &compaction("2026-01-01T00:10:00Z", 3);
    t += &prompt("Where should the cache live?", "2026-01-01T00:11:00Z", 4);
    t += &say(DECISION_WITH_REASON, "2026-01-01T00:12:00Z", 5);
    append(&path, &t);

    // Live capture of the unseen session reads only the live segment (RC1).
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON])
    );
    let offset = store.capture_offset(&session).expect("read the offset");
    assert_eq!(offset, Some(t.len() as u64));

    match d.ok(&elsewhere, replay_op(&path, &session, &cwd)) {
        Reply::Captured(r) => assert!(r.accepted, "the replay was queued"),
        other => panic!("replay replied {other:?}"),
    }
    barrier(&d, &rig, &store);
    let memories = project_memories(&store, &cwd);
    assert_eq!(
        contents(&memories),
        sorted(&[FIX_WITH_CAUSE, DECISION_WITH_REASON]),
        "the history is read, and the live capture is not stored twice"
    );
    assert!(
        project_memories(&store, &elsewhere).is_empty(),
        "the session's own folder names the project"
    );
    let queued: Vec<_> = store
        .judge_queue(1000, 10_000_000)
        .expect("read the judge queue")
        .into_iter()
        .map(|i| i.id)
        .collect();
    let replayed = memories
        .iter()
        .find(|m| m.content == FIX_WITH_CAUSE)
        .expect("the replayed capture");
    assert_eq!(
        replayed.status,
        Status::Pending,
        "never served before the judge"
    );
    assert!(queued.contains(&replayed.id), "queued for the judge");
    assert_eq!(
        replayed
            .provenance
            .extra
            .as_ref()
            .map(|e| e["trigger"].clone()),
        Some(json!("replay"))
    );
    assert_eq!(
        replayed.provenance.session_id.as_deref(),
        Some(session.as_str())
    );
    assert_eq!(
        store.capture_offset(&session).expect("read the offset"),
        offset,
        "a replay leaves the read position where it was"
    );

    // A second replay stores nothing new.
    d.ok(&elsewhere, replay_op(&path, &session, &cwd));
    barrier(&d, &rig, &store);
    assert_eq!(project_memories(&store, &cwd).len(), 2);

    // Live capture goes on from its own read position.
    let more = prompt("And the indexer?", "2026-01-01T00:13:00Z", 6)
        + &say(
            "We decided that the indexer runs after the importer, because it reads the imported rows.",
            "2026-01-01T00:14:00Z",
            7,
        );
    append(&path, &more);
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    assert_eq!(project_memories(&store, &cwd).len(), 3);
    assert_eq!(
        store.capture_offset(&session).expect("read the offset"),
        Some((t.len() + more.len()) as u64)
    );
    assert!(
        wait_until(WAIT, || d.status(&cwd).captures_queued == 0),
        "the status queue count returns to zero once the worker is done"
    );
}

/// A replay with capture off is not queued, and its paths must be absolute.
#[test]
fn s16_b172_a_replay_needs_capture_on_and_absolute_paths() {
    let mut rig = Rig::new();
    rig.settings.capture.enabled = false;
    let d = rig.start_fake();
    let cwd = rig.project("replay-off");
    let path = Path::new(&cwd).join("transcript.jsonl");
    append(
        &path,
        &(prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1)
            + &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2)),
    );
    match d.ok(&cwd, replay_op(&path, "s-off", &cwd)) {
        Reply::Captured(r) => assert!(!r.accepted, "not queued with capture off"),
        other => panic!("replay replied {other:?}"),
    }
    for op in [
        replay_op(Path::new("transcript.jsonl"), "s-off", &cwd),
        replay_op(&path, "s-off", "work/replay-off"),
    ] {
        let e = d
            .call(&cwd, op)
            .into_result()
            .expect_err("a relative path is refused");
        assert_eq!(e.code, ErrorCode::InvalidInput, "{}", e.message);
        assert!(
            e.message.contains("needs an absolute path"),
            "{}",
            e.message
        );
    }
    assert!(rig.memories().is_empty());
}
