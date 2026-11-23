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
use sm_core::config::JudgeTask;
use sm_core::error::ErrorCode;
use sm_core::journal::metrics;
use sm_core::project::resolve_project;
use sm_core::protocol::{CaptureParams, ForgetParams, HookEvent, Op, ReplayParams, Reply, Request};
use sm_core::time::Timestamp;
use sm_core::traits::ChunkPiece;
use sm_core::types::{Memory, MemoryType, NewMemory, Provenance, ProvenanceSource, Status};
use sm_daemon::Exit;
use sm_store::{Access, CaptureProgress, MemoryInput, MemoryListFilter, NewChunk, Store};

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

/// The memories of `cwd` that are served: every one not pending.
fn served(store: &Store, cwd: &str) -> Vec<Memory> {
    project_memories(store, cwd)
        .into_iter()
        .filter(|m| m.status != Status::Pending)
        .collect()
}

/// Asserts that every memory of `cwd` not served was rejected by the judge
/// and is kept, unserved, for 30 days (B-190): the store's only rejected
/// captures are these `n`, and none waits for the judge.
#[track_caller]
fn assert_rejected(store: &Store, cwd: &str, n: u64) {
    let unserved = project_memories(store, cwd)
        .into_iter()
        .filter(|m| m.status == Status::Pending)
        .count();
    let stats = store.judge_queue_stats().expect("queue stats");
    assert_eq!(
        (unserved as u64, stats.rejected, stats.pending),
        (n, n, 0),
        "{stats:?}"
    );
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
/// The summary [`compaction`] writes.
const COMPACTION_SUMMARY: &str = "This session is being continued from a previous conversation. Summary: the import and the cache were discussed.";

fn compaction(ts: &str, n: usize) -> String {
    compaction_with(ts, n, COMPACTION_SUMMARY)
}

/// [`compaction`] with `summary` as the summary.
fn compaction_with(ts: &str, n: usize, summary: &str) -> String {
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
        "message": {"role": "user", "content": summary},
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
    // Since B-179 a prompt is a text of its own, so the live segment gives
    // its prompt and its reply; the history before the boundary and the
    // summary give nothing.
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON, "Where should the cache live?"]),
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
    // The prompt is a text of its own since B-179.
    assert_eq!(
        contents(&first),
        sorted(&[FIX_WITH_CAUSE, "Why does the import fail?"])
    );
    let fix = first
        .iter()
        .find(|m| m.content == FIX_WITH_CAUSE)
        .expect("the fix")
        .id;

    match d.ok(
        &cwd,
        Op::Forget(ForgetParams {
            ids: vec![fix],
            filter: None,
            confirm: false,
        }),
    ) {
        // The fix waits for the judge, so the reply counts it and never
        // names it (B-181).
        Reply::Forget(f) => assert_eq!((f.deleted, f.pending), (Vec::new(), 1)),
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
        sorted(&[
            "Why does the import fail?",
            DECISION_WITH_REASON,
            "Where should the cache live?"
        ]),
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
    // Since B-179 the gate is gone and prompts are texts of their own, so
    // the filler and the prompts within the cap are stored pending too, for
    // the judge. The cap counts every text from the newest back (B-161):
    // count here the turns it reads whole (B-185).
    let cap = sm_capture::extract::MAX_CAPTURE_CHARS;
    let ask = |n: usize| format!("Continue with part {n}.");
    let reply = |n: usize| format!("{filler}part {n}");
    let mut total = newest.len() + "Why does the build fail?".len();
    let mut whole_from = 40;
    while whole_from > 0 {
        let n = whole_from - 1;
        let turn = reply(n).len() + ask(n).len();
        if total + turn > cap {
            break;
        }
        total += turn;
        whole_from = n;
    }
    assert!(
        (1..40).contains(&whole_from) && total + reply(whole_from - 1).len() > cap,
        "the cap falls inside a reply: {whole_from} {total}"
    );
    let stored = contents(&project_memories(&store, &cwd));
    let has = |t: &str| stored.iter().any(|c| c == t);
    assert!(has(newest), "the newest text is read and stored");
    assert!(has("Why does the build fail?"), "{stored:?}");
    for n in 0..40 {
        assert_eq!(
            has(&ask(n)),
            n >= whole_from,
            "prompt {n} is read if and only if its turn is within the cap: {stored:?}"
        );
    }
    assert!(
        stored.iter().all(|c| c == newest
            || c == "Why does the build fail?"
            || (c.starts_with("Continue with part ") && (whole_from..40).any(|n| *c == ask(n)))
            || c.starts_with("alpha beta gamma")),
        "nothing else is stored: {stored:?}"
    );
}

// --- RC4 ------------------------------------------------------------------

/// RC4: a topic word anywhere in the text is the signal. Each text below is
/// progress narration, the final text of its turn, and matches a family
/// with a strong or medium score (testing and explicit save, important,
/// explicit save, definition). The rules are only a first filter; the
/// judge rejects these (B-168), and a rejected capture is kept 30 days,
/// never served (B-190).
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
        contents(&served(&store, &cwd)),
        Vec::<String>::new(),
        "narration is not stored"
    );
    assert_rejected(&store, &cwd, 4);
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
    // Since B-179 the prompt is a text of its own; of the assistant's
    // texts, only the turn's final one is offered.
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[conclusion, "Why is the config ignored?"]),
        "only the turn's final text is a source"
    );
}

// --- RC6 ------------------------------------------------------------------

/// RC6: nothing gates a capture; a medium-tier topic match is stored as an
/// active memory and served at once. These match the api and dependency
/// families (0.6) and state nothing a later session can act on. The judge
/// is the gate: it rejects them (B-168), and they are kept 30 days, never
/// served (B-190).
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
        contents(&served(&store, &cwd)),
        Vec::<String>::new(),
        "a topic match alone is not stored"
    );
    assert_rejected(&store, &cwd, 2);
}

// --- RC7 ------------------------------------------------------------------

/// RC7: the type comes from the order of version 1's namespaces, not from
/// what the text states: anything that mentions tests passing or failing is
/// a pitfall, and a decision stated without "decided" or "chose" is a note.
/// The judge gives the type and marks spans of the text (B-168, B-191), so each
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
    let memories = served(&store, &cwd);
    assert!(
        memories.iter().all(|m| m.status == Status::Active),
        "every memory served is active"
    );
    // Nothing waits for the judge; what it did not keep it rejected (B-190).
    let rejected = project_memories(&store, &cwd).len() - memories.len();
    assert_rejected(&store, &cwd, rejected as u64);
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

// --- RC7 under the statements task -----------------------------------------
//
// Under `task = "statements"` (B-240) the judge names a type for each
// statement it keeps. Nothing of the statements call was recorded, so the
// judge here is scripted: the test shows what the daemon does with the
// types it is given, not which types the model gives. RC4 and RC6 have no
// statements version: there the only decision is the model's (whether a
// text holds a statement at all), so a scripted judge would make such a test
// assert its own script (B-256).

/// RC7 under the statements task (B-240, B-242): each kept statement takes
/// the type the judge gives it, not the type the signal rules set on the
/// pending capture from version 1's namespace order. Before the judge
/// answers, the capture that mentions a test failing and the one that
/// mentions tests passing are pitfalls by that order, and the decision
/// stated without "decided" or "chose" is a note. Afterwards none of the four
/// statements is of those types: the first statement of a capture takes its
/// row, the second becomes a row of its own, and the status line no
/// statement holds is not stored.
#[test]
fn s16_rc7_under_statements_each_statement_takes_the_type_the_judge_gives_it() {
    let incident = "The flaky clock test failed because the clock was not frozen; the fix was to inject a fixed clock.";
    let decision = "Note: we keep the cache per branch rather than per repository, because two branches share one store.";
    let status_line = "All 12 tests pass after the change.";
    let second_incident = "The importer failed on Monday because its token had expired; the fix was to renew the token before each run.";
    let second_decision =
        "We keep one writer per store rather than a pool, because SQLite locks the whole file.";
    let both = format!("{status_line} {second_incident} {second_decision}");
    let marks: [((&str, &str), &str); 4] = [
        (
            ("The flaky clock test failed", "inject a fixed clock."),
            "incident",
        ),
        (
            ("Note: we keep the cache", "two branches share one store."),
            "decision",
        ),
        (
            (
                "The importer failed on Monday",
                "the token before each run.",
            ),
            "incident",
        ),
        (
            (
                "We keep one writer per store",
                "SQLite locks the whole file.",
            ),
            "decision",
        ),
    ];
    let judge = ScriptedJudge::for_statements(Box::new(move |batch| {
        units_answer(batch, |c| {
            marks
                .iter()
                .filter(|(span, _)| c.text.contains(span.0))
                .map(|(span, ty)| statement(*span, "keep", ty, "must_keep"))
                .collect()
        })
    }));
    let mut rig = Rig::new();
    rig.settings.capture.judge.task = JudgeTask::Statements;
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rc7-statements");
    let session = unique("rc7-statements");

    judge.hold();
    for text in [incident, decision, both.as_str()] {
        stop(&d, &cwd, &session, text);
    }
    assert!(
        wait_until(WAIT, || judge.in_call()
            && project_memories(&store, &cwd).len() == 3),
        "the three captures wait for the judge"
    );
    let starting = |text: &str| {
        project_memories(&store, &cwd)
            .into_iter()
            .find(|m| m.content == text)
            .map(|m| (m.status, m.memory_type))
    };
    assert_eq!(
        starting(incident),
        Some((Status::Pending, MemoryType::Pitfall)),
        "the signal rules make a failing test a pitfall"
    );
    assert_eq!(
        starting(decision),
        Some((Status::Pending, MemoryType::Note)),
        "the signal rules make a decision without \"decided\" a note"
    );
    assert_eq!(
        starting(both.as_str()),
        Some((Status::Pending, MemoryType::Pitfall)),
        "the signal rules make a text that mentions tests passing a pitfall"
    );
    judge.release();

    assert!(
        wait_until(WAIT, || served(&store, &cwd).len() == 4
            && store.judge_queue_stats().is_ok_and(|s| s.pending == 0)),
        "four statements are stored: {:?}",
        contents(&project_memories(&store, &cwd))
    );
    let memories = served(&store, &cwd);
    assert_eq!(
        contents(&memories),
        sorted(&[incident, decision, second_incident, second_decision]),
        "each statement is stored in its own words, and nothing else"
    );
    assert!(
        memories.iter().all(|m| m.status == Status::Active),
        "every statement served is active"
    );
    let type_of = |text: &str| {
        memories
            .iter()
            .find(|m| m.content == text)
            .map(|m| m.memory_type)
    };
    for (text, want) in [
        (incident, MemoryType::Incident),
        (decision, MemoryType::Decision),
        (second_incident, MemoryType::Incident),
        (second_decision, MemoryType::Decision),
    ] {
        assert_eq!(type_of(text), Some(want), "{text}");
    }
    assert_eq!(
        project_memories(&store, &cwd).len(),
        4,
        "no capture is left unserved"
    );
    assert!(
        judge.batches().is_empty(),
        "the verdicts task was not called"
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

    // Live capture of the unseen session reads only the live segment (RC1),
    // without its summary; since B-179 its prompt is a text of its own.
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON, "Where should the cache live?"])
    );
    let offset = store.capture_offset(&session).expect("read the offset");
    assert_eq!(offset, Some(t.len() as u64));

    match d.ok(&elsewhere, replay_op(&path, &session, &cwd)) {
        Reply::Captured(r) => assert!(r.accepted, "the replay was queued"),
        other => panic!("replay replied {other:?}"),
    }
    barrier(&d, &rig, &store);
    let memories = project_memories(&store, &cwd);
    // A replay reads the whole transcript (B-179): the prompts and the
    // compaction summary are texts of their own.
    assert_eq!(
        contents(&memories),
        sorted(&[
            FIX_WITH_CAUSE,
            DECISION_WITH_REASON,
            "Why does the import fail?",
            "Where should the cache live?",
            COMPACTION_SUMMARY,
        ]),
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
    assert_eq!(project_memories(&store, &cwd).len(), 5);

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
    assert_eq!(
        project_memories(&store, &cwd).len(),
        6,
        "the new reply; the 16-character prompt is under the unit minimum"
    );
    assert_eq!(
        store.capture_offset(&session).expect("read the offset"),
        Some((t.len() + more.len()) as u64)
    );
    assert!(
        wait_until(WAIT, || d.status(&cwd).captures_queued == 0),
        "the status queue count returns to zero once the worker is done"
    );
}

/// B-176: a capture the rules made before the judge waits pending with no
/// row in the judge's queue, and its hash may already be seen (a store from
/// before the judge, or a replay made before B-176). A replay that reads
/// the same text replaces it with a capture queued for the judge; a capture
/// already in the queue is not replaced.
#[test]
fn s16_b176_a_replay_replaces_a_seen_old_rule_capture() {
    let rig = Rig::new();
    let cwd = rig.project("old-rule");
    let old = {
        let store = rig.store();
        let identity = resolve_project(Path::new(&cwd)).expect("a project");
        let project = store.upsert_project(&identity).expect("the project");
        let mut memory = NewMemory::new(
            project.id,
            FIX_WITH_CAUSE.to_owned(),
            Provenance::new(ProvenanceSource::Capture),
        );
        memory.status = Status::Pending;
        let piece = ChunkPiece {
            seq: 0,
            text: FIX_WITH_CAUSE.to_owned(),
            token_count: 20,
        };
        let id = store
            .insert_memory(
                MemoryInput {
                    memory,
                    chunks: vec![NewChunk::pending(piece)],
                },
                Access::Local,
            )
            .expect("an old-rule capture")
            .id;
        store
            .record_capture_progress(&CaptureProgress {
                session: None,
                project: project.id,
                hashes: &[sm_core::hash::content_hash(FIX_WITH_CAUSE)],
                at: Timestamp::now(),
            })
            .expect("its hash seen");
        id
    };
    let d = rig.start_fake();
    let store = rig.store();
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("old-rule");
    append(
        &path,
        &(prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1)
            + &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2)),
    );
    let queued = |store: &Store| -> Vec<_> {
        store
            .judge_queue(1000, 10_000_000)
            .expect("read the judge queue")
            .into_iter()
            .map(|i| i.id)
            .collect()
    };
    assert!(queued(&store).is_empty());

    d.ok(&cwd, replay_op(&path, &session, &cwd));
    barrier(&d, &rig, &store);
    let memories = project_memories(&store, &cwd);
    // Since B-179 the prompt is a text of its own.
    assert_eq!(
        contents(&memories),
        sorted(&[FIX_WITH_CAUSE, "Why does the import fail?"]),
        "stored once"
    );
    let fix = memories
        .iter()
        .find(|m| m.content == FIX_WITH_CAUSE)
        .expect("the fix");
    let new = fix.id;
    assert_ne!(new, old, "the old-rule capture is replaced");
    assert_eq!(fix.status, Status::Pending);
    assert!(queued(&store).contains(&new), "queued for the judge");
    let ids = |store: &Store| {
        let mut ids: Vec<_> = project_memories(store, &cwd).iter().map(|m| m.id).collect();
        ids.sort();
        ids
    };
    let before = ids(&store);

    d.ok(&cwd, replay_op(&path, &session, &cwd));
    barrier(&d, &rig, &store);
    assert_eq!(
        ids(&store),
        before,
        "a capture in the judge's queue is not replaced"
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

// --- B-184 ----------------------------------------------------------------

/// A Stop that reads the transcript at `path`, with `message` as the
/// payload's last assistant message.
fn stop_reading(d: &Daemon, cwd: &str, session: &str, path: &Path, message: &str) -> String {
    queue(
        d,
        cwd,
        session,
        HookEvent::Stop,
        json!({
            "session_id": session,
            "stop_hook_active": false,
            "transcript_path": path.to_str().expect("UTF-8"),
            "last_assistant_message": message,
        }),
    )
}

const LINKER_FINDING: &str = "The build uses the Microsoft linker from PowerShell, because the linker in the Git folder shadows it.";

/// B-184 (reviewer, step 17): capture joins a session part-way (one open
/// before setup, or before the daemon's state for it was pruned), so its
/// first Stop gives only the current turn and the earlier turns are never
/// read. A compaction summary written later restates that history. It is
/// not offered, then or after a restart, because capture did not read the
/// session from its start.
#[test]
fn s16_b184_a_session_joined_part_way_never_offers_its_summary() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("b184-joined");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("b184-joined");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    t += &prompt("Where should the cache live?", "2026-01-01T00:02:00Z", 3);
    t += &say(DECISION_WITH_REASON, "2026-01-01T00:03:00Z", 4);
    append(&path, &t);
    stop_reading(&d, &cwd, &session, &path, DECISION_WITH_REASON);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[DECISION_WITH_REASON, "Where should the cache live?"]),
        "an unseen session's Stop gives only its current turn"
    );

    // The daemon restarts: what it knows of the session comes from the
    // store.
    let exit = d.shutdown();
    assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
    let d = rig.start_fake();
    let mut more = compaction("2026-01-01T00:10:00Z", 5);
    more += &prompt(
        "Which linker does the build use?",
        "2026-01-01T00:11:00Z",
        6,
    );
    more += &say(LINKER_FINDING, "2026-01-01T00:12:00Z", 7);
    append(&path, &more);
    stop_reading(&d, &cwd, &session, &path, LINKER_FINDING);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[
            DECISION_WITH_REASON,
            "Where should the cache live?",
            LINKER_FINDING,
            "Which linker does the build use?",
        ]),
        "the summary of history capture never read is not offered"
    );
    assert_eq!(
        store.capture_offset(&session).expect("read the offset"),
        Some((t.len() + more.len()) as u64)
    );
}

/// B-184: a session whose first read began at its first line was read whole,
/// so a compaction summary written later restates only what capture has
/// read, and is offered as before (B-179).
#[test]
fn s16_b184_a_session_seen_from_its_start_offers_its_summary() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("b184-whole");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("b184-whole");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    append(&path, &t);
    stop_reading(&d, &cwd, &session, &path, FIX_WITH_CAUSE);
    barrier(&d, &rig, &store);

    let exit = d.shutdown();
    assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
    let d = rig.start_fake();
    let mut more = compaction("2026-01-01T00:10:00Z", 3);
    more += &prompt(
        "Which linker does the build use?",
        "2026-01-01T00:11:00Z",
        4,
    );
    more += &say(LINKER_FINDING, "2026-01-01T00:12:00Z", 5);
    append(&path, &more);
    stop_reading(&d, &cwd, &session, &path, LINKER_FINDING);
    barrier(&d, &rig, &store);
    assert_eq!(
        contents(&project_memories(&store, &cwd)),
        sorted(&[
            FIX_WITH_CAUSE,
            "Why does the import fail?",
            COMPACTION_SUMMARY,
            LINKER_FINDING,
            "Which linker does the build use?",
        ]),
        "the summary of a session read from its start is offered"
    );
}

// --- B-189 ----------------------------------------------------------------

const EXPORT_REQUEST: &str = "## 1. Primary Request and Intent\nThe owner asked for the export \
folder to move to the data drive. The weekly job writes the archive there. Archives older than \
a year are deleted.";

const EXPORT_PENDING: &str =
    "## 7. Pending Tasks\n- Rename the weekly job to weekly-archive and update its schedule.";

/// The first request again, with one sentence added: three of its four
/// sentence keys were offered before; its heading has no key (B-196).
const EXPORT_REQUEST_AGAIN: &str = "## 1. Primary Request and Intent\nThe owner asked for the \
export folder to move to the data drive. The weekly job writes the archive there. Archives older \
than a year are deleted. It runs every Sunday at noon.";

/// B-189: a compaction summary unit in a section that is not offered, and
/// one that repeats a unit the session offered before, are each one
/// `capture_dropped_total` line with `stage` `select`, the provisional
/// rule's metric.
#[test]
fn s16_b189_a_summary_section_or_repeat_not_offered_is_journalled() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("b189-journal");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("b189-journal");
    let mut t = prompt("Why does the import fail?", "2026-01-01T00:00:00Z", 1);
    t += &say(FIX_WITH_CAUSE, "2026-01-01T00:01:00Z", 2);
    append(&path, &t);
    let first = stop_reading(&d, &cwd, &session, &path, FIX_WITH_CAUSE);
    barrier(&d, &rig, &store);

    let mut more = compaction_with(
        "2026-01-01T00:10:00Z",
        3,
        &[EXPORT_REQUEST, EXPORT_PENDING].join("\n\n"),
    );
    more += &compaction_with("2026-01-01T00:20:00Z", 4, EXPORT_REQUEST_AGAIN);
    more += &prompt(
        "Which linker does the build use?",
        "2026-01-01T00:21:00Z",
        5,
    );
    more += &say(LINKER_FINDING, "2026-01-01T00:22:00Z", 6);
    append(&path, &more);
    let second = stop_reading(&d, &cwd, &session, &path, LINKER_FINDING);
    barrier(&d, &rig, &store);

    let stored = contents(&project_memories(&store, &cwd));
    assert!(stored.iter().any(|c| c == EXPORT_REQUEST), "{stored:?}");
    assert!(!stored.iter().any(|c| c == EXPORT_PENDING), "{stored:?}");
    assert!(
        !stored.iter().any(|c| c == EXPORT_REQUEST_AGAIN),
        "{stored:?}"
    );
    let lines = |id: &str| -> Vec<Value> {
        metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL)
            .into_iter()
            .filter(|v| v["request_id"] == id)
            .collect()
    };
    assert!(lines(&first).is_empty(), "{:?}", lines(&first));
    let dropped = lines(&second);
    let mut reasons: Vec<String> = dropped
        .iter()
        .map(|v| {
            assert_eq!(v["value"], 1.0, "{v}");
            assert_eq!(v["labels"]["stage"], "select", "{v}");
            assert_eq!(v["labels"]["tool"], "stop", "{v}");
            assert!(v["labels"]["project"].is_string(), "{v}");
            v["labels"]["reason"].as_str().expect("a reason").to_owned()
        })
        .collect();
    reasons.sort();
    assert_eq!(reasons, ["summary_repeat", "summary_section"]);
}

// --- B-185 ----------------------------------------------------------------

/// The owner's step 21 and the reviewer's Q2.1 (B-185): a decision stated
/// plainly, with none of the old decision words, read from the transcript
/// at Stop as Claude Code sends it, is stored pending and queued for the
/// judge.
#[test]
fn s16_b179_a_plainly_stated_decision_is_queued_for_the_judge() {
    const PLAIN: &str =
        "From now on the session cache lives in Redis, and the in-process map goes away.";
    let rig = Rig::new();
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("b185-plain");
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("b185-plain");
    let mut t = prompt(
        "Where should the session cache live from now on?",
        "2026-01-01T00:00:00Z",
        1,
    );
    t += &say(PLAIN, "2026-01-01T00:01:00Z", 2);
    append(&path, &t);
    stop_reading(&d, &cwd, &session, &path, PLAIN);
    barrier(&d, &rig, &store);
    let memories = project_memories(&store, &cwd);
    let decision = memories
        .iter()
        .find(|m| m.content == PLAIN)
        .expect("the decision is captured");
    assert_eq!(
        decision.status,
        Status::Pending,
        "never served before the judge"
    );
    let queued = store
        .judge_queue(1000, 10_000_000)
        .expect("read the judge queue");
    let item = queued
        .iter()
        .find(|i| i.id == decision.id)
        .expect("queued for the judge");
    assert_eq!(item.content, PLAIN);
    assert_eq!(
        item.context, "Where should the session cache live from now on?",
        "the judge reads the turn's request beside it"
    );
}
