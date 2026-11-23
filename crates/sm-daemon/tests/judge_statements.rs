//! The judge's statements task (SPEC 16, The judge; B-240 to B-245), on the
//! fake model with a scripted judge. The texts are synthetic; the cosines
//! named in the comments are the fake embedder's.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common::*;
use serde_json::{Value, json};
use sm_core::config::JudgeTask;
use sm_core::ids::MemoryId;
use sm_core::journal::metrics;
use sm_core::project::resolve_project;
use sm_core::protocol::{CaptureParams, HookEvent, Op, RecallParams, ReplayParams, Reply};
use sm_core::testing::{dot, fake_vector};
use sm_core::time::Timestamp;
use sm_core::traits::ChunkPiece;
use sm_core::types::{
    Memory, MemoryType, NewMemory, Provenance, ProvenanceSource, Status, derive_title,
};
use sm_daemon::{JudgeCandidate, JudgeFailure, JudgeFailureKind, JudgeStatement, SameItem};
use sm_store::{MemoryInput, MemoryListFilter, NewChunk, Store};

const WAIT: Duration = Duration::from_secs(20);

// --- texts ----------------------------------------------------------------

/// A decision already kept.
const KEPT: &str =
    "We decided to cap the uploader's retries at five, because the server throttles bursts.";
const KEPT_SPAN: (&str, &str) = ("We decided to cap", "throttles bursts.");
/// [`KEPT`] after a sentence of narration, as today's judge keeps it whole.
const DEC: &str = "I have looked at the uploader again. We decided to cap the uploader's retries at five, because the server throttles bursts.";
/// [`KEPT`] in other words: 0.8824 to it, 0.8146 to [`DEC`].
const S_REPEAT: &str =
    "We decided the uploader's retries are capped at five, because the server throttles bursts.";
const S_REPEAT_SPAN: (&str, &str) = ("We decided the uploader's", "throttles bursts.");
/// A new decision: under 0.34 to every other text here.
const S_FRESH: &str =
    "We decided the indexer reads rows in pages of two hundred, because memory is tight.";
const S_FRESH_SPAN: (&str, &str) = ("We decided the indexer", "memory is tight.");
/// Narration, which no statement holds.
const NARR: &str = "I have looked at the code again and ran the tests.";
/// A second decision already kept.
const K2: &str =
    "We decided the cache keeps entries for ten minutes, because the feed changes slowly.";
const K2_SPAN: (&str, &str) = ("We decided the cache keeps", "changes slowly.");
/// [`K2`] in other words: 0.9095 to it.
const S2: &str =
    "We decided the cache entries are kept for ten minutes, because the feed changes slowly.";
const S2_SPAN: (&str, &str) = ("We decided the cache entries", "changes slowly.");
const SECRET_SPAN: (&str, &str) = ("We decided the importer signs", "because the API needs it.");
const PROMPT: &str = "Why does the indexer page its reads?";
const GOOD: &str = "We decided the scheduler runs jobs nightly, because the load is low then.";
const BAD: &str = "We decided the parser skips comments, because they carry no data.";
const TOOL: &str = "The test run printed 42 passed and 0 failed in the uploader suite.";

fn msg() -> String {
    format!("{NARR} {S_REPEAT} {S_FRESH}")
}

fn anthropic_key() -> String {
    format!("sk-ant-{}", "a1B2".repeat(8))
}

fn aws_key() -> String {
    format!("AKIA{}", "Q7".repeat(8))
}

// --- helpers ----------------------------------------------------------------

/// A rig whose daemons run `judge` under the statements task.
fn statements_rig(judge: &Arc<ScriptedJudge>) -> Rig {
    let mut rig = Rig::new();
    rig.settings.capture.judge.task = JudgeTask::Statements;
    rig.use_judge(judge.clone());
    rig
}

/// One statement a script marks wherever its first words are in a
/// candidate's text.
#[derive(Debug, Clone, Copy)]
struct Mark {
    span: (&'static str, &'static str),
    memory_type: &'static str,
    grade: &'static str,
}

fn mark(
    span: (&'static str, &'static str),
    memory_type: &'static str,
    grade: &'static str,
) -> Mark {
    Mark {
        span,
        memory_type,
        grade,
    }
}

/// The statements of `marks` found in `c`; a barrier sentinel is kept whole.
fn marked(c: &JudgeCandidate, marks: &[Mark]) -> Vec<JudgeStatement> {
    if c.text.contains("barrier marker") {
        return vec![statement(
            (&c.text, &c.text),
            "keep",
            "decision",
            "must_keep",
        )];
    }
    marks
        .iter()
        .filter(|m| c.text.contains(m.span.0))
        .map(|m| statement(m.span, "keep", m.memory_type, m.grade))
        .collect()
}

fn marking(marks: Vec<Mark>) -> StatementsScript {
    Box::new(move |batch| units_answer(batch, |c| marked(c, &marks)))
}

fn bad_answer(message: &str) -> JudgeFailure {
    JudgeFailure {
        kind: JudgeFailureKind::BadAnswer,
        message: message.to_owned(),
        retry_at: None,
        cost_usd: 0.0,
    }
}

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

fn active(store: &Store, cwd: &str) -> Vec<Memory> {
    project_memories(store, cwd)
        .into_iter()
        .filter(|m| m.status == Status::Active)
        .collect()
}

fn with_content<'a>(memories: &'a [Memory], content: &str) -> &'a Memory {
    memories
        .iter()
        .find(|m| m.content == content)
        .unwrap_or_else(|| panic!("no memory holds {content:?}: {memories:?}"))
}

/// The judge record a statement's memory carries.
fn judge_extra(m: &Memory) -> Value {
    m.provenance
        .extra
        .as_ref()
        .map(|e| e["judge"].clone())
        .unwrap_or(Value::Null)
}

/// A Stop capture of `message` in a session of its own, once accepted.
fn stop(d: &Daemon, cwd: &str, message: &str) {
    let session = unique("statements");
    queue(
        d,
        cwd,
        &session,
        HookEvent::Stop,
        json!({
            "session_id": session,
            "stop_hook_active": false,
            "last_assistant_message": message,
        }),
    );
}

fn queue(d: &Daemon, cwd: &str, session: &str, event: HookEvent, payload: Value) {
    let req = d
        .req(
            cwd,
            Op::Capture(CaptureParams {
                event,
                payload,
                received_at: Timestamp::now(),
            }),
        )
        .with_session(Some(session.to_owned()));
    match d.send(&req).into_result() {
        Ok(Reply::Captured(r)) => assert!(r.accepted, "the capture was accepted"),
        Ok(other) => panic!("capture replied {other:?}"),
        Err(e) => panic!("capture failed: {}: {}", e.code.as_str(), e.message),
    }
}

/// One capture stored before the daemon starts.
struct Cap<'a> {
    text: &'a str,
    context: String,
    session: Option<&'a str>,
    /// Microseconds after the first capture's creation.
    after_us: i64,
    source: Option<&'a str>,
}

fn cap(text: &str) -> Cap<'_> {
    Cap {
        text,
        context: String::new(),
        session: None,
        after_us: 0,
        source: None,
    }
}

/// Stores `caps` as pending captures queued for the judge, in one call.
fn insert(store: &Store, cwd: &str, caps: Vec<Cap<'_>>) -> Vec<MemoryId> {
    let identity = resolve_project(Path::new(cwd)).expect("a project");
    let project = store.upsert_project(&identity).expect("the project");
    let t0 = Timestamp::now();
    let inputs = caps
        .into_iter()
        .map(|c| {
            let mut memory = NewMemory::new(
                project.id,
                c.text.to_owned(),
                Provenance::new(ProvenanceSource::Capture),
            );
            memory.status = Status::Pending;
            memory.created_at = Some(t0.add_micros(c.after_us));
            memory.provenance.session_id = c.session.map(str::to_owned);
            memory.provenance.extra = c.source.map(|s| json!({ "source": s }));
            let piece = ChunkPiece {
                seq: 0,
                text: c.text.to_owned(),
                token_count: 12,
            };
            (
                MemoryInput {
                    memory,
                    chunks: vec![NewChunk::pending(piece)],
                },
                c.context,
            )
        })
        .collect();
    store
        .insert_captures(inputs)
        .expect("store the captures")
        .into_iter()
        .map(|i| i.id)
        .collect()
}

/// Waits until no capture waits for the judge.
fn settled(store: &Store) {
    assert!(
        wait_until(WAIT, || store
            .judge_queue_stats()
            .is_ok_and(|s| s.pending == 0)),
        "every capture was judged: {:?}",
        store.judge_queue_stats()
    );
}

/// Waits until every capture queued before it has been judged: a sentinel
/// in its own project, which the scripts keep whole, is made active.
fn barrier(d: &Daemon, rig: &Rig, store: &Store) {
    let cwd = rig.project(&unique("barrier"));
    let text = format!(
        "We decided to use {} as the barrier marker, because the capture queue is first in first out.",
        unique("sentinel")
    );
    stop(d, &cwd, &text);
    assert!(
        wait_until(WAIT, || !active(store, &cwd).is_empty()
            && store.judge_queue_stats().is_ok_and(|s| s.pending == 0)),
        "the barrier was judged"
    );
}

fn usage(store: &Store) -> sm_store::JudgeUsage {
    store
        .judge_usage(&Timestamp::now().date_string())
        .expect("the judge's usage")
}

fn outcomes(rig: &Rig, n: usize) -> Vec<String> {
    let mut out: Vec<String> =
        wait_metric_lines(rig, metrics::CAPTURE_STATEMENTS_TOTAL, n, WAIT, |_| true)
            .iter()
            .map(|l| {
                l["labels"]["outcome"]
                    .as_str()
                    .expect("an outcome")
                    .to_owned()
            })
            .collect();
    out.sort();
    out
}

fn sorted(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|s| (*s).to_owned()).collect();
    out.sort();
    out
}

fn recalled(d: &Daemon, cwd: &str, query: &str) -> Vec<MemoryId> {
    recall(d, cwd, RecallParams::new(query))
        .results
        .iter()
        .map(|m| m.id)
        .collect()
}

/// A pseudo-random word of seven letters (as in `tests/judge.rs`).
fn word(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..7)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            char::from(b'a' + (x % 26) as u8)
        })
        .collect()
}

/// A decision text of mostly its own words (as in `tests/judge.rs`).
fn distinct_decision(n: u64) -> String {
    let w = |i: u64| word(n * 16 + i);
    format!(
        "We decided {} {} {} {} {} {}, because {} {} {}.",
        w(1),
        w(2),
        w(3),
        w(4),
        w(5),
        w(6),
        w(7),
        w(8),
        w(9)
    )
}

// --- B-242: each statement is its own memory ---------------------------------

/// B-242 (SPEC 16, The judge; A6): under the statements task each kept
/// statement becomes its own active memory, made of the capture's own
/// redacted sentence, with `derive_title`'s title, its own chunks and
/// vector, the type the judge gave it, and its grade and position in the
/// provenance. The first takes the capture's row. The narration no
/// statement holds is not stored, the judge never sees a secret, and with
/// no memory near either statement no duplicate step call is made.
#[test]
fn s16_b242_each_statement_is_redacted_and_chunked() {
    let anthropic = anthropic_key();
    let aws = aws_key();
    let secret = format!(
        "We decided the importer signs requests with {anthropic}, because the API needs it."
    );
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(S_FRESH_SPAN, "decision", "must_keep"),
        mark(SECRET_SPAN, "note", "good_to_have"),
    ]));
    let rig = statements_rig(&judge);
    let cwd = rig.project("statements");
    let text = format!("{NARR} {S_FRESH} {secret}");
    let id = {
        let store = rig.store();
        insert(
            &store,
            &cwd,
            vec![Cap {
                context: format!("Use the key {aws} for the import."),
                ..cap(&text)
            }],
        )[0]
    };
    let d = rig.start_fake();
    let store = rig.store();
    settled(&store);
    d.wait_embedded(&cwd);

    let seen: Vec<String> = judge
        .statement_batches()
        .iter()
        .flatten()
        .flat_map(|c| [c.text.clone(), c.context.clone()])
        .collect();
    assert!(seen.iter().any(|t| t.contains("[REDACTED:")), "{seen:?}");
    for t in &seen {
        assert!(
            !t.contains(&aws) && !t.contains(&anthropic),
            "the judge saw a secret: {t}"
        );
    }
    assert!(
        judge.batches().is_empty(),
        "the verdicts task was not called"
    );
    assert!(
        judge.same_calls().is_empty(),
        "no memory is near either statement"
    );

    let stored = active(&store, &cwd);
    assert_eq!(stored.len(), 2, "{stored:?}");
    assert_eq!(
        project_memories(&store, &cwd).len(),
        2,
        "nothing else is kept"
    );
    let fresh = with_content(&stored, S_FRESH);
    assert_eq!(fresh.id, id, "the first statement takes the capture's row");
    assert_eq!(fresh.memory_type, MemoryType::Decision);
    let signed = stored
        .iter()
        .find(|m| m.content.contains("importer signs"))
        .expect("the second statement");
    assert_ne!(signed.id, id);
    assert_eq!(signed.memory_type, MemoryType::Note);
    assert!(signed.content.contains("[REDACTED:"), "{}", signed.content);
    // The daemon's stamp, so this handle may read vectors.
    let info = sm_core::testing::fake_info();
    let indexed = rig.store();
    let index = indexed
        .configure_index(sm_core::types::IndexStamp {
            model_id: info.model_id,
            revision: info.revision,
            dim: info.dim,
            chunk_max_tokens: rig.settings.chunk.max_tokens,
            chunk_overlap: rig.settings.chunk.overlap,
        })
        .expect("the daemon's stamp");
    assert!(index.matches, "{:?}", index.differences);
    for m in &stored {
        assert!(
            !m.content.contains("ran the tests"),
            "narration was kept: {}",
            m.content
        );
        assert!(!m.content.contains(&anthropic) && !m.content.contains(&aws));
        assert_eq!(
            m.title,
            derive_title(&m.content),
            "the title is derived (A6)"
        );
        let v = indexed
            .memory_vector(&m.id)
            .expect("read a vector")
            .expect("the statement has a vector");
        assert!(
            dot(&v, &fake_vector(&m.content)) > 0.99,
            "the vector is the statement's own"
        );
    }
    for (m, index, grade) in [(fresh, 0, "must_keep"), (signed, 1, "good_to_have")] {
        let j = judge_extra(m);
        assert_eq!(j["task"], "statements", "{j}");
        assert_eq!(j["model"], "claude-test-judge", "{j}");
        assert_eq!(j["capture"], id.to_string(), "{j}");
        assert_eq!(j["statement"], index, "{j}");
        assert_eq!(j["grade"], grade, "{j}");
        assert!(j["at"].is_i64(), "microseconds, as status counts them: {j}");
        assert!(
            j.get("reason").is_none(),
            "no text written by the model is stored: {j}"
        );
    }

    let cost = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_COST_USD, 1, WAIT, |_| true);
    assert_eq!(cost.len(), 1, "{cost:?}");
    assert_eq!(cost[0]["labels"]["call"], "statements", "{cost:?}");
    assert_eq!(cost[0]["labels"]["candidates"], 1, "{cost:?}");
    assert_eq!(outcomes(&rig, 2), sorted(&["stored", "stored"]));
    let lines = wait_metric_lines(&rig, metrics::CAPTURE_STATEMENTS_TOTAL, 2, WAIT, |_| true);
    let mut graded: Vec<(String, String)> = lines
        .iter()
        .map(|l| {
            (
                l["labels"]["grade"].as_str().expect("a grade").to_owned(),
                l["labels"]["type"].as_str().expect("a type").to_owned(),
            )
        })
        .collect();
    graded.sort();
    assert_eq!(
        graded,
        vec![
            ("good_to_have".to_owned(), "note".to_owned()),
            ("must_keep".to_owned(), "decision".to_owned())
        ]
    );
    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 1, WAIT, |_| true);
    assert_eq!(judged.len(), 1, "{judged:?}");
    assert_eq!(judged[0]["labels"]["verdict"], "keep");
    assert_eq!(judged[0]["labels"]["reason"], "knowledge");
    assert_eq!(judged[0]["labels"]["statements"], 2);
    assert_eq!(judged[0]["labels"]["task"], "statements");

    let s = d.status(&cwd).judge;
    assert_eq!(s.task, JudgeTask::Statements, "{s:?}");
    assert_eq!(
        (s.statements_today, s.repeats_today, s.unchecked_today),
        (2, 0, 0),
        "{s:?}"
    );
    assert_eq!((s.calls_today, s.judged_today), (1, 1), "{s:?}");
}

/// B-242: a statement whose quote is not in the capture's text is not
/// stored, nor one the judge itself labels a duplicate; a capture left with
/// nothing is rejected and kept unserved (B-190), and no duplicate step
/// call is made.
#[test]
fn s16_b242_a_statement_whose_span_is_not_found_is_not_stored() {
    let judge = ScriptedJudge::for_statements(Box::new(|batch| {
        units_answer(batch, |_| {
            vec![
                statement(
                    ("We decided the indexer", "pages of three hundred"),
                    "keep",
                    "decision",
                    "must_keep",
                ),
                statement(S_FRESH_SPAN, "duplicate", "decision", "good_to_have"),
            ]
        })
    }));
    let rig = statements_rig(&judge);
    let cwd = rig.project("unquoted");
    let d = rig.start_fake();
    let store = rig.store();
    stop(&d, &cwd, &format!("{NARR} {S_FRESH}"));
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));
    assert!(active(&store, &cwd).is_empty());
    let kept = project_memories(&store, &cwd);
    assert_eq!(kept.len(), 1, "the rejected capture is kept: {kept:?}");
    assert_eq!(kept[0].status, Status::Pending);
    assert!(judge.same_calls().is_empty());
    assert_eq!(outcomes(&rig, 2), sorted(&["duplicate", "unquoted"]));
    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 1, WAIT, |_| true);
    assert_eq!(judged[0]["labels"]["verdict"], "reject", "{judged:?}");
    assert_eq!(judged[0]["labels"]["reason"], "nothing_kept", "{judged:?}");
    assert_eq!(judged[0]["labels"]["statements"], 2, "{judged:?}");
    let s = d.status(&cwd).judge;
    assert_eq!((s.statements_today, s.rejected), (0, 1), "{s:?}");
}

// --- B-243: the duplicate step -------------------------------------------------

/// B-243 (A4): a statement the duplicate step finds says what an active
/// memory says is not stored. It is kept unserved for 30 days with the id
/// of that memory, so the drop can be audited and undone; recall never
/// serves it. The capture's other statement is stored. The step's call
/// names the statement `s01` and the memory `k01`, as the harness did.
#[test]
fn s16_b243_a_statement_repeating_an_active_memory_is_held_with_its_id() {
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(first_same());
    let rig = statements_rig(&judge);
    let cwd = rig.project("repeat-active");
    let d = rig.start_fake();
    let store = rig.store();
    let kept = remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &msg());
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));

    let calls = judge.same_calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(
        calls[0].len(),
        1,
        "only the statement near a memory is asked"
    );
    let item: &SameItem = &calls[0][0];
    assert_eq!(
        (item.id.as_str(), item.statement.as_str()),
        ("s01", S_REPEAT)
    );
    let candidates: Vec<(&str, &str)> = item
        .candidates
        .iter()
        .map(|c| (c.id.as_str(), c.text.as_str()))
        .collect();
    assert_eq!(candidates, vec![("k01", KEPT)]);

    let all = project_memories(&store, &cwd);
    assert_eq!(all.len(), 3, "{all:?}");
    let held = with_content(&all, S_REPEAT);
    assert_eq!(held.status, Status::Pending, "the repeat is held unserved");
    let j = judge_extra(held);
    assert_eq!(j["repeat_of"], kept.to_string(), "{j}");
    assert_eq!(j["task"], "statements", "{j}");
    let fresh = with_content(&all, S_FRESH);
    assert_eq!(fresh.status, Status::Active);
    assert!(!recalled(&d, &cwd, "uploader retries capped at five").contains(&held.id));

    assert_eq!(outcomes(&rig, 2), sorted(&["repeat", "stored"]));
    let same = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_COST_USD, 2, WAIT, |l| {
        l["labels"]["call"] == "same"
    });
    assert_eq!(same.len(), 1, "{same:?}");
    assert_eq!(same[0]["labels"]["candidates"], 1, "{same:?}");
    let s = d.status(&cwd).judge;
    assert_eq!(
        (s.statements_today, s.repeats_today, s.unchecked_today),
        (1, 1, 0),
        "{s:?}"
    );
    assert_eq!(s.calls_today, 2, "both calls are counted: {s:?}");
}

/// The SQL block of README.md's section on held statements (B-260).
fn readme_held_query() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
    let readme = std::fs::read_to_string(&path).expect("read README.md");
    let (_, section) = readme
        .split_once("### Statements held as repeats")
        .expect("README.md has a section on statements held as repeats");
    let (_, block) = section
        .split_once("```sql")
        .expect("the section has an SQL block");
    let (sql, _) = block.split_once("```").expect("the SQL block ends");
    sql.to_owned()
}

/// Every row the README's query returns, each column read as text.
fn held_rows(db: &Path, sql: &str) -> (Vec<String>, Vec<Vec<Option<String>>>) {
    let c = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("open the store read-only");
    let mut q = c.prepare(sql).expect("the README's query prepares");
    let names: Vec<String> = q.column_names().map(str::to_owned).collect();
    let width = names.len();
    let rows = q
        .query_map([], |r| {
            (0..width)
                .map(|i| r.get::<_, Option<String>>(i))
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .expect("run the README's query")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("read the README's rows");
    (names, rows)
}

/// A project whose capture left [`S_REPEAT`] held as a repeat of [`KEPT`]
/// (as in the B-243 test above).
struct HeldRepeat {
    store: Store,
    d: Daemon,
    cwd: String,
    kept: MemoryId,
    held: MemoryId,
    rig: Rig,
}

fn held_repeat(name: &str) -> HeldRepeat {
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(first_same());
    let rig = statements_rig(&judge);
    let cwd = rig.project(name);
    let d = rig.start_fake();
    let store = rig.store();
    let kept = remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &msg());
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));
    let all = project_memories(&store, &cwd);
    let held = with_content(&all, S_REPEAT);
    assert_eq!(held.status, Status::Pending, "the repeat is held unserved");
    let held = held.id;
    HeldRepeat {
        store,
        d,
        cwd,
        kept,
        held,
        rig,
    }
}

/// B-260: the README says how to list the statements the duplicate step
/// held, see what each was said to repeat, and store one again with the
/// tools that exist today. Its query, run read-only on the store, lists
/// the held repeat with the memory it repeats. `revise` of the held id with
/// status active keeps it, with its id, type and text, and the query then
/// lists nothing.
#[test]
fn s16_b260_the_readme_query_lists_a_held_repeat_and_revise_keeps_it() {
    let sql = readme_held_query();
    let h = held_repeat("held-revise");
    let project = h
        .store
        .project_by_identity(&resolve_project(Path::new(&h.cwd)).expect("a project").key())
        .expect("find the project")
        .expect("the project exists");

    let (names, rows) = held_rows(&h.rig.paths.database(), &sql);
    assert_eq!(
        names,
        [
            "id",
            "project",
            "type",
            "held_as",
            "held_at_utc",
            "repeat_of",
            "repeat_status",
            "repeats",
            "statement"
        ]
    );
    assert_eq!(rows.len(), 1, "only the held statement is listed: {rows:?}");
    let row = &rows[0];
    let text = |i: usize| row[i].as_deref();
    assert_eq!(text(0), Some(h.held.to_string().as_str()), "{row:?}");
    assert_eq!(text(1), Some(project.name.as_str()), "{row:?}");
    assert_eq!(text(2), Some("decision"), "{row:?}");
    assert_eq!(text(3), Some("repeat"), "{row:?}");
    let at = text(4).expect("a time");
    assert!(
        at.len() == 19 && at.as_bytes()[10] == b' ' && at.starts_with("20"),
        "{at}"
    );
    assert_eq!(text(5), Some(h.kept.to_string().as_str()), "{row:?}");
    assert_eq!(text(6), Some("active"), "{row:?}");
    assert_eq!(text(7), Some(KEPT), "{row:?}");
    assert_eq!(text(8), Some(S_REPEAT), "{row:?}");

    let mut p = sm_core::protocol::ReviseParams::new(h.held);
    p.status = Some(Status::Active);
    let Reply::Revise(_) = h.d.ok(&h.cwd, Op::Revise(p)) else {
        panic!("not a revise reply")
    };
    let all = project_memories(&h.store, &h.cwd);
    let copies: Vec<&Memory> = all.iter().filter(|m| m.content == S_REPEAT).collect();
    assert_eq!(copies.len(), 1, "{copies:?}");
    assert_eq!(
        (copies[0].id, copies[0].status, copies[0].memory_type),
        (h.held, Status::Active, MemoryType::Decision)
    );
    assert!(recalled(&h.d, &h.cwd, "uploader retries capped at five").contains(&h.held));
    let (_, rows) = held_rows(&h.rig.paths.database(), &sql);
    assert!(rows.is_empty(), "nothing is held now: {rows:?}");
    let s = h.store.judge_queue_stats().expect("queue stats");
    assert_eq!((s.pending, s.rejected), (0, 0), "{s:?}");
}

/// B-260: the README's other way to store a held statement again:
/// `remember` of its exact text and type, in a session of its project,
/// stores it served and deletes the held copy (B-157), and the README's
/// query then lists nothing.
#[test]
fn s16_b260_remember_of_a_held_repeats_text_stores_it_and_drops_the_held_copy() {
    let sql = readme_held_query();
    let h = held_repeat("held-remember");
    let (_, rows) = held_rows(&h.rig.paths.database(), &sql);
    let ids: Vec<Option<&str>> = rows.iter().map(|r| r[0].as_deref()).collect();
    assert_eq!(ids, [Some(h.held.to_string().as_str())], "{rows:?}");

    let again = remember_item(
        &h.d,
        &h.cwd,
        sm_core::protocol::RememberItem {
            memory_type: Some(MemoryType::Decision),
            ..sm_core::protocol::RememberItem::new(S_REPEAT)
        },
    );
    let all = project_memories(&h.store, &h.cwd);
    let copies: Vec<&Memory> = all.iter().filter(|m| m.content == S_REPEAT).collect();
    assert_eq!(copies.len(), 1, "the held copy is replaced: {copies:?}");
    assert_eq!(copies[0].id, again);
    assert_ne!(copies[0].id, h.held);
    assert_eq!(
        (copies[0].status, copies[0].memory_type),
        (Status::Active, MemoryType::Decision)
    );
    h.d.wait_embedded(&h.cwd);
    assert!(recalled(&h.d, &h.cwd, "uploader retries capped at five").contains(&again));
    let (_, rows) = held_rows(&h.rig.paths.database(), &sql);
    assert!(rows.is_empty(), "nothing is held now: {rows:?}");
}

/// B-243: a statement may repeat an earlier statement of its own batch,
/// stored a moment before; it is held with the id of the memory that
/// statement became.
#[test]
fn s16_b243_a_statement_repeating_an_earlier_statement_of_its_batch_is_held() {
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(KEPT_SPAN, "decision", "must_keep"),
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(first_same());
    let rig = statements_rig(&judge);
    let cwd = rig.project("repeat-batch");
    let message = msg();
    let ids = {
        let store = rig.store();
        insert(
            &store,
            &cwd,
            vec![
                Cap {
                    session: Some("one"),
                    ..cap(KEPT)
                },
                Cap {
                    session: Some("one"),
                    after_us: 1_000,
                    ..cap(&message)
                },
            ],
        )
    };
    let _d = rig.start_fake();
    let store = rig.store();
    settled(&store);
    assert_eq!(judge.statement_batches().len(), 1, "one session, one call");
    let calls = judge.same_calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].len(), 1, "{calls:?}");
    let item = &calls[0][0];
    assert_eq!(
        (item.id.as_str(), item.statement.as_str()),
        ("s02", S_REPEAT)
    );
    let candidates: Vec<(&str, &str)> = item
        .candidates
        .iter()
        .map(|c| (c.id.as_str(), c.text.as_str()))
        .collect();
    assert_eq!(
        candidates,
        vec![("s01", KEPT)],
        "the earlier statement, by its own id"
    );

    let all = project_memories(&store, &cwd);
    let first = with_content(&all, KEPT);
    assert_eq!((first.id, first.status), (ids[0], Status::Active));
    let held = with_content(&all, S_REPEAT);
    assert_eq!(held.status, Status::Pending);
    assert_eq!(judge_extra(held)["repeat_of"], ids[0].to_string());
    let fresh = with_content(&all, S_FRESH);
    assert_eq!((fresh.id, fresh.status), (ids[1], Status::Active));
}

/// B-241: a statement with no memory or earlier statement at a cosine of
/// 0.80 or more is not asked about, and with none to ask no call is made.
#[test]
fn s16_b241_a_statement_with_no_candidate_at_0_80_makes_no_call() {
    let judge =
        ScriptedJudge::for_statements(marking(vec![mark(S_FRESH_SPAN, "decision", "must_keep")]));
    judge.set_same(Box::new(|_| Err(bad_answer("no call was expected"))));
    let rig = statements_rig(&judge);
    let cwd = rig.project("no-candidate");
    let d = rig.start_fake();
    let store = rig.store();
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &format!("{NARR} {S_FRESH}"));
    // The capture is accepted into the spool before the store holds it, so
    // no pending capture may only mean it has not been drained yet.
    assert!(wait_until(WAIT, || active(&store, &cwd).len() >= 2));
    settled(&store);
    assert!(judge.same_calls().is_empty(), "{:?}", judge.same_calls());
    let stored = active(&store, &cwd);
    assert_eq!(stored.len(), 2, "{stored:?}");
    with_content(&stored, S_FRESH);
    assert_eq!(usage(&store).calls, 1);
}

/// B-243 (A2): a bad answer from the duplicate step halves its call, the
/// first half first; a statement answered badly alone JUDGE_MAX_ATTEMPTS
/// (3) times is not stored. It is kept unserved for 30 days as unchecked,
/// and `status` counts it. The capture's other statements settle as usual.
#[test]
fn s16_b243_a_bad_same_answer_halves_the_call_then_holds_the_statement_unchecked() {
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(S2_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(Box::new(|items| {
        if items.iter().any(|i| i.statement == S2) {
            Err(bad_answer("an answer that breaks the schema"))
        } else {
            same_answer(items, |i| i.candidates.first().map(|c| c.id.clone()))
        }
    }));
    let rig = statements_rig(&judge);
    let cwd = rig.project("unchecked");
    let d = rig.start_fake();
    let store = rig.store();
    let kept = remember(&d, &cwd, KEPT);
    remember(&d, &cwd, K2);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &format!("{NARR} {S_REPEAT} {S2} {S_FRESH}"));
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 2)));

    let lengths: Vec<usize> = judge.same_calls().iter().map(Vec::len).collect();
    assert_eq!(lengths, vec![2, 1, 1, 1, 1]);
    let all = project_memories(&store, &cwd);
    let repeat = with_content(&all, S_REPEAT);
    assert_eq!(repeat.status, Status::Pending);
    assert_eq!(judge_extra(repeat)["repeat_of"], kept.to_string());
    let unchecked = with_content(&all, S2);
    assert_eq!(
        unchecked.status,
        Status::Pending,
        "an unchecked statement is not stored"
    );
    assert_eq!(judge_extra(unchecked)["repeat_of"], Value::Null);
    assert_eq!(with_content(&all, S_FRESH).status, Status::Active);
    assert_eq!(
        outcomes(&rig, 3),
        sorted(&["repeat", "stored", "unchecked"])
    );
    let s = d.status(&cwd).judge;
    assert_eq!(
        (s.statements_today, s.repeats_today, s.unchecked_today),
        (1, 1, 1),
        "{s:?}"
    );
    let u = usage(&store);
    assert_eq!((u.calls, u.failures), (6, 4), "{u:?}");
}

// --- B-244: all or nothing ------------------------------------------------------

/// B-244 (A3): the daily cap reached after the statements call, before the
/// duplicate step's, leaves the whole capture waiting: nothing is stored,
/// half or unchecked. Under a higher cap the capture is judged again and
/// settled whole.
#[test]
fn s16_b244_a_cap_part_way_leaves_the_whole_capture_waiting() {
    let judge = ScriptedJudge::for_statements(marking(vec![
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(first_same());
    let mut rig = statements_rig(&judge);
    rig.settings.capture.judge.max_calls_per_day = 1;
    let cwd = rig.project("capped");
    let d = rig.start_fake();
    let store = rig.store();
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &msg());
    let pending = wait_metric_lines(&rig, metrics::CAPTURE_PENDING, 1, WAIT, |l| {
        l["request_id"] == "judge"
    });
    assert_eq!(pending[0]["value"], 1.0, "{pending:?}");
    assert_eq!(judge.statement_batches().len(), 1);
    assert!(
        judge.same_calls().is_empty(),
        "the cap stops the second call"
    );
    let stats = store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued, stats.rejected), (1, 1, 0));
    assert_eq!(active(&store, &cwd).len(), 1, "only the remembered memory");
    assert_eq!(project_memories(&store, &cwd).len(), 2);
    assert!(metric_lines(&rig, metrics::CAPTURE_STATEMENTS_TOTAL).is_empty());

    d.shutdown();
    rig.settings.capture.judge.max_calls_per_day = 600;
    let _d = rig.start_fake();
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));
    assert_eq!(
        judge.statement_batches().len(),
        2,
        "judged again from the start"
    );
    assert_eq!(judge.same_calls().len(), 1);
    assert_eq!(active(&store, &cwd).len(), 2);
    assert_eq!(outcomes(&rig, 2), sorted(&["repeat", "stored"]));
}

/// B-244 (A3): a daemon stopped during the duplicate step's call leaves the
/// whole capture waiting, and a stop is not counted as a failure. The next
/// daemon judges it again and settles it whole.
#[test]
fn s16_b244_a_stop_part_way_leaves_the_whole_capture_waiting() {
    let judge = ScriptedJudge::for_statements(Box::new(|_| Err(bad_answer("replaced below"))));
    let weak = Arc::downgrade(&judge);
    let once = AtomicBool::new(true);
    judge.set_statements(Box::new(move |batch| {
        if once.swap(false, Ordering::SeqCst)
            && let Some(j) = weak.upgrade()
        {
            // The duplicate step's call that follows is held open.
            j.hold();
        }
        units_answer(batch, |c| {
            marked(
                c,
                &[
                    mark(S_REPEAT_SPAN, "decision", "must_keep"),
                    mark(S_FRESH_SPAN, "decision", "must_keep"),
                ],
            )
        })
    }));
    judge.set_same(first_same());
    let rig = statements_rig(&judge);
    let cwd = rig.project("stopped");
    let d = rig.start_fake();
    let store = rig.store();
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &msg());
    assert!(wait_until(WAIT, || judge.same_calls().len() == 1 && judge.in_call()));
    d.shutdown();

    let stats = store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued, stats.rejected), (1, 1, 0));
    assert_eq!(
        active(&store, &cwd).len(),
        1,
        "nothing of the capture is stored"
    );
    assert!(metric_lines(&rig, metrics::CAPTURE_STATEMENTS_TOTAL).is_empty());
    assert!(metric_lines(&rig, metrics::CAPTURE_JUDGE_FAILURES_TOTAL).is_empty());

    judge.release();
    let _d = rig.start_fake();
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));
    assert_eq!(judge.statement_batches().len(), 2);
    assert_eq!(judge.same_calls().len(), 2);
    let stored = active(&store, &cwd);
    assert_eq!(stored.len(), 2, "{stored:?}");
    with_content(&stored, S_FRESH);
}

// --- B-245: a source switched off -------------------------------------------------

/// The capture of tool output and a decision, stored before the daemon
/// starts, judged under `task` with `capture.tool_results` as given.
/// Returns the texts the judge was given.
fn judge_tool_output(task: JudgeTask, tool_results: bool) -> (Rig, Vec<String>) {
    let judge = ScriptedJudge::new(keep_all("note"));
    judge.set_statements(keep_whole("note"));
    let mut rig = Rig::new();
    rig.settings.capture.judge.task = task;
    rig.settings.capture.tool_results = tool_results;
    rig.use_judge(judge.clone());
    let cwd = rig.project("tool-output");
    {
        let store = rig.store();
        insert(
            &store,
            &cwd,
            vec![
                Cap {
                    source: Some("tool_result"),
                    ..cap(TOOL)
                },
                Cap {
                    after_us: 1_000,
                    ..cap(GOOD)
                },
            ],
        );
    }
    let _d = rig.start_fake();
    let store = rig.store();
    settled(&store);
    let texts: Vec<String> = judge
        .batches()
        .iter()
        .chain(judge.statement_batches().iter())
        .flatten()
        .map(|c| c.text.clone())
        .collect();
    let stored: Vec<String> = active(&store, &cwd)
        .into_iter()
        .map(|m| m.content)
        .collect();
    if tool_results {
        assert_eq!(stored.len(), 2, "{stored:?}");
    } else {
        assert_eq!(stored, vec![GOOD.to_owned()]);
        assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);
        let kept = project_memories(&store, &cwd);
        assert_eq!(
            with_content(&kept, TOOL).status,
            Status::Pending,
            "kept unserved for 30 days"
        );
    }
    (rig, texts)
}

/// B-245 (item 7): a capture whose source is switched off, tool output
/// with `capture.tool_results = false` (B-236), is never sent to the judge
/// under either task. It is rejected without a call, with the reason
/// `tool_results_off`, and journalled as dropped at the judge. With the
/// switch on it is judged.
#[test]
fn s16_b245_a_switched_off_source_is_rejected_without_a_call_under_either_task() {
    for task in [JudgeTask::Verdicts, JudgeTask::Statements] {
        let (rig, texts) = judge_tool_output(task, false);
        assert_eq!(texts, vec![GOOD.to_owned()], "{task:?}");
        let dropped = wait_metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL, 1, WAIT, |l| {
            l["labels"]["stage"] == "judge"
        });
        assert_eq!(dropped.len(), 1, "{task:?}: {dropped:?}");
        assert_eq!(
            dropped[0]["labels"]["reason"], "tool_results_off",
            "{task:?}"
        );

        let (_rig, texts) = judge_tool_output(task, true);
        assert!(texts.contains(&TOOL.to_owned()), "{task:?}: {texts:?}");
    }
}

// --- B-242: a transcript read again ----------------------------------------------------

/// B-242 (item 8): reading the same transcript again (the next PreCompact
/// of the session, a new session on the same file, a replay) never stores
/// its statements again, and the judge is never asked about it again.
#[test]
fn s16_b242_reading_a_transcript_again_stores_nothing_again() {
    let judge =
        ScriptedJudge::for_statements(marking(vec![mark(S_FRESH_SPAN, "decision", "must_keep")]));
    let rig = statements_rig(&judge);
    let cwd = rig.project("reread");
    let d = rig.start_fake();
    let store = rig.store();
    let path = Path::new(&cwd).join("transcript.jsonl");
    let session = unique("reread");
    let reply = format!("{NARR} {S_FRESH}");
    let t = prompt(PROMPT, "2026-01-01T00:00:00Z", 1) + &say(&reply, "2026-01-01T00:01:00Z", 2);
    append(&path, &t);
    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    let snapshot = |store: &Store| {
        let mut v: Vec<(MemoryId, Status, String)> = project_memories(store, &cwd)
            .into_iter()
            .map(|m| (m.id, m.status, m.content))
            .collect();
        v.sort();
        v
    };
    let first = snapshot(&store);
    let stored: Vec<&(MemoryId, Status, String)> =
        first.iter().filter(|m| m.1 == Status::Active).collect();
    assert_eq!(stored.len(), 1, "{first:?}");
    assert_eq!(stored[0].2, S_FRESH);

    pre_compact(&d, &cwd, &session, &path);
    barrier(&d, &rig, &store);
    pre_compact(&d, &cwd, &unique("reread-new"), &path);
    barrier(&d, &rig, &store);
    match d.ok(
        &cwd,
        Op::Replay(ReplayParams {
            transcript_path: path.to_str().expect("UTF-8").to_owned(),
            session_id: session.clone(),
            cwd: cwd.clone(),
            received_at: Timestamp::now(),
        }),
    ) {
        Reply::Captured(r) => assert!(r.accepted, "the replay was queued"),
        other => panic!("replay replied {other:?}"),
    }
    barrier(&d, &rig, &store);

    assert_eq!(snapshot(&store), first, "nothing is stored again");
    let asked = judge
        .statement_batches()
        .iter()
        .flatten()
        .filter(|c| c.text.contains("ran the tests"))
        .count();
    assert_eq!(asked, 1, "the reply was judged once");
}

fn line(v: Value) -> String {
    format!("{v}\n")
}

fn prompt(text: &str, ts: &str, n: usize) -> String {
    line(json!({
        "type": "user",
        "message": {"role": "user", "content": text},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("p{n}"),
    }))
}

fn say(text: &str, ts: &str, n: usize) -> String {
    line(json!({
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "isSidechain": false,
        "timestamp": ts,
        "uuid": format!("a{n}"),
    }))
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

fn pre_compact(d: &Daemon, cwd: &str, session: &str, path: &Path) {
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
    );
}

// --- B-243: memories made by today's judge ---------------------------------------------

/// B-243 (item 9): the duplicate step against a store that already holds
/// whole-capture memories made by the verdicts task. A statement that
/// repeats one in other words is held with its id; one whose text is a
/// memory's own is not stored again (outcome `hash`); a new one is stored.
#[test]
fn s16_b243_the_step_against_whole_capture_memories_of_the_verdicts_task() {
    let judge = ScriptedJudge::new(keep_all("decision"));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let cwd = rig.project("verdicts-store");
    let store = rig.store();
    let d = rig.start_fake();
    stop(&d, &cwd, DEC);
    stop(&d, &cwd, K2);
    assert!(wait_until(WAIT, || active(&store, &cwd).len() == 2));
    d.wait_embedded(&cwd);
    let before = active(&store, &cwd);
    let dec = with_content(&before, DEC).id;
    let k2 = with_content(&before, K2).id;
    assert_eq!(judge_extra(with_content(&before, DEC)).get("task"), None);
    d.shutdown();

    rig.settings.capture.judge.task = JudgeTask::Statements;
    judge.set_statements(marking(vec![
        mark(S_REPEAT_SPAN, "decision", "must_keep"),
        mark(K2_SPAN, "decision", "must_keep"),
        mark(S_FRESH_SPAN, "decision", "must_keep"),
    ]));
    judge.set_same(Box::new(|items| {
        same_answer(items, |i| {
            (i.statement == S_REPEAT)
                .then(|| i.candidates.first().map(|c| c.id.clone()))
                .flatten()
        })
    }));
    let d = rig.start_fake();
    stop(&d, &cwd, &format!("{NARR} {S_REPEAT} {K2} {S_FRESH}"));
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.pending == 0 && s.rejected == 1)));

    let calls = judge.same_calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    let asked: Vec<(String, Vec<(String, String)>)> = calls[0]
        .iter()
        .map(|i| {
            (
                i.id.clone(),
                i.candidates
                    .iter()
                    .map(|c| (c.id.clone(), c.text.clone()))
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        asked,
        vec![
            ("s01".to_owned(), vec![("k01".to_owned(), DEC.to_owned())]),
            ("s02".to_owned(), vec![("k02".to_owned(), K2.to_owned())]),
        ]
    );
    let all = project_memories(&store, &cwd);
    let held = with_content(&all, S_REPEAT);
    assert_eq!(held.status, Status::Pending);
    assert_eq!(judge_extra(held)["repeat_of"], dec.to_string());
    assert_eq!(
        all.iter().filter(|m| m.content == K2).count(),
        1,
        "the memory's own text is not stored again"
    );
    assert_eq!(with_content(&all, K2).id, k2);
    assert_eq!(with_content(&all, S_FRESH).status, Status::Active);
    assert_eq!(outcomes(&rig, 3), sorted(&["hash", "repeat", "stored"]));
    let _ = d;
}

// --- B-241: the queue and a bad answer -------------------------------------------------

/// B-241: under the statements task a call holds one session's captures,
/// in the order they were captured.
#[test]
fn s16_b241_the_queue_takes_one_session_per_call() {
    let judge = ScriptedJudge::for_statements(keep_whole("decision"));
    let rig = statements_rig(&judge);
    let cwd = rig.project("sessions");
    let (a1, a2, b1) = (
        distinct_decision(1),
        distinct_decision(2),
        distinct_decision(3),
    );
    {
        let store = rig.store();
        let at = |text: &str, session: &'static str, us: i64| {
            insert(
                &store,
                &cwd,
                vec![Cap {
                    session: Some(session),
                    after_us: us,
                    ..cap(text)
                }],
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        // Queued in this order; the oldest queued names the first session.
        at(&a2, "sa", 2_000_000);
        at(&b1, "sb", 1_000_000);
        at(&a1, "sa", -5_000);
    }
    let _d = rig.start_fake();
    let store = rig.store();
    settled(&store);
    let batches: Vec<Vec<String>> = judge
        .statement_batches()
        .iter()
        .map(|b| b.iter().map(|c| c.text.clone()).collect())
        .collect();
    assert_eq!(batches, vec![vec![a1, a2], vec![b1]]);
}

/// B-241: a bad statements answer halves the call, the first half first,
/// down to one capture, which is tried twice and then marked failed
/// judgement (B-206), as under the verdicts task.
#[test]
fn s16_b241_a_bad_statements_answer_halves_the_batch() {
    let judge = ScriptedJudge::for_statements(Box::new(|batch| {
        units_answer(batch, |c| {
            let memory_type = if c.text == BAD { "gossip" } else { "decision" };
            vec![statement(
                (&c.text, &c.text),
                "keep",
                memory_type,
                "must_keep",
            )]
        })
    }));
    let rig = statements_rig(&judge);
    let cwd = rig.project("bad-statements");
    {
        let store = rig.store();
        insert(
            &store,
            &cwd,
            vec![
                Cap {
                    session: Some("one"),
                    ..cap(GOOD)
                },
                Cap {
                    session: Some("one"),
                    after_us: 1_000,
                    ..cap(BAD)
                },
            ],
        );
    }
    let _d = rig.start_fake();
    let store = rig.store();
    assert!(wait_until(WAIT, || store
        .judge_queue_stats()
        .is_ok_and(|s| s.failed == 1 && s.queued == 0)));
    let lengths: Vec<usize> = judge.statement_batches().iter().map(Vec::len).collect();
    assert_eq!(lengths, vec![2, 1, 1, 1]);
    let stored: Vec<String> = active(&store, &cwd)
        .into_iter()
        .map(|m| m.content)
        .collect();
    assert_eq!(stored, vec![GOOD.to_owned()]);
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(
        judge.statement_batches().len(),
        4,
        "a failed capture is not judged again"
    );
}
