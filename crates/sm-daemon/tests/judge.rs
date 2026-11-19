//! The capture judge in a running daemon (SPEC 16, The judge; B-165 to
//! B-169). Each test runs the daemon with the fake embedder and a
//! [`ScriptedJudge`], sends captures over the real endpoint, and checks what
//! the store, `status`, recall and the journal show.
//!
//! The scripted judge stands in for Claude Code. These tests prove the
//! daemon's handling of verdicts and failures, not the model's judgement,
//! which is measured on the labelled export (`CAPTURE-RCA.md` section 8.5).
//! Every text here was written for these tests.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;
use sm_core::journal::metrics;
use sm_core::project::resolve_project;
use sm_core::protocol::{
    CaptureParams, HookEvent, JudgeBlocked, JudgeStatus, Op, RecallParams, Reply,
};
use sm_core::time::Timestamp;
use sm_core::traits::ChunkPiece;
use sm_core::types::{Memory, MemoryType, NewMemory, Provenance, ProvenanceSource, Status};
use sm_daemon::{JudgeAnswer, JudgeFailure, JudgeFailureKind, JudgeVerdict};
use sm_store::{MemoryInput, MemoryListFilter, NewChunk, Store};

const WAIT: Duration = Duration::from_secs(20);

/// A Stop message: a sentence of narration, then a decision with its reason.
const DECISION: &str = "I have looked at the uploader again. We decided to cap the uploader's retries at five, because the server throttles bursts.";
/// The span the scripted judge marks in it: the decision, its end cut
/// short (B-191).
const SPAN: (&str, &str) = ("We decided to cap the uploader's", "retries at five");
/// What is kept: the whole sentence the span found, without the narration
/// (B-173, B-191).
const KEPT: &str =
    "We decided to cap the uploader's retries at five, because the server throttles bursts.";

// --- helpers ---------------------------------------------------------------

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

/// A Stop capture of `message`; returns the request id once accepted.
fn stop(d: &Daemon, cwd: &str, message: &str) -> String {
    let session = unique("judge");
    let req = d
        .req(
            cwd,
            Op::Capture(CaptureParams {
                event: HookEvent::Stop,
                payload: json!({
                    "session_id": session,
                    "stop_hook_active": false,
                    "last_assistant_message": message,
                }),
                received_at: Timestamp::now(),
            }),
        )
        .with_session(Some(session.clone()));
    match d.send(&req).into_result() {
        Ok(Reply::Captured(r)) => assert!(r.accepted, "the capture was accepted"),
        Ok(other) => panic!("capture replied {other:?}"),
        Err(e) => panic!("capture failed: {}: {}", e.code.as_str(), e.message),
    }
    req.id
}

/// Waits until the project holds `n` memories, all with `status`.
fn wait_for(store: &Store, cwd: &str, n: usize, status: Status) -> Vec<Memory> {
    let mut last = Vec::new();
    let ok = wait_until(WAIT, || {
        last = project_memories(store, cwd);
        last.len() == n && last.iter().all(|m| m.status == status)
    });
    assert!(
        ok,
        "expected {n} {status:?} memories, found {:?}",
        last.iter()
            .map(|m| (m.status, m.content.clone()))
            .collect::<Vec<_>>()
    );
    last
}

/// The judge's status once a failed call has been recorded.
fn backing_off(d: &Daemon, cwd: &str) -> JudgeStatus {
    let mut last = None;
    let ok = wait_until(WAIT, || {
        let s = d.status(cwd).judge;
        let done = s.blocked == Some(JudgeBlocked::BackingOff);
        last = Some(s);
        done
    });
    assert!(ok, "the judge backs off: {last:?}");
    last.expect("a status")
}

fn pending_count(store: &Store) -> u64 {
    store.judge_queue_stats().expect("queue stats").pending
}

fn recalled(d: &Daemon, cwd: &str, query: &str) -> Vec<sm_core::ids::MemoryId> {
    recall(d, cwd, RecallParams::new(query))
        .results
        .iter()
        .map(|m| m.id)
        .collect()
}

/// A pseudo-random word of seven letters. Words that differ in one
/// character, such as `w8x1` and `w12x1`, can share fake-embedder buckets
/// (FNV-1a modulo 768); these do not.
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

/// A decision text of mostly its own words, so the fake embedder puts no
/// two of them near each other (the closest of the first 60 is 0.42).
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

// --- keep and reject ---------------------------------------------------------

/// A kept capture becomes active with the whole sentences of its own text
/// that the judge's span marks, and the judge's type (B-173, B-191): the
/// span stops short, the sentence is kept whole, and the narration the
/// judge left out is not kept. It records the judge in its provenance and is served; the
/// call is counted in `status` and the journal.
#[test]
fn s16_a_kept_capture_becomes_active_with_the_sentences_its_span_marks_and_its_type() {
    let judge = ScriptedJudge::new(Box::new(|batch| {
        answer(batch, |c| keep(c, SPAN, "decision"))
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("kept");
    stop(&d, &cwd, DECISION);

    let kept = wait_for(&store, &cwd, 1, Status::Active);
    let m = &kept[0];
    assert_eq!(m.content, KEPT);
    assert_eq!(m.memory_type, MemoryType::Decision);
    assert_eq!(m.provenance.source, ProvenanceSource::Capture);
    assert_eq!(
        m.provenance.extra.as_ref().map(|e| e["judge"].clone()),
        Some(json!({"model": "claude-test-judge", "reason": "knowledge"})),
        "the judge is recorded in the provenance"
    );

    let batches = judge.batches();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 1);
    assert_eq!(
        batches[0][0].id, "c01",
        "the batch place, never the memory id"
    );
    assert_eq!(batches[0][0].text, DECISION);

    d.wait_embedded(&cwd);
    assert!(
        recalled(&d, &cwd, KEPT).contains(&m.id),
        "a kept capture is served"
    );

    let s = d.status(&cwd).judge;
    assert!(s.enabled && s.can_run, "{s:?}");
    assert_eq!(s.blocked, None);
    assert_eq!(s.model, "haiku");
    assert_eq!(s.command.as_deref(), Some("claude-test"));
    assert_eq!((s.pending, s.queued), (0, 0));
    assert_eq!((s.calls_today, s.judged_today), (1, 1));
    assert_eq!(s.max_calls_per_day, 200);
    assert!((s.cost_today_usd - 0.0001).abs() < 1e-9, "{s:?}");
    assert_eq!(s.last_error, None);

    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 1, WAIT, |_| true);
    assert_eq!(judged.len(), 1, "{judged:?}");
    assert_eq!(judged[0]["labels"]["verdict"], "keep");
    assert_eq!(judged[0]["labels"]["reason"], "knowledge");
    assert_eq!(judged[0]["labels"]["type"], "decision");
    assert_eq!(judged[0]["labels"]["spans"], 1);
    assert_eq!(judged[0]["labels"]["spans_dropped"], 0);
    let cost = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_COST_USD, 1, WAIT, |_| true);
    assert_eq!(cost.len(), 1, "{cost:?}");
    assert_eq!(cost[0]["labels"]["candidates"], 1);
    let pending = metric_lines(&rig, metrics::CAPTURE_PENDING);
    assert!(
        pending
            .iter()
            .any(|l| l["request_id"] == "judge" && l["value"] == 0.0),
        "{pending:?}"
    );
}

/// A rejected capture is kept pending for 30 days, marked rejected, never
/// served and never judged again; `status` counts it as rejected, not
/// pending; the journal records the verdict with its reason and no type
/// (B-190, which replaced B-167's deletion).
#[test]
fn s16_b190_a_rejected_capture_is_kept_unserved_and_journalled_with_its_reason() {
    let judge = ScriptedJudge::new(Box::new(|batch| answer(batch, |c| reject(c, "narration"))));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rejected");
    stop(&d, &cwd, DECISION);

    assert!(
        wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0),
        "the capture was judged"
    );
    let kept = project_memories(&store, &cwd);
    assert_eq!(kept.len(), 1, "a rejected capture is kept");
    assert_eq!(kept[0].status, Status::Pending, "and stays pending");
    assert_eq!(
        kept[0].content, DECISION,
        "with the text it was captured with"
    );
    let stats = store.judge_queue_stats().expect("queue stats");
    assert_eq!((stats.pending, stats.queued, stats.rejected), (0, 0, 1));
    let s = d.status(&cwd).judge;
    assert_eq!((s.pending, s.queued, s.rejected), (0, 0, 1), "{s:?}");
    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 1, WAIT, |_| true);
    assert_eq!(judged.len(), 1, "{judged:?}");
    assert_eq!(judged[0]["labels"]["verdict"], "reject");
    assert_eq!(judged[0]["labels"]["reason"], "narration");
    assert_eq!(judged[0]["labels"].get("type"), None);
    d.wait_embedded(&cwd);
    assert!(recalled(&d, &cwd, DECISION).is_empty());

    // Never judged again, however long the daemon runs.
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(judge.calls(), 1, "a rejected capture is not judged again");
}

// --- failures ------------------------------------------------------------------

/// A failed call leaves its captures pending and unserved, shows in
/// `status`, and is retried after a wait that doubles; the first answer
/// after that settles them.
#[test]
fn s16_a_failed_call_keeps_captures_pending_and_backs_off() {
    let judge = ScriptedJudge::new(failing(
        JudgeFailureKind::CallFailed,
        "Claude Code exited with status 1",
    ));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("failed");
    stop(&d, &cwd, DECISION);

    let s = backing_off(&d, &cwd);
    assert!(!s.can_run, "{s:?}");
    assert_eq!(s.blocked, Some(JudgeBlocked::BackingOff));
    assert!(s.retry_at.is_some(), "{s:?}");
    assert_eq!(
        s.last_error.as_deref(),
        Some("Claude Code exited with status 1")
    );
    assert_eq!((s.pending, s.queued), (1, 1));
    let held = wait_for(&store, &cwd, 1, Status::Pending);
    d.wait_embedded(&cwd);
    assert!(
        !recalled(&d, &cwd, DECISION).contains(&held[0].id),
        "a pending capture is not served"
    );

    assert!(
        wait_until(WAIT, || judge.calls() >= 3),
        "the judge was retried"
    );
    let times = judge.times();
    let first = times[1] - times[0];
    let second = times[2] - times[1];
    assert!(first >= Duration::from_millis(280), "{first:?}");
    assert!(
        second >= Duration::from_millis(580),
        "the wait doubles: {first:?} then {second:?}"
    );
    let failures = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_FAILURES_TOTAL, 3, WAIT, |_| {
        true
    });
    assert!(failures.len() >= 3, "{failures:?}");
    assert_eq!(failures[0]["labels"]["kind"], "call_failed");
    assert_eq!(failures[0]["labels"]["candidates"], 1);
    assert_eq!(
        store
            .judge_usage(&Timestamp::now().date_string())
            .expect("usage")
            .judged,
        0,
        "a failed call judged nothing"
    );

    judge.set(Box::new(|batch| {
        answer(batch, |c| keep(c, SPAN, "decision"))
    }));
    let kept = wait_for(&store, &cwd, 1, Status::Active);
    assert_eq!(kept[0].content, KEPT);
    let s = d.status(&cwd).judge;
    assert!(s.can_run, "{s:?}");
    assert_eq!(s.last_error, None);
}

/// A rate limit with a reset time is waited out until that time. The reset
/// is two seconds after the first call, whenever that call comes.
#[test]
fn s16_a_rate_limit_waits_until_its_reset_time() {
    let first: Arc<Mutex<Option<Timestamp>>> = Arc::default();
    let seen = Arc::clone(&first);
    let judge = ScriptedJudge::new(Box::new(move |_| {
        let reset = *seen
            .lock()
            .unwrap()
            .get_or_insert_with(|| Timestamp::now().add_micros(2_000_000));
        Err(JudgeFailure {
            kind: JudgeFailureKind::RateLimited,
            message: "rate limited".to_owned(),
            retry_at: Some(reset),
            cost_usd: 0.0,
        })
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let cwd = rig.project("limited");
    stop(&d, &cwd, DECISION);
    let s = backing_off(&d, &cwd);
    let reset = first.lock().unwrap().expect("the judge was called");
    assert_eq!(s.retry_at, Some(reset.to_rfc3339()));
    assert!(
        wait_until(WAIT, || judge.calls() >= 2),
        "called again after the reset"
    );
    let times = judge.times();
    assert!(
        times[1] - times[0] >= Duration::from_millis(1500),
        "the reset time is waited for, not the 300 ms backoff: {:?}",
        times[1] - times[0]
    );
}

/// A bad answer fails the whole batch: nothing in it is settled and each
/// capture counts an attempt. A capture that has failed is then retried
/// alone, so one bad candidate cannot hold back the others; after its third
/// attempt it leaves the queue and waits, pending, for the prune.
#[test]
fn s16_a_bad_answer_settles_nothing_and_the_third_ends_the_queue() {
    let judge = ScriptedJudge::new(Box::new(|batch: &[sm_daemon::JudgeCandidate]| {
        let mut verdicts: Vec<JudgeVerdict> = batch
            .iter()
            .map(|c| keep(c, (&c.text, &c.text), "decision"))
            .collect();
        if let Some(last) = verdicts.last_mut() {
            last.id = "c99".to_owned();
        }
        Ok(JudgeAnswer {
            verdicts,
            cost_usd: 0.0002,
            duration_ms: 5,
            model: None,
        })
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("bad");
    stop(&d, &cwd, &distinct_decision(1));
    stop(&d, &cwd, &distinct_decision(2));

    assert!(
        wait_until(WAIT, || judge.calls() >= 5
            && store.judge_queue_stats().is_ok_and(|s| s.queued == 0)),
        "both captures left the queue after {} calls",
        judge.calls()
    );
    std::thread::sleep(Duration::from_millis(1500));
    let batches = judge.batches();
    let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
    assert_eq!(sizes, [2, 1, 1, 1, 1], "together once, then each alone");
    for text in [distinct_decision(1), distinct_decision(2)] {
        let attempts = batches.iter().flatten().filter(|c| c.text == text).count();
        assert_eq!(attempts, 3, "three attempts at {text}");
    }
    wait_for(&store, &cwd, 2, Status::Pending);
    let s = d.status(&cwd).judge;
    assert_eq!((s.pending, s.queued), (2, 0));
    assert!(
        s.last_error.as_deref().is_some_and(|e| e.contains("c99")),
        "{s:?}"
    );
    let failures = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_FAILURES_TOTAL, 5, WAIT, |_| {
        true
    });
    assert_eq!(failures.len(), 5, "one line a call: {failures:?}");
    assert!(
        failures.iter().all(|l| l["labels"]["kind"] == "bad_answer"),
        "{failures:?}"
    );
}

/// A kept verdict none of whose spans is in the capture's text settles
/// nothing (B-173, B-191): the capture stays pending and unserved, is journalled
/// with the verdict `unquoted`, and counts an attempt, so after its third it
/// leaves the queue and waits, pending, for the prune. It is not a failed
/// call: the judge does not back off.
#[test]
fn s16_a_kept_capture_whose_spans_are_not_in_its_text_stays_pending() {
    let judge = ScriptedJudge::new(Box::new(|batch| {
        answer(batch, |c| {
            keep(
                c,
                (
                    "The uploader retries a failed upload",
                    "at most five times.",
                ),
                "decision",
            )
        })
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("unquoted");
    stop(&d, &cwd, DECISION);

    assert!(
        wait_until(WAIT, || judge.calls() >= 3
            && store.judge_queue_stats().is_ok_and(|s| s.queued == 0)),
        "the capture left the queue after {} calls",
        judge.calls()
    );
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(judge.calls(), 3, "three attempts, then no more");
    let held = wait_for(&store, &cwd, 1, Status::Pending);
    assert_eq!(held[0].content, DECISION, "the capture is not changed");
    d.wait_embedded(&cwd);
    assert!(
        !recalled(&d, &cwd, DECISION).contains(&held[0].id),
        "an unsettled capture is not served"
    );
    let s = d.status(&cwd).judge;
    assert_eq!((s.pending, s.queued), (1, 0));
    assert_eq!(s.blocked, None, "{s:?}");
    assert_eq!(s.last_error, None, "{s:?}");

    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 3, WAIT, |_| true);
    assert_eq!(judged.len(), 3, "{judged:?}");
    for l in &judged {
        assert_eq!(l["labels"]["verdict"], "unquoted", "{l:?}");
        assert_eq!(l["labels"]["reason"], "knowledge", "{l:?}");
        assert_eq!(l["labels"]["spans"], 1, "{l:?}");
        assert_eq!(l["labels"]["spans_dropped"], 1, "{l:?}");
    }
    assert!(metric_lines(&rig, metrics::CAPTURE_JUDGE_FAILURES_TOTAL).is_empty());
}

// --- the cap and an unavailable judge ------------------------------------------

/// When the day's calls are used, captures stay pending and `status` says
/// so, with the time the cap resets.
#[test]
fn s16_the_daily_cap_leaves_captures_pending() {
    let judge = ScriptedJudge::new(keep_all("decision"));
    let mut rig = Rig::new();
    rig.settings.capture.judge.max_calls_per_day = 1;
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("capped");
    stop(&d, &cwd, &distinct_decision(1));
    wait_for(&store, &cwd, 1, Status::Active);
    stop(&d, &cwd, &distinct_decision(2));
    assert!(wait_until(WAIT, || pending_count(&store) == 1));
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(judge.calls(), 1, "no call past the cap");
    assert_eq!(pending_count(&store), 1);
    let s = d.status(&cwd).judge;
    assert!(!s.can_run);
    assert_eq!(s.blocked, Some(JudgeBlocked::DailyCap));
    assert_eq!((s.calls_today, s.max_calls_per_day), (1, 1));
    let reset = s.retry_at.expect("the reset time");
    assert!(reset.ends_with("T00:00:00Z"), "{reset}");
    assert!(reset > Timestamp::now().to_rfc3339(), "{reset}");
}

/// B-193: once the day's calls have cost `capture.judge.max_usd_per_day`,
/// no batch starts: captures stay pending and `status` says so, with the
/// cap and the time it resets.
#[test]
fn s16_b193_the_spending_cap_leaves_captures_pending() {
    let judge = ScriptedJudge::new(keep_all("decision"));
    let mut rig = Rig::new();
    rig.settings.capture.judge.max_usd_per_day = 0.00005;
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("spent");
    stop(&d, &cwd, &distinct_decision(1));
    wait_for(&store, &cwd, 1, Status::Active);
    stop(&d, &cwd, &distinct_decision(2));
    assert!(wait_until(WAIT, || pending_count(&store) == 1));
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(judge.calls(), 1, "no batch past the spending cap");
    assert_eq!(pending_count(&store), 1);
    let s = d.status(&cwd).judge;
    assert!(!s.can_run);
    assert_eq!(s.blocked, Some(JudgeBlocked::DailyBudget), "{s:?}");
    assert_eq!(s.max_usd_per_day, Some(0.00005));
    assert!(s.cost_today_usd >= 0.00005, "{s:?}");
    assert!(
        s.calls_today < s.max_calls_per_day,
        "the call cap is not what stops it"
    );
    let reset = s.retry_at.expect("the reset time");
    assert!(reset.ends_with("T00:00:00Z"), "{reset}");
    assert!(reset > Timestamp::now().to_rfc3339(), "{reset}");
}

/// B-193: the spending cap is checked before each batch, so a batch started
/// under it is finished, its duplicate check (B-192) too, and the day ends
/// over the cap by that batch's cost; the next batch waits.
#[test]
fn s16_b193_a_batch_started_under_the_cap_is_finished_with_its_duplicate_check() {
    assert_neighbours(RESTATED, KEPT);
    let judge = ScriptedJudge::new(keep_restated());
    judge.set_repeats(nothing_new());
    let mut rig = Rig::new();
    rig.settings.capture.judge.max_usd_per_day = 0.00005;
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("finished");
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);

    assert!(wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0));
    assert!(wait_until(WAIT, || judge.checks().len() == 1));
    assert!(wait_until(WAIT, || {
        store.judge_queue_stats().expect("stats").rejected == 1
    }));
    let u = usage(&store);
    assert_eq!((u.calls, u.judged), (2, 1), "{u:?}");
    assert!(
        (u.cost_usd - 0.0002).abs() < 1e-9,
        "over the cap by one batch: {u:?}"
    );

    stop(&d, &cwd, &distinct_decision(3));
    assert!(wait_until(WAIT, || pending_count(&store) == 1));
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(judge.calls(), 1, "the next batch waits for the next day");
    assert_eq!(judge.checks().len(), 1);
    assert_eq!(
        d.status(&cwd).judge.blocked,
        Some(JudgeBlocked::DailyBudget)
    );
}

/// Without a judge, with the judge turned off, or with Claude Code not
/// found, captures wait as pending, no call is made, and `status` names the
/// reason.
#[test]
fn s16_captures_wait_when_the_judge_cannot_run_and_status_says_why() {
    // No judge.
    {
        let rig = Rig::new();
        let d = rig.start_fake();
        let store = rig.store();
        let cwd = rig.project("none");
        stop(&d, &cwd, DECISION);
        wait_for(&store, &cwd, 1, Status::Pending);
        let s = d.status(&cwd).judge;
        assert!(!s.can_run);
        assert_eq!(s.blocked, Some(JudgeBlocked::NoJudge));
        assert_eq!(s.pending, 1);
    }
    // Turned off.
    {
        let judge = ScriptedJudge::new(keep_all("decision"));
        let mut rig = Rig::new();
        rig.settings.capture.judge.enabled = false;
        rig.use_judge(judge.clone());
        let d = rig.start_fake();
        let store = rig.store();
        let cwd = rig.project("off");
        stop(&d, &cwd, DECISION);
        wait_for(&store, &cwd, 1, Status::Pending);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(judge.calls(), 0);
        let s = d.status(&cwd).judge;
        assert!(!s.enabled && !s.can_run);
        assert_eq!(s.blocked, Some(JudgeBlocked::Disabled));
    }
    // Claude Code not found.
    {
        let judge = ScriptedJudge::new(keep_all("decision"));
        judge.set_command(None);
        let mut rig = Rig::new();
        rig.use_judge(judge.clone());
        let d = rig.start_fake();
        let store = rig.store();
        let cwd = rig.project("missing");
        stop(&d, &cwd, DECISION);
        wait_for(&store, &cwd, 1, Status::Pending);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(judge.calls(), 0);
        let s = d.status(&cwd).judge;
        assert_eq!(s.blocked, Some(JudgeBlocked::Unavailable));
        assert_eq!(s.command, None);
        assert!(
            s.last_error
                .as_deref()
                .is_some_and(|e| e.contains("Claude Code was not found")),
            "{s:?}"
        );
        // Found again: the queue is judged.
        judge.set_command(Some("claude-test"));
        wait_for(&store, &cwd, 1, Status::Active);
    }
}

// --- near duplicates -------------------------------------------------------------

/// A kept memory within the near-duplicate cosine of an active memory of
/// its project, or of a memory kept before it in the same batch, is not
/// made active (step 12): the capture is rejected as a near duplicate and
/// kept pending, unserved, for 30 days (B-190). Each capture holds a
/// decision of its own words and the shared decision [`KEPT`]; the judge
/// marks only the shared one, so the captures are far apart and their kept
/// memories the same. The journal records the drop at the judge stage.
#[test]
fn s16_b190_a_kept_memory_near_an_active_memory_is_not_made_active() {
    let judge = ScriptedJudge::new(Box::new(|batch| {
        answer(batch, |c| keep(c, SPAN, "decision"))
    }));
    let with_kept = |n: u64| format!("{} {KEPT}", distinct_decision(n));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();

    // An active memory with the text the judge keeps.
    let cwd = rig.project("near");
    let active = remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, &with_kept(10));
    assert!(wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0));
    let left = project_memories(&store, &cwd);
    let active_left: Vec<_> = left.iter().filter(|m| m.status == Status::Active).collect();
    assert_eq!(active_left.len(), 1, "{left:?}");
    assert_eq!(active_left[0].id, active);
    assert_eq!(
        left.len(),
        2,
        "the near duplicate is kept, rejected: {left:?}"
    );
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);

    // Two captures in one batch, kept with the same sentence.
    let cwd2 = rig.project("near-batch");
    stop(&d, &cwd2, &with_kept(1));
    stop(&d, &cwd2, &with_kept(2));
    assert!(wait_until(WAIT, || judge.calls() == 2 && pending_count(&store) == 0));
    assert_eq!(judge.batches()[1].len(), 2, "one batch");
    let in_batch = project_memories(&store, &cwd2);
    let kept: Vec<_> = in_batch
        .iter()
        .filter(|m| m.status == Status::Active)
        .collect();
    assert_eq!(kept.len(), 1, "{in_batch:?}");
    assert_eq!(kept[0].content, KEPT);
    assert_eq!(
        in_batch.len(),
        2,
        "the second is kept, rejected: {in_batch:?}"
    );
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 2);

    let drops = wait_metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL, 2, WAIT, |l| {
        l["labels"]["stage"] == "judge"
    });
    assert_eq!(drops.len(), 2, "{drops:?}");
    for l in &drops {
        assert_eq!(l["labels"]["reason"], "near_duplicate");
        assert!(
            l["labels"]["cosine"].as_f64().is_some_and(|c| c >= 0.90),
            "{l:?}"
        );
    }
}

// --- redaction (step 16) --------------------------------------------------------

/// Text is redacted before it reaches the judge: a capture's text through
/// the hook, and a capture already stored with a secret in its text or its
/// context (one written before a scanner rule existed). The memory is made
/// of the redacted text's own sentences, so it holds the redaction marks;
/// a quote of the judge's that is not in that text, here one holding a
/// secret, is dropped and never stored (B-173).
#[test]
fn s9_text_is_redacted_before_it_reaches_the_judge_and_after_it_answers() {
    let aws = format!("AKIA{}", "Q7".repeat(8));
    let anthropic = format!("sk-ant-{}", "a1B2".repeat(8));
    let github = format!("ghp_{}", "Zx9".repeat(12));
    let script_secret = format!("The deploy key is {github}, kept in the vault.");
    let judge = ScriptedJudge::new(Box::new(move |batch| {
        answer(batch, |c| {
            keep_spans(
                c,
                &[
                    (c.text.as_str(), c.text.as_str()),
                    (script_secret.as_str(), script_secret.as_str()),
                ],
                "note",
            )
        })
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let cwd = rig.project("secrets");

    // A capture stored with secrets in its text and context, before the
    // daemon starts.
    {
        let store = rig.store();
        let identity = resolve_project(Path::new(&cwd)).expect("a project");
        let project = store.upsert_project(&identity).expect("the project");
        let text = format!(
            "We decided the importer signs requests with {anthropic}, because the API needs it."
        );
        let mut memory = NewMemory::new(
            project.id,
            text.clone(),
            Provenance::new(ProvenanceSource::Capture),
        );
        memory.status = Status::Pending;
        let piece = ChunkPiece {
            seq: 0,
            text: text.clone(),
            token_count: 12,
        };
        store
            .insert_captures(vec![(
                MemoryInput {
                    memory,
                    chunks: vec![NewChunk::pending(piece)],
                },
                format!("Use the key {aws} for the import."),
            )])
            .expect("store a capture");
    }
    let d = rig.start_fake();
    let store = rig.store();
    assert!(wait_until(WAIT, || judge.calls() >= 1 && pending_count(&store) == 0));

    // A capture through the hook.
    let cwd2 = rig.project("secrets-hook");
    stop(
        &d,
        &cwd2,
        &format!("We decided to rotate {aws} monthly, because the audit requires it."),
    );
    assert!(wait_until(WAIT, || judge.calls() >= 2 && pending_count(&store) == 0));

    let seen: Vec<String> = judge
        .batches()
        .iter()
        .flatten()
        .flat_map(|c| [c.text.clone(), c.context.clone()])
        .collect();
    assert!(seen.iter().any(|t| t.contains("[REDACTED:")), "{seen:?}");
    for t in &seen {
        assert!(!t.contains(&aws), "the judge saw a secret: {t}");
        assert!(!t.contains(&anthropic), "the judge saw a secret: {t}");
    }

    let stored: Vec<Memory> = rig
        .memories()
        .into_iter()
        .filter(|m| m.status == Status::Active)
        .collect();
    assert_eq!(stored.len(), 2, "{stored:?}");
    for m in &stored {
        assert!(
            !m.content.contains(&github),
            "a quote not in the text was stored: {}",
            m.content
        );
        assert!(!m.content.contains("deploy key"), "{}", m.content);
        assert!(m.content.contains("[REDACTED:"), "{}", m.content);
        assert!(!m.content.contains(&aws) && !m.content.contains(&anthropic));
    }
    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 2, WAIT, |_| true);
    assert_eq!(judged.len(), 2, "{judged:?}");
    for l in &judged {
        assert_eq!(l["labels"]["verdict"], "keep", "{l:?}");
        assert_eq!(l["labels"]["spans"], 2, "{l:?}");
        assert_eq!(l["labels"]["spans_dropped"], 1, "{l:?}");
    }
}

// --- outside the hook's time, one call at a time ---------------------------------

/// The judge runs on its own thread: while a call is held open, captures
/// are still accepted at once and `status` answers. Calls never overlap and
/// a batch holds at most 25 captures.
#[test]
fn s16_the_judge_runs_outside_the_hooks_time_one_call_at_a_time() {
    let judge = ScriptedJudge::new(keep_all("decision"));
    judge.hold();
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("held");
    stop(&d, &cwd, &distinct_decision(0));
    assert!(wait_until(WAIT, || judge.in_call()), "a call is open");

    for n in 1..=40 {
        let start = Instant::now();
        stop(&d, &cwd, &distinct_decision(n));
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "a capture waited for the judge: {:?}",
            start.elapsed()
        );
    }
    let s = d.status(&cwd).judge;
    assert!(s.can_run, "{s:?}");
    assert!(
        wait_until(WAIT, || pending_count(&store) == 41),
        "41 pending: {}, drops {:?}",
        pending_count(&store),
        metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL)
    );
    assert_eq!(judge.calls(), 1, "no second call while one is open");

    judge.release();
    wait_for(&store, &cwd, 41, Status::Active);
    assert_eq!(judge.max_in_flight(), 1, "calls never overlap");
    let sizes: Vec<usize> = judge.batches().iter().map(Vec::len).collect();
    assert!(sizes.iter().all(|&n| n <= 25), "{sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 41, "{sizes:?}");
}

// --- B-181: status never shows a pending capture ----------------------------

/// B-181 (SPEC 9, 16; CAPTURE-RCA B4): `status` never shows a capture that
/// waits for the judge, neither its id nor its text, only counts. Here the
/// judge's answer copies the capture's text into a verdict's id, so the
/// check refuses the answer and its error becomes the judge's last error;
/// that error names no value the answer wrote.
#[test]
fn s16_b181_status_never_shows_a_pending_capture() {
    let judge = ScriptedJudge::new(Box::new(|batch| {
        answer(batch, |c| {
            let mut v = keep(c, SPAN, "decision");
            v.id = c.text.clone();
            v
        })
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("status");
    stop(&d, &cwd, DECISION);
    let stored = wait_for(&store, &cwd, 1, Status::Pending);

    let mut last = None;
    let ok = wait_until(WAIT, || {
        let s = d.status(&cwd);
        let done = s.judge.last_error.is_some();
        last = Some(s);
        done
    });
    assert!(
        ok,
        "the failed call is recorded: {:?}",
        last.as_ref().map(|s| &s.judge)
    );
    let s = last.expect("a status");
    assert_eq!(s.judge.pending, 1, "{:?}", s.judge);
    let shown = serde_json::to_string(&s).expect("the status as JSON");
    assert!(!shown.contains(&stored[0].id.to_string()), "{shown}");
    for word in ["uploader", "retries", "throttles"] {
        assert!(!shown.contains(word), "{word}: {shown}");
    }
}

// --- B-190: no reply shows a rejected capture --------------------------------

/// B-190 (SPEC 6, 9, 16): a capture the judge rejected is kept pending, so
/// every B-157 and B-181 guard holds for it. `status` counts it and shows
/// neither its id nor its text; recall of any status never returns it;
/// `related` and `link` refuse it; `forget` deletes it but only counts it.
#[test]
fn s16_b190_no_reply_shows_a_rejected_capture() {
    use sm_core::error::ErrorCode;
    use sm_core::protocol::{ForgetParams, LinkAction, LinkParams, RelatedParams};
    use sm_core::types::LinkEnd;

    let judge = ScriptedJudge::new(Box::new(|batch| answer(batch, |c| reject(c, "narration"))));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("rejected-replies");
    stop(&d, &cwd, DECISION);
    assert!(
        wait_until(WAIT, || judge.calls() == 1
            && store.judge_queue_stats().expect("stats").rejected == 1),
        "the capture was rejected"
    );
    let rejected = project_memories(&store, &cwd)[0].id;
    let other = remember(&d, &cwd, "The scheduler starts the uploader every night.");
    d.wait_embedded(&cwd);

    let s = d.status(&cwd);
    assert_eq!(s.judge.rejected, 1, "{:?}", s.judge);
    let shown = serde_json::to_string(&s).expect("the status as JSON");
    assert!(!shown.contains(&rejected.to_string()), "{shown}");
    for word in ["uploader", "retries", "throttles"] {
        assert!(!shown.contains(word), "{word}: {shown}");
    }

    let mut p = RecallParams::new(DECISION);
    p.status = vec![Status::Active, Status::Superseded, Status::Retracted];
    let Reply::Recall(r) = d.ok(&cwd, Op::Recall(p)) else {
        panic!("recall replied with another type");
    };
    assert!(r.results.iter().all(|m| m.id != rejected));

    let refused = |e: sm_core::error::CoreError| {
        assert_code(&e, ErrorCode::InvalidInput);
        assert!(e.message.contains("waits for the judge"), "{}", e.message);
        assert!(!e.message.contains("uploader"), "{}", e.message);
    };
    refused(d.err(
        &cwd,
        Op::Related(RelatedParams {
            id: Some(rejected),
            anchor: None,
            k: None,
            scope: None,
            limit: None,
        }),
    ));
    refused(d.err(
        &cwd,
        Op::Link(LinkParams {
            action: LinkAction::Add,
            src: Some(LinkEnd::Memory(other)),
            dst: Some(LinkEnd::Memory(rejected)),
            relation: Some(sm_core::types::Relation::RelatesTo),
            end: None,
            limit: None,
        }),
    ));
    refused(d.err(
        &cwd,
        Op::Link(LinkParams {
            action: LinkAction::List,
            src: None,
            dst: None,
            relation: None,
            end: Some(LinkEnd::Memory(rejected)),
            limit: None,
        }),
    ));

    let Reply::Forget(f) = d.ok(
        &cwd,
        Op::Forget(ForgetParams {
            ids: vec![rejected],
            filter: None,
            confirm: true,
        }),
    ) else {
        panic!("forget replied with another type");
    };
    let shown = serde_json::to_string(&f).expect("the reply as JSON");
    assert!(f.deleted.is_empty(), "{shown}");
    assert_eq!(f.pending, 1, "{shown}");
    assert!(!shown.contains(&rejected.to_string()), "{shown}");
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 0);
}

// --- the duplicate check (B-192) ---------------------------------------------

/// A Stop message whose decision restates [`KEPT`] in other words.
const RESTATED_MESSAGE: &str = "I have looked at the uploader again. The uploader's retries are capped at five, because the server throttles bursts, as we decided.";
/// The span the scripted judge marks in it.
const RESTATED_SPAN: (&str, &str) = ("The uploader's retries are capped", "as we decided.");
/// What is kept of it.
const RESTATED: &str = "The uploader's retries are capped at five, because the server throttles bursts, as we decided.";

/// Keeps [`RESTATED`] from [`RESTATED_MESSAGE`], [`KEPT`] from [`DECISION`]
/// and any other candidate whole, all as decisions.
fn keep_restated() -> Script {
    Box::new(|batch| {
        answer(batch, |c| {
            if c.text == RESTATED_MESSAGE {
                keep(c, RESTATED_SPAN, "decision")
            } else if c.text == DECISION {
                keep(c, SPAN, "decision")
            } else {
                keep(c, (&c.text, &c.text), "decision")
            }
        })
    })
}

/// The fake embedder's cosine of two texts, as the daemon computes it for a
/// one-chunk memory.
fn fake_cosine(a: &str, b: &str) -> f32 {
    sm_core::testing::dot(
        &sm_core::testing::fake_vector(a),
        &sm_core::testing::fake_vector(b),
    )
}

/// The texts of these tests are a neighbour of each other, at least 0.70,
/// and not a near duplicate, below 0.90 (B-192).
fn assert_neighbours(a: &str, b: &str) {
    let c = fake_cosine(a, b);
    assert!((0.70..0.90).contains(&c), "{c}: {a:?} and {b:?}");
}

fn usage(store: &Store) -> sm_store::JudgeUsage {
    store
        .judge_usage(&Timestamp::now().date_string())
        .expect("usage")
}

/// B-192: a kept memory whose neighbour, an active memory of its project,
/// already says everything it says is not made active: the duplicate check
/// is asked about it with the neighbour's text, and on its answer the
/// capture is rejected with the reason `repeats`, kept pending, unserved
/// and journalled. The check is counted as a call and its cost journalled.
#[test]
fn s16_b192_a_kept_memory_that_repeats_an_active_one_is_rejected_unserved() {
    assert_neighbours(RESTATED, KEPT);
    let judge = ScriptedJudge::new(keep_restated());
    judge.set_repeats(nothing_new());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("repeats");
    let active = remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);

    assert!(wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0));
    let checks = judge.checks();
    assert_eq!(checks.len(), 1, "{checks:?}");
    assert_eq!(checks[0].len(), 1, "{checks:?}");
    let item = &checks[0][0];
    assert_eq!((item.id.as_str(), item.text.as_str()), ("d01", RESTATED));
    let kept: Vec<(&str, &str)> = item
        .kept
        .iter()
        .map(|k| (k.id.as_str(), k.text.as_str()))
        .collect();
    assert_eq!(kept, vec![("k1", KEPT)]);

    let left = project_memories(&store, &cwd);
    assert_eq!(left.len(), 2, "{left:?}");
    let capture = left.iter().find(|m| m.id != active).expect("the capture");
    assert_eq!(capture.status, Status::Pending, "a repeat stays pending");
    assert_eq!(
        left.iter().find(|m| m.id == active).map(|m| m.status),
        Some(Status::Active),
        "the active memory is not changed"
    );
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);
    let s = d.status(&cwd).judge;
    assert_eq!((s.pending, s.queued, s.rejected), (0, 0, 1), "{s:?}");

    let drops = wait_metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL, 1, WAIT, |l| {
        l["labels"]["stage"] == "judge"
    });
    assert_eq!(drops.len(), 1, "{drops:?}");
    assert_eq!(drops[0]["labels"]["reason"], "repeats");
    let costs = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_COST_USD, 2, WAIT, |_| true);
    assert_eq!(costs.len(), 2, "{costs:?}");
    assert!(
        costs
            .iter()
            .any(|l| l["labels"]["call"] == "repeats" && l["labels"]["candidates"] == 1),
        "{costs:?}"
    );
    let u = usage(&store);
    assert_eq!(
        (u.calls, u.judged, u.failures),
        (2, 1, 0),
        "the check counts against the daily cap: {u:?}"
    );
    assert!((u.cost_usd - 0.0002).abs() < 1e-9, "{u:?}");

    d.wait_embedded(&cwd);
    let served = recalled(&d, &cwd, RESTATED);
    assert!(served.contains(&active), "{served:?}");
    assert!(!served.contains(&capture.id), "a repeat is never served");
}

/// B-192: a kept memory the duplicate check finds anything new in is made
/// active, whole: the sentences the judge's span found, not only the new
/// ones.
#[test]
fn s16_b192_a_kept_memory_with_anything_new_is_made_active_whole() {
    assert_neighbours(RESTATED, KEPT);
    let judge = ScriptedJudge::new(keep_restated());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("adds");
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);

    let left = wait_for(&store, &cwd, 2, Status::Active);
    assert!(left.iter().any(|m| m.content == RESTATED), "{left:?}");
    assert_eq!(judge.checks().len(), 1, "it was checked");
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 0);
}

/// B-192: a kept memory with no neighbour is made active without a check:
/// an active memory of other words is not a neighbour, and an alike one in
/// another project is not either.
#[test]
fn s16_b192_a_kept_memory_with_no_neighbour_is_not_checked() {
    let far = distinct_decision(3);
    assert!(fake_cosine(KEPT, &far) < 0.70);
    let judge = ScriptedJudge::new(keep_restated());
    judge.set_repeats(nothing_new());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let d = rig.start_fake();
    let store = rig.store();
    let other = rig.project("other");
    remember(&d, &other, RESTATED);
    d.wait_embedded(&other);
    let cwd = rig.project("alone");
    remember(&d, &cwd, &far);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, DECISION);

    let left = wait_for(&store, &cwd, 2, Status::Active);
    assert!(left.iter().any(|m| m.content == KEPT), "{left:?}");
    assert!(judge.checks().is_empty(), "{:?}", judge.checks());
    assert_eq!(usage(&store).calls, 1);
}

/// B-192: a kept memory that repeats one kept earlier in its own batch, for
/// the same project, is checked against it and rejected; the earlier one,
/// with no neighbour, is made active unchecked.
#[test]
fn s16_b192_a_kept_memory_that_repeats_one_kept_earlier_in_its_batch_is_rejected() {
    assert_neighbours(RESTATED, KEPT);
    let judge = ScriptedJudge::new(keep_restated());
    judge.set_repeats(nothing_new());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("batch-repeats");
    stop(&d, &cwd, DECISION);
    stop(&d, &cwd, RESTATED_MESSAGE);

    assert!(wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0));
    assert_eq!(judge.batches()[0].len(), 2, "one batch");
    let left = project_memories(&store, &cwd);
    let active: Vec<&Memory> = left.iter().filter(|m| m.status == Status::Active).collect();
    assert_eq!(active.len(), 1, "{left:?}");
    let checks = judge.checks();
    assert_eq!(checks.len(), 1, "{checks:?}");
    assert_eq!(
        checks[0].len(),
        1,
        "only the later one is asked: {checks:?}"
    );
    let item = &checks[0][0];
    assert_ne!(item.text, active[0].content);
    assert_eq!(item.kept.len(), 1, "{item:?}");
    assert_eq!(item.kept[0].text, active[0].content);
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);
}

/// [`RESTATED`] with one sentence added, in a Stop message whose other
/// sentence keeps it below the capture's near-duplicate cosine with
/// [`RESTATED_MESSAGE`].
const AMENDED_MESSAGE: &str = "Next I checked the queue handling in the worker module today. The uploader's retries are capped at five, because the server throttles bursts, as we decided. The delay doubles.";
const AMENDED_SPAN: (&str, &str) = ("The uploader's retries are capped", "The delay doubles.");
const AMENDED: &str = "The uploader's retries are capped at five, because the server throttles bursts, as we decided. The delay doubles.";

/// Another wording of [`RESTATED`]: a neighbour of it, and not of [`KEPT`].
const REWORDED_MESSAGE: &str = "I have looked at the uploader again. The uploader's retries are capped at five because bursts are throttled, as decided, and logged.";
const REWORDED_SPAN: (&str, &str) = ("The uploader's retries are capped", "and logged.");
const REWORDED: &str = "The uploader's retries are capped at five because bursts are throttled, as decided, and logged.";

/// Whether the project holds an active memory of `content`.
fn made_active(store: &Store, cwd: &str, content: &str) -> bool {
    project_memories(store, cwd)
        .iter()
        .any(|m| m.content == content && m.status == Status::Active)
}

/// Keeps each message of `spans` by its span and any other candidate whole,
/// all as decisions.
fn keep_by(spans: &'static [(&'static str, (&'static str, &'static str))]) -> Script {
    Box::new(move |batch| {
        answer(batch, |c| match spans.iter().find(|(m, _)| c.text == *m) {
            Some((_, span)) => keep(c, *span, "decision"),
            None => keep(c, (&c.text, &c.text), "decision"),
        })
    })
}

/// C12 review finding 3.1 (B-196): a memory kept earlier in the batch and
/// then rejected as a repeat is not made active, so it makes no later one
/// a near duplicate. The later one, never checked against the active
/// memories it is now compared with, is judged again alone: it stays
/// pending, counts an attempt and is journalled with the verdict
/// `recheck`; the second judgement makes it active.
#[test]
fn s16_b196_a_batch_member_rejected_as_a_repeat_makes_no_later_one_a_near_duplicate() {
    assert_neighbours(RESTATED, KEPT);
    assert_neighbours(AMENDED, KEPT);
    assert!(fake_cosine(AMENDED, RESTATED) >= 0.90);
    let judge = ScriptedJudge::new(keep_by(&[
        (RESTATED_MESSAGE, RESTATED_SPAN),
        (AMENDED_MESSAGE, AMENDED_SPAN),
    ]));
    judge.set_repeats(Box::new(|items| {
        repeat_answer(items, |item| item.text != RESTATED)
    }));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("batch-recheck");
    let active = remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);
    stop(&d, &cwd, AMENDED_MESSAGE);

    assert!(
        wait_until(WAIT, || made_active(&store, &cwd, AMENDED)),
        "{:?}",
        project_memories(&store, &cwd)
    );
    let left = project_memories(&store, &cwd);
    assert!(
        left.iter()
            .any(|m| m.id == active && m.status == Status::Active),
        "{left:?}"
    );
    assert_eq!(pending_count(&store), 0);
    let sizes: Vec<usize> = judge.batches().iter().map(Vec::len).collect();
    assert_eq!(sizes, [2, 1], "judged again alone");
    let checks: Vec<Vec<(String, Vec<String>)>> = judge
        .checks()
        .iter()
        .map(|c| {
            c.iter()
                .map(|i| {
                    (
                        i.text.clone(),
                        i.kept.iter().map(|k| k.text.clone()).collect(),
                    )
                })
                .collect()
        })
        .collect();
    assert_eq!(
        checks,
        [
            vec![(RESTATED.to_owned(), vec![KEPT.to_owned()])],
            vec![(AMENDED.to_owned(), vec![KEPT.to_owned()])],
        ]
    );
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);
    let judged = wait_metric_lines(&rig, metrics::CAPTURE_JUDGED_TOTAL, 3, WAIT, |_| true);
    let mut verdicts: Vec<String> = judged
        .iter()
        .map(|l| l["labels"]["verdict"].as_str().unwrap_or("").to_owned())
        .collect();
    verdicts.sort();
    assert_eq!(verdicts, ["keep", "keep", "recheck"], "{judged:?}");
    let drops = wait_metric_lines(&rig, metrics::CAPTURE_DROPPED_TOTAL, 1, WAIT, |l| {
        l["labels"]["stage"] == "judge"
    });
    let reasons: Vec<&str> = drops
        .iter()
        .map(|l| l["labels"]["reason"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(reasons, ["repeats"], "{drops:?}");
}

/// C12 review finding 3.1 (B-196): a memory the duplicate check compared
/// only with one kept earlier in its batch, which was then rejected as a
/// repeat, is not rejected on that comparison: with no neighbour left it is
/// made active.
#[test]
fn s16_b196_a_batch_member_rejected_as_a_repeat_condemns_no_later_one() {
    assert_neighbours(RESTATED, KEPT);
    assert_neighbours(REWORDED, RESTATED);
    assert!(fake_cosine(REWORDED, KEPT) < 0.70);
    let judge = ScriptedJudge::new(keep_by(&[
        (RESTATED_MESSAGE, RESTATED_SPAN),
        (REWORDED_MESSAGE, REWORDED_SPAN),
    ]));
    judge.set_repeats(nothing_new());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("batch-condemns");
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);
    stop(&d, &cwd, REWORDED_MESSAGE);

    assert!(
        wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0),
        "{:?}",
        project_memories(&store, &cwd)
    );
    assert!(
        made_active(&store, &cwd, REWORDED),
        "{:?}",
        project_memories(&store, &cwd)
    );
    assert_eq!(judge.calls(), 1, "one batch, settled at once");
    assert_eq!(judge.batches()[0].len(), 2, "one batch");
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 1);
    assert_eq!(pending_count(&store), 0);
}

/// B-192, SPEC 9: an active neighbour's text is redacted before the
/// duplicate check sees it, here one stored before a scanner rule existed.
#[test]
fn s9_b192_a_neighbours_text_is_redacted_before_the_duplicate_check() {
    let aws = format!("AKIA{}", "Q7".repeat(8));
    let secret_kept = format!("{KEPT} Its key is {aws}.");
    assert_neighbours(RESTATED, &secret_kept);
    let judge = ScriptedJudge::new(keep_restated());
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    let cwd = rig.project("neighbour-secret");
    {
        let store = rig.store();
        let identity = resolve_project(Path::new(&cwd)).expect("a project");
        let project = store.upsert_project(&identity).expect("the project");
        let memory = NewMemory::new(
            project.id,
            secret_kept.clone(),
            Provenance::new(ProvenanceSource::Agent),
        );
        let piece = ChunkPiece {
            seq: 0,
            text: secret_kept.clone(),
            token_count: 20,
        };
        store
            .insert_memory(
                MemoryInput {
                    memory,
                    chunks: vec![NewChunk::pending(piece)],
                },
                sm_store::Access::Local,
            )
            .expect("store a memory");
    }
    let d = rig.start_fake();
    let store = rig.store();
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);

    assert!(wait_until(WAIT, || judge.calls() == 1 && pending_count(&store) == 0));
    let checks = judge.checks();
    assert_eq!(checks.len(), 1, "{checks:?}");
    let shown = &checks[0][0].kept[0].text;
    assert!(!shown.contains(&aws), "the check saw a secret: {shown}");
    assert!(shown.contains("[REDACTED:"), "{shown}");
}

/// B-192: a failed duplicate check settles nothing: every capture of the
/// batch stays pending and unserved, the one that needed no check too, the
/// failure shows in `status`, and the wait before the next try doubles as
/// for any failed call. The first answer after that settles them.
#[test]
fn s16_b192_a_failed_duplicate_check_leaves_the_whole_batch_pending() {
    assert_neighbours(RESTATED, KEPT);
    let far = distinct_decision(5);
    assert!(fake_cosine(&far, KEPT) < 0.70 && fake_cosine(&far, RESTATED) < 0.70);
    let judge = ScriptedJudge::new(keep_restated());
    judge.set_repeats(failing_repeats(
        JudgeFailureKind::CallFailed,
        "Claude Code exited with status 1",
    ));
    let mut rig = Rig::new();
    rig.use_judge(judge.clone());
    rig.opts.judge_wait = Duration::from_millis(400);
    let d = rig.start_fake();
    let store = rig.store();
    let cwd = rig.project("check-fails");
    remember(&d, &cwd, KEPT);
    d.wait_embedded(&cwd);
    stop(&d, &cwd, RESTATED_MESSAGE);
    stop(&d, &cwd, &far);

    let s = backing_off(&d, &cwd);
    assert_eq!(
        s.last_error.as_deref(),
        Some("Claude Code exited with status 1")
    );
    assert_eq!((s.pending, s.queued, s.rejected), (2, 2, 0), "{s:?}");
    let left = project_memories(&store, &cwd);
    assert_eq!(
        left.iter().filter(|m| m.status == Status::Pending).count(),
        2,
        "{left:?}"
    );
    d.wait_embedded(&cwd);
    let served = recalled(&d, &cwd, &far);
    assert!(
        left.iter()
            .filter(|m| m.status == Status::Pending)
            .all(|m| !served.contains(&m.id)),
        "a pending capture is not served"
    );

    assert!(
        wait_until(WAIT, || judge.calls() >= 3),
        "the batch was tried again"
    );
    assert_eq!(judge.batches()[0].len(), 2, "one batch");
    let times = judge.times();
    let first = times[1] - times[0];
    let second = times[2] - times[1];
    assert!(first >= Duration::from_millis(280), "{first:?}");
    assert!(
        second >= Duration::from_millis(580),
        "the wait doubles: {first:?} then {second:?}"
    );
    let failures = wait_metric_lines(&rig, metrics::CAPTURE_JUDGE_FAILURES_TOTAL, 1, WAIT, |_| {
        true
    });
    assert_eq!(failures[0]["labels"]["kind"], "call_failed");
    assert_eq!(failures[0]["labels"]["candidates"], 2);
    assert_eq!(store.judge_queue_stats().expect("stats").rejected, 0);

    judge.set_repeats(all_new());
    let settled = wait_for(&store, &cwd, 3, Status::Active);
    assert!(settled.iter().any(|m| m.content == RESTATED), "{settled:?}");
    assert_eq!(d.status(&cwd).judge.last_error, None);
}
