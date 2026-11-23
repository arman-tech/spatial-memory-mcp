//! SPEC section 16, The judge (B-167): automatic captures wait in the
//! judge's queue with their context; a kept capture becomes active with the
//! judge's rewrite, a rejected one stays pending, marked rejected, for 30
//! days (B-190), and nothing else changes a capture's status. Judge usage
//! is counted per day; pending captures, rejected ones and old usage are
//! pruned.

use sm_core::protocol::RememberOutcome;
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{MemoryType, Status};

use rusqlite::OptionalExtension;

use super::*;
use crate::{JUDGE_MAX_ATTEMPTS, JudgeUsage, Settle, Settled};

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

fn capture(project: ProjectId, content: &str) -> MemoryInput {
    let mut input = note(project, content);
    input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
    input.memory.status = Status::Pending;
    input
}

/// Stores captures through the judge's path; returns their ids in order.
fn queue(store: &Store, items: &[(ProjectId, &str, &str)]) -> Vec<MemoryId> {
    store
        .insert_captures(
            items
                .iter()
                .map(|(p, text, context)| (capture(*p, text), (*context).to_owned()))
                .collect(),
        )
        .expect("insert captures")
        .into_iter()
        .map(|i| i.id)
        .collect()
}

fn keep(id: MemoryId, rewrite: &str) -> Settle {
    Settle::Keep {
        id,
        content: rewrite.to_owned(),
        title: Some("Rewritten".to_owned()),
        memory_type: MemoryType::Decision,
        chunks: vec![NewChunk::embedded(piece(0, rewrite), fake_vector(rewrite))],
        judge_model: "haiku".to_owned(),
        reason: "knowledge".to_owned(),
    }
}

fn keyword(store: &Store, text: &str) -> Vec<MemoryId> {
    store
        .keyword_search(text, QueryScope::All, &MemoryFilter::default(), 10)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

fn by_vector(store: &Store, text: &str) -> Vec<MemoryId> {
    store
        .nearest(&VectorQuery {
            vector: &fake_vector(text),
            scope: QueryScope::All,
            filter: MemoryFilter::default(),
            limit: 10,
        })
        .expect("vector search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

fn queued_ids(store: &Store) -> Vec<MemoryId> {
    store
        .judge_queue(25, 100_000)
        .expect("queue")
        .into_iter()
        .map(|i| i.id)
        .collect()
}

#[test]
fn s16_b167_captures_wait_in_the_queue_with_their_context() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let out = fx
        .store
        .insert_captures(vec![
            (
                capture(p, "the uploader retries five times"),
                "why does the uploader give up".to_owned(),
            ),
            // The same text again: a duplicate, not queued twice.
            (
                capture(p, "the uploader retries five times"),
                "again".to_owned(),
            ),
        ])
        .expect("insert captures");
    assert_eq!(out[0].outcome, RememberOutcome::Created);
    assert_eq!(out[1].outcome, RememberOutcome::Duplicate);
    assert_eq!(out[1].id, out[0].id);

    let items = fx.store.judge_queue(25, 100_000).expect("queue");
    assert_eq!(items.len(), 1);
    let item = &items[0];
    assert_eq!(item.id, out[0].id);
    assert_eq!(item.project, p);
    assert_eq!(item.content, "the uploader retries five times");
    assert_eq!(item.context, "why does the uploader give up");
    assert_eq!(item.memory_type, MemoryType::Note);
    assert_eq!(item.attempts, 0);
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued), (1, 1));
    assert_eq!(stats.oldest, Some(item.queued_at));
    // Queued, but not served.
    assert!(keyword(&fx.store, "uploader").is_empty());

    // Only pending captures go through this path.
    let mut active = capture(p, "an active memory");
    active.memory.status = Status::Active;
    assert_code(
        fx.store.insert_captures(vec![(active, String::new())]),
        ErrorCode::Internal,
    );
    assert_eq!(table_count(&fx.store, "memories"), 1);
}

#[test]
fn s16_b167_the_queue_takes_fresh_captures_by_budget_and_a_retry_alone() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids = queue(
        &fx.store,
        &[
            (p, "first capture text", ""),
            (p, "second capture text", ""),
            (p, "third capture text", ""),
        ],
    );
    let take = |items: usize, chars: usize| -> Vec<MemoryId> {
        fx.store
            .judge_queue(items, chars)
            .expect("queue")
            .into_iter()
            .map(|i| i.id)
            .collect()
    };
    assert_eq!(take(25, 100_000), ids);
    assert_eq!(take(2, 100_000), ids[..2]);
    // The character budget stops the batch, but never below one capture.
    assert_eq!(take(25, 40), ids[..2]);
    assert_eq!(take(25, 1), ids[..1]);
    assert!(take(0, 100_000).is_empty());

    // A capture a call failed on comes after the fresh ones, and alone.
    fx.store.judge_failed(&ids[..1]).expect("failed");
    assert_eq!(take(25, 100_000), ids[1..]);
    fx.store.judge_failed(&ids[1..]).expect("failed");
    let retry = fx.store.judge_queue(25, 100_000).expect("queue");
    assert_eq!(retry.len(), 1);
    assert_eq!((retry[0].id, retry[0].attempts), (ids[0], 1));

    // After the last attempt a capture is never taken again, but still
    // counts as pending until it is pruned.
    for _ in 1..JUDGE_MAX_ATTEMPTS {
        fx.store.judge_failed(&ids[..1]).expect("failed");
    }
    assert_eq!(take(25, 100_000), ids[1..2]);
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued), (3, 2));
}

#[test]
fn s16_b167_a_kept_capture_becomes_active_with_its_rewrite() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let original = "ok so the walrus thing, we went with option two in the end";
    let rewrite = "The importer uses a streaming parser for files over 1 GB.";
    let id = queue(&fx.store, &[(p, original, "which parser")])[0];
    assert!(keyword(&fx.store, "walrus").is_empty());
    assert!(!by_vector(&fx.store, rewrite).contains(&id));

    let settled = fx
        .store
        .settle_captures(vec![keep(id, rewrite)])
        .expect("settle");
    let Settled::Kept(m) = &settled[0] else {
        panic!("not kept: {settled:?}");
    };
    assert_eq!(m.id, id);
    assert_eq!(m.status, Status::Active);
    assert_eq!(m.content, rewrite);
    assert_eq!(m.content_hash, sm_core::hash::content_hash(rewrite));
    assert_eq!(m.title.as_deref(), Some("Rewritten"));
    assert_eq!(m.memory_type, MemoryType::Decision);
    assert_eq!(m.provenance.source, ProvenanceSource::Capture);
    let judge = m
        .provenance
        .extra
        .as_ref()
        .and_then(|e| e.get("judge"))
        .expect("judge recorded");
    assert_eq!(judge["model"], "haiku");
    assert_eq!(judge["reason"], "knowledge");

    // The rewrite is served; the original's words are gone with its chunks.
    assert_eq!(keyword(&fx.store, "streaming parser"), vec![id]);
    assert!(keyword(&fx.store, "walrus").is_empty());
    assert!(by_vector(&fx.store, rewrite).contains(&id));
    assert_eq!(
        rows_where(&fx.store, "chunks", "memory_id", &id.to_string()),
        1
    );
    assert_eq!(table_count(&fx.store, "capture_judge"), 0);
    assert!(queued_ids(&fx.store).is_empty());
}

fn reject(id: MemoryId, reason: &str) -> Settle {
    Settle::Reject {
        id,
        reason: reason.to_owned(),
    }
}

/// When and why a capture was rejected, from its queue row.
fn rejection(store: &Store, id: MemoryId) -> Option<(Option<i64>, Option<String>)> {
    store
        .read("reading a rejection", |tx| {
            tx.query_row(
                "SELECT rejected_at, rejected_reason FROM capture_judge WHERE memory_id = ?1",
                [id.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .ctx("reading a rejection")
        })
        .expect("read")
}

/// B-190 replaces B-167's rule that a rejected capture is deleted: it stays
/// pending with its own text and chunks, marked rejected with the judge's
/// reason, and is never served, never taken by the judge again and never
/// counted as pending; `status` counts it as rejected. A later verdict on
/// it changes nothing.
#[test]
fn s16_b190_a_rejected_capture_is_kept_pending_unqueued_and_unserved() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "Now let me run the tests.";
    let id = queue(&fx.store, &[(p, text, "")])[0];
    let before = Timestamp::now().as_micros();
    let settled = fx
        .store
        .settle_captures(vec![reject(id, "narration")])
        .expect("settle");
    assert_eq!(settled, vec![Settled::Rejected]);
    let m = fx.store.get_memory(&id).expect("kept");
    assert_eq!((m.status, m.content.as_str()), (Status::Pending, text));
    let (at, reason) = rejection(&fx.store, id).expect("a queue row");
    assert!(at.is_some_and(|t| t >= before), "{at:?}");
    assert_eq!(reason.as_deref(), Some("narration"));
    assert_eq!(
        rows_where(&fx.store, "chunks", "memory_id", &id.to_string()),
        1
    );

    assert!(queued_ids(&fx.store).is_empty());
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!(
        (stats.pending, stats.queued, stats.rejected, stats.oldest),
        (0, 0, 1, None)
    );
    assert!(keyword(&fx.store, "tests").is_empty());
    assert!(!by_vector(&fx.store, text).contains(&id));

    // A later verdict, keep or reject, changes nothing.
    let settled = fx
        .store
        .settle_captures(vec![keep(id, "a rewrite"), reject(id, "status")])
        .expect("settle");
    assert_eq!(settled, vec![Settled::Gone, Settled::Gone]);
    let m = fx.store.get_memory(&id).expect("kept");
    assert_eq!((m.status, m.content.as_str()), (Status::Pending, text));
    assert_eq!(
        rejection(&fx.store, id).and_then(|r| r.1).as_deref(),
        Some("narration")
    );

    // A reason must be 1 to 64 characters; a bad one changes nothing.
    let other = queue(&fx.store, &[(p, "another capture", "")])[0];
    for bad in ["", "   ", &"x".repeat(crate::JUDGE_REASON_MAX_CHARS + 1)] {
        assert_code(
            fx.store.settle_captures(vec![reject(other, bad)]),
            ErrorCode::InvalidInput,
        );
    }
    assert_eq!(queued_ids(&fx.store), vec![other]);
    assert_eq!(rejection(&fx.store, other).and_then(|r| r.0), None);
}

/// B-190: a rejected capture is deleted, with its queue row and chunks,
/// once it was rejected more than [`crate::JUDGE_REJECTED_KEEP_DAYS`] days
/// ago, and not before, even when it had bad answers that would have let
/// the B-178 rule prune it at 14 days.
#[test]
fn s16_b190_a_rejected_capture_is_deleted_after_30_days_and_not_before() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids = queue(
        &fx.store,
        &[(p, "a rejected capture", ""), (p, "a given-up capture", "")],
    );
    for _ in 0..JUDGE_MAX_ATTEMPTS {
        fx.store.judge_failed(&ids[1..]).expect("fail");
    }
    let now = Timestamp::now();
    fx.store
        .settle_captures(vec![
            reject(ids[0], "transient"),
            reject(ids[1], "fragment"),
        ])
        .expect("settle");
    assert_eq!(crate::JUDGE_REJECTED_KEEP_DAYS, 30);
    for days in [0, 15, 29] {
        assert_eq!(
            fx.store
                .prune_pending(now.add_micros(days * MICROS_PER_DAY), 14)
                .expect("prune"),
            0,
            "day {days}"
        );
    }
    for id in &ids {
        assert_eq!(
            fx.store.get_memory(id).expect("get").status,
            Status::Pending
        );
    }
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(31 * MICROS_PER_DAY), 14)
            .expect("prune"),
        2
    );
    for id in &ids {
        assert_code(fx.store.get_memory(id), ErrorCode::NotFound);
    }
    assert_eq!(table_count(&fx.store, "capture_judge"), 0);
    assert_eq!(table_count(&fx.store, "chunks"), 0);
    assert_eq!(fx.store.judge_queue_stats().expect("stats").rejected, 0);
}

/// B-190: a rejected capture is no near neighbour for a later capture, so
/// it never keeps better text of the same knowledge out; a capture that
/// still waits for the judge is one (B-164).
#[test]
fn s16_b190_a_rejected_capture_keeps_no_later_capture_out() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "the queue drains once a minute";
    let id = queue(&fx.store, &[(p, text, "")])[0];
    let v = fake_vector(text);
    let near = fx
        .store
        .capture_nearest(p, std::slice::from_ref(&v))
        .expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(id));
    fx.store
        .settle_captures(vec![reject(id, "restatement")])
        .expect("settle");
    assert_eq!(
        fx.store.capture_nearest(p, &[v]).expect("nearest"),
        vec![None]
    );
}

#[test]
fn s16_b167_a_verdict_never_touches_a_memory_that_is_not_pending() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "the cache is keyed by the schema version";
    let id = queue(&fx.store, &[(p, text, "")])[0];
    // A save of the same text replaces the waiting capture (B-157).
    let saved = add(&fx.store, p, text);
    assert_ne!(saved, id);
    assert_code(fx.store.get_memory(&id), ErrorCode::NotFound);

    let settled = fx
        .store
        .settle_captures(vec![
            keep(id, "anything"),
            reject(saved, "narration"),
            keep(saved, "a rewrite of a saved memory"),
        ])
        .expect("settle");
    assert_eq!(settled, vec![Settled::Gone, Settled::Gone, Settled::Gone]);
    let m = fx.store.get_memory(&saved).expect("still there");
    assert_eq!((m.status, m.content.as_str()), (Status::Active, text));
}

/// A rewrite whose content hash an active memory of the project holds is
/// not stored. Before B-190 the capture was deleted; it is now rejected
/// with the reason `duplicate`, kept pending with its own text, and never
/// served.
#[test]
fn s16_b167_b190_a_rewrite_an_active_memory_holds_is_rejected_as_a_duplicate() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let other = project(&fx.store, "beta");
    let held = "Deploys run from the release branch only.";
    let existing = add(&fx.store, p, held);
    // The same text in another project is no duplicate.
    let elsewhere = queue(&fx.store, &[(other, "deploys, release branch", "")])[0];
    let id = queue(&fx.store, &[(p, "so deploys only go out from release", "")])[0];

    let settled = fx
        .store
        .settle_captures(vec![keep(id, held), keep(elsewhere, held)])
        .expect("settle");
    assert_eq!(settled[0], Settled::Duplicate { of: existing });
    assert!(matches!(&settled[1], Settled::Kept(m) if m.project_id == other));
    let m = fx.store.get_memory(&id).expect("kept, rejected");
    assert_eq!(
        (m.status, m.content.as_str()),
        (Status::Pending, "so deploys only go out from release")
    );
    assert_eq!(
        rejection(&fx.store, id).and_then(|r| r.1).as_deref(),
        Some("duplicate")
    );
    assert!(!queued_ids(&fx.store).contains(&id));
    assert_eq!(keyword(&fx.store, "release branch").len(), 2);
    assert!(!keyword(&fx.store, "deploys").contains(&id));

    // An empty rewrite is refused and nothing is changed.
    let late = queue(&fx.store, &[(p, "one more capture", "")])[0];
    assert_code(
        fx.store.settle_captures(vec![keep(late, "  ")]),
        ErrorCode::InvalidInput,
    );
    assert_eq!(
        fx.store.get_memory(&late).expect("get").status,
        Status::Pending
    );
}

#[test]
fn s16_b167_judge_usage_adds_up_per_day() {
    let fx = fixture();
    let day = "2026-11-17";
    assert_eq!(
        fx.store.judge_usage(day).expect("usage"),
        JudgeUsage::default()
    );
    fx.store
        .record_judge_call(day, 25, false, 0.0019)
        .expect("record");
    fx.store
        .record_judge_call(day, 0, true, 0.0)
        .expect("record");
    fx.store
        .record_judge_call("2026-11-18", 3, false, 0.0004)
        .expect("record");
    let u = fx.store.judge_usage(day).expect("usage");
    assert_eq!((u.calls, u.judged, u.failures), (2, 25, 1));
    assert!((u.cost_usd - 0.0019).abs() < 1e-12, "{u:?}");
    assert_eq!(fx.store.judge_usage("2026-11-18").expect("usage").calls, 1);
    for bad in [-0.1, f64::NAN, f64::INFINITY] {
        assert_code(
            fx.store.record_judge_call(day, 1, false, bad),
            ErrorCode::InvalidInput,
        );
    }
    assert_eq!(fx.store.judge_usage(day).expect("usage").calls, 2);
}

#[test]
fn s16_b167_pending_captures_and_old_usage_are_pruned() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let queued = queue(&fx.store, &[(p, "a capture that waits", "")])[0];
    // A capture stored before the judge: pending, never queued.
    let old = fx
        .store
        .insert_memory(capture(p, "an old-rule capture"), Access::Local)
        .expect("insert")
        .id;
    let active = add(&fx.store, p, "an active memory");
    let now = Timestamp::now();
    fx.store
        .record_judge_call(&now.date_string(), 1, false, 0.0003)
        .expect("record");
    let ancient = now.add_micros(-91 * MICROS_PER_DAY).date_string();
    fx.store
        .record_judge_call(&ancient, 1, false, 0.0003)
        .expect("record");

    // Within the wait nothing goes but usage past its retention.
    assert_eq!(fx.store.prune_pending(now, 14).expect("prune"), 0);
    assert_eq!(fx.store.judge_usage(&ancient).expect("usage").calls, 0);
    assert_eq!(
        fx.store
            .judge_usage(&now.date_string())
            .expect("usage")
            .calls,
        1
    );

    // After the wait the old-rule capture goes; the queued capture, which
    // the judge has not given up on, stays (B-178).
    let later = now.add_micros(15 * MICROS_PER_DAY);
    assert_eq!(fx.store.prune_pending(later, 14).expect("prune"), 1);
    assert_code(fx.store.get_memory(&old), ErrorCode::NotFound);
    assert_eq!(
        fx.store.get_memory(&queued).expect("get").status,
        Status::Pending
    );
    assert_eq!(table_count(&fx.store, "capture_judge"), 1);
    assert_eq!(
        fx.store.get_memory(&active).expect("get").status,
        Status::Active
    );
}

/// A pending capture never expires because the judge was off, unavailable
/// or out of calls: only a capture the judge gave up on, after
/// [`JUDGE_MAX_ATTEMPTS`] bad answers, is pruned, and its wait counts from
/// its last bad answer (B-178; owner step 20).
#[test]
fn s16_b178_a_capture_the_judge_has_not_given_up_on_is_never_pruned() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids = queue(
        &fx.store,
        &[
            (p, "a capture the judge never saw", ""),
            (p, "a capture with one bad answer", ""),
            (p, "a capture the judge gave up on", ""),
        ],
    );
    let (never, once, given_up) = (ids[0], ids[1], ids[2]);
    // All three were queued a month ago, while the judge was off.
    let month_ago = Timestamp::now().as_micros() - 30 * MICROS_PER_DAY;
    fx.store
        .write("backdating the queue", |tx| {
            tx.execute("UPDATE capture_judge SET queued_at = ?1", [month_ago])
                .ctx("backdating the queue")
        })
        .expect("backdate");
    fx.store.judge_failed(&[once, given_up]).expect("fail");
    for _ in 1..JUDGE_MAX_ATTEMPTS {
        fx.store.judge_failed(&[given_up]).expect("fail");
    }
    let now = Timestamp::now();

    // The month queued does not count: the given-up capture waits 14 days
    // from its last bad answer.
    for days in [0, 1, 13] {
        assert_eq!(
            fx.store
                .prune_pending(now.add_micros(days * MICROS_PER_DAY), 14)
                .expect("prune"),
            0,
            "day {days}"
        );
    }
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(15 * MICROS_PER_DAY), 14)
            .expect("prune"),
        1
    );
    assert_code(fx.store.get_memory(&given_up), ErrorCode::NotFound);

    // A year on, the captures the judge has not given up on still wait.
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(365 * MICROS_PER_DAY), 14)
            .expect("prune"),
        0
    );
    for id in [never, once] {
        assert_eq!(
            fx.store.get_memory(&id).expect("get").status,
            Status::Pending
        );
    }
    // Both are still offered to the judge (a retry is taken alone, so the
    // count, not one batch, shows it).
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued), (2, 2));
}

/// A capture marked failed judgement (B-206) leaves the queue at once,
/// whatever its attempts, stays pending and unserved, is counted as failed,
/// and is pruned 14 days after it was marked, like one whose verdict could
/// not be applied three times.
#[test]
fn s16_b206_a_capture_marked_failed_judgement_leaves_the_queue_and_is_counted() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids = queue(
        &fx.store,
        &[
            (p, "a capture the judge answers well", ""),
            (p, "a capture answered badly twice alone", ""),
            (
                p,
                "a capture with one unapplied verdict, then answered badly",
                "",
            ),
        ],
    );
    let (fine, fresh_bad, retried_bad) = (ids[0], ids[1], ids[2]);
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued, stats.failed), (3, 3, 0));

    fx.store.judge_failed(&[retried_bad]).expect("fail");
    fx.store
        .mark_failed_judgement(&[fresh_bad, retried_bad])
        .expect("mark");
    let taken: Vec<MemoryId> = fx
        .store
        .judge_queue(25, 100_000)
        .expect("queue")
        .into_iter()
        .map(|i| i.id)
        .collect();
    assert_eq!(taken, [fine], "only the unmarked capture is offered");
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!(
        (stats.pending, stats.queued, stats.rejected, stats.failed),
        (3, 1, 0, 2)
    );
    for id in [fresh_bad, retried_bad] {
        assert_eq!(
            fx.store.get_memory(&id).expect("get").status,
            Status::Pending
        );
    }
    assert!(
        keyword(&fx.store, "answered badly").is_empty(),
        "a capture marked failed judgement is not served"
    );

    let now = Timestamp::now();
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(13 * MICROS_PER_DAY), 14)
            .expect("prune"),
        0
    );
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(15 * MICROS_PER_DAY), 14)
            .expect("prune"),
        2
    );
    assert_eq!(
        fx.store.get_memory(&fine).expect("get").status,
        Status::Pending
    );
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued, stats.failed), (1, 1, 0));
}

#[test]
fn s16_b169_active_near_ignores_pending_captures() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "the queue drains once a minute";
    let pending = queue(&fx.store, &[(p, text, "")])[0];
    let v = fake_vector(text);
    let near = fx
        .store
        .capture_nearest(p, std::slice::from_ref(&v))
        .expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(pending));
    assert_eq!(
        fx.store
            .active_near(p, std::slice::from_ref(&v), -1.0, 5)
            .expect("near"),
        vec![Vec::new()]
    );
    let active = add(&fx.store, p, text);
    let near = fx.store.active_near(p, &[v], -1.0, 5).expect("near");
    assert_eq!(near[0].iter().map(|n| n.id).collect::<Vec<_>>(), [active]);
}

/// B-192: the neighbours the duplicate check shows the judge are the
/// project's active memories at or above the floor, closest first, at most
/// `k`; another project's memories never count.
#[test]
fn s16_b192_active_near_gives_the_closest_k_at_or_above_the_floor() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let other = project(&fx.store, "beta");
    let query = "the queue drains once a minute in the nightly build";
    let texts = [
        "the queue drains once a minute in the nightly build",
        "the queue drains once a minute in the daily build",
        "the queue drains once an hour in the weekly run",
        "the parser reads utf8 only",
    ];
    let ids: Vec<MemoryId> = texts.iter().map(|t| add(&fx.store, p, t)).collect();
    add(&fx.store, other, texts[0]);
    let v = fake_vector(query);
    let cos: Vec<f32> = texts
        .iter()
        .map(|t| sm_core::testing::dot(&fake_vector(t), &v))
        .collect();
    assert!(
        cos[0] > cos[1] && cos[1] > cos[2] && cos[2] > cos[3],
        "{cos:?}"
    );
    let floor = (cos[2] + cos[3]) / 2.0;
    let near = fx
        .store
        .active_near(p, std::slice::from_ref(&v), floor, 5)
        .expect("near");
    assert_eq!(
        near[0].iter().map(|n| n.id).collect::<Vec<_>>(),
        ids[..3],
        "the three at or above the floor, closest first"
    );
    assert!((near[0][0].cosine - cos[0]).abs() < 1e-5);
    let near = fx.store.active_near(p, &[v], floor, 2).expect("near");
    assert_eq!(near[0].iter().map(|n| n.id).collect::<Vec<_>>(), ids[..2]);
}

/// B-176: a pending capture with no row in the judge's queue was made by
/// the rules before the judge and is never judged. A capture of the same
/// text under the current rules replaces it and is queued, so a replay can
/// capture it again; such a capture is no near neighbour either. A queued
/// capture of the same text is still a duplicate.
#[test]
fn s16_b176_a_capture_replaces_an_unqueued_pending_capture() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let text = "the importer retries a failed row twice";
    let old = fx
        .store
        .insert_memory(capture(p, text), Access::Local)
        .expect("an old-rule capture")
        .id;
    let elsewhere = fx
        .store
        .insert_memory(capture(q, text), Access::Local)
        .expect("an old-rule capture in another project")
        .id;
    let v = fake_vector(text);
    assert_eq!(
        fx.store
            .capture_nearest(p, std::slice::from_ref(&v))
            .expect("nearest"),
        vec![None]
    );

    let fresh = fx
        .store
        .insert_captures(vec![(capture(p, text), "context".to_owned())])
        .expect("insert a capture");
    assert_eq!(fresh[0].outcome, RememberOutcome::Created);
    assert_ne!(fresh[0].id, old);
    assert!(fx.store.find_memory(&old).expect("find").is_none());
    assert_eq!(queued_ids(&fx.store), vec![fresh[0].id]);
    assert_eq!(
        fx.store
            .get_memory(&elsewhere)
            .expect("the other project's capture")
            .status,
        Status::Pending
    );
    let near = fx
        .store
        .capture_nearest(p, std::slice::from_ref(&v))
        .expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(fresh[0].id));

    let again = fx
        .store
        .insert_captures(vec![(capture(p, text), "context".to_owned())])
        .expect("insert it again");
    assert_eq!(
        (again[0].outcome, again[0].id),
        (RememberOutcome::Duplicate, fresh[0].id)
    );
    assert_eq!(queued_ids(&fx.store), vec![fresh[0].id]);
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 2);
}

// ---------------------------------------------------------------------------
// The statements task (B-240 to B-243; STATEMENT_JUDGE_PROPOSAL.md)
// ---------------------------------------------------------------------------

/// A capture of session `session`, captured at `created` microseconds, of
/// the kind `source` (`extra.source`, B-179).
fn capture_in(
    project: ProjectId,
    session: &str,
    content: &str,
    created: i64,
    source: &str,
) -> (MemoryInput, String) {
    let mut input = capture(project, content);
    input.memory.provenance.session_id = Some(session.to_owned());
    input.memory.provenance.extra = Some(serde_json::json!({"source": source, "hook": "stop"}));
    input.memory.created_at = Some(Timestamp::from_micros(created));
    (input, format!("context of {content}"))
}

/// Sets when a capture was queued, so the queue's order is fixed.
fn set_queued(store: &Store, id: MemoryId, micros: i64) {
    store
        .write("setting queued_at", |tx| {
            tx.execute(
                "UPDATE capture_judge SET queued_at = ?2 WHERE memory_id = ?1",
                (id.to_string(), micros),
            )
            .ctx("setting queued_at")
        })
        .expect("set queued_at");
}

fn statement(index: u32, text: &str, hold: Option<crate::Hold>) -> crate::StatementSettle {
    crate::StatementSettle {
        index,
        content: text.to_owned(),
        title: Some(format!("Statement {index}")),
        memory_type: MemoryType::Decision,
        chunks: vec![NewChunk::embedded(piece(0, text), fake_vector(text))],
        grade: "must_keep".to_owned(),
        hold,
    }
}

fn statements(id: MemoryId, list: Vec<crate::StatementSettle>) -> Settle {
    Settle::Statements {
        id,
        judge_model: "haiku".to_owned(),
        statements: list,
    }
}

fn judge_extra(m: &sm_core::types::Memory) -> serde_json::Value {
    m.provenance
        .extra
        .as_ref()
        .and_then(|e| e.get("judge"))
        .cloned()
        .expect("judge recorded")
}

/// B-241: the statements task takes one session at a time: the oldest
/// waiting capture names the project and session, and that session's
/// waiting captures come in transcript order (when each was captured), up
/// to the batch's budget. A capture with an attempt goes alone, after the
/// fresh ones; a rejected one never. Each item carries its session and
/// the kind of text it holds.
#[test]
fn s16_b241_the_queue_takes_one_session_in_transcript_order() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let base = Timestamp::now().as_micros() - 3_600_000_000;
    let specs = [
        (
            p,
            "A",
            "the third statement of session A",
            base + 30,
            "reply",
        ),
        (
            p,
            "B",
            "the only statement of session B",
            base + 10,
            "prompt",
        ),
        (
            p,
            "A",
            "the first statement of session A",
            base + 10,
            "prompt",
        ),
        (q, "A", "a statement of session A elsewhere", base, "prompt"),
        (
            p,
            "A",
            "the second statement, a tool's output",
            base + 20,
            "tool_result",
        ),
        (p, "A", "a capture a call failed on", base, "prompt"),
        (p, "A", "a capture the judge rejected", base, "prompt"),
    ];
    let mut ids = Vec::new();
    for (i, (project, session, text, created, source)) in specs.iter().enumerate() {
        let out = fx
            .store
            .insert_captures(vec![capture_in(*project, session, text, *created, source)])
            .expect("insert a capture");
        set_queued(
            &fx.store,
            out[0].id,
            base + 1_000 + i64::try_from(i).expect("i"),
        );
        ids.push(out[0].id);
    }
    fx.store.judge_failed(&ids[5..6]).expect("fail");
    fx.store
        .settle_captures(vec![reject(ids[6], "transient")])
        .expect("reject");

    let take = |items: usize, chars: usize| -> Vec<MemoryId> {
        fx.store
            .judge_queue_session(items, chars)
            .expect("session queue")
            .into_iter()
            .map(|i| i.id)
            .collect()
    };
    // Session A of alpha, named by the oldest queued, in transcript order.
    assert_eq!(take(25, 100_000), vec![ids[2], ids[4], ids[0]]);
    assert_eq!(take(2, 100_000), vec![ids[2], ids[4]]);
    assert_eq!(take(1, 100_000), vec![ids[2]]);
    assert_eq!(take(25, 1), vec![ids[2]]);
    assert!(take(0, 100_000).is_empty());

    let items = fx.store.judge_queue_session(25, 100_000).expect("queue");
    assert_eq!(items[0].session.as_deref(), Some("A"));
    assert_eq!(items[0].source, sm_core::types::SourceTag::UserMessage);
    assert_eq!(items[1].source, sm_core::types::SourceTag::ToolOutput);
    assert_eq!(items[2].source, sm_core::types::SourceTag::Reply);
    assert_eq!(
        items[0].context,
        "context of the first statement of session A"
    );
    // The verdicts queue carries them too.
    let verdicts = fx.store.judge_queue(25, 100_000).expect("queue");
    assert_eq!(verdicts[0].id, ids[0]);
    assert_eq!(verdicts[0].session.as_deref(), Some("A"));
    assert_eq!(verdicts[0].source, sm_core::types::SourceTag::Reply);

    // Once session A of alpha is settled, session B comes next, then the
    // other project's, then the capture a call failed on, alone.
    fx.store
        .settle_captures(vec![
            reject(ids[2], "fragment"),
            reject(ids[4], "fragment"),
            reject(ids[0], "fragment"),
        ])
        .expect("settle");
    assert_eq!(take(25, 100_000), vec![ids[1]]);
    fx.store
        .settle_captures(vec![reject(ids[1], "fragment")])
        .expect("settle");
    assert_eq!(take(25, 100_000), vec![ids[3]]);
    fx.store
        .settle_captures(vec![reject(ids[3], "fragment")])
        .expect("settle");
    let retry = fx.store.judge_queue_session(25, 100_000).expect("queue");
    assert_eq!(retry.len(), 1);
    assert_eq!((retry[0].id, retry[0].attempts), (ids[5], 1));
}

/// B-242: each statement the judge keeps is stored as its own active
/// memory, with the capture's own words. The first takes the capture's
/// row and id; the others are new memories with the capture's provenance,
/// anchors, tags, importance and time. `extra.judge` names the task, the
/// capture, the statement's place and grade; no text the model wrote is
/// stored (owner's change A6). Each is found by keyword and by vector, and
/// the capture leaves the queue. A statement whose words another memory
/// of the project holds is not stored again.
#[test]
fn s16_b242_settle_statements_stores_each_the_first_in_the_captures_row() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "We chose SQLite for the queue. The importer retries a failed row twice.";
    let (mut input, context) = capture_in(p, "S1", text, 1_000_000, "reply");
    input.memory.tags = vec!["queue".to_owned()];
    input.memory.importance = 0.7;
    let anchor = || {
        sm_core::types::AnchorRef::new(sm_core::types::AnchorKind::File, "src/queue.rs")
            .expect("anchor")
    };
    input.memory.anchors = vec![anchor()];
    let id = fx
        .store
        .insert_captures(vec![(input, context)])
        .expect("insert")[0]
        .id;
    let first = "We chose SQLite for the queue.";
    let second = "The importer retries a failed row twice.";

    let settled = fx
        .store
        .settle_captures(vec![statements(
            id,
            vec![statement(0, first, None), statement(1, second, None)],
        )])
        .expect("settle");
    let Settled::Statements(out) = &settled[0] else {
        panic!("not statements: {settled:?}");
    };
    assert_eq!(out.len(), 2);
    let crate::StatementSettled::Stored(a) = &out[0] else {
        panic!("not stored: {out:?}");
    };
    let crate::StatementSettled::Stored(b) = &out[1] else {
        panic!("not stored: {out:?}");
    };
    assert_eq!(a.id, id);
    assert_ne!(b.id, id);
    for (m, words, index) in [(a, first, 0), (b, second, 1)] {
        assert_eq!(m.status, Status::Active);
        assert_eq!(m.content, words);
        assert_eq!(m.content_hash, sm_core::hash::content_hash(words));
        assert_eq!(
            m.title.as_deref(),
            Some(format!("Statement {index}").as_str())
        );
        assert_eq!(m.memory_type, MemoryType::Decision);
        assert_eq!(m.project_id, p);
        assert_eq!(m.created_at, Timestamp::from_micros(1_000_000));
        assert_eq!(m.tags, vec!["queue".to_owned()]);
        assert!((m.importance - 0.7).abs() < 1e-9);
        assert_eq!(m.provenance.source, ProvenanceSource::Capture);
        assert_eq!(m.provenance.session_id.as_deref(), Some("S1"));
        let extra = m.provenance.extra.as_ref().expect("extra");
        assert_eq!(extra["source"], "reply");
        let judge = judge_extra(m);
        assert_eq!(judge["model"], "haiku");
        assert_eq!(judge["task"], "statements");
        assert_eq!(judge["capture"], id.to_string());
        assert_eq!(judge["statement"], index);
        assert_eq!(judge["grade"], "must_keep");
        assert!(judge.get("reason").is_none(), "no model text is stored");
        assert_eq!(
            fx.store.get_memory(&m.id).expect("get").anchors,
            vec![anchor()]
        );
        assert_eq!(
            rows_where(&fx.store, "chunks", "memory_id", &m.id.to_string()),
            1
        );
    }
    assert_eq!(keyword(&fx.store, "SQLite"), vec![a.id]);
    assert_eq!(keyword(&fx.store, "importer"), vec![b.id]);
    assert!(by_vector(&fx.store, first).contains(&a.id));
    assert!(by_vector(&fx.store, second).contains(&b.id));
    assert_eq!(table_count(&fx.store, "capture_judge"), 0);
    assert!(
        fx.store
            .judge_queue_session(25, 100_000)
            .expect("queue")
            .is_empty()
    );
    let counts = fx
        .store
        .judge_statements_since(Timestamp::now().add_micros(-60_000_000))
        .expect("counts");
    assert_eq!((counts.stored, counts.repeats, counts.unchecked), (2, 0, 0));

    let again = queue(&fx.store, &[(p, "the importer, again: retries twice", "")])[0];
    let settled = fx
        .store
        .settle_captures(vec![statements(
            again,
            vec![
                statement(0, second, None),
                statement(1, "The importer logs each retry.", None),
            ],
        )])
        .expect("settle");
    let Settled::Statements(out) = &settled[0] else {
        panic!("not statements: {settled:?}");
    };
    assert_eq!(out[0], crate::StatementSettled::Duplicate { of: b.id });
    let crate::StatementSettled::Stored(c) = &out[1] else {
        panic!("not stored: {out:?}");
    };
    assert_eq!(c.id, again);
    assert_eq!(c.content, "The importer logs each retry.");
}

/// B-242: a capture none of whose statements is stored stays pending with
/// its own words, rejected and kept 30 days (B-190), with the first reason
/// that applies: a statement left unchecked, one that repeats another
/// memory, one whose words another memory holds, or no statement at all.
#[test]
fn s16_b242_a_capture_with_no_statement_stored_is_rejected() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let held = add(&fx.store, p, "The cache is warmed on first use.");
    let ids = queue(
        &fx.store,
        &[
            (p, "nothing worth keeping here", ""),
            (p, "the cache warms on first use, as said", ""),
            (p, "the cache is warmed, on first use", ""),
            (p, "an answer the duplicate step never checked", ""),
        ],
    );
    let repeat = crate::Hold::Repeat(crate::RepeatOf::Memory(held));
    let settled = fx
        .store
        .settle_captures(vec![
            statements(ids[0], vec![]),
            statements(
                ids[1],
                vec![statement(0, "The cache warms on first use.", Some(repeat))],
            ),
            statements(
                ids[2],
                vec![statement(0, "The cache is warmed on first use.", None)],
            ),
            statements(
                ids[3],
                vec![statement(
                    0,
                    "A statement left unchecked.",
                    Some(crate::Hold::Unchecked),
                )],
            ),
        ])
        .expect("settle");
    assert_eq!(settled[0], Settled::Statements(vec![]));
    assert_eq!(
        settled[2],
        Settled::Statements(vec![crate::StatementSettled::Duplicate { of: held }])
    );
    for (id, reason) in [
        (ids[0], "nothing_kept"),
        (ids[1], "repeats"),
        (ids[2], "duplicate"),
        (ids[3], "unchecked"),
    ] {
        let m = fx.store.get_memory(&id).expect("get");
        assert_eq!(m.status, Status::Pending, "{reason}");
        assert_eq!(
            rejection(&fx.store, id).and_then(|r| r.1).as_deref(),
            Some(reason)
        );
    }
    assert!(
        fx.store
            .judge_queue_session(25, 100_000)
            .expect("queue")
            .is_empty()
    );
}

/// B-243 (owner's changes A2 and A4): a statement the duplicate step said
/// repeats a memory, or one whose duplicate answer still failed alone, is
/// not stored. It is kept pending, never served and never queued, as its
/// own memory, rejected `repeat` or `unchecked`, with the id of the memory
/// it was said to repeat, for 30 days (B-190), then deleted. `status`
/// counts it. Revising it to active undoes the drop. The capture's other
/// statements settle as usual.
#[test]
fn s16_b243_a_held_statement_is_kept_unserved_30_days_with_what_it_repeats() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let old = add(&fx.store, p, "Deploys go out on Tuesdays.");
    let ids = queue(
        &fx.store,
        &[
            (p, "deploys on tuesdays; the flag is called ship_it", ""),
            (p, "the ship_it flag again, and a note on tags", ""),
        ],
    );
    let now = Timestamp::now();
    let settled = fx
        .store
        .settle_captures(vec![
            statements(
                ids[0],
                vec![
                    statement(
                        0,
                        "Deploys happen every Tuesday.",
                        Some(crate::Hold::Repeat(crate::RepeatOf::Memory(old))),
                    ),
                    statement(1, "The release flag is called ship_it.", None),
                ],
            ),
            statements(
                ids[1],
                vec![
                    statement(
                        0,
                        "The flag that releases is ship_it.",
                        Some(crate::Hold::Repeat(crate::RepeatOf::Statement {
                            capture: ids[0],
                            index: 1,
                        })),
                    ),
                    statement(1, "Tags are lower case.", Some(crate::Hold::Unchecked)),
                    statement(2, "Release tags start with v.", None),
                ],
            ),
        ])
        .expect("settle");
    let Settled::Statements(first) = &settled[0] else {
        panic!("{settled:?}");
    };
    let Settled::Statements(second) = &settled[1] else {
        panic!("{settled:?}");
    };
    let crate::StatementSettled::Held { id: h0, repeat_of } = first[0] else {
        panic!("{first:?}");
    };
    assert_eq!(repeat_of, Some(old));
    let crate::StatementSettled::Stored(flag) = &first[1] else {
        panic!("{first:?}");
    };
    // The first statement stored takes the capture's row.
    assert_eq!(flag.id, ids[0]);
    let crate::StatementSettled::Held { id: h1, repeat_of } = second[0] else {
        panic!("{second:?}");
    };
    assert_eq!(repeat_of, Some(flag.id));
    let crate::StatementSettled::Held { id: h2, repeat_of } = second[1] else {
        panic!("{second:?}");
    };
    assert_eq!(repeat_of, None);
    let crate::StatementSettled::Stored(tags) = &second[2] else {
        panic!("{second:?}");
    };
    assert_eq!(tags.id, ids[1]);

    for (h, reason, of, words) in [
        (h0, "repeat", Some(old), "Deploys happen every Tuesday."),
        (
            h1,
            "repeat",
            Some(flag.id),
            "The flag that releases is ship_it.",
        ),
        (h2, "unchecked", None, "Tags are lower case."),
    ] {
        let m = fx.store.get_memory(&h).expect("held");
        assert_eq!(m.status, Status::Pending);
        assert_eq!(m.content, words);
        assert_eq!(
            rejection(&fx.store, h).and_then(|r| r.1).as_deref(),
            Some(reason)
        );
        let judge = judge_extra(&m);
        assert_eq!(judge["task"], "statements");
        assert_eq!(
            judge["repeat_of"],
            of.map_or(serde_json::Value::Null, |o| serde_json::json!(
                o.to_string()
            ))
        );
        // Never served.
        assert!(!by_vector(&fx.store, words).contains(&h));
    }
    assert_eq!(keyword(&fx.store, "Tuesday"), vec![old]);
    assert!(
        fx.store
            .judge_queue_session(25, 100_000)
            .expect("queue")
            .is_empty()
    );
    let stats = fx.store.judge_queue_stats().expect("stats");
    assert_eq!((stats.rejected, stats.queued), (3, 0));
    let counts = fx
        .store
        .judge_statements_since(now.add_micros(-60_000_000))
        .expect("counts");
    assert_eq!((counts.stored, counts.repeats, counts.unchecked), (2, 2, 1));

    // Undo one drop: the owner revises it to active and it is served.
    let undone = fx
        .store
        .revise_memory(
            &h0,
            crate::Revision {
                status: Some(Status::Active),
                ..crate::Revision::default()
            },
            Access::Local,
        )
        .expect("undo");
    assert_eq!(undone.status, Status::Active);
    assert!(keyword(&fx.store, "Tuesday").contains(&h0));

    // The other two are deleted after 30 days, and not before.
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(29 * MICROS_PER_DAY), 14)
            .expect("prune"),
        0
    );
    assert_eq!(
        fx.store
            .prune_pending(now.add_micros(31 * MICROS_PER_DAY), 14)
            .expect("prune"),
        2
    );
    assert_code(fx.store.get_memory(&h1), ErrorCode::NotFound);
    assert_code(fx.store.get_memory(&h2), ErrorCode::NotFound);
    for id in [h0, flag.id, tags.id] {
        assert_eq!(
            fx.store.get_memory(&id).expect("kept").status,
            Status::Active
        );
    }
}
