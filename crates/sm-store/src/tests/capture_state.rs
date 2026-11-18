//! B-163 (RC2 in docs/v3/CAPTURE-RCA.md): a capture session's read position
//! and the hashes a project's capture has handled are kept in the store, so
//! a restarted daemon neither reads a session again nor brings back a
//! forgotten capture. B-164 (RC9): the near-duplicate lookup compares a
//! candidate with the project's active and pending memories only, and B-176
//! with only the pending captures that wait in the judge's queue.

use sm_core::error::ErrorCode;
use sm_core::testing::fake_vector;
use sm_core::time::Timestamp;
use sm_core::types::Status;

use super::*;
use crate::{CAPTURE_STATE_RETENTION_DAYS, CapturePosition, CaptureProgress};

const DAY: i64 = 86_400 * 1_000_000;

fn hashes(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// Records `session`'s offset as read part-way (B-184's `from_start` false).
fn record(store: &Store, session: Option<(&str, u64)>, p: ProjectId, h: &[String], at: Timestamp) {
    store
        .record_capture_progress(&CaptureProgress {
            session: session.map(|(s, offset)| {
                (
                    s,
                    CapturePosition {
                        offset,
                        from_start: false,
                    },
                )
            }),
            project: p,
            hashes: h,
            at,
        })
        .expect("record capture progress");
}

#[test]
fn s16_b163_a_read_position_survives_a_new_store() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    assert_eq!(fx.store.capture_offset("s-1").expect("offset"), None);
    record(&fx.store, Some(("s-1", 120)), p, &[], Timestamp::now());
    record(&fx.store, Some(("s-1", 480)), p, &[], Timestamp::now());
    let again = fx.reopen(stamp());
    assert_eq!(again.capture_offset("s-1").expect("offset"), Some(480));
    assert_eq!(again.capture_offset("s-2").expect("offset"), None);
    // A position needs a session.
    assert_code(
        again.record_capture_progress(&CaptureProgress {
            session: Some((
                " ",
                CapturePosition {
                    offset: 1,
                    from_start: false,
                },
            )),
            project: p,
            hashes: &[],
            at: Timestamp::now(),
        }),
        ErrorCode::InvalidInput,
    );
}

#[test]
fn s16_b163_a_seen_hash_belongs_to_its_project_and_outlives_its_memory() {
    let fx = fixture();
    let alpha = project(&fx.store, "alpha");
    let beta = project(&fx.store, "beta");
    let id = add(&fx.store, alpha, "captured, then forgotten");
    let h = hashes(&["h-forgotten"]);
    record(&fx.store, None, alpha, &h, Timestamp::now());
    fx.store.forget(&[id], Access::Local).expect("forget");
    let asked = hashes(&["h-forgotten", "h-new"]);
    let seen = fx.store.captures_seen(alpha, &asked).expect("seen");
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), h);
    assert!(
        fx.store
            .captures_seen(beta, &asked)
            .expect("seen")
            .is_empty()
    );
    assert!(fx.store.captures_seen(alpha, &[]).expect("seen").is_empty());
    // Still seen from a later store.
    let again = fx.reopen(stamp());
    assert_eq!(again.captures_seen(alpha, &asked).expect("seen").len(), 1);
}

/// Only a pending memory of the project with no row in the judge's queue
/// holds an unqueued capture hash: not a queued capture, not an active
/// memory, not a memory of another project.
#[test]
fn s16_b176_an_unqueued_capture_hash_is_a_pending_memory_outside_the_queue() {
    let fx = fixture();
    let alpha = project(&fx.store, "alpha");
    let beta = project(&fx.store, "beta");
    let capture = |p: ProjectId, text: &str| {
        let mut input = note(p, text);
        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
        input.memory.status = Status::Pending;
        input
    };
    let old_rule = "an old-rule capture of the importer";
    let queued = "a queued capture of the indexer";
    let active = "an active note on the cache";
    fx.store
        .insert_memory(capture(alpha, old_rule), Access::Local)
        .expect("an old-rule capture");
    fx.store
        .insert_memory(capture(beta, active), Access::Local)
        .expect("an old-rule capture in another project");
    fx.store
        .insert_captures(vec![(capture(alpha, queued), "context".to_owned())])
        .expect("queue a capture");
    add(&fx.store, alpha, active);
    let h = |t: &str| sm_core::hash::content_hash(t);
    let asked = vec![h(old_rule), h(queued), h(active), h("never stored")];
    let held = fx
        .store
        .unqueued_capture_hashes(alpha, &asked)
        .expect("unqueued");
    assert_eq!(held.into_iter().collect::<Vec<_>>(), vec![h(old_rule)]);
    let held = fx
        .store
        .unqueued_capture_hashes(beta, &asked)
        .expect("unqueued");
    assert_eq!(held.into_iter().collect::<Vec<_>>(), vec![h(active)]);
    assert!(
        fx.store
            .unqueued_capture_hashes(alpha, &[])
            .expect("unqueued")
            .is_empty()
    );
}

#[test]
fn s16_b163_capture_state_is_pruned_after_its_retention() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let now = Timestamp::now();
    let old = now.add_micros(-(CAPTURE_STATE_RETENTION_DAYS + 1) * DAY);
    let recent = now.add_micros(-(CAPTURE_STATE_RETENTION_DAYS - 1) * DAY);
    record(&fx.store, Some(("old", 10)), p, &hashes(&["h-old"]), old);
    record(
        &fx.store,
        Some(("recent", 20)),
        p,
        &hashes(&["h-recent"]),
        recent,
    );
    // Seen again recently: the later time is kept, so it stays.
    record(&fx.store, None, p, &hashes(&["h-again"]), old);
    record(&fx.store, None, p, &hashes(&["h-again"]), recent);
    assert_eq!(fx.store.prune_capture_state(now).expect("prune"), 2);
    assert_eq!(fx.store.capture_offset("old").expect("offset"), None);
    assert_eq!(fx.store.capture_offset("recent").expect("offset"), Some(20));
    let asked = hashes(&["h-old", "h-recent", "h-again"]);
    let seen: Vec<String> = fx
        .store
        .captures_seen(p, &asked)
        .expect("seen")
        .into_iter()
        .collect();
    assert_eq!(seen, hashes(&["h-again", "h-recent"]));
    assert_eq!(fx.store.prune_capture_state(now).expect("prune"), 0);
}

#[test]
fn s16_b164_the_nearest_memory_is_active_or_pending_in_the_project() {
    let fx = fixture();
    let alpha = project(&fx.store, "alpha");
    let beta = project(&fx.store, "beta");
    let text = "the importer runs before the indexer";
    let with_status = |p: ProjectId, content: &str, status: Status| {
        let mut input = note(p, content);
        input.memory.status = status;
        fx.store
            .insert_memory(input, Access::Local)
            .expect("insert")
            .id
    };
    let query = vec![fake_vector(text)];

    // Nothing in the project: no nearest memory.
    assert_eq!(
        fx.store.capture_nearest(alpha, &query).expect("nearest"),
        vec![None]
    );
    // The same text, but retracted, or in another project: still none.
    with_status(alpha, text, Status::Retracted);
    with_status(beta, text, Status::Active);
    assert_eq!(
        fx.store.capture_nearest(alpha, &query).expect("nearest"),
        vec![None]
    );

    let other = with_status(alpha, "the cache is kept per branch", Status::Active);
    let near = fx.store.capture_nearest(alpha, &query).expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(other));
    assert!(near[0].is_some_and(|n| n.cosine < 0.85), "{near:?}");

    // A pending capture that the rules before the judge made, with no row in
    // the judge's queue, does not count (B-176).
    let restated = "before the indexer the importer runs";
    let capture = || {
        let mut input = note(alpha, restated);
        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
        input.memory.status = Status::Pending;
        input
    };
    fx.store
        .insert_memory(capture(), Access::Local)
        .expect("an old-rule capture");
    let near = fx.store.capture_nearest(alpha, &query).expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(other));

    // A capture that waits in the judge's queue counts, though it is never
    // served.
    let pending = fx
        .store
        .insert_captures(vec![(capture(), "context".to_owned())])
        .expect("queue a capture")[0]
        .id;
    let near = fx.store.capture_nearest(alpha, &query).expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(pending));
    assert!(near[0].is_some_and(|n| n.cosine > 0.999), "{near:?}");

    // One answer per vector, and a vector that is not unit length is refused.
    let two = vec![
        fake_vector(text),
        fake_vector("the cache is kept per branch"),
    ];
    let near = fx.store.capture_nearest(alpha, &two).expect("nearest");
    assert_eq!(
        near.iter().map(|n| n.map(|n| n.id)).collect::<Vec<_>>(),
        vec![Some(pending), Some(other)]
    );
    assert_code(
        fx.store
            .capture_nearest(alpha, &[vec![0.5; sm_core::EMBED_DIM]]),
        ErrorCode::InvalidInput,
    );
    assert!(
        fx.store
            .capture_nearest(alpha, &[])
            .expect("none")
            .is_empty()
    );
}

#[test]
fn s16_b164_unit_mean_matches_the_stored_memory_vector() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let parts = [
        "first part of a long capture",
        "second part about something else",
    ];
    let id = fx
        .store
        .insert_memory(note_in_parts(p, &parts), Access::Local)
        .expect("insert")
        .id;
    let stored = fx
        .store
        .memory_vector(&id)
        .expect("vector")
        .expect("embedded");
    let mean = crate::unit_mean(&parts.iter().map(|t| fake_vector(t)).collect::<Vec<_>>());
    assert_eq!(stored.len(), mean.len());
    for (a, b) in stored.iter().zip(&mean) {
        assert!((a - b).abs() < 1e-6, "{a} {b}");
    }
    assert!(crate::unit_mean(&[]).is_empty());
}

#[test]
fn s16_b184_whether_a_session_was_read_from_its_start_survives_a_new_store() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let at = |offset, from_start| CapturePosition { offset, from_start };
    let save = |store: &Store, position| {
        store
            .record_capture_progress(&CaptureProgress {
                session: Some(("s-1", position)),
                project: p,
                hashes: &[],
                at: Timestamp::now(),
            })
            .expect("record capture progress");
    };
    save(&fx.store, at(120, true));
    let again = fx.reopen(stamp());
    assert_eq!(
        again.capture_position("s-1").expect("position"),
        Some(at(120, true))
    );
    assert_eq!(again.capture_offset("s-1").expect("offset"), Some(120));
    // A later position replaces both the offset and the flag.
    save(&again, at(480, false));
    assert_eq!(
        again.capture_position("s-1").expect("position"),
        Some(at(480, false))
    );
    assert_eq!(again.capture_position("s-2").expect("position"), None);
}
