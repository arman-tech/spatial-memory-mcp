//! SPEC section 4, `injections`: what each session was shown, the later
//! time kept, and pruning after 30 days.

use sm_core::time::Timestamp;

use super::*;
use crate::INJECTION_RETENTION_DAYS;

const DAY: i64 = 86_400 * 1_000_000;

/// The stored time of one injection row, in microseconds.
fn stored_at(store: &Store, session: &str, id: MemoryId) -> Option<i64> {
    store
        .read("reading an injection", |tx| {
            let mut s = tx
                .prepare("SELECT at FROM injections WHERE session_id = ?1 AND memory_id = ?2")
                .ctx("reading an injection")?;
            let mut rows = s
                .query_map((session, id.to_string()), |r| r.get::<_, i64>(0))
                .ctx("reading an injection")?;
            rows.next().transpose().ctx("reading an injection")
        })
        .expect("read")
}

#[test]
fn s4_injections_record_and_keep_the_latest_time() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let m1 = add(&fx.store, p, "first injected");
    let m2 = add(&fx.store, p, "second injected");
    let t0 = Timestamp::from_unix_seconds(1_800_000_000);

    let n = fx
        .store
        .record_injections("session-1", &[m1, m2, MemoryId::new_v7()], t0)
        .expect("record");
    assert_eq!(n, 2, "the missing memory is skipped");
    let mut both = vec![m1, m2];
    both.sort();
    assert_eq!(fx.store.injected("session-1").expect("injected"), both);
    assert!(fx.store.injected("session-2").expect("injected").is_empty());

    // An earlier time does not move the record back.
    fx.store
        .record_injections("session-1", &[m1], t0.add_micros(-DAY))
        .expect("record");
    assert_eq!(stored_at(&fx.store, "session-1", m1), Some(t0.as_micros()));
    // A later time moves it forward.
    let later = t0.add_micros(3_600 * 1_000_000);
    fx.store
        .record_injections("session-1", &[m1], later)
        .expect("record");
    assert_eq!(
        stored_at(&fx.store, "session-1", m1),
        Some(later.as_micros())
    );
    assert_eq!(stored_at(&fx.store, "session-1", m2), Some(t0.as_micros()));

    // Sessions are separate.
    fx.store
        .record_injections("session-2", &[m2], t0)
        .expect("record");
    assert_eq!(fx.store.injected("session-2").expect("injected"), vec![m2]);
    assert_eq!(table_count(&fx.store, "injections"), 3);

    for blank in ["", "   "] {
        assert_code(
            fx.store.record_injections(blank, &[m1], t0),
            ErrorCode::InvalidInput,
        );
    }
    assert_eq!(
        fx.store
            .record_injections("session-3", &[], t0)
            .expect("record nothing"),
        0
    );
}

#[test]
fn s4_injections_older_than_30_days_are_pruned() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let m = add(&fx.store, p, "shown long ago");
    let now = Timestamp::from_unix_seconds(1_800_000_000);
    let limit = now.add_micros(-INJECTION_RETENTION_DAYS * DAY);
    for (session, at) in [
        ("recent", now.add_micros(-DAY)),
        ("exactly-30-days", limit),
        ("just-older", limit.add_micros(-1)),
        ("much-older", now.add_micros(-90 * DAY)),
    ] {
        fx.store
            .record_injections(session, &[m], at)
            .expect("record");
    }
    assert_eq!(fx.store.prune_injections(now).expect("prune"), 2);
    assert_eq!(fx.store.injected("recent").expect("injected"), vec![m]);
    assert_eq!(
        fx.store.injected("exactly-30-days").expect("injected"),
        vec![m]
    );
    assert!(
        fx.store
            .injected("just-older")
            .expect("injected")
            .is_empty()
    );
    assert!(
        fx.store
            .injected("much-older")
            .expect("injected")
            .is_empty()
    );
    assert_eq!(fx.store.prune_injections(now).expect("prune"), 0);
    assert_eq!(table_count(&fx.store, "injections"), 2);
}
