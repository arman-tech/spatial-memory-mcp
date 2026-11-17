//! The liveness rule on every local path (decision B-87): a memory is
//! superseded while at least one memory of its project that is not
//! retracted supersedes it. Forgetting or retracting a superseder restores
//! what it leaves unheld; `revise` refuses to make a held memory active.
//! Also the options that keep `updated_at` (B-92, B-93), the revision that
//! records a sync mark (B-90, B-93) and the marks-only listing (B-93).

use serde_json::json;
use sm_core::protocol::ForgetFilter;
use sm_core::time::Timestamp;
use sm_core::types::{Memory, Origin, Status};

use super::*;
use crate::{
    InsertOptions, MarkRow, MemoryListFilter, ReviseOptions, Revision, SyncMark,
    UNRESOLVED_SUPERSEDES,
};

fn status(store: &Store, id: &MemoryId) -> Status {
    store.get_memory(id).expect("get").status
}

/// Inserts a local memory that supersedes `target`.
fn supersede(store: &Store, project: ProjectId, target: MemoryId, content: &str) -> MemoryId {
    let mut input = note(project, content);
    input.memory.supersedes_id = Some(target);
    store
        .insert_memory(input, Access::Local)
        .expect("insert a superseder")
        .id
}

fn set_status(store: &Store, id: &MemoryId, s: Status, access: Access) -> Memory {
    store
        .revise_memory(
            id,
            Revision {
                status: Some(s),
                ..Revision::default()
            },
            access,
        )
        .expect("revise the status")
}

/// A local memory that names `target` only in provenance
/// `extra.unresolved_supersedes`, as sync stores a file whose target was
/// missing.
fn unresolved_holder(store: &Store, project: ProjectId, target: MemoryId) -> MemoryId {
    let mut input = note(project, "holds its target through provenance");
    input.memory.provenance.extra = Some(json!({ UNRESOLVED_SUPERSEDES: target.to_string() }));
    store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id
}

fn tag_filter(tag: &str) -> ForgetFilter {
    ForgetFilter {
        scope: None,
        memory_type: Vec::new(),
        status: Vec::new(),
        anchor: None,
        tag: Some(tag.to_owned()),
        created_before: None,
    }
}

const PAST: i64 = 1_700_000_000_000_000;

#[test]
fn s4_forget_restores_a_target_left_with_no_live_superseder() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");

    // One superseder: forgetting it restores the target, stamped.
    let mut old = note(p, "retries are three");
    old.memory.created_at = Some(Timestamp::from_micros(PAST));
    let old = fx
        .store
        .insert_memory(old, Access::Local)
        .expect("insert")
        .id;
    let newer = supersede(&fx.store, p, old, "retries are five");
    assert_eq!(status(&fx.store, &old), Status::Superseded);
    assert_eq!(fx.store.forget(&[newer], Access::Local).expect("forget"), 1);
    let back = fx.store.get_memory(&old).expect("get");
    assert_eq!(back.status, Status::Active);
    assert!(back.updated_at > Timestamp::from_micros(PAST));

    // Two superseders: the target stays superseded until both are gone.
    let t = add(&fx.store, p, "the cache lives in redis");
    let s1 = supersede(&fx.store, p, t, "the cache lives in memcached");
    let s2 = supersede(&fx.store, p, t, "the cache lives in process");
    fx.store.forget(&[s1], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &t), Status::Superseded);
    fx.store.forget(&[s2], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &t), Status::Active);

    // A chain: forgetting the head restores the middle only; the middle
    // still holds the tail.
    let c = add(&fx.store, p, "deploys are manual");
    let b = supersede(&fx.store, p, c, "deploys run from CI");
    let a = supersede(&fx.store, p, b, "deploys run from CI on tags");
    fx.store.forget(&[a], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &b), Status::Active);
    assert_eq!(status(&fx.store, &c), Status::Superseded);

    // Forgetting the superseder and its target together is fine.
    let x = add(&fx.store, p, "lint with flake8");
    let y = supersede(&fx.store, p, x, "lint with ruff");
    assert_eq!(fx.store.forget(&[y, x], Access::Local).expect("forget"), 2);

    // A memory held through `extra.unresolved_supersedes` stays superseded.
    let held = add(&fx.store, p, "logs go to stdout");
    let s = supersede(&fx.store, p, held, "logs go to a file");
    unresolved_holder(&fx.store, p, held);
    fx.store.forget(&[s], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &held), Status::Superseded);
}

#[test]
fn s4_forget_never_restores_a_retracted_merged_or_shared_target() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");

    // Retracted: the target was retracted after it was superseded.
    let retracted = add(&fx.store, p, "use tabs");
    let s = supersede(&fx.store, p, retracted, "use spaces");
    set_status(&fx.store, &retracted, Status::Retracted, Access::Local);
    fx.store.forget(&[s], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &retracted), Status::Retracted);

    // Merged: consolidation holds it, not a superseder (B-73).
    let merged = add(&fx.store, p, "the api is rest");
    let keep = add(&fx.store, p, "the api is rest over http two");
    let s = supersede(&fx.store, p, merged, "the api is graphql");
    fx.store
        .merge_memories(&keep, &[merged], Access::Local)
        .expect("merge");
    fx.store.forget(&[s], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &merged), Status::Superseded);

    // Shared: its status comes from its file, which only sync reads. Only
    // sync access may supersede a shared memory.
    let mut shared = note(p, "a shared rule");
    shared.memory.origin = Origin::Shared;
    let shared = fx
        .store
        .insert_memory(shared, Access::Sync)
        .expect("insert")
        .id;
    let mut s = note(p, "a local copy that replaces it");
    s.memory.supersedes_id = Some(shared);
    let s = fx.store.insert_memory(s, Access::Sync).expect("insert").id;
    fx.store.forget(&[s], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &shared), Status::Superseded);
}

#[test]
fn s4_only_a_superseder_of_the_same_project_holds_a_target() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let t = add(&fx.store, p, "the port is 8080");
    let here = supersede(&fx.store, p, t, "the port is 9090");
    let elsewhere = supersede(&fx.store, q, t, "the port is 7070");
    fx.store.forget(&[here], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &t), Status::Active);
    assert_eq!(status(&fx.store, &elsewhere), Status::Active);
}

#[test]
fn s4_forget_by_filter_restores_targets_too() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let t = add(&fx.store, p, "builds use make");
    let mut s = note(p, "builds use just");
    s.memory.supersedes_id = Some(t);
    s.memory.tags = vec!["draft".into()];
    fx.store.insert_memory(s, Access::Local).expect("insert");
    let kept = add(&fx.store, p, "an unrelated note");
    let got = fx
        .store
        .forget_by_filter(p, &tag_filter("draft"), Access::Local)
        .expect("forget");
    assert_eq!(got.matched, 1);
    assert_eq!(status(&fx.store, &t), Status::Active);
    assert_eq!(status(&fx.store, &kept), Status::Active);

    // A superseder and its target matched together.
    let mut t2 = note(p, "tests use unittest");
    t2.memory.tags = vec!["old".into()];
    let t2 = fx
        .store
        .insert_memory(t2, Access::Local)
        .expect("insert")
        .id;
    let mut s2 = note(p, "tests use pytest");
    s2.memory.supersedes_id = Some(t2);
    s2.memory.tags = vec!["old".into()];
    fx.store.insert_memory(s2, Access::Local).expect("insert");
    let got = fx
        .store
        .forget_by_filter(p, &tag_filter("old"), Access::Local)
        .expect("forget");
    assert_eq!(got.matched, 2);
    assert_eq!(fx.store.find_memory(&t2).expect("find"), None);
}

#[test]
fn s4_retracting_a_superseder_locally_restores_its_target() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let t = add(&fx.store, p, "queues are rabbitmq");
    let s = supersede(&fx.store, p, t, "queues are kafka");

    set_status(&fx.store, &s, Status::Retracted, Access::Local);
    assert_eq!(status(&fx.store, &t), Status::Active);

    // Live again: its target is superseded again.
    set_status(&fx.store, &s, Status::Active, Access::Local);
    assert_eq!(status(&fx.store, &t), Status::Superseded);

    // Under sync access nothing else changes: sync settles supersessions
    // itself (SPEC 9, "Supersedes").
    set_status(&fx.store, &s, Status::Retracted, Access::Sync);
    assert_eq!(status(&fx.store, &t), Status::Superseded);
    set_status(&fx.store, &t, Status::Active, Access::Sync);
    set_status(&fx.store, &s, Status::Active, Access::Sync);
    assert_eq!(status(&fx.store, &t), Status::Active);

    // An unresolved holder retracted locally releases its target.
    let u_target = add(&fx.store, p, "secrets live in vault");
    let s = supersede(&fx.store, p, u_target, "secrets live in sops");
    let holder = unresolved_holder(&fx.store, p, u_target);
    fx.store.forget(&[s], Access::Local).expect("forget");
    assert_eq!(status(&fx.store, &u_target), Status::Superseded);
    set_status(&fx.store, &holder, Status::Retracted, Access::Local);
    assert_eq!(status(&fx.store, &u_target), Status::Active);
}

#[test]
fn s4_revise_refuses_to_make_a_held_memory_active() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let t = add(&fx.store, p, "the timeout is 30 s");
    let s = supersede(&fx.store, p, t, "the timeout is 60 s");
    let before = fx.store.get_memory(&t).expect("get");

    let err = fx
        .store
        .revise_memory(
            &t,
            Revision {
                status: Some(Status::Active),
                tags: Some(vec!["kept".into()]),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect_err("refused");
    assert_eq!(err.code, ErrorCode::InvalidInput);
    assert!(err.message.contains(&s.to_string()), "{}", err.message);
    assert!(
        err.next_action.contains(&s.to_string()),
        "{}",
        err.next_action
    );
    // Nothing in the refused revision was applied.
    assert_eq!(fx.store.get_memory(&t).expect("get"), before);

    // Held through provenance, from a retracted target too.
    let t2 = add(&fx.store, p, "the region is us-east-1");
    let holder = unresolved_holder(&fx.store, p, t2);
    set_status(&fx.store, &t2, Status::Retracted, Access::Local);
    let err = fx
        .store
        .revise_memory(
            &t2,
            Revision {
                status: Some(Status::Active),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect_err("refused");
    assert_eq!(err.code, ErrorCode::InvalidInput);
    assert!(err.message.contains(&holder.to_string()), "{}", err.message);

    // Sync may: it settles the supersession afterwards.
    set_status(&fx.store, &t, Status::Active, Access::Sync);
    set_status(&fx.store, &t, Status::Superseded, Access::Sync);

    // Once the superseder is retracted, the target is active again and an
    // active-to-active revision is not refused.
    set_status(&fx.store, &s, Status::Retracted, Access::Local);
    assert_eq!(status(&fx.store, &t), Status::Active);
    set_status(&fx.store, &t, Status::Active, Access::Local);
}

#[test]
fn s4_insert_can_keep_the_target_updated_at() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let past = Timestamp::from_micros(PAST);
    let mut old = note(p, "the docs are in the wiki");
    old.memory.created_at = Some(past);
    old.memory.updated_at = Some(past);
    let old = fx
        .store
        .insert_memory(old, Access::Local)
        .expect("insert")
        .id;

    let mut s = note(p, "the docs are in the repository");
    s.memory.supersedes_id = Some(old);
    let got = fx
        .store
        .insert_memories_with(
            vec![s],
            Access::Local,
            InsertOptions {
                keep_target_updated_at: true,
            },
        )
        .expect("insert");
    assert_eq!(got[0].superseded, Some(old));
    let target = fx.store.get_memory(&old).expect("get");
    assert_eq!(target.status, Status::Superseded);
    assert_eq!(target.updated_at, past);

    // The default stamps it.
    let mut other = note(p, "the docs are on a site");
    other.memory.created_at = Some(past);
    other.memory.updated_at = Some(past);
    let other = fx
        .store
        .insert_memory(other, Access::Local)
        .expect("insert")
        .id;
    supersede(&fx.store, p, other, "the docs are on a better site");
    assert!(fx.store.get_memory(&other).expect("get").updated_at > past);
}

#[test]
fn s4_revise_can_keep_updated_at() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let past = Timestamp::from_micros(PAST);
    let mut m = note(p, "status only");
    m.memory.created_at = Some(past);
    m.memory.updated_at = Some(past);
    let m = fx.store.insert_memory(m, Access::Sync).expect("insert").id;
    let keep = ReviseOptions {
        keep_updated_at: true,
        ..ReviseOptions::default()
    };
    let got = fx
        .store
        .revise_memory_with(
            &m,
            Revision {
                status: Some(Status::Superseded),
                ..Revision::default()
            },
            Access::Sync,
            &keep,
        )
        .expect("revise");
    assert_eq!(got.status, Status::Superseded);
    assert_eq!(got.updated_at, past);
    let got = set_status(&fx.store, &m, Status::Active, Access::Sync);
    assert!(got.updated_at > past);
}

#[test]
fn s9_revise_records_a_sync_mark_in_the_same_transaction() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut input = note(p, "adopted and revised in one write");
    input.memory.provenance.extra = Some(json!({"k": 1}));
    let m = fx
        .store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id;
    let mark = SyncMark {
        file_path: ".spatial-memory/shared/note/adopted--12345678.md".into(),
        commit_sha: Some("4f2a9c1".into()),
        blob_id: "a".repeat(40),
        text_hash: "t1".into(),
    };
    let opts = ReviseOptions {
        sync_mark: Some(mark.clone()),
        ..ReviseOptions::default()
    };
    let tags = Revision {
        tags: Some(vec!["shared".into()]),
        ..Revision::default()
    };

    // Local access may not record a mark; nothing changes.
    let before = fx.store.get_memory(&m).expect("get");
    assert_code(
        fx.store
            .revise_memory_with(&m, tags.clone(), Access::Local, &opts),
        ErrorCode::InvalidInput,
    );
    assert_eq!(fx.store.get_memory(&m).expect("get"), before);

    let got = fx
        .store
        .revise_memory_with(&m, tags.clone(), Access::Sync, &opts)
        .expect("revise");
    assert_eq!(got.tags, vec!["shared".to_owned()]);
    assert_eq!(got.origin, Origin::Shared);
    assert_eq!(SyncMark::of(&got), Some(mark.clone()));
    assert_eq!(
        got.provenance.extra,
        Some(json!({"k": 1, "sync": {"blob": "a".repeat(40), "text_hash": "t1"}}))
    );
    assert!(got.updated_at > before.updated_at);

    // A mark with an empty revision records the mark alone, as record_sync.
    let second = SyncMark {
        blob_id: "b".repeat(40),
        ..mark
    };
    let got = fx
        .store
        .revise_memory_with(
            &m,
            Revision::default(),
            Access::Sync,
            &ReviseOptions {
                sync_mark: Some(second.clone()),
                ..ReviseOptions::default()
            },
        )
        .expect("revise");
    assert_eq!(SyncMark::of(&got), Some(second.clone()));

    assert_code(
        fx.store.revise_memory_with(
            &MemoryId::new_v7(),
            tags,
            Access::Sync,
            &ReviseOptions {
                sync_mark: Some(second),
                ..ReviseOptions::default()
            },
        ),
        ErrorCode::NotFound,
    );
}

/// What sync derives from a whole memory, as a [`MarkRow`].
fn row_of(m: &Memory) -> MarkRow {
    let extra = m.provenance.extra.as_ref();
    MarkRow {
        id: m.id,
        status: m.status,
        origin: m.origin,
        content_hash: m.content_hash.clone(),
        supersedes_id: m.supersedes_id,
        unresolved_supersedes: extra
            .and_then(|e| e.get(UNRESOLVED_SUPERSEDES))
            .and_then(|v| v.as_str())
            .and_then(|s| MemoryId::parse(s).ok()),
        merged: extra
            .and_then(|e| e.get("merged_into"))
            .is_some_and(|v| !v.is_null()),
        file_path: m.provenance.file_path.clone(),
        mark: SyncMark::of(m),
    }
}

#[test]
fn s9_list_marks_matches_the_whole_memories() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");

    let plain = add(&fx.store, p, "a plain local memory");
    let marked = add(&fx.store, p, "a memory sync marked");
    fx.store
        .record_sync(
            &marked,
            &SyncMark {
                file_path: ".spatial-memory/shared/note/marked--12345678.md".into(),
                commit_sha: None,
                blob_id: "c".repeat(40),
                text_hash: "t3".into(),
            },
        )
        .expect("record");
    supersede(
        &fx.store,
        p,
        plain,
        "a memory that supersedes the plain one",
    );
    unresolved_holder(&fx.store, p, MemoryId::new_v7());
    let keep = add(&fx.store, p, "the kept memory of a merge");
    let merged = add(&fx.store, p, "a memory merged into it");
    fx.store
        .merge_memories(&keep, &[merged], Access::Local)
        .expect("merge");
    for extra in [
        json!([1, 2]),
        json!({"merged_into": null}),
        json!({ UNRESOLVED_SUPERSEDES: 7 }),
        json!({ UNRESOLVED_SUPERSEDES: "not an id" }),
        json!({"sync": {"blob": 1, "text_hash": "t"}}),
        json!({"sync": "broken"}),
    ] {
        let mut odd = note(p, &format!("odd provenance {extra}"));
        odd.memory.provenance.file_path = Some("somewhere.md".into());
        odd.memory.provenance.extra = Some(extra);
        fx.store.insert_memory(odd, Access::Local).expect("insert");
    }
    add(&fx.store, q, "another project's memory");

    let rows = fx.store.list_marks(p).expect("list marks");
    let whole = fx
        .store
        .list_memories(&MemoryListFilter {
            project: Some(p),
            ..MemoryListFilter::default()
        })
        .expect("list");
    assert_eq!(rows.len(), 12);
    let expected: Vec<MarkRow> = whole.iter().map(row_of).collect();
    assert_eq!(rows, expected);
    assert!(rows.iter().any(|r| r.merged));
    assert!(rows.iter().any(|r| r.unresolved_supersedes.is_some()));
    assert!(rows.iter().any(|r| r.mark.is_some()));
}
