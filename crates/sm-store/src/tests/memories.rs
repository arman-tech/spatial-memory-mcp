//! SPEC section 4 rules for memories: every field stored, duplicates
//! (B-29), supersede in one transaction, revisions, merges (B-31, B-73), the
//! forgetting cascade, shared memories refused locally, filtered forgets,
//! listing and streaming.

use serde_json::json;
use sm_core::hash::content_hash;
use sm_core::protocol::{ForgetFilter, RememberOutcome};
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope};
use sm_core::types::{
    AnchorRef, Link, LinkEnd, LinkSource, MemoryType, Origin, Relation, Scope, Status,
};

use super::*;
use crate::{
    ContentChange, ForgetCount, MapCluster, MapPointRow, MemoryListFilter, NewMapRun, Revision,
};

fn anchor(s: &str) -> AnchorRef {
    s.parse().expect("anchor")
}

fn no_filter() -> ForgetFilter {
    ForgetFilter {
        scope: None,
        memory_type: Vec::new(),
        status: Vec::new(),
        anchor: None,
        tag: None,
        created_before: None,
    }
}

fn link(src: LinkEnd, dst: LinkEnd) -> Link {
    Link {
        src,
        dst,
        relation: Relation::RelatesTo,
        source: LinkSource::Manual,
        extractor: None,
        verified_commit: None,
        verified_at: None,
    }
}

fn shared_note(project: ProjectId, content: &str) -> MemoryInput {
    let mut input = note(project, content);
    input.memory.origin = Origin::Shared;
    input.memory.provenance = Provenance::new(ProvenanceSource::Sync);
    input
}

fn keyword(store: &Store, text: &str) -> Vec<MemoryId> {
    let every = MemoryFilter {
        statuses: vec![Status::Active, Status::Superseded, Status::Retracted],
        ..MemoryFilter::default()
    };
    store
        .keyword_search(text, QueryScope::All, &every, 50)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

#[test]
fn s4_insert_stores_every_field() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let content = "Orders use soft delete. The deleted_at column marks them.";
    let mut prov = Provenance::new(ProvenanceSource::Human);
    prov.author = Some("ada".to_owned());
    prov.session_id = Some("session-1".to_owned());
    prov.commit_sha = Some("abc1234".to_owned());
    prov.file_path = Some("src/db.rs".to_owned());
    prov.tool = Some("remember".to_owned());
    prov.extra = Some(json!({"k": 1}));
    let mut m = NewMemory::new(p, content, prov.clone());
    m.memory_type = MemoryType::Decision;
    m.title = Some("Orders soft delete".to_owned());
    m.tags = vec!["db".to_owned(), " orders ".to_owned(), "db".to_owned()];
    m.anchors = vec![anchor("file:src/db.rs"), anchor("table:orders")];
    m.importance = 0.8;
    m.imported = Some(json!({"from": "v2", "id": 7}));
    let ins = fx
        .store
        .insert_memory(
            MemoryInput {
                memory: m,
                chunks: vec![
                    NewChunk::embedded(piece(0, "Orders use soft delete."), fake_vector("a")),
                    NewChunk::pending(piece(1, "The deleted_at column marks them.")),
                ],
            },
            Access::Local,
        )
        .expect("insert");
    assert_eq!(ins.outcome, RememberOutcome::Created);
    assert_eq!(ins.chunks, 2);
    assert_eq!(ins.superseded, None);

    let got = fx.store.get_memory(&ins.id).expect("get");
    assert_eq!(got.id, ins.id);
    assert_eq!(got.project_id, p);
    assert_eq!(got.memory_type, MemoryType::Decision);
    assert_eq!(got.status, Status::Active);
    assert_eq!(got.title.as_deref(), Some("Orders soft delete"));
    assert_eq!(got.content, content);
    assert_eq!(got.content_hash, content_hash(content));
    assert_eq!(got.supersedes_id, None);
    assert_eq!(got.origin, Origin::Local);
    assert_eq!(got.importance, 0.8);
    assert_eq!(got.tags, vec!["db".to_owned(), "orders".to_owned()]);
    assert_eq!(got.imported, Some(json!({"from": "v2", "id": 7})));
    assert_eq!(got.created_at, got.updated_at);
    assert_eq!(got.provenance, prov);
    let mut anchors = got.anchors.clone();
    anchors.sort_by(|a, b| (a.kind.as_str(), &a.name).cmp(&(b.kind.as_str(), &b.name)));
    assert_eq!(
        anchors,
        vec![anchor("file:src/db.rs"), anchor("table:orders")]
    );
    let id = ins.id.to_string();
    assert_eq!(rows_where(&fx.store, "chunks", "memory_id", &id), 2);
    assert_eq!(rows_where(&fx.store, "provenance", "memory_id", &id), 1);
    assert_eq!(rows_where(&fx.store, "memory_anchors", "memory_id", &id), 2);
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 1);

    // An imported memory keeps its id and creation time.
    let kept = MemoryId::new_v7();
    let created = Timestamp::from_unix_seconds(1_700_000_000);
    let mut imported = note(p, "imported from an older store");
    imported.memory.id = Some(kept);
    imported.memory.created_at = Some(created);
    let ins = fx
        .store
        .insert_memory(imported, Access::Local)
        .expect("insert");
    assert_eq!(ins.id, kept);
    let got = fx.store.get_memory(&kept).expect("get");
    assert_eq!((got.created_at, got.updated_at), (created, created));
}

#[test]
fn s4_invalid_items_fail_the_whole_batch() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut wrong_seq = note(p, "wrong seq");
    wrong_seq.chunks[0].seq = 1;
    let mut no_chunks = note(p, "no chunks");
    no_chunks.chunks.clear();
    let mut bad_importance = note(p, "bad importance");
    bad_importance.memory.importance = 1.5;
    let mut bad_tag = note(p, "bad tag");
    bad_tag.memory.tags = vec!["   ".to_owned()];
    let mut bad_vector = note(p, "bad vector");
    bad_vector.chunks[0].embedding = Some(vec![0.5; EMBED_DIM]);
    let mut short_vector = note(p, "short vector");
    short_vector.chunks[0].embedding = Some(vec![1.0]);
    let empty = note(p, "   ");
    let invalid = [
        wrong_seq,
        no_chunks,
        bad_importance,
        bad_tag,
        bad_vector,
        short_vector,
        empty,
    ];
    for bad in invalid {
        let what = bad.memory.content.clone();
        let r = fx
            .store
            .insert_memories(vec![note(p, "a valid memory"), bad], Access::Local);
        match r {
            Ok(v) => panic!("{what:?} was accepted: {v:?}"),
            Err(e) => assert_eq!(e.code, ErrorCode::InvalidInput, "{what:?}: {e:?}"),
        }
        assert_eq!(fx.store.memory_count().expect("count"), 0, "{what:?}");
    }
    assert_code(
        fx.store.insert_memories(
            vec![note(p, "valid"), note(ProjectId(999), "orphan")],
            Access::Local,
        ),
        ErrorCode::NotFound,
    );
    assert_eq!(fx.store.memory_count().expect("count"), 0);
    assert_eq!(table_count(&fx.store, "chunks"), 0);
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 0);
}

#[test]
fn s4_duplicates_return_the_existing_memory() {
    // Decision B-29.
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let first = add(&fx.store, p, "the same words");
    let again = fx
        .store
        .insert_memory(note(p, "the same words"), Access::Local)
        .expect("insert");
    assert_eq!(again.outcome, RememberOutcome::Duplicate);
    assert_eq!(again.id, first);
    assert_eq!(again.chunks, 0);
    assert_eq!(fx.store.memory_count().expect("count"), 1);

    // The same content in another project is another memory.
    let other = fx
        .store
        .insert_memory(note(q, "the same words"), Access::Local)
        .expect("insert");
    assert_eq!(other.outcome, RememberOutcome::Created);
    assert_ne!(other.id, first);

    // An existing id is a duplicate whatever the content.
    let mut same_id = note(p, "different words");
    same_id.memory.id = Some(first);
    let r = fx
        .store
        .insert_memory(same_id, Access::Local)
        .expect("insert");
    assert_eq!((r.outcome, r.id), (RememberOutcome::Duplicate, first));

    // Duplicates inside one batch.
    let batch = fx
        .store
        .insert_memories(
            vec![note(p, "batch twin"), note(p, "batch twin")],
            Access::Local,
        )
        .expect("insert");
    assert_eq!(batch[0].outcome, RememberOutcome::Created);
    assert_eq!(batch[1].outcome, RememberOutcome::Duplicate);
    assert_eq!(batch[1].id, batch[0].id);
    assert_eq!(fx.store.memory_count().expect("count"), 3);
}

#[test]
fn s4_supersede_marks_the_old_memory_in_one_transaction() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let old = add(&fx.store, p, "retries are three");
    let mut newer = note(p, "retries are five");
    newer.memory.supersedes_id = Some(old);
    let ins = fx
        .store
        .insert_memory(newer, Access::Local)
        .expect("insert");
    assert_eq!(ins.superseded, Some(old));
    let old_now = fx.store.get_memory(&old).expect("get");
    assert_eq!(old_now.status, Status::Superseded);
    assert!(old_now.updated_at >= old_now.created_at);
    let new_now = fx.store.get_memory(&ins.id).expect("get");
    assert_eq!(new_now.supersedes_id, Some(old));
    assert_eq!(new_now.status, Status::Active);

    // A missing old memory fails the insert; nothing is stored.
    let mut orphan = note(p, "supersedes nothing");
    orphan.memory.supersedes_id = Some(MemoryId::new_v7());
    assert_code(
        fx.store.insert_memory(orphan, Access::Local),
        ErrorCode::NotFound,
    );
    assert_eq!(fx.store.memory_count().expect("count"), 2);
    assert!(keyword(&fx.store, "supersedes nothing").is_empty());

    // The batch is one transaction: a later failure undoes the supersede.
    let target = add(&fx.store, p, "will stay active");
    let mut first = note(p, "replaces it");
    first.memory.supersedes_id = Some(target);
    let mut second = note(p, "fails later");
    second.memory.supersedes_id = Some(MemoryId::new_v7());
    assert_code(
        fx.store.insert_memories(vec![first, second], Access::Local),
        ErrorCode::NotFound,
    );
    assert_eq!(
        fx.store.get_memory(&target).expect("get").status,
        Status::Active
    );
    assert_eq!(fx.store.memory_count().expect("count"), 3);
}

#[test]
fn s4_revise_replaces_chunks_and_hash() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let id = fx
        .store
        .insert_memory(
            note_in_parts(p, &["old words here", "and more old words"]),
            Access::Local,
        )
        .expect("insert")
        .id;
    let before = fx.store.get_memory(&id).expect("get");
    let new_content = "fresh text entirely";
    let got = fx
        .store
        .revise_memory(
            &id,
            Revision {
                content: Some(ContentChange {
                    content: new_content.to_owned(),
                    chunks: vec![NewChunk::embedded(
                        piece(0, new_content),
                        fake_vector(new_content),
                    )],
                }),
                memory_type: Some(MemoryType::Pitfall),
                status: Some(Status::Retracted),
                tags: Some(vec!["revised".to_owned()]),
                anchors: Some(vec![anchor("service:orders")]),
                importance: Some(0.9),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect("revise");
    assert_eq!(got.content, new_content);
    assert_eq!(got.content_hash, content_hash(new_content));
    assert_ne!(got.content_hash, before.content_hash);
    assert_eq!(got.memory_type, MemoryType::Pitfall);
    assert_eq!(got.status, Status::Retracted);
    assert_eq!(got.tags, vec!["revised".to_owned()]);
    assert_eq!(got.anchors, vec![anchor("service:orders")]);
    assert_eq!(got.importance, 0.9);
    assert_eq!(got.created_at, before.created_at);
    assert!(got.updated_at >= before.updated_at);

    let ids = id.to_string();
    assert_eq!(rows_where(&fx.store, "chunks", "memory_id", &ids), 1);
    assert_eq!(table_count(&fx.store, "chunks_fts"), 1);
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 1);
    assert_eq!(
        rows_where(&fx.store, "memory_anchors", "memory_id", &ids),
        1
    );
    assert!(keyword(&fx.store, "old words").is_empty());
    assert_eq!(keyword(&fx.store, "fresh text"), vec![id]);

    // The new content is a duplicate source for later inserts.
    let dup = fx
        .store
        .insert_memory(note(p, new_content), Access::Local)
        .expect("insert");
    assert_eq!((dup.outcome, dup.id), (RememberOutcome::Duplicate, id));

    assert_code(
        fx.store
            .revise_memory(&MemoryId::new_v7(), Revision::default(), Access::Local),
        ErrorCode::NotFound,
    );
    assert_code(
        fx.store.revise_memory(
            &id,
            Revision {
                importance: Some(-0.1),
                ..Revision::default()
            },
            Access::Local,
        ),
        ErrorCode::InvalidInput,
    );
    assert_code(
        fx.store.revise_memory(
            &id,
            Revision {
                content: Some(ContentChange {
                    content: "no chunks".to_owned(),
                    chunks: Vec::new(),
                }),
                ..Revision::default()
            },
            Access::Local,
        ),
        ErrorCode::InvalidInput,
    );
    assert_eq!(fx.store.get_memory(&id).expect("get").content, new_content);
}

#[test]
fn s4_merge_supersedes_into_the_kept_memory() {
    // Decisions B-31 and B-73: a merge marks each merged memory superseded
    // and records the kept memory in its provenance; `supersedes_id` keeps
    // SPEC 4's meaning ("the memory this one replaces") and is not changed.
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let keep = add(&fx.store, p, "the kept memory");
    let a = add(&fx.store, p, "merged one");
    // `b` already replaced `x` and has its own provenance detail.
    let x = add(&fx.store, p, "an older memory b replaced");
    let mut b_input = note(p, "merged two");
    b_input.memory.supersedes_id = Some(x);
    b_input.memory.provenance.extra = Some(json!({"k": 1}));
    let b = fx
        .store
        .insert_memory(b_input, Access::Local)
        .expect("insert")
        .id;
    // `c` has provenance detail that is not a JSON object.
    let mut c_input = note(p, "merged three");
    c_input.memory.provenance.extra = Some(json!([1, 2]));
    let c = fx
        .store
        .insert_memory(c_input, Access::Local)
        .expect("insert")
        .id;

    let changed = fx
        .store
        .merge_memories(&keep, &[a, b, a, c], Access::Local)
        .expect("merge");
    assert_eq!(changed, vec![a, b, c]);
    let k = keep.to_string();
    let expected = [
        (a, None, json!({"merged_into": k})),
        (b, Some(x), json!({"k": 1, "merged_into": k})),
        (c, None, json!({"value": [1, 2], "merged_into": k})),
    ];
    for (m, supersedes, extra) in expected {
        let got = fx.store.get_memory(&m).expect("get");
        assert_eq!(got.status, Status::Superseded);
        assert_eq!(got.supersedes_id, supersedes, "supersedes_id of {m}");
        assert_eq!(got.provenance.extra, Some(extra), "provenance of {m}");
    }
    let kept = fx.store.get_memory(&keep).expect("get");
    assert_eq!(kept.status, Status::Active);
    assert_eq!(kept.supersedes_id, None);
    assert_eq!(kept.provenance.extra, None);
    assert_eq!(
        fx.store.get_memory(&x).expect("get").status,
        Status::Superseded
    );
    assert_eq!(fx.store.memory_count().expect("count"), 5);

    assert_code(
        fx.store.merge_memories(&keep, &[keep], Access::Local),
        ErrorCode::InvalidInput,
    );
    assert_code(
        fx.store
            .merge_memories(&MemoryId::new_v7(), &[a], Access::Local),
        ErrorCode::NotFound,
    );
    // A missing merged memory fails the whole group.
    let d = add(&fx.store, p, "merged four");
    assert_code(
        fx.store
            .merge_memories(&keep, &[d, MemoryId::new_v7()], Access::Local),
        ErrorCode::NotFound,
    );
    let d_now = fx.store.get_memory(&d).expect("get");
    assert_eq!(d_now.status, Status::Active);
    assert_eq!(d_now.provenance.extra, None);
}

#[test]
fn s4_forget_cascades_to_every_dependent_table() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut m = note_in_parts(p, &["forget me first", "forget me second"]);
    m.memory.anchors = vec![anchor("file:src/a.rs")];
    let m = fx.store.insert_memory(m, Access::Local).expect("insert").id;
    let n = add(&fx.store, p, "keep me for now");

    for l in [
        link(LinkEnd::Memory(m), LinkEnd::Memory(n)),
        link(LinkEnd::Memory(m), LinkEnd::Anchor(anchor("file:src/a.rs"))),
        link(
            LinkEnd::Anchor(anchor("service:billing")),
            LinkEnd::Memory(m),
        ),
    ] {
        assert!(fx.store.add_link(p, l).expect("link").created);
    }
    fx.store
        .put_map_run(NewMapRun {
            project: Some(p),
            input_hash: "cascade".to_owned(),
            params: json!({}),
            clustered: true,
            clusters: vec![MapCluster {
                region: 0,
                size: 2,
                label: None,
            }],
            points: [m, n]
                .iter()
                .map(|&id| MapPointRow {
                    memory_id: id,
                    region: Some(0),
                    x: 0.0,
                    y: 1.0,
                    central_rank: None,
                })
                .collect(),
        })
        .expect("map run");
    fx.store
        .record_injections("session-1", &[m, n], Timestamp::now())
        .expect("injections");

    let dependent = [
        ("memories", 2),
        ("provenance", 2),
        ("chunks", 3),
        ("chunks_fts", 3),
        ("chunk_vectors", 3),
        ("memory_anchors", 1),
        ("links", 3),
        ("map_points", 2),
        ("injections", 2),
    ];
    for (table, rows) in dependent {
        assert_eq!(table_count(&fx.store, table), rows, "{table} before");
    }

    assert_eq!(fx.store.forget(&[m, m], Access::Local).expect("forget"), 1);
    let after_m = [
        ("memories", 1),
        ("provenance", 1),
        ("chunks", 1),
        ("chunks_fts", 1),
        ("chunk_vectors", 1),
        ("memory_anchors", 0),
        ("links", 0),
        ("map_points", 1),
        ("injections", 1),
    ];
    for (table, rows) in after_m {
        assert_eq!(table_count(&fx.store, table), rows, "{table} after one");
    }
    assert!(keyword(&fx.store, "forget me").is_empty());
    assert_eq!(keyword(&fx.store, "keep me"), vec![n]);
    assert_code(fx.store.get_memory(&m), ErrorCode::NotFound);

    // An unknown id fails the whole call.
    assert_code(fx.store.forget(&[n, m], Access::Local), ErrorCode::NotFound);
    assert_eq!(fx.store.memory_count().expect("count"), 1);

    assert_eq!(fx.store.forget(&[n], Access::Local).expect("forget"), 1);
    for (table, _) in dependent {
        assert_eq!(table_count(&fx.store, table), 0, "{table} after both");
    }
    // The map run itself and the project's anchors stay.
    assert_eq!(table_count(&fx.store, "map_runs"), 1);
    assert!(table_count(&fx.store, "anchors") >= 1);
}

#[test]
fn s4_shared_memories_are_read_only_locally() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut shared = shared_note(p, "the team's shared memory");
    shared.memory.tags = vec!["team".to_owned()];
    let shared = fx
        .store
        .insert_memory(shared, Access::Sync)
        .expect("insert")
        .id;
    assert_eq!(
        fx.store.get_memory(&shared).expect("get").origin,
        Origin::Shared
    );
    let mut local = note(p, "a local memory");
    local.memory.tags = vec!["team".to_owned()];
    let local = fx
        .store
        .insert_memory(local, Access::Local)
        .expect("insert")
        .id;
    let other = add(&fx.store, p, "another local memory");

    assert_code(
        fx.store.forget(&[shared], Access::Local),
        ErrorCode::SharedReadOnly,
    );
    assert_code(
        fx.store.forget(&[other, shared], Access::Local),
        ErrorCode::SharedReadOnly,
    );
    assert!(fx.store.find_memory(&other).expect("find").is_some());
    assert_code(
        fx.store.revise_memory(
            &shared,
            Revision {
                title: Some("changed".to_owned()),
                ..Revision::default()
            },
            Access::Local,
        ),
        ErrorCode::SharedReadOnly,
    );
    assert_code(
        fx.store.merge_memories(&local, &[shared], Access::Local),
        ErrorCode::SharedReadOnly,
    );
    let mut newer = note(p, "replaces the shared one");
    newer.memory.supersedes_id = Some(shared);
    assert_code(
        fx.store.insert_memory(newer, Access::Local),
        ErrorCode::SharedReadOnly,
    );
    let unchanged = fx.store.get_memory(&shared).expect("get");
    assert_eq!(unchanged.status, Status::Active);
    assert_eq!(unchanged.title, None);
    assert_eq!(fx.store.memory_count().expect("count"), 3);

    // A filtered forget skips shared memories and counts them apart.
    let team = ForgetFilter {
        tag: Some("team".to_owned()),
        ..no_filter()
    };
    let expected = ForgetCount {
        matched: 1,
        shared_skipped: 1,
    };
    assert_eq!(
        fx.store
            .count_forget(p, &team, Access::Local)
            .expect("count"),
        expected
    );
    assert_eq!(
        fx.store
            .forget_by_filter(p, &team, Access::Local)
            .expect("forget"),
        expected
    );
    assert!(fx.store.find_memory(&local).expect("find").is_none());
    assert!(fx.store.find_memory(&shared).expect("find").is_some());

    // Sync may change and delete it.
    assert_eq!(
        fx.store
            .count_forget(p, &team, Access::Sync)
            .expect("count"),
        ForgetCount {
            matched: 1,
            shared_skipped: 0,
        }
    );
    let revised = fx
        .store
        .revise_memory(
            &shared,
            Revision {
                title: Some("synced title".to_owned()),
                ..Revision::default()
            },
            Access::Sync,
        )
        .expect("revise under sync");
    assert_eq!(revised.title.as_deref(), Some("synced title"));
    assert_eq!(fx.store.forget(&[shared], Access::Sync).expect("forget"), 1);
    assert_eq!(fx.store.memory_count().expect("count"), 1);
}

#[test]
fn s4_forget_by_filter_counts_what_it_deletes() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let old_time = Timestamp::from_unix_seconds(1_600_000_000);
    let cutoff = Timestamp::from_unix_seconds(1_650_000_000);
    let mut made = Vec::new();
    for (i, proj) in [p, p, p, p, q, q].into_iter().enumerate() {
        let mut m = note(proj, &format!("filtered memory {i}"));
        if i % 2 == 0 {
            m.memory.memory_type = MemoryType::Incident;
        }
        if i == 1 {
            m.memory.status = Status::Retracted;
        }
        if i < 2 || i == 4 {
            m.memory.created_at = Some(old_time);
        }
        if i == 3 {
            m.memory.anchors = vec![anchor("table:orders")];
        }
        made.push(fx.store.insert_memory(m, Access::Local).expect("insert").id);
    }

    assert_code(
        fx.store.count_forget(p, &no_filter(), Access::Local),
        ErrorCode::InvalidInput,
    );
    assert_code(
        fx.store.forget_by_filter(
            p,
            &ForgetFilter {
                scope: Some(Scope::All),
                ..no_filter()
            },
            Access::Local,
        ),
        ErrorCode::InvalidInput,
    );

    let check = |filter: ForgetFilter, expect: u64| {
        let counted = fx
            .store
            .count_forget(p, &filter, Access::Local)
            .expect("count");
        assert_eq!(counted.matched, expect, "{filter:?}");
        counted
    };
    let incidents = ForgetFilter {
        memory_type: vec![MemoryType::Incident],
        ..no_filter()
    };
    check(incidents.clone(), 2);
    check(
        ForgetFilter {
            scope: Some(Scope::All),
            ..incidents.clone()
        },
        3,
    );
    check(
        ForgetFilter {
            status: vec![Status::Retracted],
            ..no_filter()
        },
        1,
    );
    check(
        ForgetFilter {
            anchor: Some(anchor("table:orders")),
            ..no_filter()
        },
        1,
    );
    check(
        ForgetFilter {
            created_before: Some(cutoff),
            ..no_filter()
        },
        2,
    );
    check(
        ForgetFilter {
            created_before: Some(cutoff),
            scope: Some(Scope::All),
            ..no_filter()
        },
        3,
    );
    check(
        ForgetFilter {
            memory_type: vec![MemoryType::Incident],
            created_before: Some(cutoff),
            ..no_filter()
        },
        1,
    );

    let all_old = ForgetFilter {
        created_before: Some(cutoff),
        scope: Some(Scope::All),
        ..no_filter()
    };
    let counted = fx
        .store
        .count_forget(p, &all_old, Access::Local)
        .expect("count");
    let done = fx
        .store
        .forget_by_filter(p, &all_old, Access::Local)
        .expect("forget");
    assert_eq!(done, counted);
    assert_eq!(fx.store.memory_count().expect("count"), 3);
    for (i, id) in made.iter().enumerate() {
        let gone = i < 2 || i == 4;
        assert_eq!(
            fx.store.find_memory(id).expect("find").is_none(),
            gone,
            "memory {i}"
        );
    }
}

#[test]
fn s4_iter_memories_streams_every_memory() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let inputs: Vec<MemoryInput> = (0..600)
        .map(|i| note(p, &format!("streamed memory number {i}")))
        .collect();
    let inserted = fx
        .store
        .insert_memories(inputs, Access::Local)
        .expect("insert");
    let mut expected: Vec<MemoryId> = inserted.iter().map(|i| i.id).collect();
    expected.sort();
    let got: Vec<MemoryId> = fx
        .store
        .iter_memories()
        .map(|m| m.expect("memory").id)
        .collect();
    assert_eq!(got, expected);

    let empty = fixture();
    assert_eq!(empty.store.iter_memories().count(), 0);
}

#[test]
fn s4_list_memories_filters_and_pages() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let mut a = note(p, "list a");
    a.memory.memory_type = MemoryType::Runbook;
    a.memory.tags = vec!["ops".to_owned(), "db".to_owned()];
    let a = fx.store.insert_memory(a, Access::Local).expect("insert").id;
    let mut b = note(p, "list b");
    b.memory.status = Status::Superseded;
    b.memory.tags = vec!["ops".to_owned()];
    b.memory.anchors = vec![anchor("endpoint:/v1/orders")];
    let b = fx.store.insert_memory(b, Access::Local).expect("insert").id;
    let c = fx
        .store
        .insert_memory(shared_note(q, "list c"), Access::Sync)
        .expect("insert")
        .id;

    let list = |f: MemoryListFilter| -> Vec<MemoryId> {
        fx.store
            .list_memories(&f)
            .expect("list")
            .into_iter()
            .map(|m| m.id)
            .collect()
    };
    let mut every = vec![a, b, c];
    every.sort();
    assert_eq!(list(MemoryListFilter::default()), every);
    assert_eq!(
        list(MemoryListFilter {
            project: Some(q),
            ..MemoryListFilter::default()
        }),
        vec![c]
    );
    assert_eq!(
        list(MemoryListFilter {
            types: vec![MemoryType::Runbook],
            ..MemoryListFilter::default()
        }),
        vec![a]
    );
    assert_eq!(
        list(MemoryListFilter {
            statuses: vec![Status::Superseded],
            ..MemoryListFilter::default()
        }),
        vec![b]
    );
    let mut ops = vec![a, b];
    ops.sort();
    assert_eq!(
        list(MemoryListFilter {
            tags: vec!["ops".to_owned()],
            ..MemoryListFilter::default()
        }),
        ops
    );
    assert_eq!(
        list(MemoryListFilter {
            tags: vec!["ops".to_owned(), "db".to_owned()],
            ..MemoryListFilter::default()
        }),
        vec![a]
    );
    assert_eq!(
        list(MemoryListFilter {
            anchor: Some(anchor("endpoint:/v1/orders")),
            ..MemoryListFilter::default()
        }),
        vec![b]
    );
    assert_eq!(
        list(MemoryListFilter {
            origin: Some(Origin::Shared),
            ..MemoryListFilter::default()
        }),
        vec![c]
    );
    // Pages: two at a time, continuing after the last id.
    let first = list(MemoryListFilter {
        limit: 2,
        ..MemoryListFilter::default()
    });
    assert_eq!(first, every[..2]);
    let rest = list(MemoryListFilter {
        limit: 2,
        after: first.last().copied(),
        ..MemoryListFilter::default()
    });
    assert_eq!(rest, every[2..]);

    // get_memories keeps the order given and drops missing and repeated ids.
    let got: Vec<MemoryId> = fx
        .store
        .get_memories(&[c, MemoryId::new_v7(), a, c])
        .expect("get")
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(got, vec![c, a]);
    assert_eq!(fx.store.memory_count().expect("count"), 3);
}
