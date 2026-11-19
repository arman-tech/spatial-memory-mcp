//! B-157 (SPEC 16): an automatic capture is stored `pending` and is never
//! served, whatever a caller asks for; a save of the same text replaces it.

use sm_core::hash::content_hash;
use sm_core::protocol::RememberOutcome;
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{AnchorRef, Link, LinkEnd, LinkSource, Relation, Status};

use super::*;

fn capture(project: ProjectId, content: &str) -> MemoryInput {
    let mut input = note(project, content);
    input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
    input.memory.status = Status::Pending;
    input
}

/// The filters a caller could send: the default, every status by name, and
/// `pending` alone.
fn filters() -> Vec<MemoryFilter> {
    vec![
        MemoryFilter::default(),
        MemoryFilter {
            statuses: Status::ALL.to_vec(),
            ..MemoryFilter::default()
        },
        MemoryFilter {
            statuses: vec![Status::Pending],
            ..MemoryFilter::default()
        },
    ]
}

fn nearest(store: &Store, text: &str, scope: QueryScope, f: &MemoryFilter) -> Vec<MemoryId> {
    store
        .nearest(&VectorQuery {
            vector: &fake_vector(text),
            scope,
            filter: f.clone(),
            limit: 10,
        })
        .expect("vector search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

fn keyword(store: &Store, text: &str, scope: QueryScope, f: &MemoryFilter) -> Vec<MemoryId> {
    store
        .keyword_search(text, scope, f, 10)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

#[test]
fn s16_pending_captures_are_never_searched() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "the uploader retries five times before it gives up";
    let pending = fx
        .store
        .insert_memory(capture(p, text), Access::Local)
        .expect("insert a capture")
        .id;
    assert_eq!(
        fx.store.get_memory(&pending).expect("get").status,
        Status::Pending
    );
    let active = add(&fx.store, p, "the uploader logs each retry");

    for f in filters() {
        for scope in [QueryScope::Project(p), QueryScope::All] {
            let by_vector = nearest(&fx.store, text, scope, &f);
            assert!(!by_vector.contains(&pending), "{f:?} {scope:?}");
            let by_word = keyword(&fx.store, "uploader", scope, &f);
            assert!(!by_word.contains(&pending), "{f:?} {scope:?}");
            // The searches ran: the active memory is found wherever the
            // filter allows `active`.
            if f.served_statuses().contains(&Status::Active) {
                assert!(by_vector.contains(&active), "{f:?} {scope:?}");
                assert_eq!(by_word, vec![active], "{f:?} {scope:?}");
            } else {
                assert!(by_vector.is_empty() && by_word.is_empty(), "{f:?}");
            }
        }
    }
}

#[test]
fn s16_pending_captures_are_never_reached_by_the_walk() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let file: AnchorRef = "file:src/scheduler.rs".parse().expect("anchor");
    let start = add(&fx.store, p, "the scheduler starts the uploader");

    let mut input = capture(p, "the scheduler runs at night");
    input.memory.anchors = vec![file.clone()];
    let pending = fx
        .store
        .insert_memory(input, Access::Local)
        .expect("insert a capture")
        .id;
    let mut input = note(p, "the scheduler reads its hours from the config");
    input.memory.anchors = vec![file.clone()];
    let neighbour = fx
        .store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id;
    for dst in [pending, neighbour] {
        fx.store
            .add_link(
                p,
                Link {
                    src: LinkEnd::Memory(start),
                    dst: LinkEnd::Memory(dst),
                    relation: Relation::RelatesTo,
                    source: LinkSource::Manual,
                    extractor: None,
                    verified_commit: None,
                    verified_at: None,
                },
            )
            .expect("link");
    }

    for statuses in [Vec::new(), Status::ALL.to_vec(), vec![Status::Pending]] {
        let from_memory: Vec<MemoryId> = fx
            .store
            .related_to_memory(&start, &statuses)
            .expect("walk")
            .into_iter()
            .map(|h| h.memory_id)
            .collect();
        let from_anchor: Vec<MemoryId> = fx
            .store
            .related_to_anchor(p, &file, &statuses)
            .expect("walk")
            .into_iter()
            .map(|h| h.memory_id)
            .collect();
        assert!(!from_memory.contains(&pending), "{statuses:?}");
        assert!(!from_anchor.contains(&pending), "{statuses:?}");
        let wants_active = statuses.is_empty() || statuses.contains(&Status::Active);
        assert_eq!(
            from_memory.contains(&neighbour),
            wants_active,
            "{statuses:?}"
        );
        assert_eq!(
            from_anchor.contains(&neighbour),
            wants_active,
            "{statuses:?}"
        );
    }
}

#[test]
fn s16_a_save_replaces_a_pending_capture_of_the_same_text() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let text = "deploys go out on tuesdays only";

    let first = fx
        .store
        .insert_memory(capture(p, text), Access::Local)
        .expect("insert a capture");
    assert_eq!(first.outcome, RememberOutcome::Created);
    // The same capture again is a duplicate of the pending one.
    let again = fx
        .store
        .insert_memory(capture(p, text), Access::Local)
        .expect("insert a capture again");
    assert_eq!(
        (again.outcome, again.id),
        (RememberOutcome::Duplicate, first.id)
    );

    // A save of the same text is stored, and the capture is deleted with
    // its chunk, vector and keyword entries.
    let saved = fx
        .store
        .insert_memory(note(p, text), Access::Local)
        .expect("save");
    assert_eq!(saved.outcome, RememberOutcome::Created);
    assert_ne!(saved.id, first.id);
    assert!(fx.store.find_memory(&first.id).expect("find").is_none());
    assert_eq!(
        fx.store.get_memory(&saved.id).expect("get").status,
        Status::Active
    );
    assert_eq!(
        rows_where(&fx.store, "memories", "content_hash", &content_hash(text)),
        1
    );
    assert_eq!(table_count(&fx.store, "chunks"), 1);
    // The capture's keyword and vector rows went with its chunk: each index
    // holds the save's one row, read directly, not through a filter that
    // already drops pending memories.
    assert_eq!(table_count(&fx.store, "chunks_fts"), 1);
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 1);
    assert_eq!(
        keyword(
            &fx.store,
            "tuesdays",
            QueryScope::All,
            &MemoryFilter::default()
        ),
        vec![saved.id]
    );

    // A capture of text already saved is a duplicate of the save, which
    // stays active.
    let late = fx
        .store
        .insert_memory(capture(p, text), Access::Local)
        .expect("insert a late capture");
    assert_eq!(
        (late.outcome, late.id),
        (RememberOutcome::Duplicate, saved.id)
    );
    assert_eq!(
        fx.store.get_memory(&saved.id).expect("get").status,
        Status::Active
    );

    // Another project's pending capture of the same text is not touched.
    let other = fx
        .store
        .insert_memory(capture(q, text), Access::Local)
        .expect("insert in another project");
    assert_eq!(other.outcome, RememberOutcome::Created);
    let saved_in_p_again = fx
        .store
        .insert_memory(note(p, text), Access::Local)
        .expect("save again");
    assert_eq!(saved_in_p_again.outcome, RememberOutcome::Duplicate);
    assert_eq!(
        fx.store.get_memory(&other.id).expect("get").status,
        Status::Pending
    );
}

/// B-176: superseding, retracting or merging an unjudged capture would give
/// it a status a caller can ask for, and any other revise would return its
/// text, so each is refused and nothing changes. The user's own revise to
/// active still keeps it (B-167).
#[test]
fn s16_b176_a_pending_capture_is_never_superseded_retracted_merged_or_edited() {
    use crate::Revision;
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let pending = fx
        .store
        .insert_memory(capture(p, "the cache is warmed at boot"), Access::Local)
        .expect("insert a capture")
        .id;
    let other = fx
        .store
        .insert_memory(note(p, "a saved memory"), Access::Local)
        .expect("save")
        .id;
    let status = |s: Status| Revision {
        status: Some(s),
        ..Revision::default()
    };

    for s in [Status::Superseded, Status::Retracted] {
        assert_code(
            fx.store.revise_memory(&pending, status(s), Access::Local),
            ErrorCode::InvalidInput,
        );
    }
    let tags = Revision {
        tags: Some(vec!["cache".to_owned()]),
        ..Revision::default()
    };
    let title = Revision {
        title: Some("Cache warming".to_owned()),
        ..Revision::default()
    };
    for rev in [tags, title, Revision::default()] {
        assert_code(
            fx.store.revise_memory(&pending, rev, Access::Local),
            ErrorCode::InvalidInput,
        );
    }
    let mut superseder = note(p, "the cache is warmed on first use");
    superseder.memory.supersedes_id = Some(pending);
    assert_code(
        fx.store.insert_memory(superseder, Access::Local),
        ErrorCode::InvalidInput,
    );
    assert_code(
        fx.store.merge_memories(&other, &[pending], Access::Local),
        ErrorCode::InvalidInput,
    );
    assert_code(
        fx.store.merge_memories(&pending, &[other], Access::Local),
        ErrorCode::InvalidInput,
    );
    let held = fx.store.get_memory(&pending).expect("get");
    assert_eq!(held.status, Status::Pending);
    assert!(held.tags.is_empty() && held.title.is_none(), "{held:?}");
    assert_eq!(
        fx.store.get_memory(&other).expect("get").status,
        Status::Active
    );
    assert_eq!(table_count(&fx.store, "memories"), 2);
    for f in filters() {
        assert!(
            !nearest(
                &fx.store,
                "the cache is warmed at boot",
                QueryScope::All,
                &f
            )
            .contains(&pending)
        );
    }

    let kept = fx
        .store
        .revise_memory(&pending, status(Status::Active), Access::Local)
        .expect("the user's own revise to active");
    assert_eq!(kept.status, Status::Active);
}

// --- B-181: no link shows a pending capture ---------------------------------

fn manual(src: LinkEnd, dst: LinkEnd) -> Link {
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

/// Whether a link has memory `id` at either end.
fn names(link: &Link, id: MemoryId) -> bool {
    link.src == LinkEnd::Memory(id) || link.dst == LinkEnd::Memory(id)
}

fn hit_ids(walk: &[crate::WalkHit]) -> Vec<MemoryId> {
    walk.iter().map(|h| h.memory_id).collect()
}

#[track_caller]
fn assert_waits_for_the_judge<T: std::fmt::Debug>(r: Result<T, CoreError>) {
    let e = r.expect_err("a pending capture is refused");
    assert_eq!(e.code, ErrorCode::InvalidInput, "{e:?}");
    assert!(e.message.contains("waits for the judge"), "{}", e.message);
}

/// The links of a store where a pending capture sits between active
/// memories: `start` links to it, it links to `beyond` and to the anchor
/// `service`, and it shares the anchor `file` with `neighbour`. `control`
/// is linked from `start` and to `service`.
struct Linked {
    fx: Fixture,
    p: ProjectId,
    file: AnchorRef,
    service: AnchorRef,
    start: MemoryId,
    pending: MemoryId,
    neighbour: MemoryId,
    beyond: MemoryId,
    control: MemoryId,
}

fn linked() -> Linked {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let file: AnchorRef = "file:src/uploader.rs".parse().expect("anchor");
    let service: AnchorRef = "service:uploader".parse().expect("anchor");
    let start = add(&fx.store, p, "the scheduler starts the uploader");
    let mut input = capture(p, "the uploader retries five times before it gives up");
    input.memory.anchors = vec![file.clone()];
    // Queued for the judge, as every capture is now (B-196).
    let pending = fx
        .store
        .insert_captures(vec![(input, String::new())])
        .expect("insert a capture")[0]
        .id;
    let mut input = note(p, "the uploader reads its limits from the config");
    input.memory.anchors = vec![file.clone()];
    let neighbour = fx
        .store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id;
    let beyond = add(&fx.store, p, "the uploader writes a log line per retry");
    let control = add(&fx.store, p, "the scheduler runs at night");
    for (src, dst) in [
        (LinkEnd::Memory(start), LinkEnd::Memory(pending)),
        (LinkEnd::Memory(pending), LinkEnd::Memory(beyond)),
        (LinkEnd::Memory(pending), LinkEnd::Anchor(service.clone())),
        (LinkEnd::Memory(start), LinkEnd::Memory(control)),
        (LinkEnd::Memory(control), LinkEnd::Anchor(service.clone())),
    ] {
        fx.store.add_link(p, manual(src, dst)).expect("link");
    }
    Linked {
        fx,
        p,
        file,
        service,
        start,
        pending,
        neighbour,
        beyond,
        control,
    }
}

/// B-181 (SPEC 16; CAPTURE-RCA B4): the walk behind `related` never
/// passes through a capture that waits for the judge, so nothing is
/// reached through one and no `via` names one, from a memory or from an
/// anchor; a pending start is refused. Reaching it is not enough to fail:
/// the B-157 test above already shows it is never reported.
#[test]
fn s16_b181_the_walk_never_passes_through_a_pending_capture() {
    let l = linked();
    let s = &l.fx.store;
    for statuses in [Vec::new(), Status::ALL.to_vec()] {
        let from_memory = s.related_to_memory(&l.start, &statuses).expect("walk");
        let from_file = s.related_to_anchor(l.p, &l.file, &statuses).expect("walk");
        let from_service = s
            .related_to_anchor(l.p, &l.service, &statuses)
            .expect("walk");
        for (walk, what) in [
            (&from_memory, "from the start"),
            (&from_file, "from the file"),
            (&from_service, "from the service"),
        ] {
            for h in walk {
                assert!(
                    !h.via.iter().any(|e| names(e, l.pending)),
                    "{what}: {} is reached through the capture: {:?}",
                    h.memory_id,
                    h.via
                );
            }
        }
        // Through the capture, the start would reach `beyond` and
        // `neighbour`, and each anchor would reach `beyond` too.
        assert_eq!(hit_ids(&from_memory), vec![l.control], "{statuses:?}");
        assert_eq!(hit_ids(&from_file), vec![l.neighbour], "{statuses:?}");
        assert_eq!(
            hit_ids(&from_service),
            vec![l.control, l.start],
            "{statuses:?}"
        );
    }
    assert_waits_for_the_judge(s.related_to_memory(&l.pending, &[]));
}

/// B-181 (SPEC 16; CAPTURE-RCA B4): `links_of` refuses a capture that
/// waits for the judge as its end, and leaves out a link whose other end
/// is one, from a memory or from an anchor. The store keeps the links.
#[test]
fn s16_b181_links_of_never_shows_a_pending_capture() {
    let l = linked();
    let s = &l.fx.store;
    assert_waits_for_the_judge(s.links_of(
        l.p,
        &LinkEnd::Memory(l.pending),
        crate::head_only(None),
    ));
    for (end, others) in [
        (LinkEnd::Memory(l.start), vec![LinkEnd::Memory(l.control)]),
        (LinkEnd::Memory(l.beyond), Vec::new()),
        (
            LinkEnd::Anchor(l.service.clone()),
            vec![LinkEnd::Memory(l.control)],
        ),
    ] {
        let shown = s
            .links_of(l.p, &end, crate::head_only(None))
            .expect("links");
        for v in &shown {
            assert!(!names(&v.link, l.pending), "{end:?}: {:?}", v.link);
        }
        let got: Vec<LinkEnd> = shown
            .iter()
            .map(|v| {
                if v.link.src == end {
                    v.link.dst.clone()
                } else {
                    v.link.src.clone()
                }
            })
            .collect();
        assert_eq!(got, others, "{end:?}");
    }
    let kept = s.project_links(l.p).expect("links");
    assert_eq!(kept.iter().filter(|k| names(k, l.pending)).count(), 3);
}

// --- B-190: a rejected capture stays pending and is never served -----------

/// B-190 (SPEC 16): a capture the judge rejected is kept, still pending,
/// for 30 days. It is never searched, never reached by the walk, never shown
/// by `links_of`, never taken by the judge again and never a near neighbour
/// that keeps a later capture out. The walk and link checks of B-181 run
/// again with the capture rejected. The queue and near-neighbour checks are
/// made before the rejection too, so they fail if it changes nothing
/// (B-196).
#[test]
fn s16_b190_a_rejected_capture_is_never_searched_walked_or_linked() {
    let l = linked();
    let s = &l.fx.store;
    let text = "the uploader retries five times before it gives up";
    // Before the rejection the judge takes it and it is a near neighbour
    // for its own text, so the checks at the end test the rejection (B-196).
    let queued: Vec<MemoryId> = s
        .judge_queue(25, 100_000)
        .expect("queue")
        .iter()
        .map(|i| i.id)
        .collect();
    assert_eq!(queued, vec![l.pending]);
    let near = s
        .capture_nearest(l.p, &[fake_vector(text)])
        .expect("nearest");
    assert_eq!(near[0].map(|n| n.id), Some(l.pending));

    let settled = s
        .settle_captures(vec![crate::Settle::Reject {
            id: l.pending,
            reason: "narration".to_owned(),
        }])
        .expect("reject");
    assert_eq!(settled, vec![crate::Settled::Rejected]);
    let m = s.get_memory(&l.pending).expect("kept");
    assert_eq!((m.status, m.content.as_str()), (Status::Pending, text));

    for f in filters() {
        for scope in [QueryScope::Project(l.p), QueryScope::All] {
            assert!(
                !nearest(s, text, scope, &f).contains(&l.pending),
                "{f:?} {scope:?}"
            );
            assert!(
                !keyword(s, "retries", scope, &f).contains(&l.pending),
                "{f:?} {scope:?}"
            );
        }
    }
    for statuses in [Vec::new(), Status::ALL.to_vec()] {
        let from_memory = s.related_to_memory(&l.start, &statuses).expect("walk");
        let from_service = s
            .related_to_anchor(l.p, &l.service, &statuses)
            .expect("walk");
        for walk in [&from_memory, &from_service] {
            for h in walk {
                assert_ne!(h.memory_id, l.pending);
                assert!(!h.via.iter().any(|e| names(e, l.pending)), "{:?}", h.via);
            }
        }
        assert_eq!(hit_ids(&from_memory), vec![l.control], "{statuses:?}");
    }
    assert_waits_for_the_judge(s.related_to_memory(&l.pending, &[]));
    assert_waits_for_the_judge(s.links_of(
        l.p,
        &LinkEnd::Memory(l.pending),
        crate::head_only(None),
    ));
    let shown = s
        .links_of(l.p, &LinkEnd::Memory(l.start), crate::head_only(None))
        .expect("links");
    assert!(
        shown.iter().all(|v| !names(&v.link, l.pending)),
        "{shown:?}"
    );

    // Never taken by the judge again, and no near neighbour for a capture.
    assert!(s.judge_queue(25, 100_000).expect("queue").is_empty());
    let near = s
        .capture_nearest(l.p, &[fake_vector(text)])
        .expect("nearest");
    assert_ne!(near[0].map(|n| n.id), Some(l.pending));
}
