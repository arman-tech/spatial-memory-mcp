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
