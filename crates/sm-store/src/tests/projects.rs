//! SPEC section 4, `projects`: upsert by identity (B-20), rename, delete with
//! cascade, per-project counts.

use serde_json::json;
use sm_core::protocol::ProjectReply;
use sm_core::time::Timestamp;
use sm_core::types::{AnchorRef, Link, LinkEnd, LinkSource, MemoryType, Relation, Status};

use super::*;
use crate::{MapCluster, MapPointRow, NewMapRun, ProjectRecord};

fn anchor(s: &str) -> AnchorRef {
    s.parse().expect("anchor")
}

/// A second project with the same display name but another root.
fn elsewhere(name: &str) -> ProjectIdentity {
    ProjectIdentity {
        root: PathBuf::from(format!("/elsewhere/{name}")),
        root_path: format!("/elsewhere/{name}"),
        root_key: format!("/elsewhere/{name}"),
        ..identity(name)
    }
}

#[test]
fn s4_upsert_keeps_the_name_and_refreshes_the_root() {
    let fx = fixture();
    let first = fx.store.upsert_project(&identity("alpha")).expect("upsert");
    assert_eq!(first.name, "alpha");
    assert_eq!(first.root_path.as_deref(), Some("/work/alpha"));
    assert_eq!(
        fx.store
            .project_by_identity(&identity("alpha").key())
            .expect("find"),
        Some(first.clone())
    );
    assert_eq!(
        fx.store.project_by_identity("path:/nowhere").expect("find"),
        None
    );

    // The same identity is the same project, and keeps a rename.
    fx.store
        .rename_project("alpha", "Alpha Service")
        .expect("rename");
    let again = fx.store.upsert_project(&identity("alpha")).expect("upsert");
    assert_eq!(again.id, first.id);
    assert_eq!(again.name, "Alpha Service");

    // A repository with a remote is found by the remote when its clone moves.
    let cloned = ProjectIdentity {
        git_remote: Some("github.com/org/repo".to_owned()),
        ..identity("repo")
    };
    let p = fx.store.upsert_project(&cloned).expect("upsert");
    assert_eq!(p.git_remote.as_deref(), Some("github.com/org/repo"));
    let moved = ProjectIdentity {
        root: PathBuf::from("/moved/repo"),
        root_path: "/moved/repo".to_owned(),
        root_key: "/moved/repo".to_owned(),
        ..cloned.clone()
    };
    let p2 = fx.store.upsert_project(&moved).expect("upsert");
    assert_eq!(p2.id, p.id);
    assert_eq!(p2.root_path.as_deref(), Some("/moved/repo"));
    assert_eq!(fx.store.project(p.id).expect("project"), p2);
    assert_eq!(fx.store.projects().expect("projects").len(), 2);

    assert_code(fx.store.project(ProjectId(999)), ErrorCode::NotFound);
}

#[test]
fn s10_ensure_project_finds_or_creates_by_identity() {
    let fx = fixture();
    let resolved = fx.store.upsert_project(&identity("alpha")).expect("upsert");

    // A record with an identity the store already holds returns that
    // project unchanged, whatever the record's other fields say.
    let same = ProjectRecord {
        identity: identity("alpha").key(),
        name: "other name".to_owned(),
        git_remote: Some("github.com/org/other".to_owned()),
        root_path: None,
    };
    assert_eq!(
        fx.store.ensure_project(&same).expect("ensure"),
        (resolved.clone(), false)
    );

    // A remote known only from a v1 export, and a project known only by name.
    let remote = ProjectRecord {
        identity: "git:github.com/org/repo".to_owned(),
        name: "repo".to_owned(),
        git_remote: Some("github.com/org/repo".to_owned()),
        root_path: None,
    };
    let (by_remote, created) = fx.store.ensure_project(&remote).expect("ensure");
    assert!(created);
    assert_eq!(by_remote.name, "repo");
    assert_eq!(by_remote.git_remote.as_deref(), Some("github.com/org/repo"));
    assert_eq!(by_remote.root_path, None);
    let named = ProjectRecord {
        identity: "name:legacy notes".to_owned(),
        name: "legacy notes".to_owned(),
        git_remote: None,
        root_path: None,
    };
    let (by_name, created) = fx.store.ensure_project(&named).expect("ensure");
    assert!(created);
    assert_eq!(
        fx.store.ensure_project(&named).expect("ensure again"),
        (by_name.clone(), false)
    );

    // A working directory in a clone of that remote resolves to the project
    // the record created.
    let clone = ProjectIdentity {
        git_remote: Some("github.com/org/repo".to_owned()),
        ..identity("repo")
    };
    let resolved_clone = fx.store.upsert_project(&clone).expect("upsert");
    assert_eq!(resolved_clone.id, by_remote.id);

    // Every project with its identity key, in id order.
    assert_eq!(
        fx.store.project_records().expect("records"),
        vec![
            (resolved, identity("alpha").key()),
            (resolved_clone, "git:github.com/org/repo".to_owned()),
            (by_name, "name:legacy notes".to_owned()),
        ]
    );

    for bad in [
        ProjectRecord {
            identity: "legacy".to_owned(),
            ..named.clone()
        },
        ProjectRecord {
            identity: "name:".to_owned(),
            ..named.clone()
        },
        ProjectRecord {
            name: " padded".to_owned(),
            ..named.clone()
        },
        ProjectRecord {
            name: String::new(),
            ..named.clone()
        },
    ] {
        assert_code(fx.store.ensure_project(&bad), ErrorCode::InvalidInput);
    }
    assert_eq!(fx.store.projects().expect("projects").len(), 3);
}

#[test]
fn s4_rename_project_checks_names() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let id = add(&fx.store, p, "stays attached");
    add(&fx.store, p, "also attached");
    project(&fx.store, "beta");

    let reply = fx.store.rename_project("alpha", "gamma").expect("rename");
    assert_eq!(
        reply,
        ProjectReply {
            name: "gamma".to_owned(),
            renamed_from: Some("alpha".to_owned()),
            deleted: false,
            memories: 2,
        }
    );
    assert_eq!(fx.store.project(p).expect("project").name, "gamma");
    assert_eq!(fx.store.get_memory(&id).expect("get").project_id, p);
    assert!(
        fx.store
            .projects_by_name("alpha")
            .expect("by name")
            .is_empty()
    );

    for bad in ["", "   ", " gamma2", "gamma2 "] {
        assert_code(
            fx.store.rename_project("gamma", bad),
            ErrorCode::InvalidInput,
        );
    }
    assert_code(
        fx.store.rename_project("missing", "delta"),
        ErrorCode::NotFound,
    );
    assert_code(
        fx.store.rename_project("gamma", "beta"),
        ErrorCode::InvalidInput,
    );
    // Renaming to its own name is allowed and changes nothing.
    assert_eq!(
        fx.store
            .rename_project("gamma", "gamma")
            .expect("rename")
            .name,
        "gamma"
    );

    // An ambiguous name is refused.
    fx.store.upsert_project(&elsewhere("beta")).expect("upsert");
    assert_eq!(fx.store.projects_by_name("beta").expect("by name").len(), 2);
    assert_code(
        fx.store.rename_project("beta", "delta"),
        ErrorCode::InvalidInput,
    );
    assert_eq!(fx.store.projects_by_name("beta").expect("by name").len(), 2);
}

#[test]
fn s4_delete_project_cascades() {
    let fx = fixture();
    let a = project(&fx.store, "alpha");
    let b = project(&fx.store, "beta");
    let mut m = note_in_parts(a, &["alpha first part", "alpha second part"]);
    m.memory.anchors = vec![anchor("file:src/a.rs")];
    let m = fx.store.insert_memory(m, Access::Local).expect("insert").id;
    let n = add(&fx.store, a, "another alpha memory");
    // Link ends are stored in `links`; `anchors` rows come from attachments.
    let mut kept = note(b, "a beta memory");
    kept.memory.anchors = vec![anchor("table:orders")];
    let kept = fx
        .store
        .insert_memory(kept, Access::Local)
        .expect("insert")
        .id;
    for (p, l) in [
        (a, (LinkEnd::Memory(m), LinkEnd::Memory(n))),
        (
            a,
            (
                LinkEnd::Anchor(anchor("service:billing")),
                LinkEnd::Anchor(anchor("table:orders")),
            ),
        ),
        (
            b,
            (
                LinkEnd::Memory(kept),
                LinkEnd::Anchor(anchor("table:orders")),
            ),
        ),
    ] {
        fx.store
            .add_link(
                p,
                Link {
                    src: l.0,
                    dst: l.1,
                    relation: Relation::RelatesTo,
                    source: LinkSource::Manual,
                    extractor: None,
                    verified_commit: None,
                    verified_at: None,
                },
            )
            .expect("link");
    }
    for (p, ids, hash) in [(a, vec![m, n], "run-a"), (b, vec![kept], "run-b")] {
        fx.store
            .put_map_run(NewMapRun {
                project: Some(p),
                input_hash: hash.to_owned(),
                params: json!({}),
                clustered: false,
                clusters: vec![MapCluster {
                    region: 0,
                    size: u32::try_from(ids.len()).expect("size"),
                    label: None,
                }],
                points: ids
                    .iter()
                    .map(|&id| MapPointRow {
                        memory_id: id,
                        region: Some(0),
                        x: 0.5,
                        y: 0.5,
                        central_rank: None,
                    })
                    .collect(),
            })
            .expect("map run");
    }
    fx.store
        .record_injections("session-1", &[m, n, kept], Timestamp::now())
        .expect("injections");

    let reply = fx.store.delete_project("alpha").expect("delete");
    assert_eq!(
        reply,
        ProjectReply {
            name: "alpha".to_owned(),
            renamed_from: None,
            deleted: true,
            memories: 2,
        }
    );
    assert_code(fx.store.project(a), ErrorCode::NotFound);
    assert_code(fx.store.get_memory(&m), ErrorCode::NotFound);
    for (table, rows) in [
        ("projects", 1),
        ("memories", 1),
        ("provenance", 1),
        ("chunks", 1),
        ("chunks_fts", 1),
        ("chunk_vectors", 1),
        ("anchors", 1),
        ("memory_anchors", 1),
        ("links", 1),
        ("map_runs", 1),
        ("map_clusters", 1),
        ("map_points", 1),
        ("injections", 1),
    ] {
        assert_eq!(table_count(&fx.store, table), rows, "{table}");
    }
    assert_eq!(
        rows_where(&fx.store, "chunks", "memory_id", &kept.to_string()),
        1
    );
    assert_eq!(fx.store.get_memory(&kept).expect("get").project_id, b);

    assert_code(fx.store.delete_project("alpha"), ErrorCode::NotFound);
    fx.store.upsert_project(&elsewhere("beta")).expect("upsert");
    assert_code(fx.store.delete_project("beta"), ErrorCode::InvalidInput);
    assert_eq!(fx.store.memory_count().expect("count"), 1);
}

#[test]
fn s4_project_status_counts_types_statuses_and_shared() {
    let fx = fixture();
    let beta = project(&fx.store, "beta");
    let alpha = project(&fx.store, "alpha");
    let alpha2 = fx
        .store
        .upsert_project(&elsewhere("alpha"))
        .expect("upsert")
        .id;
    let mut decision = note(alpha, "a decision");
    decision.memory.memory_type = MemoryType::Decision;
    fx.store
        .insert_memory(decision, Access::Local)
        .expect("insert");
    let mut retracted = note(alpha, "a retracted note");
    retracted.memory.status = Status::Retracted;
    fx.store
        .insert_memory(retracted, Access::Local)
        .expect("insert");
    let mut shared = note(alpha, "a shared note");
    shared.memory.origin = sm_core::types::Origin::Shared;
    fx.store
        .insert_memory(shared, Access::Sync)
        .expect("insert");

    let s = fx.store.project_status(alpha).expect("status");
    assert_eq!(s.name, "alpha");
    assert_eq!(s.memories, 3);
    assert_eq!(s.shared, 1);
    let by_type: Vec<(&str, u64)> = s.by_type.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(by_type, vec![("decision", 1), ("note", 2)]);
    let by_status: Vec<(&str, u64)> = s.by_status.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(by_status, vec![("active", 2), ("retracted", 1)]);

    let empty = fx.store.project_status(beta).expect("status");
    assert_eq!((empty.memories, empty.shared), (0, 0));
    assert!(empty.by_type.is_empty() && empty.by_status.is_empty());
    assert_code(fx.store.project_status(ProjectId(999)), ErrorCode::NotFound);

    // Listed by name, then id.
    let order: Vec<ProjectId> = fx
        .store
        .projects()
        .expect("projects")
        .iter()
        .map(|p| p.id)
        .collect();
    assert_eq!(order, vec![alpha, alpha2, beta]);
    let statuses: Vec<(String, u64)> = fx
        .store
        .project_statuses()
        .expect("statuses")
        .into_iter()
        .map(|s| (s.name, s.memories))
        .collect();
    assert_eq!(
        statuses,
        vec![
            ("alpha".to_owned(), 3),
            ("alpha".to_owned(), 0),
            ("beta".to_owned(), 0)
        ]
    );
}
