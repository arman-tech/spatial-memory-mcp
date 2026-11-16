//! SPEC section 4, `map_runs`: the map cache round trip, region labels per
//! run (B-32), validation, pruning, and points removed with their memory.

use serde_json::json;
use sm_core::ids::RunId;
use sm_core::protocol::MAX_LABEL_CHARS;

use super::*;
use crate::{MapCluster, MapPointRow, NewMapRun};

fn cluster(region: u32, size: u32) -> MapCluster {
    MapCluster {
        region,
        size,
        label: None,
    }
}

fn point(memory_id: MemoryId, region: Option<u32>, x: f32, y: f32) -> MapPointRow {
    MapPointRow {
        memory_id,
        region,
        x,
        y,
        central_rank: None,
    }
}

fn run(project: Option<ProjectId>, hash: &str, ids: &[MemoryId]) -> NewMapRun {
    NewMapRun {
        project,
        input_hash: hash.to_owned(),
        params: json!({}),
        clustered: false,
        clusters: Vec::new(),
        points: ids.iter().map(|&id| point(id, None, 0.0, 0.0)).collect(),
    }
}

#[test]
fn s4_map_cache_round_trip() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids: Vec<MemoryId> = (0..4)
        .map(|i| add(&fx.store, p, &format!("mapped memory {i}")))
        .collect();
    let mut points = vec![
        MapPointRow {
            central_rank: Some(0),
            ..point(ids[0], Some(0), 0.125, -2.5)
        },
        MapPointRow {
            central_rank: Some(1),
            ..point(ids[1], Some(0), 1.0e-7, 3.402_823_5e38)
        },
        point(ids[2], Some(1), -0.333_333_34, 7.0),
        point(ids[3], None, f32::MIN_POSITIVE, -0.0),
    ];
    let params = json!({"n_neighbors": 15, "min_cluster_size": 5, "seed": 42});
    let new = NewMapRun {
        project: Some(p),
        input_hash: "hash-1".to_owned(),
        params: params.clone(),
        clustered: true,
        clusters: vec![cluster(1, 1), cluster(0, 2)],
        points: points.clone(),
    };
    let (stored, created) = fx.store.put_map_run(new.clone()).expect("store");
    assert!(created);
    assert_eq!(stored.project, Some(p));
    assert_eq!(stored.input_hash, "hash-1");
    assert_eq!(stored.params, params);
    assert_eq!(stored.memories, 4);
    assert!(stored.clustered);
    assert_eq!(stored.clusters, vec![cluster(0, 2), cluster(1, 1)]);
    points.sort_by_key(|pt| pt.memory_id);
    assert_eq!(stored.points, points, "positions round-trip exactly");

    assert_eq!(fx.store.map_run(stored.id).expect("run"), stored);
    assert_eq!(
        fx.store.map_run_by_hash("hash-1").expect("by hash"),
        Some(stored.clone())
    );
    assert_eq!(fx.store.map_run_by_hash("hash-2").expect("by hash"), None);
    assert_eq!(
        fx.store.latest_map_run(Some(p)).expect("latest"),
        Some(stored.clone())
    );
    assert_eq!(fx.store.latest_map_run(None).expect("latest"), None);
    assert_code(fx.store.map_run(RunId(999)), ErrorCode::NotFound);

    // The same input hash is a cache hit: nothing is written.
    let (again, created) = fx
        .store
        .put_map_run(NewMapRun {
            clustered: false,
            clusters: Vec::new(),
            points: Vec::new(),
            ..new.clone()
        })
        .expect("store again");
    assert!(!created);
    assert_eq!(again, stored);
    assert_eq!(table_count(&fx.store, "map_runs"), 1);

    // Labels belong to one run's region (B-32).
    fx.store
        .set_region_label(stored.id, 0, Some("Billing and payments"))
        .expect("label");
    let labelled = fx.store.map_run(stored.id).expect("run");
    assert_eq!(
        labelled.clusters[0].label.as_deref(),
        Some("Billing and payments")
    );
    assert_eq!(labelled.clusters[1].label, None);
    let longest = "é".repeat(MAX_LABEL_CHARS);
    fx.store
        .set_region_label(stored.id, 1, Some(&longest))
        .expect("label of the longest length");
    let too_long = "é".repeat(MAX_LABEL_CHARS + 1);
    for bad in ["", "   ", "two\nlines", "two\rlines", too_long.as_str()] {
        assert_code(
            fx.store.set_region_label(stored.id, 0, Some(bad)),
            ErrorCode::InvalidInput,
        );
    }
    assert_code(
        fx.store.set_region_label(stored.id, 9, Some("nowhere")),
        ErrorCode::NotFound,
    );
    assert_code(
        fx.store.set_region_label(RunId(999), 0, Some("nowhere")),
        ErrorCode::NotFound,
    );
    fx.store
        .set_region_label(stored.id, 0, None)
        .expect("clear the label");
    let cleared = fx.store.map_run(stored.id).expect("run");
    assert_eq!(cleared.clusters[0].label, None);
    assert_eq!(cleared.clusters[1].label.as_deref(), Some(longest.as_str()));

    // A new run starts without labels.
    let (second, created) = fx
        .store
        .put_map_run(NewMapRun {
            input_hash: "hash-2".to_owned(),
            ..new
        })
        .expect("second run");
    assert!(created);
    assert!(second.clusters.iter().all(|c| c.label.is_none()));
    assert_eq!(
        fx.store
            .latest_map_run(Some(p))
            .expect("latest")
            .map(|r| r.id),
        Some(second.id)
    );

    // Forgetting a memory removes its points; the run keeps its count.
    fx.store.forget(&[ids[1]], Access::Local).expect("forget");
    let after = fx.store.map_run(stored.id).expect("run");
    assert_eq!(after.memories, 4);
    assert_eq!(after.points.len(), 3);
    assert!(after.points.iter().all(|pt| pt.memory_id != ids[1]));
}

#[test]
fn s4_map_runs_are_validated() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let a = add(&fx.store, p, "in alpha");
    let b = add(&fx.store, q, "in beta");

    let labelled = NewMapRun {
        clusters: vec![MapCluster {
            label: Some("preset".to_owned()),
            ..cluster(0, 1)
        }],
        ..run(Some(p), "labelled", &[a])
    };
    let twice = NewMapRun {
        clusters: vec![cluster(0, 1), cluster(0, 1)],
        ..run(Some(p), "region twice", &[a])
    };
    let unknown_region = NewMapRun {
        clusters: vec![cluster(0, 1)],
        points: vec![point(a, Some(3), 0.0, 0.0)],
        ..run(Some(p), "unknown region", &[])
    };
    let nan = NewMapRun {
        points: vec![point(a, None, f32::NAN, 0.0)],
        ..run(Some(p), "nan", &[])
    };
    let infinite = NewMapRun {
        points: vec![point(a, None, 0.0, f32::INFINITY)],
        ..run(Some(p), "infinite", &[])
    };
    for (what, bad, code) in [
        (
            "blank hash",
            run(Some(p), "  ", &[a]),
            ErrorCode::InvalidInput,
        ),
        ("preset label", labelled, ErrorCode::InvalidInput),
        ("region twice", twice, ErrorCode::InvalidInput),
        (
            "point twice",
            run(Some(p), "twice", &[a, a]),
            ErrorCode::InvalidInput,
        ),
        ("unknown region", unknown_region, ErrorCode::InvalidInput),
        ("NaN", nan, ErrorCode::InvalidInput),
        ("infinite", infinite, ErrorCode::InvalidInput),
        (
            "other project",
            run(Some(p), "other project", &[a, b]),
            ErrorCode::InvalidInput,
        ),
        (
            "missing memory",
            run(Some(p), "missing", &[a, MemoryId::new_v7()]),
            ErrorCode::NotFound,
        ),
        (
            "missing project",
            run(Some(ProjectId(999)), "no project", &[]),
            ErrorCode::NotFound,
        ),
    ] {
        match fx.store.put_map_run(bad) {
            Ok(r) => panic!("{what}: accepted {r:?}"),
            Err(e) => assert_eq!(e.code, code, "{what}: {e:?}"),
        }
    }
    assert_eq!(table_count(&fx.store, "map_runs"), 0);
    assert_eq!(table_count(&fx.store, "map_points"), 0);

    // A run over every project may hold any project's memories.
    let (all, created) = fx
        .store
        .put_map_run(run(None, "all projects", &[a, b]))
        .expect("all-projects run");
    assert!(created);
    assert_eq!((all.project, all.points.len()), (None, 2));
    assert_eq!(
        fx.store.latest_map_run(None).expect("latest").map(|r| r.id),
        Some(all.id)
    );
    // An empty run is a valid cache entry.
    let (empty, _) = fx
        .store
        .put_map_run(run(Some(q), "empty", &[]))
        .expect("empty run");
    assert_eq!((empty.memories, empty.points.len()), (0, 0));
}

#[test]
fn s4_prune_keeps_the_newest_runs() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let a = add(&fx.store, p, "in alpha");
    let mut p_runs = Vec::new();
    for i in 0..5 {
        let (r, _) = fx
            .store
            .put_map_run(run(Some(p), &format!("p-{i}"), &[a]))
            .expect("run");
        p_runs.push(r.id);
    }
    for i in 0..2 {
        fx.store
            .put_map_run(run(None, &format!("all-{i}"), &[a]))
            .expect("run");
    }
    fx.store.put_map_run(run(Some(q), "q-0", &[])).expect("run");

    assert_eq!(fx.store.prune_map_runs(Some(p), 2).expect("prune"), 3);
    for (i, id) in p_runs.iter().enumerate() {
        let kept = fx.store.map_run(*id).is_ok();
        assert_eq!(kept, i >= 3, "run {i}");
    }
    assert_eq!(
        fx.store
            .latest_map_run(Some(p))
            .expect("latest")
            .map(|r| r.id),
        Some(p_runs[4])
    );
    assert_eq!(table_count(&fx.store, "map_runs"), 5);
    assert_eq!(fx.store.prune_map_runs(Some(p), 2).expect("prune"), 0);
    assert_eq!(fx.store.prune_map_runs(None, 0).expect("prune"), 2);
    assert_eq!(fx.store.latest_map_run(None).expect("latest"), None);
    assert!(fx.store.latest_map_run(Some(q)).expect("latest").is_some());
    assert_eq!(table_count(&fx.store, "map_runs"), 3);
    // Points of pruned runs go with them.
    assert_eq!(table_count(&fx.store, "map_points"), 2);
}
