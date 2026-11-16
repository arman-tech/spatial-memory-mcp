//! SPEC section 7 rules: the cluster-size formula, no clustering under 30
//! memories, the 2,000 cap with deterministic sampling and nearest-centre
//! assignment, the cache key, map coordinates and `journey`. Vectors come
//! from the deterministic fake embedder; no test loads the model.

mod common;

use std::collections::BTreeSet;
use std::sync::Mutex;

use common::{fake_id, fake_memories};
use sm_core::ErrorCode;
use sm_core::Result;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{MAX_JOURNEY_STEPS, MIN_JOURNEY_STEPS, MapPoint};
use sm_core::traits::{
    ClusterParams, MemoryFilter, MemoryHit, ProjectParams, QueryScope, VectorIndex, VectorQuery,
};
use sm_spatial::{
    Endpoint, JourneyRequest, MapParams, MemoryVectors, coordinates, cosine, input_hash, journey,
    memory_vector, regions, sample_order,
};

fn sorted_ids(memories: &[MemoryVectors]) -> Vec<MemoryId> {
    let mut ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();
    ids.sort_unstable();
    ids
}

// ---------------------------------------------------------------------------
// Cluster sizes and the 30-memory floor
// ---------------------------------------------------------------------------

#[test]
fn s7_min_cluster_size_formula_table() {
    for (n, min) in [(500, 11), (1000, 16), (5000, 35)] {
        assert_eq!(ClusterParams::for_n(n).min_cluster_size, min, "n = {n}");
    }
    for (n, min) in [(500, 11), (1000, 16)] {
        let run = regions(&fake_memories(n, 10), &MapParams::default()).unwrap();
        let params = run.cluster_params.unwrap();
        assert_eq!(params, ClusterParams::for_n(n));
        assert_eq!(params.min_cluster_size, min);
        assert_eq!(params.max_cluster_size, n / 4);
        assert!(run.regions.iter().all(|r| r.size() >= min), "n = {n}");
    }
}

#[test]
fn s7_no_clustering_under_30_memories() {
    let run = regions(&fake_memories(29, 3), &MapParams::default()).unwrap();
    assert!(!run.clustered);
    assert_eq!(run.cluster_params, None);
    assert_eq!(run.regions.len(), 1);
    assert_eq!(run.regions[0].size(), 29);
    assert!(run.unclustered.is_empty());

    let run = regions(&fake_memories(30, 3), &MapParams::default()).unwrap();
    assert!(run.clustered);
    assert_eq!(run.cluster_params, Some(ClusterParams::for_n(30)));
    assert_eq!(
        run.regions.iter().map(|r| r.size()).sum::<usize>() + run.unclustered.len(),
        30
    );
}

#[test]
fn s7_regions_are_numbered_largest_first_with_ranked_members() {
    let run = regions(&fake_memories(240, 6), &MapParams::default()).unwrap();
    assert!(run.regions.len() >= 2);
    for (i, r) in run.regions.iter().enumerate() {
        assert_eq!(r.region as usize, i);
        for w in r.members.windows(2) {
            assert!(w[0].centrality >= w[1].centrality);
        }
        let norm: f32 = r.centre.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    for w in run.regions.windows(2) {
        assert!(w[0].size() >= w[1].size());
    }
    let all: BTreeSet<MemoryId> = run.assignments().iter().map(|a| a.0).collect();
    assert_eq!(all.len(), 240, "every memory appears exactly once");
}

// ---------------------------------------------------------------------------
// The 2,000 cap: deterministic sample, nearest-centre assignment
// ---------------------------------------------------------------------------

#[test]
fn s7_cap_samples_2000_and_assigns_the_rest_to_the_nearest_centre() {
    let memories = fake_memories(3000, 12);
    let params = MapParams::default();
    let run = regions(&memories, &params).unwrap();
    assert!(run.clustered && run.sampled);
    assert_eq!(run.memories, 3000);
    assert_eq!(run.projected, 2000);
    assert_eq!(run.cluster_params, Some(ClusterParams::for_n(2000)));
    assert!(run.regions.len() >= 2);

    let ids = sorted_ids(&memories);
    let order = sample_order(&ids, params.project.seed);
    let sample: BTreeSet<MemoryId> = order[..2000].iter().map(|&i| ids[i]).collect();

    // Only sample memories can be noise; every other memory is assigned.
    assert!(run.unclustered.iter().all(|id| sample.contains(id)));
    let assignments = run.assignments();
    assert_eq!(assignments.len(), 3000);

    // Recompute each region's centre from its sample members alone, and
    // check every non-sample memory sits in the region whose centre is
    // nearest in the original dimensions.
    let vector_of = |id: MemoryId| -> Vec<f32> {
        let m = memories.iter().find(|m| m.id == id).unwrap();
        memory_vector(&m.chunks).unwrap()
    };
    let centres: Vec<Vec<f32>> = run
        .regions
        .iter()
        .map(|r| {
            let mut sum = vec![0.0_f64; sm_core::EMBED_DIM];
            for m in r.members.iter().filter(|m| sample.contains(&m.id)) {
                for (s, x) in sum.iter_mut().zip(vector_of(m.id)) {
                    *s += f64::from(x);
                }
            }
            let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
            sum.iter().map(|x| (x / norm) as f32).collect()
        })
        .collect();
    let mut checked = 0;
    for (id, region) in assignments.iter().filter(|a| !sample.contains(&a.0)) {
        let region = region.expect("non-sample memories are always assigned") as usize;
        let v = vector_of(*id);
        let mine = cosine(&v, &centres[region]);
        for (other, c) in centres.iter().enumerate() {
            assert!(
                mine >= cosine(&v, c) - 1e-6,
                "{id}: region {region} at {mine}, region {other} nearer"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 1000);
}

#[test]
fn s7_sample_changes_by_at_most_one_member_when_a_memory_is_added() {
    let ids: Vec<MemoryId> = (0..3000).map(fake_id).collect();
    let take = |ids: &[MemoryId]| -> BTreeSet<MemoryId> {
        sample_order(ids, 42)[..2000]
            .iter()
            .map(|&i| ids[i])
            .collect()
    };
    let before = take(&ids);
    let mut more = ids.clone();
    more.push(fake_id(3000));
    let after = take(&more);
    assert!(before.difference(&after).count() <= 1);
    // Input order does not matter.
    let mut shuffled = ids.clone();
    shuffled.reverse();
    assert_eq!(take(&shuffled), before);
}

#[test]
fn s7_projector_input_is_capped_at_2000() {
    let params = MapParams {
        max_points: 2001,
        ..MapParams::default()
    };
    let err = regions(&fake_memories(40, 2), &params).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidInput);
}

// ---------------------------------------------------------------------------
// The cache key
// ---------------------------------------------------------------------------

#[test]
fn s7_input_hash_changes_with_any_id_or_parameter() {
    let memories = fake_memories(40, 2);
    let ids = sorted_ids(&memories);
    let base = MapParams::default();
    let h = input_hash(&ids, &base);
    assert_eq!(regions(&memories, &base).unwrap().input_hash, h);

    let mut changed_id = ids.clone();
    changed_id[7] = fake_id(9999);
    assert_ne!(input_hash(&changed_id, &base), h, "an id changed");
    assert_ne!(input_hash(&ids[1..], &base), h, "an id removed");
    let mut added = ids.clone();
    added.push(fake_id(9999));
    assert_ne!(input_hash(&added, &base), h, "an id added");

    let p = base.project;
    let c = base.coords;
    let variants = [
        MapParams {
            project: ProjectParams { dims: 4, ..p },
            ..base
        },
        MapParams {
            project: ProjectParams { neighbors: 14, ..p },
            ..base
        },
        MapParams {
            project: ProjectParams { min_dist: 0.1, ..p },
            ..base
        },
        MapParams {
            project: ProjectParams { seed: 43, ..p },
            ..base
        },
        MapParams {
            coords: ProjectParams { dims: 3, ..c },
            ..base
        },
        MapParams {
            coords: ProjectParams { neighbors: 10, ..c },
            ..base
        },
        MapParams {
            coords: ProjectParams { min_dist: 0.1, ..c },
            ..base
        },
        MapParams {
            coords: ProjectParams { seed: 7, ..c },
            ..base
        },
        MapParams {
            max_points: 1999,
            ..base
        },
        MapParams {
            min_memories: 31,
            ..base
        },
    ];
    let mut seen = BTreeSet::from([h.clone()]);
    for (i, v) in variants.iter().enumerate() {
        assert!(
            seen.insert(input_hash(&ids, v)),
            "parameter variant {i} did not change the hash"
        );
    }
    // -0.0 and 0.0 are different bits, so different keys.
    let negative_zero = MapParams {
        project: ProjectParams {
            min_dist: -0.0,
            ..p
        },
        ..base
    };
    assert_ne!(input_hash(&ids, &negative_zero), h);
}

// ---------------------------------------------------------------------------
// Map coordinates
// ---------------------------------------------------------------------------

/// `regions`, then `coordinates` with its run.
fn map_points(memories: &[MemoryVectors], params: &MapParams) -> Vec<MapPoint> {
    let run = regions(memories, params).unwrap();
    coordinates(memories, params, &run).unwrap()
}

#[test]
fn s7_coordinates_cover_every_memory_in_id_order() {
    let memories = fake_memories(60, 3);
    let params = MapParams::default();
    let run = regions(&memories, &params).unwrap();
    let coords = coordinates(&memories, &params, &run).unwrap();
    assert_eq!(coords.len(), 60);
    assert_eq!(
        coords.iter().map(|c| c.id).collect::<Vec<_>>(),
        sorted_ids(&memories)
    );
    // Each point carries its memory's region from the run.
    assert_eq!(
        coords.iter().map(|c| (c.id, c.region)).collect::<Vec<_>>(),
        run.assignments()
    );
    assert!(coords.iter().all(|c| c.x.is_finite() && c.y.is_finite()));
    let distinct: BTreeSet<(u32, u32)> = coords
        .iter()
        .map(|c| (c.x.to_bits(), c.y.to_bits()))
        .collect();
    assert!(distinct.len() > 1);
}

#[test]
fn s7_coordinates_for_two_and_three_memories() {
    let two = fake_memories(2, 2);
    let coords = map_points(&two, &MapParams::default());
    let ids = sorted_ids(&two);
    let v: Vec<Vec<f32>> = ids
        .iter()
        .map(|id| memory_vector(&two.iter().find(|m| m.id == *id).unwrap().chunks).unwrap())
        .collect();
    assert_eq!((coords[0].x, coords[0].y), (0.0, 0.0));
    assert_eq!(
        (coords[1].x, coords[1].y),
        (1.0 - cosine(&v[0], &v[1]), 0.0)
    );
    let three = map_points(&fake_memories(3, 3), &MapParams::default());
    assert_eq!(three.len(), 3);
    assert!(three.iter().all(|c| c.x.is_finite() && c.y.is_finite()));
}

#[test]
fn s7_sampled_coordinates_place_the_rest_among_the_sample() {
    let memories = fake_memories(300, 4);
    let params = MapParams {
        max_points: 100,
        ..MapParams::default()
    };
    let coords = map_points(&memories, &params);
    assert_eq!(coords.len(), 300);

    let ids = sorted_ids(&memories);
    let sample: BTreeSet<MemoryId> = sample_order(&ids, 42)[..100]
        .iter()
        .map(|&i| ids[i])
        .collect();
    let sample_memories: Vec<MemoryVectors> = memories
        .iter()
        .filter(|m| sample.contains(&m.id))
        .cloned()
        .collect();
    // The sample's positions are exactly the projection of the sample alone.
    // (Region numbers can differ: the full run counts the assigned rest
    // when it numbers regions by size.)
    let alone = map_points(&sample_memories, &params);
    let position = |c: &MapPoint| (c.id, c.x.to_bits(), c.y.to_bits());
    let in_run: Vec<_> = coords
        .iter()
        .filter(|c| sample.contains(&c.id))
        .map(position)
        .collect();
    assert_eq!(in_run, alone.iter().map(position).collect::<Vec<_>>());
    // Every other memory sits at a mean of sample positions: inside their
    // bounding box.
    let (min_x, max_x) = alone
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), c| (a.min(c.x), b.max(c.x)));
    let (min_y, max_y) = alone
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), c| (a.min(c.y), b.max(c.y)));
    for c in coords.iter().filter(|c| !sample.contains(&c.id)) {
        assert!(
            (min_x..=max_x).contains(&c.x) && (min_y..=max_y).contains(&c.y),
            "{c:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// journey
// ---------------------------------------------------------------------------

/// Brute-force exact search over memory vectors, recording every query.
struct BruteIndex {
    memories: Vec<(MemoryId, Vec<f32>)>,
    queries: Mutex<Vec<(Vec<MemoryId>, QueryScope, usize)>>,
}

impl BruteIndex {
    fn new(memories: &[MemoryVectors]) -> Self {
        BruteIndex {
            memories: memories
                .iter()
                .map(|m| (m.id, memory_vector(&m.chunks).unwrap()))
                .collect(),
            queries: Mutex::new(Vec::new()),
        }
    }

    fn vector(&self, id: MemoryId) -> &[f32] {
        &self.memories.iter().find(|m| m.0 == id).unwrap().1
    }
}

impl VectorIndex for BruteIndex {
    fn nearest_memories(&self, query: &VectorQuery<'_>) -> Result<Vec<MemoryHit>> {
        self.queries
            .lock()
            .unwrap()
            .push((query.filter.exclude.clone(), query.scope, query.limit));
        let mut hits: Vec<MemoryHit> = self
            .memories
            .iter()
            .filter(|m| !query.filter.exclude.contains(&m.0))
            .map(|m| MemoryHit {
                memory_id: m.0,
                score: cosine(query.vector, &m.1),
                chunk_seq: 0,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.memory_id.cmp(&b.memory_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }
}

fn request<'a>(
    index: &'a BruteIndex,
    from: MemoryId,
    to: MemoryId,
    steps: usize,
) -> JourneyRequest<'a> {
    JourneyRequest {
        from: Endpoint {
            id: from,
            vector: index.vector(from),
        },
        to: Endpoint {
            id: to,
            vector: index.vector(to),
        },
        steps,
        scope: QueryScope::Project(ProjectId(7)),
        filter: MemoryFilter::default(),
    }
}

#[test]
fn s7_journey_endpoints_are_its_two_memories() {
    let index = BruteIndex::new(&fake_memories(80, 4));
    let (from, to) = (fake_id(0), fake_id(1));
    let path = journey(&index, &request(&index, from, to, 10)).unwrap();
    assert_eq!(path.len(), 10);
    assert_eq!(path[0].memory, Some(from));
    assert_eq!(path[9].memory, Some(to));
    assert_eq!((path[0].t, path[9].t), (0.0, 1.0));
    assert!((path[0].score.unwrap() - 1.0).abs() < 1e-6);
    assert!((path[9].score.unwrap() - 1.0).abs() < 1e-6);
    for (i, s) in path.iter().enumerate() {
        assert_eq!(s.step as usize, i);
        assert!((s.t - i as f32 / 9.0).abs() < 1e-6);
    }
    assert!((cosine(&path[0].point, index.vector(from)) - 1.0).abs() < 1e-6);
    assert!((cosine(&path[9].point, index.vector(to)) - 1.0).abs() < 1e-6);
}

#[test]
fn s7_journey_steps_are_unit_length_and_evenly_spaced() {
    let index = BruteIndex::new(&fake_memories(80, 4));
    let path = journey(&index, &request(&index, fake_id(0), fake_id(1), 12)).unwrap();
    for s in &path {
        let norm = s
            .point
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "step {}: {norm}", s.step);
    }
    let angles: Vec<f64> = path
        .windows(2)
        .map(|w| {
            f64::from(cosine(&w[0].point, &w[1].point))
                .clamp(-1.0, 1.0)
                .acos()
        })
        .collect();
    for a in &angles {
        assert!((a - angles[0]).abs() < 1e-3, "{angles:?}");
    }
}

#[test]
fn s7_journey_step_count_is_as_requested() {
    let index = BruteIndex::new(&fake_memories(80, 4));
    let (from, to) = (fake_id(0), fake_id(1));
    for steps in [
        MIN_JOURNEY_STEPS as usize,
        3,
        10,
        MAX_JOURNEY_STEPS as usize,
    ] {
        let path = journey(&index, &request(&index, from, to, steps)).unwrap();
        assert_eq!(path.len(), steps);
        assert_eq!(path.last().unwrap().memory, Some(to));
    }
    for steps in [0, 1, MAX_JOURNEY_STEPS as usize + 1] {
        let err = journey(&index, &request(&index, from, to, steps)).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput, "steps = {steps}");
    }
}

#[test]
fn s7_journey_picks_the_nearest_memory_not_already_on_the_path() {
    let memories = fake_memories(80, 4);
    let index = BruteIndex::new(&memories);
    let (from, to) = (fake_id(0), fake_id(1));
    let path = journey(&index, &request(&index, from, to, 20)).unwrap();
    let chosen: Vec<MemoryId> = path.iter().filter_map(|s| s.memory).collect();
    let distinct: BTreeSet<MemoryId> = chosen.iter().copied().collect();
    assert_eq!(distinct.len(), chosen.len(), "no memory repeats");
    assert_eq!(chosen.len(), 20, "80 memories fill every step");

    let mut on_path = vec![from, to];
    for s in &path[1..19] {
        let id = s.memory.unwrap();
        let best = index
            .memories
            .iter()
            .filter(|m| !on_path.contains(&m.0))
            .map(|m| cosine(&s.point, &m.1))
            .fold(f32::MIN, f32::max);
        assert_eq!(s.score, Some(best), "step {}", s.step);
        assert_eq!(cosine(&s.point, index.vector(id)), best);
        on_path.push(id);
    }
}

#[test]
fn s7_journey_passes_scope_and_exclusions_and_leaves_empty_steps() {
    let memories = fake_memories(6, 2);
    let index = BruteIndex::new(&memories);
    let (from, to) = (fake_id(0), fake_id(1));
    let mut req = request(&index, from, to, 10);
    req.filter.exclude = vec![fake_id(2)];
    let path = journey(&index, &req).unwrap();
    let chosen: Vec<MemoryId> = path.iter().filter_map(|s| s.memory).collect();
    // Six memories: the two endpoints, one excluded, three for the steps.
    assert_eq!(chosen.len(), 5);
    assert!(!chosen.contains(&fake_id(2)));
    assert_eq!(path.iter().filter(|s| s.memory.is_none()).count(), 5);
    assert!(path.iter().all(|s| s.memory.is_some() == s.score.is_some()));

    let queries = index.queries.lock().unwrap();
    assert_eq!(queries.len(), 8, "one query per step between the endpoints");
    for (exclude, scope, limit) in queries.iter() {
        assert_eq!(*scope, req.scope);
        for id in [fake_id(2), from, to] {
            assert!(exclude.contains(&id));
        }
        assert!(*limit >= 1);
    }
}

#[test]
fn s7_journey_rejects_one_memory_twice_and_opposite_vectors() {
    let index = BruteIndex::new(&fake_memories(10, 2));
    let err = journey(&index, &request(&index, fake_id(0), fake_id(0), 10)).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidInput);

    let v = index.vector(fake_id(0)).to_vec();
    let opposite: Vec<f32> = v.iter().map(|x| -x).collect();
    let req = JourneyRequest {
        from: Endpoint {
            id: fake_id(0),
            vector: &v,
        },
        to: Endpoint {
            id: fake_id(1),
            vector: &opposite,
        },
        steps: 5,
        scope: QueryScope::All,
        filter: MemoryFilter::default(),
    };
    assert_eq!(
        journey(&index, &req).unwrap_err().code,
        ErrorCode::InvalidInput
    );
}
