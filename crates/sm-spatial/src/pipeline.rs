//! The per-project region pipeline (SPEC section 7).
//!
//! 1. One vector per memory: the re-normalised mean of its chunk vectors.
//!    Memories are put in identifier order first, so a run is a function of
//!    the set of memories, not of the order the caller passed them in.
//! 2. Fewer than [`MapParams::min_memories`] (30): no clustering; the
//!    project is one group, region 0.
//! 3. More than [`MapParams::max_points`] (2,000): a deterministic sample of
//!    2,000 (see [`sample_order`]) is projected and clustered; every other
//!    memory joins the cluster whose centre is nearest in the original
//!    dimensions (cosine). Sample memories the clusterer marked as noise
//!    stay unclustered.
//! 4. Projection to 5 dimensions ([`crate::HolomapProjector`]), then
//!    clustering ([`crate::HdbscanClusterer`]) with
//!    [`ClusterParams::for_n`] of the number of points clustered.
//! 5. Regions are numbered by size, largest first (ties: the region holding
//!    the smallest memory identifier first). The numbers are local to the
//!    run (SPEC section 4: cluster numbers are never keys).
//! 6. Each region's members are ranked by [`CENTRALITY`].

use std::cmp::Ordering;

use sm_core::hash::{fnv1a64, sha256_hex};
use sm_core::ids::MemoryId;
use sm_core::protocol::MapPoint;
use sm_core::traits::{
    ClusterParams, Clusterer, MAX_PROJECTION_POINTS, MIN_CLUSTER_MEMORIES, ProjectParams, Projector,
};
use sm_core::{CoreError, Result};

use crate::clusterer::HdbscanClusterer;
use crate::projector::{HolomapProjector, check_project_params};
use crate::vector::{centre, cosine, dot, memory_vector};

/// The algorithm and library versions, part of every [`input_hash`]. Bump
/// the leading version whenever the pipeline's output for the same input
/// can change, including a change of either library pin.
pub const ALGORITHM: &str = "sm-spatial regions v1; holomap 0.3.0; hdbscan 0.12.0";

/// How "central" is measured (SPEC section 7, Labels), for the reply and
/// the report.
pub const CENTRALITY: &str = "cosine similarity between the memory's vector and its region's \
centre, the re-normalised mean of all the region members' vectors in the original 768 \
dimensions; higher is more central, ties broken by memory identifier";

/// One memory's input: its identifier and the vectors of its chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryVectors {
    pub id: MemoryId,
    pub chunks: Vec<Vec<f32>>,
}

/// Everything a run depends on besides the memories. The default is SPEC
/// section 7.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapParams {
    /// The projection clustered (5 dimensions, 15 neighbours, minimum
    /// distance 0, seed 42). Its seed also seeds the sample.
    pub project: ProjectParams,
    /// The projection for map coordinates: the same, with 2 dimensions.
    pub coords: ProjectParams,
    /// Most memories projected in one run (at most
    /// [`MAX_PROJECTION_POINTS`]); above it the run is sampled.
    pub max_points: usize,
    /// Fewest memories clustered; below it the project is one group.
    pub min_memories: usize,
}

impl Default for MapParams {
    fn default() -> Self {
        let project = ProjectParams::default();
        MapParams {
            project,
            coords: ProjectParams { dims: 2, ..project },
            max_points: MAX_PROJECTION_POINTS,
            min_memories: MIN_CLUSTER_MEMORIES,
        }
    }
}

impl MapParams {
    /// Checks the parameters on their own and against each other: every
    /// clustered run must hold more memories than the projection's
    /// neighbours and at least the clusterer's `min_samples`.
    pub fn validate(&self) -> Result<()> {
        check_project_params(&self.project, "project")?;
        check_project_params(&self.coords, "coords")?;
        if self.coords.dims != 2 {
            return Err(CoreError::invalid_input("coords.dims must be 2"));
        }
        if self.max_points > MAX_PROJECTION_POINTS {
            return Err(CoreError::invalid_input(format!(
                "max_points must be at most {MAX_PROJECTION_POINTS}"
            )));
        }
        let min_samples = ClusterParams::for_n(self.min_memories).rust_min_samples();
        if self.min_memories <= self.project.neighbors || self.min_memories < min_samples {
            return Err(CoreError::invalid_input(format!(
                "min_memories ({}) must exceed project.neighbors ({}) and be at least {min_samples}",
                self.min_memories, self.project.neighbors
            )));
        }
        if self.max_points < self.min_memories {
            return Err(CoreError::invalid_input(format!(
                "max_points ({}) must be at least min_memories ({})",
                self.max_points, self.min_memories
            )));
        }
        Ok(())
    }
}

/// A region member and how central it is ([`CENTRALITY`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedMember {
    pub id: MemoryId,
    pub centrality: f32,
}

/// One cluster of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionResult {
    /// The cluster number, local to the run.
    pub region: u32,
    /// Every member, most central first.
    pub members: Vec<RankedMember>,
    /// The unit centre in the original dimensions (the version 3.1
    /// two-level map clusters these).
    pub centre: Vec<f32>,
}

impl RegionResult {
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// The `k` most central members.
    pub fn central(&self, k: usize) -> &[RankedMember] {
        &self.members[..k.min(self.members.len())]
    }
}

/// The result of [`regions`] for one project.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionRun {
    /// The cache key: [`input_hash`] of the memory identifiers and the
    /// parameters.
    pub input_hash: String,
    /// Memories in the run.
    pub memories: usize,
    /// `false` below [`MapParams::min_memories`]: one group, no clustering.
    pub clustered: bool,
    /// `true` when only a sample was projected and clustered.
    pub sampled: bool,
    /// Memories projected and clustered (the sample size when sampled, 0
    /// when not clustered).
    pub projected: usize,
    /// The clusterer's parameters, `for_n(projected)`, when clustered.
    pub cluster_params: Option<ClusterParams>,
    /// Largest first.
    pub regions: Vec<RegionResult>,
    /// Memories in no region, in identifier order.
    pub unclustered: Vec<MemoryId>,
}

impl RegionRun {
    /// Every memory and its region, in identifier order.
    pub fn assignments(&self) -> Vec<(MemoryId, Option<u32>)> {
        let mut out: Vec<(MemoryId, Option<u32>)> = self
            .regions
            .iter()
            .flat_map(|r| r.members.iter().map(move |m| (m.id, Some(r.region))))
            .chain(self.unclustered.iter().map(|&id| (id, None)))
            .collect();
        out.sort_by_key(|a| a.0);
        out
    }

    /// The share of memories in no region (0 for an empty run).
    pub fn unclustered_share(&self) -> f64 {
        share(self.unclustered.len(), self.memories)
    }

    /// The share of memories in the largest region (0 for an empty run).
    pub fn largest_share(&self) -> f64 {
        share(
            self.regions.first().map_or(0, RegionResult::size),
            self.memories,
        )
    }
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// The memories in identifier order with one unit vector each.
struct Prepared {
    ids: Vec<MemoryId>,
    vectors: Vec<Vec<f32>>,
    dims: usize,
}

fn prepare(memories: &[MemoryVectors]) -> Result<Prepared> {
    let mut order: Vec<usize> = (0..memories.len()).collect();
    order.sort_by_key(|&i| memories[i].id);
    let mut ids = Vec::with_capacity(memories.len());
    let mut vectors = Vec::with_capacity(memories.len());
    let mut dims = 0;
    for i in order {
        let m = &memories[i];
        if ids.last() == Some(&m.id) {
            return Err(CoreError::invalid_input(format!(
                "memory {} appears more than once",
                m.id
            )));
        }
        let v = memory_vector(&m.chunks).map_err(|mut e| {
            e.message = format!("memory {}: {}", m.id, e.message);
            e
        })?;
        if vectors.is_empty() {
            dims = v.len();
        } else if v.len() != dims {
            return Err(CoreError::invalid_input(format!(
                "memory {} has {} dimensions, the others {dims}",
                m.id,
                v.len()
            )));
        }
        ids.push(m.id);
        vectors.push(v);
    }
    Ok(Prepared { ids, vectors, dims })
}

/// The order in which memories enter a sample: ascending
/// `fnv1a64(seed as 8 little-endian bytes ++ the identifier's 16 bytes)`,
/// ties by identifier. Returns indexes into `ids`.
///
/// The key of a memory depends only on its identifier and the seed, so the
/// sample does not depend on input order, and adding or removing one memory
/// changes at most one sample member.
pub fn sample_order(ids: &[MemoryId], seed: u64) -> Vec<usize> {
    let keys: Vec<u64> = ids
        .iter()
        .map(|id| {
            let mut bytes = [0u8; 24];
            bytes[..8].copy_from_slice(&seed.to_le_bytes());
            bytes[8..].copy_from_slice(id.as_uuid().as_bytes());
            fnv1a64(&bytes)
        })
        .collect();
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(ids[a].cmp(&ids[b])));
    order
}

/// The rows (in identifier order) projected for a run of `n` memories.
fn sample_rows(ids: &[MemoryId], params: &MapParams) -> Vec<usize> {
    if ids.len() <= params.max_points {
        return (0..ids.len()).collect();
    }
    let mut rows = sample_order(ids, params.project.seed);
    rows.truncate(params.max_points);
    rows.sort_unstable();
    rows
}

/// The cache key of a run (SPEC section 7, step 4): SHA-256, in lower-case
/// hex, of [`ALGORITHM`], every parameter in `params` and the sorted memory
/// identifiers. Input order does not matter.
///
/// The key covers identifiers, not vectors: the caller must also discard a
/// cached run when a memory in it is edited or re-embedded.
pub fn input_hash(ids: &[MemoryId], params: &MapParams) -> String {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    let project = |name: &str, p: &ProjectParams| {
        format!(
            "{name}: dims={} neighbors={} min_dist={:08x} seed={} metric=cosine init=spectral \
             spread=1 epochs=default",
            p.dims,
            p.neighbors,
            p.min_dist.to_bits(),
            p.seed
        )
    };
    let mut lines = vec![
        ALGORITHM.to_owned(),
        "vector: mean of chunk vectors, re-normalised".to_owned(),
        project("project", &params.project),
        project("coords", &params.coords),
        format!(
            "max_points={} min_memories={}",
            params.max_points, params.min_memories
        ),
        "cluster: min_cluster_size=max(5,round(sqrt(n)/2)) min_samples=5+1 \
         max_cluster_size=max(n/4,min_cluster_size) metric=euclidean selection=eom \
         single_cluster=false epsilon=0"
            .to_owned(),
        "sample: fnv1a64(seed_le ++ uuid) ascending; assign: nearest centre by cosine; \
         coords: mean of nearest sample positions"
            .to_owned(),
        format!("n={}", sorted.len()),
    ];
    lines.extend(sorted.iter().map(ToString::to_string));
    sha256_hex(lines.join("\n").as_bytes())
}

/// Runs the pipeline for one project's memories (SPEC section 7).
///
/// Fails with `invalid_input` for invalid parameters, a duplicated
/// identifier, a memory with no chunk vectors (leave out memories whose
/// embedding is pending), or vectors of different lengths.
pub fn regions(memories: &[MemoryVectors], params: &MapParams) -> Result<RegionRun> {
    params.validate()?;
    let prep = prepare(memories)?;
    let n = prep.ids.len();
    let hash = input_hash(&prep.ids, params);

    if n < params.min_memories {
        let all: Vec<Option<usize>> = vec![Some(0); n];
        return Ok(RegionRun {
            input_hash: hash,
            memories: n,
            clustered: false,
            sampled: false,
            projected: 0,
            cluster_params: None,
            regions: build_regions(&prep, &all),
            unclustered: Vec::new(),
        });
    }

    let sample = sample_rows(&prep.ids, params);
    let m = sample.len();
    let sample_vectors: Vec<Vec<f32>> = sample.iter().map(|&r| prep.vectors[r].clone()).collect();
    let points = HolomapProjector.project(&sample_vectors, &params.project)?;
    let cluster_params = ClusterParams::for_n(m);
    let labels = HdbscanClusterer.cluster(&points, &cluster_params)?;

    let mut label: Vec<Option<usize>> = vec![None; n];
    for (&r, &l) in sample.iter().zip(&labels) {
        label[r] = l;
    }
    if m < n {
        assign_rest(&prep, &sample, &labels, &mut label);
    }

    let regions = build_regions(&prep, &label);
    let unclustered = (0..n)
        .filter(|&r| label[r].is_none())
        .map(|r| prep.ids[r])
        .collect();
    Ok(RegionRun {
        input_hash: hash,
        memories: n,
        clustered: true,
        sampled: m < n,
        projected: m,
        cluster_params: Some(cluster_params),
        regions,
        unclustered,
    })
}

/// Assigns every memory outside the sample to the cluster whose centre (the
/// re-normalised mean of its sample members) is nearest by cosine. Ties go
/// to the lower cluster number. With no clusters, nothing is assigned.
fn assign_rest(
    prep: &Prepared,
    sample: &[usize],
    labels: &[Option<usize>],
    label: &mut [Option<usize>],
) {
    let k = labels.iter().flatten().max().map_or(0, |&c| c + 1);
    if k == 0 {
        return;
    }
    let mut rows_by_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (&r, &l) in sample.iter().zip(labels) {
        if let Some(c) = l {
            rows_by_cluster[c].push(r);
        }
    }
    let centres: Vec<Vec<f32>> = rows_by_cluster
        .iter()
        .map(|rows| centre(&prep.vectors, rows, prep.dims))
        .collect();
    let mut in_sample = vec![false; prep.ids.len()];
    for &r in sample {
        in_sample[r] = true;
    }
    for r in (0..prep.ids.len()).filter(|&r| !in_sample[r]) {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (c, centre) in centres.iter().enumerate() {
            let s = cosine(&prep.vectors[r], centre);
            if s > best.0 {
                best = (s, c);
            }
        }
        label[r] = Some(best.1);
    }
}

/// Groups rows by label, numbers the groups by size (largest first, ties by
/// the smallest identifier) and ranks each group's members by centrality.
fn build_regions(prep: &Prepared, label: &[Option<usize>]) -> Vec<RegionResult> {
    let k = label.iter().flatten().max().map_or(0, |&c| c + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (r, l) in label.iter().enumerate() {
        if let Some(c) = l {
            groups[*c].push(r);
        }
    }
    groups.retain(|g| !g.is_empty());
    // Rows are in identifier order, so g[0] is the group's smallest id.
    groups.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    groups
        .iter()
        .enumerate()
        .map(|(i, rows)| {
            let centre = centre(&prep.vectors, rows, prep.dims);
            let mut members: Vec<RankedMember> = rows
                .iter()
                .map(|&r| RankedMember {
                    id: prep.ids[r],
                    centrality: cosine(&prep.vectors[r], &centre),
                })
                .collect();
            members.sort_by(|a, b| b.centrality.total_cmp(&a.centrality).then(a.id.cmp(&b.id)));
            RegionResult {
                region: u32::try_from(i).unwrap_or(u32::MAX),
                members,
                centre,
            }
        })
        .collect()
}

/// Every memory's position on the 2-dimension map, in identifier order, as
/// the `points` of a `regions` reply (`coords: true`): the same pipeline's
/// projection with [`MapParams::coords`] (2 dimensions), its neighbours
/// capped at one less than the number of memories projected. Each point
/// carries the memory's region from `run`, or `None` when unclustered.
///
/// One memory sits at (0, 0); two sit on the x axis, their cosine distance
/// apart. Above [`MapParams::max_points`], the same sample as [`regions`] is
/// projected and every other memory sits at the mean position of its
/// `coords.neighbors` nearest sample memories (cosine in the original
/// dimensions, ties by identifier).
///
/// Fails with `invalid_input`, besides the cases of [`regions`], when `run`
/// was not made by [`regions`] from the same memories and parameters.
pub fn coordinates(
    memories: &[MemoryVectors],
    params: &MapParams,
    run: &RegionRun,
) -> Result<Vec<MapPoint>> {
    params.validate()?;
    let prep = prepare(memories)?;
    let n = prep.ids.len();
    let assignments = run.assignments();
    if run.input_hash != input_hash(&prep.ids, params)
        || assignments.len() != n
        || assignments.iter().zip(&prep.ids).any(|(a, id)| a.0 != *id)
    {
        return Err(CoreError::invalid_input(
            "the region run was made from other memories or parameters",
        )
        .with_next_action("run `regions` on the same memories and parameters first"));
    }
    let sample = sample_rows(&prep.ids, params);
    let m = sample.len();
    let mut xy = vec![[0.0_f32; 2]; n];
    match m {
        0 | 1 => {}
        2 => {
            let d = 1.0 - cosine(&prep.vectors[sample[0]], &prep.vectors[sample[1]]);
            xy[sample[1]] = [d, 0.0];
        }
        _ => {
            let p = ProjectParams {
                neighbors: params.coords.neighbors.min(m - 1),
                ..params.coords
            };
            let vectors: Vec<Vec<f32>> = sample.iter().map(|&r| prep.vectors[r].clone()).collect();
            let out = HolomapProjector.project(&vectors, &p)?;
            for (&r, pos) in sample.iter().zip(&out) {
                xy[r] = [pos[0], pos[1]];
            }
        }
    }
    if m < n {
        let k = params.coords.neighbors.min(m);
        let mut in_sample = vec![false; n];
        for &r in &sample {
            in_sample[r] = true;
        }
        // Most similar first, ties by row (identifier order): a total order,
        // so the k nearest are one fixed set.
        let nearest = |a: &(f32, usize), b: &(f32, usize)| -> Ordering {
            b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
        };
        for r in (0..n).filter(|&r| !in_sample[r]) {
            // Memory vectors are unit length, so the dot product is the
            // cosine.
            let mut near: Vec<(f32, usize)> = sample
                .iter()
                .map(|&s| (dot(&prep.vectors[r], &prep.vectors[s]), s))
                .collect();
            near.select_nth_unstable_by(k - 1, nearest);
            near[..k].sort_by(nearest);
            let (mut x, mut y) = (0.0_f64, 0.0_f64);
            for &(_, s) in &near[..k] {
                x += f64::from(xy[s][0]);
                y += f64::from(xy[s][1]);
            }
            xy[r] = [(x / k as f64) as f32, (y / k as f64) as f32];
        }
    }
    Ok(assignments
        .into_iter()
        .zip(xy)
        .map(|((id, region), [x, y])| MapPoint { id, region, x, y })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::ErrorCode;
    use sm_core::testing::fake_vector;

    fn id(i: usize) -> MemoryId {
        MemoryId::from_v1(&format!("unit-{i}")).0
    }

    fn mem(i: usize, text: &str) -> MemoryVectors {
        MemoryVectors {
            id: id(i),
            chunks: vec![fake_vector(text)],
        }
    }

    #[test]
    fn s7_default_params_are_spec_section_7() {
        let p = MapParams::default();
        assert_eq!(p.project, ProjectParams::default());
        assert_eq!(
            (p.project.dims, p.project.neighbors, p.project.seed),
            (5, 15, 42)
        );
        assert_eq!(p.project.min_dist, 0.0);
        assert_eq!(
            p.coords,
            ProjectParams {
                dims: 2,
                ..ProjectParams::default()
            }
        );
        assert_eq!((p.max_points, p.min_memories), (2000, 30));
        p.validate().unwrap();
    }

    #[test]
    fn s7_map_params_validation() {
        let ok = MapParams::default();
        let bad = [
            MapParams {
                max_points: 2001,
                ..ok
            },
            MapParams {
                max_points: 29,
                ..ok
            },
            MapParams {
                min_memories: 15,
                ..ok
            },
            MapParams {
                min_memories: 5,
                project: ProjectParams {
                    neighbors: 2,
                    ..ok.project
                },
                ..ok
            },
            MapParams {
                coords: ProjectParams {
                    dims: 3,
                    ..ok.coords
                },
                ..ok
            },
            MapParams {
                project: ProjectParams {
                    dims: 0,
                    ..ok.project
                },
                ..ok
            },
        ];
        for (i, p) in bad.iter().enumerate() {
            assert_eq!(
                p.validate().unwrap_err().code,
                ErrorCode::InvalidInput,
                "case {i}"
            );
        }
    }

    #[test]
    fn s7_no_clustering_under_30_memories() {
        let mems: Vec<MemoryVectors> = (0..29)
            .map(|i| mem(i, &format!("alpha beta w{i}")))
            .collect();
        let run = regions(&mems, &MapParams::default()).unwrap();
        assert!(!run.clustered);
        assert!(!run.sampled);
        assert_eq!(run.cluster_params, None);
        assert_eq!(run.regions.len(), 1);
        assert_eq!(run.regions[0].region, 0);
        assert_eq!(run.regions[0].size(), 29);
        assert!(run.unclustered.is_empty());
        assert_eq!(run.assignments().len(), 29);
        assert!(run.assignments().iter().all(|a| a.1 == Some(0)));
    }

    #[test]
    fn s7_empty_and_single_memory_runs() {
        let empty = regions(&[], &MapParams::default()).unwrap();
        assert_eq!(empty.memories, 0);
        assert!(empty.regions.is_empty() && empty.unclustered.is_empty());
        assert_eq!(empty.unclustered_share(), 0.0);
        let one = regions(&[mem(1, "solo")], &MapParams::default()).unwrap();
        assert_eq!(one.regions.len(), 1);
        assert_eq!(one.regions[0].members[0].id, id(1));
        assert!((one.regions[0].members[0].centrality - 1.0).abs() < 1e-6);
        let p = MapParams::default();
        assert!(coordinates(&[], &p, &empty).unwrap().is_empty());
        let c = coordinates(&[mem(1, "solo")], &p, &one).unwrap();
        assert_eq!(
            c,
            vec![MapPoint {
                id: id(1),
                region: Some(0),
                x: 0.0,
                y: 0.0
            }]
        );
    }

    #[test]
    fn s7_coordinates_refuse_a_run_of_other_memories_or_parameters() {
        let p = MapParams::default();
        let ab = vec![mem(1, "a"), mem(2, "b")];
        let run = regions(&ab, &p).unwrap();
        assert_eq!(coordinates(&ab, &p, &run).unwrap().len(), 2);
        let other_memories = vec![mem(1, "a"), mem(3, "c")];
        let other_params = MapParams {
            min_memories: 31,
            ..p
        };
        let mut forged = run.clone();
        forged.regions[0].members.pop();
        for err in [
            coordinates(&other_memories, &p, &run).unwrap_err(),
            coordinates(&ab, &other_params, &run).unwrap_err(),
            coordinates(&ab, &p, &forged).unwrap_err(),
        ] {
            assert_eq!(err.code, ErrorCode::InvalidInput);
            assert!(err.message.contains("region run"), "{}", err.message);
        }
    }

    #[test]
    fn s7_rejects_duplicate_ids_and_unembedded_memories() {
        let dup = vec![mem(1, "a"), mem(2, "b"), mem(1, "c")];
        let err = regions(&dup, &MapParams::default()).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        assert!(err.message.contains("more than once"));
        let pending = vec![
            mem(1, "a"),
            MemoryVectors {
                id: id(2),
                chunks: vec![],
            },
        ];
        let err = regions(&pending, &MapParams::default()).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        assert!(err.message.contains(&id(2).to_string()));
        let mixed = vec![
            mem(1, "a"),
            MemoryVectors {
                id: id(2),
                chunks: vec![vec![1.0, 0.0]],
            },
        ];
        assert_eq!(
            regions(&mixed, &MapParams::default()).unwrap_err().code,
            ErrorCode::InvalidInput
        );
    }

    #[test]
    fn s7_central_members_are_ranked_by_cosine_to_the_centre() {
        // Five memories: four close to "core" words, one further out.
        let mems = vec![
            mem(1, "core shared words here"),
            mem(2, "core shared words there"),
            mem(3, "core shared words now"),
            mem(4, "core shared other things entirely different"),
            mem(5, "unrelated distant vocabulary only"),
        ];
        let run = regions(&mems, &MapParams::default()).unwrap();
        let r = &run.regions[0];
        for w in r.members.windows(2) {
            assert!(w[0].centrality >= w[1].centrality);
        }
        assert_eq!(r.members.last().unwrap().id, id(5));
        assert_eq!(r.central(2).len(), 2);
        assert_eq!(r.central(99).len(), 5);
        let norm: f32 = r.centre.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn s7_sample_order_is_seeded_and_order_independent() {
        let ids: Vec<MemoryId> = (0..50).map(id).collect();
        let a = sample_order(&ids, 42);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>(), "a permutation");
        let reversed: Vec<MemoryId> = ids.iter().rev().copied().collect();
        let b: Vec<MemoryId> = sample_order(&reversed, 42)
            .iter()
            .map(|&i| reversed[i])
            .collect();
        let a_ids: Vec<MemoryId> = a.iter().map(|&i| ids[i]).collect();
        assert_eq!(a_ids, b);
        assert_ne!(a, sample_order(&ids, 43));
    }

    #[test]
    fn s7_input_hash_is_order_independent_hex() {
        let ids: Vec<MemoryId> = (0..10).map(id).collect();
        let mut rev = ids.clone();
        rev.reverse();
        let p = MapParams::default();
        let h = input_hash(&ids, &p);
        assert_eq!(h.len(), 64);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_eq!(h, input_hash(&rev, &p));
    }
}
