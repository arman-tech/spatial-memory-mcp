//! `regions` and `journey` (SPEC 7, SPEC 8; B-32, B-46 to B-48).
//!
//! ## The map cache
//!
//! A run is computed by `sm-spatial` over the current project's active
//! memories that have embedded chunks, one re-normalized mean vector each,
//! and stored whole in `map_runs`. The store requires a run's key to cover
//! everything the result depends on; `sm_spatial::input_hash` covers the
//! identifiers and parameters but not the vectors (B-46). The daemon's key
//! is therefore the SHA-256 of that hash, the project, and every memory's
//! identifier, `updated_at` and vector bytes, so:
//!
//! - the same data gives the cached run, across restarts;
//! - any edit (it moves `updated_at`), forget (the identifiers change) or
//!   re-embedding (the vectors change) in the project gives a new key, so
//!   the old run is never served again and is pruned later;
//! - a per-project index in memory (`Shared::maps`) skips reading the
//!   vectors on a hit until the next write drops it.
//!
//! Coordinates are computed only when asked (B-47): they cost as much again
//! as the clustering. A run without them is stored under the key with the
//! suffix `:regions` and with every point at (0, 0), which is never
//! returned (the store needs finite positions; params record
//! `"coords": false`). A run with them is stored under `:coords`. A plain
//! request is served by either; a coordinates request needs `:coords`, and
//! computing it copies the labels of the same data's `:regions` run, whose
//! cluster numbers are the same because the pipeline is deterministic
//! (AT-6). A label given to one is also given to the other.
//!
//! Each region's ten most central memories are stored with their rank, so
//! any `central` from 1 to 10 is served from the cache.

use std::collections::{BTreeMap, HashMap};

use serde_json::json;
use sm_core::error::{CoreError, Result};
use sm_core::hash::sha256_hex;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{
    DEFAULT_CENTRAL, DEFAULT_JOURNEY_STEPS, JourneyParams, JourneyReply, JourneyStep, MAX_CENTRAL,
    MapPoint, Region, RegionsParams, RegionsReply,
};
use sm_core::traits::{MemoryFilter, QueryScope};
use sm_spatial::{ALGORITHM, Endpoint, JourneyRequest, MapParams, MemoryVectors, RegionRun};
use sm_store::{MapCluster, MapPointRow, MapRun, NewMapRun};

use crate::budget::{Fitted, fit_whole, tokens_of};
use crate::dispatch::Ctx;
use crate::state::{ProjectEntry, Shared, lock};
use crate::view::Names;

/// Runs kept per project after a new one is stored.
pub(crate) const KEEP_RUNS: usize = 4;

const PLAIN: &str = "regions";
const COORDS: &str = "coords";

/// The memories a run is computed from, and the key of their state.
struct Input {
    memories: Vec<MemoryVectors>,
    state: String,
}

fn input(sh: &Shared, pid: ProjectId, params: &MapParams) -> Result<Input> {
    let vectors = sh
        .store
        .memory_vectors(QueryScope::Project(pid), &MemoryFilter::default())?;
    let ids: Vec<MemoryId> = vectors.iter().map(|(id, _)| *id).collect();
    let updated: HashMap<MemoryId, i64> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m.updated_at.as_micros()))
        .collect();
    let dims = vectors.first().map_or(0, |(_, v)| v.len());
    let mut bytes = Vec::with_capacity(96 + vectors.len() * (24 + dims * 4));
    bytes.extend_from_slice(sm_spatial::input_hash(&ids, params).as_bytes());
    bytes.extend_from_slice(&pid.0.to_le_bytes());
    for (id, v) in &vectors {
        bytes.extend_from_slice(id.0.as_bytes());
        bytes.extend_from_slice(&updated.get(id).copied().unwrap_or(0).to_le_bytes());
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    let state = sha256_hex(&bytes);
    let memories = vectors
        .into_iter()
        .map(|(id, v)| MemoryVectors {
            id,
            chunks: vec![v],
        })
        .collect();
    Ok(Input { memories, state })
}

fn key(state: &str, variant: &str) -> String {
    format!("{state}:{variant}")
}

fn index_key(pid: ProjectId, variant: &str) -> String {
    format!("{pid}:{variant}")
}

/// The other variant's key for the same data.
fn sibling(input_hash: &str) -> Option<String> {
    let (state, variant) = input_hash.rsplit_once(':')?;
    match variant {
        PLAIN => Some(key(state, COORDS)),
        COORDS => Some(key(state, PLAIN)),
        _ => None,
    }
}

/// The run for this request from the in-memory index, if still stored.
fn indexed(sh: &Shared, pid: ProjectId, coords: bool) -> Result<Option<MapRun>> {
    let variants: &[&str] = if coords { &[COORDS] } else { &[COORDS, PLAIN] };
    for v in variants {
        let id = lock(&sh.maps).get(&index_key(pid, v)).copied();
        if let Some(id) = id {
            match sh.store.map_run(id) {
                Ok(run) => return Ok(Some(run)),
                Err(e) if e.code == sm_core::error::ErrorCode::NotFound => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(None)
}

/// The stored run for this data, if any.
fn stored(sh: &Shared, state: &str, coords: bool) -> Result<Option<MapRun>> {
    let variants: &[&str] = if coords { &[COORDS] } else { &[COORDS, PLAIN] };
    for v in variants {
        if let Some(run) = sh.store.map_run_by_hash(&key(state, v))? {
            return Ok(Some(run));
        }
    }
    Ok(None)
}

fn new_run(
    pid: ProjectId,
    input_hash: String,
    run: &RegionRun,
    points: Option<&[MapPoint]>,
    params: &MapParams,
) -> NewMapRun {
    let mut rank: HashMap<MemoryId, u32> = HashMap::new();
    for r in &run.regions {
        for (i, m) in r.central(MAX_CENTRAL as usize).iter().enumerate() {
            rank.insert(m.id, i as u32);
        }
    }
    let xy: HashMap<MemoryId, (f32, f32)> = points
        .map(|ps| ps.iter().map(|p| (p.id, (p.x, p.y))).collect())
        .unwrap_or_default();
    let rows = run
        .assignments()
        .into_iter()
        .map(|(id, region)| {
            let (x, y) = xy.get(&id).copied().unwrap_or((0.0, 0.0));
            MapPointRow {
                memory_id: id,
                region,
                x,
                y,
                central_rank: rank.get(&id).copied(),
            }
        })
        .collect();
    NewMapRun {
        project: Some(pid),
        input_hash,
        params: json!({
            "algorithm": ALGORITHM,
            "spatial_input_hash": run.input_hash,
            "coords": points.is_some(),
            "sampled": run.sampled,
            "projected": run.projected,
            "max_points": params.max_points,
            "min_memories": params.min_memories,
        }),
        clustered: run.clustered,
        clusters: run
            .regions
            .iter()
            .map(|r| MapCluster {
                region: r.region,
                size: r.size() as u32,
                label: None,
            })
            .collect(),
        points: rows,
    }
}

/// Copies the labels of `from` onto the same regions of `to`.
fn copy_labels(sh: &Shared, from: &MapRun, to: &MapRun) -> Result<()> {
    for c in &from.clusters {
        if let Some(label) = &c.label
            && to.clusters.iter().any(|t| t.region == c.region)
        {
            sh.store.set_region_label(to.id, c.region, Some(label))?;
        }
    }
    Ok(())
}

/// The cached run for the request, or a new one, and whether it was cached.
fn run_for(ctx: &Ctx<'_>, pid: ProjectId, coords: bool) -> Result<(MapRun, bool)> {
    let sh = ctx.sh;
    if let Some(run) = indexed(sh, pid, coords)? {
        return Ok((run, true));
    }
    let params = MapParams::default();
    let input = input(sh, pid, &params)?;
    if let Some(run) = stored(sh, &input.state, coords)? {
        let variant = if coords { COORDS } else { PLAIN };
        lock(&sh.maps).insert(index_key(pid, variant), run.id);
        return Ok((run, true));
    }
    let computed = sm_spatial::regions(&input.memories, &params)?;
    let points = if coords {
        Some(sm_spatial::coordinates(
            &input.memories,
            &params,
            &computed,
        )?)
    } else {
        None
    };
    let variant = if coords { COORDS } else { PLAIN };
    let new = new_run(
        pid,
        key(&input.state, variant),
        &computed,
        points.as_deref(),
        &params,
    );
    let (mut run, created) = sh.store.put_map_run(new)?;
    if created
        && coords
        && let Some(plain) = sh.store.map_run_by_hash(&key(&input.state, PLAIN))?
    {
        copy_labels(sh, &plain, &run)?;
        run = sh.store.map_run(run.id)?;
    }
    sh.store.prune_map_runs(Some(pid), KEEP_RUNS)?;
    lock(&sh.maps).insert(index_key(pid, variant), run.id);
    // Stored before the deadline check, so a late reply still leaves the
    // run cached for the next call.
    ctx.check_deadline("computing the regions")?;
    Ok((run, !created))
}

fn label(ctx: &Ctx<'_>, project: &ProjectEntry, l: &sm_core::protocol::RegionLabel) -> Result<()> {
    let sh = ctx.sh;
    let run = sh.store.map_run(l.run)?;
    if run.project != Some(project.project.id) {
        return Err(CoreError::not_found(format!(
            "map run {} is not a run of project {}",
            l.run, project.project.name
        )));
    }
    sh.store.set_region_label(l.run, l.region, Some(&l.label))?;
    if let Some(other) = sibling(&run.input_hash)
        && let Some(other) = sh.store.map_run_by_hash(&other)?
        && other.clusters.iter().any(|c| c.region == l.region)
    {
        sh.store
            .set_region_label(other.id, l.region, Some(&l.label))?;
    }
    Ok(())
}

pub(crate) fn regions(ctx: &Ctx<'_>, p: &RegionsParams) -> Result<RegionsReply> {
    let sh = ctx.sh;
    let project = ctx.project()?;
    let pid = project.project.id;
    let central = p.central.unwrap_or(DEFAULT_CENTRAL).clamp(1, MAX_CENTRAL);
    if let Some(l) = &p.label {
        label(ctx, &project, l)?;
    }
    let (run, cached) = run_for(ctx, pid, p.coords)?;

    let mut by_region: BTreeMap<u32, Vec<(u32, MemoryId)>> = BTreeMap::new();
    for pt in &run.points {
        if let (Some(r), Some(rank)) = (pt.region, pt.central_rank)
            && rank < central
        {
            by_region.entry(r).or_default().push((rank, pt.memory_id));
        }
    }
    let ids: Vec<MemoryId> = by_region
        .values()
        .flat_map(|v| v.iter().map(|(_, id)| *id))
        .collect();
    let mut memories: HashMap<MemoryId, _> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let names = Names::load(&sh.store)?;
    let regions: Vec<Region> = run
        .clusters
        .iter()
        .map(|c| {
            let mut members = by_region.remove(&c.region).unwrap_or_default();
            members.sort_unstable();
            Region {
                region: c.region,
                size: c.size,
                label: c.label.clone(),
                central: members
                    .into_iter()
                    .filter_map(|(_, id)| memories.remove(&id).map(|m| names.view(m)))
                    .collect(),
            }
        })
        .collect();
    let unclustered = run.points.iter().filter(|p| p.region.is_none()).count() as u32;

    let budget = ctx.budget(p.limit);
    let used = tokens_of(&project.project.name);
    let fitted = fit_whole(regions, budget, used);
    let points = if p.coords {
        let all: Vec<MapPoint> = run
            .points
            .iter()
            .map(|pt| MapPoint {
                id: pt.memory_id,
                region: pt.region,
                x: pt.x,
                y: pt.y,
            })
            .collect();
        Some(if fitted.omitted > 0 {
            let n = all.len() as u32;
            Fitted {
                items: Vec::new(),
                truncated: n > 0,
                omitted: n,
            }
        } else {
            fit_whole(all, budget, used + tokens_of(&fitted.items))
        })
    } else {
        None
    };
    let (points_truncated, points_omitted) = points
        .as_ref()
        .map_or((false, 0), |f| (f.truncated, f.omitted));
    Ok(RegionsReply {
        run: run.id,
        project: project.project.name.clone(),
        memories: run.memories,
        clustered: run.clustered,
        cached,
        regions: fitted.items,
        unclustered,
        points: points.map(|f| f.items),
        truncated: fitted.truncated || points_truncated,
        omitted: fitted.omitted + points_omitted,
    })
}

fn embedded_vector(sh: &Shared, id: &MemoryId) -> Result<Vec<f32>> {
    sh.store.get_memory(id)?;
    sh.store.memory_vector(id)?.ok_or_else(|| {
        CoreError::invalid_input(format!("memory {id} has no embedded chunks yet"))
            .with_next_action("Wait for the embedding queue to reach it (see status), then retry.")
    })
}

pub(crate) fn journey(ctx: &Ctx<'_>, p: &JourneyParams) -> Result<JourneyReply> {
    let sh = ctx.sh;
    let steps = p.steps.unwrap_or(DEFAULT_JOURNEY_STEPS) as usize;
    let (scope, _) = ctx.scope(p.scope)?;
    let from = embedded_vector(sh, &p.from)?;
    let to = embedded_vector(sh, &p.to)?;
    let path = sm_spatial::journey(
        &sh.store,
        &JourneyRequest {
            from: Endpoint {
                id: p.from,
                vector: &from,
            },
            to: Endpoint {
                id: p.to,
                vector: &to,
            },
            steps,
            scope,
            filter: MemoryFilter::default(),
        },
    )?;
    ctx.check_deadline("the journey search")?;
    let ids: Vec<MemoryId> = path.iter().filter_map(|s| s.memory).collect();
    let memories: HashMap<MemoryId, _> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let names = Names::load(&sh.store)?;
    let steps: Vec<JourneyStep> = path
        .into_iter()
        .map(|s| JourneyStep {
            step: s.step,
            t: s.t,
            memory: s.memory.and_then(|id| memories.get(&id).cloned()).map(|m| {
                let mut v = names.view(m);
                v.score = s.score;
                v
            }),
        })
        .collect();
    let fitted = fit_whole(steps, ctx.budget(p.limit), 0);
    Ok(JourneyReply {
        from: p.from,
        to: p.to,
        steps: fitted.items,
        truncated: fitted.truncated,
        omitted: fitted.omitted,
    })
}

#[cfg(test)]
mod tests {
    use super::{COORDS, PLAIN, key, sibling};

    #[test]
    fn s7_each_run_variant_knows_its_sibling() {
        let state = "ab".repeat(32);
        assert_eq!(sibling(&key(&state, PLAIN)), Some(key(&state, COORDS)));
        assert_eq!(sibling(&key(&state, COORDS)), Some(key(&state, PLAIN)));
        assert_eq!(sibling(&state), None);
    }
}
