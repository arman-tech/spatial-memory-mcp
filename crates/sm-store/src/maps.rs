//! The map cache: projection and clustering runs keyed by their input hash,
//! with region labels per (run, region) (SPEC section 4, `map_runs`; section
//! 7; decision B-32).
//!
//! A run is computed by `sm-spatial` and stored whole. Its `input_hash` must
//! cover everything the result depends on (the project, the memories and
//! their vectors, the parameters): the same hash returns the cached run.
//! Cluster numbers are meaningful only within their run, so labels are kept
//! per (run, region) and a new run starts with none. Forgetting a memory
//! removes its points; deleting a project removes its runs.

use std::collections::HashSet;

use rusqlite::{Connection, OptionalExtension, Row};
use sm_core::error::{CoreError, Result};
use sm_core::ids::{MemoryId, ProjectId, RunId};
use sm_core::protocol::MAX_LABEL_CHARS;
use sm_core::time::Timestamp;

use crate::Store;
use crate::codec::{count, from_ts, json_value_col, memory_id_col, sql_limit, ts};
use crate::error::{Ctx, corrupt};
use crate::memories::memory_not_found;
use crate::projects::require_project;

/// One cluster of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapCluster {
    pub region: u32,
    pub size: u32,
    /// Set with [`Store::set_region_label`]; always `None` in a new run.
    pub label: Option<String>,
}

/// One memory's place in a run.
#[derive(Debug, Clone, PartialEq)]
pub struct MapPointRow {
    pub memory_id: MemoryId,
    /// `None` for a memory in no cluster.
    pub region: Option<u32>,
    pub x: f32,
    pub y: f32,
    /// 0 for the memory nearest its cluster's centre, then 1, 2, ...;
    /// `None` when it is not among the central memories.
    pub central_rank: Option<u32>,
}

/// A run to store with [`Store::put_map_run`].
#[derive(Debug, Clone, PartialEq)]
pub struct NewMapRun {
    /// `None` for a run over every project.
    pub project: Option<ProjectId>,
    pub input_hash: String,
    /// The projection and clustering parameters, as JSON.
    pub params: serde_json::Value,
    /// False below the clustering threshold (SPEC section 7).
    pub clustered: bool,
    pub clusters: Vec<MapCluster>,
    pub points: Vec<MapPointRow>,
}

/// A stored run.
#[derive(Debug, Clone, PartialEq)]
pub struct MapRun {
    pub id: RunId,
    pub project: Option<ProjectId>,
    pub input_hash: String,
    pub params: serde_json::Value,
    /// Memories in the run when it was computed. Points of memories
    /// forgotten since are gone, so `points` may be shorter.
    pub memories: u32,
    pub clustered: bool,
    pub created_at: Timestamp,
    /// By region.
    pub clusters: Vec<MapCluster>,
    /// By memory id.
    pub points: Vec<MapPointRow>,
}

fn u32_col(r: &Row<'_>, idx: usize) -> rusqlite::Result<u32> {
    r.get(idx)
}

fn check_label(label: &str) -> Result<()> {
    if label.trim().is_empty()
        || label.chars().count() > MAX_LABEL_CHARS
        || label.contains(['\n', '\r'])
    {
        return Err(CoreError::invalid_input(format!(
            "a region label must be one non-blank line of at most {MAX_LABEL_CHARS} characters"
        )));
    }
    Ok(())
}

fn check_run(run: &NewMapRun) -> Result<()> {
    if run.input_hash.trim().is_empty() {
        return Err(CoreError::invalid_input("a map run needs an input hash"));
    }
    let mut regions = HashSet::new();
    for cl in &run.clusters {
        if cl.label.is_some() {
            return Err(CoreError::invalid_input(
                "a new map run has no region labels (decision B-32)",
            ));
        }
        if !regions.insert(cl.region) {
            return Err(CoreError::invalid_input(format!(
                "region {} is listed twice",
                cl.region
            )));
        }
    }
    let mut seen = HashSet::new();
    for p in &run.points {
        if !seen.insert(p.memory_id) {
            return Err(CoreError::invalid_input(format!(
                "memory {} has two points",
                p.memory_id
            )));
        }
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "memory {} has a non-finite position",
                p.memory_id
            )));
        }
        if let Some(r) = p.region
            && !regions.contains(&r)
        {
            return Err(CoreError::invalid_input(format!(
                "memory {} is in region {r}, which the run does not list",
                p.memory_id
            )));
        }
    }
    u32::try_from(run.points.len())
        .map(drop)
        .map_err(|_| CoreError::invalid_input("a map run has too many points"))
}

const RUN_COLUMNS: &str = "id, project_id, input_hash, params, memories, clustered, created_at";

fn run_from_row(r: &Row<'_>) -> rusqlite::Result<MapRun> {
    Ok(MapRun {
        id: RunId(r.get(0)?),
        project: r.get::<_, Option<i64>>(1)?.map(ProjectId),
        input_hash: r.get(2)?,
        params: json_value_col(r, 3)?,
        memories: u32_col(r, 4)?,
        clustered: r.get(5)?,
        created_at: from_ts(r.get(6)?),
        clusters: Vec::new(),
        points: Vec::new(),
    })
}

/// Fills in a run's clusters and points.
fn load(c: &Connection, mut run: MapRun) -> Result<MapRun> {
    let mut s = c
        .prepare_cached(
            "SELECT region, size, label FROM map_clusters WHERE run_id = ?1 ORDER BY region",
        )
        .ctx("reading a map run")?;
    run.clusters = s
        .query_map([run.id.0], |r| {
            Ok(MapCluster {
                region: u32_col(r, 0)?,
                size: u32_col(r, 1)?,
                label: r.get(2)?,
            })
        })
        .ctx("reading a map run")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading a map run")?;
    let mut s = c
        .prepare_cached(
            "SELECT memory_id, region, x, y, central_rank FROM map_points
             WHERE run_id = ?1 ORDER BY memory_id",
        )
        .ctx("reading a map run")?;
    run.points = s
        .query_map([run.id.0], |r| {
            Ok(MapPointRow {
                memory_id: memory_id_col(r, 0)?,
                region: r.get(1)?,
                // Stored from f32, so the conversion back is exact.
                #[allow(clippy::cast_possible_truncation)]
                x: r.get::<_, f64>(2)? as f32,
                #[allow(clippy::cast_possible_truncation)]
                y: r.get::<_, f64>(3)? as f32,
                central_rank: r.get(4)?,
            })
        })
        .ctx("reading a map run")?
        .collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading a map run")?;
    Ok(run)
}

fn run_where(
    c: &Connection,
    where_: &str,
    param: rusqlite::types::Value,
) -> Result<Option<MapRun>> {
    let run = c
        .prepare_cached(&format!(
            "SELECT {RUN_COLUMNS} FROM map_runs WHERE {where_}"
        ))
        .and_then(|mut s| s.query_row([param], run_from_row).optional())
        .ctx("reading a map run")?;
    run.map(|r| load(c, r)).transpose()
}

impl Store {
    /// Stores a run and returns it with `true`, or, when a run with the same
    /// `input_hash` is already stored, returns that one with `false` and
    /// writes nothing. Every point's memory must exist (`not_found`
    /// otherwise: it was forgotten while the run was computed) and, for a
    /// project run, belong to the project.
    pub fn put_map_run(&self, run: NewMapRun) -> Result<(MapRun, bool)> {
        check_run(&run)?;
        let now = ts(Timestamp::now());
        self.write("storing a map run", |tx| {
            if let Some(existing) = run_where(tx, "input_hash = ?1", run.input_hash.clone().into())? {
                return Ok((existing, false));
            }
            if let Some(p) = run.project {
                require_project(tx, p)?;
            }
            let mut project_of = tx
                .prepare_cached("SELECT project_id FROM memories WHERE id = ?1")
                .ctx("checking a map run's memories")?;
            for p in &run.points {
                let project: Option<i64> = project_of
                    .query_row([p.memory_id.to_string()], |r| r.get(0))
                    .optional()
                    .ctx("checking a map run's memories")?;
                match (project, run.project) {
                    (None, _) => return Err(memory_not_found(&p.memory_id)),
                    (Some(found), Some(want)) if found != want.0 => {
                        return Err(CoreError::invalid_input(format!(
                            "memory {} is not in project {want}",
                            p.memory_id
                        )));
                    }
                    _ => {}
                }
            }
            let params = serde_json::to_string(&run.params)?;
            tx.execute(
                "INSERT INTO map_runs(project_id, input_hash, params, memories, clustered, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                (
                    run.project.map(|p| p.0),
                    &run.input_hash,
                    params,
                    sql_limit(run.points.len()),
                    run.clustered,
                    now,
                ),
            )
            .ctx("storing a map run")?;
            let id = RunId(tx.last_insert_rowid());
            let mut s = tx
                .prepare_cached("INSERT INTO map_clusters(run_id, region, size) VALUES (?1, ?2, ?3)")
                .ctx("storing a map run's clusters")?;
            for cl in &run.clusters {
                s.execute((id.0, cl.region, cl.size))
                    .ctx("storing a map run's clusters")?;
            }
            let mut s = tx
                .prepare_cached(
                    "INSERT INTO map_points(run_id, memory_id, region, x, y, central_rank)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .ctx("storing a map run's points")?;
            for p in &run.points {
                s.execute((
                    id.0,
                    p.memory_id.to_string(),
                    p.region,
                    f64::from(p.x),
                    f64::from(p.y),
                    p.central_rank,
                ))
                .ctx("storing a map run's points")?;
            }
            let stored = run_where(tx, "id = ?1", id.0.into())?
                .ok_or_else(|| corrupt("a map run", "it vanished after being stored"))?;
            Ok((stored, true))
        })
    }

    /// The run with this input hash, if cached.
    pub fn map_run_by_hash(&self, input_hash: &str) -> Result<Option<MapRun>> {
        self.read("reading a map run", |tx| {
            run_where(tx, "input_hash = ?1", input_hash.to_owned().into())
        })
    }

    /// The run with this id.
    pub fn map_run(&self, id: RunId) -> Result<MapRun> {
        self.read("reading a map run", |tx| {
            run_where(tx, "id = ?1", id.0.into())
        })?
        .ok_or_else(|| CoreError::not_found(format!("map run {id} does not exist")))
    }

    /// The newest run for `project` (`None`: the all-projects runs).
    pub fn latest_map_run(&self, project: Option<ProjectId>) -> Result<Option<MapRun>> {
        let param = project.map_or(rusqlite::types::Value::Null, |p| p.0.into());
        self.read("reading a map run", |tx| {
            run_where(
                tx,
                "project_id IS ?1 ORDER BY created_at DESC, id DESC LIMIT 1",
                param,
            )
        })
    }

    /// Names one cluster of one run, or clears its name with `None`
    /// (decision B-32). Fails with `not_found` when the run has no such
    /// region.
    pub fn set_region_label(&self, run: RunId, region: u32, label: Option<&str>) -> Result<()> {
        if let Some(l) = label {
            check_label(l)?;
        }
        self.write("labelling a region", |tx| {
            let n = tx
                .execute(
                    "UPDATE map_clusters SET label = ?3 WHERE run_id = ?1 AND region = ?2",
                    (run.0, region, label),
                )
                .ctx("labelling a region")?;
            if n == 0 {
                return Err(CoreError::not_found(format!(
                    "map run {run} has no region {region}"
                )));
            }
            Ok(())
        })
    }

    /// Deletes all but the newest `keep` runs of `project` (`None`: the
    /// all-projects runs) and returns how many were deleted.
    pub fn prune_map_runs(&self, project: Option<ProjectId>, keep: usize) -> Result<u64> {
        let param = project.map_or(rusqlite::types::Value::Null, |p| p.0.into());
        self.write("pruning map runs", |tx| {
            let n = tx
                .execute(
                    "DELETE FROM map_runs WHERE project_id IS ?1 AND id NOT IN (
                         SELECT id FROM map_runs WHERE project_id IS ?1
                         ORDER BY created_at DESC, id DESC LIMIT ?2)",
                    (param, sql_limit(keep)),
                )
                .ctx("pruning map runs")?;
            Ok(count(i64::try_from(n).unwrap_or(i64::MAX)))
        })
    }
}
