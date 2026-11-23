//! The cases that build or read a store, or run the daemon: preparing the
//! synthetic stores, the cost of a map at each size, and the map cache
//! (B-261). Each runs in a child process of its own, so its peak memory is
//! its own.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sm_core::config::Settings;
use sm_core::hash::sha256_hex;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{MAX_CENTRAL, MapPoint, Reply};
use sm_core::time::Timestamp;
use sm_core::traits::{ChunkPiece, MemoryFilter, ProjectParams, QueryScope};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_spatial::{ALGORITHM, MapParams, MemoryVectors, RegionRun};
use sm_store::{Access, MapCluster, MapPointRow, MapRun, MemoryInput, NewChunk, NewMapRun, Store};

use crate::harness::{Daemon, Env, Home, op};
use crate::meter::{self, Rec, emit, measure, ms};
use crate::pipe;
use crate::procs::Sampler;
use crate::synth::{self, Item, Structure};

pub const SIZES: [usize; 4] = [2_000, 10_000, 50_000, 100_000];
pub const BIG: usize = 100_000;

/// The daemon's run keys (`sm-daemon/src/tools/maps.rs`).
const PLAIN: &str = "regions";
const COORDS: &str = "coords";
const KEEP_RUNS: usize = 4;

pub fn work_dir() -> PathBuf {
    std::env::var_os("SM_MAP_SCALE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sm-map-scale"))
}

pub fn db_path(n: usize) -> PathBuf {
    work_dir().join(format!("syn-{n}.db"))
}

pub fn project_folder(n: usize) -> PathBuf {
    work_dir().join("projects").join(format!("p{n}"))
}

fn suffixed(p: &Path, s: &str) -> PathBuf {
    let mut o = p.as_os_str().to_owned();
    o.push(s);
    PathBuf::from(o)
}

pub fn copy_db(from: &Path, to: &Path) {
    for s in ["", "-wal", "-shm"] {
        let f = suffixed(from, s);
        if f.is_file() {
            std::fs::copy(&f, suffixed(to, s)).expect("copy a store file");
        }
    }
}

pub fn remove_db(db: &Path) {
    for s in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(suffixed(db, s));
    }
}

fn db_bytes(db: &Path) -> u64 {
    ["", "-wal", "-shm"]
        .iter()
        .filter_map(|s| std::fs::metadata(suffixed(db, s)).ok())
        .map(|m| m.len())
        .sum()
}

/// The daemon's index stamp: the model's identity and the default chunking.
pub fn stamp() -> IndexStamp {
    let s = Settings::default();
    IndexStamp {
        model_id: sm_embed::model::MODEL_ID.to_owned(),
        revision: sm_embed::model::REVISION.to_owned(),
        dim: synth::DIMS,
        chunk_max_tokens: s.chunk.max_tokens,
        chunk_overlap: s.chunk.overlap,
    }
}

fn configure(store: &Store) {
    let status = store.configure_index(stamp()).expect("configure the index");
    assert!(status.matches, "the stamp: {:?}", status.differences);
}

pub fn open(db: &Path) -> Store {
    let store = Store::open_at(db, &work_dir().join("backups")).expect("open the store");
    configure(&store);
    store
}

/// One synthetic memory as stored: one embedded chunk carrying its vector,
/// its type and its creation time.
pub fn memory_input(pid: ProjectId, item: &Item, content: String) -> MemoryInput {
    let mut memory = NewMemory::new(pid, content, Provenance::new(ProvenanceSource::Agent));
    memory.id = Some(item.id);
    memory.memory_type = synth::TYPES[item.kind];
    let t = Timestamp::from_micros(item.created_micros);
    memory.created_at = Some(t);
    memory.updated_at = Some(t);
    let text = memory.content.clone();
    let chunk = NewChunk::embedded(
        ChunkPiece {
            seq: 0,
            token_count: u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX),
            text,
        },
        item.vector.clone(),
    );
    MemoryInput {
        memory,
        chunks: vec![chunk],
    }
}

/// Builds `syn-{n}.db` (one project, `n` memories), or reuses it when it
/// already holds them.
pub fn prepare_db(n: usize) {
    std::fs::create_dir_all(work_dir()).expect("the work folder");
    let db = db_path(n);
    if db.is_file() {
        let store = open(&db);
        let have = store
            .memory_vectors(QueryScope::All, &MemoryFilter::default())
            .map(|v| v.len())
            .unwrap_or(0);
        let projects = store.projects().map(|p| p.len()).unwrap_or(0);
        drop(store);
        if have == n && projects == 1 {
            emit(&json!({"case": "prepare", "n": n, "reused": true, "bytes": db_bytes(&db)}));
            return;
        }
        remove_db(&db);
    }
    let t0 = Instant::now();
    let store = open(&db);
    let folder = project_folder(n);
    std::fs::create_dir_all(&folder).expect("the project folder");
    let identity = sm_core::project::resolve_project(&folder).expect("resolve the project");
    let pid = store.upsert_project(&identity).expect("the project").id;
    let items = synth::generate(&Structure::new(), n);
    for (b, chunk) in items.chunks(1_000).enumerate() {
        let batch = chunk
            .iter()
            .enumerate()
            .map(|(j, it)| memory_input(pid, it, it.content(n, b * 1_000 + j)))
            .collect();
        store
            .insert_memories(batch, Access::Local)
            .expect("insert memories");
    }
    drop(store);
    emit(&json!({
        "case": "prepare", "n": n, "reused": false,
        "ms": ms(t0.elapsed()), "bytes": db_bytes(&db),
    }));
}

pub fn only_project(store: &Store) -> ProjectId {
    let projects = store.projects().expect("the projects");
    assert_eq!(projects.len(), 1, "one project in a synthetic store");
    projects[0].id
}

pub fn read_vectors(store: &Store, pid: ProjectId) -> Vec<MemoryVectors> {
    store
        .memory_vectors(QueryScope::Project(pid), &MemoryFilter::default())
        .expect("the memory vectors")
        .into_iter()
        .map(|(id, v)| MemoryVectors {
            id,
            chunks: vec![v],
        })
        .collect()
}

pub fn recs_json(recs: &[Rec]) -> Value {
    Value::Array(recs.iter().map(Rec::json).collect())
}

// ---------------------------------------------------------------------------
// 1. Cost
// ---------------------------------------------------------------------------

/// The library as the daemon calls it: read, `regions`, `coordinates`.
pub fn cost_lib(n: usize) {
    let store = open(&db_path(n));
    let pid = only_project(&store);
    let params = MapParams::default();
    let (mem, r0) = measure("read vectors", || read_vectors(&store, pid));
    assert_eq!(mem.len(), n, "every memory read");
    let (run, r1) = measure("regions()", || {
        sm_spatial::regions(&mem, &params).expect("regions")
    });
    let (pts, r2) = measure("coordinates()", || {
        sm_spatial::coordinates(&mem, &params, &run).expect("coordinates")
    });
    emit(&json!({
        "case": "cost-lib", "n": n,
        "regions": run.regions.len(), "points": pts.len(),
        "unclustered": run.unclustered.len(),
        "recs": recs_json(&[r0, r1, r2]),
        "process_peak_ws": meter::now_mem().peak_working_set,
    }));
}

/// The same pipeline split into its stages by the replica in `pipe.rs`;
/// with `verify`, checked bit for bit against the library afterwards.
pub fn cost_stages(n: usize, verify: bool) {
    let store = open(&db_path(n));
    let pid = only_project(&store);
    let params = MapParams::default();
    let mut recs = Vec::new();
    let (mem, r) = measure("read vectors", || read_vectors(&store, pid));
    recs.push(r);
    let (prep, r) = measure("prepare (mean vectors, sort)", || pipe::prepare(&mem));
    recs.push(r);
    let (hash, r) = measure("input hash", || sm_spatial::input_hash(&prep.ids, &params));
    recs.push(r);
    let ((sample, sv), r) = measure("sample (order, copy)", || {
        let s = pipe::sample_rows(&prep.ids, params.max_points, params.project.seed);
        let v: Vec<Vec<f32>> = s.iter().map(|&r| prep.vectors[r].clone()).collect();
        (s, v)
    });
    recs.push(r);
    let m = sample.len();
    let (points, r) = measure("project 5-D", || pipe::project(&sv, &params.project));
    recs.push(r);
    drop(sv);
    let ((labels, cp), r) = measure("cluster", || pipe::cluster(&points));
    recs.push(r);
    let (label, r) = measure("assign the rest", || {
        let c = pipe::centres(&prep, &sample, &labels);
        pipe::assign(&prep, &sample, &labels, &c)
    });
    recs.push(r);
    let (run, r) = measure("build regions (rank members)", || {
        let regions = pipe::build_regions(&prep, &label);
        let unclustered = (0..n)
            .filter(|&r| label[r].is_none())
            .map(|r| prep.ids[r])
            .collect();
        RegionRun {
            input_hash: hash,
            memories: n,
            clustered: true,
            sampled: m < n,
            projected: m,
            cluster_params: Some(cp),
            regions,
            unclustered,
        }
    });
    recs.push(r);
    drop(prep);
    drop(points);
    let ((prep2, sample2), r) = measure("coordinates: prepare and check", || {
        let p = pipe::prepare(&mem);
        let a = run.assignments();
        assert!(
            sm_spatial::input_hash(&p.ids, &params) == run.input_hash
                && a.len() == p.ids.len()
                && a.iter().zip(&p.ids).all(|(a, id)| a.0 == *id),
            "the run matches the memories"
        );
        let s = pipe::sample_rows(&p.ids, params.max_points, params.project.seed);
        (p, s)
    });
    recs.push(r);
    let (sxy, r) = measure("project 2-D", || {
        let p = ProjectParams {
            neighbors: params.coords.neighbors.min(m - 1),
            ..params.coords
        };
        let v: Vec<Vec<f32>> = sample2.iter().map(|&r| prep2.vectors[r].clone()).collect();
        pipe::project(&v, &p)
            .into_iter()
            .map(|q| [q[0], q[1]])
            .collect::<Vec<[f32; 2]>>()
    });
    recs.push(r);
    let (xy, r) = measure("place the rest", || {
        pipe::place(&prep2, &sample2, &sxy, params.coords.neighbors.min(m))
    });
    recs.push(r);
    let mut out = json!({
        "case": "cost-stages", "n": n, "sample": m,
        "regions": run.regions.len(),
        "recs": recs_json(&recs),
        "process_peak_ws": meter::now_mem().peak_working_set,
    });
    if verify {
        let lib = sm_spatial::regions(&mem, &params).expect("regions");
        let same_regions = lib.input_hash == run.input_hash
            && lib.assignments() == run.assignments()
            && lib.unclustered == run.unclustered
            && lib.regions.len() == run.regions.len()
            && lib.regions.iter().zip(&run.regions).all(|(a, b)| {
                a.region == b.region
                    && a.members.len() == b.members.len()
                    && a.members.iter().zip(&b.members).all(|(x, y)| {
                        x.id == y.id && x.centrality.to_bits() == y.centrality.to_bits()
                    })
                    && a.centre
                        .iter()
                        .zip(&b.centre)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            });
        let pts = sm_spatial::coordinates(&mem, &params, &lib).expect("coordinates");
        let mismatches = pts
            .iter()
            .zip(prep2.ids.iter().zip(&xy))
            .filter(|(p, (id, q))| {
                p.id != **id || p.x.to_bits() != q[0].to_bits() || p.y.to_bits() != q[1].to_bits()
            })
            .count()
            + pts.len().abs_diff(xy.len());
        out["verify"] =
            json!({"regions_identical": same_regions, "coordinate_mismatches": mismatches});
    }
    emit(&out);
}

// ---------------------------------------------------------------------------
// 2. Cache
// ---------------------------------------------------------------------------

fn key(state: &str, variant: &str) -> String {
    format!("{state}:{variant}")
}

/// The daemon's `input`: the vectors, every memory's `updated_at` and the
/// SHA-256 state key, as three stages.
fn input_stages(
    store: &Store,
    pid: ProjectId,
    params: &MapParams,
    recs: &mut Vec<Rec>,
) -> (Vec<MemoryVectors>, String) {
    let (vectors, r) = measure("input: read vectors", || {
        store
            .memory_vectors(QueryScope::Project(pid), &MemoryFilter::default())
            .expect("the vectors")
    });
    recs.push(r);
    let ids: Vec<MemoryId> = vectors.iter().map(|(id, _)| *id).collect();
    let (updated, r) = measure("input: read updated_at", || {
        store
            .get_memories(&ids)
            .expect("the memories")
            .into_iter()
            .map(|m| (m.id, m.updated_at.as_micros()))
            .collect::<HashMap<MemoryId, i64>>()
    });
    recs.push(r);
    let (state, r) = measure("input: state hash", || {
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
        sha256_hex(&bytes)
    });
    recs.push(r);
    let memories = vectors
        .into_iter()
        .map(|(id, v)| MemoryVectors {
            id,
            chunks: vec![v],
        })
        .collect();
    (memories, state)
}

/// The daemon's `new_run`, unchanged.
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

/// A coordinates request that misses the cache, as `run_for` serves it.
fn request_miss(store: &Store, pid: ProjectId, params: &MapParams) -> (MapRun, Vec<Rec>) {
    let mut recs = Vec::new();
    let (mem, state) = input_stages(store, pid, params, &mut recs);
    let (hit, r) = measure("stored lookup (miss)", || {
        store
            .map_run_by_hash(&key(&state, COORDS))
            .expect("the lookup")
    });
    recs.push(r);
    assert!(hit.is_none(), "a miss");
    let (run, r) = measure("regions()", || {
        sm_spatial::regions(&mem, params).expect("regions")
    });
    recs.push(r);
    let (pts, r) = measure("coordinates()", || {
        sm_spatial::coordinates(&mem, params, &run).expect("coordinates")
    });
    recs.push(r);
    let (new, r) = measure("build the stored run", || {
        new_run(pid, key(&state, COORDS), &run, Some(&pts), params)
    });
    recs.push(r);
    let ((stored, created), r) = measure("put_map_run", || {
        store.put_map_run(new).expect("store the run")
    });
    recs.push(r);
    assert!(created, "a new run");
    let (_, r) = measure("labels lookup", || {
        store
            .map_run_by_hash(&key(&state, PLAIN))
            .expect("the lookup")
    });
    recs.push(r);
    let (_, r) = measure("prune runs", || {
        store.prune_map_runs(Some(pid), KEEP_RUNS).expect("prune")
    });
    recs.push(r);
    (stored, recs)
}

/// The cache on a copy of the 100,000-memory store, through the library
/// and the store as `run_for` uses them.
pub fn cache_lib(rep: usize) {
    let db = work_dir().join(format!("cache-lib-{rep}.db"));
    remove_db(&db);
    copy_db(&db_path(BIG), &db);
    let out = {
        let store = open(&db);
        let pid = only_project(&store);
        let params = MapParams::default();
        let (first, miss) = request_miss(&store, pid, &params);
        let (hit_run, r) = measure("index hit: map_run(id)", || {
            store.map_run(first.id).expect("the run")
        });
        let index_hit = vec![r];
        let mut stored_hit = Vec::new();
        let (_, state) = input_stages(&store, pid, &params, &mut stored_hit);
        let (found, r) = measure("stored lookup (hit)", || {
            store
                .map_run_by_hash(&key(&state, COORDS))
                .expect("the lookup")
        });
        stored_hit.push(r);
        assert!(found.is_some(), "a stored hit");
        let extra = synth::extra(&Structure::new(), BIG, 1, 7);
        let (_, r) = measure("insert one memory", || {
            store
                .insert_memories(
                    vec![memory_input(
                        pid,
                        &extra[0],
                        "map-scale cache extra".to_owned(),
                    )],
                    Access::Local,
                )
                .expect("insert")
        });
        let add = vec![r];
        let (second, after) = request_miss(&store, pid, &params);
        json!({
            "case": "cache-lib", "rep": rep,
            "points_first": hit_run.points.len(), "points_after": second.points.len(),
            "miss": recs_json(&miss), "index_hit": recs_json(&index_hit),
            "stored_hit": recs_json(&stored_hit), "add": recs_json(&add),
            "after_add": recs_json(&after),
            "db_bytes": db_bytes(&db),
        })
    };
    remove_db(&db);
    emit(&out);
}

fn daemon_request(home: &Home, d: &Daemon, cwd: &Path, limit: Option<u32>, step: &str) -> Value {
    let mut args = json!({"coords": true});
    if let Some(l) = limit {
        args["limit"] = json!(l);
    }
    let before = d.read();
    let sampler = Sampler::start(d.pid, Duration::from_millis(20));
    let t0 = Instant::now();
    let reply = home.try_call(cwd, None, op(json!({"op": "regions", "args": args})));
    let elapsed = t0.elapsed();
    let peak = sampler.stop();
    let after = d.read();
    let mut v = json!({
        "step": step, "ms": ms(elapsed), "limit": limit,
        "ws_before": before.working_set, "ws_after": after.working_set,
        "peak_ws": peak.max_working_set.max(after.working_set),
        "peak_private": peak.max_private.max(after.private),
    });
    match reply {
        Ok(Reply::Regions(r)) => {
            let bytes = serde_json::to_vec(&r).map(|b| b.len()).unwrap_or(0);
            v["cached"] = json!(r.cached);
            v["memories"] = json!(r.memories);
            v["regions"] = json!(r.regions.len());
            v["unclustered"] = json!(r.unclustered);
            v["points"] = json!(r.points.as_ref().map_or(0, Vec::len));
            v["truncated"] = json!(r.truncated);
            v["omitted"] = json!(r.omitted);
            v["reply_bytes"] = json!(bytes);
        }
        Ok(_) => v["error"] = json!("an unexpected reply"),
        Err(e) => v["error"] = json!(e),
    }
    v
}

/// The cache through the release daemon on a scratch profile holding a copy
/// of the 100,000-memory store. The added memory goes in through the store
/// while the daemon is stopped: `remember` would embed it.
pub fn cache_daemon(rep: usize) {
    let env = match Env::find() {
        Ok(e) => e,
        Err(line) => {
            emit(&json!({"case": "cache-daemon", "rep": rep, "skipped": line}));
            return;
        }
    };
    let home = Home::new(&env, "mapscale");
    let db = home.paths.database();
    std::fs::create_dir_all(db.parent().expect("a profile folder")).expect("the profile folder");
    copy_db(&db_path(BIG), &db);
    let cwd = project_folder(BIG);
    let all = Some(100_000_000);
    let mut steps = Vec::new();

    let mut d = home.start_ready();
    let pending_start = home.status().index.pending_embeddings;
    steps.push(json!({"step": "start", "ms": ms(d.ready), "ws_after": d.read().working_set}));
    steps.push(daemon_request(
        &home,
        &d,
        &cwd,
        None,
        "A first request, default limit",
    ));
    steps.push(daemon_request(&home, &d, &cwd, None, "B same again"));
    steps.push(daemon_request(&home, &d, &cwd, all, "C every point"));
    d.stop(&home);

    {
        let store = Store::open(&home.paths).expect("open the profile store");
        configure(&store);
        let pid = only_project(&store);
        let extra = synth::extra(&Structure::new(), BIG, 1, 7);
        store
            .insert_memories(
                vec![memory_input(
                    pid,
                    &extra[0],
                    "map-scale cache extra".to_owned(),
                )],
                Access::Local,
            )
            .expect("insert");
    }

    let mut d = home.start_ready();
    steps.push(json!({"step": "restart after one memory added", "ms": ms(d.ready), "ws_after": d.read().working_set}));
    steps.push(daemon_request(
        &home,
        &d,
        &cwd,
        all,
        "D after one memory added",
    ));
    steps.push(daemon_request(&home, &d, &cwd, all, "E same again"));
    d.stop(&home);

    let d = home.start_ready();
    steps.push(json!({"step": "restart", "ms": ms(d.ready), "ws_after": d.read().working_set}));
    steps.push(daemon_request(
        &home,
        &d,
        &cwd,
        all,
        "F after a restart (stored run)",
    ));
    let pending_end = home.status().index.pending_embeddings;
    drop(d);
    emit(&json!({
        "case": "cache-daemon", "rep": rep,
        "pending_embeddings": [pending_start, pending_end],
        "steps": steps,
    }));
}
