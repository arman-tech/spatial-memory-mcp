//! The analysis cases (B-261): quality, neighbours, clumping and payload on
//! the synthetic stores, the three alternatives, and the real store. Only
//! counts and numbers leave this file; no memory text, project name or
//! path is ever emitted.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use serde_json::{Value, json};
use sm_core::hash::fnv1a64;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, ProjectParams, QueryScope};
use sm_spatial::{MapParams, MemoryVectors, RegionRun, adjusted_rand_index};
use sm_store::Store;

use crate::cases::{self, BIG};
use crate::meter::{self, Rec, emit, measure, ms};
use crate::pipe::{self, Prepared};
use crate::synth::{self, Rng, Structure};

/// Queries for the neighbour measure.
const QUERIES: usize = 1_000;
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// One library run, with its rows in identifier order.
pub struct Base {
    pub run: RegionRun,
    pub prep: Prepared,
    pub sample: Vec<usize>,
    /// Region per row.
    pub label: Vec<Option<usize>>,
    pub xy: Vec<[f32; 2]>,
}

impl Base {
    fn row_of(&self) -> HashMap<MemoryId, usize> {
        self.prep
            .ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect()
    }
}

/// `sm_spatial::regions` and `coordinates` on `mem`, rows put in order.
pub fn base(mem: &[MemoryVectors], params: &MapParams) -> Base {
    let run = sm_spatial::regions(mem, params).expect("regions");
    let pts = sm_spatial::coordinates(mem, params, &run).expect("coordinates");
    let prep = pipe::prepare(mem);
    let n = prep.ids.len();
    let row: HashMap<MemoryId, usize> = prep
        .ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut label = vec![None; n];
    for (id, r) in run.assignments() {
        label[row[&id]] = r.map(|x| x as usize);
    }
    let mut xy = vec![[0.0_f32; 2]; n];
    for p in &pts {
        xy[row[&p.id]] = [p.x, p.y];
    }
    let sample = pipe::sample_rows(&prep.ids, params.max_points, params.project.seed);
    Base {
        run,
        prep,
        sample,
        label,
        xy,
    }
}

/// Topic and sub-topic per row (noise as 40 and 200), regenerated.
pub struct Truth {
    pub topic: Vec<usize>,
    pub sub: Vec<usize>,
}

pub fn truth_for(ids: &[MemoryId], n: usize) -> Truth {
    let items = synth::generate(&Structure::new(), n);
    let by_id: HashMap<MemoryId, (usize, usize)> = items
        .iter()
        .map(|it| (it.id, (it.topic_label(), it.sub_label())))
        .collect();
    let (topic, sub) = ids
        .iter()
        .map(|id| by_id.get(id).copied().expect("a generated memory"))
        .unzip();
    Truth { topic, sub }
}

fn in_sample(n: usize, sample: &[usize]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &r in sample {
        m[r] = true;
    }
    m
}

fn ari_on(truth: &[usize], label: &[Option<usize>], rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let t: Vec<usize> = rows.iter().map(|&r| truth[r]).collect();
    let p: Vec<Option<usize>> = rows.iter().map(|&r| label[r]).collect();
    adjusted_rand_index(&t, &p)
}

fn share(count: usize, of: usize) -> f64 {
    if of == 0 {
        f64::NAN
    } else {
        count as f64 / of as f64
    }
}

/// Regions against the 40 topics and 200 sub-topics.
pub fn quality(t: &Truth, label: &[Option<usize>], sample: &[usize]) -> Value {
    let n = label.len();
    let mask = in_sample(n, sample);
    let all: Vec<usize> = (0..n).collect();
    let members: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&r| t.topic[r] < synth::TOPICS)
        .collect();
    let noise: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&r| t.topic[r] == synth::TOPICS)
        .collect();
    let sampled: Vec<usize> = all.iter().copied().filter(|&r| mask[r]).collect();
    let placed: Vec<usize> = all.iter().copied().filter(|&r| !mask[r]).collect();
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for l in label.iter().flatten() {
        *sizes.entry(*l).or_default() += 1;
    }
    let mut size_list: Vec<usize> = sizes.values().copied().collect();
    size_list.sort_unstable_by(|a, b| b.cmp(a));
    let unclustered = |rows: &[usize]| -> f64 {
        share(
            rows.iter().filter(|&&r| label[r].is_none()).count(),
            rows.len(),
        )
    };
    let mt: Vec<usize> = members.iter().map(|&r| t.topic[r]).collect();
    let ms_: Vec<usize> = members.iter().map(|&r| t.sub[r]).collect();
    let mp: Vec<Option<usize>> = members.iter().map(|&r| label[r]).collect();
    json!({
        "memories": n,
        "regions": size_list.len(),
        "ari_topics_all": ari_on(&t.topic, label, &all),
        "ari_subs_all": ari_on(&t.sub, label, &all),
        "ari_topics_members": ari_on(&t.topic, label, &members),
        "ari_subs_members": ari_on(&t.sub, label, &members),
        "ari_topics_sampled": ari_on(&t.topic, label, &sampled),
        "ari_topics_placed": ari_on(&t.topic, label, &placed),
        "ari_subs_sampled": ari_on(&t.sub, label, &sampled),
        "ari_subs_placed": ari_on(&t.sub, label, &placed),
        "majority_topics": pipe::majority_classes(&mt, &mp),
        "majority_subs": pipe::majority_classes(&ms_, &mp),
        "unclustered_all": unclustered(&all),
        "unclustered_noise": unclustered(&noise),
        "unclustered_members": unclustered(&members),
        "unclustered_sampled": unclustered(&sampled),
        "largest_share": share(size_list.first().copied().unwrap_or(0), n),
        "region_sizes": size_list,
    })
}

fn overlap_stats(ov: &[&pipe::Overlap]) -> Value {
    let shares: Vec<f64> = ov.iter().map(|o| o.share).collect();
    let cos1: Vec<f64> = ov.iter().map(|o| f64::from(o.cos1)).collect();
    let cos10: Vec<f64> = ov.iter().map(|o| f64::from(o.cos10)).collect();
    let ties: Vec<f64> = ov.iter().map(|o| o.share_ties).collect();
    let tie_set: Vec<f64> = ov.iter().map(|o| o.tie_set as f64).collect();
    json!({
        "queries": ov.len(),
        "mean": pipe::mean(&shares),
        "median": pipe::median(&shares),
        "none_share": share(ov.iter().filter(|o| o.share == 0.0).count(), ov.len()),
        "mean_with_ties": pipe::mean(&ties),
        "none_share_with_ties": share(ov.iter().filter(|o| o.share_ties == 0.0).count(), ov.len()),
        "tie_set_median": pipe::median(&tie_set),
        "tie_set_share_over_10": share(ov.iter().filter(|o| o.tie_set > 10).count(), ov.len()),
        "tie_set_max": tie_set.iter().copied().fold(0.0, f64::max),
        "cos1_mean": pipe::mean(&cos1),
        "cos10_mean": pipe::mean(&cos10),
    })
}

/// The share of each memory's 10 nearest by cosine that are among its 10
/// nearest on the map: 1,000 random memories split by sample and placed,
/// and 1,000 drawn from the sample.
pub fn neighbours(prep: &Prepared, xy: &[[f32; 2]], sample: &[usize], seed: u64) -> Value {
    let n = prep.ids.len();
    let mask = in_sample(n, sample);
    let mut rng = Rng::new(seed);
    let random = rng.distinct(n, QUERIES.min(n));
    let ov = pipe::overlaps(&prep.vectors, xy, &random);
    let s: Vec<&pipe::Overlap> = ov.iter().filter(|o| mask[o.row]).collect();
    let p: Vec<&pipe::Overlap> = ov.iter().filter(|o| !mask[o.row]).collect();
    let all: Vec<&pipe::Overlap> = ov.iter().collect();
    let from_sample = if sample.len() < n {
        let picks: Vec<usize> = rng
            .distinct(sample.len(), QUERIES.min(sample.len()))
            .into_iter()
            .map(|i| sample[i])
            .collect();
        let o = pipe::overlaps(&prep.vectors, xy, &picks);
        overlap_stats(&o.iter().collect::<Vec<_>>())
    } else {
        Value::Null
    };
    let mut pair = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let a = rng.below(n);
        let b = rng.below(n);
        if a != b {
            pair.push(f64::from(pipe::dot(&prep.vectors[a], &prep.vectors[b])));
        }
    }
    json!({
        "memories": n,
        "sample": sample.len(),
        "chance": 10.0 / (n.saturating_sub(1).max(1)) as f64,
        "random_all": overlap_stats(&all),
        "random_sampled": overlap_stats(&s),
        "random_placed": overlap_stats(&p),
        "drawn_from_sample": from_sample,
        "random_pair_cos_mean": pipe::mean(&pair),
    })
}

pub fn clumping(xy: &[[f32; 2]], sample: &[usize]) -> Value {
    let c = pipe::clumping(xy, sample);
    json!({
        "placed": c.placed,
        "distinct_placed": c.distinct_placed,
        "sample_median_nn": c.sample_median_nn,
        "placed_median_nn": c.placed_median_nn,
        "placed_below": c.placed_below,
        "sampled_below": c.sampled_below,
        "placed_zero": c.placed_zero,
        "sampled_zero": c.sampled_zero,
        "diagonal": pipe::diagonal(xy),
    })
}

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------

/// What a viewer needs per memory besides its position.
#[derive(Clone, Copy)]
struct Attr {
    day: i64,
    kind: &'static str,
    status: &'static str,
}

#[derive(Serialize)]
struct JsonRow {
    x: f32,
    y: f32,
    day: i64,
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<usize>,
    status: &'static str,
}

#[derive(Serialize)]
struct JsonRowId {
    id: String,
    x: f32,
    y: f32,
    day: i64,
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<usize>,
    status: &'static str,
}

fn table(names: &[&'static str]) -> Vec<&'static str> {
    let mut v = names.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// The compact table: a 16-byte header ("SMMP", version, flags, count,
/// first day), the type and status name tables, then 14 bytes a memory
/// (x f32, y f32, day u16 from the first day, type u8, region u16 with
/// 0xFFFF for none, status u8) and, with `ids`, 16 more.
fn encode_binary(
    attrs: &[Attr],
    xy: &[[f32; 2]],
    label: &[Option<usize>],
    ids: Option<&[MemoryId]>,
) -> Vec<u8> {
    let types = table(&attrs.iter().map(|a| a.kind).collect::<Vec<_>>());
    let statuses = table(&attrs.iter().map(|a| a.status).collect::<Vec<_>>());
    let day0 = attrs.iter().map(|a| a.day).min().unwrap_or(0);
    let per = 14 + if ids.is_some() { 16 } else { 0 };
    let mut out = Vec::with_capacity(64 + attrs.len() * per);
    out.extend_from_slice(b"SMMP");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&u16::from(ids.is_some()).to_le_bytes());
    out.extend_from_slice(&u32::try_from(attrs.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&u32::try_from(day0).unwrap_or(0).to_le_bytes());
    for names in [&types, &statuses] {
        out.push(u8::try_from(names.len()).unwrap_or(u8::MAX));
        for n in names.iter() {
            out.push(u8::try_from(n.len()).unwrap_or(u8::MAX));
            out.extend_from_slice(n.as_bytes());
        }
    }
    for (i, a) in attrs.iter().enumerate() {
        out.extend_from_slice(&xy[i][0].to_le_bytes());
        out.extend_from_slice(&xy[i][1].to_le_bytes());
        let day = u16::try_from(a.day - day0).unwrap_or(u16::MAX);
        out.extend_from_slice(&day.to_le_bytes());
        out.push(u8::try_from(types.binary_search(&a.kind).unwrap_or(0)).unwrap_or(u8::MAX));
        let region = label[i].map_or(u16::MAX, |r| u16::try_from(r).unwrap_or(u16::MAX - 1));
        out.extend_from_slice(&region.to_le_bytes());
        out.push(u8::try_from(statuses.binary_search(&a.status).unwrap_or(0)).unwrap_or(u8::MAX));
        if let Some(ids) = ids {
            out.extend_from_slice(ids[i].as_uuid().as_bytes());
        }
    }
    out
}

fn encode_json(attrs: &[Attr], xy: &[[f32; 2]], label: &[Option<usize>]) -> Vec<u8> {
    let rows: Vec<JsonRow> = attrs
        .iter()
        .enumerate()
        .map(|(i, a)| JsonRow {
            x: xy[i][0],
            y: xy[i][1],
            day: a.day,
            r#type: a.kind,
            region: label[i],
            status: a.status,
        })
        .collect();
    serde_json::to_vec(&rows).expect("json")
}

fn encode_json_id(
    attrs: &[Attr],
    xy: &[[f32; 2]],
    label: &[Option<usize>],
    ids: &[MemoryId],
) -> Vec<u8> {
    let rows: Vec<JsonRowId> = attrs
        .iter()
        .enumerate()
        .map(|(i, a)| JsonRowId {
            id: ids[i].to_string(),
            x: xy[i][0],
            y: xy[i][1],
            day: a.day,
            r#type: a.kind,
            region: label[i],
            status: a.status,
        })
        .collect();
    serde_json::to_vec(&rows).expect("json")
}

fn timed_bytes(f: impl Fn() -> Vec<u8>) -> Value {
    let mut times = Vec::new();
    let mut bytes = 0;
    for _ in 0..3 {
        let t0 = Instant::now();
        let b = f();
        times.push(ms(t0.elapsed()));
        bytes = b.len();
    }
    json!({"bytes": bytes, "ms": times})
}

/// The payload of every memory's position and attributes, four ways.
fn payload(store: &Store, b: &Base) -> Value {
    let mut read = Vec::new();
    let mut attrs = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let found = store.get_memories(&b.prep.ids).expect("the memories");
        read.push(ms(t0.elapsed()));
        let by_id: HashMap<MemoryId, Attr> = found
            .into_iter()
            .map(|m| {
                (
                    m.id,
                    Attr {
                        day: m.created_at.as_micros().div_euclid(MICROS_PER_DAY),
                        kind: m.memory_type.as_str(),
                        status: m.status.as_str(),
                    },
                )
            })
            .collect();
        attrs = b
            .prep
            .ids
            .iter()
            .map(|id| *by_id.get(id).expect("each memory read"))
            .collect();
    }
    json!({
        "memories": b.prep.ids.len(),
        "read_attributes_ms": read,
        "binary": timed_bytes(|| encode_binary(&attrs, &b.xy, &b.label, None)),
        "binary_with_id": timed_bytes(|| encode_binary(&attrs, &b.xy, &b.label, Some(&b.prep.ids))),
        "json": timed_bytes(|| encode_json(&attrs, &b.xy, &b.label)),
        "json_with_id": timed_bytes(|| encode_json_id(&attrs, &b.xy, &b.label, &b.prep.ids)),
    })
}

// ---------------------------------------------------------------------------
// 3-6. Quality, neighbours, clumping and payload at each size
// ---------------------------------------------------------------------------

pub fn analyse(n: usize) {
    let store = cases::open(&cases::db_path(n));
    let pid = cases::only_project(&store);
    let mem = cases::read_vectors(&store, pid);
    let params = MapParams::default();
    let b = base(&mem, &params);
    drop(mem);
    let t = truth_for(&b.prep.ids, n);
    emit(&json!({
        "case": "analyse", "n": n,
        "quality": quality(&t, &b.label, &b.sample),
        "neighbours": neighbours(&b.prep, &b.xy, &b.sample, 0x0261_0001),
        "clumping": clumping(&b.xy, &b.sample),
        "payload": payload(&store, &b),
    }));
}

// ---------------------------------------------------------------------------
// 7a. Larger samples
// ---------------------------------------------------------------------------

/// The 100,000-memory set mapped from a sample of `s`, through the replica
/// (holomap called directly above 2,000 rows), every stage timed.
pub fn alt7a(s: usize, rep: usize) {
    let store = cases::open(&cases::db_path(BIG));
    let pid = cases::only_project(&store);
    let mem = cases::read_vectors(&store, pid);
    drop(store);
    let prep = pipe::prepare(&mem);
    drop(mem);
    let params = MapParams::default();
    let mut recs: Vec<Rec> = Vec::new();
    let (sample, r) = measure("sample", || {
        pipe::sample_rows(&prep.ids, s, params.project.seed)
    });
    recs.push(r);
    let m = sample.len();
    let (points, r) = measure("project 5-D", || {
        let v: Vec<Vec<f32>> = sample.iter().map(|&r| prep.vectors[r].clone()).collect();
        pipe::project(&v, &params.project)
    });
    recs.push(r);
    let ((labels, cp), r) = measure("cluster", || pipe::cluster(&points));
    recs.push(r);
    drop(points);
    let (label, r) = measure("assign the rest", || {
        let c = pipe::centres(&prep, &sample, &labels);
        pipe::assign(&prep, &sample, &labels, &c)
    });
    recs.push(r);
    let (regions, r) = measure("build regions", || pipe::build_regions(&prep, &label));
    recs.push(r);
    let (sxy, r) = measure("project 2-D", || {
        let p = ProjectParams {
            neighbors: params.coords.neighbors.min(m - 1),
            ..params.coords
        };
        let v: Vec<Vec<f32>> = sample.iter().map(|&r| prep.vectors[r].clone()).collect();
        pipe::project(&v, &p)
            .into_iter()
            .map(|q| [q[0], q[1]])
            .collect::<Vec<[f32; 2]>>()
    });
    recs.push(r);
    let (xy, r) = measure("place the rest", || {
        pipe::place(&prep, &sample, &sxy, params.coords.neighbors.min(m))
    });
    recs.push(r);
    let mut out = json!({
        "case": "alt7a", "s": s, "rep": rep, "n": prep.ids.len(),
        "regions": regions.len(),
        "cluster_params": format!("{cp:?}"),
        "recs": cases::recs_json(&recs),
        "process_peak_ws": meter::now_mem().peak_working_set,
    });
    if rep == 0 {
        let t = truth_for(&prep.ids, BIG);
        out["quality"] = quality(&t, &label, &sample);
        out["neighbours"] = neighbours(&prep, &xy, &sample, 0x0261_0001);
        out["clumping"] = clumping(&xy, &sample);
    }
    emit(&out);
}

// ---------------------------------------------------------------------------
// 7b. One new memory into an existing map
// ---------------------------------------------------------------------------

fn sample_key(id: &MemoryId, seed: u64) -> u64 {
    let mut bytes = [0u8; 24];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    bytes[8..].copy_from_slice(id.as_uuid().as_bytes());
    fnv1a64(&bytes)
}

fn with_extra(mem: &[MemoryVectors], e: &synth::Item) -> Vec<MemoryVectors> {
    let mut v = mem.to_vec();
    v.push(MemoryVectors {
        id: e.id,
        chunks: vec![e.vector.clone()],
    });
    v
}

/// Old labels as classes, unclustered as one more class.
fn as_classes(label: &[Option<usize>]) -> Vec<usize> {
    let none = label.iter().flatten().max().map_or(0, |m| m + 1);
    label.iter().map(|l| l.unwrap_or(none)).collect()
}

fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

/// How a rebuild with one more memory compares with the base, over the
/// base's memories.
fn compare(old: &Base, new: &Base) -> Value {
    let row_new = new.row_of();
    let pairs: Vec<(usize, usize)> = old
        .prep
        .ids
        .iter()
        .enumerate()
        .map(|(r, id)| (r, row_new[id]))
        .collect();
    let old_label: Vec<Option<usize>> = pairs.iter().map(|&(a, _)| old.label[a]).collect();
    let new_label: Vec<Option<usize>> = pairs.iter().map(|&(_, b)| new.label[b]).collect();
    let mut old_to_new: HashMap<Option<usize>, HashMap<Option<usize>, usize>> = HashMap::new();
    for (a, b) in old_label.iter().zip(&new_label) {
        *old_to_new.entry(*a).or_default().entry(*b).or_default() += 1;
    }
    let mut new_to_old: HashMap<Option<usize>, std::collections::HashSet<Option<usize>>> =
        HashMap::new();
    for (a, b) in old_label.iter().zip(&new_label) {
        new_to_old.entry(*b).or_default().insert(*a);
    }
    let same_partition =
        old_to_new.values().all(|m| m.len() == 1) && new_to_old.values().all(|s| s.len() == 1);
    let kept: usize = old_to_new
        .values()
        .map(|m| m.values().copied().max().unwrap_or(0))
        .sum();
    let same_xy = pairs.iter().all(|&(a, b)| {
        old.xy[a][0].to_bits() == new.xy[b][0].to_bits()
            && old.xy[a][1].to_bits() == new.xy[b][1].to_bits()
    });
    let diag = pipe::diagonal(&old.xy);
    let raw: Vec<f64> = pairs
        .iter()
        .map(|&(a, b)| {
            let dx = f64::from(new.xy[b][0]) - f64::from(old.xy[a][0]);
            let dy = f64::from(new.xy[b][1]) - f64::from(old.xy[a][1]);
            (dx * dx + dy * dy).sqrt() / diag
        })
        .collect();
    let pa: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(a, _)| [f64::from(old.xy[a][0]), f64::from(old.xy[a][1])])
        .collect();
    let pb: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(_, b)| [f64::from(new.xy[b][0]), f64::from(new.xy[b][1])])
        .collect();
    let diag_new = pipe::diagonal(&new.xy);
    let aligned: Vec<f64> = pipe::procrustes_residuals(&pa, &pb)
        .into_iter()
        .map(|d| d / diag_new)
        .collect();
    let old_sample: std::collections::HashSet<MemoryId> =
        old.sample.iter().map(|&r| old.prep.ids[r]).collect();
    let new_sample: std::collections::HashSet<MemoryId> =
        new.sample.iter().map(|&r| new.prep.ids[r]).collect();
    json!({
        "regions_old": old.run.regions.len(),
        "regions_new": new.run.regions.len(),
        "same_partition": same_partition,
        "ari_old_new": adjusted_rand_index(&as_classes(&old_label), &new_label),
        "changed_share": 1.0 - share(kept, pairs.len()),
        "same_xy_bits": same_xy,
        "sample_left": old_sample.difference(&new_sample).count(),
        "displacement_over_diagonal": {
            "median": pipe::median(&raw), "p90": quantile(&raw, 0.9), "max": quantile(&raw, 1.0),
        },
        "after_alignment_over_diagonal": {
            "median": pipe::median(&aligned), "p90": quantile(&aligned, 0.9), "max": quantile(&aligned, 1.0),
        },
    })
}

pub fn alt7b() {
    let store = cases::open(&cases::db_path(BIG));
    let pid = cases::only_project(&store);
    let mem = cases::read_vectors(&store, pid);
    drop(store);
    let params = MapParams::default();
    let t0 = Instant::now();
    let b = base(&mem, &params);
    let base_ms = ms(t0.elapsed());
    let k = params.coords.neighbors.min(b.sample.len());

    // The state an incremental placement keeps: each region's centre from
    // its sample members, and the sample's positions.
    let sample_region: Vec<Option<usize>> = b.sample.iter().map(|&r| b.label[r]).collect();
    let mut centre_ms = Vec::new();
    let mut centres = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        centres = pipe::centres(&b.prep, &b.sample, &sample_region);
        centre_ms.push(ms(t.elapsed()));
    }

    let extras = synth::extra(&Structure::new(), BIG, 1_000, 1);
    let mut place_ms = Vec::new();
    let mut placed: Vec<(usize, [f32; 2])> = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        placed = extras
            .iter()
            .map(|e| {
                let v = sm_spatial::memory_vector(std::slice::from_ref(&e.vector))
                    .expect("a memory vector");
                (
                    pipe::nearest_centre(&v, &centres),
                    pipe::place_one(&v, &b.prep, &b.sample, &b.xy, k),
                )
            })
            .collect();
        place_ms.push(ms(t.elapsed()));
    }

    // Which new memories would enter the sample on a rebuild.
    let seed = params.project.seed;
    let last = b
        .sample
        .iter()
        .map(|&r| (sample_key(&b.prep.ids[r], seed), b.prep.ids[r]))
        .max()
        .expect("a sample");
    let enters: Vec<bool> = extras
        .iter()
        .map(|e| (sample_key(&e.id, seed), e.id) < last)
        .collect();
    let entering = enters.iter().filter(|&&x| x).count();

    // A rebuild with one memory that stays out of the sample, and one with
    // a memory that enters it.
    let outside = enters.iter().position(|&x| !x).expect("one outside");
    let t1 = Instant::now();
    let b_out = base(&with_extra(&mem, &extras[outside]), &params);
    let rebuild_out_ms = ms(t1.elapsed());
    let row = b_out.row_of()[&extras[outside].id];
    let same_new_xy = b_out.xy[row][0].to_bits() == placed[outside].1[0].to_bits()
        && b_out.xy[row][1].to_bits() == placed[outside].1[1].to_bits();
    // The rebuild's region for the new memory, through the old-to-new map.
    let sample_rows_new = b_out.row_of();
    let region_match = {
        let new_region = b_out.label[row];
        let old_region = Some(placed[outside].0);
        // A sample member of the old region, and its region in the rebuild.
        b.sample
            .iter()
            .find(|&&r| b.label[r] == old_region)
            .map(|&r| b_out.label[sample_rows_new[&b.prep.ids[r]]] == new_region)
    };
    let cmp_out = compare(&b, &b_out);
    drop(b_out);

    let inside = enters.iter().position(|&x| x);
    let (cmp_in, rebuild_in_ms) = match inside {
        Some(i) => {
            let t2 = Instant::now();
            let b_in = base(&with_extra(&mem, &extras[i]), &params);
            let e = ms(t2.elapsed());
            (compare(&b, &b_in), json!(e))
        }
        None => (Value::Null, Value::Null),
    };

    emit(&json!({
        "case": "alt7b", "n": BIG,
        "base_ms": base_ms,
        "centres_ms": centre_ms,
        "place_1000_ms": place_ms,
        "regions": centres.len(),
        "extras": extras.len(),
        "entering": entering,
        "expected_entering_share": params.max_points as f64 / (BIG + 1) as f64,
        "outside": {
            "rebuild_ms": rebuild_out_ms,
            "new_xy_equals_incremental": same_new_xy,
            "new_region_equals_incremental": region_match,
            "comparison": cmp_out,
        },
        "inside": {"rebuild_ms": rebuild_in_ms, "comparison": cmp_in},
    }));
}

// ---------------------------------------------------------------------------
// 7c. Clustering again inside each region
// ---------------------------------------------------------------------------

pub fn alt7c(rep: usize) {
    let store = cases::open(&cases::db_path(BIG));
    let pid = cases::only_project(&store);
    let mem = cases::read_vectors(&store, pid);
    drop(store);
    let params = MapParams::default();
    let (run, r_base) = measure("regions() at 100k", || {
        sm_spatial::regions(&mem, &params).expect("regions")
    });
    let index: HashMap<MemoryId, usize> = mem.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
    let (subs, r_all) = measure("regions() inside every region", || {
        run.regions
            .iter()
            .map(|reg| {
                let sub: Vec<MemoryVectors> = reg
                    .members
                    .iter()
                    .map(|m| mem[index[&m.id]].clone())
                    .collect();
                let t = Instant::now();
                let sr = sm_spatial::regions(&sub, &params).expect("sub-regions");
                (reg.region, reg.size(), ms(t.elapsed()), sr)
            })
            .collect::<Vec<_>>()
    });
    let per_ms: Vec<f64> = subs.iter().map(|s| s.2).collect();
    let mut out = json!({
        "case": "alt7c", "rep": rep, "n": mem.len(),
        "regions": run.regions.len(),
        "recs": cases::recs_json(&[r_base, r_all]),
        "per_region_ms_sum": per_ms.iter().sum::<f64>(),
        "per_region_ms_max": per_ms.iter().copied().fold(0.0, f64::max),
        "per_region_ms_median": pipe::median(&per_ms),
        "process_peak_ws": meter::now_mem().peak_working_set,
    });
    if rep == 0 {
        let mut ids: Vec<MemoryId> = mem.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let t = truth_for(&ids, BIG);
        let top: HashMap<MemoryId, Option<u32>> = run.assignments().into_iter().collect();
        let mut strict: HashMap<MemoryId, Option<(u32, u32)>> = HashMap::new();
        let mut fallback: HashMap<MemoryId, Option<(u32, u32)>> = HashMap::new();
        let mut unclustered_regions = 0;
        let mut sizes: Vec<usize> = Vec::new();
        for (region, _, _, sr) in &subs {
            if !sr.clustered {
                unclustered_regions += 1;
            }
            sizes.push(sr.regions.len());
            for (id, s) in sr.assignments() {
                strict.insert(id, s.map(|s| (*region, s)));
                fallback.insert(id, Some((*region, s.unwrap_or(u32::MAX))));
            }
        }
        let top_l: Vec<Option<u32>> = ids.iter().map(|id| top[id]).collect();
        let strict_l: Vec<Option<(u32, u32)>> = ids
            .iter()
            .map(|id| strict.get(id).copied().flatten())
            .collect();
        let fallback_l: Vec<Option<(u32, u32)>> = ids
            .iter()
            .map(|id| fallback.get(id).copied().flatten())
            .collect();
        let (td, sd, fd) = (
            pipe::dense(&top_l),
            pipe::dense(&strict_l),
            pipe::dense(&fallback_l),
        );
        let mut sub_regions = sd.iter().flatten().copied().collect::<Vec<_>>();
        sub_regions.sort_unstable();
        sub_regions.dedup();
        let members: Vec<usize> = (0..ids.len())
            .filter(|&r| t.topic[r] < synth::TOPICS)
            .collect();
        let ms_: Vec<usize> = members.iter().map(|&r| t.sub[r]).collect();
        let sp: Vec<Option<usize>> = members.iter().map(|&r| sd[r]).collect();
        out["quality"] = json!({
            "sub_regions": sub_regions.len(),
            "regions_not_split": unclustered_regions,
            "sub_regions_per_region": sizes,
            "ari_subs_top_level": adjusted_rand_index(&t.sub, &td),
            "ari_subs_strict": adjusted_rand_index(&t.sub, &sd),
            "ari_subs_fallback": adjusted_rand_index(&t.sub, &fd),
            "ari_topics_top_level": adjusted_rand_index(&t.topic, &td),
            "ari_topics_strict": adjusted_rand_index(&t.topic, &sd),
            "majority_subs_strict": pipe::majority_classes(&ms_, &sp),
            "unclustered_strict": share(sd.iter().filter(|l| l.is_none()).count(), ids.len()),
            "unclustered_top_level": share(td.iter().filter(|l| l.is_none()).count(), ids.len()),
        });
    }
    emit(&out);
}

// ---------------------------------------------------------------------------
// The real store (a read-only copy)
// ---------------------------------------------------------------------------

struct Meta {
    project: ProjectId,
    status: &'static str,
    created: i64,
    updated: i64,
    bytes: usize,
}

/// Write activity: per day over the last 30 days, the hours with any write
/// over the last 14, and the median gap between writes, from creation and
/// update times.
fn activity(metas: &[&Meta], now: i64) -> Value {
    let mut events: Vec<i64> = Vec::new();
    for m in metas {
        events.push(m.created);
        if m.updated != m.created {
            events.push(m.updated);
        }
    }
    events.sort_unstable();
    let since30 = now - 30 * MICROS_PER_DAY;
    let since14 = now - 14 * MICROS_PER_DAY;
    let mut per_day = vec![0usize; 30];
    for &e in events.iter().filter(|&&e| e >= since30 && e <= now) {
        let d = ((e - since30) / MICROS_PER_DAY) as usize;
        per_day[d.min(29)] += 1;
    }
    let mut hours: Vec<i64> = events
        .iter()
        .filter(|&&e| e >= since14 && e <= now)
        .map(|e| e / 3_600_000_000)
        .collect();
    hours.dedup();
    let mut tens: Vec<i64> = events
        .iter()
        .filter(|&&e| e >= since14 && e <= now)
        .map(|e| e / 600_000_000)
        .collect();
    tens.dedup();
    let recent: Vec<i64> = events.iter().copied().filter(|&e| e >= since30).collect();
    let gaps: Vec<f64> = recent
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / 60_000_000.0)
        .collect();
    let active_days: Vec<usize> = per_day.iter().copied().filter(|&c| c > 0).collect();
    json!({
        "events_30d": recent.len(),
        "per_day_30d": per_day,
        "days_with_writes_30d": active_days.len(),
        "median_per_active_day": pipe::median(&active_days.iter().map(|&c| c as f64).collect::<Vec<_>>()),
        "hours_with_writes_14d": hours.len(),
        "ten_minute_windows_with_writes_14d": tens.len(),
        "median_gap_minutes_30d": pipe::median(&gaps),
    })
}

fn map_case(mem: &[MemoryVectors], params: &MapParams, label: &str) -> Value {
    let b = base(mem, params);
    let unclustered = b.label.iter().filter(|l| l.is_none()).count();
    json!({
        "params": label,
        "memories": b.prep.ids.len(),
        "sample": b.sample.len(),
        "regions": b.run.regions.len(),
        "unclustered_share": share(unclustered, b.prep.ids.len()),
        "largest_share": share(b.run.regions.first().map_or(0, |r| r.size()), b.prep.ids.len()),
        "neighbours": neighbours(&b.prep, &b.xy, &b.sample, 0x0261_0002),
        "clumping": clumping(&b.xy, &b.sample),
    })
}

pub fn real(db: &Path, backups: &Path) {
    let store = Store::open_at(db, backups).expect("open the copy");
    let stamp = store
        .stored_stamp()
        .expect("the stamp")
        .expect("a stamped index");
    let status = store.configure_index(stamp).expect("configure the index");
    assert!(status.matches, "the copy's own stamp matches itself");
    let now = Timestamp::now().as_micros();

    let (metas, r_read) = measure("read every memory (metadata)", || {
        store
            .iter_memories()
            .map(|m| {
                let m = m.expect("a memory");
                Meta {
                    project: m.project_id,
                    status: m.status.as_str(),
                    created: m.created_at.as_micros(),
                    updated: m.updated_at.as_micros(),
                    bytes: m.content.len(),
                }
            })
            .collect::<Vec<Meta>>()
    });
    let mut by_status: HashMap<&'static str, usize> = HashMap::new();
    for m in &metas {
        *by_status.entry(m.status).or_default() += 1;
    }
    let (vectors, r_vec) = measure("memory_vectors (all projects)", || {
        store
            .memory_vectors(QueryScope::All, &MemoryFilter::default())
            .expect("the vectors")
    });
    let with_vector: std::collections::HashSet<MemoryId> =
        vectors.iter().map(|(id, _)| *id).collect();
    let active_ids: Vec<MemoryId> = store
        .iter_memories()
        .filter_map(|m| m.ok())
        .filter(|m| m.status.as_str() == "active")
        .map(|m| m.id)
        .collect();
    let active_without_vector = active_ids
        .iter()
        .filter(|id| !with_vector.contains(id))
        .count();

    // Projects by their active memories with a vector, largest first.
    let project_of: HashMap<MemoryId, ProjectId> = store
        .iter_memories()
        .filter_map(|m| m.ok())
        .map(|m| (m.id, m.project_id))
        .collect();
    let mut per_project: HashMap<ProjectId, usize> = HashMap::new();
    for (id, _) in &vectors {
        if let Some(p) = project_of.get(id) {
            *per_project.entry(*p).or_default() += 1;
        }
    }
    let mut sizes: Vec<(usize, ProjectId)> = per_project.iter().map(|(p, c)| (*c, *p)).collect();
    sizes.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let largest = sizes.first().map(|s| s.1).expect("a project");

    let active_bytes: Vec<f64> = metas
        .iter()
        .filter(|m| m.status == "active")
        .map(|m| m.bytes as f64)
        .collect();
    let all_bytes: Vec<f64> = metas.iter().map(|m| m.bytes as f64).collect();

    let pooled: Vec<MemoryVectors> = vectors
        .iter()
        .map(|(id, v)| MemoryVectors {
            id: *id,
            chunks: vec![v.clone()],
        })
        .collect();
    drop(vectors);
    let mem = cases::read_vectors(&store, largest);
    let params = MapParams::default();
    let mut timings = Vec::new();
    for _ in 0..3 {
        let (run, r1) = measure("regions()", || {
            sm_spatial::regions(&mem, &params).expect("regions")
        });
        let (_, r2) = measure("coordinates()", || {
            sm_spatial::coordinates(&mem, &params, &run).expect("coordinates")
        });
        timings.push(cases::recs_json(&[r1, r2]));
    }
    let largest_case = map_case(&mem, &params, "product (max_points 2000)");
    let small = MapParams {
        max_points: 500,
        ..MapParams::default()
    };
    let small_case = if mem.len() > small.max_points {
        map_case(&mem, &small, "max_points 500 (not the product's)")
    } else {
        Value::Null
    };
    let pooled_case = map_case(&pooled, &params, "product, every project pooled");

    let largest_metas: Vec<&Meta> = metas.iter().filter(|m| m.project == largest).collect();
    let all_metas: Vec<&Meta> = metas.iter().collect();
    let largest_active_bytes: Vec<f64> = largest_metas
        .iter()
        .filter(|m| m.status == "active")
        .map(|m| m.bytes as f64)
        .collect();
    emit(&json!({
        "case": "real",
        "memories": metas.len(),
        "by_status": by_status,
        "active_with_vector": with_vector.len(),
        "active_without_vector": active_without_vector,
        "projects": sizes.len(),
        "project_sizes": sizes.iter().map(|s| s.0).collect::<Vec<_>>(),
        "content_bytes_active": {"mean": pipe::mean(&active_bytes), "median": pipe::median(&active_bytes), "count": active_bytes.len()},
        "content_bytes_all": {"mean": pipe::mean(&all_bytes), "median": pipe::median(&all_bytes)},
        "content_bytes_largest_active": {"mean": pipe::mean(&largest_active_bytes), "median": pipe::median(&largest_active_bytes)},
        "reads": cases::recs_json(&[r_read, r_vec]),
        "largest_timings": timings,
        "largest": largest_case,
        "largest_small_sample": small_case,
        "pooled": pooled_case,
        "activity_largest": activity(&largest_metas, now),
        "activity_all": activity(&all_metas, now),
        // Pending memories are not in the map (B-157); these count the
        // memories that are.
        "activity_largest_active": activity(&largest_metas.iter().copied().filter(|m| m.status == "active").collect::<Vec<_>>(), now),
        "activity_all_active": activity(&all_metas.iter().copied().filter(|m| m.status == "active").collect::<Vec<_>>(), now),
    }));
}
