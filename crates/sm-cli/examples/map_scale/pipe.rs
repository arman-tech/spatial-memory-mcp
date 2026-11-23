//! A stage-by-stage replica of the SPEC 7 pipeline in
//! `crates/sm-spatial/src/pipeline.rs`, so each stage can be timed on its
//! own, and the analysis helpers (nearest neighbours, clumping, agreement).
//!
//! The replica copies the library's code; `cost-stages` checks it bit for
//! bit against `sm_spatial::regions` and `sm_spatial::coordinates` on its
//! first run. `dot`, `unit` and `centre` are `pub(crate)` in `sm-spatial`
//! and are copied here unchanged. Above 2,000 rows (alternative (a)) the
//! projection calls `holomap` directly with the builder of
//! `HolomapProjector`, which refuses more than 2,000.

use std::cmp::Ordering;
use std::collections::HashMap;

use holomap::{Holomap, Init, Metric};
use sm_core::ids::MemoryId;
use sm_core::traits::{ClusterParams, Clusterer, MAX_PROJECTION_POINTS, ProjectParams, Projector};
use sm_spatial::{
    HdbscanClusterer, HolomapProjector, MemoryVectors, RankedMember, RegionResult, cosine,
    memory_vector, sample_order,
};

/// The memories in identifier order with one unit vector each.
pub struct Prepared {
    pub ids: Vec<MemoryId>,
    pub vectors: Vec<Vec<f32>>,
    pub dims: usize,
}

pub fn prepare(memories: &[MemoryVectors]) -> Prepared {
    let mut order: Vec<usize> = (0..memories.len()).collect();
    order.sort_by_key(|&i| memories[i].id);
    let mut ids = Vec::with_capacity(memories.len());
    let mut vectors = Vec::with_capacity(memories.len());
    let mut dims = 0;
    for i in order {
        let m = &memories[i];
        assert!(ids.last() != Some(&m.id), "a duplicated identifier");
        let v = memory_vector(&m.chunks).expect("a memory vector");
        if vectors.is_empty() {
            dims = v.len();
        }
        assert_eq!(v.len(), dims, "one dimension count");
        ids.push(m.id);
        vectors.push(v);
    }
    Prepared { ids, vectors, dims }
}

/// The first `size` rows of [`sample_order`], sorted; every row when there
/// are no more than `size`.
pub fn sample_rows(ids: &[MemoryId], size: usize, seed: u64) -> Vec<usize> {
    if ids.len() <= size {
        return (0..ids.len()).collect();
    }
    let mut rows = sample_order(ids, seed);
    rows.truncate(size);
    rows.sort_unstable();
    rows
}

/// The product's projector up to 2,000 rows; above it, `holomap` called
/// with the same builder.
pub fn project(vectors: &[Vec<f32>], p: &ProjectParams) -> Vec<Vec<f32>> {
    if vectors.len() <= MAX_PROJECTION_POINTS {
        return HolomapProjector
            .project(vectors, p)
            .expect("the projection");
    }
    let dims = vectors[0].len();
    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    let out = Holomap::builder(p.seed)
        .n_components(p.dims)
        .n_neighbors(p.neighbors)
        .min_dist(p.min_dist)
        .metric(Metric::Cosine)
        .init(Init::Spectral)
        .fit_transform(&flat, dims)
        .expect("the projection");
    assert!(out.len() == vectors.len() * p.dims && out.iter().all(|x| x.is_finite()));
    out.chunks_exact(p.dims).map(<[f32]>::to_vec).collect()
}

/// HDBSCAN with `ClusterParams::for_n` of the number of points.
pub fn cluster(points: &[Vec<f32>]) -> (Vec<Option<usize>>, ClusterParams) {
    let params = ClusterParams::for_n(points.len());
    let labels = HdbscanClusterer
        .cluster(points, &params)
        .expect("the clustering");
    (labels, params)
}

/// The centre of each cluster's sample members (`assign_rest`'s centres).
pub fn centres(prep: &Prepared, sample: &[usize], labels: &[Option<usize>]) -> Vec<Vec<f32>> {
    let k = labels.iter().flatten().max().map_or(0, |&c| c + 1);
    let mut rows_by_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (&r, &l) in sample.iter().zip(labels) {
        if let Some(c) = l {
            rows_by_cluster[c].push(r);
        }
    }
    rows_by_cluster
        .iter()
        .map(|rows| centre(&prep.vectors, rows, prep.dims))
        .collect()
}

/// The nearest centre by cosine; ties go to the lower cluster.
pub fn nearest_centre(v: &[f32], centres: &[Vec<f32>]) -> usize {
    let mut best = (f32::NEG_INFINITY, 0usize);
    for (c, centre) in centres.iter().enumerate() {
        let s = cosine(v, centre);
        if s > best.0 {
            best = (s, c);
        }
    }
    best.1
}

/// The labels of every row: the sample's from the clusterer, the rest from
/// the nearest centre (none when there are no clusters).
pub fn assign(
    prep: &Prepared,
    sample: &[usize],
    labels: &[Option<usize>],
    centres: &[Vec<f32>],
) -> Vec<Option<usize>> {
    let n = prep.ids.len();
    let mut label: Vec<Option<usize>> = vec![None; n];
    let mut in_sample = vec![false; n];
    for (&r, &l) in sample.iter().zip(labels) {
        label[r] = l;
        in_sample[r] = true;
    }
    if centres.is_empty() {
        return label;
    }
    for r in (0..n).filter(|&r| !in_sample[r]) {
        label[r] = Some(nearest_centre(&prep.vectors[r], centres));
    }
    label
}

/// `build_regions`: groups by label, largest first (ties by smallest id),
/// members ranked by centrality.
pub fn build_regions(prep: &Prepared, label: &[Option<usize>]) -> Vec<RegionResult> {
    let k = label.iter().flatten().max().map_or(0, |&c| c + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (r, l) in label.iter().enumerate() {
        if let Some(c) = l {
            groups[*c].push(r);
        }
    }
    groups.retain(|g| !g.is_empty());
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

fn nearest_first(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// Positions of every row: the sample's from `sample_xy`, every other row
/// at the mean of its `k` nearest sample rows by dot product (ties by row).
pub fn place(prep: &Prepared, sample: &[usize], sample_xy: &[[f32; 2]], k: usize) -> Vec<[f32; 2]> {
    let n = prep.ids.len();
    let mut xy = vec![[0.0_f32; 2]; n];
    let mut in_sample = vec![false; n];
    for (&r, p) in sample.iter().zip(sample_xy) {
        xy[r] = *p;
        in_sample[r] = true;
    }
    if sample.len() < n {
        for r in (0..n).filter(|&r| !in_sample[r]) {
            xy[r] = place_one(&prep.vectors[r], prep, sample, &xy, k);
        }
    }
    xy
}

/// One vector's position: the mean of its `k` nearest sample rows' `xy`.
pub fn place_one(
    v: &[f32],
    prep: &Prepared,
    sample: &[usize],
    xy: &[[f32; 2]],
    k: usize,
) -> [f32; 2] {
    let mut near: Vec<(f32, usize)> = sample
        .iter()
        .map(|&s| (dot(v, &prep.vectors[s]), s))
        .collect();
    near.select_nth_unstable_by(k - 1, nearest_first);
    near[..k].sort_by(nearest_first);
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    for &(_, s) in &near[..k] {
        x += f64::from(xy[s][0]);
        y += f64::from(xy[s][1]);
    }
    [(x / k as f64) as f32, (y / k as f64) as f32]
}

// ---------------------------------------------------------------------------
// Copies of sm-spatial's pub(crate) vector helpers (vector.rs), unchanged.
// ---------------------------------------------------------------------------

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 8];
    let (mut ca, mut cb) = (a.chunks_exact(8), b.chunks_exact(8));
    for (x, y) in (&mut ca).zip(&mut cb) {
        for ((s, &p), &q) in lanes.iter_mut().zip(x).zip(y) {
            *s += p * q;
        }
    }
    let mut tail = 0.0_f32;
    for (&p, &q) in ca.remainder().iter().zip(cb.remainder()) {
        tail += p * q;
    }
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
        + tail
}

pub fn unit(v: &[f64]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| (x / norm) as f32).collect())
}

pub fn centre(vectors: &[Vec<f32>], rows: &[usize], dims: usize) -> Vec<f32> {
    let mut sum = vec![0.0_f64; dims];
    for &r in rows {
        for (s, &x) in sum.iter_mut().zip(&vectors[r]) {
            *s += f64::from(x);
        }
    }
    unit(&sum).unwrap_or_else(|| vec![0.0; dims])
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Runs `f` over `items` on every core; results in input order.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .max(1);
    let per = items.len().div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|chunk| s.spawn(move || chunk.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a worker"))
            .collect()
    })
}

/// The `k` rows most similar to row `q` by dot product (unit vectors, so
/// cosine), itself excluded, most similar first, ties by row; with their
/// similarities.
pub fn top_cos(vectors: &[Vec<f32>], q: usize, k: usize) -> Vec<(f32, usize)> {
    let mut all: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .filter(|&(r, _)| r != q)
        .map(|(r, v)| (dot(&vectors[q], v), r))
        .collect();
    let k = k.min(all.len());
    if k == 0 {
        return Vec::new();
    }
    all.select_nth_unstable_by(k - 1, nearest_first);
    all.truncate(k);
    all.sort_by(nearest_first);
    all
}

/// The `k` rows nearest row `q` on the map (squared Euclidean in f64),
/// itself excluded, ties by row.
pub fn top_map(xy: &[[f32; 2]], q: usize, k: usize) -> Vec<usize> {
    let (qx, qy) = (f64::from(xy[q][0]), f64::from(xy[q][1]));
    let mut all: Vec<(f64, usize)> = xy
        .iter()
        .enumerate()
        .filter(|&(r, _)| r != q)
        .map(|(r, p)| {
            let dx = f64::from(p[0]) - qx;
            let dy = f64::from(p[1]) - qy;
            (dx * dx + dy * dy, r)
        })
        .collect();
    let k = k.min(all.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    all.select_nth_unstable_by(k - 1, cmp);
    all.truncate(k);
    all.into_iter().map(|(_, r)| r).collect()
}

/// For each query row: the share of its 10 nearest by cosine that are
/// among its 10 nearest on the map, the cosine of its nearest and its
/// 10th nearest.
///
/// `share` breaks ties on the map by row. Placed memories can share one
/// position exactly, so `share_ties` also counts a cosine neighbour whose
/// map distance equals the 10th nearest's, and `tie_set` is how many rows
/// lie within that distance (10 when there is no tie).
pub struct Overlap {
    pub row: usize,
    pub share: f64,
    pub share_ties: f64,
    pub tie_set: usize,
    pub cos1: f32,
    pub cos10: f32,
}

pub fn overlaps(vectors: &[Vec<f32>], xy: &[[f32; 2]], queries: &[usize]) -> Vec<Overlap> {
    par_map(queries, |&q| {
        let cos = top_cos(vectors, q, 10);
        let map = top_map(xy, q, 10);
        let hits = cos.iter().filter(|(_, r)| map.contains(r)).count();
        let (qx, qy) = (f64::from(xy[q][0]), f64::from(xy[q][1]));
        let dist = |r: usize| -> f64 {
            let dx = f64::from(xy[r][0]) - qx;
            let dy = f64::from(xy[r][1]) - qy;
            dx * dx + dy * dy
        };
        let d10 = map.iter().map(|&r| dist(r)).fold(f64::MIN, f64::max);
        let tie_set = (0..xy.len()).filter(|&r| r != q && dist(r) <= d10).count();
        let hits_ties = cos.iter().filter(|(_, r)| dist(*r) <= d10).count();
        Overlap {
            row: q,
            share: hits as f64 / cos.len().max(1) as f64,
            share_ties: hits_ties as f64 / cos.len().max(1) as f64,
            tie_set,
            cos1: cos.first().map_or(f32::NAN, |c| c.0),
            cos10: cos.last().map_or(f32::NAN, |c| c.0),
        }
    })
}

/// The distance from each of `queries` to its nearest other point among
/// `among` (both rows of `xy`), by a uniform grid.
pub fn nearest_dist(xy: &[[f32; 2]], among: &[usize], queries: &[usize]) -> Vec<f64> {
    if among.len() < 2 {
        return vec![f64::NAN; queries.len()];
    }
    let pts: Vec<(f64, f64)> = xy
        .iter()
        .map(|p| (f64::from(p[0]), f64::from(p[1])))
        .collect();
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &r in among {
        let (x, y) = pts[r];
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let area = ((x1 - x0) * (y1 - y0)).max(1e-12);
    let cell = (area / among.len() as f64).sqrt().max(1e-9);
    let key = |x: f64, y: f64| -> (i64, i64) {
        (
            ((x - x0) / cell).floor() as i64,
            ((y - y0) / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for &r in among {
        grid.entry(key(pts[r].0, pts[r].1)).or_default().push(r);
    }
    let grid = &grid;
    let pts = &pts;
    par_map(queries, |&q| {
        let (qx, qy) = pts[q];
        let (cx, cy) = key(qx, qy);
        let mut best = f64::INFINITY;
        let consider = |r: usize, best: &mut f64| {
            if r != q {
                let (x, y) = pts[r];
                let d = ((x - qx).powi(2) + (y - qy).powi(2)).sqrt();
                if d < *best {
                    *best = d;
                }
            }
        };
        let mut ring = 0_i64;
        loop {
            if ring > 64 {
                for &r in among {
                    consider(r, &mut best);
                }
                break;
            }
            for gx in (cx - ring)..=(cx + ring) {
                for gy in (cy - ring)..=(cy + ring) {
                    if (gx - cx).abs() != ring && (gy - cy).abs() != ring {
                        continue;
                    }
                    if let Some(rows) = grid.get(&(gx, gy)) {
                        for &r in rows {
                            consider(r, &mut best);
                        }
                    }
                }
            }
            // Anything outside rings 0..=ring is at least ring * cell away.
            if best <= ring as f64 * cell {
                break;
            }
            ring += 1;
        }
        best
    })
}

pub fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub fn mean(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Clumping of the placed rows: distinct positions, and the shares of
/// placed and of sampled rows whose nearest other memory (any) is closer
/// than the median distance from a sampled row to its nearest other
/// sampled row; and the shares at distance 0.
pub struct Clumping {
    pub placed: usize,
    pub distinct_placed: usize,
    pub sample_median_nn: f64,
    pub placed_below: f64,
    pub sampled_below: f64,
    pub placed_zero: f64,
    pub sampled_zero: f64,
    pub placed_median_nn: f64,
}

pub fn clumping(xy: &[[f32; 2]], sample: &[usize]) -> Clumping {
    let n = xy.len();
    let mut in_sample = vec![false; n];
    for &r in sample {
        in_sample[r] = true;
    }
    let placed: Vec<usize> = (0..n).filter(|&r| !in_sample[r]).collect();
    let all: Vec<usize> = (0..n).collect();
    let mut distinct: Vec<(u32, u32)> = placed
        .iter()
        .map(|&r| (xy[r][0].to_bits(), xy[r][1].to_bits()))
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    let sample_nn = nearest_dist(xy, sample, sample);
    let threshold = median(&sample_nn);
    let placed_nn = nearest_dist(xy, &all, &placed);
    let sampled_nn = nearest_dist(xy, &all, sample);
    let share = |v: &[f64], f: &dyn Fn(f64) -> bool| -> f64 {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().filter(|&&d| f(d)).count() as f64 / v.len() as f64
        }
    };
    Clumping {
        placed: placed.len(),
        distinct_placed: distinct.len(),
        sample_median_nn: threshold,
        placed_below: share(&placed_nn, &|d| d < threshold),
        sampled_below: share(&sampled_nn, &|d| d < threshold),
        placed_zero: share(&placed_nn, &|d| d == 0.0),
        sampled_zero: share(&sampled_nn, &|d| d == 0.0),
        placed_median_nn: median(&placed_nn),
    }
}

/// Labels as dense `usize` classes, `None` kept.
pub fn dense<T: Ord + Clone>(labels: &[Option<T>]) -> Vec<Option<usize>> {
    let mut keys: Vec<T> = labels.iter().flatten().cloned().collect();
    keys.sort();
    keys.dedup();
    labels
        .iter()
        .map(|l| l.as_ref().map(|k| keys.binary_search(k).unwrap_or(0)))
        .collect()
}

/// How many truth classes are the majority of at least one predicted group.
pub fn majority_classes(truth: &[usize], predicted: &[Option<usize>]) -> usize {
    let mut by_group: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (t, p) in truth.iter().zip(predicted) {
        if let Some(g) = p {
            *by_group.entry(*g).or_default().entry(*t).or_default() += 1;
        }
    }
    let mut winners: Vec<usize> = by_group
        .values()
        .filter_map(|counts| {
            counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(t, _)| *t)
        })
        .collect();
    winners.sort_unstable();
    winners.dedup();
    winners.len()
}

/// A 2-D similarity transform (rotation or reflection, scale, shift) that
/// best maps `a` onto `b` (least squares); returns the residual distances.
pub fn procrustes_residuals(a: &[[f64; 2]], b: &[[f64; 2]]) -> Vec<f64> {
    let n = a.len().max(1) as f64;
    let mean = |p: &[[f64; 2]]| -> [f64; 2] {
        let s = p
            .iter()
            .fold([0.0, 0.0], |acc, q| [acc[0] + q[0], acc[1] + q[1]]);
        [s[0] / n, s[1] / n]
    };
    let (ma, mb) = (mean(a), mean(b));
    let ac: Vec<[f64; 2]> = a.iter().map(|p| [p[0] - ma[0], p[1] - ma[1]]).collect();
    let bc: Vec<[f64; 2]> = b.iter().map(|p| [p[0] - mb[0], p[1] - mb[1]]).collect();
    let norm_a: f64 = ac.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum();
    [false, true]
        .into_iter()
        .map(|flip| {
            let src: Vec<[f64; 2]> = ac
                .iter()
                .map(|p| if flip { [p[0], -p[1]] } else { *p })
                .collect();
            let (mut c, mut s) = (0.0, 0.0);
            for (p, q) in src.iter().zip(&bc) {
                c += p[0] * q[0] + p[1] * q[1];
                s += p[0] * q[1] - p[1] * q[0];
            }
            let theta = s.atan2(c);
            let scale = if norm_a > 0.0 {
                (c * c + s * s).sqrt() / norm_a
            } else {
                1.0
            };
            let (ct, st) = (theta.cos(), theta.sin());
            let res: Vec<f64> = src
                .iter()
                .zip(&bc)
                .map(|(p, q)| {
                    let x = scale * (ct * p[0] - st * p[1]);
                    let y = scale * (st * p[0] + ct * p[1]);
                    ((x - q[0]).powi(2) + (y - q[1]).powi(2)).sqrt()
                })
                .collect();
            (res.iter().sum::<f64>(), res)
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .map(|(_, r)| r)
        .unwrap_or_default()
}

/// The diagonal of the bounding box of `xy`.
pub fn diagonal(xy: &[[f32; 2]]) -> f64 {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in xy {
        let (x, y) = (f64::from(p[0]), f64::from(p[1]));
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}
