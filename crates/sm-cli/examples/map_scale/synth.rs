//! The synthetic data set (B-261): unit vectors in 768 dimensions, grouped
//! into 40 topics of 5 sub-topics each, with unequal sizes and 5 % noise
//! points. Fixed before any result was seen; nothing here was tuned.
//!
//! A member of sub-topic `s` of topic `t` is
//!
//! ```text
//! unit( sqrt(0.25) g + sqrt(0.25) T[t] + sqrt(0.15) S[t][s]
//!       + sqrt(0.20) L + sqrt(0.15) E )
//! ```
//!
//! where `g`, `T[t]` and `S[t][s]` are fixed random unit directions (one
//! global, 40 topic, 200 sub-topic), `L = sum_j z_j B[t][s][j] / sqrt(8)`
//! with `z_j ~ N(0, 1)` over 8 fixed random directions of that sub-topic
//! (variation inside a sub-topic, so a memory's nearest neighbours are not
//! all equally near), and `E` is a fresh random unit vector. Random unit
//! vectors in 768 dimensions are close to orthogonal, so the expected
//! cosines are about 0.65 inside a sub-topic, 0.50 inside a topic and 0.25
//! between topics. A noise point is `unit(sqrt(0.25) g + sqrt(0.75) E)`.
//!
//! Sizes: topic `t` (0-based) has weight `(t + 1)^-0.8`, sub-topic `s`
//! weight `(s + 1)^-0.8`; the 200 cells share `0.95 n` by largest
//! remainder. Creation times are uniform over 365 days from 2025-11-23,
//! types uniform over the six, every memory active.

use sm_core::ids::MemoryId;
use sm_core::types::MemoryType;

pub const DIMS: usize = 768;
pub const TOPICS: usize = 40;
pub const SUBS: usize = 5;
pub const CELLS: usize = TOPICS * SUBS;
pub const LATENT: usize = 8;
pub const NOISE_SHARE: f64 = 0.05;
pub const SIZE_EXPONENT: f64 = 0.8;
pub const W_GLOBAL: f64 = 0.25;
pub const W_TOPIC: f64 = 0.25;
pub const W_SUB: f64 = 0.15;
pub const W_LATENT: f64 = 0.20;
pub const W_NOISE: f64 = 0.15;
const STRUCTURE_SEED: u64 = 0x5EED_0261_0000_0001;
/// 2025-11-23T00:00:00Z, the first creation day.
pub const DAY0_SECS: i64 = 1_763_856_000;
pub const DAYS: i64 = 365;
const MICROS_PER_DAY: i64 = 86_400_000_000;

pub const TYPES: [MemoryType; 6] = [
    MemoryType::Decision,
    MemoryType::Pitfall,
    MemoryType::Convention,
    MemoryType::Incident,
    MemoryType::Runbook,
    MemoryType::Note,
];

/// SplitMix64 with Box-Muller normals.
pub struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng {
            state: seed,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        self.spare = Some(r * a.sin());
        r * a.cos()
    }

    /// `k` distinct values of `0..n`, in draw order (partial Fisher-Yates).
    pub fn distinct(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + self.below(n - i);
            v.swap(i, j);
        }
        v.truncate(k);
        v
    }
}

fn unit_gaussian(rng: &mut Rng) -> Vec<f64> {
    let v: Vec<f64> = (0..DIMS).map(|_| rng.normal()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// The fixed directions, the same for every size.
pub struct Structure {
    global: Vec<f64>,
    topic: Vec<Vec<f64>>,
    sub: Vec<Vec<f64>>,
    latent: Vec<Vec<f64>>,
    /// The share of members in each cell.
    cell_weight: Vec<f64>,
}

impl Structure {
    pub fn new() -> Structure {
        let mut rng = Rng::new(STRUCTURE_SEED);
        let global = unit_gaussian(&mut rng);
        let topic = (0..TOPICS).map(|_| unit_gaussian(&mut rng)).collect();
        let sub = (0..CELLS).map(|_| unit_gaussian(&mut rng)).collect();
        let latent = (0..CELLS * LATENT)
            .map(|_| unit_gaussian(&mut rng))
            .collect();
        let tw: Vec<f64> = (0..TOPICS)
            .map(|t| ((t + 1) as f64).powf(-SIZE_EXPONENT))
            .collect();
        let sw: Vec<f64> = (0..SUBS)
            .map(|s| ((s + 1) as f64).powf(-SIZE_EXPONENT))
            .collect();
        let mut cell_weight: Vec<f64> = (0..CELLS).map(|c| tw[c / SUBS] * sw[c % SUBS]).collect();
        let total: f64 = cell_weight.iter().sum();
        for w in &mut cell_weight {
            *w /= total;
        }
        Structure {
            global,
            topic,
            sub,
            latent,
            cell_weight,
        }
    }

    /// Members per cell for `n` memories (largest remainder), and the noise
    /// count.
    pub fn counts(&self, n: usize) -> (Vec<usize>, usize) {
        let noise = (n as f64 * NOISE_SHARE).round() as usize;
        let members = n - noise;
        let exact: Vec<f64> = self
            .cell_weight
            .iter()
            .map(|w| w * members as f64)
            .collect();
        let mut counts: Vec<usize> = exact.iter().map(|x| x.floor() as usize).collect();
        let mut left = members - counts.iter().sum::<usize>();
        let mut order: Vec<usize> = (0..CELLS).collect();
        order.sort_by(|&a, &b| {
            let ra = exact[a] - exact[a].floor();
            let rb = exact[b] - exact[b].floor();
            rb.total_cmp(&ra).then(a.cmp(&b))
        });
        for c in order {
            if left == 0 {
                break;
            }
            counts[c] += 1;
            left -= 1;
        }
        (counts, noise)
    }

    fn member(&self, rng: &mut Rng, cell: usize) -> Vec<f32> {
        let t = cell / SUBS;
        let mut v: Vec<f64> = vec![0.0; DIMS];
        let add = |v: &mut [f64], d: &[f64], w: f64| {
            for (x, y) in v.iter_mut().zip(d) {
                *x += w * y;
            }
        };
        add(&mut v, &self.global, W_GLOBAL.sqrt());
        add(&mut v, &self.topic[t], W_TOPIC.sqrt());
        add(&mut v, &self.sub[cell], W_SUB.sqrt());
        let scale = W_LATENT.sqrt() / (LATENT as f64).sqrt();
        for j in 0..LATENT {
            let z = rng.normal();
            add(&mut v, &self.latent[cell * LATENT + j], scale * z);
        }
        add(&mut v, &unit_gaussian(rng), W_NOISE.sqrt());
        to_unit(&v)
    }

    fn noise(&self, rng: &mut Rng) -> Vec<f32> {
        let mut v: Vec<f64> = self.global.iter().map(|x| x * W_GLOBAL.sqrt()).collect();
        for (x, e) in v.iter_mut().zip(unit_gaussian(rng)) {
            *x += e * (1.0 - W_GLOBAL).sqrt();
        }
        to_unit(&v)
    }

    /// One memory drawn as the data set's memories are (a cell by weight,
    /// or noise with probability 0.05), created at `created`.
    fn draw(&self, rng: &mut Rng, created: i64) -> Item {
        let cell = if rng.uniform() < NOISE_SHARE {
            None
        } else {
            let u = rng.uniform();
            let mut acc = 0.0;
            let mut pick = CELLS - 1;
            for (c, w) in self.cell_weight.iter().enumerate() {
                acc += w;
                if u < acc {
                    pick = c;
                    break;
                }
            }
            Some(pick)
        };
        self.item(rng, cell, created)
    }

    fn item(&self, rng: &mut Rng, cell: Option<usize>, created: i64) -> Item {
        let id = v7_like(rng, created);
        let kind = rng.below(TYPES.len());
        let vector = match cell {
            Some(c) => self.member(rng, c),
            None => self.noise(rng),
        };
        Item {
            id,
            cell,
            created_micros: created,
            kind,
            vector,
        }
    }
}

fn to_unit(v: &[f64]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| (x / norm) as f32).collect()
}

/// A UUID version 7 layout: the creation millisecond, then random bits.
fn v7_like(rng: &mut Rng, created_micros: i64) -> MemoryId {
    let ms = (created_micros / 1000) as u64;
    let a = rng.next_u64();
    let b = rng.next_u64();
    let mut bytes = [0u8; 16];
    bytes[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    bytes[6] = 0x70 | ((a >> 8) as u8 & 0x0F);
    bytes[7] = a as u8;
    bytes[8] = 0x80 | ((b >> 56) as u8 & 0x3F);
    bytes[9..].copy_from_slice(&b.to_be_bytes()[1..]);
    let hex: String = bytes.iter().map(|x| format!("{x:02x}")).collect();
    let text = format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    MemoryId::parse(&text).expect("a well-formed UUID")
}

/// One synthetic memory.
pub struct Item {
    pub id: MemoryId,
    /// `topic * 5 + sub-topic`, or `None` for a noise point.
    pub cell: Option<usize>,
    pub created_micros: i64,
    /// Index into [`TYPES`].
    pub kind: usize,
    pub vector: Vec<f32>,
}

impl Item {
    /// The topic label, noise as label 40.
    pub fn topic_label(&self) -> usize {
        self.cell.map_or(TOPICS, |c| c / SUBS)
    }

    /// The sub-topic label, noise as label 200.
    pub fn sub_label(&self) -> usize {
        self.cell.unwrap_or(CELLS)
    }

    pub fn content(&self, n: usize, i: usize) -> String {
        format!("synthetic map-scale memory {i} of {n}")
    }
}

fn seed_for(n: usize, stream: u64) -> u64 {
    0xD47A_5E70_0000_0000 ^ ((n as u64) << 8) ^ stream
}

/// The data set of `n` memories, in random order.
pub fn generate(structure: &Structure, n: usize) -> Vec<Item> {
    let mut rng = Rng::new(seed_for(n, 0));
    let (counts, noise) = structure.counts(n);
    let mut cells: Vec<Option<usize>> = Vec::with_capacity(n);
    for (c, &k) in counts.iter().enumerate() {
        cells.extend(std::iter::repeat_n(Some(c), k));
    }
    cells.extend(std::iter::repeat_n(None, noise));
    for i in (1..cells.len()).rev() {
        let j = rng.below(i + 1);
        cells.swap(i, j);
    }
    let day0 = DAY0_SECS * 1_000_000;
    cells
        .into_iter()
        .map(|cell| {
            let created = day0 + (rng.uniform() * (DAYS * MICROS_PER_DAY) as f64) as i64;
            structure.item(&mut rng, cell, created)
        })
        .collect()
}

/// `count` new memories for a data set of `n`, created after it (day 365
/// onwards), drawn as its memories are. `stream` picks an independent set.
pub fn extra(structure: &Structure, n: usize, count: usize, stream: u64) -> Vec<Item> {
    let mut rng = Rng::new(seed_for(n, stream.wrapping_add(1)));
    let start = (DAY0_SECS + DAYS * 86_400) * 1_000_000;
    (0..count)
        .map(|i| structure.draw(&mut rng, start + i as i64 * 1_000_000))
        .collect()
}
