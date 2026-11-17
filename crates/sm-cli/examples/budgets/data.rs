//! What the cases store and ask: the spatial fixture's 320 texts, prompts of
//! 128 model tokens made from them, generated version 1 exports, and
//! seeded random vectors (fixtures are data, never instructions).

use std::path::Path;

use serde_json::{Value, json};
use sm_core::EMBED_DIM;
use sm_embed::{ModelTokenizer, TextTokenizer};

/// SplitMix64, as `sm-store`'s `search_budget.rs`: every run uses the same
/// data.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    pub fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// A uniformly random direction of `dim` components (Box-Muller).
    pub fn unit_vector_of(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f64> = (0..dim)
            .map(|_| {
                let (u1, u2) = (self.unit_interval(), self.unit_interval());
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v.into_iter().map(|x| x as f32).collect()
    }

    pub fn unit_vector(&mut self) -> Vec<f32> {
        self.unit_vector_of(EMBED_DIM)
    }
}

/// The 320 texts of `fixtures/spatial/texts.json`, in order (topic `i / 32`).
pub fn texts(repo: &Path) -> Vec<String> {
    let path = repo.join("fixtures").join("spatial").join("texts.json");
    let raw = std::fs::read_to_string(&path).expect("fixtures/spatial/texts.json");
    let rows: Vec<Value> = serde_json::from_str(&raw).expect("texts.json is a JSON array");
    rows.iter()
        .map(|r| r["text"].as_str().expect("a text").to_owned())
        .collect()
}

/// `n` distinct prompts of exactly `tokens` model tokens: texts joined from
/// offset `i`, cut at the end of the `tokens`th token.
pub fn prompts(tok: &ModelTokenizer, texts: &[String], n: usize, tokens: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let mut s = String::new();
            let mut j = i * 7;
            loop {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&texts[j % texts.len()]);
                j += 1;
                let spans = tok.spans(&s).expect("tokenize a prompt");
                if spans.len() > tokens {
                    let end = spans[tokens - 1].1;
                    s.truncate(sm_embed::tokenize::floor_boundary(&s, end));
                    return s;
                }
            }
        })
        .collect()
}

/// A version 1 JSON export of `records` one-chunk memories in `project`
/// (the format of `fixtures/migration/make_v1_export_full.py`, with a 384
/// component vector as version 1 wrote it). `tag` keeps ids and contents
/// apart between exports.
pub fn write_v1_export(
    path: &Path,
    texts: &[String],
    records: usize,
    project: &str,
    tag: u64,
) -> usize {
    let mut rng = Rng(0x5EED_0000 ^ tag);
    let mut out = Vec::with_capacity(records);
    for n in 0..records {
        let a = &texts[n % texts.len()];
        let b = &texts[(n * 13 + 5) % texts.len()];
        let content = format!("{a} Also, {b} (note {tag}-{n})");
        let id = uuid_like(&mut rng);
        let vector: Vec<f64> = rng.unit_vector_of(384).into_iter().map(f64::from).collect();
        let day = 1 + n % 28;
        let created = format!(
            "2025-09-{day:02}T10:{:02}:{:02}.{:06}",
            n % 60,
            (n / 60) % 60,
            n
        );
        out.push(json!({
            "id": id,
            "content": content,
            "vector": vector,
            "created_at": created,
            "updated_at": created,
            "last_accessed": created,
            "access_count": n % 5,
            "importance": 0.5,
            "namespace": "notes",
            "tags": ["bench"],
            "source": "manual",
            "metadata": {},
            "project": project,
            "content_hash": sm_core::hash::content_hash(&content),
            "expires_at": null,
        }));
    }
    let body = serde_json::to_vec(&out).expect("the export serialises");
    std::fs::write(path, &body).expect("write the export");
    body.len()
}

/// A random version 4 UUID in its text form.
fn uuid_like(rng: &mut Rng) -> String {
    let hi = rng.next_u64();
    let lo = rng.next_u64();
    let b: Vec<u8> = hi
        .to_be_bytes()
        .into_iter()
        .chain(lo.to_be_bytes())
        .collect();
    let mut b: [u8; 16] = b.try_into().expect("16 bytes");
    b[6] = (b[6] & 0x0F) | 0x40;
    b[8] = (b[8] & 0x3F) | 0x80;
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// The 50,000-character memory of AT-4: headings, paragraphs and a code
/// block every third part, as `sm-embed`'s `real_model.rs` builds it.
pub fn long_memory(chars: usize) -> String {
    let words = sm_embed::doctor::sample_text(chars / 3);
    let sentences: Vec<&str> = words.split(". ").collect();
    let mut out = String::from("# Long session notes\n\n");
    let mut s = 0;
    let mut section = 1;
    while out.len() < chars.saturating_sub(4_000) {
        out.push_str(&format!("## Part {section}\n\n"));
        for _ in 0..3 {
            for _ in 0..4 {
                out.push_str(sentences[s % sentences.len()]);
                out.push_str(". ");
                s += 1;
            }
            out.push_str("\n\n");
        }
        if section % 3 == 0 {
            out.push_str(&format!(
                "```rust\nfn part_{section}(orders: &[Order]) -> usize {{\n    orders.iter().filter(|o| o.failed).count()\n}}\n```\n\n"
            ));
        }
        section += 1;
    }
    while out.len() < chars {
        out.push_str(sentences[s % sentences.len()]);
        out.push_str(". ");
        s += 1;
    }
    out.truncate(chars);
    out
}

/// `n` memories for `consolidate`: the 320 texts, then variants of them
/// that differ by a short suffix, so many are near-duplicates.
pub fn consolidate_texts(texts: &[String], n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let t = &texts[i % texts.len()];
            if i < texts.len() {
                t.clone()
            } else {
                format!("{t} (seen again, round {})", i / texts.len())
            }
        })
        .collect()
}
