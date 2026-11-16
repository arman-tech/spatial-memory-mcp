//! Embeds `fixtures/spatial/texts.json` into `fixtures/spatial/vectors.f32le`
//! for sm-spatial's clustering-quality test (SPEC 11 AT-7).
//!
//! Each text is one chunk, embedded alone (a batch of one, so no padding),
//! with the document prompt `title: none | text: {text}`, by EmbeddingGemma 2
//! at the pinned revision through ONNX Runtime. The output holds one
//! 768-dimension little-endian f32 vector per text, row-major, in the order of
//! `texts.json`. The model's `sentence_embedding` output is stored as it is.
//!
//! The model is read in place from `SPATIAL_MEMORY_MODEL_DIR` or the per-user
//! cache (`sm_core::paths::model_dir`). It is never copied.
//!
//! Run from the repository root, then delete `fixtures/spatial/generator/target`:
//!
//! ```text
//! cargo run --release --manifest-path fixtures/spatial/generator/Cargo.toml
//! ```

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const DOC_PROMPT: &str = "title: none | text: ";
const DIM: usize = 768;
const MODEL_ID: &str = "embeddinggemma-2";
const THREADS: usize = 4;

/// (path in the model folder, size in bytes, SHA-256): the pins in SPEC section 5.
const FILES: &[(&str, u64, &str)] = &[
    (
        "onnx/model.onnx",
        421_421,
        "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3",
    ),
    (
        "onnx/model.onnx_data",
        1_084_170_240,
        "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2",
    ),
    (
        "tokenizer.json",
        32_170_510,
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
];

#[derive(serde::Deserialize)]
struct Entry {
    topic: String,
    text: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn sha256_file(p: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", p.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(sm_core::hash::to_hex(&h.finalize()))
}

fn run() -> Result<(), String> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("the generator has no parent folder")?;
    let texts_path = fixture.join("texts.json");
    let out_path = fixture.join("vectors.f32le");

    let raw = std::fs::read_to_string(&texts_path)
        .map_err(|e| format!("{}: {e}", texts_path.display()))?;
    let entries: Vec<Entry> =
        serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", texts_path.display()))?;

    let dir = sm_core::paths::model_dir(
        |k| std::env::var(k).ok(),
        MODEL_ID,
        sm_core::config::EMBEDDINGGEMMA_REVISION,
    )
    .map_err(|e| e.to_string())?;
    println!("model folder {}", dir.display());
    for (rel, size, sha) in FILES {
        let p = dir.join(rel);
        let len = std::fs::metadata(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?
            .len();
        if len != *size {
            return Err(format!("{} is {len} bytes, expected {size}", p.display()));
        }
        let got = sha256_file(&p)?;
        if got != *sha {
            return Err(format!("{} has SHA-256 {got}, expected {sha}", p.display()));
        }
        println!("  {rel}: size and SHA-256 match the pin");
    }

    let tk = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer: {e}"))?;
    let ort_err = |x: &dyn std::fmt::Display| format!("ONNX Runtime: {x}");
    let env = ort::init().build().map_err(|x| ort_err(&x))?;
    let mut session = Session::builder(&env)
        .map_err(|x| ort_err(&x))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|x| ort_err(&x))?
        .with_intra_threads(THREADS)
        .map_err(|x| ort_err(&x))?
        .commit_from_file(dir.join("onnx").join("model.onnx"))
        .map_err(|x| ort_err(&x))?;

    let mut bytes = Vec::with_capacity(entries.len() * DIM * 4);
    let mut topics: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut min_norm, mut max_norm, mut max_tokens) = (f32::MAX, f32::MIN, 0usize);
    for (i, entry) in entries.iter().enumerate() {
        *topics.entry(entry.topic.as_str()).or_default() += 1;
        let prompt = format!("{DOC_PROMPT}{}", entry.text);
        let ids: Vec<i64> = tk
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize text {i}: {e}"))?
            .get_ids()
            .iter()
            .map(|&x| i64::from(x))
            .collect();
        max_tokens = max_tokens.max(ids.len());
        let n = ids.len();
        let mask = vec![1i64; n];
        let empty = || Tensor::from_array(ndarray::Array2::<f32>::zeros((0, 512)));
        let out = session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, n], ids)).map_err(|x| ort_err(&x))?,
                "attention_mask" => Tensor::from_array(([1usize, n], mask)).map_err(|x| ort_err(&x))?,
                "image_features" => empty().map_err(|x| ort_err(&x))?,
                "video_features" => empty().map_err(|x| ort_err(&x))?,
                "audio_features" => empty().map_err(|x| ort_err(&x))?,
            ])
            .map_err(|x| ort_err(&x))?;
        let (_, data) = out["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|x| ort_err(&x))?;
        if data.len() != DIM {
            return Err(format!("text {i}: {} values, expected {DIM}", data.len()));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(format!("text {i}: the vector holds a non-finite value"));
        }
        let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !(0.99..=1.01).contains(&norm) {
            return Err(format!("text {i}: norm {norm}, expected about 1"));
        }
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }

    std::fs::write(&out_path, &bytes).map_err(|e| format!("{}: {e}", out_path.display()))?;
    println!("texts {} in {} topics: {topics:?}", entries.len(), topics.len());
    println!("longest input {max_tokens} tokens; norms {min_norm:.6}..{max_norm:.6}");
    println!("wrote {} ({} bytes)", out_path.display(), bytes.len());
    println!("SHA-256 {}", sm_core::hash::sha256_hex(&bytes));
    Ok(())
}
