//! The one test binary that loads the real EmbeddingGemma 2 model (SPEC 11;
//! stream B). It loads the model once and runs every case in sequence, so
//! the memory readings mean something. When the model folder is absent it
//! prints one line and passes.
//!
//! The folder is `SPATIAL_MEMORY_MODEL_DIR`, or the per-user model cache.
//! The model is read, never copied or written. Run it in release so the
//! timings mean something:
//!
//! ```text
//! cargo test -p sm-embed --release --test real_model -- --nocapture
//! ```
//!
//! Report lines go straight to standard output, past the test harness's
//! capture, so they show without `--nocapture` too.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sm_core::config::Settings;
use sm_core::eval::{Baseline, EVAL_LIMIT, EvalSet, Retrieved, evaluate};
use sm_core::traits::{ChunkParams, Chunker, DocInput, Embedder, EmbedderProvider, SessionKind};
use sm_embed::doctor::{doctor_numbers, sample_text};
use sm_embed::gemma::PAD_TOKEN;
use sm_embed::input::{NO_TITLE, document_prompt, query_prompt};
use sm_embed::model::{self, MODEL_FILES};
use sm_embed::threads::fallback_threads;
use sm_embed::{EmbedConfig, GemmaEmbedder, GemmaProvider, TextTokenizer, os, physical_cores};

const MB: f64 = 1024.0 * 1024.0;
const GB: u64 = 1024 * 1024 * 1024;

/// AT-1: the first example of README.md in the model repository at the
/// pinned revision.
const AT1_QUERY: &str = "Which planet is known as the Red Planet?";
const AT1_DOCS: [(&str, &str, f32); 4] = [
    (
        "Venus",
        "Venus is often called Earth's twin because of its similar size and proximity.",
        0.675,
    ),
    (
        "Mars",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        0.853,
    ),
    (
        "Jupiter",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        0.747,
    ),
    (
        "Saturn",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        0.778,
    ),
];
const AT1_TOLERANCE: f32 = 0.01;
const AT1_BATCH_COSINE: f32 = 0.9999;
/// FNV-1a over the input ids of the five AT-1 texts, as phase 0's self-test
/// computed it (SPIKE.md P0 a): the tokenizer has not drifted.
const AT1_TOKEN_FINGERPRINT: u64 = 0x434a_9df0_5b5b_5fc5;

/// AT-4.
const LONG_MEMORY_CHARS: usize = 50_000;
const PEAK_MEMORY_BUDGET: u64 = 2 * GB;
/// SPEC 11: daemon memory, idle after load.
const IDLE_MEMORY_BUDGET: u64 = GB;
/// SPEC 11: daemon ready from cold.
const COLD_START_BUDGET: Duration = Duration::from_secs(5);
/// SPEC 6: a hook query is cut to 128 tokens; 200 ms on the test machine.
const QUERY_TOKENS: usize = 128;
const QUERY_LATENCY_REFERENCE: Duration = Duration::from_millis(200);
const QUERY_RUNS: usize = 10;
/// D-10 at a small explicit budget.
const SMALL_BULK_THREADS: usize = 2;

/// A report line, written past the test harness's capture.
fn say(line: impl AsRef<str>) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "real_model: {}", line.as_ref());
    let _ = out.flush();
}

fn mb(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / MB)
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1000.0)
}

fn secs(d: Duration) -> String {
    format!("{:.2} s", d.as_secs_f64())
}

fn within(measured: Duration, budget: Duration) -> &'static str {
    if measured <= budget { "within" } else { "OVER" }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn fnv_ids(ids: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in ids {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

fn repo_file(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn chunk_params(settings: &Settings) -> ChunkParams {
    ChunkParams {
        max_tokens: settings.chunk.max_tokens as usize,
        overlap: settings.chunk.overlap,
        index_cap: Some(settings.capture.max_index_tokens as usize),
    }
}

/// The model folder, or `None` (with the skip line printed) when absent.
fn model_folder() -> Option<PathBuf> {
    let dir = match model::default_model_dir() {
        Ok(dir) => dir,
        Err(e) => {
            say(format!("SKIPPED: no model folder ({})", e.message));
            return None;
        }
    };
    if !dir.is_dir() {
        say(format!(
            "SKIPPED: the model folder {} is absent; set SPATIAL_MEMORY_MODEL_DIR or run `spatial-memory setup`",
            dir.display()
        ));
        return None;
    }
    Some(dir)
}

#[test]
fn s11_real_model_cases_in_sequence() {
    let Some(dir) = model_folder() else {
        return;
    };
    let settings = Settings::default();
    let physical = physical_cores();
    let logical = std::thread::available_parallelism().map_or(0, |n| n.get());
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    say(format!(
        "machine: {} {} | {} | {physical} physical cores, {logical} logical | profile {profile}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        os::cpu_name(),
    ));
    say(format!("model folder: {}", dir.display()));

    let verify_elapsed = at10_installed_model_verifies(&dir);
    let (provider, query, idle) = s11_load_and_idle_memory(&dir, verify_elapsed);
    at1_scores_match_the_reference(&provider, &query);
    let latency = s11_query_latency_at_128_tokens(&query);
    let bulk = at4_long_memory_stays_under_2_gb(&provider, &settings, physical);
    at2_retrieval_quality_holds(&provider, &settings, &bulk, &query);
    drop(bulk);
    d10_small_bulk_session_stays_within_n(&provider, &settings);
    s11_doctor_numbers(&provider, &settings, physical);
    drop(query);
    say(format!(
        "budgets summary: verify {}, load to first query session {}, idle {}, query@128 median {}, peak {}",
        secs(verify_elapsed),
        secs(idle.load),
        mb(idle.bytes),
        ms(latency),
        os::peak_memory_bytes().map_or_else(|| "unknown".to_owned(), mb),
    ));
    say("RESULT: PASS");
}

/// AT-10 on the real files: the installed model passes `verify`. Times the
/// full check against the 5-second cold-start budget.
fn at10_installed_model_verifies(dir: &Path) -> Duration {
    let report = model::verify(dir).unwrap_or_else(|e| {
        panic!(
            "AT-10: the installed model fails verify: {:?}: {} ({})",
            e.code, e.message, e.next_action
        )
    });
    assert_eq!(report.files.len(), MODEL_FILES.len());
    for f in &report.files {
        say(format!(
            "AT-10 verify {}: {} in {}",
            f.path,
            mb(f.bytes),
            secs(f.elapsed)
        ));
    }
    say(format!(
        "AT-10 PASS: full size and SHA-256 check of {} in {} ({:.0} MB/s); {} the {} cold-start budget on its own",
        mb(report.bytes()),
        secs(report.elapsed),
        report.bytes() as f64 / MB / report.elapsed.as_secs_f64().max(1e-9),
        within(report.elapsed, COLD_START_BUDGET),
        secs(COLD_START_BUDGET),
    ));
    report.elapsed
}

struct Idle {
    load: Duration,
    bytes: u64,
}

/// Model load (verify, tokenizer, then the query session, which the daemon
/// creates first and keeps) and the memory after it (SPEC 11: 1 GB).
fn s11_load_and_idle_memory(
    dir: &Path,
    verify_elapsed: Duration,
) -> (GemmaProvider, GemmaEmbedder, Idle) {
    let settings = Settings::default();
    let started = Instant::now();
    let provider = GemmaProvider::load(dir, EmbedConfig::from_settings(&settings))
        .unwrap_or_else(|e| panic!("load failed: {}: {}", e.message, e.next_action));
    let provider_elapsed = started.elapsed();
    let session_started = Instant::now();
    let query = provider
        .open_gemma(SessionKind::Query, settings.query_threads(physical_cores()))
        .unwrap_or_else(|e| panic!("query session failed: {}: {}", e.message, e.next_action));
    let session_elapsed = session_started.elapsed();
    let load = started.elapsed();
    let bytes = os::current_memory_bytes().expect("the OS reports this process's memory");
    say(format!(
        "load: provider {} (of which verify {}), query session {}, total {} ({} the {} cold-start budget)",
        secs(provider_elapsed),
        secs(verify_elapsed),
        secs(session_elapsed),
        secs(load),
        within(load, COLD_START_BUDGET),
        secs(COLD_START_BUDGET),
    ));
    say(format!(
        "idle memory after load: {} (budget {})",
        mb(bytes),
        mb(IDLE_MEMORY_BUDGET)
    ));
    assert!(
        bytes <= IDLE_MEMORY_BUDGET,
        "SPEC 11: idle memory after load is {}, over {}",
        mb(bytes),
        mb(IDLE_MEMORY_BUDGET)
    );
    (provider, query, Idle { load, bytes })
}

/// AT-1: the reference scores, unit vectors, and the same text alone and in
/// a padded batch.
fn at1_scores_match_the_reference(provider: &GemmaProvider, query: &GemmaEmbedder) {
    let tok = provider.tokenizer();
    assert_eq!(
        tok.token_id(PAD_TOKEN),
        Some(0),
        "SPEC 5: the pad token is id 0"
    );
    let mut ids = tok.encode(&query_prompt(AT1_QUERY)).unwrap().ids;
    let mut lengths = Vec::new();
    for (_, text, _) in AT1_DOCS {
        let enc = tok.encode(&document_prompt(NO_TITLE, text)).unwrap();
        lengths.push(enc.len());
        ids.extend_from_slice(&enc.ids);
    }
    let fingerprint = fnv_ids(&ids);
    say(format!(
        "AT-1 token fingerprint {fingerprint:016x}, reference {AT1_TOKEN_FINGERPRINT:016x}"
    ));
    assert_eq!(
        fingerprint, AT1_TOKEN_FINGERPRINT,
        "AT-1: the tokenizer drifted"
    );
    let longest = *lengths.iter().max().unwrap();
    assert!(
        lengths.iter().any(|&n| n < longest),
        "AT-1 needs one document shorter than another so the batch is padded: {lengths:?}"
    );

    let q = query.embed_query(AT1_QUERY, QUERY_TOKENS).unwrap();
    let docs: Vec<DocInput<'_>> = AT1_DOCS
        .iter()
        .map(|(_, text, _)| DocInput { title: None, text })
        .collect();
    let batch = query.embed_documents(&docs).unwrap();
    assert_eq!(batch.len(), AT1_DOCS.len());
    for v in std::iter::once(&q).chain(&batch) {
        assert_eq!(v.len(), sm_core::EMBED_DIM);
        let norm = dot(v, v).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "AT-1: a vector has norm {norm}");
    }
    let mut line = String::new();
    for ((name, _, reference), v) in AT1_DOCS.iter().zip(&batch) {
        let score = dot(&q, v);
        line.push_str(&format!(" {name} {score:.3} (ref {reference:.3})"));
        assert!(
            (score - reference).abs() <= AT1_TOLERANCE,
            "AT-1: {name} scores {score:.4}, reference {reference} ± {AT1_TOLERANCE}"
        );
    }
    say(format!("AT-1 scores:{line}"));

    let mut worst = 1.0f32;
    for (i, doc) in docs.iter().enumerate() {
        let alone = query.embed_documents(std::slice::from_ref(doc)).unwrap();
        worst = worst.min(dot(&alone[0], &batch[i]));
    }
    say(format!(
        "AT-1 alone against padded batch (lengths {lengths:?}): lowest cosine {worst:.6}"
    ));
    assert!(
        worst >= AT1_BATCH_COSINE,
        "AT-1: alone against batch cosine {worst} < {AT1_BATCH_COSINE}"
    );
    say("AT-1 PASS");
}

/// SPEC 11 budget: a 128-token query on the query session, warm.
fn s11_query_latency_at_128_tokens(query: &GemmaEmbedder) -> Duration {
    let question = sample_text(QUERY_TOKENS * 2);
    query.embed_query(&question, QUERY_TOKENS).unwrap();
    let mut times = Vec::with_capacity(QUERY_RUNS);
    for _ in 0..QUERY_RUNS {
        let started = Instant::now();
        query.embed_query(&question, QUERY_TOKENS).unwrap();
        times.push(started.elapsed());
    }
    let max = *times.iter().max().unwrap();
    let med = median(times);
    let report = query.threads();
    say(format!(
        "query latency at {QUERY_TOKENS} tokens on {} threads: median {} of {QUERY_RUNS}, slowest {} ({} the {} measured on the SPEC's test machine)",
        report.plan.intra_threads,
        ms(med),
        ms(max),
        within(med, QUERY_LATENCY_REFERENCE),
        ms(QUERY_LATENCY_REFERENCE),
    ));
    assert!(
        !report.plan.lower,
        "SPEC 5: the query session runs at normal priority"
    );
    med
}

/// A memory of exactly `chars` ASCII characters with headings, paragraphs
/// and fenced code, so the real chunker's rules all apply.
fn long_memory(chars: usize) -> String {
    let words = sample_text(chars / 3);
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

/// AT-4 (embedding part) with D-10 at the default budget: a 50,000-character
/// memory through the real chunker, every chunk embedded at the default
/// batch size on a bulk session opened beside the query session, as the
/// daemon runs them. Peak memory is read from the OS. Returns the bulk
/// session for AT-2.
fn at4_long_memory_stays_under_2_gb(
    provider: &GemmaProvider,
    settings: &Settings,
    physical: usize,
) -> GemmaEmbedder {
    let text = long_memory(LONG_MEMORY_CHARS);
    assert_eq!(text.chars().count(), LONG_MEMORY_CHARS);
    let chunker = provider.chunker();
    let plan = chunker.chunk(&text, &chunk_params(settings)).unwrap();
    say(format!(
        "AT-4 chunking: {} chunks, {} tokens in all, {} indexed, capped {}",
        plan.pieces.len(),
        plan.total_tokens,
        plan.indexed_tokens,
        plan.capped
    ));
    assert!(
        plan.pieces.len() > 1,
        "AT-4: a long memory makes several chunks"
    );
    let docs: Vec<DocInput<'_>> = plan
        .pieces
        .iter()
        .map(|p| DocInput {
            title: Some("Long session notes"),
            text: &p.text,
        })
        .collect();

    let n = settings.bulk_threads(physical);
    let bulk = provider.open_gemma(SessionKind::Bulk, n).unwrap();
    let started = Instant::now();
    let vectors = bulk.embed_documents(&docs).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(vectors.len(), docs.len());
    assert_eq!(
        bulk.truncated_inputs(),
        0,
        "SPEC 5 rule 6: no chunk was cut"
    );
    let peak = os::peak_memory_bytes().expect("the OS reports peak memory");
    say(format!(
        "AT-4: {} chunks at batch {} on {} bulk threads in {} ({:.2} chunks/s, first batch included)",
        docs.len(),
        settings.embed.batch_size,
        bulk.intra_threads(),
        secs(elapsed),
        docs.len() as f64 / elapsed.as_secs_f64().max(1e-9),
    ));
    say(format!(
        "AT-4 peak memory {} with the query and bulk sessions loaded (budget under {})",
        mb(peak),
        mb(PEAK_MEMORY_BUDGET)
    ));
    assert!(
        peak < PEAK_MEMORY_BUDGET,
        "AT-4: peak memory {} is not under {}",
        mb(peak),
        mb(PEAK_MEMORY_BUDGET)
    );
    say("AT-4 PASS");
    d10_bulk_threads_and_priority(&bulk, n, "default budget");
    bulk
}

/// D-10, SPEC 5 rules 1, 2 and 4: a bulk session opened with `n` threads
/// runs no more than `n`, and its threads run below normal priority as the
/// OS reads it back. When the OS refuses, the SPEC 13 fallback must show.
fn d10_bulk_threads_and_priority(bulk: &GemmaEmbedder, n: usize, label: &str) {
    let report = bulk.threads();
    let normal = os::thread_priority(os::current_thread_id())
        .expect("the OS reports this thread's priority");
    say(format!(
        "D-10 ({label}): asked {n}, intra-op {}, pool now {} (peak {}, started {}), running at most {}, lowered {}, refused {}",
        report.plan.intra_threads,
        report.pool.live,
        report.pool.peak,
        report.pool.spawned,
        report.max_running(),
        report.pool.lowered,
        report.pool.refused,
    ));
    assert!(
        report.plan.intra_threads <= n,
        "D-10: more threads than asked"
    );
    assert!(
        report.max_running() <= n,
        "D-10: {} threads ran inference, asked for {n}",
        report.max_running()
    );
    if report.plan.priority_fallback {
        let reason = report.plan.fallback_reason.clone().unwrap_or_default();
        say(format!(
            "D-10 ({label}): priority cannot be lowered here, SPEC 13 fallback: {} threads ({reason})",
            report.plan.intra_threads
        ));
        assert_eq!(
            report.plan.intra_threads,
            fallback_threads(n.min(physical_cores()))
        );
        return;
    }
    assert!(report.plan.lower);
    assert_eq!(
        report.pool.refused, 0,
        "D-10: the OS refused to lower a thread"
    );
    assert!(
        !report.pool.thread_ids.is_empty(),
        "D-10: no bulk thread was seen"
    );
    let mut read = Vec::new();
    for &tid in &report.pool.thread_ids {
        let p = os::thread_priority(tid)
            .unwrap_or_else(|| panic!("D-10: cannot read the priority of thread {tid}"));
        read.push(p);
        assert!(
            os::is_below(p, normal),
            "D-10: thread {tid} runs at {p}, not below normal {normal}"
        );
    }
    // One step below normal, the same for every thread: the runner and the
    // pool threads it starts are not lowered twice.
    assert!(
        read.windows(2).all(|w| w[0] == w[1]),
        "D-10: bulk threads read back different priorities: {read:?}"
    );
    say(format!(
        "D-10 ({label}) PASS: {} threads read back below normal: {read:?} (this thread {normal})",
        read.len()
    ));
}

/// AT-2 (embedding part): the evaluation set through the real chunker, a
/// memory scored by its best chunk, gated against the baseline.
fn at2_retrieval_quality_holds(
    provider: &GemmaProvider,
    settings: &Settings,
    bulk: &GemmaEmbedder,
    query: &GemmaEmbedder,
) {
    let set = EvalSet::load(&repo_file("fixtures/retrieval/retrieval_eval_set.json")).unwrap();
    let baseline = Baseline::load(&repo_file(
        "fixtures/retrieval/retrieval_quality_baseline.json",
    ))
    .unwrap();
    let chunker = provider.chunker();
    let params = chunk_params(settings);
    let mut owners = Vec::new();
    let mut texts = Vec::new();
    for m in &set.memories {
        for piece in chunker.chunk(&m.content, &params).unwrap().pieces {
            owners.push(m.id.clone());
            texts.push(piece.text);
        }
    }
    let docs: Vec<DocInput<'_>> = texts
        .iter()
        .map(|t| DocInput {
            title: None,
            text: t,
        })
        .collect();
    let started = Instant::now();
    let vectors = bulk.embed_documents(&docs).unwrap();
    say(format!(
        "AT-2: {} memories as {} chunks embedded in {}",
        set.memories.len(),
        docs.len(),
        secs(started.elapsed())
    ));
    let limit = settings.chunk.max_tokens as usize;
    let mut retrieved = Vec::with_capacity(set.queries.len());
    for q in &set.queries {
        let qv = query.embed_query(&q.text, limit).unwrap();
        let mut best: HashMap<&str, f32> = HashMap::new();
        for (owner, v) in owners.iter().zip(&vectors) {
            let s = dot(&qv, v);
            let e = best.entry(owner.as_str()).or_insert(f32::MIN);
            *e = e.max(s);
        }
        let mut ranked: Vec<Retrieved> = best
            .into_iter()
            .map(|(id, score)| Retrieved {
                id: id.to_owned(),
                score,
            })
            .collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        ranked.truncate(EVAL_LIMIT);
        retrieved.push(ranked);
    }
    let report = evaluate(&set, &retrieved, Some(&baseline)).unwrap();
    let m = report.metrics;
    say(format!(
        "AT-2: nDCG@10 {:.4} (baseline {:.4}), MRR {:.4} (baseline {:.4}), hit@5 {:.4} (baseline {:.4}) over {} queries",
        m.ndcg_at_10,
        baseline.ndcg_at_10,
        m.mrr,
        baseline.mrr,
        m.hit_at_5,
        baseline.hit_at_5,
        report.queries
    ));
    for q in report
        .per_query
        .iter()
        .filter(|q| !q.hit_at_5 || q.reciprocal_rank < 1.0)
    {
        say(format!(
            "AT-2 query {:?}: first relevant at {:?}, missing {:?}",
            q.text, q.first_relevant_rank, q.missing
        ));
    }
    assert!(
        report.regressions.is_empty(),
        "AT-2 gate: {}",
        report.regressions.join("; ")
    );
    say("AT-2 PASS");
}

/// D-10 at a small explicit budget, on a fresh bulk session.
fn d10_small_bulk_session_stays_within_n(provider: &GemmaProvider, settings: &Settings) {
    let bulk = provider
        .open_gemma(SessionKind::Bulk, SMALL_BULK_THREADS)
        .unwrap();
    let text = sample_text(settings.chunk.max_tokens as usize * 8);
    let plan = provider
        .chunker()
        .chunk(&text, &chunk_params(settings))
        .unwrap();
    let docs: Vec<DocInput<'_>> = plan
        .pieces
        .iter()
        .map(|p| DocInput {
            title: None,
            text: &p.text,
        })
        .collect();
    let threads_before = os::process_thread_count();
    bulk.embed_documents(&docs).unwrap();
    let threads_after = os::process_thread_count();
    say(format!(
        "D-10 (n = {SMALL_BULK_THREADS}): {} chunks embedded; process threads {threads_before:?} before, {threads_after:?} after",
        docs.len()
    ));
    d10_bulk_threads_and_priority(&bulk, SMALL_BULK_THREADS, "n = 2");
}

/// SPEC 5 rule 7: what `doctor` reports, on the real model, at the default
/// budget. Its sessions are opened after the daemon's query session, so its
/// query figure is the slower one SPEC 16 describes.
fn s11_doctor_numbers(provider: &GemmaProvider, settings: &Settings, physical: usize) {
    let chunker = provider.chunker();
    let p: &dyn EmbedderProvider = provider;
    let n = doctor_numbers(p, &chunker, settings, physical).unwrap();
    say(format!(
        "doctor: bulk {:.2} chunks/s on {} threads at batch {} ({} chunks in {}); query at {} tokens on {} threads, median {} of {}",
        n.bulk_chunks_per_sec,
        n.bulk_threads,
        n.batch_size,
        n.sample_chunks,
        secs(n.bulk_elapsed),
        n.query_tokens,
        n.query_threads,
        ms(n.query_latency_median),
        n.query_runs,
    ));
    assert!(n.bulk_chunks_per_sec > 0.0);
}
