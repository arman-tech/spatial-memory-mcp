//! The cases of stream K, Part 2. Case 15 is not here: it is these cases
//! run in Docker (see RESULTS.md). Time is printed, never asserted (B-58);
//! memory budgets print pass or FAIL, judged on the working set (B-114).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::json;
use sm_core::config::{AMBIENT_ENV, Settings};
use sm_core::ids::ProjectId;
use sm_core::journal::metrics::HOOK_LATENCY_MS;
use sm_core::protocol::{DaemonState, DoctorNumbersReply, ModelStatus, Reply};
use sm_core::traits::{ChunkPiece, MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_embed::{ModelTokenizer, TextTokenizer, physical_cores};
use sm_store::{Access, MemoryInput, NewChunk, Store};

use crate::data::{self, Rng};
use crate::harness::{Env, Home, LONG, figure, memory, ms, note, op, stats, unique};
use crate::mcp;
use crate::procs::{self, Sampler};

/// SPEC 11 says GB; the decimal unit is the stricter reading.
const GB: u64 = 1_000_000_000;
const IDLE_BUDGET: u64 = GB;
const PEAK_BUDGET: u64 = 2 * GB;
/// The SPEC 11 hook deadline.
const DEADLINE: Duration = Duration::from_millis(1_500);
const HOOK_ARGV: &[&str] = &["hook", "user-prompt-submit", "--client", "claude-code"];
/// The bulk session's runner thread (`sm-embed`'s `Runner`).
const BULK_RUNNER: &str = "sm-embed-bulk";

pub const NAMES: &[&str] = &[
    "cold",
    "idle",
    "hook",
    "import",
    "open",
    "kill",
    "remember",
    "at4",
    "recall100k",
    "consolidate",
    "at2",
    "at7",
    "at8",
    "at10",
    "doctor",
];

/// Runs the named case; `false` when there is no such case.
pub fn run(name: &str, env: &Env) -> bool {
    let started = Instant::now();
    match name {
        "cold" => cold(env),
        "idle" => idle(env),
        "hook" => hook(env),
        "import" => import(env),
        "open" => open(env),
        "kill" => kill(env),
        "remember" => remember(env),
        "at4" => at4(env),
        "recall100k" => recall100k(env),
        "consolidate" => consolidate(env),
        "at2" => at2(env),
        "at7" => at7(env),
        "at8" => at8(env),
        "at10" => at10(env),
        "doctor" => doctor(env),
        _ => return false,
    }
    note(
        name,
        format!("case finished in {:.1} s", started.elapsed().as_secs_f64()),
    );
    true
}

// ---------------------------------------------------------------------------
// Shared steps
// ---------------------------------------------------------------------------

fn tokenizer(env: &Env) -> ModelTokenizer {
    ModelTokenizer::from_file(&sm_embed::model::TOKENIZER.local_path(&env.model_dir))
        .expect("the model's tokenizer")
}

/// `n` prompts of 128 model tokens.
fn hook_prompts(env: &Env, texts: &[String], n: usize) -> Vec<String> {
    data::prompts(&tokenizer(env), texts, n, 128)
}

/// Remembers `texts` in `cwd` and waits until every chunk is embedded.
fn seed(case: &str, home: &Home, cwd: &Path, texts: &[String]) {
    let started = Instant::now();
    home.remember_all(cwd, texts);
    home.wait_embedded();
    let chunks = home.status().index.chunks;
    let took = started.elapsed().as_secs_f64();
    note(
        case,
        format!(
            "seeded {} memories ({chunks} chunks in the store) in {took:.1} s",
            texts.len()
        ),
    );
}

/// Waits until the daemon has no bulk session (no `sm-embed-bulk` thread);
/// how long that took, or `None` after `limit`.
fn wait_bulk_closed(pid: u32, limit: Duration) -> Option<Duration> {
    let started = Instant::now();
    loop {
        if !procs::thread_names(pid).contains_key(BULK_RUNNER) {
            return Some(started.elapsed());
        }
        if started.elapsed() >= limit {
            return None;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn bulk_open(pid: u32) -> bool {
    procs::thread_names(pid).contains_key(BULK_RUNNER)
}

/// The profile's store, opened as the daemon opens it: configured with the
/// daemon's model stamp and the default chunk settings.
fn open_store(home: &Home, model: &ModelStatus) -> Store {
    let settings = Settings::default();
    let store = Store::open(&home.paths).expect("open the store");
    let status = store
        .configure_index(IndexStamp {
            model_id: model.model_id.clone(),
            revision: model.revision.clone(),
            dim: model.dim as usize,
            chunk_max_tokens: settings.chunk.max_tokens,
            chunk_overlap: settings.chunk.overlap,
        })
        .expect("configure the index");
    assert!(
        status.matches,
        "the daemon's stamp: {:?}",
        status.differences
    );
    store
}

fn doctor_numbers(home: &Home) -> DoctorNumbersReply {
    match home.call(&home.tmp, op(json!({"op": "doctor_numbers"}))) {
        Reply::DoctorNumbers(n) => n,
        other => panic!("doctor_numbers: {other:?}"),
    }
}

fn numbers_line(n: &DoctorNumbersReply) -> String {
    format!(
        "bulk {:.1} chunks/s on {} threads, batch {} ({} chunks in {:.0} ms); query median {:.1} ms at {} tokens on {} threads ({} runs); {} physical cores",
        n.bulk_chunks_per_sec,
        n.bulk_threads,
        n.batch_size,
        n.sample_chunks,
        n.bulk_elapsed_ms,
        n.query_median_ms,
        n.query_tokens,
        n.query_threads,
        n.query_runs,
        n.physical_cores
    )
}

/// The chunks embedded so far: stored minus waiting.
fn embedded(home: &Home) -> u64 {
    let s = home.status();
    s.index.chunks - s.index.pending_embeddings
}

fn median(times: &[Duration]) -> Duration {
    let mut v = times.to_vec();
    v.sort();
    v[(v.len() - 1) / 2]
}

/// A prompt hook that must exit 0.
fn hook_once(home: &Home, cwd: &Path, prompt: &str) -> (Duration, bool) {
    let run = home.prompt_hook(cwd, prompt, &unique("s"));
    assert_eq!(run.code, Some(0), "the prompt hook exits 0: {}", run.stderr);
    (run.elapsed, !run.stdout.is_empty())
}

fn configured() -> (usize, usize, usize, usize) {
    let s = Settings::default();
    let physical = physical_cores();
    (
        physical,
        s.query_threads(physical),
        s.bulk_threads(physical),
        s.embed.batch_size as usize,
    )
}

fn reading_line(r: &procs::Reading) -> String {
    let file = r
        .file_backed
        .map(|f| format!(", file-backed {}", procs::mib(f)))
        .unwrap_or_default();
    format!(
        "working set {}, peak {}, private {}{file}, {} threads",
        procs::mib(r.working_set),
        procs::mib(r.peak_working_set),
        procs::mib(r.private),
        r.threads
    )
}

// ---------------------------------------------------------------------------
// 1. Cold start
// ---------------------------------------------------------------------------

fn cold(env: &Env) {
    const C: &str = "cold";
    note(
        C,
        "label: warm file cache (the model files were read recently; a cold cache is not measured)",
    );
    let mut ready = Vec::new();
    let mut first = Vec::new();
    for i in 0..5 {
        let home = Home::new(env, C);
        let mut d = home.start_ready();
        ready.push(d.ready);
        first.push(d.first_answer);
        let r = d.read();
        note(
            C,
            format!(
                "run {}: first answer {:.1} ms, ready {:.1} ms, the daemon's ready_seconds {:?}; {}",
                i + 1,
                ms(d.first_answer),
                ms(d.ready),
                d.info.ready_seconds,
                reading_line(&r)
            ),
        );
        d.stop(&home);
    }
    figure(C, "spawn to ready, warm file cache", &ready, "<= 5,000 ms");
    figure(C, "spawn to the first answer (loading)", &first, "none");
}

// ---------------------------------------------------------------------------
// 2. Idle
// ---------------------------------------------------------------------------

fn idle(env: &Env) {
    const C: &str = "idle";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    std::thread::sleep(Duration::from_secs(10));
    let r = d.read();
    memory(C, "10 s after ready", r.working_set, r.private, IDLE_BUDGET);
    note(C, format!("  {}", reading_line(&r)));
    note(
        C,
        format!(
            "  threads: {}",
            procs::names_line(&procs::thread_names(d.pid))
        ),
    );

    let proj = home.folder("idleproj");
    let texts = data::texts(&env.repo);
    seed(C, &home, &proj, &texts[..32]);
    let busy = d.read();
    note(
        C,
        format!("right after the bulk work: {}", reading_line(&busy)),
    );
    match wait_bulk_closed(d.pid, Duration::from_secs(90)) {
        Some(t) => note(
            C,
            format!(
                "the bulk session was released {:.1} s after the last chunk was embedded (bulk_grace 10 s)",
                t.as_secs_f64()
            ),
        ),
        None => note(C, "the bulk session was still open 90 s after the work"),
    }
    std::thread::sleep(Duration::from_secs(2));
    let r = d.read();
    memory(
        C,
        "after the bulk grace has passed once",
        r.working_set,
        r.private,
        IDLE_BUDGET,
    );
    note(C, format!("  {}", reading_line(&r)));
    note(
        C,
        format!(
            "  threads: {}",
            procs::names_line(&procs::thread_names(d.pid))
        ),
    );
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 3. Hook warm, and its parts (B-109)
// ---------------------------------------------------------------------------

fn hook(env: &Env) {
    const C: &str = "hook";
    let home = Home::new(env, C);
    let proj = home.folder("hookproj");
    let texts = data::texts(&env.repo);
    {
        let mut d = home.start_ready();
        seed(C, &home, &proj, &texts);
        d.stop(&home);
    }
    let prompts = hook_prompts(env, &texts, 105);
    let s = Settings::default();

    // The daemon as a session meets it: started, then idle before the first
    // prompt. Windows puts the background daemon under power throttling
    // (EcoQoS) within seconds of its start; the diagnostic below measures
    // what that costs.
    let mut d = home.start_ready();
    std::thread::sleep(Duration::from_secs(10));
    let (times, injected) = warm_hooks(&home, &proj, &prompts);
    figure(
        C,
        "prompt hook, warm, end to end (128-token prompt, 320 memories, daemon idle 10 s first, 5 warm-up runs not counted)",
        &times,
        "p95 <= 300 ms",
    );
    note(
        C,
        format!("{injected} of {} hooks injected a block", times.len()),
    );
    let journal: Vec<Duration> = home
        .journal(HOOK_LATENCY_MS)
        .iter()
        .map(|e| Duration::from_secs_f64(e.value / 1000.0))
        .collect();
    figure(
        C,
        "the hook's own journal, hook_latency_ms (warm-up included)",
        &journal,
        "none",
    );
    note(C, format!("gates: {:?}", home.gates()));
    let after = d.read();
    memory(
        C,
        "after the hooks",
        after.working_set,
        after.private,
        IDLE_BUDGET,
    );

    // The daemon's part: ambient_recall without the hook process.
    let direct = direct_recalls(&home, &proj, &prompts, &s);
    figure(
        C,
        "ambient_recall round trip, no hook process (embed + project + search + gate)",
        &direct,
        "none",
    );
    let (hook_p50, _, _) = stats(&times);
    let (direct_p50, _, _) = stats(&direct);
    note(
        C,
        format!(
            "the hook process and its pipe, by difference of medians: {:.1} ms",
            hook_p50 - direct_p50
        ),
    );

    // Query embedding: doctor's median of 5 runs at 128 tokens on the query
    // session (it opens the bulk session too).
    let n = doctor_numbers(&home);
    note(
        C,
        format!(
            "query embedding, 128 tokens: median {:.1} ms over {} runs on {} query threads (doctor_numbers)",
            n.query_median_ms, n.query_runs, n.query_threads
        ),
    );
    let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));

    // Diagnostic, not as shipped: the same daemon with Windows power
    // throttling turned off from outside.
    #[cfg(windows)]
    match procs::no_power_throttling(d.pid) {
        Ok(()) => {
            let (fast, _) = warm_hooks(&home, &proj, &prompts);
            figure(
                C,
                "diagnostic, not as shipped: prompt hook, warm, with Windows power throttling turned off for the daemon",
                &fast,
                "none",
            );
            let fast_direct = direct_recalls(&home, &proj, &prompts, &s);
            figure(
                C,
                "diagnostic: ambient_recall round trip, power throttling off",
                &fast_direct,
                "none",
            );
            let n = doctor_numbers(&home);
            note(
                C,
                format!(
                    "diagnostic: query embedding median {:.1} ms with power throttling off",
                    n.query_median_ms
                ),
            );
        }
        Err(e) => note(C, format!("diagnostic not run: {e}")),
    }

    let model = home.status().model;
    d.stop(&home);

    // Search: the store alone, as the daemon runs it, on the same file.
    let store = open_store(&home, &model);
    let identity = sm_core::project::resolve_project(&proj).expect("resolve the project");
    let project = store.upsert_project(&identity).expect("the project").id;
    let mut rng = Rng(11);
    let mut search = Vec::new();
    for i in 0..110 {
        let v = rng.unit_vector();
        let started = Instant::now();
        let hits = store
            .nearest(&VectorQuery {
                vector: &v,
                scope: QueryScope::Project(project),
                filter: MemoryFilter::default(),
                limit: s.ambient.top_k as usize,
            })
            .expect("search");
        let took = started.elapsed();
        assert!(!hits.is_empty(), "the search finds the seeded memories");
        if i >= 10 {
            search.push(took);
        }
    }
    figure(
        C,
        "search: Store::nearest, project scope, 320 memories, top 5",
        &search,
        "none",
    );
    drop(store);

    // Project resolution: run on every request (B-109), in the hook's
    // folder and in this repository's work tree (it reads `.git`).
    for (folder, what) in [
        (proj.as_path(), "the hook's folder (not a git work tree)"),
        (env.repo.as_path(), "this repository's work tree"),
    ] {
        let mut resolve = Vec::new();
        for _ in 0..100 {
            let started = Instant::now();
            let _ = sm_core::project::resolve_project(folder).expect("resolve the project");
            resolve.push(started.elapsed());
        }
        figure(
            C,
            &format!("project resolution: resolve_project in {what}"),
            &resolve,
            "none",
        );
    }
}

/// Five warm-up hooks, then one per remaining prompt: the timed runs and
/// how many injected a block.
fn warm_hooks(home: &Home, proj: &Path, prompts: &[String]) -> (Vec<Duration>, usize) {
    let mut times = Vec::new();
    let mut injected = 0;
    for (i, p) in prompts.iter().enumerate() {
        let (t, inj) = hook_once(home, proj, p);
        if i >= 5 {
            times.push(t);
            injected += usize::from(inj);
        }
    }
    (times, injected)
}

/// `ambient_recall` straight to the daemon, one new session per prompt, for
/// every prompt after the five warm-up ones.
fn direct_recalls(home: &Home, proj: &Path, prompts: &[String], s: &Settings) -> Vec<Duration> {
    let mut direct = Vec::new();
    for p in &prompts[5..] {
        let o = op(json!({"op": "ambient_recall", "args": {
            "prompt": p, "top_k": s.ambient.top_k, "scope": s.ambient.scope,
        }}));
        let session = unique("d");
        let started = Instant::now();
        let r = home
            .try_call(proj, Some(&session), o)
            .expect("ambient_recall");
        direct.push(started.elapsed());
        assert!(matches!(r, Reply::Ambient(_)), "{r:?}");
    }
    direct
}

// ---------------------------------------------------------------------------
// 4. Hooks during a bulk import
// ---------------------------------------------------------------------------

fn import(env: &Env) {
    const C: &str = "import";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("hookproj");
    let texts = data::texts(&env.repo);
    seed(C, &home, &proj, &texts);
    let prompts = hook_prompts(env, &texts, 105);
    let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));
    for p in &prompts[..5] {
        let _ = hook_once(&home, &proj, p);
    }
    let base_names = procs::thread_names(d.pid);
    let base: usize = base_names.values().sum();
    let (physical, q, b, batch) = configured();
    note(
        C,
        format!(
            "before the import: {base} threads: {}",
            procs::names_line(&base_names)
        ),
    );
    note(
        C,
        format!(
            "configured: {q} query threads, {b} bulk threads, batch {batch} ({physical} physical cores)"
        ),
    );

    let export = home.root.join("v1_export_2000.json");
    let bytes = data::write_v1_export(&export, &texts, 2_000, "bench-import", 1);
    note(
        C,
        format!("a generated version 1 export of 2,000 records, {bytes} bytes"),
    );
    let export = export.to_string_lossy().into_owned();
    let import_dir = home.folder("importproj");
    let before_chunks = home.status().index.chunks;

    let sampler = Sampler::start(d.pid, Duration::from_millis(50));
    let started = Instant::now();
    let (migrate, times, during, embedded_at) = std::thread::scope(|sc| {
        let m = sc.spawn(|| home.run(&["migrate", "--from", &export], &import_dir));
        // The hooks start once the import's chunks are queued.
        loop {
            if home.status().index.pending_embeddings > 0 {
                break;
            }
            assert!(started.elapsed() < LONG, "the import queued its chunks");
            std::thread::sleep(Duration::from_millis(20));
        }
        let queued = started.elapsed();
        let mut times = Vec::new();
        let mut injected = 0;
        for p in &prompts[5..] {
            let (t, inj) = hook_once(&home, &proj, p);
            times.push(t);
            injected += usize::from(inj);
        }
        let hooks_done = started.elapsed();
        let pending_after = home.status().index.pending_embeddings;
        home.wait_embedded();
        let embedded_at = started.elapsed();
        (
            m.join().expect("the migrate thread"),
            times,
            (queued, hooks_done, pending_after, injected),
            embedded_at,
        )
    });
    let peak = sampler.stop();
    let (queued, hooks_done, pending_after, injected) = during;
    note(
        C,
        format!(
            "migrate exited {:?} after {:.1} s: {}",
            migrate.code,
            migrate.elapsed.as_secs_f64(),
            migrate.stdout.trim().replace('\n', " | ")
        ),
    );
    note(
        C,
        format!(
            "chunks queued at {:.1} s; the 100 hooks ran from then to {:.1} s; {pending_after} chunks still waiting when they finished; all embedded at {:.1} s",
            queued.as_secs_f64(),
            hooks_done.as_secs_f64(),
            embedded_at.as_secs_f64()
        ),
    );
    figure(
        C,
        "prompt hook during the bulk import (100 runs)",
        &times,
        "p95 <= 300 ms",
    );
    note(C, format!("{injected} of 100 hooks injected a block"));
    let chunks = home.status().index.chunks - before_chunks;
    let embed_secs = (embedded_at - queued).as_secs_f64();
    note(
        C,
        format!(
            "import: {chunks} chunks embedded in {embed_secs:.1} s from the first queued, {:.1} chunks/s, hooks running for part of it",
            chunks as f64 / embed_secs
        ),
    );
    memory(
        C,
        "peak while importing (sampled every 50 ms)",
        peak.max_working_set,
        peak.max_private,
        PEAK_BUDGET,
    );
    let end = d.read();
    memory(
        C,
        "the system's peak working set since start",
        end.peak_working_set,
        peak.max_private,
        PEAK_BUDGET,
    );
    let conns = peak.names_at_max.get("sm-conn").copied().unwrap_or(0);
    let limit = base + b;
    note(
        C,
        format!(
            "threads: {} samples; peak {} ({} of them sm-conn connection threads); before {base}; allowed before + bulk {b} = {limit}, plus connections in flight = {}: {}",
            peak.samples,
            peak.max_threads,
            conns,
            limit + conns,
            if peak.max_threads <= limit + conns {
                "within"
            } else {
                "OVER"
            }
        ),
    );
    note(
        C,
        format!(
            "  threads at the peak: {}",
            procs::names_line(&peak.names_at_max)
        ),
    );
    note(
        C,
        format!(
            "  added at the peak, by name: {}",
            name_diff(&base_names, &peak.names_at_max)
        ),
    );
    // The daemon's own threads (`sm-*`), sampled by name every 50 ms.
    let own_base = procs::own(&base_names);
    let own_conns = peak.names_at_max_own.get("sm-conn").copied().unwrap_or(0);
    let own_limit = own_base + b + own_conns;
    note(
        C,
        format!(
            "the daemon's own threads (sm-*): peak {} ({own_conns} sm-conn); before {own_base}; allowed before + bulk {b} + connections = {own_limit}: {}",
            peak.max_own,
            if peak.max_own <= own_limit {
                "within"
            } else {
                "OVER"
            }
        ),
    );
    note(
        C,
        format!(
            "  added at that peak, by name: {}",
            name_diff(&base_names, &peak.names_at_max_own)
        ),
    );
    let n = doctor_numbers(&home);
    note(C, format!("doctor_numbers after: {}", numbers_line(&n)));
    d.stop(&home);
}

/// The names whose count changed, `name +n` or `name -n`.
fn name_diff(before: &BTreeMap<String, usize>, after: &BTreeMap<String, usize>) -> String {
    let mut keys: Vec<&String> = before.keys().chain(after.keys()).collect();
    keys.sort();
    keys.dedup();
    let parts: Vec<String> = keys
        .into_iter()
        .filter_map(|k| {
            let a = before.get(k).copied().unwrap_or(0);
            let b = after.get(k).copied().unwrap_or(0);
            (a != b).then(|| format!("{k} {:+}", b as i64 - a as i64))
        })
        .collect();
    if parts.is_empty() {
        "none".to_owned()
    } else {
        parts.join(", ")
    }
}

// ---------------------------------------------------------------------------
// 5. Opening the bulk session (B-97)
// ---------------------------------------------------------------------------

fn open(env: &Env) {
    const C: &str = "open";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("openproj");
    let texts = data::texts(&env.repo);
    seed(C, &home, &proj, &texts[..64]);
    let prompts = hook_prompts(env, &texts, 60);

    // A: doctor_numbers with the session closed (it opens it), then open.
    let mut closed_t = Vec::new();
    let mut open_t = Vec::new();
    for _ in 0..3 {
        let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));
        let started = Instant::now();
        let _ = doctor_numbers(&home);
        closed_t.push(started.elapsed());
        let started = Instant::now();
        let _ = doctor_numbers(&home);
        open_t.push(started.elapsed());
    }
    figure(
        C,
        "doctor_numbers with the bulk session closed (opens it)",
        &closed_t,
        "none",
    );
    figure(
        C,
        "doctor_numbers with the bulk session open",
        &open_t,
        "none",
    );
    let (c50, _, _) = stats(&closed_t);
    let (o50, _, _) = stats(&open_t);
    note(
        C,
        format!(
            "time to open the bulk session, by difference of medians: {:.1} ms",
            c50 - o50
        ),
    );

    // B: hooks back to back while one remember opens the session.
    let mut before = Vec::new();
    let mut opening = Vec::new();
    let mut embedding = Vec::new();
    let mut after = Vec::new();
    for rep in 0..3 {
        let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));
        let batch: Vec<String> = texts[64 + rep * 16..64 + (rep + 1) * 16]
            .iter()
            .map(|t| format!("{t} (open case {rep})"))
            .collect();
        let stop = AtomicBool::new(false);
        let t0 = Instant::now();
        let (runs, trigger, first, done) = std::thread::scope(|sc| {
            let hooks = sc.spawn(|| {
                let mut v = Vec::new();
                let mut i = 0;
                while !stop.load(Ordering::SeqCst) {
                    let at = t0.elapsed();
                    let (t, _) = hook_once(&home, &proj, &prompts[i % prompts.len()]);
                    v.push((at, t));
                    i += 1;
                }
                v
            });
            std::thread::sleep(Duration::from_secs(1));
            let base = embedded(&home);
            let trigger = t0.elapsed();
            home.remember_all(&proj, &batch);
            let first = loop {
                if embedded(&home) > base {
                    break t0.elapsed();
                }
                std::thread::sleep(Duration::from_millis(10));
            };
            home.wait_embedded();
            let done = t0.elapsed();
            std::thread::sleep(Duration::from_secs(2));
            stop.store(true, Ordering::SeqCst);
            (
                hooks.join().expect("the hooks thread"),
                trigger,
                first,
                done,
            )
        });
        note(
            C,
            format!(
                "rep {}: remember at {:.0} ms, first chunk embedded at {:.0} ms (+{:.0} ms: open and one batch), all 16 at {:.0} ms",
                rep + 1,
                ms(trigger),
                ms(first),
                ms(first - trigger),
                ms(done)
            ),
        );
        for (at, t) in runs {
            if at < trigger {
                before.push(t);
            } else if at < first {
                opening.push(t);
            } else if at < done {
                embedding.push(t);
            } else {
                after.push(t);
            }
        }
    }
    figure(C, "hooks before (session closed)", &before, "none");
    figure(
        C,
        "hooks started while the session was opening",
        &opening,
        "p95 <= 300 ms",
    );
    figure(
        C,
        "hooks started while its chunks were embedding",
        &embedding,
        "p95 <= 300 ms",
    );
    figure(C, "hooks after (session open, idle)", &after, "none");
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 6. The daemon killed during a prompt hook
// ---------------------------------------------------------------------------

fn kill(env: &Env) {
    const C: &str = "kill";
    let home = Home::new(env, C);
    let proj = home.folder("killproj");
    let texts = data::texts(&env.repo);
    let prompts = hook_prompts(env, &texts, 30);
    {
        let mut d = home.start_ready();
        seed(C, &home, &proj, &texts);
        d.stop(&home);
    }

    // Calibration: the hook without contact, and the hook warm.
    let mut d = home.start_ready();
    let mut off = Vec::new();
    let mut warm = Vec::new();
    for p in &prompts[..6] {
        let input = Home::prompt_input(&proj, p, &unique("s"));
        let run = home
            .spawn(HOOK_ARGV, &proj, &[(AMBIENT_ENV, "off")], &input)
            .finish();
        off.push(run.elapsed);
        warm.push(hook_once(&home, &proj, p).0);
    }
    let t_proc = median(&off[1..]);
    let t_hook = median(&warm[1..]);
    let delay = t_proc + t_hook.saturating_sub(t_proc) / 2;
    note(
        C,
        format!(
            "calibration: the hook without contact {:.1} ms, warm {:.1} ms; each kill lands {:.1} ms after the hook starts",
            ms(t_proc),
            ms(t_hook),
            ms(delay)
        ),
    );
    d.stop(&home);

    // Each run: a fresh daemon, one warm hook, then the measured hook with
    // the daemon killed `delay` after it starts. A run whose answer came
    // before the kill measured nothing; it is counted and another is run.
    let mut wall = Vec::new();
    let mut after_kill = Vec::new();
    let mut respawned = 0;
    let mut late = 0;
    let mut i = 0;
    while wall.len() < 20 && i < 40 {
        let mut d = start_retrying(&home);
        let _ = hook_once(&home, &proj, &prompts[0]);
        let running = home.spawn_prompt_hook(&proj, &prompts[1 + i % 29], &unique("s"));
        let started = running.started;
        std::thread::sleep(delay);
        let killed = Instant::now();
        d.kill();
        let run = running.finish();
        i += 1;
        assert_eq!(run.code, Some(0), "run {i}: {}", run.stderr);
        if !run.stdout.is_empty() {
            late += 1;
            continue;
        }
        wall.push(run.elapsed);
        after_kill.push((started + run.elapsed).saturating_duration_since(killed));
        // A hook that found nothing listening starts a daemon (CONTRACTS
        // 3.1): the kill landed before it connected.
        std::thread::sleep(Duration::from_millis(500));
        if sm_cli::client::is_listening(&home.endpoint) {
            respawned += 1;
            home.shut_down_any();
        }
    }
    figure(
        C,
        "prompt hook with its daemon killed mid-request, wall time",
        &wall,
        "every run <= 1,500 ms, exit 0, no output",
    );
    figure(C, "kill to hook exit", &after_kill, "none");
    let worst = wall.iter().max().copied().unwrap_or_default();
    note(
        C,
        format!(
            "worst {:.1} ms: {}; {respawned} of {} kills landed before the hook connected (it then started a daemon, which was shut down); {late} runs answered before the kill and were not counted",
            ms(worst),
            if worst <= DEADLINE { "within" } else { "OVER" },
            wall.len()
        ),
    );
    note(
        C,
        format!("gates (warm-up runs included): {:?}", home.gates()),
    );
}

/// Starts the daemon; a daemon a hook started is shut down first.
fn start_retrying(home: &Home) -> crate::harness::Daemon {
    for _ in 0..3 {
        home.shut_down_any();
        match home.start() {
            Ok(d) if d.info.state == DaemonState::Ready => return d,
            Ok(mut d) => {
                d.kill();
                panic!("the daemon is not ready: {:?}", d.info.state_error);
            }
            Err(e) => note("kill", format!("retrying a start: {e}")),
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    panic!("the daemon did not start in three tries")
}

// ---------------------------------------------------------------------------
// 7. Remember
// ---------------------------------------------------------------------------

fn remember(env: &Env) {
    const C: &str = "remember";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("rememberproj");
    let texts = data::texts(&env.repo);
    let tok = tokenizer(env);
    let items: Vec<String> = (0..23)
        .map(|i| format!("{} {} (remember run {i})", texts[i * 3], texts[i * 3 + 1]))
        .collect();
    let tokens: Vec<usize> = items
        .iter()
        .map(|t| tok.count(t).expect("count tokens"))
        .collect();
    note(
        C,
        format!(
            "texts of {} to {} model tokens (one chunk each)",
            tokens.iter().min().unwrap_or(&0),
            tokens.iter().max().unwrap_or(&0)
        ),
    );
    let rt = mcp::runtime();
    let (warm, cold, replies, first_open) = rt.block_on(async {
        let client = mcp::connect(&home, &proj).await;
        let first_open = bulk_open(d.pid);
        let mut warm = Vec::new();
        let mut cold = Vec::new();
        let mut replies = Vec::new();
        for (i, text) in items.iter().enumerate() {
            if i >= 20 {
                let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));
            }
            let (found, reply, chunks, pending) = remember_until_found(&home, &client, text).await;
            assert_eq!(chunks, 1, "one chunk");
            if i == 0 {
                note(
                    C,
                    format!(
                        "the reply came before embedding: {pending} chunk(s) waiting when it returned (queued, B-27)"
                    ),
                );
            }
            replies.push(reply);
            if i < 20 {
                warm.push(found);
            } else {
                cold.push(found);
            }
        }
        client.cancel().await.expect("close the MCP session");
        (warm, cold, replies, first_open)
    });
    note(
        C,
        format!("bulk session open before the first remember: {first_open}"),
    );
    figure(C, "remember reply over MCP", &replies, "none");
    figure(
        C,
        "remember until found by vector recall, 20 in a row (the first opens the bulk session)",
        &warm,
        "each <= 2,000 ms",
    );
    note(
        C,
        format!(
            "first of the 20 (bulk session closed): {:.1} ms",
            ms(warm[0])
        ),
    );
    figure(
        C,
        "remember until found, each after the bulk grace (session closed)",
        &cold,
        "each <= 2,000 ms",
    );
    d.stop(&home);
}

/// Remembers `text` over MCP, then waits until nothing waits for embedding
/// (status every 10 ms) and a vector recall returns it. Returns the time to
/// found, the reply time, the chunk count and the chunks waiting at reply.
async fn remember_until_found(
    home: &Home,
    client: &mcp::Client,
    text: &str,
) -> (Duration, Duration, u64, u64) {
    let started = Instant::now();
    let r = mcp::call(client, "remember", json!({"memories": [{"content": text}]})).await;
    let reply = started.elapsed();
    let m = &r["memories"][0];
    assert_eq!(m["outcome"], "created", "{r}");
    let id = m["id"].as_str().expect("an id").to_owned();
    let chunks = m["chunks"].as_u64().unwrap_or(0);
    let pending = home.status().index.pending_embeddings;
    loop {
        while home.status().index.pending_embeddings > 0 {
            assert!(started.elapsed() < LONG, "embedded in time");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let rr = mcp::call(
            client,
            "recall",
            json!({"query": text, "k": 5, "mode": "vector"}),
        )
        .await;
        let found = rr["results"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x["id"].as_str() == Some(&id)));
        if found {
            return (started.elapsed(), reply, chunks, pending);
        }
        assert!(started.elapsed() < LONG, "found in time");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// ---------------------------------------------------------------------------
// 8. AT-4: a 50,000-character memory
// ---------------------------------------------------------------------------

fn at4(env: &Env) {
    const C: &str = "at4";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("at4proj");
    let text = data::long_memory(50_000);
    let before = d.read();
    note(C, format!("before: {}", reading_line(&before)));
    let sampler = Sampler::start(d.pid, Duration::from_millis(50));
    let started = Instant::now();
    let rt = mcp::runtime();
    let (chunks, outcome) = rt.block_on(async {
        let client = mcp::connect(&home, &proj).await;
        let r = mcp::call(
            &client,
            "remember",
            json!({"memories": [{"content": text}]}),
        )
        .await;
        client.cancel().await.expect("close the MCP session");
        let m = &r["memories"][0];
        (
            m["chunks"].as_u64().unwrap_or(0),
            m["outcome"].as_str().unwrap_or("?").to_owned(),
        )
    });
    let reply = started.elapsed();
    home.wait_embedded();
    let done = started.elapsed();
    let peak = sampler.stop();
    let after = d.read();
    note(
        C,
        format!(
            "{} characters: outcome {outcome}, {chunks} chunks; reply {:.0} ms, all embedded {:.1} s ({:.1} chunks/s); multiple chunks: {}",
            text.chars().count(),
            ms(reply),
            done.as_secs_f64(),
            chunks as f64 / done.as_secs_f64(),
            chunks > 1
        ),
    );
    memory(
        C,
        "peak while indexing (sampled every 50 ms)",
        peak.max_working_set,
        peak.max_private,
        PEAK_BUDGET,
    );
    memory(
        C,
        "the system's peak working set since start",
        after.peak_working_set,
        peak.max_private,
        PEAK_BUDGET,
    );
    note(
        C,
        format!(
            "threads at most {}: {}",
            peak.max_threads,
            procs::names_line(&peak.names_at_max)
        ),
    );
    note(C, format!("after: {}", reading_line(&after)));
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 9. Recall at 100,000 chunks (B-75, B-114)
// ---------------------------------------------------------------------------

const RECALL_CHUNKS: usize = 100_000;
const CHUNKS_PER_MEMORY: usize = 2;

fn recall100k(env: &Env) {
    const C: &str = "recall100k";
    let texts = data::texts(&env.repo);
    for (layout, projects, probe, seed) in [("spread", 60, 17, 1), ("single", 1, 0, 4)] {
        let home = Home::new(env, C);
        // The daemon's own stamp, from a start on the empty profile.
        let model = {
            let mut d = home.start_ready();
            let s = home.status();
            d.stop(&home);
            s.model
        };
        let folders: Vec<PathBuf> = (0..projects)
            .map(|i| home.folder(&format!("p{i}")))
            .collect();
        let started = Instant::now();
        fill(&home, &folders, &model, seed);
        note(
            C,
            format!(
                "{layout}: {RECALL_CHUNKS} chunks ({} memories) in {projects} project(s), random unit vectors, stamped {} {}, built in {:.1} s",
                RECALL_CHUNKS / CHUNKS_PER_MEMORY,
                model.model_id,
                model.revision,
                started.elapsed().as_secs_f64()
            ),
        );
        let db_bytes = std::fs::metadata(home.paths.database()).map_or(0, |m| m.len());
        let mut d = home.start_ready();
        let st = home.status();
        note(
            C,
            format!(
                "{layout}: the daemon sees {} chunks, {} waiting, index matches: {}; database file {}; {}",
                st.index.chunks,
                st.index.pending_embeddings,
                st.index.matches,
                procs::mib(db_bytes),
                reading_line(&d.read())
            ),
        );
        assert!(st.index.matches, "the filled store matches the daemon");
        if layout == "spread" {
            let tok = tokenizer(env);
            let mut counts: Vec<usize> = texts[10..310]
                .iter()
                .map(|t| tok.spans(t).expect("tokenize a query").len())
                .collect();
            counts.sort_unstable();
            note(
                C,
                format!(
                    "queries: the spatial fixture's texts, {} to {} model tokens, median {}",
                    counts[0],
                    counts[counts.len() - 1],
                    counts[counts.len() / 2]
                ),
            );
        }
        for (scope, budget) in [("project", "p95 <= 250 ms"), ("all", "p95 <= 350 ms")] {
            let mut times = Vec::new();
            let mut returned = 0;
            for (i, q) in texts[..310].iter().enumerate() {
                let mut args = json!({"query": q, "k": 5});
                if scope == "all" {
                    args["scope"] = json!("all");
                }
                let started = Instant::now();
                let r = home.call(&folders[probe], op(json!({"op": "recall", "args": args})));
                let took = started.elapsed();
                let Reply::Recall(rr) = r else {
                    panic!("recall: {r:?}")
                };
                if i >= 10 {
                    times.push(took);
                    returned += rr.results.len();
                }
            }
            let what = if layout == "spread" && scope == "project" {
                format!(
                    "{layout}: recall, project scope (one of 60 projects, ~1,667 chunks), query embedded"
                )
            } else if scope == "project" {
                format!(
                    "{layout}: recall, project scope (the project holds all 100,000 chunks), query embedded"
                )
            } else {
                format!("{layout}: recall, all projects, query embedded")
            };
            figure(C, &what, &times, budget);
            note(
                C,
                format!(
                    "  {} results over {} queries (300 after 10 warm-up)",
                    returned,
                    times.len()
                ),
            );
            let r = d.read();
            memory(
                C,
                &format!("{layout}: after the {scope} queries"),
                r.working_set,
                r.private,
                IDLE_BUDGET,
            );
            note(C, format!("  {}", reading_line(&r)));
        }
        d.stop(&home);
    }
}

/// Fills the profile's store directly with the daemon's stamp: the memories
/// round robin over the folders' projects, two chunks each.
fn fill(home: &Home, folders: &[PathBuf], model: &ModelStatus, seed: u64) {
    let store = open_store(home, model);
    let ids: Vec<ProjectId> = folders
        .iter()
        .map(|f| {
            let identity = sm_core::project::resolve_project(f).expect("resolve a project");
            store.upsert_project(&identity).expect("a project").id
        })
        .collect();
    let mut rng = Rng(seed);
    let memories = RECALL_CHUNKS / CHUNKS_PER_MEMORY;
    let mut n = 0;
    while n < memories {
        let end = (n + 1_000).min(memories);
        let batch = (n..end)
            .map(|m| stored_memory(&mut rng, ids[m % ids.len()], m))
            .collect();
        store
            .insert_memories(batch, Access::Local)
            .expect("insert memories");
        n = end;
    }
}

fn stored_memory(rng: &mut Rng, project: ProjectId, n: usize) -> MemoryInput {
    let chunks: Vec<NewChunk> = (0..CHUNKS_PER_MEMORY)
        .map(|c| {
            let text = format!("memory {n} part {c} token{}", rng.next_u64() % 10_000);
            NewChunk::embedded(
                ChunkPiece {
                    seq: c as u32,
                    token_count: text.split_whitespace().count() as u32,
                    text,
                },
                rng.unit_vector(),
            )
        })
        .collect();
    let content = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    MemoryInput {
        memory: NewMemory::new(project, content, Provenance::new(ProvenanceSource::Agent)),
        chunks,
    }
}

// ---------------------------------------------------------------------------
// 10. Consolidate at 2,000 memories (B-103)
// ---------------------------------------------------------------------------

fn consolidate(env: &Env) {
    const C: &str = "consolidate";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("consolidateproj");
    let texts = data::texts(&env.repo);
    let items = data::consolidate_texts(&texts, 2_000);
    let started = Instant::now();
    seed(C, &home, &proj, &items);
    let chunks = home.status().index.chunks;
    let secs = started.elapsed().as_secs_f64();
    note(
        C,
        format!(
            "remember and embed: {chunks} chunks in {secs:.1} s, {:.1} chunks/s end to end",
            chunks as f64 / secs
        ),
    );
    let mut times = Vec::new();
    for i in 0..3 {
        let started = Instant::now();
        let r = home.call(
            &proj,
            op(json!({"op": "consolidate", "args": {"apply": false}})),
        );
        let took = started.elapsed();
        let Reply::Consolidate(c) = r else {
            panic!("consolidate: {r:?}")
        };
        let dups: usize = c.proposals.iter().map(|g| g.duplicates.len()).sum();
        note(
            C,
            format!(
                "run {}: {:.0} ms, {} groups proposed ({dups} duplicates), truncated {}, omitted {}",
                i + 1,
                ms(took),
                c.proposals.len(),
                c.truncated,
                c.omitted
            ),
        );
        times.push(took);
    }
    figure(
        C,
        "consolidate, proposals only, 2,000 memories in one project",
        &times,
        "none (B-103)",
    );
    let r = d.read();
    note(C, format!("after: {}", reading_line(&r)));
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 11. AT-2: doctor --eval
// ---------------------------------------------------------------------------

fn at2(env: &Env) {
    const C: &str = "at2";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let set = env.fixture("retrieval/retrieval_eval_set.json");
    let baseline = env.fixture("retrieval/retrieval_quality_baseline.json");
    let set = set.to_string_lossy().into_owned();
    let baseline = baseline.to_string_lossy().into_owned();
    let cwd = home.folder("at2");
    let run = home.run(&["doctor", "--eval", &set, &baseline], &cwd);
    note(
        C,
        format!(
            "spatial-memory doctor --eval <set> <baseline>: exit {:?} in {:.1} s",
            run.code,
            run.elapsed.as_secs_f64()
        ),
    );
    for line in run.stdout.lines().chain(run.stderr.lines()) {
        note(C, format!("  | {line}"));
    }
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 12. AT-7 live
// ---------------------------------------------------------------------------

fn at7(env: &Env) {
    const C: &str = "at7";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("at7proj");
    let texts = data::texts(&env.repo);
    seed(C, &home, &proj, &texts);
    let started = Instant::now();
    let r = home.call(&proj, op(json!({"op": "regions", "args": {}})));
    let took = started.elapsed();
    let Reply::Regions(rr) = r else {
        panic!("regions: {r:?}")
    };
    let n = f64::from(rr.memories.max(1));
    let largest = rr.regions.iter().map(|g| g.size).max().unwrap_or(0);
    let unclustered = 100.0 * f64::from(rr.unclustered) / n;
    let largest_pct = 100.0 * f64::from(largest) / n;
    note(
        C,
        format!(
            "regions over {} memories in {:.0} ms: {} regions, unclustered {} ({unclustered:.1}%), largest {} ({largest_pct:.1}%), truncated {}",
            rr.memories,
            ms(took),
            rr.regions.len(),
            rr.unclustered,
            largest,
            rr.truncated
        ),
    );
    note(
        C,
        format!(
            "AT-7: unclustered <= 25%: {}; no region over 40%: {}; the committed result was 9 regions, 7.8%, 17.8%",
            if unclustered <= 25.0 { "pass" } else { "FAIL" },
            if largest_pct <= 40.0 { "pass" } else { "FAIL" }
        ),
    );
    let sizes: Vec<String> = rr.regions.iter().map(|g| g.size.to_string()).collect();
    note(C, format!("region sizes: {}", sizes.join(", ")));
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 13. AT-8: migrate the version 1 fixture
// ---------------------------------------------------------------------------

fn at8(env: &Env) {
    const C: &str = "at8";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let cwd = home.folder("at8");
    let src = env.fixture("migration/v1_export_full.json");
    let src = src.to_string_lossy().into_owned();
    let started = Instant::now();
    let run = home.run(&["migrate", "--from", &src], &cwd);
    note(
        C,
        format!(
            "spatial-memory migrate --from fixtures/migration/v1_export_full.json: exit {:?} in {:.0} ms",
            run.code,
            ms(run.elapsed)
        ),
    );
    for line in run.stdout.lines().chain(run.stderr.lines()) {
        note(C, format!("  | {line}"));
    }
    let mut polls = 0;
    let status = loop {
        let s = home.run(&["status"], &cwd);
        polls += 1;
        if s.stdout.contains(", 0 waiting for embedding") {
            break s;
        }
        assert!(started.elapsed() < LONG, "embedded in time");
        std::thread::sleep(Duration::from_millis(50));
    };
    let total = started.elapsed();
    let st = home.status();
    note(
        C,
        format!(
            "status said 0 waiting after {:.0} ms ({polls} status runs): {} chunks, {:.1} chunks/s from the migrate's start",
            ms(total),
            st.index.chunks,
            st.index.chunks as f64 / total.as_secs_f64()
        ),
    );
    for line in status.stdout.lines() {
        note(C, format!("  | {line}"));
    }
    let memories: u64 = st.projects.iter().map(|p| p.memories).sum();
    note(
        C,
        format!(
            "projects: {}",
            st.projects
                .iter()
                .map(|p| format!("{} {}", p.name, p.memories))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    );
    note(C, format!("memories in the store: {memories}"));
    d.stop(&home);
}

// ---------------------------------------------------------------------------
// 14. AT-10: a model file with one changed byte
// ---------------------------------------------------------------------------

fn at10(env: &Env) {
    const C: &str = "at10";
    let mut home = Home::new(env, C);
    let copy = home.root.join("model-copy");
    for f in sm_embed::MODEL_FILES {
        let dst = f.local_path(&copy);
        std::fs::create_dir_all(dst.parent().expect("a parent")).expect("a folder for the copy");
        std::fs::copy(f.local_path(&env.model_dir), &dst).expect("copy a model file");
    }
    let target = sm_embed::model::MODEL_DATA.local_path(&copy);
    let at = flip_byte(&target);
    note(
        C,
        format!(
            "copied the 3 model files; flipped the byte at offset {at} of {}",
            sm_embed::model::MODEL_DATA.path
        ),
    );
    home.model_dir = copy;
    match home.start() {
        Ok(mut d) => {
            note(
                C,
                format!(
                    "the daemon answered {:?} after {:.1} s: {:?}",
                    d.info.state,
                    d.ready.as_secs_f64(),
                    d.info.state_error
                ),
            );
            note(
                C,
                format!(
                    "AT-10: not ready: {}",
                    if d.info.state == DaemonState::Failed {
                        "pass"
                    } else {
                        "FAIL"
                    }
                ),
            );
            let log = std::fs::read_to_string(home.paths.daemon_log()).unwrap_or_default();
            for line in log.lines().filter(|l| l.contains("model")) {
                note(C, format!("  daemon.log | {line}"));
            }
            let cwd = home.folder("at10");
            let run = home.run(&["doctor"], &cwd);
            note(C, format!("spatial-memory doctor: exit {:?}", run.code));
            for line in run.stdout.lines().chain(run.stderr.lines()) {
                note(C, format!("  | {line}"));
            }
            d.stop(&home);
        }
        Err(e) => note(C, format!("the daemon did not answer ready or failed: {e}")),
    }
}

/// Inverts one byte in the middle of `path`; its offset.
fn flip_byte(path: &Path) -> u64 {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open the copy");
    let at = f.metadata().expect("its size").len() / 2;
    let mut b = [0u8; 1];
    f.seek(SeekFrom::Start(at)).expect("seek");
    f.read_exact(&mut b).expect("read a byte");
    b[0] = !b[0];
    f.seek(SeekFrom::Start(at)).expect("seek");
    f.write_all(&b).expect("write the byte");
    f.sync_all().expect("sync the copy");
    at
}

// ---------------------------------------------------------------------------
// 16. Doctor
// ---------------------------------------------------------------------------

fn doctor(env: &Env) {
    const C: &str = "doctor";
    let home = Home::new(env, C);
    let mut d = home.start_ready();
    let proj = home.folder("doctorproj");
    let texts = data::texts(&env.repo);
    seed(C, &home, &proj, &texts);
    let prompts = hook_prompts(env, &texts, 30);
    let cwd = home.folder("doctorcwd");
    let mut times = Vec::new();
    for i in 0..3 {
        let run = home.run(&["doctor"], &cwd);
        times.push(run.elapsed);
        note(
            C,
            format!(
                "run {}: exit {:?} in {:.0} ms",
                i + 1,
                run.code,
                ms(run.elapsed)
            ),
        );
        for line in run.stdout.lines().chain(run.stderr.lines()) {
            if i == 0 || line.contains("chunks/s") {
                note(C, format!("  | {line}"));
            }
        }
    }
    figure(C, "spatial-memory doctor, end to end", &times, "none");
    for i in 0..3 {
        let n = doctor_numbers(&home);
        note(C, format!("doctor_numbers {}: {}", i + 1, numbers_line(&n)));
    }

    // 20 prompt hooks while doctor runs (repeated until the hooks are done).
    let _ = wait_bulk_closed(d.pid, Duration::from_secs(90));
    let stop = AtomicBool::new(false);
    let (hooks, doctors) = std::thread::scope(|sc| {
        let doctors = sc.spawn(|| {
            let mut runs: Vec<(Duration, Duration)> = Vec::new();
            let t0 = Instant::now();
            while !stop.load(Ordering::SeqCst) {
                let at = t0.elapsed();
                let run = home.run(&["doctor"], &cwd);
                runs.push((at, at + run.elapsed));
            }
            (t0, runs)
        });
        // Let the first doctor reach its numbers.
        std::thread::sleep(Duration::from_millis(300));
        let mut v: Vec<(Instant, Duration)> = Vec::new();
        for p in &prompts[..20] {
            let at = Instant::now();
            let (t, _) = hook_once(&home, &proj, p);
            v.push((at, t));
        }
        stop.store(true, Ordering::SeqCst);
        (v, doctors.join().expect("the doctor thread"))
    });
    let (t0, runs) = doctors;
    let inside = hooks
        .iter()
        .filter(|(at, t)| {
            let s = at.saturating_duration_since(t0);
            let e = s + *t;
            runs.iter().any(|(a, b)| s < *b && e > *a)
        })
        .count();
    let times: Vec<Duration> = hooks.iter().map(|(_, t)| *t).collect();
    figure(
        C,
        "prompt hooks while doctor runs (20)",
        &times,
        "p95 <= 300 ms",
    );
    note(
        C,
        format!(
            "{} doctor runs in that time; {inside} of 20 hooks overlapped a doctor run; worst {:.1} ms",
            runs.len(),
            ms(times.iter().max().copied().unwrap_or_default())
        ),
    );
    d.stop(&home);
}
