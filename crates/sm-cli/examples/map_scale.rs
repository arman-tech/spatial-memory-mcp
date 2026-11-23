//! The map at scale: a measuring spike (B-261, SPEC 7).
//!
//! ```text
//! cargo build --release -p sm-cli --bins --examples
//! target/release/examples/map_scale <step>...
//! ```
//!
//! Steps: `prepare` (the synthetic stores), `cost`, `analyse`, `cache`,
//! `alt7a`, `alt7b`, `alt7c`, `all` (every step above, in that order), and
//! `real <copy of memory.db> <backups folder>` for a read-only copy of a
//! live store. Each measurement runs in its own child process, one at a
//! time, so a stage's memory is its own. Results are printed and appended
//! to `results.jsonl` in the work folder (`SM_MAP_SCALE_DIR`, or
//! `sm-map-scale` under the temporary folder), which also holds the
//! synthetic stores.
//!
//! Figures come by three paths, named in every result: the library
//! (`sm_spatial::regions` and `coordinates`), a replica of the library's
//! pipeline stage by stage (`map_scale/pipe.rs`, checked bit for bit
//! against the library on the first run of each size), and the release
//! daemon on a scratch profile (`cache-daemon`). No model is loaded and no
//! model call is made: the synthetic vectors stand in for embeddings, and
//! the daemon is never asked to embed.

// A crate root looks for its modules beside it; these live in `budgets/`
// and `map_scale/`.
#[path = "map_scale/cases.rs"]
mod cases;
#[allow(dead_code)]
#[path = "budgets/harness.rs"]
mod harness;
#[allow(dead_code)]
#[path = "map_scale/meter.rs"]
mod meter;
#[allow(dead_code)]
#[path = "map_scale/pipe.rs"]
mod pipe;
#[allow(dead_code)]
#[path = "budgets/procs.rs"]
mod procs;
#[path = "map_scale/study.rs"]
mod study;
#[allow(dead_code)]
#[path = "map_scale/synth.rs"]
mod synth;

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use serde_json::{Value, json};

/// How long one larger-sample run may take before the rest are skipped.
const ALT7A_LIMIT: Duration = Duration::from_secs(45 * 60);
const REPS: usize = 3;

fn main() -> ExitCode {
    disinherit_own_std_handles();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("child") {
        meter::start_sampler();
        return run_child(&args[1..]);
    }
    if args.is_empty() {
        println!(
            "usage: map_scale <prepare|cost|analyse|cache|alt7a|alt7b|alt7c|all|real DB BACKUPS>..."
        );
        return ExitCode::from(2);
    }
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "prepare" => prepare(),
            "cost" => cost(),
            "analyse" => analyse(),
            "cache" => cache(),
            "alt7a" => alt7a(),
            "alt7b" => alt7b(),
            "alt7c" => alt7c(),
            "all" => {
                prepare();
                cost();
                analyse();
                cache();
                alt7a();
                alt7b();
                alt7c();
            }
            "real" => {
                let (Some(db), Some(backups)) = (args.get(i + 1), args.get(i + 2)) else {
                    println!("real needs the copy's memory.db and a backups folder");
                    return ExitCode::from(2);
                };
                real(db, backups);
                i += 2;
            }
            other => {
                println!("unknown step {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    ExitCode::SUCCESS
}

/// The standard handles are inheritable (they came from the shell); a
/// daemon started from here would hold them open and keep the shell
/// waiting. `Stdio::inherit` still works (see `client::disinherit`).
fn disinherit_own_std_handles() {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        sm_cli::client::disinherit(&[
            std::io::stdin().as_raw_handle(),
            std::io::stdout().as_raw_handle(),
            std::io::stderr().as_raw_handle(),
        ]);
    }
}

fn run_child(args: &[String]) -> ExitCode {
    let num = |i: usize| -> usize {
        args.get(i)
            .and_then(|s| s.parse().ok())
            .expect("a number argument")
    };
    match args.first().map(String::as_str) {
        Some("prepare") => cases::prepare_db(num(1)),
        Some("cost-lib") => cases::cost_lib(num(1)),
        Some("cost-stages") => cases::cost_stages(num(1), num(2) == 1),
        Some("analyse") => study::analyse(num(1)),
        Some("cache-lib") => cases::cache_lib(num(1)),
        Some("cache-daemon") => cases::cache_daemon(num(1)),
        Some("alt7a") => study::alt7a(num(1), num(2)),
        Some("alt7b") => study::alt7b(),
        Some("alt7c") => study::alt7c(num(1)),
        Some("real") => {
            let db = PathBuf::from(args.get(1).expect("the copy"));
            let backups = PathBuf::from(args.get(2).expect("a backups folder"));
            study::real(&db, &backups);
        }
        other => {
            println!("unknown child case {other:?}");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

/// Appends results to `results.jsonl`, each with its step and the time.
fn record(step: &str, results: &[Value]) {
    let dir = cases::work_dir();
    let _ = std::fs::create_dir_all(&dir);
    let at = sm_core::time::Timestamp::now().unix_seconds();
    let mut text = String::new();
    for r in results {
        let line = json!({"step": step, "at": at, "result": r});
        text.push_str(&line.to_string());
        text.push('\n');
    }
    // One write call for the whole batch.
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("results.jsonl"))
    {
        let _ = f.write_all(text.as_bytes());
    }
}

fn run(step: &str, args: &[&str]) -> Vec<Value> {
    run_timed(step, args, None).0
}

fn run_timed(step: &str, args: &[&str], limit: Option<Duration>) -> (Vec<Value>, bool) {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    println!("== {step}: {}", owned.join(" "));
    let (results, killed) = meter::child_timed(&owned, limit);
    record(step, &results);
    if killed {
        record(
            step,
            &[json!({"case": "killed", "args": owned, "limit_s": limit.map(|l| l.as_secs())})],
        );
    }
    (results, killed)
}

fn prepare() {
    for n in cases::SIZES {
        let r = run("prepare", &["prepare", &n.to_string()]);
        for v in r {
            println!("  {v}");
        }
    }
}

fn cost() {
    for n in cases::SIZES {
        let ns = n.to_string();
        let mut lib = Vec::new();
        let mut stages = Vec::new();
        for rep in 0..REPS {
            lib.extend(run("cost", &["cost-lib", &ns]));
            let verify = if rep == 0 { "1" } else { "0" };
            let s = run("cost", &["cost-stages", &ns, verify]);
            for v in &s {
                if !v["verify"].is_null() {
                    println!("  verify: {}", v["verify"]);
                }
            }
            stages.extend(s);
        }
        meter::stage_table(&format!("library, {n} memories"), &lib);
        meter::stage_table(&format!("stages (replica), {n} memories"), &stages);
    }
}

fn analyse() {
    for n in cases::SIZES {
        for v in run("analyse", &["analyse", &n.to_string()]) {
            println!("  quality {}", v["quality"]);
            println!("  neighbours {}", v["neighbours"]);
            println!("  clumping {}", v["clumping"]);
            println!("  payload {}", v["payload"]);
        }
    }
}

fn cache() {
    let mut lib = Vec::new();
    for rep in 0..REPS {
        lib.extend(run("cache", &["cache-lib", &rep.to_string()]));
    }
    for v in &lib {
        println!("  {v}");
    }
    for rep in 0..REPS {
        for v in run("cache", &["cache-daemon", &rep.to_string()]) {
            println!("  {v}");
        }
    }
}

fn alt7a() {
    let mut stop = false;
    for s in [5_000usize, 10_000, 20_000] {
        if stop {
            record(
                "alt7a",
                &[
                    json!({"case": "alt7a", "s": s, "skipped": "a smaller sample ran past the limit"}),
                ],
            );
            println!("== alt7a: sample {s} skipped, a smaller sample ran past the limit");
            continue;
        }
        let mut runs = Vec::new();
        for rep in 0..REPS {
            let (r, killed) = run_timed(
                "alt7a",
                &["alt7a", &s.to_string(), &rep.to_string()],
                Some(ALT7A_LIMIT),
            );
            runs.extend(r);
            if killed {
                stop = true;
                break;
            }
        }
        meter::stage_table(&format!("sample of {s} from 100,000 (replica)"), &runs);
        for v in &runs {
            if !v["quality"].is_null() {
                println!("  quality {}", v["quality"]);
                println!("  neighbours {}", v["neighbours"]);
                println!("  clumping {}", v["clumping"]);
            }
        }
    }
}

fn alt7b() {
    for v in run("alt7b", &["alt7b"]) {
        println!("  {v}");
    }
}

fn alt7c() {
    let mut runs = Vec::new();
    for rep in 0..REPS {
        runs.extend(run("alt7c", &["alt7c", &rep.to_string()]));
    }
    meter::stage_table("regions inside regions, 100,000 (library)", &runs);
    for v in &runs {
        if !v["quality"].is_null() {
            println!("  quality {}", v["quality"]);
        }
    }
}

fn real(db: &str, backups: &str) {
    for v in run("real", &["real", db, backups]) {
        println!("  {v}");
    }
}
