//! Real-model measurements of the SPEC 11 budgets (stream K, Part 2).
//!
//! ```text
//! cargo build --release -p sm-cli --bins --examples
//! cargo run --release -p sm-cli --example budgets -- <case>... | all
//! ```
//!
//! Each case runs the release `spatial-memory` binary in a fresh temporary
//! `SPATIAL_MEMORY_HOME`, removed at its end, with the model read from the
//! per-user cache (or `SPATIAL_MEMORY_MODEL_DIR`). Without the model it
//! prints one line and exits 0. It never downloads anything and never
//! touches the user's profile or client settings. Times are printed, never
//! asserted (B-58); memory budgets print pass or FAIL on the working set
//! (B-114).

// A crate root looks for its modules beside it; these live in `budgets/`.
#[path = "budgets/cases.rs"]
mod cases;
#[path = "budgets/data.rs"]
mod data;
#[path = "budgets/harness.rs"]
mod harness;
#[path = "budgets/mcp.rs"]
mod mcp;
#[path = "budgets/procs.rs"]
mod procs;

use std::process::ExitCode;

use sm_core::config::Settings;

fn main() -> ExitCode {
    disinherit_own_std_handles();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env = match harness::Env::find() {
        Ok(env) => env,
        Err(line) => {
            println!("{line}");
            return ExitCode::SUCCESS;
        }
    };
    let names: Vec<&str> = if args.iter().any(|a| a == "all") {
        cases::NAMES.to_vec()
    } else {
        args.iter().map(String::as_str).collect()
    };
    if names.is_empty() || names.iter().any(|n| !cases::NAMES.contains(n)) {
        println!(
            "usage: budgets <case>... | all\ncases: {}",
            cases::NAMES.join(", ")
        );
        return ExitCode::from(2);
    }
    header(&env);
    for name in names {
        println!();
        cases::run(name, &env);
    }
    ExitCode::SUCCESS
}

fn header(env: &harness::Env) {
    let s = Settings::default();
    let physical = sm_embed::physical_cores();
    let logical = std::thread::available_parallelism().map_or(0, usize::from);
    println!(
        "budgets: {} | {physical} physical cores, {logical} logical | {} {} | {} build",
        sm_embed::os::cpu_name(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!(
        "defaults: {} query threads, {} bulk threads, batch {}, chunk {} tokens | binary {} | model {}",
        s.query_threads(physical),
        s.bulk_threads(physical),
        s.embed.batch_size,
        s.chunk.max_tokens,
        env.bin.display(),
        env.model_dir.display()
    );
}

/// Windows: this process's standard handles may be inheritable (a pipe from
/// the shell); a daemon started from here would hold them open and keep the
/// shell waiting. `Stdio::inherit` still works (see `client::disinherit`).
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
