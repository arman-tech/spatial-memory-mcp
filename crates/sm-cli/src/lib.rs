//! The `spatial-memory` binary (SPEC section 3; decision D-1): the daemon,
//! the hook client, the MCP proxy, `setup`, `doctor`, `status` and the
//! command-line operations. Only the daemon holds the embedding model;
//! every other subcommand talks to it over the local endpoint (CONTRACTS
//! sections 3 and 4) and starts it, detached, when none answers.

pub mod cli;
pub mod client;
pub mod daemon;
pub mod doctor;
pub mod hook;
pub mod hook_input;
pub mod locate;
pub mod mcp;
pub mod model_step;
pub mod ops;
pub mod profile;
pub mod replay;
pub mod setup;

use std::path::PathBuf;
use std::time::Instant;

/// The program's entry point; returns the exit code.
///
/// `hook` is dispatched before the argument parser so that nothing it
/// reports as a usage error can make a hook exit non-zero or print to stderr
/// (SPEC section 16, Never blocks; version 2 parsed hook arguments itself).
pub fn main_entry() -> i32 {
    let start = Instant::now();
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    if let Some(at) = hook_position(&args) {
        let mut rest = args;
        rest.remove(at);
        return run_hook(&rest, start);
    }
    cli::run(args)
}

/// Where `hook` stands when only the global `--profile` option comes before
/// it, so `spatial-memory --profile work hook stop` takes the hook path too.
pub fn hook_position(args: &[String]) -> Option<usize> {
    let mut i = 0;
    while let Some(arg) = args.get(i) {
        match arg.as_str() {
            "hook" => return Some(i),
            "--profile" => i += 2,
            a if a.starts_with("--profile=") => i += 1,
            _ => return None,
        }
    }
    None
}

/// Runs one hook event against the real process: its stdin, stdout,
/// environment and working folder. `args` are the arguments with the `hook`
/// token removed; a leading `--profile` is still among them.
pub(crate) fn run_hook(args: &[String], start: Instant) -> i32 {
    let spawner = client::DetachedSpawn::current().unwrap_or_else(|_| client::DetachedSpawn {
        exe: PathBuf::from("spatial-memory"),
    });
    let env = hook::HookEnv {
        get_env: &profile::process_env,
        process_cwd: std::env::current_dir().unwrap_or_default(),
        spawner: &spawner,
        start,
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    hook::run(args, Box::new(std::io::stdin()), &mut out, &env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn s16_hook_is_found_after_global_profile_options_only() {
        assert_eq!(hook_position(&strings(&["hook", "stop"])), Some(0));
        assert_eq!(
            hook_position(&strings(&["--profile", "work", "hook", "stop"])),
            Some(2)
        );
        assert_eq!(
            hook_position(&strings(&["--profile=work", "hook", "stop"])),
            Some(1)
        );
        assert_eq!(hook_position(&strings(&["doctor", "hook"])), None);
        assert_eq!(hook_position(&strings(&["--profile"])), None);
        assert_eq!(hook_position(&strings(&[])), None);
    }
}
