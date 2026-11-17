//! The command line (SPEC section 3; CONTRACTS 5): every subcommand but
//! `hook`, which [`crate::main_entry`] dispatches before the parser. One
//! binary is the daemon, the MCP server, the hook and the tools (D-1).
//!
//! Errors go to stderr as `error: <code>: <message>` and `next: <next
//! action>`, with exit code 1. Usage errors are clap's, with its exit code.

use std::path::{Path, PathBuf};
use std::process::{Command as Process, Stdio};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use sm_core::config::{DEFAULT_PROFILE, ProfileName, Settings};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::paths::{self, ProfilePaths};
use sm_core::protocol::{
    DoctorEvalParams, ExportParams, ImportParams, MigrateParams, Op, ProjectDeleteParams,
    ProjectRenameParams, ReindexParams, StatusParams,
};

use crate::client::{self, Connector, DetachedSpawn, Spawn, Stop};
use crate::doctor::{self, DoctorEnv};
use crate::model_step::EmbedModelStep;
use crate::profile::{self, process_env};
use crate::setup::{self, ProcessRunner, SetupClient, SetupEnv};
use crate::{daemon, locate, mcp, ops};

/// `spatial-memory`.
#[derive(Debug, Parser)]
#[command(
    name = "spatial-memory",
    version,
    propagate_version = true,
    about = "Local semantic memory for coding agents: the MCP server, the hooks, setup and \
             maintenance."
)]
pub struct Cli {
    /// The profile to use. Overrides the plugin option, SPATIAL_MEMORY_PROFILE and the
    /// directory rules in config.toml.
    #[arg(long, global = true, value_name = "NAME")]
    pub profile: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve the thirteen memory tools over MCP on stdin and stdout.
    Mcp,
    /// Run the daemon that holds the embedding model. Clients start it when none answers.
    Daemon,
    /// Show the daemon, the model, the index and the memories of each project.
    Status,
    /// Stop this profile's daemon cleanly and wait until it has gone. Starts none.
    Stop,
    /// Run one hook event: `hook <event> --client <claude-code|cursor>`. The agent calls this.
    Hook {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
        args: Vec<String>,
    },
    /// Register the MCP server and the hooks without the plugin, or remove them.
    Setup {
        /// Remove what setup registered.
        #[arg(long)]
        remove: bool,
        /// The agent to register with.
        #[arg(long, value_enum, default_value = "claude-code")]
        client: SetupClient,
        /// Only download and verify the model; register nothing. Allowed while the plugin
        /// is enabled.
        #[arg(long, conflicts_with_all = ["remove", "client"])]
        model_only: bool,
    },
    /// Check the installation, or run a retrieval evaluation with --eval.
    Doctor {
        /// Run the queries in this evaluation set against the store.
        #[arg(long, value_name = "FILE")]
        eval: Option<PathBuf>,
        /// A baseline report to compare the evaluation with.
        #[arg(requires = "eval", value_name = "BASELINE")]
        baseline: Option<PathBuf>,
    },
    /// Load memories exported by `spatial-memory export`.
    Import {
        /// The export file.
        file: PathBuf,
    },
    /// Write the store, or one project, to a file that does not exist yet.
    Export {
        /// The file to create.
        file: PathBuf,
        /// Only this project.
        #[arg(long)]
        project: Option<String>,
    },
    /// Load memories exported from Spatial Memory 1.x or 2.x.
    Migrate {
        /// The JSON export from the old install.
        #[arg(long, value_name = "FILE")]
        from: PathBuf,
        /// The project for memories whose namespace names none.
        #[arg(long, value_name = "NAME")]
        default_project: Option<String>,
    },
    /// Rebuild the index after the model or the chunk settings changed.
    Reindex,
    /// Rename or delete a project.
    #[command(subcommand)]
    Project(ProjectCommand),
}

#[derive(Debug, Subcommand)]
pub enum ProjectCommand {
    /// Rename a project.
    Rename { from: String, to: String },
    /// Delete a project and its memories. Without --confirm, shows what would go.
    Delete {
        name: String,
        /// Delete for real.
        #[arg(long)]
        confirm: bool,
    },
}

/// Parses `args` (without the program name) and runs the command.
pub fn run(args: Vec<String>) -> i32 {
    let argv = std::iter::once(String::from("spatial-memory")).chain(args);
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    let cwd = std::env::current_dir().unwrap_or_default();
    match execute(cli, &cwd) {
        Ok(code) => code,
        Err(e) => {
            eprint!("{}", error_text(&e));
            1
        }
    }
}

/// How an error is printed on stderr.
pub fn error_text(e: &CoreError) -> String {
    format!(
        "error: {}: {}\nnext: {}\n",
        e.code, e.message, e.next_action
    )
}

fn spawner() -> DetachedSpawn {
    DetachedSpawn::current().unwrap_or_else(|_| DetachedSpawn {
        exe: PathBuf::from("spatial-memory"),
    })
}

fn home() -> Result<PathBuf, CoreError> {
    paths::home_dir().ok_or_else(|| {
        CoreError::config_invalid("the home folder is unknown")
            .with_next_action("Set HOME (USERPROFILE on Windows) and run the command again.")
    })
}

fn absolute_string(path: &Path, cwd: &Path) -> String {
    profile::absolute(path, cwd).to_string_lossy().into_owned()
}

/// Runs `<program> --version` and returns what it printed.
fn version_of(program: &Path) -> Option<String> {
    let out = Process::new(program)
        .arg("--version")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// A profile `setup` pins in what it registers: only an explicit
/// `--profile` other than `default`, so the plugin option, the variable and
/// the directory rules keep choosing otherwise.
fn profile_pin(flag: Option<&str>) -> Result<Option<ProfileName>, CoreError> {
    match flag.map(str::trim).filter(|f| !f.is_empty()) {
        None => Ok(None),
        Some(f) => {
            let name = ProfileName::parse(f)?;
            Ok((name.as_str() != DEFAULT_PROFILE).then_some(name))
        }
    }
}

fn execute(cli: Cli, cwd: &Path) -> Result<i32, CoreError> {
    let flag = cli.profile.as_deref();
    let cwd_string = cwd.to_string_lossy().into_owned();
    let op = match cli.command {
        // The process ends as soon as the daemon stops (stream J item 1).
        Command::Daemon => std::process::exit(daemon::run(flag, cwd, &process_env)),
        Command::Status => Op::Status(StatusParams {
            metrics: false,
            limit: None,
        }),
        Command::Mcp => {
            let paths = profile::resolve(flag, &process_env, cwd).map(|r| r.paths);
            let spawner: Arc<dyn Spawn> = Arc::new(spawner());
            mcp::serve(mcp::Proxy::new(paths, cwd_string, spawner))?;
            return Ok(0);
        }
        Command::Hook { args } => {
            let mut all = Vec::new();
            if let Some(p) = flag {
                all.push(String::from("--profile"));
                all.push(p.to_owned());
            }
            all.extend(args);
            return Ok(crate::run_hook(&all, Instant::now()));
        }
        Command::Stop => {
            let paths = profile::resolve(flag, &process_env, cwd)?.paths;
            println!("{}", stop_line(&paths, cwd, "stop")?);
            return Ok(0);
        }
        Command::Setup {
            model_only: true, ..
        } => return run_model_only(flag, cwd),
        Command::Setup { remove, client, .. } => return run_setup(flag, cwd, client, remove),
        Command::Doctor {
            eval: None,
            baseline: _,
        } => return Ok(run_doctor(flag, cwd)),
        Command::Doctor {
            eval: Some(set),
            baseline,
        } => Op::DoctorEval(DoctorEvalParams {
            set_path: absolute_string(&set, cwd),
            baseline_path: baseline.map(|b| absolute_string(&b, cwd)),
        }),
        Command::Import { file } => Op::Import(ImportParams {
            path: absolute_string(&file, cwd),
        }),
        Command::Export { file, project } => Op::Export(ExportParams {
            path: absolute_string(&file, cwd),
            project,
        }),
        Command::Migrate {
            from,
            default_project,
        } => Op::Migrate(MigrateParams {
            from: absolute_string(&from, cwd),
            default_project,
        }),
        Command::Reindex => Op::Reindex(ReindexParams {}),
        Command::Project(ProjectCommand::Rename { from, to }) => {
            Op::ProjectRename(ProjectRenameParams { from, to })
        }
        Command::Project(ProjectCommand::Delete { name, confirm }) => {
            Op::ProjectDelete(ProjectDeleteParams { name, confirm })
        }
    };
    op.validate()?;
    let resolved = profile::resolve(flag, &process_env, cwd)?;
    let spawner = spawner();
    let connector = Connector::new(resolved.paths, &spawner, cwd_string);
    let outcome = ops::run(op, |op| connector.call(op))?;
    print!("{}", outcome.stdout);
    eprint!("{}", outcome.stderr);
    Ok(outcome.code)
}

fn run_setup(
    flag: Option<&str>,
    cwd: &Path,
    client: SetupClient,
    remove: bool,
) -> Result<i32, CoreError> {
    let home = home()?;
    let exe = std::env::current_exe().map_err(|e| {
        CoreError::new(
            ErrorCode::Internal,
            format!("cannot find this program's path: {e}"),
        )
    })?;
    let exe = profile::absolute(&exe, cwd);
    let settings = if remove {
        Settings::default()
    } else {
        profile::resolve(flag, &process_env, cwd)?.settings()?
    };
    let runner = ProcessRunner {
        program: locate::on_path("claude", &process_env),
    };
    let model = EmbedModelStep::pinned();
    let env = SetupEnv {
        home,
        cwd: cwd.to_path_buf(),
        exe,
        runner: &runner,
        model: &model,
        settings,
        profile_pin: profile_pin(flag)?,
    };
    let report = setup::run(&env, client, remove)?;
    for line in &report.lines {
        println!("{line}");
    }
    if remove {
        // B-155: with the registrations gone, nothing of the install is
        // left running.
        let paths = profile::resolve(flag, &process_env, cwd)?.paths;
        println!("{}", stop_line(&paths, cwd, "setup --remove")?);
    }
    match report.model_error {
        None => Ok(0),
        Some(e) => Err(setup::model_failure(e)),
    }
}

/// Stops the profile's daemon (B-155) and says what happened, in the words
/// `status` uses for the daemon.
fn stop_line(paths: &ProfilePaths, cwd: &Path, reason: &str) -> Result<String, CoreError> {
    let cwd = cwd.to_string_lossy();
    let profile = paths.profile.as_str();
    Ok(
        match client::stop_daemon(paths, &cwd, reason, client::STOP_BUDGET)? {
            Stop::NotRunning => format!("Daemon: not running for profile {profile}"),
            Stop::Stopped { pid: Some(pid) } => {
                format!("Daemon: stopped (process {pid}, profile {profile})")
            }
            Stop::Stopped { pid: None } => format!("Daemon: stopped (profile {profile})"),
        },
    )
}

/// `setup --model-only` (B-153): the model step alone, so it never reads
/// the plugin state and registers nothing.
fn run_model_only(flag: Option<&str>, cwd: &Path) -> Result<i32, CoreError> {
    let settings = profile::resolve(flag, &process_env, cwd)?.settings()?;
    let report = setup::model_only(&EmbedModelStep::pinned(), &settings);
    for line in &report.lines {
        println!("{line}");
    }
    match report.model_error {
        None => Ok(0),
        Some(e) => Err(e),
    }
}

fn run_doctor(flag: Option<&str>, cwd: &Path) -> i32 {
    let spawner = spawner();
    let runner = ProcessRunner {
        program: locate::on_path("claude", &process_env),
    };
    let model = EmbedModelStep::pinned();
    let env = DoctorEnv {
        get_env: &process_env,
        cwd: cwd.to_path_buf(),
        home: paths::home_dir(),
        spawner: &spawner,
        runner: &runner,
        model: &model,
        version_of: &version_of,
        window: None,
        cold_start: None,
    };
    let checks = doctor::checks(&env, flag);
    print!("{}", doctor::render(&checks));
    doctor::exit_code(&checks)
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("spatial-memory").chain(args.iter().copied()))
    }

    #[test]
    fn s3_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn s3_every_subcommand_takes_profile_and_daemon_is_registered() {
        for args in [
            &["mcp", "--profile", "work"][..],
            &["daemon", "--profile", "work"],
            &["status", "--profile", "work"],
            &["stop", "--profile", "work"],
            &["setup", "--profile", "work"],
            &["doctor", "--profile", "work"],
            &["import", "x.jsonl", "--profile", "work"],
            &["export", "x.jsonl", "--profile", "work"],
            &["migrate", "--from", "x.json", "--profile", "work"],
            &["reindex", "--profile", "work"],
            &["project", "rename", "a", "b", "--profile", "work"],
            &["project", "delete", "a", "--profile", "work"],
        ] {
            let cli = parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
            assert_eq!(cli.profile.as_deref(), Some("work"), "{args:?}");
        }
        // The spawned command line: `daemon --profile <name>` (CONTRACTS 3.1).
        let spawned = crate::client::daemon_command(
            Path::new("spatial-memory"),
            &ProfileName::parse("work").unwrap(),
        );
        let argv: Vec<&str> = spawned.get_args().map(|a| a.to_str().unwrap()).collect();
        let cli = parse(&argv).unwrap();
        assert!(matches!(cli.command, Command::Daemon));
        assert_eq!(cli.profile.as_deref(), Some("work"));
        assert!(parse(&["daemon", "extra"]).is_err());
        assert!(parse(&["doctor", "base.json"]).is_err());
        let cli = parse(&["doctor", "--eval", "set.json", "base.json"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Doctor {
                eval: Some(_),
                baseline: Some(_)
            }
        ));
        let cli = parse(&["setup", "--client", "cursor", "--remove"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Setup {
                remove: true,
                client: SetupClient::Cursor,
                model_only: false
            }
        ));
    }

    #[test]
    fn s16_only_an_explicit_non_default_profile_is_pinned() {
        assert_eq!(profile_pin(None).unwrap(), None);
        assert_eq!(profile_pin(Some("default")).unwrap(), None);
        assert_eq!(profile_pin(Some("  ")).unwrap(), None);
        assert_eq!(profile_pin(Some("work")).unwrap().unwrap().as_str(), "work");
        assert!(profile_pin(Some("../x")).is_err());
    }

    #[test]
    fn s8_errors_print_code_message_and_next_action() {
        let e = CoreError::new(ErrorCode::NotFound, "no such project")
            .with_next_action("List projects with status.");
        assert_eq!(
            error_text(&e),
            "error: not_found: no such project\nnext: List projects with status.\n"
        );
    }
}
