//! `spatial-memory doctor` (SPEC section 5 rule 7, section 16).
//!
//! Each check passes or fails with a next action: the profile and its
//! folders, the daemon (it answers `hello`, its build matches, its state),
//! the index and the journal's measurements from `status`, the embedder's
//! numbers timed by the daemon (`doctor_numbers`, B-125), `spatial-memory`
//! on `PATH` at the plugin's release (the part of its `--version` before
//! `+`), the daemon built from the commit that program was (B-231), the
//! plugin and `setup` not both registered, the spool, and the model step
//! (decision B-43). The command exits 1 when any check fails.
//!
//! `doctor --eval <file> [baseline]` is separate: it sends `doctor_eval` and
//! prints the report (see [`crate::ops::render_eval`]).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::protocol::{
    DaemonInfo, DaemonState, DoctorNumbersReply, JudgeBlocked, JudgeStatus, Op, PROTOCOL_VERSION,
    Reply, StatusParams, StatusReply,
};
use sm_core::time::Timestamp;

use crate::client::{Connector, Spawn};
use crate::model_step::ModelStep;
use crate::profile::{self, GetEnv};
use crate::setup::{self, ClaudeRunner};
use crate::version;

/// The plugin manifest this build was made with; `doctor` compares the
/// version on `PATH` with it.
pub const PLUGIN_JSON: &str = include_str!("../../../plugin/.claude-plugin/plugin.json");

/// The plugin's version, from [`PLUGIN_JSON`].
pub fn plugin_version() -> Option<String> {
    let manifest: Value = serde_json::from_str(PLUGIN_JSON).ok()?;
    manifest.get("version")?.as_str().map(String::from)
}

/// The next action when the embedder's numbers were not measured because
/// the daemon was still loading (SPEC 5 rule 7; B-125).
pub const NUMBERS_LATER: &str = "Run `spatial-memory doctor` again in a few seconds.";

/// SPEC 11: the daemon is ready from cold in 5 seconds or less. doctor
/// waits for a daemon still loading until this long after the daemon
/// started, and fails it only after that (B-152).
pub const COLD_START_BUDGET: Duration = Duration::from_secs(5);

/// One check's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
    /// What to do about a failure, or about a number not yet measured.
    pub next_action: Option<String>,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Check {
            name,
            passed: true,
            detail: detail.into(),
            next_action: None,
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>, next: impl Into<String>) -> Self {
        Check {
            name,
            passed: false,
            detail: detail.into(),
            next_action: Some(next.into()),
        }
    }

    fn from_error(name: &'static str, e: &CoreError) -> Self {
        Check::fail(
            name,
            format!("{}: {}", e.code, e.message),
            e.next_action.clone(),
        )
    }
}

/// Everything `doctor` reads.
pub struct DoctorEnv<'a> {
    pub get_env: GetEnv<'a>,
    pub cwd: PathBuf,
    /// The user's home folder, for the plugin and `setup` registrations.
    pub home: Option<PathBuf>,
    pub spawner: &'a dyn Spawn,
    /// Asks `claude plugin list --json` (read-only).
    pub runner: &'a dyn ClaudeRunner,
    pub model: &'a dyn ModelStep,
    /// Runs `<program> --version` and returns its output.
    pub version_of: &'a dyn Fn(&Path) -> Option<String>,
    /// How long to wait for the daemon; `None` is the client's 10 s.
    pub window: Option<Duration>,
    /// How long after its start a daemon may still be loading the model
    /// before the `state` check fails; `None` is [`COLD_START_BUDGET`].
    pub cold_start: Option<Duration>,
}

/// Runs every check.
pub fn checks(env: &DoctorEnv<'_>, profile_flag: Option<&str>) -> Vec<Check> {
    let mut out = Vec::new();
    let resolved = match profile::resolve(profile_flag, env.get_env, &env.cwd) {
        Ok(r) => r,
        Err(e) => {
            out.push(Check::from_error("profile", &e));
            out.push(path_check(find_on_path(env).as_ref()));
            out.push(registration_check(env));
            out.push(mcp_enabled_check(env));
            return out;
        }
    };
    let paths = resolved.paths.clone();
    let folder = if paths.dir.is_dir() {
        "exists"
    } else {
        "not created yet; the daemon creates it"
    };
    let settings = match resolved.settings() {
        Ok(s) => {
            out.push(Check::pass(
                "profile",
                format!(
                    "{} (from {}), folder {} ({folder})",
                    paths.profile.as_str(),
                    profile::source_name(resolved.source),
                    paths.dir.display()
                ),
            ));
            Some(s)
        }
        Err(e) => {
            out.push(Check::from_error("profile", &e));
            None
        }
    };

    let mut connector = Connector::new(
        paths.clone(),
        env.spawner,
        env.cwd.to_string_lossy().into_owned(),
    );
    if let Some(w) = env.window {
        connector.window = w;
    }
    let budget = env.cold_start.unwrap_or(COLD_START_BUDGET);
    let (daemon, info) = daemon_checks(&connector, &paths.daemon_log(), budget);
    out.extend(daemon);
    let on_path = find_on_path(env);
    out.push(path_check(on_path.as_ref()));
    if let Some(info) = &info {
        out.push(commit_check(info, on_path.as_ref()));
    }
    out.push(registration_check(env));
    out.push(mcp_enabled_check(env));
    out.push(spool_check(&paths.spool_new(), &paths.spool_failed()));
    if let Some(settings) = &settings {
        out.push(match env.model.check(settings) {
            Ok(line) => Check::pass("model", line),
            Err(e) => Check::from_error("model", &e),
        });
    }
    out
}

/// The daemon's checks, and its `hello` when it answered one.
fn daemon_checks(
    connector: &Connector<'_>,
    log: &Path,
    budget: Duration,
) -> (Vec<Check>, Option<DaemonInfo>) {
    let see_log = format!("See {} for the daemon's own report.", log.display());
    // The embedder check waits for a loading daemon inside the window the
    // connection itself is given (CONTRACTS 3.1), not a second one.
    let until = Instant::now() + connector.window;
    let info = match connector.ensure_daemon() {
        Ok(Some(Reply::Hello(info))) => info,
        Ok(Some(other)) => {
            let kind = serde_json::to_value(&other)
                .ok()
                .and_then(|v| v.get("kind").and_then(Value::as_str).map(String::from))
                .unwrap_or_default();
            let check = Check::fail(
                "daemon",
                format!("the daemon answered hello with a {kind} reply"),
                see_log,
            );
            return (vec![check], None);
        }
        Ok(None) => {
            let check = Check::fail(
                "daemon",
                "the daemon is listening but answered hello with an error",
                see_log,
            );
            return (vec![check], None);
        }
        Err(e) => {
            let check = Check::fail(
                "daemon",
                format!("{}: {}", e.code, e.message),
                format!("{} {see_log}", e.next_action),
            );
            return (vec![check], None);
        }
    };
    let info = if info.state == DaemonState::Loading {
        wait_for_load(connector, info, budget)
    } else {
        info
    };
    let mut out = vec![Check::pass(
        "daemon",
        format!("pid {} answers at {}", info.pid, info.endpoint),
    )];
    if same_build(&info) {
        out.push(Check::pass(
            "build",
            format!("{} (protocol {})", info.build, info.protocol),
        ));
    } else {
        out.push(Check::fail(
            "build",
            format!(
                "the daemon is build {} protocol {}; this program is build {} protocol {}",
                info.build,
                info.protocol,
                sm_core::BUILD,
                PROTOCOL_VERSION
            ),
            "Run the command again: a client replaces a daemon of another build.",
        ));
    }
    match info.state {
        DaemonState::Ready => {
            let ready = info
                .ready_seconds
                .map(|s| format!(", ready in {s:.1} s"))
                .unwrap_or_default();
            out.push(Check::pass(
                "state",
                format!("ready with {}{ready}", info.model_id),
            ));
        }
        DaemonState::Loading => out.push(Check::fail(
            "state",
            format!(
                "still loading {} {:.1} s after the daemon started, beyond the {} s cold-start \
                 budget",
                info.model_id,
                since_start(&info).as_secs_f64(),
                budget.as_secs_f64()
            ),
            format!(
                "Run doctor again. If the model is still loading then, see {} for the daemon's \
                 own report.",
                log.display()
            ),
        )),
        DaemonState::Failed => out.push(match &info.state_error {
            Some(e) => Check::fail(
                "state",
                format!("the model failed to load: {}: {}", e.code, e.message),
                format!("{} {see_log}", e.next_action),
            ),
            None => Check::fail("state", "the model failed to load", see_log.clone()),
        }),
    }
    if info.state == DaemonState::Ready {
        let status = connector.call(Op::Status(StatusParams {
            metrics: true,
            limit: None,
        }));
        match status {
            Ok(Reply::Status(s)) => out.extend(status_checks(&s)),
            Ok(_) => out.push(Check::fail(
                "index",
                "the daemon answered status with another reply",
                see_log,
            )),
            Err(e) => out.push(Check::from_error("index", &e)),
        }
    }
    // A failed model is already a failure in `state` and `model`.
    if info.state != DaemonState::Failed {
        out.push(embedder_check(connector, until, &info.model_id));
    }
    (out, Some(info))
}

/// How long ago the daemon started, by its own `started_at`; zero when the
/// clock reads it in the future.
fn since_start(info: &DaemonInfo) -> Duration {
    let micros = Timestamp::now().as_micros() - info.started_at.as_micros();
    Duration::from_micros(u64::try_from(micros).unwrap_or(0))
}

/// A daemon still loading the model is asked `hello` again every poll
/// interval until it is ready or failed, or until `budget` has passed since
/// it started (SPEC 11, daemon ready from cold; B-152). The deadline is set
/// once, from the first answer, and is never more than `budget` from now.
fn wait_for_load(connector: &Connector<'_>, mut info: DaemonInfo, budget: Duration) -> DaemonInfo {
    let until = Instant::now() + budget.saturating_sub(since_start(&info));
    while info.state == DaemonState::Loading && Instant::now() + connector.poll <= until {
        std::thread::sleep(connector.poll);
        match connector.ensure_daemon() {
            Ok(Some(Reply::Hello(next))) => info = next,
            _ => break,
        }
    }
    info
}

/// The embedder's numbers, timed by the daemon on its own sessions (SPEC 5
/// rule 7; B-125): information, nothing is tuned from them. A daemon still
/// loading is asked again every poll interval until `until`; if it is still
/// loading then, the numbers were not measured, which is not a failure.
fn embedder_check(connector: &Connector<'_>, until: Instant, model_id: &str) -> Check {
    loop {
        match connector.call(Op::DoctorNumbers) {
            Ok(Reply::DoctorNumbers(n)) => return Check::pass("embedder", numbers_line(&n)),
            Ok(_) => {
                return Check::fail(
                    "embedder",
                    "the daemon answered doctor_numbers with another reply",
                    "Run the command again: a client replaces a daemon of another build.",
                );
            }
            Err(e) if e.code == ErrorCode::DaemonLoading => {
                if Instant::now() + connector.poll > until {
                    return not_measured(model_id);
                }
                std::thread::sleep(connector.poll);
            }
            Err(e) => return Check::from_error("embedder", &e),
        }
    }
}

/// The line for numbers not measured because the daemon was still loading.
fn not_measured(model_id: &str) -> Check {
    Check {
        name: "embedder",
        passed: true,
        detail: format!(
            "not measured: the daemon is still loading {model_id}, so chunks per second \
             and query latency were not timed"
        ),
        next_action: Some(NUMBERS_LATER.to_owned()),
    }
}

/// The bulk rate with its thread count and batch size, and the median query
/// latency with its token count and thread count.
fn numbers_line(n: &DoctorNumbersReply) -> String {
    format!(
        "bulk {:.1} chunks/s on {} threads, batch {} ({} chunks in {:.0} ms); \
         query median {:.1} ms at {} tokens on {} threads ({} runs); {} physical cores",
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

/// The index state, and the journal's measurements as information (SPEC
/// section 5 rule 7: chunks per second and query latency, nothing tuned).
fn status_checks(s: &StatusReply) -> Vec<Check> {
    let mut out = Vec::new();
    let memories: u64 = s.projects.iter().map(|p| p.memories).sum();
    if s.index.matches {
        out.push(Check::pass(
            "index",
            format!(
                "{} memories in {} projects, {} chunks, {} waiting for embedding",
                memories,
                s.projects.len(),
                s.index.chunks,
                s.index.pending_embeddings
            ),
        ));
    } else {
        out.push(Check::fail(
            "index",
            format!(
                "the index does not match the configuration: {}",
                s.index.differences.join("; ")
            ),
            "Run `spatial-memory reindex`.",
        ));
    }
    let mut lines = String::new();
    for m in s.metrics.iter().flatten().filter(|m| m.count > 0) {
        let num = |v: Option<f64>| v.map_or_else(|| String::from("-"), |x| format!("{x:.3}"));
        let _ = write!(
            lines,
            "{}{} median {} p95 {} (n={})",
            if lines.is_empty() { "" } else { "; " },
            m.metric,
            num(m.median),
            num(m.p95),
            m.count
        );
    }
    out.push(judge_check(&s.judge));
    out.push(Check::pass(
        "measurements",
        if lines.is_empty() {
            String::from("no journal entries in the last 7 days")
        } else {
            lines
        },
    ));
    out
}

/// Whether the capture judge can run, its pending captures, today's use
/// and cost, and the last 7 days of automatic capture when the daemon
/// reports them (SPEC section 16, The judge; B-165, B-200). It fails only when the judge
/// cannot run and the user did not choose that: Claude Code missing or
/// signed out, or a daemon started without a judge. A judge turned off, at
/// its daily cap, or waiting after a failed call passes, with the reason and
/// the last error.
fn judge_check(j: &JudgeStatus) -> Check {
    use crate::ops::{judge_pending, judge_runs, judge_state, judge_today, judge_week};
    let mut detail = format!(
        "{}; {}; pending {}; today {}",
        judge_state(j),
        judge_runs(j),
        judge_pending(j),
        judge_today(j)
    );
    if let (Some(week), Some(q)) = (judge_week(j), &j.quality) {
        let _ = write!(detail, "; last {} days {week}", q.days);
    }
    if let Some(e) = &j.last_error {
        let _ = write!(detail, "; last error: {e}");
    }
    match j.blocked {
        Some(JudgeBlocked::Unavailable) => Check::fail(
            "judge",
            detail,
            "Install Claude Code and sign in (run `claude` once), or name its executable in \
             capture.judge.command. Until then automatic captures wait as pending. To stop \
             judging, set capture.judge.enabled = false.",
        ),
        Some(JudgeBlocked::NoJudge) => Check::fail(
            "judge",
            detail,
            "Run `spatial-memory stop`; the next command starts a daemon with the judge.",
        ),
        _ => Check::pass("judge", detail),
    }
}

/// The `spatial-memory` that `PATH` finds, and the last word its
/// `--version` printed: `3.0.0+1b5b54a` from a build with B-231, `3.0.0`
/// from one before it.
struct OnPath {
    exe: PathBuf,
    version: Option<String>,
}

fn find_on_path(env: &DoctorEnv<'_>) -> Option<OnPath> {
    let exe = crate::locate::on_path("spatial-memory", env.get_env)?;
    let version = (env.version_of)(&exe)
        .as_deref()
        .and_then(|o| o.split_whitespace().last())
        .map(String::from);
    Some(OnPath { exe, version })
}

/// The program on `PATH` is the plugin's release. Only the part before `+`
/// is the release; the commit after it is the `commit` check's (B-231).
fn path_check(found: Option<&OnPath>) -> Check {
    let want = plugin_version();
    let Some(found) = found else {
        return Check::fail(
            "path",
            "`spatial-memory` is not on PATH, so the plugin's hooks and server cannot start",
            "Put the folder holding spatial-memory on PATH, or run `spatial-memory setup`, \
             which writes the absolute path.",
        );
    };
    let exe = found.exe.display();
    match (found.version.as_deref(), want) {
        (Some(v), Some(w)) if version::split(v).0 == w => Check::pass(
            "path",
            format!("{exe} is version {v}: release {w}, as the plugin expects"),
        ),
        (Some(v), Some(w)) => Check::fail(
            "path",
            format!("{exe} is version {v}; the plugin expects release {w}"),
            format!("Install spatial-memory {w} on PATH, or update the plugin to match."),
        ),
        (None, _) => Check::fail(
            "path",
            format!("{exe} did not report a version"),
            "Reinstall spatial-memory; `spatial-memory --version` must print the version.",
        ),
        (Some(v), None) => Check::pass("path", format!("{exe} is version {v}")),
    }
}

/// The daemon is this program's build: its protocol and release.
fn same_build(info: &DaemonInfo) -> bool {
    info.build == sm_core::BUILD && info.protocol == PROTOCOL_VERSION
}

/// The next action when the daemon of `profile` runs another commit than
/// the program on `PATH` (B-231). It names the profile even when it is the
/// default one: a command without `--profile` takes the profile from the
/// plugin's option, `SPATIAL_MEMORY_PROFILE` or a folder's rule, which may
/// name another. A client of the same release never replaces the daemon,
/// so it runs until stopped, and the client that starts it next starts it
/// from the file its own program was started from. On Windows that is the
/// file's path when the program started, even after an install renamed the
/// old file and put the new one there, so an older session starts the
/// installed program; a copy at another path, doctor among them, starts
/// itself.
fn commit_next(profile: &str) -> String {
    format!(
        "Run `spatial-memory stop --profile {profile}`, then `spatial-memory status --profile \
         {profile}`, which starts the daemon from the program on PATH. A doctor run by another \
         path starts the daemon from itself when none is running. If doctor still names \
         another commit, a copy of spatial-memory at another path started the daemon first, \
         such as one a hook or the MCP server's registration names by its full path: point \
         that registration at the program on PATH."
    )
}

/// Two stamps' commits name one commit: the same `.dirty` mark, and one
/// hash the start of the other, since git gives a short hash more than its
/// 7 characters where 7 are ambiguous in that clone.
fn same_commit(a: &str, b: &str) -> bool {
    fn split(c: &str) -> (&str, bool) {
        match c.strip_suffix(version::DIRTY) {
            Some(hash) => (hash, true),
            None => (c, false),
        }
    }
    let ((a, a_dirty), (b, b_dirty)) = (split(a), split(b));
    a_dirty == b_dirty && !a.is_empty() && !b.is_empty() && (a.starts_with(b) || b.starts_with(a))
}

/// The daemon was built from the commit the program on `PATH` was (B-231).
/// Clients and the daemon compare the release only, so two builds of one
/// release answer each other and only this check tells a daemon left from
/// an earlier install apart. It does not compare when the daemon is another
/// build than this program, which the `build` check reports; when the
/// program on `PATH` prints no commit (a build older than B-231); when it
/// is another release than the daemon, which the `path` check reports,
/// since this program's release is the plugin's; or when either was built
/// without git. A daemon that reports no commit was built before B-231, so
/// it is not the program on `PATH`, and fails.
fn commit_check(daemon: &DaemonInfo, found: Option<&OnPath>) -> Check {
    let not_compared = |why: String| Check::pass("commit", format!("not compared: {why}"));
    if !same_build(daemon) {
        return not_compared(format!(
            "the daemon is build {} protocol {}, another build than this program, which the \
             build check reports",
            daemon.build, daemon.protocol
        ));
    }
    let Some(found) = found else {
        return not_compared("`spatial-memory` is not on PATH".into());
    };
    let exe = found.exe.display();
    let Some((release, Some(want))) = found.version.as_deref().map(version::split) else {
        return not_compared(format!("{exe} does not print its commit"));
    };
    if release != daemon.build {
        return not_compared(format!(
            "{exe} is release {release} and the daemon is release {}, which the path check \
             reports",
            daemon.build
        ));
    }
    if want == version::UNKNOWN {
        return not_compared(format!("{exe} was built without git"));
    }
    match daemon.commit.as_deref() {
        Some(version::UNKNOWN) => not_compared("the daemon was built without git".into()),
        Some(c) if same_commit(c, want) => {
            let both = if c == want {
                format!("the daemon and {exe} are both commit {c}")
            } else {
                format!("the daemon is commit {c} and {exe} is commit {want}, one commit")
            };
            let dirty = if c.ends_with(version::DIRTY) {
                "; both were built with uncommitted changes, so they may still differ"
            } else {
                ""
            };
            Check::pass("commit", format!("{both}{dirty}"))
        }
        Some(c) => Check::fail(
            "commit",
            format!(
                "the daemon (pid {}) runs commit {c}; {exe} is commit {want}",
                daemon.pid
            ),
            commit_next(&daemon.profile),
        ),
        None => Check::fail(
            "commit",
            format!(
                "the daemon (pid {}) reports no commit, so it was built before daemons \
                 reported one; {exe} is commit {want}",
                daemon.pid
            ),
            commit_next(&daemon.profile),
        ),
    }
}

fn registration_check(env: &DoctorEnv<'_>) -> Check {
    let Some(home) = &env.home else {
        return Check::fail(
            "registration",
            "the home folder is unknown, so the registrations cannot be read",
            "Set HOME (USERPROFILE on Windows) and run doctor again.",
        );
    };
    let plugins = setup::plugin_findings(home, &env.cwd, env.runner);
    if let Some(old) = plugins.old {
        // The Python server's hooks would run beside version 3's (B-150).
        return Check::fail(
            "registration",
            setup::old_plugin_line(&old),
            "Disable the version 1 plugin for this folder, then run doctor again.",
        );
    }
    let manual = setup::setup_registered(home);
    match (plugins.current, manual) {
        (Some(p), Some(s)) => Check::fail(
            "registration",
            format!(
                "both the plugin ({p}) and setup ({s}) are registered, so every hook runs twice"
            ),
            "Run `spatial-memory setup --remove`, or disable the plugin.",
        ),
        (Some(p), None) => Check::pass("registration", format!("the plugin ({p})")),
        (None, Some(s)) => Check::pass("registration", format!("setup ({s})")),
        (None, None) => Check::pass(
            "registration",
            "neither the plugin nor setup is registered for Claude Code",
        ),
    }
}

/// B-226: a project entry in `~/.claude.json` whose `disabledMcpServers`
/// names spatial-memory. It fails for this folder while the user-scope
/// server is registered, because Claude Code then does not start the
/// server here; other folders are named, and pass.
fn mcp_enabled_check(env: &DoctorEnv<'_>) -> Check {
    let Some(home) = &env.home else {
        return Check::fail(
            "mcp_enabled",
            "the home folder is unknown, so ~/.claude.json cannot be read",
            "Set HOME (USERPROFILE on Windows) and run doctor again.",
        );
    };
    let file = setup::claude_json_path(home);
    let disabled = setup::disabled_for(home, &env.cwd);
    if disabled.is_empty() {
        return Check::pass(
            "mcp_enabled",
            format!("no project in {} disables spatial-memory", file.display()),
        );
    }
    let found = format!(
        "`disabledMcpServers` in {} names spatial-memory for {}",
        file.display(),
        disabled.list()
    );
    let there = format!(
        "To use it there, run `claude` in that folder and {}.",
        setup::ENABLE_IN_MCP
    );
    let registered = setup::registered_server(home).is_some();
    if !registered {
        return Check {
            name: "mcp_enabled",
            passed: true,
            detail: format!(
                "the user-scope server is not registered; {found}, so it will not start there \
                 once `spatial-memory setup` registers it"
            ),
            next_action: Some(there),
        };
    }
    if disabled.here.is_empty() {
        return Check {
            name: "mcp_enabled",
            passed: true,
            detail: format!(
                "{found}, so Claude Code does not start the server there; here it does"
            ),
            next_action: Some(there),
        };
    }
    Check::fail(
        "mcp_enabled",
        format!(
            "{found}, so Claude Code does not start the server in this folder and its tools are \
             missing here; the hooks still run"
        ),
        format!(
            "Run `claude` in this folder and {}. Then run doctor again.",
            setup::ENABLE_IN_MCP
        ),
    )
}

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                .count()
        })
        .unwrap_or(0)
}

fn spool_check(new: &Path, failed: &Path) -> Check {
    let waiting = count_files(new);
    let bad = count_files(failed);
    if bad > 0 {
        return Check::fail(
            "spool",
            format!("{waiting} waiting, {bad} could not be loaded"),
            format!(
                "Read the files in {}; they were set aside and will not be retried.",
                failed.display()
            ),
        );
    }
    Check::pass(
        "spool",
        format!("{waiting} capture hand-offs waiting in {}", new.display()),
    )
}

/// The command's exit code: 1 when any check failed, else 0.
pub fn exit_code(checks: &[Check]) -> i32 {
    i32::from(checks.iter().any(|c| !c.passed))
}

/// The report: one line per check, its next action when it has one, and a
/// summary.
pub fn render(checks: &[Check]) -> String {
    let mut s = String::new();
    for c in checks {
        let mark = if c.passed { "ok  " } else { "FAIL" };
        let _ = writeln!(s, "{mark} {}: {}", c.name, c.detail);
        if let Some(next) = &c.next_action {
            let _ = writeln!(s, "     next: {next}");
        }
    }
    let failed = checks.iter().filter(|c| !c.passed).count();
    let _ = writeln!(s, "{} checks, {failed} failed", checks.len());
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16_plugin_manifest_carries_the_build_version() {
        assert_eq!(plugin_version().as_deref(), Some(sm_core::BUILD));
    }

    #[test]
    fn s16_spool_with_failed_files_fails_with_a_next_action() {
        let tmp = tempfile::tempdir().unwrap();
        let new = tmp.path().join("new");
        let failed = tmp.path().join("failed");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::create_dir_all(&failed).unwrap();
        std::fs::write(new.join("a.json"), "{}").unwrap();
        let ok = spool_check(&new, &failed);
        assert!(ok.passed);
        assert!(ok.detail.starts_with("1 capture"));
        std::fs::write(failed.join("b.json"), "{}").unwrap();
        let bad = spool_check(&new, &failed);
        assert!(!bad.passed);
        assert!(bad.next_action.unwrap().contains("set aside"));
    }

    #[test]
    fn s16_report_lists_failures_with_their_next_action() {
        let text = render(&[
            Check::pass("profile", "default"),
            Check::fail("daemon", "absent", "Start it."),
        ]);
        assert!(text.contains("ok   profile: default"));
        assert!(text.contains("FAIL daemon: absent\n     next: Start it."));
        assert!(text.ends_with("2 checks, 1 failed\n"));
    }

    fn sample_numbers() -> DoctorNumbersReply {
        DoctorNumbersReply {
            physical_cores: 16,
            query_threads: 4,
            bulk_threads: 12,
            batch_size: 16,
            sample_chunks: 16,
            query_tokens: 128,
            query_runs: 5,
            bulk_elapsed_ms: 927.4,
            bulk_chunks_per_sec: 17.26,
            query_median_ms: 47.21,
        }
    }

    #[test]
    fn s5_rule7_numbers_line_gives_both_numbers_with_threads_and_batch() {
        assert_eq!(
            numbers_line(&sample_numbers()),
            "bulk 17.3 chunks/s on 12 threads, batch 16 (16 chunks in 927 ms); \
             query median 47.2 ms at 128 tokens on 4 threads (5 runs); 16 physical cores"
        );
    }

    fn sample_judge() -> JudgeStatus {
        JudgeStatus {
            enabled: true,
            can_run: true,
            blocked: None,
            retry_at: None,
            model: "haiku".into(),
            command: Some("claude".into()),
            pending: 2,
            queued: 1,
            rejected: 0,
            failed: 0,
            calls_today: 5,
            max_calls_per_day: 200,
            judged_today: 60,
            cost_today_usd: 0.0094,
            max_usd_per_day: Some(3.0),
            last_error: None,
            quality: None,
        }
    }

    /// The judge check passes when the judge can run, is turned off, is at
    /// its call or spending cap (B-193) or waits after a failure, and fails with a next action when
    /// Claude Code is not available or the daemon has no judge (B-165).
    #[test]
    fn s16_judge_check_fails_only_when_the_judge_cannot_run_and_was_not_turned_off() {
        let ok = judge_check(&sample_judge());
        assert!(ok.passed);
        assert_eq!(
            ok.detail,
            "can run; model haiku through claude; pending 2 captures, 1 waiting for the judge; \
             today 5 of 200 calls, 60 judged, about 0.0094 US dollars of 3.00 (Claude Code's \
             estimate)"
        );
        for blocked in [
            JudgeBlocked::Disabled,
            JudgeBlocked::CaptureDisabled,
            JudgeBlocked::DailyCap,
            JudgeBlocked::DailyBudget,
            JudgeBlocked::BackingOff,
        ] {
            let mut j = sample_judge();
            j.can_run = false;
            j.blocked = Some(blocked);
            j.last_error = Some("rate limited".into());
            let c = judge_check(&j);
            assert!(c.passed, "{blocked:?}: {c:?}");
            assert!(c.detail.starts_with("cannot run: "), "{c:?}");
            assert!(c.detail.ends_with("; last error: rate limited"), "{c:?}");
        }
        for blocked in [JudgeBlocked::Unavailable, JudgeBlocked::NoJudge] {
            let mut j = sample_judge();
            j.can_run = false;
            j.blocked = Some(blocked);
            let c = judge_check(&j);
            assert!(!c.passed, "{blocked:?}");
            assert!(c.next_action.is_some(), "{blocked:?}");
        }
    }

    /// The judge check says what automatic capture did over the last 7
    /// days when the daemon reports it (B-200).
    #[test]
    fn s16_b200_judge_check_shows_the_last_7_days() {
        let mut j = sample_judge();
        j.quality = Some(sm_core::journal::CaptureQuality {
            days: 7,
            judged: 30,
            kept: 21,
            rejected: 9,
            dropped_after_keep: 0,
            forgotten: 1,
            forgotten_pending: 0,
            calls: 4,
            cost_usd: 0.0084,
        });
        j.last_error = Some("rate limited".into());
        let c = judge_check(&j);
        assert!(c.passed);
        assert!(
            c.detail.ends_with(
                "estimate); last 7 days 30 judged: 21 kept, 9 rejected; 1 captured memories \
                 removed with forget; 4 judge calls, about 0.0084 US dollars (Claude Code's \
                 estimate); last error: rate limited"
            ),
            "{}",
            c.detail
        );
    }

    #[test]
    fn s5_rule7_not_measured_is_not_a_failure_and_says_run_doctor_again() {
        let line = not_measured("m");
        assert!(line.passed);
        assert!(line.detail.starts_with("not measured"), "{}", line.detail);
        assert_eq!(line.next_action.as_deref(), Some(NUMBERS_LATER));
        let others = vec![Check::pass("profile", "default")];
        let mut with = others.clone();
        with.push(line);
        assert_eq!(exit_code(&with), exit_code(&others));
        assert_eq!(exit_code(&with), 0);
        let text = render(&with);
        assert!(
            text.contains(
                "ok   embedder: not measured: the daemon is still loading m, so chunks per \
                 second and query latency were not timed\n     next: Run `spatial-memory \
                 doctor` again in a few seconds.\n"
            ),
            "{text}"
        );
        assert!(text.ends_with("2 checks, 0 failed\n"));
    }

    fn on_path(version: Option<&str>) -> OnPath {
        OnPath {
            exe: PathBuf::from("bin/spatial-memory"),
            version: version.map(String::from),
        }
    }

    fn daemon_built_from(commit: Option<&str>) -> DaemonInfo {
        DaemonInfo {
            protocol: PROTOCOL_VERSION,
            build: sm_core::BUILD.into(),
            commit: commit.map(String::from),
            pid: 42,
            profile: "default".into(),
            state: DaemonState::Ready,
            state_error: None,
            started_at: Timestamp::now(),
            ready_seconds: Some(1.0),
            model_id: "m".into(),
            endpoint: "e".into(),
        }
    }

    /// B-231: the `path` check compares the release before `+` only, so a
    /// build with its commit and one from before B-231 both pass at the
    /// plugin's release, and another release fails.
    #[test]
    fn s16_b231_path_check_compares_the_release_before_the_plus() {
        let w = sm_core::BUILD;
        let stamped = format!("{w}+1b5b54a.dirty");
        let c = path_check(Some(&on_path(Some(&stamped))));
        assert!(c.passed, "{c:?}");
        assert!(c.detail.contains(&stamped), "{c:?}");
        assert!(path_check(Some(&on_path(Some(w)))).passed);
        let other = path_check(Some(&on_path(Some("2.9.9+1b5b54a"))));
        assert!(!other.passed, "{other:?}");
        assert!(other.next_action.is_some(), "{other:?}");
        assert!(!path_check(Some(&on_path(None))).passed);
        assert!(!path_check(None).passed);
    }

    /// B-231: the `commit` check passes when the daemon was built from the
    /// commit the program on PATH was, fails when it names another or none
    /// (a daemon from before B-231), and does not compare when either side
    /// cannot say.
    #[test]
    fn s16_b231_commit_check_compares_the_daemon_with_the_program_on_path() {
        let w = sm_core::BUILD;
        let at = |c: &str| format!("{w}+{c}");
        let same = commit_check(
            &daemon_built_from(Some("1b5b54a")),
            Some(&on_path(Some(&at("1b5b54a")))),
        );
        assert!(same.passed, "{same:?}");
        assert_eq!(same.next_action, None);
        assert!(same.detail.contains("both commit 1b5b54a"), "{same:?}");
        assert!(!same.detail.contains("uncommitted"), "{same:?}");

        let dirty = commit_check(
            &daemon_built_from(Some("1b5b54a.dirty")),
            Some(&on_path(Some(&at("1b5b54a.dirty")))),
        );
        assert!(dirty.passed, "{dirty:?}");
        assert!(dirty.detail.contains("may still differ"), "{dirty:?}");

        for daemon in [Some("f9acf44"), Some("1b5b54a.dirty"), None] {
            let c = commit_check(
                &daemon_built_from(daemon),
                Some(&on_path(Some(&at("1b5b54a")))),
            );
            assert!(!c.passed, "{daemon:?}: {c:?}");
            assert_eq!(c.next_action, Some(commit_next("default")));
            assert!(c.detail.contains("pid 42"), "{c:?}");
            assert!(c.detail.contains("commit 1b5b54a"), "{c:?}");
            match daemon {
                Some(d) => assert!(c.detail.contains(&format!("runs commit {d}")), "{c:?}"),
                None => assert!(c.detail.contains("reports no commit"), "{c:?}"),
            }
        }

        let not_compared = [
            commit_check(&daemon_built_from(Some("1b5b54a")), None),
            commit_check(&daemon_built_from(Some("1b5b54a")), Some(&on_path(None))),
            commit_check(&daemon_built_from(Some("1b5b54a")), Some(&on_path(Some(w)))),
            commit_check(
                &daemon_built_from(Some("1b5b54a")),
                Some(&on_path(Some(&at(version::UNKNOWN)))),
            ),
            commit_check(
                &daemon_built_from(Some(version::UNKNOWN)),
                Some(&on_path(Some(&at("1b5b54a")))),
            ),
            commit_check(&daemon_built_from(None), Some(&on_path(Some(w)))),
        ];
        for c in not_compared {
            assert!(c.passed, "{c:?}");
            assert!(c.detail.starts_with("not compared: "), "{c:?}");
            assert_eq!(c.next_action, None, "{c:?}");
        }
    }

    /// B-231: the next action stops and starts the daemon doctor asked,
    /// so it names that daemon's profile, the default one too: a command
    /// without `--profile` may take another from `SPATIAL_MEMORY_PROFILE`.
    #[test]
    fn s16_b231_commit_next_names_the_daemons_profile() {
        let work = commit_next("work");
        assert!(
            work.contains("`spatial-memory stop --profile work`"),
            "{work}"
        );
        assert!(
            work.contains("`spatial-memory status --profile work`"),
            "{work}"
        );
        let default = commit_next(sm_core::config::DEFAULT_PROFILE);
        assert!(
            default.contains("`spatial-memory stop --profile default`"),
            "{default}"
        );
        assert!(
            default.contains("`spatial-memory status --profile default`"),
            "{default}"
        );

        let mut daemon = daemon_built_from(Some("f9acf44"));
        daemon.profile = "work".into();
        let c = commit_check(
            &daemon,
            Some(&on_path(Some(&format!("{}+1b5b54a", sm_core::BUILD)))),
        );
        assert!(!c.passed, "{c:?}");
        assert_eq!(c.next_action, Some(work));
    }

    /// B-231: a daemon of another release or protocol fails the `build`
    /// check, whose next action differs; the `commit` check does not fail
    /// it a second time with a next action that would not help.
    #[test]
    fn s16_b231_commit_check_leaves_another_build_to_the_build_check() {
        let on = on_path(Some(&format!("{}+1b5b54a", sm_core::BUILD)));
        let mut release = daemon_built_from(Some("f9acf44"));
        release.build = "2.9.9".into();
        let mut protocol = daemon_built_from(None);
        protocol.protocol = PROTOCOL_VERSION - 1;
        for daemon in [release, protocol] {
            let c = commit_check(&daemon, Some(&on));
            assert!(c.passed, "{c:?}");
            assert!(c.detail.starts_with("not compared: "), "{c:?}");
            assert!(c.detail.contains("build check"), "{c:?}");
            assert_eq!(c.next_action, None, "{c:?}");
        }
    }

    /// B-231: the program on PATH of another release than the daemon fails
    /// the `path` check, whose next action differs; the `commit` check does
    /// not fail it a second time with one that would start the daemon from
    /// that program.
    #[test]
    fn s16_b231_commit_check_leaves_another_release_on_path_to_the_path_check() {
        let daemon = daemon_built_from(Some("f9acf44"));
        let other = on_path(Some("2.9.9+1b5b54a"));
        assert!(!path_check(Some(&other)).passed);
        let c = commit_check(&daemon, Some(&other));
        assert!(c.passed, "{c:?}");
        assert!(c.detail.starts_with("not compared: "), "{c:?}");
        assert!(c.detail.contains("release 2.9.9"), "{c:?}");
        assert!(c.detail.contains("path check"), "{c:?}");
        assert_eq!(c.next_action, None, "{c:?}");
    }

    /// B-231: git gives a short hash more than 7 characters where 7 are
    /// ambiguous in that clone, so two builds of one commit may print it at
    /// two lengths. They pass; a `.dirty` mark on one side only, or another
    /// hash, fails.
    #[test]
    fn s16_b231_commit_check_reads_one_commit_at_two_lengths() {
        let at = |c: &str| on_path(Some(&format!("{}+{c}", sm_core::BUILD)));
        for (daemon, path) in [("1b5b54ab", "1b5b54a"), ("1b5b54a", "1b5b54ab")] {
            let c = commit_check(&daemon_built_from(Some(daemon)), Some(&at(path)));
            assert!(c.passed, "{daemon} {path}: {c:?}");
            assert!(c.detail.contains("one commit"), "{c:?}");
        }
        let dirty = commit_check(
            &daemon_built_from(Some("1b5b54ab.dirty")),
            Some(&at("1b5b54a.dirty")),
        );
        assert!(dirty.passed, "{dirty:?}");
        assert!(dirty.detail.contains("may still differ"), "{dirty:?}");
        for (daemon, path) in [
            ("1b5b54ab.dirty", "1b5b54a"),
            ("1b5b54ab", "1b5b54a.dirty"),
            ("1b5b54c", "1b5b54a"),
            ("1b5b54c1", "1b5b54a"),
        ] {
            let c = commit_check(&daemon_built_from(Some(daemon)), Some(&at(path)));
            assert!(!c.passed, "{daemon} {path}: {c:?}");
        }
        assert!(!same_commit("", "1b5b54a"));
        assert!(!same_commit(".dirty", "1b5b54a.dirty"));
    }
}
