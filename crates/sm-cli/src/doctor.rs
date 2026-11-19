//! `spatial-memory doctor` (SPEC section 5 rule 7, section 16).
//!
//! Each check passes or fails with a next action: the profile and its
//! folders, the daemon (it answers `hello`, its build matches, its state),
//! the index and the journal's measurements from `status`, the embedder's
//! numbers timed by the daemon (`doctor_numbers`, B-125), `spatial-memory`
//! on `PATH` at the plugin's version, the plugin and `setup` not both
//! registered, the spool, and the model step (decision B-43). The command
//! exits 1 when any check fails.
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
            out.push(path_check(env));
            out.push(registration_check(env));
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
    out.extend(daemon_checks(&connector, &paths.daemon_log(), budget));
    out.push(path_check(env));
    out.push(registration_check(env));
    out.push(spool_check(&paths.spool_new(), &paths.spool_failed()));
    if let Some(settings) = &settings {
        out.push(match env.model.check(settings) {
            Ok(line) => Check::pass("model", line),
            Err(e) => Check::from_error("model", &e),
        });
    }
    out
}

fn daemon_checks(connector: &Connector<'_>, log: &Path, budget: Duration) -> Vec<Check> {
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
            return vec![Check::fail(
                "daemon",
                format!("the daemon answered hello with a {kind} reply"),
                see_log,
            )];
        }
        Ok(None) => {
            return vec![Check::fail(
                "daemon",
                "the daemon is listening but answered hello with an error",
                see_log,
            )];
        }
        Err(e) => {
            return vec![Check::fail(
                "daemon",
                format!("{}: {}", e.code, e.message),
                format!("{} {see_log}", e.next_action),
            )];
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
    if info.build == sm_core::BUILD && info.protocol == PROTOCOL_VERSION {
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
    out
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

fn path_check(env: &DoctorEnv<'_>) -> Check {
    let want = plugin_version();
    let Some(found) = crate::locate::on_path("spatial-memory", env.get_env) else {
        return Check::fail(
            "path",
            "`spatial-memory` is not on PATH, so the plugin's hooks and server cannot start",
            "Put the folder holding spatial-memory on PATH, or run `spatial-memory setup`, \
             which writes the absolute path.",
        );
    };
    let output = (env.version_of)(&found);
    let version = output
        .as_deref()
        .and_then(|o| o.split_whitespace().last())
        .map(String::from);
    match (version, want) {
        (Some(v), Some(w)) if v == w => Check::pass(
            "path",
            format!("{} is version {v}, as the plugin expects", found.display()),
        ),
        (Some(v), Some(w)) => Check::fail(
            "path",
            format!("{} is version {v}; the plugin expects {w}", found.display()),
            format!("Install spatial-memory {w} on PATH, or update the plugin to match."),
        ),
        (None, _) => Check::fail(
            "path",
            format!("{} did not report a version", found.display()),
            "Reinstall spatial-memory; `spatial-memory --version` must print the version.",
        ),
        (Some(v), None) => Check::pass("path", format!("{} is version {v}", found.display())),
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
}
