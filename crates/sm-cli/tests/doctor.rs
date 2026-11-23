//! `doctor`'s embedder line (SPEC 5 rule 7, decision B-125): the daemon
//! times its own sessions with `doctor_numbers` and `doctor` prints both
//! numbers; a daemon that stays loading gives "not measured" with the next
//! action, which is not a failure. `doctor::checks` runs in this process
//! with a fake `claude` and a fake model step, because the binary's
//! `doctor` would start the real `claude` from PATH; the daemon is either
//! the production daemon in this process with the fake model, or the mock.

mod common;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::{BIN, CountingSpawn, MockDaemon, Mode, Sandbox, daemon_info, loading_error};
use sm_cli::doctor::{self, Check, DoctorEnv, NUMBERS_LATER};
use sm_cli::model_step::ModelStep;
use sm_cli::setup::{ClaudeRunner, RunOutput};
use sm_core::config::{ProfileName, Settings};
use sm_core::error::{CoreError, Result};
use sm_core::paths::{HOME_ENV, ProfilePaths};
use sm_core::protocol::{
    DaemonState, DoctorNumbersReply, IndexStatus, ModelStatus, Op, Reply, Request, StatusReply,
};
use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
use sm_core::traits::EmbedderInfo;
use sm_daemon::{
    DaemonHandle, DaemonOptions, Judge, JudgeAnswer, JudgeCandidate, JudgeFailure,
    JudgeFailureKind, LoadedModel, ModelLoader, RepeatAnswer, RepeatItem, SameAnswer, SameItem,
    ShareAdapter, Started, StatementsAnswer,
};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A `claude` that answers the one read-only question `doctor` asks, with
/// this plugin list.
struct FakeClaude(String);

impl FakeClaude {
    fn empty() -> Self {
        FakeClaude(String::from("[]"))
    }
}

impl ClaudeRunner for FakeClaude {
    fn run(&self, args: &[String]) -> io::Result<RunOutput> {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        assert_eq!(a, ["plugin", "list", "--json"], "doctor only reads");
        Ok(RunOutput {
            success: true,
            stdout: self.0.clone(),
            stderr: String::new(),
        })
    }
}

/// A model step that reports the model present without touching one.
struct FakeModelStep;

impl ModelStep for FakeModelStep {
    fn ensure(&self, _settings: &Settings) -> std::result::Result<String, CoreError> {
        panic!("doctor never downloads the model")
    }
    fn check(&self, _settings: &Settings) -> std::result::Result<String, CoreError> {
        Ok(String::from("fake model present"))
    }
}

/// The fake model, loaded at once.
struct FakeLoader;

impl ModelLoader for FakeLoader {
    fn info(&self) -> EmbedderInfo {
        fake_info()
    }

    fn load(&self) -> Result<LoadedModel> {
        Ok(LoadedModel {
            provider: Arc::new(FakeProvider::new()),
            chunker: Arc::new(SimpleChunker),
        })
    }
}

/// A capture judge that can run, as the binary's daemon has one (B-168).
/// These tests capture nothing, so a call, the judge's or the duplicate
/// check's (B-192), is a test fault: it fails, and the judge check would
/// show the failure.
#[derive(Debug)]
struct IdleJudge;

impl Judge for IdleJudge {
    fn model(&self) -> &str {
        "haiku"
    }

    fn command(&self) -> Option<String> {
        Some(String::from("claude-test"))
    }

    fn judge(
        &self,
        _batch: &[JudgeCandidate],
        _stop: &AtomicBool,
    ) -> std::result::Result<JudgeAnswer, JudgeFailure> {
        Err(JudgeFailure {
            kind: JudgeFailureKind::CallFailed,
            message: String::from("the doctor tests capture nothing to judge"),
            retry_at: None,
            cost_usd: 0.0,
        })
    }

    fn repeats(
        &self,
        _items: &[RepeatItem],
        _stop: &AtomicBool,
    ) -> std::result::Result<RepeatAnswer, JudgeFailure> {
        Err(JudgeFailure {
            kind: JudgeFailureKind::CallFailed,
            message: String::from("the doctor tests capture nothing to check for duplicates"),
            retry_at: None,
            cost_usd: 0.0,
        })
    }

    fn statements(
        &self,
        _batch: &[JudgeCandidate],
        _stop: &AtomicBool,
    ) -> std::result::Result<StatementsAnswer, JudgeFailure> {
        Err(JudgeFailure {
            kind: JudgeFailureKind::CallFailed,
            message: String::from("the doctor tests capture nothing to split into statements"),
            retry_at: None,
            cost_usd: 0.0,
        })
    }

    fn same(
        &self,
        _items: &[SameItem],
        _stop: &AtomicBool,
    ) -> std::result::Result<SameAnswer, JudgeFailure> {
        Err(JudgeFailure {
            kind: JudgeFailureKind::CallFailed,
            message: String::from("the doctor tests capture no statement to check"),
            retry_at: None,
            cost_usd: 0.0,
        })
    }
}

/// The production daemon in this process on the sandbox's profile, with
/// the fake model, a judge and 8 physical cores, once it is ready.
fn daemon(sb: &Sandbox) -> DaemonHandle {
    daemon_with(sb, None)
}

/// [`daemon`], reporting `commit` in `hello` as a daemon built from it
/// does (B-231).
fn daemon_with(sb: &Sandbox, commit: Option<&str>) -> DaemonHandle {
    let paths = sb.paths();
    let opts = DaemonOptions {
        lock_wait: Duration::from_millis(300),
        bind_wait: Duration::from_millis(300),
        retry_every: Duration::from_millis(10),
        physical_cores: Some(8),
        shutdown_grace: Duration::from_secs(5),
        judge: Some(Arc::new(IdleJudge)),
        commit: commit.map(String::from),
        ..DaemonOptions::default()
    };
    let started = sm_daemon::start(
        paths.clone(),
        Settings::default(),
        Arc::new(FakeLoader),
        Arc::new(ShareAdapter),
        opts,
    )
    .expect("the daemon starts");
    let Started::Running(handle) = started else {
        panic!("another daemon holds the sandbox's lock");
    };
    let hello = Request::new(String::new(), paths.profile.clone(), Op::Hello);
    let start = Instant::now();
    loop {
        let state =
            sm_cli::client::exchange(handle.endpoint(), &hello, Some(Duration::from_secs(5)))
                .ok()
                .and_then(|r| r.into_result().ok());
        if let Some(Reply::Hello(info)) = state
            && info.state == DaemonState::Ready
        {
            return handle;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the fake model loaded"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The folder holding the binary under test, so the `path` check finds it.
fn bin_dir() -> PathBuf {
    Path::new(BIN).parent().expect("a folder").to_path_buf()
}

/// Runs every check against the sandbox's profile. PATH holds only the
/// binary's folder and the version it reports is the plugin's, so every
/// check but the daemon's can pass.
fn run_checks(sb: &Sandbox, spawner: &CountingSpawn, window: Duration) -> Vec<Check> {
    run_checks_with(sb, spawner, window, &FakeClaude::empty())
}

fn run_checks_with(
    sb: &Sandbox,
    spawner: &CountingSpawn,
    window: Duration,
    claude: &FakeClaude,
) -> Vec<Check> {
    run_checks_in(sb, spawner, window, claude, &sb.tmp, Some(SHORT_COLD_START))
}

/// The cold-start budget the tests give a daemon that stays loading, so
/// they finish inside their window; `None` is doctor's own 5 s (B-152).
const SHORT_COLD_START: Duration = Duration::from_millis(200);

fn run_checks_in(
    sb: &Sandbox,
    spawner: &CountingSpawn,
    window: Duration,
    claude: &FakeClaude,
    cwd: &Path,
    cold_start: Option<Duration>,
) -> Vec<Check> {
    let version_of = |_: &Path| doctor::plugin_version().map(|v| format!("spatial-memory {v}"));
    run_checks_versioned(sb, spawner, window, claude, cwd, cold_start, &version_of)
}

/// [`run_checks_in`] with `version_of` saying what the program on PATH
/// prints for `--version`.
fn run_checks_versioned(
    sb: &Sandbox,
    spawner: &CountingSpawn,
    window: Duration,
    claude: &FakeClaude,
    cwd: &Path,
    cold_start: Option<Duration>,
    version_of: &dyn Fn(&Path) -> Option<String>,
) -> Vec<Check> {
    let home = sb.home.to_string_lossy().into_owned();
    let path = bin_dir().to_string_lossy().into_owned();
    let get_env = move |k: &str| match k {
        HOME_ENV => Some(home.clone()),
        "PATH" => Some(path.clone()),
        "PATHEXT" => Some(String::from(".EXE")),
        _ => None,
    };
    let env = DoctorEnv {
        get_env: &get_env,
        cwd: cwd.to_path_buf(),
        home: Some(sb.user_home.clone()),
        spawner,
        runner: claude,
        model: &FakeModelStep,
        version_of,
        window: Some(window),
        cold_start,
    };
    doctor::checks(&env, None)
}

/// [`run_checks`] as `doctor --profile <profile>` runs them.
fn run_checks_for_profile(
    sb: &Sandbox,
    spawner: &CountingSpawn,
    window: Duration,
    profile: &str,
) -> Vec<Check> {
    let home = sb.home.to_string_lossy().into_owned();
    let path = bin_dir().to_string_lossy().into_owned();
    let get_env = move |k: &str| match k {
        HOME_ENV => Some(home.clone()),
        "PATH" => Some(path.clone()),
        "PATHEXT" => Some(String::from(".EXE")),
        _ => None,
    };
    let version_of = |_: &Path| doctor::plugin_version().map(|v| format!("spatial-memory {v}"));
    let claude = FakeClaude::empty();
    let env = DoctorEnv {
        get_env: &get_env,
        cwd: sb.tmp.clone(),
        home: Some(sb.user_home.clone()),
        spawner,
        runner: &claude,
        model: &FakeModelStep,
        version_of: &version_of,
        window: Some(window),
        cold_start: Some(SHORT_COLD_START),
    };
    doctor::checks(&env, Some(profile))
}

fn embedder(checks: &[Check]) -> &Check {
    checks
        .iter()
        .find(|c| c.name == "embedder")
        .unwrap_or_else(|| panic!("no embedder line in {checks:#?}"))
}

fn without_embedder(checks: &[Check]) -> Vec<Check> {
    checks
        .iter()
        .filter(|c| c.name != "embedder")
        .cloned()
        .collect()
}

fn doctor_numbers_requests(mock: &MockDaemon) -> usize {
    mock.requests()
        .iter()
        .filter(|r| matches!(r.call, Op::DoctorNumbers))
        .count()
}

/// SPEC 5 rule 7 with B-125: against the production daemon (fake model,
/// 8 physical cores) `doctor` prints both numbers, each with the thread
/// count the settings give for 8 cores, the bulk batch size and the query
/// token count, and every check passes.
#[test]
fn s5_rule7_doctor_prints_both_numbers_from_an_in_process_daemon() {
    let _serial = serial();
    let sb = Sandbox::with_daemon();
    let _daemon = daemon(&sb);
    let spawner = CountingSpawn::default();
    let checks = run_checks(&sb, &spawner, Duration::from_secs(10));
    let report = doctor::render(&checks);
    println!("{report}");
    let settings = Settings::default();
    let e = embedder(&checks);
    assert!(e.passed, "{e:?}");
    assert_eq!(e.next_action, None, "{e:?}");
    let bulk = format!(
        "chunks/s on {} threads, batch {} (",
        settings.bulk_threads(8),
        settings.embed.batch_size
    );
    let query = format!(
        "at {} tokens on {} threads (",
        sm_embed::doctor::QUERY_TOKENS,
        settings.query_threads(8)
    );
    assert!(e.detail.starts_with("bulk "), "{}", e.detail);
    assert!(e.detail.contains(&bulk), "{} lacks {bulk}", e.detail);
    assert!(e.detail.contains("; query median "), "{}", e.detail);
    assert!(e.detail.contains(&query), "{} lacks {query}", e.detail);
    assert!(e.detail.ends_with("; 8 physical cores"), "{}", e.detail);
    assert!(report.contains("ok   embedder: bulk "), "{report}");
    assert_eq!(doctor::exit_code(&checks), 0, "{report}");
    assert_eq!(spawner.count.load(Ordering::SeqCst), 0, "a daemon answered");
}

/// SPEC 5 rule 7 with B-125: a daemon that stays loading for the whole
/// window gives "not measured" with the next action. `doctor` asked again
/// while it waited, the line is not a failure, and the exit code is the one
/// the other checks give (here 1, from the `state` check: still loading
/// after the cold-start budget, which this test sets to 200 ms; B-152).
#[test]
fn s5_rule7_doctor_says_not_measured_while_the_daemon_stays_loading() {
    let _serial = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Loading);
    let spawner = CountingSpawn::default();
    let start = Instant::now();
    let checks = run_checks(&sb, &spawner, Duration::from_millis(500));
    let took = start.elapsed();
    let report = doctor::render(&checks);
    println!("{report}");
    let e = embedder(&checks);
    assert!(e.passed, "{e:?}");
    assert!(e.detail.starts_with("not measured: "), "{}", e.detail);
    assert_eq!(e.next_action.as_deref(), Some(NUMBERS_LATER));
    assert!(report.contains("ok   embedder: not measured"), "{report}");
    assert!(
        report.contains(&format!("next: {NUMBERS_LATER}")),
        "{report}"
    );
    assert_eq!(
        doctor::exit_code(&checks),
        doctor::exit_code(&without_embedder(&checks))
    );
    assert_eq!(doctor::exit_code(&checks), 1, "the state check fails");
    let state = checks
        .iter()
        .find(|c| c.name == "state")
        .expect("a state line");
    assert!(
        !state.passed && state.detail.contains("cold-start budget"),
        "{state:?}"
    );
    assert!(doctor_numbers_requests(&mock) >= 2, "doctor asked again");
    assert!(
        took < Duration::from_secs(5),
        "{took:?}: it gave up in the window"
    );
    assert_eq!(spawner.count.load(Ordering::SeqCst), 0, "a daemon answered");
}

/// SPEC 5 rule 7 with B-125: a ready daemon whose numbers are still not
/// available (it answers `doctor_numbers` with `daemon_loading` for the
/// whole window) leaves every other check passing, and `doctor` exits 0,
/// as it would without the line.
#[test]
fn s5_rule7_doctor_exits_0_when_only_the_numbers_were_not_measured() {
    let _serial = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    mock.set_responder(Arc::new(|request: &Request| match &request.call {
        Op::Status(_) => Some(Ok(Reply::Status(Box::new(StatusReply {
            daemon: daemon_info(request, DaemonState::Ready, sm_core::BUILD),
            model: ModelStatus {
                model_id: String::from("fake"),
                revision: String::from("0"),
                dim: 8,
                loaded: true,
            },
            index: IndexStatus {
                matches: true,
                differences: Vec::new(),
                chunks: 0,
                pending_embeddings: 0,
            },
            current_project: None,
            projects: Vec::new(),
            database: String::from("memory.db"),
            database_bytes: 0,
            spool_pending: 0,
            captures_queued: 0,
            judge: judge_ok(),
            metrics: Some(Vec::new()),
            omitted: 0,
            client_build: None,
        })))),
        Op::DoctorNumbers => Some(Err(loading_error())),
        _ => None,
    }));
    let spawner = CountingSpawn::default();
    let checks = run_checks(&sb, &spawner, Duration::from_millis(500));
    let report = doctor::render(&checks);
    println!("{report}");
    let e = embedder(&checks);
    assert!(e.passed, "{e:?}");
    assert!(e.detail.starts_with("not measured: "), "{}", e.detail);
    assert_eq!(e.next_action.as_deref(), Some(NUMBERS_LATER));
    assert_eq!(doctor::exit_code(&without_embedder(&checks)), 0, "{report}");
    assert_eq!(doctor::exit_code(&checks), 0, "{report}");
    assert!(doctor_numbers_requests(&mock) >= 2, "doctor asked again");
}

/// SPEC 16 with B-234: a daemon of this program's protocol and release
/// that answers `doctor_numbers` with another kind of reply fails the
/// `embedder` check, and the detail names the kind. Its next action names
/// the stop that replaces the daemon, then `status`, and the daemon's log.
/// Running doctor again would only ask the same daemon: no client replaces
/// a daemon of its own release, and this one passed the envelope's check.
/// doctor asks once, starts nothing, and every other check passes. The
/// profile is not the default one, and the daemon's `hello` names yet
/// another, so the next action is shown to name the profile doctor asked
/// at, whose log it names too, and neither a fixed one nor the daemon's.
#[test]
fn s16_b234_a_wrong_reply_to_doctor_numbers_names_the_stop_that_replaces_the_daemon() {
    let _serial = serial();
    let sb = Sandbox::new();
    let paths = ProfilePaths::new(sb.home.clone(), ProfileName::parse("work").unwrap());
    let mock = MockDaemon::start(&paths, Mode::Answer);
    mock.set_responder(Arc::new(|request: &Request| match &request.call {
        Op::Hello => Some(Ok(Reply::Hello(hello_naming(
            request,
            sm_core::BUILD,
            "elsewhere",
        )))),
        Op::Status(_) | Op::DoctorNumbers => {
            Some(Ok(Reply::Status(Box::new(ready_status(request)))))
        }
        _ => None,
    }));
    let spawner = CountingSpawn::default();
    let checks = run_checks_for_profile(&sb, &spawner, Duration::from_millis(500), "work");
    let report = doctor::render(&checks);
    println!("{report}");
    assert!(
        report.starts_with("ok   profile: work (from --profile)"),
        "{report}"
    );
    let e = embedder(&checks);
    assert!(!e.passed, "{e:?}");
    assert_eq!(
        e.detail, "the daemon answered doctor_numbers with a status reply",
        "{e:?}"
    );
    let next = e.next_action.as_deref().unwrap_or_default();
    for want in [
        String::from("Run `spatial-memory stop --profile work`, then "),
        String::from("`spatial-memory status --profile work`"),
        format!(
            "see {} for the daemon's own report",
            paths.daemon_log().display()
        ),
    ] {
        assert!(next.contains(&want), "{next:?} lacks {want:?}");
    }
    assert!(
        !next.contains("a client replaces a daemon of another build"),
        "{next:?}"
    );
    assert!(!next.contains("Run the command again"), "{next:?}");
    assert!(!next.contains("elsewhere"), "{next:?}");
    assert_eq!(doctor_numbers_requests(&mock), 1, "doctor asked once");
    assert_eq!(doctor::exit_code(&without_embedder(&checks)), 0, "{report}");
    assert_eq!(doctor::exit_code(&checks), 1, "{report}");
    assert_eq!(spawner.count.load(Ordering::SeqCst), 0, "a daemon answered");
}

/// SPEC 16 with B-234: a daemon whose `hello` names another build that
/// does not rank newer, under this program's envelope, fails the `build`
/// check. Its next action names the stop that replaces the daemon at the
/// profile doctor asked at, then `status`, then that profile's log, not a
/// rerun: the envelope matched, so no client replaces that daemon, and
/// doctor sends it no `shutdown`. doctor starts nothing, and every other
/// check passes. The daemon's `hello` names another profile than doctor's,
/// so the next action is shown to name doctor's.
#[test]
fn s16_b234_a_hello_naming_another_build_names_the_stop_that_replaces_the_daemon() {
    let _serial = serial();
    let sb = Sandbox::new();
    let paths = ProfilePaths::new(sb.home.clone(), ProfileName::parse("work").unwrap());
    let mock = MockDaemon::start(&paths, Mode::Answer);
    mock.set_responder(Arc::new(|request: &Request| match &request.call {
        Op::Hello => Some(Ok(Reply::Hello(hello_naming(
            request,
            "2.9.9",
            "elsewhere",
        )))),
        Op::Status(_) => Some(Ok(Reply::Status(Box::new(ready_status(request))))),
        Op::DoctorNumbers => Some(Ok(Reply::DoctorNumbers(some_numbers()))),
        _ => None,
    }));
    let spawner = CountingSpawn::default();
    let checks = run_checks_for_profile(&sb, &spawner, Duration::from_millis(500), "work");
    let report = doctor::render(&checks);
    println!("{report}");
    let b = named(&checks, "build");
    assert!(!b.passed, "{b:?}");
    assert!(
        b.detail.starts_with("the daemon is build 2.9.9 protocol "),
        "{b:?}"
    );
    let next = b.next_action.as_deref().unwrap_or_default();
    for want in [
        String::from("Run `spatial-memory stop --profile work`, then "),
        String::from("`spatial-memory status --profile work`"),
        format!(
            "see {} for the daemon's own report",
            paths.daemon_log().display()
        ),
    ] {
        assert!(next.contains(&want), "{next:?} lacks {want:?}");
    }
    assert!(!next.contains("Run the command again"), "{next:?}");
    assert!(!next.contains("elsewhere"), "{next:?}");
    let ops: Vec<&str> = mock.requests().iter().map(|r| r.call.name()).collect();
    assert!(!ops.contains(&"shutdown"), "{ops:?}");
    let others: Vec<Check> = checks
        .iter()
        .filter(|c| c.name != "build")
        .cloned()
        .collect();
    assert_eq!(doctor::exit_code(&others), 0, "{report}");
    assert_eq!(spawner.count.load(Ordering::SeqCst), 0, "a daemon answered");
}

/// A ready `hello` of this build's protocol that names `build` and
/// `profile`, whatever the request's profile was.
fn hello_naming(request: &Request, build: &str, profile: &str) -> sm_core::protocol::DaemonInfo {
    let mut info = daemon_info(request, DaemonState::Ready, build);
    profile.clone_into(&mut info.profile);
    info
}

/// Embedder numbers for a daemon that measured them.
fn some_numbers() -> DoctorNumbersReply {
    DoctorNumbersReply {
        physical_cores: 8,
        query_threads: 4,
        bulk_threads: 6,
        batch_size: 4,
        sample_chunks: 4,
        query_tokens: 128,
        query_runs: 5,
        bulk_elapsed_ms: 100.0,
        bulk_chunks_per_sec: 40.0,
        query_median_ms: 20.0,
    }
}

fn registration(checks: &[Check]) -> &Check {
    checks
        .iter()
        .find(|c| c.name == "registration")
        .unwrap_or_else(|| panic!("no registration line in {checks:#?}"))
}

/// B-150: the version 1 plugin enabled for this folder is a failure, never
/// taken for the version 3 plugin; disabled at local scope it is not there.
#[test]
fn s16_doctor_fails_while_the_version_1_plugin_is_enabled_for_the_folder() {
    let _serial = serial();
    let sb = Sandbox::new();
    let key = "spatial-memory@spatial-memory-marketplace";
    let claude_dir = sb.tmp.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    )
    .unwrap();
    let list = serde_json::json!([{
        "id": key,
        "version": "1.11.4",
        "scope": "project",
        "enabled": true,
        "projectEnabled": true,
        "projectPath": sb.tmp.to_string_lossy(),
    }])
    .to_string();
    let claude = FakeClaude(list);
    let spawner = CountingSpawn::default();
    let checks = run_checks_with(&sb, &spawner, Duration::from_millis(200), &claude);
    let r = registration(&checks);
    assert!(!r.passed, "{r:?}");
    assert!(r.detail.contains("version 1 plugin"), "{r:?}");
    assert!(r.detail.contains("1.11.4"), "{r:?}");
    assert!(
        r.detail.contains("claude plugin disable"),
        "the line says how: {r:?}"
    );

    // The six projects' state: disabled in the local settings.
    std::fs::write(
        claude_dir.join("settings.local.json"),
        format!(r#"{{"enabledPlugins": {{"{key}": false}}}}"#),
    )
    .unwrap();
    let checks = run_checks_with(&sb, &spawner, Duration::from_millis(200), &claude);
    let r = registration(&checks);
    assert!(r.passed, "{r:?}");
    assert!(!r.detail.contains("version 1"), "{r:?}");
}

fn named<'a>(checks: &'a [Check], name: &str) -> &'a Check {
    checks
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no {name} line in {checks:#?}"))
}

/// `cwd` as Claude Code may spell it in `~/.claude.json`: on Windows with
/// forward slashes and another case, as the entry found on 2026-11-21 was.
fn spelled_otherwise(cwd: &Path) -> String {
    let s = cwd.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.replace('\\', "/").to_uppercase()
    } else {
        s
    }
}

/// Writes `~/.claude.json` with the user-scope server and these projects'
/// `disabledMcpServers` lists.
fn write_claude_json(sb: &Sandbox, registered: bool, projects: &[(&str, &[&str])]) -> PathBuf {
    let mut map = serde_json::Map::new();
    if registered {
        map.insert(
            String::from("mcpServers"),
            serde_json::json!({"spatial-memory": {"type": "stdio", "command": BIN, "args": ["mcp"]}}),
        );
    }
    let mut entries = serde_json::Map::new();
    for (key, disabled) in projects {
        entries.insert(
            (*key).to_owned(),
            serde_json::json!({"allowedTools": [], "disabledMcpServers": disabled}),
        );
    }
    map.insert(String::from("projects"), serde_json::Value::Object(entries));
    let path = sb.user_home.join(".claude.json");
    std::fs::write(&path, serde_json::to_string_pretty(&map).unwrap()).unwrap();
    path
}

/// B-226: a project entry in `~/.claude.json` whose `disabledMcpServers`
/// names spatial-memory hides the user-scope server in that folder, as an
/// entry left from version 1 did in this repository while doctor passed.
/// For this folder it fails, with the /mcp next action; for other folders
/// it passes and names them; doctor never edits the file.
#[test]
fn s16_b226_doctor_fails_while_the_server_is_disabled_for_this_folder() {
    let _serial = serial();
    let sb = Sandbox::new();
    let spawner = CountingSpawn::default();
    let window = Duration::from_millis(200);
    let here = spelled_otherwise(&sb.tmp);
    let other = sb
        .user_home
        .join("elsewhere")
        .to_string_lossy()
        .into_owned();

    // No file: nothing disables it.
    let checks = run_checks(&sb, &spawner, window);
    let c = named(&checks, "mcp_enabled");
    assert!(c.passed, "{c:?}");
    assert_eq!(c.next_action, None, "{c:?}");

    // Registered, and this folder's entry names it beside another server.
    let file = write_claude_json(&sb, true, &[(&here, &["other", "spatial-memory"])]);
    let before = std::fs::read(&file).unwrap();
    let checks = run_checks(&sb, &spawner, window);
    let c = named(&checks, "mcp_enabled");
    let report = doctor::render(&checks);
    assert!(!c.passed, "{report}");
    assert_eq!(doctor::exit_code(&checks), 1, "{report}");
    assert!(c.detail.contains("disabledMcpServers"), "{c:?}");
    assert!(c.detail.contains(&here), "the entry is named: {c:?}");
    assert!(
        c.detail.contains(&file.display().to_string()),
        "the file is named: {c:?}"
    );
    let next = c.next_action.as_deref().unwrap_or_default();
    assert!(next.contains("/mcp") && next.contains("Enable"), "{c:?}");
    assert_eq!(std::fs::read(&file).unwrap(), before, "doctor only reads");
    assert!(
        registration(&checks).passed,
        "the registration itself is fine"
    );

    // Another server only, or not registered: this folder is not hidden.
    write_claude_json(&sb, true, &[(&here, &["other"])]);
    let checks = run_checks(&sb, &spawner, window);
    let c = named(&checks, "mcp_enabled");
    assert!(c.passed, "{c:?}");
    assert_eq!(c.next_action, None, "{c:?}");

    // Only another folder: a pass that names it and says what to do there.
    write_claude_json(&sb, true, &[(&here, &[]), (&other, &["spatial-memory"])]);
    let checks = run_checks(&sb, &spawner, window);
    let c = named(&checks, "mcp_enabled");
    assert!(c.passed, "{c:?}");
    assert!(c.detail.contains(&other), "{c:?}");
    assert!(!c.detail.contains(&here), "{c:?}");
    let next = c.next_action.as_deref().unwrap_or_default();
    assert!(next.contains("/mcp") && next.contains("Enable"), "{c:?}");

    // Not registered at user scope: nothing is hidden yet, and the entry
    // that would hide it is still named.
    write_claude_json(&sb, false, &[(&here, &["spatial-memory"])]);
    let checks = run_checks(&sb, &spawner, window);
    let c = named(&checks, "mcp_enabled");
    assert!(c.passed, "{c:?}");
    assert!(c.detail.contains(&here), "{c:?}");
    assert!(c.next_action.is_some(), "{c:?}");
}

/// What the binary under test prints for `--version`.
fn bin_version(_: &Path) -> Option<String> {
    let out = std::process::Command::new(BIN)
        .arg("--version")
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// B-231: `spatial-memory --version` prints `<release>+<commit>`. The
/// `path` check compares the release before `+` with the plugin's, and the
/// `commit` check compares the daemon's commit with the one after `+`. The
/// binary under test, against a daemon built from the same commit, passes
/// both and doctor exits 0. Before B-231 the `path` check took the whole
/// stamp for the release and failed.
#[test]
fn s16_b231_doctor_reads_the_release_before_the_plus_and_compares_the_commit() {
    use sm_cli::version::{COMMIT, STAMP};
    let _serial = serial();
    let printed = bin_version(Path::new(BIN)).expect("the binary prints its version");
    assert_eq!(printed, format!("spatial-memory {STAMP}"));
    let sb = Sandbox::with_daemon();
    let _daemon = daemon_with(&sb, Some(COMMIT));
    let spawner = CountingSpawn::default();
    let checks = run_checks_versioned(
        &sb,
        &spawner,
        Duration::from_secs(10),
        &FakeClaude::empty(),
        &sb.tmp,
        Some(SHORT_COLD_START),
        &bin_version,
    );
    let report = doctor::render(&checks);
    println!("{report}");
    let path = named(&checks, "path");
    assert!(path.passed, "{report}");
    assert!(path.detail.contains(STAMP), "{path:?}");
    let commit = named(&checks, "commit");
    assert!(commit.passed, "{report}");
    assert_eq!(commit.next_action, None, "{commit:?}");
    // Without git the build says `unknown`, and doctor does not compare.
    if COMMIT != "unknown" {
        assert!(commit.detail.contains(COMMIT), "{commit:?}");
    }
    assert_eq!(doctor::exit_code(&checks), 0, "{report}");
}

/// B-231: a daemon built from another commit than the program on PATH
/// fails the `commit` check, which names both commits and says how to start
/// the daemon again from PATH. A daemon older than B-231, which reports no
/// commit, fails it too. The `build` and `path` checks still pass: it is one
/// release.
#[test]
fn s16_b231_doctor_fails_while_the_daemon_runs_another_commit() {
    let _serial = serial();
    let on_path = |_: &Path| Some(format!("spatial-memory {}+abc1234", sm_core::BUILD));
    for (daemon_commit, named_as) in [(Some("def5678"), "def5678"), (None, "no commit")] {
        let sb = Sandbox::with_daemon();
        let _daemon = daemon_with(&sb, daemon_commit);
        let spawner = CountingSpawn::default();
        let checks = run_checks_versioned(
            &sb,
            &spawner,
            Duration::from_secs(10),
            &FakeClaude::empty(),
            &sb.tmp,
            Some(SHORT_COLD_START),
            &on_path,
        );
        let report = doctor::render(&checks);
        println!("{report}");
        let c = named(&checks, "commit");
        assert!(!c.passed, "{report}");
        assert!(c.detail.contains("abc1234"), "{c:?}");
        assert!(c.detail.contains(named_as), "{c:?}");
        let next = c.next_action.as_deref().unwrap_or_default();
        assert!(next.contains("spatial-memory stop"), "{c:?}");
        assert!(named(&checks, "build").passed, "{report}");
        assert!(named(&checks, "path").passed, "{report}");
        assert_eq!(doctor::exit_code(&checks), 1, "{report}");
    }
}

/// B-232: doctor meeting a daemon of a newer build sends it no shutdown
/// and starts no daemon although the lock is free. The daemon check fails
/// with `version_mismatch`, naming the daemon's build and how to run the
/// installed program.
#[test]
fn s16_b232_doctor_never_replaces_a_newer_daemon() {
    let _serial = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Newer);
    let spawner = CountingSpawn::default();
    let checks = run_checks(&sb, &spawner, Duration::from_millis(500));
    println!("{}", doctor::render(&checks));
    let d = named(&checks, "daemon");
    assert!(!d.passed, "{d:?}");
    assert!(d.detail.starts_with("version_mismatch: "), "{d:?}");
    let theirs = format!("build {}", common::NEWER_BUILD);
    assert!(d.detail.contains(&theirs), "{d:?}");
    let next = d.next_action.as_deref().unwrap_or_default();
    assert!(
        next.contains("Reconnect") && next.contains("`/mcp`"),
        "{d:?}"
    );
    let ops: Vec<&str> = mock.requests().iter().map(|r| r.call.name()).collect();
    assert!(!ops.contains(&"shutdown"), "{ops:?}");
    assert_eq!(spawner.count(), 0);
    assert!(!mock.is_closed());
}

/// The length and modification time of the database and its write-ahead
/// log, which a committed write changes.
fn store_files(sb: &Sandbox) -> Vec<Option<(u64, std::time::SystemTime)>> {
    let db = sb.paths().database();
    let wal = db.with_file_name("memory.db-wal");
    [db, wal]
        .iter()
        .map(|p| {
            std::fs::metadata(p)
                .ok()
                .map(|m| (m.len(), m.modified().expect("a modification time")))
        })
        .collect()
}

/// B-151 (D18): doctor only reads. Run in a folder the store has never
/// seen, against the production daemon, it passes, the database and its
/// log are as they were, and no project was created for the folder.
#[test]
fn s16_doctor_never_creates_a_project_or_writes_to_the_store() {
    let _serial = serial();
    let sb = Sandbox::with_daemon();
    let handle = daemon(&sb);
    let fresh = sb.tmp.join("never-seen");
    std::fs::create_dir_all(&fresh).unwrap();
    let before = store_files(&sb);
    assert!(before[0].is_some(), "the daemon made the database");
    let spawner = CountingSpawn::default();
    let checks = run_checks_in(
        &sb,
        &spawner,
        Duration::from_secs(10),
        &FakeClaude::empty(),
        &fresh,
        Some(SHORT_COLD_START),
    );
    let report = doctor::render(&checks);
    println!("{report}");
    assert_eq!(doctor::exit_code(&checks), 0, "{report}");
    assert_eq!(store_files(&sb), before, "doctor wrote to the store");
    drop(handle);
    let store = sm_store::Store::open(&sb.paths()).expect("the store opens");
    let names: Vec<String> = store
        .projects()
        .expect("the projects")
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert!(names.is_empty(), "doctor created {names:?}");
}

/// A judge that can run, with nothing pending.
fn judge_ok() -> sm_core::protocol::JudgeStatus {
    sm_core::protocol::JudgeStatus {
        enabled: true,
        can_run: true,
        blocked: None,
        retry_at: None,
        model: "haiku".into(),
        command: Some("claude".into()),
        pending: 0,
        queued: 0,
        rejected: 0,
        failed: 0,
        calls_today: 0,
        max_calls_per_day: 200,
        judged_today: 0,
        cost_today_usd: 0.0,
        max_usd_per_day: Some(3.0),
        last_error: None,
        quality: None,
        task: sm_core::config::JudgeTask::Verdicts,
        statements_today: 0,
        repeats_today: 0,
        unchecked_today: 0,
    }
}

/// The `status` reply of a ready mock daemon with an index that matches.
fn ready_status(request: &Request) -> StatusReply {
    StatusReply {
        daemon: daemon_info(request, DaemonState::Ready, sm_core::BUILD),
        model: ModelStatus {
            model_id: String::from("fake"),
            revision: String::from("0"),
            dim: 8,
            loaded: true,
        },
        index: IndexStatus {
            matches: true,
            differences: Vec::new(),
            chunks: 0,
            pending_embeddings: 0,
        },
        current_project: None,
        projects: Vec::new(),
        database: String::from("memory.db"),
        database_bytes: 0,
        spool_pending: 0,
        captures_queued: 0,
        judge: judge_ok(),
        metrics: Some(Vec::new()),
        omitted: 0,
        client_build: None,
    }
}

/// B-152 (D19): a daemon still loading when doctor asks is waited for, up
/// to the cold-start budget from its start (SPEC 11, 5 seconds), and not
/// failed. This mock loads for 400 ms after doctor's first `hello`, then is
/// ready; every check passes and `hello` was asked again.
#[test]
fn s11_doctor_waits_for_a_model_still_loading_within_the_cold_start_budget() {
    let _serial = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let first: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    let loading = move || {
        let mut first = first.lock().unwrap();
        let at = *first.get_or_insert_with(Instant::now);
        at.elapsed() < Duration::from_millis(400)
    };
    mock.set_responder(Arc::new(move |request: &Request| {
        let still = loading();
        match &request.call {
            Op::Hello => {
                let state = if still {
                    DaemonState::Loading
                } else {
                    DaemonState::Ready
                };
                Some(Ok(Reply::Hello(daemon_info(
                    request,
                    state,
                    sm_core::BUILD,
                ))))
            }
            _ if still => Some(Err(loading_error())),
            Op::Status(_) => Some(Ok(Reply::Status(Box::new(ready_status(request))))),
            Op::DoctorNumbers => Some(Ok(Reply::DoctorNumbers(DoctorNumbersReply {
                physical_cores: 8,
                query_threads: 4,
                bulk_threads: 6,
                batch_size: 4,
                sample_chunks: 4,
                query_tokens: 128,
                query_runs: 5,
                bulk_elapsed_ms: 100.0,
                bulk_chunks_per_sec: 40.0,
                query_median_ms: 20.0,
            }))),
            _ => None,
        }
    }));
    let spawner = CountingSpawn::default();
    // doctor's own budget, 5 s from the daemon's start.
    let checks = run_checks_in(
        &sb,
        &spawner,
        Duration::from_secs(2),
        &FakeClaude::empty(),
        &sb.tmp,
        None,
    );
    let report = doctor::render(&checks);
    println!("{report}");
    let state = checks
        .iter()
        .find(|c| c.name == "state")
        .unwrap_or_else(|| panic!("no state line in {report}"));
    assert!(state.passed, "{state:?}");
    assert_eq!(doctor::exit_code(&checks), 0, "{report}");
    let hellos = mock
        .requests()
        .iter()
        .filter(|r| matches!(r.call, Op::Hello))
        .count();
    assert!(hellos >= 2, "doctor asked hello again: {hellos}");
    assert_eq!(spawner.count.load(Ordering::SeqCst), 0, "the mock answered");
}
