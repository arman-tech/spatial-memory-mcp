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
use sm_core::config::Settings;
use sm_core::error::{CoreError, Result};
use sm_core::paths::HOME_ENV;
use sm_core::protocol::{
    DaemonState, DoctorNumbersReply, IndexStatus, ModelStatus, Op, Reply, Request, StatusReply,
};
use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
use sm_core::traits::EmbedderInfo;
use sm_daemon::{
    DaemonHandle, DaemonOptions, Judge, JudgeAnswer, JudgeCandidate, JudgeFailure,
    JudgeFailureKind, LoadedModel, ModelLoader, RepeatAnswer, RepeatItem, ShareAdapter, Started,
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
}

/// The production daemon in this process on the sandbox's profile, with
/// the fake model, a judge and 8 physical cores, once it is ready.
fn daemon(sb: &Sandbox) -> DaemonHandle {
    let paths = sb.paths();
    let opts = DaemonOptions {
        lock_wait: Duration::from_millis(300),
        bind_wait: Duration::from_millis(300),
        retry_every: Duration::from_millis(10),
        physical_cores: Some(8),
        shutdown_grace: Duration::from_secs(5),
        judge: Some(Arc::new(IdleJudge)),
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
    let home = sb.home.to_string_lossy().into_owned();
    let path = bin_dir().to_string_lossy().into_owned();
    let get_env = move |k: &str| match k {
        HOME_ENV => Some(home.clone()),
        "PATH" => Some(path.clone()),
        "PATHEXT" => Some(String::from(".EXE")),
        _ => None,
    };
    let version_of = |_: &Path| doctor::plugin_version().map(|v| format!("spatial-memory {v}"));
    let env = DoctorEnv {
        get_env: &get_env,
        cwd: cwd.to_path_buf(),
        home: Some(sb.user_home.clone()),
        spawner,
        runner: claude,
        model: &FakeModelStep,
        version_of: &version_of,
        window: Some(window),
        cold_start,
    };
    doctor::checks(&env, None)
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
        calls_today: 0,
        max_calls_per_day: 200,
        judged_today: 0,
        cost_today_usd: 0.0,
        max_usd_per_day: Some(3.0),
        last_error: None,
        quality: None,
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
