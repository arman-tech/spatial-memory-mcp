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
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::{BIN, CountingSpawn, MockDaemon, Mode, Sandbox, daemon_info, loading_error};
use sm_cli::doctor::{self, Check, DoctorEnv, NUMBERS_LATER};
use sm_cli::model_step::ModelStep;
use sm_cli::setup::{ClaudeRunner, RunOutput};
use sm_core::config::Settings;
use sm_core::error::{CoreError, Result};
use sm_core::paths::HOME_ENV;
use sm_core::protocol::{DaemonState, IndexStatus, ModelStatus, Op, Reply, Request, StatusReply};
use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
use sm_core::traits::EmbedderInfo;
use sm_daemon::{DaemonHandle, DaemonOptions, LoadedModel, ModelLoader, ShareAdapter, Started};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A `claude` that answers the one read-only question `doctor` asks.
struct FakeClaude;

impl ClaudeRunner for FakeClaude {
    fn run(&self, args: &[String]) -> io::Result<RunOutput> {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        assert_eq!(a, ["plugin", "list", "--json"], "doctor only reads");
        Ok(RunOutput {
            success: true,
            stdout: String::from("[]"),
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

/// The production daemon in this process on the sandbox's profile, with
/// the fake model and 8 physical cores, once it is ready.
fn daemon(sb: &Sandbox) -> DaemonHandle {
    let paths = sb.paths();
    let opts = DaemonOptions {
        lock_wait: Duration::from_millis(300),
        bind_wait: Duration::from_millis(300),
        retry_every: Duration::from_millis(10),
        physical_cores: Some(8),
        shutdown_grace: Duration::from_secs(5),
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
        cwd: sb.tmp.clone(),
        home: Some(sb.user_home.clone()),
        spawner,
        runner: &FakeClaude,
        model: &FakeModelStep,
        version_of: &version_of,
        window: Some(window),
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
/// the other checks give (here 1, from the `state` check's "still
/// loading").
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
