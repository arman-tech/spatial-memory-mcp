//! Sharing and migration end to end through the binary (SPEC 9 and 10,
//! decision B-79): `share` and `sync` through `spatial-memory mcp`, and
//! `migrate`, `export` and `import` as subcommands. The daemon runs in this
//! process on the sandbox's endpoint, with the fake embedder and the
//! production sharing adapter, so every reply checked here was made by
//! `sm-share`: a shared file in the repository, an export file, counts read
//! from a file. The git commands these tests run use an empty global
//! configuration and no system one.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::{Run, Sandbox};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Value, json};
use sm_core::config::Settings;
use sm_core::error::Result;
use sm_core::protocol::{DaemonState, Op, Reply, Request};
use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
use sm_core::traits::EmbedderInfo;
use sm_daemon::{DaemonHandle, DaemonOptions, LoadedModel, ModelLoader, ShareAdapter, Started};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The fake model, loaded at once.
struct FakeModel;

impl ModelLoader for FakeModel {
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

/// A daemon in this process on the sandbox's profile and endpoint, with the
/// production adapter, once its model is ready. Dropping it shuts it down.
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
        Arc::new(FakeModel),
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

/// Runs the binary in `dir`.
fn run_in(sb: &Sandbox, dir: &Path, args: &[&str]) -> Run {
    let start = Instant::now();
    let out = sb
        .command(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("run spatial-memory");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code(),
        elapsed: start.elapsed(),
    }
}

/// Runs git in `dir`, isolated from the user's and the system's settings.
fn git(dir: &Path, config: &Path, args: &[&str]) {
    let mut cmd = Command::new("git");
    cmd.args(args).current_dir(dir);
    for var in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
    ] {
        cmd.env_remove(var);
    }
    let out = cmd
        .env("GIT_CONFIG_GLOBAL", config)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_AUTHOR_NAME", "Test Author")
        .env("GIT_AUTHOR_EMAIL", "author@example.invalid")
        .env("GIT_COMMITTER_NAME", "Test Author")
        .env("GIT_COMMITTER_EMAIL", "author@example.invalid")
        .output()
        .expect("run git");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A git repository with no commits, under the sandbox's `{TMP}`.
fn repo(sb: &Sandbox, name: &str) -> (PathBuf, PathBuf) {
    let config = sb.tmp.join("gitconfig");
    std::fs::write(&config, "").unwrap();
    let dir = sb.tmp.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    git(&dir, &config, &["init", "--quiet", "--initial-branch=main"]);
    (dir, config)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// `spatial-memory mcp` in `dir`, with an rmcp client attached.
async fn mcp_in(sb: &Sandbox, dir: &Path) -> RunningService<RoleClient, ()> {
    let mut cmd = sb.command(&["mcp"]);
    cmd.current_dir(dir).stderr(Stdio::null());
    let child = TokioChildProcess::new(tokio::process::Command::from(cmd)).expect("spawn mcp");
    ().serve(child).await.expect("the MCP session starts")
}

/// The structured result of a tool call that succeeded.
async fn call_ok(client: &RunningService<RoleClient, ()>, tool: &str, args: Value) -> Value {
    let Value::Object(args) = args else {
        panic!("arguments are an object")
    };
    let result: CallToolResult = client
        .call_tool(CallToolRequestParams::new(tool.to_owned()).with_arguments(args))
        .await
        .expect("tools/call answers");
    assert_ne!(result.is_error, Some(true), "{tool}: {result:?}");
    result
        .structured_content
        .unwrap_or_else(|| panic!("{tool} has structured content"))
}

#[test]
fn s9_share_and_sync_through_the_mcp_binary_reach_the_adapter() {
    let _g = serial();
    let author = Sandbox::with_daemon();
    let (repo, config) = repo(&author, "billing");
    let content = "Invoices are numbered per tenant, never globally.";

    let id = {
        let _daemon = daemon(&author);
        runtime().block_on(async {
            let client = mcp_in(&author, &repo).await;
            let remembered = call_ok(
                &client,
                "remember",
                json!({"memories": [{"content": content, "type": "decision"}]}),
            )
            .await;
            let id = remembered["memories"][0]["id"]
                .as_str()
                .expect("an id")
                .to_owned();
            let shared = call_ok(&client, "share", json!({"ids": [id]})).await;
            assert_eq!(
                shared["written"].as_array().map(Vec::len),
                Some(1),
                "{shared}"
            );
            assert_eq!(shared["written"][0]["id"], id.as_str());
            let rel = shared["written"][0]["path"].as_str().expect("a path");
            assert!(rel.starts_with(".spatial-memory/shared/"), "{rel}");
            let text = std::fs::read_to_string(repo.join(rel)).expect("the shared file");
            assert!(text.contains(content), "{text}");
            assert!(text.contains(&id), "the file names the memory: {text}");
            client.cancel().await.unwrap();
            id
        })
    };
    git(&repo, &config, &["add", "-A"]);
    git(
        &repo,
        &config,
        &["commit", "--quiet", "-m", "share a memory"],
    );

    // A teammate's store: another profile home, the same repository.
    let reader = Sandbox::with_daemon();
    let _daemon = daemon(&reader);
    runtime().block_on(async {
        let client = mcp_in(&reader, &repo).await;
        let dry = call_ok(&client, "sync", json!({"dry_run": true})).await;
        assert_eq!(dry["dry_run"], true, "{dry}");
        assert_eq!(dry["added"], json!([id]), "{dry}");
        let synced = call_ok(&client, "sync", json!({})).await;
        assert_eq!(synced["dry_run"], false, "{synced}");
        assert_eq!(synced["added"], json!([id]), "{synced}");
        assert!(synced.get("errors").is_none(), "{synced}");
        let again = call_ok(&client, "sync", json!({})).await;
        assert!(again.get("added").is_none(), "{again}");
        assert_eq!(again["unchanged"], 1, "{again}");
        client.cancel().await.unwrap();
    });
}

#[test]
fn s10_migrate_export_and_import_subcommands_reach_the_adapter() {
    let _g = serial();
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/migration/v1_export_sample.json"
    ));
    let source = Sandbox::with_daemon();
    let file = source.tmp.join("export.json");
    {
        let _daemon = daemon(&source);
        let migrated = run_in(
            &source,
            &source.tmp,
            &["migrate", "--from", fixture.to_str().expect("a UTF-8 path")],
        );
        assert_eq!(migrated.code, Some(0), "{migrated:?}");
        assert!(
            migrated.stdout.starts_with("Imported "),
            "{}",
            migrated.stdout
        );
        for line in ["  records read:        10\n", "  inserted:            10\n"] {
            assert!(
                migrated.stdout.contains(line),
                "{line}: {}",
                migrated.stdout
            );
        }

        // A relative path is resolved against the client's folder.
        let exported = run_in(&source, &source.tmp, &["export", "export.json"]);
        assert_eq!(exported.code, Some(0), "{exported:?}");
        assert!(
            exported.stdout.starts_with("Exported 10 memories and "),
            "{}",
            exported.stdout
        );
        assert!(file.is_file(), "the export file was written");

        // sm-share refuses to overwrite; the refusal is the client's error.
        let twice = run_in(&source, &source.tmp, &["export", "export.json"]);
        assert_eq!(twice.code, Some(1), "{twice:?}");
        assert!(
            twice.stderr.starts_with("error: invalid_input: "),
            "{}",
            twice.stderr
        );
        assert!(twice.stderr.contains("already exists"), "{}", twice.stderr);
        assert!(twice.stderr.contains("\nnext: "), "{}", twice.stderr);
    }

    let target = Sandbox::with_daemon();
    let _daemon = daemon(&target);
    let path = file.to_str().expect("a UTF-8 path");
    let imported = run_in(&target, &target.tmp, &["import", path]);
    assert_eq!(imported.code, Some(0), "{imported:?}");
    assert!(
        imported.stdout.starts_with(&format!("Imported {path}\n")),
        "{}",
        imported.stdout
    );
    for line in ["  records read:        10\n", "  inserted:            10\n"] {
        assert!(
            imported.stdout.contains(line),
            "{line}: {}",
            imported.stdout
        );
    }
    let again = run_in(&target, &target.tmp, &["import", path]);
    assert_eq!(again.code, Some(0), "{again:?}");
    for line in ["  inserted:            0\n", "  already present:     10\n"] {
        assert!(again.stdout.contains(line), "{line}: {}", again.stdout);
    }
}
