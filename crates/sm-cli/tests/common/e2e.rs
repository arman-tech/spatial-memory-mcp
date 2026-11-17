//! End-to-end support (stream K, part 1): the real daemon with the fake
//! model, in this process or in a child process, and direct requests a test
//! uses to read what the binary's runs left in the store.
//!
//! The fake model can be wrapped so a test sees the queries the daemon
//! embedded (`Model::Recording`) or so one query blocks until the process is
//! killed (`Model::Slow`). A child daemon is this test binary run again on
//! one test, `at3_child_daemon_serves_only_when_asked`, with
//! `SM_E2E_CHILD_HOME` set; without the variable that test returns at once.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sm_core::config::{ProfileName, Settings};
use sm_core::error::{CoreError, Result};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    CaptureParams, DaemonState, ExportParams, HookEvent, Op, RememberItem, RememberParams, Reply,
    Request, StatusParams, StatusReply,
};
use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
use sm_core::time::Timestamp;
use sm_core::traits::{DocInput, Embedder, EmbedderInfo, EmbedderProvider, SessionKind};
use sm_core::types::Client;
use sm_daemon::{DaemonHandle, DaemonOptions, LoadedModel, ModelLoader, ShareAdapter, Started};

use crate::common::Sandbox;

/// The variable that makes the child test run a daemon on this home.
pub const CHILD_HOME_ENV: &str = "SM_E2E_CHILD_HOME";
/// The variable that makes the child's model slow, naming its marker file.
pub const CHILD_MARKER_ENV: &str = "SM_E2E_CHILD_MARKER";
/// The test the child runs.
pub const CHILD_TEST: &str = "at3_child_daemon_serves_only_when_asked";
/// A query holding this word blocks in a slow model.
pub const SLOW_PROBE: &str = "slowprobe";
/// How long a slow query blocks: far past the prompt hook's deadline.
pub const SLOW_FOR: Duration = Duration::from_secs(10);
/// How long a test waits for the daemon to finish something.
pub const WAIT: Duration = Duration::from_secs(20);

// ---------------------------------------------------------------------------
// The fake model, wrapped
// ---------------------------------------------------------------------------

/// What the query session does besides embedding.
#[derive(Clone)]
pub enum Model {
    Plain,
    /// Records every query text.
    Recording(Arc<Mutex<Vec<String>>>),
    /// A query holding [`SLOW_PROBE`] writes the marker, then blocks for
    /// [`SLOW_FOR`].
    Slow(PathBuf),
}

pub struct TestLoader(pub Model);

impl ModelLoader for TestLoader {
    fn info(&self) -> EmbedderInfo {
        fake_info()
    }

    fn load(&self) -> Result<LoadedModel> {
        Ok(LoadedModel {
            provider: Arc::new(TestProvider {
                inner: FakeProvider::new(),
                model: self.0.clone(),
            }),
            chunker: Arc::new(SimpleChunker),
        })
    }
}

struct TestProvider {
    inner: FakeProvider,
    model: Model,
}

impl EmbedderProvider for TestProvider {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }

    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        let inner = self.inner.open(kind, threads)?;
        if kind == SessionKind::Query && !matches!(self.model, Model::Plain) {
            return Ok(Box::new(TestEmbedder {
                inner,
                model: self.model.clone(),
            }));
        }
        Ok(inner)
    }
}

struct TestEmbedder {
    inner: Box<dyn Embedder>,
    model: Model,
}

impl Embedder for TestEmbedder {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }

    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        match &self.model {
            Model::Plain => {}
            Model::Recording(seen) => seen.lock().unwrap().push(text.to_owned()),
            Model::Slow(marker) => {
                if text.contains(SLOW_PROBE) {
                    let _ = std::fs::write(marker, b"embedding");
                    std::thread::sleep(SLOW_FOR);
                }
            }
        }
        self.inner.embed_query(text, max_tokens)
    }

    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(docs)
    }
}

// ---------------------------------------------------------------------------
// The daemon in this process
// ---------------------------------------------------------------------------

/// The test daemon's options: short waits, and the spool drained every
/// second instead of every minute (a test setting, not a default).
pub fn options(endpoint: Option<Endpoint>) -> DaemonOptions {
    DaemonOptions {
        endpoint,
        lock_wait: Duration::from_millis(300),
        bind_wait: Duration::from_millis(300),
        retry_every: Duration::from_millis(10),
        physical_cores: Some(8),
        shutdown_grace: Duration::from_secs(5),
        spool_interval: Duration::from_secs(1),
        ..DaemonOptions::default()
    }
}

/// Waits until a daemon on `endpoint` says ready.
pub fn wait_ready(paths: &ProfilePaths, endpoint: &Endpoint, timeout: Duration) -> bool {
    let hello = Request::new(String::new(), paths.profile.clone(), Op::Hello);
    let until = Instant::now() + timeout;
    loop {
        let state = sm_cli::client::exchange(endpoint, &hello, Some(Duration::from_secs(5)))
            .ok()
            .and_then(|r| r.into_result().ok());
        if let Some(Reply::Hello(info)) = state
            && info.state == DaemonState::Ready
        {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The real daemon in this process, with the production sharing adapter,
/// once its model is ready. `endpoint` `None` is the profile's own.
pub fn start_daemon(
    paths: &ProfilePaths,
    model: Model,
    endpoint: Option<Endpoint>,
) -> DaemonHandle {
    let started = sm_daemon::start(
        paths.clone(),
        Settings::default(),
        Arc::new(TestLoader(model)),
        Arc::new(ShareAdapter),
        options(endpoint),
    )
    .expect("the daemon starts");
    let Started::Running(handle) = started else {
        panic!("another daemon holds the sandbox's lock");
    };
    assert!(
        wait_ready(paths, handle.endpoint(), WAIT),
        "the fake model loaded"
    );
    handle
}

// ---------------------------------------------------------------------------
// Direct requests
// ---------------------------------------------------------------------------

/// Requests sent straight to a daemon, for what a test sets up or reads.
pub struct Direct {
    pub paths: ProfilePaths,
    pub endpoint: Endpoint,
    /// Where export files go.
    pub scratch: PathBuf,
    /// The folder barriers are made in.
    pub projects: PathBuf,
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// A name no other call in this process gives.
pub fn unique(prefix: &str) -> String {
    format!(
        "{prefix}{}x{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

impl Direct {
    pub fn new(sb: &Sandbox, endpoint: Endpoint) -> Direct {
        let scratch = sb.tmp.join("direct");
        std::fs::create_dir_all(&scratch).unwrap();
        Direct {
            paths: sb.paths(),
            endpoint,
            scratch,
            projects: sb.tmp.clone(),
        }
    }

    /// One request; `cwd` is the client's folder.
    pub fn try_call(
        &self,
        cwd: &Path,
        session: Option<&str>,
        op: Op,
    ) -> std::result::Result<Reply, CoreError> {
        let request = Request::new(
            cwd.to_string_lossy().into_owned(),
            self.paths.profile.clone(),
            op,
        )
        .with_session(session.map(str::to_owned))
        .with_client(Client::ClaudeCode);
        match sm_cli::client::exchange(&self.endpoint, &request, Some(Duration::from_secs(30))) {
            Ok(response) => response.into_result(),
            Err(e) => panic!("the daemon answers: {e:?}"),
        }
    }

    pub fn call(&self, cwd: &Path, op: Op) -> Reply {
        let name = op.name();
        self.try_call(cwd, None, op)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
    }

    pub fn status(&self) -> StatusReply {
        let op = Op::Status(StatusParams {
            metrics: false,
            limit: Some(100_000),
        });
        match self.call(&self.scratch, op) {
            Reply::Status(s) => *s,
            other => panic!("status: {other:?}"),
        }
    }

    /// Memories in the project with this name; 0 when it does not exist.
    pub fn project_memories(&self, name: &str) -> u64 {
        self.status()
            .projects
            .iter()
            .filter(|p| p.name == name)
            .map(|p| p.memories)
            .sum()
    }

    /// Waits until nothing waits for embedding.
    pub fn wait_embedded(&self) {
        assert!(
            wait_until(WAIT, || self.status().index.pending_embeddings == 0),
            "every chunk was embedded"
        );
    }

    pub fn remember_item(&self, cwd: &Path, item: RememberItem) -> String {
        let op = Op::Remember(RememberParams {
            memories: vec![item],
        });
        match self.call(cwd, op) {
            Reply::Remember(r) => r.memories[0].id.to_string(),
            other => panic!("remember: {other:?}"),
        }
    }

    pub fn remember(&self, cwd: &Path, content: &str) -> String {
        self.remember_item(cwd, RememberItem::new(content))
    }

    /// Hands a capture to the daemon, as a hook would.
    pub fn capture(&self, cwd: &Path, session: Option<&str>, event: HookEvent, payload: Value) {
        let op = Op::Capture(CaptureParams {
            event,
            payload,
            received_at: Timestamp::now(),
        });
        match self.try_call(cwd, session, op) {
            Ok(Reply::Captured(c)) => assert!(c.accepted, "the capture was accepted"),
            other => panic!("capture: {other:?}"),
        }
    }

    /// Every stored memory, as the export file writes it, by project name.
    pub fn export(&self) -> BTreeMap<String, Vec<Value>> {
        let path = self.scratch.join(format!("{}.jsonl", unique("export")));
        let op = Op::Export(ExportParams {
            path: path.to_string_lossy().into_owned(),
            project: None,
        });
        match self.call(&self.scratch, op) {
            Reply::Export(_) => {}
            other => panic!("export: {other:?}"),
        }
        let text = std::fs::read_to_string(&path).expect("the export file");
        let mut names = BTreeMap::new();
        let mut memories: Vec<Value> = Vec::new();
        for line in text.lines() {
            let v: Value = serde_json::from_str(line).expect("an export line");
            if let Some(p) = v.get("project") {
                names.insert(
                    p["identity"].as_str().unwrap_or_default().to_owned(),
                    p["name"].as_str().unwrap_or_default().to_owned(),
                );
            } else if let Some(m) = v.get("memory") {
                memories.push(m.clone());
            }
        }
        let mut by_name: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for m in memories {
            let key = m["project"].as_str().unwrap_or_default();
            let name = names.get(key).cloned().unwrap_or_else(|| key.to_owned());
            by_name.entry(name).or_default().push(m);
        }
        by_name
    }

    /// Waits until every capture handed over before it has been processed
    /// (one capture worker, first in first out).
    pub fn barrier(&self) {
        let name = unique("bar");
        let cwd = self.projects.join(&name);
        std::fs::create_dir_all(&cwd).unwrap();
        let text = format!("Decided to use {} for the barrier.", unique("sentinel"));
        self.capture(
            &cwd,
            None,
            HookEvent::PostToolUse,
            json!({"tool_name": "Edit", "tool_input": {"new_string": text}, "tool_response": "ok"}),
        );
        assert!(
            wait_until(WAIT, || self.project_memories(&name) > 0),
            "the barrier capture was stored"
        );
    }
}

/// Polls `f` every 20 ms until it holds or `timeout` passes.
pub fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let until = Instant::now() + timeout;
    loop {
        if f() {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Every journal entry for `metric` in the profile, in no set order.
pub fn journal(paths: &ProfilePaths, metric: &str) -> Vec<sm_core::journal::Entry> {
    let Ok(entries) = std::fs::read_dir(paths.metrics_dir()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.filter_map(std::result::Result::ok) {
        let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
        for line in text.lines() {
            let e: sm_core::journal::Entry = serde_json::from_str(line).expect("journal line");
            if e.metric == metric {
                out.push(e);
            }
        }
    }
    out
}

/// The `content` of each exported memory, sorted.
pub fn contents(memories: &[Value]) -> Vec<String> {
    let mut out: Vec<String> = memories
        .iter()
        .map(|m| m["content"].as_str().unwrap_or_default().to_owned())
        .collect();
    out.sort();
    out
}

/// Version 2's redaction markers (`[REDACTED_AWS_KEY]`) as version 3 writes
/// them (`[REDACTED:aws_key]`).
pub fn v3_markers(s: &str) -> String {
    const OPEN: &str = "[REDACTED_";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(OPEN) {
        out.push_str(&rest[..i]);
        let after = &rest[i + OPEN.len()..];
        let kind_len = after
            .find(|c: char| !(c.is_ascii_uppercase() || c == '_'))
            .unwrap_or(after.len());
        if kind_len > 0 && after[kind_len..].starts_with(']') {
            out.push_str(&format!(
                "[REDACTED:{}]",
                after[..kind_len].to_ascii_lowercase()
            ));
            rest = &after[kind_len + 1..];
        } else {
            out.push_str(OPEN);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// The daemon in a child process
// ---------------------------------------------------------------------------

/// A daemon in a child process on a sandbox's profile and endpoint. Killed
/// and reaped when dropped; declare it after the sandbox so it goes first.
pub struct ChildDaemon {
    child: Option<Child>,
    pub paths: ProfilePaths,
    log: PathBuf,
}

impl ChildDaemon {
    /// Starts the child and waits until it says ready. `slow_marker` makes
    /// its model slow.
    pub fn spawn(sb: &Sandbox, slow_marker: Option<&Path>) -> ChildDaemon {
        let paths = sb.paths();
        let log = sb.tmp.join(format!("{}.log", unique("child")));
        let mut cmd = Command::new(std::env::current_exe().expect("the test binary"));
        cmd.args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD_HOME_ENV, &sb.home)
            .env_remove(CHILD_MARKER_ENV)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).expect("the child's log"));
        if let Some(marker) = slow_marker {
            cmd.env(CHILD_MARKER_ENV, marker);
        }
        let child = cmd.spawn().expect("spawn the child daemon");
        let mut me = ChildDaemon {
            child: Some(child),
            paths,
            log,
        };
        let endpoint = me.paths.endpoint();
        let until = Instant::now() + WAIT;
        loop {
            if wait_ready(&me.paths, &endpoint, Duration::from_millis(200)) {
                return me;
            }
            let exited = me.child.as_mut().and_then(|c| c.try_wait().ok().flatten());
            if exited.is_some() || Instant::now() >= until {
                let text = std::fs::read_to_string(&me.log).unwrap_or_default();
                me.kill();
                panic!("the child daemon did not become ready ({exited:?}): {text}");
            }
        }
    }

    /// Kills the child and waits for it; then waits for its lock.
    pub fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            crate::common::wait_for_lock_free(&self.paths, Duration::from_secs(10));
        }
    }
}

impl Drop for ChildDaemon {
    fn drop(&mut self) {
        self.kill();
    }
}

/// The child's body: a daemon on `home` until it is shut down or killed.
/// It ends itself after two minutes whatever happens, so a test that dies
/// cannot leave it running.
pub fn child_main(home: &str) {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(120));
        std::process::exit(3);
    });
    let model = match std::env::var_os(CHILD_MARKER_ENV) {
        Some(m) => Model::Slow(PathBuf::from(m)),
        None => Model::Plain,
    };
    let paths = ProfilePaths::new(PathBuf::from(home), ProfileName::default_profile());
    let started = sm_daemon::start(
        paths,
        Settings::default(),
        Arc::new(TestLoader(model)),
        Arc::new(ShareAdapter),
        options(None),
    )
    .expect("the child daemon starts");
    let Started::Running(handle) = started else {
        std::process::exit(2);
    };
    handle.wait();
    std::process::exit(0);
}
