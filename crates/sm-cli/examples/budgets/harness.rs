//! The measurement harness: the release binary in a private temporary home,
//! its real daemon, direct requests, hook and command runs, figures.
//!
//! Every process runs with a cleared environment, as the integration tests'
//! sandbox does: the user folders point inside the temporary home, the model
//! folder is the per-user cache (read only), and the model address is one
//! nothing answers, so nothing is ever downloaded.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sm_core::config::{AMBIENT_ENV, ProfileName};
use sm_core::journal::Entry;
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{DaemonInfo, DaemonState, Op, Reply, Request, StatusReply};
use sm_core::types::Client;

use crate::procs::{self, Reading};

/// What a process needs to start on each platform, nothing that names user
/// data (the integration tests' list).
const PASS_THROUGH: &[&str] = &[
    "PATH",
    "SystemRoot",
    "SYSTEMROOT",
    "windir",
    "ComSpec",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
];

/// A model address nothing answers: the model is never downloaded.
const NO_MODEL_URL: &str = "http://127.0.0.1:9/none";

/// The longest any wait in a case may take before the case gives up.
pub const LONG: Duration = Duration::from_secs(30 * 60);

/// Where the binary, the model and the fixtures are.
pub struct Env {
    pub bin: PathBuf,
    pub model_dir: PathBuf,
    pub repo: PathBuf,
}

impl Env {
    /// `Err` holds the one line printed when the program cannot run.
    pub fn find() -> Result<Env, String> {
        let exe = std::env::current_exe().map_err(|e| format!("budgets: no own path: {e}"))?;
        // target/<profile>/examples/budgets(.exe) -> target/<profile>
        let bin = exe
            .parent()
            .and_then(Path::parent)
            .map(|d| d.join(format!("spatial-memory{}", std::env::consts::EXE_SUFFIX)))
            .ok_or_else(|| "budgets: cannot find the target folder".to_owned())?;
        if !bin.is_file() {
            return Err(format!(
                "budgets: {} is missing; build it first with `cargo build --release -p sm-cli --bins --examples`",
                bin.display()
            ));
        }
        let model_dir = sm_embed::model::default_model_dir()
            .map_err(|e| format!("budgets: skipped, no model folder: {}", e.message))?;
        let missing: Vec<&str> = sm_embed::MODEL_FILES
            .iter()
            .filter(|f| !f.local_path(&model_dir).is_file())
            .map(|f| f.path)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "budgets: skipped, the model is not in {} (missing {})",
                model_dir.display(),
                missing.join(", ")
            ));
        }
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("the repository root")
            .to_path_buf();
        Ok(Env {
            bin,
            model_dir,
            repo,
        })
    }

    pub fn fixture(&self, rel: &str) -> PathBuf {
        self.repo.join("fixtures").join(rel)
    }
}

/// A process that has finished.
#[derive(Debug)]
pub struct Run {
    pub stdout: String,
    pub stderr: String,
    pub code: Option<i32>,
    pub elapsed: Duration,
}

/// A process still running, its stdin being written.
pub struct Running {
    pub child: Child,
    pub started: Instant,
    writer: Option<JoinHandle<()>>,
}

impl Running {
    pub fn finish(mut self) -> Run {
        let out = std::mem::replace(&mut self.child, dummy_child())
            .wait_with_output()
            .expect("wait for spatial-memory");
        let elapsed = self.started.elapsed();
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        Run {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            code: out.status.code(),
            elapsed,
        }
    }
}

/// A finished child to swap in when `Running` gives its own up.
fn dummy_child() -> Child {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", "exit 0"]);
        c
    } else {
        Command::new("true")
    };
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn a trivial process");
    let _ = child.wait();
    child
}

/// A fresh `SPATIAL_MEMORY_HOME` with its own user folders, removed at drop
/// after any daemon on its endpoint has been shut down.
pub struct Home {
    pub root: PathBuf,
    pub home: PathBuf,
    pub user: PathBuf,
    pub tmp: PathBuf,
    pub paths: ProfilePaths,
    pub endpoint: Endpoint,
    pub model_dir: PathBuf,
    bin: PathBuf,
}

impl Home {
    pub fn new(env: &Env, case: &str) -> Home {
        let root = tempfile::Builder::new()
            .prefix(&format!("sm-budgets-{case}-"))
            .tempdir()
            .expect("a temporary folder")
            .keep();
        let home = root.join("h");
        let user = root.join("u");
        let tmp = root.join("t");
        for d in [&home, &user, &tmp] {
            std::fs::create_dir_all(d).expect("a folder in the temporary home");
        }
        let paths = ProfilePaths::new(home.clone(), ProfileName::default_profile());
        let endpoint = paths.endpoint();
        Home {
            root,
            home,
            user,
            tmp,
            paths,
            endpoint,
            model_dir: env.model_dir.clone(),
            bin: env.bin.clone(),
        }
    }

    /// A folder under the home's `t`, created; a project of its own.
    pub fn folder(&self, name: &str) -> PathBuf {
        let d = self.tmp.join(name);
        std::fs::create_dir_all(&d).expect("a project folder");
        d
    }

    /// The binary with a cleared, private environment, in `cwd`.
    pub fn command(&self, args: &[&str], cwd: &Path) -> Command {
        let mut cmd = Command::new(&self.bin);
        cmd.args(args).env_clear();
        for key in PASS_THROUGH {
            if let Some(v) = std::env::var_os(key) {
                cmd.env(key, v);
            }
        }
        // The binary under test comes first on PATH, so `doctor`'s path check
        // finds it and not another `spatial-memory` (version 1, say).
        if let Some(dir) = self.bin.parent() {
            let rest = std::env::var_os("PATH").unwrap_or_default();
            let dirs = std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&rest));
            if let Ok(path) = std::env::join_paths(dirs) {
                cmd.env("PATH", path);
            }
        }
        let u = &self.user;
        cmd.env("HOME", u)
            .env("USERPROFILE", u)
            .env("APPDATA", u.join("AppData").join("Roaming"))
            .env("LOCALAPPDATA", u.join("AppData").join("Local"))
            .env("XDG_CONFIG_HOME", u.join(".config"))
            .env("XDG_DATA_HOME", u.join(".local").join("share"))
            .env("XDG_CACHE_HOME", u.join(".cache"))
            .env(sm_core::paths::MODEL_DIR_ENV, &self.model_dir)
            .env(sm_cli::model_step::MODEL_URL_ENV, NO_MODEL_URL)
            .env(sm_core::paths::HOME_ENV, &self.home)
            .current_dir(cwd);
        cmd
    }

    /// Starts the binary with piped streams and writes `stdin` to it.
    pub fn spawn(&self, args: &[&str], cwd: &Path, env: &[(&str, &str)], stdin: &[u8]) -> Running {
        let mut cmd = self.command(args, cwd);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let input = stdin.to_vec();
        let started = Instant::now();
        let mut child = cmd.spawn().expect("spawn spatial-memory");
        let mut pipe = child.stdin.take().expect("stdin pipe");
        let writer = std::thread::spawn(move || {
            let _ = pipe.write_all(&input);
        });
        Running {
            child,
            started,
            writer: Some(writer),
        }
    }

    pub fn run(&self, args: &[&str], cwd: &Path) -> Run {
        self.spawn(args, cwd, &[], b"").finish()
    }

    /// The prompt hook's input for `prompt` in `cwd`, a session of its own.
    pub fn prompt_input(cwd: &Path, prompt: &str, session: &str) -> Vec<u8> {
        let cwd = cwd.to_string_lossy().replace('\\', "/");
        serde_json::to_vec(&json!({"session_id": session, "cwd": cwd, "prompt": prompt}))
            .expect("hook input")
    }

    /// `hook user-prompt-submit --client claude-code` with ambient recall on.
    pub fn spawn_prompt_hook(&self, cwd: &Path, prompt: &str, session: &str) -> Running {
        self.spawn(
            &["hook", "user-prompt-submit", "--client", "claude-code"],
            cwd,
            &[(AMBIENT_ENV, "on")],
            &Self::prompt_input(cwd, prompt, session),
        )
    }

    pub fn prompt_hook(&self, cwd: &Path, prompt: &str, session: &str) -> Run {
        self.spawn_prompt_hook(cwd, prompt, session).finish()
    }

    /// `spatial-memory daemon` in the foreground, its streams closed.
    pub fn spawn_daemon(&self) -> (Child, Instant) {
        let mut cmd = self.command(&["daemon"], &self.tmp);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let started = Instant::now();
        (cmd.spawn().expect("spawn the daemon"), started)
    }

    /// The daemon's `hello`, or `None` when nothing answers.
    pub fn hello(&self) -> Option<DaemonInfo> {
        let req = Request::new(String::new(), self.paths.profile.clone(), Op::Hello);
        match sm_cli::client::exchange(&self.endpoint, &req, Some(Duration::from_secs(5)))
            .ok()?
            .into_result()
            .ok()?
        {
            Reply::Hello(info) => Some(info),
            _ => None,
        }
    }

    /// Starts the daemon and waits until it is ready (or has failed, or
    /// exited). Polls `hello` every 10 ms.
    pub fn start(&self) -> Result<Daemon, String> {
        let (mut child, started) = self.spawn_daemon();
        let pid = child.id();
        let mut first_answer = None;
        let until = Instant::now() + Duration::from_secs(300);
        loop {
            if let Some(info) = self.hello() {
                first_answer.get_or_insert_with(|| started.elapsed());
                if info.pid != pid {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("another daemon (pid {}) answered", info.pid));
                }
                match info.state {
                    DaemonState::Ready | DaemonState::Failed => {
                        let ready = started.elapsed();
                        return Ok(Daemon {
                            child,
                            pid,
                            first_answer: first_answer.unwrap_or(ready),
                            ready,
                            info,
                        });
                    }
                    _ => {}
                }
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(format!(
                    "the daemon exited with {status} before it was ready"
                ));
            }
            if Instant::now() >= until {
                let _ = child.kill();
                let _ = child.wait();
                return Err("the daemon was not ready within 300 s".to_owned());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// [`Home::start`], which must reach `ready`.
    pub fn start_ready(&self) -> Daemon {
        let d = self.start().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            d.info.state,
            DaemonState::Ready,
            "the daemon is ready: {:?}",
            d.info.state_error
        );
        d
    }

    pub fn request(&self, cwd: &Path, session: Option<&str>, op: Op) -> Request {
        Request::new(
            cwd.to_string_lossy().into_owned(),
            self.paths.profile.clone(),
            op,
        )
        .with_session(session.map(str::to_owned))
        .with_client(Client::ClaudeCode)
    }

    /// One request; any failure is an `Err` with its text.
    pub fn try_call(&self, cwd: &Path, session: Option<&str>, op: Op) -> Result<Reply, String> {
        let req = self.request(cwd, session, op);
        match sm_cli::client::exchange(&self.endpoint, &req, Some(Duration::from_secs(600))) {
            Ok(r) => r
                .into_result()
                .map_err(|e| format!("{:?}: {}", e.code, e.message)),
            Err(e) => Err(format!("{e:?}")),
        }
    }

    pub fn call(&self, cwd: &Path, op: Op) -> Reply {
        let name = op.name();
        self.try_call(cwd, None, op)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    pub fn status(&self) -> StatusReply {
        match self.call(&self.tmp, op(json!({"op": "status", "args": {}}))) {
            Reply::Status(s) => *s,
            other => panic!("status: {other:?}"),
        }
    }

    /// Waits until nothing waits for embedding; returns how long it took.
    pub fn wait_embedded(&self) -> Duration {
        let started = Instant::now();
        loop {
            if self.status().index.pending_embeddings == 0 {
                return started.elapsed();
            }
            assert!(started.elapsed() < LONG, "every chunk embedded in time");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Remembers `texts` in `cwd`, 100 a request; returns the ids.
    pub fn remember_all(&self, cwd: &Path, texts: &[String]) -> Vec<String> {
        let mut ids = Vec::with_capacity(texts.len());
        for batch in texts.chunks(sm_core::protocol::MAX_REMEMBER_BATCH) {
            let memories: Vec<Value> = batch.iter().map(|t| json!({"content": t})).collect();
            let reply = self.call(
                cwd,
                op(json!({"op": "remember", "args": {"memories": memories}})),
            );
            let Reply::Remember(r) = reply else {
                panic!("remember: {reply:?}")
            };
            ids.extend(r.memories.iter().map(|m| m.id.to_string()));
        }
        ids
    }

    /// Every journal entry for `metric`.
    pub fn journal(&self, metric: &str) -> Vec<Entry> {
        let Ok(entries) = std::fs::read_dir(self.paths.metrics_dir()) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in entries.filter_map(Result::ok) {
            let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
            for line in text.lines() {
                if let Ok(e) = serde_json::from_str::<Entry>(line)
                    && e.metric == metric
                {
                    out.push(e);
                }
            }
        }
        out
    }

    /// The `gate` labels of the hook journal, counted.
    pub fn gates(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for e in self.journal(sm_core::journal::metrics::HOOK_GATE_TOTAL) {
            let gate = e.labels.get("gate").and_then(Value::as_str).unwrap_or("?");
            *out.entry(gate.to_owned()).or_insert(0) += 1;
        }
        out
    }

    /// Shuts down whatever daemon answers on this home's endpoint and waits
    /// for the endpoint to close.
    pub fn shut_down_any(&self) {
        if !sm_cli::client::is_listening(&self.endpoint) {
            return;
        }
        let req = self.request(
            &self.tmp,
            None,
            op(json!({"op": "shutdown", "args": {"reason": "budgets"}})),
        );
        let _ = sm_cli::client::exchange(&self.endpoint, &req, Some(Duration::from_secs(10)));
        let until = Instant::now() + Duration::from_secs(30);
        while sm_cli::client::is_listening(&self.endpoint) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Removes the temporary home; returns whether it is gone.
    pub fn remove(&self) -> bool {
        for _ in 0..50 {
            match std::fs::remove_dir_all(&self.root) {
                Ok(()) => return true,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return true,
                Err(_) => std::thread::sleep(Duration::from_millis(200)),
            }
        }
        !self.root.exists()
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        self.shut_down_any();
        if !self.remove() {
            println!("  note: could not remove {}", self.root.display());
        }
    }
}

/// The daemon this program started.
pub struct Daemon {
    pub child: Child,
    pub pid: u32,
    /// From spawning to the first `hello` answer.
    pub first_answer: Duration,
    /// From spawning to the first `hello` that said ready (or failed).
    pub ready: Duration,
    pub info: DaemonInfo,
}

impl Daemon {
    pub fn read(&self) -> Reading {
        procs::read(self.pid).expect("the daemon's process can be read")
    }

    /// Sends `shutdown` and waits for the process to exit (kills it after
    /// 30 s). Returns the time to exit.
    pub fn stop(&mut self, home: &Home) -> Duration {
        let started = Instant::now();
        let req = home.request(
            &home.tmp,
            None,
            op(json!({"op": "shutdown", "args": {"reason": "budgets"}})),
        );
        let _ = sm_cli::client::exchange(&home.endpoint, &req, Some(Duration::from_secs(10)));
        let until = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(Some(_)) = self.child.try_wait() {
                return started.elapsed();
            }
            if Instant::now() >= until {
                println!("  note: the daemon did not exit within 30 s of shutdown; killed");
                self.kill();
                return started.elapsed();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            self.kill();
        }
    }
}

/// An op from its JSON form, `{"op": ..., "args": ...}`.
pub fn op(v: Value) -> Op {
    serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("op {v}: {e}"))
}

/// A name no other call in this process gives.
pub fn unique(prefix: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{prefix}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

/// Milliseconds of a duration.
pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The 50th and 95th percentiles (nearest rank) and the maximum, in ms.
pub fn stats(times: &[Duration]) -> (f64, f64, f64) {
    let mut v: Vec<f64> = times.iter().map(|d| ms(*d)).collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let rank = |p: f64| ((p * n as f64).ceil() as usize).clamp(1, n) - 1;
    (v[rank(0.5)], v[rank(0.95)], v[n - 1])
}

/// One timing figure: the case, the count, p50, p95, max and the budget.
/// Time is printed, never asserted (B-58).
pub fn figure(case: &str, what: &str, times: &[Duration], budget: &str) {
    let (p50, p95, max) = stats(times);
    println!(
        "[{case}] {what}: n {} | p50 {p50:.1} ms | p95 {p95:.1} ms | max {max:.1} ms | budget {budget}",
        times.len()
    );
}

/// One memory figure, judged on the working set (B-114): pass or fail.
pub fn memory(case: &str, what: &str, working_set: u64, private: u64, budget: u64) -> bool {
    let pass = working_set <= budget;
    println!(
        "[{case}] {what}: working set {} | private {} | budget {} on the working set | {}",
        procs::mib(working_set),
        procs::mib(private),
        procs::mib(budget),
        if pass { "pass" } else { "FAIL" }
    );
    pass
}

/// A line of the case's notes.
pub fn note(case: &str, text: impl AsRef<str>) {
    println!("[{case}] {}", text.as_ref());
}
