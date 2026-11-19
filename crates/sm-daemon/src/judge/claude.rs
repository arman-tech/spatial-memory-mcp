//! The judge as a Claude Code call (SPEC section 16, The judge; B-158,
//! B-168; `docs/v3/SPIKE-JUDGE.md`).
//!
//! One call runs `claude -p` once, with the flags the spike proved: no
//! settings, plugins, hooks, tools, MCP servers or slash commands, no
//! session saved, a prompt and a schema of `docs/v3/spikes/judge/`, and
//! the input as one JSON array on standard input. The judge's call sends
//! the candidates with `prompt.txt` and `schema.json`; the duplicate
//! check's call sends kept memories and their nearest active ones with
//! `repeats-prompt.txt` and `repeats-schema.json` (B-192), with the same
//! flags. The child runs in
//! `%LOCALAPPDATA%\spatial-memory\judge\work` (kept empty), with an
//! environment built from a whitelist plus `SPATIAL_MEMORY_JUDGE=1` and
//! `DISABLE_PROMPT_CACHING=1` (B-182), at a lower priority (below normal on Windows, nice 10 elsewhere), and is
//! ended at its timeout or when the daemon stops.

use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
use sm_core::config::{JUDGE_MARKER_ENV, JudgeSettings};
use sm_core::time::Timestamp;

use super::{
    Judge, JudgeAnswer, JudgeCandidate, JudgeFailure, JudgeFailureKind, JudgeVerdict, RepeatAnswer,
    RepeatItem, RepeatVerdict,
};

/// The judge's instructions, version 10 (`docs/v3/SPIKE-JUDGE.md`, B-173,
/// B-179, B-182, B-191): the judge marks each statement worth keeping by
/// its first and last words.
pub const PROMPT: &str = include_str!("../../../../docs/v3/spikes/judge/prompt.txt");

/// The answer's JSON schema: one verdict per candidate.
pub const SCHEMA: &str = include_str!("../../../../docs/v3/spikes/judge/schema.json");

/// The duplicate check's instructions (B-192): for each kept memory, whether
/// any statement of it is new against the active memories most alike. The
/// version measured on the design set (`CAPTURE-RCA.md` section 8.10).
pub const REPEATS_PROMPT: &str =
    include_str!("../../../../docs/v3/spikes/judge/repeats-prompt.txt");

/// The duplicate check's JSON schema: one answer per kept memory.
pub const REPEATS_SCHEMA: &str =
    include_str!("../../../../docs/v3/spikes/judge/repeats-schema.json");

/// How long one call may run before it is ended. The spike's largest batch
/// (25 candidates) took 11.5 seconds.
pub const CALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Bytes of standard output (and of standard error) read from one call; the
/// rest is read and dropped. A batch's answer is a few kilobytes.
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// Characters of an error message kept from Claude Code's output.
const MAX_MESSAGE_CHARS: usize = 300;

/// Claude Code's documented switch that turns prompt caching off. Each call
/// sends the prompt once and never again within the cache's lifetime, so
/// writing it to the cache only costs more: about 17 per cent per candidate
/// on the design set, with the same verdicts (B-182).
pub const NO_PROMPT_CACHING_ENV: &str = "DISABLE_PROMPT_CACHING";

/// The environment variables passed to Claude Code; nothing else of the
/// daemon's environment is (B-158). No `ANTHROPIC_*` variable and no
/// `CLAUDECODE`, so a daemon started inside a Claude Code session does not
/// hand that session's state on.
pub const PASS_ENV: &[&str] = &[
    "PATH",
    "USERPROFILE",
    "HOME",
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "ComSpec",
    "PATHEXT",
    "windir",
    "ProgramData",
    "USERNAME",
    "USER",
    "LOGNAME",
    "LANG",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_GIT_BASH_PATH",
    "SystemDrive",
    "HOMEDRIVE",
    "HOMEPATH",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "CommonProgramFiles",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "OS",
];

/// The judge that runs Claude Code under the user's own sign-in.
#[derive(Debug, Clone)]
pub struct ClaudeJudge {
    model: String,
    setting: Option<String>,
    work_dir: PathBuf,
    timeout: Duration,
}

impl ClaudeJudge {
    /// A judge with the model and executable of `settings`, run in
    /// `work_dir` (created when missing).
    pub fn new(settings: &JudgeSettings, work_dir: PathBuf) -> ClaudeJudge {
        ClaudeJudge {
            model: settings.model.clone(),
            setting: settings.command.clone(),
            work_dir,
            timeout: CALL_TIMEOUT,
        }
    }

    /// The same judge with another call timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> ClaudeJudge {
        self.timeout = timeout;
        self
    }

    fn executable(&self) -> Option<PathBuf> {
        find_claude_in(
            self.setting.as_deref(),
            std::env::var_os("PATH").as_deref(),
            sm_core::paths::home_dir().as_deref(),
        )
    }

    /// The command that runs one call with `prompt` and `schema`; its
    /// arguments and environment are checked by the tests (B-182, B-192).
    pub(crate) fn command_line(&self, exe: &Path, prompt: &str, schema: &str) -> Command {
        let mut cmd = Command::new(exe);
        cmd.args([
            "-p",
            "--model",
            self.model.as_str(),
            "--safe-mode",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            r#"{"mcpServers":{}}"#,
            "--tools",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--permission-prompts",
            "none",
            "--system-prompt",
            prompt,
            "--json-schema",
            schema,
            "--output-format",
            "stream-json",
            "--verbose",
        ])
        .current_dir(&self.work_dir)
        .env_clear()
        .envs(passed_env())
        .env(JUDGE_MARKER_ENV, "1")
        .env(NO_PROMPT_CACHING_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        below_normal(&mut cmd);
        cmd
    }
}

impl Judge for ClaudeJudge {
    fn model(&self) -> &str {
        &self.model
    }

    fn command(&self) -> Option<String> {
        self.executable().map(|p| p.display().to_string())
    }

    fn judge(
        &self,
        batch: &[JudgeCandidate],
        stop: &AtomicBool,
    ) -> Result<JudgeAnswer, JudgeFailure> {
        let out = self.run(PROMPT, SCHEMA, batch, "the candidates", stop)?;
        parse_output(&out.stdout, out.code, &out.stderr)
    }

    fn repeats(
        &self,
        items: &[RepeatItem],
        stop: &AtomicBool,
    ) -> Result<RepeatAnswer, JudgeFailure> {
        let out = self.run(
            REPEATS_PROMPT,
            REPEATS_SCHEMA,
            items,
            "the kept memories",
            stop,
        )?;
        parse_repeats_output(&out.stdout, out.code, &out.stderr)
    }
}

/// What one finished call printed, and how it exited.
struct Finished {
    stdout: Vec<u8>,
    code: Option<i32>,
    stderr: Vec<u8>,
}

impl ClaudeJudge {
    /// Runs one call with `prompt` and `schema` and `input` on standard
    /// input; returns its standard output, exit code and standard error.
    fn run<T: serde::Serialize + ?Sized>(
        &self,
        prompt: &str,
        schema: &str,
        input: &T,
        what: &str,
        stop: &AtomicBool,
    ) -> Result<Finished, JudgeFailure> {
        let exe = self.executable().ok_or_else(|| {
            failure(
                JudgeFailureKind::Unavailable,
                "Claude Code was not found: install it, or name its executable in \
                 capture.judge.command",
            )
        })?;
        std::fs::create_dir_all(&self.work_dir).map_err(|e| {
            failure(
                JudgeFailureKind::CallFailed,
                format!("cannot create the judge's folder: {e}"),
            )
        })?;
        let input = serde_json::to_vec(input).map_err(|e| {
            failure(
                JudgeFailureKind::CallFailed,
                format!("cannot write {what}: {e}"),
            )
        })?;
        let mut child = self
            .command_line(&exe, prompt, schema)
            .spawn()
            .map_err(|e| {
                let kind = if e.kind() == std::io::ErrorKind::NotFound {
                    JudgeFailureKind::Unavailable
                } else {
                    JudgeFailureKind::CallFailed
                };
                failure(kind, format!("cannot start Claude Code: {e}"))
            })?;
        lower_priority(&child);
        let pipes = Pipes::start(&mut child, input);
        let pipes = match pipes {
            Ok(p) => p,
            Err(e) => {
                end(&mut child);
                return Err(failure(
                    JudgeFailureKind::CallFailed,
                    format!("cannot read Claude Code's output: {e}"),
                ));
            }
        };
        let status = match wait(&mut child, self.timeout, stop) {
            Ok(status) => status,
            Err(f) => {
                end(&mut child);
                pipes.abandon();
                return Err(f);
            }
        };
        let (stdout, stderr) = pipes.finish();
        Ok(Finished {
            stdout,
            code: status.code(),
            stderr,
        })
    }
}

fn failure(kind: JudgeFailureKind, message: impl Into<String>) -> JudgeFailure {
    JudgeFailure {
        kind,
        message: message.into(),
        retry_at: None,
        cost_usd: 0.0,
    }
}

fn passed_env() -> Vec<(OsString, OsString)> {
    PASS_ENV
        .iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (OsString::from(k), v)))
        .collect()
}

#[cfg(windows)]
fn below_normal(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    /// `BELOW_NORMAL_PRIORITY_CLASS`.
    const BELOW_NORMAL: u32 = 0x0000_4000;
    /// `CREATE_NO_WINDOW`: no console window for the child.
    const NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(BELOW_NORMAL | NO_WINDOW);
}

#[cfg(not(windows))]
fn below_normal(_cmd: &mut Command) {}

#[cfg(unix)]
#[allow(unsafe_code)]
fn lower_priority(child: &Child) {
    // `id_t` is `u32` on Linux and macOS, the same type as the child's id.
    let pid: libc::id_t = child.id();
    // SAFETY: `setpriority` takes plain integers and reads or writes no
    // memory of this process; it changes only the scheduling priority of
    // the child just started. A failure (the child already gone) is
    // harmless and ignored.
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, pid, 10);
    }
}

#[cfg(not(unix))]
fn lower_priority(_child: &Child) {}

/// Ends the child and reaps it.
fn end(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Waits for the child, polling so the timeout and the daemon's stop are
/// seen.
fn wait(
    child: &mut Child,
    timeout: Duration,
    stop: &AtomicBool,
) -> Result<ExitStatus, JudgeFailure> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(e) => {
                return Err(failure(
                    JudgeFailureKind::CallFailed,
                    format!("cannot wait for Claude Code: {e}"),
                ));
            }
        }
        if stop.load(Ordering::SeqCst) {
            return Err(failure(JudgeFailureKind::Stopped, "the daemon is stopping"));
        }
        if Instant::now() >= deadline {
            return Err(failure(
                JudgeFailureKind::Timeout,
                format!("Claude Code did not answer within {} s", timeout.as_secs()),
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The child's three pipes, each served by its own thread so none fills.
struct Pipes {
    stdin: Option<JoinHandle<()>>,
    stdout: JoinHandle<Vec<u8>>,
    stderr: JoinHandle<Vec<u8>>,
}

impl Pipes {
    fn start(child: &mut Child, input: Vec<u8>) -> std::io::Result<Pipes> {
        let missing = || std::io::Error::other("a pipe of the child is missing");
        let mut stdin = child.stdin.take().ok_or_else(missing)?;
        let stdout = child.stdout.take().ok_or_else(missing)?;
        let stderr = child.stderr.take().ok_or_else(missing)?;
        let stdout = drain("sm-judge-out", stdout)?;
        let stderr = drain("sm-judge-err", stderr)?;
        // A child that exits before it reads (signed out) breaks the pipe;
        // that write error is the child's answer, not ours.
        let stdin = std::thread::Builder::new()
            .name("sm-judge-in".to_owned())
            .spawn(move || {
                let _ = stdin.write_all(&input);
            })?;
        Ok(Pipes {
            stdin: Some(stdin),
            stdout,
            stderr,
        })
    }

    /// The output read, once the child has exited. A pipe still open after
    /// 5 seconds (a grandchild holding it) is left with what was read.
    fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !(self.stdout.is_finished() && self.stderr.is_finished()) && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        if let Some(t) = self.stdin.take()
            && t.is_finished()
        {
            let _ = t.join();
        }
        let take = |t: JoinHandle<Vec<u8>>| {
            if t.is_finished() {
                t.join().unwrap_or_default()
            } else {
                Vec::new()
            }
        };
        (take(self.stdout), take(self.stderr))
    }

    /// The child was ended: its pipes close and the threads finish alone.
    fn abandon(self) {}
}

fn drain(name: &str, mut from: impl Read + Send + 'static) -> std::io::Result<JoinHandle<Vec<u8>>> {
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let mut kept = Vec::new();
            let mut buf = [0u8; 16 * 1024];
            loop {
                match from.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let room = MAX_OUTPUT_BYTES.saturating_sub(kept.len());
                        kept.extend_from_slice(&buf[..n.min(room)]);
                    }
                }
            }
            kept
        })
}

/// The Claude Code executable (B-158). `setting` is
/// `capture.judge.command`: a path, or a bare name looked for on
/// `path_var`. A `.cmd` or `.bat` wrapper (npm's) is replaced by the
/// `claude.exe` of the package beside it, so no `cmd.exe` parses the
/// arguments. Without a setting: on Windows `claude.exe` on `PATH`, then
/// the npm package's `claude.exe` beside a `claude.cmd` or `claude` on
/// `PATH`, then `~\.local\bin\claude.exe` (the native installer's); on
/// other systems `claude` on `PATH`, then `~/.local/bin/claude`. `None`
/// when nothing is found.
pub fn find_claude_in(
    setting: Option<&str>,
    path_var: Option<&OsStr>,
    home: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(s) = setting {
        let p = PathBuf::from(s.trim());
        let named = if p.components().count() > 1 || p.is_absolute() {
            p
        } else {
            bare(path_var, p.as_os_str())?
        };
        return if is_wrapper(&named) {
            npm_binary(&named)
        } else {
            named.is_file().then_some(named)
        };
    }
    if cfg!(windows) {
        on_path(path_var, OsStr::new("claude.exe"))
            .or_else(|| {
                ["claude.cmd", "claude"]
                    .iter()
                    .filter_map(|n| on_path(path_var, OsStr::new(n)))
                    .find_map(|w| npm_binary(&w))
            })
            .or_else(|| {
                home.map(|h| h.join(".local").join("bin").join("claude.exe"))
                    .filter(|p| p.is_file())
            })
    } else {
        on_path(path_var, OsStr::new("claude")).or_else(|| {
            home.map(|h| h.join(".local").join("bin").join("claude"))
                .filter(|p| p.is_file())
        })
    }
}

/// A bare name of the setting on `PATH`. On Windows a name without an
/// extension is tried with `.exe`, then `.cmd`, then as it is.
fn bare(path_var: Option<&OsStr>, name: &OsStr) -> Option<PathBuf> {
    if cfg!(windows) && Path::new(name).extension().is_none() {
        for ext in ["exe", "cmd"] {
            let mut n = name.to_owned();
            n.push(".");
            n.push(ext);
            if let Some(p) = on_path(path_var, &n) {
                return Some(p);
            }
        }
    }
    on_path(path_var, name)
}

fn on_path(path_var: Option<&OsStr>, name: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(path_var?)
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

fn is_wrapper(p: &Path) -> bool {
    p.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"))
}

/// The `claude.exe` of the npm package installed beside a wrapper.
fn npm_binary(wrapper: &Path) -> Option<PathBuf> {
    let exe = wrapper
        .parent()?
        .join("node_modules")
        .join("@anthropic-ai")
        .join("claude-code")
        .join("bin")
        .join("claude.exe");
    exe.is_file().then_some(exe)
}

#[derive(Deserialize)]
struct Answer {
    verdicts: Vec<JudgeVerdict>,
}

#[derive(Deserialize)]
struct RepeatsAnswer {
    answers: Vec<RepeatVerdict>,
}

/// A call's result that reported no error: its structured output, cost,
/// duration and model.
struct Structured {
    value: Value,
    cost_usd: f64,
    duration_ms: u64,
    model: Option<String>,
}

/// Reads one call's `stream-json` output (B-158). Only lines that start
/// with `{` are read. A `rate_limit_event` whose status is `rejected`, a
/// result with `is_error`, a non-zero exit, no result, and a result whose
/// `structured_output` is missing or breaks the schema each fail the
/// whole batch. No memory text is put in a failure's message: Claude
/// Code's own error text and the tail of standard error are, cut to 300
/// characters. A schema error is cut before the first value or name the
/// answer wrote (see [`schema_error`]), since the model writes the answer
/// after reading the captures, and the message reaches `status` (B-181).
pub fn parse_output(
    stdout: &[u8],
    exit_code: Option<i32>,
    stderr: &[u8],
) -> Result<JudgeAnswer, JudgeFailure> {
    let r = parse_result(stdout, exit_code, stderr)?;
    let answer: Answer = serde_json::from_value(r.value).map_err(|e| JudgeFailure {
        kind: JudgeFailureKind::BadAnswer,
        message: format!("the answer breaks the schema: {}", schema_error(&e)),
        retry_at: None,
        cost_usd: r.cost_usd,
    })?;
    Ok(JudgeAnswer {
        verdicts: answer.verdicts,
        cost_usd: r.cost_usd,
        duration_ms: r.duration_ms,
        model: r.model,
    })
}

/// Reads the duplicate check's output as [`parse_output`] reads the
/// judge's (B-192).
pub fn parse_repeats_output(
    stdout: &[u8],
    exit_code: Option<i32>,
    stderr: &[u8],
) -> Result<RepeatAnswer, JudgeFailure> {
    let r = parse_result(stdout, exit_code, stderr)?;
    let answer: RepeatsAnswer = serde_json::from_value(r.value).map_err(|e| JudgeFailure {
        kind: JudgeFailureKind::BadAnswer,
        message: format!(
            "the duplicate check's answer breaks the schema: {}",
            schema_error(&e)
        ),
        retry_at: None,
        cost_usd: r.cost_usd,
    })?;
    Ok(RepeatAnswer {
        verdicts: answer.answers,
        cost_usd: r.cost_usd,
        duration_ms: r.duration_ms,
    })
}

/// The structured output of one call, or why the call failed.
fn parse_result(
    stdout: &[u8],
    exit_code: Option<i32>,
    stderr: &[u8],
) -> Result<Structured, JudgeFailure> {
    let text = String::from_utf8_lossy(stdout);
    let mut result: Option<Value> = None;
    let mut rejected: Option<Option<Timestamp>> = None;
    let mut model: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("system") if v.get("subtype").and_then(Value::as_str) == Some("init") => {
                model = v.get("model").and_then(Value::as_str).map(str::to_owned);
            }
            Some("rate_limit_event") => {
                let info = &v["rate_limit_info"];
                if info.get("status").and_then(Value::as_str) == Some("rejected") {
                    rejected = Some(reset_time(&info["resetsAt"]));
                }
            }
            Some("result") => result = Some(v),
            _ => {}
        }
    }
    let cost = result
        .as_ref()
        .and_then(|r| r.get("total_cost_usd"))
        .and_then(Value::as_f64)
        .filter(|c| c.is_finite() && *c >= 0.0)
        .unwrap_or(0.0);
    let fail = |kind, message: String, retry_at| JudgeFailure {
        kind,
        message,
        retry_at,
        cost_usd: cost,
    };
    if let Some(retry_at) = rejected {
        return Err(fail(
            JudgeFailureKind::RateLimited,
            "Claude Code reported that the plan's rate limit was reached".to_owned(),
            retry_at,
        ));
    }
    let Some(r) = result else {
        let tail = clip_tail(&String::from_utf8_lossy(stderr));
        let exit = exit_code.map_or_else(|| "no exit code".to_owned(), |c| format!("exit {c}"));
        return Err(fail(
            JudgeFailureKind::CallFailed,
            if tail.is_empty() {
                format!("Claude Code gave no result ({exit})")
            } else {
                format!("Claude Code gave no result ({exit}): {tail}")
            },
            None,
        ));
    };
    if r.get("is_error").and_then(Value::as_bool) == Some(true) {
        let message = r
            .get("result")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| r.get("subtype").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_else(|| "an error".to_owned());
        let message = clip(&message);
        let lower = message.to_lowercase();
        let status = r.get("api_error_status").and_then(Value::as_u64);
        if status == Some(429) || lower.contains("rate limit") || lower.contains("usage limit") {
            return Err(fail(
                JudgeFailureKind::RateLimited,
                format!("Claude Code reported a limit: {message}"),
                None,
            ));
        }
        if lower.contains("not logged in") || lower.contains("/login") {
            return Err(fail(
                JudgeFailureKind::Unavailable,
                format!("Claude Code is not signed in ({message}): run `claude` once and sign in"),
                None,
            ));
        }
        return Err(fail(
            JudgeFailureKind::CallFailed,
            format!("Claude Code reported an error: {message}"),
            None,
        ));
    }
    if exit_code != Some(0) {
        let exit = exit_code.map_or_else(|| "no exit code".to_owned(), |c| format!("exit {c}"));
        return Err(fail(
            JudgeFailureKind::CallFailed,
            format!("Claude Code ended with {exit}"),
            None,
        ));
    }
    let structured = match r.get("structured_output") {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
        Some(Value::Null) | None => None,
        Some(v) => Some(v.clone()),
    };
    let Some(structured) = structured else {
        return Err(fail(
            JudgeFailureKind::BadAnswer,
            "the answer has no structured output".to_owned(),
            None,
        ));
    };
    Ok(Structured {
        value: structured,
        cost_usd: cost,
        duration_ms: r.get("duration_ms").and_then(Value::as_u64).unwrap_or(0),
        model,
    })
}

/// `resetsAt`: Unix seconds, Unix milliseconds (above 10^12) or an RFC 3339
/// string.
fn reset_time(v: &Value) -> Option<Timestamp> {
    match v {
        Value::Number(n) => {
            let n = n.as_f64()?;
            if !n.is_finite() || n <= 0.0 {
                return None;
            }
            let seconds = if n > 1e12 { n / 1000.0 } else { n };
            Some(Timestamp::from_unix_seconds(seconds as i64))
        }
        Value::String(s) => Timestamp::parse(s).ok(),
        _ => None,
    }
}

/// A schema error without the answer's own text (B-181). serde quotes a
/// value it read (`string "..."`) and backticks a field or variant name it
/// did not know, and the model wrote both, so the message is cut at the
/// first quote or backtick.
fn schema_error(e: &serde_json::Error) -> String {
    let s = e.to_string();
    let head = s.find(['"', '`']).map_or(s.as_str(), |cut| &s[..cut]);
    clip(head.trim_end().trim_end_matches(':'))
}

fn clip(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= MAX_MESSAGE_CHARS {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(MAX_MESSAGE_CHARS).collect();
    out.push_str("...");
    out
}

fn clip_tail(s: &str) -> String {
    let s = s.trim();
    let n = s.chars().count();
    if n <= MAX_MESSAGE_CHARS {
        return s.to_owned();
    }
    let mut out = "...".to_owned();
    out.extend(s.chars().skip(n - MAX_MESSAGE_CHARS));
    out
}
