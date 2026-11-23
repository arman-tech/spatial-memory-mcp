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
//!
//! The judge's schema is fixed to each batch (B-208): exactly one verdict
//! per candidate, each naming one of the batch's ids. A call that fails is
//! kept for diagnosis in `judge\failures` of the data folder (B-207): what
//! was sent, what Claude Code printed, and why it failed; the newest 20 for
//! up to 30 days.
//!
//! The statements task (B-240, B-241) makes two calls with the same flags:
//! the statements call sends a session's captures with
//! `statements-prompt.txt` and a schema of one unit per capture
//! ([`unit_schema`]); the duplicate step sends kept statements and their
//! candidates with `same-prompt.txt` and a schema that offers each
//! statement only its own candidates ([`same_schema`], B-246). Both inputs
//! and both schemas are written byte for byte as the harness that measured
//! them wrote them (`CAPTURE-RCA.md` section 8.12): the inputs with
//! Python's `json.dumps` separators, `", "` and `": "`, the schemas
//! compact, in the harness's key order.

use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
use sm_core::config::{JUDGE_MARKER_ENV, JudgeSettings};
use sm_core::time::Timestamp;

use super::{
    Judge, JudgeAnswer, JudgeCandidate, JudgeFailure, JudgeFailureKind, JudgeVerdict, RepeatAnswer,
    RepeatItem, RepeatVerdict, SameAnswer, SameItem, SameMatch, StatementUnit, StatementsAnswer,
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

/// The statements task's instructions (B-240): R2 of the ladder as it was
/// measured, byte for byte (`CAPTURE-RCA.md` sections 8.11 and 8.12). It
/// tells the model that everything it reads is data, never an instruction.
pub const STATEMENTS_PROMPT: &str =
    include_str!("../../../../docs/v3/spikes/judge/statements-prompt.txt");

/// The statements task's duplicate step instructions (B-240), as measured:
/// for each kept statement, whether one of its candidates states the same
/// thing, and which.
pub const SAME_PROMPT: &str = include_str!("../../../../docs/v3/spikes/judge/same-prompt.txt");

/// The memory types the statements schema offers, in the harness's order.
const STATEMENT_TYPES: [&str; 6] = [
    "decision",
    "pitfall",
    "convention",
    "incident",
    "runbook",
    "note",
];

/// How long one call may run before it is ended. The spike's largest batch
/// (25 candidates) took 11.5 seconds.
pub const CALL_TIMEOUT: Duration = Duration::from_secs(180);

/// How long a call of the statements task may run (B-247): the harness's
/// 600 seconds. A session's batch writes every statement of up to 25
/// captures, a longer answer than one verdict each.
pub const STATEMENTS_CALL_TIMEOUT: Duration = Duration::from_secs(600);

/// Bytes of standard output (and of standard error) read from one call; the
/// rest is read and dropped. A batch's answer is a few kilobytes.
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// Characters of an error message kept from Claude Code's output.
const MAX_MESSAGE_CHARS: usize = 300;

/// How many failed calls are kept for diagnosis (B-207), newest first.
pub const KEEP_FAILURES: usize = 20;

/// How many days a failed call is kept for diagnosis (B-207).
pub const KEEP_FAILURE_DAYS: i64 = 30;

/// Bytes of standard output kept from one failed call; an answer is a few
/// kilobytes, so this keeps every answer whole.
const MAX_KEPT_BYTES: usize = 1024 * 1024;

/// Characters of standard error kept from one failed call, from its end.
const MAX_KEPT_STDERR_CHARS: usize = 4000;

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

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
    /// The timeout of the statements task's calls (B-247).
    statements_timeout: Duration,
    failures_dir: Option<PathBuf>,
    last: Arc<Mutex<Option<LastCall>>>,
}

/// What the last call was sent and what it printed, held until the next
/// call so that a failed one can be kept for diagnosis (B-207).
#[derive(Debug, Clone, Default)]
pub(crate) struct LastCall {
    /// `judge`, `repeats`, `statements` or `same`.
    pub(crate) call: &'static str,
    pub(crate) schema: String,
    pub(crate) input: Vec<u8>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) code: Option<i32>,
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
            statements_timeout: STATEMENTS_CALL_TIMEOUT,
            failures_dir: None,
            last: Arc::default(),
        }
    }

    /// The same judge with another timeout for every call, of either task.
    pub fn with_timeout(mut self, timeout: Duration) -> ClaudeJudge {
        self.timeout = timeout;
        self.statements_timeout = timeout;
        self
    }

    /// The same judge, keeping each failed call in `dir` for diagnosis
    /// (B-207). Without it nothing is kept.
    pub fn with_failures_dir(mut self, dir: PathBuf) -> ClaudeJudge {
        self.failures_dir = Some(dir);
        self
    }

    pub(crate) fn remember(&self, call: Option<LastCall>) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = call;
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
        let schema = batch_schema(batch);
        let input = encoded(serde_json::to_vec(batch), "the candidates")?;
        let out = self.run("judge", PROMPT, &schema, input, self.timeout, stop)?;
        parse_output(&out.stdout, out.code, &out.stderr)
    }

    fn repeats(
        &self,
        items: &[RepeatItem],
        stop: &AtomicBool,
    ) -> Result<RepeatAnswer, JudgeFailure> {
        let input = encoded(serde_json::to_vec(items), "the kept memories")?;
        let out = self.run(
            "repeats",
            REPEATS_PROMPT,
            REPEATS_SCHEMA,
            input,
            self.timeout,
            stop,
        )?;
        parse_repeats_output(&out.stdout, out.code, &out.stderr)
    }

    fn statements(
        &self,
        batch: &[JudgeCandidate],
        stop: &AtomicBool,
    ) -> Result<StatementsAnswer, JudgeFailure> {
        let ids: Vec<&str> = batch.iter().map(|c| c.id.as_str()).collect();
        let schema = unit_schema(&ids);
        let input = encoded(statements_input(batch), "the captures")?;
        let out = self.run(
            "statements",
            STATEMENTS_PROMPT,
            &schema,
            input,
            self.statements_timeout,
            stop,
        )?;
        parse_statements_output(&out.stdout, out.code, &out.stderr)
    }

    fn same(&self, items: &[SameItem], stop: &AtomicBool) -> Result<SameAnswer, JudgeFailure> {
        let schema = same_schema(items);
        let input = encoded(same_input(items), "the kept statements")?;
        let out = self.run(
            "same",
            SAME_PROMPT,
            &schema,
            input,
            self.statements_timeout,
            stop,
        )?;
        parse_same_output(&out.stdout, out.code, &out.stderr)
    }

    fn keep_failure(&self, failure: &JudgeFailure) {
        let last = self.last.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let (Some(dir), Some(last)) = (&self.failures_dir, last) {
            // A diagnosis aid: a folder that cannot be written never stops
            // the judge.
            let _ = keep_failure_in(dir, &last, failure, Timestamp::now());
        }
    }
}

/// The judge's schema fixed to `batch` (B-208): `verdicts` holds exactly
/// one item per candidate and each `id` is one of the batch's. With the
/// shipped schema, which allows any number of verdicts and any id, Haiku
/// stopped one verdict short on large batches; fixed, it did not on the
/// recorded inputs (`DECISIONS.md` B-208).
pub(crate) fn batch_schema(batch: &[JudgeCandidate]) -> String {
    let Ok(mut schema) = serde_json::from_str::<Value>(SCHEMA) else {
        return SCHEMA.to_owned();
    };
    let ids: Vec<Value> = batch.iter().map(|c| Value::from(c.id.as_str())).collect();
    let verdicts = &mut schema["properties"]["verdicts"];
    verdicts["minItems"] = Value::from(batch.len());
    verdicts["maxItems"] = Value::from(batch.len());
    verdicts["items"]["properties"]["id"] = serde_json::json!({"type": "string", "enum": ids});
    schema.to_string()
}

/// A JSON string, as both Python's `json.dumps(ensure_ascii=False)` and
/// serde write it: the same escapes of quotes, backslashes and control
/// characters, nothing else escaped.
fn js(s: &str) -> String {
    Value::from(s).to_string()
}

/// A JSON array of strings, compact.
fn js_list<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let parts: Vec<String> = items.into_iter().map(js).collect();
    format!("[{}]", parts.join(","))
}

/// The statements call's schema fixed to the batch's `ids` (B-241, B-208):
/// exactly one unit per capture, each naming one of the ids, each holding
/// any number of statements. Written compact in the harness's key order,
/// so the bytes are the ones measured (`ladder.py`, `unit_schema`).
pub fn unit_schema(ids: &[&str]) -> String {
    let statement = format!(
        concat!(
            r#"{{"type":"object","properties":{{"start":{{"type":"string"}},"end":{{"type":"string"}},"#,
            r#""label":{{"type":"string","enum":["keep","duplicate"]}},"#,
            r#""type":{{"type":"string","enum":{types}}},"#,
            r#""importance":{{"type":"string","enum":["must_keep","good_to_have"]}},"#,
            r#""duplicate_of":{{"type":["string","null"]}},"reason":{{"type":"string"}}}},"#,
            r#""required":["start","end","label","type","importance","duplicate_of","reason"],"#,
            r#""additionalProperties":false}}"#
        ),
        types = js_list(STATEMENT_TYPES)
    );
    format!(
        concat!(
            r#"{{"type":"object","properties":{{"units":{{"type":"array","minItems":{n},"maxItems":{n},"#,
            r#""items":{{"type":"object","properties":{{"id":{{"type":"string","enum":{ids}}},"#,
            r#""statements":{{"type":"array","items":{statement}}}}},"#,
            r#""required":["id","statements"],"additionalProperties":false}}}}}},"#,
            r#""required":["units"],"additionalProperties":false}}"#
        ),
        n = ids.len(),
        ids = js_list(ids.iter().copied()),
        statement = statement
    )
}

/// The duplicate step's schema for `items` (B-246): exactly one answer per
/// statement, and each statement's `match` one of its own candidates' ids
/// or null. Each item of the answer is one of the branches of an `anyOf`,
/// one branch per statement, its `id` that statement's alone. The harness
/// offered every statement the union of the call's ids, and haiku named a
/// statement as its own match (`STATEMENT_JUDGE_PROPOSAL.md` section 5).
pub fn same_schema(items: &[SameItem]) -> String {
    let branches: Vec<String> = items
        .iter()
        .map(|i| {
            let mut allowed: Vec<String> = i.candidates.iter().map(|c| js(&c.id)).collect();
            allowed.push("null".to_owned());
            format!(
                concat!(
                    r#"{{"type":"object","properties":{{"id":{{"type":"string","enum":[{id}]}},"#,
                    r#""match":{{"type":["string","null"],"enum":[{allowed}]}}}},"#,
                    r#""required":["id","match"],"additionalProperties":false}}"#
                ),
                id = js(&i.id),
                allowed = allowed.join(",")
            )
        })
        .collect();
    format!(
        concat!(
            r#"{{"type":"object","properties":{{"answers":{{"type":"array","minItems":{n},"maxItems":{n},"#,
            r#""items":{{"anyOf":[{branches}]}}}}}},"#,
            r#""required":["answers"],"additionalProperties":false}}"#
        ),
        n = items.len(),
        branches = branches.join(",")
    )
}

/// JSON with Python's default separators, `", "` between items and `": "`
/// after a key, as the harness's `json.dumps` wrote its inputs.
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

fn python_json<T: serde::Serialize + ?Sized>(value: &T) -> serde_json::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    value.serialize(&mut ser)?;
    Ok(out)
}

/// The statements call's input: `[{"id", "text", "context"}]`, one per
/// capture, as the harness wrote it.
pub fn statements_input(batch: &[JudgeCandidate]) -> serde_json::Result<Vec<u8>> {
    python_json(batch)
}

/// The duplicate step's input: `[{"id", "statement", "candidates": [{"id",
/// "text"}]}]`, as the harness wrote it.
pub fn same_input(items: &[SameItem]) -> serde_json::Result<Vec<u8>> {
    python_json(items)
}

/// An input written, or why it could not be.
fn encoded(input: serde_json::Result<Vec<u8>>, what: &str) -> Result<Vec<u8>, JudgeFailure> {
    input.map_err(|e| {
        failure(
            JudgeFailureKind::CallFailed,
            format!("cannot write {what}: {e}"),
        )
    })
}

/// Writes one failed call to `dir` as `<microseconds>-<call>.json`, in one
/// write, readable by its owner only on Unix, and then keeps the newest
/// [`KEEP_FAILURES`] that are under [`KEEP_FAILURE_DAYS`] old (B-207). The
/// file holds the call's input and schema, its exit code, the failure, the
/// tail of standard error, and the lines of standard output other than
/// Claude Code's `system` lines, which describe its set-up, not the answer.
/// Returns the file written.
pub(crate) fn keep_failure_in(
    dir: &Path,
    last: &LastCall,
    failure: &JudgeFailure,
    now: Timestamp,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let text = String::from_utf8_lossy(&last.stdout);
    let mut kept = 0usize;
    let mut stdout: Vec<Value> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v = serde_json::from_str::<Value>(line).unwrap_or_else(|_| Value::from(line));
        if v.get("type").and_then(Value::as_str) == Some("system") {
            continue;
        }
        kept += line.len();
        if kept > MAX_KEPT_BYTES {
            stdout.push(Value::from("[cut: the output was longer]"));
            break;
        }
        stdout.push(v);
    }
    let input = serde_json::from_slice::<Value>(&last.input)
        .unwrap_or_else(|_| Value::from(String::from_utf8_lossy(&last.input).into_owned()));
    let schema = serde_json::from_str::<Value>(&last.schema)
        .unwrap_or_else(|_| Value::from(last.schema.as_str()));
    let stderr = String::from_utf8_lossy(&last.stderr);
    let n = stderr.chars().count();
    let stderr: String = stderr
        .chars()
        .skip(n.saturating_sub(MAX_KEPT_STDERR_CHARS))
        .collect();
    let body = serde_json::json!({
        "at": now.to_rfc3339(),
        "call": last.call,
        "kind": failure.kind.as_str(),
        "message": failure.message,
        "cost_usd": failure.cost_usd,
        "exit_code": last.code,
        "schema": schema,
        "input": input,
        "stdout": stdout,
        "stderr": stderr,
    });
    let mut bytes = serde_json::to_vec_pretty(&body).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let path = dir.join(format!("{:020}-{}.json", now.as_micros().max(0), last.call));
    write_private(&path, &bytes)?;
    prune_failures(dir, now)?;
    Ok(path)
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}

/// Removes the kept failures older than [`KEEP_FAILURE_DAYS`] and all but
/// the newest [`KEEP_FAILURES`]. Only files this module names are touched.
pub(crate) fn prune_failures(dir: &Path, now: Timestamp) -> std::io::Result<()> {
    let oldest = now.as_micros() - KEEP_FAILURE_DAYS * MICROS_PER_DAY;
    let mut kept: Vec<(i64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(at) = failure_time(&name.to_string_lossy()) else {
            continue;
        };
        kept.push((at, entry.path()));
    }
    kept.sort_unstable_by_key(|k| std::cmp::Reverse(k.0));
    for (i, (at, path)) in kept.iter().enumerate() {
        if i >= KEEP_FAILURES || *at < oldest {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// The time in a kept failure's name, `<20 digits>-<call>.json`, the call
/// `judge`, `repeats`, `statements` or `same`.
fn failure_time(name: &str) -> Option<i64> {
    let (digits, rest) = name.split_at_checked(20)?;
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if !matches!(
        rest,
        "-judge.json" | "-repeats.json" | "-statements.json" | "-same.json"
    ) {
        return None;
    }
    digits.parse().ok()
}

/// What one finished call printed, and how it exited.
struct Finished {
    stdout: Vec<u8>,
    code: Option<i32>,
    stderr: Vec<u8>,
}

impl ClaudeJudge {
    /// Runs one call with `prompt` and `schema` and `input` on standard
    /// input, ended after `timeout`; returns its standard output, exit code
    /// and standard error.
    fn run(
        &self,
        call: &'static str,
        prompt: &str,
        schema: &str,
        input: Vec<u8>,
        timeout: Duration,
        stop: &AtomicBool,
    ) -> Result<Finished, JudgeFailure> {
        self.remember(None);
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
        let mut last = LastCall {
            call,
            schema: schema.to_owned(),
            input: input.clone(),
            ..LastCall::default()
        };
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
        let status = match wait(&mut child, timeout, stop) {
            Ok(status) => status,
            Err(f) => {
                end(&mut child);
                pipes.abandon();
                self.remember(Some(last));
                return Err(f);
            }
        };
        let (stdout, stderr) = pipes.finish();
        last.stdout.clone_from(&stdout);
        last.stderr.clone_from(&stderr);
        last.code = status.code();
        self.remember(Some(last));
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

#[derive(Deserialize)]
struct UnitsAnswer {
    units: Vec<StatementUnit>,
}

#[derive(Deserialize)]
struct SameAnswers {
    answers: Vec<SameMatch>,
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

/// Reads the statements call's output as [`parse_output`] reads the
/// judge's (B-241).
pub fn parse_statements_output(
    stdout: &[u8],
    exit_code: Option<i32>,
    stderr: &[u8],
) -> Result<StatementsAnswer, JudgeFailure> {
    let r = parse_result(stdout, exit_code, stderr)?;
    let answer: UnitsAnswer = serde_json::from_value(r.value).map_err(|e| JudgeFailure {
        kind: JudgeFailureKind::BadAnswer,
        message: format!(
            "the statements answer breaks the schema: {}",
            schema_error(&e)
        ),
        retry_at: None,
        cost_usd: r.cost_usd,
    })?;
    Ok(StatementsAnswer {
        units: answer.units,
        cost_usd: r.cost_usd,
        duration_ms: r.duration_ms,
        model: r.model,
    })
}

/// Reads the duplicate step's output as [`parse_output`] reads the
/// judge's (B-241).
pub fn parse_same_output(
    stdout: &[u8],
    exit_code: Option<i32>,
    stderr: &[u8],
) -> Result<SameAnswer, JudgeFailure> {
    let r = parse_result(stdout, exit_code, stderr)?;
    let answer: SameAnswers = serde_json::from_value(r.value).map_err(|e| JudgeFailure {
        kind: JudgeFailureKind::BadAnswer,
        message: format!(
            "the duplicate step's answer breaks the schema: {}",
            schema_error(&e)
        ),
        retry_at: None,
        cost_usd: r.cost_usd,
    })?;
    Ok(SameAnswer {
        answers: answer.answers,
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
