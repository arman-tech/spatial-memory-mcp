//! `spatial-memory hook <event> --client <claude-code|cursor>` (SPEC section
//! 16; CONTRACTS section 7, the hook client).
//!
//! The hook never blocks and never fails: every path ends with the event's
//! output (or nothing) and exit code 0. Time is counted from the start of
//! `main`. The prompt hook has `ambient::DEADLINE_MS` (1,500 ms) in all; a
//! capture hook has `ambient::CAPTURE_HOOK_BUDGET_MS` (200 ms) to hand its
//! request to the daemon or to the spool.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use sm_core::ambient;
use sm_core::config::{self, AmbientSwitches};
use sm_core::error::ErrorCode;
use sm_core::journal::{HookGate, Journal, metrics};
use sm_core::protocol::{
    AmbientParams, AmbientReply, CaptureParams, HookEvent, Op, Outcome, Reply, Request,
};
use sm_core::spool;
use sm_core::time::Timestamp;
use sm_core::types::Client;

use crate::client::{self, ExchangeError, Spawn};
use crate::hook_input::{self, HookArgs};
use crate::profile::{self, GetEnv};

/// Time kept back from the capture budget to write the spool file.
pub const SPOOL_RESERVE: Duration = Duration::from_millis(40);
/// Time kept back from the prompt deadline to print, journal and exit. It
/// also covers process start before `main`, which the hook cannot see:
/// AT-3 measures from outside the process. Measured on Windows, start, journal
/// and exit together took 13 to 21 ms past the point where the hook gave up.
pub const PRINT_RESERVE: Duration = Duration::from_millis(100);

/// What the hook needs from its process, so tests can supply their own.
pub struct HookEnv<'a> {
    pub get_env: GetEnv<'a>,
    /// The process working folder, used when the payload has no valid `cwd`.
    pub process_cwd: PathBuf,
    pub spawner: &'a dyn Spawn,
    /// When `main` started; every budget counts from here.
    pub start: Instant,
}

/// Runs the hook and writes its output. Always returns exit code 0.
pub fn run(
    args: &[String],
    stdin: Box<dyn Read + Send>,
    out: &mut dyn Write,
    env: &HookEnv<'_>,
) -> i32 {
    if let Some(text) = output(args, stdin, env) {
        // UTF-8 with no byte-order mark: Rust strings are UTF-8 and nothing
        // is written before them.
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }
    0
}

/// The hook's stdout for these arguments and this stdin, `None` for none.
pub fn output(args: &[String], stdin: Box<dyn Read + Send>, env: &HookEnv<'_>) -> Option<String> {
    let (event, client_flag, profile_flag) = match hook_input::parse_hook_args(args) {
        HookArgs::Usage => return Some(hook_input::USAGE.to_owned()),
        // Version 2 printed nothing and did not read stdin.
        HookArgs::Unknown => return None,
        HookArgs::Event {
            event,
            client_flag,
            profile,
        } => (event, client_flag, profile),
    };
    let deadline = env.start + budget(event);
    let read_by = if event.is_capture() {
        deadline.checked_sub(SPOOL_RESERVE).unwrap_or(deadline)
    } else {
        deadline.checked_sub(PRINT_RESERVE).unwrap_or(deadline)
    };
    let raw = read_stdin(stdin, read_by.saturating_duration_since(Instant::now()))?;
    // Invalid JSON: version 2 logged it and printed nothing.
    let parsed = hook_input::parse_stdin(&raw).ok()?;
    let client = hook_input::detect_client(&parsed, &client_flag);
    let mut data = hook_input::normalize_stdin(parsed, &client);
    hook_input::validate_common(&mut data, cfg!(windows));
    let call = HookCall {
        event,
        client,
        profile_flag,
        env,
        deadline,
    };
    match event {
        HookEvent::SessionStart => call.session_start(&data),
        HookEvent::UserPromptSubmit => call.prompt(data),
        HookEvent::PostToolUse | HookEvent::PreCompact | HookEvent::Stop => call.capture(data),
    }
}

/// The whole time an event may take, from the start of `main`.
pub fn budget(event: HookEvent) -> Duration {
    if event.is_capture() {
        Duration::from_millis(ambient::CAPTURE_HOOK_BUDGET_MS)
    } else {
        Duration::from_millis(ambient::DEADLINE_MS)
    }
}

/// Reads at most [`hook_input::STDIN_LIMIT_BYTES`] bytes on a helper thread,
/// so a stdin that is never closed cannot hold the hook past `limit`.
fn read_stdin(stdin: Box<dyn Read + Send>, limit: Duration) -> Option<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("sm-hook-stdin".into())
        .spawn(move || {
            let mut buf = Vec::new();
            let read = stdin
                .take(hook_input::STDIN_LIMIT_BYTES as u64)
                .read_to_end(&mut buf)
                .map(|_| buf);
            let _ = tx.send(read);
        })
        .ok()?;
    rx.recv_timeout(limit).ok()?.ok()
}

/// The session identifier of a hook payload, if it carries one.
fn session_of(data: &Map<String, Value>) -> Option<String> {
    data.get("session_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// What a capture hook prints once stdin has parsed, whatever happens to the
/// capture: the continue line for `post-tool-use` and `stop`, nothing for
/// `pre-compact` (AT-14; version 2 printed it even when its pipeline
/// failed).
pub fn capture_stdout(event: HookEvent) -> Option<String> {
    match event {
        HookEvent::PostToolUse | HookEvent::Stop => Some(ambient::CONTINUE_JSON.to_owned()),
        _ => None,
    }
}

struct HookCall<'a> {
    event: HookEvent,
    client: String,
    profile_flag: Option<String>,
    env: &'a HookEnv<'a>,
    deadline: Instant,
}

impl HookCall<'_> {
    /// The working folder sent to the daemon and used for profile rules: the
    /// payload's checked `cwd`, else the process's.
    fn cwd(&self, data: &Map<String, Value>) -> String {
        data.get("cwd")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map_or_else(
                || self.env.process_cwd.to_string_lossy().into_owned(),
                str::to_owned,
            )
    }

    /// The request envelope fields every hook request carries. An unknown
    /// `--client` value (version 2 accepted any) is sent as no client.
    fn request(
        &self,
        data: &Map<String, Value>,
        cwd: String,
        profile: &config::ProfileName,
        call: Op,
    ) -> Request {
        self.request_for(session_of(data), cwd, profile, call)
    }

    fn request_for(
        &self,
        session: Option<String>,
        cwd: String,
        profile: &config::ProfileName,
        call: Op,
    ) -> Request {
        let request = Request::new(cwd, profile.clone(), call).with_session(session);
        match Client::from_str(&self.client) {
            Ok(c) => request.with_client(c),
            Err(_) => request,
        }
    }

    fn session_start(&self, data: &Map<String, Value>) -> Option<String> {
        // Cursor's sessionStart output is dropped (B-24); no daemon contact.
        if self.client == hook_input::CURSOR {
            return None;
        }
        ambient::session_start_output(data.get("source").and_then(Value::as_str))
    }

    // -----------------------------------------------------------------------
    // Capture: post-tool-use, pre-compact, stop
    // -----------------------------------------------------------------------

    fn capture(&self, data: Map<String, Value>) -> Option<String> {
        let printed = capture_stdout(self.event);
        if self.event == HookEvent::Stop
            && data
                .get("stop_hook_active")
                .is_some_and(hook_input::py_truthy)
        {
            // The stop-hook loop guard: Claude Code is already continuing
            // because of a stop hook, so nothing is captured.
            return printed;
        }
        if data.is_empty() {
            return printed;
        }
        let cwd = self.cwd(&data);
        let Ok(resolved) = profile::resolve(
            self.profile_flag.as_deref(),
            self.env.get_env,
            Path::new(&cwd),
        ) else {
            // No profile, no store to write to: capture nothing rather than
            // guess one (SPEC 9: never land memories in the wrong store).
            return printed;
        };
        let paths = resolved.paths;
        let request = self.request_for(
            session_of(&data),
            cwd.clone(),
            &paths.profile,
            Op::Capture(CaptureParams {
                event: self.event,
                payload: Value::Object(data),
                received_at: Timestamp::now(),
            }),
        );
        let endpoint = paths.endpoint();
        let send_by = self
            .deadline
            .checked_sub(SPOOL_RESERVE)
            .unwrap_or(self.deadline);
        let left = send_by.saturating_duration_since(Instant::now());
        let outcome = if left.is_zero() {
            Err(ExchangeError::Timeout { connected: false })
        } else {
            client::exchange(&endpoint, &request, Some(left))
        };
        // `accepted: false` (capture off for this profile or project) is an
        // answer too: the daemon has the request and chose to drop it.
        if let Ok(response) = &outcome
            && let Outcome::Ok(reply) = &response.result
            && matches!(**reply, Reply::Captured(_))
        {
            return printed;
        }
        // The daemon did not accept the capture in time, refused it, or
        // cannot be reached: the same request goes to the spool, and the
        // hook never sends it again.
        //
        // A capture spooled after 200 ms may still reach the daemon from the
        // abandoned connection: the exchange thread can finish writing after
        // the hook has given up on the reply. The daemon then sees the same
        // capture twice, once live and once from the spool; its content-hash
        // dedupe drops the second (SPEC 16, capture pipeline step 4).
        let _ = spool::write_spool(&paths.spool_tmp(), &paths.spool_new(), &request);
        match outcome {
            Err(ExchangeError::Absent) => {
                // Start a daemon for next time; do not wait for it.
                let _ = client::try_start(&paths, self.env.spawner);
            }
            Err(ExchangeError::Mismatch) => {
                let budget = self.deadline.saturating_duration_since(Instant::now());
                if client::retire_other_build(&endpoint, &cwd, &paths.profile, budget) {
                    let _ = client::try_start(&paths, self.env.spawner);
                }
            }
            _ => {}
        }
        printed
    }

    // -----------------------------------------------------------------------
    // The prompt hook: user-prompt-submit
    // -----------------------------------------------------------------------

    fn prompt(&self, data: Map<String, Value>) -> Option<String> {
        let prompt = data
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let cwd = self.cwd(&data);
        let resolved = profile::resolve(
            self.profile_flag.as_deref(),
            self.env.get_env,
            Path::new(&cwd),
        )
        .ok()?;
        let paths = resolved.paths.clone();
        let settings = resolved.settings();
        let include_prompts = settings.as_ref().is_ok_and(|s| s.journal.include_prompts);
        let mut result = PromptResult::gate(HookGate::Error);
        let mut request_id = sm_core::protocol::new_request_id();
        if let Ok(settings) = &settings {
            let switches = config::resolve_ambient(&settings.ambient, self.env.get_env);
            if !switches.enabled {
                // AT-17: switched off means no connection, no lock, no spawn.
                result = PromptResult::gate(HookGate::Disabled);
            } else if !ambient::prompt_passes(&prompt) {
                result = PromptResult::gate(HookGate::ShortPrompt);
            } else {
                let request =
                    self.ambient_request(&data, cwd.clone(), &paths.profile, &prompt, switches);
                request_id.clone_from(&request.id);
                result = self.ask(&paths, &cwd, &request);
            }
        }
        let stdout = match (&result.gate, &result.text) {
            (HookGate::Injected, Some(text)) if !text.is_empty() => {
                Some(ambient::prompt_hook_stdout(text))
            }
            _ => None,
        };
        // The journal makes a missing profile folder with the default mode
        // (when the hook never started a daemon, as with ambient off), so
        // it is made user-only first (B-117). Windows is unchanged.
        #[cfg(unix)]
        let _ = client::create_profile_dir(&paths.dir);
        let journal = Journal::new(paths.metrics_dir(), include_prompts);
        self.journal(&journal, &request_id, &result, &prompt);
        stdout
    }

    fn ambient_request(
        &self,
        data: &Map<String, Value>,
        cwd: String,
        profile: &config::ProfileName,
        prompt: &str,
        switches: AmbientSwitches,
    ) -> Request {
        let left = self
            .deadline
            .checked_sub(PRINT_RESERVE)
            .unwrap_or(self.deadline)
            .saturating_duration_since(Instant::now());
        let millis = u64::try_from(left.as_millis()).unwrap_or(u64::MAX);
        self.request(
            data,
            cwd,
            profile,
            Op::AmbientRecall(AmbientParams {
                prompt: prompt.to_owned(),
                top_k: switches.top_k,
                scope: switches.scope,
            }),
        )
        .with_deadline_ms(millis)
    }

    /// Sends `ambient_recall` and maps every outcome to a gate. The hook
    /// starts a missing daemon and retires one of another build, but never
    /// waits for the new one: this prompt goes without recall.
    fn ask(
        &self,
        paths: &sm_core::paths::ProfilePaths,
        cwd: &str,
        request: &Request,
    ) -> PromptResult {
        let left = Duration::from_millis(request.deadline_ms.unwrap_or(0));
        if left.is_zero() {
            return PromptResult::gate(HookGate::Timeout);
        }
        let endpoint = paths.endpoint();
        match client::exchange(&endpoint, request, Some(left)) {
            Ok(response) => match response.into_result() {
                Ok(Reply::Ambient(reply)) => PromptResult::from_reply(reply),
                Ok(_) => PromptResult::gate(HookGate::Error),
                Err(e) if e.code == ErrorCode::DaemonLoading => {
                    PromptResult::gate(HookGate::Loading)
                }
                Err(e) if e.code == ErrorCode::Timeout => PromptResult::gate(HookGate::Timeout),
                Err(_) => PromptResult::gate(HookGate::Error),
            },
            Err(ExchangeError::Absent) => {
                let _ = client::try_start(paths, self.env.spawner);
                PromptResult::gate(HookGate::Error)
            }
            Err(ExchangeError::Mismatch) => {
                let budget = self
                    .deadline
                    .checked_sub(PRINT_RESERVE)
                    .unwrap_or(self.deadline)
                    .saturating_duration_since(Instant::now());
                if client::retire_other_build(&endpoint, cwd, &paths.profile, budget) {
                    let _ = client::try_start(paths, self.env.spawner);
                }
                PromptResult::gate(HookGate::Error)
            }
            Err(ExchangeError::Timeout { .. }) => PromptResult::gate(HookGate::Timeout),
            // A refused endpoint (another user's socket) is never used and
            // never replaced by a spawn.
            Err(ExchangeError::Io(_) | ExchangeError::Refused(_)) => {
                PromptResult::gate(HookGate::Error)
            }
        }
    }

    /// `hook_gate_total`, `hook_top_score` and `hook_latency_ms` (B-34).
    fn journal(&self, journal: &Journal, request_id: &str, result: &PromptResult, prompt: &str) {
        let latency_ms = self.env.start.elapsed().as_secs_f64() * 1000.0;
        let mut labels = BTreeMap::new();
        labels.insert("gate".to_owned(), Value::from(result.gate.as_str()));
        labels.insert("client".to_owned(), Value::from(self.client.as_str()));
        if let Some(project) = &result.project {
            labels.insert("project".to_owned(), Value::from(project.as_str()));
        }
        if !result.injected.is_empty() {
            labels.insert("injected".to_owned(), Value::from(result.injected.clone()));
        }
        if !result.top_scores.is_empty() {
            labels.insert(
                "top_scores".to_owned(),
                Value::from(
                    result
                        .top_scores
                        .iter()
                        .map(|s| f64::from(*s))
                        .collect::<Vec<_>>(),
                ),
            );
        }
        if journal.include_prompts() && !prompt.is_empty() {
            labels.insert("prompt".to_owned(), Value::from(prompt));
        }
        journal.record(request_id, metrics::HOOK_GATE_TOTAL, 1.0, labels.clone());
        if let Some(top) = result.top_scores.first() {
            let mut l = BTreeMap::new();
            l.insert("gate".to_owned(), Value::from(result.gate.as_str()));
            if let Some(project) = &result.project {
                l.insert("project".to_owned(), Value::from(project.as_str()));
            }
            journal.record(request_id, metrics::HOOK_TOP_SCORE, f64::from(*top), l);
        }
        let mut l = BTreeMap::new();
        l.insert("gate".to_owned(), Value::from(result.gate.as_str()));
        journal.record(request_id, metrics::HOOK_LATENCY_MS, latency_ms, l);
    }
}

/// The prompt hook's outcome, for its output and its journal lines.
#[derive(Debug, Clone)]
struct PromptResult {
    gate: HookGate,
    text: Option<String>,
    injected: Vec<String>,
    top_scores: Vec<f32>,
    project: Option<String>,
}

impl PromptResult {
    fn gate(gate: HookGate) -> Self {
        PromptResult {
            gate,
            text: None,
            injected: Vec::new(),
            top_scores: Vec::new(),
            project: None,
        }
    }

    fn from_reply(reply: AmbientReply) -> Self {
        PromptResult {
            gate: reply.gate,
            text: reply.text,
            injected: reply.injected.iter().map(ToString::to_string).collect(),
            top_scores: reply.top_scores,
            project: reply.project,
        }
    }
}
