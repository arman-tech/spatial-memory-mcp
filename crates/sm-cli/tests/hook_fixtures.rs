//! Every hook fixture against the mock daemon (SPEC 16; stream D, "all five
//! hook events"): each payload in `fixtures/capture/hook_payloads/` and each
//! applicable case in `event_aliases.json`, run through the real binary
//! (`spatial-memory hook ...`) for Claude Code and for Cursor.
//!
//! Each case runs twice:
//!
//! - **answer**: a mock daemon answers. The printed output must equal the
//!   fixture's, and the mock must have received exactly the request version
//!   3 sends for it: one `capture` for a capture event with a non-empty
//!   payload (filtering is the daemon's job now), one `ambient_recall` with
//!   the prompt and `top_k` for a prompt that recalls, and no connection
//!   at all for everything else.
//! - **absent**: no daemon. The output is the same except that a prompt
//!   that would have recalled prints nothing, and a capture that would have
//!   been sent is in `spool/new/` as a readable capture request.
//!
//! Where version 3 differs from version 2 on purpose the expectation is
//! overridden here and the reason is given (B-24: Cursor's `sessionStart`
//! output is dropped).

mod common;

use std::sync::Mutex;

use serde_json::Value;
use sm_core::error::CoreError;
use sm_core::journal::HookGate;
use sm_core::protocol::{HookEvent, Op, Request};
use sm_core::types::Client;

use common::*;

/// The process-heavy tests in this binary run one at a time so that the
/// 200 ms capture budget is measured on an unloaded machine.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Version 3 drops Cursor's `sessionStart` output (B-24).
const B24_CURSOR_SESSION_START: &[&str] = &[
    "cursor_sessionStart_detected_by_version",
    "cursor_sessionStart_synthesized_source",
    "contract_cursor_SessionStart_startup",
];

fn event_of(name: &str) -> Option<HookEvent> {
    match name {
        "PostToolUse" => Some(HookEvent::PostToolUse),
        "PreCompact" => Some(HookEvent::PreCompact),
        "Stop" => Some(HookEvent::Stop),
        "SessionStart" => Some(HookEvent::SessionStart),
        "UserPromptSubmit" => Some(HookEvent::UserPromptSubmit),
        _ => None,
    }
}

/// Python truthiness of a JSON value, as version 2's loop guard read it.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// What version 3 must do with one case.
#[derive(Debug, Clone)]
struct Expect {
    id: String,
    argv: Vec<String>,
    env: Vec<(String, String)>,
    stdin: Vec<u8>,
    /// The output with a daemon that answers.
    stdout: String,
    /// The capture sent, with its client, if any.
    capture: Option<(HookEvent, Option<Client>)>,
    /// The `ambient_recall` sent, if any: prompt and `top_k`.
    recall: Option<(String, u32)>,
    /// The mock's answer to `ambient_recall`.
    ambient: Result<sm_core::protocol::AmbientReply, CoreError>,
}

/// Builds the expectation for a fixture case. `contact` is false for cases
/// that never reach the daemon whatever their payload (unknown events,
/// usage).
fn expect(case: &Value, tmp: &str, contact: bool) -> Expect {
    let id = case["id"].as_str().unwrap().to_owned();
    let argv = case_argv(case);
    let env = case_env(case, tmp);
    let stdin = stdin_bytes(case, tmp);
    let mut stdout = case["expected"]["stdout"].as_str().unwrap().to_owned();
    if B24_CURSOR_SESSION_START.contains(&id.as_str()) {
        stdout.clear();
    }
    let event = event_of(case["event"].as_str().unwrap_or_default());
    let client = case["client"]
        .as_str()
        .and_then(|c| c.parse::<Client>().ok());
    let json = case
        .get("stdin_json")
        .map(|v| expand_tmp(v, tmp))
        .and_then(|v| v.as_object().cloned());
    let mut capture = None;
    let mut recall = None;
    let mut ambient = Ok(empty_ambient());
    if contact && let Some(event) = event {
        let payload = json.clone().unwrap_or_default();
        if event.is_capture() && !payload.is_empty() {
            let guard =
                event == HookEvent::Stop && payload.get("stop_hook_active").is_some_and(truthy);
            if !guard {
                capture = Some((event, client));
            }
        }
        if event == HookEvent::UserPromptSubmit
            && let Some(fake) = case.get("fake_recall")
        {
            let prompt = payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let top_k = env
                .iter()
                .find(|(k, _)| k == "SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K")
                .map_or(5, |(_, v)| v.trim().parse().unwrap());
            recall = Some((prompt, top_k));
            ambient = if let Some(msg) = fake.get("message").and_then(Value::as_str) {
                Err(CoreError::internal(msg))
            } else if fake["memories"].as_array().is_some_and(|m| !m.is_empty()) {
                let text = stdout.strip_suffix('\n').unwrap_or(&stdout).to_owned();
                Ok(ambient_with(HookGate::Injected, Some(text.as_str())))
            } else {
                Ok(empty_ambient())
            };
        }
    }
    Expect {
        id,
        argv,
        env,
        stdin,
        stdout,
        capture,
        recall,
        ambient,
    }
}

/// Runs one case with a daemon that answers; returns the failures.
fn check_answer(sandbox: &Sandbox, case: &Value, contact: bool) -> Vec<String> {
    let tmp = sandbox.tmp_token();
    create_files(case, &tmp);
    let e = expect(case, &tmp, contact);
    let mock = MockDaemon::start(&sandbox.paths(), Mode::Answer);
    mock.set_ambient(e.ambient.clone());
    let argv: Vec<&str> = e.argv.iter().map(String::as_str).collect();
    let run = sandbox.hook(&argv, &e.env, &e.stdin);
    mock.stop();
    let mut failures = Vec::new();
    let id = &e.id;
    if run.code != Some(0) {
        failures.push(format!("{id} [answer]: exit code {:?}", run.code));
    }
    if run.stdout != e.stdout {
        failures.push(format!(
            "{id} [answer]: stdout {:?}, expected {:?}",
            run.stdout, e.stdout
        ));
    }
    let requests = mock.requests();
    match (&e.capture, &e.recall) {
        (Some((event, client)), _) => {
            let captures: Vec<&Request> = requests
                .iter()
                .filter(|r| matches!(r.call, Op::Capture(_)))
                .collect();
            if requests.len() != 1 || captures.len() != 1 {
                failures.push(format!(
                    "{id} [answer]: expected one capture, got {:?}",
                    requests.iter().map(|r| r.call.name()).collect::<Vec<_>>()
                ));
            } else {
                let r = captures[0];
                let Op::Capture(p) = &r.call else {
                    unreachable!()
                };
                if p.event != *event {
                    failures.push(format!("{id} [answer]: capture event {:?}", p.event));
                }
                if r.client != *client {
                    failures.push(format!("{id} [answer]: client {:?}", r.client));
                }
                if p.payload.as_object().is_none_or(|m| m.is_empty()) {
                    failures.push(format!("{id} [answer]: empty capture payload"));
                }
                if r.deadline_ms.is_some() {
                    failures.push(format!("{id} [answer]: capture carries a deadline"));
                }
            }
            if !sandbox.spooled().is_empty() {
                failures.push(format!("{id} [answer]: an accepted capture was spooled"));
            }
        }
        (None, Some((prompt, top_k))) => {
            if requests.len() != 1 {
                failures.push(format!(
                    "{id} [answer]: expected one ambient_recall, got {}",
                    requests.len()
                ));
            } else if let Op::AmbientRecall(p) = &requests[0].call {
                if p.prompt != *prompt || p.top_k != *top_k {
                    failures.push(format!(
                        "{id} [answer]: ambient_recall prompt {:?} top_k {}, expected {:?} {}",
                        p.prompt, p.top_k, prompt, top_k
                    ));
                }
                if requests[0].deadline_ms.is_none_or(|d| d > 1500) {
                    failures.push(format!(
                        "{id} [answer]: deadline_ms {:?}",
                        requests[0].deadline_ms
                    ));
                }
            } else {
                failures.push(format!(
                    "{id} [answer]: sent {} instead of ambient_recall",
                    requests[0].call.name()
                ));
            }
        }
        (None, None) => {
            if mock.connections() != 0 {
                failures.push(format!(
                    "{id} [answer]: expected no connection, got {} ({:?})",
                    mock.connections(),
                    requests.iter().map(|r| r.call.name()).collect::<Vec<_>>()
                ));
            }
        }
    }
    failures
}

/// Runs one case with no daemon; returns the failures.
fn check_absent(sandbox: &Sandbox, case: &Value, contact: bool) -> Vec<String> {
    let tmp = sandbox.tmp_token();
    create_files(case, &tmp);
    let e = expect(case, &tmp, contact);
    let argv: Vec<&str> = e.argv.iter().map(String::as_str).collect();
    let run = sandbox.hook(&argv, &e.env, &e.stdin);
    let mut failures = Vec::new();
    let id = &e.id;
    let expected = if e.recall.is_some() {
        String::new()
    } else {
        e.stdout.clone()
    };
    if run.code != Some(0) {
        failures.push(format!("{id} [absent]: exit code {:?}", run.code));
    }
    if run.stdout != expected {
        failures.push(format!(
            "{id} [absent]: stdout {:?}, expected {:?}",
            run.stdout, expected
        ));
    }
    let spooled = sandbox.spooled();
    match &e.capture {
        Some((event, client)) => {
            if spooled.len() != 1 {
                failures.push(format!(
                    "{id} [absent]: {} spool files, expected 1",
                    spooled.len()
                ));
            } else {
                match sm_core::spool::read_spool(&spooled[0]) {
                    Ok(r) => match &r.call {
                        Op::Capture(p) if p.event == *event && r.client == *client => {}
                        other => failures.push(format!(
                            "{id} [absent]: spooled {} {:?}",
                            other.name(),
                            r.client
                        )),
                    },
                    Err(err) => failures.push(format!("{id} [absent]: unreadable spool: {err}")),
                }
            }
        }
        None => {
            if !spooled.is_empty() {
                failures.push(format!("{id} [absent]: spooled {} files", spooled.len()));
            }
        }
    }
    failures
}

fn report(failures: Vec<String>, ran: usize) {
    assert!(
        failures.is_empty(),
        "{} of {ran} checks failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn s16_every_hook_payload_against_an_answering_daemon() {
    let _g = serial();
    let cases = hook_payloads();
    assert_eq!(cases.len(), 75, "hook_payloads/ holds 75 cases");
    let mut failures = Vec::new();
    let (mut captures, mut recalls) = (0, 0);
    for case in &cases {
        let sandbox = Sandbox::new();
        let e = expect(case, &sandbox.tmp_token(), true);
        captures += usize::from(e.capture.is_some());
        recalls += usize::from(e.recall.is_some());
        failures.extend(check_answer(&sandbox, case, true));
    }
    // Every capture payload that is a non-empty object, outside the stop
    // loop guard, is sent; six prompts recall.
    assert_eq!((captures, recalls), (42, 6));
    report(failures, cases.len());
}

#[test]
fn at15_every_hook_payload_with_no_daemon() {
    let _g = serial();
    let cases = hook_payloads();
    let mut failures = Vec::new();
    for case in &cases {
        let sandbox = Sandbox::new();
        failures.extend(check_absent(&sandbox, case, true));
    }
    report(failures, cases.len());
}

/// `stdout_contract`: every event for both clients.
#[test]
fn s16_stdout_contract_for_both_clients() {
    let _g = serial();
    let aliases = event_aliases();
    let cases = aliases["stdout_contract"].as_array().unwrap();
    assert_eq!(cases.len(), 14);
    let mut failures = Vec::new();
    for case in cases {
        failures.extend(check_answer(&Sandbox::new(), case, true));
        failures.extend(check_absent(&Sandbox::new(), case, true));
    }
    report(failures, cases.len() * 2);
}

/// `cli_cases`: argument handling, usage, encodings, oversized input. None
/// of them reaches the daemon. Version 2's usage text named
/// `dispatcher.py`; version 3 prints the `cli_v2` text the fixture records
/// for `spatial-memory hook`.
#[test]
fn s16_cli_cases() {
    let _g = serial();
    let aliases = event_aliases();
    let usage = aliases["usage"]["cli_v2"].as_str().unwrap();
    assert_eq!(usage, sm_cli::hook_input::USAGE);
    let cases = aliases["cli_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 17);
    let mut failures = Vec::new();
    for case in cases {
        let mut case = case.clone();
        let stdout = case["expected"]["stdout"].as_str().unwrap().to_owned();
        if stdout.starts_with("Usage: dispatcher.py") {
            case["expected"]["stdout"] = Value::from(usage);
        }
        failures.extend(check_answer(&Sandbox::new(), &case, false));
        failures.extend(check_absent(&Sandbox::new(), &case, false));
    }
    report(failures, cases.len() * 2);
}

/// `dispatch_cases`: version 2's `dispatch(event, client, data)`, run as
/// the hook with `data` on stdin. Version 2's `dispatch` returned `None`
/// for a skipped tool and its `main` still printed the continue line; the
/// whole process is compared here, and version 3 sends the capture (the
/// daemon filters).
#[test]
fn s16_dispatch_cases() {
    let _g = serial();
    let aliases = event_aliases();
    let cases = aliases["dispatch_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    let start = sm_core::ambient::session_start_output(Some("startup")).unwrap();
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let event = case["event"].as_str().unwrap();
        let client = case["client"].as_str().unwrap();
        let stdout = match name {
            "session_start_startup" => start.clone(),
            "post_tool_use_skip_tool" | "stop_loop_guard" => {
                sm_core::ambient::CONTINUE_JSON.to_owned()
            }
            _ => String::new(),
        };
        let fixture = serde_json::json!({
            "id": format!("dispatch_{name}"),
            "event": event,
            "client": client,
            "argv": [event, "--client", client],
            "env": case["env"],
            "stdin_json": case["data"],
            "expected": {"stdout": stdout},
        });
        let contact = event_of(event).is_some();
        failures.extend(check_answer(&Sandbox::new(), &fixture, contact));
        failures.extend(check_absent(&Sandbox::new(), &fixture, contact));
    }
    report(failures, cases.len() * 2);
}

/// `session_start_handler_cases`: only `startup` and `resume` print, and
/// what they print is the fixture's object.
#[test]
fn s16_session_start_handler_cases() {
    let _g = serial();
    let aliases = event_aliases();
    let cases = aliases["session_start_handler_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let sandbox = Sandbox::new();
        let mock = MockDaemon::start(&sandbox.paths(), Mode::Answer);
        let stdin = serde_json::to_vec(&case["data"]).unwrap();
        let run = sandbox.hook(&["session-start", "--client", "claude-code"], &[], &stdin);
        mock.stop();
        let ok = match &case["expected"] {
            Value::Null => run.stdout.is_empty(),
            expected => {
                run.stdout.ends_with('\n')
                    && serde_json::from_str::<Value>(&run.stdout).ok().as_ref() == Some(expected)
            }
        };
        if !ok || run.code != Some(0) || mock.connections() != 0 {
            failures.push(format!(
                "case {i} {}: stdout {:?} code {:?} connections {}",
                case["data"],
                run.stdout,
                run.code,
                mock.connections()
            ));
        }
    }
    report(failures, cases.len());
}
