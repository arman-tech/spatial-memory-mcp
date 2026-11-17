//! The hooks end to end (stream K, part 1): `spatial-memory hook` runs as
//! a separate process against the real daemon with the fake model, and
//! every check reads what the daemon stored, journaled or embedded.
//!
//! - AT-3: a prompt hook whose daemon is killed while it waits, and one
//!   whose daemon died before it connected, exit 0 with nothing printed
//!   inside the 1,500 ms deadline. The daemon runs in a child process (this
//!   binary run again on `at3_child_daemon_serves_only_when_asked`) so it
//!   can be killed.
//! - AT-13: fixture payloads for Edit, Write and Bash store strong text as
//!   active captured memories, store nothing for weak text, and redact the
//!   fake credential.
//! - AT-14: PreCompact and Stop store only what PostToolUse missed; the
//!   loop guard stores nothing; PreCompact prints nothing.
//! - AT-15: with no daemon each capture hook exits 0 inside 200 ms with its
//!   payload spooled; a daemon started afterwards stores the same as a live
//!   one and empties the spool.
//! - AT-17: ambient recall off makes no contact (seen from the daemon's
//!   side); the printed block is version 2's format plus ids and footer; a
//!   capture over `capture.max_index_tokens` is stored whole, indexed to
//!   the cap and journaled as a cap hit.

mod common;
#[path = "common/e2e.rs"]
mod e2e;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::{
    MockDaemon, Mode, Run, Sandbox, case_argv, create_files, env_of, expand_tmp, load_json,
    payload, shut_down_daemon, stats_ms, take_daemon_lock,
};
use e2e::{ChildDaemon, Direct, Model, SLOW_PROBE, WAIT, contents, journal, unique, v3_markers};
use serde_json::{Value, json};
use sm_core::ambient::{self, Candidate};
use sm_core::config::{AMBIENT_ENV, AMBIENT_TOP_K_ENV, Settings};
use sm_core::error::CoreError;
use sm_core::ids::MemoryId;
use sm_core::journal::{HookGate, metrics};
use sm_core::protocol::{HookEvent, Op, Request};
use sm_core::testing::{SimpleChunker, dot, fake_vector};
use sm_core::traits::{ChunkParams, Chunker};
use sm_core::types::Client;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const PROMPT_LIMIT: Duration = Duration::from_millis(ambient::DEADLINE_MS);
const CAPTURE_LIMIT: Duration = Duration::from_millis(ambient::CAPTURE_HOOK_BUDGET_MS);
const PROMPT_ARGV: &[&str] = &["user-prompt-submit", "--client", "claude-code"];
const POST_ARGV: &[&str] = &["post-tool-use", "--client", "claude-code"];
const PRE_COMPACT_ARGV: &[&str] = &["pre-compact", "--client", "claude-code"];
const STOP_ARGV: &[&str] = &["stop", "--client", "claude-code"];

/// The capture payloads, one per event and client.
const CAPTURES: &[(HookEvent, Client, &str)] = &[
    (
        HookEvent::PostToolUse,
        Client::ClaudeCode,
        "cc_post_tool_use_edit_decision",
    ),
    (
        HookEvent::PreCompact,
        Client::ClaudeCode,
        "cc_pre_compact_runs_pipeline",
    ),
    (HookEvent::Stop, Client::ClaudeCode, "cc_stop_runs_pipeline"),
    (
        HookEvent::PostToolUse,
        Client::Cursor,
        "cursor_postToolUse_edit_decision",
    ),
    (
        HookEvent::PreCompact,
        Client::Cursor,
        "cursor_preCompact_status_trigger",
    ),
    (
        HookEvent::Stop,
        Client::Cursor,
        "cursor_stop_status_trigger",
    ),
];

fn expected_capture_stdout(event: HookEvent) -> &'static str {
    match event {
        HookEvent::PreCompact => "",
        _ => ambient::CONTINUE_JSON,
    }
}

fn report(label: &str, times: &[Duration]) -> f64 {
    let (median, p95, worst) = stats_ms(times);
    eprintln!(
        "{label}: n={} median {median:.1} ms, p95 {p95:.1} ms, worst {worst:.1} ms",
        times.len()
    );
    worst
}

/// A folder under the sandbox's `{TMP}`, made now.
fn folder(sb: &Sandbox, name: &str) -> PathBuf {
    let dir = sb.tmp.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn slashes(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Runs a fixture payload with `{TMP}` set to `dir`, its files made there,
/// and its session (or Cursor conversation) set to `session`. A Claude Code
/// payload without `cwd` gets `dir`, so the capture lands in that folder's
/// project. The fixture's version 2 variables are not passed: version 3
/// reads neither.
fn fixture_run(
    sb: &Sandbox,
    name: &str,
    dir: &Path,
    session: &str,
    env: &[(String, String)],
) -> Run {
    let c = payload(name);
    let tmp = slashes(dir);
    create_files(&c, &tmp);
    let mut stdin = expand_tmp(&c["stdin_json"], &tmp);
    if c["client"] == "cursor" {
        stdin["conversation_id"] = json!(session);
    } else {
        stdin["session_id"] = json!(session);
        if stdin.get("cwd").is_none() {
            stdin["cwd"] = json!(tmp);
        }
    }
    let argv = case_argv(&c);
    let args: Vec<&str> = argv.iter().map(String::as_str).collect();
    sb.hook(&args, env, &serde_json::to_vec(&stdin).unwrap())
}

/// The name of a folder, which is its project's name.
fn name_of(dir: &Path) -> String {
    dir.file_name().unwrap().to_string_lossy().into_owned()
}

/// Journal entries for `metric` whose `project` label is `project`.
fn journal_for(sb: &Sandbox, metric: &str, project: &str) -> Vec<sm_core::journal::Entry> {
    journal(&sb.paths(), metric)
        .into_iter()
        .filter(|e| e.labels.get("project").and_then(Value::as_str) == Some(project))
        .collect()
}

// ---------------------------------------------------------------------------
// The child daemon
// ---------------------------------------------------------------------------

/// Not a test of its own: the body of the child daemon the AT-3 and AT-15
/// tests start. Without `SM_E2E_CHILD_HOME` it returns at once.
#[test]
fn at3_child_daemon_serves_only_when_asked() {
    if let Some(home) = std::env::var_os(e2e::CHILD_HOME_ENV) {
        e2e::child_main(&home.to_string_lossy());
    }
}

// ---------------------------------------------------------------------------
// AT-3: the prompt hook and a daemon that dies
// ---------------------------------------------------------------------------

const KILL_RUNS: usize = 20;

#[test]
fn at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_is_killed_mid_request() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let dir = folder(&sb, "at3killed");
    let marker = sb.tmp.join("slow.marker");
    let env = env_of(&[(AMBIENT_ENV, "on")]);
    let mut times = Vec::new();
    let mut after_kill = Vec::new();
    for i in 0..KILL_RUNS {
        let _ = std::fs::remove_file(&marker);
        let mut child = ChildDaemon::spawn(&sb, Some(&marker));
        let stdin = json!({
            "session_id": unique("s"),
            "cwd": slashes(&dir),
            "prompt": format!("{SLOW_PROBE} what does the billing deploy need"),
        })
        .to_string();
        let (run, started, killed) = std::thread::scope(|s| {
            let started = Instant::now();
            let hook = s.spawn(|| sb.hook(PROMPT_ARGV, &env, stdin.as_bytes()));
            assert!(
                e2e::wait_until(Duration::from_secs(10), || marker.exists()),
                "run {i}: the daemon began embedding the prompt"
            );
            let killed = Instant::now();
            child.kill();
            (hook.join().unwrap(), started, killed)
        });
        assert_eq!(run.code, Some(0), "run {i}: {}", run.stderr);
        assert_eq!(run.stdout, "", "run {i}");
        assert!(run.elapsed < PROMPT_LIMIT, "run {i}: {:?}", run.elapsed);
        times.push(run.elapsed);
        after_kill.push((started + run.elapsed).saturating_duration_since(killed));
    }
    assert_eq!(
        sb.gates(),
        vec![HookGate::Error.as_str().to_owned(); KILL_RUNS],
        "every run ended on the broken connection, not on its deadline"
    );
    report("AT-3 daemon killed mid-request, hook wall time", &times);
    report(
        "AT-3 daemon killed mid-request, kill to hook exit",
        &after_kill,
    );
}

#[test]
fn at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_died_before_it_connected() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let paths = sb.paths();
    let dir = folder(&sb, "at3gone");
    let env = env_of(&[(AMBIENT_ENV, "on")]);
    let runs = 5;
    let mut times = Vec::new();
    for i in 0..runs {
        let mut child = ChildDaemon::spawn(&sb, None);
        child.kill();
        let stdin = json!({
            "session_id": unique("s"),
            "cwd": slashes(&dir),
            "prompt": "what does the billing deploy need today",
        })
        .to_string();
        let run = sb.hook(PROMPT_ARGV, &env, stdin.as_bytes());
        assert_eq!(run.code, Some(0), "run {i}: {}", run.stderr);
        assert_eq!(run.stdout, "", "run {i}");
        assert!(run.elapsed < PROMPT_LIMIT, "run {i}: {:?}", run.elapsed);
        times.push(run.elapsed);
        // The hook found nothing listening and started the binary's own
        // daemon on its way out; stop it before the next run.
        assert!(
            e2e::wait_until(Duration::from_secs(15), || {
                sm_cli::client::is_listening(&paths.endpoint())
            }),
            "run {i}: the hook started a daemon"
        );
        assert!(shut_down_daemon(&paths, "test run over"), "run {i}");
    }
    assert_eq!(sb.gates(), vec![HookGate::Error.as_str().to_owned(); runs]);
    report("AT-3 daemon dead before connect, hook wall time", &times);
}

// ---------------------------------------------------------------------------
// AT-13: capture through the binary
// ---------------------------------------------------------------------------

#[test]
fn at13_capture_hooks_through_the_binary_store_strong_text_redacted_and_weak_text_not_at_all() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let handle = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let direct = Direct::new(&sb, handle.endpoint().clone());
    let cases = [
        "cc_post_tool_use_edit_decision",
        "cc_post_tool_use_write_solution",
        "cc_post_tool_use_bash_aws_key_redacted",
        "cc_post_tool_use_bash_trivial",
    ];
    let mut dirs = Vec::new();
    for name in cases {
        let dir = folder(&sb, &unique("at13"));
        let run = fixture_run(&sb, name, &dir, &unique("s"), &[]);
        assert_eq!(run.code, Some(0), "{name}: {}", run.stderr);
        assert_eq!(run.stdout, ambient::CONTINUE_JSON, "{name}");
        dirs.push((name, dir));
    }
    direct.barrier();
    let stored = direct.export();
    for (name, dir) in &dirs {
        let c = payload(name);
        let mut want: Vec<String> = c["expected"]["queue_files"]
            .as_array()
            .expect("the fixture's expected queue files")
            .iter()
            .map(|q| v3_markers(q["content"].as_str().unwrap()))
            .collect();
        want.sort();
        let got = stored.get(&name_of(dir)).cloned().unwrap_or_default();
        assert_eq!(contents(&got), want, "{name}");
        for m in &got {
            assert_eq!(m["status"], "active", "{name}: {m}");
            assert_eq!(m["provenance"]["source"], "capture", "{name}: {m}");
        }
    }
    assert!(
        !stored.contains_key(&name_of(&dirs[3].1)),
        "the trivial Bash output stored nothing"
    );
    let aws = &stored[&name_of(&dirs[2].1)][0];
    let text = aws.to_string();
    assert!(!text.contains("AKIA1234567890ABCDEF"), "{text}");
    assert!(
        aws["content"]
            .as_str()
            .unwrap()
            .contains("[REDACTED:aws_key]"),
        "{text}"
    );
    handle.shutdown("test over");
}

// ---------------------------------------------------------------------------
// AT-14: the transcript stores only what PostToolUse missed
// ---------------------------------------------------------------------------

const CAPTURED_EDIT: &str = "Decided to use PostgreSQL because it handles JSONB well.";
const CAPTURED_WRITE: &str = "The fix is to add the missing import statement.";
const MISSED: &str = "Decided to use Redis for the session cache.";

/// A folder with a copy of `rust/at14_session.jsonl`, and PostToolUse
/// captures of two of its texts in `session`, through the binary. Returns
/// the folder and the transcript path.
fn at14_session(sb: &Sandbox, prefix: &str, session: &str) -> (PathBuf, String) {
    let dir = folder(sb, &unique(prefix));
    let transcript = dir.join("at14_session.jsonl");
    std::fs::copy(
        common::fixture_dir()
            .join("rust")
            .join("at14_session.jsonl"),
        &transcript,
    )
    .expect("copy the transcript");
    for (tool, input) in [
        ("Edit", json!({"new_string": CAPTURED_EDIT})),
        ("Write", json!({"content": CAPTURED_WRITE})),
    ] {
        let stdin = json!({
            "session_id": session,
            "cwd": slashes(&dir),
            "hook_event_name": "PostToolUse",
            "tool_name": tool,
            "tool_input": input,
            "tool_response": "ok",
        });
        let run = sb.hook(POST_ARGV, &[], &serde_json::to_vec(&stdin).unwrap());
        assert_eq!(run.code, Some(0), "{tool}: {}", run.stderr);
        assert_eq!(run.stdout, ambient::CONTINUE_JSON, "{tool}");
    }
    (dir, slashes(&transcript))
}

fn the_three() -> Vec<String> {
    let mut want = vec![
        CAPTURED_EDIT.to_owned(),
        CAPTURED_WRITE.to_owned(),
        MISSED.to_owned(),
    ];
    want.sort();
    want
}

#[test]
fn at14_pre_compact_and_stop_through_the_binary_store_only_what_post_tool_use_missed() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let handle = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let direct = Direct::new(&sb, handle.endpoint().clone());

    // PreCompact.
    let session = unique("s");
    let (dir, transcript) = at14_session(&sb, "at14pre", &session);
    let stdin = json!({
        "session_id": session,
        "transcript_path": transcript,
        "cwd": slashes(&dir),
        "hook_event_name": "PreCompact",
        "trigger": "auto",
    });
    let run = sb.hook(PRE_COMPACT_ARGV, &[], &serde_json::to_vec(&stdin).unwrap());
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(run.stdout, "", "PreCompact prints nothing");
    direct.barrier();
    let stored = direct.export();
    let pre = &stored[&name_of(&dir)];
    assert_eq!(contents(pre), the_three());
    assert_eq!(
        journal_for(&sb, metrics::CAPTURE_SIZE_TOKENS, &name_of(&dir)).len(),
        3,
        "two PostToolUse captures and one from the transcript: the two \
         texts PostToolUse stored were not offered again"
    );
    let missed = pre
        .iter()
        .find(|m| m["content"] == MISSED)
        .expect("the missed text");
    assert_eq!(missed["tags"], json!(["transcript", "decision"]));
    assert_eq!(missed["provenance"]["source"], "capture");
    assert_eq!(missed["provenance"]["extra"]["trigger"], "auto", "{missed}");

    // Stop.
    let session = unique("s");
    let (dir, transcript) = at14_session(&sb, "at14stop", &session);
    let stdin = json!({
        "session_id": session,
        "transcript_path": transcript,
        "cwd": slashes(&dir),
        "hook_event_name": "Stop",
        "stop_hook_active": false,
    });
    let run = sb.hook(STOP_ARGV, &[], &serde_json::to_vec(&stdin).unwrap());
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(run.stdout, ambient::CONTINUE_JSON);
    direct.barrier();
    let stored = direct.export();
    let stop = &stored[&name_of(&dir)];
    assert_eq!(contents(stop), the_three());
    assert_eq!(
        journal_for(&sb, metrics::CAPTURE_SIZE_TOKENS, &name_of(&dir)).len(),
        3
    );
    let missed = stop
        .iter()
        .find(|m| m["content"] == MISSED)
        .expect("the missed text");
    assert_eq!(
        missed["provenance"]["extra"]["trigger"], "session_end",
        "{missed}"
    );

    // The loop guard: a Stop already continuing stores nothing.
    let session = unique("s");
    let dir = folder(&sb, &unique("at14guard"));
    let transcript = dir.join("at14_session.jsonl");
    std::fs::copy(
        common::fixture_dir()
            .join("rust")
            .join("at14_session.jsonl"),
        &transcript,
    )
    .unwrap();
    let stdin = json!({
        "session_id": session,
        "transcript_path": slashes(&transcript),
        "cwd": slashes(&dir),
        "hook_event_name": "Stop",
        "stop_hook_active": true,
    });
    let run = sb.hook(STOP_ARGV, &[], &serde_json::to_vec(&stdin).unwrap());
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(run.stdout, ambient::CONTINUE_JSON);
    direct.barrier();
    assert!(!direct.export().contains_key(&name_of(&dir)));
    assert!(journal_for(&sb, metrics::CAPTURE_SIZE_TOKENS, &name_of(&dir)).is_empty());
    handle.shutdown("test over");
}

// ---------------------------------------------------------------------------
// AT-15: the spool when no daemon runs
// ---------------------------------------------------------------------------

const SPOOL_RUNS: usize = 5;

#[test]
fn at15_capture_hooks_spool_inside_200_ms_with_no_daemon_and_a_started_daemon_stores_the_same() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let paths = sb.paths();
    // A daemon ran here and is gone.
    ChildDaemon::spawn(&sb, None).kill();
    // Held so the hooks' own start attempts find a daemon starting and
    // spawn nothing; released before the daemon below starts.
    let lock = take_daemon_lock(&paths).expect("the profile lock");
    let mut worst = Vec::new();
    for (i, (event, client, name)) in CAPTURES.iter().enumerate() {
        let dir = folder(&sb, &format!("c{i}-sp"));
        let mut times = Vec::new();
        for r in 0..SPOOL_RUNS {
            let before = sb.spooled().len();
            let run = fixture_run(&sb, name, &dir, &unique("s"), &[]);
            assert_eq!(run.code, Some(0), "{name} run {r}: {}", run.stderr);
            assert_eq!(
                run.stdout,
                expected_capture_stdout(*event),
                "{name} run {r}"
            );
            assert!(
                run.elapsed < CAPTURE_LIMIT,
                "{name} run {r}: {:?}",
                run.elapsed
            );
            assert_eq!(sb.spooled().len(), before + 1, "{name} run {r} spooled");
            times.push(run.elapsed);
        }
        let w = report(&format!("AT-15 no daemon, {event:?} {client}"), &times);
        worst.push((name, w));
    }
    drop(lock);

    let _child = ChildDaemon::spawn(&sb, None);
    let direct = Direct::new(&sb, paths.endpoint());
    assert!(
        e2e::wait_until(WAIT, || sb.spooled().is_empty()),
        "the daemon drained the spool"
    );
    direct.barrier();
    for (i, (_, _, name)) in CAPTURES.iter().enumerate() {
        let dir = folder(&sb, &format!("c{i}-lv"));
        for r in 0..SPOOL_RUNS {
            let run = fixture_run(&sb, name, &dir, &unique("s"), &[]);
            assert_eq!(run.code, Some(0), "{name} live run {r}: {}", run.stderr);
        }
    }
    direct.barrier();
    let stored = direct.export();
    for (i, (_, _, name)) in CAPTURES.iter().enumerate() {
        let spooled = contents(
            stored
                .get(&format!("c{i}-sp"))
                .map_or(&[][..], Vec::as_slice),
        );
        let live = contents(
            stored
                .get(&format!("c{i}-lv"))
                .map_or(&[][..], Vec::as_slice),
        );
        assert!(
            !spooled.is_empty(),
            "{name}: the spooled captures were stored"
        );
        assert_eq!(spooled, live, "{name}: spooled and live store the same");
    }
    let failed = std::fs::read_dir(paths.spool_failed())
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(failed, 0, "nothing in spool/failed");
    for (name, w) in worst {
        eprintln!("AT-15 worst {name}: {w:.1} ms");
    }
}

// ---------------------------------------------------------------------------
// AT-17: ambient recall
// ---------------------------------------------------------------------------

/// A second endpoint next to the profile's, for the real daemon behind a
/// relay.
fn relay_endpoint(e: &sm_core::paths::Endpoint) -> sm_core::paths::Endpoint {
    use sm_core::paths::Endpoint;
    match e {
        Endpoint::NamedPipe(name) => Endpoint::NamedPipe(format!("{name}-relay")),
        Endpoint::UnixSocket(p) => Endpoint::UnixSocket(p.with_extension("relay")),
    }
}

/// Off: the hook prints nothing, journals `disabled`, and nothing reaches
/// the daemon. The real daemon sits behind a relay on the profile's
/// endpoint, which counts every connection; the daemon's model records
/// every query it embeds. Switched on, the same prompt reaches both, so
/// the counts can see contact.
#[test]
fn at17_ambient_recall_off_makes_no_contact_with_the_daemon() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let paths = sb.paths();
    let relay = relay_endpoint(&paths.endpoint());
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let handle = e2e::start_daemon(
        &paths,
        Model::Recording(Arc::clone(&seen)),
        Some(relay.clone()),
    );
    let direct = Direct::new(&sb, relay.clone());
    let dir = folder(&sb, "at17off");
    direct.remember(
        &dir,
        "release checklist steps for the billing service deploy",
    );
    direct.wait_embedded();

    let mock = MockDaemon::start(&paths, Mode::Answer);
    let to = relay.clone();
    mock.set_responder(Arc::new(move |req: &Request| {
        if matches!(req.call, Op::Shutdown(_)) {
            return None;
        }
        Some(
            match sm_cli::client::exchange(&to, req, Some(Duration::from_secs(5))) {
                Ok(response) => response.into_result(),
                Err(e) => Err(CoreError::internal(format!("relay: {e:?}"))),
            },
        )
    }));

    let cursor = payload("cursor_userPromptSubmit_block");
    let cursor_argv = case_argv(&cursor);
    let cursor_args: Vec<&str> = cursor_argv.iter().map(String::as_str).collect();
    let runs: [(&[&str], Value); 2] = [
        (
            PROMPT_ARGV,
            json!({"session_id": unique("s"), "cwd": slashes(&dir), "prompt": "what is on the billing deploy checklist"}),
        ),
        (
            cursor_args.as_slice(),
            json!({"conversation_id": unique("c"), "workspace_roots": [slashes(&dir)], "prompt": "which steps does the billing deploy need"}),
        ),
    ];
    for (args, stdin) in &runs {
        let prompt = stdin["prompt"].as_str().unwrap();
        let bytes = serde_json::to_vec(stdin).unwrap();
        let off = sb.hook(args, &env_of(&[(AMBIENT_ENV, "off")]), &bytes);
        assert_eq!(off.code, Some(0), "{}", off.stderr);
        assert_eq!(off.stdout, "", "{args:?}");
        assert_eq!(mock.connections(), 0, "{args:?}: no connection while off");
        assert!(mock.requests().is_empty());
        assert!(
            !seen.lock().unwrap().iter().any(|q| q == prompt),
            "the daemon embedded no prompt while off"
        );
    }
    assert_eq!(
        sb.gates(),
        vec![HookGate::Disabled.as_str().to_owned(); runs.len()]
    );

    // The control: switched on, the same prompts reach the daemon.
    for (n, (args, stdin)) in runs.iter().enumerate() {
        let prompt = stdin["prompt"].as_str().unwrap();
        let bytes = serde_json::to_vec(stdin).unwrap();
        let on = sb.hook(args, &env_of(&[(AMBIENT_ENV, "on")]), &bytes);
        assert_eq!(on.code, Some(0), "{}", on.stderr);
        assert!(mock.connections() > n, "{args:?}: on connects");
        assert_eq!(
            seen.lock().unwrap().iter().filter(|q| *q == prompt).count(),
            1,
            "{args:?}: on, the daemon embedded the prompt once"
        );
    }
    mock.stop();
    handle.shutdown("test over");
}

const PROMPT: &str = "release checklist steps for the billing service deploy";

/// Version 2's renderer, from the fixture's `rules.format`: no ids, no
/// footer, no trailing newline.
fn v2_render(items: &[(f64, &str)]) -> String {
    let mut lines = vec![ambient::HEADER.to_owned()];
    for (i, (similarity, content)) in items.iter().enumerate() {
        let flat = content.replace('\n', " ");
        let flat = flat.trim();
        let line = if flat.chars().count() > 200 {
            let cut: String = flat.chars().take(197).collect();
            format!("{cut}...")
        } else {
            flat.to_owned()
        };
        lines.push(format!("  {}. ({similarity:.2}) {line}", i + 1));
    }
    lines.join("\n")
}

#[test]
fn at17_the_block_through_the_binary_is_version_2s_format_with_ids_and_the_footer() {
    let _g = serial();
    // The renderer is version 2's: check it on the fixture's case first.
    let fixture = load_json(&common::fixture_dir().join("ambient_recall_cases.json"));
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "recall_returns_formatted_block")
        .expect("the fixture case");
    let items: Vec<(f64, &str)> = case["fake_recall"]["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["similarity"].as_f64().unwrap(),
                m["content"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(v2_render(&items), case["computed_result"].as_str().unwrap());

    let sb = Sandbox::with_daemon();
    let handle = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let direct = Direct::new(&sb, handle.endpoint().clone());
    let dir = folder(&sb, "at17block");
    // Three lines: one over 200 characters with line breaks in it.
    let multi = format!("{} now", [PROMPT; 4].join("\n"));
    let texts = [
        PROMPT.to_owned(),
        format!("{PROMPT} today"),
        format!("{PROMPT} today quickly"),
        format!("{PROMPT} today quickly please"),
        "unrelated note about garden tomatoes and summer rain".to_owned(),
        multi.clone(),
    ];
    let q = fake_vector(PROMPT);
    let mut seeded: Vec<(String, String, f32)> = texts
        .iter()
        .map(|t| {
            (
                t.clone(),
                direct.remember(&dir, t),
                dot(&q, &fake_vector(t)),
            )
        })
        .collect();
    seeded.sort_by(|a, b| b.2.total_cmp(&a.2));
    for w in seeded.windows(2) {
        assert!(w[0].2 - w[1].2 > 1e-3, "distinct fake scores: {seeded:?}");
    }
    let top: Vec<&(String, String, f32)> = seeded.iter().take(3).collect();
    assert!(top.iter().any(|t| t.0 == multi), "{seeded:?}");
    assert!(multi.chars().count() > 200 && multi.contains('\n'));
    direct.wait_embedded();

    let session = unique("s");
    let stdin = json!({"session_id": session, "cwd": slashes(&dir), "prompt": PROMPT});
    let run = sb.hook(
        PROMPT_ARGV,
        &env_of(&[(AMBIENT_ENV, "on"), (AMBIENT_TOP_K_ENV, "3")]),
        &serde_json::to_vec(&stdin).unwrap(),
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);

    // What the hook journaled: the ids injected and their exact scores.
    let gates = journal(&sb.paths(), metrics::HOOK_GATE_TOTAL);
    assert_eq!(gates.len(), 1);
    let entry = &gates[0];
    assert_eq!(entry.labels["gate"], HookGate::Injected.as_str());
    let injected: Vec<String> = entry.labels["injected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let scores: Vec<f32> = entry.labels["top_scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let want_ids: Vec<String> = top.iter().map(|t| t.1.clone()).collect();
    assert_eq!(injected, want_ids, "the best three, best first");
    assert_eq!(scores.len(), 3);
    for (s, t) in scores.iter().zip(&top) {
        assert!((s - t.2).abs() < 1e-5, "{s} against the fake score {}", t.2);
    }

    let by_id: BTreeMap<&str, &str> = seeded
        .iter()
        .map(|(text, id, _)| (id.as_str(), text.as_str()))
        .collect();
    let candidates: Vec<Candidate> = injected
        .iter()
        .zip(&scores)
        .map(|(id, score)| Candidate {
            id: MemoryId::parse(id).unwrap(),
            score: *score,
            text: by_id[id.as_str()].to_owned(),
        })
        .collect();
    let refs: Vec<&Candidate> = candidates.iter().collect();
    let block = ambient::format_block(&refs).expect("a block");
    assert_eq!(run.stdout, ambient::prompt_hook_stdout(&block.text));
    assert!(run.stdout.starts_with(&format!("{}\n", ambient::HEADER)));
    assert!(run.stdout.ends_with(&format!("{}\n", ambient::FOOTER)));

    // Without the ids and the footer, it is version 2's block byte for byte.
    let mut v3 = run.stdout.clone();
    for id in &injected {
        v3 = v3.replace(&format!(" [{id}]"), "");
    }
    let v3 = v3
        .trim_end_matches('\n')
        .strip_suffix(ambient::FOOTER)
        .expect("the footer")
        .trim_end_matches('\n')
        .to_owned();
    let v2_items: Vec<(f64, &str)> = injected
        .iter()
        .zip(&scores)
        .map(|(id, s)| (f64::from(*s), by_id[id.as_str()]))
        .collect();
    assert_eq!(v3, v2_render(&v2_items));
    let long = v3
        .lines()
        .find(|l| l.ends_with("..."))
        .expect("the cut line");
    assert_eq!(long.chars().count(), "  1. (1.00) ".chars().count() + 200);
    handle.shutdown("test over");
}

#[test]
fn at17_a_capture_over_the_index_cap_is_stored_whole_indexed_to_the_cap_and_journaled() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let handle = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let direct = Direct::new(&sb, handle.endpoint().clone());
    let settings = Settings::default();
    let cap = settings.capture.max_index_tokens as usize;
    let filler: Vec<String> = (0..cap + 800).map(|i| format!("filler{i}")).collect();
    let text = format!("{CAPTURED_EDIT} {}", filler.join(" "));
    let plan = SimpleChunker
        .chunk(
            &text,
            &ChunkParams {
                max_tokens: settings.chunk.max_tokens as usize,
                overlap: settings.chunk.overlap,
                index_cap: Some(cap),
            },
        )
        .unwrap();
    assert!(plan.capped && plan.indexed_tokens == cap && plan.total_tokens > cap);

    direct.barrier();
    direct.wait_embedded();
    let chunks_before = direct.status().index.chunks;
    let dir = folder(&sb, "at17cap");
    let stdin = json!({
        "session_id": unique("s"),
        "cwd": slashes(&dir),
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"new_string": text},
        "tool_response": "ok",
    });
    let bytes = serde_json::to_vec(&stdin).unwrap();
    assert!(
        bytes.len() < sm_cli::hook_input::STDIN_LIMIT_BYTES,
        "under the hook's stdin limit"
    );
    let run = sb.hook(POST_ARGV, &[], &bytes);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(run.stdout, ambient::CONTINUE_JSON);
    direct.barrier();
    direct.wait_embedded();

    let stored = direct.export();
    let got = &stored[&name_of(&dir)];
    assert_eq!(got.len(), 1);
    assert_eq!(got[0]["content"], text.as_str(), "stored whole");
    // The barrier added one single-chunk memory.
    assert_eq!(
        direct.status().index.chunks - chunks_before,
        plan.pieces.len() as u64 + 1,
        "indexed to the cap"
    );
    let sizes = journal_for(&sb, metrics::CAPTURE_SIZE_TOKENS, &name_of(&dir));
    assert_eq!(sizes.len(), 1);
    assert_eq!(sizes[0].value, plan.total_tokens as f64);
    assert_eq!(sizes[0].labels["indexed_tokens"].as_f64(), Some(cap as f64));
    let hits = journal_for(&sb, metrics::CAPTURE_CAP_HIT_TOTAL, &name_of(&dir));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].request_id, sizes[0].request_id);
    assert_eq!(hits[0].value, 1.0);
    handle.shutdown("test over");
}
