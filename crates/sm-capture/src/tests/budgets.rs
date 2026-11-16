//! Time for the whole pipeline on large inputs: a 200,000-character Bash
//! output and a 1 MB transcript delta. SPEC section 11 sets no budget for
//! this path (the daemon runs it after it has replied `accepted`), so these
//! tests check the outcome and print the time for the report; the two
//! classifier budgets of `signal_cases.json` are asserted in `signals.rs`.
//! Run on a release build: `cargo test -p sm-capture --all-features
//! --release -- --nocapture`.

use std::time::{Duration, Instant};

use serde_json::json;
use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;

use super::{project, temp_dir};
use crate::extract::MAX_CAPTURE_CHARS;
use crate::pipeline::{CaptureInput, CaptureOutcome, TextSource, capture};
use crate::session::SessionState;

const BASH_OUTPUT_CHARS: usize = 200_000;
const TRANSCRIPT_BYTES: usize = 1 << 20;

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Runs `f` `runs` times and returns the median time and the last result.
fn median<T>(runs: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut times = Vec::with_capacity(runs);
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        let out = f();
        times.push(start.elapsed());
        last = Some(out);
    }
    times.sort();
    (times[times.len() / 2], last.expect("at least one run"))
}

fn runs() -> usize {
    if cfg!(debug_assertions) { 3 } else { 11 }
}

/// A build log of exactly `chars` characters, with one strong signal and
/// one fake credential in it, so every step of the pipeline does work.
fn bash_output(chars: usize) -> String {
    let mut out = String::with_capacity(chars + 128);
    out.push_str("The fix was to raise the pool size to 32.\n");
    out.push_str("export AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF\n");
    let mut i = 0usize;
    while out.chars().count() < chars {
        out.push_str(&format!(
            "   Compiling crate-{i:05} v0.{}.{} (/build/src/crate-{i:05}) in {}.{:02}s\n",
            i % 17,
            i % 7,
            i % 9,
            i % 100
        ));
        i += 1;
    }
    out.chars().take(chars).collect()
}

#[test]
fn s16_pipeline_time_on_a_200k_bash_output() {
    let dir = temp_dir();
    let project = project(dir.path());
    let settings = CaptureSettings::default();
    let output = bash_output(BASH_OUTPUT_CHARS);
    assert_eq!(output.chars().count(), BASH_OUTPUT_CHARS);
    let payload = json!({
        "session_id": "budget",
        "tool_name": "Bash",
        "tool_input": {"command": "cargo build --workspace"},
        "tool_response": output,
    });
    let input = CaptureInput {
        event: HookEvent::PostToolUse,
        payload: &payload,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: 0,
    };
    let (time, outcome): (Duration, CaptureOutcome) = median(runs(), || {
        let mut state = SessionState::new();
        capture(&input, &mut state)
    });
    assert_eq!(outcome.skipped, None);
    assert_eq!(outcome.source, Some(TextSource::ToolCall));
    assert!(outcome.cap_hit, "the extracted text is over the cap");
    assert_eq!(outcome.candidates.len(), 1, "{:?}", outcome.drops);
    let c = &outcome.candidates[0];
    assert!(c.content.contains("[REDACTED:aws_key]"));
    assert!(!c.content.contains("AKIA1234567890ABCDEF"));
    println!(
        "BUDGET pipeline_bash_200k_chars: {:.3} ms median of {} ({} build; extracted {} chars, \
         cut to {}; no SPEC 11 budget)",
        time.as_secs_f64() * 1000.0,
        runs(),
        profile(),
        outcome.extracted_chars,
        MAX_CAPTURE_CHARS
    );
}

/// A transcript of at least `bytes` bytes: assistant texts, each unique,
/// every fourth with a strong signal, between user turns and tool results.
fn transcript(bytes: usize) -> (String, usize) {
    let mut out = String::with_capacity(bytes + 4096);
    let mut assistant_chars = 0usize;
    let mut i = 0usize;
    while out.len() < bytes {
        let text = if i.is_multiple_of(4) {
            format!(
                "Decided to use option {i} for the cache because the benchmark favoured it. \
                 The pool held {} connections and the p99 stayed flat across the run, so the \
                 change is kept for entry {i}.",
                i % 64
            )
        } else {
            format!(
                "Looking at entry {i} now. The output lists {} files and nothing in it needs a \
                 change; moving on to the next part of the build once the check completes.",
                i % 97
            )
        };
        assistant_chars += text.chars().count();
        let assistant = json!({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "isSidechain": false,
            "timestamp": "2026-01-01T00:00:00Z",
            "uuid": format!("a{i}"),
        });
        let user = json!({
            "type": "user",
            "message": {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": format!("t{i}"),
                "content": format!("{} lines of tool output for step {i}", i * 3),
            }]},
            "uuid": format!("u{i}"),
        });
        out.push_str(&assistant.to_string());
        out.push('\n');
        out.push_str(&user.to_string());
        out.push('\n');
        i += 1;
    }
    (out, assistant_chars)
}

#[test]
fn s16_pipeline_time_on_a_1mb_transcript_delta() {
    let dir = temp_dir();
    let project = project(dir.path());
    let settings = CaptureSettings::default();
    let (text, assistant_chars) = transcript(TRANSCRIPT_BYTES);
    assert!(
        assistant_chars > MAX_CAPTURE_CHARS,
        "more assistant text than the cap"
    );
    let path = dir.path().join("session.jsonl");
    std::fs::write(&path, &text).expect("write the transcript");
    let payload = json!({
        "session_id": "budget",
        "transcript_path": path.to_str().expect("a UTF-8 path"),
        "trigger": "auto",
    });
    let input = CaptureInput {
        event: HookEvent::PreCompact,
        payload: &payload,
        settings: &settings,
        project: &project,
        commit_sha: None,
        transcript_offset: 0,
    };
    let (time, outcome) = median(runs(), || {
        let mut state = SessionState::new();
        capture(&input, &mut state)
    });
    assert_eq!(outcome.skipped, None);
    assert_eq!(outcome.source, Some(TextSource::Transcript));
    assert_eq!(outcome.new_offset, Some(text.len() as u64));
    assert!(outcome.cap_hit, "the assistant text is over the cap");
    assert!(!outcome.candidates.is_empty());
    println!(
        "BUDGET pipeline_transcript_1mb_delta: {:.3} ms median of {} ({} build; {} bytes, \
         {} assistant chars, {} candidates; no SPEC 11 budget)",
        time.as_secs_f64() * 1000.0,
        runs(),
        profile(),
        text.len(),
        assistant_chars,
        outcome.candidates.len()
    );
}
