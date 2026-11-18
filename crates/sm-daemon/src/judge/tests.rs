//! Unit tests of the judge: the output parser, the executable search, the
//! answer check and the redaction before a call (SPEC 16, The judge; SPEC
//! 9; B-168). Every text here is synthetic.

use std::path::Path;

use sm_capture::RegexSecretScanner;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::time::Timestamp;
use sm_core::types::MemoryType;
use sm_store::JudgeItem;

use super::*;

fn verdict(id: &str, keep: bool, reason: &str, ty: &str, memory: &str) -> JudgeVerdict {
    JudgeVerdict {
        id: id.to_owned(),
        keep,
        reason: reason.to_owned(),
        memory_type: ty.to_owned(),
        memory: memory.to_owned(),
    }
}

fn batch(n: usize) -> Vec<JudgeCandidate> {
    (1..=n)
        .map(|i| JudgeCandidate {
            id: format!("c{i:02}"),
            text: format!("candidate text {i}"),
            context: String::new(),
        })
        .collect()
}

const INIT: &str = r#"{"type":"system","subtype":"init","model":"claude-haiku-5-5","tools":[]}"#;

fn result_line(structured: &str) -> String {
    format!(
        r#"{{"type":"result","subtype":"success","is_error":false,"duration_ms":10500,"total_cost_usd":0.00187,"result":"","structured_output":{structured}}}"#
    )
}

#[test]
fn s16_parse_output_reads_the_verdicts_cost_and_model_of_an_answer() {
    let out = format!(
        "{INIT}\n{}\n{}\n",
        r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1900000000}}"#,
        result_line(
            r#"{"verdicts":[{"id":"c01","keep":true,"reason":"knowledge","type":"decision","memory":"The build uses one lock file."},{"id":"c02","keep":false,"reason":"narration","type":"note","memory":""}]}"#
        )
    );
    let a = parse_output(out.as_bytes(), Some(0), b"").expect("an answer");
    assert_eq!(a.verdicts.len(), 2);
    assert_eq!(
        a.verdicts[0],
        verdict(
            "c01",
            true,
            "knowledge",
            "decision",
            "The build uses one lock file."
        )
    );
    assert!(!a.verdicts[1].keep);
    assert!((a.cost_usd - 0.00187).abs() < 1e-12);
    assert_eq!(a.duration_ms, 10500);
    assert_eq!(a.model.as_deref(), Some("claude-haiku-5-5"));
}

#[test]
fn s16_parse_output_takes_a_structured_output_given_as_a_json_string() {
    let out = result_line(r#""{\"verdicts\":[]}""#);
    let a = parse_output(out.as_bytes(), Some(0), b"").expect("an answer");
    assert!(a.verdicts.is_empty());
    assert_eq!(a.model, None);
}

#[test]
fn s16_parse_output_counts_signed_out_as_unavailable() {
    let out = format!(
        "{INIT}\n{}\n",
        r#"{"type":"result","subtype":"success","is_error":true,"duration_ms":40,"total_cost_usd":0,"result":"Not logged in · Please run /login"}"#
    );
    let f = parse_output(out.as_bytes(), Some(1), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::Unavailable);
    assert!(f.message.contains("sign in"), "{}", f.message);
    assert_eq!(f.cost_usd, 0.0);
}

#[test]
fn s16_parse_output_gives_the_reset_time_of_a_rejected_rate_limit() {
    let out = format!(
        "{INIT}\n{}\n{}\n",
        r#"{"type":"rate_limit_event","rate_limit_info":{"status":"rejected","resetsAt":1900000000}}"#,
        r#"{"type":"result","subtype":"success","is_error":true,"duration_ms":40,"total_cost_usd":0,"result":"Claude AI usage limit reached"}"#
    );
    let f = parse_output(out.as_bytes(), Some(1), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::RateLimited);
    assert_eq!(
        f.retry_at,
        Some(Timestamp::from_unix_seconds(1_900_000_000))
    );
}

#[test]
fn s16_parse_output_counts_a_usage_limit_error_as_rate_limited() {
    let out = r#"{"type":"result","subtype":"success","is_error":true,"api_error_status":429,"total_cost_usd":0,"result":"API Error: 429"}"#;
    let f = parse_output(out.as_bytes(), Some(1), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::RateLimited);
    assert_eq!(f.retry_at, None);
}

#[test]
fn s16_parse_output_without_a_result_is_a_failed_call_with_the_stderr_tail() {
    let f = parse_output(b"not json\n", Some(2), b"something broke").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::CallFailed);
    assert!(f.message.contains("something broke"), "{}", f.message);
}

#[test]
fn s16_parse_output_with_another_error_is_a_failed_call() {
    let out = r#"{"type":"result","subtype":"error_during_execution","is_error":true,"total_cost_usd":0.0004,"result":"API Error: 500 overloaded"}"#;
    let f = parse_output(out.as_bytes(), Some(1), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::CallFailed);
    assert!((f.cost_usd - 0.0004).abs() < 1e-12);
}

#[test]
fn s16_parse_output_without_structured_output_is_a_bad_answer() {
    let out = r#"{"type":"result","subtype":"success","is_error":false,"total_cost_usd":0.001,"result":"Here are my verdicts"}"#;
    let f = parse_output(out.as_bytes(), Some(0), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::BadAnswer);
    assert!(
        (f.cost_usd - 0.001).abs() < 1e-12,
        "the cost is still counted"
    );
}

#[test]
fn s16_parse_output_with_verdicts_of_the_wrong_shape_is_a_bad_answer() {
    let out = result_line(r#"{"verdicts":[{"id":"c01","keep":"yes"}]}"#);
    let f = parse_output(out.as_bytes(), Some(0), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::BadAnswer);
}

#[test]
fn s16_parse_output_with_a_nonzero_exit_and_an_answer_is_a_failed_call() {
    let out = result_line(r#"{"verdicts":[]}"#);
    let f = parse_output(out.as_bytes(), Some(3), b"").expect_err("a failure");
    assert_eq!(f.kind, JudgeFailureKind::CallFailed);
}

#[test]
fn s16_check_accepts_one_verdict_per_candidate_in_any_order() {
    let c = batch(2);
    let v = vec![
        verdict("c02", false, "status", "whatever", "ignored"),
        verdict(
            "c01",
            true,
            "knowledge",
            "pitfall",
            "  Port 80 needs admin rights.  ",
        ),
    ];
    let checked = check(&c, &v).expect("valid");
    assert_eq!(checked.len(), 2);
    assert!(checked[0].keep);
    assert_eq!(checked[0].memory_type, MemoryType::Pitfall);
    assert_eq!(checked[0].memory, "Port 80 needs admin rights.");
    assert!(!checked[1].keep);
    assert_eq!(checked[1].reason, "status");
    assert_eq!(checked[1].memory, "", "a rejected rewrite is never kept");
}

#[test]
fn s16_check_refuses_an_answer_that_does_not_judge_each_candidate_once() {
    let c = batch(2);
    let k = |id: &str| verdict(id, true, "knowledge", "note", "A fact.");
    assert!(
        check(&c, &[k("c01")])
            .unwrap_err()
            .contains("c02 was not judged")
    );
    assert!(
        check(&c, &[k("c01"), k("c01"), k("c02")])
            .unwrap_err()
            .contains("twice")
    );
    assert!(
        check(&c, &[k("c01"), k("c02"), k("c03")])
            .unwrap_err()
            .contains("not in the batch")
    );
}

#[test]
fn s16_check_refuses_a_verdict_that_breaks_the_schema() {
    let c = batch(1);
    let bad = [
        verdict("c01", true, "narration", "note", "A fact."),
        verdict("c01", false, "knowledge", "note", ""),
        verdict("c01", true, "knowledge", "insight", "A fact."),
        verdict("c01", false, "because", "note", ""),
        verdict("c01", true, "knowledge", "note", "   "),
        verdict(
            "c01",
            true,
            "knowledge",
            "note",
            &"x".repeat(MAX_REWRITE_CHARS + 1),
        ),
    ];
    for v in bad {
        assert!(check(&c, std::slice::from_ref(&v)).is_err(), "{v:?}");
    }
    let longest = verdict(
        "c01",
        true,
        "knowledge",
        "note",
        &"x".repeat(MAX_REWRITE_CHARS),
    );
    assert!(check(&c, &[longest]).is_ok());
}

#[test]
fn s16_check_errors_never_quote_a_rewrite() {
    let c = batch(1);
    let secretish = "the rewrite text that must not reach the log";
    for v in [
        verdict(
            "c01",
            true,
            "knowledge",
            "note",
            &"y".repeat(MAX_REWRITE_CHARS + 5),
        ),
        verdict("c01", true, "narration", "note", secretish),
        verdict("c01", true, "knowledge", "bogus", secretish),
    ] {
        let e = check(&c, &[v]).unwrap_err();
        assert!(!e.contains(secretish) && !e.contains("yyyy"), "{e}");
    }
}

fn item(content: &str, context: &str) -> JudgeItem {
    JudgeItem {
        id: MemoryId::new_v7(),
        project: ProjectId(1),
        content: content.to_owned(),
        context: context.to_owned(),
        memory_type: MemoryType::Note,
        attempts: 0,
        queued_at: Timestamp::now(),
    }
}

/// Step 16: text and context are redacted again before they reach the
/// judge, whatever the store holds.
#[test]
fn s9_text_and_context_are_redacted_before_they_reach_the_judge() {
    let key = format!("AKIA{}", "Q7".repeat(8));
    let token = format!("sk-ant-{}", "a1B2".repeat(8));
    let items = [item(
        &format!("Deploy with the key {key} from the vault."),
        &format!("the user said: use {token}"),
    )];
    let c = candidates_of(&RegexSecretScanner::new(), &items);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].id, "c01");
    assert!(!c[0].text.contains(&key), "{}", c[0].text);
    assert!(c[0].text.contains("Deploy with the key"));
    assert!(!c[0].context.contains(&token), "{}", c[0].context);
    let sent = serde_json::to_string(&c).expect("serializes");
    assert!(!sent.contains(&key) && !sent.contains(&token));
}

#[test]
fn s16_candidates_are_numbered_c01_upward_never_by_memory_id() {
    let items: Vec<JudgeItem> = (0..12).map(|i| item(&format!("text {i}"), "")).collect();
    let c = candidates_of(&RegexSecretScanner::new(), &items);
    assert_eq!(c[0].id, "c01");
    assert_eq!(c[11].id, "c12");
    for (ci, it) in c.iter().zip(&items) {
        assert!(
            !serde_json::to_string(ci)
                .unwrap()
                .contains(&it.id.to_string())
        );
    }
}

fn touch(p: &Path) {
    std::fs::create_dir_all(p.parent().expect("a parent")).expect("creates");
    std::fs::write(p, b"").expect("writes");
}

#[cfg(windows)]
const EXE: &str = "claude.exe";
#[cfg(not(windows))]
const EXE: &str = "claude";

#[test]
fn s16_find_claude_looks_on_path_then_in_the_home_local_bin() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = tmp.path().join("bin");
    let home = tmp.path().join("home");
    let path_var = std::env::join_paths([&bin]).expect("joins");
    assert_eq!(find_claude_in(None, Some(&path_var), Some(&home)), None);
    let local = home.join(".local").join("bin").join(EXE);
    touch(&local);
    assert_eq!(
        find_claude_in(None, Some(&path_var), Some(&home)),
        Some(local)
    );
    let on_path = bin.join(EXE);
    touch(&on_path);
    assert_eq!(
        find_claude_in(None, Some(&path_var), Some(&home)),
        Some(on_path),
        "PATH comes first"
    );
}

#[test]
fn s16_find_claude_takes_the_setting_as_a_path_or_a_name_on_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = tmp.path().join("bin");
    let path_var = std::env::join_paths([&bin]).expect("joins");
    let custom = tmp.path().join("tools").join(EXE);
    touch(&custom);
    let setting = custom.display().to_string();
    assert_eq!(
        find_claude_in(Some(&setting), Some(&path_var), None),
        Some(custom.clone())
    );
    assert_eq!(
        find_claude_in(
            Some(&tmp.path().join("missing").display().to_string()),
            Some(&path_var),
            None
        ),
        None,
        "a setting that names nothing finds nothing, with no fallback"
    );
    let named = bin.join(EXE);
    touch(&named);
    assert_eq!(
        find_claude_in(Some("claude"), Some(&path_var), None),
        Some(named)
    );
}

/// An npm install puts a `claude.cmd` wrapper on PATH; the judge runs the
/// binary beside it, so no command shell parses the arguments.
#[cfg(windows)]
#[test]
fn s16_find_claude_maps_an_npm_wrapper_to_its_binary() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = tmp.path().join("npm");
    let path_var = std::env::join_paths([&bin]).expect("joins");
    touch(&bin.join("claude.cmd"));
    assert_eq!(
        find_claude_in(None, Some(&path_var), None),
        None,
        "a wrapper with no binary beside it is not run"
    );
    let exe = bin
        .join("node_modules")
        .join("@anthropic-ai")
        .join("claude-code")
        .join("bin")
        .join("claude.exe");
    touch(&exe);
    assert_eq!(find_claude_in(None, Some(&path_var), None), Some(exe));
}

#[test]
fn s16_the_prompt_and_schema_are_the_ones_the_spike_proved() {
    assert!(PROMPT.contains("knowledge"));
    let schema: serde_json::Value = serde_json::from_str(SCHEMA).expect("the schema is JSON");
    let reasons = schema
        .pointer("/properties/verdicts/items/properties/reason/enum")
        .and_then(|v| v.as_array())
        .expect("the schema lists the reasons");
    let listed: Vec<&str> = reasons.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(
        listed, REASONS,
        "check() accepts exactly the schema's reasons"
    );
}

#[test]
fn s16_the_failure_kinds_are_the_journal_labels() {
    use JudgeFailureKind::*;
    let names: Vec<&str> = [
        Unavailable,
        RateLimited,
        Timeout,
        CallFailed,
        BadAnswer,
        Stopped,
    ]
    .iter()
    .map(|k| k.as_str())
    .collect();
    assert_eq!(
        names,
        [
            "unavailable",
            "rate_limited",
            "timeout",
            "call_failed",
            "bad_answer",
            "stopped"
        ]
    );
}

#[test]
fn s16_the_next_utc_day_starts_at_midnight() {
    let t = Timestamp::parse("2026-11-17T23:59:59Z").expect("parses");
    assert_eq!(next_day(t).to_rfc3339(), "2026-11-18T00:00:00Z");
    let midnight = Timestamp::parse("2026-11-18T00:00:00Z").expect("parses");
    assert_eq!(next_day(midnight).to_rfc3339(), "2026-11-19T00:00:00Z");
}
