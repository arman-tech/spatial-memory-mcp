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

fn verdict(id: &str, keep: bool, reason: &str, ty: &str, quotes: &[&str]) -> JudgeVerdict {
    JudgeVerdict {
        id: id.to_owned(),
        keep,
        reason: reason.to_owned(),
        memory_type: ty.to_owned(),
        quotes: quotes.iter().map(|q| (*q).to_owned()).collect(),
    }
}

fn candidate(id: &str, text: &str) -> JudgeCandidate {
    JudgeCandidate {
        id: id.to_owned(),
        text: text.to_owned(),
        context: String::new(),
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
            r#"{"verdicts":[{"id":"c01","keep":true,"reason":"knowledge","type":"decision","quotes":["The build uses one lock file."]},{"id":"c02","keep":false,"reason":"narration","type":"note","quotes":[]}]}"#
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
            &["The build uses one lock file."]
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
    let c = [
        candidate(
            "c01",
            "The server is up. Port 80 needs admin rights on this machine.",
        ),
        candidate("c02", "All the tests pass."),
    ];
    let v = vec![
        verdict("c02", false, "status", "whatever", &["All the tests pass."]),
        verdict(
            "c01",
            true,
            "knowledge",
            "pitfall",
            &["  Port 80 needs admin rights  "],
        ),
    ];
    let checked = check(&c, &v).expect("valid");
    assert_eq!(checked.len(), 2);
    assert!(checked[0].keep);
    assert_eq!(checked[0].memory_type, MemoryType::Pitfall);
    assert_eq!(
        checked[0].memory, "Port 80 needs admin rights on this machine.",
        "the sentence the quote found, whole, from the text"
    );
    assert_eq!((checked[0].quotes, checked[0].dropped), (1, 0));
    assert_eq!(checked[0].unsettled, None);
    assert!(!checked[1].keep);
    assert_eq!(checked[1].reason, "status");
    assert_eq!(checked[1].memory, "", "a rejected capture keeps nothing");
}

#[test]
fn s16_check_refuses_an_answer_that_does_not_judge_each_candidate_once() {
    let c = batch(2);
    let k = |id: &str| verdict(id, true, "knowledge", "note", &["candidate text"]);
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
    let q: &[&str] = &["candidate text 1"];
    let bad = [
        verdict("c01", true, "narration", "note", q),
        verdict("c01", false, "knowledge", "note", &[]),
        verdict("c01", true, "knowledge", "insight", q),
        verdict("c01", false, "because", "note", &[]),
        verdict("c01", true, "knowledge", "note", &[]),
        verdict("c01", true, "knowledge", "note", &["   ", ""]),
    ];
    for v in bad {
        assert!(check(&c, std::slice::from_ref(&v)).is_err(), "{v:?}");
    }
    assert!(check(&c, &[verdict("c01", true, "knowledge", "note", q)]).is_ok());
}

/// B-173: a kept verdict whose quotes are not in the text is not an error
/// of the batch; it leaves that capture unsettled, with nothing to store.
#[test]
fn s16_check_leaves_a_kept_capture_unquoted_when_no_quote_is_in_its_text() {
    let c = [candidate("c01", "The cache is keyed by path.")];
    let v = verdict(
        "c01",
        true,
        "knowledge",
        "note",
        &["The cache is keyed by hash.", "It is cleared on start."],
    );
    let checked = check(&c, &[v]).expect("not an error of the batch");
    assert!(checked[0].keep);
    assert_eq!(checked[0].unsettled, Some("unquoted"));
    assert_eq!(checked[0].memory, "", "nothing to store");
    assert_eq!((checked[0].quotes, checked[0].dropped), (2, 2));
}

/// B-173: the sentences the quotes found are kept up to
/// [`MAX_REWRITE_CHARS`] characters; beyond that the capture is unsettled.
#[test]
fn s16_check_leaves_a_kept_capture_too_long_past_the_longest_memory() {
    for (n, unsettled) in [
        (MAX_REWRITE_CHARS, None),
        (MAX_REWRITE_CHARS + 1, Some("too_long")),
    ] {
        let text = "x".repeat(n);
        let c = [candidate("c01", &text)];
        let v = verdict("c01", true, "knowledge", "note", &[text.as_str()]);
        let checked = check(&c, &[v]).expect("not an error of the batch");
        assert_eq!(checked[0].unsettled, unsettled, "{n}");
        assert_eq!(
            checked[0].memory.len(),
            if unsettled.is_none() { n } else { 0 }
        );
    }
}

#[test]
fn s16_check_errors_never_repeat_a_quote() {
    let c = batch(1);
    let secretish = "the quoted text that must not reach the log";
    for v in [
        verdict("c01", true, "narration", "note", &[secretish]),
        verdict("c01", true, "knowledge", "bogus", &[secretish]),
        verdict("c01", false, "knowledge", "note", &[secretish]),
    ] {
        let e = check(&c, &[v]).unwrap_err();
        assert!(!e.contains(secretish), "{e}");
    }
}

// --- quotes (B-173) ---------------------------------------------------------

fn quotes_of(text: &str, quotes: &[&str]) -> quote::Quoted {
    let quotes: Vec<String> = quotes.iter().map(|q| (*q).to_owned()).collect();
    quote::quoted(text, &quotes)
}

#[test]
fn s16_a_quote_is_found_whatever_its_markdown_marks_and_spacing() {
    let text = "The **cache** is keyed\n  by `path`.\nIt is cleared on start.";
    let q = quotes_of(text, &["The cache is keyed by path."]);
    assert_eq!(
        q.memory, "The **cache** is keyed\n  by `path`.",
        "copied from the text"
    );
    assert_eq!((q.quotes, q.dropped), (1, 0));
    let typographic = quotes_of(
        "The log says \u{201C}idle\u{201D} when the queue is empty.",
        &["The log says \"idle\" when the queue is empty"],
    );
    assert_eq!(
        typographic.memory,
        "The log says \u{201C}idle\u{201D} when the queue is empty."
    );
}

#[test]
fn s16_a_quote_that_is_not_in_the_text_is_dropped() {
    let text = "The cache is keyed by path. Logs rotate daily.";
    let none = quotes_of(text, &["The cache is keyed by hash."]);
    assert_eq!(none.memory, "");
    assert_eq!((none.quotes, none.dropped), (1, 1));
    let one = quotes_of(text, &["The cache is keyed by hash.", "Logs rotate daily."]);
    assert_eq!(one.memory, "Logs rotate daily.", "only what the text says");
    assert_eq!((one.quotes, one.dropped), (2, 1));
    let reworded = quotes_of(text, &["The cache is keyed by its path."]);
    assert_eq!(reworded.memory, "", "a reworded quote is not in the text");
}

#[test]
fn s16_a_quote_cut_short_brings_its_whole_sentence() {
    let text = "Retries run only for idempotent requests, never for uploads. Logs rotate daily.";
    let q = quotes_of(text, &["Retries run"]);
    assert_eq!(
        q.memory, "Retries run only for idempotent requests, never for uploads.",
        "the qualifiers of the sentence are kept"
    );
    let middle = quotes_of(text, &["for idempotent requests"]);
    assert_eq!(middle.memory, q.memory);
    let across = quotes_of(text, &["never for uploads. Logs rotate"]);
    assert_eq!(
        across.memory, text,
        "a quote across two sentences brings both"
    );
}

#[test]
fn s16_a_quote_starts_and_ends_at_a_word_boundary() {
    let text = "The importer runs first.";
    assert_eq!(quotes_of(text, &["port"]).memory, "");
    assert_eq!(quotes_of(text, &["importer run"]).memory, "");
    assert_eq!(quotes_of(text, &["importer runs"]).memory, text);
}

#[test]
fn s16_an_abbreviation_does_not_end_a_sentence() {
    let text = "Use a fixed seed, e.g. the value 7, in tests. Nothing else changes.";
    let q = quotes_of(text, &["Use a fixed seed"]);
    assert_eq!(q.memory, "Use a fixed seed, e.g. the value 7, in tests.");
}

#[test]
fn s16_quoted_sentences_keep_the_texts_order_and_merge_where_they_touch() {
    let text = "A is first. B is second. C is third. D is fourth.";
    let touching = quotes_of(text, &["C is third", "A is first", "B is second"]);
    assert_eq!(touching.memory, "A is first. B is second. C is third.");
    let apart = quotes_of(text, &["D is fourth", "A is first"]);
    assert_eq!(apart.memory, "A is first.\nD is fourth.");
    let twice = quotes_of(text, &["B is second", "B is second."]);
    assert_eq!(twice.memory, "B is second.");
}

#[test]
fn s16_a_quotes_list_marker_quotation_marks_and_closing_stop_are_ignored() {
    let text = "Rules:\n- Never edit the lock file by hand.\n- Run the formatter first.\n1. Pin every version.";
    let q = quotes_of(
        text,
        &[
            "Never edit the lock file by hand",
            "\"- Run the formatter first.\"",
            "2. Pin every version",
        ],
    );
    assert_eq!(
        q.memory,
        "- Never edit the lock file by hand.\n- Run the formatter first.\n1. Pin every version."
    );
    assert_eq!(q.dropped, 0);
    let blank = quotes_of(text, &["", "  "]);
    assert_eq!(
        (blank.memory.as_str(), blank.quotes),
        ("", 0),
        "blank quotes are not counted"
    );
}

/// B-174: a quoted nested item without its parent reads as part of the
/// item the memory puts before it, which changes what it is about.
#[test]
fn s16_a_quoted_nested_line_brings_the_lines_it_is_read_under() {
    let text = "Two defects:\n\
                1. The importer stored a whole session as one memory.\n   \
                - Cause: the turn boundary was missed.\n\
                2. The doctor failed in every project.\n   \
                - Cause: it read the wrong settings key.\n";
    let q = quotes_of(
        text,
        &[
            "The importer stored a whole session as one memory",
            "it read the wrong settings key",
        ],
    );
    assert_eq!(
        q.memory,
        "1. The importer stored a whole session as one memory.\n\
         2. The doctor failed in every project.\n   \
         - Cause: it read the wrong settings key.",
        "the second cause stays under the doctor's item"
    );
    assert_eq!((q.quotes, q.dropped), (2, 0), "a parent is not a quote");

    let deep = "- Release:\n  - Windows:\n    - Sign the installer first.\n- Logs rotate daily.";
    assert_eq!(
        quotes_of(deep, &["Sign the installer first"]).memory,
        &deep[..deep.find("\n- Logs").unwrap()],
        "every parent up to a line that is not indented"
    );

    let continued = "- The importer runs first.\n  It must finish before the indexer starts.\n- Logs rotate daily.";
    assert_eq!(
        quotes_of(continued, &["It must finish before the indexer starts"]).memory,
        "- The importer runs first.\n  It must finish before the indexer starts.",
        "an indented continuation is read under its item"
    );

    let top = "Rules:\n- Never edit the lock file by hand.";
    assert_eq!(
        quotes_of(top, &["Never edit the lock file by hand"]).memory,
        "- Never edit the lock file by hand.",
        "a line that is not indented brings nothing"
    );
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
