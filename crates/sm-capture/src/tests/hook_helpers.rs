//! `hook_helpers_cases.json`: the session identifier and transcript path
//! rules this crate applies (SPEC section 16, Pipeline in the daemon,
//! step 1). The rest of the file (the working directory check, stdin,
//! stdout, the project root and the error log) is stream D's or is replaced
//! by the journal.
//!
//! Path validation differs by platform, so the file runs as a
//! `#[cfg(windows)]` and `#[cfg(unix)]` pair, each comparing with the
//! fixture's `computed_win32` or `computed_posix_simulated`.

use serde_json::{Value, json};

use super::checker::{check_all, resolve};
use super::{Suite, case_id, fixture, list, short, temp_dir, with_tmp};
use crate::transcript::{
    MAX_SESSION_ID_CHARS, check_transcript_path, is_windows_device_name, read_transcript,
    sanitize_session_id, validate_transcript_path,
};

fn session(id: &str) -> Value {
    json!(sanitize_session_id(id).unwrap_or(""))
}

/// Version 1's `validate_transcript_path` is a string check; this crate's
/// [`check_transcript_path`] is the same check.
fn transcript_path(path: &str) -> Value {
    json!(check_transcript_path(path).unwrap_or(""))
}

fn run(platform_key: &str, expected_differences: &[&str]) {
    let f = fixture("hook_helpers_cases.json");
    let mut suite = Suite::new(&format!("hook_helpers_cases.json ({platform_key})"));
    let dir = temp_dir();
    let tmp = dir
        .path()
        .to_str()
        .expect("a UTF-8 temporary path")
        .to_owned();
    let probes = &f["probes"];

    for case in list(&f, "sanitize_session_id_cases") {
        let id = case_id(case);
        let arg = resolve(&case["args"]["session_id"]);
        let observed = json!({"result": session(arg.as_str().expect("session_id"))});
        suite.case(&id, check_all(&observed, &resolve(&case["expect"])));
        suite.computed(
            &id,
            "computed_result",
            &observed["result"],
            &resolve(&case["computed_result"]),
        );
    }
    for probe in list(probes, "sanitize_session_id") {
        let id = case_id(probe);
        let arg = resolve(&probe["args"]["session_id"]);
        let observed = session(arg.as_str().expect("session_id"));
        suite.computed(
            &id,
            "computed_result",
            &observed,
            &resolve(&probe["computed_result"]),
        );
        suite.case(&id, Vec::new());
    }

    for case in list(&f, "validate_transcript_path_cases") {
        let id = case_id(case);
        let arg = with_tmp(&case["args"]["path"], &tmp);
        let observed = json!({"result": transcript_path(arg.as_str().expect("path"))});
        suite.case(&id, check_all(&observed, &with_tmp(&case["expect"], &tmp)));
        suite.computed(
            &id,
            platform_key,
            &observed["result"],
            &with_tmp(&case[platform_key], &tmp),
        );
    }
    for probe in list(probes, "validate_transcript_path") {
        let id = case_id(probe);
        let arg = with_tmp(&probe["args"]["path"], &tmp);
        let observed = transcript_path(arg.as_str().expect("path"));
        suite.computed(
            &id,
            platform_key,
            &observed,
            &with_tmp(&probe[platform_key], &tmp),
        );
        suite.case(&id, Vec::new());
    }

    // The reader applies the same check before it opens anything.
    for case in list(&f, "sanitization_integration_cases") {
        let id = case_id(case);
        match case["function"].as_str() {
            Some("transcript_reader.read_transcript_delta") => {
                let path = case["args"]["transcript_path"]
                    .as_str()
                    .expect("transcript_path");
                let read = read_transcript(path, 0);
                let entries: Vec<Value> = read
                    .entries
                    .iter()
                    .map(|e| json!({"role": e.role, "text": e.text, "timestamp": e.timestamp, "uuid": e.uuid, "entry_type": e.entry_type}))
                    .collect();
                let observed =
                    json!({"result": {"entries": entries, "new_offset": read.new_offset}});
                suite.case(&id, check_all(&observed, &case["expect"]));
                suite.computed(
                    &id,
                    "computed_result",
                    &observed["result"],
                    &case["computed_result"],
                );
            }
            _ => suite.not_applicable(
                &id,
                "load_state and save_state keep the per-session offset, which is the daemon's \
                 (stream E); this crate's reader is stateless",
            ),
        }
    }

    // This crate's additions to version 1's string check: the file must
    // exist and be a regular file, and its name must not be a Windows
    // device name (on every platform).
    let file = dir.path().join("transcript.jsonl");
    std::fs::write(&file, b"{}\n").expect("write a transcript");
    let mut problems = Vec::new();
    let file_str = file.to_str().expect("UTF-8");
    if validate_transcript_path(file_str).as_deref() != Some(file.as_path()) {
        problems.push(format!("a regular file was rejected: {file_str}"));
    }
    if validate_transcript_path(&tmp).is_some() {
        problems.push("a directory was accepted".to_owned());
    }
    let missing = dir.path().join("missing.jsonl");
    if validate_transcript_path(missing.to_str().expect("UTF-8")).is_some() {
        problems.push("a missing file was accepted".to_owned());
    }
    let device = dir.path().join("CON.jsonl");
    if validate_transcript_path(device.to_str().expect("UTF-8")).is_some() {
        problems.push("a device name was accepted".to_owned());
    }
    suite.case("v3.validate_transcript_path.regular_file_only", problems);

    let mut problems = Vec::new();
    if f["constants"]["max_session_id_length"].as_u64() != Some(MAX_SESSION_ID_CHARS as u64) {
        problems.push(format!("max session id {MAX_SESSION_ID_CHARS}"));
    }
    // `_WINDOWS_DEVICE_RE`'s recommended Rust form, applied to names.
    for (name, device) in [
        ("CON", true),
        ("con", true),
        ("nul.txt", true),
        ("LPT9", true),
        ("COM1.log", true),
        ("COM0", false),
        ("COM10", false),
        ("CONSOLE", false),
        ("CONIN", false),
        ("CON\n", false),
        ("CON.", false),
    ] {
        if is_windows_device_name(name) != device {
            problems.push(format!("is_windows_device_name({name:?}) is not {device}"));
        }
    }
    suite.case("constants_and_regexes", problems);

    let not_ours = [
        (
            "validate_cwd_cases",
            "the working directory check is stream D's; the daemon resolves the project from `cwd` with sm-core's resolve_project (CONTRACTS 2)",
        ),
        ("read_stdin_cases", "stdin is stream D's"),
        ("write_stdout_response_cases", "stdout is stream D's"),
        (
            "get_project_root_cases",
            "the project root is stream D's and sm-core's resolve_project",
        ),
        (
            "log_hook_error_cases",
            "the hook error log is replaced by the journal",
        ),
    ];
    for (section, reason) in not_ours {
        for case in list(&f, section) {
            suite.not_applicable(&format!("{section}.{}", case_id(case)), reason);
        }
    }
    for (section, reason) in [
        ("validate_cwd", "the working directory check is stream D's"),
        ("read_stdin", "stdin is stream D's"),
        ("get_project_root", "the project root is stream D's"),
        (
            "log_hook_error",
            "the hook error log is replaced by the journal",
        ),
        (
            "_resolve_log_dir",
            "the hook error log is replaced by the journal",
        ),
    ] {
        for probe in list(probes, section) {
            suite.not_applicable(&format!("probes.{section}.{}", case_id(probe)), reason);
        }
    }
    for key in [
        "_MAX_CWD_LENGTH",
        "_MAX_LOG_SIZE",
        "read_stdin_limit_chars",
        "log_file_name",
        "rotated_log_file_name",
    ] {
        if f["constants"].get(key).is_some() {
            suite.not_applicable(
                &format!("constants.{key}"),
                &format!(
                    "stream D's or the journal's: {}",
                    short(&f["constants"][key])
                ),
            );
        }
    }

    suite.finish(expected_differences);
}

/// The Windows half: `computed_win32`.
#[cfg(windows)]
#[test]
fn s16_hook_helpers_cases_windows() {
    run("computed_win32", &["probe.trailing_newline_accepted"]);
}

/// The Unix half: `computed_posix_simulated`.
#[cfg(unix)]
#[test]
fn s16_hook_helpers_cases_unix() {
    run(
        "computed_posix_simulated",
        &["probe.trailing_newline_accepted"],
    );
}
