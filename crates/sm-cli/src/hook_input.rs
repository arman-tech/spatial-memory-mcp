//! What a hook reads, under the version-2 rules (SPEC 16, "Events" and
//! "Clients"): the event name and its aliases, the `--client` flag, client
//! detection, stdin, Cursor's field names, and the checks on `session_id`,
//! `transcript_path` and `cwd`.
//!
//! The rules and their cases are in `fixtures/capture/event_aliases.json` and
//! `fixtures/capture/hook_helpers_cases.json`; the tests below run them.

use serde_json::{Map, Value};
use sm_core::protocol::HookEvent;

/// The most stdin the hook reads, in bytes. Version 2's dispatcher read this
/// many bytes (`sys.stdin.buffer.read(524_288)`); anything beyond is ignored,
/// which usually leaves invalid JSON and so no output.
pub const STDIN_LIMIT_BYTES: usize = 524_288;

/// Printed for `spatial-memory hook` with no event, `--help` or `-h`.
pub const USAGE: &str = "Usage: spatial-memory hook <event> [--client <client>]\nEvents: session-start, post-tool-use, pre-compact, stop, user-prompt-submit\n";

/// The default value of `--client`.
pub const CLAUDE_CODE: &str = "claude-code";
/// The client name that turns on Cursor's field names.
pub const CURSOR: &str = "cursor";

const MAX_SESSION_ID_CHARS: usize = 128;
const MAX_CWD_CHARS: usize = 4096;

/// The hook's command line after `hook`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookArgs {
    /// No event, `--help` or `-h`: print [`USAGE`].
    Usage,
    /// An event name that is not one of the aliases: print nothing and do
    /// not read stdin.
    Unknown,
    /// A known event.
    Event {
        event: HookEvent,
        /// The raw `--client` value ([`parse_client_flag`]).
        client_flag: String,
        /// `--profile <name>` or `--profile=<name>`, if given.
        profile: Option<String>,
    },
}

/// Parses the arguments after `hook`. `--profile` is a version-3 addition:
/// it is taken out wherever it appears, and the version-2 rules apply to
/// what is left.
pub fn parse_hook_args(args: &[String]) -> HookArgs {
    let (profile, rest) = take_profile(args);
    match rest.first().map(String::as_str) {
        None | Some("--help" | "-h") => HookArgs::Usage,
        Some(raw) => match normalize_event(raw) {
            None => HookArgs::Unknown,
            Some(event) => HookArgs::Event {
                event,
                client_flag: parse_client_flag(&rest),
                profile,
            },
        },
    }
}

/// Removes `--profile <value>` and `--profile=<value>` from the arguments.
/// A trailing `--profile` with no value is dropped. The last one given wins.
fn take_profile(args: &[String]) -> (Option<String>, Vec<String>) {
    let mut profile = None;
    let mut rest = Vec::with_capacity(args.len());
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--profile" {
            if let Some(value) = iter.next() {
                profile = Some(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--profile=") {
            profile = Some(value.to_owned());
        } else {
            rest.push(arg.clone());
        }
    }
    (profile, rest)
}

/// The event for a name, or `None` for an unknown one. The names are matched
/// exactly: PascalCase, camelCase, kebab-case and Cursor's
/// `afterMCPExecution`.
pub fn normalize_event(raw: &str) -> Option<HookEvent> {
    Some(match raw {
        "SessionStart" | "sessionStart" | "session-start" => HookEvent::SessionStart,
        "PostToolUse"
        | "postToolUse"
        | "post-tool-use"
        | "afterMCPExecution"
        | "after-mcp-execution" => HookEvent::PostToolUse,
        "PreCompact" | "preCompact" | "pre-compact" => HookEvent::PreCompact,
        "Stop" | "stop" => HookEvent::Stop,
        "UserPromptSubmit" | "userPromptSubmit" | "user-prompt-submit" => {
            HookEvent::UserPromptSubmit
        }
        _ => return None,
    })
}

/// The value after the first `--client` that has one, else `claude-code`.
/// `--client=x` is not recognised, as in version 2.
pub fn parse_client_flag(argv: &[String]) -> String {
    argv.windows(2)
        .find(|pair| pair[0] == "--client")
        .map_or_else(|| CLAUDE_CODE.to_owned(), |pair| pair[1].clone())
}

/// The client: an explicit flag other than `claude-code` wins; otherwise
/// Cursor when the payload carries a truthy `cursor_version` or
/// `cursorVersion`; otherwise the flag, or `claude-code` when it is empty.
pub fn detect_client(data: &Map<String, Value>, explicit: &str) -> String {
    if !explicit.is_empty() && explicit != CLAUDE_CODE {
        return explicit.to_owned();
    }
    if data.get("cursor_version").is_some_and(py_truthy)
        || data.get("cursorVersion").is_some_and(py_truthy)
    {
        return CURSOR.to_owned();
    }
    if explicit.is_empty() {
        CLAUDE_CODE.to_owned()
    } else {
        explicit.to_owned()
    }
}

/// Renames Cursor's fields to Claude Code's. Other clients' payloads are
/// returned unchanged.
pub fn normalize_stdin(mut data: Map<String, Value>, client: &str) -> Map<String, Value> {
    if client != CURSOR {
        return data;
    }
    if !data.contains_key("session_id")
        && let Some(id) = data.remove("conversation_id")
    {
        data.insert("session_id".to_owned(), id);
    }
    if !data.contains_key("cwd")
        && let Some(Value::Array(roots)) = data.get("workspace_roots")
        && let Some(first) = roots.first()
    {
        let cwd = py_str(first);
        data.insert("cwd".to_owned(), Value::String(cwd));
    }
    if let Some(Value::String(cwd)) = data.get_mut("cwd")
        && let Some(fixed) = cursor_drive_path(cwd)
    {
        *cwd = fixed;
    }
    if !data.contains_key("tool_response") {
        let moved = data
            .remove("tool_output")
            .or_else(|| data.remove("result_json"));
        if let Some(response) = moved {
            data.insert("tool_response".to_owned(), response);
        }
    }
    if !data.contains_key("trigger")
        && let Some(status) = data.remove("status")
    {
        data.insert("trigger".to_owned(), status);
    }
    data.entry("source")
        .or_insert_with(|| Value::String("startup".to_owned()));
    data
}

/// Cursor writes Windows folders as `/c:/Users/...`; this gives
/// `C:/Users/...`. `None` when the path does not have that shape.
fn cursor_drive_path(cwd: &str) -> Option<String> {
    let mut chars = cwd.chars();
    let (Some('/'), Some(letter), Some(':')) = (chars.next(), chars.next(), chars.next()) else {
        return None;
    };
    let mut fixed: String = letter.to_uppercase().collect();
    fixed.push(':');
    fixed.push_str(chars.as_str());
    Some(fixed)
}

/// Checks `session_id`, `transcript_path` and `cwd`. A present value is
/// turned into text as Python's `str()` would; non-empty text is replaced
/// by its checked form, which is empty when the check fails. Missing keys
/// and empty values are left as they were. `windows` picks the path rules.
pub fn validate_common(data: &mut Map<String, Value>, windows: bool) {
    for key in ["session_id", "transcript_path", "cwd"] {
        let Some(value) = data.get(key) else { continue };
        let text = py_str(value);
        if text.is_empty() {
            continue;
        }
        let checked = match key {
            "session_id" => sanitize_session_id(&text),
            "transcript_path" => validate_transcript_path(&text, windows),
            _ => validate_cwd(&text, windows),
        };
        data.insert(key.to_owned(), Value::String(checked));
    }
}

/// The session id unchanged when it is 1 to 128 ASCII letters, digits,
/// hyphens or underscores and not a Windows device name; otherwise empty.
/// Version 2's regex also accepted one trailing newline (a Python `$`
/// quirk); this rejects it, as the fixture recommends.
pub fn sanitize_session_id(id: &str) -> String {
    let ok = !id.is_empty()
        && id.chars().count() <= MAX_SESSION_ID_CHARS
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        && !is_device_name(id);
    if ok { id.to_owned() } else { String::new() }
}

/// The transcript path unchanged when it is absolute and has no `..`
/// anywhere; otherwise empty.
pub fn validate_transcript_path(path: &str, windows: bool) -> String {
    if path.is_empty() || path.contains("..") || !is_abs(path, windows) {
        return String::new();
    }
    path.to_owned()
}

/// The working folder unchanged when it is absolute, at most 4,096
/// characters, has no `..` and does not end in a Windows device name;
/// otherwise empty. No file-system access.
pub fn validate_cwd(cwd: &str, windows: bool) -> String {
    if cwd.is_empty()
        || cwd.chars().count() > MAX_CWD_CHARS
        || cwd.contains("..")
        || !is_abs(cwd, windows)
    {
        return String::new();
    }
    let base = basename(cwd, windows);
    if !base.is_empty() && is_device_name(base) {
        return String::new();
    }
    cwd.to_owned()
}

/// `os.path.isabs` as Python 3.13 has it. On Windows a path needs a drive
/// and a root (`C:\x`, `C:/x`) or two leading separators (UNC); `/x` and
/// `C:x` are not absolute.
pub fn is_abs(path: &str, windows: bool) -> bool {
    if !windows {
        return path.starts_with('/');
    }
    let head: Vec<char> = path
        .chars()
        .take(3)
        .map(|c| if c == '/' { '\\' } else { c })
        .collect();
    (head.len() == 3 && head[1] == ':' && head[2] == '\\')
        || (head.len() >= 2 && head[0] == '\\' && head[1] == '\\')
}

/// `os.path.basename`: the text after the last separator. On Windows the
/// separators are `/` and `\`, and the drive or UNC share is skipped first.
pub fn basename(path: &str, windows: bool) -> &str {
    if !windows {
        return path.rsplit('/').next().unwrap_or("");
    }
    win_after_root(path)
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
}

/// The part of a Windows path after its drive and root, as Python 3.13's
/// `ntpath.splitroot` splits it.
fn win_after_root(path: &str) -> &str {
    let norm: Vec<u8> = path
        .bytes()
        .map(|b| if b == b'/' { b'\\' } else { b })
        .collect();
    let find_sep = |from: usize| {
        norm.get(from..)
            .and_then(|tail| tail.iter().position(|&b| b == b'\\'))
            .map(|i| i + from)
    };
    if norm.first() == Some(&b'\\') {
        if norm.get(1) != Some(&b'\\') {
            // Rooted without a drive, such as `\Windows`.
            return &path[1..];
        }
        // UNC (`\\server\share`, `\\?\UNC\server\share`) or a device.
        let start = if norm.len() >= 8 && norm[..8].eq_ignore_ascii_case(b"\\\\?\\UNC\\") {
            8
        } else {
            2
        };
        let Some(first) = find_sep(start) else {
            return "";
        };
        let Some(second) = find_sep(first + 1) else {
            return "";
        };
        return &path[second + 1..];
    }
    let mut chars = path.char_indices();
    chars.next();
    if let Some((colon, ':')) = chars.next() {
        let after = &path[colon + 1..];
        return after.strip_prefix(['\\', '/']).unwrap_or(after);
    }
    path
}

/// Whether a name is a Windows device: `CON`, `NUL`, `PRN`, `AUX`,
/// `COM1`-`COM9` or `LPT1`-`LPT9` in any case, alone or followed by a dot
/// and at least one more character (no newline).
pub fn is_device_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    let upper = |i: usize| bytes.get(i).map(u8::to_ascii_uppercase);
    let stem = match (upper(0), upper(1), upper(2)) {
        (Some(b'C'), Some(b'O'), Some(b'N'))
        | (Some(b'N'), Some(b'U'), Some(b'L'))
        | (Some(b'P'), Some(b'R'), Some(b'N'))
        | (Some(b'A'), Some(b'U'), Some(b'X')) => 3,
        (Some(b'C'), Some(b'O'), Some(b'M')) | (Some(b'L'), Some(b'P'), Some(b'T'))
            if matches!(bytes.get(3), Some(b'1'..=b'9')) =>
        {
            4
        }
        _ => return false,
    };
    // The stem is ASCII, so `stem` is a character boundary.
    let rest = &name[stem..];
    match rest.strip_prefix('.') {
        None => rest.is_empty(),
        Some(ext) => !ext.is_empty() && !ext.contains('\n'),
    }
}

/// Python's truthiness for a JSON value.
pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n
            .as_i64()
            .map(|i| i != 0)
            .or_else(|| n.as_u64().map(|u| u != 0))
            .unwrap_or_else(|| n.as_f64().is_some_and(|f| f != 0.0)),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `str()` of a JSON value. Lists and objects come out as JSON
/// rather than Python's repr; both start with `[` or `{`, which none of the
/// checks above accepts, so the checked result is the same. An integer too
/// large for 64 bits arrives as a float here, where Python kept it exact.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_float(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed
/// notation for exponents from -4 to 15 (always with a fractional part) and
/// in scientific notation otherwise (`1e-05`, `1.5e+20`).
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            let zeros = "0".repeat(point.unsigned_abs() as usize);
            format!("{sign}0.{zeros}{digits}")
        } else {
            let point = point.unsigned_abs() as usize;
            if point >= digits.len() {
                let zeros = "0".repeat(point - digits.len());
                format!("{sign}{digits}{zeros}.0")
            } else {
                format!("{sign}{}.{}", &digits[..point], &digits[point..])
            }
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exponent.unsigned_abs())
    }
}

/// Parses what the hook read from stdin, as version 2's dispatcher did.
///
/// - Empty, or only ASCII whitespace: an empty object.
/// - Otherwise the bytes are decoded as `json.loads` decodes bytes (a UTF-8,
///   UTF-16 or UTF-32 byte-order mark, or the zero-byte pattern of UTF-16 or
///   UTF-32 without one; UTF-8 by default) and parsed as JSON.
/// - A decoding or JSON error is `Err`: the hook prints nothing.
/// - JSON that is not an object: an empty object.
pub fn parse_stdin(raw: &[u8]) -> Result<Map<String, Value>, String> {
    if raw
        .iter()
        .all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
    {
        return Ok(Map::new());
    }
    let text = decode_json_bytes(raw)?;
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Ok(Map::new()),
        Err(e) => Err(format!("invalid JSON on stdin: {e}")),
    }
}

/// Decodes JSON bytes as Python's `json.detect_encoding` chooses.
fn decode_json_bytes(b: &[u8]) -> Result<String, String> {
    const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
    if b.starts_with(&[0x00, 0x00, 0xFE, 0xFF]) {
        return decode_utf32(&b[4..], true);
    }
    if b.starts_with(&[0xFF, 0xFE, 0x00, 0x00]) {
        return decode_utf32(&b[4..], false);
    }
    if b.starts_with(&[0xFE, 0xFF]) {
        return decode_utf16(&b[2..], true);
    }
    if b.starts_with(&[0xFF, 0xFE]) {
        return decode_utf16(&b[2..], false);
    }
    if let Some(rest) = b.strip_prefix(UTF8_BOM) {
        return decode_utf8(rest);
    }
    if b.len() >= 4 {
        if b[0] == 0 {
            return if b[1] != 0 {
                decode_utf16(b, true)
            } else {
                decode_utf32(b, true)
            };
        }
        if b[1] == 0 {
            return if b[2] != 0 || b[3] != 0 {
                decode_utf16(b, false)
            } else {
                decode_utf32(b, false)
            };
        }
    } else if b.len() == 2 {
        if b[0] == 0 {
            return decode_utf16(b, true);
        }
        if b[1] == 0 {
            return decode_utf16(b, false);
        }
    }
    decode_utf8(b)
}

fn decode_utf8(b: &[u8]) -> Result<String, String> {
    String::from_utf8(b.to_vec()).map_err(|e| format!("stdin is not valid UTF-8: {e}"))
}

fn decode_utf16(b: &[u8], big_endian: bool) -> Result<String, String> {
    if !b.len().is_multiple_of(2) {
        return Err("stdin is truncated UTF-16".to_owned());
    }
    let units = b.chunks_exact(2).map(|pair| {
        let pair = [pair[0], pair[1]];
        if big_endian {
            u16::from_be_bytes(pair)
        } else {
            u16::from_le_bytes(pair)
        }
    });
    char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .map_err(|e| format!("stdin is not valid UTF-16: {e}"))
}

fn decode_utf32(b: &[u8], big_endian: bool) -> Result<String, String> {
    if !b.len().is_multiple_of(4) {
        return Err("stdin is truncated UTF-32".to_owned());
    }
    b.chunks_exact(4)
        .map(|quad| {
            let quad = [quad[0], quad[1], quad[2], quad[3]];
            let code = if big_endian {
                u32::from_be_bytes(quad)
            } else {
                u32::from_le_bytes(quad)
            };
            char::from_u32(code)
                .ok_or_else(|| format!("stdin has an invalid UTF-32 code {code:#x}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/capture")
            .join(name);
        let text = std::fs::read_to_string(&path).expect("fixture readable");
        serde_json::from_str(&text).expect("fixture is JSON")
    }

    /// Expands the fixtures' `{"$concat": [...]}` and `{"$repeat": s,
    /// "count": n}` builders into strings.
    fn build(value: &Value) -> Value {
        match value {
            Value::Object(map) if map.contains_key("$concat") => {
                let parts = map["$concat"].as_array().expect("$concat is a list");
                let mut out = String::new();
                for part in parts {
                    match build(part) {
                        Value::String(s) => out.push_str(&s),
                        other => panic!("$concat part is not text: {other}"),
                    }
                }
                Value::String(out)
            }
            Value::Object(map) if map.contains_key("$repeat") => {
                let unit = map["$repeat"].as_str().expect("$repeat is text");
                let count = map["count"].as_u64().expect("count") as usize;
                Value::String(unit.repeat(count))
            }
            Value::Object(map) => {
                Value::Object(map.iter().map(|(k, v)| (k.clone(), build(v))).collect())
            }
            Value::Array(items) => Value::Array(items.iter().map(build).collect()),
            other => other.clone(),
        }
    }

    /// Replaces `{TMP}` with an absolute folder for the platform the case
    /// is checked under.
    fn with_tmp(value: &Value, windows: bool) -> Value {
        let tmp = if windows {
            r"C:\Temp\sm-fixture"
        } else {
            "/tmp/sm-fixture"
        };
        match value {
            Value::String(s) => Value::String(s.replace("{TMP}", tmp)),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), with_tmp(v, windows)))
                    .collect(),
            ),
            Value::Array(items) => {
                Value::Array(items.iter().map(|v| with_tmp(v, windows)).collect())
            }
            other => other.clone(),
        }
    }

    fn pascal(event: HookEvent) -> &'static str {
        match event {
            HookEvent::SessionStart => "SessionStart",
            HookEvent::PostToolUse => "PostToolUse",
            HookEvent::PreCompact => "PreCompact",
            HookEvent::Stop => "Stop",
            HookEvent::UserPromptSubmit => "UserPromptSubmit",
        }
    }

    fn strings(value: &Value) -> Vec<String> {
        value
            .as_array()
            .expect("list")
            .iter()
            .map(|v| v.as_str().expect("text").to_owned())
            .collect()
    }

    #[test]
    fn s16_event_aliases_match_version_2() {
        let fx = fixture("event_aliases.json");
        let aliases = fx["aliases"]["v2"].as_object().expect("aliases.v2");
        for (alias, canonical) in aliases {
            let got = normalize_event(alias).map(pascal);
            assert_eq!(got, canonical.as_str(), "alias {alias}");
        }
        let cases = fx["normalize_event_cases"].as_array().expect("cases");
        assert!(cases.len() >= 26);
        for case in cases {
            let input = case["input"].as_str().expect("input");
            let expected = case["expected"].as_str();
            assert_eq!(
                normalize_event(input).map(pascal),
                expected,
                "event {input:?}"
            );
        }
    }

    #[test]
    fn s16_client_flag_matches_version_2() {
        let fx = fixture("event_aliases.json");
        for case in fx["parse_client_flag_cases"].as_array().expect("cases") {
            let argv = strings(&case["argv"]);
            assert_eq!(
                parse_client_flag(&argv),
                case["expected"].as_str().expect("expected"),
                "argv {argv:?}"
            );
        }
    }

    #[test]
    fn s16_client_detection_matches_version_2() {
        let fx = fixture("event_aliases.json");
        for case in fx["detect_client_cases"].as_array().expect("cases") {
            let data = case["data"].as_object().expect("data");
            let explicit = case["explicit"].as_str().expect("explicit");
            assert_eq!(
                detect_client(data, explicit),
                case["expected"].as_str().expect("expected"),
                "case {case}"
            );
        }
    }

    #[test]
    fn s16_cursor_normalisation_matches_version_2() {
        let fx = fixture("event_aliases.json");
        let cases = fx["normalize_stdin_cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 23);
        for case in cases {
            let data = case["data"].as_object().expect("data").clone();
            let client = case["client"].as_str().expect("client");
            let got = normalize_stdin(data.clone(), client);
            assert_eq!(Value::Object(got.clone()), case["expected"], "case {case}");
            // "returns_same_object" is Python identity; here it means the
            // payload comes back untouched.
            if case["returns_same_object"] == json!(true) {
                assert_eq!(got, data);
            }
        }
    }

    #[test]
    fn s16_validate_common_matches_version_2() {
        // The fixture was computed on Windows (Python 3.13), so the cases
        // run under the Windows path rules on every host.
        let fx = fixture("event_aliases.json");
        let cases = fx["validate_common_cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 13);
        for case in cases {
            let mut data = with_tmp(&case["data"], true)
                .as_object()
                .expect("data")
                .clone();
            validate_common(&mut data, true);
            assert_eq!(
                Value::Object(data),
                with_tmp(&case["expected"], true),
                "case {case}"
            );
        }
    }

    /// Runs a hook_helpers function case under both platforms' rules, using
    /// `computed_win32` and `computed_posix_simulated` where the case has
    /// them, else `expect`'s value under the platform the case names.
    fn check_helper(case: &Value, run: &dyn Fn(&str, bool) -> String) {
        let args = build(&case["args"]);
        let input = args
            .as_object()
            .and_then(|a| a.values().next())
            .and_then(Value::as_str)
            .expect("one text argument")
            .to_owned();
        let mut checked = 0;
        for (key, windows) in [
            ("computed_win32", true),
            ("computed_posix_simulated", false),
        ] {
            if let Some(expected) = case.get(key).filter(|v| !v.is_null()) {
                let input = with_tmp(&Value::String(input.clone()), windows);
                let input = input.as_str().expect("text");
                let expected = with_tmp(&build(expected), windows);
                assert_eq!(
                    run(input, windows),
                    expected.as_str().expect("text"),
                    "{key} {case}"
                );
                checked += 1;
            }
        }
        if checked == 0 {
            let expected = case["expect"][0]["value"].clone();
            let expected = build(&expected);
            let platforms: &[bool] = match case.get("test_runs_on").and_then(Value::as_str) {
                Some("win32") => &[true],
                Some("posix") => &[false],
                _ => &[true, false],
            };
            for &windows in platforms {
                let given = with_tmp(&Value::String(input.clone()), windows);
                let want = with_tmp(&expected, windows);
                assert_eq!(
                    run(given.as_str().expect("text"), windows),
                    want.as_str().expect("text"),
                    "windows={windows} {case}"
                );
            }
        }
    }

    #[test]
    fn s16_session_id_rules_match_version_2() {
        let fx = fixture("hook_helpers_cases.json");
        let run = |s: &str, _windows: bool| sanitize_session_id(s);
        let cases = fx["sanitize_session_id_cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 21);
        for case in cases {
            check_helper(case, &run);
        }
        for probe in fx["probes"]["sanitize_session_id"]
            .as_array()
            .expect("probes")
        {
            let args = build(&probe["args"]);
            let id = args["session_id"].as_str().expect("id");
            let want = build(&probe["computed_result"]);
            let want = want.as_str().expect("text");
            if probe["name"] == "probe.trailing_newline_accepted" {
                // The Python `$` quirk; the fixture recommends rejecting it.
                assert_eq!(want, "abc\n");
                assert_eq!(sanitize_session_id(id), "");
            } else {
                assert_eq!(sanitize_session_id(id), want, "{probe}");
            }
        }
    }

    #[test]
    fn s16_transcript_path_rules_match_version_2() {
        let fx = fixture("hook_helpers_cases.json");
        let run = |s: &str, windows: bool| validate_transcript_path(s, windows);
        for case in fx["validate_transcript_path_cases"]
            .as_array()
            .expect("cases")
        {
            check_helper(case, &run);
        }
        for probe in fx["probes"]["validate_transcript_path"]
            .as_array()
            .expect("probes")
        {
            check_helper(probe, &run);
        }
    }

    #[test]
    fn s16_cwd_rules_match_version_2() {
        let fx = fixture("hook_helpers_cases.json");
        let run = |s: &str, windows: bool| validate_cwd(s, windows);
        for case in fx["validate_cwd_cases"].as_array().expect("cases") {
            check_helper(case, &run);
        }
        for probe in fx["probes"]["validate_cwd"].as_array().expect("probes") {
            check_helper(probe, &run);
        }
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// The bytes a cli case writes to stdin.
    fn case_stdin(case: &Value) -> Vec<u8> {
        if let Some(h) = case.get("stdin_hex").and_then(Value::as_str) {
            return hex(h);
        }
        if let Some(j) = case.get("stdin_json") {
            return j.to_string().into_bytes();
        }
        match build(&case["stdin_text"]) {
            Value::String(s) => s.into_bytes(),
            other => panic!("stdin_text is not text: {other}"),
        }
    }

    #[test]
    fn s16_stdin_rules_match_the_version_2_dispatcher() {
        let fx = fixture("event_aliases.json");
        for case in fx["cli_cases"].as_array().expect("cases") {
            let argv = strings(&case["argv"]);
            let parsed = parse_hook_args(&argv);
            let stdout = case["expected"]["stdout"].as_str().expect("stdout");
            let logged = !case["expected"]["hook_errors_log"].is_null();
            match parsed {
                HookArgs::Usage => assert!(stdout.starts_with("Usage:"), "{case}"),
                HookArgs::Unknown => assert_eq!(stdout, "", "{case}"),
                HookArgs::Event { .. } => {
                    let mut raw = case_stdin(case);
                    raw.truncate(STDIN_LIMIT_BYTES);
                    // version 2 logged exactly the inputs it could not parse.
                    assert_eq!(parse_stdin(&raw).is_err(), logged, "{case}");
                }
            }
        }
    }

    #[test]
    fn s16_stdin_bom_and_wide_encodings_decode() {
        let mut utf8 = vec![0xEF, 0xBB, 0xBF];
        utf8.extend_from_slice(br#"{"a": 1}"#);
        assert_eq!(
            parse_stdin(&utf8).unwrap(),
            json!({"a": 1}).as_object().unwrap().clone()
        );
        let text = r#"{"a": "é"}"#;
        let le: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let be: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
        let utf32: Vec<u8> = text
            .chars()
            .flat_map(|c| (c as u32).to_le_bytes())
            .collect();
        let want = json!({"a": "é"}).as_object().unwrap().clone();
        assert_eq!(parse_stdin(&le).unwrap(), want);
        assert_eq!(parse_stdin(&be).unwrap(), want);
        assert_eq!(parse_stdin(&utf32).unwrap(), want);
        assert_eq!(parse_stdin(b"[1, 2]").unwrap(), Map::new());
        assert_eq!(parse_stdin(b" \x0c\n").unwrap(), Map::new());
        assert!(parse_stdin(b"{\"a\": 1} x").is_err());
    }

    #[test]
    fn s16_hook_args_take_profile_anywhere() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_hook_args(&args(&[])), HookArgs::Usage);
        assert_eq!(parse_hook_args(&args(&["-h"])), HookArgs::Usage);
        assert_eq!(
            parse_hook_args(&args(&["--profile", "work"])),
            HookArgs::Usage
        );
        assert_eq!(parse_hook_args(&args(&["nope"])), HookArgs::Unknown);
        assert_eq!(
            parse_hook_args(&args(&["--profile", "work", "stop", "--client", "cursor"])),
            HookArgs::Event {
                event: HookEvent::Stop,
                client_flag: "cursor".to_owned(),
                profile: Some("work".to_owned()),
            }
        );
        assert_eq!(
            parse_hook_args(&args(&["Stop", "--profile=home"])),
            HookArgs::Event {
                event: HookEvent::Stop,
                client_flag: CLAUDE_CODE.to_owned(),
                profile: Some("home".to_owned()),
            }
        );
    }

    #[test]
    fn s16_python_text_and_truthiness() {
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(12345)), "12345");
        assert_eq!(py_str(&json!(-7)), "-7");
        assert_eq!(py_str(&json!(1.0)), "1.0");
        assert_eq!(py_str(&json!(0.5)), "0.5");
        assert_eq!(py_str(&json!(0.00001)), "1e-05");
        assert_eq!(py_str(&json!(1.5e20)), "1.5e+20");
        assert_eq!(py_str(&json!(1e16)), "1e+16");
        assert_eq!(py_str(&json!(123456.789)), "123456.789");
        assert_eq!(py_str(&json!(0.0001)), "0.0001");
        for falsy in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!py_truthy(&falsy), "{falsy}");
        }
        for truthy in [
            json!(true),
            json!(1),
            json!(-0.5),
            json!("0"),
            json!([0]),
            json!({"a": 0}),
        ] {
            assert!(py_truthy(&truthy), "{truthy}");
        }
    }
}
