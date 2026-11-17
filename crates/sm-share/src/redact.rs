//! Redaction of the strings `migrate` and `import` store besides content
//! and title (SPEC 9 "Secret redaction on write"; the lead's redaction
//! decision in the stream F report).
//!
//! Every string value of a JSON tree is scanned. A string that is the value
//! of an object key (or an item of an array that is) is scanned a second
//! time as `key: value`, so a keyed secret such as `{"password": "hunter2"}`
//! is found the way `password: hunter2` is found in text. A string that
//! holds a JSON object or array is read as JSON and scanned the same way.
//! Only values are replaced; keys are kept as they are. Text that already
//! holds a `[REDACTED:<kind>]` marker is not changed again (B-64).

use serde_json::Value;
use sm_core::traits::{SecretMatch, SecretScanner, redact_matches};

/// Redacts every string value of `value` in place; true when one changed.
pub(crate) fn redact_value(scanner: &dyn SecretScanner, value: &mut Value) -> bool {
    redact_tree(scanner, value, None)
}

fn redact_tree(scanner: &dyn SecretScanner, value: &mut Value, key: Option<&str>) -> bool {
    match value {
        Value::String(s) => redact_keyed(scanner, s, key),
        Value::Array(items) => {
            let mut changed = false;
            for v in items {
                changed |= redact_tree(scanner, v, key);
            }
            changed
        }
        Value::Object(map) => {
            let mut changed = false;
            for (k, v) in map.iter_mut() {
                changed |= redact_tree(scanner, v, Some(k.as_str()));
            }
            changed
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// Redacts one text (content, a title, a provenance field) in place, as
/// text; true when it changed.
pub(crate) fn redact_string(scanner: &dyn SecretScanner, s: &mut String) -> bool {
    redact_text(scanner, s, None)
}

/// Redacts an optional string in place; true when it changed.
pub(crate) fn redact_opt(scanner: &dyn SecretScanner, s: &mut Option<String>) -> bool {
    s.as_mut().is_some_and(|s| redact_string(scanner, s))
}

/// True when the scanner finds a secret in `s`.
pub(crate) fn holds_secret(scanner: &dyn SecretScanner, s: &str) -> bool {
    !scanner.scan(s).is_empty()
}

/// Redacts a string value of a JSON tree.
fn redact_keyed(scanner: &dyn SecretScanner, s: &mut String, key: Option<&str>) -> bool {
    let embedded = redact_embedded_json(scanner, s);
    redact_text(scanner, s, key) || embedded
}

/// A string that holds a JSON object or array (version 1's CSV and Parquet
/// exports keep `metadata` and `tags` that way) is redacted value by value,
/// so its keys count; when a value changed, the string becomes the redacted
/// JSON, written compactly. Any other string is left to the text scan.
fn redact_embedded_json(scanner: &dyn SecretScanner, s: &mut String) -> bool {
    let t = s.trim_start();
    if !(t.starts_with('{') || t.starts_with('[')) {
        return false;
    }
    let Ok(mut v) = serde_json::from_str::<Value>(s) else {
        return false;
    };
    if !redact_value(scanner, &mut v) {
        return false;
    }
    *s = v.to_string();
    true
}

/// Redacts `s` as text; with a key, also what the scanner finds in
/// `key: s` past the key.
fn redact_text(scanner: &dyn SecretScanner, s: &mut String, key: Option<&str>) -> bool {
    let mut matches = scanner.scan(s);
    if let Some(key) = key {
        let prefix = key.len() + 2;
        let joined = format!("{key}: {s}");
        for m in scanner.scan(&joined) {
            if m.end > prefix {
                matches.push(SecretMatch {
                    kind: m.kind,
                    start: m.start.max(prefix) - prefix,
                    end: m.end - prefix,
                });
            }
        }
    }
    if matches.is_empty() {
        return false;
    }
    let r = redact_matches(s, matches);
    if r.text == *s {
        return false;
    }
    *s = r.text;
    true
}
