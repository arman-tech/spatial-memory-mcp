//! `redaction_cases.json` and `redaction_patterns.json`: the secret scanner
//! (SPEC section 16, Pipeline in the daemon, step 3; AT-9).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};
use sm_core::traits::SecretScanner;

use super::checker::{json_eq, text};
use super::{Suite, case_id, fixture, list, short};
use crate::secrets::{
    FALSE_POSITIVE_REGEX, KEY_PATTERNS, RegexSecretScanner, V3_VALUE_PATTERNS, VALUE_PATTERNS,
    python_placeholder,
};

static PYTHON_PLACEHOLDER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[REDACTED_([A-Z_]+)\]").expect("a valid expression"));

/// Python's `[REDACTED_<KIND>]` written as version 3's `[REDACTED:<kind>]`
/// (the fixture's `v3_note`).
fn v3(s: &str) -> String {
    PYTHON_PLACEHOLDER
        .replace_all(s, |c: &regex::Captures<'_>| {
            format!("[REDACTED:{}]", c[1].to_ascii_lowercase())
        })
        .into_owned()
}

/// The scanner's result as version 1's `RedactionResult`, with the kinds.
fn observed(input: &str) -> Value {
    let r = RegexSecretScanner::new().report(input);
    json!({
        "redacted_text": r.redaction.text,
        "redaction_count": r.count(),
        "should_skip": r.should_skip,
        "kinds": r.redaction.kinds,
    })
}

/// The strings of a list, `$concat` and `$repeat` expanded: a fake
/// credential in `expect` is built at run time, like the input (B-215).
fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|l| l.iter().map(|s| v3(&text(s))).collect())
        .unwrap_or_default()
}

/// The dict-form `expect` of `redaction_cases.json` (its `expect_keys`).
fn check_expect(obs: &Value, expect: &Value) -> Vec<String> {
    let text = obs["redacted_text"].as_str().unwrap_or_default();
    let count = obs["redaction_count"].as_u64().unwrap_or(0);
    let mut problems = Vec::new();
    let Some(expect) = expect.as_object() else {
        return vec![format!("expect is not an object: {expect}")];
    };
    for (key, value) in expect {
        let ok = match key.as_str() {
            "contains" => strings(value).iter().all(|s| text.contains(s.as_str())),
            "not_contains" => strings(value).iter().all(|s| !text.contains(s.as_str())),
            "contains_any" => strings(value).iter().any(|s| text.contains(s.as_str())),
            "redaction_count_min" => value.as_u64().is_some_and(|n| count >= n),
            "redaction_count" => value.as_u64() == Some(count),
            "should_skip" => obs["should_skip"] == *value,
            "redacted_text" => value.as_str().map(v3).as_deref() == Some(text),
            other => {
                problems.push(format!("unknown expect key {other}"));
                continue;
            }
        };
        if !ok {
            problems.push(format!("{key} {value}: got {}", short(obs)));
        }
    }
    problems
}

/// The AT-9 properties for one input: every offset on a `char` boundary,
/// every kind in `v3_mapping`, nothing found in the redacted text, and a
/// second redaction that changes nothing.
fn stability(input: &str, kinds: &[String]) -> Vec<String> {
    let scanner = RegexSecretScanner::new();
    let mut problems = Vec::new();
    for m in scanner.scan(input) {
        if !(m.start < m.end
            && m.end <= input.len()
            && input.is_char_boundary(m.start)
            && input.is_char_boundary(m.end))
        {
            problems.push(format!("match {}..{} is not a char range", m.start, m.end));
        }
        if !kinds.contains(&m.kind) {
            problems.push(format!("kind {} is not in v3_mapping", m.kind));
        }
    }
    let once = scanner.redact(input);
    let left = scanner.scan(&once.text);
    if !left.is_empty() {
        problems.push(format!(
            "scan(redact(x).text) found {left:?} in {:?}",
            once.text
        ));
    }
    let twice = scanner.redact(&once.text);
    if twice.text != once.text {
        problems.push(format!(
            "redacting twice changed {:?} to {:?}",
            once.text, twice.text
        ));
    }
    let report = scanner.report(&once.text);
    if report.count() != 0 {
        problems.push(format!("report(redact(x).text) counted {}", report.count()));
    }
    problems
}

/// Every case of `redaction_cases.json`, then the AT-9 stability properties
/// over all of them.
#[test]
fn at9_redaction_cases() {
    let f = fixture("redaction_cases.json");
    let kinds: Vec<String> = fixture("redaction_patterns.json")["v3_mapping"]["kinds"]
        .as_array()
        .expect("v3_mapping.kinds")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let mut suite = Suite::new("redaction_cases.json");
    let mut inputs = Vec::new();
    for case in list(&f, "cases") {
        let id = case_id(case);
        let input = text(&case["input"]);
        let obs = observed(&input);
        let mut problems = check_expect(&obs, &case["expect"]);
        problems.extend(stability(&input, &kinds));
        suite.case(&id, problems);
        let py = &case["computed_result"];
        let py_v3 = json!({
            "redacted_text": py["redacted_text"].as_str().map(v3),
            "redaction_count": py["redaction_count"],
            "should_skip": py["should_skip"],
            "kinds": py["kinds"],
        });
        suite.computed(&id, "computed_result", &obs, &py_v3);
        inputs.push(input);
    }

    // The stability properties once more, over every input concatenated
    // and over each redacted output, so markers next to fresh secrets are
    // covered too.
    let all = inputs.join("\n");
    suite.case("stability_all_inputs_joined", stability(&all, &kinds));
    let scanner = RegexSecretScanner::new();
    let redacted_then_raw: String = inputs
        .iter()
        .map(|i| format!("{} {i}", scanner.redact(i).text))
        .collect::<Vec<_>>()
        .join("\n");
    suite.case(
        "stability_redacted_next_to_raw",
        stability(&redacted_then_raw, &kinds),
    );

    suite.finish(&[]);
}

/// AT-9 (B-215): `redaction_cases.json` stores no whole fake credential.
/// Each one is built at run time from `$concat` and `$repeat` parts, so the
/// scanner, run over the file's own text, finds nothing but the fixture's
/// `[REDACTED_<KIND>]` markers. Every case with expected kinds still holds
/// a credential once built. Prints kinds and offsets, never a value.
#[test]
fn at9_redaction_fixture_stores_no_whole_credential() {
    let scanner = RegexSecretScanner::new();
    let path = super::fixture_path("redaction_cases.json");
    let raw = std::fs::read_to_string(&path).expect("the fixture");
    let whole: Vec<String> = scanner
        .scan(&raw)
        .into_iter()
        .filter(|m| !raw[m.start..m.end].contains("[REDACTED_"))
        .map(|m| format!("{} at byte {}", m.kind, m.start))
        .collect();
    assert!(whole.is_empty(), "whole credentials in the file: {whole:?}");

    let f = fixture("redaction_cases.json");
    let mut built = 0;
    for case in list(&f, "cases") {
        let expected = !case["computed_result"]["kinds"]
            .as_array()
            .expect("kinds")
            .is_empty();
        let found = !scanner.scan(&text(&case["input"])).is_empty();
        assert_eq!(
            found,
            expected,
            "{}: a credential once built",
            case_id(case)
        );
        built += usize::from(found);
    }
    assert_eq!(built, 33, "the fixture's credential cases");
}

/// The 19 key patterns, the 3 value patterns and the false positive filter
/// against `redaction_patterns.json`: order, tier, kind (`v3_kind`), the
/// private key flag, the expression with Python's flags written inline, and
/// the placeholder.
#[test]
fn at9_redaction_patterns_table() {
    let f = fixture("redaction_patterns.json");
    let mut suite = Suite::new("redaction_patterns.json");
    let with_flags = |p: &Value| -> String {
        if let Some(r) = p.get("rust_regex").and_then(Value::as_str) {
            return r.to_owned();
        }
        let mut prefix = String::new();
        for flag in p["flags"].as_array().into_iter().flatten() {
            match flag.as_str() {
                Some("IGNORECASE") => prefix.push_str("(?i)"),
                Some("DOTALL") => prefix.push_str("(?s)"),
                other => panic!("unexpected flag {other:?}"),
            }
        }
        format!("{prefix}{}", p["regex"].as_str().unwrap_or_default())
    };

    let key = list(&f, "tier1_tier2");
    let mut count = Vec::new();
    if key.len() != KEY_PATTERNS.len() {
        count.push(format!(
            "fixture {} key patterns, crate {}",
            key.len(),
            KEY_PATTERNS.len()
        ));
    }
    for (i, p) in key.iter().enumerate() {
        let id = case_id(p);
        let Some(ours) = KEY_PATTERNS.get(i) else {
            suite.case(&id, vec![format!("no key pattern at {i}")]);
            continue;
        };
        let mut problems = Vec::new();
        if p["order"].as_u64() != Some(i as u64) {
            problems.push(format!("order {} is not {i}", p["order"]));
        }
        if ours.name != id {
            problems.push(format!("name {} != {id}", ours.name));
        }
        if p["tier"].as_u64() != Some(u64::from(ours.tier)) {
            problems.push(format!("tier {} != {}", ours.tier, p["tier"]));
        }
        if p["v3_kind"].as_str() != Some(ours.kind) {
            problems.push(format!("kind {} != {}", ours.kind, p["v3_kind"]));
        }
        if p["is_private_key"].as_bool() != Some(ours.private_key) {
            problems.push(format!(
                "private_key {} != {}",
                ours.private_key, p["is_private_key"]
            ));
        }
        if ours.url_password != p.get("rust_replacement").is_some() {
            problems.push(format!(
                "url_password {} disagrees with the fixture",
                ours.url_password
            ));
        }
        let expected = with_flags(p);
        if ours.regex != expected {
            problems.push(format!("regex {:?} != {expected:?}", ours.regex));
        }
        if !ours.url_password
            && p["python_replacement"].as_str() != Some(&python_placeholder(ours.kind))
        {
            problems.push(format!(
                "placeholder {} != {}",
                python_placeholder(ours.kind),
                p["python_replacement"]
            ));
        }
        if p["v3_placeholder"].as_str() != Some(&format!("[REDACTED:{}]", ours.kind)) {
            problems.push(format!("v3 placeholder {}", p["v3_placeholder"]));
        }
        suite.case(&id, problems);
    }
    suite.case("tier1_tier2_count", count);

    let value = list(&f, "tier3");
    let mut count = Vec::new();
    if value.len() != VALUE_PATTERNS.len() {
        count.push(format!(
            "fixture {} value patterns, crate {}",
            value.len(),
            VALUE_PATTERNS.len()
        ));
    }
    for (i, p) in value.iter().enumerate() {
        let id = format!("tier3.{}", case_id(p));
        let Some(ours) = VALUE_PATTERNS.get(i) else {
            suite.case(&id, vec![format!("no value pattern at {i}")]);
            continue;
        };
        let mut problems = Vec::new();
        if p["name"].as_str() != Some(ours.name) {
            problems.push(format!("name {} != {}", ours.name, p["name"]));
        }
        if p["v3_kind"].as_str() != Some(ours.kind) {
            problems.push(format!("kind {} != {}", ours.kind, p["v3_kind"]));
        }
        if p["value_group"].as_u64() != Some(1) {
            problems.push(format!("value group {}", p["value_group"]));
        }
        let expected = with_flags(p);
        if ours.regex != expected {
            problems.push(format!("regex {:?} != {expected:?}", ours.regex));
        }
        if p["placeholder"].as_str() != Some(&python_placeholder(ours.kind)) {
            problems.push(format!("placeholder {}", p["placeholder"]));
        }
        suite.case(&id, problems);
    }
    suite.case("tier3_count", count);

    let fp = &f["false_positive_filter"];
    let mut problems = Vec::new();
    if fp["rust_regex"].as_str() != Some(FALSE_POSITIVE_REGEX) {
        problems.push(format!(
            "false positive filter {FALSE_POSITIVE_REGEX:?} != {}",
            fp["rust_regex"]
        ));
    }
    suite.case("false_positive_filter", problems);

    // Every kind the patterns can report is in `v3_mapping`, and every
    // mapped kind is reported by some pattern.
    let mapped: Vec<&str> = f["v3_mapping"]["kinds"]
        .as_array()
        .expect("v3_mapping.kinds")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut ours: Vec<&str> = KEY_PATTERNS
        .iter()
        .map(|p| p.kind)
        .chain(VALUE_PATTERNS.iter().map(|p| p.kind))
        .chain(V3_VALUE_PATTERNS.iter().map(|p| p.kind))
        .collect();
    ours.sort_unstable();
    ours.dedup();
    let mut problems = Vec::new();
    if !json_eq(&json!(ours), &json!(mapped)) {
        problems.push(format!("kinds {ours:?} != v3_mapping {mapped:?}"));
    }
    suite.case("v3_mapping", problems);
    suite.finish(&[]);
}

/// Version 3's AWS secret access key pattern (B-86; SPEC 9 "Secret
/// redaction on write"; migration fixture record 4, AT-8): the environment,
/// credentials-file, JSON and short forms are redacted, only the value is
/// replaced, placeholders and prose are left alone, and the AT-9 stability
/// properties hold.
#[test]
fn at9_aws_secret_access_key_is_redacted() {
    let secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    let kinds: Vec<String> = fixture("redaction_patterns.json")["v3_mapping"]["kinds"]
        .as_array()
        .expect("v3_mapping.kinds")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let scanner = RegexSecretScanner::new();
    let mut problems = Vec::new();

    let redacted = [
        (
            format!("export AWS_SECRET_ACCESS_KEY={secret}"),
            "export AWS_SECRET_ACCESS_KEY=[REDACTED:aws_key]".to_owned(),
            1,
        ),
        (
            format!("[default]\naws_secret_access_key = {secret}\n"),
            "[default]\naws_secret_access_key = [REDACTED:aws_key]\n".to_owned(),
            1,
        ),
        (
            format!(r#"{{"aws_secret_access_key": "{secret}"}}"#),
            r#"{"aws_secret_access_key": "[REDACTED:aws_key]"}"#.to_owned(),
            1,
        ),
        (
            format!("SecretAccessKey: {secret}"),
            "SecretAccessKey: [REDACTED:aws_key]".to_owned(),
            1,
        ),
        (
            format!("AWS_SECRET_KEY='{secret}'"),
            "AWS_SECRET_KEY='[REDACTED:aws_key]'".to_owned(),
            1,
        ),
        // Record 4 of the migration fixture: the key id and the secret.
        (
            format!(
                "Example from the AWS docs: AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE AWS_SECRET_ACCESS_KEY={secret}"
            ),
            "Example from the AWS docs: AWS_ACCESS_KEY_ID=[REDACTED:aws_key] AWS_SECRET_ACCESS_KEY=[REDACTED:aws_key]".to_owned(),
            2,
        ),
    ];
    for (input, expected, count) in &redacted {
        let report = scanner.report(input);
        if report.redaction.text != *expected {
            problems.push(format!(
                "{input:?} gave {:?}, not {expected:?}",
                report.redaction.text
            ));
        }
        if report.count() != *count {
            problems.push(format!("{input:?} counted {}, not {count}", report.count()));
        }
        if report.redaction.kinds.iter().any(|k| k != "aws_key") {
            problems.push(format!("{input:?} kinds {:?}", report.redaction.kinds));
        }
        problems.extend(stability(input, &kinds));
    }

    let kept = [
        "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}",
        "aws_secret_access_key = <your-secret-access-key>",
        "aws_secret_access_key: changeme",
        "Rotate the secret access key every 90 days.",
    ];
    for input in kept {
        let report = scanner.report(input);
        if report.count() != 0 || report.redaction.text != input {
            problems.push(format!(
                "{input:?} should be kept, gave {:?}",
                report.redaction.text
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// B-137 (SPEC 9 "Secret redaction on write", the cloud access keys): the
/// AWS secret access key is redacted in the forms version 3's first pattern
/// missed, with a hyphenated keyword, Ruby's `=>`, a space, an XML element
/// and Terraform's `secret_key`. Only the value is replaced, a value that
/// is not 40 key characters is left alone in the space, XML and Terraform
/// forms, and the AT-9 stability properties hold.
#[test]
fn s9_aws_secret_access_key_forms_with_other_separators_are_redacted() {
    let secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    assert_eq!(secret.len(), 40);
    let kinds: Vec<String> = fixture("redaction_patterns.json")["v3_mapping"]["kinds"]
        .as_array()
        .expect("v3_mapping.kinds")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let scanner = RegexSecretScanner::new();
    let mut problems = Vec::new();
    let r = "[REDACTED:aws_key]";

    let redacted = [
        (
            format!("aws-secret-access-key: {secret}"),
            format!("aws-secret-access-key: {r}"),
        ),
        (
            format!("aws s3 ls --aws-secret-access-key={secret}"),
            format!("aws s3 ls --aws-secret-access-key={r}"),
        ),
        (
            format!("credentials = {{ :secret_access_key => '{secret}' }}"),
            format!("credentials = {{ :secret_access_key => '{r}' }}"),
        ),
        (
            format!("aws configure set aws_secret_access_key {secret}"),
            format!("aws configure set aws_secret_access_key {r}"),
        ),
        (
            format!("<SecretAccessKey>{secret}</SecretAccessKey>"),
            format!("<SecretAccessKey>{r}</SecretAccessKey>"),
        ),
        (
            format!("provider \"aws\" {{\n  secret_key = \"{secret}\"\n}}\n"),
            format!("provider \"aws\" {{\n  secret_key = \"{r}\"\n}}\n"),
        ),
        (format!("secret_key {secret}"), format!("secret_key {r}")),
    ];
    for (input, expected) in &redacted {
        let report = scanner.report(input);
        if report.redaction.text != *expected {
            problems.push(format!(
                "{input:?} gave {:?}, not {expected:?}",
                report.redaction.text
            ));
        }
        if report.count() != 1 {
            problems.push(format!("{input:?} counted {}, not 1", report.count()));
        }
        if report.redaction.kinds.iter().any(|k| k != "aws_key") {
            problems.push(format!("{input:?} kinds {:?}", report.redaction.kinds));
        }
        problems.extend(stability(input, &kinds));
    }

    let kept = [
        // Not 40 key characters after a space, an element or `secret_key`.
        "aws configure set aws_secret_access_key from-the-vault".to_owned(),
        "<SecretAccessKey>short</SecretAccessKey>".to_owned(),
        "secret_key = var.aws_secret_key".to_owned(),
        format!("secret_key {secret}A"),
        // Prose.
        format!("The secret key {secret} is an example."),
    ];
    for input in &kept {
        let report = scanner.report(input);
        if report.count() != 0 || report.redaction.text != *input {
            problems.push(format!(
                "{input:?} should be kept, gave {:?}",
                report.redaction.text
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
