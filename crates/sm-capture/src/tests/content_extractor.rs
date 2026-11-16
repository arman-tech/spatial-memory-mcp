//! `content_extractor_cases.json`: version 2's PostToolUse extraction with
//! one 200,000-character safety cap, and the tool skip rules (SPEC section
//! 16, Pipeline in the daemon, step 1; AT-17).

use serde_json::{Value, json};

use super::checker::{json_eq, resolve};
use super::{Suite, case_id, fixture, list, short};
use crate::extract::{
    MAX_CAPTURE_CHARS, SKIP_TOOLS, SPATIAL_MEMORY_PREFIX, TRUNCATION_MARKER, V1_TOOL_NAMES,
    extract_bash, extract_content, extract_edit, extract_mcp_tool, extract_notebook_edit,
    extract_write, safe_get, skip_rule, tool_response_text, truncate,
};

/// The `tool_response` argument as the string this crate's extractors take.
/// A string is itself. Python's extractors were also called with `None`,
/// which they treat as empty; the Rust functions take `&str`, so `null`
/// becomes `""` here. Any other value is the JSON text the pipeline makes of
/// it ([`tool_response_text`]), where Python's `str()` wrote a Python
/// literal: that is a recorded difference.
fn response(args: &Value) -> String {
    match args.get("tool_response") {
        None | Some(Value::Null) => String::new(),
        Some(v) => tool_response_text(&json!({ "tool_response": v })),
    }
}

/// Runs one of the fixture's functions on its (resolved) `args`.
fn run(function: &str, args: &Value) -> String {
    let input = &args["tool_input"];
    let tool_name = args.get("tool_name").and_then(Value::as_str).unwrap_or("");
    match function {
        "_truncate" => {
            let max = args["max_len"].as_u64().expect("max_len") as usize;
            truncate(args["text"].as_str().expect("text"), max).0
        }
        "_safe_get" => safe_get(
            &args["d"],
            args["key"].as_str().expect("key"),
            args.get("default").and_then(Value::as_str).unwrap_or(""),
        ),
        "_extract_edit" => extract_edit(input),
        "_extract_write" => extract_write(input),
        "_extract_bash" => extract_bash(input, &response(args)),
        "_extract_notebook_edit" => extract_notebook_edit(input),
        "_extract_mcp_tool" => extract_mcp_tool(input, &response(args), tool_name),
        "extract_content" => extract_content(tool_name, input, &response(args)).text,
        other => panic!("unknown function {other}"),
    }
}

/// The dict-form `expect` of the extractor cases.
fn check_expect(out: &str, expect: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let Some(expect) = expect.as_object() else {
        return vec![format!("expect is not an object: {expect}")];
    };
    let len = out.chars().count() as u64;
    for (key, value) in expect {
        let value = resolve(value);
        let ok = match key.as_str() {
            "equals" => value.as_str() == Some(out),
            "contains" => value.as_array().is_some_and(|l| {
                l.iter()
                    .all(|s| s.as_str().is_some_and(|s| out.contains(s)))
            }),
            "not_contains" => value.as_array().is_some_and(|l| {
                l.iter()
                    .all(|s| s.as_str().is_some_and(|s| !out.contains(s)))
            }),
            "ends_with" => value.as_str().is_some_and(|s| out.ends_with(s)),
            "not_ends_with" => value.as_str().is_some_and(|s| !out.ends_with(s)),
            "length" => value.as_u64() == Some(len),
            "length_max" => value.as_u64().is_some_and(|m| len <= m),
            other => {
                problems.push(format!("unknown expect key {other}"));
                continue;
            }
        };
        if !ok {
            problems.push(format!(
                "{key} {}: got {} ({len} chars)",
                short(&value),
                short(&Value::String(out.to_owned()))
            ));
        }
    }
    problems
}

fn skip_observed(tool_name: &str) -> Value {
    match skip_rule(tool_name) {
        Some(skip) => json!({"skip": true, "reason": skip.reason()}),
        None => json!({"skip": false, "reason": ""}),
    }
}

/// Every extractor case, the boundary probes, the master cases version 2
/// superseded, the contract constants and the skip rules.
#[test]
fn s16_content_extractor_cases() {
    let f = fixture("content_extractor_cases.json");
    let mut suite = Suite::new("content_extractor_cases.json");

    for case in list(&f, "cases") {
        let id = case_id(case);
        let function = case["function"].as_str().expect("function");
        let args = resolve(&case["args"]);
        let out = run(function, &args);
        suite.case(&id, check_expect(&out, &case["expect"]));
        let observed = Value::String(out);
        suite.computed(
            &id,
            "computed_result",
            &observed,
            &resolve(&case["computed_result"]),
        );
        if let Some(eq) = case
            .get("equivalent_extract_content")
            .filter(|v| !v.is_null())
        {
            let tool_name = eq["tool_name"].as_str().expect("tool_name");
            let through = extract_content(tool_name, &args["tool_input"], &response(&args)).text;
            suite.computed(
                &id,
                "equivalent_extract_content",
                &Value::String(through),
                &resolve(&eq["computed_result"]),
            );
        }
    }

    for probe in list(&f, "boundary_probes") {
        let id = case_id(probe);
        let args = resolve(&probe["args"]);
        let out = run(probe["function"].as_str().expect("function"), &args);
        suite.computed(
            &id,
            "computed_length",
            &json!(out.chars().count()),
            &probe["computed_length"],
        );
        suite.computed(
            &id,
            "computed_result",
            &Value::String(out),
            &resolve(&probe["computed_result"]),
        );
        suite.case(&id, Vec::new());
    }

    // Master's per-field and total caps, which version 2 removed: the
    // result must be version 2's, so master's own expectation (and its
    // computed value) is expected to differ.
    for case in list(&f, "superseded_by_v2") {
        let id = case_id(case);
        let args = resolve(&case["args"]);
        let out = Value::String(run(case["function"].as_str().expect("function"), &args));
        let v2 = resolve(&case["computed_result_v2"]);
        let mut problems = Vec::new();
        if !json_eq(&out, &v2) {
            problems.push(format!("v2 result {} != {}", short(&out), short(&v2)));
        }
        suite.case(&id, problems);
        suite.computed(
            &id,
            "computed_result_master",
            &out,
            &resolve(&case["computed_result_master"]),
        );
    }

    let contract = &f["contract_v2"];
    let mut problems = Vec::new();
    if contract["safety_cap"].as_u64() != Some(MAX_CAPTURE_CHARS as u64) {
        problems.push(format!(
            "safety cap {MAX_CAPTURE_CHARS} != {}",
            contract["safety_cap"]
        ));
    }
    if TRUNCATION_MARKER != "..." {
        problems.push(format!("marker {TRUNCATION_MARKER:?}"));
    }
    suite.case("contract_v2.safety_cap", problems);

    // No per-field cap: each extractor passes 300,000 characters through.
    let big = "q".repeat(300_000);
    let mut problems = Vec::new();
    let lens = [
        extract_edit(&json!({"new_string": big})).chars().count(),
        extract_write(&json!({"content": big})).chars().count(),
        extract_notebook_edit(&json!({"new_source": big}))
            .chars()
            .count(),
        extract_bash(&json!({"command": "c"}), &big).chars().count(),
        extract_mcp_tool(&json!({}), &big, "").chars().count(),
    ];
    if lens.iter().any(|&n| n < 300_000) {
        problems.push(format!("a per-tool extractor cut its field: {lens:?}"));
    }
    let capped = extract_content("Write", &json!({"content": big}), "");
    if !(capped.cap_hit
        && capped.original_chars == 300_000
        && capped.text.chars().count() == MAX_CAPTURE_CHARS + TRUNCATION_MARKER.len())
    {
        problems.push(format!(
            "extract_content: cap_hit {}, original {}, length {}",
            capped.cap_hit,
            capped.original_chars,
            capped.text.chars().count()
        ));
    }
    suite.case("contract_v2.invariant_assertions", problems);

    // The dispatch table: the four named tools, any `mcp__` tool, and
    // nothing for any other name, Cursor's `MCP:<name>` included.
    let mut problems = Vec::new();
    for (tool, expected) in [
        ("Edit", "e"),
        ("Write", "w"),
        ("Bash", "Command: b\nOutput: r"),
        ("NotebookEdit", "n"),
        ("mcp__db__query", "Tool: mcp__db__query\nResponse: r"),
        ("MCP:run_query", ""),
        ("Glob", ""),
    ] {
        let input = json!({"new_string": "e", "content": "w", "command": "b", "new_source": "n"});
        let got = extract_content(tool, &input, "r");
        if got.text != expected || got.cap_hit {
            problems.push(format!("{tool}: {:?}", got.text));
        }
    }
    suite.case("contract_v2.dispatch", problems);

    let rules = &f["skip_rules"];
    for case in list(rules, "cases") {
        let id = format!(
            "skip_rules:{}",
            case["tool_name"].as_str().unwrap_or_default()
        );
        let obs = skip_observed(case["tool_name"].as_str().expect("tool_name"));
        let mut problems = Vec::new();
        if !json_eq(&obs, &case["expect"]) {
            problems.push(format!("expected {}, got {obs}", case["expect"]));
        }
        suite.case(&id, problems);
    }
    for probe in list(rules, "computed_probes") {
        let id = format!(
            "skip_rules.probe:{}",
            probe["tool_name"].as_str().unwrap_or_default()
        );
        let obs = skip_observed(probe["tool_name"].as_str().expect("tool_name"));
        suite.computed(&id, "computed_result", &obs, &probe["computed_result"]);
        suite.case(&id, Vec::new());
    }
    let mut problems = Vec::new();
    let mut fixture_skip: Vec<&str> = list(rules, "skip_tools")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    fixture_skip.sort_unstable();
    let mut ours = SKIP_TOOLS.to_vec();
    ours.sort_unstable();
    if ours != fixture_skip {
        problems.push(format!("skip tools {ours:?} != {fixture_skip:?}"));
    }
    if rules["spatial_memory_prefix"].as_str() != Some(SPATIAL_MEMORY_PREFIX) {
        problems.push(format!("prefix {SPATIAL_MEMORY_PREFIX}"));
    }
    let names: Vec<&str> = list(rules, "spatial_memory_tool_names")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if names != V1_TOOL_NAMES {
        problems.push(format!(
            "version 1 tool names {V1_TOOL_NAMES:?} != {names:?}"
        ));
    }
    for core in ["recall", "remember", "forget", "hybrid_recall", "nearby"] {
        if !V1_TOOL_NAMES.contains(&core) {
            problems.push(format!("{core} missing"));
        }
    }
    suite.case("skip_rules.constant_assertions", problems);

    suite.finish(&[
        "non_string_tool_response_coerced",
        "master_edit_truncation",
        "master_write_truncation",
        "master_bash_large_output_truncated",
        "master_total_cap_enforced",
    ]);
}
