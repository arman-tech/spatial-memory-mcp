//! Tests for sm-capture.
//!
//! Each `fixtures/capture` file has its own test, named for its rule, that
//! runs every case through one checker ([`checker`]) and reports every
//! failing case together. A [`Suite`] counts the cases ported, passing and
//! not applicable (each with its reason), and lists every place where the
//! result satisfies `expect` but differs from a `computed_*` value. Those
//! differences are decisions: each suite names the cases it expects to
//! differ, and fails when a difference appears or disappears unannounced.

mod acceptance;
mod budgets;
mod checker;
mod content_extractor;
mod hook_helpers;
mod hook_payloads;
mod lean;
mod overlap;
mod pipeline_notes;
mod quote;
mod redaction;
mod replay;
mod rules;
mod selection;
mod signals;
mod transcript_cases;
mod turns;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};
use sm_core::config::CaptureSettings;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::HookEvent;

use crate::candidate::Candidate;
use crate::secrets::python_placeholder;

/// The capture fixtures, read in place.
pub const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/capture");

/// The path of a capture fixture.
pub fn fixture_path(rel: &str) -> PathBuf {
    Path::new(FIXTURES).join(rel)
}

/// A capture fixture file, parsed.
pub fn fixture(rel: &str) -> Value {
    let path = fixture_path(rel);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// The list at `key` of a fixture section.
pub fn list<'a>(section: &'a Value, key: &str) -> &'a [Value] {
    section
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixture section has no list {key:?}"))
}

/// A case's identifier: its `name`, `id` or `tool_name`.
pub fn case_id(case: &Value) -> String {
    ["name", "id", "tool_name"]
        .iter()
        .find_map(|k| case.get(*k).and_then(Value::as_str))
        .unwrap_or("<unnamed>")
        .to_owned()
}

/// A project rooted at `root`, capture enabled.
pub fn project(root: &Path) -> ProjectIdentity {
    let root_path = root.to_string_lossy().replace('\\', "/");
    ProjectIdentity {
        root: root.to_path_buf(),
        root_key: root_path.to_lowercase(),
        root_path,
        git_remote: None,
        repo: None,
        name: "fixture-project".to_owned(),
        capture_enabled: true,
        warnings: Vec::new(),
    }
}

/// The tools versions 1 and 2 captured by default. Version 3 captures none
/// by default (B-160), so a test of tool capture names these, as a user who
/// wants tool capture does in `capture.tools`.
pub const V1_TOOLS: [&str; 4] = ["Edit", "Write", "Bash", "NotebookEdit"];

/// The tool settings of [`settings_with_tools`] with [`V1_TOOLS`] (B-160).
pub fn v1_tool_settings() -> CaptureSettings {
    settings_with_tools(&V1_TOOLS)
}

/// The default capture settings with `capture.tool_results` on, so that
/// text read from files and command output is offered. It is off by default
/// (B-236), so a test of that text turns it on, as a user who wants it does.
pub fn tool_results_on() -> CaptureSettings {
    CaptureSettings {
        tool_results: true,
        ..CaptureSettings::default()
    }
}

/// The settings of a user who wants tool capture: `tools` in place of the
/// default list, and `capture.tool_results` on, without which no tool but
/// those that write a file is captured (B-160, B-221, B-236).
pub fn settings_with_tools(tools: &[&str]) -> CaptureSettings {
    CaptureSettings {
        tools: tools.iter().map(|t| (*t).to_owned()).collect(),
        ..tool_results_on()
    }
}

/// A new temporary directory.
pub fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("a temporary directory")
}

/// Copies fixture files into `dir`, byte for byte. The transcript reader
/// rejects any path holding `..`, which the in-place fixture path does.
pub fn copy_fixtures(dir: &Path, rels: &[&str]) {
    for rel in rels {
        let name = Path::new(rel).file_name().expect("a file name");
        std::fs::copy(fixture_path(rel), dir.join(name))
            .unwrap_or_else(|e| panic!("cannot copy {rel}: {e}"));
    }
}

/// Replaces `{TMP}` in every string of `v`.
pub fn with_tmp(v: &Value, tmp: &str) -> Value {
    match v {
        Value::String(s) => Value::String(s.replace("{TMP}", tmp)),
        Value::Array(items) => Value::Array(items.iter().map(|x| with_tmp(x, tmp)).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, x)| (k.clone(), with_tmp(x, tmp)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The tally of one fixture file.
pub struct Suite {
    file: String,
    ported: usize,
    failed: Vec<(String, Vec<String>)>,
    not_applicable: Vec<(String, String)>,
    differences: Vec<(String, String)>,
}

impl Suite {
    /// A tally for fixture file `file`.
    pub fn new(file: &str) -> Self {
        Suite {
            file: file.to_owned(),
            ported: 0,
            failed: Vec::new(),
            not_applicable: Vec::new(),
            differences: Vec::new(),
        }
    }

    /// A ported case; it passes when `problems` is empty.
    pub fn case(&mut self, id: &str, problems: Vec<String>) {
        self.ported += 1;
        if !problems.is_empty() {
            self.failed.push((id.to_owned(), problems));
        }
    }

    /// A case that does not apply to this crate, with the reason.
    pub fn not_applicable(&mut self, id: &str, reason: &str) {
        self.not_applicable.push((id.to_owned(), reason.to_owned()));
    }

    /// Compares a result with Python's computed value and records a
    /// difference. Returns whether they agree.
    pub fn computed(&mut self, id: &str, what: &str, v3: &Value, python: &Value) -> bool {
        let same = checker::json_eq(v3, python);
        if !same {
            self.difference(
                id,
                format!("{what}: v3 {}, Python {}", short(v3), short(python)),
            );
        }
        same
    }

    /// Records a difference from a computed value.
    pub fn difference(&mut self, id: &str, detail: String) {
        self.differences.push((id.to_owned(), detail));
    }

    /// Prints the tally and panics when a case failed, or when the cases
    /// that differ from Python's computed values are not exactly
    /// `expected_differences`.
    pub fn finish(self, expected_differences: &[&str]) {
        let passing = self.ported - self.failed.len();
        println!(
            "COVERAGE {}: ported {}, passing {}, not applicable {}",
            self.file,
            self.ported,
            passing,
            self.not_applicable.len()
        );
        for (id, reason) in &self.not_applicable {
            println!("  n/a {id}: {reason}");
        }
        for (id, detail) in &self.differences {
            println!("  differs {id}: {detail}");
        }
        let found: BTreeSet<&str> = self.differences.iter().map(|(id, _)| id.as_str()).collect();
        let expected: BTreeSet<&str> = expected_differences.iter().copied().collect();
        let mut report = Vec::new();
        for (id, problems) in &self.failed {
            report.push(format!("FAILED {id}:\n    {}", problems.join("\n    ")));
        }
        for id in found.difference(&expected) {
            report.push(format!("UNANNOUNCED DIFFERENCE {id}"));
        }
        for id in expected.difference(&found) {
            report.push(format!("ANNOUNCED DIFFERENCE NOT FOUND {id}"));
        }
        assert!(
            report.is_empty(),
            "{}: {} problem(s)\n{}",
            self.file,
            report.len(),
            report.join("\n")
        );
    }
}

/// Claude Code's name for an event, as version 1 wrote `source_hook` and a
/// transcript context's `hook_event`. Version 3's provenance records the
/// command-line name instead ([`HookEvent::as_str`]).
pub fn claude_name(event: HookEvent) -> &'static str {
    match event {
        HookEvent::SessionStart => "SessionStart",
        HookEvent::PostToolUse => "PostToolUse",
        HookEvent::PreCompact => "PreCompact",
        HookEvent::Stop => "Stop",
        HookEvent::UserPromptSubmit => "UserPromptSubmit",
    }
}

/// A candidate as version 1's queue file, without the fields this crate
/// does not own: `version` and `timestamp` (the spool's), `project_root_dir`
/// (the candidate belongs to `CaptureInput.project`) and `client` (stream D
/// normalises a Cursor payload first). The context is built from the
/// provenance as version 1's `_build_context` and `_build_transcript_context`
/// built it from the payload.
pub fn queue_fields(c: &Candidate) -> Value {
    let p = &c.provenance;
    let mut context = Map::new();
    if c.source_hook == HookEvent::PostToolUse {
        for (key, value) in [
            ("tool_name", &p.tool),
            ("session_id", &p.session_id),
            ("file_path", &p.file_path),
        ] {
            if let Some(v) = value {
                context.insert(key.to_owned(), json!(v));
            }
        }
    } else {
        if let Some(s) = &p.session_id {
            context.insert("session_id".to_owned(), json!(s));
        }
        context.insert("hook_event".to_owned(), json!(claude_name(c.source_hook)));
        if let Some(t) = p.extra.as_ref().and_then(|e| e.get("trigger")) {
            context.insert("trigger".to_owned(), t.clone());
        }
    }
    json!({
        "content": c.content,
        "source_hook": claude_name(c.source_hook),
        "suggested_namespace": c.v1_namespace,
        "suggested_tags": c.tags,
        "suggested_importance": c.importance,
        "signal_tier": c.tier.number(),
        "signal_patterns_matched": c.patterns,
        "context": context,
    })
}

/// The kind of transcript text a candidate came from, as its provenance
/// records it (B-179): `reply`, `prompt`, `summary`, `tool_input` or
/// `tool_result`; empty for a PostToolUse candidate.
pub fn source_of(c: &Candidate) -> &str {
    c.provenance
        .extra
        .as_ref()
        .and_then(|e| e.get("source"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// Version 1's queue file without the fields [`queue_fields`] leaves out.
pub fn without_unowned(queue_file: &Value) -> Value {
    let mut map = queue_file.as_object().cloned().unwrap_or_default();
    for key in ["version", "timestamp", "project_root_dir", "client"] {
        map.remove(key);
    }
    Value::Object(map)
}

/// `text` with each version 3 marker `[REDACTED:<kind>]` written as version
/// 1's placeholder `[REDACTED_<KIND>]`, to compare with version 1's output.
pub fn python_form(text: &str) -> String {
    static MARKER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\[REDACTED:([a-z_]+)\]").expect("a valid expression"));
    MARKER
        .replace_all(text, |c: &regex::Captures<'_>| python_placeholder(&c[1]))
        .into_owned()
}

/// A value written short enough for a failure message.
pub fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 240 {
        let head: String = s.chars().take(240).collect();
        format!("{head}... ({} chars)", s.chars().count())
    } else {
        s
    }
}
