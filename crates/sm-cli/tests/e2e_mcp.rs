//! `remember`, `recall` and `share` end to end through `spatial-memory mcp`
//! (stream K, part 1, items 6 and 10): AT-16 and AT-9 at the binary's level.
//! The daemon runs in this process on the sandbox's endpoint, with the fake
//! embedder and the production sharing adapter; every call goes through the
//! binary's MCP proxy as `tools/call`. The git commands use an empty global
//! configuration and no system one.

mod common;
#[path = "common/e2e.rs"]
mod e2e;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, MutexGuard};

use common::Sandbox;
use e2e::{Direct, Model, unique, v3_markers};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Value, json};
use sm_core::testing::{dot, fake_vector};
use sm_daemon::RECALL_COLLAPSE_COSINE as COLLAPSE_COSINE;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Runs git in `dir`, isolated from the user's and the system's settings.
fn git(dir: &Path, config: &Path, args: &[&str]) {
    let mut cmd = Command::new("git");
    cmd.args(args).current_dir(dir);
    for var in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
    ] {
        cmd.env_remove(var);
    }
    let out = cmd
        .env("GIT_CONFIG_GLOBAL", config)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_AUTHOR_NAME", "Test Author")
        .env("GIT_AUTHOR_EMAIL", "author@example.invalid")
        .env("GIT_COMMITTER_NAME", "Test Author")
        .env("GIT_COMMITTER_EMAIL", "author@example.invalid")
        .output()
        .expect("run git");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A git repository with no commits, under the sandbox's `{TMP}`.
fn repo(sb: &Sandbox, name: &str) -> PathBuf {
    let config = sb.tmp.join("gitconfig");
    std::fs::write(&config, "").unwrap();
    let dir = sb.tmp.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    git(&dir, &config, &["init", "--quiet", "--initial-branch=main"]);
    dir
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// `spatial-memory mcp` in `dir`, with an rmcp client attached.
async fn mcp_in(sb: &Sandbox, dir: &Path) -> RunningService<RoleClient, ()> {
    let mut cmd = sb.command(&["mcp"]);
    cmd.current_dir(dir).stderr(Stdio::null());
    let child = TokioChildProcess::new(tokio::process::Command::from(cmd)).expect("spawn mcp");
    ().serve(child).await.expect("the MCP session starts")
}

async fn call(client: &RunningService<RoleClient, ()>, tool: &str, args: Value) -> CallToolResult {
    let Value::Object(args) = args else {
        panic!("arguments are an object")
    };
    client
        .call_tool(CallToolRequestParams::new(tool.to_owned()).with_arguments(args))
        .await
        .expect("tools/call answers")
}

/// The structured result of a tool call that succeeded.
async fn call_ok(client: &RunningService<RoleClient, ()>, tool: &str, args: Value) -> Value {
    let result = call(client, tool, args).await;
    assert_ne!(result.is_error, Some(true), "{tool}: {result:?}");
    result
        .structured_content
        .unwrap_or_else(|| panic!("{tool} has structured content"))
}

/// The structured error of a tool call that failed.
async fn call_err(client: &RunningService<RoleClient, ()>, tool: &str, args: Value) -> Value {
    let result = call(client, tool, args).await;
    assert_eq!(result.is_error, Some(true), "{tool}: {result:?}");
    result
        .structured_content
        .unwrap_or_else(|| panic!("{tool}'s error has structured content"))
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|s| s.as_str().expect("a string").to_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Every file under `dir`, recursively; empty when it does not exist.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            out.extend(files_under(&path));
        } else {
            out.push(path);
        }
    }
    out
}

// --- AT-16: importance does not change recall's order ------------------------------

/// AT-16 through the binary (stream K item 6). Two memories that tie on
/// score (each holds two of the query's four words and three of its own,
/// so the fake embedder scores them the same) are remembered through
/// `spatial-memory mcp` with importance 0.2 and 0.9 in one folder and 0.9
/// and 0.2 in another. In both, `recall` (auto and vector) returns them in
/// the same order, the order of their ids (B-70), so importance did not
/// move either. They are not near-identical to each other, so recall shows
/// both (B-197).
#[test]
fn at16_importance_does_not_change_recall_order_through_mcp() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let daemon = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let direct = Direct::new(&sb, daemon.endpoint().clone());
    let q = "garden shed deploy checklist";
    let x = "garden shed notes for spring";
    let y = "deploy checklist notes for spring";
    let qv = fake_vector(q);
    assert_eq!(
        dot(&qv, &fake_vector(x)),
        dot(&qv, &fake_vector(y)),
        "two of the query's words each tie on score"
    );
    assert!(
        dot(&fake_vector(x), &fake_vector(y)) < COLLAPSE_COSINE,
        "not near-identical, so neither hides the other"
    );

    let mut orders = Vec::new();
    for (label, ix, iy) in [("lowfirst", 0.2, 0.9), ("highfirst", 0.9, 0.2)] {
        let dir = sb.tmp.join(unique(label));
        std::fs::create_dir_all(&dir).unwrap();
        let (id_x, id_y) = runtime().block_on(async {
            let client = mcp_in(&sb, &dir).await;
            let mut ids = Vec::new();
            for (text, importance) in [(x, ix), (y, iy)] {
                let r = call_ok(
                    &client,
                    "remember",
                    json!({"memories": [{"content": text, "importance": importance}]}),
                )
                .await;
                assert_eq!(r["memories"][0]["outcome"], "created", "{r}");
                ids.push(r["memories"][0]["id"].as_str().expect("an id").to_owned());
            }
            client.cancel().await.unwrap();
            (ids[0].clone(), ids[1].clone())
        });
        // The tie is ranked by id; X was remembered first, so its id is the
        // lower one and the order is [X, Y] whatever the importances.
        assert!(id_x < id_y, "{label}: ids in the order remembered");
        direct.wait_embedded();

        runtime().block_on(async {
            let client = mcp_in(&sb, &dir).await;
            for mode in [None, Some("vector")] {
                let mut args = json!({"query": q, "k": 5});
                if let Some(m) = mode {
                    args["mode"] = json!(m);
                }
                let r = call_ok(&client, "recall", args).await;
                let results = r["results"].as_array().expect("results");
                let ids: Vec<&str> = results
                    .iter()
                    .map(|m| m["id"].as_str().expect("an id"))
                    .collect();
                assert_eq!(
                    ids,
                    vec![id_x.as_str(), id_y.as_str()],
                    "{label} {mode:?}: {r}"
                );
                assert_eq!(results[0]["importance"], json!(ix), "{label}: {r}");
                assert_eq!(results[1]["importance"], json!(iy), "{label}: {r}");
                assert_eq!(
                    results[0]["score"], results[1]["score"],
                    "{label}: the tie holds through the binary"
                );
            }
            client.cancel().await.unwrap();
        });
        orders.push((ix, iy));
    }
    assert_eq!(orders.len(), 2, "both ways round were recalled");
    eprintln!("AT-16 through mcp: [X, Y] with importances 0.2/0.9 and 0.9/0.2");
}

// --- AT-9: remember redacts, share writes no credential ---------------------------

/// AT-9 through the binary (stream K item 10). Every case of
/// `redaction_cases.json` with a fake credential is remembered through
/// `spatial-memory mcp`, the reply lists the kinds, and `share` writes each
/// one with version 3's marker and without the credential. The empty input
/// is refused. A credential in a tag or an anchor cannot be redacted, so
/// `share` refuses it with `secret_detected` and writes nothing at all.
#[test]
fn at9_remember_redacts_and_share_writes_no_credential_through_mcp() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let _daemon = e2e::start_daemon(&sb.paths(), Model::Plain, None);
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/capture/redaction_cases.json"
    ));
    let f: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).expect("the fixture")).unwrap();
    let cases = f["cases"].as_array().expect("cases");

    let dir = repo(&sb, "vault");
    runtime().block_on(async {
        let client = mcp_in(&sb, &dir).await;
        let mut stored: Vec<(String, String, &Value)> = Vec::new();
        let mut duplicates = 0;
        for case in cases {
            let name = case["name"].as_str().expect("a name");
            let input = case["input"].as_str().expect("an input");
            let kinds = strings(&case["computed_result"]["kinds"]);
            if input.trim().is_empty() {
                let e = call_err(
                    &client,
                    "remember",
                    json!({"memories": [{"content": input}]}),
                )
                .await;
                assert_eq!(e["code"], "invalid_input", "{name}: {e}");
                assert!(
                    !e["next_action"].as_str().unwrap_or_default().is_empty(),
                    "{name}: {e}"
                );
                continue;
            }
            if kinds.is_empty() {
                continue;
            }
            let r = call_ok(
                &client,
                "remember",
                json!({"memories": [{"content": input}]}),
            )
            .await;
            let m = &r["memories"][0];
            assert_eq!(strings(&m["redactions"]), kinds, "{name}: {r}");
            let id = m["id"].as_str().expect("an id").to_owned();
            // Two inputs that redact to the same text are one memory: the
            // second is a duplicate and names the first one's id.
            match m["outcome"].as_str() {
                Some("created") => assert!(!stored.iter().any(|s| s.1 == id), "{name}: {r}"),
                Some("duplicate") => {
                    assert!(stored.iter().any(|s| s.1 == id), "{name}: {r}");
                    duplicates += 1;
                }
                other => panic!("{name}: outcome {other:?}"),
            }
            stored.push((name.to_owned(), id, case));
        }
        assert!(stored.len() >= 25, "{} credential cases", stored.len());

        let mut ids: Vec<&str> = stored.iter().map(|s| s.1.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len() + duplicates, stored.len());
        let mut written = Vec::new();
        for batch in ids.chunks(100) {
            let shared = call_ok(&client, "share", json!({"ids": batch})).await;
            assert!(shared.get("refused").is_none(), "{shared}");
            written.extend(shared["written"].as_array().expect("written").clone());
        }
        assert_eq!(written.len(), ids.len(), "one file per memory");
        for (name, id, case) in &stored {
            let w = written
                .iter()
                .find(|w| w["id"] == id.as_str())
                .unwrap_or_else(|| panic!("{name}: written"));
            let rel = w["path"].as_str().expect("a path");
            assert!(rel.starts_with(".spatial-memory/shared/"), "{name}: {rel}");
            let text = std::fs::read_to_string(dir.join(rel)).expect("the shared file");
            let input = case["input"].as_str().unwrap_or_default();
            for s in strings(&case["expect"]["not_contains"]) {
                assert!(!text.contains(&s), "{name}: {s} in the shared file");
            }
            for s in strings(&case["expect"]["contains"]) {
                assert!(text.contains(&v3_markers(&s)), "{name}: {s} missing: {text}");
            }
            let any = strings(&case["expect"]["contains_any"]);
            if !any.is_empty() {
                assert!(
                    any.iter().any(|s| text.contains(&v3_markers(s))),
                    "{name}: none of {any:?}: {text}"
                );
            }
            let want = v3_markers(
                case["computed_result"]["redacted_text"]
                    .as_str()
                    .expect("a redacted text"),
            );
            assert!(text.contains(&want), "{name}: the redacted body: {text}");
            assert_ne!(want, input, "{name}: the case has a credential");
        }
        eprintln!(
            "AT-9 through mcp: {} cases remembered redacted ({} memories, {duplicates} duplicates) and shared without the credential",
            stored.len(),
            ids.len()
        );
        client.cancel().await.unwrap();
    });

    // A credential in front matter: refused, and not one file written.
    let fresh = repo(&sb, "frontmatter");
    let ghp = format!("ghp_{}", "A".repeat(40));
    runtime().block_on(async {
        let client = mcp_in(&sb, &fresh).await;
        let r = call_ok(
            &client,
            "remember",
            json!({"memories": [
                {"content": "Rotate the deploy key every quarter.", "tags": ["AKIAIOSFODNN7EXAMPLE"]},
                {"content": "The CI token lives in the vault.", "anchors": [format!("service:{ghp}")]},
            ]}),
        )
        .await;
        let ids: Vec<String> = r["memories"]
            .as_array()
            .expect("memories")
            .iter()
            .map(|m| m["id"].as_str().expect("an id").to_owned())
            .collect();
        assert_eq!(ids.len(), 2, "{r}");
        let shared = call_ok(&client, "share", json!({"ids": ids})).await;
        assert_eq!(shared["written"], json!([]), "{shared}");
        let refused = shared["refused"].as_array().expect("refused");
        assert_eq!(refused.len(), 2, "{shared}");
        for r in refused {
            let e = &r["error"];
            assert_eq!(e["code"], "secret_detected", "{r}");
            assert!(
                e["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("front matter"),
                "{r}"
            );
            assert!(
                !e["next_action"].as_str().unwrap_or_default().is_empty(),
                "{r}"
            );
        }
        client.cancel().await.unwrap();
    });
    let files = files_under(&fresh.join(".spatial-memory").join("shared"));
    assert!(files.is_empty(), "no file written: {files:?}");
}
