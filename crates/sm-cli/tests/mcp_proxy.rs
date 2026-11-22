//! The MCP proxy over stdio (SPEC 8, CONTRACTS 11): an rmcp client drives
//! `spatial-memory mcp` as a child process, as spike d did, with the mock
//! daemon on the sandbox's endpoint.

mod common;

use std::sync::{Arc, Mutex, MutexGuard};

use common::{MockDaemon, Mode, Sandbox};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult, JsonObject};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Value, json};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::ids::MemoryId;
use sm_core::protocol::{ForgetReply, Op, Reply, Tool};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// `spatial-memory mcp` in the sandbox, with an rmcp client attached.
async fn connect(sb: &Sandbox) -> RunningService<RoleClient, ()> {
    let mut cmd = sb.command(&["mcp"]);
    cmd.stderr(std::process::Stdio::null());
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

fn error_of(result: &CallToolResult) -> Value {
    assert_eq!(result.is_error, Some(true), "{result:?}");
    result
        .structured_content
        .clone()
        .expect("an error carries structured content")
}

#[test]
fn s8_the_proxy_lists_the_thirteen_tools_with_the_parameter_type_schemas() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    runtime().block_on(async {
        let client = connect(&sb).await;
        let info = client.peer_info().expect("server info");
        let server = info.server_info.as_ref().expect("server name and version");
        assert_eq!(server.name, "spatial-memory");
        // The release, then the commit the program was built from (B-231).
        assert_eq!(server.version, sm_cli::version::STAMP);
        assert_eq!(sm_cli::version::split(&server.version).0, sm_core::BUILD);
        let tools = client.list_all_tools().await.expect("tools/list");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        let expected: Vec<&str> = Tool::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(names, expected);
        assert_eq!(tools.len(), 13);
        for (def, tool) in tools.iter().zip(Tool::ALL.iter()) {
            assert_eq!(
                Value::Object((*def.input_schema).clone()),
                Value::from(tool.input_schema()),
                "{tool} input schema"
            );
            let output = def.output_schema.as_ref().expect("an output schema");
            assert_eq!(
                Value::Object(JsonObject::clone(output)),
                Value::from(tool.output_schema()),
                "{tool} output schema"
            );
            assert!(def.description.as_ref().is_some_and(|d| !d.is_empty()));
        }
        client.cancel().await.unwrap();
    });
    // Listing is not a call: the proxy connects lazily.
    assert_eq!(mock.connections(), 0);
    assert!(!sb.lock_file_exists());
}

#[test]
fn s8_a_call_with_an_unknown_field_is_refused_before_the_daemon() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    runtime().block_on(async {
        let client = connect(&sb).await;
        let result = call(
            &client,
            "recall",
            json!({"query": "pool", "colour": "blue"}),
        )
        .await;
        let err = error_of(&result);
        assert_eq!(err["code"], "invalid_input");
        assert!(err["message"].as_str().unwrap().contains("colour"), "{err}");
        assert!(!err["next_action"].as_str().unwrap().is_empty());
        // A tool that does not exist is a protocol error.
        let unknown = client
            .call_tool(CallToolRequestParams::new("remember_everything"))
            .await;
        assert!(unknown.is_err(), "{unknown:?}");
        client.cancel().await.unwrap();
    });
    assert!(mock.requests().is_empty());
}

#[test]
fn s8_a_daemon_error_reaches_the_client_with_its_code_and_next_action() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let refusal = CoreError::new(ErrorCode::NotFound, "no memory has that anchor")
        .with_next_action("Recall without the anchor, or link one first.");
    let answer = refusal.clone();
    let deleted = MemoryId::new_v7();
    mock.set_responder(Arc::new(move |request| match &request.call {
        Op::Recall(_) => Some(Err(answer.clone())),
        Op::Forget(p) => Some(Ok(Reply::Forget(ForgetReply {
            deleted: p.ids.clone(),
            pending: 0,
        }))),
        _ => None,
    }));
    runtime().block_on(async {
        let client = connect(&sb).await;
        let result = call(
            &client,
            "recall",
            json!({"query": "pool", "anchor": "file:src/db.rs"}),
        )
        .await;
        let err = error_of(&result);
        assert_eq!(err, serde_json::to_value(&refusal).unwrap());
        assert_eq!(err["code"], "not_found");
        assert_eq!(
            err["next_action"],
            "Recall without the anchor, or link one first."
        );
        // A reply of the tool's own kind is the structured result.
        let ok = call(&client, "forget", json!({"ids": [deleted.to_string()]})).await;
        assert_ne!(ok.is_error, Some(true), "{ok:?}");
        assert_eq!(
            ok.structured_content,
            Some(json!({"deleted": [deleted.to_string()]}))
        );
        client.cancel().await.unwrap();
    });
    let ops: Vec<&str> = mock.requests().iter().map(|r| r.call.name()).collect();
    assert_eq!(ops, ["hello", "recall", "hello", "forget"]);
    for r in mock.requests() {
        assert_eq!(r.cwd, sb.tmp.to_string_lossy());
        assert_eq!(r.session_id, None, "the proxy has no session");
    }
}

#[test]
fn s8_a_loading_daemon_reaches_the_client_as_daemon_loading() {
    let _g = serial();
    let sb = Sandbox::new();
    let _mock = MockDaemon::start(&sb.paths(), Mode::Loading);
    runtime().block_on(async {
        let client = connect(&sb).await;
        let result = call(&client, "recall", json!({"query": "pool"})).await;
        let err = error_of(&result);
        assert_eq!(err["code"], "daemon_loading");
        assert_eq!(err["next_action"], "Retry in a few seconds.");
        client.cancel().await.unwrap();
    });
}
