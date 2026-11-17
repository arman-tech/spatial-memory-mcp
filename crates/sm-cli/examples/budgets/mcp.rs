//! `spatial-memory mcp` with an rmcp client attached, as an agent runs it.

use std::path::Path;
use std::process::Stdio;

use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::Value;

use crate::harness::Home;

pub type Client = RunningService<RoleClient, ()>;

pub fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a tokio runtime")
}

/// The MCP server in `cwd`, with a client attached.
pub async fn connect(home: &Home, cwd: &Path) -> Client {
    let mut cmd = home.command(&["mcp"], cwd);
    cmd.stderr(Stdio::null());
    let child = TokioChildProcess::new(tokio::process::Command::from(cmd))
        .expect("spawn spatial-memory mcp");
    ().serve(child).await.expect("the MCP session starts")
}

/// A tool call that must succeed; its structured content.
pub async fn call(client: &Client, tool: &str, args: Value) -> Value {
    let Value::Object(map) = args else {
        panic!("{tool}: arguments are an object")
    };
    let result = client
        .call_tool(CallToolRequestParams::new(tool.to_owned()).with_arguments(map))
        .await
        .unwrap_or_else(|e| panic!("{tool}: tools/call: {e}"));
    assert_ne!(result.is_error, Some(true), "{tool}: {result:?}");
    result
        .structured_content
        .unwrap_or_else(|| panic!("{tool} has structured content"))
}
