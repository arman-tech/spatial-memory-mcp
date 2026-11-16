//! `spatial-memory mcp`: the MCP proxy (SPEC section 8; CONTRACTS 4).
//!
//! It serves exactly the thirteen tools of [`Tool::ALL`] over stdio with
//! rmcp. Each tool's input schema is [`Tool::input_schema`] and its output
//! schema [`Tool::output_schema`], so the schemas come from the frozen
//! parameter types. A call is parsed with [`Tool::parse_args`], which
//! refuses unknown fields (B-28), and forwarded to the daemon through the
//! waiting client. A refused call or a daemon error comes back as a tool
//! result with `isError: true` whose structured content is the error's
//! `code`, `message` and `next_action`, so the agent can read it.
//!
//! The proxy holds no model and opens no database. It connects lazily: the
//! first tool call starts the daemon when none answers, so a health check
//! such as `claude mcp get` does not.

use std::sync::Arc;

use rmcp::model::{
    self as m, CallToolRequestParams, CallToolResponse, CallToolResult, ErrorData, Implementation,
    JsonObject, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde_json::Value;
use sm_core::error::CoreError;
use sm_core::paths::ProfilePaths;
use sm_core::protocol::{Reply, Tool};

use crate::client::{Connector, Spawn};

/// What the server tells the agent about itself.
pub const INSTRUCTIONS: &str = "Spatial Memory stores what you learn about a project and finds it \
    again by meaning. Use `recall` before re-deriving something, `remember` for decisions, \
    solutions, errors and patterns worth keeping, and `reinforce` when a recalled memory helped. \
    Errors carry a code, a message and a next action.";

/// The SPEC section 8 description of each tool.
pub fn description(tool: Tool) -> &'static str {
    match tool {
        Tool::Remember => {
            "Store one or many memories with type, anchors and provenance. Returns identifiers \
             and any redactions made."
        }
        Tool::Recall => {
            "Search. Parameters: `query`, `scope` (project or all), `type`, `anchor`, `status`, \
             `mode` (auto, vector, keyword), `k`."
        }
        Tool::Related => "Neighbours of a memory or anchor by vector and by link walk.",
        Tool::Revise => "Edit a memory, change its status, or supersede it with a new one.",
        Tool::Forget => {
            "Delete by identifier or filter. Requires `confirm: true` for more than one memory."
        }
        Tool::Link => "Add or remove a link between anchors or memories; list links for an anchor.",
        Tool::Regions => {
            "Clusters for a project, with central memories and optional 2-dimension coordinates."
        }
        Tool::Journey => "Path between two memories.",
        Tool::Consolidate => {
            "Find near-duplicate memories and propose merges. Applies only with `apply: true`."
        }
        Tool::Share => "Write selected memories to the repository's shared folder for review.",
        Tool::Sync => "Load reviewed shared memories from the repository into the local store.",
        Tool::Status => "Daemon health, model, counts per project, index state.",
        Tool::Reinforce => {
            "Mark a recalled memory as having helped. Records the rank it was found at. Does not \
             change scores in 3.0."
        }
    }
}

/// A generated schema as the JSON object MCP carries.
pub fn schema_object(schema: impl Into<Value>) -> JsonObject {
    match schema.into() {
        Value::Object(map) => map,
        // A schema of `true` accepts anything: the empty object says the same.
        _ => JsonObject::new(),
    }
}

/// The MCP description of one tool.
pub fn tool_definition(tool: Tool) -> m::Tool {
    m::Tool::new(
        tool.as_str(),
        description(tool),
        Arc::new(schema_object(tool.input_schema())),
    )
    .with_raw_output_schema(Arc::new(schema_object(tool.output_schema())))
}

/// The thirteen tool definitions, in [`Tool::ALL`] order.
pub fn tool_definitions() -> Vec<m::Tool> {
    Tool::ALL.iter().copied().map(tool_definition).collect()
}

/// The tool result for an error: `isError: true`, the error as structured
/// content and as text.
pub fn error_result(error: &CoreError) -> CallToolResult {
    let value = serde_json::to_value(error).unwrap_or_else(|_| {
        serde_json::json!({
            "code": error.code.as_str(),
            "message": error.message,
            "next_action": error.next_action,
        })
    });
    CallToolResult::structured_error(value)
}

/// The tool result for a daemon reply. The reply must be of the tool's own
/// kind; its `data` is the structured content.
pub fn reply_result(tool: Tool, reply: &Reply) -> CallToolResult {
    let mut value = match serde_json::to_value(reply) {
        Ok(Value::Object(map)) => map,
        Ok(_) | Err(_) => {
            return error_result(&CoreError::internal(format!(
                "the daemon's reply to {tool} could not be encoded"
            )));
        }
    };
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if kind != tool.as_str() {
        return error_result(&CoreError::internal(format!(
            "the daemon answered {tool} with a {kind} reply"
        )));
    }
    let data = value
        .remove("data")
        .unwrap_or(Value::Object(JsonObject::new()));
    CallToolResult::structured(data)
}

/// The proxy: the profile resolved at start, the client's working folder
/// and the way to start a daemon.
pub struct Proxy {
    paths: Result<ProfilePaths, CoreError>,
    cwd: String,
    spawner: Arc<dyn Spawn>,
    tools: Vec<m::Tool>,
}

impl Proxy {
    /// `paths` is the resolved profile, or the error resolving it gave; in
    /// that case every tool call returns that error.
    pub fn new(
        paths: Result<ProfilePaths, CoreError>,
        cwd: String,
        spawner: Arc<dyn Spawn>,
    ) -> Self {
        Proxy {
            paths,
            cwd,
            spawner,
            tools: tool_definitions(),
        }
    }

    /// Runs one `tools/call`.
    pub async fn call(&self, request: CallToolRequestParams) -> Result<CallToolResult, ErrorData> {
        let Ok(tool) = request.name.parse::<Tool>() else {
            return Err(ErrorData::invalid_params(
                format!("unknown tool: {}", request.name),
                None,
            ));
        };
        let args = request.arguments.map_or(Value::Null, Value::Object);
        let op = match tool
            .parse_args(args)
            .and_then(|op| op.validate().map(|()| op))
        {
            Ok(op) => op,
            Err(e) => return Ok(error_result(&e)),
        };
        let paths = match &self.paths {
            Ok(p) => p.clone(),
            Err(e) => return Ok(error_result(e)),
        };
        let spawner = Arc::clone(&self.spawner);
        let cwd = self.cwd.clone();
        let joined = tokio::task::spawn_blocking(move || {
            Connector::new(paths, spawner.as_ref(), cwd).call(op)
        })
        .await;
        let reply = joined.unwrap_or_else(|e| {
            Err(CoreError::internal(format!(
                "the {tool} call stopped unexpectedly: {e}"
            )))
        });
        Ok(match reply {
            Ok(reply) => reply_result(tool, &reply),
            Err(e) => error_result(&e),
        })
    }
}

impl ServerHandler for Proxy {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("spatial-memory", sm_core::BUILD))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.tools.clone()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.call(request).await.map(Into::into)
    }

    fn get_tool(&self, name: &str) -> Option<m::Tool> {
        self.tools.iter().find(|t| t.name == name).cloned()
    }
}

/// Serves the proxy on stdin and stdout until the client disconnects.
/// Returns the exit code.
pub fn serve(proxy: Proxy) -> Result<(), CoreError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CoreError::internal(format!("cannot start the async runtime: {e}")))?;
    runtime.block_on(async move {
        let service = proxy
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|e| CoreError::internal(format!("the MCP session did not start: {e}")))?;
        service.waiting().await.map_err(|e| {
            CoreError::internal(format!("the MCP session ended with an error: {e}"))
        })?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use sm_core::error::ErrorCode;

    use super::*;

    #[test]
    fn s8_every_tool_has_a_description_and_both_schemas() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 13);
        for (def, tool) in defs.iter().zip(Tool::ALL.iter()) {
            assert_eq!(def.name, tool.as_str());
            assert!(!description(*tool).is_empty());
            assert_eq!(
                Value::Object((*def.input_schema).clone()),
                Value::from(tool.input_schema())
            );
            let out = def.output_schema.as_ref().expect("output schema");
            assert_eq!(
                Value::Object((**out).clone()),
                Value::from(tool.output_schema())
            );
        }
    }

    #[test]
    fn s8_errors_carry_code_message_and_next_action() {
        let e = CoreError::new(ErrorCode::DaemonLoading, "the model is loading")
            .with_next_action("Retry in a few seconds.");
        let r = error_result(&e);
        assert_eq!(r.is_error, Some(true));
        let s = r.structured_content.unwrap();
        assert_eq!(s["code"], "daemon_loading");
        assert_eq!(s["message"], "the model is loading");
        assert_eq!(s["next_action"], "Retry in a few seconds.");
    }

    #[test]
    fn s8_a_reply_of_another_kind_is_an_internal_error() {
        let r = reply_result(Tool::Recall, &Reply::ShuttingDown);
        assert_eq!(r.is_error, Some(true));
        assert_eq!(r.structured_content.unwrap()["code"], "internal");
    }
}
