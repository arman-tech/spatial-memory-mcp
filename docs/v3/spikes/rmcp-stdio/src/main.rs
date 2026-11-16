//! Phase 0 spike d: an rmcp 3.10.0 stdio server exposing one tool, called by an MCP client.
//! `spike-rmcp-stdio server` serves on stdin/stdout. `spike-rmcp-stdio client` spawns the
//! server as a child process, lists its tools and calls the tool 50 times.
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolRequestParams, ServerCapabilities, ServerConfig},
    schemars, tool, tool_handler, tool_router,
    transport::{ConfigureCommandExt, TokioChildProcess, stdio},
};
use std::time::Instant;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct EchoRequest {
    /// Text to upper-case.
    pub text: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallRequest {
    /// The user's prompt.
    pub prompt: String,
}

fn stub_block(prompt: &str) -> String {
    let words = prompt.split_whitespace().count();
    format!("[spatial-memory] ambient recall:
  1. (0.80) 0193a8f0 spike stub for a {words}-word prompt")
}

#[derive(Debug, Clone)]
pub struct Spike {
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl Spike {
    fn new() -> Self {
        Self { tool_router: Self::tool_router() }
    }

    #[tool(description = "Spike stand-in for ambient recall: returns a fixed injection block")]
    fn ambient_recall(&self, Parameters(RecallRequest { prompt }): Parameters<RecallRequest>) -> String {
        stub_block(&prompt)
    }

    #[tool(description = "Return the text upper-cased")]
    fn echo_upper(&self, Parameters(EchoRequest { text }): Parameters<EchoRequest>) -> String {
        text.to_uppercase()
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Spike {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("phase 0 spike d")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "hook" {
        use std::io::{Read, Write};
        let mut input = String::new();
        let _ = std::io::stdin().read_to_string(&mut input);
        let v: serde_json::Value = serde_json::from_str(&input).unwrap_or_default();
        let event = std::env::args().nth(2).unwrap_or_default();
        if let Ok(path) = std::env::var("SPIKE_HOOK_LOG") {
            let line = format!("{} {} {}
", event, v["hook_event_name"].as_str().unwrap_or(""), v["source"].as_str().unwrap_or(""));
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(line.as_bytes());
            }
        }
        if event == "user-prompt-submit" {
            let prompt = v["prompt"].as_str().unwrap_or("");
            println!("{}", stub_block(prompt));
        } else if event == "session-start" {
            println!("Project memory is available through the spatial-memory tools.");
        }
        return Ok(());
    }
    if mode == "server" {
        let svc = Spike::new().serve(stdio()).await?;
        svc.waiting().await?;
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let t0 = Instant::now();
    let child = TokioChildProcess::new(tokio::process::Command::new(exe).configure(|c| {
        c.arg("server");
    }))?;
    let client = ().serve(child).await?;
    let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tools = client.list_all_tools().await?;
    let names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    println!("tools: {names:?}");
    println!("schema: {}", serde_json::to_string(&tools[0].input_schema)?);
    let mut times = Vec::new();
    let mut last = String::new();
    for i in 0..50 {
        let mut args = serde_json::Map::new();
        args.insert("text".into(), serde_json::Value::String(format!("hello {i}")));
        let t = Instant::now();
        let r = client.call_tool(CallToolRequestParams::new("echo_upper").with_arguments(args)).await?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        last = serde_json::to_string(&r.content)?;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("last result: {last}");
    println!("init+spawn ms: {init_ms:.2}");
    println!("call ms median {:.3} p95 {:.3} max {:.3}", times[25], times[47], times[49]);
    client.cancel().await?;
    Ok(())
}
