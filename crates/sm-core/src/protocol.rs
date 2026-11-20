//! The daemon message schema (SPEC section 3, Transport and rule 1).
//!
//! Every client (the MCP proxy, the hook, the command line) talks to the
//! daemon over a Unix domain socket or a Windows named pipe with
//! length-prefixed JSON frames:
//!
//! ```text
//! +----------------------+---------------------------------------+
//! | length: u32, LE (4B) | body: UTF-8 JSON, `length` bytes      |
//! +----------------------+---------------------------------------+
//! ```
//!
//! A connection carries any number of request and response pairs, strictly
//! in turn; either side may close it between pairs. Every request carries the
//! protocol version, the client's build, a request id, the client's working
//! directory and, when the client has one, the session id.
//!
//! **Frozen forever** (so any two builds can always negotiate a restart): the
//! frame format, the top-level fields `v`, `build` and `id`, the
//! `call.op` tag, the `shutdown` op and the `version_mismatch` error. A daemon
//! that receives a request whose `v` or `build` differs from its own replies
//! `version_mismatch` to everything except `shutdown`, which it always
//! accepts; the client then sends `shutdown` and starts a daemon of its own
//! build ([`Envelope`]). Everything else here is frozen for the 3.0 contract
//! (SPEC section 14) and may change only together with [`PROTOCOL_VERSION`].
//!
//! The thirteen tool parameter types double as the MCP input schemas (SPEC
//! section 8, Conventions: JSON Schema generated from Rust types), so they
//! reject unknown fields: an agent that sends a version 1 parameter such as
//! `namespace` gets an error naming the fields that exist.

use std::collections::BTreeMap;
use std::io::{ErrorKind, Read, Write};

use schemars::JsonSchema;
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize};

use crate::config::{MAX_K, MIN_OUTPUT_TOKEN_BUDGET, ProfileName};
use crate::error::{CoreError, ErrorCode, Result};
use crate::eval::EvalReport;
use crate::ids::{MemoryId, RunId};
use crate::journal::{CaptureQuality, HookGate, MetricSummary};
use crate::time::Timestamp;
use crate::types::{
    AnchorRef, Client, Link, LinkEnd, MatchKind, MemoryType, Origin, Provenance, RecallMode,
    Relation, Scope, Status, check_importance,
};

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

/// The message schema version. Bumped on any change to the types in this
/// module, even between unreleased builds, whose `build` is the same crate
/// version (B-171); client and daemon must agree exactly.
pub const PROTOCOL_VERSION: u32 = 3;

/// Bytes in the length prefix.
pub const FRAME_PREFIX_BYTES: usize = 4;

/// Largest frame body either side accepts. A capture payload larger than
/// this is spooled instead of sent (the spool has no size limit).
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

fn connection_error(e: &std::io::Error) -> CoreError {
    CoreError::new(
        ErrorCode::DaemonUnavailable,
        format!("connection error: {e}"),
    )
}

fn frame_too_large(len: usize) -> CoreError {
    CoreError::invalid_input(format!(
        "message of {len} bytes exceeds the {MAX_FRAME_BYTES}-byte frame limit"
    ))
}

/// Encodes one message as a complete frame (prefix and body in one buffer,
/// so it goes out in one write call; spike decision 3).
pub fn encode_frame<T: Serialize>(message: &T) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; FRAME_PREFIX_BYTES];
    serde_json::to_writer(&mut buf, message)
        .map_err(|e| CoreError::internal(format!("cannot encode message: {e}")))?;
    let len = buf.len() - FRAME_PREFIX_BYTES;
    if len > MAX_FRAME_BYTES {
        return Err(frame_too_large(len));
    }
    let prefix = u32::try_from(len)
        .map_err(|_| frame_too_large(len))?
        .to_le_bytes();
    buf[..FRAME_PREFIX_BYTES].copy_from_slice(&prefix);
    Ok(buf)
}

/// Reads the body length from a prefix. Zero-length and oversized frames are
/// refused before any body byte is read.
pub fn frame_len(prefix: [u8; FRAME_PREFIX_BYTES]) -> Result<usize> {
    let len = u32::from_le_bytes(prefix) as usize;
    if len == 0 {
        return Err(CoreError::invalid_input("empty frame"));
    }
    if len > MAX_FRAME_BYTES {
        return Err(frame_too_large(len));
    }
    Ok(len)
}

/// Decodes a frame body.
pub fn decode_body<T: DeserializeOwned>(body: &[u8]) -> Result<T> {
    serde_json::from_slice(body)
        .map_err(|e| CoreError::invalid_input(format!("malformed message: {e}")))
}

/// Writes one message as one frame and flushes.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, message: &T) -> Result<()> {
    let frame = encode_frame(message)?;
    w.write_all(&frame).map_err(|e| connection_error(&e))?;
    w.flush().map_err(|e| connection_error(&e))
}

/// Reads one frame body. `Ok(None)` is a clean close before a new frame; a
/// close inside a frame is an error.
pub fn read_frame<R: Read>(r: &mut R) -> Result<Option<Vec<u8>>> {
    let mut prefix = [0u8; FRAME_PREFIX_BYTES];
    let mut got = 0;
    while got < FRAME_PREFIX_BYTES {
        match r.read(&mut prefix[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => {
                return Err(CoreError::new(
                    ErrorCode::DaemonUnavailable,
                    "connection closed inside a frame prefix",
                ));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(connection_error(&e)),
        }
    }
    let len = frame_len(prefix)?;
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| connection_error(&e))?;
    Ok(Some(body))
}

/// Reads and decodes one message; `Ok(None)` on a clean close.
pub fn read_message<R: Read, T: DeserializeOwned>(r: &mut R) -> Result<Option<T>> {
    match read_frame(r)? {
        Some(body) => decode_body(&body).map(Some),
        None => Ok(None),
    }
}

/// A new request id: the last 12 hex digits of a UUID version 7 (time and
/// randomness; unique enough to pair a request with its journal lines).
pub fn new_request_id() -> String {
    let simple = uuid::Uuid::now_v7().simple().to_string();
    simple[simple.len() - 12..].to_owned()
}

// ---------------------------------------------------------------------------
// Version handshake
// ---------------------------------------------------------------------------

/// The frozen part of every message, read before the full decode so a daemon
/// can answer a client of another build that sends ops it does not know.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Envelope {
    pub v: u32,
    pub build: String,
    pub id: String,
    #[serde(default)]
    pub call: Option<OpTag>,
}

/// The `op` tag of a request's call.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OpTag {
    pub op: String,
}

impl Envelope {
    /// Reads the envelope of a frame body, ignoring everything else.
    pub fn peek(body: &[u8]) -> Result<Envelope> {
        decode_body(body)
    }

    /// Whether the call is `shutdown`, which every build accepts.
    pub fn is_shutdown(&self) -> bool {
        self.call.as_ref().is_some_and(|c| c.op == "shutdown")
    }

    /// `version_mismatch` unless the sender is this exact protocol and build.
    pub fn check_version(&self) -> Result<()> {
        check_version(self.v, &self.build)
    }
}

fn check_version(v: u32, build: &str) -> Result<()> {
    if v == PROTOCOL_VERSION && build == crate::BUILD {
        Ok(())
    } else {
        Err(CoreError::new(
            ErrorCode::VersionMismatch,
            format!(
                "peer speaks protocol {v} build {build}; this is protocol {PROTOCOL_VERSION} build {}",
                crate::BUILD
            ),
        ))
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// One request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// [`PROTOCOL_VERSION`] of the sender.
    pub v: u32,
    /// [`crate::BUILD`] of the sender.
    pub build: String,
    /// Pairs the request with its response and its journal lines.
    pub id: String,
    /// The client's working directory, as the client received it. The
    /// daemon resolves the project from it (SPEC section 9, Scope).
    pub cwd: String,
    /// The agent session, when the client has one (the hook does; the MCP
    /// proxy does not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Which agent the hook serves (`--client`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<Client>,
    /// The profile the client resolved. A daemon serves one profile and
    /// refuses a request for another with `invalid_input`.
    pub profile: ProfileName,
    /// Milliseconds the client will wait from sending. Work still running
    /// past it is abandoned with `timeout`. `None` waits for completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    pub call: Op,
}

impl Request {
    /// A request from this build with a fresh id.
    pub fn new(cwd: impl Into<String>, profile: ProfileName, call: Op) -> Self {
        Request {
            v: PROTOCOL_VERSION,
            build: crate::BUILD.to_owned(),
            id: new_request_id(),
            cwd: cwd.into(),
            session_id: None,
            client: None,
            profile,
            deadline_ms: None,
            call,
        }
    }

    #[must_use]
    pub fn with_session(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id.filter(|s| !s.is_empty());
        self
    }

    #[must_use]
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    #[must_use]
    pub fn with_deadline_ms(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }
}

/// What a request asks for. Serialised as `{"op": "<name>", "args": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "args", rename_all = "snake_case")]
pub enum Op {
    /// Daemon identity and state; also the readiness probe.
    Hello,
    /// Exit after replying. Accepted from any build.
    Shutdown(ShutdownParams),

    // The thirteen MCP tools (SPEC section 8).
    Remember(RememberParams),
    Recall(RecallParams),
    Related(RelatedParams),
    Revise(ReviseParams),
    Forget(ForgetParams),
    Link(LinkParams),
    Regions(RegionsParams),
    Journey(JourneyParams),
    Consolidate(ConsolidateParams),
    Share(ShareParams),
    Sync(SyncParams),
    Status(StatusParams),
    Reinforce(ReinforceParams),

    // Hooks (SPEC sections 6 and 16).
    AmbientRecall(AmbientParams),
    Capture(CaptureParams),
    /// `spatial-memory replay`: a session's stored transcript through the
    /// capture pipeline again, queued as a capture is (SPEC 16, Replay;
    /// B-172).
    Replay(ReplayParams),

    // Command line (SPEC section 8, Moved to the command line).
    Import(ImportParams),
    Export(ExportParams),
    Migrate(MigrateParams),
    Reindex(ReindexParams),
    ProjectRename(ProjectRenameParams),
    ProjectDelete(ProjectDeleteParams),
    DoctorEval(DoctorEvalParams),
    /// The embedder's numbers for `doctor` (SPEC 5 rule 7; B-125).
    DoctorNumbers,
}

impl Op {
    /// The `op` tag.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Hello => "hello",
            Op::Shutdown(_) => "shutdown",
            Op::Remember(_) => "remember",
            Op::Recall(_) => "recall",
            Op::Related(_) => "related",
            Op::Revise(_) => "revise",
            Op::Forget(_) => "forget",
            Op::Link(_) => "link",
            Op::Regions(_) => "regions",
            Op::Journey(_) => "journey",
            Op::Consolidate(_) => "consolidate",
            Op::Share(_) => "share",
            Op::Sync(_) => "sync",
            Op::Status(_) => "status",
            Op::Reinforce(_) => "reinforce",
            Op::AmbientRecall(_) => "ambient_recall",
            Op::Capture(_) => "capture",
            Op::Replay(_) => "replay",
            Op::Import(_) => "import",
            Op::Export(_) => "export",
            Op::Migrate(_) => "migrate",
            Op::Reindex(_) => "reindex",
            Op::ProjectRename(_) => "project_rename",
            Op::ProjectDelete(_) => "project_delete",
            Op::DoctorEval(_) => "doctor_eval",
            Op::DoctorNumbers => "doctor_numbers",
        }
    }

    /// Checks the parameters that do not depend on the store.
    pub fn validate(&self) -> Result<()> {
        match self {
            Op::Hello | Op::Status(_) | Op::Sync(_) | Op::Reindex(_) | Op::DoctorNumbers => Ok(()),
            Op::Shutdown(_) => Ok(()),
            Op::Remember(p) => p.validate(),
            Op::Recall(p) => p.validate(),
            Op::Related(p) => p.validate(),
            Op::Revise(p) => p.validate(),
            Op::Forget(p) => p.validate(),
            Op::Link(p) => p.validate(),
            Op::Regions(p) => p.validate(),
            Op::Journey(p) => p.validate(),
            Op::Consolidate(p) => p.validate(),
            Op::Share(p) => p.validate(),
            Op::Reinforce(p) => p.validate(),
            Op::AmbientRecall(p) => p.validate(),
            Op::Capture(_) => Ok(()),
            Op::Replay(p) => p.validate(),
            Op::Import(p) => non_blank("path", &p.path),
            Op::Export(p) => non_blank("path", &p.path),
            Op::Migrate(p) => non_blank("from", &p.from),
            Op::ProjectRename(p) => {
                non_blank("from", &p.from)?;
                non_blank("to", &p.to)
            }
            Op::ProjectDelete(p) => non_blank("name", &p.name),
            Op::DoctorEval(p) => non_blank("set_path", &p.set_path),
        }
    }
}

string_enum! {
    /// The thirteen MCP tools (SPEC section 8).
    pub enum Tool {
        Remember => "remember",
        Recall => "recall",
        Related => "related",
        Revise => "revise",
        Forget => "forget",
        Link => "link",
        Regions => "regions",
        Journey => "journey",
        Consolidate => "consolidate",
        Share => "share",
        Sync => "sync",
        Status => "status",
        Reinforce => "reinforce",
    }
}

impl Tool {
    /// The JSON Schema of the tool's input, generated from its parameter type.
    pub fn input_schema(self) -> schemars::Schema {
        match self {
            Tool::Remember => tool_schema::<RememberParams>(),
            Tool::Recall => tool_schema::<RecallParams>(),
            Tool::Related => tool_schema::<RelatedParams>(),
            Tool::Revise => tool_schema::<ReviseParams>(),
            Tool::Forget => tool_schema::<ForgetParams>(),
            Tool::Link => tool_schema::<LinkParams>(),
            Tool::Regions => tool_schema::<RegionsParams>(),
            Tool::Journey => tool_schema::<JourneyParams>(),
            Tool::Consolidate => tool_schema::<ConsolidateParams>(),
            Tool::Share => tool_schema::<ShareParams>(),
            Tool::Sync => tool_schema::<SyncParams>(),
            Tool::Status => tool_schema::<StatusParams>(),
            Tool::Reinforce => tool_schema::<ReinforceParams>(),
        }
    }

    /// The JSON Schema of the tool's reply.
    pub fn output_schema(self) -> schemars::Schema {
        match self {
            Tool::Remember => tool_schema::<RememberReply>(),
            Tool::Recall => tool_schema::<RecallReply>(),
            Tool::Related => tool_schema::<RelatedReply>(),
            Tool::Revise => tool_schema::<ReviseReply>(),
            Tool::Forget => tool_schema::<ForgetReply>(),
            Tool::Link => tool_schema::<LinkReply>(),
            Tool::Regions => tool_schema::<RegionsReply>(),
            Tool::Journey => tool_schema::<JourneyReply>(),
            Tool::Consolidate => tool_schema::<ConsolidateReply>(),
            Tool::Share => tool_schema::<ShareReply>(),
            Tool::Sync => tool_schema::<SyncReply>(),
            Tool::Status => tool_schema::<StatusReply>(),
            Tool::Reinforce => tool_schema::<ReinforceReply>(),
        }
    }

    /// Builds the op from the tool's JSON arguments (an MCP `tools/call`).
    pub fn parse_args(self, args: serde_json::Value) -> Result<Op> {
        fn de<T: DeserializeOwned>(tool: Tool, args: serde_json::Value) -> Result<T> {
            serde_json::from_value(args)
                .map_err(|e| CoreError::invalid_input(format!("invalid arguments for {tool}: {e}")))
        }
        let args = if args.is_null() {
            serde_json::json!({})
        } else {
            args
        };
        let op = match self {
            Tool::Remember => Op::Remember(de(self, args)?),
            Tool::Recall => Op::Recall(de(self, args)?),
            Tool::Related => Op::Related(de(self, args)?),
            Tool::Revise => Op::Revise(de(self, args)?),
            Tool::Forget => Op::Forget(de(self, args)?),
            Tool::Link => Op::Link(de(self, args)?),
            Tool::Regions => Op::Regions(de(self, args)?),
            Tool::Journey => Op::Journey(de(self, args)?),
            Tool::Consolidate => Op::Consolidate(de(self, args)?),
            Tool::Share => Op::Share(de(self, args)?),
            Tool::Sync => Op::Sync(de(self, args)?),
            Tool::Status => Op::Status(de(self, args)?),
            Tool::Reinforce => Op::Reinforce(de(self, args)?),
        };
        op.validate()?;
        Ok(op)
    }
}

/// A tool's schema, generated from `T` with the default settings, with no
/// `format` on any integer or number (B-154). Claude Code checks tool
/// schemas with a validator that knows no `uint32`, `uint64`, `uint8`,
/// `int64`, `float` or `double`, and warns for each one. The bounds an
/// unsigned type needs stay as `minimum`.
fn tool_schema<T: JsonSchema>() -> schemars::Schema {
    schemars::generate::SchemaSettings::default()
        .with_transform(schemars::transform::RecursiveTransform(drop_number_format))
        .into_generator()
        .into_root_schema_for::<T>()
}

/// Removes `format` from a schema whose type is an integer or a number.
fn drop_number_format(schema: &mut schemars::Schema) {
    let numeric = |t: &serde_json::Value| matches!(t.as_str(), Some("integer" | "number"));
    let is_numeric = schema
        .get("type")
        .is_some_and(|t| numeric(t) || t.as_array().is_some_and(|a| a.iter().any(numeric)));
    if is_numeric {
        schema.remove("format");
    }
}

// ---------------------------------------------------------------------------
// Parameter helpers
// ---------------------------------------------------------------------------

/// Accepts one value or an array of values (agents often send `"type":
/// "decision"` for a list-valued filter). `null` is an empty list.
fn one_or_many<'de, D, T>(d: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Null => Ok(Vec::new()),
        v @ serde_json::Value::Array(_) => serde_json::from_value(v).map_err(D::Error::custom),
        v => serde_json::from_value(v)
            .map(|x| vec![x])
            .map_err(D::Error::custom),
    }
}

fn non_blank(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(CoreError::invalid_input(format!(
            "{name} must not be empty"
        )))
    } else {
        Ok(())
    }
}

fn check_k(k: Option<u32>) -> Result<()> {
    match k {
        Some(0) => Err(CoreError::invalid_input("k must be at least 1")),
        Some(k) if k > MAX_K => Err(CoreError::invalid_input(format!(
            "k must be at most {MAX_K}"
        ))),
        _ => Ok(()),
    }
}

/// `limit` is an explicit token budget for the reply (SPEC section 8,
/// Conventions); without it the configured `output.token_budget` applies.
fn check_limit(limit: Option<u32>) -> Result<()> {
    match limit {
        Some(l) if l < MIN_OUTPUT_TOKEN_BUDGET => Err(CoreError::invalid_input(format!(
            "limit is a token budget and must be at least {MIN_OUTPUT_TOKEN_BUDGET}"
        ))),
        _ => Ok(()),
    }
}

fn check_opt_importance(name: &str, value: Option<f64>) -> Result<()> {
    match value {
        Some(v) => check_importance(v)
            .map(|_| ())
            .map_err(|e| CoreError::invalid_input(format!("{name}: {}", e.message))),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Tool parameters (SPEC section 8)
// ---------------------------------------------------------------------------

/// Most memories one `remember` call stores.
pub const MAX_REMEMBER_BATCH: usize = 100;

/// `remember`: store one or many memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RememberParams {
    /// The memories to store, 1 to 100.
    pub memories: Vec<RememberItem>,
}

impl RememberParams {
    pub fn validate(&self) -> Result<()> {
        if self.memories.is_empty() || self.memories.len() > MAX_REMEMBER_BATCH {
            return Err(CoreError::invalid_input(format!(
                "memories must hold 1 to {MAX_REMEMBER_BATCH} items, got {}",
                self.memories.len()
            )));
        }
        for (i, m) in self.memories.iter().enumerate() {
            m.validate()
                .map_err(|e| CoreError::invalid_input(format!("memories[{i}]: {}", e.message)))?;
        }
        Ok(())
    }
}

/// One memory to store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RememberItem {
    /// The text, stored whole. Secrets are redacted before storing.
    pub content: String,
    /// A short title; derived from the first line of `content` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// decision, pitfall, convention, incident, runbook or note (default).
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<MemoryType>,
    /// What the memory is about, as `kind:name` (for example
    /// `file:src/db.rs`, `table:orders`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<AnchorRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// 0 to 1, default 0.5. Stored, not used for ranking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f64>,
    /// An older memory this one replaces; it is marked superseded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<MemoryId>,
    /// The file the memory came from, relative to the project root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// Who wrote it, when not the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

impl RememberItem {
    /// A note with only content.
    pub fn new(content: impl Into<String>) -> Self {
        RememberItem {
            content: content.into(),
            title: None,
            memory_type: None,
            anchors: Vec::new(),
            tags: Vec::new(),
            importance: None,
            supersedes: None,
            file_path: None,
            author: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        non_blank("content", &self.content)?;
        check_opt_importance("importance", self.importance)?;
        crate::types::normalize_tags(self.tags.iter()).map(|_| ())
    }
}

/// `recall`: search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecallParams {
    pub query: String,
    /// project (default) or all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    /// Memory types to keep; all when empty. One value or a list.
    #[serde(
        default,
        rename = "type",
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub memory_type: Vec<MemoryType>,
    /// Keep only memories about this anchor (`kind:name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<AnchorRef>,
    /// Statuses to keep: active, superseded or retracted; active only when
    /// empty. One value or a list. `pending` is refused: an automatic
    /// capture is never served before the judge keeps it (B-157).
    #[serde(
        default,
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub status: Vec<Status>,
    /// auto (default: vector, plus keyword hits for identifier-like
    /// queries), vector or keyword.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<RecallMode>,
    /// Results, 1 to 50; default `recall.k` (5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    /// Keep only memories with at least this importance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_importance: Option<f64>,
    /// Token budget for the reply; default `output.token_budget` (2,000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

impl RecallParams {
    pub fn new(query: impl Into<String>) -> Self {
        RecallParams {
            query: query.into(),
            scope: None,
            memory_type: Vec::new(),
            anchor: None,
            status: Vec::new(),
            mode: None,
            k: None,
            min_importance: None,
            limit: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        non_blank("query", &self.query)?;
        if self.status.contains(&Status::Pending) {
            return Err(not_served());
        }
        check_k(self.k)?;
        check_opt_importance("min_importance", self.min_importance)?;
        check_limit(self.limit)
    }
}

/// The error for a request to serve `pending` memories (B-157).
fn not_served() -> CoreError {
    CoreError::invalid_input("pending captures are never served").with_next_action(
        "Ask for active, superseded or retracted. A capture is served once the judge keeps it.",
    )
}

/// `related`: neighbours of a memory or an anchor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelatedParams {
    /// Start from this memory. Give exactly one of `id` and `anchor`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<MemoryId>,
    /// Start from this anchor (`kind:name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<AnchorRef>,
    /// Nearest memories by vector, 1 to 50; default `recall.k`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    /// project (default) or all, for the vector list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// Link hops `related` walks (SPEC section 6, Related).
pub const RELATED_MAX_HOPS: u8 = 2;

impl RelatedParams {
    pub fn validate(&self) -> Result<()> {
        if self.id.is_some() == self.anchor.is_some() {
            return Err(CoreError::invalid_input(
                "give exactly one of id and anchor",
            ));
        }
        check_k(self.k)?;
        check_limit(self.limit)
    }
}

/// `revise`: edit a memory, change its status, or supersede it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReviseParams {
    pub id: MemoryId,
    /// New content; the memory is re-chunked and re-embedded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<MemoryType>,
    /// active, superseded or retracted. `pending` is set only by automatic
    /// capture; setting a pending capture `active` keeps it by hand (B-157).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<Status>,
    /// Replaces the tag list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Replaces the anchor list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchors: Option<Vec<AnchorRef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f64>,
    /// Stores this as a new memory that supersedes `id`; `id` becomes
    /// superseded in the same transaction. Excludes every other change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersede_with: Option<RememberItem>,
}

impl ReviseParams {
    pub fn new(id: MemoryId) -> Self {
        ReviseParams {
            id,
            content: None,
            title: None,
            memory_type: None,
            status: None,
            tags: None,
            anchors: None,
            importance: None,
            supersede_with: None,
        }
    }

    fn has_edit(&self) -> bool {
        self.content.is_some()
            || self.title.is_some()
            || self.memory_type.is_some()
            || self.status.is_some()
            || self.tags.is_some()
            || self.anchors.is_some()
            || self.importance.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        match (&self.supersede_with, self.has_edit()) {
            (Some(_), true) => Err(CoreError::invalid_input(
                "supersede_with excludes every other change; revise the new memory afterwards",
            )),
            (Some(new), false) => new.validate(),
            (None, false) => Err(CoreError::invalid_input("nothing to change")),
            (None, true) => {
                if self.status == Some(Status::Pending) {
                    return Err(CoreError::invalid_input(
                        "status pending is set only by automatic capture",
                    )
                    .with_next_action("Use active, superseded or retracted."));
                }
                if let Some(c) = &self.content {
                    non_blank("content", c)?;
                }
                if let Some(tags) = &self.tags {
                    crate::types::normalize_tags(tags.iter())?;
                }
                check_opt_importance("importance", self.importance)
            }
        }
    }
}

/// `forget`: delete by identifier or by filter. More than one memory needs
/// `confirm: true`; without it the call fails with `confirm_required` and
/// the message says how many memories would go.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetParams {
    /// Memories to delete. Give `ids` or `filter`, not both.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<MemoryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<ForgetFilter>,
    #[serde(default)]
    pub confirm: bool,
}

/// Which memories a filtered `forget` deletes. At least one criterion
/// besides `scope` is required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetFilter {
    /// project (default) or all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    #[serde(
        default,
        rename = "type",
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub memory_type: Vec<MemoryType>,
    /// Statuses to match, `pending` included; any status when empty.
    #[serde(
        default,
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub status: Vec<Status>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<AnchorRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    /// Only memories created before this time (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_before: Option<Timestamp>,
}

impl ForgetParams {
    pub fn validate(&self) -> Result<()> {
        match (&self.filter, self.ids.is_empty()) {
            (Some(_), false) => Err(CoreError::invalid_input("give ids or filter, not both")),
            (None, true) => Err(CoreError::invalid_input("give ids or filter")),
            (None, false) => Ok(()),
            (Some(f), true) => {
                let any = !f.memory_type.is_empty()
                    || !f.status.is_empty()
                    || f.anchor.is_some()
                    || f.tag.is_some()
                    || f.created_before.is_some();
                if any {
                    Ok(())
                } else {
                    Err(CoreError::invalid_input(
                        "filter needs at least one of type, status, anchor, tag, created_before",
                    ))
                }
            }
        }
    }
}

string_enum! {
    /// What `link` does.
    pub enum LinkAction {
        Add => "add",
        Remove => "remove",
        List => "list",
    }
}

/// `link`: add or remove a link between anchors or memories, or list the
/// links of one end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LinkParams {
    pub action: LinkAction,
    /// add, remove: the source (`kind:name` or `memory:<id>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<LinkEnd>,
    /// add, remove: the destination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dst: Option<LinkEnd>,
    /// add, remove: calls, reads, writes, depends_on, relates_to, caused_by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<Relation>,
    /// list: links into or out of this end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<LinkEnd>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

impl LinkParams {
    pub fn validate(&self) -> Result<()> {
        check_limit(self.limit)?;
        match self.action {
            LinkAction::Add | LinkAction::Remove => {
                if self.src.is_none() || self.dst.is_none() || self.relation.is_none() {
                    return Err(CoreError::invalid_input(format!(
                        "{} needs src, dst and relation",
                        self.action
                    )));
                }
                if self.end.is_some() {
                    return Err(CoreError::invalid_input("end is only for list"));
                }
                if self.src == self.dst {
                    return Err(CoreError::invalid_input("src and dst are the same"));
                }
                Ok(())
            }
            LinkAction::List => {
                if self.end.is_none() {
                    return Err(CoreError::invalid_input("list needs end"));
                }
                if self.src.is_some() || self.dst.is_some() || self.relation.is_some() {
                    return Err(CoreError::invalid_input("list takes only end and limit"));
                }
                Ok(())
            }
        }
    }
}

/// Central memories `regions` returns per cluster by default.
pub const DEFAULT_CENTRAL: u32 = 3;
/// Most central memories per cluster.
pub const MAX_CENTRAL: u32 = 10;
/// Longest region label.
pub const MAX_LABEL_CHARS: usize = 80;

/// `regions`: clusters of the current project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegionsParams {
    /// Include 2-dimension coordinates for every memory.
    #[serde(default)]
    pub coords: bool,
    /// Central memories per cluster, 1 to 10; default 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub central: Option<u32>,
    /// Store a name for one cluster of a run, then return the regions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<RegionLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// A name the agent gives a cluster. Cluster numbers are only meaningful
/// within their run (SPEC section 4, Cluster numbers are never keys).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegionLabel {
    pub run: RunId,
    pub region: u32,
    pub label: String,
}

impl RegionsParams {
    pub fn validate(&self) -> Result<()> {
        if let Some(c) = self.central
            && (c == 0 || c > MAX_CENTRAL)
        {
            return Err(CoreError::invalid_input(format!(
                "central must be 1 to {MAX_CENTRAL}"
            )));
        }
        if let Some(l) = &self.label {
            non_blank("label", &l.label)?;
            if l.label.chars().count() > MAX_LABEL_CHARS || l.label.contains(['\n', '\r']) {
                return Err(CoreError::invalid_input(format!(
                    "label must be one line of at most {MAX_LABEL_CHARS} characters"
                )));
            }
        }
        check_limit(self.limit)
    }
}

/// Default interpolation steps for `journey` (version 2's default).
pub const DEFAULT_JOURNEY_STEPS: u32 = 10;
pub const MIN_JOURNEY_STEPS: u32 = 2;
pub const MAX_JOURNEY_STEPS: u32 = 20;

/// `journey`: the nearest memory at each step on the path between two.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JourneyParams {
    pub from: MemoryId,
    pub to: MemoryId,
    /// 2 to 20; default 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// project (default) or all, for the memories found along the path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

impl JourneyParams {
    pub fn validate(&self) -> Result<()> {
        if self.from == self.to {
            return Err(CoreError::invalid_input("from and to are the same memory"));
        }
        if let Some(s) = self.steps
            && !(MIN_JOURNEY_STEPS..=MAX_JOURNEY_STEPS).contains(&s)
        {
            return Err(CoreError::invalid_input(format!(
                "steps must be {MIN_JOURNEY_STEPS} to {MAX_JOURNEY_STEPS}"
            )));
        }
        check_limit(self.limit)
    }
}

/// Default similarity for `consolidate` (version 2's default).
pub const DEFAULT_CONSOLIDATE_THRESHOLD: f64 = 0.85;
pub const MIN_CONSOLIDATE_THRESHOLD: f64 = 0.70;
pub const MAX_CONSOLIDATE_THRESHOLD: f64 = 0.99;
/// Default and most duplicate groups proposed (version 2's values).
pub const DEFAULT_MAX_GROUPS: u32 = 50;
pub const MAX_MAX_GROUPS: u32 = 100;

/// `consolidate`: find near-duplicates and propose merges. Nothing changes
/// unless `apply` is true, and then only the `groups` given are merged: each
/// merged memory is marked superseded and the kept memory is unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConsolidateParams {
    /// Cosine similarity at or above which two memories are duplicates,
    /// 0.70 to 0.99; default 0.85.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(
        default,
        rename = "type",
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub memory_type: Vec<MemoryType>,
    /// Apply the listed `groups`.
    #[serde(default)]
    pub apply: bool,
    /// With `apply`: the merges to make, usually copied from a proposal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<MergeGroup>,
    /// Most groups proposed, 1 to 100; default 50.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_groups: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// One merge: `merge` memories are superseded in favour of `keep`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeGroup {
    pub keep: MemoryId,
    pub merge: Vec<MemoryId>,
}

impl ConsolidateParams {
    pub fn validate(&self) -> Result<()> {
        if let Some(t) = self.threshold
            && !(MIN_CONSOLIDATE_THRESHOLD..=MAX_CONSOLIDATE_THRESHOLD).contains(&t)
        {
            return Err(CoreError::invalid_input(format!(
                "threshold must be {MIN_CONSOLIDATE_THRESHOLD} to {MAX_CONSOLIDATE_THRESHOLD}"
            )));
        }
        if let Some(g) = self.max_groups
            && (g == 0 || g > MAX_MAX_GROUPS)
        {
            return Err(CoreError::invalid_input(format!(
                "max_groups must be 1 to {MAX_MAX_GROUPS}"
            )));
        }
        if self.apply && self.groups.is_empty() {
            return Err(CoreError::invalid_input(
                "apply needs groups; run without apply to get proposals",
            ));
        }
        if !self.apply && !self.groups.is_empty() {
            return Err(CoreError::invalid_input(
                "groups are only used with apply: true",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for g in &self.groups {
            if g.merge.is_empty() {
                return Err(CoreError::invalid_input(
                    "each group needs at least one memory to merge",
                ));
            }
            for id in std::iter::once(&g.keep).chain(&g.merge) {
                if !seen.insert(*id) {
                    return Err(CoreError::invalid_input(format!(
                        "memory {id} appears in more than one place"
                    )));
                }
            }
        }
        check_limit(self.limit)
    }
}

/// `share`: write memories to the repository's shared folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShareParams {
    /// 1 to 100 memories of the current project.
    pub ids: Vec<MemoryId>,
}

impl ShareParams {
    pub fn validate(&self) -> Result<()> {
        if self.ids.is_empty() || self.ids.len() > MAX_REMEMBER_BATCH {
            return Err(CoreError::invalid_input(format!(
                "ids must hold 1 to {MAX_REMEMBER_BATCH} items"
            )));
        }
        Ok(())
    }
}

/// `sync`: load the repository's shared memories into the local store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct SyncParams {
    /// Report what would change without changing it.
    #[serde(default)]
    pub dry_run: bool,
}

/// `status`: daemon health, model, counts per project, index state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct StatusParams {
    /// Include the journal's 7-day medians and 95th percentiles.
    #[serde(default)]
    pub metrics: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// `reinforce`: mark a recalled memory as having helped. Records the rank
/// it was found at; does not change scores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReinforceParams {
    pub id: MemoryId,
    /// The 1-based rank it held in the recall that found it. When absent,
    /// the daemon uses the rank from the latest recall in this project that
    /// returned it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
}

impl ReinforceParams {
    pub fn validate(&self) -> Result<()> {
        if self.rank == Some(0) {
            return Err(CoreError::invalid_input("rank is 1-based"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hook and command-line parameters
// ---------------------------------------------------------------------------

/// `shutdown`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShutdownParams {
    /// Why, for the daemon log: `version_mismatch`, `cli`, `test`.
    pub reason: String,
}

/// The prompt hook's query. The hook resolves the switches (SPEC section
/// 6) before contacting the daemon; the daemon applies the floor, the
/// no-repeat rule and the block budget, and records what it injected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbientParams {
    pub prompt: String,
    /// Result cap, 1 to 50.
    pub top_k: u32,
    pub scope: Scope,
}

impl AmbientParams {
    pub fn validate(&self) -> Result<()> {
        non_blank("prompt", &self.prompt)?;
        check_k(Some(self.top_k))
    }
}

string_enum! {
    /// The hook events, as named on the command line (`spatial-memory hook
    /// <event>`; SPEC section 16, Registration).
    pub enum HookEvent {
        SessionStart => "session-start",
        PostToolUse => "post-tool-use",
        PreCompact => "pre-compact",
        Stop => "stop",
        UserPromptSubmit => "user-prompt-submit",
    }
}

impl HookEvent {
    /// Whether the event hands work to the capture pipeline.
    pub fn is_capture(self) -> bool {
        matches!(
            self,
            HookEvent::PostToolUse | HookEvent::PreCompact | HookEvent::Stop
        )
    }
}

/// A capture hand-off: the hook's input, already normalised to Claude
/// Code's field names (`session_id`, `cwd`, `tool_name`, `tool_input`,
/// `tool_response`, `transcript_path`, `last_assistant_message`,
/// `stop_hook_active`, ...). The daemon replies as soon as it has queued the
/// work. The same request, written to the spool, is how a capture survives
/// a stopped daemon ([`crate::spool`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureParams {
    pub event: HookEvent,
    pub payload: serde_json::Value,
    /// When the hook read the payload.
    pub received_at: Timestamp,
}

/// `spatial-memory replay`: one session's stored transcript, run through
/// the current capture pipeline as if the session had been seen from its
/// start (SPEC 16, Replay; B-172). The daemon replies `Reply::Captured` as
/// soon as the work is queued; the candidates are stored pending, as any
/// capture's are. Like a capture, the request is spooled until stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayParams {
    /// Absolute path of the transcript, resolved by the client.
    pub transcript_path: String,
    /// The session the transcript records: the file's name without
    /// `.jsonl`.
    pub session_id: String,
    /// The session's working directory, which names its project: the
    /// `cwd` the transcript records, absolute.
    pub cwd: String,
    /// When the client asked.
    pub received_at: Timestamp,
}

impl ReplayParams {
    /// Every text field is set; the paths' form is checked by the daemon.
    pub fn validate(&self) -> Result<()> {
        non_blank("transcript_path", &self.transcript_path)?;
        non_blank("session_id", &self.session_id)?;
        non_blank("cwd", &self.cwd)
    }
}

/// `spatial-memory import <file>`: a version 3 export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportParams {
    /// Absolute path, resolved by the client.
    pub path: String,
}

/// `spatial-memory export <file>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportParams {
    /// Absolute path, resolved by the client. An existing file is refused.
    pub path: String,
    /// One project by name; every project when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

/// `spatial-memory migrate --from <file>`: a version 1 or 2 JSON export
/// (SPEC section 10).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrateParams {
    /// Absolute path, resolved by the client.
    pub from: String,
    /// Project for records without a `project` field; the project of the
    /// request's working directory when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_project: Option<String>,
}

/// `spatial-memory reindex`: back up, re-chunk every memory with the
/// configured settings, re-stamp `meta` and queue the embedding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ReindexParams {}

/// `spatial-memory project rename <from> <to>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectRenameParams {
    pub from: String,
    pub to: String,
}

/// `spatial-memory project delete <name>`; refused with `confirm_required`
/// unless `confirm` is set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectDeleteParams {
    pub name: String,
    #[serde(default)]
    pub confirm: bool,
}

/// `spatial-memory doctor --eval <file>` (SPEC section 11, Evaluation on real
/// memories). A set with `memories` is run against a scratch store built
/// from them; a set without is run against the live store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoctorEvalParams {
    /// Absolute path of the evaluation set.
    pub set_path: String,
    /// Absolute path of a baseline to gate against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// One response. `id` repeats the request's id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub v: u32,
    pub build: String,
    pub id: String,
    pub result: Outcome,
}

/// `{"ok": <reply>}` or `{"err": <error>}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok(Box<Reply>),
    Err(CoreError),
}

impl Response {
    pub fn ok(id: impl Into<String>, reply: Reply) -> Self {
        Response {
            v: PROTOCOL_VERSION,
            build: crate::BUILD.to_owned(),
            id: id.into(),
            result: Outcome::Ok(Box::new(reply)),
        }
    }

    pub fn err(id: impl Into<String>, error: CoreError) -> Self {
        Response {
            v: PROTOCOL_VERSION,
            build: crate::BUILD.to_owned(),
            id: id.into(),
            result: Outcome::Err(error),
        }
    }

    /// The reply to a request from another build.
    pub fn version_mismatch(envelope: &Envelope) -> Self {
        let error = envelope
            .check_version()
            .err()
            .unwrap_or_else(|| CoreError::new(ErrorCode::VersionMismatch, "version mismatch"));
        Response::err(envelope.id.clone(), error)
    }

    /// `version_mismatch` unless the daemon is this exact protocol and build.
    pub fn check_version(&self) -> Result<()> {
        check_version(self.v, &self.build)
    }

    pub fn into_result(self) -> Result<Reply> {
        match self.result {
            Outcome::Ok(r) => Ok(*r),
            Outcome::Err(e) => Err(e),
        }
    }
}

/// What a request produced. Serialised as `{"kind": "<name>", "data": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Reply {
    Hello(DaemonInfo),
    ShuttingDown,
    Remember(RememberReply),
    Recall(RecallReply),
    Related(RelatedReply),
    Revise(ReviseReply),
    Forget(ForgetReply),
    Link(LinkReply),
    Regions(RegionsReply),
    Journey(JourneyReply),
    Consolidate(ConsolidateReply),
    Share(ShareReply),
    Sync(SyncReply),
    Status(Box<StatusReply>),
    Reinforce(ReinforceReply),
    Ambient(AmbientReply),
    Captured(CaptureReply),
    /// `import` and `migrate`.
    Import(ImportReport),
    Export(ExportReport),
    Reindex(ReindexReport),
    /// `project_rename` and `project_delete`.
    Project(ProjectReply),
    Eval(Box<EvalReport>),
    DoctorNumbers(DoctorNumbersReply),
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

string_enum! {
    /// Where the daemon is in its life.
    pub enum DaemonState {
        /// Loading and verifying the model; search answers `daemon_loading`.
        Loading => "loading",
        /// Serving everything.
        Ready => "ready",
        /// The model could not be loaded (`state_error` says why). Requests
        /// that need it fail with that error; the rest are served.
        Failed => "failed",
    }
}

/// `hello`, and the daemon part of `status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DaemonInfo {
    pub protocol: u32,
    pub build: String,
    pub pid: u32,
    pub profile: String,
    pub state: DaemonState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_error: Option<CoreError>,
    pub started_at: Timestamp,
    /// Seconds from start to ready (`embedder_cold_start_seconds`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_seconds: Option<f64>,
    pub model_id: String,
    /// The socket path or pipe name.
    pub endpoint: String,
}

/// A memory as every tool returns it: enough to cite (SPEC section 8,
/// Conventions). Empty and default fields are left out to save tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MemoryView {
    pub id: MemoryId,
    /// Project name.
    pub project: String,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
    pub status: Status,
    pub origin: Origin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub content: String,
    /// The view was cut to fit the token budget: `content` was shortened, or
    /// a field (title, tags, anchors) was shed.
    #[serde(default, skip_serializing_if = "is_false")]
    pub content_truncated: bool,
    /// Cosine similarity of the best chunk, for search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Below the similarity floor (explicit recall still returns these).
    #[serde(default, skip_serializing_if = "is_false")]
    pub weak: bool,
    /// 1-based position in the result list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    /// vector or keyword, for search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_by: Option<MatchKind>,
    pub importance: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<AnchorRef>,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes_id: Option<MemoryId>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

string_enum! {
    /// What `remember` did with one item.
    pub enum RememberOutcome {
        Created => "created",
        /// The project already holds a memory with the same content hash;
        /// nothing was stored and `id` is the existing memory.
        Duplicate => "duplicate",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RememberReply {
    /// One entry per input item, in input order.
    pub memories: Vec<Remembered>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Remembered {
    pub id: MemoryId,
    pub project: String,
    pub outcome: RememberOutcome,
    /// Kinds of secret replaced with `[REDACTED:<kind>]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redactions: Vec<String>,
    /// Chunks stored for search.
    pub chunks: u32,
    /// The older memory this one superseded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded: Option<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RecallReply {
    pub results: Vec<MemoryView>,
    /// Every result is below the similarity floor.
    #[serde(default, skip_serializing_if = "is_false")]
    pub weak_only: bool,
    /// Some content was cut to fit the token budget.
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    /// Results left out to fit the token budget.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelatedReply {
    pub from: LinkEnd,
    /// Nearest memories by vector.
    pub by_vector: Vec<MemoryView>,
    /// Memories reached through `links` and `memory_anchors`, up to two hops.
    pub by_links: Vec<LinkedMemory>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkedMemory {
    pub memory: MemoryView,
    /// 1 or 2.
    pub hops: u8,
    /// The edges walked; an anchor shared by two memories counts as a hop
    /// with relation `relates_to`.
    pub via: Vec<Link>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReviseReply {
    /// The memory after the change; for `supersede_with`, the new memory.
    pub memory: MemoryView,
    /// The memory that became superseded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded: Option<MemoryId>,
    /// Content or title changed, so chunks were rebuilt.
    #[serde(default, skip_serializing_if = "is_false")]
    pub reembedded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ForgetReply {
    /// The memories deleted, except captures waiting for the judge.
    pub deleted: Vec<MemoryId>,
    /// Captures waiting for the judge (`pending`) that were deleted too.
    /// They are counted, never listed: no reply shows a pending capture's
    /// id (B-181).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pending: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkReply {
    pub action: LinkAction,
    /// add: the link stored (an existing identical link is returned as is).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<Link>,
    /// remove: links deleted (0 or 1).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub removed: u32,
    /// list: the links of `end`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkView>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkView {
    pub link: Link,
    /// `verified_commit` is no longer an ancestor of the project head.
    /// Absent for links with no commit, or when it cannot be decided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stale: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RegionsReply {
    /// The cached clustering run these numbers belong to.
    pub run: RunId,
    pub project: String,
    pub memories: u32,
    /// False below 30 memories: the project is one region (SPEC section 7).
    pub clustered: bool,
    /// The run came from the cache.
    pub cached: bool,
    pub regions: Vec<Region>,
    /// Memories in no cluster.
    pub unclustered: u32,
    /// With `coords`: every memory's 2-dimension position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<MapPoint>>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Region {
    /// Cluster number, valid only within `run`.
    pub region: u32,
    pub size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The memories nearest the cluster's centre, most central first.
    pub central: Vec<MemoryView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MapPoint {
    pub id: MemoryId,
    /// `None` for an unclustered memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<u32>,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JourneyReply {
    pub from: MemoryId,
    pub to: MemoryId,
    pub steps: Vec<JourneyStep>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JourneyStep {
    /// 0-based step.
    pub step: u32,
    /// Position on the path, 0 at `from` and 1 at `to`.
    pub t: f32,
    /// The nearest memory not already on the path, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConsolidateReply {
    pub applied: bool,
    /// Without `apply`: the proposed groups.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposals: Vec<DuplicateGroup>,
    /// With `apply`: the merges made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merged: Vec<MergeGroup>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DuplicateGroup {
    /// The suggested survivor: the newest active memory of the group.
    pub keep: MemoryView,
    pub duplicates: Vec<Duplicate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Duplicate {
    pub memory: MemoryView,
    /// Cosine similarity to `keep`.
    pub similarity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShareReply {
    pub written: Vec<SharedFile>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refused: Vec<Refusal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SharedFile {
    pub id: MemoryId,
    /// Path relative to the repository root, with `/` separators.
    pub path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redactions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Refusal {
    pub id: MemoryId,
    pub error: CoreError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SyncReply {
    pub dry_run: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<MemoryId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub updated: Vec<MemoryId>,
    /// Shared memories whose file is gone from the repository.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<MemoryId>,
    pub unchanged: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<FileIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FileIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StatusReply {
    pub daemon: DaemonInfo,
    pub model: ModelStatus,
    pub index: IndexStatus,
    /// The project of the request's working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_project: Option<String>,
    pub projects: Vec<ProjectStatus>,
    pub database: String,
    pub database_bytes: u64,
    /// Capture hand-offs waiting in the spool.
    pub spool_pending: u64,
    /// Capture and replay jobs the daemon holds, queued or being processed
    /// (B-172). Zero when the worker has nothing left to do.
    #[serde(default)]
    pub captures_queued: u64,
    /// The capture judge (SPEC section 16, The judge; B-165).
    pub judge: JudgeStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<MetricSummary>>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelStatus {
    pub model_id: String,
    pub revision: String,
    pub dim: u32,
    pub loaded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct IndexStatus {
    /// The store's stamp matches the configured model and chunk settings.
    pub matches: bool,
    /// What differs, when it does not (`index_mismatch`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub differences: Vec<String>,
    pub chunks: u64,
    /// Chunks stored but not yet embedded (bulk queue).
    pub pending_embeddings: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProjectStatus {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_remote: Option<String>,
    pub memories: u64,
    /// Count by type name.
    pub by_type: BTreeMap<String, u64>,
    /// Count by status name.
    pub by_status: BTreeMap<String, u64>,
    pub shared: u64,
}

string_enum! {
    /// Why the capture judge cannot run now (B-165).
    pub enum JudgeBlocked {
        /// `capture.judge.enabled` is false.
        Disabled => "disabled",
        /// Automatic capture is off (`capture.enabled` is false).
        CaptureDisabled => "capture_disabled",
        /// This daemon was started without a judge.
        NoJudge => "no_judge",
        /// Claude Code was not found, `capture.judge.command` names
        /// nothing that runs, or Claude Code is not signed in.
        Unavailable => "unavailable",
        /// Today's calls reached `capture.judge.max_calls_per_day`.
        DailyCap => "daily_cap",
        /// Today's cost reached `capture.judge.max_usd_per_day` (B-193).
        DailyBudget => "daily_budget",
        /// The last call failed; the judge waits before it tries again.
        BackingOff => "backing_off",
    }
}

/// The capture judge's state in `status` (SPEC section 16, The judge;
/// B-165). Days are UTC days.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct JudgeStatus {
    /// `capture.judge.enabled`.
    pub enabled: bool,
    /// Whether the judge can run now. Captures wait as pending while it
    /// cannot.
    pub can_run: bool,
    /// Why it cannot, when it cannot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<JudgeBlocked>,
    /// When a backing-off judge tries again, in RFC 3339 UTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_at: Option<String>,
    /// `capture.judge.model`.
    pub model: String,
    /// The Claude Code executable the judge runs, when one was found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Pending captures in every project, not counting rejected ones: they
    /// are never served.
    pub pending: u64,
    /// Pending captures waiting for the judge; waiting never deletes them
    /// (B-178). The rest were captured before the judge existed, or the
    /// judge gave up on them (`failed`), and wait only to be deleted after
    /// `capture.judge.pending_max_days`.
    pub queued: u64,
    /// Captures the judge rejected, kept 30 days and never served (B-190).
    #[serde(default)]
    pub rejected: u64,
    /// Pending captures the judge gave up on, never served: those marked
    /// failed judgement after two bad answers alone (B-206), and those
    /// whose verdict could not be applied three times. Counted in
    /// `pending`; 0 from a daemon before it.
    #[serde(default)]
    pub failed: u64,
    /// Judge calls today, failed or not.
    pub calls_today: u32,
    /// `capture.judge.max_calls_per_day`.
    pub max_calls_per_day: u32,
    /// Captures judged today.
    pub judged_today: u64,
    /// Today's judge cost in US dollars, as Claude Code reports it. Under a
    /// Claude subscription this estimates plan usage; nothing is charged.
    pub cost_today_usd: f64,
    /// `capture.judge.max_usd_per_day` (B-193); absent from a daemon
    /// before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_usd_per_day: Option<f64>,
    /// Why the last call failed, when the judge has failed since it last
    /// succeeded in this daemon. It never holds memory text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// Automatic capture over the last 7 days, from the journal: captures
    /// judged, kept and rejected, captured memories removed with `forget`,
    /// and the judge's calls and cost (B-200). Absent from a daemon before
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<CaptureQuality>,
}

string_enum! {
    /// Where `reinforce` got the rank it recorded.
    pub enum RankSource {
        Given => "given",
        LastRecall => "last_recall",
        Unknown => "unknown",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReinforceReply {
    pub id: MemoryId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    pub rank_source: RankSource,
}

/// The prompt hook's answer. The hook, not the daemon, records the journal
/// line, because only the hook knows whether its deadline passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbientReply {
    pub gate: HookGate,
    /// The block to print, when `gate` is `injected`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Memories in the block, now recorded in `injections`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub injected: Vec<MemoryId>,
    /// The best scores before the floor, highest first, at most `top_k`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_scores: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureReply {
    /// Queued for the pipeline. False when capture is disabled for the
    /// profile or the project, or the event carries nothing to capture.
    pub accepted: bool,
}

/// `import` and `migrate`. `read` equals `inserted + skipped_existing +
/// issues.len()` (AT-8: the report matches the row count).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportReport {
    pub source: String,
    pub read: u64,
    pub inserted: u64,
    /// Already present by id (an interrupted import is resumed this way).
    pub skipped_existing: u64,
    /// Memories that had a secret redacted.
    pub redacted: u64,
    pub projects_created: Vec<String>,
    pub issues: Vec<ImportIssue>,
    /// Chunks queued for embedding; keyword search works meanwhile.
    pub pending_embeddings: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportIssue {
    /// 0-based record position in the file.
    pub index: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportReport {
    pub path: String,
    pub projects: u64,
    pub memories: u64,
    pub links: u64,
}

/// The embedder's numbers, timed by the daemon on its own sessions (SPEC 5
/// rule 7; B-56, B-125): bulk throughput on the bulk session and the
/// median query latency on the query session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoctorNumbersReply {
    pub physical_cores: u32,
    pub query_threads: u32,
    pub bulk_threads: u32,
    pub batch_size: u32,
    /// Chunks in the timed bulk sample.
    pub sample_chunks: u32,
    /// Tokens in each timed query.
    pub query_tokens: u32,
    /// Timed query runs the median is taken over.
    pub query_runs: u32,
    pub bulk_elapsed_ms: f64,
    pub bulk_chunks_per_sec: f64,
    pub query_median_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReindexReport {
    /// The database copy made first (SPEC section 4, Housekeeping).
    pub backup: String,
    pub memories: u64,
    pub chunks_queued: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectReply {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renamed_from: Option<String>,
    pub deleted: bool,
    /// Memories moved by a rename or removed by a delete.
    pub memories: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AnchorKind;

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    fn profile() -> ProfileName {
        ProfileName::default_profile()
    }

    #[test]
    fn s3_frame_round_trip_and_size_limit() {
        let req = Request::new(
            "/work/repo",
            profile(),
            Op::Recall(RecallParams::new("orders table")),
        );
        let frame = encode_frame(&req).unwrap();
        let len = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        assert_eq!(len, frame.len() - FRAME_PREFIX_BYTES);

        // Two frames back to back on one stream, then a clean close.
        let mut stream = frame.clone();
        stream.extend_from_slice(&frame);
        let mut cursor = std::io::Cursor::new(stream);
        let a: Request = read_message(&mut cursor).unwrap().unwrap();
        let b: Request = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(a, req);
        assert_eq!(b, req);
        assert!(read_message::<_, Request>(&mut cursor).unwrap().is_none());

        // A close inside a prefix or a body is an error, not a clean close.
        let mut short = std::io::Cursor::new(frame[..2].to_vec());
        assert!(read_frame(&mut short).is_err());
        let mut cut = std::io::Cursor::new(frame[..frame.len() - 1].to_vec());
        assert!(read_frame(&mut cut).is_err());

        // Empty and oversized frames are refused before the body is read.
        assert!(frame_len([0, 0, 0, 0]).is_err());
        let too_big = u32::try_from(MAX_FRAME_BYTES + 1).unwrap().to_le_bytes();
        assert_eq!(
            frame_len(too_big).unwrap_err().code,
            ErrorCode::InvalidInput
        );
        assert_eq!(frame_len(1024u32.to_le_bytes()).unwrap(), 1024);

        let mut sink = Vec::new();
        write_frame(&mut sink, &req).unwrap();
        assert_eq!(sink, frame);
    }

    #[test]
    fn s3_request_carries_version_cwd_and_session() {
        let req = Request::new(r"C:\work\repo", profile(), Op::Hello)
            .with_session(Some("abc-123".into()))
            .with_client(Client::ClaudeCode)
            .with_deadline_ms(1500);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["v"], PROTOCOL_VERSION);
        assert_eq!(json["build"], crate::BUILD);
        assert_eq!(json["id"].as_str().unwrap().len(), 12);
        assert_eq!(json["cwd"], r"C:\work\repo");
        assert_eq!(json["session_id"], "abc-123");
        assert_eq!(json["client"], "claude-code");
        assert_eq!(json["profile"], "default");
        assert_eq!(json["deadline_ms"], 1500);
        assert_eq!(json["call"], serde_json::json!({"op": "hello"}));

        // An empty session id is no session id.
        let none = Request::new("/x", profile(), Op::Hello).with_session(Some(String::new()));
        assert!(
            serde_json::to_value(&none)
                .unwrap()
                .get("session_id")
                .is_none()
        );

        // Request ids differ.
        assert_ne!(new_request_id(), new_request_id());
    }

    #[test]
    fn s3_op_tags_are_stable() {
        let ops = vec![
            Op::Hello,
            Op::Shutdown(ShutdownParams {
                reason: "test".into(),
            }),
            Op::AmbientRecall(AmbientParams {
                prompt: "how do we migrate the orders table".into(),
                top_k: 5,
                scope: Scope::Project,
            }),
            Op::Capture(CaptureParams {
                event: HookEvent::PostToolUse,
                payload: serde_json::json!({"tool_name": "Edit"}),
                received_at: Timestamp::from_micros(1),
            }),
            Op::Replay(ReplayParams {
                transcript_path: "/home/u/.claude/projects/p/s-1.jsonl".into(),
                session_id: "s-1".into(),
                cwd: "/work/repo".into(),
                received_at: Timestamp::from_micros(1),
            }),
            Op::Reindex(ReindexParams {}),
            Op::ProjectDelete(ProjectDeleteParams {
                name: "p".into(),
                confirm: false,
            }),
            Op::DoctorEval(DoctorEvalParams {
                set_path: "/e.json".into(),
                baseline_path: None,
            }),
            Op::DoctorNumbers,
        ];
        for op in ops {
            let json = serde_json::to_value(&op).unwrap();
            assert_eq!(json["op"], op.name(), "{json}");
            let back: Op = serde_json::from_value(json).unwrap();
            assert_eq!(back, op);
        }
        assert_eq!(
            serde_json::to_value(Op::DoctorNumbers).unwrap(),
            serde_json::json!({"op": "doctor_numbers"})
        );
        let numbers = Reply::DoctorNumbers(DoctorNumbersReply {
            physical_cores: 16,
            query_threads: 4,
            bulk_threads: 12,
            batch_size: 16,
            sample_chunks: 16,
            query_tokens: 128,
            query_runs: 5,
            bulk_elapsed_ms: 927.0,
            bulk_chunks_per_sec: 17.26,
            query_median_ms: 47.2,
        });
        let json = serde_json::to_value(Response::ok("n", numbers.clone())).unwrap();
        assert_eq!(json["result"]["ok"]["kind"], "doctor_numbers");
        assert_eq!(json["result"]["ok"]["data"]["query_tokens"], 128);
        let back: Response = serde_json::from_value(json).unwrap();
        assert_eq!(back.into_result().unwrap(), numbers);
        let capture = serde_json::to_value(Op::Capture(CaptureParams {
            event: HookEvent::PreCompact,
            payload: serde_json::json!({}),
            received_at: Timestamp::from_micros(0),
        }))
        .unwrap();
        assert_eq!(capture["args"]["event"], "pre-compact");
    }

    #[test]
    fn s3_version_mismatch_reply_shape() {
        // A client of another build sends an op this build does not know.
        let foreign = serde_json::json!({
            "v": PROTOCOL_VERSION + 1, "build": "9.9.9", "id": "aaaaaaaaaaaa",
            "cwd": "/x", "profile": "default", "call": {"op": "teleport", "args": {}}
        });
        let body = serde_json::to_vec(&foreign).unwrap();
        assert!(decode_body::<Request>(&body).is_err());
        let env = Envelope::peek(&body).unwrap();
        assert!(!env.is_shutdown());
        assert_eq!(
            env.check_version().unwrap_err().code,
            ErrorCode::VersionMismatch
        );

        let resp = Response::version_mismatch(&env);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "aaaaaaaaaaaa");
        assert_eq!(json["v"], PROTOCOL_VERSION);
        assert_eq!(json["result"]["err"]["code"], "version_mismatch");
        assert!(json["result"]["err"]["next_action"].as_str().unwrap().len() > 10);

        // Shutdown from any build is recognised without a full decode.
        let shutdown = serde_json::json!({
            "v": 0, "build": "2.0.0", "id": "b", "call": {"op": "shutdown", "args": {"reason": "version_mismatch"}}
        });
        let env = Envelope::peek(&serde_json::to_vec(&shutdown).unwrap()).unwrap();
        assert!(env.is_shutdown());

        // Same build passes; the client checks the daemon's reply the same way.
        let ok = Response::ok("c", Reply::ShuttingDown);
        ok.check_version().unwrap();
        let json = serde_json::to_value(&ok).unwrap();
        assert_eq!(
            json["result"]["ok"],
            serde_json::json!({"kind": "shutting_down"})
        );
        let back: Response = serde_json::from_value(json).unwrap();
        assert_eq!(back.into_result().unwrap(), Reply::ShuttingDown);
    }

    #[test]
    fn s8_thirteen_tools_with_generated_schemas() {
        assert_eq!(Tool::ALL.len(), 13);
        for tool in Tool::ALL {
            let input = serde_json::to_value(tool.input_schema()).unwrap();
            assert_eq!(input["type"], "object", "{tool}");
            assert_eq!(
                input["additionalProperties"], false,
                "{tool} must reject unknown fields"
            );
            let output = serde_json::to_value(tool.output_schema()).unwrap();
            assert_eq!(output["type"], "object", "{tool}");
        }
        let recall = serde_json::to_value(Tool::Recall.input_schema()).unwrap();
        let props = recall["properties"].as_object().unwrap();
        for p in [
            "query",
            "scope",
            "type",
            "anchor",
            "status",
            "mode",
            "k",
            "min_importance",
            "limit",
        ] {
            assert!(props.contains_key(p), "recall lacks {p}");
        }
        assert_eq!(recall["required"], serde_json::json!(["query"]));
    }

    /// Every `format` in `v`: its path, whether its node is an integer or a
    /// number, and the format.
    fn formats(v: &serde_json::Value, path: &str, out: &mut Vec<(String, bool, String)>) {
        match v {
            serde_json::Value::Object(map) => {
                let is_number = |t: &serde_json::Value| {
                    t.as_str().is_some_and(|t| t == "integer" || t == "number")
                };
                let numeric = map.get("type").is_some_and(|t| {
                    is_number(t) || t.as_array().is_some_and(|a| a.iter().any(is_number))
                });
                if let Some(f) = map.get("format") {
                    out.push((path.to_owned(), numeric, f.to_string()));
                }
                for (k, child) in map {
                    formats(child, &format!("{path}/{k}"), out);
                }
            }
            serde_json::Value::Array(items) => {
                for (i, child) in items.iter().enumerate() {
                    formats(child, &format!("{path}/{i}"), out);
                }
            }
            _ => {}
        }
    }

    /// B-154 (D21): Claude Code validates tool schemas with a validator
    /// that knows no `uint32`, and warned `Unknown format "uint32" ignored
    /// in schema at path "#/properties/chunks"` on the first install. No
    /// integer or number in any tool's input or output schema carries a
    /// `format`; the bounds that matter are `minimum` and `maximum`. Every
    /// other format is one JSON Schema 2020-12 defines.
    #[test]
    fn s8_no_tool_schema_carries_a_number_format() {
        const STANDARD: [&str; 19] = [
            "date-time",
            "date",
            "time",
            "duration",
            "email",
            "idn-email",
            "hostname",
            "idn-hostname",
            "ipv4",
            "ipv6",
            "uri",
            "uri-reference",
            "iri",
            "iri-reference",
            "uuid",
            "uri-template",
            "json-pointer",
            "relative-json-pointer",
            "regex",
        ];
        let mut found = Vec::new();
        for tool in Tool::ALL {
            for (side, schema) in [
                ("input", tool.input_schema()),
                ("output", tool.output_schema()),
            ] {
                let v = serde_json::to_value(schema).unwrap();
                formats(&v, &format!("{tool} {side} #"), &mut found);
            }
        }
        let numeric: Vec<_> = found.iter().filter(|(_, n, _)| *n).collect();
        assert!(
            numeric.is_empty(),
            "number formats in tool schemas: {numeric:#?}"
        );
        let unknown: Vec<_> = found
            .iter()
            .filter(|(_, _, f)| !STANDARD.iter().any(|s| f.trim_matches('"') == *s))
            .collect();
        assert!(unknown.is_empty(), "non-standard formats: {unknown:#?}");
        // The unsigned bound the format stood for is kept.
        let remember = serde_json::to_value(Tool::Remember.output_schema()).unwrap();
        let chunks = &remember["$defs"]["Remembered"]["properties"]["chunks"];
        assert_eq!(chunks["type"], "integer", "{remember:#}");
        assert_eq!(chunks["minimum"], 0, "{remember:#}");
    }

    #[test]
    fn s8_tool_args_parse_and_validate() {
        let op = Tool::Recall
            .parse_args(serde_json::json!({"query": "orders", "type": "decision", "status": ["active", "superseded"]}))
            .unwrap();
        let Op::Recall(p) = op else {
            panic!("not recall")
        };
        assert_eq!(p.memory_type, vec![MemoryType::Decision]);
        assert_eq!(p.status, vec![Status::Active, Status::Superseded]);

        // Version 1 parameters are refused by name.
        let err = Tool::Recall
            .parse_args(serde_json::json!({"query": "q", "namespace": "x"}))
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        assert!(err.message.contains("namespace"), "{}", err.message);

        // Bad enum values name the allowed ones.
        let err = Tool::Recall
            .parse_args(serde_json::json!({"query": "q", "type": "idea"}))
            .unwrap_err();
        assert!(err.message.contains("decision"), "{}", err.message);

        for (tool, args) in [
            (Tool::Recall, serde_json::json!({"query": "  "})),
            (Tool::Recall, serde_json::json!({"query": "q", "k": 51})),
            (
                Tool::Recall,
                serde_json::json!({"query": "q", "limit": 199}),
            ),
            (Tool::Remember, serde_json::json!({"memories": []})),
            (
                Tool::Remember,
                serde_json::json!({"memories": [{"content": "x", "importance": 1.5}]}),
            ),
            (Tool::Related, serde_json::json!({})),
            (Tool::Revise, serde_json::json!({"id": id(1).to_string()})),
            (Tool::Forget, serde_json::json!({})),
            (
                Tool::Forget,
                serde_json::json!({"filter": {"scope": "all"}}),
            ),
            (
                Tool::Link,
                serde_json::json!({"action": "add", "src": "table:orders"}),
            ),
            (Tool::Link, serde_json::json!({"action": "list"})),
            (
                Tool::Journey,
                serde_json::json!({"from": id(1).to_string(), "to": id(1).to_string()}),
            ),
            (
                Tool::Journey,
                serde_json::json!({"from": id(1).to_string(), "to": id(2).to_string(), "steps": 21}),
            ),
            (Tool::Consolidate, serde_json::json!({"threshold": 0.5})),
            (Tool::Consolidate, serde_json::json!({"apply": true})),
            (Tool::Share, serde_json::json!({"ids": []})),
            (
                Tool::Reinforce,
                serde_json::json!({"id": id(1).to_string(), "rank": 0}),
            ),
        ] {
            let err = tool.parse_args(args.clone()).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidInput, "{tool} {args}");
        }

        // The smallest limit accepted is the floor itself (B-111).
        Tool::Recall
            .parse_args(serde_json::json!({"query": "q", "limit": MIN_OUTPUT_TOKEN_BUDGET}))
            .unwrap();

        // Valid minimal calls.
        Tool::Status.parse_args(serde_json::Value::Null).unwrap();
        Tool::Sync.parse_args(serde_json::json!({})).unwrap();
        Tool::Remember
            .parse_args(
                serde_json::json!({"memories": [{"content": "Use WAL mode", "type": "decision",
                "anchors": ["database:main"], "tags": ["sqlite"]}]}),
            )
            .unwrap();
        Tool::Forget
            .parse_args(serde_json::json!({"filter": {"tag": "old"}}))
            .unwrap();
        Tool::Link
            .parse_args(serde_json::json!({"action": "add", "src": "service:api", "dst": "table:orders", "relation": "reads"}))
            .unwrap();
        Tool::Consolidate
            .parse_args(serde_json::json!({"apply": true, "groups": [{"keep": id(1).to_string(), "merge": [id(2).to_string()]}]}))
            .unwrap();
        let err = Tool::Consolidate
            .parse_args(serde_json::json!({"apply": true, "groups": [
                {"keep": id(1).to_string(), "merge": [id(2).to_string()]},
                {"keep": id(3).to_string(), "merge": [id(2).to_string()]}]}))
            .unwrap_err();
        assert!(err.message.contains("more than one place"));
    }

    #[test]
    fn s8_revise_supersede_excludes_other_edits() {
        let mut p = ReviseParams::new(id(1));
        p.supersede_with = Some(RememberItem::new("new decision"));
        p.validate().unwrap();
        p.status = Some(Status::Retracted);
        assert!(p.validate().is_err());
        let mut q = ReviseParams::new(id(1));
        q.status = Some(Status::Retracted);
        q.validate().unwrap();
    }

    #[test]
    fn s16_b172_a_replay_names_its_transcript_session_and_folder() {
        let p = ReplayParams {
            transcript_path: "/home/u/.claude/projects/p/s-1.jsonl".into(),
            session_id: "s-1".into(),
            cwd: "/work/repo".into(),
            received_at: Timestamp::from_micros(1),
        };
        Op::Replay(p.clone()).validate().unwrap();
        for blank in [
            ReplayParams {
                transcript_path: " ".into(),
                ..p.clone()
            },
            ReplayParams {
                session_id: String::new(),
                ..p.clone()
            },
            ReplayParams {
                cwd: String::new(),
                ..p.clone()
            },
        ] {
            let e = Op::Replay(blank).validate().unwrap_err();
            assert_eq!(e.code, crate::error::ErrorCode::InvalidInput, "{e:?}");
        }
    }

    #[test]
    fn s8_memory_view_omits_empty_fields() {
        let view = MemoryView {
            id: id(7),
            project: "shop".into(),
            memory_type: MemoryType::Decision,
            status: Status::Active,
            origin: Origin::Local,
            title: None,
            content: "Orders are soft-deleted.".into(),
            content_truncated: false,
            score: Some(0.81),
            weak: false,
            rank: Some(1),
            matched_by: Some(MatchKind::Vector),
            importance: 0.5,
            tags: Vec::new(),
            anchors: vec![AnchorRef::new(AnchorKind::Table, "orders").unwrap()],
            provenance: Provenance::new(crate::types::ProvenanceSource::Agent),
            supersedes_id: None,
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_value(&view).unwrap();
        let obj = json.as_object().unwrap();
        for absent in [
            "title",
            "content_truncated",
            "weak",
            "tags",
            "supersedes_id",
        ] {
            assert!(!obj.contains_key(absent), "{absent} should be omitted");
        }
        assert_eq!(json["type"], "decision");
        assert_eq!(json["anchors"], serde_json::json!(["table:orders"]));
        let back: MemoryView = serde_json::from_value(json).unwrap();
        assert_eq!(back, view);

        let reply = Response::ok(
            "r",
            Reply::Recall(RecallReply {
                results: vec![view],
                weak_only: false,
                truncated: false,
                omitted: 0,
            }),
        );
        let back: Response = serde_json::from_slice(&serde_json::to_vec(&reply).unwrap()).unwrap();
        assert_eq!(back, reply);
    }

    #[test]
    fn s16_hook_events_use_version_2_names() {
        let names: Vec<&str> = HookEvent::ALL.iter().map(|e| e.as_str()).collect();
        assert_eq!(
            names,
            [
                "session-start",
                "post-tool-use",
                "pre-compact",
                "stop",
                "user-prompt-submit"
            ]
        );
        let capture: Vec<HookEvent> = HookEvent::ALL
            .iter()
            .copied()
            .filter(|e| e.is_capture())
            .collect();
        assert_eq!(
            capture,
            [
                HookEvent::PostToolUse,
                HookEvent::PreCompact,
                HookEvent::Stop
            ]
        );
    }

    #[test]
    fn s16_recall_and_revise_refuse_pending() {
        // B-157: no request can ask for unjudged captures, and only capture
        // sets `pending`.
        let mut r = RecallParams::new("deploy days");
        for statuses in [vec![Status::Pending], vec![Status::Active, Status::Pending]] {
            r.status = statuses;
            let e = r.validate().unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidInput);
            assert!(e.message.contains("never served"), "{}", e.message);
        }
        r.status = vec![Status::Active, Status::Superseded, Status::Retracted];
        r.validate().unwrap();

        let id = MemoryId::new_v7();
        let mut v = ReviseParams::new(id);
        v.status = Some(Status::Pending);
        let e = v.validate().unwrap_err();
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(
            e.message.contains("only by automatic capture"),
            "{}",
            e.message
        );
        // Keeping a pending capture by hand is allowed.
        v.status = Some(Status::Active);
        v.validate().unwrap();

        // A filtered forget may name pending captures.
        let f: ForgetParams = serde_json::from_value(serde_json::json!({
            "filter": {"status": "pending"}, "confirm": true
        }))
        .unwrap();
        assert_eq!(f.filter.unwrap().status, [Status::Pending]);
    }
}
