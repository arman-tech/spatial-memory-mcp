//! Domain types (SPEC section 4, Data model).

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{CoreError, ErrorCode, Result};
use crate::ids::{MemoryId, ProjectId};
use crate::time::Timestamp;

string_enum! {
    /// What kind of knowledge a memory holds (SPEC section 4, Types and status).
    #[derive(Default)]
    pub enum MemoryType {
        Decision => "decision",
        Pitfall => "pitfall",
        Convention => "convention",
        Incident => "incident",
        Runbook => "runbook",
        #[default]
        Note => "note",
    }
}

string_enum! {
    /// Lifecycle state. Recall returns active memories only unless asked.
    /// `pending` is an automatic capture that has not been judged yet
    /// (SPEC 16, B-157): it is never served, whatever a caller asks for.
    #[derive(Default)]
    pub enum Status {
        #[default]
        Active => "active",
        Superseded => "superseded",
        Retracted => "retracted",
        Pending => "pending",
    }
}

impl Status {
    /// Whether recall, ambient recall, related, maps and share may return a
    /// memory with this status. Only `pending` is never served (B-157).
    pub fn is_served(self) -> bool {
        self != Status::Pending
    }
}

string_enum! {
    /// Where the memory lives: only in the local store, or in the
    /// repository's shared folder (SPEC section 9).
    #[derive(Default)]
    pub enum Origin {
        #[default]
        Local => "local",
        Shared => "shared",
    }
}

string_enum! {
    /// Who or what produced a memory (SPEC section 4, `provenance.source`).
    pub enum ProvenanceSource {
        Agent => "agent",
        Human => "human",
        Capture => "capture",
        Import => "import",
        Sync => "sync",
    }
}

string_enum! {
    /// Things in the codebase a memory can be about (SPEC section 4, `anchors.kind`).
    pub enum AnchorKind {
        Project => "project",
        Service => "service",
        Database => "database",
        Table => "table",
        Endpoint => "endpoint",
        File => "file",
        Symbol => "symbol",
        Commit => "commit",
    }
}

string_enum! {
    /// The closed set of link relations (SPEC section 4, Relations).
    pub enum Relation {
        Calls => "calls",
        Reads => "reads",
        Writes => "writes",
        DependsOn => "depends_on",
        RelatesTo => "relates_to",
        CausedBy => "caused_by",
    }
}

string_enum! {
    /// How a link was made. Only `manual` is produced in 3.0 (SPEC section 12).
    pub enum LinkSource {
        Manual => "manual",
        Extracted => "extracted",
    }
}

string_enum! {
    /// Search scope. Every read defaults to the current project; `all` must be
    /// passed explicitly (SPEC section 9, Scope).
    #[derive(Default)]
    pub enum Scope {
        #[default]
        Project => "project",
        All => "all",
    }
}

string_enum! {
    /// Recall path (SPEC section 8, `recall`; section 6). `auto` runs vector
    /// search and adds keyword hits when the query looks like an identifier.
    #[derive(Default)]
    pub enum RecallMode {
        #[default]
        Auto => "auto",
        Vector => "vector",
        Keyword => "keyword",
    }
}

string_enum! {
    /// Which path found a result. Keyword hits are appended to vector hits and
    /// marked; the two are never score-fused (SPEC section 6).
    pub enum MatchKind {
        Vector => "vector",
        Keyword => "keyword",
    }
}

string_enum! {
    /// The client a hook runs for (SPEC section 16, Clients).
    #[derive(Default)]
    pub enum Client {
        #[default]
        ClaudeCode => "claude-code",
        Cursor => "cursor",
    }
}

/// Longest anchor name, in characters.
pub const MAX_ANCHOR_NAME_CHARS: usize = 256;
/// Longest tag, in characters.
pub const MAX_TAG_CHARS: usize = 64;
/// Longest derived title, in characters.
pub const MAX_DERIVED_TITLE_CHARS: usize = 80;

/// Characters that cannot appear in an anchor name or a tag, because both are
/// written as items of a one-line list in shared front matter (SPEC section 9).
fn is_list_breaking(c: char) -> bool {
    c.is_control() || matches!(c, ',' | '[' | ']')
}

fn check_list_item(what: &str, s: &str, max_chars: usize) -> Result<()> {
    if s.is_empty() {
        return Err(CoreError::invalid_input(format!("{what} is empty")));
    }
    if s.trim() != s {
        return Err(CoreError::invalid_input(format!(
            "{what} {s:?} has leading or trailing whitespace"
        )));
    }
    if s.chars().count() > max_chars {
        return Err(CoreError::invalid_input(format!(
            "{what} {s:?} is longer than {max_chars} characters"
        )));
    }
    if let Some(c) = s.chars().find(|c| is_list_breaking(*c)) {
        return Err(CoreError::invalid_input(format!(
            "{what} {s:?} contains {c:?}; control characters, ',', '[' and ']' are not allowed"
        )));
    }
    Ok(())
}

/// `kind:name`, for example `table:dbo.Orders` or `file:src/lib.rs`. The name
/// may itself contain colons (`symbol:Foo::bar`). File anchors are
/// repository-relative with `/` separators when the file is inside the
/// project, absolute otherwise (see `project::normalize_file_anchor`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnchorRef {
    pub kind: AnchorKind,
    pub name: String,
}

impl AnchorRef {
    pub fn new(kind: AnchorKind, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        check_list_item("anchor name", &name, MAX_ANCHOR_NAME_CHARS)?;
        Ok(AnchorRef { kind, name })
    }
}

impl fmt::Display for AnchorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.name)
    }
}

impl FromStr for AnchorRef {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        let (kind, name) = s.split_once(':').ok_or_else(|| {
            CoreError::invalid_input(format!("anchor {s:?} must be written kind:name"))
        })?;
        AnchorRef::new(kind.parse()?, name)
    }
}

impl Serialize for AnchorRef {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AnchorRef {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse()
            .map_err(|e: CoreError| serde::de::Error::custom(e.message))
    }
}

impl JsonSchema for AnchorRef {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("AnchorRef")
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": "^(project|service|database|table|endpoint|file|symbol|commit):[^,\\[\\]]+$",
            "description": "kind:name, where kind is project, service, database, table, endpoint, file, symbol or commit. Examples: table:dbo.Orders, file:src/orders.rs, symbol:OrderService.GetOrderAsync."
        })
    }
}

/// One end of a link: a memory or an anchor. Written `memory:<uuid>` or
/// `kind:name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LinkEnd {
    Memory(MemoryId),
    Anchor(AnchorRef),
}

impl LinkEnd {
    /// The `src_kind`/`dst_kind` column value: `memory` or the anchor kind.
    pub fn kind_str(&self) -> &'static str {
        match self {
            LinkEnd::Memory(_) => "memory",
            LinkEnd::Anchor(a) => a.kind.as_str(),
        }
    }
}

impl fmt::Display for LinkEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkEnd::Memory(id) => write!(f, "memory:{id}"),
            LinkEnd::Anchor(a) => fmt::Display::fmt(a, f),
        }
    }
}

impl FromStr for LinkEnd {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        match s.strip_prefix("memory:") {
            Some(id) => Ok(LinkEnd::Memory(MemoryId::parse(id)?)),
            None => Ok(LinkEnd::Anchor(s.parse()?)),
        }
    }
}

impl Serialize for LinkEnd {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for LinkEnd {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse()
            .map_err(|e: CoreError| serde::de::Error::custom(e.message))
    }
}

impl JsonSchema for LinkEnd {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("LinkEnd")
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "memory:<memory id>, or an anchor written kind:name."
        })
    }
}

/// A directed edge (SPEC section 4, `links`). A link is stale when its
/// `verified_commit` is no longer an ancestor of the project head; stale
/// links are reported, never deleted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Link {
    pub src: LinkEnd,
    pub dst: LinkEnd,
    pub relation: Relation,
    pub source: LinkSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extractor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified_at: Option<Timestamp>,
}

/// Where a memory came from (SPEC section 4, `provenance`). `tool` and
/// `extra` extend the specification's key columns: `tool` is the capturing
/// tool (SPEC section 16, Type and provenance) and `extra` keeps version 1's
/// `source` and `metadata` on migration (SPEC section 10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Provenance {
    pub source: ProvenanceSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl Provenance {
    pub fn new(source: ProvenanceSource) -> Self {
        Provenance {
            source,
            author: None,
            session_id: None,
            commit_sha: None,
            file_path: None,
            tool: None,
            extra: None,
        }
    }
}

/// A `projects` row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    /// Normalised remote, `host/path` (see `project::normalize_remote_url`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_remote: Option<String>,
    /// Canonical root folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_path: Option<String>,
}

/// A stored memory with its provenance and anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Memory {
    pub id: MemoryId,
    pub project_id: ProjectId,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub content: String,
    pub content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes_id: Option<MemoryId>,
    pub origin: Origin,
    pub importance: f64,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Version 1 fields kept for reference only (SPEC section 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported: Option<serde_json::Value>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub provenance: Provenance,
    #[serde(default)]
    pub anchors: Vec<AnchorRef>,
}

/// Input for inserting a memory. The store fills what is `None`: a new UUID
/// version 7, the content hash of `content`, and the current time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewMemory {
    pub id: Option<MemoryId>,
    pub project_id: ProjectId,
    pub memory_type: MemoryType,
    pub status: Status,
    pub title: Option<String>,
    pub content: String,
    pub content_hash: Option<String>,
    pub supersedes_id: Option<MemoryId>,
    pub origin: Origin,
    pub importance: f64,
    pub tags: Vec<String>,
    pub imported: Option<serde_json::Value>,
    pub created_at: Option<Timestamp>,
    pub updated_at: Option<Timestamp>,
    pub provenance: Provenance,
    pub anchors: Vec<AnchorRef>,
}

impl NewMemory {
    /// An active local note with default importance and no extras.
    pub fn new(project_id: ProjectId, content: impl Into<String>, provenance: Provenance) -> Self {
        NewMemory {
            id: None,
            project_id,
            memory_type: MemoryType::Note,
            status: Status::Active,
            title: None,
            content: content.into(),
            content_hash: None,
            supersedes_id: None,
            origin: Origin::Local,
            importance: DEFAULT_IMPORTANCE,
            tags: Vec::new(),
            imported: None,
            created_at: None,
            updated_at: None,
            provenance,
            anchors: Vec::new(),
        }
    }
}

/// Importance of an explicit `remember` without one.
pub const DEFAULT_IMPORTANCE: f64 = 0.5;

/// Checks an importance value: finite and within 0 to 1.
pub fn check_importance(value: f64) -> Result<f64> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(CoreError::invalid_input(format!(
            "importance {value} must be between 0 and 1"
        )))
    }
}

/// Validates tags from a tool call: each trimmed, non-empty, at most 64
/// characters, with no control characters, `,`, `[` or `]`. Duplicates are
/// removed keeping the first; case is kept.
pub fn normalize_tags<I, S>(tags: I) -> Result<Vec<String>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let t = t.as_ref().trim();
        check_list_item("tag", t, MAX_TAG_CHARS)?;
        if !out.iter().any(|x| x == t) {
            out.push(t.to_owned());
        }
    }
    Ok(out)
}

/// Makes a tag from imported data that may break the rules: forbidden
/// characters become `-`, whitespace runs become one space, and the result is
/// cut to 64 characters. `None` when nothing is left.
pub fn sanitize_tag(raw: &str) -> Option<String> {
    let replaced: String = raw
        .chars()
        .map(|c| {
            if is_list_breaking(c) && !c.is_whitespace() {
                '-'
            } else {
                c
            }
        })
        .collect();
    let collapsed = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = collapsed.chars().take(MAX_TAG_CHARS).collect();
    let cut = cut.trim().to_owned();
    if cut.is_empty() { None } else { Some(cut) }
}

/// A title for a memory that has none: the first line with text, without
/// leading `#` marks, with whitespace collapsed, cut to 80 characters.
/// `None` when the content has no text.
pub fn derive_title(content: &str) -> Option<String> {
    for line in content.lines() {
        let text = line.trim().trim_start_matches('#');
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            continue;
        }
        let cut: String = collapsed.chars().take(MAX_DERIVED_TITLE_CHARS).collect();
        return Some(cut.trim_end().to_owned());
    }
    None
}

/// `meta` table keys (SPEC section 4, `meta`).
pub mod meta_keys {
    pub const SCHEMA_VERSION: &str = "schema_version";
    pub const EMBED_MODEL: &str = "embed_model";
    pub const EMBED_REVISION: &str = "embed_revision";
    pub const EMBED_DIM: &str = "embed_dim";
    pub const CHUNK_MAX_TOKENS: &str = "chunk_max_tokens";
    pub const CHUNK_OVERLAP: &str = "chunk_overlap";
}

/// What the index was built with. A daemon whose configured stamp differs
/// from the stored one refuses to search until `reindex` has run (SPEC
/// section 4, Model stamp; AT-11).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct IndexStamp {
    pub model_id: String,
    pub revision: String,
    pub dim: usize,
    pub chunk_max_tokens: u32,
    pub chunk_overlap: f64,
}

impl IndexStamp {
    /// The `meta` rows that record this stamp. The overlap is written with
    /// Rust's shortest round-trip form (`0.125`).
    pub fn to_meta(&self) -> Vec<(&'static str, String)> {
        vec![
            (meta_keys::EMBED_MODEL, self.model_id.clone()),
            (meta_keys::EMBED_REVISION, self.revision.clone()),
            (meta_keys::EMBED_DIM, self.dim.to_string()),
            (
                meta_keys::CHUNK_MAX_TOKENS,
                self.chunk_max_tokens.to_string(),
            ),
            (meta_keys::CHUNK_OVERLAP, format!("{}", self.chunk_overlap)),
        ]
    }

    /// Reads a stamp from `meta`. `Ok(None)` when no stamp has been written
    /// (a new store); an error when it is partial or unreadable.
    pub fn from_meta(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let keys = [
            meta_keys::EMBED_MODEL,
            meta_keys::EMBED_REVISION,
            meta_keys::EMBED_DIM,
            meta_keys::CHUNK_MAX_TOKENS,
            meta_keys::CHUNK_OVERLAP,
        ];
        let values: Vec<Option<String>> = keys.iter().map(|k| get(k)).collect();
        if values.iter().all(Option::is_none) {
            return Ok(None);
        }
        let bad = |key: &str| {
            CoreError::new(
                ErrorCode::Storage,
                format!("index stamp key {key} is missing or invalid"),
            )
            .with_next_action("Run `spatial-memory reindex` to rebuild the index and its stamp.")
        };
        let v = |i: usize| values[i].clone().ok_or_else(|| bad(keys[i]));
        Ok(Some(IndexStamp {
            model_id: v(0)?,
            revision: v(1)?,
            dim: v(2)?.parse().map_err(|_| bad(keys[2]))?,
            chunk_max_tokens: v(3)?.parse().map_err(|_| bad(keys[3]))?,
            chunk_overlap: v(4)?.parse().map_err(|_| bad(keys[4]))?,
        }))
    }

    /// Human-readable differences between the stored stamp (`self`) and the
    /// configured one. Empty when they match.
    pub fn differences(&self, configured: &IndexStamp) -> Vec<String> {
        let mut out = Vec::new();
        if self.model_id != configured.model_id {
            out.push(format!(
                "model {} in the index, {} configured",
                self.model_id, configured.model_id
            ));
        }
        if self.revision != configured.revision {
            out.push(format!(
                "model revision {} in the index, {} configured",
                self.revision, configured.revision
            ));
        }
        if self.dim != configured.dim {
            out.push(format!(
                "dimension {} in the index, {} configured",
                self.dim, configured.dim
            ));
        }
        if self.chunk_max_tokens != configured.chunk_max_tokens {
            out.push(format!(
                "chunk.max_tokens {} in the index, {} configured",
                self.chunk_max_tokens, configured.chunk_max_tokens
            ));
        }
        if self.chunk_overlap != configured.chunk_overlap {
            out.push(format!(
                "chunk.overlap {} in the index, {} configured",
                self.chunk_overlap, configured.chunk_overlap
            ));
        }
        out
    }

    /// `Err(index_mismatch)` naming every difference, or `Ok(())`.
    pub fn check_matches(&self, configured: &IndexStamp) -> Result<()> {
        let diffs = self.differences(configured);
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(CoreError::new(
                ErrorCode::IndexMismatch,
                format!(
                    "the index was built with different settings: {}",
                    diffs.join("; ")
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_enums_use_spec_strings() {
        let types: Vec<&str> = MemoryType::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(
            types,
            [
                "decision",
                "pitfall",
                "convention",
                "incident",
                "runbook",
                "note"
            ]
        );
        let statuses: Vec<&str> = Status::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(statuses, ["active", "superseded", "retracted", "pending"]);
        let kinds: Vec<&str> = AnchorKind::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "project", "service", "database", "table", "endpoint", "file", "symbol", "commit"
            ]
        );
        let rel: Vec<&str> = Relation::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(
            rel,
            [
                "calls",
                "reads",
                "writes",
                "depends_on",
                "relates_to",
                "caused_by"
            ]
        );
        let src: Vec<&str> = ProvenanceSource::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(src, ["agent", "human", "capture", "import", "sync"]);
        assert_eq!(Client::ClaudeCode.as_str(), "claude-code");
        assert_eq!("cursor".parse::<Client>().unwrap(), Client::Cursor);
        assert!("Decision".parse::<MemoryType>().is_err());
    }

    #[test]
    fn s8_enum_schema_lists_values() {
        let schema = schemars::schema_for!(Status);
        let v = serde_json::to_value(&schema).unwrap();
        assert_eq!(
            v["enum"],
            serde_json::json!(["active", "superseded", "retracted", "pending"])
        );
    }

    #[test]
    fn s4_anchor_refs_parse_and_print() {
        let a: AnchorRef = "symbol:Foo::bar".parse().unwrap();
        assert_eq!(a.kind, AnchorKind::Symbol);
        assert_eq!(a.name, "Foo::bar");
        assert_eq!(a.to_string(), "symbol:Foo::bar");
        let e: AnchorRef = "endpoint:GET /api/orders/{id}".parse().unwrap();
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            "\"endpoint:GET /api/orders/{id}\""
        );
        for bad in [
            "table",
            "nope:x",
            "table:",
            "table: x",
            "table:a,b",
            "table:[x]",
            "table:a\nb",
        ] {
            assert!(bad.parse::<AnchorRef>().is_err(), "{bad:?}");
        }
        let long = format!("table:{}", "x".repeat(257));
        assert!(long.parse::<AnchorRef>().is_err());
    }

    #[test]
    fn s4_link_ends_distinguish_memories_and_anchors() {
        let m: LinkEnd = "memory:0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d"
            .parse()
            .unwrap();
        assert!(matches!(m, LinkEnd::Memory(_)));
        assert_eq!(m.kind_str(), "memory");
        let a: LinkEnd = "service:order-api".parse().unwrap();
        assert_eq!(a.kind_str(), "service");
        assert_eq!(a.to_string(), "service:order-api");
        assert!("memory:nope".parse::<LinkEnd>().is_err());
    }

    #[test]
    fn s9_tags_are_validated_and_deduplicated() {
        let t = normalize_tags([" sql ", "sql", "SQL", "concurrency"]).unwrap();
        assert_eq!(t, ["sql", "SQL", "concurrency"]);
        assert!(normalize_tags(["a,b"]).is_err());
        assert!(normalize_tags([""]).is_err());
        assert!(normalize_tags(["x".repeat(65)]).is_err());
        assert_eq!(sanitize_tag(" a,b [c] "), Some("a-b -c-".to_owned()));
        assert_eq!(sanitize_tag(" \t "), None);
        assert_eq!(sanitize_tag(&"y".repeat(80)).unwrap().chars().count(), 64);
    }

    #[test]
    fn s4_derive_title_takes_first_text_line() {
        assert_eq!(
            derive_title("\n\n# Release checklist\n\nbody").as_deref(),
            Some("Release checklist")
        );
        assert_eq!(
            derive_title("###\n  Two   words  here ").as_deref(),
            Some("Two words here")
        );
        assert_eq!(derive_title(" \n\t"), None);
        let long = "word ".repeat(40);
        assert!(derive_title(&long).unwrap().chars().count() <= 80);
    }

    #[test]
    fn s4_importance_must_be_in_range() {
        assert!(check_importance(0.0).is_ok());
        assert!(check_importance(1.0).is_ok());
        assert!(check_importance(1.01).is_err());
        assert!(check_importance(f64::NAN).is_err());
    }

    fn stamp() -> IndexStamp {
        IndexStamp {
            model_id: "embeddinggemma-2".into(),
            revision: "daa72c51243991dfcaf9f9137d2c573d8f7790c0".into(),
            dim: 768,
            chunk_max_tokens: 256,
            chunk_overlap: 0.125,
        }
    }

    #[test]
    fn at11_index_stamp_round_trips_through_meta() {
        let s = stamp();
        let rows = s.to_meta();
        let get = |k: &str| {
            rows.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(IndexStamp::from_meta(get).unwrap(), Some(s.clone()));
        assert_eq!(IndexStamp::from_meta(|_| None).unwrap(), None);
        let partial = |k: &str| (k == meta_keys::EMBED_MODEL).then(|| "x".to_owned());
        assert!(IndexStamp::from_meta(partial).is_err());
    }

    #[test]
    fn at11_changed_model_or_chunk_size_is_a_mismatch() {
        let stored = stamp();
        assert!(stored.check_matches(&stamp()).is_ok());
        let mut other = stamp();
        other.chunk_max_tokens = 512;
        let e = stored.check_matches(&other).unwrap_err();
        assert_eq!(e.code, ErrorCode::IndexMismatch);
        assert!(e.message.contains("chunk.max_tokens 256"));
        let mut other = stamp();
        other.model_id = "arctic-embed-m".into();
        assert_eq!(stored.differences(&other).len(), 1);
    }
}
