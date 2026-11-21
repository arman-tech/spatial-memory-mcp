//! Configuration (SPEC section 5, Configuration; section 6, the hook's gates
//! and switches; section 9, Profiles; section 15, defaults; section 16,
//! Controls). Decision B-11 records the layering.
//!
//! Two files, both optional:
//!
//! - `<root>/config.toml`, shared by every profile. Besides settings it may
//!   hold `[[profile_rule]]` entries (`path`, `profile`) that pick a profile
//!   by working directory.
//! - `<root>/<profile>/config.toml`, the profile's own settings.
//!
//! The profile file is merged over the root file table by table, then the
//! result is read into [`Settings`] and validated. An unknown key is an error
//! (`config_invalid`), so a misspelt key is reported rather than ignored.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::hash::python_strip;
use crate::project::path_key;
use crate::types::{FILE_WRITE_TOOLS, Scope};

string_enum! {
    /// The embedding model (SPEC section 5, `embed.model`). Changing it
    /// requires `reindex`.
    pub enum EmbedModel {
        /// EmbeddingGemma 2 through ONNX Runtime, pinned in SPEC section 5.
        EmbeddingGemma2 => "embeddinggemma-2",
        /// Snowflake Arctic Embed M through `fastembed`, the fallback.
        ArcticEmbedM => "arctic-embed-m",
    }
}

#[allow(clippy::derivable_impls)]
impl Default for EmbedModel {
    fn default() -> Self {
        EmbedModel::EmbeddingGemma2
    }
}

/// The pinned EmbeddingGemma 2 revision (SPEC section 5, Model, pinned).
pub const EMBEDDINGGEMMA_REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

/// Version 2's default result cap for the prompt hook (SPEC section 6).
pub const DEFAULT_AMBIENT_TOP_K: u32 = 5;
/// Upper bound on any `k` or `top_k` (decision B-11).
pub const MAX_K: u32 = 50;

/// `[chunk]` (SPEC section 5). Both keys are stamped into the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChunkSettings {
    /// Chunk content limit in model tokens, 128 to 1,024.
    pub max_tokens: u32,
    /// Window overlap for an oversized piece, 0 to 0.25.
    pub overlap: f64,
}

impl Default for ChunkSettings {
    fn default() -> Self {
        ChunkSettings {
            max_tokens: 256,
            overlap: 0.125,
        }
    }
}

/// `[embed]` (SPEC section 5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbedSettings {
    pub model: EmbedModel,
    /// Chunks per bulk inference, 1 to 16.
    pub batch_size: u32,
    /// Threads for the query session; `None` means 4. Clamped to the
    /// physical core count at use.
    pub query_threads: Option<u32>,
    /// ONNX Runtime thread spinning; off by default.
    pub spinning: bool,
}

impl Default for EmbedSettings {
    fn default() -> Self {
        EmbedSettings {
            model: EmbedModel::default(),
            batch_size: 4,
            query_threads: None,
            spinning: false,
        }
    }
}

/// `[cpu]` (SPEC section 5, Using the machine's cores).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuSettings {
    /// Share of physical cores bulk work may use, 0.1 to 1.0.
    pub budget: f64,
}

impl Default for CpuSettings {
    fn default() -> Self {
        CpuSettings { budget: 0.75 }
    }
}

/// The tools PostToolUse captures by default (SPEC section 16): none.
///
/// A tool call's material is a command and its output, or code; none of it
/// is a conclusion, and 36 of 37 such captures in the owner's export were
/// not worth keeping (RC8, decision B-160). Versions 1 and 2 captured
/// `Edit`, `Write`, `Bash` and `NotebookEdit`; a user who wants that names
/// them in `capture.tools`.
pub const DEFAULT_CAPTURE_TOOLS: [&str; 0] = [];

/// Paths never captured, by default (SPEC section 16, Controls: "`.env`
/// files and key files"; decision B-21 lists them).
pub const DEFAULT_CAPTURE_IGNORE: [&str; 13] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
];

/// `[capture]` (SPEC section 16, Controls and Index cap).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureSettings {
    pub enabled: bool,
    /// Tool names PostToolUse captures, as Claude Code names them.
    pub tools: Vec<String>,
    /// Glob patterns for paths never captured. A pattern without `/` is
    /// matched against every path component; one with `/` against the
    /// project-relative path. `*` and `?` stop at `/`; `**` does not.
    pub ignore: Vec<String>,
    /// Tokens of one capture that are indexed (embedded); the text is
    /// stored whole. 256 and up.
    pub max_index_tokens: u32,
    /// Whether capture offers text read from files and command output: the
    /// prose of a Bash, Read or Grep result in the transcript (B-179's
    /// `tool_result`), and a PostToolUse capture of any tool but Edit,
    /// MultiEdit, Write and NotebookEdit. Recall tags such a memory
    /// "file or command output, unverified" (B-220). `false` stops every
    /// such capture, replay included; nothing already stored is changed
    /// (decision B-221). On by default.
    pub tool_results: bool,
    /// `[capture.judge]`: the Claude Code call that decides which captures
    /// are kept (SPEC section 16, The judge; decision B-165).
    pub judge: JudgeSettings,
}

/// The judge's model by default: Claude Code's alias for its smallest
/// current model (B-158, B-165).
pub const DEFAULT_JUDGE_MODEL: &str = "haiku";

/// Judge calls a day, by default (B-165, B-209). Duplicate checks (B-192)
/// count as calls. From 30 days of the owner's transcripts, the 95th
/// percentile day is about 487 calls and the busiest day about 565. The cap
/// leaves headroom above both. On a busier day the rest wait. The dollar cap
/// ([`DEFAULT_JUDGE_MAX_USD_PER_DAY`]) is the hard limit.
pub const DEFAULT_JUDGE_MAX_CALLS_PER_DAY: u32 = 600;

/// The judge's spending a UTC day, by default, in US dollars at API prices
/// as Claude Code reports them (B-193). The owner's typical day costs
/// about 0.43 with haiku and the duplicate check, the busiest day of the
/// 31 measured about 1.40 (`CAPTURE-RCA.md` section 8.11).
pub const DEFAULT_JUDGE_MAX_USD_PER_DAY: f64 = 3.0;

/// Days before a capture the judge gave up on is deleted, by default
/// (B-165, B-178).
pub const DEFAULT_JUDGE_PENDING_MAX_DAYS: u32 = 14;

/// `[capture.judge]` (SPEC section 16, The judge; decision B-165).
///
/// Every automatic capture waits as `pending` until the judge keeps it
/// (`active`) or rejects it (kept pending and never served for 30 days,
/// then deleted; B-190). With the judge off, when its daily
/// cap is reached or when Claude Code cannot be run, captures wait, and
/// waiting alone never deletes one (B-178). A capture the judge gave up on,
/// after three bad answers, is deleted `pending_max_days` after the last
/// one; a capture from before schema version 4, which the judge never
/// takes, is deleted `pending_max_days` after the upgrade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JudgeSettings {
    pub enabled: bool,
    /// The `--model` given to Claude Code: an alias such as `haiku` or a
    /// full model name.
    pub model: String,
    /// Judge calls a UTC day, 1 and up. Each call counts, failed or not,
    /// the duplicate check's too (B-192).
    pub max_calls_per_day: u32,
    /// US dollars a UTC day, more than 0: once the day's calls have cost
    /// this much, by Claude Code's own estimate at API prices, no batch
    /// starts until the next UTC day (B-193). A batch started under it is
    /// finished, so a day can end over it by one batch's cost.
    pub max_usd_per_day: f64,
    /// Days before a capture the judge gave up on, or one from before schema
    /// version 4, is deleted, 1 to 365 (B-178).
    pub pending_max_days: u32,
    /// The Claude Code executable. When absent, `claude.exe` (or `claude`)
    /// is looked for on `PATH`, then in `~/.local/bin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

impl Default for JudgeSettings {
    fn default() -> Self {
        JudgeSettings {
            enabled: true,
            model: DEFAULT_JUDGE_MODEL.to_owned(),
            max_calls_per_day: DEFAULT_JUDGE_MAX_CALLS_PER_DAY,
            max_usd_per_day: DEFAULT_JUDGE_MAX_USD_PER_DAY,
            pending_max_days: DEFAULT_JUDGE_PENDING_MAX_DAYS,
            command: None,
        }
    }
}

impl Default for CaptureSettings {
    fn default() -> Self {
        CaptureSettings {
            enabled: true,
            tools: DEFAULT_CAPTURE_TOOLS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            ignore: DEFAULT_CAPTURE_IGNORE
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            max_index_tokens: 8192,
            tool_results: true,
            judge: JudgeSettings::default(),
        }
    }
}

impl CaptureSettings {
    /// Whether `tool` is captured.
    pub fn captures_tool(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }

    /// Whether a PostToolUse capture of `tool` is allowed by
    /// [`CaptureSettings::tool_results`]: the tools that write a file
    /// always are; any other tool's capture is text read from a file or
    /// command output (B-221).
    pub fn allows_tool_output(&self, tool: &str) -> bool {
        self.tool_results || FILE_WRITE_TOOLS.contains(&tool)
    }

    /// Whether a path is excluded from capture. `path` may be relative to
    /// the project or absolute, with either separator.
    pub fn is_ignored(&self, path: &str) -> bool {
        self.is_ignored_for(path, cfg!(windows))
    }

    fn is_ignored_for(&self, path: &str, windows: bool) -> bool {
        let path = path.replace('\\', "/");
        let path = path.trim_start_matches("./");
        self.ignore.iter().any(|pattern| {
            let pattern = pattern.replace('\\', "/");
            if pattern.contains('/') {
                let pattern = pattern.trim_start_matches('/');
                glob_match(pattern, path, windows) || {
                    // An absolute path still matches a relative pattern at
                    // any depth: `config/*.json` matches `/x/config/a.json`.
                    let suffixed = format!("**/{pattern}");
                    glob_match(&suffixed, path, windows)
                }
            } else {
                path.split('/')
                    .filter(|c| !c.is_empty())
                    .any(|c| glob_match(&pattern, c, windows))
            }
        })
    }
}

/// Glob matching for capture ignore patterns: `*` matches any run of
/// characters except `/`, `?` one character except `/`, and `**` any run
/// including `/` (and `**/` also matches nothing). Case-insensitive when
/// `fold_case` is set.
pub fn glob_match(pattern: &str, text: &str, fold_case: bool) -> bool {
    let fold = |s: &str| {
        if fold_case {
            s.to_lowercase()
        } else {
            s.to_owned()
        }
    };
    let p: Vec<char> = fold(pattern).chars().collect();
    let t: Vec<char> = fold(text).chars().collect();
    glob_rec(&p, &t)
}

fn glob_rec(p: &[char], t: &[char]) -> bool {
    match p.first() {
        None => t.is_empty(),
        Some('*') if p.get(1) == Some(&'*') => {
            let mut rest = &p[2..];
            // `**/` may match nothing, including the slash.
            if rest.first() == Some(&'/') && glob_rec(&rest[1..], t) {
                return true;
            }
            while rest.first() == Some(&'*') {
                rest = &rest[1..];
            }
            (0..=t.len()).any(|i| glob_rec(rest, &t[i..]))
        }
        Some('*') => {
            let rest = &p[1..];
            let limit = t.iter().position(|c| *c == '/').unwrap_or(t.len());
            (0..=limit).any(|i| glob_rec(rest, &t[i..]))
        }
        Some('?') => t.first().is_some_and(|c| *c != '/') && glob_rec(&p[1..], &t[1..]),
        Some(c) => t.first() == Some(c) && glob_rec(&p[1..], &t[1..]),
    }
}

/// `[recall]` (SPEC section 6, Recall pipeline and Similarity floor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecallSettings {
    /// Cosine similarity under which explicit recall marks a result weak and
    /// the hook drops it. Model-specific and provisional; its journal metric
    /// is `hook_top_score`.
    pub similarity_floor: f64,
    /// Default number of results, 1 to 50.
    pub k: u32,
}

impl Default for RecallSettings {
    fn default() -> Self {
        RecallSettings {
            similarity_floor: 0.70,
            k: 5,
        }
    }
}

/// `[ambient]`: the prompt hook (SPEC section 6, Ambient recall). The
/// environment switches of [`resolve_ambient`] override these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AmbientSettings {
    pub enabled: bool,
    /// Result cap, 0 to 50; 0 disables the hook.
    pub top_k: u32,
    pub scope: Scope,
}

impl Default for AmbientSettings {
    fn default() -> Self {
        AmbientSettings {
            enabled: true,
            top_k: DEFAULT_AMBIENT_TOP_K,
            scope: Scope::Project,
        }
    }
}

/// `[journal]` (SPEC section 6, Metrics journal).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JournalSettings {
    /// Write prompt text into hook journal lines. Off by default.
    pub include_prompts: bool,
}

/// `[daemon]` (SPEC section 15, Daemon when idle).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonSettings {
    /// Minutes without a request before the daemon exits; 0 means never.
    pub idle_exit_minutes: u32,
}

/// Default `output.token_budget` (SPEC section 8, Tools).
pub const DEFAULT_OUTPUT_TOKEN_BUDGET: u32 = 2000;
/// Smallest `output.token_budget` and `limit` accepted: room for one whole
/// result and its provenance. A memory view alone is about 85 to 90 tokens,
/// and a reply's fixed fields add up to about 30 (B-111).
pub const MIN_OUTPUT_TOKEN_BUDGET: u32 = 200;

/// `[output]` (SPEC section 8: "No tool returns more than a configurable
/// token budget (default 2,000) without an explicit `limit`").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputSettings {
    /// Estimated tokens (characters / 4) a tool result may hold before it is
    /// cut and marked `truncated`.
    pub token_budget: u32,
}

impl Default for OutputSettings {
    fn default() -> Self {
        OutputSettings {
            token_budget: DEFAULT_OUTPUT_TOKEN_BUDGET,
        }
    }
}

/// Every setting, with SPEC defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub chunk: ChunkSettings,
    pub embed: EmbedSettings,
    pub cpu: CpuSettings,
    pub capture: CaptureSettings,
    pub recall: RecallSettings,
    pub ambient: AmbientSettings,
    pub journal: JournalSettings,
    pub daemon: DaemonSettings,
    pub output: OutputSettings,
}

fn range_error(key: &str, value: impl std::fmt::Display, range: &str) -> CoreError {
    CoreError::config_invalid(format!("{key} = {value} is outside {range}"))
}

impl Settings {
    /// Checks every range in SPEC sections 5, 6 and 16.
    pub fn validate(&self) -> Result<()> {
        let c = &self.chunk;
        if !(128..=1024).contains(&c.max_tokens) {
            return Err(range_error("chunk.max_tokens", c.max_tokens, "128 to 1024"));
        }
        if !(c.overlap.is_finite() && (0.0..=0.25).contains(&c.overlap)) {
            return Err(range_error("chunk.overlap", c.overlap, "0 to 0.25"));
        }
        let e = &self.embed;
        if !(1..=16).contains(&e.batch_size) {
            return Err(range_error("embed.batch_size", e.batch_size, "1 to 16"));
        }
        if e.query_threads == Some(0) {
            return Err(range_error(
                "embed.query_threads",
                0,
                "1 to the physical core count",
            ));
        }
        let b = self.cpu.budget;
        if !(b.is_finite() && (0.1..=1.0).contains(&b)) {
            return Err(range_error("cpu.budget", b, "0.1 to 1.0"));
        }
        let cap = &self.capture;
        if cap.max_index_tokens < 256 {
            return Err(range_error(
                "capture.max_index_tokens",
                cap.max_index_tokens,
                "256 and up",
            ));
        }
        for tool in &cap.tools {
            if tool.is_empty() || tool.chars().any(char::is_whitespace) {
                return Err(CoreError::config_invalid(format!(
                    "capture.tools has an invalid name {tool:?}"
                )));
            }
        }
        for pattern in &cap.ignore {
            if python_strip(pattern).is_empty() {
                return Err(CoreError::config_invalid(
                    "capture.ignore has an empty pattern",
                ));
            }
        }
        let j = &cap.judge;
        if j.model.is_empty() || j.model.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(CoreError::config_invalid(format!(
                "capture.judge.model has an invalid name {:?}",
                j.model
            )));
        }
        if j.max_calls_per_day == 0 {
            return Err(range_error(
                "capture.judge.max_calls_per_day",
                0,
                "1 and up",
            ));
        }
        if !(j.max_usd_per_day.is_finite() && j.max_usd_per_day > 0.0) {
            return Err(range_error(
                "capture.judge.max_usd_per_day",
                j.max_usd_per_day,
                "more than 0",
            ));
        }
        if !(1..=365).contains(&j.pending_max_days) {
            return Err(range_error(
                "capture.judge.pending_max_days",
                j.pending_max_days,
                "1 to 365",
            ));
        }
        if j.command.as_deref().is_some_and(|c| c.trim().is_empty()) {
            return Err(CoreError::config_invalid("capture.judge.command is empty"));
        }
        let f = self.recall.similarity_floor;
        if !(f.is_finite() && (0.0..=1.0).contains(&f)) {
            return Err(range_error("recall.similarity_floor", f, "0 to 1"));
        }
        if !(1..=MAX_K).contains(&self.recall.k) {
            return Err(range_error("recall.k", self.recall.k, "1 to 50"));
        }
        if self.ambient.top_k > MAX_K {
            return Err(range_error("ambient.top_k", self.ambient.top_k, "0 to 50"));
        }
        if self.output.token_budget < MIN_OUTPUT_TOKEN_BUDGET {
            return Err(range_error(
                "output.token_budget",
                self.output.token_budget,
                "200 and up",
            ));
        }
        Ok(())
    }

    /// Threads for the query session: `embed.query_threads` (4 by default),
    /// never more than the physical cores (SPEC section 5, rules 1 and 2).
    pub fn query_threads(&self, physical_cores: usize) -> usize {
        let wanted = self.embed.query_threads.unwrap_or(4) as usize;
        wanted.min(physical_cores.max(1)).max(1)
    }

    /// Threads for the bulk session: `cpu.budget` × physical cores, rounded
    /// down, at least 1 (SPEC section 5, rule 3).
    pub fn bulk_threads(&self, physical_cores: usize) -> usize {
        let physical = physical_cores.max(1);
        let n = (self.cpu.budget * physical as f64).floor() as usize;
        n.clamp(1, physical)
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// The file name of both configuration files.
pub const CONFIG_FILE: &str = "config.toml";

/// `[[profile_rule]]` in the root `config.toml`: working directories under
/// `path` use `profile` (SPEC section 9, Profiles).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileRule {
    pub path: String,
    pub profile: String,
}

/// The root configuration file, split into its settings table and its
/// profile rules.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RootConfig {
    pub settings: toml::Table,
    pub profile_rules: Vec<ProfileRule>,
}

fn toml_error(what: &str, e: toml::de::Error) -> CoreError {
    CoreError::config_invalid(format!("{what}: {}", e.message().trim()))
}

/// Parses the root `config.toml`.
pub fn parse_root_config(text: &str) -> Result<RootConfig> {
    let mut table: toml::Table = toml::from_str(text).map_err(|e| toml_error("config.toml", e))?;
    let rules = match table.remove("profile_rule") {
        None => Vec::new(),
        Some(v) => {
            let rules: Vec<ProfileRule> = v
                .try_into()
                .map_err(|e| toml_error("config.toml [[profile_rule]]", e))?;
            for r in &rules {
                ProfileName::parse(&r.profile)?;
                if r.path.trim().is_empty() {
                    return Err(CoreError::config_invalid(
                        "a [[profile_rule]] has an empty path",
                    ));
                }
            }
            rules
        }
    };
    Ok(RootConfig {
        settings: table,
        profile_rules: rules,
    })
}

/// Parses a profile's `config.toml`, which may not hold profile rules.
pub fn parse_profile_config(text: &str) -> Result<toml::Table> {
    let table: toml::Table =
        toml::from_str(text).map_err(|e| toml_error("profile config.toml", e))?;
    if table.contains_key("profile_rule") {
        return Err(CoreError::config_invalid(
            "[[profile_rule]] belongs in the root config.toml, not in a profile's",
        ));
    }
    Ok(table)
}

/// Merges `over` into `base`: tables merge key by key, anything else is
/// replaced.
pub fn merge_tables(base: &mut toml::Table, over: toml::Table) {
    for (key, value) in over {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge_tables(b, o),
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

/// Builds validated settings from the root table and the profile's table.
pub fn settings_from_tables(root: toml::Table, profile: Option<toml::Table>) -> Result<Settings> {
    let mut merged = root;
    if let Some(p) = profile {
        merge_tables(&mut merged, p);
    }
    let settings: Settings = toml::Value::Table(merged)
        .try_into()
        .map_err(|e| toml_error("config.toml", e))?;
    settings.validate()?;
    Ok(settings)
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CoreError::config_invalid(format!(
            "cannot read {}: {e}",
            path.display()
        ))),
    }
}

/// Reads `<root>/config.toml`; a missing file is an empty configuration.
pub fn load_root_config(root_dir: &Path) -> Result<RootConfig> {
    match read_optional(&root_dir.join(CONFIG_FILE))? {
        Some(text) => parse_root_config(&text),
        None => Ok(RootConfig::default()),
    }
}

/// Reads both files for `profile` and returns validated settings.
pub fn load_settings(
    root_dir: &Path,
    root: &RootConfig,
    profile: &ProfileName,
) -> Result<Settings> {
    let profile_table = match read_optional(&root_dir.join(profile.as_str()).join(CONFIG_FILE))? {
        Some(text) => Some(parse_profile_config(&text)?),
        None => None,
    };
    settings_from_tables(root.settings.clone(), profile_table)
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// The profile used when nothing selects another.
pub const DEFAULT_PROFILE: &str = "default";
/// `--profile` takes precedence over these variables, in this order.
pub const PLUGIN_PROFILE_ENV: &str = "CLAUDE_PLUGIN_OPTION_PROFILE";
pub const PROFILE_ENV: &str = "SPATIAL_MEMORY_PROFILE";

/// A profile name: 1 to 32 characters of `a-z`, `0-9` and `-`, not starting
/// with `-` (decision B-3; this also keeps `_models` free).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProfileName(String);

impl ProfileName {
    pub fn parse(s: &str) -> Result<Self> {
        let ok = !s.is_empty()
            && s.len() <= 32
            && s.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            && !s.starts_with('-');
        if ok {
            Ok(ProfileName(s.to_owned()))
        } else {
            Err(CoreError::config_invalid(format!(
                "invalid profile name {s:?}: use 1 to 32 characters of a-z, 0-9 and '-', not starting with '-'"
            )))
        }
    }

    pub fn default_profile() -> Self {
        ProfileName(DEFAULT_PROFILE.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ProfileName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ProfileName {
    type Error = CoreError;
    fn try_from(s: String) -> Result<Self> {
        ProfileName::parse(&s)
    }
}

impl From<ProfileName> for String {
    fn from(p: ProfileName) -> String {
        p.0
    }
}

/// What selected the profile, for `status` and `doctor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileSource {
    Flag,
    PluginOption,
    Environment,
    DirectoryRule,
    Default,
}

/// Picks the profile: `--profile`, then the plugin option, then
/// `SPATIAL_MEMORY_PROFILE`, then the longest `[[profile_rule]]` whose path
/// contains `cwd`, then `default` (SPEC section 9, Profiles; decision B-16).
/// Empty values count as unset. A value that is not a valid name is an
/// error, not a silent fallback, so memories never land in the wrong store.
pub fn resolve_profile(
    flag: Option<&str>,
    get_env: impl Fn(&str) -> Option<String>,
    cwd: Option<&Path>,
    rules: &[ProfileRule],
) -> Result<(ProfileName, ProfileSource)> {
    resolve_profile_for(flag, get_env, cwd, rules, cfg!(windows))
}

fn resolve_profile_for(
    flag: Option<&str>,
    get_env: impl Fn(&str) -> Option<String>,
    cwd: Option<&Path>,
    rules: &[ProfileRule],
    windows: bool,
) -> Result<(ProfileName, ProfileSource)> {
    let present = |v: Option<String>| {
        v.map(|s| python_strip(&s).to_owned())
            .filter(|s| !s.is_empty())
    };
    if let Some(f) = flag.map(python_strip).filter(|s| !s.is_empty()) {
        return Ok((ProfileName::parse(f)?, ProfileSource::Flag));
    }
    if let Some(p) = present(get_env(PLUGIN_PROFILE_ENV)) {
        return Ok((ProfileName::parse(&p)?, ProfileSource::PluginOption));
    }
    if let Some(p) = present(get_env(PROFILE_ENV)) {
        return Ok((ProfileName::parse(&p)?, ProfileSource::Environment));
    }
    if let Some(cwd) = cwd {
        let cwd_key = path_key(&cwd.to_string_lossy(), windows);
        let best = rules
            .iter()
            .filter_map(|r| {
                let key = path_key(&r.path, windows);
                let inside = cwd_key == key
                    || (key.ends_with('/') && cwd_key.starts_with(&key))
                    || cwd_key
                        .strip_prefix(&key)
                        .is_some_and(|rest| rest.starts_with('/'));
                inside.then_some((key.len(), r))
            })
            .max_by_key(|(len, _)| *len);
        if let Some((_, rule)) = best {
            return Ok((
                ProfileName::parse(&rule.profile)?,
                ProfileSource::DirectoryRule,
            ));
        }
    }
    Ok((ProfileName::default_profile(), ProfileSource::Default))
}

// ---------------------------------------------------------------------------
// Ambient recall switches (SPEC section 6, Hook input and output)
// ---------------------------------------------------------------------------

/// The plugin's `userConfig` switch (SPEC section 16, Plugin options).
pub const PLUGIN_AMBIENT_ENV: &str = "CLAUDE_PLUGIN_OPTION_AMBIENT_RECALL";
/// Version 2's switches, unchanged names.
pub const AMBIENT_ENV: &str = "SPATIAL_MEMORY_AMBIENT_RECALL";
pub const AMBIENT_TOP_K_ENV: &str = "SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K";
pub const AMBIENT_SCOPE_ENV: &str = "SPATIAL_MEMORY_AMBIENT_RECALL_SCOPE";

/// Set, to `1`, in the environment of every capture-judge call (B-158). A
/// hook that sees it, with any non-empty value, exits 0 before it reads
/// stdin, prints, journals or contacts the daemon, so a judge call can never
/// recall, capture or start a daemon even if its hooks were loaded.
pub const JUDGE_MARKER_ENV: &str = "SPATIAL_MEMORY_JUDGE";

/// The hook's effective switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmbientSwitches {
    pub enabled: bool,
    pub top_k: u32,
    pub scope: Scope,
}

/// Python's `int(s)` for the forms version 2 could meet: surrounding
/// whitespace, an optional sign, ASCII digits with single underscores
/// between them. `None` where Python would raise `ValueError`.
pub fn python_int(s: &str) -> Option<i64> {
    let s = python_strip(s);
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let mut value: i64 = 0;
    for b in digits.bytes() {
        match b {
            b'_' => {}
            b'0'..=b'9' => value = value.saturating_mul(10).saturating_add(i64::from(b - b'0')),
            _ => return None,
        }
    }
    Some(if neg { -value } else { value })
}

fn plugin_bool(v: &str) -> Option<bool> {
    match python_strip(v).to_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Some(true),
        "false" | "0" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// The hook's switches: the plugin option first, then version 2's
/// variables, then `[ambient]`. Version 2's rules are kept exactly:
/// `SPATIAL_MEMORY_AMBIENT_RECALL` enables only when its stripped,
/// lower-cased value is `on`; `..._TOP_K` that does not parse means 5, and
/// zero or less disables the hook. The cap never exceeds [`MAX_K`].
pub fn resolve_ambient(
    settings: &AmbientSettings,
    get_env: impl Fn(&str) -> Option<String>,
) -> AmbientSwitches {
    let mut enabled = settings.enabled;
    let plugin = get_env(PLUGIN_AMBIENT_ENV).and_then(|v| plugin_bool(&v));
    if let Some(p) = plugin {
        enabled = p;
    } else if let Some(v) = get_env(AMBIENT_ENV) {
        enabled = python_strip(&v).to_lowercase() == "on";
    }
    let mut top_k = settings.top_k;
    if let Some(v) = get_env(AMBIENT_TOP_K_ENV) {
        match python_int(&v) {
            Some(n) if n <= 0 => {
                top_k = 0;
            }
            Some(n) => top_k = u32::try_from(n).unwrap_or(MAX_K).min(MAX_K),
            None => top_k = DEFAULT_AMBIENT_TOP_K,
        }
    }
    if top_k == 0 {
        enabled = false;
    }
    let scope = get_env(AMBIENT_SCOPE_ENV)
        .and_then(|v| python_strip(&v).to_lowercase().parse::<Scope>().ok())
        .unwrap_or(settings.scope);
    AmbientSwitches {
        enabled,
        top_k,
        scope,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::error::ErrorCode;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn s5_defaults_match_the_spec() {
        let s = Settings::default();
        s.validate().unwrap();
        assert_eq!(s.chunk.max_tokens, 256);
        assert_eq!(s.chunk.overlap, 0.125);
        assert_eq!(s.embed.model, EmbedModel::EmbeddingGemma2);
        assert_eq!(s.embed.model.as_str(), "embeddinggemma-2");
        assert_eq!(s.embed.batch_size, 4);
        assert!(!s.embed.spinning);
        assert_eq!(s.cpu.budget, 0.75);
        assert_eq!(s.capture.max_index_tokens, 8192);
        assert!(s.capture.enabled);
        assert!(s.capture.tools.is_empty(), "B-160: no tool by default");
        assert!(s.capture.tool_results, "B-221: on by default");
        assert_eq!(
            s.capture.judge,
            JudgeSettings {
                enabled: true,
                model: "haiku".into(),
                max_calls_per_day: 600,
                max_usd_per_day: 3.0,
                pending_max_days: 14,
                command: None,
            },
            "B-165, B-193"
        );
        assert_eq!(s.recall.similarity_floor, 0.70);
        assert_eq!(s.recall.k, 5);
        assert_eq!(
            s.ambient,
            AmbientSettings {
                enabled: true,
                top_k: 5,
                scope: Scope::Project
            }
        );
        assert!(!s.journal.include_prompts);
        assert_eq!(s.daemon.idle_exit_minutes, 0);
        assert_eq!(s.output.token_budget, 2000);
    }

    #[test]
    fn s5_thread_counts_follow_physical_cores() {
        let s = Settings::default();
        assert_eq!(s.query_threads(16), 4);
        assert_eq!(s.query_threads(2), 2);
        assert_eq!(s.query_threads(0), 1);
        assert_eq!(s.bulk_threads(16), 12);
        assert_eq!(s.bulk_threads(2), 1);
        assert_eq!(s.bulk_threads(1), 1);
        let mut full = Settings::default();
        full.cpu.budget = 1.0;
        full.embed.query_threads = Some(64);
        assert_eq!(full.bulk_threads(16), 16);
        assert_eq!(full.query_threads(16), 16);
    }

    #[test]
    fn s5_out_of_range_values_are_config_invalid() {
        let bad = [
            "[chunk]\nmax_tokens = 127",
            "[chunk]\nmax_tokens = 1025",
            "[chunk]\noverlap = 0.3",
            "[embed]\nbatch_size = 0",
            "[embed]\nbatch_size = 17",
            "[embed]\nquery_threads = 0",
            "[embed]\nmodel = \"bge\"",
            "[cpu]\nbudget = 0.05",
            "[cpu]\nbudget = 1.5",
            "[capture]\nmax_index_tokens = 255",
            "[capture]\ntools = [\"\"]",
            "[capture.judge]\nmodel = \"\"",
            "[capture.judge]\nmodel = \"claude haiku\"",
            "[capture.judge]\nmax_calls_per_day = 0",
            "[capture.judge]\nmax_usd_per_day = 0",
            "[capture.judge]\nmax_usd_per_day = -1.5",
            "[capture.judge]\nmax_usd_per_day = nan",
            "[capture.judge]\nmax_usd_per_day = inf",
            "[capture.judge]\npending_max_days = 0",
            "[capture.judge]\npending_max_days = 366",
            "[capture.judge]\ncommand = \" \"",
            "[capture.judge]\nenable = false",
            "[recall]\nsimilarity_floor = 1.5",
            "[recall]\nk = 0",
            "[ambient]\ntop_k = 51",
            "[ambient]\nscope = \"everywhere\"",
            "[output]\ntoken_budget = 199",
            "[chunk]\nmax_token = 256",
            "[unknown]\nx = 1",
        ];
        for text in bad {
            let root = parse_root_config(text).map(|r| r.settings);
            let e = root
                .and_then(|t| settings_from_tables(t, None))
                .unwrap_err();
            assert_eq!(e.code, ErrorCode::ConfigInvalid, "{text}");
        }
        let ok = "[chunk]\nmax_tokens = 1024\noverlap = 0\n[embed]\nmodel = \"arctic-embed-m\"\n";
        let s = settings_from_tables(parse_root_config(ok).unwrap().settings, None).unwrap();
        assert_eq!(s.embed.model, EmbedModel::ArcticEmbedM);
    }

    #[test]
    fn s9_profile_config_overrides_root_key_by_key() {
        let root = parse_root_config(
            "[chunk]\nmax_tokens = 512\n[capture]\nenabled = false\n\n[[profile_rule]]\npath = \"/work\"\nprofile = \"work\"\n",
        )
        .unwrap();
        assert_eq!(root.profile_rules.len(), 1);
        let profile =
            parse_profile_config("[chunk]\noverlap = 0.25\n[capture]\nenabled = true\n").unwrap();
        let s = settings_from_tables(root.settings, Some(profile)).unwrap();
        assert_eq!(s.chunk.max_tokens, 512);
        assert_eq!(s.chunk.overlap, 0.25);
        assert!(s.capture.enabled);
        let e =
            parse_profile_config("[[profile_rule]]\npath = \"/x\"\nprofile = \"y\"\n").unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigInvalid);
        let e = parse_root_config("[[profile_rule]]\npath = \"/x\"\nprofile = \"Bad Name\"\n")
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigInvalid);
    }

    #[test]
    fn s9_settings_load_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let profile = ProfileName::parse("work").unwrap();
        std::fs::create_dir_all(dir.path().join("work")).unwrap();
        std::fs::write(dir.path().join(CONFIG_FILE), "[recall]\nk = 7\n").unwrap();
        std::fs::write(
            dir.path().join("work").join(CONFIG_FILE),
            "[recall]\nk = 9\n",
        )
        .unwrap();
        let root = load_root_config(dir.path()).unwrap();
        assert_eq!(
            load_settings(dir.path(), &root, &profile).unwrap().recall.k,
            9
        );
        let other = ProfileName::default_profile();
        assert_eq!(
            load_settings(dir.path(), &root, &other).unwrap().recall.k,
            7
        );
        let empty = tempfile::tempdir().unwrap();
        let root = load_root_config(empty.path()).unwrap();
        assert_eq!(
            load_settings(empty.path(), &root, &other).unwrap(),
            Settings::default()
        );
    }

    #[test]
    fn at12_profile_order_is_flag_plugin_env_rule_default() {
        let rules = vec![
            ProfileRule {
                path: "C:\\Work".into(),
                profile: "work".into(),
            },
            ProfileRule {
                path: "c:/work/client-a".into(),
                profile: "client-a".into(),
            },
        ];
        let cwd = Path::new("C:/WORK/Client-A/api");
        let all = env(&[(PLUGIN_PROFILE_ENV, "plugin"), (PROFILE_ENV, "envp")]);
        let r = |flag, get: &dyn Fn(&str) -> Option<String>, cwd| {
            resolve_profile_for(flag, get, cwd, &rules, true).unwrap()
        };
        assert_eq!(r(Some("flag"), &all, Some(cwd)).1, ProfileSource::Flag);
        assert_eq!(
            r(None, &all, Some(cwd)),
            (
                ProfileName::parse("plugin").unwrap(),
                ProfileSource::PluginOption
            )
        );
        let only_env = env(&[(PLUGIN_PROFILE_ENV, "  "), (PROFILE_ENV, "envp")]);
        assert_eq!(r(None, &only_env, Some(cwd)).1, ProfileSource::Environment);
        let none = env(&[]);
        assert_eq!(
            r(None, &none, Some(cwd)),
            (
                ProfileName::parse("client-a").unwrap(),
                ProfileSource::DirectoryRule
            )
        );
        assert_eq!(
            r(None, &none, Some(Path::new("c:\\work\\other")))
                .0
                .as_str(),
            "work"
        );
        assert_eq!(
            r(None, &none, Some(Path::new("c:/workshop"))).1,
            ProfileSource::Default
        );
        assert_eq!(r(None, &none, None).0.as_str(), "default");
        let bad = env(&[(PROFILE_ENV, "Bad Name")]);
        assert_eq!(
            resolve_profile_for(None, &bad, None, &rules, true)
                .unwrap_err()
                .code,
            ErrorCode::ConfigInvalid
        );
        // On Linux and macOS case matters.
        let unix_rules = vec![ProfileRule {
            path: "/home/me/work/".into(),
            profile: "work".into(),
        }];
        let p = resolve_profile_for(
            None,
            &none,
            Some(Path::new("/home/me/work/x")),
            &unix_rules,
            false,
        )
        .unwrap();
        assert_eq!(p.1, ProfileSource::DirectoryRule);
        let p = resolve_profile_for(
            None,
            &none,
            Some(Path::new("/home/me/Work/x")),
            &unix_rules,
            false,
        )
        .unwrap();
        assert_eq!(p.1, ProfileSource::Default);
    }

    #[test]
    fn s9_profile_names_are_restricted() {
        for ok in [
            "default",
            "work",
            "a",
            "client-a-2",
            "x234567890123456789012345678901z",
        ] {
            ProfileName::parse(ok).unwrap();
        }
        for bad in [
            "",
            "-x",
            "Work",
            "_models",
            "a b",
            "a/b",
            "x2345678901234567890123456789012z",
            "é",
        ] {
            assert!(ProfileName::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn at17_version_2_ambient_switches_keep_their_rules() {
        let d = AmbientSettings::default();
        let on = resolve_ambient(&d, env(&[]));
        assert_eq!(
            on,
            AmbientSwitches {
                enabled: true,
                top_k: 5,
                scope: Scope::Project
            }
        );
        for v in ["on", " ON ", "On\n"] {
            assert!(
                resolve_ambient(&d, env(&[(AMBIENT_ENV, v)])).enabled,
                "{v:?}"
            );
        }
        for v in ["off", "", "1", "true", "yes", "enabled"] {
            assert!(
                !resolve_ambient(&d, env(&[(AMBIENT_ENV, v)])).enabled,
                "{v:?}"
            );
        }
        let k = |v: &str| resolve_ambient(&d, env(&[(AMBIENT_TOP_K_ENV, v)]));
        assert_eq!(k("3").top_k, 3);
        assert_eq!(k(" +8 ").top_k, 8);
        assert_eq!(k("1_0").top_k, 10);
        assert_eq!(k("abc").top_k, 5);
        assert_eq!(k("3.5").top_k, 5);
        assert_eq!(k("").top_k, 5);
        assert!(k("abc").enabled);
        assert!(!k("0").enabled);
        assert!(!k("-2").enabled);
        assert_eq!(k("1000").top_k, MAX_K);
        let s = resolve_ambient(&d, env(&[(AMBIENT_SCOPE_ENV, " ALL ")]));
        assert_eq!(s.scope, Scope::All);
        let s = resolve_ambient(&d, env(&[(AMBIENT_SCOPE_ENV, "namespace")]));
        assert_eq!(s.scope, Scope::Project);
    }

    #[test]
    fn s16_plugin_option_is_read_before_version_2_variables() {
        let d = AmbientSettings::default();
        let s = resolve_ambient(
            &d,
            env(&[(PLUGIN_AMBIENT_ENV, "false"), (AMBIENT_ENV, "on")]),
        );
        assert!(!s.enabled);
        let s = resolve_ambient(
            &d,
            env(&[(PLUGIN_AMBIENT_ENV, "true"), (AMBIENT_ENV, "off")]),
        );
        assert!(s.enabled);
        // An unrecognised plugin value falls through to version 2's switch.
        let s = resolve_ambient(
            &d,
            env(&[(PLUGIN_AMBIENT_ENV, "maybe"), (AMBIENT_ENV, "off")]),
        );
        assert!(!s.enabled);
        let off = AmbientSettings {
            enabled: false,
            ..d
        };
        assert!(!resolve_ambient(&off, env(&[])).enabled);
    }

    #[test]
    fn s16_capture_ignore_patterns_cover_env_and_key_files() {
        let c = CaptureSettings::default();
        let ignored = [
            ".env",
            "app/.env",
            "C:\\repo\\.env.local",
            "certs/server.pem",
            "/home/me/.ssh/id_rsa",
            "/home/me/.ssh/id_ed25519.pub",
            "keys/prod.key",
            "store.p12",
        ];
        for p in ignored {
            assert!(c.is_ignored_for(p, false), "{p}");
        }
        for p in [
            "src/env.rs",
            "docs/keys.md",
            ".envrc.md",
            "README.md",
            "environment/x.rs",
        ] {
            assert!(!c.is_ignored_for(p, false), "{p}");
        }
        assert!(c.is_ignored_for("certs/SERVER.PEM", true));
        assert!(!c.is_ignored_for("certs/SERVER.PEM", false));
        let custom = CaptureSettings {
            ignore: vec!["secrets/**".into(), "config/*.json".into(), "build".into()],
            ..CaptureSettings::default()
        };
        assert!(custom.is_ignored_for("secrets/a/b.txt", false));
        assert!(custom.is_ignored_for("config/app.json", false));
        assert!(custom.is_ignored_for("/abs/repo/config/app.json", false));
        assert!(!custom.is_ignored_for("config/sub/app.json", false));
        assert!(custom.is_ignored_for("x/build/out.txt", false));
        // Ignore patterns leave the tool list alone, which names no tool by
        // default (B-160).
        assert!(!custom.captures_tool("Edit"));
        let named = CaptureSettings {
            tools: vec!["Edit".into()],
            ..custom
        };
        assert!(named.captures_tool("Edit"));
        assert!(!named.captures_tool("Read"));
        assert!(named.is_ignored_for("secrets/a/b.txt", false));
    }

    #[test]
    fn s16_judge_settings_read_and_layer() {
        let root = parse_root_config(concat!(
            "[capture]\ntools = [\"Edit\"]\n",
            "[capture.judge]\nmodel = \"sonnet\"\nmax_calls_per_day = 50\n",
            "max_usd_per_day = 0.5\n",
        ))
        .unwrap();
        let profile =
            parse_profile_config("[capture.judge]\nenabled = false\npending_max_days = 3\n")
                .unwrap();
        let s = settings_from_tables(root.settings, Some(profile)).unwrap();
        assert_eq!(s.capture.tools, vec!["Edit".to_owned()]);
        assert_eq!(
            s.capture.judge,
            JudgeSettings {
                enabled: false,
                model: "sonnet".into(),
                max_calls_per_day: 50,
                max_usd_per_day: 0.5,
                pending_max_days: 3,
                command: None,
            }
        );
        let c = settings_from_tables(
            parse_root_config("[capture.judge]\ncommand = \"C:/tools/claude.exe\"\n")
                .unwrap()
                .settings,
            None,
        )
        .unwrap();
        assert_eq!(
            c.capture.judge.command.as_deref(),
            Some("C:/tools/claude.exe")
        );
        assert!(c.capture.judge.enabled);
    }

    /// B-221: `capture.tool_results` is on by default, read from either
    /// file, the profile's value winning; off, it allows a PostToolUse
    /// capture only of the tools that write a file.
    #[test]
    fn s16_b221_tool_results_switch_reads_and_layers() {
        let d = CaptureSettings::default();
        assert!(d.tool_results, "on by default");
        for tool in ["Bash", "Read", "mcp__db__query", "Edit", "Write"] {
            assert!(d.allows_tool_output(tool), "{tool}");
        }
        let off = settings_from_tables(
            parse_root_config(
                "[capture]
tool_results = false
",
            )
            .unwrap()
            .settings,
            None,
        )
        .unwrap();
        assert!(!off.capture.tool_results);
        for tool in FILE_WRITE_TOOLS {
            assert!(off.capture.allows_tool_output(tool), "{tool}");
        }
        for tool in ["Bash", "Read", "Grep", "mcp__db__query"] {
            assert!(!off.capture.allows_tool_output(tool), "{tool}");
        }
        let back_on = settings_from_tables(
            parse_root_config(
                "[capture]
tool_results = false
",
            )
            .unwrap()
            .settings,
            Some(
                parse_profile_config(
                    "[capture]
tool_results = true
",
                )
                .unwrap(),
            ),
        )
        .unwrap();
        assert!(back_on.capture.tool_results);
        let wrong = parse_root_config(
            "[capture]
tool_results = \"no\"
",
        )
        .unwrap();
        let e = settings_from_tables(wrong.settings, None).unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigInvalid);
    }

    #[test]
    fn s16_glob_rules() {
        assert!(glob_match("*.pem", "a.pem", false));
        assert!(!glob_match("*.pem", "d/a.pem", false));
        assert!(glob_match("**/*.pem", "a.pem", false));
        assert!(glob_match("**/*.pem", "d/e/a.pem", false));
        assert!(glob_match("a/**/b", "a/b", false));
        assert!(glob_match("a/**/b", "a/x/y/b", false));
        assert!(glob_match("id_?sa", "id_rsa", false));
        assert!(!glob_match("?", "/", false));
        assert!(glob_match("ÄB*", "äbc", true));
    }

    #[test]
    fn s6_python_int_matches_python() {
        assert_eq!(python_int("5"), Some(5));
        assert_eq!(python_int(" -3 "), Some(-3));
        assert_eq!(python_int("+0"), Some(0));
        assert_eq!(python_int("1_000"), Some(1000));
        for bad in ["", " ", "1__0", "_1", "1_", "0x10", "1.0", "five", "--1"] {
            assert_eq!(python_int(bad), None, "{bad:?}");
        }
    }
}
