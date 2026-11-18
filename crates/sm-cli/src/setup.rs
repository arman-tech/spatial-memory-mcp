//! `spatial-memory setup` (SPEC section 16, Registration): the fallback for
//! people who do not use the plugin.
//!
//! For Claude Code it registers the MCP server by calling `claude mcp add` at
//! user scope (never by editing Claude Code's own files), merges the five
//! hooks into `~/.claude/settings.json` after a backup, and refuses while
//! the plugin is enabled so nothing is registered twice. For Cursor it
//! writes `~/.cursor/mcp.json` and `~/.cursor/hooks.json` with plain
//! command strings. `--remove` undoes either. Running it twice changes
//! nothing the second time.
//!
//! Every outside effect goes through [`SetupEnv`]: the home folder, the
//! `claude` runner and the model step, so tests use temporary folders and a
//! fake runner and never touch the real files.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use sm_core::config::{ProfileName, Settings};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::protocol::HookEvent;

use crate::model_step::ModelStep;

/// The MCP server name in every registration.
pub const SERVER_NAME: &str = "spatial-memory";
/// The plugin name; `enabledPlugins` keys are `spatial-memory@<marketplace>`.
pub const PLUGIN_NAME: &str = "spatial-memory";
/// The program name a handler must run to count as ours.
pub const PROGRAM_NAME: &str = "spatial-memory";

/// One Claude Code hook as `setup` and the plugin register it (SPEC section
/// 16, Registration; the plugin's `hooks/hooks.json` holds the same table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HookSpec {
    /// The Claude Code event name, the key under `hooks`.
    pub key: &'static str,
    pub event: HookEvent,
    /// Seconds.
    pub timeout: u64,
    /// Claude Code runs the handler without waiting (`"async": true`).
    pub is_async: bool,
    pub matcher: Option<&'static str>,
}

/// The five Claude Code hooks.
pub const CLAUDE_HOOKS: [HookSpec; 5] = [
    HookSpec {
        key: "SessionStart",
        event: HookEvent::SessionStart,
        timeout: 5,
        is_async: false,
        matcher: Some("startup|resume"),
    },
    HookSpec {
        key: "UserPromptSubmit",
        event: HookEvent::UserPromptSubmit,
        timeout: 5,
        is_async: false,
        matcher: None,
    },
    HookSpec {
        key: "PostToolUse",
        event: HookEvent::PostToolUse,
        timeout: 10,
        is_async: true,
        matcher: None,
    },
    HookSpec {
        key: "PreCompact",
        event: HookEvent::PreCompact,
        timeout: 15,
        is_async: false,
        matcher: None,
    },
    HookSpec {
        key: "Stop",
        event: HookEvent::Stop,
        timeout: 15,
        is_async: false,
        matcher: None,
    },
];

/// One Cursor hook: the event key, the event, the timeout in seconds and
/// the loop limit (version 2's Cursor registration, kept unchanged).
pub const CURSOR_HOOKS: [(&str, HookEvent, u64, Option<u64>); 3] = [
    ("postToolUse", HookEvent::PostToolUse, 10, None),
    ("preCompact", HookEvent::PreCompact, 15, None),
    ("stop", HookEvent::Stop, 15, Some(1)),
];

/// Which client `setup` registers with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SetupClient {
    ClaudeCode,
    Cursor,
}

/// What one `claude` run produced.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RunOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Runs the `claude` command line. The binary starts the program directly
/// (no shell); tests record the calls instead.
pub trait ClaudeRunner: Send + Sync {
    fn run(&self, args: &[String]) -> io::Result<RunOutput>;
}

/// The real runner: `claude` found on `PATH` (on Windows that includes the
/// npm `claude.cmd` shim), started with no shell and a closed stdin.
#[derive(Debug, Clone)]
pub struct ProcessRunner {
    pub program: Option<PathBuf>,
}

impl ClaudeRunner for ProcessRunner {
    fn run(&self, args: &[String]) -> io::Result<RunOutput> {
        let program = self
            .program
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "`claude` is not on PATH"))?;
        let out = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .output()?;
        Ok(RunOutput {
            success: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

/// Everything `setup` touches.
pub struct SetupEnv<'a> {
    /// The user's home folder: `.claude/settings.json`, `.claude.json` and
    /// `.cursor/` live here.
    pub home: PathBuf,
    /// The working folder, for the project and local settings files that
    /// may enable the plugin.
    pub cwd: PathBuf,
    /// This program's absolute path, written into every registration.
    pub exe: PathBuf,
    pub runner: &'a dyn ClaudeRunner,
    pub model: &'a dyn ModelStep,
    /// The profile's settings, for the model step.
    pub settings: Settings,
    /// A profile to pin in every registration: only an explicit `--profile`
    /// other than `default`, so directory rules keep working otherwise.
    pub profile_pin: Option<ProfileName>,
}

/// What `setup` did, line by line, and the model step's failure if any.
/// Everything else has been done when `model_error` is set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SetupReport {
    pub lines: Vec<String>,
    pub model_error: Option<CoreError>,
}

/// Runs `setup` (or `setup --remove`) for one client.
pub fn run(
    env: &SetupEnv<'_>,
    client: SetupClient,
    remove: bool,
) -> Result<SetupReport, CoreError> {
    let mut report = SetupReport::default();
    match (client, remove) {
        (SetupClient::ClaudeCode, false) => {
            let found = plugins_found(&env.home, &env.cwd, env.runner);
            if let Some(current) = found.iter().find(|f| f.is_current()) {
                return Err(plugin_refusal(&current.evidence));
            }
            report.lines.push(register_mcp(env)?);
            report.lines.push(install_claude_hooks(env)?);
            if let Some(old) = found.iter().find(|f| !f.is_current()) {
                report
                    .lines
                    .push(format!("note: {}", old_plugin_line(&old.evidence)));
            }
        }
        (SetupClient::ClaudeCode, true) => {
            report.lines.push(unregister_mcp(env)?);
            report.lines.push(remove_claude_hooks(env)?);
            return Ok(report);
        }
        (SetupClient::Cursor, false) => {
            report.lines.push(install_cursor_mcp(env)?);
            report.lines.push(install_cursor_hooks(env)?);
        }
        (SetupClient::Cursor, true) => {
            report.lines.push(remove_cursor_mcp(env)?);
            report.lines.push(remove_cursor_hooks(env)?);
            return Ok(report);
        }
    }
    let model = model_only(env.model, &env.settings);
    report.lines.extend(model.lines);
    report.model_error = model.model_error;
    Ok(report)
}

/// `setup --model-only` (B-153): downloads the model if a file is missing
/// or wrong and verifies every file, and does nothing else. It runs no
/// `claude` command and writes no settings file, so it is allowed while the
/// plugin is enabled: the plugin route needs the model too.
pub fn model_only(model: &dyn ModelStep, settings: &Settings) -> SetupReport {
    let mut report = SetupReport::default();
    match model.ensure(settings) {
        Ok(line) => report.lines.push(format!("model: {line}")),
        Err(e) => report.model_error = Some(e),
    }
    report
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/// `~/.claude/settings.json`.
pub fn claude_settings_path(home: &Path) -> PathBuf {
    home.join(".claude").join("settings.json")
}

/// `~/.claude.json`, where `claude mcp add --scope user` stores the server
/// (SPEC section 13, measured with Claude Code 2.1.325).
pub fn claude_json_path(home: &Path) -> PathBuf {
    home.join(".claude.json")
}

/// `~/.cursor/hooks.json`.
pub fn cursor_hooks_path(home: &Path) -> PathBuf {
    home.join(".cursor").join("hooks.json")
}

/// `~/.cursor/mcp.json`.
pub fn cursor_mcp_path(home: &Path) -> PathBuf {
    home.join(".cursor").join("mcp.json")
}

/// A settings file as read: its text and its top-level object.
type ReadFile = (String, Map<String, Value>);

/// A JSON file's text and object, `None` when the file does not exist. An
/// empty file counts as `{}`. Anything but an object is refused, and
/// nothing is written.
fn read_object(path: &Path) -> Result<Option<ReadFile>, CoreError> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CoreError::storage(format!(
                "cannot read {}: {e}",
                path.display()
            )));
        }
    };
    let body = text.trim_start_matches('\u{feff}');
    if body.trim().is_empty() {
        return Ok(Some((text, Map::new())));
    }
    match serde_json::from_str::<Value>(body) {
        Ok(Value::Object(map)) => Ok(Some((text, map))),
        Ok(_) => Err(not_an_object(path)),
        Err(e) => Err(CoreError::config_invalid(format!(
            "{} is not valid JSON: {e}",
            path.display()
        ))
        .with_next_action(format!(
            "Fix the JSON in {}, then run setup again; nothing was changed.",
            path.display()
        ))),
    }
}

fn not_an_object(path: &Path) -> CoreError {
    CoreError::config_invalid(format!("{} does not hold a JSON object", path.display()))
        .with_next_action(format!(
            "Fix {}, then run setup again; nothing was changed.",
            path.display()
        ))
}

/// Reads a JSON object for inspection only; a missing or unreadable file
/// counts as empty.
fn peek_object(path: &Path) -> Map<String, Value> {
    read_object(path)
        .ok()
        .flatten()
        .map(|(_, m)| m)
        .unwrap_or_default()
}

/// A backup path next to `path` that does not exist yet.
fn backup_path(path: &Path) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut n = 0u32;
    loop {
        let suffix = if n == 0 {
            String::new()
        } else {
            format!("-{n}")
        };
        let candidate = path.with_file_name(format!("{name}.spatial-memory-{millis}{suffix}.bak"));
        if !candidate.exists() {
            return candidate;
        }
        n += 1;
    }
}

/// Writes `map` to `path`: a backup of the existing file first, then the new
/// text to a temporary file in the same folder, renamed over the original.
/// Returns the backup path, `None` when the file did not exist.
fn write_object(
    path: &Path,
    existed: bool,
    map: Map<String, Value>,
) -> Result<Option<PathBuf>, CoreError> {
    let io_err = |what: &str, p: &Path, e: io::Error| {
        CoreError::storage(format!("cannot {what} {}: {e}", p.display()))
    };
    let backup = if existed {
        let b = backup_path(path);
        std::fs::copy(path, &b).map_err(|e| io_err("back up", path, e))?;
        Some(b)
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io_err("create", parent, e))?;
        }
        None
    };
    let mut text = serde_json::to_string_pretty(&Value::Object(map))
        .map_err(|e| CoreError::internal(format!("cannot serialise settings: {e}")))?;
    text.push('\n');
    let tmp = path.with_file_name(format!(
        "{}.spatial-memory.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    std::fs::write(&tmp, text.as_bytes()).map_err(|e| io_err("write", &tmp, e))?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io_err("replace", path, e));
    }
    Ok(backup)
}

/// Applies `change` to the object in `path` and writes it back when it
/// changed. Returns a line saying what happened.
fn edit_file(
    path: &Path,
    what: &str,
    change: impl FnOnce(&mut Map<String, Value>) -> Result<bool, CoreError>,
) -> Result<String, CoreError> {
    let (existed, mut map) = match read_object(path)? {
        Some((_, map)) => (true, map),
        None => (false, Map::new()),
    };
    let before = map.clone();
    let changed = change(&mut map)?;
    if !changed || map == before {
        return Ok(format!("{what}: unchanged in {}", path.display()));
    }
    if !existed && map.is_empty() {
        return Ok(format!("{what}: nothing to change"));
    }
    match write_object(path, existed, map)? {
        Some(backup) => Ok(format!(
            "{what}: updated {} (backup {})",
            path.display(),
            backup.display()
        )),
        None => Ok(format!("{what}: created {}", path.display())),
    }
}

// ---------------------------------------------------------------------------
// Recognising our handlers
// ---------------------------------------------------------------------------

/// Whether a program path names `spatial-memory` (any folder, `.exe` or
/// not, any case).
pub fn program_is_ours(program: &str) -> bool {
    let p = program.trim().trim_matches('"');
    let base = p.rsplit(['/', '\\']).next().unwrap_or(p);
    let lower = base.to_ascii_lowercase();
    let stem = lower.strip_suffix(".exe").unwrap_or(&lower);
    stem == PROGRAM_NAME
}

/// Splits a shell command line into words, honouring double quotes.
fn shell_words(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut any = false;
    for c in command.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                any = true;
            }
            c if c.is_whitespace() && !quoted => {
                if any {
                    words.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            c => {
                cur.push(c);
                any = true;
            }
        }
    }
    if any {
        words.push(cur);
    }
    words
}

/// Whether a hook handler runs `spatial-memory hook ...`: the exec form this
/// version writes, or the shell form version 2 wrote.
pub fn handler_is_ours(handler: &Value) -> bool {
    let Some(command) = handler.get("command").and_then(Value::as_str) else {
        return false;
    };
    match handler.get("args").and_then(Value::as_array) {
        Some(args) => {
            program_is_ours(command) && args.first().and_then(Value::as_str) == Some("hook")
        }
        None => {
            let words = shell_words(command);
            words.first().is_some_and(|w| program_is_ours(w))
                && words.get(1).map(String::as_str) == Some("hook")
        }
    }
}

// ---------------------------------------------------------------------------
// Claude Code: the plugin check
// ---------------------------------------------------------------------------

/// The first major version of the plugin that runs this binary. Earlier
/// versions (1.x, the Python server) share its name and marketplace.
pub const PLUGIN_MAJOR: u64 = 3;

/// An enabled spatial-memory plugin found for one folder.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FoundPlugin {
    /// The installed version for this folder, when the plugin list gives it.
    version: Option<String>,
    evidence: String,
}

impl FoundPlugin {
    /// Version 3 or later, or a version that could not be read: the plugin
    /// is then taken to be this one, so setup never registers twice.
    fn is_current(&self) -> bool {
        self.version
            .as_deref()
            .and_then(major_version)
            .is_none_or(|m| m >= PLUGIN_MAJOR)
    }
}

/// The leading number of a version such as `1.11.4`.
fn major_version(version: &str) -> Option<u64> {
    let digits: String = version
        .trim()
        .trim_start_matches(['v', 'V'])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn is_our_plugin(id: &str) -> bool {
    id.split_once('@')
        .is_some_and(|(name, _)| name == PLUGIN_NAME)
}

/// One install from `claude plugin list --json` that applies to `cwd`.
struct Install {
    id: String,
    version: Option<String>,
    scope_rank: u8,
    on: bool,
}

/// The plugin list's installs of ours that apply to `cwd`: user scope
/// everywhere, project and local scope only in their own folder (compared
/// as `path_key` does, so case and slash direction do not matter on
/// Windows). `None` when the list cannot be read.
fn applicable_installs(cwd: &Path, runner: &dyn ClaudeRunner) -> Option<Vec<Install>> {
    let args: Vec<String> = ["plugin", "list", "--json"].map(String::from).to_vec();
    let out = runner.run(&args).ok().filter(|o| o.success)?;
    let list: Value = serde_json::from_str(out.stdout.trim()).ok()?;
    let here = sm_core::project::project_path_key(cwd);
    let installs = list
        .as_array()?
        .iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?;
            if !is_our_plugin(id) {
                return None;
            }
            let scope = item.get("scope").and_then(Value::as_str).unwrap_or("user");
            let scope_rank = match scope {
                "local" => 0,
                "project" => 1,
                _ => 2,
            };
            if scope_rank < 2 {
                let path = item.get("projectPath").and_then(Value::as_str)?;
                if sm_core::project::path_key(path, cfg!(windows)) != here {
                    return None;
                }
            }
            let flag = |k: &str| item.get(k).and_then(Value::as_bool) == Some(true);
            Some(Install {
                id: id.to_owned(),
                version: item
                    .get("version")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                scope_rank,
                on: flag("enabled") || flag("projectEnabled"),
            })
        })
        .collect();
    Some(installs)
}

/// Every spatial-memory plugin enabled for `cwd`, with its version.
///
/// Claude Code records an enabled plugin in the `enabledPlugins` setting,
/// keyed `plugin@marketplace`; for one key the local settings win over the
/// project settings, which win over the user settings, so a key set to
/// false in `.claude/settings.local.json` is disabled whatever the other
/// files say (B-150). `claude plugin list --json` (read-only) gives the
/// installed version for this folder, and, for a key no settings file
/// names, whether its default leaves it on. Installs for other folders do
/// not count. Managed settings are not read; the list covers them.
fn plugins_found(home: &Path, cwd: &Path, runner: &dyn ClaudeRunner) -> Vec<FoundPlugin> {
    let files = [
        cwd.join(".claude").join("settings.local.json"),
        cwd.join(".claude").join("settings.json"),
        claude_settings_path(home),
    ];
    let mut decided: Vec<(String, bool, PathBuf)> = Vec::new();
    for file in &files {
        let map = peek_object(file);
        let Some(Value::Object(enabled)) = map.get("enabledPlugins") else {
            continue;
        };
        for (key, value) in enabled {
            if !is_our_plugin(key) || decided.iter().any(|(k, _, _)| k == key) {
                continue;
            }
            decided.push((key.clone(), value.as_bool() == Some(true), file.clone()));
        }
    }
    let installs = applicable_installs(cwd, runner).unwrap_or_default();
    let version_of = |key: &str| {
        installs
            .iter()
            .filter(|i| i.id == key)
            .min_by_key(|i| i.scope_rank)
            .and_then(|i| i.version.clone())
    };
    let shown = |v: &Option<String>| match v {
        Some(v) => format!("version {v}"),
        None => String::from("version unknown"),
    };
    let mut found = Vec::new();
    for (key, _, file) in decided.iter().filter(|(_, on, _)| *on) {
        let version = version_of(key);
        found.push(FoundPlugin {
            evidence: format!(
                "enabledPlugins sets \"{key}\": true in {}; {}",
                file.display(),
                shown(&version)
            ),
            version,
        });
    }
    // A key no settings file names: the install for the narrowest scope.
    let mut listed: Vec<&str> = Vec::new();
    let mut by_scope: Vec<&Install> = installs.iter().collect();
    by_scope.sort_by_key(|i| i.scope_rank);
    for install in by_scope {
        let named = decided.iter().any(|(k, _, _)| *k == install.id);
        if named || listed.contains(&install.id.as_str()) {
            continue;
        }
        listed.push(&install.id);
        if install.on {
            found.push(FoundPlugin {
                evidence: format!(
                    "`claude plugin list --json` reports {} as enabled; {}",
                    install.id,
                    shown(&install.version)
                ),
                version: install.version.clone(),
            });
        }
    }
    found
}

/// Whether the version 3 spatial-memory plugin is enabled for `cwd`, with
/// the evidence. A plugin whose version cannot be read counts as version 3.
pub fn plugin_enabled(home: &Path, cwd: &Path, runner: &dyn ClaudeRunner) -> Option<String> {
    plugins_found(home, cwd, runner)
        .into_iter()
        .find(FoundPlugin::is_current)
        .map(|f| f.evidence)
}

/// The spatial-memory plugins enabled for one folder, read once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginFindings {
    /// The version 3 plugin, or one whose version could not be read.
    pub current: Option<String>,
    /// A version 1 or 2 plugin (the Python server). It is never taken for
    /// the version 3 plugin (B-150).
    pub old: Option<String>,
}

/// The current and the old spatial-memory plugin enabled for `cwd`, with
/// the evidence for each.
pub fn plugin_findings(home: &Path, cwd: &Path, runner: &dyn ClaudeRunner) -> PluginFindings {
    let found = plugins_found(home, cwd, runner);
    PluginFindings {
        current: found
            .iter()
            .find(|f| f.is_current())
            .map(|f| f.evidence.clone()),
        old: found
            .iter()
            .find(|f| !f.is_current())
            .map(|f| f.evidence.clone()),
    }
}

/// What setup and doctor say about an enabled old plugin.
pub fn old_plugin_line(evidence: &str) -> String {
    format!(
        "the version 1 plugin is enabled for this folder ({evidence}). It runs the Python server \
         and its own hooks beside version 3; disable it here with `claude plugin disable \
         spatial-memory@<marketplace> --scope local`"
    )
}

fn plugin_refusal(evidence: &str) -> CoreError {
    CoreError::config_invalid(format!(
        "the spatial-memory plugin is enabled ({evidence}). The plugin already registers the \
         server and the five hooks, and a plugin's hooks run separately from the same hooks in a \
         settings file, so setup would register everything twice"
    ))
    .with_next_action(
        "Keep using the plugin and skip setup, or disable the plugin with /plugin or \
         `claude plugin disable spatial-memory@<marketplace>` and run setup again.",
    )
}

// ---------------------------------------------------------------------------
// Claude Code: the MCP server
// ---------------------------------------------------------------------------

fn exe_string(env: &SetupEnv<'_>) -> String {
    env.exe.to_string_lossy().into_owned()
}

/// The server's arguments: `mcp`, plus `--profile <name>` when pinned.
pub fn mcp_args(pin: Option<&ProfileName>) -> Vec<String> {
    let mut args = vec![String::from("mcp")];
    if let Some(p) = pin {
        args.push(String::from("--profile"));
        args.push(p.as_str().to_owned());
    }
    args
}

/// The user-scope entry `claude mcp add` wrote, if any.
pub fn registered_server(home: &Path) -> Option<Value> {
    let map = peek_object(&claude_json_path(home));
    map.get("mcpServers")?.get(SERVER_NAME).cloned()
}

fn run_claude(runner: &dyn ClaudeRunner, args: Vec<String>) -> Result<(), CoreError> {
    let shown = format!("claude {}", args.join(" "));
    match runner.run(&args) {
        Ok(out) if out.success => Ok(()),
        Ok(out) => Err(CoreError::config_invalid(format!(
            "`{shown}` failed: {}",
            out.stderr.trim()
        ))
        .with_next_action(format!(
            "Run `{shown}` yourself to see why, then run setup again."
        ))),
        Err(e) => Err(
            CoreError::config_invalid(format!("cannot run `{shown}`: {e}")).with_next_action(
                "Install Claude Code or put `claude` on PATH, then run setup again.",
            ),
        ),
    }
}

fn register_mcp(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let exe = exe_string(env);
    let args = mcp_args(env.profile_pin.as_ref());
    let existing = registered_server(&env.home);
    if let Some(entry) = &existing {
        let same_command = entry.get("command").and_then(Value::as_str) == Some(exe.as_str());
        let same_args = entry.get("args") == Some(&json!(args));
        if same_command && same_args {
            return Ok(format!(
                "MCP server: already registered at user scope ({exe} {})",
                args.join(" ")
            ));
        }
        run_claude(
            env.runner,
            ["mcp", "remove", "--scope", "user", SERVER_NAME]
                .map(String::from)
                .to_vec(),
        )?;
    }
    let mut add: Vec<String> = ["mcp", "add", "--scope", "user", SERVER_NAME, "--"]
        .map(String::from)
        .to_vec();
    add.push(exe.clone());
    add.extend(args.iter().cloned());
    run_claude(env.runner, add)?;
    Ok(format!(
        "MCP server: {} with `claude mcp add --scope user` ({exe} {})",
        if existing.is_some() {
            "re-registered"
        } else {
            "registered"
        },
        args.join(" ")
    ))
}

fn unregister_mcp(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    if registered_server(&env.home).is_none() {
        return Ok(String::from("MCP server: not registered at user scope"));
    }
    run_claude(
        env.runner,
        ["mcp", "remove", "--scope", "user", SERVER_NAME]
            .map(String::from)
            .to_vec(),
    )?;
    Ok(String::from(
        "MCP server: removed with `claude mcp remove --scope user`",
    ))
}

// ---------------------------------------------------------------------------
// Claude Code: the hooks
// ---------------------------------------------------------------------------

/// The exec-form handler `setup` writes for one hook.
pub fn claude_handler(spec: &HookSpec, exe: &str, pin: Option<&ProfileName>) -> Value {
    let mut args = vec![
        String::from("hook"),
        spec.event.as_str().to_owned(),
        String::from("--client"),
        String::from("claude-code"),
    ];
    if let Some(p) = pin {
        args.push(String::from("--profile"));
        args.push(p.as_str().to_owned());
    }
    let mut h = Map::new();
    h.insert("type".into(), json!("command"));
    h.insert("command".into(), json!(exe));
    h.insert("args".into(), json!(args));
    h.insert("timeout".into(), json!(spec.timeout));
    if spec.is_async {
        h.insert("async".into(), json!(true));
    }
    Value::Object(h)
}

fn claude_group(spec: &HookSpec, handler: Value) -> Value {
    match spec.matcher {
        Some(m) => json!({"matcher": m, "hooks": [handler]}),
        None => json!({"hooks": [handler]}),
    }
}

fn matcher_of(group: &Value) -> Option<&str> {
    group
        .get("matcher")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
}

/// Every handler of ours: (event key, its group's matcher, the handler).
fn our_claude_handlers(root: &Map<String, Value>) -> Vec<(String, Option<String>, Value)> {
    let mut found = Vec::new();
    let Some(Value::Object(hooks)) = root.get("hooks") else {
        return found;
    };
    for (key, groups) in hooks {
        let Some(groups) = groups.as_array() else {
            continue;
        };
        for group in groups {
            let Some(handlers) = group.get("hooks").and_then(Value::as_array) else {
                continue;
            };
            for h in handlers.iter().filter(|h| handler_is_ours(h)) {
                found.push((key.clone(), matcher_of(group).map(String::from), h.clone()));
            }
        }
    }
    found
}

/// Removes every handler of ours, and any group, event list or `hooks`
/// object that removing them left empty. Returns whether anything went.
fn strip_claude(root: &mut Map<String, Value>) -> bool {
    let Some(Value::Object(hooks)) = root.get_mut("hooks") else {
        return false;
    };
    let mut removed = false;
    let keys: Vec<String> = hooks.keys().cloned().collect();
    for key in keys {
        let Some(Value::Array(groups)) = hooks.get_mut(&key) else {
            continue;
        };
        let mut here = false;
        groups.retain_mut(|group| {
            let Some(Value::Array(handlers)) = group.get_mut("hooks") else {
                return true;
            };
            let before = handlers.len();
            handlers.retain(|h| !handler_is_ours(h));
            if handlers.len() == before {
                return true;
            }
            here = true;
            !handlers.is_empty()
        });
        if here {
            removed = true;
            if groups.is_empty() {
                hooks.remove(&key);
            }
        }
    }
    if removed && hooks.is_empty() {
        root.remove("hooks");
    }
    removed
}

/// Installs the five hooks into a settings object. Returns whether it
/// changed: an object that already holds exactly our five handlers is left
/// alone.
pub fn merge_claude_hooks(
    root: &mut Map<String, Value>,
    exe: &str,
    pin: Option<&ProfileName>,
    path: &Path,
) -> Result<bool, CoreError> {
    let expected: Vec<(String, Option<String>, Value)> = CLAUDE_HOOKS
        .iter()
        .map(|s| {
            (
                s.key.to_owned(),
                s.matcher.map(String::from),
                claude_handler(s, exe, pin),
            )
        })
        .collect();
    let found = our_claude_handlers(root);
    if found.len() == expected.len() && expected.iter().all(|e| found.contains(e)) {
        return Ok(false);
    }
    if root.get("hooks").is_some_and(|h| !h.is_object()) {
        return Err(not_an_object(path));
    }
    strip_claude(root);
    let hooks = root
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(hooks) = hooks else {
        return Err(not_an_object(path));
    };
    for spec in &CLAUDE_HOOKS {
        let list = hooks
            .entry(spec.key)
            .or_insert_with(|| Value::Array(Vec::new()));
        let Value::Array(list) = list else {
            return Err(CoreError::config_invalid(format!(
                "hooks.{} in {} is not a list",
                spec.key,
                path.display()
            ))
            .with_next_action(format!(
                "Fix {}, then run setup again; nothing was changed.",
                path.display()
            )));
        };
        list.push(claude_group(spec, claude_handler(spec, exe, pin)));
    }
    Ok(true)
}

fn install_claude_hooks(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = claude_settings_path(&env.home);
    let exe = exe_string(env);
    edit_file(&path, "hooks", |map| {
        merge_claude_hooks(map, &exe, env.profile_pin.as_ref(), &path)
    })
}

fn remove_claude_hooks(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = claude_settings_path(&env.home);
    edit_file(&path, "hooks", |map| Ok(strip_claude(map)))
}

/// Whether `setup` has registered anything for Claude Code: our hooks in
/// `~/.claude/settings.json` or the user-scope server. Returns where.
pub fn setup_registered(home: &Path) -> Option<String> {
    let settings = claude_settings_path(home);
    if !our_claude_handlers(&peek_object(&settings)).is_empty() {
        return Some(format!("hooks in {}", settings.display()));
    }
    registered_server(home).map(|_| {
        format!(
            "the user-scope MCP server in {}",
            claude_json_path(home).display()
        )
    })
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

/// A program path as one shell word.
fn shell_quote(program: &str) -> String {
    if program.contains(char::is_whitespace) {
        format!("\"{program}\"")
    } else {
        program.to_owned()
    }
}

/// The plain command string Cursor runs for one event.
pub fn cursor_command(exe: &str, event: HookEvent, pin: Option<&ProfileName>) -> String {
    let mut s = format!(
        "{} hook {} --client cursor",
        shell_quote(exe),
        event.as_str()
    );
    if let Some(p) = pin {
        s.push_str(" --profile ");
        s.push_str(p.as_str());
    }
    s
}

fn cursor_entry(
    exe: &str,
    event: HookEvent,
    timeout: u64,
    loop_limit: Option<u64>,
    pin: Option<&ProfileName>,
) -> Value {
    let mut e = Map::new();
    e.insert("command".into(), json!(cursor_command(exe, event, pin)));
    e.insert("timeout".into(), json!(timeout));
    if let Some(l) = loop_limit {
        e.insert("loop_limit".into(), json!(l));
    }
    Value::Object(e)
}

fn our_cursor_entries(root: &Map<String, Value>) -> Vec<(String, Value)> {
    let mut found = Vec::new();
    if let Some(Value::Object(hooks)) = root.get("hooks") {
        for (key, list) in hooks {
            for e in list.as_array().into_iter().flatten() {
                if handler_is_ours(e) {
                    found.push((key.clone(), e.clone()));
                }
            }
        }
    }
    found
}

fn strip_cursor(root: &mut Map<String, Value>) -> bool {
    let Some(Value::Object(hooks)) = root.get_mut("hooks") else {
        return false;
    };
    let mut removed = false;
    let keys: Vec<String> = hooks.keys().cloned().collect();
    for key in keys {
        let Some(Value::Array(list)) = hooks.get_mut(&key) else {
            continue;
        };
        let before = list.len();
        list.retain(|e| !handler_is_ours(e));
        if list.len() != before {
            removed = true;
            if list.is_empty() {
                hooks.remove(&key);
            }
        }
    }
    if removed && hooks.is_empty() {
        root.remove("hooks");
    }
    removed
}

/// Installs the three Cursor hooks into a `hooks.json` object.
pub fn merge_cursor_hooks(
    root: &mut Map<String, Value>,
    exe: &str,
    pin: Option<&ProfileName>,
    path: &Path,
) -> Result<bool, CoreError> {
    let expected: Vec<(String, Value)> = CURSOR_HOOKS
        .iter()
        .map(|(key, event, timeout, limit)| {
            (
                (*key).to_owned(),
                cursor_entry(exe, *event, *timeout, *limit, pin),
            )
        })
        .collect();
    let found = our_cursor_entries(root);
    let versioned = root.contains_key("version");
    if versioned && found.len() == expected.len() && expected.iter().all(|e| found.contains(e)) {
        return Ok(false);
    }
    if root.get("hooks").is_some_and(|h| !h.is_object()) {
        return Err(not_an_object(path));
    }
    strip_cursor(root);
    if !versioned {
        root.insert("version".into(), json!(1));
    }
    let hooks = root
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(hooks) = hooks else {
        return Err(not_an_object(path));
    };
    for (key, entry) in expected {
        let list = hooks
            .entry(key.clone())
            .or_insert_with(|| Value::Array(Vec::new()));
        let Value::Array(list) = list else {
            return Err(CoreError::config_invalid(format!(
                "hooks.{key} in {} is not a list",
                path.display()
            )));
        };
        list.push(entry);
    }
    Ok(true)
}

fn install_cursor_hooks(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = cursor_hooks_path(&env.home);
    let exe = exe_string(env);
    edit_file(&path, "Cursor hooks", |map| {
        merge_cursor_hooks(map, &exe, env.profile_pin.as_ref(), &path)
    })
}

fn remove_cursor_hooks(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = cursor_hooks_path(&env.home);
    edit_file(&path, "Cursor hooks", |map| {
        let removed = strip_cursor(map);
        // `version` alone is what an install into a missing file left.
        if removed && map.len() == 1 && map.get("version") == Some(&json!(1)) {
            map.remove("version");
        }
        Ok(removed)
    })
}

fn install_cursor_mcp(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = cursor_mcp_path(&env.home);
    let exe = exe_string(env);
    let args = mcp_args(env.profile_pin.as_ref());
    edit_file(&path, "Cursor MCP server", |map| {
        let servers = map
            .entry("mcpServers")
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(servers) = servers else {
            return Err(not_an_object(&path));
        };
        let entry = servers
            .entry(SERVER_NAME)
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(entry) = entry else {
            return Err(not_an_object(&path));
        };
        let before = entry.clone();
        entry.insert("command".into(), json!(exe));
        entry.insert("args".into(), json!(args));
        Ok(*entry != before)
    })
}

fn remove_cursor_mcp(env: &SetupEnv<'_>) -> Result<String, CoreError> {
    let path = cursor_mcp_path(&env.home);
    edit_file(&path, "Cursor MCP server", |map| {
        let Some(Value::Object(servers)) = map.get_mut("mcpServers") else {
            return Ok(false);
        };
        if servers.remove(SERVER_NAME).is_none() {
            return Ok(false);
        }
        if servers.is_empty() {
            map.remove("mcpServers");
        }
        Ok(true)
    })
}

/// What one judge call of 25 captures cost with haiku, in US dollars at API
/// prices, by Claude Code's own estimate (`SPIKE-JUDGE.md`, batch size 25).
pub const JUDGE_CALL_USD: f64 = 0.00187;

/// Judge calls on a typical day of the owner's use after the mechanical
/// fixes (`SPIKE-JUDGE.md`, "A day of the owner's use").
pub const JUDGE_TYPICAL_CALLS: u32 = 35;

/// The capture judge notice `setup` prints before it finishes (SPEC section
/// 16, The judge; B-170): automatic capture runs Claude Code in the
/// background under the user's own sign-in, roughly what that costs, and how
/// to turn it off. `config` is the settings file named for turning it off;
/// `claude` is the Claude Code executable the judge would run, if any.
pub fn judge_notice(settings: &Settings, config: &Path, claude: Option<&str>) -> Vec<String> {
    let capture = &settings.capture;
    let judge = &capture.judge;
    let config = config.display();
    let days = judge.pending_max_days;
    if !capture.enabled {
        return vec![String::from(
            "capture: automatic capture is off (capture.enabled = false), so nothing is sent to \
             Claude Code. Memories are stored only when you ask.",
        )];
    }
    if !judge.enabled {
        return vec![format!(
            "capture: the capture judge is off (capture.judge.enabled = false). Automatic \
             captures wait as pending, are never recalled, and are deleted after {days} days. To \
             turn the judge on, set `enabled = true` under `[capture.judge]` in {config}, then \
             run `spatial-memory stop`."
        )];
    }
    let cap = judge.max_calls_per_day;
    let model = &judge.model;
    let measured = if model == sm_core::config::DEFAULT_JUDGE_MODEL {
        String::new()
    } else {
        format!(" These figures are for haiku; {model} costs more.")
    };
    let mut lines = vec![
        format!(
            "capture: automatic capture uses your Claude plan in the background. The daemon \
             runs Claude Code (`claude -p --model {model}`) to keep or reject each automatic \
             capture: in batches of up to 25, one call at a time, at lower priority, at most \
             {cap} calls a day."
        ),
        format!(
            "capture: that is about {:.4} US dollars a call, about {:.2} on a typical day \
             ({JUDGE_TYPICAL_CALLS} calls) and at most {:.2} a day at the cap, by Claude Code's \
             own estimate at API prices.{measured} Signed in with a Claude subscription, it is \
             plan usage, not a charge; signed in with an Anthropic Console account, it is billed.",
            JUDGE_CALL_USD,
            JUDGE_CALL_USD * f64::from(JUDGE_TYPICAL_CALLS),
            JUDGE_CALL_USD * f64::from(cap),
        ),
        format!(
            "capture: to turn it off, set `enabled = false` under `[capture.judge]` in {config}, \
             then run `spatial-memory stop`. Captures then wait as pending, are never recalled, \
             and are deleted after {days} days. `spatial-memory status` shows the judge's calls \
             and cost today."
        ),
    ];
    if claude.is_none() {
        lines.push(format!(
            "capture: Claude Code was not found, so the judge cannot run yet: captures wait as \
             pending and are deleted after {days} days. Install Claude Code and sign in, or name \
             its executable in capture.judge.command."
        ));
    }
    lines
}

/// The failure `setup` reports when only the model step failed.
pub fn model_failure(e: CoreError) -> CoreError {
    let message = format!(
        "everything else was registered, but the model step failed: {}",
        e.message
    );
    CoreError {
        code: e.code,
        message,
        next_action: e.next_action,
    }
}

impl SetupReport {
    /// Whether every step, the model step included, succeeded.
    pub fn succeeded(&self) -> bool {
        self.model_error.is_none()
    }
}

/// The `ErrorCode` a refusal carries, for tests.
pub const REFUSAL_CODE: ErrorCode = ErrorCode::ConfigInvalid;

#[cfg(test)]
mod tests {
    use super::*;

    /// Setup says that capture uses the plan, roughly how much, and how to
    /// turn it off; with the judge or capture off, or no Claude Code, it says
    /// what happens instead (SPEC section 16, The judge; B-170).
    #[test]
    fn s16_setup_notice_says_capture_uses_the_plan_its_cost_and_how_to_turn_it_off() {
        let config = Path::new("C:/sm/config.toml");
        let mut settings = Settings::default();
        let on = judge_notice(&settings, config, Some("claude"));
        assert_eq!(on.len(), 3, "{on:#?}");
        let text = on.join("\n");
        assert!(
            text.contains("automatic capture uses your Claude plan in the background"),
            "{text}"
        );
        assert!(text.contains("`claude -p --model haiku`"), "{text}");
        assert!(text.contains("at most 200 calls a day"), "{text}");
        assert!(
            text.contains(
                "about 0.0019 US dollars a call, about 0.07 on a typical day (35 calls) and at \
                 most 0.37 a day at the cap"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "set `enabled = false` under `[capture.judge]` in C:/sm/config.toml, then run \
                 `spatial-memory stop`"
            ),
            "{text}"
        );
        assert!(text.contains("deleted after 14 days"), "{text}");
        assert!(!text.contains("costs more"), "{text}");

        settings.capture.judge.model = String::from("sonnet");
        settings.capture.judge.max_calls_per_day = 20;
        let text = judge_notice(&settings, config, None).join("\n");
        assert!(text.contains("at most 0.04 a day at the cap"), "{text}");
        assert!(text.contains("sonnet costs more"), "{text}");
        assert!(text.contains("Claude Code was not found"), "{text}");

        settings.capture.judge.enabled = false;
        let off = judge_notice(&settings, config, Some("claude"));
        assert_eq!(off.len(), 1);
        assert!(off[0].contains("the capture judge is off"), "{off:?}");
        assert!(off[0].contains("never recalled"), "{off:?}");
        assert!(!off[0].contains("plan"), "{off:?}");

        settings.capture.enabled = false;
        let none = judge_notice(&settings, config, Some("claude"));
        assert_eq!(none.len(), 1);
        assert!(
            none[0].contains("nothing is sent to Claude Code"),
            "{none:?}"
        );
    }

    #[test]
    fn s16_handlers_are_recognised_in_exec_and_shell_form() {
        assert!(handler_is_ours(&json!({
            "type": "command",
            "command": "C:\\Program Files\\sm\\spatial-memory.exe",
            "args": ["hook", "stop", "--client", "claude-code"]
        })));
        assert!(handler_is_ours(&json!({
            "type": "command",
            "command": "spatial-memory hook post-tool-use --client claude-code"
        })));
        assert!(handler_is_ours(&json!({
            "command": "\"C:\\Program Files\\sm\\spatial-memory.exe\" hook stop --client cursor"
        })));
        assert!(!handler_is_ours(&json!({
            "type": "command",
            "command": "spatial-memory",
            "args": ["mcp"]
        })));
        assert!(!handler_is_ours(
            &json!({"type": "command", "command": "other-tool hook stop"})
        ));
        assert!(!handler_is_ours(&json!({"type": "prompt", "prompt": "x"})));
        assert!(program_is_ours("/usr/local/bin/spatial-memory"));
        assert!(program_is_ours("C:\\x\\Spatial-Memory.EXE"));
        assert!(!program_is_ours("spatial-memory-mcp"));
    }

    #[test]
    fn s16_shell_words_honour_quotes() {
        assert_eq!(
            shell_words("\"a b\\c.exe\" hook  stop"),
            vec!["a b\\c.exe", "hook", "stop"]
        );
        assert!(shell_words("   ").is_empty());
    }

    #[test]
    fn s16_cursor_command_is_a_plain_string_quoted_when_needed() {
        let pin = ProfileName::parse("work").unwrap();
        assert_eq!(
            cursor_command("/opt/sm/spatial-memory", HookEvent::Stop, None),
            "/opt/sm/spatial-memory hook stop --client cursor"
        );
        assert_eq!(
            cursor_command(
                "C:\\Program Files\\spatial-memory.exe",
                HookEvent::PreCompact,
                Some(&pin)
            ),
            "\"C:\\Program Files\\spatial-memory.exe\" hook pre-compact --client cursor --profile work"
        );
    }
}
