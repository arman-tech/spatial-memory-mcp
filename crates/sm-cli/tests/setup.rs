//! `setup` registration (SPEC 16, Registration). The library runs against a
//! temporary home with a fake `claude` that records its arguments and keeps
//! `~/.claude.json` as `claude mcp add --scope user` and `claude mcp remove
//! --scope user` would; the real `claude mcp` commands are never run. The
//! binary runs in the sandbox for the cases that start no `claude`.

mod common;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use common::Sandbox;
use serde_json::{Map, Value, json};
use sm_cli::model_step::ModelStep;
use sm_cli::setup::{
    self, CLAUDE_HOOKS, CURSOR_HOOKS, ClaudeRunner, REFUSAL_CODE, RunOutput, SetupClient, SetupEnv,
};
use sm_core::config::{ProfileName, Settings};
use sm_core::error::CoreError;

/// A `claude` stand-in.
struct FakeClaude {
    home: PathBuf,
    calls: Mutex<Vec<Vec<String>>>,
    /// What `claude plugin list --json` prints.
    plugin_list: String,
}

impl FakeClaude {
    fn new(home: &Path) -> Self {
        FakeClaude {
            home: home.to_path_buf(),
            calls: Mutex::new(Vec::new()),
            plugin_list: String::from("[]"),
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.join(" "))
            .collect()
    }

    fn edit_claude_json(&self, change: impl FnOnce(&mut Map<String, Value>)) {
        let path = setup::claude_json_path(&self.home);
        let mut map = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Map<String, Value>>(&t).ok())
            .unwrap_or_default();
        change(&mut map);
        std::fs::write(&path, serde_json::to_string_pretty(&map).unwrap()).unwrap();
    }
}

impl ClaudeRunner for FakeClaude {
    fn run(&self, args: &[String]) -> io::Result<RunOutput> {
        self.calls.lock().unwrap().push(args.to_vec());
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let stdout = match a.as_slice() {
            ["plugin", "list", "--json"] => self.plugin_list.clone(),
            [
                "mcp",
                "add",
                "--scope",
                "user",
                name,
                "--",
                command,
                rest @ ..,
            ] => {
                let entry = json!({"type": "stdio", "command": command, "args": rest, "env": {}});
                let name = (*name).to_owned();
                self.edit_claude_json(|m| {
                    let servers = m.entry("mcpServers").or_insert_with(|| json!({}));
                    servers.as_object_mut().unwrap().insert(name, entry);
                });
                String::new()
            }
            ["mcp", "remove", "--scope", "user", name] => {
                self.edit_claude_json(|m| {
                    if let Some(Value::Object(s)) = m.get_mut("mcpServers") {
                        s.remove(*name);
                    }
                });
                String::new()
            }
            other => panic!("unexpected claude call: {other:?}"),
        };
        Ok(RunOutput {
            success: true,
            stdout,
            stderr: String::new(),
        })
    }
}

/// A model step that succeeds without touching any model.
struct FakeModel;

impl ModelStep for FakeModel {
    fn ensure(&self, _settings: &Settings) -> Result<String, CoreError> {
        Ok(String::from("fake model present"))
    }
    fn check(&self, _settings: &Settings) -> Result<String, CoreError> {
        Ok(String::from("fake model present"))
    }
}

struct Home {
    _dir: tempfile::TempDir,
    home: PathBuf,
    cwd: PathBuf,
    exe: PathBuf,
}

impl Home {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let cwd = dir.path().join("project");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        let exe = dir.path().join("Program Files").join(if cfg!(windows) {
            "spatial-memory.exe"
        } else {
            "spatial-memory"
        });
        Home {
            _dir: dir,
            home,
            cwd,
            exe,
        }
    }

    fn env<'a>(&self, runner: &'a FakeClaude) -> SetupEnv<'a> {
        SetupEnv {
            home: self.home.clone(),
            cwd: self.cwd.clone(),
            exe: self.exe.clone(),
            runner,
            model: &FakeModel,
            settings: Settings::default(),
            profile_pin: None,
        }
    }

    fn exe(&self) -> String {
        self.exe.to_string_lossy().into_owned()
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn parse(path: &Path) -> Value {
    serde_json::from_str(&read(path)).unwrap()
}

/// The backups `setup` made next to `path`.
fn backups(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let prefix = format!("{name}.spatial-memory-");
    let mut found: Vec<PathBuf> = std::fs::read_dir(path.parent().unwrap())
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    let n = p.file_name().unwrap().to_string_lossy();
                    n.starts_with(&prefix) && n.ends_with(".bak")
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

/// A user's own settings: a hook of theirs and other keys.
const USER_SETTINGS: &str = r#"{
  "theme": "dark",
  "permissions": {
    "allow": ["Bash(cargo test:*)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [{"type": "command", "command": "cargo fmt"}]
      }
    ]
  }
}
"#;

fn write_user_settings(home: &Path, text: &str) -> PathBuf {
    let path = setup::claude_settings_path(home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, text).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Claude Code
// ---------------------------------------------------------------------------

#[test]
fn s16_setup_twice_gives_the_same_settings_file_and_one_backup_per_change() {
    let h = Home::new();
    let settings = write_user_settings(&h.home, USER_SETTINGS);
    let claude = FakeClaude::new(&h.home);
    let first = setup::run(&h.env(&claude), SetupClient::ClaudeCode, false).unwrap();
    assert_eq!(first.model_error, None);
    let after_first = read(&settings);
    let backups_first = backups(&settings);
    assert_eq!(backups_first.len(), 1, "one change, one backup");
    assert_eq!(
        read(&backups_first[0]),
        USER_SETTINGS,
        "the backup is the original"
    );

    // The user's keys and hook stay; ours are the five exec-form handlers.
    let v = parse(&settings);
    assert_eq!(v["theme"], "dark");
    assert_eq!(v["permissions"]["allow"][0], "Bash(cargo test:*)");
    assert_eq!(
        v["hooks"]["PostToolUse"][0]["hooks"][0]["command"],
        "cargo fmt"
    );
    for spec in &CLAUDE_HOOKS {
        let groups = v["hooks"][spec.key].as_array().unwrap();
        let ours: Vec<&Value> = groups
            .iter()
            .filter(|g| g["hooks"][0]["command"] == h.exe().as_str())
            .collect();
        assert_eq!(ours.len(), 1, "{}", spec.key);
        let handler = &ours[0]["hooks"][0];
        assert_eq!(handler, &setup::claude_handler(spec, &h.exe(), None));
        assert_eq!(handler["type"], "command");
        assert_eq!(
            handler["args"],
            json!(["hook", spec.event.as_str(), "--client", "claude-code"])
        );
        assert_eq!(handler["timeout"], spec.timeout);
        assert_eq!(handler.get("async").is_some(), spec.is_async);
        assert_eq!(ours[0].get("matcher").and_then(Value::as_str), spec.matcher);
    }
    let server = setup::registered_server(&h.home).unwrap();
    assert_eq!(server["command"], h.exe().as_str());
    assert_eq!(server["args"], json!(["mcp"]));
    assert_eq!(
        claude.calls(),
        [
            "plugin list --json".to_owned(),
            format!("mcp add --scope user spatial-memory -- {} mcp", h.exe()),
        ]
    );

    // The second run changes nothing and backs nothing up.
    let second = setup::run(&h.env(&claude), SetupClient::ClaudeCode, false).unwrap();
    assert_eq!(read(&settings), after_first, "the same settings file");
    assert_eq!(
        backups(&settings),
        backups_first,
        "no backup without a change"
    );
    assert!(
        second
            .lines
            .iter()
            .any(|l| l.contains("already registered")),
        "{second:?}"
    );
    assert!(
        second.lines.iter().any(|l| l.contains("unchanged")),
        "{second:?}"
    );
    assert_eq!(
        claude.calls().len(),
        3,
        "only the plugin check: {:?}",
        claude.calls()
    );
}

#[test]
fn s16_setup_remove_restores_the_original() {
    let h = Home::new();
    let settings = write_user_settings(&h.home, USER_SETTINGS);
    let claude = FakeClaude::new(&h.home);
    setup::run(&h.env(&claude), SetupClient::ClaudeCode, false).unwrap();
    let removed = setup::run(&h.env(&claude), SetupClient::ClaudeCode, true).unwrap();
    assert_eq!(removed.model_error, None);
    assert_eq!(
        parse(&settings),
        serde_json::from_str::<Value>(USER_SETTINGS).unwrap(),
        "the original settings"
    );
    assert_eq!(setup::registered_server(&h.home), None);
    assert_eq!(backups(&settings).len(), 2, "one backup per change");
    assert_eq!(
        claude.calls().last().unwrap(),
        "mcp remove --scope user spatial-memory"
    );
    // A second removal finds nothing to do.
    let again = setup::run(&h.env(&claude), SetupClient::ClaudeCode, true).unwrap();
    assert!(
        again.lines.iter().any(|l| l.contains("not registered")),
        "{again:?}"
    );
    assert_eq!(backups(&settings).len(), 2);

    // With no settings file before setup, removal leaves an empty object.
    let fresh = Home::new();
    let claude = FakeClaude::new(&fresh.home);
    setup::run(&fresh.env(&claude), SetupClient::ClaudeCode, false).unwrap();
    let path = setup::claude_settings_path(&fresh.home);
    assert!(backups(&path).is_empty(), "nothing to back up");
    setup::run(&fresh.env(&claude), SetupClient::ClaudeCode, true).unwrap();
    assert_eq!(parse(&path), json!({}));
}

#[test]
fn s16_setup_pins_an_explicit_profile_in_every_registration() {
    let h = Home::new();
    let claude = FakeClaude::new(&h.home);
    let mut env = h.env(&claude);
    env.profile_pin = Some(ProfileName::parse("work").unwrap());
    setup::run(&env, SetupClient::ClaudeCode, false).unwrap();
    let v = parse(&setup::claude_settings_path(&h.home));
    for spec in &CLAUDE_HOOKS {
        let args = &v["hooks"][spec.key][0]["hooks"][0]["args"];
        assert_eq!(args[4], "--profile");
        assert_eq!(args[5], "work");
    }
    assert_eq!(
        setup::registered_server(&h.home).unwrap()["args"],
        json!(["mcp", "--profile", "work"])
    );
}

#[test]
fn s16_setup_refuses_while_the_plugin_is_enabled() {
    // Enabled in user, project or local settings, or by the plugin list.
    let user = |h: &Home| {
        write_user_settings(
            &h.home,
            r#"{"enabledPlugins": {"spatial-memory@spatial-memory": true}}"#,
        )
    };
    let project = |h: &Home| {
        let p = h.cwd.join(".claude").join("settings.json");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            r#"{"enabledPlugins": {"spatial-memory@team-tools": true}}"#,
        )
        .unwrap();
        p
    };
    let local = |h: &Home| {
        let p = h.cwd.join(".claude").join("settings.local.json");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            r#"{"enabledPlugins": {"spatial-memory@spatial-memory": true}}"#,
        )
        .unwrap();
        p
    };
    for (name, enable) in [
        ("user", &user as &dyn Fn(&Home) -> PathBuf),
        ("project", &project),
        ("local", &local),
    ] {
        let h = Home::new();
        let file = enable(&h);
        let before = read(&file);
        let claude = FakeClaude::new(&h.home);
        let err = setup::run(&h.env(&claude), SetupClient::ClaudeCode, false)
            .expect_err("refused while the plugin is enabled");
        assert_eq!(err.code, REFUSAL_CODE, "{name}");
        assert!(
            err.message.contains("plugin is enabled"),
            "{name}: {}",
            err.message
        );
        assert!(err.next_action.contains("claude plugin disable"), "{name}");
        assert_eq!(read(&file), before, "{name}: nothing written");
        // Only the read-only plugin list, for the installed version: the
        // version 1 and 3 plugins can share one key (B-150).
        assert_eq!(claude.calls(), ["plugin list --json"], "{name}");
        assert_eq!(setup::registered_server(&h.home), None, "{name}");
    }

    // The plugin list reports it enabled (installed with its default on).
    let h = Home::new();
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list =
        r#"[{"id": "spatial-memory@spatial-memory", "version": "3.0.0", "scope": "user", "enabled": true}]"#
            .to_owned();
    let err = setup::run(&h.env(&claude), SetupClient::ClaudeCode, false).unwrap_err();
    assert_eq!(err.code, REFUSAL_CODE);
    assert!(
        err.message.contains("claude plugin list --json"),
        "{}",
        err.message
    );
    assert_eq!(claude.calls(), ["plugin list --json"]);
    assert!(!setup::claude_settings_path(&h.home).exists());

    // Disabled everywhere: setup goes ahead.
    let h = Home::new();
    write_user_settings(
        &h.home,
        r#"{"enabledPlugins": {"spatial-memory@spatial-memory": false}}"#,
    );
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list =
        r#"[{"id": "spatial-memory@spatial-memory", "enabled": false}]"#.to_owned();
    setup::run(&h.env(&claude), SetupClient::ClaudeCode, false).expect("disabled plugin");
    assert!(setup::registered_server(&h.home).is_some());
    // The plugin's removal is never refused.
    let h2 = Home::new();
    user(&h2);
    let claude = FakeClaude::new(&h2.home);
    setup::run(&h2.env(&claude), SetupClient::ClaudeCode, true).expect("remove is allowed");
}

/// One `claude plugin list --json` item for this folder's install.
fn install(h: &Home, id: &str, version: &str, scope: &str, on: bool) -> Value {
    json!({
        "id": id,
        "version": version,
        "scope": scope,
        "enabled": on,
        "projectEnabled": on,
        "projectPath": h.cwd.to_string_lossy(),
    })
}

fn write_project_settings(h: &Home, file: &str, text: &str) -> PathBuf {
    let p = h.cwd.join(".claude").join(file);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, text).unwrap();
    p
}

#[test]
fn s16_a_plugin_disabled_at_local_scope_counts_as_disabled() {
    // The state of the six projects whose version 1 plugin was turned off
    // for version 3 (B-150): the shared project settings enable it, the
    // local settings disable it. Local settings win, and the plugin list
    // must not bring the plugin back, whatever it reports for the key.
    let h = Home::new();
    let key = "spatial-memory@spatial-memory-marketplace";
    write_project_settings(
        &h,
        "settings.json",
        &format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    );
    write_project_settings(
        &h,
        "settings.local.json",
        &format!(r#"{{"enabledPlugins": {{"{key}": false}}}}"#),
    );
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list = json!([install(&h, key, "3.0.0", "project", true)]).to_string();
    assert_eq!(setup::plugin_enabled(&h.home, &h.cwd, &claude), None);
    setup::run(&h.env(&claude), SetupClient::ClaudeCode, false)
        .expect("a plugin disabled at local scope does not block setup");
    assert!(setup::registered_server(&h.home).is_some());
}

#[test]
fn s16_the_version_1_plugin_is_never_taken_for_the_version_3_plugin() {
    let key = "spatial-memory@spatial-memory-marketplace";
    // Enabled in the project settings, installed for this folder at 1.11.4.
    let h = Home::new();
    write_project_settings(
        &h,
        "settings.json",
        &format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    );
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list = json!([install(&h, key, "1.11.4", "project", true)]).to_string();
    assert_eq!(setup::plugin_enabled(&h.home, &h.cwd, &claude), None);
    let report = setup::run(&h.env(&claude), SetupClient::ClaudeCode, false)
        .expect("the version 1 plugin does not block setup");
    assert!(
        report.lines.iter().any(|l| l.contains("version 1 plugin")
            && l.contains("1.11.4")
            && l.contains("claude plugin disable")),
        "setup says the old plugin should be disabled: {:?}",
        report.lines
    );

    // Reported enabled only by the plugin list, at user scope, at 1.11.1.
    let h = Home::new();
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list = json!([install(&h, key, "1.11.1", "user", true)]).to_string();
    assert_eq!(setup::plugin_enabled(&h.home, &h.cwd, &claude), None);

    // Version 3 enabled for another folder only: not enabled here.
    let h = Home::new();
    let mut claude = FakeClaude::new(&h.home);
    let mut other = install(&h, key, "3.0.0", "project", true);
    other["projectPath"] = json!(h.home.join("elsewhere").to_string_lossy());
    claude.plugin_list = json!([other]).to_string();
    assert_eq!(setup::plugin_enabled(&h.home, &h.cwd, &claude), None);

    // Version 3 enabled here is still refused, whichever way it is enabled.
    let h = Home::new();
    write_project_settings(
        &h,
        "settings.json",
        &format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    );
    let mut claude = FakeClaude::new(&h.home);
    claude.plugin_list = json!([install(&h, key, "3.0.0", "project", true)]).to_string();
    let found = setup::plugin_enabled(&h.home, &h.cwd, &claude).expect("version 3 is seen");
    assert!(found.contains("3.0.0"), "{found}");
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

#[test]
fn s16_cursor_files_hold_plain_command_strings() {
    let h = Home::new();
    let mcp = setup::cursor_mcp_path(&h.home);
    std::fs::create_dir_all(mcp.parent().unwrap()).unwrap();
    let original_mcp = "{\n  \"mcpServers\": {\n    \"other\": {\"command\": \"other-server\"}\n  },\n  \"note\": \"keep\"\n}\n";
    std::fs::write(&mcp, original_mcp).unwrap();
    let claude = FakeClaude::new(&h.home);
    let report = setup::run(&h.env(&claude), SetupClient::Cursor, false).unwrap();
    assert_eq!(report.model_error, None);
    assert!(claude.calls().is_empty(), "Cursor setup runs no claude");

    let hooks_path = setup::cursor_hooks_path(&h.home);
    let hooks = parse(&hooks_path);
    assert_eq!(hooks["version"], 1);
    for (key, event, timeout, limit) in &CURSOR_HOOKS {
        let list = hooks["hooks"][*key].as_array().unwrap();
        assert_eq!(list.len(), 1, "{key}");
        let command = list[0]["command"].as_str().expect("a plain string");
        assert_eq!(command, setup::cursor_command(&h.exe(), *event, None));
        // The program path has a space, so it is quoted as one word.
        assert_eq!(
            command,
            format!("\"{}\" hook {} --client cursor", h.exe(), event.as_str())
        );
        assert_eq!(list[0]["timeout"], *timeout);
        assert_eq!(list[0].get("loop_limit").and_then(Value::as_u64), *limit);
    }
    let m = parse(&mcp);
    assert_eq!(m["mcpServers"]["other"]["command"], "other-server");
    assert_eq!(m["note"], "keep");
    assert_eq!(
        m["mcpServers"]["spatial-memory"],
        json!({"command": h.exe(), "args": ["mcp"]})
    );
    assert_eq!(backups(&mcp).len(), 1);
    assert!(backups(&hooks_path).is_empty());

    // Twice: the same files, no new backups.
    let (hooks_text, mcp_text) = (read(&hooks_path), read(&mcp));
    setup::run(&h.env(&claude), SetupClient::Cursor, false).unwrap();
    assert_eq!(read(&hooks_path), hooks_text);
    assert_eq!(read(&mcp), mcp_text);
    assert_eq!(backups(&mcp).len(), 1);
    assert!(backups(&hooks_path).is_empty());

    // Removal restores the MCP file and empties the hooks file.
    setup::run(&h.env(&claude), SetupClient::Cursor, true).unwrap();
    assert_eq!(
        parse(&mcp),
        serde_json::from_str::<Value>(original_mcp).unwrap()
    );
    assert_eq!(parse(&hooks_path), json!({}));
}

#[test]
fn s16_cursor_setup_keeps_the_users_own_hooks() {
    let h = Home::new();
    let path = setup::cursor_hooks_path(&h.home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = r#"{"version": 1, "hooks": {"stop": [{"command": "./notify.sh"}], "beforeShellExecution": [{"command": "./audit.sh"}]}}"#;
    std::fs::write(&path, original).unwrap();
    let claude = FakeClaude::new(&h.home);
    setup::run(&h.env(&claude), SetupClient::Cursor, false).unwrap();
    let v = parse(&path);
    assert_eq!(v["hooks"]["stop"][0]["command"], "./notify.sh");
    assert_eq!(v["hooks"]["stop"].as_array().unwrap().len(), 2);
    assert_eq!(
        v["hooks"]["beforeShellExecution"][0]["command"],
        "./audit.sh"
    );
    setup::run(&h.env(&claude), SetupClient::Cursor, true).unwrap();
    assert_eq!(
        parse(&path),
        serde_json::from_str::<Value>(original).unwrap()
    );
}

// ---------------------------------------------------------------------------
// The binary, in the sandbox
// ---------------------------------------------------------------------------

#[test]
fn s16_the_binary_registers_with_cursor_in_the_users_home_only() {
    let sb = Sandbox::new();
    let run = sb.run(&["setup", "--client", "cursor"], &[], b"");
    // Everything is registered; the model step is the failure. The sandbox
    // names an empty model folder and a local model address that answers
    // 404, so the step tries the first pinned file there and nothing else.
    assert!(run.stdout.contains("Cursor MCP server: created"), "{run:?}");
    assert!(run.stdout.contains("Cursor hooks: created"), "{run:?}");
    assert_ne!(run.code, Some(0), "{run:?}");
    assert!(run.stderr.contains("model step failed"), "{run:?}");
    assert!(run.stderr.contains("error: model_missing:"), "{run:?}");
    assert!(run.stderr.contains(&sb.model_url), "{run:?}");
    assert!(run.stderr.contains("\nnext: "), "{run:?}");
    let asked = sb.model_requests();
    assert_eq!(asked.len(), 1, "{asked:?}");
    assert!(asked[0].ends_with("/onnx/model.onnx"), "{asked:?}");
    let model_dir = sb.user_home.join("no-model");
    for f in sm_embed::model::MODEL_FILES {
        assert!(!f.local_path(&model_dir).exists(), "{}", f.path);
    }
    let hooks = parse(&setup::cursor_hooks_path(&sb.user_home));
    let command = hooks["hooks"]["stop"][0]["command"].as_str().unwrap();
    assert!(command.ends_with("hook stop --client cursor"), "{command}");
    let removed = sb.run(&["setup", "--client", "cursor", "--remove"], &[], b"");
    assert_eq!(removed.code, Some(0), "{removed:?}");
    assert_eq!(parse(&setup::cursor_mcp_path(&sb.user_home)), json!({}));
}

#[test]
fn s16_the_binary_refuses_claude_code_setup_while_the_plugin_is_enabled() {
    let sb = Sandbox::new();
    let settings = write_user_settings(
        &sb.user_home,
        r#"{"enabledPlugins": {"spatial-memory@spatial-memory": true}}"#,
    );
    let before = read(&settings);
    let run = sb.run(&["setup"], &[], b"");
    assert_ne!(run.code, Some(0), "{run:?}");
    assert!(run.stderr.contains("plugin is enabled"), "{run:?}");
    assert!(run.stderr.contains("claude plugin disable"), "{run:?}");
    assert_eq!(read(&settings), before);
    // Nothing registered. Setup now asks the real `claude` on PATH for its
    // plugin list (B-150), and Claude Code creates its own `~/.claude.json`
    // in an empty sandbox home when it runs, so the file may exist; it must
    // hold no spatial-memory server.
    assert_eq!(setup::registered_server(&sb.user_home), None);
}

/// Everything in the sandbox home: each file's path and contents.
fn home_files(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn s16_setup_model_only_downloads_the_model_and_registers_nothing() {
    // B-153 (D20): the plugin route needs the model too, and `setup`
    // refuses while the plugin is enabled. `setup --model-only` runs the
    // model step alone: no `claude` call, no settings file, no server.
    let h = Home::new();
    let key = "spatial-memory@spatial-memory";
    write_user_settings(
        &h.home,
        &format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    );
    let before = home_files(&h.home);
    let claude = FakeClaude::new(&h.home);
    let report = setup::model_only(&FakeModel, &Settings::default());
    assert_eq!(report.lines, ["model: fake model present"]);
    assert_eq!(report.model_error, None);
    assert!(claude.calls().is_empty(), "{:?}", claude.calls());
    assert_eq!(home_files(&h.home), before, "nothing written in the home");

    // The binary, with the plugin enabled in the user settings. The model
    // step is the only thing that runs: it asks the sandbox's model address
    // for the first file, which answers 404, so the command fails with
    // model_missing and names no registration.
    let sb = Sandbox::new();
    write_user_settings(
        &sb.user_home,
        &format!(r#"{{"enabledPlugins": {{"{key}": true}}}}"#),
    );
    let before = home_files(&sb.user_home);
    let run = sb.run(&["setup", "--model-only"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    assert!(run.stderr.contains("error: model_missing:"), "{run:?}");
    assert!(!run.stderr.contains("plugin is enabled"), "{run:?}");
    assert!(!run.stderr.contains("registered"), "{run:?}");
    assert!(!run.stdout.contains("MCP server"), "{run:?}");
    assert!(!run.stdout.contains("hooks"), "{run:?}");
    let asked = sb.model_requests();
    assert_eq!(asked.len(), 1, "{asked:?}");
    assert!(asked[0].ends_with("/onnx/model.onnx"), "{asked:?}");
    assert_eq!(
        home_files(&sb.user_home),
        before,
        "nothing written in the home"
    );

    // It cannot be combined with registering or removing.
    for args in [
        &["setup", "--model-only", "--remove"][..],
        &["setup", "--model-only", "--client", "cursor"],
    ] {
        let run = sb.run(args, &[], b"");
        assert_eq!(run.code, Some(2), "{args:?}: {run:?}");
    }
}
