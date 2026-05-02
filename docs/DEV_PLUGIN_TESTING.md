# Testing the spatial-memory Claude Code plugin from local source

This guide walks through installing, verifying, and uninstalling the
**development** build of the `spatial-memory` Claude Code plugin —
i.e., from a local clone of this repository, **not** the PyPI release.
It covers both the user-level (default) install scope and the
project-level scope, plus a manual `.claude/settings.json` fallback
path that does not use the plugin marketplace.

---

## Assumptions

These are stated up front so you can stop early if any of them does not
match your environment.

1. **Local source, not PyPI.** You are testing changes on the
   `feature/v2-implementations` branch (or a working copy thereof) and
   must execute that source — not `uvx --from spatial-memory-mcp`.
2. **OS.** Linux or macOS is the primary target. Windows is supported
   with one caveat called out in [Pitfalls](#common-pitfalls).
3. **Claude Code v2.1.37 or newer** is installed and on `PATH` as
   `claude`. The plugin and marketplace slash commands referenced below
   ship with that version.
4. **Python 3.10+** with `pip`, available on `PATH` as `python` (the
   plugin's `.mcp.json` and `hooks.json` invoke the bare `python`
   command — see [Pitfalls](#common-pitfalls) if you only have
   `python3`).
5. **Editable install is mandatory.** `plugin/.mcp.json` is currently
   committed in *dev* mode (`python -m spatial_memory`), so the
   `spatial_memory` package must be importable in whatever Python
   environment Claude Code's stdio child process inherits.
6. **`${CLAUDE_PLUGIN_ROOT}` is provided by Claude Code.** All four
   plugin hooks rely on it to locate the cloned `plugin/hooks/`
   directory at runtime; you do not need to set it yourself.
7. **Default `memory_path` is CWD-relative.** Without
   `SPATIAL_MEMORY_MEMORY_PATH`, the server creates `./.spatial-memory`
   in whatever directory Claude Code launched the MCP child from. This
   surprises people; pin it explicitly when in doubt.
8. **Repo layout.** This guide assumes the clone lives at
   `~/code/spatial-memory-mcp`. Substitute your absolute path
   throughout — relative paths will not work for `claude plugin
   marketplace add`.

---

## Prerequisites

```bash
claude --version            # expect >= 2.1.37
python --version            # expect >= 3.10
git --version
```

If `python` does not exist but `python3` does, see
[Pitfalls](#common-pitfalls) before proceeding.

---

## One-time prep (do this once per clone)

```bash
# 1. Clone (or `git switch feature/v2-implementations` in an existing clone).
git clone https://github.com/arman-tech/spatial-memory-mcp.git ~/code/spatial-memory-mcp
cd ~/code/spatial-memory-mcp
git switch feature/v2-implementations

# 2. Create + activate a venv that Claude Code can inherit.
#    (If you rely on a system-wide python, skip the venv but be aware
#    Claude Code's stdio child process must be able to `import spatial_memory`.)
python -m venv .venv
source .venv/bin/activate

# 3. Editable install with dev extras.
pip install -e ".[dev]"

# 4. Sanity-check the CLI is wired up.
spatial-memory --version
spatial-memory doctor          # config + paths report; non-fatal on warnings
```

> **Optional model warm-up.** The first MCP call downloads
> `Snowflake/snowflake-arctic-embed-m-long` (~580 MB) into the
> Hugging Face cache. If you would rather pay that cost outside of
> Claude Code, run `spatial-memory doctor` once more after any cold
> install — it loads the embedder eagerly and surfaces failures
> immediately.

> **Confirm the plugin is in dev mode.** This branch ships
> `plugin/.mcp.json` already pointing at `python -m spatial_memory`,
> which is the dev launcher. If `spatial-memory plugin-mode status`
> reports `prod`, run `spatial-memory plugin-mode dev` to flip it.

---

## Path A — User-level install (default scope)

User-level installs live under `~/.claude/` and apply to every project
you open with Claude Code. This is the right scope for ongoing
dogfooding of the plugin while you iterate on it.

### A1. Install

```bash
# Register the local clone as a marketplace. Use an absolute path —
# Claude Code does not resolve `~` or relative paths here.
claude plugin marketplace add /absolute/path/to/spatial-memory-mcp

# Install the plugin from that marketplace at user scope (the default).
claude plugin install spatial-memory@spatial-memory-marketplace
```

Claude Code copies/symlinks the plugin into
`~/.claude/plugins/spatial-memory-marketplace/spatial-memory/` and
registers the four hooks in
`~/.claude/plugins/.../hooks.json` for every session.

### A2. Verify

Inside any Claude Code session:

```
/plugin
```

You should see `spatial-memory@spatial-memory-marketplace` listed as
enabled.

```
/mcp
```

You should see `spatial-memory` listed and `connected`. Expand it and
confirm 25 tools (remember, recall, hybrid_recall, journey, wander,
etc.) are exposed.

Then in the chat, type any prompt — the `UserPromptSubmit` hook should
inject an ambient-recall context block (when at least one memory
exists). To confirm a memory landed:

```
Use the remember tool to store: "Ambient recall smoke test on $(date)" in namespace "smoketest".
```

Then send a fresh prompt that semantically overlaps ("ambient recall
smoke test"); the dispatcher should surface the just-stored memory in
the injected block. Inspect the metrics journal directly to confirm
hooks fired:

```bash
ls ~/code/spatial-memory-mcp/.spatial-memory/metrics/
tail -n 5 ~/code/spatial-memory-mcp/.spatial-memory/metrics/*.jsonl
```

(If you set `SPATIAL_MEMORY_MEMORY_PATH` somewhere else, look there
instead — see [Pitfalls](#common-pitfalls).)

### A3. Uninstall

```bash
claude plugin uninstall spatial-memory@spatial-memory-marketplace
claude plugin marketplace remove spatial-memory-marketplace

# Optional clean-up:
rm -rf ~/.spatial-memory                   # only if the server wrote to $HOME
rm -rf /path/where/you/launched/claude/.spatial-memory  # CWD-relative default
pip uninstall spatial-memory-mcp           # only if you want to drop the editable install
```

Restart Claude Code (or run `/plugin` and confirm the entry is gone).
Hooks are unregistered automatically when the plugin is uninstalled.

---

## Path B — Project-level install (`--scope project`)

Project-level installs live under `<project>/.claude/plugins/` and
apply only when Claude Code is launched inside that project tree. This
is the right scope when you want a *single* repo to opt into the
in-development plugin without polluting other projects.

### B1. Install

```bash
cd /path/to/the/project/you/want/to/test/in   # e.g. cd ~/code/some-other-repo

# Same marketplace as Path A — marketplaces themselves are user-scoped,
# so you only need to add it once per machine.
claude plugin marketplace add /absolute/path/to/spatial-memory-mcp   # skip if already added

# Install at project scope.
claude plugin install spatial-memory@spatial-memory-marketplace --scope project
```

The plugin manifest is now committed under
`<project>/.claude/plugins/`. **Commit or .gitignore that directory
intentionally** depending on whether you want collaborators to inherit
the dev plugin install.

### B2. Verify

Same as [A2](#a2-verify), but launch Claude Code from inside the
project root. `/plugin` should list `spatial-memory` only when the
session is rooted in that project.

To confirm scope:

```bash
ls .claude/plugins/      # project-scope artifacts
ls ~/.claude/plugins/    # should NOT contain spatial-memory if you only did Path B
```

### B3. Uninstall

```bash
cd /path/to/the/project
claude plugin uninstall spatial-memory@spatial-memory-marketplace --scope project

# Marketplace registration is user-scoped; only remove if no other project still uses it.
claude plugin marketplace remove spatial-memory-marketplace

# Optional clean-up:
rm -rf .spatial-memory                     # local DB if memory_path was project-relative
rm -rf .claude/plugins                     # if no other project-scope plugins remain
```

---

## Optional alternative — manual `.claude/settings.json` (no marketplace)

If you want to bypass the plugin system entirely (for example to test
the `setup-hooks` CLI itself, or to install on a Claude Code build that
predates plugins), use the bundled installer:

```bash
cd ~/code/spatial-memory-mcp

# Print what would be written, scoped to the user's settings.json.
spatial-memory setup-hooks --client claude-code --print-only

# Apply it. Add --no-session-start on Windows (see Pitfalls).
spatial-memory setup-hooks --client claude-code

# This writes hook entries into ~/.claude/settings.json that point at
# `spatial-memory hook <event> --client claude-code`. The MCP server
# itself is not registered this way — you still need to add it via
# `~/.claude/mcp.json` or `claude mcp add` if you want the tool surface.
```

To uninstall the manual path, edit `~/.claude/settings.json` and
remove the `hooks` block the installer added (or revert the file from
your VCS / backup). Then restart Claude Code.

---

## Smoke tests (apply to every path)

Run these once after installing, to prove every layer is wired up:

1. **MCP connection.** `/mcp` shows `spatial-memory` as `connected`
   with 25 tools.
2. **Auto-injected behavioural instructions.** Ask Claude Code "What
   does the spatial-memory MCP server tell you to do?" — it should
   paraphrase the 3-tier auto-save instructions emitted by
   `_get_server_instructions()` in `spatial_memory/server.py`.
3. **Ambient recall.** Store a distinctive memory via `remember`, then
   send a new prompt with an overlapping phrase. The injected context
   block should reference the stored memory. Disable with
   `SPATIAL_MEMORY_AMBIENT_RECALL=off` and confirm the block goes away.
4. **PostToolUse hook.** Use any tool, then check
   `<memory_path>/metrics/*.jsonl` — there should be a fresh
   `post_tool_use` event line.
5. **PreCompact hook.** Run `/compact` (or let auto-compaction fire)
   and confirm a `pre_compact` event lands in the same journal.
6. **Stop hook.** End the Claude Code session normally; the next time
   you open the journal, the final lines should include a `stop`
   event.

---

## Iteration tips

- **Editing `plugin/hooks/*.py`.** Changes propagate without
  reinstalling the plugin — Claude Code resolves `${CLAUDE_PLUGIN_ROOT}`
  at hook-fire time. You may need to start a new Claude Code session
  for stdout-buffered output to refresh, but no reinstall is needed.
- **Editing `spatial_memory/*.py`.** Because you did `pip install -e`,
  edits are live for the *next* MCP child process. Restart Claude Code
  or run `/mcp` → `restart spatial-memory` to pick them up.
- **Editing `plugin/.mcp.json` or `plugin/hooks/hooks.json`.** Claude
  Code caches plugin manifests at install time. Run `claude plugin
  uninstall ... && claude plugin install ...` after any manifest edit.
  The `spatial-memory plugin-mode dev|prod` CLI does both: it edits
  the JSON *and* prints a reinstall hint.
- **Cold-cache pause on first use.** First `recall` after install can
  take 10–30 s while sentence-transformers downloads weights. Expected,
  one-time per HF cache.

---

## Common pitfalls

- **`python` vs `python3`.** Both `plugin/.mcp.json` and
  `plugin/hooks/hooks.json` invoke the bare `python` command. On
  systems where only `python3` exists, either symlink
  `~/.local/bin/python -> $(which python3)` or run
  `spatial-memory plugin-mode prod` to switch the MCP server to the
  `uvx` launcher (which does not need `python` on `PATH`). Hooks are
  trickier — they invoke `python` directly and have no `uvx`
  alternative; symlinking is the supported workaround.
- **CWD-relative `memory_path`.** With no `SPATIAL_MEMORY_MEMORY_PATH`,
  the database lands wherever Claude Code launched the stdio child
  from — usually the directory where you ran `claude`, *not* the repo
  root. Set `SPATIAL_MEMORY_MEMORY_PATH=~/.spatial-memory` (or some
  absolute path) in your shell profile if you want a stable single
  location.
- **Editable install in the wrong env.** Claude Code spawns the MCP
  child using whatever `python` is first on `PATH` for the user
  session. If you `pip install -e` inside a venv but Claude Code
  launches from a shell that does not have that venv activated, the
  child will fail with `ModuleNotFoundError: spatial_memory`. Either
  put the venv's `bin/` first on the system `PATH`, install editable
  into the system Python, or use a venv-activating wrapper.
- **Windows + `SessionStart` plugin hook.** Claude Code v2.1.37 freezes
  terminal input when a *plugin* registers a `SessionStart` hook on
  Windows. The plugin's `hooks.json` deliberately omits
  `SessionStart` for that reason. If you want session-start ingestion
  on Windows, install hooks via
  `spatial-memory setup-hooks --client claude-code` (the
  `.claude/settings.json` path), which writes the hook outside of the
  plugin manifest. Use `--no-session-start` if you do not want it at
  all.
- **`plugin/.mcp.json` mode.** This branch ships dev mode. Check with
  `spatial-memory plugin-mode status` if surprised; flip with
  `spatial-memory plugin-mode dev` or `... prod`. Reinstall the plugin
  after every flip — Claude Code caches the manifest at install time.
- **Cursor users.** The plugin marketplace is a Claude Code feature.
  Cursor uses a different config surface (`.cursor/mcp.json` +
  `.cursor/hooks.json`). Use `spatial-memory init --client cursor`
  inside the target project, *not* the steps above. The
  `UserPromptSubmit` ambient-recall feature is Claude-Code-only in
  v2.0; Cursor sessions do not get prompt-time recall injection.
- **Two marketplace registrations of the same path.** Claude Code
  rejects duplicate marketplace `name`s. If you previously added the
  PyPI/remote install (`claude plugin marketplace add
  arman-tech/spatial-memory-mcp`), remove it first before pointing the
  same name at a local clone.

---

## Quick reference

| Action                          | User scope                                                                    | Project scope                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Add local marketplace           | `claude plugin marketplace add /abs/path/to/spatial-memory-mcp`               | Same — marketplaces are user-scoped                                                                    |
| Install plugin                  | `claude plugin install spatial-memory@spatial-memory-marketplace`             | `claude plugin install spatial-memory@spatial-memory-marketplace --scope project`                      |
| Where it lives                  | `~/.claude/plugins/...`                                                       | `<project>/.claude/plugins/...`                                                                        |
| Uninstall                       | `claude plugin uninstall spatial-memory@spatial-memory-marketplace`           | `claude plugin uninstall spatial-memory@spatial-memory-marketplace --scope project`                    |
| Remove marketplace registration | `claude plugin marketplace remove spatial-memory-marketplace`                 | Same                                                                                                   |

For the architectural background behind the v2.0 hook surface, see
`internal-docs/v2-architectural-decisions.md` (D3 ambient recall, D7
metrics journal). For env-var reference, see `.env.example` and
`docs/CONFIGURATION.md`.
