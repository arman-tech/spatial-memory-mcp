# Client Setup Guide

Installation and configuration for spatial-memory-mcp across AI coding clients.

## Quick Reference

| Client | Tier | Hooks? | Config File | Setup Command |
|--------|------|--------|-------------|---------------|
| **Claude Code** | 1 (full) | Native | `.claude/settings.json` | Plugin install or `setup-hooks --client claude-code` |
| **Cursor** | 2 (adapter) | Via adapter | `.cursor/hooks.json` | `setup-hooks --client cursor` |
| **Windsurf** | 3 (MCP only) | No | `~/.codeium/windsurf/mcp_config.json` | `setup-hooks --client windsurf` |
| **Antigravity** | 3 (MCP only) | No | `~/.gemini/antigravity/mcp_config.json` | `setup-hooks --client antigravity` |
| **VS Code Copilot** | 3 (MCP only) | No | `.vscode/mcp.json` | `setup-hooks --client vscode-copilot` |

### What the tiers mean

- **Tier 1**: Full native hook support. Auto-captures decisions, solutions, and errors in the background. Recall nudge at session start.
- **Tier 2**: Hooks via adapter script. Same auto-capture features as Tier 1, translated through a compatibility layer.
- **Tier 3**: MCP tools only. No auto-capture hooks. Uses rules/instructions files to guide the LLM toward proactive memory use.

---

## Prerequisites

- Python 3.10+
- `pip install spatial-memory-mcp`

Verify:

```bash
python -m spatial_memory --version
```

---

## Claude Code (Tier 1)

Claude Code has full native hook support. Two installation options:

### Option A: Plugin Install (Zero-Config)

```
/plugin marketplace add arman-tech/spatial-memory-mcp
/plugin install spatial-memory@spatial-memory-marketplace
```

This installs hooks and the MCP server automatically. No manual configuration needed.

### Option B: Manual CLI Install

Generate the configuration:

```bash
python -m spatial_memory setup-hooks --client claude-code --json
```

Then apply the output:

**1. Hooks** — Add to `.claude/settings.json` under the `hooks` key:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "\"<python>\" \"<hooks_dir>/session_start.py\"",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"<python>\" \"<hooks_dir>/post_tool_use.py\"",
            "timeout": 10,
            "async": true
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"<python>\" \"<hooks_dir>/pre_compact.py\"",
            "timeout": 15
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"<python>\" \"<hooks_dir>/stop.py\"",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

> Replace `<python>` and `<hooks_dir>` with the actual paths from `setup-hooks --json` output.

**2. MCP Server** — Add to your MCP configuration (`.mcp.json` or Claude Desktop config):

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "<python>",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

### Verification

Start a new session. You should see a recall nudge in the context:

> SPATIAL MEMORY: Call `recall` with a brief summary of the user's apparent task/context to load relevant memories from previous sessions.

Run a few tool calls and check that queue files appear in `.spatial-memory/queue/`.

---

## Cursor (Tier 2)

Cursor supports hooks via a compatibility adapter that translates between Cursor's native format (camelCase events, `conversation_id`) and Claude Code's format (PascalCase events, `session_id`).

### Installation

Generate the configuration:

```bash
python -m spatial_memory setup-hooks --client cursor --json
```

**1. Hooks** — Save to `.cursor/hooks.json`:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": "\"<python>\" \"<hooks_dir>/cursor_adapter.py\" session_start",
        "timeout": 5,
        "matcher": "startup|resume"
      }
    ],
    "postToolUse": [
      {
        "command": "\"<python>\" \"<hooks_dir>/cursor_adapter.py\" post_tool_use",
        "timeout": 10
      }
    ],
    "preCompact": [
      {
        "command": "\"<python>\" \"<hooks_dir>/cursor_adapter.py\" pre_compact",
        "timeout": 15
      }
    ],
    "stop": [
      {
        "command": "\"<python>\" \"<hooks_dir>/cursor_adapter.py\" stop",
        "timeout": 15,
        "loop_limit": 1
      }
    ]
  }
}
```

> Replace `<python>` and `<hooks_dir>` with the actual paths from `setup-hooks --json` output.

**2. MCP Server** — Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "<python>",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

### How the adapter works

All hook commands go through `cursor_adapter.py <hook_name>`, which:

1. Reads Cursor's stdin JSON (camelCase keys, `conversation_id`)
2. Translates to Claude Code format (PascalCase keys, `session_id`)
3. Delegates to the actual hook script via subprocess
4. Translates the output back to Cursor format

This is invisible to the user. The adapter is fail-open: any error results in silent pass-through.

---

## Windsurf (Tier 3)

Windsurf does not support hooks. Cognitive offloading layers 1-2 (server-side gating and session extraction) work fully via MCP. Layer 3 (auto-capture) is unavailable; a rules file guides the LLM toward proactive memory use.

### Installation

Generate the configuration:

```bash
python -m spatial_memory setup-hooks --client windsurf --json
```

**1. MCP Server** — Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "<python>",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

**2. Rules** — Create `.windsurf/rules/spatial-memory.md`:

```markdown
At the start of each conversation, call `recall` with a brief
summary of the user's task to load relevant memories. When you
make decisions, discover solutions, or fix bugs, call `remember`
to save them for future sessions.
```

---

## Antigravity (Tier 3)

Antigravity (Gemini-based) does not support hooks. Same as Windsurf: layers 1-2 work via MCP, layer 3 needs a rules file.

### Installation

Generate the configuration:

```bash
python -m spatial_memory setup-hooks --client antigravity --json
```

**1. MCP Server** — Add to `~/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "<python>",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

**2. Rules** — Add to `~/.gemini/GEMINI.md` or `.agent/rules/spatial-memory.md`:

```markdown
At the start of each conversation, call `recall` with a brief
summary of the user's task to load relevant memories. When you
make decisions, discover solutions, or fix bugs, call `remember`
to save them for future sessions.
```

### Aliases

The setup tool accepts `gemini` as an alias for `antigravity`:

```bash
python -m spatial_memory setup-hooks --client gemini --json
```

---

## VS Code Copilot (Tier 3)

VS Code Copilot does not support hooks. Uses the `servers` key (not `mcpServers`) in its MCP config.

### Installation

Generate the configuration:

```bash
python -m spatial_memory setup-hooks --client vscode-copilot --json
```

**1. MCP Server** — Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "spatial-memory": {
      "type": "stdio",
      "command": "<python>",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

> Note: VS Code Copilot uses `servers` with a `type` field, not `mcpServers`.

**2. Instructions** — Create `.github/copilot-instructions.md`:

```markdown
At the start of each conversation, call `recall` with a brief
summary of the user's task to load relevant memories. When you
make decisions, discover solutions, or fix bugs, call `remember`
to save them for future sessions.
```

### Aliases

The setup tool accepts `copilot`, `vscode`, or `vs-code`:

```bash
python -m spatial_memory setup-hooks --client copilot --json
```

---

## Feature Comparison

| Feature | Tier 1 (Claude Code) | Tier 2 (Cursor) | Tier 3 (Windsurf, Antigravity, VS Code) |
|---------|---------------------|-----------------|----------------------------------------|
| MCP tools (remember, recall, etc.) | Yes | Yes | Yes |
| Server-side gating (Layer 1) | Yes | Yes | Yes |
| Session extraction (Layer 2) | Yes | Yes | Yes |
| Auto-capture hooks (Layer 3) | Native | Via adapter | No |
| SessionStart recall nudge | Yes | Yes | No |
| PostToolUse background capture | Yes (async) | Yes | No |
| PreCompact transcript scan | Yes | Yes | No |
| Stop session-end capture | Yes | Yes | No |
| Rules/instructions file | Not needed | Not needed | Required for best results |
| Plugin install (zero-config) | Yes | No | No |

---

## Using the `setup_hooks` MCP Tool

If spatial-memory-mcp is already running as an MCP server, you can generate configuration from within a session:

```
Call the setup_hooks tool with client="cursor"
```

The tool returns the same JSON as the CLI command, plus human-readable instructions.

---

## Troubleshooting

### Hooks not firing (Claude Code)

1. Verify hooks are in `.claude/settings.json` (not `.claude/settings.local.json`)
2. Check the Python path is valid: run the `command` value manually in a terminal
3. Look for errors in `~/.claude/logs/`

### Hooks not firing (Cursor)

1. Verify `.cursor/hooks.json` has the correct `version: 1` wrapper
2. Confirm `cursor_adapter.py` exists at the path shown in the hook command
3. All hooks are fail-open: errors are silently swallowed. Run the adapter manually to debug:
   ```bash
   echo '{"conversation_id":"test"}' | python <hooks_dir>/cursor_adapter.py session_start
   ```

### MCP server not starting

1. Verify the package is installed: `python -m spatial_memory --version`
2. First run downloads the embedding model (~80MB). Allow time for completion.
3. Check `SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED` is set to `"true"` (string, not boolean)

### Queue files not appearing

1. Check that `.spatial-memory/queue/` exists in your project root
2. Verify `CLAUDE_PROJECT_DIR` is set (Claude Code sets this automatically)
3. Run a tool that produces content (e.g., Edit, Write, Bash) — read-only tools like Read and Grep are intentionally skipped

### Tier 3 clients not remembering

Tier 3 clients rely on the LLM reading the rules file and choosing to call `remember`. If memories aren't being saved:

1. Verify the rules file is in the correct location for your client
2. Add more explicit instructions in the rules file
3. Consider manually calling `remember` for important information

---

## Client Aliases

| Alias | Resolves To |
|-------|-------------|
| `claude` | `claude-code` |
| `claudecode` | `claude-code` |
| `gemini` | `antigravity` |
| `copilot` | `vscode-copilot` |
| `vscode` | `vscode-copilot` |
| `vs-code` | `vscode-copilot` |

---

## Next Steps

- [HOOKS_ARCHITECTURE.md](HOOKS_ARCHITECTURE.md) — How the hook system works internally
- [COGNITIVE_OFFLOADING_DESIGN.md](COGNITIVE_OFFLOADING_DESIGN.md) — Full design document
- [CONFIGURATION.md](CONFIGURATION.md) — All server settings
- [API.md](API.md) — Complete MCP tool reference
