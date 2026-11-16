# Spatial Memory plugin for Claude Code

Spatial Memory 3.0 gives Claude Code a local memory. It captures decisions, solutions, errors and patterns from your sessions, stores them per project on your machine, and recalls them by meaning. Everything runs in one local binary, `spatial-memory`; nothing is sent to a network service.

## Installing

```
/plugin install spatial-memory --marketplace arman-tech/spatial-memory-mcp
```

This works in one step on Claude Code 2.1.275 or later. On an earlier version, add the marketplace first with `/plugin marketplace add arman-tech/spatial-memory-mcp`, then run `/plugin install spatial-memory@spatial-memory-marketplace`.

## What the plugin registers

- **MCP server** `spatial-memory`: runs `spatial-memory mcp` and offers the thirteen tools (`remember`, `recall`, `related`, `revise`, `forget`, `link`, `regions`, `journey`, `consolidate`, `share`, `sync`, `status`, `reinforce`).
- **Hooks**, all in exec form:
  - SessionStart (`startup` and `resume`), 5 seconds: a short note that project memory is available.
  - UserPromptSubmit, 5 seconds: ambient recall adds relevant memories to the prompt.
  - PostToolUse, 10 seconds, asynchronous: capture.
  - PreCompact and Stop, 15 seconds each: capture.

## Requirements

The `spatial-memory` binary must be on `PATH` for the application that starts Claude Code. Check it with `spatial-memory doctor`, which also reports whether the binary's version matches this plugin's.

Do not use the plugin and `spatial-memory setup` together: they register the same hooks and server. `setup` refuses to run while the plugin is enabled, and `doctor` fails if both are registered.

## Options

- **profile**: the profile to use. Leave it empty to use `SPATIAL_MEMORY_PROFILE`, a folder rule from the configuration file, or the default profile.
- **ambient_recall**: `on` or `off`. With `off`, the prompt hook prints nothing and contacts nothing.
- **binary_path**: not applied in 3.0. Claude Code substitutes plugin options into a hook's arguments but not into its command, so the plugin always runs `spatial-memory` from `PATH`. If `PATH` differs between your terminal and the application that starts Claude Code (common for desktop applications on macOS), disable the plugin and run `spatial-memory setup`, which registers the absolute path of the binary.

## Removing

Disable or uninstall the plugin with `/plugin`. Your memories stay in the profile folder until you delete them.
