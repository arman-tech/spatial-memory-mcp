"""Auto-configure Cursor for spatial-memory-mcp.

Creates MCP config, hooks config, and rules files for Cursor.
Claude Code uses plugin marketplace install and does not need ``init``.

Usage::

    spatial-memory init --client cursor           # project scope
    spatial-memory init --client cursor --global  # global scope
    spatial-memory init --client cursor --force   # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project name validation
# ---------------------------------------------------------------------------

_PROJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")


def _validate_project_name(name: str) -> str:
    """Validate and return a project name, or raise ValueError.

    Rules:
    - 1-100 characters
    - Alphanumeric, dots, hyphens, underscores only
    - Must start with alphanumeric character
    """
    if not name:
        raise ValueError(
            "Could not determine project name from current directory.\n"
            "Run this command from a named project directory, or use --global."
        )
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name: {name!r}. "
            "Must be 1-100 chars, alphanumeric/dots/hyphens/underscores, "
            "starting with alphanumeric."
        )
    return name


# ---------------------------------------------------------------------------
# Cursor path resolution
# ---------------------------------------------------------------------------


def _get_cursor_paths(global_scope: bool) -> dict[str, Path]:
    """Resolve Cursor config file paths.

    Args:
        global_scope: If True, use ``~/.cursor/`` instead of ``.cursor/``.

    Returns:
        Dict with ``mcp_json``, ``hooks_json``, and ``rules_file`` paths.
    """
    if global_scope:
        base = Path.home() / ".cursor"
    else:
        base = Path(".cursor")

    return {
        "mcp_json": base / "mcp.json",
        "hooks_json": base / "hooks.json",
        "rules_file": base / "rules" / "spatial-memory.mdc",
    }


# ---------------------------------------------------------------------------
# JSON merge helper
# ---------------------------------------------------------------------------


def _merge_json(existing_path: Path, new_data: dict[str, Any], key: str) -> dict[str, Any]:
    """Read existing JSON, merge new data under key, return merged dict.

    If the file doesn't exist, returns ``{key: new_data[key]}``.
    If the file has invalid JSON, raises ``ValueError``.
    """
    if not existing_path.exists():
        return new_data

    raw = existing_path.read_text(encoding="utf-8").strip()
    if not raw:
        return new_data

    try:
        existing = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {existing_path}: {e}\n"
            f"Fix the file manually or use --force to overwrite."
        ) from e

    if not isinstance(existing, dict):
        raise ValueError(f"Expected JSON object in {existing_path}, got {type(existing).__name__}")

    # Merge: new data takes precedence for keys under the merge key
    if key in existing and isinstance(existing[key], dict) and key in new_data:
        existing[key].update(new_data[key])
    else:
        existing.update(new_data)

    return existing


# ---------------------------------------------------------------------------
# Rules file content
# ---------------------------------------------------------------------------

_RULES_CONTENT = """\
---
description: Spatial Memory MCP - automatic memory recall
globs:
alwaysApply: true
---

# Spatial Memory

At the start of each conversation, call the `recall` MCP tool with a brief
summary of the user's apparent task or context to load relevant memories
from previous sessions.

When you encounter decisions, error solutions, architecture choices, or
important patterns, the hook system will automatically capture them.
"""


def _write_rules_file(path: Path) -> None:
    """Write the Cursor rules .mdc file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_RULES_CONTENT, encoding="utf-8")


# ---------------------------------------------------------------------------
# Cursor init
# ---------------------------------------------------------------------------


def _init_cursor(
    paths: dict[str, Path],
    force: bool,
    global_scope: bool = False,
    project: str | None = None,
    mode: str = "prod",
) -> int:
    """Write Cursor config files.

    Args:
        paths: File paths from ``_get_cursor_paths()``.
        force: Overwrite existing files without merging.
        global_scope: If True, skip project scoping (global memories).
        project: Explicit project name. If None, derived from cwd.
        mode: ``"prod"`` for uvx (default), ``"dev"`` for local python.

    Returns:
        Exit code (0 success, 1 error).
    """
    from spatial_memory.tools.setup_hooks import generate_hook_config

    if global_scope:
        resolved_project = ""
    elif project:
        try:
            resolved_project = _validate_project_name(project)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    else:
        try:
            resolved_project = _validate_project_name(Path.cwd().name)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1

    config = generate_hook_config(client="cursor", project=resolved_project, mode=mode)

    # 1. MCP config
    mcp_path = paths["mcp_json"]
    mcp_data = config["mcp_config"]
    try:
        if force or not mcp_path.exists():
            mcp_path.parent.mkdir(parents=True, exist_ok=True)
            mcp_path.write_text(json.dumps(mcp_data, indent=2) + "\n", encoding="utf-8")
            print(f"  Created {mcp_path}")
        else:
            merged = _merge_json(mcp_path, mcp_data, "mcpServers")
            mcp_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            print(f"  Updated {mcp_path}")
    except ValueError as e:
        print(f"WARNING: {e}")
        return 1
    except OSError as e:
        print(f"ERROR: Could not write {mcp_path}: {e}")
        return 1

    # 2. Hooks config
    hooks_path = paths["hooks_json"]
    hooks_data = config["hooks"]
    try:
        if force or not hooks_path.exists():
            hooks_path.parent.mkdir(parents=True, exist_ok=True)
            hooks_path.write_text(json.dumps(hooks_data, indent=2) + "\n", encoding="utf-8")
            print(f"  Created {hooks_path}")
        else:
            merged = _merge_json(hooks_path, hooks_data, "hooks")
            hooks_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            print(f"  Updated {hooks_path}")
    except ValueError as e:
        print(f"WARNING: {e}")
        return 1
    except OSError as e:
        print(f"ERROR: Could not write {hooks_path}: {e}")
        return 1

    # 3. Rules file
    rules_path = paths["rules_file"]
    try:
        _write_rules_file(rules_path)
        print(f"  Created {rules_path}")
    except OSError as e:
        print(f"ERROR: Could not write {rules_path}: {e}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_init(args: argparse.Namespace) -> int:
    """Run the init command.

    Args:
        args: Parsed CLI args with ``client``, ``global_scope``, ``force``.

    Returns:
        Exit code (0 success, 1 error).
    """
    client: str = args.client

    if client != "cursor":
        print(f"Error: init is only supported for 'cursor'. Got: {client!r}")
        print("Claude Code uses plugin marketplace install instead.")
        return 1

    global_scope: bool = getattr(args, "global_scope", False)
    force: bool = getattr(args, "force", False)
    project: str | None = getattr(args, "project", None)
    mode: str = getattr(args, "mode", "prod")
    scope_label = "global" if global_scope else "project"

    print(f"Spatial Memory - Init ({scope_label} scope, {mode} mode)")
    paths = _get_cursor_paths(global_scope)

    # Check if already configured
    all_exist = all(p.exists() for p in paths.values())
    if all_exist and not force:
        print("Already configured. Use --force to overwrite.")
        return 0

    result = _init_cursor(paths, force, global_scope=global_scope, project=project, mode=mode)

    if result == 0:
        print(f"\nDone. Cursor {scope_label} config created.")
        if not global_scope:
            print("Restart Cursor to activate.")
    return result
