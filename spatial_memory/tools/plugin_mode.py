"""Switch the Claude Code plugin between dev (local source) and prod (PyPI/uvx) modes.

Modifies ``plugin/.mcp.json`` in-place, swapping only ``command`` and ``args``
while preserving ``env`` and other keys.

Usage::

    spatial-memory plugin-mode dev      # local: python -m spatial_memory
    spatial-memory plugin-mode prod     # PyPI:  uvx --from spatial-memory-mcp ...
    spatial-memory plugin-mode status   # show current mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DEV_SERVER: dict[str, Any] = {
    "command": "python",
    "args": ["-m", "spatial_memory"],
}

_PROD_SERVER: dict[str, Any] = {
    "command": "uvx",
    "args": ["--from", "spatial-memory-mcp", "spatial-memory", "serve"],
}

_MCP_JSON = Path(__file__).resolve().parent.parent.parent / "plugin" / ".mcp.json"

_SERVER_KEY = "spatial-memory"


def detect_mode(mcp_data: dict[str, Any]) -> str:
    """Return ``"dev"``, ``"prod"``, or ``"unknown"`` based on the command field."""
    server = mcp_data.get("mcpServers", {}).get(_SERVER_KEY, {})
    cmd = server.get("command", "")
    if cmd == "python":
        return "dev"
    if cmd == "uvx":
        return "prod"
    return "unknown"


def run_plugin_mode(args: argparse.Namespace) -> int:
    """Execute the plugin-mode subcommand.

    Returns:
        Exit code (0 success, 1 error).
    """
    target: str = args.target_mode
    mcp_path: Path = getattr(args, "_mcp_path", _MCP_JSON)

    if not mcp_path.exists():
        print(f"Error: {mcp_path} not found.")
        return 1

    raw = mcp_path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    current = detect_mode(data)

    if target == "status":
        print(f"Current mode: {current}")
        print(f"Config: {mcp_path}")
        return 0

    if current == target:
        print(f"Already in {target} mode. No changes made.")
        return 0

    # Swap command/args, preserve everything else
    server = data.get("mcpServers", {}).get(_SERVER_KEY, {})
    new_server = _DEV_SERVER if target == "dev" else _PROD_SERVER
    server["command"] = new_server["command"]
    server["args"] = new_server["args"]

    mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Switched to {target} mode.")
    print(f"Updated: {mcp_path}")
    print()
    print("NOTE: Claude Code caches the plugin at install time.")
    print("Run the following to apply changes:")
    print("  1. /mcp  (in Claude Code, remove spatial-memory)")
    print("  2. Re-install the plugin from the plugin/ directory")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch plugin dev/prod mode")
    parser.add_argument("target_mode", choices=["dev", "prod", "status"])
    sys.exit(run_plugin_mode(parser.parse_args()))
