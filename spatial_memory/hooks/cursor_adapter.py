"""Cursor-to-Claude-Code stdin/stdout adapter for hook scripts.

Translates between Cursor's native hook format (camelCase events,
``conversation_id``, flat structure) and Claude Code's format (PascalCase
events, ``session_id``, nested structure), then delegates to the actual
hook script via subprocess.

**Stdlib-only**: No third-party imports.  This file may be copied into the
plugin directory and must run without ``spatial_memory`` installed in the
import path.

Usage::

    python cursor_adapter.py <hook_name>

Where ``hook_name`` is one of: ``session_start``, ``post_tool_use``,
``pre_compact``, ``stop``.

**Fail-open**: All exceptions are caught; the script always exits 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOOKS_DIR = Path(__file__).resolve().parent

# Map adapter hook names to their target script filenames
_HOOK_SCRIPTS: dict[str, str] = {
    "session_start": "session_start.py",
    "post_tool_use": "post_tool_use.py",
    "pre_compact": "pre_compact.py",
    "stop": "stop.py",
}

# Cursor camelCase -> Claude Code PascalCase event names
_EVENT_MAP: dict[str, str] = {
    "sessionStart": "SessionStart",
    "postToolUse": "PostToolUse",
    "preCompact": "PreCompact",
    "stop": "Stop",
}


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def translate_input(data: dict[str, Any], hook_name: str) -> dict[str, Any]:
    """Translate Cursor stdin JSON to Claude Code format.

    Args:
        data: Parsed Cursor stdin JSON.
        hook_name: The adapter hook name (e.g. ``"session_start"``).

    Returns:
        Translated dict in Claude Code format.
    """
    translated = dict(data)

    # conversation_id -> session_id
    if "conversation_id" in translated:
        translated["session_id"] = translated.pop("conversation_id")

    # Translate event name from camelCase to PascalCase
    event = translated.get("hook_event_name", "")
    if event in _EVENT_MAP:
        translated["hook_event_name"] = _EVENT_MAP[event]

    # SessionStart: synthesize "source" field (Cursor has no startup/resume)
    if hook_name == "session_start" and "source" not in translated:
        translated["source"] = "startup"

    return translated


def translate_session_start_output(data: dict[str, Any]) -> dict[str, Any]:
    """Translate Claude Code SessionStart output to Cursor format.

    Claude Code format::

        {"hookSpecificOutput": {"additionalContext": "..."}}

    Cursor format::

        {"additional_context": "...", "continue": true}

    Args:
        data: Parsed Claude Code stdout JSON.

    Returns:
        Translated dict in Cursor format.
    """
    hook_output = data.get("hookSpecificOutput", {})
    context = hook_output.get("additionalContext", "")

    result: dict[str, Any] = {"continue": True}
    if context:
        result["additional_context"] = context
    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Adapter entrypoint. Fail-open: catches all exceptions, always exits 0."""
    try:
        if len(sys.argv) < 2:
            return

        hook_name = sys.argv[1]
        if hook_name not in _HOOK_SCRIPTS:
            return

        # Read Cursor stdin
        raw = sys.stdin.read(524_288)
        if not raw or not raw.strip():
            return

        data = json.loads(raw)
        if not isinstance(data, dict):
            return

        # Translate input
        translated = translate_input(data, hook_name)

        # Find the target hook script
        script_path = _HOOKS_DIR / _HOOK_SCRIPTS[hook_name]
        if not script_path.exists():
            return

        # Delegate to actual hook script via subprocess
        python = sys.executable or "python"
        result = subprocess.run(
            [python, str(script_path)],
            input=json.dumps(translated),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )

        # For session_start: translate output back to Cursor format
        if hook_name == "session_start" and result.stdout and result.stdout.strip():
            try:
                output_data = json.loads(result.stdout)
                cursor_output = translate_session_start_output(output_data)
                json.dump(cursor_output, sys.stdout)
                sys.stdout.write("\n")
                sys.stdout.flush()
            except (json.JSONDecodeError, KeyError):
                pass

        # For informational hooks (post_tool_use, pre_compact, stop):
        # Cursor doesn't require specific output, pass through if any
        elif result.stdout and result.stdout.strip():
            sys.stdout.write(result.stdout)
            sys.stdout.flush()

    except Exception:
        pass  # Fail-open


if __name__ == "__main__":
    main()
