"""Shared utilities for hook script entrypoints.

Provides common functions used by ``post_tool_use.py``, ``pre_compact.py``,
and ``stop.py``.  Each entrypoint loads this module via its own minimal
``_load_hook_module("hook_helpers")`` bootstrap.

**STDLIB-ONLY**: Only ``json``, ``os``, ``re``, ``sys`` imports allowed.
"""

from __future__ import annotations

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
"""Safe characters for session IDs used as filenames."""

_WINDOWS_DEVICE_RE = re.compile(r"^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\..+)?$", re.IGNORECASE)
"""Windows reserved device names that must not be used as filenames."""


def sanitize_session_id(session_id: str) -> str:
    """Sanitize a session ID for safe use as a filename component.

    Returns the session ID unchanged if it passes validation, or an
    empty string if it contains unsafe characters.

    Args:
        session_id: Raw session ID from stdin JSON.

    Returns:
        The validated session ID, or ``""`` if invalid.
    """
    if not session_id:
        return ""

    # Length cap: prevent absurdly long filenames
    if len(session_id) > 128:
        return ""

    # Must contain only safe characters
    if not _SESSION_ID_RE.match(session_id):
        return ""

    # Reject Windows device names
    if _WINDOWS_DEVICE_RE.match(session_id):
        return ""

    return session_id


def validate_transcript_path(path: str) -> str:
    """Validate a transcript path for safe use.

    Rejects paths containing traversal sequences (``..``) or relative
    paths.  The transcript path comes from the Claude Code host process
    and should always be absolute.

    Args:
        path: Raw transcript path from stdin JSON.

    Returns:
        The validated path, or ``""`` if invalid.
    """
    if not path:
        return ""

    # Reject path traversal sequences
    if ".." in path:
        return ""

    # Reject relative paths (must be absolute)
    # On Windows, absolute paths start with drive letter or UNC
    # On POSIX, absolute paths start with /
    if not os.path.isabs(path):
        return ""

    return path


# ---------------------------------------------------------------------------
# stdin / stdout helpers
# ---------------------------------------------------------------------------


def read_stdin() -> dict[str, object]:
    """Read and parse JSON from stdin.

    Returns:
        Parsed dict, or empty dict on any error.
    """
    try:
        raw = sys.stdin.read(524_288)  # 512KB limit
        if not raw or not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def write_stdout_response() -> None:
    """Write the standard hook response to stdout and flush.

    Output: ``{"continue": true, "suppressOutput": true}``
    """
    try:
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass


def get_project_root(cwd: str = "") -> str:
    """Resolve the project root from environment or explicit CWD.

    Resolution order:
    1. ``$CLAUDE_PROJECT_DIR`` (set by some CI / wrapper environments)
    2. *cwd* parameter (from hook stdin ``cwd`` field)
    3. Empty string (caller falls back to CWD-relative paths)

    Args:
        cwd: Explicit working directory from hook stdin data.
    """
    return os.environ.get("CLAUDE_PROJECT_DIR", "") or cwd
