"""Shared utilities for hook script entrypoints.

Provides common functions used by ``post_tool_use.py``, ``pre_compact.py``,
and ``stop.py``.  Each entrypoint loads this module via its own minimal
``_load_hook_module("hook_helpers")`` bootstrap.

**STDLIB-ONLY**: Only ``json``, ``os``, ``re``, ``sys``, ``time`` imports allowed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
"""Safe characters for session IDs used as filenames."""

_WINDOWS_DEVICE_RE = re.compile(r"^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\..+)?$", re.IGNORECASE)
"""Windows reserved device names that must not be used as filenames."""

_MAX_CWD_LENGTH = 4096
"""Maximum allowed length for a CWD path."""

_MAX_LOG_SIZE = 1_048_576
"""Maximum log file size in bytes before rotation (1MB)."""


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


def validate_cwd(cwd: str) -> str:
    """Validate a working directory path.

    Rejects empty, overly long, traversal-containing, relative, and
    Windows device name paths.  All string ops, no I/O.

    Args:
        cwd: Raw working directory path.

    Returns:
        The validated path, or ``""`` if invalid.
    """
    if not cwd:
        return ""

    if len(cwd) > _MAX_CWD_LENGTH:
        return ""

    if ".." in cwd:
        return ""

    if not os.path.isabs(cwd):
        return ""

    # Reject Windows device names as the final path component
    basename = os.path.basename(cwd)
    if basename and _WINDOWS_DEVICE_RE.match(basename):
        return ""

    return cwd


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------


def _resolve_log_dir(cwd: str = "") -> str:
    """Resolve the directory for hook error logs.

    Resolution order:
    1. ``$SPATIAL_MEMORY_MEMORY_PATH``
    2. ``{cwd}/.spatial-memory/``
    3. Empty string (caller should skip logging)
    """
    memory_path = os.environ.get("SPATIAL_MEMORY_MEMORY_PATH", "")
    if memory_path:
        return memory_path
    if cwd:
        return os.path.join(cwd, ".spatial-memory")
    return ""


def log_hook_error(exc: BaseException, hook_name: str, cwd: str = "") -> None:
    """Append an error entry to ``hook-errors.log``.

    Rotates the log file when it exceeds ``_MAX_LOG_SIZE`` (1MB).
    This function **must never raise** — all exceptions are swallowed.

    Args:
        exc: The exception to log.
        hook_name: Name of the hook that failed.
        cwd: Working directory for log path resolution.
    """
    try:
        log_dir = _resolve_log_dir(cwd)
        if not log_dir:
            return

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "hook-errors.log")

        # Rotate if too large
        try:
            if os.path.exists(log_path) and os.path.getsize(log_path) > _MAX_LOG_SIZE:
                rotated = log_path + ".1"
                if os.path.exists(rotated):
                    os.remove(rotated)
                os.rename(log_path, rotated)
        except OSError:
            pass

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{timestamp}] {hook_name}: {type(exc).__name__}: {exc}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # Logger must never raise


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
