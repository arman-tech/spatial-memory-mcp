"""SessionStart hook script for Claude Code cognitive offloading.

Outputs ``additionalContext`` nudging Claude to call ``recall`` at session
start, so relevant memories are loaded automatically.

Fires on ``startup`` and ``resume`` events only.  ``clear`` and ``compact``
are ignored (recall nudge only matters at interaction start).

**Minimal**: No importlib bootstrap, no sibling imports — just ``json`` + ``sys``.
No file I/O, no environment reads.

**Fail-open**: All exceptions are caught; the script always exits 0.
Never exit 2 — this hook is non-blocking and should never disrupt the user.

Exit codes:
    0 — Always.  Success, skip, or silent failure.
"""

from __future__ import annotations

import json
import sys

_RECALL_NUDGE = (
    "SPATIAL MEMORY: Call `recall` with a brief summary of the user's "
    "apparent task/context to load relevant memories from previous sessions."
)

_TRIGGER_SOURCES = frozenset({"startup", "resume"})


def main() -> None:
    """Entrypoint. Fail-open: catches all exceptions, always exits 0."""
    try:
        raw = sys.stdin.read(524_288)
        if not raw or not raw.strip():
            return

        data = json.loads(raw)
        if not isinstance(data, dict):
            return

        source = data.get("source", "")
        if source not in _TRIGGER_SOURCES:
            return

        response = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": _RECALL_NUDGE,
            }
        }
        json.dump(response, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()

    except Exception:
        pass
