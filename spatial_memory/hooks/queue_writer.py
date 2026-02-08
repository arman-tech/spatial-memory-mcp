"""Stdlib-only Maildir atomic queue writer for hook scripts.

Writes queue files compatible with ``core/queue_file.QueueFile.from_json()``.
Uses the Maildir protocol: write to ``tmp/``, then ``os.replace()`` to ``new/``
for atomic delivery.

Cross-reference: ``spatial_memory/core/queue_file.py`` — server-side parser.
Cross-reference: ``spatial_memory/core/queue_constants.py`` — canonical constants.

**STDLIB-ONLY**: Only ``json``, ``os``, ``random``, ``time``, ``pathlib``,
``datetime`` imports allowed.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Embedded constants (from core/queue_constants.py — keep in sync)
# ---------------------------------------------------------------------------

QUEUE_DIR_NAME = "pending-saves"
QUEUE_FILE_VERSION = 1

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_queue_dir() -> Path:
    """Determine the queue directory path.

    Resolution order:
    1. ``$SPATIAL_MEMORY_MEMORY_PATH/pending-saves/``
    2. ``~/.spatial-memory/pending-saves/`` (fallback)

    Returns:
        Path to the queue directory (may not exist yet).
    """
    memory_path = os.environ.get("SPATIAL_MEMORY_MEMORY_PATH", "")
    if memory_path:
        return Path(memory_path) / QUEUE_DIR_NAME
    return Path.home() / ".spatial-memory" / QUEUE_DIR_NAME


def write_queue_file(
    content: str,
    source_hook: str,
    project_root_dir: str = "",
    suggested_namespace: str = "default",
    suggested_tags: list[str] | None = None,
    suggested_importance: float = 0.5,
    signal_tier: int = 1,
    signal_patterns_matched: list[str] | None = None,
    context: dict[str, object] | None = None,
    client: str = "claude-code",
) -> Path:
    """Write a queue file atomically using Maildir protocol.

    Writes to ``tmp/`` first, then uses ``os.replace()`` to move to ``new/``
    for atomic delivery.  Directories are auto-created if needed.

    Args:
        content: The memory content text.
        source_hook: Hook that triggered this save (e.g. "PostToolUse").
        project_root_dir: Absolute path to the project root.
        suggested_namespace: Namespace for the memory.
        suggested_tags: Tags for categorization.
        suggested_importance: Importance score (0.0-1.0).
        signal_tier: Signal tier (1, 2, or 3).
        signal_patterns_matched: Pattern type names that matched.
        context: Additional context dict.
        client: Client identifier.

    Returns:
        Path to the delivered file in ``new/``.
    """
    queue_dir = get_queue_dir()
    tmp_dir = queue_dir / "tmp"
    new_dir = queue_dir / "new"

    # Auto-create directories
    tmp_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique, sortable filename
    filename = _make_filename()

    # Build JSON payload matching QueueFile.from_json() contract
    payload = {
        "version": QUEUE_FILE_VERSION,
        "content": content,
        "source_hook": source_hook,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root_dir": project_root_dir,
        "suggested_namespace": suggested_namespace,
        "suggested_tags": suggested_tags or [],
        "suggested_importance": min(1.0, max(0.0, suggested_importance)),
        "signal_tier": signal_tier,
        "signal_patterns_matched": signal_patterns_matched or [],
        "context": context or {},
        "client": client,
    }

    # Write to tmp/, then atomic move to new/
    tmp_path = tmp_dir / filename
    new_path = new_dir / filename

    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp_path), str(new_path))

    return new_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_filename() -> str:
    """Generate a unique, time-sortable filename.

    Format: ``{time_ns}-{pid}-{random_hex}.json``
    """
    return f"{time.time_ns()}-{os.getpid()}-{random.randbytes(4).hex()}.json"
