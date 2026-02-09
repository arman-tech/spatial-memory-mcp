"""Dedup against PostToolUse queue using hash comparison.

Reads content hashes from recent queue files in ``new/`` to detect
whether transcript-extracted text was already captured by the
PostToolUse hook.

**STDLIB-ONLY**: Only ``hashlib``, ``json``, ``pathlib`` imports allowed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_queued_hashes(queue_dir: Path, max_age_files: int = 50) -> set[str]:
    """Read content hashes from recent queue files in ``new/``.

    Scans at most *max_age_files* (newest first by filename, which is
    time-sortable) to bound I/O.

    Args:
        queue_dir: Path to the queue directory (contains ``new/``).
        max_age_files: Maximum number of files to read.

    Returns:
        Set of SHA-256 hex digests of normalized content values.
    """
    new_dir = queue_dir / "new"
    if not new_dir.is_dir():
        return set()

    hashes: set[str] = set()

    try:
        # Sort by filename descending (newest first — filenames are time-sortable)
        files = sorted(new_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
    except OSError:
        return set()

    for file_path in files[:max_age_files]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            content = data.get("content", "")
            if isinstance(content, str) and content.strip():
                h = _content_hash(content)
                hashes.add(h)
        except (json.JSONDecodeError, OSError, ValueError):
            continue

    return hashes


def is_duplicate(content: str, queued_hashes: set[str]) -> bool:
    """Check if content was already queued by PostToolUse.

    Compares the SHA-256 of the normalized (stripped + lowercased) content
    against the set of hashes from recent queue files.  This catches exact
    duplicates modulo whitespace and case, but not semantic near-duplicates.

    Args:
        content: Text to check for duplication.
        queued_hashes: Set of hashes from ``get_queued_hashes()``.

    Returns:
        ``True`` if the normalized content matches an existing queue entry.
    """
    if not content or not queued_hashes:
        return False

    return _content_hash(content) in queued_hashes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    """Compute SHA-256 hex digest of normalized content."""
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
