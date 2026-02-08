"""Content hashing utilities for deduplication.

Provides SHA-256 hashing of normalized content for exact-match deduplication.
The normalization must match the logic used in db_migrations.py backfill.
"""

from __future__ import annotations

import hashlib


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of normalized content.

    Normalization: strip whitespace + lowercase, matching the migration
    backfill logic in db_migrations.py.

    Args:
        content: Raw content string.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    normalized = content.strip().lower() if content else ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
