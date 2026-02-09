"""Extract text content from transcript entries.

Simple extraction and truncation module.  Takes a list of
``TranscriptEntry`` objects and returns their text content, capped at
configurable size limits.

**STDLIB-ONLY**: No imports beyond typing.
"""

from __future__ import annotations

from spatial_memory.hooks.models import TranscriptEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ENTRY_TEXT: int = 5_000
"""Maximum characters per individual entry text."""

MAX_COMBINED_TEXT: int = 10_000
"""Maximum total characters across all extracted texts."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_assistant_text(entries: list[TranscriptEntry]) -> list[str]:
    """Extract text content from assistant entries.

    Returns a list of text strings, each truncated to ``MAX_ENTRY_TEXT``.
    Only includes entries with ``role='assistant'`` and non-empty text.
    The total output is capped at ``MAX_COMBINED_TEXT`` characters.

    Args:
        entries: Transcript entries to extract text from.

    Returns:
        List of non-empty text strings.
    """
    if not entries:
        return []

    texts: list[str] = []
    total_chars = 0

    for entry in entries:
        if entry.role != "assistant":
            continue

        text = entry.text.strip()
        if not text:
            continue

        # Truncate individual entry
        if len(text) > MAX_ENTRY_TEXT:
            text = text[:MAX_ENTRY_TEXT]

        # Check combined budget
        if total_chars + len(text) > MAX_COMBINED_TEXT:
            remaining = MAX_COMBINED_TEXT - total_chars
            if remaining > 0:
                texts.append(text[:remaining])
            break

        texts.append(text)
        total_chars += len(text)

    return texts
