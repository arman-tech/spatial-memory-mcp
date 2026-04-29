"""Extract text content from transcript entries.

Simple extraction and truncation module.  Takes a list of
``TranscriptEntry`` objects and returns their text content, capped by a
single 200_000-character safety cap.

Per spec §8.2 (D6) the v1 per-entry / combined caps are gone: token-based
capture sizing now lives in the upstream capture pipeline. The single
``_SAFETY_CAP`` is applied at the per-entry stage (effectively a no-op for
entries below 200K) and again at the combined stage where the last entry is
tail-dropped with a ``...`` suffix once the cumulative length would
overflow.

**STDLIB-ONLY**: No imports beyond typing.
"""

from __future__ import annotations

from spatial_memory.hooks.models import TranscriptEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAFETY_CAP = 200_000
"""Single tail-drop safety cap on transcript text (US-019, spec §8.2)."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_assistant_text(entries: list[TranscriptEntry]) -> list[str]:
    """Extract text content from assistant entries.

    Returns a list of text strings. Per-entry length is bounded by
    ``_SAFETY_CAP`` (effectively a no-op for typical conversational turns).
    The combined length across all returned strings is bounded by the same
    cap; the last appended entry is tail-dropped with a ``...`` suffix when
    truncation fires.

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

        # Per-entry cap (no-op for entries below _SAFETY_CAP)
        if len(text) > _SAFETY_CAP:
            text = text[:_SAFETY_CAP] + "..."

        # Combined-output cap with tail-drop suffix
        if total_chars + len(text) > _SAFETY_CAP:
            remaining = _SAFETY_CAP - total_chars
            if remaining > 0:
                texts.append(text[:remaining] + "...")
            break

        texts.append(text)
        total_chars += len(text)

    return texts
