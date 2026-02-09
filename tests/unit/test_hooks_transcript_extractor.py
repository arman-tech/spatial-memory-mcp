"""Unit tests for spatial_memory.hooks.transcript_extractor.

Tests cover:
1. Basic extraction — assistant text returned, non-assistant filtered
2. Truncation — individual entry limit, combined text budget
3. Edge cases — empty input, whitespace entries
"""

from __future__ import annotations

import pytest

from spatial_memory.hooks.models import TranscriptEntry
from spatial_memory.hooks.transcript_extractor import (
    MAX_COMBINED_TEXT,
    MAX_ENTRY_TEXT,
    extract_assistant_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(text: str = "Some text.", role: str = "assistant", uuid: str = "u1") -> TranscriptEntry:
    return TranscriptEntry(
        role=role, text=text, timestamp="2026-01-01T00:00:00Z", uuid=uuid, entry_type=role
    )


# =============================================================================
# Basic extraction
# =============================================================================


@pytest.mark.unit
class TestExtractAssistantText:
    """Test basic text extraction from entries."""

    def test_extracts_assistant_text(self) -> None:
        entries = [_entry("Decided to use Redis.")]
        result = extract_assistant_text(entries)
        assert result == ["Decided to use Redis."]

    def test_filters_non_assistant(self) -> None:
        entries = [
            _entry("User message", role="user"),
            _entry("Assistant text", role="assistant"),
        ]
        result = extract_assistant_text(entries)
        assert len(result) == 1
        assert result[0] == "Assistant text"

    def test_multiple_assistant_entries(self) -> None:
        entries = [
            _entry("First.", uuid="u1"),
            _entry("Second.", uuid="u2"),
        ]
        result = extract_assistant_text(entries)
        assert len(result) == 2
        assert result[0] == "First."
        assert result[1] == "Second."

    def test_strips_whitespace(self) -> None:
        entries = [_entry("  Hello world.  ")]
        result = extract_assistant_text(entries)
        assert result == ["Hello world."]


# =============================================================================
# Truncation
# =============================================================================


@pytest.mark.unit
class TestExtractTruncation:
    """Test individual and combined text limits."""

    def test_individual_entry_truncated(self) -> None:
        long_text = "x" * (MAX_ENTRY_TEXT + 1000)
        entries = [_entry(long_text)]
        result = extract_assistant_text(entries)
        assert len(result) == 1
        assert len(result[0]) == MAX_ENTRY_TEXT

    def test_combined_budget_enforced(self) -> None:
        # Create entries that individually fit but exceed combined budget
        entry_size = MAX_COMBINED_TEXT // 3 + 100
        entries = [
            _entry("a" * entry_size, uuid="u1"),
            _entry("b" * entry_size, uuid="u2"),
            _entry("c" * entry_size, uuid="u3"),
            _entry("d" * entry_size, uuid="u4"),
        ]
        result = extract_assistant_text(entries)
        total = sum(len(t) for t in result)
        assert total <= MAX_COMBINED_TEXT

    def test_partial_entry_at_budget_boundary(self) -> None:
        """When the budget runs out mid-entry, truncate that entry."""
        half = MAX_COMBINED_TEXT // 2
        entries = [
            _entry("a" * half, uuid="u1"),
            _entry("b" * (half + 1000), uuid="u2"),
        ]
        result = extract_assistant_text(entries)
        assert len(result) == 2
        total = sum(len(t) for t in result)
        assert total <= MAX_COMBINED_TEXT


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.unit
class TestExtractEdgeCases:
    """Test edge cases."""

    def test_empty_list(self) -> None:
        assert extract_assistant_text([]) == []

    def test_whitespace_only_entries_filtered(self) -> None:
        entries = [_entry("   "), _entry("\n\n"), _entry("Valid.")]
        result = extract_assistant_text(entries)
        assert result == ["Valid."]

    def test_empty_text_entries_filtered(self) -> None:
        entries = [_entry(""), _entry("Has content.")]
        result = extract_assistant_text(entries)
        assert result == ["Has content."]

    def test_all_non_assistant_returns_empty(self) -> None:
        entries = [_entry("text", role="user"), _entry("text", role="system")]
        assert extract_assistant_text(entries) == []
