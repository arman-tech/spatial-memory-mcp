"""Unit tests for spatial_memory.hooks.transcript_extractor (US-019 / spec §8.2, D6).

The v2.0 contract replaces the per-entry cap (``MAX_ENTRY_TEXT`` = 5_000) and
the combined cap (``MAX_COMBINED_TEXT`` = 10_000) with a single 200_000-char
safety cap applied at both stages. The per-entry stage is effectively a
no-op for entries under 200K — the combined-output stage is what bounds
the result list with a tail-drop ``...`` suffix on the last entry.

Tests cover:
1. Basic extraction — assistant text returned, non-assistant filtered
2. Single-entry intact below the safety cap (acceptance criterion 5)
3. Combined entries totalling 250K truncate to ~200K with ``...`` suffix (AC 6)
4. Edge cases — empty input, whitespace entries
5. Mirror integrity — plugin/hooks/transcript_extractor.py byte-identical
"""

from __future__ import annotations

import filecmp
import importlib
from pathlib import Path

import pytest

from spatial_memory.hooks.models import TranscriptEntry
from spatial_memory.hooks.transcript_extractor import (
    _SAFETY_CAP,
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
# Module-level invariants (US-019 acceptance criteria 1-4)
# =============================================================================


@pytest.mark.unit
class TestSafetyCapInvariants:
    """The single 200K safety cap replaces both v1 caps."""

    def test_safety_cap_is_200_000(self) -> None:
        assert _SAFETY_CAP == 200_000

    def test_legacy_caps_removed(self) -> None:
        """v1 cap names must not be re-exported from either module."""
        from spatial_memory.hooks import transcript_extractor as te

        for legacy in ("MAX_ENTRY_TEXT", "MAX_COMBINED_TEXT"):
            assert not hasattr(te, legacy), (
                f"Legacy cap {legacy!r} must not be re-exported (US-019, spec §8.2)"
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
# Truncation contract — US-019 acceptance criteria 5 & 6
# =============================================================================


@pytest.mark.unit
class TestSafetyCapTruncation:
    """Test the single 200K safety cap behavior."""

    def test_50k_single_assistant_turn_intact(self) -> None:
        """AC 5: 50K-char single assistant turn is preserved (was 5K cap in v1)."""
        long_text = "x" * 50_000
        entries = [_entry(long_text)]
        result = extract_assistant_text(entries)
        assert len(result) == 1
        assert result[0] == long_text, (
            "50K single assistant turn must NOT be per-entry truncated"
        )

    def test_under_safety_cap_passes_through(self) -> None:
        """An entry of 199K (under safety cap) must pass through intact."""
        entries = [_entry("a" * 199_000)]
        result = extract_assistant_text(entries)
        assert len(result) == 1
        assert len(result[0]) == 199_000
        assert not result[0].endswith("...")

    def test_250k_combined_truncated_to_safety_cap(self) -> None:
        """AC 6: list totalling 250K chars truncates to ~200K with '...' suffix."""
        # 125K + 125K = 250K total; expect first intact + second tail-dropped.
        entries = [
            _entry("a" * 125_000, uuid="u1"),
            _entry("b" * 125_000, uuid="u2"),
        ]
        result = extract_assistant_text(entries)
        total = sum(len(t) for t in result)
        # Allow up to _SAFETY_CAP + len("...") = 200_003
        assert total <= _SAFETY_CAP + 3, (
            f"Combined length {total} exceeds _SAFETY_CAP+3 ({_SAFETY_CAP + 3})"
        )
        # The last appended entry must carry the '...' suffix marking truncation.
        assert result[-1].endswith("..."), (
            "Final entry must carry '...' suffix when combined truncation fires"
        )

    def test_single_entry_above_safety_cap_truncated_with_suffix(self) -> None:
        """A single 250K entry truncates to _SAFETY_CAP + '...'."""
        entries = [_entry("a" * 250_000)]
        result = extract_assistant_text(entries)
        assert len(result) == 1
        assert len(result[0]) == _SAFETY_CAP + 3
        assert result[0].endswith("...")

    def test_combined_below_cap_returns_all_intact(self) -> None:
        """When the combined total fits, every entry passes through unchanged."""
        # 50K + 50K + 50K = 150K (well under 200K)
        entries = [
            _entry("a" * 50_000, uuid="u1"),
            _entry("b" * 50_000, uuid="u2"),
            _entry("c" * 50_000, uuid="u3"),
        ]
        result = extract_assistant_text(entries)
        assert len(result) == 3
        for r in result:
            assert not r.endswith("..."), "No truncation expected when combined < cap"


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


# =============================================================================
# Mirror integrity
# =============================================================================


@pytest.mark.unit
class TestMirrorIntegrity:
    """Acceptance criteria 1-4: both transcript_extractor.py copies must mirror."""

    def test_files_byte_identical(self) -> None:
        spatial = Path("spatial_memory/hooks/transcript_extractor.py")
        plugin = Path("plugin/hooks/transcript_extractor.py")
        assert spatial.exists() and plugin.exists()
        assert filecmp.cmp(spatial, plugin, shallow=False), (
            "plugin/hooks/transcript_extractor.py must mirror "
            "spatial_memory/hooks/transcript_extractor.py byte-for-byte"
        )

    def test_plugin_module_imports_and_exports_safety_cap(self) -> None:
        plugin_te = importlib.import_module("plugin.hooks.transcript_extractor")
        assert plugin_te._SAFETY_CAP == 200_000
        assert not hasattr(plugin_te, "MAX_ENTRY_TEXT")
        assert not hasattr(plugin_te, "MAX_COMBINED_TEXT")
