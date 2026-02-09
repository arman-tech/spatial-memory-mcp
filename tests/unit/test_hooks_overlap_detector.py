"""Unit tests for spatial_memory.hooks.overlap_detector.

Tests cover:
1. get_queued_hashes — reading hashes from queue files
2. is_duplicate — content comparison against hash set
3. Edge cases — missing dir, malformed files, empty content
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_memory.hooks.overlap_detector import (
    _content_hash,
    get_queued_hashes,
    is_duplicate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_queue_file(new_dir: Path, content: str, filename: str = "1.json") -> None:
    """Write a minimal queue file to the new/ directory."""
    payload = {"content": content, "version": 1}
    (new_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


# =============================================================================
# get_queued_hashes
# =============================================================================


@pytest.mark.unit
class TestGetQueuedHashes:
    """Test hash collection from queue files."""

    def test_reads_hashes_from_queue_files(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "Decided to use Redis.", "001.json")
        _write_queue_file(new_dir, "The fix was resetting cache.", "002.json")

        hashes = get_queued_hashes(tmp_path)
        assert len(hashes) == 2

    def test_empty_new_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        hashes = get_queued_hashes(tmp_path)
        assert hashes == set()

    def test_missing_new_dir(self, tmp_path: Path) -> None:
        hashes = get_queued_hashes(tmp_path)
        assert hashes == set()

    def test_max_age_files_limits_reads(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        for i in range(10):
            _write_queue_file(new_dir, f"Content {i}", f"{i:03d}.json")

        hashes = get_queued_hashes(tmp_path, max_age_files=3)
        assert len(hashes) == 3

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "bad.json").write_text("not json")
        _write_queue_file(new_dir, "Good content.", "good.json")

        hashes = get_queued_hashes(tmp_path)
        assert len(hashes) == 1

    def test_empty_content_skipped(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "", "empty.json")
        _write_queue_file(new_dir, "Real content.", "real.json")

        hashes = get_queued_hashes(tmp_path)
        assert len(hashes) == 1

    def test_whitespace_content_skipped(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "   ", "ws.json")

        hashes = get_queued_hashes(tmp_path)
        assert hashes == set()


# =============================================================================
# is_duplicate
# =============================================================================


@pytest.mark.unit
class TestIsDuplicate:
    """Test content duplication detection."""

    def test_exact_match_detected(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "Decided to use Redis.")

        hashes = get_queued_hashes(tmp_path)
        assert is_duplicate("Decided to use Redis.", hashes) is True

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "Decided to use Redis.")

        hashes = get_queued_hashes(tmp_path)
        assert is_duplicate("decided to use redis.", hashes) is True

    def test_whitespace_normalized(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "  Decided to use Redis.  ")

        hashes = get_queued_hashes(tmp_path)
        assert is_duplicate("Decided to use Redis.", hashes) is True

    def test_different_content_not_duplicate(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _write_queue_file(new_dir, "Decided to use Redis.")

        hashes = get_queued_hashes(tmp_path)
        assert is_duplicate("Decided to use PostgreSQL.", hashes) is False

    def test_empty_content_not_duplicate(self) -> None:
        assert is_duplicate("", {"some_hash"}) is False

    def test_empty_hashes_not_duplicate(self) -> None:
        assert is_duplicate("Some content.", set()) is False

    def test_both_empty(self) -> None:
        assert is_duplicate("", set()) is False


# =============================================================================
# _content_hash
# =============================================================================


@pytest.mark.unit
class TestContentHash:
    """Test the internal hash function."""

    def test_deterministic(self) -> None:
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_case_normalized(self) -> None:
        h1 = _content_hash("Hello World")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_whitespace_stripped(self) -> None:
        h1 = _content_hash("  hello  ")
        h2 = _content_hash("hello")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2
