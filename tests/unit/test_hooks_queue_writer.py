"""Unit tests for spatial_memory.hooks.queue_writer.

Tests cover:
1. Queue directory discovery — env var, default, edge cases
2. Filename generation — format, sortability, uniqueness
3. Atomic write — file in new/ not tmp/, valid JSON, auto-create dirs
4. JSON schema — all 12 fields, types, defaults, ISO timestamp
5. Round-trip — write then parse with QueueFile.from_json() succeeds
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from spatial_memory.hooks.queue_writer import (
    QUEUE_DIR_NAME,
    QUEUE_FILE_VERSION,
    _make_filename,
    get_queue_dir,
    write_queue_file,
)

# =============================================================================
# Queue Directory Discovery
# =============================================================================


@pytest.mark.unit
class TestQueueDirectoryDiscovery:
    """Test get_queue_dir() resolution logic."""

    def test_env_var_takes_precedence(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": str(tmp_path)}):
            result = get_queue_dir()
            assert result == tmp_path / QUEUE_DIR_NAME

    def test_default_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            # Remove the env var if it exists
            env = os.environ.copy()
            env.pop("SPATIAL_MEMORY_MEMORY_PATH", None)
            with patch.dict(os.environ, env, clear=True):
                result = get_queue_dir()
                assert result == Path.home() / ".spatial-memory" / QUEUE_DIR_NAME

    def test_empty_env_var_uses_default(self) -> None:
        with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": ""}):
            result = get_queue_dir()
            assert result == Path.home() / ".spatial-memory" / QUEUE_DIR_NAME

    def test_queue_dir_name_matches_constant(self) -> None:
        """Embedded constant matches core/queue_constants.py value."""
        from spatial_memory.core.queue_constants import QUEUE_DIR_NAME as CANONICAL

        assert QUEUE_DIR_NAME == CANONICAL


# =============================================================================
# Filename Generation
# =============================================================================


@pytest.mark.unit
class TestFilenameGeneration:
    """Test _make_filename() format and properties."""

    def test_format(self) -> None:
        filename = _make_filename()
        assert filename.endswith(".json")
        parts = filename[:-5].split("-")  # strip .json
        assert len(parts) >= 3  # time_ns, pid, random_hex
        # First part should be numeric (time_ns)
        assert parts[0].isdigit()

    def test_sortability(self) -> None:
        """Filenames generated in sequence should sort chronologically."""
        names = [_make_filename() for _ in range(10)]
        assert names == sorted(names)

    def test_uniqueness(self) -> None:
        """Multiple filenames should all be unique."""
        names = {_make_filename() for _ in range(100)}
        assert len(names) == 100


# =============================================================================
# Atomic Write
# =============================================================================


@pytest.mark.unit
class TestAtomicWrite:
    """Test atomic write protocol (tmp/ -> new/)."""

    def test_file_lands_in_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                result_path = write_queue_file(
                    content="Test content",
                    source_hook="PostToolUse",
                )
                assert result_path.parent.name == "new"
                assert result_path.exists()

    def test_tmp_dir_empty_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                write_queue_file(content="Test", source_hook="PostToolUse")
                tmp_dir = Path(tmpdir) / QUEUE_DIR_NAME / "tmp"
                assert list(tmp_dir.iterdir()) == []

    def test_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                result_path = write_queue_file(
                    content="Valid JSON test",
                    source_hook="Stop",
                )
                data = json.loads(result_path.read_text(encoding="utf-8"))
                assert isinstance(data, dict)
                assert data["content"] == "Valid JSON test"

    def test_auto_create_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir) / "deep" / "nested" / "path"
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": str(deep)}):
                result_path = write_queue_file(
                    content="Auto-create test",
                    source_hook="PostToolUse",
                )
                assert result_path.exists()


# =============================================================================
# JSON Schema
# =============================================================================


@pytest.mark.unit
class TestJsonSchema:
    """Test that written JSON matches QueueFile.from_json() contract."""

    def _write_and_read(self, **kwargs: object) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                if "content" not in kwargs:
                    kwargs["content"] = "Schema test content"
                if "source_hook" not in kwargs:
                    kwargs["source_hook"] = "PostToolUse"
                result_path = write_queue_file(**kwargs)  # type: ignore[arg-type]
                return json.loads(result_path.read_text(encoding="utf-8"))

    def test_all_12_fields_present(self) -> None:
        data = self._write_and_read()
        expected_fields = {
            "version",
            "content",
            "source_hook",
            "timestamp",
            "project_root_dir",
            "suggested_namespace",
            "suggested_tags",
            "suggested_importance",
            "signal_tier",
            "signal_patterns_matched",
            "context",
            "client",
        }
        assert set(data.keys()) == expected_fields

    def test_version_matches_constant(self) -> None:
        data = self._write_and_read()
        assert data["version"] == QUEUE_FILE_VERSION
        # Also verify against canonical constant
        from spatial_memory.core.queue_constants import QUEUE_FILE_VERSION as CANONICAL

        assert data["version"] == CANONICAL

    def test_default_values(self) -> None:
        data = self._write_and_read()
        assert data["project_root_dir"] == ""
        assert data["suggested_namespace"] == "default"
        assert data["suggested_tags"] == []
        assert data["suggested_importance"] == 0.5
        assert data["signal_tier"] == 1
        assert data["signal_patterns_matched"] == []
        assert data["context"] == {}
        assert data["client"] == "claude-code"

    def test_custom_values(self) -> None:
        data = self._write_and_read(
            project_root_dir="/home/user/project",
            suggested_namespace="decisions",
            suggested_tags=["python", "architecture"],
            suggested_importance=0.9,
            signal_tier=2,
            signal_patterns_matched=["decision"],
            context={"file": "app.py"},
            client="custom-client",
        )
        assert data["project_root_dir"] == "/home/user/project"
        assert data["suggested_namespace"] == "decisions"
        assert data["suggested_tags"] == ["python", "architecture"]
        assert data["suggested_importance"] == 0.9
        assert data["signal_tier"] == 2
        assert data["signal_patterns_matched"] == ["decision"]
        assert data["context"] == {"file": "app.py"}
        assert data["client"] == "custom-client"

    def test_iso_timestamp_format(self) -> None:
        data = self._write_and_read()
        ts = data["timestamp"]
        # Should match ISO 8601 UTC format: YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), f"Bad timestamp: {ts}"

    def test_content_preserved_exactly(self) -> None:
        text = "Unicode: 数据库  \n\tTabs and newlines"
        data = self._write_and_read(content=text)
        assert data["content"] == text

    def test_types_correct(self) -> None:
        data = self._write_and_read(
            suggested_tags=["a", "b"],
            suggested_importance=0.7,
        )
        assert isinstance(data["version"], int)
        assert isinstance(data["content"], str)
        assert isinstance(data["suggested_tags"], list)
        assert isinstance(data["suggested_importance"], float)
        assert isinstance(data["signal_tier"], int)
        assert isinstance(data["context"], dict)

    def test_source_hook_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                for hook in ["PostToolUse", "PreCompact", "Stop"]:
                    path = write_queue_file(content="Test", source_hook=hook)
                    data = json.loads(path.read_text(encoding="utf-8"))
                    assert data["source_hook"] == hook


# =============================================================================
# Round-Trip with QueueFile.from_json()
# =============================================================================


@pytest.mark.unit
class TestRoundTrip:
    """Critical: write then parse with QueueFile.from_json() must succeed."""

    def test_basic_roundtrip(self) -> None:
        from spatial_memory.core.queue_file import QueueFile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                path = write_queue_file(
                    content="Round-trip test: decided to use PostgreSQL",
                    source_hook="PostToolUse",
                    project_root_dir="/home/user/project",
                    suggested_namespace="decisions",
                    suggested_tags=["postgresql", "database"],
                    suggested_importance=0.8,
                    signal_tier=1,
                    signal_patterns_matched=["decision"],
                    context={"file": "app.py"},
                )
                data = json.loads(path.read_text(encoding="utf-8"))
                qf = QueueFile.from_json(data)

                assert qf.version == QUEUE_FILE_VERSION
                assert qf.content == "Round-trip test: decided to use PostgreSQL"
                assert qf.source_hook == "PostToolUse"
                assert qf.project_root_dir == "/home/user/project"
                assert qf.suggested_namespace == "decisions"
                assert qf.suggested_tags == ["postgresql", "database"]
                assert qf.suggested_importance == 0.8
                assert qf.signal_tier == 1
                assert qf.signal_patterns_matched == ["decision"]
                assert qf.context == {"file": "app.py"}
                assert qf.client == "claude-code"

    def test_minimal_roundtrip(self) -> None:
        """Minimal arguments should still produce valid QueueFile."""
        from spatial_memory.core.queue_file import QueueFile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                path = write_queue_file(
                    content="Minimal test content",
                    source_hook="Stop",
                )
                data = json.loads(path.read_text(encoding="utf-8"))
                qf = QueueFile.from_json(data)
                assert qf.content == "Minimal test content"
                assert qf.source_hook == "Stop"

    def test_roundtrip_with_unicode(self) -> None:
        """Unicode content survives the round-trip."""
        from spatial_memory.core.queue_file import QueueFile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPATIAL_MEMORY_MEMORY_PATH": tmpdir}):
                content = "数据库设计: decided to use PostgreSQL for 日本語 support"
                path = write_queue_file(
                    content=content,
                    source_hook="PreCompact",
                    suggested_tags=["unicode", "i18n"],
                )
                data = json.loads(path.read_text(encoding="utf-8"))
                qf = QueueFile.from_json(data)
                assert qf.content == content
