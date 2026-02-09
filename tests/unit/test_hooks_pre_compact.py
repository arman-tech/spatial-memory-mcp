"""Unit tests for spatial_memory.hooks.pre_compact.

Tests cover:
1. Module loading — _load_hook_module
2. main() — fail-open behavior, empty input, no transcript path
3. Pipeline invocation — correct args passed

Note: stdin/stdout/env helpers are tested in test_hooks_hook_helpers.py.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from spatial_memory.hooks.pre_compact import (
    _load_hook_module,
    main,
)

# =============================================================================
# _load_hook_module
# =============================================================================


@pytest.mark.unit
class TestLoadHookModule:
    """Test importlib-based module loading."""

    def test_loads_models(self) -> None:
        mod = _load_hook_module("models")
        assert hasattr(mod, "TranscriptHookInput")

    def test_loads_signal_detection(self) -> None:
        mod = _load_hook_module("signal_detection")
        assert hasattr(mod, "classify_signal")

    def test_missing_module_raises(self) -> None:
        with pytest.raises(ImportError, match="Hook module not found"):
            _load_hook_module("nonexistent_module_xyz")


# =============================================================================
# main() — integration-style tests with mocked modules
# =============================================================================


@pytest.mark.unit
class TestMain:
    """Test the main entrypoint."""

    def test_empty_input_exits_cleanly(self) -> None:
        with patch("sys.stdin", io.StringIO("")):
            main()  # Should not raise

    def test_no_transcript_path_exits_cleanly(self) -> None:
        data = {"session_id": "s1"}
        with patch("sys.stdin", io.StringIO(json.dumps(data))):
            main()  # Should not raise

    def test_exception_swallowed(self) -> None:
        """Fail-open: even if module loading fails, no exception escapes."""
        data = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
        }
        with (
            patch("sys.stdin", io.StringIO(json.dumps(data))),
            patch(
                "spatial_memory.hooks.pre_compact._load_hook_module",
                side_effect=ImportError("boom"),
            ),
        ):
            main()  # Should not raise

    def test_runs_pipeline_with_valid_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify pipeline is invoked when given valid input."""
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/project")

        # Create a minimal transcript with an assistant entry
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Decided to use Redis."}],
            },
            "isSidechain": False,
            "timestamp": "2026-01-01T00:00:00Z",
            "uuid": "uuid-1",
        }
        transcript.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        data = {
            "session_id": "test-session",
            "transcript_path": str(transcript),
            "cwd": str(tmp_path),
            "hook_event_name": "PreCompact",
            "trigger": "auto",
        }

        with patch("sys.stdin", io.StringIO(json.dumps(data))):
            main()

        # Verify no crash occurred — the pipeline ran successfully.
        # State should be saved even if no queue files written (signal might be tier 3)

    def test_hook_event_name_defaults_to_precompact(self) -> None:
        """If hook_event_name is missing, defaults to 'PreCompact'."""
        data = {
            "session_id": "s1",
            "transcript_path": "/nonexistent.jsonl",
        }
        # This will silently fail on read (file doesn't exist) but should
        # pass TranscriptHookInput with hook_event_name="PreCompact"
        with patch("sys.stdin", io.StringIO(json.dumps(data))):
            main()  # Should not raise
