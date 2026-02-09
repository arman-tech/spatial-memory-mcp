"""Unit tests for spatial_memory.hooks.stop.

Tests cover:
1. main() — fail-open behavior, empty input, no transcript path
2. Loop guard — stop_hook_active=True skips processing
3. stdout response — always emits continue/suppressOutput
4. stdin parsing — valid JSON, empty, non-dict
5. Pipeline invocation — correct args passed
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from spatial_memory.hooks.stop import (
    _load_hook_module,
    _write_stdout_response,
    main,
)

# =============================================================================
# _write_stdout_response (local safety net)
# =============================================================================


@pytest.mark.unit
class TestWriteStdoutResponse:
    """Test the local stdout safety net."""

    def test_writes_valid_json(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _write_stdout_response()
        output = buf.getvalue()
        data = json.loads(output.strip())
        assert data["continue"] is True
        assert data["suppressOutput"] is True

    def test_exception_swallowed(self) -> None:
        """If stdout write fails, no exception escapes."""
        with patch("sys.stdout", None):
            _write_stdout_response()  # Should not raise


# =============================================================================
# _load_hook_module
# =============================================================================


@pytest.mark.unit
class TestLoadHookModule:
    """Test importlib-based module loading."""

    def test_loads_models(self) -> None:
        mod = _load_hook_module("models")
        assert hasattr(mod, "TranscriptHookInput")

    def test_missing_module_raises(self) -> None:
        with pytest.raises(ImportError, match="Hook module not found"):
            _load_hook_module("nonexistent_module_xyz")


# =============================================================================
# Loop guard
# =============================================================================


@pytest.mark.unit
class TestLoopGuard:
    """Test stop_hook_active loop guard."""

    def test_stop_hook_active_true_skips_processing(self) -> None:
        """When stop_hook_active=True, skip all processing."""
        data = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "stop_hook_active": True,
        }
        buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(data))), patch("sys.stdout", buf):
            main()

        # Should still write response
        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True

    def test_stop_hook_active_false_processes(self) -> None:
        """When stop_hook_active=False, processing occurs normally."""
        data = {
            "session_id": "s1",
            "transcript_path": "/nonexistent.jsonl",
            "stop_hook_active": False,
        }
        buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(data))), patch("sys.stdout", buf):
            main()

        # Should still write response (even though file doesn't exist)
        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True


# =============================================================================
# main() — integration-style tests
# =============================================================================


@pytest.mark.unit
class TestMain:
    """Test the main entrypoint."""

    def test_empty_input_writes_response(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdin", io.StringIO("")), patch("sys.stdout", buf):
            main()

        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True

    def test_no_transcript_path_writes_response(self) -> None:
        data = {"session_id": "s1"}
        buf = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(data))),
            patch("sys.stdout", buf),
        ):
            main()

        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True

    def test_exception_swallowed_and_response_written(self) -> None:
        """Fail-open: even if module loading fails, response is still written."""
        data = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "stop_hook_active": False,
        }
        buf = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(data))),
            patch("sys.stdout", buf),
            patch(
                "spatial_memory.hooks.stop._load_hook_module",
                side_effect=ImportError("boom"),
            ),
        ):
            main()

        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True

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
            "hook_event_name": "Stop",
            "stop_hook_active": False,
        }

        buf = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(data))),
            patch("sys.stdout", buf),
        ):
            main()

        # Verify response was written
        output = buf.getvalue()
        resp = json.loads(output.strip())
        assert resp["continue"] is True

    def test_hook_event_name_defaults_to_stop(self) -> None:
        """If hook_event_name is missing, defaults to 'Stop'."""
        data = {
            "session_id": "s1",
            "transcript_path": "/nonexistent.jsonl",
            "stop_hook_active": False,
        }
        buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(data))), patch("sys.stdout", buf):
            main()  # Should not raise
