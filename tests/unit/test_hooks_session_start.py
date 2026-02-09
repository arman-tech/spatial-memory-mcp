"""Unit tests for spatial_memory.hooks.session_start.

Tests cover:
1. main() — startup/resume trigger emits additionalContext
2. Ignored sources — clear, compact, unknown sources produce no output
3. Fail-open — empty input, invalid JSON, non-dict JSON
4. Output format — valid JSON with correct structure
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from spatial_memory.hooks.session_start import (
    _RECALL_NUDGE,
    _TRIGGER_SOURCES,
    main,
)

# =============================================================================
# Trigger sources
# =============================================================================


@pytest.mark.unit
class TestTriggerSources:
    """Verify which sources trigger the recall nudge."""

    def test_startup_emits_context(self) -> None:
        data = json.dumps({"source": "startup", "session_id": "s1"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        output = json.loads(stdout.getvalue().strip())
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert output["hookSpecificOutput"]["additionalContext"] == _RECALL_NUDGE

    def test_resume_emits_context(self) -> None:
        data = json.dumps({"source": "resume", "session_id": "s2"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        output = json.loads(stdout.getvalue().strip())
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_clear_produces_no_output(self) -> None:
        data = json.dumps({"source": "clear"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_compact_produces_no_output(self) -> None:
        data = json.dumps({"source": "compact"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_unknown_source_produces_no_output(self) -> None:
        data = json.dumps({"source": "something_else"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_missing_source_produces_no_output(self) -> None:
        data = json.dumps({"session_id": "s1"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_trigger_sources_frozenset(self) -> None:
        assert _TRIGGER_SOURCES == {"startup", "resume"}


# =============================================================================
# Fail-open behavior
# =============================================================================


@pytest.mark.unit
class TestFailOpen:
    """Verify fail-open behavior on invalid input."""

    def test_empty_stdin(self) -> None:
        stdin = io.StringIO("")
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_whitespace_only_stdin(self) -> None:
        stdin = io.StringIO("   \n  ")
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_invalid_json(self) -> None:
        stdin = io.StringIO("not json at all")
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()  # Should not raise
        assert stdout.getvalue() == ""

    def test_non_dict_json(self) -> None:
        stdin = io.StringIO(json.dumps([1, 2, 3]))
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        assert stdout.getvalue() == ""

    def test_stdin_read_error(self) -> None:
        """If stdin.read raises, no exception escapes."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.side_effect = OSError("broken pipe")
            main()  # Should not raise

    def test_stdout_write_error_swallowed(self) -> None:
        """If stdout write fails on a valid trigger, no exception escapes."""
        data = json.dumps({"source": "startup"})
        stdin = io.StringIO(data)
        with patch("sys.stdin", stdin), patch("sys.stdout", None):
            main()  # Should not raise


# =============================================================================
# Output format
# =============================================================================


@pytest.mark.unit
class TestOutputFormat:
    """Verify the JSON output structure."""

    def test_output_is_single_line_json(self) -> None:
        data = json.dumps({"source": "startup"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        lines = stdout.getvalue().strip().split("\n")
        assert len(lines) == 1
        output = json.loads(lines[0])
        assert "hookSpecificOutput" in output

    def test_no_continue_or_suppress_keys(self) -> None:
        """SessionStart hooks use hookSpecificOutput, not continue/suppress."""
        data = json.dumps({"source": "startup"})
        stdin = io.StringIO(data)
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            main()
        output = json.loads(stdout.getvalue().strip())
        assert "continue" not in output
        assert "suppressOutput" not in output

    def test_recall_nudge_content(self) -> None:
        """The nudge text mentions recall."""
        assert "recall" in _RECALL_NUDGE.lower()
        assert "memories" in _RECALL_NUDGE.lower()
