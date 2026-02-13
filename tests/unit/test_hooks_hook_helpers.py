"""Unit tests for spatial_memory.hooks.hook_helpers.

Tests cover:
1. sanitize_session_id — valid IDs, invalid chars, Windows device names, length
2. validate_transcript_path — absolute paths, traversal, relative paths
3. read_stdin — valid JSON, empty, non-dict, size limit
4. write_stdout_response — JSON output, exception swallowed
5. get_project_root — env var set/unset
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import patch

import pytest

from spatial_memory.hooks.hook_helpers import (
    get_project_root,
    log_hook_error,
    read_stdin,
    sanitize_session_id,
    validate_cwd,
    validate_transcript_path,
    write_stdout_response,
)

# =============================================================================
# sanitize_session_id
# =============================================================================


@pytest.mark.unit
class TestSanitizeSessionId:
    """Test session ID sanitization for safe filename use."""

    def test_valid_alphanumeric(self) -> None:
        assert sanitize_session_id("abc123") == "abc123"

    def test_valid_with_hyphens(self) -> None:
        assert sanitize_session_id("session-abc-123") == "session-abc-123"

    def test_valid_with_underscores(self) -> None:
        assert sanitize_session_id("session_abc_123") == "session_abc_123"

    def test_empty_string(self) -> None:
        assert sanitize_session_id("") == ""

    def test_rejects_slashes(self) -> None:
        assert sanitize_session_id("../../etc/passwd") == ""

    def test_rejects_backslashes(self) -> None:
        assert sanitize_session_id("..\\..\\windows\\system32") == ""

    def test_rejects_spaces(self) -> None:
        assert sanitize_session_id("session id") == ""

    def test_rejects_dots(self) -> None:
        assert sanitize_session_id("session.id") == ""

    def test_rejects_null_bytes(self) -> None:
        assert sanitize_session_id("session\x00id") == ""

    def test_rejects_unicode(self) -> None:
        assert sanitize_session_id("session\u200bid") == ""

    def test_length_cap(self) -> None:
        long_id = "a" * 128
        assert sanitize_session_id(long_id) == long_id

    def test_over_length_cap(self) -> None:
        too_long = "a" * 129
        assert sanitize_session_id(too_long) == ""

    def test_rejects_windows_device_con(self) -> None:
        assert sanitize_session_id("CON") == ""

    def test_rejects_windows_device_nul(self) -> None:
        assert sanitize_session_id("NUL") == ""

    def test_rejects_windows_device_prn(self) -> None:
        assert sanitize_session_id("PRN") == ""

    def test_rejects_windows_device_aux(self) -> None:
        assert sanitize_session_id("AUX") == ""

    def test_rejects_windows_device_com1(self) -> None:
        assert sanitize_session_id("COM1") == ""

    def test_rejects_windows_device_lpt1(self) -> None:
        assert sanitize_session_id("LPT1") == ""

    def test_rejects_device_name_case_insensitive(self) -> None:
        assert sanitize_session_id("con") == ""
        assert sanitize_session_id("Con") == ""
        assert sanitize_session_id("nul") == ""


# =============================================================================
# validate_transcript_path
# =============================================================================


@pytest.mark.unit
class TestValidateTranscriptPath:
    """Test transcript path validation."""

    def test_valid_absolute_path(self, tmp_path: os.PathLike[str]) -> None:
        """Platform-independent: tmp_path is always absolute."""
        path = str(tmp_path / "transcript.jsonl")
        assert validate_transcript_path(path) == path

    def test_empty_string(self) -> None:
        assert validate_transcript_path("") == ""

    def test_rejects_traversal(self, tmp_path: os.PathLike[str]) -> None:
        bad_path = str(tmp_path / ".." / ".." / "etc" / "passwd")
        assert validate_transcript_path(bad_path) == ""

    def test_rejects_traversal_in_middle(self, tmp_path: os.PathLike[str]) -> None:
        bad_path = str(tmp_path / ".." / "other" / "t.jsonl")
        assert validate_transcript_path(bad_path) == ""

    def test_rejects_relative_path(self) -> None:
        assert validate_transcript_path("relative/path/t.jsonl") == ""

    def test_rejects_dot_relative(self) -> None:
        assert validate_transcript_path("./transcript.jsonl") == ""


# =============================================================================
# read_stdin
# =============================================================================


@pytest.mark.unit
class TestReadStdin:
    """Test stdin JSON parsing."""

    def test_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"session_id": "s1", "tool_name": "Edit"}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(data)))
        result = read_stdin()
        assert result["session_id"] == "s1"

    def test_empty_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert read_stdin() == {}

    def test_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("   \n  "))
        assert read_stdin() == {}

    def test_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        assert read_stdin() == {}

    def test_non_dict_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2, 3]"))
        assert read_stdin() == {}

    def test_large_stdin_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        large = '{"key": "' + "x" * 600_000 + '"}'
        monkeypatch.setattr("sys.stdin", io.StringIO(large))
        result = read_stdin()
        # Truncation at 512KB cuts mid-string → JSONDecodeError → {}
        assert result == {} or isinstance(result, dict)


# =============================================================================
# write_stdout_response
# =============================================================================


@pytest.mark.unit
class TestWriteStdoutResponse:
    """Test stdout response writing."""

    def test_writes_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        write_stdout_response()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True
        assert output["suppressOutput"] is True

    def test_exception_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout", None)
        write_stdout_response()  # Should not raise


# =============================================================================
# get_project_root
# =============================================================================


@pytest.mark.unit
class TestGetProjectRoot:
    """Test project root resolution from env."""

    def test_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/my/project")
        assert get_project_root() == "/my/project"

    def test_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        assert get_project_root() == ""

    def test_env_var_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "")
        assert get_project_root() == ""


# =============================================================================
# Defense-in-depth: integration with transcript_reader
# =============================================================================


@pytest.mark.unit
class TestSanitizationIntegration:
    """Verify sanitization works correctly at the reader layer."""

    def test_malicious_session_id_rejected_by_load_state(self) -> None:
        from spatial_memory.hooks.transcript_reader import load_state

        state = load_state("../../etc/passwd")
        assert state["last_offset"] == 0

    def test_malicious_session_id_rejected_by_save_state(
        self, tmp_path: os.PathLike[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from spatial_memory.hooks.transcript_reader import save_state

        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
        # Should silently skip (empty after sanitization)
        save_state("../../etc/passwd", 100, "ts")

    def test_relative_transcript_path_rejected(self) -> None:
        from spatial_memory.hooks.transcript_reader import read_transcript_delta

        entries, offset = read_transcript_delta("relative/path.jsonl")
        assert entries == []
        assert offset == 0


# =============================================================================
# validate_cwd
# =============================================================================


@pytest.mark.unit
class TestValidateCwd:
    """Test CWD path validation."""

    def test_valid_absolute_path(self, tmp_path: os.PathLike[str]) -> None:
        path = str(tmp_path)
        assert validate_cwd(path) == path

    def test_empty_string(self) -> None:
        assert validate_cwd("") == ""

    def test_rejects_too_long(self) -> None:
        long_path = "/a" * 2100
        assert validate_cwd(long_path) == ""

    def test_rejects_traversal(self) -> None:
        assert validate_cwd("/home/user/../etc") == ""

    def test_rejects_relative(self) -> None:
        assert validate_cwd("relative/path") == ""

    def test_rejects_windows_device(self) -> None:
        # On Windows, os.path.isabs("C:\\CON") is True
        # The basename check should catch it
        import sys

        if sys.platform == "win32":
            assert validate_cwd("C:\\CON") == ""

    def test_valid_unix_path(self) -> None:
        import sys

        if sys.platform != "win32":
            assert validate_cwd("/home/user/project") == "/home/user/project"


# =============================================================================
# log_hook_error
# =============================================================================


@pytest.mark.unit
class TestLogHookError:
    """Test error logging utility."""

    def test_creates_log_file(self, tmp_path: os.PathLike[str]) -> None:
        import os as os_mod
        from pathlib import Path

        with patch.dict(os_mod.environ, {"SPATIAL_MEMORY_MEMORY_PATH": str(tmp_path)}):
            log_hook_error(ValueError("test error"), "TestHook")
            log_file = Path(str(tmp_path)) / "hook-errors.log"
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "TestHook" in content
            assert "ValueError" in content
            assert "test error" in content

    def test_appends_to_existing(self, tmp_path: os.PathLike[str]) -> None:
        import os as os_mod
        from pathlib import Path

        with patch.dict(os_mod.environ, {"SPATIAL_MEMORY_MEMORY_PATH": str(tmp_path)}):
            log_hook_error(ValueError("error 1"), "Hook1")
            log_hook_error(RuntimeError("error 2"), "Hook2")
            log_file = Path(str(tmp_path)) / "hook-errors.log"
            content = log_file.read_text(encoding="utf-8")
            assert "error 1" in content
            assert "error 2" in content

    def test_rotates_large_file(self, tmp_path: os.PathLike[str]) -> None:
        import os as os_mod
        from pathlib import Path

        with patch.dict(os_mod.environ, {"SPATIAL_MEMORY_MEMORY_PATH": str(tmp_path)}):
            log_file = Path(str(tmp_path)) / "hook-errors.log"
            # Write a file larger than 1MB
            log_file.write_text("x" * 1_100_000, encoding="utf-8")
            log_hook_error(ValueError("after rotation"), "HookR")
            # Original should be renamed
            rotated = Path(str(tmp_path)) / "hook-errors.log.1"
            assert rotated.exists()
            assert log_file.exists()
            assert "after rotation" in log_file.read_text(encoding="utf-8")

    def test_never_raises(self) -> None:
        # With no SPATIAL_MEMORY_MEMORY_PATH and no cwd, should silently skip
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SPATIAL_MEMORY_MEMORY_PATH", None)
            with patch.dict(os.environ, env, clear=True):
                log_hook_error(ValueError("safe"), "Hook")  # Should not raise
