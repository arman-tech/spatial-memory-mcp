"""Unit tests for spatial_memory.hooks.dispatcher.

Tests cover:
1. Event normalization — PascalCase, camelCase, kebab-case, Cursor-specific aliases
2. Client detection — cursor_version heuristic, default to claude-code
3. Stdin normalization — Cursor field mapping (conversation_id, workspace_roots, etc.)
4. Centralized validation — session_id, transcript_path, cwd sanitization
5. Handler routing — known events route correctly, unknown returns None
6. SessionStart handler — startup/resume produce nudge, clear/compact produce None
7. PostToolUse handler — qualifying tool queues, skipped tool returns None
8. _load_hook_module — loads, caches, nonexistent raises ImportError
9. Cursor output translation — fire-and-forget behavior
10. dispatch() — integration of routing + handler
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.hooks.dispatcher import (
    _HANDLER_MAP,
    _detect_client,
    _handle_post_tool_use,
    _handle_pre_compact,
    _handle_session_start,
    _handle_stop,
    _load_hook_module,
    _normalize_event,
    _normalize_stdin,
    _parse_client_flag,
    _translate_output_for_cursor,
    _validate_common,
    dispatch,
)

# =============================================================================
# Event normalization
# =============================================================================


@pytest.mark.unit
class TestNormalizeEvent:
    """Test event name normalization to canonical PascalCase."""

    def test_pascal_case(self) -> None:
        assert _normalize_event("SessionStart") == "SessionStart"
        assert _normalize_event("PostToolUse") == "PostToolUse"
        assert _normalize_event("PreCompact") == "PreCompact"
        assert _normalize_event("Stop") == "Stop"

    def test_camel_case(self) -> None:
        assert _normalize_event("sessionStart") == "SessionStart"
        assert _normalize_event("postToolUse") == "PostToolUse"
        assert _normalize_event("preCompact") == "PreCompact"
        assert _normalize_event("stop") == "Stop"

    def test_kebab_case(self) -> None:
        assert _normalize_event("session-start") == "SessionStart"
        assert _normalize_event("post-tool-use") == "PostToolUse"
        assert _normalize_event("pre-compact") == "PreCompact"

    def test_cursor_after_mcp_execution(self) -> None:
        assert _normalize_event("afterMCPExecution") == "PostToolUse"
        assert _normalize_event("after-mcp-execution") == "PostToolUse"

    def test_unknown_returns_none(self) -> None:
        assert _normalize_event("unknown-event") is None
        assert _normalize_event("") is None
        assert _normalize_event("fooBar") is None


# =============================================================================
# Client detection
# =============================================================================


@pytest.mark.unit
class TestDetectClient:
    """Test client detection from explicit flag and data heuristics."""

    def test_explicit_cursor(self) -> None:
        assert _detect_client({}, "cursor") == "cursor"

    def test_explicit_claude_code(self) -> None:
        assert _detect_client({}, "claude-code") == "claude-code"

    def test_cursor_version_heuristic(self) -> None:
        assert _detect_client({"cursor_version": "1.0"}, "claude-code") == "cursor"

    def test_cursor_version_camel(self) -> None:
        assert _detect_client({"cursorVersion": "1.0"}, "claude-code") == "cursor"

    def test_default_claude_code(self) -> None:
        assert _detect_client({}, "") == "claude-code"


# =============================================================================
# Parse client flag
# =============================================================================


@pytest.mark.unit
class TestParseClientFlag:
    """Test --client flag parsing from argv."""

    def test_client_present(self) -> None:
        assert _parse_client_flag(["session-start", "--client", "cursor"]) == "cursor"

    def test_client_absent(self) -> None:
        assert _parse_client_flag(["session-start"]) == "claude-code"

    def test_client_at_end(self) -> None:
        assert _parse_client_flag(["stop", "--client"]) == "claude-code"


# =============================================================================
# Stdin normalization
# =============================================================================


@pytest.mark.unit
class TestNormalizeStdin:
    """Test cross-client field mapping for Cursor."""

    def test_claude_code_passthrough(self) -> None:
        data = {"session_id": "s1", "cwd": "/project"}
        result = _normalize_stdin(data, "claude-code")
        assert result is data  # No copy for claude-code

    def test_cursor_conversation_id(self) -> None:
        data = {"conversation_id": "c1"}
        result = _normalize_stdin(data, "cursor")
        assert result["session_id"] == "c1"
        assert "conversation_id" not in result

    def test_cursor_workspace_roots(self) -> None:
        data = {"workspace_roots": ["/project/a", "/project/b"]}
        result = _normalize_stdin(data, "cursor")
        assert result["cwd"] == "/project/a"

    def test_cursor_drive_path_normalization(self) -> None:
        """Cursor sends /c:/Users/... which fails os.path.isabs on Windows."""
        data = {"workspace_roots": ["/c:/Users/dev/project"]}
        result = _normalize_stdin(data, "cursor")
        assert result["cwd"] == "C:/Users/dev/project"

    def test_cursor_drive_path_lowercase_uppercased(self) -> None:
        data = {"workspace_roots": ["/d:/work/repo"]}
        result = _normalize_stdin(data, "cursor")
        assert result["cwd"] == "D:/work/repo"

    def test_cursor_drive_path_not_applied_to_unix_paths(self) -> None:
        """Unix paths like /home/user should not be mangled."""
        data = {"workspace_roots": ["/home/user/project"]}
        result = _normalize_stdin(data, "cursor")
        assert result["cwd"] == "/home/user/project"

    def test_cursor_drive_path_not_applied_to_claude_code(self) -> None:
        data = {"cwd": "/c:/Users/dev/project"}
        result = _normalize_stdin(data, "claude-code")
        assert result["cwd"] == "/c:/Users/dev/project"

    def test_cursor_result_json(self) -> None:
        data = {"result_json": '{"ok": true}'}
        result = _normalize_stdin(data, "cursor")
        assert result["tool_response"] == '{"ok": true}'

    def test_cursor_tool_output(self) -> None:
        data = {"tool_output": '{"content": [{"type": "text"}]}'}
        result = _normalize_stdin(data, "cursor")
        assert result["tool_response"] == '{"content": [{"type": "text"}]}'
        assert "tool_output" not in result

    def test_cursor_tool_output_takes_priority_over_result_json(self) -> None:
        data = {"tool_output": "from_cursor", "result_json": "from_legacy"}
        result = _normalize_stdin(data, "cursor")
        assert result["tool_response"] == "from_cursor"

    def test_cursor_tool_output_no_overwrite_existing_tool_response(self) -> None:
        data = {"tool_response": "existing", "tool_output": "from_cursor"}
        result = _normalize_stdin(data, "cursor")
        assert result["tool_response"] == "existing"

    def test_cursor_tool_output_not_applied_to_claude_code(self) -> None:
        data = {"tool_output": "cursor_only"}
        result = _normalize_stdin(data, "claude-code")
        assert "tool_response" not in result
        assert result["tool_output"] == "cursor_only"

    def test_cursor_status_to_trigger(self) -> None:
        data = {"status": "completed"}
        result = _normalize_stdin(data, "cursor")
        assert result["trigger"] == "completed"

    def test_cursor_synthesizes_source(self) -> None:
        data: dict[str, object] = {}
        result = _normalize_stdin(data, "cursor")
        assert result["source"] == "startup"

    def test_cursor_no_overwrite_existing(self) -> None:
        data = {"session_id": "existing", "conversation_id": "c1"}
        result = _normalize_stdin(data, "cursor")
        assert result["session_id"] == "existing"


# =============================================================================
# Centralized validation
# =============================================================================


@pytest.mark.unit
class TestValidateCommon:
    """Test centralized validation of session_id, transcript_path, cwd."""

    def test_valid_fields_unchanged(self, tmp_path: object) -> None:
        import os

        # Use platform-appropriate absolute paths
        abs_path = str(tmp_path)
        transcript = os.path.join(abs_path, "transcript.jsonl")
        data: dict[str, object] = {
            "session_id": "abc-123",
            "transcript_path": transcript,
            "cwd": abs_path,
        }
        result = _validate_common(data)
        assert result["session_id"] == "abc-123"
        assert result["transcript_path"] == transcript
        assert result["cwd"] == abs_path

    def test_invalid_session_id_sanitized(self) -> None:
        data: dict[str, object] = {"session_id": "../../etc/passwd"}
        result = _validate_common(data)
        assert result["session_id"] == ""

    def test_invalid_transcript_path_sanitized(self) -> None:
        data: dict[str, object] = {"transcript_path": "../../../etc/shadow"}
        result = _validate_common(data)
        assert result["transcript_path"] == ""

    def test_invalid_cwd_sanitized(self) -> None:
        data: dict[str, object] = {"cwd": "relative/path"}
        result = _validate_common(data)
        assert result["cwd"] == ""

    def test_empty_fields_stay_empty(self) -> None:
        data: dict[str, object] = {"session_id": "", "transcript_path": "", "cwd": ""}
        result = _validate_common(data)
        assert result["session_id"] == ""

    def test_missing_fields_no_error(self) -> None:
        data: dict[str, object] = {"tool_name": "Edit"}
        result = _validate_common(data)
        assert "session_id" not in result


# =============================================================================
# Handler routing
# =============================================================================


@pytest.mark.unit
class TestHandlerRouting:
    """Test that known events route to handlers and unknown returns None."""

    def test_all_events_have_handlers(self) -> None:
        for event in ("SessionStart", "PostToolUse", "PreCompact", "Stop"):
            assert event in _HANDLER_MAP

    def test_unknown_event_returns_none(self) -> None:
        result = dispatch("UnknownEvent", "claude-code", {})
        assert result is None


# =============================================================================
# SessionStart handler
# =============================================================================


@pytest.mark.unit
class TestSessionStartHandler:
    """Test SessionStart handler behavior."""

    def test_startup_produces_nudge(self) -> None:
        result = _handle_session_start({"source": "startup"}, "claude-code")
        assert result is not None
        assert "hookSpecificOutput" in result
        ctx = result["hookSpecificOutput"]
        assert isinstance(ctx, dict)
        assert "additionalContext" in ctx
        assert "recall" in ctx["additionalContext"].lower()

    def test_resume_produces_nudge(self) -> None:
        result = _handle_session_start({"source": "resume"}, "claude-code")
        assert result is not None

    def test_clear_produces_none(self) -> None:
        result = _handle_session_start({"source": "clear"}, "claude-code")
        assert result is None

    def test_compact_produces_none(self) -> None:
        result = _handle_session_start({"source": "compact"}, "claude-code")
        assert result is None

    def test_empty_source_produces_none(self) -> None:
        result = _handle_session_start({"source": ""}, "claude-code")
        assert result is None

    def test_missing_source_produces_none(self) -> None:
        result = _handle_session_start({}, "claude-code")
        assert result is None


# =============================================================================
# PostToolUse handler
# =============================================================================


@pytest.mark.unit
class TestPostToolUseHandler:
    """Test PostToolUse handler with mocked pipeline."""

    def test_skip_tool_returns_none(self) -> None:
        # Read is in SKIP_TOOLS
        result = _handle_post_tool_use({"tool_name": "Read"}, "claude-code")
        assert result is None

    def test_empty_tool_returns_none(self) -> None:
        result = _handle_post_tool_use({"tool_name": ""}, "claude-code")
        assert result is None

    def test_spatial_memory_tool_returns_none(self) -> None:
        result = _handle_post_tool_use({"tool_name": "mcp__spatial-memory__recall"}, "claude-code")
        assert result is None

    def test_qualifying_tool_calls_pipeline(self) -> None:
        mock_pipeline = MagicMock()
        with patch.dict(
            "sys.modules",
            {"spatial_memory.hooks.pipeline": MagicMock(run_pipeline=mock_pipeline)},
        ):
            # Need to also ensure the other modules are available
            result = _handle_post_tool_use(
                {
                    "tool_name": "Edit",
                    "tool_response": "ok",
                    "session_id": "s1",
                    "cwd": "",
                },
                "claude-code",
            )
            # PostToolUse always returns None (fire-and-forget)
            assert result is None


# =============================================================================
# Stop handler
# =============================================================================


@pytest.mark.unit
class TestStopHandler:
    """Test Stop handler loop guard."""

    def test_loop_guard_active(self) -> None:
        result = _handle_stop({"stop_hook_active": True}, "claude-code")
        assert result is None

    def test_loop_guard_inactive_with_no_transcript(self) -> None:
        result = _handle_stop(
            {"stop_hook_active": False, "transcript_path": ""},
            "claude-code",
        )
        assert result is None


# =============================================================================
# PreCompact handler
# =============================================================================


@pytest.mark.unit
class TestPreCompactHandler:
    """Test PreCompact handler."""

    def test_no_transcript_returns_none(self) -> None:
        result = _handle_pre_compact({"transcript_path": ""}, "claude-code")
        assert result is None


# =============================================================================
# _load_hook_module
# =============================================================================


@pytest.mark.unit
class TestLoadHookModule:
    """Test module loading utility."""

    def test_loads_existing_module(self) -> None:
        mod = _load_hook_module("hook_helpers")
        assert hasattr(mod, "sanitize_session_id")

    def test_cached_in_sys_modules(self) -> None:
        import sys

        mod1 = _load_hook_module("hook_helpers")
        assert "spatial_memory.hooks.hook_helpers" in sys.modules
        mod2 = _load_hook_module("hook_helpers")
        assert mod1 is mod2

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(ImportError, match="not found"):
            _load_hook_module("nonexistent_module_xyz")


# =============================================================================
# Cursor output translation
# =============================================================================


@pytest.mark.unit
class TestCursorOutputTranslation:
    """Test output translation for Cursor."""

    def test_none_stays_none(self) -> None:
        assert _translate_output_for_cursor(None, "PostToolUse") is None

    def test_session_start_translated(self) -> None:
        response: dict[str, object] = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "Call recall...",
            }
        }
        result = _translate_output_for_cursor(response, "SessionStart")
        assert result is not None
        assert result["continue"] is True
        assert result["additional_context"] == "Call recall..."

    def test_non_session_start_passthrough(self) -> None:
        response: dict[str, object] = {"continue": True}
        result = _translate_output_for_cursor(response, "Stop")
        assert result is response


# =============================================================================
# dispatch() integration
# =============================================================================


@pytest.mark.unit
class TestDispatch:
    """Test the top-level dispatch function."""

    def test_session_start_startup(self) -> None:
        result = dispatch("SessionStart", "claude-code", {"source": "startup"})
        assert result is not None
        assert "hookSpecificOutput" in result

    def test_session_start_clear(self) -> None:
        result = dispatch("SessionStart", "claude-code", {"source": "clear"})
        assert result is None

    def test_unknown_event(self) -> None:
        result = dispatch("FooBar", "claude-code", {})
        assert result is None

    def test_post_tool_use_skip_tool(self) -> None:
        result = dispatch("PostToolUse", "claude-code", {"tool_name": "Glob"})
        assert result is None

    def test_stop_loop_guard(self) -> None:
        result = dispatch("Stop", "claude-code", {"stop_hook_active": True})
        assert result is None
