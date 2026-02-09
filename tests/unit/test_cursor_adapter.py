"""Unit tests for spatial_memory.hooks.cursor_adapter.

Tests cover:
1. Stdin field translation (conversation_id -> session_id)
2. Event name translation (camelCase -> PascalCase)
3. SessionStart source synthesis
4. SessionStart output translation
5. Informational hook pass-through
6. Unknown hook name handling
7. Empty/invalid stdin fail-open behavior
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.hooks.cursor_adapter import (
    _EVENT_MAP,
    _HOOK_SCRIPTS,
    main,
    translate_input,
    translate_session_start_output,
)

# =============================================================================
# translate_input
# =============================================================================


@pytest.mark.unit
class TestTranslateInput:
    """Test Cursor -> Claude Code input translation."""

    def test_conversation_id_becomes_session_id(self) -> None:
        data = {"conversation_id": "conv-123", "tool_name": "Edit"}
        result = translate_input(data, "post_tool_use")
        assert result["session_id"] == "conv-123"
        assert "conversation_id" not in result

    def test_session_id_preserved_if_no_conversation_id(self) -> None:
        data = {"session_id": "sess-456", "tool_name": "Edit"}
        result = translate_input(data, "post_tool_use")
        assert result["session_id"] == "sess-456"

    def test_event_name_translated_camel_to_pascal(self) -> None:
        for camel, pascal in _EVENT_MAP.items():
            data = {"hook_event_name": camel}
            result = translate_input(data, "post_tool_use")
            assert result["hook_event_name"] == pascal

    def test_unknown_event_name_preserved(self) -> None:
        data = {"hook_event_name": "customEvent"}
        result = translate_input(data, "post_tool_use")
        assert result["hook_event_name"] == "customEvent"

    def test_session_start_synthesizes_source(self) -> None:
        data = {"conversation_id": "conv-123"}
        result = translate_input(data, "session_start")
        assert result["source"] == "startup"

    def test_session_start_preserves_explicit_source(self) -> None:
        data = {"conversation_id": "conv-123", "source": "resume"}
        result = translate_input(data, "session_start")
        assert result["source"] == "resume"

    def test_non_session_start_no_source_added(self) -> None:
        data = {"conversation_id": "conv-123"}
        result = translate_input(data, "post_tool_use")
        assert "source" not in result

    def test_original_data_not_mutated(self) -> None:
        data = {"conversation_id": "conv-123", "hook_event_name": "postToolUse"}
        original_copy = dict(data)
        translate_input(data, "post_tool_use")
        assert data == original_copy


# =============================================================================
# translate_session_start_output
# =============================================================================


@pytest.mark.unit
class TestTranslateSessionStartOutput:
    """Test Claude Code -> Cursor output translation for SessionStart."""

    def test_translates_additional_context(self) -> None:
        data = {
            "hookSpecificOutput": {
                "additionalContext": "Call recall with context",
            }
        }
        result = translate_session_start_output(data)
        assert result["additional_context"] == "Call recall with context"
        assert result["continue"] is True

    def test_empty_context_omitted(self) -> None:
        data = {"hookSpecificOutput": {"additionalContext": ""}}
        result = translate_session_start_output(data)
        assert "additional_context" not in result
        assert result["continue"] is True

    def test_missing_hook_output(self) -> None:
        result = translate_session_start_output({})
        assert result["continue"] is True
        assert "additional_context" not in result


# =============================================================================
# main() entrypoint
# =============================================================================


@pytest.mark.unit
class TestMain:
    """Test the adapter main() entrypoint."""

    def test_no_args_exits_silently(self) -> None:
        with patch("sys.argv", ["cursor_adapter.py"]):
            main()  # Should not raise

    def test_unknown_hook_name_exits_silently(self) -> None:
        with patch("sys.argv", ["cursor_adapter.py", "unknown_hook"]):
            main()  # Should not raise

    def test_empty_stdin_exits_silently(self) -> None:
        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=""))),
        ):
            main()  # Should not raise

    def test_invalid_json_stdin_exits_silently(self) -> None:
        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value="not-json"))),
        ):
            main()  # Should not raise

    def test_non_dict_json_exits_silently(self) -> None:
        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value="[1,2,3]"))),
        ):
            main()  # Should not raise

    @patch("subprocess.run")
    def test_delegates_to_hook_script(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        stdin_data = json.dumps(
            {
                "conversation_id": "conv-123",
                "hook_event_name": "postToolUse",
                "tool_name": "Edit",
            }
        )

        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
        ):
            main()

        mock_run.assert_called_once()
        call_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args[1].get("input")
        parsed = json.loads(call_input)
        assert parsed["session_id"] == "conv-123"
        assert parsed["hook_event_name"] == "PostToolUse"

    @patch("subprocess.run")
    def test_session_start_output_translated(self, mock_run: MagicMock) -> None:
        claude_output = json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "Call recall now",
                }
            }
        )
        mock_run.return_value = MagicMock(stdout=claude_output, returncode=0)
        stdin_data = json.dumps({"conversation_id": "conv-123"})

        stdout_capture: list[str] = []

        with (
            patch("sys.argv", ["cursor_adapter.py", "session_start"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdout.write = lambda s: stdout_capture.append(s)
            mock_stdout.flush = lambda: None
            main()

        # Find the JSON output (json.dump writes in chunks, so find the part with content)
        full_output = "".join(stdout_capture)
        parsed = json.loads(full_output.strip())
        assert parsed["continue"] is True
        assert parsed["additional_context"] == "Call recall now"

    def test_hook_scripts_map_valid(self) -> None:
        """All mapped hook scripts correspond to known hooks."""
        assert set(_HOOK_SCRIPTS.keys()) == {
            "session_start",
            "post_tool_use",
            "pre_compact",
            "stop",
        }

    # -----------------------------------------------------------------
    # M-6: Subprocess failure scenarios (fail-open behavior)
    # -----------------------------------------------------------------

    @patch("subprocess.run")
    def test_subprocess_timeout_caught_silently(self, mock_run: MagicMock) -> None:
        """TimeoutExpired from subprocess.run() is caught; main() exits 0."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=30)
        stdin_data = json.dumps({"conversation_id": "conv-123"})

        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
        ):
            main()  # Should not raise

    @patch("subprocess.run")
    def test_subprocess_oserror_caught_silently(self, mock_run: MagicMock) -> None:
        """OSError from subprocess.run() is caught; main() exits 0."""
        mock_run.side_effect = OSError("No such file or directory")
        stdin_data = json.dumps({"conversation_id": "conv-123"})

        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
        ):
            main()  # Should not raise

    @patch("subprocess.run")
    def test_subprocess_nonzero_exit_ignored(self, mock_run: MagicMock) -> None:
        """Non-zero exit code from delegated script is silently ignored."""
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        stdin_data = json.dumps({"conversation_id": "conv-123"})

        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
        ):
            main()  # Should not raise

    @patch("subprocess.run")
    def test_session_start_invalid_json_output_caught(self, mock_run: MagicMock) -> None:
        """SessionStart script returns invalid JSON; caught and no output written."""
        mock_run.return_value = MagicMock(stdout="not-valid-json{[", returncode=0)
        stdin_data = json.dumps({"conversation_id": "conv-123"})
        stdout_capture: list[str] = []

        with (
            patch("sys.argv", ["cursor_adapter.py", "session_start"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdout.write = lambda s: stdout_capture.append(s)
            mock_stdout.flush = lambda: None
            main()

        # Invalid JSON silently caught — no translated output written
        assert "".join(stdout_capture) == ""

    # -----------------------------------------------------------------
    # M-8: Informational hook pass-through (elif branch)
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("hook_name", ["post_tool_use", "pre_compact", "stop"])
    @patch("subprocess.run")
    def test_informational_hook_stdout_passthrough(
        self, mock_run: MagicMock, hook_name: str
    ) -> None:
        """Non-session_start hooks pass stdout through without translation."""
        hook_stdout = f'{{"status": "ok from {hook_name}"}}'
        mock_run.return_value = MagicMock(stdout=hook_stdout, returncode=0)
        stdin_data = json.dumps({"conversation_id": "conv-123"})
        stdout_capture: list[str] = []

        with (
            patch("sys.argv", ["cursor_adapter.py", hook_name]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdout.write = lambda s: stdout_capture.append(s)
            mock_stdout.flush = lambda: None
            main()

        assert "".join(stdout_capture) == hook_stdout

    @patch("subprocess.run")
    def test_informational_hook_whitespace_stdout_not_passed(self, mock_run: MagicMock) -> None:
        """Whitespace-only stdout from informational hook is not passed through."""
        mock_run.return_value = MagicMock(stdout="   \n\t  ", returncode=0)
        stdin_data = json.dumps({"conversation_id": "conv-123"})
        stdout_capture: list[str] = []

        with (
            patch("sys.argv", ["cursor_adapter.py", "post_tool_use"]),
            patch("sys.stdin", MagicMock(read=MagicMock(return_value=stdin_data))),
            patch(
                "spatial_memory.hooks.cursor_adapter._HOOKS_DIR",
                MagicMock(**{"__truediv__": lambda self, x: MagicMock(exists=lambda: True)}),
            ),
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdout.write = lambda s: stdout_capture.append(s)
            mock_stdout.flush = lambda: None
            main()

        assert "".join(stdout_capture) == ""
