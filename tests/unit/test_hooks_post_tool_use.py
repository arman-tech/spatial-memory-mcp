"""Unit tests for spatial_memory.hooks.post_tool_use.

Tests cover:
1. _read_stdin — valid JSON, empty, invalid JSON, non-dict
2. _write_stdout — continue:true, suppressOutput:true, valid JSON
3. _get_project_root — env var set/unset/empty
4. _load_hook_module — loads each module, nonexistent raises
5. main() fail-open — empty stdin, invalid JSON, pipeline exception, import error
6. End-to-end — Edit with decision queues, Bash trivial skips, Read skips,
   spatial-memory skips, AWS key redacted, private key causes skip
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spatial_memory.hooks.post_tool_use import (
    _get_project_root,
    _load_hook_module,
    _read_stdin,
    _write_stdout_and_exit,
    main,
)

# =============================================================================
# _read_stdin
# =============================================================================


@pytest.mark.unit
class TestReadStdin:
    """Test stdin JSON parsing."""

    def test_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"tool_name": "Edit", "tool_response": "ok"}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(data)))
        result = _read_stdin()
        assert result["tool_name"] == "Edit"

    def test_empty_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        result = _read_stdin()
        assert result == {}

    def test_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        result = _read_stdin()
        assert result == {}

    def test_non_dict_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2, 3]"))
        result = _read_stdin()
        assert result == {}

    def test_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("   \n  "))
        result = _read_stdin()
        assert result == {}


# =============================================================================
# _write_stdout_and_exit
# =============================================================================


@pytest.mark.unit
class TestWriteStdout:
    """Test stdout JSON output."""

    def test_output_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        _write_stdout_and_exit()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True
        assert output["suppressOutput"] is True

    def test_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        _write_stdout_and_exit()
        # Should not raise
        json.loads(buf.getvalue().strip())


# =============================================================================
# _get_project_root
# =============================================================================


@pytest.mark.unit
class TestGetProjectRoot:
    """Test project root resolution from env."""

    def test_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/my/project")
        assert _get_project_root() == "/my/project"

    def test_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        assert _get_project_root() == ""

    def test_env_var_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "")
        assert _get_project_root() == ""


# =============================================================================
# _load_hook_module
# =============================================================================


@pytest.mark.unit
class TestLoadHookModule:
    """Test importlib-based module loading."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "models",
            "content_extractor",
            "pipeline",
            "signal_detection",
            "redaction",
            "queue_writer",
        ],
    )
    def test_loads_existing_modules(self, module_name: str) -> None:
        mod = _load_hook_module(module_name)
        assert mod is not None

    def test_nonexistent_module_raises(self) -> None:
        with pytest.raises(ImportError, match="not found"):
            _load_hook_module("nonexistent_module_xyz")

    def test_loaded_module_has_expected_attr(self) -> None:
        mod = _load_hook_module("models")
        assert hasattr(mod, "HookInput")
        assert hasattr(mod, "should_skip_tool")


# =============================================================================
# main() fail-open
# =============================================================================


@pytest.mark.unit
class TestMainFailOpen:
    """Test that main() always exits cleanly (never raises)."""

    def test_empty_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        # Should not raise
        main()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True

    def test_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("{bad json"))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        main()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True

    def test_pipeline_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even if the pipeline raises, main() should not."""
        data = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {"new_string": "decided to use X"},
                "tool_response": "ok",
            }
        )
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        # Patch _load_hook_module to raise on pipeline
        original_load = _load_hook_module

        def _patched_load(name: str):
            if name == "pipeline":
                mod = MagicMock()
                mod.run_pipeline.side_effect = RuntimeError("boom")
                return mod
            return original_load(name)

        monkeypatch.setattr("spatial_memory.hooks.post_tool_use._load_hook_module", _patched_load)
        main()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True

    def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even if module loading fails, main() should not raise."""
        data = json.dumps({"tool_name": "Edit", "tool_response": "ok"})
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        def _bad_load(name: str):
            raise ImportError("module missing")

        monkeypatch.setattr("spatial_memory.hooks.post_tool_use._load_hook_module", _bad_load)
        main()
        output = json.loads(buf.getvalue().strip())
        assert output["continue"] is True


# =============================================================================
# End-to-end tests
# =============================================================================


@pytest.mark.unit
class TestEndToEnd:
    """Integration-style tests using real Phase A modules via importlib."""

    def _run_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tool_name: str,
        tool_input: dict,
        tool_response: str,
        tmp_path: Path,
    ) -> tuple[str, list[Path]]:
        """Run main() with given input and return (stdout, queue_files)."""
        data = json.dumps(
            {
                "session_id": "test-sess",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_response": tool_response,
                "hook_event_name": "PostToolUse",
            }
        )
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))

        main()

        new_dir = tmp_path / "pending-saves" / "new"
        queue_files = sorted(new_dir.glob("*.json")) if new_dir.exists() else []
        return buf.getvalue(), queue_files

    def test_edit_with_decision_queued(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Edit with decision-signal content should produce a queue file."""
        stdout, files = self._run_main(
            monkeypatch,
            tool_name="Edit",
            tool_input={
                "file_path": "/src/db.py",
                "new_string": "Decided to use PostgreSQL because it handles JSONB well.",
            },
            tool_response="ok",
            tmp_path=tmp_path,
        )
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["signal_tier"] == 1
        assert payload["source_hook"] == "PostToolUse"
        assert "decision" in payload["signal_patterns_matched"]
        assert payload["suggested_namespace"] == "decisions"
        assert payload["client"] == "claude-code"

    def test_edit_with_decision_round_trips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Queue file should parse via QueueFile.from_json()."""
        from spatial_memory.core.queue_file import QueueFile

        _, files = self._run_main(
            monkeypatch,
            tool_name="Edit",
            tool_input={
                "file_path": "/src/db.py",
                "new_string": "The fix is to add the missing import statement.",
            },
            tool_response="ok",
            tmp_path=tmp_path,
        )
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        qf = QueueFile.from_json(payload)
        assert qf.content
        assert qf.signal_tier in (1, 2)

    def test_bash_trivial_skips(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Bash with no signal content should not queue."""
        _, files = self._run_main(
            monkeypatch,
            tool_name="Bash",
            tool_input={"command": "ls -la"},
            tool_response="total 42\ndrwxr-xr-x ...",
            tmp_path=tmp_path,
        )
        assert len(files) == 0

    def test_read_tool_skips(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Read tool should be skipped entirely."""
        _, files = self._run_main(
            monkeypatch,
            tool_name="Read",
            tool_input={"file_path": "/src/app.py"},
            tool_response="file contents...",
            tmp_path=tmp_path,
        )
        assert len(files) == 0

    def test_spatial_memory_tool_skips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Spatial-memory MCP tools should be skipped."""
        _, files = self._run_main(
            monkeypatch,
            tool_name="mcp__spatial-memory__remember",
            tool_input={"content": "test"},
            tool_response="stored",
            tmp_path=tmp_path,
        )
        assert len(files) == 0

    def test_aws_key_redacted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """AWS key in content should be redacted before queuing."""
        _, files = self._run_main(
            monkeypatch,
            tool_name="Bash",
            tool_input={"command": "cat .env"},
            tool_response=("The fix is to set AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF in .env"),
            tmp_path=tmp_path,
        )
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert "AKIA1234567890ABCDEF" not in payload["content"]
        assert "[REDACTED_AWS_KEY]" in payload["content"]

    def test_private_key_causes_skip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Private key in content should trigger redaction skip."""
        private_key = (
            "The fix is to use this key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbIkG1HbqP/MNW3Lzv2gU...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        _, files = self._run_main(
            monkeypatch,
            tool_name="Bash",
            tool_input={"command": "cat key.pem"},
            tool_response=private_key,
            tmp_path=tmp_path,
        )
        assert len(files) == 0

    def test_stdout_always_valid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """stdout should always contain valid JSON with continue:true."""
        stdout, _ = self._run_main(
            monkeypatch,
            tool_name="Edit",
            tool_input={"new_string": "hello"},
            tool_response="ok",
            tmp_path=tmp_path,
        )
        output = json.loads(stdout.strip())
        assert output["continue"] is True
        assert output["suppressOutput"] is True

    def test_write_with_solution_queued(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Write tool with solution-signal content should queue."""
        _, files = self._run_main(
            monkeypatch,
            tool_name="Write",
            tool_input={
                "file_path": "/docs/fix.md",
                "content": "The solution is to increase the timeout to 30 seconds.",
            },
            tool_response="ok",
            tmp_path=tmp_path,
        )
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert "solution" in payload["signal_patterns_matched"]
        assert payload["suggested_namespace"] == "troubleshooting"
