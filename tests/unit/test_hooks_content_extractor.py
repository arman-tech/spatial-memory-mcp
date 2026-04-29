"""Unit tests for spatial_memory.hooks.content_extractor (US-018 / spec §8.1, D6).

The v2.0 contract replaces three per-field caps (MAX_EXTRACT_LENGTH=10_000,
_FIELD_SOFT_CAP=5_000, _BASH_OUTPUT_CAP=3_000) with a single tail-drop safety
cap of 200_000 characters applied only at the final extraction boundary.

Tests cover:
1. _extract_edit / _extract_write — basic, file_path, missing/empty keys
2. _extract_bash — command+output (incl. 50K output captured intact)
3. _extract_notebook_edit — basic, path, missing keys
4. _extract_mcp_tool — basic, tool name, empty response
5. extract_content — dispatch table + 200K safety cap with '...' suffix
6. _truncate — short unchanged, long truncated with '...'
7. _safe_get — existing key, missing, non-string coercion
8. Drift guard — ``plugin.hooks.content_extractor`` mirrors the spatial_memory copy
"""

from __future__ import annotations

import filecmp
import importlib
import inspect
from pathlib import Path

import pytest

from spatial_memory.hooks.content_extractor import (
    _SAFETY_CAP,
    _extract_bash,
    _extract_edit,
    _extract_mcp_tool,
    _extract_notebook_edit,
    _extract_write,
    _safe_get,
    _truncate,
    extract_content,
)

# =============================================================================
# Module-level invariants (US-018 acceptance criteria 1-4)
# =============================================================================


@pytest.mark.unit
class TestSafetyCapInvariants:
    """The single 200K safety cap replaces all three v1 caps."""

    def test_safety_cap_is_200_000(self) -> None:
        assert _SAFETY_CAP == 200_000

    def test_legacy_caps_removed(self) -> None:
        """v1 per-field caps must not be re-exported."""
        from spatial_memory.hooks import content_extractor as ce

        for legacy in ("MAX_EXTRACT_LENGTH", "_FIELD_SOFT_CAP", "_BASH_OUTPUT_CAP"):
            assert not hasattr(ce, legacy), (
                f"Legacy cap {legacy!r} must not be re-exported (US-018, spec §8.1)"
            )


# =============================================================================
# _truncate
# =============================================================================


@pytest.mark.unit
class TestTruncate:
    """Test _truncate helper."""

    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello", 10) == "hello"

    def test_long_text_truncated(self) -> None:
        result = _truncate("a" * 100, 10)
        assert len(result) == 13  # 10 + "..."
        assert result.endswith("...")

    def test_exact_length_unchanged(self) -> None:
        text = "a" * 10
        assert _truncate(text, 10) == text

    def test_empty_string(self) -> None:
        assert _truncate("", 10) == ""

    def test_one_over_truncated(self) -> None:
        result = _truncate("a" * 11, 10)
        assert result == "a" * 10 + "..."


# =============================================================================
# _safe_get
# =============================================================================


@pytest.mark.unit
class TestSafeGet:
    """Test _safe_get helper."""

    def test_existing_string_key(self) -> None:
        assert _safe_get({"key": "value"}, "key") == "value"

    def test_missing_key_returns_default(self) -> None:
        assert _safe_get({}, "key") == ""

    def test_missing_key_with_custom_default(self) -> None:
        assert _safe_get({}, "key", "fallback") == "fallback"

    def test_non_string_coercion(self) -> None:
        assert _safe_get({"key": 42}, "key") == "42"

    def test_none_value_returns_default(self) -> None:
        assert _safe_get({"key": None}, "key") == ""

    def test_bool_coercion(self) -> None:
        assert _safe_get({"key": True}, "key") == "True"

    def test_list_coercion(self) -> None:
        assert _safe_get({"key": [1, 2]}, "key") == "[1, 2]"


# =============================================================================
# _extract_edit
# =============================================================================


@pytest.mark.unit
class TestExtractEdit:
    """Test Edit tool extractor."""

    def test_basic(self) -> None:
        result = _extract_edit({"new_string": "x = 1"}, "ok")
        assert "x = 1" in result

    def test_with_file_path(self) -> None:
        result = _extract_edit({"file_path": "/src/app.py", "new_string": "x = 1"}, "ok")
        assert "File: /src/app.py" in result
        assert "x = 1" in result

    def test_missing_new_string(self) -> None:
        result = _extract_edit({"file_path": "/src/app.py"}, "ok")
        assert result == ""

    def test_empty_new_string(self) -> None:
        result = _extract_edit({"new_string": ""}, "ok")
        assert result == ""

    def test_50k_new_string_intact(self) -> None:
        """Per-field caps removed: 50K Edit content is preserved at the field level."""
        long_content = "x" * 50_000
        result = _extract_edit({"new_string": long_content}, "ok")
        assert long_content in result, "50K new_string must not be field-truncated"


# =============================================================================
# _extract_write
# =============================================================================


@pytest.mark.unit
class TestExtractWrite:
    """Test Write tool extractor."""

    def test_basic(self) -> None:
        result = _extract_write({"content": "hello world"}, "ok")
        assert "hello world" in result

    def test_with_file_path(self) -> None:
        result = _extract_write({"file_path": "/src/new.py", "content": "print('hi')"}, "ok")
        assert "File: /src/new.py" in result
        assert "print('hi')" in result

    def test_missing_content(self) -> None:
        result = _extract_write({"file_path": "/src/new.py"}, "ok")
        assert result == ""

    def test_empty_content(self) -> None:
        result = _extract_write({"content": ""}, "ok")
        assert result == ""

    def test_50k_content_intact(self) -> None:
        long_content = "y" * 50_000
        result = _extract_write({"content": long_content}, "ok")
        assert long_content in result


# =============================================================================
# _extract_bash
# =============================================================================


@pytest.mark.unit
class TestExtractBash:
    """Test Bash tool extractor."""

    def test_command_and_output(self) -> None:
        result = _extract_bash({"command": "echo hello"}, "hello")
        assert "Command: echo hello" in result
        assert "Output: hello" in result

    def test_command_only(self) -> None:
        result = _extract_bash({"command": "ls"}, "")
        assert "Command: ls" in result
        assert "Output" not in result

    def test_missing_command(self) -> None:
        result = _extract_bash({}, "some output")
        assert result == ""

    def test_50k_bash_output_captured_intact(self) -> None:
        """Acceptance criterion 5: 50K-char Bash output preserved (was truncated to 3K in v1)."""
        big_output = "z" * 50_000
        result = _extract_bash({"command": "cat big.log"}, big_output)
        assert "Command: cat big.log" in result
        assert big_output in result, (
            "50K Bash output must NOT be field-truncated under US-018 (was 3K in v1)"
        )

    def test_empty_command(self) -> None:
        result = _extract_bash({"command": ""}, "output")
        assert result == ""


# =============================================================================
# _extract_notebook_edit
# =============================================================================


@pytest.mark.unit
class TestExtractNotebookEdit:
    """Test NotebookEdit tool extractor."""

    def test_basic(self) -> None:
        result = _extract_notebook_edit({"new_source": "import pandas"}, "ok")
        assert "import pandas" in result

    def test_with_path(self) -> None:
        result = _extract_notebook_edit(
            {"notebook_path": "/nb/analysis.ipynb", "new_source": "df.head()"}, "ok"
        )
        assert "Notebook: /nb/analysis.ipynb" in result
        assert "df.head()" in result

    def test_missing_new_source(self) -> None:
        result = _extract_notebook_edit({"notebook_path": "/nb/x.ipynb"}, "ok")
        assert result == ""

    def test_empty_new_source(self) -> None:
        result = _extract_notebook_edit({"new_source": ""}, "ok")
        assert result == ""


# =============================================================================
# _extract_mcp_tool
# =============================================================================


@pytest.mark.unit
class TestExtractMcpTool:
    """Test MCP tool extractor."""

    def test_basic(self) -> None:
        result = _extract_mcp_tool({}, "some response", tool_name="mcp__db__query")
        assert "Tool: mcp__db__query" in result
        assert "Response: some response" in result

    def test_no_tool_name(self) -> None:
        result = _extract_mcp_tool({}, "response")
        assert "Tool:" not in result
        assert "Response: response" in result

    def test_empty_response(self) -> None:
        result = _extract_mcp_tool({}, "", tool_name="mcp__db__query")
        assert "Tool: mcp__db__query" in result
        assert "Response" not in result


# =============================================================================
# extract_content (dispatch)
# =============================================================================


@pytest.mark.unit
class TestExtractContent:
    """Test public extract_content dispatch + safety cap."""

    def test_edit_dispatch(self) -> None:
        result = extract_content("Edit", {"new_string": "code"}, "ok")
        assert "code" in result

    def test_write_dispatch(self) -> None:
        result = extract_content("Write", {"content": "data"}, "ok")
        assert "data" in result

    def test_bash_dispatch(self) -> None:
        result = extract_content("Bash", {"command": "echo hi"}, "hi")
        assert "Command: echo hi" in result

    def test_notebook_edit_dispatch(self) -> None:
        result = extract_content("NotebookEdit", {"new_source": "code"}, "ok")
        assert "code" in result

    def test_mcp_tool_dispatch(self) -> None:
        result = extract_content("mcp__db__query", {}, "results")
        assert "Tool: mcp__db__query" in result
        assert "Response: results" in result

    def test_unknown_tool_returns_empty(self) -> None:
        result = extract_content("UnknownTool", {"x": "y"}, "response")
        assert result == ""

    def test_250k_input_truncated_to_safety_cap(self) -> None:
        """Acceptance criterion 6: 250K input truncates to ~200K with '...' suffix."""
        huge_content = "a" * 250_000
        result = extract_content("Write", {"content": huge_content}, "ok")
        # _SAFETY_CAP + len(prefix "File: ...\n") + "..." trail
        assert len(result) <= _SAFETY_CAP + 10, (
            f"Result {len(result)} exceeds _SAFETY_CAP+10 ({_SAFETY_CAP + 10})"
        )
        assert result.endswith("...")

    def test_under_safety_cap_passes_through(self) -> None:
        """Anything below 200K must not be truncated by extract_content."""
        content = "b" * 199_000
        result = extract_content("Write", {"content": content}, "ok")
        assert content in result
        assert not result.endswith("...")

    def test_non_string_tool_response_coerced(self) -> None:
        result = extract_content("Bash", {"command": "echo"}, {"success": True})  # type: ignore[arg-type]
        assert "Command: echo" in result

    def test_none_tool_response(self) -> None:
        result = extract_content("Bash", {"command": "echo"}, None)  # type: ignore[arg-type]
        assert "Command: echo" in result


# =============================================================================
# Mirror integrity — plugin.hooks.content_extractor must equal the package copy
# =============================================================================


@pytest.mark.unit
class TestMirrorIntegrity:
    """Acceptance criteria 1-4: both content_extractor.py copies must be identical."""

    def test_files_byte_identical(self) -> None:
        spatial = Path("spatial_memory/hooks/content_extractor.py")
        plugin = Path("plugin/hooks/content_extractor.py")
        assert spatial.exists() and plugin.exists()
        assert filecmp.cmp(spatial, plugin, shallow=False), (
            "plugin/hooks/content_extractor.py must mirror "
            "spatial_memory/hooks/content_extractor.py byte-for-byte"
        )

    def test_plugin_module_imports_and_exports_safety_cap(self) -> None:
        plugin_ce = importlib.import_module("plugin.hooks.content_extractor")
        assert plugin_ce._SAFETY_CAP == 200_000
        assert not hasattr(plugin_ce, "MAX_EXTRACT_LENGTH")
        assert not hasattr(plugin_ce, "_FIELD_SOFT_CAP")
        assert not hasattr(plugin_ce, "_BASH_OUTPUT_CAP")

    def test_per_tool_extractors_are_two_arg(self) -> None:
        """Acceptance: per-tool extractors drop their individual cap arguments."""
        for fn in (_extract_edit, _extract_write, _extract_bash, _extract_notebook_edit):
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values() if p.kind != inspect.Parameter.KEYWORD_ONLY
            ]
            assert len(params) == 2, (
                f"{fn.__name__} must take exactly two positional args (tool_input, tool_response); "
                f"found {[p.name for p in sig.parameters.values()]}"
            )
