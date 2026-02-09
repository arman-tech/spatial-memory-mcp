"""Unit tests for spatial_memory.hooks.content_extractor.

Tests cover:
1. _extract_edit — basic, file_path, missing/empty keys, truncation
2. _extract_write — basic, file_path, truncation
3. _extract_bash — command+output, large output truncation, empty
4. _extract_notebook_edit — basic, path
5. _extract_mcp_tool — basic, tool name, empty response
6. extract_content — dispatch for each tool type, unknown returns empty, total cap
7. _truncate — short unchanged, long truncated, exact length
8. _safe_get — existing key, missing, non-string coercion
"""

from __future__ import annotations

import pytest

from spatial_memory.hooks.content_extractor import (
    _BASH_OUTPUT_CAP,
    _FIELD_SOFT_CAP,
    MAX_EXTRACT_LENGTH,
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

    def test_truncation(self) -> None:
        long_content = "x" * (_FIELD_SOFT_CAP + 100)
        result = _extract_edit({"new_string": long_content}, "ok")
        assert len(result) <= _FIELD_SOFT_CAP + 10  # + "..." + margin


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

    def test_truncation(self) -> None:
        long_content = "y" * (_FIELD_SOFT_CAP + 200)
        result = _extract_write({"content": long_content}, "ok")
        assert len(result) <= _FIELD_SOFT_CAP + 10


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

    def test_large_output_truncated(self) -> None:
        big_output = "z" * (_BASH_OUTPUT_CAP + 500)
        result = _extract_bash({"command": "cat big.log"}, big_output)
        assert "Command: cat big.log" in result
        # Output should be capped
        output_line = [line for line in result.split("\n") if line.startswith("Output:")][0]
        # Cap + "Output: " prefix + "..."
        assert len(output_line) < _BASH_OUTPUT_CAP + 20

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
    """Test public extract_content dispatch function."""

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

    def test_total_cap_enforced(self) -> None:
        huge_content = "a" * (MAX_EXTRACT_LENGTH + 1000)
        result = extract_content("Write", {"content": huge_content}, "ok")
        assert len(result) <= MAX_EXTRACT_LENGTH + 10  # + "..." margin

    def test_non_string_tool_response_coerced(self) -> None:
        result = extract_content("Bash", {"command": "echo"}, {"success": True})  # type: ignore[arg-type]
        assert "Command: echo" in result

    def test_none_tool_response(self) -> None:
        result = extract_content("Bash", {"command": "echo"}, None)  # type: ignore[arg-type]
        assert "Command: echo" in result
