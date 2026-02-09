"""Unit tests for spatial_memory.hooks.models.

Tests cover:
1. HookInput — defaults, complete construction, frozen immutability
2. should_skip_tool — all SKIP_TOOLS, spatial-memory tools, processable tools
3. ProcessingResult — defaults, mutation
4. Constants — frozenset type, expected contents, prefix value
"""

from __future__ import annotations

import pytest

from spatial_memory.hooks.models import (
    SKIP_TOOLS,
    SPATIAL_MEMORY_PREFIX,
    HookInput,
    ProcessingResult,
    should_skip_tool,
)

# =============================================================================
# HookInput
# =============================================================================


@pytest.mark.unit
class TestHookInput:
    """Test HookInput frozen dataclass."""

    def test_defaults(self) -> None:
        h = HookInput()
        assert h.session_id == ""
        assert h.tool_name == ""
        assert h.tool_input == {}
        assert h.tool_response == ""
        assert h.tool_use_id == ""
        assert h.transcript_path == ""
        assert h.cwd == ""
        assert h.hook_event_name == ""
        assert h.permission_mode == ""

    def test_complete_construction(self) -> None:
        h = HookInput(
            session_id="sess-1",
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            tool_response="ok",
            tool_use_id="toolu_123",
            transcript_path="/tmp/transcript.jsonl",
            cwd="/home/user/project",
            hook_event_name="PostToolUse",
            permission_mode="default",
        )
        assert h.session_id == "sess-1"
        assert h.tool_name == "Edit"
        assert h.tool_input == {"file_path": "/src/app.py"}
        assert h.tool_response == "ok"
        assert h.tool_use_id == "toolu_123"
        assert h.transcript_path == "/tmp/transcript.jsonl"
        assert h.cwd == "/home/user/project"
        assert h.hook_event_name == "PostToolUse"
        assert h.permission_mode == "default"

    def test_frozen_immutability(self) -> None:
        h = HookInput(tool_name="Bash")
        with pytest.raises(AttributeError):
            h.tool_name = "Edit"  # type: ignore[misc]

    def test_frozen_cannot_set_new_attribute(self) -> None:
        h = HookInput()
        with pytest.raises(AttributeError):
            h.new_field = "value"  # type: ignore[attr-defined]

    def test_tool_input_default_is_independent(self) -> None:
        h1 = HookInput()
        h2 = HookInput()
        assert h1.tool_input is not h2.tool_input


# =============================================================================
# ProcessingResult
# =============================================================================


@pytest.mark.unit
class TestProcessingResult:
    """Test ProcessingResult mutable dataclass."""

    def test_defaults(self) -> None:
        r = ProcessingResult()
        assert r.skipped is False
        assert r.skip_reason == ""
        assert r.queued is False
        assert r.queue_path == ""
        assert r.signal_tier == 3
        assert r.signal_score == 0.0
        assert r.patterns_matched == []
        assert r.redaction_count == 0
        assert r.content_length == 0

    def test_mutation(self) -> None:
        r = ProcessingResult()
        r.skipped = True
        r.skip_reason = "test"
        r.signal_tier = 1
        r.signal_score = 0.9
        assert r.skipped is True
        assert r.skip_reason == "test"
        assert r.signal_tier == 1
        assert r.signal_score == 0.9

    def test_patterns_matched_default_is_independent(self) -> None:
        r1 = ProcessingResult()
        r2 = ProcessingResult()
        r1.patterns_matched.append("decision")
        assert r2.patterns_matched == []


# =============================================================================
# should_skip_tool
# =============================================================================


@pytest.mark.unit
class TestShouldSkipTool:
    """Test tool filtering logic."""

    @pytest.mark.parametrize("tool_name", sorted(SKIP_TOOLS))
    def test_skip_tools_are_skipped(self, tool_name: str) -> None:
        skip, reason = should_skip_tool(tool_name)
        assert skip is True
        assert f"skip_tool:{tool_name}" == reason

    def test_empty_tool_name_skipped(self) -> None:
        skip, reason = should_skip_tool("")
        assert skip is True
        assert reason == "empty_tool_name"

    @pytest.mark.parametrize(
        "tool_name",
        [
            "mcp__spatial-memory__remember",
            "mcp__spatial-memory__recall",
            "mcp__spatial-memory__hybrid_recall",
            "mcp__spatial-memory__nearby",
            "mcp__spatial-memory__forget",
            "mcp__spatial-memory__forget_batch",
            "mcp__spatial-memory__journey",
            "mcp__spatial-memory__wander",
            "mcp__spatial-memory__regions",
            "mcp__spatial-memory__visualize",
            "mcp__spatial-memory__decay",
            "mcp__spatial-memory__reinforce",
            "mcp__spatial-memory__extract",
            "mcp__spatial-memory__consolidate",
            "mcp__spatial-memory__health",
            "mcp__spatial-memory__stats",
            "mcp__spatial-memory__namespaces",
            "mcp__spatial-memory__delete_namespace",
            "mcp__spatial-memory__rename_namespace",
            "mcp__spatial-memory__export_memories",
            "mcp__spatial-memory__import_memories",
            "mcp__spatial-memory__remember_batch",
        ],
    )
    def test_all_spatial_memory_tools_skipped(self, tool_name: str) -> None:
        skip, reason = should_skip_tool(tool_name)
        assert skip is True
        assert reason == "spatial_memory_tool"

    @pytest.mark.parametrize(
        "tool_name",
        ["Edit", "Write", "Bash", "NotebookEdit", "mcp__other__something"],
    )
    def test_processable_tools_not_skipped(self, tool_name: str) -> None:
        skip, reason = should_skip_tool(tool_name)
        assert skip is False
        assert reason == ""


# =============================================================================
# Constants
# =============================================================================


@pytest.mark.unit
class TestConstants:
    """Test module-level constants."""

    def test_skip_tools_is_frozenset(self) -> None:
        assert isinstance(SKIP_TOOLS, frozenset)

    def test_skip_tools_expected_contents(self) -> None:
        expected = {"Read", "Glob", "Grep", "LSP", "WebSearch", "WebFetch", "Task"}
        assert SKIP_TOOLS == expected

    def test_spatial_memory_prefix(self) -> None:
        assert SPATIAL_MEMORY_PREFIX == "mcp__spatial-memory__"
