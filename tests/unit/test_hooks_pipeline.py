"""Unit tests for spatial_memory.hooks.pipeline.

Uses mock callables for all injected dependencies (no real signal_detection/
redaction/queue_writer).

Tests cover:
1. Pipeline skip filters — read-only, spatial-memory, empty tool name
2. Pipeline extraction — empty/whitespace skip, valid proceeds
3. Pipeline signal gating — tier 3 skipped, tier 1/2 proceed
4. Pipeline redaction gating — should_skip stops, clean proceeds
5. Pipeline queue write — correct args passed
6. Pipeline result — skipped/queued fields populated
7. score_to_importance — tier 1/2 mappings, bounds
8. derive_namespace — each pattern type, priority, valid format
9. _build_context — tool_name, session_id, file_path conditional
10. _derive_tags — tool name, patterns
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest

from spatial_memory.hooks.models import HookInput
from spatial_memory.hooks.pipeline import (
    _build_context,
    _derive_tags,
    derive_namespace,
    run_pipeline,
    score_to_importance,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSignalResult(NamedTuple):
    tier: int
    score: float
    patterns_matched: list[str]


class FakeRedactionResult(NamedTuple):
    redacted_text: str
    redaction_count: int
    should_skip: bool


def _make_classify(tier: int = 1, score: float = 0.9, patterns: list[str] | None = None):
    """Return a classify_fn that always returns the given signal."""
    result = FakeSignalResult(tier=tier, score=score, patterns_matched=patterns or ["decision"])
    return MagicMock(return_value=result)


def _make_redact(text: str = "clean", count: int = 0, skip: bool = False):
    """Return a redact_fn that always returns the given redaction."""
    result = FakeRedactionResult(redacted_text=text, redaction_count=count, should_skip=skip)
    return MagicMock(return_value=result)


def _make_extract(content: str = "Decided to use PostgreSQL"):
    """Return an extract_fn that always returns the given content."""
    return MagicMock(return_value=content)


def _make_write(path: str = "/queue/new/file.json"):
    """Return a write_fn that always returns a Path."""
    return MagicMock(return_value=Path(path))


def _default_hook_input(**overrides) -> HookInput:
    defaults = {
        "session_id": "sess-1",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/src/app.py", "new_string": "x = 1"},
        "tool_response": "ok",
    }
    defaults.update(overrides)
    return HookInput(**defaults)


# =============================================================================
# Pipeline skip filters
# =============================================================================


@pytest.mark.unit
class TestPipelineSkipFilter:
    """Test that filtered tools produce skipped results."""

    def test_read_tool_skipped(self) -> None:
        result = run_pipeline(
            _default_hook_input(tool_name="Read"),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert "skip_tool:Read" in result.skip_reason

    def test_spatial_memory_tool_skipped(self) -> None:
        result = run_pipeline(
            _default_hook_input(tool_name="mcp__spatial-memory__remember"),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "spatial_memory_tool"

    def test_empty_tool_name_skipped(self) -> None:
        result = run_pipeline(
            _default_hook_input(tool_name=""),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "empty_tool_name"

    def test_glob_skipped(self) -> None:
        result = run_pipeline(
            _default_hook_input(tool_name="Glob"),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True

    def test_skip_does_not_call_extract(self) -> None:
        extract = _make_extract()
        run_pipeline(
            _default_hook_input(tool_name="Read"),
            extract_fn=extract,
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        extract.assert_not_called()


# =============================================================================
# Pipeline extraction
# =============================================================================


@pytest.mark.unit
class TestPipelineExtraction:
    """Test content extraction step."""

    def test_empty_extraction_skips(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract(""),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "empty_content"

    def test_whitespace_extraction_skips(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("   \n  "),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "empty_content"

    def test_valid_extraction_proceeds(self) -> None:
        classify = _make_classify()
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("valid content"),
            classify_fn=classify,
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        classify.assert_called_once()

    def test_content_length_recorded(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("some content"),
            classify_fn=_make_classify(),
            redact_fn=_make_redact(text="some content"),
            write_fn=_make_write(),
        )
        assert result.content_length == len("some content")


# =============================================================================
# Pipeline signal gating
# =============================================================================


@pytest.mark.unit
class TestPipelineSignalGating:
    """Test tier 3 gating in the pipeline."""

    def test_tier_3_skipped(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("some text"),
            classify_fn=_make_classify(tier=3, score=0.2, patterns=[]),
            redact_fn=_make_redact(),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "tier_3_no_signal"
        assert result.signal_tier == 3

    def test_tier_1_proceeds(self) -> None:
        redact = _make_redact(text="clean")
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided to use X"),
            classify_fn=_make_classify(tier=1, score=0.9),
            redact_fn=redact,
            write_fn=_make_write(),
        )
        redact.assert_called_once()

    def test_tier_2_proceeds(self) -> None:
        redact = _make_redact(text="clean")
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("the convention is X"),
            classify_fn=_make_classify(tier=2, score=0.6),
            redact_fn=redact,
            write_fn=_make_write(),
        )
        redact.assert_called_once()

    def test_signal_result_recorded(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided to use X"),
            classify_fn=_make_classify(tier=1, score=0.85, patterns=["decision"]),
            redact_fn=_make_redact(text="decided to use X"),
            write_fn=_make_write(),
        )
        assert result.signal_tier == 1
        assert result.signal_score == 0.85
        assert result.patterns_matched == ["decision"]


# =============================================================================
# Pipeline redaction gating
# =============================================================================


@pytest.mark.unit
class TestPipelineRedactionGating:
    """Test redaction should_skip gating."""

    def test_should_skip_stops_pipeline(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("secret stuff"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="[REDACTED]", count=5, skip=True),
            write_fn=_make_write(),
        )
        assert result.skipped is True
        assert result.skip_reason == "redaction_skip"
        assert result.redaction_count == 5

    def test_clean_content_proceeds(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("clean content"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="clean content", count=0, skip=False),
            write_fn=write,
        )
        write.assert_called_once()

    def test_redacted_but_not_skipped_proceeds(self) -> None:
        write = _make_write()
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("has key=AKIA1234"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="has key=[REDACTED_AWS_KEY]", count=1, skip=False),
            write_fn=write,
        )
        write.assert_called_once()
        assert result.redaction_count == 1
        assert result.queued is True


# =============================================================================
# Pipeline queue write
# =============================================================================


@pytest.mark.unit
class TestPipelineQueueWrite:
    """Test that queue write receives correct arguments."""

    def test_correct_args_passed(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(tool_name="Bash", session_id="sess-42"),
            extract_fn=_make_extract("decided to use PostgreSQL"),
            classify_fn=_make_classify(tier=1, score=0.9, patterns=["decision"]),
            redact_fn=_make_redact(text="decided to use PostgreSQL"),
            write_fn=write,
            project_root="/home/user/project",
        )
        write.assert_called_once()
        kwargs = write.call_args[1]
        assert kwargs["content"] == "decided to use PostgreSQL"
        assert kwargs["source_hook"] == "PostToolUse"
        assert kwargs["project_root_dir"] == "/home/user/project"
        assert kwargs["client"] == "claude-code"
        assert kwargs["signal_tier"] == 1
        assert kwargs["signal_patterns_matched"] == ["decision"]

    def test_queue_path_recorded(self) -> None:
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided X"),
            write_fn=_make_write("/queue/new/abc.json"),
        )
        assert result.queued is True
        assert result.queue_path == str(Path("/queue/new/abc.json"))

    def test_namespace_derived_from_patterns(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("error text"),
            classify_fn=_make_classify(tier=1, score=0.8, patterns=["error"]),
            redact_fn=_make_redact(text="error text"),
            write_fn=write,
        )
        kwargs = write.call_args[1]
        assert kwargs["suggested_namespace"] == "troubleshooting"

    def test_tags_include_tool_and_patterns(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(tool_name="Bash"),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1, patterns=["decision"]),
            redact_fn=_make_redact(text="decided X"),
            write_fn=write,
        )
        kwargs = write.call_args[1]
        assert "bash" in kwargs["suggested_tags"]
        assert "decision" in kwargs["suggested_tags"]

    def test_context_has_tool_name(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(tool_name="Edit", session_id="s1"),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided X"),
            write_fn=write,
        )
        kwargs = write.call_args[1]
        assert kwargs["context"]["tool_name"] == "Edit"
        assert kwargs["context"]["session_id"] == "s1"


# =============================================================================
# score_to_importance
# =============================================================================


@pytest.mark.unit
class TestScoreToImportance:
    """Test importance scoring."""

    def test_tier_1_high_score(self) -> None:
        assert score_to_importance(0.9, 1) == 0.9

    def test_tier_1_low_score_floors_at_07(self) -> None:
        assert score_to_importance(0.5, 1) == 0.7

    def test_tier_2_high_score(self) -> None:
        result = score_to_importance(0.75, 2)
        assert result == pytest.approx(0.6)

    def test_tier_2_low_score_floors_at_04(self) -> None:
        assert score_to_importance(0.3, 2) == 0.4

    def test_tier_1_exact_threshold(self) -> None:
        assert score_to_importance(0.7, 1) == 0.7

    def test_tier_2_exact_threshold(self) -> None:
        assert score_to_importance(0.5, 2) == max(0.4, 0.5 * 0.8)


# =============================================================================
# derive_namespace
# =============================================================================


@pytest.mark.unit
class TestDeriveNamespace:
    """Test namespace derivation from patterns."""

    def test_decision_pattern(self) -> None:
        assert derive_namespace(["decision"]) == "decisions"

    def test_error_pattern(self) -> None:
        assert derive_namespace(["error"]) == "troubleshooting"

    def test_solution_pattern(self) -> None:
        assert derive_namespace(["solution"]) == "troubleshooting"

    def test_pattern_pattern(self) -> None:
        assert derive_namespace(["pattern"]) == "patterns"

    def test_convention_pattern(self) -> None:
        assert derive_namespace(["convention"]) == "patterns"

    def test_workaround_pattern(self) -> None:
        assert derive_namespace(["workaround"]) == "patterns"

    def test_configuration_pattern(self) -> None:
        assert derive_namespace(["configuration"]) == "patterns"

    def test_important_pattern(self) -> None:
        assert derive_namespace(["important"]) == "notes"

    def test_explicit_pattern(self) -> None:
        assert derive_namespace(["explicit"]) == "notes"

    def test_definition_pattern(self) -> None:
        assert derive_namespace(["definition"]) == "definitions"

    def test_decision_has_priority_over_error(self) -> None:
        assert derive_namespace(["decision", "error"]) == "decisions"

    def test_empty_patterns_returns_captured(self) -> None:
        assert derive_namespace([]) == "captured"

    def test_unknown_pattern_returns_captured(self) -> None:
        assert derive_namespace(["something_unknown"]) == "captured"

    @pytest.mark.parametrize(
        "ns",
        [
            "decisions",
            "troubleshooting",
            "patterns",
            "notes",
            "definitions",
            "captured",
        ],
    )
    def test_all_namespaces_are_valid_format(self, ns: str) -> None:
        # Must match NAMESPACE_PATTERN: ^[a-zA-Z][a-zA-Z0-9_-]{0,62}$
        import re

        assert re.match(r"^[a-zA-Z][a-zA-Z0-9_-]{0,62}$", ns)


# =============================================================================
# _build_context
# =============================================================================


@pytest.mark.unit
class TestBuildContext:
    """Test context dict building."""

    def test_tool_name_present(self) -> None:
        ctx = _build_context(HookInput(tool_name="Edit"))
        assert ctx["tool_name"] == "Edit"

    def test_session_id_present(self) -> None:
        ctx = _build_context(HookInput(session_id="sess-1", tool_name="Edit"))
        assert ctx["session_id"] == "sess-1"

    def test_file_path_from_tool_input(self) -> None:
        ctx = _build_context(
            HookInput(
                tool_name="Edit",
                tool_input={"file_path": "/src/app.py"},
            )
        )
        assert ctx["file_path"] == "/src/app.py"

    def test_no_file_path_when_absent(self) -> None:
        ctx = _build_context(HookInput(tool_name="Bash"))
        assert "file_path" not in ctx

    def test_empty_tool_name_excluded(self) -> None:
        ctx = _build_context(HookInput())
        assert "tool_name" not in ctx

    def test_empty_session_id_excluded(self) -> None:
        ctx = _build_context(HookInput(tool_name="Edit"))
        assert "session_id" not in ctx


# =============================================================================
# _derive_tags
# =============================================================================


@pytest.mark.unit
class TestDeriveTags:
    """Test tag derivation."""

    def test_tool_name_included(self) -> None:
        tags = _derive_tags(HookInput(tool_name="Bash"), [])
        assert "bash" in tags

    def test_patterns_included(self) -> None:
        tags = _derive_tags(HookInput(tool_name="Edit"), ["decision", "important"])
        assert "edit" in tags
        assert "decision" in tags
        assert "important" in tags

    def test_no_duplicates(self) -> None:
        tags = _derive_tags(HookInput(tool_name="Edit"), ["edit"])
        assert tags.count("edit") == 1

    def test_empty_tool_name(self) -> None:
        tags = _derive_tags(HookInput(), ["decision"])
        assert "decision" in tags
        assert "" not in tags


# =============================================================================
# M-5: Protocol types — structural subtyping of NamedTuples
# =============================================================================


@pytest.mark.unit
class TestProtocolTypes:
    """M-5: NamedTuples from signal_detection/redaction satisfy pipeline Protocols."""

    def test_signal_result_satisfies_protocol(self) -> None:
        """FakeSignalResult (NamedTuple) works as classify_fn return type."""
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided to use X"),
            classify_fn=_make_classify(tier=1, score=0.9, patterns=["decision"]),
            redact_fn=_make_redact(text="decided to use X"),
            write_fn=_make_write(),
        )
        assert result.signal_tier == 1
        assert result.signal_score == 0.9

    def test_redaction_result_satisfies_protocol(self) -> None:
        """FakeRedactionResult (NamedTuple) works as redact_fn return type."""
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided to use X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided to use X", count=2, skip=False),
            write_fn=_make_write(),
        )
        assert result.redaction_count == 2
        assert result.queued is True


# =============================================================================
# Client parameter passthrough
# =============================================================================


@pytest.mark.unit
class TestClientParam:
    """Test that client parameter is passed to write_fn."""

    def test_default_client_is_claude_code(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided X"),
            write_fn=write,
        )
        kwargs = write.call_args[1]
        assert kwargs["client"] == "claude-code"

    def test_custom_client_passed(self) -> None:
        write = _make_write()
        run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided X"),
            write_fn=write,
            client="cursor",
        )
        kwargs = write.call_args[1]
        assert kwargs["client"] == "cursor"


# =============================================================================
# None from write_fn (rate limited)
# =============================================================================


@pytest.mark.unit
class TestWriteFnNone:
    """Test handling when write_fn returns None (rate limited)."""

    def test_rate_limited_skipped(self) -> None:
        write = MagicMock(return_value=None)
        result = run_pipeline(
            _default_hook_input(),
            extract_fn=_make_extract("decided X"),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="decided X"),
            write_fn=write,
        )
        assert result.skipped is True
        assert result.skip_reason == "rate_limited"
        assert result.queued is False
