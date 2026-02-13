"""Unit tests for spatial_memory.hooks.transcript_pipeline.

Uses mock callables for all injected dependencies. Tests cover:
1. Pipeline reads delta and extracts text
2. Signal gating — tier 3 skipped, tier 1/2 proceed
3. Dedup gating — duplicates skipped
4. Redaction gating — should_skip stops processing
5. Queue write — correct args passed
6. State persistence — load/save called correctly
7. Edge cases — no entries, no texts, empty transcript
8. Context and tags — transcript-specific metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest

from spatial_memory.hooks.models import TranscriptEntry, TranscriptHookInput
from spatial_memory.hooks.transcript_pipeline import (
    TranscriptPipelineResult,
    _build_transcript_context,
    _derive_transcript_tags,
    run_transcript_pipeline,
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


def _entry(text: str = "Decided to use Redis.", uuid: str = "u1") -> TranscriptEntry:
    return TranscriptEntry(
        role="assistant",
        text=text,
        timestamp="2026-01-01T00:00:00Z",
        uuid=uuid,
        entry_type="assistant",
    )


def _hook_input(**overrides) -> TranscriptHookInput:
    defaults = {
        "session_id": "sess-1",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/project",
        "hook_event_name": "PreCompact",
        "trigger": "auto",
    }
    defaults.update(overrides)
    return TranscriptHookInput(**defaults)


def _make_read(entries: list[TranscriptEntry] | None = None, offset: int = 1000) -> MagicMock:
    entries = entries if entries is not None else [_entry()]
    return MagicMock(return_value=(entries, offset))


def _make_extract(texts: list[str] | None = None) -> MagicMock:
    return MagicMock(return_value=texts if texts is not None else ["Decided to use Redis."])


def _make_classify(
    tier: int = 1, score: float = 0.9, patterns: list[str] | None = None
) -> MagicMock:
    result = FakeSignalResult(tier=tier, score=score, patterns_matched=patterns or ["decision"])
    return MagicMock(return_value=result)


def _make_redact(text: str = "clean", count: int = 0, skip: bool = False) -> MagicMock:
    result = FakeRedactionResult(redacted_text=text, redaction_count=count, should_skip=skip)
    return MagicMock(return_value=result)


def _make_write(path: str = "/queue/new/file.json") -> MagicMock:
    return MagicMock(return_value=Path(path))


def _make_load_state(offset: int = 0) -> MagicMock:
    return MagicMock(return_value={"last_offset": offset, "last_timestamp": ""})


def _make_save_state() -> MagicMock:
    return MagicMock()


def _make_get_hashes(hashes: set[str] | None = None) -> MagicMock:
    return MagicMock(return_value=hashes or set())


def _make_is_dup(dup: bool = False) -> MagicMock:
    return MagicMock(return_value=dup)


def _run(**overrides) -> TranscriptPipelineResult:
    """Run pipeline with sensible defaults; override any dependency."""
    kwargs = {
        "hook_input": _hook_input(),
        "read_fn": _make_read(),
        "extract_fn": _make_extract(),
        "classify_fn": _make_classify(),
        "redact_fn": _make_redact(text="Decided to use Redis."),
        "write_fn": _make_write(),
        "load_state_fn": _make_load_state(),
        "save_state_fn": _make_save_state(),
        "get_queued_hashes_fn": _make_get_hashes(),
        "is_duplicate_fn": _make_is_dup(),
        "queue_dir": Path("/queue"),
        "project_root": "/project",
    }
    kwargs.update(overrides)
    hook_input = kwargs.pop("hook_input")
    return run_transcript_pipeline(hook_input, **kwargs)


# =============================================================================
# Basic pipeline flow
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineFlow:
    """Test the basic pipeline flow from read to write."""

    def test_reads_and_extracts(self) -> None:
        read_fn = _make_read()
        extract_fn = _make_extract()
        result = _run(read_fn=read_fn, extract_fn=extract_fn)

        read_fn.assert_called_once()
        extract_fn.assert_called_once()
        assert result.entries_scanned == 1
        assert result.texts_extracted == 1

    def test_full_pipeline_queues_entry(self) -> None:
        write_fn = _make_write()
        result = _run(write_fn=write_fn)

        write_fn.assert_called_once()
        assert result.entries_queued == 1
        assert result.entries_with_signal == 1

    def test_result_has_correct_source_hook(self) -> None:
        result = _run(hook_input=_hook_input(hook_event_name="Stop"))
        assert result.source_hook == "Stop"


# =============================================================================
# Signal gating
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineSignalGating:
    """Test tier-based gating in transcript pipeline."""

    def test_tier_3_skipped(self) -> None:
        write_fn = _make_write()
        result = _run(
            classify_fn=_make_classify(tier=3, score=0.2, patterns=[]),
            write_fn=write_fn,
        )

        write_fn.assert_not_called()
        assert result.entries_queued == 0
        assert result.entries_with_signal == 0

    def test_tier_1_proceeds(self) -> None:
        write_fn = _make_write()
        result = _run(
            classify_fn=_make_classify(tier=1, score=0.9),
            write_fn=write_fn,
        )

        write_fn.assert_called_once()
        assert result.entries_queued == 1

    def test_tier_2_proceeds(self) -> None:
        write_fn = _make_write()
        result = _run(
            classify_fn=_make_classify(tier=2, score=0.6),
            write_fn=write_fn,
        )

        write_fn.assert_called_once()
        assert result.entries_queued == 1


# =============================================================================
# Dedup gating
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineDedup:
    """Test dedup against PostToolUse queue."""

    def test_duplicate_skipped(self) -> None:
        write_fn = _make_write()
        result = _run(
            is_duplicate_fn=_make_is_dup(dup=True),
            write_fn=write_fn,
        )

        write_fn.assert_not_called()
        assert result.entries_skipped_dup == 1
        assert result.entries_queued == 0

    def test_non_duplicate_proceeds(self) -> None:
        write_fn = _make_write()
        result = _run(
            is_duplicate_fn=_make_is_dup(dup=False),
            write_fn=write_fn,
        )

        write_fn.assert_called_once()
        assert result.entries_queued == 1


# =============================================================================
# Redaction gating
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineRedaction:
    """Test redaction gating."""

    def test_should_skip_stops_processing(self) -> None:
        write_fn = _make_write()
        result = _run(
            redact_fn=_make_redact(text="[REDACTED]", count=5, skip=True),
            write_fn=write_fn,
        )

        write_fn.assert_not_called()
        assert result.entries_skipped_redaction == 1

    def test_clean_content_proceeds(self) -> None:
        write_fn = _make_write()
        result = _run(
            redact_fn=_make_redact(text="clean text", count=0, skip=False),
            write_fn=write_fn,
        )

        write_fn.assert_called_once()
        assert result.entries_queued == 1


# =============================================================================
# Queue write args
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineWriteArgs:
    """Test that queue write receives correct arguments."""

    def test_correct_args_passed(self) -> None:
        write_fn = _make_write()
        _run(
            hook_input=_hook_input(session_id="s1", hook_event_name="PreCompact"),
            classify_fn=_make_classify(tier=1, score=0.9, patterns=["decision"]),
            redact_fn=_make_redact(text="Decided to use Redis."),
            write_fn=write_fn,
            project_root="/home/user/project",
        )

        write_fn.assert_called_once()
        kwargs = write_fn.call_args[1]
        assert kwargs["content"] == "Decided to use Redis."
        assert kwargs["source_hook"] == "PreCompact"
        assert kwargs["project_root_dir"] == "/home/user/project"
        assert kwargs["client"] == "claude-code"
        assert kwargs["signal_tier"] == 1
        assert kwargs["signal_patterns_matched"] == ["decision"]

    def test_namespace_derived_from_patterns(self) -> None:
        write_fn = _make_write()
        _run(
            classify_fn=_make_classify(tier=1, score=0.8, patterns=["error"]),
            write_fn=write_fn,
        )

        kwargs = write_fn.call_args[1]
        assert kwargs["suggested_namespace"] == "troubleshooting"

    def test_tags_include_transcript_and_patterns(self) -> None:
        write_fn = _make_write()
        _run(
            classify_fn=_make_classify(tier=1, patterns=["decision"]),
            write_fn=write_fn,
        )

        kwargs = write_fn.call_args[1]
        assert "transcript" in kwargs["suggested_tags"]
        assert "decision" in kwargs["suggested_tags"]

    def test_context_has_session_and_hook_event(self) -> None:
        write_fn = _make_write()
        _run(
            hook_input=_hook_input(session_id="s1", hook_event_name="PreCompact", trigger="auto"),
            write_fn=write_fn,
        )

        kwargs = write_fn.call_args[1]
        ctx = kwargs["context"]
        assert ctx["session_id"] == "s1"
        assert ctx["hook_event"] == "PreCompact"
        assert ctx["trigger"] == "auto"


# =============================================================================
# State persistence
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineState:
    """Test state load/save in pipeline."""

    def test_loads_state_at_start(self) -> None:
        load_fn = _make_load_state(offset=500)
        read_fn = _make_read(entries=[], offset=500)
        _run(load_state_fn=load_fn, read_fn=read_fn)

        load_fn.assert_called_once_with("sess-1", project_root="/project")
        # read_fn should receive the loaded offset
        read_fn.assert_called_once_with("/tmp/transcript.jsonl", 500)

    def test_saves_state_after_processing(self) -> None:
        save_fn = _make_save_state()
        _run(save_state_fn=save_fn, read_fn=_make_read(offset=2000))

        save_fn.assert_called_once()
        args = save_fn.call_args[0]
        assert args[0] == "sess-1"  # session_id
        assert args[1] == 2000  # new offset

    def test_saves_state_even_with_no_entries(self) -> None:
        save_fn = _make_save_state()
        _run(
            read_fn=_make_read(entries=[], offset=500),
            load_state_fn=_make_load_state(offset=0),
            save_state_fn=save_fn,
        )

        save_fn.assert_called_once()

    def test_no_state_save_when_offset_unchanged(self) -> None:
        save_fn = _make_save_state()
        _run(
            read_fn=_make_read(entries=[], offset=0),
            load_state_fn=_make_load_state(offset=0),
            save_state_fn=save_fn,
        )

        save_fn.assert_not_called()


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.unit
class TestTranscriptPipelineEdgeCases:
    """Test edge cases in transcript pipeline."""

    def test_no_entries_returned(self) -> None:
        write_fn = _make_write()
        result = _run(
            read_fn=_make_read(entries=[], offset=100),
            load_state_fn=_make_load_state(offset=0),
            write_fn=write_fn,
        )

        write_fn.assert_not_called()
        assert result.entries_scanned == 0
        assert result.entries_queued == 0

    def test_no_texts_extracted(self) -> None:
        write_fn = _make_write()
        result = _run(extract_fn=_make_extract(texts=[]), write_fn=write_fn)

        write_fn.assert_not_called()
        assert result.texts_extracted == 0

    def test_multiple_texts_processed(self) -> None:
        write_fn = _make_write()
        entries = [_entry("First.", uuid="u1"), _entry("Second.", uuid="u2")]
        result = _run(
            read_fn=_make_read(entries=entries),
            extract_fn=_make_extract(texts=["First.", "Second."]),
            write_fn=write_fn,
        )

        assert write_fn.call_count == 2
        assert result.entries_queued == 2

    def test_empty_text_in_list_skipped(self) -> None:
        write_fn = _make_write()
        result = _run(
            extract_fn=_make_extract(texts=["", "  ", "Valid."]),
            write_fn=write_fn,
        )

        write_fn.assert_called_once()
        assert result.entries_queued == 1


# =============================================================================
# Helper functions
# =============================================================================


@pytest.mark.unit
class TestBuildTranscriptContext:
    """Test _build_transcript_context."""

    def test_includes_session_id(self) -> None:
        ctx = _build_transcript_context(_hook_input(session_id="s1"))
        assert ctx["session_id"] == "s1"

    def test_includes_hook_event(self) -> None:
        ctx = _build_transcript_context(_hook_input(hook_event_name="Stop"))
        assert ctx["hook_event"] == "Stop"

    def test_includes_trigger(self) -> None:
        ctx = _build_transcript_context(_hook_input(trigger="manual"))
        assert ctx["trigger"] == "manual"

    def test_empty_fields_excluded(self) -> None:
        ctx = _build_transcript_context(TranscriptHookInput())
        assert ctx == {}


@pytest.mark.unit
class TestDeriveTranscriptTags:
    """Test _derive_transcript_tags."""

    def test_always_includes_transcript(self) -> None:
        tags = _derive_transcript_tags([])
        assert "transcript" in tags

    def test_includes_patterns(self) -> None:
        tags = _derive_transcript_tags(["decision", "important"])
        assert "transcript" in tags
        assert "decision" in tags
        assert "important" in tags

    def test_no_duplicates(self) -> None:
        tags = _derive_transcript_tags(["transcript"])
        assert tags.count("transcript") == 1


# =============================================================================
# Client parameter passthrough
# =============================================================================


@pytest.mark.unit
class TestClientParam:
    """Test that client parameter is passed to write_fn."""

    def test_default_client_is_claude_code(self) -> None:
        write = _make_write()
        run_transcript_pipeline(
            _hook_input(),
            read_fn=_make_read(),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="clean"),
            write_fn=write,
            load_state_fn=_make_load_state(),
            save_state_fn=_make_save_state(),
            get_queued_hashes_fn=MagicMock(return_value=set()),
            is_duplicate_fn=MagicMock(return_value=False),
            queue_dir=Path("/queue"),
        )
        kwargs = write.call_args[1]
        assert kwargs["client"] == "claude-code"

    def test_custom_client_passed(self) -> None:
        write = _make_write()
        run_transcript_pipeline(
            _hook_input(),
            read_fn=_make_read(),
            extract_fn=_make_extract(),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="clean"),
            write_fn=write,
            load_state_fn=_make_load_state(),
            save_state_fn=_make_save_state(),
            get_queued_hashes_fn=MagicMock(return_value=set()),
            is_duplicate_fn=MagicMock(return_value=False),
            queue_dir=Path("/queue"),
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

    def test_rate_limited_stops_loop(self) -> None:
        write = MagicMock(return_value=None)
        result = run_transcript_pipeline(
            _hook_input(),
            read_fn=_make_read(),
            extract_fn=_make_extract(["text1", "text2", "text3"]),
            classify_fn=_make_classify(tier=1),
            redact_fn=_make_redact(text="clean"),
            write_fn=write,
            load_state_fn=_make_load_state(),
            save_state_fn=_make_save_state(),
            get_queued_hashes_fn=MagicMock(return_value=set()),
            is_duplicate_fn=MagicMock(return_value=False),
            queue_dir=Path("/queue"),
        )
        # Write called once, returned None, loop stopped
        assert write.call_count == 1
        assert result.entries_queued == 0
