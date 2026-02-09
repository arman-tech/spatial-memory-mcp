"""Unit tests for spatial_memory.hooks.transcript_reader.

Tests cover:
1. read_transcript_delta — basic reading, offset seeking, bytes pre-filter
2. Entry parsing — assistant text extraction, sidechain filtering
3. Edge cases — empty file, malformed JSON, oversized lines, offset beyond EOF
4. State persistence — load_state, save_state
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_memory.hooks.transcript_reader import (
    MAX_LINE_BYTES,
    _parse_entry,
    load_state,
    read_transcript_delta,
    save_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def _make_assistant_entry(
    text: str = "The fix was to reset the cache.",
    uuid: str = "uuid-1",
    timestamp: str = "2026-01-01T00:00:00Z",
    sidechain: bool = False,
) -> dict:
    """Create a valid assistant transcript entry."""
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
        "isSidechain": sidechain,
        "timestamp": timestamp,
        "uuid": uuid,
    }


def _make_user_entry(content: str = "Hello") -> dict:
    """Create a user transcript entry."""
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "uuid-user",
    }


def _make_tool_result_entry() -> dict:
    """Create a user entry with tool_result content (array)."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "uuid-tool",
    }


def _make_progress_entry() -> dict:
    return {"type": "progress", "data": {"status": "running"}}


def _make_file_history_entry() -> dict:
    return {"type": "file-history-snapshot", "files": {"/a.py": "abc"}}


# =============================================================================
# read_transcript_delta — basic reading
# =============================================================================


@pytest.mark.unit
class TestReadTranscriptDelta:
    """Test basic transcript reading."""

    def test_reads_assistant_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(path, [_make_assistant_entry(text="Decided to use Redis.")])

        entries, offset = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "Decided to use Redis."
        assert entries[0].role == "assistant"
        assert offset > 0

    def test_skips_user_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [
                _make_user_entry(),
                _make_assistant_entry(text="The solution is X."),
            ],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "The solution is X."

    def test_skips_tool_result_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [_make_tool_result_entry(), _make_assistant_entry(text="Done.")],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1

    def test_skips_progress_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [_make_progress_entry(), _make_assistant_entry(text="OK.")],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1

    def test_skips_file_history_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [_make_file_history_entry(), _make_assistant_entry(text="OK.")],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1

    def test_skips_sidechain_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [
                _make_assistant_entry(text="Main text.", sidechain=False),
                _make_assistant_entry(text="Subagent text.", sidechain=True),
            ],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "Main text."

    def test_multiple_assistant_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [
                _make_assistant_entry(text="First.", uuid="u1"),
                _make_user_entry(),
                _make_assistant_entry(text="Second.", uuid="u2"),
            ],
        )

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 2
        assert entries[0].text == "First."
        assert entries[1].text == "Second."


# =============================================================================
# read_transcript_delta — offset seeking
# =============================================================================


@pytest.mark.unit
class TestReadTranscriptDeltaOffset:
    """Test offset-based delta reading."""

    def test_reads_from_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(
            path,
            [
                _make_assistant_entry(text="Old entry.", uuid="u1"),
                _make_assistant_entry(text="New entry.", uuid="u2"),
            ],
        )

        # First read: get all
        entries1, offset1 = read_transcript_delta(str(path), 0)
        assert len(entries1) == 2

        # Second read from offset: no new entries
        entries2, offset2 = read_transcript_delta(str(path), offset1)
        assert len(entries2) == 0
        assert offset2 == offset1

    def test_reads_appended_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(path, [_make_assistant_entry(text="First.", uuid="u1")])

        _, offset1 = read_transcript_delta(str(path))

        # Append a new entry
        with open(path, "a", encoding="utf-8") as f:
            json.dump(_make_assistant_entry(text="Appended.", uuid="u2"), f)
            f.write("\n")

        entries2, offset2 = read_transcript_delta(str(path), offset1)
        assert len(entries2) == 1
        assert entries2[0].text == "Appended."
        assert offset2 > offset1

    def test_offset_beyond_file_size_resets(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        _write_jsonl(path, [_make_assistant_entry(text="Entry.")])

        entries, _ = read_transcript_delta(str(path), 999_999_999)
        assert len(entries) == 1  # Should reset and read from start


# =============================================================================
# read_transcript_delta — edge cases
# =============================================================================


@pytest.mark.unit
class TestReadTranscriptDeltaEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_path(self) -> None:
        entries, offset = read_transcript_delta("")
        assert entries == []
        assert offset == 0

    def test_nonexistent_file(self) -> None:
        entries, offset = read_transcript_delta("/nonexistent/transcript.jsonl")
        assert entries == []
        assert offset == 0

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        entries, offset = read_transcript_delta(str(path))
        assert entries == []
        assert offset == 0

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("not valid json\n")
            json.dump(_make_assistant_entry(text="Valid."), f)
            f.write("\n")

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "Valid."

    def test_oversized_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        with open(path, "wb") as f:
            # Write an oversized line (just bytes, not valid JSON)
            f.write(b'"assistant" ' + b"x" * (MAX_LINE_BYTES + 100) + b"\n")
            # Write a valid entry
            valid = json.dumps(_make_assistant_entry(text="Small.")).encode("utf-8")
            f.write(valid + b"\n")

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "Small."

    def test_assistant_with_tool_use_blocks_extracts_text_only(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Analysis text here."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Edit",
                        "input": {"file_path": "/a.py"},
                    },
                ],
            },
            "isSidechain": False,
            "timestamp": "2026-01-01T00:00:00Z",
            "uuid": "uuid-mixed",
        }
        _write_jsonl(path, [entry])

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 1
        assert entries[0].text == "Analysis text here."

    def test_assistant_with_only_tool_use_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {},
                    }
                ],
            },
            "isSidechain": False,
            "timestamp": "2026-01-01T00:00:00Z",
            "uuid": "uuid-toolonly",
        }
        _write_jsonl(path, [entry])

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 0

    def test_assistant_with_empty_text_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        entry = _make_assistant_entry(text="   ")
        _write_jsonl(path, [entry])

        entries, _ = read_transcript_delta(str(path))
        assert len(entries) == 0


# =============================================================================
# _parse_entry — unit tests
# =============================================================================


@pytest.mark.unit
class TestParseEntry:
    """Test the internal _parse_entry function."""

    def test_valid_assistant(self) -> None:
        data = _make_assistant_entry(text="Hello.", uuid="u1", timestamp="2026-01-01T00:00:00Z")
        entry = _parse_entry(data)
        assert entry is not None
        assert entry.role == "assistant"
        assert entry.text == "Hello."
        assert entry.uuid == "u1"
        assert entry.timestamp == "2026-01-01T00:00:00Z"
        assert entry.entry_type == "assistant"

    def test_non_assistant_type(self) -> None:
        assert _parse_entry({"type": "user"}) is None

    def test_sidechain_filtered(self) -> None:
        data = _make_assistant_entry(sidechain=True)
        assert _parse_entry(data) is None

    def test_missing_message(self) -> None:
        assert _parse_entry({"type": "assistant"}) is None

    def test_non_dict_message(self) -> None:
        assert _parse_entry({"type": "assistant", "message": "string"}) is None

    def test_non_list_content(self) -> None:
        data = {
            "type": "assistant",
            "message": {"role": "assistant", "content": "string"},
        }
        assert _parse_entry(data) is None

    def test_multiple_text_blocks_joined(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "text", "text": "Part two."},
                ],
            },
            "isSidechain": False,
            "timestamp": "2026-01-01T00:00:00Z",
            "uuid": "uuid-multi",
        }
        entry = _parse_entry(data)
        assert entry is not None
        assert "Part one." in entry.text
        assert "Part two." in entry.text


# =============================================================================
# State persistence
# =============================================================================


@pytest.mark.unit
class TestStatePersistence:
    """Test load_state and save_state."""

    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))

        save_state("session-abc", 12345, "2026-01-01T00:00:00Z")
        state = load_state("session-abc")
        assert state["last_offset"] == 12345
        assert state["last_timestamp"] == "2026-01-01T00:00:00Z"

    def test_load_missing_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))

        state = load_state("nonexistent-session")
        assert state["last_offset"] == 0
        assert state["last_timestamp"] == ""

    def test_load_empty_session_id(self) -> None:
        state = load_state("")
        assert state["last_offset"] == 0

    def test_save_empty_session_id_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
        # Should not raise
        save_state("", 100, "ts")
        # No state file created
        state_dir = tmp_path / "pending-saves" / "state"
        assert not state_dir.exists() or len(list(state_dir.glob("*.json"))) == 0

    def test_save_overwrites_previous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))

        save_state("s1", 100, "ts1")
        save_state("s1", 200, "ts2")
        state = load_state("s1")
        assert state["last_offset"] == 200
        assert state["last_timestamp"] == "ts2"

    def test_load_corrupted_json_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
        state_dir = tmp_path / "pending-saves" / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "bad.json").write_text("not json")

        state = load_state("bad")
        assert state["last_offset"] == 0
