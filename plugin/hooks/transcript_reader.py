"""JSONL transcript delta reader with offset tracking.

Reads new entries from a Claude Code session transcript since the last
processed byte offset.  Uses binary seek + readline for performance
(skip already-processed bytes, then bytes pre-filter before JSON parsing).

State persistence uses a small JSON sidecar file per session in the
queue ``state/`` directory.

**STDLIB-ONLY**: Only ``json``, ``os``, ``pathlib`` imports allowed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from spatial_memory.hooks.hook_helpers import sanitize_session_id, validate_transcript_path
from spatial_memory.hooks.models import TranscriptEntry
from spatial_memory.hooks.queue_writer import get_queue_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LINE_BYTES: int = 1_048_576  # 1MB — skip oversized lines (e.g. base64 payloads)

_ASSISTANT_MARKER: bytes = b'"assistant"'
"""Bytes pre-filter: only parse lines containing this keyword."""


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    """Return the state directory path (``{queue_dir}/state/``)."""
    return get_queue_dir() / "state"


def load_state(session_id: str) -> dict[str, int | str]:
    """Load offset state for a session.

    Sanitizes *session_id* before using it as a filename component
    (defense-in-depth — entrypoints also sanitize).

    Returns:
        ``{"last_offset": <int>, "last_timestamp": <str>}``
        or defaults if no state file exists or session_id is invalid.
    """
    session_id = sanitize_session_id(session_id)
    if not session_id:
        return {"last_offset": 0, "last_timestamp": ""}

    state_path = _state_dir() / f"{session_id}.json"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "last_offset": int(data.get("last_offset", 0)),
            "last_timestamp": str(data.get("last_timestamp", "")),
        }
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        return {"last_offset": 0, "last_timestamp": ""}


def save_state(session_id: str, offset: int, timestamp: str) -> None:
    """Persist offset state for a session.

    Sanitizes *session_id* before using it as a filename component
    (defense-in-depth — entrypoints also sanitize).

    Creates the state directory if it doesn't exist.  Uses tmp + rename
    for best-effort atomicity.
    """
    session_id = sanitize_session_id(session_id)
    if not session_id:
        return

    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    state_path = state_dir / f"{session_id}.json"
    tmp_path = state_dir / f"{session_id}.tmp"

    payload = json.dumps(
        {"last_offset": offset, "last_timestamp": timestamp},
        ensure_ascii=False,
    )
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(str(tmp_path), str(state_path))


# ---------------------------------------------------------------------------
# Transcript reading
# ---------------------------------------------------------------------------


def read_transcript_delta(
    transcript_path: str,
    last_offset: int = 0,
) -> tuple[list[TranscriptEntry], int]:
    """Read new entries from transcript JSONL since *last_offset*.

    Uses binary mode for seek + readline, with a bytes pre-filter
    (``b'"assistant"' in line``) to skip non-assistant lines without
    JSON parsing (~10-15x speedup at 90% filter rate).

    Validates *transcript_path* before use (defense-in-depth — entrypoints
    also validate).  Returns ``([], 0)`` for invalid paths.

    Args:
        transcript_path: Path to the JSONL transcript file.
        last_offset: Byte position to seek to (0 = start of file).

    Returns:
        ``(entries, new_offset)`` where *new_offset* is the byte position
        after the last successfully parsed line.  Returns ``([], last_offset)``
        on ``OSError`` during reading.
    """
    transcript_path = validate_transcript_path(transcript_path)
    if not transcript_path:
        return [], 0

    path = Path(transcript_path)
    try:
        file_size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return [], 0

    if file_size == 0:
        return [], 0

    # Reset offset if it's beyond file size (file was truncated/rotated)
    if last_offset > file_size:
        last_offset = 0

    entries: list[TranscriptEntry] = []

    try:
        with open(path, "rb") as f:
            f.seek(last_offset)

            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break

                # Skip oversized lines
                if len(line_bytes) > MAX_LINE_BYTES:
                    continue

                # Skip empty lines
                stripped = line_bytes.strip()
                if not stripped:
                    continue

                # Bytes pre-filter: only parse lines containing "assistant"
                if _ASSISTANT_MARKER not in line_bytes:
                    continue

                # Decode and parse
                try:
                    line_str = line_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if not isinstance(data, dict):
                    continue

                entry = _parse_entry(data)
                if entry is not None:
                    entries.append(entry)

            new_offset = f.tell()
    except OSError:
        return [], last_offset

    return entries, new_offset


def _parse_entry(data: dict[str, object]) -> TranscriptEntry | None:
    """Parse a single JSONL entry dict into a TranscriptEntry.

    Returns ``None`` if the entry is not an assistant message with text,
    or if it's a sidechain (subagent) message.
    """
    entry_type = data.get("type", "")
    if entry_type != "assistant":
        return None

    # Skip sidechain (subagent) entries
    if data.get("isSidechain", False):
        return None

    message = data.get("message")
    if not isinstance(message, dict):
        return None

    # message.content is ALWAYS an array for assistant messages
    content_blocks = message.get("content")
    if not isinstance(content_blocks, list):
        return None

    # Extract text from text blocks only (skip tool_use blocks)
    text_parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_val = block.get("text", "")
            if isinstance(text_val, str) and text_val.strip():
                text_parts.append(text_val.strip())

    if not text_parts:
        return None

    combined_text = "\n\n".join(text_parts)

    return TranscriptEntry(
        role="assistant",
        text=combined_text,
        timestamp=str(data.get("timestamp", "")),
        uuid=str(data.get("uuid", "")),
        entry_type=str(entry_type),
    )
