"""Orchestrator for transcript-based hook processing (PreCompact / Stop).

Chains transcript reading, text extraction, signal classification, dedup
against the PostToolUse queue, secret redaction, and queue writing into a
single pipeline.  All external dependencies are **injected as callable
parameters** (DIP) so the pipeline is fully testable with mock functions.

**STDLIB-ONLY**: Only ``dataclasses``, ``pathlib`` and typing imports
allowed (besides sibling module types).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from spatial_memory.hooks.models import (
    RedactionResultProtocol,
    SignalResultProtocol,
    TranscriptEntry,
    TranscriptHookInput,
)
from spatial_memory.hooks.pipeline import derive_namespace, score_to_importance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_DEDUP_FILES: int = 50
"""Maximum recent queue files to scan for PostToolUse dedup hashes."""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TranscriptPipelineResult:
    """Result of transcript processing."""

    entries_scanned: int = 0
    entries_with_signal: int = 0
    entries_queued: int = 0
    entries_skipped_dup: int = 0
    entries_skipped_redaction: int = 0
    new_offset: int = 0
    texts_extracted: int = 0
    source_hook: str = ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_transcript_context(
    hook_input: TranscriptHookInput,
) -> dict[str, str]:
    """Build a context dict for transcript-sourced queue files."""
    ctx: dict[str, str] = {}
    if hook_input.session_id:
        ctx["session_id"] = hook_input.session_id
    if hook_input.hook_event_name:
        ctx["hook_event"] = hook_input.hook_event_name
    if hook_input.trigger:
        ctx["trigger"] = hook_input.trigger
    return ctx


def _derive_transcript_tags(patterns: list[str]) -> list[str]:
    """Derive tags for transcript-sourced content."""
    tags: list[str] = ["transcript"]
    for p in patterns:
        if p not in tags:
            tags.append(p)
    return tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_transcript_pipeline(
    hook_input: TranscriptHookInput,
    *,
    read_fn: Callable[[str, int], tuple[list[TranscriptEntry], int]],
    extract_fn: Callable[[list[TranscriptEntry]], list[str]],
    classify_fn: Callable[[str], SignalResultProtocol],
    redact_fn: Callable[[str], RedactionResultProtocol],
    write_fn: Callable[..., Path],
    load_state_fn: Callable[..., dict[str, int | str]],
    save_state_fn: Callable[..., None],
    get_queued_hashes_fn: Callable[[Path, int], set[str]],
    is_duplicate_fn: Callable[[str, set[str]], bool],
    queue_dir: Path,
    project_root: str = "",
) -> TranscriptPipelineResult:
    """Run the transcript processing pipeline.

    Steps:
        1. Load state (last offset for this session)
        2. Read delta (new transcript entries since last offset)
        3. Extract text from assistant entries
        4. Load dedup hashes from PostToolUse queue
        5. For each text block:
           a. Classify signal
           b. Gate: skip tier 3
           c. Dedup check against PostToolUse queue
           d. Redact secrets
           e. Gate: skip if redaction says should_skip
           f. Compute metadata
           g. Write queue file
        6. Save state (updated offset)

    All dependencies are injected as callables for testability.
    """
    source_hook = hook_input.hook_event_name or "Transcript"
    result = TranscriptPipelineResult(source_hook=source_hook)

    # Step 1: Load state
    state = load_state_fn(hook_input.session_id, project_root=project_root)
    last_offset = int(state.get("last_offset", 0))

    # Step 2: Read delta
    entries, new_offset = read_fn(hook_input.transcript_path, last_offset)
    result.entries_scanned = len(entries)
    result.new_offset = new_offset

    if not entries:
        # Save state even if no entries (offset may have advanced past non-assistant lines)
        if new_offset > last_offset:
            save_state_fn(hook_input.session_id, new_offset, "", project_root=project_root)
        return result

    # Step 3: Extract text
    texts = extract_fn(entries)
    result.texts_extracted = len(texts)

    if not texts:
        if new_offset > last_offset:
            save_state_fn(hook_input.session_id, new_offset, "", project_root=project_root)
        return result

    # Step 4: Load dedup hashes
    queued_hashes = get_queued_hashes_fn(queue_dir, DEFAULT_MAX_DEDUP_FILES)

    # Step 5: Process each text block
    context = _build_transcript_context(hook_input)

    for text in texts:
        if not text or not text.strip():
            continue

        # 5a: Classify signal
        signal = classify_fn(text)

        # 5b: Gate tier 3
        if signal.tier == 3:
            continue

        result.entries_with_signal += 1

        # 5c: Dedup check
        if is_duplicate_fn(text, queued_hashes):
            result.entries_skipped_dup += 1
            continue

        # 5d: Redact secrets
        redaction = redact_fn(text)

        # 5e: Gate should_skip
        if redaction.should_skip:
            result.entries_skipped_redaction += 1
            continue

        # 5f: Compute metadata
        importance = score_to_importance(signal.score, signal.tier)
        namespace = derive_namespace(list(signal.patterns_matched))
        tags = _derive_transcript_tags(list(signal.patterns_matched))

        # 5g: Write queue file
        write_fn(
            content=redaction.redacted_text,
            source_hook=source_hook,
            project_root_dir=project_root,
            suggested_namespace=namespace,
            suggested_tags=tags,
            suggested_importance=importance,
            signal_tier=signal.tier,
            signal_patterns_matched=list(signal.patterns_matched),
            context=context,
            client="claude-code",
        )

        result.entries_queued += 1

    # Step 6: Save state
    last_timestamp = ""
    if entries:
        last_timestamp = entries[-1].timestamp
    save_state_fn(hook_input.session_id, new_offset, last_timestamp, project_root=project_root)

    return result
