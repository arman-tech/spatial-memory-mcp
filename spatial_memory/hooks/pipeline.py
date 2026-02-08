"""Application-layer orchestration for the PostToolUse hook.

Chains tool filtering, content extraction, signal classification, secret
redaction, and queue writing into a single pipeline.  All external
dependencies are **injected as callable parameters** (DIP) so the pipeline
is fully testable with mock functions.

**STDLIB-ONLY**: Only ``pathlib`` and typing imports allowed (besides
sibling ``models`` module loaded via relative import).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from spatial_memory.hooks.models import HookInput, ProcessingResult, should_skip_tool

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _score_to_importance(score: float, tier: int) -> float:
    """Map signal score + tier to a suggested importance value.

    - Tier 1: importance = max(0.7, score)
    - Tier 2: importance = max(0.4, score * 0.8)
    """
    if tier == 1:
        return max(0.7, score)
    return max(0.4, score * 0.8)


_NAMESPACE_MAP: list[tuple[list[str], str]] = [
    (["decision"], "decisions"),
    (["error", "solution"], "troubleshooting"),
    (["pattern", "convention", "workaround", "configuration"], "patterns"),
    (["important", "explicit"], "notes"),
    (["definition"], "definitions"),
]


def _derive_namespace(patterns: list[str]) -> str:
    """Derive a suggested namespace from matched signal patterns.

    Checks patterns against a priority-ordered mapping and returns the
    first match.  Falls back to ``"captured"`` if no mapping matches.
    """
    if not patterns:
        return "captured"

    for pattern_keys, namespace in _NAMESPACE_MAP:
        for p in patterns:
            if p in pattern_keys:
                return namespace

    return "captured"


def _build_context(hook_input: HookInput) -> dict[str, str]:
    """Build a flat context dict from the hook input."""
    ctx: dict[str, str] = {}
    if hook_input.tool_name:
        ctx["tool_name"] = hook_input.tool_name
    if hook_input.session_id:
        ctx["session_id"] = hook_input.session_id

    # Extract file_path if present in tool_input
    file_path = hook_input.tool_input.get("file_path")
    if isinstance(file_path, str) and file_path:
        ctx["file_path"] = file_path

    return ctx


def _derive_tags(hook_input: HookInput, patterns: list[str]) -> list[str]:
    """Derive suggested tags from the hook input and matched patterns."""
    tags: list[str] = []

    # Add tool name as tag
    if hook_input.tool_name:
        tags.append(hook_input.tool_name.lower())

    # Add matched signal patterns as tags
    for p in patterns:
        if p not in tags:
            tags.append(p)

    return tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    hook_input: HookInput,
    *,
    extract_fn: Callable[[str, dict[str, object], str], str],
    classify_fn: Callable[[str], Any],
    redact_fn: Callable[[str], Any],
    write_fn: Callable[..., Path],
    project_root: str = "",
) -> ProcessingResult:
    """Run the PostToolUse processing pipeline.

    Steps:
        1. Filter tool (skip read-only / spatial-memory tools)
        2. Extract content from tool invocation
        3. Classify signal tier
        4. Gate: skip tier 3 (no signal)
        5. Redact secrets
        6. Gate: skip if redaction says should_skip
        7. Compute metadata (namespace, importance, tags, context)
        8. Write queue file

    Args:
        hook_input: Parsed hook input data.
        extract_fn: Content extractor — ``(tool_name, tool_input, tool_response) -> str``.
        classify_fn: Signal classifier — ``(text) -> SignalResult``.
        redact_fn: Secret redactor — ``(text) -> RedactionResult``.
        write_fn: Queue file writer — ``write_queue_file(content, ...) -> Path``.
        project_root: Project root directory path.

    Returns:
        ProcessingResult describing what happened.
    """
    result = ProcessingResult()

    # Step 1: Filter tool
    skip, reason = should_skip_tool(hook_input.tool_name)
    if skip:
        result.skipped = True
        result.skip_reason = reason
        return result

    # Step 2: Extract content
    content = extract_fn(
        hook_input.tool_name,
        hook_input.tool_input,
        hook_input.tool_response,
    )
    if not content or not content.strip():
        result.skipped = True
        result.skip_reason = "empty_content"
        return result

    result.content_length = len(content)

    # Step 3: Classify signal
    signal = classify_fn(content)
    result.signal_tier = signal.tier
    result.signal_score = signal.score
    result.patterns_matched = list(signal.patterns_matched)

    # Step 4: Gate tier 3
    if signal.tier == 3:
        result.skipped = True
        result.skip_reason = "tier_3_no_signal"
        return result

    # Step 5: Redact secrets
    redaction = redact_fn(content)
    result.redaction_count = redaction.redaction_count

    # Step 6: Gate should_skip
    if redaction.should_skip:
        result.skipped = True
        result.skip_reason = "redaction_skip"
        return result

    redacted_content = redaction.redacted_text

    # Step 7: Compute metadata
    importance = _score_to_importance(signal.score, signal.tier)
    namespace = _derive_namespace(result.patterns_matched)
    tags = _derive_tags(hook_input, result.patterns_matched)
    context = _build_context(hook_input)

    # Step 8: Write queue file
    queue_path = write_fn(
        content=redacted_content,
        source_hook="PostToolUse",
        project_root_dir=project_root,
        suggested_namespace=namespace,
        suggested_tags=tags,
        suggested_importance=importance,
        signal_tier=signal.tier,
        signal_patterns_matched=result.patterns_matched,
        context=context,
        client="claude-code",
    )

    result.queued = True
    result.queue_path = str(queue_path)

    return result
