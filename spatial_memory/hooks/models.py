"""Domain types and tool filtering constants for hook scripts.

Provides immutable data types for hook input/output and tool classification
logic.  Used by the pipeline orchestrator and the CLI entrypoint.

**STDLIB-ONLY**: Only ``dataclasses`` and ``typing`` imports allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tool filtering constants
# ---------------------------------------------------------------------------

SKIP_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "LSP",
        "WebSearch",
        "WebFetch",
        "Task",
    }
)
"""Tools that never produce memory-worthy output (read-only / delegating)."""

SPATIAL_MEMORY_PREFIX: str = "mcp__spatial-memory__"
"""MCP tool prefix for spatial-memory's own tools (avoid recursive capture)."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookInput:
    """Parsed representation of Claude Code PostToolUse stdin JSON.

    Frozen dataclass — immutable after construction.
    """

    session_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, object] = field(default_factory=dict)
    tool_response: str = ""
    tool_use_id: str = ""
    transcript_path: str = ""
    cwd: str = ""
    hook_event_name: str = ""
    permission_mode: str = ""


@dataclass
class ProcessingResult:
    """Result of the PostToolUse processing pipeline."""

    skipped: bool = False
    skip_reason: str = ""
    queued: bool = False
    queue_path: str = ""
    signal_tier: int = 3
    signal_score: float = 0.0
    patterns_matched: list[str] = field(default_factory=list)
    redaction_count: int = 0
    content_length: int = 0


# ---------------------------------------------------------------------------
# Tool filter logic
# ---------------------------------------------------------------------------


def should_skip_tool(tool_name: str) -> tuple[bool, str]:
    """Check whether a tool invocation should be skipped.

    Returns:
        ``(skip, reason)`` — *skip* is ``True`` if the tool should not be
        processed, with a human-readable *reason*.
    """
    if not tool_name:
        return True, "empty_tool_name"

    if tool_name in SKIP_TOOLS:
        return True, f"skip_tool:{tool_name}"

    if tool_name.startswith(SPATIAL_MEMORY_PREFIX):
        return True, "spatial_memory_tool"

    return False, ""
