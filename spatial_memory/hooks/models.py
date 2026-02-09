"""Domain types and tool filtering constants for hook scripts.

Provides immutable data types for hook input/output and tool classification
logic.  Used by the pipeline orchestrator and the CLI entrypoint.

**STDLIB-ONLY**: Only ``dataclasses`` and ``typing`` imports allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
# Protocol types for pipeline callables
# ---------------------------------------------------------------------------


class SignalResultProtocol(Protocol):
    """Structural type for signal classification results."""

    tier: int
    score: float
    patterns_matched: list[str]


class RedactionResultProtocol(Protocol):
    """Structural type for redaction results."""

    redacted_text: str
    redaction_count: int
    should_skip: bool


class WriteQueueFileProtocol(Protocol):
    """Structural type for queue file writer callable.

    Uses ``Protocol`` instead of ``Callable[..., Path]`` to preserve
    parameter names, types, and defaults in the type signature.
    """

    def __call__(
        self,
        content: str,
        source_hook: str,
        project_root_dir: str = "",
        suggested_namespace: str = "default",
        suggested_tags: list[str] | None = None,
        suggested_importance: float = 0.5,
        signal_tier: int = 1,
        signal_patterns_matched: list[str] | None = None,
        context: dict[str, object] | None = None,
        client: str = "claude-code",
    ) -> Path: ...


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
# Transcript data types (Phase C)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptEntry:
    """Single entry parsed from a Claude Code JSONL transcript.

    Only ``assistant`` entries with non-empty text are interesting for
    memory capture.  Other entry types (``user``, ``file-history-snapshot``,
    ``progress``) are filtered out by the reader.
    """

    role: str = ""  # "user" | "assistant"
    text: str = ""  # extracted text content (from content blocks)
    timestamp: str = ""  # ISO timestamp
    uuid: str = ""  # message UUID
    entry_type: str = ""  # raw entry type from JSONL


@dataclass(frozen=True)
class TranscriptHookInput:
    """Parsed stdin for PreCompact / Stop hooks.

    Separate from ``HookInput`` because the field sets differ fundamentally:
    ``HookInput`` has tool-specific fields; this has transcript/trigger fields.

    This is a **union type** — it carries fields from both PreCompact and Stop
    hooks.  Each entrypoint only populates the fields relevant to its event.
    Splitting into two dataclasses would add complexity without benefit since
    the pipeline treats them identically.
    """

    session_id: str = ""
    transcript_path: str = ""
    cwd: str = ""
    permission_mode: str = ""
    hook_event_name: str = ""
    # PreCompact-specific
    trigger: str = ""  # "manual" | "auto" | "session_end" (Stop sets this)
    custom_instructions: str = ""  # Reserved for future PreCompact use
    # Stop-specific
    stop_hook_active: bool = False


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
