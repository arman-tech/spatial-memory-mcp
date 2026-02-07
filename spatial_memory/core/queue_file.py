"""Data model and parsing for queue files. Pure data, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spatial_memory.core.queue_constants import QUEUE_FILE_VERSION


@dataclass
class QueueFile:
    """Parsed queue file contents."""

    version: int
    content: str
    source_hook: str
    timestamp: str
    project_root_dir: str
    suggested_namespace: str
    suggested_tags: list[str]
    suggested_importance: float
    signal_tier: int
    signal_patterns_matched: list[str]
    context: dict[str, Any]
    client: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> QueueFile:
        """Parse and validate a queue file JSON dict.

        Args:
            data: Raw JSON dictionary from queue file.

        Returns:
            Validated QueueFile instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        version = data.get("version")
        if version != QUEUE_FILE_VERSION:
            raise ValueError(
                f"Unsupported queue file version: {version} (expected {QUEUE_FILE_VERSION})"
            )

        content = data.get("content", "")
        if not content or not content.strip():
            raise ValueError("Queue file content must not be empty")

        importance = data.get("suggested_importance", 0.5)
        if not isinstance(importance, (int, float)) or importance < 0 or importance > 1:
            raise ValueError(
                f"suggested_importance must be a number between 0 and 1, got {importance}"
            )

        signal_tier = data.get("signal_tier", 1)
        if not isinstance(signal_tier, int) or signal_tier not in (1, 2, 3):
            raise ValueError(f"signal_tier must be 1, 2, or 3, got {signal_tier}")

        suggested_tags = data.get("suggested_tags", [])
        if not isinstance(suggested_tags, list) or not all(
            isinstance(t, str) for t in suggested_tags
        ):
            raise ValueError("suggested_tags must be a list of strings")

        return cls(
            version=version,
            content=content,
            source_hook=data.get("source_hook", ""),
            timestamp=data.get("timestamp", ""),
            project_root_dir=data.get("project_root_dir", ""),
            suggested_namespace=data.get("suggested_namespace", "default"),
            suggested_tags=suggested_tags,
            suggested_importance=float(importance),
            signal_tier=signal_tier,
            signal_patterns_matched=data.get("signal_patterns_matched", []),
            context=data.get("context", {}),
            client=data.get("client", ""),
        )


@dataclass
class ProcessedResult:
    """Result of processing a single queue file."""

    filename: str
    status: str  # "stored", "rejected_exact", "rejected_similar", "rejected_quality", "error"
    memory_id: str | None = None
    content_summary: str = ""
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
