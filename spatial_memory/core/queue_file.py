"""Data model and parsing for queue files. Pure data, no I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from spatial_memory.core.queue_constants import QUEUE_FILE_VERSION

# Windows device names that must be rejected in untrusted paths
_WINDOWS_DEVICE_RE = re.compile(r"^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\..+)?$", re.IGNORECASE)


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

        # Validate signal_patterns_matched
        signal_patterns = data.get("signal_patterns_matched", [])
        if not isinstance(signal_patterns, list) or not all(
            isinstance(p, str) for p in signal_patterns
        ):
            raise ValueError("signal_patterns_matched must be a list of strings")

        # Validate context (deep validation: nesting, size, serializability)
        context = data.get("context", {})
        if not isinstance(context, dict):
            raise ValueError(f"context must be a dict, got {type(context).__name__}")
        try:
            from spatial_memory.core.validation import validate_metadata

            validate_metadata(context, validate_keys=False)
        except Exception as e:
            raise ValueError(f"Invalid context: {e}") from e

        # Validate namespace
        suggested_namespace = data.get("suggested_namespace", "default")
        if not isinstance(suggested_namespace, str):
            raise ValueError(
                f"suggested_namespace must be a string, got {type(suggested_namespace).__name__}"
            )

        from spatial_memory.core.validation import NAMESPACE_PATTERN

        if not NAMESPACE_PATTERN.match(suggested_namespace):
            raise ValueError(
                f"Invalid suggested_namespace: {suggested_namespace}. "
                "Must start with a letter, contain only letters/numbers/dash/underscore, "
                "and be max 63 characters."
            )

        # Validate content length (defense in depth — also checked in remember())
        from spatial_memory.core.validation import MAX_CONTENT_LENGTH

        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(
                f"content exceeds maximum length of {MAX_CONTENT_LENGTH} characters "
                f"(got {len(content)})"
            )

        # Validate project_root_dir (untrusted path from queue file)
        project_root_dir = data.get("project_root_dir", "")
        if project_root_dir:
            if not isinstance(project_root_dir, str):
                raise ValueError(
                    f"project_root_dir must be a string, got {type(project_root_dir).__name__}"
                )
            if "\x00" in project_root_dir:
                raise ValueError("project_root_dir contains null bytes")
            if len(project_root_dir) > 1024:
                raise ValueError(
                    f"project_root_dir too long ({len(project_root_dir)} chars, max 1024)"
                )
            # Reject UNC paths to prevent SSRF via SMB on Windows
            if project_root_dir.startswith("\\\\") or project_root_dir.startswith("//"):
                raise ValueError("project_root_dir must not be a UNC path")
            # Reject Windows device names
            stem = project_root_dir.split("\\")[0].split("/")[0]
            if _WINDOWS_DEVICE_RE.match(stem):
                raise ValueError(f"project_root_dir must not be a Windows device name: {stem}")

        return cls(
            version=version,
            content=content,
            source_hook=data.get("source_hook", ""),
            timestamp=data.get("timestamp", ""),
            project_root_dir=project_root_dir,
            suggested_namespace=suggested_namespace,
            suggested_tags=suggested_tags,
            suggested_importance=float(importance),
            signal_tier=signal_tier,
            signal_patterns_matched=signal_patterns,
            context=context,
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
