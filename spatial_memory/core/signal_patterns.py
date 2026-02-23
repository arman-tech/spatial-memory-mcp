"""Shared signal patterns for extraction and quality gating.

These patterns are used by both ``lifecycle_ops`` (extraction) and
``quality_gate`` (scoring).  Extracting them into a standalone module
breaks the direct import dependency between those two modules.
"""

from __future__ import annotations

# Default extraction patterns: (regex_pattern, base_confidence, pattern_type)
# These patterns identify memory-worthy content in conversation text
SIGNAL_PATTERNS: list[tuple[str, float, str]] = [
    # Decisions
    (
        r"(?:decided|chose|going with|selected|will use)\s+(.+?)(?:\.|$)",
        0.8,
        "decision",
    ),
    # Facts/Definitions (requires definitional structure: "X is a/an/the Y")
    (
        r"\b\w+\s+(?:is\s+(?:a|an|the)\s+|means\s+|refers\s+to\s+)\S.+?(?:\.|$)",
        0.6,
        "definition",
    ),
    # Important points
    (
        r"(?:important|note|remember|key point)[:\s]+(.+?)(?:\.|$)",
        0.9,
        "important",
    ),
    # Solutions/Fixes
    (
        r"(?:the (?:fix|solution|approach) (?:is|was))\s+(.+?)(?:\.|$)",
        0.85,
        "solution",
    ),
    # Error diagnoses
    (
        r"(?:the (?:issue|problem|bug) was)\s+(.+?)(?:\.|$)",
        0.8,
        "error",
    ),
    # Explicit save requests
    (
        r"(?:save|remember|note|store)(?:\s+that)?\s+(.+?)(?:\.|$)",
        0.95,
        "explicit",
    ),
    # Patterns/Learnings
    (
        r"(?:the trick is|the key is|pattern:)\s+(.+?)(?:\.|$)",
        0.85,
        "pattern",
    ),
    # Testing
    (
        r"(?:tests?\s+(?:failed|passed|pass|fail)|test coverage|assertion\s+(?:error|failed))",
        0.7,
        "testing",
    ),
    # Performance
    (
        r"(?:bottleneck|latency|throughput|optimization|(?:faster|slower)\s+when)",
        0.7,
        "performance",
    ),
    # Environment
    (
        r"(?:requires?\s+version|upgraded?\s+to|compatible\s+with"
        r"|(?:python|node|npm|java)\s+version)",
        0.6,
        "environment",
    ),
    # Dependencies
    (
        r"(?:dependenc(?:y|ies)|depends\s+on|(?:package|library)\s+version"
        r"|(?:pip|npm|yarn)\s+install)",
        0.6,
        "dependency",
    ),
    # API
    (
        r"(?:endpoint|api\s+returns?|(?:request|response)\s+body"
        r"|webhook|rest\s+api|graphql)",
        0.6,
        "api",
    ),
    # Procedures
    (
        r"(?:steps?\s+to\s+|to\s+reproduce|step\s+\d|first,\s+.+?then)",
        0.6,
        "procedure",
    ),
    # Workflows
    (
        r"(?:(?:process|pipeline)\s+for|workflow\s+(?:is|for)|run\s+(?:before|after))",
        0.6,
        "workflow",
    ),
]
