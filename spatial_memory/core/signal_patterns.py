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
    # Facts/Definitions
    (
        r"(.+?)\s+(?:is|are|means|refers to)\s+(.+?)(?:\.|$)",
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
]
