"""Stdlib-only signal detection engine for hook scripts.

Classifies text into signal tiers (1 = auto-save, 2 = ask first, 3 = skip)
based on regex pattern matching.  This is the hook-side equivalent of
``core/signal_patterns.py`` (server-side extraction patterns), extended with
additional patterns from the design doc.

Cross-reference: ``spatial_memory/core/signal_patterns.py`` — canonical
server-side patterns.  Keep both files in sync when adding new patterns.

**STDLIB-ONLY**: Only ``re`` and typing imports allowed.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SignalResult(NamedTuple):
    """Result of signal classification."""

    tier: int  # 1 (auto-save), 2 (ask first), or 3 (skip)
    score: float  # 0.0-1.0 max confidence across matched patterns
    patterns_matched: list[str]  # pattern type names that matched


def classify_signal(text: str) -> SignalResult:
    """Classify text into a signal tier based on pattern matching.

    Args:
        text: The text to classify.

    Returns:
        SignalResult with tier, max confidence score, and matched pattern names.
    """
    if not text or not text.strip():
        return SignalResult(tier=3, score=0.0, patterns_matched=[])

    text_lower = text.lower()
    max_score = 0.0
    matched: list[str] = []

    for keywords, pattern, confidence, pattern_type in _PATTERNS:
        # Keyword pre-filter: skip regex if no keyword substring matches
        if not any(kw in text_lower for kw in keywords):
            continue
        if pattern.search(text):
            if pattern_type not in matched:
                matched.append(pattern_type)
            if confidence > max_score:
                max_score = confidence

    # Tier assignment based on max score
    if max_score >= 0.8:
        tier = 1
    elif max_score >= 0.5:
        tier = 2
    else:
        tier = 3

    return SignalResult(tier=tier, score=max_score, patterns_matched=matched)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each entry: (keyword_prefixes, compiled_regex, confidence, pattern_type)
#
# keyword_prefixes is a tuple of lowercase substrings used as a fast pre-filter
# before running the regex.  At least one must appear in text.lower().

_PatternEntry = tuple[tuple[str, ...], re.Pattern[str], float, str]

_PATTERNS: list[_PatternEntry] = [
    # --- Tier 1 patterns (confidence >= 0.8) ---
    # Decisions
    (
        ("decided", "chose", "going with", "selected", "will use"),
        re.compile(
            r"\b(?:decided|chose|going with|selected|will use)\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.8,
        "decision",
    ),
    # Important points
    (
        ("important", "note", "remember", "key point"),
        re.compile(
            r"\b(?:important|note|remember|key point)[:\s]+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.9,
        "important",
    ),
    # Solutions/fixes
    (
        ("the fix", "the solution", "the approach"),
        re.compile(
            r"the (?:fix|solution|approach) (?:is|was)\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.85,
        "solution",
    ),
    # Error diagnoses
    (
        ("the issue", "the problem", "the bug"),
        re.compile(
            r"the (?:issue|problem|bug) was\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.8,
        "error",
    ),
    # Explicit save requests
    (
        ("save", "remember", "note", "store"),
        re.compile(
            r"\b(?:save|remember|note|store)(?:\s+that)?\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.95,
        "explicit",
    ),
    # Patterns/learnings
    (
        ("the trick is", "the key is", "pattern:"),
        re.compile(
            r"(?:the trick is|the key is|pattern:)\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.85,
        "pattern",
    ),
    # Resolved by
    (
        ("resolved by",),
        re.compile(r"resolved by\s+.+?(?:\.|$)", re.IGNORECASE),
        0.8,
        "solution",
    ),
    # Broke/failed because
    (
        ("broke because", "failed because"),
        re.compile(
            r"(?:broke|failed) because\s+.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.8,
        "error",
    ),
    # Error was due to
    (
        ("the error was due to",),
        re.compile(r"the error was due to\s+.+?(?:\.|$)", re.IGNORECASE),
        0.8,
        "error",
    ),
    # Architecture decisions
    (
        ("the architecture will be",),
        re.compile(r"the architecture will be\s+.+?(?:\.|$)", re.IGNORECASE),
        0.8,
        "decision",
    ),
    # --- Tier 2 patterns (confidence 0.5-0.79) ---
    # Definitions (requires definitional structure: "X is a/an/the Y")
    (
        (" is a ", " is an ", " is the ", " means ", " refers to "),
        re.compile(
            r"\b\w+\s+(?:is\s+(?:a|an|the)\s+|means\s+|refers\s+to\s+)\S.+?(?:\.|$)",
            re.IGNORECASE,
        ),
        0.6,
        "definition",
    ),
    # Conventions
    (
        ("convention here is",),
        re.compile(r"convention here is\s+.+?(?:\.|$)", re.IGNORECASE),
        0.7,
        "convention",
    ),
    # Watch out for
    (
        ("watch out for",),
        re.compile(r"watch out for\s+.+?(?:\.|$)", re.IGNORECASE),
        0.7,
        "workaround",
    ),
    # The workaround is
    (
        ("the workaround is",),
        re.compile(r"the workaround is\s+.+?(?:\.|$)", re.IGNORECASE),
        0.75,
        "workaround",
    ),
    # Configuration
    (
        ("you need to set",),
        re.compile(r"you need to set\s+.+?(?:\.|$)", re.IGNORECASE),
        0.65,
        "configuration",
    ),
    # Testing
    (
        ("test failed", "test passed", "tests pass", "test fail", "test coverage", "assertion"),
        re.compile(
            r"(?:tests?\s+(?:failed|passed|pass|fail)|test coverage|assertion\s+(?:error|failed))",
            re.IGNORECASE,
        ),
        0.7,
        "testing",
    ),
    # Performance
    (
        ("bottleneck", "latency", "throughput", "optimization", "faster when", "slower when"),
        re.compile(
            r"(?:bottleneck|latency|throughput|optimization|(?:faster|slower)\s+when)",
            re.IGNORECASE,
        ),
        0.7,
        "performance",
    ),
    # Environment
    (
        ("requires version", "upgraded to", "compatible with", "python version", "node version"),
        re.compile(
            r"(?:requires?\s+version|upgraded?\s+to|compatible\s+with"
            r"|(?:python|node|npm|java)\s+version)",
            re.IGNORECASE,
        ),
        0.6,
        "environment",
    ),
    # Dependencies
    (
        (
            "dependency",
            "dependencies",
            "depends on",
            "package version",
            "library version",
            "pip install",
            "npm install",
        ),
        re.compile(
            r"(?:dependenc(?:y|ies)|depends\s+on|(?:package|library)\s+version"
            r"|(?:pip|npm|yarn)\s+install)",
            re.IGNORECASE,
        ),
        0.6,
        "dependency",
    ),
    # API
    (
        (
            "endpoint",
            "api returns",
            "api return",
            "request body",
            "response body",
            "webhook",
            "rest api",
            "graphql",
        ),
        re.compile(
            r"(?:endpoint|api\s+returns?|(?:request|response)\s+body"
            r"|webhook|rest\s+api|graphql)",
            re.IGNORECASE,
        ),
        0.6,
        "api",
    ),
    # Procedures
    (
        ("steps to", "step to", "to reproduce", "step 1", "first,"),
        re.compile(
            r"(?:steps?\s+to\s+|to\s+reproduce|step\s+\d|first,\s+.+?then)",
            re.IGNORECASE,
        ),
        0.6,
        "procedure",
    ),
    # Workflows
    (
        ("process for", "pipeline for", "workflow", "run before", "run after"),
        re.compile(
            r"(?:(?:process|pipeline)\s+for|workflow\s+(?:is|for)|run\s+(?:before|after))",
            re.IGNORECASE,
        ),
        0.6,
        "workflow",
    ),
]
