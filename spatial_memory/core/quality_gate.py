"""Quality gate for memory storage.

Scores incoming content to prevent low-quality saves.
Reuses EXTRACTION_PATTERNS from lifecycle_ops for signal detection.

Formula (from Cognitive Offloading Design):
    Score = signal_score x 0.3
          + content_length_score x 0.2
          + structure_score x 0.2
          + context_richness x 0.3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from spatial_memory.core.lifecycle_ops import EXTRACTION_PATTERNS


@dataclass
class QualityScore:
    """Result of quality gate evaluation."""

    total: float
    signal_score: float
    content_length_score: float
    structure_score: float
    context_richness: float


def score_memory_quality(
    content: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> QualityScore:
    """Score memory content quality for the quality gate.

    Args:
        content: The memory content text.
        tags: Optional tags associated with the memory.
        metadata: Optional metadata dict.

    Returns:
        QualityScore with component scores and total.
    """
    signal = _score_signal(content)
    length = _score_content_length(content)
    structure = _score_structure(content, tags)
    richness = _score_context_richness(content)

    total = signal * 0.3 + length * 0.2 + structure * 0.2 + richness * 0.3

    return QualityScore(
        total=total,
        signal_score=signal,
        content_length_score=length,
        structure_score=structure,
        context_richness=richness,
    )


def _score_signal(content: str) -> float:
    """Check content against EXTRACTION_PATTERNS, return max confidence found."""
    max_confidence = 0.0
    content_lower = content.lower()

    for pattern, base_confidence, _pattern_type in EXTRACTION_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            max_confidence = max(max_confidence, base_confidence)

    return max_confidence


def _score_content_length(content: str) -> float:
    """Score based on content length."""
    length = len(content)

    if length < 20:
        return 0.0
    if length < 100:
        return 0.5
    if length <= 500:
        return 1.0
    if length <= 2000:
        return 0.8
    return 0.7


def _score_structure(content: str, tags: list[str] | None) -> float:
    """Score based on structural quality indicators."""
    score = 0.0

    # Has tags (+0.3)
    if tags:
        score += 0.3

    # Has reasoning words (+0.3)
    reasoning_pattern = r"\b(?:because|so that|due to|in order to|since|therefore)\b"
    if re.search(reasoning_pattern, content, re.IGNORECASE):
        score += 0.3

    # Has specific names/paths (+0.4)
    # File paths, function names (word.word or word/word patterns), or PascalCase identifiers
    specifics_pattern = r"(?:\w+[./]\w+|[A-Z][a-z]+[A-Z]\w+)"
    if re.search(specifics_pattern, content):
        score += 0.4

    return min(1.0, score)


def _score_context_richness(content: str) -> float:
    """Score based on contextual reference richness."""
    score = 0.0

    # References files (+0.25)
    if re.search(r"\.\w{1,5}\b", content) and re.search(r"[\w/\\]", content):
        # More specific: looks for file-like patterns (path/to/file.ext)
        if re.search(r"[\w./\\-]+\.\w{1,5}\b", content):
            score += 0.25

    # References functions (+0.25)
    if re.search(r"\w+\(", content):
        score += 0.25

    # References versions (+0.25)
    if re.search(r"v?\d+\.\d+", content):
        score += 0.25

    # References URLs or code (+0.25)
    if re.search(r"https?://", content) or re.search(r"`[^`]+`", content):
        score += 0.25

    return min(1.0, score)
