"""Core algorithms for memory lifecycle operations.

This module contains the pure algorithmic implementations for:
- Decay: Time/access-based importance reduction
- Reinforcement: Importance boosting
- Extraction: Pattern-based memory extraction from text
- Consolidation: Duplicate detection and merging

These functions are pure computations with no I/O dependencies,
enabling easy testing and reuse across different contexts.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias for vectors
Vector = NDArray[np.float32]


# =============================================================================
# Decay Algorithms
# =============================================================================


def calculate_decay_factor(
    days_since_access: float,
    access_count: int,
    base_importance: float,
    decay_function: Literal["exponential", "linear", "step"],
    half_life_days: float,
    access_weight: float,
) -> float:
    """
    Calculate decay factor for a memory based on time and access patterns.

    Implements a modified half-life regression algorithm inspired by Ebbinghaus
    forgetting curve, Duolingo HLR, and FSRS. The decay factor represents how
    much the importance should be multiplied by (1.0 = no decay, 0.0 = full decay).

    The effective half-life is adaptive:
    - More accesses = longer half-life (slower decay)
    - Higher importance = longer half-life (slower decay)

    Args:
        days_since_access: Days since the memory was last accessed.
            Negative values are treated as 0.
        access_count: Number of times the memory has been accessed.
            Capped at 20 for half-life calculation to prevent overflow.
        base_importance: Current importance of the memory (0-1).
        decay_function: Type of decay curve.
            - "exponential": Smooth decay following 2^(-t/half_life)
            - "linear": Constant rate decay reaching 0 at 2*half_life
            - "step": Discrete thresholds at half_life intervals
        half_life_days: Base half-life in days for exponential decay.
        access_weight: Weight of access count in decay calculation (0-1).
            Higher values make access count more influential vs time.

    Returns:
        Decay factor (0.0 - 1.0) to multiply with importance.

    Example:
        >>> factor = calculate_decay_factor(
        ...     days_since_access=30,
        ...     access_count=5,
        ...     base_importance=0.7,
        ...     decay_function="exponential",
        ...     half_life_days=30,
        ...     access_weight=0.3,
        ... )
        >>> 0.0 <= factor <= 1.0
        True
    """
    # Clamp inputs to valid ranges
    days_since_access = max(0.0, days_since_access)
    access_count = max(0, access_count)
    base_importance = max(0.0, min(1.0, base_importance))
    access_weight = max(0.0, min(1.0, access_weight))
    half_life_days = max(1.0, half_life_days)

    # Adaptive half-life: more accesses = longer half-life (slower decay)
    # Each access adds 50% to half-life, capped at 20 accesses
    access_bonus = 0.5
    access_factor = (1 + access_bonus) ** min(access_count, 20)

    # Higher base importance also slows decay
    importance_factor = 1 + base_importance

    # Calculate effective half-life
    effective_half_life = half_life_days * access_factor * importance_factor

    # Calculate time-based decay
    if decay_function == "exponential":
        # Exponential decay: importance halves every half_life days
        time_decay = 2.0 ** (-days_since_access / effective_half_life)
    elif decay_function == "linear":
        # Linear decay: reaches zero at 2x half_life
        time_decay = max(0.0, 1.0 - days_since_access / (2 * effective_half_life))
    elif decay_function == "step":
        # Step function: discrete drops at half_life intervals
        if days_since_access < effective_half_life:
            time_decay = 1.0
        elif days_since_access < 2 * effective_half_life:
            time_decay = 0.5
        else:
            time_decay = 0.25
    else:
        # Default to exponential for unknown functions
        logger.warning("Unknown decay function '%s', using exponential", decay_function)
        time_decay = 2.0 ** (-days_since_access / effective_half_life)

    # Calculate access-based stability (memories accessed more are more stable)
    # log1p(x)/log(100) normalizes access count to ~1.0 at 99 accesses
    if access_count > 0:
        access_stability = min(1.0, math.log1p(access_count) / math.log(100))
    else:
        access_stability = 0.0

    # Blend time decay with access-based stability
    # access_weight controls the balance (0 = pure time decay, 1 = pure access stability)
    decay_factor = (1 - access_weight) * time_decay + access_weight * access_stability

    return float(max(0.0, min(1.0, decay_factor)))


def apply_decay(
    current_importance: float,
    decay_factor: float,
    min_importance: float,
) -> float:
    """
    Apply decay to importance with a minimum floor.

    Applies the calculated decay factor to the current importance score,
    ensuring the result never falls below the specified minimum. This
    prevents memories from becoming completely unfindable due to decay.

    Args:
        current_importance: Current importance score (0-1).
        decay_factor: Decay factor from calculate_decay_factor (0-1).
        min_importance: Minimum importance floor (0-1).
            Memories will not decay below this threshold.

    Returns:
        New importance score after decay, clamped to [min_importance, 1.0].

    Example:
        >>> apply_decay(current_importance=0.8, decay_factor=0.5, min_importance=0.1)
        0.4
        >>> apply_decay(current_importance=0.2, decay_factor=0.3, min_importance=0.1)
        0.1
    """
    # Clamp inputs
    current_importance = max(0.0, min(1.0, current_importance))
    decay_factor = max(0.0, min(1.0, decay_factor))
    min_importance = max(0.0, min(1.0, min_importance))

    # Apply decay
    decayed = current_importance * decay_factor

    # Enforce floor and ceiling
    return max(min_importance, min(1.0, decayed))


# =============================================================================
# Reinforcement Algorithms
# =============================================================================


def calculate_reinforcement(
    current_importance: float,
    boost_type: Literal["additive", "multiplicative", "set_value"],
    boost_amount: float,
    max_importance: float = 1.0,
) -> tuple[float, float]:
    """
    Calculate new importance after reinforcement.

    Computes the reinforced importance based on the specified boost type.
    This implements the memory strengthening counterpart to decay, allowing
    frequently accessed or explicitly important memories to maintain or
    increase their importance.

    Args:
        current_importance: Current importance score (0-1).
        boost_type: Type of boost to apply.
            - "additive": new = current + boost_amount
            - "multiplicative": new = current * (1 + boost_amount)
            - "set_value": new = boost_amount (direct override)
        boost_amount: Amount of boost to apply.
            For additive/multiplicative: the increment/factor.
            For set_value: the target importance.
        max_importance: Maximum allowed importance (default 1.0).
            Results are capped to this value.

    Returns:
        Tuple of (new_importance, actual_boost) where:
        - new_importance: The resulting importance after reinforcement
        - actual_boost: The actual change applied (may be less than requested
          if capped by max_importance)

    Example:
        >>> calculate_reinforcement(0.5, "additive", 0.1)
        (0.6, 0.1)
        >>> calculate_reinforcement(0.5, "multiplicative", 0.2)
        (0.6, 0.1)
        >>> calculate_reinforcement(0.5, "set_value", 0.8)
        (0.8, 0.3)
        >>> calculate_reinforcement(0.9, "additive", 0.2)  # Capped at 1.0
        (1.0, 0.1)
    """
    # Clamp inputs
    current_importance = max(0.0, min(1.0, current_importance))
    boost_amount = max(0.0, boost_amount)
    max_importance = max(0.0, min(1.0, max_importance))

    # Calculate new importance based on boost type
    if boost_type == "additive":
        new_importance = current_importance + boost_amount
    elif boost_type == "multiplicative":
        new_importance = current_importance * (1.0 + boost_amount)
    elif boost_type == "set_value":
        new_importance = boost_amount
    else:
        # Unknown boost type, return unchanged
        logger.warning("Unknown boost type '%s', returning unchanged", boost_type)
        return current_importance, 0.0

    # Cap at maximum
    new_importance = min(max_importance, max(0.0, new_importance))
    actual_boost = new_importance - current_importance

    return new_importance, actual_boost


# =============================================================================
# Extraction Algorithms
# =============================================================================


@dataclass
class ExtractionCandidate:
    """
    A candidate memory extracted from text.

    Attributes:
        content: The extracted text content.
        confidence: Confidence score (0-1) that this is a valid memory.
        pattern_type: Type of pattern that matched (e.g., "decision", "solution").
        start_pos: Start position in the original text.
        end_pos: End position in the original text.
    """

    content: str
    confidence: float
    pattern_type: str
    start_pos: int
    end_pos: int


# Default extraction patterns: (regex_pattern, base_confidence, pattern_type)
# These patterns identify memory-worthy content in conversation text
EXTRACTION_PATTERNS: list[tuple[str, float, str]] = [
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


def score_extraction_confidence(content: str, base_confidence: float) -> float:
    """
    Adjust extraction confidence based on content quality signals.

    Analyzes the extracted content for quality indicators that suggest
    higher or lower confidence in the extraction being meaningful.

    Args:
        content: Extracted text content to analyze.
        base_confidence: Base confidence from the pattern match.

    Returns:
        Adjusted confidence score clamped to [0.0, 1.0].

    Quality signals that increase confidence:
    - Longer content (10+ words): +0.1
    - Technical terms present: +0.05
    - Code snippets present: +0.1
    - URL references: +0.05
    - Proper sentence structure: +0.05

    Quality signals that decrease confidence:
    - Very short content (< 5 words): -0.1
    - All caps (shouting/headers): -0.15
    - Excessive punctuation: -0.1
    """
    score = base_confidence

    # Word count analysis
    words = content.split()
    word_count = len(words)

    if word_count >= 10:
        # Longer content is typically more informative
        score += 0.1
    elif word_count < 5:
        # Very short content may be incomplete
        score -= 0.1

    # Technical terms boost confidence (domain-specific knowledge)
    tech_terms = {
        "api",
        "database",
        "function",
        "class",
        "config",
        "error",
        "server",
        "client",
        "query",
        "model",
        "endpoint",
        "module",
        "package",
        "library",
        "framework",
        "method",
        "variable",
        "parameter",
        "exception",
        "async",
        "schema",
        "interface",
        "type",
    }
    content_lower = content.lower()
    if any(term in content_lower for term in tech_terms):
        score += 0.05

    # Code presence boosts confidence (concrete implementation details)
    has_code_block = "```" in content
    has_indented_code = bool(re.search(r"^\s{4,}", content, re.MULTILINE))
    has_inline_code = bool(re.search(r"`[^`]+`", content))

    if has_code_block or has_indented_code:
        score += 0.1
    elif has_inline_code:
        score += 0.05

    # URL presence (references external knowledge)
    if re.search(r"https?://", content):
        score += 0.05

    # All caps penalty (likely headers or shouting)
    if content.isupper() and len(content) > 10:
        score -= 0.15

    # Excessive punctuation penalty
    punct_count = sum(1 for c in content if c in "!?...")
    if word_count > 0 and punct_count > word_count / 2:
        score -= 0.1

    # Proper sentence structure (starts with capital, ends with punctuation)
    if content and content[0].isupper() and content[-1] in ".!?":
        score += 0.05

    # Clamp to valid range
    return max(0.0, min(1.0, score))


def extract_candidates(
    text: str,
    patterns: list[tuple[str, float, str]] | None = None,
    min_confidence: float = 0.5,
    max_candidates: int = 20,
) -> list[ExtractionCandidate]:
    """
    Extract memory candidates from text using pattern matching.

    Scans the input text for patterns that indicate memory-worthy content,
    such as decisions, definitions, solutions, and explicit save requests.
    Returns candidates with confidence scores for further filtering.

    Args:
        text: Text to extract memories from.
            Can be conversation transcript, documentation, notes, etc.
        patterns: List of (regex, base_confidence, pattern_type) tuples.
            If None, uses EXTRACTION_PATTERNS default.
        min_confidence: Minimum confidence threshold (0-1).
            Candidates below this are filtered out.
        max_candidates: Maximum number of candidates to return.

    Returns:
        List of extraction candidates sorted by confidence (descending).
        Limited to max_candidates entries.

    Example:
        >>> text = "The fix is to add a null check before accessing the property."
        >>> candidates = extract_candidates(text, min_confidence=0.5)
        >>> len(candidates) >= 1
        True
    """
    if not text or not text.strip():
        return []

    if patterns is None:
        patterns = EXTRACTION_PATTERNS

    candidates: list[ExtractionCandidate] = []

    for pattern, base_confidence, pattern_type in patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                # Extract content from first capture group or full match
                if match.groups():
                    # Join multiple groups if present (for definition patterns)
                    groups = [g for g in match.groups() if g]
                    content = " ".join(groups).strip()
                else:
                    content = match.group(0).strip()

                # Skip too short or too long
                if len(content) < 10 or len(content) > 5000:
                    continue

                # Adjust confidence based on content quality
                confidence = score_extraction_confidence(content, base_confidence)

                if confidence >= min_confidence:
                    candidates.append(
                        ExtractionCandidate(
                            content=content,
                            confidence=confidence,
                            pattern_type=pattern_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                        )
                    )
        except re.error as e:
            logger.warning("Invalid regex pattern '%s': %s", pattern, e)
            continue

    # Deduplicate overlapping extractions (keep highest confidence)
    candidates = dedupe_overlapping_extractions(candidates)

    # Sort by confidence and limit
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:max_candidates]


def dedupe_overlapping_extractions(
    candidates: list[ExtractionCandidate],
) -> list[ExtractionCandidate]:
    """Remove overlapping extractions, keeping highest confidence.

    Args:
        candidates: List of extraction candidates.

    Returns:
        Deduplicated list.
    """
    if not candidates:
        return []

    # Sort by start position, then by confidence (highest first)
    sorted_candidates = sorted(candidates, key=lambda c: (c.start_pos, -c.confidence))

    result: list[ExtractionCandidate] = []
    last_end = -1

    for candidate in sorted_candidates:
        # Skip if overlaps with previous kept candidate
        if candidate.start_pos < last_end:
            continue

        result.append(candidate)
        last_end = candidate.end_pos

    return result


# =============================================================================
# Consolidation Algorithms
# =============================================================================


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Computes the ratio of shared words to total unique words. This provides
    a simple lexical similarity measure that complements semantic (vector)
    similarity.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Jaccard similarity coefficient in range [0, 1].
        0 = no shared words, 1 = identical word sets.

    Example:
        >>> jaccard_similarity("hello world", "hello there")
        0.333...
        >>> jaccard_similarity("hello world", "hello world")
        1.0
        >>> jaccard_similarity("", "hello")
        0.0
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize to lowercase words
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def cosine_similarity_vectors(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity (-1 to 1).
    """
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return dot_product / (norm1 * norm2)


def combined_similarity(
    vector_similarity: float,
    content_overlap: float,
    content_weight: float = 0.3,
) -> float:
    """Calculate combined similarity score.

    Args:
        vector_similarity: Cosine similarity of embeddings.
        content_overlap: Jaccard similarity of content.
        content_weight: Weight for content similarity (0-1).

    Returns:
        Combined similarity score (0-1).
    """
    return (1 - content_weight) * vector_similarity + content_weight * content_overlap


@dataclass
class ConsolidationGroup:
    """
    A group of similar memories to consolidate.

    Attributes:
        member_indices: Indices of memories in this group.
        representative_idx: Index of the representative memory.
        avg_similarity: Average pairwise similarity within the group.
    """

    member_indices: list[int]
    representative_idx: int
    avg_similarity: float


def find_duplicate_groups(
    memory_ids: list[str],
    vectors: NDArray[np.float32],
    contents: list[str],
    threshold: float,
    content_weight: float = 0.3,
) -> list[list[int]]:
    """
    Find groups of duplicate memories using Union-Find algorithm.

    Identifies clusters of similar memories based on a combination of
    vector similarity (semantic) and content overlap (lexical). Uses
    Union-Find for efficient grouping of transitively similar memories.

    This is an efficient numpy-based implementation that computes all
    pairwise similarities in batch using matrix operations.

    Args:
        memory_ids: List of memory IDs (for logging/debugging).
        vectors: 2D array of shape (n_memories, embedding_dim).
            Should be normalized vectors for cosine similarity.
        contents: List of memory content strings.
        threshold: Minimum combined similarity to consider duplicates (0-1).
            Higher values = stricter matching.
        content_weight: Weight of content (Jaccard) similarity vs vector.
            0.0 = pure vector similarity
            1.0 = pure content similarity
            Default 0.3 balances both.

    Returns:
        List of duplicate groups, where each group is a list of indices
        into the original arrays. Only groups with 2+ members are returned.

    Example:
        >>> ids = ["a", "b", "c"]
        >>> vectors = np.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=np.float32)
        >>> contents = ["hello world", "hello world!", "goodbye"]
        >>> groups = find_duplicate_groups(ids, vectors, contents, threshold=0.8)
        >>> len(groups) >= 1  # First two should group together
        True
    """
    n = len(memory_ids)

    if n == 0:
        return []

    if n < 2:
        return []

    if n != vectors.shape[0] or n != len(contents):
        raise ValueError(
            f"Mismatched lengths: memory_ids={n}, vectors={vectors.shape[0]}, "
            f"contents={len(contents)}"
        )

    # Union-Find data structure
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        """Find with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        """Union by rank."""
        px, py = find(x), find(y)
        if px != py:
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

    # Calculate pairwise vector similarities using cosine similarity
    # For normalized vectors: cosine_sim = dot product
    # Normalize vectors first
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    normalized_vectors = vectors / norms

    # Compute cosine similarity matrix efficiently
    vector_sim = np.dot(normalized_vectors, normalized_vectors.T)

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Vector similarity (already computed)
            v_sim = float(vector_sim[i, j])

            # Content similarity (Jaccard)
            c_sim = jaccard_similarity(contents[i], contents[j])

            # Combined score
            combined = (1 - content_weight) * v_sim + content_weight * c_sim

            if combined >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Return only groups with duplicates (2+ members)
    return [g for g in groups.values() if len(g) > 1]


def find_duplicate_groups_with_callbacks(
    memory_count: int,
    get_vector_similarity: Any,  # callable[[int, int], float]
    get_content_similarity: Any,  # callable[[int, int], float]
    threshold: float,
    content_weight: float = 0.3,
) -> list[ConsolidationGroup]:
    """
    Find groups of duplicate memories using Union-Find with callback functions.

    This is an alternative implementation that uses callback functions instead
    of precomputed arrays. Useful when vectors/contents are lazily loaded.

    Args:
        memory_count: Number of memories.
        get_vector_similarity: Function(i, j) -> float for vector similarity.
        get_content_similarity: Function(i, j) -> float for content similarity.
        threshold: Minimum combined similarity for grouping.
        content_weight: Weight for content similarity.

    Returns:
        List of consolidation groups with indices.
    """
    if memory_count < 2:
        return []

    # Union-Find data structure
    parent = list(range(memory_count))
    rank = [0] * memory_count

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

    # Track similarities for average calculation
    similarities: dict[tuple[int, int], float] = {}

    # Compare all pairs
    for i in range(memory_count):
        for j in range(i + 1, memory_count):
            v_sim = get_vector_similarity(i, j)
            c_sim = get_content_similarity(i, j)
            combined = combined_similarity(v_sim, c_sim, content_weight)

            if combined >= threshold:
                union(i, j)
                similarities[(i, j)] = combined

    # Group by root
    groups_dict: dict[int, list[int]] = {}
    for i in range(memory_count):
        root = find(i)
        groups_dict.setdefault(root, []).append(i)

    # Convert to ConsolidationGroup objects (only groups with 2+ members)
    result: list[ConsolidationGroup] = []
    for members in groups_dict.values():
        if len(members) < 2:
            continue

        # Calculate average similarity
        total_sim = 0.0
        pair_count = 0
        for idx_i in range(len(members)):
            for idx_j in range(idx_i + 1, len(members)):
                mi, mj = members[idx_i], members[idx_j]
                key = (min(mi, mj), max(mi, mj))
                if key in similarities:
                    total_sim += similarities[key]
                    pair_count += 1

        avg_sim = total_sim / pair_count if pair_count > 0 else threshold

        result.append(
            ConsolidationGroup(
                member_indices=members,
                representative_idx=members[0],  # Will be updated by caller
                avg_similarity=avg_sim,
            )
        )

    return result


def select_representative(
    members: list[dict[str, Any]],
    strategy: Literal[
        "keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"
    ],
) -> int:
    """
    Select the representative memory index based on strategy.

    Determines which memory should be kept as the canonical version
    when merging a group of similar memories.

    Args:
        members: List of memory dictionaries with 'created_at', 'importance' keys.
        strategy: Selection strategy.
            - "keep_newest": Most recently created memory
            - "keep_oldest": Oldest memory (canonical/original)
            - "keep_highest_importance": Most important memory
            - "merge_content": Longest content (most comprehensive)

    Returns:
        Index of the representative memory within the list.
    """
    if not members:
        return 0

    if strategy == "keep_newest":
        return max(range(len(members)), key=lambda i: members[i].get("created_at", 0))
    elif strategy == "keep_oldest":
        return min(
            range(len(members)), key=lambda i: members[i].get("created_at", float("inf"))
        )
    elif strategy == "keep_highest_importance":
        return max(range(len(members)), key=lambda i: members[i].get("importance", 0))
    elif strategy == "merge_content":
        # For merge, pick the longest content as base
        return max(range(len(members)), key=lambda i: len(members[i].get("content", "")))
    else:
        return 0


def merge_memory_content(contents: list[str], separator: str = "\n\n---\n\n") -> str:
    """
    Merge multiple memory contents into one.

    Combines content from multiple memories, removing duplicates while
    preserving the order of first occurrence.

    Args:
        contents: List of content strings to merge.
        separator: Separator between merged contents.

    Returns:
        Merged content string with duplicates removed.
    """
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_contents: list[str] = []
    for content in contents:
        normalized = content.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_contents.append(content.strip())

    return separator.join(unique_contents)


def merge_memory_metadata(memories: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge metadata from multiple memories.

    Consolidates metadata intelligently:
    - created_at: earliest (preserve provenance)
    - last_accessed: latest (preserve recency)
    - access_count: sum (preserve total usage)
    - importance: max (preserve significance)
    - tags: union (preserve all categorization)
    - metadata: merged with special 'consolidated_from' field

    Args:
        memories: List of memory dictionaries.

    Returns:
        Merged metadata dictionary.
    """
    if not memories:
        return {}

    created_dates: list[Any] = [
        m.get("created_at") for m in memories if m.get("created_at") is not None
    ]
    accessed_dates: list[Any] = [
        m.get("last_accessed") for m in memories if m.get("last_accessed") is not None
    ]

    result: dict[str, Any] = {
        "created_at": min(created_dates) if created_dates else None,
        "last_accessed": max(accessed_dates) if accessed_dates else None,
        "access_count": sum(m.get("access_count", 0) for m in memories),
        "importance": max((m.get("importance", 0) for m in memories), default=0.5),
        "tags": list(set(tag for m in memories for tag in m.get("tags", []))),
        "source": "consolidated",
        "metadata": {
            "consolidated_from": [m.get("id") for m in memories if m.get("id")],
        },
    }

    return result


def merge_memories(
    memories: list[dict[str, Any]],
    vectors: list[NDArray[np.float32]],
    strategy: Literal["keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"],
) -> tuple[dict[str, Any], NDArray[np.float32]]:
    """
    Merge multiple memories into one according to strategy.

    Combines similar memories using the specified strategy for content
    selection, while intelligently merging metadata like timestamps,
    access counts, and tags.

    Args:
        memories: List of memory dictionaries to merge.
            Each must have: content, created_at, last_accessed,
            access_count, importance, tags, id.
        vectors: List of embedding vectors corresponding to memories.
        strategy: How to select/merge content.
            - "keep_newest": Use content from most recently created memory
            - "keep_oldest": Use content from oldest memory (canonical)
            - "keep_highest_importance": Use content from most important memory
            - "merge_content": Combine all content with separators

    Returns:
        Tuple of (merged_memory_dict, merged_vector) where:
        - merged_memory_dict contains the merged content and metadata
        - merged_vector is the weighted average of input vectors (normalized)

    Raises:
        ValueError: If memories list is empty or lengths mismatch.

    Example:
        >>> memories = [
        ...     {"content": "A", "created_at": dt1, "importance": 0.5, ...},
        ...     {"content": "B", "created_at": dt2, "importance": 0.8, ...},
        ... ]
        >>> vectors = [v1, v2]
        >>> merged, vec = merge_memories(memories, vectors, "keep_highest_importance")
        >>> merged["content"]  # "B" (higher importance)
    """
    if not memories:
        raise ValueError("Cannot merge empty list of memories")

    if len(memories) != len(vectors):
        raise ValueError(
            f"Mismatched lengths: memories={len(memories)}, vectors={len(vectors)}"
        )

    # Select primary memory based on strategy
    if strategy == "keep_newest":
        primary = max(memories, key=lambda m: m["created_at"])
        content = primary["content"]
    elif strategy == "keep_oldest":
        primary = min(memories, key=lambda m: m["created_at"])
        content = primary["content"]
    elif strategy == "keep_highest_importance":
        primary = max(memories, key=lambda m: m.get("importance", 0.5))
        content = primary["content"]
    elif strategy == "merge_content":
        # Combine all content with separator
        contents = [m["content"] for m in memories]
        content = merge_memory_content(contents)
    else:
        # Default to keeping highest importance
        logger.warning("Unknown strategy '%s', using keep_highest_importance", strategy)
        primary = max(memories, key=lambda m: m.get("importance", 0.5))
        content = primary["content"]

    # Merge metadata from all memories
    merged = merge_memory_metadata(memories)
    merged["content"] = content

    # Calculate merged vector as weighted average by content length
    vectors_array = np.array(vectors, dtype=np.float32)
    weights = np.array([len(m["content"]) for m in memories], dtype=np.float32)
    total_weight = weights.sum()

    if total_weight > 0:
        weights = weights / total_weight
        merged_vector = np.sum(vectors_array * weights[:, np.newaxis], axis=0)
    else:
        # Fallback to simple average
        merged_vector = np.mean(vectors_array, axis=0)

    # Normalize the merged vector
    norm = np.linalg.norm(merged_vector)
    if norm > 1e-10:
        merged_vector = merged_vector / norm

    return merged, merged_vector.astype(np.float32)


def cosine_similarity_matrix(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute pairwise cosine similarity matrix for a set of vectors.

    Efficient batch computation of cosine similarity between all pairs
    of vectors using matrix multiplication.

    Args:
        vectors: 2D array of shape (n_vectors, embedding_dim).

    Returns:
        2D array of shape (n_vectors, n_vectors) containing pairwise
        cosine similarities. Values range from -1 to 1.

    Example:
        >>> vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        >>> sim = cosine_similarity_matrix(vectors)
        >>> abs(sim[0, 0] - 1.0) < 0.001  # Self-similarity = 1
        True
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors / norms

    # Cosine similarity = dot product of normalized vectors
    similarity: NDArray[np.float32] = np.dot(normalized, normalized.T).astype(np.float32)
    return similarity
