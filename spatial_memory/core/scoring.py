"""Pluggable scoring strategies for cross-corpus similarity.

Pure computation layer -- no I/O, no database access.

Each strategy takes a ScoringContext with all available data and returns
a final similarity score in [0, 1]. Strategies are registered in a
module-level registry and looked up by name string.

Follows the same registry pattern as consolidation_strategies.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from spatial_memory.core.lifecycle_ops import jaccard_similarity

# =============================================================================
# Scoring Context
# =============================================================================


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """All available data for scoring a candidate match.

    Immutable value object passed to strategies so they can pick
    whichever signals they need without coupling to the data source.
    """

    vector_similarity: float
    query_content: str | None
    candidate_content: str
    query_namespace: str | None
    candidate_namespace: str
    query_project: str | None
    candidate_project: str
    candidate_importance: float
    candidate_tags: list[str]
    candidate_metadata: dict[str, Any]


# =============================================================================
# Strategy Protocol
# =============================================================================


class ScoringStrategy(Protocol):
    """Strategy for computing a final similarity score."""

    @property
    def name(self) -> str:
        """Unique identifier for this strategy."""
        ...

    def score(self, ctx: ScoringContext) -> float:
        """Compute final similarity from context.

        Args:
            ctx: All available data about the query and candidate.

        Returns:
            Final similarity score in [0, 1].
        """
        ...


# =============================================================================
# Strategy Implementations
# =============================================================================


class VectorOnlyScoring:
    """Pass-through cosine similarity.

    Fastest strategy -- no additional computation. Best for proactive
    surfacing where raw semantic similarity is sufficient.
    """

    @property
    def name(self) -> str:
        return "vector_only"

    def score(self, ctx: ScoringContext) -> float:
        return ctx.vector_similarity


class VectorContentScoring:
    """Weighted blend of vector similarity and content overlap.

    Uses Jaccard word overlap as lexical signal, matching the existing
    combined_similarity() formula: (1 - w) * vector + w * jaccard.

    Best for consolidation/dedup where lexical overlap matters.
    """

    def __init__(self, content_weight: float = 0.3) -> None:
        self._content_weight = content_weight

    @property
    def name(self) -> str:
        return "vector_content"

    def score(self, ctx: ScoringContext) -> float:
        if ctx.query_content is None:
            return ctx.vector_similarity

        content_overlap = jaccard_similarity(ctx.query_content, ctx.candidate_content)
        w = self._content_weight
        return (1 - w) * ctx.vector_similarity + w * content_overlap


class VectorMetadataScoring:
    """Vector similarity boosted by tag overlap and importance.

    Adds a metadata bonus based on shared tags and candidate importance.
    Best for cross-app discovery where shared tags signal meaningful links.

    Formula:
        base = vector_similarity
        tag_bonus = tag_overlap_ratio * tag_weight
        importance_bonus = candidate_importance * importance_weight
        final = base + (1 - base) * (tag_bonus + importance_bonus)

    The (1 - base) factor prevents boosting past 1.0 and ensures
    high-similarity matches aren't over-boosted.
    """

    def __init__(
        self,
        tag_weight: float = 0.15,
        importance_weight: float = 0.05,
    ) -> None:
        self._tag_weight = tag_weight
        self._importance_weight = importance_weight

    @property
    def name(self) -> str:
        return "vector_metadata"

    def score(self, ctx: ScoringContext) -> float:
        base = ctx.vector_similarity

        # Tag overlap bonus
        tag_bonus = 0.0
        query_tags = ctx.candidate_metadata.get("query_tags")
        if query_tags and ctx.candidate_tags:
            query_set = set(query_tags) if isinstance(query_tags, list) else set()
            candidate_set = set(ctx.candidate_tags)
            union = query_set | candidate_set
            if union:
                tag_bonus = len(query_set & candidate_set) / len(union) * self._tag_weight

        # Importance bonus
        importance_bonus = ctx.candidate_importance * self._importance_weight

        # Boost from remaining headroom so result stays in [0, 1]
        return base + (1 - base) * (tag_bonus + importance_bonus)


# =============================================================================
# Strategy Registry
# =============================================================================

_STRATEGY_REGISTRY: dict[str, ScoringStrategy] = {
    "vector_only": VectorOnlyScoring(),
    "vector_content": VectorContentScoring(),
    "vector_metadata": VectorMetadataScoring(),
}


def get_scoring_strategy(name: str, **kwargs: Any) -> ScoringStrategy:
    """Get a scoring strategy by name.

    For the default instances (no custom kwargs), returns the cached
    singleton from the registry. When kwargs are provided, creates
    a fresh instance with custom parameters.

    Args:
        name: Strategy name (vector_only, vector_content, vector_metadata).
        **kwargs: Optional parameters forwarded to the strategy constructor.

    Returns:
        ScoringStrategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    if kwargs:
        factory = _STRATEGY_CLASSES.get(name)
        if factory is None:
            valid = ", ".join(_STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown scoring strategy: {name}. Valid: {valid}")
        return factory(**kwargs)

    strategy = _STRATEGY_REGISTRY.get(name)
    if strategy is None:
        valid = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown scoring strategy: {name}. Valid: {valid}")
    return strategy


def list_scoring_strategies() -> list[str]:
    """List all registered scoring strategy names."""
    return list(_STRATEGY_REGISTRY.keys())


# Class registry for custom-parameter instantiation
_StrategyType = type[VectorOnlyScoring] | type[VectorContentScoring] | type[VectorMetadataScoring]
_STRATEGY_CLASSES: dict[str, _StrategyType] = {
    "vector_only": VectorOnlyScoring,
    "vector_content": VectorContentScoring,
    "vector_metadata": VectorMetadataScoring,
}
