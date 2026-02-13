"""Tests for cross-corpus scoring strategies."""

import pytest

from spatial_memory.core.scoring import (
    ScoringContext,
    VectorContentScoring,
    VectorMetadataScoring,
    VectorOnlyScoring,
    get_scoring_strategy,
    list_scoring_strategies,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _ctx(
    *,
    vector_similarity: float = 0.8,
    query_content: str | None = "hello world test",
    candidate_content: str = "hello world example",
    candidate_importance: float = 0.5,
    candidate_tags: list[str] | None = None,
    candidate_metadata: dict | None = None,
    query_namespace: str | None = "ns-a",
    candidate_namespace: str = "ns-b",
    query_project: str | None = "proj-1",
    candidate_project: str = "proj-2",
) -> ScoringContext:
    """Create a ScoringContext with sensible defaults."""
    return ScoringContext(
        vector_similarity=vector_similarity,
        query_content=query_content,
        candidate_content=candidate_content,
        query_namespace=query_namespace,
        candidate_namespace=candidate_namespace,
        query_project=query_project,
        candidate_project=candidate_project,
        candidate_importance=candidate_importance,
        candidate_tags=candidate_tags or [],
        candidate_metadata=candidate_metadata or {},
    )


# =============================================================================
# VectorOnlyScoring Tests
# =============================================================================


@pytest.mark.unit
class TestVectorOnlyScoring:
    """Tests for VectorOnlyScoring strategy."""

    def test_name(self) -> None:
        assert VectorOnlyScoring().name == "vector_only"

    def test_passthrough(self) -> None:
        """Should return raw vector similarity unchanged."""
        strategy = VectorOnlyScoring()
        assert strategy.score(_ctx(vector_similarity=0.85)) == 0.85

    def test_zero(self) -> None:
        assert VectorOnlyScoring().score(_ctx(vector_similarity=0.0)) == 0.0

    def test_one(self) -> None:
        assert VectorOnlyScoring().score(_ctx(vector_similarity=1.0)) == 1.0

    def test_ignores_content(self) -> None:
        """Content should have no effect on score."""
        strategy = VectorOnlyScoring()
        s1 = strategy.score(_ctx(query_content="aaa", candidate_content="zzz"))
        s2 = strategy.score(_ctx(query_content="aaa", candidate_content="aaa"))
        assert s1 == s2


# =============================================================================
# VectorContentScoring Tests
# =============================================================================


@pytest.mark.unit
class TestVectorContentScoring:
    """Tests for VectorContentScoring strategy."""

    def test_name(self) -> None:
        assert VectorContentScoring().name == "vector_content"

    def test_identical_content_boosts(self) -> None:
        """Identical content (jaccard=1.0) should boost the score."""
        strategy = VectorContentScoring(content_weight=0.3)
        ctx = _ctx(
            vector_similarity=0.7,
            query_content="hello world",
            candidate_content="hello world",
        )
        score = strategy.score(ctx)
        # (1-0.3)*0.7 + 0.3*1.0 = 0.49 + 0.3 = 0.79
        assert score == pytest.approx(0.79, abs=1e-6)

    def test_disjoint_content_lowers(self) -> None:
        """Completely disjoint content (jaccard=0) should lower the score."""
        strategy = VectorContentScoring(content_weight=0.3)
        ctx = _ctx(
            vector_similarity=0.7,
            query_content="alpha beta",
            candidate_content="gamma delta",
        )
        score = strategy.score(ctx)
        # (1-0.3)*0.7 + 0.3*0.0 = 0.49
        assert score == pytest.approx(0.49, abs=1e-6)

    def test_no_query_content_falls_back_to_vector(self) -> None:
        """When query_content is None, should return raw vector similarity."""
        strategy = VectorContentScoring(content_weight=0.3)
        ctx = _ctx(vector_similarity=0.9, query_content=None)
        assert strategy.score(ctx) == 0.9

    def test_custom_content_weight(self) -> None:
        """Custom weight should shift the blend."""
        strategy = VectorContentScoring(content_weight=0.5)
        ctx = _ctx(
            vector_similarity=0.8,
            query_content="hello world",
            candidate_content="hello world",
        )
        score = strategy.score(ctx)
        # (1-0.5)*0.8 + 0.5*1.0 = 0.4 + 0.5 = 0.9
        assert score == pytest.approx(0.9, abs=1e-6)

    def test_matches_combined_similarity(self) -> None:
        """Should produce the same result as lifecycle_ops.combined_similarity."""
        from spatial_memory.core.lifecycle_ops import (
            combined_similarity,
            jaccard_similarity,
        )

        strategy = VectorContentScoring(content_weight=0.3)
        q = "the quick brown fox"
        c = "the quick red fox"
        vec_sim = 0.75
        ctx = _ctx(vector_similarity=vec_sim, query_content=q, candidate_content=c)

        expected = combined_similarity(vec_sim, jaccard_similarity(q, c), 0.3)
        assert strategy.score(ctx) == pytest.approx(expected, abs=1e-9)


# =============================================================================
# VectorMetadataScoring Tests
# =============================================================================


@pytest.mark.unit
class TestVectorMetadataScoring:
    """Tests for VectorMetadataScoring strategy."""

    def test_name(self) -> None:
        assert VectorMetadataScoring().name == "vector_metadata"

    def test_no_tags_no_importance(self) -> None:
        """With no tags and zero importance, score equals vector_similarity."""
        strategy = VectorMetadataScoring()
        ctx = _ctx(
            vector_similarity=0.8,
            candidate_tags=[],
            candidate_importance=0.0,
            candidate_metadata={},
        )
        assert strategy.score(ctx) == 0.8

    def test_shared_tags_boost(self) -> None:
        """Shared tags should boost the score above raw vector similarity."""
        strategy = VectorMetadataScoring(tag_weight=0.15, importance_weight=0.0)
        ctx = _ctx(
            vector_similarity=0.8,
            candidate_tags=["python", "testing"],
            candidate_importance=0.0,
            candidate_metadata={"query_tags": ["python", "testing"]},
        )
        score = strategy.score(ctx)
        # tag overlap = 2/2 = 1.0, bonus = 1.0 * 0.15 = 0.15
        # final = 0.8 + (1 - 0.8) * 0.15 = 0.8 + 0.03 = 0.83
        assert score == pytest.approx(0.83, abs=1e-6)

    def test_partial_tag_overlap(self) -> None:
        """Partial tag overlap should give partial boost."""
        strategy = VectorMetadataScoring(tag_weight=0.15, importance_weight=0.0)
        ctx = _ctx(
            vector_similarity=0.8,
            candidate_tags=["python", "testing"],
            candidate_importance=0.0,
            candidate_metadata={"query_tags": ["python", "docs"]},
        )
        score = strategy.score(ctx)
        # overlap = 1/3 (python in both, testing and docs unique)
        # bonus = (1/3) * 0.15 = 0.05
        # final = 0.8 + 0.2 * 0.05 = 0.81
        assert score == pytest.approx(0.81, abs=1e-6)

    def test_importance_boost(self) -> None:
        """High importance should boost the score."""
        strategy = VectorMetadataScoring(tag_weight=0.0, importance_weight=0.1)
        ctx = _ctx(
            vector_similarity=0.8,
            candidate_tags=[],
            candidate_importance=1.0,
            candidate_metadata={},
        )
        score = strategy.score(ctx)
        # bonus = 1.0 * 0.1 = 0.1
        # final = 0.8 + 0.2 * 0.1 = 0.82
        assert score == pytest.approx(0.82, abs=1e-6)

    def test_never_exceeds_one(self) -> None:
        """Score should never exceed 1.0 even with maximum boosts."""
        strategy = VectorMetadataScoring(tag_weight=0.5, importance_weight=0.5)
        ctx = _ctx(
            vector_similarity=0.99,
            candidate_tags=["a"],
            candidate_importance=1.0,
            candidate_metadata={"query_tags": ["a"]},
        )
        score = strategy.score(ctx)
        assert score <= 1.0

    def test_no_query_tags_in_metadata(self) -> None:
        """Missing query_tags in metadata should still work (no tag bonus)."""
        strategy = VectorMetadataScoring(tag_weight=0.15, importance_weight=0.0)
        ctx = _ctx(
            vector_similarity=0.8,
            candidate_tags=["python"],
            candidate_importance=0.0,
            candidate_metadata={},
        )
        assert strategy.score(ctx) == 0.8


# =============================================================================
# ScoringContext Tests
# =============================================================================


@pytest.mark.unit
class TestScoringContext:
    """Tests for the ScoringContext dataclass."""

    def test_frozen_immutability(self) -> None:
        ctx = _ctx()
        with pytest.raises(AttributeError):
            ctx.vector_similarity = 0.1  # type: ignore[misc]

    def test_slots_no_dict(self) -> None:
        ctx = _ctx()
        assert not hasattr(ctx, "__dict__")


# =============================================================================
# Registry Tests
# =============================================================================


@pytest.mark.unit
class TestScoringRegistry:
    """Tests for the scoring strategy registry."""

    def test_get_vector_only(self) -> None:
        strategy = get_scoring_strategy("vector_only")
        assert strategy.name == "vector_only"

    def test_get_vector_content(self) -> None:
        strategy = get_scoring_strategy("vector_content")
        assert strategy.name == "vector_content"

    def test_get_vector_metadata(self) -> None:
        strategy = get_scoring_strategy("vector_metadata")
        assert strategy.name == "vector_metadata"

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown scoring strategy"):
            get_scoring_strategy("nonexistent")

    def test_custom_kwargs_creates_new_instance(self) -> None:
        """Passing kwargs should create a fresh instance, not return singleton."""
        default = get_scoring_strategy("vector_content")
        custom = get_scoring_strategy("vector_content", content_weight=0.5)
        assert default is not custom

    def test_custom_kwargs_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scoring strategy"):
            get_scoring_strategy("bad_name", foo=1)

    def test_list_strategies(self) -> None:
        names = list_scoring_strategies()
        assert "vector_only" in names
        assert "vector_content" in names
        assert "vector_metadata" in names
        assert len(names) == 3


# =============================================================================
# Parametrized Cross-Strategy Tests
# =============================================================================


@pytest.mark.unit
class TestAllStrategies:
    """Tests that apply to every scoring strategy."""

    @pytest.mark.parametrize("name", ["vector_only", "vector_content", "vector_metadata"])
    def test_score_in_range(self, name: str) -> None:
        """All strategies should return scores in [0, 1]."""
        strategy = get_scoring_strategy(name)
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ctx = _ctx(
                vector_similarity=v,
                candidate_tags=["a", "b"],
                candidate_importance=0.8,
                candidate_metadata={"query_tags": ["a", "c"]},
            )
            score = strategy.score(ctx)
            assert 0.0 <= score <= 1.0, f"{name} returned {score} for input {v}"

    @pytest.mark.parametrize("name", ["vector_only", "vector_content", "vector_metadata"])
    def test_zero_vector_returns_low_score(self, name: str) -> None:
        """Zero vector similarity should produce a low score."""
        strategy = get_scoring_strategy(name)
        ctx = _ctx(vector_similarity=0.0, candidate_importance=0.0, candidate_metadata={})
        score = strategy.score(ctx)
        assert score <= 0.2, f"{name} returned {score} for zero vector sim"

    @pytest.mark.parametrize("name", ["vector_only", "vector_content", "vector_metadata"])
    def test_perfect_vector_returns_high_score(self, name: str) -> None:
        """Perfect vector similarity should produce a high score."""
        strategy = get_scoring_strategy(name)
        ctx = _ctx(
            vector_similarity=1.0,
            query_content="hello world",
            candidate_content="hello world",
        )
        score = strategy.score(ctx)
        assert score >= 0.9, f"{name} returned {score} for perfect vector sim"
