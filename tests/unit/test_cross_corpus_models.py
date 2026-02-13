"""Tests for cross-corpus similarity data models and port protocols."""

from datetime import datetime, timezone

import pytest

from spatial_memory.core.models import (
    BatchSimilarityResult,
    CorpusSimilaritySummary,
    CrossCorpusMatch,
    SimilarityConfig,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_match(
    *,
    memory_id: str = "mem-1",
    similarity: float = 0.9,
    namespace: str = "ns-a",
    project: str = "proj-1",
    query_namespace: str | None = "ns-b",
    scoring_strategy: str = "vector_only",
) -> CrossCorpusMatch:
    """Create a CrossCorpusMatch for testing."""
    return CrossCorpusMatch(
        memory_id=memory_id,
        content="test content",
        similarity=similarity,
        raw_vector_similarity=similarity,
        namespace=namespace,
        project=project,
        importance=0.5,
        tags=["tag1"],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        scoring_strategy=scoring_strategy,
        query_namespace=query_namespace,
    )


# =============================================================================
# CrossCorpusMatch Tests
# =============================================================================


@pytest.mark.unit
class TestCrossCorpusMatch:
    """Tests for CrossCorpusMatch dataclass."""

    def test_create_with_required_fields(self) -> None:
        """CrossCorpusMatch should store all required fields."""
        match = _make_match()
        assert match.memory_id == "mem-1"
        assert match.similarity == 0.9
        assert match.namespace == "ns-a"
        assert match.project == "proj-1"
        assert match.scoring_strategy == "vector_only"

    def test_optional_fields_default_to_none(self) -> None:
        """Optional query provenance fields should default to None."""
        match = CrossCorpusMatch(
            memory_id="m1",
            content="c",
            similarity=0.5,
            raw_vector_similarity=0.5,
            namespace="ns",
            project="p",
            importance=0.5,
            tags=[],
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            scoring_strategy="vector_only",
        )
        assert match.query_namespace is None
        assert match.query_project is None
        assert match.query_memory_id is None

    def test_frozen_immutability(self) -> None:
        """CrossCorpusMatch should be immutable (frozen dataclass)."""
        match = _make_match()
        with pytest.raises(AttributeError):
            match.similarity = 0.1  # type: ignore[misc]

    def test_slots_no_dict(self) -> None:
        """Slots dataclass should not have __dict__."""
        match = _make_match()
        assert not hasattr(match, "__dict__")

    def test_equality(self) -> None:
        """Two matches with same values should be equal."""
        m1 = _make_match(memory_id="a", similarity=0.8)
        m2 = _make_match(memory_id="a", similarity=0.8)
        assert m1 == m2

    def test_inequality(self) -> None:
        """Matches with different values should not be equal."""
        m1 = _make_match(memory_id="a")
        m2 = _make_match(memory_id="b")
        assert m1 != m2


# =============================================================================
# BatchSimilarityResult Tests
# =============================================================================


@pytest.mark.unit
class TestBatchSimilarityResult:
    """Tests for BatchSimilarityResult dataclass."""

    def test_top_match_returns_first(self) -> None:
        """top_match should return the first match in the list."""
        matches = [_make_match(similarity=0.9), _make_match(similarity=0.7)]
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=matches)
        assert result.top_match is not None
        assert result.top_match.similarity == 0.9

    def test_top_match_returns_none_when_empty(self) -> None:
        """top_match should return None for empty matches."""
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=[])
        assert result.top_match is None

    def test_cross_namespace_count_all_different(self) -> None:
        """cross_namespace_count should count matches from different namespaces."""
        matches = [
            _make_match(namespace="ns-x", query_namespace="ns-a"),
            _make_match(namespace="ns-y", query_namespace="ns-a"),
            _make_match(namespace="ns-z", query_namespace="ns-a"),
        ]
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=matches)
        assert result.cross_namespace_count == 3

    def test_cross_namespace_count_some_same(self) -> None:
        """cross_namespace_count should exclude matches from the query namespace."""
        matches = [
            _make_match(namespace="ns-a", query_namespace="ns-a"),  # Same as query
            _make_match(namespace="ns-b", query_namespace="ns-a"),  # Different
        ]
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=matches)
        assert result.cross_namespace_count == 1

    def test_cross_namespace_count_empty(self) -> None:
        """cross_namespace_count should be 0 for empty matches."""
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=[])
        assert result.cross_namespace_count == 0

    def test_frozen_immutability(self) -> None:
        """BatchSimilarityResult should be immutable."""
        result = BatchSimilarityResult(query_index=0, query_memory_id="q1", matches=[])
        with pytest.raises(AttributeError):
            result.query_index = 1  # type: ignore[misc]


# =============================================================================
# CorpusSimilaritySummary Tests
# =============================================================================


@pytest.mark.unit
class TestCorpusSimilaritySummary:
    """Tests for CorpusSimilaritySummary dataclass."""

    def test_create_with_defaults(self) -> None:
        """CorpusSimilaritySummary should have sensible defaults."""
        summary = CorpusSimilaritySummary(
            total_memories_analyzed=100,
            namespaces_analyzed=["ns-a", "ns-b"],
            bridges_found=5,
            potential_duplicates=2,
        )
        assert summary.total_memories_analyzed == 100
        assert summary.top_bridges == []

    def test_create_with_bridges(self) -> None:
        """CorpusSimilaritySummary should store top bridges."""
        bridges = [_make_match(similarity=0.95)]
        summary = CorpusSimilaritySummary(
            total_memories_analyzed=100,
            namespaces_analyzed=["ns-a"],
            bridges_found=1,
            potential_duplicates=0,
            top_bridges=bridges,
        )
        assert len(summary.top_bridges) == 1
        assert summary.top_bridges[0].similarity == 0.95


# =============================================================================
# SimilarityConfig Tests
# =============================================================================


@pytest.mark.unit
class TestSimilarityConfig:
    """Tests for SimilarityConfig dataclass."""

    def test_defaults(self) -> None:
        """SimilarityConfig should have reasonable defaults."""
        config = SimilarityConfig()
        assert config.default_min_similarity == 0.5
        assert config.default_limit == 10
        assert config.max_batch_size == 100
        assert config.scoring_strategy == "vector_only"
        assert config.content_weight == 0.3
        assert config.ann_candidate_multiplier == 3

    def test_custom_values(self) -> None:
        """SimilarityConfig should accept custom values."""
        config = SimilarityConfig(
            default_min_similarity=0.8,
            default_limit=20,
            max_batch_size=50,
            scoring_strategy="vector_content",
            content_weight=0.5,
            ann_candidate_multiplier=5,
        )
        assert config.default_min_similarity == 0.8
        assert config.default_limit == 20
        assert config.scoring_strategy == "vector_content"
        assert config.ann_candidate_multiplier == 5


# =============================================================================
# Port Protocol Compliance Tests
# =============================================================================


@pytest.mark.unit
class TestPortProtocolCompliance:
    """Verify that protocol interfaces are importable and well-formed."""

    def test_similarity_query_port_importable(self) -> None:
        """SimilarityQueryPort should be importable."""
        from spatial_memory.ports.similarity import SimilarityQueryPort

        assert SimilarityQueryPort is not None

    def test_batch_similarity_port_importable(self) -> None:
        """BatchSimilarityPort should be importable."""
        from spatial_memory.ports.similarity import BatchSimilarityPort

        assert BatchSimilarityPort is not None

    def test_corpus_analysis_port_importable(self) -> None:
        """CorpusAnalysisPort should be importable."""
        from spatial_memory.ports.similarity import CorpusAnalysisPort

        assert CorpusAnalysisPort is not None

    def test_ports_exported_from_package(self) -> None:
        """All ports should be accessible from the ports package."""
        from spatial_memory.ports import (
            BatchSimilarityPort,
            CorpusAnalysisPort,
            SimilarityQueryPort,
        )

        assert SimilarityQueryPort is not None
        assert BatchSimilarityPort is not None
        assert CorpusAnalysisPort is not None

    def test_similarity_query_port_has_find_method(self) -> None:
        """SimilarityQueryPort should define find_similar_across_corpus."""
        from spatial_memory.ports.similarity import SimilarityQueryPort

        assert hasattr(SimilarityQueryPort, "find_similar_across_corpus")

    def test_batch_similarity_port_has_batch_method(self) -> None:
        """BatchSimilarityPort should define find_similar_batch."""
        from spatial_memory.ports.similarity import BatchSimilarityPort

        assert hasattr(BatchSimilarityPort, "find_similar_batch")

    def test_corpus_analysis_port_has_bridge_method(self) -> None:
        """CorpusAnalysisPort should define find_cross_namespace_bridges."""
        from spatial_memory.ports.similarity import CorpusAnalysisPort

        assert hasattr(CorpusAnalysisPort, "find_cross_namespace_bridges")

    def test_corpus_analysis_port_has_summary_method(self) -> None:
        """CorpusAnalysisPort should define get_corpus_overlap_summary."""
        from spatial_memory.ports.similarity import CorpusAnalysisPort

        assert hasattr(CorpusAnalysisPort, "get_corpus_overlap_summary")
