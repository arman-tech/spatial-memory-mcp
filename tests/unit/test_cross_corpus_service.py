"""Tests for CrossCorpusSimilarityService."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.core.errors import MemoryNotFoundError
from spatial_memory.core.models import (
    Memory,
    MemoryResult,
    MemorySource,
    SimilarityConfig,
)
from spatial_memory.core.scoring import VectorContentScoring, VectorOnlyScoring
from spatial_memory.services.similarity import CrossCorpusSimilarityService

# =============================================================================
# Fixtures
# =============================================================================

_NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _make_result(
    *,
    id: str = "mem-1",
    content: str = "test content",
    similarity: float = 0.9,
    namespace: str = "ns-a",
    project: str = "proj-1",
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> MemoryResult:
    """Create a MemoryResult for testing."""
    return MemoryResult(
        id=id,
        content=content,
        similarity=similarity,
        namespace=namespace,
        project=project,
        importance=importance,
        tags=tags or [],
        created_at=_NOW,
        metadata={},
    )


def _make_raw_dict(
    *,
    id: str = "mem-1",
    content: str = "test content",
    similarity: float = 0.9,
    namespace: str = "ns-a",
    project: str = "proj-1",
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> dict:
    """Create a raw dict result for batch_vector_search."""
    return {
        "id": id,
        "content": content,
        "similarity": similarity,
        "namespace": namespace,
        "project": project,
        "importance": importance,
        "tags": tags or [],
        "created_at": _NOW,
        "metadata": {},
    }


def _make_service(
    search_results: list[MemoryResult] | None = None,
    batch_results: list[list[dict]] | None = None,
    namespaces: list[str] | None = None,
    namespace_counts: dict[str, int] | None = None,
    clustering_data: dict[str, tuple[list[str], np.ndarray]] | None = None,
    config: SimilarityConfig | None = None,
    scoring_strategy=None,
    memory_repository=None,
) -> CrossCorpusSimilarityService:
    """Create a service with mocked dependencies."""
    repo = MagicMock()
    repo.search.return_value = search_results or []
    repo.batch_vector_search.return_value = batch_results or []

    if clustering_data:

        def _get_vectors(namespace=None, project=None, max_memories=10000):
            if namespace in clustering_data:
                return clustering_data[namespace]
            return ([], np.array([]))

        repo.get_vectors_for_clustering.side_effect = _get_vectors

    ns_provider = MagicMock()
    ns_provider.get_namespaces.return_value = namespaces or []
    if namespace_counts:
        ns_provider.count.side_effect = lambda namespace=None: namespace_counts.get(namespace, 0)
    else:
        ns_provider.count.return_value = 0

    return CrossCorpusSimilarityService(
        repository=repo,
        namespace_provider=ns_provider,
        config=config or SimilarityConfig(),
        scoring_strategy=scoring_strategy,
        memory_repository=memory_repository,
    )


# =============================================================================
# find_similar_across_corpus Tests
# =============================================================================


@pytest.mark.unit
class TestFindSimilarAcrossCorpus:
    """Tests for find_similar_across_corpus."""

    def test_returns_matches_sorted_by_similarity(self) -> None:
        results = [
            _make_result(id="a", similarity=0.7),
            _make_result(id="b", similarity=0.9),
            _make_result(id="c", similarity=0.8),
        ]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(np.zeros(384), min_similarity=0.5)

        assert [m.memory_id for m in matches] == ["b", "c", "a"]

    def test_respects_min_similarity_filter(self) -> None:
        results = [
            _make_result(id="a", similarity=0.9),
            _make_result(id="b", similarity=0.4),
        ]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(np.zeros(384), min_similarity=0.5)

        assert len(matches) == 1
        assert matches[0].memory_id == "a"

    def test_excludes_namespace(self) -> None:
        results = [
            _make_result(id="a", namespace="ns-skip"),
            _make_result(id="b", namespace="ns-keep"),
        ]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            exclude_namespace="ns-skip",
        )

        assert len(matches) == 1
        assert matches[0].memory_id == "b"

    def test_excludes_project(self) -> None:
        results = [
            _make_result(id="a", project="proj-skip"),
            _make_result(id="b", project="proj-keep"),
        ]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            exclude_project="proj-skip",
        )

        assert len(matches) == 1
        assert matches[0].memory_id == "b"

    def test_excludes_ids(self) -> None:
        results = [
            _make_result(id="skip-me", similarity=0.95),
            _make_result(id="keep-me", similarity=0.8),
        ]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            exclude_ids={"skip-me"},
        )

        assert len(matches) == 1
        assert matches[0].memory_id == "keep-me"

    def test_respects_limit(self) -> None:
        results = [_make_result(id=f"m-{i}", similarity=0.9 - i * 0.01) for i in range(10)]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(np.zeros(384), limit=3, min_similarity=0.0)

        assert len(matches) == 3

    def test_overfetches_candidates(self) -> None:
        """Should request limit * ann_candidate_multiplier candidates."""
        svc = _make_service(config=SimilarityConfig(ann_candidate_multiplier=5))

        svc.find_similar_across_corpus(np.zeros(384), limit=10)

        call_args = svc._repo.search.call_args
        assert call_args.kwargs.get("limit") == 50 or call_args[1].get("limit") == 50

    def test_tracks_provenance(self) -> None:
        results = [_make_result(id="a", similarity=0.9)]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            query_namespace="source-ns",
            query_project="source-proj",
        )

        assert matches[0].query_namespace == "source-ns"
        assert matches[0].query_project == "source-proj"

    def test_records_scoring_strategy(self) -> None:
        results = [_make_result(id="a", similarity=0.9)]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(np.zeros(384), min_similarity=0.0)

        assert matches[0].scoring_strategy == "vector_only"

    def test_preserves_raw_vector_similarity(self) -> None:
        results = [_make_result(id="a", similarity=0.85)]
        svc = _make_service(search_results=results)

        matches = svc.find_similar_across_corpus(np.zeros(384), min_similarity=0.0)

        assert matches[0].raw_vector_similarity == 0.85


# =============================================================================
# find_similar_batch Tests
# =============================================================================


@pytest.mark.unit
class TestFindSimilarBatch:
    """Tests for find_similar_batch."""

    def test_returns_parallel_results(self) -> None:
        batch = [
            [_make_raw_dict(id="a1", similarity=0.9)],
            [_make_raw_dict(id="b1", similarity=0.8)],
        ]
        svc = _make_service(batch_results=batch)

        results = svc.find_similar_batch(
            [np.zeros(384), np.zeros(384)],
            min_similarity=0.0,
        )

        assert len(results) == 2
        assert results[0].query_index == 0
        assert results[1].query_index == 1

    def test_applies_per_query_exclusions(self) -> None:
        batch = [
            [
                _make_raw_dict(id="skip", similarity=0.9),
                _make_raw_dict(id="keep", similarity=0.8),
            ],
        ]
        svc = _make_service(batch_results=batch)

        results = svc.find_similar_batch(
            [np.zeros(384)],
            min_similarity=0.0,
            exclude_ids=[{"skip"}],
        )

        assert len(results[0].matches) == 1
        assert results[0].matches[0].memory_id == "keep"

    def test_enforces_batch_size_limit(self) -> None:
        svc = _make_service(config=SimilarityConfig(max_batch_size=2))

        with pytest.raises(ValueError, match="exceeds maximum"):
            svc.find_similar_batch(
                [np.zeros(384)] * 3,
                min_similarity=0.0,
            )

    def test_assigns_query_memory_ids(self) -> None:
        batch = [[_make_raw_dict(id="a1", similarity=0.9)]]
        svc = _make_service(batch_results=batch)

        results = svc.find_similar_batch(
            [np.zeros(384)],
            min_similarity=0.0,
            query_memory_ids=["source-1"],
        )

        assert results[0].query_memory_id == "source-1"

    def test_empty_batch(self) -> None:
        svc = _make_service(batch_results=[])

        results = svc.find_similar_batch([], min_similarity=0.0)

        assert results == []

    def test_min_similarity_filters_batch(self) -> None:
        batch = [
            [
                _make_raw_dict(id="high", similarity=0.9),
                _make_raw_dict(id="low", similarity=0.3),
            ],
        ]
        svc = _make_service(batch_results=batch)

        results = svc.find_similar_batch(
            [np.zeros(384)],
            min_similarity=0.5,
        )

        assert len(results[0].matches) == 1
        assert results[0].matches[0].memory_id == "high"


# =============================================================================
# find_cross_namespace_bridges Tests
# =============================================================================


@pytest.mark.unit
class TestFindCrossNamespaceBridges:
    """Tests for find_cross_namespace_bridges."""

    def test_finds_bridges_between_namespaces(self) -> None:
        clustering_data = {
            "ns-a": (["m1"], np.array([[1.0, 0.0]])),
            "ns-b": (["m2"], np.array([[0.9, 0.1]])),
        }
        batch_results_for_ns_a = [[_make_raw_dict(id="m2", namespace="ns-b", similarity=0.95)]]
        batch_results_for_ns_b = [[_make_raw_dict(id="m1", namespace="ns-a", similarity=0.95)]]

        repo = MagicMock()

        call_count = {"n": 0}

        def _batch_search(query_vectors, limit_per_query=3, namespace=None, project=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return batch_results_for_ns_a
            return batch_results_for_ns_b

        repo.batch_vector_search.side_effect = _batch_search
        repo.get_vectors_for_clustering.side_effect = (
            lambda namespace=None, project=None, max_memories=10000: clustering_data.get(
                namespace, ([], np.array([]))
            )
        )

        ns_provider = MagicMock()
        ns_provider.get_namespaces.return_value = ["ns-a", "ns-b"]

        svc = CrossCorpusSimilarityService(
            repository=repo,
            namespace_provider=ns_provider,
        )

        bridges = svc.find_cross_namespace_bridges(min_similarity=0.8)

        assert len(bridges) >= 1
        assert bridges[0].similarity >= 0.8

    def test_deduplicates_symmetric_pairs(self) -> None:
        """A->B and B->A should produce only one bridge."""
        clustering_data = {
            "ns-a": (["m1"], np.array([[1.0, 0.0]])),
            "ns-b": (["m2"], np.array([[0.9, 0.1]])),
        }

        repo = MagicMock()
        repo.batch_vector_search.return_value = [
            [_make_raw_dict(id="m2", namespace="ns-b", similarity=0.95)]
        ]
        repo.get_vectors_for_clustering.side_effect = (
            lambda namespace=None, project=None, max_memories=10000: clustering_data.get(
                namespace, ([], np.array([]))
            )
        )

        ns_provider = MagicMock()
        ns_provider.get_namespaces.return_value = ["ns-a", "ns-b"]

        svc = CrossCorpusSimilarityService(repository=repo, namespace_provider=ns_provider)

        bridges = svc.find_cross_namespace_bridges(min_similarity=0.8)

        # Both ns-a and ns-b search and find each other, but dedup means only 1
        bridge_ids = [(b.query_memory_id, b.memory_id) for b in bridges]
        normalized = set()
        for qid, mid in bridge_ids:
            normalized.add((min(qid or "", mid), max(qid or "", mid)))
        assert len(normalized) == len(bridges)

    def test_returns_empty_with_single_namespace(self) -> None:
        svc = _make_service(namespaces=["only-one"])

        bridges = svc.find_cross_namespace_bridges()

        assert bridges == []

    def test_respects_namespace_filter(self) -> None:
        svc = _make_service(namespaces=["ns-a", "ns-b", "ns-c"])

        bridges = svc.find_cross_namespace_bridges(namespace_filter=["ns-a"])

        # Only 1 namespace after filter, need >= 2
        assert bridges == []

    def test_respects_max_bridges(self) -> None:
        clustering_data = {
            "ns-a": (["m1", "m2", "m3"], np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])),
            "ns-b": (["m4", "m5", "m6"], np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])),
        }

        repo = MagicMock()
        repo.batch_vector_search.return_value = [
            [_make_raw_dict(id=f"x{i}", namespace="ns-b", similarity=0.9)] for i in range(3)
        ]
        repo.get_vectors_for_clustering.side_effect = (
            lambda namespace=None, project=None, max_memories=10000: clustering_data.get(
                namespace, ([], np.array([]))
            )
        )

        ns_provider = MagicMock()
        ns_provider.get_namespaces.return_value = ["ns-a", "ns-b"]

        svc = CrossCorpusSimilarityService(repository=repo, namespace_provider=ns_provider)

        bridges = svc.find_cross_namespace_bridges(min_similarity=0.5, max_bridges=2)

        assert len(bridges) <= 2


# =============================================================================
# get_corpus_overlap_summary Tests
# =============================================================================


@pytest.mark.unit
class TestGetCorpusOverlapSummary:
    """Tests for get_corpus_overlap_summary."""

    def test_returns_summary_with_counts(self) -> None:
        svc = _make_service(
            namespaces=["ns-a"],
            namespace_counts={"ns-a": 50},
        )

        summary = svc.get_corpus_overlap_summary()

        assert summary.total_memories_analyzed == 50
        assert summary.namespaces_analyzed == ["ns-a"]
        assert summary.bridges_found == 0

    def test_counts_potential_duplicates(self) -> None:
        """Bridges with similarity >= 0.95 should be counted as duplicates."""
        clustering_data = {
            "ns-a": (["m1"], np.array([[1.0, 0.0]])),
            "ns-b": (["m2"], np.array([[0.9, 0.1]])),
        }

        repo = MagicMock()
        repo.batch_vector_search.return_value = [
            [_make_raw_dict(id="m2", namespace="ns-b", similarity=0.97)]
        ]
        repo.get_vectors_for_clustering.side_effect = (
            lambda namespace=None, project=None, max_memories=10000: clustering_data.get(
                namespace, ([], np.array([]))
            )
        )

        ns_provider = MagicMock()
        ns_provider.get_namespaces.return_value = ["ns-a", "ns-b"]
        ns_provider.count.return_value = 10

        svc = CrossCorpusSimilarityService(repository=repo, namespace_provider=ns_provider)

        summary = svc.get_corpus_overlap_summary()

        assert summary.potential_duplicates >= 1


# =============================================================================
# find_similar_to_memory Tests
# =============================================================================


@pytest.mark.unit
class TestFindSimilarToMemory:
    """Tests for find_similar_to_memory convenience method."""

    def test_resolves_memory_and_searches(self) -> None:
        memory = Memory(
            id="src-1",
            content="source content",
            namespace="ns-a",
            project="proj-1",
            tags=["tag1"],
            importance=0.5,
            source=MemorySource.MANUAL,
        )
        vector = np.array([1.0, 0.0, 0.0])

        mem_repo = MagicMock()
        mem_repo.get_with_vector.return_value = (memory, vector)

        results = [_make_result(id="found-1", similarity=0.9, namespace="ns-b")]
        svc = _make_service(
            search_results=results,
            memory_repository=mem_repo,
        )

        matches = svc.find_similar_to_memory("src-1", min_similarity=0.0)

        assert len(matches) == 1
        assert matches[0].memory_id == "found-1"
        mem_repo.get_with_vector.assert_called_once_with("src-1")

    def test_excludes_self_from_results(self) -> None:
        memory = Memory(
            id="src-1",
            content="content",
            namespace="ns-a",
            project="proj-1",
            tags=[],
            importance=0.5,
            source=MemorySource.MANUAL,
        )
        vector = np.array([1.0, 0.0])

        mem_repo = MagicMock()
        mem_repo.get_with_vector.return_value = (memory, vector)

        results = [
            _make_result(id="src-1", similarity=1.0),
            _make_result(id="other", similarity=0.9),
        ]
        svc = _make_service(
            search_results=results,
            memory_repository=mem_repo,
        )

        matches = svc.find_similar_to_memory("src-1", min_similarity=0.0)

        assert all(m.memory_id != "src-1" for m in matches)

    def test_raises_memory_not_found(self) -> None:
        mem_repo = MagicMock()
        mem_repo.get_with_vector.return_value = None

        svc = _make_service(memory_repository=mem_repo)

        with pytest.raises(MemoryNotFoundError):
            svc.find_similar_to_memory("nonexistent")

    def test_raises_without_memory_repository(self) -> None:
        svc = _make_service()

        with pytest.raises(RuntimeError, match="memory_repository"):
            svc.find_similar_to_memory("any-id")

    def test_exclude_same_namespace(self) -> None:
        memory = Memory(
            id="src-1",
            content="content",
            namespace="ns-a",
            project="proj-1",
            tags=[],
            importance=0.5,
            source=MemorySource.MANUAL,
        )
        vector = np.array([1.0, 0.0])

        mem_repo = MagicMock()
        mem_repo.get_with_vector.return_value = (memory, vector)

        results = [
            _make_result(id="same-ns", namespace="ns-a", similarity=0.95),
            _make_result(id="diff-ns", namespace="ns-b", similarity=0.85),
        ]
        svc = _make_service(
            search_results=results,
            memory_repository=mem_repo,
        )

        matches = svc.find_similar_to_memory(
            "src-1",
            min_similarity=0.0,
            exclude_same_namespace=True,
        )

        assert all(m.namespace != "ns-a" for m in matches)


# =============================================================================
# Scoring Strategy Integration Tests
# =============================================================================


@pytest.mark.unit
class TestScoringStrategyIntegration:
    """Test that different scoring strategies affect results."""

    def test_vector_only_uses_raw_similarity(self) -> None:
        results = [_make_result(id="a", similarity=0.85)]
        svc = _make_service(
            search_results=results,
            scoring_strategy=VectorOnlyScoring(),
        )

        matches = svc.find_similar_across_corpus(np.zeros(384), min_similarity=0.0)

        assert matches[0].similarity == 0.85
        assert matches[0].raw_vector_similarity == 0.85

    def test_vector_content_penalizes_low_overlap(self) -> None:
        results = [
            _make_result(id="a", content="alpha beta", similarity=0.8),
        ]
        svc = _make_service(
            search_results=results,
            scoring_strategy=VectorContentScoring(content_weight=0.3),
        )

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            query_content="gamma delta",
        )

        # Disjoint content => jaccard=0 => (0.7)*0.8 + (0.3)*0 = 0.56
        assert matches[0].similarity < 0.8
        assert matches[0].raw_vector_similarity == 0.8

    def test_strategy_can_rerank(self) -> None:
        """Content scoring can reorder results compared to vector-only."""
        results = [
            _make_result(id="a", content="hello world", similarity=0.8),
            _make_result(id="b", content="goodbye world", similarity=0.82),
        ]
        svc = _make_service(
            search_results=results,
            scoring_strategy=VectorContentScoring(content_weight=0.5),
        )

        matches = svc.find_similar_across_corpus(
            np.zeros(384),
            min_similarity=0.0,
            query_content="hello world",
        )

        # "a" has identical content (jaccard=1), "b" has partial overlap
        # a: 0.5*0.8 + 0.5*1.0 = 0.9
        # b: 0.5*0.82 + 0.5*0.5 = 0.66
        assert matches[0].memory_id == "a"

    def test_with_scoring_strategy_returns_new_instance(self) -> None:
        """with_scoring_strategy() returns a new service using the given strategy."""
        results = [_make_result(id="a", content="hello world", similarity=0.8)]
        original = _make_service(
            search_results=results,
            scoring_strategy=VectorOnlyScoring(),
        )

        replaced = original.with_scoring_strategy(VectorContentScoring(content_weight=0.5))

        # Different instance
        assert replaced is not original

        # Original still uses vector-only (score == raw similarity)
        orig_matches = original.find_similar_across_corpus(
            np.zeros(384), min_similarity=0.0, query_content="goodbye"
        )
        assert orig_matches[0].similarity == 0.8

        # Replaced uses vector-content (penalises disjoint content)
        repl_matches = replaced.find_similar_across_corpus(
            np.zeros(384), min_similarity=0.0, query_content="goodbye"
        )
        assert repl_matches[0].similarity < 0.8
