"""Unit tests for UtilityService with mocked dependencies.

Tests the utility management operations:
- stats: Get database statistics and health metrics
- namespaces: List namespaces with memory counts
- delete_namespace: Delete all memories in a namespace
- rename_namespace: Rename namespace (move all memories)
- hybrid_recall: Combined vector + FTS search

Uses mocked repositories and embedding services for isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.core.errors import (
    NamespaceNotFoundError,
    ValidationError,
)
from spatial_memory.core.models import (
    DeleteNamespaceResult,
    HybridRecallResult,
    NamespacesResult,
    RenameNamespaceResult,
    StatsResult,
    UtilityConfig,
)


# =============================================================================
# Test UUIDs (valid format)
# =============================================================================

TEST_UUID_1 = "11111111-1111-1111-1111-111111111111"
TEST_UUID_2 = "22222222-2222-2222-2222-222222222222"
TEST_UUID_3 = "33333333-3333-3333-3333-333333333333"


# =============================================================================
# Helper functions
# =============================================================================


def make_vector(dims: int = 384, seed: int | None = None) -> np.ndarray:
    """Create a random unit vector."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    vec = rng.standard_normal(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return np.asarray(vec / norm, dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository for unit tests.

    Returns a MagicMock that satisfies MemoryRepositoryProtocol.
    Configure specific behavior in individual tests.
    """
    repo = MagicMock()

    # Default returns
    repo.get_stats.return_value = {
        "total_memories": 100,
        "namespaces": {"default": 60, "work": 40},
        "storage_bytes": 1024 * 1024,  # 1MB
        "storage_mb": 1.0,
        "has_vector_index": True,
        "has_fts_index": True,
        "num_fragments": 5,
        "needs_compaction": False,
        "table_version": 3,
        "indices": [
            {
                "name": "vector_idx",
                "index_type": "IVF_PQ",
                "column": "vector",
                "num_indexed_rows": 100,
                "status": "ready",
            },
        ],
    }
    repo.get_namespace_stats.return_value = {
        "namespace": "default",
        "memory_count": 60,
        "oldest_memory": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "newest_memory": datetime(2026, 1, 30, tzinfo=timezone.utc),
        "avg_content_length": 150.0,
    }
    repo.get_namespaces.return_value = ["default", "work"]
    repo.delete_by_namespace.return_value = 60
    repo.rename_namespace.return_value = 60
    repo.count.return_value = 100
    repo.hybrid_search.return_value = []

    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service for unit tests."""
    embeddings = MagicMock()
    embeddings.dimensions = 384
    embeddings.embed = MagicMock(return_value=make_vector(seed=42))
    embeddings.embed_batch = MagicMock(
        return_value=[make_vector(seed=i) for i in range(10)]
    )
    return embeddings


@pytest.fixture
def utility_service(
    mock_repository: MagicMock,
    mock_embeddings: MagicMock,
) -> "UtilityService":
    """UtilityService with mocked dependencies."""
    from spatial_memory.services.utility import UtilityService

    return UtilityService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=UtilityConfig(),
    )


# Import here after defining fixtures to allow TDD red phase
try:
    from spatial_memory.services.utility import UtilityService
except ImportError:
    UtilityService = None  # type: ignore


# =============================================================================
# TestStats
# =============================================================================


class TestStats:
    """Tests for UtilityService.stats() - database statistics operation."""

    def test_stats_returns_result(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should return StatsResult with database information."""
        result = utility_service.stats()

        assert isinstance(result, StatsResult)
        assert result.total_memories == 100
        assert result.storage_mb == 1.0
        mock_repository.get_stats.assert_called_once()

    def test_stats_includes_namespace_breakdown(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should include memory counts per namespace."""
        result = utility_service.stats()

        assert "default" in result.memories_by_namespace
        assert "work" in result.memories_by_namespace
        assert result.memories_by_namespace["default"] == 60
        assert result.memories_by_namespace["work"] == 40

    def test_stats_includes_index_info(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should include index information when requested."""
        result = utility_service.stats(include_index_details=True)

        assert result.has_vector_index is True
        assert result.has_fts_index is True
        assert len(result.indices) > 0

    def test_stats_filters_by_namespace(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should filter by namespace when specified."""
        mock_repository.get_stats.return_value = {
            "total_memories": 60,
            "namespaces": {"default": 60},
            "storage_bytes": 512 * 1024,
            "storage_mb": 0.5,
            "has_vector_index": True,
            "has_fts_index": True,
            "num_fragments": 2,
            "needs_compaction": False,
            "table_version": 3,
            "indices": [],
        }

        result = utility_service.stats(namespace="default")

        assert result.total_memories == 60
        mock_repository.get_stats.assert_called_once()
        call_args = mock_repository.get_stats.call_args
        assert call_args.kwargs.get("namespace") == "default" or call_args.args == (
            "default",
        )

    def test_stats_handles_empty_database(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should handle empty database gracefully."""
        mock_repository.get_stats.return_value = {
            "total_memories": 0,
            "namespaces": {},
            "storage_bytes": 0,
            "storage_mb": 0.0,
            "has_vector_index": False,
            "has_fts_index": False,
            "num_fragments": 0,
            "needs_compaction": False,
            "table_version": 1,
            "indices": [],
        }

        result = utility_service.stats()

        assert result.total_memories == 0
        assert result.storage_mb == 0.0
        assert len(result.memories_by_namespace) == 0

    def test_stats_includes_compaction_status(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """stats() should include compaction status."""
        mock_repository.get_stats.return_value["needs_compaction"] = True

        result = utility_service.stats()

        assert result.needs_compaction is True


# =============================================================================
# TestNamespaces
# =============================================================================


class TestNamespaces:
    """Tests for UtilityService.namespaces() - namespace listing operation."""

    def test_namespaces_returns_result(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """namespaces() should return NamespacesResult with namespace list."""
        result = utility_service.namespaces()

        assert isinstance(result, NamespacesResult)
        assert result.total_namespaces == 2
        mock_repository.get_namespaces.assert_called_once()

    def test_namespaces_includes_stats(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """namespaces() should include stats per namespace when requested."""
        result = utility_service.namespaces(include_stats=True)

        assert len(result.namespaces) == 2
        for ns_info in result.namespaces:
            assert ns_info.memory_count >= 0

    def test_namespaces_without_stats(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """namespaces() without stats should just list names."""
        result = utility_service.namespaces(include_stats=False)

        assert len(result.namespaces) == 2
        # Names should still be present
        ns_names = [ns.name for ns in result.namespaces]
        assert "default" in ns_names
        assert "work" in ns_names

    def test_namespaces_calculates_total(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """namespaces() should calculate total memories across all namespaces."""
        result = utility_service.namespaces()

        assert result.total_memories >= 0

    def test_namespaces_handles_empty(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """namespaces() should handle empty database gracefully."""
        mock_repository.get_namespaces.return_value = []

        result = utility_service.namespaces()

        assert result.total_namespaces == 0
        assert len(result.namespaces) == 0


# =============================================================================
# TestDeleteNamespace
# =============================================================================


class TestDeleteNamespace:
    """Tests for UtilityService.delete_namespace() - namespace deletion operation."""

    def test_delete_namespace_dry_run(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() with dry_run=True should preview without deleting."""
        mock_repository.count.return_value = 60

        result = utility_service.delete_namespace(
            namespace="default",
            confirm=False,
            dry_run=True,
        )

        assert isinstance(result, DeleteNamespaceResult)
        assert result.dry_run is True
        assert result.memories_deleted == 60
        mock_repository.delete_by_namespace.assert_not_called()

    def test_delete_namespace_requires_confirmation(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() without confirmation should fail when not dry_run."""
        mock_repository.count.return_value = 60

        with pytest.raises(ValidationError, match="confirm"):
            utility_service.delete_namespace(
                namespace="default",
                confirm=False,
                dry_run=False,
            )

    def test_delete_namespace_with_confirmation(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() with confirm=True should delete memories."""
        mock_repository.count.return_value = 60
        mock_repository.delete_by_namespace.return_value = 60

        result = utility_service.delete_namespace(
            namespace="default",
            confirm=True,
            dry_run=False,
        )

        assert result.success is True
        assert result.memories_deleted == 60
        assert result.dry_run is False
        mock_repository.delete_by_namespace.assert_called_once_with("default")

    def test_delete_namespace_validates_namespace(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() should validate namespace name."""
        with pytest.raises(ValidationError):
            utility_service.delete_namespace(
                namespace="",
                confirm=True,
                dry_run=False,
            )

    def test_delete_namespace_handles_not_found(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() should handle non-existent namespace."""
        mock_repository.count.return_value = 0

        result = utility_service.delete_namespace(
            namespace="nonexistent",
            dry_run=True,
        )

        assert result.memories_deleted == 0

    def test_delete_namespace_returns_count(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """delete_namespace() should return count of deleted memories."""
        mock_repository.count.return_value = 100
        mock_repository.delete_by_namespace.return_value = 100

        result = utility_service.delete_namespace(
            namespace="work",
            confirm=True,
            dry_run=False,
        )

        assert result.memories_deleted == 100


# =============================================================================
# TestRenameNamespace
# =============================================================================


class TestRenameNamespace:
    """Tests for UtilityService.rename_namespace() - namespace rename operation."""

    def test_rename_namespace_returns_result(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should return RenameNamespaceResult."""
        mock_repository.rename_namespace.return_value = 60

        result = utility_service.rename_namespace(
            old_namespace="default",
            new_namespace="archive",
        )

        assert isinstance(result, RenameNamespaceResult)
        assert result.old_namespace == "default"
        assert result.new_namespace == "archive"
        assert result.memories_renamed == 60
        assert result.success is True

    def test_rename_namespace_calls_repository(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should call repository method."""
        mock_repository.rename_namespace.return_value = 60

        utility_service.rename_namespace(
            old_namespace="work",
            new_namespace="projects",
        )

        mock_repository.rename_namespace.assert_called_once_with("work", "projects")

    def test_rename_namespace_validates_old_namespace(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should validate old namespace name."""
        with pytest.raises(ValidationError):
            utility_service.rename_namespace(
                old_namespace="",
                new_namespace="new",
            )

    def test_rename_namespace_validates_new_namespace(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should validate new namespace name."""
        with pytest.raises(ValidationError):
            utility_service.rename_namespace(
                old_namespace="old",
                new_namespace="",
            )

    def test_rename_namespace_handles_not_found(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should handle non-existent namespace."""
        mock_repository.rename_namespace.side_effect = NamespaceNotFoundError(
            "nonexistent"
        )

        with pytest.raises(NamespaceNotFoundError):
            utility_service.rename_namespace(
                old_namespace="nonexistent",
                new_namespace="new",
            )

    def test_rename_namespace_same_name(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """rename_namespace() should reject renaming to same name."""
        with pytest.raises(ValidationError, match="same"):
            utility_service.rename_namespace(
                old_namespace="default",
                new_namespace="default",
            )


# =============================================================================
# TestHybridRecall
# =============================================================================


class TestHybridRecall:
    """Tests for UtilityService.hybrid_recall() - hybrid search operation."""

    def test_hybrid_recall_returns_result(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should return HybridRecallResult."""
        mock_repository.hybrid_search.return_value = []

        result = utility_service.hybrid_recall(
            query="database configuration",
            alpha=0.5,
        )

        assert isinstance(result, HybridRecallResult)
        assert result.query == "database configuration"
        assert result.alpha == 0.5
        assert result.search_type == "hybrid"

    def test_hybrid_recall_generates_embedding(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should generate query embedding."""
        mock_repository.hybrid_search.return_value = []

        utility_service.hybrid_recall(query="test query")

        mock_embeddings.embed.assert_called_once_with("test query")

    def test_hybrid_recall_calls_hybrid_search(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should call repository hybrid_search."""
        mock_repository.hybrid_search.return_value = []

        utility_service.hybrid_recall(
            query="test",
            alpha=0.7,
            limit=10,
            namespace="work",
        )

        mock_repository.hybrid_search.assert_called_once()
        call_kwargs = mock_repository.hybrid_search.call_args.kwargs
        assert call_kwargs.get("alpha") == 0.7
        assert call_kwargs.get("limit") == 10
        assert call_kwargs.get("namespace") == "work"

    def test_hybrid_recall_validates_alpha_range(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """hybrid_recall() should validate alpha is between 0 and 1."""
        with pytest.raises(ValidationError):
            utility_service.hybrid_recall(query="test", alpha=-0.1)

        with pytest.raises(ValidationError):
            utility_service.hybrid_recall(query="test", alpha=1.5)

    def test_hybrid_recall_validates_query(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
    ) -> None:
        """hybrid_recall() should validate query is not empty."""
        with pytest.raises(ValidationError, match="[Qq]uery"):
            utility_service.hybrid_recall(query="")

    def test_hybrid_recall_respects_limit(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should pass limit to repository."""
        mock_repository.hybrid_search.return_value = []

        utility_service.hybrid_recall(query="test", limit=20)

        call_kwargs = mock_repository.hybrid_search.call_args.kwargs
        assert call_kwargs.get("limit") == 20

    def test_hybrid_recall_respects_min_similarity(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should filter by min_similarity."""
        from spatial_memory.core.models import MemoryResult

        # Return some results from repository
        mock_repository.hybrid_search.return_value = [
            MemoryResult(
                id=TEST_UUID_1,
                content="High similarity result",
                similarity=0.9,
                namespace="default",
                tags=[],
                importance=0.5,
                created_at=datetime.now(timezone.utc),
                metadata={},
            ),
            MemoryResult(
                id=TEST_UUID_2,
                content="Low similarity result",
                similarity=0.3,
                namespace="default",
                tags=[],
                importance=0.5,
                created_at=datetime.now(timezone.utc),
                metadata={},
            ),
        ]

        result = utility_service.hybrid_recall(
            query="test",
            min_similarity=0.5,
        )

        # Should only include high similarity result
        assert result.total == 1
        assert len(result.memories) == 1
        assert result.memories[0].similarity >= 0.5

    def test_hybrid_recall_transforms_results(
        self,
        utility_service: "UtilityService",
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """hybrid_recall() should transform results to HybridMemoryMatch."""
        from spatial_memory.core.models import MemoryResult

        mock_repository.hybrid_search.return_value = [
            MemoryResult(
                id=TEST_UUID_1,
                content="Test content",
                similarity=0.85,
                namespace="default",
                tags=["tag1"],
                importance=0.7,
                created_at=datetime.now(timezone.utc),
                metadata={"key": "value"},
            ),
        ]

        result = utility_service.hybrid_recall(query="test")

        assert len(result.memories) == 1
        match = result.memories[0]
        assert match.id == TEST_UUID_1
        assert match.content == "Test content"
        assert match.similarity == 0.85
        assert match.namespace == "default"
        assert "tag1" in match.tags
        assert match.importance == 0.7


# =============================================================================
# TestUtilityServiceInitialization
# =============================================================================


class TestUtilityServiceInitialization:
    """Tests for UtilityService initialization and configuration."""

    def test_utility_service_uses_default_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """UtilityService should use default config when not provided."""
        from spatial_memory.services.utility import UtilityService

        service = UtilityService(
            repository=mock_repository,
            embeddings=mock_embeddings,
        )

        assert service._config is not None
        assert service._config.hybrid_default_alpha == 0.5

    def test_utility_service_uses_custom_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """UtilityService should use provided config."""
        from spatial_memory.services.utility import UtilityService

        custom_config = UtilityConfig(
            hybrid_default_alpha=0.7,
            namespace_batch_size=2000,
        )

        service = UtilityService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            config=custom_config,
        )

        assert service._config.hybrid_default_alpha == 0.7
        assert service._config.namespace_batch_size == 2000

    def test_utility_service_requires_repository(
        self,
        mock_embeddings: MagicMock,
    ) -> None:
        """UtilityService should require a repository."""
        from spatial_memory.services.utility import UtilityService

        with pytest.raises(TypeError):
            UtilityService(embeddings=mock_embeddings)  # type: ignore

    def test_utility_service_requires_embeddings(
        self,
        mock_repository: MagicMock,
    ) -> None:
        """UtilityService should require an embedding service."""
        from spatial_memory.services.utility import UtilityService

        with pytest.raises(TypeError):
            UtilityService(repository=mock_repository)  # type: ignore
