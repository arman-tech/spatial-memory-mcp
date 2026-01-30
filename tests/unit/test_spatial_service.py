"""Unit tests for SpatialService with mocked dependencies.

Tests the spatial navigation and exploration operations:
- Journey: SLERP interpolation between memories
- Wander: Temperature-based random walk
- Regions: HDBSCAN clustering
- Visualize: UMAP projection

Uses mocked repositories and embedding services for isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatial_memory.core.errors import (
    ClusteringError,
    InsufficientMemoriesError,
    MemoryNotFoundError,
    ValidationError,
    VisualizationError,
)
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.services.spatial import SpatialConfig, SpatialService


# =============================================================================
# Test UUIDs (valid format)
# =============================================================================

START_UUID = "11111111-1111-1111-1111-111111111111"
END_UUID = "22222222-2222-2222-2222-222222222222"
NONEXISTENT_UUID = "00000000-0000-0000-0000-000000000000"


# =============================================================================
# Helper functions
# =============================================================================

def make_memory(
    id: str,
    content: str | None = None,
    namespace: str = "default",
    importance: float = 0.5,
) -> Memory:
    """Create a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=id,
        content=content or f"Memory content for {id}",
        namespace=namespace,
        importance=importance,
        tags=[],
        source=MemorySource.MANUAL,
        metadata={},
        created_at=now,
        updated_at=now,
        last_accessed=now,
        access_count=0,
    )


def make_memory_result(
    id: str,
    content: str | None = None,
    similarity: float = 0.8,
    namespace: str = "default",
    importance: float = 0.5,
) -> MemoryResult:
    """Create a MemoryResult object for testing."""
    return MemoryResult(
        id=id,
        content=content or f"Memory content for {id}",
        similarity=similarity,
        namespace=namespace,
        tags=[],
        importance=importance,
        created_at=datetime.now(timezone.utc),
        metadata={},
    )


def make_vector(dims: int = 384, seed: int | None = None) -> np.ndarray:
    """Create a random unit vector."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    vec = rng.standard_normal(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository with spatial methods.

    Returns a MagicMock that satisfies MemoryRepositoryProtocol with
    spatial-specific methods configured.
    """
    repo = MagicMock()

    # Predefined vectors for start and end memories
    start_vector = np.array([1.0] + [0.0] * 383, dtype=np.float32)
    end_vector = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)

    # Setup get_with_vector to return (Memory, vector) tuples
    def get_with_vector(id: str) -> tuple[Memory, np.ndarray] | None:
        if id == NONEXISTENT_UUID:
            return None
        if id == START_UUID:
            return (make_memory(id), start_vector)
        if id == END_UUID:
            return (make_memory(id), end_vector)
        # Return random vector for other IDs
        return (make_memory(id), make_vector(seed=hash(id) % (2**32)))

    repo.get_with_vector = MagicMock(side_effect=get_with_vector)

    # Setup get to return Memory objects
    def get_memory(id: str) -> Memory | None:
        if id == NONEXISTENT_UUID:
            return None
        return make_memory(id)

    repo.get = MagicMock(side_effect=get_memory)

    # Setup search to return MemoryResult list
    def search(
        vector: np.ndarray,
        limit: int = 10,
        namespace: str | None = None,
    ) -> list[MemoryResult]:
        results = []
        for i in range(limit):
            results.append(make_memory_result(
                id=f"search-result-{i}-{hash(tuple(vector[:5].tolist())) % 10000:04d}",
                similarity=0.95 - (i * 0.05),
            ))
        return results

    repo.search = MagicMock(side_effect=search)

    # Setup get_all to return list of (Memory, vector) tuples for regions/visualize
    def get_all(
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[tuple[Memory, np.ndarray]]:
        rng = np.random.default_rng(42)
        results = []
        for i in range(min(limit, 100)):
            mem = make_memory(f"mem-{i:04d}")
            vec = rng.standard_normal(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            results.append((mem, vec))
        return results

    repo.get_all = MagicMock(side_effect=get_all)

    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service for unit tests."""
    embeddings = MagicMock()
    embeddings.dimensions = 384
    embeddings.embed = MagicMock(
        return_value=np.random.randn(384).astype(np.float32)
    )
    embeddings.embed_batch = MagicMock(
        return_value=[np.random.randn(384).astype(np.float32)]
    )
    return embeddings


@pytest.fixture
def spatial_service(mock_repository: MagicMock, mock_embeddings: MagicMock) -> SpatialService:
    """SpatialService with mocked dependencies."""
    return SpatialService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=SpatialConfig(),
    )


# =============================================================================
# TestJourney
# =============================================================================


class TestJourney:
    """Tests for SpatialService.journey() - SLERP interpolation."""

    def test_journey_returns_result(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """journey() should return JourneyResult with path steps."""
        result = spatial_service.journey(
            start_id=START_UUID,
            end_id=END_UUID,
            steps=5,
        )

        assert result.start_id == START_UUID
        assert result.end_id == END_UUID
        assert len(result.steps) == 5

    def test_journey_validates_min_steps(self, spatial_service: SpatialService) -> None:
        """journey() should raise ValidationError for steps < 2."""
        with pytest.raises(ValidationError, match="at least 2 steps"):
            spatial_service.journey(
                start_id=START_UUID,
                end_id=END_UUID,
                steps=1,
            )

    def test_journey_validates_max_steps(self, spatial_service: SpatialService) -> None:
        """journey() should raise ValidationError for steps exceeding max."""
        with pytest.raises(ValidationError, match="[Mm]aximum"):
            spatial_service.journey(
                start_id=START_UUID,
                end_id=END_UUID,
                steps=100,
            )

    def test_journey_validates_uuid(self, spatial_service: SpatialService) -> None:
        """journey() should raise ValidationError for invalid UUID."""
        with pytest.raises(ValidationError, match="UUID"):
            spatial_service.journey(
                start_id="invalid-not-a-uuid",
                end_id=END_UUID,
            )

    def test_journey_raises_not_found(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """journey() should raise MemoryNotFoundError for unknown memory."""
        with pytest.raises(MemoryNotFoundError):
            spatial_service.journey(
                start_id=NONEXISTENT_UUID,
                end_id=END_UUID,
            )

    def test_journey_with_namespace_filter(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """journey() should pass namespace to search operations."""
        spatial_service.journey(
            start_id=START_UUID,
            end_id=END_UUID,
            steps=3,
            namespace="work",
        )

        # Verify namespace was passed to search
        calls = mock_repository.search.call_args_list
        for call in calls:
            assert call.kwargs.get("namespace") == "work"


# =============================================================================
# TestWander
# =============================================================================


class TestWander:
    """Tests for SpatialService.wander() - random walk exploration."""

    def test_wander_returns_result(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """wander() should return WanderResult with steps."""
        result = spatial_service.wander(
            start_id=START_UUID,
            steps=3,
            temperature=0.5,
        )

        assert result.start_id == START_UUID
        assert len(result.steps) <= 3
        assert result.total_distance >= 0.0

    def test_wander_validates_temperature_too_low(
        self, spatial_service: SpatialService
    ) -> None:
        """wander() should raise ValidationError for temperature < 0.0."""
        with pytest.raises(ValidationError, match="[Tt]emperature"):
            spatial_service.wander(start_id=START_UUID, temperature=-0.1)

    def test_wander_validates_temperature_too_high(
        self, spatial_service: SpatialService
    ) -> None:
        """wander() should raise ValidationError for temperature > 1.0."""
        with pytest.raises(ValidationError, match="[Tt]emperature"):
            spatial_service.wander(start_id=START_UUID, temperature=1.5)

    def test_wander_validates_steps_at_least_one(
        self, spatial_service: SpatialService
    ) -> None:
        """wander() should raise ValidationError for steps < 1."""
        with pytest.raises(ValidationError, match="at least 1 step"):
            spatial_service.wander(start_id=START_UUID, steps=0)

    def test_wander_validates_steps_max(self, spatial_service: SpatialService) -> None:
        """wander() should raise ValidationError for steps exceeding max."""
        with pytest.raises(ValidationError, match="[Mm]aximum"):
            spatial_service.wander(start_id=START_UUID, steps=100)

    def test_wander_with_namespace(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """wander() should pass namespace to search."""
        spatial_service.wander(
            start_id=START_UUID,
            steps=2,
            namespace="personal",
        )

        call_args = mock_repository.search.call_args
        assert call_args.kwargs.get("namespace") == "personal"


# =============================================================================
# TestRegions
# =============================================================================


class TestRegions:
    """Tests for SpatialService.regions() - HDBSCAN clustering."""

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.hdbscan")
    def test_regions_returns_clusters(
        self,
        mock_hdbscan: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """regions() should return RegionsResult with clusters."""
        # Mock HDBSCAN clusterer
        clusterer = MagicMock()
        # Labels: 0, 0, 1, 1, -1 repeated 20 times for 100 memories
        labels = np.array([0, 0, 1, 1, -1] * 20)
        clusterer.fit_predict.return_value = labels
        mock_hdbscan.HDBSCAN.return_value = clusterer

        result = spatial_service.regions()

        assert result.total_memories == 100
        assert len(result.clusters) > 0
        assert result.noise_count >= 0

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    def test_regions_raises_when_hdbscan_unavailable(
        self, spatial_service: SpatialService
    ) -> None:
        """regions() should raise ClusteringError when HDBSCAN not installed."""
        with pytest.raises(ClusteringError, match="HDBSCAN.*not available"):
            spatial_service.regions()

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", True)
    def test_regions_raises_insufficient_memories(
        self, spatial_service: SpatialService, mock_repository: MagicMock
    ) -> None:
        """regions() should raise InsufficientMemoriesError for too few memories."""
        # Return only 1 memory (less than min_cluster_size default of 2)
        mock_repository.get_all.side_effect = None  # Clear any existing side_effect
        mock_repository.get_all.return_value = [
            (make_memory("mem-1"), make_vector(seed=1))
        ]

        with pytest.raises(InsufficientMemoriesError):
            spatial_service.regions()

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.hdbscan")
    def test_regions_with_namespace(
        self,
        mock_hdbscan: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """regions() should pass namespace filter to repository."""
        clusterer = MagicMock()
        clusterer.fit_predict.return_value = np.array([0] * 100)
        mock_hdbscan.HDBSCAN.return_value = clusterer

        spatial_service.regions(namespace="projects")

        call_args = mock_repository.get_all.call_args
        assert call_args.kwargs.get("namespace") == "projects"

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.hdbscan")
    def test_regions_max_clusters_limit(
        self,
        mock_hdbscan: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """regions() should respect max_clusters parameter."""
        clusterer = MagicMock()
        # Create 5 clusters
        clusterer.fit_predict.return_value = np.array(
            [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20
        )
        mock_hdbscan.HDBSCAN.return_value = clusterer

        result = spatial_service.regions(max_clusters=2)

        assert len(result.clusters) <= 2


# =============================================================================
# TestVisualize
# =============================================================================


class TestVisualize:
    """Tests for SpatialService.visualize() - UMAP projection."""

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_returns_result(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() should return VisualizeResult."""
        # Mock UMAP reducer
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize()

        assert len(result.nodes) == 100
        assert result.format == "json"

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", False)
    def test_visualize_raises_when_umap_unavailable(
        self, spatial_service: SpatialService
    ) -> None:
        """visualize() should raise VisualizationError when UMAP not installed."""
        with pytest.raises(VisualizationError, match="UMAP.*not available"):
            spatial_service.visualize()

    def test_visualize_validates_dimensions(
        self, spatial_service: SpatialService
    ) -> None:
        """visualize() should raise ValidationError for invalid dimensions."""
        with pytest.raises(ValidationError, match="must be 2 or 3"):
            spatial_service.visualize(dimensions=4)  # type: ignore

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_json_format(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() with format=json should return valid JSON output."""
        import json

        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize(format="json")

        # Output should be valid JSON
        parsed = json.loads(result.output)
        assert "nodes" in parsed
        assert "edges" in parsed

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_mermaid_format(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() with format=mermaid should return Mermaid diagram."""
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize(format="mermaid")

        assert result.format == "mermaid"
        assert result.output.startswith("graph LR")

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_svg_format(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() with format=svg should return SVG markup."""
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize(format="svg")

        assert result.format == "svg"
        assert "<svg" in result.output
        assert "</svg>" in result.output

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_includes_edges(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() with include_edges=True should compute similarity edges."""
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize(include_edges=True)

        # Edges list exists (may be empty if no pairs exceed threshold)
        assert hasattr(result, "edges")
        assert isinstance(result.edges, list)

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_excludes_edges(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() with include_edges=False should not compute edges."""
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize(include_edges=False)

        assert len(result.edges) == 0

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_raises_for_too_few_memories(
        self,
        mock_umap: MagicMock,
        spatial_service: SpatialService,
        mock_repository: MagicMock,
    ) -> None:
        """visualize() should raise error if fewer than 5 memories."""
        # Return only 2 memories (less than required 5)
        mock_repository.get_all.side_effect = None  # Clear any existing side_effect
        mock_repository.get_all.return_value = [
            (make_memory("mem-1"), make_vector(seed=1)),
            (make_memory("mem-2"), make_vector(seed=2)),
        ]

        with pytest.raises(InsufficientMemoriesError):
            spatial_service.visualize()


# =============================================================================
# TestSpatialServiceInitialization
# =============================================================================


class TestSpatialServiceInitialization:
    """Tests for SpatialService initialization and configuration."""

    def test_spatial_service_uses_default_config(
        self, mock_repository: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """SpatialService should use default config when not provided."""
        service = SpatialService(
            repository=mock_repository,
            embeddings=mock_embeddings,
        )

        assert service._config is not None
        assert service._config.journey_default_steps == 10

    def test_spatial_service_uses_custom_config(
        self, mock_repository: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """SpatialService should use provided config."""
        custom_config = SpatialConfig(
            journey_default_steps=5,
            wander_default_temperature=0.8,
        )

        service = SpatialService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            config=custom_config,
        )

        assert service._config.journey_default_steps == 5
        assert service._config.wander_default_temperature == 0.8

    def test_spatial_service_requires_repository(
        self, mock_embeddings: MagicMock
    ) -> None:
        """SpatialService should require a repository."""
        with pytest.raises(TypeError):
            SpatialService(embeddings=mock_embeddings)  # type: ignore

    def test_spatial_service_requires_embeddings(
        self, mock_repository: MagicMock
    ) -> None:
        """SpatialService should require an embedding service."""
        with pytest.raises(TypeError):
            SpatialService(repository=mock_repository)  # type: ignore
