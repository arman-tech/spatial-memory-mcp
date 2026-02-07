"""Tests for WP7: Per-Project Decay.

Verifies that:
1. decay() accepts and passes project parameter to repository
2. decay(project="X") only fetches memories for project X
3. decay(project="X", namespace="Y") filters by both
4. decay() with no project decays all memories (existing behavior)
5. Tool schema includes project parameter
6. _handle_decay resolves and passes project to service
7. Decay calculation is unchanged (algorithm stays the same)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.adapters.project_detection import (
    ProjectDetector,
    ProjectIdentity,
)
from spatial_memory.core.errors import ValidationError
from spatial_memory.core.models import Memory, MemorySource
from spatial_memory.services.lifecycle import (
    DecayResult,
    LifecycleConfig,
    LifecycleService,
)
from tests.unit.conftest import make_server_with_mocks as _make_server_with_mocks

# =============================================================================
# Test UUIDs
# =============================================================================

UUID_1 = "11111111-1111-1111-1111-111111111111"
UUID_2 = "22222222-2222-2222-2222-222222222222"
UUID_3 = "33333333-3333-3333-3333-333333333333"


# =============================================================================
# Helpers
# =============================================================================


def make_memory(
    id: str,
    content: str | None = None,
    namespace: str = "default",
    project: str = "",
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed: datetime | None = None,
) -> Memory:
    """Create a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=id,
        content=content or f"Memory content for {id}",
        namespace=namespace,
        project=project,
        importance=importance,
        tags=[],
        source=MemorySource.MANUAL,
        metadata={},
        created_at=now,
        updated_at=now,
        last_accessed=last_accessed or now,
        access_count=access_count,
    )


def make_vector(dims: int = 384, seed: int = 0) -> np.ndarray:
    """Create a random unit vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dims).astype(np.float32)
    return np.asarray(vec / np.linalg.norm(vec), dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository for unit tests."""
    repo = MagicMock()
    repo.get_all.return_value = []
    repo.update_batch.return_value = (0, [])
    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service."""
    embeddings = MagicMock()
    embeddings.dimensions = 384
    embeddings.embed = MagicMock(return_value=make_vector(seed=42))
    return embeddings


@pytest.fixture
def lifecycle_service(
    mock_repository: MagicMock,
    mock_embeddings: MagicMock,
) -> LifecycleService:
    """LifecycleService with mocked dependencies."""
    return LifecycleService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=LifecycleConfig(),
    )


# =============================================================================
# LifecycleService.decay() — project parameter
# =============================================================================


@pytest.mark.unit
class TestDecayProjectFiltering:
    """Tests for project filtering in LifecycleService.decay()."""

    def test_decay_with_project_passes_to_repo(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X') should pass project='X' to repo.get_all()."""
        lifecycle_service.decay(project="github.com/org/repo", dry_run=True)

        mock_repository.get_all.assert_called_once()
        call_kwargs = mock_repository.get_all.call_args.kwargs
        assert call_kwargs["project"] == "github.com/org/repo"

    def test_decay_without_project_passes_none(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() with no project should pass project=None to repo."""
        lifecycle_service.decay(dry_run=True)

        mock_repository.get_all.assert_called_once()
        call_kwargs = mock_repository.get_all.call_args.kwargs
        assert call_kwargs["project"] is None

    def test_decay_with_project_and_namespace(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X', namespace='Y') should filter by both."""
        lifecycle_service.decay(
            project="github.com/org/repo",
            namespace="work",
            dry_run=True,
        )

        mock_repository.get_all.assert_called_once()
        call_kwargs = mock_repository.get_all.call_args.kwargs
        assert call_kwargs["project"] == "github.com/org/repo"
        assert call_kwargs["namespace"] == "work"

    def test_decay_only_affects_project_memories(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X') should only process memories returned by repo."""
        now = datetime.now(timezone.utc)
        # Repo returns only project X's memories (filtering done by repo)
        project_memory = make_memory(
            UUID_1,
            project="github.com/org/repo",
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [
            (project_memory, make_vector(seed=1)),
        ]

        result = lifecycle_service.decay(
            project="github.com/org/repo",
            dry_run=True,
        )

        assert result.memories_analyzed == 1
        assert result.memories_decayed == 1

    def test_decay_empty_project_returns_zero(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X') with no matching memories returns empty result."""
        mock_repository.get_all.return_value = []

        result = lifecycle_service.decay(
            project="github.com/empty/repo",
            dry_run=True,
        )

        assert result.memories_analyzed == 0
        assert result.memories_decayed == 0
        assert result.decayed_memories == []


# =============================================================================
# Decay algorithm unchanged
# =============================================================================


@pytest.mark.unit
class TestDecayAlgorithmUnchanged:
    """Verify decay calculation is the same with or without project."""

    def test_decay_factor_same_with_project(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """Decay factor should be identical whether project is set or not."""
        now = datetime.now(timezone.utc)
        memory = make_memory(
            UUID_1,
            importance=1.0,
            access_count=0,
            last_accessed=now - timedelta(days=30),
        )
        mock_repository.get_all.return_value = [(memory, make_vector(seed=1))]

        # Without project
        result_no_project = lifecycle_service.decay(
            half_life_days=30.0,
            access_weight=0.0,
            dry_run=True,
        )

        # With project (same memory returned by repo)
        mock_repository.get_all.return_value = [(memory, make_vector(seed=1))]
        result_with_project = lifecycle_service.decay(
            project="github.com/org/repo",
            half_life_days=30.0,
            access_weight=0.0,
            dry_run=True,
        )

        assert len(result_no_project.decayed_memories) == 1
        assert len(result_with_project.decayed_memories) == 1
        # Same decay factor (approx due to wall-clock time between calls)
        assert result_no_project.decayed_memories[0].decay_factor == pytest.approx(
            result_with_project.decayed_memories[0].decay_factor, rel=1e-6
        )
        # Same new importance
        assert result_no_project.decayed_memories[0].new_importance == pytest.approx(
            result_with_project.decayed_memories[0].new_importance, rel=1e-6
        )


# =============================================================================
# Tool schema
# =============================================================================


@pytest.mark.unit
class TestDecayToolSchema:
    """Verify the decay tool schema includes the project parameter."""

    def test_decay_schema_has_project(self) -> None:
        """The decay tool schema should include a 'project' property."""
        from spatial_memory.tools.definitions import TOOLS

        decay_tool = next(t for t in TOOLS if t.name == "decay")
        props = decay_tool.inputSchema["properties"]
        assert "project" in props
        assert props["project"]["type"] == "string"

    def test_decay_schema_has_agent_id(self) -> None:
        """The decay tool schema should still include _agent_id."""
        from spatial_memory.tools.definitions import TOOLS

        decay_tool = next(t for t in TOOLS if t.name == "decay")
        props = decay_tool.inputSchema["properties"]
        assert "_agent_id" in props


# =============================================================================
# Server handler
# =============================================================================


@pytest.mark.unit
class TestHandleDecayProject:
    """Verify _handle_decay resolves and passes project."""

    def test_handle_decay_passes_project(self) -> None:
        """_handle_decay should call _resolve_project and pass to service."""
        server, mocks = _make_server_with_mocks()
        lifecycle = mocks["lifecycle"]
        lifecycle.decay.return_value = DecayResult(
            memories_analyzed=0,
            memories_decayed=0,
            avg_decay_factor=1.0,
            decayed_memories=[],
            dry_run=True,
        )

        server._handle_decay(
            {
                "project": "github.com/org/repo",
                "dry_run": True,
            }
        )

        lifecycle.decay.assert_called_once()
        call_kwargs = lifecycle.decay.call_args.kwargs
        assert call_kwargs["project"] == "github.com/org/repo"

    def test_handle_decay_star_project(self) -> None:
        """_handle_decay with project='*' should pass None (cross-project)."""
        server, mocks = _make_server_with_mocks()
        lifecycle = mocks["lifecycle"]
        lifecycle.decay.return_value = DecayResult(
            memories_analyzed=0,
            memories_decayed=0,
            avg_decay_factor=1.0,
            decayed_memories=[],
            dry_run=True,
        )

        server._handle_decay(
            {
                "project": "*",
                "dry_run": True,
            }
        )

        lifecycle.decay.assert_called_once()
        call_kwargs = lifecycle.decay.call_args.kwargs
        assert call_kwargs["project"] is None

    def test_handle_decay_no_project_auto_detects(self) -> None:
        """_handle_decay with no project should auto-detect."""
        detector = MagicMock(spec=ProjectDetector)
        detector.detect.return_value = ProjectIdentity(
            project_id="github.com/auto/detected",
            source="git_remote",
        )
        server, mocks = _make_server_with_mocks(project_detector=detector)
        lifecycle = mocks["lifecycle"]
        lifecycle.decay.return_value = DecayResult(
            memories_analyzed=0,
            memories_decayed=0,
            avg_decay_factor=1.0,
            decayed_memories=[],
            dry_run=True,
        )

        server._handle_decay({"dry_run": True})

        lifecycle.decay.assert_called_once()
        call_kwargs = lifecycle.decay.call_args.kwargs
        assert call_kwargs["project"] == "github.com/auto/detected"

    def test_handle_decay_preserves_other_params(self) -> None:
        """_handle_decay should pass all other parameters correctly."""
        server, mocks = _make_server_with_mocks()
        lifecycle = mocks["lifecycle"]
        lifecycle.decay.return_value = DecayResult(
            memories_analyzed=0,
            memories_decayed=0,
            avg_decay_factor=1.0,
            decayed_memories=[],
            dry_run=False,
        )

        server._handle_decay(
            {
                "project": "github.com/org/repo",
                "namespace": "work",
                "decay_function": "linear",
                "half_life_days": 60.0,
                "min_importance": 0.2,
                "access_weight": 0.5,
                "dry_run": False,
            }
        )

        lifecycle.decay.assert_called_once()
        call_kwargs = lifecycle.decay.call_args.kwargs
        assert call_kwargs["project"] == "github.com/org/repo"
        assert call_kwargs["namespace"] == "work"
        assert call_kwargs["decay_function"] == "linear"
        assert call_kwargs["half_life_days"] == 60.0
        assert call_kwargs["min_importance"] == 0.2
        assert call_kwargs["access_weight"] == 0.5
        assert call_kwargs["dry_run"] is False


# =============================================================================
# Project validation
# =============================================================================


@pytest.mark.unit
class TestDecayProjectValidation:
    """Tests for project validation in decay()."""

    def test_decay_rejects_empty_project(
        self,
        lifecycle_service: LifecycleService,
    ) -> None:
        """decay(project='') should raise ValidationError."""
        with pytest.raises(ValidationError, match="Project cannot be empty"):
            lifecycle_service.decay(project="", dry_run=True)

    def test_decay_rejects_invalid_project(
        self,
        lifecycle_service: LifecycleService,
    ) -> None:
        """decay() with invalid project characters should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid project format"):
            lifecycle_service.decay(project="'; DROP TABLE --", dry_run=True)

    def test_decay_accepts_valid_project_formats(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should accept various valid project ID formats."""
        valid_projects = [
            "github.com/org/repo",
            "my-project",
            "gitlab.com/org/sub/repo",
            "project_v2",
            "192.168.1.1:8080/repo",
        ]
        for proj in valid_projects:
            mock_repository.get_all.return_value = []
            result = lifecycle_service.decay(project=proj, dry_run=True)
            assert result.memories_analyzed == 0

    def test_decay_rejects_overlong_project(
        self,
        lifecycle_service: LifecycleService,
    ) -> None:
        """decay() with project > 255 chars should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid project format"):
            lifecycle_service.decay(project="a" * 256, dry_run=True)


# =============================================================================
# dry_run=False write path with project
# =============================================================================


@pytest.mark.unit
class TestDecayWritePathWithProject:
    """Verify decay(dry_run=False, project=...) calls update_batch."""

    def test_decay_dry_run_false_with_project_calls_update_batch(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X', dry_run=False) should call update_batch."""
        now = datetime.now(timezone.utc)
        memory = make_memory(
            UUID_1,
            project="github.com/org/repo",
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [(memory, make_vector(seed=1))]
        mock_repository.update_batch.return_value = (1, [])

        result = lifecycle_service.decay(
            project="github.com/org/repo",
            dry_run=False,
        )

        assert result.dry_run is False
        assert result.memories_decayed == 1
        mock_repository.update_batch.assert_called_once()

    def test_decay_dry_run_true_with_project_skips_update(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay(project='X', dry_run=True) should NOT call update_batch."""
        now = datetime.now(timezone.utc)
        memory = make_memory(
            UUID_1,
            project="github.com/org/repo",
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [(memory, make_vector(seed=1))]

        result = lifecycle_service.decay(
            project="github.com/org/repo",
            dry_run=True,
        )

        assert result.dry_run is True
        mock_repository.update_batch.assert_not_called()
