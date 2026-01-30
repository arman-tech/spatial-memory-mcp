"""Pytest fixtures for Spatial Memory tests."""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.core.utils import utc_now

# ---------------------------------------------------------------------------
# Basic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage() -> Generator[Path, None, None]:
    """Provide temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_storage: Path) -> Generator[Settings, None, None]:
    """Provide test settings with temp storage."""
    settings = Settings(
        memory_path=temp_storage / "test-memory",
        embedding_model="all-MiniLM-L6-v2",
        log_level="DEBUG",
    )
    override_settings(settings)
    yield settings
    reset_settings()


@pytest.fixture
def database(test_settings: Settings) -> Generator[Database, None, None]:
    """Provide initialized database."""
    db = Database(test_settings.memory_path)
    db.connect()
    yield db
    db.close()


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Provide embedding service."""
    return EmbeddingService("all-MiniLM-L6-v2")


@pytest.fixture
def sample_memories() -> list[dict[str, Any]]:
    """Provide sample memory data."""
    return [
        {
            "content": "React uses a virtual DOM for efficient rendering",
            "tags": ["react", "frontend"],
        },
        {
            "content": "Vue provides reactive data binding",
            "tags": ["vue", "frontend"],
        },
        {
            "content": "PostgreSQL is a powerful relational database",
            "tags": ["database", "backend"],
        },
        {
            "content": "Redis is used for caching and session storage",
            "tags": ["cache", "backend"],
        },
        {
            "content": "Docker containers provide consistent environments",
            "tags": ["devops", "containers"],
        },
    ]


# ---------------------------------------------------------------------------
# Mock fixtures for unit testing (Clean Architecture)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository for unit tests.

    Returns a MagicMock that satisfies MemoryRepositoryProtocol.
    Configure return values in individual tests.
    """
    repo = MagicMock()
    repo.add.return_value = str(uuid.uuid4())
    repo.add_batch.return_value = [str(uuid.uuid4())]
    repo.get.return_value = None
    repo.get_with_vector.return_value = None
    repo.delete.return_value = True
    repo.delete_batch.return_value = 1
    repo.search.return_value = []
    repo.update_access.return_value = None
    repo.update.return_value = None
    repo.count.return_value = 0
    repo.get_namespaces.return_value = []
    repo.get_all.return_value = []
    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service for unit tests.

    Returns a MagicMock that satisfies EmbeddingServiceProtocol.
    """
    emb = MagicMock()
    emb.dimensions = 384
    emb.embed.return_value = np.random.randn(384).astype(np.float32)
    emb.embed_batch.return_value = [np.random.randn(384).astype(np.float32)]
    return emb


@pytest.fixture
def memory_service(mock_repository: MagicMock, mock_embeddings: MagicMock) -> Any:
    """MemoryService with mocked dependencies.

    Note: Returns Any type since MemoryService may not exist yet (TDD Red phase).
    """
    from spatial_memory.services.memory import MemoryService

    return MemoryService(repository=mock_repository, embeddings=mock_embeddings)


# ---------------------------------------------------------------------------
# Factory fixtures for creating test data
# ---------------------------------------------------------------------------


@pytest.fixture
def make_memory() -> Any:
    """Factory fixture for creating Memory objects.

    Usage:
        def test_something(make_memory):
            memory = make_memory(content="Test content")
    """

    def _make_memory(
        id: str | None = None,
        content: str = "Test memory content",
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: MemorySource = MemorySource.MANUAL,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        last_accessed: datetime | None = None,
        access_count: int = 0,
    ) -> Memory:
        now = utc_now()
        return Memory(
            id=id or str(uuid.uuid4()),
            content=content,
            namespace=namespace,
            tags=tags or [],
            importance=importance,
            source=source,
            metadata=metadata or {},
            created_at=created_at or now,
            updated_at=updated_at or now,
            last_accessed=last_accessed or now,
            access_count=access_count,
        )

    return _make_memory


@pytest.fixture
def make_memory_result() -> Any:
    """Factory fixture for creating MemoryResult objects.

    Usage:
        def test_something(make_memory_result):
            result = make_memory_result(content="Test", similarity=0.9)
    """

    def _make_memory_result(
        id: str | None = None,
        content: str = "Test memory content",
        similarity: float = 0.8,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryResult:
        return MemoryResult(
            id=id or str(uuid.uuid4()),
            content=content,
            similarity=similarity,
            namespace=namespace,
            tags=tags or [],
            importance=importance,
            created_at=created_at or utc_now(),
            metadata=metadata or {},
        )

    return _make_memory_result


@pytest.fixture
def make_vector() -> Any:
    """Factory fixture for creating embedding vectors.

    Usage:
        def test_something(make_vector):
            vec = make_vector()  # random 384-dim vector
            vec = make_vector(dims=512)  # custom dimensions
    """

    def _make_vector(dims: int = 384, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        vec = np.random.randn(dims).astype(np.float32)
        # Normalize to unit length (like real embeddings)
        return vec / np.linalg.norm(vec)

    return _make_vector
