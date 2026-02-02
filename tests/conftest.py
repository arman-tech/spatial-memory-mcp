"""Pytest fixtures for Spatial Memory tests.

ARCHITECTURE NOTE:
- This file provides MOCK fixtures for unit tests (fast, no model loading)
- Real embedding_service is in tests/integration/conftest.py (session-scoped)
- Tests requiring real embeddings should be in tests/integration/

PERFORMANCE:
- Unit tests use mock_embeddings (instant, no 80MB model load)
- Integration tests load model ONCE per session via session_embedding_service
"""

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
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.core.utils import utc_now


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Tests with real database/embeddings")
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")
    config.addinivalue_line("markers", "requires_model: Tests needing embedding model loaded")


# ---------------------------------------------------------------------------
# Basic fixtures (no heavy dependencies)
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


# ---------------------------------------------------------------------------
# Deterministic Mock Vector Helper
# ---------------------------------------------------------------------------


def get_deterministic_vector(content: str, dims: int = 384) -> np.ndarray:
    """Generate a deterministic mock vector based on content hash.

    This allows tests to get consistent vectors without loading the real model.
    Useful for database tests that need valid vectors but don't test embeddings.
    """
    seed = hash(content) % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def mock_vector_generator() -> Any:
    """Factory for generating deterministic mock vectors.

    Usage:
        def test_something(mock_vector_generator):
            vec = mock_vector_generator("some content")  # Always same vector for same content
    """
    return get_deterministic_vector


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
    repo.delete_batch.return_value = (1, ["mock-deleted-id"])
    repo.search.return_value = []
    repo.update_access.return_value = None
    repo.update.return_value = None
    repo.count.return_value = 0
    repo.get_namespaces.return_value = []
    repo.get_all.return_value = []

    # Phase 5 Protocol Extensions: Utility & Export/Import Operations
    repo.delete_by_namespace.return_value = 0
    repo.rename_namespace.return_value = 0
    repo.get_stats.return_value = {
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
    repo.get_namespace_stats.return_value = {
        "namespace": "default",
        "memory_count": 0,
        "oldest_memory": None,
        "newest_memory": None,
        "avg_content_length": None,
    }
    repo.get_all_for_export.return_value = iter([])
    repo.bulk_import.return_value = (0, [])
    repo.hybrid_search.return_value = []

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
