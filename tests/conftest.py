"""Pytest fixtures for Spatial Memory tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService


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
def sample_memories() -> list[dict]:
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
