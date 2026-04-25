"""Integration test: v1.x database is hard-rejected by the v2 factory."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.config import Settings
from spatial_memory.core.errors import ConfigurationError
from spatial_memory.factory import ServiceFactory
from tests.fixtures.scripts.build_v1_database_sample import build as build_v1_db

pytestmark = pytest.mark.integration


def _stub_embeddings(dim: int = 768) -> MagicMock:
    """Return a mock that satisfies EmbeddingServiceProtocol without loading a real model."""
    emb = MagicMock()
    emb.dimensions = dim
    emb.backend = "pytorch"

    def make_vec() -> np.ndarray:
        v = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    emb.embed.return_value = make_vec()
    emb.embed_batch.return_value = [make_vec()]
    emb.embed_query.return_value = make_vec()
    emb.embed_passage.return_value = make_vec()
    emb.embed_query_batch.return_value = [make_vec()]
    emb.embed_passage_batch.return_value = [make_vec()]
    return emb


def test_v1_database_rejected(tmp_path: Path) -> None:
    """A v1.x database (memories present, no system_metadata) raises §8.1."""
    fixture = build_v1_db(tmp_path / "v1_db")

    # Settings configured for arctic-m-long (768-dim) against the 384-dim v1 fixture.
    # Use a stub embedding service so no real model load happens.
    settings = Settings(
        memory_path=fixture,
        embedding_model="arctic-m-long",
        # Disable filelock to keep the test fast/portable.
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))

    with pytest.raises(ConfigurationError) as exc_info:
        factory.create_all()

    msg = str(exc_info.value)
    assert "v1.x database" in msg
    assert "5 memories" in msg
    assert "384-dim" in msg
    assert "768-dim" in msg
    assert "arctic-m-long" in msg
    assert "MIGRATION_V1_TO_V2.md" in msg
    assert "spatial-memory import" in msg

    # Cleanup fixture
    shutil.rmtree(fixture, ignore_errors=True)
