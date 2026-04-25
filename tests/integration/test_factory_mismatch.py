"""Integration tests for ServiceFactory._validate_embedding_model (spec §6.3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.config import Settings
from spatial_memory.core.errors import ConfigurationError
from spatial_memory.factory import ServiceFactory

pytestmark = pytest.mark.integration


def _stub_embeddings(dim: int) -> MagicMock:
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


def _settings_for(memory_path: Path, model_alias: str) -> Settings:
    return Settings(
        memory_path=memory_path,
        embedding_model=model_alias,
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )


def test_match_after_first_ingest(tmp_path: Path) -> None:
    """Once stored alias matches configured alias, validation passes silently."""
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()  # fresh DB — no exception

    # Simulate first ingest pinning metadata.
    assert container.database is not None
    container.database.set_system_metadata("embedding_model", "arctic-m-long")
    container.database.close()

    # Second factory build with same alias must not raise.
    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "arctic-m-long"),
        embeddings=_stub_embeddings(dim=768),
    )
    container2 = factory2.create_all()
    assert container2.database is not None


def test_same_dim_mismatch_points_at_reembed(tmp_path: Path) -> None:
    """Configured nomic vs stored arctic-m-long (both 768-dim) -> §8.2 error."""
    # First, build a DB and pin stored alias to 'arctic-m-long'.
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()
    assert container.database is not None
    container.database.set_system_metadata("embedding_model", "arctic-m-long")
    container.database.close()

    # Now configure 'nomic' (also 768-dim) -> same-dim mismatch.
    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "nomic"),
        embeddings=_stub_embeddings(dim=768),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        factory2.create_all()

    msg = str(exc_info.value)
    assert "configured=nomic" in msg
    assert "stored=arctic-m-long" in msg
    assert "both 768-dim" in msg
    assert "spatial-memory reembed --to=nomic" in msg


def test_cross_dim_mismatch_points_at_export_import(tmp_path: Path) -> None:
    """Configured arctic-m-long (768) vs stored minilm (384) -> §8.3 error."""
    # Pin stored alias to 'minilm' in a 384-dim DB.
    settings = _settings_for(tmp_path / "db", "minilm")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=384))
    container = factory.create_all()
    assert container.database is not None
    container.database.set_system_metadata("embedding_model", "minilm")
    container.database.close()

    # Configure 'arctic-m-long' (768-dim) -> cross-dim mismatch.
    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "arctic-m-long"),
        embeddings=_stub_embeddings(dim=768),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        factory2.create_all()

    msg = str(exc_info.value)
    assert "dim=384" in msg
    assert "dim=768" in msg
    assert "spatial-memory export" in msg
    assert "spatial-memory import" in msg
