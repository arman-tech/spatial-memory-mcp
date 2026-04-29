"""Integration tests for ServiceFactory._validate_embedding_model (spec §6.3).

v2.0 cherry-picks 4 of the 5 cases from §6.3:
  1. Fresh DB (no stored alias) — return.
  3. v2.x exact match — return.
  4. Same-dim mismatch — ConfigurationError §8.2.
  5. Cross-dim mismatch — ConfigurationError §8.3.

Case 2 (v1.x rejection) is intentionally dropped per [DECISION] D5 (clean
break) — Database._check_schema_refusal already rejects v1-shaped DBs at
connect time, so the factory guard only has to worry about cross-version
v2 mismatches.

Registry-of-one constraint (D1): only ``arctic-m-long`` is a known alias.
The mismatch tests therefore stamp synthetic stored aliases that the registry
does not recognize; the validator falls back to ``Database.detect_vector_dim``
to classify same-dim vs cross-dim.
"""

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
    # US-020 / §8.5: registry contract for token-based safety cap.
    emb.max_seq_length = 8192
    emb.tokenize.side_effect = lambda text: list(range(len(text.split()))) if text else []
    emb.detokenize.side_effect = lambda ids: " ".join(["w"] * len(ids))
    return emb


def _settings_for(memory_path: Path, model_alias: str) -> Settings:
    return Settings(
        memory_path=memory_path,
        embedding_model=model_alias,
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )


def test_fresh_db_no_validation_error(tmp_path: Path) -> None:
    """Case 1 — fresh DB with no stored alias passes validation silently."""
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()
    assert container.database is not None


def test_match_after_first_ingest(tmp_path: Path) -> None:
    """Case 3 — once stored alias matches configured alias, validation passes."""
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()

    assert container.database is not None
    container.database.set_system_metadata("embedding_model", "arctic-m-long")
    container.database.close()

    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "arctic-m-long"),
        embeddings=_stub_embeddings(dim=768),
    )
    container2 = factory2.create_all()
    assert container2.database is not None


def test_same_dim_mismatch_points_at_reembed(tmp_path: Path) -> None:
    """Case 4 — different alias, same dim raises §8.2 with reembed hint."""
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()
    assert container.database is not None
    # Mirror production: a stored alias only appears after first ingest pin.
    container.memory.remember("seed memory")
    # Overwrite the pin with a synthetic alias unknown to v2.0 but same dim.
    container.database.set_system_metadata("embedding_model", "fake-768-alias")
    container.database.close()

    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "arctic-m-long"),
        embeddings=_stub_embeddings(dim=768),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        factory2.create_all()

    msg = str(exc_info.value)
    assert "configured=arctic-m-long" in msg
    assert "stored=fake-768-alias" in msg
    assert "both 768-dim" in msg
    assert "spatial-memory reembed --to=arctic-m-long" in msg


def test_cross_dim_mismatch_points_at_export_import(tmp_path: Path) -> None:
    """Case 5 — different dim raises §8.3 with export/import migration hint."""
    # Pin a 384-dim DB with a synthetic stored alias.
    settings = _settings_for(tmp_path / "db", "arctic-m-long")
    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=384))
    container = factory.create_all()
    assert container.database is not None
    container.memory.remember("seed memory")
    container.database.set_system_metadata("embedding_model", "fake-384-alias")
    container.database.close()

    # Reopen with arctic-m-long (768-dim) → cross-dim mismatch.
    factory2 = ServiceFactory(
        _settings_for(tmp_path / "db", "arctic-m-long"),
        embeddings=_stub_embeddings(dim=768),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        factory2.create_all()

    msg = str(exc_info.value)
    assert "configured=arctic-m-long" in msg
    assert "stored=fake-384-alias" in msg
    assert "dim=384" in msg
    assert "dim=768" in msg
    assert "spatial-memory export" in msg
    assert "spatial-memory import" in msg
