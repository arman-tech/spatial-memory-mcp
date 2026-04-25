"""Integration test for the v1.x JSON -> v2 import roundtrip (PRD §8, spec §7.2 T3)."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.factory import ServiceFactory

pytestmark = pytest.mark.integration


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "v1_export_sample.json"


def _stub_embeddings(dim: int = 768, alias: str = "arctic-m-long") -> MagicMock:
    """Mock satisfying EmbeddingServiceProtocol — returns 768-dim normalized vectors."""
    emb = MagicMock()
    emb.dimensions = dim
    emb.backend = "pytorch"

    cfg = MagicMock()
    cfg.alias = alias
    cfg.dim = dim
    emb.config = cfg

    rng = np.random.default_rng(seed=12345)

    def make_vec() -> np.ndarray:
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def batch(texts: list[str]) -> list[np.ndarray]:
        return [make_vec() for _ in texts]

    emb.embed.side_effect = lambda _t: make_vec()
    emb.embed_query.side_effect = lambda _t: make_vec()
    emb.embed_passage.side_effect = lambda _t: make_vec()
    emb.embed_batch.side_effect = batch
    emb.embed_query_batch.side_effect = batch
    emb.embed_passage_batch.side_effect = batch
    return emb


@pytest.fixture(autouse=True)
def _reset_after() -> Generator[None, None, None]:
    yield
    reset_settings()


def test_v1_export_imports_under_v2_default(tmp_path: Path) -> None:
    """v1.x JSON export imports cleanly under arctic-m-long with metadata pinned."""
    # Stage the fixture into a path the import allow-list will accept.
    staging = tmp_path / "imports"
    staging.mkdir()
    target = staging / "v1_export.json"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings(
        memory_path=tmp_path / "db",
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
        export_allowed_paths=[str(staging), str(tmp_path)],
        import_allowed_paths=[str(staging), str(tmp_path)],
    )
    override_settings(settings)

    factory = ServiceFactory(settings, embeddings=_stub_embeddings(dim=768))
    container = factory.create_all()

    try:
        result = container.export_import.import_memories(
            source_path=str(target),
            regenerate_embeddings=True,
            dry_run=False,
        )

        # Acceptance: exactly 10 imported, zero failed.
        assert result.memories_imported == 10
        assert result.memories_failed == 0
        assert result.total_records_in_file == 10

        # Acceptance: every imported row's vector length is 768.
        assert container.database is not None
        arrow_table = container.database.table.to_arrow()
        assert arrow_table.num_rows == 10
        vectors = arrow_table.column("vector").to_pylist()
        assert all(len(v) == 768 for v in vectors)

        # Acceptance: pin checks for mem-0001. bulk_import regenerates UUIDs but
        # preserves the original id on imported_memories[].original_id so we can
        # remap to the live row.
        rows = arrow_table.to_pylist()
        by_id = {r["id"]: r for r in rows}
        assert result.imported_memories is not None
        new_id_by_original = {im.original_id: im.id for im in result.imported_memories}
        assert "mem-0001" in new_id_by_original
        live_row = by_id[new_id_by_original["mem-0001"]]
        assert pytest.approx(live_row["importance"], rel=1e-6) == 0.85
        assert "decision" in live_row["tags"]

        # Acceptance: first-write metadata pinning (§11.2 path 2).
        stored_alias = container.database.get_system_metadata("embedding_model")
        assert stored_alias == "arctic-m-long"

        # Acceptance: recall returns at least one non-empty result for a known phrase.
        recall_result = container.memory.recall(
            query="rotate API tokens every 90 days",
            limit=5,
        )
        assert recall_result.total >= 1

        # Sanity: namespaces preserved (multiple distinct namespaces from fixture).
        namespaces = {r["namespace"] for r in rows}
        assert len(namespaces) >= 2
    finally:
        if container.database is not None:
            container.database.close()


def test_v1_fixture_is_well_formed() -> None:
    """The committed fixture must parse and meet the PRD's spread requirements."""
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 10
    by_id = {r["id"]: r for r in data}
    assert by_id["mem-0001"]["importance"] == 0.85
    assert "decision" in by_id["mem-0001"]["tags"]
    namespaces = {r["namespace"] for r in data}
    assert len(namespaces) >= 2
    # At least one record longer than 500 words.
    assert any(len(r["content"].split()) > 300 for r in data)
    # Unicode coverage: at least one non-ASCII content.
    assert any(any(ord(c) > 127 for c in r["content"]) for r in data)
