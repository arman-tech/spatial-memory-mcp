"""Integration tests for `spatial-memory doctor` (PRD §7)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.__main__ import run_doctor
from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.factory import ServiceFactory

pytestmark = pytest.mark.integration


def _stub_embeddings(dim: int = 768, alias: str = "arctic-m-long") -> MagicMock:
    """Mock satisfying EmbeddingServiceProtocol — used to seed real DBs."""
    emb = MagicMock()
    emb.dimensions = dim
    emb.backend = "pytorch"

    cfg = MagicMock()
    cfg.alias = alias
    cfg.dim = dim
    emb.config = cfg

    rng = np.random.default_rng(seed=42)

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


@pytest.fixture
def patched_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ServiceFactory and EmbeddingService to use stubs (avoids real model load)."""
    original_init = ServiceFactory.__init__

    def patched_init(self, settings, embeddings=None, repository=None) -> None:  # type: ignore[no-untyped-def]
        original_init(
            self,
            settings,
            embeddings=embeddings or _stub_embeddings(dim=768),
            repository=repository,
        )

    monkeypatch.setattr(ServiceFactory, "__init__", patched_init)

    class _FakeEmbeddingService:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self._stub = _stub_embeddings(dim=768)

        def embed_passage(self, text: str) -> np.ndarray:
            return self._stub.embed_passage(text)

    monkeypatch.setattr("spatial_memory.core.embeddings.EmbeddingService", _FakeEmbeddingService)


def _doctor_args(json_mode: bool = False) -> argparse.Namespace:
    return argparse.Namespace(json=json_mode, verbose=False)


def _settings_for(memory_path: Path, alias: str = "arctic-m-long") -> Settings:
    return Settings(
        memory_path=memory_path,
        embedding_model=alias,
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )


def _seed(settings: Settings, *, count: int, alias: str, dim: int) -> None:
    """Build a real DB and seed it with `count` memories pinned to `alias`."""
    factory_emb = _stub_embeddings(dim=dim, alias=alias)
    factory = ServiceFactory(settings, embeddings=factory_emb)
    container = factory.create_all()
    try:
        assert container.database is not None
        container.database.set_system_metadata("embedding_model", alias)
        for i in range(count):
            container.memory.remember(
                content=f"doctor seed #{i}",
                namespace="default",
                importance=0.5,
            )
    finally:
        if container.database is not None:
            container.database.close()


def test_doctor_fresh_db(
    tmp_path: Path, patched_factory: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fresh DB: exit 0, prints configured model + 0 stored memories."""
    settings = _settings_for(tmp_path / "fresh-db")
    override_settings(settings)

    exit_code = run_doctor(_doctor_args())
    out = capsys.readouterr().out

    assert exit_code == 0, out
    assert "arctic-m-long" in out
    assert "Memories:" in out
    assert "Status: OK" in out


def test_doctor_matched_db(
    tmp_path: Path, patched_factory: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Matched DB: exit 0, match indicator present."""
    settings = _settings_for(tmp_path / "matched-db", alias="arctic-m-long")
    override_settings(settings)
    _seed(settings, count=3, alias="arctic-m-long", dim=768)

    exit_code = run_doctor(_doctor_args())
    out = capsys.readouterr().out

    assert exit_code == 0, out
    assert "arctic-m-long" in out
    assert "Match:            match" in out
    assert "Status: OK" in out


def test_doctor_mismatched_db_surfaces_migration_steps(
    tmp_path: Path, patched_factory: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mismatched DB: exit 1, prints migration steps mirroring §6.3 factory error."""
    db_path = tmp_path / "mismatch-db"
    seed_settings = _settings_for(db_path, alias="arctic-m-long")
    _seed(seed_settings, count=4, alias="arctic-m-long", dim=768)

    # Now reconfigure for a different same-dim model so doctor reports mismatch.
    new_settings = _settings_for(db_path, alias="nomic")
    override_settings(new_settings)

    exit_code = run_doctor(_doctor_args())
    out = capsys.readouterr().out

    assert exit_code == 1, out
    assert "configured=nomic" in out
    assert "stored=arctic-m-long" in out
    assert "spatial-memory reembed --to=nomic" in out
    assert "Status: WARNINGS" in out


def test_doctor_v1_db_returns_2(
    tmp_path: Path, patched_factory: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """v1.x-shaped DB (memories present, no system_metadata): exit 2 with migration recipe."""
    from tests.fixtures.scripts.build_v1_database_sample import build as build_v1_db

    fixture = build_v1_db(tmp_path / "v1-db")
    settings = _settings_for(fixture, alias="arctic-m-long")
    override_settings(settings)

    exit_code = run_doctor(_doctor_args())
    out = capsys.readouterr().out

    assert exit_code == 2, out
    assert "v1.x database" in out
    assert "MIGRATION_V1_TO_V2.md" in out
    assert "spatial-memory import" in out
    assert "Status: ERRORS" in out


def test_doctor_json_mode(
    tmp_path: Path, patched_factory: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json emits a parseable structured object with the same fields."""
    settings = _settings_for(tmp_path / "json-db", alias="arctic-m-long")
    override_settings(settings)
    _seed(settings, count=2, alias="arctic-m-long", dim=768)

    exit_code = run_doctor(_doctor_args(json_mode=True))
    out = capsys.readouterr().out

    assert exit_code == 0, out
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["embedding"]["configured_alias"] == "arctic-m-long"
    assert payload["embedding"]["stored_alias"] == "arctic-m-long"
    assert payload["embedding"]["match"] == "match"
    assert payload["database"]["memories"] == 2
    assert "schema_version" in payload["database"]
    assert "vector_index" in payload["health"]
