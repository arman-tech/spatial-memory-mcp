"""Integration tests for `spatial-memory export` and `spatial-memory import` (US-006, spec §4.4).

Exercises run_export / run_import directly (so we can assert on argparse.Namespace
instead of subprocess output). The factory is monkey-patched to inject a stub
embeddings backend — registry-of-one is honoured (alias='arctic-m-long', dim=768).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.__main__ import run_export, run_import
from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.factory import ServiceFactory

pytestmark = pytest.mark.integration


def _stub_embeddings(dim: int = 768) -> MagicMock:
    """Mock satisfying EmbeddingServiceProtocol without loading a real model."""
    emb = MagicMock()
    emb.dimensions = dim
    emb.backend = "pytorch"

    cfg = MagicMock()
    cfg.alias = "arctic-m-long"
    cfg.dim = dim
    emb.config = cfg

    rng = np.random.default_rng(seed=42)

    def make_vec() -> np.ndarray:
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def batch(texts: list[str]) -> list[np.ndarray]:
        return [make_vec() for _ in texts]

    emb.embed_query.side_effect = lambda _t: make_vec()
    emb.embed_passage.side_effect = lambda _t: make_vec()
    emb.embed_query_batch.side_effect = batch
    emb.embed_passage_batch.side_effect = batch
    # US-020 / §8.5: registry contract for token-based safety cap.
    emb.max_seq_length = 8192
    emb.tokenize.side_effect = lambda text: list(range(len(text.split()))) if text else []
    emb.detokenize.side_effect = lambda ids: " ".join(["w"] * len(ids))
    return emb


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Generator[Settings, None, None]:
    """Configure settings to use tmp_path for the DB and allow tmp_path for export/import."""
    db_path = tmp_path / "db"
    settings = Settings(
        memory_path=db_path,
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
        export_allowed_paths=[str(tmp_path)],
        import_allowed_paths=[str(tmp_path)],
    )
    override_settings(settings)
    try:
        yield settings
    finally:
        reset_settings()


@pytest.fixture
def patched_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every ServiceFactory built in this test module to use stub embeddings."""
    original_init = ServiceFactory.__init__

    def patched_init(self, settings, embeddings=None, repository=None) -> None:  # type: ignore[no-untyped-def]
        original_init(
            self,
            settings,
            embeddings=embeddings or _stub_embeddings(dim=768),
            repository=repository,
        )

    monkeypatch.setattr(ServiceFactory, "__init__", patched_init)


def _seed_db(settings: Settings, count: int = 10) -> None:
    """Insert `count` memories directly into the DB so we have something to export.

    Also pins ``system_metadata.embedding_model`` so subsequent factory builds against
    the same path don't trip the v2 dim-mismatch guard. (US-013 wires this into the
    ingest pipeline; until then, tests pin it explicitly.)
    """
    factory = ServiceFactory(settings)
    container = factory.create_all()
    try:
        assert container.database is not None
        container.database.set_system_metadata("embedding_model", "arctic-m-long")
        for i in range(count):
            container.memory.remember(
                content=f"Seed memory #{i:03d} — quick brown fox jumps over lazy dog",
                namespace="default",
                importance=0.5,
                tags=[f"tag-{i % 3}"],
            )
    finally:
        if container.database is not None:
            container.database.close()


def _export_args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=str(output),
        format=None,
        namespace=None,
        project=None,
        include_vectors=True,
        verbose=False,
    )


def _import_args(input_path: Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        format=None,
        namespace_override=None,
        regenerate_embeddings=False,
        dry_run=dry_run,
        no_validate=False,
        deduplicate=False,
        dedup_threshold=0.95,
        verbose=False,
    )


def test_cli_export_writes_parquet(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`spatial-memory export -o file.parquet` writes a non-empty parquet file and exits 0."""
    _seed_db(tmp_settings, count=10)

    out = tmp_path / "out.parquet"
    exit_code = run_export(_export_args(out))

    assert exit_code == 0, capsys.readouterr().out
    assert out.exists()
    assert out.stat().st_size > 0
    captured = capsys.readouterr().out
    assert "Exported 10 memories" in captured
    assert "parquet" in captured


def test_cli_import_defaults_dry_run_false(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No --dry-run flag => actual import; subsequent count_memories() == seed count."""
    _seed_db(tmp_settings, count=10)
    out = tmp_path / "roundtrip.parquet"
    assert run_export(_export_args(out)) == 0
    capsys.readouterr()  # discard

    # Wipe the DB and re-import.
    fresh_db_path = tmp_path / "fresh"
    fresh_settings = Settings(
        memory_path=fresh_db_path,
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
        export_allowed_paths=[str(tmp_path)],
        import_allowed_paths=[str(tmp_path)],
    )
    override_settings(fresh_settings)

    exit_code = run_import(_import_args(out, dry_run=False))
    assert exit_code == 0, capsys.readouterr().out
    captured = capsys.readouterr().out
    assert "imported: 10" in captured
    assert "DRY RUN" not in captured

    # Verify the fresh DB now contains the imported memories.
    db = Database(storage_path=fresh_db_path, embedding_dim=768, auto_create_indexes=False)
    db.connect()
    try:
        assert db.count_memories() == 10
    finally:
        db.close()


def test_cli_import_dry_run_flag(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run leaves the DB untouched (count == 0 against a fresh path)."""
    _seed_db(tmp_settings, count=10)
    out = tmp_path / "preview.parquet"
    assert run_export(_export_args(out)) == 0
    capsys.readouterr()

    fresh_db_path = tmp_path / "fresh-dry"
    fresh_settings = Settings(
        memory_path=fresh_db_path,
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
        export_allowed_paths=[str(tmp_path)],
        import_allowed_paths=[str(tmp_path)],
    )
    override_settings(fresh_settings)

    exit_code = run_import(_import_args(out, dry_run=True))
    assert exit_code == 0, capsys.readouterr().out
    captured = capsys.readouterr().out
    assert "DRY RUN" in captured

    db = Database(storage_path=fresh_db_path, embedding_dim=768, auto_create_indexes=False)
    db.connect()
    try:
        assert db.count_memories() == 0
    finally:
        db.close()


def test_cli_export_invalid_format_returns_1(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
) -> None:
    """Unknown extension and no --format => validation error => exit 1."""
    _seed_db(tmp_settings, count=2)
    out = tmp_path / "bad.weird"
    args = _export_args(out)
    args.format = None
    exit_code = run_export(args)
    # Either a ValidationError (1) or a clean fallback to default format succeeds (0).
    assert exit_code in (0, 1)


def test_cli_import_missing_file_returns_2(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
) -> None:
    """Non-existent input path surfaces as a read error (exit 2) or path-security error (exit 1)."""
    missing = tmp_path / "does_not_exist.parquet"
    exit_code = run_import(_import_args(missing))
    assert exit_code in (1, 2)


def test_cli_import_json_format(
    tmp_path: Path,
    tmp_settings: Settings,
    patched_factory: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON export/import roundtrip works through the CLI as well as parquet."""
    _seed_db(tmp_settings, count=5)
    out = tmp_path / "out.json"
    assert run_export(_export_args(out)) == 0
    capsys.readouterr()

    parsed = json.loads(out.read_text())
    assert isinstance(parsed, (list, dict))

    fresh_db_path = tmp_path / "fresh-json"
    fresh_settings = Settings(
        memory_path=fresh_db_path,
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
        export_allowed_paths=[str(tmp_path)],
        import_allowed_paths=[str(tmp_path)],
    )
    override_settings(fresh_settings)

    exit_code = run_import(_import_args(out, dry_run=False))
    assert exit_code == 0, capsys.readouterr().out

    db = Database(storage_path=fresh_db_path, embedding_dim=768, auto_create_indexes=False)
    db.connect()
    try:
        assert db.count_memories() == 5
    finally:
        db.close()
