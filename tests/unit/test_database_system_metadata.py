"""Unit tests for Database system_metadata table helpers (v2.0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.utils import utc_now_naive

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(
        storage_path=tmp_path / "lancedb",
        embedding_dim=384,
        auto_create_indexes=False,
        enable_fts=False,
        filelock_enabled=False,
    )
    database.connect()
    return database


def test_get_returns_none_when_key_absent(db: Database) -> None:
    assert db.get_system_metadata("missing-key") is None


def test_set_then_get_round_trip(db: Database) -> None:
    db.set_system_metadata("embedding_model", "arctic-m-long")
    assert db.get_system_metadata("embedding_model") == "arctic-m-long"


def test_set_is_upsert(db: Database) -> None:
    db.set_system_metadata("embedding_model", "minilm")
    db.set_system_metadata("embedding_model", "arctic-m-long")
    assert db.get_system_metadata("embedding_model") == "arctic-m-long"


def test_ensure_system_metadata_table_is_idempotent(db: Database) -> None:
    db._ensure_system_metadata_table()
    # Second call must not raise.
    db._ensure_system_metadata_table()
    db._ensure_system_metadata_table()


def test_detect_vector_dim_returns_zero_for_empty_db(db: Database) -> None:
    assert db.detect_vector_dim() == 0


def test_count_memories_zero_for_empty_db(db: Database) -> None:
    assert db.count_memories() == 0


def test_detect_vector_dim_returns_768_after_insert(tmp_path: Path) -> None:
    """Per spec §7.1 — after inserting a 768-dim vector, detect_vector_dim returns 768."""
    database = Database(
        storage_path=tmp_path / "lancedb",
        embedding_dim=768,
        auto_create_indexes=False,
        enable_fts=False,
        filelock_enabled=False,
    )
    database.connect()
    vec = np.random.default_rng(0).standard_normal(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    now = utc_now_naive()
    database.table.add(
        [
            {
                "id": "m-001",
                "content": "hello",
                "vector": vec.tolist(),
                "created_at": now,
                "updated_at": now,
                "last_accessed": now,
                "access_count": 0,
                "importance": 0.5,
                "namespace": "default",
                "tags": [],
                "source": "test",
                "metadata": "{}",
                "project": "",
                "content_hash": "abc",
                "expires_at": None,
            }
        ]
    )
    assert database.detect_vector_dim() == 768
    assert database.count_memories() == 1


def test_set_system_metadata_when_table_missing_creates_it(tmp_path: Path) -> None:
    database = Database(
        storage_path=tmp_path / "lancedb",
        embedding_dim=384,
        auto_create_indexes=False,
        enable_fts=False,
        filelock_enabled=False,
    )
    database.connect()
    # Drop the table the connect-time hook created so we can verify auto-creation.
    if database._db is not None and database.SYSTEM_METADATA_TABLE in database._db.table_names():
        database._db.drop_table(database.SYSTEM_METADATA_TABLE)
    assert database.get_system_metadata("k") is None  # missing -> None
    database.set_system_metadata("k", "v")  # auto-creates
    assert database.get_system_metadata("k") == "v"
