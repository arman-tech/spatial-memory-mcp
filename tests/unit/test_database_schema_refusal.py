"""Unit tests for v1-database schema refusal (US-001 / R11 no-row leg).

Per spec D5 (clean-break) and US-001 acceptance: opening a database whose
memories table is populated but whose system_metadata.schema_version row is
absent must raise a clear ``StorageError`` that points at the export/import
migration path. Both flavors of "missing" are covered:

* the system_metadata table itself is absent (legacy v1 layout), and
* the table exists but no schema_version row was ever written.

The version-mismatch leg of R11 (schema_version present but != '2.0.0') is
the responsibility of US-009 and is intentionally NOT exercised here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.errors import StorageError
from spatial_memory.core.utils import utc_now_naive

pytestmark = pytest.mark.unit


def _seed_v1_shaped_memory(db: Database) -> None:
    """Insert one normalized vector so the memories table is non-empty."""
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(db.embedding_dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    now = utc_now_naive()
    db.table.add(
        [
            {
                "id": "v1-survivor",
                "content": "ancient memory",
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
                "content_hash": "deadbeef",
                "expires_at": None,
            }
        ]
    )


def _make_db(tmp_path: Path) -> Database:
    return Database(
        storage_path=tmp_path / "lancedb",
        embedding_dim=384,
        auto_create_indexes=False,
        enable_fts=False,
        filelock_enabled=False,
    )


def _open_fresh(tmp_path: Path) -> Database:
    """Return a brand-new Database object pointed at the same on-disk path."""
    fresh = _make_db(tmp_path)
    fresh.connect()
    return fresh


def test_refuses_to_open_when_memories_present_and_schema_version_missing(
    tmp_path: Path,
) -> None:
    """No-row leg: schema_version was never written → must refuse.

    Fresh ``connect()`` auto-stamps schema_version on empty DBs, so to
    simulate genuine v1 layout we delete the stamp row after seeding.
    """
    db = _make_db(tmp_path)
    db.connect()
    _seed_v1_shaped_memory(db)
    assert db._db is not None
    metadata_table = db._db.open_table(db.SYSTEM_METADATA_TABLE)
    metadata_table.delete("key = 'schema_version'")
    db.close()

    with pytest.raises(StorageError) as excinfo:
        _open_fresh(tmp_path)

    msg = str(excinfo.value)
    assert "v1" in msg.lower()
    assert "export" in msg.lower()
    assert "import" in msg.lower()


def test_refuses_when_system_metadata_table_was_never_created(tmp_path: Path) -> None:
    """Drop-the-table leg: simulates a true v1 layout that never had system_metadata."""
    db = _make_db(tmp_path)
    db.connect()
    _seed_v1_shaped_memory(db)
    if db._db is not None and db.SYSTEM_METADATA_TABLE in db._db.table_names():
        db._db.drop_table(db.SYSTEM_METADATA_TABLE)
    db.close()

    with pytest.raises(StorageError) as excinfo:
        _open_fresh(tmp_path)

    msg = str(excinfo.value)
    assert "schema_version" in msg or "v1" in msg.lower()
    assert "export" in msg.lower()
    assert "import" in msg.lower()


def test_opens_cleanly_when_schema_version_row_present(tmp_path: Path) -> None:
    """v2-shaped DB (schema_version row written) must open without raising."""
    db = _make_db(tmp_path)
    db.connect()
    _seed_v1_shaped_memory(db)
    db.set_system_metadata("schema_version", "2.0.0")
    db.close()

    fresh = _open_fresh(tmp_path)
    assert fresh.get_system_metadata("schema_version") == "2.0.0"
    fresh.close()


def test_opens_cleanly_when_memories_table_empty(tmp_path: Path) -> None:
    """Fresh install (no memories yet) must open without raising even with no schema_version."""
    db = _make_db(tmp_path)
    db.connect()  # connect already runs the refusal check; no rows -> no refusal
    db.close()

    fresh = _open_fresh(tmp_path)
    assert fresh.count_memories() == 0
    fresh.close()
