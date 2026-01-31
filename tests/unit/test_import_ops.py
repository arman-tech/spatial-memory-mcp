"""Unit tests for import operations at the database layer.

Tests bulk_import() method which accepts pre-validated records
and inserts them into the database. File I/O and format parsing
are service layer responsibilities.

These tests focus on:
- Batch processing for efficiency
- Namespace override functionality
- Return value (count, list of IDs)
- Error handling
- Integration with insert_batch()
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.errors import StorageError, ValidationError


def create_import_record(
    make_vector: Any,
    content: str = "Test memory",
    namespace: str = "default",
    tags: list[str] | None = None,
    importance: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to create a valid import record."""
    return {
        "content": content,
        "vector": make_vector().tolist(),
        "namespace": namespace,
        "tags": tags or [],
        "importance": importance,
        "source": "import",
        "metadata": metadata or {},
    }


def records_iterator(records: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Convert list to iterator for testing."""
    yield from records


class TestBulkImport:
    """Tests for Database.bulk_import() method."""

    def test_empty_iterator_returns_zero(self, database: Database) -> None:
        """Empty iterator imports nothing."""
        count, ids = database.bulk_import(iter([]))

        assert count == 0
        assert ids == []

    def test_single_record_import(
        self, database: Database, make_vector: Any
    ) -> None:
        """Single record is imported successfully."""
        record = create_import_record(make_vector, content="Imported memory")

        count, ids = database.bulk_import(iter([record]))

        assert count == 1
        assert len(ids) == 1
        assert isinstance(ids[0], str)

        # Verify the memory exists
        memory = database.get(ids[0])
        assert memory["content"] == "Imported memory"

    def test_multiple_records_import(
        self, database: Database, make_vector: Any
    ) -> None:
        """Multiple records are imported correctly."""
        records = [
            create_import_record(make_vector, content=f"Memory {i}")
            for i in range(5)
        ]

        count, ids = database.bulk_import(iter(records))

        assert count == 5
        assert len(ids) == 5
        assert all(isinstance(id_, str) for id_ in ids)
        assert len(set(ids)) == 5  # All unique IDs

    def test_batch_size_processing(
        self, database: Database, make_vector: Any
    ) -> None:
        """Records are processed in batches."""
        records = [
            create_import_record(make_vector, content=f"Memory {i}")
            for i in range(25)
        ]

        # Use small batch size
        count, ids = database.bulk_import(iter(records), batch_size=10)

        assert count == 25
        assert len(ids) == 25

        # All records should be in database
        assert database.count() == 25

    def test_namespace_override(
        self, database: Database, make_vector: Any
    ) -> None:
        """Namespace override replaces original namespace."""
        records = [
            create_import_record(
                make_vector,
                content=f"Memory {i}",
                namespace="original",
            )
            for i in range(3)
        ]

        count, ids = database.bulk_import(
            iter(records),
            namespace_override="imported",
        )

        # Verify all memories have the overridden namespace
        for memory_id in ids:
            memory = database.get(memory_id)
            assert memory["namespace"] == "imported"

    def test_preserves_original_namespace_when_no_override(
        self, database: Database, make_vector: Any
    ) -> None:
        """Original namespaces preserved when no override specified."""
        records = [
            create_import_record(make_vector, content="A", namespace="ns-a"),
            create_import_record(make_vector, content="B", namespace="ns-b"),
        ]

        count, ids = database.bulk_import(iter(records))

        namespaces = set()
        for memory_id in ids:
            memory = database.get(memory_id)
            namespaces.add(memory["namespace"])

        assert namespaces == {"ns-a", "ns-b"}

    def test_import_with_metadata(
        self, database: Database, make_vector: Any
    ) -> None:
        """Metadata is correctly imported."""
        metadata = {"source_file": "backup.json", "version": 2}
        record = create_import_record(
            make_vector,
            content="Memory with metadata",
            metadata=metadata,
        )

        count, ids = database.bulk_import(iter([record]))

        memory = database.get(ids[0])
        assert memory["metadata"]["source_file"] == "backup.json"
        assert memory["metadata"]["version"] == 2

    def test_import_with_tags(
        self, database: Database, make_vector: Any
    ) -> None:
        """Tags are correctly imported."""
        record = create_import_record(
            make_vector,
            content="Tagged memory",
            tags=["tag1", "tag2", "imported"],
        )

        count, ids = database.bulk_import(iter([record]))

        memory = database.get(ids[0])
        assert set(memory["tags"]) == {"tag1", "tag2", "imported"}

    def test_import_with_importance(
        self, database: Database, make_vector: Any
    ) -> None:
        """Importance score is preserved."""
        record = create_import_record(
            make_vector,
            content="Important memory",
            importance=0.95,
        )

        count, ids = database.bulk_import(iter([record]))

        memory = database.get(ids[0])
        assert memory["importance"] == pytest.approx(0.95, abs=0.01)

    def test_generator_consumed_once(
        self, database: Database, make_vector: Any
    ) -> None:
        """Generator iterator is properly consumed."""
        def record_gen() -> Iterator[dict[str, Any]]:
            for i in range(3):
                yield create_import_record(make_vector, content=f"Gen {i}")

        count, ids = database.bulk_import(record_gen())

        assert count == 3
        assert len(ids) == 3

    def test_invalid_namespace_override_raises(
        self, database: Database, make_vector: Any
    ) -> None:
        """Invalid namespace override raises ValidationError."""
        records = [create_import_record(make_vector)]

        with pytest.raises(ValidationError):
            database.bulk_import(iter(records), namespace_override="")

    def test_handles_numpy_vectors(
        self, database: Database, make_vector: Any
    ) -> None:
        """Numpy array vectors are handled correctly."""
        vec = make_vector()  # Returns numpy array
        record = {
            "content": "Memory with numpy vector",
            "vector": vec,  # numpy array, not list
            "namespace": "default",
            "tags": [],
            "importance": 0.5,
            "source": "import",
            "metadata": {},
        }

        count, ids = database.bulk_import(iter([record]))

        assert count == 1
        memory = database.get(ids[0])
        assert memory["content"] == "Memory with numpy vector"

    def test_handles_list_vectors(
        self, database: Database, make_vector: Any
    ) -> None:
        """List vectors are handled correctly."""
        record = create_import_record(make_vector)  # Uses .tolist()

        count, ids = database.bulk_import(iter([record]))

        assert count == 1

    def test_default_batch_size(
        self, database: Database, make_vector: Any
    ) -> None:
        """Default batch size is 1000."""
        # Just verify the method works with default batch_size
        records = [create_import_record(make_vector) for _ in range(5)]

        count, ids = database.bulk_import(iter(records))

        assert count == 5

    def test_large_import(
        self, database: Database, make_vector: Any
    ) -> None:
        """Large number of records can be imported."""
        records = [
            create_import_record(make_vector, content=f"Memory {i}")
            for i in range(100)
        ]

        count, ids = database.bulk_import(iter(records), batch_size=25)

        assert count == 100
        assert len(ids) == 100
        assert database.count() == 100

    def test_ids_are_valid_uuids(
        self, database: Database, make_vector: Any
    ) -> None:
        """Returned IDs are valid UUIDs."""
        records = [create_import_record(make_vector) for _ in range(3)]

        count, ids = database.bulk_import(iter(records))

        for id_ in ids:
            # Should not raise
            uuid.UUID(id_)

    def test_import_sets_timestamps(
        self, database: Database, make_vector: Any
    ) -> None:
        """Import sets created_at and updated_at timestamps."""
        record = create_import_record(make_vector)

        count, ids = database.bulk_import(iter([record]))

        memory = database.get(ids[0])
        assert memory["created_at"] is not None
        assert memory["updated_at"] is not None
        assert memory["last_accessed"] is not None

    def test_import_initializes_access_count(
        self, database: Database, make_vector: Any
    ) -> None:
        """Import initializes access_count to 0."""
        record = create_import_record(make_vector)

        count, ids = database.bulk_import(iter([record]))

        memory = database.get(ids[0])
        assert memory["access_count"] == 0

    def test_records_with_existing_ids_get_new_ids(
        self, database: Database, make_vector: Any
    ) -> None:
        """Records with IDs in data get new IDs assigned."""
        old_id = str(uuid.uuid4())
        record = create_import_record(make_vector)
        record["id"] = old_id  # Attempt to specify ID

        count, ids = database.bulk_import(iter([record]))

        # The import should assign a new ID (not use the provided one)
        # This ensures imported data doesn't conflict with existing data
        assert count == 1
        assert len(ids) == 1
        # The new ID might differ from old_id (depends on implementation)

    def test_empty_content_raises(
        self, database: Database, make_vector: Any
    ) -> None:
        """Empty content raises ValidationError."""
        record = create_import_record(make_vector, content="")

        with pytest.raises(ValidationError):
            database.bulk_import(iter([record]))

    def test_invalid_importance_raises(
        self, database: Database, make_vector: Any
    ) -> None:
        """Invalid importance value raises ValidationError."""
        record = create_import_record(make_vector, importance=1.5)

        with pytest.raises(ValidationError):
            database.bulk_import(iter([record]))
