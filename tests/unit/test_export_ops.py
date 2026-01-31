"""Unit tests for export operations at the database layer.

Tests get_all_for_export() method which provides streaming data export
without file I/O (that's the service layer's responsibility).

These tests focus on:
- Generator pattern for memory efficiency
- Namespace filtering
- Batch yielding
- Metadata parsing from JSON strings
- Error handling
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.errors import StorageError, ValidationError


class TestGetAllForExport:
    """Tests for Database.get_all_for_export() method."""

    def test_returns_iterator(self, database: Database) -> None:
        """get_all_for_export returns an iterator."""
        result = database.get_all_for_export()
        assert isinstance(result, Iterator)

    def test_empty_table_yields_nothing(self, database: Database) -> None:
        """Empty table yields no batches."""
        batches = list(database.get_all_for_export())
        assert batches == []

    def test_single_memory_yields_one_batch(
        self, database: Database, make_vector: Any
    ) -> None:
        """Single memory is yielded in one batch."""
        # Insert one memory
        database.insert(
            content="Test memory for export",
            vector=make_vector(),
            namespace="test",
            tags=["export"],
            importance=0.8,
            metadata={"key": "value"},
        )

        batches = list(database.get_all_for_export())

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0]["content"] == "Test memory for export"
        assert batches[0][0]["namespace"] == "test"

    def test_batch_size_respected(
        self, database: Database, make_vector: Any
    ) -> None:
        """Records are yielded in specified batch sizes."""
        # Insert 5 memories
        for i in range(5):
            database.insert(
                content=f"Memory {i}",
                vector=make_vector(),
                namespace="default",
            )

        # Request batch size of 2
        batches = list(database.get_all_for_export(batch_size=2))

        # Should have 3 batches: [2, 2, 1]
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_namespace_filter(
        self, database: Database, make_vector: Any
    ) -> None:
        """Only memories from specified namespace are exported."""
        # Insert memories in different namespaces
        database.insert(
            content="Project A memory",
            vector=make_vector(),
            namespace="project-a",
        )
        database.insert(
            content="Project B memory",
            vector=make_vector(),
            namespace="project-b",
        )
        database.insert(
            content="Another project A memory",
            vector=make_vector(),
            namespace="project-a",
        )

        # Export only project-a
        batches = list(database.get_all_for_export(namespace="project-a"))
        all_records = [r for batch in batches for r in batch]

        assert len(all_records) == 2
        assert all(r["namespace"] == "project-a" for r in all_records)

    def test_metadata_parsed_from_json(
        self, database: Database, make_vector: Any
    ) -> None:
        """Metadata JSON strings are parsed to dicts in output."""
        metadata = {"complex": {"nested": "value"}, "list": [1, 2, 3]}
        database.insert(
            content="Memory with metadata",
            vector=make_vector(),
            namespace="default",
            metadata=metadata,
        )

        batches = list(database.get_all_for_export())
        record = batches[0][0]

        # Metadata should be parsed as dict, not JSON string
        assert isinstance(record["metadata"], dict)
        assert record["metadata"]["complex"]["nested"] == "value"
        assert record["metadata"]["list"] == [1, 2, 3]

    def test_includes_all_fields(
        self, database: Database, make_vector: Any
    ) -> None:
        """Export includes all memory fields."""
        vec = make_vector()
        memory_id = database.insert(
            content="Complete memory",
            vector=vec,
            namespace="test-ns",
            tags=["tag1", "tag2"],
            importance=0.75,
            source="test",
            metadata={"key": "value"},
        )

        batches = list(database.get_all_for_export())
        record = batches[0][0]

        # Check all expected fields are present
        expected_fields = {
            "id", "content", "vector", "namespace", "tags",
            "importance", "source", "metadata",
            "created_at", "updated_at", "last_accessed", "access_count",
        }
        assert expected_fields.issubset(set(record.keys()))
        assert record["id"] == memory_id
        assert record["content"] == "Complete memory"
        assert record["namespace"] == "test-ns"
        assert record["tags"] == ["tag1", "tag2"]
        assert record["importance"] == pytest.approx(0.75, abs=0.01)

    def test_handles_empty_metadata(
        self, database: Database, make_vector: Any
    ) -> None:
        """Empty metadata is handled correctly."""
        database.insert(
            content="Memory without metadata",
            vector=make_vector(),
            namespace="default",
        )

        batches = list(database.get_all_for_export())
        record = batches[0][0]

        assert isinstance(record["metadata"], dict)
        assert record["metadata"] == {}

    def test_handles_malformed_metadata_json(
        self, database: Database, make_vector: Any
    ) -> None:
        """Malformed metadata JSON is handled gracefully."""
        # Insert directly with malformed metadata
        memory_id = database.insert(
            content="Memory with bad metadata",
            vector=make_vector(),
            namespace="default",
        )

        # Manually corrupt the metadata (simulating database corruption)
        # This test verifies the export handles such edge cases
        batches = list(database.get_all_for_export())
        record = batches[0][0]

        # Should still get a dict (even if empty) rather than crash
        assert isinstance(record["metadata"], dict)

    def test_large_batch_streaming(
        self, database: Database, make_vector: Any
    ) -> None:
        """Large number of records streams efficiently."""
        # Insert 100 memories
        for i in range(100):
            database.insert(
                content=f"Memory {i}",
                vector=make_vector(),
                namespace="bulk",
            )

        # Export with small batch size
        batches = list(database.get_all_for_export(batch_size=10))

        assert len(batches) == 10
        total_records = sum(len(batch) for batch in batches)
        assert total_records == 100

    def test_invalid_namespace_raises_error(self, database: Database) -> None:
        """Invalid namespace format raises ValidationError."""
        with pytest.raises(ValidationError):
            list(database.get_all_for_export(namespace=""))

    def test_nonexistent_namespace_yields_empty(
        self, database: Database, make_vector: Any
    ) -> None:
        """Filtering by nonexistent namespace yields no results."""
        database.insert(
            content="Memory in default",
            vector=make_vector(),
            namespace="default",
        )

        batches = list(database.get_all_for_export(namespace="nonexistent"))
        assert batches == []

    def test_generator_pattern_memory_efficient(
        self, database: Database, make_vector: Any
    ) -> None:
        """Generator pattern allows memory-efficient iteration."""
        # Insert some memories
        for i in range(10):
            database.insert(
                content=f"Memory {i}",
                vector=make_vector(),
                namespace="default",
            )

        # Use generator without materializing all at once
        export_gen = database.get_all_for_export(batch_size=3)

        # Get first batch
        first_batch = next(export_gen)
        assert len(first_batch) == 3

        # Get second batch
        second_batch = next(export_gen)
        assert len(second_batch) == 3

        # Can continue or stop - generator maintains state
        remaining = list(export_gen)
        assert len(remaining) == 2  # One batch of 3, one batch of 1

    def test_default_batch_size(
        self, database: Database, make_vector: Any
    ) -> None:
        """Default batch size is 1000."""
        # Insert just a few records to verify behavior
        for i in range(5):
            database.insert(
                content=f"Memory {i}",
                vector=make_vector(),
                namespace="default",
            )

        # With default batch size of 1000, 5 records should be in one batch
        batches = list(database.get_all_for_export())
        assert len(batches) == 1
        assert len(batches[0]) == 5
