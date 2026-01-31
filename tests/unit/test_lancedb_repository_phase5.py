"""Unit tests for Phase 5.2 LanceDBMemoryRepository adapter methods.

These tests verify the Phase 5 protocol extensions in LanceDBMemoryRepository:
- delete_by_namespace: Delete all memories in a namespace
- rename_namespace: Rename all memories from one namespace to another
- get_stats: Get comprehensive database statistics
- get_namespace_stats: Get statistics for a specific namespace
- get_all_for_export: Stream memories for export in batches
- bulk_import: Import memories from an iterator of records

Tests use mocked Database instances to isolate the adapter layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.errors import (
    NamespaceNotFoundError,
    StorageError,
    ValidationError,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_database() -> MagicMock:
    """Create a mock Database instance for testing."""
    db = MagicMock()
    return db


@pytest.fixture
def repository(mock_database: MagicMock) -> LanceDBMemoryRepository:
    """Create a LanceDBMemoryRepository with mocked database."""
    return LanceDBMemoryRepository(mock_database)


# ===========================================================================
# delete_by_namespace Tests
# ===========================================================================


class TestDeleteByNamespace:
    """Tests for LanceDBMemoryRepository.delete_by_namespace()."""

    def test_delete_by_namespace_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should delegate to database."""
        mock_database.delete_by_namespace.return_value = 5

        result = repository.delete_by_namespace("test-namespace")

        assert result == 5
        mock_database.delete_by_namespace.assert_called_once_with("test-namespace")

    def test_delete_by_namespace_returns_zero_for_empty_namespace(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should return 0 when namespace has no memories."""
        mock_database.delete_by_namespace.return_value = 0

        result = repository.delete_by_namespace("empty-namespace")

        assert result == 0
        mock_database.delete_by_namespace.assert_called_once_with("empty-namespace")

    def test_delete_by_namespace_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should propagate ValidationError."""
        mock_database.delete_by_namespace.side_effect = ValidationError(
            "Invalid namespace"
        )

        with pytest.raises(ValidationError, match="Invalid namespace"):
            repository.delete_by_namespace("invalid/namespace")

    def test_delete_by_namespace_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should propagate StorageError."""
        mock_database.delete_by_namespace.side_effect = StorageError("Database error")

        with pytest.raises(StorageError, match="Database error"):
            repository.delete_by_namespace("test-namespace")

    def test_delete_by_namespace_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should wrap unexpected errors in StorageError."""
        mock_database.delete_by_namespace.side_effect = RuntimeError("Unexpected")

        with pytest.raises(StorageError, match="Failed to delete namespace"):
            repository.delete_by_namespace("test-namespace")


# ===========================================================================
# rename_namespace Tests
# ===========================================================================


class TestRenameNamespace:
    """Tests for LanceDBMemoryRepository.rename_namespace()."""

    def test_rename_namespace_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should delegate to database."""
        mock_database.rename_namespace.return_value = 10

        result = repository.rename_namespace("old-ns", "new-ns")

        assert result == 10
        mock_database.rename_namespace.assert_called_once_with("old-ns", "new-ns")

    def test_rename_namespace_returns_zero_for_empty_namespace(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should return 0 for empty source namespace."""
        mock_database.rename_namespace.return_value = 0

        result = repository.rename_namespace("empty", "new")

        assert result == 0

    def test_rename_namespace_raises_namespace_not_found(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should propagate NamespaceNotFoundError."""
        mock_database.rename_namespace.side_effect = NamespaceNotFoundError(
            "nonexistent"
        )

        with pytest.raises(NamespaceNotFoundError):
            repository.rename_namespace("nonexistent", "new-ns")

    def test_rename_namespace_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should propagate ValidationError."""
        mock_database.rename_namespace.side_effect = ValidationError("Invalid name")

        with pytest.raises(ValidationError, match="Invalid name"):
            repository.rename_namespace("old", "invalid/name")

    def test_rename_namespace_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should propagate StorageError."""
        mock_database.rename_namespace.side_effect = StorageError("Database error")

        with pytest.raises(StorageError, match="Database error"):
            repository.rename_namespace("old", "new")

    def test_rename_namespace_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should wrap unexpected errors in StorageError."""
        mock_database.rename_namespace.side_effect = RuntimeError("Unexpected")

        with pytest.raises(StorageError, match="Failed to rename namespace"):
            repository.rename_namespace("old", "new")


# ===========================================================================
# get_stats Tests
# ===========================================================================


class TestGetStats:
    """Tests for LanceDBMemoryRepository.get_stats()."""

    def test_get_stats_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should delegate to database."""
        expected_stats = {
            "total_memories": 100,
            "memories_by_namespace": {"default": 50, "work": 50},
            "storage_bytes": 1024000,
            "storage_mb": 0.98,
            "has_vector_index": True,
            "has_fts_index": True,
        }
        mock_database.get_stats.return_value = expected_stats

        result = repository.get_stats()

        assert result == expected_stats
        mock_database.get_stats.assert_called_once_with(None)

    def test_get_stats_with_namespace_filter(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should pass namespace filter to database."""
        mock_database.get_stats.return_value = {"total_memories": 50}

        repository.get_stats(namespace="work")

        mock_database.get_stats.assert_called_once_with("work")

    def test_get_stats_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should propagate ValidationError."""
        mock_database.get_stats.side_effect = ValidationError("Invalid namespace")

        with pytest.raises(ValidationError, match="Invalid namespace"):
            repository.get_stats(namespace="invalid/ns")

    def test_get_stats_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should propagate StorageError."""
        mock_database.get_stats.side_effect = StorageError("Database error")

        with pytest.raises(StorageError, match="Database error"):
            repository.get_stats()

    def test_get_stats_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should wrap unexpected errors in StorageError."""
        mock_database.get_stats.side_effect = RuntimeError("Unexpected")

        with pytest.raises(StorageError, match="Failed to get stats"):
            repository.get_stats()


# ===========================================================================
# get_namespace_stats Tests
# ===========================================================================


class TestGetNamespaceStats:
    """Tests for LanceDBMemoryRepository.get_namespace_stats()."""

    def test_get_namespace_stats_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should delegate to database."""
        expected_stats = {
            "namespace": "work",
            "memory_count": 25,
            "oldest_memory": "2024-01-01T00:00:00Z",
            "newest_memory": "2024-06-01T00:00:00Z",
        }
        mock_database.get_namespace_stats.return_value = expected_stats

        result = repository.get_namespace_stats("work")

        assert result == expected_stats
        mock_database.get_namespace_stats.assert_called_once_with("work")

    def test_get_namespace_stats_raises_namespace_not_found(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should propagate NamespaceNotFoundError."""
        mock_database.get_namespace_stats.side_effect = NamespaceNotFoundError(
            "nonexistent"
        )

        with pytest.raises(NamespaceNotFoundError):
            repository.get_namespace_stats("nonexistent")

    def test_get_namespace_stats_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should propagate ValidationError."""
        mock_database.get_namespace_stats.side_effect = ValidationError(
            "Invalid namespace"
        )

        with pytest.raises(ValidationError, match="Invalid namespace"):
            repository.get_namespace_stats("invalid/ns")

    def test_get_namespace_stats_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should propagate StorageError."""
        mock_database.get_namespace_stats.side_effect = StorageError("Database error")

        with pytest.raises(StorageError, match="Database error"):
            repository.get_namespace_stats("work")

    def test_get_namespace_stats_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should wrap unexpected errors in StorageError."""
        mock_database.get_namespace_stats.side_effect = RuntimeError("Unexpected")

        with pytest.raises(StorageError, match="Failed to get namespace stats"):
            repository.get_namespace_stats("work")


# ===========================================================================
# get_all_for_export Tests
# ===========================================================================


class TestGetAllForExport:
    """Tests for LanceDBMemoryRepository.get_all_for_export()."""

    def test_get_all_for_export_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should delegate to database."""
        batch1 = [{"id": "1", "content": "test1"}]
        batch2 = [{"id": "2", "content": "test2"}]

        def mock_generator():
            yield batch1
            yield batch2

        mock_database.get_all_for_export.return_value = mock_generator()

        result = list(repository.get_all_for_export())

        assert len(result) == 2
        assert result[0] == batch1
        assert result[1] == batch2
        mock_database.get_all_for_export.assert_called_once_with(None, 1000)

    def test_get_all_for_export_with_namespace_filter(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should pass namespace filter to database."""
        mock_database.get_all_for_export.return_value = iter([])

        list(repository.get_all_for_export(namespace="work"))

        mock_database.get_all_for_export.assert_called_once_with("work", 1000)

    def test_get_all_for_export_with_custom_batch_size(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should pass batch_size to database."""
        mock_database.get_all_for_export.return_value = iter([])

        list(repository.get_all_for_export(batch_size=500))

        mock_database.get_all_for_export.assert_called_once_with(None, 500)

    def test_get_all_for_export_yields_batches(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should yield batches from database."""
        batches = [
            [{"id": str(i), "content": f"content{i}"} for i in range(3)],
            [{"id": str(i + 3), "content": f"content{i + 3}"} for i in range(3)],
        ]

        def mock_generator():
            yield from batches

        mock_database.get_all_for_export.return_value = mock_generator()

        result = list(repository.get_all_for_export())

        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    def test_get_all_for_export_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should propagate ValidationError."""

        def mock_generator():
            raise ValidationError("Invalid namespace")
            yield  # noqa: B901 - yield is unreachable but needed for generator

        mock_database.get_all_for_export.return_value = mock_generator()

        with pytest.raises(ValidationError, match="Invalid namespace"):
            list(repository.get_all_for_export(namespace="invalid/ns"))

    def test_get_all_for_export_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should propagate StorageError."""

        def mock_generator():
            raise StorageError("Database error")
            yield  # noqa: B901 - yield is unreachable but needed for generator

        mock_database.get_all_for_export.return_value = mock_generator()

        with pytest.raises(StorageError, match="Database error"):
            list(repository.get_all_for_export())

    def test_get_all_for_export_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should wrap unexpected errors in StorageError."""

        def mock_generator():
            raise RuntimeError("Unexpected")
            yield  # noqa: B901 - yield is unreachable but needed for generator

        mock_database.get_all_for_export.return_value = mock_generator()

        with pytest.raises(StorageError, match="Failed to export"):
            list(repository.get_all_for_export())

    def test_get_all_for_export_returns_empty_for_empty_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should return empty iterator for empty database."""
        mock_database.get_all_for_export.return_value = iter([])

        result = list(repository.get_all_for_export())

        assert result == []


# ===========================================================================
# bulk_import Tests
# ===========================================================================


class TestBulkImport:
    """Tests for LanceDBMemoryRepository.bulk_import()."""

    def test_bulk_import_delegates_to_database(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should delegate to database."""
        records = [
            {"content": "test1", "vector": [0.1] * 384},
            {"content": "test2", "vector": [0.2] * 384},
        ]
        mock_database.bulk_import.return_value = (2, ["id1", "id2"])

        result = repository.bulk_import(iter(records))

        assert result == (2, ["id1", "id2"])
        mock_database.bulk_import.assert_called_once()

    def test_bulk_import_with_custom_batch_size(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should pass batch_size to database."""
        mock_database.bulk_import.return_value = (0, [])

        repository.bulk_import(iter([]), batch_size=500)

        call_args = mock_database.bulk_import.call_args
        assert call_args[0][1] == 500  # batch_size argument

    def test_bulk_import_with_namespace_override(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should pass namespace_override to database."""
        mock_database.bulk_import.return_value = (0, [])

        repository.bulk_import(iter([]), namespace_override="imported")

        call_args = mock_database.bulk_import.call_args
        assert call_args[0][2] == "imported"  # namespace_override argument

    def test_bulk_import_raises_validation_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should propagate ValidationError."""
        mock_database.bulk_import.side_effect = ValidationError("Invalid record")

        with pytest.raises(ValidationError, match="Invalid record"):
            repository.bulk_import(iter([]))

    def test_bulk_import_raises_storage_error(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should propagate StorageError."""
        mock_database.bulk_import.side_effect = StorageError("Database error")

        with pytest.raises(StorageError, match="Database error"):
            repository.bulk_import(iter([]))

    def test_bulk_import_wraps_unexpected_errors(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should wrap unexpected errors in StorageError."""
        mock_database.bulk_import.side_effect = RuntimeError("Unexpected")

        with pytest.raises(StorageError, match="Failed to bulk import"):
            repository.bulk_import(iter([]))

    def test_bulk_import_returns_zero_for_empty_iterator(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should return (0, []) for empty iterator."""
        mock_database.bulk_import.return_value = (0, [])

        result = repository.bulk_import(iter([]))

        assert result == (0, [])

    def test_bulk_import_returns_correct_counts(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should return correct import counts."""
        mock_database.bulk_import.return_value = (100, [f"id{i}" for i in range(100)])

        result = repository.bulk_import(iter([{"content": "test"}] * 100))

        assert result[0] == 100
        assert len(result[1]) == 100


# ===========================================================================
# Integration-style Tests (Still Unit Tests with Mocks)
# ===========================================================================


class TestPhase5ProtocolCompliance:
    """Tests ensuring the adapter implements the Phase 5 protocol correctly."""

    def test_all_phase5_methods_exist(
        self, repository: LanceDBMemoryRepository
    ) -> None:
        """Verify all Phase 5 protocol methods exist on the repository."""
        phase5_methods = [
            "delete_by_namespace",
            "rename_namespace",
            "get_stats",
            "get_namespace_stats",
            "get_all_for_export",
            "bulk_import",
        ]

        for method_name in phase5_methods:
            assert hasattr(repository, method_name), f"Missing method: {method_name}"
            assert callable(
                getattr(repository, method_name)
            ), f"Not callable: {method_name}"

    def test_delete_by_namespace_returns_int(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """delete_by_namespace should return int."""
        mock_database.delete_by_namespace.return_value = 5

        result = repository.delete_by_namespace("test")

        assert isinstance(result, int)

    def test_rename_namespace_returns_int(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """rename_namespace should return int."""
        mock_database.rename_namespace.return_value = 10

        result = repository.rename_namespace("old", "new")

        assert isinstance(result, int)

    def test_get_stats_returns_dict(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_stats should return dict."""
        mock_database.get_stats.return_value = {}

        result = repository.get_stats()

        assert isinstance(result, dict)

    def test_get_namespace_stats_returns_dict(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_namespace_stats should return dict."""
        mock_database.get_namespace_stats.return_value = {}

        result = repository.get_namespace_stats("test")

        assert isinstance(result, dict)

    def test_get_all_for_export_returns_iterator(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """get_all_for_export should return an iterator/generator."""
        mock_database.get_all_for_export.return_value = iter([[]])

        result = repository.get_all_for_export()

        # Check it's iterable (generators are iterators)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_bulk_import_returns_tuple(
        self, repository: LanceDBMemoryRepository, mock_database: MagicMock
    ) -> None:
        """bulk_import should return tuple of (int, list[str])."""
        mock_database.bulk_import.return_value = (5, ["id1", "id2"])

        result = repository.bulk_import(iter([]))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], list)
