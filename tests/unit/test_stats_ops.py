"""Unit tests for database stats operations.

Tests the get_stats() and get_namespace_stats() database methods:
- Basic statistics retrieval
- Namespace-filtered statistics
- Health metrics integration
- Error handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatial_memory.core.database import Database, HealthMetrics, IndexStats
from spatial_memory.core.errors import StorageError


# =============================================================================
# TestGetStats
# =============================================================================


class TestGetStats:
    """Tests for get_stats() method."""

    def test_get_stats_returns_total_memories(self, database: Database) -> None:
        """get_stats should return total memory count."""
        # Add some test data
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory 1", vec, namespace="test")
        database.insert("Test memory 2", vec, namespace="test")

        stats = database.get_stats()

        assert stats["total_memories"] == 2

    def test_get_stats_returns_namespaces(self, database: Database) -> None:
        """get_stats should return memory counts by namespace."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="ns1")
        database.insert("Memory 2", vec, namespace="ns1")
        database.insert("Memory 3", vec, namespace="ns2")

        stats = database.get_stats()

        assert "namespaces" in stats
        assert stats["namespaces"]["ns1"] == 2
        assert stats["namespaces"]["ns2"] == 1

    def test_get_stats_returns_storage_info(self, database: Database) -> None:
        """get_stats should return storage bytes and MB."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory", vec)

        stats = database.get_stats()

        assert "storage_bytes" in stats
        assert "storage_mb" in stats
        assert stats["storage_bytes"] >= 0
        assert stats["storage_mb"] >= 0.0

    def test_get_stats_returns_index_info(self, database: Database) -> None:
        """get_stats should return vector and FTS index status."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory", vec)

        stats = database.get_stats()

        assert "has_vector_index" in stats
        assert "has_fts_index" in stats
        assert isinstance(stats["has_vector_index"], bool)
        assert isinstance(stats["has_fts_index"], bool)

    def test_get_stats_returns_fragment_info(self, database: Database) -> None:
        """get_stats should return fragment and compaction info."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory", vec)

        stats = database.get_stats()

        assert "num_fragments" in stats
        assert "needs_compaction" in stats
        assert isinstance(stats["num_fragments"], int)
        assert isinstance(stats["needs_compaction"], bool)

    def test_get_stats_returns_table_version(self, database: Database) -> None:
        """get_stats should return current table version."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory", vec)

        stats = database.get_stats()

        assert "table_version" in stats
        assert isinstance(stats["table_version"], int)
        assert stats["table_version"] >= 0

    def test_get_stats_returns_indices_list(self, database: Database) -> None:
        """get_stats should return list of index information."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Test memory", vec)

        stats = database.get_stats()

        assert "indices" in stats
        assert isinstance(stats["indices"], list)
        # Each index should have required fields
        for idx in stats["indices"]:
            assert "name" in idx
            assert "index_type" in idx
            assert "num_indexed_rows" in idx
            assert "status" in idx

    def test_get_stats_with_namespace_filter(self, database: Database) -> None:
        """get_stats with namespace filter should only count that namespace."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="ns1")
        database.insert("Memory 2", vec, namespace="ns1")
        database.insert("Memory 3", vec, namespace="ns2")

        stats = database.get_stats(namespace="ns1")

        # When filtered by namespace, total_memories should only count that namespace
        assert stats["total_memories"] == 2

    def test_get_stats_empty_database(self, database: Database) -> None:
        """get_stats on empty database should return zeros."""
        stats = database.get_stats()

        assert stats["total_memories"] == 0
        assert stats["namespaces"] == {}

    def test_get_stats_raises_storage_error_on_failure(
        self, database: Database
    ) -> None:
        """get_stats should raise StorageError on database failure."""
        from unittest.mock import patch, PropertyMock

        # Mock the table property to raise an exception (defeats auto-reconnect)
        with patch.object(
            type(database), "table", new_callable=PropertyMock
        ) as mock_table:
            mock_table.side_effect = Exception("Database connection failed")

            with pytest.raises(StorageError) as exc_info:
                database.get_stats()

            assert "Failed to get stats" in str(exc_info.value)


# =============================================================================
# TestGetNamespaceStats
# =============================================================================


class TestGetNamespaceStats:
    """Tests for get_namespace_stats() method."""

    def test_get_namespace_stats_returns_memory_count(
        self, database: Database
    ) -> None:
        """get_namespace_stats should return memory count for namespace."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="test_ns")
        database.insert("Memory 2", vec, namespace="test_ns")
        database.insert("Memory 3", vec, namespace="other")

        stats = database.get_namespace_stats("test_ns")

        assert stats["namespace"] == "test_ns"
        assert stats["memory_count"] == 2

    def test_get_namespace_stats_returns_date_ranges(
        self, database: Database
    ) -> None:
        """get_namespace_stats should return oldest and newest memory dates."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="test_ns")
        database.insert("Memory 2", vec, namespace="test_ns")

        stats = database.get_namespace_stats("test_ns")

        assert "oldest_memory" in stats
        assert "newest_memory" in stats
        # Both should be datetime or None
        if stats["memory_count"] > 0:
            assert stats["oldest_memory"] is not None
            assert stats["newest_memory"] is not None

    def test_get_namespace_stats_returns_avg_content_length(
        self, database: Database
    ) -> None:
        """get_namespace_stats should return average content length."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Short", vec, namespace="test_ns")
        database.insert("A much longer content string", vec, namespace="test_ns")

        stats = database.get_namespace_stats("test_ns")

        assert "avg_content_length" in stats
        # Average of len("Short")=5 and len("A much longer...")=28 = 16.5
        assert stats["avg_content_length"] is not None
        assert stats["avg_content_length"] > 0

    def test_get_namespace_stats_empty_namespace(self, database: Database) -> None:
        """get_namespace_stats for non-existent namespace should return zeros."""
        stats = database.get_namespace_stats("nonexistent")

        assert stats["namespace"] == "nonexistent"
        assert stats["memory_count"] == 0
        assert stats["oldest_memory"] is None
        assert stats["newest_memory"] is None

    def test_get_namespace_stats_validates_namespace(
        self, database: Database
    ) -> None:
        """get_namespace_stats should validate namespace input."""
        # Invalid namespace with SQL injection attempt
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError):
            database.get_namespace_stats("'; DROP TABLE memories; --")

    def test_get_namespace_stats_raises_storage_error_on_failure(
        self, database: Database
    ) -> None:
        """get_namespace_stats should raise StorageError on database failure."""
        from unittest.mock import patch, PropertyMock

        # Mock the table property to raise an exception (defeats auto-reconnect)
        with patch.object(
            type(database), "table", new_callable=PropertyMock
        ) as mock_table:
            mock_table.side_effect = Exception("Database connection failed")

            with pytest.raises(StorageError) as exc_info:
                database.get_namespace_stats("test")

            assert "Failed to get namespace stats" in str(exc_info.value)


# =============================================================================
# TestStatsIntegration
# =============================================================================


class TestStatsIntegration:
    """Integration tests for stats operations."""

    def test_stats_consistent_with_count(self, database: Database) -> None:
        """get_stats total should match count() method."""
        vec = np.random.randn(384).astype(np.float32)
        for i in range(5):
            database.insert(f"Memory {i}", vec, namespace="test")

        stats = database.get_stats()
        count = database.count()

        assert stats["total_memories"] == count

    def test_namespace_stats_consistent_with_filtered_count(
        self, database: Database
    ) -> None:
        """get_namespace_stats count should match filtered count()."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="ns1")
        database.insert("Memory 2", vec, namespace="ns1")
        database.insert("Memory 3", vec, namespace="ns2")

        ns_stats = database.get_namespace_stats("ns1")
        count = database.count(namespace="ns1")

        assert ns_stats["memory_count"] == count

    def test_all_namespaces_sum_to_total(self, database: Database) -> None:
        """Sum of namespace counts should equal total memories."""
        vec = np.random.randn(384).astype(np.float32)
        database.insert("Memory 1", vec, namespace="ns1")
        database.insert("Memory 2", vec, namespace="ns1")
        database.insert("Memory 3", vec, namespace="ns2")
        database.insert("Memory 4", vec, namespace="ns3")

        stats = database.get_stats()
        namespace_sum = sum(stats["namespaces"].values())

        assert namespace_sum == stats["total_memories"]
