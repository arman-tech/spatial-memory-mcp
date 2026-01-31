"""Tests for database safety safeguards.

These tests verify the safety mechanisms added to prevent:
- Infinite loops in batch operations
- Data loss from non-atomic operations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.errors import StorageError


# ===========================================================================
# TestRenameNamespaceSafeguards
# ===========================================================================


class TestRenameNamespaceSafeguards:
    """Tests for rename_namespace safety mechanisms."""

    def test_rename_namespace_iteration_limit(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Verify rename_namespace raises StorageError when iteration limit exceeded.

        This tests the safety mechanism that prevents infinite loops when
        merge_insert silently fails to update records (records keep appearing
        in search results with the old namespace).
        """
        # Insert a test memory so namespace exists
        vec = embedding_service.embed("test content")
        database.insert(content="test content", vector=vec, namespace="old-ns")

        # Store original table reference
        original_table = database.table

        # Create a mock that always returns records with old namespace
        # This simulates merge_insert silently failing to update the namespace
        def mock_search_factory():
            mock_search = MagicMock()
            mock_where = MagicMock()
            mock_limit = MagicMock()

            # Always return the same record with old namespace
            mock_limit.to_list.return_value = [
                {
                    "id": "fake-id-12345678-1234-1234-1234-123456789012",
                    "namespace": "old-ns",
                    "content": "test content",
                    "vector": [0.1] * 384,
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "expires_at": None,
                    "importance": 0.5,
                    "access_count": 0,
                    "last_accessed": "2024-01-01T00:00:00Z",
                }
            ]
            mock_where.limit.return_value = mock_limit
            mock_search.where.return_value = mock_where
            return mock_search

        # Mock merge_insert to do nothing (simulating silent failure)
        mock_merge = MagicMock()
        mock_chain = MagicMock()
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain
        mock_chain.execute.return_value = None
        mock_merge.return_value = mock_chain

        # Patch table search, merge_insert, and get_namespaces to simulate the scenario
        # get_namespaces must return old-ns so the check passes
        with patch.object(
            original_table, "search", side_effect=lambda: mock_search_factory()
        ), patch.object(original_table, "merge_insert", mock_merge), patch.object(
            database, "get_namespaces", return_value=["old-ns"]
        ):
            with pytest.raises(StorageError, match="exceeded maximum iterations"):
                database.rename_namespace("old-ns", "new-ns")


# ===========================================================================
# TestSetMemoryTTLSafeguards
# ===========================================================================


class TestSetMemoryTTLSafeguards:
    """Tests for set_memory_ttl atomicity."""

    def test_set_memory_ttl_preserves_data_on_failure(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Verify TTL update is atomic - no data loss on failure.

        The set_memory_ttl method uses merge_insert instead of delete+insert,
        which means if the operation fails, the original data is preserved.
        """
        # Insert a memory with important data
        vec = embedding_service.embed("important data")
        memory_id = database.insert(content="important data", vector=vec)

        # Store original table reference
        original_table = database.table

        # Mock merge_insert to fail
        mock_merge = MagicMock()
        mock_chain = MagicMock()
        mock_chain.when_matched_update_all.return_value = mock_chain
        mock_chain.when_not_matched_insert_all.return_value = mock_chain
        mock_chain.execute.side_effect = Exception("Database failure")
        mock_merge.return_value = mock_chain

        # Patch only the merge_insert method to fail
        with patch.object(original_table, "merge_insert", mock_merge):
            with pytest.raises(StorageError, match="Failed to set memory TTL"):
                database.set_memory_ttl(memory_id, ttl_days=7)

        # Memory should still exist (atomic - no delete happened)
        preserved = database.get(memory_id)
        assert preserved is not None
        assert preserved["content"] == "important data"
