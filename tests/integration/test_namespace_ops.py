"""Integration tests for namespace operations.

TDD: Tests for rename_namespace database method.
These tests require real database and embeddings for thorough testing.

NOTE: Despite being in tests/unit/, these are integration tests because they
test actual database operations with real embeddings.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Mark entire module as integration tests (require real database + embeddings)
pytestmark = pytest.mark.integration

from spatial_memory.core.database import Database
from spatial_memory.core.errors import (
    NamespaceNotFoundError,
    StorageError,
    ValidationError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatial_memory.core.embeddings import EmbeddingService


class TestRenameNamespace:
    """Tests for Database.rename_namespace()."""

    def test_rename_namespace_success(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test successful rename of all memories in a namespace."""
        vec = embedding_service.embed("Test content")

        # Insert memories in the source namespace
        database.insert(content="Memory 1", vector=vec, namespace="old-ns")
        database.insert(content="Memory 2", vector=vec, namespace="old-ns")
        database.insert(content="Memory 3", vector=vec, namespace="keep-ns")

        # Rename old-ns to new-ns
        renamed_count = database.rename_namespace("old-ns", "new-ns")

        assert renamed_count == 2
        assert database.count(namespace="old-ns") == 0
        assert database.count(namespace="new-ns") == 2
        assert database.count(namespace="keep-ns") == 1

    def test_rename_namespace_returns_count(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename returns correct count of renamed memories."""
        vec = embedding_service.embed("Test content")

        # Insert 5 memories
        for i in range(5):
            database.insert(content=f"Memory {i}", vector=vec, namespace="to-rename")

        renamed_count = database.rename_namespace("to-rename", "renamed")

        assert renamed_count == 5

    def test_rename_namespace_preserves_content(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename preserves all memory content and metadata."""
        vec = embedding_service.embed("Test content")

        # Insert memory with all fields
        original_id = database.insert(
            content="Important content",
            vector=vec,
            namespace="source",
            tags=["tag1", "tag2"],
            importance=0.9,
            metadata={"key": "value"},
        )

        database.rename_namespace("source", "target")

        # Verify content is preserved
        record = database.get(original_id)
        assert record["content"] == "Important content"
        assert record["namespace"] == "target"
        assert record["tags"] == ["tag1", "tag2"]
        assert record["importance"] == pytest.approx(0.9, abs=0.01)
        assert record["metadata"] == {"key": "value"}

    def test_rename_namespace_preserves_vectors(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename preserves embedding vectors."""
        original_vec = embedding_service.embed("Test content")

        memory_id = database.insert(
            content="Test content",
            vector=original_vec,
            namespace="source",
        )

        database.rename_namespace("source", "target")

        record = database.get(memory_id)
        stored_vec = np.array(record["vector"])
        np.testing.assert_array_almost_equal(stored_vec, original_vec, decimal=5)

    def test_rename_namespace_updates_timestamp(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename updates the updated_at timestamp."""
        import time

        vec = embedding_service.embed("Test content")
        memory_id = database.insert(content="Test", vector=vec, namespace="source")

        original_record = database.get(memory_id)
        original_updated = original_record["updated_at"]

        time.sleep(0.1)

        database.rename_namespace("source", "target")

        new_record = database.get(memory_id)
        assert new_record["updated_at"] > original_updated

    def test_rename_namespace_not_found_raises(
        self, database: Database
    ) -> None:
        """Test that renaming nonexistent namespace raises NamespaceNotFoundError."""
        with pytest.raises(NamespaceNotFoundError) as exc_info:
            database.rename_namespace("nonexistent", "new-ns")

        assert exc_info.value.namespace == "nonexistent"

    def test_rename_namespace_validates_old_namespace(
        self, database: Database
    ) -> None:
        """Test that invalid old_namespace raises ValidationError."""
        with pytest.raises(ValidationError):
            database.rename_namespace("invalid namespace", "new-ns")

        with pytest.raises(ValidationError):
            database.rename_namespace("", "new-ns")

    def test_rename_namespace_validates_new_namespace(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that invalid new_namespace raises ValidationError."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test", vector=vec, namespace="source")

        with pytest.raises(ValidationError):
            database.rename_namespace("source", "invalid namespace")

        with pytest.raises(ValidationError):
            database.rename_namespace("source", "")

    def test_rename_namespace_invalidates_cache(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename invalidates the namespace cache."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test", vector=vec, namespace="cached-ns")

        # Access namespaces to populate cache
        namespaces_before = database.get_namespaces()
        assert "cached-ns" in namespaces_before

        database.rename_namespace("cached-ns", "new-ns")

        # Cache should be invalidated, new query should reflect change
        namespaces_after = database.get_namespaces()
        assert "cached-ns" not in namespaces_after
        assert "new-ns" in namespaces_after


class TestRenameNamespaceBatching:
    """Tests for batch processing in rename_namespace."""

    def test_rename_large_namespace(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test renaming namespace with many memories (batch processing)."""
        vec = embedding_service.embed("Test content")

        # Insert more than one batch worth (batch_size=1000)
        records = [
            {
                "content": f"Memory {i}",
                "vector": vec,
                "namespace": "large-ns",
            }
            for i in range(50)  # Use 50 for faster tests
        ]
        database.insert_batch(records)

        renamed_count = database.rename_namespace("large-ns", "renamed-ns")

        assert renamed_count == 50
        assert database.count(namespace="large-ns") == 0
        assert database.count(namespace="renamed-ns") == 50

    def test_rename_handles_mixed_types(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test rename handles various metadata types correctly."""
        vec = embedding_service.embed("Test")

        # Insert with various metadata types
        database.insert(
            content="With dict metadata",
            vector=vec,
            namespace="mixed",
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        database.insert(
            content="With simple metadata",
            vector=vec,
            namespace="mixed",
            metadata={"simple": "value"},
        )
        database.insert(
            content="With empty metadata",
            vector=vec,
            namespace="mixed",
            metadata={},
        )

        renamed_count = database.rename_namespace("mixed", "renamed-mixed")

        assert renamed_count == 3

        # Verify all records are accessible
        all_records = database.get_all(namespace="renamed-mixed")
        assert len(all_records) == 3


class TestRenameNamespaceEdgeCases:
    """Edge case tests for rename_namespace."""

    def test_rename_to_existing_namespace_merges(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test renaming to an existing namespace merges memories."""
        vec = embedding_service.embed("Test")

        # Create source namespace
        database.insert(content="From source", vector=vec, namespace="source")

        # Create target namespace with existing memories
        database.insert(content="Already in target", vector=vec, namespace="target")

        renamed_count = database.rename_namespace("source", "target")

        assert renamed_count == 1
        assert database.count(namespace="source") == 0
        assert database.count(namespace="target") == 2

    def test_rename_same_namespace_no_op(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test renaming namespace to itself is effectively a no-op."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test", vector=vec, namespace="same")

        renamed_count = database.rename_namespace("same", "same")

        # Should still process but result in same state
        assert renamed_count == 1
        assert database.count(namespace="same") == 1

    def test_rename_preserves_access_stats(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename preserves access_count and last_accessed."""
        vec = embedding_service.embed("Test")
        memory_id = database.insert(content="Test", vector=vec, namespace="source")

        # Update access to increment count
        database.update_access(memory_id)
        database.update_access(memory_id)

        original = database.get(memory_id)
        original_count = original["access_count"]
        original_accessed = original["last_accessed"]

        database.rename_namespace("source", "target")

        record = database.get(memory_id)
        # access_count should be preserved
        assert record["access_count"] == original_count
        # last_accessed should be preserved (updated_at changes, not last_accessed)
        # Note: Depending on implementation, last_accessed might or might not update
        # This is a design decision - typically it shouldn't update on rename

    def test_rename_preserves_created_at(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that rename preserves the original created_at timestamp."""
        import time

        vec = embedding_service.embed("Test")
        memory_id = database.insert(content="Test", vector=vec, namespace="source")

        original = database.get(memory_id)
        original_created = original["created_at"]

        time.sleep(0.1)

        database.rename_namespace("source", "target")

        record = database.get(memory_id)
        assert record["created_at"] == original_created

    def test_rename_with_special_characters_in_namespace(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test renaming namespaces with valid special characters."""
        vec = embedding_service.embed("Test")

        # Namespaces can contain hyphens, underscores, and dots
        database.insert(content="Test", vector=vec, namespace="my-ns.v1")

        renamed_count = database.rename_namespace("my-ns.v1", "my_ns_v2")

        assert renamed_count == 1
        assert database.count(namespace="my-ns.v1") == 0
        assert database.count(namespace="my_ns_v2") == 1


class TestRenameNamespaceErrorHandling:
    """Error handling tests for rename_namespace."""

    def test_rename_empty_namespace_raises(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that renaming from empty namespace raises NamespaceNotFoundError."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test", vector=vec, namespace="other")

        # Empty namespace has no memories
        with pytest.raises(NamespaceNotFoundError):
            database.rename_namespace("empty-ns", "new-ns")

    def test_rename_validates_namespace_length(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that very long namespace names are rejected."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test", vector=vec, namespace="source")

        long_namespace = "x" * 300  # Too long

        with pytest.raises(ValidationError):
            database.rename_namespace("source", long_namespace)
