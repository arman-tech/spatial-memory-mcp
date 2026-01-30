"""Unit tests for MemoryService.

TDD: These tests define the expected behavior for the MemoryService.
Tests use mocked repositories and embedding services for isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.core.errors import MemoryNotFoundError, ValidationError
from spatial_memory.core.models import Memory


class TestRemember:
    """Tests for MemoryService.remember() - storing new memories."""

    def test_remember_stores_memory_and_returns_id(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember() should embed content, store memory, and return result."""
        # Given
        expected_id = "test-uuid-1234"
        test_vector = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.embed.return_value = test_vector
        mock_repository.add.return_value = expected_id

        # When
        result = memory_service.remember(
            content="Test content for memory",
            namespace="default",
        )

        # Then
        assert result.id == expected_id
        assert result.content == "Test content for memory"
        assert result.namespace == "default"
        mock_embeddings.embed.assert_called_once_with("Test content for memory")
        mock_repository.add.assert_called_once()

        # Verify the Memory object passed to repository
        call_args = mock_repository.add.call_args
        memory_arg = call_args[0][0]
        vector_arg = call_args[0][1]
        assert isinstance(memory_arg, Memory)
        assert memory_arg.content == "Test content for memory"
        assert memory_arg.namespace == "default"
        np.testing.assert_array_equal(vector_arg, test_vector)

    def test_remember_with_optional_parameters(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember() should accept and store optional parameters."""
        # Given
        expected_id = "test-uuid-5678"
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_repository.add.return_value = expected_id

        # When
        result = memory_service.remember(
            content="Tagged important memory",
            namespace="work",
            tags=["important", "project-x"],
            importance=0.9,
            metadata={"source": "meeting"},
        )

        # Then
        assert result.id == expected_id
        assert result.namespace == "work"

        # Verify optional params passed through
        call_args = mock_repository.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.tags == ["important", "project-x"]
        assert memory_arg.importance == 0.9
        assert memory_arg.metadata == {"source": "meeting"}

    def test_remember_validates_empty_content(
        self,
        memory_service: Any,
    ) -> None:
        """remember() should raise ValidationError for empty content."""
        with pytest.raises(ValidationError, match="[Cc]ontent.*empty"):
            memory_service.remember(content="")

    def test_remember_validates_whitespace_only_content(
        self,
        memory_service: Any,
    ) -> None:
        """remember() should raise ValidationError for whitespace-only content."""
        with pytest.raises(ValidationError, match="[Cc]ontent.*empty"):
            memory_service.remember(content="   \t\n  ")

    def test_remember_validates_importance_range(
        self,
        memory_service: Any,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember() should raise ValidationError for invalid importance."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)

        with pytest.raises(ValidationError, match="[Ii]mportance"):
            memory_service.remember(content="Test", importance=1.5)

        with pytest.raises(ValidationError, match="[Ii]mportance"):
            memory_service.remember(content="Test", importance=-0.1)

    def test_remember_uses_default_namespace(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember() should use 'default' namespace when not specified."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_repository.add.return_value = "test-id"

        memory_service.remember(content="Test content")

        call_args = mock_repository.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.namespace == "default"


class TestRememberBatch:
    """Tests for MemoryService.remember_batch() - storing multiple memories."""

    def test_remember_batch_stores_multiple_memories(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember_batch() should store multiple memories efficiently."""
        # Given
        expected_ids = ["id-1", "id-2", "id-3"]
        vectors = [np.array([0.1] * 384, dtype=np.float32) for _ in range(3)]
        mock_embeddings.embed_batch.return_value = vectors
        mock_repository.add_batch.return_value = expected_ids

        contents = [
            {"content": "Memory 1"},
            {"content": "Memory 2"},
            {"content": "Memory 3"},
        ]

        # When
        result = memory_service.remember_batch(contents)

        # Then
        assert result.ids == expected_ids
        assert result.count == 3
        mock_embeddings.embed_batch.assert_called_once()
        mock_repository.add_batch.assert_called_once()

    def test_remember_batch_with_mixed_namespaces(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """remember_batch() should handle mixed namespaces."""
        vectors = [np.array([0.1] * 384, dtype=np.float32) for _ in range(2)]
        mock_embeddings.embed_batch.return_value = vectors
        mock_repository.add_batch.return_value = ["id-1", "id-2"]

        contents = [
            {"content": "Work memory", "namespace": "work"},
            {"content": "Personal memory", "namespace": "personal"},
        ]

        result = memory_service.remember_batch(contents)

        assert result.count == 2
        call_args = mock_repository.add_batch.call_args
        memories_arg = call_args[0][0]
        assert memories_arg[0].namespace == "work"
        assert memories_arg[1].namespace == "personal"

    def test_remember_batch_validates_empty_list(
        self,
        memory_service: Any,
    ) -> None:
        """remember_batch() should raise ValidationError for empty list."""
        with pytest.raises(ValidationError, match="[Ee]mpty"):
            memory_service.remember_batch([])


class TestRecall:
    """Tests for MemoryService.recall() - searching memories."""

    def test_recall_returns_similar_memories(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        make_memory_result: Any,
    ) -> None:
        """recall() should embed query and return similar memories."""
        # Given
        query_vector = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.embed.return_value = query_vector

        expected_results = [
            make_memory_result(id="mem-1", content="Similar memory 1", similarity=0.95),
            make_memory_result(id="mem-2", content="Similar memory 2", similarity=0.87),
        ]
        mock_repository.search.return_value = expected_results

        # When
        result = memory_service.recall(query="test query", limit=5)

        # Then
        assert len(result.memories) == 2
        assert result.total == 2
        assert result.memories[0].similarity == 0.95
        assert result.memories[0].id == "mem-1"
        mock_embeddings.embed.assert_called_once_with("test query")
        mock_repository.search.assert_called_once()

    def test_recall_filters_by_namespace(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """recall() should pass namespace filter to repository."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_repository.search.return_value = []

        memory_service.recall(query="test", namespace="work", limit=10)

        call_args = mock_repository.search.call_args
        assert call_args.kwargs.get("namespace") == "work"
        assert call_args.kwargs.get("limit") == 10

    def test_recall_filters_by_min_similarity(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        make_memory_result: Any,
    ) -> None:
        """recall() should filter results below min_similarity threshold."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_repository.search.return_value = [
            make_memory_result(similarity=0.9),
            make_memory_result(similarity=0.7),
            make_memory_result(similarity=0.5),
        ]

        result = memory_service.recall(query="test", min_similarity=0.6)

        # Should filter out the 0.5 result
        assert len(result.memories) == 2
        assert all(m.similarity >= 0.6 for m in result.memories)

    def test_recall_updates_access_for_returned_memories(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        make_memory_result: Any,
    ) -> None:
        """recall() should update access stats for returned memories (batch)."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_repository.search.return_value = [
            make_memory_result(id="mem-1", similarity=0.9),
            make_memory_result(id="mem-2", similarity=0.8),
        ]
        mock_repository.update_access_batch.return_value = 2

        memory_service.recall(query="test")

        # Should call batch update with all memory IDs
        mock_repository.update_access_batch.assert_called_once()
        call_args = mock_repository.update_access_batch.call_args[0][0]
        assert "mem-1" in call_args
        assert "mem-2" in call_args

    def test_recall_validates_empty_query(
        self,
        memory_service: Any,
    ) -> None:
        """recall() should raise ValidationError for empty query."""
        with pytest.raises(ValidationError, match="[Qq]uery.*empty"):
            memory_service.recall(query="")

    def test_recall_validates_limit_range(
        self,
        memory_service: Any,
        mock_embeddings: MagicMock,
    ) -> None:
        """recall() should raise ValidationError for invalid limit."""
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)

        with pytest.raises(ValidationError, match="[Ll]imit"):
            memory_service.recall(query="test", limit=0)

        with pytest.raises(ValidationError, match="[Ll]imit"):
            memory_service.recall(query="test", limit=-1)


class TestNearby:
    """Tests for MemoryService.nearby() - finding neighbors of a memory."""

    def test_nearby_finds_neighbors(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        make_memory: Any,
        make_memory_result: Any,
        make_vector: Any,
    ) -> None:
        """nearby() should find memories similar to a reference memory."""
        # Given
        ref_memory = make_memory(id="ref-id", content="Reference memory")
        ref_vector = make_vector()
        mock_repository.get_with_vector.return_value = (ref_memory, ref_vector)

        neighbors = [
            make_memory_result(id="neighbor-1", similarity=0.92),
            make_memory_result(id="neighbor-2", similarity=0.85),
        ]
        mock_repository.search.return_value = neighbors

        # When
        result = memory_service.nearby(memory_id="ref-id", limit=5)

        # Then
        assert result.reference.id == "ref-id"
        assert len(result.neighbors) == 2
        assert result.neighbors[0].similarity == 0.92
        mock_repository.get_with_vector.assert_called_once_with("ref-id")

    def test_nearby_excludes_reference_memory(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        make_memory: Any,
        make_memory_result: Any,
        make_vector: Any,
    ) -> None:
        """nearby() should exclude the reference memory from results."""
        ref_memory = make_memory(id="ref-id")
        ref_vector = make_vector()
        mock_repository.get_with_vector.return_value = (ref_memory, ref_vector)

        # Repository returns results including the reference
        mock_repository.search.return_value = [
            make_memory_result(id="ref-id", similarity=1.0),  # Reference itself
            make_memory_result(id="neighbor-1", similarity=0.9),
        ]

        result = memory_service.nearby(memory_id="ref-id", limit=5)

        # Reference should be excluded from neighbors
        neighbor_ids = [n.id for n in result.neighbors]
        assert "ref-id" not in neighbor_ids
        assert len(result.neighbors) == 1

    def test_nearby_raises_for_nonexistent_memory(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
    ) -> None:
        """nearby() should raise MemoryNotFoundError for unknown ID."""
        mock_repository.get_with_vector.return_value = None

        with pytest.raises(MemoryNotFoundError):
            memory_service.nearby(memory_id="nonexistent-id")

    def test_nearby_filters_by_namespace(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
        make_memory: Any,
        make_vector: Any,
    ) -> None:
        """nearby() should pass namespace filter to search."""
        ref_memory = make_memory(id="ref-id")
        ref_vector = make_vector()
        mock_repository.get_with_vector.return_value = (ref_memory, ref_vector)
        mock_repository.search.return_value = []

        memory_service.nearby(memory_id="ref-id", namespace="work", limit=5)

        call_args = mock_repository.search.call_args
        assert call_args.kwargs.get("namespace") == "work"


class TestForget:
    """Tests for MemoryService.forget() - deleting memories."""

    def test_forget_deletes_single_memory(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
    ) -> None:
        """forget() should delete a single memory by ID."""
        mock_repository.delete.return_value = True

        result = memory_service.forget(memory_id="mem-to-delete")

        assert result.deleted == 1
        assert result.ids == ["mem-to-delete"]
        mock_repository.delete.assert_called_once_with("mem-to-delete")

    def test_forget_returns_zero_for_nonexistent(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
    ) -> None:
        """forget() should return deleted=0 for nonexistent memory."""
        mock_repository.delete.return_value = False

        result = memory_service.forget(memory_id="nonexistent")

        assert result.deleted == 0


class TestForgetBatch:
    """Tests for MemoryService.forget_batch() - deleting multiple memories."""

    def test_forget_batch_deletes_multiple_memories(
        self,
        memory_service: Any,
        mock_repository: MagicMock,
    ) -> None:
        """forget_batch() should delete multiple memories."""
        mock_repository.delete_batch.return_value = 3

        result = memory_service.forget_batch(memory_ids=["id-1", "id-2", "id-3"])

        assert result.deleted == 3
        mock_repository.delete_batch.assert_called_once_with(["id-1", "id-2", "id-3"])

    def test_forget_batch_validates_empty_list(
        self,
        memory_service: Any,
    ) -> None:
        """forget_batch() should raise ValidationError for empty list."""
        with pytest.raises(ValidationError, match="[Ee]mpty"):
            memory_service.forget_batch(memory_ids=[])


class TestMemoryServiceInitialization:
    """Tests for MemoryService initialization and configuration."""

    def test_memory_service_requires_repository(
        self,
        mock_embeddings: MagicMock,
    ) -> None:
        """MemoryService should require a repository."""
        from spatial_memory.services.memory import MemoryService

        with pytest.raises(TypeError):
            MemoryService(embeddings=mock_embeddings)  # type: ignore

    def test_memory_service_requires_embeddings(
        self,
        mock_repository: MagicMock,
    ) -> None:
        """MemoryService should require an embedding service."""
        from spatial_memory.services.memory import MemoryService

        with pytest.raises(TypeError):
            MemoryService(repository=mock_repository)  # type: ignore
