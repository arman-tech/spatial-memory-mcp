"""LanceDB repository adapter implementing MemoryRepositoryProtocol.

This adapter wraps the Database class to provide a clean interface
for the service layer, following Clean Architecture principles.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from spatial_memory.core.errors import MemoryNotFoundError, StorageError, ValidationError
from spatial_memory.core.models import Memory, MemoryResult, MemorySource

if TYPE_CHECKING:
    from spatial_memory.core.database import Database

logger = logging.getLogger(__name__)


class LanceDBMemoryRepository:
    """Repository implementation using LanceDB.

    Implements MemoryRepositoryProtocol for use with MemoryService.
    """

    def __init__(self, database: Database) -> None:
        """Initialize the repository.

        Args:
            database: LanceDB database wrapper instance.
        """
        self._db = database

    def add(self, memory: Memory, vector: np.ndarray) -> str:
        """Add a memory with its embedding vector.

        Args:
            memory: The Memory object to store.
            vector: The embedding vector for the memory.

        Returns:
            The generated memory ID (UUID string).

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            return self._db.insert(
                content=memory.content,
                vector=vector,
                namespace=memory.namespace,
                tags=memory.tags,
                importance=memory.importance,
                source=memory.source.value,
                metadata=memory.metadata,
            )
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in add: {e}")
            raise StorageError(f"Failed to add memory: {e}") from e

    def add_batch(
        self,
        memories: list[Memory],
        vectors: list[np.ndarray],
    ) -> list[str]:
        """Add multiple memories efficiently.

        Args:
            memories: List of Memory objects to store.
            vectors: List of embedding vectors (same order as memories).

        Returns:
            List of generated memory IDs.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            records = []
            for memory, vector in zip(memories, vectors):
                records.append({
                    "content": memory.content,
                    "vector": vector,
                    "namespace": memory.namespace,
                    "tags": memory.tags,
                    "importance": memory.importance,
                    "source": memory.source.value,
                    "metadata": memory.metadata,
                })
            return self._db.insert_batch(records)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in add_batch: {e}")
            raise StorageError(f"Failed to add batch: {e}") from e

    def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            The Memory object, or None if not found.

        Raises:
            ValidationError: If memory_id is invalid.
            StorageError: If database operation fails.
        """
        try:
            record = self._db.get(memory_id)
            return self._record_to_memory(record)
        except MemoryNotFoundError:
            return None
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get: {e}")
            raise StorageError(f"Failed to get memory: {e}") from e

    def get_with_vector(self, memory_id: str) -> tuple[Memory, np.ndarray] | None:
        """Get a memory and its vector by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            Tuple of (Memory, vector), or None if not found.

        Raises:
            ValidationError: If memory_id is invalid.
            StorageError: If database operation fails.
        """
        try:
            record = self._db.get(memory_id)
            memory = self._record_to_memory(record)
            vector = np.array(record["vector"], dtype=np.float32)
            return (memory, vector)
        except MemoryNotFoundError:
            return None
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_with_vector: {e}")
            raise StorageError(f"Failed to get memory with vector: {e}") from e

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValidationError: If memory_id is invalid.
            StorageError: If database operation fails.
        """
        try:
            self._db.delete(memory_id)
            return True
        except MemoryNotFoundError:
            return False
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in delete: {e}")
            raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories atomically.

        Delegates to Database.delete_batch for proper encapsulation.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Number of memories actually deleted.

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        try:
            return self._db.delete_batch(memory_ids)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in delete_batch: {e}")
            raise StorageError(f"Failed to delete batch: {e}") from e

    def search(
        self,
        vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
    ) -> list[MemoryResult]:
        """Search for similar memories by vector.

        Args:
            vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.

        Returns:
            List of MemoryResult objects with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            results = self._db.vector_search(vector, limit=limit, namespace=namespace)
            return [self._record_to_memory_result(r) for r in results]
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise StorageError(f"Failed to search: {e}") from e

    def update_access(self, memory_id: str) -> None:
        """Update access timestamp and count for a memory.

        Args:
            memory_id: The memory UUID.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        try:
            self._db.update_access(memory_id)
        except (ValidationError, MemoryNotFoundError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in update_access: {e}")
            raise StorageError(f"Failed to update access: {e}") from e

    def update_access_batch(self, memory_ids: list[str]) -> int:
        """Update access timestamp and count for multiple memories.

        Delegates to Database.update_access_batch for proper encapsulation.

        Args:
            memory_ids: List of memory UUIDs.

        Returns:
            Number of memories successfully updated.

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        try:
            return self._db.update_access_batch(memory_ids)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in update_access_batch: {e}")
            raise StorageError(f"Batch access update failed: {e}") from e

    def update(self, memory_id: str, updates: dict[str, Any]) -> None:
        """Update a memory's fields.

        Args:
            memory_id: The memory UUID.
            updates: Fields to update.

        Raises:
            ValidationError: If input validation fails.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        try:
            self._db.update(memory_id, updates)
        except (ValidationError, MemoryNotFoundError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in update: {e}")
            raise StorageError(f"Failed to update memory: {e}") from e

    def count(self, namespace: str | None = None) -> int:
        """Count memories.

        Args:
            namespace: Filter to specific namespace.

        Returns:
            Number of memories.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            return self._db.count(namespace=namespace)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in count: {e}")
            raise StorageError(f"Failed to count memories: {e}") from e

    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces.

        Returns:
            List of namespace names.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            return self._db.get_namespaces()
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_namespaces: {e}")
            raise StorageError(f"Failed to get namespaces: {e}") from e

    def get_all(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[Memory, np.ndarray]]:
        """Get all memories with their vectors.

        Args:
            namespace: Filter to specific namespace.
            limit: Maximum number of results.

        Returns:
            List of (Memory, vector) tuples.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            records = self._db.get_all(namespace=namespace, limit=limit)
            results = []
            for record in records:
                memory = self._record_to_memory(record)
                vector = np.array(record["vector"], dtype=np.float32)
                results.append((memory, vector))
            return results
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_all: {e}")
            raise StorageError(f"Failed to get all memories: {e}") from e

    def _record_to_memory(self, record: dict[str, Any]) -> Memory:
        """Convert a database record to a Memory object.

        Args:
            record: Dictionary from database.

        Returns:
            Memory object.
        """
        # Handle source enum
        source_value = record.get("source", "manual")
        try:
            source = MemorySource(source_value)
        except ValueError:
            source = MemorySource.MANUAL

        return Memory(
            id=record["id"],
            content=record["content"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            last_accessed=record["last_accessed"],
            access_count=record["access_count"],
            importance=record["importance"],
            namespace=record["namespace"],
            tags=record.get("tags", []),
            source=source,
            metadata=record.get("metadata", {}),
        )

    def _record_to_memory_result(self, record: dict[str, Any]) -> MemoryResult:
        """Convert a search result record to a MemoryResult object.

        Args:
            record: Dictionary from database search.

        Returns:
            MemoryResult object.
        """
        # Clamp similarity to valid range [0, 1]
        # Cosine distance can sometimes produce values slightly outside this range
        similarity = record.get("similarity", 0.0)
        similarity = max(0.0, min(1.0, similarity))

        return MemoryResult(
            id=record["id"],
            content=record["content"],
            similarity=similarity,
            namespace=record["namespace"],
            tags=record.get("tags", []),
            importance=record["importance"],
            created_at=record["created_at"],
            metadata=record.get("metadata", {}),
        )
