"""LanceDB repository adapter implementing MemoryRepositoryProtocol.

This adapter wraps the Database class to provide a clean interface
for the service layer, following Clean Architecture principles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatial_memory.core.errors import MemoryNotFoundError
from spatial_memory.core.models import Memory, MemoryResult, MemorySource

if TYPE_CHECKING:
    from spatial_memory.core.database import Database


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
        """
        return self._db.insert(
            content=memory.content,
            vector=vector,
            namespace=memory.namespace,
            tags=memory.tags,
            importance=memory.importance,
            source=memory.source.value,
            metadata=memory.metadata,
        )

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
        """
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

    def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            The Memory object, or None if not found.
        """
        try:
            record = self._db.get(memory_id)
            return self._record_to_memory(record)
        except MemoryNotFoundError:
            return None

    def get_with_vector(self, memory_id: str) -> tuple[Memory, np.ndarray] | None:
        """Get a memory and its vector by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            Tuple of (Memory, vector), or None if not found.
        """
        try:
            record = self._db.get(memory_id)
            memory = self._record_to_memory(record)
            vector = np.array(record["vector"], dtype=np.float32)
            return (memory, vector)
        except MemoryNotFoundError:
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory UUID.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self._db.delete(memory_id)
            return True
        except MemoryNotFoundError:
            return False

    def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Number of memories actually deleted.
        """
        deleted_count = 0
        for memory_id in memory_ids:
            try:
                self._db.delete(memory_id)
                deleted_count += 1
            except MemoryNotFoundError:
                continue
        return deleted_count

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
        """
        results = self._db.vector_search(vector, limit=limit, namespace=namespace)
        return [self._record_to_memory_result(r) for r in results]

    def update_access(self, memory_id: str) -> None:
        """Update access timestamp and count for a memory.

        Args:
            memory_id: The memory UUID.
        """
        self._db.update_access(memory_id)

    def update(self, memory_id: str, updates: dict[str, Any]) -> None:
        """Update a memory's fields.

        Args:
            memory_id: The memory UUID.
            updates: Fields to update.
        """
        self._db.update(memory_id, updates)

    def count(self, namespace: str | None = None) -> int:
        """Count memories.

        Args:
            namespace: Filter to specific namespace.

        Returns:
            Number of memories.
        """
        return self._db.count(namespace=namespace)

    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces.

        Returns:
            List of namespace names.
        """
        return self._db.get_namespaces()

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
        """
        records = self._db.get_all(namespace=namespace, limit=limit)
        results = []
        for record in records:
            memory = self._record_to_memory(record)
            vector = np.array(record["vector"], dtype=np.float32)
            results.append((memory, vector))
        return results

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
