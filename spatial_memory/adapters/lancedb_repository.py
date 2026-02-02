"""LanceDB repository adapter implementing MemoryRepositoryProtocol.

This adapter wraps the Database class to provide a clean interface
for the service layer, following Clean Architecture principles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
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
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        include_vector: bool = False,
    ) -> list[MemoryResult]:
        """Search for similar memories by vector.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.
            include_vector: Whether to include embedding vectors in results.
                Defaults to False to reduce response size.

        Returns:
            List of MemoryResult objects with similarity scores.
            If include_vector=True, each result includes its embedding vector.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            results = self._db.vector_search(
                query_vector,
                limit=limit,
                namespace=namespace,
                include_vector=include_vector,
            )
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

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        limit: int = 5,
        namespace: str | None = None,
        alpha: float = 0.5,
    ) -> list[MemoryResult]:
        """Search using both vector similarity and full-text search.

        Args:
            query_vector: Query embedding vector.
            query_text: Query text for FTS.
            limit: Maximum results.
            namespace: Optional namespace filter.
            alpha: Balance between vector (1.0) and FTS (0.0).

        Returns:
            List of matching memories ranked by combined score.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            results = self._db.hybrid_search(
                query=query_text,
                query_vector=query_vector,
                limit=limit,
                namespace=namespace,
                alpha=alpha,
            )
            return [self._record_to_memory_result(r) for r in results]
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in hybrid_search: {e}")
            raise StorageError(f"Failed to perform hybrid search: {e}") from e

    def get_health_metrics(self) -> dict[str, Any]:
        """Get database health metrics.

        Returns:
            Dictionary with health metrics.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            metrics = self._db.get_health_metrics()
            return asdict(metrics)
        except Exception as e:
            logger.error(f"Unexpected error in get_health_metrics: {e}")
            raise StorageError(f"Failed to get health metrics: {e}") from e

    def optimize(self) -> dict[str, Any]:
        """Run optimization and compaction.

        Returns:
            Dictionary with optimization results.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            return self._db.optimize()
        except Exception as e:
            logger.error(f"Unexpected error in optimize: {e}")
            raise StorageError(f"Failed to optimize database: {e}") from e

    def export_to_parquet(self, path: Path) -> int:
        """Export memories to Parquet file.

        Args:
            path: Output file path.

        Returns:
            Number of records exported.

        Raises:
            StorageError: If export fails.
        """
        try:
            result = self._db.export_to_parquet(output_path=path)
            rows_exported = result.get("rows_exported", 0)
            if not isinstance(rows_exported, int):
                raise StorageError("Invalid export result: rows_exported is not an integer")
            return rows_exported
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in export_to_parquet: {e}")
            raise StorageError(f"Failed to export to Parquet: {e}") from e

    def import_from_parquet(
        self,
        path: Path,
        namespace_override: str | None = None,
    ) -> int:
        """Import memories from Parquet file.

        Args:
            path: Input file path.
            namespace_override: Override namespace for imported memories.

        Returns:
            Number of records imported.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If import fails.
        """
        try:
            result = self._db.import_from_parquet(
                parquet_path=path,
                namespace_override=namespace_override,
            )
            rows_imported = result.get("rows_imported", 0)
            if not isinstance(rows_imported, int):
                raise StorageError("Invalid import result: rows_imported is not an integer")
            return rows_imported
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in import_from_parquet: {e}")
            raise StorageError(f"Failed to import from Parquet: {e}") from e

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

        # Include vector if present in record (when include_vector=True in search)
        vector = None
        if "vector" in record and record["vector"] is not None:
            # Convert to list for JSON serialization
            vec = record["vector"]
            vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)

        return MemoryResult(
            id=record["id"],
            content=record["content"],
            similarity=similarity,
            namespace=record["namespace"],
            tags=record.get("tags", []),
            importance=record["importance"],
            created_at=record["created_at"],
            metadata=record.get("metadata", {}),
            vector=vector,
        )

    # ========================================================================
    # Spatial Operations (Phase 4B)
    # ========================================================================

    def get_vectors_for_clustering(
        self,
        namespace: str | None = None,
        max_memories: int = 10_000,
    ) -> tuple[list[str], np.ndarray]:
        """Extract memory IDs and vectors efficiently for clustering.

        Optimized for memory efficiency with large datasets. Used by
        spatial operations like HDBSCAN clustering for region detection.

        Args:
            namespace: Filter to specific namespace.
            max_memories: Maximum memories to fetch.

        Returns:
            Tuple of (memory_ids, vectors_array) where vectors_array
            is a 2D numpy array of shape (n_memories, embedding_dim).

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            return self._db.get_vectors_for_clustering(
                namespace=namespace,
                max_memories=max_memories,
            )
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_vectors_for_clustering: {e}")
            raise StorageError(f"Failed to get vectors for clustering: {e}") from e

    def batch_vector_search(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Search for memories near multiple query points.

        Efficient for operations like journey interpolation where multiple
        points need to find nearby memories. Uses parallel execution when
        beneficial.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum results per query vector.
            namespace: Filter to specific namespace.

        Returns:
            List of result lists (one per query vector). Each result
            is a dict containing memory fields and similarity score.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            return self._db.batch_vector_search(
                query_vectors=query_vectors,
                limit_per_query=limit_per_query,
                namespace=namespace,
                parallel=len(query_vectors) > 3,  # Use parallel for larger batches
                max_workers=4,
            )
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in batch_vector_search: {e}")
            raise StorageError(f"Failed to perform batch vector search: {e}") from e

    def vector_search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories by vector (returns raw dict).

        Lower-level search that returns raw dictionary results instead
        of MemoryResult objects. Useful for spatial operations that need
        direct access to all fields including vectors.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.

        Returns:
            List of memory records as dictionaries with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            return self._db.vector_search(
                query_vector=query_vector,
                limit=limit,
                namespace=namespace,
            )
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in vector_search: {e}")
            raise StorageError(f"Failed to perform vector search: {e}") from e

    # ========================================================================
    # Phase 5 Protocol Extensions: Utility & Export/Import Operations
    # ========================================================================

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all memories in a namespace.

        Args:
            namespace: The namespace whose memories should be deleted.

        Returns:
            Number of memories deleted.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            return self._db.delete_by_namespace(namespace)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in delete_by_namespace: {e}")
            raise StorageError(f"Failed to delete namespace: {e}") from e

    def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
        """Rename all memories from one namespace to another.

        Args:
            old_namespace: The current namespace name (source).
            new_namespace: The new namespace name (target).

        Returns:
            Number of memories renamed.

        Raises:
            ValidationError: If namespace names are invalid.
            NamespaceNotFoundError: If old_namespace doesn't exist.
            StorageError: If database operation fails.
        """
        from spatial_memory.core.errors import NamespaceNotFoundError

        try:
            return self._db.rename_namespace(old_namespace, new_namespace)
        except (ValidationError, NamespaceNotFoundError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in rename_namespace: {e}")
            raise StorageError(f"Failed to rename namespace: {e}") from e

    def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Args:
            namespace: Filter statistics to a specific namespace.
                If None, returns statistics for all namespaces.

        Returns:
            Dictionary containing statistics.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            return self._db.get_stats(namespace)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_stats: {e}")
            raise StorageError(f"Failed to get stats: {e}") from e

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        """Get statistics for a specific namespace.

        Args:
            namespace: The namespace to get statistics for.

        Returns:
            Dictionary containing namespace statistics.

        Raises:
            ValidationError: If namespace is invalid.
            NamespaceNotFoundError: If namespace doesn't exist.
            StorageError: If database operation fails.
        """
        from spatial_memory.core.errors import NamespaceNotFoundError

        try:
            return self._db.get_namespace_stats(namespace)
        except (ValidationError, NamespaceNotFoundError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_namespace_stats: {e}")
            raise StorageError(f"Failed to get namespace stats: {e}") from e

    def get_all_for_export(
        self,
        namespace: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[list[dict[str, Any]]]:
        """Stream all memories for export in batches.

        Args:
            namespace: Filter to a specific namespace.
                If None, exports all namespaces.
            batch_size: Number of records per yielded batch.

        Yields:
            Batches of memory dictionaries.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            yield from self._db.get_all_for_export(namespace, batch_size)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_all_for_export: {e}")
            raise StorageError(f"Failed to export: {e}") from e

    def bulk_import(
        self,
        records: Iterator[dict[str, Any]],
        batch_size: int = 1000,
        namespace_override: str | None = None,
    ) -> tuple[int, list[str]]:
        """Import memories from an iterator of records.

        Args:
            records: Iterator of memory dictionaries.
            batch_size: Number of records per database insert batch.
            namespace_override: If provided, overrides the namespace
                field for all imported records.

        Returns:
            Tuple of (records_imported, list_of_new_ids).

        Raises:
            ValidationError: If records contain invalid data.
            StorageError: If database operation fails.
        """
        try:
            return self._db.bulk_import(records, batch_size, namespace_override)
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in bulk_import: {e}")
            raise StorageError(f"Failed to bulk import: {e}") from e
