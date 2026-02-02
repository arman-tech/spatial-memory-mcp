"""Protocol interfaces for repository and embedding services.

These protocols define the contracts between the service layer and infrastructure.
Using typing.Protocol enables structural subtyping (duck typing with type checking).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from spatial_memory.core.errors import NamespaceNotFoundError
from spatial_memory.core.models import Memory, MemoryResult


class MemoryRepositoryProtocol(Protocol):
    """Protocol for memory storage and retrieval operations.

    Implementations must provide all methods defined here.
    The LanceDBMemoryRepository is the primary implementation.
    """

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Number of memories actually deleted.

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        ...

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
        ...

    def update_access(self, memory_id: str) -> None:
        """Update access timestamp and count for a memory.

        Args:
            memory_id: The memory UUID.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        ...

    def update_access_batch(self, memory_ids: list[str]) -> int:
        """Update access timestamp and count for multiple memories.

        Args:
            memory_ids: List of memory UUIDs.

        Returns:
            Number of memories successfully updated.

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        ...

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
        ...

    def get_batch(self, memory_ids: list[str]) -> dict[str, Memory]:
        """Get multiple memories by ID in a single query.

        Args:
            memory_ids: List of memory UUIDs to retrieve.

        Returns:
            Dict mapping memory_id to Memory object. Missing IDs are not included.

        Raises:
            ValidationError: If any memory_id format is invalid.
            StorageError: If database operation fails.
        """
        ...

    def update_batch(
        self, updates: list[tuple[str, dict[str, Any]]]
    ) -> tuple[int, list[str]]:
        """Update multiple memories in a single batch operation.

        Args:
            updates: List of (memory_id, updates_dict) tuples.

        Returns:
            Tuple of (success_count, list of failed memory_ids).

        Raises:
            StorageError: If database operation fails completely.
        """
        ...

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
        ...

    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces.

        Returns:
            List of namespace names.

        Raises:
            StorageError: If database operation fails.
        """
        ...

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
        ...

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
        ...

    def get_health_metrics(self) -> dict[str, Any]:
        """Get database health metrics.

        Returns:
            Dictionary with health metrics.

        Raises:
            StorageError: If database operation fails.
        """
        ...

    def optimize(self) -> dict[str, Any]:
        """Run optimization and compaction.

        Returns:
            Dictionary with optimization results.

        Raises:
            StorageError: If database operation fails.
        """
        ...

    def export_to_parquet(self, path: Path) -> int:
        """Export memories to Parquet file.

        Args:
            path: Output file path.

        Returns:
            Number of records exported.

        Raises:
            StorageError: If export fails.
        """
        ...

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
        ...

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
        ...

    def batch_vector_search(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Search for memories near multiple query points.

        Efficient for operations like journey interpolation where multiple
        points need to find nearby memories. Supports parallel execution.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum results per query vector.
            namespace: Filter to specific namespace.
            include_vector: Whether to include embedding vectors in results.
                Defaults to False to reduce response size.

        Returns:
            List of result lists (one per query vector). Each result
            is a dict containing memory fields and similarity score.
            If include_vector=True, each dict includes the 'vector' field.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        ...

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
        ...

    # -------------------------------------------------------------------------
    # Phase 5 Protocol Extensions: Utility & Export/Import Operations
    # -------------------------------------------------------------------------

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all memories in a namespace.

        Removes all memories belonging to the specified namespace from storage.
        This is a destructive operation that cannot be undone.

        Args:
            namespace: The namespace whose memories should be deleted.

        Returns:
            Number of memories deleted.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        ...

    def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
        """Rename all memories from one namespace to another.

        Atomically updates the namespace field for all memories belonging
        to the source namespace. Uses batch updates for data integrity.

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
        ...

    def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Retrieves statistics about the memory database including total counts,
        storage size, index information, and health metrics. Uses efficient
        database-level aggregation queries.

        Args:
            namespace: Filter statistics to a specific namespace.
                If None, returns statistics for all namespaces.

        Returns:
            Dictionary containing:
                - total_memories: Total count of memories
                - memories_by_namespace: Dict mapping namespace to count
                - storage_bytes: Total storage size in bytes
                - storage_mb: Total storage size in megabytes
                - has_vector_index: Whether vector index exists
                - has_fts_index: Whether full-text search index exists
                - num_fragments: Number of storage fragments
                - needs_compaction: Whether compaction is recommended
                - table_version: Current table version number
                - indices: List of index information dicts

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        ...

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        """Get statistics for a specific namespace.

        Retrieves detailed statistics for a single namespace including
        memory count, date ranges, and storage estimates.

        Args:
            namespace: The namespace to get statistics for.

        Returns:
            Dictionary containing:
                - namespace: The namespace name
                - memory_count: Number of memories in namespace
                - oldest_memory: Datetime of oldest memory (or None)
                - newest_memory: Datetime of newest memory (or None)
                - avg_content_length: Average content length (optional)

        Raises:
            ValidationError: If namespace is invalid.
            NamespaceNotFoundError: If namespace doesn't exist.
            StorageError: If database operation fails.
        """
        ...

    def get_all_for_export(
        self,
        namespace: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[list[dict[str, Any]]]:
        """Stream all memories for export in batches.

        Memory-efficient export using generator pattern. Yields batches
        of memory records suitable for writing to export files. This method
        handles only data retrieval; file I/O is the service layer's concern.

        Args:
            namespace: Filter to a specific namespace.
                If None, exports all namespaces.
            batch_size: Number of records per yielded batch.

        Yields:
            Batches of memory dictionaries. Each dictionary contains all
            memory fields including the embedding vector.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        ...

    def bulk_import(
        self,
        records: Iterator[dict[str, Any]],
        batch_size: int = 1000,
        namespace_override: str | None = None,
    ) -> tuple[int, list[str]]:
        """Import memories from an iterator of records.

        Supports streaming import for large datasets. Accepts pre-parsed
        and pre-validated records; file parsing and validation are handled
        by the service layer.

        Args:
            records: Iterator of memory dictionaries. Each dictionary
                should contain required fields (content, vector, etc.).
            batch_size: Number of records per database insert batch.
            namespace_override: If provided, overrides the namespace
                field for all imported records.

        Returns:
            Tuple of (records_imported, list_of_new_ids) where:
                - records_imported: Total number of records successfully imported
                - list_of_new_ids: List of generated memory IDs

        Raises:
            ValidationError: If records contain invalid data.
            StorageError: If database operation fails.
        """
        ...


class EmbeddingServiceProtocol(Protocol):
    """Protocol for text embedding generation.

    Implementations can use local models (sentence-transformers)
    or API-based services (OpenAI).
    """

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...
