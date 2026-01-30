"""Protocol interfaces for repository and embedding services.

These protocols define the contracts between the service layer and infrastructure.
Using typing.Protocol enables structural subtyping (duck typing with type checking).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

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
    ) -> list[list[dict[str, Any]]]:
        """Search for memories near multiple query points.

        Efficient for operations like journey interpolation where multiple
        points need to find nearby memories. Supports parallel execution.

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
