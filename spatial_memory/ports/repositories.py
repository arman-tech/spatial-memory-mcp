"""Protocol interfaces for repository and embedding services.

These protocols define the contracts between the service layer and infrastructure.
Using typing.Protocol enables structural subtyping (duck typing with type checking).
"""

from __future__ import annotations

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
