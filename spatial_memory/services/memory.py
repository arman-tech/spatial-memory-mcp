"""Memory service for core operations.

This service provides the application layer for memory operations:
- remember: Store new memories
- recall: Search for similar memories
- nearby: Find neighbors of a memory
- forget: Delete memories

The service uses dependency injection for repository and embedding services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spatial_memory.core.errors import MemoryNotFoundError, ValidationError
from spatial_memory.core.models import Memory, MemorySource

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spatial_memory.core.models import MemoryResult
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


@dataclass
class RememberResult:
    """Result of storing a memory."""

    id: str
    content: str
    namespace: str


@dataclass
class RememberBatchResult:
    """Result of storing multiple memories."""

    ids: list[str]
    count: int


@dataclass
class RecallResult:
    """Result of recalling memories."""

    memories: list[MemoryResult]
    total: int


@dataclass
class NearbyResult:
    """Result of finding nearby memories."""

    reference: Memory
    neighbors: list[MemoryResult]


@dataclass
class ForgetResult:
    """Result of forgetting memories."""

    deleted: int
    ids: list[str] = field(default_factory=list)


class MemoryService:
    """Service for memory operations.

    Uses Clean Architecture - depends on protocol interfaces, not implementations.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> None:
        """Initialize the memory service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
        """
        self._repo = repository
        self._embeddings = embeddings

    def _validate_content(self, content: str) -> None:
        """Validate memory content.

        Args:
            content: Content to validate.

        Raises:
            ValidationError: If content is empty or whitespace-only.
        """
        if not content or not content.strip():
            raise ValidationError("Content cannot be empty")

    def _validate_importance(self, importance: float) -> None:
        """Validate importance score.

        Args:
            importance: Importance to validate.

        Raises:
            ValidationError: If importance is out of range.
        """
        if not 0.0 <= importance <= 1.0:
            raise ValidationError("Importance must be between 0.0 and 1.0")

    def remember(
        self,
        content: str,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> RememberResult:
        """Store a new memory.

        Args:
            content: Text content of the memory.
            namespace: Namespace for organization.
            tags: Optional list of tags.
            importance: Importance score (0-1).
            metadata: Optional metadata dict.

        Returns:
            RememberResult with the new memory's ID.

        Raises:
            ValidationError: If input validation fails.
        """
        # Validate inputs
        self._validate_content(content)
        self._validate_importance(importance)

        # Generate embedding
        vector = self._embeddings.embed(content)

        # Create memory object (ID will be assigned by repository)
        memory = Memory(
            id="",  # Will be replaced by repository
            content=content,
            namespace=namespace,
            tags=tags or [],
            importance=importance,
            source=MemorySource.MANUAL,
            metadata=metadata or {},
        )

        # Store in repository
        memory_id = self._repo.add(memory, vector)

        return RememberResult(
            id=memory_id,
            content=content,
            namespace=namespace,
        )

    def remember_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> RememberBatchResult:
        """Store multiple memories efficiently.

        Args:
            memories: List of dicts with content and optional fields.
                Each dict can have: content, namespace, tags, importance, metadata.

        Returns:
            RememberBatchResult with IDs and count.

        Raises:
            ValidationError: If input validation fails.
        """
        if not memories:
            raise ValidationError("Memory list cannot be empty")

        # Validate all memories first
        for mem_dict in memories:
            content = mem_dict.get("content", "")
            self._validate_content(content)
            importance = mem_dict.get("importance", 0.5)
            self._validate_importance(importance)

        # Extract content for batch embedding
        contents = [m["content"] for m in memories]
        vectors = self._embeddings.embed_batch(contents)

        # Create Memory objects
        memory_objects: list[Memory] = []
        for mem_dict in memories:
            memory = Memory(
                id="",  # Will be replaced by repository
                content=mem_dict["content"],
                namespace=mem_dict.get("namespace", "default"),
                tags=mem_dict.get("tags", []),
                importance=mem_dict.get("importance", 0.5),
                source=MemorySource.MANUAL,
                metadata=mem_dict.get("metadata", {}),
            )
            memory_objects.append(memory)

        # Store in repository
        ids = self._repo.add_batch(memory_objects, vectors)

        return RememberBatchResult(
            ids=ids,
            count=len(ids),
        )

    def recall(
        self,
        query: str,
        limit: int = 5,
        namespace: str | None = None,
        min_similarity: float = 0.0,
    ) -> RecallResult:
        """Search for similar memories.

        Args:
            query: Query text to search for.
            limit: Maximum number of results.
            namespace: Optional namespace filter.
            min_similarity: Minimum similarity threshold.

        Returns:
            RecallResult with matching memories.

        Raises:
            ValidationError: If input validation fails.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if limit < 1:
            raise ValidationError("Limit must be at least 1")

        # Generate query embedding
        vector = self._embeddings.embed(query)

        # Search repository
        results = self._repo.search(vector, limit=limit, namespace=namespace)

        # Filter by minimum similarity
        filtered_results = [r for r in results if r.similarity >= min_similarity]

        # Update access stats for returned memories (batch for efficiency)
        if filtered_results:
            memory_ids = [r.id for r in filtered_results]
            try:
                self._repo.update_access_batch(memory_ids)
            except Exception as e:
                # Log but don't fail the search if access update fails
                logger.warning(f"Failed to update access stats: {e}")

        return RecallResult(
            memories=filtered_results,
            total=len(filtered_results),
        )

    def nearby(
        self,
        memory_id: str,
        limit: int = 5,
        namespace: str | None = None,
    ) -> NearbyResult:
        """Find memories similar to a reference memory.

        Args:
            memory_id: ID of the reference memory.
            limit: Maximum number of neighbors.
            namespace: Optional namespace filter.

        Returns:
            NearbyResult with reference and neighbors.

        Raises:
            MemoryNotFoundError: If reference memory not found.
        """
        # Get reference memory with its vector
        result = self._repo.get_with_vector(memory_id)
        if result is None:
            raise MemoryNotFoundError(memory_id)

        reference_memory, reference_vector = result

        # Search for similar memories (request limit+1 to account for self)
        search_results = self._repo.search(
            reference_vector,
            limit=limit + 1,
            namespace=namespace,
        )

        # Filter out the reference memory itself
        neighbors = [r for r in search_results if r.id != memory_id]

        # Limit to requested count
        neighbors = neighbors[:limit]

        return NearbyResult(
            reference=reference_memory,
            neighbors=neighbors,
        )

    def forget(
        self,
        memory_id: str,
    ) -> ForgetResult:
        """Delete a memory.

        Args:
            memory_id: ID of memory to delete.

        Returns:
            ForgetResult with deletion count.
        """
        deleted = self._repo.delete(memory_id)

        return ForgetResult(
            deleted=1 if deleted else 0,
            ids=[memory_id] if deleted else [],
        )

    def forget_batch(
        self,
        memory_ids: list[str],
    ) -> ForgetResult:
        """Delete multiple memories.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            ForgetResult with deletion count.

        Raises:
            ValidationError: If input validation fails.
        """
        if not memory_ids:
            raise ValidationError("Memory ID list cannot be empty")

        deleted_count = self._repo.delete_batch(memory_ids)

        return ForgetResult(
            deleted=deleted_count,
            ids=memory_ids[:deleted_count] if deleted_count > 0 else [],
        )
