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
from typing import TYPE_CHECKING, Any, Protocol

from spatial_memory.core.errors import MemoryNotFoundError, ValidationError
from spatial_memory.core.hashing import compute_content_hash
from spatial_memory.core.models import Memory, MemorySource
from spatial_memory.core.quality_gate import score_memory_quality
from spatial_memory.core.validation import validate_content, validate_importance

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spatial_memory.core.db_idempotency import IdempotencyRecord
    from spatial_memory.core.models import MemoryResult
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


class IdempotencyProviderProtocol(Protocol):
    """Protocol for idempotency key storage and lookup.

    Implementations should handle key-to-memory-id mappings with TTL support.
    """

    def get_by_idempotency_key(self, key: str) -> IdempotencyRecord | None:
        """Look up an idempotency record by key.

        Args:
            key: The idempotency key to look up.

        Returns:
            IdempotencyRecord if found and not expired, None otherwise.
        """
        ...

    def store_idempotency_key(
        self,
        key: str,
        memory_id: str,
        ttl_hours: float = 24.0,
    ) -> None:
        """Store an idempotency key mapping.

        Args:
            key: The idempotency key.
            memory_id: The memory ID that was created.
            ttl_hours: Time-to-live in hours (default: 24 hours).
        """
        ...


@dataclass
class DedupCheckResult:
    """Result of deduplication check."""

    status: str  # "new", "exact_duplicate", "likely_duplicate", "potential_duplicate"
    existing_memory: Memory | None = None
    similarity: float = 0.0


@dataclass
class RememberResult:
    """Result of storing a memory."""

    id: str
    content: str
    namespace: str
    deduplicated: bool = False
    status: str = "stored"
    quality_score: float | None = None
    existing_memory_id: str | None = None
    existing_memory_content: str | None = None
    similarity: float | None = None


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
        idempotency_provider: IdempotencyProviderProtocol | None = None,
    ) -> None:
        """Initialize the memory service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
            idempotency_provider: Optional provider for idempotency key support.
        """
        self._repo = repository
        self._embeddings = embeddings
        self._idempotency = idempotency_provider

    # Use centralized validation functions
    _validate_content = staticmethod(validate_content)
    _validate_importance = staticmethod(validate_importance)

    def _check_dedup(
        self,
        content: str,
        content_hash: str,
        vector: Any,
        namespace: str,
        project: str,
        dedup_threshold: float,
    ) -> DedupCheckResult:
        """Check for duplicate memories using hash and vector similarity.

        Args:
            content: The memory content.
            content_hash: SHA-256 of normalized content.
            vector: Embedding vector for similarity search.
            namespace: Namespace to scope the check.
            project: Project to scope the check.
            dedup_threshold: Similarity threshold for likely-duplicate rejection.

        Returns:
            DedupCheckResult indicating duplicate status.
        """
        # Layer 1: Exact content hash match
        existing = self._repo.find_by_content_hash(
            content_hash, namespace=namespace, project=project or None
        )
        if existing:
            return DedupCheckResult(
                status="exact_duplicate",
                existing_memory=existing,
                similarity=1.0,
            )

        # Layer 2: Vector similarity check
        results = self._repo.search(vector, limit=1, namespace=namespace, project=project or None)
        if results:
            top = results[0]
            if top.similarity >= 0.80:
                # Build Memory from the search result directly — the vector search
                # already returned all the fields we need. This avoids an extra
                # repo.get() round trip (~2-5ms) per near-duplicate detection.
                existing = Memory(
                    id=top.id,
                    content=top.content,
                    namespace=top.namespace,
                    project=top.project,
                    tags=top.tags,
                    importance=top.importance,
                    created_at=top.created_at,
                    last_accessed=top.last_accessed or top.created_at,
                    access_count=top.access_count,
                    metadata=top.metadata,
                )
                if top.similarity >= dedup_threshold:
                    return DedupCheckResult(
                        status="likely_duplicate",
                        existing_memory=existing,
                        similarity=top.similarity,
                    )
                return DedupCheckResult(
                    status="potential_duplicate",
                    existing_memory=existing,
                    similarity=top.similarity,
                )

        return DedupCheckResult(status="new")

    def remember(
        self,
        content: str,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        project: str = "",
        cognitive_offloading_enabled: bool = False,
        signal_threshold: float = 0.3,
        dedup_threshold: float = 0.85,
    ) -> RememberResult:
        """Store a new memory.

        Args:
            content: Text content of the memory.
            namespace: Namespace for organization.
            tags: Optional list of tags.
            importance: Importance score (0-1).
            metadata: Optional metadata dict.
            idempotency_key: Optional key for idempotent requests. If provided
                and a memory was already created with this key, returns the
                existing memory ID with deduplicated=True.
            project: Project scope for the memory.
            cognitive_offloading_enabled: Whether to run dedup + quality gate.
            signal_threshold: Quality gate minimum score (reject below this).
            dedup_threshold: Vector similarity threshold for likely-duplicate rejection.

        Returns:
            RememberResult with the new memory's ID. If idempotency_key was
            provided and matched an existing request, deduplicated=True.

        Raises:
            ValidationError: If input validation fails.
        """
        # Check idempotency key first (before any expensive operations)
        if idempotency_key and self._idempotency:
            existing = self._idempotency.get_by_idempotency_key(idempotency_key)
            if existing:
                logger.debug(
                    "Idempotency key '%s' matched existing memory '%s'",
                    idempotency_key,
                    existing.memory_id,
                )
                # Return cached result - fetch the memory to get content
                cached_memory = self._repo.get(existing.memory_id)
                if cached_memory:
                    return RememberResult(
                        id=existing.memory_id,
                        content=cached_memory.content,
                        namespace=cached_memory.namespace,
                        deduplicated=True,
                    )
                # Memory was deleted but key exists - proceed with new insert
                logger.warning(
                    "Idempotency key '%s' references deleted memory '%s', creating new memory",
                    idempotency_key,
                    existing.memory_id,
                )

        # Validate inputs
        self._validate_content(content)
        self._validate_importance(importance)

        # Generate embedding
        vector = self._embeddings.embed(content)

        # Always compute content hash (cheap, forward-compatible)
        content_hash = compute_content_hash(content)

        if cognitive_offloading_enabled:
            # Dedup check
            dedup = self._check_dedup(
                content, content_hash, vector, namespace, project, dedup_threshold
            )

            if dedup.status == "exact_duplicate" and dedup.existing_memory:
                logger.debug("Exact duplicate detected for content hash %s...", content_hash[:12])
                return RememberResult(
                    id=dedup.existing_memory.id,
                    content=content,
                    namespace=namespace,
                    deduplicated=True,
                    status="rejected_exact",
                    existing_memory_id=dedup.existing_memory.id,
                    existing_memory_content=dedup.existing_memory.content,
                    similarity=1.0,
                )

            if dedup.status == "likely_duplicate" and dedup.existing_memory:
                logger.debug(
                    "Likely duplicate (similarity=%.3f) for memory %s",
                    dedup.similarity,
                    dedup.existing_memory.id,
                )
                return RememberResult(
                    id=dedup.existing_memory.id,
                    content=content,
                    namespace=namespace,
                    deduplicated=True,
                    status="rejected_similar",
                    existing_memory_id=dedup.existing_memory.id,
                    existing_memory_content=dedup.existing_memory.content,
                    similarity=dedup.similarity,
                )

            if dedup.status == "potential_duplicate" and dedup.existing_memory:
                logger.debug(
                    "Potential duplicate (similarity=%.3f) for memory %s",
                    dedup.similarity,
                    dedup.existing_memory.id,
                )
                return RememberResult(
                    id="",
                    content=content,
                    namespace=namespace,
                    deduplicated=False,
                    status="potential_duplicate",
                    existing_memory_id=dedup.existing_memory.id,
                    existing_memory_content=dedup.existing_memory.content,
                    similarity=dedup.similarity,
                )

            # Quality gate (runs after dedup — no point scoring if duplicate)
            quality = score_memory_quality(content, tags, metadata)
            if quality.total < signal_threshold:
                logger.debug(
                    "Quality gate rejected: score=%.3f < threshold=%s",
                    quality.total,
                    signal_threshold,
                )
                return RememberResult(
                    id="",
                    content=content,
                    namespace=namespace,
                    status="rejected_quality",
                    quality_score=quality.total,
                )
            if quality.total < 0.5:
                # Store with reduced importance for borderline quality
                importance = min(importance, quality.total)

        # Create memory object (ID will be assigned by repository)
        memory = Memory(
            id="",  # Will be replaced by repository
            content=content,
            namespace=namespace,
            tags=tags or [],
            importance=importance,
            source=MemorySource.MANUAL,
            metadata=metadata or {},
            project=project,
            content_hash=content_hash,
        )

        # Store in repository
        memory_id = self._repo.add(memory, vector)

        # Store idempotency key mapping if provided
        if idempotency_key and self._idempotency:
            try:
                self._idempotency.store_idempotency_key(idempotency_key, memory_id)
            except Exception as e:
                # Log but don't fail the memory creation
                logger.warning("Failed to store idempotency key '%s': %s", idempotency_key, e)

        return RememberResult(
            id=memory_id,
            content=content,
            namespace=namespace,
            deduplicated=False,
        )

    def remember_batch(
        self,
        memories: list[dict[str, Any]],
        project: str = "",
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
                project=project,
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
        project: str | None = None,
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
        results = self._repo.search(vector, limit=limit, namespace=namespace, project=project)

        # Filter by minimum similarity
        filtered_results = [r for r in results if r.similarity >= min_similarity]

        # Update access stats for returned memories (batch for efficiency)
        if filtered_results:
            memory_ids = [r.id for r in filtered_results]
            try:
                self._repo.update_access_batch(memory_ids)
            except Exception as e:
                # Log but don't fail the search if access update fails
                logger.warning("Failed to update access stats: %s", e)

        return RecallResult(
            memories=filtered_results,
            total=len(filtered_results),
        )

    def nearby(
        self,
        memory_id: str,
        limit: int = 5,
        namespace: str | None = None,
        project: str | None = None,
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
            project=project,
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

        deleted_count, deleted_ids = self._repo.delete_batch(memory_ids)

        return ForgetResult(
            deleted=deleted_count,
            ids=deleted_ids,
        )
