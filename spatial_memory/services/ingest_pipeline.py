"""Ingest pipeline for memory storage.

Extracts the dedup + quality gate + storage logic from MemoryService.remember()
into a dedicated pipeline class, following the Single Responsibility Principle.

The pipeline is NOT thread-safe — callers must provide synchronization
(e.g., MemoryService wraps calls with _remember_lock).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spatial_memory.core.hashing import compute_content_hash
from spatial_memory.core.models import Memory, MemorySource
from spatial_memory.core.quality_gate import score_memory_quality

# Default capacity for the in-memory hash LRU cache.
DEFAULT_HASH_CACHE_SIZE = 10_000

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryNamespaceProtocol,
        MemoryRepositoryProtocol,
    )


@dataclass
class IngestConfig:
    """Per-invocation configuration for the ingest pipeline.

    Passed on each ingest() call because server handler and queue processor
    use different threshold values.
    """

    cognitive_offloading_enabled: bool = False
    dedup_threshold: float = 0.85
    signal_threshold: float = 0.3


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


class IngestPipeline:
    """Pipeline for dedup checking, quality gating, and storing memories.

    Steps (when cognitive_offloading_enabled=True):
      1. Compute content hash
      2. Hash dedup check -> reject_exact if match
      3. Generate embedding
      4. Vector dedup check -> reject_similar or potential_duplicate
      5. Quality gate -> reject_quality if below threshold
      6. Importance adjustment (borderline quality)
      7. Construct Memory object
      8. Store via repository

    When disabled: skip steps 2, 4, 5, 6.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        hash_cache_size: int = DEFAULT_HASH_CACHE_SIZE,
    ) -> None:
        self._repo = repository
        self._embeddings = embeddings
        self._hash_cache_size = hash_cache_size
        # LRU cache of content hashes stored/seen by this pipeline instance.
        # Hash dedup only queries the DB when the hash is in this dict,
        # making the common case (new content) O(1) instead of an O(n)
        # full table scan.  Cross-session exact duplicates are still caught
        # by vector dedup (layer 2) at similarity ~1.0.
        # Uses OrderedDict as an LRU set (values are unused sentinels).
        self._stored_hashes: OrderedDict[str, None] = OrderedDict()

    def ingest(
        self,
        content: str,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        project: str = "",
        config: IngestConfig | None = None,
    ) -> RememberResult:
        """Run the full ingest pipeline: dedup, quality gate, store.

        Args:
            content: Text content of the memory.
            namespace: Namespace for organization.
            tags: Optional list of tags.
            importance: Importance score (0-1).
            metadata: Optional metadata dict.
            project: Project scope for the memory.
            config: Pipeline configuration for this invocation.

        Returns:
            RememberResult with the outcome.
        """
        if config is None:
            config = IngestConfig()

        # Always compute content hash (cheap, forward-compatible)
        content_hash = compute_content_hash(content)

        if config.cognitive_offloading_enabled:
            # Layer 1 dedup: O(1) in-memory hash check.
            # Only queries the DB if we previously stored this hash (same
            # pipeline instance).  New content skips the DB scan entirely.
            # Cross-session exact duplicates are caught by vector dedup
            # (layer 2) at similarity ~1.0.
            hash_dedup = self._check_hash_dedup(content_hash, namespace, project)

            if hash_dedup.status == "exact_duplicate" and hash_dedup.existing_memory:
                logger.debug("Exact duplicate detected for content hash %s...", content_hash[:12])
                return RememberResult(
                    id=hash_dedup.existing_memory.id,
                    content=content,
                    namespace=namespace,
                    deduplicated=True,
                    status="rejected_exact",
                    existing_memory_id=hash_dedup.existing_memory.id,
                    existing_memory_content=hash_dedup.existing_memory.content,
                    similarity=1.0,
                )

            # Generate embedding (skipped above for exact duplicates)
            vector = self._embeddings.embed(content)

            # Layer 2 dedup: vector similarity check
            vec_dedup = self._check_vector_dedup(vector, namespace, project, config.dedup_threshold)

            if vec_dedup.status == "likely_duplicate" and vec_dedup.existing_memory:
                logger.debug(
                    "Likely duplicate (similarity=%.3f) for memory %s",
                    vec_dedup.similarity,
                    vec_dedup.existing_memory.id,
                )
                return RememberResult(
                    id=vec_dedup.existing_memory.id,
                    content=content,
                    namespace=namespace,
                    deduplicated=True,
                    status="rejected_similar",
                    existing_memory_id=vec_dedup.existing_memory.id,
                    existing_memory_content=vec_dedup.existing_memory.content,
                    similarity=vec_dedup.similarity,
                )

            if vec_dedup.status == "potential_duplicate" and vec_dedup.existing_memory:
                logger.debug(
                    "Potential duplicate (similarity=%.3f) for memory %s",
                    vec_dedup.similarity,
                    vec_dedup.existing_memory.id,
                )
                return RememberResult(
                    id="",
                    content=content,
                    namespace=namespace,
                    deduplicated=False,
                    status="potential_duplicate",
                    existing_memory_id=vec_dedup.existing_memory.id,
                    existing_memory_content=vec_dedup.existing_memory.content,
                    similarity=vec_dedup.similarity,
                )

            # Quality gate (runs after dedup — no point scoring if duplicate)
            quality = score_memory_quality(content, tags, metadata)
            if quality.total < config.signal_threshold:
                logger.debug(
                    "Quality gate rejected: score=%.3f < threshold=%s",
                    quality.total,
                    config.signal_threshold,
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
        else:
            # No cognitive offloading — just embed
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
            project=project,
            content_hash=content_hash,
        )

        # Store in repository
        memory_id = self._repo.add(memory, vector)

        # Track hash so future exact duplicates are caught by layer 1
        self._add_to_hash_cache(content_hash)

        return RememberResult(
            id=memory_id,
            content=content,
            namespace=namespace,
            deduplicated=False,
        )

    def _check_hash_dedup(
        self,
        content_hash: str,
        namespace: str,
        project: str,
    ) -> DedupCheckResult:
        """Layer 1: Check for exact content hash match.

        Uses the in-memory ``_stored_hashes`` set to decide whether a DB
        lookup is worthwhile.  If the hash has never been stored by this
        pipeline instance, we skip the O(n) DB scan and return "new".

        Args:
            content_hash: SHA-256 of normalized content.
            namespace: Namespace to scope the check.
            project: Project to scope the check.

        Returns:
            DedupCheckResult with status "exact_duplicate" or "new".
        """
        if content_hash not in self._stored_hashes:
            return DedupCheckResult(status="new")

        existing = self._repo.find_by_content_hash(
            content_hash, namespace=namespace, project=project or None
        )
        if existing:
            return DedupCheckResult(
                status="exact_duplicate",
                existing_memory=existing,
                similarity=1.0,
            )
        return DedupCheckResult(status="new")

    def _check_vector_dedup(
        self,
        vector: Any,
        namespace: str,
        project: str,
        dedup_threshold: float,
    ) -> DedupCheckResult:
        """Layer 2: Check for near-duplicate via vector similarity.

        Called after embedding has been generated (needed for both dedup
        and eventual storage).

        Args:
            vector: Embedding vector for similarity search.
            namespace: Namespace to scope the check.
            project: Project to scope the check.
            dedup_threshold: Similarity threshold for likely-duplicate rejection.

        Returns:
            DedupCheckResult indicating duplicate status.
        """
        results = self._repo.search(vector, limit=1, namespace=namespace, project=project or None)
        if results:
            top = results[0]
            if top.similarity >= 0.80:
                # Build Memory from the search result directly — the vector search
                # already returned all the fields we need. This avoids an extra
                # repo.get() round trip (~2-5ms) per near-duplicate detection.
                existing = Memory.from_search_result(top)
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

    def _add_to_hash_cache(self, content_hash: str) -> None:
        """Add a hash to the LRU cache, evicting oldest if at capacity."""
        if content_hash in self._stored_hashes:
            self._stored_hashes.move_to_end(content_hash)
            return
        if len(self._stored_hashes) >= self._hash_cache_size:
            self._stored_hashes.popitem(last=False)
        self._stored_hashes[content_hash] = None

    @property
    def hash_cache_capacity(self) -> int:
        """Maximum number of hashes the dedup cache can hold."""
        return self._hash_cache_size

    def seed_hashes(self, hashes: list[str]) -> None:
        """Seed the hash cache with existing content hashes.

        Hashes are added in order, so the last entries are the most
        recently used.

        Args:
            hashes: List of content hash strings to seed.
        """
        for h in hashes:
            self._add_to_hash_cache(h)
        logger.info(
            "Seeded hash cache with %d entries (capacity %d)",
            len(self._stored_hashes),
            self._hash_cache_size,
        )

    def seed_from_repository(self, repository: MemoryNamespaceProtocol) -> None:
        """Seed the hash cache from the database for cross-session dedup.

        Called during startup to enable cross-session exact-duplicate
        detection via layer 1 (hash dedup).  Non-fatal: logs a warning
        and continues with an empty cache on failure.

        Args:
            repository: Repository implementing MemoryNamespaceProtocol.
        """
        try:
            hashes = repository.get_all_content_hashes(limit=self._hash_cache_size)
            if hashes:
                self.seed_hashes(hashes)
        except Exception as e:
            logger.warning("Failed to seed hash cache from repository: %s", e)
