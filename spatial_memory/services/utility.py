"""Utility service for database management operations.

This service provides the application layer for utility operations:
- stats: Get database statistics and health metrics
- namespaces: List namespaces with memory counts
- delete_namespace: Delete all memories in a namespace
- rename_namespace: Rename namespace (move all memories)
- hybrid_recall: Combined vector + FTS search

The service uses dependency injection for repository and embedding services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spatial_memory.core.errors import (
    NamespaceNotFoundError,
    NamespaceOperationError,
    ValidationError,
)
from spatial_memory.core.models import (
    DeleteNamespaceResult,
    HybridMemoryMatch,
    HybridRecallResult,
    IndexInfo,
    NamespaceInfo,
    NamespacesResult,
    RenameNamespaceResult,
    StatsResult,
    UtilityConfig,
)
from spatial_memory.core.validation import validate_namespace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


class UtilityService:
    """Service for database utility operations.

    Uses Clean Architecture - depends on protocol interfaces, not implementations.
    Provides database statistics, namespace management, and hybrid search.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: UtilityConfig | None = None,
    ) -> None:
        """Initialize the utility service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
            config: Optional configuration (uses defaults if not provided).
        """
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or UtilityConfig()

    def stats(
        self,
        namespace: str | None = None,
        include_index_details: bool = True,
    ) -> StatsResult:
        """Get comprehensive database statistics.

        Retrieves statistics about the memory database including total counts,
        storage size, index information, and health metrics.

        Args:
            namespace: Filter statistics to a specific namespace.
                If None, returns statistics for all namespaces.
            include_index_details: Include detailed index statistics.

        Returns:
            StatsResult with database information.

        Raises:
            ValidationError: If namespace is invalid.
            NamespaceOperationError: If stats retrieval fails.
        """
        # Validate namespace if provided
        if namespace is not None:
            namespace = validate_namespace(namespace)

        try:
            # Get stats from repository
            raw_stats = self._repo.get_stats(namespace=namespace)

            # Transform indices to IndexInfo objects
            indices: list[IndexInfo] = []
            if include_index_details:
                for idx_data in raw_stats.get("indices", []):
                    indices.append(
                        IndexInfo(
                            name=idx_data.get("name", "unknown"),
                            index_type=idx_data.get("index_type", "unknown"),
                            column=idx_data.get("column", "unknown"),
                            num_indexed_rows=idx_data.get("num_indexed_rows", 0),
                            status=idx_data.get("status", "unknown"),
                        )
                    )

            # Get namespace breakdown from raw_stats
            memories_by_namespace: dict[str, int] = raw_stats.get("namespaces", {})

            # Calculate estimated vector bytes (dims * 4 bytes per float * num memories)
            total_memories = raw_stats.get("total_memories", 0)
            embedding_dims = self._embeddings.dimensions
            estimated_vector_bytes = total_memories * embedding_dims * 4

            return StatsResult(
                total_memories=total_memories,
                memories_by_namespace=memories_by_namespace,
                storage_bytes=raw_stats.get("storage_bytes", 0),
                storage_mb=raw_stats.get("storage_mb", 0.0),
                estimated_vector_bytes=estimated_vector_bytes,
                has_vector_index=raw_stats.get("has_vector_index", False),
                has_fts_index=raw_stats.get("has_fts_index", False),
                indices=indices,
                num_fragments=raw_stats.get("num_fragments", 0),
                needs_compaction=raw_stats.get("needs_compaction", False),
                table_version=raw_stats.get("table_version", 1),
                oldest_memory_date=raw_stats.get("oldest_memory_date"),
                newest_memory_date=raw_stats.get("newest_memory_date"),
                avg_content_length=raw_stats.get("avg_content_length"),
            )

        except ValidationError:
            raise
        except Exception as e:
            raise NamespaceOperationError(f"Failed to get stats: {e}") from e

    def namespaces(
        self,
        include_stats: bool = True,
    ) -> NamespacesResult:
        """List all namespaces with optional statistics.

        Args:
            include_stats: Include memory counts and date ranges per namespace.

        Returns:
            NamespacesResult with namespace list and totals.

        Raises:
            NamespaceOperationError: If namespace listing fails.
        """
        try:
            # Get list of namespaces from repository
            namespace_names = self._repo.get_namespaces()

            namespaces: list[NamespaceInfo] = []
            total_memories = 0

            for ns_name in namespace_names:
                if include_stats:
                    # Get detailed stats for each namespace
                    try:
                        ns_stats = self._repo.get_namespace_stats(ns_name)
                        memory_count = ns_stats.get("memory_count", 0)
                        namespaces.append(
                            NamespaceInfo(
                                name=ns_name,
                                memory_count=memory_count,
                                oldest_memory=ns_stats.get("oldest_memory"),
                                newest_memory=ns_stats.get("newest_memory"),
                            )
                        )
                        total_memories += memory_count
                    except Exception as e:
                        logger.warning(f"Failed to get stats for namespace {ns_name}: {e}")
                        # Still include namespace with zero count
                        namespaces.append(
                            NamespaceInfo(
                                name=ns_name,
                                memory_count=0,
                            )
                        )
                else:
                    # Just include name without stats
                    namespaces.append(
                        NamespaceInfo(
                            name=ns_name,
                            memory_count=0,
                        )
                    )

            # If we didn't include stats, get total from count
            if not include_stats:
                total_memories = self._repo.count()

            return NamespacesResult(
                namespaces=namespaces,
                total_namespaces=len(namespaces),
                total_memories=total_memories,
            )

        except Exception as e:
            raise NamespaceOperationError(f"Failed to list namespaces: {e}") from e

    def delete_namespace(
        self,
        namespace: str,
        confirm: bool = False,
        dry_run: bool = True,
    ) -> DeleteNamespaceResult:
        """Delete all memories in a namespace.

        DESTRUCTIVE OPERATION - requires explicit confirmation unless dry_run.

        Args:
            namespace: Namespace to delete.
            confirm: Set to True to confirm deletion (required for non-dry-run).
            dry_run: Preview deletion without executing (default True).

        Returns:
            DeleteNamespaceResult with deletion details.

        Raises:
            ValidationError: If namespace is invalid or confirmation missing.
            NamespaceOperationError: If deletion fails.
        """
        # Validate namespace
        namespace = validate_namespace(namespace)

        try:
            # Get count of memories that would be deleted
            memory_count = self._repo.count(namespace=namespace)

            # If dry run, just return preview
            if dry_run:
                return DeleteNamespaceResult(
                    namespace=namespace,
                    memories_deleted=memory_count,
                    success=True,
                    message=f"DRY RUN: Would delete {memory_count} memories from namespace '{namespace}'",
                    dry_run=True,
                )

            # Not a dry run - require confirmation
            if not confirm:
                raise ValidationError(
                    f"Deletion of {memory_count} memories in namespace '{namespace}' "
                    "requires confirm=True. Use dry_run=True to preview first."
                )

            # Perform actual deletion
            deleted_count = self._repo.delete_by_namespace(namespace)

            return DeleteNamespaceResult(
                namespace=namespace,
                memories_deleted=deleted_count,
                success=True,
                message=f"Successfully deleted {deleted_count} memories from namespace '{namespace}'",
                dry_run=False,
            )

        except ValidationError:
            raise
        except Exception as e:
            raise NamespaceOperationError(
                f"Failed to delete namespace '{namespace}': {e}"
            ) from e

    def rename_namespace(
        self,
        old_namespace: str,
        new_namespace: str,
    ) -> RenameNamespaceResult:
        """Rename all memories from one namespace to another.

        Atomically updates the namespace field for all memories belonging
        to the source namespace.

        Args:
            old_namespace: The current namespace name (source).
            new_namespace: The new namespace name (target).

        Returns:
            RenameNamespaceResult with rename details.

        Raises:
            ValidationError: If namespace names are invalid or the same.
            NamespaceNotFoundError: If old_namespace doesn't exist.
            NamespaceOperationError: If rename fails.
        """
        # Validate both namespaces
        old_namespace = validate_namespace(old_namespace)
        new_namespace = validate_namespace(new_namespace)

        # Check they are different
        if old_namespace == new_namespace:
            raise ValidationError(
                f"Cannot rename namespace to same name: '{old_namespace}'"
            )

        try:
            # Call repository to perform rename
            renamed_count = self._repo.rename_namespace(old_namespace, new_namespace)

            return RenameNamespaceResult(
                old_namespace=old_namespace,
                new_namespace=new_namespace,
                memories_renamed=renamed_count,
                success=True,
                message=f"Successfully renamed {renamed_count} memories from '{old_namespace}' to '{new_namespace}'",
            )

        except NamespaceNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            raise NamespaceOperationError(
                f"Failed to rename namespace '{old_namespace}' to '{new_namespace}': {e}"
            ) from e

    def hybrid_recall(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 5,
        namespace: str | None = None,
        min_similarity: float = 0.0,
    ) -> HybridRecallResult:
        """Search using combined vector similarity and keyword matching.

        Performs hybrid search combining semantic similarity (vector search)
        and keyword matching (full-text search). Alpha parameter controls
        the balance: 1.0 = pure vector, 0.0 = pure keyword, 0.5 = balanced.

        Args:
            query: Search query text.
            alpha: Balance between vector (1.0) and keyword (0.0) search.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            HybridRecallResult with matching memories.

        Raises:
            ValidationError: If input validation fails.
        """
        # Validate query
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        # Validate alpha
        if alpha < self._config.hybrid_min_alpha or alpha > self._config.hybrid_max_alpha:
            raise ValidationError(
                f"Alpha must be between {self._config.hybrid_min_alpha} "
                f"and {self._config.hybrid_max_alpha}, got {alpha}"
            )

        # Validate namespace if provided
        if namespace is not None:
            namespace = validate_namespace(namespace)

        # Generate query embedding
        query_vector = self._embeddings.embed(query)

        # Perform hybrid search
        results = self._repo.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            limit=limit,
            namespace=namespace,
            alpha=alpha,
        )

        # Transform results to HybridMemoryMatch and filter by min_similarity
        memories: list[HybridMemoryMatch] = []
        for result in results:
            if result.similarity >= min_similarity:
                memories.append(
                    HybridMemoryMatch(
                        id=result.id,
                        content=result.content,
                        similarity=result.similarity,
                        namespace=result.namespace,
                        tags=list(result.tags),
                        importance=result.importance,
                        created_at=result.created_at,
                        metadata=dict(result.metadata),
                        # These may be populated if repository provides them
                        vector_score=getattr(result, "vector_score", None),
                        fts_score=getattr(result, "fts_score", None),
                        combined_score=result.similarity,
                    )
                )

        return HybridRecallResult(
            query=query,
            alpha=alpha,
            memories=memories,
            total=len(memories),
            search_type="hybrid",
        )
