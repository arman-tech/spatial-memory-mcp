"""Statistics and health monitoring for LanceDB database.

Provides database statistics, namespace statistics, health metrics,
and the HealthMetrics/IndexStats dataclasses used across the codebase.

This module is part of the database.py refactoring to separate concerns:
- StatsManager handles all statistics and health-related operations
- Database class delegates to StatsManager for these operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from spatial_memory.core.db_indexes import _get_index_attr
from spatial_memory.core.errors import StorageError, ValidationError
from spatial_memory.core.validation import (
    sanitize_string as _sanitize_string,
)
from spatial_memory.core.validation import (
    validate_namespace as _validate_namespace,
)

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


# ============================================================================
# Health Metrics Dataclasses
# ============================================================================


@dataclass
class IndexStats:
    """Statistics for a single index."""

    name: str
    index_type: str
    num_indexed_rows: int
    num_unindexed_rows: int
    needs_update: bool


@dataclass
class HealthMetrics:
    """Database health and performance metrics."""

    total_rows: int
    total_bytes: int
    total_bytes_mb: float
    num_fragments: int
    num_small_fragments: int
    needs_compaction: bool
    has_vector_index: bool
    has_fts_index: bool
    indices: list[IndexStats]
    version: int
    error: str | None = None


class StatsManagerProtocol(Protocol):
    """Protocol defining what StatsManager needs from Database.

    This protocol enables loose coupling between StatsManager and Database,
    preventing circular imports while maintaining type safety.
    """

    @property
    def table(self) -> LanceTable:
        """Access to the LanceDB table."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of embedding vectors."""
        ...

    _has_vector_index: bool | None
    _has_fts_index: bool | None


class StatsManager:
    """Manages statistics and health monitoring operations.

    Handles database statistics, namespace statistics, health metrics,
    and table-level stats retrieval.

    Example:
        manager = StatsManager(database)
        stats = manager.get_stats()
        health = manager.get_health_metrics()
    """

    def __init__(self, db: StatsManagerProtocol) -> None:
        """Initialize the stats manager.

        Args:
            db: Database instance (satisfies StatsManagerProtocol).
        """
        self._db = db

    def _get_table_stats(self) -> dict[str, Any]:
        """Get table statistics with best-effort fragment info."""
        try:
            count = self._db.table.count_rows()
            stats: dict[str, Any] = {
                "num_rows": count,
                "num_fragments": 0,
                "num_small_fragments": 0,
            }

            # Try to get fragment stats from table.stats() if available
            try:
                if hasattr(self._db.table, "stats"):
                    table_stats = self._db.table.stats()
                    if isinstance(table_stats, dict):
                        stats["num_fragments"] = table_stats.get("num_fragments", 0)
                        stats["num_small_fragments"] = table_stats.get("num_small_fragments", 0)
                    elif hasattr(table_stats, "num_fragments"):
                        stats["num_fragments"] = table_stats.num_fragments
                        stats["num_small_fragments"] = getattr(
                            table_stats, "num_small_fragments", 0
                        )
            except Exception as e:
                logger.debug(f"Could not get fragment stats: {e}")

            return stats
        except Exception as e:
            logger.warning(f"Could not get table stats: {e}")
            return {}

    def get_health_metrics(self) -> HealthMetrics:
        """Get comprehensive health and performance metrics.

        Returns:
            HealthMetrics dataclass with all metrics.
        """
        try:
            count = self._db.table.count_rows()

            # Estimate size (rough approximation)
            # vector (dim * 4 bytes) + avg content size estimate
            estimated_bytes = count * (self._db.embedding_dim * 4 + 1000)

            # Check indexes
            indices: list[IndexStats] = []
            try:
                for idx in self._db.table.list_indices():
                    indices.append(
                        IndexStats(
                            name=str(_get_index_attr(idx, "name", "unknown")),
                            index_type=str(_get_index_attr(idx, "index_type", "unknown")),
                            num_indexed_rows=count,  # Approximate
                            num_unindexed_rows=0,
                            needs_update=False,
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")

            return HealthMetrics(
                total_rows=count,
                total_bytes=estimated_bytes,
                total_bytes_mb=estimated_bytes / (1024 * 1024),
                num_fragments=0,
                num_small_fragments=0,
                needs_compaction=False,
                has_vector_index=self._db._has_vector_index or False,
                has_fts_index=self._db._has_fts_index or False,
                indices=indices,
                version=0,
            )

        except Exception as e:
            return HealthMetrics(
                total_rows=0,
                total_bytes=0,
                total_bytes_mb=0,
                num_fragments=0,
                num_small_fragments=0,
                needs_compaction=False,
                has_vector_index=False,
                has_fts_index=False,
                indices=[],
                version=0,
                error=str(e),
            )

    def get_stats(self, namespace: str | None = None, project: str | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Uses efficient LanceDB queries for aggregations.

        Args:
            namespace: Filter stats to specific namespace (None = all).
            project: Filter stats to specific project (None = all).

        Returns:
            Dictionary with statistics including:
                - total_memories: Total count of memories
                - namespaces: Dict mapping namespace to count
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
        try:
            metrics = self.get_health_metrics()

            # Get memory counts by namespace using efficient Arrow aggregation
            # Use pure Arrow operations (no pandas dependency)
            search = self._db.table.search().select(["namespace"])
            if project:
                safe_proj = _sanitize_string(project)
                search = search.where(f"project = '{safe_proj}'")
            ns_arrow = search.to_arrow()

            # Count by namespace using Arrow's to_pylist()
            ns_counts: dict[str, int] = {}
            for record in ns_arrow.to_pylist():
                ns = record["namespace"]
                ns_counts[ns] = ns_counts.get(ns, 0) + 1

            # Filter if namespace specified
            if namespace:
                namespace = _validate_namespace(namespace)
                if namespace in ns_counts:
                    ns_counts = {namespace: ns_counts[namespace]}
                else:
                    ns_counts = {}

            total = sum(ns_counts.values()) if ns_counts else 0

            return {
                "total_memories": total if namespace else metrics.total_rows,
                "namespaces": ns_counts,
                "storage_bytes": metrics.total_bytes,
                "storage_mb": metrics.total_bytes_mb,
                "num_fragments": metrics.num_fragments,
                "needs_compaction": metrics.needs_compaction,
                "has_vector_index": metrics.has_vector_index,
                "has_fts_index": metrics.has_fts_index,
                "table_version": metrics.version,
                "indices": [
                    {
                        "name": idx.name,
                        "index_type": idx.index_type,
                        "num_indexed_rows": idx.num_indexed_rows,
                        "status": "ready" if not idx.needs_update else "needs_update",
                    }
                    for idx in metrics.indices
                ],
            }
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get stats: {e}") from e

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        """Get statistics for a specific namespace.

        Args:
            namespace: The namespace to get statistics for.

        Returns:
            Dictionary containing:
                - namespace: The namespace name
                - memory_count: Number of memories in namespace
                - oldest_memory: Datetime of oldest memory (or None)
                - newest_memory: Datetime of newest memory (or None)
                - avg_content_length: Average content length (or None if empty)

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        namespace = _validate_namespace(namespace)
        safe_ns = _sanitize_string(namespace)

        try:
            # Get count efficiently
            filter_expr = f"namespace = '{safe_ns}'"
            count_results = (
                self._db.table.search()
                .where(filter_expr)
                .select(["id"])
                .limit(1000000)  # High limit to count all
                .to_list()
            )
            memory_count = len(count_results)

            if memory_count == 0:
                return {
                    "namespace": namespace,
                    "memory_count": 0,
                    "oldest_memory": None,
                    "newest_memory": None,
                    "avg_content_length": None,
                }

            # Get oldest memory (sort ascending, limit 1)
            oldest_records = (
                self._db.table.search().where(filter_expr).select(["created_at"]).limit(1).to_list()
            )
            oldest = oldest_records[0]["created_at"] if oldest_records else None

            # Get newest memory - need to fetch more and find max since LanceDB
            # doesn't support ORDER BY DESC efficiently
            # Sample up to 1000 records for stats to avoid loading everything
            sample_size = min(memory_count, 1000)
            sample_records = (
                self._db.table.search()
                .where(filter_expr)
                .select(["created_at", "content"])
                .limit(sample_size)
                .to_list()
            )

            # Find newest from sample (for large namespaces this is approximate)
            if sample_records:
                created_times = [r["created_at"] for r in sample_records]
                newest = max(created_times)
                # Calculate average content length from sample
                content_lengths = [len(r.get("content", "")) for r in sample_records]
                avg_content_length = sum(content_lengths) / len(content_lengths)
            else:
                newest = oldest
                avg_content_length = None

            return {
                "namespace": namespace,
                "memory_count": memory_count,
                "oldest_memory": oldest,
                "newest_memory": newest,
                "avg_content_length": avg_content_length,
            }

        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get namespace stats: {e}") from e
