"""Namespace and data retrieval operations for LanceDB database.

Provides namespace management, counting, and bulk data retrieval operations.

This module is part of the database.py refactoring to separate concerns:
- NamespaceManager handles namespace operations and read-only data retrieval
- Database class delegates to NamespaceManager for these operations
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from spatial_memory.core.errors import StorageError, ValidationError
from spatial_memory.core.utils import utc_now
from spatial_memory.core.validation import (
    sanitize_string as _sanitize_string,
)
from spatial_memory.core.validation import (
    validate_namespace as _validate_namespace,
)

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


class NamespaceManagerProtocol(Protocol):
    """Protocol defining what NamespaceManager needs from Database.

    This protocol enables loose coupling between NamespaceManager and Database,
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

    # Namespace cache fields
    _cached_namespaces: set[str] | None
    _namespace_cache_time: float
    _namespace_cache_lock: threading.Lock
    _NAMESPACE_CACHE_TTL: float

    def _invalidate_namespace_cache(self) -> None:
        """Invalidate the namespace cache after modifications."""
        ...

    def _invalidate_count_cache(self) -> None:
        """Invalidate the row count cache after modifications."""
        ...

    def _track_modification(self, count: int = 1) -> None:
        """Track database modifications for auto-compaction."""
        ...


class NamespaceManager:
    """Manages namespace operations and read-only data retrieval.

    Handles namespace renaming, counting, data retrieval, and
    vector fetching for clustering operations.

    Example:
        manager = NamespaceManager(database)
        namespaces = manager.get_namespaces()
        count = manager.count(namespace="default")
    """

    def __init__(self, db: NamespaceManagerProtocol) -> None:
        """Initialize the namespace manager.

        Args:
            db: Database instance (satisfies NamespaceManagerProtocol).
        """
        self._db = db

    def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
        """Rename all memories from one namespace to another.

        Uses atomic batch update via merge_insert for data integrity.
        On partial failure, attempts to rollback renamed records to original namespace.

        Args:
            old_namespace: Source namespace name.
            new_namespace: Target namespace name.

        Returns:
            Number of memories renamed.

        Raises:
            ValidationError: If namespace names are invalid.
            NamespaceNotFoundError: If old_namespace doesn't exist.
            StorageError: If database operation fails.
        """
        from spatial_memory.core.errors import NamespaceNotFoundError

        old_namespace = _validate_namespace(old_namespace)
        new_namespace = _validate_namespace(new_namespace)
        safe_old = _sanitize_string(old_namespace)
        _sanitize_string(new_namespace)  # Validate but don't store unused result

        try:
            # Check if source namespace exists using direct count query
            # (avoids namespace cache which may be stale during mutations)
            ns_count: int = self._db.table.count_rows(f"namespace = '{safe_old}'")
            if ns_count == 0:
                raise NamespaceNotFoundError(old_namespace)

            # Short-circuit if renaming to same namespace (no-op)
            if old_namespace == new_namespace:
                logger.debug(f"Namespace '{old_namespace}' renamed to itself ({ns_count} records)")
                return ns_count

            # Track renamed IDs for rollback capability
            renamed_ids: list[str] = []

            # Fetch all records in batches with iteration safeguards
            batch_size = 1000
            max_iterations = 10000  # Safety cap: 10M records at 1000/batch
            updated = 0
            iteration = 0
            previous_updated = 0

            while True:
                iteration += 1

                # Safety limit to prevent infinite loops
                if iteration > max_iterations:
                    raise StorageError(
                        f"rename_namespace exceeded maximum iterations ({max_iterations}). "
                        f"Updated {updated} records before stopping. "
                        "This may indicate a database consistency issue."
                    )

                records = (
                    self._db.table.search()
                    .where(f"namespace = '{safe_old}'")
                    .limit(batch_size)
                    .to_list()
                )

                if not records:
                    break

                # Track IDs in this batch for potential rollback
                batch_ids = [r["id"] for r in records]

                # Update namespace field
                for r in records:
                    r["namespace"] = new_namespace
                    r["updated_at"] = utc_now()
                    if isinstance(r.get("metadata"), dict):
                        r["metadata"] = json.dumps(r["metadata"])
                    if isinstance(r.get("vector"), np.ndarray):
                        r["vector"] = r["vector"].tolist()

                try:
                    # Atomic upsert
                    (
                        self._db.table.merge_insert("id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .execute(records)
                    )
                    # Only track as renamed after successful update
                    renamed_ids.extend(batch_ids)
                except Exception as batch_error:
                    # Batch failed - attempt rollback of previously renamed records
                    if renamed_ids:
                        logger.warning(
                            f"Batch {iteration} failed, attempting rollback of "
                            f"{len(renamed_ids)} previously renamed records"
                        )
                        rollback_error = self._rollback_namespace_rename(renamed_ids, old_namespace)
                        if rollback_error:
                            raise StorageError(
                                f"Namespace rename failed at batch {iteration} and "
                                f"rollback also failed. {len(renamed_ids)} records may be "
                                f"in inconsistent state (partially in '{new_namespace}'). "
                                f"Original error: {batch_error}. "
                                f"Rollback error: {rollback_error}"
                            ) from batch_error
                        else:
                            logger.info(
                                f"Rollback successful, reverted {len(renamed_ids)} records "
                                f"back to namespace '{old_namespace}'"
                            )
                    raise StorageError(
                        f"Failed to rename namespace (rolled back): {batch_error}"
                    ) from batch_error

                updated += len(records)

                # Detect stalled progress (same batch being processed repeatedly)
                if updated == previous_updated:
                    raise StorageError(
                        f"rename_namespace stalled at {updated} records. "
                        "merge_insert may have failed silently."
                    )
                previous_updated = updated

            self._db._invalidate_namespace_cache()
            logger.debug(f"Renamed {updated} memories from '{old_namespace}' to '{new_namespace}'")
            return updated

        except (ValidationError, NamespaceNotFoundError):
            raise
        except Exception as e:
            raise StorageError(f"Failed to rename namespace: {e}") from e

    def _rollback_namespace_rename(
        self, memory_ids: list[str], target_namespace: str
    ) -> Exception | None:
        """Attempt to revert renamed records back to original namespace.

        Args:
            memory_ids: List of memory IDs to revert.
            target_namespace: Namespace to revert records to.

        Returns:
            None if rollback succeeded, Exception if it failed.
        """
        try:
            if not memory_ids:
                return None

            _sanitize_string(target_namespace)  # Validate namespace
            now = utc_now()

            # Process in batches for large rollbacks
            batch_size = 1000
            for i in range(0, len(memory_ids), batch_size):
                batch_ids = memory_ids[i : i + batch_size]
                id_list = ", ".join(f"'{_sanitize_string(mid)}'" for mid in batch_ids)

                # Fetch records that need rollback
                records = self._db.table.search().where(f"id IN ({id_list})").to_list()

                if not records:
                    continue

                # Revert namespace
                for r in records:
                    r["namespace"] = target_namespace
                    r["updated_at"] = now
                    if isinstance(r.get("metadata"), dict):
                        r["metadata"] = json.dumps(r["metadata"])
                    if isinstance(r.get("vector"), np.ndarray):
                        r["vector"] = r["vector"].tolist()

                # Atomic upsert to restore original namespace
                (
                    self._db.table.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(records)
                )

            self._db._invalidate_namespace_cache()
            logger.debug(f"Rolled back {len(memory_ids)} records to namespace '{target_namespace}'")
            return None

        except Exception as e:
            logger.error(f"Namespace rename rollback failed: {e}")
            return e

    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces (cached with TTL, thread-safe).

        Uses double-checked locking to avoid race conditions where another
        thread could see stale data between cache check and update.

        Returns:
            Sorted list of namespace names.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            now = time.time()

            # First check with lock (quick path if cache is valid)
            with self._db._namespace_cache_lock:
                if (
                    self._db._cached_namespaces is not None
                    and (now - self._db._namespace_cache_time) <= self._db._NAMESPACE_CACHE_TTL
                ):
                    return sorted(self._db._cached_namespaces)

            # Fetch from database (outside lock to avoid blocking)
            results = self._db.table.search().select(["namespace"]).to_list()
            namespaces = set(r["namespace"] for r in results)

            # Double-checked locking: re-check and update atomically
            with self._db._namespace_cache_lock:
                # Another thread may have populated cache while we were fetching
                if self._db._cached_namespaces is None:
                    self._db._cached_namespaces = namespaces
                    self._db._namespace_cache_time = now
                # Return fresh data regardless (it's at least as current)
                return sorted(namespaces)

        except Exception as e:
            raise StorageError(f"Failed to get namespaces: {e}") from e

    def count(self, namespace: str | None = None, project: str | None = None) -> int:
        """Count memories.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.

        Returns:
            Number of memories.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            filters: list[str] = []
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                filters.append(f"namespace = '{safe_ns}'")
            if project:
                safe_proj = _sanitize_string(project)
                filters.append(f"project = '{safe_proj}'")
            if filters:
                count: int = self._db.table.count_rows(" AND ".join(filters))
                return count
            count = self._db.table.count_rows()
            return count
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to count memories: {e}") from e

    def get_all(
        self,
        namespace: str | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories, optionally filtered by namespace and project.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.
            limit: Maximum number of results.

        Returns:
            List of memory records.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            search = self._db.table.search()

            filters: list[str] = []
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                filters.append(f"namespace = '{safe_ns}'")
            if project:
                safe_proj = _sanitize_string(project)
                filters.append(f"project = '{safe_proj}'")
            if filters:
                search = search.where(" AND ".join(filters))

            if limit:
                search = search.limit(limit)

            results: list[dict[str, Any]] = search.to_list()

            for record in results:
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}

            return results
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get all memories: {e}") from e

    def get_vectors_for_clustering(
        self,
        namespace: str | None = None,
        project: str | None = None,
        max_memories: int = 10_000,
    ) -> tuple[list[str], np.ndarray]:
        """Fetch all vectors for clustering operations (e.g., HDBSCAN).

        Optimized for memory efficiency with large datasets.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.
            max_memories: Maximum memories to fetch.

        Returns:
            Tuple of (memory_ids, vectors_array).

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            # Build query selecting only needed columns
            search = self._db.table.search()

            filters: list[str] = []
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                filters.append(f"namespace = '{safe_ns}'")
            if project:
                safe_proj = _sanitize_string(project)
                filters.append(f"project = '{safe_proj}'")
            if filters:
                search = search.where(" AND ".join(filters))

            # Select only id and vector to minimize memory usage
            search = search.select(["id", "vector"]).limit(max_memories)

            results = search.to_list()

            if not results:
                return [], np.array([], dtype=np.float32).reshape(0, self._db.embedding_dim)

            ids = [r["id"] for r in results]
            vectors = np.array([r["vector"] for r in results], dtype=np.float32)

            return ids, vectors

        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to fetch vectors for clustering: {e}") from e

    def get_vectors_as_arrow(
        self,
        namespace: str | None = None,
        project: str | None = None,
        columns: list[str] | None = None,
    ) -> Any:
        """Get memories as Arrow table for efficient processing.

        Arrow tables enable zero-copy data sharing and efficient columnar
        operations. Use this for large-scale analytics.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.
            columns: Columns to select (None = all).

        Returns:
            PyArrow Table with selected data.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            search = self._db.table.search()

            filters: list[str] = []
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                filters.append(f"namespace = '{safe_ns}'")
            if project:
                safe_proj = _sanitize_string(project)
                filters.append(f"project = '{safe_proj}'")
            if filters:
                search = search.where(" AND ".join(filters))

            if columns:
                search = search.select(columns)

            return search.to_arrow()

        except Exception as e:
            raise StorageError(f"Failed to get Arrow table: {e}") from e
