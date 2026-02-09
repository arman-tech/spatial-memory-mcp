"""CRUD operations for LanceDB database.

Provides insert, get, update, delete, and batch operations.

This module is part of the database.py refactoring to separate concerns:
- CrudManager handles all CRUD operations
- Database class delegates to CrudManager for these operations
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pyarrow as pa

from spatial_memory.core.errors import (
    DimensionMismatchError,
    MemoryNotFoundError,
    PartialBatchInsertError,
    StorageError,
    ValidationError,
)
from spatial_memory.core.utils import utc_now
from spatial_memory.core.validation import (
    sanitize_string as _sanitize_string,
)
from spatial_memory.core.validation import (
    validate_metadata as _validate_metadata,
)
from spatial_memory.core.validation import (
    validate_namespace as _validate_namespace,
)
from spatial_memory.core.validation import (
    validate_tags as _validate_tags,
)
from spatial_memory.core.validation import (
    validate_uuid as _validate_uuid,
)

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


class CrudManagerProtocol(Protocol):
    """Protocol defining what CrudManager needs from Database.

    This protocol enables loose coupling between CrudManager and Database,
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

    @property
    def auto_create_indexes(self) -> bool:
        """Whether to automatically create indexes."""
        ...

    @property
    def vector_index_threshold(self) -> int:
        """Row count to trigger vector index creation."""
        ...

    _has_vector_index: bool | None

    @property
    def default_memory_ttl_days(self) -> int | None:
        """Default TTL for memories in days."""
        ...

    @property
    def enable_memory_expiration(self) -> bool:
        """Whether memory expiration is enabled."""
        ...

    def _get_cached_row_count(self) -> int:
        """Get cached row count for performance."""
        ...

    def _invalidate_count_cache(self) -> None:
        """Invalidate the row count cache after modifications."""
        ...

    def _invalidate_namespace_cache(self) -> None:
        """Invalidate the namespace cache after modifications."""
        ...

    def _track_modification(self, count: int = 1) -> None:
        """Track database modifications for auto-compaction."""
        ...

    def ensure_indexes(self, force: bool = False) -> dict[str, bool]:
        """Ensure all appropriate indexes exist."""
        ...

    def _reset_index_state(self) -> None:
        """Reset all index tracking flags."""
        ...


class CrudManager:
    """Manages CRUD operations for the database.

    Handles insert, get, update, delete, and batch variants for memories.

    Example:
        manager = CrudManager(database)
        memory_id = manager.insert(content="hello", vector=vec)
    """

    # Maximum batch size to prevent memory exhaustion
    MAX_BATCH_SIZE = 10_000

    def __init__(self, db: CrudManagerProtocol) -> None:
        """Initialize the CRUD manager.

        Args:
            db: Database instance (satisfies CrudManagerProtocol).
        """
        self._db = db

    def _to_arrow_table(self, records: list[dict[str, Any]]) -> pa.Table:
        """Convert record dicts to a PyArrow table matching the LanceDB schema.

        LanceDB's ``add_columns`` creates non-nullable columns, but PyArrow
        infers nullable types from plain dicts. This mismatch causes
        ``Append with different schema`` errors. Building a ``pa.Table``
        with the existing table schema ensures nullability flags match.
        """
        return pa.Table.from_pylist(records, schema=self._db.table.schema)

    def insert(
        self,
        content: str,
        vector: np.ndarray,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        project: str = "",
        content_hash: str = "",
        _skip_field_validation: bool = False,
    ) -> str:
        """Insert a new memory.

        Args:
            content: Text content of the memory.
            vector: Embedding vector.
            namespace: Namespace for organization.
            tags: List of tags.
            importance: Importance score (0-1).
            source: Source of the memory.
            metadata: Additional metadata.
            project: Project scope.
            content_hash: SHA-256 content hash.
            _skip_field_validation: Skip redundant field validation when the
                caller (e.g. repository adapter) has already validated inputs.
                Vector dimension check is always performed.

        Returns:
            The generated memory ID.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if not _skip_field_validation:
            # Validate inputs
            namespace = _validate_namespace(namespace)
            tags = _validate_tags(tags)
            metadata = _validate_metadata(metadata)
            if not content or len(content) > 100000:
                raise ValidationError("Content must be between 1 and 100000 characters")
            if not 0.0 <= importance <= 1.0:
                raise ValidationError("Importance must be between 0.0 and 1.0")
        else:
            tags = tags or []
            metadata = metadata or {}

        # Validate vector dimensions (always — only layer with embedding_dim)
        if len(vector) != self._db.embedding_dim:
            raise DimensionMismatchError(
                expected_dim=self._db.embedding_dim,
                actual_dim=len(vector),
            )

        memory_id = str(uuid.uuid4())
        now = utc_now()

        # Calculate expires_at if default TTL is configured
        expires_at = None
        if self._db.default_memory_ttl_days is not None:
            expires_at = now + timedelta(days=self._db.default_memory_ttl_days)

        record = {
            "id": memory_id,
            "content": content,
            "vector": vector.tolist(),
            "created_at": now,
            "updated_at": now,
            "last_accessed": now,
            "access_count": 0,
            "importance": importance,
            "namespace": namespace,
            "tags": tags,
            "source": source,
            "metadata": json.dumps(metadata),
            "project": project,
            "content_hash": content_hash,
            "expires_at": expires_at,
        }

        try:
            self._db.table.add(self._to_arrow_table([record]))
            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()
            logger.debug(f"Inserted memory {memory_id}")
            return memory_id
        except Exception as e:
            raise StorageError(f"Failed to insert memory: {e}") from e

    def insert_batch(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 1000,
        atomic: bool = False,
    ) -> list[str]:
        """Insert multiple memories efficiently with batching.

        Args:
            records: List of memory records with content, vector, and optional fields.
            batch_size: Records per batch (default: 1000, max: 10000).
            atomic: If True, rollback all inserts on partial failure.
                When atomic=True and a batch fails:
                - Attempts to delete already-inserted records
                - If rollback succeeds, raises the original StorageError
                - If rollback fails, raises PartialBatchInsertError with succeeded_ids

        Returns:
            List of generated memory IDs.

        Raises:
            ValidationError: If input validation fails or batch_size exceeds maximum.
            StorageError: If database operation fails (and rollback succeeds when atomic=True).
            PartialBatchInsertError: If atomic=True and rollback fails after partial insert.
        """
        if batch_size > self.MAX_BATCH_SIZE:
            raise ValidationError(
                f"batch_size ({batch_size}) exceeds maximum {self.MAX_BATCH_SIZE}"
            )

        all_ids: list[str] = []
        total_requested = len(records)

        # Process in batches for large inserts
        for batch_index, i in enumerate(range(0, len(records), batch_size)):
            batch = records[i : i + batch_size]
            now = utc_now()
            memory_ids: list[str] = []
            prepared_records: list[dict[str, Any]] = []

            for record in batch:
                # Validate each record
                namespace = _validate_namespace(record.get("namespace", "default"))
                tags = _validate_tags(record.get("tags"))
                metadata = _validate_metadata(record.get("metadata"))
                content = record.get("content", "")
                if not content or len(content) > 100000:
                    raise ValidationError("Content must be between 1 and 100000 characters")

                importance = record.get("importance", 0.5)
                if not 0.0 <= importance <= 1.0:
                    raise ValidationError("Importance must be between 0.0 and 1.0")

                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)

                raw_vector = record["vector"]
                if isinstance(raw_vector, np.ndarray):
                    vector_list = raw_vector.tolist()
                else:
                    vector_list = raw_vector

                # Validate vector dimensions
                if len(vector_list) != self._db.embedding_dim:
                    raise DimensionMismatchError(
                        expected_dim=self._db.embedding_dim,
                        actual_dim=len(vector_list),
                        record_index=i + len(memory_ids),
                    )

                # Calculate expires_at if default TTL is configured
                expires_at = None
                if self._db.default_memory_ttl_days is not None:
                    expires_at = now + timedelta(days=self._db.default_memory_ttl_days)

                prepared = {
                    "id": memory_id,
                    "content": content,
                    "vector": vector_list,
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "importance": importance,
                    "namespace": namespace,
                    "tags": tags,
                    "source": record.get("source", "manual"),
                    "metadata": json.dumps(metadata),
                    "project": record.get("project", ""),
                    "content_hash": record.get("content_hash", ""),
                    "expires_at": expires_at,
                }
                prepared_records.append(prepared)

            try:
                self._db.table.add(self._to_arrow_table(prepared_records))
                all_ids.extend(memory_ids)
                self._db._invalidate_count_cache()
                self._db._track_modification(len(memory_ids))
                self._db._invalidate_namespace_cache()
                logger.debug(f"Inserted batch {batch_index + 1}: {len(memory_ids)} memories")
            except Exception as e:
                if atomic and all_ids:
                    # Attempt rollback of previously inserted records
                    logger.warning(
                        f"Batch {batch_index + 1} failed, attempting rollback of "
                        f"{len(all_ids)} previously inserted records"
                    )
                    rollback_error = self._rollback_batch_insert(all_ids)
                    if rollback_error:
                        # Rollback failed - raise PartialBatchInsertError
                        raise PartialBatchInsertError(
                            message=f"Batch insert failed and rollback also failed: {e}",
                            succeeded_ids=all_ids,
                            total_requested=total_requested,
                            failed_batch_index=batch_index,
                        ) from e
                    else:
                        # Rollback succeeded - raise original error
                        logger.info(f"Rollback successful, deleted {len(all_ids)} records")
                        raise StorageError(f"Failed to insert batch (rolled back): {e}") from e
                raise StorageError(f"Failed to insert batch: {e}") from e

        # Check if we should create indexes after large insert
        if self._db.auto_create_indexes and len(all_ids) >= 1000:
            count = self._db._get_cached_row_count()
            if count >= self._db.vector_index_threshold and not self._db._has_vector_index:
                logger.info("Dataset crossed index threshold, creating indexes...")
                try:
                    self._db.ensure_indexes()
                except Exception as e:
                    logger.warning(f"Auto-index creation failed: {e}")

        logger.debug(f"Inserted {len(all_ids)} memories total")
        return all_ids

    def _rollback_batch_insert(self, memory_ids: list[str]) -> Exception | None:
        """Attempt to delete records inserted during a failed batch operation.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            None if rollback succeeded, Exception if it failed.
        """
        try:
            if not memory_ids:
                return None

            # Use delete_batch for efficient rollback
            id_list = ", ".join(f"'{_sanitize_string(mid)}'" for mid in memory_ids)
            self._db.table.delete(f"id IN ({id_list})")
            self._db._invalidate_count_cache()
            self._db._track_modification(len(memory_ids))
            self._db._invalidate_namespace_cache()
            logger.debug(f"Rolled back {len(memory_ids)} records")
            return None
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return e

    def search_by_content_hash(
        self,
        content_hash: str,
        namespace: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any] | None:
        """Find a memory by its content hash.

        Args:
            content_hash: SHA-256 hex digest to search for.
            namespace: Optional namespace filter.
            project: Optional project filter.

        Returns:
            The memory record dict, or None if not found.

        Raises:
            StorageError: If database operation fails.
        """
        safe_hash = _sanitize_string(content_hash)
        filters = [f"content_hash = '{safe_hash}'"]

        if namespace is not None:
            safe_ns = _sanitize_string(namespace)
            filters.append(f"namespace = '{safe_ns}'")
        if project is not None:
            safe_proj = _sanitize_string(project)
            filters.append(f"project = '{safe_proj}'")

        where_clause = " AND ".join(filters)

        try:
            results = (
                self._db.table.search()
                .where(where_clause)
                .select(
                    [
                        "id",
                        "content",
                        "namespace",
                        "project",
                        "metadata",
                        "created_at",
                        "updated_at",
                        "last_accessed",
                        "importance",
                        "tags",
                        "source",
                        "access_count",
                        "content_hash",
                    ]
                )
                .limit(1)
                .to_list()
            )
            if not results:
                return None
            record: dict[str, Any] = results[0]
            record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
            return record
        except Exception as e:
            raise StorageError(f"Failed to search by content hash: {e}") from e

    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The memory record.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        # Validate and sanitize memory_id
        memory_id = _validate_uuid(memory_id)
        safe_id = _sanitize_string(memory_id)

        try:
            results = self._db.table.search().where(f"id = '{safe_id}'").limit(1).to_list()
            if not results:
                raise MemoryNotFoundError(memory_id)

            record: dict[str, Any] = results[0]
            record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
            return record
        except MemoryNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get memory: {e}") from e

    def get_batch(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple memories by ID in a single query.

        Args:
            memory_ids: List of memory UUIDs to retrieve.

        Returns:
            Dict mapping memory_id to memory record. Missing IDs are not included.

        Raises:
            ValidationError: If any memory_id format is invalid.
            StorageError: If database operation fails.
        """
        if not memory_ids:
            return {}

        # Validate all IDs first
        validated_ids: list[str] = []
        for memory_id in memory_ids:
            try:
                validated_id = _validate_uuid(memory_id)
                validated_ids.append(_sanitize_string(validated_id))
            except Exception as e:
                logger.debug(f"Invalid memory ID {memory_id}: {e}")
                continue

        if not validated_ids:
            return {}

        try:
            # Batch fetch with single IN query
            id_list = ", ".join(f"'{mid}'" for mid in validated_ids)
            results = self._db.table.search().where(f"id IN ({id_list})").to_list()

            # Build result map
            result_map: dict[str, dict[str, Any]] = {}
            for record in results:
                # Deserialize metadata
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
                result_map[record["id"]] = record

            return result_map
        except Exception as e:
            raise StorageError(f"Failed to batch get memories: {e}") from e

    def update(self, memory_id: str, updates: dict[str, Any]) -> None:
        """Update a memory using atomic merge_insert.

        Uses LanceDB's merge_insert API for atomic upserts, eliminating
        race conditions from delete-then-insert patterns.

        Args:
            memory_id: The memory ID.
            updates: Fields to update.

        Raises:
            ValidationError: If input validation fails.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        # Validate memory_id
        memory_id = _validate_uuid(memory_id)

        # First verify the memory exists
        existing = self.get(memory_id)

        # Prepare updates
        updates["updated_at"] = utc_now()
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])
        if "vector" in updates and isinstance(updates["vector"], np.ndarray):
            updates["vector"] = updates["vector"].tolist()

        # Merge existing with updates
        for key, value in updates.items():
            existing[key] = value

        # Ensure metadata is serialized as JSON string for storage
        if isinstance(existing.get("metadata"), dict):
            existing["metadata"] = json.dumps(existing["metadata"])

        # Ensure vector is a list, not numpy array
        if isinstance(existing.get("vector"), np.ndarray):
            existing["vector"] = existing["vector"].tolist()

        try:
            # Atomic upsert using merge_insert
            # Requires BTREE index on 'id' column (created in create_scalar_indexes)
            (
                self._db.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([existing])
            )
            logger.debug(f"Updated memory {memory_id} (atomic merge_insert)")
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

    def update_batch(self, updates: list[tuple[str, dict[str, Any]]]) -> tuple[int, list[str]]:
        """Update multiple memories using atomic merge_insert.

        Args:
            updates: List of (memory_id, updates_dict) tuples.

        Returns:
            Tuple of (success_count, list of failed memory_ids).

        Raises:
            StorageError: If database operation fails completely.
        """
        if not updates:
            return 0, []

        now = utc_now()
        records_to_update: list[dict[str, Any]] = []
        failed_ids: list[str] = []

        # Validate all IDs and collect them
        validated_updates: list[tuple[str, dict[str, Any]]] = []
        for memory_id, update_dict in updates:
            try:
                validated_id = _validate_uuid(memory_id)
                validated_updates.append((_sanitize_string(validated_id), update_dict))
            except Exception as e:
                logger.debug(f"Invalid memory ID {memory_id}: {e}")
                failed_ids.append(memory_id)

        if not validated_updates:
            return 0, failed_ids

        # Batch fetch all records
        validated_ids = [vid for vid, _ in validated_updates]
        try:
            id_list = ", ".join(f"'{mid}'" for mid in validated_ids)
            all_records = self._db.table.search().where(f"id IN ({id_list})").to_list()
        except Exception as e:
            logger.error(f"Failed to batch fetch records for update: {e}")
            raise StorageError(f"Failed to batch fetch for update: {e}") from e

        # Build lookup map
        record_map: dict[str, dict[str, Any]] = {}
        for record in all_records:
            record_map[record["id"]] = record

        # Apply updates to found records
        update_dict_map = dict(validated_updates)
        for memory_id in validated_ids:
            if memory_id not in record_map:
                logger.debug(f"Memory {memory_id} not found for batch update")
                failed_ids.append(memory_id)
                continue

            record = record_map[memory_id]
            update_dict = update_dict_map[memory_id]

            # Apply updates
            record["updated_at"] = now
            for key, value in update_dict.items():
                if key == "metadata" and isinstance(value, dict):
                    record[key] = json.dumps(value)
                elif key == "vector" and isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                else:
                    record[key] = value

            # Ensure metadata is serialized
            if isinstance(record.get("metadata"), dict):
                record["metadata"] = json.dumps(record["metadata"])

            # Ensure vector is a list
            if isinstance(record.get("vector"), np.ndarray):
                record["vector"] = record["vector"].tolist()

            records_to_update.append(record)

        if not records_to_update:
            return 0, failed_ids

        try:
            # Atomic batch upsert
            (
                self._db.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records_to_update)
            )
            success_count = len(records_to_update)
            logger.debug(
                f"Batch updated {success_count}/{len(updates)} memories (atomic merge_insert)"
            )
            return success_count, failed_ids
        except Exception as e:
            logger.error(f"Failed to batch update: {e}")
            raise StorageError(f"Failed to batch update: {e}") from e

    def delete(self, memory_id: str) -> None:
        """Delete a memory.

        Args:
            memory_id: The memory ID.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        # Validate memory_id
        memory_id = _validate_uuid(memory_id)
        safe_id = _sanitize_string(memory_id)

        # First verify the memory exists
        self.get(memory_id)

        try:
            self._db.table.delete(f"id = '{safe_id}'")
            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()
            logger.debug(f"Deleted memory {memory_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_batch(self, memory_ids: list[str]) -> tuple[int, list[str]]:
        """Delete multiple memories atomically using IN clause.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Tuple of (count_deleted, list_of_deleted_ids) where:
                - count_deleted: Number of memories actually deleted
                - list_of_deleted_ids: IDs that were actually deleted

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        if not memory_ids:
            return (0, [])

        # Validate all IDs first (fail fast)
        validated_ids: list[str] = []
        for memory_id in memory_ids:
            validated_id = _validate_uuid(memory_id)
            validated_ids.append(_sanitize_string(validated_id))

        try:
            # First, check which IDs actually exist
            id_list = ", ".join(f"'{mid}'" for mid in validated_ids)
            filter_expr = f"id IN ({id_list})"
            existing_records = (
                self._db.table.search()
                .where(filter_expr)
                .select(["id"])
                .limit(len(validated_ids))
                .to_list()
            )
            existing_ids = [r["id"] for r in existing_records]

            if not existing_ids:
                return (0, [])

            # Delete only existing IDs
            existing_id_list = ", ".join(f"'{mid}'" for mid in existing_ids)
            delete_expr = f"id IN ({existing_id_list})"
            self._db.table.delete(delete_expr)

            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()

            logger.debug(f"Batch deleted {len(existing_ids)} memories")
            return (len(existing_ids), existing_ids)
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete batch: {e}") from e

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all memories in a namespace.

        Args:
            namespace: The namespace to delete.

        Returns:
            Number of deleted records.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        namespace = _validate_namespace(namespace)
        safe_ns = _sanitize_string(namespace)

        try:
            count_before: int = self._db.table.count_rows()
            self._db.table.delete(f"namespace = '{safe_ns}'")
            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()
            count_after: int = self._db.table.count_rows()
            deleted = count_before - count_after
            logger.debug(f"Deleted {deleted} memories in namespace '{namespace}'")
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete by namespace: {e}") from e

    def clear_all(self, reset_indexes: bool = True) -> int:
        """Clear all memories from the database.

        This is primarily for testing purposes to reset database state
        between tests while maintaining the connection.

        Args:
            reset_indexes: If True, also reset index tracking flags.
                          This allows tests to verify index creation behavior.

        Returns:
            Number of deleted records.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            count: int = self._db.table.count_rows()
            if count > 0:
                # Delete all rows - use simpler predicate that definitely matches
                self._db.table.delete("true")

                # Verify deletion worked
                remaining = self._db.table.count_rows()
                if remaining > 0:
                    logger.warning(
                        f"clear_all: {remaining} records remain after delete, "
                        f"attempting cleanup again"
                    )
                    # Try alternative delete approach
                    self._db.table.delete("id IS NOT NULL")

            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()

            # Reset index tracking flags for test isolation
            if reset_indexes:
                self._db._reset_index_state()

            logger.debug(f"Cleared all {count} memories from database")
            return count
        except Exception as e:
            raise StorageError(f"Failed to clear all memories: {e}") from e
