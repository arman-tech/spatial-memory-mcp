"""Export, import, access tracking, and TTL operations for LanceDB database.

Provides export/import functionality, batch access updates, and TTL management.

This module is part of the database.py refactoring to separate concerns:
- ExportImportManager handles all export, import, and TTL operations
- Database class delegates to ExportImportManager for these operations
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator, Iterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pyarrow.parquet as pq

from spatial_memory.core.errors import StorageError, ValidationError
from spatial_memory.core.utils import utc_now
from spatial_memory.core.validation import (
    sanitize_string as _sanitize_string,
)
from spatial_memory.core.validation import (
    validate_namespace as _validate_namespace,
)
from spatial_memory.core.validation import (
    validate_uuid as _validate_uuid,
)

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


class ExportImportManagerProtocol(Protocol):
    """Protocol defining what ExportImportManager needs from Database.

    This protocol enables loose coupling between ExportImportManager and Database,
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
    def enable_memory_expiration(self) -> bool:
        """Whether memory expiration is enabled."""
        ...

    @property
    def default_memory_ttl_days(self) -> int | None:
        """Default TTL for memories in days."""
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

    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a memory by ID."""
        ...

    def insert_batch(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 1000,
        atomic: bool = False,
    ) -> list[str]:
        """Insert multiple memories efficiently."""
        ...

    def get_vectors_as_arrow(
        self,
        namespace: str | None = None,
        project: str | None = None,
        columns: list[str] | None = None,
    ) -> Any:
        """Get memories as Arrow table."""
        ...


class ExportImportManager:
    """Manages export, import, access tracking, and TTL operations.

    Handles streaming export, bulk import, Parquet I/O, batch access
    updates, and memory TTL management.

    Example:
        manager = ExportImportManager(database)
        stats = manager.export_to_parquet(Path("backup.parquet"))
    """

    def __init__(self, db: ExportImportManagerProtocol) -> None:
        """Initialize the export/import manager.

        Args:
            db: Database instance (satisfies ExportImportManagerProtocol).
        """
        self._db = db

    def get_all_for_export(
        self,
        namespace: str | None = None,
        project: str | None = None,
        batch_size: int = 1000,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Stream all memories for export in batches.

        Memory-efficient export using generator pattern.

        Args:
            namespace: Optional namespace filter.
            project: Optional project filter.
            batch_size: Records per batch.

        Yields:
            Batches of memory dictionaries.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            search = self._db.table.search()

            filters: list[str] = []
            if namespace is not None:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                filters.append(f"namespace = '{safe_ns}'")
            if project:
                safe_proj = _sanitize_string(project)
                filters.append(f"project = '{safe_proj}'")
            if filters:
                search = search.where(" AND ".join(filters))

            # Use Arrow for efficient streaming
            arrow_table = search.to_arrow()
            records = arrow_table.to_pylist()

            # Yield in batches
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                # Process metadata
                for record in batch:
                    if isinstance(record.get("metadata"), str):
                        try:
                            record["metadata"] = json.loads(record["metadata"])
                        except json.JSONDecodeError:
                            record["metadata"] = {}

                yield batch

        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to stream export: {e}") from e

    def bulk_import(
        self,
        records: Iterator[dict[str, Any]],
        batch_size: int = 1000,
        namespace_override: str | None = None,
    ) -> tuple[int, list[str]]:
        """Import memories from an iterator of records.

        Supports streaming import for large datasets.

        Args:
            records: Iterator of memory dictionaries.
            batch_size: Records per database insert batch.
            namespace_override: Override namespace for all records.

        Returns:
            Tuple of (records_imported, list_of_new_ids).

        Raises:
            ValidationError: If records contain invalid data.
            StorageError: If database operation fails.
        """
        if namespace_override is not None:
            namespace_override = _validate_namespace(namespace_override)

        imported = 0
        all_ids: list[str] = []
        batch: list[dict[str, Any]] = []

        try:
            for record in records:
                prepared = self._prepare_import_record(record, namespace_override)
                batch.append(prepared)

                if len(batch) >= batch_size:
                    ids = self._db.insert_batch(batch, batch_size=batch_size)
                    all_ids.extend(ids)
                    imported += len(ids)
                    batch = []

            # Import remaining
            if batch:
                ids = self._db.insert_batch(batch, batch_size=batch_size)
                all_ids.extend(ids)
                imported += len(ids)

            return imported, all_ids

        except (ValidationError, StorageError):
            raise
        except Exception as e:
            raise StorageError(f"Bulk import failed: {e}") from e

    def _prepare_import_record(
        self,
        record: dict[str, Any],
        namespace_override: str | None = None,
    ) -> dict[str, Any]:
        """Prepare a record for import.

        Args:
            record: The raw record from import file.
            namespace_override: Optional namespace override.

        Returns:
            Prepared record suitable for insert_batch.
        """
        # Required fields
        content = record.get("content", "")
        vector = record.get("vector", [])

        # Convert vector to numpy if needed
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        # Get namespace (override if specified)
        namespace = namespace_override or record.get("namespace", "default")

        # Optional fields with defaults
        tags = record.get("tags", [])
        importance = record.get("importance", 0.5)
        source = record.get("source", "import")
        metadata = record.get("metadata", {})

        return {
            "content": content,
            "vector": vector,
            "namespace": namespace,
            "tags": tags,
            "importance": importance,
            "source": source,
            "metadata": metadata,
        }

    def update_access_batch(self, memory_ids: list[str]) -> int:
        """Update access timestamp and count for multiple memories using atomic merge_insert.

        Uses LanceDB's merge_insert API for atomic batch upserts, eliminating
        race conditions from delete-then-insert patterns.

        Args:
            memory_ids: List of memory UUIDs to update.

        Returns:
            Number of memories successfully updated.
        """
        if not memory_ids:
            return 0

        now = utc_now()
        records_to_update: list[dict[str, Any]] = []

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
            return 0

        # Batch fetch all records with single IN query (fixes N+1 pattern)
        try:
            id_list = ", ".join(f"'{mid}'" for mid in validated_ids)
            all_records = self._db.table.search().where(f"id IN ({id_list})").to_list()
        except Exception as e:
            logger.error(f"Failed to batch fetch records for access update: {e}")
            return 0

        # Build lookup map for found records
        found_ids = set()
        for record in all_records:
            found_ids.add(record["id"])
            record["last_accessed"] = now
            record["access_count"] = record["access_count"] + 1

            # Ensure proper serialization for metadata
            if isinstance(record.get("metadata"), dict):
                record["metadata"] = json.dumps(record["metadata"])

            # Ensure vector is a list, not numpy array
            if isinstance(record.get("vector"), np.ndarray):
                record["vector"] = record["vector"].tolist()

            records_to_update.append(record)

        # Log any IDs that weren't found
        missing_ids = set(validated_ids) - found_ids
        for missing_id in missing_ids:
            logger.debug(f"Memory {missing_id} not found for access update")

        if not records_to_update:
            return 0

        try:
            # Atomic batch upsert using merge_insert
            # Requires BTREE index on 'id' column (created in create_scalar_indexes)
            (
                self._db.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records_to_update)
            )
            updated = len(records_to_update)
            logger.debug(
                f"Batch updated access for {updated}/{len(memory_ids)} memories "
                "(atomic merge_insert)"
            )
            return updated
        except Exception as e:
            logger.error(f"Failed to batch update access: {e}")
            return 0

    def export_to_parquet(
        self,
        output_path: Path,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Export memories to Parquet file for backup.

        Parquet provides efficient compression and fast read performance
        for large datasets.

        Args:
            output_path: Path to save Parquet file.
            namespace: Export only this namespace (None = all).

        Returns:
            Export statistics (rows_exported, output_path, size_mb).

        Raises:
            StorageError: If export fails.
        """
        try:
            # Get all data as Arrow table (efficient)
            arrow_table = self._db.get_vectors_as_arrow(namespace=namespace)

            # Ensure parent directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to Parquet with compression
            pq.write_table(
                arrow_table,
                output_path,
                compression="zstd",  # Good compression + fast decompression
            )

            size_bytes = output_path.stat().st_size

            logger.info(
                f"Exported {arrow_table.num_rows} memories to {output_path} "
                f"({size_bytes / (1024 * 1024):.2f} MB)"
            )

            return {
                "rows_exported": arrow_table.num_rows,
                "output_path": str(output_path),
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
            }

        except Exception as e:
            raise StorageError(f"Export failed: {e}") from e

    def import_from_parquet(
        self,
        parquet_path: Path,
        namespace_override: str | None = None,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Import memories from Parquet backup.

        Args:
            parquet_path: Path to Parquet file.
            namespace_override: Override namespace for all imported memories.
            batch_size: Records per batch during import.

        Returns:
            Import statistics (rows_imported, source).

        Raises:
            StorageError: If import fails.
        """
        try:
            parquet_path = Path(parquet_path)
            if not parquet_path.exists():
                raise StorageError(f"Parquet file not found: {parquet_path}")

            table = pq.read_table(parquet_path)
            total_rows = table.num_rows

            logger.info(f"Importing {total_rows} memories from {parquet_path}")

            # Convert to list of dicts for processing
            records = table.to_pylist()

            # Override namespace if requested
            if namespace_override:
                namespace_override = _validate_namespace(namespace_override)
                for record in records:
                    record["namespace"] = namespace_override

            # Regenerate IDs to avoid conflicts
            for record in records:
                record["id"] = str(uuid.uuid4())
                # Ensure metadata is properly formatted
                if isinstance(record.get("metadata"), str):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}

            # After reading from parquet, serialize metadata back to JSON string
            # Parquet may read metadata as dict/struct, but the database expects JSON string
            for record in records:
                if "metadata" in record and isinstance(record["metadata"], dict):
                    record["metadata"] = json.dumps(record["metadata"])

            # Insert in batches
            imported = 0
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # Convert to format expected by insert
                prepared = []
                for r in batch:
                    # Ensure metadata is a JSON string for storage
                    metadata = r.get("metadata", {})
                    if isinstance(metadata, dict):
                        metadata = json.dumps(metadata)
                    elif metadata is None:
                        metadata = "{}"

                    prepared.append(
                        {
                            "content": r["content"],
                            "vector": r["vector"],
                            "namespace": r["namespace"],
                            "tags": r.get("tags", []),
                            "importance": r.get("importance", 0.5),
                            "source": r.get("source", "import"),
                            "metadata": metadata,
                            "expires_at": r.get("expires_at"),  # Preserve TTL from source
                        }
                    )
                self._db.table.add(prepared)
                imported += len(batch)
                logger.debug(f"Imported batch: {imported}/{total_rows}")

            logger.info(f"Successfully imported {imported} memories")

            return {
                "rows_imported": imported,
                "source": str(parquet_path),
            }

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Import failed: {e}") from e

    def set_memory_ttl(self, memory_id: str, ttl_days: int | None) -> None:
        """Set TTL for a specific memory.

        Args:
            memory_id: Memory ID.
            ttl_days: Days until expiration, or None to remove TTL.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        memory_id = _validate_uuid(memory_id)

        # Verify memory exists
        existing = self._db.get(memory_id)

        if ttl_days is not None:
            if ttl_days <= 0:
                raise ValidationError("TTL days must be positive")
            expires_at = utc_now() + timedelta(days=ttl_days)
        else:
            expires_at = None

        # Prepare record with TTL update
        existing["expires_at"] = expires_at
        existing["updated_at"] = utc_now()

        # Ensure proper serialization for LanceDB
        if isinstance(existing.get("metadata"), dict):
            existing["metadata"] = json.dumps(existing["metadata"])
        if isinstance(existing.get("vector"), np.ndarray):
            existing["vector"] = existing["vector"].tolist()

        try:
            # Atomic upsert using merge_insert (same pattern as update() method)
            # This prevents data loss if the operation fails partway through
            (
                self._db.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([existing])
            )
            logger.debug(f"Set TTL for memory {memory_id}: expires_at={expires_at}")
        except Exception as e:
            raise StorageError(f"Failed to set memory TTL: {e}") from e

    def cleanup_expired_memories(self) -> int:
        """Delete memories that have passed their expiration time.

        Returns:
            Number of deleted memories.

        Raises:
            StorageError: If cleanup fails.
        """
        if not self._db.enable_memory_expiration:
            logger.debug("Memory expiration is disabled, skipping cleanup")
            return 0

        try:
            now = utc_now()
            count_before: int = self._db.table.count_rows()

            # Delete expired memories using timestamp comparison
            # LanceDB uses ISO 8601 format for timestamp comparisons
            predicate = f"expires_at IS NOT NULL AND expires_at < timestamp '{now.isoformat()}'"
            self._db.table.delete(predicate)

            count_after: int = self._db.table.count_rows()
            deleted: int = count_before - count_after

            if deleted > 0:
                self._db._invalidate_count_cache()
                self._db._track_modification(deleted)
                self._db._invalidate_namespace_cache()
                logger.info(f"Cleaned up {deleted} expired memories")

            return deleted
        except Exception as e:
            raise StorageError(f"Failed to cleanup expired memories: {e}") from e
