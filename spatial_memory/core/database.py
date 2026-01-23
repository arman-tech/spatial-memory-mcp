"""LanceDB database wrapper for Spatial Memory MCP Server."""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lancedb
import numpy as np
import pyarrow as pa

from spatial_memory.core.errors import MemoryNotFoundError, StorageError, ValidationError
from spatial_memory.core.utils import utc_now

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


def _sanitize_string(value: str) -> str:
    """Sanitize a string value for use in LanceDB filter expressions.

    Prevents SQL injection by escaping special characters and validating input.

    Args:
        value: The string value to sanitize.

    Returns:
        Sanitized string safe for use in filter expressions.

    Raises:
        ValidationError: If the value contains invalid characters.
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value).__name__}")

    # Check for obviously malicious patterns
    dangerous_patterns = [
        r";\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
        r"--\s*$",
        r"/\*.*\*/",
        r"'\s*OR\s*'",
        r"'\s*AND\s*'",
        r"'\s*UNION\s+(?:ALL\s+)?SELECT",  # UNION-based SQL injection
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValidationError(f"Invalid characters in value: {value[:50]}")

    # Escape single quotes by doubling them (standard SQL escaping)
    return value.replace("'", "''")


def _validate_uuid(value: str) -> str:
    """Validate that a value is a valid UUID format.

    Args:
        value: The value to validate.

    Returns:
        The validated UUID string.

    Raises:
        ValidationError: If the value is not a valid UUID.
    """
    try:
        # Attempt to parse as UUID to validate format
        uuid.UUID(value)
        return value
    except (ValueError, AttributeError) as e:
        raise ValidationError(f"Invalid UUID format: {value}") from e


def _validate_namespace(value: str) -> str:
    """Validate namespace format.

    Args:
        value: The namespace to validate.

    Returns:
        The validated namespace string.

    Raises:
        ValidationError: If the namespace is invalid.
    """
    if not value:
        raise ValidationError("Namespace cannot be empty")
    if len(value) > 256:
        raise ValidationError("Namespace too long (max 256 characters)")
    # Allow alphanumeric, dash, underscore, dot
    if not re.match(r"^[\w\-\.]+$", value):
        raise ValidationError(f"Invalid namespace format: {value}")
    return value


class Database:
    """LanceDB wrapper for memory storage and retrieval.

    Supports context manager protocol for safe resource management.

    Example:
        with Database(path) as db:
            db.insert(content="Hello", vector=vec)
    """

    def __init__(self, storage_path: Path, embedding_dim: int = 384) -> None:
        """Initialize the database connection.

        Args:
            storage_path: Path to LanceDB storage directory.
            embedding_dim: Dimension of embedding vectors.
        """
        self.storage_path = Path(storage_path)
        self.embedding_dim = embedding_dim
        self._db: lancedb.DBConnection | None = None
        self._table: LanceTable | None = None

    def __enter__(self) -> Database:
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def connect(self) -> None:
        """Connect to the database and ensure table exists."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.storage_path))
            self._ensure_table()
            logger.info(f"Connected to LanceDB at {self.storage_path}")
        except Exception as e:
            raise StorageError(f"Failed to connect to database: {e}") from e

    def _ensure_table(self) -> None:
        """Ensure the memories table exists."""
        if self._db is None:
            raise StorageError("Database not connected")

        existing_tables = self._db.list_tables()
        if "memories" not in existing_tables:
            # Create table with schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("created_at", pa.timestamp("us")),
                pa.field("updated_at", pa.timestamp("us")),
                pa.field("last_accessed", pa.timestamp("us")),
                pa.field("access_count", pa.int32()),
                pa.field("importance", pa.float32()),
                pa.field("namespace", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source", pa.string()),
                pa.field("metadata", pa.string()),
            ])
            self._table = self._db.create_table("memories", schema=schema)
            logger.info("Created memories table")
        else:
            self._table = self._db.open_table("memories")
            logger.debug("Opened existing memories table")

    @property
    def table(self) -> LanceTable:
        """Get the memories table, connecting if needed."""
        if self._table is None:
            self.connect()
        assert self._table is not None  # connect() sets this or raises
        return self._table

    def close(self) -> None:
        """Close the database connection."""
        self._table = None
        self._db = None
        logger.debug("Database connection closed")

    def insert(
        self,
        content: str,
        vector: np.ndarray,
        namespace: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
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

        Returns:
            The generated memory ID.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        # Validate inputs
        namespace = _validate_namespace(namespace)
        if not content or len(content) > 100000:
            raise ValidationError("Content must be between 1 and 100000 characters")
        if not 0.0 <= importance <= 1.0:
            raise ValidationError("Importance must be between 0.0 and 1.0")

        memory_id = str(uuid.uuid4())
        now = utc_now()

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
            "tags": tags or [],
            "source": source,
            "metadata": json.dumps(metadata or {}),
        }

        try:
            self.table.add([record])
            logger.debug(f"Inserted memory {memory_id}")
            return memory_id
        except Exception as e:
            raise StorageError(f"Failed to insert memory: {e}") from e

    def insert_batch(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple memories efficiently.

        Args:
            records: List of memory records with content, vector, and optional fields.

        Returns:
            List of generated memory IDs.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        now = utc_now()
        memory_ids = []
        prepared_records = []

        for record in records:
            # Validate each record
            namespace = _validate_namespace(record.get("namespace", "default"))
            content = record.get("content", "")
            if not content or len(content) > 100000:
                raise ValidationError("Content must be between 1 and 100000 characters")

            importance = record.get("importance", 0.5)
            if not 0.0 <= importance <= 1.0:
                raise ValidationError("Importance must be between 0.0 and 1.0")

            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)

            raw_vector = record["vector"]
            vector_list = raw_vector.tolist() if isinstance(raw_vector, np.ndarray) else raw_vector
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
                "tags": record.get("tags", []),
                "source": record.get("source", "manual"),
                "metadata": json.dumps(record.get("metadata", {})),
            }
            prepared_records.append(prepared)

        try:
            self.table.add(prepared_records)
            logger.debug(f"Inserted {len(memory_ids)} memories")
            return memory_ids
        except Exception as e:
            raise StorageError(f"Failed to insert batch: {e}") from e

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
            results = self.table.search().where(f"id = '{safe_id}'").limit(1).to_list()
            if not results:
                raise MemoryNotFoundError(memory_id)

            record = results[0]
            record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
            return cast(dict[str, Any], record)
        except MemoryNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get memory: {e}") from e

    def update(self, memory_id: str, updates: dict[str, Any]) -> None:
        """Update a memory.

        Uses a backup-and-restore pattern to ensure atomicity: if the add
        operation fails after delete, the original record is restored.

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
        safe_id = _sanitize_string(memory_id)

        # First verify the memory exists and create a backup for atomicity
        existing = self.get(memory_id)
        backup_record = copy.deepcopy(existing)

        # Ensure backup metadata is serialized for potential restore
        if isinstance(backup_record.get("metadata"), dict):
            backup_record["metadata"] = json.dumps(backup_record["metadata"])
        if isinstance(backup_record.get("vector"), np.ndarray):
            backup_record["vector"] = backup_record["vector"].tolist()

        # Prepare updates
        updates["updated_at"] = utc_now()
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])
        if "vector" in updates and isinstance(updates["vector"], np.ndarray):
            updates["vector"] = updates["vector"].tolist()

        try:
            # LanceDB update: delete and re-insert
            self.table.delete(f"id = '{safe_id}'")

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
                self.table.add([existing])
                logger.debug(f"Updated memory {memory_id}")
            except Exception as add_error:
                # Add failed after delete - restore the backup to prevent data loss
                logger.error(f"Failed to add updated record, restoring backup: {add_error}")
                try:
                    self.table.add([backup_record])
                    logger.info(f"Successfully restored backup for memory {memory_id}")
                except Exception as restore_error:
                    # Critical: both add and restore failed
                    logger.critical(
                        f"CRITICAL: Failed to restore backup for memory {memory_id}. "
                        f"Original error: {add_error}, Restore error: {restore_error}"
                    )
                raise StorageError(
                    f"Failed to update memory (add failed): {add_error}"
                ) from add_error
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

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
            self.table.delete(f"id = '{safe_id}'")
            logger.debug(f"Deleted memory {memory_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete memory: {e}") from e

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
            count_before: int = self.table.count_rows()
            self.table.delete(f"namespace = '{safe_ns}'")
            count_after: int = self.table.count_rows()
            deleted = count_before - count_after
            logger.debug(f"Deleted {deleted} memories in namespace '{namespace}'")
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete by namespace: {e}") from e

    def vector_search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories by vector.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.

        Returns:
            List of memory records with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            search = self.table.search(query_vector.tolist())

            # Build filter with sanitized namespace
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

            results: list[dict[str, Any]] = search.limit(limit).to_list()

            # Process results
            for record in results:
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
                # LanceDB returns _distance, convert to similarity
                if "_distance" in record:
                    # Cosine distance to similarity: 1 - distance (for normalized vectors)
                    record["similarity"] = 1 - record["_distance"]
                    del record["_distance"]

            return results
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to search: {e}") from e

    def get_all(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories, optionally filtered by namespace.

        Args:
            namespace: Filter to specific namespace.
            limit: Maximum number of results.

        Returns:
            List of memory records.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            search = self.table.search()

            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

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

    def count(self, namespace: str | None = None) -> int:
        """Count memories.

        Args:
            namespace: Filter to specific namespace.

        Returns:
            Number of memories.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                results = self.table.search().where(f"namespace = '{safe_ns}'").to_list()
                return len(results)
            count: int = self.table.count_rows()
            return count
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to count memories: {e}") from e

    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces.

        Returns:
            List of namespace names.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            results = self.table.search().select(["namespace"]).to_list()
            namespaces = set(r["namespace"] for r in results)
            return sorted(namespaces)
        except Exception as e:
            raise StorageError(f"Failed to get namespaces: {e}") from e

    def update_access(self, memory_id: str) -> None:
        """Update access timestamp and count for a memory.

        Args:
            memory_id: The memory ID.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        existing = self.get(memory_id)
        self.update(memory_id, {
            "last_accessed": utc_now(),
            "access_count": existing["access_count"] + 1,
        })
