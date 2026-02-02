"""Idempotency key management for LanceDB database.

Provides idempotency key storage and lookup to prevent duplicate
memory creation from retried requests.

This module is part of the database.py refactoring to separate concerns:
- IdempotencyManager handles all idempotency key operations
- Database class delegates to IdempotencyManager for these operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa

from spatial_memory.core.errors import StorageError, ValidationError
from spatial_memory.core.utils import to_aware_utc, utc_now
from spatial_memory.core.validation import sanitize_string as _sanitize_string

if TYPE_CHECKING:
    import lancedb
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyRecord:
    """Record for idempotency key tracking."""

    key: str
    memory_id: str
    created_at: Any  # datetime
    expires_at: Any  # datetime


class IdempotencyManagerProtocol(Protocol):
    """Protocol defining what IdempotencyManager needs from Database.

    This protocol enables loose coupling between IdempotencyManager and Database,
    preventing circular imports while maintaining type safety.
    """

    @property
    def _db(self) -> lancedb.DBConnection | None:
        """Access to the LanceDB connection."""
        ...


class IdempotencyManager:
    """Manages idempotency keys for deduplication.

    Provides storage and lookup of idempotency keys to prevent
    duplicate memory creation from retried requests.

    Keys are stored with TTL and automatically expire. The main
    table is created lazily on first access.

    Example:
        idem_mgr = IdempotencyManager(database)
        existing = idem_mgr.get_by_idempotency_key("request-123")
        if existing:
            return existing.memory_id  # Return cached result
        # ... create new memory ...
        idem_mgr.store_idempotency_key("request-123", new_memory_id)
    """

    def __init__(self, db: IdempotencyManagerProtocol) -> None:
        """Initialize the idempotency manager.

        Args:
            db: Database instance providing connection access.
        """
        self._db_ref = db
        self._table: LanceTable | None = None

    def _ensure_idempotency_table(self) -> None:
        """Ensure the idempotency keys table exists with proper indexes."""
        db_conn = self._db_ref._db
        if db_conn is None:
            raise StorageError("Database not connected")

        existing_tables_result = db_conn.list_tables()
        if hasattr(existing_tables_result, "tables"):
            existing_tables = existing_tables_result.tables
        else:
            existing_tables = existing_tables_result

        if "idempotency_keys" not in existing_tables:
            schema = pa.schema([
                pa.field("key", pa.string()),
                pa.field("memory_id", pa.string()),
                pa.field("created_at", pa.timestamp("us")),
                pa.field("expires_at", pa.timestamp("us")),
            ])
            table = db_conn.create_table("idempotency_keys", schema=schema)
            logger.info("Created idempotency_keys table")

            # Create BTREE index on key column for fast lookups
            try:
                table.create_scalar_index("key", index_type="BTREE", replace=True)
                logger.info("Created BTREE index on idempotency_keys.key")
            except Exception as e:
                logger.warning(f"Could not create index on idempotency_keys.key: {e}")

    @property
    def idempotency_table(self) -> LanceTable:
        """Get the idempotency keys table, creating if needed."""
        db_conn = self._db_ref._db
        if db_conn is None:
            raise StorageError("Database not connected")
        self._ensure_idempotency_table()
        return db_conn.open_table("idempotency_keys")

    def get_by_idempotency_key(self, key: str) -> IdempotencyRecord | None:
        """Look up an idempotency record by key.

        Args:
            key: The idempotency key to look up.

        Returns:
            IdempotencyRecord if found and not expired, None otherwise.

        Raises:
            StorageError: If database operation fails.
        """
        if not key:
            return None

        try:
            safe_key = _sanitize_string(key)
            results = (
                self.idempotency_table.search()
                .where(f"key = '{safe_key}'")
                .limit(1)
                .to_list()
            )

            if not results:
                return None

            record = results[0]
            now = utc_now()

            # Check if expired (convert DB naive datetime to aware for comparison)
            expires_at = record.get("expires_at")
            if expires_at is not None:
                expires_at_aware = to_aware_utc(expires_at)
                if expires_at_aware < now:
                    # Expired - clean it up and return None
                    logger.debug(f"Idempotency key '{key}' has expired")
                    return None

            return IdempotencyRecord(
                key=record["key"],
                memory_id=record["memory_id"],
                created_at=record["created_at"],
                expires_at=record["expires_at"],
            )

        except Exception as e:
            raise StorageError(f"Failed to look up idempotency key: {e}") from e

    def store_idempotency_key(
        self,
        key: str,
        memory_id: str,
        ttl_hours: float = 24.0,
    ) -> None:
        """Store an idempotency key mapping.

        Note: This method should be called within a write lock context
        (the Database class handles locking).

        Args:
            key: The idempotency key.
            memory_id: The memory ID that was created.
            ttl_hours: Time-to-live in hours (default: 24 hours).

        Raises:
            ValidationError: If inputs are invalid.
            StorageError: If database operation fails.
        """
        if not key:
            raise ValidationError("Idempotency key cannot be empty")
        if not memory_id:
            raise ValidationError("Memory ID cannot be empty")
        if ttl_hours <= 0:
            raise ValidationError("TTL must be positive")

        now = utc_now()
        expires_at = now + timedelta(hours=ttl_hours)

        record = {
            "key": key,
            "memory_id": memory_id,
            "created_at": now,
            "expires_at": expires_at,
        }

        try:
            self.idempotency_table.add([record])
            logger.debug(
                f"Stored idempotency key '{key}' -> memory '{memory_id}' "
                f"(expires in {ttl_hours}h)"
            )
        except Exception as e:
            raise StorageError(f"Failed to store idempotency key: {e}") from e

    def cleanup_expired_idempotency_keys(self) -> int:
        """Remove expired idempotency keys.

        Note: This method should be called within a write lock context
        (the Database class handles locking).

        Returns:
            Number of keys removed.

        Raises:
            StorageError: If cleanup fails.
        """
        try:
            now = utc_now()
            count_before = self.idempotency_table.count_rows()

            # Delete expired keys
            predicate = f"expires_at < timestamp '{now.isoformat()}'"
            self.idempotency_table.delete(predicate)

            count_after = self.idempotency_table.count_rows()
            deleted = count_before - count_after

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired idempotency keys")

            return deleted
        except Exception as e:
            raise StorageError(f"Failed to cleanup idempotency keys: {e}") from e
