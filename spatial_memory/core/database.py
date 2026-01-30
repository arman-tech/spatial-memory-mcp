"""LanceDB database wrapper for Spatial Memory MCP Server.

Enterprise-grade implementation with:
- Connection pooling (singleton pattern)
- Automatic index creation (IVF-PQ, FTS, scalar)
- Hybrid search with RRF reranking
- Batch operations and streaming
- Maintenance and optimization utilities
- Health metrics and monitoring
- Retry logic for transient errors
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import lancedb
import lancedb.index
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spatial_memory.core.errors import MemoryNotFoundError, StorageError, ValidationError
from spatial_memory.core.utils import utc_now

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)

# Type variable for retry decorator
F = TypeVar("F", bound=Callable[..., Any])

# All known vector index types for detection
VECTOR_INDEX_TYPES = frozenset({
    "IVF_PQ", "IVF_FLAT", "HNSW",
    "IVF_HNSW_PQ", "IVF_HNSW_SQ",
    "HNSW_PQ", "HNSW_SQ",
})

# ============================================================================
# Connection Pool (Singleton Pattern with LRU Eviction)
# ============================================================================

# OrderedDict for LRU tracking - most recently used at end
_connection_cache: OrderedDict[str, lancedb.DBConnection] = OrderedDict()
_connection_lock = threading.Lock()
_max_pool_size: int = 10  # Default, can be overridden


def set_connection_pool_max_size(max_size: int) -> None:
    """Set the maximum connection pool size.

    Args:
        max_size: Maximum number of connections to cache.
    """
    global _max_pool_size
    _max_pool_size = max_size


def _get_or_create_connection(
    storage_path: Path,
    read_consistency_interval_ms: int = 0,
) -> lancedb.DBConnection:
    """Get cached connection or create new one (thread-safe with LRU eviction).

    Args:
        storage_path: Path to LanceDB storage directory.
        read_consistency_interval_ms: Read consistency interval in milliseconds.

    Returns:
        LanceDB connection instance.
    """
    path_key = str(storage_path.absolute())

    with _connection_lock:
        if path_key in _connection_cache:
            # Move to end for LRU tracking
            _connection_cache.move_to_end(path_key)
            return _connection_cache[path_key]

        # Evict oldest connection if at capacity
        while len(_connection_cache) >= _max_pool_size:
            evicted_key, evicted_conn = _connection_cache.popitem(last=False)
            try:
                if hasattr(evicted_conn, "close"):
                    evicted_conn.close()
            except Exception as e:
                logger.warning(f"Error closing evicted connection {evicted_key}: {e}")
            logger.debug(f"Evicted connection from pool: {evicted_key}")

        # Create new connection with read consistency if specified
        connect_kwargs: dict[str, Any] = {}
        if read_consistency_interval_ms > 0:
            connect_kwargs["read_consistency_interval"] = timedelta(
                milliseconds=read_consistency_interval_ms
            )

        _connection_cache[path_key] = lancedb.connect(path_key, **connect_kwargs)
        logger.info(f"Created new LanceDB connection for {path_key}")
        return _connection_cache[path_key]


def clear_connection_cache() -> None:
    """Clear the connection cache, properly closing connections.

    Should be called during shutdown or testing cleanup.
    """
    global _connection_cache
    with _connection_lock:
        for path_key, conn in list(_connection_cache.items()):
            try:
                # LanceDB connections may have cleanup methods
                if hasattr(conn, "close"):
                    conn.close()
                elif hasattr(conn, "__del__"):
                    # Trigger cleanup if available
                    pass
            except Exception as e:
                logger.warning(f"Error closing connection {path_key}: {e}")
        _connection_cache.clear()
        logger.debug("Connection cache cleared")


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_on_storage_error(
    max_attempts: int = 3,
    backoff: float = 0.5,
) -> Callable[[F], F]:
    """Retry decorator for transient storage errors.

    Args:
        max_attempts: Maximum number of retry attempts.
        backoff: Initial backoff time in seconds (doubles each attempt).

    Returns:
        Decorated function with retry logic.

    Note:
        - Decorator values are STATIC: Parameters are fixed at class definition
          time, not instance creation time. This means the instance config values
          (max_retry_attempts, retry_backoff_seconds) exist for external tooling
          or future dynamic use, but do NOT affect this decorator's behavior.
        - Does NOT retry concurrent modification or conflict errors as these
          require application-level resolution (e.g., refresh and retry).
    """
    # Patterns indicating non-retryable errors
    non_retryable_patterns = ("concurrent", "conflict", "version mismatch")

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (
                    StorageError, OSError, ConnectionError, TimeoutError
                ) as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check for non-retryable errors - raise immediately
                    if any(pattern in error_str for pattern in non_retryable_patterns):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                        raise

                    # Check if we've exhausted retries
                    if attempt == max_attempts - 1:
                        raise

                    # Retry with exponential backoff
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts})"
                        f": {e}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            # Should never reach here, but satisfy type checker
            if last_error:
                raise last_error
            return None
        return cast(F, wrapper)
    return decorator


# ============================================================================
# Health Metrics
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

    Enterprise-grade features:
    - Connection pooling via singleton pattern with LRU eviction
    - Automatic index creation based on dataset size
    - Hybrid search with RRF reranking and alpha parameter
    - Batch operations for efficiency
    - Row count caching for search performance (thread-safe)
    - Maintenance and optimization utilities

    Thread Safety:
        The module-level connection pool is thread-safe. However, individual
        Database instances should NOT be shared across threads without external
        synchronization. Each thread should create its own Database instance,
        which will share the underlying pooled connection safely.

    Supports context manager protocol for safe resource management.

    Example:
        with Database(path) as db:
            db.insert(content="Hello", vector=vec)
    """

    # Cache refresh interval for row count (seconds)
    _COUNT_CACHE_TTL = 60.0
    # Cache refresh interval for namespaces (seconds) - longer because namespaces change less often
    _NAMESPACE_CACHE_TTL = 300.0

    def __init__(
        self,
        storage_path: Path,
        embedding_dim: int = 384,
        auto_create_indexes: bool = True,
        vector_index_threshold: int = 10_000,
        enable_fts: bool = True,
        index_nprobes: int = 20,
        index_refine_factor: int = 5,
        max_retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        read_consistency_interval_ms: int = 0,
        index_wait_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the database connection.

        Args:
            storage_path: Path to LanceDB storage directory.
            embedding_dim: Dimension of embedding vectors.
            auto_create_indexes: Automatically create indexes when thresholds met.
            vector_index_threshold: Row count to trigger vector index creation.
            enable_fts: Enable full-text search index.
            index_nprobes: Number of partitions to search (higher = better recall).
            index_refine_factor: Re-rank top (refine_factor * limit) for accuracy.
            max_retry_attempts: Maximum retry attempts for transient errors.
            retry_backoff_seconds: Initial backoff time for retries.
            read_consistency_interval_ms: Read consistency interval (0 = strong).
            index_wait_timeout_seconds: Timeout for waiting on index creation.
        """
        self.storage_path = Path(storage_path)
        self.embedding_dim = embedding_dim
        self.auto_create_indexes = auto_create_indexes
        self.vector_index_threshold = vector_index_threshold
        self.enable_fts = enable_fts
        self.index_nprobes = index_nprobes
        self.index_refine_factor = index_refine_factor
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.read_consistency_interval_ms = read_consistency_interval_ms
        self.index_wait_timeout_seconds = index_wait_timeout_seconds
        self._db: lancedb.DBConnection | None = None
        self._table: LanceTable | None = None
        self._has_vector_index: bool | None = None
        self._has_fts_index: bool | None = None
        # Row count cache for performance (avoid count_rows() on every search)
        self._cached_row_count: int | None = None
        self._count_cache_time: float = 0.0
        # Thread-safe lock for row count cache
        self._cache_lock = threading.Lock()
        # Namespace cache for performance
        self._cached_namespaces: set[str] | None = None
        self._namespace_cache_time: float = 0.0
        self._namespace_cache_lock = threading.Lock()

    def __enter__(self) -> Database:
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def connect(self) -> None:
        """Connect to the database using pooled connections."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            # Use connection pooling with read consistency support
            self._db = _get_or_create_connection(
                self.storage_path,
                read_consistency_interval_ms=self.read_consistency_interval_ms,
            )
            self._ensure_table()
            logger.info(f"Connected to LanceDB at {self.storage_path}")
        except Exception as e:
            raise StorageError(f"Failed to connect to database: {e}") from e

    def _ensure_table(self) -> None:
        """Ensure the memories table exists with appropriate indexes."""
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

            # Create FTS index on new table if enabled
            if self.enable_fts:
                self._create_fts_index()
        else:
            self._table = self._db.open_table("memories")
            logger.debug("Opened existing memories table")

            # Check existing indexes
            self._check_existing_indexes()

    def _check_existing_indexes(self) -> None:
        """Check which indexes already exist using robust detection."""
        try:
            indices = self.table.list_indices()

            self._has_vector_index = False
            self._has_fts_index = False

            for idx in indices:
                index_name = idx.get("name", "").lower()
                index_type = idx.get("index_type", "").upper()
                columns = idx.get("columns", [])

                # Vector index detection: check index_type or column name
                if index_type in VECTOR_INDEX_TYPES:
                    self._has_vector_index = True
                elif "vector" in columns or "vector" in index_name:
                    self._has_vector_index = True

                # FTS index detection: check index_type or name patterns
                if index_type == "FTS":
                    self._has_fts_index = True
                elif "fts" in index_name or "content" in index_name:
                    self._has_fts_index = True

            logger.debug(
                f"Existing indexes: vector={self._has_vector_index}, "
                f"fts={self._has_fts_index}"
            )
        except Exception as e:
            logger.warning(f"Could not check existing indexes: {e}")
            self._has_vector_index = None
            self._has_fts_index = None

    def _create_fts_index(self) -> None:
        """Create full-text search index on content field."""
        try:
            self.table.create_fts_index(
                "content",
                use_tantivy=False,  # Use Lance native FTS
                with_position=True,  # Enable phrase queries
            )
            self._has_fts_index = True
            logger.info("Created FTS index on content field")
        except Exception as e:
            # Check if index already exists (not an error)
            if "already exists" in str(e).lower():
                self._has_fts_index = True
                logger.debug("FTS index already exists")
            else:
                logger.warning(f"FTS index creation failed: {e}")

    @property
    def table(self) -> LanceTable:
        """Get the memories table, connecting if needed."""
        if self._table is None:
            self.connect()
        assert self._table is not None  # connect() sets this or raises
        return self._table

    def close(self) -> None:
        """Close the database connection (connection remains pooled)."""
        self._table = None
        self._db = None
        self._has_vector_index = None
        self._has_fts_index = None
        with self._cache_lock:
            self._cached_row_count = None
            self._count_cache_time = 0.0
        with self._namespace_cache_lock:
            self._cached_namespaces = None
            self._namespace_cache_time = 0.0
        logger.debug("Database connection closed")

    def _get_cached_row_count(self) -> int:
        """Get row count with caching for performance (thread-safe).

        Avoids calling count_rows() on every search operation.
        Cache is invalidated on insert/delete or after TTL expires.

        Returns:
            Cached or fresh row count.
        """
        now = time.time()
        with self._cache_lock:
            if (
                self._cached_row_count is None
                or (now - self._count_cache_time) > self._COUNT_CACHE_TTL
            ):
                self._cached_row_count = self.table.count_rows()
                self._count_cache_time = now
            return self._cached_row_count

    def _invalidate_count_cache(self) -> None:
        """Invalidate the row count cache after modifications (thread-safe)."""
        with self._cache_lock:
            self._cached_row_count = None
            self._count_cache_time = 0.0

    def _invalidate_namespace_cache(self) -> None:
        """Invalidate the namespace cache after modifications (thread-safe)."""
        with self._namespace_cache_lock:
            self._cached_namespaces = None
            self._namespace_cache_time = 0.0

    # ========================================================================
    # Index Management
    # ========================================================================

    def create_vector_index(self, force: bool = False) -> bool:
        """Create IVF-PQ vector index for similarity search.

        Automatically determines optimal parameters based on dataset size.

        Args:
            force: Force index creation regardless of dataset size.

        Returns:
            True if index was created, False if skipped.

        Raises:
            StorageError: If index creation fails.
        """
        count = self.table.count_rows()

        # Check threshold
        if count < self.vector_index_threshold and not force:
            logger.info(
                f"Dataset has {count} rows, below threshold {self.vector_index_threshold}. "
                "Skipping vector index creation."
            )
            return False

        # Check if already exists
        if self._has_vector_index and not force:
            logger.info("Vector index already exists")
            return False

        # Determine optimal parameters based on dataset size
        if count < 100_000:
            num_partitions = 64
            num_sub_vectors = 48  # 384 / 48 = 8 dims per sub-vector
        elif count < 1_000_000:
            num_partitions = 256
            num_sub_vectors = 96  # 384 / 96 = 4 dims per sub-vector
        else:
            num_partitions = 1024
            num_sub_vectors = 96

        # Validate embedding_dim % num_sub_vectors == 0 (required for IVF-PQ)
        if self.embedding_dim % num_sub_vectors != 0:
            # Find a valid divisor from common sub-vector counts
            valid_divisors = [48, 32, 24, 16, 12, 8, 4]
            found_divisor = False
            for divisor in valid_divisors:
                if self.embedding_dim % divisor == 0:
                    logger.info(
                        f"Adjusted num_sub_vectors from {num_sub_vectors} to {divisor} "
                        f"for embedding_dim={self.embedding_dim}"
                    )
                    num_sub_vectors = divisor
                    found_divisor = True
                    break

            if not found_divisor:
                raise StorageError(
                    f"Cannot create IVF-PQ index: embedding_dim={self.embedding_dim} "
                    "has no suitable divisor for sub-vectors. "
                    f"Tried divisors: {valid_divisors}"
                )

        logger.info(
            f"Creating IVF-PQ vector index: {num_partitions} partitions, "
            f"{num_sub_vectors} sub-vectors for {count} rows"
        )

        try:
            # Use config-based API for LanceDB 0.27+
            self.table.create_index(
                "vector",
                config=lancedb.index.IvfPq(
                    num_partitions=num_partitions,
                    num_sub_vectors=num_sub_vectors,
                    distance_type="cosine",
                ),
                replace=True,
            )

            # Wait for index to be ready with configurable timeout
            self._wait_for_index_ready("vector", self.index_wait_timeout_seconds)

            self._has_vector_index = True
            logger.info("Vector index created successfully")

            # Compact after index creation
            self.table.compact_files()
            return True

        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise StorageError(f"Vector index creation failed: {e}") from e

    def _wait_for_index_ready(
        self,
        column_name: str,
        timeout_seconds: float,
        poll_interval: float = 0.5,
    ) -> None:
        """Wait for an index on the specified column to be ready.

        Args:
            column_name: Name of the column the index is on (e.g., "vector").
                         LanceDB typically names indexes as "{column_name}_idx".
            timeout_seconds: Maximum time to wait.
            poll_interval: Time between status checks.
        """
        if timeout_seconds <= 0:
            return

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                indices = self.table.list_indices()
                for idx in indices:
                    idx_name = idx.get("name", "").lower()
                    idx_columns = idx.get("columns", [])

                    # Match by column name in index metadata, or index name contains column
                    if column_name in idx_columns or column_name in idx_name:
                        # Index exists, check if it's ready
                        status = idx.get("status", "ready")
                        if status.lower() in ("ready", "complete", "built"):
                            logger.debug(f"Index on {column_name} is ready")
                            return
                        break
            except Exception as e:
                logger.debug(f"Error checking index status: {e}")

            time.sleep(poll_interval)

        logger.warning(
            f"Timeout waiting for index on {column_name} after {timeout_seconds}s"
        )

    def create_scalar_indexes(self) -> None:
        """Create scalar indexes for frequently filtered columns.

        Creates:
        - BTREE on id (fast lookups, upserts)
        - BTREE on timestamps and importance (range queries)
        - BITMAP on namespace and source (low cardinality)
        - LABEL_LIST on tags (array contains queries)

        Raises:
            StorageError: If index creation fails critically.
        """
        # BTREE indexes for range queries and lookups
        btree_columns = [
            "id",  # Fast lookups and merge_insert
            "created_at",
            "updated_at",
            "last_accessed",
            "importance",
            "access_count",
        ]

        for column in btree_columns:
            try:
                self.table.create_index(
                    column,
                    config=lancedb.index.BTree(),
                    replace=True,
                )
                logger.debug(f"Created BTREE index on {column}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create BTREE index on {column}: {e}")

        # BITMAP indexes for low-cardinality columns
        bitmap_columns = ["namespace", "source"]

        for column in bitmap_columns:
            try:
                self.table.create_index(
                    column,
                    config=lancedb.index.Bitmap(),
                    replace=True,
                )
                logger.debug(f"Created BITMAP index on {column}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create BITMAP index on {column}: {e}")

        # LABEL_LIST index for tags array (supports array_has_any queries)
        # Use config API if available (LanceDB 0.27+), fallback to legacy
        try:
            if hasattr(lancedb.index, "LabelList"):
                self.table.create_index(
                    "tags",
                    config=lancedb.index.LabelList(),
                    replace=True,
                )
            else:
                # Fallback to legacy API
                self.table.create_index("tags", index_type="LABEL_LIST", replace=True)
            logger.debug("Created LABEL_LIST index on tags")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Could not create LABEL_LIST index on tags: {e}")

        logger.info("Scalar indexes created")

    def ensure_indexes(self, force: bool = False) -> dict[str, bool]:
        """Ensure all appropriate indexes exist.

        Args:
            force: Force index creation regardless of thresholds.

        Returns:
            Dict indicating which indexes were created.
        """
        results = {
            "vector_index": False,
            "scalar_indexes": False,
            "fts_index": False,
        }

        count = self.table.count_rows()

        # Vector index
        if self.auto_create_indexes or force:
            if count >= self.vector_index_threshold or force:
                results["vector_index"] = self.create_vector_index(force=force)

        # Scalar indexes (always create if > 1000 rows)
        if count >= 1000 or force:
            try:
                self.create_scalar_indexes()
                results["scalar_indexes"] = True
            except Exception as e:
                logger.warning(f"Scalar index creation partially failed: {e}")

        # FTS index
        if self.enable_fts and not self._has_fts_index:
            try:
                self._create_fts_index()
                results["fts_index"] = True
            except Exception as e:
                logger.warning(f"FTS index creation failed in ensure_indexes: {e}")

        return results

    # ========================================================================
    # Maintenance & Optimization
    # ========================================================================

    def optimize(self) -> dict[str, Any]:
        """Run optimization and maintenance tasks.

        Performs:
        - File compaction (merges small fragments)
        - Index optimization

        Returns:
            Statistics about optimization performed.
        """
        try:
            stats_before = self._get_table_stats()

            # Compact small fragments
            needs_compaction = stats_before.get("num_small_fragments", 0) > 10
            if needs_compaction:
                logger.info("Compacting fragments...")
                self.table.compact_files()

            # Optimize indexes
            logger.info("Optimizing indexes...")
            self.table.optimize()

            stats_after = self._get_table_stats()

            return {
                "fragments_before": stats_before.get("num_fragments", 0),
                "fragments_after": stats_after.get("num_fragments", 0),
                "compaction_performed": needs_compaction,
                "total_rows": stats_after.get("num_rows", 0),
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"error": str(e)}

    def _get_table_stats(self) -> dict[str, Any]:
        """Get table statistics with best-effort fragment info."""
        try:
            count = self.table.count_rows()
            stats: dict[str, Any] = {
                "num_rows": count,
                "num_fragments": 0,
                "num_small_fragments": 0,
            }

            # Try to get fragment stats from table.stats() if available
            try:
                if hasattr(self.table, "stats"):
                    table_stats = self.table.stats()
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
            count = self.table.count_rows()

            # Estimate size (rough approximation)
            # vector (dim * 4 bytes) + avg content size estimate
            estimated_bytes = count * (self.embedding_dim * 4 + 1000)

            # Check indexes
            indices: list[IndexStats] = []
            try:
                for idx in self.table.list_indices():
                    indices.append(IndexStats(
                        name=idx.get("name", "unknown"),
                        index_type=idx.get("index_type", "unknown"),
                        num_indexed_rows=count,  # Approximate
                        num_unindexed_rows=0,
                        needs_update=False,
                    ))
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")

            return HealthMetrics(
                total_rows=count,
                total_bytes=estimated_bytes,
                total_bytes_mb=estimated_bytes / (1024 * 1024),
                num_fragments=0,
                num_small_fragments=0,
                needs_compaction=False,
                has_vector_index=self._has_vector_index or False,
                has_fts_index=self._has_fts_index or False,
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

    @retry_on_storage_error(max_attempts=3, backoff=0.5)
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
            self._invalidate_count_cache()
            self._invalidate_namespace_cache()
            logger.debug(f"Inserted memory {memory_id}")
            return memory_id
        except Exception as e:
            raise StorageError(f"Failed to insert memory: {e}") from e

    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def insert_batch(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 1000,
    ) -> list[str]:
        """Insert multiple memories efficiently with batching.

        Args:
            records: List of memory records with content, vector, and optional fields.
            batch_size: Records per batch (default: 1000).

        Returns:
            List of generated memory IDs.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        all_ids: list[str] = []

        # Process in batches for large inserts
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            now = utc_now()
            memory_ids: list[str] = []
            prepared_records: list[dict[str, Any]] = []

            for record in batch:
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
                if isinstance(raw_vector, np.ndarray):
                    vector_list = raw_vector.tolist()
                else:
                    vector_list = raw_vector
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
                all_ids.extend(memory_ids)
                self._invalidate_count_cache()
                self._invalidate_namespace_cache()
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(memory_ids)} memories")
            except Exception as e:
                raise StorageError(f"Failed to insert batch: {e}") from e

        # Check if we should create indexes after large insert
        if self.auto_create_indexes and len(all_ids) >= 1000:
            count = self._get_cached_row_count()
            if count >= self.vector_index_threshold and not self._has_vector_index:
                logger.info("Dataset crossed index threshold, creating indexes...")
                try:
                    self.ensure_indexes()
                except Exception as e:
                    logger.warning(f"Auto-index creation failed: {e}")

        logger.debug(f"Inserted {len(all_ids)} memories total")
        return all_ids

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
            return record
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
            self._invalidate_count_cache()
            self._invalidate_namespace_cache()
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
            self._invalidate_count_cache()
            self._invalidate_namespace_cache()
            count_after: int = self.table.count_rows()
            deleted = count_before - count_after
            logger.debug(f"Deleted {deleted} memories in namespace '{namespace}'")
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete by namespace: {e}") from e

    def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories atomically using IN clause.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Number of memories actually deleted.

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        if not memory_ids:
            return 0

        # Validate all IDs first (fail fast)
        validated_ids: list[str] = []
        for memory_id in memory_ids:
            validated_id = _validate_uuid(memory_id)
            validated_ids.append(_sanitize_string(validated_id))

        try:
            count_before: int = self.table.count_rows()

            # Build IN clause for atomic batch delete
            id_list = ", ".join(f"'{mid}'" for mid in validated_ids)
            filter_expr = f"id IN ({id_list})"
            self.table.delete(filter_expr)

            self._invalidate_count_cache()
            self._invalidate_namespace_cache()

            count_after: int = self.table.count_rows()
            deleted = count_before - count_after

            logger.debug(f"Batch deleted {deleted} memories")
            return deleted
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete batch: {e}") from e

    def update_access_batch(self, memory_ids: list[str]) -> int:
        """Update access timestamp and count for multiple memories.

        LanceDB requires delete+insert for updates, so this method loops through
        IDs individually but handles errors gracefully to continue on failures.

        Args:
            memory_ids: List of memory UUIDs to update.

        Returns:
            Number of memories successfully updated.
        """
        if not memory_ids:
            return 0

        updated = 0
        now = utc_now()

        for memory_id in memory_ids:
            try:
                validated_id = _validate_uuid(memory_id)
                safe_id = _sanitize_string(validated_id)

                # Get current record
                results = self.table.search().where(f"id = '{safe_id}'").limit(1).to_list()
                if not results:
                    logger.debug(f"Memory {memory_id} not found for access update")
                    continue

                record = results[0]
                record["last_accessed"] = now
                record["access_count"] = record["access_count"] + 1

                # Ensure proper serialization for metadata
                if isinstance(record.get("metadata"), dict):
                    record["metadata"] = json.dumps(record["metadata"])

                # Ensure vector is a list, not numpy array
                if isinstance(record.get("vector"), np.ndarray):
                    record["vector"] = record["vector"].tolist()

                # Delete and re-insert (LanceDB update pattern)
                self.table.delete(f"id = '{safe_id}'")
                self.table.add([record])
                updated += 1

            except Exception as e:
                logger.debug(f"Failed to update access for {memory_id}: {e}")
                continue

        logger.debug(f"Batch updated access for {updated}/{len(memory_ids)} memories")
        return updated

    def _create_retry_decorator(self) -> Callable[[F], F]:
        """Create a retry decorator using instance settings."""
        return retry_on_storage_error(
            max_attempts=self.max_retry_attempts,
            backoff=self.retry_backoff_seconds,
        )

    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def vector_search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        min_similarity: float = 0.0,
        nprobes: int | None = None,
        refine_factor: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories by vector with performance tuning.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.
            min_similarity: Minimum similarity threshold (0-1).
            nprobes: Number of partitions to search (higher = better recall).
                     Only effective when vector index exists. Defaults to instance config.
            refine_factor: Re-rank top (refine_factor * limit) for accuracy.
                          Defaults to instance config.

        Returns:
            List of memory records with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            search = self.table.search(query_vector.tolist())

            # Distance type for queries (cosine for semantic similarity)
            # Note: When vector index exists, the index's metric is used
            search = search.distance_type("cosine")

            # Apply performance tuning when index exists (use cached count)
            count = self._get_cached_row_count()
            if count > self.vector_index_threshold and self._has_vector_index:
                # Use provided values or fall back to instance config
                actual_nprobes = nprobes if nprobes is not None else self.index_nprobes
                # Increase nprobes for very large datasets
                if count > 100_000 and actual_nprobes < 30:
                    actual_nprobes = 30
                search = search.nprobes(actual_nprobes)

                # Apply refine factor for better accuracy
                if refine_factor is not None:
                    actual_refine = refine_factor
                else:
                    actual_refine = self.index_refine_factor
                search = search.refine_factor(actual_refine)

            # Build filter with sanitized namespace
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

            # Fetch extra if filtering by similarity
            fetch_limit = limit * 2 if min_similarity > 0.0 else limit
            results: list[dict[str, Any]] = search.limit(fetch_limit).to_list()

            # Process results
            filtered_results: list[dict[str, Any]] = []
            for record in results:
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
                # LanceDB returns _distance, convert to similarity
                if "_distance" in record:
                    # Cosine distance to similarity: 1 - distance
                    # Clamp to [0, 1] (cosine distance can exceed 1 for unnormalized)
                    similarity = max(0.0, min(1.0, 1 - record["_distance"]))
                    record["similarity"] = similarity
                    del record["_distance"]

                # Apply similarity threshold
                if record.get("similarity", 0) >= min_similarity:
                    filtered_results.append(record)
                    if len(filtered_results) >= limit:
                        break

            return filtered_results
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to search: {e}") from e

    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword matching.

        Uses LinearCombinationReranker to balance vector and keyword scores
        based on the alpha parameter.

        Args:
            query: Text query for full-text search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results.
            namespace: Filter to namespace.
            alpha: Balance between vector (1.0) and keyword (0.0).
                   0.5 = balanced (recommended).

        Returns:
            List of memory records with combined scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            # Check if FTS is available
            if not self._has_fts_index:
                logger.debug("FTS index not available, falling back to vector search")
                return self.vector_search(query_vector, limit=limit, namespace=namespace)

            # Create hybrid search with explicit vector column specification
            # Required when using external embeddings (not LanceDB built-in)
            search = (
                self.table.search(query, query_type="hybrid")
                .vector(query_vector.tolist())
                .vector_column_name("vector")
            )

            # Apply alpha parameter using LinearCombinationReranker
            # alpha=1.0 means full vector, alpha=0.0 means full FTS
            try:
                from lancedb.rerankers import LinearCombinationReranker

                reranker = LinearCombinationReranker(weight=alpha)
                search = search.rerank(reranker)
            except ImportError:
                logger.debug("LinearCombinationReranker not available, using default reranking")
            except Exception as e:
                logger.debug(f"Could not apply reranker: {e}")

            # Apply namespace filter
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

            results: list[dict[str, Any]] = search.limit(limit).to_list()

            # Process results
            for record in results:
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
                if "_distance" in record:
                    record["similarity"] = max(0.0, min(1.0, 1 - record["_distance"]))
                    del record["_distance"]
                # Mark as hybrid result with alpha value
                record["search_type"] = "hybrid"
                record["alpha"] = alpha

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return self.vector_search(query_vector, limit=limit, namespace=namespace)

    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def batch_vector_search(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[list[dict[str, Any]]]:
        """Search for similar memories using multiple query vectors.

        Efficient for operations like journey interpolation where multiple
        points need to find nearby memories.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum results per query vector.
            namespace: Filter to specific namespace.
            parallel: Execute searches in parallel using ThreadPoolExecutor.
            max_workers: Maximum worker threads for parallel execution.

        Returns:
            List of result lists (one per query vector).

        Raises:
            StorageError: If database operation fails.
        """
        if not query_vectors:
            return []

        # Build namespace filter once
        where_clause: str | None = None
        if namespace:
            namespace = _validate_namespace(namespace)
            safe_ns = _sanitize_string(namespace)
            where_clause = f"namespace = '{safe_ns}'"

        def search_single(vec: np.ndarray) -> list[dict[str, Any]]:
            """Execute a single vector search."""
            search = self.table.search(vec.tolist()).distance_type("cosine")

            if where_clause:
                search = search.where(where_clause)

            results: list[dict[str, Any]] = search.limit(limit_per_query).to_list()

            # Process results
            for record in results:
                meta = record["metadata"]
                record["metadata"] = json.loads(meta) if meta else {}
                if "_distance" in record:
                    record["similarity"] = max(0.0, min(1.0, 1 - record["_distance"]))
                    del record["_distance"]

            return results

        try:
            if parallel and len(query_vectors) > 1:
                # Use ThreadPoolExecutor for parallel execution
                from concurrent.futures import ThreadPoolExecutor

                workers = min(max_workers, len(query_vectors))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # Map preserves order
                    all_results = list(executor.map(search_single, query_vectors))
            else:
                # Sequential execution
                all_results = [search_single(vec) for vec in query_vectors]

            return all_results

        except Exception as e:
            raise StorageError(f"Batch vector search failed: {e}") from e

    def get_vectors_for_clustering(
        self,
        namespace: str | None = None,
        max_memories: int = 10_000,
    ) -> tuple[list[str], np.ndarray]:
        """Fetch all vectors for clustering operations (e.g., HDBSCAN).

        Optimized for memory efficiency with large datasets.

        Args:
            namespace: Filter to specific namespace.
            max_memories: Maximum memories to fetch.

        Returns:
            Tuple of (memory_ids, vectors_array).

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            # Build query selecting only needed columns
            search = self.table.search()

            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

            # Select only id and vector to minimize memory usage
            search = search.select(["id", "vector"]).limit(max_memories)

            results = search.to_list()

            if not results:
                return [], np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

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
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Get memories as Arrow table for efficient processing.

        Arrow tables enable zero-copy data sharing and efficient columnar
        operations. Use this for large-scale analytics.

        Args:
            namespace: Filter to specific namespace.
            columns: Columns to select (None = all).

        Returns:
            PyArrow Table with selected data.

        Raises:
            StorageError: If database operation fails.
        """
        try:
            search = self.table.search()

            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

            if columns:
                search = search.select(columns)

            return search.to_arrow()

        except Exception as e:
            raise StorageError(f"Failed to get Arrow table: {e}") from e

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
                # Use count_rows with filter predicate for efficiency
                count: int = self.table.count_rows(f"namespace = '{safe_ns}'")
                return count
            count = self.table.count_rows()
            return count
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to count memories: {e}") from e

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
            with self._namespace_cache_lock:
                if (
                    self._cached_namespaces is not None
                    and (now - self._namespace_cache_time) <= self._NAMESPACE_CACHE_TTL
                ):
                    return sorted(self._cached_namespaces)

            # Fetch from database (outside lock to avoid blocking)
            results = self.table.search().select(["namespace"]).to_list()
            namespaces = set(r["namespace"] for r in results)

            # Double-checked locking: re-check and update atomically
            with self._namespace_cache_lock:
                # Another thread may have populated cache while we were fetching
                if self._cached_namespaces is None:
                    self._cached_namespaces = namespaces
                    self._namespace_cache_time = now
                # Return fresh data regardless (it's at least as current)
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

    # ========================================================================
    # Backup & Export
    # ========================================================================

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
            arrow_table = self.get_vectors_as_arrow(namespace=namespace)

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

            # Insert in batches
            imported = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                # Convert to format expected by insert
                prepared = []
                for r in batch:
                    prepared.append({
                        "content": r["content"],
                        "vector": r["vector"],
                        "namespace": r["namespace"],
                        "tags": r.get("tags", []),
                        "importance": r.get("importance", 0.5),
                        "source": r.get("source", "import"),
                        "metadata": r.get("metadata", {}),
                    })
                self.table.add(prepared)
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
