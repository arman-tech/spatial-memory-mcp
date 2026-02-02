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

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterator
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
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from spatial_memory.core.connection_pool import ConnectionPool
from spatial_memory.core.db_idempotency import IdempotencyManager, IdempotencyRecord
from spatial_memory.core.db_indexes import IndexManager
from spatial_memory.core.db_migrations import CURRENT_SCHEMA_VERSION, MigrationManager
from spatial_memory.core.db_search import SearchManager
from spatial_memory.core.db_versioning import VersionManager
from spatial_memory.core.errors import (
    DimensionMismatchError,
    FileLockError,
    MemoryNotFoundError,
    PartialBatchInsertError,
    StorageError,
    ValidationError,
)
from spatial_memory.core.filesystem import (
    detect_filesystem_type,
    get_filesystem_warning_message,
    is_network_filesystem,
)
from spatial_memory.core.utils import utc_now

# Import centralized validation functions
from spatial_memory.core.validation import (
    sanitize_string as _sanitize_string_impl,
)
from spatial_memory.core.validation import (
    validate_metadata as _validate_metadata_impl,
)
from spatial_memory.core.validation import (
    validate_namespace as _validate_namespace_impl,
)
from spatial_memory.core.validation import (
    validate_tags as _validate_tags_impl,
)
from spatial_memory.core.validation import (
    validate_uuid as _validate_uuid_impl,
)

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

# Global connection pool instance
_connection_pool = ConnectionPool(max_size=10)


def set_connection_pool_max_size(max_size: int) -> None:
    """Set the maximum connection pool size.

    Args:
        max_size: Maximum number of connections to cache.
    """
    _connection_pool.max_size = max_size


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
    return _connection_pool.get_or_create(path_key, read_consistency_interval_ms)


def clear_connection_cache() -> None:
    """Clear the connection cache, properly closing connections.

    Should be called during shutdown or testing cleanup.
    """
    _connection_pool.close_all()


def invalidate_connection(storage_path: Path) -> bool:
    """Invalidate a specific cached connection.

    Use when a database connection becomes stale (e.g., database was
    deleted and recreated externally).

    Args:
        storage_path: Path to the database to invalidate.

    Returns:
        True if a connection was invalidated, False if not found in cache.
    """
    path_key = str(storage_path.absolute())
    return _connection_pool.invalidate(path_key)


# ============================================================================
# Retry Decorator
# ============================================================================

# Default retry settings (can be overridden per-call)
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5


def retry_on_storage_error(
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
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


def with_write_lock(func: F) -> F:
    """Decorator to acquire write lock for mutation operations.

    Serializes write operations per Database instance to prevent
    LanceDB version conflicts during concurrent writes.

    Uses RLock to allow nested calls (e.g., bulk_import -> insert_batch).
    """
    @wraps(func)
    def wrapper(self: Database, *args: Any, **kwargs: Any) -> Any:
        with self._write_lock:
            return func(self, *args, **kwargs)
    return cast(F, wrapper)


def with_stale_connection_recovery(func: F) -> F:
    """Decorator to auto-recover from stale connection errors.

    Detects when a database operation fails due to stale metadata
    (e.g., database was recreated while connection was cached),
    reconnects, and retries the operation once.
    """
    @wraps(func)
    def wrapper(self: Database, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if _connection_pool.is_stale_connection_error(e):
                logger.warning(
                    f"Stale connection detected in {func.__name__}, reconnecting..."
                )
                self.reconnect()
                return func(self, *args, **kwargs)
            raise
    return cast(F, wrapper)


# ============================================================================
# Cross-Process Lock Manager
# ============================================================================


class ProcessLockManager:
    """Cross-process file lock manager with reentrant support.

    Wraps FileLock with thread-local depth tracking to support nested calls
    (e.g., bulk_import() -> insert_batch()). Each thread can re-acquire the
    lock without blocking.

    Thread Safety:
        - Lock depth is tracked per-thread using threading.local
        - The underlying FileLock handles cross-process synchronization
        - Multiple threads in the same process can hold the lock via RLock behavior

    Example:
        lock = ProcessLockManager(Path("/tmp/db.lock"), timeout=30.0)
        with lock:
            # Protected region
            with lock:  # Nested call - same thread can re-acquire
                pass
    """

    def __init__(
        self,
        lock_path: Path,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
        enabled: bool = True,
    ) -> None:
        """Initialize the process lock manager.

        Args:
            lock_path: Path to the lock file.
            timeout: Maximum seconds to wait for lock acquisition.
            poll_interval: Seconds between lock acquisition attempts.
            enabled: If False, all lock operations are no-ops.
        """
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.enabled = enabled

        # Create FileLock only if enabled
        self._lock: FileLock | None = None
        if enabled:
            try:
                self._lock = FileLock(str(lock_path), timeout=timeout)
            except Exception as e:
                # Fallback to disabled mode if lock file can't be created
                # (e.g., read-only filesystem)
                logger.warning(
                    f"Could not create file lock at {lock_path}: {e}. "
                    "Falling back to disabled mode."
                )
                self.enabled = False

        # Thread-local storage for lock depth tracking
        self._local = threading.local()

    def _get_depth(self) -> int:
        """Get current lock depth for this thread."""
        return getattr(self._local, "depth", 0)

    def _set_depth(self, depth: int) -> None:
        """Set lock depth for this thread."""
        self._local.depth = depth

    def acquire(self) -> bool:
        """Acquire the lock (reentrant for same thread).

        Returns:
            True if lock was newly acquired, False if already held by this thread.

        Raises:
            FileLockError: If lock cannot be acquired within timeout.
        """
        if not self.enabled or self._lock is None:
            return True

        depth = self._get_depth()
        if depth > 0:
            # Already held by this thread - increment depth
            self._set_depth(depth + 1)
            return False  # Not newly acquired

        try:
            self._lock.acquire(timeout=self.timeout, poll_interval=self.poll_interval)
            self._set_depth(1)
            return True
        except FileLockTimeout:
            raise FileLockError(
                lock_path=str(self.lock_path),
                timeout=self.timeout,
                message=(
                    f"Timed out waiting {self.timeout}s for file lock at "
                    f"{self.lock_path}. Another process may be holding the lock."
                ),
            )

    def release(self) -> bool:
        """Release the lock (decrements depth, releases when depth reaches 0).

        Returns:
            True if lock was released, False if still held (depth > 0).
        """
        if not self.enabled or self._lock is None:
            return True

        depth = self._get_depth()
        if depth <= 0:
            return True  # Not holding the lock

        if depth == 1:
            self._lock.release()
            self._set_depth(0)
            return True
        else:
            self._set_depth(depth - 1)
            return False  # Still holding

    def __enter__(self) -> ProcessLockManager:
        """Enter context manager - acquire lock."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager - release lock."""
        self.release()


def with_process_lock(func: F) -> F:
    """Decorator to acquire process-level file lock for write operations.

    Must be applied BEFORE (outer) @with_write_lock to ensure:
    1. Cross-process lock acquired first
    2. Then intra-process thread lock
    3. Releases in reverse order

    Usage:
        @with_process_lock  # Outer - cross-process
        @with_write_lock    # Inner - intra-process
        def insert(self, ...):
            ...
    """
    @wraps(func)
    def wrapper(self: Database, *args: Any, **kwargs: Any) -> Any:
        if self._process_lock is None:
            return func(self, *args, **kwargs)
        with self._process_lock:
            return func(self, *args, **kwargs)
    return cast(F, wrapper)


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


# Backward compatibility aliases - use centralized validation module
_sanitize_string = _sanitize_string_impl
_validate_uuid = _validate_uuid_impl


def _get_index_attr(idx: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from an index object (handles both dict and IndexConfig).

    LanceDB 0.27+ returns IndexConfig objects, while older versions use dicts.

    Args:
        idx: Index object (dict or IndexConfig).
        attr: Attribute name to retrieve.
        default: Default value if attribute not found.

    Returns:
        The attribute value or default.
    """
    if isinstance(idx, dict):
        return idx.get(attr, default)
    return getattr(idx, attr, default)


_validate_namespace = _validate_namespace_impl
_validate_tags = _validate_tags_impl
_validate_metadata = _validate_metadata_impl


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
        max_retry_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        read_consistency_interval_ms: int = 0,
        index_wait_timeout_seconds: float = 30.0,
        fts_stem: bool = True,
        fts_remove_stop_words: bool = True,
        fts_language: str = "English",
        index_type: str = "IVF_PQ",
        hnsw_m: int = 20,
        hnsw_ef_construction: int = 300,
        enable_memory_expiration: bool = False,
        default_memory_ttl_days: int | None = None,
        filelock_enabled: bool = True,
        filelock_timeout: float = 30.0,
        filelock_poll_interval: float = 0.1,
        acknowledge_network_filesystem_risk: bool = False,
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
            fts_stem: Enable stemming in FTS (running -> run).
            fts_remove_stop_words: Remove stop words in FTS (the, is, etc.).
            filelock_enabled: Enable cross-process file locking.
            filelock_timeout: Timeout in seconds for acquiring filelock.
            filelock_poll_interval: Interval between lock acquisition attempts.
            fts_language: Language for FTS stemming.
            index_type: Vector index type (IVF_PQ, IVF_FLAT, or HNSW_SQ).
            hnsw_m: HNSW connections per node (4-64).
            hnsw_ef_construction: HNSW build-time search width (100-1000).
            enable_memory_expiration: Enable automatic memory expiration.
            default_memory_ttl_days: Default TTL for memories in days (None = no expiration).
            acknowledge_network_filesystem_risk: Suppress network filesystem warnings.
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
        self.fts_stem = fts_stem
        self.fts_remove_stop_words = fts_remove_stop_words
        self.fts_language = fts_language
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.enable_memory_expiration = enable_memory_expiration
        self.default_memory_ttl_days = default_memory_ttl_days
        self.filelock_enabled = filelock_enabled
        self.filelock_timeout = filelock_timeout
        self.filelock_poll_interval = filelock_poll_interval
        self.acknowledge_network_filesystem_risk = acknowledge_network_filesystem_risk
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
        # Write lock for serializing mutations (prevents LanceDB version conflicts)
        self._write_lock = threading.RLock()
        # Cross-process lock (initialized in connect())
        self._process_lock: ProcessLockManager | None = None
        # Auto-compaction tracking
        self._modification_count: int = 0
        self._auto_compaction_threshold: int = 100  # Compact after this many modifications
        self._auto_compaction_enabled: bool = True
        # Version manager (initialized in connect())
        self._version_manager: VersionManager | None = None
        # Index manager (initialized in connect())
        self._index_manager: IndexManager | None = None
        # Search manager (initialized in connect())
        self._search_manager: SearchManager | None = None
        # Idempotency manager (initialized in connect())
        self._idempotency_manager: IdempotencyManager | None = None

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

            # Check for network filesystem and warn if detected
            if not self.acknowledge_network_filesystem_risk:
                if is_network_filesystem(self.storage_path):
                    fs_type = detect_filesystem_type(self.storage_path)
                    warning_msg = get_filesystem_warning_message(fs_type, self.storage_path)
                    logger.warning(warning_msg)

            # Initialize cross-process lock manager
            if self.filelock_enabled:
                lock_path = self.storage_path / ".spatial-memory.lock"
                self._process_lock = ProcessLockManager(
                    lock_path=lock_path,
                    timeout=self.filelock_timeout,
                    poll_interval=self.filelock_poll_interval,
                    enabled=self.filelock_enabled,
                )
            else:
                self._process_lock = None

            # Use connection pooling with read consistency support
            self._db = _get_or_create_connection(
                self.storage_path,
                read_consistency_interval_ms=self.read_consistency_interval_ms,
            )
            self._ensure_table()
            # Initialize remaining managers (IndexManager already initialized in _ensure_table)
            self._version_manager = VersionManager(self)
            self._search_manager = SearchManager(self)
            self._idempotency_manager = IdempotencyManager(self)
            logger.info(f"Connected to LanceDB at {self.storage_path}")

            # Check for pending schema migrations
            self._check_pending_migrations()
        except Exception as e:
            raise StorageError(f"Failed to connect to database: {e}") from e

    def _check_pending_migrations(self) -> None:
        """Check for pending migrations and warn if any exist.

        This method checks the schema version and logs a warning if there
        are pending migrations. It does not auto-apply migrations - that
        requires explicit user action via the CLI.
        """
        try:
            manager = MigrationManager(self, embeddings=None)
            manager.register_builtin_migrations()

            current_version = manager.get_current_version()
            pending = manager.get_pending_migrations()

            if pending:
                pending_versions = [m.version for m in pending]
                logger.warning(
                    f"Database schema version {current_version} is outdated. "
                    f"{len(pending)} migration(s) pending: {', '.join(pending_versions)}. "
                    f"Target version: {CURRENT_SCHEMA_VERSION}. "
                    f"Run 'spatial-memory migrate' to apply migrations."
                )
        except Exception as e:
            # Don't fail connection due to migration check errors
            logger.debug(f"Migration check skipped: {e}")

    def _ensure_table(self) -> None:
        """Ensure the memories table exists with appropriate indexes.

        Uses retry logic to handle race conditions when multiple processes
        attempt to create/open the table simultaneously.
        """
        if self._db is None:
            raise StorageError("Database not connected")

        max_retries = 3
        retry_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                existing_tables_result = self._db.list_tables()
                # Handle both old (list) and new (object with .tables) LanceDB API
                if hasattr(existing_tables_result, 'tables'):
                    existing_tables = existing_tables_result.tables
                else:
                    existing_tables = existing_tables_result

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
                        pa.field("expires_at", pa.timestamp("us")),  # TTL support - nullable
                    ])
                    try:
                        self._table = self._db.create_table("memories", schema=schema)
                        logger.info("Created memories table")
                    except Exception as create_err:
                        # Table might have been created by another process
                        if "already exists" in str(create_err).lower():
                            logger.debug("Table created by another process, opening it")
                            self._table = self._db.open_table("memories")
                        else:
                            raise

                    # Initialize IndexManager immediately after table is set
                    self._index_manager = IndexManager(self)

                    # Create FTS index on new table if enabled
                    if self.enable_fts:
                        self._index_manager.create_fts_index()
                else:
                    self._table = self._db.open_table("memories")
                    logger.debug("Opened existing memories table")

                    # Initialize IndexManager immediately after table is set
                    self._index_manager = IndexManager(self)

                    # Check existing indexes
                    self._index_manager.check_existing_indexes()

                # Success - exit retry loop
                return

            except Exception as e:
                error_msg = str(e).lower()
                # Retry on transient race conditions
                if attempt < max_retries - 1 and (
                    "not found" in error_msg
                    or "does not exist" in error_msg
                    or "already exists" in error_msg
                ):
                    logger.debug(
                        f"Table operation failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {retry_delay}s: {e}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

    def _check_existing_indexes(self) -> None:
        """Check which indexes already exist. Delegates to IndexManager."""
        if self._index_manager is None:
            raise StorageError("Database not connected")
        self._index_manager.check_existing_indexes()
        # Sync local state for backward compatibility
        self._has_vector_index = self._index_manager.has_vector_index
        self._has_fts_index = self._index_manager.has_fts_index

    def _create_fts_index(self) -> None:
        """Create FTS index. Delegates to IndexManager."""
        if self._index_manager is None:
            raise StorageError("Database not connected")
        self._index_manager.create_fts_index()
        # Sync local state for backward compatibility
        self._has_fts_index = self._index_manager.has_fts_index

    @property
    def table(self) -> LanceTable:
        """Get the memories table, connecting if needed."""
        if self._table is None:
            self.connect()
        assert self._table is not None  # connect() sets this or raises
        return self._table

    def close(self) -> None:
        """Close the database connection and remove from pool.

        This invalidates the pooled connection so that subsequent
        Database instances will create fresh connections.
        """
        # Invalidate pooled connection first
        invalidate_connection(self.storage_path)

        # Clear local state
        self._table = None
        self._db = None
        self._has_vector_index = None
        self._has_fts_index = None
        self._version_manager = None
        self._index_manager = None
        self._search_manager = None
        self._idempotency_manager = None
        with self._cache_lock:
            self._cached_row_count = None
            self._count_cache_time = 0.0
        with self._namespace_cache_lock:
            self._cached_namespaces = None
            self._namespace_cache_time = 0.0
        logger.debug("Database connection closed and removed from pool")

    def reconnect(self) -> None:
        """Invalidate cached connection and reconnect.

        Use when the database connection becomes stale (e.g., database was
        deleted and recreated externally, or metadata references missing files).

        This method:
        1. Closes the current connection state
        2. Invalidates the pooled connection for this path
        3. Creates a fresh connection to the database
        """
        logger.info(f"Reconnecting to database at {self.storage_path}")
        self.close()
        invalidate_connection(self.storage_path)
        self.connect()

    def _is_stale_connection_error(self, error: Exception) -> bool:
        """Check if an error indicates a stale/corrupted connection.

        Args:
            error: The exception to check.

        Returns:
            True if the error indicates a stale connection.
        """
        return ConnectionPool.is_stale_connection_error(error)

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

    def _track_modification(self, count: int = 1) -> None:
        """Track database modifications and trigger auto-compaction if threshold reached.

        Args:
            count: Number of modifications to track (default 1).
        """
        if not self._auto_compaction_enabled:
            return

        self._modification_count += count
        if self._modification_count >= self._auto_compaction_threshold:
            # Reset counter before compacting to avoid re-triggering
            self._modification_count = 0
            try:
                stats = self._get_table_stats()
                # Only compact if there are enough fragments to justify it
                if stats.get("num_small_fragments", 0) >= 5:
                    logger.info(
                        f"Auto-compaction triggered after {self._auto_compaction_threshold} "
                        f"modifications ({stats.get('num_small_fragments', 0)} small fragments)"
                    )
                    self.table.compact_files()
                    logger.debug("Auto-compaction completed")
            except Exception as e:
                # Don't fail operations due to compaction issues
                logger.debug(f"Auto-compaction skipped: {e}")

    def set_auto_compaction(
        self,
        enabled: bool = True,
        threshold: int | None = None,
    ) -> None:
        """Configure auto-compaction behavior.

        Args:
            enabled: Whether auto-compaction is enabled.
            threshold: Number of modifications before auto-compact (default: 100).
        """
        self._auto_compaction_enabled = enabled
        if threshold is not None:
            if threshold < 10:
                raise ValueError("Auto-compaction threshold must be at least 10")
            self._auto_compaction_threshold = threshold

    # ========================================================================
    # Index Management (delegates to IndexManager)
    # ========================================================================

    def create_vector_index(self, force: bool = False) -> bool:
        """Create vector index for similarity search. Delegates to IndexManager.

        Args:
            force: Force index creation regardless of dataset size.

        Returns:
            True if index was created, False if skipped.

        Raises:
            StorageError: If index creation fails.
        """
        if self._index_manager is None:
            raise StorageError("Database not connected")
        result = self._index_manager.create_vector_index(force=force)
        # Sync local state only when index was created or modified
        if result:
            self._has_vector_index = self._index_manager.has_vector_index
        return result

    def create_scalar_indexes(self) -> None:
        """Create scalar indexes for frequently filtered columns. Delegates to IndexManager.

        Raises:
            StorageError: If index creation fails critically.
        """
        if self._index_manager is None:
            raise StorageError("Database not connected")
        self._index_manager.create_scalar_indexes()

    def ensure_indexes(self, force: bool = False) -> dict[str, bool]:
        """Ensure all appropriate indexes exist. Delegates to IndexManager.

        Args:
            force: Force index creation regardless of thresholds.

        Returns:
            Dict indicating which indexes were created.
        """
        if self._index_manager is None:
            raise StorageError("Database not connected")
        results = self._index_manager.ensure_indexes(force=force)
        # Sync local state for backward compatibility
        self._has_vector_index = self._index_manager.has_vector_index
        self._has_fts_index = self._index_manager.has_fts_index
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

    @with_stale_connection_recovery
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
                        name=str(_get_index_attr(idx, "name", "unknown")),
                        index_type=str(_get_index_attr(idx, "index_type", "unknown")),
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

    @with_process_lock
    @with_write_lock
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
        tags = _validate_tags(tags)
        metadata = _validate_metadata(metadata)
        if not content or len(content) > 100000:
            raise ValidationError("Content must be between 1 and 100000 characters")
        if not 0.0 <= importance <= 1.0:
            raise ValidationError("Importance must be between 0.0 and 1.0")

        # Validate vector dimensions
        if len(vector) != self.embedding_dim:
            raise DimensionMismatchError(
                expected_dim=self.embedding_dim,
                actual_dim=len(vector),
            )

        memory_id = str(uuid.uuid4())
        now = utc_now()

        # Calculate expires_at if default TTL is configured
        expires_at = None
        if self.default_memory_ttl_days is not None:
            expires_at = now + timedelta(days=self.default_memory_ttl_days)

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
            "expires_at": expires_at,
        }

        try:
            self.table.add([record])
            self._invalidate_count_cache()
            self._track_modification()
            self._invalidate_namespace_cache()
            logger.debug(f"Inserted memory {memory_id}")
            return memory_id
        except Exception as e:
            raise StorageError(f"Failed to insert memory: {e}") from e

    # Maximum batch size to prevent memory exhaustion
    MAX_BATCH_SIZE = 10_000

    @with_process_lock
    @with_write_lock
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
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
            batch = records[i:i + batch_size]
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
                if len(vector_list) != self.embedding_dim:
                    raise DimensionMismatchError(
                        expected_dim=self.embedding_dim,
                        actual_dim=len(vector_list),
                        record_index=i + len(memory_ids),
                    )

                # Calculate expires_at if default TTL is configured
                expires_at = None
                if self.default_memory_ttl_days is not None:
                    expires_at = now + timedelta(days=self.default_memory_ttl_days)

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
                    "expires_at": expires_at,
                }
                prepared_records.append(prepared)

            try:
                self.table.add(prepared_records)
                all_ids.extend(memory_ids)
                self._invalidate_count_cache()
                self._track_modification(len(memory_ids))
                self._invalidate_namespace_cache()
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
            self.table.delete(f"id IN ({id_list})")
            self._invalidate_count_cache()
            self._track_modification(len(memory_ids))
            self._invalidate_namespace_cache()
            logger.debug(f"Rolled back {len(memory_ids)} records")
            return None
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return e

    @with_stale_connection_recovery
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
            results = self.table.search().where(f"id IN ({id_list})").to_list()

            # Build result map
            result_map: dict[str, dict[str, Any]] = {}
            for record in results:
                # Deserialize metadata
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}
                result_map[record["id"]] = record

            return result_map
        except Exception as e:
            raise StorageError(f"Failed to batch get memories: {e}") from e

    @with_process_lock
    @with_write_lock
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
                self.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([existing])
            )
            logger.debug(f"Updated memory {memory_id} (atomic merge_insert)")
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

    @with_process_lock
    @with_write_lock
    def update_batch(
        self, updates: list[tuple[str, dict[str, Any]]]
    ) -> tuple[int, list[str]]:
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
            all_records = self.table.search().where(f"id IN ({id_list})").to_list()
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
                self.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records_to_update)
            )
            success_count = len(records_to_update)
            logger.debug(
                f"Batch updated {success_count}/{len(updates)} memories "
                "(atomic merge_insert)"
            )
            return success_count, failed_ids
        except Exception as e:
            logger.error(f"Failed to batch update: {e}")
            raise StorageError(f"Failed to batch update: {e}") from e

    @with_process_lock
    @with_write_lock
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
            self._track_modification()
            self._invalidate_namespace_cache()
            logger.debug(f"Deleted memory {memory_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete memory: {e}") from e

    @with_process_lock
    @with_write_lock
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
            self._track_modification()
            self._invalidate_namespace_cache()
            count_after: int = self.table.count_rows()
            deleted = count_before - count_after
            logger.debug(f"Deleted {deleted} memories in namespace '{namespace}'")
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete by namespace: {e}") from e

    @with_process_lock
    @with_write_lock
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
            count: int = self.table.count_rows()
            if count > 0:
                # Delete all rows - use simpler predicate that definitely matches
                self.table.delete("true")

                # Verify deletion worked
                remaining = self.table.count_rows()
                if remaining > 0:
                    logger.warning(
                        f"clear_all: {remaining} records remain after delete, "
                        f"attempting cleanup again"
                    )
                    # Try alternative delete approach
                    self.table.delete("id IS NOT NULL")

            self._invalidate_count_cache()
            self._track_modification()
            self._invalidate_namespace_cache()

            # Reset index tracking flags for test isolation
            if reset_indexes:
                self._has_vector_index = None
                self._has_fts_index = False
                self._has_scalar_indexes = False

            logger.debug(f"Cleared all {count} memories from database")
            return count
        except Exception as e:
            raise StorageError(f"Failed to clear all memories: {e}") from e

    @with_process_lock
    @with_write_lock
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
            # Check if source namespace exists
            existing = self.get_namespaces()
            if old_namespace not in existing:
                raise NamespaceNotFoundError(old_namespace)

            # Short-circuit if renaming to same namespace (no-op)
            if old_namespace == new_namespace:
                count = self.count(namespace=old_namespace)
                logger.debug(f"Namespace '{old_namespace}' renamed to itself ({count} records)")
                return count

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
                    self.table.search()
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
                        self.table.merge_insert("id")
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
                        rollback_error = self._rollback_namespace_rename(
                            renamed_ids, old_namespace
                        )
                        if rollback_error:
                            raise StorageError(
                                f"Namespace rename failed at batch {iteration} and "
                                f"rollback also failed. {len(renamed_ids)} records may be "
                                f"in inconsistent state (partially in '{new_namespace}'). "
                                f"Original error: {batch_error}. Rollback error: {rollback_error}"
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

            self._invalidate_namespace_cache()
            logger.debug(
                f"Renamed {updated} memories from '{old_namespace}' to '{new_namespace}'"
            )
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
                batch_ids = memory_ids[i:i + batch_size]
                id_list = ", ".join(f"'{_sanitize_string(mid)}'" for mid in batch_ids)

                # Fetch records that need rollback
                records = (
                    self.table.search()
                    .where(f"id IN ({id_list})")
                    .to_list()
                )

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
                    self.table.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(records)
                )

            self._invalidate_namespace_cache()
            logger.debug(f"Rolled back {len(memory_ids)} records to namespace '{target_namespace}'")
            return None

        except Exception as e:
            logger.error(f"Namespace rename rollback failed: {e}")
            return e

    @with_stale_connection_recovery
    def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Uses efficient LanceDB queries for aggregations.

        Args:
            namespace: Filter stats to specific namespace (None = all).

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
            ns_arrow = self.table.search().select(["namespace"]).to_arrow()

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
                self.table.search()
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
                self.table.search()
                .where(filter_expr)
                .select(["created_at"])
                .limit(1)
                .to_list()
            )
            oldest = oldest_records[0]["created_at"] if oldest_records else None

            # Get newest memory - need to fetch more and find max since LanceDB
            # doesn't support ORDER BY DESC efficiently
            # Sample up to 1000 records for stats to avoid loading everything
            sample_size = min(memory_count, 1000)
            sample_records = (
                self.table.search()
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

    def get_all_for_export(
        self,
        namespace: str | None = None,
        batch_size: int = 1000,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Stream all memories for export in batches.

        Memory-efficient export using generator pattern.

        Args:
            namespace: Optional namespace filter.
            batch_size: Records per batch.

        Yields:
            Batches of memory dictionaries.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        try:
            search = self.table.search()

            if namespace is not None:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'")

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

    @with_process_lock
    @with_write_lock
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
                    ids = self.insert_batch(batch, batch_size=batch_size)
                    all_ids.extend(ids)
                    imported += len(ids)
                    batch = []

            # Import remaining
            if batch:
                ids = self.insert_batch(batch, batch_size=batch_size)
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

    @with_process_lock
    @with_write_lock
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
                self.table.search()
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
            self.table.delete(delete_expr)

            self._invalidate_count_cache()
            self._track_modification()
            self._invalidate_namespace_cache()

            logger.debug(f"Batch deleted {len(existing_ids)} memories")
            return (len(existing_ids), existing_ids)
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete batch: {e}") from e

    @with_process_lock
    @with_write_lock
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
            all_records = self.table.search().where(f"id IN ({id_list})").to_list()
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
                self.table.merge_insert("id")
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

    def _create_retry_decorator(self) -> Callable[[F], F]:
        """Create a retry decorator using instance settings."""
        return retry_on_storage_error(
            max_attempts=self.max_retry_attempts,
            backoff=self.retry_backoff_seconds,
        )

    # ========================================================================
    # Search Operations (delegates to SearchManager)
    # ========================================================================

    def _calculate_search_params(
        self,
        count: int,
        limit: int,
        nprobes_override: int | None = None,
        refine_factor_override: int | None = None,
    ) -> tuple[int, int]:
        """Calculate optimal search parameters. Delegates to SearchManager."""
        if self._search_manager is None:
            raise StorageError("Database not connected")
        return self._search_manager.calculate_search_params(
            count, limit, nprobes_override, refine_factor_override
        )

    @with_stale_connection_recovery
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def vector_search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        min_similarity: float = 0.0,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        include_vector: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for similar memories by vector. Delegates to SearchManager.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.
            min_similarity: Minimum similarity threshold (0-1).
            nprobes: Number of partitions to search.
            refine_factor: Re-rank top (refine_factor * limit) for accuracy.
            include_vector: Whether to include vector embeddings in results.

        Returns:
            List of memory records with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if self._search_manager is None:
            raise StorageError("Database not connected")
        return self._search_manager.vector_search(
            query_vector=query_vector,
            limit=limit,
            namespace=namespace,
            min_similarity=min_similarity,
            nprobes=nprobes,
            refine_factor=refine_factor,
            include_vector=include_vector,
        )

    @with_stale_connection_recovery
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def batch_vector_search_native(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        min_similarity: float = 0.0,
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Batch search using native LanceDB. Delegates to SearchManager.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum number of results per query.
            namespace: Filter to specific namespace.
            min_similarity: Minimum similarity threshold (0-1).
            include_vector: Whether to include vector embeddings in results.

        Returns:
            List of result lists, one per query vector.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if self._search_manager is None:
            raise StorageError("Database not connected")
        return self._search_manager.batch_vector_search_native(
            query_vectors=query_vectors,
            limit_per_query=limit_per_query,
            namespace=namespace,
            min_similarity=min_similarity,
            include_vector=include_vector,
        )

    @with_stale_connection_recovery
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        alpha: float = 0.5,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector and keyword. Delegates to SearchManager.

        Args:
            query: Text query for full-text search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results.
            namespace: Filter to namespace.
            alpha: Balance between vector (1.0) and keyword (0.0).
            min_similarity: Minimum similarity threshold.

        Returns:
            List of memory records with combined scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if self._search_manager is None:
            raise StorageError("Database not connected")
        return self._search_manager.hybrid_search(
            query=query,
            query_vector=query_vector,
            limit=limit,
            namespace=namespace,
            alpha=alpha,
            min_similarity=min_similarity,
        )

    @with_stale_connection_recovery
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def batch_vector_search(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        parallel: bool = False,  # Deprecated
        max_workers: int = 4,  # Deprecated
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Search using multiple query vectors. Delegates to SearchManager.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum results per query vector.
            namespace: Filter to specific namespace.
            parallel: Deprecated, kept for backward compatibility.
            max_workers: Deprecated, kept for backward compatibility.
            include_vector: Whether to include vector embeddings in results.

        Returns:
            List of result lists (one per query vector).

        Raises:
            StorageError: If database operation fails.
        """
        if self._search_manager is None:
            raise StorageError("Database not connected")
        return self._search_manager.batch_vector_search(
            query_vectors=query_vectors,
            limit_per_query=limit_per_query,
            namespace=namespace,
            parallel=parallel,
            max_workers=max_workers,
            include_vector=include_vector,
        )

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

    @with_stale_connection_recovery
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

    @with_stale_connection_recovery
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

    @with_process_lock
    @with_write_lock
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
        # Note: self.update() also has @with_process_lock and @with_write_lock,
        # but both support reentrancy within the same thread (no deadlock):
        # - ProcessLockManager tracks depth via threading.local
        # - RLock allows same thread to re-acquire
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

            # After reading from parquet, serialize metadata back to JSON string
            # Parquet may read metadata as dict/struct, but the database expects JSON string
            for record in records:
                if "metadata" in record and isinstance(record["metadata"], dict):
                    record["metadata"] = json.dumps(record["metadata"])

            # Insert in batches
            imported = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                # Convert to format expected by insert
                prepared = []
                for r in batch:
                    # Ensure metadata is a JSON string for storage
                    metadata = r.get("metadata", {})
                    if isinstance(metadata, dict):
                        metadata = json.dumps(metadata)
                    elif metadata is None:
                        metadata = "{}"

                    prepared.append({
                        "content": r["content"],
                        "vector": r["vector"],
                        "namespace": r["namespace"],
                        "tags": r.get("tags", []),
                        "importance": r.get("importance", 0.5),
                        "source": r.get("source", "import"),
                        "metadata": metadata,
                        "expires_at": r.get("expires_at"),  # Preserve TTL from source
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

    # ========================================================================
    # TTL (Time-To-Live) Management
    # ========================================================================

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
        existing = self.get(memory_id)

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
                self.table.merge_insert("id")
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
        if not self.enable_memory_expiration:
            logger.debug("Memory expiration is disabled, skipping cleanup")
            return 0

        try:
            now = utc_now()
            count_before = self.table.count_rows()

            # Delete expired memories using timestamp comparison
            # LanceDB uses ISO 8601 format for timestamp comparisons
            predicate = (
                f"expires_at IS NOT NULL AND expires_at < timestamp '{now.isoformat()}'"
            )
            self.table.delete(predicate)

            count_after = self.table.count_rows()
            deleted = count_before - count_after

            if deleted > 0:
                self._invalidate_count_cache()
                self._track_modification(deleted)
                self._invalidate_namespace_cache()
                logger.info(f"Cleaned up {deleted} expired memories")

            return deleted
        except Exception as e:
            raise StorageError(f"Failed to cleanup expired memories: {e}") from e

    # ========================================================================
    # Snapshot / Version Management (delegated to VersionManager)
    # ========================================================================

    def create_snapshot(self, tag: str) -> int:
        """Create a named snapshot of the current table state.

        Delegates to VersionManager. See VersionManager.create_snapshot for details.
        """
        if self._version_manager is None:
            raise StorageError("Database not connected")
        return self._version_manager.create_snapshot(tag)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List available versions/snapshots.

        Delegates to VersionManager. See VersionManager.list_snapshots for details.
        """
        if self._version_manager is None:
            raise StorageError("Database not connected")
        return self._version_manager.list_snapshots()

    def restore_snapshot(self, version: int) -> None:
        """Restore table to a specific version.

        Delegates to VersionManager. See VersionManager.restore_snapshot for details.
        """
        if self._version_manager is None:
            raise StorageError("Database not connected")
        self._version_manager.restore_snapshot(version)

    def get_current_version(self) -> int:
        """Get the current table version number.

        Delegates to VersionManager. See VersionManager.get_current_version for details.
        """
        if self._version_manager is None:
            raise StorageError("Database not connected")
        return self._version_manager.get_current_version()

    # ========================================================================
    # Idempotency Key Management (delegates to IdempotencyManager)
    # ========================================================================

    @property
    def idempotency_table(self) -> LanceTable:
        """Get the idempotency keys table. Delegates to IdempotencyManager."""
        if self._idempotency_manager is None:
            raise StorageError("Database not connected")
        return self._idempotency_manager.idempotency_table

    def get_by_idempotency_key(self, key: str) -> IdempotencyRecord | None:
        """Look up an idempotency record by key. Delegates to IdempotencyManager.

        Args:
            key: The idempotency key to look up.

        Returns:
            IdempotencyRecord if found and not expired, None otherwise.

        Raises:
            StorageError: If database operation fails.
        """
        if self._idempotency_manager is None:
            raise StorageError("Database not connected")
        return self._idempotency_manager.get_by_idempotency_key(key)

    @with_process_lock
    @with_write_lock
    def store_idempotency_key(
        self,
        key: str,
        memory_id: str,
        ttl_hours: float = 24.0,
    ) -> None:
        """Store an idempotency key mapping. Delegates to IdempotencyManager.

        Args:
            key: The idempotency key.
            memory_id: The memory ID that was created.
            ttl_hours: Time-to-live in hours (default: 24 hours).

        Raises:
            ValidationError: If inputs are invalid.
            StorageError: If database operation fails.
        """
        if self._idempotency_manager is None:
            raise StorageError("Database not connected")
        self._idempotency_manager.store_idempotency_key(key, memory_id, ttl_hours)

    @with_process_lock
    @with_write_lock
    def cleanup_expired_idempotency_keys(self) -> int:
        """Remove expired idempotency keys. Delegates to IdempotencyManager.

        Returns:
            Number of keys removed.

        Raises:
            StorageError: If cleanup fails.
        """
        if self._idempotency_manager is None:
            raise StorageError("Database not connected")
        return self._idempotency_manager.cleanup_expired_idempotency_keys()
