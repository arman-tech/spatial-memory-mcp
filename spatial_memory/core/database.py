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

import logging
import threading
import time
from collections.abc import Callable, Generator, Iterator
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import lancedb
import lancedb.index
import numpy as np
import pyarrow as pa
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from spatial_memory.core.connection_pool import ConnectionPool
from spatial_memory.core.db_crud import CrudManager
from spatial_memory.core.db_export import ExportImportManager
from spatial_memory.core.db_idempotency import IdempotencyManager, IdempotencyRecord
from spatial_memory.core.db_indexes import IndexManager
from spatial_memory.core.db_migrations import CURRENT_SCHEMA_VERSION, MigrationManager
from spatial_memory.core.db_namespace import NamespaceManager
from spatial_memory.core.db_search import SearchManager
from spatial_memory.core.db_stats import HealthMetrics, StatsManager
from spatial_memory.core.db_versioning import VersionManager
from spatial_memory.core.errors import (
    FileLockError,
    StorageError,
)
from spatial_memory.core.filesystem import (
    detect_filesystem_type,
    get_filesystem_warning_message,
    is_network_filesystem,
)
from spatial_memory.core.utils import utc_now

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)

# Type variable for retry decorator
F = TypeVar("F", bound=Callable[..., Any])

# All known vector index types for detection
VECTOR_INDEX_TYPES = frozenset(
    {
        "IVF_PQ",
        "IVF_FLAT",
        "HNSW",
        "IVF_HNSW_PQ",
        "IVF_HNSW_SQ",
        "HNSW_PQ",
        "HNSW_SQ",
    }
)

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
                except (StorageError, OSError, ConnectionError, TimeoutError) as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check for non-retryable errors - raise immediately
                    if any(pattern in error_str for pattern in non_retryable_patterns):
                        logger.warning(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    # Check if we've exhausted retries
                    if attempt == max_attempts - 1:
                        raise

                    # Retry with exponential backoff
                    wait_time = backoff * (2**attempt)
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
                logger.warning(f"Stale connection detected in {func.__name__}, reconnecting...")
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
        # Stats manager (initialized in connect())
        self._stats_manager: StatsManager | None = None
        # Namespace manager (initialized in connect())
        self._namespace_manager: NamespaceManager | None = None
        # Export/import manager (initialized in connect())
        self._export_manager: ExportImportManager | None = None
        # CRUD manager (initialized in connect())
        self._crud_manager: CrudManager | None = None

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
            self._stats_manager = StatsManager(self)
            self._namespace_manager = NamespaceManager(self)
            self._export_manager = ExportImportManager(self)
            self._crud_manager = CrudManager(self)
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
                if hasattr(existing_tables_result, "tables"):
                    existing_tables = existing_tables_result.tables
                else:
                    existing_tables = existing_tables_result

                if "memories" not in existing_tables:
                    # Create table with schema
                    schema = pa.schema(
                        [
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
                            pa.field("project", pa.string()),
                            pa.field("content_hash", pa.string()),
                            pa.field("expires_at", pa.timestamp("us")),  # TTL support - nullable
                        ]
                    )
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
        self._stats_manager = None
        self._namespace_manager = None
        self._export_manager = None
        self._crud_manager = None
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
                stats = (
                    self._stats_manager._get_table_stats()
                    if self._stats_manager is not None
                    else {}
                )
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

    def _reset_index_state(self) -> None:
        """Reset all index tracking flags (used by clear_all for test isolation)."""
        self._has_vector_index = None
        self._has_fts_index = None
        if self._index_manager is not None:
            self._index_manager.reset_index_state()

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
            if self._stats_manager is None:
                raise StorageError("Database not connected")
            stats_before = self._stats_manager._get_table_stats()

            # Compact small fragments
            needs_compaction = stats_before.get("num_small_fragments", 0) > 10
            if needs_compaction:
                logger.info("Compacting fragments...")
                self.table.compact_files()

            # Optimize indexes
            logger.info("Optimizing indexes...")
            self.table.optimize()

            stats_after = self._stats_manager._get_table_stats()

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
        """Get table statistics with best-effort fragment info. Delegates to StatsManager."""
        if self._stats_manager is None:
            return {}
        return self._stats_manager._get_table_stats()

    @with_stale_connection_recovery
    def get_health_metrics(self) -> HealthMetrics:
        """Get comprehensive health and performance metrics. Delegates to StatsManager.

        Returns:
            HealthMetrics dataclass with all metrics.
        """
        if self._stats_manager is None:
            raise StorageError("Database not connected")
        return self._stats_manager.get_health_metrics()

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
        project: str = "",
        content_hash: str = "",
        _skip_field_validation: bool = False,
    ) -> str:
        """Insert a new memory. Delegates to CrudManager.

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
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.insert(
            content=content,
            vector=vector,
            namespace=namespace,
            tags=tags,
            importance=importance,
            source=source,
            metadata=metadata,
            project=project,
            content_hash=content_hash,
            _skip_field_validation=_skip_field_validation,
        )

    @with_process_lock
    @with_write_lock
    @retry_on_storage_error(max_attempts=3, backoff=0.5)
    def insert_batch(
        self,
        records: list[dict[str, Any]],
        batch_size: int = 1000,
        atomic: bool = False,
    ) -> list[str]:
        """Insert multiple memories efficiently. Delegates to CrudManager.

        Args:
            records: List of memory records with content, vector, and optional fields.
            batch_size: Records per batch (default: 1000, max: 10000).
            atomic: If True, rollback all inserts on partial failure.

        Returns:
            List of generated memory IDs.

        Raises:
            ValidationError: If input validation fails or batch_size exceeds maximum.
            StorageError: If database operation fails (and rollback succeeds when atomic=True).
            PartialBatchInsertError: If atomic=True and rollback fails after partial insert.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.insert_batch(
            records=records, batch_size=batch_size, atomic=atomic
        )

    def _rollback_batch_insert(self, memory_ids: list[str]) -> Exception | None:
        """Attempt to rollback a failed batch insert. Delegates to CrudManager."""
        if self._crud_manager is None:
            return StorageError("Database not connected")
        return self._crud_manager._rollback_batch_insert(memory_ids)

    @with_stale_connection_recovery
    def search_by_content_hash(
        self,
        content_hash: str,
        namespace: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any] | None:
        """Find a memory by its content hash. Delegates to CrudManager.

        Args:
            content_hash: SHA-256 hex digest to search for.
            namespace: Optional namespace filter.
            project: Optional project filter.

        Returns:
            The memory record dict, or None if not found.

        Raises:
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.search_by_content_hash(
            content_hash, namespace=namespace, project=project
        )

    @with_stale_connection_recovery
    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a memory by ID. Delegates to CrudManager.

        Args:
            memory_id: The memory ID.

        Returns:
            The memory record.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.get(memory_id)

    def get_batch(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get multiple memories by ID. Delegates to CrudManager.

        Args:
            memory_ids: List of memory UUIDs to retrieve.

        Returns:
            Dict mapping memory_id to memory record. Missing IDs are not included.

        Raises:
            ValidationError: If any memory_id format is invalid.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.get_batch(memory_ids)

    @with_process_lock
    @with_write_lock
    def update(self, memory_id: str, updates: dict[str, Any]) -> None:
        """Update a memory. Delegates to CrudManager.

        Args:
            memory_id: The memory ID.
            updates: Fields to update.

        Raises:
            ValidationError: If input validation fails.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        self._crud_manager.update(memory_id, updates)

    @with_process_lock
    @with_write_lock
    def update_batch(self, updates: list[tuple[str, dict[str, Any]]]) -> tuple[int, list[str]]:
        """Update multiple memories. Delegates to CrudManager.

        Args:
            updates: List of (memory_id, updates_dict) tuples.

        Returns:
            Tuple of (success_count, list of failed memory_ids).

        Raises:
            StorageError: If database operation fails completely.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.update_batch(updates)

    @with_process_lock
    @with_write_lock
    def delete(self, memory_id: str) -> None:
        """Delete a memory. Delegates to CrudManager.

        Args:
            memory_id: The memory ID.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        self._crud_manager.delete(memory_id)

    @with_process_lock
    @with_write_lock
    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all memories in a namespace. Delegates to CrudManager.

        Args:
            namespace: The namespace to delete.

        Returns:
            Number of deleted records.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.delete_by_namespace(namespace)

    @with_process_lock
    @with_write_lock
    def clear_all(self, reset_indexes: bool = True) -> int:
        """Clear all memories from the database. Delegates to CrudManager.

        Args:
            reset_indexes: If True, also reset index tracking flags.

        Returns:
            Number of deleted records.

        Raises:
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.clear_all(reset_indexes=reset_indexes)

    @with_process_lock
    @with_write_lock
    def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
        """Rename all memories from one namespace to another. Delegates to NamespaceManager.

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
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.rename_namespace(old_namespace, new_namespace)

    @with_stale_connection_recovery
    def get_stats(self, namespace: str | None = None, project: str | None = None) -> dict[str, Any]:
        """Get comprehensive database statistics. Delegates to StatsManager.

        Args:
            namespace: Filter stats to specific namespace (None = all).
            project: Filter stats to specific project (None = all).

        Returns:
            Dictionary with statistics.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        if self._stats_manager is None:
            raise StorageError("Database not connected")
        return self._stats_manager.get_stats(namespace=namespace, project=project)

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        """Get statistics for a specific namespace. Delegates to StatsManager.

        Args:
            namespace: The namespace to get statistics for.

        Returns:
            Dictionary with namespace statistics.

        Raises:
            ValidationError: If namespace is invalid.
            StorageError: If database operation fails.
        """
        if self._stats_manager is None:
            raise StorageError("Database not connected")
        return self._stats_manager.get_namespace_stats(namespace)

    def get_all_for_export(
        self,
        namespace: str | None = None,
        project: str | None = None,
        batch_size: int = 1000,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Stream all memories for export. Delegates to ExportImportManager.

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
        if self._export_manager is None:
            raise StorageError("Database not connected")
        yield from self._export_manager.get_all_for_export(
            namespace=namespace, project=project, batch_size=batch_size
        )

    @with_process_lock
    @with_write_lock
    def bulk_import(
        self,
        records: Iterator[dict[str, Any]],
        batch_size: int = 1000,
        namespace_override: str | None = None,
    ) -> tuple[int, list[str]]:
        """Import memories from an iterator. Delegates to ExportImportManager.

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
        if self._export_manager is None:
            raise StorageError("Database not connected")
        return self._export_manager.bulk_import(
            records=records, batch_size=batch_size, namespace_override=namespace_override
        )

    @with_process_lock
    @with_write_lock
    def delete_batch(self, memory_ids: list[str]) -> tuple[int, list[str]]:
        """Delete multiple memories atomically. Delegates to CrudManager.

        Args:
            memory_ids: List of memory UUIDs to delete.

        Returns:
            Tuple of (count_deleted, list_of_deleted_ids).

        Raises:
            ValidationError: If any memory_id is invalid.
            StorageError: If database operation fails.
        """
        if self._crud_manager is None:
            raise StorageError("Database not connected")
        return self._crud_manager.delete_batch(memory_ids)

    @with_process_lock
    @with_write_lock
    def update_access_batch(self, memory_ids: list[str]) -> int:
        """Update access for multiple memories. Delegates to ExportImportManager.

        Args:
            memory_ids: List of memory UUIDs to update.

        Returns:
            Number of memories successfully updated.
        """
        if self._export_manager is None:
            raise StorageError("Database not connected")
        return self._export_manager.update_access_batch(memory_ids)

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
        project: str | None = None,
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
            project: Filter to specific project.
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
            project=project,
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
        project: str | None = None,
        min_similarity: float = 0.0,
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Batch search using native LanceDB. Delegates to SearchManager.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum number of results per query.
            namespace: Filter to specific namespace.
            project: Filter to specific project.
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
            project=project,
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
        project: str | None = None,
        alpha: float = 0.5,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector and keyword. Delegates to SearchManager.

        Args:
            query: Text query for full-text search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results.
            namespace: Filter to namespace.
            project: Filter to specific project.
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
            project=project,
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
        project: str | None = None,
        parallel: bool = False,  # Deprecated
        max_workers: int = 4,  # Deprecated
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Search using multiple query vectors. Delegates to SearchManager.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum results per query vector.
            namespace: Filter to specific namespace.
            project: Filter to specific project.
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
            project=project,
            parallel=parallel,
            max_workers=max_workers,
            include_vector=include_vector,
        )

    def get_vectors_for_clustering(
        self,
        namespace: str | None = None,
        project: str | None = None,
        max_memories: int = 10_000,
    ) -> tuple[list[str], np.ndarray]:
        """Fetch all vectors for clustering operations. Delegates to NamespaceManager.

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
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.get_vectors_for_clustering(
            namespace=namespace, project=project, max_memories=max_memories
        )

    def get_vectors_as_arrow(
        self,
        namespace: str | None = None,
        project: str | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Get memories as Arrow table. Delegates to NamespaceManager.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.
            columns: Columns to select (None = all).

        Returns:
            PyArrow Table with selected data.

        Raises:
            StorageError: If database operation fails.
        """
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.get_vectors_as_arrow(
            namespace=namespace, project=project, columns=columns
        )

    def get_all(
        self,
        namespace: str | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all memories, optionally filtered. Delegates to NamespaceManager.

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
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.get_all(namespace=namespace, project=project, limit=limit)

    @with_stale_connection_recovery
    def count(self, namespace: str | None = None, project: str | None = None) -> int:
        """Count memories. Delegates to NamespaceManager.

        Args:
            namespace: Filter to specific namespace.
            project: Filter to specific project.

        Returns:
            Number of memories.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.count(namespace=namespace, project=project)

    @with_stale_connection_recovery
    def get_namespaces(self) -> list[str]:
        """Get all unique namespaces. Delegates to NamespaceManager.

        Returns:
            Sorted list of namespace names.

        Raises:
            StorageError: If database operation fails.
        """
        if self._namespace_manager is None:
            raise StorageError("Database not connected")
        return self._namespace_manager.get_namespaces()

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
        self.update(
            memory_id,
            {
                "last_accessed": utc_now(),
                "access_count": existing["access_count"] + 1,
            },
        )

    # ========================================================================
    # Backup, Export, Import & TTL (delegates to ExportImportManager)
    # ========================================================================

    def export_to_parquet(
        self,
        output_path: Path,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Export memories to Parquet file. Delegates to ExportImportManager.

        Args:
            output_path: Path to save Parquet file.
            namespace: Export only this namespace (None = all).

        Returns:
            Export statistics (rows_exported, output_path, size_mb).

        Raises:
            StorageError: If export fails.
        """
        if self._export_manager is None:
            raise StorageError("Database not connected")
        return self._export_manager.export_to_parquet(output_path, namespace=namespace)

    @with_process_lock
    @with_write_lock
    def import_from_parquet(
        self,
        parquet_path: Path,
        namespace_override: str | None = None,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Import memories from Parquet backup. Delegates to ExportImportManager.

        Args:
            parquet_path: Path to Parquet file.
            namespace_override: Override namespace for all imported memories.
            batch_size: Records per batch during import.

        Returns:
            Import statistics (rows_imported, source).

        Raises:
            StorageError: If import fails.
        """
        if self._export_manager is None:
            raise StorageError("Database not connected")
        return self._export_manager.import_from_parquet(
            parquet_path, namespace_override=namespace_override, batch_size=batch_size
        )

    @with_process_lock
    @with_write_lock
    def set_memory_ttl(self, memory_id: str, ttl_days: int | None) -> None:
        """Set TTL for a specific memory. Delegates to ExportImportManager.

        Args:
            memory_id: Memory ID.
            ttl_days: Days until expiration, or None to remove TTL.

        Raises:
            ValidationError: If memory_id is invalid.
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If database operation fails.
        """
        if self._export_manager is None:
            raise StorageError("Database not connected")
        self._export_manager.set_memory_ttl(memory_id, ttl_days)

    @with_process_lock
    @with_write_lock
    def cleanup_expired_memories(self) -> int:
        """Delete expired memories. Delegates to ExportImportManager.

        Returns:
            Number of deleted memories.

        Raises:
            StorageError: If cleanup fails.
        """
        if self._export_manager is None:
            raise StorageError("Database not connected")
        return self._export_manager.cleanup_expired_memories()

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
