"""Index management for LanceDB database.

Provides vector, FTS, and scalar index creation and management.

This module is part of the database.py refactoring to separate concerns:
- IndexManager handles all index-related operations
- Database class delegates to IndexManager for these operations
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Protocol

from spatial_memory.core.errors import StorageError

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)

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


class IndexManagerProtocol(Protocol):
    """Protocol defining what IndexManager needs from Database.

    This protocol enables loose coupling between IndexManager and Database,
    preventing circular imports while maintaining type safety.
    """

    @property
    def table(self) -> LanceTable:
        """Access to the LanceDB table."""
        ...

    # Configuration properties
    @property
    def enable_fts(self) -> bool:
        """Whether FTS is enabled."""
        ...

    @property
    def fts_language(self) -> str:
        """FTS language."""
        ...

    @property
    def fts_stem(self) -> bool:
        """FTS stemming enabled."""
        ...

    @property
    def fts_remove_stop_words(self) -> bool:
        """FTS stop words removal enabled."""
        ...

    @property
    def index_type(self) -> str:
        """Vector index type."""
        ...

    @property
    def vector_index_threshold(self) -> int:
        """Row count threshold for vector index."""
        ...

    @property
    def auto_create_indexes(self) -> bool:
        """Auto-create indexes when thresholds met."""
        ...

    @property
    def hnsw_m(self) -> int:
        """HNSW M parameter."""
        ...

    @property
    def hnsw_ef_construction(self) -> int:
        """HNSW ef_construction parameter."""
        ...

    @property
    def index_wait_timeout_seconds(self) -> float:
        """Timeout for waiting on index creation."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        ...


class IndexManager:
    """Manages vector, FTS, and scalar indexes.

    Handles index creation, detection, and optimization for
    LanceDB tables.

    Example:
        index_mgr = IndexManager(database)
        index_mgr.ensure_indexes()
        if not index_mgr.has_vector_index:
            index_mgr.create_vector_index()
    """

    def __init__(self, db: IndexManagerProtocol) -> None:
        """Initialize the index manager.

        Args:
            db: Database instance providing table and config access.
        """
        self._db = db
        self._has_vector_index: bool | None = None
        self._has_fts_index: bool | None = None
        self._has_scalar_indexes: bool = False

    @property
    def has_vector_index(self) -> bool | None:
        """Whether vector index exists."""
        return self._has_vector_index

    @has_vector_index.setter
    def has_vector_index(self, value: bool | None) -> None:
        self._has_vector_index = value

    @property
    def has_fts_index(self) -> bool | None:
        """Whether FTS index exists."""
        return self._has_fts_index

    @has_fts_index.setter
    def has_fts_index(self, value: bool | None) -> None:
        self._has_fts_index = value

    @property
    def has_scalar_indexes(self) -> bool:
        """Whether scalar indexes exist."""
        return self._has_scalar_indexes

    @has_scalar_indexes.setter
    def has_scalar_indexes(self, value: bool) -> None:
        self._has_scalar_indexes = value

    def reset_index_state(self) -> None:
        """Reset all index state flags."""
        self._has_vector_index = None
        self._has_fts_index = None
        self._has_scalar_indexes = False

    def check_existing_indexes(self) -> None:
        """Check which indexes already exist using robust detection."""
        try:
            indices = self._db.table.list_indices()

            self._has_vector_index = False
            self._has_fts_index = False

            for idx in indices:
                index_name = str(_get_index_attr(idx, "name", "")).lower()
                index_type = str(_get_index_attr(idx, "index_type", "")).upper()
                columns = _get_index_attr(idx, "columns", [])

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
                f"Existing indexes: vector={self._has_vector_index}, fts={self._has_fts_index}"
            )
        except Exception as e:
            logger.warning(f"Could not check existing indexes: {e}")
            self._has_vector_index = None
            self._has_fts_index = None

    def create_fts_index(self) -> None:
        """Create full-text search index with optimized settings."""
        try:
            self._db.table.create_fts_index(
                "content",
                use_tantivy=False,  # Use Lance native FTS
                language=self._db.fts_language,
                stem=self._db.fts_stem,
                remove_stop_words=self._db.fts_remove_stop_words,
                with_position=True,  # Enable phrase queries
                lower_case=True,  # Case-insensitive search
            )
            self._has_fts_index = True
            logger.info(
                f"Created FTS index with stemming={self._db.fts_stem}, "
                f"stop_words={self._db.fts_remove_stop_words}"
            )
        except Exception as e:
            # Check if index already exists (not an error)
            if "already exists" in str(e).lower():
                self._has_fts_index = True
                logger.debug("FTS index already exists")
            else:
                logger.warning(f"FTS index creation failed: {e}")

    def create_vector_index(self, force: bool = False) -> bool:
        """Create vector index for similarity search.

        Supports IVF_PQ, IVF_FLAT, and HNSW_SQ index types based on configuration.
        Automatically determines optimal parameters based on dataset size.

        Args:
            force: Force index creation regardless of dataset size.

        Returns:
            True if index was created, False if skipped.

        Raises:
            StorageError: If index creation fails.
        """
        count = self._db.table.count_rows()

        # Check threshold
        if count < self._db.vector_index_threshold and not force:
            logger.info(
                f"Dataset has {count} rows, below threshold {self._db.vector_index_threshold}. "
                "Skipping vector index creation."
            )
            return False

        # Check if already exists
        if self._has_vector_index and not force:
            logger.info("Vector index already exists")
            return False

        # Handle HNSW_SQ index type
        if self._db.index_type == "HNSW_SQ":
            return self._create_hnsw_index(count)

        # IVF-based index creation (IVF_PQ or IVF_FLAT)
        return self._create_ivf_index(count)

    def _create_hnsw_index(self, count: int) -> bool:
        """Create HNSW-SQ vector index.

        HNSW (Hierarchical Navigable Small World) provides better recall than IVF
        at the cost of higher memory usage. Good for datasets where recall is critical.

        Args:
            count: Number of rows in the table.

        Returns:
            True if index was created.

        Raises:
            StorageError: If index creation fails.
        """
        logger.info(
            f"Creating HNSW_SQ vector index: m={self._db.hnsw_m}, "
            f"ef_construction={self._db.hnsw_ef_construction} for {count} rows"
        )

        try:
            self._db.table.create_index(
                metric="cosine",
                vector_column_name="vector",
                index_type="HNSW_SQ",
                replace=True,
                m=self._db.hnsw_m,
                ef_construction=self._db.hnsw_ef_construction,
            )

            # Wait for index to be ready with configurable timeout
            self._wait_for_index_ready("vector", self._db.index_wait_timeout_seconds)

            self._has_vector_index = True
            logger.info("HNSW_SQ vector index created successfully")

            # Optimize after index creation (may fail in some environments)
            try:
                self._db.table.optimize()
            except Exception as optimize_error:
                logger.debug(f"Optimization after index creation skipped: {optimize_error}")

            return True

        except Exception as e:
            logger.error(f"Failed to create HNSW_SQ vector index: {e}")
            raise StorageError(f"HNSW_SQ vector index creation failed: {e}") from e

    def _create_ivf_index(self, count: int) -> bool:
        """Create IVF-PQ or IVF-FLAT vector index.

        Uses sqrt rule for partitions: num_partitions = sqrt(count), clamped to [16, 4096].
        Uses 48 sub-vectors for <500K rows (8 dims each for 384-dim vectors),
        96 sub-vectors for >=500K rows (4 dims each).

        Args:
            count: Number of rows in the table.

        Returns:
            True if index was created.

        Raises:
            StorageError: If index creation fails.
        """
        # Use sqrt rule for partitions, clamped to [16, 4096]
        num_partitions = int(math.sqrt(count))
        num_partitions = max(16, min(num_partitions, 4096))

        # Choose num_sub_vectors based on dataset size
        # <500K: 48 sub-vectors (8 dims each for 384-dim, more precision)
        # >=500K: 96 sub-vectors (4 dims each, more compression)
        if count < 500_000:
            num_sub_vectors = 48
        else:
            num_sub_vectors = 96

        # Validate embedding_dim % num_sub_vectors == 0 (required for IVF-PQ)
        if self._db.embedding_dim % num_sub_vectors != 0:
            # Find a valid divisor from common sub-vector counts
            valid_divisors = [96, 48, 32, 24, 16, 12, 8, 4]
            found_divisor = False
            for divisor in valid_divisors:
                if self._db.embedding_dim % divisor == 0:
                    logger.info(
                        f"Adjusted num_sub_vectors from {num_sub_vectors} to {divisor} "
                        f"for embedding_dim={self._db.embedding_dim}"
                    )
                    num_sub_vectors = divisor
                    found_divisor = True
                    break

            if not found_divisor:
                raise StorageError(
                    f"Cannot create IVF-PQ index: embedding_dim={self._db.embedding_dim} "
                    "has no suitable divisor for sub-vectors. "
                    f"Tried divisors: {valid_divisors}"
                )

        # IVF-PQ requires minimum rows for training (sample_rate * num_partitions / 256)
        # Default sample_rate=256, so we need at least 256 rows
        # Also, IVF requires num_partitions < num_vectors for KMeans training
        sample_rate = 256  # default
        if count < 256:
            # Use IVF_FLAT for very small datasets (no PQ training required)
            logger.info(
                f"Dataset too small for IVF-PQ ({count} rows < 256). Using IVF_FLAT index instead."
            )
            index_type = "IVF_FLAT"
            sample_rate = max(16, count // 4)  # Lower sample rate for small data
        else:
            valid_types = ("IVF_PQ", "IVF_FLAT")
            index_type = self._db.index_type if self._db.index_type in valid_types else "IVF_PQ"

        # Ensure num_partitions < num_vectors for KMeans clustering
        if num_partitions >= count:
            num_partitions = max(1, count // 4)  # Use 1/4 of count, minimum 1
            logger.info(f"Adjusted num_partitions to {num_partitions} for {count} rows")

        logger.info(
            f"Creating {index_type} vector index: {num_partitions} partitions, "
            f"{num_sub_vectors} sub-vectors for {count} rows"
        )

        try:
            # LanceDB 0.27+ API: parameters passed directly to create_index
            index_kwargs: dict[str, Any] = {
                "metric": "cosine",
                "num_partitions": num_partitions,
                "vector_column_name": "vector",
                "index_type": index_type,
                "replace": True,
                "sample_rate": sample_rate,
            }

            # num_sub_vectors only applies to PQ-based indexes
            if "PQ" in index_type:
                index_kwargs["num_sub_vectors"] = num_sub_vectors

            self._db.table.create_index(**index_kwargs)

            # Wait for index to be ready with configurable timeout
            self._wait_for_index_ready("vector", self._db.index_wait_timeout_seconds)

            self._has_vector_index = True
            logger.info(f"{index_type} vector index created successfully")

            # Optimize after index creation (may fail in some environments)
            try:
                self._db.table.optimize()
            except Exception as optimize_error:
                logger.debug(f"Optimization after index creation skipped: {optimize_error}")

            return True

        except Exception as e:
            logger.error(f"Failed to create {index_type} vector index: {e}")
            raise StorageError(f"{index_type} vector index creation failed: {e}") from e

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
                indices = self._db.table.list_indices()
                for idx in indices:
                    idx_name = str(_get_index_attr(idx, "name", "")).lower()
                    idx_columns = _get_index_attr(idx, "columns", [])

                    # Match by column name in index metadata, or index name contains column
                    if column_name in idx_columns or column_name in idx_name:
                        # Index exists, check if it's ready
                        status = str(_get_index_attr(idx, "status", "ready"))
                        if status.lower() in ("ready", "complete", "built"):
                            logger.debug(f"Index on {column_name} is ready")
                            return
                        break
            except Exception as e:
                logger.debug(f"Error checking index status: {e}")

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for index on {column_name} after {timeout_seconds}s")

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
            "content_hash",  # Dedup layer 1: exact-match hash lookup
            "created_at",
            "updated_at",
            "last_accessed",
            "importance",
            "access_count",
            "expires_at",  # TTL expiration queries
        ]

        for column in btree_columns:
            try:
                self._db.table.create_scalar_index(
                    column,
                    index_type="BTREE",
                    replace=True,
                )
                logger.debug(f"Created BTREE index on {column}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create BTREE index on {column}: {e}")

        # BITMAP indexes for low-cardinality columns
        bitmap_columns = ["namespace", "source", "project"]

        for column in bitmap_columns:
            try:
                self._db.table.create_scalar_index(
                    column,
                    index_type="BITMAP",
                    replace=True,
                )
                logger.debug(f"Created BITMAP index on {column}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create BITMAP index on {column}: {e}")

        # LABEL_LIST index for tags array (supports array_has_any queries)
        try:
            self._db.table.create_scalar_index(
                "tags",
                index_type="LABEL_LIST",
                replace=True,
            )
            logger.debug("Created LABEL_LIST index on tags")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Could not create LABEL_LIST index on tags: {e}")

        self._has_scalar_indexes = True
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

        count = self._db.table.count_rows()

        # Vector index
        if self._db.auto_create_indexes or force:
            if count >= self._db.vector_index_threshold or force:
                results["vector_index"] = self.create_vector_index(force=force)

        # Scalar indexes (always create if > 1000 rows)
        if count >= 1000 or force:
            try:
                self.create_scalar_indexes()
                results["scalar_indexes"] = True
            except Exception as e:
                logger.warning(f"Scalar index creation partially failed: {e}")

        # FTS index
        if self._db.enable_fts and not self._has_fts_index:
            try:
                self.create_fts_index()
                results["fts_index"] = True
            except Exception as e:
                logger.warning(f"FTS index creation failed in ensure_indexes: {e}")

        return results
