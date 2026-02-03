"""Search operations for LanceDB database.

Provides vector search, hybrid search, and batch search functionality.

This module is part of the database.py refactoring to separate concerns:
- SearchManager handles all search-related operations
- Database class delegates to SearchManager for these operations
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

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


class SearchManagerProtocol(Protocol):
    """Protocol defining what SearchManager needs from Database.

    This protocol enables loose coupling between SearchManager and Database,
    preventing circular imports while maintaining type safety.
    """

    @property
    def table(self) -> LanceTable:
        """Access to the LanceDB table."""
        ...

    @property
    def index_nprobes(self) -> int:
        """Base nprobes for search."""
        ...

    @property
    def index_refine_factor(self) -> int:
        """Base refine factor for search."""
        ...

    @property
    def vector_index_threshold(self) -> int:
        """Row count threshold for vector index."""
        ...

    def _get_cached_row_count(self) -> int:
        """Get cached row count."""
        ...

    @property
    def _has_vector_index(self) -> bool | None:
        """Whether vector index exists."""
        ...

    @property
    def _has_fts_index(self) -> bool | None:
        """Whether FTS index exists."""
        ...


class SearchManager:
    """Manages search operations for vector and hybrid queries.

    Handles vector similarity search, batch search, and hybrid
    search combining vector and keyword matching.

    Example:
        search_mgr = SearchManager(database)
        results = search_mgr.vector_search(query_vector, limit=10)
        batch_results = search_mgr.batch_vector_search_native([vec1, vec2])
    """

    def __init__(self, db: SearchManagerProtocol) -> None:
        """Initialize the search manager.

        Args:
            db: Database instance providing table and config access.
        """
        self._db = db

    def calculate_search_params(
        self,
        count: int,
        limit: int,
        nprobes_override: int | None = None,
        refine_factor_override: int | None = None,
    ) -> tuple[int, int]:
        """Calculate optimal search parameters based on dataset size and limit.

        Dynamically tunes nprobes and refine_factor for optimal recall/speed tradeoff.

        Args:
            count: Number of rows in the dataset.
            limit: Number of results requested.
            nprobes_override: Optional override for nprobes (uses this if provided).
            refine_factor_override: Optional override for refine_factor.

        Returns:
            Tuple of (nprobes, refine_factor).

        Scaling rules:
            - nprobes: Base from config, scaled up for larger datasets
              - <100K: config value (default 20)
              - 100K-1M: max(config, 30)
              - 1M-10M: max(config, 50)
              - >10M: max(config, 100)
            - refine_factor: Base from config, scaled up for small limits
              - limit <= 5: config value * 2
              - limit <= 20: config value
              - limit > 20: max(config // 2, 2)
        """
        # Calculate nprobes based on dataset size
        if nprobes_override is not None:
            nprobes = nprobes_override
        else:
            base_nprobes = self._db.index_nprobes
            if count < 100_000:
                nprobes = base_nprobes
            elif count < 1_000_000:
                nprobes = max(base_nprobes, 30)
            elif count < 10_000_000:
                nprobes = max(base_nprobes, 50)
            else:
                nprobes = max(base_nprobes, 100)

        # Calculate refine_factor based on limit
        if refine_factor_override is not None:
            refine_factor = refine_factor_override
        else:
            base_refine = self._db.index_refine_factor
            if limit <= 5:
                # Small limits need more refinement for accuracy
                refine_factor = base_refine * 2
            elif limit <= 20:
                refine_factor = base_refine
            else:
                # Large limits can use less refinement
                refine_factor = max(base_refine // 2, 2)

        return nprobes, refine_factor

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
        """Search for similar memories by vector with performance tuning.

        Note: This method should be called through the Database class which
        applies stale connection recovery and retry decorators.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            namespace: Filter to specific namespace.
            min_similarity: Minimum similarity threshold (0-1).
            nprobes: Number of partitions to search (higher = better recall).
                     Only effective when vector index exists. Defaults to dynamic calculation.
            refine_factor: Re-rank top (refine_factor * limit) for accuracy.
                          Defaults to dynamic calculation based on limit.
            include_vector: Whether to include vector embeddings in results.
                           Defaults to False to reduce response size.

        Returns:
            List of memory records with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            search = self._db.table.search(query_vector.tolist())

            # Distance type for queries (cosine for semantic similarity)
            # Note: When vector index exists, the index's metric is used
            search = search.distance_type("cosine")

            # Apply performance tuning when index exists (use cached count)
            count = self._db._get_cached_row_count()
            if count > self._db.vector_index_threshold and self._db._has_vector_index:
                # Use dynamic calculation for search params
                actual_nprobes, actual_refine = self.calculate_search_params(
                    count, limit, nprobes, refine_factor
                )
                search = search.nprobes(actual_nprobes)
                search = search.refine_factor(actual_refine)

            # Build filter with sanitized namespace
            # prefilter=True applies namespace filter BEFORE vector search for better performance
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'", prefilter=True)

            # Vector projection: exclude vector column to reduce response size
            if not include_vector:
                search = search.select(
                    [
                        "id",
                        "content",
                        "namespace",
                        "metadata",
                        "created_at",
                        "updated_at",
                        "last_accessed",
                        "importance",
                        "tags",
                        "source",
                        "access_count",
                        "expires_at",
                    ]
                )

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

    def batch_vector_search_native(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        min_similarity: float = 0.0,
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Batch search for similar memories using native LanceDB batch search.

        Searches for multiple query vectors in a single database operation,
        much more efficient than individual searches. Uses LanceDB's native
        batch search API which returns results with query_index for grouping.

        Note: This method should be called through the Database class which
        applies stale connection recovery and retry decorators.

        Args:
            query_vectors: List of query embedding vectors.
            limit_per_query: Maximum number of results per query.
            namespace: Filter to specific namespace (applied to all queries).
            min_similarity: Minimum similarity threshold (0-1).
            include_vector: Whether to include vector embeddings in results.

        Returns:
            List of result lists, one per query vector (same order as input).
            Each result list contains memory records with similarity scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        if not query_vectors:
            return []

        try:
            # Convert all vectors to lists for LanceDB
            vector_lists = [v.tolist() for v in query_vectors]

            # LanceDB native batch search
            search = self._db.table.search(vector_lists)
            search = search.distance_type("cosine")

            # Apply performance tuning when index exists
            count = self._db._get_cached_row_count()
            if count > self._db.vector_index_threshold and self._db._has_vector_index:
                actual_nprobes, actual_refine = self.calculate_search_params(
                    count, limit_per_query, None, None
                )
                search = search.nprobes(actual_nprobes)
                search = search.refine_factor(actual_refine)

            # Apply namespace filter
            if namespace:
                namespace = _validate_namespace(namespace)
                safe_ns = _sanitize_string(namespace)
                search = search.where(f"namespace = '{safe_ns}'", prefilter=True)

            # Vector projection
            if not include_vector:
                search = search.select(
                    [
                        "id",
                        "content",
                        "namespace",
                        "metadata",
                        "created_at",
                        "updated_at",
                        "last_accessed",
                        "importance",
                        "tags",
                        "source",
                        "access_count",
                    ]
                )

            # Execute search and get results
            # LanceDB returns results with _query_index to identify which query
            # each result belongs to
            # Use Arrow operations (no pandas dependency)
            search = search.limit(limit_per_query)
            results = search.to_arrow().to_pylist()

            # Initialize result lists (one per query)
            num_queries = len(query_vectors)
            batch_results: list[list[dict[str, Any]]] = [[] for _ in range(num_queries)]

            if not results:
                return batch_results

            # Group results by query index
            for record in results:
                query_idx = int(record.get("_query_index", 0))
                if query_idx >= num_queries:
                    continue

                # Convert distance to similarity (cosine distance -> similarity)
                distance = record.get("_distance", 0)
                similarity = 1.0 - distance

                if similarity < min_similarity:
                    continue

                # Clean up internal fields
                record.pop("_distance", None)
                record.pop("_query_index", None)
                record.pop("_relevance_score", None)

                # Add similarity score
                record["similarity"] = similarity

                # Deserialize metadata
                if record.get("metadata"):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        record["metadata"] = {}
                else:
                    record["metadata"] = {}

                batch_results[query_idx].append(record)

            return batch_results
        except ValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to batch search: {e}") from e

    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        limit: int = 5,
        namespace: str | None = None,
        alpha: float = 0.5,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword matching.

        Uses LinearCombinationReranker to balance vector and keyword scores
        based on the alpha parameter.

        Note: This method should be called through the Database class which
        applies stale connection recovery and retry decorators.

        Args:
            query: Text query for full-text search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results.
            namespace: Filter to namespace.
            alpha: Balance between vector (1.0) and keyword (0.0).
                   0.5 = balanced (recommended).
            min_similarity: Minimum similarity threshold (0.0-1.0).
                           Results below this threshold are filtered out.

        Returns:
            List of memory records with combined scores.

        Raises:
            ValidationError: If input validation fails.
            StorageError: If database operation fails.
        """
        try:
            # Check if FTS is available
            if not self._db._has_fts_index:
                logger.debug("FTS index not available, falling back to vector search")
                return self.vector_search(query_vector, limit=limit, namespace=namespace)

            # Create hybrid search with explicit vector column specification
            # Required when using external embeddings (not LanceDB built-in)
            search = (
                self._db.table.search(query, query_type="hybrid")
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

            # Process results - normalize scores and clean up internal columns
            processed_results: list[dict[str, Any]] = []
            for record in results:
                record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else {}

                # Compute similarity from various score columns
                # Priority: _relevance_score > _distance > _score > default
                similarity: float
                if "_relevance_score" in record:
                    # Reranker output - use directly (already 0-1 range)
                    similarity = float(record["_relevance_score"])
                    del record["_relevance_score"]
                elif "_distance" in record:
                    # Vector distance - convert to similarity
                    similarity = max(0.0, min(1.0, 1 - float(record["_distance"])))
                    del record["_distance"]
                elif "_score" in record:
                    # BM25 score - normalize using score/(1+score)
                    score = float(record["_score"])
                    similarity = score / (1.0 + score)
                    del record["_score"]
                else:
                    # No score column - use default
                    similarity = 0.5

                record["similarity"] = similarity

                # Mark as hybrid result with alpha value
                record["search_type"] = "hybrid"
                record["alpha"] = alpha

                # Apply min_similarity filter
                if similarity >= min_similarity:
                    processed_results.append(record)

            return processed_results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return self.vector_search(query_vector, limit=limit, namespace=namespace)

    def batch_vector_search(
        self,
        query_vectors: list[np.ndarray],
        limit_per_query: int = 3,
        namespace: str | None = None,
        parallel: bool = False,  # Deprecated: native batch is always efficient
        max_workers: int = 4,  # Deprecated: native batch handles parallelism
        include_vector: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Search for similar memories using multiple query vectors.

        Uses native LanceDB batch search for efficiency. A single database
        operation searches all vectors simultaneously.

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
        # Delegate to native batch search implementation
        return self.batch_vector_search_native(
            query_vectors=query_vectors,
            limit_per_query=limit_per_query,
            namespace=namespace,
            min_similarity=0.0,
            include_vector=include_vector,
        )
