"""Protocol interfaces for cross-corpus similarity operations.

Segregated by consumer profile following the Interface Segregation Principle:
- SimilarityQueryPort: Single-vector queries (discovery, surfacing)
- BatchSimilarityPort: Multi-vector queries (consolidation, dedup)
- CorpusAnalysisPort: Corpus-wide analytics (cross-app discovery)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from spatial_memory.core.models import (
    CorpusSimilaritySummary,
    CrossCorpusMatch,
)


class SimilarityQueryPort(Protocol):
    """Single-query similarity for proactive surfacing and discovery.

    Consumers: MCP tool handlers, proactive surfacing pipeline.
    """

    def find_similar_across_corpus(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        min_similarity: float = 0.5,
        exclude_namespace: str | None = None,
        exclude_project: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[CrossCorpusMatch]:
        """Find similar items across all namespaces and projects.

        Args:
            query_vector: The vector to search with.
            limit: Maximum results.
            min_similarity: Floor for similarity score.
            exclude_namespace: Skip results from this namespace.
            exclude_project: Skip results from this project.
            exclude_ids: Specific memory IDs to exclude.

        Returns:
            Matches sorted by similarity descending.
        """
        ...


class BatchSimilarityPort(Protocol):
    """Batch similarity for consolidation and dedup pipelines.

    Consumers: LifecycleService.consolidate(), IngestPipeline dedup.
    """

    def find_similar_batch(
        self,
        query_vectors: list[np.ndarray],
        *,
        limit_per_query: int = 5,
        min_similarity: float = 0.7,
        exclude_ids: list[set[str]] | None = None,
    ) -> list[list[CrossCorpusMatch]]:
        """Find similar items for multiple vectors in a single operation.

        Args:
            query_vectors: Vectors to search (one result list per vector).
            limit_per_query: Max results per query vector.
            min_similarity: Floor for similarity score.
            exclude_ids: Per-query exclusion sets (parallel with query_vectors).

        Returns:
            List of match lists, parallel with query_vectors.
        """
        ...


class CorpusAnalysisPort(Protocol):
    """Corpus-wide analysis for cross-app discovery and reporting.

    Consumers: Admin tools, cross-project relationship discovery.
    """

    def find_cross_namespace_bridges(
        self,
        *,
        min_similarity: float = 0.8,
        max_bridges: int = 50,
        namespace_filter: list[str] | None = None,
    ) -> list[CrossCorpusMatch]:
        """Find memories that bridge different namespaces.

        Identifies memories in different namespaces/projects that are
        semantically very similar -- potential knowledge links or duplicates.

        Args:
            min_similarity: Minimum similarity to consider a bridge.
            max_bridges: Maximum bridges to return.
            namespace_filter: Only consider these namespaces (None = all).

        Returns:
            Bridges sorted by similarity descending.
        """
        ...

    def get_corpus_overlap_summary(
        self,
        *,
        namespace_filter: list[str] | None = None,
    ) -> CorpusSimilaritySummary:
        """Get summary statistics about cross-namespace similarity.

        Args:
            namespace_filter: Only consider these namespaces (None = all).

        Returns:
            Summary with bridge counts, overlap matrix, and top bridges.
        """
        ...
