"""Cross-corpus similarity service.

Finds similar memories across the entire corpus using ANN-based search
with pluggable scoring strategies. Read-only service -- no mutations.

Implements three ISP-segregated ports:
- SimilarityQueryPort: Single-vector queries
- BatchSimilarityPort: Multi-vector batch queries
- CorpusAnalysisPort: Corpus-wide analytics
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from spatial_memory.core.errors import MemoryNotFoundError
from spatial_memory.core.models import (
    BatchSimilarityResult,
    CorpusSimilaritySummary,
    CrossCorpusMatch,
    SimilarityConfig,
)
from spatial_memory.core.scoring import (
    ScoringContext,
    get_scoring_strategy,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from spatial_memory.core.scoring import ScoringStrategy
    from spatial_memory.ports.repositories import (
        MemoryNamespaceProtocol,
        MemoryRepositoryProtocol,
        MemorySearchProtocol,
    )


class CrossCorpusSimilarityService:
    """Finds similar memories across the entire corpus.

    Uses ANN (via LanceDB vector index) for candidate discovery,
    then applies pluggable scoring strategies for final ranking.

    Read-only service -- depends only on search and namespace protocols.
    """

    def __init__(
        self,
        repository: MemorySearchProtocol,
        namespace_provider: MemoryNamespaceProtocol,
        config: SimilarityConfig | None = None,
        scoring_strategy: ScoringStrategy | None = None,
        memory_repository: MemoryRepositoryProtocol | None = None,
    ) -> None:
        self._repo = repository
        self._ns_provider = namespace_provider
        self._config = config or SimilarityConfig()
        self._scoring = scoring_strategy or get_scoring_strategy(self._config.scoring_strategy)
        self._memory_repo = memory_repository

    # =========================================================================
    # SimilarityQueryPort
    # =========================================================================

    def find_similar_across_corpus(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        min_similarity: float = 0.5,
        exclude_namespace: str | None = None,
        exclude_project: str | None = None,
        exclude_ids: set[str] | None = None,
        query_content: str | None = None,
        query_namespace: str | None = None,
        query_project: str | None = None,
    ) -> list[CrossCorpusMatch]:
        """Find similar items across all namespaces and projects.

        Over-fetches candidates via ANN, applies exclusions and scoring,
        then returns top matches.

        Args:
            query_vector: The vector to search with.
            limit: Maximum results to return.
            min_similarity: Floor for final scored similarity.
            exclude_namespace: Skip results from this namespace.
            exclude_project: Skip results from this project.
            exclude_ids: Specific memory IDs to exclude.
            query_content: Optional query text for content-aware scoring.
            query_namespace: Source namespace for provenance tracking.
            query_project: Source project for provenance tracking.

        Returns:
            Matches sorted by similarity descending.
        """
        overfetch = limit * self._config.ann_candidate_multiplier
        candidates = self._repo.search(
            query_vector,
            limit=overfetch,
            namespace=None,
            project=None,
        )

        matches: list[CrossCorpusMatch] = []
        exclude = exclude_ids or set()

        for result in candidates:
            if result.id in exclude:
                continue
            if exclude_namespace and result.namespace == exclude_namespace:
                continue
            if exclude_project and result.project == exclude_project:
                continue

            ctx = ScoringContext(
                vector_similarity=result.similarity,
                query_content=query_content,
                candidate_content=result.content,
                query_namespace=query_namespace,
                candidate_namespace=result.namespace,
                query_project=query_project,
                candidate_project=result.project,
                candidate_importance=result.importance,
                candidate_tags=result.tags,
                candidate_metadata=result.metadata,
            )
            final_score = self._scoring.score(ctx)

            if final_score < min_similarity:
                continue

            matches.append(
                CrossCorpusMatch(
                    memory_id=result.id,
                    content=result.content,
                    similarity=final_score,
                    raw_vector_similarity=result.similarity,
                    namespace=result.namespace,
                    project=result.project,
                    importance=result.importance,
                    tags=result.tags,
                    created_at=result.created_at,
                    scoring_strategy=self._scoring.name,
                    query_namespace=query_namespace,
                    query_project=query_project,
                )
            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:limit]

    # =========================================================================
    # BatchSimilarityPort
    # =========================================================================

    def find_similar_batch(
        self,
        query_vectors: list[np.ndarray],
        *,
        limit_per_query: int = 5,
        min_similarity: float = 0.7,
        exclude_ids: list[set[str]] | None = None,
        query_contents: list[str | None] | None = None,
        query_memory_ids: list[str | None] | None = None,
    ) -> list[BatchSimilarityResult]:
        """Find similar items for multiple vectors in a single operation.

        Args:
            query_vectors: Vectors to search (one result set per vector).
            limit_per_query: Max results per query vector.
            min_similarity: Floor for similarity score.
            exclude_ids: Per-query exclusion sets (parallel with query_vectors).
            query_contents: Per-query content for content-aware scoring.
            query_memory_ids: Per-query source memory IDs for provenance.

        Returns:
            List of BatchSimilarityResult, parallel with query_vectors.

        Raises:
            ValueError: If batch size exceeds max_batch_size.
        """
        if len(query_vectors) > self._config.max_batch_size:
            raise ValueError(
                f"Batch size {len(query_vectors)} exceeds maximum {self._config.max_batch_size}"
            )

        overfetch = limit_per_query * self._config.ann_candidate_multiplier
        batch_results = self._repo.batch_vector_search(
            query_vectors,
            limit_per_query=overfetch,
            namespace=None,
            project=None,
        )

        results: list[BatchSimilarityResult] = []

        for i, raw_candidates in enumerate(batch_results):
            exclude = exclude_ids[i] if exclude_ids and i < len(exclude_ids) else set()
            query_content = (
                query_contents[i] if query_contents and i < len(query_contents) else None
            )
            query_mem_id = (
                query_memory_ids[i] if query_memory_ids and i < len(query_memory_ids) else None
            )

            matches = self._score_raw_candidates(
                raw_candidates,
                min_similarity=min_similarity,
                limit=limit_per_query,
                exclude_ids=exclude,
                query_content=query_content,
            )

            results.append(
                BatchSimilarityResult(
                    query_index=i,
                    query_memory_id=query_mem_id,
                    matches=matches,
                )
            )

        return results

    # =========================================================================
    # CorpusAnalysisPort
    # =========================================================================

    def find_cross_namespace_bridges(
        self,
        *,
        min_similarity: float = 0.8,
        max_bridges: int = 50,
        namespace_filter: list[str] | None = None,
    ) -> list[CrossCorpusMatch]:
        """Find memories that bridge different namespaces.

        For each namespace, samples memories and searches for similar
        items in other namespaces via ANN. Deduplicates symmetric pairs.

        Args:
            min_similarity: Minimum similarity to consider a bridge.
            max_bridges: Maximum bridges to return.
            namespace_filter: Only consider these namespaces (None = all).

        Returns:
            Bridges sorted by similarity descending.
        """
        namespaces = self._ns_provider.get_namespaces()
        if namespace_filter:
            ns_set = set(namespace_filter)
            namespaces = [ns for ns in namespaces if ns in ns_set]

        if len(namespaces) < 2:
            return []

        sample_size = self._config.corpus_analysis_sample_size
        seen_pairs: set[tuple[str, str]] = set()
        bridges: list[CrossCorpusMatch] = []

        for ns in namespaces:
            mem_ids, vectors = self._repo.get_vectors_for_clustering(
                namespace=ns,
                max_memories=sample_size,
            )

            if len(mem_ids) == 0:
                continue

            vector_list = [vectors[j] for j in range(len(mem_ids))]

            batch_results = self._repo.batch_vector_search(
                vector_list,
                limit_per_query=5,
                namespace=None,
                project=None,
            )

            for j, raw_candidates in enumerate(batch_results):
                source_id = mem_ids[j]

                for cand in raw_candidates:
                    cand_id = str(cand.get("id", ""))
                    cand_ns = str(cand.get("namespace", ""))
                    cand_sim = float(cand.get("_distance", cand.get("similarity", 0.0)))

                    if cand_ns == ns:
                        continue
                    if namespace_filter and cand_ns not in ns_set:
                        continue
                    if cand_sim < min_similarity:
                        continue

                    pair = (min(source_id, cand_id), max(source_id, cand_id))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    bridges.append(
                        CrossCorpusMatch(
                            memory_id=cand_id,
                            content=str(cand.get("content", "")),
                            similarity=cand_sim,
                            raw_vector_similarity=cand_sim,
                            namespace=cand_ns,
                            project=str(cand.get("project", "")),
                            importance=float(cand.get("importance", 0.5)),
                            tags=cand.get("tags", []),
                            created_at=cand.get("created_at") or datetime.now(timezone.utc),
                            scoring_strategy="vector_only",
                            query_namespace=ns,
                            query_memory_id=source_id,
                        )
                    )

        bridges.sort(key=lambda m: m.similarity, reverse=True)
        return bridges[:max_bridges]

    def get_corpus_overlap_summary(
        self,
        *,
        namespace_filter: list[str] | None = None,
    ) -> CorpusSimilaritySummary:
        """Get summary statistics about cross-namespace similarity.

        Args:
            namespace_filter: Only consider these namespaces (None = all).

        Returns:
            Summary with bridge counts and top bridges.
        """
        namespaces = self._ns_provider.get_namespaces()
        if namespace_filter:
            ns_set = set(namespace_filter)
            namespaces = [ns for ns in namespaces if ns in ns_set]

        total_analyzed = 0
        for ns in namespaces:
            total_analyzed += self._ns_provider.count(namespace=ns)

        bridges = self.find_cross_namespace_bridges(
            min_similarity=0.8,
            max_bridges=100,
            namespace_filter=namespace_filter,
        )

        duplicates = sum(1 for b in bridges if b.similarity >= 0.95)

        return CorpusSimilaritySummary(
            total_memories_analyzed=total_analyzed,
            namespaces_analyzed=namespaces,
            bridges_found=len(bridges),
            potential_duplicates=duplicates,
            top_bridges=bridges[:10],
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def find_similar_to_memory(
        self,
        memory_id: str,
        *,
        limit: int = 10,
        min_similarity: float = 0.5,
        exclude_same_namespace: bool = False,
    ) -> list[CrossCorpusMatch]:
        """Find items similar to an existing memory.

        Convenience method that resolves a memory ID to its vector,
        then delegates to find_similar_across_corpus.

        Args:
            memory_id: ID of the source memory.
            limit: Maximum results.
            min_similarity: Floor for similarity score.
            exclude_same_namespace: If True, excludes the source namespace.

        Returns:
            Matches sorted by similarity descending.

        Raises:
            MemoryNotFoundError: If memory_id doesn't exist.
            RuntimeError: If memory_repository was not provided.
        """
        if self._memory_repo is None:
            raise RuntimeError(
                "find_similar_to_memory requires memory_repository to be provided in constructor"
            )
        result = self._memory_repo.get_with_vector(memory_id)
        if result is None:
            raise MemoryNotFoundError(memory_id)

        memory, vector = result

        return self.find_similar_across_corpus(
            vector,
            limit=limit,
            min_similarity=min_similarity,
            exclude_namespace=memory.namespace if exclude_same_namespace else None,
            exclude_ids={memory_id},
            query_content=memory.content,
            query_namespace=memory.namespace,
            query_project=memory.project,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _score_raw_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        *,
        min_similarity: float,
        limit: int,
        exclude_ids: set[str],
        query_content: str | None = None,
    ) -> list[CrossCorpusMatch]:
        """Score and filter raw dict candidates from batch_vector_search."""
        matches: list[CrossCorpusMatch] = []

        for cand in raw_candidates:
            cand_id = str(cand.get("id", ""))
            if cand_id in exclude_ids:
                continue

            raw_sim = float(cand.get("_distance", cand.get("similarity", 0.0)))

            ctx = ScoringContext(
                vector_similarity=raw_sim,
                query_content=query_content,
                candidate_content=str(cand.get("content", "")),
                query_namespace=None,
                candidate_namespace=str(cand.get("namespace", "")),
                query_project=None,
                candidate_project=str(cand.get("project", "")),
                candidate_importance=float(cand.get("importance", 0.5)),
                candidate_tags=cand.get("tags", []),
                candidate_metadata=cand.get("metadata", {}),
            )
            final_score = self._scoring.score(ctx)

            if final_score < min_similarity:
                continue

            matches.append(
                CrossCorpusMatch(
                    memory_id=cand_id,
                    content=str(cand.get("content", "")),
                    similarity=final_score,
                    raw_vector_similarity=raw_sim,
                    namespace=str(cand.get("namespace", "")),
                    project=str(cand.get("project", "")),
                    importance=float(cand.get("importance", 0.5)),
                    tags=cand.get("tags", []),
                    created_at=cand.get("created_at") or datetime.now(timezone.utc),
                    scoring_strategy=self._scoring.name,
                )
            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:limit]
