"""Spatial service for exploration operations.

This service provides the spatial layer for memory exploration:
- journey: SLERP interpolation between two memories
- wander: Temperature-based random walk through memory space
- regions: HDBSCAN clustering to discover memory regions
- visualize: UMAP projection for 2D/3D visualization

The service uses dependency injection for repository and embedding services.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from spatial_memory.core.errors import (
    ClusteringError,
    InsufficientMemoriesError,
    JourneyError,
    MemoryNotFoundError,
    ValidationError,
    VisualizationError,
    WanderError,
)
from spatial_memory.core.models import (
    JourneyResult,
    JourneyStep,
    MemoryResult,
    RegionCluster,
    RegionsResult,
    VisualizationEdge,
    VisualizationNode,
    VisualizationResult,
    WanderResult,
    WanderStep,
)
from spatial_memory.core.validation import validate_namespace, validate_uuid

logger = logging.getLogger(__name__)

# Check optional dependency availability at import time
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.debug("HDBSCAN not available - regions operation will be disabled")

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.debug("UMAP not available - visualize operation will be disabled")

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


@dataclass
class SpatialConfig:
    """Configuration for spatial operations.

    Attributes:
        journey_default_steps: Default number of interpolation steps for journey.
        journey_max_steps: Maximum allowed steps for journey.
        journey_neighbors_per_step: Number of neighbors to find per interpolation point.
        wander_default_steps: Default number of steps for random walk.
        wander_max_steps: Maximum allowed steps for wander.
        wander_default_temperature: Default temperature (randomness) for wander.
        wander_avoid_recent: Number of recent memories to avoid revisiting.
        wander_candidates_per_step: Number of candidate neighbors per step.
        regions_min_cluster_size: Minimum cluster size for HDBSCAN.
        regions_max_memories: Maximum memories to consider for clustering.
        visualize_n_neighbors: UMAP n_neighbors parameter.
        visualize_min_dist: UMAP min_dist parameter.
        visualize_max_memories: Maximum memories to include in visualization.
        visualize_similarity_threshold: Minimum similarity for edge creation.
    """

    # Journey parameters
    journey_default_steps: int = 10
    journey_max_steps: int = 20
    journey_neighbors_per_step: int = 3

    # Wander parameters
    wander_default_steps: int = 10
    wander_max_steps: int = 20
    wander_default_temperature: float = 0.5
    wander_avoid_recent: int = 5
    wander_candidates_per_step: int = 10

    # Regions parameters
    regions_min_cluster_size: int = 3
    regions_max_memories: int = 10_000

    # Visualize parameters
    visualize_n_neighbors: int = 15
    visualize_min_dist: float = 0.1
    visualize_max_memories: int = 500
    visualize_similarity_threshold: float = 0.7


# Color palette for cluster visualization
CLUSTER_COLORS = [
    "#4285F4",  # Blue
    "#EA4335",  # Red
    "#FBBC04",  # Yellow
    "#34A853",  # Green
    "#FF6D01",  # Orange
    "#46BDC6",  # Cyan
    "#7B1FA2",  # Purple
    "#E91E63",  # Pink
    "#009688",  # Teal
    "#795548",  # Brown
]


class SpatialService:
    """Service for spatial exploration of memory space.

    Uses Clean Architecture - depends on protocol interfaces, not implementations.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: SpatialConfig | None = None,
    ) -> None:
        """Initialize the spatial service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
            config: Optional configuration (uses defaults if not provided).
        """
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or SpatialConfig()

    def journey(
        self,
        start_id: str,
        end_id: str,
        steps: int | None = None,
        namespace: str | None = None,
    ) -> JourneyResult:
        """Find a path between two memories using SLERP interpolation.

        Spherical Linear Interpolation (SLERP) creates smooth paths through
        embedding space, finding actual memories closest to each interpolation
        point.

        Args:
            start_id: Starting memory UUID.
            end_id: Ending memory UUID.
            steps: Number of interpolation steps (default from config).
            namespace: Optional namespace filter for intermediate memories.

        Returns:
            JourneyResult with path steps.

        Raises:
            ValidationError: If input validation fails.
            MemoryNotFoundError: If start or end memory not found.
            JourneyError: If path cannot be computed.
        """
        # Validate inputs
        start_id = validate_uuid(start_id)
        end_id = validate_uuid(end_id)
        if namespace is not None:
            namespace = validate_namespace(namespace)

        # Get step count
        actual_steps = steps if steps is not None else self._config.journey_default_steps
        if actual_steps < 2:
            raise ValidationError("Journey requires at least 2 steps")
        if actual_steps > self._config.journey_max_steps:
            raise ValidationError(
                f"Maximum journey steps is {self._config.journey_max_steps}"
            )

        # Get start and end memories with vectors
        start_result = self._repo.get_with_vector(start_id)
        if start_result is None:
            raise MemoryNotFoundError(start_id)
        start_memory, start_vector = start_result

        end_result = self._repo.get_with_vector(end_id)
        if end_result is None:
            raise MemoryNotFoundError(end_id)
        end_memory, end_vector = end_result

        try:
            # Generate interpolation points using SLERP
            interpolated_vectors, t_values = self._slerp_interpolate(
                start_vector, end_vector, actual_steps
            )

            # Find nearest memories for each interpolation point
            # Use batch search for efficiency
            search_results = self._batch_vector_search(
                interpolated_vectors,
                limit_per_query=self._config.journey_neighbors_per_step,
                namespace=namespace,
            )

            # Build journey steps
            journey_steps: list[JourneyStep] = []
            steps_with_memories = 0

            for step_num, (interp_vec, t_val, neighbors) in enumerate(
                zip(interpolated_vectors, t_values, search_results)
            ):
                # Calculate distance from interpolation point to nearest memory
                distance_to_path = float("inf")
                if neighbors:
                    for neighbor in neighbors:
                        dist = self._cosine_distance(
                            interp_vec, self._get_vector_for_memory(neighbor.id)
                        )
                        if dist < distance_to_path:
                            distance_to_path = dist
                    steps_with_memories += 1

                # Use 0.0 if no memories found (inf means no distance calculated)
                # Clamp to 0.0 to handle floating point precision errors (e.g., -4.89e-08)
                final_distance = 0.0 if distance_to_path == float("inf") else max(0.0, distance_to_path)
                journey_steps.append(
                    JourneyStep(
                        step=step_num,
                        t=t_val,
                        position=interp_vec.tolist(),
                        nearby_memories=neighbors,
                        distance_to_path=final_distance,
                    )
                )

            # Calculate path coverage
            path_coverage = steps_with_memories / len(journey_steps) if journey_steps else 0.0

            return JourneyResult(
                start_id=start_id,
                end_id=end_id,
                steps=journey_steps,
                path_coverage=path_coverage,
            )

        except Exception as e:
            if isinstance(e, (ValidationError, MemoryNotFoundError)):
                raise
            raise JourneyError(f"Failed to compute journey: {e}") from e

    def wander(
        self,
        start_id: str,
        steps: int | None = None,
        temperature: float | None = None,
        namespace: str | None = None,
    ) -> WanderResult:
        """Perform a random walk through memory space.

        Temperature controls randomness:
        - 0.0 = Always pick the most similar (greedy)
        - 0.5 = Balanced exploration
        - 1.0 = Highly random selection

        Args:
            start_id: Starting memory UUID.
            steps: Number of steps to wander (default from config).
            temperature: Randomness factor 0.0-1.0 (default from config).
            namespace: Optional namespace filter.

        Returns:
            WanderResult with path taken.

        Raises:
            ValidationError: If input validation fails.
            MemoryNotFoundError: If start memory not found.
            WanderError: If walk cannot continue.
        """
        # Validate inputs
        start_id = validate_uuid(start_id)
        if namespace is not None:
            namespace = validate_namespace(namespace)

        # Get parameters
        actual_steps = steps if steps is not None else self._config.wander_default_steps
        if actual_steps < 1:
            raise ValidationError("Wander requires at least 1 step")
        if actual_steps > self._config.wander_max_steps:
            raise ValidationError(
                f"Maximum wander steps is {self._config.wander_max_steps}"
            )

        actual_temp = (
            temperature
            if temperature is not None
            else self._config.wander_default_temperature
        )
        if not 0.0 <= actual_temp <= 1.0:
            raise ValidationError("Temperature must be between 0.0 and 1.0")

        # Verify start memory exists
        start_result = self._repo.get_with_vector(start_id)
        if start_result is None:
            raise MemoryNotFoundError(start_id)
        current_memory, current_vector = start_result

        try:
            wander_steps: list[WanderStep] = []
            visited_ids: set[str] = {start_id}
            recent_ids: list[str] = [start_id]
            total_distance = 0.0
            prev_vector = current_vector

            for step_num in range(actual_steps):
                # Find candidates from current position
                neighbors = self._repo.search(
                    current_vector,
                    limit=self._config.wander_candidates_per_step + len(visited_ids),
                    namespace=namespace,
                )

                # Filter out recently visited
                candidates = [
                    n
                    for n in neighbors
                    if n.id not in recent_ids[-self._config.wander_avoid_recent :]
                ]

                if not candidates:
                    # No unvisited candidates - allow revisiting older memories
                    candidates = [n for n in neighbors if n.id not in visited_ids]

                if not candidates:
                    logger.warning(
                        f"Wander ended early at step {step_num}: no candidates"
                    )
                    break

                # Select next memory based on temperature
                next_memory, selection_prob = self._temperature_select(
                    candidates, actual_temp
                )

                # Calculate distance traveled
                next_result = self._repo.get_with_vector(next_memory.id)
                if next_result is None:
                    logger.warning(f"Memory {next_memory.id} disappeared during wander")
                    break
                _, next_vector = next_result

                step_distance = self._cosine_distance(prev_vector, next_vector)
                total_distance += step_distance

                wander_steps.append(
                    WanderStep(
                        step=step_num,
                        memory=next_memory,
                        similarity_to_previous=next_memory.similarity,
                        selection_probability=selection_prob,
                    )
                )

                visited_ids.add(next_memory.id)
                recent_ids.append(next_memory.id)
                current_vector = next_vector
                prev_vector = next_vector

            return WanderResult(
                start_id=start_id,
                steps=wander_steps,
                total_distance=total_distance,
            )

        except Exception as e:
            if isinstance(e, (ValidationError, MemoryNotFoundError)):
                raise
            raise WanderError(f"Wander failed: {e}") from e

    def regions(
        self,
        namespace: str | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
    ) -> RegionsResult:
        """Discover memory regions using HDBSCAN clustering.

        HDBSCAN automatically determines the number of clusters and
        identifies outliers (noise points).

        Args:
            namespace: Optional namespace filter.
            min_cluster_size: Minimum points per cluster (default from config).
            max_clusters: Maximum clusters to return (None = all).

        Returns:
            RegionsResult with discovered clusters.

        Raises:
            ValidationError: If input validation fails.
            ClusteringError: If clustering fails or HDBSCAN unavailable.
            InsufficientMemoriesError: If not enough memories for clustering.
        """
        if not HDBSCAN_AVAILABLE:
            raise ClusteringError(
                "HDBSCAN is not available. Install with: pip install hdbscan"
            )

        # Validate inputs
        if namespace is not None:
            namespace = validate_namespace(namespace)

        actual_min_size = (
            min_cluster_size
            if min_cluster_size is not None
            else self._config.regions_min_cluster_size
        )
        if actual_min_size < 2:
            raise ValidationError("Minimum cluster size must be at least 2")

        try:
            # Fetch all vectors for clustering
            all_memories = self._repo.get_all(
                namespace=namespace, limit=self._config.regions_max_memories
            )

            if len(all_memories) < actual_min_size:
                raise InsufficientMemoriesError(
                    required=actual_min_size,
                    available=len(all_memories),
                    operation="regions",
                )

            # Extract IDs and vectors
            memory_map = {m.id: (m, v) for m, v in all_memories}
            memory_ids = list(memory_map.keys())
            vectors = np.array([v for _, v in all_memories], dtype=np.float32)

            # Run HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=actual_min_size,
                metric="euclidean",  # Works well with normalized vectors
                cluster_selection_method="eom",  # Excess of Mass
            )
            labels = clusterer.fit_predict(vectors)

            # Process clusters
            clusters: list[RegionCluster] = []
            unique_labels = set(labels)

            # Remove noise label (-1) for cluster processing
            cluster_labels = [label for label in unique_labels if label >= 0]

            for cluster_id in cluster_labels:
                # Get indices of memories in this cluster
                cluster_indices = [
                    i for i, lbl in enumerate(labels) if lbl == cluster_id
                ]
                cluster_vectors = vectors[cluster_indices]
                cluster_ids = [memory_ids[i] for i in cluster_indices]

                # Find centroid and closest memory to centroid
                centroid = cluster_vectors.mean(axis=0)
                distances_to_centroid = np.linalg.norm(
                    cluster_vectors - centroid, axis=1
                )
                centroid_idx = int(np.argmin(distances_to_centroid))
                centroid_memory_id = cluster_ids[centroid_idx]

                # Calculate coherence (inverse of average intra-cluster distance)
                avg_dist = float(distances_to_centroid.mean())
                max_possible_dist = 2.0  # Max distance for normalized vectors
                coherence = max(0.0, min(1.0, 1.0 - (avg_dist / max_possible_dist)))

                # Get representative and sample memories
                rep_memory, _ = memory_map[centroid_memory_id]
                rep_result = self._memory_to_result(rep_memory, 1.0)

                sample_results: list[MemoryResult] = []
                for sid in cluster_ids[:5]:
                    mem, _ = memory_map[sid]
                    # Calculate similarity to centroid for the sample
                    mem_vec = memory_map[sid][1]
                    sim = 1.0 - self._cosine_distance(centroid, mem_vec)
                    sample_results.append(self._memory_to_result(mem, sim))

                # Extract keywords from sample content
                sample_contents = [m.content for m in sample_results]
                keywords = self._extract_keywords(" ".join(sample_contents), n=5)

                clusters.append(
                    RegionCluster(
                        cluster_id=cluster_id,
                        size=len(cluster_ids),
                        representative_memory=rep_result,
                        sample_memories=sample_results[:3],
                        coherence=coherence,
                        keywords=keywords,
                    )
                )

            # Sort by size (largest first)
            clusters.sort(key=lambda c: c.size, reverse=True)

            # Limit clusters if requested
            if max_clusters is not None and len(clusters) > max_clusters:
                clusters = clusters[:max_clusters]

            # Count noise points
            noise_count = sum(1 for lbl in labels if lbl == -1)

            # Calculate silhouette score if possible
            clustering_quality = 0.0
            if len(cluster_labels) >= 2:
                try:
                    from sklearn.metrics import silhouette_score
                    # Filter out noise points for silhouette calculation
                    mask = labels >= 0
                    if mask.sum() >= 2:
                        clustering_quality = float(
                            silhouette_score(vectors[mask], labels[mask])
                        )
                except ImportError:
                    pass  # sklearn not available, skip quality calculation

            return RegionsResult(
                clusters=clusters,
                noise_count=noise_count,
                total_memories=len(memory_ids),
                clustering_quality=clustering_quality,
            )

        except (ValidationError, InsufficientMemoriesError, ClusteringError):
            raise
        except Exception as e:
            raise ClusteringError(f"Clustering failed: {e}") from e

    def visualize(
        self,
        memory_ids: list[str] | None = None,
        namespace: str | None = None,
        format: Literal["json", "mermaid", "svg"] = "json",
        dimensions: Literal[2, 3] = 2,
        include_edges: bool = True,
    ) -> VisualizationResult:
        """Generate a visualization of memory space using UMAP projection.

        Args:
            memory_ids: Specific memories to visualize (None = auto-select).
            namespace: Namespace filter when auto-selecting.
            format: Output format (json, mermaid, or svg).
            dimensions: Number of dimensions (2 or 3).
            include_edges: Include similarity edges between nodes.

        Returns:
            VisualizationResult with visualization data and formatted output.

        Raises:
            ValidationError: If input validation fails.
            VisualizationError: If visualization fails or UMAP unavailable.
            InsufficientMemoriesError: If not enough memories.
        """
        if not UMAP_AVAILABLE:
            raise VisualizationError(
                "UMAP is not available. Install with: pip install umap-learn"
            )

        # Validate inputs
        if namespace is not None:
            namespace = validate_namespace(namespace)

        if memory_ids is not None:
            memory_ids = [validate_uuid(mid) for mid in memory_ids]

        if dimensions not in (2, 3):
            raise ValidationError("Dimensions must be 2 or 3")

        try:
            # Get memories to visualize
            if memory_ids:
                memories_with_vectors: list[tuple[Any, np.ndarray]] = []
                for mid in memory_ids[: self._config.visualize_max_memories]:
                    result = self._repo.get_with_vector(mid)
                    if result:
                        memories_with_vectors.append(result)
            else:
                memories_with_vectors = self._repo.get_all(
                    namespace=namespace, limit=self._config.visualize_max_memories
                )

            if len(memories_with_vectors) < 5:
                raise InsufficientMemoriesError(
                    required=5,
                    available=len(memories_with_vectors),
                    operation="visualize",
                )

            # Extract vectors
            vectors = np.array(
                [v for _, v in memories_with_vectors], dtype=np.float32
            )

            # Run UMAP projection
            n_neighbors = min(
                self._config.visualize_n_neighbors, len(vectors) - 1
            )
            reducer = umap.UMAP(
                n_components=dimensions,
                n_neighbors=n_neighbors,
                min_dist=self._config.visualize_min_dist,
                metric="cosine",
                random_state=42,  # Reproducibility
            )
            embedding = reducer.fit_transform(vectors)

            # Optionally run clustering for coloring
            cluster_labels = [-1] * len(memories_with_vectors)

            if HDBSCAN_AVAILABLE and len(memories_with_vectors) >= 10:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=3,
                        metric="euclidean",
                    )
                    cluster_labels = clusterer.fit_predict(vectors).tolist()
                except Exception as e:
                    logger.debug(f"Clustering for visualization failed: {e}")

            # Build visualization nodes
            nodes: list[VisualizationNode] = []
            for i, (memory, _) in enumerate(memories_with_vectors):
                # Create short label from content
                content = memory.content
                label = content[:50] + "..." if len(content) > 50 else content
                label = label.replace("\n", " ")

                nodes.append(
                    VisualizationNode(
                        id=memory.id,
                        x=float(embedding[i, 0]),
                        y=float(embedding[i, 1]) if dimensions >= 2 else 0.0,
                        label=label,
                        cluster=cluster_labels[i],
                        importance=memory.importance,
                        highlighted=False,
                    )
                )

            # Build edges if requested
            edges: list[VisualizationEdge] = []
            if include_edges:
                # Calculate pairwise similarities and create edges for high similarity
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        similarity = 1.0 - self._cosine_distance(vectors[i], vectors[j])
                        if similarity >= self._config.visualize_similarity_threshold:
                            edges.append(
                                VisualizationEdge(
                                    from_id=nodes[i].id,
                                    to_id=nodes[j].id,
                                    weight=similarity,
                                )
                            )

            # Calculate bounds
            x_coords = [n.x for n in nodes]
            y_coords = [n.y for n in nodes]
            bounds = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
            }

            # Format output
            output = self._format_output(nodes, edges, format)

            return VisualizationResult(
                nodes=nodes,
                edges=edges,
                bounds=bounds,
                format=format,
                output=output,
            )

        except (ValidationError, InsufficientMemoriesError, VisualizationError):
            raise
        except Exception as e:
            raise VisualizationError(f"Visualization failed: {e}") from e

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _memory_to_result(self, memory: Any, similarity: float) -> MemoryResult:
        """Convert a Memory object to a MemoryResult.

        Args:
            memory: Memory object.
            similarity: Similarity score.

        Returns:
            MemoryResult object.
        """
        return MemoryResult(
            id=memory.id,
            content=memory.content,
            similarity=max(0.0, min(1.0, similarity)),
            namespace=memory.namespace,
            tags=memory.tags,
            importance=memory.importance,
            created_at=memory.created_at,
            metadata=memory.metadata,
        )

    def _slerp_interpolate(
        self,
        start_vec: np.ndarray,
        end_vec: np.ndarray,
        num_steps: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Spherical Linear Interpolation between two vectors.

        SLERP maintains constant angular velocity along the geodesic path
        between two points on a hypersphere, making it ideal for semantic
        interpolation in embedding space.

        Args:
            start_vec: Starting vector.
            end_vec: Ending vector.
            num_steps: Number of interpolation points.

        Returns:
            Tuple of (interpolated vectors, t values).
        """
        # Normalize vectors
        start_norm = start_vec / (np.linalg.norm(start_vec) + 1e-10)
        end_norm = end_vec / (np.linalg.norm(end_vec) + 1e-10)

        # Calculate angle between vectors
        dot = np.clip(np.dot(start_norm, end_norm), -1.0, 1.0)
        omega = np.arccos(dot)

        t_values = list(np.linspace(0, 1, num_steps))

        # Handle nearly parallel vectors (use linear interpolation)
        if omega < 1e-6:
            linear_interp = [
                start_vec + t * (end_vec - start_vec)
                for t in t_values
            ]
            return linear_interp, t_values

        sin_omega = np.sin(omega)

        interpolated: list[np.ndarray] = []
        for t in t_values:
            coef_start = np.sin((1 - t) * omega) / sin_omega
            coef_end = np.sin(t * omega) / sin_omega
            vec = coef_start * start_norm + coef_end * end_norm
            interpolated.append(vec)

        return interpolated, t_values

    def _batch_vector_search(
        self,
        vectors: list[np.ndarray],
        limit_per_query: int,
        namespace: str | None,
    ) -> list[list[MemoryResult]]:
        """Perform batch vector search.

        Delegates to repository if batch search is available, otherwise
        performs individual searches.

        Args:
            vectors: List of query vectors.
            limit_per_query: Results per query.
            namespace: Optional namespace filter.

        Returns:
            List of result lists.
        """
        # Fall back to individual searches (repository handles batch internally)
        results: list[list[MemoryResult]] = []
        for vec in vectors:
            neighbors = self._repo.search(vec, limit=limit_per_query, namespace=namespace)
            results.append(neighbors)
        return results

    def _get_vector_for_memory(self, memory_id: str) -> np.ndarray:
        """Get the vector for a memory.

        Args:
            memory_id: Memory UUID.

        Returns:
            The memory's vector.
        """
        result = self._repo.get_with_vector(memory_id)
        if result is None:
            # Return zero vector if memory not found (shouldn't happen in practice)
            return np.zeros(self._embeddings.dimensions, dtype=np.float32)
        _, vector = result
        return vector

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine distance (0 = identical, 2 = opposite).
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1.0  # Maximum distance for zero vectors

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - similarity)

    def _temperature_select(
        self,
        candidates: list[MemoryResult],
        temperature: float,
    ) -> tuple[MemoryResult, float]:
        """Select a candidate using temperature-based sampling.

        Args:
            candidates: List of candidate memories with similarity scores.
            temperature: Randomness factor (0 = greedy, 1 = uniform random).

        Returns:
            Tuple of (selected memory, selection probability).
        """
        if not candidates:
            raise WanderError("No candidates for temperature selection")

        if temperature == 0.0:
            # Greedy: pick highest similarity
            return max(candidates, key=lambda c: c.similarity), 1.0

        if temperature >= 1.0:
            # Random: uniform selection
            prob = 1.0 / len(candidates)
            return random.choice(candidates), prob

        # Temperature-based softmax selection
        similarities = np.array([c.similarity for c in candidates])

        # Scale by inverse temperature (lower temp = sharper distribution)
        scaled = similarities / (temperature + 1e-10)
        scaled = scaled - scaled.max()  # Numerical stability
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum()

        # Sample according to probabilities
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx], float(probs[idx])

    def _extract_keywords(self, text: str, n: int = 5) -> list[str]:
        """Extract top keywords from text using simple frequency analysis.

        Args:
            text: Text to analyze.
            n: Number of keywords to extract.

        Returns:
            List of top keywords.
        """
        # Simple keyword extraction using word frequency
        # Remove common stop words and short words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "also", "now", "and", "but", "or", "if", "it", "its",
            "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
        }

        # Tokenize and filter
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        filtered = [w for w in words if w not in stop_words and len(w) > 2]

        # Count frequencies
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(n)]

    def _format_output(
        self,
        nodes: list[VisualizationNode],
        edges: list[VisualizationEdge],
        format: Literal["json", "mermaid", "svg"],
    ) -> str:
        """Format visualization data for output.

        Args:
            nodes: Visualization nodes.
            edges: Visualization edges.
            format: Output format.

        Returns:
            Formatted string output.
        """
        if format == "json":
            return json.dumps(
                {
                    "nodes": [
                        {
                            "id": n.id,
                            "x": n.x,
                            "y": n.y,
                            "label": n.label,
                            "cluster": n.cluster,
                            "importance": n.importance,
                        }
                        for n in nodes
                    ],
                    "edges": [
                        {
                            "from": e.from_id,
                            "to": e.to_id,
                            "weight": e.weight,
                        }
                        for e in edges
                    ],
                },
                indent=2,
            )

        elif format == "mermaid":
            lines = ["graph LR"]

            # Add nodes with short IDs
            node_aliases = {n.id: f"N{i}" for i, n in enumerate(nodes)}
            for node in nodes:
                alias = node_aliases[node.id]
                # Escape special characters in label
                safe_label = node.label.replace('"', "'").replace("\n", " ")[:30]
                lines.append(f'    {alias}["{safe_label}"]')

            # Add edges
            for edge in edges:
                from_alias = node_aliases.get(edge.from_id)
                to_alias = node_aliases.get(edge.to_id)
                if from_alias and to_alias:
                    lines.append(f"    {from_alias} --> {to_alias}")

            return "\n".join(lines)

        elif format == "svg":
            return self._generate_svg(nodes, edges)

        else:
            raise ValidationError(f"Unknown format: {format}")

    def _generate_svg(
        self,
        nodes: list[VisualizationNode],
        edges: list[VisualizationEdge],
    ) -> str:
        """Generate SVG visualization.

        Args:
            nodes: Visualization nodes.
            edges: Visualization edges.

        Returns:
            SVG string.
        """
        width, height = 800, 600
        padding = 50

        # Calculate scale to fit nodes
        x_coords = [n.x for n in nodes]
        y_coords = [n.y for n in nodes]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1

        def scale_x(x: float) -> float:
            return padding + (x - x_min) / x_range * (width - 2 * padding)

        def scale_y(y: float) -> float:
            return padding + (y - y_min) / y_range * (height - 2 * padding)

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            "  <style>",
            "    .node { cursor: pointer; }",
            "    .node circle { stroke: #333; stroke-width: 1; }",
            "    .node text { font-size: 10px; fill: #333; }",
            "    .edge { stroke: #ccc; stroke-width: 1; opacity: 0.5; }",
            "  </style>",
        ]

        # Draw edges
        for edge in edges:
            from_node = next((n for n in nodes if n.id == edge.from_id), None)
            to_node = next((n for n in nodes if n.id == edge.to_id), None)
            if from_node and to_node:
                x1, y1 = scale_x(from_node.x), scale_y(from_node.y)
                x2, y2 = scale_x(to_node.x), scale_y(to_node.y)
                svg_lines.append(
                    f'  <line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" '
                    f'x2="{x2:.1f}" y2="{y2:.1f}" />'
                )

        # Draw nodes
        for node in nodes:
            x, y = scale_x(node.x), scale_y(node.y)
            radius = 5 + node.importance * 5  # Scale by importance
            if node.cluster >= 0:
                color = CLUSTER_COLORS[node.cluster % len(CLUSTER_COLORS)]
            else:
                color = "#999"

            svg_lines.append('  <g class="node">')
            svg_lines.append(
                f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                f'fill="{color}" />'
            )
            # Add truncated label
            short_label = node.label[:20] + "..." if len(node.label) > 20 else node.label
            # Escape XML special characters
            short_label = (
                short_label.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            svg_lines.append(
                f'    <text x="{x:.1f}" y="{y + radius + 12:.1f}" '
                f'text-anchor="middle">{short_label}</text>'
            )
            svg_lines.append("  </g>")

        svg_lines.append("</svg>")
        return "\n".join(svg_lines)
