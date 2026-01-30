# Phase 4B Implementation Plan: Spatial Operations

## Executive Summary

This plan details the implementation of four spatial operations (journey, wander, regions, visualize) for the Spatial Memory MCP Server. Based on analysis from 4 specialized agents, this follows Clean Architecture principles, TDD methodology, and industry standards.

**Current State:** Phase 4A complete with 392 tests passing, 7 of 19 tools implemented
**Phase 4B Goal:** Add journey, wander, regions, visualize tools (4 tools, bringing total to 11)

---

## 1. Architecture Decision: Separate SpatialService

### Rationale
- **Single Responsibility**: MemoryService handles CRUD; SpatialService handles navigation/exploration
- **Dependency Isolation**: HDBSCAN/UMAP may be optional; separate service enables graceful degradation
- **Testing Isolation**: Different mocking strategies for CRUD vs spatial algorithms

### File Structure

```
spatial_memory/
├── core/
│   ├── spatial_ops.py      # NEW: Core algorithms (SLERP, temperature selection, etc.)
│   └── ...
├── services/
│   ├── memory.py           # Existing CRUD operations
│   └── spatial.py          # NEW: SpatialService
├── ports/
│   └── repositories.py     # EXTEND: Add spatial protocol methods
└── server.py               # UPDATE: Register 4 new tools
```

---

## 2. Core Algorithms Module

### File: `spatial_memory/core/spatial_ops.py`

```python
"""Core spatial algorithms for memory navigation and exploration."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from spatial_memory.core.database import Database

logger = logging.getLogger(__name__)

T = TypeVar("T")
Vector = NDArray[np.float32]


# =============================================================================
# Vector Operations
# =============================================================================

def normalize(v: Vector) -> Vector:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def normalize_batch(vectors: Vector, copy: bool = True) -> Vector:
    """Normalize multiple vectors efficiently."""
    if copy:
        vectors = vectors.copy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors /= norms
    return vectors


# =============================================================================
# SLERP (Spherical Linear Interpolation)
# =============================================================================

def slerp(v0: Vector, v1: Vector, t: float) -> Vector:
    """
    Spherical linear interpolation between two unit vectors.

    Handles edge cases:
    - Parallel vectors (omega ≈ 0): Falls back to linear interpolation
    - Antipodal vectors (omega ≈ π): Chooses arbitrary perpendicular path

    Args:
        v0: Starting unit vector.
        v1: Ending unit vector.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit vector at parameter t.
    """
    v0 = normalize(v0.astype(np.float64))
    v1 = normalize(v1.astype(np.float64))

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)

    # Handle nearly parallel vectors (dot ~ 1.0)
    if dot > 0.9995:
        result = v0 + t * (v1 - v0)
        return normalize(result.astype(np.float32))

    # Handle nearly antipodal vectors (dot ~ -1.0)
    if dot < -0.9995:
        perp = _find_perpendicular(v0)
        half_angle = np.pi * t
        result = v0 * np.cos(half_angle) + perp * np.sin(half_angle)
        return result.astype(np.float32)

    # Standard SLERP
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    return (s0 * v0 + s1 * v1).astype(np.float32)


def _find_perpendicular(v: Vector) -> Vector:
    """Find a unit vector perpendicular to v."""
    min_idx = np.argmin(np.abs(v))
    basis = np.zeros_like(v)
    basis[min_idx] = 1.0
    perp = basis - np.dot(v, basis) * v
    return normalize(perp)


def slerp_path(
    v0: Vector,
    v1: Vector,
    steps: int,
    include_endpoints: bool = True
) -> list[Vector]:
    """Generate N interpolation steps between two vectors using SLERP."""
    if steps < 1:
        raise ValueError("steps must be at least 1")

    vectors = []
    if include_endpoints:
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            vectors.append(slerp(v0, v1, t))
    else:
        for i in range(steps):
            t = (i + 1) / (steps + 1)
            vectors.append(slerp(v0, v1, t))

    return vectors


# =============================================================================
# Temperature-based Selection (for Wander)
# =============================================================================

def softmax_with_temperature(
    scores: NDArray[np.float64],
    temperature: float = 1.0
) -> NDArray[np.float64]:
    """
    Compute softmax probabilities with temperature scaling.

    Temperature controls randomness:
    - T -> 0: Deterministic (always pick highest score)
    - T = 1: Standard softmax
    - T -> inf: Uniform random selection
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative")

    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        return np.array([], dtype=np.float64)

    if len(scores) == 1:
        return np.array([1.0], dtype=np.float64)

    # Handle temperature = 0 (greedy/deterministic)
    if temperature < 1e-10:
        result = np.zeros_like(scores)
        result[np.argmax(scores)] = 1.0
        return result

    # Scale scores by temperature
    scaled = scores / temperature
    scaled_shifted = scaled - np.max(scaled)  # Numerical stability
    exp_scores = np.exp(scaled_shifted)
    return exp_scores / np.sum(exp_scores)


def temperature_select(
    items: Sequence[T],
    scores: NDArray[np.float64],
    temperature: float = 1.0,
    rng: np.random.Generator | None = None
) -> T:
    """Select an item using temperature-scaled softmax probabilities."""
    if len(items) != len(scores):
        raise ValueError("items and scores must have same length")

    if rng is None:
        rng = np.random.default_rng()

    probabilities = softmax_with_temperature(scores, temperature)
    idx = rng.choice(len(items), p=probabilities)
    return items[idx]


# =============================================================================
# HDBSCAN Clustering (for Regions)
# =============================================================================

@dataclass
class ClusterInfo:
    """Information about a discovered cluster."""
    cluster_id: int
    size: int
    centroid: Vector
    centroid_memory_id: str
    sample_memory_ids: list[str]
    coherence: float  # Average pairwise similarity within cluster
    keywords: list[str] = field(default_factory=list)


def configure_hdbscan(
    n_samples: int,
    min_cluster_size: int | None = None,
    min_samples: int | None = None
) -> dict:
    """Configure HDBSCAN parameters based on dataset characteristics."""
    if min_cluster_size is None:
        min_cluster_size = max(3, int(np.sqrt(n_samples) / 2))
        min_cluster_size = min(min_cluster_size, 50)

    if min_samples is None:
        min_samples = max(2, min_cluster_size // 2)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": "euclidean",  # Use with normalized vectors
        "cluster_selection_method": "eom",
        "core_dist_n_jobs": -1,
    }


# =============================================================================
# UMAP Projection (for Visualize)
# =============================================================================

def configure_umap(
    n_samples: int,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> dict:
    """Configure UMAP parameters for memory visualization."""
    return {
        "n_components": n_components,
        "n_neighbors": min(n_neighbors, n_samples - 1),
        "min_dist": min_dist,
        "metric": "cosine",
        "random_state": random_state,
        "low_memory": n_samples > 5000,
    }
```

---

## 3. Domain Models

### File: `spatial_memory/core/models.py` (additions)

```python
# Add to existing models.py

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JourneyStep:
    """A step in the semantic journey."""
    step: int
    t: float  # Interpolation parameter [0, 1]
    position: list[float]  # Interpolated vector as list
    nearby_memories: list[MemoryResult]
    distance_to_path: float = 0.0


@dataclass
class JourneyResult:
    """Result of journey between two memories."""
    start_id: str
    end_id: str
    steps: list[JourneyStep]
    path_coverage: float  # Percentage of steps with nearby memories


@dataclass
class WanderStep:
    """A step in a semantic random walk."""
    step: int
    memory: MemoryResult
    similarity_to_previous: float
    selection_probability: float


@dataclass
class WanderResult:
    """Result of semantic random walk."""
    start_id: str
    steps: list[WanderStep]
    total_distance: float


@dataclass
class RegionCluster:
    """A discovered semantic region/cluster."""
    cluster_id: int
    size: int
    representative_memory: MemoryResult
    sample_memories: list[MemoryResult]
    coherence: float
    keywords: list[str]


@dataclass
class RegionsResult:
    """Result of clustering operation."""
    clusters: list[RegionCluster]
    noise_count: int
    total_memories: int
    clustering_quality: float


@dataclass
class VisualizationNode:
    """A node in the 2D/3D visualization."""
    id: str
    x: float
    y: float
    z: float | None = None
    label: str = ""
    cluster_id: int = -1
    importance: float = 0.5


@dataclass
class VisualizationEdge:
    """An edge connecting two memories in visualization."""
    source: str
    target: str
    weight: float


@dataclass
class VisualizationResult:
    """Result of visualization projection."""
    nodes: list[VisualizationNode]
    edges: list[VisualizationEdge]
    bounds: dict[str, float]
    format: str
    output: str  # JSON, Mermaid, or SVG string
```

---

## 4. Port Protocol Extensions

### File: `spatial_memory/ports/repositories.py` (additions)

```python
# Add to MemoryRepositoryProtocol

class MemoryRepositoryProtocol(Protocol):
    # ... existing methods ...

    def get_vectors_for_clustering(
        self,
        namespace: str | None = None,
        max_memories: int = 10_000,
    ) -> tuple[list[str], NDArray[np.float32]]:
        """
        Extract memory IDs and vectors efficiently for clustering.

        Returns:
            Tuple of (memory_ids, vectors_array).
        """
        ...

    def batch_vector_search(
        self,
        query_vectors: list[NDArray[np.float32]],
        limit_per_query: int = 3,
        namespace: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Search for memories near multiple query points.

        Returns:
            List of result lists, one per query vector.
        """
        ...
```

---

## 5. SpatialService Implementation

### File: `spatial_memory/services/spatial.py`

```python
"""Service for spatial memory navigation and exploration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from spatial_memory.core.errors import (
    ClusteringError,
    InsufficientMemoriesError,
    MemoryNotFoundError,
    ValidationError,
    VisualizationError,
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
from spatial_memory.core.spatial_ops import (
    ClusterInfo,
    configure_hdbscan,
    configure_umap,
    normalize,
    normalize_batch,
    slerp_path,
    softmax_with_temperature,
    temperature_select,
)
from spatial_memory.core.validation import validate_namespace, validate_uuid

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )

logger = logging.getLogger(__name__)


# Check optional dependencies at import time
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None  # type: ignore

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None  # type: ignore


@dataclass
class SpatialConfig:
    """Configuration for spatial operations."""

    # Journey
    journey_default_steps: int = 10
    journey_max_steps: int = 20
    journey_neighbors_per_step: int = 3

    # Wander
    wander_default_steps: int = 10
    wander_max_steps: int = 20
    wander_default_temperature: float = 0.5
    wander_avoid_recent: int = 5
    wander_candidates_per_step: int = 10

    # Regions
    regions_min_cluster_size: int = 3
    regions_max_memories: int = 10_000

    # Visualize
    visualize_n_neighbors: int = 15
    visualize_min_dist: float = 0.1
    visualize_max_memories: int = 500
    visualize_default_dimensions: int = 2
    visualize_similarity_threshold: float = 0.7


class SpatialService:
    """Service for spatial navigation and exploration of memory space."""

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: SpatialConfig | None = None,
    ) -> None:
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or SpatialConfig()

    # =========================================================================
    # Journey: SLERP interpolation between two memories
    # =========================================================================

    def journey(
        self,
        start_id: str,
        end_id: str,
        steps: int | None = None,
        namespace: str | None = None,
    ) -> JourneyResult:
        """
        Navigate semantic space between two memories using SLERP.

        Args:
            start_id: Starting memory UUID.
            end_id: Ending memory UUID.
            steps: Number of interpolation steps (default: config value).
            namespace: Optional namespace filter for nearby search.

        Returns:
            JourneyResult with path and discovered memories.

        Raises:
            MemoryNotFoundError: If start or end memory not found.
            ValidationError: If parameters are invalid.
        """
        # Validate
        validate_uuid(start_id)
        validate_uuid(end_id)

        steps = steps or self._config.journey_default_steps
        if steps < 2:
            raise ValidationError("Journey requires at least 2 steps")
        if steps > self._config.journey_max_steps:
            raise ValidationError(
                f"Journey steps cannot exceed {self._config.journey_max_steps}"
            )

        if namespace is not None:
            validate_namespace(namespace)

        # Get start and end memories
        start_memory = self._repo.get_with_vector(start_id)
        end_memory = self._repo.get_with_vector(end_id)

        start_vector = np.array(start_memory["vector"], dtype=np.float32)
        end_vector = np.array(end_memory["vector"], dtype=np.float32)

        # Generate interpolation path
        interpolated = slerp_path(start_vector, end_vector, steps)

        # Batch search for nearby memories at all interpolation points
        search_results = self._repo.batch_vector_search(
            query_vectors=interpolated,
            limit_per_query=self._config.journey_neighbors_per_step + 2,
            namespace=namespace,
        )

        # Build journey steps
        seen_ids = {start_id, end_id}
        journey_steps = []

        for i, (vector, results) in enumerate(zip(interpolated, search_results)):
            t = i / (steps - 1) if steps > 1 else 0.0

            # Find nearby memories not yet seen
            nearby = []
            for r in results:
                if r["id"] not in seen_ids:
                    nearby.append(MemoryResult.from_dict(r))
                    seen_ids.add(r["id"])
                    if len(nearby) >= self._config.journey_neighbors_per_step:
                        break

            journey_steps.append(JourneyStep(
                step=i,
                t=t,
                position=vector.tolist(),
                nearby_memories=nearby,
                distance_to_path=0.0,  # Will be computed if needed
            ))

        # Calculate path coverage
        steps_with_memories = sum(1 for s in journey_steps if s.nearby_memories)
        path_coverage = steps_with_memories / steps if steps > 0 else 0.0

        return JourneyResult(
            start_id=start_id,
            end_id=end_id,
            steps=journey_steps,
            path_coverage=path_coverage,
        )

    # =========================================================================
    # Wander: Random walk exploration
    # =========================================================================

    def wander(
        self,
        start_id: str | None = None,
        steps: int | None = None,
        temperature: float | None = None,
        namespace: str | None = None,
    ) -> WanderResult:
        """
        Explore memory space through random walk.

        Args:
            start_id: Starting memory UUID (random if not provided).
            steps: Number of steps (default: config value).
            temperature: Randomness (0.1=focused, 2.0=random).
            namespace: Optional namespace filter.

        Returns:
            WanderResult with path taken.

        Raises:
            MemoryNotFoundError: If start memory not found.
            ValidationError: If parameters are invalid.
        """
        steps = steps or self._config.wander_default_steps
        temperature = temperature or self._config.wander_default_temperature

        if steps < 1:
            raise ValidationError("Wander requires at least 1 step")
        if steps > self._config.wander_max_steps:
            raise ValidationError(
                f"Wander steps cannot exceed {self._config.wander_max_steps}"
            )
        if not 0.1 <= temperature <= 5.0:
            raise ValidationError("Temperature must be between 0.1 and 5.0")

        if namespace is not None:
            validate_namespace(namespace)

        # Get starting memory
        if start_id is not None:
            validate_uuid(start_id)
            current = self._repo.get_with_vector(start_id)
        else:
            # Pick random starting memory
            all_memories = self._repo.get_all(namespace=namespace, limit=100)
            if not all_memories:
                raise ValidationError("No memories to explore")
            current = all_memories[np.random.randint(len(all_memories))]
            start_id = current["id"]

        # Walk
        visited_ids = [start_id]
        current_vector = np.array(current["vector"], dtype=np.float32)
        wander_steps = []
        total_distance = 0.0
        rng = np.random.default_rng()

        for step_num in range(steps):
            # Search for candidates
            results = self._repo.vector_search(
                query_vector=current_vector,
                limit=self._config.wander_candidates_per_step +
                      self._config.wander_avoid_recent + 1,
                namespace=namespace,
            )

            # Filter candidates
            candidates = []
            for r in results:
                # Skip self and recently visited
                if r["id"] in visited_ids[-self._config.wander_avoid_recent:]:
                    continue
                # Skip very similar (near-duplicates)
                if r.get("similarity", 0) > 0.95:
                    continue
                candidates.append(r)
                if len(candidates) >= self._config.wander_candidates_per_step:
                    break

            if not candidates:
                logger.warning(f"Wander stuck at step {step_num}: no valid candidates")
                break

            # Temperature-based selection
            scores = np.array([c.get("similarity", 0.5) for c in candidates])
            selected = temperature_select(candidates, scores, temperature, rng)

            # Record step
            prob = softmax_with_temperature(scores, temperature)
            selected_idx = candidates.index(selected)

            wander_steps.append(WanderStep(
                step=step_num,
                memory=MemoryResult.from_dict(selected),
                similarity_to_previous=selected.get("similarity", 0),
                selection_probability=float(prob[selected_idx]),
            ))

            # Update state
            total_distance += 1.0 - selected.get("similarity", 0)
            visited_ids.append(selected["id"])
            current_vector = np.array(selected["vector"], dtype=np.float32)

        return WanderResult(
            start_id=start_id,
            steps=wander_steps,
            total_distance=total_distance,
        )

    # =========================================================================
    # Regions: HDBSCAN clustering
    # =========================================================================

    def regions(
        self,
        namespace: str | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
    ) -> RegionsResult:
        """
        Discover semantic clusters in memory space using HDBSCAN.

        Args:
            namespace: Optional namespace filter.
            min_cluster_size: Minimum memories per cluster.
            max_clusters: Maximum clusters to return.

        Returns:
            RegionsResult with discovered clusters.

        Raises:
            InsufficientMemoriesError: If too few memories.
            ClusteringError: If HDBSCAN fails.
        """
        if not HDBSCAN_AVAILABLE:
            raise ClusteringError(
                "HDBSCAN not installed. Install with: pip install hdbscan"
            )

        min_cluster_size = min_cluster_size or self._config.regions_min_cluster_size

        if namespace is not None:
            validate_namespace(namespace)

        # Get vectors
        ids, vectors = self._repo.get_vectors_for_clustering(
            namespace=namespace,
            max_memories=self._config.regions_max_memories,
        )

        if len(ids) < min_cluster_size * 2:
            raise InsufficientMemoriesError(
                required=min_cluster_size * 2,
                available=len(ids),
                operation="regions",
            )

        # Normalize for cosine distance
        vectors = normalize_batch(vectors)

        # Configure and run HDBSCAN
        params = configure_hdbscan(len(ids), min_cluster_size)
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(vectors)

        # Compute silhouette score if possible
        clustering_quality = 0.0
        unique_labels = set(labels) - {-1}
        if len(unique_labels) > 1:
            try:
                from sklearn.metrics import silhouette_score
                mask = labels != -1
                if mask.sum() >= 2:
                    clustering_quality = float(silhouette_score(
                        vectors[mask], labels[mask], metric="euclidean"
                    ))
            except ImportError:
                pass

        # Build cluster info
        clusters = []

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_ids = [ids[i] for i in np.where(mask)[0]]
            cluster_vectors = vectors[mask]

            # Centroid
            centroid = np.mean(cluster_vectors, axis=0)
            centroid = normalize(centroid)

            # Find representative (closest to centroid)
            distances = 1 - np.dot(cluster_vectors, centroid)
            rep_idx = np.argmin(distances)
            rep_id = cluster_ids[rep_idx]
            rep_memory = self._repo.get(rep_id)

            # Sample memories
            sample_size = min(5, len(cluster_ids))
            sample_indices = np.random.choice(len(cluster_ids), sample_size, replace=False)
            sample_ids = [cluster_ids[i] for i in sample_indices]
            sample_memories = [
                MemoryResult.from_dict(self._repo.get(mid)) for mid in sample_ids
            ]

            # Coherence
            coherence = 0.0
            if len(cluster_vectors) >= 2:
                sim_matrix = np.dot(cluster_vectors[:100], cluster_vectors[:100].T)
                n = len(sim_matrix)
                coherence = float(np.mean(sim_matrix[np.triu_indices(n, k=1)]))

            # Keywords (simple: first few words from representative)
            keywords = self._extract_keywords(rep_memory.get("content", ""), n=5)

            clusters.append(RegionCluster(
                cluster_id=int(label),
                size=int(mask.sum()),
                representative_memory=MemoryResult.from_dict(rep_memory),
                sample_memories=sample_memories,
                coherence=coherence,
                keywords=keywords,
            ))

        # Sort by size, limit if requested
        clusters.sort(key=lambda c: c.size, reverse=True)
        if max_clusters:
            clusters = clusters[:max_clusters]

        return RegionsResult(
            clusters=clusters,
            noise_count=int((labels == -1).sum()),
            total_memories=len(ids),
            clustering_quality=clustering_quality,
        )

    def _extract_keywords(self, text: str, n: int = 5) -> list[str]:
        """Extract top N keywords from text (simple implementation)."""
        # Simple word frequency approach
        words = text.lower().split()
        # Filter short words and common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "can", "to", "of",
                      "in", "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "again", "further", "then", "once", "and",
                      "but", "if", "or", "because", "until", "while", "this", "that"}
        filtered = [w for w in words if len(w) > 2 and w not in stop_words]

        # Count frequencies
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        # Return top N
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:n]]

    # =========================================================================
    # Visualize: UMAP projection
    # =========================================================================

    def visualize(
        self,
        memory_ids: list[str] | None = None,
        namespace: str | None = None,
        format: Literal["json", "mermaid", "svg"] = "json",
        dimensions: int = 2,
        include_edges: bool = True,
    ) -> VisualizationResult:
        """
        Generate visualization of memory space.

        Args:
            memory_ids: Specific memories to visualize (or all in namespace).
            namespace: Namespace filter (if memory_ids not specified).
            format: Output format (json, mermaid, svg).
            dimensions: 2 or 3 dimensional projection.
            include_edges: Include similarity edges between memories.

        Returns:
            VisualizationResult with nodes, edges, and formatted output.

        Raises:
            VisualizationError: If UMAP fails or too few memories.
        """
        if not UMAP_AVAILABLE:
            raise VisualizationError(
                "UMAP not installed. Install with: pip install umap-learn"
            )

        if dimensions not in (2, 3):
            raise ValidationError("Dimensions must be 2 or 3")

        if namespace is not None:
            validate_namespace(namespace)

        # Get memories
        if memory_ids:
            for mid in memory_ids:
                validate_uuid(mid)
            memories = [self._repo.get_with_vector(mid) for mid in memory_ids]
        else:
            ids, vectors = self._repo.get_vectors_for_clustering(
                namespace=namespace,
                max_memories=self._config.visualize_max_memories,
            )
            memories = []
            for mid in ids:
                try:
                    memories.append(self._repo.get_with_vector(mid))
                except Exception:
                    pass

        if len(memories) < 3:
            raise VisualizationError("Need at least 3 memories to visualize")

        # Extract vectors
        ids = [m["id"] for m in memories]
        vectors = np.array([m["vector"] for m in memories], dtype=np.float32)

        # Run UMAP
        params = configure_umap(
            len(ids),
            n_components=dimensions,
            n_neighbors=self._config.visualize_n_neighbors,
            min_dist=self._config.visualize_min_dist,
        )
        reducer = umap.UMAP(**params)
        coords = reducer.fit_transform(vectors)

        # Build nodes
        nodes = []
        for i, memory in enumerate(memories):
            node = VisualizationNode(
                id=memory["id"],
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if dimensions == 3 else None,
                label=memory.get("content", "")[:50],
                cluster_id=-1,
                importance=memory.get("importance", 0.5),
            )
            nodes.append(node)

        # Build edges (similarity connections)
        edges = []
        if include_edges:
            sim_threshold = self._config.visualize_similarity_threshold
            vectors_norm = normalize_batch(vectors)
            sim_matrix = np.dot(vectors_norm, vectors_norm.T)

            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    if sim_matrix[i, j] >= sim_threshold:
                        edges.append(VisualizationEdge(
                            source=ids[i],
                            target=ids[j],
                            weight=float(sim_matrix[i, j]),
                        ))

        # Calculate bounds
        bounds = {
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
        }
        if dimensions == 3:
            bounds["z_min"] = float(coords[:, 2].min())
            bounds["z_max"] = float(coords[:, 2].max())

        # Format output
        output = self._format_output(nodes, edges, format)

        return VisualizationResult(
            nodes=nodes,
            edges=edges,
            bounds=bounds,
            format=format,
            output=output,
        )

    def _format_output(
        self,
        nodes: list[VisualizationNode],
        edges: list[VisualizationEdge],
        format: str,
    ) -> str:
        """Format visualization output."""
        import json

        if format == "json":
            return json.dumps({
                "nodes": [
                    {
                        "id": n.id,
                        "x": n.x,
                        "y": n.y,
                        "z": n.z,
                        "label": n.label,
                        "importance": n.importance,
                    }
                    for n in nodes
                ],
                "edges": [
                    {"source": e.source, "target": e.target, "weight": e.weight}
                    for e in edges
                ],
            }, indent=2)

        elif format == "mermaid":
            lines = ["graph TD"]
            for n in nodes:
                label = n.label.replace('"', '\\"')[:30]
                lines.append(f'    {n.id[:8]}["{label}"]')
            for e in edges:
                lines.append(f"    {e.source[:8]} --- {e.target[:8]}")
            return "\n".join(lines)

        elif format == "svg":
            # Simple SVG output
            width, height = 800, 600
            lines = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '  <style>.node { fill: #4f46e5; } .edge { stroke: #94a3b8; }</style>',
            ]

            # Scale coordinates
            x_min = min(n.x for n in nodes)
            x_max = max(n.x for n in nodes)
            y_min = min(n.y for n in nodes)
            y_max = max(n.y for n in nodes)

            def scale_x(x: float) -> float:
                return 50 + (x - x_min) / (x_max - x_min + 1e-10) * (width - 100)

            def scale_y(y: float) -> float:
                return 50 + (y - y_min) / (y_max - y_min + 1e-10) * (height - 100)

            node_coords = {n.id: (scale_x(n.x), scale_y(n.y)) for n in nodes}

            # Draw edges
            for e in edges:
                if e.source in node_coords and e.target in node_coords:
                    x1, y1 = node_coords[e.source]
                    x2, y2 = node_coords[e.target]
                    lines.append(
                        f'  <line class="edge" x1="{x1}" y1="{y1}" '
                        f'x2="{x2}" y2="{y2}" stroke-width="1"/>'
                    )

            # Draw nodes
            for n in nodes:
                x, y = node_coords[n.id]
                r = 5 + n.importance * 5
                lines.append(f'  <circle class="node" cx="{x}" cy="{y}" r="{r}"/>')

            lines.append("</svg>")
            return "\n".join(lines)

        raise ValidationError(f"Unknown format: {format}")
```

---

## 6. Error Types

### File: `spatial_memory/core/errors.py` (additions)

```python
# Add these exceptions

class InsufficientMemoriesError(SpatialMemoryError):
    """Raised when operation requires more memories than available."""

    def __init__(self, required: int, available: int, operation: str) -> None:
        self.required = required
        self.available = available
        self.operation = operation
        super().__init__(
            f"{operation} requires at least {required} memories, "
            f"but only {available} available"
        )


class JourneyError(SpatialMemoryError):
    """Raised when journey path cannot be computed."""
    pass


class WanderError(SpatialMemoryError):
    """Raised when wander cannot continue."""
    pass
```

---

## 7. Repository Implementation

### File: `spatial_memory/adapters/lancedb_repository.py` (additions)

```python
# Add these methods to LanceDBMemoryRepository

def get_vectors_for_clustering(
    self,
    namespace: str | None = None,
    max_memories: int = 10_000,
) -> tuple[list[str], np.ndarray]:
    """Extract memory IDs and vectors efficiently for clustering."""
    return self._db.get_vectors_for_clustering(namespace, max_memories)

def batch_vector_search(
    self,
    query_vectors: list[np.ndarray],
    limit_per_query: int = 3,
    namespace: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Search for memories near multiple query points."""
    return self._db.batch_vector_search(
        query_vectors=query_vectors,
        limit_per_query=limit_per_query,
        namespace=namespace,
    )
```

### File: `spatial_memory/core/database.py` (additions)

```python
# Add these methods to Database class

def get_vectors_for_clustering(
    self,
    namespace: str | None = None,
    max_memories: int = 10_000,
) -> tuple[list[str], np.ndarray]:
    """
    Extract memory IDs and vectors efficiently for clustering.

    Uses Arrow format for efficient column extraction.
    """
    search = self.table.search()

    # Apply namespace filter if provided
    if namespace:
        search = search.where(f"namespace = '{sanitize_string(namespace)}'")

    # Select only needed columns
    search = search.select(["id", "vector"])

    # Limit results
    search = search.limit(max_memories)

    # Execute and convert to Arrow
    results = search.to_arrow()

    ids = results["id"].to_pylist()
    vectors = np.array(results["vector"].to_pylist(), dtype=np.float32)

    return ids, vectors

def batch_vector_search(
    self,
    query_vectors: list[np.ndarray],
    limit_per_query: int = 3,
    namespace: str | None = None,
    parallel: bool = True,
    max_workers: int = 4,
) -> list[list[dict[str, Any]]]:
    """
    Search for memories near multiple query points.

    Executes searches in parallel for efficiency.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def search_single(vector: np.ndarray) -> list[dict[str, Any]]:
        return self.vector_search(
            query_vector=vector,
            limit=limit_per_query,
            namespace=namespace,
        )

    if parallel and len(query_vectors) > 1:
        results = [None] * len(query_vectors)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(search_single, v): i
                for i, v in enumerate(query_vectors)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
    else:
        return [search_single(v) for v in query_vectors]
```

---

## 8. MCP Tool Definitions

### File: `spatial_memory/server.py` (additions to tools list)

```python
# Add these tool definitions

JOURNEY_TOOL = {
    "name": "journey",
    "description": (
        "Navigate semantic space between two memories using spherical "
        "interpolation (SLERP). Discovers memories along the conceptual path."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "start_id": {
                "type": "string",
                "description": "Starting memory UUID",
            },
            "end_id": {
                "type": "string",
                "description": "Ending memory UUID",
            },
            "steps": {
                "type": "integer",
                "minimum": 2,
                "maximum": 20,
                "default": 10,
                "description": "Number of interpolation steps",
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace filter for nearby search",
            },
        },
        "required": ["start_id", "end_id"],
    },
}

WANDER_TOOL = {
    "name": "wander",
    "description": (
        "Explore memory space through random walk. Uses temperature-based "
        "selection to balance exploration and exploitation."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "start_id": {
                "type": "string",
                "description": "Starting memory UUID (random if not provided)",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Number of exploration steps",
            },
            "temperature": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 5.0,
                "default": 0.5,
                "description": "Randomness (0.1=focused, 2.0=very random)",
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace filter",
            },
        },
        "required": [],
    },
}

REGIONS_TOOL = {
    "name": "regions",
    "description": (
        "Discover semantic clusters in memory space using HDBSCAN. "
        "Returns cluster info with representative memories and keywords."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Optional namespace filter",
            },
            "min_cluster_size": {
                "type": "integer",
                "minimum": 2,
                "maximum": 50,
                "default": 3,
                "description": "Minimum memories per cluster",
            },
            "max_clusters": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum clusters to return",
            },
        },
        "required": [],
    },
}

VISUALIZE_TOOL = {
    "name": "visualize",
    "description": (
        "Project memories to 2D/3D for visualization using UMAP. "
        "Returns coordinates and optional similarity edges."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific memory UUIDs to visualize",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace filter (if memory_ids not specified)",
            },
            "format": {
                "type": "string",
                "enum": ["json", "mermaid", "svg"],
                "default": "json",
                "description": "Output format",
            },
            "dimensions": {
                "type": "integer",
                "enum": [2, 3],
                "default": 2,
                "description": "Projection dimensionality",
            },
            "include_edges": {
                "type": "boolean",
                "default": True,
                "description": "Include similarity edges",
            },
        },
        "required": [],
    },
}
```

---

## 9. Testing Strategy

### Test Files Required

| File | Tests | Priority |
|------|-------|----------|
| `tests/unit/test_spatial_ops.py` | SLERP, temperature selection, helpers | P0 |
| `tests/unit/test_spatial_service.py` | Journey, wander, regions, visualize (mocked) | P0 |
| `tests/integration/test_spatial_tools.py` | End-to-end MCP tools | P0 |

### Unit Tests: `tests/unit/test_spatial_ops.py`

```python
"""Tests for core spatial operations."""

import numpy as np
import pytest

from spatial_memory.core.spatial_ops import (
    normalize,
    slerp,
    slerp_path,
    softmax_with_temperature,
    temperature_select,
)


class TestNormalize:
    def test_normalize_unit_vector(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_scales_to_unit(self):
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)


class TestSlerp:
    def test_slerp_t0_returns_start(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.0)
        np.testing.assert_array_almost_equal(result, v0, decimal=5)

    def test_slerp_t1_returns_end(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 1.0)
        np.testing.assert_array_almost_equal(result, v1, decimal=5)

    def test_slerp_midpoint_is_unit(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_handles_parallel_vectors(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([1.0, 0.0001, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_handles_antipodal_vectors(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        # Should be perpendicular to both
        assert abs(np.dot(result, v0)) < 0.1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5


class TestSlerpPath:
    def test_slerp_path_generates_correct_count(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=5)
        assert len(path) == 5

    def test_slerp_path_all_unit_vectors(self):
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=10)
        for v in path:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_slerp_path_invalid_steps(self):
        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError):
            slerp_path(v0, v1, steps=0)


class TestSoftmaxWithTemperature:
    def test_softmax_sums_to_one(self):
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=1.0)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_temperature_zero_is_argmax(self):
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=0.0)
        assert probs[0] == 1.0
        assert probs[1] == 0.0
        assert probs[2] == 0.0

    def test_softmax_high_temperature_is_uniform(self):
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=100.0)
        # Should be nearly uniform
        assert all(0.25 < p < 0.45 for p in probs)

    def test_softmax_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax_with_temperature(np.array([1.0]), temperature=-1.0)


class TestTemperatureSelect:
    def test_temperature_select_returns_item(self):
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7, 0.3])
        result = temperature_select(items, scores, temperature=1.0)
        assert result in items

    def test_temperature_select_greedy(self):
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7, 0.3])
        # With temperature=0, should always pick highest score
        for _ in range(10):
            result = temperature_select(items, scores, temperature=0.01)
            assert result == "a"
```

### Unit Tests: `tests/unit/test_spatial_service.py`

```python
"""Tests for SpatialService with mocked dependencies."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from spatial_memory.core.errors import (
    InsufficientMemoriesError,
    MemoryNotFoundError,
    ValidationError,
)
from spatial_memory.services.spatial import SpatialConfig, SpatialService


@pytest.fixture
def mock_repository():
    """Mock repository with spatial methods."""
    repo = MagicMock()

    # Sample memory data
    def make_memory(id: str, vector: list[float]) -> dict:
        return {
            "id": id,
            "content": f"Memory {id}",
            "vector": vector,
            "namespace": "default",
            "importance": 0.5,
        }

    # Setup get_with_vector to return memories
    def get_with_vector(id: str) -> dict:
        vectors = {
            "start-uuid": [1.0] + [0.0] * 383,
            "end-uuid": [0.0, 1.0] + [0.0] * 382,
        }
        if id not in vectors:
            raise MemoryNotFoundError(id)
        return make_memory(id, vectors[id])

    repo.get_with_vector = MagicMock(side_effect=get_with_vector)
    repo.get = MagicMock(side_effect=get_with_vector)

    # Setup batch_vector_search
    def batch_search(query_vectors, limit_per_query, namespace):
        return [
            [make_memory(f"nearby-{i}-{j}", [0.5] * 384)
             for j in range(limit_per_query)]
            for i in range(len(query_vectors))
        ]

    repo.batch_vector_search = MagicMock(side_effect=batch_search)

    # Setup get_vectors_for_clustering
    repo.get_vectors_for_clustering = MagicMock(return_value=(
        [f"mem-{i}" for i in range(100)],
        np.random.randn(100, 384).astype(np.float32),
    ))

    return repo


@pytest.fixture
def mock_embeddings():
    """Mock embedding service."""
    embeddings = MagicMock()
    embeddings.embed = MagicMock(return_value=np.random.randn(384).astype(np.float32))
    return embeddings


@pytest.fixture
def spatial_service(mock_repository, mock_embeddings):
    """SpatialService with mocked dependencies."""
    return SpatialService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=SpatialConfig(),
    )


class TestJourney:
    def test_journey_returns_result(self, spatial_service):
        result = spatial_service.journey(
            start_id="start-uuid",
            end_id="end-uuid",
            steps=5,
        )
        assert result.start_id == "start-uuid"
        assert result.end_id == "end-uuid"
        assert len(result.steps) == 5

    def test_journey_validates_min_steps(self, spatial_service):
        with pytest.raises(ValidationError, match="at least 2 steps"):
            spatial_service.journey("start-uuid", "end-uuid", steps=1)

    def test_journey_validates_max_steps(self, spatial_service):
        with pytest.raises(ValidationError, match="cannot exceed"):
            spatial_service.journey("start-uuid", "end-uuid", steps=100)

    def test_journey_validates_uuid(self, spatial_service):
        with pytest.raises(ValidationError, match="UUID"):
            spatial_service.journey("invalid", "end-uuid")

    def test_journey_raises_not_found(self, spatial_service, mock_repository):
        mock_repository.get_with_vector.side_effect = MemoryNotFoundError("bad-id")
        with pytest.raises(MemoryNotFoundError):
            spatial_service.journey("bad-id", "end-uuid")


class TestWander:
    def test_wander_returns_result(self, spatial_service, mock_repository):
        # Setup vector_search for wander
        mock_repository.vector_search = MagicMock(return_value=[
            {"id": f"nearby-{i}", "content": f"Memory {i}",
             "vector": [0.5] * 384, "similarity": 0.8 - i*0.1}
            for i in range(10)
        ])

        result = spatial_service.wander(start_id="start-uuid", steps=3)
        assert result.start_id == "start-uuid"
        assert len(result.steps) <= 3

    def test_wander_validates_temperature(self, spatial_service):
        with pytest.raises(ValidationError, match="Temperature"):
            spatial_service.wander(temperature=10.0)

    def test_wander_validates_steps(self, spatial_service):
        with pytest.raises(ValidationError, match="at least 1 step"):
            spatial_service.wander(steps=0)


class TestRegions:
    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.hdbscan")
    def test_regions_returns_clusters(
        self, mock_hdbscan, spatial_service, mock_repository
    ):
        # Mock HDBSCAN
        clusterer = MagicMock()
        clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1] * 20)
        mock_hdbscan.HDBSCAN.return_value = clusterer

        result = spatial_service.regions()
        assert result.total_memories == 100
        assert len(result.clusters) > 0

    @patch("spatial_memory.services.spatial.HDBSCAN_AVAILABLE", False)
    def test_regions_raises_when_hdbscan_unavailable(self, spatial_service):
        from spatial_memory.core.errors import ClusteringError
        with pytest.raises(ClusteringError, match="HDBSCAN not installed"):
            spatial_service.regions()

    def test_regions_raises_insufficient_memories(
        self, spatial_service, mock_repository
    ):
        mock_repository.get_vectors_for_clustering.return_value = (
            ["mem-1"],
            np.random.randn(1, 384).astype(np.float32),
        )
        with pytest.raises(InsufficientMemoriesError):
            spatial_service.regions()


class TestVisualize:
    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", True)
    @patch("spatial_memory.services.spatial.umap")
    def test_visualize_returns_result(
        self, mock_umap, spatial_service, mock_repository
    ):
        # Mock UMAP
        reducer = MagicMock()
        reducer.fit_transform.return_value = np.random.randn(100, 2).astype(np.float32)
        mock_umap.UMAP.return_value = reducer

        result = spatial_service.visualize()
        assert len(result.nodes) == 100
        assert result.format == "json"

    @patch("spatial_memory.services.spatial.UMAP_AVAILABLE", False)
    def test_visualize_raises_when_umap_unavailable(self, spatial_service):
        from spatial_memory.core.errors import VisualizationError
        with pytest.raises(VisualizationError, match="UMAP not installed"):
            spatial_service.visualize()

    def test_visualize_validates_dimensions(self, spatial_service):
        with pytest.raises(ValidationError, match="must be 2 or 3"):
            spatial_service.visualize(dimensions=4)
```

---

## 10. Configuration Additions

### File: `spatial_memory/config.py` (additions to Settings)

```python
# Add these fields to Settings class

# Spatial Operations - Journey
max_journey_steps: int = Field(
    default=20,
    ge=2,
    le=50,
    description="Maximum steps in a journey",
)
journey_neighbors_per_step: int = Field(
    default=3,
    ge=1,
    le=10,
    description="Nearby memories to find per journey step",
)

# Spatial Operations - Wander
max_wander_steps: int = Field(
    default=20,
    ge=1,
    le=50,
    description="Maximum steps in a wander",
)
wander_default_temperature: float = Field(
    default=0.5,
    ge=0.1,
    le=5.0,
    description="Default temperature for wander randomness",
)

# Spatial Operations - Regions
regions_max_memories: int = Field(
    default=10_000,
    ge=100,
    description="Maximum memories for clustering",
)

# Spatial Operations - Visualize
visualize_similarity_threshold: float = Field(
    default=0.7,
    ge=0.0,
    le=1.0,
    description="Minimum similarity for visualization edges",
)
```

---

## 11. Dependencies

### Verify in `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing ...
    "hdbscan>=0.8.33",
    "umap-learn>=0.5.5",
    "scikit-learn>=1.0.0",  # For silhouette score
]
```

---

## 12. Implementation Order

1. **Add error types** to `spatial_memory/core/errors.py`
2. **Add domain models** to `spatial_memory/core/models.py`
3. **Create `spatial_memory/core/spatial_ops.py`** with core algorithms
4. **Extend repository protocol** in `spatial_memory/ports/repositories.py`
5. **Add database methods** to `spatial_memory/core/database.py`
6. **Implement repository methods** in `spatial_memory/adapters/lancedb_repository.py`
7. **Create `spatial_memory/services/spatial.py`** (TDD: write tests first)
8. **Add config options** to `spatial_memory/config.py`
9. **Register tools** in `spatial_memory/server.py`
10. **Run all tests** and verify coverage

---

## 13. Verification Commands

```bash
# 1. Type checking
mypy spatial_memory --strict

# 2. Linting
ruff check spatial_memory tests

# 3. Unit tests
pytest tests/unit/test_spatial_ops.py tests/unit/test_spatial_service.py -v

# 4. Integration tests
pytest tests/integration/test_spatial_tools.py -v

# 5. All tests
pytest tests/ -v --cov=spatial_memory --cov-report=term-missing

# 6. Verify tool count
python -c "from spatial_memory.server import SpatialMemoryServer; print(len(SpatialMemoryServer()._tools))"
# Expected: 11 (7 existing + 4 new)
```

---

## 14. Success Criteria

### Functional
- [ ] journey() finds path between two memories using SLERP
- [ ] wander() explores with temperature-based selection
- [ ] regions() discovers clusters using HDBSCAN
- [ ] visualize() projects to 2D/3D using UMAP
- [ ] All 4 tools registered in MCP server

### Quality
- [ ] 50+ new tests passing
- [ ] Total test count ~450
- [ ] mypy strict: 0 errors
- [ ] ruff: 0 issues

### Performance
- [ ] Journey: <500ms for 20 steps
- [ ] Regions: <5s for 10K memories
- [ ] Visualize: <3s for 500 memories

---

## 15. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| HDBSCAN/UMAP not installed | Runtime check with clear error message |
| Large dataset clustering slow | Add `max_memories` limit with config |
| SLERP edge cases | Comprehensive tests for parallel/antipodal vectors |
| Temperature selection bias | Test randomness distribution |
| Memory pressure | Chunk large operations, use float32 |
