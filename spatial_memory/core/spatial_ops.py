"""Core spatial algorithms for memory navigation and exploration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

T = TypeVar("T")
Vector = NDArray[np.float32]


# =============================================================================
# Vector Operations
# =============================================================================


def normalize(v: Vector) -> Vector:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector to normalize.

    Returns:
        Unit vector in same direction as v, or zero vector if input norm is negligible.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def normalize_batch(vectors: Vector, copy: bool = True) -> Vector:
    """
    Normalize multiple vectors efficiently.

    Args:
        vectors: 2D array of shape (n_vectors, n_dimensions).
        copy: If True, creates a copy before modifying. If False, modifies in place.

    Returns:
        Array of unit vectors with same shape as input.
    """
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

    SLERP produces a constant-speed path along the great circle connecting two
    points on the unit sphere. This is more geometrically correct than linear
    interpolation for normalized embedding vectors.

    Handles edge cases:
    - Parallel vectors (omega ~ 0): Falls back to linear interpolation
    - Antipodal vectors (omega ~ pi): Chooses arbitrary perpendicular path

    Args:
        v0: Starting unit vector.
        v1: Ending unit vector.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit vector at parameter t.

    Example:
        >>> v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        >>> mid = slerp(v0, v1, 0.5)
        >>> np.linalg.norm(mid)  # Always unit length
        1.0
    """
    # Work in float64 for numerical stability
    v0 = normalize(v0.astype(np.float64))
    v1 = normalize(v1.astype(np.float64))

    # Compute dot product, clamp to [-1, 1] for numerical stability
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)

    # Handle nearly parallel vectors (dot ~ 1.0)
    # Linear interpolation is a good approximation when angle is very small
    if dot > 0.9995:
        result = v0 + t * (v1 - v0)
        return normalize(result.astype(np.float32))

    # Handle nearly antipodal vectors (dot ~ -1.0)
    # Choose an arbitrary perpendicular path
    if dot < -0.9995:
        perp = _find_perpendicular(v0)
        half_angle = np.pi * t
        result = v0 * np.cos(half_angle) + perp * np.sin(half_angle)
        return result.astype(np.float32)

    # Standard SLERP formula
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    return (s0 * v0 + s1 * v1).astype(np.float32)


def _find_perpendicular(v: Vector) -> Vector:
    """
    Find a unit vector perpendicular to v.

    Uses the approach of creating a vector from the standard basis that differs
    most from v, then applying Gram-Schmidt orthogonalization.

    Args:
        v: Input unit vector.

    Returns:
        A unit vector orthogonal to v.
    """
    # Find the component with smallest absolute value
    min_idx = np.argmin(np.abs(v))

    # Create a basis vector that differs most from v
    basis = np.zeros_like(v)
    basis[min_idx] = 1.0

    # Gram-Schmidt: subtract projection of basis onto v
    perp = basis - np.dot(v, basis) * v
    return normalize(perp)


def slerp_path(
    v0: Vector,
    v1: Vector,
    steps: int,
    include_endpoints: bool = True,
) -> list[Vector]:
    """
    Generate N interpolation steps between two vectors using SLERP.

    Creates a path of evenly-spaced points along the great circle connecting
    two embedding vectors. Useful for exploring the semantic space between
    two memories.

    Args:
        v0: Starting vector.
        v1: Ending vector.
        steps: Number of vectors to generate.
        include_endpoints: If True, path starts at v0 and ends at v1.
            If False, generates intermediate points only.

    Returns:
        List of interpolated unit vectors.

    Raises:
        ValueError: If steps < 1.

    Example:
        >>> v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        >>> path = slerp_path(v0, v1, steps=5)
        >>> len(path)
        5
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    vectors: list[Vector] = []

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
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """
    Compute softmax probabilities with temperature scaling.

    Temperature controls the randomness of the resulting distribution:
    - T -> 0: Deterministic (all probability mass on highest score)
    - T = 1: Standard softmax
    - T -> inf: Uniform random selection

    Uses numerically stable computation by shifting scores before exponentiation.

    Args:
        scores: Array of raw scores (higher = better).
        temperature: Temperature parameter (must be >= 0).

    Returns:
        Probability distribution over scores (sums to 1).

    Raises:
        ValueError: If temperature is negative.

    Example:
        >>> scores = np.array([0.9, 0.7, 0.3])
        >>> probs = softmax_with_temperature(scores, temperature=1.0)
        >>> np.sum(probs)  # Always sums to 1
        1.0
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative")

    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        return np.array([], dtype=np.float64)

    if len(scores) == 1:
        return np.array([1.0], dtype=np.float64)

    # Handle temperature = 0 (greedy/deterministic selection)
    if temperature < 1e-10:
        result = np.zeros_like(scores)
        result[np.argmax(scores)] = 1.0
        return result

    # Scale scores by temperature
    scaled = scores / temperature

    # Subtract max for numerical stability (prevents overflow in exp)
    scaled_shifted = scaled - np.max(scaled)
    exp_scores = np.exp(scaled_shifted)

    return exp_scores / np.sum(exp_scores)


def temperature_select(
    items: Sequence[T],
    scores: NDArray[np.float64],
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> T:
    """
    Select an item using temperature-scaled softmax probabilities.

    Combines softmax_with_temperature with random selection. Lower temperatures
    favor higher-scored items, while higher temperatures approach uniform random.

    Args:
        items: Sequence of items to choose from.
        scores: Score for each item (higher = more likely to be selected).
        temperature: Controls randomness (0.1=focused, 2.0=random).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Selected item from the sequence.

    Raises:
        ValueError: If items and scores have different lengths.

    Example:
        >>> items = ["a", "b", "c"]
        >>> scores = np.array([0.9, 0.7, 0.3])
        >>> # Low temperature: almost always picks "a"
        >>> temperature_select(items, scores, temperature=0.1)
        'a'
    """
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
    """
    Information about a discovered cluster.

    Represents a semantic region in the memory space discovered by HDBSCAN
    clustering. Contains metadata about the cluster including its size,
    central tendency, and sample members.

    Attributes:
        cluster_id: Unique identifier for this cluster (-1 indicates noise).
        size: Number of memories in this cluster.
        centroid: Mean vector of all memories in the cluster (normalized).
        centroid_memory_id: ID of the memory closest to the centroid.
        sample_memory_ids: IDs of representative sample memories.
        coherence: Average pairwise similarity within the cluster (0-1).
        keywords: Extracted topic keywords for this cluster.
    """

    cluster_id: int
    size: int
    centroid: Vector
    centroid_memory_id: str
    sample_memory_ids: list[str]
    coherence: float
    keywords: list[str] = field(default_factory=list)


def configure_hdbscan(
    n_samples: int,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> dict:
    """
    Configure HDBSCAN parameters based on dataset characteristics.

    Provides sensible defaults for HDBSCAN clustering on embedding vectors.
    The min_cluster_size is computed adaptively based on dataset size if not
    provided explicitly.

    Args:
        n_samples: Number of samples in the dataset.
        min_cluster_size: Minimum number of points to form a cluster.
            If None, computed as sqrt(n_samples)/2, clamped to [3, 50].
        min_samples: Minimum samples in neighborhood for core point.
            If None, set to min_cluster_size // 2.

    Returns:
        Dictionary of HDBSCAN parameters ready to use with hdbscan.HDBSCAN().

    Example:
        >>> params = configure_hdbscan(1000)
        >>> params["min_cluster_size"]
        15
        >>> import hdbscan  # doctest: +SKIP
        >>> clusterer = hdbscan.HDBSCAN(**params)  # doctest: +SKIP
    """
    if min_cluster_size is None:
        # Adaptive min_cluster_size based on dataset size
        min_cluster_size = max(3, int(np.sqrt(n_samples) / 2))
        min_cluster_size = min(min_cluster_size, 50)

    if min_samples is None:
        min_samples = max(2, min_cluster_size // 2)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": "euclidean",  # Use with normalized vectors for cosine distance
        "cluster_selection_method": "eom",  # Excess of Mass for varied cluster sizes
        "core_dist_n_jobs": -1,  # Use all available cores
    }


# =============================================================================
# UMAP Projection (for Visualize)
# =============================================================================


def configure_umap(
    n_samples: int,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Configure UMAP parameters for memory visualization.

    Provides sensible defaults for projecting high-dimensional embedding
    vectors to 2D or 3D for visualization. Parameters are adjusted based
    on the number of samples.

    Args:
        n_samples: Number of samples to project.
        n_components: Target dimensionality (2 for 2D, 3 for 3D visualization).
        n_neighbors: Size of local neighborhood for manifold approximation.
            Larger values capture more global structure.
        min_dist: Minimum distance between points in embedded space.
            Smaller values create tighter clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of UMAP parameters ready to use with umap.UMAP().

    Example:
        >>> params = configure_umap(500, n_components=2)
        >>> params["n_neighbors"]
        15
        >>> import umap  # doctest: +SKIP
        >>> reducer = umap.UMAP(**params)  # doctest: +SKIP
    """
    return {
        "n_components": n_components,
        # n_neighbors cannot exceed n_samples - 1
        "n_neighbors": min(n_neighbors, n_samples - 1),
        "min_dist": min_dist,
        "metric": "cosine",  # Natural metric for embeddings
        "random_state": random_state,
        # Enable low memory mode for large datasets
        "low_memory": n_samples > 5000,
    }
