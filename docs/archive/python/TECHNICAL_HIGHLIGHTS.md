# Technical Highlights

A deep dive into the key algorithms and design decisions that differentiate Spatial Memory MCP from traditional vector databases.

## Table of Contents

- [Cognitive Memory Model vs Traditional Storage](#cognitive-memory-model-vs-traditional-storage)
- [SLERP: Spherical Linear Interpolation](#slerp-spherical-linear-interpolation)
- [Temperature-Based Random Walks](#temperature-based-random-walks)
- [HDBSCAN Clustering](#hdbscan-clustering)
- [UMAP Projection](#umap-projection)
- [ONNX Runtime Optimization](#onnx-runtime-optimization)
- [scipy Integration](#scipy-integration)
- [References](#references)

---

## Cognitive Memory Model vs Traditional Storage

Traditional vector databases focus on a single question: *"How do we store things so we can find them later?"*

Spatial Memory MCP asks a different question: *"How do we give AI systems memory that works like memory—with forgetting, reinforcement, association, and discovery?"*

### The Difference in Practice

| Aspect | Traditional Vector DB | Spatial Memory MCP |
|--------|----------------------|-------------------|
| Storage model | Static embeddings | Dynamic, evolving memories |
| Retrieval | Query → Results | Query → Results + Exploration |
| Time dimension | None | Decay, reinforcement, access tracking |
| Organization | Manual tagging | Self-organizing clusters |
| Discovery | Only what you search for | Serendipitous connections |

### How It Works at Runtime

When the MCP server runs, it implements a cognitive memory model inspired by established memory research:

1. **Memory Decay** (Ebbinghaus Forgetting Curve): Memories fade over time if not accessed. The system supports exponential, linear, and step decay functions with configurable half-life periods.

2. **Reinforcement Learning**: Every access, retrieval, or reference boosts importance scores. Frequently needed knowledge rises to the surface.

3. **Consolidation**: Similar memories merge intelligently using configurable strategies (keep_newest, keep_oldest, keep_highest_importance, merge_content).

4. **Auto-Extraction**: Pattern matching identifies facts, decisions, and key information from conversation transcripts.

This model is inspired by:
- Ebbinghaus's forgetting curve research (1885)
- Duolingo's half-life regression algorithm
- The FSRS (Free Spaced Repetition Scheduler) algorithm

### Memory Lifecycle Diagram

```mermaid
flowchart TB
    subgraph Creation["Memory Creation"]
        A[New Information] --> B[Generate Embedding]
        B --> C[Store with Importance]
    end

    subgraph Evolution["Memory Evolution"]
        C --> D{Time Passes}
        D -->|Not Accessed| E[Decay]
        D -->|Accessed| F[Reinforce]
        E --> G[Lower Importance]
        F --> H[Higher Importance]
    end

    subgraph Maintenance["Memory Maintenance"]
        G --> I{Below Threshold?}
        H --> J{Similar Memories?}
        I -->|Yes| K[Fade Away]
        I -->|No| L[Persist]
        J -->|Yes| M[Consolidate]
        J -->|No| L
        M --> L
    end

    subgraph Discovery["Spatial Discovery"]
        L --> N[Recall]
        L --> O[Journey]
        L --> P[Wander]
        L --> Q[Regions]
    end

    style Creation fill:#e1f5fe
    style Evolution fill:#fff3e0
    style Maintenance fill:#f3e5f5
    style Discovery fill:#e8f5e9
```

---

## SLERP: Spherical Linear Interpolation

### What It Is

SLERP (Spherical Linear Interpolation) is a method for interpolating between two points on a sphere while maintaining constant angular velocity. In the context of embeddings, it creates a smooth path through semantic space.

### Why It Matters

Embedding vectors are typically normalized to unit length, meaning they all lie on a hypersphere. When navigating between two concepts:

- **Linear interpolation** cuts through the sphere, producing vectors of varying length
- **SLERP** follows the geodesic (great circle) path, maintaining unit length throughout

This makes SLERP geometrically correct for exploring the semantic space between two memories.

### Implementation

The SLERP formula:

```
slerp(v0, v1, t) = sin((1-t)θ) / sin(θ) * v0 + sin(tθ) / sin(θ) * v1
```

Where `θ` is the angle between vectors and `t` is the interpolation parameter (0.0 to 1.0).

From `spatial_memory/core/spatial_ops.py`:

```python
def slerp(v0: Vector, v1: Vector, t: float) -> Vector:
    """
    Spherical linear interpolation between two unit vectors.

    SLERP produces a constant-speed path along the great circle connecting two
    points on the unit sphere. This is more geometrically correct than linear
    interpolation for normalized embedding vectors.
    """
    # Normalize and compute angle
    v0 = normalize(v0.astype(np.float64))
    v1 = normalize(v1.astype(np.float64))
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)

    # Handle edge cases (parallel/antipodal vectors)
    if dot > 0.9995:  # Nearly parallel - use linear interpolation
        result = v0 + t * (v1 - v0)
        return normalize(result.astype(np.float32))

    # Standard SLERP
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega
    return (s0 * v0 + s1 * v1).astype(np.float32)
```

### Use Case: The `journey` Tool

The `journey` tool uses SLERP to navigate between two memories:

1. Generate interpolation points along the SLERP path
2. Find actual memories closest to each interpolation point
3. Return the discovered conceptual path

Example: Journey from "machine learning basics" to "production deployment" might reveal: feature engineering → model validation → containerization → monitoring.

### SLERP vs Linear Interpolation

```mermaid
graph LR
    subgraph Linear["Linear Interpolation (Wrong)"]
        A1((Start)) -.->|"cuts through sphere"| B1((End))
    end

    subgraph SLERP["SLERP (Correct)"]
        A2((Start)) -->|"t=0.25"| M1(( ))
        M1 -->|"t=0.5"| M2(( ))
        M2 -->|"t=0.75"| M3(( ))
        M3 --> B2((End))
    end

    style A1 fill:#ef5350
    style B1 fill:#ef5350
    style A2 fill:#66bb6a
    style B2 fill:#66bb6a
    style M1 fill:#81c784
    style M2 fill:#81c784
    style M3 fill:#81c784
```

### Journey Tool Flow

```mermaid
flowchart LR
    A[Start Memory] --> B[Get Vector v0]
    C[End Memory] --> D[Get Vector v1]
    B --> E[SLERP Path Generation]
    D --> E
    E --> F["Interpolation Points<br/>t=0, 0.25, 0.5, 0.75, 1.0"]
    F --> G[For Each Point]
    G --> H[Find Nearest Memories]
    H --> I[Build Journey Steps]
    I --> J[Return Path with<br/>Discovered Memories]

    style A fill:#bbdefb
    style C fill:#c8e6c9
    style J fill:#fff9c4
```

---

## Temperature-Based Random Walks

### What It Is

Temperature-based selection is a technique from reinforcement learning that balances exploration vs exploitation. The `wander` tool uses it for serendipitous discovery through memory space.

### The Temperature Parameter

Temperature controls the randomness of selection:

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Greedy: Always pick the most similar memory |
| 0.5 | Balanced: Mix of focused and exploratory |
| 1.0 | Highly random: Nearly uniform selection |

### Implementation

The softmax function with temperature scaling:

```python
def softmax_with_temperature(scores: NDArray, temperature: float) -> NDArray:
    """
    Temperature controls the randomness of the resulting distribution:
    - T -> 0: Deterministic (all probability mass on highest score)
    - T = 1: Standard softmax
    - T -> inf: Uniform random selection
    """
    if temperature < 1e-10:  # Greedy selection
        result = np.zeros_like(scores)
        result[np.argmax(scores)] = 1.0
        return result

    # Scale by temperature and apply softmax
    scaled = scores / temperature
    scaled_shifted = scaled - np.max(scaled)  # Numerical stability
    exp_scores = np.exp(scaled_shifted)
    return exp_scores / np.sum(exp_scores)
```

### Use Case: The `wander` Tool

Each step of the random walk:

1. Find candidate neighbors from current position
2. Convert similarity scores to selection probabilities using temperature
3. Sample from the probability distribution
4. Move to selected memory, repeat

Lower temperature creates focused, thematic walks. Higher temperature enables unexpected conceptual jumps.

### Wander Algorithm Flow

```mermaid
flowchart TB
    A[Start at Memory] --> B[Find Neighbor Candidates]
    B --> C[Get Similarity Scores]
    C --> D[Apply Temperature Scaling]
    D --> E[Softmax → Probabilities]
    E --> F[Sample Next Memory]
    F --> G{More Steps?}
    G -->|Yes| H[Move to Selected Memory]
    H --> B
    G -->|No| I[Return Walk Path]

    subgraph Temperature Effect
        T1["T=0.0: Greedy<br/>Always highest similarity"]
        T2["T=0.5: Balanced<br/>Weighted random"]
        T3["T=1.0: Exploratory<br/>Nearly uniform"]
    end

    style A fill:#bbdefb
    style I fill:#fff9c4
    style T1 fill:#ffcdd2
    style T2 fill:#fff9c4
    style T3 fill:#c8e6c9
```

### Temperature Effect on Selection

```mermaid
graph TB
    subgraph "Low Temperature (T=0.1)"
        L1[Memory A: 0.9] -->|"P=0.95"| L2((Selected))
        L3[Memory B: 0.7] -.->|"P=0.04"| L2
        L4[Memory C: 0.5] -.->|"P=0.01"| L2
    end

    subgraph "High Temperature (T=1.0)"
        H1[Memory A: 0.9] -->|"P=0.45"| H2((Maybe))
        H3[Memory B: 0.7] -->|"P=0.33"| H2
        H4[Memory C: 0.5] -->|"P=0.22"| H2
    end

    style L2 fill:#66bb6a
    style H2 fill:#fff176
```

---

## HDBSCAN Clustering

### What It Is

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that:

- Automatically determines the number of clusters
- Identifies outliers (noise points)
- Works with varying density clusters
- Requires minimal parameter tuning

### Why HDBSCAN Over K-Means

| Feature | K-Means | HDBSCAN |
|---------|---------|---------|
| Cluster count | Must specify | Auto-determined |
| Cluster shape | Spherical only | Arbitrary shapes |
| Outlier handling | Forces all points into clusters | Identifies noise |
| Density variance | Assumes uniform | Handles varying density |

### Implementation

From `spatial_memory/services/spatial.py`:

```python
def regions(self, namespace: str | None = None,
            min_cluster_size: int | None = None) -> RegionsResult:
    """Discover memory regions using HDBSCAN clustering."""

    # Fetch all vectors
    all_memories = self._repo.get_all(namespace=namespace, limit=10_000)
    vectors = np.array([v for _, v in all_memories], dtype=np.float32)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=actual_min_size,
        metric="euclidean",  # Works with normalized vectors
        cluster_selection_method="eom",  # Excess of Mass for varied sizes
    )
    labels = clusterer.fit_predict(vectors)
```

### Configuration

The system auto-configures HDBSCAN based on dataset size:

```python
def configure_hdbscan(n_samples: int, min_cluster_size: int | None = None):
    if min_cluster_size is None:
        # Adaptive: sqrt(n)/2, clamped to [3, 50]
        min_cluster_size = max(3, int(np.sqrt(n_samples) / 2))
        min_cluster_size = min(min_cluster_size, 50)
```

### Use Case: The `regions` Tool

Discovers natural topic clusters in your knowledge base:
- Returns cluster centroids with representative memories
- Extracts keywords describing each cluster
- Identifies orphan memories (noise points)
- Reports clustering quality (silhouette score)

### HDBSCAN Clustering Process

```mermaid
flowchart TB
    A[Load All Memory Vectors] --> B[Build Density Graph]
    B --> C[Construct Hierarchy]
    C --> D[Extract Flat Clusters]
    D --> E{For Each Cluster}

    E --> F[Compute Centroid]
    F --> G[Find Representative Memory]
    G --> H[Extract Keywords]
    H --> I[Calculate Coherence]

    E --> J[Identify Noise Points]

    I --> K[Sort by Size]
    J --> K
    K --> L[Return RegionsResult]

    subgraph Output
        L --> M[Clusters with Keywords]
        L --> N[Noise Count]
        L --> O[Quality Score]
    end

    style A fill:#bbdefb
    style L fill:#fff9c4
    style M fill:#c8e6c9
    style N fill:#ffcdd2
    style O fill:#e1bee7
```

### K-Means vs HDBSCAN

```mermaid
graph TB
    subgraph "K-Means (K=3)"
        K1((●)) --- K2((●))
        K2 --- K3((●))
        K4[Forced into cluster]
        K5[Must specify K]
    end

    subgraph "HDBSCAN"
        H1((●)) --- H2((●))
        H3((●)) --- H4((●))
        H5[○ Noise point]
        H6[Auto-detected clusters]
    end

    style K4 fill:#ffcdd2
    style K5 fill:#ffcdd2
    style H5 fill:#e0e0e0
    style H6 fill:#c8e6c9
```

---

## UMAP Projection

### What It Is

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that projects high-dimensional embeddings into 2D or 3D for visualization while preserving both local and global structure.

### Why UMAP Over t-SNE

| Feature | t-SNE | UMAP |
|---------|-------|------|
| Speed | Slower | Faster |
| Global structure | Poor | Better preserved |
| Scalability | Limited | Handles larger datasets |
| Reproducibility | Random each run | Deterministic with seed |

### Implementation

From `spatial_memory/services/spatial.py`:

```python
def visualize(self, memory_ids: list[str] | None = None,
              dimensions: Literal[2, 3] = 2) -> VisualizationResult:
    """Generate visualization using UMAP projection."""

    # Configure UMAP
    reducer = umap.UMAP(
        n_components=dimensions,
        n_neighbors=min(15, len(vectors) - 1),
        min_dist=0.1,
        metric="cosine",  # Natural for embeddings
        random_state=42,  # Reproducibility
    )
    embedding = reducer.fit_transform(vectors)
```

### Configuration Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_neighbors` | 15 | Larger values = more global structure |
| `min_dist` | 0.1 | Smaller values = tighter clusters |
| `metric` | cosine | Natural for embedding similarity |

### Use Case: The `visualize` Tool

Generates visualizations in multiple formats:
- **JSON**: Node/edge data for custom rendering
- **Mermaid**: Graph diagrams for documentation
- **SVG**: Standalone visual with color-coded clusters

### UMAP Visualization Pipeline

```mermaid
flowchart LR
    subgraph Input
        A["High-Dimensional<br/>Embeddings<br/>(384 dimensions)"]
    end

    subgraph UMAP["UMAP Processing"]
        B[Build k-NN Graph]
        C[Optimize Low-Dim Layout]
        D[Preserve Local Structure]
    end

    subgraph Output
        E["2D/3D Coordinates"]
        F[Similarity Edges]
        G[Cluster Colors]
    end

    A --> B --> C --> D --> E
    D --> F
    D --> G

    subgraph Formats
        E --> H[JSON]
        E --> I[Mermaid]
        E --> J[SVG]
    end

    style A fill:#bbdefb
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
```

### Dimensionality Reduction Concept

```mermaid
graph LR
    subgraph "384-D Space (Embeddings)"
        A1[●]
        A2[●]
        A3[●]
        A4[●]
        A5[●]
    end

    subgraph "2-D Space (Visualization)"
        B1((●))
        B2((●))
        B3((●))
        B4((●))
        B5((●))
    end

    A1 -.->|UMAP| B1
    A2 -.->|preserves| B2
    A3 -.->|neighbors| B3
    A4 -.->|"& structure"| B4
    A5 -.->|" "| B5

    style B1 fill:#ef5350
    style B2 fill:#ef5350
    style B3 fill:#66bb6a
    style B4 fill:#66bb6a
    style B5 fill:#42a5f5
```

---

## ONNX Runtime Optimization

### What It Is

ONNX (Open Neural Network Exchange) Runtime is an inference engine that provides optimized execution of neural networks. Spatial Memory MCP uses it for embedding generation.

### Performance Benefits

| Metric | PyTorch | ONNX Runtime |
|--------|---------|--------------|
| Inference speed | Baseline | 2-3x faster |
| Memory usage | Higher | ~60% less |
| GPU requirement | Optional | CPU-optimized |

### Implementation

From `spatial_memory/core/embeddings.py`:

```python
def _detect_backend(requested: EmbeddingBackend) -> Literal["onnx", "pytorch"]:
    """Detect which backend to use."""
    if requested == "auto":
        if _is_onnx_available():
            return "onnx"
        return "pytorch"
    # ...

def _load_local_model(self) -> None:
    """Load model with appropriate backend."""
    self._active_backend = _detect_backend(self._requested_backend)

    if self._active_backend == "onnx":
        self._model = SentenceTransformer(
            self.model_name,
            backend="onnx",
        )
        logger.info("Using ONNX Runtime backend (2-3x faster inference)")
```

### Automatic Detection

The system automatically detects ONNX availability:

```python
def _is_onnx_available() -> bool:
    """Check if ONNX Runtime and Optimum are available."""
    try:
        import onnxruntime
        import optimum.onnxruntime
        return True
    except ImportError:
        return False
```

### Installation

To enable ONNX Runtime:

```bash
pip install sentence-transformers[onnx]
```

### Backend Selection Flow

```mermaid
flowchart TB
    A[Initialize EmbeddingService] --> B{Backend Setting?}

    B -->|"auto"| C{ONNX Available?}
    B -->|"onnx"| D{ONNX Available?}
    B -->|"pytorch"| E[Use PyTorch]

    C -->|Yes| F[Use ONNX Runtime]
    C -->|No| E

    D -->|Yes| F
    D -->|No| G[Raise ConfigurationError]

    F --> H["2-3x Faster Inference<br/>60% Less Memory"]
    E --> I["Standard Inference<br/>Full Compatibility"]

    style F fill:#c8e6c9
    style H fill:#c8e6c9
    style E fill:#fff9c4
    style I fill:#fff9c4
    style G fill:#ffcdd2
```

### Embedding Generation Pipeline

```mermaid
flowchart LR
    A[Text Input] --> B{Cached?}
    B -->|Yes| C[Return Cached]
    B -->|No| D[Generate Embedding]
    D --> E{Backend}
    E -->|ONNX| F[ONNX Runtime]
    E -->|PyTorch| G[PyTorch]
    F --> H[Normalize Vector]
    G --> H
    H --> I[Cache Result]
    I --> J[Return Embedding]
    C --> J

    style A fill:#bbdefb
    style J fill:#c8e6c9
    style F fill:#e8f5e9
    style G fill:#fff3e0
```

---

## scipy Integration

### What It Is

scipy is a fundamental library for scientific computing in Python. Spatial Memory MCP uses it specifically for efficient similarity calculations.

### Why scipy

The primary use is `scipy.spatial.distance.cdist` for computing pairwise distances:

```python
from scipy.spatial.distance import cdist

# Vectorized pairwise cosine similarity
distances = cdist(normalized_vectors, normalized_vectors, metric="cosine")
similarities = 1.0 - distances
```

### Performance Comparison

| Method | Operation | Performance |
|--------|-----------|-------------|
| Naive loops | N² iterations | Slow |
| NumPy dot product | Matrix multiplication | Good |
| scipy.cdist | Optimized C implementation | Best |

### Implementation

From `spatial_memory/services/spatial.py`:

```python
def _compute_pairwise_similarities(self, vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities using vectorized operations."""

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms < 1e-10, 1.0, norms)

    if SCIPY_AVAILABLE:
        # scipy.cdist with cosine metric (returns distances)
        distances = cdist(normalized, normalized, metric="cosine")
        similarities = 1.0 - distances
    else:
        # Fallback: numpy dot product
        similarities = normalized @ normalized.T

    return similarities
```

### Graceful Degradation

scipy is an optional dependency. If not available, the system falls back to NumPy matrix operations:

```python
try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("scipy not available - using fallback for similarity calculations")
```

### Where scipy Is Used

1. **Visualization edges**: Computing similarity edges between all node pairs
2. **Clustering quality**: Silhouette score calculation (via sklearn, which uses scipy)
3. **Batch similarity**: Efficient pairwise comparisons in consolidation

### Pairwise Similarity Computation

```mermaid
flowchart TB
    A["N Vectors<br/>(N × 384)"] --> B[Normalize to Unit Length]
    B --> C{scipy Available?}

    C -->|Yes| D["scipy.cdist<br/>(Optimized C)"]
    C -->|No| E["NumPy @ operator<br/>(Matrix Multiply)"]

    D --> F["Distance Matrix<br/>(N × N)"]
    E --> G["Similarity Matrix<br/>(N × N)"]

    F -->|"1 - distance"| H[Similarity Matrix]
    G --> H

    H --> I[Filter by Threshold]
    I --> J[Create Edges]

    style D fill:#c8e6c9
    style E fill:#fff9c4
    style H fill:#bbdefb
```

### Performance Scaling

```mermaid
graph LR
    subgraph "Naive Loops O(N²)"
        N1["100 vectors: 10K ops"]
        N2["1000 vectors: 1M ops"]
        N3["10K vectors: 100M ops"]
    end

    subgraph "scipy.cdist O(N²) but Optimized"
        S1["100 vectors: Fast"]
        S2["1000 vectors: Fast"]
        S3["10K vectors: Manageable"]
    end

    N1 -.->|"~100x slower"| S1
    N2 -.->|"~100x slower"| S2
    N3 -.->|"~100x slower"| S3

    style N1 fill:#ffcdd2
    style N2 fill:#ffcdd2
    style N3 fill:#ffcdd2
    style S1 fill:#c8e6c9
    style S2 fill:#c8e6c9
    style S3 fill:#fff9c4
```

---

## References

### Memory Research

1. **Ebbinghaus, H. (1885)**. *Memory: A Contribution to Experimental Psychology*. The foundational research on the forgetting curve. [Read](https://psychclassics.yorku.ca/Ebbinghaus/index.htm)

2. **Settles, B. & Meeder, B. (2016)**. *A Trainable Spaced Repetition Model for Language Learning*. Duolingo's half-life regression algorithm. [ACL Anthology](https://aclanthology.org/P16-1174/)

3. **FSRS Algorithm**. Free Spaced Repetition Scheduler. [GitHub](https://github.com/open-spaced-repetition/fsrs4anki)

### Algorithm References

4. **SLERP**: Shoemake, K. (1985). *Animating rotation with quaternion curves*. SIGGRAPH '85.

5. **HDBSCAN**: Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. [Paper](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)

6. **UMAP**: McInnes, L., Healy, J., Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. [arXiv](https://arxiv.org/abs/1802.03426)

### Libraries

- [ONNX Runtime](https://onnxruntime.ai/)
- [sentence-transformers](https://www.sbert.net/)
- [LanceDB](https://lancedb.github.io/lancedb/)
- [hdbscan](https://hdbscan.readthedocs.io/)
- [umap-learn](https://umap-learn.readthedocs.io/)
- [scipy](https://scipy.org/)
