# Phase 4 Implementation Plan: Spatial Memory MCP Server

## Executive Summary

Based on comprehensive analysis by 4 specialized agents (Tech Lead, LanceDB Expert, Codebase Explorer, Code Reviewer), this plan addresses **71 identified gaps** across the codebase to transform the spatial-memory-mcp server into a production-ready enterprise system.

**Current State:** Phase 1-3 complete with 285 tests passing, 7 of 19 tools implemented (37%)
**Phase 4 Goal:** Complete spatial operations, add resilience patterns, fix technical debt, achieve production readiness

---

## Gap Analysis Summary

| Category | Total Gaps | Critical | High | Medium | Low |
|----------|-----------|----------|------|--------|-----|
| MCP Protocol | 6 | 1 | 4 | 1 | 0 |
| Clean Architecture | 6 | 0 | 2 | 3 | 1 |
| Test Coverage | 8 | 2 | 1 | 4 | 1 |
| Security | 8 | 1 | 2 | 3 | 2 |
| Documentation | 10 | 0 | 3 | 2 | 5 |
| LanceDB Patterns | 12 | 3 | 4 | 3 | 2 |
| Code Quality | 15 | 3 | 4 | 5 | 3 |
| Performance | 8 | 0 | 0 | 4 | 4 |
| **TOTAL** | **71** | **8** | **16** | **25** | **22** |

---

## Phase 4 Structure

### Phase 4A: Critical Fixes & Technical Debt (Week 1)

#### 4A.1 Fix Static Retry Decorator (Critical)
**File:** `spatial_memory/core/database.py:101-161`
**Issue:** Retry decorator uses hardcoded values, ignoring instance config (`max_retry_attempts`, `retry_backoff_seconds`)

```python
# BEFORE (static - ignores instance config)
@retry_on_storage_error(max_attempts=3, backoff=0.5)
def insert(self, ...):

# AFTER (dynamic - uses instance config)
def insert(self, ...):
    return self._with_retry(self._insert_impl, ...)

def _with_retry(self, func: Callable, *args, **kwargs) -> Any:
    """Execute function with instance-configured retry logic."""
    for attempt in range(self.max_retry_attempts):
        try:
            return func(*args, **kwargs)
        except StorageError as e:
            if attempt == self.max_retry_attempts - 1:
                raise
            time.sleep(self.retry_backoff_seconds * (2 ** attempt))
```

**Tests:**
- `test_retry_uses_instance_config`
- `test_retry_respects_max_attempts`
- `test_retry_respects_backoff_seconds`

---

#### 4A.2 Use merge_insert for Atomic Updates (Critical)
**File:** `spatial_memory/core/database.py:1297-1370`
**Issue:** Delete+insert pattern creates fragmentation and has race condition window

```python
# BEFORE (non-atomic)
self.table.delete(f"id = '{safe_id}'")
self.table.add([existing])

# AFTER (atomic upsert)
(
    self.table.merge_insert("id")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute([existing])
)
```

**Tests:**
- `test_update_atomic_under_concurrent_access`
- `test_update_no_data_loss_on_failure`
- `test_merge_insert_reduces_fragmentation`

---

#### 4A.3 Extract Duplicate Code Patterns (High)
**Issue:** JSON metadata parsing duplicated 6+ times

```python
# NEW: spatial_memory/core/helpers.py
def deserialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Deserialize database record fields."""
    if record.get("metadata"):
        record["metadata"] = json.loads(record["metadata"])
    else:
        record["metadata"] = {}

    if "_distance" in record:
        record["similarity"] = max(0.0, min(1.0, 1 - record["_distance"]))
        del record["_distance"]

    return record
```

---

#### 4A.4 Consolidate Validation Logic (High)
**Issue:** Validation spread across database.py and memory.py

```python
# NEW: spatial_memory/core/validation.py
class ValidationService:
    """Centralized input validation."""

    MAX_CONTENT_LENGTH = 100_000
    MAX_TAGS = 20
    MAX_TAG_LENGTH = 50
    MAX_METADATA_SIZE = 10_000

    @staticmethod
    def validate_content(content: str) -> None: ...

    @staticmethod
    def validate_namespace(namespace: str) -> None: ...

    @staticmethod
    def validate_importance(importance: float) -> None: ...

    @staticmethod
    def validate_tags(tags: list[str] | None) -> list[str]: ...

    @staticmethod
    def validate_metadata(metadata: dict | None) -> dict: ...

    @staticmethod
    def validate_uuid(value: str) -> str: ...
```

---

#### 4A.5 Consolidate ConfigurationError (Minor)
**Issue:** Defined in both `config.py` and `errors.py`

**Action:** Remove from `config.py`, import from `errors.py`

---

### Phase 4B: Spatial Operations (Week 2)

#### 4B.1 Journey Tool - SLERP Interpolation
**New Files:**
- `spatial_memory/services/spatial.py`
- `tests/unit/test_spatial_service.py`
- `tests/integration/test_spatial_tools.py`

```python
# spatial_memory/services/spatial.py
class SpatialService:
    """Service for spatial memory navigation and exploration."""

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings

    def journey(
        self,
        start_id: str,
        end_id: str,
        steps: int = 5,
        namespace: str | None = None,
    ) -> list[JourneyStep]:
        """Navigate semantic space between two memories using SLERP.

        Args:
            start_id: Starting memory ID
            end_id: Ending memory ID
            steps: Number of interpolation steps (3-20)
            namespace: Optional namespace filter

        Returns:
            List of JourneyStep with interpolated positions and nearby memories
        """
        # 1. Get start and end vectors
        start_memory = self._repository.get(start_id)
        end_memory = self._repository.get(end_id)

        # 2. SLERP interpolation between vectors
        interpolated_vectors = self._slerp_interpolate(
            start_memory.vector,
            end_memory.vector,
            steps,
        )

        # 3. For each interpolated point, find nearest memories
        journey_steps = []
        for i, vector in enumerate(interpolated_vectors):
            nearby = self._repository.vector_search(
                vector, limit=3, namespace=namespace
            )
            journey_steps.append(JourneyStep(
                step=i,
                position=vector.tolist(),
                progress=i / (steps - 1),
                nearby_memories=[MemoryResult.from_dict(m) for m in nearby],
            ))

        return journey_steps

    def _slerp_interpolate(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        steps: int,
    ) -> list[np.ndarray]:
        """Spherical linear interpolation between two vectors."""
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Calculate angle
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        theta = np.arccos(dot)

        # Generate interpolated vectors
        vectors = []
        for i in range(steps):
            t = i / (steps - 1)
            if theta < 1e-6:
                # Vectors are nearly identical, use linear interpolation
                v = (1 - t) * v1 + t * v2
            else:
                # SLERP formula
                v = (np.sin((1 - t) * theta) / np.sin(theta)) * v1_norm + \
                    (np.sin(t * theta) / np.sin(theta)) * v2_norm
            vectors.append(v)

        return vectors
```

**MCP Tool Definition:**
```python
{
    "name": "journey",
    "description": "Navigate semantic space between two memories using spherical interpolation (SLERP). Discovers memories along the conceptual path.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "start_id": {"type": "string", "description": "Starting memory UUID"},
            "end_id": {"type": "string", "description": "Ending memory UUID"},
            "steps": {"type": "integer", "minimum": 3, "maximum": 20, "default": 5},
            "namespace": {"type": "string", "description": "Optional namespace filter"},
        },
        "required": ["start_id", "end_id"],
    },
}
```

---

#### 4B.2 Wander Tool - Random Walk Discovery
```python
def wander(
    self,
    start_id: str | None = None,
    steps: int = 5,
    randomness: float = 0.3,
    namespace: str | None = None,
) -> list[WanderStep]:
    """Explore memory space through random walk.

    Args:
        start_id: Optional starting memory (random if not provided)
        steps: Number of exploration steps
        randomness: How much to deviate from current direction (0-1)
        namespace: Optional namespace filter

    Returns:
        List of WanderStep with discovered memories and paths
    """
    if start_id is None:
        # Start from random memory
        all_memories = self._repository.get_all(namespace=namespace, limit=100)
        if not all_memories:
            raise ValidationError("No memories to explore")
        start_memory = random.choice(all_memories)
    else:
        start_memory = self._repository.get(start_id)

    visited = {start_memory["id"]}
    current_vector = np.array(start_memory["vector"])
    steps_taken = [WanderStep(
        step=0,
        memory=MemoryResult.from_dict(start_memory),
        direction=None,
    )]

    for step in range(1, steps):
        # Add random perturbation to current vector
        noise = np.random.randn(len(current_vector)) * randomness
        exploration_vector = current_vector + noise
        exploration_vector = exploration_vector / np.linalg.norm(exploration_vector)

        # Find nearest unvisited memory
        candidates = self._repository.vector_search(
            exploration_vector, limit=10, namespace=namespace
        )

        next_memory = None
        for candidate in candidates:
            if candidate["id"] not in visited:
                next_memory = candidate
                break

        if next_memory is None:
            break  # No more unvisited memories

        visited.add(next_memory["id"])
        current_vector = np.array(next_memory["vector"])

        steps_taken.append(WanderStep(
            step=step,
            memory=MemoryResult.from_dict(next_memory),
            direction=exploration_vector.tolist(),
        ))

    return steps_taken
```

---

#### 4B.3 Regions Tool - HDBSCAN Clustering
```python
def regions(
    self,
    namespace: str | None = None,
    min_cluster_size: int = 5,
    max_clusters: int | None = None,
) -> list[ClusterInfo]:
    """Discover semantic clusters in memory space using HDBSCAN.

    Args:
        namespace: Optional namespace filter
        min_cluster_size: Minimum memories per cluster
        max_clusters: Maximum number of clusters to return

    Returns:
        List of ClusterInfo with cluster statistics and representative memories
    """
    import hdbscan

    # Get all memories
    memories = self._repository.get_all(namespace=namespace, limit=10000)
    if len(memories) < min_cluster_size:
        raise ValidationError(f"Need at least {min_cluster_size} memories for clustering")

    # Extract vectors
    vectors = np.array([m["vector"] for m in memories])

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vectors)

    # Build cluster info
    clusters = []
    unique_labels = set(labels) - {-1}  # -1 is noise

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_memories = [m for m, is_member in zip(memories, mask) if is_member]
        cluster_vectors = vectors[mask]

        # Calculate centroid
        centroid = cluster_vectors.mean(axis=0)

        # Find representative memory (closest to centroid)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        representative_idx = np.argmin(distances)
        representative = cluster_memories[representative_idx]

        # Generate cluster summary from top keywords
        all_content = " ".join(m["content"][:200] for m in cluster_memories)
        keywords = self._extract_keywords(all_content, n=5)

        clusters.append(ClusterInfo(
            cluster_id=int(label),
            size=len(cluster_memories),
            centroid=centroid.tolist(),
            representative_memory=MemoryResult.from_dict(representative),
            keywords=keywords,
            coherence=float(1 - distances.mean()),
        ))

    # Sort by size, limit if needed
    clusters.sort(key=lambda c: c.size, reverse=True)
    if max_clusters:
        clusters = clusters[:max_clusters]

    return clusters
```

---

#### 4B.4 Visualize Tool - JSON/Mermaid/SVG Output
```python
def visualize(
    self,
    memory_ids: list[str] | None = None,
    namespace: str | None = None,
    format: Literal["json", "mermaid", "svg"] = "json",
    include_edges: bool = True,
    similarity_threshold: float = 0.7,
) -> VisualizationData:
    """Generate visualization of memory space.

    Args:
        memory_ids: Specific memories to visualize (or all in namespace)
        namespace: Namespace filter
        format: Output format (json, mermaid, svg)
        include_edges: Include similarity edges between memories
        similarity_threshold: Minimum similarity for edge inclusion

    Returns:
        VisualizationData with nodes, edges, and formatted output
    """
    # Get memories
    if memory_ids:
        memories = [self._repository.get(mid) for mid in memory_ids]
    else:
        memories = self._repository.get_all(namespace=namespace, limit=500)

    if not memories:
        raise ValidationError("No memories to visualize")

    # Project to 2D using UMAP
    import umap

    vectors = np.array([m["vector"] for m in memories])
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords_2d = reducer.fit_transform(vectors)

    # Build nodes
    nodes = []
    for i, memory in enumerate(memories):
        nodes.append(VisualizationNode(
            id=memory["id"],
            label=memory["content"][:50],
            x=float(coords_2d[i, 0]),
            y=float(coords_2d[i, 1]),
            namespace=memory["namespace"],
            importance=memory["importance"],
        ))

    # Build edges (similarity connections)
    edges = []
    if include_edges:
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(vectors)

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    edges.append(VisualizationEdge(
                        source=memories[i]["id"],
                        target=memories[j]["id"],
                        weight=float(sim),
                    ))

    # Format output
    if format == "json":
        output = self._format_json(nodes, edges)
    elif format == "mermaid":
        output = self._format_mermaid(nodes, edges)
    elif format == "svg":
        output = self._format_svg(nodes, edges)

    return VisualizationData(
        nodes=nodes,
        edges=edges,
        format=format,
        output=output,
    )
```

---

### Phase 4C: Resilience & Scalability (Week 3)

#### 4C.1 Circuit Breaker Pattern
**New File:** `spatial_memory/core/resilience.py`

```python
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, TypeVar
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""

    failure_threshold: int = 5
    reset_timeout_seconds: float = 60.0
    half_open_requests: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitOpenError(
                        f"Circuit is open, retry after {self._time_until_reset():.1f}s"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_requests:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
```

**Integration Points:**
- Wrap `EmbeddingService._embed_openai()` calls
- Wrap external API calls in hybrid search

---

#### 4C.2 Cursor-Based Pagination
**New File:** `spatial_memory/core/pagination.py`

```python
@dataclass
class PaginatedResult(Generic[T]):
    """Paginated result with cursor for next page."""

    items: list[T]
    total: int
    cursor: str | None
    has_more: bool

    @classmethod
    def encode_cursor(cls, offset: int, filters: dict[str, Any]) -> str:
        """Encode pagination state into opaque cursor."""
        import base64
        import json
        payload = {"offset": offset, "filters": filters}
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    @classmethod
    def decode_cursor(cls, cursor: str) -> tuple[int, dict[str, Any]]:
        """Decode cursor back to pagination state."""
        import base64
        import json
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return payload["offset"], payload["filters"]
```

**Integration:**
- Add `cursor` parameter to `recall`, `nearby`, `regions` tools
- Return `PaginatedResult` instead of plain list for large results

---

#### 4C.3 Graceful Degradation Manager
**New File:** `spatial_memory/core/degradation.py`

```python
class DegradationMode(Enum):
    FULL = "full"           # All features available
    VECTOR_ONLY = "vector_only"  # FTS unavailable
    READ_ONLY = "read_only"      # Writes disabled
    CACHED_ONLY = "cached_only"  # Only cached embeddings
    MINIMAL = "minimal"     # Basic operations only

@dataclass
class ServiceDegradationManager:
    """Manages graceful degradation when components fail."""

    embedding_circuit: CircuitBreaker
    fts_available: bool = True
    write_enabled: bool = True

    def get_search_mode(self) -> str:
        """Determine available search capabilities."""
        if not self.fts_available:
            return "vector_only"
        if self.embedding_circuit._state == CircuitState.OPEN:
            return "cached_only"
        return "hybrid"

    def should_allow_write(self) -> bool:
        """Check if write operations are allowed."""
        return self.write_enabled and \
               self.embedding_circuit._state != CircuitState.OPEN
```

---

#### 4C.4 Async Operation Support
**Enhancement to:** `spatial_memory/server.py`

```python
async def _handle_tool_async(
    self,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Handle tool calls asynchronously."""
    loop = asyncio.get_event_loop()

    # Run blocking database operations in thread pool
    result = await loop.run_in_executor(
        self._executor,  # ThreadPoolExecutor
        self._handle_tool_impl,
        name,
        arguments,
    )

    return result
```

---

### Phase 4D: Lifecycle Operations (Week 4)

#### 4D.1 Consolidate Tool - Merge Similar Memories
```python
def consolidate(
    self,
    namespace: str | None = None,
    similarity_threshold: float = 0.95,
    dry_run: bool = True,
) -> ConsolidationResult:
    """Merge highly similar memories to reduce redundancy.

    Args:
        namespace: Namespace to consolidate
        similarity_threshold: Minimum similarity to consider duplicates
        dry_run: If True, only report what would be merged

    Returns:
        ConsolidationResult with merge groups and statistics
    """
    memories = self._repository.get_all(namespace=namespace)
    vectors = np.array([m["vector"] for m in memories])

    # Find similarity pairs above threshold
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)

    # Build merge groups using union-find
    merge_groups = []
    merged = set()

    for i in range(len(memories)):
        if i in merged:
            continue

        group = [memories[i]]
        for j in range(i + 1, len(memories)):
            if j in merged:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                group.append(memories[j])
                merged.add(j)

        if len(group) > 1:
            merged.add(i)
            merge_groups.append(group)

    if not dry_run:
        for group in merge_groups:
            self._merge_memory_group(group)

    return ConsolidationResult(
        groups_found=len(merge_groups),
        memories_merged=sum(len(g) - 1 for g in merge_groups),
        dry_run=dry_run,
        merge_groups=[
            MergeGroup(
                keep_id=g[0]["id"],
                merge_ids=[m["id"] for m in g[1:]],
                similarity=float(sim_matrix[
                    memories.index(g[0]),
                    memories.index(g[1])
                ]),
            )
            for g in merge_groups
        ],
    )
```

---

#### 4D.2 Decay Tool - Reduce Stale Importance
```python
def decay(
    self,
    namespace: str | None = None,
    decay_factor: float = 0.95,
    min_importance: float = 0.1,
    older_than_days: int = 30,
) -> DecayResult:
    """Apply importance decay to stale memories.

    Args:
        namespace: Namespace to process
        decay_factor: Multiply importance by this factor
        min_importance: Don't decay below this threshold
        older_than_days: Only decay memories older than this

    Returns:
        DecayResult with affected memories count
    """
    cutoff = utc_now() - timedelta(days=older_than_days)
    memories = self._repository.get_all(namespace=namespace)

    decayed_count = 0
    for memory in memories:
        if memory["last_accessed"] < cutoff:
            new_importance = max(
                min_importance,
                memory["importance"] * decay_factor,
            )
            if new_importance != memory["importance"]:
                self._repository.update(memory["id"], {
                    "importance": new_importance,
                })
                decayed_count += 1

    return DecayResult(
        memories_processed=len(memories),
        memories_decayed=decayed_count,
        decay_factor=decay_factor,
    )
```

---

#### 4D.3 Reinforce Tool - Boost Useful Memories
```python
def reinforce(
    self,
    memory_ids: list[str],
    boost_factor: float = 1.1,
    max_importance: float = 1.0,
) -> ReinforceResult:
    """Boost importance of useful memories.

    Args:
        memory_ids: Memories to reinforce
        boost_factor: Multiply importance by this factor
        max_importance: Don't boost above this threshold

    Returns:
        ReinforceResult with affected memories
    """
    reinforced = []
    for memory_id in memory_ids:
        memory = self._repository.get(memory_id)
        new_importance = min(
            max_importance,
            memory["importance"] * boost_factor,
        )

        if new_importance != memory["importance"]:
            self._repository.update(memory_id, {
                "importance": new_importance,
            })
            reinforced.append(memory_id)

    return ReinforceResult(
        memories_reinforced=len(reinforced),
        reinforced_ids=reinforced,
    )
```

---

### Phase 4E: Utility Operations & Polish (Week 5)

#### 4E.1 Stats Tool
```python
def stats(
    self,
    namespace: str | None = None,
) -> MemoryStats:
    """Get comprehensive memory statistics.

    Returns:
        MemoryStats with counts, distributions, health metrics
    """
    return MemoryStats(
        total_memories=self._repository.count(namespace),
        namespaces=self._repository.get_namespaces(),
        importance_distribution=self._calculate_importance_distribution(namespace),
        access_statistics=self._calculate_access_stats(namespace),
        storage_size_bytes=self._get_storage_size(),
        index_status=self._get_index_status(),
        oldest_memory=self._get_oldest_memory(namespace),
        newest_memory=self._get_newest_memory(namespace),
        most_accessed=self._get_most_accessed(namespace, limit=5),
    )
```

---

#### 4E.2 Namespaces Tool
```python
def namespaces(
    self,
    action: Literal["list", "create", "delete", "stats"] = "list",
    namespace: str | None = None,
) -> NamespaceResult:
    """Manage memory namespaces.

    Actions:
        list: List all namespaces with counts
        create: Create namespace (no-op, created on first use)
        delete: Delete all memories in namespace
        stats: Get detailed namespace statistics
    """
    if action == "list":
        ns_list = self._repository.get_namespaces()
        return NamespaceResult(
            namespaces=[
                NamespaceInfo(name=ns, count=self._repository.count(ns))
                for ns in ns_list
            ]
        )
    elif action == "delete":
        if not namespace:
            raise ValidationError("namespace required for delete action")
        count = self._repository.delete_by_namespace(namespace)
        return NamespaceResult(deleted_count=count)
    elif action == "stats":
        if not namespace:
            raise ValidationError("namespace required for stats action")
        return NamespaceResult(
            stats=self._get_namespace_stats(namespace)
        )
```

---

#### 4E.3 Wire Rate Limiter to MCP Tools
**File:** `spatial_memory/server.py`

```python
class SpatialMemoryServer:
    def __init__(self):
        # ... existing init ...
        self._embedding_limiter = RateLimiter(
            rate=settings.embedding_rate_limit,
            capacity=int(settings.embedding_rate_limit * 2),
        )
        self._batch_limiter = RateLimiter(
            rate=settings.batch_rate_limit,
            capacity=int(settings.batch_rate_limit * 2),
        )

    def _handle_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Apply rate limiting based on tool type
        if name in ("remember", "recall", "nearby"):
            if not self._embedding_limiter.acquire():
                raise RateLimitError("Embedding rate limit exceeded")

        if name in ("remember_batch", "forget_batch"):
            if not self._batch_limiter.acquire():
                raise RateLimitError("Batch rate limit exceeded")

        # ... rest of handler ...
```

---

### Phase 4F: LanceDB Optimizations (Week 6)

#### 4F.1 Add Reranker Options
```python
class RerankerType(str, Enum):
    LINEAR = "linear"
    RRF = "rrf"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    COLBERT = "colbert"

def hybrid_search(
    self,
    query: str,
    query_vector: np.ndarray,
    limit: int = 5,
    namespace: str | None = None,
    alpha: float = 0.5,
    reranker: RerankerType = RerankerType.LINEAR,
    reranker_model: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search with configurable reranking."""
    # ... implementation as detailed in LanceDB analysis ...
```

---

#### 4F.2 Add Distance Range Filtering
```python
def vector_search(
    self,
    query_vector: np.ndarray,
    limit: int = 5,
    namespace: str | None = None,
    min_similarity: float = 0.0,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Search with distance range filtering."""
    search = self.table.search(query_vector.tolist())

    # Convert similarity to distance for filtering
    if min_similarity > 0.0 and max_distance is None:
        max_distance = 1.0 - min_similarity

    if max_distance is not None:
        search = search.distance_range(upper_bound=max_distance)

    # ... rest of implementation ...
```

---

#### 4F.3 Automatic Maintenance Routine
```python
def run_maintenance(self) -> dict[str, Any]:
    """Run full database maintenance routine.

    Performs:
    1. Compact small fragments
    2. Optimize indexes
    3. Cleanup old versions
    4. Cleanup expired memories
    """
    results = {}

    # 1. Compaction
    stats = self._get_table_stats()
    if stats.get("num_small_fragments", 0) > self.compaction_threshold:
        self.table.compact_files()
        results["compaction"] = "performed"

    # 2. Index optimization
    self.table.optimize()
    results["optimize"] = "performed"

    # 3. Version cleanup
    self.table.cleanup_old_versions(
        older_than=timedelta(days=self.version_retention_days),
        delete_unverified=True,
    )
    results["version_cleanup"] = "performed"

    # 4. Expired memory cleanup
    expired_count = self.cleanup_expired_memories()
    results["expired_cleanup"] = expired_count

    return results
```

---

## Testing Strategy

### New Test Files Required

| File | Tests | Priority |
|------|-------|----------|
| `tests/unit/test_spatial_service.py` | Journey, Wander, Regions, Visualize | P0 |
| `tests/unit/test_lifecycle_service.py` | Consolidate, Decay, Reinforce | P1 |
| `tests/unit/test_validation_service.py` | Centralized validation | P0 |
| `tests/unit/test_circuit_breaker.py` | Circuit states, timeouts | P0 |
| `tests/unit/test_pagination.py` | Cursor encoding/decoding | P0 |
| `tests/integration/test_spatial_tools.py` | End-to-end spatial ops | P0 |
| `tests/integration/test_lifecycle_tools.py` | End-to-end lifecycle ops | P1 |
| `tests/integration/test_resilience.py` | Degradation modes | P1 |
| `tests/load/test_concurrent_operations.py` | 100+ concurrent requests | P2 |
| `tests/performance/test_benchmarks.py` | 100K, 1M record performance | P2 |

### Test Coverage Targets

| Component | Current | Target |
|-----------|---------|--------|
| Spatial Service | 0% | 90% |
| Lifecycle Service | 0% | 90% |
| Circuit Breaker | 0% | 95% |
| Pagination | 0% | 95% |
| Validation Service | 0% | 95% |
| Overall | ~80% | 90% |

---

## Documentation Deliverables

| Document | Description | Priority |
|----------|-------------|----------|
| `API.md` | Complete tool reference with examples | P0 |
| `DEPLOYMENT.md` | Docker, Claude Desktop, production setup | P1 |
| `CONFIGURATION.md` | All 40+ config options explained | P1 |
| `PERFORMANCE.md` | Tuning guide, benchmarks | P2 |
| `DEVELOPMENT.md` | Contributing guide | P2 |
| `examples/journey_demo.py` | Journey tool example | P1 |
| `examples/clustering_demo.py` | Regions tool example | P1 |

---

## Configuration Additions

```python
# spatial_memory/config.py additions

class Settings(BaseSettings):
    # ... existing ...

    # Circuit Breaker
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_timeout_seconds: float = Field(default=60.0, ge=1.0)
    circuit_half_open_requests: int = Field(default=3, ge=1)

    # Reranking
    default_reranker: str = Field(default="linear")
    cohere_api_key: str | None = Field(default=None)
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Lifecycle
    auto_decay_enabled: bool = Field(default=False)
    decay_interval_hours: int = Field(default=24)
    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    min_importance_after_decay: float = Field(default=0.1, ge=0.0, le=1.0)

    # Maintenance
    auto_maintenance_enabled: bool = Field(default=False)
    maintenance_interval_hours: int = Field(default=24)
    version_retention_days: int = Field(default=7, ge=1)

    # Pagination
    default_page_size: int = Field(default=20, ge=1, le=100)
    max_page_size: int = Field(default=100, ge=1, le=1000)
```

---

## File Structure After Phase 4

```
spatial_memory/
├── __init__.py
├── server.py                    # MCP server (updated with 19 tools)
├── config.py                    # Settings (expanded)
├── core/
│   ├── __init__.py
│   ├── database.py              # LanceDB operations (refactored)
│   ├── embeddings.py            # Embedding service
│   ├── errors.py                # Exception hierarchy
│   ├── models.py                # Domain models
│   ├── connection_pool.py       # Connection pooling
│   ├── rate_limiter.py          # Rate limiting
│   ├── health.py                # Health checks
│   ├── logging.py               # Secure logging
│   ├── metrics.py               # Prometheus metrics
│   ├── validation.py            # NEW: Centralized validation
│   ├── helpers.py               # NEW: Common utilities
│   ├── resilience.py            # NEW: Circuit breaker
│   ├── pagination.py            # NEW: Cursor pagination
│   └── degradation.py           # NEW: Graceful degradation
├── services/
│   ├── __init__.py
│   ├── memory.py                # Memory service
│   ├── spatial.py               # NEW: Spatial operations
│   ├── lifecycle.py             # NEW: Lifecycle operations
│   └── utility.py               # NEW: Stats, namespaces
├── ports/
│   └── repositories.py          # Protocol interfaces
└── adapters/
    └── lancedb_repository.py    # LanceDB adapter

tests/
├── conftest.py
├── unit/
│   ├── test_memory_service.py
│   ├── test_spatial_service.py       # NEW
│   ├── test_lifecycle_service.py     # NEW
│   ├── test_validation_service.py    # NEW
│   ├── test_circuit_breaker.py       # NEW
│   └── test_pagination.py            # NEW
├── integration/
│   ├── test_mcp_server.py
│   ├── test_spatial_tools.py         # NEW
│   ├── test_lifecycle_tools.py       # NEW
│   └── test_resilience.py            # NEW
├── load/
│   └── test_concurrent_operations.py # NEW
└── performance/
    └── test_benchmarks.py            # NEW

docs/
├── API.md                       # NEW
├── DEPLOYMENT.md                # NEW
├── CONFIGURATION.md             # NEW
├── PERFORMANCE.md               # NEW
├── METRICS.md
└── troubleshooting.md

examples/
├── journey_demo.py              # NEW
├── clustering_demo.py           # NEW
└── demo_config_logging.py
```

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | 4A | Critical fixes, technical debt, validation consolidation |
| 2 | 4B | Spatial operations (journey, wander, regions, visualize) |
| 3 | 4C | Resilience (circuit breaker, pagination, degradation, async) |
| 4 | 4D | Lifecycle operations (consolidate, decay, reinforce) |
| 5 | 4E | Utility operations, rate limiting, polish |
| 6 | 4F | LanceDB optimizations, documentation, final testing |

---

## Success Criteria

### Functional
- [ ] All 19 MCP tools implemented and tested
- [ ] Circuit breaker protects against cascade failures
- [ ] Pagination supports 100K+ result sets
- [ ] Graceful degradation when components fail
- [ ] Rate limiting prevents resource exhaustion

### Quality
- [ ] 90%+ test coverage
- [ ] All tests pass (target: 400+ tests)
- [ ] mypy strict mode: 0 errors
- [ ] ruff: 0 issues
- [ ] No critical/high security issues

### Performance
- [ ] Recall latency <100ms for 10K memories
- [ ] Journey latency <500ms for 10 steps
- [ ] Regions clustering <5s for 10K memories
- [ ] Batch insert: 1000 memories/second

### Documentation
- [ ] Complete API reference
- [ ] Deployment guide with Docker
- [ ] Configuration reference
- [ ] Performance tuning guide

---

## Verification Commands

```bash
# After each phase:

# 1. Type checking
mypy spatial_memory --strict

# 2. Linting
ruff check spatial_memory tests

# 3. Unit tests
pytest tests/unit -v --cov=spatial_memory --cov-report=term-missing

# 4. Integration tests
pytest tests/integration -v

# 5. All tests
pytest tests/ -v

# 6. Performance tests (Phase 4E+)
pytest tests/performance -v --benchmark-only

# 7. Load tests (Phase 4E+)
pytest tests/load -v --timeout=300
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| UMAP/HDBSCAN scaling issues | Add fallback to simpler algorithms for large datasets |
| Circuit breaker over-triggers | Tune thresholds, add manual override |
| Pagination cursor security | Sign cursors with HMAC |
| Rate limiter too aggressive | Make limits configurable per-tool |
| Breaking changes to LanceDB API | Pin version, add compatibility layer |

---

## Approval Checklist

Before starting implementation:

- [ ] Review and approve this plan
- [ ] Confirm timeline is acceptable
- [ ] Identify any missing requirements
- [ ] Confirm testing strategy
- [ ] Approve new dependencies (if any)
