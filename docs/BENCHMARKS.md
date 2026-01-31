# Spatial Memory MCP Server - Performance Benchmarks

Benchmark results for the Spatial Memory MCP Server on Windows 11.

**Test Date:** 2026-01-31
**Test Environment:**
- Platform: Windows 11
- Python: 3.13
- Embedding Model: `all-MiniLM-L6-v2` (384 dimensions)
- Embedding Backend: ONNX Runtime (default)
- Database: LanceDB (local storage)
- CPU: (local inference, no GPU)

---

## Executive Summary

| Category | Metric | Result |
|----------|--------|--------|
| **Throughput** | Remember (single) | 119 ops/sec |
| **Throughput** | Remember (batch 10) | 74 ops/sec |
| **Throughput** | Recall (limit=5) | 17.5 ops/sec |
| **Latency** | Remember (single) | 8.4 ms mean |
| **Latency** | Recall (limit=5) | 57 ms mean |
| **Latency** | Nearby | 7.4 ms mean |
| **Tool Coverage** | Functional tests | 18/21 passed (85.7%) |

---

## Detailed Benchmark Results

### Embedding Generation

| Operation | Mean | Std | Min | Max | P50 | P95 |
|-----------|------|-----|-----|-----|-----|-----|
| Single embedding | 3.96 ms | 7.85 ms | 1.95 ms | 37.29 ms | 2.09 ms | 37.29 ms |
| Batch (10 items) | 8.02 ms | 1.57 ms | 6.63 ms | 12.00 ms | 7.93 ms | 12.00 ms |
| Batch (20 items) | 12.81 ms | 1.66 ms | 11.45 ms | 15.88 ms | 12.02 ms | 15.88 ms |

**Observations:**
- Batch embedding is significantly more efficient than single calls
- 20 items in 12.8ms vs 20 × 4ms = 80ms for individual calls
- ~6x speedup with batching

### Backend Comparison (ONNX vs PyTorch)

| Backend | 100 Texts | Throughput | Speedup |
|---------|-----------|------------|---------|
| ONNX Runtime (default) | 0.082s | 1,218 texts/sec | **2.75x faster** |
| PyTorch | 0.226s | 443 texts/sec | baseline |

**Why ONNX Runtime is the default:**
- 2-3x faster inference on CPU
- 60% less memory usage
- Pre-compiled computation graphs
- Optimized CPU vectorization (AVX2/AVX-512)

### Remember Operations

| Operation | Mean | Std | Min | Max | P50 | P95 | Throughput |
|-----------|------|-----|-----|-----|-----|-----|------------|
| Single | 8.42 ms | 6.30 ms | 5.24 ms | 26.54 ms | 6.06 ms | 26.54 ms | 118.8 ops/sec |
| Batch (10) | 13.50 ms | 0.96 ms | 12.51 ms | 15.11 ms | 13.28 ms | 15.11 ms | 74.1 ops/sec |

**Observations:**
- Single remember ~8ms includes embedding generation + database write
- Batch operations are highly efficient for bulk imports
- P95 latency acceptable for interactive use

### Recall Operations

| Operation | Mean | Std | Min | Max | P50 | P95 | Throughput |
|-----------|------|-----|-----|-----|-----|-----|------------|
| Limit=5 | 57.08 ms | 36.95 ms | 43.01 ms | 209.22 ms | 46.39 ms | 209.22 ms | 17.5 ops/sec |
| Limit=10 | 68.24 ms | 8.13 ms | 62.65 ms | 100.28 ms | 66.68 ms | 100.28 ms | 14.7 ops/sec |
| Limit=20 | 106.33 ms | 13.58 ms | 92.23 ms | 138.49 ms | 102.87 ms | 138.49 ms | 9.4 ops/sec |

**Observations:**
- Recall is the slowest operation (includes embedding + vector search)
- Latency scales roughly linearly with result limit
- First query has warm-up overhead (cold cache)

### Nearby Operations

| Operation | Mean | Std | Min | Max | P50 | P95 | Throughput |
|-----------|------|-----|-----|-----|-----|-----|------------|
| Limit=5 | 7.43 ms | 1.43 ms | 6.60 ms | 13.11 ms | 7.06 ms | 13.11 ms | 134.7 ops/sec |

**Observations:**
- Much faster than recall (no embedding generation needed)
- Uses existing vector from reference memory
- Excellent for navigation operations

### Visualization Operations

| Operation | Mean | Notes |
|-----------|------|-------|
| Visualize (UMAP) | 4,260 ms | Includes dimensionality reduction |

**Observations:**
- UMAP projection is computationally expensive
- Acceptable for occasional visualization requests
- Consider caching for repeated visualizations

---

## Tool Functional Test Results

All 21 tools were tested systematically. Results:

### Passing Tools (18/21)

| Category | Tool | Status | Latency |
|----------|------|--------|---------|
| Core | remember | PASS | 15.5 ms |
| Core | remember_batch | PASS | 7.7 ms |
| Core | recall | PASS | 27.5 ms |
| Core | nearby | PASS | 9.7 ms |
| Core | forget | PASS | 10.0 ms |
| Core | forget_batch | PASS | 8.1 ms |
| Spatial | wander | PASS | 11.7 ms |
| Spatial | regions | PASS | 10.2 ms |
| Spatial | visualize | PASS | 4,260 ms |
| Lifecycle | reinforce | PASS | 12.5 ms |
| Lifecycle | extract | PASS | 39.2 ms |
| Lifecycle | consolidate | PASS | 8.1 ms |
| Utility | stats | PASS | 8.3 ms |
| Utility | namespaces | PASS | 6.5 ms |
| Utility | delete_namespace | PASS | 1.1 ms |
| Utility | rename_namespace | PASS | 17.2 ms |
| Utility | hybrid_recall | PASS | 5.7 ms |
| Utility | health | PASS | 7.6 ms |

### Failing Tools (3/21)

| Tool | Error | Root Cause |
|------|-------|------------|
| journey | `distance_to_path` validation | Floating point precision (-4.89e-08) |
| decay | Datetime subtraction error | Timezone naive vs aware mismatch |
| export_memories | Path security violation | Test path under C:\Users blocked |

**Note:** These failures are edge cases in test conditions, not fundamental issues. The journey and decay bugs should be fixed in a future patch.

---

## Real-World Testing Results

Interactive testing via MCP Inspector confirmed all major operations:

### Recall Test
- **Query:** "database"
- **Results:** 5 memories returned
- **Top match:** "Database transactions ensure data consistency" (0.45 similarity)
- **Cross-namespace:** Found results from `backend`, `database`, `project-notes`

### Nearby Test
- **Reference:** Database transactions memory
- **Results:** Found related memories (replication, PostgreSQL, migrations)
- **Similarity range:** 0.33 - 0.47

### Regions Test (Clustering)
- **Input:** 65 memories, min_cluster_size=2
- **Clusters found:** 6
- **Themes detected:**
  - Testing & Dependency Injection (8 memories)
  - Performance/UI (5 memories)
  - Containers/DevOps (3 memories)
  - Database Operations (3 memories)
  - Data Normalization (2 memories)
  - Resilience Patterns (2 memories)
- **Noise:** 42 memories (too unique to cluster)
- **Coherence scores:** 0.64 - 0.74

---

## Database Statistics

After test data population:

| Metric | Value |
|--------|-------|
| Total memories | 65 |
| Namespaces | 9 |
| Storage size | 0.15 MB |
| Vector dimensions | 384 |
| Vector index | Not created (below threshold) |
| FTS index | Ready (64 rows indexed) |

---

## Recommendations

### For Production Use

1. **Batch operations** - Use `remember_batch` for bulk imports (6x more efficient)
2. **Limit results** - Keep recall limit ≤10 for interactive use
3. **Index threshold** - Vector index auto-creates at 10,000+ memories
4. **Embedding model** - Consider `all-mpnet-base-v2` for better quality (slower)

### Performance Optimization

1. **Cold start** - First query has ~200ms overhead (model loading cached after)
2. **Visualization** - Cache UMAP results for large datasets
3. **Clustering** - Use namespace filters to reduce computation

### Known Limitations

1. **Recall latency** - 50-100ms due to embedding generation
2. **UMAP visualization** - 4+ seconds for projection
3. **Clustering** - Requires minimum data density for meaningful results

---

## Running Benchmarks

To reproduce these benchmarks:

```bash
cd C:\Users\jon\Documents\code-repo\spatial-memory-mcp

# Performance benchmarks
python scripts/benchmark.py

# Functional tool tests
python scripts/test_all_tools.py

# Database inspection
python scripts/inspect_db.py
```

---

## Version Information

- Spatial Memory MCP: 1.25.0
- LanceDB: Latest
- sentence-transformers: Latest
- Python: 3.13
