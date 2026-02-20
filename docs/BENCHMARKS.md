# Spatial Memory MCP Server - Performance Benchmarks

Benchmark results for the Spatial Memory MCP Server on Windows 11.

**Test Date:** 2026-02-19
**Version:** 1.11.3
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
| **Throughput** | Remember (single) | 112 ops/sec |
| **Throughput** | Recall (limit=5) | 27.4 ops/sec |
| **Throughput** | Nearby (limit=5) | 110 ops/sec |
| **Latency** | Remember (single) | 8.9 ms mean |
| **Latency** | Recall (limit=5) | 36.5 ms mean |
| **Latency** | Nearby | 9.1 ms mean |
| **Tool Coverage** | Functional tests | 25/25 passed (100%) |

---

## Detailed Benchmark Results

### Embedding Generation

| Operation | Mean | Std | Min | Max | P50 | P95 |
|-----------|------|-----|-----|-----|-----|-----|
| Single embedding | 0.17 ms | 0.74 ms | 0.00 ms | 3.30 ms | 0.00 ms | 3.30 ms |
| Batch (10 items) | 2.82 ms | 8.89 ms | 0.01 ms | 28.13 ms | 0.01 ms | 28.13 ms |
| Batch (20 items) | 2.39 ms | 7.49 ms | 0.02 ms | 23.72 ms | 0.02 ms | 23.72 ms |

**Observations:**
- Batch embedding is significantly more efficient than single calls
- ONNX caching makes repeated embeddings near-instant after warm-up
- First call includes model warm-up overhead

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
| Single | 8.92 ms | 8.39 ms | 4.64 ms | 41.19 ms | 5.77 ms | 41.19 ms | 112.2 ops/sec |
| Batch (10) | 38.09 ms | 6.93 ms | 31.41 ms | 48.33 ms | 39.26 ms | 48.33 ms | 26.3 ops/sec |

**Observations:**
- Single remember ~9ms includes embedding generation + database write
- Batch operations amortize overhead across multiple items
- P95 latency acceptable for interactive use

### Recall Operations

| Operation | Mean | Std | Min | Max | P50 | P95 | Throughput |
|-----------|------|-----|-----|-----|-----|-----|------------|
| Limit=5 | 36.48 ms | 18.82 ms | 25.35 ms | 105.74 ms | 28.87 ms | 105.74 ms | 27.4 ops/sec |
| Limit=10 | 31.33 ms | 8.63 ms | 26.85 ms | 63.91 ms | 28.59 ms | 63.91 ms | 31.9 ops/sec |
| Limit=20 | 33.02 ms | 8.89 ms | 27.19 ms | 68.70 ms | 30.89 ms | 68.70 ms | 30.3 ops/sec |

**Observations:**
- Recall includes embedding generation + vector search
- Latency is consistent across limit sizes (dominated by embedding time)
- First query has warm-up overhead (cold cache)

### Nearby Operations

| Operation | Mean | Std | Min | Max | P50 | P95 | Throughput |
|-----------|------|-----|-----|-----|-----|-----|------------|
| Limit=5 | 9.12 ms | 1.61 ms | 7.87 ms | 15.42 ms | 8.90 ms | 15.42 ms | 109.7 ops/sec |

**Observations:**
- Much faster than recall (no embedding generation needed)
- Uses existing vector from reference memory
- Excellent for navigation operations

### Visualization Operations

| Operation | Mean | Notes |
|-----------|------|-------|
| Visualize (UMAP) | 4,261 ms | Includes dimensionality reduction |

**Observations:**
- UMAP projection is computationally expensive
- Acceptable for occasional visualization requests
- Consider caching for repeated visualizations

---

## Tool Functional Test Results

All 25 tools were tested systematically. Results:

### All Tools Passing (25/25)

| Category | Tool | Status | Latency |
|----------|------|--------|---------|
| Core | remember | PASS | 22.0 ms |
| Core | remember_batch | PASS | 17.8 ms |
| Core | recall | PASS | 22.3 ms |
| Core | nearby | PASS | 11.0 ms |
| Core | forget | PASS | 12.6 ms |
| Core | forget_batch | PASS | 8.0 ms |
| Spatial | journey | PASS | 12.7 ms |
| Spatial | wander | PASS | 9.8 ms |
| Spatial | regions | PASS | 16.8 ms |
| Spatial | visualize | PASS | 4,261 ms |
| Lifecycle | decay | PASS | 3.7 ms |
| Lifecycle | reinforce | PASS | 12.3 ms |
| Lifecycle | extract | PASS | 42.1 ms |
| Lifecycle | consolidate | PASS | 11.7 ms |
| Utility | stats | PASS | 5.8 ms |
| Utility | namespaces | PASS | 15.6 ms |
| Utility | delete_namespace | PASS | 0.8 ms |
| Utility | rename_namespace | PASS | 19.3 ms |
| Utility | export_memories | PASS | 6.3 ms |
| Utility | import_memories | PASS | 4.9 ms |
| Utility | hybrid_recall | PASS | 24.7 ms |
| Utility | health | PASS | 3.1 ms |
| Cross-corpus | discover_connections | PASS | 16.8 ms |
| Cross-corpus | corpus_bridges | PASS | 33.0 ms |
| Setup | setup_hooks | PASS | 1.1 ms |

---

## Recommendations

### For Production Use

1. **Batch operations** - Use `remember_batch` for bulk imports
2. **Limit results** - Keep recall limit <=10 for interactive use
3. **Index threshold** - Vector index auto-creates at 10,000+ memories
4. **Embedding model** - Consider `all-mpnet-base-v2` for better quality (slower)

### Performance Optimization

1. **Cold start** - First query has ~200ms overhead (model loading cached after)
2. **Visualization** - Cache UMAP results for large datasets
3. **Clustering** - Use namespace filters to reduce computation

### Known Limitations

1. **Recall latency** - 30-40ms due to embedding generation
2. **UMAP visualization** - 4+ seconds for projection
3. **Clustering** - Requires minimum data density for meaningful results

---

## Running Benchmarks

To reproduce these benchmarks:

```bash
cd spatial-memory-mcp

# Performance benchmarks
python scripts/benchmark.py

# Functional tool tests (all 25 tools)
python scripts/test_all_tools.py

# Database inspection
python scripts/inspect_db.py
```

---

## Version Information

- Spatial Memory MCP: 1.11.3
- LanceDB: Latest
- sentence-transformers: Latest
- Python: 3.13
