# Configuration Almanac

**Spatial Memory MCP Server** — Complete Configuration Reference

Every environment variable, what it does, when to change it, and all possible variations.

> All variables use the prefix `SPATIAL_MEMORY_`. Set them via environment variables, a `.env` file,
> or in your MCP client configuration (`.mcp.json`, `claude_desktop_config.json`).
> All settings have sensible defaults — none are required.

---

## Table of Contents

- [How Configuration Works](#how-configuration-works)
- [Quick Reference](#quick-reference)
- [Storage & Paths](#storage--paths)
- [Embedding Configuration](#embedding-configuration)
- [OpenAI Integration](#openai-integration)
- [Logging & Observability](#logging--observability)
- [Spatial Operation Limits](#spatial-operation-limits)
- [Vector Indexing](#vector-indexing)
- [Full-Text Search](#full-text-search)
- [Hybrid Search](#hybrid-search)
- [Clustering (HDBSCAN)](#clustering-hdbscan)
- [Visualization (UMAP)](#visualization-umap)
- [Performance & Retry Logic](#performance--retry-logic)
- [Connection Pool](#connection-pool)
- [Cross-Process Locking](#cross-process-locking)
- [Read Consistency](#read-consistency)
- [Rate Limiting](#rate-limiting)
- [Circuit Breaker](#circuit-breaker)
- [Response Caching](#response-caching)
- [Idempotency](#idempotency)
- [Memory TTL & Expiration](#memory-ttl--expiration)
- [Memory Lifecycle: Decay](#memory-lifecycle-decay)
- [Memory Lifecycle: Auto-Decay](#memory-lifecycle-auto-decay)
- [Memory Lifecycle: Reinforcement](#memory-lifecycle-reinforcement)
- [Memory Lifecycle: Extraction](#memory-lifecycle-extraction)
- [Memory Lifecycle: Consolidation](#memory-lifecycle-consolidation)
- [Export Operations](#export-operations)
- [Import Operations](#import-operations)
- [Security & Destructive Operations](#security--destructive-operations)
- [Cognitive Offloading (Proposed v2.0)](#cognitive-offloading-proposed-v20)
- [Deprecated Parameters](#deprecated-parameters)
- [Recipes](#recipes)

---

## How Configuration Works

### Precedence (highest to lowest)

1. **Environment variables** — `export SPATIAL_MEMORY_LOG_LEVEL=DEBUG`
2. **`.env` file** — loaded from the working directory
3. **Built-in defaults** — defined in `spatial_memory/config.py`

MCP client configurations (`.mcp.json`, `claude_desktop_config.json`) set environment variables
in the server process, so they follow rule 1.

### Setting a value

**`.mcp.json`** (recommended):
```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "env": {
        "SPATIAL_MEMORY_MEMORY_PATH": "./my-memory",
        "SPATIAL_MEMORY_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

**`.env` file:**
```env
SPATIAL_MEMORY_MEMORY_PATH=./my-memory
SPATIAL_MEMORY_LOG_LEVEL=DEBUG
```

**Shell:**
```bash
export SPATIAL_MEMORY_MEMORY_PATH=./my-memory
spatial-memory
```

### Type conventions

| Type in this doc | Format | Example |
|------------------|--------|---------|
| `str` | Plain text | `INFO` |
| `int` | Whole number | `100` |
| `float` | Decimal number | `0.5` |
| `bool` | `true` or `false` | `true` |
| `path` | File system path | `./data` or `/home/user/data` |
| `list[str]` | JSON array | `["./exports", "./backups"]` |
| `secret` | Sensitive string (masked in logs) | `sk-...` |

---

## Quick Reference

### Essential (most users configure these)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MEMORY_PATH` | `./.spatial-memory` | Where data is stored |
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Which embedding model to use |
| `SPATIAL_MEMORY_LOG_LEVEL` | `INFO` | Logging verbosity |
| `SPATIAL_MEMORY_AUTO_DECAY_ENABLED` | `true` | Whether memories fade over time |

### Optional (power users)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_EMBEDDING_BACKEND` | `auto` | Force ONNX or PyTorch |
| `SPATIAL_MEMORY_OPENAI_API_KEY` | — | Use OpenAI for embeddings |
| `SPATIAL_MEMORY_ENABLE_FTS_INDEX` | `true` | Full-text search support |
| `SPATIAL_MEMORY_RESPONSE_CACHE_ENABLED` | `true` | Cache repeated queries |
| `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_ENABLED` | `true` | Per-agent rate limiting |

---

## Storage & Paths

### `SPATIAL_MEMORY_MEMORY_PATH`

| | |
|---|---|
| **Type** | `path` |
| **Default** | `./.spatial-memory` |
| **Since** | v0.1.0 |

The directory where LanceDB stores all vector data, indexes, and metadata. If the directory does
not exist, it is created automatically on startup.

**When to change:** Set this to a permanent location when running in production. The default
(relative path) depends on the working directory, which varies by MCP client.

**Values:**
- Relative path — resolved from the server's working directory
- Absolute path — recommended for production (`/home/user/.spatial-memory`)
- Network path — **not recommended** (see `SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FILESYSTEM_RISK`)

**Example:**
```json
"SPATIAL_MEMORY_MEMORY_PATH": "/home/user/.spatial-memory"
```

**Caveats:**
- On Windows, use forward slashes or escaped backslashes: `C:/Users/me/.spatial-memory`
- The MCP server's working directory varies by client. With `uvx`, it may be a cache directory. Always use an absolute path to avoid surprises.

---

### `SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FILESYSTEM_RISK`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.6.0 |

Suppresses the startup warning about network filesystems (NFS, SMB/CIFS, SSHFS). File-based
locking does not work reliably on network filesystems, which can cause data corruption if
multiple MCP server instances write concurrently.

**When to change:** Only set to `true` if you have confirmed that only one server instance
accesses the storage directory, and it happens to be on a network mount.

**Values:**
- `false` (default) — warn at startup if network filesystem detected
- `true` — suppress the warning

**Related:** `SPATIAL_MEMORY_FILELOCK_ENABLED`

---

## Embedding Configuration

### `SPATIAL_MEMORY_EMBEDDING_MODEL`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `all-MiniLM-L6-v2` |
| **Since** | v0.1.0 |

The embedding model used to convert text into vector representations. This is the most impactful
configuration choice — it determines search quality, speed, and storage size.

**When to change:** If you need higher quality search results (use a larger model) or want to use
OpenAI's hosted embeddings instead of local inference.

**Values — Local models** (no API key, runs on your machine):

| Model | Dimensions | Speed | Quality | Notes |
|-------|-----------|-------|---------|-------|
| `all-MiniLM-L6-v2` (default) | 384 | Fast | Good | Best balance for most users |
| `all-mpnet-base-v2` | 768 | Slower | Better | Larger model, more accurate similarity |
| Any HuggingFace model | Varies | Varies | Varies | Must be sentence-transformers compatible |

**Values — OpenAI models** (requires `SPATIAL_MEMORY_OPENAI_API_KEY`):

| Model | Dimensions | Cost | Quality | Notes |
|-------|-----------|------|---------|-------|
| `openai:text-embedding-3-small` | 1536 | $0.02/1M tokens | Good | Fast, cost-effective |
| `openai:text-embedding-3-large` | 3072 | $0.13/1M tokens | Best | Highest quality |
| `openai:text-embedding-ada-002` | 1536 | $0.10/1M tokens | Good | Legacy, use 3-small instead |

**Example:**
```json
"SPATIAL_MEMORY_EMBEDDING_MODEL": "openai:text-embedding-3-small"
```

**Caveats:**
- Changing the model after storing memories **breaks search**. Existing vectors are incompatible
  with the new model's dimensions/space. You must re-embed or start fresh.
- The first run downloads the model (~80MB for MiniLM). Ensure internet connectivity.
- OpenAI models require network access for every embedding operation.

**Related:** `SPATIAL_MEMORY_EMBEDDING_DIMENSIONS`, `SPATIAL_MEMORY_EMBEDDING_BACKEND`, `SPATIAL_MEMORY_OPENAI_API_KEY`

---

### `SPATIAL_MEMORY_EMBEDDING_DIMENSIONS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `384` |
| **Since** | v0.1.0 |

The number of dimensions in the embedding vectors. Auto-detected for known models. Only set this
manually if using a custom model that the auto-detection doesn't recognize.

**When to change:** Almost never. The system auto-detects dimensions for all standard models.
Only set this if you see dimension mismatch errors with a custom model.

**Values:**
- `384` — for `all-MiniLM-L6-v2` (default)
- `768` — for `all-mpnet-base-v2`
- `1536` — for OpenAI `text-embedding-3-small` / `ada-002`
- `3072` — for OpenAI `text-embedding-3-large`

**Related:** `SPATIAL_MEMORY_EMBEDDING_MODEL`

---

### `SPATIAL_MEMORY_EMBEDDING_BACKEND`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `auto` |
| **Since** | v1.6.0 |

Controls which inference engine is used for local embedding models. ONNX Runtime provides 2-3x
faster inference and 60% less memory than PyTorch.

**When to change:** If you want to force a specific backend for benchmarking or troubleshooting,
or if ONNX Runtime is causing issues on your platform.

**Values:**
- `auto` (default) — uses ONNX if available, falls back to PyTorch
- `onnx` — force ONNX Runtime (fails if not installed)
- `pytorch` — force PyTorch (always available with sentence-transformers)

**Example:**
```json
"SPATIAL_MEMORY_EMBEDDING_BACKEND": "onnx"
```

**Performance comparison:**

| Backend | Speed | Memory | Install |
|---------|-------|--------|---------|
| ONNX Runtime | 2-3x faster | 60% less | `pip install sentence-transformers[onnx]` |
| PyTorch | Baseline | Baseline | Included with sentence-transformers |

**Caveats:**
- Setting `onnx` when ONNX Runtime is not installed raises `ConfigurationError` at startup.
- This setting is ignored for OpenAI models (API-based, no local inference).

**Related:** `SPATIAL_MEMORY_EMBEDDING_MODEL`

---

## OpenAI Integration

### `SPATIAL_MEMORY_OPENAI_API_KEY`

| | |
|---|---|
| **Type** | `secret` |
| **Default** | `null` (not set) |
| **Since** | v0.1.0 |

Your OpenAI API key. Required only when using an OpenAI embedding model
(`SPATIAL_MEMORY_EMBEDDING_MODEL` starts with `openai:`).

**When to change:** When you want to use OpenAI's embedding API instead of local inference.

**Example:**
```json
"SPATIAL_MEMORY_OPENAI_API_KEY": "sk-proj-abc123..."
```

**Caveats:**
- Stored as `SecretStr` — masked in logs and debug output.
- If set but using a local model, the key is ignored.
- If using an `openai:*` model without this key, the server raises `ConfigurationError` at startup.

**Related:** `SPATIAL_MEMORY_EMBEDDING_MODEL`, `SPATIAL_MEMORY_OPENAI_EMBEDDING_MODEL`

---

### `SPATIAL_MEMORY_OPENAI_EMBEDDING_MODEL`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `text-embedding-3-small` |
| **Since** | v0.1.0 |

The specific OpenAI model to use when `SPATIAL_MEMORY_EMBEDDING_MODEL` is set to an `openai:*`
value. In practice, the model is extracted from the `openai:` prefix of `EMBEDDING_MODEL`, so
this setting is rarely needed independently.

**When to change:** Almost never. Use `SPATIAL_MEMORY_EMBEDDING_MODEL=openai:text-embedding-3-large` instead.

**Related:** `SPATIAL_MEMORY_OPENAI_API_KEY`, `SPATIAL_MEMORY_EMBEDDING_MODEL`

---

## Logging & Observability

### `SPATIAL_MEMORY_LOG_LEVEL`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `INFO` |
| **Since** | v0.1.0 |

Controls the verbosity of server logs. Logs are emitted to stderr (captured by MCP client).

**Values:**
- `DEBUG` — verbose output including SQL queries, embedding timings, cache hits/misses. Use for troubleshooting.
- `INFO` (default) — startup messages, configuration summary, warnings.
- `WARNING` — only warnings and errors. Use for production.
- `ERROR` — only errors.

**Example:**
```json
"SPATIAL_MEMORY_LOG_LEVEL": "DEBUG"
```

---

### `SPATIAL_MEMORY_LOG_FORMAT`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `text` |
| **Since** | v1.5.3 |

Controls the log output format.

**Values:**
- `text` (default) — human-readable format: `2026-02-05 10:30:00 INFO message`
- `json` — structured JSON per line. Use for log aggregation tools (Datadog, ELK, etc.)

---

### `SPATIAL_MEMORY_INCLUDE_REQUEST_META`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.3 |

When enabled, every MCP tool response includes a `_meta` object with `request_id` and optional
timing information. Useful for debugging and correlating logs with responses.

**When to change:** Enable for debugging performance issues or tracing request flow.

**Example response with `_meta`:**
```json
{
  "memories": [...],
  "_meta": {
    "request_id": "req_abc123",
    "duration_ms": 42,
    "agent_id": "agent_xyz"
  }
}
```

**Related:** `SPATIAL_MEMORY_INCLUDE_TIMING_BREAKDOWN`

---

### `SPATIAL_MEMORY_INCLUDE_TIMING_BREAKDOWN`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.3 |

When enabled (requires `SPATIAL_MEMORY_INCLUDE_REQUEST_META=true`), the `_meta` object includes
a `timing_ms` breakdown showing how long each phase of the operation took.

**When to change:** Enable to profile where time is spent (embedding, search, database write, etc.).

**Related:** `SPATIAL_MEMORY_INCLUDE_REQUEST_META`

---

### `SPATIAL_MEMORY_LOG_INCLUDE_TRACE_CONTEXT`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.3 |

Adds `[req=...][agent=...]` trace context to log messages for request/agent correlation.

**When to change:** Disable if the extra context clutters your logs.

---

## Spatial Operation Limits

### `SPATIAL_MEMORY_MAX_JOURNEY_STEPS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | 2–20 |
| **Since** | v0.1.0 |

Maximum number of interpolation steps in a `journey` operation. Each step uses SLERP to
interpolate between two memory embeddings, discovering concepts along the path.

**When to change:** Lower for faster journeys with fewer waypoints; the maximum of 20 is a
hard cap to prevent excessive database queries.

---

### `SPATIAL_MEMORY_MAX_WANDER_STEPS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | 1–20 |
| **Since** | v0.1.0 |

Maximum steps in a `wander` random walk through memory space.

**When to change:** Lower for shorter explorations. Each step involves a database query.

---

### `SPATIAL_MEMORY_MAX_VISUALIZE_MEMORIES`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `500` |
| **Since** | v0.1.0 |

Maximum number of memories included in a `visualize` operation. UMAP projection and similarity
calculations scale quadratically, so this caps compute cost.

**When to change:** Lower if visualizations are slow; increase if you need a broader view
and have the compute budget.

---

### `SPATIAL_MEMORY_REGIONS_MAX_MEMORIES`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Since** | v0.1.0 |

Maximum memories considered for HDBSCAN clustering in the `regions` tool. Limits the dataset
size fed to the clustering algorithm.

**When to change:** Increase if you have many memories and want broader cluster detection.
Decrease if regions is slow.

---

### `SPATIAL_MEMORY_VISUALIZE_SIMILARITY_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.7` |
| **Range** | 0.0–1.0 |
| **Since** | v0.1.0 |

Minimum cosine similarity between two memories to draw an edge in visualizations. Controls
how densely connected the visualization graph is.

**When to change:**
- Increase (e.g., `0.85`) for sparser, cleaner graphs showing only strong connections
- Decrease (e.g., `0.5`) for denser graphs showing weaker associations

---

## Vector Indexing

These settings control how LanceDB indexes vectors for fast approximate nearest-neighbor search.
Indexing only activates when the dataset exceeds `vector_index_threshold`.

### `SPATIAL_MEMORY_AUTO_CREATE_INDEXES`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v0.1.0 |

Whether to automatically create vector and FTS indexes when dataset size thresholds are met.

**When to change:** Disable if you want full control over index creation timing (e.g., during
bulk imports where you want to index once at the end).

---

### `SPATIAL_MEMORY_VECTOR_INDEX_THRESHOLD`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10000` |
| **Range** | >= 1000 |
| **Since** | v0.1.0 |

The number of rows at which a vector index is created. Below this threshold, LanceDB uses
brute-force (exact) search, which is faster for small datasets.

**When to change:**
- Lower (e.g., `5000`) if you want indexing to kick in earlier
- Higher if you prefer exact search for longer (more accurate, slower at scale)

---

### `SPATIAL_MEMORY_INDEX_TYPE`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `IVF_PQ` |
| **Since** | v1.6.0 |

The vector index algorithm. Each trades off between speed, accuracy, and memory usage.

**Values:**

| Type | Speed | Accuracy | Memory | Best For |
|------|-------|----------|--------|----------|
| `IVF_PQ` (default) | Fast | Good | Low | Most use cases, large datasets |
| `IVF_FLAT` | Medium | High | High | When accuracy matters more than speed |
| `HNSW_SQ` | Fastest | Good | Medium | Low-latency requirements |

**Caveats:** Changing the index type requires rebuilding the index. Existing indexes are not
automatically converted.

**Related:** `SPATIAL_MEMORY_INDEX_NPROBES`, `SPATIAL_MEMORY_HNSW_M`

---

### `SPATIAL_MEMORY_INDEX_NPROBES`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | >= 1 |
| **Since** | v0.1.0 |

Number of IVF partitions to search during a query. Only applies to `IVF_PQ` and `IVF_FLAT`
index types. Higher values search more partitions, improving recall at the cost of speed.

**Guidance:**
- `10` — fast, may miss some results
- `20` (default) — good balance
- `50+` — thorough, slower

**Related:** `SPATIAL_MEMORY_INDEX_TYPE`, `SPATIAL_MEMORY_INDEX_REFINE_FACTOR`

---

### `SPATIAL_MEMORY_INDEX_REFINE_FACTOR`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5` |
| **Range** | >= 1 |
| **Since** | v0.1.0 |

After approximate search returns candidates, the top `(refine_factor * limit)` results are
re-ranked using exact distance for improved accuracy.

**Guidance:**
- `1` — no refinement (fastest)
- `5` (default) — good accuracy
- `10+` — maximum accuracy at the cost of speed

---

### `SPATIAL_MEMORY_HNSW_M`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | 4–64 |
| **Since** | v1.6.0 |

Number of bi-directional connections per node in the HNSW graph. Only applies when
`INDEX_TYPE=HNSW_SQ`. Higher values improve search quality but increase index size.

**Guidance:**
- `8` — smaller index, faster build, lower quality
- `20` (default) — balanced
- `48` — high quality, large index

**Related:** `SPATIAL_MEMORY_HNSW_EF_CONSTRUCTION`, `SPATIAL_MEMORY_INDEX_TYPE`

---

### `SPATIAL_MEMORY_HNSW_EF_CONSTRUCTION`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `300` |
| **Range** | 100–1000 |
| **Since** | v1.6.0 |

Search width during HNSW index construction. Higher values produce a better quality graph
but take longer to build. Only affects build time, not query time.

**Guidance:**
- `100` — fast build, adequate quality
- `300` (default) — good quality
- `500+` — maximum quality, slow build

**Related:** `SPATIAL_MEMORY_HNSW_M`

---

### `SPATIAL_MEMORY_INDEX_WAIT_TIMEOUT_SECONDS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `30.0` |
| **Range** | >= 0.0 |
| **Since** | v1.5.3 |

Maximum time to wait for index creation to complete. Index creation happens asynchronously
in LanceDB; this is the timeout before giving up.

**When to change:** Increase for very large datasets where index creation takes longer.

---

## Full-Text Search

These settings control the full-text search (FTS) index used by `hybrid_recall`.

### `SPATIAL_MEMORY_ENABLE_FTS_INDEX`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v0.1.0 |

Enables the Tantivy-based full-text search index. Required for `hybrid_recall` to combine
vector and keyword search.

**When to change:** Disable if you only use vector search (`recall`) and want to save
storage space and index build time.

---

### `SPATIAL_MEMORY_FTS_STEM`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.6.0 |

Enables word stemming in FTS indexing. Stemming reduces words to their root form
(e.g., "running" -> "run", "decided" -> "decid") so that related word forms match.

**When to change:** Disable if stemming causes incorrect matches in your domain
(e.g., "postgres" stemming to "postgr" matching "postgraduate").

---

### `SPATIAL_MEMORY_FTS_REMOVE_STOP_WORDS`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.6.0 |

Removes common stop words ("the", "is", "at", etc.) from the FTS index. This improves
search relevance by ignoring noise words.

**When to change:** Disable if stop words are meaningful in your content (rare).

---

### `SPATIAL_MEMORY_FTS_LANGUAGE`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `English` |
| **Since** | v1.6.0 |

Language for FTS stemming and stop word lists. Uses Tantivy's language support.

**Values:** `English`, `French`, `German`, `Spanish`, `Italian`, `Portuguese`, `Russian`,
`Dutch`, `Swedish`, `Norwegian`, `Danish`, `Finnish`, `Hungarian`, `Romanian`, `Turkish`,
`Arabic`, `Hindi`, `Tamil`, `Greek`

**When to change:** If your memories are primarily in a non-English language.

---

## Hybrid Search

### `SPATIAL_MEMORY_HYBRID_DEFAULT_ALPHA`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.5` |
| **Range** | 0.0–1.0 |
| **Since** | v1.5.0 |

Default balance between vector similarity and keyword matching in `hybrid_recall`. Users
can override this per-call with the `alpha` parameter.

**Values:**
- `1.0` — pure vector search (semantic similarity only)
- `0.5` (default) — equal weight to vector and keyword
- `0.0` — pure keyword search (full-text match only)

**When to change:** Adjust based on your search patterns. If you search for specific terms
(function names, error codes), lower alpha gives better results.

---

### `SPATIAL_MEMORY_HYBRID_MIN_ALPHA`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.0` |
| **Range** | 0.0–1.0 |
| **Since** | v1.5.0 |

Minimum allowed alpha value. Restricts how far toward pure keyword search a user can go.

**When to change:** Set to `0.2` if you want to ensure vector similarity always has some weight.

---

### `SPATIAL_MEMORY_HYBRID_MAX_ALPHA`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `1.0` |
| **Range** | 0.0–1.0 |
| **Since** | v1.5.0 |

Maximum allowed alpha value. Restricts how far toward pure vector search a user can go.

---

## Clustering (HDBSCAN)

### `SPATIAL_MEMORY_REGIONS_MAX_MEMORIES`

See [Spatial Operation Limits](#spatial_memory_regions_max_memories).

---

## Visualization (UMAP)

### `SPATIAL_MEMORY_UMAP_N_NEIGHBORS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `15` |
| **Range** | >= 2 |
| **Since** | v0.1.0 |

UMAP neighborhood size. Controls the balance between local and global structure in
the 2D/3D projection.

**Guidance:**
- `5` — preserves very local structure (tight clusters, disconnected)
- `15` (default) — balanced
- `50` — preserves more global structure (connected, less cluster separation)

**When to change:** If visualizations show all points in a single blob, try lower values.
If clusters appear disconnected when they shouldn't be, try higher values.

---

### `SPATIAL_MEMORY_UMAP_MIN_DIST`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |
| **Range** | 0.0–1.0 |
| **Since** | v0.1.0 |

Minimum distance between points in UMAP projection. Controls how tightly packed the clusters are.

**Guidance:**
- `0.0` — points can overlap, tightest clusters
- `0.1` (default) — slight separation within clusters
- `0.5` — spread out, less distinct clusters
- `1.0` — maximum spread, uniform distribution

---

## Performance & Retry Logic

### `SPATIAL_MEMORY_MAX_RETRY_ATTEMPTS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `3` |
| **Range** | >= 1 |
| **Since** | v0.1.0 |

Maximum number of retry attempts for transient storage errors (file locks, I/O errors).
Uses exponential backoff between retries.

**Guidance:**
- `1` — no retries (fail fast)
- `3` (default) — handles most transient issues
- `5+` — for unreliable storage (USB drives, network-adjacent filesystems)

**Related:** `SPATIAL_MEMORY_RETRY_BACKOFF_SECONDS`

---

### `SPATIAL_MEMORY_RETRY_BACKOFF_SECONDS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.5` |
| **Range** | >= 0.1 |
| **Since** | v0.1.0 |

Initial backoff time between retry attempts. Doubles on each subsequent attempt
(exponential backoff: 0.5s, 1.0s, 2.0s...).

**Guidance:**
- `0.1` — aggressive retries (LAN/local storage)
- `0.5` (default) — balanced
- `2.0` — gentle retries (API-based backends)

**Related:** `SPATIAL_MEMORY_MAX_RETRY_ATTEMPTS`

---

### `SPATIAL_MEMORY_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Range** | >= 100 |
| **Since** | v0.1.0 |

Internal batch size for large operations (namespace listing, bulk reads). Controls how
many records are processed per database round-trip.

**When to change:** Lower if you're on memory-constrained systems. Higher if you have
large datasets and want faster bulk operations.

---

### `SPATIAL_MEMORY_COMPACTION_THRESHOLD`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10` |
| **Range** | >= 1 |
| **Since** | v1.6.0 |

Number of small file fragments that trigger automatic LanceDB compaction. LanceDB stores
data as immutable fragments; frequent writes create many small files. Compaction merges
them for better read performance.

**Guidance:**
- `5` — compact aggressively (frequent writes, slower writes)
- `10` (default) — balanced
- `20` — compact less often (fewer write-heavy workloads)

---

## Connection Pool

### `SPATIAL_MEMORY_CONNECTION_POOL_MAX_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10` |
| **Range** | 1–100 |
| **Since** | v1.5.3 |

Maximum number of LanceDB connections in the pool. Each namespace gets its own connection;
LRU eviction removes least-recently-used connections when the pool is full.

**When to change:** Increase if you use many namespaces concurrently and see connection
churn in debug logs. Decrease to save memory.

---

## Cross-Process Locking

### `SPATIAL_MEMORY_FILELOCK_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.6.0 |

Enables file-based locking to prevent data corruption when multiple MCP server instances
access the same storage directory.

**When to change:** Disable only if you are certain only one instance will ever access
the storage, and you want to eliminate lock overhead.

**Caveats:** Disabling file locking while running multiple instances will cause data corruption.

**Related:** `SPATIAL_MEMORY_FILELOCK_TIMEOUT`, `SPATIAL_MEMORY_FILELOCK_POLL_INTERVAL`

---

### `SPATIAL_MEMORY_FILELOCK_TIMEOUT`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `30.0` |
| **Range** | 1.0–300.0 |
| **Since** | v1.6.0 |

Maximum time in seconds to wait for acquiring a file lock before raising an error.

**When to change:** Increase if you see timeout errors during heavy concurrent usage.
Decrease if you want faster failure detection.

**Related:** `SPATIAL_MEMORY_FILELOCK_ENABLED`, `SPATIAL_MEMORY_FILELOCK_POLL_INTERVAL`

---

### `SPATIAL_MEMORY_FILELOCK_POLL_INTERVAL`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |
| **Range** | 0.01–1.0 |
| **Since** | v1.6.0 |

Interval in seconds between lock acquisition attempts while waiting for a lock.

**Guidance:**
- `0.01` — aggressive polling (lower latency, more CPU)
- `0.1` (default) — balanced
- `0.5` — gentle polling (lower CPU, higher latency)

**Related:** `SPATIAL_MEMORY_FILELOCK_ENABLED`, `SPATIAL_MEMORY_FILELOCK_TIMEOUT`

---

## Read Consistency

### `SPATIAL_MEMORY_READ_CONSISTENCY_INTERVAL_MS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `0` |
| **Range** | >= 0 |
| **Since** | v1.5.3 |

Interval in milliseconds for LanceDB read consistency checks. Controls how quickly reads
see recently written data.

**Values:**
- `0` (default) — strong consistency. Every read sees the latest write. Best for
  single-instance deployments.
- `100+` — eventual consistency. Reads may not see writes from the last N ms, but
  reads are faster. Only useful for high-throughput scenarios.

**When to change:** Almost never for single-developer use. The default (strong consistency)
is correct for MCP server workloads.

---

## Rate Limiting

### `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.3 |

Enables per-agent rate limiting. Each agent (identified by `_agent_id` parameter on tools)
gets its own rate limit bucket. Prevents a single runaway agent from monopolizing the server.

**When to change:** Disable if rate limiting interferes with legitimate high-throughput
operations (e.g., bulk imports via script).

---

### `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_RATE`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `25.0` |
| **Range** | 1.0–1000.0 |
| **Since** | v1.5.3 |

Maximum operations per second per agent. Uses a token bucket algorithm.

**Guidance:**
- `10` — conservative, limits burst behavior
- `25` (default) — generous for interactive use
- `100+` — effectively unlimited for most workflows

---

### `SPATIAL_MEMORY_RATE_LIMIT_MAX_TRACKED_AGENTS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | 1–1000 |
| **Since** | v1.5.3 |

Maximum number of distinct agents tracked for rate limiting. Uses LRU eviction — when the
limit is reached, the least recently active agent's bucket is removed.

**When to change:** Increase if you have many concurrent agents hitting the server.

---

### `SPATIAL_MEMORY_EMBEDDING_RATE_LIMIT`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `100.0` |
| **Range** | >= 1.0 |
| **Since** | v1.5.3 |

Maximum embedding operations per second. Applies globally (not per-agent). Protects against
overwhelming the embedding model or OpenAI API with too many concurrent requests.

**When to change:** Lower for OpenAI to stay within API rate limits. The default (100/s)
is fine for local models.

---

## Circuit Breaker

### `SPATIAL_MEMORY_CIRCUIT_BREAKER_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.3 |

Enables the circuit breaker pattern for the OpenAI embedding API. When consecutive failures
exceed the threshold, the circuit "opens" and requests fail fast instead of waiting for
timeouts. The circuit resets after a cooldown period.

**When to change:** Disable if you want every request to attempt the API call regardless
of recent failures.

**Caveats:** Only applies to OpenAI embedding calls. Local embedding models don't use the
circuit breaker.

---

### `SPATIAL_MEMORY_CIRCUIT_BREAKER_FAILURE_THRESHOLD`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5` |
| **Range** | 1–100 |
| **Since** | v1.5.3 |

Number of consecutive failures before the circuit opens (all subsequent requests fail immediately).

**Guidance:**
- `3` — sensitive, opens quickly
- `5` (default) — balanced
- `10` — tolerant, allows more failures before tripping

---

### `SPATIAL_MEMORY_CIRCUIT_BREAKER_RESET_TIMEOUT`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `60.0` |
| **Range** | 5.0–600.0 |
| **Since** | v1.5.3 |

Seconds to wait after the circuit opens before attempting a "half-open" probe request.
If the probe succeeds, the circuit closes (normal operation resumes). If it fails, the
circuit stays open for another timeout period.

**Guidance:**
- `30` — recover quickly from transient outages
- `60` (default) — balanced
- `300` — conservative, waits longer before retrying

---

## Response Caching

### `SPATIAL_MEMORY_RESPONSE_CACHE_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.3 |

Enables in-memory caching of responses for idempotent read operations (`recall`,
`hybrid_recall`, `stats`, `namespaces`, etc.). Repeated identical queries return
cached results instead of re-querying the database.

**When to change:** Disable if you need guaranteed fresh results on every call, or if
memory usage is a concern.

---

### `SPATIAL_MEMORY_RESPONSE_CACHE_MAX_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Range** | 100–100,000 |
| **Since** | v1.5.3 |

Maximum number of cached responses. Uses LRU eviction — the least recently accessed entry
is removed when the cache is full.

**Guidance:**
- `100` — minimal cache, low memory
- `1000` (default) — good hit rate for most workloads
- `10000` — large cache for heavy read workloads

---

### `SPATIAL_MEMORY_RESPONSE_CACHE_DEFAULT_TTL`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `60.0` |
| **Range** | 1.0–3600.0 |
| **Since** | v1.5.3 |

Default time-to-live in seconds for cached responses. After this time, the cached entry
expires and the next request re-queries the database.

**Guidance:**
- `10` — near real-time freshness
- `60` (default) — 1 minute staleness acceptable
- `300` — 5 minutes, good for slow-changing data

---

### `SPATIAL_MEMORY_RESPONSE_CACHE_REGIONS_TTL`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `300.0` |
| **Range** | 60.0–3600.0 |
| **Since** | v1.5.3 |

TTL in seconds specifically for `regions()` responses, which are computationally expensive
(HDBSCAN clustering). Cached longer than other responses by default.

**When to change:** Lower if you're actively adding memories and want clustering results
to update faster.

---

## Idempotency

### `SPATIAL_MEMORY_IDEMPOTENCY_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.3 |

Enables idempotency key support for write operations (`remember`, `remember_batch`).
When a client sends the same `idempotency_key` twice, the second call returns the
cached result instead of creating a duplicate memory.

**When to change:** Disable if you don't use idempotency keys and want to save the
overhead of tracking them.

---

### `SPATIAL_MEMORY_IDEMPOTENCY_KEY_TTL_HOURS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `24.0` |
| **Range** | 1.0–168.0 |
| **Since** | v1.5.3 |

How long (in hours) to remember idempotency keys. After this time, a previously-used
key can be reused and will create a new memory.

**Guidance:**
- `1` — short memory, keys expire quickly
- `24` (default) — one day
- `168` — one week (maximum)

---

## Memory TTL & Expiration

### `SPATIAL_MEMORY_ENABLE_MEMORY_EXPIRATION`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.6.0 |

Enables automatic deletion of memories after their TTL expires. This is a hard delete,
not decay — expired memories are permanently removed.

**When to change:** Enable if you want memories to automatically clean up after a fixed
period. This is different from decay (which reduces importance but keeps the memory).

**Related:** `SPATIAL_MEMORY_DEFAULT_MEMORY_TTL_DAYS`

---

### `SPATIAL_MEMORY_DEFAULT_MEMORY_TTL_DAYS`

| | |
|---|---|
| **Type** | `int` or `null` |
| **Default** | `null` (no expiration) |
| **Since** | v1.6.0 |

Default time-to-live in days for new memories. After this many days, the memory is
eligible for automatic deletion (if expiration is enabled).

**Values:**
- `null` (default) — memories never expire
- `7` — one week
- `30` — one month
- `365` — one year

**Caveats:** Requires `SPATIAL_MEMORY_ENABLE_MEMORY_EXPIRATION=true` to have any effect.

---

## Memory Lifecycle: Decay

These settings control the manual `decay` tool, which reduces importance of memories
based on time and access patterns.

### `SPATIAL_MEMORY_DECAY_DEFAULT_HALF_LIFE_DAYS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `30.0` |
| **Range** | 1.0–365.0 |
| **Since** | v1.6.0 |

Default half-life for exponential decay. After this many days without access, a memory's
importance drops to 50% of its original value.

**Guidance:**
- `7` — aggressive decay, only recent memories matter
- `30` (default) — balanced, ~monthly cycle
- `90` — slow decay, memories stay relevant for a quarter
- `365` — very slow, nearly permanent memories

**Example:** A memory with importance 1.0 and half-life 30 days:
- After 30 days: importance ≈ 0.50
- After 60 days: importance ≈ 0.25
- After 90 days: importance ≈ 0.125

**Related:** `SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION`, `SPATIAL_MEMORY_DECAY_MIN_IMPORTANCE_FLOOR`

---

### `SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `exponential` |
| **Since** | v1.8.0 |

The mathematical function used for the manual `decay` tool.

**Values:**
- `exponential` (default) — smooth decay: `2^(-t/half_life)`. Most natural, models the
  Ebbinghaus forgetting curve. Memory importance halves every `half_life` days.
- `linear` — constant-rate decay: reaches 0 at `2 * half_life` days. Memory importance
  decreases by a fixed amount per day.
- `step` — discrete drops at `half_life` intervals: `1.0 → 0.5 → 0.25 → 0.125`.
  Importance stays constant until the next step.

**Related:** `SPATIAL_MEMORY_AUTO_DECAY_FUNCTION` (same options, for auto-decay)

---

### `SPATIAL_MEMORY_DECAY_MIN_IMPORTANCE_FLOOR`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |
| **Range** | 0.0–0.5 |
| **Since** | v1.6.0 |

Minimum importance a memory can decay to. Prevents memories from becoming completely
invisible in search results.

**Guidance:**
- `0.0` — memories can decay to nothing (effectively hidden)
- `0.1` (default) — always findable, just deprioritized
- `0.3` — memories stay moderately visible even after long neglect

---

### `SPATIAL_MEMORY_DECAY_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `500` |
| **Range** | >= 100 |
| **Since** | v1.6.0 |

Number of memories processed per batch when running manual decay operations. Controls
memory usage and database round-trips during `decay` tool calls.

---

## Memory Lifecycle: Auto-Decay

Auto-decay automatically reduces memory importance during `recall` operations based on
time since last access. Unlike the manual `decay` tool, this runs transparently on every
search.

### `SPATIAL_MEMORY_AUTO_DECAY_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.7.0 |

Master switch for auto-decay. When enabled, every `recall` and `hybrid_recall` calculates
`effective_importance` for returned memories and re-ranks results accordingly.

**When to change:** Disable if you want static importance values (memories never fade).

**Values:**
- `true` (default) — memories that aren't accessed gradually become less prominent
- `false` — all memories maintain their original importance forever

---

### `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.7.0 |

Whether to write decayed importance values back to the database. When enabled, decay is
permanent (future searches use the decayed value). When disabled, decay is calculated
in-memory per-query and the stored importance is unchanged.

**Values:**
- `true` (default) — permanent decay, accumulates over time
- `false` — ephemeral decay, recalculated on each query. Good for experimentation.

**Related:** `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_BATCH_SIZE`, `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_FLUSH_INTERVAL_SECONDS`

---

### `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `100` |
| **Range** | 10–1000 |
| **Since** | v1.7.0 |

How many decay updates to accumulate before flushing to the database. The background
thread batches updates for efficiency.

**Guidance:**
- `10` — flush frequently (lower latency, more I/O)
- `100` (default) — balanced
- `500` — batch heavily (higher latency, less I/O)

---

### `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_FLUSH_INTERVAL_SECONDS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `5.0` |
| **Range** | 1.0–60.0 |
| **Since** | v1.7.0 |

Interval in seconds between background flush operations. The decay manager thread wakes
up at this interval to write pending decay updates to the database.

**Guidance:**
- `1.0` — near real-time persistence
- `5.0` (default) — balanced
- `30.0` — lazy persistence, fewer I/O operations

---

### `SPATIAL_MEMORY_AUTO_DECAY_MIN_CHANGE_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.01` |
| **Range** | 0.001–0.1 |
| **Since** | v1.7.0 |

Minimum importance change required before a decay update is queued for persistence.
Prevents writing to the database for negligible changes.

**Values:**
- `0.001` — persist even tiny changes (1/10th of a percent)
- `0.01` (default) — persist changes >= 1%
- `0.05` — only persist significant changes (>= 5%)

---

### `SPATIAL_MEMORY_AUTO_DECAY_MAX_QUEUE_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10000` |
| **Range** | 1000–100,000 |
| **Since** | v1.7.0 |

Maximum number of pending decay updates in the queue. If the queue fills up (because
the background thread can't keep up), new updates are dropped silently to prevent memory
exhaustion.

**When to change:** Increase if you see "decay queue full" warnings in logs during
heavy recall workloads.

---

### `SPATIAL_MEMORY_AUTO_DECAY_FUNCTION`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `exponential` |
| **Since** | v1.8.0 |

The decay function used during automatic decay in recall operations. Same options as
`SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION`.

**Values:**
- `exponential` (default) — smooth Ebbinghaus forgetting curve
- `linear` — constant-rate decay
- `step` — discrete importance drops

**Note:** In a future version, this may be merged with `SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION`
into a single `SPATIAL_MEMORY_DECAY_FUNCTION` setting.

**Related:** `SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION`

---

## Memory Lifecycle: Reinforcement

### `SPATIAL_MEMORY_REINFORCE_DEFAULT_BOOST`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |
| **Range** | 0.01–0.5 |
| **Since** | v1.6.0 |

Default amount to boost a memory's importance when using the `reinforce` tool. Applied
additively by default (importance += boost).

**Guidance:**
- `0.05` — gentle boost
- `0.1` (default) — noticeable boost
- `0.3` — strong boost

---

### `SPATIAL_MEMORY_REINFORCE_MAX_IMPORTANCE`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `1.0` |
| **Range** | 0.5–1.0 |
| **Since** | v1.6.0 |

Cap on importance after reinforcement. Prevents importance from exceeding this value
regardless of how many times a memory is reinforced.

---

## Memory Lifecycle: Extraction

### `SPATIAL_MEMORY_EXTRACT_MAX_TEXT_LENGTH`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `50000` |
| **Range** | >= 1000 |
| **Since** | v1.6.0 |

Maximum character length of text accepted by the `extract` tool. Text longer than this
is truncated before processing.

**Guidance:**
- `10000` — limit to short conversations
- `50000` (default) — handles most sessions
- `100000` — for very long transcripts

---

### `SPATIAL_MEMORY_EXTRACT_MAX_CANDIDATES`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `20` |
| **Range** | >= 1 |
| **Since** | v1.6.0 |

Maximum number of memory candidates the `extract` tool will generate from a single text.

**Guidance:**
- `5` — only the most prominent facts
- `20` (default) — thorough extraction
- `50` — exhaustive extraction

---

### `SPATIAL_MEMORY_EXTRACT_DEFAULT_IMPORTANCE`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.4` |
| **Range** | 0.0–1.0 |
| **Since** | v1.6.0 |

Default importance assigned to auto-extracted memories. Set lower than manual memories
(which default to 0.5) to indicate lower confidence.

---

### `SPATIAL_MEMORY_EXTRACT_DEFAULT_NAMESPACE`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `extracted` |
| **Since** | v1.6.0 |

Default namespace for auto-extracted memories. Keeps extractions separate from manually
created memories for easy identification.

---

## Memory Lifecycle: Consolidation

### `SPATIAL_MEMORY_CONSOLIDATE_MIN_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.7` |
| **Range** | 0.5–0.99 |
| **Since** | v1.6.0 |

Minimum similarity threshold for the `consolidate` tool to consider two memories as
potential duplicates for merging.

**Guidance:**
- `0.7` (default) — catches broadly similar memories
- `0.85` — only merge closely related memories
- `0.95` — only merge near-exact duplicates

**Caveats:** This is for post-hoc consolidation (merging existing memories). For
ingest-time deduplication (preventing duplicates during `remember`), see the cognitive
offloading `SPATIAL_MEMORY_DEDUP_VECTOR_THRESHOLD` setting.

---

### `SPATIAL_MEMORY_CONSOLIDATE_CONTENT_WEIGHT`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.3` |
| **Range** | 0.0–1.0 |
| **Since** | v1.6.0 |

Weight given to textual content overlap (vs. vector similarity) when computing
consolidation similarity. `1.0` means only content overlap matters; `0.0` means
only vector similarity matters.

**Guidance:**
- `0.0` — pure vector similarity
- `0.3` (default) — mostly vector, some text overlap
- `0.7` — mostly text overlap (for domain-specific content)

---

### `SPATIAL_MEMORY_CONSOLIDATE_MAX_BATCH`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Range** | >= 100 |
| **Since** | v1.6.0 |

Maximum memories per consolidation pass. Large namespaces are processed in chunks of
this size using streaming consolidation.

---

## Export Operations

### `SPATIAL_MEMORY_EXPORT_DEFAULT_FORMAT`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `parquet` |
| **Since** | v1.5.0 |

Default format for the `export_memories` tool when no format is specified.

**Values:**
- `parquet` (default) — binary columnar format. Smallest files, fastest I/O, includes vectors.
  Recommended for backups.
- `json` — human-readable. Good for inspection and interoperability.
- `csv` — tabular format. Vectors are excluded by default (see `CSV_INCLUDE_VECTORS`).
  Good for spreadsheet import.

---

### `SPATIAL_MEMORY_EXPORT_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `5000` |
| **Range** | >= 100 |
| **Since** | v1.5.0 |

Records per batch during streaming export. Controls memory usage for large exports.

---

### `SPATIAL_MEMORY_MAX_EXPORT_RECORDS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000000` |
| **Range** | 1000–10,000,000 |
| **Since** | v1.5.0 |

Maximum total records per export operation. Safety limit to prevent accidental full-database
exports from consuming disk space.

---

### `SPATIAL_MEMORY_CSV_INCLUDE_VECTORS`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.0 |

Whether to include embedding vectors in CSV exports. Vectors are arrays of 384+ floats,
which make CSV files extremely large and hard to read.

**When to change:** Enable only if you need vectors in CSV format for a specific tool
that doesn't support Parquet.

---

### `SPATIAL_MEMORY_EXPORT_ALLOWED_PATHS`

| | |
|---|---|
| **Type** | `list[str]` |
| **Default** | `["./exports", "./backups"]` |
| **Since** | v1.5.0 |

Directories where the `export_memories` tool is allowed to write files. Paths are
relative to `SPATIAL_MEMORY_MEMORY_PATH`. This is a security measure to prevent
arbitrary file writes.

**Example:** `["./exports", "./backups", "/shared/data"]`

**Related:** `SPATIAL_MEMORY_EXPORT_ALLOW_SYMLINKS`

---

### `SPATIAL_MEMORY_EXPORT_ALLOW_SYMLINKS`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.0 |

Whether to follow symlinks in export paths. Disabled by default to prevent symlink-based
path traversal attacks.

---

## Import Operations

### `SPATIAL_MEMORY_IMPORT_ALLOWED_PATHS`

| | |
|---|---|
| **Type** | `list[str]` |
| **Default** | `["./imports", "./backups"]` |
| **Since** | v1.5.0 |

Directories where the `import_memories` tool is allowed to read files. Security measure
to prevent arbitrary file reads.

---

### `SPATIAL_MEMORY_IMPORT_ALLOW_SYMLINKS`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.0 |

Whether to follow symlinks in import paths.

---

### `SPATIAL_MEMORY_IMPORT_MAX_FILE_SIZE_MB`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `100.0` |
| **Range** | 1.0–1000.0 |
| **Since** | v1.5.0 |

Maximum file size in megabytes for import operations. Safety limit to prevent loading
unexpectedly large files into memory.

---

### `SPATIAL_MEMORY_IMPORT_MAX_RECORDS`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `100000` |
| **Range** | 1000–10,000,000 |
| **Since** | v1.5.0 |

Maximum number of records per import operation.

---

### `SPATIAL_MEMORY_IMPORT_FAIL_FAST`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.0 |

Whether to stop import on the first validation error. When `false`, the import continues
and reports all errors at the end.

**Values:**
- `false` (default) — collect all errors, report at end
- `true` — stop on first error (faster feedback for debugging)

---

### `SPATIAL_MEMORY_IMPORT_VALIDATE_VECTORS`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.0 |

Whether to validate that imported vector dimensions match the configured embedding model.
Prevents importing vectors from a different model that would break search.

**When to change:** Disable only if you're importing vectors you know are correct but from
a non-standard source.

---

### `SPATIAL_MEMORY_IMPORT_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Range** | >= 100 |
| **Since** | v1.5.0 |

Records per batch during import. Controls memory usage and transaction size.

---

### `SPATIAL_MEMORY_IMPORT_DEDUPLICATE_DEFAULT`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Since** | v1.5.0 |

Whether to deduplicate imported records by default. When enabled, records that are
similar to existing memories (above the threshold) are skipped.

---

### `SPATIAL_MEMORY_IMPORT_DEDUP_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.95` |
| **Range** | 0.7–0.99 |
| **Since** | v1.5.0 |

Similarity threshold for import deduplication. Records with similarity above this
threshold to any existing memory are considered duplicates and skipped.

---

## Security & Destructive Operations

### `SPATIAL_MEMORY_DESTRUCTIVE_CONFIRM_THRESHOLD`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `100` |
| **Range** | >= 1 |
| **Since** | v1.5.0 |

Operations affecting more than this many records require explicit confirmation
(e.g., `delete_namespace` with `confirm=true`).

---

### `SPATIAL_MEMORY_DESTRUCTIVE_REQUIRE_NAMESPACE_CONFIRMATION`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `true` |
| **Since** | v1.5.0 |

Whether `delete_namespace` requires explicit confirmation. When `true`, the caller
must pass `confirm=true` to actually delete.

---

### `SPATIAL_MEMORY_NAMESPACE_BATCH_SIZE`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `1000` |
| **Range** | >= 100 |
| **Since** | v1.5.0 |

Batch size for namespace-level operations (listing, counting, renaming).

---

## Cognitive Offloading (Proposed v2.0)

> These parameters are proposed for the cognitive offloading feature (v2.0). They are
> documented here for completeness but are **not yet implemented** in the current version.

### `SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `false` |
| **Proposed** | v2.0 |

Master switch for cognitive offloading. When enabled, the server processes queued memories
from client hooks (PostToolUse, PreCompact, Stop), runs signal detection, and applies
smart gating (quality + dedup) on incoming memories.

**Values:**
- `false` (default) — traditional behavior, LLM decides what to save
- `true` — server-side intelligence + client hooks for automatic memory capture

---

### `SPATIAL_MEMORY_EXTRACTION_INTERVAL_MINUTES`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10` |
| **Proposed** | v2.0 |

How often (in minutes) the Layer 2 session extraction engine automatically extracts
memories from the in-memory tool interaction log.

**Guidance:**
- `5` — aggressive extraction (captures more, more duplicates)
- `10` (default) — balanced
- `30` — relaxed extraction (captures less, fewer duplicates)

---

### `SPATIAL_MEMORY_SIGNAL_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.3` |
| **Range** | 0.0–1.0 |
| **Proposed** | v2.0 |

Minimum signal score for content to be considered memory-worthy. The signal score
is computed from regex pattern matches against Tier 1/2/3 signal phrases.

**Guidance:**
- `0.1` — capture almost everything (noisy)
- `0.3` (default) — capture content with at least weak signals
- `0.6` — only capture content with strong Tier 1 signals

---

### `SPATIAL_MEMORY_QUEUE_POLL_INTERVAL_SECONDS`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `30` |
| **Proposed** | v2.0 |

How often (in seconds) the background queue processor checks for new memory files
written by client hooks. Follows the Maildir queue pattern (polling `pending-saves/new/`).

**Guidance:**
- `10` — process hook captures quickly (more CPU wake-ups)
- `30` (default) — balanced
- `60` — lazy processing (lower overhead, higher save latency)

---

### `SPATIAL_MEMORY_DEDUP_VECTOR_THRESHOLD`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.85` |
| **Range** | 0.0–1.0 |
| **Proposed** | v2.0 |

Vector similarity threshold for ingest-time deduplication. Memories with similarity above
this value to any existing memory in the same project are rejected as duplicates.

**Values:**
- `0.80` — aggressive dedup (may reject related but distinct memories)
- `0.85` (default) — balanced
- `0.90` — permissive (allows more near-duplicates through)

**Comparison with `CONSOLIDATE_MIN_THRESHOLD`:** This setting prevents duplicates at ingest
time. `CONSOLIDATE_MIN_THRESHOLD` (default 0.7) merges existing duplicates after the fact.
The ingest threshold is intentionally stricter to avoid false positive rejection.

---

### `SPATIAL_MEMORY_PROJECT`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `null` (auto-detect) |
| **Proposed** | v2.0 |

Explicit project identifier. When set, all memories are tagged with this project without
running the auto-detection cascade. This is priority 5 in the 7-level project detection
chain.

**When to change:** Set when running the MCP server for a specific project and auto-detection
isn't working (no git remote, unusual directory layout, etc.).

**Values:**
- Not set (default) — auto-detect from git remote URL, env vars, file paths
- `github.com/org/repo` — explicit project identity
- `local/my-project` — local project without remote

---

## Deprecated Parameters

The following parameters are defined in `config.py` but are **unused** — they have no
effect on server behavior. They will be removed in a future version.

| Parameter | Reason | Alternative |
|-----------|--------|-------------|
| `SPATIAL_MEMORY_DEFAULT_NAMESPACE` | Never read from config. Tool schemas already default to `"default"`. | Pass `namespace` on tool calls |
| `SPATIAL_MEMORY_DEFAULT_IMPORTANCE` | Never read from config. Tool schemas already default to `0.5`. | Pass `importance` on tool calls |
| `SPATIAL_MEMORY_MAX_BATCH_SIZE` | Never enforced. Batch tools have their own validation. | Tool-level `max_batch_size` param |
| `SPATIAL_MEMORY_MAX_RECALL_LIMIT` | Never enforced. Recall `limit` param already capped at 100 in tool schema. | Tool-level `limit` param |
| `SPATIAL_MEMORY_MIN_CLUSTER_SIZE` | Config value ignored; hardcoded in `SpatialConfig`. | Pass `min_cluster_size` on `regions` tool call |
| `SPATIAL_MEMORY_WARM_UP_ON_START` | Feature never implemented. | — |
| `SPATIAL_MEMORY_BATCH_RATE_LIMIT` | Never referenced in code. | `SPATIAL_MEMORY_EMBEDDING_RATE_LIMIT` |
| `SPATIAL_MEMORY_BACKPRESSURE_QUEUE_ENABLED` | Feature never implemented (marked "future"). | Replaced by cognitive offloading queue system |
| `SPATIAL_MEMORY_BACKPRESSURE_QUEUE_MAX_SIZE` | Related to above. | `SPATIAL_MEMORY_AUTO_DECAY_MAX_QUEUE_SIZE` |

**Stale `.env.example` entries** — The following entries appear in `.env.example` but do
not correspond to any field in `config.py`:
- `SPATIAL_MEMORY_DECAY_TIME_WEIGHT` — legacy, removed
- `SPATIAL_MEMORY_DECAY_ACCESS_WEIGHT` — legacy, removed
- `SPATIAL_MEMORY_DECAY_DAYS_THRESHOLD` — legacy, removed

---

## Recipes

### Minimal Production Setup

```json
{
  "env": {
    "SPATIAL_MEMORY_MEMORY_PATH": "/home/user/.spatial-memory",
    "SPATIAL_MEMORY_LOG_LEVEL": "WARNING"
  }
}
```

### OpenAI Embeddings

```json
{
  "env": {
    "SPATIAL_MEMORY_EMBEDDING_MODEL": "openai:text-embedding-3-small",
    "SPATIAL_MEMORY_OPENAI_API_KEY": "sk-..."
  }
}
```

### Maximum Quality (Slower)

```json
{
  "env": {
    "SPATIAL_MEMORY_EMBEDDING_MODEL": "all-mpnet-base-v2",
    "SPATIAL_MEMORY_INDEX_TYPE": "IVF_FLAT",
    "SPATIAL_MEMORY_INDEX_NPROBES": "50",
    "SPATIAL_MEMORY_INDEX_REFINE_FACTOR": "10"
  }
}
```

### Disable Auto-Decay

```json
{
  "env": {
    "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "false"
  }
}
```

### Read-Only Decay (Preview Without Persisting)

```json
{
  "env": {
    "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true",
    "SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED": "false"
  }
}
```

### Debug Performance

```json
{
  "env": {
    "SPATIAL_MEMORY_LOG_LEVEL": "DEBUG",
    "SPATIAL_MEMORY_INCLUDE_REQUEST_META": "true",
    "SPATIAL_MEMORY_INCLUDE_TIMING_BREAKDOWN": "true"
  }
}
```

### Multi-Language Memories

```json
{
  "env": {
    "SPATIAL_MEMORY_FTS_LANGUAGE": "French",
    "SPATIAL_MEMORY_FTS_STEM": "true"
  }
}
```

### High-Throughput Batch Processing

```json
{
  "env": {
    "SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_RATE": "200",
    "SPATIAL_MEMORY_EMBEDDING_RATE_LIMIT": "500",
    "SPATIAL_MEMORY_RESPONSE_CACHE_ENABLED": "false",
    "SPATIAL_MEMORY_BATCH_SIZE": "5000"
  }
}
```

### Restrictive Security (Shared Server)

```json
{
  "env": {
    "SPATIAL_MEMORY_EXPORT_ALLOW_SYMLINKS": "false",
    "SPATIAL_MEMORY_IMPORT_ALLOW_SYMLINKS": "false",
    "SPATIAL_MEMORY_IMPORT_MAX_FILE_SIZE_MB": "10",
    "SPATIAL_MEMORY_DESTRUCTIVE_REQUIRE_NAMESPACE_CONFIRMATION": "true"
  }
}
```

### Cognitive Offloading (v2.0, proposed)

```json
{
  "env": {
    "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true",
    "SPATIAL_MEMORY_SIGNAL_THRESHOLD": "0.3",
    "SPATIAL_MEMORY_DEDUP_VECTOR_THRESHOLD": "0.85",
    "SPATIAL_MEMORY_EXTRACTION_INTERVAL_MINUTES": "10",
    "SPATIAL_MEMORY_QUEUE_POLL_INTERVAL_SECONDS": "30"
  }
}
```
