# Metrics

Spatial Memory v2.0 ships **two** metric paths:

1. **JSONL metrics journal** — always-on, no dependencies, the canonical
   observability sink. Every metric helper appends one line per observation.
2. **Prometheus** — optional, opt-in via the `prometheus_client` dependency.
   When installed, the same helpers also update Prometheus counters /
   histograms / gauges. When absent, the calls degrade to journal-only
   no-ops with zero overhead.

Operators reading metrics back without Prometheus should use either the
JSONL files directly or `stats(include_observations=True)`, which exposes
rolling percentiles computed from the journal.

---

## Journal (always-on)

**Location**: `<memory_path>/metrics/<YYYY-MM-DD>-<pid>-<startepoch>.jsonl`,
where `<memory_path>` defaults to `.spatial-memory/` (override with
`SPATIAL_MEMORY_MEMORY_PATH`). A typical path looks like
`.spatial-memory/metrics/2026-04-29-12345-1714398732.jsonl`.

**Filename scheme**: per-process, day-stamped. The combination of date,
PID, and process start epoch makes filenames unique even across rapid
restarts within the same minute, so concurrent processes never need to
coordinate locks.

**Retention**: 7 days. `prune_journal()` removes files whose date stem
is older than the cutoff; it is invoked from the maintenance path and
fails open on `OSError` so a misbehaving filesystem cannot break a
request path.

**Atomic-append guarantee**: each line is written in a single `write()`
call sized below `PIPE_BUF` (4096 bytes), so multi-thread interleaving
inside a single process stays line-clean.

**Read path** (no Prometheus required):
```python
# MCP / CLI tool: returns p50/p95/p99 percentiles over the recent journal
stats(include_observations=True)
```

---

## v2.0 Metrics (§10.2)

The five v2.0 metrics each inform a deferred design decision documented
in `internal-docs/v2-architectural-decisions.md`. Without that mapping the metrics rot,
because the *purpose* (and therefore the read query and threshold) lives
in the design note rather than in the code that emits them. Keep this
table in sync if you add a new v2.0 metric.

| Metric | Type | Labels | Informs decision |
|--------|------|--------|------------------|
| `spatial_memory_capture_size_tokens` | Histogram | `stage` (`pre_cap` / `post_cap`) | **D2** — chunking pipeline |
| `spatial_memory_capture_cap_hit_total` | Counter | `tool` | **D2** — chunking pipeline |
| `spatial_memory_recall_result_count` | Histogram | — | **D2 / D3** — HyDE query expansion |
| `spatial_memory_reinforced_memory_rank` | Histogram | — | **D2** — cross-encoder reranker |
| `spatial_memory_embedder_cold_start_seconds` | Gauge | — | **D1** — preload-at-startup contract |

### `capture_size_tokens` — D2 (chunking pipeline)

Observed both before and after the 200K-byte safety cap is applied
(see §8.1 / §8.2 / US-018 / US-019). The `pre_cap` / `post_cap` paired
samples answer: "how many captures are getting truncated, and by how
much?" The answer determines whether D2's chunking pipeline gives the
operator a real recall improvement or whether they were already inside
the cap.

```python
from spatial_memory.core.metrics import record_capture_size
record_capture_size("pre_cap", tokens=12480)
record_capture_size("post_cap", tokens=8192)
```

### `capture_cap_hit_total` — D2 (chunking pipeline)

Counter incremented every time the cap actually truncates a capture,
labelled by emitting tool (`content_extractor`, `transcript_extractor`,
`embed_passage`, …). Used together with `capture_size_tokens` to size
the chunker's window before D2 ships.

### `recall_result_count` — D2 / D3 (HyDE)

Histogram of how many memories `recall` / `hybrid_recall` actually
returns per call. A long tail of single-result and zero-result calls is
the empirical justification for D3's HyDE query expansion: if recall is
already returning healthy result counts, HyDE costs latency for no win.

### `reinforced_memory_rank` — D2 (cross-encoder reranker)

Histogram of the rank (1-indexed) at which a memory was sitting when it
got reinforced. If reinforcement is concentrated at rank 1–3, the existing
vector + BM25 hybrid is already ranking well; if reinforcement happens
deep in the result list, D2's cross-encoder reranker has a real lift to
deliver.

### `embedder_cold_start_seconds` — D1 (preload-at-startup)

Gauge set exactly once per process, on first model load. D1's preload
contract says cold start belongs to startup, not the first request — this
metric is the verification: a non-zero gauge from a request-path
percentile means we regressed against the contract.

---

## Recording from your own code

```python
from spatial_memory.core.metrics import (
    record_capture_size,
    record_capture_cap_hit,
    record_recall_result_count,
    record_reinforced_memory_rank,
    record_embedder_cold_start,
    record_search_similarity,
    record_embedding_latency,
)

record_capture_size("pre_cap", 4096)
record_capture_cap_hit("content_extractor")
record_recall_result_count(7)
record_reinforced_memory_rank(2)
record_embedder_cold_start(3.42)
record_search_similarity(0.91)
record_embedding_latency(0.234, model="arctic-m-long")
```

Every helper writes to the journal unconditionally and to Prometheus when
the optional dependency is installed.

---

## Pre-existing Prometheus metrics

These predate v2.0 and are unchanged:

- `spatial_memory_requests_total{tool, status}` — Counter
- `spatial_memory_request_duration_seconds{tool}` — Histogram
- `spatial_memory_memories_total{namespace}` — Gauge
- `spatial_memory_index_status{index_type}` — Gauge
- `spatial_memory_search_similarity_score` — Histogram (also journaled in v2.0)
- `spatial_memory_embedding_latency_seconds{model}` — Histogram (also journaled in v2.0; `model` is `arctic-m-long` post-D1)

---

## Prometheus setup (optional)

```bash
pip install spatial-memory-mcp[metrics]
# or:
pip install prometheus_client
```

Expose the metrics endpoint:

```python
from prometheus_client import start_http_server
start_http_server(8000)  # http://localhost:8000/metrics
```

Scrape config:

```yaml
scrape_configs:
  - job_name: 'spatial-memory-mcp'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

### Useful queries

```promql
# Capture truncation rate by tool
rate(spatial_memory_capture_cap_hit_total[5m])

# Distribution of capture sizes pre vs post cap
histogram_quantile(0.95,
  rate(spatial_memory_capture_size_tokens_bucket{stage="pre_cap"}[5m]))

# Recall result-count tail (long tail ⇒ D3 HyDE candidate)
histogram_quantile(0.10, rate(spatial_memory_recall_result_count_bucket[5m]))

# Reinforcement rank distribution (deep ranks ⇒ D2 reranker candidate)
histogram_quantile(0.95, rate(spatial_memory_reinforced_memory_rank_bucket[5m]))

# Cold-start gauge — must be 0 on the request path (preload-at-startup, D1)
spatial_memory_embedder_cold_start_seconds
```

---

## Performance impact

When `prometheus_client` is **not** installed, only the journal write
fires — a single appended line per observation, ≤ 4096 bytes, no lock
acquisition. The fail-open `try/except OSError` swallows transient
filesystem errors so a full disk cannot break a request path.

When `prometheus_client` **is** installed, both paths fire. Prometheus
adds ~1–5 µs per call.
