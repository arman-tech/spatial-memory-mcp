# Prometheus Metrics

The Spatial Memory MCP Server includes optional Prometheus metrics for monitoring and observability.

## Installation

Metrics are **optional** and require the `prometheus_client` library. Install it with:

```bash
# Install with metrics support
pip install spatial-memory-mcp[metrics]

# Or install prometheus_client separately
pip install prometheus_client
```

The server will work perfectly fine without `prometheus_client` installed. When not available, all metrics calls become no-ops with zero performance overhead.

## Available Metrics

### Request Metrics

#### `spatial_memory_requests_total`
- **Type**: Counter
- **Labels**: `tool`, `status`
- **Description**: Total number of MCP tool requests
- **Example values**:
  - `{tool="recall", status="success"}` - Successful recall operations
  - `{tool="remember", status="error"}` - Failed remember operations

#### `spatial_memory_request_duration_seconds`
- **Type**: Histogram
- **Labels**: `tool`
- **Description**: Request duration in seconds
- **Buckets**: 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0 seconds

### Memory Metrics

#### `spatial_memory_memories_total`
- **Type**: Gauge
- **Labels**: `namespace`
- **Description**: Total number of memories per namespace
- **Usage**: Track memory growth over time

#### `spatial_memory_index_status`
- **Type**: Gauge
- **Labels**: `index_type`
- **Description**: Index status (1=exists, 0=missing)
- **Values**: `vector`, `fts`, `scalar`

### Search Metrics

#### `spatial_memory_search_similarity_score`
- **Type**: Histogram
- **Labels**: None
- **Description**: Distribution of search result similarity scores
- **Buckets**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
- **Usage**: Monitor search quality and relevance

### Embedding Metrics

#### `spatial_memory_embedding_latency_seconds`
- **Type**: Histogram
- **Labels**: `model`
- **Description**: Time to generate embeddings
- **Buckets**: 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0 seconds
- **Models**: `local`, `openai`

## Usage in Code

### Automatic Tool Metrics

All MCP tool calls are automatically instrumented. The server wraps each tool call with metrics recording:

```python
# This happens automatically for all tool calls
with record_request("recall", "success"):
    # ... handle tool ...
```

### Manual Metrics Recording

If you're extending the codebase, you can record custom metrics:

```python
from spatial_memory.core.metrics import (
    record_request,
    record_search_similarity,
    record_embedding_latency,
    update_memory_count,
    update_index_status,
    is_available,
)

# Check if metrics are available
if is_available():
    print("Metrics enabled!")

# Record a request with automatic timing
with record_request("custom_operation", "success"):
    # ... do work ...
    pass

# Record search similarity scores
record_search_similarity(0.85)

# Record embedding generation time
record_embedding_latency(0.234, model="openai")

# Update memory counts
update_memory_count("default", 1000)

# Update index status
update_index_status("vector", True)
```

### Error Handling

The metrics system automatically records errors:

```python
with record_request("my_tool", "success"):
    # If an exception is raised, status is automatically changed to "error"
    raise ValueError("Something went wrong")
# Metrics will show: spatial_memory_requests_total{tool="my_tool", status="error"}
```

## Exposing Metrics

To expose metrics to Prometheus, you need to add an HTTP endpoint. Here's an example:

```python
from prometheus_client import start_http_server

# Start metrics server on port 8000
start_http_server(8000)

# Now metrics are available at http://localhost:8000/metrics
```

For production deployments, consider:
1. Using a separate process for metrics export
2. Adding authentication to the metrics endpoint
3. Using a metrics gateway for short-lived processes

## Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'spatial-memory-mcp'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

## Example Queries

### Request Rate
```promql
# Requests per second by tool
rate(spatial_memory_requests_total[5m])

# Error rate
rate(spatial_memory_requests_total{status="error"}[5m])
```

### Latency
```promql
# 95th percentile request duration
histogram_quantile(0.95, rate(spatial_memory_request_duration_seconds_bucket[5m]))

# Average embedding latency by model
rate(spatial_memory_embedding_latency_seconds_sum[5m])
  / rate(spatial_memory_embedding_latency_seconds_count[5m])
```

### Memory Growth
```promql
# Total memories across all namespaces
sum(spatial_memory_memories_total)

# Memory growth rate
deriv(spatial_memory_memories_total[1h])
```

### Search Quality
```promql
# Average similarity score
rate(spatial_memory_search_similarity_score_sum[5m])
  / rate(spatial_memory_search_similarity_score_count[5m])

# Percentage of high-quality results (similarity > 0.8)
sum(rate(spatial_memory_search_similarity_score_bucket{le="1.0"}[5m]))
  - sum(rate(spatial_memory_search_similarity_score_bucket{le="0.8"}[5m]))
```

## Grafana Dashboard

A sample Grafana dashboard is available in `docs/grafana-dashboard.json` (coming soon).

Key panels:
- Request rate and error rate over time
- Request latency percentiles (p50, p95, p99)
- Memory count by namespace
- Search similarity distribution
- Embedding latency by model
- Index health status

## Performance Impact

When `prometheus_client` is **not** installed:
- Zero overhead - all metrics calls are no-ops
- No additional imports or dependencies
- Code compiles away at module load time

When `prometheus_client` **is** installed:
- Minimal overhead (~1-5 microseconds per metric update)
- Memory usage: ~100KB for metrics registry
- No impact on MCP protocol communication

## Architecture

The metrics module uses a graceful degradation pattern:

```python
# At module load time
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # All metrics become None

# At runtime
if PROMETHEUS_AVAILABLE:
    REQUESTS_TOTAL.labels(tool=tool, status=status).inc()
# If not available, this becomes a no-op
```

This design ensures:
1. No runtime import checks (checked once at module load)
2. No AttributeError exceptions when prometheus_client is missing
3. Clean separation between optional and required dependencies
4. Easy testing of both scenarios

## Testing

The test suite includes comprehensive tests for both scenarios:

```bash
# Test with prometheus_client installed
pip install prometheus_client
pytest tests/test_metrics.py

# Test without prometheus_client (mocked)
# Tests automatically verify no-op behavior
pytest tests/test_metrics.py::TestMetricsWithoutPrometheus
```

## Best Practices

1. **Don't log metrics failures**: If `prometheus_client` has issues, don't spam logs
2. **Use context managers**: `record_request()` automatically handles timing and errors
3. **Label cardinality**: Be careful with high-cardinality labels (namespaces are fine, user IDs are not)
4. **Histogram buckets**: Default buckets are tuned for typical MCP operations
5. **Gauge updates**: Update memory counts and index status during maintenance operations

## Troubleshooting

### Metrics not appearing
- Check if `prometheus_client` is installed: `pip list | grep prometheus`
- Check logs for "Prometheus metrics enabled" message
- Verify metrics endpoint is accessible: `curl http://localhost:8000/metrics`

### High memory usage
- Check label cardinality: `curl http://localhost:8000/metrics | wc -l`
- Consider reducing retention or scrape frequency
- Use recording rules in Prometheus for aggregations

### Incorrect values
- Verify labels match your queries
- Check if metrics are being reset (server restarts)
- Use `increase()` instead of `rate()` for counters over short windows
