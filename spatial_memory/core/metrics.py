"""Prometheus metrics for Spatial Memory MCP Server.

This module provides optional Prometheus metrics. If prometheus_client is not
installed, no-op stubs are provided so the code works without metrics.

Usage:
    from spatial_memory.core.metrics import (
        record_request,
        record_search_similarity,
        record_embedding_latency,
        update_memory_count,
    )

    with record_request("recall", "success"):
        # ... do work
        pass

    record_search_similarity(0.85)
    record_embedding_latency(0.234, model="arctic-m-long")
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter as CounterType
    from prometheus_client import Gauge as GaugeType
    from prometheus_client import Histogram as HistogramType

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    Gauge = None  # type: ignore

# Metrics definitions (only created if prometheus_client available)
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUESTS_TOTAL: CounterType = Counter(
        "spatial_memory_requests_total",
        "Total number of requests",
        ["tool", "status"],
    )
    REQUEST_DURATION: HistogramType = Histogram(
        "spatial_memory_request_duration_seconds",
        "Request duration in seconds",
        ["tool"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # Memory metrics
    MEMORIES_TOTAL: GaugeType = Gauge(
        "spatial_memory_memories_total",
        "Total number of memories",
        ["namespace"],
    )
    INDEX_STATUS: GaugeType = Gauge(
        "spatial_memory_index_status",
        "Index status (1=exists, 0=missing)",
        ["index_type"],
    )

    # Search metrics
    SEARCH_SIMILARITY: HistogramType = Histogram(
        "spatial_memory_search_similarity_score",
        "Search result similarity scores",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    )

    # Embedding metrics
    EMBEDDING_LATENCY: HistogramType = Histogram(
        "spatial_memory_embedding_latency_seconds",
        "Embedding generation latency in seconds",
        ["model"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )

    # v2.0 metrics (US-017 / spec §10.2)
    CAPTURE_SIZE_TOKENS: HistogramType = Histogram(
        "spatial_memory_capture_size_tokens",
        "Capture size in tokens, before/after the cap is applied",
        ["stage"],
        buckets=(128, 512, 2048, 4096, 8192, 16384, 65536),
    )
    CAPTURE_CAP_HIT_TOTAL: CounterType = Counter(
        "spatial_memory_capture_cap_hit_total",
        "Number of times the capture cap was hit, by tool",
        ["tool"],
    )
    RECALL_RESULT_COUNT: HistogramType = Histogram(
        "spatial_memory_recall_result_count",
        "Number of results returned by recall",
        buckets=(0, 1, 2, 3, 5, 10, 20, 50),
    )
    REINFORCED_MEMORY_RANK: HistogramType = Histogram(
        "spatial_memory_reinforced_memory_rank",
        "Rank of memories that get reinforced (1 = top hit)",
        buckets=(1, 2, 3, 5, 10, 20, 50, 100),
    )
    EMBEDDER_COLD_START_SECONDS: GaugeType = Gauge(
        "spatial_memory_embedder_cold_start_seconds",
        "Embedder cold-start latency in seconds (set once at first model load)",
    )
else:
    # No-op stubs when prometheus_client is not available
    REQUESTS_TOTAL = None  # type: ignore
    REQUEST_DURATION = None  # type: ignore
    MEMORIES_TOTAL = None  # type: ignore
    INDEX_STATUS = None  # type: ignore
    SEARCH_SIMILARITY = None  # type: ignore
    EMBEDDING_LATENCY = None  # type: ignore
    CAPTURE_SIZE_TOKENS = None  # type: ignore
    CAPTURE_CAP_HIT_TOTAL = None  # type: ignore
    RECALL_RESULT_COUNT = None  # type: ignore
    REINFORCED_MEMORY_RANK = None  # type: ignore
    EMBEDDER_COLD_START_SECONDS = None  # type: ignore


@contextmanager
def record_request(tool: str, status: str = "success") -> Generator[None, None, None]:
    """Context manager to record request metrics.

    Args:
        tool: Name of the tool being called.
        status: Status of the request (success, error, etc.).

    Yields:
        None

    Example:
        with record_request("recall", "success"):
            # ... do work
            pass
    """
    if not PROMETHEUS_AVAILABLE:
        yield
        return

    start = time.monotonic()
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        REQUESTS_TOTAL.labels(tool=tool, status=status).inc()
        REQUEST_DURATION.labels(tool=tool).observe(duration)


def record_search_similarity(similarity: float) -> None:
    """Record a search result similarity score.

    Updates the Prometheus histogram (when ``prometheus_client`` is installed)
    and unconditionally appends a JSONL journal line so the §10.4
    ``RecentObservations.search_similarity_*`` percentiles remain populated
    even when Prometheus is absent (US-021 / spec §10.1).

    Args:
        similarity: Similarity score between 0.0 and 1.0.

    Example:
        record_search_similarity(0.85)
    """
    if PROMETHEUS_AVAILABLE:
        SEARCH_SIMILARITY.observe(similarity)
    from spatial_memory.core.metrics_journal import journal_observation

    journal_observation("search_similarity_score", similarity, {})


def record_embedding_latency(duration: float, model: str = "local") -> None:
    """Record embedding generation latency.

    Updates the Prometheus histogram (when ``prometheus_client`` is installed)
    and unconditionally appends a JSONL journal line so the §10.4
    ``RecentObservations.embedding_latency_*_ms`` percentiles remain populated
    even when Prometheus is absent (US-021 / spec §10.1). The unit is seconds
    on both paths; §10.4 multiplies by 1000 at percentile-compute time.

    Args:
        duration: Time taken to generate embeddings in seconds.
        model: Model identifier (always ``"arctic-m-long"`` post-D1).

    Example:
        record_embedding_latency(0.234, model="arctic-m-long")
    """
    if PROMETHEUS_AVAILABLE:
        EMBEDDING_LATENCY.labels(model=model).observe(duration)
    from spatial_memory.core.metrics_journal import journal_observation

    journal_observation("embedding_latency_seconds", duration, {"model": model})


def update_memory_count(namespace: str, count: int) -> None:
    """Update memory count for a namespace.

    Args:
        namespace: The namespace identifier.
        count: Total number of memories in the namespace.

    Example:
        update_memory_count("default", 1000)
    """
    if PROMETHEUS_AVAILABLE:
        MEMORIES_TOTAL.labels(namespace=namespace).set(count)


def update_index_status(index_type: str, exists: bool) -> None:
    """Update index status.

    Args:
        index_type: Type of index (e.g., "vector", "fts", "scalar").
        exists: Whether the index exists.

    Example:
        update_index_status("vector", True)
    """
    if PROMETHEUS_AVAILABLE:
        INDEX_STATUS.labels(index_type=index_type).set(1 if exists else 0)


def is_available() -> bool:
    """Check if Prometheus metrics are available.

    Returns:
        True if prometheus_client is installed, False otherwise.

    Example:
        if is_available():
            print("Metrics are available")
    """
    return PROMETHEUS_AVAILABLE


# ---------------------------------------------------------------------------
# v2.0 helpers (US-017 / spec §10.2)
#
# Each helper updates the Prometheus metric (when prometheus_client is
# installed) AND appends one line to the always-on metrics journal. The
# journal write is the always-on observability path; Prometheus is optional.
# ---------------------------------------------------------------------------


def record_capture_size(stage: str, tokens: int) -> None:
    """Record the size in tokens of a capture, pre- or post-cap.

    Args:
        stage: ``"pre_cap"`` or ``"post_cap"``.
        tokens: Token count.
    """
    from spatial_memory.core.metrics_journal import journal_observation

    if PROMETHEUS_AVAILABLE:
        CAPTURE_SIZE_TOKENS.labels(stage=stage).observe(tokens)
    journal_observation("capture_size_tokens", tokens, {"stage": stage})


def record_capture_cap_hit(tool: str) -> None:
    """Record that the capture cap was hit by ``tool``."""
    from spatial_memory.core.metrics_journal import journal_observation

    if PROMETHEUS_AVAILABLE:
        CAPTURE_CAP_HIT_TOTAL.labels(tool=tool).inc()
    journal_observation("capture_cap_hit_total", 1, {"tool": tool})


def record_recall_result_count(count: int) -> None:
    """Record the number of results returned by a recall call."""
    from spatial_memory.core.metrics_journal import journal_observation

    if PROMETHEUS_AVAILABLE:
        RECALL_RESULT_COUNT.observe(count)
    journal_observation("recall_result_count", count, {})


def record_reinforced_memory_rank(rank: int) -> None:
    """Record the rank (1-indexed) of a memory that got reinforced."""
    from spatial_memory.core.metrics_journal import journal_observation

    if PROMETHEUS_AVAILABLE:
        REINFORCED_MEMORY_RANK.observe(rank)
    journal_observation("reinforced_memory_rank", rank, {})


def record_embedder_cold_start(seconds: float) -> None:
    """Record the embedder cold-start latency, in seconds."""
    from spatial_memory.core.metrics_journal import journal_observation

    if PROMETHEUS_AVAILABLE:
        EMBEDDER_COLD_START_SECONDS.set(seconds)
    journal_observation("embedder_cold_start_seconds", seconds, {})
