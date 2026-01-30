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
    record_embedding_latency(0.234, model="openai")
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
else:
    # No-op stubs when prometheus_client is not available
    REQUESTS_TOTAL = None  # type: ignore
    REQUEST_DURATION = None  # type: ignore
    MEMORIES_TOTAL = None  # type: ignore
    INDEX_STATUS = None  # type: ignore
    SEARCH_SIMILARITY = None  # type: ignore
    EMBEDDING_LATENCY = None  # type: ignore


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

    Args:
        similarity: Similarity score between 0.0 and 1.0.

    Example:
        record_search_similarity(0.85)
    """
    if PROMETHEUS_AVAILABLE:
        SEARCH_SIMILARITY.observe(similarity)


def record_embedding_latency(duration: float, model: str = "local") -> None:
    """Record embedding generation latency.

    Args:
        duration: Time taken to generate embeddings in seconds.
        model: Model identifier (e.g., "local", "openai").

    Example:
        record_embedding_latency(0.234, model="openai")
    """
    if PROMETHEUS_AVAILABLE:
        EMBEDDING_LATENCY.labels(model=model).observe(duration)


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
