"""Tests for Prometheus metrics module.

Tests both scenarios:
1. With prometheus_client installed (using real metrics)
2. Without prometheus_client installed (using no-op stubs)

Test order matters: Integration tests must run before "without prometheus" tests
because the latter modifies global module state.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest


class TestMetricsWithPrometheus:
    """Test metrics module when prometheus_client is available."""

    @pytest.fixture(autouse=True)
    def setup_method(self) -> None:
        """Ensure prometheus_client is available for these tests."""
        pytest.importorskip("prometheus_client")
        # Don't reload the module - reuse existing metrics
        # Prometheus metrics are singletons and reloading causes registration conflicts

    def test_prometheus_is_available(self) -> None:
        """Test that metrics detects prometheus_client correctly."""
        from spatial_memory.core.metrics import PROMETHEUS_AVAILABLE, is_available

        assert PROMETHEUS_AVAILABLE is True
        assert is_available() is True

    def test_metrics_objects_are_created(self) -> None:
        """Test that metric objects are created when prometheus is available."""
        from spatial_memory.core.metrics import (
            EMBEDDING_LATENCY,
            INDEX_STATUS,
            MEMORIES_TOTAL,
            REQUEST_DURATION,
            REQUESTS_TOTAL,
            SEARCH_SIMILARITY,
        )

        assert REQUESTS_TOTAL is not None
        assert REQUEST_DURATION is not None
        assert MEMORIES_TOTAL is not None
        assert INDEX_STATUS is not None
        assert SEARCH_SIMILARITY is not None
        assert EMBEDDING_LATENCY is not None

    def test_record_request_success(self) -> None:
        """Test recording a successful request."""
        from spatial_memory.core.metrics import REQUESTS_TOTAL, record_request

        # Get initial count
        initial_count = REQUESTS_TOTAL.labels(tool="recall", status="success")._value.get()

        # Record a request
        with record_request("recall", "success"):
            time.sleep(0.01)  # Simulate work

        # Check count increased
        final_count = REQUESTS_TOTAL.labels(tool="recall", status="success")._value.get()
        assert final_count == initial_count + 1

    def test_record_request_error(self) -> None:
        """Test recording a failed request."""
        from spatial_memory.core.metrics import REQUESTS_TOTAL, record_request

        # Get initial count
        initial_count = REQUESTS_TOTAL.labels(tool="forget", status="error")._value.get()

        # Record a failed request
        with pytest.raises(ValueError):
            with record_request("forget", "success"):
                raise ValueError("Test error")

        # Check error count increased (status should be changed to "error")
        final_count = REQUESTS_TOTAL.labels(tool="forget", status="error")._value.get()
        assert final_count == initial_count + 1

    def test_record_request_duration(self) -> None:
        """Test recording request duration."""
        from spatial_memory.core.metrics import REQUEST_DURATION, record_request

        # Get initial count
        metric = REQUEST_DURATION.labels(tool="test_remember_duration")
        # Histogram uses ._sum and ._count child metrics
        initial_sum = metric._sum.get()

        # Record a request with duration
        with record_request("test_remember_duration"):
            time.sleep(0.02)  # Simulate work

        # Check that a duration was recorded
        final_sum = metric._sum.get()
        assert final_sum > initial_sum

    def test_record_search_similarity(self) -> None:
        """Test recording search similarity scores."""
        from spatial_memory.core.metrics import SEARCH_SIMILARITY, record_search_similarity

        # Get initial sum
        initial_sum = SEARCH_SIMILARITY._sum.get()

        # Record similarity scores
        record_search_similarity(0.85)
        record_search_similarity(0.92)

        # Check sum increased
        final_sum = SEARCH_SIMILARITY._sum.get()
        assert final_sum > initial_sum

    def test_record_embedding_latency(self) -> None:
        """Test recording embedding latency."""
        from spatial_memory.core.metrics import EMBEDDING_LATENCY, record_embedding_latency

        # Get initial sums
        local_metric = EMBEDDING_LATENCY.labels(model="test_local")
        openai_metric = EMBEDDING_LATENCY.labels(model="test_openai")
        initial_local = local_metric._sum.get()
        initial_openai = openai_metric._sum.get()

        # Record latencies
        record_embedding_latency(0.123, model="test_local")
        record_embedding_latency(0.234, model="test_openai")
        record_embedding_latency(0.156, model="test_local")

        # Check sums increased
        assert local_metric._sum.get() > initial_local
        assert openai_metric._sum.get() > initial_openai

    def test_update_memory_count(self) -> None:
        """Test updating memory count gauge."""
        from spatial_memory.core.metrics import MEMORIES_TOTAL, update_memory_count

        # Update memory counts
        update_memory_count("default", 100)
        update_memory_count("work", 50)

        # Check gauge values
        assert MEMORIES_TOTAL.labels(namespace="default")._value.get() == 100
        assert MEMORIES_TOTAL.labels(namespace="work")._value.get() == 50

        # Update again
        update_memory_count("default", 150)
        assert MEMORIES_TOTAL.labels(namespace="default")._value.get() == 150

    def test_update_index_status(self) -> None:
        """Test updating index status gauge."""
        from spatial_memory.core.metrics import INDEX_STATUS, update_index_status

        # Update index statuses
        update_index_status("vector", True)
        update_index_status("fts", False)
        update_index_status("scalar", True)

        # Check gauge values
        assert INDEX_STATUS.labels(index_type="vector")._value.get() == 1
        assert INDEX_STATUS.labels(index_type="fts")._value.get() == 0
        assert INDEX_STATUS.labels(index_type="scalar")._value.get() == 1


class TestMetricsIntegration:
    """Integration tests for metrics module.

    These tests require prometheus_client to be installed and the metrics
    module to have been loaded with prometheus available.

    Note: Due to Prometheus registry being a global singleton, if the metrics
    module was first loaded without prometheus_client, we cannot reload it
    without registry conflicts. These tests must run before TestMetricsWithoutPrometheus.
    """

    @pytest.fixture(autouse=True)
    def ensure_prometheus_loaded(self) -> None:
        """Ensure metrics module is loaded with prometheus available."""
        pytest.importorskip("prometheus_client")

        # If the module was previously loaded without prometheus, skip
        # (we can't reload due to Prometheus registry conflicts)
        if "spatial_memory.core.metrics" in sys.modules:
            import spatial_memory.core.metrics as metrics_module

            if not metrics_module.PROMETHEUS_AVAILABLE:
                pytest.skip(
                    "Metrics module was loaded without prometheus_client. "
                    "Run integration tests separately: pytest tests/test_metrics.py::TestMetricsIntegration"
                )

    def test_metrics_work_across_modules(self) -> None:
        """Test that metrics can be imported and used across different modules."""
        pytest.importorskip("prometheus_client")

        # Import in different ways (don't reload to avoid registry conflicts)
        from spatial_memory.core import metrics
        from spatial_memory.core.metrics import (
            record_embedding_latency,
            record_request,
            record_search_similarity,
            update_index_status,
            update_memory_count,
        )

        # All should work
        with record_request("test_module"):
            pass
        record_search_similarity(0.5)
        record_embedding_latency(0.1)
        update_memory_count("test_module", 1)
        update_index_status("test_module", True)

        # Should be able to check availability
        assert metrics.is_available() is True

    def test_metrics_context_manager_timing(self) -> None:
        """Test that context manager accurately measures duration."""
        pytest.importorskip("prometheus_client")

        from spatial_memory.core.metrics import REQUEST_DURATION, record_request

        # Get initial sum (Histogram stores sum of all observed values)
        metric = REQUEST_DURATION.labels(tool="test_timing")
        initial_sum = metric._sum.get()

        # Record a request with known duration
        sleep_time = 0.05
        with record_request("test_timing"):
            time.sleep(sleep_time)

        # Check that duration was recorded (sum should have increased)
        final_sum = metric._sum.get()
        duration_recorded = final_sum - initial_sum

        # Verify a duration was recorded and it's reasonably close to sleep time
        # Allow 30ms tolerance for timing variability
        assert duration_recorded > 0
        assert abs(duration_recorded - sleep_time) < 0.03

    def test_multiple_tools_tracked_independently(self) -> None:
        """Test that different tools are tracked independently."""
        pytest.importorskip("prometheus_client")

        from spatial_memory.core.metrics import REQUESTS_TOTAL, record_request

        # Get initial counts for different tools
        recall_initial = REQUESTS_TOTAL.labels(tool="test_recall", status="success")._value.get()
        remember_initial = REQUESTS_TOTAL.labels(
            tool="test_remember", status="success"
        )._value.get()

        # Record requests for different tools
        with record_request("test_recall"):
            pass
        with record_request("test_recall"):
            pass
        with record_request("test_remember"):
            pass

        # Check counts
        recall_final = REQUESTS_TOTAL.labels(tool="test_recall", status="success")._value.get()
        remember_final = REQUESTS_TOTAL.labels(
            tool="test_remember", status="success"
        )._value.get()

        assert recall_final == recall_initial + 2
        assert remember_final == remember_initial + 1


class TestMetricsWithoutPrometheus:
    """Test metrics module when prometheus_client is NOT available.

    These tests verify that the metrics module degrades gracefully when
    prometheus_client is not installed. We mock the import to simulate
    the unavailable scenario.

    IMPORTANT: These tests MUST run last because they modify global module state
    that cannot be cleanly restored (Prometheus registry conflicts).
    """

    @pytest.fixture(autouse=True)
    def mock_prometheus_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Mock prometheus_client as unavailable."""
        import builtins

        # Block prometheus_client import
        import_orig = builtins.__import__

        def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "prometheus_client":
                raise ImportError("prometheus_client not installed")
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_mock)

        # Force reload of metrics module without prometheus
        if "spatial_memory.core.metrics" in sys.modules:
            del sys.modules["spatial_memory.core.metrics"]

        yield

        # Note: We intentionally don't restore the module here because
        # Prometheus metrics are singletons. The module will be reimported
        # by subsequent tests that need it.

    def test_prometheus_is_not_available(self) -> None:
        """Test that metrics detects missing prometheus_client correctly."""
        from spatial_memory.core.metrics import PROMETHEUS_AVAILABLE, is_available

        assert PROMETHEUS_AVAILABLE is False
        assert is_available() is False

    def test_metrics_objects_are_none(self) -> None:
        """Test that metric objects are None when prometheus is unavailable."""
        from spatial_memory.core.metrics import (
            EMBEDDING_LATENCY,
            INDEX_STATUS,
            MEMORIES_TOTAL,
            REQUEST_DURATION,
            REQUESTS_TOTAL,
            SEARCH_SIMILARITY,
        )

        assert REQUESTS_TOTAL is None
        assert REQUEST_DURATION is None
        assert MEMORIES_TOTAL is None
        assert INDEX_STATUS is None
        assert SEARCH_SIMILARITY is None
        assert EMBEDDING_LATENCY is None

    def test_record_request_no_op(self) -> None:
        """Test that record_request is a no-op without prometheus."""
        from spatial_memory.core.metrics import record_request

        # Should not raise any errors
        with record_request("recall", "success"):
            time.sleep(0.01)

    def test_record_request_no_op_with_error(self) -> None:
        """Test that record_request handles errors gracefully without prometheus."""
        from spatial_memory.core.metrics import record_request

        # Should still propagate the exception
        with pytest.raises(ValueError):
            with record_request("forget", "success"):
                raise ValueError("Test error")

    def test_record_search_similarity_no_op(self) -> None:
        """Test that record_search_similarity is a no-op without prometheus."""
        from spatial_memory.core.metrics import record_search_similarity

        # Should not raise any errors
        record_search_similarity(0.85)
        record_search_similarity(0.92)

    def test_record_embedding_latency_no_op(self) -> None:
        """Test that record_embedding_latency is a no-op without prometheus."""
        from spatial_memory.core.metrics import record_embedding_latency

        # Should not raise any errors
        record_embedding_latency(0.123, model="local")
        record_embedding_latency(0.234, model="openai")
        record_embedding_latency(0.156)

    def test_update_memory_count_no_op(self) -> None:
        """Test that update_memory_count is a no-op without prometheus."""
        from spatial_memory.core.metrics import update_memory_count

        # Should not raise any errors
        update_memory_count("default", 100)
        update_memory_count("work", 50)

    def test_update_index_status_no_op(self) -> None:
        """Test that update_index_status is a no-op without prometheus."""
        from spatial_memory.core.metrics import update_index_status

        # Should not raise any errors
        update_index_status("vector", True)
        update_index_status("fts", False)
        update_index_status("scalar", True)
