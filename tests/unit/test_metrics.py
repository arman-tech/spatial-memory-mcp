"""Unit tests for spatial_memory.core.metrics (US-017 / spec §10.2).

Pin the public contract for the five v2.0 metrics + helpers before the
implementation lands. Each helper has two responsibilities:

1. Update the Prometheus metric (when ``prometheus_client`` is installed).
2. Append a JSON line to the always-on metrics journal — *unconditionally*,
   regardless of whether Prometheus is present.

The journal write is the always-on observability path; Prometheus is optional
bookkeeping on top. These tests encode that contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _patch_journal(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, dict]]:
    """Capture every journal_observation call across the module under test.

    The metrics module imports ``journal_observation`` lazily inside each
    helper to avoid a hard import-time dependency on the journal facility,
    so we patch the journal module's symbol and rely on lazy resolution.
    """
    captured: list[tuple[str, float, dict]] = []

    def _fake(metric: str, value: float, labels: dict) -> None:
        captured.append((metric, value, labels))

    monkeypatch.setattr("spatial_memory.core.metrics_journal.journal_observation", _fake)
    return captured


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — module exports the five metric symbols + 5 helpers
# ---------------------------------------------------------------------------


def test_module_exports_five_new_metrics() -> None:
    from spatial_memory.core import metrics

    assert hasattr(metrics, "CAPTURE_SIZE_TOKENS")
    assert hasattr(metrics, "CAPTURE_CAP_HIT_TOTAL")
    assert hasattr(metrics, "RECALL_RESULT_COUNT")
    assert hasattr(metrics, "REINFORCED_MEMORY_RANK")
    assert hasattr(metrics, "EMBEDDER_COLD_START_SECONDS")


def test_module_exports_five_new_helpers() -> None:
    from spatial_memory.core import metrics

    assert callable(metrics.record_capture_size)
    assert callable(metrics.record_capture_cap_hit)
    assert callable(metrics.record_recall_result_count)
    assert callable(metrics.record_reinforced_memory_rank)
    assert callable(metrics.record_embedder_cold_start)


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — Prometheus metric movement (when available)
# ---------------------------------------------------------------------------


def test_record_capture_size_moves_histogram() -> None:
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent — see prometheus-absent tests below")

    before = (
        REGISTRY.get_sample_value("spatial_memory_capture_size_tokens_count", {"stage": "pre_cap"})
        or 0
    )
    metrics.record_capture_size("pre_cap", 4096)
    after = (
        REGISTRY.get_sample_value("spatial_memory_capture_size_tokens_count", {"stage": "pre_cap"})
        or 0
    )
    assert after == before + 1


def test_record_capture_cap_hit_increments_counter() -> None:
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    before = (
        REGISTRY.get_sample_value("spatial_memory_capture_cap_hit_total", {"tool": "remember"}) or 0
    )
    metrics.record_capture_cap_hit("remember")
    after = (
        REGISTRY.get_sample_value("spatial_memory_capture_cap_hit_total", {"tool": "remember"}) or 0
    )
    assert after == before + 1


def test_record_recall_result_count_moves_histogram() -> None:
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    before = REGISTRY.get_sample_value("spatial_memory_recall_result_count_count") or 0
    metrics.record_recall_result_count(5)
    after = REGISTRY.get_sample_value("spatial_memory_recall_result_count_count") or 0
    assert after == before + 1


def test_record_reinforced_memory_rank_moves_histogram() -> None:
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    before = REGISTRY.get_sample_value("spatial_memory_reinforced_memory_rank_count") or 0
    metrics.record_reinforced_memory_rank(3)
    after = REGISTRY.get_sample_value("spatial_memory_reinforced_memory_rank_count") or 0
    assert after == before + 1


def test_record_embedder_cold_start_sets_gauge() -> None:
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    metrics.record_embedder_cold_start(2.5)
    value = REGISTRY.get_sample_value("spatial_memory_embedder_cold_start_seconds")
    assert value == 2.5


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — Each helper calls journal_observation unconditionally
# ---------------------------------------------------------------------------


def test_record_capture_size_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    metrics.record_capture_size("pre_cap", 8192)
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "capture_size_tokens"
    assert value == 8192
    assert labels == {"stage": "pre_cap"}


def test_record_capture_cap_hit_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    metrics.record_capture_cap_hit("recall")
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "capture_cap_hit_total"
    assert value == 1
    assert labels == {"tool": "recall"}


def test_record_recall_result_count_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    metrics.record_recall_result_count(10)
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "recall_result_count"
    assert value == 10
    assert labels == {}


def test_record_reinforced_memory_rank_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    metrics.record_reinforced_memory_rank(7)
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "reinforced_memory_rank"
    assert value == 7
    assert labels == {}


def test_record_embedder_cold_start_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    metrics.record_embedder_cold_start(1.234)
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "embedder_cold_start_seconds"
    assert value == 1.234
    assert labels == {}


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — Helpers do not raise when prometheus_client absent;
# journal still fires
# ---------------------------------------------------------------------------


def test_helpers_do_not_raise_when_prometheus_absent(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    from spatial_memory.core import metrics

    with patch.object(metrics, "PROMETHEUS_AVAILABLE", False):
        metrics.record_capture_size("post_cap", 2048)
        metrics.record_capture_cap_hit("recall")
        metrics.record_recall_result_count(0)
        metrics.record_reinforced_memory_rank(1)
        metrics.record_embedder_cold_start(0.5)

    assert len(_patch_journal) == 5


# ---------------------------------------------------------------------------
# Histogram bucket fidelity — pin the buckets specified in spec §10.2
# ---------------------------------------------------------------------------


def test_capture_size_tokens_buckets_match_spec() -> None:
    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    expected_upper_bounds = {128.0, 512.0, 2048.0, 4096.0, 8192.0, 16384.0, 65536.0}
    actual = set(metrics.CAPTURE_SIZE_TOKENS._upper_bounds)
    assert expected_upper_bounds.issubset(actual), (
        f"Spec buckets {expected_upper_bounds} not found in {actual}"
    )


def test_recall_result_count_buckets_match_spec() -> None:
    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    expected = {0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0}
    actual = set(metrics.RECALL_RESULT_COUNT._upper_bounds)
    assert expected.issubset(actual), f"Spec buckets {expected} not found in {actual}"


def test_reinforced_memory_rank_buckets_match_spec() -> None:
    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent")

    expected = {1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0}
    actual = set(metrics.REINFORCED_MEMORY_RANK._upper_bounds)
    assert expected.issubset(actual), f"Spec buckets {expected} not found in {actual}"


# ---------------------------------------------------------------------------
# Existing helpers must still exist and remain unchanged
# ---------------------------------------------------------------------------


def test_existing_helpers_still_exported() -> None:
    """Regression: US-017 must not remove the v1 helpers."""
    from spatial_memory.core import metrics

    assert callable(metrics.record_request)
    assert callable(metrics.record_search_similarity)
    assert callable(metrics.record_embedding_latency)
    assert callable(metrics.update_memory_count)
    assert callable(metrics.update_index_status)
    assert callable(metrics.is_available)


# ---------------------------------------------------------------------------
# US-021 / §10.1 — record_search_similarity + record_embedding_latency
# now also write to the JSONL journal so the §10.4 RecentObservations block
# is non-empty when prometheus_client is absent.
# ---------------------------------------------------------------------------


def test_record_search_similarity_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    """§10.1: record_search_similarity must journal alongside Prometheus."""
    from spatial_memory.core import metrics

    metrics.record_search_similarity(0.85)
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "search_similarity_score"
    assert value == pytest.approx(0.85)
    assert labels == {}


def test_record_embedding_latency_journals(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    """§10.1: record_embedding_latency must journal alongside Prometheus.

    The unit is seconds (Prometheus convention; spec §10.4 unit note); the
    `model` label is carried through verbatim so §10.4 can group by model
    when the day comes.
    """
    from spatial_memory.core import metrics

    metrics.record_embedding_latency(0.234, model="arctic-m-long")
    assert len(_patch_journal) == 1
    metric, value, labels = _patch_journal[0]
    assert metric == "embedding_latency_seconds"
    assert value == pytest.approx(0.234)
    assert labels == {"model": "arctic-m-long"}


def test_record_search_similarity_moves_histogram() -> None:
    """§10.1: with prometheus_client installed, the histogram count moves."""
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent — see prometheus-absent test below")

    before = REGISTRY.get_sample_value("spatial_memory_search_similarity_score_count") or 0
    metrics.record_search_similarity(0.5)
    after = REGISTRY.get_sample_value("spatial_memory_search_similarity_score_count") or 0
    assert after == before + 1


def test_record_embedding_latency_moves_histogram() -> None:
    """§10.1: with prometheus_client installed, the histogram count moves."""
    from prometheus_client import REGISTRY

    from spatial_memory.core import metrics

    if not metrics.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client absent — see prometheus-absent test below")

    before = (
        REGISTRY.get_sample_value(
            "spatial_memory_embedding_latency_seconds_count", {"model": "arctic-m-long"}
        )
        or 0
    )
    metrics.record_embedding_latency(0.1, model="arctic-m-long")
    after = (
        REGISTRY.get_sample_value(
            "spatial_memory_embedding_latency_seconds_count", {"model": "arctic-m-long"}
        )
        or 0
    )
    assert after == before + 1


def test_existing_helpers_journal_when_prometheus_absent(
    _patch_journal: list[tuple[str, float, dict]],
) -> None:
    """§10.1: search_similarity + embedding_latency journal even without prometheus.

    This is the entire point of the wire-up: when prometheus_client is not
    installed, the §10.4 ``RecentObservations.search_similarity_*`` and
    ``embedding_latency_*_ms`` fields would silently be empty unless these
    helpers also fed the journal.
    """
    from spatial_memory.core import metrics

    with patch.object(metrics, "PROMETHEUS_AVAILABLE", False):
        metrics.record_search_similarity(0.7)
        metrics.record_embedding_latency(0.05, model="arctic-m-long")

    assert len(_patch_journal) == 2
    assert _patch_journal[0][0] == "search_similarity_score"
    assert _patch_journal[1][0] == "embedding_latency_seconds"
