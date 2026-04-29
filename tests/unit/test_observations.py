"""Unit tests for compute_recent_observations (US-023 / spec §10.4).

Builds synthetic JSONL journals under a tmp_path and verifies the
percentile rollups the stats() endpoint will surface.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_jsonl(
    journal_dir: Path,
    entries: list[dict[str, object]],
    filename: str = "2026-04-29-1234-1700000000.jsonl",
) -> Path:
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


def _entry(
    metric: str, value: float, ts: datetime | None = None, **labels: object
) -> dict[str, object]:
    return {
        "ts": (ts or datetime.now(timezone.utc)).isoformat(),
        "request_id": "test",
        "metric": metric,
        "value": value,
        "labels": labels,
    }


def test_compute_observations_missing_dir_returns_empty(tmp_path: Path) -> None:
    """No journal dir → all p50/p95 are None, capture_cap_hit_rate=0."""
    from spatial_memory.core.observations import compute_recent_observations

    obs = compute_recent_observations(tmp_path / "does-not-exist", window_days=7)

    assert obs.capture_cap_hit_rate == 0.0
    assert obs.capture_size_p50_tokens is None
    assert obs.capture_size_p95_tokens is None
    assert obs.recall_result_count_p50 is None
    assert obs.recall_result_count_p95 is None
    assert obs.reinforced_memory_rank_p50 is None
    assert obs.reinforced_memory_rank_p95 is None
    assert obs.search_similarity_p50 is None
    assert obs.search_similarity_p95 is None
    assert obs.embedding_latency_p50_ms is None
    assert obs.embedding_latency_p95_ms is None
    assert obs.window_days == 7
    assert obs.sample_size == 0


def test_compute_observations_empty_journal_files(tmp_path: Path) -> None:
    """Journal files exist but contain no entries → same as missing dir."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    _write_jsonl(journal_dir, [])

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.sample_size == 0
    assert obs.capture_cap_hit_rate == 0.0
    assert obs.capture_size_p50_tokens is None
    assert obs.search_similarity_p50 is None


def test_compute_observations_single_metric_percentiles(tmp_path: Path) -> None:
    """100 capture_size_tokens entries 1..100 → p50≈50, p95≈95."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    entries = [_entry("capture_size_tokens", float(i), stage="pre_cap") for i in range(1, 101)]
    _write_jsonl(journal_dir, entries)

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.sample_size == 100
    assert obs.capture_size_p50_tokens is not None
    assert obs.capture_size_p95_tokens is not None
    # quantiles(n=100) on 1..100 → p50 ≈ 50, p95 ≈ 95 (within ±2)
    assert 48 <= obs.capture_size_p50_tokens <= 52
    assert 93 <= obs.capture_size_p95_tokens <= 97


def test_compute_observations_embedding_latency_in_ms(tmp_path: Path) -> None:
    """embedding_latency_seconds journals seconds; output multiplies by 1000."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    # 100 latency entries from 0.001s..0.100s (1ms..100ms)
    entries = [
        _entry("embedding_latency_seconds", i / 1000.0, model="arctic-m-long")
        for i in range(1, 101)
    ]
    _write_jsonl(journal_dir, entries)

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.embedding_latency_p50_ms is not None
    assert obs.embedding_latency_p95_ms is not None
    # Expect p50≈50ms, p95≈95ms (raw seconds × 1000)
    assert 48 <= obs.embedding_latency_p50_ms <= 52
    assert 93 <= obs.embedding_latency_p95_ms <= 97


def test_compute_observations_capture_cap_hit_rate(tmp_path: Path) -> None:
    """cap_hit_rate = capture_cap_hit_total / capture_size_tokens entries."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    entries: list[dict[str, object]] = []
    # 10 capture events
    for _ in range(10):
        entries.append(_entry("capture_size_tokens", 1000, stage="pre_cap"))
    # 3 of them hit the cap
    for _ in range(3):
        entries.append(_entry("capture_cap_hit_total", 1, tool="ingest"))
    _write_jsonl(journal_dir, entries)

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.capture_cap_hit_rate == pytest.approx(0.3, rel=1e-6)


def test_compute_observations_optional_pairs_none_when_absent(tmp_path: Path) -> None:
    """If a metric never appears in journal, its p50/p95 must be None (not 0)."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    # Only capture_size_tokens — nothing for reinforce/search/recall
    entries = [_entry("capture_size_tokens", 100, stage="pre_cap")]
    _write_jsonl(journal_dir, entries)

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.recall_result_count_p50 is None
    assert obs.recall_result_count_p95 is None
    assert obs.reinforced_memory_rank_p50 is None
    assert obs.reinforced_memory_rank_p95 is None
    assert obs.search_similarity_p50 is None
    assert obs.search_similarity_p95 is None
    assert obs.embedding_latency_p50_ms is None
    assert obs.embedding_latency_p95_ms is None


def test_compute_observations_filters_outside_window(tmp_path: Path) -> None:
    """Entries older than window_days must not contribute to percentiles."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    now = datetime.now(timezone.utc)
    fresh = [_entry("recall_result_count", 5, ts=now)]
    stale = [_entry("recall_result_count", 999, ts=now - timedelta(days=30))]
    _write_jsonl(journal_dir, fresh + stale, filename="fresh.jsonl")

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.sample_size == 1
    assert obs.recall_result_count_p50 == 5
    assert obs.recall_result_count_p95 == 5


def test_compute_observations_handles_corrupt_lines(tmp_path: Path) -> None:
    """Malformed JSON lines must be skipped, not raise."""
    from spatial_memory.core.observations import compute_recent_observations

    journal_dir = tmp_path / "metrics"
    journal_dir.mkdir(parents=True)
    path = journal_dir / "test.jsonl"
    path.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps(_entry("recall_result_count", 1)),
                "{partial:",
                json.dumps(_entry("recall_result_count", 2)),
            ]
        )
    )

    obs = compute_recent_observations(journal_dir, window_days=7)

    assert obs.sample_size == 2
    assert obs.recall_result_count_p50 is not None
