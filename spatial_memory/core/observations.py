"""Compute :class:`RecentObservations` percentiles from the JSONL journal.

The journal is the always-on observability sink (see
:mod:`spatial_memory.core.metrics_journal`). This module aggregates entries
from the last ``window_days`` into the p50/p95 rollups exposed by
``stats(include_observations=True)`` per spec §10.4 (D7).

Design choices
--------------
* The journal stores latencies in **seconds** (Prometheus convention). The
  ``embedding_latency_*_ms`` outputs are pre-multiplied by 1000 here so
  consumers don't have to remember the unit.
* Optional p50/p95 pairs (``reinforced_memory_rank``, ``search_similarity``,
  ``embedding_latency``) return ``None`` when the metric had no observations
  in the window — distinguishes "no data" from "data is zero".
* ``capture_cap_hit_rate`` is computed as
  ``capture_cap_hit_total_count / capture_size_tokens_count`` so the rate is
  meaningful regardless of the absolute capture volume.
* Corrupt or unparseable JSON lines are silently skipped — the journal must
  never break the request path it summarises.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spatial_memory.core.models import RecentObservations


def compute_recent_observations(
    journal_dir: Path,
    window_days: int = 7,
) -> RecentObservations:
    """Aggregate JSONL entries in ``journal_dir`` into a :class:`RecentObservations`.

    Args:
        journal_dir: Directory containing per-process daily JSONL files.
        window_days: Lookback window for percentile computation.

    Returns:
        :class:`RecentObservations` with required scalars populated and any
        absent metric's p50/p95 left ``None``.
    """
    if not journal_dir.exists():
        return RecentObservations(window_days=window_days, sample_size=0)

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    values: dict[str, list[float]] = defaultdict(list)
    sample_size = 0

    for jsonl in journal_dir.glob("*.jsonl"):
        try:
            text = jsonl.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(str(entry["ts"]))
                if ts < cutoff:
                    continue
                metric = str(entry["metric"])
                value = float(entry["value"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
            values[metric].append(value)
            sample_size += 1

    capture_total = len(values.get("capture_size_tokens", []))
    capture_caps = len(values.get("capture_cap_hit_total", []))
    cap_hit_rate = (capture_caps / capture_total) if capture_total > 0 else 0.0

    return RecentObservations(
        capture_cap_hit_rate=cap_hit_rate,
        capture_size_p50_tokens=_percentile(values.get("capture_size_tokens"), 50),
        capture_size_p95_tokens=_percentile(values.get("capture_size_tokens"), 95),
        recall_result_count_p50=_percentile(values.get("recall_result_count"), 50),
        recall_result_count_p95=_percentile(values.get("recall_result_count"), 95),
        reinforced_memory_rank_p50=_percentile(values.get("reinforced_memory_rank"), 50),
        reinforced_memory_rank_p95=_percentile(values.get("reinforced_memory_rank"), 95),
        search_similarity_p50=_percentile(values.get("search_similarity_score"), 50),
        search_similarity_p95=_percentile(values.get("search_similarity_score"), 95),
        embedding_latency_p50_ms=_percentile_seconds_to_ms(
            values.get("embedding_latency_seconds"), 50
        ),
        embedding_latency_p95_ms=_percentile_seconds_to_ms(
            values.get("embedding_latency_seconds"), 95
        ),
        window_days=window_days,
        sample_size=sample_size,
    )


def _percentile(samples: list[float] | None, p: int) -> float | None:
    """Return the p-th percentile, or ``None`` if ``samples`` is empty.

    For a single sample, returns it directly (statistics.quantiles requires
    at least two data points).
    """
    if not samples:
        return None
    if len(samples) == 1:
        return float(samples[0])
    cuts = statistics.quantiles(samples, n=100)
    # quantiles(n=100) returns 99 cut points: [Q1, Q2, ..., Q99]
    # p=50 → index 49, p=95 → index 94
    return float(cuts[p - 1])


def _percentile_seconds_to_ms(samples: list[float] | None, p: int) -> float | None:
    raw = _percentile(samples, p)
    return None if raw is None else raw * 1000.0
