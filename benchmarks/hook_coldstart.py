"""Cold-start benchmark for cognitive offloading pipeline.

Measures Python-side processing latency for operations a client-side hook
would trigger:
1. Import: Load quality_gate and lifecycle_ops modules
2. Signal match: Run score_memory_quality() against sample strings
3. Queue file write: Serialize QueueFile to JSON, write tmp/, rename to new/

Usage:
    python benchmarks/hook_coldstart.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample inputs covering the quality gate spectrum
# ---------------------------------------------------------------------------

SAMPLE_INPUTS: list[tuple[str, str]] = [
    (
        "high-signal decision",
        "Decided to use PostgreSQL because it handles JSONB natively and "
        "supports full-text search, which eliminates the need for a separate "
        "Elasticsearch cluster in our deployment.",
    ),
    (
        "solution with file path",
        "The fix was in core/database.py — the connection pool was exhausted "
        "because close() wasn't called in the finally block of search(). "
        "Added a context manager wrapper around LanceDB connections.",
    ),
    (
        "trivial greeting",
        "ok thanks",
    ),
    (
        "medium-signal pattern",
        "The trick is to always batch embed before dedup checking, since "
        "generating embeddings one at a time adds ~50ms per call.",
    ),
    (
        "rich multi-line",
        "Upgraded sentence-transformers from v2.2.2 to v3.0.1 because the "
        "new ONNX backend in encode() reduces cold-start from 1.2s to 0.4s. "
        "Changed EmbeddingService.__init__() to pass model_kwargs={'onnx': True}. "
        "See https://huggingface.co/docs/sentence-transformers for migration "
        "guide. Benchmark: `pytest benchmarks/ -k perf` shows 3x throughput.",
    ),
]

# Thresholds (median must be below these, or exit code 1)
QUALITY_GATE_THRESHOLD_MS = 50.0
QUEUE_WRITE_THRESHOLD_MS = 10.0


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"


def bench_import(iterations: int = 100) -> list[float]:
    """Benchmark importing quality_gate and lifecycle_ops modules."""
    timings: list[float] = []
    for _ in range(iterations):
        # Force re-import by removing from sys.modules
        sys.modules.pop("spatial_memory.core.quality_gate", None)
        sys.modules.pop("spatial_memory.core.lifecycle_ops", None)

        start = time.perf_counter()
        import spatial_memory.core.lifecycle_ops  # noqa: F401
        import spatial_memory.core.quality_gate  # noqa: F401

        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return timings


def bench_signal_match(iterations: int = 100) -> list[float]:
    """Benchmark score_memory_quality() against all sample strings."""
    from spatial_memory.core.quality_gate import score_memory_quality

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        for _label, content in SAMPLE_INPUTS:
            score_memory_quality(content, tags=["test"], metadata=None)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return timings


def bench_queue_write(iterations: int = 100) -> list[float]:
    """Benchmark serializing a QueueFile to JSON and writing via Maildir rename."""
    from spatial_memory.core.queue_constants import QUEUE_FILE_VERSION

    queue_data = {
        "version": QUEUE_FILE_VERSION,
        "content": SAMPLE_INPUTS[0][1],
        "source_hook": "prompt-submit",
        "timestamp": "2025-01-15T10:30:00Z",
        "project_root_dir": "/home/user/code/my-project",
        "suggested_namespace": "decisions",
        "suggested_tags": ["postgresql", "architecture"],
        "suggested_importance": 0.8,
        "signal_tier": 1,
        "signal_patterns_matched": ["decision"],
        "context": {},
        "client": "claude-code",
    }
    payload = json.dumps(queue_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir) / "tmp"
        new_dir = Path(tmpdir) / "new"
        tmp_dir.mkdir()
        new_dir.mkdir()

        timings: list[float] = []
        for i in range(iterations):
            filename = f"{i:06d}-bench.json"
            tmp_path = tmp_dir / filename
            new_path = new_dir / filename

            start = time.perf_counter()
            tmp_path.write_text(payload, encoding="utf-8")
            os.rename(str(tmp_path), str(new_path))
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

            # Cleanup for next iteration
            new_path.unlink()

    return timings


def print_results(label: str, timings: list[float]) -> float:
    """Print formatted results table row and return median in seconds."""
    mn = min(timings)
    median = statistics.median(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95)]
    mx = max(timings)
    print(
        f"  {label:<20s}  {_fmt_ms(mn):>10s}  {_fmt_ms(median):>10s}  "
        f"{_fmt_ms(p95):>10s}  {_fmt_ms(mx):>10s}"
    )
    return median


def main() -> int:
    """Run all benchmarks and report results."""
    iterations = 100
    print(f"Cold-start benchmark ({iterations} iterations per operation)")
    print("=" * 78)
    print()
    print(f"  {'Operation':<20s}  {'Min':>10s}  {'Median':>10s}  {'P95':>10s}  {'Max':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    import_median = print_results("Import", bench_import(iterations))
    signal_median = print_results("Signal match (5x)", bench_signal_match(iterations))
    queue_median = print_results("Queue file write", bench_queue_write(iterations))

    print()
    print("Threshold check:")

    failed = False
    if signal_median * 1000 > QUALITY_GATE_THRESHOLD_MS:
        print(
            f"  FAIL: Signal match median {_fmt_ms(signal_median)} "
            f"> {QUALITY_GATE_THRESHOLD_MS}ms threshold"
        )
        failed = True
    else:
        print(
            f"  PASS: Signal match median {_fmt_ms(signal_median)} "
            f"<= {QUALITY_GATE_THRESHOLD_MS}ms threshold"
        )

    if queue_median * 1000 > QUEUE_WRITE_THRESHOLD_MS:
        print(
            f"  FAIL: Queue write median {_fmt_ms(queue_median)} "
            f"> {QUEUE_WRITE_THRESHOLD_MS}ms threshold"
        )
        failed = True
    else:
        print(
            f"  PASS: Queue write median {_fmt_ms(queue_median)} "
            f"<= {QUEUE_WRITE_THRESHOLD_MS}ms threshold"
        )

    # Import is informational only (no threshold)
    print(f"  INFO: Import median {_fmt_ms(import_median)} (no threshold)")

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
