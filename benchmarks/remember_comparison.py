"""Compare remember() latency with cognitive_offloading_enabled=True vs False.

Measures the overhead that dedup + quality gate adds to the remember() path
using a real LanceDB database and embedding model.

When cognitive_offloading_enabled=True, remember() runs two extra steps:
  1. Dedup check (content hash lookup + vector similarity search)
  2. Quality gate (score_memory_quality() regex scoring)

When False, it skips both and goes straight to embed + store.

Usage:
    python benchmarks/remember_comparison.py
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.services.memory import MemoryService

# High-quality content that passes both dedup and quality gate.
# Each iteration gets a unique number so content is never duplicated.
CONTENT_TEMPLATE = (
    "Decided to use approach #{i} for the API layer because it provides "
    "better error handling in core/handlers.py and reduces latency by 30%. "
    "Updated services/api.py with the new retry logic."
)

# Threshold: enabled overhead should stay under this (median)
OVERHEAD_THRESHOLD_MS = 50.0

WARMUP_ITERATIONS = 5
BENCH_ITERATIONS = 30


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"


def run_benchmark(
    enabled: bool, iterations: int, svc: MemoryService, offset: int = 0
) -> list[float]:
    """Run remember() with the given flag and measure latency per call."""
    timings: list[float] = []
    ns = "bench-on" if enabled else "bench-off"
    for i in range(iterations):
        content = CONTENT_TEMPLATE.format(i=i + offset)
        start = time.perf_counter()
        svc.remember(
            content=content,
            namespace=ns,
            tags=["benchmark"],
            importance=0.8,
            cognitive_offloading_enabled=enabled,
        )
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return timings


def print_stats(label: str, timings: list[float]) -> float:
    """Print formatted stats row and return median in seconds."""
    mn = min(timings)
    median = statistics.median(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95)]
    mx = max(timings)
    print(
        f"  {label:<30s}  {_fmt_ms(mn):>10s}  {_fmt_ms(median):>10s}  "
        f"{_fmt_ms(p95):>10s}  {_fmt_ms(mx):>10s}"
    )
    return median


def main() -> int:
    """Run comparison benchmark and report results."""
    print("Loading embedding model...")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bench-db"
        settings = Settings(memory_path=path, embedding_model="all-MiniLM-L6-v2")
        override_settings(settings)

        db = Database(path)
        db.connect()
        repo = LanceDBMemoryRepository(db)
        svc = MemoryService(repository=repo, embeddings=embeddings)

        # Warmup: populate the DB so LanceDB schema creation and first-write
        # compaction don't skew the measured iterations.
        print(f"Warming up ({WARMUP_ITERATIONS} writes)...")
        run_benchmark(False, WARMUP_ITERATIONS, svc, offset=90000)

        print(f"\nremember() comparison ({BENCH_ITERATIONS} iterations, unique content each)")
        print("=" * 88)
        print()
        print(f"  {'Operation':<30s}  {'Min':>10s}  {'Median':>10s}  {'P95':>10s}  {'Max':>10s}")
        print(f"  {'-' * 30}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        off_timings = run_benchmark(False, BENCH_ITERATIONS, svc, offset=0)
        on_timings = run_benchmark(True, BENCH_ITERATIONS, svc, offset=50000)

        off_median = print_stats("cognitive_offloading=False", off_timings)
        on_median = print_stats("cognitive_offloading=True", on_timings)

        overhead = on_median - off_median
        pct = (overhead / off_median * 100) if off_median > 0 else 0

        print()
        print(f"  Overhead (median): {_fmt_ms(overhead)} ({pct:+.1f}%)")
        print()
        print("  Breakdown of extra work when enabled:")
        print("    - Content hash computation (SHA-256 of normalized content)")
        print("    - Dedup layer 1: content hash lookup in DB")
        print("    - Dedup layer 2: vector similarity search (top-1)")
        print("    - Quality gate: score_memory_quality() regex scoring")

        print()
        print("Threshold check:")
        failed = False
        if overhead * 1000 > OVERHEAD_THRESHOLD_MS:
            print(
                f"  FAIL: Overhead median {_fmt_ms(overhead)} > {OVERHEAD_THRESHOLD_MS}ms threshold"
            )
            failed = True
        else:
            print(
                f"  PASS: Overhead median {_fmt_ms(overhead)} "
                f"<= {OVERHEAD_THRESHOLD_MS}ms threshold"
            )

        print()
        db.close()
        reset_settings()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
