"""Performance benchmarks for Spatial Memory MCP Server.

Run: python scripts/benchmark.py

Measures latency and throughput for key operations.
"""

import time
import statistics
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.services.memory import MemoryService


# Test data
SAMPLE_MEMORIES = [
    "Use repository pattern for data access layer abstraction",
    "React components should be small and focused on a single responsibility",
    "PostgreSQL performs better than MySQL for complex analytical queries",
    "TypeScript adds static typing to JavaScript for better developer experience",
    "Docker containers provide consistent deployment environments",
    "Redis is excellent for caching and session management",
    "GraphQL allows clients to request exactly the data they need",
    "Kubernetes orchestrates container deployment and scaling",
    "Event sourcing stores all changes as a sequence of events",
    "CQRS separates read and write models for better scalability",
    "Microservices enable independent deployment and scaling of components",
    "API rate limiting protects services from abuse and overload",
    "JWT tokens are stateless and work well for distributed systems",
    "Database indexing dramatically improves query performance",
    "Load balancing distributes traffic across multiple server instances",
    "Circuit breakers prevent cascade failures in distributed systems",
    "Message queues decouple producers from consumers",
    "Blue-green deployments enable zero-downtime releases",
    "Feature flags allow gradual rollout of new functionality",
    "Observability requires logs, metrics, and distributed tracing",
]


class Benchmark:
    """Performance benchmark runner."""

    def __init__(self, db_path: str = "./.benchmark-memory"):
        """Initialize benchmark with a test database."""
        self.db_path = db_path
        self.results: dict[str, list[float]] = {}

    def setup(self) -> None:
        """Set up test database and services."""
        print("Setting up benchmark environment...")

        # Clean previous test data
        import shutil
        if Path(self.db_path).exists():
            shutil.rmtree(self.db_path)

        # Initialize services
        self.embeddings = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.db = Database(
            storage_path=self.db_path,
            embedding_dim=self.embeddings.dimensions,
            auto_create_indexes=True,
        )
        self.db.connect()
        self.repository = LanceDBMemoryRepository(self.db)
        self.service = MemoryService(
            repository=self.repository,
            embeddings=self.embeddings,
        )
        print(f"  Embedding model: all-MiniLM-L6-v2 ({self.embeddings.dimensions} dims)")
        print(f"  Database: {self.db_path}")
        print()

    def teardown(self) -> None:
        """Clean up test database."""
        self.db.close()
        import shutil
        if Path(self.db_path).exists():
            shutil.rmtree(self.db_path)

    def measure(self, name: str, func, iterations: int = 10) -> None:
        """Measure operation latency."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        self.results[name] = times

    def run_benchmarks(self) -> None:
        """Run all benchmarks."""

        # 1. Embedding generation
        print("=" * 60)
        print("EMBEDDING BENCHMARKS")
        print("=" * 60)

        self.measure(
            "embed_single",
            lambda: self.embeddings.embed("Test content for embedding"),
            iterations=20,
        )
        self._print_result("embed_single", "Single embedding generation")

        self.measure(
            "embed_batch_10",
            lambda: self.embeddings.embed_batch(SAMPLE_MEMORIES[:10]),
            iterations=10,
        )
        self._print_result("embed_batch_10", "Batch embedding (10 items)")

        self.measure(
            "embed_batch_20",
            lambda: self.embeddings.embed_batch(SAMPLE_MEMORIES),
            iterations=10,
        )
        self._print_result("embed_batch_20", "Batch embedding (20 items)")

        # 2. Remember operations
        print("\n" + "=" * 60)
        print("REMEMBER BENCHMARKS")
        print("=" * 60)

        counter = [0]
        def remember_single():
            counter[0] += 1
            self.service.remember(
                content=f"Benchmark memory {counter[0]}: {SAMPLE_MEMORIES[counter[0] % len(SAMPLE_MEMORIES)]}",
                namespace="benchmark",
                importance=0.5,
            )

        self.measure("remember_single", remember_single, iterations=20)
        self._print_result("remember_single", "Single remember operation")

        batch_counter = [0]
        def remember_batch():
            batch_counter[0] += 1
            memories = [
                {"content": f"Batch {batch_counter[0]} item {i}: {m}", "namespace": "benchmark-batch"}
                for i, m in enumerate(SAMPLE_MEMORIES[:10])
            ]
            self.service.remember_batch(memories)

        self.measure("remember_batch_10", remember_batch, iterations=5)
        self._print_result("remember_batch_10", "Batch remember (10 items)")

        # 3. Recall operations (need data first)
        print("\n" + "=" * 60)
        print("RECALL BENCHMARKS")
        print("=" * 60)

        # Ensure we have enough data
        print("  Populating test data for recall benchmarks...")
        for i, content in enumerate(SAMPLE_MEMORIES * 5):  # 100 memories
            self.service.remember(
                content=f"Recall test {i}: {content}",
                namespace="recall-test",
            )
        print(f"  Added 100 test memories")

        queries = [
            "database performance optimization",
            "container deployment kubernetes",
            "authentication security tokens",
            "microservices architecture patterns",
            "caching strategies redis",
        ]
        query_idx = [0]

        def recall_single():
            query = queries[query_idx[0] % len(queries)]
            query_idx[0] += 1
            self.service.recall(query=query, limit=5, namespace="recall-test")

        self.measure("recall_limit_5", recall_single, iterations=20)
        self._print_result("recall_limit_5", "Recall (limit=5)")

        def recall_limit_10():
            query = queries[query_idx[0] % len(queries)]
            query_idx[0] += 1
            self.service.recall(query=query, limit=10, namespace="recall-test")

        self.measure("recall_limit_10", recall_limit_10, iterations=20)
        self._print_result("recall_limit_10", "Recall (limit=10)")

        def recall_limit_20():
            query = queries[query_idx[0] % len(queries)]
            query_idx[0] += 1
            self.service.recall(query=query, limit=20, namespace="recall-test")

        self.measure("recall_limit_20", recall_limit_20, iterations=20)
        self._print_result("recall_limit_20", "Recall (limit=20)")

        # 4. Nearby operations
        print("\n" + "=" * 60)
        print("NEARBY BENCHMARKS")
        print("=" * 60)

        # Get a memory ID to use
        result = self.service.recall("database", limit=1, namespace="recall-test")
        if result.memories:
            test_id = result.memories[0].id

            def nearby_5():
                self.service.nearby(memory_id=test_id, limit=5)

            self.measure("nearby_limit_5", nearby_5, iterations=20)
            self._print_result("nearby_limit_5", "Nearby (limit=5)")

        # 5. Summary
        self._print_summary()

    def _print_result(self, key: str, description: str) -> None:
        """Print benchmark result."""
        times = self.results[key]
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        min_t = min(times)
        max_t = max(times)
        p50 = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max_t

        print(f"\n  {description}:")
        print(f"    Mean:   {avg:7.2f} ms")
        print(f"    Std:    {std:7.2f} ms")
        print(f"    Min:    {min_t:7.2f} ms")
        print(f"    Max:    {max_t:7.2f} ms")
        print(f"    P50:    {p50:7.2f} ms")
        print(f"    P95:    {p95:7.2f} ms")

        # Throughput for key operations
        if "remember" in key or "recall" in key:
            throughput = 1000 / avg  # ops/sec
            print(f"    Throughput: {throughput:.1f} ops/sec")

    def _print_summary(self) -> None:
        """Print summary table."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'Operation':<25} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Ops/sec':<10}")
        print("-" * 60)

        for key, times in self.results.items():
            avg = statistics.mean(times)
            p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
            throughput = 1000 / avg
            print(f"{key:<25} {avg:<12.2f} {p95:<12.2f} {throughput:<10.1f}")


def main():
    """Run benchmarks."""
    print("\n" + "=" * 60)
    print("SPATIAL MEMORY MCP SERVER - PERFORMANCE BENCHMARKS")
    print("=" * 60 + "\n")

    benchmark = Benchmark()
    try:
        benchmark.setup()
        benchmark.run_benchmarks()
    finally:
        benchmark.teardown()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
