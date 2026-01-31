"""Systematically test all 21 MCP tools.

Run: python scripts/test_all_tools.py

Tests each tool and reports success/failure.
"""

import json
import sys
import time
import shutil
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_memory.server import SpatialMemoryServer


class ToolTester:
    """Test all MCP tools systematically."""

    def __init__(self, db_path: str = "./.test-memory"):
        self.db_path = db_path
        self.server: SpatialMemoryServer | None = None
        self.results: dict[str, dict[str, Any]] = {}
        self.memory_ids: list[str] = []

    def setup(self) -> None:
        """Set up test environment."""
        print("\n" + "=" * 70)
        print("SPATIAL MEMORY MCP - COMPREHENSIVE TOOL TEST")
        print("=" * 70)

        # Clean previous test data
        if Path(self.db_path).exists():
            shutil.rmtree(self.db_path)

        # Set environment for test database
        import os
        os.environ["SPATIAL_MEMORY_MEMORY_PATH"] = self.db_path

        # Create server
        self.server = SpatialMemoryServer()
        print(f"\nTest database: {self.db_path}")
        print("Server initialized successfully\n")

    def teardown(self) -> None:
        """Clean up."""
        if self.server:
            self.server.close()
        if Path(self.db_path).exists():
            shutil.rmtree(self.db_path)
        # Clean up export/import test directories
        for dir_path in [Path("./exports"), Path("./imports")]:
            if dir_path.exists():
                shutil.rmtree(dir_path)

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool and return result."""
        return self.server._handle_tool(name, arguments)

    def test_tool(self, name: str, arguments: dict, description: str) -> bool:
        """Test a single tool."""
        print(f"  Testing: {name}")
        print(f"    {description}")

        start = time.perf_counter()
        try:
            result = self.call_tool(name, arguments)
            elapsed = (time.perf_counter() - start) * 1000

            self.results[name] = {
                "status": "PASS",
                "elapsed_ms": elapsed,
                "result": result,
            }
            print(f"    [PASS] PASS ({elapsed:.1f}ms)")
            return True

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.results[name] = {
                "status": "FAIL",
                "elapsed_ms": elapsed,
                "error": str(e),
            }
            print(f"    [FAIL] FAIL: {e}")
            return False

    def run_tests(self) -> None:
        """Run all tool tests."""

        # ============================================================
        # CORE OPERATIONS
        # ============================================================
        print("\n" + "-" * 70)
        print("CORE OPERATIONS")
        print("-" * 70)

        # 1. remember
        self.test_tool(
            "remember",
            {
                "content": "Use repository pattern for data access",
                "namespace": "test",
                "tags": ["patterns", "architecture"],
                "importance": 0.8,
            },
            "Store a single memory"
        )
        if "remember" in self.results and self.results["remember"]["status"] == "PASS":
            self.memory_ids.append(self.results["remember"]["result"]["id"])

        # 2. remember_batch
        self.test_tool(
            "remember_batch",
            {
                "memories": [
                    {"content": "React uses virtual DOM for updates", "namespace": "test", "tags": ["react"]},
                    {"content": "TypeScript adds static typing", "namespace": "test", "tags": ["typescript"]},
                    {"content": "Docker containers are portable", "namespace": "test", "tags": ["docker"]},
                    {"content": "Redis provides fast caching", "namespace": "test", "tags": ["redis"]},
                    {"content": "PostgreSQL handles complex queries", "namespace": "test", "tags": ["database"]},
                ]
            },
            "Store multiple memories in batch"
        )
        if "remember_batch" in self.results and self.results["remember_batch"]["status"] == "PASS":
            self.memory_ids.extend(self.results["remember_batch"]["result"]["ids"])

        # 3. recall
        self.test_tool(
            "recall",
            {"query": "database patterns", "limit": 3, "namespace": "test"},
            "Search for similar memories"
        )

        # 4. nearby
        if self.memory_ids:
            self.test_tool(
                "nearby",
                {"memory_id": self.memory_ids[0], "limit": 3},
                "Find memories near a specific memory"
            )

        # 5. forget (we'll create a temp memory first)
        temp_result = self.call_tool("remember", {"content": "Temporary memory to delete", "namespace": "temp"})
        temp_id = temp_result["id"]
        self.test_tool(
            "forget",
            {"memory_id": temp_id},
            "Delete a single memory"
        )

        # 6. forget_batch
        temp_batch = self.call_tool("remember_batch", {
            "memories": [
                {"content": "Temp 1", "namespace": "temp"},
                {"content": "Temp 2", "namespace": "temp"},
            ]
        })
        self.test_tool(
            "forget_batch",
            {"memory_ids": temp_batch["ids"]},
            "Delete multiple memories"
        )

        # ============================================================
        # SPATIAL OPERATIONS
        # ============================================================
        print("\n" + "-" * 70)
        print("SPATIAL OPERATIONS")
        print("-" * 70)

        # Need at least 2 memories for journey
        if len(self.memory_ids) >= 2:
            # 7. journey
            self.test_tool(
                "journey",
                {"start_id": self.memory_ids[0], "end_id": self.memory_ids[-1], "steps": 5},
                "Navigate semantic path between two memories"
            )

            # 8. wander
            self.test_tool(
                "wander",
                {"start_id": self.memory_ids[0], "steps": 3, "temperature": 0.5},
                "Random walk through memory space"
            )

        # 9. regions (need more memories)
        # Add more memories for clustering
        self.call_tool("remember_batch", {
            "memories": [
                {"content": "Python is great for data science", "namespace": "test"},
                {"content": "Machine learning models need training data", "namespace": "test"},
                {"content": "Neural networks have multiple layers", "namespace": "test"},
                {"content": "Deep learning requires GPU acceleration", "namespace": "test"},
            ]
        })
        self.test_tool(
            "regions",
            {"namespace": "test", "min_cluster_size": 2},
            "Discover semantic clusters"
        )

        # 10. visualize
        self.test_tool(
            "visualize",
            {"namespace": "test", "format": "json", "dimensions": 2},
            "Generate 2D visualization"
        )

        # ============================================================
        # LIFECYCLE OPERATIONS
        # ============================================================
        print("\n" + "-" * 70)
        print("LIFECYCLE OPERATIONS")
        print("-" * 70)

        # 11. decay
        self.test_tool(
            "decay",
            {"namespace": "test", "decay_function": "exponential", "dry_run": True},
            "Apply time-based decay (dry run)"
        )

        # 12. reinforce
        if self.memory_ids:
            self.test_tool(
                "reinforce",
                {"memory_ids": [self.memory_ids[0]], "boost_type": "additive", "boost_amount": 0.1},
                "Boost memory importance"
            )

        # 13. extract
        self.test_tool(
            "extract",
            {
                "text": "We decided to use PostgreSQL for the database. The solution was to add an index.",
                "namespace": "extracted",
                "min_confidence": 0.5,
            },
            "Extract memories from text"
        )

        # 14. consolidate
        # Add some similar memories first
        self.call_tool("remember_batch", {
            "memories": [
                {"content": "Use caching for better performance", "namespace": "consolidate-test"},
                {"content": "Caching improves application performance", "namespace": "consolidate-test"},
                {"content": "Cache data to improve performance", "namespace": "consolidate-test"},
            ]
        })
        self.test_tool(
            "consolidate",
            {"namespace": "consolidate-test", "similarity_threshold": 0.8, "dry_run": True},
            "Find and merge similar memories (dry run)"
        )

        # ============================================================
        # UTILITY OPERATIONS
        # ============================================================
        print("\n" + "-" * 70)
        print("UTILITY OPERATIONS")
        print("-" * 70)

        # 15. stats
        self.test_tool(
            "stats",
            {"include_index_details": True},
            "Get database statistics"
        )

        # 16. namespaces
        self.test_tool(
            "namespaces",
            {"include_stats": True},
            "List all namespaces"
        )

        # 17. delete_namespace (dry run)
        self.test_tool(
            "delete_namespace",
            {"namespace": "consolidate-test", "dry_run": True},
            "Preview namespace deletion"
        )

        # 18. rename_namespace
        self.call_tool("remember", {"content": "Test memory", "namespace": "old-name"})
        self.test_tool(
            "rename_namespace",
            {"old_namespace": "old-name", "new_namespace": "new-name"},
            "Rename a namespace"
        )

        # 19. export_memories
        # Use ./exports directory which is in the default allowed paths
        exports_dir = Path("./exports")
        exports_dir.mkdir(exist_ok=True)
        export_path = str(exports_dir / "test_export.json")
        self.test_tool(
            "export_memories",
            {"output_path": export_path, "format": "json", "namespace": "test"},
            "Export memories to JSON"
        )

        # 20. import_memories (dry run)
        if Path(export_path).exists():
            # Move export file to imports directory for import test
            imports_dir = Path("./imports")
            imports_dir.mkdir(exist_ok=True)
            import_path = str(imports_dir / "test_import.json")
            shutil.copy(export_path, import_path)
            self.test_tool(
                "import_memories",
                {"source_path": import_path, "format": "json", "dry_run": True},
                "Validate import file (dry run)"
            )

        # 21. hybrid_recall
        self.test_tool(
            "hybrid_recall",
            {"query": "database caching performance", "alpha": 0.5, "limit": 3},
            "Combined vector + keyword search"
        )

        # 22. health (bonus)
        self.test_tool(
            "health",
            {"verbose": True},
            "Check system health"
        )

    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        total = len(self.results)

        print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"  Success Rate: {passed/total*100:.1f}%\n")

        print(f"{'Tool':<25} {'Status':<8} {'Time (ms)':<12}")
        print("-" * 50)

        for name, result in self.results.items():
            status = result["status"]
            elapsed = result["elapsed_ms"]
            status_symbol = "[PASS]" if status == "PASS" else "[FAIL]"
            print(f"{name:<25} {status_symbol} {status:<6} {elapsed:>8.1f}")

        if failed > 0:
            print("\n" + "-" * 70)
            print("FAILURES:")
            print("-" * 70)
            for name, result in self.results.items():
                if result["status"] == "FAIL":
                    print(f"\n  {name}: {result.get('error', 'Unknown error')}")

        print("\n" + "=" * 70)


def main():
    """Run all tool tests."""
    tester = ToolTester()
    try:
        tester.setup()
        tester.run_tests()
        tester.print_summary()
    finally:
        tester.teardown()


if __name__ == "__main__":
    main()
