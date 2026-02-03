"""Integration tests for concurrent write operations.

Tests that the write lock properly serializes concurrent writes to prevent
LanceDB version conflicts and data loss.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from spatial_memory.core.database import Database


class TestConcurrentWrites:
    """Test concurrent write operations don't lose data."""

    def test_concurrent_inserts_no_data_loss(
        self,
        temp_storage: Path,
        session_embedding_service,
    ) -> None:
        """Multiple threads inserting simultaneously should not lose data.

        This test reproduces the original bug where 4 agents saving 19 memories
        only persisted 5 due to LanceDB version conflicts.
        """
        # Create a fresh database for this test (not shared)
        db = Database(temp_storage / "concurrent-test")
        db.connect()

        try:
            num_threads = 4
            memories_per_thread = 5
            total_expected = num_threads * memories_per_thread

            results: list[str] = []
            results_lock = threading.Lock()
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def insert_memories(thread_id: int) -> list[str]:
                """Insert memories from a single thread."""
                thread_ids = []
                for i in range(memories_per_thread):
                    content = f"Memory from thread {thread_id}, item {i}"
                    vector = np.random.rand(384).astype(np.float32)
                    try:
                        memory_id = db.insert(
                            content=content,
                            vector=vector,
                            namespace=f"thread-{thread_id}",
                            tags=[f"thread-{thread_id}"],
                        )
                        thread_ids.append(memory_id)
                    except Exception as e:
                        with errors_lock:
                            errors.append(e)
                return thread_ids

            # Run concurrent inserts
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(insert_memories, tid) for tid in range(num_threads)]

                for future in as_completed(futures):
                    try:
                        thread_ids = future.result()
                        with results_lock:
                            results.extend(thread_ids)
                    except Exception as e:
                        with errors_lock:
                            errors.append(e)

            # Verify no errors occurred
            assert not errors, f"Errors during concurrent inserts: {errors}"

            # Verify all IDs were returned
            assert len(results) == total_expected, (
                f"Expected {total_expected} IDs returned, got {len(results)}"
            )

            # Verify all memories actually exist in database
            actual_count = db.count()
            assert actual_count == total_expected, (
                f"Expected {total_expected} memories in database, "
                f"found {actual_count} (data loss detected!)"
            )

            # Verify each memory can be retrieved
            for memory_id in results:
                memory = db.get(memory_id)
                assert memory is not None, f"Memory {memory_id} not found"

        finally:
            db.close()

    def test_concurrent_inserts_and_deletes(
        self,
        temp_storage: Path,
    ) -> None:
        """Concurrent inserts and deletes should not corrupt data."""
        db = Database(temp_storage / "concurrent-mixed")
        db.connect()

        try:
            # Pre-populate with some memories to delete
            initial_ids = []
            for i in range(10):
                memory_id = db.insert(
                    content=f"Initial memory {i}",
                    vector=np.random.rand(384).astype(np.float32),
                    namespace="initial",
                )
                initial_ids.append(memory_id)

            new_ids: list[str] = []
            new_ids_lock = threading.Lock()
            deleted_count = [0]  # Use list for mutability in closure
            deleted_lock = threading.Lock()
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def insert_worker() -> None:
                """Insert new memories."""
                for i in range(5):
                    try:
                        memory_id = db.insert(
                            content=f"New memory {i}",
                            vector=np.random.rand(384).astype(np.float32),
                            namespace="new",
                        )
                        with new_ids_lock:
                            new_ids.append(memory_id)
                    except Exception as e:
                        with errors_lock:
                            errors.append(e)

            def delete_worker() -> None:
                """Delete initial memories."""
                for memory_id in initial_ids[:5]:  # Delete first 5
                    try:
                        db.delete(memory_id)
                        with deleted_lock:
                            deleted_count[0] += 1
                    except Exception as e:
                        with errors_lock:
                            errors.append(e)

            # Run concurrent operations
            threads = [
                threading.Thread(target=insert_worker),
                threading.Thread(target=insert_worker),
                threading.Thread(target=delete_worker),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify no errors
            assert not errors, f"Errors during concurrent operations: {errors}"

            # Verify expected count: 10 initial - 5 deleted + 10 new = 15
            expected = 10 - 5 + 10  # 15
            actual = db.count()
            assert actual == expected, f"Expected {expected} memories, found {actual}"

        finally:
            db.close()

    def test_concurrent_batch_inserts(
        self,
        temp_storage: Path,
    ) -> None:
        """Concurrent batch inserts should not lose data."""
        db = Database(temp_storage / "concurrent-batch")
        db.connect()

        try:
            num_threads = 3
            batch_size = 10
            total_expected = num_threads * batch_size

            all_ids: list[str] = []
            ids_lock = threading.Lock()
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def batch_insert_worker(thread_id: int) -> None:
                """Insert a batch of memories."""
                records = [
                    {
                        "content": f"Batch memory t{thread_id}-{i}",
                        "vector": np.random.rand(384).astype(np.float32),
                        "namespace": f"batch-{thread_id}",
                    }
                    for i in range(batch_size)
                ]
                try:
                    ids = db.insert_batch(records)
                    with ids_lock:
                        all_ids.extend(ids)
                except Exception as e:
                    with errors_lock:
                        errors.append(e)

            # Run concurrent batch inserts
            threads = [
                threading.Thread(target=batch_insert_worker, args=(tid,))
                for tid in range(num_threads)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify no errors
            assert not errors, f"Errors during concurrent batch inserts: {errors}"

            # Verify all data persisted
            actual = db.count()
            assert actual == total_expected, (
                f"Expected {total_expected} memories, found {actual} "
                f"(lost {total_expected - actual})"
            )

        finally:
            db.close()

    def test_concurrent_updates_no_corruption(
        self,
        temp_storage: Path,
    ) -> None:
        """Concurrent updates to same memory should not corrupt data."""
        db = Database(temp_storage / "concurrent-update")
        db.connect()

        try:
            # Create a single memory to update concurrently
            memory_id = db.insert(
                content="Original content",
                vector=np.random.rand(384).astype(np.float32),
                namespace="test",
                importance=0.5,
            )

            update_count = [0]
            update_lock = threading.Lock()
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def update_worker(worker_id: int) -> None:
                """Update the memory multiple times."""
                for i in range(5):
                    try:
                        db.update(
                            memory_id,
                            {
                                "content": f"Updated by worker {worker_id}, iteration {i}",
                            },
                        )
                        with update_lock:
                            update_count[0] += 1
                    except Exception as e:
                        with errors_lock:
                            errors.append(e)

            # Run concurrent updates
            threads = [threading.Thread(target=update_worker, args=(wid,)) for wid in range(3)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Verify no errors
            assert not errors, f"Errors during concurrent updates: {errors}"

            # Verify memory still exists and is valid
            memory = db.get(memory_id)
            assert memory is not None
            assert "Updated by worker" in memory["content"]

            # Verify count is still 1 (no duplicates created)
            assert db.count() == 1

        finally:
            db.close()

    def test_rlock_allows_nested_calls(
        self,
        temp_storage: Path,
    ) -> None:
        """Verify RLock allows nested write operations (bulk_import -> insert_batch)."""
        db = Database(temp_storage / "nested-lock")
        db.connect()

        try:
            # bulk_import internally calls insert_batch, both have @with_write_lock
            # This should not deadlock thanks to RLock
            records = iter(
                [
                    {
                        "content": f"Import record {i}",
                        "vector": np.random.rand(384).astype(np.float32),
                        "namespace": "import",
                    }
                    for i in range(5)
                ]
            )

            # This would deadlock with a regular Lock, but works with RLock
            imported_count, ids = db.bulk_import(records, batch_size=2)

            assert imported_count == 5
            assert len(ids) == 5
            assert db.count() == 5

        finally:
            db.close()

    def test_high_contention_stress_test(
        self,
        temp_storage: Path,
    ) -> None:
        """Stress test with high contention to verify lock robustness."""
        db = Database(temp_storage / "stress-test")
        db.connect()

        try:
            num_threads = 8
            ops_per_thread = 10

            success_count = [0]
            count_lock = threading.Lock()
            errors: list[Exception] = []
            errors_lock = threading.Lock()

            def stress_worker(worker_id: int) -> None:
                """Perform mixed operations."""
                for i in range(ops_per_thread):
                    try:
                        # Insert
                        memory_id = db.insert(
                            content=f"Stress test w{worker_id}-{i}",
                            vector=np.random.rand(384).astype(np.float32),
                            namespace=f"stress-{worker_id}",
                        )

                        # Update
                        db.update(memory_id, {"importance": 0.8})

                        with count_lock:
                            success_count[0] += 1

                    except Exception as e:
                        with errors_lock:
                            errors.append(e)

            # Launch many concurrent workers
            start_time = time.time()
            threads = [
                threading.Thread(target=stress_worker, args=(wid,)) for wid in range(num_threads)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            elapsed = time.time() - start_time

            # Verify no errors
            assert not errors, f"Errors during stress test: {errors[:5]}..."

            # Verify expected count
            expected = num_threads * ops_per_thread
            actual = db.count()
            assert actual == expected, f"Expected {expected} memories, found {actual}"

            # Log performance info (not a hard assertion)
            ops_total = expected * 2  # insert + update per iteration
            ops_per_sec = ops_total / elapsed if elapsed > 0 else 0
            print(f"\nStress test: {ops_total} ops in {elapsed:.2f}s ({ops_per_sec:.1f} ops/sec)")

        finally:
            db.close()
