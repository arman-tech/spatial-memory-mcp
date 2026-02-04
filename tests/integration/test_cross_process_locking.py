"""Integration tests for cross-process file locking.

Tests that the filelock mechanism properly serializes writes across
multiple processes to prevent data corruption in multi-instance scenarios.

Note: LanceDB is not fork-safe. On Linux, multiprocessing uses fork() by default,
which causes issues. We use 'spawn' start method to ensure compatibility.
"""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Force spawn method on all platforms for LanceDB compatibility
# LanceDB is not fork-safe, so we must use spawn
_mp_context = multiprocessing.get_context("spawn")

# =============================================================================
# Helper Functions for Multiprocess Tests
# =============================================================================


def _insert_memories_worker(
    storage_path: str,
    worker_id: int,
    num_inserts: int,
    results_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    """Worker function that inserts memories from a separate process.

    Args:
        storage_path: Path to the database.
        worker_id: Unique identifier for this worker.
        num_inserts: Number of memories to insert.
        results_queue: Queue to report results back to main process.
    """
    try:
        from spatial_memory.core.database import Database

        db = Database(
            Path(storage_path),
            embedding_dim=384,
            filelock_enabled=True,
            filelock_timeout=30.0,
        )
        db.connect()

        inserted_ids: list[str] = []
        for i in range(num_inserts):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory_id = db.insert(
                content=f"Worker {worker_id} memory {i}",
                vector=vec,
                namespace=f"worker_{worker_id}",
                tags=[f"worker_{worker_id}"],
                importance=0.5,
            )
            inserted_ids.append(memory_id)

        db.close()
        results_queue.put(
            {
                "worker_id": worker_id,
                "success": True,
                "inserted_count": len(inserted_ids),
                "inserted_ids": inserted_ids,
                "error": None,
            }
        )
    except Exception as e:
        results_queue.put(
            {
                "worker_id": worker_id,
                "success": False,
                "inserted_count": 0,
                "inserted_ids": [],
                "error": str(e),
            }
        )


def _try_acquire_lock_worker(
    storage_path: str,
    timeout: float,
    results_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    """Worker that attempts to acquire the lock and holds it briefly.

    Args:
        storage_path: Path to the database.
        timeout: Lock acquisition timeout.
        results_queue: Queue to report results.
    """
    try:
        from spatial_memory.core.database import ProcessLockManager

        lock_path = Path(storage_path) / ".spatial-memory.lock"
        lock = ProcessLockManager(
            lock_path=lock_path,
            timeout=timeout,
            poll_interval=0.05,
            enabled=True,
        )

        acquired = lock.acquire()
        if acquired:
            # Hold the lock for a bit
            time.sleep(0.5)
            lock.release()

        results_queue.put(
            {
                "acquired": True,
                "error": None,
            }
        )
    except Exception as e:
        results_queue.put(
            {
                "acquired": False,
                "error": str(e),
            }
        )


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCrossProcessLocking:
    """Tests for cross-process file locking behavior."""

    def test_multiple_processes_no_data_loss(self) -> None:
        """Multiple processes inserting concurrently should not lose data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            num_workers = 3
            inserts_per_worker = 5

            # Pre-create the database to avoid table creation race conditions
            from spatial_memory.core.database import Database

            init_db = Database(
                storage_path,
                embedding_dim=384,
                filelock_enabled=True,
            )
            init_db.connect()
            init_db.close()

            # Create results queue using spawn context
            results_queue: "multiprocessing.Queue[dict[str, Any]]" = _mp_context.Queue()

            # Start worker processes using spawn context
            processes: list[multiprocessing.Process] = []
            for worker_id in range(num_workers):
                p = _mp_context.Process(
                    target=_insert_memories_worker,
                    args=(str(storage_path), worker_id, inserts_per_worker, results_queue),
                )
                processes.append(p)

            # Start all processes
            for p in processes:
                p.start()

            # Wait for all to complete
            for p in processes:
                p.join(timeout=60)

            # Collect results
            results: list[dict[str, Any]] = []
            while not results_queue.empty():
                results.append(results_queue.get())

            # Verify all workers succeeded
            assert len(results) == num_workers, (
                f"Expected {num_workers} results, got {len(results)}"
            )

            for result in results:
                assert result["success"], f"Worker {result['worker_id']} failed: {result['error']}"
                assert result["inserted_count"] == inserts_per_worker

            # Verify total records in database
            from spatial_memory.core.database import Database

            db = Database(storage_path, embedding_dim=384)
            db.connect()
            total_count = db.table.count_rows()
            db.close()

            expected_total = num_workers * inserts_per_worker
            assert total_count == expected_total, (
                f"Expected {expected_total} total records, got {total_count}. "
                "Data may have been lost due to concurrent writes."
            )

    def test_lock_timeout_raises_error(self) -> None:
        """Lock acquisition should raise FileLockError on timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # First, acquire the lock in the main process
            from spatial_memory.core.database import ProcessLockManager

            lock_path = storage_path / ".spatial-memory.lock"
            main_lock = ProcessLockManager(
                lock_path=lock_path,
                timeout=30.0,
                poll_interval=0.01,
                enabled=True,
            )
            main_lock.acquire()

            try:
                # Try to acquire from a worker with short timeout
                results_queue: "multiprocessing.Queue[dict[str, Any]]" = _mp_context.Queue()

                p = _mp_context.Process(
                    target=_try_acquire_lock_worker,
                    args=(str(storage_path), 0.5, results_queue),  # Short timeout
                )
                p.start()
                p.join(timeout=10)

                # Worker should have failed to acquire
                result = results_queue.get()
                assert result["acquired"] is False
                assert "FileLockError" in result["error"] or "Timed out" in result["error"]
            finally:
                main_lock.release()

    def test_database_operations_serialize_correctly(self) -> None:
        """Database write operations should serialize across processes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create initial database
            from spatial_memory.core.database import Database

            db = Database(
                storage_path,
                embedding_dim=384,
                filelock_enabled=True,
            )
            db.connect()

            # Insert initial record
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            db.insert(
                content="Initial memory",
                vector=vec,
                namespace="test",
            )

            # Count before parallel inserts
            count_before = db.table.count_rows()
            assert count_before == 1

            db.close()

            # Run parallel inserts from subprocesses using spawn context
            results_queue: "multiprocessing.Queue[dict[str, Any]]" = _mp_context.Queue()

            processes: list[multiprocessing.Process] = []
            for worker_id in range(2):
                p = _mp_context.Process(
                    target=_insert_memories_worker,
                    args=(str(storage_path), worker_id, 3, results_queue),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join(timeout=30)

            # Verify results
            db = Database(storage_path, embedding_dim=384)
            db.connect()
            final_count = db.table.count_rows()
            db.close()

            # 1 initial + 2 workers * 3 each = 7 total
            assert final_count == 7, f"Expected 7, got {final_count}"

    def test_filelock_disabled_still_works(self) -> None:
        """Database should work correctly when filelock is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            from spatial_memory.core.database import Database

            db = Database(
                storage_path,
                embedding_dim=384,
                filelock_enabled=False,
            )
            db.connect()

            # _process_lock should be None when disabled
            assert db._process_lock is None

            # Operations should still work
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory_id = db.insert(
                content="Test memory",
                vector=vec,
                namespace="test",
            )

            assert memory_id is not None
            assert db.table.count_rows() == 1

            db.close()


@pytest.mark.integration
class TestFileLockErrorDetails:
    """Tests for FileLockError exception details."""

    def test_filelock_error_contains_path(self) -> None:
        """FileLockError should contain the lock path."""
        from spatial_memory.core.errors import FileLockError

        error = FileLockError(
            lock_path="/path/to/lock",
            timeout=30.0,
        )

        assert error.lock_path == "/path/to/lock"
        assert error.timeout == 30.0
        assert "/path/to/lock" in str(error)
        assert "30" in str(error)

    def test_filelock_error_custom_message(self) -> None:
        """FileLockError should support custom message."""
        from spatial_memory.core.errors import FileLockError

        error = FileLockError(
            lock_path="/path/to/lock",
            timeout=30.0,
            message="Custom error message",
        )

        assert str(error) == "Custom error message"
        assert error.message == "Custom error message"


@pytest.mark.integration
class TestReconnectWithFilelock:
    """Tests for reconnect behavior with filelock."""

    def test_reconnect_reinitializes_process_lock(self) -> None:
        """Reconnect should reinitialize the process lock manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            from spatial_memory.core.database import Database

            db = Database(
                storage_path,
                embedding_dim=384,
                filelock_enabled=True,
            )
            db.connect()

            # Get reference to original lock
            original_lock = db._process_lock
            assert original_lock is not None

            # Reconnect
            db.reconnect()

            # Should have a new lock manager
            assert db._process_lock is not None
            # The lock manager should still be functional
            assert db._process_lock.enabled is True

            # Operations should still work
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory_id = db.insert(
                content="After reconnect",
                vector=vec,
            )
            assert memory_id is not None

            db.close()
