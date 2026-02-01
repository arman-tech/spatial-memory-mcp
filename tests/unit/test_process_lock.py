"""Unit tests for ProcessLockManager cross-process locking.

Tests the ProcessLockManager class that provides cross-process file locking
with reentrant support for nested database operations.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.core.database import ProcessLockManager
from spatial_memory.core.errors import FileLockError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_lock_path() -> Path:
    """Provide a temporary lock file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.lock"


@pytest.fixture
def lock_manager(temp_lock_path: Path) -> ProcessLockManager:
    """Provide an enabled ProcessLockManager."""
    return ProcessLockManager(
        lock_path=temp_lock_path,
        timeout=5.0,
        poll_interval=0.01,
        enabled=True,
    )


@pytest.fixture
def disabled_lock_manager(temp_lock_path: Path) -> ProcessLockManager:
    """Provide a disabled ProcessLockManager."""
    return ProcessLockManager(
        lock_path=temp_lock_path,
        timeout=5.0,
        poll_interval=0.01,
        enabled=False,
    )


# =============================================================================
# Disabled Mode Tests
# =============================================================================


@pytest.mark.unit
class TestDisabledMode:
    """Tests for disabled lock manager behavior."""

    def test_disabled_acquire_always_succeeds(
        self, disabled_lock_manager: ProcessLockManager
    ) -> None:
        """Disabled lock manager should always succeed on acquire."""
        assert disabled_lock_manager.acquire() is True
        assert disabled_lock_manager.acquire() is True
        assert disabled_lock_manager.acquire() is True

    def test_disabled_release_always_succeeds(
        self, disabled_lock_manager: ProcessLockManager
    ) -> None:
        """Disabled lock manager should always succeed on release."""
        assert disabled_lock_manager.release() is True
        assert disabled_lock_manager.release() is True

    def test_disabled_context_manager(
        self, disabled_lock_manager: ProcessLockManager
    ) -> None:
        """Disabled lock manager should work as context manager."""
        executed = False
        with disabled_lock_manager:
            executed = True
        assert executed

    def test_disabled_nested_context_managers(
        self, disabled_lock_manager: ProcessLockManager
    ) -> None:
        """Disabled lock manager should allow unlimited nesting."""
        depth = 0
        with disabled_lock_manager:
            depth += 1
            with disabled_lock_manager:
                depth += 1
                with disabled_lock_manager:
                    depth += 1
        assert depth == 3


# =============================================================================
# Reentrancy Tests
# =============================================================================


@pytest.mark.unit
class TestReentrancy:
    """Tests for reentrant lock behavior within same thread."""

    def test_same_thread_can_reacquire(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Same thread should be able to acquire lock multiple times."""
        # First acquisition
        assert lock_manager.acquire() is True
        assert lock_manager._get_depth() == 1

        # Second acquisition (same thread)
        assert lock_manager.acquire() is False  # Not newly acquired
        assert lock_manager._get_depth() == 2

        # Third acquisition (same thread)
        assert lock_manager.acquire() is False
        assert lock_manager._get_depth() == 3

        # Release in reverse order
        assert lock_manager.release() is False  # Still held
        assert lock_manager._get_depth() == 2

        assert lock_manager.release() is False  # Still held
        assert lock_manager._get_depth() == 1

        assert lock_manager.release() is True  # Actually released
        assert lock_manager._get_depth() == 0

    def test_nested_context_managers(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Nested context managers should work correctly."""
        depths: list[int] = []

        with lock_manager:
            depths.append(lock_manager._get_depth())
            with lock_manager:
                depths.append(lock_manager._get_depth())
                with lock_manager:
                    depths.append(lock_manager._get_depth())
                depths.append(lock_manager._get_depth())
            depths.append(lock_manager._get_depth())
        depths.append(lock_manager._get_depth())

        assert depths == [1, 2, 3, 2, 1, 0]


# =============================================================================
# Context Manager Tests
# =============================================================================


@pytest.mark.unit
class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_basic(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Context manager should acquire and release lock."""
        assert lock_manager._get_depth() == 0

        with lock_manager as lm:
            assert lm is lock_manager
            assert lock_manager._get_depth() == 1

        assert lock_manager._get_depth() == 0

    def test_context_manager_exception_releases_lock(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Lock should be released even if exception occurs."""
        with pytest.raises(ValueError):
            with lock_manager:
                assert lock_manager._get_depth() == 1
                raise ValueError("Test exception")

        # Lock should be released after exception
        assert lock_manager._get_depth() == 0

    def test_context_manager_nested_exception(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Nested locks should unwind properly on exception."""
        with pytest.raises(ValueError):
            with lock_manager:
                assert lock_manager._get_depth() == 1
                with lock_manager:
                    assert lock_manager._get_depth() == 2
                    raise ValueError("Test exception")

        assert lock_manager._get_depth() == 0


# =============================================================================
# Thread-Local Isolation Tests
# =============================================================================


@pytest.mark.unit
class TestThreadLocalIsolation:
    """Tests for thread-local depth tracking isolation."""

    def test_different_threads_have_independent_depth(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Different threads should have independent depth tracking."""
        depths: dict[str, list[int]] = {"main": [], "worker": []}
        barrier = threading.Barrier(2)

        def worker() -> None:
            # Wait for main thread to acquire
            barrier.wait()
            # Worker's depth should start at 0
            depths["worker"].append(lock_manager._get_depth())
            with lock_manager:
                depths["worker"].append(lock_manager._get_depth())
            depths["worker"].append(lock_manager._get_depth())

        worker_thread = threading.Thread(target=worker)
        worker_thread.start()

        with lock_manager:
            depths["main"].append(lock_manager._get_depth())
            barrier.wait()
            # Give worker time to check depth
            time.sleep(0.1)
            with lock_manager:
                depths["main"].append(lock_manager._get_depth())

        depths["main"].append(lock_manager._get_depth())
        worker_thread.join()

        # Main thread went: 1 -> 2 -> 0
        assert depths["main"] == [1, 2, 0]
        # Worker thread started fresh: 0 -> 1 -> 0
        assert depths["worker"] == [0, 1, 0]


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_timeout_raises_file_lock_error(self, temp_lock_path: Path) -> None:
        """Should raise FileLockError when timeout is exceeded."""
        # Create a mock that simulates timeout
        with patch("spatial_memory.core.database.FileLock") as mock_filelock_class:
            from filelock import Timeout as FileLockTimeout

            mock_lock = MagicMock()
            mock_lock.acquire.side_effect = FileLockTimeout(str(temp_lock_path))
            mock_filelock_class.return_value = mock_lock

            lock_manager = ProcessLockManager(
                lock_path=temp_lock_path,
                timeout=0.1,
                poll_interval=0.01,
                enabled=True,
            )

            with pytest.raises(FileLockError) as exc_info:
                lock_manager.acquire()

            assert str(temp_lock_path) in str(exc_info.value.lock_path)
            assert exc_info.value.timeout == 0.1

    def test_release_without_acquire_is_safe(
        self, lock_manager: ProcessLockManager
    ) -> None:
        """Releasing without acquiring should be safe (no-op)."""
        assert lock_manager._get_depth() == 0
        result = lock_manager.release()
        assert result is True
        assert lock_manager._get_depth() == 0

    def test_fallback_to_disabled_on_lock_creation_error(
        self, temp_lock_path: Path
    ) -> None:
        """Should fallback to disabled mode if lock file cannot be created."""
        with patch("spatial_memory.core.database.FileLock") as mock_filelock_class:
            mock_filelock_class.side_effect = OSError("Permission denied")

            lock_manager = ProcessLockManager(
                lock_path=temp_lock_path,
                timeout=5.0,
                poll_interval=0.01,
                enabled=True,
            )

            # Should fallback to disabled mode
            assert lock_manager.enabled is False
            assert lock_manager._lock is None

            # Operations should still work (as no-ops)
            assert lock_manager.acquire() is True
            assert lock_manager.release() is True


# =============================================================================
# Integration with Database Decorator Tests
# =============================================================================


@pytest.mark.unit
class TestWithProcessLockDecorator:
    """Tests for the with_process_lock decorator integration."""

    def test_decorator_passes_through_when_lock_is_none(self) -> None:
        """Decorator should pass through when _process_lock is None."""
        from spatial_memory.core.database import with_process_lock

        class MockDatabase:
            _process_lock = None
            call_count = 0

            @with_process_lock
            def write_operation(self) -> str:
                self.call_count += 1
                return "success"

        db = MockDatabase()
        result = db.write_operation()

        assert result == "success"
        assert db.call_count == 1

    def test_decorator_acquires_lock_when_present(
        self, temp_lock_path: Path
    ) -> None:
        """Decorator should acquire lock when _process_lock is set."""
        from spatial_memory.core.database import with_process_lock

        lock_manager = ProcessLockManager(
            lock_path=temp_lock_path,
            timeout=5.0,
            enabled=True,
        )

        class MockDatabase:
            _process_lock = lock_manager
            depths_during_call: list[int] = []

            @with_process_lock
            def write_operation(self) -> str:
                self.depths_during_call.append(self._process_lock._get_depth())
                return "success"

        db = MockDatabase()

        # Before call, depth is 0
        assert lock_manager._get_depth() == 0

        result = db.write_operation()

        # During call, depth was 1
        assert db.depths_during_call == [1]

        # After call, depth is back to 0
        assert lock_manager._get_depth() == 0
        assert result == "success"

    def test_decorator_handles_nested_calls(self, temp_lock_path: Path) -> None:
        """Decorator should handle nested calls via reentrancy."""
        from spatial_memory.core.database import with_process_lock

        lock_manager = ProcessLockManager(
            lock_path=temp_lock_path,
            timeout=5.0,
            enabled=True,
        )

        class MockDatabase:
            _process_lock = lock_manager
            depths: list[int] = []

            @with_process_lock
            def outer_operation(self) -> str:
                self.depths.append(self._process_lock._get_depth())
                return self.inner_operation()

            @with_process_lock
            def inner_operation(self) -> str:
                self.depths.append(self._process_lock._get_depth())
                return "inner_done"

        db = MockDatabase()
        result = db.outer_operation()

        assert result == "inner_done"
        assert db.depths == [1, 2]  # Nested lock depths
        assert lock_manager._get_depth() == 0  # Released after


# =============================================================================
# Enabled Property Tests
# =============================================================================


@pytest.mark.unit
class TestEnabledProperty:
    """Tests for enabled/disabled state handling."""

    def test_enabled_true_creates_lock(self, temp_lock_path: Path) -> None:
        """Enabled=True should create the FileLock."""
        lock_manager = ProcessLockManager(
            lock_path=temp_lock_path,
            timeout=5.0,
            enabled=True,
        )
        assert lock_manager.enabled is True
        assert lock_manager._lock is not None

    def test_enabled_false_does_not_create_lock(
        self, temp_lock_path: Path
    ) -> None:
        """Enabled=False should not create the FileLock."""
        lock_manager = ProcessLockManager(
            lock_path=temp_lock_path,
            timeout=5.0,
            enabled=False,
        )
        assert lock_manager.enabled is False
        assert lock_manager._lock is None
