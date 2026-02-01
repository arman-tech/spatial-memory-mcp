"""Unit tests for CircuitBreaker.

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure counting
- Reset timeout behavior
- Thread safety
- CircuitOpenError raised when open
- Successful recovery after half-open probe
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.core.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_states_exist(self) -> None:
        """CircuitState should have CLOSED, OPEN, and HALF_OPEN states."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_default_message(self) -> None:
        """CircuitOpenError should have a default message."""
        error = CircuitOpenError()
        assert "open" in str(error).lower()

    def test_circuit_open_error_custom_message(self) -> None:
        """CircuitOpenError should accept custom message."""
        error = CircuitOpenError("Custom error message")
        assert str(error) == "Custom error message"

    def test_circuit_open_error_time_until_retry(self) -> None:
        """CircuitOpenError should store time_until_retry."""
        error = CircuitOpenError("Test", time_until_retry=30.5)
        assert error.time_until_retry == 30.5

    def test_circuit_open_error_is_exception(self) -> None:
        """CircuitOpenError should be an Exception."""
        assert issubclass(CircuitOpenError, Exception)


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_default_initialization(self) -> None:
        """CircuitBreaker should initialize with default values."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_custom_parameters(self) -> None:
        """CircuitBreaker should accept custom parameters."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0,
            half_open_max_calls=2,
            name="test_breaker",
        )
        stats = breaker.stats
        assert stats["failure_threshold"] == 3
        assert stats["reset_timeout"] == 30.0

    def test_invalid_failure_threshold(self) -> None:
        """CircuitBreaker should reject invalid failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=0)
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=-1)

    def test_invalid_reset_timeout(self) -> None:
        """CircuitBreaker should reject invalid reset_timeout."""
        with pytest.raises(ValueError, match="reset_timeout"):
            CircuitBreaker(reset_timeout=0)
        with pytest.raises(ValueError, match="reset_timeout"):
            CircuitBreaker(reset_timeout=-1.0)

    def test_invalid_half_open_max_calls(self) -> None:
        """CircuitBreaker should reject invalid half_open_max_calls."""
        with pytest.raises(ValueError, match="half_open_max_calls"):
            CircuitBreaker(half_open_max_calls=0)

    def test_repr(self) -> None:
        """CircuitBreaker should have informative repr."""
        breaker = CircuitBreaker(name="my_service", failure_threshold=5)
        repr_str = repr(breaker)
        assert "my_service" in repr_str
        assert "closed" in repr_str
        assert "0/5" in repr_str


class TestCircuitBreakerClosedState:
    """Tests for CircuitBreaker in CLOSED state."""

    def test_successful_call_returns_result(self) -> None:
        """Successful call should return function result."""
        breaker = CircuitBreaker()

        def success_func() -> str:
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call_with_args(self) -> None:
        """Successful call should pass arguments correctly."""
        breaker = CircuitBreaker()

        def add(a: int, b: int) -> int:
            return a + b

        result = breaker.call(add, 2, 3)
        assert result == 5

    def test_successful_call_with_kwargs(self) -> None:
        """Successful call should pass keyword arguments correctly."""
        breaker = CircuitBreaker()

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = breaker.call(greet, "World", greeting="Hi")
        assert result == "Hi, World!"

    def test_failure_increments_count(self) -> None:
        """Failed call should increment failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        def fail_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self) -> None:
        """Successful call should reset failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        def fail_func() -> None:
            raise ValueError("Test error")

        def success_func() -> str:
            return "ok"

        # Accumulate some failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.failure_count == 3

        # Success should reset count
        breaker.call(success_func)
        assert breaker.failure_count == 0


class TestCircuitBreakerOpenTransition:
    """Tests for CLOSED -> OPEN transition."""

    def test_opens_at_threshold(self) -> None:
        """Circuit should open when failure threshold is reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        def fail_func() -> None:
            raise ValueError("Test error")

        # First two failures keep circuit closed
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
            assert breaker.state == CircuitState.CLOSED

        # Third failure opens the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    def test_open_rejects_calls(self) -> None:
        """Open circuit should reject calls with CircuitOpenError."""
        breaker = CircuitBreaker(failure_threshold=2)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Subsequent calls should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(fail_func)

        assert exc_info.value.time_until_retry is not None

    def test_open_provides_time_until_retry(self) -> None:
        """CircuitOpenError should include time until retry."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Immediate rejection should have ~60s until retry
        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(fail_func)

        assert exc_info.value.time_until_retry is not None
        # Allow some tolerance for test execution time
        assert 59.0 <= exc_info.value.time_until_retry <= 60.0


class TestCircuitBreakerHalfOpenTransition:
    """Tests for OPEN -> HALF_OPEN transition."""

    def test_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit should transition to HALF_OPEN after reset_timeout."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # State should now be HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN

    @patch("time.monotonic")
    def test_half_open_transition_mocked_time(
        self, mock_monotonic: MagicMock
    ) -> None:
        """Circuit should transition to HALF_OPEN based on monotonic time."""
        mock_monotonic.return_value = 1000.0

        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit at t=1000
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

        # Still OPEN at t=1059
        mock_monotonic.return_value = 1059.0
        assert breaker.state == CircuitState.OPEN

        # HALF_OPEN at t=1060
        mock_monotonic.return_value = 1060.0
        assert breaker.state == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpenState:
    """Tests for CircuitBreaker in HALF_OPEN state."""

    def test_half_open_allows_probe_calls(self) -> None:
        """HALF_OPEN state should allow limited probe calls."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.05,
            half_open_max_calls=1,
        )

        def fail_func() -> None:
            raise ValueError("Test error")

        def success_func() -> str:
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for half-open
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        # First probe call should be allowed
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_rejects_excess_calls(self) -> None:
        """HALF_OPEN state should reject calls beyond max_calls."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.05,
            half_open_max_calls=1,
        )

        call_count = 0

        def slow_success() -> str:
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate slow call
            return "success"

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for half-open
        time.sleep(0.1)

        # Start first probe in a thread
        def first_probe() -> None:
            breaker.call(slow_success)

        thread = threading.Thread(target=first_probe)
        thread.start()

        # Give time for first probe to start
        time.sleep(0.02)

        # Second call should be rejected
        with pytest.raises(CircuitOpenError):
            breaker.call(slow_success)

        thread.join()
        assert call_count == 1  # Only one call was allowed

    def test_half_open_success_closes_circuit(self) -> None:
        """Successful probe in HALF_OPEN should close circuit."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

        def fail_func() -> None:
            raise ValueError("Test error")

        def success_func() -> str:
            return "recovered"

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for half-open
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful probe
        result = breaker.call(success_func)
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Failed probe in HALF_OPEN should reopen circuit."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for half-open
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failed probe
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Tests for manual circuit reset."""

    def test_reset_from_open(self) -> None:
        """reset() should transition from OPEN to CLOSED."""
        breaker = CircuitBreaker(failure_threshold=1)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_reset_from_half_open(self) -> None:
        """reset() should transition from HALF_OPEN to CLOSED."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Wait for half-open
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self) -> None:
        """reset() should clear failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Accumulate failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.failure_count == 3

        # Reset
        breaker.reset()
        assert breaker.failure_count == 0


class TestCircuitBreakerStatistics:
    """Tests for circuit breaker statistics."""

    def test_stats_structure(self) -> None:
        """stats should return complete statistics."""
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        stats = breaker.stats

        assert "state" in stats
        assert "failure_count" in stats
        assert "failure_threshold" in stats
        assert "total_calls" in stats
        assert "total_failures" in stats
        assert "total_rejections" in stats
        assert "reset_timeout" in stats
        assert "time_until_retry" in stats

    def test_stats_track_calls(self) -> None:
        """stats should track total calls."""
        breaker = CircuitBreaker(failure_threshold=5)

        def success_func() -> str:
            return "ok"

        for _ in range(3):
            breaker.call(success_func)

        assert breaker.stats["total_calls"] == 3

    def test_stats_track_failures(self) -> None:
        """stats should track total failures."""
        breaker = CircuitBreaker(failure_threshold=5)

        def fail_func() -> None:
            raise ValueError("Test error")

        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.stats["total_failures"] == 3

    def test_stats_track_rejections(self) -> None:
        """stats should track rejections when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=2)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        # Try to call when open (rejected)
        for _ in range(3):
            with pytest.raises(CircuitOpenError):
                breaker.call(fail_func)

        assert breaker.stats["total_rejections"] == 3

    def test_stats_time_until_retry_when_open(self) -> None:
        """stats should show time_until_retry when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)

        def fail_func() -> None:
            raise ValueError("Test error")

        # Open circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        stats = breaker.stats
        assert stats["time_until_retry"] is not None
        assert 59.0 <= stats["time_until_retry"] <= 60.0

    def test_stats_time_until_retry_none_when_closed(self) -> None:
        """stats should show time_until_retry as None when closed."""
        breaker = CircuitBreaker()
        assert breaker.stats["time_until_retry"] is None


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety of CircuitBreaker."""

    def test_concurrent_calls(self) -> None:
        """Circuit breaker should handle concurrent calls safely."""
        breaker = CircuitBreaker(failure_threshold=100)
        results: list[str] = []
        lock = threading.Lock()

        def success_func(n: int) -> str:
            result = f"result-{n}"
            with lock:
                results.append(result)
            return result

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(breaker.call, success_func, i) for i in range(50)]
            for future in futures:
                future.result()

        assert len(results) == 50
        assert breaker.stats["total_calls"] == 50

    def test_concurrent_failures(self) -> None:
        """Circuit breaker should handle concurrent failures safely."""
        breaker = CircuitBreaker(failure_threshold=10)
        failure_exceptions: list[Exception] = []
        open_errors: list[CircuitOpenError] = []
        lock = threading.Lock()

        def fail_func() -> None:
            raise ValueError("Concurrent failure")

        def try_call() -> None:
            try:
                breaker.call(fail_func)
            except CircuitOpenError as e:
                with lock:
                    open_errors.append(e)
            except ValueError as e:
                with lock:
                    failure_exceptions.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(try_call) for _ in range(50)]
            for future in futures:
                future.result()

        # Some calls should have failed, some should have been rejected
        total = len(failure_exceptions) + len(open_errors)
        assert total == 50
        # At least some should have been rejected (circuit should have opened)
        assert len(open_errors) > 0

    def test_state_consistency_under_load(self) -> None:
        """Circuit state should remain consistent under concurrent access."""
        breaker = CircuitBreaker(failure_threshold=5)
        states_seen: list[CircuitState] = []
        lock = threading.Lock()

        def record_state() -> None:
            state = breaker.state
            with lock:
                states_seen.append(state)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_state) for _ in range(100)]
            for future in futures:
                future.result()

        # All states should be valid
        for state in states_seen:
            assert state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)


class TestCircuitBreakerRecovery:
    """Tests for circuit recovery scenarios."""

    def test_full_recovery_cycle(self) -> None:
        """Test complete recovery: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        call_count = 0

        def fail_func() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Service unavailable")

        def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "recovered"

        # 1. Start in CLOSED
        assert breaker.state == CircuitState.CLOSED

        # 2. Failures open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # 3. Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # 4. Successful probe closes circuit
        result = breaker.call(success_func)
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED

    def test_failed_recovery_cycle(self) -> None:
        """Test failed recovery: CLOSED -> OPEN -> HALF_OPEN -> OPEN."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        def fail_func() -> None:
            raise ValueError("Still failing")

        # 1. Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

        # 2. Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # 3. Failed probe reopens circuit
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        assert breaker.state == CircuitState.OPEN

    def test_multiple_recovery_attempts(self) -> None:
        """Test multiple recovery attempts before success."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
        attempt = 0

        def sometimes_fail() -> str:
            nonlocal attempt
            attempt += 1
            # Fail on attempts 1, 2, 3 (initial + 2 probes), succeed on attempt 4
            if attempt < 4:
                raise ValueError(f"Attempt {attempt} failed")
            return "finally recovered"

        # Open circuit (attempt 1)
        with pytest.raises(ValueError):
            breaker.call(sometimes_fail)
        assert breaker.state == CircuitState.OPEN

        # First recovery attempt fails (attempt 2)
        time.sleep(0.1)
        with pytest.raises(ValueError):
            breaker.call(sometimes_fail)
        assert breaker.state == CircuitState.OPEN

        # Second recovery attempt fails (attempt 3)
        time.sleep(0.1)
        with pytest.raises(ValueError):
            breaker.call(sometimes_fail)
        assert breaker.state == CircuitState.OPEN

        # Third recovery attempt succeeds (attempt 4)
        time.sleep(0.1)
        result = breaker.call(sometimes_fail)
        assert result == "finally recovered"
        assert breaker.state == CircuitState.CLOSED
