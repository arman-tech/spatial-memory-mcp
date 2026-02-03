"""Circuit breaker pattern implementation for fault tolerance.

The circuit breaker prevents cascading failures by fast-failing requests
when a service is unhealthy, allowing time for recovery.

State transitions:
    CLOSED (normal) -> OPEN (failures >= threshold)
    OPEN -> HALF_OPEN (after reset_timeout)
    HALF_OPEN -> CLOSED (probe succeeds)
    HALF_OPEN -> OPEN (probe fails)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states.

    CLOSED: Normal operation, requests pass through.
    OPEN: Circuit is tripped, requests are rejected immediately.
    HALF_OPEN: Testing recovery, limited requests allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected.

    This exception indicates that the circuit breaker is preventing
    requests from reaching a failing service.
    """

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        time_until_retry: float | None = None,
    ) -> None:
        """Initialize CircuitOpenError.

        Args:
            message: Error description.
            time_until_retry: Seconds until the circuit will transition to HALF_OPEN.
        """
        super().__init__(message)
        self.time_until_retry = time_until_retry


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    Monitors failures and opens the circuit when failures exceed a threshold,
    preventing further calls until a reset timeout has elapsed.

    Example:
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)

        try:
            result = breaker.call(my_api_call, arg1, arg2)
        except CircuitOpenError:
            # Service is unhealthy, use fallback
            result = fallback_value

    Thread Safety:
        This class is thread-safe. All state transitions and counters
        are protected by a lock.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        name: str | None = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit.
            reset_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN.
            half_open_max_calls: Maximum concurrent calls allowed in HALF_OPEN state.
            name: Optional name for logging purposes.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be positive")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")

        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls
        self._name = name or "circuit_breaker"

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state.

        This property also handles automatic transition from OPEN to HALF_OPEN
        when the reset timeout has elapsed.

        Returns:
            Current circuit state.
        """
        with self._lock:
            return self._get_state_unlocked()

    def _get_state_unlocked(self) -> CircuitState:
        """Get state without acquiring lock (must be called with lock held)."""
        if self._state == CircuitState.OPEN:
            if self._should_transition_to_half_open():
                self._transition_to_half_open()
        return self._state

    def _should_transition_to_half_open(self) -> bool:
        """Check if reset timeout has elapsed (must be called with lock held)."""
        if self._last_failure_time is None:
            return False
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self._reset_timeout

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state (must be called with lock held)."""
        logger.info(f"[{self._name}] Circuit transitioning from OPEN to HALF_OPEN")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count

    @property
    def stats(self) -> dict[str, int | str | float | None]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with state, counters, and timing information.
        """
        with self._lock:
            state = self._get_state_unlocked()
            time_until_retry = None
            if state == CircuitState.OPEN and self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                time_until_retry = max(0.0, self._reset_timeout - elapsed)

            return {
                "state": state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self._failure_threshold,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_rejections": self._total_rejections,
                "reset_timeout": self._reset_timeout,
                "time_until_retry": time_until_retry,
            }

    def call(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Return value of the function.

        Raises:
            CircuitOpenError: If circuit is OPEN and not ready for retry.
            Exception: Any exception raised by the function.
        """
        with self._lock:
            self._total_calls += 1
            current_state = self._get_state_unlocked()

            if current_state == CircuitState.OPEN:
                self._total_rejections += 1
                time_until_retry = None
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    time_until_retry = max(0.0, self._reset_timeout - elapsed)
                raise CircuitOpenError(
                    f"[{self._name}] Circuit is OPEN, rejecting call",
                    time_until_retry=time_until_retry,
                )

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    self._total_rejections += 1
                    raise CircuitOpenError(
                        f"[{self._name}] Circuit is HALF_OPEN, max probe calls reached",
                        time_until_retry=0.0,
                    )
                self._half_open_calls += 1

        # Execute function outside the lock
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        """Handle successful call.

        In CLOSED state: Reset failure count.
        In HALF_OPEN state: Transition to CLOSED.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"[{self._name}] Probe succeeded, circuit transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call.

        In CLOSED state: Increment failure count, open circuit if threshold reached.
        In HALF_OPEN state: Transition back to OPEN.

        Args:
            error: The exception that was raised.
        """
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"[{self._name}] Probe failed ({error!r}), circuit transitioning back to OPEN"
                )
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    logger.warning(
                        f"[{self._name}] Failure threshold reached "
                        f"({self._failure_count}/{self._failure_threshold}), "
                        f"circuit transitioning to OPEN"
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state.

        This clears all failure counters and transitions the circuit
        to CLOSED state regardless of current state.
        """
        with self._lock:
            logger.info(f"[{self._name}] Circuit manually reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CircuitBreaker(name={self._name!r}, state={self.state.value}, "
            f"failures={self.failure_count}/{self._failure_threshold})"
        )
