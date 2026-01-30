"""Token bucket rate limiter for API calls."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter.

    Limits the rate of operations using a token bucket algorithm:
    - Bucket holds up to `capacity` tokens
    - Tokens are added at `rate` per second
    - Each operation consumes tokens

    Example:
        limiter = RateLimiter(rate=10.0, capacity=20)  # 10 ops/sec, burst of 20
        if limiter.acquire():
            # perform operation
        else:
            # rate limited, try again later

        # Or blocking wait:
        limiter.wait()  # waits until token available
        # perform operation
    """

    def __init__(self, rate: float, capacity: int | None = None) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Tokens added per second.
            capacity: Maximum tokens in bucket (default: rate * 2).
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity or int(rate * 2)
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait (None = no limit).

        Returns:
            True if tokens were acquired, False if timeout.
        """
        start = time.monotonic()
        while True:
            if self.acquire(tokens):
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return False

            # Sleep for estimated time to get a token
            with self._lock:
                wait_time = (tokens - self._tokens) / self.rate
            time.sleep(min(wait_time, 0.1))  # Cap at 100ms to check timeout

    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "tokens_available": self._tokens,
                "capacity": self.capacity,
                "rate": self.rate,
            }
