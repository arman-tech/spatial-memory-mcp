"""Tests for rate limiter."""

import time

import pytest

from spatial_memory.core.rate_limiter import RateLimiter


class TestRateLimiterBasics:
    """Test basic rate limiter functionality."""

    def test_initialization(self) -> None:
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(rate=10.0, capacity=20)
        assert limiter.rate == 10.0
        assert limiter.capacity == 20

        stats = limiter.stats()
        assert stats["rate"] == 10.0
        assert stats["capacity"] == 20
        assert stats["tokens_available"] == 20

    def test_default_capacity(self) -> None:
        """Test default capacity is rate * 2."""
        limiter = RateLimiter(rate=10.0)
        assert limiter.capacity == 20

    def test_invalid_rate(self) -> None:
        """Test invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be positive"):
            RateLimiter(rate=0.0)

        with pytest.raises(ValueError, match="rate must be positive"):
            RateLimiter(rate=-1.0)


class TestRateLimiterAcquire:
    """Test non-blocking acquire."""

    def test_acquire_single_token(self) -> None:
        """Test acquiring a single token."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        assert limiter.acquire() is True

        stats = limiter.stats()
        assert 8.9 <= stats["tokens_available"] <= 9.1  # Allow for timing variance

    def test_acquire_multiple_tokens(self) -> None:
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        assert limiter.acquire(tokens=5) is True

        stats = limiter.stats()
        assert 4.9 <= stats["tokens_available"] <= 5.1  # Allow for timing variance

    def test_acquire_fails_when_insufficient(self) -> None:
        """Test acquire returns False when insufficient tokens."""
        limiter = RateLimiter(rate=10.0, capacity=10)

        # Consume all tokens
        assert limiter.acquire(tokens=10) is True

        # Next acquire should fail
        assert limiter.acquire() is False

        stats = limiter.stats()
        assert stats["tokens_available"] < 0.1  # Nearly zero, allow for timing variance

    def test_tokens_refill_over_time(self) -> None:
        """Test tokens refill over time."""
        limiter = RateLimiter(rate=10.0, capacity=10)

        # Consume all tokens
        assert limiter.acquire(tokens=10) is True
        assert limiter.acquire() is False

        # Wait for refill (10 tokens/sec = 0.1 sec per token)
        time.sleep(0.15)  # Should refill ~1.5 tokens

        # Should be able to acquire 1 token now
        assert limiter.acquire() is True
        assert limiter.acquire() is False  # But not 2

    def test_capacity_cap(self) -> None:
        """Test tokens don't exceed capacity."""
        limiter = RateLimiter(rate=100.0, capacity=10)

        # Wait long enough to refill many tokens
        time.sleep(0.2)  # Would refill 20 tokens without cap

        # Should still only have capacity worth
        stats = limiter.stats()
        assert stats["tokens_available"] <= 10


class TestRateLimiterWait:
    """Test blocking wait."""

    def test_wait_immediate_success(self) -> None:
        """Test wait returns immediately when tokens available."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        start = time.monotonic()
        assert limiter.wait() is True
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should be nearly instant

    def test_wait_blocks_until_available(self) -> None:
        """Test wait blocks until tokens are available."""
        limiter = RateLimiter(rate=10.0, capacity=10)

        # Consume all tokens
        limiter.acquire(tokens=10)

        # Wait should block for ~0.1 seconds (1 token at 10/sec)
        start = time.monotonic()
        assert limiter.wait() is True
        elapsed = time.monotonic() - start
        assert 0.08 < elapsed < 0.15  # Allow some variance

    def test_wait_with_timeout_success(self) -> None:
        """Test wait with timeout succeeds when tokens available in time."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        limiter.acquire(tokens=10)

        # Wait with generous timeout
        assert limiter.wait(timeout=0.5) is True

    def test_wait_with_timeout_failure(self) -> None:
        """Test wait with timeout returns False when timeout exceeded."""
        limiter = RateLimiter(rate=1.0, capacity=10)  # Slow rate
        limiter.acquire(tokens=10)

        # Wait with short timeout - should timeout before refill
        start = time.monotonic()
        assert limiter.wait(timeout=0.05) is False
        elapsed = time.monotonic() - start
        assert elapsed < 0.15  # Should timeout around 0.05, allow for variance

    def test_wait_multiple_tokens(self) -> None:
        """Test wait can acquire multiple tokens."""
        limiter = RateLimiter(rate=20.0, capacity=10)
        limiter.acquire(tokens=10)

        # Wait for 5 tokens (20/sec = 0.25 sec for 5 tokens)
        start = time.monotonic()
        assert limiter.wait(tokens=5) is True
        elapsed = time.monotonic() - start
        assert 0.2 < elapsed < 0.35


class TestRateLimiterBurstTraffic:
    """Test rate limiter with burst traffic."""

    def test_burst_within_capacity(self) -> None:
        """Test burst traffic within capacity succeeds."""
        limiter = RateLimiter(rate=10.0, capacity=20)

        # Burst of 15 requests (within capacity)
        for _ in range(15):
            assert limiter.acquire() is True

        # Next 5 should succeed
        for _ in range(5):
            assert limiter.acquire() is True

        # Now exhausted
        assert limiter.acquire() is False

    def test_burst_exceeds_capacity(self) -> None:
        """Test burst traffic exceeding capacity is rate limited."""
        limiter = RateLimiter(rate=10.0, capacity=10)

        # First 10 succeed (burst capacity)
        for _ in range(10):
            assert limiter.acquire() is True

        # Next requests fail until refill
        assert limiter.acquire() is False

    def test_steady_state_rate(self) -> None:
        """Test steady-state rate approaches configured rate."""
        limiter = RateLimiter(rate=20.0, capacity=40)

        # Consume all initial tokens
        limiter.acquire(tokens=40)

        # Now measure steady-state rate
        start = time.monotonic()
        count = 0
        duration = 0.5  # Test for 0.5 seconds

        while time.monotonic() - start < duration:
            if limiter.wait(timeout=0.05):
                count += 1

        elapsed = time.monotonic() - start
        actual_rate = count / elapsed

        # Should be close to 20/sec (allow 20% variance for timing)
        assert 16 < actual_rate < 24


class TestRateLimiterThreadSafety:
    """Test rate limiter thread safety."""

    def test_concurrent_acquire(self) -> None:
        """Test concurrent acquire calls are thread-safe."""
        import threading

        limiter = RateLimiter(rate=100.0, capacity=100)
        success_count = []
        lock = threading.Lock()

        def worker() -> None:
            if limiter.acquire():
                with lock:
                    success_count.append(1)

        threads = [threading.Thread(target=worker) for _ in range(150)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have acquired ~100 tokens (capacity), allow for some refill during test
        assert 99 <= len(success_count) <= 105

    def test_concurrent_wait(self) -> None:
        """Test concurrent wait calls are thread-safe."""
        import threading

        limiter = RateLimiter(rate=50.0, capacity=50)
        success_count = []
        lock = threading.Lock()

        def worker() -> None:
            if limiter.wait(timeout=1.0):
                with lock:
                    success_count.append(1)

        threads = [threading.Thread(target=worker) for _ in range(75)]

        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        # All should eventually succeed (within timeout)
        assert len(success_count) == 75

        # Should take roughly 0.5 seconds (75 tokens at 50/sec = 1.5s, but we start with 50)
        # 50 instant + 25 more at 50/sec = 0.5s
        assert 0.4 < elapsed < 1.0


class TestRateLimiterStats:
    """Test rate limiter statistics."""

    def test_stats_updates(self) -> None:
        """Test stats reflect current state."""
        limiter = RateLimiter(rate=10.0, capacity=20)

        # Initial state
        stats = limiter.stats()
        assert stats["tokens_available"] == 20

        # After acquire
        limiter.acquire(tokens=5)
        stats = limiter.stats()
        assert 14.9 <= stats["tokens_available"] <= 15.1  # Allow for timing variance

        # After refill
        time.sleep(0.2)  # Refill ~2 tokens
        stats = limiter.stats()
        assert 16 < stats["tokens_available"] < 18
