"""Tests for rate limiter."""

import asyncio
import time

import pytest

from spatial_memory.core.rate_limiter import AgentAwareRateLimiter, RateLimiter

# =============================================================================
# Sync Tests — basics, acquire, burst, thread safety, stats
# =============================================================================


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
        time.sleep(0.25)  # Should refill ~2.5 tokens (extra buffer for slow CI)

        # Should be able to acquire 1 token now
        assert limiter.acquire() is True
        # Don't assert second acquire fails - timing variance makes this flaky

    def test_capacity_cap(self) -> None:
        """Test tokens don't exceed capacity."""
        limiter = RateLimiter(rate=100.0, capacity=10)

        # Wait long enough to refill many tokens
        time.sleep(0.2)  # Would refill 20 tokens without cap

        # Should still only have capacity worth
        stats = limiter.stats()
        assert stats["tokens_available"] <= 10


class TestTimeUntilAvailable:
    """Test time_until_available() deficit estimation."""

    def test_returns_zero_when_tokens_available(self) -> None:
        """time_until_available returns 0.0 when tokens are available."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        assert limiter.time_until_available() == 0.0
        assert limiter.time_until_available(tokens=10) == 0.0

    def test_returns_correct_deficit(self) -> None:
        """time_until_available returns estimated seconds for deficit."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        limiter.acquire(tokens=10)

        estimate = limiter.time_until_available(tokens=1)
        # Need 1 token at 10/sec = 0.1 seconds
        assert 0.05 < estimate < 0.15

    def test_agent_aware_returns_max_of_both(self) -> None:
        """AgentAwareRateLimiter returns the worse of global and per-agent."""
        limiter = AgentAwareRateLimiter(global_rate=10.0, per_agent_rate=5.0, max_agents=5)
        # Exhaust global tokens
        for _ in range(20):
            limiter._global.acquire()

        estimate = limiter.time_until_available(agent_id="test", tokens=1)
        # Global is exhausted, per-agent is full — should reflect global deficit
        assert estimate > 0.0

    def test_agent_aware_without_agent_id(self) -> None:
        """AgentAwareRateLimiter without agent_id uses global only."""
        limiter = AgentAwareRateLimiter(global_rate=10.0, per_agent_rate=5.0)
        assert limiter.time_until_available() == 0.0


# =============================================================================
# Async Tests — wait() behavior
# =============================================================================


class TestRateLimiterWait:
    """Test async wait."""

    @pytest.mark.asyncio
    async def test_wait_immediate_success(self) -> None:
        """Test wait returns immediately when tokens available."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        start = time.monotonic()
        assert await limiter.wait() is True
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_wait_blocks_until_available(self) -> None:
        """Test wait yields until tokens are available."""
        limiter = RateLimiter(rate=10.0, capacity=10)

        # Consume all tokens
        limiter.acquire(tokens=10)

        # Wait should yield for ~0.1 seconds (1 token at 10/sec)
        start = time.monotonic()
        assert await limiter.wait() is True
        elapsed = time.monotonic() - start
        assert 0.06 < elapsed < 0.30  # Allow for CI timing variance

    @pytest.mark.asyncio
    async def test_wait_with_timeout_success(self) -> None:
        """Test wait with timeout succeeds when tokens available in time."""
        limiter = RateLimiter(rate=10.0, capacity=10)
        limiter.acquire(tokens=10)

        # Wait with generous timeout
        assert await limiter.wait(timeout=0.5) is True

    @pytest.mark.asyncio
    async def test_wait_with_timeout_failure(self) -> None:
        """Test wait with timeout returns False when timeout exceeded."""
        limiter = RateLimiter(rate=1.0, capacity=10)  # Slow rate
        limiter.acquire(tokens=10)

        # Wait with short timeout - should timeout before refill
        start = time.monotonic()
        assert await limiter.wait(timeout=0.05) is False
        elapsed = time.monotonic() - start
        assert elapsed < 0.35  # Should timeout around 0.05, allow for CI variance

    @pytest.mark.asyncio
    async def test_wait_multiple_tokens(self) -> None:
        """Test wait can acquire multiple tokens."""
        limiter = RateLimiter(rate=20.0, capacity=10)
        limiter.acquire(tokens=10)

        # Wait for 5 tokens (20/sec = 0.25 sec for 5 tokens)
        start = time.monotonic()
        assert await limiter.wait(tokens=5) is True
        elapsed = time.monotonic() - start
        assert 0.15 < elapsed < 0.50  # Allow for CI timing variance


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

    @pytest.mark.asyncio
    async def test_steady_state_rate(self) -> None:
        """Test steady-state rate approaches configured rate."""
        limiter = RateLimiter(rate=20.0, capacity=40)

        # Consume all initial tokens
        limiter.acquire(tokens=40)

        # Now measure steady-state rate
        start = time.monotonic()
        count = 0
        duration = 0.5  # Test for 0.5 seconds

        while time.monotonic() - start < duration:
            if await limiter.wait(timeout=0.05):
                count += 1

        elapsed = time.monotonic() - start
        actual_rate = count / elapsed

        # Should be close to 20/sec (allow wider variance for CI timing)
        assert 12 < actual_rate < 28


class TestRateLimiterThreadSafety:
    """Test rate limiter thread safety for sync operations."""

    def test_concurrent_acquire(self) -> None:
        """Test concurrent acquire calls are thread-safe."""
        import threading

        limiter = RateLimiter(rate=100.0, capacity=100)
        success_count: list[int] = []
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
        # Range widened to account for CI timing variations
        assert 95 <= len(success_count) <= 115


class TestRateLimiterConcurrentWait:
    """Test async concurrent wait behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_wait(self) -> None:
        """Test concurrent async wait calls all succeed."""
        limiter = RateLimiter(rate=50.0, capacity=50)
        success_count: list[int] = []

        async def worker() -> None:
            if await limiter.wait(timeout=2.0):
                success_count.append(1)

        start = time.monotonic()
        await asyncio.gather(*[worker() for _ in range(75)])
        elapsed = time.monotonic() - start

        # All should eventually succeed (within timeout)
        assert len(success_count) == 75

        # 50 instant + 25 more at 50/sec = 0.5s
        assert 0.3 < elapsed < 1.5


class TestRateLimiterStats:
    """Test rate limiter statistics."""

    @pytest.mark.asyncio
    async def test_stats_updates(self) -> None:
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
        await asyncio.sleep(0.2)  # Refill ~2 tokens
        stats = limiter.stats()
        assert 15 < stats["tokens_available"] < 20  # Allow for CI timing variance


# =============================================================================
# Async Edge Case Tests
# =============================================================================


class TestAsyncEdgeCases:
    """Edge cases for async wait() behavior."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_waits(self) -> None:
        """Multiple coroutines waiting simultaneously all succeed."""
        limiter = RateLimiter(rate=100.0, capacity=5)
        limiter.acquire(tokens=5)  # Exhaust capacity

        results: list[bool] = []

        async def acquire_one() -> None:
            result = await limiter.wait(timeout=2.0)
            results.append(result)

        start = time.monotonic()
        await asyncio.gather(*[acquire_one() for _ in range(15)])
        elapsed = time.monotonic() - start

        assert all(results)
        assert len(results) == 15
        # 15 tokens at 100/sec = 0.15s, allow CI variance
        assert 0.05 < elapsed < 0.50

    @pytest.mark.asyncio
    async def test_wait_cancellation(self) -> None:
        """Cancelled wait does not consume tokens; limiter works after."""
        limiter = RateLimiter(rate=1.0, capacity=1)
        limiter.acquire(tokens=1)  # Exhaust

        async def long_wait() -> bool:
            return await limiter.wait(timeout=10.0)

        task = asyncio.create_task(long_wait())
        await asyncio.sleep(0.05)  # Let it start waiting
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # After cancellation, limiter should still work correctly
        # Wait for token refill (1 token/sec)
        await asyncio.sleep(1.1)
        result = await limiter.wait(timeout=0.5)
        assert result is True

    @pytest.mark.asyncio
    async def test_timeout_precision_async(self) -> None:
        """Timeout returns in approximately expected time."""
        limiter = RateLimiter(rate=0.5, capacity=1)  # Very slow rate
        limiter.acquire(tokens=1)

        start = time.monotonic()
        result = await limiter.wait(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        # Should be close to 0.1s (allow wider tolerance for CI)
        assert 0.05 < elapsed < 0.25

    @pytest.mark.asyncio
    async def test_mixed_acquire_and_wait(self) -> None:
        """Concurrent sync acquire() + async wait() don't corrupt state."""
        limiter = RateLimiter(rate=100.0, capacity=100)

        acquired: list[int] = []
        waited: list[int] = []

        async def do_wait() -> None:
            if await limiter.wait(timeout=0.5):
                waited.append(1)

        # Mix acquire and wait operations
        for _ in range(50):
            limiter.acquire()
            acquired.append(1)

        await asyncio.gather(*[do_wait() for _ in range(30)])

        assert len(acquired) == 50
        assert len(waited) == 30  # All waits should succeed (enough capacity + refill)

    @pytest.mark.asyncio
    async def test_wait_timeout_zero(self) -> None:
        """timeout=0.0 returns immediately without sleeping."""
        limiter = RateLimiter(rate=1.0, capacity=1)
        limiter.acquire(tokens=1)  # Exhaust

        start = time.monotonic()
        result = await limiter.wait(timeout=0.0)
        elapsed = time.monotonic() - start

        assert result is False
        # Should return within a few ms (no actual sleep)
        assert elapsed < 0.02

    @pytest.mark.asyncio
    async def test_agent_aware_async_wait(self) -> None:
        """AgentAwareRateLimiter async wait with multiple agents."""
        limiter = AgentAwareRateLimiter(global_rate=100.0, per_agent_rate=20.0, max_agents=5)

        results: dict[str, list[int]] = {"a1": [], "a2": [], "a3": []}

        async def agent_worker(agent_id: str, count: int) -> None:
            for _ in range(count):
                if await limiter.wait(agent_id=agent_id, timeout=2.0):
                    results[agent_id].append(1)

        await asyncio.gather(
            agent_worker("a1", 10),
            agent_worker("a2", 10),
            agent_worker("a3", 10),
        )

        # All should succeed within timeout
        assert len(results["a1"]) == 10
        assert len(results["a2"]) == 10
        assert len(results["a3"]) == 10

    @pytest.mark.asyncio
    async def test_refill_accuracy_during_async_wait(self) -> None:
        """Token refill timing stays accurate across async sleeps."""
        limiter = RateLimiter(rate=100.0, capacity=100)
        limiter.acquire(tokens=100)

        # 10 concurrent waits, each for 1 token
        start = time.monotonic()
        results = await asyncio.gather(*[limiter.wait(timeout=1.0) for _ in range(10)])
        elapsed = time.monotonic() - start

        assert all(results)
        # 10 tokens at 100/sec = 0.1s, allow CI variance
        assert 0.03 < elapsed < 0.40

    @pytest.mark.asyncio
    async def test_event_loop_yields_during_wait(self) -> None:
        """Other coroutines run while rate limiter waits (proves non-blocking)."""
        limiter = RateLimiter(rate=5.0, capacity=1)
        limiter.acquire(tokens=1)

        other_task_ran = False

        async def other_task() -> None:
            nonlocal other_task_ran
            await asyncio.sleep(0.05)
            other_task_ran = True

        # Start other task while waiting for token
        task = asyncio.create_task(other_task())
        start = time.monotonic()
        result = await limiter.wait(timeout=1.0)
        elapsed = time.monotonic() - start

        await task

        # Wait should succeed after ~0.2s (5/sec = 0.2s per token)
        assert result is True
        assert 0.1 < elapsed < 0.5
        # Other task must have run (proves event loop wasn't blocked)
        assert other_task_ran is True
