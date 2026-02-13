"""Token bucket rate limiter for API calls."""

from __future__ import annotations

import asyncio
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

        # Or async wait:
        await limiter.wait()  # yields to event loop until token available
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

    def can_acquire(self, tokens: int = 1) -> bool:
        """Check if tokens could be acquired without consuming them.

        Args:
            tokens: Number of tokens to check.

        Returns:
            True if tokens are available, False otherwise.
        """
        with self._lock:
            self._refill()
            return self._tokens >= tokens

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

    def time_until_available(self, tokens: int = 1) -> float:
        """Estimate seconds until tokens are available.

        Returns 0.0 if tokens are already available. Uses lazy refill
        to compute the estimate based on current token state.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Estimated seconds to wait (0.0 if immediately available).
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self.rate

    def _acquire_or_estimate(self, tokens: int = 1) -> float:
        """Atomically acquire tokens or return deficit estimate.

        Combines acquire() and time_until_available() in a single lock
        acquisition, eliminating the TOCTOU window that causes
        time_until_available() to return 0.0 after a failed acquire()
        when tokens refill between the two calls.

        Returns:
            0.0 if tokens were acquired, or estimated seconds to wait.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            return (tokens - self._tokens) / self.rate

    def _peek_and_deficit(self, tokens: int = 1) -> tuple[bool, float]:
        """Atomically check availability and compute deficit without consuming.

        Used by AgentAwareRateLimiter to peek at sub-limiter state before
        deciding whether to consume from both.

        Returns:
            (available, deficit) -- available is True if tokens can be
            acquired, deficit is estimated seconds to wait (0.0 if available).
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return True, 0.0
            return False, (tokens - self._tokens) / self.rate

    async def wait(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Wait asynchronously until tokens are available.

        Uses _acquire_or_estimate() for atomic acquire-or-sleep decisions,
        eliminating the TOCTOU window between separate acquire() and
        time_until_available() calls. The threading.Lock is only held
        during the instant check, never across an await.

        Cancellation-safe: tokens are only consumed on success. If the
        coroutine is cancelled during sleep, no tokens are consumed and
        no rollback is needed.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait in seconds (None = no limit).

        Returns:
            True if tokens were acquired, False if timeout expired.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            wait_time = self._acquire_or_estimate(tokens)
            if wait_time == 0.0:
                return True

            if deadline is not None:
                wait_time = min(wait_time, deadline - time.monotonic())
                if wait_time <= 0:
                    return False

            await asyncio.sleep(wait_time)

    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "tokens_available": self._tokens,
                "capacity": self.capacity,
                "rate": self.rate,
            }


class AgentAwareRateLimiter:
    """Rate limiter with per-agent and global limits.

    Provides two-tier rate limiting:
    - Global limit: Shared across all agents/requests
    - Per-agent limit: Individual limit for each agent ID

    A request must pass BOTH limits to proceed. This prevents:
    - Any single agent from consuming all available capacity
    - Overall system overload from too many concurrent agents

    Example:
        limiter = AgentAwareRateLimiter(
            global_rate=100.0,  # 100 total ops/sec
            per_agent_rate=25.0,  # Each agent limited to 25 ops/sec
            max_agents=20,  # Track up to 20 agents
        )

        if limiter.acquire(agent_id="agent-123"):
            # perform operation
        else:
            # rate limited

    Thread Safety:
        This class is thread-safe. The global limiter and per-agent dict
        are protected by appropriate locks.
    """

    def __init__(
        self,
        global_rate: float = 100.0,
        per_agent_rate: float = 25.0,
        max_agents: int = 20,
    ) -> None:
        """Initialize the agent-aware rate limiter.

        Args:
            global_rate: Tokens per second for global limit.
            per_agent_rate: Tokens per second for each agent.
            max_agents: Maximum number of agent limiters to track.
                When exceeded, oldest (by last access) agents are evicted.
        """
        if global_rate <= 0:
            raise ValueError("global_rate must be positive")
        if per_agent_rate <= 0:
            raise ValueError("per_agent_rate must be positive")
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1")

        self._global = RateLimiter(rate=global_rate)
        self._per_agent: dict[str, RateLimiter] = {}
        self._per_agent_rate = per_agent_rate
        self._max_agents = max_agents
        self._lock = threading.Lock()
        # Track last access time for LRU eviction
        self._last_access: dict[str, float] = {}

    def _get_agent_limiter(self, agent_id: str) -> RateLimiter:
        """Get or create a rate limiter for an agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            RateLimiter for the agent.
        """
        with self._lock:
            now = time.monotonic()

            if agent_id not in self._per_agent:
                # Evict oldest agent if at capacity
                if len(self._per_agent) >= self._max_agents:
                    self._evict_oldest_agent()

                self._per_agent[agent_id] = RateLimiter(rate=self._per_agent_rate)

            self._last_access[agent_id] = now
            return self._per_agent[agent_id]

    def _evict_oldest_agent(self) -> None:
        """Evict the least recently accessed agent limiter.

        Must be called with self._lock held.
        """
        if not self._last_access:
            return

        oldest_agent = min(self._last_access, key=self._last_access.get)  # type: ignore[arg-type]
        del self._per_agent[oldest_agent]
        del self._last_access[oldest_agent]
        logger.debug(f"Evicted rate limiter for agent: {oldest_agent}")

    def acquire(self, agent_id: str | None = None, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Must pass BOTH global AND per-agent limits (if agent_id provided).
        Tokens are only consumed if both limits pass.

        Args:
            agent_id: Optional agent identifier. If None, only global limit applies.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        # If no agent_id, only global limit applies
        if agent_id is None:
            return self._global.acquire(tokens)

        # Check both limits first without consuming
        agent_limiter = self._get_agent_limiter(agent_id)

        if not self._global.can_acquire(tokens):
            return False

        if not agent_limiter.can_acquire(tokens):
            return False

        # Both limits pass, now actually consume tokens from both
        # Note: Small race window here, but acceptable for rate limiting
        self._global.acquire(tokens)
        agent_limiter.acquire(tokens)

        return True

    def time_until_available(self, agent_id: str | None = None, tokens: int = 1) -> float:
        """Estimate seconds until tokens are available from both limiters.

        Args:
            agent_id: Optional agent identifier.
            tokens: Number of tokens needed.

        Returns:
            Estimated seconds to wait (0.0 if immediately available).
        """
        global_wait = self._global.time_until_available(tokens)

        if agent_id is None:
            return global_wait

        agent_limiter = self._get_agent_limiter(agent_id)
        agent_wait = agent_limiter.time_until_available(tokens)

        return max(global_wait, agent_wait)

    def _acquire_or_estimate(self, agent_id: str | None = None, tokens: int = 1) -> float:
        """Atomically acquire tokens or return deficit estimate.

        Uses _peek_and_deficit() on each sub-limiter for atomic per-limiter
        estimates. If either sub-limiter has a deficit, returns that estimate
        (guaranteed > 0). If both are available, consumes from both.

        Returns:
            0.0 if tokens were acquired, or estimated seconds to wait.
        """
        if agent_id is None:
            return self._global._acquire_or_estimate(tokens)

        agent_limiter = self._get_agent_limiter(agent_id)

        # Atomic peek on each sub-limiter (no consuming)
        global_ok, global_deficit = self._global._peek_and_deficit(tokens)
        agent_ok, agent_deficit = agent_limiter._peek_and_deficit(tokens)

        if not global_ok or not agent_ok:
            # At least one has a deficit -- always > 0
            return max(global_deficit, agent_deficit)

        # Both available -- consume from both
        global_acquired = self._global.acquire(tokens)
        agent_acquired = agent_limiter.acquire(tokens)

        if not global_acquired or not agent_acquired:
            # Peek showed available but acquire failed (cross-lock race).
            # Self-healing: leaked token refills in 1/rate seconds.
            logger.warning(
                "Rate limiter peek/acquire race: global=%s agent=%s (tokens will refill naturally)",
                global_acquired,
                agent_acquired,
            )
            # Return a minimum deficit so the caller sleeps instead of hot-looping
            return max(
                0.0 if global_acquired else global_deficit,
                0.0 if agent_acquired else agent_deficit,
            ) or (1.0 / min(self._global.rate, self._per_agent_rate))

        return 0.0

    async def wait(
        self,
        agent_id: str | None = None,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """Wait asynchronously until tokens are available from both limiters.

        Uses _acquire_or_estimate() for atomic acquire-or-sleep decisions
        on each sub-limiter. See RateLimiter.wait() for cancellation safety.

        Args:
            agent_id: Optional agent identifier.
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait in seconds (None = no limit).

        Returns:
            True if tokens were acquired, False if timeout expired.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            wait_time = self._acquire_or_estimate(agent_id, tokens)
            if wait_time == 0.0:
                return True

            if deadline is not None:
                wait_time = min(wait_time, deadline - time.monotonic())
                if wait_time <= 0:
                    return False

            await asyncio.sleep(wait_time)

    def stats(self, agent_id: str | None = None) -> dict[str, Any]:
        """Get rate limiter statistics.

        Args:
            agent_id: Optional agent ID to include agent-specific stats.

        Returns:
            Dictionary with global and optionally per-agent statistics.
        """
        result: dict[str, Any] = {
            "global": self._global.stats(),
            "active_agents": len(self._per_agent),
            "max_agents": self._max_agents,
            "per_agent_rate": self._per_agent_rate,
        }

        if agent_id and agent_id in self._per_agent:
            result["agent"] = self._per_agent[agent_id].stats()

        return result

    def reset_agent(self, agent_id: str) -> bool:
        """Reset rate limiter for a specific agent.

        Useful for testing or when an agent reconnects.

        Args:
            agent_id: The agent identifier.

        Returns:
            True if agent was found and reset, False if not found.
        """
        with self._lock:
            if agent_id in self._per_agent:
                del self._per_agent[agent_id]
                del self._last_access[agent_id]
                return True
            return False

    def clear_all_agents(self) -> int:
        """Clear all per-agent rate limiters.

        Returns:
            Number of agents cleared.
        """
        with self._lock:
            count = len(self._per_agent)
            self._per_agent.clear()
            self._last_access.clear()
            return count
