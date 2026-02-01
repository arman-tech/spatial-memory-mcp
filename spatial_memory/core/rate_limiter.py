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

        Args:
            agent_id: Optional agent identifier. If None, only global limit applies.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        # Check global limit first (cheaper)
        if not self._global.acquire(tokens):
            return False

        # If no agent_id, only global limit applies
        if agent_id is None:
            return True

        # Check per-agent limit
        agent_limiter = self._get_agent_limiter(agent_id)
        if not agent_limiter.acquire(tokens):
            # Failed per-agent limit, but we already consumed global tokens
            # This is acceptable - prevents gaming by switching agents
            return False

        return True

    def wait(
        self,
        agent_id: str | None = None,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """Wait until tokens are available from both limiters.

        Args:
            agent_id: Optional agent identifier.
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait (None = no limit).

        Returns:
            True if tokens were acquired, False if timeout.
        """
        start = time.monotonic()

        while True:
            if self.acquire(agent_id, tokens):
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return False

            # Sleep briefly before retry
            time.sleep(0.01)  # 10ms

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
