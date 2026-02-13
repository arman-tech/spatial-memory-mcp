"""Protocol interfaces for rate limiting.

Defines what async callers need from rate limiters, following the
Dependency Inversion Principle. The server depends on these protocols,
not on concrete rate limiter classes.
"""

from __future__ import annotations

from typing import Protocol


class AsyncRateLimiterPort(Protocol):
    """Async rate limiter for the MCP call_tool boundary.

    Consumer: SpatialMemoryServer.call_tool()
    """

    async def wait(self, *, timeout: float | None = None) -> bool:
        """Wait until a token is available.

        Args:
            timeout: Maximum wait time in seconds (None = no limit).

        Returns:
            True if token acquired, False if timeout expired.
        """
        ...


class AsyncAgentRateLimiterPort(Protocol):
    """Async agent-aware rate limiter for the MCP call_tool boundary.

    Consumer: SpatialMemoryServer.call_tool()
    """

    async def wait(self, *, agent_id: str | None = None, timeout: float | None = None) -> bool:
        """Wait until a token is available for the given agent.

        Args:
            agent_id: Optional agent identifier for per-agent limiting.
            timeout: Maximum wait time in seconds (None = no limit).

        Returns:
            True if token acquired, False if timeout expired.
        """
        ...
