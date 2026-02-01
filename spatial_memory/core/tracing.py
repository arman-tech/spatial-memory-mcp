"""Request tracing and timing utilities for Spatial Memory MCP Server.

This module provides request context tracking and timing utilities to support
observability and debugging. It uses contextvars for safe async propagation.

Usage:
    from spatial_memory.core.tracing import (
        RequestContext,
        TimingContext,
        get_current_context,
        set_context,
        clear_context,
    )

    # Set request context
    ctx = RequestContext(
        request_id="abc123def456",
        agent_id="agent-1",
        tool_name="recall",
        started_at=utc_now(),
        namespace="default",
    )
    token = set_context(ctx)
    try:
        # ... do work
        pass
    finally:
        clear_context(token)

    # Measure operation timings
    timing = TimingContext()
    with timing.measure("embedding"):
        # ... generate embedding
        pass
    with timing.measure("search"):
        # ... perform search
        pass
    print(f"Total: {timing.total_ms():.2f}ms")
    print(f"Timings: {timing.timings}")
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextvars import Token

from spatial_memory.core.utils import utc_now


@dataclass
class RequestContext:
    """Context information for a request.

    Attributes:
        request_id: Unique identifier for the request (first 12 chars of UUID).
        agent_id: Optional identifier for the calling agent.
        tool_name: Name of the MCP tool being called.
        started_at: When the request started.
        namespace: Optional namespace being operated on.
    """

    request_id: str
    agent_id: str | None
    tool_name: str
    started_at: datetime
    namespace: str | None = None

    @classmethod
    def create(
        cls,
        tool_name: str,
        agent_id: str | None = None,
        namespace: str | None = None,
    ) -> RequestContext:
        """Create a new request context with auto-generated ID.

        Args:
            tool_name: Name of the MCP tool being called.
            agent_id: Optional identifier for the calling agent.
            namespace: Optional namespace being operated on.

        Returns:
            A new RequestContext with generated request_id and started_at.

        Example:
            ctx = RequestContext.create("recall", agent_id="agent-1")
        """
        return cls(
            request_id=uuid.uuid4().hex[:12],
            agent_id=agent_id,
            tool_name=tool_name,
            started_at=utc_now(),
            namespace=namespace,
        )

    def elapsed_ms(self) -> float:
        """Calculate elapsed time since request started.

        Returns:
            Elapsed time in milliseconds.
        """
        return (utc_now() - self.started_at).total_seconds() * 1000


# Context variable for request tracking
_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "request_context", default=None
)


def get_current_context() -> RequestContext | None:
    """Get the current request context.

    Returns:
        The current RequestContext, or None if not in a request context.

    Example:
        ctx = get_current_context()
        if ctx:
            print(f"Request {ctx.request_id} for tool {ctx.tool_name}")
    """
    return _context.get()


def set_context(ctx: RequestContext) -> Token[RequestContext | None]:
    """Set the current request context.

    Args:
        ctx: The request context to set.

    Returns:
        A token that can be used to reset the context.

    Example:
        ctx = RequestContext.create("recall")
        token = set_context(ctx)
        try:
            # ... do work
            pass
        finally:
            clear_context(token)
    """
    return _context.set(ctx)


def clear_context(token: Token[RequestContext | None]) -> None:
    """Reset the context to its previous value.

    Args:
        token: The token returned from set_context().

    Example:
        token = set_context(ctx)
        try:
            # ... do work
            pass
        finally:
            clear_context(token)
    """
    _context.reset(token)


@contextmanager
def request_context(
    tool_name: str,
    agent_id: str | None = None,
    namespace: str | None = None,
) -> Generator[RequestContext, None, None]:
    """Context manager for request tracing.

    Creates a RequestContext, sets it as current, and clears it on exit.

    Args:
        tool_name: Name of the MCP tool being called.
        agent_id: Optional identifier for the calling agent.
        namespace: Optional namespace being operated on.

    Yields:
        The created RequestContext.

    Example:
        with request_context("recall", agent_id="agent-1") as ctx:
            print(f"Request {ctx.request_id}")
            # ... do work
    """
    ctx = RequestContext.create(tool_name, agent_id=agent_id, namespace=namespace)
    token = set_context(ctx)
    try:
        yield ctx
    finally:
        clear_context(token)


@dataclass
class TimingContext:
    """Context for measuring operation timings.

    Tracks timing of multiple named operations within a request.
    Uses perf_counter for high-precision timing.

    Attributes:
        timings: Dictionary mapping operation names to durations in milliseconds.
        start: Start time of the context (perf_counter value).

    Example:
        timing = TimingContext()
        with timing.measure("embedding"):
            embed = generate_embedding(text)
        with timing.measure("search"):
            results = search(embed)
        print(f"Total: {timing.total_ms():.2f}ms")
        print(f"Embedding: {timing.timings['embedding']:.2f}ms")
    """

    timings: dict[str, float] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Measure the duration of a named operation.

        Args:
            name: Name of the operation being measured.

        Yields:
            None

        Example:
            with timing.measure("database_query"):
                results = db.query(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = (time.perf_counter() - t0) * 1000

    def total_ms(self) -> float:
        """Calculate total elapsed time since context creation.

        Returns:
            Total elapsed time in milliseconds.

        Example:
            timing = TimingContext()
            # ... do some work
            print(f"Total time: {timing.total_ms():.2f}ms")
        """
        return (time.perf_counter() - self.start) * 1000

    def summary(self) -> dict[str, float]:
        """Get a summary of all timings including total.

        Returns:
            Dictionary with all named timings plus 'total_ms' and 'other_ms'.
            'other_ms' is time not accounted for by named operations.

        Example:
            timing = TimingContext()
            with timing.measure("op1"):
                time.sleep(0.01)
            summary = timing.summary()
            # {'op1': 10.0, 'total_ms': 10.5, 'other_ms': 0.5}
        """
        total = self.total_ms()
        measured = sum(self.timings.values())
        return {
            **self.timings,
            "total_ms": total,
            "other_ms": max(0.0, total - measured),
        }


def format_context_prefix() -> str:
    """Format the current context as a log prefix.

    Returns:
        A string like "[req=abc123][agent=agent-1]" or "" if no context.

    Example:
        prefix = format_context_prefix()
        logger.info(f"{prefix}Processing request...")
    """
    ctx = get_current_context()
    if ctx is None:
        return ""

    parts = [f"[req={ctx.request_id}]"]
    if ctx.agent_id:
        parts.append(f"[agent={ctx.agent_id}]")
    return "".join(parts)
