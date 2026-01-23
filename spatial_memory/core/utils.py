"""Shared utility functions for Spatial Memory MCP Server."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get current UTC datetime (timezone-aware).

    Returns:
        A timezone-aware datetime object representing the current time in UTC.

    Example:
        from spatial_memory.core.utils import utc_now
        now = utc_now()
        print(now.isoformat())  # 2024-01-15T10:30:00+00:00
    """
    return datetime.now(timezone.utc)
