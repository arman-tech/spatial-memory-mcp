"""Shared utility functions for Spatial Memory MCP Server.

This module provides timezone-aware datetime utilities that handle the
impedance mismatch between Python's timezone-aware datetimes and LanceDB's
naive datetime storage.

Design Principles:
    - Use timezone-aware datetimes for business logic and API responses
    - Use naive UTC datetimes for database operations (LanceDB compatibility)
    - Centralize all timezone conversion logic here for consistency
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get current UTC datetime (timezone-aware).

    Use this for business logic, API responses, and when timezone
    information should be preserved.

    Returns:
        A timezone-aware datetime object representing the current time in UTC.

    Example:
        >>> from spatial_memory.core.utils import utc_now
        >>> now = utc_now()
        >>> now.tzinfo is not None
        True
        >>> print(now.isoformat())  # 2024-01-15T10:30:00+00:00
    """
    return datetime.now(timezone.utc)


def utc_now_naive() -> datetime:
    """Get current UTC datetime as naive (no timezone info).

    Use this for database comparisons where LanceDB stores naive timestamps.
    This replaces the deprecated datetime.utcnow() function.

    Returns:
        A naive datetime object representing the current time in UTC.

    Example:
        >>> from spatial_memory.core.utils import utc_now_naive
        >>> now = utc_now_naive()
        >>> now.tzinfo is None
        True
        >>> print(now.isoformat())  # 2024-01-15T10:30:00
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_naive_utc(dt: datetime) -> datetime:
    """Convert any datetime to naive UTC for database operations.

    This handles the conversion safely regardless of input type:
    - If timezone-aware: converts to UTC, then strips tzinfo
    - If naive: assumes already UTC, returns as-is

    Args:
        dt: A datetime object (naive or timezone-aware).

    Returns:
        A naive datetime object representing the time in UTC.

    Example:
        >>> from datetime import datetime, timezone
        >>> from spatial_memory.core.utils import to_naive_utc
        >>> aware = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        >>> naive = to_naive_utc(aware)
        >>> naive.tzinfo is None
        True
        >>> naive.hour
        10
    """
    if dt.tzinfo is not None:
        # Convert to UTC first (handles non-UTC timezones), then strip tzinfo
        dt = dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=None)
    return dt


def to_aware_utc(dt: datetime) -> datetime:
    """Convert any datetime to timezone-aware UTC.

    This handles the conversion safely regardless of input type:
    - If naive: assumes UTC, adds tzinfo
    - If aware: converts to UTC

    Args:
        dt: A datetime object (naive or timezone-aware).

    Returns:
        A timezone-aware datetime object in UTC.

    Example:
        >>> from datetime import datetime
        >>> from spatial_memory.core.utils import to_aware_utc
        >>> naive = datetime(2024, 1, 15, 10, 30)
        >>> aware = to_aware_utc(naive)
        >>> aware.tzinfo is not None
        True
        >>> aware.hour
        10
    """
    if dt.tzinfo is None:
        # Assume naive datetime is already UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
