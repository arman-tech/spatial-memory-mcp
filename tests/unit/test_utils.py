"""Unit tests for core utility functions.

Tests the timezone-aware datetime utilities that handle the impedance
mismatch between Python's timezone-aware datetimes and LanceDB's naive
datetime storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from spatial_memory.core.utils import (
    to_aware_utc,
    to_naive_utc,
    utc_now,
    utc_now_naive,
)


class TestUtcNow:
    """Tests for utc_now() function."""

    def test_returns_datetime(self) -> None:
        """Should return a datetime object."""
        result = utc_now()
        assert isinstance(result, datetime)

    def test_is_timezone_aware(self) -> None:
        """Should return a timezone-aware datetime."""
        result = utc_now()
        assert result.tzinfo is not None

    def test_is_utc_timezone(self) -> None:
        """Should be in UTC timezone."""
        result = utc_now()
        assert result.tzinfo == timezone.utc

    def test_is_approximately_now(self) -> None:
        """Should return current time (within 1 second tolerance)."""
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)

        assert before <= result <= after

    def test_consecutive_calls_are_ordered(self) -> None:
        """Consecutive calls should return increasing times."""
        first = utc_now()
        second = utc_now()

        assert first <= second


class TestUtcNowNaive:
    """Tests for utc_now_naive() function."""

    def test_returns_datetime(self) -> None:
        """Should return a datetime object."""
        result = utc_now_naive()
        assert isinstance(result, datetime)

    def test_is_timezone_naive(self) -> None:
        """Should return a naive datetime (no timezone info)."""
        result = utc_now_naive()
        assert result.tzinfo is None

    def test_is_approximately_now(self) -> None:
        """Should return current UTC time (within 1 second tolerance)."""
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        result = utc_now_naive()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        assert before <= result <= after

    def test_matches_utc_now_without_tzinfo(self) -> None:
        """Should match utc_now() when tzinfo is stripped."""
        aware = utc_now()
        naive = utc_now_naive()

        # Allow 1 second tolerance for execution time
        diff = abs((aware.replace(tzinfo=None) - naive).total_seconds())
        assert diff < 1.0

    def test_can_compare_with_lancedb_style_datetimes(self) -> None:
        """Should be able to compare with naive datetimes (LanceDB compatibility)."""
        now = utc_now_naive()

        # Simulate a LanceDB-stored naive datetime
        lancedb_stored = datetime(2024, 1, 15, 10, 30, 0)

        # This should NOT raise TypeError
        assert now > lancedb_stored
        assert (now - lancedb_stored).total_seconds() > 0


class TestToNaiveUtc:
    """Tests for to_naive_utc() function."""

    def test_converts_aware_utc_to_naive(self) -> None:
        """Should strip tzinfo from UTC datetime."""
        aware = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = to_naive_utc(aware)

        assert result.tzinfo is None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_converts_non_utc_timezone_to_naive_utc(self) -> None:
        """Should convert non-UTC timezone to UTC before stripping tzinfo."""
        # Create a datetime in UTC+5 timezone
        utc_plus_5 = timezone(timedelta(hours=5))
        aware = datetime(2024, 1, 15, 15, 30, 0, tzinfo=utc_plus_5)

        result = to_naive_utc(aware)

        assert result.tzinfo is None
        # 15:30 UTC+5 = 10:30 UTC
        assert result.hour == 10
        assert result.minute == 30

    def test_returns_naive_unchanged(self) -> None:
        """Should return naive datetime unchanged (assumed already UTC)."""
        naive = datetime(2024, 1, 15, 10, 30, 0)
        result = to_naive_utc(naive)

        assert result is naive  # Same object
        assert result.tzinfo is None

    def test_preserves_microseconds(self) -> None:
        """Should preserve microsecond precision."""
        aware = datetime(2024, 1, 15, 10, 30, 0, 123456, tzinfo=timezone.utc)
        result = to_naive_utc(aware)

        assert result.microsecond == 123456

    def test_handles_negative_offset(self) -> None:
        """Should correctly convert negative timezone offset."""
        # UTC-8 (e.g., Pacific Standard Time)
        utc_minus_8 = timezone(timedelta(hours=-8))
        aware = datetime(2024, 1, 15, 2, 30, 0, tzinfo=utc_minus_8)

        result = to_naive_utc(aware)

        # 02:30 UTC-8 = 10:30 UTC
        assert result.hour == 10
        assert result.minute == 30

    def test_handles_date_boundary_crossing(self) -> None:
        """Should correctly handle date boundary when converting."""
        # UTC+12 at 01:00 = previous day 13:00 UTC
        utc_plus_12 = timezone(timedelta(hours=12))
        aware = datetime(2024, 1, 16, 1, 0, 0, tzinfo=utc_plus_12)

        result = to_naive_utc(aware)

        assert result.day == 15
        assert result.hour == 13


class TestToAwareUtc:
    """Tests for to_aware_utc() function."""

    def test_adds_tzinfo_to_naive(self) -> None:
        """Should add UTC tzinfo to naive datetime."""
        naive = datetime(2024, 1, 15, 10, 30, 0)
        result = to_aware_utc(naive)

        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_converts_non_utc_to_utc(self) -> None:
        """Should convert non-UTC timezone to UTC."""
        utc_plus_5 = timezone(timedelta(hours=5))
        aware = datetime(2024, 1, 15, 15, 30, 0, tzinfo=utc_plus_5)

        result = to_aware_utc(aware)

        assert result.tzinfo == timezone.utc
        # 15:30 UTC+5 = 10:30 UTC
        assert result.hour == 10
        assert result.minute == 30

    def test_returns_utc_aware_unchanged(self) -> None:
        """Should return UTC-aware datetime with same values."""
        aware = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = to_aware_utc(aware)

        assert result.tzinfo == timezone.utc
        assert result.hour == 10

    def test_preserves_microseconds(self) -> None:
        """Should preserve microsecond precision."""
        naive = datetime(2024, 1, 15, 10, 30, 0, 654321)
        result = to_aware_utc(naive)

        assert result.microsecond == 654321

    def test_handles_negative_offset(self) -> None:
        """Should correctly convert negative timezone offset."""
        utc_minus_8 = timezone(timedelta(hours=-8))
        aware = datetime(2024, 1, 15, 2, 30, 0, tzinfo=utc_minus_8)

        result = to_aware_utc(aware)

        assert result.tzinfo == timezone.utc
        # 02:30 UTC-8 = 10:30 UTC
        assert result.hour == 10


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_naive_to_aware_to_naive(self) -> None:
        """Converting naive -> aware -> naive should preserve value."""
        original = datetime(2024, 1, 15, 10, 30, 0, 123456)

        aware = to_aware_utc(original)
        back_to_naive = to_naive_utc(aware)

        assert back_to_naive == original

    def test_aware_utc_to_naive_to_aware(self) -> None:
        """Converting aware UTC -> naive -> aware should preserve value."""
        original = datetime(2024, 1, 15, 10, 30, 0, 123456, tzinfo=timezone.utc)

        naive = to_naive_utc(original)
        back_to_aware = to_aware_utc(naive)

        assert back_to_aware == original

    def test_utc_now_and_utc_now_naive_are_equivalent(self) -> None:
        """utc_now() stripped should equal utc_now_naive()."""
        aware = utc_now()
        naive = utc_now_naive()

        # Strip tzinfo from aware for comparison
        aware_stripped = aware.replace(tzinfo=None)

        # Allow 1 second tolerance
        diff = abs((aware_stripped - naive).total_seconds())
        assert diff < 1.0


class TestLanceDBCompatibility:
    """Tests verifying LanceDB compatibility scenarios."""

    def test_can_compare_naive_datetimes_for_decay(self) -> None:
        """Should be able to compare timestamps for decay calculation."""
        now = utc_now_naive()

        # Simulate a memory's last_accessed from LanceDB
        last_accessed = datetime(2024, 1, 1, 0, 0, 0)  # Naive UTC

        # This is the decay calculation pattern
        days_since_access = (now - last_accessed).total_seconds() / 86400

        assert days_since_access > 0
        assert isinstance(days_since_access, float)

    def test_can_normalize_mixed_timestamps(self) -> None:
        """Should handle mix of aware and naive timestamps."""
        # Some memories might have aware timestamps (from API)
        aware_timestamp = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Some might have naive timestamps (from LanceDB)
        naive_timestamp = datetime(2024, 6, 15, 10, 0, 0)

        # Both should normalize to comparable naive UTC
        normalized_aware = to_naive_utc(aware_timestamp)
        normalized_naive = to_naive_utc(naive_timestamp)

        # They should be equal after normalization
        assert normalized_aware == normalized_naive

    def test_decay_calculation_with_mixed_inputs(self) -> None:
        """Should correctly calculate decay with mixed timestamp types."""
        now = utc_now_naive()

        # Memory with aware timestamp (e.g., from API input)
        aware_last_accessed = datetime(
            now.year, now.month, now.day, now.hour, now.minute,
            tzinfo=timezone.utc
        ) - timedelta(days=30)

        # Normalize for comparison
        normalized = to_naive_utc(aware_last_accessed)
        days_since = (now - normalized).total_seconds() / 86400

        # Should be approximately 30 days
        assert 29.9 < days_since < 30.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_to_naive_utc_at_midnight(self) -> None:
        """Should handle midnight correctly."""
        aware = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = to_naive_utc(aware)

        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_to_naive_utc_at_end_of_day(self) -> None:
        """Should handle 23:59:59 correctly."""
        aware = datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
        result = to_naive_utc(aware)

        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 59

    def test_to_aware_utc_at_year_boundary(self) -> None:
        """Should handle year boundary correctly."""
        naive = datetime(2024, 12, 31, 23, 59, 59)
        result = to_aware_utc(naive)

        assert result.year == 2024
        assert result.month == 12
        assert result.day == 31

    def test_handles_leap_year(self) -> None:
        """Should handle leap year dates correctly."""
        # Feb 29 in a leap year
        aware = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)
        naive = to_naive_utc(aware)
        back = to_aware_utc(naive)

        assert naive.month == 2
        assert naive.day == 29
        assert back == aware
