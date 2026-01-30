"""Tests for connection pool."""

import tempfile
from pathlib import Path

import pytest

from spatial_memory.core.connection_pool import ConnectionPool


class TestConnectionPoolBasics:
    """Test basic connection pool functionality."""

    def test_pool_initialization(self) -> None:
        """Test pool initializes with correct defaults."""
        pool = ConnectionPool(max_size=5)
        assert pool.max_size == 5
        assert len(pool) == 0

    def test_pool_stats(self) -> None:
        """Test pool statistics."""
        pool = ConnectionPool(max_size=3)
        stats = pool.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 3
        assert stats["uris"] == []

    def test_invalid_max_size(self) -> None:
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ConnectionPool(max_size=0)


class TestConnectionPoolOperations:
    """Test connection pool operations."""

    def test_get_or_create_single_connection(self) -> None:
        """Test creating a single connection."""
        pool = ConnectionPool(max_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            uri = str(Path(tmpdir) / "test-db")
            conn1 = pool.get_or_create(uri)
            assert conn1 is not None
            assert len(pool) == 1

            # Second call should return same connection
            conn2 = pool.get_or_create(uri)
            assert conn2 is conn1
            assert len(pool) == 1

    def test_multiple_connections(self) -> None:
        """Test creating multiple connections."""
        pool = ConnectionPool(max_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            uris = [str(Path(tmpdir) / f"db{i}") for i in range(3)]

            conns = [pool.get_or_create(uri) for uri in uris]
            assert len(pool) == 3
            assert len(set(id(c) for c in conns)) == 3  # All unique

            # Verify stats
            stats = pool.stats()
            assert stats["size"] == 3
            assert set(stats["uris"]) == set(uris)

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when pool is full."""
        pool = ConnectionPool(max_size=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            uri1 = str(Path(tmpdir) / "db1")
            uri2 = str(Path(tmpdir) / "db2")
            uri3 = str(Path(tmpdir) / "db3")

            # Fill pool
            conn1 = pool.get_or_create(uri1)
            conn2 = pool.get_or_create(uri2)
            assert len(pool) == 2

            # Access uri1 again (moves to end)
            conn1_again = pool.get_or_create(uri1)
            assert conn1 is conn1_again

            # Add third connection - should evict uri2 (oldest)
            conn3 = pool.get_or_create(uri3)
            assert len(pool) == 2

            stats = pool.stats()
            assert uri1 in stats["uris"]
            assert uri3 in stats["uris"]
            assert uri2 not in stats["uris"]

    def test_read_consistency_interval(self) -> None:
        """Test read consistency interval parameter."""
        pool = ConnectionPool(max_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            uri = str(Path(tmpdir) / "test-db")
            conn = pool.get_or_create(uri, read_consistency_interval_ms=1000)
            assert conn is not None


class TestConnectionPoolCleanup:
    """Test connection pool cleanup."""

    def test_close_all(self) -> None:
        """Test closing all connections."""
        pool = ConnectionPool(max_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            uris = [str(Path(tmpdir) / f"db{i}") for i in range(3)]
            for uri in uris:
                pool.get_or_create(uri)

            assert len(pool) == 3

            pool.close_all()
            assert len(pool) == 0

            stats = pool.stats()
            assert stats["size"] == 0
            assert stats["uris"] == []


class TestConnectionPoolMaxSizeChange:
    """Test changing max_size dynamically."""

    def test_increase_max_size(self) -> None:
        """Test increasing max_size."""
        pool = ConnectionPool(max_size=2)
        assert pool.max_size == 2

        pool.max_size = 5
        assert pool.max_size == 5

    def test_decrease_max_size_triggers_eviction(self) -> None:
        """Test decreasing max_size triggers eviction."""
        pool = ConnectionPool(max_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4 connections
            uris = [str(Path(tmpdir) / f"db{i}") for i in range(4)]
            for uri in uris:
                pool.get_or_create(uri)

            assert len(pool) == 4

            # Reduce max_size to 2 - should evict 2 oldest
            pool.max_size = 2
            assert len(pool) == 2

            # Only the 2 most recent should remain
            stats = pool.stats()
            assert uris[2] in stats["uris"]
            assert uris[3] in stats["uris"]
            assert uris[0] not in stats["uris"]
            assert uris[1] not in stats["uris"]

    def test_set_invalid_max_size(self) -> None:
        """Test setting invalid max_size raises ValueError."""
        pool = ConnectionPool(max_size=5)

        with pytest.raises(ValueError, match="max_size must be >= 1"):
            pool.max_size = 0

        with pytest.raises(ValueError, match="max_size must be >= 1"):
            pool.max_size = -1


class TestConnectionPoolThreadSafety:
    """Test connection pool thread safety."""

    def test_concurrent_access(self) -> None:
        """Test concurrent access to pool."""
        import threading

        pool = ConnectionPool(max_size=10)
        errors = []

        def worker(uri: str) -> None:
            try:
                conn = pool.get_or_create(uri)
                assert conn is not None
            except Exception as e:
                errors.append(e)

        with tempfile.TemporaryDirectory() as tmpdir:
            uri = str(Path(tmpdir) / "test-db")
            threads = [threading.Thread(target=worker, args=(uri,)) for _ in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(pool) == 1  # All threads should share same connection
