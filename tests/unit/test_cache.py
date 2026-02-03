"""Unit tests for ResponseCache.

Tests cover:
- Basic get/set operations
- TTL-based expiration
- LRU eviction at capacity
- Namespace invalidation
- Thread safety with concurrent access
- Statistics tracking
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

from spatial_memory.core.cache import CacheEntry, CacheStats, ResponseCache


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_stores_all_fields(self) -> None:
        """CacheEntry should store value and timing metadata."""
        entry = CacheEntry(
            value={"data": "test"},
            expires_at=100.0,
            created_at=50.0,
        )

        assert entry.value == {"data": "test"}
        assert entry.expires_at == 100.0
        assert entry.created_at == 50.0


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_cache_stats_stores_all_fields(self) -> None:
        """CacheStats should store all statistics."""
        stats = CacheStats(
            hits=100,
            misses=25,
            evictions=10,
            size=500,
            max_size=1000,
        )

        assert stats.hits == 100
        assert stats.misses == 25
        assert stats.evictions == 10
        assert stats.size == 500
        assert stats.max_size == 1000

    def test_hit_rate_calculation(self) -> None:
        """hit_rate should calculate correct ratio."""
        stats = CacheStats(hits=75, misses=25, evictions=0, size=0, max_size=100)
        assert stats.hit_rate == 0.75

    def test_hit_rate_with_zero_requests(self) -> None:
        """hit_rate should return 0.0 when no requests."""
        stats = CacheStats(hits=0, misses=0, evictions=0, size=0, max_size=100)
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        """hit_rate should be 1.0 when all requests are hits."""
        stats = CacheStats(hits=100, misses=0, evictions=0, size=50, max_size=100)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self) -> None:
        """hit_rate should be 0.0 when all requests are misses."""
        stats = CacheStats(hits=0, misses=100, evictions=0, size=0, max_size=100)
        assert stats.hit_rate == 0.0


class TestResponseCacheInit:
    """Tests for ResponseCache initialization."""

    def test_default_initialization(self) -> None:
        """ResponseCache should initialize with defaults."""
        cache = ResponseCache()

        assert cache.max_size == 1000
        assert cache.default_ttl == 60.0
        stats = cache.stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_custom_initialization(self) -> None:
        """ResponseCache should accept custom max_size and default_ttl."""
        cache = ResponseCache(max_size=500, default_ttl=30.0)

        assert cache.max_size == 500
        assert cache.default_ttl == 30.0

    def test_invalid_max_size_zero(self) -> None:
        """ResponseCache should reject max_size of zero."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ResponseCache(max_size=0)

    def test_invalid_max_size_negative(self) -> None:
        """ResponseCache should reject negative max_size."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ResponseCache(max_size=-1)

    def test_invalid_default_ttl_zero(self) -> None:
        """ResponseCache should reject default_ttl of zero."""
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            ResponseCache(default_ttl=0.0)

    def test_invalid_default_ttl_negative(self) -> None:
        """ResponseCache should reject negative default_ttl."""
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            ResponseCache(default_ttl=-1.0)


class TestBasicGetSet:
    """Tests for basic get/set operations."""

    def test_get_returns_none_for_missing_key(self) -> None:
        """get() should return None for keys that don't exist."""
        cache = ResponseCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_set_and_get_string_value(self) -> None:
        """Should store and retrieve string values."""
        cache = ResponseCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_set_and_get_dict_value(self) -> None:
        """Should store and retrieve dict values."""
        cache = ResponseCache()
        data = {"memories": [1, 2, 3], "total": 3}
        cache.set("key1", data)
        assert cache.get("key1") == data

    def test_set_and_get_list_value(self) -> None:
        """Should store and retrieve list values."""
        cache = ResponseCache()
        data = [{"id": 1}, {"id": 2}]
        cache.set("key1", data)
        assert cache.get("key1") == data

    def test_set_and_get_none_value(self) -> None:
        """Should distinguish between None value and missing key."""
        cache = ResponseCache()
        # Set None explicitly
        cache.set("key1", None)
        # get() returns None for both cases, but internally the key exists
        # We can verify by checking stats
        cache.get("key1")  # Should be a hit
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 0

    def test_set_overwrites_existing_key(self) -> None:
        """set() should overwrite existing values."""
        cache = ResponseCache()
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_get_increments_hit_counter(self) -> None:
        """Successful get() should increment hits."""
        cache = ResponseCache()
        cache.set("key1", "value1")

        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.stats()
        assert stats.hits == 3

    def test_get_increments_miss_counter(self) -> None:
        """Failed get() should increment misses."""
        cache = ResponseCache()

        cache.get("missing1")
        cache.get("missing2")

        stats = cache.stats()
        assert stats.misses == 2

    def test_set_with_invalid_ttl_zero(self) -> None:
        """set() should reject zero TTL."""
        cache = ResponseCache()
        with pytest.raises(ValueError, match="ttl must be positive"):
            cache.set("key1", "value1", ttl=0.0)

    def test_set_with_invalid_ttl_negative(self) -> None:
        """set() should reject negative TTL."""
        cache = ResponseCache()
        with pytest.raises(ValueError, match="ttl must be positive"):
            cache.set("key1", "value1", ttl=-1.0)


class TestTTLExpiry:
    """Tests for TTL-based expiration."""

    def test_entry_expires_after_ttl(self) -> None:
        """Entries should expire after their TTL."""
        cache = ResponseCache(default_ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired now
        assert cache.get("key1") is None

    def test_custom_ttl_overrides_default(self) -> None:
        """Custom TTL in set() should override default_ttl."""
        cache = ResponseCache(default_ttl=10.0)  # Long default
        cache.set("key1", "value1", ttl=0.1)  # Short custom TTL

        assert cache.get("key1") == "value1"

        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_expired_entry_counts_as_miss(self) -> None:
        """Accessing expired entry should increment miss counter."""
        cache = ResponseCache(default_ttl=0.05)
        cache.set("key1", "value1")

        cache.get("key1")  # Hit
        time.sleep(0.1)
        cache.get("key1")  # Miss (expired)

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_expired_entry_is_removed(self) -> None:
        """Expired entries should be removed from cache on access."""
        cache = ResponseCache(default_ttl=0.05)
        cache.set("key1", "value1")

        time.sleep(0.1)
        cache.get("key1")  # Triggers removal

        stats = cache.stats()
        assert stats.size == 0

    def test_different_entries_different_ttls(self) -> None:
        """Different entries can have different TTLs."""
        cache = ResponseCache(default_ttl=1.0)
        cache.set("short", "value1", ttl=0.05)
        cache.set("long", "value2", ttl=1.0)

        time.sleep(0.1)

        assert cache.get("short") is None  # Expired
        assert cache.get("long") == "value2"  # Still valid


class TestLRUEviction:
    """Tests for LRU eviction at capacity."""

    def test_eviction_when_at_capacity(self) -> None:
        """Should evict LRU entry when cache is full."""
        cache = ResponseCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_eviction_counter_increments(self) -> None:
        """Evictions should be counted."""
        cache = ResponseCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Evicts key1
        cache.set("key4", "value4")  # Evicts key2

        stats = cache.stats()
        assert stats.evictions == 2

    def test_get_updates_lru_order(self) -> None:
        """Accessing an entry should make it most recently used."""
        cache = ResponseCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1, making it most recently used
        cache.get("key1")

        # Add new entry - should evict key2 (now LRU)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Preserved
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_set_existing_key_updates_lru_order(self) -> None:
        """Updating an existing key should make it most recently used."""
        cache = ResponseCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Update key1, making it most recently used
        cache.set("key1", "updated")

        # Add new entry - should evict key2 (now LRU)
        cache.set("key4", "value4")

        assert cache.get("key1") == "updated"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key_does_not_evict(self) -> None:
        """Updating existing key should not trigger eviction."""
        cache = ResponseCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Update should not evict
        cache.set("key1", "updated")

        stats = cache.stats()
        assert stats.evictions == 0
        assert stats.size == 3

    def test_max_size_one(self) -> None:
        """Cache with max_size=1 should work correctly."""
        cache = ResponseCache(max_size=1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", "value2")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        stats = cache.stats()
        assert stats.evictions == 1
        assert stats.size == 1


class TestNamespaceInvalidation:
    """Tests for namespace-based invalidation."""

    def test_invalidate_namespace_removes_matching_keys(self) -> None:
        """invalidate_namespace should remove all keys containing namespace."""
        cache = ResponseCache()

        cache.set("recall:default:query1:5", "result1")
        cache.set("recall:default:query2:10", "result2")
        cache.set("recall:work:query1:5", "result3")
        cache.set("nearby:default:id1:5", "result4")

        count = cache.invalidate_namespace("default")

        assert count == 3
        assert cache.get("recall:default:query1:5") is None
        assert cache.get("recall:default:query2:10") is None
        assert cache.get("nearby:default:id1:5") is None
        assert cache.get("recall:work:query1:5") == "result3"

    def test_invalidate_namespace_returns_zero_when_no_matches(self) -> None:
        """invalidate_namespace should return 0 when no keys match."""
        cache = ResponseCache()

        cache.set("recall:work:query:5", "result")

        count = cache.invalidate_namespace("nonexistent")

        assert count == 0
        assert cache.get("recall:work:query:5") == "result"

    def test_invalidate_namespace_on_empty_cache(self) -> None:
        """invalidate_namespace should handle empty cache gracefully."""
        cache = ResponseCache()

        count = cache.invalidate_namespace("any")

        assert count == 0

    def test_invalidate_namespace_partial_match(self) -> None:
        """invalidate_namespace should match partial strings."""
        cache = ResponseCache()

        cache.set("prefix_namespace_suffix", "value1")
        cache.set("namespace", "value2")
        cache.set("other", "value3")

        count = cache.invalidate_namespace("namespace")

        assert count == 2
        assert cache.get("prefix_namespace_suffix") is None
        assert cache.get("namespace") is None
        assert cache.get("other") == "value3"


class TestInvalidateAll:
    """Tests for invalidate_all."""

    def test_invalidate_all_clears_cache(self) -> None:
        """invalidate_all should clear all entries."""
        cache = ResponseCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        count = cache.invalidate_all()

        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_invalidate_all_returns_zero_on_empty_cache(self) -> None:
        """invalidate_all should return 0 on empty cache."""
        cache = ResponseCache()

        count = cache.invalidate_all()

        assert count == 0

    def test_invalidate_all_resets_size(self) -> None:
        """invalidate_all should reset size to 0."""
        cache = ResponseCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate_all()

        stats = cache.stats()
        assert stats.size == 0


class TestStats:
    """Tests for cache statistics."""

    def test_stats_returns_correct_values(self) -> None:
        """stats() should return accurate statistics."""
        cache = ResponseCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 2
        assert stats.max_size == 100

    def test_reset_stats_clears_counters(self) -> None:
        """reset_stats() should clear hit/miss/eviction counters."""
        cache = ResponseCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Eviction
        cache.get("key3")  # Hit
        cache.get("missing")  # Miss

        cache.reset_stats()
        stats = cache.stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        # Size should NOT be reset
        assert stats.size == 2

    def test_stats_after_reset_and_new_operations(self) -> None:
        """Stats should accumulate correctly after reset."""
        cache = ResponseCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit

        cache.reset_stats()

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1


class TestThreadSafety:
    """Tests for thread safety with concurrent access."""

    def test_concurrent_reads_and_writes(self) -> None:
        """Cache should handle concurrent reads and writes safely."""
        cache = ResponseCache(max_size=100)
        num_threads = 10
        ops_per_thread = 100
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(ops_per_thread):
                    key = f"key_{thread_id}_{i}"
                    cache.set(key, f"value_{thread_id}_{i}")
                    cache.get(key)
                    cache.get(f"missing_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cache.stats()
        # All operations should complete
        assert stats.hits + stats.misses == num_threads * ops_per_thread * 2

    def test_concurrent_invalidation(self) -> None:
        """Cache should handle concurrent invalidation safely."""
        cache = ResponseCache(max_size=1000)
        errors: list[Exception] = []

        # Pre-populate cache
        for i in range(100):
            cache.set(f"ns1:key_{i}", f"value_{i}")
            cache.set(f"ns2:key_{i}", f"value_{i}")

        def reader() -> None:
            try:
                for i in range(100):
                    cache.get(f"ns1:key_{i}")
                    cache.get(f"ns2:key_{i}")
            except Exception as e:
                errors.append(e)

        def invalidator() -> None:
            try:
                for _ in range(5):
                    cache.invalidate_namespace("ns1")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)] + [
            threading.Thread(target=invalidator) for _ in range(2)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_eviction(self) -> None:
        """Cache should handle concurrent eviction safely."""
        cache = ResponseCache(max_size=10)
        num_threads = 5
        ops_per_thread = 50
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(ops_per_thread):
                    cache.set(f"key_{thread_id}_{i}", f"value_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cache.stats()
        # Should have evicted most entries due to small max_size
        assert stats.evictions > 0
        assert stats.size <= 10

    def test_thread_pool_executor(self) -> None:
        """Cache should work correctly with ThreadPoolExecutor."""
        cache = ResponseCache(max_size=500)

        def task(i: int) -> tuple[int, Any]:
            key = f"task_key_{i}"
            cache.set(key, {"task": i, "data": list(range(10))})
            return i, cache.get(key)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(task, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        # All tasks should complete successfully
        assert len(results) == 100
        # All results should be valid (not None due to race condition)
        for i, result in results:
            # Result should match the set value
            assert result is not None
            assert result["task"] == i


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_key(self) -> None:
        """Cache should handle empty string keys."""
        cache = ResponseCache()
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"

    def test_unicode_keys(self) -> None:
        """Cache should handle unicode keys."""
        cache = ResponseCache()
        cache.set("key_with_emoji_\U0001f600", "value1")
        cache.set("key_\u4e2d\u6587", "value2")

        assert cache.get("key_with_emoji_\U0001f600") == "value1"
        assert cache.get("key_\u4e2d\u6587") == "value2"

    def test_very_long_key(self) -> None:
        """Cache should handle very long keys."""
        cache = ResponseCache()
        long_key = "k" * 10000
        cache.set(long_key, "long_key_value")
        assert cache.get(long_key) == "long_key_value"

    def test_complex_nested_value(self) -> None:
        """Cache should handle complex nested values."""
        cache = ResponseCache()
        complex_value: dict[str, Any] = {
            "memories": [
                {"id": "1", "content": "test", "metadata": {"nested": {"deep": True}}},
            ],
            "stats": {"count": 1, "elapsed": 0.05},
            "flags": [True, False, None],
        }
        cache.set("complex", complex_value)
        assert cache.get("complex") == complex_value

    def test_callable_value(self) -> None:
        """Cache should handle callable values (not call them)."""
        cache = ResponseCache()

        def my_func() -> str:
            return "called"

        cache.set("func", my_func)
        result = cache.get("func")
        assert result is my_func  # Should return the function, not call it

    def test_rapid_set_get_same_key(self) -> None:
        """Cache should handle rapid set/get on same key."""
        cache = ResponseCache()

        for i in range(1000):
            cache.set("rapid", i)
            result = cache.get("rapid")
            assert result == i

    def test_invalidate_namespace_with_special_characters(self) -> None:
        """invalidate_namespace should work with special characters."""
        cache = ResponseCache()

        cache.set("ns:special/chars:key", "value1")
        cache.set("ns:other:key", "value2")

        count = cache.invalidate_namespace("special/chars")

        assert count == 1
        assert cache.get("ns:special/chars:key") is None
        assert cache.get("ns:other:key") == "value2"
