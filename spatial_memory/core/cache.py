"""Response cache with TTL and LRU eviction for Spatial Memory MCP Server.

This module provides a thread-safe response cache using:
- LRU (Least Recently Used) eviction when at capacity
- TTL (Time To Live) based expiration checked on get()
- Namespace-based invalidation for targeted cache clearing

Usage:
    from spatial_memory.core.cache import ResponseCache

    cache = ResponseCache(max_size=1000, default_ttl=60.0)

    # Basic get/set
    cache.set("recall:default:query:5", results, ttl=30.0)
    cached = cache.get("recall:default:query:5")

    # Namespace invalidation
    cache.invalidate_namespace("default")  # Clears all keys containing "default"

    # Stats
    stats = cache.stats()
    print(f"Hit rate: {stats.hits / (stats.hits + stats.misses):.2%}")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with value and expiration metadata.

    Attributes:
        value: The cached value.
        expires_at: Monotonic timestamp when this entry expires.
        created_at: Monotonic timestamp when this entry was created.
    """

    value: Any
    expires_at: float
    created_at: float


@dataclass
class CacheStats:
    """Statistics about cache performance and usage.

    Attributes:
        hits: Number of successful cache hits.
        misses: Number of cache misses (key not found or expired).
        evictions: Number of entries evicted due to capacity limits.
        size: Current number of entries in the cache.
        max_size: Maximum capacity of the cache.
    """

    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0, or 0.0 if no requests.
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ResponseCache:
    """Thread-safe LRU cache with TTL expiration.

    This cache is designed for caching MCP tool responses. Keys are strings
    typically formatted as "tool:namespace:query:limit" to enable targeted
    namespace invalidation.

    The cache uses:
    - OrderedDict for O(1) LRU operations
    - time.monotonic() for TTL (immune to system clock changes)
    - threading.Lock() for thread safety

    Example:
        cache = ResponseCache(max_size=1000, default_ttl=60.0)

        # Set with default TTL
        cache.set("recall:ns:query:10", result)

        # Set with custom TTL
        cache.set("recall:ns:query:10", result, ttl=30.0)

        # Get (returns None on miss)
        result = cache.get("recall:ns:query:10")

        # Invalidate all entries for a namespace
        count = cache.invalidate_namespace("ns")

        # Get statistics
        stats = cache.stats()
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 60.0) -> None:
        """Initialize the response cache.

        Args:
            max_size: Maximum number of entries to store. Must be positive.
            default_ttl: Default time-to-live in seconds. Must be positive.

        Raises:
            ValueError: If max_size or default_ttl is not positive.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        """Get a value from the cache.

        Returns the cached value if the key exists and has not expired.
        On cache hit, the entry is moved to the end (most recently used).
        On cache miss (not found or expired), returns None and increments
        the miss counter.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if not found or expired.

        Example:
            result = cache.get("recall:default:test:5")
            if result is not None:
                print("Cache hit!")
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check if expired
            if time.monotonic() > entry.expires_at:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache miss (expired): %s", key)
                return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit: %s", key)
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in the cache.

        If the key already exists, it is updated and moved to the end.
        If the cache is at capacity, the least recently used entry is evicted.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. Uses default_ttl if not specified.
                 Must be positive if specified.

        Raises:
            ValueError: If ttl is not positive.

        Example:
            # With default TTL
            cache.set("recall:default:test:5", result)

            # With custom TTL
            cache.set("recall:default:test:5", result, ttl=120.0)
        """
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")

        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.monotonic()
        entry = CacheEntry(
            value=value,
            expires_at=now + effective_ttl,
            created_at=now,
        )

        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                logger.debug("Cache update: %s (ttl=%.1fs)", key, effective_ttl)
                return

            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                # popitem(last=False) removes the first item (LRU)
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("Cache eviction (LRU): %s", evicted_key)

            # Add new entry
            self._cache[key] = entry
            logger.debug("Cache set: %s (ttl=%.1fs)", key, effective_ttl)

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries containing the given namespace.

        This is useful when data in a namespace changes and cached
        query results should be refreshed.

        Args:
            namespace: The namespace string to match. All keys containing
                       this string will be invalidated.

        Returns:
            The number of entries invalidated.

        Example:
            # After modifying memories in "work" namespace
            count = cache.invalidate_namespace("work")
            print(f"Invalidated {count} cached entries")
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache if namespace in key]
            for key in keys_to_remove:
                del self._cache[key]

            if keys_to_remove:
                logger.debug(
                    "Cache invalidate namespace '%s': %d entries",
                    namespace,
                    len(keys_to_remove),
                )

            return len(keys_to_remove)

    def invalidate_all(self) -> int:
        """Clear the entire cache.

        Returns:
            The number of entries cleared.

        Example:
            count = cache.invalidate_all()
            print(f"Cleared {count} cached entries")
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug("Cache cleared: %d entries", count)
            return count

    def stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats with hits, misses, evictions, size, and max_size.

        Example:
            stats = cache.stats()
            print(f"Hit rate: {stats.hit_rate:.2%}")
            print(f"Size: {stats.size}/{stats.max_size}")
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._cache),
                max_size=self._max_size,
            )

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters to zero.

        This does not clear the cache itself, only the statistics.

        Example:
            cache.reset_stats()
        """
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.debug("Cache stats reset")

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @property
    def default_ttl(self) -> float:
        """Get the default TTL in seconds."""
        return self._default_ttl
