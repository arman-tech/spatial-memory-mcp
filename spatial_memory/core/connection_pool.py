"""Thread-safe LRU connection pool for LanceDB."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import lancedb

if TYPE_CHECKING:
    from lancedb import DBConnection

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe LRU connection pool for LanceDB connections.

    This class encapsulates connection management with:
    - LRU eviction when at max capacity
    - Thread-safe access with locking
    - Connection reuse for same URIs

    Example:
        pool = ConnectionPool(max_size=10)
        conn = pool.get_or_create("/path/to/db")
        # ... use connection
        pool.close_all()  # cleanup
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the connection pool.

        Args:
            max_size: Maximum number of connections to cache.
        """
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._connections: OrderedDict[str, DBConnection] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Get the maximum pool size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set the maximum pool size."""
        if value < 1:
            raise ValueError("max_size must be >= 1")
        with self._lock:
            self._max_size = value
            # Evict if now over capacity
            while len(self._connections) > self._max_size:
                self._evict_oldest()

    def get_or_create(
        self,
        uri: str,
        read_consistency_interval_ms: int = 0,
        **kwargs: Any,
    ) -> DBConnection:
        """Get existing connection or create new one.

        Args:
            uri: Database URI/path.
            read_consistency_interval_ms: Read consistency interval.
            **kwargs: Additional args for lancedb.connect().

        Returns:
            Database connection.
        """
        with self._lock:
            # Check if exists
            if uri in self._connections:
                # Move to end (most recently used)
                self._connections.move_to_end(uri)
                return self._connections[uri]

            # Evict oldest if at capacity
            while len(self._connections) >= self._max_size:
                self._evict_oldest()

            # Create new connection
            from datetime import timedelta
            conn_kwargs = dict(kwargs)
            if read_consistency_interval_ms > 0:
                conn_kwargs["read_consistency_interval"] = timedelta(
                    milliseconds=read_consistency_interval_ms
                )

            conn = lancedb.connect(uri, **conn_kwargs)
            self._connections[uri] = conn
            logger.debug(f"Created new connection for {uri} (pool size: {len(self._connections)})")
            return conn

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) connection."""
        if self._connections:
            uri, conn = self._connections.popitem(last=False)
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"Error closing evicted connection {uri}: {e}")
            logger.debug(f"Evicted connection for {uri}")

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for uri, conn in list(self._connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Error closing connection {uri}: {e}")
            self._connections.clear()
            logger.info("Closed all pooled connections")

    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "size": len(self._connections),
                "max_size": self._max_size,
                "uris": list(self._connections.keys()),
            }

    def __len__(self) -> int:
        """Return number of connections in pool."""
        with self._lock:
            return len(self._connections)
