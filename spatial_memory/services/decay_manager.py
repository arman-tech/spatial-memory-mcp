"""Automatic decay manager for real-time importance decay.

This service provides automatic decay calculation during recall operations,
re-ranking search results based on time-decayed importance. Updates are
optionally persisted to the database in the background.

Architecture:
    recall() / hybrid_recall()
            │
            ▼
    DecayManager.apply_decay_to_results()  ← Real-time (~20-50μs)
            │
       ┌────┴────┐
       ▼         ▼
    [Re-ranked   [Background Queue]
     Results]          │
                       ▼
                [Batch Persist Thread]
                       │
                       ▼
                [LanceDB Update]
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from spatial_memory.core.lifecycle_ops import apply_decay, calculate_decay_factor
from spatial_memory.core.models import AutoDecayConfig
from spatial_memory.core.utils import to_aware_utc, utc_now

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import MemoryRepositoryProtocol

logger = logging.getLogger(__name__)


@dataclass
class DecayUpdate:
    """A pending decay update for a memory."""

    memory_id: str
    old_importance: float
    new_importance: float
    timestamp: float  # time.monotonic() for deduplication


class DecayManager:
    """Manages automatic decay calculation and persistence.

    This service calculates effective importance during search operations
    using exponential decay based on time since last access. Results are
    re-ranked by multiplying similarity with effective importance.

    Background persistence is optional and uses a daemon thread with
    batched updates to minimize database overhead.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        config: AutoDecayConfig | None = None,
    ) -> None:
        """Initialize the decay manager.

        Args:
            repository: Repository for persisting decay updates.
            config: Configuration for decay behavior.
        """
        self._repo = repository
        self._config = config or AutoDecayConfig()

        # Threading primitives
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Update queue with backpressure (deque with maxlen)
        # Using maxlen for automatic backpressure - oldest items dropped
        self._update_queue: deque[DecayUpdate] = deque(
            maxlen=self._config.max_queue_size
        )

        # Track pending updates by memory_id for deduplication
        self._pending_updates: dict[str, DecayUpdate] = {}

        # Statistics
        self._stats_lock = threading.Lock()
        self._updates_queued = 0
        self._updates_persisted = 0
        self._updates_deduplicated = 0

    @property
    def enabled(self) -> bool:
        """Whether auto-decay is enabled."""
        return self._config.enabled

    @property
    def persist_enabled(self) -> bool:
        """Whether persistence is enabled."""
        return self._config.persist_enabled

    def start(self) -> None:
        """Start the background persistence worker.

        Safe to call multiple times - will only start if not already running.
        """
        if not self._config.enabled or not self._config.persist_enabled:
            logger.debug("Auto-decay persistence disabled, skipping worker start")
            return

        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.debug("Decay worker already running")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._background_worker,
            name="decay-persist-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Auto-decay background worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker gracefully.

        Flushes any pending updates before stopping.

        Args:
            timeout: Maximum time to wait for worker shutdown.
        """
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return

        logger.info("Stopping auto-decay background worker...")
        self._shutdown_event.set()

        # Wait for worker to finish
        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning("Decay worker did not stop within timeout")
        else:
            logger.info(
                f"Auto-decay worker stopped. "
                f"Queued: {self._updates_queued}, "
                f"Persisted: {self._updates_persisted}, "
                f"Deduplicated: {self._updates_deduplicated}"
            )

    def calculate_effective_importance(
        self,
        stored_importance: float,
        last_accessed: datetime,
        access_count: int,
    ) -> float:
        """Calculate time-decayed effective importance.

        Uses the unified decay algorithm from lifecycle_ops, supporting
        exponential, linear, and step decay functions with adaptive half-life
        based on access count and importance.

        Args:
            stored_importance: The stored importance value (0-1).
            last_accessed: When the memory was last accessed.
            access_count: Number of times the memory has been accessed.

        Returns:
            Effective importance after decay (clamped to min_importance_floor).
        """
        if not self._config.enabled:
            return stored_importance

        # Calculate days since last access
        # Normalize last_accessed to timezone-aware UTC (database may return naive)
        now = utc_now()
        last_accessed_aware = to_aware_utc(last_accessed)
        delta = now - last_accessed_aware
        days_since_access = delta.total_seconds() / 86400.0  # seconds in a day

        if days_since_access <= 0:
            return stored_importance

        # Use the unified decay algorithm from lifecycle_ops
        decay_factor = calculate_decay_factor(
            days_since_access=days_since_access,
            access_count=access_count,
            base_importance=stored_importance,
            decay_function=self._config.decay_function,
            half_life_days=self._config.half_life_days,
            access_weight=self._config.access_weight,
        )

        return apply_decay(
            current_importance=stored_importance,
            decay_factor=decay_factor,
            min_importance=self._config.min_importance_floor,
        )

    def apply_decay_to_results(
        self,
        results: list[dict[str, Any]],
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Apply decay to search results and optionally re-rank.

        Calculates effective_importance for each result and optionally
        re-ranks results by multiplying similarity with effective_importance.

        Args:
            results: List of memory result dictionaries.
            rerank: Whether to re-rank by adjusted score (similarity × effective_importance).

        Returns:
            Results with effective_importance added, optionally re-ranked.
        """
        if not self._config.enabled or not results:
            return results

        updates_to_queue: list[DecayUpdate] = []

        for result in results:
            # Extract required fields
            stored_importance = result.get("importance", 0.5)
            last_accessed = result.get("last_accessed")
            access_count = result.get("access_count", 0)
            memory_id = result.get("id", "")

            # Handle datetime parsing if needed
            if isinstance(last_accessed, str):
                try:
                    last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    last_accessed = utc_now()
            elif last_accessed is None:
                last_accessed = utc_now()

            # Calculate effective importance
            effective_importance = self.calculate_effective_importance(
                stored_importance=stored_importance,
                last_accessed=last_accessed,
                access_count=access_count,
            )

            # Add to result
            result["effective_importance"] = effective_importance

            # Check if we should queue an update
            if self._config.persist_enabled and memory_id:
                change = abs(stored_importance - effective_importance)
                if change >= self._config.min_change_threshold:
                    updates_to_queue.append(
                        DecayUpdate(
                            memory_id=memory_id,
                            old_importance=stored_importance,
                            new_importance=effective_importance,
                            timestamp=time.monotonic(),
                        )
                    )

        # Queue updates in bulk
        if updates_to_queue:
            self._queue_updates(updates_to_queue)

        # Re-rank by adjusted score if requested
        if rerank:
            # Calculate adjusted score: similarity × effective_importance
            for result in results:
                similarity = result.get("similarity", 0.0)
                effective = result.get("effective_importance", result.get("importance", 0.5))
                result["_adjusted_score"] = similarity * effective

            # Sort by adjusted score (descending)
            results.sort(key=lambda r: r.get("_adjusted_score", 0.0), reverse=True)

            # Remove temporary score field
            for result in results:
                result.pop("_adjusted_score", None)

        return results

    def _queue_updates(self, updates: list[DecayUpdate]) -> None:
        """Queue updates for background persistence with deduplication.

        Latest update per memory_id wins - prevents duplicate writes.

        Args:
            updates: List of decay updates to queue.
        """
        with self._lock:
            for update in updates:
                # Deduplicate: keep latest update per memory_id
                existing = self._pending_updates.get(update.memory_id)
                if existing is not None:
                    with self._stats_lock:
                        self._updates_deduplicated += 1

                self._pending_updates[update.memory_id] = update
                self._update_queue.append(update)

                with self._stats_lock:
                    self._updates_queued += 1

    def _background_worker(self) -> None:
        """Background worker that batches and persists decay updates."""
        logger.debug("Decay background worker started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for flush interval or shutdown
                self._shutdown_event.wait(timeout=self._config.persist_flush_interval_seconds)

                # Collect batch of updates
                batch = self._collect_batch()

                if batch:
                    self._persist_batch(batch)

            except Exception as e:
                logger.error(f"Error in decay background worker: {e}", exc_info=True)
                # Don't crash the worker on transient errors
                time.sleep(1.0)

        # Final flush on shutdown
        try:
            batch = self._collect_batch()
            if batch:
                logger.debug(f"Final flush: {len(batch)} updates")
                self._persist_batch(batch)
        except Exception as e:
            logger.error(f"Error in final decay flush: {e}", exc_info=True)

        logger.debug("Decay background worker stopped")

    def _collect_batch(self) -> list[DecayUpdate]:
        """Collect a batch of updates for persistence.

        Returns:
            List of unique updates (latest per memory_id).
        """
        with self._lock:
            if not self._pending_updates:
                return []

            # Get unique updates (already deduplicated in _pending_updates)
            batch = list(self._pending_updates.values())[:self._config.persist_batch_size]

            # Clear processed updates from pending dict
            for update in batch:
                self._pending_updates.pop(update.memory_id, None)

            return batch

    def _persist_batch(self, batch: list[DecayUpdate]) -> None:
        """Persist a batch of decay updates to the database.

        Args:
            batch: List of decay updates to persist.
        """
        if not batch:
            return

        # Build update tuples for batch update
        updates = [
            (update.memory_id, {"importance": update.new_importance})
            for update in batch
        ]

        try:
            success_count, failed_ids = self._repo.update_batch(updates)

            with self._stats_lock:
                self._updates_persisted += success_count

            if failed_ids:
                logger.warning(f"Failed to persist decay for {len(failed_ids)} memories")

            logger.debug(f"Persisted decay updates for {success_count} memories")

        except Exception as e:
            logger.error(f"Failed to persist decay batch: {e}")
            # Re-queue failed updates? For now, just log and continue
            # In a production system, you might want retry logic here

    def get_stats(self) -> dict[str, Any]:
        """Get decay manager statistics.

        Returns:
            Dictionary with queue and persistence stats.
        """
        with self._stats_lock:
            return {
                "enabled": self._config.enabled,
                "persist_enabled": self._config.persist_enabled,
                "updates_queued": self._updates_queued,
                "updates_persisted": self._updates_persisted,
                "updates_deduplicated": self._updates_deduplicated,
                "pending_updates": len(self._pending_updates),
                "queue_size": len(self._update_queue),
                "queue_max_size": self._config.max_queue_size,
                "worker_alive": (
                    self._worker_thread is not None and self._worker_thread.is_alive()
                ),
            }
