"""Unit tests for DecayManager.

Tests cover:
- Effective importance calculation with exponential decay
- Access count slowing decay
- Queue updates with deduplication
- Apply decay to results with re-ranking
- Background thread lifecycle
"""

from __future__ import annotations

import time
from datetime import timedelta
from unittest.mock import MagicMock

from spatial_memory.core.models import AutoDecayConfig
from spatial_memory.core.utils import utc_now
from spatial_memory.services.decay_manager import DecayManager, DecayUpdate


class TestDecayManagerInit:
    """Tests for DecayManager initialization."""

    def test_default_initialization(self) -> None:
        """DecayManager should initialize with default config."""
        repo = MagicMock()
        manager = DecayManager(repository=repo)

        assert manager.enabled is True
        assert manager.persist_enabled is True

    def test_disabled_initialization(self) -> None:
        """DecayManager should respect disabled config."""
        repo = MagicMock()
        config = AutoDecayConfig(enabled=False)
        manager = DecayManager(repository=repo, config=config)

        assert manager.enabled is False

    def test_custom_config(self) -> None:
        """DecayManager should accept custom config."""
        repo = MagicMock()
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=False,
            half_life_days=60.0,
            min_importance_floor=0.05,
            access_weight=0.5,
        )
        manager = DecayManager(repository=repo, config=config)

        assert manager.enabled is True
        assert manager.persist_enabled is False


class TestCalculateEffectiveImportance:
    """Tests for effective importance calculation."""

    def test_no_decay_when_disabled(self) -> None:
        """Should return stored importance when disabled."""
        repo = MagicMock()
        config = AutoDecayConfig(enabled=False)
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        result = manager.calculate_effective_importance(
            stored_importance=0.8,
            last_accessed=now - timedelta(days=30),
            access_count=0,
        )

        assert result == 0.8

    def test_no_decay_for_recently_accessed(self) -> None:
        """Should not decay memories accessed just now (with pure time decay).

        Note: With the unified algorithm, access_weight blends time_decay with
        access_stability. To test pure time-based decay (no access stability),
        we use access_weight=0.0.
        """
        repo = MagicMock()
        config = AutoDecayConfig(
            half_life_days=30.0,
            access_weight=0.0,  # Pure time decay, no access stability blending
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        result = manager.calculate_effective_importance(
            stored_importance=0.8,
            last_accessed=now,
            access_count=0,
        )

        # Use approximate comparison due to floating point precision
        assert abs(result - 0.8) < 0.001

    def test_decay_at_half_life(self) -> None:
        """Should decay with adaptive half-life based on importance.

        The new unified algorithm uses:
        - effective_half_life = half_life * access_factor * importance_factor
        - importance_factor = 1 + base_importance
        - For importance=1.0, effective_half_life = 30 * 1 * 2 = 60 days
        - At 30 days: decay_factor = 2^(-30/60) = ~0.707
        """
        repo = MagicMock()
        config = AutoDecayConfig(
            half_life_days=30.0,
            min_importance_floor=0.0,  # No floor for precise test
            access_weight=0.0,  # No access count effect
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        result = manager.calculate_effective_importance(
            stored_importance=1.0,
            last_accessed=now - timedelta(days=30),
            access_count=0,
        )

        # With importance=1.0, effective_half_life = 30 * 2 = 60 days
        # decay_factor = 2^(-30/60) = 0.707
        # result = 1.0 * 0.707 = 0.707
        expected = 2 ** (-30 / 60)  # ~0.707
        assert abs(result - expected) < 0.01

    def test_decay_at_effective_half_life(self) -> None:
        """Should decay to half at effective half-life (considering importance factor)."""
        repo = MagicMock()
        config = AutoDecayConfig(
            half_life_days=30.0,
            min_importance_floor=0.0,
            access_weight=0.0,
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        # For importance=1.0, effective_half_life = 60 days
        # So at 60 days, we should see ~0.5 decay
        result = manager.calculate_effective_importance(
            stored_importance=1.0,
            last_accessed=now - timedelta(days=60),
            access_count=0,
        )

        # At effective half-life (60 days), decay_factor = 0.5
        # result = 1.0 * 0.5 = 0.5
        assert abs(result - 0.5) < 0.01

    def test_importance_floor_respected(self) -> None:
        """Should not decay below min_importance_floor."""
        repo = MagicMock()
        config = AutoDecayConfig(
            half_life_days=1.0,  # Very fast decay
            min_importance_floor=0.1,
            access_weight=0.0,
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        result = manager.calculate_effective_importance(
            stored_importance=1.0,
            last_accessed=now - timedelta(days=365),  # Very old
            access_count=0,
        )

        assert result == 0.1  # Should hit the floor

    def test_access_count_slows_decay(self) -> None:
        """Higher access count should slow decay.

        The new unified algorithm uses:
        - access_factor = 1.5^min(access_count, 20)
        - For 10 accesses: access_factor = 1.5^10 ≈ 57.67
        - importance_factor = 1 + 1.0 = 2
        - effective_half_life = 30 * 57.67 * 2 = ~3460 days
        - Plus access_weight blends time decay with access stability
        """
        repo = MagicMock()
        config = AutoDecayConfig(
            half_life_days=30.0,
            min_importance_floor=0.0,
            access_weight=0.3,
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        last_accessed = now - timedelta(days=30)

        # No access count - normal decay
        result_no_access = manager.calculate_effective_importance(
            stored_importance=1.0,
            last_accessed=last_accessed,
            access_count=0,
        )

        # With 10 accesses - much slower decay
        result_with_access = manager.calculate_effective_importance(
            stored_importance=1.0,
            last_accessed=last_accessed,
            access_count=10,
        )

        # Memory with more accesses should have higher effective importance
        assert result_with_access > result_no_access

        # Without access: effective_half_life = 30 * 1 * 2 = 60 days
        # pure time_decay = 2^(-30/60) = 0.707
        # access_stability = 0 (no accesses)
        # decay_factor = 0.7 * 0.707 + 0.3 * 0 = 0.495
        # But the decay may be slightly different due to blending
        assert result_no_access < 0.6  # Should show significant decay

        # With 10 accesses, the decay should be much slower
        # The access stability component helps preserve importance
        assert result_with_access > 0.7  # Should retain most importance


class TestApplyDecayToResults:
    """Tests for applying decay to search results."""

    def test_empty_results(self) -> None:
        """Should handle empty results."""
        repo = MagicMock()
        manager = DecayManager(repository=repo)

        result = manager.apply_decay_to_results([])
        assert result == []

    def test_disabled_returns_unchanged(self) -> None:
        """Should return results unchanged when disabled."""
        repo = MagicMock()
        config = AutoDecayConfig(enabled=False)
        manager = DecayManager(repository=repo, config=config)

        results = [
            {"id": "1", "similarity": 0.9, "importance": 0.8},
        ]
        result = manager.apply_decay_to_results(results)

        assert result == results
        assert "effective_importance" not in result[0]

    def test_adds_effective_importance(self) -> None:
        """Should add effective_importance to results."""
        repo = MagicMock()
        config = AutoDecayConfig(
            persist_enabled=False,  # Disable persistence
            access_weight=0.0,  # Pure time decay for predictable test
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        results = [
            {
                "id": "1",
                "similarity": 0.9,
                "importance": 0.8,
                "last_accessed": now,
                "access_count": 0,
            },
        ]

        result = manager.apply_decay_to_results(results)

        assert "effective_importance" in result[0]
        # Use approximate comparison due to floating point precision
        assert abs(result[0]["effective_importance"] - 0.8) < 0.001

    def test_reranks_by_adjusted_score(self) -> None:
        """Should re-rank results by similarity * effective_importance."""
        repo = MagicMock()
        config = AutoDecayConfig(
            persist_enabled=False,
            half_life_days=30.0,
            min_importance_floor=0.0,
            access_weight=0.0,
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        results = [
            {
                "id": "old_high_sim",
                "similarity": 0.9,
                "importance": 1.0,
                "last_accessed": now - timedelta(days=60),  # Very old
                "access_count": 0,
            },
            {
                "id": "new_lower_sim",
                "similarity": 0.7,
                "importance": 1.0,
                "last_accessed": now,  # Recent
                "access_count": 0,
            },
        ]

        result = manager.apply_decay_to_results(results, rerank=True)

        # With new unified algorithm (importance_factor = 1 + base_importance):
        # - effective_half_life for importance=1.0 is 30 * 2 = 60 days
        # Old memory: at 60 days, effective_importance ≈ 0.5, adjusted = 0.9 * 0.5 = 0.45
        # New memory: effective_importance = 1.0, adjusted = 0.7 * 1.0 = 0.7
        # New memory should be ranked first despite lower similarity
        assert result[0]["id"] == "new_lower_sim"
        assert result[1]["id"] == "old_high_sim"

    def test_no_rerank_when_disabled(self) -> None:
        """Should preserve order when rerank=False."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=False)
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        results = [
            {"id": "1", "similarity": 0.5, "importance": 0.1,
             "last_accessed": now, "access_count": 0},
            {"id": "2", "similarity": 0.9, "importance": 1.0,
             "last_accessed": now, "access_count": 0},
        ]

        result = manager.apply_decay_to_results(results, rerank=False)

        # Order should be preserved
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_handles_missing_access_fields(self) -> None:
        """Should handle results without last_accessed/access_count."""
        repo = MagicMock()
        config = AutoDecayConfig(
            persist_enabled=False,
            access_weight=0.0,  # Pure time decay for predictable test
        )
        manager = DecayManager(repository=repo, config=config)

        results = [
            {"id": "1", "similarity": 0.9, "importance": 0.8},
        ]

        result = manager.apply_decay_to_results(results)

        # Should not crash, effective_importance should approximately equal stored importance
        # When last_accessed is None, it defaults to now(), so no time decay occurs
        assert "effective_importance" in result[0]
        # Use approximate comparison due to floating point precision
        assert abs(result[0]["effective_importance"] - 0.8) < 0.001


class TestQueueUpdates:
    """Tests for update queue with deduplication."""

    def test_deduplicates_updates(self) -> None:
        """Should keep only latest update per memory_id."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=True, min_change_threshold=0.0)
        manager = DecayManager(repository=repo, config=config)

        # Manually queue multiple updates for same memory
        updates = [
            DecayUpdate("mem1", 0.8, 0.7, time.monotonic()),
            DecayUpdate("mem1", 0.7, 0.6, time.monotonic() + 0.001),
            DecayUpdate("mem1", 0.6, 0.5, time.monotonic() + 0.002),
        ]

        manager._queue_updates(updates)

        # Should only have one pending update for mem1
        assert len(manager._pending_updates) == 1
        assert manager._pending_updates["mem1"].new_importance == 0.5

        # Stats should show deduplication
        stats = manager.get_stats()
        assert stats["updates_deduplicated"] == 2

    def test_queues_different_memories(self) -> None:
        """Should queue updates for different memories."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=True, min_change_threshold=0.0)
        manager = DecayManager(repository=repo, config=config)

        updates = [
            DecayUpdate("mem1", 0.8, 0.7, time.monotonic()),
            DecayUpdate("mem2", 0.9, 0.8, time.monotonic()),
            DecayUpdate("mem3", 0.7, 0.6, time.monotonic()),
        ]

        manager._queue_updates(updates)

        assert len(manager._pending_updates) == 3


class TestBackgroundWorker:
    """Tests for background thread lifecycle."""

    def test_start_creates_thread(self) -> None:
        """start() should create and start worker thread."""
        repo = MagicMock()
        repo.update_batch = MagicMock(return_value=(0, []))
        config = AutoDecayConfig(persist_enabled=True)
        manager = DecayManager(repository=repo, config=config)

        manager.start()

        assert manager._worker_thread is not None
        assert manager._worker_thread.is_alive()

        manager.stop()

    def test_stop_gracefully_shuts_down(self) -> None:
        """stop() should gracefully shut down worker."""
        repo = MagicMock()
        repo.update_batch = MagicMock(return_value=(0, []))
        config = AutoDecayConfig(
            persist_enabled=True,
            persist_flush_interval_seconds=0.1,
        )
        manager = DecayManager(repository=repo, config=config)

        manager.start()
        assert manager._worker_thread is not None
        assert manager._worker_thread.is_alive()

        manager.stop(timeout=2.0)

        assert not manager._worker_thread.is_alive()

    def test_worker_not_started_when_disabled(self) -> None:
        """start() should not create thread when persistence disabled."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=False)
        manager = DecayManager(repository=repo, config=config)

        manager.start()

        assert manager._worker_thread is None

    def test_multiple_start_calls_safe(self) -> None:
        """Multiple start() calls should be safe."""
        repo = MagicMock()
        repo.update_batch = MagicMock(return_value=(0, []))
        config = AutoDecayConfig(persist_enabled=True)
        manager = DecayManager(repository=repo, config=config)

        manager.start()
        thread1 = manager._worker_thread

        manager.start()
        thread2 = manager._worker_thread

        # Should be the same thread
        assert thread1 is thread2

        manager.stop()

    def test_worker_persists_batch(self) -> None:
        """Worker should persist batches to repository."""
        repo = MagicMock()
        repo.update_batch = MagicMock(return_value=(2, []))
        config = AutoDecayConfig(
            persist_enabled=True,
            persist_flush_interval_seconds=0.1,
            min_change_threshold=0.0,
        )
        manager = DecayManager(repository=repo, config=config)

        # Queue updates before starting worker
        updates = [
            DecayUpdate("mem1", 0.8, 0.7, time.monotonic()),
            DecayUpdate("mem2", 0.9, 0.8, time.monotonic()),
        ]
        manager._queue_updates(updates)

        manager.start()

        # Wait for flush
        time.sleep(0.3)

        manager.stop()

        # Should have called update_batch
        repo.update_batch.assert_called()


class TestMinChangeThreshold:
    """Tests for minimum change threshold filtering."""

    def test_ignores_small_changes(self) -> None:
        """Should not queue updates below min_change_threshold."""
        repo = MagicMock()
        config = AutoDecayConfig(
            persist_enabled=True,
            min_change_threshold=0.05,  # 5% threshold
            access_weight=0.0,  # Pure time decay for predictable test
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        # Result with only ~1% change (below threshold)
        # With access_weight=0.0 and importance=0.8, effective_half_life = 30 * 1.8 = 54 days
        # At 1 day: time_decay = 2^(-1/54) ≈ 0.987
        # Change: 0.8 - 0.8*0.987 ≈ 0.01 (1.3%)
        # Use even shorter time for smaller change
        results = [
            {
                "id": "1",
                "similarity": 0.9,
                "importance": 0.80,
                "last_accessed": now - timedelta(hours=6),  # Very small decay
                "access_count": 0,
            },
        ]

        manager.apply_decay_to_results(results)

        # Should not queue any updates (change too small)
        assert len(manager._pending_updates) == 0

    def test_queues_large_changes(self) -> None:
        """Should queue updates above min_change_threshold."""
        repo = MagicMock()
        config = AutoDecayConfig(
            persist_enabled=True,
            min_change_threshold=0.01,  # 1% threshold
            half_life_days=30.0,
            min_importance_floor=0.0,
        )
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        # Result with significant decay
        # With unified algorithm: effective_half_life = 30 * 2 (importance=1.0) = 60 days
        # At 30 days: decay_factor = 2^(-30/60) = ~0.707, so ~30% decay
        results = [
            {
                "id": "1",
                "similarity": 0.9,
                "importance": 1.0,
                "last_accessed": now - timedelta(days=30),  # ~30% decay
                "access_count": 0,
            },
        ]

        manager.apply_decay_to_results(results)

        # Should queue the update (30% > 1% threshold)
        assert len(manager._pending_updates) == 1


class TestGetStats:
    """Tests for statistics reporting."""

    def test_reports_initial_stats(self) -> None:
        """Should report initial zero stats."""
        repo = MagicMock()
        manager = DecayManager(repository=repo)

        stats = manager.get_stats()

        assert stats["enabled"] is True
        assert stats["persist_enabled"] is True
        assert stats["updates_queued"] == 0
        assert stats["updates_persisted"] == 0
        assert stats["updates_deduplicated"] == 0
        assert stats["pending_updates"] == 0

    def test_tracks_queued_updates(self) -> None:
        """Should track number of queued updates."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=True, min_change_threshold=0.0)
        manager = DecayManager(repository=repo, config=config)

        updates = [
            DecayUpdate("mem1", 0.8, 0.7, time.monotonic()),
            DecayUpdate("mem2", 0.9, 0.8, time.monotonic()),
        ]
        manager._queue_updates(updates)

        stats = manager.get_stats()
        assert stats["updates_queued"] == 2


class TestDatetimeParsing:
    """Tests for datetime parsing in apply_decay_to_results."""

    def test_handles_iso_string_datetime(self) -> None:
        """Should handle ISO format datetime strings."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=False)
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        results = [
            {
                "id": "1",
                "similarity": 0.9,
                "importance": 0.8,
                "last_accessed": now.isoformat(),  # String format
                "access_count": 5,
            },
        ]

        result = manager.apply_decay_to_results(results)

        # Should parse and calculate without error
        assert "effective_importance" in result[0]

    def test_handles_datetime_object(self) -> None:
        """Should handle datetime objects directly."""
        repo = MagicMock()
        config = AutoDecayConfig(persist_enabled=False)
        manager = DecayManager(repository=repo, config=config)

        now = utc_now()
        results = [
            {
                "id": "1",
                "similarity": 0.9,
                "importance": 0.8,
                "last_accessed": now,  # datetime object
                "access_count": 5,
            },
        ]

        result = manager.apply_decay_to_results(results)

        assert "effective_importance" in result[0]
