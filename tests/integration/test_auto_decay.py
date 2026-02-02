"""Integration tests for auto-decay feature.

Tests cover:
- Decay persists to database
- Recall returns effective_importance
- Hybrid recall returns effective_importance
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.models import AutoDecayConfig, Memory, MemorySource
from spatial_memory.core.utils import utc_now
from spatial_memory.services.decay_manager import DecayManager
from spatial_memory.services.memory import MemoryService
from spatial_memory.services.utility import UtilityConfig, UtilityService


pytestmark = pytest.mark.integration


class TestAutoDecayPersistence:
    """Tests for auto-decay persistence to database."""

    def test_decay_persists_to_database(
        self,
        module_database: Database,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """Decay updates should persist to database via background worker."""
        # Create a decay manager with fast flush interval
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=True,
            persist_flush_interval_seconds=0.5,
            min_change_threshold=0.0,  # Persist all changes
            half_life_days=1.0,  # Fast decay for testing
            min_importance_floor=0.0,
        )
        decay_manager = DecayManager(
            repository=module_repository,
            config=config,
        )

        # Create a memory with old last_accessed
        past_time = utc_now() - timedelta(days=2)  # 2 days ago
        memory = Memory(
            id="",
            content="Test memory for decay persistence",
            namespace="default",
            importance=1.0,
            source=MemorySource.MANUAL,
        )
        vector = session_embedding_service.embed(memory.content)

        # Insert the memory
        memory_id = module_repository.add(memory, vector)

        # Manually update last_accessed to simulate old memory
        module_database.update(memory_id, {
            "last_accessed": past_time,
            "access_count": 0,
        })

        # Start the background worker
        decay_manager.start()

        try:
            # Apply decay to results (this should queue an update)
            results = [{
                "id": memory_id,
                "similarity": 0.9,
                "importance": 1.0,
                "last_accessed": past_time,
                "access_count": 0,
            }]

            processed = decay_manager.apply_decay_to_results(results)

            # Verify effective_importance was calculated
            assert "effective_importance" in processed[0]
            effective = processed[0]["effective_importance"]
            assert effective < 1.0  # Should have decayed

            # Wait for background flush
            time.sleep(1.0)

            # Check that the importance was updated in database
            updated_memory = module_repository.get(memory_id)
            assert updated_memory is not None

            # The database importance should now reflect the decayed value
            # (within some tolerance due to timing)
            assert updated_memory.importance < 1.0

        finally:
            decay_manager.stop()


class TestRecallWithEffectiveImportance:
    """Tests for recall returning effective_importance."""

    def test_recall_returns_effective_importance(
        self,
        module_database: Database,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """Recall should include effective_importance in results."""
        # Create memory service
        memory_service = MemoryService(
            repository=module_repository,
            embeddings=session_embedding_service,
        )

        # Create decay manager
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=False,  # Don't persist for this test
            half_life_days=30.0,
            min_importance_floor=0.1,
        )
        decay_manager = DecayManager(
            repository=module_repository,
            config=config,
        )

        # Create two memories - one recent, one old
        recent_result = memory_service.remember(
            content="Recent memory about cats",
            importance=0.8,
        )

        old_result = memory_service.remember(
            content="Old memory about cats",
            importance=0.8,
        )

        # Make the old memory actually old
        past_time = utc_now() - timedelta(days=60)  # 60 days ago
        module_database.update(old_result.id, {
            "last_accessed": past_time,
            "access_count": 0,
        })

        # Recall memories
        recall_result = memory_service.recall(
            query="cats",
            limit=10,
        )

        # Convert to dict and apply decay
        memories_list = [
            {
                "id": m.id,
                "similarity": m.similarity,
                "importance": m.importance,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
            }
            for m in recall_result.memories
        ]

        processed = decay_manager.apply_decay_to_results(memories_list)

        # Find the two memories
        recent_mem = next((m for m in processed if m["id"] == recent_result.id), None)
        old_mem = next((m for m in processed if m["id"] == old_result.id), None)

        assert recent_mem is not None
        assert old_mem is not None

        # Both should have effective_importance
        assert "effective_importance" in recent_mem
        assert "effective_importance" in old_mem

        # Recent memory should have higher effective_importance
        assert recent_mem["effective_importance"] > old_mem["effective_importance"]


class TestHybridRecallWithEffectiveImportance:
    """Tests for hybrid_recall returning effective_importance."""

    def test_hybrid_recall_returns_effective_importance(
        self,
        module_database: Database,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """Hybrid recall should include effective_importance in results."""
        # Create utility service
        utility_service = UtilityService(
            repository=module_repository,
            embeddings=session_embedding_service,
            config=UtilityConfig(),
        )

        # Create decay manager
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=False,
            half_life_days=30.0,
            min_importance_floor=0.1,
        )
        decay_manager = DecayManager(
            repository=module_repository,
            config=config,
        )

        # Create memories
        memory1 = Memory(
            id="",
            content="Python programming best practices",
            namespace="default",
            importance=0.9,
            source=MemorySource.MANUAL,
        )
        memory2 = Memory(
            id="",
            content="Python testing with pytest",
            namespace="default",
            importance=0.9,
            source=MemorySource.MANUAL,
        )

        vector1 = session_embedding_service.embed(memory1.content)
        vector2 = session_embedding_service.embed(memory2.content)

        id1 = module_repository.add(memory1, vector1)
        id2 = module_repository.add(memory2, vector2)

        # Make memory1 old
        past_time = utc_now() - timedelta(days=60)
        module_database.update(id1, {
            "last_accessed": past_time,
            "access_count": 0,
        })

        # Hybrid recall
        hybrid_result = utility_service.hybrid_recall(
            query="Python programming",
            alpha=0.5,
            limit=10,
        )

        # Convert to dict and apply decay
        memories_list = [
            {
                "id": m.id,
                "similarity": m.similarity,
                "importance": m.importance,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
            }
            for m in hybrid_result.memories
        ]

        processed = decay_manager.apply_decay_to_results(memories_list)

        # All results should have effective_importance
        for mem in processed:
            assert "effective_importance" in mem


class TestAutoDecayReranking:
    """Tests for auto-decay re-ranking behavior."""

    def test_decay_reranks_results(
        self,
        module_database: Database,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """Auto-decay should re-rank results favoring recent memories."""
        # Create decay manager
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=False,
            half_life_days=10.0,  # Fast decay
            min_importance_floor=0.0,
            access_weight=0.0,
        )
        decay_manager = DecayManager(
            repository=module_repository,
            config=config,
        )

        now = utc_now()

        # Create memories with different ages but same initial importance
        memories_data = [
            {"content": "Machine learning basics", "age_days": 0},
            {"content": "Machine learning advanced", "age_days": 30},
            {"content": "Machine learning expert", "age_days": 60},
        ]

        memory_ids = []
        for data in memories_data:
            memory = Memory(
                id="",
                content=data["content"],
                namespace="default",
                importance=1.0,
                source=MemorySource.MANUAL,
            )
            vector = session_embedding_service.embed(memory.content)
            mem_id = module_repository.add(memory, vector)
            memory_ids.append(mem_id)

            # Set last_accessed
            past_time = now - timedelta(days=data["age_days"])
            module_database.update(mem_id, {
                "last_accessed": past_time,
                "access_count": 0,
            })

        # Simulate search results with similar similarity scores
        # but different ages
        results = []
        for i, mem_id in enumerate(memory_ids):
            results.append({
                "id": mem_id,
                "similarity": 0.9,  # Same similarity
                "importance": 1.0,
                "last_accessed": now - timedelta(days=memories_data[i]["age_days"]),
                "access_count": 0,
            })

        # Apply decay with reranking
        processed = decay_manager.apply_decay_to_results(results, rerank=True)

        # After reranking, the newest memory should be first
        # (it has highest effective_importance, so highest adjusted_score)
        assert processed[0]["id"] == memory_ids[0]  # age=0, highest effective
        assert processed[1]["id"] == memory_ids[1]  # age=30
        assert processed[2]["id"] == memory_ids[2]  # age=60, lowest effective

        # Verify effective_importance decreases with age
        assert processed[0]["effective_importance"] > processed[1]["effective_importance"]
        assert processed[1]["effective_importance"] > processed[2]["effective_importance"]


class TestAccessCountSlowsDecay:
    """Tests for access count slowing decay."""

    def test_frequently_accessed_memory_decays_slower(
        self,
        module_database: Database,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """Frequently accessed memories should decay slower."""
        config = AutoDecayConfig(
            enabled=True,
            persist_enabled=False,
            half_life_days=30.0,
            min_importance_floor=0.0,
            access_weight=0.3,
        )
        decay_manager = DecayManager(
            repository=module_repository,
            config=config,
        )

        now = utc_now()
        past_time = now - timedelta(days=30)  # 30 days ago

        # Create two memories with same age but different access counts
        memory1 = Memory(
            id="",
            content="Rarely accessed memory",
            namespace="default",
            importance=1.0,
            source=MemorySource.MANUAL,
        )
        memory2 = Memory(
            id="",
            content="Frequently accessed memory",
            namespace="default",
            importance=1.0,
            source=MemorySource.MANUAL,
        )

        vector1 = session_embedding_service.embed(memory1.content)
        vector2 = session_embedding_service.embed(memory2.content)

        id1 = module_repository.add(memory1, vector1)
        id2 = module_repository.add(memory2, vector2)

        # Set same last_accessed but different access counts
        module_database.update(id1, {
            "last_accessed": past_time,
            "access_count": 0,
        })
        module_database.update(id2, {
            "last_accessed": past_time,
            "access_count": 50,  # Frequently accessed
        })

        # Apply decay
        results = [
            {
                "id": id1,
                "similarity": 0.9,
                "importance": 1.0,
                "last_accessed": past_time,
                "access_count": 0,
            },
            {
                "id": id2,
                "similarity": 0.9,
                "importance": 1.0,
                "last_accessed": past_time,
                "access_count": 50,
            },
        ]

        processed = decay_manager.apply_decay_to_results(results, rerank=False)

        # Memory with more accesses should have higher effective_importance
        rarely_accessed = next(m for m in processed if m["id"] == id1)
        frequently_accessed = next(m for m in processed if m["id"] == id2)

        assert frequently_accessed["effective_importance"] > rarely_accessed["effective_importance"]
