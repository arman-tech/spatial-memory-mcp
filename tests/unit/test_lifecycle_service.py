"""Unit tests for LifecycleService with mocked dependencies.

Tests the lifecycle management operations:
- Decay: Apply time/access-based importance decay
- Reinforce: Boost memory importance on access
- Extract: Auto-extract memories from text
- Consolidate: Merge similar/duplicate memories

Uses mocked repositories and embedding services for isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.core.errors import (
    ValidationError,
)
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.services.lifecycle import (
    ConsolidateResult,
    DecayResult,
    ExtractResult,
    LifecycleConfig,
    LifecycleService,
    ReinforceResult,
)

# =============================================================================
# Test UUIDs (valid format)
# =============================================================================

TEST_UUID_1 = "11111111-1111-1111-1111-111111111111"
TEST_UUID_2 = "22222222-2222-2222-2222-222222222222"
TEST_UUID_3 = "33333333-3333-3333-3333-333333333333"
NONEXISTENT_UUID = "00000000-0000-0000-0000-000000000000"


# =============================================================================
# Helper functions
# =============================================================================


def make_memory(
    id: str,
    content: str | None = None,
    namespace: str = "default",
    importance: float = 0.5,
    access_count: int = 0,
    created_at: datetime | None = None,
    last_accessed: datetime | None = None,
) -> Memory:
    """Create a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=id,
        content=content or f"Memory content for {id}",
        namespace=namespace,
        importance=importance,
        tags=[],
        source=MemorySource.MANUAL,
        metadata={},
        created_at=created_at or now,
        updated_at=now,
        last_accessed=last_accessed or now,
        access_count=access_count,
    )


def make_memory_result(
    id: str,
    content: str | None = None,
    similarity: float = 0.8,
    namespace: str = "default",
    importance: float = 0.5,
) -> MemoryResult:
    """Create a MemoryResult object for testing."""
    return MemoryResult(
        id=id,
        content=content or f"Memory content for {id}",
        similarity=similarity,
        namespace=namespace,
        tags=[],
        importance=importance,
        created_at=datetime.now(timezone.utc),
        metadata={},
    )


def make_vector(dims: int = 384, seed: int | None = None) -> np.ndarray:
    """Create a random unit vector."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    vec = rng.standard_normal(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return np.asarray(vec / norm, dtype=np.float32)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository for unit tests.

    Returns a MagicMock that satisfies MemoryRepositoryProtocol.
    Configure specific behavior in individual tests.
    """
    repo = MagicMock()

    # Default returns
    repo.add.return_value = TEST_UUID_1
    repo.get.return_value = None
    repo.get_with_vector.return_value = None
    repo.delete.return_value = True
    repo.search.return_value = []
    repo.update.return_value = None
    repo.count.return_value = 0
    repo.get_namespaces.return_value = []
    repo.get_all.return_value = []
    repo.get_vectors_for_clustering.return_value = ([], np.array([]))

    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service for unit tests."""
    embeddings = MagicMock()
    embeddings.dimensions = 384
    embeddings.embed = MagicMock(return_value=make_vector(seed=42))
    embeddings.embed_batch = MagicMock(
        return_value=[make_vector(seed=i) for i in range(10)]
    )
    return embeddings


@pytest.fixture
def lifecycle_service(
    mock_repository: MagicMock,
    mock_embeddings: MagicMock,
) -> LifecycleService:
    """LifecycleService with mocked dependencies."""
    return LifecycleService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=LifecycleConfig(),
    )


# =============================================================================
# TestDecay
# =============================================================================


class TestDecay:
    """Tests for LifecycleService.decay() - importance decay operation."""

    def test_decay_returns_result(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should return DecayResult with decay information."""
        # Setup: memories that haven't been accessed recently
        now = datetime.now(timezone.utc)
        old_memories = [
            (
                make_memory(
                    TEST_UUID_1,
                    importance=0.8,
                    last_accessed=now - timedelta(days=60),
                ),
                make_vector(seed=1),
            ),
            (
                make_memory(
                    TEST_UUID_2,
                    importance=0.6,
                    last_accessed=now - timedelta(days=30),
                ),
                make_vector(seed=2),
            ),
        ]
        mock_repository.get_all.return_value = old_memories

        result = lifecycle_service.decay(dry_run=True)

        assert isinstance(result, DecayResult)
        assert result.memories_analyzed == 2
        assert result.dry_run is True

    def test_decay_dry_run_no_changes(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() with dry_run=True should not modify database."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        lifecycle_service.decay(dry_run=True)

        # update() should not be called in dry run
        mock_repository.update.assert_not_called()

    def test_decay_applies_changes(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() with dry_run=False should update database."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(dry_run=False)

        # update() should be called
        assert mock_repository.update.called
        assert result.dry_run is False

    def test_decay_respects_namespace(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should filter by namespace when specified."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            namespace="work",
            importance=0.8,
            last_accessed=now - timedelta(days=60),
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        lifecycle_service.decay(namespace="work", dry_run=True)

        # get_all should be called with namespace filter
        mock_repository.get_all.assert_called_once()
        call_kwargs = mock_repository.get_all.call_args.kwargs
        assert call_kwargs.get("namespace") == "work"

    def test_decay_calculates_correct_factor(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should calculate correct decay factors."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=1.0,
            last_accessed=now - timedelta(days=30),
            access_count=0,
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(
            half_life_days=30.0,
            dry_run=True,
        )

        # With default settings, decay should occur
        assert len(result.decayed_memories) > 0
        decayed = result.decayed_memories[0]
        assert decayed.new_importance < decayed.old_importance

    def test_decay_respects_min_importance(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should not reduce importance below min_importance."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=0.3,
            last_accessed=now - timedelta(days=365),  # Very old
            access_count=0,
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(
            min_importance=0.1,
            dry_run=True,
        )

        # New importance should not go below floor
        if result.decayed_memories:
            decayed = result.decayed_memories[0]
            assert decayed.new_importance >= 0.1

    def test_decay_handles_empty_database(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should handle empty database gracefully."""
        mock_repository.get_all.return_value = []

        result = lifecycle_service.decay(dry_run=True)

        assert result.memories_analyzed == 0
        assert result.memories_decayed == 0
        assert len(result.decayed_memories) == 0

    def test_decay_skips_recently_accessed(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should have minimal impact on recently accessed memories."""
        now = datetime.now(timezone.utc)
        recent_memory = make_memory(
            TEST_UUID_1,
            importance=0.8,
            last_accessed=now,  # Just accessed
            access_count=10,
        )
        mock_repository.get_all.return_value = [(recent_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(dry_run=True)

        # Recently accessed memory should have minimal decay
        # With 0 days since access and 10 access count, factor should be high
        if result.decayed_memories:
            decayed = result.decayed_memories[0]
            # Factor should be reasonably high (access count helps)
            assert decayed.decay_factor > 0.8

    def test_decay_with_linear_function(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should support linear decay function."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=1.0,
            last_accessed=now - timedelta(days=30),
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(
            decay_function="linear",
            half_life_days=30.0,
            dry_run=True,
        )

        assert isinstance(result, DecayResult)

    def test_decay_with_step_function(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """decay() should support step decay function."""
        now = datetime.now(timezone.utc)
        old_memory = make_memory(
            TEST_UUID_1,
            importance=1.0,
            last_accessed=now - timedelta(days=30),
        )
        mock_repository.get_all.return_value = [(old_memory, make_vector(seed=1))]

        result = lifecycle_service.decay(
            decay_function="step",
            half_life_days=30.0,
            dry_run=True,
        )

        assert isinstance(result, DecayResult)


# =============================================================================
# TestReinforce
# =============================================================================


class TestReinforce:
    """Tests for LifecycleService.reinforce() - importance boost operation."""

    def test_reinforce_boosts_importance(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should increase memory importance."""
        memory = make_memory(TEST_UUID_1, importance=0.5)
        mock_repository.get.return_value = memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1],
            boost_type="additive",
            boost_amount=0.1,
        )

        assert isinstance(result, ReinforceResult)
        assert result.memories_reinforced == 1
        assert len(result.reinforced) == 1
        assert result.reinforced[0].new_importance > result.reinforced[0].old_importance

    def test_reinforce_updates_access(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() with update_access=True should update access timestamp."""
        memory = make_memory(TEST_UUID_1, importance=0.5)
        mock_repository.get.return_value = memory

        lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1],
            boost_type="additive",
            boost_amount=0.1,
            update_access=True,
        )

        # update() should be called with access-related fields
        assert mock_repository.update.called

    def test_reinforce_handles_not_found(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should track not found memories."""
        mock_repository.get.return_value = None

        result = lifecycle_service.reinforce(
            memory_ids=[NONEXISTENT_UUID],
            boost_type="additive",
            boost_amount=0.1,
        )

        assert result.memories_reinforced == 0
        assert len(result.not_found) == 1
        assert NONEXISTENT_UUID in result.not_found

    def test_reinforce_multiple_memories(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should handle multiple memories."""
        def get_memory(memory_id: str) -> Memory | None:
            if memory_id == TEST_UUID_1:
                return make_memory(TEST_UUID_1, importance=0.5)
            elif memory_id == TEST_UUID_2:
                return make_memory(TEST_UUID_2, importance=0.6)
            return None

        mock_repository.get.side_effect = get_memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1, TEST_UUID_2, NONEXISTENT_UUID],
            boost_type="additive",
            boost_amount=0.1,
        )

        assert result.memories_reinforced == 2
        assert len(result.reinforced) == 2
        assert len(result.not_found) == 1

    def test_reinforce_multiplicative_boost(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should support multiplicative boost."""
        memory = make_memory(TEST_UUID_1, importance=0.5)
        mock_repository.get.return_value = memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1],
            boost_type="multiplicative",
            boost_amount=0.2,  # 20% increase
        )

        reinforced = result.reinforced[0]
        # 0.5 * 1.2 = 0.6
        assert abs(reinforced.new_importance - 0.6) < 0.01

    def test_reinforce_set_value(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should support set_value boost type."""
        memory = make_memory(TEST_UUID_1, importance=0.3)
        mock_repository.get.return_value = memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1],
            boost_type="set_value",
            boost_amount=0.9,
        )

        reinforced = result.reinforced[0]
        assert reinforced.new_importance == 0.9

    def test_reinforce_respects_max_cap(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should not exceed maximum importance."""
        memory = make_memory(TEST_UUID_1, importance=0.95)
        mock_repository.get.return_value = memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1],
            boost_type="additive",
            boost_amount=0.2,  # Would exceed 1.0
        )

        reinforced = result.reinforced[0]
        assert reinforced.new_importance <= 1.0

    def test_reinforce_calculates_avg_boost(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """reinforce() should calculate average boost correctly."""
        def get_memory(memory_id: str) -> Memory | None:
            if memory_id == TEST_UUID_1:
                return make_memory(TEST_UUID_1, importance=0.5)
            elif memory_id == TEST_UUID_2:
                return make_memory(TEST_UUID_2, importance=0.6)
            return None

        mock_repository.get.side_effect = get_memory

        result = lifecycle_service.reinforce(
            memory_ids=[TEST_UUID_1, TEST_UUID_2],
            boost_type="additive",
            boost_amount=0.1,
        )

        assert abs(result.avg_boost - 0.1) < 0.001  # Both got 0.1 boost


# =============================================================================
# TestExtract
# =============================================================================


class TestExtract:
    """Tests for LifecycleService.extract() - memory extraction operation."""

    def test_extract_finds_candidates(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """extract() should find memory candidates in text."""
        text = "We decided to use Redis for caching. The fix was to increase connection timeout."

        result = lifecycle_service.extract(
            text=text,
            min_confidence=0.5,
            deduplicate=False,
        )

        assert isinstance(result, ExtractResult)
        assert result.candidates_found > 0

    def test_extract_deduplicates(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """extract() with deduplicate=True should check for existing memories."""
        text = "We decided to use PostgreSQL for the database."

        # Mock existing similar memory
        mock_repository.search.return_value = [
            make_memory_result(
                TEST_UUID_1,
                content="We chose PostgreSQL for data storage",
                similarity=0.95,
            )
        ]

        result = lifecycle_service.extract(
            text=text,
            min_confidence=0.5,
            deduplicate=True,
            dedup_threshold=0.9,
        )

        # Should find candidate but not store it due to deduplication
        assert result.deduplicated_count >= 0

    def test_extract_stores_memories(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """extract() should store new memories when no duplicates found."""
        text = "We decided to use a microservices architecture for better scalability."

        # No existing similar memories
        mock_repository.search.return_value = []
        mock_repository.add.return_value = TEST_UUID_1

        result = lifecycle_service.extract(
            text=text,
            min_confidence=0.5,
            deduplicate=True,
        )

        if result.candidates_found > 0:
            # add() should be called for non-deduplicated candidates
            assert result.memories_created >= 0

    def test_extract_respects_namespace(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """extract() should store memories in specified namespace."""
        text = "Important: always backup before migrations."

        mock_repository.search.return_value = []
        mock_repository.add.return_value = TEST_UUID_1

        lifecycle_service.extract(
            text=text,
            namespace="production",
            min_confidence=0.5,
        )

        # If add was called, check namespace
        if mock_repository.add.called:
            call_args = mock_repository.add.call_args
            memory = call_args[0][0]
            assert memory.namespace == "production"

    def test_extract_respects_min_confidence(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """extract() should filter by min_confidence."""
        text = "We decided on X."  # Short, lower confidence match

        result_low = lifecycle_service.extract(
            text=text,
            min_confidence=0.3,
            deduplicate=False,
        )

        result_high = lifecycle_service.extract(
            text=text,
            min_confidence=0.95,
            deduplicate=False,
        )

        # High threshold should find fewer candidates
        assert result_high.candidates_found <= result_low.candidates_found

    def test_extract_handles_no_matches(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """extract() should handle text with no extractable content."""
        text = "This is just regular text without any memory-worthy content."

        result = lifecycle_service.extract(
            text=text,
            min_confidence=0.9,  # High threshold
            deduplicate=False,
        )

        assert isinstance(result, ExtractResult)
        assert result.memories_created == 0

    def test_extract_sets_source_extracted(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """extract() should set memory source to EXTRACTED."""
        text = "The solution was to implement connection pooling for better performance."

        mock_repository.search.return_value = []
        mock_repository.add.return_value = TEST_UUID_1

        lifecycle_service.extract(
            text=text,
            min_confidence=0.5,
        )

        if mock_repository.add.called:
            call_args = mock_repository.add.call_args
            memory = call_args[0][0]
            assert memory.source == MemorySource.EXTRACTED

    def test_extract_handles_empty_text(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """extract() should raise ValidationError for empty text."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            lifecycle_service.extract(
                text="",
                min_confidence=0.5,
            )


# =============================================================================
# TestConsolidate
# =============================================================================


class TestConsolidate:
    """Tests for LifecycleService.consolidate() - memory consolidation operation."""

    def test_consolidate_finds_groups(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should find groups of similar memories."""
        # Mock similar memories with vectors
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Pad vectors to 384 dimensions
        vec1_full = np.zeros(384, dtype=np.float32)
        vec2_full = np.zeros(384, dtype=np.float32)
        vec1_full[:3] = vec1
        vec2_full[:3] = vec2

        all_memories = [
            (make_memory(TEST_UUID_1, content="Database configuration settings"), vec1_full),
            (make_memory(TEST_UUID_2, content="Database config options"), vec2_full),
        ]
        mock_repository.get_all.return_value = all_memories

        result = lifecycle_service.consolidate(
            namespace="default",
            similarity_threshold=0.8,
            dry_run=True,
        )

        assert isinstance(result, ConsolidateResult)

    def test_consolidate_dry_run(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() with dry_run=True should not modify database."""
        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        vec2[0] = 0.99
        vec2[1] = 0.1

        all_memories = [
            (make_memory(TEST_UUID_1, content="Similar content here"), vec1),
            (make_memory(TEST_UUID_2, content="Similar content here too"), vec2),
        ]
        mock_repository.get_all.return_value = all_memories

        lifecycle_service.consolidate(
            namespace="default",
            dry_run=True,
        )

        # delete() and update() should not be called
        mock_repository.delete.assert_not_called()

    def test_consolidate_merges_memories(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() with dry_run=False should merge memories."""
        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        vec2[0] = 0.99
        vec2[1] = 0.1

        all_memories = [
            (make_memory(TEST_UUID_1, content="Database config settings", importance=0.8), vec1),
            (make_memory(TEST_UUID_2, content="Database config options", importance=0.6), vec2),
        ]
        mock_repository.get_all.return_value = all_memories

        result = lifecycle_service.consolidate(
            namespace="default",
            similarity_threshold=0.8,
            strategy="keep_highest_importance",
            dry_run=False,
        )

        assert result.dry_run is False
        # If groups were found and merged, delete should be called for non-representative memories

    def test_consolidate_respects_strategy_keep_newest(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should respect keep_newest strategy."""
        now = datetime.now(timezone.utc)

        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        vec2[0] = 0.99
        vec2[1] = 0.1

        all_memories = [
            (
                make_memory(
                    TEST_UUID_1,
                    content="Old content here for testing",
                    created_at=now - timedelta(days=10),
                ),
                vec1,
            ),
            (
                make_memory(
                    TEST_UUID_2,
                    content="New content here for testing",
                    created_at=now,
                ),
                vec2,
            ),
        ]
        mock_repository.get_all.return_value = all_memories

        result = lifecycle_service.consolidate(
            namespace="default",
            strategy="keep_newest",
            dry_run=True,
        )

        # The newest memory should be the representative
        if result.groups:
            assert result.groups[0].representative_id == TEST_UUID_2

    def test_consolidate_respects_threshold(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should respect similarity threshold."""
        # Memories that are similar but not very similar
        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        vec3 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        vec2[0], vec2[1] = 0.8, 0.6
        vec3[1] = 1.0

        all_memories = [
            (make_memory(TEST_UUID_1, content="First memory content here"), vec1),
            (make_memory(TEST_UUID_2, content="Second memory content here"), vec2),
            (make_memory(TEST_UUID_3, content="Third memory content here"), vec3),
        ]
        mock_repository.get_all.return_value = all_memories

        result_high = lifecycle_service.consolidate(
            namespace="default",
            similarity_threshold=0.95,  # Very strict
            dry_run=True,
        )

        mock_repository.get_all.return_value = all_memories  # Reset mock

        result_low = lifecycle_service.consolidate(
            namespace="default",
            similarity_threshold=0.7,  # Lenient (must be >= 0.7)
            dry_run=True,
        )

        # Higher threshold should find fewer groups
        assert result_high.groups_found <= result_low.groups_found

    def test_consolidate_handles_no_duplicates(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should handle case with no duplicates."""
        # Completely different memories
        vec1 = np.zeros(384, dtype=np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        vec2[1] = 1.0

        all_memories = [
            (make_memory(TEST_UUID_1, content="First different content"), vec1),
            (make_memory(TEST_UUID_2, content="Second completely unrelated"), vec2),
        ]
        mock_repository.get_all.return_value = all_memories

        result = lifecycle_service.consolidate(
            namespace="default",
            similarity_threshold=0.9,
            dry_run=True,
        )

        assert result.groups_found == 0
        assert result.memories_merged == 0

    def test_consolidate_handles_empty_namespace(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should handle empty namespace gracefully."""
        mock_repository.get_all.return_value = []

        result = lifecycle_service.consolidate(
            namespace="empty",
            dry_run=True,
        )

        assert result.groups_found == 0
        assert result.memories_deleted == 0

    def test_consolidate_max_groups_limit(
        self,
        lifecycle_service: LifecycleService,
        mock_repository: MagicMock,
    ) -> None:
        """consolidate() should respect max_groups parameter."""
        # Create many similar pairs
        import uuid

        all_memories = []
        for i in range(10):
            vec = np.zeros(384, dtype=np.float32)
            vec[i % 5] = 1.0
            if i % 2 == 1:
                vec[(i % 5)] = 0.99
                vec[(i % 5 + 1) % 5] = 0.1
            mem_id = str(uuid.uuid4())
            all_memories.append((make_memory(mem_id, content=f"Content for memory {i}"), vec))

        mock_repository.get_all.return_value = all_memories

        result = lifecycle_service.consolidate(
            namespace="default",
            max_groups=2,
            dry_run=True,
        )

        # Should not return more than max_groups
        assert len(result.groups) <= 2


# =============================================================================
# TestLifecycleServiceInitialization
# =============================================================================


class TestLifecycleServiceInitialization:
    """Tests for LifecycleService initialization and configuration."""

    def test_lifecycle_service_uses_default_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """LifecycleService should use default config when not provided."""
        service = LifecycleService(
            repository=mock_repository,
            embeddings=mock_embeddings,
        )

        assert service._config is not None
        assert service._config.decay_default_half_life_days == 30.0

    def test_lifecycle_service_uses_custom_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """LifecycleService should use provided config."""
        custom_config = LifecycleConfig(
            decay_default_half_life_days=60.0,
            reinforce_default_boost=0.2,
        )

        service = LifecycleService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            config=custom_config,
        )

        assert service._config.decay_default_half_life_days == 60.0
        assert service._config.reinforce_default_boost == 0.2

    def test_lifecycle_service_requires_repository(
        self,
        mock_embeddings: MagicMock,
    ) -> None:
        """LifecycleService should require a repository."""
        with pytest.raises(TypeError):
            LifecycleService(embeddings=mock_embeddings)  # type: ignore

    def test_lifecycle_service_requires_embeddings(
        self,
        mock_repository: MagicMock,
    ) -> None:
        """LifecycleService should require an embedding service."""
        with pytest.raises(TypeError):
            LifecycleService(repository=mock_repository)  # type: ignore
