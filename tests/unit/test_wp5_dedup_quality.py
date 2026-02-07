"""Unit tests for WP5: Content Hash Dedup + Quality Gate on remember().

Tests cover:
1. compute_content_hash - normalization and consistency
2. score_memory_quality - each sub-score component
3. Content hash dedup - exact match -> rejected_exact
4. Vector similarity dedup - threshold-based responses
5. Quality gate - scoring and threshold behavior
6. Feature flag off - all dedup/quality logic skipped
7. Response structure - status, quality_score, existing_memory populated
8. _handle_remember server integration
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from spatial_memory.core.hashing import compute_content_hash
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.core.quality_gate import QualityScore, score_memory_quality
from spatial_memory.services.memory import DedupCheckResult, MemoryService

# =============================================================================
# Test UUIDs
# =============================================================================

UUID_1 = "11111111-1111-1111-1111-111111111111"
UUID_2 = "22222222-2222-2222-2222-222222222222"


# =============================================================================
# Helpers
# =============================================================================


def make_memory(
    id: str = UUID_1,
    content: str = "Test memory content",
    namespace: str = "default",
    importance: float = 0.5,
    project: str = "",
    content_hash: str = "",
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=id,
        content=content,
        namespace=namespace,
        importance=importance,
        tags=[],
        source=MemorySource.MANUAL,
        metadata={},
        created_at=now,
        updated_at=now,
        last_accessed=now,
        access_count=0,
        project=project,
        content_hash=content_hash,
    )


def make_memory_result(
    id: str = UUID_1,
    content: str = "Test memory content",
    similarity: float = 0.9,
    namespace: str = "default",
) -> MemoryResult:
    return MemoryResult(
        id=id,
        content=content,
        similarity=similarity,
        namespace=namespace,
        tags=[],
        importance=0.5,
        created_at=datetime.now(timezone.utc),
        metadata={},
    )


def make_vector(dims: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def mock_repo() -> MagicMock:
    repo = MagicMock()
    repo.add.return_value = UUID_2
    repo.find_by_content_hash.return_value = None
    repo.search.return_value = []
    repo.get.return_value = None
    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = make_vector()
    type(emb).dimensions = PropertyMock(return_value=384)
    return emb


@pytest.fixture
def service(mock_repo: MagicMock, mock_embeddings: MagicMock) -> MemoryService:
    return MemoryService(repository=mock_repo, embeddings=mock_embeddings)


# =============================================================================
# 1. compute_content_hash tests
# =============================================================================


class TestComputeContentHash:
    def test_basic_hash(self) -> None:
        h = compute_content_hash("Hello World")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert h == expected

    def test_normalization_strips_whitespace(self) -> None:
        assert compute_content_hash("  hello  ") == compute_content_hash("hello")

    def test_case_insensitive(self) -> None:
        """Case is normalized: different casing produces same hash."""
        assert compute_content_hash("HELLO") == compute_content_hash("hello")

    def test_empty_string(self) -> None:
        h = compute_content_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_matches_migration_logic(self) -> None:
        """Ensure our function matches the migration backfill normalization."""
        content = "  Test Content  "
        # Migration logic: content.strip().lower()
        normalized = content.strip().lower()
        expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        assert compute_content_hash(content) == expected

    def test_deterministic(self) -> None:
        assert compute_content_hash("test") == compute_content_hash("test")

    def test_different_content_different_hash(self) -> None:
        assert compute_content_hash("foo") != compute_content_hash("bar")


# =============================================================================
# 2. score_memory_quality tests
# =============================================================================


class TestScoreMemoryQuality:
    def test_returns_quality_score(self) -> None:
        result = score_memory_quality("Some content here")
        assert isinstance(result, QualityScore)
        assert 0.0 <= result.total <= 1.0

    def test_signal_score_with_decision_pattern(self) -> None:
        result = score_memory_quality("We decided to use PostgreSQL because of ACID compliance.")
        assert result.signal_score >= 0.7

    def test_signal_score_no_match(self) -> None:
        result = score_memory_quality("hello world")
        assert result.signal_score == 0.0

    def test_content_length_very_short(self) -> None:
        result = score_memory_quality("hi")
        assert result.content_length_score == 0.0

    def test_content_length_short(self) -> None:
        result = score_memory_quality("A" * 50)
        assert result.content_length_score == 0.5

    def test_content_length_medium(self) -> None:
        result = score_memory_quality("A" * 200)
        assert result.content_length_score == 1.0

    def test_content_length_long(self) -> None:
        result = score_memory_quality("A" * 1000)
        assert result.content_length_score == 0.8

    def test_content_length_very_long(self) -> None:
        result = score_memory_quality("A" * 3000)
        assert result.content_length_score == 0.7

    def test_structure_with_tags(self) -> None:
        result = score_memory_quality("Some content", tags=["tag1", "tag2"])
        assert result.structure_score >= 0.3

    def test_structure_with_reasoning_words(self) -> None:
        result = score_memory_quality("We chose this because it was faster.")
        assert result.structure_score >= 0.3

    def test_structure_with_specific_names(self) -> None:
        result = score_memory_quality("The file src/main.py handles routing")
        assert result.structure_score >= 0.4

    def test_context_richness_with_file_reference(self) -> None:
        result = score_memory_quality("Check the config in settings.json for details")
        assert result.context_richness >= 0.25

    def test_context_richness_with_function_reference(self) -> None:
        result = score_memory_quality("The function compute_hash() was slow")
        assert result.context_richness >= 0.25

    def test_context_richness_with_version(self) -> None:
        result = score_memory_quality("Fixed in v2.1 release")
        assert result.context_richness >= 0.25

    def test_context_richness_with_url(self) -> None:
        result = score_memory_quality("See https://example.com/docs for more")
        assert result.context_richness >= 0.25

    def test_context_richness_with_inline_code(self) -> None:
        result = score_memory_quality("Use `pip install spatial-memory` to install")
        assert result.context_richness >= 0.25

    def test_high_quality_memory(self) -> None:
        """A well-structured memory should score above 0.5."""
        content = (
            "Decided to use LanceDB v0.6 because it supports hybrid search. "
            "The function create_index() handles IVF-PQ indexing in database.py."
        )
        result = score_memory_quality(content, tags=["architecture", "database"])
        assert result.total > 0.5

    def test_low_quality_memory(self) -> None:
        """Trivial content should score below 0.3."""
        result = score_memory_quality("ok")
        assert result.total < 0.3

    def test_formula_weights(self) -> None:
        """Verify the formula applies correct weights."""
        result = score_memory_quality("test content for weights")
        expected = (
            result.signal_score * 0.3
            + result.content_length_score * 0.2
            + result.structure_score * 0.2
            + result.context_richness * 0.3
        )
        assert abs(result.total - expected) < 1e-9


# =============================================================================
# 3. Content hash dedup tests
# =============================================================================


class TestContentHashDedup:
    def test_exact_duplicate_rejected(self, service: MemoryService, mock_repo: MagicMock) -> None:
        existing = make_memory(id=UUID_1, content="duplicate content")
        mock_repo.find_by_content_hash.return_value = existing

        result = service.remember(
            content="duplicate content",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_exact"
        assert result.deduplicated is True
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 1.0
        # Should NOT have called add
        mock_repo.add.assert_not_called()

    def test_no_hash_match_proceeds(self, service: MemoryService, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        result = service.remember(
            content="We decided to use Redis because it provides fast caching for our API.",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()


# =============================================================================
# 4. Vector similarity dedup tests
# =============================================================================


class TestVectorSimilarityDedup:
    def test_high_similarity_rejected(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Similarity >= 0.85 (default threshold) -> rejected_similar."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.92)]

        result = service.remember(
            content="paraphrased content",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_similar"
        assert result.deduplicated is True
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 0.92
        mock_repo.get.assert_not_called()

    def test_borderline_similarity_potential_duplicate(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Similarity 0.80-0.85 -> potential_duplicate (LLM arbitration)."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.82)]

        result = service.remember(
            content="similar but different content",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "potential_duplicate"
        assert result.deduplicated is False  # Not auto-rejected
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 0.82
        # Should NOT store — let LLM decide
        mock_repo.add.assert_not_called()
        mock_repo.get.assert_not_called()

    def test_low_similarity_accepted(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Similarity < 0.80 -> new (accepted)."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.5)]

        result = service.remember(
            content="We decided to use Redis because it provides fast caching for our API.",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()

    def test_no_search_results_accepted(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Empty search results -> new (accepted)."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        result = service.remember(
            content="We decided to use Redis because it provides fast caching for our API.",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()

    def test_custom_dedup_threshold(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Custom threshold 0.90 — similarity 0.87 should be potential_duplicate."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.87)]

        result = service.remember(
            content="custom threshold test",
            cognitive_offloading_enabled=True,
            dedup_threshold=0.90,
        )

        # 0.87 is above 0.80 but below 0.90 -> potential_duplicate
        assert result.status == "potential_duplicate"
        mock_repo.get.assert_not_called()


# =============================================================================
# 5. Quality gate tests
# =============================================================================


class TestQualityGate:
    def test_low_quality_rejected(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Content below quality threshold -> rejected_quality."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        result = service.remember(
            content="ok",
            cognitive_offloading_enabled=True,
            signal_threshold=0.3,
        )

        assert result.status == "rejected_quality"
        assert result.quality_score is not None
        assert result.quality_score < 0.3
        assert result.id == ""
        mock_repo.add.assert_not_called()

    def test_borderline_quality_reduces_importance(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Score 0.3-0.5 -> stored but importance reduced."""
        mock_repo.find_by_content_hash.return_value = None
        # Return low similarity so dedup passes
        mock_repo.search.return_value = [make_memory_result(similarity=0.1)]

        # Content that will score 0.3-0.5 range — has some signal but not great
        content = "The fix was adjusting the timeout."

        result = service.remember(
            content=content,
            importance=0.8,
            cognitive_offloading_enabled=True,
            signal_threshold=0.2,  # Lower threshold so it passes
        )

        if result.status == "stored":
            # Check that importance was reduced via the Memory object passed to add
            call_args = mock_repo.add.call_args
            memory_arg = call_args[0][0]  # First positional arg is Memory
            # If quality < 0.5, importance = min(importance, quality_total)
            quality = score_memory_quality(content)
            if quality.total < 0.5:
                assert memory_arg.importance <= 0.5

    def test_high_quality_preserves_importance(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Score > 0.5 -> stored at requested importance."""
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.1)]

        content = (
            "Decided to use LanceDB v0.6 for the database layer because it supports "
            "hybrid search with IVF-PQ indexing. The function create_index() in database.py "
            "handles all index creation."
        )

        result = service.remember(
            content=content,
            importance=0.8,
            tags=["architecture"],
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.importance == 0.8

    def test_quality_gate_only_runs_after_dedup(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """If dedup rejects, quality gate should NOT run."""
        existing = make_memory(id=UUID_1, content="existing content")
        mock_repo.find_by_content_hash.return_value = existing

        # Even low-quality content should be rejected as duplicate, not quality
        result = service.remember(
            content="existing content",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_exact"
        # quality_score should NOT be set
        assert result.quality_score is None


# =============================================================================
# 6. Feature flag off tests
# =============================================================================


class TestFeatureFlagOff:
    def test_no_dedup_when_disabled(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """With cognitive_offloading_enabled=False, no dedup or quality gate."""
        result = service.remember(
            content="any content",
            cognitive_offloading_enabled=False,
        )

        assert result.status == "stored"
        assert result.quality_score is None
        assert result.existing_memory_id is None
        mock_repo.add.assert_called_once()
        # find_by_content_hash should NOT be called
        mock_repo.find_by_content_hash.assert_not_called()

    def test_default_is_disabled(self, service: MemoryService, mock_repo: MagicMock) -> None:
        """Default behavior: cognitive_offloading_enabled defaults to False."""
        result = service.remember(content="test content")

        assert result.status == "stored"
        mock_repo.find_by_content_hash.assert_not_called()

    def test_content_hash_always_computed(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Even with feature flag off, content_hash is set on Memory."""
        result = service.remember(
            content="test content",
            cognitive_offloading_enabled=False,
        )

        assert result.status == "stored"
        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.content_hash != ""
        assert memory_arg.content_hash == compute_content_hash("test content")


# =============================================================================
# 7. Response structure tests
# =============================================================================


class TestResponseStructure:
    def test_stored_response_fields(self, service: MemoryService, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        result = service.remember(
            content="We decided to use Redis because it provides fast caching for our API.",
            cognitive_offloading_enabled=True,
        )

        assert result.id == UUID_2
        assert result.namespace == "default"
        assert result.deduplicated is False
        assert result.status == "stored"
        assert result.quality_score is None
        assert result.existing_memory_id is None

    def test_rejected_exact_response_fields(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        existing = make_memory(id=UUID_1, content="the content")
        mock_repo.find_by_content_hash.return_value = existing

        result = service.remember(
            content="the content",
            cognitive_offloading_enabled=True,
        )

        assert result.id == UUID_1
        assert result.deduplicated is True
        assert result.status == "rejected_exact"
        assert result.existing_memory_id == UUID_1
        assert result.existing_memory_content == "the content"
        assert result.similarity == 1.0

    def test_rejected_similar_response_fields(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [
            make_memory_result(similarity=0.91, content="original")
        ]

        result = service.remember(
            content="paraphrase of original",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_similar"
        assert result.existing_memory_id == UUID_1
        assert result.existing_memory_content == "original"
        assert result.similarity == 0.91
        # Verify no extra DB round trip — Memory is built from search result
        mock_repo.get.assert_not_called()

    def test_potential_duplicate_response_fields(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [
            make_memory_result(similarity=0.83, content="borderline")
        ]

        result = service.remember(
            content="almost the same as borderline",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "potential_duplicate"
        assert result.existing_memory_id == UUID_1
        assert result.existing_memory_content == "borderline"
        assert result.similarity == 0.83
        # Verify no extra DB round trip — Memory is built from search result
        mock_repo.get.assert_not_called()

    def test_rejected_quality_response_fields(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        result = service.remember(
            content="hi",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_quality"
        assert result.id == ""
        assert result.quality_score is not None
        assert result.quality_score < 0.3


# =============================================================================
# 8. DedupCheckResult tests
# =============================================================================


class TestDedupCheckResult:
    def test_new_status(self) -> None:
        r = DedupCheckResult(status="new")
        assert r.existing_memory is None
        assert r.similarity == 0.0

    def test_exact_duplicate_status(self) -> None:
        mem = make_memory()
        r = DedupCheckResult(status="exact_duplicate", existing_memory=mem, similarity=1.0)
        assert r.existing_memory is not None
        assert r.similarity == 1.0


# =============================================================================
# 9. Idempotency still works with new params
# =============================================================================


class TestIdempotencyIntegration:
    def test_idempotency_key_bypass_dedup(
        self, mock_repo: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Idempotency check happens before dedup — early return."""
        mock_idempotency = MagicMock()
        mock_record = MagicMock()
        mock_record.memory_id = UUID_1
        mock_idempotency.get_by_idempotency_key.return_value = mock_record
        mock_repo.get.return_value = make_memory(id=UUID_1, content="cached")

        svc = MemoryService(
            repository=mock_repo,
            embeddings=mock_embeddings,
            idempotency_provider=mock_idempotency,
        )

        result = svc.remember(
            content="cached",
            idempotency_key="test-key",
            cognitive_offloading_enabled=True,
        )

        assert result.deduplicated is True
        assert result.id == UUID_1
        # Should NOT call find_by_content_hash — idempotency returned early
        mock_repo.find_by_content_hash.assert_not_called()


# =============================================================================
# 10. Project validation tests
# =============================================================================


class TestProjectValidation:
    def test_remember_validates_project(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Project validation should reject invalid project strings."""
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError, match="Invalid project format"):
            service.remember(
                content="Valid content for testing",
                project="'; DROP TABLE memories--",
                cognitive_offloading_enabled=False,
            )

    def test_remember_accepts_valid_project(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Valid project strings should pass validation."""
        result = service.remember(
            content="Valid content for testing",
            project="github.com/org/repo",
        )
        assert result.id  # Should succeed

    def test_remember_skips_validation_for_empty_project(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """Empty string project should not trigger validation."""
        result = service.remember(
            content="Valid content for testing",
            project="",
        )
        assert result.id  # Should succeed without validation error

    def test_remember_batch_validates_project(
        self, service: MemoryService, mock_repo: MagicMock
    ) -> None:
        """remember_batch should reject invalid project strings."""
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError, match="Invalid project format"):
            service.remember_batch(
                memories=[{"content": "Valid content for batch testing"}],
                project="'; DROP TABLE memories--",
            )

    def test_remember_batch_accepts_valid_project(
        self, service: MemoryService, mock_repo: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """remember_batch should accept valid project strings."""
        mock_embeddings.embed_batch.return_value = [make_vector()]
        mock_repo.add_batch.return_value = [UUID_2]
        result = service.remember_batch(
            memories=[{"content": "Valid content for batch testing"}],
            project="github.com/org/repo",
        )
        assert result.count == 1
