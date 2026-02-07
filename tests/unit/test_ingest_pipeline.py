"""Unit tests for IngestPipeline.

Tests the pipeline in isolation with mocked repository and embeddings.
Covers: basic ingest, hash dedup, vector dedup, quality gate, config handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from spatial_memory.core.hashing import compute_content_hash
from spatial_memory.core.models import Memory, MemoryResult, MemorySource
from spatial_memory.core.quality_gate import score_memory_quality
from spatial_memory.services.ingest_pipeline import (
    DedupCheckResult,
    IngestConfig,
    IngestPipeline,
    RememberResult,
)

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
    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = make_vector()
    type(emb).dimensions = PropertyMock(return_value=384)
    return emb


@pytest.fixture
def pipeline(mock_repo: MagicMock, mock_embeddings: MagicMock) -> IngestPipeline:
    return IngestPipeline(repository=mock_repo, embeddings=mock_embeddings)


# =============================================================================
# TestIngestBasic
# =============================================================================


class TestIngestBasic:
    def test_stores_memory_and_returns_result(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        result = pipeline.ingest(content="some content")

        assert result.id == UUID_2
        assert result.status == "stored"
        assert result.deduplicated is False
        assert result.namespace == "default"
        mock_repo.add.assert_called_once()

    def test_computes_content_hash(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        pipeline.ingest(content="test content")

        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.content_hash == compute_content_hash("test content")

    def test_passes_tags_and_metadata(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        pipeline.ingest(
            content="test",
            tags=["a", "b"],
            metadata={"key": "val"},
        )

        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.tags == ["a", "b"]
        assert memory_arg.metadata == {"key": "val"}

    def test_passes_namespace_and_project(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        pipeline.ingest(content="test", namespace="ns", project="proj")

        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.namespace == "ns"
        assert memory_arg.project == "proj"

    def test_default_config_when_none(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        """When config is None, uses defaults (no cognitive offloading)."""
        result = pipeline.ingest(content="test content", config=None)

        assert result.status == "stored"
        # No dedup checks when cognitive_offloading_enabled=False (default)
        mock_repo.find_by_content_hash.assert_not_called()


# =============================================================================
# TestIngestHashDedup
# =============================================================================


class TestIngestHashDedup:
    def test_exact_match_rejected(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        existing = make_memory(id=UUID_1, content="duplicate content")
        mock_repo.find_by_content_hash.return_value = existing
        # Simulate that this hash was stored in a previous call
        pipeline._stored_hashes.add(compute_content_hash("duplicate content"))

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(content="duplicate content", config=config)

        assert result.status == "rejected_exact"
        assert result.deduplicated is True
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 1.0
        mock_repo.add.assert_not_called()

    def test_no_match_proceeds(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(
            content="We decided to use Redis because it provides fast caching for our API.",
            config=config,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()

    def test_skipped_when_disabled(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        config = IngestConfig(cognitive_offloading_enabled=False)
        result = pipeline.ingest(content="anything", config=config)

        assert result.status == "stored"
        mock_repo.find_by_content_hash.assert_not_called()


# =============================================================================
# TestIngestVectorDedup
# =============================================================================


class TestIngestVectorDedup:
    def test_high_similarity_rejected(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.92)]

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(content="paraphrased content", config=config)

        assert result.status == "rejected_similar"
        assert result.deduplicated is True
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 0.92

    def test_borderline_flagged(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.82)]

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(content="similar content", config=config)

        assert result.status == "potential_duplicate"
        assert result.deduplicated is False
        assert result.existing_memory_id == UUID_1
        assert result.similarity == 0.82
        mock_repo.add.assert_not_called()

    def test_low_similarity_accepted(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.5)]

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(
            content="We decided to use Redis because it provides fast caching for our API.",
            config=config,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()

    def test_no_search_results_accepted(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(
            content="We decided to use Redis because it provides fast caching for our API.",
            config=config,
        )

        assert result.status == "stored"
        mock_repo.add.assert_called_once()

    def test_custom_threshold_respected(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.87)]

        config = IngestConfig(cognitive_offloading_enabled=True, dedup_threshold=0.90)
        result = pipeline.ingest(content="custom threshold test", config=config)

        # 0.87 is above 0.80 but below 0.90 -> potential_duplicate
        assert result.status == "potential_duplicate"


# =============================================================================
# TestIngestQualityGate
# =============================================================================


class TestIngestQualityGate:
    def test_low_quality_rejected(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = []

        config = IngestConfig(cognitive_offloading_enabled=True, signal_threshold=0.3)
        result = pipeline.ingest(content="ok", config=config)

        assert result.status == "rejected_quality"
        assert result.quality_score is not None
        assert result.quality_score < 0.3
        assert result.id == ""
        mock_repo.add.assert_not_called()

    def test_borderline_reduces_importance(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.1)]

        content = "The fix was adjusting the timeout."
        config = IngestConfig(cognitive_offloading_enabled=True, signal_threshold=0.2)
        result = pipeline.ingest(content=content, importance=0.8, config=config)

        if result.status == "stored":
            call_args = mock_repo.add.call_args
            memory_arg = call_args[0][0]
            quality = score_memory_quality(content)
            if quality.total < 0.5:
                assert memory_arg.importance <= 0.5

    def test_high_quality_preserves_importance(
        self, pipeline: IngestPipeline, mock_repo: MagicMock
    ) -> None:
        mock_repo.find_by_content_hash.return_value = None
        mock_repo.search.return_value = [make_memory_result(similarity=0.1)]

        content = (
            "Decided to use LanceDB v0.6 for the database layer because it supports "
            "hybrid search with IVF-PQ indexing. The function create_index() in database.py "
            "handles all index creation."
        )
        config = IngestConfig(cognitive_offloading_enabled=True)
        result = pipeline.ingest(
            content=content,
            importance=0.8,
            tags=["architecture"],
            config=config,
        )

        assert result.status == "stored"
        call_args = mock_repo.add.call_args
        memory_arg = call_args[0][0]
        assert memory_arg.importance == 0.8

    def test_skipped_when_disabled(self, pipeline: IngestPipeline, mock_repo: MagicMock) -> None:
        """Low quality content still stored when cognitive offloading disabled."""
        config = IngestConfig(cognitive_offloading_enabled=False)
        result = pipeline.ingest(content="ok", config=config)

        assert result.status == "stored"
        assert result.quality_score is None
        mock_repo.add.assert_called_once()


# =============================================================================
# TestIngestConfig
# =============================================================================


class TestIngestConfig:
    def test_default_values(self) -> None:
        config = IngestConfig()
        assert config.cognitive_offloading_enabled is False
        assert config.dedup_threshold == 0.85
        assert config.signal_threshold == 0.3

    def test_custom_values(self) -> None:
        config = IngestConfig(
            cognitive_offloading_enabled=True,
            dedup_threshold=0.90,
            signal_threshold=0.5,
        )
        assert config.cognitive_offloading_enabled is True
        assert config.dedup_threshold == 0.90
        assert config.signal_threshold == 0.5


# =============================================================================
# TestDedupCheckResult
# =============================================================================


class TestDedupCheckResultPipeline:
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
# TestRememberResult
# =============================================================================


class TestRememberResultPipeline:
    def test_default_values(self) -> None:
        r = RememberResult(id="abc", content="test", namespace="default")
        assert r.deduplicated is False
        assert r.status == "stored"
        assert r.quality_score is None
        assert r.existing_memory_id is None
        assert r.existing_memory_content is None
        assert r.similarity is None

    def test_rejected_values(self) -> None:
        r = RememberResult(
            id="abc",
            content="test",
            namespace="default",
            deduplicated=True,
            status="rejected_exact",
            existing_memory_id="xyz",
            existing_memory_content="original",
            similarity=1.0,
        )
        assert r.deduplicated is True
        assert r.status == "rejected_exact"
        assert r.existing_memory_id == "xyz"
        assert r.similarity == 1.0
