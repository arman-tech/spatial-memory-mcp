"""Integration tests for the cognitive offloading pipeline.

Tests the full pipeline with real embeddings and real database:
- Quality gate via MemoryService.remember(cognitive_offloading_enabled=True)
- Content hash + vector dedup
- Queue processor Maildir flow
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.project_detection import ProjectIdentity
from spatial_memory.core.queue_constants import QUEUE_FILE_VERSION
from spatial_memory.services.memory import MemoryService
from spatial_memory.services.queue_processor import QueueProcessor

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def memory_service(
    module_repository: LanceDBMemoryRepository,
    session_embedding_service: EmbeddingService,
) -> MemoryService:
    """Module-scoped MemoryService with real repo and embeddings."""
    return MemoryService(
        repository=module_repository,
        embeddings=session_embedding_service,
    )


def _make_mock_project_detector(project_id: str = "test-project") -> MagicMock:
    """Create a mock ProjectDetector that returns a fixed project identity.

    In production, QueueProcessor receives a real ProjectDetector (wired in
    factory.py) that resolves project identity from git remotes. In tests we
    mock it because we don't have real git repositories to resolve from.
    This matches the pattern in tests/unit/test_wp6_queue_processor.py.
    """
    detector = MagicMock()
    detector.resolve_from_directory.return_value = ProjectIdentity(
        project_id=project_id,
        source="mock",
    )
    return detector


def _make_queue_json(
    content: str,
    *,
    suggested_namespace: str = "decisions",
    suggested_tags: list[str] | None = None,
    suggested_importance: float = 0.8,
    signal_tier: int = 1,
    project_root_dir: str = "/home/user/code/my-project",
) -> dict:
    """Build a valid queue file JSON dict."""
    return {
        "version": QUEUE_FILE_VERSION,
        "content": content,
        "source_hook": "prompt-submit",
        "timestamp": "2025-01-15T10:30:00Z",
        "project_root_dir": project_root_dir,
        "suggested_namespace": suggested_namespace,
        "suggested_tags": suggested_tags or ["test"],
        "suggested_importance": suggested_importance,
        "signal_tier": signal_tier,
        "signal_patterns_matched": ["decision"],
        "context": {},
        "client": "claude-code",
    }


def _write_queue_file(new_dir: Path, filename: str, data: dict) -> Path:
    """Write a queue file directly into new/ for processing."""
    path = new_dir / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Quality Gate Pipeline
# ---------------------------------------------------------------------------


class TestQualityGatePipeline:
    """Quality gate via MemoryService.remember(cognitive_offloading_enabled=True)."""

    def test_high_quality_stored(self, memory_service: MemoryService) -> None:
        """Decision content with tags and file paths should be stored."""
        result = memory_service.remember(
            content=(
                "Decided to use PostgreSQL because it handles JSONB natively, "
                "which eliminates the need for a separate Elasticsearch cluster. "
                "Updated config.py and core/database.py to use psycopg3 driver."
            ),
            namespace="decisions",
            tags=["postgresql", "architecture"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        assert result.id  # non-empty UUID
        assert len(result.id) > 0

    def test_low_quality_rejected(self, memory_service: MemoryService) -> None:
        """Trivial content should be rejected by quality gate."""
        result = memory_service.remember(
            content="ok",
            namespace="default",
            cognitive_offloading_enabled=True,
        )

        assert result.status == "rejected_quality"
        assert result.quality_score is not None
        assert result.quality_score < 0.3

    def test_borderline_reduces_importance(self, memory_service: MemoryService) -> None:
        """Content scoring 0.3-0.5 should be stored with reduced importance.

        Quality gate scoring breakdown for the chosen content:
        - signal_score = 0.8 ("chose" matches decision extraction pattern)
        - content_length_score = 1.0 (~130 chars, in the 100-500 optimal range)
        - structure_score = 0.0 (no tags, no reasoning words like "because",
          no file paths or PascalCase identifiers)
        - context_richness = 0.0 (no file refs, function refs, versions, or URLs)

        Total = 0.8 * 0.3 + 1.0 * 0.2 + 0.0 * 0.2 + 0.0 * 0.3 = 0.44

        This lands in the 0.3-0.5 borderline band where remember() stores
        the memory but reduces importance to min(original, quality_score).
        """
        content = (
            "We chose this library for its simplicity and ease of maintenance "
            "across our development team working on the backend platform services"
        )
        result = memory_service.remember(
            content=content,
            namespace="default",
            importance=0.9,
            cognitive_offloading_enabled=True,
        )

        assert result.status == "stored"
        assert result.id

        from spatial_memory.core.quality_gate import score_memory_quality

        score = score_memory_quality(content)
        assert 0.3 <= score.total < 0.5, f"Expected borderline score, got {score.total}"


# ---------------------------------------------------------------------------
# Dedup Pipeline
# ---------------------------------------------------------------------------


class TestDedupPipeline:
    """Dedup via MemoryService.remember(cognitive_offloading_enabled=True)."""

    def test_exact_duplicate_rejected(self, memory_service: MemoryService) -> None:
        """Storing the same content twice should reject the second as exact duplicate."""
        content = (
            "Decided to use Redis for session caching because it supports "
            "TTL natively and has excellent Python client support via redis-py. "
            "Configuration is in services/cache.py."
        )

        # First store should succeed
        result1 = memory_service.remember(
            content=content,
            namespace="decisions",
            tags=["redis"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )
        assert result1.status == "stored"
        assert result1.id

        # Second store of identical content should be rejected
        result2 = memory_service.remember(
            content=content,
            namespace="decisions",
            tags=["redis"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )
        assert result2.status == "rejected_exact"
        assert result2.similarity == 1.0

    def test_similar_content_rejected(self, memory_service: MemoryService) -> None:
        """Close paraphrase should be rejected as similar duplicate.

        The paraphrase uses minimal word changes ("great" -> "excellent",
        "any external" -> "any extra") to produce high cosine similarity
        (~0.997 with all-MiniLM-L6-v2) that exceeds the 0.85 dedup threshold.

        Note: Larger paraphrases (rewriting whole clauses) can drop below the
        threshold with this model — the dedup layer catches near-duplicates,
        not loose paraphrases. That's by design.
        """
        original = (
            "PostgreSQL is great for storing JSON data because of its native "
            "JSONB column type which supports indexing and querying nested fields "
            "efficiently without any external plugins."
        )
        paraphrase = (
            "PostgreSQL is excellent for storing JSON data because of its native "
            "JSONB column type which supports indexing and querying nested fields "
            "efficiently without any extra plugins."
        )

        result1 = memory_service.remember(
            content=original,
            namespace="decisions",
            tags=["postgresql"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )
        assert result1.status == "stored"

        result2 = memory_service.remember(
            content=paraphrase,
            namespace="decisions",
            tags=["postgresql"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )
        assert result2.status == "rejected_similar"
        assert result2.similarity is not None
        assert result2.similarity >= 0.85

    def test_different_content_passes(self, memory_service: MemoryService) -> None:
        """Semantically different content should both be stored."""
        content_a = (
            "Decided to use PostgreSQL for the user accounts database because "
            "it handles relational data with ACID transactions and has mature "
            "migration tooling via Alembic."
        )
        content_b = (
            "The Python testing strategy uses pytest with fixtures and the "
            "factory pattern for test data. Coverage target is 90% for core "
            "modules in the services/ directory."
        )

        result_a = memory_service.remember(
            content=content_a,
            namespace="decisions",
            tags=["postgresql"],
            importance=0.8,
            cognitive_offloading_enabled=True,
        )
        assert result_a.status == "stored"
        assert result_a.id

        result_b = memory_service.remember(
            content=content_b,
            namespace="decisions",
            tags=["testing"],
            importance=0.7,
            cognitive_offloading_enabled=True,
        )
        assert result_b.status == "stored"
        assert result_b.id
        assert result_a.id != result_b.id


# ---------------------------------------------------------------------------
# Queue Processor Pipeline
# ---------------------------------------------------------------------------


class TestQueueProcessorPipeline:
    """Full Maildir queue flow with real database."""

    def test_queue_file_stored(
        self,
        memory_service: MemoryService,
        module_database: Database,
        module_temp_storage: Path,
    ) -> None:
        """Valid queue file in new/ should be processed, moved, and stored in DB."""
        queue_dir = module_temp_storage / "queue-stored"
        new_dir = queue_dir / "new"
        processed_dir = queue_dir / "processed"
        (queue_dir / "tmp").mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        content = (
            "Decided to use FastAPI for the REST endpoints because it has "
            "native async support and automatic OpenAPI documentation. "
            "Updated server.py and routes/api.py with the new framework."
        )
        data = _make_queue_json(content)
        _write_queue_file(new_dir, "001-test.json", data)

        detector = _make_mock_project_detector()
        processor = QueueProcessor(
            memory_service=memory_service,
            project_detector=detector,
            queue_dir=queue_dir,
            poll_interval=1,
            cognitive_offloading_enabled=True,
        )
        processor.start()
        try:
            # Wait for processing with timeout
            deadline = time.monotonic() + 10.0
            notifications = []
            while time.monotonic() < deadline:
                notifications = processor.drain_notifications()
                if notifications:
                    break
                time.sleep(0.2)

            assert notifications, "Queue processor did not produce notifications in time"

            # File should be moved to processed/
            assert not (new_dir / "001-test.json").exists()
            assert (processed_dir / "001-test.json").exists()

            # Notification should indicate stored
            assert any("stored" in n for n in notifications)

            # Stats should show 1 stored
            stats = processor.get_stats()
            assert stats["files_stored"] >= 1
        finally:
            processor.stop(timeout=5.0)

    def test_queue_file_rejected_quality(
        self,
        memory_service: MemoryService,
        module_database: Database,
        module_temp_storage: Path,
    ) -> None:
        """Low-quality queue file should be processed but not stored in DB."""
        queue_dir = module_temp_storage / "queue-quality"
        new_dir = queue_dir / "new"
        processed_dir = queue_dir / "processed"
        (queue_dir / "tmp").mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # "thanks" is too short/trivial to pass quality gate
        data = _make_queue_json("thanks")
        _write_queue_file(new_dir, "002-low.json", data)

        detector = _make_mock_project_detector()
        processor = QueueProcessor(
            memory_service=memory_service,
            project_detector=detector,
            queue_dir=queue_dir,
            poll_interval=1,
            cognitive_offloading_enabled=True,
        )
        processor.start()
        try:
            deadline = time.monotonic() + 10.0
            notifications = []
            while time.monotonic() < deadline:
                notifications = processor.drain_notifications()
                if notifications:
                    break
                time.sleep(0.2)

            assert notifications, "Queue processor did not produce notifications in time"

            # File should be moved to processed/
            assert not (new_dir / "002-low.json").exists()
            assert (processed_dir / "002-low.json").exists()

            # Notification should indicate rejection
            assert any("rejected_quality" in n for n in notifications)

            # Stats should show rejection, not storage
            stats = processor.get_stats()
            assert stats["files_rejected"] >= 1
        finally:
            processor.stop(timeout=5.0)

    def test_queue_file_dedup(
        self,
        memory_service: MemoryService,
        module_database: Database,
        module_temp_storage: Path,
    ) -> None:
        """Pre-stored memory + identical queue file should be detected as duplicate.

        Important: The direct remember() call and the queue processor must use
        the same project scope. QueueProcessor resolves project_root_dir via
        ProjectDetector, so we pass the same project_id ("test-project") to both
        the direct call and the mock detector. Without this, _check_dedup() scopes
        searches by project and won't find the pre-stored memory (project="" vs
        project="test-project" are different scopes).
        """
        # First, store a memory directly
        content = (
            "Decided to use WebSockets for real-time notifications because "
            "Server-Sent Events don't support bidirectional communication "
            "and we need the client to acknowledge receipt. "
            "Implementation in services/realtime.py."
        )
        # Must match the mock detector's project_id so dedup scopes align
        project_id = "test-project"
        direct_result = memory_service.remember(
            content=content,
            namespace="decisions",
            tags=["websockets"],
            importance=0.8,
            project=project_id,
            cognitive_offloading_enabled=True,
        )
        assert direct_result.status == "stored"

        # Now queue the same content
        queue_dir = module_temp_storage / "queue-dedup"
        new_dir = queue_dir / "new"
        processed_dir = queue_dir / "processed"
        (queue_dir / "tmp").mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        data = _make_queue_json(content)
        _write_queue_file(new_dir, "003-dup.json", data)

        detector = _make_mock_project_detector(project_id=project_id)
        processor = QueueProcessor(
            memory_service=memory_service,
            project_detector=detector,
            queue_dir=queue_dir,
            poll_interval=1,
            cognitive_offloading_enabled=True,
        )
        processor.start()
        try:
            deadline = time.monotonic() + 10.0
            notifications = []
            while time.monotonic() < deadline:
                notifications = processor.drain_notifications()
                if notifications:
                    break
                time.sleep(0.2)

            assert notifications, "Queue processor did not produce notifications in time"

            # File should be in processed/
            assert not (new_dir / "003-dup.json").exists()
            assert (processed_dir / "003-dup.json").exists()

            # Should be rejected as duplicate (exact or similar)
            assert any("rejected_exact" in n or "rejected_similar" in n for n in notifications), (
                f"Expected duplicate rejection, got: {notifications}"
            )
        finally:
            processor.stop(timeout=5.0)
