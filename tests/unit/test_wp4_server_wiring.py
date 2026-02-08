"""Tests for WP4: Tool Schema + Server Wiring.

Verifies that:
1. _resolve_project correctly handles *, explicit, and auto-detect cases
2. Server handlers pass project through to service methods
3. project field appears in response dicts when set
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from spatial_memory.adapters.project_detection import (
    ProjectDetector,
    ProjectIdentity,
)
from spatial_memory.core.models import (
    HybridMemoryMatch,
    HybridRecallResult,
    MemoryResult,
    StatsResult,
)
from tests.unit.conftest import make_server_with_mocks as _make_server_with_mocks

NOW = datetime.now(timezone.utc)


def _make_memory_result(**overrides: object) -> MemoryResult:
    defaults: dict = {
        "id": "test-id",
        "content": "test content",
        "similarity": 0.9,
        "namespace": "default",
        "project": "github.com/org/repo",
        "importance": 0.5,
        "created_at": NOW,
        "tags": [],
        "metadata": {},
    }
    defaults.update(overrides)
    return MemoryResult(**defaults)


# ===========================================================================
# _resolve_project tests
# ===========================================================================


@pytest.mark.unit
class TestResolveProject:
    """Tests for SpatialMemoryServer._resolve_project."""

    def test_star_returns_none(self) -> None:
        """project='*' should return None (cross-project)."""
        server, _ = _make_server_with_mocks()
        args = {"project": "*", "query": "test"}
        result = server._resolve_project(args)
        assert result is None
        # project key is NOT consumed (uses .get, not .pop)
        assert "project" in args

    def test_explicit_project_passes_through(self) -> None:
        """Explicit project string should be returned."""
        server, _ = _make_server_with_mocks()
        args = {"project": "github.com/org/repo", "query": "test"}
        result = server._resolve_project(args)
        assert result == "github.com/org/repo"
        assert "project" in args

    def test_omitted_auto_detects(self) -> None:
        """Omitted project should trigger auto-detection cascade."""
        detector = MagicMock(spec=ProjectDetector)
        detector.detect.return_value = ProjectIdentity(
            project_id="github.com/detected/repo",
            source="env_var",
        )
        server, _ = _make_server_with_mocks(project_detector=detector)
        args = {"query": "test"}
        result = server._resolve_project(args)
        assert result == "github.com/detected/repo"
        detector.detect.assert_called_once_with(explicit_project=None)

    def test_none_explicit_auto_detects(self) -> None:
        """Explicit None should also trigger auto-detection."""
        detector = MagicMock(spec=ProjectDetector)
        detector.detect.return_value = ProjectIdentity(
            project_id="",
            source="fallback",
        )
        server, _ = _make_server_with_mocks(project_detector=detector)
        args = {"project": None}
        result = server._resolve_project(args)
        # Empty project_id => returns None
        assert result is None

    def test_preserves_project_key(self) -> None:
        """_resolve_project should not mutate the arguments dict."""
        server, _ = _make_server_with_mocks()
        args = {"project": "test-proj", "other": "value"}
        server._resolve_project(args)
        assert "project" in args
        assert args == {"project": "test-proj", "other": "value"}

    def test_rejects_invalid_project(self) -> None:
        """_resolve_project should reject invalid project strings."""
        from spatial_memory.core.errors import ValidationError

        server, _ = _make_server_with_mocks()
        args = {"project": "'; DROP TABLE memories--", "query": "test"}
        with pytest.raises(ValidationError, match="Invalid project format"):
            server._resolve_project(args)

    def test_star_bypasses_validation(self) -> None:
        """_resolve_project with '*' should return None without validation."""
        server, _ = _make_server_with_mocks()
        args = {"project": "*"}
        result = server._resolve_project(args)
        assert result is None

    def test_none_project_skips_validation(self) -> None:
        """_resolve_project with None should skip validation and auto-detect."""
        server, _ = _make_server_with_mocks()
        args = {"project": None}
        # Should not raise - None triggers auto-detection, not validation
        server._resolve_project(args)


# ===========================================================================
# Handler project pass-through tests
# ===========================================================================


@pytest.mark.unit
class TestHandlerProjectPassThrough:
    """Verify handlers extract project and pass it to services."""

    def test_remember_passes_project(self) -> None:
        """_handle_remember should pass project to memory service."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            id: str = "new-id"
            content: str = "test"
            namespace: str = "default"
            deduplicated: bool = False
            status: str = "stored"
            quality_score: float | None = None
            existing_memory_id: str | None = None
            existing_memory_content: str | None = None
            similarity: float | None = None

        server, mocks = _make_server_with_mocks()
        mocks["memory"].remember.return_value = FakeResult()
        args = {"content": "test", "project": "github.com/org/repo"}
        server._handle_remember(args)
        mocks["memory"].remember.assert_called_once()
        call_kwargs = mocks["memory"].remember.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_recall_passes_project(self) -> None:
        """_handle_recall should pass project to memory service."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeRecallResult:
            memories: list = field(default_factory=list)
            total: int = 0

        server, mocks = _make_server_with_mocks()
        mocks["memory"].recall.return_value = FakeRecallResult()
        args = {"query": "test", "project": "github.com/org/repo"}
        server._handle_recall(args)
        mocks["memory"].recall.assert_called_once()
        call_kwargs = mocks["memory"].recall.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_recall_star_project_passes_none(self) -> None:
        """_handle_recall with project='*' should pass project=None."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeRecallResult:
            memories: list = field(default_factory=list)
            total: int = 0

        server, mocks = _make_server_with_mocks()
        mocks["memory"].recall.return_value = FakeRecallResult()
        args = {"query": "test", "project": "*"}
        server._handle_recall(args)
        call_kwargs = mocks["memory"].recall.call_args
        assert call_kwargs.kwargs.get("project") is None

    def test_hybrid_recall_passes_project(self) -> None:
        """_handle_hybrid_recall should pass project to utility service."""
        server, mocks = _make_server_with_mocks()
        mocks["utility"].hybrid_recall.return_value = HybridRecallResult(
            query="test",
            alpha=0.5,
            memories=[],
            total=0,
            search_type="hybrid",
        )
        args = {"query": "test", "project": "github.com/org/repo"}
        server._handle_hybrid_recall(args)
        mocks["utility"].hybrid_recall.assert_called_once()
        call_kwargs = mocks["utility"].hybrid_recall.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_regions_passes_project(self) -> None:
        """_handle_regions should pass project to spatial service."""
        from spatial_memory.core.models import RegionsResult

        server, mocks = _make_server_with_mocks()
        mocks["spatial"].regions.return_value = RegionsResult(
            clusters=[],
            total_memories=0,
            noise_count=0,
            clustering_quality=0.0,
        )
        args = {"project": "github.com/org/repo"}
        server._handle_regions(args)
        mocks["spatial"].regions.assert_called_once()
        call_kwargs = mocks["spatial"].regions.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_stats_passes_project(self) -> None:
        """_handle_stats should pass project to utility service."""
        server, mocks = _make_server_with_mocks()
        mocks["utility"].stats.return_value = StatsResult(
            total_memories=0,
            memories_by_namespace={},
            storage_bytes=0,
            storage_mb=0.0,
            estimated_vector_bytes=0,
            has_vector_index=False,
            has_fts_index=False,
            indices=[],
            num_fragments=0,
            needs_compaction=False,
            table_version=1,
        )
        args = {"project": "github.com/org/repo"}
        server._handle_stats(args)
        mocks["utility"].stats.assert_called_once()
        call_kwargs = mocks["utility"].stats.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_extract_passes_project(self) -> None:
        """_handle_extract should pass project to lifecycle service."""
        from spatial_memory.core.models import ExtractResult

        server, mocks = _make_server_with_mocks()
        mocks["lifecycle"].extract.return_value = ExtractResult(
            candidates_found=0,
            memories_created=0,
            deduplicated_count=0,
            extractions=[],
        )
        args = {"text": "some text", "project": "github.com/org/repo"}
        server._handle_extract(args)
        mocks["lifecycle"].extract.assert_called_once()
        call_kwargs = mocks["lifecycle"].extract.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"

    def test_consolidate_passes_project(self) -> None:
        """_handle_consolidate should pass project to lifecycle service."""
        from spatial_memory.core.models import ConsolidateResult

        server, mocks = _make_server_with_mocks()
        mocks["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=0,
            memories_merged=0,
            memories_deleted=0,
            groups=[],
            dry_run=True,
        )
        args = {"namespace": "default", "project": "github.com/org/repo"}
        server._handle_consolidate(args)
        mocks["lifecycle"].consolidate.assert_called_once()
        call_kwargs = mocks["lifecycle"].consolidate.call_args
        assert call_kwargs.kwargs.get("project") == "github.com/org/repo"


# ===========================================================================
# Response dict project field tests
# ===========================================================================


@pytest.mark.unit
class TestResponseProjectField:
    """Verify that response dicts include project when set."""

    def test_recall_response_includes_project(self) -> None:
        """Recall response should include project in memory dicts."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeRecallResult:
            memories: list = field(default_factory=list)
            total: int = 0

        mem = _make_memory_result(project="github.com/org/repo")
        server, mocks = _make_server_with_mocks()
        mocks["memory"].recall.return_value = FakeRecallResult(memories=[mem], total=1)
        args = {"query": "test", "project": "*"}
        result = server._handle_recall(args)
        assert result["memories"][0]["project"] == "github.com/org/repo"

    def test_recall_response_omits_empty_project(self) -> None:
        """Recall response should omit project when empty."""
        from dataclasses import dataclass, field

        @dataclass
        class FakeRecallResult:
            memories: list = field(default_factory=list)
            total: int = 0

        mem = _make_memory_result(project="")
        server, mocks = _make_server_with_mocks()
        mocks["memory"].recall.return_value = FakeRecallResult(memories=[mem], total=1)
        args = {"query": "test", "project": "*"}
        result = server._handle_recall(args)
        assert "project" not in result["memories"][0]

    def test_hybrid_recall_response_includes_project(self) -> None:
        """Hybrid recall response should include project in memory dicts."""
        mem = HybridMemoryMatch(
            id="test-id",
            content="test",
            similarity=0.9,
            namespace="default",
            project="github.com/org/repo",
            tags=[],
            importance=0.5,
            created_at=NOW,
            metadata={},
        )
        server, mocks = _make_server_with_mocks()
        mocks["utility"].hybrid_recall.return_value = HybridRecallResult(
            query="test",
            alpha=0.5,
            memories=[mem],
            total=1,
            search_type="hybrid",
        )
        args = {"query": "test", "project": "*"}
        result = server._handle_hybrid_recall(args)
        assert result["memories"][0]["project"] == "github.com/org/repo"
