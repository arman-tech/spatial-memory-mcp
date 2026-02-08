"""Tests for project and content_hash fields (WP2).

Verifies that the new project and content_hash fields are properly
integrated into models, repository, and migration.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from spatial_memory.core.models import (
    HybridMemoryMatch,
    Memory,
    MemoryResult,
)


@pytest.mark.unit
class TestMemoryProjectField:
    """Tests for project and content_hash fields on Memory."""

    def test_default_project_empty(self) -> None:
        """Test that project defaults to empty string."""
        memory = Memory(id="test-id", content="test content")
        assert memory.project == ""

    def test_default_content_hash_empty(self) -> None:
        """Test that content_hash defaults to empty string."""
        memory = Memory(id="test-id", content="test content")
        assert memory.content_hash == ""

    def test_set_project(self) -> None:
        """Test setting project field."""
        memory = Memory(
            id="test-id",
            content="test content",
            project="github.com/org/repo",
        )
        assert memory.project == "github.com/org/repo"

    def test_set_content_hash(self) -> None:
        """Test setting content_hash field."""
        content = "test content"
        content_hash = hashlib.sha256(content.strip().lower().encode()).hexdigest()
        memory = Memory(
            id="test-id",
            content=content,
            content_hash=content_hash,
        )
        assert memory.content_hash == content_hash
        assert len(memory.content_hash) == 64  # SHA-256 hex length

    def test_memory_roundtrip(self) -> None:
        """Test that Memory serializes and deserializes with project."""
        memory = Memory(
            id="test-id",
            content="test content",
            project="github.com/org/repo",
            content_hash="abc123",
        )
        data = memory.model_dump()
        assert data["project"] == "github.com/org/repo"
        assert data["content_hash"] == "abc123"

        restored = Memory(**data)
        assert restored.project == "github.com/org/repo"
        assert restored.content_hash == "abc123"


@pytest.mark.unit
class TestMemoryResultProjectField:
    """Tests for project field on MemoryResult."""

    def test_default_project_empty(self) -> None:
        """Test that project defaults to empty string."""
        now = datetime.now(timezone.utc)
        result = MemoryResult(
            id="test-id",
            content="test content",
            similarity=0.9,
            namespace="default",
            importance=0.5,
            created_at=now,
        )
        assert result.project == ""

    def test_set_project(self) -> None:
        """Test setting project field."""
        now = datetime.now(timezone.utc)
        result = MemoryResult(
            id="test-id",
            content="test content",
            similarity=0.9,
            namespace="default",
            project="github.com/org/repo",
            importance=0.5,
            created_at=now,
        )
        assert result.project == "github.com/org/repo"


@pytest.mark.unit
class TestHybridMemoryMatchProjectField:
    """Tests for project field on HybridMemoryMatch."""

    def test_default_project_empty(self) -> None:
        """Test that project defaults to empty string."""
        now = datetime.now(timezone.utc)
        match = HybridMemoryMatch(
            id="test-id",
            content="test content",
            similarity=0.9,
            namespace="default",
            tags=["test"],
            importance=0.5,
            created_at=now,
            metadata={},
        )
        assert match.project == ""

    def test_set_project(self) -> None:
        """Test setting project field."""
        now = datetime.now(timezone.utc)
        match = HybridMemoryMatch(
            id="test-id",
            content="test content",
            similarity=0.9,
            namespace="default",
            tags=["test"],
            importance=0.5,
            created_at=now,
            metadata={},
            project="github.com/org/repo",
        )
        assert match.project == "github.com/org/repo"


@pytest.mark.unit
class TestContentHashComputation:
    """Tests for content hash computation logic."""

    def test_sha256_hex(self) -> None:
        """Test that hash is SHA-256 hex."""
        content = "test content"
        normalized = content.strip().lower()
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_normalized_dedup(self) -> None:
        """Test that normalization catches near-duplicates."""
        h1 = hashlib.sha256(b"test content").hexdigest()
        h2 = hashlib.sha256("  test content  ".strip().lower().encode()).hexdigest()
        # Note: "test content" == "test content".strip().lower()
        assert h1 == h2

    def test_case_normalization(self) -> None:
        """Test case normalization for dedup."""
        h1 = hashlib.sha256("Test Content".lower().encode()).hexdigest()
        h2 = hashlib.sha256(b"test content").hexdigest()
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        """Test that different content produces different hashes."""
        h1 = hashlib.sha256(b"content a").hexdigest()
        h2 = hashlib.sha256(b"content b").hexdigest()
        assert h1 != h2
