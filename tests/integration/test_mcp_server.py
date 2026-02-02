"""Integration tests for the MCP server.

These tests verify the full stack works together with real (but isolated)
database and embedding service instances.
"""

from __future__ import annotations

import pytest

from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.server import SpatialMemoryServer

# ---------------------------------------------------------------------------
# Alias Fixtures (use shared fixtures from conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def server(integration_server: SpatialMemoryServer) -> SpatialMemoryServer:
    """Use shared server fixture from conftest."""
    return integration_server


@pytest.fixture
def embeddings(embedding_service: EmbeddingService) -> EmbeddingService:
    """Use shared embedding service from conftest."""
    return embedding_service


class TestRememberTool:
    """Integration tests for the remember tool."""

    def test_remember_stores_and_returns_id(self, server: SpatialMemoryServer) -> None:
        """remember tool should store a memory and return its ID."""
        result = server._handle_tool("remember", {
            "content": "Python is a programming language",
        })

        assert "id" in result
        assert result["content"] == "Python is a programming language"
        assert result["namespace"] == "default"

    def test_remember_with_all_options(self, server: SpatialMemoryServer) -> None:
        """remember tool should accept all optional parameters."""
        result = server._handle_tool("remember", {
            "content": "Important technical decision",
            "namespace": "architecture",
            "tags": ["decision", "important"],
            "importance": 0.9,
            "metadata": {"author": "test"},
        })

        assert "id" in result
        assert result["namespace"] == "architecture"

    def test_remember_validates_content(self, server: SpatialMemoryServer) -> None:
        """remember tool should validate empty content."""
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError, match="[Cc]ontent.*empty"):
            server._handle_tool("remember", {
                "content": "",
            })


class TestRememberBatchTool:
    """Integration tests for the remember_batch tool."""

    def test_remember_batch_stores_multiple(self, server: SpatialMemoryServer) -> None:
        """remember_batch tool should store multiple memories."""
        result = server._handle_tool("remember_batch", {
            "memories": [
                {"content": "Memory one"},
                {"content": "Memory two"},
                {"content": "Memory three"},
            ],
        })

        assert result["count"] == 3
        assert len(result["ids"]) == 3


class TestRecallTool:
    """Integration tests for the recall tool."""

    def test_recall_finds_similar_memories(self, server: SpatialMemoryServer) -> None:
        """recall tool should find semantically similar memories."""
        # First, store some memories
        server._handle_tool("remember", {
            "content": "Python is great for data science",
        })
        server._handle_tool("remember", {
            "content": "JavaScript runs in the browser",
        })
        server._handle_tool("remember", {
            "content": "Python has excellent ML libraries like scikit-learn",
        })

        # Search for data science related memories
        result = server._handle_tool("recall", {
            "query": "machine learning with Python",
            "limit": 5,
        })

        assert "memories" in result
        assert result["total"] >= 1
        # Python ML memory should have high similarity
        memories = result["memories"]
        assert len(memories) > 0
        # The first result should be about Python/ML
        assert "python" in memories[0]["content"].lower() or "ml" in memories[0]["content"].lower()

    def test_recall_with_namespace_filter(self, server: SpatialMemoryServer) -> None:
        """recall tool should filter by namespace."""
        # Store memories in different namespaces
        server._handle_tool("remember", {
            "content": "Work project deadline",
            "namespace": "work",
        })
        server._handle_tool("remember", {
            "content": "Personal vacation plans",
            "namespace": "personal",
        })

        # Search only in work namespace
        result = server._handle_tool("recall", {
            "query": "deadline",
            "namespace": "work",
        })

        assert result["total"] >= 1
        # All results should be from work namespace
        for mem in result["memories"]:
            assert mem["namespace"] == "work"

    def test_recall_with_min_similarity(self, server: SpatialMemoryServer) -> None:
        """recall tool should filter by minimum similarity."""
        server._handle_tool("remember", {
            "content": "React is a JavaScript framework",
        })

        result = server._handle_tool("recall", {
            "query": "TypeScript type system",
            "min_similarity": 0.9,  # Very high threshold
        })

        # Should filter out low similarity results
        for mem in result.get("memories", []):
            assert mem["similarity"] >= 0.9


class TestNearbyTool:
    """Integration tests for the nearby tool."""

    def test_nearby_finds_similar_memories(self, server: SpatialMemoryServer) -> None:
        """nearby tool should find memories similar to a reference."""
        # Store related memories
        result1 = server._handle_tool("remember", {
            "content": "Docker containers provide isolation",
        })
        server._handle_tool("remember", {
            "content": "Kubernetes orchestrates containers",
        })
        server._handle_tool("remember", {
            "content": "Cooking pasta requires boiling water",
        })

        # Find memories near the Docker one
        result = server._handle_tool("nearby", {
            "memory_id": result1["id"],
            "limit": 5,
        })

        assert "reference" in result
        assert result["reference"]["id"] == result1["id"]
        assert "neighbors" in result
        # Kubernetes should be found as a neighbor
        neighbor_contents = [n["content"] for n in result["neighbors"]]
        assert any("kubernetes" in c.lower() or "container" in c.lower()
                   for c in neighbor_contents)

    def test_nearby_with_nonexistent_memory(self, server: SpatialMemoryServer) -> None:
        """nearby tool should handle nonexistent memory ID."""
        from spatial_memory.core.errors import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            server._handle_tool("nearby", {
                "memory_id": "00000000-0000-0000-0000-000000000000",
            })


class TestForgetTool:
    """Integration tests for the forget tool."""

    def test_forget_deletes_memory(self, server: SpatialMemoryServer) -> None:
        """forget tool should delete a memory."""
        from spatial_memory.core.errors import MemoryNotFoundError

        # Store a memory
        remember_result = server._handle_tool("remember", {
            "content": "Temporary memory to delete",
        })
        memory_id = remember_result["id"]

        # Delete it
        forget_result = server._handle_tool("forget", {
            "memory_id": memory_id,
        })

        assert forget_result["deleted"] == 1

        # Verify it's gone by trying to find it
        with pytest.raises(MemoryNotFoundError):
            server._handle_tool("nearby", {
                "memory_id": memory_id,
            })

    def test_forget_nonexistent_returns_zero(self, server: SpatialMemoryServer) -> None:
        """forget tool should return deleted=0 for nonexistent ID."""
        result = server._handle_tool("forget", {
            "memory_id": "00000000-0000-0000-0000-000000000000",
        })

        assert result["deleted"] == 0


class TestForgetBatchTool:
    """Integration tests for the forget_batch tool."""

    def test_forget_batch_deletes_multiple(self, server: SpatialMemoryServer) -> None:
        """forget_batch tool should delete multiple memories."""
        # Store multiple memories
        ids = []
        for i in range(3):
            result = server._handle_tool("remember", {
                "content": f"Memory {i} to delete",
            })
            ids.append(result["id"])

        # Delete them all
        result = server._handle_tool("forget_batch", {
            "memory_ids": ids,
        })

        assert result["deleted"] == 3


class TestToolErrorHandling:
    """Integration tests for error handling in tools."""

    def test_unknown_tool_returns_error(self, server: SpatialMemoryServer) -> None:
        """Unknown tool name should raise ValidationError."""
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError, match="Unknown tool"):
            server._handle_tool("nonexistent_tool", {})

    def test_invalid_importance_returns_error(self, server: SpatialMemoryServer) -> None:
        """Invalid importance value should raise validation error."""
        from spatial_memory.core.errors import ValidationError

        with pytest.raises(ValidationError, match="[Ii]mportance"):
            server._handle_tool("remember", {
                "content": "Test content",
                "importance": 2.0,  # Out of range
            })


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_memory_lifecycle(self, server: SpatialMemoryServer) -> None:
        """Test complete workflow: remember, recall, nearby, forget."""
        # 1. Remember some related content
        mem1 = server._handle_tool("remember", {
            "content": "Machine learning uses statistical models",
            "namespace": "tech",
            "tags": ["ml", "stats"],
        })
        mem2 = server._handle_tool("remember", {
            "content": "Deep learning is a subset of machine learning",
            "namespace": "tech",
            "tags": ["ml", "dl"],
        })
        mem3 = server._handle_tool("remember", {
            "content": "Cooking requires following recipes",
            "namespace": "hobbies",
        })

        # 2. Recall ML-related memories
        recall_result = server._handle_tool("recall", {
            "query": "neural networks and AI",
            "namespace": "tech",
            "limit": 10,
        })

        assert recall_result["total"] >= 2
        # Tech memories should rank high
        tech_ids = {mem1["id"], mem2["id"]}
        found_ids = {m["id"] for m in recall_result["memories"]}
        assert tech_ids & found_ids  # At least one tech memory found

        # 3. Find memories nearby the first one
        nearby_result = server._handle_tool("nearby", {
            "memory_id": mem1["id"],
            "limit": 5,
        })

        assert nearby_result["reference"]["id"] == mem1["id"]
        # Deep learning memory should be a neighbor
        neighbor_ids = {n["id"] for n in nearby_result["neighbors"]}
        assert mem2["id"] in neighbor_ids

        # 4. Forget one memory
        forget_result = server._handle_tool("forget", {
            "memory_id": mem3["id"],
        })
        assert forget_result["deleted"] == 1

        # 5. Verify forgotten memory is gone
        from spatial_memory.core.errors import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            server._handle_tool("nearby", {
                "memory_id": mem3["id"],
            })
