"""Tests for data models."""

from datetime import datetime, timezone

import pytest

from spatial_memory.core.models import (
    Filter,
    FilterGroup,
    FilterOperator,
    Memory,
    MemoryResult,
    MemorySource,
)


class TestMemory:
    """Tests for Memory model."""

    def test_create_memory(self) -> None:
        """Test basic memory creation."""
        memory = Memory(id="test-123", content="Test content")
        assert memory.id == "test-123"
        assert memory.content == "Test content"
        assert memory.importance == 0.5  # default
        assert memory.namespace == "default"
        assert memory.source == MemorySource.MANUAL

    def test_memory_with_metadata(self) -> None:
        """Test memory with all fields."""
        memory = Memory(
            id="test-456",
            content="Full memory",
            importance=0.9,
            namespace="project-alpha",
            tags=["tag1", "tag2"],
            source=MemorySource.EXTRACTED,
            metadata={"key": "value"},
        )
        assert memory.importance == 0.9
        assert memory.namespace == "project-alpha"
        assert memory.tags == ["tag1", "tag2"]
        assert memory.source == MemorySource.EXTRACTED
        assert memory.metadata == {"key": "value"}

    def test_memory_timestamps_are_utc(self) -> None:
        """Test that timestamps are timezone-aware UTC."""
        memory = Memory(id="test-789", content="Test")
        assert memory.created_at.tzinfo is not None
        assert memory.updated_at.tzinfo is not None
        assert memory.last_accessed.tzinfo is not None

    def test_memory_importance_bounds(self) -> None:
        """Test importance validation."""
        # Valid
        Memory(id="test", content="test", importance=0.0)
        Memory(id="test", content="test", importance=1.0)

        # Invalid
        with pytest.raises(ValueError):
            Memory(id="test", content="test", importance=-0.1)

        with pytest.raises(ValueError):
            Memory(id="test", content="test", importance=1.1)

    def test_memory_content_max_length(self) -> None:
        """Test content max length validation."""
        # Should work with large content up to limit
        long_content = "x" * 100000
        memory = Memory(id="test", content=long_content)
        assert len(memory.content) == 100000

        # Should fail with content over limit
        with pytest.raises(ValueError):
            Memory(id="test", content="x" * 100001)


class TestMemoryResult:
    """Tests for MemoryResult model."""

    def test_create_memory_result(self) -> None:
        """Test memory result creation."""
        result = MemoryResult(
            id="test-123",
            content="Test content",
            similarity=0.95,
            namespace="default",
            importance=0.7,
            created_at=datetime.now(timezone.utc),
        )
        assert result.id == "test-123"
        assert result.similarity == 0.95
        assert result.importance == 0.7

    def test_similarity_bounds(self) -> None:
        """Test similarity value bounds."""
        # Valid
        MemoryResult(
            id="test",
            content="test",
            similarity=0.0,
            namespace="default",
            importance=0.5,
            created_at=datetime.now(timezone.utc),
        )
        MemoryResult(
            id="test",
            content="test",
            similarity=1.0,
            namespace="default",
            importance=0.5,
            created_at=datetime.now(timezone.utc),
        )

        # Invalid
        with pytest.raises(ValueError):
            MemoryResult(
                id="test",
                content="test",
                similarity=-0.1,
                namespace="default",
                importance=0.5,
                created_at=datetime.now(timezone.utc),
            )


class TestFilter:
    """Tests for Filter model."""

    def test_create_filter(self) -> None:
        """Test filter creation."""
        f = Filter(field="importance", operator=FilterOperator.GT, value=0.5)
        assert f.field == "importance"
        assert f.operator == FilterOperator.GT
        assert f.value == 0.5

    def test_filter_operators(self) -> None:
        """Test all filter operators."""
        operators = [
            FilterOperator.EQ,
            FilterOperator.NE,
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
            FilterOperator.IN,
            FilterOperator.NIN,
            FilterOperator.CONTAINS,
        ]
        for op in operators:
            f = Filter(field="test", operator=op, value="value")
            assert f.operator == op

    def test_filter_with_typed_values(self) -> None:
        """Test filter with various typed values."""
        # String value
        f1 = Filter(field="namespace", operator=FilterOperator.EQ, value="default")
        assert f1.value == "default"

        # Int value
        f2 = Filter(field="access_count", operator=FilterOperator.GT, value=5)
        assert f2.value == 5

        # Float value
        f3 = Filter(field="importance", operator=FilterOperator.GTE, value=0.7)
        assert f3.value == 0.7

        # Bool value
        f4 = Filter(field="archived", operator=FilterOperator.EQ, value=True)
        assert f4.value is True

        # List of strings (for IN operator)
        f5 = Filter(field="tags", operator=FilterOperator.IN, value=["tag1", "tag2"])
        assert f5.value == ["tag1", "tag2"]

        # List of ints
        f6 = Filter(field="cluster_id", operator=FilterOperator.IN, value=[1, 2, 3])
        assert f6.value == [1, 2, 3]

        # Datetime value
        now = datetime.now(timezone.utc)
        f7 = Filter(field="created_at", operator=FilterOperator.GT, value=now)
        assert f7.value == now


class TestFilterGroup:
    """Tests for FilterGroup model."""

    def test_create_filter_group(self) -> None:
        """Test basic filter group creation."""
        f1 = Filter(field="importance", operator=FilterOperator.GT, value=0.5)
        f2 = Filter(field="namespace", operator=FilterOperator.EQ, value="default")
        group = FilterGroup(operator="and", filters=[f1, f2])

        assert group.operator == "and"
        assert len(group.filters) == 2

    def test_filter_group_or_operator(self) -> None:
        """Test filter group with OR operator."""
        f1 = Filter(field="importance", operator=FilterOperator.GT, value=0.8)
        f2 = Filter(field="tags", operator=FilterOperator.CONTAINS, value="important")
        group = FilterGroup(operator="or", filters=[f1, f2])

        assert group.operator == "or"
        assert len(group.filters) == 2

    def test_filter_group_empty_filters_raises(self) -> None:
        """Test that empty filters list raises validation error."""
        with pytest.raises(ValueError):
            FilterGroup(operator="and", filters=[])

    def test_filter_group_nested(self) -> None:
        """Test nested filter groups."""
        f1 = Filter(field="importance", operator=FilterOperator.GT, value=0.5)
        f2 = Filter(field="access_count", operator=FilterOperator.GT, value=10)
        inner_group = FilterGroup(operator="and", filters=[f1, f2])

        f3 = Filter(field="namespace", operator=FilterOperator.EQ, value="priority")
        outer_group = FilterGroup(operator="or", filters=[inner_group, f3])

        assert outer_group.operator == "or"
        assert len(outer_group.filters) == 2
        assert isinstance(outer_group.filters[0], FilterGroup)
        assert isinstance(outer_group.filters[1], Filter)

    def test_filter_group_single_filter(self) -> None:
        """Test filter group with single filter (minimum valid case)."""
        f = Filter(field="importance", operator=FilterOperator.GT, value=0.5)
        group = FilterGroup(filters=[f])

        assert group.operator == "and"  # Default
        assert len(group.filters) == 1

    def test_filter_group_default_operator(self) -> None:
        """Test that default operator is 'and'."""
        f = Filter(field="importance", operator=FilterOperator.GT, value=0.5)
        group = FilterGroup(filters=[f])

        assert group.operator == "and"
