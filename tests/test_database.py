"""Tests for database operations."""

import tempfile
from pathlib import Path

import pytest

from spatial_memory.core.database import (
    Database,
    _sanitize_string,
    _validate_namespace,
    _validate_uuid,
)
from spatial_memory.core.errors import (
    MemoryNotFoundError,
    ValidationError,
)


class TestSanitization:
    """Tests for input sanitization functions."""

    def test_sanitize_string_escapes_quotes(self) -> None:
        """Test that single quotes are escaped."""
        assert _sanitize_string("test") == "test"
        assert _sanitize_string("test's") == "test''s"
        assert _sanitize_string("it's a 'test'") == "it''s a ''test''"

    def test_sanitize_string_blocks_injection(self) -> None:
        """Test that SQL injection patterns are blocked."""
        with pytest.raises(ValidationError):
            _sanitize_string("'; DROP TABLE memories; --")

        with pytest.raises(ValidationError):
            _sanitize_string("' OR '1'='1")

        with pytest.raises(ValidationError):
            _sanitize_string("test /* comment */")

    def test_validate_uuid_valid(self) -> None:
        """Test valid UUID validation."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert _validate_uuid(valid_uuid) == valid_uuid

    def test_validate_uuid_invalid(self) -> None:
        """Test invalid UUID rejection."""
        with pytest.raises(ValidationError):
            _validate_uuid("not-a-uuid")

        with pytest.raises(ValidationError):
            _validate_uuid("550e8400-e29b-41d4-a716")  # Too short

        with pytest.raises(ValidationError):
            _validate_uuid("")

    def test_validate_namespace_valid(self) -> None:
        """Test valid namespace validation."""
        assert _validate_namespace("default") == "default"
        assert _validate_namespace("project-alpha") == "project-alpha"
        assert _validate_namespace("my_namespace") == "my_namespace"
        assert _validate_namespace("ns.v1") == "ns.v1"

    def test_validate_namespace_invalid(self) -> None:
        """Test invalid namespace rejection."""
        with pytest.raises(ValidationError):
            _validate_namespace("")

        with pytest.raises(ValidationError):
            _validate_namespace("invalid namespace")  # Space

        with pytest.raises(ValidationError):
            _validate_namespace("invalid/namespace")  # Slash

        with pytest.raises(ValidationError):
            _validate_namespace("x" * 300)  # Too long


class TestDatabaseContextManager:
    """Tests for database context manager."""

    def test_context_manager(self) -> None:
        """Test database context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test-db"

            with Database(db_path) as db:
                assert db._db is not None
                assert db._table is not None

            # After exit, should be closed
            assert db._db is None
            assert db._table is None


class TestDatabaseOperations:
    """Tests for database CRUD operations."""

    def test_insert_and_get(self, database: Database, embedding_service) -> None:
        """Test inserting and retrieving a memory."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            namespace="test",
            tags=["tag1", "tag2"],
            importance=0.8,
        )

        # Retrieve
        record = database.get(memory_id)
        assert record["content"] == "Test content"
        assert record["namespace"] == "test"
        assert record["tags"] == ["tag1", "tag2"]
        assert record["importance"] == pytest.approx(0.8, abs=0.01)

    def test_get_nonexistent_raises(self, database: Database) -> None:
        """Test that getting nonexistent memory raises error."""
        with pytest.raises(MemoryNotFoundError):
            database.get("550e8400-e29b-41d4-a716-446655440000")

    def test_get_invalid_uuid_raises(self, database: Database) -> None:
        """Test that invalid UUID raises validation error."""
        with pytest.raises(ValidationError):
            database.get("not-a-uuid")

    def test_insert_validates_content(self, database: Database, embedding_service) -> None:
        """Test that insert validates content."""
        vec = embedding_service.embed("Test")

        with pytest.raises(ValidationError):
            database.insert(content="", vector=vec)

    def test_insert_validates_namespace(self, database: Database, embedding_service) -> None:
        """Test that insert validates namespace."""
        vec = embedding_service.embed("Test")

        with pytest.raises(ValidationError):
            database.insert(content="Test", vector=vec, namespace="invalid namespace")

    def test_update(self, database: Database, embedding_service) -> None:
        """Test updating a memory."""
        vec = embedding_service.embed("Original content")
        memory_id = database.insert(content="Original content", vector=vec)

        database.update(memory_id, {"importance": 0.9})

        record = database.get(memory_id)
        assert record["importance"] == pytest.approx(0.9, abs=0.01)

    def test_delete(self, database: Database, embedding_service) -> None:
        """Test deleting a memory."""
        vec = embedding_service.embed("To be deleted")
        memory_id = database.insert(content="To be deleted", vector=vec)

        database.delete(memory_id)

        with pytest.raises(MemoryNotFoundError):
            database.get(memory_id)

    def test_vector_search(self, database: Database, embedding_service) -> None:
        """Test vector similarity search."""
        # Insert some memories
        contents = [
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Databases store data",
        ]
        for content in contents:
            vec = embedding_service.embed(content)
            database.insert(content=content, vector=vec)

        # Search for programming-related
        query_vec = embedding_service.embed("coding languages")
        results = database.vector_search(query_vec, limit=2)

        assert len(results) == 2
        assert "similarity" in results[0]
        assert results[0]["similarity"] >= results[1]["similarity"]

    def test_count(self, database: Database, embedding_service) -> None:
        """Test counting memories."""
        assert database.count() == 0

        vec = embedding_service.embed("Test")
        database.insert(content="Test 1", vector=vec, namespace="ns1")
        database.insert(content="Test 2", vector=vec, namespace="ns1")
        database.insert(content="Test 3", vector=vec, namespace="ns2")

        assert database.count() == 3
        assert database.count(namespace="ns1") == 2
        assert database.count(namespace="ns2") == 1

    def test_get_namespaces(self, database: Database, embedding_service) -> None:
        """Test getting unique namespaces."""
        vec = embedding_service.embed("Test")
        database.insert(content="Test 1", vector=vec, namespace="alpha")
        database.insert(content="Test 2", vector=vec, namespace="beta")
        database.insert(content="Test 3", vector=vec, namespace="alpha")

        namespaces = database.get_namespaces()
        assert sorted(namespaces) == ["alpha", "beta"]


class TestInsertBatch:
    """Tests for batch insertion operations."""

    def test_insert_batch_success(self, database: Database, embedding_service) -> None:
        """Test successful batch insertion of multiple memories."""
        records = [
            {
                "content": "First memory content",
                "vector": embedding_service.embed("First memory content"),
                "namespace": "batch-test",
                "tags": ["tag1"],
                "importance": 0.7,
            },
            {
                "content": "Second memory content",
                "vector": embedding_service.embed("Second memory content"),
                "namespace": "batch-test",
                "tags": ["tag2"],
                "importance": 0.8,
            },
            {
                "content": "Third memory content",
                "vector": embedding_service.embed("Third memory content"),
                "namespace": "batch-test",
                "tags": ["tag3"],
                "importance": 0.9,
            },
        ]

        memory_ids = database.insert_batch(records)

        assert len(memory_ids) == 3
        assert all(isinstance(mid, str) for mid in memory_ids)
        assert len(set(memory_ids)) == 3  # All IDs unique

        # Verify all records were inserted correctly
        for i, memory_id in enumerate(memory_ids):
            record = database.get(memory_id)
            assert record["content"] == records[i]["content"]
            assert record["namespace"] == "batch-test"
            assert record["importance"] == pytest.approx(records[i]["importance"], abs=0.01)

    def test_insert_batch_empty_list(self, database: Database) -> None:
        """Test batch insertion with empty list returns empty list."""
        memory_ids = database.insert_batch([])
        assert memory_ids == []
        assert database.count() == 0

    def test_insert_batch_validates_content(self, database: Database, embedding_service) -> None:
        """Test that batch insertion validates content for each record."""
        records = [
            {
                "content": "Valid content",
                "vector": embedding_service.embed("Valid content"),
            },
            {
                "content": "",  # Invalid: empty content
                "vector": embedding_service.embed("Test"),
            },
        ]

        with pytest.raises(ValidationError):
            database.insert_batch(records)

    def test_insert_batch_validates_namespace(self, database: Database, embedding_service) -> None:
        """Test that batch insertion validates namespace for each record."""
        records = [
            {
                "content": "Valid content",
                "vector": embedding_service.embed("Valid content"),
                "namespace": "invalid namespace",  # Invalid: contains space
            },
        ]

        with pytest.raises(ValidationError):
            database.insert_batch(records)

    def test_insert_batch_validates_importance(self, database: Database, embedding_service) -> None:
        """Test that batch insertion validates importance for each record."""
        # Test importance too high
        records_high = [
            {
                "content": "Valid content",
                "vector": embedding_service.embed("Valid content"),
                "importance": 1.5,  # Invalid: > 1.0
            },
        ]

        with pytest.raises(ValidationError, match="Importance must be between 0.0 and 1.0"):
            database.insert_batch(records_high)

        # Test importance too low
        records_low = [
            {
                "content": "Valid content",
                "vector": embedding_service.embed("Valid content"),
                "importance": -0.1,  # Invalid: < 0.0
            },
        ]

        with pytest.raises(ValidationError, match="Importance must be between 0.0 and 1.0"):
            database.insert_batch(records_low)

    def test_insert_batch_mixed_namespaces(self, database: Database, embedding_service) -> None:
        """Test batch insertion with records in different namespaces."""
        records = [
            {
                "content": "Memory in ns1",
                "vector": embedding_service.embed("Memory in ns1"),
                "namespace": "ns1",
            },
            {
                "content": "Memory in ns2",
                "vector": embedding_service.embed("Memory in ns2"),
                "namespace": "ns2",
            },
            {
                "content": "Another in ns1",
                "vector": embedding_service.embed("Another in ns1"),
                "namespace": "ns1",
            },
        ]

        memory_ids = database.insert_batch(records)

        assert len(memory_ids) == 3
        assert database.count(namespace="ns1") == 2
        assert database.count(namespace="ns2") == 1

    def test_insert_batch_default_values(self, database: Database, embedding_service) -> None:
        """Test that batch insertion uses default values correctly."""
        records = [
            {
                "content": "Minimal record",
                "vector": embedding_service.embed("Minimal record"),
            },
        ]

        memory_ids = database.insert_batch(records)

        record = database.get(memory_ids[0])
        assert record["namespace"] == "default"
        assert record["importance"] == pytest.approx(0.5, abs=0.01)
        assert record["tags"] == []
        assert record["source"] == "manual"

    def test_insert_batch_with_metadata(self, database: Database, embedding_service) -> None:
        """Test batch insertion with metadata."""
        records = [
            {
                "content": "Memory with metadata",
                "vector": embedding_service.embed("Memory with metadata"),
                "metadata": {"key": "value", "count": 42},
            },
        ]

        memory_ids = database.insert_batch(records)

        record = database.get(memory_ids[0])
        assert record["metadata"] == {"key": "value", "count": 42}

    def test_insert_batch_with_list_vector(self, database: Database, embedding_service) -> None:
        """Test batch insertion accepts vector as list (not just numpy array)."""
        vec = embedding_service.embed("Test").tolist()  # Convert to list
        records = [
            {
                "content": "Memory with list vector",
                "vector": vec,
            },
        ]

        memory_ids = database.insert_batch(records)
        assert len(memory_ids) == 1

        record = database.get(memory_ids[0])
        assert record["content"] == "Memory with list vector"


class TestDeleteByNamespace:
    """Tests for namespace deletion operations."""

    def test_delete_by_namespace_success(self, database: Database, embedding_service) -> None:
        """Test successful deletion of all memories in a namespace."""
        vec = embedding_service.embed("Test")
        database.insert(content="Memory 1", vector=vec, namespace="to-delete")
        database.insert(content="Memory 2", vector=vec, namespace="to-delete")
        database.insert(content="Memory 3", vector=vec, namespace="to-keep")

        deleted_count = database.delete_by_namespace("to-delete")

        assert deleted_count == 2
        assert database.count(namespace="to-delete") == 0
        assert database.count(namespace="to-keep") == 1

    def test_delete_by_namespace_empty(self, database: Database) -> None:
        """Test deletion from namespace with no memories."""
        deleted_count = database.delete_by_namespace("nonexistent")
        assert deleted_count == 0

    def test_delete_by_namespace_validates_namespace(self, database: Database) -> None:
        """Test that delete_by_namespace validates the namespace format."""
        with pytest.raises(ValidationError):
            database.delete_by_namespace("invalid namespace")

        with pytest.raises(ValidationError):
            database.delete_by_namespace("")

    def test_delete_by_namespace_returns_correct_count(
        self, database: Database, embedding_service
    ) -> None:
        """Test that delete_by_namespace returns accurate deletion count."""
        vec = embedding_service.embed("Test")

        # Insert 5 memories in the namespace
        for i in range(5):
            database.insert(content=f"Memory {i}", vector=vec, namespace="count-test")

        assert database.count(namespace="count-test") == 5

        deleted_count = database.delete_by_namespace("count-test")

        assert deleted_count == 5
        assert database.count(namespace="count-test") == 0


class TestGetAll:
    """Tests for get_all operations."""

    def test_get_all_returns_all_memories(self, database: Database, embedding_service) -> None:
        """Test that get_all returns all memories."""
        vec = embedding_service.embed("Test")
        database.insert(content="Memory 1", vector=vec, namespace="ns1")
        database.insert(content="Memory 2", vector=vec, namespace="ns2")
        database.insert(content="Memory 3", vector=vec, namespace="ns1")

        results = database.get_all()

        assert len(results) == 3
        contents = [r["content"] for r in results]
        assert "Memory 1" in contents
        assert "Memory 2" in contents
        assert "Memory 3" in contents

    def test_get_all_empty_database(self, database: Database) -> None:
        """Test get_all on empty database."""
        results = database.get_all()
        assert results == []

    def test_get_all_with_namespace_filter(self, database: Database, embedding_service) -> None:
        """Test get_all with namespace filter."""
        vec = embedding_service.embed("Test")
        database.insert(content="Memory 1", vector=vec, namespace="ns1")
        database.insert(content="Memory 2", vector=vec, namespace="ns2")
        database.insert(content="Memory 3", vector=vec, namespace="ns1")

        results = database.get_all(namespace="ns1")

        assert len(results) == 2
        for r in results:
            assert r["namespace"] == "ns1"

    def test_get_all_with_limit(self, database: Database, embedding_service) -> None:
        """Test get_all with limit."""
        vec = embedding_service.embed("Test")
        for i in range(10):
            database.insert(content=f"Memory {i}", vector=vec)

        results = database.get_all(limit=3)

        assert len(results) == 3

    def test_get_all_with_namespace_and_limit(
        self, database: Database, embedding_service
    ) -> None:
        """Test get_all with both namespace filter and limit."""
        vec = embedding_service.embed("Test")
        for i in range(5):
            database.insert(content=f"Memory {i}", vector=vec, namespace="limited")
        database.insert(content="Other memory", vector=vec, namespace="other")

        results = database.get_all(namespace="limited", limit=2)

        assert len(results) == 2
        for r in results:
            assert r["namespace"] == "limited"

    def test_get_all_includes_metadata(self, database: Database, embedding_service) -> None:
        """Test that get_all deserializes metadata correctly."""
        vec = embedding_service.embed("Test")
        database.insert(
            content="Memory with metadata",
            vector=vec,
            metadata={"key": "value", "nested": {"a": 1}},
        )

        results = database.get_all()

        assert len(results) == 1
        assert results[0]["metadata"] == {"key": "value", "nested": {"a": 1}}

    def test_get_all_validates_namespace(self, database: Database) -> None:
        """Test that get_all validates namespace format."""
        with pytest.raises(ValidationError):
            database.get_all(namespace="invalid namespace")


class TestUpdateAccess:
    """Tests for update_access operations."""

    def test_update_access_increments_count(
        self, database: Database, embedding_service
    ) -> None:
        """Test that update_access increments access_count."""
        vec = embedding_service.embed("Test")
        memory_id = database.insert(content="Test memory", vector=vec)

        # Initial access count should be 0
        record = database.get(memory_id)
        assert record["access_count"] == 0

        # Update access
        database.update_access(memory_id)

        record = database.get(memory_id)
        assert record["access_count"] == 1

        # Update access again
        database.update_access(memory_id)

        record = database.get(memory_id)
        assert record["access_count"] == 2

    def test_update_access_updates_timestamp(
        self, database: Database, embedding_service
    ) -> None:
        """Test that update_access updates last_accessed timestamp."""
        import time

        vec = embedding_service.embed("Test")
        memory_id = database.insert(content="Test memory", vector=vec)

        record_before = database.get(memory_id)
        original_accessed = record_before["last_accessed"]

        # Small delay to ensure timestamp difference
        time.sleep(0.1)

        database.update_access(memory_id)

        record_after = database.get(memory_id)
        new_accessed = record_after["last_accessed"]

        assert new_accessed > original_accessed

    def test_update_access_nonexistent_raises(self, database: Database) -> None:
        """Test that update_access raises error for nonexistent memory."""
        with pytest.raises(MemoryNotFoundError):
            database.update_access("550e8400-e29b-41d4-a716-446655440000")

    def test_update_access_invalid_uuid_raises(self, database: Database) -> None:
        """Test that update_access raises error for invalid UUID."""
        with pytest.raises(ValidationError):
            database.update_access("not-a-uuid")


class TestUnionInjection:
    """Tests for UNION-based SQL injection prevention."""

    def test_sanitize_string_blocks_union_injection(self) -> None:
        """Test that UNION injection patterns are blocked."""
        with pytest.raises(ValidationError):
            _sanitize_string("' UNION SELECT * FROM users --")

        with pytest.raises(ValidationError):
            _sanitize_string("' UNION ALL SELECT password FROM users")

        with pytest.raises(ValidationError):
            _sanitize_string("test' UNION SELECT 1,2,3")


class TestTagsValidation:
    """Tests for tags validation."""

    def test_valid_tags(self, database: Database, embedding_service) -> None:
        """Test valid tags are accepted."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            tags=["python", "api-test", "my_tag123"]
        )
        record = database.get(memory_id)
        assert record["tags"] == ["python", "api-test", "my_tag123"]

    def test_empty_tags_list(self, database: Database, embedding_service) -> None:
        """Test that empty tags list is accepted."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            tags=[]
        )
        record = database.get(memory_id)
        assert record["tags"] == []

    def test_none_tags(self, database: Database, embedding_service) -> None:
        """Test that None tags is converted to empty list."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            tags=None
        )
        record = database.get(memory_id)
        assert record["tags"] == []

    def test_too_many_tags_raises(self, database: Database, embedding_service) -> None:
        """Test that >100 tags raises ValidationError."""
        vec = embedding_service.embed("Test content")
        too_many_tags = [f"tag{i}" for i in range(101)]

        with pytest.raises(ValidationError, match="Maximum 100 tags allowed"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=too_many_tags
            )

    def test_invalid_tag_format_raises(self, database: Database, embedding_service) -> None:
        """Test that invalid characters raise ValidationError."""
        vec = embedding_service.embed("Test content")

        # Tag with space
        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=["has space"]
            )

        # Tag with dot
        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=["has.dot"]
            )

        # Tag with special characters
        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=["has@symbol"]
            )

    def test_tag_too_long_raises(self, database: Database, embedding_service) -> None:
        """Test that tags >50 chars raise ValidationError."""
        vec = embedding_service.embed("Test content")
        long_tag = "a" * 51

        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=[long_tag]
            )

    def test_tag_empty_string_raises(self, database: Database, embedding_service) -> None:
        """Test that empty string tags raise ValidationError."""
        vec = embedding_service.embed("Test content")

        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=[""]
            )

    def test_tag_non_string_raises(self, database: Database, embedding_service) -> None:
        """Test that non-string tags raise ValidationError."""
        vec = embedding_service.embed("Test content")

        with pytest.raises(ValidationError, match="Tag must be a string"):
            database.insert(
                content="Test content",
                vector=vec,
                tags=[123]  # type: ignore
            )


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_valid_metadata(self, database: Database, embedding_service) -> None:
        """Test valid metadata is accepted."""
        vec = embedding_service.embed("Test content")
        metadata = {
            "author": "test_user",
            "version": 1,
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            metadata=metadata
        )
        record = database.get(memory_id)
        assert record["metadata"] == metadata

    def test_empty_metadata(self, database: Database, embedding_service) -> None:
        """Test that empty metadata dict is accepted."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            metadata={}
        )
        record = database.get(memory_id)
        assert record["metadata"] == {}

    def test_none_metadata(self, database: Database, embedding_service) -> None:
        """Test that None metadata is converted to empty dict."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            metadata=None
        )
        record = database.get(memory_id)
        assert record["metadata"] == {}

    def test_metadata_too_large_raises(self, database: Database, embedding_service) -> None:
        """Test that >64KB metadata raises ValidationError."""
        vec = embedding_service.embed("Test content")
        # Create metadata that exceeds 64KB
        large_value = "x" * 70000
        large_metadata = {"data": large_value}

        with pytest.raises(ValidationError, match="Metadata exceeds 64KB limit"):
            database.insert(
                content="Test content",
                vector=vec,
                metadata=large_metadata
            )

    def test_non_serializable_metadata_raises(self, database: Database, embedding_service) -> None:
        """Test that non-JSON-serializable metadata raises."""
        vec = embedding_service.embed("Test content")

        # Function is not JSON-serializable
        non_serializable = {"func": lambda x: x}

        with pytest.raises(ValidationError, match="Metadata must be JSON-serializable"):
            database.insert(
                content="Test content",
                vector=vec,
                metadata=non_serializable  # type: ignore
            )

    def test_metadata_not_dict_raises(self, database: Database, embedding_service) -> None:
        """Test that non-dict metadata raises ValidationError."""
        vec = embedding_service.embed("Test content")

        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            database.insert(
                content="Test content",
                vector=vec,
                metadata="not a dict"  # type: ignore
            )


class TestBatchValidation:
    """Tests for batch insert validation."""

    def test_batch_with_valid_tags(self, database: Database, embedding_service) -> None:
        """Test batch insert with valid tags."""
        records = [
            {
                "content": "First memory",
                "vector": embedding_service.embed("First memory"),
                "tags": ["tag1", "tag2"],
            },
            {
                "content": "Second memory",
                "vector": embedding_service.embed("Second memory"),
                "tags": ["tag3"],
            },
        ]
        ids = database.insert_batch(records)
        assert len(ids) == 2

        record1 = database.get(ids[0])
        assert record1["tags"] == ["tag1", "tag2"]

    def test_batch_with_invalid_tags_raises(self, database: Database, embedding_service) -> None:
        """Test batch insert with invalid tags raises ValidationError."""
        records = [
            {
                "content": "First memory",
                "vector": embedding_service.embed("First memory"),
                "tags": ["valid-tag"],
            },
            {
                "content": "Second memory",
                "vector": embedding_service.embed("Second memory"),
                "tags": ["invalid tag"],  # Has space
            },
        ]

        with pytest.raises(ValidationError, match="Invalid tag format"):
            database.insert_batch(records)

    def test_batch_with_valid_metadata(self, database: Database, embedding_service) -> None:
        """Test batch insert with valid metadata."""
        records = [
            {
                "content": "First memory",
                "vector": embedding_service.embed("First memory"),
                "metadata": {"key": "value1"},
            },
            {
                "content": "Second memory",
                "vector": embedding_service.embed("Second memory"),
                "metadata": {"key": "value2"},
            },
        ]
        ids = database.insert_batch(records)
        assert len(ids) == 2

    def test_batch_with_invalid_metadata_raises(
        self, database: Database, embedding_service
    ) -> None:
        """Test batch insert with invalid metadata raises ValidationError."""
        records = [
            {
                "content": "First memory",
                "vector": embedding_service.embed("First memory"),
                "metadata": {"valid": "data"},
            },
            {
                "content": "Second memory",
                "vector": embedding_service.embed("Second memory"),
                "metadata": "not a dict",  # Invalid
            },
        ]

        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            database.insert_batch(records)  # type: ignore


class TestAtomicUpdate:
    """Tests for atomic update operations using merge_insert."""

    def test_update_preserves_all_fields(
        self, database: Database, embedding_service
    ) -> None:
        """Test that update preserves fields not being updated."""
        vec = embedding_service.embed("Test content")
        memory_id = database.insert(
            content="Test content",
            vector=vec,
            namespace="test-ns",
            tags=["tag1", "tag2"],
            importance=0.7,
            metadata={"key": "value"},
        )

        # Update only importance
        database.update(memory_id, {"importance": 0.9})

        record = database.get(memory_id)
        assert record["content"] == "Test content"
        assert record["namespace"] == "test-ns"
        assert record["tags"] == ["tag1", "tag2"]
        assert record["importance"] == pytest.approx(0.9, abs=0.01)
        assert record["metadata"] == {"key": "value"}

    def test_update_multiple_fields(
        self, database: Database, embedding_service
    ) -> None:
        """Test that update can modify multiple fields at once."""
        vec = embedding_service.embed("Original content")
        memory_id = database.insert(
            content="Original content",
            vector=vec,
            importance=0.5,
            metadata={"original": True},
        )

        # Update multiple fields
        database.update(
            memory_id,
            {
                "importance": 0.8,
                "metadata": {"original": False, "updated": True},
            },
        )

        record = database.get(memory_id)
        assert record["importance"] == pytest.approx(0.8, abs=0.01)
        assert record["metadata"] == {"original": False, "updated": True}

    def test_update_sets_updated_at(
        self, database: Database, embedding_service
    ) -> None:
        """Test that update automatically sets updated_at timestamp."""
        import time

        vec = embedding_service.embed("Test content")
        memory_id = database.insert(content="Test content", vector=vec)

        record_before = database.get(memory_id)
        original_updated = record_before["updated_at"]

        time.sleep(0.1)

        database.update(memory_id, {"importance": 0.9})

        record_after = database.get(memory_id)
        assert record_after["updated_at"] > original_updated

    def test_update_with_vector(self, database: Database, embedding_service) -> None:
        """Test that update can modify the vector field."""
        vec1 = embedding_service.embed("Original content")
        memory_id = database.insert(content="Original content", vector=vec1)

        vec2 = embedding_service.embed("Updated content")
        database.update(memory_id, {"vector": vec2})

        record = database.get(memory_id)
        # Vector should be different from original
        import numpy as np

        stored_vec = np.array(record["vector"])
        assert not np.allclose(stored_vec, vec1, atol=0.01)

    def test_update_nonexistent_raises(self, database: Database) -> None:
        """Test that update raises error for nonexistent memory."""
        with pytest.raises(MemoryNotFoundError):
            database.update(
                "550e8400-e29b-41d4-a716-446655440000", {"importance": 0.5}
            )

    def test_update_invalid_uuid_raises(self, database: Database) -> None:
        """Test that update raises error for invalid UUID."""
        with pytest.raises(ValidationError):
            database.update("not-a-uuid", {"importance": 0.5})


class TestUpdateAccessBatch:
    """Tests for batch access update operations using merge_insert."""

    def test_update_access_batch_success(
        self, database: Database, embedding_service
    ) -> None:
        """Test successful batch access update."""
        vec = embedding_service.embed("Test content")
        memory_ids = [
            database.insert(content=f"Memory {i}", vector=vec) for i in range(3)
        ]

        # All should have access_count = 0 initially
        for mid in memory_ids:
            assert database.get(mid)["access_count"] == 0

        updated = database.update_access_batch(memory_ids)
        assert updated == 3

        # All should now have access_count = 1
        for mid in memory_ids:
            record = database.get(mid)
            assert record["access_count"] == 1

    def test_update_access_batch_partial_success(
        self, database: Database, embedding_service
    ) -> None:
        """Test batch access update with some nonexistent IDs."""
        vec = embedding_service.embed("Test content")
        valid_id = database.insert(content="Valid memory", vector=vec)

        # Mix of valid and invalid IDs
        memory_ids = [
            valid_id,
            "550e8400-e29b-41d4-a716-446655440000",  # Nonexistent
        ]

        updated = database.update_access_batch(memory_ids)
        assert updated == 1  # Only the valid one was updated

        record = database.get(valid_id)
        assert record["access_count"] == 1

    def test_update_access_batch_empty_list(self, database: Database) -> None:
        """Test batch access update with empty list."""
        updated = database.update_access_batch([])
        assert updated == 0

    def test_update_access_batch_updates_timestamp(
        self, database: Database, embedding_service
    ) -> None:
        """Test that batch access update updates last_accessed timestamps."""
        import time

        vec = embedding_service.embed("Test content")
        memory_id = database.insert(content="Test memory", vector=vec)

        original_accessed = database.get(memory_id)["last_accessed"]
        time.sleep(0.1)

        database.update_access_batch([memory_id])

        new_accessed = database.get(memory_id)["last_accessed"]
        assert new_accessed > original_accessed
