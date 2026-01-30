"""Comprehensive tests for centralized validation module.

Tests follow TDD principles with 95%+ coverage target.
"""

from __future__ import annotations

import pytest

from spatial_memory.core.errors import ValidationError
from spatial_memory.core.validation import (
    MAX_CONTENT_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    sanitize_string,
    validate_content,
    validate_importance,
    validate_metadata,
    validate_namespace,
    validate_tags,
    validate_uuid,
)


class TestValidateUUID:
    """Tests for UUID validation."""

    def test_valid_uuid_v4(self) -> None:
        """Test valid UUID v4 format."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_uuid(valid_uuid) == valid_uuid

    def test_valid_uuid_v1(self) -> None:
        """Test valid UUID v1 format."""
        valid_uuid = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        assert validate_uuid(valid_uuid) == valid_uuid

    def test_valid_uuid_with_uppercase(self) -> None:
        """Test valid UUID with uppercase letters."""
        valid_uuid = "550E8400-E29B-41D4-A716-446655440000"
        assert validate_uuid(valid_uuid) == valid_uuid

    def test_invalid_uuid_format(self) -> None:
        """Test invalid UUID format."""
        with pytest.raises(ValidationError, match="Invalid UUID format"):
            validate_uuid("not-a-uuid")

    def test_invalid_uuid_empty(self) -> None:
        """Test empty UUID."""
        with pytest.raises(ValidationError, match="Invalid UUID format"):
            validate_uuid("")

    def test_invalid_uuid_wrong_length(self) -> None:
        """Test UUID with wrong length."""
        with pytest.raises(ValidationError, match="Invalid UUID format"):
            validate_uuid("550e8400-e29b-41d4-a716")

    def test_valid_uuid_no_hyphens(self) -> None:
        """Test UUID without hyphens (Python's UUID accepts this)."""
        # Python's uuid.UUID accepts UUIDs without hyphens
        result = validate_uuid("550e8400e29b41d4a716446655440000")
        assert result == "550e8400e29b41d4a716446655440000"


class TestValidateNamespace:
    """Tests for namespace validation."""

    def test_valid_namespace_simple(self) -> None:
        """Test simple valid namespace."""
        assert validate_namespace("default") == "default"

    def test_valid_namespace_with_dash(self) -> None:
        """Test namespace with dash."""
        assert validate_namespace("my-namespace") == "my-namespace"

    def test_valid_namespace_with_underscore(self) -> None:
        """Test namespace with underscore."""
        assert validate_namespace("my_namespace") == "my_namespace"

    def test_valid_namespace_with_dot(self) -> None:
        """Test namespace with dot."""
        assert validate_namespace("my.namespace") == "my.namespace"

    def test_valid_namespace_mixed_chars(self) -> None:
        """Test namespace with mixed allowed characters."""
        assert validate_namespace("my-test_namespace.v1") == "my-test_namespace.v1"

    def test_invalid_namespace_empty(self) -> None:
        """Test empty namespace."""
        with pytest.raises(ValidationError, match="Namespace cannot be empty"):
            validate_namespace("")

    def test_invalid_namespace_too_long(self) -> None:
        """Test namespace exceeding max length."""
        long_namespace = "a" * 257
        with pytest.raises(ValidationError, match="Namespace too long"):
            validate_namespace(long_namespace)

    def test_invalid_namespace_special_chars(self) -> None:
        """Test namespace with invalid characters."""
        with pytest.raises(ValidationError, match="Invalid namespace format"):
            validate_namespace("my@namespace")

    def test_invalid_namespace_space(self) -> None:
        """Test namespace with space."""
        with pytest.raises(ValidationError, match="Invalid namespace format"):
            validate_namespace("my namespace")

    def test_invalid_namespace_slash(self) -> None:
        """Test namespace with slash."""
        with pytest.raises(ValidationError, match="Invalid namespace format"):
            validate_namespace("my/namespace")


class TestValidateContent:
    """Tests for content validation."""

    def test_valid_content(self) -> None:
        """Test valid content."""
        content = "This is valid content"
        validate_content(content)  # Should not raise

    def test_valid_content_long(self) -> None:
        """Test valid long content."""
        content = "x" * 50000
        validate_content(content)  # Should not raise

    def test_valid_content_max_length(self) -> None:
        """Test content at maximum length."""
        content = "x" * MAX_CONTENT_LENGTH
        validate_content(content)  # Should not raise

    def test_invalid_content_empty(self) -> None:
        """Test empty content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            validate_content("")

    def test_invalid_content_whitespace_only(self) -> None:
        """Test whitespace-only content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            validate_content("   \n\t   ")

    def test_invalid_content_too_long(self) -> None:
        """Test content exceeding max length."""
        content = "x" * (MAX_CONTENT_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_content(content)


class TestValidateImportance:
    """Tests for importance validation."""

    def test_valid_importance_zero(self) -> None:
        """Test importance value of 0.0."""
        validate_importance(0.0)  # Should not raise

    def test_valid_importance_one(self) -> None:
        """Test importance value of 1.0."""
        validate_importance(1.0)  # Should not raise

    def test_valid_importance_middle(self) -> None:
        """Test importance value in middle range."""
        validate_importance(0.5)  # Should not raise

    def test_valid_importance_precision(self) -> None:
        """Test importance with high precision."""
        validate_importance(0.123456)  # Should not raise

    def test_invalid_importance_negative(self) -> None:
        """Test negative importance."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_importance(-0.1)

    def test_invalid_importance_above_one(self) -> None:
        """Test importance above 1.0."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_importance(1.1)

    def test_invalid_importance_large_negative(self) -> None:
        """Test large negative importance."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_importance(-100.0)


class TestValidateTags:
    """Tests for tags validation."""

    def test_valid_tags_empty_list(self) -> None:
        """Test empty tags list."""
        assert validate_tags([]) == []

    def test_valid_tags_none(self) -> None:
        """Test None tags."""
        assert validate_tags(None) == []

    def test_valid_tags_single(self) -> None:
        """Test single tag."""
        assert validate_tags(["test"]) == ["test"]

    def test_valid_tags_multiple(self) -> None:
        """Test multiple tags."""
        tags = ["tag1", "tag2", "tag3"]
        assert validate_tags(tags) == tags

    def test_valid_tags_with_dash(self) -> None:
        """Test tags with dash."""
        tags = ["my-tag"]
        assert validate_tags(tags) == tags

    def test_valid_tags_with_underscore(self) -> None:
        """Test tags with underscore."""
        tags = ["my_tag"]
        assert validate_tags(tags) == tags

    def test_valid_tags_alphanumeric(self) -> None:
        """Test alphanumeric tags."""
        tags = ["tag123", "456tag"]
        assert validate_tags(tags) == tags

    def test_valid_tags_max_length(self) -> None:
        """Test tag at maximum length."""
        tag = "a" * MAX_TAG_LENGTH
        assert validate_tags([tag]) == [tag]

    def test_invalid_tags_too_many(self) -> None:
        """Test exceeding maximum number of tags."""
        tags = [f"tag{i}" for i in range(MAX_TAGS + 1)]
        with pytest.raises(ValidationError, match=f"Maximum {MAX_TAGS} tags allowed"):
            validate_tags(tags)

    def test_invalid_tags_not_string(self) -> None:
        """Test non-string tag."""
        with pytest.raises(ValidationError, match="Tag must be a string"):
            validate_tags([123])  # type: ignore[list-item]

    def test_invalid_tags_too_long(self) -> None:
        """Test tag exceeding max length."""
        tag = "a" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValidationError, match="Invalid tag format"):
            validate_tags([tag])

    def test_invalid_tags_empty_string(self) -> None:
        """Test empty string tag."""
        with pytest.raises(ValidationError, match="Invalid tag format"):
            validate_tags([""])

    def test_invalid_tags_special_chars(self) -> None:
        """Test tag with invalid characters."""
        with pytest.raises(ValidationError, match="Invalid tag format"):
            validate_tags(["tag@test"])

    def test_invalid_tags_space(self) -> None:
        """Test tag with space."""
        with pytest.raises(ValidationError, match="Invalid tag format"):
            validate_tags(["tag test"])

    def test_invalid_tags_starting_with_dash(self) -> None:
        """Test tag starting with dash."""
        with pytest.raises(ValidationError, match="Invalid tag format"):
            validate_tags(["-tag"])


class TestValidateMetadata:
    """Tests for metadata validation."""

    def test_valid_metadata_empty(self) -> None:
        """Test empty metadata dict."""
        assert validate_metadata({}) == {}

    def test_valid_metadata_none(self) -> None:
        """Test None metadata."""
        assert validate_metadata(None) == {}

    def test_valid_metadata_simple(self) -> None:
        """Test simple metadata."""
        metadata = {"key": "value"}
        assert validate_metadata(metadata) == metadata

    def test_valid_metadata_nested(self) -> None:
        """Test nested metadata."""
        metadata = {"nested": {"key": "value"}}
        assert validate_metadata(metadata) == metadata

    def test_valid_metadata_mixed_types(self) -> None:
        """Test metadata with mixed types."""
        metadata = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "null": None,
        }
        assert validate_metadata(metadata) == metadata

    def test_invalid_metadata_not_dict(self) -> None:
        """Test non-dict metadata."""
        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            validate_metadata("not a dict")  # type: ignore[arg-type]

    def test_invalid_metadata_too_large(self) -> None:
        """Test metadata exceeding size limit."""
        # Create metadata that exceeds MAX_METADATA_SIZE (64KB) when serialized
        large_value = "x" * 70000  # 70KB string
        metadata = {"data": large_value}
        with pytest.raises(ValidationError, match="exceeds.*limit"):
            validate_metadata(metadata)

    def test_invalid_metadata_not_serializable(self) -> None:
        """Test non-JSON-serializable metadata."""
        metadata = {"func": lambda x: x}  # Functions can't be serialized
        with pytest.raises(ValidationError, match="must be JSON-serializable"):
            validate_metadata(metadata)


class TestSanitizeString:
    """Tests for string sanitization."""

    def test_sanitize_simple_string(self) -> None:
        """Test sanitizing simple string."""
        assert sanitize_string("hello") == "hello"

    def test_sanitize_string_with_single_quote(self) -> None:
        """Test sanitizing string with single quote."""
        assert sanitize_string("it's") == "it''s"

    def test_sanitize_string_with_multiple_quotes(self) -> None:
        """Test sanitizing string with multiple quotes."""
        assert sanitize_string("'hello'") == "''hello''"

    def test_sanitize_string_empty(self) -> None:
        """Test sanitizing empty string."""
        assert sanitize_string("") == ""

    def test_sanitize_invalid_sql_injection_drop(self) -> None:
        """Test SQL injection attempt with DROP."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("'; DROP TABLE users--")

    def test_sanitize_invalid_sql_injection_delete(self) -> None:
        """Test SQL injection attempt with DELETE."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("'; DELETE FROM users--")

    def test_sanitize_invalid_sql_injection_update(self) -> None:
        """Test SQL injection attempt with UPDATE."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("'; UPDATE users SET admin=1--")

    def test_sanitize_invalid_sql_injection_union(self) -> None:
        """Test SQL injection attempt with UNION."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("' UNION SELECT * FROM users--")

    def test_sanitize_invalid_sql_injection_or(self) -> None:
        """Test SQL injection attempt with OR."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("' OR '1'='1")

    def test_sanitize_invalid_sql_injection_and(self) -> None:
        """Test SQL injection attempt with AND."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("' AND '1'='1")

    def test_sanitize_invalid_sql_comment(self) -> None:
        """Test SQL comment injection."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("test--")

    def test_sanitize_invalid_multiline_comment(self) -> None:
        """Test SQL multiline comment injection."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("test /* comment */")

    def test_sanitize_invalid_not_string(self) -> None:
        """Test non-string input."""
        with pytest.raises(ValidationError, match="Expected string"):
            sanitize_string(123)  # type: ignore[arg-type]

    def test_sanitize_case_insensitive_sql_injection(self) -> None:
        """Test SQL injection with mixed case."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            sanitize_string("'; dRoP tAbLe users--")
