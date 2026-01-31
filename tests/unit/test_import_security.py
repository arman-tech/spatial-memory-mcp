"""Unit tests for import validation module.

Security-critical tests for import data validation covering:
- Required field validation (content)
- Optional field validation (namespace, tags, importance, metadata, vector)
- Vector dimension validation
- Schema validation with proper error types
- Fail-fast and max-errors modes

These tests follow TDD - written BEFORE implementation.
"""

from __future__ import annotations

from typing import Any

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_record() -> dict[str, Any]:
    """Return a valid import record with all fields."""
    return {
        "content": "This is a valid memory content.",
        "namespace": "test-namespace",
        "tags": ["tag1", "tag2"],
        "importance": 0.75,
        "metadata": {"source": "test", "version": 1},
        "vector": [0.1] * 384,  # 384-dim vector
    }


@pytest.fixture
def minimal_valid_record() -> dict[str, Any]:
    """Return a valid record with only required field (content)."""
    return {
        "content": "Minimal valid content.",
    }


@pytest.fixture
def validator() -> Any:
    """Create an ImportValidator with default config."""
    from spatial_memory.core.import_security import (
        ImportValidationConfig,
        ImportValidator,
    )

    config = ImportValidationConfig(expected_vector_dim=384)
    return ImportValidator(config)


@pytest.fixture
def validator_fail_fast() -> Any:
    """Create an ImportValidator with fail_fast=True."""
    from spatial_memory.core.import_security import (
        ImportValidationConfig,
        ImportValidator,
    )

    config = ImportValidationConfig(
        expected_vector_dim=384,
        fail_fast=True,
    )
    return ImportValidator(config)


@pytest.fixture
def validator_max_errors() -> Any:
    """Create an ImportValidator with max_errors=5."""
    from spatial_memory.core.import_security import (
        ImportValidationConfig,
        ImportValidator,
    )

    config = ImportValidationConfig(
        expected_vector_dim=384,
        max_errors=5,
    )
    return ImportValidator(config)


# =============================================================================
# TestImportValidationConfig
# =============================================================================


class TestImportValidationConfig:
    """Tests for ImportValidationConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has correct default values."""
        from spatial_memory.core.import_security import ImportValidationConfig

        config = ImportValidationConfig(expected_vector_dim=384)

        assert config.expected_vector_dim == 384
        assert config.fail_fast is False
        assert config.max_errors == 100

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        from spatial_memory.core.import_security import ImportValidationConfig

        config = ImportValidationConfig(
            expected_vector_dim=512,
            fail_fast=True,
            max_errors=50,
        )

        assert config.expected_vector_dim == 512
        assert config.fail_fast is True
        assert config.max_errors == 50


# =============================================================================
# TestValidRecords
# =============================================================================


class TestValidRecords:
    """Tests for valid record validation."""

    def test_valid_record_passes(self, validator: Any, valid_record: dict[str, Any]) -> None:
        """Valid record with all fields passes validation."""
        is_valid, errors = validator.validate_record(valid_record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_minimal_valid_record_passes(
        self, validator: Any, minimal_valid_record: dict[str, Any]
    ) -> None:
        """Minimal valid record (content only) passes validation."""
        is_valid, errors = validator.validate_record(minimal_valid_record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_valid_record_with_empty_optional_fields(self, validator: Any) -> None:
        """Valid record with empty optional fields passes."""
        record = {
            "content": "Valid content here.",
            "tags": [],
            "metadata": {},
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestContentValidation
# =============================================================================


class TestContentValidation:
    """Tests for content field validation."""

    def test_missing_content_fails(self, validator: Any) -> None:
        """Missing content field fails validation."""
        record: dict[str, Any] = {
            "namespace": "test",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "content" in errors[0].lower()
        assert "required" in errors[0].lower()

    def test_empty_content_fails(self, validator: Any) -> None:
        """Empty content string fails validation."""
        record = {
            "content": "",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "content" in errors[0].lower()
        assert "empty" in errors[0].lower() or "non-empty" in errors[0].lower()

    def test_whitespace_only_content_fails(self, validator: Any) -> None:
        """Content with only whitespace fails validation."""
        record = {
            "content": "   \t\n  ",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "content" in errors[0].lower()

    def test_non_string_content_fails(self, validator: Any) -> None:
        """Non-string content type fails validation."""
        record = {
            "content": 12345,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "content" in errors[0].lower()
        assert "string" in errors[0].lower()

    def test_none_content_fails(self, validator: Any) -> None:
        """None content fails validation."""
        record = {
            "content": None,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) >= 1


# =============================================================================
# TestVectorDimensionValidation
# =============================================================================


class TestVectorDimensionValidation:
    """Tests for vector dimension validation."""

    def test_correct_vector_dimensions_pass(self, validator: Any) -> None:
        """Vector with correct dimensions passes."""
        record = {
            "content": "Test content",
            "vector": [0.1] * 384,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_wrong_vector_dimensions_fail(self, validator: Any) -> None:
        """Vector with wrong dimensions fails."""
        record = {
            "content": "Test content",
            "vector": [0.1] * 512,  # Wrong: 512 instead of 384
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "dimension" in errors[0].lower() or "vector" in errors[0].lower()
        assert "384" in errors[0]
        assert "512" in errors[0]

    def test_empty_vector_fails(self, validator: Any) -> None:
        """Empty vector fails validation."""
        record = {
            "content": "Test content",
            "vector": [],
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "dimension" in errors[0].lower() or "vector" in errors[0].lower()

    def test_non_list_vector_fails(self, validator: Any) -> None:
        """Non-list vector type fails validation."""
        record = {
            "content": "Test content",
            "vector": "not a vector",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "vector" in errors[0].lower()

    def test_vector_with_non_numeric_values_fails(self, validator: Any) -> None:
        """Vector with non-numeric values fails."""
        record = {
            "content": "Test content",
            "vector": ["a", "b", "c"] + [0.1] * 381,  # Non-numeric at start
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) >= 1
        assert "vector" in errors[0].lower()

    def test_none_vector_allowed(self, validator: Any) -> None:
        """None/missing vector is allowed (will be generated)."""
        record = {
            "content": "Test content without vector",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestImportanceValidation
# =============================================================================


class TestImportanceValidation:
    """Tests for importance field validation."""

    def test_valid_importance_passes(self, validator: Any) -> None:
        """Valid importance values pass."""
        for importance in [0.0, 0.5, 1.0, 0.33]:
            record = {
                "content": "Test content",
                "importance": importance,
            }

            is_valid, errors = validator.validate_record(record, 384, 0)

            assert is_valid is True, f"importance={importance} should be valid"
            assert errors == []

    def test_importance_below_zero_fails(self, validator: Any) -> None:
        """Importance below 0 fails validation."""
        record = {
            "content": "Test content",
            "importance": -0.1,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "importance" in errors[0].lower()
        assert "0" in errors[0] and "1" in errors[0]

    def test_importance_above_one_fails(self, validator: Any) -> None:
        """Importance above 1 fails validation."""
        record = {
            "content": "Test content",
            "importance": 1.5,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "importance" in errors[0].lower()

    def test_non_numeric_importance_fails(self, validator: Any) -> None:
        """Non-numeric importance fails validation."""
        record = {
            "content": "Test content",
            "importance": "high",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "importance" in errors[0].lower()

    def test_missing_importance_uses_default(self, validator: Any) -> None:
        """Missing importance is allowed (uses default)."""
        record = {
            "content": "Test content without importance",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestTagsValidation
# =============================================================================


class TestTagsValidation:
    """Tests for tags field validation."""

    def test_valid_tags_pass(self, validator: Any) -> None:
        """Valid tags list passes."""
        record = {
            "content": "Test content",
            "tags": ["python", "testing", "validation"],
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_empty_tags_list_passes(self, validator: Any) -> None:
        """Empty tags list passes."""
        record = {
            "content": "Test content",
            "tags": [],
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_non_list_tags_fails(self, validator: Any) -> None:
        """Non-list tags type fails validation."""
        record = {
            "content": "Test content",
            "tags": "python, testing",  # String instead of list
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "tags" in errors[0].lower()
        assert "list" in errors[0].lower()

    def test_tags_with_non_string_items_fails(self, validator: Any) -> None:
        """Tags list with non-string items fails."""
        record = {
            "content": "Test content",
            "tags": ["valid", 123, None],  # Non-string items
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) >= 1
        assert "tags" in errors[0].lower()

    def test_missing_tags_uses_default(self, validator: Any) -> None:
        """Missing tags is allowed (uses default empty list)."""
        record = {
            "content": "Test content without tags",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestNamespaceValidation
# =============================================================================


class TestNamespaceValidation:
    """Tests for namespace field validation."""

    def test_valid_namespace_passes(self, validator: Any) -> None:
        """Valid namespace values pass."""
        valid_namespaces = ["default", "test-ns", "my_namespace", "project123"]
        for ns in valid_namespaces:
            record = {
                "content": "Test content",
                "namespace": ns,
            }

            is_valid, errors = validator.validate_record(record, 384, 0)

            assert is_valid is True, f"namespace='{ns}' should be valid"
            assert errors == []

    def test_empty_namespace_fails(self, validator: Any) -> None:
        """Empty namespace string fails validation."""
        record = {
            "content": "Test content",
            "namespace": "",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "namespace" in errors[0].lower()

    def test_non_string_namespace_fails(self, validator: Any) -> None:
        """Non-string namespace fails validation."""
        record = {
            "content": "Test content",
            "namespace": 12345,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "namespace" in errors[0].lower()

    def test_namespace_with_invalid_chars_fails(self, validator: Any) -> None:
        """Namespace with invalid characters fails validation."""
        invalid_namespaces = ["ns/path", "ns..name", "ns\\path", "ns;drop"]
        for ns in invalid_namespaces:
            record = {
                "content": "Test content",
                "namespace": ns,
            }

            is_valid, errors = validator.validate_record(record, 384, 0)

            assert is_valid is False, f"namespace='{ns}' should be invalid"
            assert len(errors) >= 1
            assert "namespace" in errors[0].lower()

    def test_missing_namespace_uses_default(self, validator: Any) -> None:
        """Missing namespace is allowed (uses default)."""
        record = {
            "content": "Test content without namespace",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestMetadataValidation
# =============================================================================


class TestMetadataValidation:
    """Tests for metadata field validation."""

    def test_valid_metadata_passes(self, validator: Any) -> None:
        """Valid metadata dict passes."""
        record = {
            "content": "Test content",
            "metadata": {"key": "value", "number": 42, "nested": {"a": 1}},
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_empty_metadata_passes(self, validator: Any) -> None:
        """Empty metadata dict passes."""
        record = {
            "content": "Test content",
            "metadata": {},
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_non_dict_metadata_fails(self, validator: Any) -> None:
        """Non-dict metadata type fails validation."""
        record = {
            "content": "Test content",
            "metadata": "not a dict",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "metadata" in errors[0].lower()
        assert "dict" in errors[0].lower()

    def test_list_metadata_fails(self, validator: Any) -> None:
        """List metadata fails validation."""
        record = {
            "content": "Test content",
            "metadata": ["not", "a", "dict"],
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1
        assert "metadata" in errors[0].lower()

    def test_missing_metadata_uses_default(self, validator: Any) -> None:
        """Missing metadata is allowed (uses default empty dict)."""
        record = {
            "content": "Test content without metadata",
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestMultipleErrors
# =============================================================================


class TestMultipleErrors:
    """Tests for handling multiple validation errors."""

    def test_multiple_errors_collected(self, validator: Any) -> None:
        """Multiple validation errors are collected."""
        record = {
            "content": "",  # Error 1: empty content
            "importance": 2.0,  # Error 2: out of range
            "tags": "not-a-list",  # Error 3: wrong type
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 3

    def test_fail_fast_stops_on_first_error(self, validator_fail_fast: Any) -> None:
        """Fail-fast mode stops on first error."""
        record = {
            "content": "",  # Error 1
            "importance": 2.0,  # Error 2
            "tags": "not-a-list",  # Error 3
        }

        is_valid, errors = validator_fail_fast.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) == 1

    def test_max_errors_limits_collection(self, validator_max_errors: Any) -> None:
        """Max errors limit prevents collecting too many errors."""
        # Record with more than 5 errors
        record = {
            "content": "",  # Error 1
            "namespace": "",  # Error 2
            "importance": 2.0,  # Error 3
            "tags": "not-a-list",  # Error 4
            "metadata": "not-a-dict",  # Error 5
            "vector": "not-a-vector",  # Error 6 (should not be collected)
        }

        is_valid, errors = validator_max_errors.validate_record(record, 384, 0)

        assert is_valid is False
        assert len(errors) <= 5


# =============================================================================
# TestRecordIndex
# =============================================================================


class TestRecordIndex:
    """Tests for record index in error messages."""

    def test_record_index_in_error_message(self, validator: Any) -> None:
        """Record index is included in error messages."""
        record = {
            "content": "",  # Invalid
        }

        is_valid, errors = validator.validate_record(record, 384, 42)

        assert is_valid is False
        assert len(errors) == 1
        # Error message should reference record 42
        assert "42" in errors[0] or "record 42" in errors[0].lower()


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_vector_boundary_values_valid(self, validator: Any) -> None:
        """Vector with boundary float values is valid."""
        record = {
            "content": "Test content",
            "vector": [1e-10, -1e-10, 0.0, 1.0, -1.0] + [0.1] * 379,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_importance_boundary_values(self, validator: Any) -> None:
        """Importance at exact boundaries is valid."""
        for importance in [0.0, 1.0]:
            record = {
                "content": "Test content",
                "importance": importance,
            }

            is_valid, errors = validator.validate_record(record, 384, 0)

            assert is_valid is True, f"importance={importance} should be valid"

    def test_unicode_content_valid(self, validator: Any) -> None:
        """Unicode content is valid."""
        unicode_text = "Unicode: Chinese: \u4e2d\u6587 Japanese: \u65e5\u672c\u8a9e"
        record = {
            "content": unicode_text,
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_very_long_content_valid(self, validator: Any) -> None:
        """Very long content is valid (no max length)."""
        record = {
            "content": "x" * 100000,  # 100KB of content
        }

        is_valid, errors = validator.validate_record(record, 384, 0)

        assert is_valid is True
        assert errors == []

    def test_integer_importance_valid(self, validator: Any) -> None:
        """Integer importance (0 or 1) is valid."""
        for importance in [0, 1]:
            record = {
                "content": "Test content",
                "importance": importance,
            }

            is_valid, errors = validator.validate_record(record, 384, 0)

            assert is_valid is True, f"importance={importance} should be valid"


# =============================================================================
# TestValidateBatch
# =============================================================================


class TestValidateBatch:
    """Tests for batch validation functionality."""

    def test_validate_batch_all_valid(self, validator: Any) -> None:
        """Batch validation with all valid records."""
        from spatial_memory.core.import_security import BatchValidationResult

        records = [
            {"content": f"Valid content {i}"} for i in range(5)
        ]

        result = validator.validate_batch(records, 384)

        assert isinstance(result, BatchValidationResult)
        assert result.valid_count == 5
        assert result.invalid_count == 0
        assert len(result.errors) == 0
        assert result.is_valid is True

    def test_validate_batch_some_invalid(self, validator: Any) -> None:
        """Batch validation with some invalid records."""
        records = [
            {"content": "Valid 1"},
            {"content": ""},  # Invalid
            {"content": "Valid 2"},
            {"content": None},  # Invalid
            {"content": "Valid 3"},
        ]

        result = validator.validate_batch(records, 384)

        assert result.valid_count == 3
        assert result.invalid_count == 2
        assert len(result.errors) == 2
        assert result.is_valid is False

    def test_validate_batch_fail_fast(self, validator_fail_fast: Any) -> None:
        """Batch validation with fail_fast stops on first invalid."""
        records = [
            {"content": "Valid 1"},
            {"content": ""},  # Invalid - should stop here
            {"content": "Valid 2"},
            {"content": ""},  # Should not reach
        ]

        result = validator_fail_fast.validate_batch(records, 384)

        assert result.valid_count == 1
        assert result.invalid_count == 1
        assert len(result.errors) == 1
        assert result.is_valid is False
