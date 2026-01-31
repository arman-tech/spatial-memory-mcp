"""Unit tests for Phase 5.1 error types.

These tests verify the new error types for Phase 5 utility operations:
- ExportError
- MemoryImportError (renamed to avoid shadowing Python's built-in ImportError)
- NamespaceOperationError
- PathSecurityError
- FileSizeLimitError
- DimensionMismatchError (extends ValidationError)
- SchemaValidationError (extends ValidationError)
- ImportRecordLimitError
"""

import pytest

from spatial_memory.core.errors import (
    DimensionMismatchError,
    ExportError,
    FileSizeLimitError,
    ImportRecordLimitError,
    MemoryImportError,
    NamespaceOperationError,
    PathSecurityError,
    SchemaValidationError,
    SpatialMemoryError,
    ValidationError,
)


class TestExportError:
    """Tests for ExportError."""

    def test_export_error_is_spatial_memory_error(self):
        """ExportError should extend SpatialMemoryError."""
        error = ExportError("Export failed")
        assert isinstance(error, SpatialMemoryError)

    def test_export_error_message(self):
        """ExportError should store message correctly."""
        error = ExportError("Failed to write parquet file")
        assert str(error) == "Failed to write parquet file"

    def test_export_error_can_be_raised_and_caught(self):
        """ExportError should be raisable and catchable."""
        with pytest.raises(ExportError) as exc_info:
            raise ExportError("Export operation failed")
        assert "Export operation failed" in str(exc_info.value)


class TestMemoryImportError:
    """Tests for MemoryImportError.

    Note: Named MemoryImportError to avoid shadowing Python's built-in ImportError.
    """

    def test_memory_import_error_is_spatial_memory_error(self):
        """MemoryImportError should extend SpatialMemoryError."""
        error = MemoryImportError("Import failed")
        assert isinstance(error, SpatialMemoryError)

    def test_memory_import_error_message(self):
        """MemoryImportError should store message correctly."""
        error = MemoryImportError("Failed to parse JSON file")
        assert str(error) == "Failed to parse JSON file"

    def test_memory_import_error_does_not_shadow_builtin(self):
        """MemoryImportError should not shadow Python's built-in ImportError."""
        # Python's built-in ImportError should still work
        try:
            raise ImportError("Module not found")
        except ImportError as e:
            assert "Module not found" in str(e)
            # Verify it's the built-in, not ours
            assert not isinstance(e, SpatialMemoryError)


class TestNamespaceOperationError:
    """Tests for NamespaceOperationError."""

    def test_namespace_operation_error_is_spatial_memory_error(self):
        """NamespaceOperationError should extend SpatialMemoryError."""
        error = NamespaceOperationError("Namespace operation failed")
        assert isinstance(error, SpatialMemoryError)

    def test_namespace_operation_error_message(self):
        """NamespaceOperationError should store message correctly."""
        error = NamespaceOperationError("Cannot delete non-empty namespace")
        assert str(error) == "Cannot delete non-empty namespace"


class TestPathSecurityError:
    """Tests for PathSecurityError."""

    def test_path_security_error_is_spatial_memory_error(self):
        """PathSecurityError should extend SpatialMemoryError."""
        error = PathSecurityError(
            path="../../../etc/passwd",
            violation_type="traversal_attempt",
        )
        assert isinstance(error, SpatialMemoryError)

    def test_path_security_error_stores_fields(self):
        """PathSecurityError should store path and violation_type fields."""
        error = PathSecurityError(
            path="/etc/passwd",
            violation_type="outside_allowed_directories",
        )
        assert error.path == "/etc/passwd"
        assert error.violation_type == "outside_allowed_directories"

    def test_path_security_error_default_message(self):
        """PathSecurityError should generate default message from fields."""
        error = PathSecurityError(
            path="../secret",
            violation_type="traversal_attempt",
        )
        assert "Path security violation" in str(error)
        assert "traversal_attempt" in str(error)
        assert "../secret" in str(error)

    def test_path_security_error_custom_message(self):
        """PathSecurityError should accept custom message."""
        custom_msg = "Symlink to /etc/passwd detected"
        error = PathSecurityError(
            path="/exports/symlink",
            violation_type="symlink_attack",
            message=custom_msg,
        )
        assert str(error) == custom_msg
        assert error.message == custom_msg

    @pytest.mark.parametrize(
        "path,violation_type",
        [
            ("../../../etc/passwd", "traversal_attempt"),
            ("..\\..\\Windows\\System32", "traversal_attempt"),
            ("/etc/shadow", "sensitive_directory"),
            ("exports/link", "symlink_outside_allowed"),
            ("backup.exe", "invalid_extension"),
        ],
    )
    def test_path_security_error_various_violations(self, path, violation_type):
        """PathSecurityError should handle various violation types."""
        error = PathSecurityError(path=path, violation_type=violation_type)
        assert error.path == path
        assert error.violation_type == violation_type


class TestFileSizeLimitError:
    """Tests for FileSizeLimitError."""

    def test_file_size_limit_error_is_spatial_memory_error(self):
        """FileSizeLimitError should extend SpatialMemoryError."""
        error = FileSizeLimitError(
            path="/imports/large.json",
            actual_size_bytes=150 * 1024 * 1024,  # 150MB
            max_size_bytes=100 * 1024 * 1024,  # 100MB
        )
        assert isinstance(error, SpatialMemoryError)

    def test_file_size_limit_error_stores_fields(self):
        """FileSizeLimitError should store all fields."""
        actual = 150 * 1024 * 1024
        maximum = 100 * 1024 * 1024
        error = FileSizeLimitError(
            path="/imports/large.parquet",
            actual_size_bytes=actual,
            max_size_bytes=maximum,
        )
        assert error.path == "/imports/large.parquet"
        assert error.actual_size_bytes == actual
        assert error.max_size_bytes == maximum

    def test_file_size_limit_error_message_format(self):
        """FileSizeLimitError should format message with MB values."""
        error = FileSizeLimitError(
            path="/imports/huge.csv",
            actual_size_bytes=157286400,  # 150MB
            max_size_bytes=104857600,  # 100MB
        )
        message = str(error)
        assert "exceeds size limit" in message
        assert "/imports/huge.csv" in message
        assert "150" in message  # Actual MB
        assert "100" in message  # Max MB

    def test_file_size_limit_error_small_file(self):
        """FileSizeLimitError should handle small file sizes correctly."""
        error = FileSizeLimitError(
            path="test.json",
            actual_size_bytes=1024,  # 1KB
            max_size_bytes=512,  # 0.5KB
        )
        assert error.actual_size_bytes == 1024
        assert error.max_size_bytes == 512


class TestDimensionMismatchError:
    """Tests for DimensionMismatchError."""

    def test_dimension_mismatch_error_is_validation_error(self):
        """DimensionMismatchError should extend ValidationError."""
        error = DimensionMismatchError(
            expected_dim=384,
            actual_dim=512,
        )
        assert isinstance(error, ValidationError)
        assert isinstance(error, SpatialMemoryError)

    def test_dimension_mismatch_error_stores_fields(self):
        """DimensionMismatchError should store dimension fields."""
        error = DimensionMismatchError(
            expected_dim=384,
            actual_dim=768,
            record_index=42,
        )
        assert error.expected_dim == 384
        assert error.actual_dim == 768
        assert error.record_index == 42

    def test_dimension_mismatch_error_message_without_index(self):
        """DimensionMismatchError should format message without record index."""
        error = DimensionMismatchError(
            expected_dim=384,
            actual_dim=512,
        )
        message = str(error)
        assert "dimension mismatch" in message.lower()
        assert "384" in message
        assert "512" in message
        assert "record" not in message.lower()

    def test_dimension_mismatch_error_message_with_index(self):
        """DimensionMismatchError should format message with record index."""
        error = DimensionMismatchError(
            expected_dim=384,
            actual_dim=1024,
            record_index=99,
        )
        message = str(error)
        assert "dimension mismatch" in message.lower()
        assert "384" in message
        assert "1024" in message
        assert "record 99" in message.lower()


class TestSchemaValidationError:
    """Tests for SchemaValidationError."""

    def test_schema_validation_error_is_validation_error(self):
        """SchemaValidationError should extend ValidationError."""
        error = SchemaValidationError(
            field="content",
            error="Field is required",
        )
        assert isinstance(error, ValidationError)
        assert isinstance(error, SpatialMemoryError)

    def test_schema_validation_error_stores_fields(self):
        """SchemaValidationError should store all fields."""
        error = SchemaValidationError(
            field="namespace",
            error="Invalid characters",
            record_index=15,
        )
        assert error.field == "namespace"
        assert error.error == "Invalid characters"
        assert error.record_index == 15

    def test_schema_validation_error_message_without_index(self):
        """SchemaValidationError should format message without record index."""
        error = SchemaValidationError(
            field="importance",
            error="Must be between 0 and 1",
        )
        message = str(error)
        assert "validation failed" in message.lower()
        assert "importance" in message
        assert "Must be between 0 and 1" in message
        assert "record" not in message.lower()

    def test_schema_validation_error_message_with_index(self):
        """SchemaValidationError should format message with record index."""
        error = SchemaValidationError(
            field="tags",
            error="Must be a list",
            record_index=7,
        )
        message = str(error)
        assert "validation failed" in message.lower()
        assert "tags" in message
        assert "Must be a list" in message
        assert "record 7" in message.lower()


class TestImportRecordLimitError:
    """Tests for ImportRecordLimitError."""

    def test_import_record_limit_error_is_spatial_memory_error(self):
        """ImportRecordLimitError should extend SpatialMemoryError."""
        error = ImportRecordLimitError(
            actual_count=150000,
            max_count=100000,
        )
        assert isinstance(error, SpatialMemoryError)

    def test_import_record_limit_error_stores_fields(self):
        """ImportRecordLimitError should store count fields."""
        error = ImportRecordLimitError(
            actual_count=500000,
            max_count=100000,
        )
        assert error.actual_count == 500000
        assert error.max_count == 100000

    def test_import_record_limit_error_message(self):
        """ImportRecordLimitError should format message with counts."""
        error = ImportRecordLimitError(
            actual_count=250000,
            max_count=100000,
        )
        message = str(error)
        assert "250000" in message
        assert "100000" in message
        assert "records" in message.lower()


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_phase5_errors_extend_spatial_memory_error(self):
        """All Phase 5 errors should extend SpatialMemoryError."""
        errors = [
            ExportError("test"),
            MemoryImportError("test"),
            NamespaceOperationError("test"),
            PathSecurityError(path="/test", violation_type="test"),
            FileSizeLimitError(path="/test", actual_size_bytes=1, max_size_bytes=0),
            ImportRecordLimitError(actual_count=1, max_count=0),
        ]
        for error in errors:
            assert isinstance(error, SpatialMemoryError)

    def test_validation_subclasses_extend_validation_error(self):
        """Validation subclasses should extend ValidationError."""
        validation_errors = [
            DimensionMismatchError(expected_dim=1, actual_dim=2),
            SchemaValidationError(field="test", error="test"),
        ]
        for error in validation_errors:
            assert isinstance(error, ValidationError)
            assert isinstance(error, SpatialMemoryError)

    def test_errors_can_be_caught_by_base_class(self):
        """All Phase 5 errors should be catchable by SpatialMemoryError."""
        try:
            raise PathSecurityError(path="/test", violation_type="test")
        except SpatialMemoryError as e:
            assert isinstance(e, PathSecurityError)

        try:
            raise DimensionMismatchError(expected_dim=1, actual_dim=2)
        except ValidationError as e:
            assert isinstance(e, DimensionMismatchError)
