"""Unit tests for the consolidated security facade module.

These tests verify the FileSecurityManager and convenience functions
provide a proper interface to the underlying security validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatial_memory.core.errors import FileSizeLimitError, PathSecurityError
from spatial_memory.core.security import (
    DEFAULT_SECURITY_CONFIG,
    FileSecurityManager,
    SecurityConfig,
    validate_export_path,
    validate_import_file,
    validate_import_path,
    validate_import_records,
)

if TYPE_CHECKING:
    pass


MB = 1024 * 1024  # 1 megabyte in bytes


# =============================================================================
# SecurityConfig Tests
# =============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Default config should have sensible security values."""
        config = SecurityConfig()

        assert "./exports" in config.export_allowed_paths
        assert "./backups" in config.export_allowed_paths
        assert "./imports" in config.import_allowed_paths
        assert config.max_import_size_mb == 100.0
        assert config.allow_symlinks is False
        assert config.max_import_records == 100_000

    def test_config_immutability(self) -> None:
        """SecurityConfig should be immutable after creation."""
        config = SecurityConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.max_import_size_mb = 200.0  # type: ignore

    def test_config_validation_rejects_invalid_size(self) -> None:
        """Config should reject invalid max_import_size_mb."""
        with pytest.raises(ValueError, match="must be positive"):
            SecurityConfig(max_import_size_mb=0)

        with pytest.raises(ValueError, match="must be positive"):
            SecurityConfig(max_import_size_mb=-100)

        with pytest.raises(ValueError, match="cannot exceed"):
            SecurityConfig(max_import_size_mb=20_000)

    def test_config_max_size_bytes_conversion(self) -> None:
        """Config should correctly convert MB to bytes."""
        config = SecurityConfig(max_import_size_mb=50.0)
        assert config.max_import_size_bytes == 50 * MB

    def test_config_with_custom_paths(self) -> None:
        """Config should accept custom allowed paths."""
        config = SecurityConfig(
            export_allowed_paths=("/data/exports",),
            import_allowed_paths=("/data/imports", "/data/shared"),
        )

        assert config.export_allowed_paths == ("/data/exports",)
        assert config.import_allowed_paths == ("/data/imports", "/data/shared")


# =============================================================================
# FileSecurityManager Tests
# =============================================================================


class TestFileSecurityManager:
    """Tests for FileSecurityManager class."""

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> dict[str, Path]:
        """Create temporary directories for testing."""
        exports = tmp_path / "exports"
        imports = tmp_path / "imports"
        backups = tmp_path / "backups"

        exports.mkdir()
        imports.mkdir()
        backups.mkdir()

        return {
            "root": tmp_path,
            "exports": exports,
            "imports": imports,
            "backups": backups,
        }

    @pytest.fixture
    def manager(self, temp_dirs: dict[str, Path]) -> FileSecurityManager:
        """Create a manager with temp directories as allowed paths."""
        config = SecurityConfig(
            export_allowed_paths=(str(temp_dirs["exports"]), str(temp_dirs["backups"])),
            import_allowed_paths=(str(temp_dirs["imports"]), str(temp_dirs["backups"])),
            max_import_size_mb=10.0,
            allow_symlinks=False,
        )
        return FileSecurityManager(config)

    def test_manager_uses_default_config(self) -> None:
        """Manager should use default config when none provided."""
        manager = FileSecurityManager()
        assert manager.config == DEFAULT_SECURITY_CONFIG

    def test_manager_uses_custom_config(self) -> None:
        """Manager should use provided config."""
        config = SecurityConfig(max_import_size_mb=50.0)
        manager = FileSecurityManager(config)
        assert manager.config.max_import_size_mb == 50.0

    # Export path validation tests

    def test_validate_export_path_accepts_valid_path(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Valid export paths should be accepted."""
        valid_path = temp_dirs["exports"] / "backup.parquet"
        result = manager.validate_export_path(valid_path)
        assert result is not None
        assert result.suffix == ".parquet"

    def test_validate_export_path_rejects_traversal(
        self,
        manager: FileSecurityManager,
    ) -> None:
        """Path traversal attempts should be rejected."""
        with pytest.raises(PathSecurityError) as exc_info:
            manager.validate_export_path("../../../etc/passwd")
        assert exc_info.value.violation_type in ("traversal_attempt", "path_outside_allowlist")

    def test_validate_export_path_rejects_invalid_extension(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Invalid file extensions should be rejected."""
        bad_path = temp_dirs["exports"] / "script.exe"
        with pytest.raises(PathSecurityError) as exc_info:
            manager.validate_export_path(bad_path)
        assert exc_info.value.violation_type == "invalid_extension"

    def test_validate_export_path_rejects_outside_allowlist(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Paths outside allowed directories should be rejected."""
        outside_path = temp_dirs["root"] / "forbidden" / "data.parquet"
        with pytest.raises(PathSecurityError) as exc_info:
            manager.validate_export_path(outside_path)
        assert exc_info.value.violation_type in ("path_outside_allowlist", "sensitive_directory")

    # Import path validation tests

    def test_validate_import_path_accepts_valid_file(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Valid import paths with existing files should be accepted."""
        valid_file = temp_dirs["imports"] / "data.json"
        valid_file.write_text('{"test": true}')

        result = manager.validate_import_path(valid_file)
        assert result.exists()
        assert result.suffix == ".json"

    def test_validate_import_path_rejects_nonexistent(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Non-existent files should be rejected for imports."""
        missing_file = temp_dirs["imports"] / "missing.json"
        with pytest.raises(PathSecurityError) as exc_info:
            manager.validate_import_path(missing_file)
        assert exc_info.value.violation_type == "file_not_found"

    def test_validate_import_path_rejects_oversized_file(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Oversized files should be rejected."""
        large_file = temp_dirs["imports"] / "large.json"
        # Manager has 10MB limit, create 11MB file
        large_file.write_bytes(b"x" * (11 * MB))

        with pytest.raises(FileSizeLimitError) as exc_info:
            manager.validate_import_path(large_file)
        assert exc_info.value.actual_size_bytes > exc_info.value.max_size_bytes

    def test_validate_import_path_rejects_directory(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """Directories should be rejected for imports."""
        subdir = temp_dirs["imports"] / "subdir"
        subdir.mkdir()

        with pytest.raises(PathSecurityError) as exc_info:
            manager.validate_import_path(subdir)
        assert exc_info.value.violation_type in ("not_a_file", "invalid_extension")

    # TOCTOU-safe import tests

    def test_validate_and_open_import_returns_handle(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """validate_and_open_import should return open file handle."""
        test_file = temp_dirs["imports"] / "test.json"
        test_file.write_text('{"content": "test data"}')

        path, handle = manager.validate_and_open_import(test_file)
        try:
            content = handle.read()
            assert b"test data" in content
        finally:
            handle.close()

    def test_validate_and_open_import_handle_is_binary(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """File handle should be opened in binary mode."""
        test_file = temp_dirs["imports"] / "binary.json"
        test_file.write_bytes(b'{"binary": true}')

        path, handle = manager.validate_and_open_import(test_file)
        try:
            assert "b" in handle.mode
        finally:
            handle.close()

    # Boolean check method tests

    def test_is_path_safe_returns_true_for_valid(
        self,
        manager: FileSecurityManager,
        temp_dirs: dict[str, Path],
    ) -> None:
        """is_path_safe should return True for valid paths."""
        valid_export = temp_dirs["exports"] / "data.parquet"
        assert manager.is_path_safe(valid_export, "export") is True

        valid_import = temp_dirs["imports"] / "data.json"
        valid_import.write_text("{}")
        assert manager.is_path_safe(valid_import, "import") is True

    def test_is_path_safe_returns_false_for_invalid(
        self,
        manager: FileSecurityManager,
    ) -> None:
        """is_path_safe should return False for invalid paths."""
        assert manager.is_path_safe("../../../etc/passwd", "export") is False
        assert manager.is_path_safe("/nonexistent/path.json", "import") is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> dict[str, Path]:
        """Create temporary directories for testing."""
        exports = tmp_path / "exports"
        imports = tmp_path / "imports"

        exports.mkdir()
        imports.mkdir()

        return {
            "root": tmp_path,
            "exports": exports,
            "imports": imports,
        }

    @pytest.fixture
    def config(self, temp_dirs: dict[str, Path]) -> SecurityConfig:
        """Create a config for temp directories."""
        return SecurityConfig(
            export_allowed_paths=(str(temp_dirs["exports"]),),
            import_allowed_paths=(str(temp_dirs["imports"]),),
        )

    def test_validate_export_path_function(
        self,
        config: SecurityConfig,
        temp_dirs: dict[str, Path],
    ) -> None:
        """validate_export_path function should work."""
        path = temp_dirs["exports"] / "test.parquet"
        result = validate_export_path(path, config)
        assert result is not None

    def test_validate_import_path_function(
        self,
        config: SecurityConfig,
        temp_dirs: dict[str, Path],
    ) -> None:
        """validate_import_path function should work."""
        path = temp_dirs["imports"] / "test.json"
        path.write_text("{}")
        result = validate_import_path(path, config)
        assert result.exists()

    def test_validate_import_file_function(
        self,
        config: SecurityConfig,
        temp_dirs: dict[str, Path],
    ) -> None:
        """validate_import_file function should work."""
        path = temp_dirs["imports"] / "test.json"
        path.write_text('{"data": "test"}')

        canonical, handle = validate_import_file(path, config)
        try:
            content = handle.read()
            assert b"test" in content
        finally:
            handle.close()


# =============================================================================
# Import Record Validation Tests
# =============================================================================


class TestImportRecordValidation:
    """Tests for import record validation."""

    def test_validate_import_records_valid_batch(self) -> None:
        """Valid records should pass validation."""
        records = [
            {"content": "Memory 1", "namespace": "test"},
            {"content": "Memory 2", "importance": 0.8},
            {"content": "Memory 3", "tags": ["tag1", "tag2"]},
        ]

        result = validate_import_records(records, expected_vector_dim=384)

        assert result.is_valid
        assert result.valid_count == 3
        assert result.invalid_count == 0
        assert len(result.errors) == 0

    def test_validate_import_records_missing_content(self) -> None:
        """Records missing content should fail validation."""
        records = [
            {"namespace": "test"},  # Missing content
            {"content": "Valid memory"},
        ]

        result = validate_import_records(records, expected_vector_dim=384)

        assert not result.is_valid
        assert result.valid_count == 1
        assert result.invalid_count == 1

    def test_validate_import_records_wrong_vector_dim(self) -> None:
        """Records with wrong vector dimensions should fail."""
        records = [
            {
                "content": "Memory with wrong vector",
                "vector": [0.1] * 256,  # Wrong dimension
            },
        ]

        result = validate_import_records(records, expected_vector_dim=384)

        assert not result.is_valid
        assert result.invalid_count == 1
        error_msg = str(result.errors[0][1])
        assert "dimension" in error_msg.lower()

    def test_validate_import_records_invalid_importance(self) -> None:
        """Records with invalid importance should fail."""
        records = [
            {"content": "Memory 1", "importance": 1.5},  # > 1.0
            {"content": "Memory 2", "importance": -0.1},  # < 0.0
        ]

        result = validate_import_records(records, expected_vector_dim=384)

        assert not result.is_valid
        assert result.invalid_count == 2

    def test_validate_import_records_fail_fast(self) -> None:
        """fail_fast should stop on first error."""
        records = [
            {},  # Missing content
            {},  # Also missing content
            {},  # Also missing content
        ]

        result = validate_import_records(
            records, expected_vector_dim=384, fail_fast=True
        )

        # Should stop after first error
        assert result.invalid_count == 1
        assert len(result.errors) == 1


# =============================================================================
# Integration with Settings Tests
# =============================================================================


class TestSettingsIntegration:
    """Tests for integration with application Settings."""

    def test_create_security_manager_from_settings(self) -> None:
        """Factory function should create manager from settings."""
        from spatial_memory.core.security import create_security_manager_from_settings

        # This will use default settings from environment/defaults
        manager = create_security_manager_from_settings()

        assert manager is not None
        assert manager.config is not None
        # Config should have values from settings
        assert len(manager.config.export_allowed_paths) > 0
        assert len(manager.config.import_allowed_paths) > 0
