"""Unit tests for Phase 5 configuration options.

Tests the new config fields for:
- Export settings (allowed paths, symlinks)
- Import settings (allowed paths, symlinks, file size, records, validation)
- Destructive operation settings (confirmation threshold)
- Export/Import operational settings (formats, batch sizes, deduplication)

Following TDD: Write tests FIRST, then implement.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spatial_memory.config import Settings

# =============================================================================
# Export Settings Tests
# =============================================================================


class TestExportSettings:
    """Test export configuration options."""

    def test_export_allowed_paths_default(self) -> None:
        """Test default export allowed paths."""
        settings = Settings()
        assert settings.export_allowed_paths == ["./exports", "./backups"]

    def test_export_allowed_paths_custom(self) -> None:
        """Test custom export allowed paths."""
        settings = Settings(export_allowed_paths=["./custom/exports", "/data/backups"])
        assert settings.export_allowed_paths == ["./custom/exports", "/data/backups"]

    def test_export_allowed_paths_empty_list(self) -> None:
        """Test empty export allowed paths list."""
        settings = Settings(export_allowed_paths=[])
        assert settings.export_allowed_paths == []

    def test_export_allow_symlinks_default_false(self) -> None:
        """Test symlinks are disallowed by default for security."""
        settings = Settings()
        assert settings.export_allow_symlinks is False

    def test_export_allow_symlinks_can_enable(self) -> None:
        """Test symlinks can be explicitly enabled."""
        settings = Settings(export_allow_symlinks=True)
        assert settings.export_allow_symlinks is True


# =============================================================================
# Import Settings Tests
# =============================================================================


class TestImportSettings:
    """Test import configuration options."""

    def test_import_allowed_paths_default(self) -> None:
        """Test default import allowed paths."""
        settings = Settings()
        assert settings.import_allowed_paths == ["./imports", "./backups"]

    def test_import_allowed_paths_custom(self) -> None:
        """Test custom import allowed paths."""
        settings = Settings(import_allowed_paths=["./data/imports"])
        assert settings.import_allowed_paths == ["./data/imports"]

    def test_import_allow_symlinks_default_false(self) -> None:
        """Test symlinks are disallowed by default for security."""
        settings = Settings()
        assert settings.import_allow_symlinks is False

    def test_import_allow_symlinks_can_enable(self) -> None:
        """Test symlinks can be explicitly enabled."""
        settings = Settings(import_allow_symlinks=True)
        assert settings.import_allow_symlinks is True

    def test_import_max_file_size_mb_default(self) -> None:
        """Test default max file size is 100MB."""
        settings = Settings()
        assert settings.import_max_file_size_mb == 100.0

    def test_import_max_file_size_mb_custom(self) -> None:
        """Test custom max file size."""
        settings = Settings(import_max_file_size_mb=500.0)
        assert settings.import_max_file_size_mb == 500.0

    def test_import_max_file_size_mb_min_validation(self) -> None:
        """Test minimum file size limit of 1MB."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_max_file_size_mb=0.5)
        assert "import_max_file_size_mb" in str(exc_info.value)

    def test_import_max_file_size_mb_max_validation(self) -> None:
        """Test maximum file size limit of 1000MB."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_max_file_size_mb=1500.0)
        assert "import_max_file_size_mb" in str(exc_info.value)

    def test_import_max_file_size_mb_boundary_min(self) -> None:
        """Test boundary at minimum 1MB."""
        settings = Settings(import_max_file_size_mb=1.0)
        assert settings.import_max_file_size_mb == 1.0

    def test_import_max_file_size_mb_boundary_max(self) -> None:
        """Test boundary at maximum 1000MB."""
        settings = Settings(import_max_file_size_mb=1000.0)
        assert settings.import_max_file_size_mb == 1000.0

    def test_import_max_records_default(self) -> None:
        """Test default max records is 100,000."""
        settings = Settings()
        assert settings.import_max_records == 100_000

    def test_import_max_records_custom(self) -> None:
        """Test custom max records."""
        settings = Settings(import_max_records=500_000)
        assert settings.import_max_records == 500_000

    def test_import_max_records_min_validation(self) -> None:
        """Test minimum records limit of 1000."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_max_records=500)
        assert "import_max_records" in str(exc_info.value)

    def test_import_max_records_max_validation(self) -> None:
        """Test maximum records limit of 10,000,000."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_max_records=20_000_000)
        assert "import_max_records" in str(exc_info.value)

    def test_import_max_records_boundary_min(self) -> None:
        """Test boundary at minimum 1000 records."""
        settings = Settings(import_max_records=1000)
        assert settings.import_max_records == 1000

    def test_import_max_records_boundary_max(self) -> None:
        """Test boundary at maximum 10,000,000 records."""
        settings = Settings(import_max_records=10_000_000)
        assert settings.import_max_records == 10_000_000

    def test_import_fail_fast_default_false(self) -> None:
        """Test fail_fast is disabled by default (continue on errors)."""
        settings = Settings()
        assert settings.import_fail_fast is False

    def test_import_fail_fast_can_enable(self) -> None:
        """Test fail_fast can be enabled."""
        settings = Settings(import_fail_fast=True)
        assert settings.import_fail_fast is True

    def test_import_validate_vectors_default_true(self) -> None:
        """Test vector validation is enabled by default."""
        settings = Settings()
        assert settings.import_validate_vectors is True

    def test_import_validate_vectors_can_disable(self) -> None:
        """Test vector validation can be disabled."""
        settings = Settings(import_validate_vectors=False)
        assert settings.import_validate_vectors is False


# =============================================================================
# Destructive Operation Settings Tests
# =============================================================================


class TestDestructiveOperationSettings:
    """Test destructive operation configuration options."""

    def test_destructive_confirm_threshold_default(self) -> None:
        """Test default confirmation threshold is 100 records."""
        settings = Settings()
        assert settings.destructive_confirm_threshold == 100

    def test_destructive_confirm_threshold_custom(self) -> None:
        """Test custom confirmation threshold."""
        settings = Settings(destructive_confirm_threshold=50)
        assert settings.destructive_confirm_threshold == 50

    def test_destructive_confirm_threshold_min_validation(self) -> None:
        """Test minimum confirmation threshold of 1."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(destructive_confirm_threshold=0)
        assert "destructive_confirm_threshold" in str(exc_info.value)

    def test_destructive_confirm_threshold_boundary_min(self) -> None:
        """Test boundary at minimum 1 record."""
        settings = Settings(destructive_confirm_threshold=1)
        assert settings.destructive_confirm_threshold == 1

    def test_destructive_require_namespace_confirmation_default_true(self) -> None:
        """Test namespace confirmation is required by default."""
        settings = Settings()
        assert settings.destructive_require_namespace_confirmation is True

    def test_destructive_require_namespace_confirmation_can_disable(self) -> None:
        """Test namespace confirmation can be disabled."""
        settings = Settings(destructive_require_namespace_confirmation=False)
        assert settings.destructive_require_namespace_confirmation is False


# =============================================================================
# Export/Import Operational Settings Tests
# =============================================================================


class TestExportImportOperationalSettings:
    """Test export/import operational configuration options."""

    def test_export_default_format_default_parquet(self) -> None:
        """Test default export format is parquet."""
        settings = Settings()
        assert settings.export_default_format == "parquet"

    def test_export_default_format_json(self) -> None:
        """Test JSON export format."""
        settings = Settings(export_default_format="json")
        assert settings.export_default_format == "json"

    def test_export_default_format_csv(self) -> None:
        """Test CSV export format."""
        settings = Settings(export_default_format="csv")
        assert settings.export_default_format == "csv"

    def test_export_batch_size_default(self) -> None:
        """Test default export batch size is 5000."""
        settings = Settings()
        assert settings.export_batch_size == 5000

    def test_export_batch_size_custom(self) -> None:
        """Test custom export batch size."""
        settings = Settings(export_batch_size=10000)
        assert settings.export_batch_size == 10000

    def test_export_batch_size_min_validation(self) -> None:
        """Test minimum export batch size of 100."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(export_batch_size=50)
        assert "export_batch_size" in str(exc_info.value)

    def test_export_batch_size_boundary_min(self) -> None:
        """Test boundary at minimum 100."""
        settings = Settings(export_batch_size=100)
        assert settings.export_batch_size == 100

    def test_import_batch_size_default(self) -> None:
        """Test default import batch size is 1000."""
        settings = Settings()
        assert settings.import_batch_size == 1000

    def test_import_batch_size_custom(self) -> None:
        """Test custom import batch size."""
        settings = Settings(import_batch_size=2000)
        assert settings.import_batch_size == 2000

    def test_import_batch_size_min_validation(self) -> None:
        """Test minimum import batch size of 100."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_batch_size=50)
        assert "import_batch_size" in str(exc_info.value)

    def test_import_batch_size_boundary_min(self) -> None:
        """Test boundary at minimum 100."""
        settings = Settings(import_batch_size=100)
        assert settings.import_batch_size == 100

    def test_import_deduplicate_default_false(self) -> None:
        """Test deduplication is disabled by default."""
        settings = Settings()
        assert settings.import_deduplicate_default is False

    def test_import_deduplicate_can_enable(self) -> None:
        """Test deduplication can be enabled."""
        settings = Settings(import_deduplicate_default=True)
        assert settings.import_deduplicate_default is True

    def test_import_dedup_threshold_default(self) -> None:
        """Test default deduplication threshold is 0.95."""
        settings = Settings()
        assert settings.import_dedup_threshold == 0.95

    def test_import_dedup_threshold_custom(self) -> None:
        """Test custom deduplication threshold."""
        settings = Settings(import_dedup_threshold=0.85)
        assert settings.import_dedup_threshold == 0.85

    def test_import_dedup_threshold_min_validation(self) -> None:
        """Test minimum deduplication threshold of 0.7."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_dedup_threshold=0.5)
        assert "import_dedup_threshold" in str(exc_info.value)

    def test_import_dedup_threshold_max_validation(self) -> None:
        """Test maximum deduplication threshold of 0.99."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(import_dedup_threshold=1.0)
        assert "import_dedup_threshold" in str(exc_info.value)

    def test_import_dedup_threshold_boundary_min(self) -> None:
        """Test boundary at minimum 0.7."""
        settings = Settings(import_dedup_threshold=0.7)
        assert settings.import_dedup_threshold == 0.7

    def test_import_dedup_threshold_boundary_max(self) -> None:
        """Test boundary at maximum 0.99."""
        settings = Settings(import_dedup_threshold=0.99)
        assert settings.import_dedup_threshold == 0.99


# =============================================================================
# Environment Variable Tests
# =============================================================================


class TestEnvironmentVariables:
    """Test that Phase 5 config options can be set via environment variables."""

    def test_export_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test export settings from environment variables."""
        monkeypatch.setenv("SPATIAL_MEMORY_EXPORT_ALLOW_SYMLINKS", "true")
        settings = Settings()
        assert settings.export_allow_symlinks is True

    def test_import_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test import settings from environment variables."""
        monkeypatch.setenv("SPATIAL_MEMORY_IMPORT_MAX_FILE_SIZE_MB", "250.0")
        monkeypatch.setenv("SPATIAL_MEMORY_IMPORT_MAX_RECORDS", "50000")
        settings = Settings()
        assert settings.import_max_file_size_mb == 250.0
        assert settings.import_max_records == 50000

    def test_destructive_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test destructive operation settings from environment variables."""
        monkeypatch.setenv("SPATIAL_MEMORY_DESTRUCTIVE_CONFIRM_THRESHOLD", "200")
        settings = Settings()
        assert settings.destructive_confirm_threshold == 200

    def test_operational_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test operational settings from environment variables."""
        monkeypatch.setenv("SPATIAL_MEMORY_EXPORT_DEFAULT_FORMAT", "json")
        monkeypatch.setenv("SPATIAL_MEMORY_EXPORT_BATCH_SIZE", "3000")
        monkeypatch.setenv("SPATIAL_MEMORY_IMPORT_BATCH_SIZE", "2500")
        monkeypatch.setenv("SPATIAL_MEMORY_IMPORT_DEDUP_THRESHOLD", "0.90")
        settings = Settings()
        assert settings.export_default_format == "json"
        assert settings.export_batch_size == 3000
        assert settings.import_batch_size == 2500
        assert settings.import_dedup_threshold == 0.90


# =============================================================================
# Field Description Tests
# =============================================================================


class TestFieldDescriptions:
    """Test that all Phase 5 config fields have proper descriptions."""

    def test_export_fields_have_descriptions(self) -> None:
        """Test export fields have descriptions."""
        schema = Settings.model_json_schema()
        props = schema["properties"]

        assert "description" in props["export_allowed_paths"]
        assert "description" in props["export_allow_symlinks"]

    def test_import_fields_have_descriptions(self) -> None:
        """Test import fields have descriptions."""
        schema = Settings.model_json_schema()
        props = schema["properties"]

        assert "description" in props["import_allowed_paths"]
        assert "description" in props["import_allow_symlinks"]
        assert "description" in props["import_max_file_size_mb"]
        assert "description" in props["import_max_records"]
        assert "description" in props["import_fail_fast"]
        assert "description" in props["import_validate_vectors"]

    def test_destructive_fields_have_descriptions(self) -> None:
        """Test destructive operation fields have descriptions."""
        schema = Settings.model_json_schema()
        props = schema["properties"]

        assert "description" in props["destructive_confirm_threshold"]
        assert "description" in props["destructive_require_namespace_confirmation"]

    def test_operational_fields_have_descriptions(self) -> None:
        """Test operational fields have descriptions."""
        schema = Settings.model_json_schema()
        props = schema["properties"]

        assert "description" in props["export_default_format"]
        assert "description" in props["export_batch_size"]
        assert "description" in props["import_batch_size"]
        assert "description" in props["import_deduplicate_default"]
        assert "description" in props["import_dedup_threshold"]
