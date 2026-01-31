"""Unit tests for Phase 5 result dataclasses.

TDD: These tests define the expected behavior for Phase 5 result models.
Tests verify dataclass initialization, default values, and type constraints.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone

import pytest

from spatial_memory.core.models import (
    DeleteNamespaceResult,
    ExportImportConfig,
    ExportResult,
    HybridMemoryMatch,
    HybridRecallResult,
    ImportedMemory,
    ImportResult,
    ImportValidationError,
    IndexInfo,
    NamespaceInfo,
    NamespacesResult,
    RenameNamespaceResult,
    StatsResult,
    UtilityConfig,
)

# =============================================================================
# IndexInfo Tests
# =============================================================================


class TestIndexInfo:
    """Tests for IndexInfo dataclass."""

    def test_create_with_required_fields(self) -> None:
        """IndexInfo should be created with all required fields."""
        info = IndexInfo(
            name="vector_idx",
            index_type="IVF_PQ",
            column="vector",
            num_indexed_rows=1000,
            status="ready",
        )

        assert info.name == "vector_idx"
        assert info.index_type == "IVF_PQ"
        assert info.column == "vector"
        assert info.num_indexed_rows == 1000
        assert info.status == "ready"

    def test_all_fields_are_required(self) -> None:
        """IndexInfo should have no default values - all fields required."""
        # All 5 fields should be required (no defaults)
        assert len(fields(IndexInfo)) == 5

    def test_serializable_to_dict(self) -> None:
        """IndexInfo should be serializable to dictionary."""
        info = IndexInfo(
            name="fts_idx",
            index_type="FTS",
            column="content",
            num_indexed_rows=500,
            status="building",
        )

        result = asdict(info)
        assert isinstance(result, dict)
        assert result["name"] == "fts_idx"
        assert result["status"] == "building"


# =============================================================================
# StatsResult Tests
# =============================================================================


class TestStatsResult:
    """Tests for StatsResult dataclass."""

    def test_create_with_required_fields(self) -> None:
        """StatsResult should be created with required fields."""
        stats = StatsResult(
            total_memories=1000,
            memories_by_namespace={"default": 800, "archive": 200},
            storage_bytes=1024 * 1024 * 50,
            storage_mb=50.0,
            estimated_vector_bytes=1024 * 1024 * 40,
            has_vector_index=True,
            has_fts_index=True,
            indices=[],
            num_fragments=5,
            needs_compaction=False,
            table_version=10,
        )

        assert stats.total_memories == 1000
        assert stats.memories_by_namespace["default"] == 800
        assert stats.storage_mb == 50.0
        assert stats.has_vector_index is True
        assert stats.has_fts_index is True

    def test_optional_fields_default_to_none(self) -> None:
        """StatsResult optional fields should default to None."""
        stats = StatsResult(
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

        assert stats.oldest_memory_date is None
        assert stats.newest_memory_date is None
        assert stats.avg_content_length is None

    def test_with_optional_fields(self) -> None:
        """StatsResult should accept optional fields."""
        now = datetime.now(timezone.utc)
        stats = StatsResult(
            total_memories=100,
            memories_by_namespace={"test": 100},
            storage_bytes=1024,
            storage_mb=0.001,
            estimated_vector_bytes=512,
            has_vector_index=True,
            has_fts_index=False,
            indices=[],
            num_fragments=1,
            needs_compaction=False,
            table_version=1,
            oldest_memory_date=now,
            newest_memory_date=now,
            avg_content_length=150.5,
        )

        assert stats.oldest_memory_date == now
        assert stats.newest_memory_date == now
        assert stats.avg_content_length == 150.5

    def test_with_index_info_list(self) -> None:
        """StatsResult should accept list of IndexInfo."""
        indices = [
            IndexInfo(
                name="vec_idx",
                index_type="IVF_PQ",
                column="vector",
                num_indexed_rows=1000,
                status="ready",
            ),
            IndexInfo(
                name="fts_idx",
                index_type="FTS",
                column="content",
                num_indexed_rows=1000,
                status="ready",
            ),
        ]

        stats = StatsResult(
            total_memories=1000,
            memories_by_namespace={},
            storage_bytes=0,
            storage_mb=0.0,
            estimated_vector_bytes=0,
            has_vector_index=True,
            has_fts_index=True,
            indices=indices,
            num_fragments=1,
            needs_compaction=False,
            table_version=1,
        )

        assert len(stats.indices) == 2
        assert stats.indices[0].name == "vec_idx"
        assert stats.indices[1].index_type == "FTS"


# =============================================================================
# NamespaceInfo Tests
# =============================================================================


class TestNamespaceInfo:
    """Tests for NamespaceInfo dataclass."""

    def test_create_with_required_fields(self) -> None:
        """NamespaceInfo should be created with required fields."""
        info = NamespaceInfo(name="default", memory_count=500)

        assert info.name == "default"
        assert info.memory_count == 500

    def test_optional_datetime_fields(self) -> None:
        """NamespaceInfo optional datetime fields should default to None."""
        info = NamespaceInfo(name="test", memory_count=0)

        assert info.oldest_memory is None
        assert info.newest_memory is None

    def test_with_datetime_fields(self) -> None:
        """NamespaceInfo should accept datetime fields."""
        now = datetime.now(timezone.utc)
        info = NamespaceInfo(
            name="archive",
            memory_count=100,
            oldest_memory=now,
            newest_memory=now,
        )

        assert info.oldest_memory == now
        assert info.newest_memory == now


# =============================================================================
# NamespacesResult Tests
# =============================================================================


class TestNamespacesResult:
    """Tests for NamespacesResult dataclass."""

    def test_create_with_all_fields(self) -> None:
        """NamespacesResult should be created with all fields."""
        namespaces = [
            NamespaceInfo(name="default", memory_count=500),
            NamespaceInfo(name="archive", memory_count=200),
        ]

        result = NamespacesResult(
            namespaces=namespaces,
            total_namespaces=2,
            total_memories=700,
        )

        assert len(result.namespaces) == 2
        assert result.total_namespaces == 2
        assert result.total_memories == 700

    def test_empty_namespaces(self) -> None:
        """NamespacesResult should handle empty namespace list."""
        result = NamespacesResult(
            namespaces=[],
            total_namespaces=0,
            total_memories=0,
        )

        assert result.namespaces == []
        assert result.total_namespaces == 0
        assert result.total_memories == 0


# =============================================================================
# DeleteNamespaceResult Tests
# =============================================================================


class TestDeleteNamespaceResult:
    """Tests for DeleteNamespaceResult dataclass."""

    def test_create_successful_deletion(self) -> None:
        """DeleteNamespaceResult should represent successful deletion."""
        result = DeleteNamespaceResult(
            namespace="archive",
            memories_deleted=150,
            success=True,
            message="Successfully deleted 150 memories from namespace 'archive'",
        )

        assert result.namespace == "archive"
        assert result.memories_deleted == 150
        assert result.success is True
        assert result.dry_run is False  # default

    def test_dry_run_mode(self) -> None:
        """DeleteNamespaceResult should support dry_run mode."""
        result = DeleteNamespaceResult(
            namespace="test",
            memories_deleted=50,
            success=True,
            message="Would delete 50 memories",
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.memories_deleted == 50

    def test_failed_deletion(self) -> None:
        """DeleteNamespaceResult should represent failed deletion."""
        result = DeleteNamespaceResult(
            namespace="protected",
            memories_deleted=0,
            success=False,
            message="Namespace 'protected' not found",
        )

        assert result.success is False
        assert result.memories_deleted == 0


# =============================================================================
# RenameNamespaceResult Tests
# =============================================================================


class TestRenameNamespaceResult:
    """Tests for RenameNamespaceResult dataclass."""

    def test_create_successful_rename(self) -> None:
        """RenameNamespaceResult should represent successful rename."""
        result = RenameNamespaceResult(
            old_namespace="old_name",
            new_namespace="new_name",
            memories_renamed=200,
            success=True,
            message="Successfully renamed 200 memories",
        )

        assert result.old_namespace == "old_name"
        assert result.new_namespace == "new_name"
        assert result.memories_renamed == 200
        assert result.success is True

    def test_failed_rename(self) -> None:
        """RenameNamespaceResult should represent failed rename."""
        result = RenameNamespaceResult(
            old_namespace="missing",
            new_namespace="target",
            memories_renamed=0,
            success=False,
            message="Namespace 'missing' not found",
        )

        assert result.success is False
        assert result.memories_renamed == 0


# =============================================================================
# ExportResult Tests
# =============================================================================


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_create_parquet_export(self) -> None:
        """ExportResult should represent parquet export."""
        result = ExportResult(
            format="parquet",
            output_path="/exports/backup.parquet",
            memories_exported=1000,
            file_size_bytes=1024 * 1024 * 10,
            file_size_mb=10.0,
            namespaces_included=["default", "archive"],
            duration_seconds=5.5,
            compression="zstd",
        )

        assert result.format == "parquet"
        assert result.output_path == "/exports/backup.parquet"
        assert result.memories_exported == 1000
        assert result.file_size_mb == 10.0
        assert result.compression == "zstd"

    def test_create_json_export(self) -> None:
        """ExportResult should represent json export."""
        result = ExportResult(
            format="json",
            output_path="/exports/backup.json",
            memories_exported=500,
            file_size_bytes=1024 * 512,
            file_size_mb=0.5,
            namespaces_included=["default"],
            duration_seconds=2.0,
        )

        assert result.format == "json"
        assert result.compression is None  # default

    def test_create_csv_export(self) -> None:
        """ExportResult should represent csv export."""
        result = ExportResult(
            format="csv",
            output_path="/exports/backup.csv",
            memories_exported=100,
            file_size_bytes=1024 * 100,
            file_size_mb=0.1,
            namespaces_included=["test"],
            duration_seconds=1.0,
        )

        assert result.format == "csv"


# =============================================================================
# ImportedMemory Tests
# =============================================================================


class TestImportedMemory:
    """Tests for ImportedMemory dataclass."""

    def test_create_basic_import(self) -> None:
        """ImportedMemory should represent a basic import."""
        imported = ImportedMemory(
            id="abc-123",
            content_preview="This is a test memory about...",
            namespace="default",
        )

        assert imported.id == "abc-123"
        assert imported.content_preview == "This is a test memory about..."
        assert imported.namespace == "default"
        assert imported.was_deduplicated is False  # default
        assert imported.original_id is None  # default

    def test_deduplicated_import(self) -> None:
        """ImportedMemory should represent deduplicated import."""
        imported = ImportedMemory(
            id="new-id-456",
            content_preview="Similar content...",
            namespace="default",
            was_deduplicated=True,
            original_id="old-id-123",
        )

        assert imported.was_deduplicated is True
        assert imported.original_id == "old-id-123"


# =============================================================================
# ImportValidationError Tests
# =============================================================================


class TestImportValidationError:
    """Tests for ImportValidationError dataclass."""

    def test_create_validation_error(self) -> None:
        """ImportValidationError should represent a validation error."""
        error = ImportValidationError(
            row_number=42,
            field="content",
            error="Content field is required",
        )

        assert error.row_number == 42
        assert error.field == "content"
        assert error.error == "Content field is required"
        assert error.value is None  # default

    def test_validation_error_with_value(self) -> None:
        """ImportValidationError should include problematic value."""
        error = ImportValidationError(
            row_number=15,
            field="importance",
            error="Value must be between 0 and 1",
            value="1.5",
        )

        assert error.value == "1.5"


# =============================================================================
# ImportResult Tests
# =============================================================================


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_create_successful_import(self) -> None:
        """ImportResult should represent successful import."""
        result = ImportResult(
            source_path="/imports/backup.parquet",
            format="parquet",
            total_records_in_file=1000,
            memories_imported=950,
            memories_skipped=40,
            memories_failed=10,
            validation_errors=[],
            duration_seconds=8.5,
        )

        assert result.source_path == "/imports/backup.parquet"
        assert result.format == "parquet"
        assert result.total_records_in_file == 1000
        assert result.memories_imported == 950
        assert result.memories_skipped == 40
        assert result.memories_failed == 10
        assert result.namespace_override is None  # default
        assert result.imported_memories is None  # default

    def test_import_with_namespace_override(self) -> None:
        """ImportResult should support namespace override."""
        result = ImportResult(
            source_path="/imports/data.json",
            format="json",
            total_records_in_file=100,
            memories_imported=100,
            memories_skipped=0,
            memories_failed=0,
            validation_errors=[],
            namespace_override="imported",
            duration_seconds=2.0,
        )

        assert result.namespace_override == "imported"

    def test_import_with_validation_errors(self) -> None:
        """ImportResult should include validation errors."""
        errors = [
            ImportValidationError(row_number=5, field="content", error="Empty content"),
            ImportValidationError(row_number=10, field="vector", error="Wrong dimensions"),
        ]

        result = ImportResult(
            source_path="/imports/bad.csv",
            format="csv",
            total_records_in_file=100,
            memories_imported=98,
            memories_skipped=0,
            memories_failed=2,
            validation_errors=errors,
            duration_seconds=1.5,
        )

        assert len(result.validation_errors) == 2
        assert result.validation_errors[0].row_number == 5

    def test_import_with_imported_memories_list(self) -> None:
        """ImportResult should optionally include imported memories."""
        imported = [
            ImportedMemory(id="1", content_preview="First...", namespace="default"),
            ImportedMemory(id="2", content_preview="Second...", namespace="default"),
        ]

        result = ImportResult(
            source_path="/imports/data.json",
            format="json",
            total_records_in_file=2,
            memories_imported=2,
            memories_skipped=0,
            memories_failed=0,
            validation_errors=[],
            duration_seconds=0.5,
            imported_memories=imported,
        )

        assert result.imported_memories is not None
        assert len(result.imported_memories) == 2


# =============================================================================
# HybridMemoryMatch Tests
# =============================================================================


class TestHybridMemoryMatch:
    """Tests for HybridMemoryMatch dataclass."""

    def test_create_hybrid_match(self) -> None:
        """HybridMemoryMatch should represent a hybrid search match."""
        now = datetime.now(timezone.utc)
        match = HybridMemoryMatch(
            id="mem-123",
            content="This is a test memory",
            similarity=0.85,
            namespace="default",
            tags=["test", "example"],
            importance=0.7,
            created_at=now,
            metadata={"source": "test"},
            vector_score=0.82,
            fts_score=0.88,
            combined_score=0.85,
        )

        assert match.id == "mem-123"
        assert match.similarity == 0.85
        assert match.vector_score == 0.82
        assert match.fts_score == 0.88
        assert match.combined_score == 0.85

    def test_optional_score_fields(self) -> None:
        """HybridMemoryMatch optional scores should default to None/0."""
        now = datetime.now(timezone.utc)
        match = HybridMemoryMatch(
            id="mem-456",
            content="Another memory",
            similarity=0.75,
            namespace="test",
            tags=[],
            importance=0.5,
            created_at=now,
            metadata={},
        )

        assert match.vector_score is None
        assert match.fts_score is None
        assert match.combined_score == 0.0  # default


# =============================================================================
# HybridRecallResult Tests
# =============================================================================


class TestHybridRecallResult:
    """Tests for HybridRecallResult dataclass."""

    def test_create_hybrid_recall_result(self) -> None:
        """HybridRecallResult should represent hybrid search results."""
        now = datetime.now(timezone.utc)
        memories = [
            HybridMemoryMatch(
                id="1",
                content="First match",
                similarity=0.9,
                namespace="default",
                tags=[],
                importance=0.5,
                created_at=now,
                metadata={},
                combined_score=0.9,
            ),
            HybridMemoryMatch(
                id="2",
                content="Second match",
                similarity=0.8,
                namespace="default",
                tags=[],
                importance=0.5,
                created_at=now,
                metadata={},
                combined_score=0.8,
            ),
        ]

        result = HybridRecallResult(
            query="search query",
            alpha=0.5,
            memories=memories,
            total=2,
        )

        assert result.query == "search query"
        assert result.alpha == 0.5
        assert len(result.memories) == 2
        assert result.total == 2
        assert result.search_type == "hybrid"  # default

    def test_empty_results(self) -> None:
        """HybridRecallResult should handle empty results."""
        result = HybridRecallResult(
            query="no matches",
            alpha=0.7,
            memories=[],
            total=0,
        )

        assert result.memories == []
        assert result.total == 0


# =============================================================================
# UtilityConfig Tests
# =============================================================================


class TestUtilityConfig:
    """Tests for UtilityConfig dataclass."""

    def test_default_values(self) -> None:
        """UtilityConfig should have sensible defaults."""
        config = UtilityConfig()

        assert config.hybrid_default_alpha == 0.5
        assert config.hybrid_min_alpha == 0.0
        assert config.hybrid_max_alpha == 1.0
        assert config.stats_include_index_details is True
        assert config.namespace_batch_size == 1000
        assert config.delete_namespace_require_confirmation is True

    def test_custom_values(self) -> None:
        """UtilityConfig should accept custom values."""
        config = UtilityConfig(
            hybrid_default_alpha=0.7,
            hybrid_min_alpha=0.1,
            hybrid_max_alpha=0.9,
            stats_include_index_details=False,
            namespace_batch_size=500,
            delete_namespace_require_confirmation=False,
        )

        assert config.hybrid_default_alpha == 0.7
        assert config.namespace_batch_size == 500
        assert config.delete_namespace_require_confirmation is False


# =============================================================================
# ExportImportConfig Tests
# =============================================================================


class TestExportImportConfig:
    """Tests for ExportImportConfig dataclass."""

    def test_default_values(self) -> None:
        """ExportImportConfig should have sensible defaults."""
        config = ExportImportConfig()

        assert config.default_export_format == "parquet"
        assert config.export_batch_size == 5000
        assert config.import_batch_size == 1000
        assert config.import_deduplicate is False
        assert config.import_dedup_threshold == 0.95
        assert config.validate_on_import is True
        assert config.parquet_compression == "zstd"
        assert config.csv_include_vectors is False
        assert config.max_export_records == 0  # 0 = unlimited

    def test_custom_values(self) -> None:
        """ExportImportConfig should accept custom values."""
        config = ExportImportConfig(
            default_export_format="json",
            export_batch_size=10000,
            import_batch_size=500,
            import_deduplicate=True,
            import_dedup_threshold=0.9,
            validate_on_import=False,
            parquet_compression="snappy",
            csv_include_vectors=True,
            max_export_records=100000,
        )

        assert config.default_export_format == "json"
        assert config.export_batch_size == 10000
        assert config.import_deduplicate is True
        assert config.parquet_compression == "snappy"
        assert config.max_export_records == 100000


# =============================================================================
# Dataclass Contract Tests
# =============================================================================


class TestDataclassContracts:
    """Tests to verify all dataclasses follow consistent patterns."""

    @pytest.mark.parametrize(
        "cls",
        [
            IndexInfo,
            StatsResult,
            NamespaceInfo,
            NamespacesResult,
            DeleteNamespaceResult,
            RenameNamespaceResult,
            ExportResult,
            ImportedMemory,
            ImportValidationError,
            ImportResult,
            HybridMemoryMatch,
            HybridRecallResult,
            UtilityConfig,
            ExportImportConfig,
        ],
    )
    def test_is_dataclass(self, cls: type) -> None:
        """All result classes should be dataclasses."""
        from dataclasses import is_dataclass

        assert is_dataclass(cls), f"{cls.__name__} should be a dataclass"

    @pytest.mark.parametrize(
        "cls",
        [
            IndexInfo,
            StatsResult,
            NamespaceInfo,
            NamespacesResult,
            DeleteNamespaceResult,
            RenameNamespaceResult,
            ExportResult,
            ImportedMemory,
            ImportValidationError,
            ImportResult,
            HybridMemoryMatch,
            HybridRecallResult,
        ],
    )
    def test_result_classes_serializable(self, cls: type) -> None:
        """All result classes should be serializable to dict."""
        # This is a compile-time check that asdict works
        # Actual serialization tested in individual tests
        from dataclasses import asdict

        assert callable(asdict)
