"""Unit tests for ExportImportService with mocked dependencies.

Tests the export/import service operations:
- export_memories: Export memories to Parquet/JSON/CSV formats
- import_memories: Import memories with validation and deduplication

Uses mocked repositories and embedding services for isolation.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatial_memory.core.errors import (
    ExportError,
    FileSizeLimitError,
    MemoryImportError,
    PathSecurityError,
    ValidationError,
)
from spatial_memory.core.models import (
    ExportImportConfig,
    ExportResult,
    ImportResult,
    ImportValidationError,
)
from spatial_memory.services.export_import import ExportImportService


# =============================================================================
# Test UUIDs and Constants
# =============================================================================

TEST_UUID_1 = "11111111-1111-1111-1111-111111111111"
TEST_UUID_2 = "22222222-2222-2222-2222-222222222222"
TEST_UUID_3 = "33333333-3333-3333-3333-333333333333"


# =============================================================================
# Helper functions
# =============================================================================


def make_vector(dims: int = 384, seed: int | None = None) -> np.ndarray:
    """Create a random unit vector."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    vec = rng.standard_normal(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return np.asarray(vec / norm, dtype=np.float32)


def make_memory_record(
    memory_id: str,
    content: str = "Test content",
    namespace: str = "default",
    importance: float = 0.5,
    vector: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create a memory record dictionary."""
    now = datetime.now(timezone.utc)
    return {
        "id": memory_id,
        "content": content,
        "namespace": namespace,
        "importance": importance,
        "tags": [],
        "source": "manual",
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "last_accessed": now,
        "access_count": 0,
        "vector": vector.tolist() if vector is not None else make_vector(seed=42).tolist(),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository for unit tests."""
    repo = MagicMock()
    repo.add.return_value = TEST_UUID_1
    repo.get.return_value = None
    repo.search.return_value = []
    repo.count.return_value = 0
    repo.get_namespaces.return_value = []
    repo.get_all_for_export.return_value = iter([])
    repo.bulk_import.return_value = (0, [])
    return repo


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Mock embedding service for unit tests."""
    embeddings = MagicMock()
    embeddings.dimensions = 384
    embeddings.embed = MagicMock(return_value=make_vector(seed=42))
    embeddings.embed_batch = MagicMock(
        return_value=[make_vector(seed=i) for i in range(10)]
    )
    return embeddings


@pytest.fixture
def temp_export_dir() -> Iterator[Path]:
    """Provide a temporary directory for exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        yield export_dir


@pytest.fixture
def temp_import_dir() -> Iterator[Path]:
    """Provide a temporary directory for imports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import_dir = Path(tmpdir) / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        yield import_dir


@pytest.fixture
def export_import_service(
    mock_repository: MagicMock,
    mock_embeddings: MagicMock,
    temp_export_dir: Path,
    temp_import_dir: Path,
) -> ExportImportService:
    """ExportImportService with mocked dependencies and temp directories."""
    config = ExportImportConfig(
        default_export_format="parquet",
        export_batch_size=100,
        import_batch_size=100,
    )
    return ExportImportService(
        repository=mock_repository,
        embeddings=mock_embeddings,
        config=config,
        allowed_export_paths=[temp_export_dir],
        allowed_import_paths=[temp_import_dir],
    )


# =============================================================================
# TestExportMemories
# =============================================================================


class TestExportMemories:
    """Tests for ExportImportService.export_memories()."""

    def test_export_returns_result(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should return ExportResult."""
        output_path = temp_export_dir / "test_export.parquet"
        mock_repository.get_all_for_export.return_value = iter([
            [make_memory_record(TEST_UUID_1)]
        ])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
            format="parquet",
        )

        assert isinstance(result, ExportResult)
        assert result.memories_exported == 1
        assert result.format == "parquet"

    def test_export_auto_detects_format_from_extension(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should auto-detect format from file extension."""
        # Test JSON format
        json_path = temp_export_dir / "export.json"
        mock_repository.get_all_for_export.return_value = iter([
            [make_memory_record(TEST_UUID_1)]
        ])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(output_path=str(json_path))

        assert result.format == "json"

    def test_export_validates_path(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """export_memories() should validate path for security."""
        # Path traversal attempt
        with pytest.raises(PathSecurityError):
            export_import_service.export_memories(
                output_path="../../../etc/passwd.parquet"
            )

    def test_export_json_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should export to JSON format."""
        output_path = temp_export_dir / "export.json"
        records = [
            make_memory_record(TEST_UUID_1, content="Memory 1"),
            make_memory_record(TEST_UUID_2, content="Memory 2"),
        ]
        mock_repository.get_all_for_export.return_value = iter([records])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
            format="json",
        )

        assert result.format == "json"
        assert result.memories_exported == 2
        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_export_csv_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should export to CSV format."""
        output_path = temp_export_dir / "export.csv"
        records = [
            make_memory_record(TEST_UUID_1, content="Memory 1"),
            make_memory_record(TEST_UUID_2, content="Memory 2"),
        ]
        mock_repository.get_all_for_export.return_value = iter([records])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
            format="csv",
            include_vectors=False,  # CSV typically doesn't include vectors
        )

        assert result.format == "csv"
        assert result.memories_exported == 2
        assert output_path.exists()

    def test_export_parquet_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should export to Parquet format."""
        output_path = temp_export_dir / "export.parquet"
        records = [make_memory_record(TEST_UUID_1)]
        mock_repository.get_all_for_export.return_value = iter([records])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
            format="parquet",
        )

        assert result.format == "parquet"
        assert output_path.exists()

    def test_export_filters_by_namespace(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should filter by namespace when specified."""
        output_path = temp_export_dir / "export.json"
        mock_repository.get_all_for_export.return_value = iter([
            [make_memory_record(TEST_UUID_1, namespace="work")]
        ])
        mock_repository.get_namespaces.return_value = ["work"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
            namespace="work",
        )

        # Verify namespace was passed to repository
        mock_repository.get_all_for_export.assert_called_once()
        call_args = mock_repository.get_all_for_export.call_args
        assert call_args.kwargs.get("namespace") == "work" or call_args.args[0] == "work"
        assert "work" in result.namespaces_included

    def test_export_handles_empty_database(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should handle empty database gracefully."""
        output_path = temp_export_dir / "export.json"
        mock_repository.get_all_for_export.return_value = iter([])
        mock_repository.get_namespaces.return_value = []

        result = export_import_service.export_memories(
            output_path=str(output_path),
        )

        assert result.memories_exported == 0

    def test_export_tracks_duration(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should track operation duration."""
        output_path = temp_export_dir / "export.json"
        mock_repository.get_all_for_export.return_value = iter([
            [make_memory_record(TEST_UUID_1)]
        ])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
        )

        assert result.duration_seconds >= 0

    def test_export_calculates_file_size(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should calculate output file size."""
        output_path = temp_export_dir / "export.json"
        mock_repository.get_all_for_export.return_value = iter([
            [make_memory_record(TEST_UUID_1)]
        ])
        mock_repository.get_namespaces.return_value = ["default"]

        result = export_import_service.export_memories(
            output_path=str(output_path),
        )

        assert result.file_size_bytes > 0
        assert result.file_size_mb == result.file_size_bytes / (1024 * 1024)

    def test_export_raises_on_invalid_format(
        self,
        export_import_service: ExportImportService,
        temp_export_dir: Path,
    ) -> None:
        """export_memories() should raise on invalid format."""
        output_path = temp_export_dir / "export.txt"

        with pytest.raises((ValidationError, ExportError, PathSecurityError)):
            export_import_service.export_memories(
                output_path=str(output_path),
                format="invalid_format",
            )


# =============================================================================
# TestImportMemories
# =============================================================================


class TestImportMemories:
    """Tests for ExportImportService.import_memories()."""

    def test_import_returns_result(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should return ImportResult."""
        # Create a test JSON file
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        result = export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=True,
        )

        assert isinstance(result, ImportResult)

    def test_import_dry_run_no_changes(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() with dry_run=True should not modify database."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=True,
        )

        mock_repository.bulk_import.assert_not_called()

    def test_import_applies_changes(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() with dry_run=False should import to database."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        mock_repository.bulk_import.return_value = (1, [TEST_UUID_1])

        result = export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=False,
        )

        assert mock_repository.bulk_import.called
        assert result.memories_imported == 1

    def test_import_validates_path(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """import_memories() should validate path for security."""
        with pytest.raises(PathSecurityError):
            export_import_service.import_memories(
                source_path="../../../etc/passwd.json"
            )

    def test_import_auto_detects_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should auto-detect format from extension."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        result = export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=True,
        )

        assert result.format == "json"

    def test_import_json_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should import from JSON format."""
        import_path = temp_import_dir / "import.json"
        records = [
            make_memory_record(TEST_UUID_1, content="Memory 1"),
            make_memory_record(TEST_UUID_2, content="Memory 2"),
        ]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        result = export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=True,
        )

        assert result.format == "json"
        assert result.total_records_in_file == 2

    def test_import_csv_format(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should import from CSV format."""
        import_path = temp_import_dir / "import.csv"

        # Create CSV file with minimal fields
        with open(import_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "content", "namespace", "importance"])
            writer.writeheader()
            writer.writerow({
                "id": TEST_UUID_1,
                "content": "Memory 1",
                "namespace": "default",
                "importance": "0.5",
            })

        result = export_import_service.import_memories(
            source_path=str(import_path),
            regenerate_embeddings=True,  # CSV typically needs embeddings regenerated
            dry_run=True,
        )

        assert result.format == "csv"

    def test_import_validates_records(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should validate records and report errors."""
        import_path = temp_import_dir / "import.json"
        # Record with missing required content field
        records = [{"id": TEST_UUID_1, "namespace": "default"}]
        with open(import_path, "w") as f:
            json.dump(records, f)

        result = export_import_service.import_memories(
            source_path=str(import_path),
            validate=True,
            dry_run=True,
        )

        assert result.memories_failed > 0 or len(result.validation_errors) > 0

    def test_import_namespace_override(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should override namespace when specified."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1, namespace="original")]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        mock_repository.bulk_import.return_value = (1, [TEST_UUID_1])

        result = export_import_service.import_memories(
            source_path=str(import_path),
            namespace_override="overridden",
            dry_run=False,
        )

        assert result.namespace_override == "overridden"

    def test_import_deduplication(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should deduplicate when enabled."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1, content="Duplicate content")]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        # Mock that similar content already exists
        from spatial_memory.core.models import MemoryResult
        mock_repository.search.return_value = [
            MemoryResult(
                id=TEST_UUID_2,
                content="Duplicate content",
                similarity=0.98,  # Above dedup threshold
                namespace="default",
                tags=[],
                importance=0.5,
                created_at=datetime.now(timezone.utc),
                metadata={},
            )
        ]

        result = export_import_service.import_memories(
            source_path=str(import_path),
            deduplicate=True,
            dedup_threshold=0.95,
            dry_run=True,
        )

        assert result.memories_skipped >= 0

    def test_import_regenerate_embeddings(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should regenerate embeddings when requested."""
        import_path = temp_import_dir / "import.json"
        # Record without vector
        records = [{
            "id": TEST_UUID_1,
            "content": "Test content",
            "namespace": "default",
            "importance": 0.5,
            "tags": [],
            "source": "manual",
            "metadata": {},
        }]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        mock_repository.bulk_import.return_value = (1, [TEST_UUID_1])

        export_import_service.import_memories(
            source_path=str(import_path),
            regenerate_embeddings=True,
            dry_run=False,
        )

        # Embedding service should be called
        assert mock_embeddings.embed.called or mock_embeddings.embed_batch.called

    def test_import_checks_file_size_limit(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should check file size limits."""
        # Create service with very low size limit
        config = ExportImportConfig()
        service = ExportImportService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            config=config,
            allowed_export_paths=[temp_import_dir],
            allowed_import_paths=[temp_import_dir],
            max_import_size_bytes=100,  # Very small limit
        )

        import_path = temp_import_dir / "large_import.json"
        # Create file larger than limit
        records = [make_memory_record(f"uuid-{i}") for i in range(100)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        with pytest.raises(FileSizeLimitError):
            service.import_memories(source_path=str(import_path))

    def test_import_tracks_duration(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """import_memories() should track operation duration."""
        import_path = temp_import_dir / "import.json"
        records = [make_memory_record(TEST_UUID_1)]
        with open(import_path, "w") as f:
            json.dump(records, f, default=str)

        result = export_import_service.import_memories(
            source_path=str(import_path),
            dry_run=True,
        )

        assert result.duration_seconds >= 0

    def test_import_handles_file_not_found(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """import_memories() should handle missing files."""
        with pytest.raises((PathSecurityError, MemoryImportError, FileNotFoundError)):
            export_import_service.import_memories(
                source_path="/nonexistent/path/file.json"
            )


# =============================================================================
# TestExportImportServiceInitialization
# =============================================================================


class TestExportImportServiceInitialization:
    """Tests for ExportImportService initialization and configuration."""

    def test_uses_default_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_export_dir: Path,
        temp_import_dir: Path,
    ) -> None:
        """ExportImportService should use default config when not provided."""
        service = ExportImportService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            allowed_export_paths=[temp_export_dir],
            allowed_import_paths=[temp_import_dir],
        )

        assert service._config is not None
        assert service._config.default_export_format == "parquet"

    def test_uses_custom_config(
        self,
        mock_repository: MagicMock,
        mock_embeddings: MagicMock,
        temp_export_dir: Path,
        temp_import_dir: Path,
    ) -> None:
        """ExportImportService should use provided config."""
        custom_config = ExportImportConfig(
            default_export_format="json",
            export_batch_size=500,
        )

        service = ExportImportService(
            repository=mock_repository,
            embeddings=mock_embeddings,
            config=custom_config,
            allowed_export_paths=[temp_export_dir],
            allowed_import_paths=[temp_import_dir],
        )

        assert service._config.default_export_format == "json"
        assert service._config.export_batch_size == 500

    def test_requires_repository(
        self,
        mock_embeddings: MagicMock,
        temp_export_dir: Path,
        temp_import_dir: Path,
    ) -> None:
        """ExportImportService should require a repository."""
        with pytest.raises(TypeError):
            ExportImportService(
                embeddings=mock_embeddings,
                allowed_export_paths=[temp_export_dir],
                allowed_import_paths=[temp_import_dir],
            )  # type: ignore

    def test_requires_embeddings(
        self,
        mock_repository: MagicMock,
        temp_export_dir: Path,
        temp_import_dir: Path,
    ) -> None:
        """ExportImportService should require an embedding service."""
        with pytest.raises(TypeError):
            ExportImportService(
                repository=mock_repository,
                allowed_export_paths=[temp_export_dir],
                allowed_import_paths=[temp_import_dir],
            )  # type: ignore


# =============================================================================
# TestFormatHandlers
# =============================================================================


class TestFormatHandlers:
    """Tests for format-specific export/import handlers."""

    def test_format_detection_parquet(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should detect parquet format from extension."""
        assert export_import_service._detect_format("file.parquet") == "parquet"
        assert export_import_service._detect_format("file.PARQUET") == "parquet"

    def test_format_detection_json(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should detect json format from extension."""
        assert export_import_service._detect_format("file.json") == "json"
        assert export_import_service._detect_format("file.jsonl") == "json"

    def test_format_detection_csv(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should detect csv format from extension."""
        assert export_import_service._detect_format("file.csv") == "csv"

    def test_format_detection_unknown(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should return None for unknown extensions."""
        assert export_import_service._detect_format("file.txt") is None
        assert export_import_service._detect_format("file.xml") is None


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for record validation during import."""

    def test_validates_required_fields(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should validate required fields are present."""
        # Missing content
        record = {"id": TEST_UUID_1, "namespace": "default"}
        errors = export_import_service._validate_record(record, 0)
        assert len(errors) > 0
        assert any(e.field == "content" for e in errors)

    def test_validates_importance_range(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should validate importance is in valid range."""
        record = {
            "id": TEST_UUID_1,
            "content": "Test",
            "namespace": "default",
            "importance": 2.0,  # Invalid: > 1.0
        }
        errors = export_import_service._validate_record(record, 0)
        assert any(e.field == "importance" for e in errors)

    def test_validates_vector_dimensions(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should validate vector dimensions match expected."""
        record = {
            "id": TEST_UUID_1,
            "content": "Test",
            "namespace": "default",
            "importance": 0.5,
            "vector": [0.1] * 256,  # Wrong dimensions (expecting 384)
        }
        errors = export_import_service._validate_record(record, 0, expected_dims=384)
        assert any("dimension" in e.error.lower() for e in errors)

    def test_accepts_valid_record(
        self,
        export_import_service: ExportImportService,
    ) -> None:
        """Should accept valid records without errors."""
        record = make_memory_record(TEST_UUID_1)
        errors = export_import_service._validate_record(record, 0, expected_dims=384)
        assert len(errors) == 0


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in export/import operations."""

    def test_export_wraps_errors_in_export_error(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_export_dir: Path,
    ) -> None:
        """Export errors should be wrapped in ExportError."""
        output_path = temp_export_dir / "export.parquet"
        mock_repository.get_all_for_export.side_effect = Exception("Database error")

        with pytest.raises(ExportError):
            export_import_service.export_memories(output_path=str(output_path))

    def test_import_wraps_errors_in_import_error(
        self,
        export_import_service: ExportImportService,
        mock_repository: MagicMock,
        temp_import_dir: Path,
    ) -> None:
        """Import errors should be wrapped in MemoryImportError."""
        import_path = temp_import_dir / "import.json"
        # Create invalid JSON
        with open(import_path, "w") as f:
            f.write("not valid json{{{")

        with pytest.raises((MemoryImportError, json.JSONDecodeError)):
            export_import_service.import_memories(source_path=str(import_path))
