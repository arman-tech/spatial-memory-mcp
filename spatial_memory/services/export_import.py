"""Export/Import service for memory data portability.

This service provides the application layer for memory export/import operations:
- export_memories: Export memories to Parquet/JSON/CSV formats
- import_memories: Import memories with validation and deduplication

The service uses dependency injection for repository and embedding services,
following Clean Architecture principles. File I/O and format conversion are
handled at this service layer, while the repository handles only data access.

Security is enforced through PathValidator for all file operations.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from io import TextIOWrapper
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator

import numpy as np

from spatial_memory.core.errors import (
    ExportError,
    FileSizeLimitError,
    ImportRecordLimitError,
    MemoryImportError,
    PathSecurityError,
    ValidationError,
)
from spatial_memory.core.file_security import PathValidator
from spatial_memory.core.models import (
    ExportImportConfig,
    ExportResult,
    ImportedMemory,
    ImportResult,
    ImportValidationError,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_FORMATS = frozenset({"parquet", "json", "csv"})

EXTENSION_TO_FORMAT: dict[str, str] = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
}

# Required fields for import validation
REQUIRED_IMPORT_FIELDS = frozenset({"content"})

# Default import size limit (100 MB)
DEFAULT_MAX_IMPORT_SIZE_BYTES = 100 * 1024 * 1024


# =============================================================================
# Service Implementation
# =============================================================================


class ExportImportService:
    """Service for memory export and import operations.

    Uses Clean Architecture - depends on protocol interfaces, not implementations.
    Handles file I/O and format conversion at the service layer while delegating
    data access to the repository.

    Security Features:
    - Path validation to prevent traversal attacks
    - File size limits for imports
    - Symlink detection (optional)
    - Extension validation

    Example:
        service = ExportImportService(
            repository=repo,
            embeddings=emb,
            allowed_export_paths=[Path("./exports")],
            allowed_import_paths=[Path("./imports")],
        )

        # Export memories
        result = service.export_memories(
            output_path="./exports/backup.parquet",
            namespace="work",
        )

        # Import memories
        result = service.import_memories(
            source_path="./imports/restore.json",
            dry_run=False,
        )
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: ExportImportConfig | None = None,
        allowed_export_paths: Sequence[str | Path] | None = None,
        allowed_import_paths: Sequence[str | Path] | None = None,
        allow_symlinks: bool = False,
        max_import_size_bytes: int | None = None,
    ) -> None:
        """Initialize the export/import service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
            config: Optional configuration (uses defaults if not provided).
            allowed_export_paths: Directories where exports are permitted.
            allowed_import_paths: Directories where imports are permitted.
            allow_symlinks: Whether to allow following symlinks (default False).
            max_import_size_bytes: Maximum import file size in bytes.
        """
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or ExportImportConfig()

        # Set up path validator
        export_paths = allowed_export_paths or [Path("./exports"), Path("./backups")]
        import_paths = allowed_import_paths or [Path("./imports"), Path("./backups")]

        self._path_validator = PathValidator(
            allowed_export_paths=export_paths,
            allowed_import_paths=import_paths,
            allow_symlinks=allow_symlinks,
        )

        self._max_import_size_bytes = (
            max_import_size_bytes or DEFAULT_MAX_IMPORT_SIZE_BYTES
        )

    def export_memories(
        self,
        output_path: str,
        format: str | None = None,
        namespace: str | None = None,
        include_vectors: bool = True,
    ) -> ExportResult:
        """Export memories to a file.

        Streams data from repository and writes to the specified format.
        Supports Parquet (recommended for full fidelity), JSON, and CSV.

        Args:
            output_path: Path for output file. Extension determines format
                if format parameter is not specified.
            format: Export format (parquet, json, csv). Auto-detected from
                extension if not specified.
            namespace: Export only this namespace (all if not specified).
            include_vectors: Include embedding vectors in export (default True).
                Note: CSV exports may set this to False for readability.

        Returns:
            ExportResult with export statistics.

        Raises:
            ExportError: If export operation fails.
            PathSecurityError: If path validation fails.
            ValidationError: If input validation fails.
        """
        start_time = time.monotonic()

        # Validate and resolve path
        try:
            canonical_path = self._path_validator.validate_export_path(output_path)
        except (PathSecurityError, ValueError) as e:
            raise PathSecurityError(
                path=output_path,
                violation_type="export_path_validation_failed",
                message=str(e),
            ) from e

        # Detect or validate format
        detected_format = format or self._detect_format(output_path)
        if detected_format is None:
            detected_format = self._config.default_export_format

        if detected_format not in SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported export format: {detected_format}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        try:
            # Check export record limit before starting
            if self._config.max_export_records > 0:
                memory_count = self._repo.count(namespace=namespace)
                if memory_count > self._config.max_export_records:
                    raise ExportError(
                        f"Export would contain {memory_count} records, "
                        f"exceeding limit of {self._config.max_export_records}. "
                        "Consider filtering by namespace or increasing max_export_records."
                    )

            # Ensure parent directory exists
            canonical_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream data from repository
            batches = self._repo.get_all_for_export(
                namespace=namespace,
                batch_size=self._config.export_batch_size,
            )

            # Get namespaces for result
            if namespace:
                namespaces_included = [namespace]
            else:
                namespaces_included = self._repo.get_namespaces()

            # Export based on format
            if detected_format == "parquet":
                memories_exported = self._export_parquet(
                    canonical_path, batches, include_vectors
                )
            elif detected_format == "json":
                memories_exported = self._export_json(
                    canonical_path, batches, include_vectors
                )
            elif detected_format == "csv":
                memories_exported = self._export_csv(
                    canonical_path, batches, include_vectors
                )
            else:
                raise ExportError(f"Unsupported format: {detected_format}")

            # Calculate file size
            if canonical_path.exists():
                file_size_bytes = canonical_path.stat().st_size
            else:
                file_size_bytes = 0

            duration_seconds = time.monotonic() - start_time

            return ExportResult(
                format=detected_format,
                output_path=str(canonical_path),
                memories_exported=memories_exported,
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_bytes / (1024 * 1024),
                namespaces_included=namespaces_included,
                duration_seconds=duration_seconds,
                compression="zstd" if detected_format == "parquet" else None,
            )

        except (ExportError, PathSecurityError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ExportError(f"Export operation failed: {e}") from e

    def import_memories(
        self,
        source_path: str,
        format: str | None = None,
        namespace_override: str | None = None,
        deduplicate: bool = False,
        dedup_threshold: float = 0.95,
        validate: bool = True,
        regenerate_embeddings: bool = False,
        dry_run: bool = True,
    ) -> ImportResult:
        """Import memories from a file.

        Parses the file, validates records, optionally deduplicates against
        existing memories, and imports to the repository.

        Args:
            source_path: Path to source file.
            format: Import format (parquet, json, csv). Auto-detected from
                extension if not specified.
            namespace_override: Override namespace for all imported memories.
            deduplicate: Skip records similar to existing memories (default False).
            dedup_threshold: Similarity threshold for deduplication (0.7-0.99).
            validate: Validate records before import (default True).
            regenerate_embeddings: Generate new embeddings for imported memories.
                Required if source lacks vectors or dimensions don't match.
            dry_run: Validate without importing (default True). Set to False
                to actually import the memories.

        Returns:
            ImportResult with import statistics and validation errors.

        Raises:
            MemoryImportError: If import operation fails.
            PathSecurityError: If path validation fails.
            FileSizeLimitError: If file exceeds size limit.
            ValidationError: If input validation fails.
        """
        start_time = time.monotonic()

        # Detect or validate format BEFORE opening file
        detected_format = format or self._detect_format(source_path)
        if detected_format is None:
            raise ValidationError(
                f"Cannot detect format from path: {source_path}. "
                "Please specify format explicitly."
            )

        if detected_format not in SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported import format: {detected_format}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        # Validate dedup threshold
        if deduplicate and not 0.7 <= dedup_threshold <= 0.99:
            raise ValidationError(
                "dedup_threshold must be between 0.7 and 0.99"
            )

        # ATOMIC: Validate and open file in one step (prevents TOCTOU)
        # The file handle MUST be used for reading, not re-opened by path
        try:
            canonical_path, file_handle = self._path_validator.validate_and_open_import_file(
                source_path,
                max_size_bytes=self._max_import_size_bytes,
            )
        except PathSecurityError as e:
            raise e
        except FileSizeLimitError as e:
            raise e
        except ValueError as e:
            raise PathSecurityError(
                path=source_path,
                violation_type="import_path_validation_failed",
                message=str(e),
            ) from e

        try:
            # Parse file using the ALREADY OPEN file handle (TOCTOU safe)
            if detected_format == "parquet":
                records_iter = self._parse_parquet_from_handle(file_handle, canonical_path)
            elif detected_format == "json":
                records_iter = self._parse_json_from_handle(file_handle)
            elif detected_format == "csv":
                records_iter = self._parse_csv_from_handle(file_handle)
            else:
                raise MemoryImportError(f"Unsupported format: {detected_format}")

            # Stream records with early termination to prevent memory exhaustion.
            # Check limit during iteration, not after loading all records.
            max_records = self._config.max_import_records
            records: list[dict[str, Any]] = []

            for record in records_iter:
                records.append(record)
                # Fail fast if limit exceeded - prevents memory exhaustion from large files
                if max_records > 0 and len(records) > max_records:
                    raise ImportRecordLimitError(
                        actual_count=len(records),
                        max_count=max_records,
                    )
        finally:
            # Ensure file is closed even if parsing fails
            file_handle.close()

        try:

            # Process records
            total_records = 0
            valid_records: list[dict[str, Any]] = []
            validation_errors: list[ImportValidationError] = []
            skipped_count = 0
            failed_count = 0
            imported_memories: list[ImportedMemory] = []

            for idx, record in enumerate(records):
                total_records += 1

                # Validate record if requested
                if validate:
                    expected_dims = (
                        self._embeddings.dimensions
                        if not regenerate_embeddings
                        else None
                    )
                    errors = self._validate_record(record, idx, expected_dims)
                    if errors:
                        validation_errors.extend(errors)
                        failed_count += 1
                        continue

                # Apply namespace override
                if namespace_override:
                    record["namespace"] = namespace_override

                # Handle embeddings
                if regenerate_embeddings or "vector" not in record:
                    if not dry_run:
                        vector = self._embeddings.embed(record["content"])
                        record["vector"] = vector.tolist()
                    else:
                        # In dry run, just mark that we would regenerate
                        record["_needs_embedding"] = True

                # Deduplicate if requested
                if deduplicate and not dry_run:
                    is_duplicate = self._check_duplicate(
                        record, dedup_threshold
                    )
                    if is_duplicate is True:
                        skipped_count += 1
                        continue
                    # If is_duplicate is None (check failed), proceed with import
                    # This is a conservative policy - import on failure

                valid_records.append(record)

            # Import if not dry run
            memories_imported = 0
            imported_ids: list[str] = []

            if not dry_run and valid_records:
                # Filter out internal fields
                import_records = [
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in valid_records
                ]

                memories_imported, imported_ids = self._repo.bulk_import(
                    iter(import_records),
                    batch_size=self._config.import_batch_size,
                    namespace_override=namespace_override,
                )

                # Build imported memories list
                for record, new_id in zip(valid_records, imported_ids):
                    content = record.get("content", "")
                    preview = content[:100] + "..." if len(content) > 100 else content
                    imported_memories.append(
                        ImportedMemory(
                            id=new_id,
                            content_preview=preview,
                            namespace=record.get("namespace", "default"),
                            was_deduplicated=False,
                            original_id=record.get("id"),
                        )
                    )
            elif dry_run:
                # In dry run, count valid records as "would be imported"
                memories_imported = len(valid_records)

            duration_seconds = time.monotonic() - start_time

            return ImportResult(
                source_path=str(canonical_path),
                format=detected_format,
                total_records_in_file=total_records,
                memories_imported=memories_imported,
                memories_skipped=skipped_count,
                memories_failed=failed_count,
                validation_errors=validation_errors,
                duration_seconds=duration_seconds,
                namespace_override=namespace_override,
                imported_memories=imported_memories if not dry_run else None,
            )

        except (MemoryImportError, PathSecurityError, ValidationError, FileSizeLimitError):
            raise
        except json.JSONDecodeError as e:
            raise MemoryImportError(f"Invalid JSON in import file: {e}") from e
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise MemoryImportError(f"Import operation failed: {e}") from e

    # =========================================================================
    # Format Detection
    # =========================================================================

    def _detect_format(self, path: str) -> str | None:
        """Detect format from file extension.

        Args:
            path: File path.

        Returns:
            Format string or None if unknown.
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        return EXTENSION_TO_FORMAT.get(ext)

    # =========================================================================
    # Export Format Handlers
    # =========================================================================

    def _export_parquet(
        self,
        path: Path,
        batches: Iterator[list[dict[str, Any]]],
        include_vectors: bool,
    ) -> int:
        """Export to Parquet format.

        Args:
            path: Output file path.
            batches: Iterator of record batches.
            include_vectors: Whether to include embedding vectors.

        Returns:
            Number of records exported.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ExportError(
                "pyarrow is required for Parquet export. "
                "Install with: pip install pyarrow"
            ) from e

        all_records: list[dict[str, Any]] = []
        for batch in batches:
            for record in batch:
                # Convert for Parquet compatibility
                processed = self._prepare_record_for_export(record, include_vectors)
                # Parquet needs metadata as string to avoid empty struct issues
                if "metadata" in processed:
                    if isinstance(processed["metadata"], dict):
                        processed["metadata"] = json.dumps(processed["metadata"])
                all_records.append(processed)

        if not all_records:
            # Write empty parquet file
            schema = pa.schema([
                ("id", pa.string()),
                ("content", pa.string()),
                ("namespace", pa.string()),
                ("importance", pa.float32()),
                ("tags", pa.list_(pa.string())),
                ("source", pa.string()),
                ("metadata", pa.string()),
                ("created_at", pa.timestamp("us", tz="UTC")),
                ("updated_at", pa.timestamp("us", tz="UTC")),
                ("last_accessed", pa.timestamp("us", tz="UTC")),
                ("access_count", pa.int32()),
            ])
            if include_vectors:
                schema = schema.append(pa.field("vector", pa.list_(pa.float32())))
            table = pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)
        else:
            table = pa.Table.from_pylist(all_records)

        pq.write_table(
            table,
            path,
            compression=self._config.parquet_compression,
        )

        return len(all_records)

    def _export_json(
        self,
        path: Path,
        batches: Iterator[list[dict[str, Any]]],
        include_vectors: bool,
    ) -> int:
        """Export to JSON format.

        Args:
            path: Output file path.
            batches: Iterator of record batches.
            include_vectors: Whether to include embedding vectors.

        Returns:
            Number of records exported.
        """
        all_records: list[dict[str, Any]] = []
        for batch in batches:
            for record in batch:
                processed = self._prepare_record_for_export(record, include_vectors)
                all_records.append(processed)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, default=self._json_serializer, indent=2)

        return len(all_records)

    def _export_csv(
        self,
        path: Path,
        batches: Iterator[list[dict[str, Any]]],
        include_vectors: bool,
    ) -> int:
        """Export to CSV format.

        Args:
            path: Output file path.
            batches: Iterator of record batches.
            include_vectors: Whether to include embedding vectors.

        Returns:
            Number of records exported.
        """
        all_records: list[dict[str, Any]] = []
        for batch in batches:
            for record in batch:
                processed = self._prepare_record_for_export(record, include_vectors)
                # Convert complex types to strings for CSV
                processed["tags"] = json.dumps(processed.get("tags", []))
                processed["metadata"] = json.dumps(processed.get("metadata", {}))
                if include_vectors and "vector" in processed:
                    processed["vector"] = json.dumps(processed["vector"])
                # Convert datetimes to ISO format
                for key in ["created_at", "updated_at", "last_accessed"]:
                    if key in processed and processed[key] is not None:
                        if isinstance(processed[key], datetime):
                            processed[key] = processed[key].isoformat()
                all_records.append(processed)

        if not all_records:
            # Write empty CSV with header
            fieldnames = [
                "id", "content", "namespace", "importance", "tags",
                "source", "metadata", "created_at", "updated_at",
                "last_accessed", "access_count"
            ]
            if include_vectors:
                fieldnames.append("vector")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return 0

        fieldnames = list(all_records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        return len(all_records)

    def _prepare_record_for_export(
        self,
        record: dict[str, Any],
        include_vectors: bool,
    ) -> dict[str, Any]:
        """Prepare a record for export.

        Args:
            record: Raw record from repository.
            include_vectors: Whether to include embedding vectors.

        Returns:
            Processed record suitable for export.
        """
        processed = dict(record)

        # Handle vector
        if not include_vectors:
            processed.pop("vector", None)
        elif "vector" in processed:
            # Ensure vector is a list, not numpy array
            vec = processed["vector"]
            if isinstance(vec, np.ndarray):
                processed["vector"] = vec.tolist()

        # Handle metadata - ensure it's JSON serializable
        if "metadata" in processed:
            meta = processed["metadata"]
            if isinstance(meta, str):
                try:
                    processed["metadata"] = json.loads(meta)
                except json.JSONDecodeError:
                    processed["metadata"] = {}

        return processed

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # =========================================================================
    # Import Format Handlers (TOCTOU-safe versions using file handles)
    # =========================================================================

    def _parse_parquet_from_handle(
        self, file_handle: BinaryIO, path: Path
    ) -> Iterator[dict[str, Any]]:
        """Parse Parquet from an already-open file handle (TOCTOU-safe).

        Args:
            file_handle: Open binary file handle.
            path: Original path (for error messages only).

        Yields:
            Memory records as dictionaries.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise MemoryImportError(
                "pyarrow is required for Parquet import. "
                "Install with: pip install pyarrow"
            ) from e

        try:
            # PyArrow can read from file-like objects
            table = pq.read_table(file_handle)
            records = table.to_pylist()

            for record in records:
                # Convert metadata from string if needed
                if "metadata" in record and isinstance(record["metadata"], str):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}
                yield record
        except Exception as e:
            raise MemoryImportError(f"Failed to parse Parquet file {path}: {e}") from e

    def _parse_json_from_handle(self, file_handle: BinaryIO) -> Iterator[dict[str, Any]]:
        """Parse JSON from an already-open file handle (TOCTOU-safe).

        Args:
            file_handle: Open binary file handle.

        Yields:
            Memory records as dictionaries.
        """
        # Read and decode content
        content = file_handle.read().decode("utf-8").strip()

        # Handle both JSON array and JSON Lines formats
        if content.startswith("["):
            # JSON array
            records = json.loads(content)
            for record in records:
                yield record
        else:
            # JSON Lines (one object per line)
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _parse_csv_from_handle(self, file_handle: BinaryIO) -> Iterator[dict[str, Any]]:
        """Parse CSV from an already-open file handle (TOCTOU-safe).

        Args:
            file_handle: Open binary file handle.

        Yields:
            Memory records as dictionaries.
        """
        # Wrap binary handle in text wrapper for CSV reader
        text_handle = TextIOWrapper(file_handle, encoding="utf-8", newline="")
        try:
            reader = csv.DictReader(text_handle)

            for row in reader:
                record: dict[str, Any] = dict(row)

                # Convert string fields to appropriate types
                if "importance" in record:
                    try:
                        record["importance"] = float(record["importance"])
                    except (ValueError, TypeError):
                        record["importance"] = 0.5

                if "access_count" in record:
                    try:
                        record["access_count"] = int(record["access_count"])
                    except (ValueError, TypeError):
                        record["access_count"] = 0

                # Parse JSON fields
                if "tags" in record and isinstance(record["tags"], str):
                    try:
                        record["tags"] = json.loads(record["tags"])
                    except json.JSONDecodeError:
                        record["tags"] = []

                if "metadata" in record and isinstance(record["metadata"], str):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}

                if "vector" in record and isinstance(record["vector"], str):
                    try:
                        record["vector"] = json.loads(record["vector"])
                    except json.JSONDecodeError:
                        # Remove invalid vector
                        del record["vector"]

                yield record
        finally:
            # Detach text wrapper to prevent it from closing the underlying handle
            text_handle.detach()

    # =========================================================================
    # DEPRECATED: Legacy Import Format Handlers
    # These methods open files by path and are NOT TOCTOU-safe.
    # Use _parse_*_from_handle methods instead for secure imports.
    # =========================================================================

    def _parse_parquet(self, path: Path) -> Iterator[dict[str, Any]]:
        """Parse Parquet file.

        .. deprecated::
            This method is NOT TOCTOU-safe. Use _parse_parquet_from_handle instead.

        Args:
            path: Input file path.

        Yields:
            Memory records as dictionaries.
        """
        import warnings

        warnings.warn(
            "_parse_parquet is deprecated and not TOCTOU-safe. "
            "Use _parse_parquet_from_handle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise MemoryImportError(
                "pyarrow is required for Parquet import. "
                "Install with: pip install pyarrow"
            ) from e

        table = pq.read_table(path)
        records = table.to_pylist()

        for record in records:
            # Convert metadata from string if needed
            if "metadata" in record and isinstance(record["metadata"], str):
                try:
                    record["metadata"] = json.loads(record["metadata"])
                except json.JSONDecodeError:
                    record["metadata"] = {}
            yield record

    def _parse_json(self, path: Path) -> Iterator[dict[str, Any]]:
        """Parse JSON file.

        .. deprecated::
            This method is NOT TOCTOU-safe. Use _parse_json_from_handle instead.

        Args:
            path: Input file path.

        Yields:
            Memory records as dictionaries.
        """
        import warnings

        warnings.warn(
            "_parse_json is deprecated and not TOCTOU-safe. "
            "Use _parse_json_from_handle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()

        # Handle both JSON array and JSON Lines formats
        if content.startswith("["):
            # JSON array
            records = json.loads(content)
            for record in records:
                yield record
        else:
            # JSON Lines (one object per line)
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _parse_csv(self, path: Path) -> Iterator[dict[str, Any]]:
        """Parse CSV file.

        .. deprecated::
            This method is NOT TOCTOU-safe. Use _parse_csv_from_handle instead.

        Args:
            path: Input file path.

        Yields:
            Memory records as dictionaries.
        """
        import warnings

        warnings.warn(
            "_parse_csv is deprecated and not TOCTOU-safe. "
            "Use _parse_csv_from_handle instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record: dict[str, Any] = dict(row)

                # Convert string fields to appropriate types
                if "importance" in record:
                    try:
                        record["importance"] = float(record["importance"])
                    except (ValueError, TypeError):
                        record["importance"] = 0.5

                if "access_count" in record:
                    try:
                        record["access_count"] = int(record["access_count"])
                    except (ValueError, TypeError):
                        record["access_count"] = 0

                # Parse JSON fields
                if "tags" in record and isinstance(record["tags"], str):
                    try:
                        record["tags"] = json.loads(record["tags"])
                    except json.JSONDecodeError:
                        record["tags"] = []

                if "metadata" in record and isinstance(record["metadata"], str):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}

                if "vector" in record and isinstance(record["vector"], str):
                    try:
                        record["vector"] = json.loads(record["vector"])
                    except json.JSONDecodeError:
                        # Remove invalid vector
                        del record["vector"]

                yield record

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_record(
        self,
        record: dict[str, Any],
        row_number: int,
        expected_dims: int | None = None,
    ) -> list[ImportValidationError]:
        """Validate a single import record.

        Args:
            record: Record to validate.
            row_number: Row number for error reporting.
            expected_dims: Expected vector dimensions (None to skip check).

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[ImportValidationError] = []

        # Check required fields
        for field in REQUIRED_IMPORT_FIELDS:
            if field not in record or record[field] is None:
                errors.append(
                    ImportValidationError(
                        row_number=row_number,
                        field=field,
                        error=f"Required field '{field}' is missing",
                        value=None,
                    )
                )
            elif field == "content" and not str(record[field]).strip():
                errors.append(
                    ImportValidationError(
                        row_number=row_number,
                        field=field,
                        error="Content cannot be empty",
                        value=str(record[field])[:50],
                    )
                )

        # Validate importance range
        if "importance" in record:
            importance = record["importance"]
            try:
                importance_float = float(importance)
                if not 0.0 <= importance_float <= 1.0:
                    errors.append(
                        ImportValidationError(
                            row_number=row_number,
                            field="importance",
                            error="Importance must be between 0.0 and 1.0",
                            value=str(importance),
                        )
                    )
            except (ValueError, TypeError):
                errors.append(
                    ImportValidationError(
                        row_number=row_number,
                        field="importance",
                        error="Importance must be a number",
                        value=str(importance)[:50],
                    )
                )

        # Validate vector dimensions
        if expected_dims is not None and "vector" in record:
            vector = record["vector"]
            if vector is not None:
                try:
                    if isinstance(vector, (list, np.ndarray)):
                        actual_dims = len(vector)
                        if actual_dims != expected_dims:
                            errors.append(
                                ImportValidationError(
                                    row_number=row_number,
                                    field="vector",
                                    error=f"Vector dimension mismatch: expected {expected_dims}, got {actual_dims}",
                                    value=f"[{actual_dims} dimensions]",
                                )
                            )
                except (TypeError, AttributeError):
                    errors.append(
                        ImportValidationError(
                            row_number=row_number,
                            field="vector",
                            error="Vector must be an array of numbers",
                            value=str(type(vector)),
                        )
                    )

        return errors

    # =========================================================================
    # Deduplication
    # =========================================================================

    def _check_duplicate(
        self,
        record: dict[str, Any],
        threshold: float,
    ) -> bool | None:
        """Check if record is a duplicate of an existing memory.

        Args:
            record: Record to check.
            threshold: Similarity threshold for deduplication.

        Returns:
            True if record is a duplicate.
            False if no duplicate found.
            None if the check failed (let caller decide policy).
        """
        try:
            # Get vector for comparison
            if "vector" in record and record["vector"] is not None:
                vector = np.array(record["vector"], dtype=np.float32)
            else:
                # Generate embedding for comparison
                vector = self._embeddings.embed(record["content"])

            # Search for similar existing memories
            namespace = record.get("namespace")
            results = self._repo.search(vector, limit=5, namespace=namespace)

            # Check if any result exceeds threshold
            for result in results:
                if result.similarity >= threshold:
                    logger.debug(
                        f"Duplicate found: similarity {result.similarity:.3f} "
                        f">= threshold {threshold:.3f}"
                    )
                    return True

            return False

        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return None
