"""Custom exceptions for Spatial Memory MCP Server."""

from pathlib import Path


def sanitize_path_for_error(path: str | Path) -> str:
    """Extract only the filename from a path for safe error messages.

    Prevents leaking full system paths in error messages which could
    expose sensitive directory structure information.

    Args:
        path: Full path or filename.

    Returns:
        Just the filename portion.
    """
    if isinstance(path, Path):
        return path.name
    return Path(path).name


class SpatialMemoryError(Exception):
    """Base exception for all spatial memory errors."""

    pass


class MemoryNotFoundError(SpatialMemoryError):
    """Raised when a memory ID doesn't exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class NamespaceNotFoundError(SpatialMemoryError):
    """Raised when a namespace doesn't exist."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        super().__init__(f"Namespace not found: {namespace}")


class EmbeddingError(SpatialMemoryError):
    """Raised when embedding generation fails."""

    pass


class StorageError(SpatialMemoryError):
    """Raised when database operations fail."""

    pass


class PartialBatchInsertError(StorageError):
    """Raised when batch insert partially fails.

    Provides information about which records were successfully inserted
    before the failure, enabling recovery or rollback.
    """

    def __init__(
        self,
        message: str,
        succeeded_ids: list[str],
        total_requested: int,
        failed_batch_index: int | None = None,
    ) -> None:
        """Initialize with details about partial failure.

        Args:
            message: Error description.
            succeeded_ids: IDs of successfully inserted records.
            total_requested: Total number of records requested to insert.
            failed_batch_index: Index of the batch that failed (if batched).
        """
        self.succeeded_ids = succeeded_ids
        self.total_requested = total_requested
        self.failed_batch_index = failed_batch_index
        super().__init__(
            f"{message}. Inserted {len(succeeded_ids)}/{total_requested} records before failure."
        )


class ValidationError(SpatialMemoryError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(SpatialMemoryError):
    """Raised when configuration is invalid."""

    pass


class ClusteringError(SpatialMemoryError):
    """Raised when clustering fails (e.g., too few memories)."""

    pass


class VisualizationError(SpatialMemoryError):
    """Raised when visualization generation fails."""

    pass


class InsufficientMemoriesError(SpatialMemoryError):
    """Raised when operation requires more memories than available."""

    def __init__(self, required: int, available: int, operation: str) -> None:
        self.required = required
        self.available = available
        self.operation = operation
        super().__init__(
            f"{operation} requires at least {required} memories, but only {available} available"
        )


class JourneyError(SpatialMemoryError):
    """Raised when journey path cannot be computed."""

    pass


class WanderError(SpatialMemoryError):
    """Raised when wander cannot continue."""

    pass


class DecayError(SpatialMemoryError):
    """Raised when decay calculation or application fails."""

    pass


class ReinforcementError(SpatialMemoryError):
    """Raised when reinforcement fails."""

    pass


class ExtractionError(SpatialMemoryError):
    """Raised when memory extraction fails."""

    pass


class ConsolidationError(SpatialMemoryError):
    """Raised when consolidation fails."""

    pass


# =============================================================================
# Phase 5 Error Types - Utility Operations
# =============================================================================


class ExportError(SpatialMemoryError):
    """Raised when memory export fails."""

    pass


class MemoryImportError(SpatialMemoryError):
    """Raised when memory import fails.

    Note: Named MemoryImportError to avoid shadowing Python's built-in ImportError.
    """

    pass


class NamespaceOperationError(SpatialMemoryError):
    """Raised when namespace operation fails."""

    pass


class PathSecurityError(SpatialMemoryError):
    """Raised when a file path violates security constraints.

    Examples:
        - Path traversal attempt (../)
        - Path outside allowed directories
        - Symlink to disallowed location
        - Invalid file extension

    Note:
        Error messages only include the filename, not the full path,
        to avoid leaking system directory structure.
    """

    def __init__(
        self,
        path: str,
        violation_type: str,
        message: str | None = None,
    ) -> None:
        self.path = path
        self.violation_type = violation_type
        safe_name = sanitize_path_for_error(path)
        self.message = message or f"Path security violation ({violation_type}): {safe_name}"
        super().__init__(self.message)


class FileSizeLimitError(SpatialMemoryError):
    """Raised when a file exceeds size limits.

    Note:
        Error messages only include the filename, not the full path,
        to avoid leaking system directory structure.
    """

    def __init__(
        self,
        path: str,
        actual_size_bytes: int,
        max_size_bytes: int,
    ) -> None:
        self.path = path
        self.actual_size_bytes = actual_size_bytes
        self.max_size_bytes = max_size_bytes
        actual_mb = actual_size_bytes / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        safe_name = sanitize_path_for_error(path)
        super().__init__(
            f"File exceeds size limit: {safe_name} is {actual_mb:.2f}MB (max: {max_mb:.2f}MB)"
        )


class DimensionMismatchError(ValidationError):
    """Raised when imported vectors have wrong dimensions."""

    def __init__(
        self,
        expected_dim: int,
        actual_dim: int,
        record_index: int | None = None,
    ) -> None:
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.record_index = record_index
        location = f" at record {record_index}" if record_index is not None else ""
        super().__init__(
            f"Vector dimension mismatch{location}: expected {expected_dim}, got {actual_dim}"
        )


class SchemaValidationError(ValidationError):
    """Raised when import data fails schema validation."""

    def __init__(
        self,
        field: str,
        error: str,
        record_index: int | None = None,
    ) -> None:
        self.field = field
        self.error = error
        self.record_index = record_index
        location = f" at record {record_index}" if record_index is not None else ""
        super().__init__(f"Schema validation failed for '{field}'{location}: {error}")


class ImportRecordLimitError(SpatialMemoryError):
    """Raised when import file contains too many records."""

    def __init__(
        self,
        actual_count: int,
        max_count: int,
    ) -> None:
        self.actual_count = actual_count
        self.max_count = max_count
        super().__init__(f"Import file contains {actual_count} records (max: {max_count})")


# =============================================================================
# Cross-Process Locking Error
# =============================================================================


class FileLockError(SpatialMemoryError):
    """Raised when cross-process file lock cannot be acquired."""

    def __init__(
        self,
        lock_path: str,
        timeout: float,
        message: str | None = None,
    ) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        safe_name = sanitize_path_for_error(lock_path)
        self.message = message or f"Failed to acquire file lock at {safe_name} after {timeout}s"
        super().__init__(self.message)


# =============================================================================
# Migration Error
# =============================================================================


class MigrationError(SpatialMemoryError):
    """Raised when a database migration fails."""

    pass


class BackfillError(MigrationError):
    """Raised when a migration's data backfill fails but schema changes succeeded.

    This is a non-fatal migration error: the structural schema change (e.g.,
    adding columns) completed, but populating existing rows with computed values
    failed.  Callers should record the migration as applied (schema is correct)
    but surface the error so operators can re-run the backfill later.
    """

    pass
