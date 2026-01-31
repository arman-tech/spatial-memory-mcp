"""Consolidated security module for file path operations.

This module provides a unified security facade for all file path validation
operations in the spatial-memory-mcp project. It consolidates path traversal
prevention, symlink attack prevention, file size limits, and allowed directory
configuration into a single, production-ready interface.

Security Architecture:
----------------------
The security system implements defense-in-depth with multiple layers:

1. **Input Validation Layer**
   - Pattern-based detection of known attack vectors
   - URL decoding to catch encoded attacks
   - Null byte injection prevention

2. **Path Canonicalization Layer**
   - Resolution of symbolic path elements
   - Detection of traversal after normalization
   - UNC path blocking (Windows network shares)

3. **Access Control Layer**
   - Directory allowlist enforcement
   - Sensitive directory blocking
   - File extension validation

4. **Runtime Protection Layer**
   - Symlink detection and optional blocking
   - File size limit enforcement
   - TOCTOU (Time-of-Check-Time-of-Use) prevention

Threat Model:
-------------
This module defends against:

- **Path Traversal Attacks**: Attempts to access files outside allowed
  directories using sequences like ../, URL encoding (%2e%2e), or
  double encoding (%252e%252e).

- **Symlink Attacks**: Creating symlinks in allowed directories that
  point to sensitive files elsewhere.

- **UNC Path Attacks**: Accessing network shares on Windows using
  paths like \\\\server\\share.

- **Large File DoS**: Uploading excessively large files to exhaust
  disk space or memory.

- **Extension Spoofing**: Attempting to import/export executable or
  script files by disguising extensions.

- **TOCTOU Race Conditions**: Swapping files between validation and
  actual file operations.

Usage Example:
--------------
    from spatial_memory.core.security import (
        FileSecurityManager,
        SecurityConfig,
        validate_export_path,
        validate_import_file,
    )

    # Using the high-level API
    config = SecurityConfig(
        export_allowed_paths=["./exports", "./backups"],
        import_allowed_paths=["./imports"],
        max_import_size_mb=100.0,
        allow_symlinks=False,
    )
    manager = FileSecurityManager(config)

    # Validate export path
    safe_path = manager.validate_export_path("./exports/backup.parquet")

    # Validate and open import file atomically (TOCTOU-safe)
    path, handle = manager.validate_and_open_import("./imports/data.json")
    try:
        content = handle.read()
    finally:
        handle.close()

    # Or use convenience functions with default settings
    safe_path = validate_export_path("./exports/backup.parquet", config)

Module Dependencies:
--------------------
- spatial_memory.core.errors: Custom exception types
- spatial_memory.core.file_security: PathValidator implementation
- spatial_memory.core.import_security: Import record validation
- spatial_memory.config: Application configuration

Author: Spatial Memory MCP Team
Security Review: Phase 5 Implementation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from spatial_memory.core.errors import (
    DimensionMismatchError,
    FileSizeLimitError,
    PathSecurityError,
    SchemaValidationError,
    ValidationError,
)
from spatial_memory.core.file_security import (
    PATH_TRAVERSAL_PATTERNS,
    SENSITIVE_DIRECTORIES,
    VALID_EXTENSIONS,
    PathValidator,
)
from spatial_memory.core.import_security import (
    BatchValidationResult,
    ImportValidationConfig,
    ImportValidator,
)

__all__ = [
    # Configuration
    "SecurityConfig",
    "DEFAULT_SECURITY_CONFIG",
    # Manager class
    "FileSecurityManager",
    # Convenience functions
    "validate_export_path",
    "validate_import_path",
    "validate_import_file",
    "validate_import_records",
    # Error types (re-exported for convenience)
    "PathSecurityError",
    "FileSizeLimitError",
    "DimensionMismatchError",
    "SchemaValidationError",
    "ValidationError",
    # Constants (re-exported for inspection)
    "PATH_TRAVERSAL_PATTERNS",
    "SENSITIVE_DIRECTORIES",
    "VALID_EXTENSIONS",
    # Validation result types
    "BatchValidationResult",
    "ImportValidationConfig",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """Immutable security configuration for file operations.

    This configuration is intentionally immutable (frozen=True) to prevent
    runtime modification of security settings after initialization.

    Attributes:
        export_allowed_paths: Directories where exports are permitted.
            Paths can be absolute or relative. Relative paths are resolved
            against the current working directory.

        import_allowed_paths: Directories where imports are permitted.
            Same resolution rules as export_allowed_paths.

        max_import_size_mb: Maximum file size for imports in megabytes.
            Files exceeding this size will be rejected to prevent DoS
            attacks through disk/memory exhaustion.

        allow_symlinks: Whether to allow following symlinks. Default False
            for security - symlinks can escape allowed directories.

        max_import_records: Maximum number of records in an import file.
            Prevents memory exhaustion from very large files.

    Example:
        config = SecurityConfig(
            export_allowed_paths=["./exports", "/data/backups"],
            import_allowed_paths=["./imports"],
            max_import_size_mb=100.0,
            allow_symlinks=False,
            max_import_records=100_000,
        )
    """

    export_allowed_paths: tuple[str, ...] = field(
        default_factory=lambda: ("./exports", "./backups")
    )
    import_allowed_paths: tuple[str, ...] = field(
        default_factory=lambda: ("./imports", "./backups")
    )
    max_import_size_mb: float = 100.0
    allow_symlinks: bool = False
    max_import_records: int = 100_000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Convert lists to tuples if needed (for immutability)
        if isinstance(self.export_allowed_paths, list):
            object.__setattr__(
                self, "export_allowed_paths", tuple(self.export_allowed_paths)
            )
        if isinstance(self.import_allowed_paths, list):
            object.__setattr__(
                self, "import_allowed_paths", tuple(self.import_allowed_paths)
            )

        # Validate ranges
        if self.max_import_size_mb <= 0:
            raise ValueError("max_import_size_mb must be positive")
        if self.max_import_size_mb > 10_000:
            raise ValueError("max_import_size_mb cannot exceed 10GB")
        if self.max_import_records <= 0:
            raise ValueError("max_import_records must be positive")

    @property
    def max_import_size_bytes(self) -> int:
        """Return maximum import size in bytes."""
        return int(self.max_import_size_mb * 1024 * 1024)


# Default security configuration for general use
DEFAULT_SECURITY_CONFIG = SecurityConfig()


# =============================================================================
# File Security Manager
# =============================================================================


class FileSecurityManager:
    """Unified security manager for all file path operations.

    This class provides a thread-safe, production-ready interface for
    validating file paths used in export/import operations. It encapsulates
    all security checks into a single interface.

    Thread Safety:
        This class is thread-safe. The underlying PathValidator is stateless
        and only reads from immutable configuration.

    Example:
        manager = FileSecurityManager(config)

        # Export validation (file doesn't need to exist)
        safe_path = manager.validate_export_path("./exports/backup.parquet")

        # Import validation (file must exist, size checked)
        safe_path = manager.validate_import_path("./imports/data.json")

        # TOCTOU-safe import (validates and opens atomically)
        path, handle = manager.validate_and_open_import("./imports/data.json")
        try:
            data = handle.read()
        finally:
            handle.close()
    """

    def __init__(self, config: SecurityConfig | None = None) -> None:
        """Initialize the security manager.

        Args:
            config: Security configuration. Uses DEFAULT_SECURITY_CONFIG
                if not provided.
        """
        self._config = config or DEFAULT_SECURITY_CONFIG

        # Initialize the underlying PathValidator
        self._validator = PathValidator(
            allowed_export_paths=list(self._config.export_allowed_paths),
            allowed_import_paths=list(self._config.import_allowed_paths),
            allow_symlinks=self._config.allow_symlinks,
        )

        logger.debug(
            "FileSecurityManager initialized with config: "
            f"export_paths={self._config.export_allowed_paths}, "
            f"import_paths={self._config.import_allowed_paths}, "
            f"max_size_mb={self._config.max_import_size_mb}, "
            f"allow_symlinks={self._config.allow_symlinks}"
        )

    @property
    def config(self) -> SecurityConfig:
        """Return the security configuration (read-only)."""
        return self._config

    def validate_export_path(self, path: str | Path) -> Path:
        """Validate a path for export operations.

        Performs all security checks without requiring the file to exist.
        Parent directories will be created during export if needed.

        Security checks performed:
        1. Path traversal pattern detection
        2. URL decoding and re-checking
        3. UNC path blocking
        4. Path canonicalization
        5. Extension validation (.parquet, .json, .csv only)
        6. Symlink detection (if file exists and symlinks disabled)
        7. Allowlist validation

        Args:
            path: The path to validate. Can be absolute or relative.

        Returns:
            Canonicalized Path object that is safe to use.

        Raises:
            PathSecurityError: If the path fails any security check.
            ValueError: If the path is empty or contains null bytes.

        Example:
            safe_path = manager.validate_export_path("./exports/backup.parquet")
            # Use safe_path for actual file writing
        """
        return self._validator.validate_export_path(path)

    def validate_import_path(self, path: str | Path) -> Path:
        """Validate a path for import operations.

        Performs all export validation checks plus:
        - File existence verification
        - Directory rejection (must be a file)
        - File size limit enforcement

        Args:
            path: The path to validate. Can be absolute or relative.

        Returns:
            Canonicalized Path object that is safe to use.

        Raises:
            PathSecurityError: If the path fails any security check.
            FileSizeLimitError: If the file exceeds the size limit.
            ValueError: If the path is empty or contains null bytes.

        Example:
            safe_path = manager.validate_import_path("./imports/data.json")
            # Use safe_path for actual file reading
        """
        return self._validator.validate_import_path(
            path, max_size_bytes=self._config.max_import_size_bytes
        )

    def validate_and_open_import(
        self, path: str | Path
    ) -> tuple[Path, BinaryIO]:
        """Atomically validate and open a file for import.

        This method prevents TOCTOU (Time-of-Check-Time-of-Use) race
        conditions by opening the file FIRST, then validating properties
        on the open file descriptor.

        IMPORTANT: The caller MUST use the returned file handle for reading.
        DO NOT re-open the file by path after this call.

        Args:
            path: The path to validate and open.

        Returns:
            Tuple of (canonical_path, file_handle). The file handle is
            opened in binary read mode ('rb'). Caller is responsible
            for closing it.

        Raises:
            PathSecurityError: If the path fails any security check.
            FileSizeLimitError: If the file exceeds the size limit.
            ValueError: If the path is empty or contains null bytes.

        Example:
            path, handle = manager.validate_and_open_import("./imports/data.json")
            try:
                data = handle.read()
                # Process data...
            finally:
                handle.close()
        """
        return self._validator.validate_and_open_import_file(
            path, max_size_bytes=self._config.max_import_size_bytes
        )

    def validate_import_records(
        self,
        records: list[dict[str, object]],
        expected_vector_dim: int,
        fail_fast: bool = False,
    ) -> BatchValidationResult:
        """Validate a batch of import records for schema compliance.

        Validates each record against the import schema:
        - Required fields (content)
        - Optional field types (namespace, tags, importance, metadata, vector)
        - Vector dimension matching
        - Value constraints (importance range, etc.)

        Args:
            records: List of record dictionaries to validate.
            expected_vector_dim: Expected vector dimensions.
            fail_fast: If True, stop on first error.

        Returns:
            BatchValidationResult with validation statistics and errors.

        Example:
            result = manager.validate_import_records(records, expected_dim=384)
            if not result.is_valid:
                for idx, errors in result.errors:
                    print(f"Record {idx} errors: {errors}")
        """
        config = ImportValidationConfig(
            expected_vector_dim=expected_vector_dim,
            fail_fast=fail_fast,
            max_errors=100,
        )
        validator = ImportValidator(config)
        return validator.validate_batch(records, expected_vector_dim)

    def is_path_safe(self, path: str | Path, operation: str = "export") -> bool:
        """Check if a path passes security validation.

        Convenience method that returns a boolean instead of raising.
        Useful for pre-flight checks in UI code.

        Args:
            path: The path to check.
            operation: Either "export" or "import".

        Returns:
            True if the path passes all security checks, False otherwise.

        Example:
            if manager.is_path_safe(user_input, "export"):
                # Proceed with export
                pass
            else:
                # Show error to user
                pass
        """
        try:
            if operation == "export":
                self.validate_export_path(path)
            elif operation == "import":
                self.validate_import_path(path)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            return True
        except (PathSecurityError, FileSizeLimitError, ValueError):
            return False


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_export_path(
    path: str | Path,
    config: SecurityConfig | None = None,
) -> Path:
    """Convenience function to validate an export path.

    Creates a FileSecurityManager with the provided config and validates
    the path. For repeated validations, prefer creating a manager instance.

    Args:
        path: The path to validate.
        config: Security configuration (uses defaults if not provided).

    Returns:
        Canonicalized Path object that is safe to use.

    Raises:
        PathSecurityError: If the path fails security checks.
    """
    manager = FileSecurityManager(config)
    return manager.validate_export_path(path)


def validate_import_path(
    path: str | Path,
    config: SecurityConfig | None = None,
) -> Path:
    """Convenience function to validate an import path.

    Creates a FileSecurityManager with the provided config and validates
    the path. For repeated validations, prefer creating a manager instance.

    Args:
        path: The path to validate.
        config: Security configuration (uses defaults if not provided).

    Returns:
        Canonicalized Path object that is safe to use.

    Raises:
        PathSecurityError: If the path fails security checks.
        FileSizeLimitError: If the file exceeds size limits.
    """
    manager = FileSecurityManager(config)
    return manager.validate_import_path(path)


def validate_import_file(
    path: str | Path,
    config: SecurityConfig | None = None,
) -> tuple[Path, BinaryIO]:
    """Convenience function to validate and open an import file atomically.

    This is the TOCTOU-safe way to open import files. The returned file
    handle MUST be used for reading - do not re-open by path.

    Args:
        path: The path to validate and open.
        config: Security configuration (uses defaults if not provided).

    Returns:
        Tuple of (canonical_path, file_handle).

    Raises:
        PathSecurityError: If the path fails security checks.
        FileSizeLimitError: If the file exceeds size limits.
    """
    manager = FileSecurityManager(config)
    return manager.validate_and_open_import(path)


def validate_import_records(
    records: list[dict[str, object]],
    expected_vector_dim: int,
    fail_fast: bool = False,
) -> BatchValidationResult:
    """Convenience function to validate import records.

    Args:
        records: List of record dictionaries to validate.
        expected_vector_dim: Expected vector dimensions.
        fail_fast: If True, stop on first error.

    Returns:
        BatchValidationResult with validation statistics.
    """
    config = ImportValidationConfig(
        expected_vector_dim=expected_vector_dim,
        fail_fast=fail_fast,
        max_errors=100,
    )
    validator = ImportValidator(config)
    return validator.validate_batch(records, expected_vector_dim)


# =============================================================================
# Factory Function for Settings Integration
# =============================================================================


def create_security_manager_from_settings() -> FileSecurityManager:
    """Create a FileSecurityManager from application settings.

    Reads security configuration from the application Settings object
    (environment variables or .env file).

    Returns:
        Configured FileSecurityManager instance.

    Example:
        manager = create_security_manager_from_settings()
        safe_path = manager.validate_export_path(user_path)
    """
    from spatial_memory.config import get_settings

    settings = get_settings()

    config = SecurityConfig(
        export_allowed_paths=tuple(settings.export_allowed_paths),
        import_allowed_paths=tuple(settings.import_allowed_paths),
        max_import_size_mb=settings.import_max_file_size_mb,
        allow_symlinks=settings.export_allow_symlinks or settings.import_allow_symlinks,
        max_import_records=settings.import_max_records,
    )

    return FileSecurityManager(config)
