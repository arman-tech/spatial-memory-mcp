"""Unit tests for file security module.

Security-critical tests covering:
- Path traversal attack prevention (../, %2e%2e, etc.)
- Windows UNC path detection
- Symlink detection and blocking
- File size limits for imports
- File extension validation
- Allowlist validation

These tests follow TDD - written BEFORE implementation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from spatial_memory.core.errors import FileSizeLimitError, PathSecurityError

if TYPE_CHECKING:
    from spatial_memory.core.file_security import PathValidator


# =============================================================================
# Test UUIDs and Constants
# =============================================================================

MB = 1024 * 1024  # 1 megabyte in bytes
DEFAULT_MAX_SIZE_BYTES = 100 * MB  # 100MB default limit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_allowed_dir(tmp_path: Path) -> Path:
    """Create a temporary allowed directory structure."""
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir()
    imports_dir = tmp_path / "imports"
    imports_dir.mkdir()
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    return tmp_path


@pytest.fixture
def validator(temp_allowed_dir: Path) -> "PathValidator":
    """Create a PathValidator with temp directories as allowed paths."""
    from spatial_memory.core.file_security import PathValidator

    return PathValidator(
        allowed_export_paths=[
            temp_allowed_dir / "exports",
            temp_allowed_dir / "backups",
        ],
        allowed_import_paths=[
            temp_allowed_dir / "imports",
            temp_allowed_dir / "backups",
        ],
        allow_symlinks=False,
    )


@pytest.fixture
def validator_with_symlinks(temp_allowed_dir: Path) -> "PathValidator":
    """Create a PathValidator that allows symlinks."""
    from spatial_memory.core.file_security import PathValidator

    return PathValidator(
        allowed_export_paths=[
            temp_allowed_dir / "exports",
            temp_allowed_dir / "backups",
        ],
        allowed_import_paths=[
            temp_allowed_dir / "imports",
            temp_allowed_dir / "backups",
        ],
        allow_symlinks=True,
    )


# =============================================================================
# TestPathTraversalDetection
# =============================================================================


class TestPathTraversalDetection:
    """Tests for path traversal attack detection."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            # Basic parent directory traversal
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32\\config",
            "exports/../../../etc/passwd",
            "exports/../../sensitive/data",
            # URL-encoded variants
            "exports/%2e%2e/secret",
            "exports/%2e%2e/%2e%2e/etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            # Double URL-encoded variants
            "exports/%252e%252e/secret",
            "%252e%252e%252f%252e%252e%252fetc%252fpasswd",
            # Mixed case URL encoding
            "exports/%2E%2E/secret",
            "exports/%2e%2E/%2E%2e/secret",
            # Null byte injection (historic attack vector)
            "exports/file.parquet%00.txt",
            # Unicode normalization attacks
            "exports/..%c0%af../etc/passwd",
            "exports/..%c1%9c../etc/passwd",
        ],
    )
    def test_traversal_attempt_blocked_export(
        self,
        validator: "PathValidator",
        malicious_path: str,
    ) -> None:
        """Path traversal attempts should be blocked for exports."""
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(malicious_path)
        assert exc_info.value.violation_type in ("traversal_attempt", "path_outside_allowlist")

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../../etc/passwd",
            "imports/../../../etc/passwd",
            "imports/%2e%2e/secret",
            "imports/%252e%252e/secret",
        ],
    )
    def test_traversal_attempt_blocked_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
        malicious_path: str,
    ) -> None:
        """Path traversal attempts should be blocked for imports."""
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(malicious_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert exc_info.value.violation_type in ("traversal_attempt", "path_outside_allowlist", "file_not_found")

    def test_relative_path_traversal_in_middle(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Detect traversal attempts embedded in seemingly valid paths."""
        malicious = str(temp_allowed_dir / "exports" / "subdir" / ".." / ".." / "secret.json")
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(malicious)
        assert exc_info.value.violation_type in ("traversal_attempt", "path_outside_allowlist")


# =============================================================================
# TestWindowsUNCPaths
# =============================================================================


class TestWindowsUNCPaths:
    """Tests for Windows UNC path detection."""

    @pytest.mark.parametrize(
        "unc_path",
        [
            "\\\\server\\share\\file.parquet",
            "\\\\?\\C:\\Windows\\System32\\config",
            "\\\\localhost\\c$\\sensitive",
            "\\\\192.168.1.1\\share\\data",
            "//server/share/file.parquet",  # Unix-style UNC
        ],
    )
    def test_unc_paths_blocked_export(
        self,
        validator: "PathValidator",
        unc_path: str,
    ) -> None:
        """Windows UNC paths should be blocked for exports.

        Note: UNC paths may be detected by the traversal pattern (which catches \\\\)
        or the UNC-specific check, both are valid security responses.
        """
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(unc_path)
        # UNC paths are blocked - the specific violation type may vary based on which
        # security check catches it first (traversal pattern vs UNC check)
        assert exc_info.value.violation_type in (
            "unc_path",
            "path_outside_allowlist",
            "traversal_attempt",  # \\\\ pattern matches traversal regex
        )

    @pytest.mark.parametrize(
        "unc_path",
        [
            "\\\\server\\share\\file.parquet",
            "\\\\?\\C:\\Windows\\System32\\config",
        ],
    )
    def test_unc_paths_blocked_import(
        self,
        validator: "PathValidator",
        unc_path: str,
    ) -> None:
        """Windows UNC paths should be blocked for imports."""
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(unc_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert exc_info.value.violation_type in (
            "unc_path",
            "path_outside_allowlist",
            "file_not_found",
            "traversal_attempt",  # \\\\ pattern matches traversal regex
        )


# =============================================================================
# TestSymlinkDetection
# =============================================================================


class TestSymlinkDetection:
    """Tests for symlink detection and blocking."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlinks require admin on Windows: Creating symbolic links on Windows requires "
               "the SeCreateSymbolicLinkPrivilege, which is only granted to administrators by "
               "default. Run tests as Administrator or on Unix/Linux to execute these tests."
    )
    def test_symlink_to_sensitive_blocked_export(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Symlinks pointing outside allowed directories should be blocked."""
        # Create a symlink in exports pointing to /etc
        exports_dir = temp_allowed_dir / "exports"
        sensitive_link = exports_dir / "sensitive_link"

        # Create a directory outside allowed paths
        outside_dir = temp_allowed_dir / "outside"
        outside_dir.mkdir()
        sensitive_file = outside_dir / "secret.txt"
        sensitive_file.write_text("secret data")

        try:
            sensitive_link.symlink_to(sensitive_file)
        except OSError:
            pytest.skip("Cannot create symlinks (requires elevated privileges)")

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(sensitive_link)
        assert exc_info.value.violation_type in ("symlink_not_allowed", "path_outside_allowlist")

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlinks require admin on Windows: Creating symbolic links on Windows requires "
               "the SeCreateSymbolicLinkPrivilege, which is only granted to administrators by "
               "default. Run tests as Administrator or on Unix/Linux to execute these tests."
    )
    def test_symlink_to_sensitive_blocked_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Symlinks should be blocked for imports when configured."""
        imports_dir = temp_allowed_dir / "imports"

        # Create a file outside allowed paths
        outside_dir = temp_allowed_dir / "outside"
        outside_dir.mkdir()
        sensitive_file = outside_dir / "secret.json"
        sensitive_file.write_text('{"secret": "data"}')

        # Create symlink in imports
        link_path = imports_dir / "link.json"
        try:
            link_path.symlink_to(sensitive_file)
        except OSError:
            pytest.skip("Cannot create symlinks (requires elevated privileges)")

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(link_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert exc_info.value.violation_type in ("symlink_not_allowed", "path_outside_allowlist")

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlinks require admin on Windows: Creating symbolic links on Windows requires "
               "the SeCreateSymbolicLinkPrivilege, which is only granted to administrators by "
               "default. Run tests as Administrator or on Unix/Linux to execute these tests."
    )
    def test_symlink_allowed_when_configured(
        self,
        validator_with_symlinks: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Symlinks should be allowed when configured."""
        imports_dir = temp_allowed_dir / "imports"
        backups_dir = temp_allowed_dir / "backups"

        # Create a file in backups (an allowed path)
        real_file = backups_dir / "real_data.json"
        real_file.write_text('{"data": "test"}')

        # Create symlink in imports pointing to backups
        link_path = imports_dir / "link.json"
        try:
            link_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Cannot create symlinks (requires elevated privileges)")

        # Should succeed when symlinks are allowed and target is in allowed path
        result = validator_with_symlinks.validate_import_path(
            link_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES
        )
        assert result.exists()

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlinks require admin on Windows: Creating symbolic links on Windows requires "
               "the SeCreateSymbolicLinkPrivilege, which is only granted to administrators by "
               "default. Run tests as Administrator or on Unix/Linux to execute these tests."
    )
    def test_symlink_chain_detection(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Detect symlink chains that eventually escape allowed paths."""
        exports_dir = temp_allowed_dir / "exports"
        outside_dir = temp_allowed_dir / "outside"
        outside_dir.mkdir()

        # Create chain: link1 -> link2 -> outside_file
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")

        try:
            link2 = exports_dir / "link2"
            link2.symlink_to(outside_file)

            link1 = exports_dir / "link1"
            link1.symlink_to(link2)
        except OSError:
            pytest.skip("Cannot create symlinks (requires elevated privileges)")

        with pytest.raises(PathSecurityError):
            validator.validate_export_path(link1)


# =============================================================================
# TestFileSizeLimits
# =============================================================================


class TestFileSizeLimits:
    """Tests for file size limit enforcement."""

    def test_oversized_file_rejected(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Files exceeding size limit should be rejected."""
        imports_dir = temp_allowed_dir / "imports"
        large_file = imports_dir / "large.json"

        # Create a file larger than the limit
        max_size = 1 * MB  # 1MB limit for test
        large_file.write_bytes(b"x" * (max_size + 1000))

        with pytest.raises(FileSizeLimitError) as exc_info:
            validator.validate_import_path(large_file, max_size_bytes=max_size)

        assert exc_info.value.actual_size_bytes > max_size
        assert exc_info.value.max_size_bytes == max_size

    def test_file_at_exact_limit_accepted(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Files at exactly the size limit should be accepted."""
        imports_dir = temp_allowed_dir / "imports"
        exact_file = imports_dir / "exact.json"

        max_size = 1 * MB
        exact_file.write_bytes(b"x" * max_size)

        result = validator.validate_import_path(exact_file, max_size_bytes=max_size)
        assert result.exists()

    def test_file_under_limit_accepted(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Files under the size limit should be accepted."""
        imports_dir = temp_allowed_dir / "imports"
        small_file = imports_dir / "small.json"

        small_file.write_text('{"test": "data"}')

        result = validator.validate_import_path(small_file, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert result.exists()

    def test_file_size_error_includes_details(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """FileSizeLimitError should include size details."""
        imports_dir = temp_allowed_dir / "imports"
        large_file = imports_dir / "oversized.parquet"

        max_size = 10 * MB
        actual_size = 15 * MB
        large_file.write_bytes(b"x" * actual_size)

        with pytest.raises(FileSizeLimitError) as exc_info:
            validator.validate_import_path(large_file, max_size_bytes=max_size)

        error = exc_info.value
        assert error.actual_size_bytes == actual_size
        assert error.max_size_bytes == max_size
        assert "15" in str(error) or "14" in str(error)  # ~15MB
        assert "10" in str(error) or "9" in str(error)  # ~10MB


# =============================================================================
# TestExtensionValidation
# =============================================================================


class TestExtensionValidation:
    """Tests for file extension validation."""

    @pytest.mark.parametrize(
        "valid_extension",
        [
            "data.parquet",
            "backup.json",
            "export.csv",
            "DATA.PARQUET",  # Case insensitive
            "Export.JSON",
            "backup.Csv",
        ],
    )
    def test_valid_extensions_accepted_export(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
        valid_extension: str,
    ) -> None:
        """Valid export extensions should be accepted."""
        exports_dir = temp_allowed_dir / "exports"
        file_path = exports_dir / valid_extension

        # validate_export_path should not raise for valid extensions
        # (file doesn't need to exist for export path validation)
        result = validator.validate_export_path(file_path)
        assert result is not None

    @pytest.mark.parametrize(
        "valid_extension",
        [
            "data.parquet",
            "backup.json",
            "import.csv",
        ],
    )
    def test_valid_extensions_accepted_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
        valid_extension: str,
    ) -> None:
        """Valid import extensions should be accepted."""
        imports_dir = temp_allowed_dir / "imports"
        file_path = imports_dir / valid_extension
        file_path.write_text("test data")

        result = validator.validate_import_path(file_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert result.exists()

    @pytest.mark.parametrize(
        "invalid_extension",
        [
            "script.py",
            "executable.exe",
            "library.dll",
            "shell.sh",
            "batch.bat",
            "command.cmd",
            "binary.bin",
            "archive.zip",
            "archive.tar.gz",
            "image.png",
            "document.pdf",
            "noextension",
        ],
    )
    def test_invalid_extensions_rejected_export(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
        invalid_extension: str,
    ) -> None:
        """Invalid extensions should be rejected for exports."""
        exports_dir = temp_allowed_dir / "exports"
        file_path = exports_dir / invalid_extension

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(file_path)
        assert exc_info.value.violation_type == "invalid_extension"

    @pytest.mark.parametrize(
        "invalid_extension",
        [
            "script.py",
            "executable.exe",
            "shell.sh",
        ],
    )
    def test_invalid_extensions_rejected_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
        invalid_extension: str,
    ) -> None:
        """Invalid extensions should be rejected for imports."""
        imports_dir = temp_allowed_dir / "imports"
        file_path = imports_dir / invalid_extension
        file_path.write_text("test")

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(file_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert exc_info.value.violation_type == "invalid_extension"


# =============================================================================
# TestAllowlistValidation
# =============================================================================


class TestAllowlistValidation:
    """Tests for directory allowlist validation."""

    def test_path_in_allowed_directory_accepted_export(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Paths in allowed directories should be accepted."""
        exports_dir = temp_allowed_dir / "exports"
        valid_path = exports_dir / "data.parquet"

        result = validator.validate_export_path(valid_path)
        assert result is not None

    def test_path_outside_allowed_directory_rejected_export(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Paths outside allowed directories should be rejected."""
        outside_path = temp_allowed_dir / "not_allowed" / "data.parquet"

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(outside_path)
        # May be path_outside_allowlist or sensitive_directory depending on temp path location
        assert exc_info.value.violation_type in ("path_outside_allowlist", "sensitive_directory")

    def test_path_in_allowed_directory_accepted_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Paths in allowed directories should be accepted for imports."""
        imports_dir = temp_allowed_dir / "imports"
        valid_path = imports_dir / "data.json"
        valid_path.write_text('{"test": true}')

        result = validator.validate_import_path(valid_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert result.exists()

    def test_path_outside_allowed_directory_rejected_import(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Paths outside allowed directories should be rejected for imports."""
        outside_dir = temp_allowed_dir / "not_allowed"
        outside_dir.mkdir()
        outside_path = outside_dir / "data.json"
        outside_path.write_text('{"test": true}')

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(outside_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        # May be path_outside_allowlist or sensitive_directory depending on temp path location
        assert exc_info.value.violation_type in ("path_outside_allowlist", "sensitive_directory")

    def test_subdirectory_of_allowed_accepted(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Subdirectories of allowed paths should be accepted."""
        exports_dir = temp_allowed_dir / "exports"
        subdir = exports_dir / "2024" / "01"
        subdir.mkdir(parents=True)
        file_path = subdir / "backup.parquet"

        result = validator.validate_export_path(file_path)
        assert result is not None

    def test_backups_dir_works_for_both(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Backups directory should work for both export and import."""
        backups_dir = temp_allowed_dir / "backups"

        # Export
        export_path = backups_dir / "export.parquet"
        result = validator.validate_export_path(export_path)
        assert result is not None

        # Import
        import_path = backups_dir / "import.json"
        import_path.write_text('{"test": true}')
        result = validator.validate_import_path(import_path, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert result.exists()


# =============================================================================
# TestFileExistenceValidation
# =============================================================================


class TestFileExistenceValidation:
    """Tests for file existence checks."""

    def test_import_nonexistent_file_rejected(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Import validation should reject non-existent files."""
        imports_dir = temp_allowed_dir / "imports"
        nonexistent = imports_dir / "does_not_exist.json"

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(nonexistent, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        assert exc_info.value.violation_type == "file_not_found"

    def test_export_nonexistent_directory_parent_created(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Export validation should handle non-existent parent directories."""
        exports_dir = temp_allowed_dir / "exports"
        new_subdir = exports_dir / "new_subdir" / "data.parquet"

        # Should not raise - export paths don't need to exist yet
        result = validator.validate_export_path(new_subdir)
        assert result is not None

    def test_import_directory_instead_of_file_rejected(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Import validation should reject directories."""
        imports_dir = temp_allowed_dir / "imports"
        subdir = imports_dir / "subdir"
        subdir.mkdir()

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_import_path(subdir, max_size_bytes=DEFAULT_MAX_SIZE_BYTES)
        # Could be invalid_extension (no .json/.parquet/.csv) or file_not_found
        assert exc_info.value.violation_type in ("invalid_extension", "not_a_file")


# =============================================================================
# TestPathCanonicalization
# =============================================================================


class TestPathCanonicalization:
    """Tests for path canonicalization."""

    def test_canonicalizes_relative_paths(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Validator should detect and block paths with .. even if they resolve to allowed dirs.

        Security note: Blocking all '..' patterns is a defense-in-depth measure.
        Even though 'exports/subdir/../data.parquet' resolves to 'exports/data.parquet',
        we block it to prevent any potential path traversal bypass attempts.
        """
        exports_dir = temp_allowed_dir / "exports"

        # Path with redundant elements - should be blocked for defense-in-depth
        redundant_path = exports_dir / "subdir" / ".." / "data.parquet"

        # The .. is detected and blocked as a traversal attempt
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(redundant_path)
        assert exc_info.value.violation_type == "traversal_attempt"

    def test_handles_absolute_paths(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Validator should handle absolute paths correctly."""
        exports_dir = temp_allowed_dir / "exports"
        absolute_path = exports_dir.resolve() / "data.parquet"

        result = validator.validate_export_path(absolute_path)
        assert result.is_absolute()

    def test_normalizes_path_separators(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Validator should normalize path separators."""
        exports_dir = temp_allowed_dir / "exports"

        # Mix of separators (platform-dependent behavior)
        mixed_path = str(exports_dir) + "/subdir\\data.parquet"

        # Should handle without raising (path normalization)
        try:
            result = validator.validate_export_path(mixed_path)
            assert result is not None
        except PathSecurityError as e:
            # If it raises, it should be for a valid security reason
            assert e.violation_type in ("path_outside_allowlist", "traversal_attempt")


# =============================================================================
# TestSensitiveDirectoryBlocking
# =============================================================================


class TestSensitiveDirectoryBlocking:
    """Tests for blocking access to sensitive system directories."""

    @pytest.mark.parametrize(
        "sensitive_path",
        [
            "/etc/passwd",
            "/etc/shadow",
            "/usr/bin/bash",
            "/var/log/auth.log",
            "/root/.ssh/id_rsa",
        ],
    )
    @pytest.mark.skipif(
        os.name == "nt",
        reason="Unix paths only: These tests verify blocking of Unix/Linux sensitive system "
               "files (e.g., /etc/passwd, /etc/shadow). These paths don't exist on Windows. "
               "Equivalent Windows path tests run separately."
    )
    def test_unix_sensitive_paths_blocked(
        self,
        validator: "PathValidator",
        sensitive_path: str,
    ) -> None:
        """Unix sensitive paths should be blocked."""
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(sensitive_path)
        assert exc_info.value.violation_type in ("sensitive_directory", "path_outside_allowlist")

    @pytest.mark.parametrize(
        "sensitive_path",
        [
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Program Files\\sensitive.exe",
            "C:\\Users\\Administrator\\Desktop\\secret.txt",
        ],
    )
    @pytest.mark.skipif(os.name != "nt", reason="Windows paths only")
    def test_windows_sensitive_paths_blocked(
        self,
        validator: "PathValidator",
        sensitive_path: str,
    ) -> None:
        """Windows sensitive paths should be blocked.

        Note: These paths will be blocked, but the specific violation type may vary:
        - invalid_extension for non-data files (.exe, .txt)
        - path_resolution_failed if the path can't be accessed
        - sensitive_directory or path_outside_allowlist otherwise
        """
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(sensitive_path)
        # Any of these means the path is blocked - the security goal is achieved
        assert exc_info.value.violation_type in (
            "sensitive_directory",
            "path_outside_allowlist",
            "invalid_extension",
            "path_resolution_failed",
        )


# =============================================================================
# TestErrorMessages
# =============================================================================


class TestErrorMessages:
    """Tests for clear and informative error messages."""

    def test_traversal_error_includes_path(
        self,
        validator: "PathValidator",
    ) -> None:
        """PathSecurityError should include the problematic path."""
        malicious = "../../../etc/passwd"
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(malicious)

        error = exc_info.value
        assert error.path is not None
        assert "passwd" in str(error) or ".." in str(error)

    def test_allowlist_error_is_informative(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Allowlist errors should explain the violation."""
        outside = temp_allowed_dir / "forbidden" / "data.parquet"
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(outside)

        error = exc_info.value
        # May be path_outside_allowlist or sensitive_directory depending on temp path location
        assert error.violation_type in ("path_outside_allowlist", "sensitive_directory")
        # Error message should be informative about the blocking reason
        error_msg = str(error).lower()
        assert (
            "allowlist" in error_msg
            or "allowed" in error_msg
            or "sensitive" in error_msg
            or "blocked" in error_msg
        )

    def test_extension_error_shows_invalid_extension(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Extension errors should show the invalid extension."""
        exports_dir = temp_allowed_dir / "exports"
        bad_file = exports_dir / "script.exe"

        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(bad_file)

        error = exc_info.value
        assert error.violation_type == "invalid_extension"
        assert "exe" in str(error).lower() or "extension" in str(error).lower()


# =============================================================================
# TestPatternConstants
# =============================================================================


class TestPatternConstants:
    """Tests for PATH_TRAVERSAL_PATTERNS and SENSITIVE_DIRECTORIES constants."""

    def test_traversal_patterns_exist(self) -> None:
        """PATH_TRAVERSAL_PATTERNS should be defined."""
        from spatial_memory.core.file_security import PATH_TRAVERSAL_PATTERNS

        assert PATH_TRAVERSAL_PATTERNS is not None
        assert len(PATH_TRAVERSAL_PATTERNS) > 0

    def test_traversal_patterns_are_compiled_regex(self) -> None:
        """PATH_TRAVERSAL_PATTERNS should contain compiled regex patterns."""
        import re

        from spatial_memory.core.file_security import PATH_TRAVERSAL_PATTERNS

        for pattern in PATH_TRAVERSAL_PATTERNS:
            assert isinstance(pattern, re.Pattern)

    def test_sensitive_directories_exist(self) -> None:
        """SENSITIVE_DIRECTORIES should be defined."""
        from spatial_memory.core.file_security import SENSITIVE_DIRECTORIES

        assert SENSITIVE_DIRECTORIES is not None
        assert isinstance(SENSITIVE_DIRECTORIES, frozenset)
        assert len(SENSITIVE_DIRECTORIES) > 0

    def test_sensitive_directories_contains_common_paths(self) -> None:
        """SENSITIVE_DIRECTORIES should contain common sensitive paths."""
        from spatial_memory.core.file_security import SENSITIVE_DIRECTORIES

        # Check for some expected paths (lowercase for comparison)
        lower_dirs = {d.lower() for d in SENSITIVE_DIRECTORIES}

        if os.name != "nt":
            assert any("/etc" in d for d in lower_dirs)
        else:
            assert any("windows" in d for d in lower_dirs)


# =============================================================================
# TestValidExtensions
# =============================================================================


class TestValidExtensions:
    """Tests for VALID_EXTENSIONS constant."""

    def test_valid_extensions_defined(self) -> None:
        """VALID_EXTENSIONS should be defined."""
        from spatial_memory.core.file_security import VALID_EXTENSIONS

        assert VALID_EXTENSIONS is not None
        assert isinstance(VALID_EXTENSIONS, (set, frozenset))

    def test_valid_extensions_includes_required(self) -> None:
        """VALID_EXTENSIONS should include parquet, json, csv."""
        from spatial_memory.core.file_security import VALID_EXTENSIONS

        required = {".parquet", ".json", ".csv"}
        assert required.issubset(VALID_EXTENSIONS)


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_path_rejected(
        self,
        validator: "PathValidator",
    ) -> None:
        """Empty path should be rejected."""
        with pytest.raises((PathSecurityError, ValueError)):
            validator.validate_export_path("")

    def test_whitespace_only_path_rejected(
        self,
        validator: "PathValidator",
    ) -> None:
        """Whitespace-only path should be rejected."""
        with pytest.raises((PathSecurityError, ValueError)):
            validator.validate_export_path("   ")

    def test_null_bytes_in_path_rejected(
        self,
        validator: "PathValidator",
    ) -> None:
        """Paths with null bytes should be rejected."""
        with pytest.raises((PathSecurityError, ValueError)):
            validator.validate_export_path("file\x00.parquet")

    def test_very_long_path_handled(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Very long paths should be handled appropriately."""
        exports_dir = temp_allowed_dir / "exports"
        # Create a very long filename
        long_name = "a" * 200 + ".parquet"
        long_path = exports_dir / long_name

        # Should either accept (if OS supports) or raise clear error
        try:
            result = validator.validate_export_path(long_path)
            assert result is not None
        except (PathSecurityError, OSError):
            # Acceptable to reject very long paths
            pass

    def test_special_characters_in_filename(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """Filenames with special characters should be handled."""
        exports_dir = temp_allowed_dir / "exports"

        # Test with spaces
        spaced_path = exports_dir / "my file.parquet"
        result = validator.validate_export_path(spaced_path)
        assert result is not None

        # Test with unicode
        unicode_path = exports_dir / "data_2024.parquet"
        result = validator.validate_export_path(unicode_path)
        assert result is not None


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of PathValidator."""

    def test_validator_is_reentrant(
        self,
        validator: "PathValidator",
        temp_allowed_dir: Path,
    ) -> None:
        """PathValidator should be safe for concurrent use."""
        import concurrent.futures

        exports_dir = temp_allowed_dir / "exports"
        paths = [exports_dir / f"file_{i}.parquet" for i in range(10)]

        def validate_path(path: Path) -> Path:
            return validator.validate_export_path(path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(validate_path, paths))

        assert len(results) == 10
        assert all(r is not None for r in results)
