"""File security module for path validation and attack prevention.

This module provides security-critical path validation to prevent:
- Path traversal attacks (../, %2e%2e, etc.)
- Windows UNC path attacks
- Symlink-based escapes from allowed directories
- File size limit bypass
- Invalid file extension attacks

Security is implemented through defense-in-depth:
1. Pattern-based detection of known attack vectors
2. Path canonicalization to resolve symbolic elements
3. Allowlist validation to restrict accessible directories
4. Extension validation to limit file types
5. Symlink resolution and validation
"""

from __future__ import annotations

import errno
import os
import re
import stat
import urllib.parse
from collections.abc import Sequence
from io import BufferedReader
from pathlib import Path
from typing import BinaryIO

from spatial_memory.core.errors import FileSizeLimitError, PathSecurityError

# =============================================================================
# Security Constants
# =============================================================================

# Regex patterns to detect path traversal attempts
# These patterns detect various encoding schemes used to bypass filters
PATH_TRAVERSAL_PATTERNS: list[re.Pattern[str]] = [
    # Basic parent directory traversal
    re.compile(r"\.\."),
    # URL-encoded .. (%2e = '.')
    re.compile(r"%2e%2e", re.IGNORECASE),
    # Double URL-encoded .. (%252e = '%2e')
    re.compile(r"%252e%252e", re.IGNORECASE),
    # Windows UNC paths (\\server\share or \\?\)
    re.compile(r"^\\\\"),
    # Unix-style UNC paths (//server/share)
    re.compile(r"^//"),
    # Null byte injection (historic attack, blocked by modern OSes but still checked)
    re.compile(r"%00|\x00"),
    # Overlong UTF-8 encoding of '.' (CVE-2000-0884 style)
    re.compile(r"%c0%ae|%c0%2e|%c1%9c", re.IGNORECASE),
]

# Sensitive system directories that should never be accessible
# These are common targets for path traversal attacks
SENSITIVE_DIRECTORIES: frozenset[str] = frozenset(
    {
        # Unix/Linux sensitive directories
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/var/log",
        "/root",
        "/home",
        "/tmp",
        "/var/tmp",
        "/proc",
        "/sys",
        "/dev",
        # macOS specific
        "/System",
        "/Library",
        "/private",
        # Windows sensitive directories
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
        "C:\\Users",
        "C:\\System32",
        "C:\\SysWOW64",
    }
)

# Valid file extensions for export/import operations
# Only data formats are allowed - no executables or scripts
VALID_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".parquet",
        ".json",
        ".csv",
    }
)

# Maximum number of URL decode iterations to catch double/triple encoding attacks
# Three passes catches: single encoding (%2e), double (%252e), and triple (%25252e)
MAX_URL_DECODE_ITERATIONS = 3


# =============================================================================
# PathValidator Class
# =============================================================================


class PathValidator:
    """Validates file paths for security constraints.

    This class implements defense-in-depth path validation:
    1. Detects path traversal patterns in raw input
    2. Canonicalizes paths to resolve symbolic elements
    3. Validates against allowed directories (allowlist)
    4. Validates file extensions
    5. Detects and optionally blocks symlinks

    Thread Safety: This class is thread-safe. All methods are stateless
    and only read from immutable configuration.

    Example:
        validator = PathValidator(
            allowed_export_paths=[Path("/data/exports")],
            allowed_import_paths=[Path("/data/imports")],
        )

        # Validate export path
        safe_path = validator.validate_export_path("/data/exports/backup.parquet")

        # Validate import path with size check
        safe_path = validator.validate_import_path(
            "/data/imports/restore.json",
            max_size_bytes=100 * 1024 * 1024,
        )
    """

    def __init__(
        self,
        allowed_export_paths: Sequence[str | Path],
        allowed_import_paths: Sequence[str | Path],
        allow_symlinks: bool = False,
    ) -> None:
        """Initialize the PathValidator.

        Args:
            allowed_export_paths: Directories where exports are permitted.
            allowed_import_paths: Directories where imports are permitted.
            allow_symlinks: Whether to allow following symlinks. Default False
                for security - symlinks can be used to escape allowed directories.
        """
        # Convert and resolve allowed paths to absolute paths
        self._allowed_export_paths: tuple[Path, ...] = tuple(
            Path(p).resolve() for p in allowed_export_paths
        )
        self._allowed_import_paths: tuple[Path, ...] = tuple(
            Path(p).resolve() for p in allowed_import_paths
        )
        self._allow_symlinks = allow_symlinks

    def validate_export_path(self, path: str | Path) -> Path:
        """Validate a path for export operations.

        Performs security checks without requiring the file to exist.
        Parent directories will be created if needed during export.

        Args:
            path: The path to validate. Can be absolute or relative.

        Returns:
            Canonicalized Path object that is safe to use.

        Raises:
            PathSecurityError: If the path fails any security check.
            ValueError: If the path is empty or invalid.
        """
        # Basic input validation
        path_str = str(path).strip() if path else ""
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Check for null bytes
        if "\x00" in path_str:
            raise ValueError("Path cannot contain null bytes")

        # Step 1: Detect path traversal patterns in raw input
        self._check_traversal_patterns(path_str)

        # Step 2: Detect UNC paths
        self._check_unc_path(path_str)

        # Step 3: URL decode and check again (defense in depth)
        decoded = self._url_decode_path(path_str)
        if decoded != path_str:
            self._check_traversal_patterns(decoded)

        # Step 4: Convert to Path and canonicalize
        path_obj = Path(path_str)

        # Resolve without strict (file doesn't need to exist for export)
        # We resolve parents to detect traversal attempts
        try:
            # For non-existent paths, resolve what we can
            if path_obj.exists():
                canonical = path_obj.resolve()
            else:
                # Resolve existing parents, keep filename
                parent = path_obj.parent
                while not parent.exists() and parent != parent.parent:
                    parent = parent.parent
                if parent.exists():
                    resolved_parent = parent.resolve()
                    # Build the rest of the path
                    if parent != path_obj:
                        relative = path_obj.relative_to(parent)
                    else:
                        relative = Path(path_obj.name)
                    canonical = resolved_parent / relative
                else:
                    canonical = path_obj.absolute()
        except (OSError, ValueError) as e:
            raise PathSecurityError(
                path=path_str,
                violation_type="path_resolution_failed",
                message=f"Failed to resolve path: {e}",
            )

        # Step 5: Check for traversal in canonical path (defense in depth)
        canonical_str = str(canonical)
        if ".." in canonical_str:
            raise PathSecurityError(
                path=path_str,
                violation_type="traversal_attempt",
                message=f"Path contains traversal after canonicalization: {path_str}",
            )

        # Step 6: Validate extension
        self._validate_extension(canonical)

        # Step 7: Check symlink (if path exists)
        if canonical.exists() and not self._allow_symlinks:
            self._check_symlink(canonical, path_str)

        # Step 8: Validate against allowlist
        self._validate_allowlist(canonical, self._allowed_export_paths, path_str)

        return canonical

    def validate_import_path(self, path: str | Path, max_size_bytes: int) -> Path:
        """Validate a path for import operations.

        Performs all security checks and additionally verifies:
        - File exists
        - File is not a directory
        - File size is within limits

        Args:
            path: The path to validate. Can be absolute or relative.
            max_size_bytes: Maximum allowed file size in bytes.

        Returns:
            Canonicalized Path object that is safe to use.

        Raises:
            PathSecurityError: If the path fails any security check.
            FileSizeLimitError: If the file exceeds the size limit.
            ValueError: If the path is empty or invalid.
        """
        # Basic input validation
        path_str = str(path).strip() if path else ""
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Check for null bytes
        if "\x00" in path_str:
            raise ValueError("Path cannot contain null bytes")

        # Step 1: Detect path traversal patterns in raw input
        self._check_traversal_patterns(path_str)

        # Step 2: Detect UNC paths
        self._check_unc_path(path_str)

        # Step 3: URL decode and check again
        decoded = self._url_decode_path(path_str)
        if decoded != path_str:
            self._check_traversal_patterns(decoded)

        # Step 4: Convert to Path
        path_obj = Path(path_str)

        # Step 5: Check file exists
        if not path_obj.exists():
            raise PathSecurityError(
                path=path_str,
                violation_type="file_not_found",
                message=f"File does not exist: {path_str}",
            )

        # Step 6: Check it's a file, not a directory
        if path_obj.is_dir():
            raise PathSecurityError(
                path=path_str,
                violation_type="not_a_file",
                message=f"Path is a directory, not a file: {path_str}",
            )

        # Step 7: Canonicalize (resolve symlinks unless blocked)
        try:
            canonical = path_obj.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(
                path=path_str,
                violation_type="path_resolution_failed",
                message=f"Failed to resolve path: {e}",
            )

        # Step 8: Check for traversal in canonical path
        canonical_str = str(canonical)
        if ".." in canonical_str:
            raise PathSecurityError(
                path=path_str,
                violation_type="traversal_attempt",
                message=f"Path contains traversal after canonicalization: {path_str}",
            )

        # Step 9: Validate extension
        self._validate_extension(canonical)

        # Step 10: Check symlink
        if not self._allow_symlinks:
            self._check_symlink(path_obj, path_str)

        # Step 11: Validate against allowlist
        self._validate_allowlist(canonical, self._allowed_import_paths, path_str)

        # Step 12: Check file size
        try:
            file_size = canonical.stat().st_size
        except OSError as e:
            raise PathSecurityError(
                path=path_str,
                violation_type="stat_failed",
                message=f"Failed to get file size: {e}",
            )

        if file_size > max_size_bytes:
            raise FileSizeLimitError(
                path=path_str,
                actual_size_bytes=file_size,
                max_size_bytes=max_size_bytes,
            )

        return canonical

    def validate_and_open_import_file(
        self, path: str | Path, max_size_bytes: int
    ) -> tuple[Path, BinaryIO]:
        """Atomically validate and open a file for import.

        This method prevents TOCTOU (Time-of-Check-Time-of-Use) race conditions
        by opening the file FIRST, then validating properties on the open file
        descriptor. The caller MUST use the returned file handle for reading.

        Args:
            path: The path to validate and open.
            max_size_bytes: Maximum allowed file size in bytes.

        Returns:
            Tuple of (canonical_path, open_file_handle). The file handle is
            opened in binary read mode. Caller is responsible for closing it.

        Raises:
            PathSecurityError: If the path fails any security check.
            FileSizeLimitError: If the file exceeds the size limit.
            ValueError: If the path is empty or invalid.
        """
        # Basic input validation
        path_str = str(path).strip() if path else ""
        if not path_str:
            raise ValueError("Path cannot be empty")

        if "\x00" in path_str:
            raise ValueError("Path cannot contain null bytes")

        # Step 1: Detect path traversal patterns in raw input
        self._check_traversal_patterns(path_str)

        # Step 2: Detect UNC paths
        self._check_unc_path(path_str)

        # Step 3: URL decode and check again
        decoded = self._url_decode_path(path_str)
        if decoded != path_str:
            self._check_traversal_patterns(decoded)

        # Step 4: Validate extension BEFORE opening
        path_obj = Path(path_str)
        self._validate_extension(path_obj)

        # Step 5: Check basic allowlist BEFORE opening (path-based check)
        # We'll re-verify after opening but this catches obvious violations early
        try:
            preliminary_canonical = path_obj.resolve()
        except (OSError, RuntimeError):
            # Can't resolve - will fail when we try to open
            pass
        else:
            # Quick check that we're in allowed territory
            in_allowed = False
            for allowed in self._allowed_import_paths:
                try:
                    if preliminary_canonical.is_relative_to(allowed):
                        in_allowed = True
                        break
                except AttributeError:
                    try:
                        preliminary_canonical.relative_to(allowed)
                        in_allowed = True
                        break
                    except ValueError:
                        continue

            if not in_allowed:
                allowed_str = ", ".join(str(p) for p in self._allowed_import_paths)
                raise PathSecurityError(
                    path=path_str,
                    violation_type="path_outside_allowlist",
                    message=f"Path is not in allowed directories. Allowed: {allowed_str}",
                )

        # Step 5.5: Pre-check symlinks BEFORE opening (defense in depth)
        # This catches obvious symlinks before we open the file
        if not self._allow_symlinks:
            # Check the path itself and all parents
            self._check_symlink(path_obj, path_str)

        # Step 6: ATOMICALLY open the file using low-level os.open
        # This prevents TOCTOU - all subsequent checks use the open descriptor
        # On Unix, use O_NOFOLLOW to prevent opening through symlinks at OS level
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW") and not self._allow_symlinks:
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(str(path_obj), flags)
        except FileNotFoundError:
            # FileNotFoundError is a subclass of OSError, so catch it first
            raise PathSecurityError(
                path=path_str,
                violation_type="file_not_found",
                message=f"File does not exist: {path_str}",
            )
        except IsADirectoryError:
            raise PathSecurityError(
                path=path_str,
                violation_type="not_a_file",
                message=f"Path is a directory, not a file: {path_str}",
            )
        except PermissionError as e:
            raise PathSecurityError(
                path=path_str,
                violation_type="permission_denied",
                message=f"Permission denied: {e}",
            )
        except OSError as e:
            # O_NOFOLLOW causes ELOOP (or EMLINK on some systems) if path is a symlink
            if e.errno in (errno.ELOOP, getattr(errno, "EMLINK", None)):
                raise PathSecurityError(
                    path=path_str,
                    violation_type="symlink_not_allowed",
                    message=f"Symlinks are not allowed: {path_str}",
                )
            raise PathSecurityError(
                path=path_str,
                violation_type="open_failed",
                message=f"Failed to open file: {e}",
            )

        # From this point, we must close fd on any error
        try:
            # Step 7: Get file stats from the OPEN descriptor (not the path!)
            try:
                fd_stat = os.fstat(fd)
            except OSError as e:
                raise PathSecurityError(
                    path=path_str,
                    violation_type="stat_failed",
                    message=f"Failed to stat file: {e}",
                )

            # Step 8: Verify it's a regular file (not directory, device, etc.)
            if not stat.S_ISREG(fd_stat.st_mode):
                raise PathSecurityError(
                    path=path_str,
                    violation_type="not_a_regular_file",
                    message=f"Path is not a regular file: {path_str}",
                )

            # Step 9: Re-check symlink status after open (detect race conditions)
            # If a symlink appeared between our pre-check and open, detect it here
            if not self._allow_symlinks and path_obj.is_symlink():
                raise PathSecurityError(
                    path=path_str,
                    violation_type="symlink_race_detected",
                    message=f"Symlink detected after open (possible race condition): {path_str}",
                )

            # Step 10: Check file size using the open descriptor
            if fd_stat.st_size > max_size_bytes:
                raise FileSizeLimitError(
                    path=path_str,
                    actual_size_bytes=fd_stat.st_size,
                    max_size_bytes=max_size_bytes,
                )

            # Step 11: Resolve the canonical path for the file we actually opened
            # Use /proc/self/fd on Linux or os.path.realpath
            try:
                canonical = path_obj.resolve(strict=True)
            except (OSError, RuntimeError) as e:
                raise PathSecurityError(
                    path=path_str,
                    violation_type="path_resolution_failed",
                    message=f"Failed to resolve path: {e}",
                )

            # Step 12: Final allowlist validation on canonical path
            self._validate_allowlist(canonical, self._allowed_import_paths, path_str)

            # Step 13: Convert fd to a Python file object
            # The file object now owns the fd and will close it
            file_handle: BinaryIO = os.fdopen(fd, "rb")

            return canonical, file_handle

        except Exception:
            # Close fd on any error (before it's converted to file object)
            os.close(fd)
            raise

    def _check_traversal_patterns(self, path_str: str) -> None:
        """Check for path traversal patterns in the input string.

        Args:
            path_str: The path string to check.

        Raises:
            PathSecurityError: If a traversal pattern is detected.
        """
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if pattern.search(path_str):
                raise PathSecurityError(
                    path=path_str,
                    violation_type="traversal_attempt",
                    message=f"Path traversal pattern detected: {path_str}",
                )

    def _check_unc_path(self, path_str: str) -> None:
        """Check for Windows UNC paths.

        Args:
            path_str: The path string to check.

        Raises:
            PathSecurityError: If a UNC path is detected.
        """
        # Check for Windows UNC (\\server\share)
        if path_str.startswith("\\\\"):
            raise PathSecurityError(
                path=path_str,
                violation_type="unc_path",
                message=f"UNC paths are not allowed: {path_str}",
            )
        # Check for \\?\ prefix (Windows extended path)
        if path_str.startswith("\\\\?\\"):
            raise PathSecurityError(
                path=path_str,
                violation_type="unc_path",
                message=f"Extended UNC paths are not allowed: {path_str}",
            )
        # Check for Unix-style UNC (//server/share)
        if path_str.startswith("//"):
            raise PathSecurityError(
                path=path_str,
                violation_type="unc_path",
                message=f"UNC-style paths are not allowed: {path_str}",
            )

    def _url_decode_path(self, path_str: str) -> str:
        """URL decode a path string (multiple passes for double encoding).

        Args:
            path_str: The path string to decode.

        Returns:
            The decoded path string.
        """
        decoded = path_str
        # Multiple passes to catch double/triple encoding
        for _ in range(MAX_URL_DECODE_ITERATIONS):
            new_decoded = urllib.parse.unquote(decoded)
            if new_decoded == decoded:
                break
            decoded = new_decoded
        return decoded

    def _validate_extension(self, path: Path) -> None:
        """Validate that the file has an allowed extension.

        Args:
            path: The path to validate.

        Raises:
            PathSecurityError: If the extension is not allowed.
        """
        # Get extension (lowercase for comparison)
        ext = path.suffix.lower()

        if ext not in VALID_EXTENSIONS:
            allowed = ", ".join(sorted(VALID_EXTENSIONS))
            raise PathSecurityError(
                path=str(path),
                violation_type="invalid_extension",
                message=f"Invalid file extension '{ext}'. Allowed: {allowed}",
            )

    def _check_symlink(self, path: Path, original_path_str: str) -> None:
        """Check if path or any parent is a symlink.

        Args:
            path: The path to check.
            original_path_str: Original path string for error messages.

        Raises:
            PathSecurityError: If symlinks are found and not allowed.
        """
        # Check the path itself
        if path.is_symlink():
            raise PathSecurityError(
                path=original_path_str,
                violation_type="symlink_not_allowed",
                message=f"Symlinks are not allowed: {original_path_str}",
            )

        # Check all parents
        current = path
        while current != current.parent:
            current = current.parent
            if current.is_symlink():
                raise PathSecurityError(
                    path=original_path_str,
                    violation_type="symlink_not_allowed",
                    message=f"Path contains symlink in parent directory: {original_path_str}",
                )

    def _validate_allowlist(
        self,
        canonical_path: Path,
        allowed_paths: tuple[Path, ...],
        original_path_str: str,
    ) -> None:
        """Validate that the canonical path is within allowed directories.

        Args:
            canonical_path: The canonicalized path to validate.
            allowed_paths: Tuple of allowed directory paths.
            original_path_str: Original path string for error messages.

        Raises:
            PathSecurityError: If the path is outside all allowed directories.
        """
        # Check if canonical path is under any allowed directory FIRST
        # If explicitly allowed, skip sensitive directory check
        for allowed in allowed_paths:
            try:
                # Use is_relative_to for Python 3.9+
                if canonical_path.is_relative_to(allowed):
                    return  # Path is explicitly allowed, skip all other checks
            except AttributeError:
                # Fallback for older Python
                try:
                    canonical_path.relative_to(allowed)
                    return  # Path is explicitly allowed, skip all other checks
                except ValueError:
                    continue

        # Path not in allowlist - now check sensitive directories
        canonical_str = str(canonical_path)
        for sensitive in SENSITIVE_DIRECTORIES:
            if canonical_str.startswith(sensitive):
                raise PathSecurityError(
                    path=original_path_str,
                    violation_type="sensitive_directory",
                    message=f"Access to sensitive directory is blocked: {sensitive}",
                )

        # Path is not in any allowed directory
        allowed_str = ", ".join(str(p) for p in allowed_paths)
        raise PathSecurityError(
            path=original_path_str,
            violation_type="path_outside_allowlist",
            message=f"Path is not in allowed directories. Allowed: {allowed_str}",
        )
