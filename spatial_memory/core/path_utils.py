"""Path utilities for project detection.

Provides path normalization, blocklist detection, and platform-aware
root path identification for the project detection cascade.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Platform-aware blocklisted roots (too broad to be valid project roots)
_BLOCKLISTED_ROOTS: set[Path] | None = None


def normalize_path(path: str | Path) -> Path:
    """Normalize a filesystem path.

    Resolves symlinks, expands ~ and environment variables,
    and normalizes case on Windows.

    Args:
        path: Path to normalize.

    Returns:
        Normalized absolute Path.
    """
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    return p


def get_blocklisted_roots() -> set[Path]:
    """Get platform-aware blocklisted root directories.

    These are directories too broad to be valid project roots
    (home dir, filesystem root, temp directories, etc.)

    Returns:
        Set of blocklisted Path objects.
    """
    global _BLOCKLISTED_ROOTS
    if _BLOCKLISTED_ROOTS is not None:
        return _BLOCKLISTED_ROOTS

    roots: set[Path] = set()

    # Home directory
    home = Path.home()
    roots.add(home)

    # Filesystem roots
    if sys.platform == "win32":
        # Add common Windows drive roots
        for letter in "CDEFG":
            roots.add(Path(f"{letter}:\\"))
            roots.add(Path(f"{letter}:/"))
        # Temp directories
        temp = os.environ.get("TEMP") or os.environ.get("TMP")
        if temp:
            roots.add(normalize_path(temp))
    else:
        roots.add(Path("/"))
        roots.add(Path("/tmp"))
        roots.add(Path("/var/tmp"))

    _BLOCKLISTED_ROOTS = roots
    return _BLOCKLISTED_ROOTS


def is_blocklisted(path: Path) -> bool:
    """Check if a path is a blocklisted root.

    Args:
        path: Path to check.

    Returns:
        True if the path is a blocklisted root directory.
    """
    normalized = normalize_path(path)
    blocklist = get_blocklisted_roots()
    return normalized in blocklist


def reset_blocklist_cache() -> None:
    """Reset the blocklist cache. Used in testing."""
    global _BLOCKLISTED_ROOTS
    _BLOCKLISTED_ROOTS = None
