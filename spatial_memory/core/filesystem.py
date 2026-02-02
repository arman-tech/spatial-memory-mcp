"""Filesystem detection utilities for identifying network filesystems.

This module provides utilities to detect if a path is on a network filesystem
(NFS, SMB/CIFS) where file-based locking may not work reliably.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class FilesystemType(Enum):
    """Types of filesystems that can be detected."""

    LOCAL = "local"
    NFS = "nfs"
    SMB = "smb"
    CIFS = "cifs"
    NETWORK_UNKNOWN = "network_unknown"
    UNKNOWN = "unknown"


def detect_filesystem_type(path: Path) -> FilesystemType:
    """Detect the filesystem type for a given path.

    Args:
        path: Path to check. Will resolve to absolute path.

    Returns:
        FilesystemType indicating the detected filesystem.
        Returns LOCAL for local filesystems, specific types for
        network filesystems, or UNKNOWN if detection fails.
    """
    try:
        resolved = path.resolve()

        if platform.system() == "Windows":
            return _detect_windows(resolved)
        else:
            return _detect_unix(resolved)
    except Exception as e:
        logger.debug(f"Filesystem detection failed for {path}: {e}")
        return FilesystemType.UNKNOWN


def _detect_windows(path: Path) -> FilesystemType:
    """Detect filesystem type on Windows.

    Uses GetDriveTypeW to check if drive is remote.
    """
    try:
        import ctypes

        # Get the drive letter (e.g., "C:\\")
        drive = str(path)[:3] if len(str(path)) >= 3 else str(path)

        # Ensure it ends with backslash for GetDriveTypeW
        if not drive.endswith("\\"):
            drive = drive + "\\"

        # DRIVE_REMOTE = 4
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)

        if drive_type == 4:  # DRIVE_REMOTE
            logger.debug(f"Detected remote drive: {drive}")
            return FilesystemType.NETWORK_UNKNOWN
        else:
            return FilesystemType.LOCAL

    except Exception as e:
        logger.debug(f"Windows filesystem detection failed: {e}")
        return FilesystemType.UNKNOWN


def _detect_unix(path: Path) -> FilesystemType:
    """Detect filesystem type on Unix-like systems.

    Uses 'df -T' or 'mount' to determine filesystem type.
    """
    try:
        # Try using df -T first (more portable)
        result = subprocess.run(
            ["df", "-T", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            output = result.stdout.lower()
            # Check for common network filesystem types
            if "nfs" in output:
                return FilesystemType.NFS
            if "cifs" in output:
                return FilesystemType.CIFS
            if "smb" in output:
                return FilesystemType.SMB
            if "fuse.sshfs" in output:
                return FilesystemType.NETWORK_UNKNOWN
            # If none of the above, assume local
            return FilesystemType.LOCAL

    except subprocess.TimeoutExpired:
        logger.debug("df command timed out - may indicate network filesystem issue")
        return FilesystemType.NETWORK_UNKNOWN
    except FileNotFoundError:
        # df not available, try alternative
        pass
    except Exception as e:
        logger.debug(f"df command failed: {e}")

    # Fallback: try reading /proc/mounts on Linux
    try:
        if os.path.exists("/proc/mounts"):
            with open("/proc/mounts") as f:
                mounts = f.read().lower()
                path_str = str(path).lower()
                # Find the mount point for this path
                for line in mounts.split("\n"):
                    parts = line.split()
                    if len(parts) >= 3:
                        mount_point = parts[1]
                        fs_type = parts[2]
                        if path_str.startswith(mount_point):
                            if "nfs" in fs_type:
                                return FilesystemType.NFS
                            if "cifs" in fs_type or "smb" in fs_type:
                                return FilesystemType.SMB
    except Exception as e:
        logger.debug(f"/proc/mounts check failed: {e}")

    return FilesystemType.LOCAL


def is_network_filesystem(path: Path) -> bool:
    """Check if a path is on a network filesystem.

    Args:
        path: Path to check.

    Returns:
        True if the path appears to be on a network filesystem
        (NFS, SMB, CIFS, or unknown network type).
    """
    fs_type = detect_filesystem_type(path)
    return fs_type in (
        FilesystemType.NFS,
        FilesystemType.SMB,
        FilesystemType.CIFS,
        FilesystemType.NETWORK_UNKNOWN,
    )


def get_filesystem_warning_message(fs_type: FilesystemType, path: Path) -> str:
    """Generate a warning message for network filesystem detection.

    Args:
        fs_type: The detected filesystem type.
        path: The path that was checked.

    Returns:
        A warning message string explaining the risks.
    """
    return (
        f"WARNING: Storage path appears to be on a network filesystem ({fs_type.value}). "
        f"Path: {path}\n"
        f"File-based locking does not work reliably on network filesystems. "
        f"Running multiple instances against this storage may cause data corruption. "
        f"To suppress this warning, set SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FS_RISK=true "
        f"or use a local filesystem path."
    )
