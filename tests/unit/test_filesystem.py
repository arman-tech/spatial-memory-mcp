"""Tests for filesystem detection utilities."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.core.filesystem import (
    FilesystemType,
    detect_filesystem_type,
    get_filesystem_warning_message,
    is_network_filesystem,
)


class TestFilesystemType:
    """Tests for FilesystemType enum."""

    def test_enum_values(self) -> None:
        """Test that all expected enum values exist."""
        assert FilesystemType.LOCAL.value == "local"
        assert FilesystemType.NFS.value == "nfs"
        assert FilesystemType.SMB.value == "smb"
        assert FilesystemType.CIFS.value == "cifs"
        assert FilesystemType.NETWORK_UNKNOWN.value == "network_unknown"
        assert FilesystemType.UNKNOWN.value == "unknown"


class TestIsNetworkFilesystem:
    """Tests for is_network_filesystem function."""

    def test_local_is_not_network(self) -> None:
        """Test that LOCAL is not considered a network filesystem."""
        with patch(
            "spatial_memory.core.filesystem.detect_filesystem_type",
            return_value=FilesystemType.LOCAL,
        ):
            assert is_network_filesystem(Path("/some/path")) is False

    def test_unknown_is_not_network(self) -> None:
        """Test that UNKNOWN is not considered a network filesystem."""
        with patch(
            "spatial_memory.core.filesystem.detect_filesystem_type",
            return_value=FilesystemType.UNKNOWN,
        ):
            assert is_network_filesystem(Path("/some/path")) is False

    @pytest.mark.parametrize(
        "fs_type",
        [
            FilesystemType.NFS,
            FilesystemType.SMB,
            FilesystemType.CIFS,
            FilesystemType.NETWORK_UNKNOWN,
        ],
    )
    def test_network_types_detected(self, fs_type: FilesystemType) -> None:
        """Test that network filesystem types are correctly identified."""
        with patch(
            "spatial_memory.core.filesystem.detect_filesystem_type",
            return_value=fs_type,
        ):
            assert is_network_filesystem(Path("/some/path")) is True


class TestGetFilesystemWarningMessage:
    """Tests for get_filesystem_warning_message function."""

    def test_warning_message_contains_type(self) -> None:
        """Test that warning message contains the filesystem type."""
        test_path = Path("/mnt/nfs")
        msg = get_filesystem_warning_message(FilesystemType.NFS, test_path)
        assert "nfs" in msg.lower()
        assert str(test_path) in msg  # Path representation varies by platform

    def test_warning_message_explains_risk(self) -> None:
        """Test that warning message explains the risk."""
        msg = get_filesystem_warning_message(FilesystemType.SMB, Path("/mnt/share"))
        assert "locking" in msg.lower()
        assert "data corruption" in msg.lower()

    def test_warning_message_suggests_env_var(self) -> None:
        """Test that warning message suggests environment variable."""
        msg = get_filesystem_warning_message(FilesystemType.CIFS, Path("/net"))
        assert "SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FS_RISK" in msg


class TestDetectFilesystemType:
    """Tests for detect_filesystem_type function."""

    def test_handles_exception_gracefully(self) -> None:
        """Test that exceptions return UNKNOWN."""
        with patch("pathlib.Path.resolve", side_effect=OSError("Test error")):
            result = detect_filesystem_type(Path("/some/path"))
            assert result == FilesystemType.UNKNOWN


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
class TestWindowsDetection:
    """Tests for Windows-specific filesystem detection."""

    def test_local_drive_detected(self) -> None:
        """Test that local drives are detected as LOCAL."""
        with patch("ctypes.windll.kernel32.GetDriveTypeW", return_value=3):  # DRIVE_FIXED
            result = detect_filesystem_type(Path("C:\\Users\\test"))
            assert result == FilesystemType.LOCAL

    def test_remote_drive_detected(self) -> None:
        """Test that remote drives are detected as NETWORK_UNKNOWN."""
        with patch("ctypes.windll.kernel32.GetDriveTypeW", return_value=4):  # DRIVE_REMOTE
            result = detect_filesystem_type(Path("Z:\\share"))
            assert result == FilesystemType.NETWORK_UNKNOWN


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
class TestUnixDetection:
    """Tests for Unix-specific filesystem detection."""

    def test_nfs_detected_from_df(self) -> None:
        """Test that NFS is detected from df output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Filesystem     Type  Size  Used Avail Use% Mounted on\nnfs-server:/   nfs4  100G   50G   50G  50% /mnt/nfs"

        with patch("subprocess.run", return_value=mock_result):
            result = detect_filesystem_type(Path("/mnt/nfs/data"))
            assert result == FilesystemType.NFS

    def test_cifs_detected_from_df(self) -> None:
        """Test that CIFS is detected from df output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Filesystem     Type  Size  Used Avail Use% Mounted on\n//server/share cifs  100G   50G   50G  50% /mnt/share"

        with patch("subprocess.run", return_value=mock_result):
            result = detect_filesystem_type(Path("/mnt/share/data"))
            assert result == FilesystemType.CIFS

    def test_local_detected_from_df(self) -> None:
        """Test that local filesystems are detected from df output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Filesystem     Type  Size  Used Avail Use% Mounted on\n/dev/sda1      ext4  100G   50G   50G  50% /"

        with patch("subprocess.run", return_value=mock_result):
            result = detect_filesystem_type(Path("/home/user"))
            assert result == FilesystemType.LOCAL

    def test_sshfs_detected_as_network_unknown(self) -> None:
        """Test that SSHFS is detected as NETWORK_UNKNOWN."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Filesystem     Type       Size  Used Avail Use% Mounted on\nuser@host:     fuse.sshfs 100G   50G   50G  50% /mnt/ssh"

        with patch("subprocess.run", return_value=mock_result):
            result = detect_filesystem_type(Path("/mnt/ssh"))
            assert result == FilesystemType.NETWORK_UNKNOWN
