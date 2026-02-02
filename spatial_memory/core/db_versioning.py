"""Snapshot and version management for LanceDB database.

Provides snapshot creation, listing, and restoration capabilities
leveraging LanceDB's built-in versioning system.

This module is part of the database.py refactoring to separate concerns:
- VersionManager handles all snapshot/version operations
- Database class delegates to VersionManager for these operations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from spatial_memory.core.errors import StorageError, ValidationError

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable

logger = logging.getLogger(__name__)


class VersionManagerProtocol(Protocol):
    """Protocol defining what VersionManager needs from Database.

    This protocol enables loose coupling between VersionManager and Database,
    preventing circular imports while maintaining type safety.
    """

    @property
    def table(self) -> LanceTable:
        """Access to the LanceDB table."""
        ...

    def _invalidate_count_cache(self) -> None:
        """Invalidate the row count cache."""
        ...

    def _track_modification(self, count: int = 1) -> None:
        """Track a modification for auto-compaction."""
        ...

    def _invalidate_namespace_cache(self) -> None:
        """Invalidate the namespace cache."""
        ...


class VersionManager:
    """Manages database snapshots and version control.

    Leverages LanceDB's native versioning to provide:
    - Snapshot creation with semantic tags
    - Version listing
    - Point-in-time restoration

    LanceDB automatically versions data on every write. This manager
    provides a clean interface for working with those versions.

    Example:
        version_mgr = VersionManager(database)
        version = version_mgr.create_snapshot("backup-2024-01")
        snapshots = version_mgr.list_snapshots()
        version_mgr.restore_snapshot(version)
    """

    def __init__(self, db: VersionManagerProtocol) -> None:
        """Initialize the version manager.

        Args:
            db: Database instance providing table and cache access.
        """
        self._db = db

    def create_snapshot(self, tag: str) -> int:
        """Create a named snapshot of the current table state.

        LanceDB automatically versions data on every write. This method
        returns the current version number which can be used with restore_snapshot().

        Args:
            tag: Semantic version tag (e.g., "v1.0.0", "backup-2024-01").
                 Note: Tag is logged for reference but LanceDB tracks versions
                 numerically. Consider storing tag->version mappings externally
                 if tag-based retrieval is needed.

        Returns:
            Version number of the snapshot.

        Raises:
            StorageError: If snapshot creation fails.
        """
        try:
            version = self._db.table.version
            logger.info(f"Created snapshot '{tag}' at version {version}")
            return version
        except Exception as e:
            raise StorageError(f"Failed to create snapshot: {e}") from e

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List available versions/snapshots.

        Returns:
            List of version information dictionaries. Each dict contains
            at minimum 'version' key. Additional fields depend on LanceDB
            version and available metadata.

        Raises:
            StorageError: If listing fails.
        """
        try:
            versions_info: list[dict[str, Any]] = []

            # Try to get version history if available
            if hasattr(self._db.table, "list_versions"):
                try:
                    versions = self._db.table.list_versions()
                    for v in versions:
                        if isinstance(v, dict):
                            versions_info.append(v)
                        elif hasattr(v, "version"):
                            versions_info.append({
                                "version": v.version,
                                "timestamp": getattr(v, "timestamp", None),
                            })
                        else:
                            versions_info.append({"version": v})
                except Exception as e:
                    logger.debug(f"list_versions not fully supported: {e}")

            # Always include current version
            if not versions_info:
                versions_info.append({"version": self._db.table.version})

            return versions_info
        except Exception as e:
            logger.warning(f"Could not list snapshots: {e}")
            return [{"version": 0, "error": str(e)}]

    def restore_snapshot(self, version: int) -> None:
        """Restore table to a specific version.

        This creates a NEW version that reflects the old state
        (doesn't delete history).

        Args:
            version: The version number to restore to.

        Raises:
            ValidationError: If version is invalid.
            StorageError: If restore fails.
        """
        if version < 0:
            raise ValidationError("Version must be non-negative")

        try:
            self._db.table.restore(version)
            self._db._invalidate_count_cache()
            self._db._track_modification()
            self._db._invalidate_namespace_cache()
            logger.info(f"Restored to version {version}")
        except Exception as e:
            raise StorageError(f"Failed to restore snapshot: {e}") from e

    def get_current_version(self) -> int:
        """Get the current table version number.

        Returns:
            Current version number.

        Raises:
            StorageError: If version cannot be retrieved.
        """
        try:
            return self._db.table.version
        except Exception as e:
            raise StorageError(f"Failed to get current version: {e}") from e
