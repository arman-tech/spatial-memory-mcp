"""Tests for v1.1.0 migration (project + content_hash columns)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, PropertyMock

import pytest

from spatial_memory.core.db_migrations import (
    CURRENT_SCHEMA_VERSION,
    AddProjectAndHashMigration,
    MigrationManager,
)
from spatial_memory.core.errors import BackfillError


@pytest.mark.unit
class TestMigrationV110:
    """Tests for AddProjectAndHashMigration."""

    def test_version(self) -> None:
        """Test migration version is 1.1.0."""
        migration = AddProjectAndHashMigration()
        assert migration.version == "1.1.0"

    def test_description(self) -> None:
        """Test migration has a description."""
        migration = AddProjectAndHashMigration()
        assert "project" in migration.description.lower()
        assert "content_hash" in migration.description.lower()

    def test_current_schema_version(self) -> None:
        """Test CURRENT_SCHEMA_VERSION is updated."""
        assert CURRENT_SCHEMA_VERSION == "1.1.0"


@pytest.mark.unit
class TestMigrationRegistration:
    """Tests for migration registration."""

    def test_builtin_migrations_include_v110(self) -> None:
        """Test that v1.1.0 migration is registered."""
        # We can't create a real MigrationManager without a Database,
        # but we can verify the migration class exists and is importable
        migration = AddProjectAndHashMigration()
        assert migration.version == "1.1.0"
        assert callable(migration.up)
        assert callable(migration.down)


@pytest.mark.unit
class TestBackfillErrorRaises:
    """Tests that backfill failure raises BackfillError with accurate message."""

    def _make_migration_with_failing_backfill(self) -> tuple[AddProjectAndHashMigration, MagicMock]:
        """Create a migration with a mock DB where backfill will fail."""
        migration = AddProjectAndHashMigration()
        mock_db = MagicMock()
        mock_table = MagicMock()
        type(mock_db).table = PropertyMock(return_value=mock_table)
        mock_table.add_columns.return_value = None
        mock_table.to_arrow.side_effect = RuntimeError("disk full")
        return migration, mock_db

    def test_backfill_failure_raises_backfill_error(self) -> None:
        """Backfill failure should raise BackfillError, not silently swallow."""
        migration, mock_db = self._make_migration_with_failing_backfill()

        with pytest.raises(BackfillError, match="Content hash backfill failed"):
            migration.up(mock_db)

    def test_backfill_error_chains_original_cause(self) -> None:
        """BackfillError should chain the original exception as __cause__."""
        migration, mock_db = self._make_migration_with_failing_backfill()

        with pytest.raises(BackfillError) as exc_info:
            migration.up(mock_db)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "disk full" in str(exc_info.value.__cause__)

    def test_backfill_error_message_is_accurate(self) -> None:
        """Error message should mention dedup layers and recovery command."""
        migration, mock_db = self._make_migration_with_failing_backfill()

        with pytest.raises(BackfillError) as exc_info:
            migration.up(mock_db)

        msg = str(exc_info.value)

        # Should NOT claim hashes will be computed on access (old, wrong message)
        assert "computed on access" not in msg

        # Should mention layer 1 dedup won't work for pre-existing records
        assert "layer 1" in msg
        # Should mention layer 2 still works
        assert "layer 2" in msg
        # Should mention backfill command
        assert "backfill-hashes" in msg


@pytest.mark.unit
class TestRunPendingSnapshotRestore:
    """Tests that run_pending restores snapshot on migration failure."""

    def _make_manager_with_failing_migration(self) -> tuple[MigrationManager, MagicMock]:
        """Create a MigrationManager with a mock DB and a failing migration registered."""
        mock_db = MagicMock()
        # create_snapshot returns a version number
        mock_db.create_snapshot.return_value = 42

        manager = MigrationManager(mock_db)

        # Bypass schema table check
        manager._schema_table_checked = True

        # Mock get_current_version to return "0.0.0" (so our migration is "pending")
        manager.get_current_version = MagicMock(return_value="0.0.0")  # type: ignore[method-assign]

        # Register a migration that fails
        failing_migration = MagicMock()
        failing_migration.version = "1.0.0"
        failing_migration.description = "test migration"
        failing_migration.up.side_effect = RuntimeError("migration boom")

        manager._migrations["1.0.0"] = failing_migration

        return manager, mock_db

    def test_restore_snapshot_called_on_failure(self) -> None:
        """When migration fails, restore_snapshot should be called."""
        manager, mock_db = self._make_manager_with_failing_migration()

        result = manager.run_pending(dry_run=False)

        assert len(result.errors) == 1
        assert "migration boom" in result.errors[0]
        mock_db.restore_snapshot.assert_called_once_with(42)

    def test_restore_snapshot_failure_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If restore_snapshot itself fails, that error should be logged."""
        manager, mock_db = self._make_manager_with_failing_migration()
        mock_db.restore_snapshot.side_effect = RuntimeError("restore failed")

        with caplog.at_level(logging.DEBUG, logger="spatial_memory.core.db_migrations"):
            result = manager.run_pending(dry_run=False)

        assert len(result.errors) == 1

        # Verify error about failed restore is logged
        restore_errors = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and "Failed to restore snapshot" in r.message
        ]
        assert len(restore_errors) == 1
        assert "restore failed" in restore_errors[0].message

    def test_successful_restore_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Successful snapshot restore should be logged at INFO level."""
        manager, mock_db = self._make_manager_with_failing_migration()

        with caplog.at_level(logging.DEBUG, logger="spatial_memory.core.db_migrations"):
            manager.run_pending(dry_run=False)

        restore_info = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "Restored pre-migration snapshot" in r.message
        ]
        assert len(restore_info) == 1
        assert "42" in restore_info[0].message


@pytest.mark.unit
class TestRunPendingBackfillError:
    """Tests that run_pending handles BackfillError: records migration but surfaces error."""

    def _make_manager_with_backfill_failing_migration(
        self,
    ) -> tuple[MigrationManager, MagicMock]:
        """Create a MigrationManager with a migration that raises BackfillError."""
        mock_db = MagicMock()
        mock_db.create_snapshot.return_value = 42

        manager = MigrationManager(mock_db)
        manager._schema_table_checked = True
        manager.get_current_version = MagicMock(return_value="0.0.0")  # type: ignore[method-assign]

        failing_migration = MagicMock()
        failing_migration.version = "1.1.0"
        failing_migration.description = "add project and content_hash"
        failing_migration.up.side_effect = BackfillError("backfill failed: disk full")

        manager._migrations["1.1.0"] = failing_migration

        return manager, mock_db

    def test_migration_recorded_despite_backfill_failure(self) -> None:
        """Migration should be recorded as applied when only backfill fails."""
        manager, mock_db = self._make_manager_with_backfill_failing_migration()

        result = manager.run_pending(dry_run=False)

        assert "1.1.0" in result.migrations_applied
        assert result.current_version == "1.1.0"
        # Schema is correct so no snapshot restore should happen
        mock_db.restore_snapshot.assert_not_called()

    def test_backfill_error_surfaces_in_result(self) -> None:
        """BackfillError should appear in result.errors for caller visibility."""
        manager, mock_db = self._make_manager_with_backfill_failing_migration()

        result = manager.run_pending(dry_run=False)

        assert len(result.errors) == 1
        assert "Backfill warning" in result.errors[0]
        assert "disk full" in result.errors[0]

    def test_no_snapshot_restore_on_backfill_error(self) -> None:
        """BackfillError should NOT trigger snapshot restore (schema is correct)."""
        manager, mock_db = self._make_manager_with_backfill_failing_migration()

        manager.run_pending(dry_run=False)

        mock_db.restore_snapshot.assert_not_called()

    def test_backfill_error_logged_as_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """BackfillError should be logged at WARNING level, not ERROR."""
        manager, mock_db = self._make_manager_with_backfill_failing_migration()

        with caplog.at_level(logging.DEBUG, logger="spatial_memory.core.db_migrations"):
            manager.run_pending(dry_run=False)

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "backfill incomplete" in r.message
        ]
        assert len(warning_records) == 1
        assert "1.1.0" in warning_records[0].message

    def test_subsequent_migrations_still_run_after_backfill_error(self) -> None:
        """Migrations after a BackfillError should still be applied (no break)."""
        mock_db = MagicMock()
        mock_db.create_snapshot.return_value = 42

        manager = MigrationManager(mock_db)
        manager._schema_table_checked = True
        manager.get_current_version = MagicMock(return_value="0.0.0")  # type: ignore[method-assign]

        # First migration: backfill fails
        m1 = MagicMock()
        m1.version = "1.1.0"
        m1.description = "add columns"
        m1.up.side_effect = BackfillError("backfill failed")

        # Second migration: succeeds
        m2 = MagicMock()
        m2.version = "1.2.0"
        m2.description = "next change"
        m2.up.return_value = None

        manager._migrations["1.1.0"] = m1
        manager._migrations["1.2.0"] = m2

        result = manager.run_pending(dry_run=False)

        assert "1.1.0" in result.migrations_applied
        assert "1.2.0" in result.migrations_applied
        assert result.current_version == "1.2.0"
        assert len(result.errors) == 1  # Only the backfill warning


@pytest.mark.unit
class TestCompareVersionsDefensive:
    """Tests for H6: _compare_versions handles malformed version strings."""

    def test_valid_versions(self) -> None:
        """Normal semantic versions compare correctly."""
        assert MigrationManager._compare_versions("1.0.0", "1.1.0") == -1
        assert MigrationManager._compare_versions("1.1.0", "1.0.0") == 1
        assert MigrationManager._compare_versions("1.0.0", "1.0.0") == 0

    def test_malformed_version_treated_as_zero(self) -> None:
        """Malformed version strings should not crash, treated as 0.0.0."""
        result = MigrationManager._compare_versions("abc.1.0", "1.0.0")
        assert result == -1  # (0,0,0) < (1,0,0)

    def test_empty_string_version(self) -> None:
        """Empty string version should not crash."""
        result = MigrationManager._compare_versions("", "1.0.0")
        assert result == -1  # (0,0,0) < (1,0,0)

    def test_two_part_version(self) -> None:
        """Two-part version should parse fine (different tuple length)."""
        result = MigrationManager._compare_versions("1.0", "1.0.0")
        assert result == -1  # (1,0) < (1,0,0) in Python tuple comparison

    def test_both_malformed(self) -> None:
        """Both malformed should return 0 (both treated as 0.0.0)."""
        result = MigrationManager._compare_versions("bad", "also-bad")
        assert result == 0
