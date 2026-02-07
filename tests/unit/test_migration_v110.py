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
class TestBackfillErrorLogging:
    """Tests that backfill failure is logged at ERROR level with accurate message."""

    def test_backfill_failure_logs_error_not_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Backfill failure should log at ERROR level, not WARNING."""
        migration = AddProjectAndHashMigration()

        # Create a mock Database whose table succeeds for add_columns
        # but fails during backfill (to_arrow raises).
        mock_db = MagicMock()
        mock_table = MagicMock()
        type(mock_db).table = PropertyMock(return_value=mock_table)

        # add_columns succeeds
        mock_table.add_columns.return_value = None

        # to_arrow fails to simulate backfill failure
        mock_table.to_arrow.side_effect = RuntimeError("disk full")

        with caplog.at_level(logging.DEBUG, logger="spatial_memory.core.db_migrations"):
            migration.up(mock_db)

        # Verify ERROR level log exists
        error_records = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR
            and "Content hash backfill failed" in r.message
        ]
        assert len(error_records) == 1, (
            f"Expected exactly 1 ERROR log about backfill failure, "
            f"found {len(error_records)}. All records: {[r.message for r in caplog.records]}"
        )

        # Verify no WARNING about backfill (the old behavior)
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "backfill" in r.message.lower()
        ]
        assert len(warning_records) == 0, (
            "Backfill failure should NOT produce a WARNING log"
        )

    def test_backfill_error_message_is_accurate(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Error message should mention dedup layers, not claim hashes will be computed."""
        migration = AddProjectAndHashMigration()

        mock_db = MagicMock()
        mock_table = MagicMock()
        type(mock_db).table = PropertyMock(return_value=mock_table)
        mock_table.add_columns.return_value = None
        mock_table.to_arrow.side_effect = RuntimeError("disk full")

        with caplog.at_level(logging.DEBUG, logger="spatial_memory.core.db_migrations"):
            migration.up(mock_db)

        error_records = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR
            and "Content hash backfill failed" in r.message
        ]
        assert len(error_records) == 1
        msg = error_records[0].message

        # Should NOT claim hashes will be computed on access (old, wrong message)
        assert "computed on access" not in msg

        # Should mention layer 1 dedup won't work for pre-existing records
        assert "layer 1" in msg
        # Should mention layer 2 still works
        assert "layer 2" in msg
        # Should mention backfill command
        assert "backfill-hashes" in msg

    def test_backfill_failure_does_not_raise(self) -> None:
        """Backfill failure should be non-fatal; migration should not raise."""
        migration = AddProjectAndHashMigration()

        mock_db = MagicMock()
        mock_table = MagicMock()
        type(mock_db).table = PropertyMock(return_value=mock_table)
        mock_table.add_columns.return_value = None
        mock_table.to_arrow.side_effect = RuntimeError("disk full")

        # Should NOT raise
        migration.up(mock_db)


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
            if r.levelno == logging.ERROR
            and "Failed to restore snapshot" in r.message
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
            if r.levelno == logging.INFO
            and "Restored pre-migration snapshot" in r.message
        ]
        assert len(restore_info) == 1
        assert "42" in restore_info[0].message


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
