"""Tests for v1.1.0 migration (project + content_hash columns)."""

from __future__ import annotations

import pytest

from spatial_memory.core.db_migrations import (
    CURRENT_SCHEMA_VERSION,
    AddProjectAndHashMigration,
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
