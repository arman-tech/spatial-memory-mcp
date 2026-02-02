"""Schema migration system for LanceDB database.

This module provides a migration framework for managing schema changes
over time. It supports:
- Forward migrations (up)
- Rollback migrations (down)
- Dry-run mode for previewing changes
- Automatic snapshot creation before migrations

Migrations are versioned using semantic versioning (e.g., "1.0.0", "1.1.0").

Usage:
    from spatial_memory.core.db_migrations import MigrationManager

    manager = MigrationManager(database)
    manager.register_builtin_migrations()

    # Check pending migrations
    pending = manager.get_pending_migrations()

    # Run migrations
    applied = manager.run_pending(dry_run=False)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from spatial_memory.core.errors import MigrationError, StorageError
from spatial_memory.core.utils import utc_now

if TYPE_CHECKING:
    from spatial_memory.core.database import Database
    from spatial_memory.ports.repositories import EmbeddingServiceProtocol

logger = logging.getLogger(__name__)


# Schema metadata table name
SCHEMA_VERSIONS_TABLE = "_schema_versions"

# Current schema version
CURRENT_SCHEMA_VERSION = "1.0.0"


# =============================================================================
# Migration Data Types
# =============================================================================


@dataclass
class MigrationRecord:
    """Record of an applied migration.

    Stored in the _schema_versions table to track migration history.
    """

    version: str
    description: str
    applied_at: datetime
    embedding_model: str | None = None
    embedding_dimensions: int | None = None


@dataclass
class MigrationResult:
    """Result of running migrations.

    Attributes:
        migrations_applied: List of version strings that were applied.
        dry_run: Whether this was a dry run.
        current_version: Version after migrations.
        errors: List of error messages if any migrations failed.
    """

    migrations_applied: list[str] = field(default_factory=list)
    dry_run: bool = True
    current_version: str = "0.0.0"
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Migration Protocol/Base Class
# =============================================================================


class Migration(ABC):
    """Abstract base class for schema migrations.

    Each migration should:
    1. Have a unique version string (semantic versioning)
    2. Implement up() for forward migration
    3. Optionally implement down() for rollback
    4. Provide a description of what the migration does

    Example:
        class Migration001AddExpiresAt(Migration):
            version = "1.1.0"
            description = "Add expires_at column for TTL support"

            def up(self, db: Database, embeddings: EmbeddingServiceProtocol | None) -> None:
                # Add new column or modify schema
                pass

            def down(self, db: Database) -> None:
                # Rollback changes (optional)
                raise NotImplementedError("Rollback not supported")
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g., '1.1.0')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the migration."""
        ...

    @abstractmethod
    def up(
        self,
        db: Database,
        embeddings: EmbeddingServiceProtocol | None = None,
    ) -> None:
        """Apply the migration forward.

        Args:
            db: Database instance to migrate.
            embeddings: Optional embedding service for re-embedding operations.

        Raises:
            MigrationError: If migration fails.
        """
        ...

    def down(self, db: Database) -> None:
        """Rollback the migration (optional).

        By default, rollback is not supported. Override this method
        to enable rollback for a specific migration.

        Args:
            db: Database instance to rollback.

        Raises:
            NotImplementedError: If rollback is not supported.
        """
        raise NotImplementedError(
            f"Rollback not supported for migration {self.version}"
        )


# =============================================================================
# Migration Manager
# =============================================================================


class MigrationManager:
    """Manages database schema migrations.

    The manager:
    - Tracks applied migrations in a metadata table
    - Supports forward migrations (up) and rollbacks (down)
    - Creates snapshots before applying migrations for safety
    - Supports dry-run mode for previewing changes

    Example:
        manager = MigrationManager(database)
        manager.register_builtin_migrations()

        # Preview pending migrations
        pending = manager.get_pending_migrations()
        for m in pending:
            print(f"Pending: {m.version} - {m.description}")

        # Apply migrations
        result = manager.run_pending(dry_run=False)
        print(f"Applied: {result.migrations_applied}")
    """

    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingServiceProtocol | None = None,
    ) -> None:
        """Initialize the migration manager.

        Args:
            db: Database instance to manage.
            embeddings: Optional embedding service for migrations that re-embed.
        """
        self._db = db
        self._embeddings = embeddings
        self._migrations: dict[str, Migration] = {}
        self._schema_table_checked = False

    def _ensure_schema_table(self) -> None:
        """Ensure the schema versions table exists.

        Creates the table if it doesn't exist. This is called lazily
        on first access to avoid issues with fresh databases.
        """
        if self._schema_table_checked:
            return

        try:
            lance_db = self._db._db
            table_names = lance_db.table_names()

            if SCHEMA_VERSIONS_TABLE not in table_names:
                # Create schema versions table
                schema = pa.schema([
                    pa.field("version", pa.string()),
                    pa.field("description", pa.string()),
                    pa.field("applied_at", pa.timestamp("us")),
                    pa.field("embedding_model", pa.string()),
                    pa.field("embedding_dimensions", pa.int32()),
                ])
                # Create empty table with schema
                empty_table = pa.table(
                    {
                        "version": pa.array([], type=pa.string()),
                        "description": pa.array([], type=pa.string()),
                        "applied_at": pa.array([], type=pa.timestamp("us")),
                        "embedding_model": pa.array([], type=pa.string()),
                        "embedding_dimensions": pa.array([], type=pa.int32()),
                    },
                    schema=schema,
                )
                lance_db.create_table(SCHEMA_VERSIONS_TABLE, empty_table)
                logger.info("Created schema versions table")

            self._schema_table_checked = True
        except Exception as e:
            raise StorageError(f"Failed to create schema versions table: {e}") from e

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration instance to register.

        Raises:
            ValueError: If a migration with the same version already exists.
        """
        if migration.version in self._migrations:
            raise ValueError(f"Migration {migration.version} already registered")
        self._migrations[migration.version] = migration

    def register_builtin_migrations(self) -> None:
        """Register all built-in migrations.

        Called automatically to set up standard migrations.
        """
        # Register the initial schema migration
        self.register(InitialSchemaMigration())
        # Future migrations would be registered here:
        # self.register(Migration001AddExpiresAt())

    def get_current_version(self) -> str:
        """Get the current schema version from the database.

        Returns:
            Current version string, or "0.0.0" if no migrations applied.
        """
        self._ensure_schema_table()

        try:
            lance_db = self._db._db
            table = lance_db.open_table(SCHEMA_VERSIONS_TABLE)
            arrow_table = table.to_arrow()

            if arrow_table.num_rows == 0:
                return "0.0.0"

            # Get the latest version by comparing all versions
            versions = arrow_table.column("version").to_pylist()
            if not versions:
                return "0.0.0"

            # Find the maximum version using semantic comparison
            return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
        except Exception as e:
            logger.warning(f"Could not get current version: {e}")
            return "0.0.0"

    def get_applied_migrations(self) -> list[MigrationRecord]:
        """Get list of all applied migrations.

        Returns:
            List of MigrationRecord for each applied migration.
        """
        self._ensure_schema_table()

        try:
            lance_db = self._db._db
            table = lance_db.open_table(SCHEMA_VERSIONS_TABLE)
            arrow_table = table.to_arrow()

            if arrow_table.num_rows == 0:
                return []

            records = []
            versions = arrow_table.column("version").to_pylist()
            descriptions = arrow_table.column("description").to_pylist()
            applied_ats = arrow_table.column("applied_at").to_pylist()
            embedding_models = arrow_table.column("embedding_model").to_pylist()
            embedding_dims = arrow_table.column("embedding_dimensions").to_pylist()

            for i in range(arrow_table.num_rows):
                # Handle timestamp conversion
                applied_at = applied_ats[i]
                if hasattr(applied_at, "as_py"):
                    applied_at = applied_at.as_py()

                records.append(MigrationRecord(
                    version=versions[i],
                    description=descriptions[i],
                    applied_at=applied_at,
                    embedding_model=embedding_models[i],
                    embedding_dimensions=embedding_dims[i],
                ))
            return records
        except Exception as e:
            logger.warning(f"Could not get applied migrations: {e}")
            return []

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations that haven't been applied yet.

        Returns:
            List of Migration instances that need to be applied.
        """
        current = self.get_current_version()
        pending = []

        for version in sorted(self._migrations.keys()):
            if self._compare_versions(version, current) > 0:
                pending.append(self._migrations[version])

        return pending

    def run_pending(self, dry_run: bool = True) -> MigrationResult:
        """Run all pending migrations.

        Args:
            dry_run: If True, preview migrations without applying.

        Returns:
            MigrationResult with applied migrations and any errors.
        """
        pending = self.get_pending_migrations()
        result = MigrationResult(
            dry_run=dry_run,
            current_version=self.get_current_version(),
        )

        if not pending:
            logger.info("No pending migrations")
            return result

        for migration in pending:
            logger.info(
                f"{'Would apply' if dry_run else 'Applying'} migration "
                f"{migration.version}: {migration.description}"
            )

            if not dry_run:
                try:
                    # Create snapshot before migration
                    snapshot_tag = f"pre-migration-{migration.version}"
                    snapshot_version = self._db.create_snapshot(snapshot_tag)
                    logger.info(f"Created pre-migration snapshot at version {snapshot_version}")

                    # Apply migration
                    migration.up(self._db, self._embeddings)

                    # Record migration
                    self._record_migration(migration)

                    result.migrations_applied.append(migration.version)
                    result.current_version = migration.version
                except Exception as e:
                    error_msg = f"Migration {migration.version} failed: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

                    # Stop on first error
                    break
            else:
                result.migrations_applied.append(migration.version)

        return result

    def rollback(self, target_version: str) -> MigrationResult:
        """Rollback to a specific version.

        Args:
            target_version: Version to rollback to.

        Returns:
            MigrationResult with rolled back migrations and any errors.

        Raises:
            MigrationError: If rollback fails.
        """
        current = self.get_current_version()
        result = MigrationResult(
            dry_run=False,
            current_version=current,
        )

        if self._compare_versions(target_version, current) >= 0:
            logger.info(f"Already at or before version {target_version}")
            return result

        # Find migrations to rollback (newest first)
        to_rollback = []
        for version in sorted(self._migrations.keys(), reverse=True):
            if self._compare_versions(version, target_version) > 0:
                if self._compare_versions(version, current) <= 0:
                    to_rollback.append(self._migrations[version])

        for migration in to_rollback:
            logger.info(f"Rolling back migration {migration.version}")
            try:
                migration.down(self._db)
                # Remove migration record
                self._remove_migration_record(migration.version)
                result.migrations_applied.append(migration.version)
            except NotImplementedError:
                error_msg = f"Rollback not supported for {migration.version}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                break
            except Exception as e:
                error_msg = f"Rollback of {migration.version} failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                break

        result.current_version = self.get_current_version()
        return result

    def _record_migration(self, migration: Migration) -> None:
        """Record that a migration was applied.

        Args:
            migration: Migration that was applied.
        """
        self._ensure_schema_table()

        try:
            lance_db = self._db._db
            table = lance_db.open_table(SCHEMA_VERSIONS_TABLE)

            # Get embedding info if available
            embedding_model = None
            embedding_dim = None
            if self._embeddings:
                embedding_model = getattr(self._embeddings, "model_name", None)
                embedding_dim = getattr(self._embeddings, "dimensions", None)

            record = pa.table({
                "version": [migration.version],
                "description": [migration.description],
                "applied_at": [utc_now()],
                "embedding_model": [embedding_model],
                "embedding_dimensions": [embedding_dim],
            })
            table.add(record)
        except Exception as e:
            raise MigrationError(f"Failed to record migration: {e}") from e

    def _remove_migration_record(self, version: str) -> None:
        """Remove a migration record (for rollback).

        Args:
            version: Version to remove.
        """
        self._ensure_schema_table()

        try:
            lance_db = self._db._db
            table = lance_db.open_table(SCHEMA_VERSIONS_TABLE)
            table.delete(f'version = "{version}"')
        except Exception as e:
            logger.warning(f"Failed to remove migration record: {e}")

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two semantic version strings.

        Args:
            v1: First version.
            v2: Second version.

        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
        """
        def parse(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split("."))

        p1, p2 = parse(v1), parse(v2)
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1
        return 0


# =============================================================================
# Built-in Migrations
# =============================================================================


class InitialSchemaMigration(Migration):
    """Initial schema setup migration.

    This migration represents the initial schema state. It doesn't
    actually change anything but serves as a baseline version marker.
    """

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Initial schema version"

    def up(
        self,
        db: Database,
        embeddings: EmbeddingServiceProtocol | None = None,
    ) -> None:
        """No-op for initial schema - just marks version."""
        logger.info("Initial schema version marker applied")

    def down(self, db: Database) -> None:
        """Cannot rollback initial schema."""
        raise NotImplementedError("Cannot rollback initial schema")


# =============================================================================
# Helper Functions
# =============================================================================


def check_migration_status(db: Database) -> dict[str, Any]:
    """Check the migration status of a database.

    Args:
        db: Database to check.

    Returns:
        Dictionary with migration status information.
    """
    manager = MigrationManager(db)
    manager.register_builtin_migrations()

    current = manager.get_current_version()
    pending = manager.get_pending_migrations()

    return {
        "current_version": current,
        "target_version": CURRENT_SCHEMA_VERSION,
        "pending_count": len(pending),
        "pending_migrations": [
            {"version": m.version, "description": m.description}
            for m in pending
        ],
        "needs_migration": len(pending) > 0,
    }
