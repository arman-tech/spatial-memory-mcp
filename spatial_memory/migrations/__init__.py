"""Database migrations for spatial-memory-mcp.

This package contains database migration scripts for schema changes.
Each migration is a module that exports a Migration class.

Migrations should be numbered sequentially (001, 002, etc.) and use
semantic versioning for their version string.

Example:
    # migrations/001_add_expires_at.py
    from spatial_memory.core.db_migrations import Migration

    class Migration001AddExpiresAt(Migration):
        version = "1.1.0"
        description = "Add expires_at column for TTL support"

        def up(self, db, embeddings=None):
            # Apply migration
            pass

        def down(self, db):
            # Rollback migration (optional)
            raise NotImplementedError("Rollback not supported")
"""

from spatial_memory.core.db_migrations import (
    CURRENT_SCHEMA_VERSION,
    Migration,
    MigrationManager,
    MigrationResult,
    check_migration_status,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Migration",
    "MigrationManager",
    "MigrationResult",
    "check_migration_status",
]
