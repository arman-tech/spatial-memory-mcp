"""Entry point for running the Spatial Memory MCP Server and CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import NoReturn

logger = logging.getLogger(__name__)


def run_server() -> None:
    """Run the Spatial Memory MCP Server."""
    from spatial_memory.server import main as server_main

    asyncio.run(server_main())


def run_migrate(args: argparse.Namespace) -> int:
    """Run database migrations.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from spatial_memory.config import get_settings
    from spatial_memory.core.database import Database
    from spatial_memory.core.db_migrations import (
        CURRENT_SCHEMA_VERSION,
        MigrationManager,
    )
    from spatial_memory.core.embeddings import EmbeddingService

    settings = get_settings()

    # Set up logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    print("Spatial Memory Migration Tool")
    print(f"Target schema version: {CURRENT_SCHEMA_VERSION}")
    print(f"Database path: {settings.memory_path}")
    print()

    try:
        # Create embedding service if needed for migrations
        embeddings = None
        if not args.dry_run:
            # Only load embeddings for actual migrations (some may need re-embedding)
            print("Loading embedding service...")
            embeddings = EmbeddingService(
                model_name=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                backend=settings.embedding_backend,  # type: ignore[arg-type]
            )

        # Connect to database
        print("Connecting to database...")
        db = Database(
            storage_path=settings.memory_path,
            embedding_dim=embeddings.dimensions if embeddings else 384,
            auto_create_indexes=settings.auto_create_indexes,
        )
        db.connect()

        # Create migration manager
        manager = MigrationManager(db, embeddings)
        manager.register_builtin_migrations()

        current_version = manager.get_current_version()
        print(f"Current schema version: {current_version}")

        if args.status:
            # Just show status, don't run migrations
            pending = manager.get_pending_migrations()
            if pending:
                print(f"\nPending migrations ({len(pending)}):")
                for migration in pending:
                    print(f"  - {migration.version}: {migration.description}")
            else:
                print("\nNo pending migrations. Database is up to date.")

            applied = manager.get_applied_migrations()
            if applied:
                print(f"\nApplied migrations ({len(applied)}):")
                for record in applied:
                    applied_ts = record.applied_at
                    print(f"  - {record.version}: {record.description} (applied: {applied_ts})")

            db.close()
            return 0

        if args.rollback:
            # Rollback to specified version
            print(f"\nRolling back to version {args.rollback}...")
            result = manager.rollback(args.rollback)

            if result.errors:
                print("\nRollback failed with errors:")
                for error in result.errors:
                    print(f"  - {error}")
                db.close()
                return 1

            if result.migrations_applied:
                print("\nRolled back migrations:")
                for v in result.migrations_applied:
                    print(f"  - {v}")
                print(f"\nCurrent version: {result.current_version}")
            else:
                print("\nNo migrations to rollback.")

            db.close()
            return 0

        # Run pending migrations
        pending = manager.get_pending_migrations()
        if not pending:
            print("\nNo pending migrations. Database is up to date.")
            db.close()
            return 0

        print(f"\nPending migrations ({len(pending)}):")
        for m in pending:
            print(f"  - {m.version}: {m.description}")

        if args.dry_run:
            print("\n[DRY RUN] Would apply the above migrations.")
            print("Run without --dry-run to apply.")
            db.close()
            return 0

        # Confirm before applying
        if not args.yes:
            print()
            response = input("Apply these migrations? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                print("Aborted.")
                db.close()
                return 0

        print("\nApplying migrations...")
        result = manager.run_pending(dry_run=False)

        if result.errors:
            print("\nMigration failed with errors:")
            for error in result.errors:
                print(f"  - {error}")
            print("\nSome migrations may have been applied. Check database state.")
            db.close()
            return 1

        print(f"\nSuccessfully applied {len(result.migrations_applied)} migration(s):")
        for v in result.migrations_applied:
            print(f"  - {v}")
        print(f"\nCurrent version: {result.current_version}")

        db.close()
        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=args.verbose)
        print(f"\nError: {e}")
        return 1


def run_backfill_project(args: argparse.Namespace) -> int:
    """Backfill the project field for existing memories.

    Uses the same ProjectDetector cascade as the server to auto-detect the
    project identifier. If --project is provided, compares it against the
    auto-detected value and warns on mismatch.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from spatial_memory.adapters.project_detection import (
        ProjectDetectionConfig,
        ProjectDetector,
    )
    from spatial_memory.config import get_settings
    from spatial_memory.core.database import Database

    settings = get_settings()

    # Set up logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    namespace_filter: str | None = args.namespace
    dry_run: bool = args.dry_run
    batch_size: int = args.batch_size

    # Auto-detect project using the same cascade as the server
    # Pass cwd so level 2 (file path walk) can find .git
    config = ProjectDetectionConfig(explicit_project=settings.project)
    detector = ProjectDetector(config=config)
    detected = detector.detect(file_path=os.getcwd())
    detected_project = detected.project_id

    print("Spatial Memory - Backfill Project")
    print(f"Database path: {settings.memory_path}")

    # Resolve project name: explicit --project or auto-detected
    explicit_project: str | None = args.project
    if explicit_project:
        project_name = explicit_project
        print(f"Auto-detected project: {detected_project or '(none)'} (source: {detected.source})")
        print(f"Explicit --project:    {project_name}")
        if detected_project and project_name != detected_project:
            print()
            print(
                f"WARNING: The project you specified ('{project_name}') does not match "
                f"what the auto-detection cascade resolves ('{detected_project}')."
            )
            print(
                "New memories created by the server will use the auto-detected value. "
                "Using a different value for backfill will cause a project mismatch."
            )
            if not args.force:
                print(
                    "\nUse --force to proceed anyway, or omit --project to use "
                    "the auto-detected value."
                )
                return 1
            print("--force specified, proceeding with explicit value.\n")
    else:
        if not detected_project:
            print("Auto-detected project: (none)")
            print(
                "\nError: Could not auto-detect a project identity. "
                "Run this command from within a git repository, "
                "or specify --project explicitly."
            )
            return 1
        project_name = detected_project
        print(f"Auto-detected project: {project_name} (source: {detected.source})")

    if namespace_filter:
        print(f"Namespace filter: {namespace_filter}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    try:
        # Connect to database
        print("Connecting to database...")
        db = Database(
            storage_path=settings.memory_path,
            embedding_dim=384,  # Not needed for update operations
            auto_create_indexes=False,
        )
        db.connect()

        if db.table is None:
            print("No memories table found. Nothing to backfill.")
            db.close()
            return 0

        # Read all records to find those needing backfill
        print("Scanning memories...")
        arrow_table = db.table.to_arrow()

        if arrow_table.num_rows == 0:
            print("No memories found. Nothing to backfill.")
            db.close()
            return 0

        ids = arrow_table.column("id").to_pylist()
        namespaces = arrow_table.column("namespace").to_pylist()

        # Check if project column exists
        if "project" not in arrow_table.column_names:
            print("Error: 'project' column not found. Run 'spatial-memory migrate' first.")
            db.close()
            return 1

        projects = arrow_table.column("project").to_pylist()

        # Find records to update
        to_update: list[tuple[str, dict[str, str]]] = []
        for mem_id, ns, proj in zip(ids, namespaces, projects):
            # Skip if already has a project
            if proj:
                continue
            # Skip if namespace filter doesn't match
            if namespace_filter and ns != namespace_filter:
                continue
            to_update.append((mem_id, {"project": project_name}))

        if not to_update:
            total_with_project = sum(1 for p in projects if p)
            print("No memories need backfilling.")
            print(f"  Total memories: {arrow_table.num_rows}")
            print(f"  Already have project: {total_with_project}")
            if namespace_filter:
                ns_count = sum(1 for ns in namespaces if ns == namespace_filter)
                print(f"  In namespace '{namespace_filter}': {ns_count}")
            db.close()
            return 0

        # Show preview
        total_empty = sum(1 for p in projects if not p)
        print(f"Memories to update: {len(to_update)}")
        print(f"  Total memories: {arrow_table.num_rows}")
        print(f"  Without project: {total_empty}")
        if namespace_filter:
            print(f"  Matching namespace '{namespace_filter}': {len(to_update)}")
        print(f"  Will assign project: '{project_name}'")

        if dry_run:
            print(f"\n[DRY RUN] Would update {len(to_update)} memories.")
            print("Run without --dry-run to apply.")
            db.close()
            return 0

        # Confirm before applying
        if not args.yes:
            print()
            response = (
                input(f"Assign project '{project_name}' to {len(to_update)} memories? [y/N] ")
                .strip()
                .lower()
            )
            if response not in ("y", "yes"):
                print("Aborted.")
                db.close()
                return 0

        # Apply updates in batches
        print(f"\nUpdating {len(to_update)} memories...")
        total_updated = 0

        for i in range(0, len(to_update), batch_size):
            batch = to_update[i : i + batch_size]
            success_count, failed = db.update_batch(batch)
            total_updated += success_count
            if failed:
                print(f"  Warning: {len(failed)} updates failed in batch")
            progress = min(i + batch_size, len(to_update))
            print(f"  Progress: {progress}/{len(to_update)} ({total_updated} updated)")

        print(
            f"\nDone. Updated {total_updated}/{len(to_update)} memories "
            f"with project '{project_name}'."
        )

        db.close()
        return 0

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=args.verbose)
        print(f"\nError: {e}")
        return 1


def run_version() -> None:
    """Print version information."""
    from spatial_memory import __version__

    print(f"spatial-memory {__version__}")


def run_setup_hooks(args: argparse.Namespace) -> int:
    """Generate hook configuration for cognitive offloading.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    import json as json_mod

    from spatial_memory.tools.setup_hooks import generate_hook_config

    try:
        config = generate_hook_config(
            client=args.client,
            python_path=args.python_path or "",
            include_session_start=not args.no_session_start,
            include_mcp_config=not args.no_mcp_config,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if args.json:
        print(json_mod.dumps(config, indent=2))
        return 0

    print("Spatial Memory - Hook Configuration")
    print(f"Client: {config['client']}")
    print(f"Python: {config['paths']['python']}")
    print(f"Hooks dir: {config['paths']['hooks_dir']}")
    print()

    if config.get("hooks"):
        print("Hooks config:")
        print(json_mod.dumps({"hooks": config["hooks"]}, indent=2))
        print()

    if config.get("mcp_config"):
        print("MCP server config:")
        print(json_mod.dumps(config["mcp_config"], indent=2))
        print()

    print(config.get("instructions", ""))
    return 0


def run_instructions() -> None:
    """Print the MCP server instructions that are auto-injected into Claude's context."""
    from spatial_memory.server import SpatialMemoryServer

    instructions = SpatialMemoryServer._get_server_instructions()
    print(instructions)


def main() -> NoReturn:
    """Main entry point with subcommand support."""
    parser = argparse.ArgumentParser(
        prog="spatial-memory",
        description="Spatial Memory MCP Server and CLI tools",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # Server command (default)
    subparsers.add_parser(
        "serve",
        help="Start the MCP server (default if no command given)",
    )

    # Instructions command
    subparsers.add_parser(
        "instructions",
        help="Show the MCP instructions injected into Claude's context",
    )

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Run database migrations",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migrations without applying",
    )
    migrate_parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status and exit",
    )
    migrate_parser.add_argument(
        "--rollback",
        metavar="VERSION",
        help="Rollback to specified version (e.g., 1.0.0)",
    )
    migrate_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    migrate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Setup-hooks command
    setup_hooks_parser = subparsers.add_parser(
        "setup-hooks",
        help="Generate hook configuration for cognitive offloading",
    )
    from spatial_memory.tools.setup_hooks import SUPPORTED_CLIENTS

    setup_hooks_parser.add_argument(
        "--client",
        default="claude-code",
        choices=list(SUPPORTED_CLIENTS),
        help="Target client (default: claude-code)",
    )
    setup_hooks_parser.add_argument(
        "--python-path",
        default="",
        help="Python interpreter path (default: auto-detect)",
    )
    setup_hooks_parser.add_argument(
        "--no-session-start",
        action="store_true",
        help="Exclude the SessionStart hook",
    )
    setup_hooks_parser.add_argument(
        "--no-mcp-config",
        action="store_true",
        help="Exclude MCP server configuration",
    )
    setup_hooks_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON only (for piping)",
    )

    # Backfill-project command
    backfill_parser = subparsers.add_parser(
        "backfill-project",
        help="Assign a project name to existing memories",
        description=(
            "Assigns a project identifier to memories with an empty project field. "
            "By default, auto-detects the project from the current directory using "
            "the same cascade as the server (git remote URL, env vars, config). "
            "Use --project to override, but a warning is shown if it differs from "
            "the auto-detected value."
        ),
    )
    backfill_parser.add_argument(
        "--project",
        default=None,
        help=(
            "Project identifier to assign. If omitted, auto-detects from the "
            "current directory (e.g., github.com/org/repo from git remote)"
        ),
    )
    backfill_parser.add_argument(
        "--namespace",
        default=None,
        help="Only update memories in this namespace",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )
    backfill_parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if --project differs from auto-detected value",
    )
    backfill_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records per batch update (default: 500)",
    )
    backfill_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    backfill_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.version:
        run_version()
        sys.exit(0)

    if args.command == "instructions":
        run_instructions()
        sys.exit(0)
    elif args.command == "setup-hooks":
        sys.exit(run_setup_hooks(args))
    elif args.command == "migrate":
        sys.exit(run_migrate(args))
    elif args.command == "backfill-project":
        sys.exit(run_backfill_project(args))
    elif args.command == "serve" or args.command is None:
        # Default to running the server
        run_server()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
