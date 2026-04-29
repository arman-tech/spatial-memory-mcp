"""Entry point for running the Spatial Memory MCP Server and CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from spatial_memory.config import Settings
    from spatial_memory.factory import ServiceContainer

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


def run_consolidate(args: argparse.Namespace) -> int:
    """Run memory consolidation for a namespace.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from spatial_memory.config import get_settings
    from spatial_memory.factory import ServiceFactory

    settings = get_settings()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    if not args.verbose:
        # Suppress noisy third-party library warnings
        for name in ("optimum", "onnxruntime", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        import warnings

        warnings.filterwarnings("ignore", message="Multiple distributions")
        warnings.filterwarnings("ignore", message="Multiple ONNX files")

    dry_run = not args.no_dry_run
    mode = "dry run" if dry_run else "LIVE"
    print(f'Consolidating namespace "{args.namespace}" ({mode})...')
    print(f"Database path: {settings.memory_path}")
    print()

    try:
        factory = ServiceFactory(settings)
        if args.verbose:
            embeddings = factory.create_embedding_service()
            database = factory.create_database(embeddings.dimensions)
        else:
            # Suppress noisy stdout from optimum/onnxruntime during model load
            import contextlib
            import io

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                embeddings = factory.create_embedding_service()
                database = factory.create_database(embeddings.dimensions)
        repository = factory.create_repository(database)
        lifecycle = factory.create_lifecycle_service(repository, embeddings)

        result = lifecycle.consolidate(
            namespace=args.namespace,
            project=args.project,
            similarity_threshold=args.similarity,
            strategy=args.strategy,
            dry_run=dry_run,
            max_groups=args.max_groups,
        )

        if result.groups_found == 0:
            print("No duplicate groups found.")
            database.close()
            return 0

        total_members = sum(len(g.member_ids) for g in result.groups)
        print(f"Found {result.groups_found} duplicate group(s) ({total_members} memories total)")
        print()

        if args.verbose:
            for i, group in enumerate(result.groups, 1):
                print(f"Group {i} (similarity: {group.avg_similarity:.2f})")
                for mid in group.member_ids:
                    label = "Keep" if mid == group.representative_id else "Merge"
                    print(f"  {label:>5}:  {mid}")
                print()

        action = "would be" if dry_run else "were"
        print(
            f"Summary: {result.groups_found} group(s), "
            f"{result.memories_merged} memories {action} merged, "
            f"{result.memories_deleted} {action} deleted"
        )
        if dry_run:
            print("Re-run with --no-dry-run to apply changes.")

        database.close()
        return 0

    except Exception as e:
        logger.error(f"Consolidation failed: {e}", exc_info=args.verbose)
        print(f"\nError: {e}")
        return 1


def run_namespaces(args: argparse.Namespace) -> int:
    """List all namespaces in the database.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from spatial_memory.config import get_settings
    from spatial_memory.core.database import Database

    settings = get_settings()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        db = Database(
            storage_path=settings.memory_path,
            embedding_dim=384,  # Not needed for namespace listing
            auto_create_indexes=False,
        )
        db.connect()

        if db.table is None:
            print("No memories table found.")
            db.close()
            return 0

        # Read namespaces directly from the table
        arrow_table = db.table.to_arrow()
        if arrow_table.num_rows == 0:
            print("No memories found.")
            db.close()
            return 0

        namespaces_col = arrow_table.column("namespace").to_pylist()

        # Count per namespace
        counts: dict[str, int] = {}
        for ns in namespaces_col:
            counts[ns] = counts.get(ns, 0) + 1

        # Sort by name
        print(f"{'Namespace':<30} {'Memories':>10}")
        print(f"{'-' * 30} {'-' * 10}")
        for ns in sorted(counts):
            print(f"{ns:<30} {counts[ns]:>10}")
        print(f"{'-' * 30} {'-' * 10}")
        print(f"{'Total':<30} {arrow_table.num_rows:>10}")
        print(f"\n{len(counts)} namespace(s)")

        db.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}", exc_info=args.verbose)
        print(f"\nError: {e}")
        return 1


def _build_container_quiet(verbose: bool) -> tuple[Settings, ServiceContainer]:
    """Build a ServiceContainer, suppressing model-load chatter unless verbose."""
    from spatial_memory.config import get_settings
    from spatial_memory.factory import ServiceFactory

    settings = get_settings()
    factory = ServiceFactory(settings)
    if verbose:
        return settings, factory.create_all()

    import contextlib
    import io
    import warnings

    for name in ("optimum", "onnxruntime", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Multiple distributions")
    warnings.filterwarnings("ignore", message="Multiple ONNX files")

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        container = factory.create_all()
    return settings, container


def run_export(args: argparse.Namespace) -> int:
    """Export memories to a file via ExportImportService.

    Exit codes:
      0 success
      1 validation error (bad format / no namespace match / config error)
      2 write error
    """
    from spatial_memory.core.errors import (
        ConfigurationError,
        ExportError,
        PathSecurityError,
        ValidationError,
    )

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    container: ServiceContainer | None = None
    try:
        _settings, container = _build_container_quiet(args.verbose)
    except ConfigurationError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 2

    try:
        result = container.export_import.export_memories(
            output_path=args.output,
            format=args.format,
            namespace=args.namespace,
            project=args.project,
            include_vectors=args.include_vectors,
        )
        print(f"Exported {result.memories_exported} memories")
        print(f"Output: {result.output_path}")
        print(f"Format: {result.format}")
        print(f"Size:   {result.file_size_mb:.2f} MB")
        if result.namespaces_included:
            print(f"Namespaces: {', '.join(result.namespaces_included)}")
        return 0
    except (ValidationError, PathSecurityError) as e:
        print(f"Error: {e}")
        return 1
    except ExportError as e:
        print(f"Error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 2
    finally:
        if container is not None and container.database is not None:
            container.database.close()


def run_import(args: argparse.Namespace) -> int:
    """Import memories from a file via ExportImportService.

    Exit codes:
      0 success
      1 validation/import errors (>0 records failed)
      2 read error
    """
    from spatial_memory.core.errors import (
        ConfigurationError,
        FileSizeLimitError,
        MemoryImportError,
        PathSecurityError,
        ValidationError,
    )

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    container: ServiceContainer | None = None
    try:
        _settings, container = _build_container_quiet(args.verbose)
    except ConfigurationError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 2

    try:
        result = container.export_import.import_memories(
            source_path=args.input,
            format=args.format,
            namespace_override=args.namespace_override,
            deduplicate=args.deduplicate,
            dedup_threshold=args.dedup_threshold,
            validate=not args.no_validate,
            regenerate_embeddings=args.regenerate_embeddings,
            dry_run=args.dry_run,
        )
        mode = "DRY RUN — no changes applied" if args.dry_run else "imported"
        print(f"{mode}: {result.memories_imported}/{result.total_records_in_file} records")
        if result.memories_skipped:
            print(f"Skipped: {result.memories_skipped}")
        if result.memories_failed:
            print(f"Failed:  {result.memories_failed}")
        if result.validation_errors:
            print(f"Validation errors ({len(result.validation_errors)}):")
            for err in result.validation_errors[:10]:
                print(f"  row {err.row_number} [{err.field}]: {err.error}")
            if len(result.validation_errors) > 10:
                print(f"  ... and {len(result.validation_errors) - 10} more")
        if result.namespace_override:
            print(f"Namespace override: {result.namespace_override}")
        print(f"Duration: {result.duration_seconds:.2f}s")

        if result.memories_failed > 0:
            return 1
        return 0
    except (ValidationError, PathSecurityError, FileSizeLimitError) as e:
        print(f"Error: {e}")
        return 1
    except MemoryImportError as e:
        print(f"Error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 2
    finally:
        if container is not None and container.database is not None:
            container.database.close()


def run_version() -> None:
    """Print version information."""
    from spatial_memory import __version__

    print(f"spatial-memory {__version__}")


_OPENAI_R17_MESSAGE = (
    "v2.0 removed the OpenAI embedding provider (D1). "
    "Either accept the local arctic-m-long default (export from v1, install v2 fresh, "
    "import) or pin v1.x."
)


def _doctor_collect(verbose: bool) -> dict[str, Any]:
    """Collect diagnostic information for the doctor subcommand.

    Returns a dict with database/embedding/health sections. The dict carries an
    internal "_status" key with values: "ok", "warning", "error" used to derive
    the exit code. Any keys prefixed with "_" are stripped before --json output.

    R17 mitigation: a configured ``openai:*`` prefix is detected BEFORE
    ``resolve_model`` runs so v1 users see an actionable migration message
    instead of a generic registry rejection.
    """
    from spatial_memory.config import get_settings
    from spatial_memory.core.db_migrations import CURRENT_SCHEMA_VERSION, MigrationManager
    from spatial_memory.core.embedding_registry import resolve_model

    settings = get_settings()

    # R17: surface legacy openai:* configuration before any model resolution.
    if settings.embedding_model.startswith("openai:"):
        return {
            "database": {
                "path": str(settings.memory_path),
                "exists": Path(settings.memory_path).exists(),
                "memories": 0,
                "schema_version": None,
                "schema_target": CURRENT_SCHEMA_VERSION,
            },
            "embedding": {
                "configured_alias": settings.embedding_model,
                "configured_hf_name": None,
                "configured_dim": 0,
                "configured_max_seq_length": None,
                "query_prefix": None,
                "passage_prefix": None,
                "stored_alias": None,
                "stored_dim": 0,
                "match": "openai-legacy",
                "service_responsive": None,
            },
            "health": {"vector_index": None, "fts_index": None},
            "_status": "error",
            "_messages": [
                _OPENAI_R17_MESSAGE,
                f"Currently configured: SPATIAL_MEMORY_EMBEDDING_MODEL={settings.embedding_model}",
            ],
        }

    configured_cfg = resolve_model(settings.embedding_model)

    report: dict[str, Any] = {
        "database": {
            "path": str(settings.memory_path),
            "exists": Path(settings.memory_path).exists(),
            "memories": 0,
            "schema_version": None,
            "schema_target": CURRENT_SCHEMA_VERSION,
        },
        "embedding": {
            "configured_alias": configured_cfg.alias,
            "configured_hf_name": configured_cfg.hf_name,
            "configured_dim": configured_cfg.dim,
            "configured_max_seq_length": configured_cfg.max_seq_length,
            "query_prefix": configured_cfg.query_prefix,
            "passage_prefix": configured_cfg.passage_prefix,
            "stored_alias": None,
            "stored_dim": 0,
            "match": None,
            "service_responsive": False,
        },
        "health": {
            "vector_index": None,
            "fts_index": None,
        },
        "_status": "ok",
        "_messages": [],
    }

    from spatial_memory.core.database import Database

    try:
        db = Database(
            storage_path=settings.memory_path,
            embedding_dim=configured_cfg.dim,
            auto_create_indexes=False,
        )
        db.connect()
    except Exception as e:
        report["_status"] = "error"
        report["_messages"].append(f"Cannot open database: {e}")
        return report

    try:
        try:
            report["database"]["memories"] = db.count_memories()
        except Exception:
            report["database"]["memories"] = 0

        try:
            mgr = MigrationManager(db)
            mgr.register_builtin_migrations()
            report["database"]["schema_version"] = mgr.get_current_version()
        except Exception:
            report["database"]["schema_version"] = "unknown"

        stored_alias = db.get_system_metadata("embedding_model")
        stored_dim = db.detect_vector_dim()
        report["embedding"]["stored_alias"] = stored_alias
        report["embedding"]["stored_dim"] = stored_dim

        # R17 (defense in depth): also check stored alias for legacy openai:*.
        if stored_alias and stored_alias.startswith("openai:"):
            report["embedding"]["match"] = "openai-legacy"
            report["_status"] = "error"
            report["_messages"].append(_OPENAI_R17_MESSAGE)
            report["_messages"].append(
                f"Stored DB metadata: embedding_model={stored_alias}"
            )
            return report

        memories = report["database"]["memories"]
        configured_alias = configured_cfg.alias

        if stored_alias is None:
            report["embedding"]["match"] = "fresh"
            report["_messages"].append(
                f"Fresh database — first ingest will pin embedding_model='{configured_alias}'."
            )
        elif stored_alias == configured_alias:
            report["embedding"]["match"] = "match"
        elif configured_cfg.dim and stored_dim and configured_cfg.dim == stored_dim:
            report["embedding"]["match"] = "same-dim-mismatch"
            report["_status"] = "warning"
            report["_messages"].append(
                f"Embedding model mismatch: configured={configured_alias} vs "
                f"stored={stored_alias} (both {configured_cfg.dim}-dim). To realign, run:\n"
                f"  spatial-memory reembed --to={configured_alias}"
            )
        else:
            report["embedding"]["match"] = "cross-dim-mismatch"
            report["_status"] = "warning"
            report["_messages"].append(
                f"Embedding model mismatch: configured={configured_alias} "
                f"(dim={configured_cfg.dim}) vs stored={stored_alias} (dim={stored_dim}). "
                f"To migrate, export from old model and re-import under new:\n"
                f"  spatial-memory export -o backup.json\n"
                f"  rm -rf {settings.memory_path}\n"
                f"  spatial-memory import -i backup.json"
            )

        try:
            report["health"]["vector_index"] = bool(db._has_vector_index)
            report["health"]["fts_index"] = bool(db._has_fts_index)
        except Exception:
            pass

        if (
            report["health"]["vector_index"] is False
            and memories >= 10_000
            and report["_status"] == "ok"
        ):
            report["_status"] = "warning"
            report["_messages"].append(
                f"Vector index missing despite {memories} memories — recall will be slow."
            )

        if configured_cfg.dim > 0:
            try:
                import contextlib
                import io

                from spatial_memory.core.embeddings import EmbeddingService

                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    svc = EmbeddingService(model_name=settings.embedding_model)
                    _ = svc.embed_passage("doctor health check")
                report["embedding"]["service_responsive"] = True
            except Exception as e:
                report["embedding"]["service_responsive"] = False
                if report["_status"] == "ok":
                    report["_status"] = "error"
                report["_messages"].append(f"Embedding service failed to load: {e}")
        else:
            report["embedding"]["service_responsive"] = None
    finally:
        db.close()

    return report


def _doctor_format_text(report: dict[str, Any]) -> str:
    db = report["database"]
    emb = report["embedding"]
    health = report["health"]

    lines: list[str] = []
    lines.append("Spatial Memory — Doctor Report")
    lines.append("=" * 50)

    lines.append("\nDatabase:")
    lines.append(f"  Path:           {db['path']}")
    lines.append(f"  Exists:         {db['exists']}")
    lines.append(f"  Memories:       {db['memories']}")
    lines.append(
        f"  Schema version: {db['schema_version']} (target: {db['schema_target']})"
    )

    lines.append("\nEmbedding model:")
    lines.append(f"  Configured alias: {emb['configured_alias']}")
    lines.append(f"  HF name:          {emb['configured_hf_name']}")
    lines.append(f"  Configured dim:   {emb['configured_dim']}")
    lines.append(f"  Max seq length:   {emb['configured_max_seq_length']}")
    lines.append(f"  Query prefix:     {emb['query_prefix']!r}")
    lines.append(f"  Passage prefix:   {emb['passage_prefix']!r}")
    lines.append(f"  Stored alias:     {emb['stored_alias']}")
    lines.append(f"  Stored dim:       {emb['stored_dim']}")
    lines.append(f"  Match:            {emb['match']}")
    lines.append(f"  Service OK:       {emb['service_responsive']}")

    lines.append("\nHealth:")
    lines.append(f"  Vector index: {health['vector_index']}")
    lines.append(f"  FTS index:    {health['fts_index']}")

    if report["_messages"]:
        lines.append("\nNotes:")
        for msg in report["_messages"]:
            lines.append(f"  - {msg}")

    status_label = {
        "ok": "OK",
        "warning": "WARNINGS",
        "error": "ERRORS",
    }.get(report["_status"], "UNKNOWN")
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"Status: {status_label}")
    return "\n".join(lines)


def run_doctor(args: argparse.Namespace) -> int:
    """Print a diagnostic report on the current configuration and database.

    Exit codes:
      0 all green
      1 warnings (mismatch, missing index)
      2 errors (DB missing, model load failed, legacy openai:* configuration)
    """
    import json as json_mod

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        report = _doctor_collect(args.verbose)
    except Exception as e:
        logger.error(f"doctor failed: {e}", exc_info=args.verbose)
        print(f"Error: {e}")
        return 2

    if args.json:
        public = {k: v for k, v in report.items() if not k.startswith("_")}
        public["status"] = report["_status"]
        public["messages"] = report["_messages"]
        print(json_mod.dumps(public, indent=2, default=str))
    else:
        print(_doctor_format_text(report))

    if report["_status"] == "error":
        return 2
    if report["_status"] == "warning":
        return 1
    return 0


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


def _dispatch_hook() -> None:
    """Load and run dispatcher.py via importlib (bypasses heavy package init).

    Called when ``sys.argv[1] == "hook"``.  Edge cases (no args, --help)
    print usage and return.
    """
    import importlib.util as ilu

    argv_rest = sys.argv[2:]  # everything after "hook"
    if not argv_rest or argv_rest[0] in ("--help", "-h"):
        print(
            "Usage: spatial-memory hook <event> [--client <client>]\n"
            "Events: session-start, post-tool-use, pre-compact, stop"
        )
        return

    dispatcher_path = Path(__file__).resolve().parent / "hooks" / "dispatcher.py"
    spec = ilu.spec_from_file_location("spatial_memory.hooks.dispatcher", str(dispatcher_path))
    if spec is None or spec.loader is None:
        return
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Shift argv so dispatcher sees event as argv[1]
    sys.argv = [str(dispatcher_path)] + argv_rest
    mod.main()


def main() -> NoReturn:
    """Main entry point with subcommand support."""
    # Fast-path: bypass argparse entirely for hook dispatch
    if len(sys.argv) >= 2 and sys.argv[1] == "hook":
        try:
            _dispatch_hook()
        except Exception:
            pass  # Fail-open
        sys.exit(0)

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
    setup_hooks_parser.add_argument(
        "--client",
        default="claude-code",
        choices=["claude-code", "cursor"],
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

    # Consolidate command
    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Merge duplicate memories in a namespace",
        description=(
            "Finds and merges semantically similar memories within a namespace. "
            "Dry run is the default (safe). Use --no-dry-run to actually apply merges."
        ),
    )
    consolidate_parser.add_argument(
        "namespace",
        help="Namespace to consolidate",
    )
    consolidate_parser.add_argument(
        "--similarity",
        type=float,
        default=0.85,
        help="Similarity threshold (0.7-0.99, default: 0.85)",
    )
    consolidate_parser.add_argument(
        "--strategy",
        default="keep_highest_importance",
        choices=["keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"],
        help="Merge strategy (default: keep_highest_importance)",
    )
    consolidate_parser.add_argument(
        "--max-groups",
        type=int,
        default=50,
        help="Max duplicate groups to process (default: 50)",
    )
    consolidate_parser.add_argument(
        "--project",
        default=None,
        help="Project scope (default: auto-detect)",
    )
    consolidate_parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually perform merges (default is dry run)",
    )
    consolidate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed per-group info",
    )

    # Namespaces command
    namespaces_parser = subparsers.add_parser(
        "namespaces",
        help="List all namespaces with memory counts",
    )
    namespaces_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export memories to a file (parquet/json/csv)",
        description=(
            "Export memories from the database to a file. Format is inferred "
            "from the output path extension (.parquet, .json, .csv) when "
            "--format is omitted."
        ),
    )
    export_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path (extension determines format if --format omitted)",
    )
    export_parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv"],
        default=None,
        help="Export format (default: inferred from --output extension)",
    )
    export_parser.add_argument(
        "--namespace",
        default=None,
        help="Export only memories in this namespace (default: all)",
    )
    export_parser.add_argument(
        "--project",
        default=None,
        help="Export only memories matching this project (default: all)",
    )
    export_parser.add_argument(
        "--include-vectors",
        dest="include_vectors",
        action="store_true",
        default=True,
        help="Include embedding vectors in export (default)",
    )
    export_parser.add_argument(
        "--no-include-vectors",
        dest="include_vectors",
        action="store_false",
        help="Omit embedding vectors (smaller files; requires regenerate on import)",
    )
    export_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Import command
    import_parser = subparsers.add_parser(
        "import",
        help="Import memories from a file (parquet/json/csv)",
        description=(
            "Import memories from a file into the database. NOTE: --dry-run "
            "defaults to FALSE on the CLI (the opposite of the MCP tool default). "
            "Pass --dry-run to validate without writing."
        ),
    )
    import_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Source file path (extension determines format if --format omitted)",
    )
    import_parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv"],
        default=None,
        help="Import format (default: inferred from --input extension)",
    )
    import_parser.add_argument(
        "--namespace-override",
        default=None,
        help="Override the namespace for all imported memories",
    )
    import_parser.add_argument(
        "--regenerate-embeddings",
        action="store_true",
        default=False,
        help="Generate new embeddings (required if source lacks vectors or dims differ)",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate without writing (CLI default is False — opposite of MCP tool)",
    )
    import_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip per-record validation (default: validate)",
    )
    import_parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=False,
        help="Skip records similar to existing memories",
    )
    import_parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Deduplication similarity threshold (0.7-0.99, default: 0.95)",
    )
    import_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Print diagnostics: configured vs stored embedding model, health, indexes",
        description=(
            "Prints a diagnostic report on the database and configured embedding "
            "model. Surfaces fresh / mismatch / legacy-openai states with the same "
            "format as the factory startup error so users get a copy-pasteable "
            "migration recipe."
        ),
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a structured JSON object instead of text",
    )
    doctor_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Plugin-mode command
    plugin_mode_parser = subparsers.add_parser(
        "plugin-mode",
        help="Switch plugin between dev (local source) and prod (PyPI/uvx) modes",
    )
    plugin_mode_parser.add_argument(
        "target_mode",
        choices=["dev", "prod", "status"],
        help="Target mode: dev (python -m), prod (uvx), or status",
    )

    # Init command (Cursor only)
    init_parser = subparsers.add_parser(
        "init",
        help="Auto-configure a client for spatial-memory",
        description="Creates MCP config, hooks, and rules files for Cursor.",
    )
    init_parser.add_argument(
        "--client",
        required=True,
        choices=["cursor"],
        help="Target client to configure",
    )
    init_parser.add_argument(
        "--project",
        default=None,
        help="Project name for memory scoping (default: current directory name)",
    )
    init_parser.add_argument(
        "--mode",
        default="prod",
        choices=["dev", "prod"],
        help="Server mode: prod (uvx, default) or dev (local python -m)",
    )
    init_parser.add_argument(
        "--global",
        dest="global_scope",
        action="store_true",
        help="Configure globally (~/.cursor/) instead of project scope",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration",
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
    elif args.command == "consolidate":
        sys.exit(run_consolidate(args))
    elif args.command == "namespaces":
        sys.exit(run_namespaces(args))
    elif args.command == "export":
        sys.exit(run_export(args))
    elif args.command == "import":
        sys.exit(run_import(args))
    elif args.command == "doctor":
        sys.exit(run_doctor(args))
    elif args.command == "plugin-mode":
        from spatial_memory.tools.plugin_mode import run_plugin_mode

        sys.exit(run_plugin_mode(args))
    elif args.command == "init":
        from spatial_memory.tools.init_client import run_init

        sys.exit(run_init(args))
    elif args.command == "serve" or args.command is None:
        # Default to running the server
        run_server()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
