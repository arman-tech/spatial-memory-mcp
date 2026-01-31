"""Integration test fixtures with module-scoped database and session-scoped embeddings.

Fixture Architecture
====================

This module implements a multi-tier fixture strategy optimized for performance
while maintaining test isolation:

1. SESSION-SCOPED (loaded once per pytest run):
   - session_embedding_service: Loads the ~80MB embedding model once

2. MODULE-SCOPED (shared within each test file):
   - module_temp_storage: Temporary directory for database files
   - module_settings: Configuration with isolated storage path
   - module_database: LanceDB connection (expensive to create)

3. FUNCTION-SCOPED with AUTOUSE (runs for every test):
   - cleanup_database: Clears all data after each test for isolation

Performance Impact
==================

- Before: ~500ms per test (creating new database each time)
- After: ~50ms per test (reusing database, just clearing data)

The database connection and schema creation happen once per test module,
while data cleanup between tests is fast (~5ms).

Backward Compatibility
======================

Aliases are provided so existing tests continue to work without modification:
- integration_database -> module_database
- integration_temp_storage -> module_temp_storage
- integration_settings -> module_settings
- database -> integration_database
- temp_storage -> integration_temp_storage
- embedding_service -> session_embedding_service
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.server import SpatialMemoryServer


# ===========================================================================
# Session-Scoped Fixtures (loaded once per pytest run)
# ===========================================================================


@pytest.fixture(scope="session")
def session_embedding_service() -> Generator[EmbeddingService, None, None]:
    """Session-scoped embedding service.

    Loads the ~80MB all-MiniLM-L6-v2 model once per test session instead of
    once per test, reducing memory usage and initialization time by 80-90%.

    Yields:
        EmbeddingService: Initialized embedding service ready for use.
    """
    service = EmbeddingService("all-MiniLM-L6-v2")
    yield service


# ===========================================================================
# Module-Scoped Fixtures (shared within each test file)
# ===========================================================================


@pytest.fixture(scope="module")
def module_temp_storage() -> Generator[Path, None, None]:
    """Module-scoped temporary storage directory.

    Creates a temporary directory that is shared across all tests in a module.
    The directory is cleaned up when the module finishes.

    Yields:
        Path: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def module_settings(module_temp_storage: Path) -> Generator[Settings, None, None]:
    """Module-scoped settings with isolated storage path.

    Provides consistent settings for all tests in a module, pointing to
    the module's temporary storage directory.

    Args:
        module_temp_storage: Temporary directory for this test module.

    Yields:
        Settings: Configured settings instance.
    """
    # Create exports/imports directories for Phase 5 tests
    exports_dir = module_temp_storage / "exports"
    imports_dir = module_temp_storage / "imports"
    exports_dir.mkdir(exist_ok=True)
    imports_dir.mkdir(exist_ok=True)

    settings = Settings(
        memory_path=module_temp_storage / "test-memory",
        embedding_model="all-MiniLM-L6-v2",
        log_level="DEBUG",
        # Phase 5: Configure allowed paths for export/import tests
        export_allowed_paths=[str(exports_dir), str(module_temp_storage)],
        import_allowed_paths=[str(imports_dir), str(module_temp_storage)],
    )
    override_settings(settings)
    yield settings
    reset_settings()


@pytest.fixture(scope="module")
def module_database(module_settings: Settings) -> Generator[Database, None, None]:
    """Module-scoped database connection.

    Creates a single database connection shared across all tests in a module.
    This avoids the overhead of creating a new database for each test (~500ms).
    Test isolation is maintained via the cleanup_database fixture.

    Args:
        module_settings: Settings with the database path.

    Yields:
        Database: Connected database instance.
    """
    db = Database(module_settings.memory_path)
    db.connect()
    yield db
    db.close()


# ===========================================================================
# Function-Scoped Autouse Cleanup (runs after every test)
# ===========================================================================


@pytest.fixture(autouse=True)
def cleanup_database(module_database: Database) -> Generator[None, None, None]:
    """Cleanup database before and after each test for isolation.

    This autouse fixture ensures test isolation by clearing all data from the
    database both before and after each test. The pre-test cleanup ensures
    tests always start with a clean slate, even if a previous test's cleanup
    failed or crashed.

    Using clear_all() is much faster than recreating the database connection
    and schema (~5ms vs ~500ms).

    Args:
        module_database: The shared database connection to clean.

    Yields:
        None: Control returns to the test after pre-cleanup.
    """
    # Pre-test cleanup: ensure clean state even if previous cleanup failed
    try:
        module_database.clear_all()
    except Exception:
        pass

    yield

    # Post-test cleanup: prepare for next test
    try:
        module_database.clear_all()
    except Exception:
        pass  # Ignore cleanup errors to avoid masking test failures


# ===========================================================================
# Module-Scoped Repository and Server
# ===========================================================================


@pytest.fixture(scope="module")
def module_repository(module_database: Database) -> LanceDBMemoryRepository:
    """Module-scoped repository adapter.

    Args:
        module_database: The shared database connection.

    Returns:
        LanceDBMemoryRepository: Repository for memory operations.
    """
    return LanceDBMemoryRepository(module_database)


@pytest.fixture(scope="module")
def module_server(
    module_repository: LanceDBMemoryRepository,
    session_embedding_service: EmbeddingService,
) -> SpatialMemoryServer:
    """Module-scoped MCP server.

    Args:
        module_repository: The shared repository.
        session_embedding_service: The session-scoped embedding service.

    Returns:
        SpatialMemoryServer: Configured server instance.
    """
    return SpatialMemoryServer(
        repository=module_repository,
        embeddings=session_embedding_service,
    )


# ===========================================================================
# Backward Compatibility Aliases
# ===========================================================================
# These aliases ensure existing tests continue to work without modification.
# They delegate to the new module-scoped fixtures.


@pytest.fixture
def integration_temp_storage(module_temp_storage: Path) -> Path:
    """Alias for module_temp_storage.

    Provides backward compatibility for tests using 'integration_temp_storage'.

    Args:
        module_temp_storage: The module-scoped temporary directory.

    Returns:
        Path: The temporary directory path.
    """
    return module_temp_storage


@pytest.fixture
def integration_settings(module_settings: Settings) -> Settings:
    """Alias for module_settings.

    Provides backward compatibility for tests using 'integration_settings'.

    Args:
        module_settings: The module-scoped settings.

    Returns:
        Settings: The settings instance.
    """
    return module_settings


@pytest.fixture
def integration_database(module_database: Database) -> Database:
    """Alias for module_database.

    Provides backward compatibility for tests using 'integration_database'.

    Args:
        module_database: The module-scoped database.

    Returns:
        Database: The database instance.
    """
    return module_database


@pytest.fixture
def integration_repository(module_repository: LanceDBMemoryRepository) -> LanceDBMemoryRepository:
    """Alias for module_repository.

    Provides backward compatibility for tests using 'integration_repository'.

    Args:
        module_repository: The module-scoped repository.

    Returns:
        LanceDBMemoryRepository: The repository instance.
    """
    return module_repository


@pytest.fixture
def integration_server(module_server: SpatialMemoryServer) -> SpatialMemoryServer:
    """Alias for module_server.

    Provides backward compatibility for tests using 'integration_server'.

    Args:
        module_server: The module-scoped server.

    Returns:
        SpatialMemoryServer: The server instance.
    """
    return module_server


# ---------------------------------------------------------------------------
# Short Aliases for Convenience
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_service(session_embedding_service: EmbeddingService) -> EmbeddingService:
    """Short alias for session_embedding_service.

    Provides backward compatibility for tests using 'embedding_service'.

    Args:
        session_embedding_service: The session-scoped embedding service.

    Returns:
        EmbeddingService: The embedding service instance.
    """
    return session_embedding_service


@pytest.fixture
def database(integration_database: Database) -> Database:
    """Short alias for integration_database.

    Provides backward compatibility for tests using 'database'.

    Args:
        integration_database: The integration database alias.

    Returns:
        Database: The database instance.
    """
    return integration_database


@pytest.fixture
def temp_storage(integration_temp_storage: Path, request: pytest.FixtureRequest) -> Path:
    """Per-test temporary storage directory.

    Creates a unique subdirectory for each test to avoid path conflicts
    when tests create their own Database instances.

    Args:
        integration_temp_storage: The module-level temp storage.
        request: Pytest request fixture for test name.

    Returns:
        Path: Unique temporary directory for this test.
    """
    # Create a unique subdirectory for this test
    test_dir = integration_temp_storage / f"test-{request.node.name}"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def test_settings(integration_settings: Settings) -> Settings:
    """Alias for integration_settings.

    Provides backward compatibility for tests using 'test_settings'.

    Args:
        integration_settings: The integration settings alias.

    Returns:
        Settings: The settings instance.
    """
    return integration_settings
