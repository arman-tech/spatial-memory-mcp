"""Shared fixtures for unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from spatial_memory.adapters.project_detection import (
    ProjectDetectionConfig,
    ProjectDetector,
)


def make_server_with_mocks(
    *,
    project_detector: ProjectDetector | None = None,
) -> tuple:
    """Build a SpatialMemoryServer instance with mocked services.

    Returns (server, mocks_dict) where mocks_dict has keys for each service.
    """
    # We avoid constructing a full SpatialMemoryServer because it calls
    # get_settings / ServiceFactory.  Instead, build the minimum needed
    # by patching __init__.
    from spatial_memory.server import SpatialMemoryServer

    # Create mock services
    memory_svc = MagicMock()
    spatial_svc = MagicMock()
    lifecycle_svc = MagicMock()
    utility_svc = MagicMock()
    export_import_svc = MagicMock()
    detector = project_detector or ProjectDetector(ProjectDetectionConfig())

    with patch.object(SpatialMemoryServer, "__init__", lambda self, **kw: None):
        server = SpatialMemoryServer()

    # Wire attributes the handlers need
    server._memory_service = memory_svc
    server._spatial_service = spatial_svc
    server._lifecycle_service = lifecycle_svc
    server._utility_service = utility_svc
    server._export_import_service = export_import_svc
    server._project_detector = detector
    server._decay_manager = None
    server._cache_enabled = False
    server._cache = None
    # Provide minimal settings for handlers that read config
    settings_mock = MagicMock()
    settings_mock.cognitive_offloading_enabled = False
    settings_mock.signal_threshold = 0.3
    settings_mock.dedup_vector_threshold = 0.85
    server._settings = settings_mock

    mocks = {
        "memory": memory_svc,
        "spatial": spatial_svc,
        "lifecycle": lifecycle_svc,
        "utility": utility_svc,
        "export_import": export_import_svc,
    }
    return server, mocks
