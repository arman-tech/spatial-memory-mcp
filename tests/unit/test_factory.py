"""Unit tests for ServiceFactory.

Tests verify that factory wiring and initialization calls happen correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from spatial_memory.factory import ServiceFactory

# ===========================================================================
# TestSeedFromRepository
# ===========================================================================


@pytest.mark.unit
class TestSeedFromRepository:
    """Verify that create_all() seeds the ingest pipeline from the repository."""

    def test_seed_from_repository_called_during_create_all(self) -> None:
        """create_all() should call ingest_pipeline.seed_from_repository(repository)."""
        # Build mock embeddings with a .dimensions property
        mock_embeddings = MagicMock()
        type(mock_embeddings).dimensions = PropertyMock(return_value=384)
        type(mock_embeddings).backend = PropertyMock(return_value="mock")

        # Build mock repository
        mock_repository = MagicMock()

        # Create a minimal settings mock
        mock_settings = MagicMock()
        mock_settings.cognitive_offloading_enabled = False
        mock_settings.auto_decay_enabled = False
        mock_settings.rate_limit_per_agent_enabled = False
        mock_settings.embedding_rate_limit = 10.0
        mock_settings.response_cache_enabled = False
        mock_settings.project = ""

        factory = ServiceFactory(
            settings=mock_settings,
            repository=mock_repository,
            embeddings=mock_embeddings,
        )

        # Patch create_ingest_pipeline to return a mock whose seed_from_repository
        # we can inspect.
        mock_pipeline = MagicMock()
        with patch.object(factory, "create_ingest_pipeline", return_value=mock_pipeline):
            factory.create_all()

        # Assert seed_from_repository was called (with the repository)
        mock_pipeline.seed_from_repository.assert_called_once_with(mock_repository)
