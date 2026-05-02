"""Unit tests for ServiceFactory.

Tests verify that factory wiring and initialization calls happen correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from spatial_memory.factory import ServiceFactory

# ===========================================================================
# TestSeedFromRepository
# ===========================================================================


@pytest.mark.unit
class TestSeedFromRepository:
    """Verify that create_all() seeds the ingest pipeline from the repository."""

    def test_seed_from_repository_called_during_create_all(self, tmp_path: Path) -> None:
        """create_all() should call ingest_pipeline.seed_from_repository(repository)."""
        # Build mock embeddings with a .dimensions property
        mock_embeddings = MagicMock()
        type(mock_embeddings).dimensions = PropertyMock(return_value=384)
        type(mock_embeddings).backend = PropertyMock(return_value="mock")

        # Build mock repository
        mock_repository = MagicMock()

        # Create a minimal settings mock. memory_path must be a real Path —
        # ServiceFactory.__init__ unconditionally calls set_journal_dir(
        # settings.memory_path / "metrics"), and a bare MagicMock there leaks
        # a stringified <MagicMock>/__truediv__() directory under repo root
        # when journal_observation later materializes the path.
        mock_settings = MagicMock()
        mock_settings.memory_path = tmp_path
        mock_settings.embedding_model = "arctic-m-long"
        mock_settings.cognitive_offloading_enabled = False
        mock_settings.auto_decay_enabled = False
        mock_settings.rate_limit_per_agent_enabled = False
        mock_settings.embedding_rate_limit = 10.0
        mock_settings.response_cache_enabled = False
        mock_settings.project = ""
        mock_settings.cross_corpus_min_similarity = 0.5
        mock_settings.cross_corpus_max_batch = 100
        mock_settings.cross_corpus_scoring = "vector_only"
        mock_settings.cross_corpus_content_weight = 0.3

        # Patch the journal-wiring side effects too — this is a pure unit test
        # and shouldn't touch the metrics-journal module's process state.
        with patch("spatial_memory.factory.set_journal_dir"), patch(
            "spatial_memory.factory.prune_journal"
        ):
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


# ===========================================================================
# US-022 / spec §7.2 — factory wires set_journal_dir + prune_journal
# ===========================================================================


@pytest.mark.unit
class TestJournalWiring:
    """ServiceFactory.__init__ must pin journal dir to settings.memory_path / 'metrics'.

    This closes the chicken-and-egg gap: the journal directory must follow
    Settings.memory_path (not the SPATIAL_MEMORY_MEMORY_PATH env var) for the
    duration of factory and downstream service construction.
    """

    def _make_settings(self, memory_path: Path) -> MagicMock:
        s = MagicMock()
        s.memory_path = memory_path
        s.embedding_model = "arctic-m-long"
        return s

    def test_factory_init_calls_set_journal_dir_with_memory_path(self, tmp_path: Path) -> None:
        """AC1: factory.__init__ calls set_journal_dir(settings.memory_path / 'metrics')."""
        captured: list[Path] = []
        with patch(
            "spatial_memory.factory.set_journal_dir",
            side_effect=lambda p: captured.append(p),
        ):
            with patch("spatial_memory.factory.prune_journal"):
                ServiceFactory(settings=self._make_settings(tmp_path))

        assert len(captured) == 1
        assert captured[0] == tmp_path / "metrics"

    def test_factory_init_calls_prune_journal_at_startup(self, tmp_path: Path) -> None:
        """AC2: factory.__init__ calls prune_journal() once."""
        with patch("spatial_memory.factory.set_journal_dir"):
            with patch("spatial_memory.factory.prune_journal") as mock_prune:
                ServiceFactory(settings=self._make_settings(tmp_path))

        assert mock_prune.call_count == 1

    def test_factory_init_set_journal_dir_overrides_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC4 (negative control): Settings.memory_path beats SPATIAL_MEMORY_MEMORY_PATH.

        When the env var points elsewhere, the factory must still pin the journal
        directory to settings.memory_path / 'metrics'. This pins the §7.2 contract
        that Settings is authoritative.
        """
        env_dir = tmp_path / "from-env"
        explicit_dir = tmp_path / "from-settings"
        monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(env_dir))

        captured: list[Path] = []
        with patch(
            "spatial_memory.factory.set_journal_dir",
            side_effect=lambda p: captured.append(p),
        ):
            with patch("spatial_memory.factory.prune_journal"):
                ServiceFactory(settings=self._make_settings(explicit_dir))

        assert len(captured) == 1
        assert captured[0] == explicit_dir / "metrics"
        assert captured[0] != env_dir / "metrics"
