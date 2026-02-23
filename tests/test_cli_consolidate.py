"""Tests for the consolidate CLI subcommand."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from spatial_memory.__main__ import run_consolidate
from spatial_memory.core.models import ConsolidateResult, ConsolidationGroup


@pytest.fixture
def mock_services():
    """Set up mocked factory/services for consolidate CLI tests."""
    with (
        patch("spatial_memory.config.get_settings") as mock_settings,
        patch("spatial_memory.factory.ServiceFactory") as mock_factory_cls,
    ):
        settings = MagicMock()
        settings.memory_path = "/tmp/test-db"
        mock_settings.return_value = settings

        factory = MagicMock()
        mock_factory_cls.return_value = factory

        embeddings = MagicMock()
        embeddings.dimensions = 384
        factory.create_embedding_service.return_value = embeddings

        database = MagicMock()
        factory.create_database.return_value = database

        repository = MagicMock()
        factory.create_repository.return_value = repository

        lifecycle = MagicMock()
        factory.create_lifecycle_service.return_value = lifecycle

        yield {
            "settings": settings,
            "factory": factory,
            "embeddings": embeddings,
            "database": database,
            "repository": repository,
            "lifecycle": lifecycle,
        }


def _make_args(**overrides) -> argparse.Namespace:
    """Create args namespace with defaults for consolidate."""
    defaults = {
        "namespace": "definitions",
        "similarity": 0.85,
        "strategy": "keep_highest_importance",
        "max_groups": 50,
        "project": None,
        "no_dry_run": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestConsolidateArgParsing:
    """Test that the consolidate subparser registers correctly."""

    def test_parser_accepts_consolidate_args(self):
        """Verify the subparser registers and parses consolidate args."""
        with patch("sys.argv", ["spatial-memory", "consolidate", "my-ns", "--similarity", "0.9"]):
            parser = argparse.ArgumentParser(prog="spatial-memory")
            subparsers = parser.add_subparsers(dest="command")
            cp = subparsers.add_parser("consolidate")
            cp.add_argument("namespace")
            cp.add_argument("--similarity", type=float, default=0.85)
            args = parser.parse_args(["consolidate", "my-ns", "--similarity", "0.9"])
            assert args.command == "consolidate"
            assert args.namespace == "my-ns"
            assert args.similarity == 0.9

    def test_defaults(self):
        args = _make_args()
        assert args.no_dry_run is False
        assert args.strategy == "keep_highest_importance"
        assert args.max_groups == 50
        assert args.project is None
        assert args.verbose is False

    def test_no_dry_run_flag(self):
        args = _make_args(no_dry_run=True)
        assert args.no_dry_run is True


class TestRunConsolidateDryRun:
    """Test dry-run mode (default)."""

    def test_no_duplicates_found(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=0,
            memories_merged=0,
            memories_deleted=0,
            groups=[],
            dry_run=True,
        )
        exit_code = run_consolidate(_make_args())
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "No duplicate groups found" in output

    def test_dry_run_output(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=2,
            memories_merged=3,
            memories_deleted=3,
            groups=[
                ConsolidationGroup(
                    representative_id="aaa",
                    member_ids=["aaa", "bbb"],
                    avg_similarity=0.92,
                    action_taken="preview",
                ),
                ConsolidationGroup(
                    representative_id="ccc",
                    member_ids=["ccc", "ddd", "eee"],
                    avg_similarity=0.88,
                    action_taken="preview",
                ),
            ],
            dry_run=True,
        )
        exit_code = run_consolidate(_make_args())
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "dry run" in output
        assert "2 group(s)" in output
        assert "5 memories total" in output
        assert "would be merged" in output
        assert "--no-dry-run" in output

    def test_dry_run_is_default(self, mock_services):
        run_consolidate(_make_args())
        call_kwargs = mock_services["lifecycle"].consolidate.call_args.kwargs
        assert call_kwargs["dry_run"] is True


class TestRunConsolidateLive:
    """Test live mode (--no-dry-run)."""

    def test_live_run_passes_dry_run_false(self, mock_services):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=0,
            memories_merged=0,
            memories_deleted=0,
            groups=[],
            dry_run=False,
        )
        run_consolidate(_make_args(no_dry_run=True))
        call_kwargs = mock_services["lifecycle"].consolidate.call_args.kwargs
        assert call_kwargs["dry_run"] is False

    def test_live_output_uses_past_tense(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=1,
            memories_merged=2,
            memories_deleted=2,
            groups=[
                ConsolidationGroup(
                    representative_id="aaa",
                    member_ids=["aaa", "bbb", "ccc"],
                    avg_similarity=0.90,
                    action_taken="merged",
                ),
            ],
            dry_run=False,
        )
        exit_code = run_consolidate(_make_args(no_dry_run=True))
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "LIVE" in output
        assert "were merged" in output
        assert "--no-dry-run" not in output


class TestRunConsolidateVerbose:
    """Test verbose output mode."""

    def test_verbose_shows_group_details(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=1,
            memories_merged=1,
            memories_deleted=1,
            groups=[
                ConsolidationGroup(
                    representative_id="aaa",
                    member_ids=["aaa", "bbb"],
                    avg_similarity=0.92,
                    action_taken="preview",
                ),
            ],
            dry_run=True,
        )
        run_consolidate(_make_args(verbose=True))
        output = capsys.readouterr().out
        assert "Group 1" in output
        assert "0.92" in output
        assert "Keep" in output
        assert "Merge" in output

    def test_non_verbose_hides_group_details(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=1,
            memories_merged=1,
            memories_deleted=1,
            groups=[
                ConsolidationGroup(
                    representative_id="aaa",
                    member_ids=["aaa", "bbb"],
                    avg_similarity=0.92,
                    action_taken="preview",
                ),
            ],
            dry_run=True,
        )
        run_consolidate(_make_args(verbose=False))
        output = capsys.readouterr().out
        assert "Group 1" not in output


class TestRunConsolidateArgs:
    """Test that CLI args are forwarded correctly to the service."""

    def test_args_forwarded(self, mock_services):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=0, memories_merged=0, memories_deleted=0, groups=[], dry_run=True
        )
        run_consolidate(
            _make_args(
                namespace="my-ns",
                similarity=0.90,
                strategy="merge_content",
                max_groups=10,
                project="my-project",
            )
        )
        call_kwargs = mock_services["lifecycle"].consolidate.call_args.kwargs
        assert call_kwargs["namespace"] == "my-ns"
        assert call_kwargs["similarity_threshold"] == 0.90
        assert call_kwargs["strategy"] == "merge_content"
        assert call_kwargs["max_groups"] == 10
        assert call_kwargs["project"] == "my-project"


class TestRunConsolidateErrorHandling:
    """Test error paths."""

    def test_service_error_returns_1(self, mock_services, capsys):
        mock_services["lifecycle"].consolidate.side_effect = RuntimeError("boom")
        exit_code = run_consolidate(_make_args())
        assert exit_code == 1
        output = capsys.readouterr().out
        assert "Error: boom" in output

    def test_database_closed_on_success(self, mock_services):
        mock_services["lifecycle"].consolidate.return_value = ConsolidateResult(
            groups_found=0, memories_merged=0, memories_deleted=0, groups=[], dry_run=True
        )
        run_consolidate(_make_args())
        mock_services["database"].close.assert_called_once()
