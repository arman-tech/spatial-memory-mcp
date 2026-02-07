"""Tests for project_detection module."""

import configparser
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from spatial_memory.adapters.project_detection import (
    ProjectDetectionConfig,
    ProjectDetector,
    ProjectIdentity,
)


@pytest.mark.unit
class TestProjectDetector:
    """Tests for ProjectDetector."""

    def test_explicit_project(self) -> None:
        """Test level 1: explicit project parameter."""
        detector = ProjectDetector()
        result = detector.detect(explicit_project="github.com/org/repo")
        assert result.project_id == "github.com/org/repo"
        assert result.source == "explicit"

    def test_explicit_wildcard(self) -> None:
        """Test level 1: wildcard passes through."""
        detector = ProjectDetector()
        result = detector.detect(explicit_project="*")
        assert result.project_id == "*"
        assert result.source == "explicit"

    def test_explicit_empty_falls_through(self) -> None:
        """Test that empty explicit project falls through."""
        detector = ProjectDetector()
        result = detector.detect(explicit_project="")
        # Falls to level 7 since nothing else matches
        assert result.source == "fallback"
        assert result.project_id == ""

    def test_file_path_detection(self, tmp_path: Path) -> None:
        """Test level 2: detect project from file path."""
        # Create git repo structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        # Write git config with remote
        config = configparser.ConfigParser()
        config['remote "origin"'] = {"url": "https://github.com/org/repo.git"}
        config_file = git_dir / "config"
        with open(config_file, "w") as f:
            config.write(f)

        detector = ProjectDetector()
        subdir = tmp_path / "src"
        subdir.mkdir()
        result = detector.detect(file_path=subdir)

        assert result.project_id == "github.com/org/repo"
        assert result.source == "file_path"
        assert result.git_root == tmp_path

    def test_env_var_detection(self, tmp_path: Path) -> None:
        """Test level 3: detect project from environment variable."""
        # Create git repo
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        config = configparser.ConfigParser()
        config['remote "origin"'] = {"url": "https://github.com/env/project.git"}
        with open(git_dir / "config", "w") as f:
            config.write(f)

        config_obj = ProjectDetectionConfig(env_var_names=["TEST_PROJECT_DIR"])
        detector = ProjectDetector(config=config_obj)

        with patch.dict(os.environ, {"TEST_PROJECT_DIR": str(tmp_path)}):
            result = detector.detect()

        assert result.project_id == "github.com/env/project"
        assert result.source == "env_var"

    def test_config_setting(self) -> None:
        """Test level 5: use config project setting."""
        config = ProjectDetectionConfig(
            explicit_project="custom-project-id",
            env_var_names=[],  # Disable env var check
        )
        detector = ProjectDetector(config=config)
        result = detector.detect()
        assert result.project_id == "custom-project-id"
        assert result.source == "config"

    def test_single_project_heuristic(self) -> None:
        """Test level 6: single-project heuristic."""
        counter = lambda: {"github.com/org/repo": 42}  # noqa: E731
        config = ProjectDetectionConfig(env_var_names=[])
        detector = ProjectDetector(config=config, project_counter=counter)
        result = detector.detect()
        assert result.project_id == "github.com/org/repo"
        assert result.source == "single_project"

    def test_single_project_ignores_empty(self) -> None:
        """Test that single-project heuristic ignores empty project IDs."""
        counter = lambda: {"": 100, "github.com/org/repo": 5}  # noqa: E731
        config = ProjectDetectionConfig(env_var_names=[])
        detector = ProjectDetector(config=config, project_counter=counter)
        result = detector.detect()
        assert result.project_id == "github.com/org/repo"
        assert result.source == "single_project"

    def test_multiple_projects_falls_through(self) -> None:
        """Test that multiple projects don't match level 6."""
        counter = lambda: {"github.com/org/a": 10, "github.com/org/b": 5}  # noqa: E731
        config = ProjectDetectionConfig(env_var_names=[])
        detector = ProjectDetector(config=config, project_counter=counter)
        result = detector.detect()
        assert result.source == "fallback"

    def test_fallback(self) -> None:
        """Test level 7: fallback returns empty project."""
        config = ProjectDetectionConfig(env_var_names=[])
        detector = ProjectDetector(config=config)
        result = detector.detect()
        assert result.project_id == ""
        assert result.source == "fallback"

    def test_cascade_order(self, tmp_path: Path) -> None:
        """Test that cascade stops at first match."""
        # Explicit overrides everything
        counter = lambda: {"github.com/org/repo": 42}  # noqa: E731
        config = ProjectDetectionConfig(
            explicit_project="config-project",
            env_var_names=[],
        )
        detector = ProjectDetector(config=config, project_counter=counter)
        result = detector.detect(explicit_project="override")
        assert result.project_id == "override"
        assert result.source == "explicit"

    def test_no_remote_uses_dir_name(self, tmp_path: Path) -> None:
        """Test that a git repo without remotes uses directory name."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        # Write empty git config (no remotes)
        config = configparser.ConfigParser()
        with open(git_dir / "config", "w") as f:
            config.write(f)

        detector = ProjectDetector()
        result = detector.detect(file_path=tmp_path)
        assert result.project_id == tmp_path.name
        assert result.source == "file_path"


@pytest.mark.unit
class TestProjectIdentity:
    """Tests for ProjectIdentity dataclass."""

    def test_basic_identity(self) -> None:
        """Test creating a basic identity."""
        identity = ProjectIdentity(project_id="github.com/org/repo", source="explicit")
        assert identity.project_id == "github.com/org/repo"
        assert identity.source == "explicit"
        assert identity.git_root is None
        assert identity.remote_url is None
        assert identity.sub_project is None

    def test_full_identity(self, tmp_path: Path) -> None:
        """Test creating a full identity with all fields."""
        identity = ProjectIdentity(
            project_id="github.com/org/repo",
            source="file_path",
            git_root=tmp_path,
            remote_url="https://github.com/org/repo.git",
            sub_project="packages/api",
        )
        assert identity.git_root == tmp_path
        assert identity.remote_url == "https://github.com/org/repo.git"
        assert identity.sub_project == "packages/api"

    def test_frozen(self) -> None:
        """Test that identity is immutable."""
        identity = ProjectIdentity(project_id="test", source="explicit")
        with pytest.raises(AttributeError):
            identity.project_id = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestResolveFromDirectoryOSError:
    """Tests for H9: path.resolve() OSError handling."""

    def test_resolve_oserror_falls_back_to_absolute(self) -> None:
        """When path.resolve() raises OSError, should fall back to absolute()."""
        from unittest.mock import MagicMock

        detector = ProjectDetector(ProjectDetectionConfig())

        # Create a mock path where resolve() raises OSError
        mock_path = MagicMock(spec=Path)
        mock_path.resolve.side_effect = OSError("broken symlink")
        mock_path.absolute.return_value = Path("/fallback/path")
        mock_path.__str__ = lambda self: "/fallback/path"

        # Mock find_git_root to return None (simplifies test)
        with patch("spatial_memory.adapters.project_detection.find_git_root", return_value=None):
            result = detector._resolve_from_directory(mock_path, source="test")

        # Should not crash, should return None (no git root)
        assert result is None
