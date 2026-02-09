"""Tests for monorepo_detection module."""

import json
from pathlib import Path

import pytest

from spatial_memory.adapters.monorepo_detection import (
    detect_sub_project,
    is_workspace_root,
)


@pytest.mark.unit
class TestIsWorkspaceRoot:
    """Tests for is_workspace_root."""

    def test_npm_workspace(self, tmp_path: Path) -> None:
        """Test detecting npm workspace."""
        pkg = {"name": "root", "workspaces": ["packages/*"]}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert is_workspace_root(tmp_path) is True

    def test_npm_non_workspace(self, tmp_path: Path) -> None:
        """Test that a regular npm project is not a workspace."""
        pkg = {"name": "simple-package"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert is_workspace_root(tmp_path) is False

    def test_pnpm_workspace(self, tmp_path: Path) -> None:
        """Test detecting pnpm workspace."""
        (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - packages/*\n")
        assert is_workspace_root(tmp_path) is True

    def test_cargo_workspace(self, tmp_path: Path) -> None:
        """Test detecting Cargo workspace."""
        (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/*"]\n')
        assert is_workspace_root(tmp_path) is True

    def test_cargo_non_workspace(self, tmp_path: Path) -> None:
        """Test that a regular Cargo project is not a workspace."""
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\n')
        assert is_workspace_root(tmp_path) is False

    def test_go_workspace(self, tmp_path: Path) -> None:
        """Test detecting Go workspace."""
        (tmp_path / "go.work").write_text("go 1.21\nuse ./cmd\n")
        assert is_workspace_root(tmp_path) is True

    def test_python_uv_workspace(self, tmp_path: Path) -> None:
        """Test detecting Python/uv workspace."""
        (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
        assert is_workspace_root(tmp_path) is True

    def test_python_non_workspace(self, tmp_path: Path) -> None:
        """Test that a regular Python project is not a workspace."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "mylib"\n')
        assert is_workspace_root(tmp_path) is False

    def test_nx_workspace(self, tmp_path: Path) -> None:
        """Test detecting Nx workspace."""
        (tmp_path / "nx.json").write_text("{}")
        assert is_workspace_root(tmp_path) is True

    def test_no_workspace(self, tmp_path: Path) -> None:
        """Test empty directory is not a workspace."""
        assert is_workspace_root(tmp_path) is False


@pytest.mark.unit
class TestDetectSubProject:
    """Tests for detect_sub_project."""

    def test_sub_project_with_package_json(self, tmp_path: Path) -> None:
        """Test detecting sub-project with package.json."""
        # Setup monorepo structure
        git_root = tmp_path
        (git_root / ".git").mkdir()
        pkg = tmp_path / "packages" / "api"
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text('{"name": "api"}')
        test_file = pkg / "src" / "index.ts"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        result = detect_sub_project(test_file, git_root)
        assert result == "packages/api"

    def test_sub_project_at_root(self, tmp_path: Path) -> None:
        """Test that root level returns None."""
        git_root = tmp_path
        (git_root / ".git").mkdir()
        (git_root / "package.json").write_text('{"name": "root"}')
        test_file = git_root / "src" / "index.ts"
        test_file.parent.mkdir()
        test_file.touch()

        result = detect_sub_project(test_file, git_root)
        assert result is None

    def test_no_sub_project(self, tmp_path: Path) -> None:
        """Test when no sub-project markers found."""
        git_root = tmp_path
        (git_root / ".git").mkdir()
        test_file = git_root / "src" / "deep" / "file.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        result = detect_sub_project(test_file, git_root)
        assert result is None

    def test_at_git_root(self, tmp_path: Path) -> None:
        """Test with file path at git root."""
        git_root = tmp_path
        (git_root / ".git").mkdir()

        result = detect_sub_project(git_root, git_root)
        assert result is None

    def test_cargo_sub_project(self, tmp_path: Path) -> None:
        """Test detecting Cargo sub-project."""
        git_root = tmp_path
        (git_root / ".git").mkdir()
        crate = tmp_path / "crates" / "core"
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text('[package]\nname = "core"\n')
        test_file = crate / "src" / "lib.rs"
        test_file.parent.mkdir()
        test_file.touch()

        result = detect_sub_project(test_file, git_root)
        assert result == "crates/core"
