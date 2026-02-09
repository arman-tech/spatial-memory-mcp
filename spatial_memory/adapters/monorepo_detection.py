"""Monorepo and sub-project detection.

Detects workspace roots and sub-project boundaries within monorepos
using marker files for common build systems.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Directories to skip when walking up (performance optimization)
SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        ".tox",
        ".nox",
        ".eggs",
        ".git",
        ".hg",
        ".svn",
    }
)


def _check_npm_workspace(root: Path) -> bool:
    """Check if root is an npm/yarn workspace."""
    pkg_json = root / "package.json"
    if not pkg_json.exists():
        return False
    try:
        data: dict[str, Any] = json.loads(pkg_json.read_text(encoding="utf-8"))
        return "workspaces" in data
    except (json.JSONDecodeError, OSError):
        return False


def _check_pnpm_workspace(root: Path) -> bool:
    """Check if root has a pnpm workspace."""
    return (root / "pnpm-workspace.yaml").exists()


def _check_cargo_workspace(root: Path) -> bool:
    """Check if root is a Cargo workspace."""
    cargo_toml = root / "Cargo.toml"
    if not cargo_toml.exists():
        return False
    try:
        content = cargo_toml.read_text(encoding="utf-8")
        return "[workspace]" in content
    except OSError:
        return False


def _check_go_workspace(root: Path) -> bool:
    """Check if root is a Go workspace."""
    return (root / "go.work").exists()


def _check_python_workspace(root: Path) -> bool:
    """Check if root is a Python/uv workspace."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        content = pyproject.read_text(encoding="utf-8")
        # uv workspaces use [tool.uv.workspace]
        return "[tool.uv.workspace]" in content
    except OSError:
        return False


def _check_gradle_multi(root: Path) -> bool:
    """Check if root is a Gradle multi-project build."""
    return (root / "settings.gradle").exists() or (root / "settings.gradle.kts").exists()


def _check_maven_multi(root: Path) -> bool:
    """Check if root is a Maven multi-module project."""
    pom = root / "pom.xml"
    if not pom.exists():
        return False
    try:
        content = pom.read_text(encoding="utf-8")
        return "<modules>" in content
    except OSError:
        return False


def _check_bazel_workspace(root: Path) -> bool:
    """Check if root is a Bazel workspace."""
    return (root / "WORKSPACE").exists() or (root / "WORKSPACE.bazel").exists()


def _check_nx_workspace(root: Path) -> bool:
    """Check if root is an Nx workspace."""
    return (root / "nx.json").exists()


def _check_lerna_workspace(root: Path) -> bool:
    """Check if root is a Lerna workspace."""
    return (root / "lerna.json").exists()


# Workspace root markers: filename -> detector function
WORKSPACE_MARKERS: dict[str, Any] = {
    "package.json": _check_npm_workspace,
    "pnpm-workspace.yaml": _check_pnpm_workspace,
    "Cargo.toml": _check_cargo_workspace,
    "go.work": _check_go_workspace,
    "pyproject.toml": _check_python_workspace,
    "settings.gradle": _check_gradle_multi,
    "settings.gradle.kts": _check_gradle_multi,
    "pom.xml": _check_maven_multi,
    "WORKSPACE": _check_bazel_workspace,
    "WORKSPACE.bazel": _check_bazel_workspace,
    "nx.json": _check_nx_workspace,
    "lerna.json": _check_lerna_workspace,
}

# Sub-project marker files (presence indicates a sub-project boundary)
SUB_PROJECT_MARKERS: frozenset[str] = frozenset(
    {
        "package.json",
        "Cargo.toml",
        "go.mod",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "build.gradle",
        "build.gradle.kts",
        "pom.xml",
        "BUILD",
        "BUILD.bazel",
        "CMakeLists.txt",
        "Makefile",
        "project.clj",
        "mix.exs",
        "Gemfile",
    }
)


def is_workspace_root(directory: Path) -> bool:
    """Check if a directory is a monorepo workspace root.

    Args:
        directory: Directory to check.

    Returns:
        True if the directory is a workspace root.
    """
    for marker_file, checker in WORKSPACE_MARKERS.items():
        if (directory / marker_file).exists():
            if checker(directory):
                return True
    return False


def detect_sub_project(file_path: Path, git_root: Path) -> str | None:
    """Detect the sub-project a file belongs to within a monorepo.

    Walks from the file's directory toward the git root, looking for
    sub-project marker files. Returns the relative path from git_root
    to the sub-project boundary.

    Args:
        file_path: Path to the file (or directory within the project).
        git_root: Path to the git repository root.

    Returns:
        Relative sub-project path (e.g., "packages/api"), or None
        if the file is at the workspace root level.
    """
    if file_path.is_file():
        current = file_path.parent
    else:
        current = file_path

    try:
        git_root = git_root.resolve()
        current = current.resolve()
    except OSError:
        return None

    # Don't detect sub-project at the git root itself
    if current == git_root:
        return None

    # Walk up from file toward git_root
    while current != git_root:
        # Check if any part of the path is in SKIP_DIRS
        if current.name in SKIP_DIRS or current.name.endswith(".egg-info"):
            current = current.parent
            continue

        # Check for sub-project markers
        for marker in SUB_PROJECT_MARKERS:
            if (current / marker).exists():
                try:
                    rel = current.relative_to(git_root)
                    rel_str = str(rel).replace("\\", "/")
                    if rel_str and rel_str != ".":
                        return rel_str
                except ValueError:
                    pass
                return None

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None
