"""Tests for path_utils module."""

import sys
from pathlib import Path

import pytest

from spatial_memory.core.path_utils import (
    get_blocklisted_roots,
    is_blocklisted,
    normalize_path,
    reset_blocklist_cache,
)


@pytest.mark.unit
class TestNormalizePath:
    """Tests for normalize_path."""

    def test_string_to_path(self) -> None:
        """Test converting string to Path."""
        result = normalize_path("/some/path")
        assert isinstance(result, Path)

    def test_expands_user_home(self, tmp_path: Path) -> None:
        """Test that ~ is expanded."""
        result = normalize_path("~")
        assert result == Path.home().resolve()

    def test_resolves_path(self, tmp_path: Path) -> None:
        """Test that path is resolved."""
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        result = normalize_path(str(subdir))
        assert result.is_absolute()

    def test_path_input(self, tmp_path: Path) -> None:
        """Test with Path input."""
        result = normalize_path(tmp_path)
        assert isinstance(result, Path)
        assert result.is_absolute()


@pytest.mark.unit
class TestBlocklist:
    """Tests for blocklist functions."""

    def setup_method(self) -> None:
        """Reset cache before each test."""
        reset_blocklist_cache()

    def test_home_is_blocklisted(self) -> None:
        """Test that home directory is blocklisted."""
        assert is_blocklisted(Path.home())

    def test_root_is_blocklisted(self) -> None:
        """Test that root is blocklisted."""
        if sys.platform == "win32":
            assert is_blocklisted(Path("C:\\"))
        else:
            assert is_blocklisted(Path("/"))

    def test_project_dir_not_blocklisted(self, tmp_path: Path) -> None:
        """Test that a regular directory is not blocklisted."""
        project = tmp_path / "my-project"
        project.mkdir()
        assert not is_blocklisted(project)

    def test_get_blocklisted_roots_returns_set(self) -> None:
        """Test that get_blocklisted_roots returns a set."""
        roots = get_blocklisted_roots()
        assert isinstance(roots, set)
        assert len(roots) > 0

    def test_cache_works(self) -> None:
        """Test that the cache is reused."""
        roots1 = get_blocklisted_roots()
        roots2 = get_blocklisted_roots()
        assert roots1 is roots2
