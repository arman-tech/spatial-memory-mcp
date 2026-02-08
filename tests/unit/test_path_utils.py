"""Tests for path_utils module."""

import sys
from pathlib import Path

import pytest

from spatial_memory.adapters.path_utils import (
    get_blocklisted_roots,
    is_blocklisted,
    normalize_path,
    normalize_user_path,
    reset_blocklist_cache,
)


@pytest.mark.unit
class TestNormalizePath:
    """Tests for normalize_path (safe for untrusted input)."""

    def test_string_to_path(self) -> None:
        """Test converting string to Path."""
        result = normalize_path("/some/path")
        assert isinstance(result, Path)

    def test_does_not_expand_user_home(self) -> None:
        """Test that ~ is NOT expanded (safe for untrusted input)."""
        result = normalize_path("~")
        # The resolved path should NOT equal the home directory
        assert result != Path.home().resolve()

    def test_does_not_expand_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables are NOT expanded."""
        monkeypatch.setenv("SPATIAL_TEST_SECRET", "leaked_value")
        if sys.platform == "win32":
            result = normalize_path("%SPATIAL_TEST_SECRET%/subdir")
            result_str = str(result)
            assert "leaked_value" not in result_str
        else:
            result = normalize_path("$SPATIAL_TEST_SECRET/subdir")
            result_str = str(result)
            assert "leaked_value" not in result_str

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
class TestNormalizeUserPath:
    """Tests for normalize_user_path (trusted input only)."""

    def test_expands_user_home(self) -> None:
        """Test that ~ IS expanded for trusted paths."""
        result = normalize_user_path("~")
        assert result == Path.home().resolve()

    def test_expands_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables ARE expanded for trusted paths."""
        monkeypatch.setenv("SPATIAL_TEST_DIR", "/some/test/dir")
        if sys.platform == "win32":
            result = normalize_user_path("%SPATIAL_TEST_DIR%")
        else:
            result = normalize_user_path("$SPATIAL_TEST_DIR")
        result_str = str(result)
        assert "SPATIAL_TEST_DIR" not in result_str

    def test_string_to_path(self) -> None:
        """Test converting string to Path."""
        result = normalize_user_path("/some/path")
        assert isinstance(result, Path)

    def test_resolves_path(self, tmp_path: Path) -> None:
        """Test that path is resolved."""
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        result = normalize_user_path(str(subdir))
        assert result.is_absolute()

    def test_path_input(self, tmp_path: Path) -> None:
        """Test with Path input."""
        result = normalize_user_path(tmp_path)
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


@pytest.mark.unit
class TestBlocklistThreadSafety:
    """Tests for H10: thread-safe blocklist initialization."""

    def setup_method(self) -> None:
        """Reset cache before each test."""
        reset_blocklist_cache()

    def test_concurrent_initialization(self) -> None:
        """Multiple threads initializing blocklist should not crash."""
        import threading

        results: list[set[Path]] = []
        errors: list[Exception] = []

        def get_roots() -> None:
            try:
                roots = get_blocklisted_roots()
                results.append(roots)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_roots) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent initialization: {errors}"
        assert len(results) == 10
        # All threads should get the same set object (cached)
        for r in results:
            assert r is results[0]
