"""Unit tests for scripts/sync_plugin_hooks.py.

Tests cover:
1. get_hook_files() — returns correct files, excludes __init__.py
2. _check_stdlib_only() — detects third-party imports
3. sync(check_only=True) — detects missing/out-of-sync files
4. sync(check_only=False) — copies files
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import the sync module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sync_plugin_hooks import (  # noqa: E402
    _check_stdlib_only,
    get_hook_files,
    sync,
)

# =============================================================================
# get_hook_files
# =============================================================================


@pytest.mark.unit
class TestGetHookFiles:
    """Test hook file discovery."""

    def test_returns_py_files(self) -> None:
        files = get_hook_files()
        assert len(files) > 0
        for f in files:
            assert f.suffix == ".py"

    def test_excludes_init(self) -> None:
        files = get_hook_files()
        names = [f.name for f in files]
        assert "__init__.py" not in names

    def test_includes_session_start(self) -> None:
        files = get_hook_files()
        names = [f.name for f in files]
        assert "session_start.py" in names

    def test_includes_entrypoints(self) -> None:
        files = get_hook_files()
        names = [f.name for f in files]
        assert "post_tool_use.py" in names
        assert "pre_compact.py" in names
        assert "stop.py" in names


# =============================================================================
# _check_stdlib_only
# =============================================================================


@pytest.mark.unit
class TestStdlibCheck:
    """Test stdlib-only import validation."""

    def test_clean_file(self, tmp_path: Path) -> None:
        p = tmp_path / "clean.py"
        p.write_text("import json\nimport os\n", encoding="utf-8")
        assert _check_stdlib_only(p) == []

    def test_third_party_import(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.py"
        p.write_text("import numpy\n", encoding="utf-8")
        violations = _check_stdlib_only(p)
        assert len(violations) == 1
        assert "numpy" in violations[0]

    def test_third_party_from_import(self, tmp_path: Path) -> None:
        p = tmp_path / "bad2.py"
        p.write_text("from lancedb import connect\n", encoding="utf-8")
        violations = _check_stdlib_only(p)
        assert len(violations) == 1
        assert "lancedb" in violations[0]

    def test_syntax_error(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.py"
        p.write_text("def (\n", encoding="utf-8")
        violations = _check_stdlib_only(p)
        assert len(violations) == 1
        assert "syntax error" in violations[0]

    def test_all_hook_files_are_stdlib_only(self) -> None:
        """Verify all hook source files pass the stdlib check."""
        for path in get_hook_files():
            violations = _check_stdlib_only(path)
            assert violations == [], f"{path.name} has violations: {violations}"


# =============================================================================
# sync
# =============================================================================


@pytest.mark.unit
class TestSync:
    """Test sync check mode against the actual repo state."""

    def test_check_passes_after_sync(self) -> None:
        """After running sync, check mode should pass."""
        result = sync(check_only=True)
        assert result == 0

    def test_check_detects_missing_file(self, tmp_path: Path) -> None:
        """If a file is missing from dest, check fails."""
        # Create a fake source dir with one file
        src = tmp_path / "src"
        src.mkdir()
        (src / "test_module.py").write_text("import json\n", encoding="utf-8")

        dst = tmp_path / "dst"
        dst.mkdir()
        # Don't create the file in dst

        with (
            patch("sync_plugin_hooks.SRC_DIR", src),
            patch("sync_plugin_hooks.DST_DIR", dst),
        ):
            result = sync(check_only=True)
            assert result == 1

    def test_copy_creates_files(self, tmp_path: Path) -> None:
        """Sync without check_only should copy files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "test_module.py").write_text("import json\n", encoding="utf-8")

        dst = tmp_path / "dst"
        # Don't create dst yet

        with (
            patch("sync_plugin_hooks.SRC_DIR", src),
            patch("sync_plugin_hooks.DST_DIR", dst),
        ):
            result = sync(check_only=False)
            assert result == 0
            assert (dst / "test_module.py").exists()
            assert (dst / "test_module.py").read_text(encoding="utf-8") == "import json\n"
