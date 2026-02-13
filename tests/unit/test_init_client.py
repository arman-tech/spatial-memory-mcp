"""Unit tests for spatial_memory.tools.init_client.

Tests cover:
1. Cursor init — 3 files created, content valid
2. Merge preserves existing config
3. Already configured — exit 0
4. Force overwrites
5. Error cases — invalid JSON without --force
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_memory.tools.init_client import (
    _get_cursor_paths,
    _merge_json,
    _validate_project_name,
    _write_rules_file,
    run_init,
)

# =============================================================================
# Path resolution
# =============================================================================


@pytest.mark.unit
class TestGetCursorPaths:
    """Test Cursor config path resolution."""

    def test_project_scope(self) -> None:
        paths = _get_cursor_paths(global_scope=False)
        assert paths["mcp_json"] == Path(".cursor/mcp.json")
        assert paths["hooks_json"] == Path(".cursor/hooks.json")
        assert paths["rules_file"] == Path(".cursor/rules/spatial-memory.mdc")

    def test_global_scope(self) -> None:
        paths = _get_cursor_paths(global_scope=True)
        assert paths["mcp_json"].parent == Path.home() / ".cursor"
        assert paths["hooks_json"].parent == Path.home() / ".cursor"


# =============================================================================
# JSON merge
# =============================================================================


@pytest.mark.unit
class TestMergeJson:
    """Test JSON merge utility."""

    def test_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        new_data = {"mcpServers": {"spatial-memory": {"command": "test"}}}
        result = _merge_json(path, new_data, "mcpServers")
        assert result == new_data

    def test_merge_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"mcpServers": {"other": {"command": "other"}}}),
            encoding="utf-8",
        )
        new_data = {"mcpServers": {"spatial-memory": {"command": "test"}}}
        result = _merge_json(path, new_data, "mcpServers")
        assert "other" in result["mcpServers"]
        assert "spatial-memory" in result["mcpServers"]

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("not valid json{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _merge_json(path, {}, "key")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("", encoding="utf-8")
        new_data = {"key": "value"}
        result = _merge_json(path, new_data, "key")
        assert result == new_data


# =============================================================================
# Rules file
# =============================================================================


@pytest.mark.unit
class TestWriteRulesFile:
    """Test rules .mdc file writing."""

    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "rules" / "spatial-memory.mdc"
        _write_rules_file(path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "recall" in content.lower()
        assert "alwaysApply: true" in content


# =============================================================================
# Project name validation
# =============================================================================


@pytest.mark.unit
class TestValidateProjectName:
    """Test _validate_project_name guard."""

    def test_valid_simple_name(self) -> None:
        assert _validate_project_name("tic-tac-toe") == "tic-tac-toe"

    def test_valid_with_dots(self) -> None:
        assert _validate_project_name("my.project.v2") == "my.project.v2"

    def test_valid_with_underscores(self) -> None:
        assert _validate_project_name("my_project") == "my_project"

    def test_valid_single_char(self) -> None:
        assert _validate_project_name("a") == "a"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not determine project name"):
            _validate_project_name("")

    def test_starts_with_hyphen_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid project name"):
            _validate_project_name("-bad-name")

    def test_starts_with_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid project name"):
            _validate_project_name(".hidden")

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid project name"):
            _validate_project_name("my project!")

    def test_spaces_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid project name"):
            _validate_project_name("my project")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid project name"):
            _validate_project_name("a" * 101)

    def test_max_length_ok(self) -> None:
        name = "a" * 100
        assert _validate_project_name(name) == name


# =============================================================================
# run_init
# =============================================================================


@pytest.mark.unit
class TestRunInit:
    """Test the init command end-to-end."""

    def _make_args(
        self,
        client: str = "cursor",
        global_scope: bool = False,
        force: bool = False,
        project: str | None = None,
        mode: str = "prod",
    ) -> object:
        class Args:
            pass

        args = Args()
        args.client = client  # type: ignore[attr-defined]
        args.global_scope = global_scope  # type: ignore[attr-defined]
        args.force = force  # type: ignore[attr-defined]
        args.project = project  # type: ignore[attr-defined]
        args.mode = mode  # type: ignore[attr-defined]
        return args

    def test_cursor_creates_3_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = run_init(self._make_args())  # type: ignore[arg-type]
        assert result == 0
        assert (tmp_path / ".cursor" / "mcp.json").exists()
        assert (tmp_path / ".cursor" / "hooks.json").exists()
        assert (tmp_path / ".cursor" / "rules" / "spatial-memory.mdc").exists()

    def test_already_configured_exits_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]
        result = run_init(self._make_args())  # type: ignore[arg-type]
        assert result == 0

    def test_force_overwrites(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]
        result = run_init(self._make_args(force=True))  # type: ignore[arg-type]
        assert result == 0

    def test_mcp_json_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        assert "mcpServers" in data
        assert "spatial-memory" in data["mcpServers"]

    def test_hooks_json_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
        assert "version" in data
        assert "hooks" in data

    def test_unsupported_client(self) -> None:
        result = run_init(self._make_args(client="windsurf"))  # type: ignore[arg-type]
        assert result == 1

    def test_explicit_project_in_mcp_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args(project="my-game"))  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        env = data["mcpServers"]["spatial-memory"]["env"]
        assert env["SPATIAL_MEMORY_PROJECT"] == "my-game"

    def test_cwd_fallback_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        env = data["mcpServers"]["spatial-memory"]["env"]
        assert env["SPATIAL_MEMORY_PROJECT"] == tmp_path.name

    def test_global_scope_no_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Global scope uses ~/.cursor/ so we mock _get_cursor_paths to use tmp_path
        monkeypatch.chdir(tmp_path)
        import spatial_memory.tools.init_client as mod

        monkeypatch.setattr(
            mod,
            "_get_cursor_paths",
            lambda global_scope: {
                "mcp_json": tmp_path / ".cursor" / "mcp.json",
                "hooks_json": tmp_path / ".cursor" / "hooks.json",
                "rules_file": tmp_path / ".cursor" / "rules" / "spatial-memory.mdc",
            },
        )
        run_init(self._make_args(global_scope=True))  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        env = data["mcpServers"]["spatial-memory"]["env"]
        assert "SPATIAL_MEMORY_PROJECT" not in env

    def test_invalid_project_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = run_init(self._make_args(project="-bad-name"))  # type: ignore[arg-type]
        assert result == 1
        assert not (tmp_path / ".cursor" / "mcp.json").exists()

    def test_prod_mode_uses_uvx(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args())  # type: ignore[arg-type]  # default is prod
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        server = data["mcpServers"]["spatial-memory"]
        assert server["command"] == "uvx"
        assert "--from" in server["args"]
        assert "spatial-memory-mcp" in server["args"]

    def test_dev_mode_uses_python(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        run_init(self._make_args(mode="dev"))  # type: ignore[arg-type]
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        server = data["mcpServers"]["spatial-memory"]
        assert server["command"] != "uvx"
        assert server["args"] == ["-m", "spatial_memory"]
