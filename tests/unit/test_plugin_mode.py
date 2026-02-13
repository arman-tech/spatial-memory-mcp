"""Tests for spatial_memory.tools.plugin_mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from spatial_memory.tools.plugin_mode import (
    _DEV_SERVER,
    _PROD_SERVER,
    detect_mode,
    run_plugin_mode,
)


def _make_mcp(server_block: dict, env: dict | None = None) -> dict:
    """Build a minimal .mcp.json dict."""
    entry = {**server_block}
    if env is not None:
        entry["env"] = env
    return {"mcpServers": {"spatial-memory": entry}}


# ── detect_mode ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestDetectMode:
    def test_dev(self):
        assert detect_mode(_make_mcp(_DEV_SERVER)) == "dev"

    def test_prod(self):
        assert detect_mode(_make_mcp(_PROD_SERVER)) == "prod"

    def test_unknown(self):
        assert detect_mode(_make_mcp({"command": "node", "args": []})) == "unknown"

    def test_empty(self):
        assert detect_mode({}) == "unknown"


# ── run_plugin_mode ──────────────────────────────────────────────────


def _write_mcp(path: Path, server_block: dict, env: dict | None = None) -> None:
    data = _make_mcp(server_block, env)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _args(mode: str, mcp_path: Path) -> argparse.Namespace:
    ns = argparse.Namespace(target_mode=mode)
    ns._mcp_path = mcp_path  # noqa: SLF001
    return ns


@pytest.mark.unit
class TestRunPluginMode:
    def test_status(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        mcp = tmp_path / ".mcp.json"
        _write_mcp(mcp, _PROD_SERVER)
        rc = run_plugin_mode(_args("status", mcp))
        assert rc == 0
        assert "prod" in capsys.readouterr().out

    def test_prod_to_dev(self, tmp_path: Path):
        mcp = tmp_path / ".mcp.json"
        env = {"SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"}
        _write_mcp(mcp, _PROD_SERVER, env=env)

        rc = run_plugin_mode(_args("dev", mcp))
        assert rc == 0

        data = json.loads(mcp.read_text(encoding="utf-8"))
        server = data["mcpServers"]["spatial-memory"]
        assert server["command"] == "python"
        assert server["args"] == ["-m", "spatial_memory"]
        # env preserved
        assert server["env"] == env

    def test_dev_to_prod(self, tmp_path: Path):
        mcp = tmp_path / ".mcp.json"
        env = {"MY_KEY": "val"}
        _write_mcp(mcp, _DEV_SERVER, env=env)

        rc = run_plugin_mode(_args("prod", mcp))
        assert rc == 0

        data = json.loads(mcp.read_text(encoding="utf-8"))
        server = data["mcpServers"]["spatial-memory"]
        assert server["command"] == "uvx"
        assert "--from" in server["args"]
        assert server["env"] == env

    def test_already_in_mode(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        mcp = tmp_path / ".mcp.json"
        _write_mcp(mcp, _DEV_SERVER)

        rc = run_plugin_mode(_args("dev", mcp))
        assert rc == 0
        assert "Already in dev mode" in capsys.readouterr().out

    def test_missing_file(self, tmp_path: Path):
        mcp = tmp_path / "does_not_exist.json"
        rc = run_plugin_mode(_args("dev", mcp))
        assert rc == 1
