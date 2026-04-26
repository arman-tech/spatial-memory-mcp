"""Unit tests for the setup pre-warm command and sentinel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_memory.tools.setup_runner import (
    SCRIPT_NAME,
    SENTINEL_FILENAME,
    SENTINEL_SCHEMA,
    _build_mcp_entry,
    _ensure_mcp_entry,
    _evict_local_path_uv_cache,
    read_sentinel,
    strict_startup_error,
    write_sentinel,
)


class TestSentinelReadWrite:
    def test_read_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert read_sentinel(tmp_path) is None

    def test_read_returns_none_when_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / SENTINEL_FILENAME).write_text("not json{", encoding="utf-8")
        assert read_sentinel(tmp_path) is None

    def test_write_then_read_round_trip(self, tmp_path: Path) -> None:
        write_sentinel(tmp_path, model="arctic-m-long", dim=768, version="2.0.0")
        sentinel = read_sentinel(tmp_path)
        assert sentinel is not None
        assert sentinel["embedding_model"] == "arctic-m-long"
        assert sentinel["embedding_dim"] == 768
        assert sentinel["spatial_memory_version"] == "2.0.0"
        assert sentinel["schema"] == SENTINEL_SCHEMA
        assert "initialized_at" in sentinel

    def test_write_creates_memory_path_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "spatial-memory"
        write_sentinel(target, model="arctic-m-long", dim=768, version="2.0.0")
        assert (target / SENTINEL_FILENAME).exists()

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        write_sentinel(tmp_path, model="arctic-m-long", dim=768, version="2.0.0")
        write_sentinel(tmp_path, model="bge-small", dim=384, version="2.0.1")
        sentinel = read_sentinel(tmp_path)
        assert sentinel is not None
        assert sentinel["embedding_model"] == "bge-small"
        assert sentinel["embedding_dim"] == 384


class TestStrictStartupError:
    def test_missing_sentinel_returns_actionable_error(self, tmp_path: Path) -> None:
        err = strict_startup_error(tmp_path, "arctic-m-long")
        assert err is not None
        assert "spatial-memory setup" in err
        assert "arctic-m-long" in err
        assert str(tmp_path) in err

    def test_matching_sentinel_returns_none(self, tmp_path: Path) -> None:
        write_sentinel(tmp_path, model="arctic-m-long", dim=768, version="2.0.0")
        assert strict_startup_error(tmp_path, "arctic-m-long") is None

    def test_model_mismatch_returns_re_run_message(self, tmp_path: Path) -> None:
        write_sentinel(tmp_path, model="bge-small", dim=384, version="2.0.0")
        err = strict_startup_error(tmp_path, "arctic-m-long")
        assert err is not None
        assert "does not match" in err
        assert "bge-small" in err
        assert "arctic-m-long" in err
        assert "--force" in err

    def test_corrupt_sentinel_treated_as_missing(self, tmp_path: Path) -> None:
        (tmp_path / SENTINEL_FILENAME).write_text("{not json", encoding="utf-8")
        err = strict_startup_error(tmp_path, "arctic-m-long")
        assert err is not None
        assert "not initialized" in err


class TestBuildMcpEntry:
    def test_entry_shape(self, tmp_path: Path) -> None:
        entry = _build_mcp_entry(
            source="spatial-memory-mcp",
            memory_path=tmp_path,
            model_alias="arctic-m-long",
        )
        assert entry["command"] == "uvx"
        assert entry["args"] == ["--from", "spatial-memory-mcp", SCRIPT_NAME]
        assert entry["env"]["SPATIAL_MEMORY_MEMORY_PATH"] == str(tmp_path)
        assert entry["env"]["SPATIAL_MEMORY_EMBEDDING_MODEL"] == "arctic-m-long"

    def test_entry_uses_local_path_source(self, tmp_path: Path) -> None:
        entry = _build_mcp_entry(
            source="/abs/path/to/repo",
            memory_path=tmp_path,
            model_alias="arctic-m-long",
        )
        assert entry["args"][1] == "/abs/path/to/repo"


class TestEnsureMcpEntry:
    def _desired(self, tmp_path: Path) -> dict:
        return _build_mcp_entry("spatial-memory-mcp", tmp_path, "arctic-m-long")

    def test_creates_when_missing(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        config, status = _ensure_mcp_entry(mcp_path, self._desired(tmp_path), force=False)
        assert status == "created"
        assert mcp_path.exists()
        on_disk = json.loads(mcp_path.read_text())
        assert on_disk["mcpServers"]["spatial-memory"]["command"] == "uvx"

    def test_adds_entry_when_other_servers_present(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text(json.dumps({"mcpServers": {"other": {"command": "echo"}}}))

        config, status = _ensure_mcp_entry(mcp_path, self._desired(tmp_path), force=False)
        assert status == "added entry"
        # Other entries are preserved
        assert config["mcpServers"]["other"]["command"] == "echo"
        assert "spatial-memory" in config["mcpServers"]

    def test_idempotent_when_matches(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        desired = self._desired(tmp_path)
        _ensure_mcp_entry(mcp_path, desired, force=False)

        config, status = _ensure_mcp_entry(mcp_path, desired, force=False)
        assert status == "already matches"

    def test_raises_on_mismatch_without_force(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        _ensure_mcp_entry(mcp_path, self._desired(tmp_path), force=False)

        new_desired = _build_mcp_entry("spatial-memory-mcp", tmp_path, "bge-small")
        with pytest.raises(RuntimeError, match="does not match"):
            _ensure_mcp_entry(mcp_path, new_desired, force=False)

    def test_force_overwrites_mismatched_entry(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        _ensure_mcp_entry(mcp_path, self._desired(tmp_path), force=False)

        new_desired = _build_mcp_entry("spatial-memory-mcp", tmp_path, "bge-small")
        config, status = _ensure_mcp_entry(mcp_path, new_desired, force=True)
        assert status == "overwritten (--force)"
        assert config["mcpServers"]["spatial-memory"]["env"][
            "SPATIAL_MEMORY_EMBEDDING_MODEL"
        ] == "bge-small"

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        mcp_path = tmp_path / ".mcp.json"
        mcp_path.write_text("not valid {")
        with pytest.raises(RuntimeError, match="could not parse"):
            _ensure_mcp_entry(mcp_path, self._desired(tmp_path), force=False)


class TestEvictLocalPathUvCache:
    """`_evict_local_path_uv_cache` removes uv's stale path-keyed wheel cache.

    The function is gated to local-path sources because uv's path-build cache
    is keyed by source path (not mtime), so editing source files does not
    invalidate it. PyPI sources are version-keyed and out of scope.
    """

    def _fake_uv_cache(
        self,
        cache_root: Path,
        *,
        with_wheel: bool = True,
        with_archive: bool = True,
        unrelated: bool = False,
    ) -> dict[str, Path]:
        created: dict[str, Path] = {}
        if with_wheel:
            wheel_dir = cache_root / "sdists-v9" / "path" / "deadbeef" / "abcd1234"
            wheel_dir.mkdir(parents=True)
            wheel = wheel_dir / "spatial_memory_mcp-2.0.0-py3-none-any.whl"
            wheel.write_bytes(b"PK\x03\x04stub")
            created["wheel_hash_dir"] = wheel_dir.parent
        if with_archive:
            arch_dir = cache_root / "archive-v0" / "ABCDEFGHIJKLMNOPQR"
            (arch_dir / "spatial_memory_mcp-2.0.0.dist-info").mkdir(parents=True)
            (arch_dir / "spatial_memory" / "core").mkdir(parents=True)
            created["archive_dir"] = arch_dir
        if unrelated:
            other = cache_root / "archive-v0" / "ZZZZUNRELATEDZZZ"
            (other / "some_other_pkg-1.0.dist-info").mkdir(parents=True)
            created["unrelated"] = other
        return created

    def test_pypi_source_is_no_op(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        cache_root = tmp_path / ".cache" / "uv"
        created = self._fake_uv_cache(cache_root)

        removed = _evict_local_path_uv_cache("spatial-memory-mcp")

        assert removed == []
        assert created["wheel_hash_dir"].exists()
        assert created["archive_dir"].exists()

    def test_local_path_evicts_wheel_and_archive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        cache_root = home / ".cache" / "uv"
        created = self._fake_uv_cache(cache_root, unrelated=True)
        repo = tmp_path / "repo"
        repo.mkdir()

        removed = _evict_local_path_uv_cache(str(repo))

        assert created["wheel_hash_dir"] in removed
        assert created["archive_dir"] in removed
        assert not created["wheel_hash_dir"].exists()
        assert not created["archive_dir"].exists()
        # Unrelated archive entries are untouched.
        assert created["unrelated"].exists()

    def test_nonexistent_path_is_no_op(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        self._fake_uv_cache(tmp_path / ".cache" / "uv")

        removed = _evict_local_path_uv_cache(str(tmp_path / "does-not-exist"))

        assert removed == []

    def test_empty_cache_returns_empty_list(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        repo = tmp_path / "repo"
        repo.mkdir()

        assert _evict_local_path_uv_cache(str(repo)) == []


class TestCheckOnlyFlag:
    """Tests for `setup --check`."""

    def test_check_exits_zero_when_initialized(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from spatial_memory.config import Settings, override_settings, reset_settings
        from spatial_memory.tools.setup_runner import run_setup

        write_sentinel(tmp_path, model="arctic-m-long", dim=768, version="2.0.0")
        settings = Settings(memory_path=tmp_path, embedding_model="arctic-m-long")
        override_settings(settings)
        try:
            args = type("A", (), {"check": True, "force": False, "model": None, "verbose": False})()
            assert run_setup(args) == 0
        finally:
            reset_settings()

    def test_check_exits_nonzero_when_missing(self, tmp_path: Path) -> None:
        from spatial_memory.config import Settings, override_settings, reset_settings
        from spatial_memory.tools.setup_runner import run_setup

        settings = Settings(memory_path=tmp_path, embedding_model="arctic-m-long")
        override_settings(settings)
        try:
            args = type("A", (), {"check": True, "force": False, "model": None, "verbose": False})()
            assert run_setup(args) == 1
        finally:
            reset_settings()
