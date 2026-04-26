"""Pre-warm setup command for spatial-memory.

Runs out-of-band before MCP startup so the model + dependencies are cached
on disk and the MCP handshake completes within the 30s client timeout.

Writes a `.initialized` sentinel JSON to the configured memory_path on
success; the MCP server's strict startup check refuses to start without it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENTINEL_FILENAME = ".initialized"
SENTINEL_SCHEMA = 1
DEFAULT_SOURCE = "spatial-memory-mcp"
SCRIPT_NAME = "spatial-memory-mcp"


def _w(msg: str = "") -> None:
    """Write to stderr (progress goes to stderr; stdout reserved for results)."""
    print(msg, file=sys.stderr, flush=True)


def _phase(idx: int, total: int, title: str) -> None:
    _w(f"\n[{idx}/{total}] {title}")


def _ok(label: str, detail: str = "") -> None:
    _w(f"      ok    {label}" + (f"   {detail}" if detail else ""))


def _info(label: str, value: Any) -> None:
    # Right-pad the label to ~22 cols (long enough for "sentence-transformers")
    # and always emit at least one separating space so labels never butt up
    # against their values.
    _w(f"      {label:<22} {value}")


def read_sentinel(memory_path: Path) -> dict[str, Any] | None:
    """Read the `.initialized` sentinel. Returns None if missing or invalid."""
    sentinel = memory_path / SENTINEL_FILENAME
    if not sentinel.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(sentinel.read_text(encoding="utf-8"))
        return data
    except (OSError, json.JSONDecodeError):
        return None


def write_sentinel(
    memory_path: Path,
    *,
    model: str,
    dim: int,
    version: str,
) -> None:
    """Write `.initialized` sentinel JSON to memory_path."""
    memory_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SENTINEL_SCHEMA,
        "spatial_memory_version": version,
        "embedding_model": model,
        "embedding_dim": dim,
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }
    (memory_path / SENTINEL_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def strict_startup_error(memory_path: Path, configured_model: str) -> str | None:
    """Validate the sentinel against current config. Returns an actionable
    error message if the MCP server should refuse to start, or None if OK.
    """
    sentinel = read_sentinel(memory_path)
    if sentinel is None:
        return _format_missing_sentinel(memory_path, configured_model)
    stored_model = sentinel.get("embedding_model")
    if stored_model != configured_model:
        return _format_model_mismatch(memory_path, configured_model, stored_model)
    return None


def _format_missing_sentinel(memory_path: Path, model: str) -> str:
    return textwrap.dedent(
        f"""
        spatial-memory-mcp is not initialized.

        Run this once before starting your MCP client:

            uvx --from spatial-memory-mcp spatial-memory setup

        This downloads the embedding model and validates your config.
        Without it the MCP handshake will time out before the model can load.

        Configured:
          memory_path:     {memory_path}
          embedding_model: {model}
        """
    ).strip()


def _format_model_mismatch(memory_path: Path, configured: str, stored: str | None) -> str:
    return textwrap.dedent(
        f"""
        spatial-memory-mcp setup state does not match the current configuration.

        Configured model:   {configured}
        Setup ran for:      {stored}

        Re-run setup so the new model is cached:

            uvx --from spatial-memory-mcp spatial-memory setup --force

        memory_path: {memory_path}
        """
    ).strip()


def _format_failure(stage: str, exc: BaseException, memory_path: Path) -> str:
    free_gb: float | None
    try:
        target = memory_path.parent if memory_path.exists() else Path.home()
        free_gb = shutil.disk_usage(target).free / (1024**3)
    except OSError:
        free_gb = None
    diag_lines = [
        "",
        f"setup failed during: {stage}",
        f"  error:        {type(exc).__name__}: {exc}",
        f"  memory_path:  {memory_path}",
    ]
    if free_gb is not None:
        diag_lines.append(f"  free disk:    {free_gb:.1f} GB")
    diag_lines += [
        "",
        "Common causes:",
        "  - Network unreachable while fetching the embedding model",
        "  - Insufficient disk space (model + deps need ~3 GB on first install)",
        "  - Invalid embedding_model alias",
        "",
        "Re-run with --verbose for a full traceback, or `spatial-memory doctor`",
        "for a configuration report.",
    ]
    return "\n".join(diag_lines)


def _build_mcp_entry(source: str, memory_path: Path, model_alias: str) -> dict[str, Any]:
    """Generate the canonical .mcp.json entry for spatial-memory."""
    return {
        "command": "uvx",
        "args": ["--from", source, SCRIPT_NAME],
        "env": {
            "SPATIAL_MEMORY_MEMORY_PATH": str(memory_path),
            "SPATIAL_MEMORY_EMBEDDING_MODEL": model_alias,
        },
    }


def _ensure_mcp_entry(
    mcp_path: Path,
    desired: dict[str, Any],
    *,
    force: bool,
) -> tuple[dict[str, Any], str]:
    """Read or create `mcp_path` and ensure its spatial-memory entry matches `desired`.

    Returns (full_config, status) where status is one of:
        "created", "added entry", "already matches", "overwritten (--force)".

    Raises RuntimeError if an existing entry diverges and `force` is False.
    """
    if not mcp_path.exists():
        config: dict[str, Any] = {"mcpServers": {"spatial-memory": desired}}
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config, "created"

    try:
        config = json.loads(mcp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse {mcp_path}: {e}") from e

    if not isinstance(config, dict):
        raise RuntimeError(f"{mcp_path} top-level is not a JSON object")

    servers = config.setdefault("mcpServers", {})
    existing = servers.get("spatial-memory")

    if existing is None:
        servers["spatial-memory"] = desired
        mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config, "added entry"

    if existing == desired:
        return config, "already matches"

    if force:
        servers["spatial-memory"] = desired
        mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config, "overwritten (--force)"

    raise RuntimeError(
        f"existing spatial-memory entry in {mcp_path} does not match the values "
        f"setup was invoked with. Re-run with --force to overwrite, or edit "
        f"{mcp_path} by hand to align."
    )


def _evict_local_path_uv_cache(source: str) -> list[Path]:
    """Evict uv's path-keyed wheel cache + extracted archives for spatial-memory.

    uv keys its source-build cache under `~/.cache/uv/sdists-v9/path/<hash>/`
    by source path, NOT by source mtime. When a developer iterates on a local
    repo (`uvx --from /abs/path/to/repo ...`), edits to source files do not
    invalidate this cache — uv reuses the stale wheel forever. This causes
    `--force` to be a no-op for the package code itself.

    PyPI sources (`--from spatial-memory-mcp`) are version-keyed and
    self-invalidate, so we only act on local paths.

    Returns the list of removed cache directories (empty if none matched).
    """
    try:
        source_path = Path(source).expanduser().resolve()
    except (OSError, RuntimeError):
        return []
    if not source_path.is_dir():
        return []

    cache_root = Path.home() / ".cache" / "uv"
    removed: list[Path] = []

    # Path-keyed built wheels: sdists-v9/path/<hash>/<id>/spatial_memory_mcp-*.whl
    path_root = cache_root / "sdists-v9" / "path"
    if path_root.is_dir():
        for wheel in path_root.glob("*/*/spatial_memory_mcp-*.whl"):
            target = wheel.parent.parent  # the <hash>/ dir
            if target.is_dir() and target not in removed:
                shutil.rmtree(target, ignore_errors=True)
                removed.append(target)

    # Extracted archives: archive-v0/<id>/spatial_memory_mcp-*.dist-info
    archive_root = cache_root / "archive-v0"
    if archive_root.is_dir():
        for dist_info in archive_root.glob("*/spatial_memory_mcp-*.dist-info"):
            target = dist_info.parent  # the <id>/ dir
            if target.is_dir() and target not in removed:
                shutil.rmtree(target, ignore_errors=True)
                removed.append(target)

    return removed


def _warm_uvx_cache(uvx_args: list[str]) -> None:
    """Run `uvx <uvx_args> --version` so uvx materializes its venv into the
    user's uv cache. Streams uvx output through unmodified.
    """
    import subprocess

    if shutil.which("uvx") is None:
        raise RuntimeError(
            "`uvx` not found on PATH. Install uv first: "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
    cmd = ["uvx", *uvx_args, "--version"]
    _w(f"      cmd  {' '.join(cmd)}")
    _w("")
    subprocess.run(cmd, check=True)


def run_setup(args: argparse.Namespace) -> int:
    """Six-phase pre-warm. Writes sentinel on success."""
    from spatial_memory import __version__
    from spatial_memory.config import get_settings, override_settings
    from spatial_memory.core.embedding_registry import resolve_model

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"embedding_model": args.model})
        override_settings(settings)

    memory_path: Path = settings.memory_path
    model_alias = settings.embedding_model

    if args.check:
        err = strict_startup_error(memory_path, model_alias)
        if err is None:
            _w("ok: spatial-memory is initialized")
            sentinel = read_sentinel(memory_path)
            if sentinel:
                _info("model:", sentinel.get("embedding_model"))
                _info("dim:", sentinel.get("embedding_dim"))
                _info("when:", sentinel.get("initialized_at"))
            return 0
        _w(err)
        return 1

    if not args.force and read_sentinel(memory_path) is not None:
        # Sentinel exists and matches (we'll re-validate per-phase below);
        # still re-run idempotently so users get a confirming report.
        pass

    total = 6
    _w("=" * 60)
    _w(f"spatial-memory setup  v{__version__}")
    _w("=" * 60)

    # ---------------- Phase 1: Python environment ---------------------------
    _phase(1, total, "Python environment")
    _info("python", sys.version.split()[0])
    try:
        import importlib.metadata as md

        for dist in ("lancedb", "sentence-transformers", "mcp", "numpy"):
            try:
                _info(dist, md.version(dist))
            except md.PackageNotFoundError:
                _info(dist, "NOT INSTALLED")
                exc = RuntimeError(f"{dist} missing")
                _w(_format_failure("Python environment", exc, memory_path))
                return 2
    except Exception as e:
        _w(_format_failure("Python environment", e, memory_path))
        return 2
    _ok("dependencies importable")

    # ---------------- Phase 2: MCP config + uvx warm-up ---------------------
    _phase(2, total, "MCP config + uvx warm-up")
    if args.no_mcp_config:
        _ok("skipped (--no-mcp-config)")
    else:
        if args.force:
            evicted = _evict_local_path_uv_cache(args.source)
            if evicted:
                _info("evicted uv cache", f"{len(evicted)} dir(s)")
                for p in evicted:
                    _w(f"        - {p}")
        mcp_path = Path(args.mcp_config) if args.mcp_config else Path.cwd() / ".mcp.json"
        desired = _build_mcp_entry(args.source, memory_path, model_alias)
        try:
            config, status = _ensure_mcp_entry(mcp_path, desired, force=args.force)
        except RuntimeError as e:
            _w(_format_failure("MCP config", e, memory_path))
            return 2
        _info("path", mcp_path)
        _info("entry status", status)

        entry = config["mcpServers"]["spatial-memory"]
        if entry.get("command") == "uvx":
            _w("      warming uvx venv (first run downloads ~2 GB; cache persists in ~/.cache/uv/)")
            try:
                _warm_uvx_cache(entry["args"])
            except Exception as e:
                _w(_format_failure("uvx warm-up", e, memory_path))
                if args.verbose:
                    raise
                return 2
            _ok("uvx venv cached")
        else:
            _ok(f"command={entry.get('command')!r} - skipping uvx warm-up")

    # ---------------- Phase 3: Embedding model ------------------------------
    _phase(3, total, f"Embedding model: {model_alias}")
    try:
        cfg = resolve_model(model_alias)
        _info("hf_name", cfg.hf_name or model_alias)
        _info("dim", cfg.dim or "(probed at load)")
        _info("max_seq", cfg.max_seq_length or "(probed at load)")
        _info("cache", str(Path.home() / ".cache" / "huggingface" / "hub"))
        _w("      loading (HuggingFace progress bars stream below if download needed)")
        _w("")

        from spatial_memory.core.embeddings import EmbeddingService

        svc = EmbeddingService(model_name=model_alias)
        # Force load by accessing dimensions
        dim = svc.dimensions
        _ok(f"loaded   dim={dim}   backend={svc.backend}")
    except Exception as e:
        _w(_format_failure("Embedding model load", e, memory_path))
        if args.verbose:
            raise
        return 2

    # ---------------- Phase 4: Smoke test -----------------------------------
    _phase(4, total, "Smoke test")
    try:
        vec = svc.embed_passage("hello world")
        if len(vec) != dim:
            raise RuntimeError(f"smoke-test dim mismatch: got {len(vec)} expected {dim}")
        _ok(f'embed("hello world") -> {len(vec)}-dim vector')
    except Exception as e:
        _w(_format_failure("Smoke test", e, memory_path))
        if args.verbose:
            raise
        return 2

    # ---------------- Phase 5: Config ---------------------------------------
    _phase(5, total, "Config")
    _info("memory_path", memory_path)
    _info("model", model_alias)
    _info("log_level", settings.log_level)
    _ok("config valid")

    # ---------------- Phase 6: Database -------------------------------------
    _phase(6, total, "Database")
    try:
        from spatial_memory.core.database import Database

        db = Database(storage_path=memory_path, embedding_dim=dim)
        db.connect()
        try:
            count = db.count_memories() if hasattr(db, "count_memories") else 0
            _info("path", memory_path)
            _info("memories", count)
            _ok("database ready")
        finally:
            db.close()
    except Exception as e:
        _w(_format_failure("Database init", e, memory_path))
        if args.verbose:
            raise
        return 2

    # ---------------- Sentinel ----------------------------------------------
    try:
        write_sentinel(memory_path, model=model_alias, dim=dim, version=__version__)
    except Exception as e:
        _w(_format_failure("Writing sentinel", e, memory_path))
        return 2

    _w("")
    _w("=" * 60)
    _w("ready. spatial-memory-mcp will start within the MCP handshake window.")
    if not args.no_mcp_config:
        cfg_path = Path(args.mcp_config) if args.mcp_config else Path.cwd() / ".mcp.json"
        _w(f"MCP config: {cfg_path}")
        _w("Start your MCP client (Claude Code, Cursor, etc.) and the server will connect.")
    else:
        _w("MCP client config was not touched (--no-mcp-config). Add the spatial-memory")
        _w("entry to your client config manually before starting it.")
    return 0
