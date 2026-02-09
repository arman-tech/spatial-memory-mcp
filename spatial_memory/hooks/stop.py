"""Stop hook script for Claude Code cognitive offloading.

Scans the session transcript for assistant text (decisions, analyses,
solutions) that PostToolUse missed, classifies signals, redacts secrets,
and writes qualifying content to the Maildir queue before session end.

**Loop guard**: When ``stop_hook_active`` is ``True`` in the input, the
hook skips all processing to prevent infinite loops.

**Fail-open**: All exceptions are caught; the script always exits 0.
Never exit 2 — this hook is non-blocking and should never disrupt the user.

**Stdout response**: Always emits ``{"continue": true, "suppressOutput": true}``
so Claude stops normally without showing hook output.

**Stdlib-only imports at module level**.  Sibling hook modules are loaded
via ``importlib.util.spec_from_file_location()`` to bypass
``spatial_memory/__init__.py`` which pulls in heavy dependencies
(lancedb, numpy, sentence-transformers).

Exit codes:
    0 — Always.  Success, skip, or silent failure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Module loading (bypass spatial_memory/__init__.py)
# ---------------------------------------------------------------------------

_HOOKS_DIR = Path(__file__).resolve().parent


def _load_hook_module(module_name: str) -> ModuleType:
    """Load a sibling module by file path, bypassing the package __init__.

    Args:
        module_name: Module name without ``.py`` extension
            (e.g. ``"signal_detection"``).

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the module file does not exist.
    """
    module_path = _HOOKS_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(f"Hook module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        f"spatial_memory.hooks.{module_name}",
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclasses can resolve the module
    full_name = f"spatial_memory.hooks.{module_name}"
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# stdout safety net (must work even if hook_helpers fails to load)
# ---------------------------------------------------------------------------


def _write_stdout_response() -> None:
    """Write the standard hook response to stdout.

    Uses a string literal to avoid depending on ``json`` import at call time.
    This function is the fail-safe fallback — it MUST succeed even if no
    sibling modules could be loaded.
    """
    try:
        sys.stdout.write('{"continue": true, "suppressOutput": true}\n')
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Entrypoint. Fail-open: catches all exceptions, always exits 0."""
    try:
        # Load shared helpers first (stdin, env, sanitization)
        helpers = _load_hook_module("hook_helpers")

        data = helpers.read_stdin()
        if not data:
            _write_stdout_response()
            return

        # LOOP GUARD: If stop_hook_active is True, Claude is already
        # continuing due to a stop hook. Skip processing to prevent
        # infinite loops / unnecessary work.
        if data.get("stop_hook_active", False):
            _write_stdout_response()
            return

        # Load sibling modules via importlib (no heavy deps)
        models_mod = _load_hook_module("models")
        reader_mod = _load_hook_module("transcript_reader")
        extractor_mod = _load_hook_module("transcript_extractor")
        pipeline_mod = _load_hook_module("transcript_pipeline")
        signal_mod = _load_hook_module("signal_detection")
        redaction_mod = _load_hook_module("redaction")
        writer_mod = _load_hook_module("queue_writer")
        overlap_mod = _load_hook_module("overlap_detector")

        # Sanitize inputs before building the dataclass
        session_id = helpers.sanitize_session_id(str(data.get("session_id", "")))
        transcript_path = helpers.validate_transcript_path(str(data.get("transcript_path", "")))

        # Build TranscriptHookInput
        hook_input = models_mod.TranscriptHookInput(
            session_id=session_id,
            transcript_path=transcript_path,
            cwd=str(data.get("cwd", "")),
            permission_mode=str(data.get("permission_mode", "")),
            hook_event_name=str(data.get("hook_event_name", "Stop")),
            trigger="session_end",
            stop_hook_active=bool(data.get("stop_hook_active", False)),
        )

        # Skip if no transcript path (empty after validation = invalid)
        if not hook_input.transcript_path:
            _write_stdout_response()
            return

        project_root = helpers.get_project_root(cwd=hook_input.cwd)
        queue_dir = writer_mod.get_queue_dir(project_root=project_root)

        pipeline_mod.run_transcript_pipeline(
            hook_input,
            read_fn=reader_mod.read_transcript_delta,
            extract_fn=extractor_mod.extract_assistant_text,
            classify_fn=signal_mod.classify_signal,
            redact_fn=redaction_mod.redact_secrets,
            write_fn=writer_mod.write_queue_file,
            load_state_fn=reader_mod.load_state,
            save_state_fn=reader_mod.save_state,
            get_queued_hashes_fn=overlap_mod.get_queued_hashes,
            is_duplicate_fn=overlap_mod.is_duplicate,
            queue_dir=queue_dir,
            project_root=project_root,
        )
    except Exception:
        pass  # Fail-open

    _write_stdout_response()


if __name__ == "__main__":
    main()
