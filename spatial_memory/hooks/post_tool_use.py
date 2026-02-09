"""PostToolUse hook script for Claude Code cognitive offloading.

Receives tool call JSON on stdin, classifies signals, redacts secrets,
writes qualifying content to the Maildir queue for server-side processing.

**Fail-open**: All exceptions are caught; the script always exits 0.
Never exit 2 — this hook is non-blocking and should never disrupt the user.

**Stdlib-only imports at module level**.  Sibling hook modules are loaded
via ``importlib.util.spec_from_file_location()`` to bypass
``spatial_memory/__init__.py`` which pulls in heavy dependencies
(lancedb, numpy, sentence-transformers).

Exit codes:
    0 — Always.  Success, skip, or silent failure.
"""

from __future__ import annotations

import importlib.util
import json
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

        # Read input
        data = helpers.read_stdin()
        if not data:
            _write_stdout_response()
            return

        # Early exit: check tool filter before loading heavy modules
        models_mod = _load_hook_module("models")
        tool_name = str(data.get("tool_name", ""))
        skip, _reason = models_mod.should_skip_tool(tool_name)
        if skip:
            _write_stdout_response()
            return

        # Load remaining sibling modules (only reached for non-skipped tools)
        extractor_mod = _load_hook_module("content_extractor")
        pipeline_mod = _load_hook_module("pipeline")
        signal_mod = _load_hook_module("signal_detection")
        redaction_mod = _load_hook_module("redaction")
        writer_mod = _load_hook_module("queue_writer")

        # Parse tool_response to string if needed
        tool_response = data.get("tool_response", "")
        if not isinstance(tool_response, str):
            try:
                tool_response = json.dumps(tool_response, ensure_ascii=False)
            except (TypeError, ValueError):
                tool_response = str(tool_response) if tool_response else ""

        # Build HookInput
        tool_input = data.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}

        hook_input = models_mod.HookInput(
            session_id=str(data.get("session_id", "")),
            tool_name=str(data.get("tool_name", "")),
            tool_input=tool_input,
            tool_response=tool_response,
            tool_use_id=str(data.get("tool_use_id", "")),
            transcript_path=str(data.get("transcript_path", "")),
            cwd=str(data.get("cwd", "")),
            hook_event_name=str(data.get("hook_event_name", "")),
            permission_mode=str(data.get("permission_mode", "")),
        )

        # Run pipeline
        pipeline_mod.run_pipeline(
            hook_input,
            extract_fn=extractor_mod.extract_content,
            classify_fn=signal_mod.classify_signal,
            redact_fn=redaction_mod.redact_secrets,
            write_fn=writer_mod.write_queue_file,
            project_root=helpers.get_project_root(),
        )

    except Exception:
        # Fail-open: swallow all errors silently
        pass

    _write_stdout_response()


if __name__ == "__main__":
    main()
