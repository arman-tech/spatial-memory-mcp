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
import os
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
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# stdin / stdout helpers
# ---------------------------------------------------------------------------


def _read_stdin() -> dict[str, object]:
    """Read and parse JSON from stdin.

    Returns:
        Parsed dict, or empty dict on any error.
    """
    try:
        raw = sys.stdin.read()
        if not raw or not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _write_stdout_and_exit() -> None:
    """Write the standard PostToolUse response to stdout and exit 0.

    Output: ``{"continue": true, "suppressOutput": true}``
    """
    try:
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass


def _get_project_root() -> str:
    """Resolve the project root from environment.

    Uses ``$CLAUDE_PROJECT_DIR`` (preferred), falling back to empty string.
    """
    return os.environ.get("CLAUDE_PROJECT_DIR", "")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Entrypoint. Fail-open: catches all exceptions, always exits 0."""
    try:
        # Read input
        data = _read_stdin()
        if not data:
            _write_stdout_and_exit()
            return

        # Load sibling modules via importlib (no heavy deps)
        models_mod = _load_hook_module("models")
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
            project_root=_get_project_root(),
        )

    except Exception:
        # Fail-open: swallow all errors silently
        pass

    _write_stdout_and_exit()


if __name__ == "__main__":
    main()
