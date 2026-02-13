"""Unified dispatcher for all hook events.

Routes hook invocations to the correct handler based on event name.
Supports both Claude Code (PascalCase) and Cursor (camelCase) event formats.

**STDLIB-ONLY** at module level.  Sibling hook modules are loaded lazily
via ``_load_hook_module()`` to bypass ``spatial_memory/__init__.py``.

CLI usage::

    echo '{"source":"startup"}' | python -m spatial_memory hook session-start --client claude-code
    echo '{"tool_name":"Edit",...}' | python dispatcher.py post-tool-use --client claude-code

All exceptions are caught (fail-open).  Exit code is always 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable as _Callable
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOOKS_DIR = Path(__file__).resolve().parent

_EVENT_ALIASES: dict[str, str] = {
    # PascalCase (Claude Code canonical)
    "SessionStart": "SessionStart",
    "PostToolUse": "PostToolUse",
    "PreCompact": "PreCompact",
    "Stop": "Stop",
    # camelCase (Cursor)
    "sessionStart": "SessionStart",
    "postToolUse": "PostToolUse",
    "preCompact": "PreCompact",
    "stop": "Stop",
    # kebab-case (CLI)
    "session-start": "SessionStart",
    "post-tool-use": "PostToolUse",
    "pre-compact": "PreCompact",
    # Cursor-specific
    "afterMCPExecution": "PostToolUse",
    "after-mcp-execution": "PostToolUse",
}

# ---------------------------------------------------------------------------
# Module loading (single copy, replaces triplication)
# ---------------------------------------------------------------------------

_module_cache: dict[str, ModuleType] = {}


def _load_hook_module(module_name: str) -> ModuleType:
    """Load a sibling module by file path, bypassing the package __init__.

    Results are cached in ``sys.modules`` to avoid redundant loads.

    Args:
        module_name: Module name without ``.py`` extension.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the module file does not exist.
    """
    full_name = f"spatial_memory.hooks.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = _HOOKS_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(f"Hook module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(full_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Event normalization + client detection
# ---------------------------------------------------------------------------


def _normalize_event(raw: str) -> str | None:
    """Normalize an event name to canonical PascalCase.

    Returns ``None`` if the event is not recognized.
    """
    return _EVENT_ALIASES.get(raw)


def _parse_client_flag(argv: list[str]) -> str:
    """Extract --client value from argv, default to claude-code."""
    for i, arg in enumerate(argv):
        if arg == "--client" and i + 1 < len(argv):
            return argv[i + 1]
    return "claude-code"


def _detect_client(data: dict[str, object], explicit: str) -> str:
    """Detect client from explicit flag or stdin data heuristics."""
    if explicit and explicit != "claude-code":
        return explicit
    if data.get("cursor_version") or data.get("cursorVersion"):
        return "cursor"
    return explicit or "claude-code"


# ---------------------------------------------------------------------------
# Stdin normalization (cross-client field mapping)
# ---------------------------------------------------------------------------


def _normalize_stdin(data: dict[str, object], client: str) -> dict[str, object]:
    """Normalize stdin fields for cross-client compatibility.

    Cursor uses ``conversation_id``, ``workspace_roots``, ``result_json``,
    and ``status`` where Claude Code uses ``session_id``, ``cwd``,
    ``tool_response``, and ``trigger``.
    """
    if client != "cursor":
        return data

    result = dict(data)

    # conversation_id -> session_id
    if "conversation_id" in result and "session_id" not in result:
        result["session_id"] = result.pop("conversation_id")

    # workspace_roots -> cwd (take first)
    if "workspace_roots" in result and "cwd" not in result:
        roots = result.get("workspace_roots")
        if isinstance(roots, list) and roots:
            result["cwd"] = str(roots[0])

    # Fix Cursor's Unix-style drive paths on Windows: /c:/Users/... -> C:/Users/...
    # Cursor sends workspace_roots with leading slash before drive letter.
    # os.path.isabs("/c:/...") returns False on Windows, breaking validate_cwd().
    cwd = result.get("cwd", "")
    if isinstance(cwd, str) and len(cwd) >= 3 and cwd[0] == "/" and cwd[2] == ":":
        result["cwd"] = cwd[1].upper() + cwd[2:]

    # tool_output / result_json -> tool_response
    if "tool_output" in result and "tool_response" not in result:
        result["tool_response"] = result.pop("tool_output")
    elif "result_json" in result and "tool_response" not in result:
        result["tool_response"] = result.pop("result_json")

    # status -> trigger (for Stop)
    if "status" in result and "trigger" not in result:
        result["trigger"] = result.pop("status")

    # Synthesize source for SessionStart if missing
    if "source" not in result:
        result["source"] = "startup"

    return result


# ---------------------------------------------------------------------------
# Centralized validation
# ---------------------------------------------------------------------------


def _validate_common(data: dict[str, object]) -> dict[str, object]:
    """Apply centralized validation to stdin data.

    Sanitizes session_id, transcript_path, and cwd in-place.
    """
    helpers = _load_hook_module("hook_helpers")

    session_id = str(data.get("session_id", ""))
    if session_id:
        data["session_id"] = helpers.sanitize_session_id(session_id)

    transcript_path = str(data.get("transcript_path", ""))
    if transcript_path:
        data["transcript_path"] = helpers.validate_transcript_path(transcript_path)

    cwd = str(data.get("cwd", ""))
    if cwd:
        data["cwd"] = helpers.validate_cwd(cwd)

    return data


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

_RECALL_NUDGE = (
    "SPATIAL MEMORY: Call `recall` with a brief summary of the user's "
    "apparent task/context to load relevant memories from previous sessions."
)

_TRIGGER_SOURCES = frozenset({"startup", "resume"})


def _handle_session_start(data: dict[str, object], client: str) -> dict[str, object] | None:
    """Handle SessionStart — self-contained, no module loads.

    Returns a response dict or None if the event should be skipped.
    """
    source = str(data.get("source", ""))
    if source not in _TRIGGER_SOURCES:
        return None

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": _RECALL_NUDGE,
        }
    }


def _handle_post_tool_use(data: dict[str, object], client: str) -> dict[str, object] | None:
    """Handle PostToolUse — loads modules, runs pipeline."""
    helpers = _load_hook_module("hook_helpers")
    models_mod = _load_hook_module("models")

    tool_name = str(data.get("tool_name", ""))
    skip, _reason = models_mod.should_skip_tool(tool_name)
    if skip:
        return None

    extractor_mod = _load_hook_module("content_extractor")
    pipeline_mod = _load_hook_module("pipeline")
    signal_mod = _load_hook_module("signal_detection")
    redaction_mod = _load_hook_module("redaction")
    writer_mod = _load_hook_module("queue_writer")

    # Parse tool_response
    tool_response = data.get("tool_response", "")
    if not isinstance(tool_response, str):
        try:
            tool_response = json.dumps(tool_response, ensure_ascii=False)
        except (TypeError, ValueError):
            tool_response = str(tool_response) if tool_response else ""

    tool_input = data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    hook_input = models_mod.HookInput(
        session_id=str(data.get("session_id", "")),
        tool_name=tool_name,
        tool_input=tool_input,
        tool_response=tool_response,
        tool_use_id=str(data.get("tool_use_id", "")),
        transcript_path=str(data.get("transcript_path", "")),
        cwd=str(data.get("cwd", "")),
        hook_event_name=str(data.get("hook_event_name", "")),
        permission_mode=str(data.get("permission_mode", "")),
    )

    pipeline_mod.run_pipeline(
        hook_input,
        extract_fn=extractor_mod.extract_content,
        classify_fn=signal_mod.classify_signal,
        redact_fn=redaction_mod.redact_secrets,
        write_fn=writer_mod.write_queue_file,
        project_root=helpers.get_project_root(cwd=hook_input.cwd),
        client=client,
    )

    return None  # PostToolUse has no meaningful response


def _handle_transcript(
    data: dict[str, object], client: str, event_name: str
) -> dict[str, object] | None:
    """Shared handler for PreCompact and Stop (transcript-based events)."""
    helpers = _load_hook_module("hook_helpers")
    models_mod = _load_hook_module("models")
    reader_mod = _load_hook_module("transcript_reader")
    extractor_mod = _load_hook_module("transcript_extractor")
    pipeline_mod = _load_hook_module("transcript_pipeline")
    signal_mod = _load_hook_module("signal_detection")
    redaction_mod = _load_hook_module("redaction")
    writer_mod = _load_hook_module("queue_writer")
    overlap_mod = _load_hook_module("overlap_detector")

    session_id = str(data.get("session_id", ""))
    transcript_path = str(data.get("transcript_path", ""))

    if not transcript_path:
        return None

    # Stop-specific: trigger defaults to "session_end"
    trigger = str(data.get("trigger", ""))
    if event_name == "Stop" and not trigger:
        trigger = "session_end"

    hook_input = models_mod.TranscriptHookInput(
        session_id=session_id,
        transcript_path=transcript_path,
        cwd=str(data.get("cwd", "")),
        permission_mode=str(data.get("permission_mode", "")),
        hook_event_name=event_name,
        trigger=trigger,
        custom_instructions=str(data.get("custom_instructions", "")),
        stop_hook_active=bool(data.get("stop_hook_active", False)),
    )

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
        client=client,
    )

    return None


def _handle_pre_compact(data: dict[str, object], client: str) -> dict[str, object] | None:
    """Handle PreCompact."""
    return _handle_transcript(data, client, "PreCompact")


def _handle_stop(data: dict[str, object], client: str) -> dict[str, object] | None:
    """Handle Stop — includes loop guard."""
    if data.get("stop_hook_active", False):
        return None  # Loop guard
    return _handle_transcript(data, client, "Stop")


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HandlerFn = _Callable[[dict[str, object], str], dict[str, object] | None]

_HANDLER_MAP: dict[str, _HandlerFn] = {
    "SessionStart": _handle_session_start,
    "PostToolUse": _handle_post_tool_use,
    "PreCompact": _handle_pre_compact,
    "Stop": _handle_stop,
}


# ---------------------------------------------------------------------------
# Primary dispatch function (testable surface)
# ---------------------------------------------------------------------------


def dispatch(event: str, client: str, data: dict[str, object]) -> dict[str, object] | None:
    """Dispatch a hook event to the appropriate handler.

    Args:
        event: Canonical event name (PascalCase).
        client: Client identifier.
        data: Parsed stdin JSON.

    Returns:
        Response dict, or ``None`` if no output needed.
    """
    handler = _HANDLER_MAP.get(event)
    if handler is None:
        return None
    return handler(data, client)


# ---------------------------------------------------------------------------
# Cursor output translation
# ---------------------------------------------------------------------------


def _translate_output_for_cursor(
    response: dict[str, object] | None, event: str
) -> dict[str, object] | None:
    """Translate output for Cursor (fire-and-forget — Cursor ignores stdout)."""
    if response is None:
        return None

    # SessionStart: translate hookSpecificOutput to flat format
    if event == "SessionStart":
        hook_output = response.get("hookSpecificOutput", {})
        if isinstance(hook_output, dict):
            context = hook_output.get("additionalContext", "")
            result: dict[str, object] = {"continue": True}
            if context:
                result["additional_context"] = context
            return result

    return response


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python dispatcher.py <event> [--client <client>]``.

    Reads stdin, dispatches to handler, writes stdout response.
    Always exits 0 (fail-open).
    """
    try:
        # Hand-parse args (no argparse — speed matters)
        argv = sys.argv[1:]

        if not argv or argv[0] in ("--help", "-h"):
            print(
                "Usage: dispatcher.py <event> [--client <client>]\n"
                "Events: session-start, post-tool-use, pre-compact, stop"
            )
            return

        raw_event = argv[0]
        event = _normalize_event(raw_event)
        if event is None:
            return

        client_flag = _parse_client_flag(argv)

        # Read stdin
        raw = sys.stdin.buffer.read(524_288)
        if not raw or not raw.strip():
            data: dict[str, object] = {}
        else:
            parsed = json.loads(raw)
            data = parsed if isinstance(parsed, dict) else {}

        # Detect client and normalize
        client = _detect_client(data, client_flag)
        data = _normalize_stdin(data, client)
        data = _validate_common(data)

        # Dispatch
        response = dispatch(event, client, data)

        # Translate for Cursor if needed
        if client == "cursor":
            response = _translate_output_for_cursor(response, event)

        # Write stdout response
        if event == "SessionStart" and response is not None:
            json.dump(response, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif event in ("PostToolUse", "Stop"):
            # PostToolUse and Stop always emit continue/suppressOutput
            sys.stdout.write('{"continue": true, "suppressOutput": true}\n')
            sys.stdout.flush()
        # PreCompact: no stdout output

    except Exception as exc:
        # Fail-open: log error if possible, never crash
        try:
            helpers = _load_hook_module("hook_helpers")
            log_cwd = ""
            try:
                log_cwd = str(data.get("cwd", ""))
            except Exception:
                pass
            log_event = ""
            try:
                log_event = raw_event
            except Exception:
                pass
            helpers.log_hook_error(exc, f"dispatcher:{log_event}", log_cwd)
        except Exception:
            pass


if __name__ == "__main__":
    main()
