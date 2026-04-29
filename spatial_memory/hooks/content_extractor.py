"""Per-tool content extraction with dispatch table.

Extracts memory-worthy text from tool invocations.  Each tool type has
a dedicated extractor function; the dispatch table maps tool names to
extractors (OCP — add extractors without modifying ``extract_content``).

**STDLIB-ONLY**: Only ``json`` imports allowed.

Per spec §8.1 (D6), there is a single 200_000-character safety cap applied
at the final extraction boundary; v1 per-field caps are removed because
token-based capture sizing now lives in the upstream capture pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_SAFETY_CAP = 200_000
"""Single tail-drop safety cap on total extracted content (US-018, spec §8.1)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters, appending ``...`` if trimmed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _safe_get(d: dict[str, object], key: str, default: str = "") -> str:
    """Get a string value from *d*, coercing non-strings via ``str()``."""
    value = d.get(key, default)
    if isinstance(value, str):
        return value
    return str(value) if value is not None else default


# ---------------------------------------------------------------------------
# Per-tool extractors
# ---------------------------------------------------------------------------


def _extract_edit(tool_input: dict[str, object], tool_response: str) -> str:
    file_path = _safe_get(tool_input, "file_path")
    new_string = _safe_get(tool_input, "new_string")
    if not new_string:
        return ""
    parts: list[str] = []
    if file_path:
        parts.append(f"File: {file_path}")
    parts.append(new_string)
    return "\n".join(parts)


def _extract_write(tool_input: dict[str, object], tool_response: str) -> str:
    file_path = _safe_get(tool_input, "file_path")
    content = _safe_get(tool_input, "content")
    if not content:
        return ""
    parts: list[str] = []
    if file_path:
        parts.append(f"File: {file_path}")
    parts.append(content)
    return "\n".join(parts)


def _extract_bash(tool_input: dict[str, object], tool_response: str) -> str:
    command = _safe_get(tool_input, "command")
    if not command:
        return ""
    parts = [f"Command: {command}"]
    if tool_response:
        parts.append(f"Output: {tool_response}")
    return "\n".join(parts)


def _extract_notebook_edit(tool_input: dict[str, object], tool_response: str) -> str:
    notebook_path = _safe_get(tool_input, "notebook_path")
    new_source = _safe_get(tool_input, "new_source")
    if not new_source:
        return ""
    parts: list[str] = []
    if notebook_path:
        parts.append(f"Notebook: {notebook_path}")
    parts.append(new_source)
    return "\n".join(parts)


def _extract_mcp_tool(
    tool_input: dict[str, object], tool_response: str, *, tool_name: str = ""
) -> str:
    parts: list[str] = []
    if tool_name:
        parts.append(f"Tool: {tool_name}")
    if tool_response:
        parts.append(f"Response: {tool_response}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatch table  (OCP: extend by adding entries, not modifying logic)
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Callable[[dict[str, object], str], str]] = {
    "Edit": _extract_edit,
    "Write": _extract_write,
    "Bash": _extract_bash,
    "NotebookEdit": _extract_notebook_edit,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_content(tool_name: str, tool_input: dict[str, object], tool_response: str) -> str:
    """Extract memory-worthy text from a tool invocation.

    Uses the dispatch table for known tools; falls back to the MCP extractor
    for ``mcp__*`` prefixed tools.  Returns empty string if nothing useful
    can be extracted.

    Args:
        tool_name: The Claude Code tool name (e.g. ``"Edit"``, ``"Bash"``).
        tool_input: The tool's input arguments dict.
        tool_response: The tool's response as a string.

    Returns:
        Extracted text, or ``""`` if nothing memory-worthy.
    """
    # tool_response is guaranteed to be a string by the entry point
    # (post_tool_use.py normalizes it before constructing HookInput)
    extractor = _EXTRACTORS.get(tool_name)
    if extractor is not None:
        result = extractor(tool_input, tool_response)
    elif tool_name.startswith("mcp__"):
        result = _extract_mcp_tool(tool_input, tool_response, tool_name=tool_name)
    else:
        # Unknown tool — no extraction
        return ""

    return _truncate(result, _SAFETY_CAP)
