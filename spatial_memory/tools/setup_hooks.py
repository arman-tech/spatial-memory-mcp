"""Generate hook configuration for Claude Code and other AI coding clients.

Uses the **Strategy + Template Method** pattern to produce correct
configuration for each supported client.  The public API is the
backward-compatible ``generate_hook_config()`` facade.

Supported clients (Tier 1–3):

* **Tier 1** (full hooks): Claude Code
* **Tier 2** (hooks via adapter): Cursor
* **Tier 3** (MCP only): Windsurf, Antigravity, VS Code Copilot
"""

from __future__ import annotations

import importlib.resources
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# =============================================================================
# Client capabilities
# =============================================================================


@dataclass(frozen=True)
class ClientCapabilities:
    """Declares what a client supports."""

    hooks: bool
    mcp: bool
    rules: bool
    hook_format: str  # "claude-code", "cursor-native", "none"


# =============================================================================
# Shared helpers
# =============================================================================


def _resolve_hooks_dir() -> str:
    """Resolve the absolute path to the installed hooks directory."""
    ref = importlib.resources.files("spatial_memory") / "hooks"
    return str(ref)


def _resolve_python() -> str:
    """Resolve the current Python interpreter path."""
    return sys.executable


def _build_standard_mcp_config(python: str) -> dict[str, Any]:
    """Build the standard ``mcpServers`` MCP config used by most clients.

    Args:
        python: Path to the Python interpreter.

    Returns:
        Dict with ``mcpServers`` key containing spatial-memory config.
    """
    return {
        "mcpServers": {
            "spatial-memory": {
                "command": python,
                "args": ["-m", "spatial_memory"],
                "env": {
                    "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true",
                },
            }
        }
    }


# =============================================================================
# Strategy ABC (Template Method)
# =============================================================================


class ClientConfigStrategy(ABC):
    """Base class for per-client config generation.

    Subclasses implement the three abstract hooks.  The ``generate()``
    template method orchestrates the overall response shape.
    """

    name: str
    capabilities: ClientCapabilities

    def __init__(self, python: str, hooks_dir: str) -> None:
        self.python = python
        self.hooks_dir = hooks_dir

    def generate(
        self,
        *,
        include_session_start: bool = True,
        include_mcp_config: bool = True,
    ) -> dict[str, Any]:
        """Template method — assembles the full response dict.

        Args:
            include_session_start: Include the SessionStart hook.
            include_mcp_config: Include MCP server configuration.

        Returns:
            Config dict with ``client``, ``hooks``, ``mcp_config``,
            ``instructions``, ``paths``, and ``capabilities``.
        """
        result: dict[str, Any] = {
            "client": self.name,
            "paths": {
                "python": self.python,
                "hooks_dir": self.hooks_dir,
            },
            "capabilities": {
                "hooks": self.capabilities.hooks,
                "mcp": self.capabilities.mcp,
                "rules": self.capabilities.rules,
                "hook_format": self.capabilities.hook_format,
            },
        }

        result["hooks"] = self.build_hooks(include_session_start=include_session_start)

        if include_mcp_config:
            result["mcp_config"] = self.build_mcp_config()
        else:
            result["mcp_config"] = None

        result["instructions"] = self.build_instructions()

        return result

    @abstractmethod
    def build_hooks(self, *, include_session_start: bool = True) -> dict[str, Any] | None:
        """Build hooks config, or None if unsupported."""

    @abstractmethod
    def build_mcp_config(self) -> dict[str, Any] | None:
        """Build MCP server config."""

    @abstractmethod
    def build_instructions(self) -> str:
        """Build human-readable setup instructions."""


# =============================================================================
# Tier 1: Claude Code (full native hooks)
# =============================================================================


class ClaudeCodeStrategy(ClientConfigStrategy):
    name = "claude-code"
    capabilities = ClientCapabilities(
        hooks=True,
        mcp=True,
        rules=False,
        hook_format="claude-code",
    )

    def build_hooks(self, *, include_session_start: bool = True) -> dict[str, Any]:
        hooks: dict[str, list[dict[str, Any]]] = {}

        if include_session_start:
            hooks["SessionStart"] = [
                {
                    "matcher": "startup|resume",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f'"{self.python}" "{self.hooks_dir}/session_start.py"',
                            "timeout": 5,
                        }
                    ],
                }
            ]

        hooks["PostToolUse"] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{self.python}" "{self.hooks_dir}/post_tool_use.py"',
                        "timeout": 10,
                        "async": True,
                    }
                ],
            }
        ]

        hooks["PreCompact"] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{self.python}" "{self.hooks_dir}/pre_compact.py"',
                        "timeout": 15,
                    }
                ],
            }
        ]

        hooks["Stop"] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f'"{self.python}" "{self.hooks_dir}/stop.py"',
                        "timeout": 15,
                    }
                ],
            }
        ]

        return hooks

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python)

    def build_instructions(self) -> str:
        return (
            "Add the 'hooks' config to your .claude/settings.json under the "
            "'hooks' key. Add the 'mcp_config' to your MCP server configuration. "
            "Or install the spatial-memory plugin for zero-config setup:\n\n"
            "  /plugin marketplace add arman-tech/spatial-memory-mcp\n"
            "  /plugin install spatial-memory@spatial-memory-marketplace"
        )


# =============================================================================
# Tier 2: Cursor (hooks via adapter script)
# =============================================================================


class CursorStrategy(ClientConfigStrategy):
    name = "cursor"
    capabilities = ClientCapabilities(
        hooks=True,
        mcp=True,
        rules=True,
        hook_format="cursor-native",
    )

    def build_hooks(self, *, include_session_start: bool = True) -> dict[str, Any]:
        adapter = f"{self.hooks_dir}/cursor_adapter.py"
        hooks: dict[str, list[dict[str, Any]]] = {}

        if include_session_start:
            hooks["sessionStart"] = [
                {
                    "command": f'"{self.python}" "{adapter}" session_start',
                    "timeout": 5,
                    "matcher": "startup|resume",
                }
            ]

        hooks["postToolUse"] = [
            {
                "command": f'"{self.python}" "{adapter}" post_tool_use',
                "timeout": 10,
            }
        ]

        hooks["preCompact"] = [
            {
                "command": f'"{self.python}" "{adapter}" pre_compact',
                "timeout": 15,
            }
        ]

        hooks["stop"] = [
            {
                "command": f'"{self.python}" "{adapter}" stop',
                "timeout": 15,
                "loop_limit": 1,
            }
        ]

        return {"version": 1, "hooks": hooks}

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python)

    def build_instructions(self) -> str:
        return (
            "1. Save the 'hooks' config to .cursor/hooks.json\n"
            "2. Add the 'mcp_config' to .cursor/mcp.json\n\n"
            "The hooks use an adapter script that translates between Cursor's "
            "native format and the spatial-memory hook scripts."
        )


# =============================================================================
# Tier 3: Windsurf (MCP only, rules for guidance)
# =============================================================================


class WindsurfStrategy(ClientConfigStrategy):
    name = "windsurf"
    capabilities = ClientCapabilities(
        hooks=False,
        mcp=True,
        rules=True,
        hook_format="none",
    )

    def build_hooks(self, *, include_session_start: bool = True) -> None:
        return None

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python)

    def build_instructions(self) -> str:
        return (
            "Windsurf does not support hooks in a compatible format.\n\n"
            "1. Add the 'mcp_config' to ~/.codeium/windsurf/mcp_config.json\n"
            "2. Create .windsurf/rules/spatial-memory.md with this content:\n\n"
            "   ---\n"
            "   At the start of each conversation, call `recall` with a brief\n"
            "   summary of the user's task to load relevant memories. When you\n"
            "   make decisions, discover solutions, or fix bugs, call `remember`\n"
            "   to save them for future sessions.\n"
            "   ---\n\n"
            "Cognitive offloading layers 1-2 (server-side) work fully via MCP.\n"
            "Layer 3 (auto-capture) relies on the rule to prompt proactive memory use."
        )


# =============================================================================
# Tier 3: Antigravity / Gemini (MCP only, rules for guidance)
# =============================================================================


class AntigravityStrategy(ClientConfigStrategy):
    name = "antigravity"
    capabilities = ClientCapabilities(
        hooks=False,
        mcp=True,
        rules=True,
        hook_format="none",
    )

    def build_hooks(self, *, include_session_start: bool = True) -> None:
        return None

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python)

    def build_instructions(self) -> str:
        return (
            "Antigravity does not support hooks.\n\n"
            "1. Add the 'mcp_config' to ~/.gemini/antigravity/mcp_config.json\n"
            "2. Add memory guidance to ~/.gemini/GEMINI.md or "
            ".agent/rules/spatial-memory.md:\n\n"
            "   ---\n"
            "   At the start of each conversation, call `recall` with a brief\n"
            "   summary of the user's task to load relevant memories. When you\n"
            "   make decisions, discover solutions, or fix bugs, call `remember`\n"
            "   to save them for future sessions.\n"
            "   ---\n\n"
            "Cognitive offloading layers 1-2 (server-side) work fully via MCP."
        )


# =============================================================================
# Tier 3: VS Code Copilot (MCP only, custom instructions)
# =============================================================================


class VSCodeCopilotStrategy(ClientConfigStrategy):
    name = "vscode-copilot"
    capabilities = ClientCapabilities(
        hooks=False,
        mcp=True,
        rules=True,
        hook_format="none",
    )

    def build_hooks(self, *, include_session_start: bool = True) -> None:
        return None

    def build_mcp_config(self) -> dict[str, Any]:
        return {
            "servers": {
                "spatial-memory": {
                    "type": "stdio",
                    "command": self.python,
                    "args": ["-m", "spatial_memory"],
                    "env": {
                        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true",
                    },
                }
            }
        }

    def build_instructions(self) -> str:
        return (
            "VS Code Copilot does not support hooks.\n\n"
            "1. Add the 'mcp_config' to .vscode/mcp.json\n"
            "2. Create .github/copilot-instructions.md with this content:\n\n"
            "   ---\n"
            "   At the start of each conversation, call `recall` with a brief\n"
            "   summary of the user's task to load relevant memories. When you\n"
            "   make decisions, discover solutions, or fix bugs, call `remember`\n"
            "   to save them for future sessions.\n"
            "   ---\n\n"
            "Cognitive offloading layers 1-2 (server-side) work fully via MCP."
        )


# =============================================================================
# Registry + alias resolution
# =============================================================================

_STRATEGY_REGISTRY: dict[str, type[ClientConfigStrategy]] = {
    "claude-code": ClaudeCodeStrategy,
    "cursor": CursorStrategy,
    "windsurf": WindsurfStrategy,
    "antigravity": AntigravityStrategy,
    "vscode-copilot": VSCodeCopilotStrategy,
}

_CLIENT_ALIASES: dict[str, str] = {
    "claude": "claude-code",
    "claudecode": "claude-code",
    "gemini": "antigravity",
    "copilot": "vscode-copilot",
    "vscode": "vscode-copilot",
    "vs-code": "vscode-copilot",
}

SUPPORTED_CLIENTS: tuple[str, ...] = tuple(_STRATEGY_REGISTRY.keys())


def _resolve_client_name(client: str) -> str:
    """Resolve a client name or alias to a canonical name.

    Args:
        client: Client name or alias.

    Returns:
        Canonical client name.

    Raises:
        ValueError: If the client is unknown.
    """
    normalized = client.lower().strip()
    if normalized in _STRATEGY_REGISTRY:
        return normalized
    if normalized in _CLIENT_ALIASES:
        return _CLIENT_ALIASES[normalized]
    raise ValueError(
        f"Unknown client: {client!r}. Supported clients: {', '.join(SUPPORTED_CLIENTS)}"
    )


# =============================================================================
# Public facade (backward-compatible)
# =============================================================================


def generate_hook_config(
    *,
    client: str = "claude-code",
    python_path: str = "",
    include_session_start: bool = True,
    include_mcp_config: bool = True,
) -> dict[str, Any]:
    """Generate hook configuration for the specified client.

    Args:
        client: Target client name or alias.
        python_path: Python interpreter path. Defaults to ``sys.executable``.
        include_session_start: Include the SessionStart hook.
        include_mcp_config: Include MCP server configuration.

    Returns:
        Dict with ``client``, ``hooks``, ``mcp_config``, ``instructions``,
        ``paths``, and ``capabilities``.

    Raises:
        ValueError: If the client name is unknown.
    """
    canonical = _resolve_client_name(client)
    python = python_path or _resolve_python()
    hooks_dir = _resolve_hooks_dir()

    strategy_cls = _STRATEGY_REGISTRY[canonical]
    strategy = strategy_cls(python=python, hooks_dir=hooks_dir)

    return strategy.generate(
        include_session_start=include_session_start,
        include_mcp_config=include_mcp_config,
    )
