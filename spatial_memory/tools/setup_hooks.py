"""Generate hook configuration for Claude Code and Cursor.

Uses the **Strategy + Template Method** pattern to produce correct
configuration for each supported client.  The public API is the
backward-compatible ``generate_hook_config()`` facade.

Supported clients:

* **Claude Code** (Tier 1): Full native hooks
* **Cursor** (Tier 2): Hooks via ``afterMCPExecution`` + ``stop``
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


# Server command/args for dev and prod modes (mirrors plugin_mode.py)
_DEV_SERVER: dict[str, Any] = {
    "command": "python",
    "args": ["-m", "spatial_memory"],
}

_PROD_SERVER: dict[str, Any] = {
    "command": "uvx",
    "args": ["--from", "spatial-memory-mcp", "spatial-memory", "serve"],
}


def _build_standard_mcp_config(
    python: str, project: str = "", mode: str = "prod"
) -> dict[str, Any]:
    """Build the standard ``mcpServers`` MCP config used by most clients.

    Args:
        python: Path to the Python interpreter.
        project: Project identifier for memory scoping. If empty, omitted.
        mode: ``"prod"`` for uvx (default), ``"dev"`` for local python.

    Returns:
        Dict with ``mcpServers`` key containing spatial-memory config.
    """
    env: dict[str, str] = {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true",
    }
    if project:
        env["SPATIAL_MEMORY_PROJECT"] = project

    if mode == "dev":
        server = {
            "command": python,
            "args": list(_DEV_SERVER["args"]),
        }
    else:
        server = {
            "command": _PROD_SERVER["command"],
            "args": list(_PROD_SERVER["args"]),
        }

    server["env"] = env

    return {"mcpServers": {"spatial-memory": server}}


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

    def __init__(self, python: str, hooks_dir: str, project: str = "", mode: str = "prod") -> None:
        self.python = python
        self.hooks_dir = hooks_dir
        self.project = project
        self.mode = mode

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
                            "command": "spatial-memory hook session-start --client claude-code",
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
                        "command": "spatial-memory hook post-tool-use --client claude-code",
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
                        "command": "spatial-memory hook pre-compact --client claude-code",
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
                        "command": "spatial-memory hook stop --client claude-code",
                        "timeout": 15,
                    }
                ],
            }
        ]

        return hooks

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python, project=self.project, mode=self.mode)

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
        hooks: dict[str, list[dict[str, Any]]] = {}

        # postToolUse: fires after any tool use (MCP + built-in Edit/Write/Bash)
        hooks["postToolUse"] = [
            {
                "command": "spatial-memory hook post-tool-use --client cursor",
                "timeout": 10,
            }
        ]

        # preCompact: captures context before Cursor compacts conversation
        hooks["preCompact"] = [
            {
                "command": "spatial-memory hook pre-compact --client cursor",
                "timeout": 15,
            }
        ]

        # stop: fires when agent finishes response
        hooks["stop"] = [
            {
                "command": "spatial-memory hook stop --client cursor",
                "timeout": 15,
                "loop_limit": 1,
            }
        ]

        return {"version": 1, "hooks": hooks}

    def build_mcp_config(self) -> dict[str, Any]:
        return _build_standard_mcp_config(self.python, project=self.project, mode=self.mode)

    def build_instructions(self) -> str:
        return (
            "1. Save the 'hooks' config to .cursor/hooks.json\n"
            "2. Add the 'mcp_config' to .cursor/mcp.json\n"
            "3. Create .cursor/rules/spatial-memory.mdc with memory instructions\n\n"
            "Or run: spatial-memory init --client cursor"
        )


# =============================================================================
# Registry + alias resolution
# =============================================================================

_STRATEGY_REGISTRY: dict[str, type[ClientConfigStrategy]] = {
    "claude-code": ClaudeCodeStrategy,
    "cursor": CursorStrategy,
}

_CLIENT_ALIASES: dict[str, str] = {
    "claude": "claude-code",
    "claudecode": "claude-code",
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
    project: str = "",
    mode: str = "prod",
) -> dict[str, Any]:
    """Generate hook configuration for the specified client.

    Args:
        client: Target client name or alias.
        python_path: Python interpreter path. Defaults to ``sys.executable``.
        include_session_start: Include the SessionStart hook.
        include_mcp_config: Include MCP server configuration.
        project: Project identifier for memory scoping. If empty, omitted.
        mode: ``"prod"`` for uvx (default), ``"dev"`` for local python.

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
    strategy = strategy_cls(python=python, hooks_dir=hooks_dir, project=project, mode=mode)

    return strategy.generate(
        include_session_start=include_session_start,
        include_mcp_config=include_mcp_config,
    )
