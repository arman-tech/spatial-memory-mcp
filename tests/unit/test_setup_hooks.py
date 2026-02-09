"""Unit tests for spatial_memory.tools.setup_hooks.

Tests cover all 5 client strategies, optional parameters, CLI invocation,
registry/alias resolution, and client capabilities.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from spatial_memory.tools.setup_hooks import (
    _STRATEGY_REGISTRY,
    SUPPORTED_CLIENTS,
    ClaudeCodeStrategy,
    ClientCapabilities,
    CursorStrategy,
    _resolve_client_name,
    generate_hook_config,
)

# =============================================================================
# Claude Code client (Tier 1)
# =============================================================================


@pytest.mark.unit
class TestClaudeCodeConfig:
    """Test hook config generation for Claude Code."""

    def test_default_produces_all_hooks(self) -> None:
        config = generate_hook_config()
        assert config["client"] == "claude-code"
        assert "SessionStart" in config["hooks"]
        assert "PostToolUse" in config["hooks"]
        assert "PreCompact" in config["hooks"]
        assert "Stop" in config["hooks"]

    def test_hooks_have_command_type(self) -> None:
        config = generate_hook_config()
        for entries in config["hooks"].values():
            for entry in entries:
                for hook in entry["hooks"]:
                    assert hook["type"] == "command"

    def test_post_tool_use_is_async(self) -> None:
        config = generate_hook_config()
        ptu_hooks = config["hooks"]["PostToolUse"]
        assert ptu_hooks[0]["hooks"][0]["async"] is True

    def test_session_start_has_matcher(self) -> None:
        config = generate_hook_config()
        ss = config["hooks"]["SessionStart"]
        assert ss[0]["matcher"] == "startup|resume"

    def test_mcp_config_uses_mcp_servers_key(self) -> None:
        config = generate_hook_config()
        assert "mcpServers" in config["mcp_config"]
        assert "spatial-memory" in config["mcp_config"]["mcpServers"]

    def test_mcp_config_has_env(self) -> None:
        config = generate_hook_config()
        env = config["mcp_config"]["mcpServers"]["spatial-memory"]["env"]
        assert env["SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED"] == "true"

    def test_paths_included(self) -> None:
        config = generate_hook_config()
        assert "python" in config["paths"]
        assert "hooks_dir" in config["paths"]

    def test_instructions_mention_plugin(self) -> None:
        config = generate_hook_config()
        assert "plugin" in config["instructions"].lower()

    def test_python_path_defaults_to_sys_executable(self) -> None:
        config = generate_hook_config()
        assert config["paths"]["python"] == sys.executable

    def test_custom_python_path(self) -> None:
        config = generate_hook_config(python_path="/usr/bin/python3.12")
        assert config["paths"]["python"] == "/usr/bin/python3.12"
        for entries in config["hooks"].values():
            for entry in entries:
                for hook in entry["hooks"]:
                    assert "/usr/bin/python3.12" in hook["command"]

    def test_capabilities_report(self) -> None:
        config = generate_hook_config()
        caps = config["capabilities"]
        assert caps["hooks"] is True
        assert caps["mcp"] is True
        assert caps["hook_format"] == "claude-code"

    def test_hooks_contain_correct_scripts(self) -> None:
        config = generate_hook_config()
        hooks = config["hooks"]
        assert "session_start.py" in hooks["SessionStart"][0]["hooks"][0]["command"]
        assert "post_tool_use.py" in hooks["PostToolUse"][0]["hooks"][0]["command"]
        assert "pre_compact.py" in hooks["PreCompact"][0]["hooks"][0]["command"]
        assert "stop.py" in hooks["Stop"][0]["hooks"][0]["command"]


# =============================================================================
# Cursor client (Tier 2)
# =============================================================================


@pytest.mark.unit
class TestCursorConfig:
    """Test hook config generation for Cursor (native format via adapter)."""

    def test_cursor_has_hooks(self) -> None:
        config = generate_hook_config(client="cursor")
        assert config["hooks"] is not None
        assert "version" in config["hooks"]
        assert config["hooks"]["version"] == 1

    def test_cursor_uses_camel_case_events(self) -> None:
        config = generate_hook_config(client="cursor")
        hooks = config["hooks"]["hooks"]
        assert "sessionStart" in hooks
        assert "postToolUse" in hooks
        assert "preCompact" in hooks
        assert "stop" in hooks

    def test_cursor_hooks_call_adapter(self) -> None:
        config = generate_hook_config(client="cursor")
        hooks = config["hooks"]["hooks"]
        for event_hooks in hooks.values():
            for entry in event_hooks:
                assert "cursor_adapter.py" in entry["command"]

    def test_cursor_stop_has_loop_limit(self) -> None:
        config = generate_hook_config(client="cursor")
        stop = config["hooks"]["hooks"]["stop"][0]
        assert stop["loop_limit"] == 1

    def test_cursor_hooks_flat_structure(self) -> None:
        """Cursor hooks are flat (command, timeout) — no nested 'hooks' array."""
        config = generate_hook_config(client="cursor")
        for event_hooks in config["hooks"]["hooks"].values():
            for entry in event_hooks:
                assert "command" in entry
                assert "timeout" in entry
                assert "hooks" not in entry  # flat, not nested

    def test_cursor_mcp_uses_standard_format(self) -> None:
        config = generate_hook_config(client="cursor")
        assert "mcpServers" in config["mcp_config"]

    def test_cursor_capabilities(self) -> None:
        config = generate_hook_config(client="cursor")
        caps = config["capabilities"]
        assert caps["hooks"] is True
        assert caps["hook_format"] == "cursor-native"

    def test_cursor_instructions_mention_hooks_json(self) -> None:
        config = generate_hook_config(client="cursor")
        assert "hooks.json" in config["instructions"]


# =============================================================================
# Windsurf client (Tier 3)
# =============================================================================


@pytest.mark.unit
class TestWindsurfConfig:
    """Test hook config for Windsurf (MCP only + rules)."""

    def test_windsurf_no_hooks(self) -> None:
        config = generate_hook_config(client="windsurf")
        assert config["hooks"] is None

    def test_windsurf_has_mcp_config(self) -> None:
        config = generate_hook_config(client="windsurf")
        assert "mcpServers" in config["mcp_config"]

    def test_windsurf_instructions_mention_rules(self) -> None:
        config = generate_hook_config(client="windsurf")
        assert "rules" in config["instructions"].lower()

    def test_windsurf_instructions_mention_config_path(self) -> None:
        config = generate_hook_config(client="windsurf")
        assert "codeium" in config["instructions"].lower()

    def test_windsurf_capabilities(self) -> None:
        config = generate_hook_config(client="windsurf")
        caps = config["capabilities"]
        assert caps["hooks"] is False
        assert caps["mcp"] is True
        assert caps["rules"] is True
        assert caps["hook_format"] == "none"

    def test_windsurf_instructions_mention_layers(self) -> None:
        config = generate_hook_config(client="windsurf")
        assert "layer" in config["instructions"].lower()


# =============================================================================
# Antigravity / Gemini client (Tier 3)
# =============================================================================


@pytest.mark.unit
class TestAntigravityConfig:
    """Test hook config for Antigravity (MCP only + rules)."""

    def test_antigravity_no_hooks(self) -> None:
        config = generate_hook_config(client="antigravity")
        assert config["hooks"] is None

    def test_antigravity_has_mcp_config(self) -> None:
        config = generate_hook_config(client="antigravity")
        assert "mcpServers" in config["mcp_config"]

    def test_antigravity_instructions_mention_gemini(self) -> None:
        config = generate_hook_config(client="antigravity")
        assert "gemini" in config["instructions"].lower()

    def test_antigravity_capabilities(self) -> None:
        config = generate_hook_config(client="antigravity")
        caps = config["capabilities"]
        assert caps["hooks"] is False
        assert caps["mcp"] is True
        assert caps["rules"] is True

    def test_antigravity_accessible_via_gemini_alias(self) -> None:
        config = generate_hook_config(client="gemini")
        assert config["client"] == "antigravity"

    def test_antigravity_instructions_mention_agent_rules(self) -> None:
        config = generate_hook_config(client="antigravity")
        assert ".agent/rules" in config["instructions"]


# =============================================================================
# VS Code Copilot client (Tier 3)
# =============================================================================


@pytest.mark.unit
class TestVSCodeCopilotConfig:
    """Test hook config for VS Code Copilot (MCP only + instructions)."""

    def test_vscode_copilot_no_hooks(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        assert config["hooks"] is None

    def test_vscode_copilot_uses_servers_key(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        assert "servers" in config["mcp_config"]
        assert "mcpServers" not in config["mcp_config"]

    def test_vscode_copilot_has_stdio_type(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        server = config["mcp_config"]["servers"]["spatial-memory"]
        assert server["type"] == "stdio"

    def test_vscode_copilot_instructions_mention_copilot_instructions_md(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        assert "copilot-instructions.md" in config["instructions"]

    def test_vscode_copilot_instructions_mention_vscode_mcp(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        assert ".vscode/mcp.json" in config["instructions"]

    def test_vscode_copilot_capabilities(self) -> None:
        config = generate_hook_config(client="vscode-copilot")
        caps = config["capabilities"]
        assert caps["hooks"] is False
        assert caps["mcp"] is True
        assert caps["hook_format"] == "none"

    def test_vscode_copilot_accessible_via_copilot_alias(self) -> None:
        config = generate_hook_config(client="copilot")
        assert config["client"] == "vscode-copilot"


# =============================================================================
# Optional parameters
# =============================================================================


@pytest.mark.unit
class TestOptionalParams:
    """Test optional parameter behavior."""

    def test_exclude_session_start_claude_code(self) -> None:
        config = generate_hook_config(include_session_start=False)
        assert "SessionStart" not in config["hooks"]
        assert "PostToolUse" in config["hooks"]

    def test_exclude_session_start_cursor(self) -> None:
        config = generate_hook_config(client="cursor", include_session_start=False)
        assert "sessionStart" not in config["hooks"]["hooks"]
        assert "postToolUse" in config["hooks"]["hooks"]

    def test_exclude_mcp_config(self) -> None:
        config = generate_hook_config(include_mcp_config=False)
        assert config["mcp_config"] is None

    def test_exclude_mcp_config_tier3(self) -> None:
        config = generate_hook_config(client="windsurf", include_mcp_config=False)
        assert config["mcp_config"] is None


# =============================================================================
# CLI subcommand
# =============================================================================


@pytest.mark.unit
class TestCLI:
    """Test the setup-hooks CLI subcommand."""

    def test_json_output_claude_code(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "claude-code"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        assert data["client"] == "claude-code"
        assert "capabilities" in data

    def test_json_output_cursor(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "cursor"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        data = json.loads(mock_print.call_args[0][0])
        assert data["hooks"] is not None
        assert data["hooks"]["version"] == 1

    def test_json_output_windsurf(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "windsurf"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        data = json.loads(mock_print.call_args[0][0])
        assert data["hooks"] is None

    def test_json_output_antigravity(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "antigravity"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        data = json.loads(mock_print.call_args[0][0])
        assert data["client"] == "antigravity"

    def test_json_output_vscode_copilot(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "vscode-copilot"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        data = json.loads(mock_print.call_args[0][0])
        assert "servers" in data["mcp_config"]

    def test_human_readable_output(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "claude-code"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = False

        with patch("builtins.print") as mock_print:
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 0
        assert mock_print.call_count > 3

    def test_unknown_client_returns_error(self) -> None:
        from spatial_memory.__main__ import run_setup_hooks

        class Args:
            client = "unknown-client"
            python_path = ""
            no_session_start = False
            no_mcp_config = False
            json = True

        with patch("builtins.print"):
            result = run_setup_hooks(Args())  # type: ignore[arg-type]

        assert result == 1


# =============================================================================
# Registry + alias resolution
# =============================================================================


@pytest.mark.unit
class TestRegistry:
    """Test client registry and alias resolution."""

    def test_all_supported_clients_in_registry(self) -> None:
        for client in SUPPORTED_CLIENTS:
            assert client in _STRATEGY_REGISTRY

    def test_alias_claude(self) -> None:
        assert _resolve_client_name("claude") == "claude-code"

    def test_alias_claudecode(self) -> None:
        assert _resolve_client_name("claudecode") == "claude-code"

    def test_alias_gemini(self) -> None:
        assert _resolve_client_name("gemini") == "antigravity"

    def test_alias_copilot(self) -> None:
        assert _resolve_client_name("copilot") == "vscode-copilot"

    def test_alias_vscode(self) -> None:
        assert _resolve_client_name("vscode") == "vscode-copilot"

    def test_alias_vs_code(self) -> None:
        assert _resolve_client_name("vs-code") == "vscode-copilot"

    def test_unknown_client_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown client"):
            _resolve_client_name("nonexistent")

    def test_case_insensitive(self) -> None:
        assert _resolve_client_name("Claude-Code") == "claude-code"
        assert _resolve_client_name("CURSOR") == "cursor"


# =============================================================================
# Client capabilities
# =============================================================================


@pytest.mark.unit
class TestClientCapabilities:
    """Test ClientCapabilities dataclass and strategy consistency."""

    def test_capabilities_frozen(self) -> None:
        caps = ClientCapabilities(hooks=True, mcp=True, rules=False, hook_format="claude-code")
        with pytest.raises(AttributeError):
            caps.hooks = False  # type: ignore[misc]

    def test_every_strategy_has_capabilities(self) -> None:
        for name, cls in _STRATEGY_REGISTRY.items():
            assert hasattr(cls, "capabilities"), f"{name} missing capabilities"
            assert isinstance(cls.capabilities, ClientCapabilities)

    def test_every_strategy_has_name(self) -> None:
        for canonical, cls in _STRATEGY_REGISTRY.items():
            assert cls.name == canonical

    def test_tier1_has_hooks(self) -> None:
        assert ClaudeCodeStrategy.capabilities.hooks is True

    def test_tier2_has_hooks(self) -> None:
        assert CursorStrategy.capabilities.hooks is True

    def test_tier3_no_hooks(self) -> None:
        for name in ("windsurf", "antigravity", "vscode-copilot"):
            cls = _STRATEGY_REGISTRY[name]
            assert cls.capabilities.hooks is False, f"{name} should not have hooks"
