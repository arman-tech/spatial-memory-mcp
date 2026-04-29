"""Unit tests for spatial_memory.tools.setup_hooks.

Tests cover Claude Code and Cursor client strategies, optional parameters,
CLI invocation, registry/alias resolution, and client capabilities.
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
        assert "UserPromptSubmit" in config["hooks"]

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

    def test_capabilities_report(self) -> None:
        config = generate_hook_config()
        caps = config["capabilities"]
        assert caps["hooks"] is True
        assert caps["mcp"] is True
        assert caps["hook_format"] == "claude-code"

    def test_hooks_use_spatial_memory_hook_commands(self) -> None:
        config = generate_hook_config()
        hooks = config["hooks"]
        ss_cmd = hooks["SessionStart"][0]["hooks"][0]["command"]
        ptu_cmd = hooks["PostToolUse"][0]["hooks"][0]["command"]
        pc_cmd = hooks["PreCompact"][0]["hooks"][0]["command"]
        stop_cmd = hooks["Stop"][0]["hooks"][0]["command"]
        ups_cmd = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert "spatial-memory hook session-start" in ss_cmd
        assert "spatial-memory hook post-tool-use" in ptu_cmd
        assert "spatial-memory hook pre-compact" in pc_cmd
        assert "spatial-memory hook stop" in stop_cmd
        assert "spatial-memory hook user-prompt-submit" in ups_cmd

    def test_hooks_specify_client_claude_code(self) -> None:
        config = generate_hook_config()
        for entries in config["hooks"].values():
            for entry in entries:
                for hook in entry["hooks"]:
                    assert "--client claude-code" in hook["command"]


# =============================================================================
# Cursor client (Tier 2)
# =============================================================================


@pytest.mark.unit
class TestCursorConfig:
    """Test hook config generation for Cursor."""

    def test_cursor_has_hooks(self) -> None:
        config = generate_hook_config(client="cursor")
        assert config["hooks"] is not None
        assert "version" in config["hooks"]
        assert config["hooks"]["version"] == 1

    def test_cursor_uses_post_tool_use_and_stop(self) -> None:
        config = generate_hook_config(client="cursor")
        hooks = config["hooks"]["hooks"]
        assert "postToolUse" in hooks
        assert "stop" in hooks

    def test_cursor_hooks_use_spatial_memory_hook(self) -> None:
        config = generate_hook_config(client="cursor")
        hooks = config["hooks"]["hooks"]
        for event_hooks in hooks.values():
            for entry in event_hooks:
                assert "spatial-memory hook" in entry["command"]
                assert "--client cursor" in entry["command"]

    def test_cursor_stop_has_loop_limit(self) -> None:
        config = generate_hook_config(client="cursor")
        stop = config["hooks"]["hooks"]["stop"][0]
        assert stop["loop_limit"] == 1

    def test_cursor_hooks_flat_structure(self) -> None:
        """Cursor hooks are flat (command, timeout) -- no nested 'hooks' array."""
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

    def test_cursor_instructions_mention_init(self) -> None:
        config = generate_hook_config(client="cursor")
        assert "init" in config["instructions"]

    def test_cursor_instructions_mention_mdc(self) -> None:
        config = generate_hook_config(client="cursor")
        assert ".mdc" in config["instructions"]

    def test_cursor_3_event_types(self) -> None:
        config = generate_hook_config(client="cursor")
        hooks = config["hooks"]["hooks"]
        assert len(hooks) == 3
        assert "postToolUse" in hooks
        assert "preCompact" in hooks
        assert "stop" in hooks


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
        # postToolUse still present (it's the PostToolUse equivalent)
        assert "postToolUse" in config["hooks"]["hooks"]
        assert "stop" in config["hooks"]["hooks"]

    def test_exclude_mcp_config(self) -> None:
        config = generate_hook_config(include_mcp_config=False)
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

    def test_supported_clients_are_claude_code_and_cursor(self) -> None:
        assert set(SUPPORTED_CLIENTS) == {"claude-code", "cursor"}

    def test_alias_claude(self) -> None:
        assert _resolve_client_name("claude") == "claude-code"

    def test_alias_claudecode(self) -> None:
        assert _resolve_client_name("claudecode") == "claude-code"

    def test_unknown_client_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown client"):
            _resolve_client_name("nonexistent")

    def test_unsupported_windsurf_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown client"):
            _resolve_client_name("windsurf")

    def test_unsupported_antigravity_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown client"):
            _resolve_client_name("antigravity")

    def test_unsupported_vscode_copilot_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown client"):
            _resolve_client_name("vscode-copilot")

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


# =============================================================================
# US-025 / spec §8.4: UserPromptSubmit install matrix
# =============================================================================


@pytest.mark.unit
class TestUserPromptSubmitInstall:
    """Verify both install paths register the new UserPromptSubmit hook.

    Spec §8.4 (D3): plugin/hooks/hooks.json and the CLI install matrix
    (ClaudeCodeStrategy) must both fire the UserPromptSubmit event so users
    on either install path get ambient recall.
    """

    def test_claude_code_user_prompt_submit_present(self) -> None:
        config = generate_hook_config()
        assert "UserPromptSubmit" in config["hooks"]
        ups = config["hooks"]["UserPromptSubmit"]
        assert isinstance(ups, list) and len(ups) >= 1

    def test_claude_code_user_prompt_submit_command_invokes_event(self) -> None:
        config = generate_hook_config()
        cmd = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert "user-prompt-submit" in cmd
        assert "--client claude-code" in cmd

    def test_claude_code_user_prompt_submit_type_is_command(self) -> None:
        config = generate_hook_config()
        hook = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        assert hook["type"] == "command"

    def test_claude_code_user_prompt_submit_timeout_is_5(self) -> None:
        """Spec §8.4: timeout=5 (matches Claude Code's UserPromptSubmit envelope)."""
        config = generate_hook_config()
        hook = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        assert hook["timeout"] == 5

    def test_cursor_does_not_register_user_prompt_submit(self) -> None:
        """Cursor's hook surface has no UserPromptSubmit-equivalent in v2.0.

        We deliberately do NOT add it to CursorStrategy. The gap is
        documented in CLAUDE.md (see test_cursor_gap_documented_in_claude_md).
        """
        config = generate_hook_config(client="cursor")
        cursor_hooks = config["hooks"]["hooks"]
        for event_name, entries in cursor_hooks.items():
            for entry in entries:
                cmd = entry.get("command", "")
                assert "user-prompt-submit" not in cmd, (
                    f"Cursor event {event_name!r} unexpectedly registered "
                    f"user-prompt-submit: {cmd!r}"
                )

    def test_plugin_hooks_json_has_user_prompt_submit(self) -> None:
        """Spec §8.4: plugin/hooks/hooks.json must register UserPromptSubmit."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        hooks_json = repo_root / "plugin" / "hooks" / "hooks.json"
        data = json.loads(hooks_json.read_text(encoding="utf-8"))

        assert "UserPromptSubmit" in data["hooks"]
        ups = data["hooks"]["UserPromptSubmit"]
        assert isinstance(ups, list) and len(ups) >= 1

        hook = ups[0]["hooks"][0]
        assert hook["type"] == "command"
        assert "user-prompt-submit" in hook["command"]
        assert hook["timeout"] == 5

    def test_cursor_gap_documented_in_claude_md(self) -> None:
        """AC4: Cursor's missing UserPromptSubmit must be documented somewhere
        the user can find it (CLAUDE.md or README)."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [repo_root / "CLAUDE.md", repo_root / "README.md"]
        text = ""
        for path in candidates:
            if path.exists():
                text += path.read_text(encoding="utf-8").lower()

        assert (
            "ambient recall" in text or "user-prompt-submit" in text or "userpromptsubmit" in text
        )
        assert "claude" in text and "cursor" in text
