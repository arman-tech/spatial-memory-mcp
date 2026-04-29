"""Unit tests for spatial_memory.hooks.user_prompt_submit (US-024 / spec §8.3).

The hook module exposes ``handle(payload, client) -> str | None``. Behaviour
is gated by ``SPATIAL_MEMORY_AMBIENT_RECALL`` (default ``"on"``) and is
fail-open: any exception in the recall path becomes ``None``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _payload(prompt: str = "How do I configure the embedding model?") -> dict[str, Any]:
    return {
        "session_id": "abc123",
        "cwd": "/tmp/proj",
        "prompt": prompt,
    }


def _import_module() -> Any:
    from spatial_memory.hooks import user_prompt_submit

    return user_prompt_submit


def test_module_exports_handle_callable() -> None:
    """AC: Module exports ``handle(payload, client)`` callable."""
    mod = _import_module()
    assert hasattr(mod, "handle")
    assert callable(mod.handle)


def test_ambient_recall_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC: SPATIAL_MEMORY_AMBIENT_RECALL=off ⇒ handle returns None."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "off")
    assert mod.handle(_payload(), "claude-code") is None


def test_empty_prompt_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC: empty prompt ⇒ handle returns None (no point recalling)."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")
    assert mod.handle(_payload(prompt=""), "claude-code") is None
    assert mod.handle(_payload(prompt="   "), "claude-code") is None


def test_missing_prompt_field_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive: payload without prompt key ⇒ handle returns None."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")
    payload: dict[str, Any] = {"session_id": "abc123", "cwd": "/tmp/proj"}
    assert mod.handle(payload, "claude-code") is None


def test_recall_raises_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC: recall raises ⇒ handle returns None (fail-open at module level)."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")

    fake_factory = MagicMock()
    fake_factory.create_all.side_effect = RuntimeError("DB exploded")

    with patch.object(mod, "_build_factory", return_value=fake_factory):
        result = mod.handle(_payload(), "claude-code")

    assert result is None


def test_recall_returns_formatted_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC: successful recall returns a non-empty formatted top-K block."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K", "3")

    from datetime import datetime, timezone

    from spatial_memory.core.models import HybridMemoryMatch, HybridRecallResult

    matches = [
        HybridMemoryMatch(
            id=f"id-{i}",
            content=f"memory content {i}",
            similarity=0.9 - 0.1 * i,
            namespace="default",
            tags=[],
            importance=0.5,
            created_at=datetime.now(timezone.utc),
            metadata={},
            combined_score=0.9 - 0.1 * i,
        )
        for i in range(3)
    ]
    recall_result = HybridRecallResult(
        query="How do I configure the embedding model?",
        alpha=0.5,
        memories=matches,
        total=3,
        search_type="hybrid",
    )

    fake_utility = MagicMock()
    fake_utility.hybrid_recall.return_value = recall_result

    fake_services = MagicMock()
    fake_services.utility = fake_utility

    fake_factory = MagicMock()
    fake_factory.create_all.return_value = fake_services

    with patch.object(mod, "_build_factory", return_value=fake_factory):
        out = mod.handle(_payload(), "claude-code")

    assert out is not None
    assert isinstance(out, str)
    # Spec §8.3: top-K block format includes content
    for i in range(3):
        assert f"memory content {i}" in out
    # Recall request used the configured top_k
    args, kwargs = fake_utility.hybrid_recall.call_args
    assert kwargs.get("limit") == 3 or (len(args) >= 2 and args[1] == 3)


def test_recall_empty_results_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No memories matched ⇒ no ambient block (None)."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")

    from spatial_memory.core.models import HybridRecallResult

    recall_result = HybridRecallResult(
        query="x",
        alpha=0.5,
        memories=[],
        total=0,
        search_type="hybrid",
    )

    fake_utility = MagicMock()
    fake_utility.hybrid_recall.return_value = recall_result
    fake_services = MagicMock()
    fake_services.utility = fake_utility
    fake_factory = MagicMock()
    fake_factory.create_all.return_value = fake_services

    with patch.object(mod, "_build_factory", return_value=fake_factory):
        out = mod.handle(_payload(), "claude-code")

    assert out is None


def test_hook_env_names_round_trip_with_settings_fields() -> None:
    """R15 guard (spec §13:1151): every ``SPATIAL_MEMORY_AMBIENT_RECALL*`` env
    name read by the hook must match ``Settings.env_prefix + field.upper()``
    for a field that actually exists on ``Settings``.

    Catches the failure mode where a future rename of either the hook env
    name or the Settings field silently desyncs the operator-facing surface
    (operators set the documented env var, hook reads a stale one).
    """
    import ast
    import inspect

    from spatial_memory.config import Settings
    from spatial_memory.hooks import user_prompt_submit as hook

    src = inspect.getsource(hook)
    env_names: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "getenv"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            name = node.args[0].value
            if name.startswith("SPATIAL_MEMORY_AMBIENT_RECALL"):
                env_names.add(name)

    assert env_names, "hook should read at least one SPATIAL_MEMORY_AMBIENT_RECALL* env"

    prefix = Settings.model_config["env_prefix"]
    settings_fields = set(Settings.model_fields.keys())

    for env_name in env_names:
        assert env_name.startswith(prefix), (
            f"hook env var {env_name!r} does not start with Settings.env_prefix={prefix!r}"
        )
        field_name = env_name[len(prefix) :].lower()
        assert field_name in settings_fields, (
            f"hook reads {env_name!r} which would auto-derive from a "
            f"Settings field named {field_name!r}, but no such field exists. "
            f"Either rename the hook env var or add the Settings field."
        )


def test_namespace_env_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE forwarded to hybrid_recall(namespace=...)."""
    mod = _import_module()
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE", "work")

    from spatial_memory.core.models import HybridRecallResult

    recall_result = HybridRecallResult(
        query="x", alpha=0.5, memories=[], total=0, search_type="hybrid"
    )

    fake_utility = MagicMock()
    fake_utility.hybrid_recall.return_value = recall_result
    fake_services = MagicMock()
    fake_services.utility = fake_utility
    fake_factory = MagicMock()
    fake_factory.create_all.return_value = fake_services

    with patch.object(mod, "_build_factory", return_value=fake_factory):
        mod.handle(_payload(), "claude-code")

    _, kwargs = fake_utility.hybrid_recall.call_args
    assert kwargs.get("namespace") == "work"
