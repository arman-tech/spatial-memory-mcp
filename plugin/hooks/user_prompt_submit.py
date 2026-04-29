"""UserPromptSubmit hook — ambient recall on user prompt (US-024 / spec §8.3, D3).

Listens for the ``UserPromptSubmit`` event, runs ``hybrid_recall`` against the
configured namespace, and returns a formatted top-K block printed verbatim by
the dispatcher. Disabled when ``SPATIAL_MEMORY_AMBIENT_RECALL`` is anything
other than ``"on"`` (default ``"on"``).

Design choices
--------------
* **Stdlib-only at import time.** Heavy imports (``ServiceFactory``, models)
  are deferred to the request path so hook startup stays cheap.
* **Fail-open.** The dispatcher already swallows exceptions, but we catch
  here too — an exception in the recall path must yield ``None`` so the
  dispatcher emits no stdout. AC: "recall raises ⇒ handle returns None".
* **Empty payload / off switch / empty prompt** all short-circuit before
  any service is built, so this path adds <1ms to typical prompts.
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_TOP_K = 5
_DEFAULT_RECALL = "on"


def _build_factory() -> Any:
    """Lazy-build a ServiceFactory. Indirection lets tests patch this."""
    from spatial_memory.config import Settings
    from spatial_memory.factory import ServiceFactory

    return ServiceFactory(Settings())


def _format_top_k(memories: list[Any]) -> str:
    """Render the recall block injected into the model context.

    The header tags the block as a memory injection so downstream readers
    (and human reviewers of transcripts) can recognise it. Each memory line
    is truncated at 200 characters to keep the block compact.
    """
    lines = ["[spatial-memory] ambient recall:"]
    for i, mem in enumerate(memories, start=1):
        content = str(mem.content).replace("\n", " ").strip()
        if len(content) > 200:
            content = content[:197] + "..."
        lines.append(f"  {i}. ({mem.similarity:.2f}) {content}")
    return "\n".join(lines)


def handle(payload: dict[str, Any], client: str) -> str | None:
    """Run ambient recall for a user prompt, returning a formatted block or None.

    Args:
        payload: Hook stdin payload. Expected to contain ``prompt`` (str).
        client: Client identifier (``claude-code`` / ``cursor``). Currently
            unused — block format is client-agnostic.

    Returns:
        A formatted top-K recall block, or ``None`` to suppress output.
    """
    if os.getenv("SPATIAL_MEMORY_AMBIENT_RECALL", _DEFAULT_RECALL).strip().lower() != "on":
        return None

    prompt_raw = payload.get("prompt")
    if not isinstance(prompt_raw, str):
        return None
    prompt = prompt_raw.strip()
    if not prompt:
        return None

    try:
        top_k_env = os.getenv("SPATIAL_MEMORY_AMBIENT_TOP_K")
        top_k = int(top_k_env) if top_k_env else _DEFAULT_TOP_K
    except ValueError:
        top_k = _DEFAULT_TOP_K
    if top_k <= 0:
        return None

    namespace = os.getenv("SPATIAL_MEMORY_AMBIENT_NAMESPACE") or None

    try:
        factory = _build_factory()
        services = factory.create_all()
        result = services.utility.hybrid_recall(
            query=prompt,
            limit=top_k,
            namespace=namespace,
        )
    except Exception:
        return None

    memories = getattr(result, "memories", None) or []
    if not memories:
        return None

    return _format_top_k(memories)
