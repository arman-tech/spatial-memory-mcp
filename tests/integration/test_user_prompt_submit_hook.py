"""Integration test for the UserPromptSubmit ambient-recall hook (US-024 / spec §8.3).

Builds a real ``ServiceFactory`` against a tmp DB, seeds a small corpus,
then exercises ``spatial_memory.hooks.user_prompt_submit.handle`` end-to-end
to confirm a non-empty formatted top-K block is returned.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.factory import ServiceFactory

pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


@pytest.fixture(autouse=True)
def _reset_after() -> Generator[None, None, None]:
    yield
    reset_settings()


def test_handle_returns_formatted_top_k_against_seeded_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC9: real recall against a seeded namespace returns formatted top-K output."""
    settings = Settings(
        memory_path=tmp_path / "db",
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )
    override_settings(settings)

    factory = ServiceFactory(settings)
    container = factory.create_all()

    seeds = [
        "Configure the embedding model via SPATIAL_MEMORY_EMBEDDING_MODEL env var.",
        "Use the arctic-m-long alias for v2.0 — it's the only registered model.",
        "Hybrid recall combines vector similarity and BM25 keyword search.",
        "Lance DB stores vectors in Parquet fragments under the memory path.",
        "Reset settings between tests with reset_settings() to avoid singleton leak.",
    ]
    for content in seeds:
        container.memory.remember(content=content, namespace="default", importance=0.5)

    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_RECALL", "on")
    monkeypatch.setenv("SPATIAL_MEMORY_AMBIENT_TOP_K", "3")

    from spatial_memory.hooks import user_prompt_submit

    # Indirect the factory build so we get the same in-process container
    # rather than constructing a second one (which would mismatch tmp paths).
    monkeypatch.setattr(user_prompt_submit, "_build_factory", lambda: factory)

    out = user_prompt_submit.handle(
        {"prompt": "How do I configure which embedding model is used?"},
        client="claude-code",
    )

    assert out is not None
    assert isinstance(out, str)
    assert "[spatial-memory] ambient recall" in out
    # At least one of the seeded contents should appear in the top-K block
    assert any(seed[:30] in out for seed in seeds)
