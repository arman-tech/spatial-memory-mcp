"""Tests for the session-scoped shared_embedding_service fixture (US-011 / R14).

Per spec §13 R14: every gate that constructs ``EmbeddingService.from_config(Settings())``
triggers a fresh ~768 MB arctic-m-long download on cold-cache CI runners. The
mitigation is a session-scoped pytest fixture in ``tests/conftest.py`` that
loads the v2 default once and lets every consumer reuse the instance.

These tests assert the wiring directly — they don't substitute for the
in-suite uses by other integration tests, but they prove the fixture is
defined, the session scope is honored, and the resolved instance honors the
production-default registry config (768-dim).
"""

from __future__ import annotations

import pytest

from spatial_memory.core.embeddings import EmbeddingService

pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


def test_shared_embedding_service_is_embedding_service(
    shared_embedding_service: EmbeddingService,
) -> None:
    """Fixture yields a real EmbeddingService instance."""
    assert isinstance(shared_embedding_service, EmbeddingService)


def test_shared_embedding_service_uses_v2_default_dim(
    shared_embedding_service: EmbeddingService,
) -> None:
    """Fixture honors the production registry default (768 for arctic-m-long)."""
    assert shared_embedding_service.dimensions == shared_embedding_service.config.dim
    assert shared_embedding_service.dimensions == 768


def test_shared_embedding_service_embeds_query(
    shared_embedding_service: EmbeddingService,
) -> None:
    """A round-trip embed_query call returns a vector at the registry's dim."""
    vec = shared_embedding_service.embed_query("hello world")
    assert vec.shape == (shared_embedding_service.dimensions,)


def test_shared_embedding_service_is_session_scoped(
    shared_embedding_service: EmbeddingService,
    session_embedding_service: EmbeddingService,
) -> None:
    """session_embedding_service must delegate to shared_embedding_service.

    Both fixture names should yield the same instance so existing consumers of
    session_embedding_service transitively reuse the shared session-scoped
    instance instead of triggering a second model load.
    """
    assert shared_embedding_service is session_embedding_service
