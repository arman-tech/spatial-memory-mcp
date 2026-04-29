"""Unit tests for spatial_memory.core.embedding_registry (v2.0 — registry-of-one).

Per spec §6 / D1: arctic-m-long is the only built-in alias. All other
names — legacy MiniLM short/long forms, BGE/e5/nomic/arctic-s/xs aliases,
unknown HF ids, OpenAI-prefixed names — must raise ValueError.
"""

from __future__ import annotations

import pytest

from spatial_memory.core.embedding_registry import (
    BUILTIN_MODELS,
    EmbeddingModelConfig,
    list_aliases,
    resolve_model,
)

pytestmark = pytest.mark.unit


def test_arctic_m_long_defaults() -> None:
    cfg = BUILTIN_MODELS["arctic-m-long"]
    assert cfg.alias == "arctic-m-long"
    assert cfg.hf_name == "Snowflake/snowflake-arctic-embed-m-long"
    assert cfg.dim == 768
    assert cfg.query_prefix == "Represent this sentence for searching relevant passages: "
    assert cfg.passage_prefix == ""
    assert cfg.normalize is True
    assert cfg.max_seq_length == 8192
    assert cfg.trust_remote_code is True


def test_builtin_models_is_registry_of_one() -> None:
    """D1: arctic-m-long is the only built-in alias."""
    assert set(BUILTIN_MODELS.keys()) == {"arctic-m-long"}
    assert list_aliases() == ["arctic-m-long"]


def test_resolve_arctic_m_long_returns_config() -> None:
    cfg = resolve_model("arctic-m-long")
    assert isinstance(cfg, EmbeddingModelConfig)
    assert cfg.alias == "arctic-m-long"


def test_resolve_minilm_alias_raises() -> None:
    """Legacy MiniLM short alias must not auto-resolve under D1."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        resolve_model("minilm")


def test_resolve_minilm_legacy_long_form_raises() -> None:
    """Legacy MiniLM HF long form must not auto-resolve under D1/D5."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        resolve_model("all-MiniLM-L6-v2")


def test_resolve_unknown_hf_id_raises() -> None:
    """Pre-D1 BGE alias is gone; passthrough is gone — unknown names must raise."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        resolve_model("BAAI/bge-small-en-v1.5")


def test_resolve_openai_prefixed_name_raises() -> None:
    """D1 drops the OpenAI provider; openai:* names fail like any other unknown alias."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        resolve_model("openai:text-embedding-3-small")


def test_no_legacy_name_map_attribute() -> None:
    """LEGACY_NAME_MAP must not exist (D1/D5 clean break)."""
    import spatial_memory.core.embedding_registry as reg

    assert not hasattr(reg, "LEGACY_NAME_MAP")


def test_config_is_frozen() -> None:
    cfg = BUILTIN_MODELS["arctic-m-long"]
    with pytest.raises(Exception):
        cfg.dim = 999  # type: ignore[misc]
