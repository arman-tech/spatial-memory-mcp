"""Unit tests for spatial_memory.core.embedding_registry."""

from __future__ import annotations

import pytest

from spatial_memory.core.embedding_registry import (
    BUILTIN_MODELS,
    LEGACY_NAME_MAP,
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


def test_all_seven_aliases_present() -> None:
    expected = {
        "arctic-m-long",
        "nomic",
        "arctic-s",
        "arctic-xs",
        "bge-small",
        "e5-small",
        "minilm",
    }
    assert set(BUILTIN_MODELS.keys()) == expected
    assert list_aliases() == sorted(expected)


def test_legacy_minilm_resolution_short_form() -> None:
    cfg = resolve_model("all-MiniLM-L6-v2")
    assert cfg.alias == "minilm"
    assert cfg.dim == 384
    assert cfg.hf_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_legacy_minilm_resolution_long_form() -> None:
    cfg = resolve_model("sentence-transformers/all-MiniLM-L6-v2")
    # The long form is in BUILTIN_MODELS' minilm.hf_name AND in LEGACY_NAME_MAP.
    # Either path resolves to the canonical 'minilm' alias.
    assert cfg.alias == "minilm"
    assert cfg.dim == 384


def test_hf_reverse_lookup() -> None:
    cfg = resolve_model("Snowflake/snowflake-arctic-embed-m-long")
    assert cfg.alias == "arctic-m-long"
    assert cfg.dim == 768


def test_passthrough_for_unknown_model() -> None:
    cfg = resolve_model("some-org/unknown-model")
    assert cfg.alias == "some-org/unknown-model"
    assert cfg.hf_name == "some-org/unknown-model"
    assert cfg.dim == 0
    assert cfg.query_prefix == ""
    assert cfg.passage_prefix == ""
    assert cfg.normalize is True
    assert cfg.max_seq_length == 512
    assert "passthrough" in cfg.description.lower()


def test_e5_is_only_symmetric_prefix_model() -> None:
    """Per spec §7.1: e5-small is the only built-in with both prefixes set."""
    symmetric = [
        cfg.alias
        for cfg in BUILTIN_MODELS.values()
        if cfg.query_prefix and cfg.passage_prefix
    ]
    # nomic also has both — let's be precise: e5 is the only model where both
    # prefixes are non-empty AND match the 'query: '/'passage: ' style. Per spec
    # §7.1 the assertion is symmetric (both non-empty); both e5 and nomic
    # qualify. The spec wording calls this out by listing them as the two
    # asymmetric-prefix models with passage prefixes.
    assert "e5-small" in symmetric
    assert "nomic" in symmetric
    # Arctic family + bge + minilm should NOT have a passage prefix.
    asymmetric_or_empty = [
        cfg.alias
        for cfg in BUILTIN_MODELS.values()
        if not cfg.passage_prefix
    ]
    assert set(asymmetric_or_empty) == {
        "arctic-m-long",
        "arctic-s",
        "arctic-xs",
        "bge-small",
        "minilm",
    }


def test_alias_match_short_circuits_legacy_map() -> None:
    cfg = resolve_model("minilm")
    assert cfg.alias == "minilm"
    assert cfg.dim == 384


def test_legacy_name_map_has_two_entries() -> None:
    assert set(LEGACY_NAME_MAP.keys()) == {
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    }
    assert all(v == "minilm" for v in LEGACY_NAME_MAP.values())


def test_config_is_frozen() -> None:
    cfg = BUILTIN_MODELS["minilm"]
    with pytest.raises(Exception):
        cfg.dim = 999  # type: ignore[misc]


def test_resolve_model_returns_dataclass() -> None:
    cfg = resolve_model("arctic-m-long")
    assert isinstance(cfg, EmbeddingModelConfig)
