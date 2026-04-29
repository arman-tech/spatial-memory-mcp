"""Embedding model registry for Spatial Memory MCP v2.

Single source of truth for built-in embedding aliases. All other code reads
dimension/prefix/normalize/max_seq_length information from this registry.

Per spec D1 / D5: arctic-m-long is the only built-in alias in v2.0. There
is no legacy-name remapping, no passthrough fallback, and no auto-resolution
of arbitrary HF identifiers — every other name raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Frozen configuration for an embedding model alias."""

    alias: str
    hf_name: str
    dim: int
    query_prefix: str
    passage_prefix: str
    normalize: bool
    max_seq_length: int
    description: str
    trust_remote_code: bool = False


_ARCTIC_PREFIX = "Represent this sentence for searching relevant passages: "


BUILTIN_MODELS: dict[str, EmbeddingModelConfig] = {
    "arctic-m-long": EmbeddingModelConfig(
        alias="arctic-m-long",
        hf_name="Snowflake/snowflake-arctic-embed-m-long",
        dim=768,
        query_prefix=_ARCTIC_PREFIX,
        passage_prefix="",
        normalize=True,
        max_seq_length=8192,
        description="Snowflake Arctic Embed M Long — 768-dim, 8K context, asymmetric.",
        trust_remote_code=True,
    ),
}


def resolve_model(name: str) -> EmbeddingModelConfig:
    """Resolve a model alias to its config.

    v2.0 supports a single built-in alias (``arctic-m-long``). Any other name
    — legacy MiniLM forms, HF identifiers, ``openai:*`` strings, etc. — is
    rejected with ``ValueError``. Migration from a v1 database is handled
    via the export/import path; see the v2.0 spec D1/D5.
    """
    if name not in BUILTIN_MODELS:
        raise ValueError(
            f"Unknown embedding model {name!r}. v2.0 supports only: {sorted(BUILTIN_MODELS)}"
        )
    return BUILTIN_MODELS[name]


def list_aliases() -> list[str]:
    """Return the sorted list of built-in alias names."""
    return sorted(BUILTIN_MODELS.keys())
