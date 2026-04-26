"""Embedding model registry for Spatial Memory MCP.

Single source of truth for built-in embedding aliases. All other code reads
dimension/prefix/normalize/max_seq_length information from this registry.
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
    # True when the HF repo ships custom modeling code that must be
    # executed at load time (sentence-transformers requires the explicit
    # trust_remote_code=True). Snowflake arctic-m-long and Nomic v1.5
    # both fall in this bucket; standard BERT-derived models do not.
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
    "nomic": EmbeddingModelConfig(
        alias="nomic",
        hf_name="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        normalize=True,
        max_seq_length=8192,
        description="Nomic Embed Text v1.5 — 768-dim, 8K context, asymmetric prefixes.",
        trust_remote_code=True,
    ),
    "arctic-s": EmbeddingModelConfig(
        alias="arctic-s",
        hf_name="Snowflake/snowflake-arctic-embed-s",
        dim=384,
        query_prefix=_ARCTIC_PREFIX,
        passage_prefix="",
        normalize=True,
        max_seq_length=512,
        description="Snowflake Arctic Embed S — 384-dim, 512 context.",
    ),
    "arctic-xs": EmbeddingModelConfig(
        alias="arctic-xs",
        hf_name="Snowflake/snowflake-arctic-embed-xs",
        dim=384,
        query_prefix=_ARCTIC_PREFIX,
        passage_prefix="",
        normalize=True,
        max_seq_length=512,
        description="Snowflake Arctic Embed XS — 384-dim, 512 context, smallest.",
    ),
    "bge-small": EmbeddingModelConfig(
        alias="bge-small",
        hf_name="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix=_ARCTIC_PREFIX,
        passage_prefix="",
        normalize=True,
        max_seq_length=512,
        description="BAAI BGE Small EN v1.5 — 384-dim, 512 context.",
    ),
    "e5-small": EmbeddingModelConfig(
        alias="e5-small",
        hf_name="intfloat/e5-small-v2",
        dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalize=True,
        max_seq_length=512,
        description="intfloat E5 Small v2 — 384-dim, 512 context, symmetric prefixes.",
    ),
    "minilm": EmbeddingModelConfig(
        alias="minilm",
        hf_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        query_prefix="",
        passage_prefix="",
        normalize=True,
        max_seq_length=256,
        description="all-MiniLM-L6-v2 — 384-dim, 256 context, no prefixes (legacy v1 default).",
    ),
}


LEGACY_NAME_MAP: dict[str, str] = {
    "all-MiniLM-L6-v2": "minilm",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
}


def resolve_model(name: str) -> EmbeddingModelConfig:
    """Resolve a model name (alias, legacy name, HF id, or unknown) to a config.

    Resolution order:
      1. Exact alias match in BUILTIN_MODELS
      2. Legacy name map -> alias
      3. Reverse lookup by HF name
      4. Passthrough config (dim=0, no prefixes, normalize=True, max_seq_length=512)
    """
    if name in BUILTIN_MODELS:
        return BUILTIN_MODELS[name]

    if name in LEGACY_NAME_MAP:
        return BUILTIN_MODELS[LEGACY_NAME_MAP[name]]

    for config in BUILTIN_MODELS.values():
        if config.hf_name == name:
            return config

    return EmbeddingModelConfig(
        alias=name,
        hf_name=name,
        dim=0,
        query_prefix="",
        passage_prefix="",
        normalize=True,
        max_seq_length=512,
        description="Passthrough — unknown model, dim auto-detected.",
    )


def list_aliases() -> list[str]:
    """Return the sorted list of built-in alias names."""
    return sorted(BUILTIN_MODELS.keys())
