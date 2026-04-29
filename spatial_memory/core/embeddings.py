"""Embedding service for Spatial Memory MCP Server."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import numpy as np

from spatial_memory.core.embedding_registry import (
    EmbeddingModelConfig,
    resolve_model,
)
from spatial_memory.core.errors import ConfigurationError, EmbeddingError
from spatial_memory.core.metrics import record_embedding_latency

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from spatial_memory.config import Settings

logger = logging.getLogger(__name__)

# Backend type for embedding inference
EmbeddingBackend = Literal["auto", "onnx", "pytorch"]


def _is_onnx_available() -> bool:
    """Check if ONNX Runtime and Optimum are available.

    Sentence-transformers requires both onnxruntime and optimum for ONNX support.
    """
    try:
        import onnxruntime  # noqa: F401
        import optimum.onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _detect_backend(requested: EmbeddingBackend) -> Literal["onnx", "pytorch"]:
    """Detect which backend to use.

    Args:
        requested: The requested backend ('auto', 'onnx', or 'pytorch').

    Returns:
        The actual backend to use ('onnx' or 'pytorch').
    """
    if requested == "pytorch":
        return "pytorch"
    elif requested == "onnx":
        if not _is_onnx_available():
            raise ConfigurationError(
                "ONNX Runtime requested but not fully installed. "
                "Install with: pip install sentence-transformers[onnx]"
            )
        return "onnx"
    else:  # auto
        if _is_onnx_available():
            return "onnx"
        return "pytorch"


def _mask_api_key(text: str) -> str:
    """Mask generic api_key=... patterns in error messages."""
    return re.sub(
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+',
        "api_key=***MASKED***",
        text,
        flags=re.IGNORECASE,
    )


class EmbeddingService:
    """Service for generating text embeddings via local sentence-transformers.

    Under v2.0 (D1 registry-of-one) this is the only embedding path: every
    model_name resolves to an entry in ``embedding_registry.BUILTIN_MODELS``
    and unknown names raise ``ValueError`` at construction. Uses ONNX Runtime
    by default for faster inference, falling back to PyTorch when ONNX isn't
    available or the model demands ``trust_remote_code=True``.
    """

    def __init__(
        self,
        model_name: str = "arctic-m-long",
        backend: EmbeddingBackend = "auto",
        cache_max_size: int = 1000,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Registry alias (e.g. ``arctic-m-long``). Unknown
                aliases raise ``ValueError`` via ``resolve_model``.
            backend: Inference backend. 'auto' uses ONNX if available
                (default), 'onnx' forces ONNX Runtime, 'pytorch' forces
                PyTorch.
            cache_max_size: Maximum number of embeddings to cache (LRU
                eviction). Default is 1000. Set to 0 to disable caching.
        """
        self.model_name = model_name
        # Resolve model config from the v2 registry up front so the rest of
        # the service can read prefix/dim/max_seq_length without repeating
        # the lookup.
        self._config: EmbeddingModelConfig = resolve_model(model_name)
        self._model: SentenceTransformer | None = None
        self._dimensions: int | None = None

        # Determine backend for local models
        self._requested_backend = backend
        self._active_backend: Literal["onnx", "pytorch"] | None = None

        # Embedding cache (LRU with max size)
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_lock = threading.Lock()

    @classmethod
    def from_config(cls, settings: Settings) -> EmbeddingService:
        """Construct from a ``Settings`` instance and run a single warm-up.

        The warm-up call forces the model + tokenizer to load eagerly so
        the first user-facing request doesn't pay the cold-start penalty.
        """
        from typing import cast

        svc = cls(
            model_name=settings.embedding_model,
            backend=cast(EmbeddingBackend, settings.embedding_backend),
        )
        svc.embed_query("warmup")
        return svc

    def _load_local_model(self) -> None:
        """Load local sentence-transformers model with ONNX or PyTorch backend."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Detect which backend to use. Models that require trust_remote_code
            # (e.g. arctic-m-long) are forced to PyTorch — the ONNX path in
            # sentence-transformers does not reliably plumb the flag through
            # to AutoConfig.from_pretrained.
            requested = self._requested_backend
            if self._config.trust_remote_code and requested != "pytorch":
                logger.info(
                    "Forcing PyTorch backend for %s (trust_remote_code=True)",
                    self.model_name,
                )
                requested = "pytorch"
            self._active_backend = _detect_backend(requested)

            logger.info(
                f"Loading embedding model: {self.model_name} (backend: {self._active_backend})"
            )

            # Load model with appropriate backend. Use the resolved HF name
            # so registry aliases like "arctic-m-long" download the correct
            # model from Hugging Face instead of being treated as an HF id.
            hf_id = self._config.hf_name or self.model_name
            trust = self._config.trust_remote_code
            if self._active_backend == "onnx":
                # Use ONNX Runtime backend for faster inference
                self._model = SentenceTransformer(
                    hf_id,
                    backend="onnx",
                    trust_remote_code=trust,
                )
                logger.info("Using ONNX Runtime backend (2-3x faster inference)")
            else:
                # Use default PyTorch backend
                self._model = SentenceTransformer(hf_id, trust_remote_code=trust)
                logger.info("Using PyTorch backend")

            # Honor the registry's max_seq_length (e.g. 8192 for arctic-m-long)
            # so long passages aren't silently truncated to the model default.
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self._config.max_seq_length

            self._dimensions = _read_embedding_dimension(self._model)
            logger.info(f"Loaded model with {self._dimensions} dimensions")
        except Exception as e:
            masked_error = _mask_api_key(str(e))
            raise EmbeddingError(f"Failed to load embedding model: {masked_error}") from e

    def _get_cache_key(self, text: str, role: str = "passage") -> str:
        """Generate role-aware cache key from text content.

        Role is part of the hash so query-prefixed and passage-prefixed
        embeddings of the same text occupy different cache slots. Uses MD5
        for speed (not security) — collisions are acceptable for a cache.
        """
        return hashlib.md5(f"{role}:{text}".encode(), usedforsecurity=False).hexdigest()

    def _apply_prefix(self, text: str, role: str) -> str:
        """Prepend the registry-defined prefix for the given role."""
        if role == "query":
            return f"{self._config.query_prefix}{text}"
        if role == "passage":
            return f"{self._config.passage_prefix}{text}"
        raise ValueError(f"Unknown embedding role: {role!r}")

    @property
    def config(self) -> EmbeddingModelConfig:
        """Return the resolved registry config for this service."""
        return self._config

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is None:
            self._load_local_model()
        return self._dimensions  # type: ignore

    @property
    def backend(self) -> str:
        """Get the active embedding backend ('onnx' or 'pytorch')."""
        if self._active_backend is None:
            self._load_local_model()
        return self._active_backend or "pytorch"

    @property
    def max_seq_length(self) -> int:
        """Registry-defined maximum sequence length for the active model."""
        return self._config.max_seq_length

    def tokenize(self, text: str) -> list[int]:
        """Encode ``text`` to model token IDs (no special tokens added)."""
        self._load_local_model()
        assert self._model is not None
        encoded = self._model.tokenizer.encode(text, add_special_tokens=False)
        return list(encoded)

    def detokenize(self, ids: list[int]) -> str:
        """Decode token IDs back to a string (special tokens skipped)."""
        self._load_local_model()
        assert self._model is not None
        return str(self._model.tokenizer.decode(ids, skip_special_tokens=True))

    def embed(self, text: str) -> np.ndarray:
        """Generate a passage-role embedding (backward-compatible shim)."""
        return self.embed_passage(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate passage-role embeddings (backward-compatible shim)."""
        return self.embed_passage_batch(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Generate a query-role embedding (prepends config.query_prefix)."""
        return self._embed_one(text, role="query")

    def embed_passage(self, text: str) -> np.ndarray:
        """Generate a passage-role embedding (prepends config.passage_prefix)."""
        return self._embed_one(text, role="passage")

    def embed_query_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate query-role embeddings for a batch of texts."""
        return self._embed_many(texts, role="query")

    def embed_passage_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate passage-role embeddings for a batch of texts."""
        return self._embed_many(texts, role="passage")

    def _embed_one(self, text: str, *, role: str) -> np.ndarray:
        prefixed = self._apply_prefix(text, role)
        cache_key = self._get_cache_key(text, role=role)

        with self._cache_lock:
            if cache_key in self._embed_cache:
                self._embed_cache.move_to_end(cache_key)
                cached: np.ndarray = self._embed_cache[cache_key].copy()
                return cached

        embedding = self._embed_local([prefixed])[0]

        if self._cache_max_size > 0:
            with self._cache_lock:
                if cache_key not in self._embed_cache:
                    while len(self._embed_cache) >= self._cache_max_size:
                        self._embed_cache.popitem(last=False)
                    self._embed_cache[cache_key] = embedding.copy()
                else:
                    self._embed_cache.move_to_end(cache_key)

        return embedding

    def _embed_many(self, texts: list[str], *, role: str) -> list[np.ndarray]:
        if not texts:
            logger.debug("embed batch called with empty input, returning empty list")
            return []

        prefixed = [self._apply_prefix(t, role) for t in texts]

        if self._cache_max_size <= 0:
            return self._embed_local(prefixed)

        results: list[np.ndarray | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []  # (index, prefixed_text)

        with self._cache_lock:
            for i, raw in enumerate(texts):
                cache_key = self._get_cache_key(raw, role=role)
                if cache_key in self._embed_cache:
                    self._embed_cache.move_to_end(cache_key)
                    results[i] = self._embed_cache[cache_key].copy()
                else:
                    to_embed.append((i, prefixed[i]))

        if to_embed:
            uncached_texts = [t for _, t in to_embed]
            new_embeddings = self._embed_local(uncached_texts)

            with self._cache_lock:
                for (idx, _prefixed), embedding in zip(to_embed, new_embeddings):
                    results[idx] = embedding
                    cache_key = self._get_cache_key(texts[idx], role=role)
                    if cache_key not in self._embed_cache:
                        while len(self._embed_cache) >= self._cache_max_size:
                            self._embed_cache.popitem(last=False)
                        self._embed_cache[cache_key] = embedding.copy()

        return [r for r in results if r is not None]

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of entries cleared."""
        with self._cache_lock:
            count = len(self._embed_cache)
            self._embed_cache.clear()
            return count

    def _embed_local(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using the local model and record latency."""
        self._load_local_model()
        assert self._model is not None  # _load_local_model() sets this or raises

        start = time.perf_counter()
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [emb for emb in embeddings]
        except Exception as e:
            masked_error = _mask_api_key(str(e))
            raise EmbeddingError(f"Failed to generate embeddings: {masked_error}") from e
        finally:
            record_embedding_latency(time.perf_counter() - start, model=self.model_name)


def _read_embedding_dimension(model: SentenceTransformer) -> int:
    """Read embedding dimension across sentence-transformers versions.

    sentence-transformers 5.x renamed ``get_sentence_embedding_dimension``
    to ``get_embedding_dimension`` and emits a ``FutureWarning`` on the
    legacy name. Prefer the new accessor when present, swallow the warning
    on the fallback so callers running with ``-W error`` don't trip.
    """
    new_accessor = getattr(model, "get_embedding_dimension", None)
    if callable(new_accessor):
        return int(new_accessor())
    legacy = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(legacy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return int(legacy())
    raise EmbeddingError(
        "SentenceTransformer model exposes neither get_embedding_dimension "
        "nor get_sentence_embedding_dimension"
    )
