"""Embedding service for Spatial Memory MCP Server."""

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from spatial_memory.core.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from spatial_memory.core.errors import ConfigurationError, EmbeddingError

if TYPE_CHECKING:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

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

# Type variable for retry decorator
F = TypeVar("F", bound=Callable[..., Any])


def _mask_api_key(text: str) -> str:
    """Mask API keys in error messages.

    Args:
        text: Error message text that might contain API keys.

    Returns:
        Text with API keys masked.
    """
    # Mask OpenAI keys (sk-...)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '***OPENAI_KEY***', text)
    # Mask generic api_key patterns
    text = re.sub(
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+',
        'api_key=***MASKED***',
        text,
        flags=re.IGNORECASE
    )
    return text


def retry_on_api_error(
    max_attempts: int = 3,
    backoff: float = 1.0,
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> Callable[[F], F]:
    """Retry decorator for transient API errors.

    Args:
        max_attempts: Maximum number of retry attempts.
        backoff: Initial backoff time in seconds (doubles each attempt).
        retryable_status_codes: HTTP status codes that should trigger retry.

    Returns:
        Decorated function with retry logic.
    """
    # Non-retryable auth errors
    non_retryable_codes = (401, 403)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check for OpenAI-specific errors
                    status_code = None
                    if hasattr(e, "status_code"):
                        status_code = e.status_code
                    elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                        status_code = e.response.status_code

                    # Don't retry auth errors
                    if status_code in non_retryable_codes:
                        logger.warning(f"Non-retryable API error (status {status_code}): {e}")
                        raise

                    # Check if we should retry
                    should_retry = (
                        status_code in retryable_status_codes
                        or "rate" in str(e).lower()
                        or "timeout" in str(e).lower()
                        or "connection" in str(e).lower()
                    )

                    if not should_retry or attempt == max_attempts - 1:
                        raise

                    # Retry with exponential backoff
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)

            if last_error:
                raise last_error
            return None

        return wrapper  # type: ignore

    return decorator


class EmbeddingService:
    """Service for generating text embeddings.

    Supports local sentence-transformers models and optional OpenAI API.
    Uses ONNX Runtime by default for 2-3x faster inference.
    Optionally uses a circuit breaker for fault tolerance with external services.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | Any | None = None,
        backend: EmbeddingBackend = "auto",
        circuit_breaker: CircuitBreaker | None = None,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        cache_max_size: int = 1000,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Model name. Use 'openai:model-name' for OpenAI models.
            openai_api_key: OpenAI API key (required for OpenAI models).
                Can be a string or a SecretStr (pydantic).
            backend: Inference backend. 'auto' uses ONNX if available (default),
                'onnx' forces ONNX Runtime, 'pytorch' forces PyTorch.
            circuit_breaker: Optional pre-configured circuit breaker instance.
                If provided, other circuit breaker parameters are ignored.
            circuit_breaker_enabled: Whether to enable circuit breaker for OpenAI calls.
                Defaults to True. Only applies to OpenAI models.
            circuit_breaker_failure_threshold: Number of consecutive failures before
                opening the circuit. Default is 5.
            circuit_breaker_reset_timeout: Seconds to wait before attempting recovery.
                Default is 60.0 seconds.
            cache_max_size: Maximum number of embeddings to cache (LRU eviction).
                Default is 1000. Set to 0 to disable caching.
        """
        self.model_name = model_name
        # Handle both plain strings and SecretStr (pydantic)
        if openai_api_key is not None and hasattr(openai_api_key, 'get_secret_value'):
            self._openai_api_key: str | None = openai_api_key.get_secret_value()
        else:
            self._openai_api_key = openai_api_key
        self._model: SentenceTransformer | None = None
        self._openai_client: OpenAI | None = None
        self._dimensions: int | None = None

        # Determine backend for local models
        self._requested_backend = backend
        self._active_backend: Literal["onnx", "pytorch"] | None = None

        # Embedding cache (LRU with max size)
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_lock = threading.Lock()

        # Determine if using OpenAI
        self.use_openai = model_name.startswith("openai:")
        if self.use_openai:
            self.openai_model = model_name.split(":", 1)[1]
            if not self._openai_api_key:
                raise ConfigurationError(
                    "OpenAI API key required for OpenAI embedding models"
                )

        # Circuit breaker for OpenAI API calls (optional)
        if circuit_breaker is not None:
            self._circuit_breaker: CircuitBreaker | None = circuit_breaker
        elif circuit_breaker_enabled and self.use_openai:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_breaker_failure_threshold,
                reset_timeout=circuit_breaker_reset_timeout,
                name=f"embedding_service_{model_name}",
            )
            logger.info(
                f"Circuit breaker enabled for embedding service "
                f"(threshold={circuit_breaker_failure_threshold}, "
                f"timeout={circuit_breaker_reset_timeout}s)"
            )
        else:
            self._circuit_breaker = None

    def _load_local_model(self) -> None:
        """Load local sentence-transformers model with ONNX or PyTorch backend."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Detect which backend to use
            self._active_backend = _detect_backend(self._requested_backend)

            logger.info(
                f"Loading embedding model: {self.model_name} "
                f"(backend: {self._active_backend})"
            )

            # Load model with appropriate backend
            if self._active_backend == "onnx":
                # Use ONNX Runtime backend for faster inference
                self._model = SentenceTransformer(
                    self.model_name,
                    backend="onnx",
                )
                logger.info(
                    f"Using ONNX Runtime backend (2-3x faster inference)"
                )
            else:
                # Use default PyTorch backend
                self._model = SentenceTransformer(self.model_name)
                logger.info(
                    f"Using PyTorch backend"
                )

            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model with {self._dimensions} dimensions"
            )
        except Exception as e:
            masked_error = _mask_api_key(str(e))
            raise EmbeddingError(f"Failed to load embedding model: {masked_error}") from e

    def _load_openai_client(self) -> None:
        """Load OpenAI client."""
        if self._openai_client is not None:
            return

        try:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self._openai_api_key)
            # Set dimensions based on model
            model_dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dimensions = model_dimensions.get(self.openai_model, 1536)
            logger.info(
                f"Initialized OpenAI client for {self.openai_model} "
                f"({self._dimensions} dimensions)"
            )
        except ImportError:
            raise ConfigurationError(
                "OpenAI package not installed. Run: pip install openai"
            )
        except Exception as e:
            masked_error = _mask_api_key(str(e))
            raise EmbeddingError(f"Failed to initialize OpenAI client: {masked_error}") from e

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content.

        Uses MD5 for speed (not security) - collisions are acceptable for cache.
        """
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is None:
            if self.use_openai:
                self._load_openai_client()
            else:
                self._load_local_model()
        return self._dimensions  # type: ignore

    @property
    def backend(self) -> str:
        """Get the active embedding backend.

        Returns:
            'openai' for OpenAI API, 'onnx' or 'pytorch' for local models.
        """
        if self.use_openai:
            return "openai"
        if self._active_backend is None:
            # Force model load to determine backend
            self._load_local_model()
        return self._active_backend or "pytorch"

    @property
    def circuit_state(self) -> CircuitState | None:
        """Get the current circuit breaker state.

        Returns:
            CircuitState if circuit breaker is enabled, None otherwise.
        """
        if self._circuit_breaker is None:
            return None
        return self._circuit_breaker.state

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Get the circuit breaker instance.

        Returns:
            CircuitBreaker if enabled, None otherwise.
        """
        return self._circuit_breaker

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        cache_key = self._get_cache_key(text)

        # Check cache first
        with self._cache_lock:
            if cache_key in self._embed_cache:
                # Move to end (most recently used) and return copy
                self._embed_cache.move_to_end(cache_key)
                return self._embed_cache[cache_key].copy()

        # Generate embedding (outside lock to allow concurrent generation)
        if self.use_openai:
            embedding = self._embed_openai_with_circuit_breaker([text])[0]
        else:
            embedding = self._embed_local([text])[0]

        # Cache the result (if caching enabled)
        if self._cache_max_size > 0:
            with self._cache_lock:
                # Check if another thread already cached it
                if cache_key not in self._embed_cache:
                    # Evict oldest entries if at capacity
                    while len(self._embed_cache) >= self._cache_max_size:
                        self._embed_cache.popitem(last=False)
                    self._embed_cache[cache_key] = embedding.copy()
                else:
                    # Another thread cached it, move to end
                    self._embed_cache.move_to_end(cache_key)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Uses cache for already-embedded texts and only generates
        embeddings for texts not in cache.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            logger.debug("embed_batch called with empty input, returning empty list")
            return []

        # If caching disabled, generate all embeddings directly
        if self._cache_max_size <= 0:
            if self.use_openai:
                return self._embed_openai_with_circuit_breaker(texts)
            else:
                return self._embed_local(texts)

        # Check cache for each text
        results: list[np.ndarray | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)

        with self._cache_lock:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embed_cache:
                    self._embed_cache.move_to_end(cache_key)
                    results[i] = self._embed_cache[cache_key].copy()
                else:
                    texts_to_embed.append((i, text))

        # Generate embeddings for uncached texts
        if texts_to_embed:
            uncached_texts = [t for _, t in texts_to_embed]
            if self.use_openai:
                new_embeddings = self._embed_openai_with_circuit_breaker(uncached_texts)
            else:
                new_embeddings = self._embed_local(uncached_texts)

            # Store results and cache them
            with self._cache_lock:
                for (idx, text), embedding in zip(texts_to_embed, new_embeddings):
                    results[idx] = embedding
                    cache_key = self._get_cache_key(text)
                    if cache_key not in self._embed_cache:
                        # Evict oldest entries if at capacity
                        while len(self._embed_cache) >= self._cache_max_size:
                            self._embed_cache.popitem(last=False)
                        self._embed_cache[cache_key] = embedding.copy()

        # Type assertion - all results should be filled
        return [r for r in results if r is not None]

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of entries cleared."""
        with self._cache_lock:
            count = len(self._embed_cache)
            self._embed_cache.clear()
            return count

    def _embed_local(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using local model.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self._load_local_model()
        assert self._model is not None  # _load_local_model() sets this or raises

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

    def _embed_openai_with_circuit_breaker(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using OpenAI API with circuit breaker protection.

        Wraps the OpenAI embedding call with a circuit breaker to prevent
        cascading failures when the API is unavailable.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If circuit is open or embedding generation fails.
        """
        if self._circuit_breaker is None:
            # No circuit breaker, call directly
            return self._embed_openai(texts)

        try:
            return self._circuit_breaker.call(self._embed_openai, texts)
        except CircuitOpenError as e:
            logger.warning(
                f"Circuit breaker is open for embedding service, "
                f"time until retry: {e.time_until_retry:.1f}s"
                if e.time_until_retry is not None
                else "Circuit breaker is open for embedding service"
            )
            raise EmbeddingError(
                f"Embedding service temporarily unavailable (circuit open). "
                f"Try again in {e.time_until_retry:.0f} seconds."
                if e.time_until_retry is not None
                else "Embedding service temporarily unavailable (circuit open)."
            ) from e

    @retry_on_api_error(max_attempts=3, backoff=1.0)
    def _embed_openai(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using OpenAI API with retry logic.

        Automatically retries on transient errors (429 rate limit, 5xx server errors).
        Does not retry on auth errors (401, 403).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self._load_openai_client()
        assert self._openai_client is not None  # _load_openai_client() sets this or raises

        try:
            response = self._openai_client.embeddings.create(
                model=self.openai_model,
                input=texts,
            )
            embeddings = []
            for item in response.data:
                emb = np.array(item.embedding, dtype=np.float32)
                # Normalize
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return embeddings
        except Exception as e:
            masked_error = _mask_api_key(str(e))
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {masked_error}") from e
