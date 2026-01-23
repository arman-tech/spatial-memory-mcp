"""Embedding service for Spatial Memory MCP Server."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from spatial_memory.core.errors import ConfigurationError, EmbeddingError

if TYPE_CHECKING:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings.

    Supports local sentence-transformers models and optional OpenAI API.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | None = None,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Model name. Use 'openai:model-name' for OpenAI models.
            openai_api_key: OpenAI API key (required for OpenAI models).
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self._model: SentenceTransformer | None = None
        self._openai_client: OpenAI | None = None
        self._dimensions: int | None = None

        # Determine if using OpenAI
        self.use_openai = model_name.startswith("openai:")
        if self.use_openai:
            self.openai_model = model_name.split(":", 1)[1]
            if not openai_api_key:
                raise ConfigurationError(
                    "OpenAI API key required for OpenAI embedding models"
                )

    def _load_local_model(self) -> None:
        """Load local sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded model with {self._dimensions} dimensions"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def _load_openai_client(self) -> None:
        """Load OpenAI client."""
        if self._openai_client is not None:
            return

        try:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.openai_api_key)
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
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}") from e

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is None:
            if self.use_openai:
                self._load_openai_client()
            else:
                self._load_local_model()
        return self._dimensions  # type: ignore

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        if self.use_openai:
            return self._embed_openai([text])[0]
        else:
            return self._embed_local([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            logger.debug("embed_batch called with empty input, returning empty list")
            return []

        if self.use_openai:
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

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
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    def _embed_openai(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using OpenAI API.

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
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {e}") from e
