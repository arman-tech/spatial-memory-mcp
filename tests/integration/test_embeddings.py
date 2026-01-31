"""Tests for embedding service.

These tests intentionally use the real embedding model to verify its behavior.
"""

import numpy as np
import pytest

from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.errors import ConfigurationError

# Mark entire module as integration tests (require real embedding model)
pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_dimensions(self, embedding_service: EmbeddingService) -> None:
        """Test embedding dimensions."""
        assert embedding_service.dimensions == 384

    def test_embed_single(self, embedding_service: EmbeddingService) -> None:
        """Test single text embedding."""
        vec = embedding_service.embed("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        # Check normalization
        assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_embed_batch(self, embedding_service: EmbeddingService) -> None:
        """Test batch text embedding."""
        texts = ["Hello", "World", "Test"]
        vecs = embedding_service.embed_batch(texts)
        assert len(vecs) == 3
        for vec in vecs:
            assert isinstance(vec, np.ndarray)
            assert vec.shape == (384,)
            assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_embed_batch_empty(self, embedding_service: EmbeddingService) -> None:
        """Test empty batch embedding."""
        vecs = embedding_service.embed_batch([])
        assert vecs == []

    def test_similar_texts_have_similar_embeddings(
        self, embedding_service: EmbeddingService
    ) -> None:
        """Test that similar texts produce similar embeddings."""
        vec1 = embedding_service.embed("Python programming language")
        vec2 = embedding_service.embed("Python coding language")
        vec3 = embedding_service.embed("Cooking recipes for dinner")

        # Cosine similarity (vectors are normalized)
        sim_12 = np.dot(vec1, vec2)
        sim_13 = np.dot(vec1, vec3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13

    def test_openai_requires_api_key(self) -> None:
        """Test that OpenAI model requires API key."""
        with pytest.raises(ConfigurationError):
            EmbeddingService("openai:text-embedding-3-small")

    def test_openai_model_name_parsing(self) -> None:
        """Test OpenAI model name parsing."""
        # This won't load the model, just test parsing
        svc = EmbeddingService.__new__(EmbeddingService)
        svc.model_name = "openai:text-embedding-3-small"
        svc.openai_api_key = "fake-key"
        svc._model = None
        svc._openai_client = None
        svc._dimensions = None
        svc.use_openai = True
        svc.openai_model = "text-embedding-3-small"

        assert svc.use_openai is True
        assert svc.openai_model == "text-embedding-3-small"
