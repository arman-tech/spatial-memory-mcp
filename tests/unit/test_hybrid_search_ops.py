"""Tests for hybrid search operations (Phase 5.2).

This module tests the NEW behaviors introduced in Phase 5.2:
1. Score column variant handling (_distance, _score, _relevance_score)
2. Score normalization to 0-1 range
3. min_similarity threshold filtering
4. search_type and alpha added to results
5. Score column cleanup from results

Note: Alpha parameter behavior and FTS fallback are already tested in
test_enterprise_features.py::TestHybridSearchAlpha - DO NOT DUPLICATE those tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestScoreColumnVariantHandling:
    """Test handling of different score columns returned by LanceDB."""

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock table for testing."""
        table = MagicMock()
        return table

    @pytest.fixture
    def db_with_mock_table(self, temp_storage, mock_table) -> Any:
        """Create a database with mocked table for testing."""
        from spatial_memory.core.database import Database

        db = Database(temp_storage / "test-db")
        db._table = mock_table
        db._has_fts_index = True
        db._cached_row_count = 100
        db._count_cache_time = float("inf")  # Never expire
        return db

    def test_handles_distance_column(self, db_with_mock_table: Any) -> None:
        """Test conversion of _distance to similarity."""
        # Mock search returning _distance column (vector search style)
        mock_results = [
            {"id": "1", "content": "test1", "metadata": "{}", "_distance": 0.2},
            {"id": "2", "content": "test2", "metadata": "{}", "_distance": 0.5},
            {"id": "3", "content": "test3", "metadata": "{}", "_distance": 0.8},
        ]

        # Configure mock chain
        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test query",
            query_vector=query_vector,
            limit=5,
        )

        # Verify _distance was converted to similarity (1 - distance)
        assert len(results) == 3
        assert results[0]["similarity"] == pytest.approx(0.8, rel=0.01)  # 1 - 0.2
        assert results[1]["similarity"] == pytest.approx(0.5, rel=0.01)  # 1 - 0.5
        assert results[2]["similarity"] == pytest.approx(0.2, rel=0.01)  # 1 - 0.8

        # Verify _distance column was removed
        for r in results:
            assert "_distance" not in r

    def test_handles_score_column(self, db_with_mock_table: Any) -> None:
        """Test normalization of _score (BM25) to similarity."""
        # Mock search returning _score column (FTS style)
        mock_results = [
            {"id": "1", "content": "test1", "metadata": "{}", "_score": 2.0},
            {"id": "2", "content": "test2", "metadata": "{}", "_score": 1.0},
            {"id": "3", "content": "test3", "metadata": "{}", "_score": 0.5},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test query",
            query_vector=query_vector,
            limit=5,
        )

        # Verify _score was normalized: similarity = score / (1 + score)
        assert len(results) == 3
        assert results[0]["similarity"] == pytest.approx(2.0 / 3.0, rel=0.01)  # 2/(1+2)
        assert results[1]["similarity"] == pytest.approx(0.5, rel=0.01)  # 1/(1+1)
        assert results[2]["similarity"] == pytest.approx(1.0 / 3.0, rel=0.01)  # 0.5/(1+0.5)

        # Verify _score column was removed
        for r in results:
            assert "_score" not in r

    def test_handles_relevance_score_column(self, db_with_mock_table: Any) -> None:
        """Test handling of _relevance_score (reranker output)."""
        # Mock search returning _relevance_score column (reranker style)
        mock_results = [
            {"id": "1", "content": "test1", "metadata": "{}", "_relevance_score": 0.95},
            {"id": "2", "content": "test2", "metadata": "{}", "_relevance_score": 0.75},
            {"id": "3", "content": "test3", "metadata": "{}", "_relevance_score": 0.50},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test query",
            query_vector=query_vector,
            limit=5,
        )

        # Verify _relevance_score used directly as similarity
        assert len(results) == 3
        assert results[0]["similarity"] == pytest.approx(0.95, rel=0.01)
        assert results[1]["similarity"] == pytest.approx(0.75, rel=0.01)
        assert results[2]["similarity"] == pytest.approx(0.50, rel=0.01)

        # Verify _relevance_score column was removed
        for r in results:
            assert "_relevance_score" not in r

    def test_handles_no_score_column(self, db_with_mock_table: Any) -> None:
        """Test fallback when no score column is present."""
        # Mock search returning no score column
        mock_results = [
            {"id": "1", "content": "test1", "metadata": "{}"},
            {"id": "2", "content": "test2", "metadata": "{}"},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test query",
            query_vector=query_vector,
            limit=5,
        )

        # Should fall back to default similarity of 0.5
        assert len(results) == 2
        for r in results:
            assert r["similarity"] == pytest.approx(0.5, rel=0.01)


class TestScoreNormalization:
    """Test score normalization formulas."""

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock table for testing."""
        return MagicMock()

    @pytest.fixture
    def db_with_mock_table(self, temp_storage, mock_table) -> Any:
        """Create a database with mocked table for testing."""
        from spatial_memory.core.database import Database

        db = Database(temp_storage / "test-db")
        db._table = mock_table
        db._has_fts_index = True
        db._cached_row_count = 100
        db._count_cache_time = float("inf")
        return db

    def test_distance_normalization_clamped_to_0_1(
        self, db_with_mock_table: Any
    ) -> None:
        """Test that distance-to-similarity conversion is clamped to [0, 1]."""
        # Edge case: distance > 1 (can happen with unnormalized vectors)
        mock_results = [
            {"id": "1", "content": "test", "metadata": "{}", "_distance": 1.5},
            {"id": "2", "content": "test", "metadata": "{}", "_distance": -0.2},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=5,
        )

        # Similarity should be clamped: 1 - 1.5 = -0.5 -> 0.0
        assert results[0]["similarity"] == pytest.approx(0.0, rel=0.01)
        # Similarity: 1 - (-0.2) = 1.2 -> 1.0
        assert results[1]["similarity"] == pytest.approx(1.0, rel=0.01)

    def test_bm25_score_normalization_formula(self, db_with_mock_table: Any) -> None:
        """Test BM25 score normalization: similarity = score / (1 + score)."""
        # Test various BM25 scores
        mock_results = [
            {"id": "1", "content": "t", "metadata": "{}", "_score": 0.0},  # 0/(1+0) = 0
            {"id": "2", "content": "t", "metadata": "{}", "_score": 1.0},  # 1/2 = 0.5
            {"id": "3", "content": "t", "metadata": "{}", "_score": 9.0},  # 9/10 = 0.9
            {"id": "4", "content": "t", "metadata": "{}", "_score": 99.0},  # 99/100 = 0.99
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
        )

        assert results[0]["similarity"] == pytest.approx(0.0, rel=0.01)
        assert results[1]["similarity"] == pytest.approx(0.5, rel=0.01)
        assert results[2]["similarity"] == pytest.approx(0.9, rel=0.01)
        assert results[3]["similarity"] == pytest.approx(0.99, rel=0.01)

    def test_relevance_score_used_directly(self, db_with_mock_table: Any) -> None:
        """Test that _relevance_score is used directly (already 0-1)."""
        mock_results = [
            {"id": "1", "content": "t", "metadata": "{}", "_relevance_score": 0.123},
            {"id": "2", "content": "t", "metadata": "{}", "_relevance_score": 0.999},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
        )

        # _relevance_score should be used as-is
        assert results[0]["similarity"] == pytest.approx(0.123, rel=0.01)
        assert results[1]["similarity"] == pytest.approx(0.999, rel=0.01)


class TestMinSimilarityThreshold:
    """Test min_similarity threshold filtering."""

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock table for testing."""
        return MagicMock()

    @pytest.fixture
    def db_with_mock_table(self, temp_storage, mock_table) -> Any:
        """Create a database with mocked table for testing."""
        from spatial_memory.core.database import Database

        db = Database(temp_storage / "test-db")
        db._table = mock_table
        db._has_fts_index = True
        db._cached_row_count = 100
        db._count_cache_time = float("inf")
        return db

    def test_filters_below_threshold(self, db_with_mock_table: Any) -> None:
        """Test that results below min_similarity are filtered out."""
        # Results with varying distances (will be converted to similarity)
        mock_results = [
            {"id": "1", "content": "high", "metadata": "{}", "_distance": 0.1},  # sim=0.9
            {"id": "2", "content": "med", "metadata": "{}", "_distance": 0.3},  # sim=0.7
            {"id": "3", "content": "low", "metadata": "{}", "_distance": 0.6},  # sim=0.4
            {"id": "4", "content": "vlow", "metadata": "{}", "_distance": 0.9},  # sim=0.1
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
            min_similarity=0.5,
        )

        # Only results with similarity >= 0.5 should be returned
        assert len(results) == 2
        assert results[0]["content"] == "high"
        assert results[1]["content"] == "med"

    def test_min_similarity_zero_returns_all(self, db_with_mock_table: Any) -> None:
        """Test that min_similarity=0 returns all results."""
        mock_results = [
            {"id": "1", "content": "t1", "metadata": "{}", "_distance": 0.9},  # sim=0.1
            {"id": "2", "content": "t2", "metadata": "{}", "_distance": 0.95},  # sim=0.05
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
            min_similarity=0.0,
        )

        # All results should be returned
        assert len(results) == 2

    def test_min_similarity_one_filters_all(self, db_with_mock_table: Any) -> None:
        """Test that min_similarity=1.0 filters most results."""
        mock_results = [
            {"id": "1", "content": "t1", "metadata": "{}", "_distance": 0.0},  # sim=1.0
            {"id": "2", "content": "t2", "metadata": "{}", "_distance": 0.01},  # sim=0.99
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
            min_similarity=1.0,
        )

        # Only exact match (sim=1.0) should pass
        assert len(results) == 1
        assert results[0]["similarity"] == pytest.approx(1.0, rel=0.01)

    def test_min_similarity_with_bm25_score(self, db_with_mock_table: Any) -> None:
        """Test min_similarity threshold works with BM25 scores."""
        # BM25 scores: normalized to similarity = score/(1+score)
        mock_results = [
            {"id": "1", "content": "high", "metadata": "{}", "_score": 9.0},  # sim=0.9
            {"id": "2", "content": "med", "metadata": "{}", "_score": 1.0},  # sim=0.5
            {"id": "3", "content": "low", "metadata": "{}", "_score": 0.25},  # sim=0.2
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=10,
            min_similarity=0.4,
        )

        # Only high (0.9) and med (0.5) should pass the 0.4 threshold
        assert len(results) == 2
        assert results[0]["content"] == "high"
        assert results[1]["content"] == "med"


class TestResultMetadataEnhancement:
    """Test that search_type and alpha are added to results."""

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock table for testing."""
        return MagicMock()

    @pytest.fixture
    def db_with_mock_table(self, temp_storage, mock_table) -> Any:
        """Create a database with mocked table for testing."""
        from spatial_memory.core.database import Database

        db = Database(temp_storage / "test-db")
        db._table = mock_table
        db._has_fts_index = True
        db._cached_row_count = 100
        db._count_cache_time = float("inf")
        return db

    def test_search_type_added_to_results(self, db_with_mock_table: Any) -> None:
        """Test that search_type='hybrid' is added to all results."""
        mock_results = [
            {"id": "1", "content": "test1", "metadata": "{}", "_distance": 0.1},
            {"id": "2", "content": "test2", "metadata": "{}", "_distance": 0.2},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=5,
        )

        for r in results:
            assert r["search_type"] == "hybrid"

    def test_alpha_value_added_to_results(self, db_with_mock_table: Any) -> None:
        """Test that alpha value is added to all results."""
        mock_results = [
            {"id": "1", "content": "test", "metadata": "{}", "_distance": 0.1},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)

        # Test with different alpha values
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            results = db_with_mock_table.hybrid_search(
                query="test",
                query_vector=query_vector,
                limit=5,
                alpha=alpha,
            )
            assert len(results) == 1
            assert results[0]["alpha"] == alpha


class TestScoreColumnCleanup:
    """Test that internal score columns are removed from results."""

    @pytest.fixture
    def mock_table(self) -> MagicMock:
        """Create a mock table for testing."""
        return MagicMock()

    @pytest.fixture
    def db_with_mock_table(self, temp_storage, mock_table) -> Any:
        """Create a database with mocked table for testing."""
        from spatial_memory.core.database import Database

        db = Database(temp_storage / "test-db")
        db._table = mock_table
        db._has_fts_index = True
        db._cached_row_count = 100
        db._count_cache_time = float("inf")
        return db

    def test_distance_column_removed(self, db_with_mock_table: Any) -> None:
        """Test that _distance column is removed from results."""
        mock_results = [
            {"id": "1", "content": "test", "metadata": "{}", "_distance": 0.1},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=5,
        )

        assert "_distance" not in results[0]
        assert "similarity" in results[0]

    def test_score_column_removed(self, db_with_mock_table: Any) -> None:
        """Test that _score column is removed from results."""
        mock_results = [
            {"id": "1", "content": "test", "metadata": "{}", "_score": 2.0},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=5,
        )

        assert "_score" not in results[0]
        assert "similarity" in results[0]

    def test_relevance_score_column_removed(self, db_with_mock_table: Any) -> None:
        """Test that _relevance_score column is removed from results."""
        mock_results = [
            {"id": "1", "content": "test", "metadata": "{}", "_relevance_score": 0.9},
        ]

        search_mock = MagicMock()
        search_mock.vector.return_value = search_mock
        search_mock.vector_column_name.return_value = search_mock
        search_mock.rerank.return_value = search_mock
        search_mock.where.return_value = search_mock
        search_mock.limit.return_value = search_mock
        search_mock.to_list.return_value = mock_results

        db_with_mock_table._table.search.return_value = search_mock

        query_vector = np.random.randn(384).astype(np.float32)
        results = db_with_mock_table.hybrid_search(
            query="test",
            query_vector=query_vector,
            limit=5,
        )

        assert "_relevance_score" not in results[0]
        assert "similarity" in results[0]
