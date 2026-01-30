"""Unit tests for core spatial operations.

Tests the mathematical algorithms used for spatial navigation:
- Vector normalization
- SLERP (spherical linear interpolation)
- Temperature-based softmax selection
"""

from __future__ import annotations

import numpy as np
import pytest

from spatial_memory.core.spatial_ops import (
    normalize,
    slerp,
    slerp_path,
    softmax_with_temperature,
    temperature_select,
)


class TestNormalize:
    """Tests for normalize() - vector normalization to unit length."""

    def test_normalize_unit_vector(self) -> None:
        """normalize() should return unit vector unchanged."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_scales_to_unit(self) -> None:
        """normalize() should scale vectors to unit length."""
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # 3-4-5 triangle
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        # Expected: [0.6, 0.8, 0.0]
        np.testing.assert_array_almost_equal(result, [0.6, 0.8, 0.0], decimal=5)

    def test_normalize_zero_vector(self) -> None:
        """normalize() should return zero vector for zero input."""
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_preserves_direction(self) -> None:
        """normalize() should preserve vector direction."""
        v = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        result = normalize(v)
        # Direction should be [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        expected = v / np.linalg.norm(v)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_normalize_handles_high_dimensional(self) -> None:
        """normalize() should work with high-dimensional vectors."""
        v = np.random.randn(384).astype(np.float32)
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5


class TestSlerp:
    """Tests for slerp() - spherical linear interpolation."""

    def test_slerp_t0_returns_start(self) -> None:
        """slerp() at t=0 should return the start vector."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.0)
        np.testing.assert_array_almost_equal(result, v0, decimal=5)

    def test_slerp_t1_returns_end(self) -> None:
        """slerp() at t=1 should return the end vector."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 1.0)
        np.testing.assert_array_almost_equal(result, v1, decimal=5)

    def test_slerp_midpoint_is_unit(self) -> None:
        """slerp() midpoint should be a unit vector."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_midpoint_equidistant(self) -> None:
        """slerp() midpoint should be equidistant from both endpoints."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)

        # Angular distances should be equal
        dot_to_start = np.dot(result, v0)
        dot_to_end = np.dot(result, v1)
        assert abs(dot_to_start - dot_to_end) < 1e-5

    def test_slerp_handles_parallel_vectors(self) -> None:
        """slerp() should handle nearly parallel vectors (linear fallback)."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([1.0, 0.0001, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        # Result should still be unit length
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5
        # Should be close to both vectors
        assert np.dot(result, v0) > 0.999

    def test_slerp_handles_antipodal_vectors(self) -> None:
        """slerp() should handle antipodal vectors (opposite directions)."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        result = slerp(v0, v1, 0.5)
        # Result should be perpendicular to both (on the great circle)
        assert abs(np.dot(result, v0)) < 0.1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_all_points_unit_length(self) -> None:
        """slerp() should produce unit vectors for all t values."""
        v0 = normalize(np.array([1.0, 0.5, 0.2], dtype=np.float32))
        v1 = normalize(np.array([0.2, 1.0, 0.5], dtype=np.float32))

        for t in np.linspace(0, 1, 11):
            result = slerp(v0, v1, t)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_slerp_high_dimensional(self) -> None:
        """slerp() should work with high-dimensional vectors (e.g., 384-dim embeddings)."""
        rng = np.random.default_rng(42)
        v0 = normalize(rng.standard_normal(384).astype(np.float32))
        v1 = normalize(rng.standard_normal(384).astype(np.float32))

        result = slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5
        assert result.shape == (384,)


class TestSlerpPath:
    """Tests for slerp_path() - generating interpolation paths."""

    def test_slerp_path_generates_correct_count(self) -> None:
        """slerp_path() should generate the specified number of points."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=5)
        assert len(path) == 5

    def test_slerp_path_all_unit_vectors(self) -> None:
        """slerp_path() should produce all unit vectors."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=10)

        for v in path:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_slerp_path_invalid_steps_raises_valueerror(self) -> None:
        """slerp_path() should raise ValueError for steps < 1."""
        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        with pytest.raises(ValueError, match="steps must be at least 1"):
            slerp_path(v0, v1, steps=0)

        with pytest.raises(ValueError, match="steps must be at least 1"):
            slerp_path(v0, v1, steps=-1)

    def test_slerp_path_includes_endpoints(self) -> None:
        """slerp_path() with include_endpoints=True should include start and end."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=5, include_endpoints=True)

        np.testing.assert_array_almost_equal(path[0], v0, decimal=5)
        np.testing.assert_array_almost_equal(path[-1], v1, decimal=5)

    def test_slerp_path_excludes_endpoints(self) -> None:
        """slerp_path() with include_endpoints=False should exclude start and end."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=5, include_endpoints=False)

        # First and last should NOT be the endpoints
        assert np.dot(path[0], v0) < 0.99
        assert np.dot(path[-1], v1) < 0.99

    def test_slerp_path_single_step(self) -> None:
        """slerp_path() with steps=1 should return single point."""
        v0 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v1 = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        path = slerp_path(v0, v1, steps=1)

        assert len(path) == 1


class TestSoftmaxWithTemperature:
    """Tests for softmax_with_temperature() - temperature-scaled softmax."""

    def test_softmax_sums_to_one(self) -> None:
        """softmax_with_temperature() output should sum to 1.0."""
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=1.0)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_softmax_temperature_zero_is_argmax(self) -> None:
        """softmax_with_temperature() at T=0 should be one-hot (argmax)."""
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=0.0)

        # Highest score gets probability 1.0
        assert probs[0] == 1.0
        assert probs[1] == 0.0
        assert probs[2] == 0.0

    def test_softmax_temperature_very_low_is_argmax(self) -> None:
        """softmax_with_temperature() at very low T should approach argmax."""
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=0.001)

        # Highest score should dominate
        assert probs[0] > 0.99

    def test_softmax_high_temperature_is_uniform(self) -> None:
        """softmax_with_temperature() at high T should approach uniform."""
        scores = np.array([0.9, 0.7, 0.3])
        probs = softmax_with_temperature(scores, temperature=100.0)

        # Should be nearly uniform (each ~0.33)
        assert all(0.25 < p < 0.45 for p in probs)
        uniform = 1.0 / len(scores)
        for p in probs:
            assert abs(p - uniform) < 0.1

    def test_softmax_negative_temperature_raises(self) -> None:
        """softmax_with_temperature() should raise for negative temperature."""
        scores = np.array([0.9, 0.7, 0.3])

        with pytest.raises(ValueError, match="[Tt]emperature.*non-negative"):
            softmax_with_temperature(scores, temperature=-1.0)

    def test_softmax_preserves_ordering(self) -> None:
        """softmax_with_temperature() should preserve score ordering in probabilities."""
        scores = np.array([0.9, 0.7, 0.3, 0.5])
        probs = softmax_with_temperature(scores, temperature=1.0)

        # Probabilities should maintain the same ordering as scores
        assert probs[0] > probs[1] > probs[3] > probs[2]

    def test_softmax_empty_array(self) -> None:
        """softmax_with_temperature() should handle empty arrays."""
        scores = np.array([])
        probs = softmax_with_temperature(scores, temperature=1.0)

        assert len(probs) == 0

    def test_softmax_single_element(self) -> None:
        """softmax_with_temperature() with single element should return [1.0]."""
        scores = np.array([0.5])
        probs = softmax_with_temperature(scores, temperature=1.0)

        assert len(probs) == 1
        assert probs[0] == 1.0

    def test_softmax_numerical_stability(self) -> None:
        """softmax_with_temperature() should be numerically stable with large values."""
        # Large values that would overflow without proper handling
        scores = np.array([1000.0, 1001.0, 999.0])
        probs = softmax_with_temperature(scores, temperature=1.0)

        # Should not have NaN or Inf
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-6


class TestTemperatureSelect:
    """Tests for temperature_select() - temperature-based item selection."""

    def test_temperature_select_returns_item(self) -> None:
        """temperature_select() should return an item from the list."""
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7, 0.3])
        result = temperature_select(items, scores, temperature=1.0)

        assert result in items

    def test_temperature_select_greedy_with_low_temperature(self) -> None:
        """temperature_select() with low temperature should select highest score."""
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7, 0.3])

        # With very low temperature, should always pick highest score
        for _ in range(10):
            result = temperature_select(items, scores, temperature=0.01)
            assert result == "a"

    def test_temperature_select_random_with_high_temperature(self) -> None:
        """temperature_select() with high temperature should select varied items."""
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7, 0.3])

        # With high temperature, should get variety
        results = set()
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = temperature_select(items, scores, temperature=10.0, rng=rng)
            results.add(result)

        # Should see at least 2 different items (likely all 3)
        assert len(results) >= 2

    def test_temperature_select_with_objects(self) -> None:
        """temperature_select() should work with any sequence type."""
        items = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
        scores = np.array([0.9, 0.7, 0.3])

        result = temperature_select(items, scores, temperature=1.0)
        assert result in items

    def test_temperature_select_mismatched_length_raises(self) -> None:
        """temperature_select() should raise for mismatched items/scores length."""
        items = ["a", "b", "c"]
        scores = np.array([0.9, 0.7])  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            temperature_select(items, scores, temperature=1.0)

    def test_temperature_select_uses_provided_rng(self) -> None:
        """temperature_select() should use provided RNG for reproducibility."""
        items = ["a", "b", "c"]
        scores = np.array([0.5, 0.5, 0.5])  # Equal scores

        # Same seed should give same results
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        results1 = [temperature_select(items, scores, 1.0, rng1) for _ in range(10)]
        results2 = [temperature_select(items, scores, 1.0, rng2) for _ in range(10)]

        assert results1 == results2

    def test_temperature_select_single_item(self) -> None:
        """temperature_select() with single item should always return it."""
        items = ["only"]
        scores = np.array([0.5])

        for _ in range(10):
            result = temperature_select(items, scores, temperature=1.0)
            assert result == "only"
