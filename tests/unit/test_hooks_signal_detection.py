"""Unit tests for spatial_memory.hooks.signal_detection.

Tests cover:
1. Tier 1 patterns — each pattern detects correctly
2. Tier 2 patterns — each pattern detects correctly
3. Tier 3 (no signal) — noise text classified as Tier 3
4. Scoring — max confidence, multiple matches, bounds
5. Edge cases — empty string, long text, Unicode, case insensitivity, multiline
6. Performance — classify_signal under 1ms
"""

from __future__ import annotations

import time

import pytest

from spatial_memory.hooks.signal_detection import SignalResult, classify_signal

# =============================================================================
# Tier 1 Patterns
# =============================================================================


@pytest.mark.unit
class TestTier1Patterns:
    """Each Tier 1 pattern detects correctly and returns tier=1."""

    def test_decision_decided(self) -> None:
        result = classify_signal("We decided to use PostgreSQL for the database.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_decision_chose(self) -> None:
        result = classify_signal("The team chose React over Vue.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_decision_going_with(self) -> None:
        result = classify_signal("Going with Redis for caching.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_decision_will_use(self) -> None:
        result = classify_signal("We will use Docker for deployment.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_important_note(self) -> None:
        result = classify_signal("Important: always run migrations before deploy.")
        assert result.tier == 1
        assert "important" in result.patterns_matched

    def test_solution_fix_was(self) -> None:
        result = classify_signal("The fix was adding a null check in the handler.")
        assert result.tier == 1
        assert "solution" in result.patterns_matched

    def test_solution_approach_is(self) -> None:
        result = classify_signal("The approach is to batch all writes together.")
        assert result.tier == 1
        assert "solution" in result.patterns_matched

    def test_error_issue_was(self) -> None:
        result = classify_signal("The issue was a race condition in the pool.")
        assert result.tier == 1
        assert "error" in result.patterns_matched

    def test_explicit_save(self) -> None:
        result = classify_signal("Save that the API endpoint changed to v2.")
        assert result.tier == 1
        assert "explicit" in result.patterns_matched

    def test_pattern_trick_is(self) -> None:
        result = classify_signal("The trick is to pre-warm the cache on startup.")
        assert result.tier == 1
        assert "pattern" in result.patterns_matched

    def test_resolved_by(self) -> None:
        result = classify_signal("Resolved by upgrading the library to v3.0.")
        assert result.tier == 1
        assert "solution" in result.patterns_matched

    def test_broke_because(self) -> None:
        result = classify_signal("The pipeline broke because of a missing env var.")
        assert result.tier == 1
        assert "error" in result.patterns_matched

    def test_failed_because(self) -> None:
        result = classify_signal("The deployment failed because the image was too large.")
        assert result.tier == 1
        assert "error" in result.patterns_matched

    def test_error_was_due_to(self) -> None:
        result = classify_signal("The error was due to an expired certificate.")
        assert result.tier == 1
        assert "error" in result.patterns_matched

    def test_architecture_will_be(self) -> None:
        result = classify_signal("The architecture will be event-driven with CQRS.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched


# =============================================================================
# Tier 2 Patterns
# =============================================================================


@pytest.mark.unit
class TestTier2Patterns:
    """Each Tier 2 pattern detects correctly and returns tier=2."""

    def test_definition_is(self) -> None:
        result = classify_signal("A mutex is a synchronization primitive.")
        assert result.tier == 2
        assert "definition" in result.patterns_matched

    def test_definition_means(self) -> None:
        result = classify_signal("ACID means atomicity, consistency, isolation, durability.")
        assert result.tier == 2
        assert "definition" in result.patterns_matched

    def test_convention(self) -> None:
        result = classify_signal("The convention here is to use camelCase for variables.")
        assert result.tier == 2
        assert "convention" in result.patterns_matched

    def test_watch_out_for(self) -> None:
        result = classify_signal("Watch out for null pointer exceptions in the parser.")
        assert result.tier == 2
        assert "workaround" in result.patterns_matched

    def test_workaround_is(self) -> None:
        result = classify_signal("The workaround is to restart the service every hour.")
        assert result.tier == 2
        assert "workaround" in result.patterns_matched

    def test_configuration(self) -> None:
        result = classify_signal("You need to set MAX_POOL_SIZE to 20 in production.")
        assert result.tier == 2
        assert "configuration" in result.patterns_matched


# =============================================================================
# Tier 3 (No Signal)
# =============================================================================


@pytest.mark.unit
class TestTier3NoSignal:
    """Noise text should be classified as Tier 3 with score 0.0."""

    def test_greeting(self) -> None:
        result = classify_signal("Hello, thanks for the help!")
        assert result.tier == 3
        assert result.score == 0.0
        assert result.patterns_matched == []

    def test_trivial(self) -> None:
        result = classify_signal("ok thanks")
        assert result.tier == 3

    def test_code_snippet(self) -> None:
        result = classify_signal("x = 42\nprint(x)")
        assert result.tier == 3

    def test_random_words(self) -> None:
        result = classify_signal("apple banana cherry")
        assert result.tier == 3
        assert result.score == 0.0


# =============================================================================
# Scoring
# =============================================================================


@pytest.mark.unit
class TestScoring:
    """Test score calculation and bounds."""

    def test_max_confidence_explicit(self) -> None:
        """Explicit save has highest confidence (0.95)."""
        result = classify_signal("Remember that the config uses port 8080.")
        assert result.score == 0.95

    def test_multiple_matches(self) -> None:
        """Text matching multiple patterns returns all types."""
        text = "Decided to use Redis. The fix was adding a TTL."
        result = classify_signal(text)
        assert result.tier == 1
        assert len(result.patterns_matched) >= 2
        assert "decision" in result.patterns_matched
        assert "solution" in result.patterns_matched

    def test_score_bounds(self) -> None:
        """Score is always between 0.0 and 1.0."""
        for text in [
            "",
            "hello",
            "Decided to use X.",
            "Save that Y. The fix was Z. Important: W.",
        ]:
            result = classify_signal(text)
            assert 0.0 <= result.score <= 1.0


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases for signal classification."""

    def test_empty_string(self) -> None:
        result = classify_signal("")
        assert result == SignalResult(tier=3, score=0.0, patterns_matched=[])

    def test_whitespace_only(self) -> None:
        result = classify_signal("   \n\t  ")
        assert result.tier == 3

    def test_unicode_text(self) -> None:
        result = classify_signal("Decided to use PostgreSQL for 数据库管理.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_case_insensitivity(self) -> None:
        """Patterns should match regardless of case."""
        result = classify_signal("DECIDED TO USE Kubernetes for orchestration.")
        assert result.tier == 1
        assert "decision" in result.patterns_matched

    def test_multiline_input(self) -> None:
        text = "Line 1: some context.\nThe fix was updating the schema.\nLine 3: done."
        result = classify_signal(text)
        assert result.tier == 1
        assert "solution" in result.patterns_matched

    def test_long_text(self) -> None:
        """Long text should still work without errors."""
        text = "Random padding. " * 1000 + "Decided to use Rust."
        result = classify_signal(text)
        assert result.tier == 1
        assert "decision" in result.patterns_matched


# =============================================================================
# Performance
# =============================================================================


@pytest.mark.unit
class TestPerformance:
    """classify_signal should complete in under 1ms for typical inputs."""

    def test_classify_under_1ms(self) -> None:
        texts = [
            "Decided to use PostgreSQL because it handles JSONB natively.",
            "The fix was in core/database.py — the pool was exhausted.",
            "ok thanks",
            "The trick is to batch embed before dedup checking.",
            "Hello world, nothing interesting here at all.",
        ]
        # Warm up
        for t in texts:
            classify_signal(t)

        # Measure
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            for t in texts:
                classify_signal(t)
        elapsed = time.perf_counter() - start

        # Average per call
        avg_ms = (elapsed / (iterations * len(texts))) * 1000
        assert avg_ms < 1.0, f"Average {avg_ms:.3f}ms per call exceeds 1ms threshold"

    def test_definition_pattern_no_redos(self) -> None:
        """H-3: Definition pattern must complete on 100K input in <100ms."""
        text = "a " * 50_000 + " is "
        start = time.perf_counter()
        classify_signal(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Definition pattern took {elapsed_ms:.1f}ms (ReDoS?)"


# =============================================================================
# M-2: False Positive Resistance (substring keywords)
# =============================================================================


@pytest.mark.unit
class TestFalsePositiveSubstring:
    """M-2: Short keywords must not match as substrings of longer words."""

    def test_false_positive_restore(self) -> None:
        """'restore' should NOT match the 'store' keyword for 'explicit'."""
        result = classify_signal("restore the database from backup.")
        assert "explicit" not in result.patterns_matched

    def test_false_positive_notebook(self) -> None:
        """'notebook' should NOT match the 'note' keyword for 'important'."""
        result = classify_signal("Open the notebook to review the data.")
        assert "important" not in result.patterns_matched

    def test_false_positive_datastore(self) -> None:
        """'datastore' should NOT match the 'store' keyword for 'explicit'."""
        result = classify_signal("The datastore is ready for queries.")
        assert "explicit" not in result.patterns_matched

    def test_true_positive_store_still_works(self) -> None:
        """'Store that config' should still match as tier 1 explicit."""
        result = classify_signal("Store that config in the settings file.")
        assert result.tier == 1
        assert "explicit" in result.patterns_matched
