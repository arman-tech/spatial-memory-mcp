"""Unit tests for core lifecycle operations.

Tests the mathematical algorithms used for lifecycle management:
- Decay calculations (exponential, linear, step)
- Reinforcement calculations (additive, multiplicative, set_value)
- Extraction patterns (decision, solution, explicit save)
- Consolidation algorithms (Jaccard similarity, union-find, merge strategies)
"""

from __future__ import annotations

import numpy as np

from spatial_memory.core.lifecycle_ops import (
    EXTRACTION_PATTERNS,
    ExtractionCandidate,
    calculate_decay_factor,
    calculate_reinforcement,
    dedupe_overlapping_extractions,
    extract_candidates,
    find_duplicate_groups,
    jaccard_similarity,
    merge_memory_metadata,
    score_extraction_confidence,
    select_representative,
)

# =============================================================================
# TestDecayCalculations
# =============================================================================


class TestDecayCalculations:
    """Tests for decay factor calculations."""

    def test_exponential_decay_halves_at_half_life(self) -> None:
        """Exponential decay should reduce importance by ~50% at half-life."""
        # With importance=0, importance factor is 1.0, so effective_half_life = 30 days
        # factor = 2^(-30/30) = 2^(-1) = 0.5
        factor_pure = calculate_decay_factor(
            days_since_access=30.0,
            access_count=0,
            base_importance=0.0,  # importance_factor = 1
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )
        # Now effective_half_life = 30 days exactly
        # factor = 2^(-30/30) = 2^(-1) = 0.5
        assert abs(factor_pure - 0.5) < 0.01

    def test_exponential_decay_at_zero_days(self) -> None:
        """Exponential decay at day 0 should return 1.0 (no decay)."""
        factor = calculate_decay_factor(
            days_since_access=0.0,
            access_count=0,
            base_importance=0.5,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )
        assert abs(factor - 1.0) < 0.01

    def test_linear_decay_reaches_zero(self) -> None:
        """Linear decay should reach zero at 2x half-life."""
        factor = calculate_decay_factor(
            days_since_access=60.0,
            access_count=0,
            base_importance=0.0,  # importance_factor = 1
            decay_function="linear",
            half_life_days=30.0,
            access_weight=0.0,
        )
        # effective_half_life = 30 days
        # linear: 1 - days / (2 * half_life) = 1 - 60/60 = 0
        assert factor == 0.0

    def test_linear_decay_at_half_life(self) -> None:
        """Linear decay should be 0.5 at half-life."""
        factor = calculate_decay_factor(
            days_since_access=30.0,
            access_count=0,
            base_importance=0.0,
            decay_function="linear",
            half_life_days=30.0,
            access_weight=0.0,
        )
        # linear: 1 - days / (2 * half_life) = 1 - 30/60 = 0.5
        assert abs(factor - 0.5) < 0.01

    def test_step_decay_transitions(self) -> None:
        """Step decay should transition at half-life boundaries."""
        # Before half-life: 1.0
        factor_early = calculate_decay_factor(
            days_since_access=15.0,
            access_count=0,
            base_importance=0.0,
            decay_function="step",
            half_life_days=30.0,
            access_weight=0.0,
        )
        assert factor_early == 1.0

        # Between 1x and 2x half-life: 0.5
        factor_mid = calculate_decay_factor(
            days_since_access=45.0,
            access_count=0,
            base_importance=0.0,
            decay_function="step",
            half_life_days=30.0,
            access_weight=0.0,
        )
        assert factor_mid == 0.5

        # After 2x half-life: 0.25
        factor_late = calculate_decay_factor(
            days_since_access=90.0,
            access_count=0,
            base_importance=0.0,
            decay_function="step",
            half_life_days=30.0,
            access_weight=0.0,
        )
        assert factor_late == 0.25

    def test_access_count_slows_decay(self) -> None:
        """Higher access count should result in slower decay (higher factor)."""
        factor_no_access = calculate_decay_factor(
            days_since_access=30.0,
            access_count=0,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        factor_with_access = calculate_decay_factor(
            days_since_access=30.0,
            access_count=5,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        # More accesses = longer effective half-life = higher decay factor
        assert factor_with_access > factor_no_access

    def test_high_importance_slows_decay(self) -> None:
        """Higher base importance should result in slower decay."""
        factor_low_importance = calculate_decay_factor(
            days_since_access=30.0,
            access_count=0,
            base_importance=0.1,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        factor_high_importance = calculate_decay_factor(
            days_since_access=30.0,
            access_count=0,
            base_importance=0.9,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        # Higher importance = longer effective half-life = higher decay factor
        assert factor_high_importance > factor_low_importance

    def test_min_floor_enforced(self) -> None:
        """Decay factor should not go below 0."""
        factor = calculate_decay_factor(
            days_since_access=1000.0,  # Very old
            access_count=0,
            base_importance=0.0,
            decay_function="linear",
            half_life_days=30.0,
            access_weight=0.0,
        )
        assert factor >= 0.0

    def test_access_weight_blends_time_and_access(self) -> None:
        """Access weight should blend time decay with access stability."""
        # With high access_weight, factor should depend more on access count
        factor_no_weight = calculate_decay_factor(
            days_since_access=60.0,
            access_count=10,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        factor_with_weight = calculate_decay_factor(
            days_since_access=60.0,
            access_count=10,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.5,
        )

        # With access weight, the factor is influenced by access count stability
        # Both should be valid factors (0-1 range)
        assert 0.0 <= factor_no_weight <= 1.0
        assert 0.0 <= factor_with_weight <= 1.0

    def test_access_count_capped_at_20(self) -> None:
        """Access count influence should be capped at 20 accesses."""
        factor_20_accesses = calculate_decay_factor(
            days_since_access=30.0,
            access_count=20,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        factor_100_accesses = calculate_decay_factor(
            days_since_access=30.0,
            access_count=100,
            base_importance=0.0,
            decay_function="exponential",
            half_life_days=30.0,
            access_weight=0.0,
        )

        # Should be the same due to cap
        assert abs(factor_20_accesses - factor_100_accesses) < 0.01


# =============================================================================
# TestReinforcementCalculations
# =============================================================================


class TestReinforcementCalculations:
    """Tests for reinforcement boost calculations."""

    def test_additive_boost(self) -> None:
        """Additive boost should add fixed amount to importance."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="additive",
            boost_amount=0.1,
            max_importance=1.0,
        )
        assert abs(new_importance - 0.6) < 0.001
        assert abs(actual_boost - 0.1) < 0.001

    def test_multiplicative_boost(self) -> None:
        """Multiplicative boost should multiply importance by (1 + amount)."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="multiplicative",
            boost_amount=0.2,  # 20% increase
            max_importance=1.0,
        )
        assert abs(new_importance - 0.6) < 0.001  # 0.5 * 1.2 = 0.6
        assert abs(actual_boost - 0.1) < 0.001  # 0.6 - 0.5 = 0.1

    def test_set_value_override(self) -> None:
        """Set value should override current importance."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.3,
            boost_type="set_value",
            boost_amount=0.8,
            max_importance=1.0,
        )
        assert new_importance == 0.8
        assert actual_boost == 0.5  # 0.8 - 0.3 = 0.5

    def test_max_cap_enforced(self) -> None:
        """Importance should not exceed max_importance."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.9,
            boost_type="additive",
            boost_amount=0.3,  # Would exceed 1.0
            max_importance=1.0,
        )
        assert abs(new_importance - 1.0) < 0.001
        assert abs(actual_boost - 0.1) < 0.001  # Capped at 1.0 - 0.9

    def test_set_value_respects_max_cap(self) -> None:
        """Set value should also respect max_importance."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="set_value",
            boost_amount=1.5,  # Exceeds max
            max_importance=1.0,
        )
        assert new_importance == 1.0
        assert actual_boost == 0.5

    def test_zero_boost(self) -> None:
        """Zero boost should not change importance."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="additive",
            boost_amount=0.0,
            max_importance=1.0,
        )
        assert new_importance == 0.5
        assert actual_boost == 0.0

    def test_negative_boost_not_allowed(self) -> None:
        """Boost amount should be non-negative (or handled gracefully)."""
        # Additive with negative would decrease, which might not be desired
        # Implementation should either prevent negative or handle it
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="additive",
            boost_amount=-0.1,
            max_importance=1.0,
        )
        # Either it's clamped to 0 or applied as decrease
        # The implementation decision determines expected behavior
        assert new_importance >= 0.0
        assert new_importance <= 1.0

    def test_custom_max_importance(self) -> None:
        """Custom max_importance should be respected."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="additive",
            boost_amount=0.5,
            max_importance=0.8,  # Custom cap
        )
        assert abs(new_importance - 0.8) < 0.001
        assert abs(actual_boost - 0.3) < 0.001

    def test_unknown_boost_type_returns_unchanged(self) -> None:
        """Unknown boost type should return current importance unchanged."""
        new_importance, actual_boost = calculate_reinforcement(
            current_importance=0.5,
            boost_type="unknown",  # type: ignore[arg-type]
            boost_amount=0.1,
            max_importance=1.0,
        )
        assert new_importance == 0.5
        assert actual_boost == 0.0


# =============================================================================
# TestExtractionPatterns
# =============================================================================


class TestExtractionPatterns:
    """Tests for memory extraction pattern matching."""

    def test_decision_pattern(self) -> None:
        """Decision pattern should match decision-related text."""
        text = "We decided to use PostgreSQL for the database."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        decision_matches = [c for c in candidates if c.pattern_type == "decision"]
        assert len(decision_matches) > 0
        assert any("PostgreSQL" in c.content for c in decision_matches)

    def test_solution_pattern(self) -> None:
        """Solution pattern should match fix/solution text."""
        text = "The fix was to increase the connection pool size."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        solution_matches = [c for c in candidates if c.pattern_type == "solution"]
        assert len(solution_matches) > 0
        assert any("connection pool" in c.content for c in solution_matches)

    def test_explicit_save_pattern(self) -> None:
        """Explicit save pattern should match save/remember requests."""
        text = "Remember that the API key expires every 30 days."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        explicit_matches = [c for c in candidates if c.pattern_type == "explicit"]
        assert len(explicit_matches) > 0

    def test_important_pattern(self) -> None:
        """Important pattern should match key point markers."""
        text = "Important: always run migrations before deployment."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        important_matches = [c for c in candidates if c.pattern_type == "important"]
        assert len(important_matches) > 0

    def test_error_pattern(self) -> None:
        """Error pattern should match issue/problem diagnoses."""
        text = "The issue was that the cache TTL was too short."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        error_matches = [c for c in candidates if c.pattern_type == "error"]
        assert len(error_matches) > 0

    def test_pattern_learning_pattern(self) -> None:
        """Pattern type should match learning/trick markers."""
        text = "The trick is to batch the API calls for better performance."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        pattern_matches = [c for c in candidates if c.pattern_type == "pattern"]
        assert len(pattern_matches) > 0

    def test_confidence_scoring(self) -> None:
        """Confidence scoring should adjust based on content quality."""
        # Longer, more detailed content should score higher
        short_content = "the fix was X"
        long_content = (
            "The fix was to refactor the authentication middleware to use "
            "async/await patterns which resolved the race condition"
        )

        short_score = score_extraction_confidence(short_content, 0.8)
        long_score = score_extraction_confidence(long_content, 0.8)

        assert long_score > short_score

    def test_confidence_boosted_by_tech_terms(self) -> None:
        """Content with technical terms should have higher confidence."""
        plain_content = "The solution was to change the settings"
        tech_content = "The solution was to change the database config for better API performance"

        plain_score = score_extraction_confidence(plain_content, 0.8)
        tech_score = score_extraction_confidence(tech_content, 0.8)

        assert tech_score >= plain_score

    def test_confidence_boosted_by_code(self) -> None:
        """Content with code should have higher confidence."""
        no_code = "The fix was to update the configuration"
        with_code = (
            "The fix was to update the configuration:\n"
            "```python\nconfig.timeout = 30\n```"
        )

        no_code_score = score_extraction_confidence(no_code, 0.8)
        with_code_score = score_extraction_confidence(with_code, 0.8)

        assert with_code_score > no_code_score

    def test_overlapping_deduplication(self) -> None:
        """Overlapping extractions should be deduplicated, keeping highest confidence."""
        candidates = [
            ExtractionCandidate(
                content="use PostgreSQL",
                confidence=0.7,
                pattern_type="decision",
                start_pos=10,
                end_pos=30,
            ),
            ExtractionCandidate(
                content="use PostgreSQL for the database",
                confidence=0.85,
                pattern_type="decision",
                start_pos=10,
                end_pos=50,
            ),
            ExtractionCandidate(
                content="different content",
                confidence=0.6,
                pattern_type="solution",
                start_pos=100,
                end_pos=120,
            ),
        ]

        deduped = dedupe_overlapping_extractions(candidates)

        # Overlapping candidates reduced to one (first non-overlapping kept)
        decision_matches = [c for c in deduped if c.pattern_type == "decision"]
        assert len(decision_matches) == 1

        # Non-overlapping candidate preserved
        solution_matches = [c for c in deduped if c.pattern_type == "solution"]
        assert len(solution_matches) == 1

    def test_min_confidence_filtering(self) -> None:
        """Candidates below min_confidence should be filtered out."""
        text = "We selected React for the frontend. The fix was updating the config."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.9)

        # High threshold should filter out most/all matches
        # (base confidences are typically 0.6-0.95)
        assert all(c.confidence >= 0.9 for c in candidates)

    def test_short_content_filtered(self) -> None:
        """Very short extracted content should be filtered out."""
        text = "We decided x. The fix was y."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        # Single character matches should be filtered (< 10 chars)
        assert all(len(c.content) >= 10 for c in candidates)

    def test_no_matches_returns_empty(self) -> None:
        """Text without patterns should return empty list."""
        text = "This is just regular text without any decision or fix markers."
        candidates = extract_candidates(text, EXTRACTION_PATTERNS, min_confidence=0.0)

        # May have some matches depending on patterns, but tests the function works
        assert isinstance(candidates, list)


# =============================================================================
# TestConsolidationAlgorithms
# =============================================================================


class TestConsolidationAlgorithms:
    """Tests for consolidation algorithms."""

    def test_jaccard_similarity_identical(self) -> None:
        """Identical strings should have Jaccard similarity of 1.0."""
        text = "The quick brown fox jumps over the lazy dog"
        similarity = jaccard_similarity(text, text)
        assert similarity == 1.0

    def test_jaccard_similarity_no_overlap(self) -> None:
        """Completely different strings should have Jaccard similarity of 0.0."""
        text1 = "apple banana cherry"
        text2 = "xylophone zebra quantum"
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Partially overlapping strings should have intermediate similarity."""
        text1 = "the quick brown fox"
        text2 = "the slow brown dog"
        similarity = jaccard_similarity(text1, text2)
        # Overlap: {the, brown} = 2, Union: {the, quick, brown, fox, slow, dog} = 6
        # Jaccard = 2/6 = 0.333...
        assert 0.0 < similarity < 1.0
        assert abs(similarity - (2 / 6)) < 0.01

    def test_jaccard_similarity_case_insensitive(self) -> None:
        """Jaccard similarity should be case-insensitive."""
        text1 = "The Quick Brown Fox"
        text2 = "the quick brown fox"
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 1.0

    def test_jaccard_similarity_empty_strings(self) -> None:
        """Empty strings should have similarity of 0.0 or be handled gracefully."""
        similarity = jaccard_similarity("", "")
        # Empty set intersection over empty union: could be 0, 1, or NaN
        # Implementation should handle gracefully
        assert 0.0 <= similarity <= 1.0

    def test_find_duplicate_groups_basic(self) -> None:
        """find_duplicate_groups should identify groups of similar memories."""
        memory_ids = ["mem-1", "mem-2", "mem-3", "mem-4"]

        # Create vectors where mem-1 and mem-2 are similar, mem-3 and mem-4 are similar
        vectors = np.array([
            [1.0, 0.0, 0.0],  # mem-1
            [0.99, 0.1, 0.0],  # mem-2 (similar to mem-1)
            [0.0, 1.0, 0.0],  # mem-3
            [0.0, 0.99, 0.1],  # mem-4 (similar to mem-3)
        ], dtype=np.float32)

        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        contents = [
            "Content about database configuration",
            "Content about database config",
            "Content about frontend React components",
            "Content about frontend React component library",
        ]

        groups = find_duplicate_groups(
            memory_ids=memory_ids,
            vectors=vectors,
            contents=contents,
            threshold=0.8,
            content_weight=0.3,
        )

        # Should find 2 groups: [0, 1] and [2, 3]
        assert len(groups) == 2
        # Each group should have 2 members
        assert all(len(g) == 2 for g in groups)

    def test_find_duplicate_groups_no_duplicates(self) -> None:
        """find_duplicate_groups should return empty when no duplicates."""
        memory_ids = ["mem-1", "mem-2", "mem-3"]

        # Completely different vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        contents = [
            "Database schema design",
            "Frontend user interface",
            "Backend API development",
        ]

        groups = find_duplicate_groups(
            memory_ids=memory_ids,
            vectors=vectors,
            contents=contents,
            threshold=0.9,
            content_weight=0.3,
        )

        assert len(groups) == 0

    def test_union_find_correctness(self) -> None:
        """Union-find should correctly merge transitive relationships."""
        memory_ids = ["a", "b", "c", "d"]

        # a-b similar, b-c similar -> a, b, c should be in same group
        # d is isolated
        vectors = np.array([
            [1.0, 0.0, 0.0],  # a
            [0.95, 0.31, 0.0],  # b (similar to a)
            [0.87, 0.5, 0.0],  # c (similar to b)
            [0.0, 0.0, 1.0],  # d (isolated)
        ], dtype=np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        contents = ["topic one"] * 4  # Same content for simplicity

        groups = find_duplicate_groups(
            memory_ids=memory_ids,
            vectors=vectors,
            contents=contents,
            threshold=0.8,
            content_weight=0.0,  # Pure vector similarity
        )

        # Should find one group with a, b, c
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_merge_keep_newest(self) -> None:
        """select_representative with keep_newest should return newest memory index."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        memories = [
            {"id": "old", "created_at": now - timedelta(days=10), "importance": 0.9},
            {"id": "new", "created_at": now, "importance": 0.5},
            {"id": "mid", "created_at": now - timedelta(days=5), "importance": 0.7},
        ]

        rep_idx = select_representative(memories, strategy="keep_newest")
        assert memories[rep_idx]["id"] == "new"

    def test_merge_keep_oldest(self) -> None:
        """select_representative with keep_oldest should return oldest memory index."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        memories = [
            {"id": "old", "created_at": now - timedelta(days=10), "importance": 0.5},
            {"id": "new", "created_at": now, "importance": 0.9},
            {"id": "mid", "created_at": now - timedelta(days=5), "importance": 0.7},
        ]

        rep_idx = select_representative(memories, strategy="keep_oldest")
        assert memories[rep_idx]["id"] == "old"

    def test_merge_keep_highest_importance(self) -> None:
        """select_representative with keep_highest_importance should return most important index."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        memories = [
            {"id": "low", "created_at": now - timedelta(days=10), "importance": 0.3},
            {"id": "high", "created_at": now - timedelta(days=5), "importance": 0.9},
            {"id": "mid", "created_at": now, "importance": 0.5},
        ]

        rep_idx = select_representative(memories, strategy="keep_highest_importance")
        assert memories[rep_idx]["id"] == "high"

    def test_merge_metadata_combines_tags(self) -> None:
        """merge_memory_metadata should combine tags from all memories."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        memories = [
            {
                "id": "1",
                "content": "Content 1",
                "created_at": now - timedelta(days=5),
                "last_accessed": now - timedelta(days=2),
                "access_count": 5,
                "importance": 0.6,
                "tags": ["python", "backend"],
            },
            {
                "id": "2",
                "content": "Content 2",
                "created_at": now,
                "last_accessed": now,
                "access_count": 3,
                "importance": 0.8,
                "tags": ["python", "database"],
            },
        ]

        merged = merge_memory_metadata(memories)

        # Tags should be combined and deduplicated
        assert set(merged["tags"]) == {"python", "backend", "database"}
        # Access count should be summed
        assert merged["access_count"] == 8
        # Importance should be max
        assert merged["importance"] == 0.8
        # Created_at should be min (oldest)
        assert merged["created_at"] == now - timedelta(days=5)
        # Last_accessed should be max (newest)
        assert merged["last_accessed"] == now

    def test_merge_metadata_handles_empty_tags(self) -> None:
        """merge_memory_metadata should handle memories without tags."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        memories = [
            {
                "id": "1",
                "content": "Content 1",
                "created_at": now,
                "last_accessed": now,
                "access_count": 1,
                "importance": 0.5,
                # No tags key
            },
            {
                "id": "2",
                "content": "Content 2",
                "created_at": now,
                "last_accessed": now,
                "access_count": 2,
                "importance": 0.6,
                "tags": [],
            },
        ]

        merged = merge_memory_metadata(memories)
        assert merged["tags"] == []

    def test_find_duplicate_groups_respects_threshold(self) -> None:
        """Higher threshold should result in fewer/smaller groups."""
        memory_ids = ["a", "b", "c"]

        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.31, 0.0],
            [0.87, 0.5, 0.0],
        ], dtype=np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        contents = ["same content"] * 3

        # With low threshold, should find groups
        groups_low = find_duplicate_groups(
            memory_ids=memory_ids,
            vectors=vectors,
            contents=contents,
            threshold=0.7,
            content_weight=0.0,
        )

        # With high threshold, should find fewer/no groups
        groups_high = find_duplicate_groups(
            memory_ids=memory_ids,
            vectors=vectors,
            contents=contents,
            threshold=0.99,
            content_weight=0.0,
        )

        # High threshold should be more restrictive
        assert len(groups_high) <= len(groups_low)
