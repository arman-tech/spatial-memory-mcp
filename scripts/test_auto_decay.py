#!/usr/bin/env python
"""Manual test script to verify auto-decay feature is working.

Run from the project root:
    python scripts/test_auto_decay.py

This script demonstrates:
1. Decay calculation based on memory age
2. Access count slowing decay
3. Re-ranking of results by effective importance
"""

import sys
from pathlib import Path

# Ensure we use the local development version, not installed package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import timedelta

from spatial_memory.core.models import AutoDecayConfig
from spatial_memory.core.utils import utc_now
from spatial_memory.services.decay_manager import DecayManager


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_basic_decay() -> None:
    """Test that older memories have lower effective importance."""
    print_header("Test 1: Basic Decay by Age")

    config = AutoDecayConfig(
        enabled=True,
        persist_enabled=False,
        half_life_days=30.0,
        min_importance_floor=0.1,
        access_weight=0.0,  # Disable access count effect for this test
    )
    decay_manager = DecayManager(repository=None, config=config)

    now = utc_now()
    results = [
        {
            "id": "1",
            "content": "Fresh memory (just now)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now,
            "access_count": 0,
        },
        {
            "id": "2",
            "content": "30-day old memory (1 half-life)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now - timedelta(days=30),
            "access_count": 0,
        },
        {
            "id": "3",
            "content": "60-day old memory (2 half-lives)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now - timedelta(days=60),
            "access_count": 0,
        },
        {
            "id": "4",
            "content": "90-day old memory (3 half-lives)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now - timedelta(days=90),
            "access_count": 0,
        },
    ]

    processed = decay_manager.apply_decay_to_results(results, rerank=False)

    print("All memories have same similarity (0.9) and stored importance (1.0)")
    print("Half-life: 30 days\n")

    for r in processed:
        eff = r["effective_importance"]
        decay_pct = (1.0 - eff) * 100
        print(f"  {r['content']}")
        print(f"    effective_importance: {eff:.4f} ({decay_pct:.1f}% decayed)")
        print()

    # Verify decay is working
    assert processed[0]["effective_importance"] > processed[1]["effective_importance"]
    assert processed[1]["effective_importance"] > processed[2]["effective_importance"]
    assert processed[2]["effective_importance"] > processed[3]["effective_importance"]
    print("PASS: Older memories have lower effective importance")


def test_access_count_slows_decay() -> None:
    """Test that frequently accessed memories decay slower."""
    print_header("Test 2: Access Count Slows Decay")

    config = AutoDecayConfig(
        enabled=True,
        persist_enabled=False,
        half_life_days=30.0,
        min_importance_floor=0.1,
        access_weight=0.3,  # Each access adds 30% to half-life
    )
    decay_manager = DecayManager(repository=None, config=config)

    now = utc_now()
    past = now - timedelta(days=30)  # 30 days ago

    results = [
        {
            "id": "1",
            "content": "Rarely accessed (0 times)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": past,
            "access_count": 0,
        },
        {
            "id": "2",
            "content": "Sometimes accessed (10 times)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": past,
            "access_count": 10,
        },
        {
            "id": "3",
            "content": "Frequently accessed (50 times)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": past,
            "access_count": 50,
        },
    ]

    processed = decay_manager.apply_decay_to_results(results, rerank=False)

    print("All memories are 30 days old with same similarity and stored importance")
    print("access_weight: 0.3 (each access adds 30% to effective half-life)\n")

    for r in processed:
        eff = r["effective_importance"]
        access = r["access_count"]
        effective_half_life = 30.0 * (1.0 + 0.3 * access)
        print(f"  {r['content']}")
        print(f"    access_count: {access}")
        print(f"    effective_half_life: {effective_half_life:.1f} days")
        print(f"    effective_importance: {eff:.4f}")
        print()

    # Verify access count slows decay
    assert processed[2]["effective_importance"] > processed[1]["effective_importance"]
    assert processed[1]["effective_importance"] > processed[0]["effective_importance"]
    print("PASS: Frequently accessed memories decay slower")


def test_reranking() -> None:
    """Test that results are re-ranked by similarity * effective_importance."""
    print_header("Test 3: Re-ranking by Adjusted Score")

    config = AutoDecayConfig(
        enabled=True,
        persist_enabled=False,
        half_life_days=30.0,
        min_importance_floor=0.1,
        access_weight=0.0,
    )
    decay_manager = DecayManager(repository=None, config=config)

    now = utc_now()

    # Old memory has higher similarity but will be outranked due to decay
    results = [
        {
            "id": "old",
            "content": "Old but highly similar",
            "similarity": 0.95,  # Higher similarity
            "importance": 1.0,
            "last_accessed": now - timedelta(days=60),  # 60 days old
            "access_count": 0,
        },
        {
            "id": "new",
            "content": "Fresh but less similar",
            "similarity": 0.70,  # Lower similarity
            "importance": 1.0,
            "last_accessed": now,  # Just accessed
            "access_count": 0,
        },
    ]

    print("Before re-ranking (by similarity):")
    print(f"  1. Old memory: similarity=0.95")
    print(f"  2. Fresh memory: similarity=0.70\n")

    processed = decay_manager.apply_decay_to_results(results, rerank=True)

    print("After re-ranking (by similarity * effective_importance):")
    for i, r in enumerate(processed, 1):
        adj_score = r["similarity"] * r["effective_importance"]
        print(f"  {i}. {r['content']}")
        print(f"     similarity: {r['similarity']:.2f}")
        print(f"     effective_importance: {r['effective_importance']:.4f}")
        print(f"     adjusted_score: {adj_score:.4f}")
        print()

    # The fresh memory should now rank first despite lower similarity
    assert processed[0]["id"] == "new", "Fresh memory should rank first after decay"
    print("PASS: Fresh memory outranks old memory after decay adjustment")


def test_minimum_floor() -> None:
    """Test that importance never goes below the minimum floor."""
    print_header("Test 4: Minimum Importance Floor")

    config = AutoDecayConfig(
        enabled=True,
        persist_enabled=False,
        half_life_days=30.0,
        min_importance_floor=0.1,  # 10% floor
        access_weight=0.0,
    )
    decay_manager = DecayManager(repository=None, config=config)

    now = utc_now()

    results = [
        {
            "id": "1",
            "content": "Very old memory (1 year)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now - timedelta(days=365),
            "access_count": 0,
        },
        {
            "id": "2",
            "content": "Ancient memory (5 years)",
            "similarity": 0.9,
            "importance": 1.0,
            "last_accessed": now - timedelta(days=365 * 5),
            "access_count": 0,
        },
    ]

    processed = decay_manager.apply_decay_to_results(results, rerank=False)

    print(f"Minimum importance floor: {config.min_importance_floor}")
    print()

    for r in processed:
        eff = r["effective_importance"]
        print(f"  {r['content']}")
        print(f"    effective_importance: {eff:.4f}")
        print()

    # Verify floor is respected
    for r in processed:
        assert r["effective_importance"] >= config.min_importance_floor
    print(f"PASS: All effective_importance values >= {config.min_importance_floor}")


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  AUTO-DECAY FEATURE VERIFICATION")
    print("=" * 60)

    test_basic_decay()
    test_access_count_slows_decay()
    test_reranking()
    test_minimum_floor()

    print_header("ALL TESTS PASSED!")
    print("The auto-decay feature is working correctly.\n")


if __name__ == "__main__":
    main()
