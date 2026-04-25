"""Nightly retrieval-quality eval (PRD §9, spec §7.2 T5).

Parameterized recall@5 across the registry. Default ``pytest tests/`` does NOT
execute these tests — the ``slow`` marker keeps them gated. Run explicitly:

    pytest tests/integration/test_retrieval_quality.py -v -m slow
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.factory import ServiceFactory

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.requires_model,
]


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_eval_set.json"

# Per spec §7.2 / [DECISION] D10 — conservative starting floors. Recalibrate
# after first real nightly run if a model lands consistently above its floor.
MINIMUM_FLOORS: dict[str, float] = {
    "arctic-m-long": 0.75,
    "nomic": 0.70,
    "arctic-s": 0.60,
    "arctic-xs": 0.55,
    "bge-small": 0.60,
    "minilm": 0.40,
}


@pytest.fixture(autouse=True)
def _reset_after() -> Generator[None, None, None]:
    yield
    reset_settings()


def _ensure_dependency(module_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError:
        pytest.skip(f"{module_name!r} not available in this sandbox; nightly eval requires it")


@pytest.mark.parametrize("alias", list(MINIMUM_FLOORS.keys()))
def test_recall_at_5(alias: str, tmp_path: Path) -> None:
    """For each registry alias: ingest 30 memories, run 15 queries, assert recall@5 floor."""
    _ensure_dependency("sentence_transformers")

    eval_set = json.loads(FIXTURE.read_text(encoding="utf-8"))
    memories: list[dict[str, str]] = eval_set["memories"]
    queries: list[dict[str, object]] = eval_set["queries"]
    assert len(memories) == 30, "fixture corpus must be exactly 30 memories"
    assert len(queries) == 15, "fixture must have exactly 15 queries"

    settings = Settings(
        memory_path=tmp_path / f"db-{alias}",
        embedding_model=alias,
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )
    override_settings(settings)

    factory = ServiceFactory(settings)
    container = factory.create_all()

    try:
        # Ingest all eval memories. Use the original ids for traceability.
        original_to_new: dict[str, str] = {}
        for m in memories:
            new_id = container.memory.remember(
                content=m["content"],
                namespace="eval",
                importance=0.5,
            )
            original_to_new[m["id"]] = new_id.id if hasattr(new_id, "id") else str(new_id)

        # Run all queries; compute recall@5.
        hits = 0
        for q in queries:
            result = container.memory.recall(query=q["text"], limit=5)  # type: ignore[arg-type]
            top5_originals = set()
            for match in result.memories:
                # Reverse-map repository ids back to original fixture ids.
                for orig, new in original_to_new.items():
                    if new == match.id:
                        top5_originals.add(orig)
                        break
            ground_truth = set(q["ground_truth_ids"])  # type: ignore[arg-type]
            if ground_truth & top5_originals:
                hits += 1

        recall_at_5 = hits / len(queries)
        floor = MINIMUM_FLOORS[alias]
        assert recall_at_5 >= floor, (
            f"{alias}: recall@5 = {recall_at_5:.3f} < floor {floor:.3f} "
            f"({hits}/{len(queries)} queries hit). "
            "Investigate model swap, prefix routing, or fixture drift."
        )
    finally:
        if container.database is not None:
            container.database.close()
