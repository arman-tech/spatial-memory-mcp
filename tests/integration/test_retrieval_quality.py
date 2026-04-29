"""Single-model retrieval-quality eval (US-008, spec §4.6, R8 ε-tolerance gate).

Default ``pytest tests/`` does NOT execute this test — the ``slow`` marker
keeps it gated. Run explicitly:

    pytest tests/integration/test_retrieval_quality.py -v -m slow

Why a single model: registry-of-one (D1) — only ``arctic-m-long`` is a known
alias in v2.0, so the source SHA's parametrize-over-six-aliases shape is
collapsed to a single canonical run.

Why ε-tolerance instead of absolute floors: per R8 mitigation, embedding
non-determinism on long passages + minor library/seed shifts can move scores
by 1-2% between identical runs. Absolute floors would either be too tight
(flaky) or too loose (useless). Comparing against a committed baseline with
small ε absorbs numerical noise while still catching real regressions.
"""

from __future__ import annotations

import json
import math
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


EVAL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_eval_set.json"
BASELINE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_quality_baseline.json"
)

# Per R8 mitigation — these ε values are tight enough to catch real degradations
# (a 2-point nDCG drop or a single missed query in 15) but loose enough to absorb
# numerical noise from embedding non-determinism on long text.
EPS_NDCG = 0.02
EPS_MRR = 0.02
EPS_HIT_AT_K = 0.07  # one missed query in 15 = 0.067; ε just clears that


@pytest.fixture(autouse=True)
def _reset_after() -> Generator[None, None, None]:
    yield
    reset_settings()


def _ensure_dependency(module_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError:
        pytest.skip(f"{module_name!r} not available; nightly eval requires it")


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(retrieved_ids: list[str], ground_truth: set[str], k: int) -> float:
    gains = [1.0 if rid in ground_truth else 0.0 for rid in retrieved_ids[:k]]
    ideal = [1.0] * min(len(ground_truth), k)
    dcg = _dcg(gains)
    idcg = _dcg(ideal) if ideal else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(retrieved_ids: list[str], ground_truth: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def _hit_at_k(retrieved_ids: list[str], ground_truth: set[str], k: int) -> bool:
    return bool(set(retrieved_ids[:k]) & ground_truth)


def test_arctic_m_long_no_regression_beyond_epsilon(tmp_path: Path) -> None:
    """Single arctic-m-long run; assert no regression beyond ε vs committed baseline."""
    _ensure_dependency("sentence_transformers")

    eval_set = json.loads(EVAL_FIXTURE.read_text(encoding="utf-8"))
    memories: list[dict[str, str]] = eval_set["memories"]
    queries: list[dict[str, object]] = eval_set["queries"]
    assert len(memories) == 30, "fixture corpus must be exactly 30 memories"
    assert len(queries) == 15, "fixture must have exactly 15 queries"

    settings = Settings(
        memory_path=tmp_path / "db-arctic",
        embedding_model="arctic-m-long",
        filelock_enabled=False,
        auto_create_indexes=False,
        enable_fts_index=False,
    )
    override_settings(settings)

    factory = ServiceFactory(settings)
    container = factory.create_all()
    try:
        original_to_new: dict[str, str] = {}
        for m in memories:
            new = container.memory.remember(
                content=m["content"], namespace="eval", importance=0.5
            )
            original_to_new[m["id"]] = new.id if hasattr(new, "id") else str(new)
        new_to_original = {v: k for k, v in original_to_new.items()}

        ndcg_scores: list[float] = []
        mrr_scores: list[float] = []
        hits_at_5 = 0
        for q in queries:
            result = container.memory.recall(query=q["text"], limit=10)  # type: ignore[arg-type]
            retrieved_ids = [
                new_to_original.get(m.id, "") for m in result.memories
            ]
            ground_truth = set(q["ground_truth_ids"])  # type: ignore[arg-type]
            ndcg_scores.append(_ndcg_at_k(retrieved_ids, ground_truth, 10))
            mrr_scores.append(_mrr(retrieved_ids, ground_truth))
            if _hit_at_k(retrieved_ids, ground_truth, 5):
                hits_at_5 += 1

        measured = {
            "nDCG@10": sum(ndcg_scores) / len(ndcg_scores),
            "MRR": sum(mrr_scores) / len(mrr_scores),
            "hit@5": hits_at_5 / len(queries),
        }

        if not BASELINE_FIXTURE.exists():
            pytest.fail(
                f"Baseline file missing at {BASELINE_FIXTURE}.\n"
                f"Measured this run:\n"
                f"  nDCG@10 = {measured['nDCG@10']:.4f}\n"
                f"  MRR     = {measured['MRR']:.4f}\n"
                f"  hit@5   = {measured['hit@5']:.4f}\n"
                f"Commit a baseline JSON with these values to enable regression gating."
            )

        baseline = json.loads(BASELINE_FIXTURE.read_text(encoding="utf-8"))
        regressions: list[str] = []
        if measured["nDCG@10"] < baseline["nDCG@10"] - EPS_NDCG:
            regressions.append(
                f"nDCG@10: measured {measured['nDCG@10']:.4f} < "
                f"baseline {baseline['nDCG@10']:.4f} - ε={EPS_NDCG}"
            )
        if measured["MRR"] < baseline["MRR"] - EPS_MRR:
            regressions.append(
                f"MRR: measured {measured['MRR']:.4f} < "
                f"baseline {baseline['MRR']:.4f} - ε={EPS_MRR}"
            )
        if measured["hit@5"] < baseline["hit@5"] - EPS_HIT_AT_K:
            regressions.append(
                f"hit@5: measured {measured['hit@5']:.4f} < "
                f"baseline {baseline['hit@5']:.4f} - ε={EPS_HIT_AT_K}"
            )
        if regressions:
            pytest.fail(
                "Retrieval quality regression beyond ε:\n  "
                + "\n  ".join(regressions)
                + f"\n\nMeasured: {measured}\nBaseline: {baseline}"
            )
    finally:
        if container.database is not None:
            container.database.close()
