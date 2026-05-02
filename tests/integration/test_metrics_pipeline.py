"""Integration test for v2.0 metrics pipeline (spec §10.4 verification gate L719).

Pins the §10.4 contract end-to-end with the real arctic-m-long embedder +
LanceDB repository: a recall + a reinforce must produce non-empty
``recall_result_count`` and ``reinforced_memory_rank`` observations in the
JSONL metrics journal.

The unit suite (``tests/unit/test_metrics.py``,
``tests/unit/test_metrics_journal.py``) already pins the helpers in isolation;
here we exercise the production wiring so a regression in
``MemoryService.recall`` or ``LifecycleService.reinforce`` (e.g. someone
removes the ``record_*`` call) is caught at the integration layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.metrics_journal import set_journal_dir
from spatial_memory.services.lifecycle import LifecycleService
from spatial_memory.services.memory import MemoryService

pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


@pytest.fixture
def journal_dir(tmp_path: Path) -> Path:
    """Pin the metrics journal to a tmp directory for this test only."""
    target = tmp_path / "metrics"
    target.mkdir()
    set_journal_dir(target)
    return target


def _read_journal_observations(journal_dir: Path) -> list[dict]:
    """Slurp every JSONL line written under ``journal_dir`` into dicts."""
    out: list[dict] = []
    for jsonl in journal_dir.glob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_recall_then_reinforce_emits_v2_metrics_to_journal(
    module_repository: LanceDBMemoryRepository,
    session_embedding_service: EmbeddingService,
    journal_dir: Path,
) -> None:
    """§10.4 L719: recall + reinforce produce non-empty observations.

    Builds real services against the module-scoped repository + session-scoped
    embedder, seeds a small corpus, recalls against it, then reinforces the
    returned IDs in recall order. The journal must contain at least one
    ``recall_result_count`` and one ``reinforced_memory_rank`` line.
    """
    memory = MemoryService(
        repository=module_repository,
        embeddings=session_embedding_service,
    )
    lifecycle = LifecycleService(
        repository=module_repository,
        embeddings=session_embedding_service,
    )

    seeds = [
        "Configure the embedding model via SPATIAL_MEMORY_EMBEDDING_MODEL.",
        "v2.0 ships a registry-of-one — only arctic-m-long is supported.",
        "Hybrid recall combines vector similarity with BM25 keyword search.",
        "LanceDB stores vectors in Parquet fragments under the memory path.",
        "Reinforce boosts importance and updates the access timestamp.",
    ]
    for content in seeds:
        memory.remember(content=content, namespace="default", importance=0.5)

    # Real recall — wires record_recall_result_count → journal_observation.
    result = memory.recall(
        query="how do I configure the embedding model?",
        limit=3,
        namespace="default",
    )
    recalled_ids = [m.id for m in result.memories]
    assert recalled_ids, "Recall must return at least one memory against the seeded corpus"

    # Reinforce in recall order — wires record_reinforced_memory_rank per ID.
    reinforce_result = lifecycle.reinforce(memory_ids=recalled_ids)
    assert reinforce_result.memories_reinforced == len(recalled_ids), (
        f"Expected all {len(recalled_ids)} recalled IDs to reinforce; "
        f"got {reinforce_result.memories_reinforced}"
    )

    # Journal must contain both v2.0 metrics emitted by the pipeline above.
    observations = _read_journal_observations(journal_dir)
    assert observations, "Journal must contain at least one observation after recall+reinforce"

    recall_counts = [o for o in observations if o["metric"] == "recall_result_count"]
    rank_obs = [o for o in observations if o["metric"] == "reinforced_memory_rank"]

    assert recall_counts, (
        "recall_result_count missing from journal — MemoryService.recall must call "
        "record_recall_result_count per spec §10.2 / D7"
    )
    # Exactly one recall call → exactly one count observation, equal to len(recalled).
    assert len(recall_counts) == 1, (
        f"Expected 1 recall_result_count observation, got {len(recall_counts)}"
    )
    assert recall_counts[0]["value"] == len(recalled_ids), (
        f"recall_result_count={recall_counts[0]['value']} != len(recalled)="
        f"{len(recalled_ids)}; histogram value must equal returned-result count"
    )

    assert rank_obs, (
        "reinforced_memory_rank missing from journal — LifecycleService.reinforce must "
        "call record_reinforced_memory_rank per spec §10.2 / D7"
    )
    # One rank observation per reinforced memory; ranks are 1-indexed positions
    # of memory_ids in the recall-order list passed to reinforce.
    assert len(rank_obs) == len(recalled_ids), (
        f"Expected {len(recalled_ids)} reinforced_memory_rank observations, "
        f"got {len(rank_obs)}"
    )
    observed_ranks = sorted(int(o["value"]) for o in rank_obs)
    expected_ranks = list(range(1, len(recalled_ids) + 1))
    assert observed_ranks == expected_ranks, (
        f"Observed ranks {observed_ranks} != expected 1-indexed positions {expected_ranks}; "
        "callers preserve recall order so ranks must be a contiguous 1..N range"
    )
