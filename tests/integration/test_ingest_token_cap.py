"""Integration tests for IngestPipeline server-side token cap (US-020 / §8.5).

End-to-end coverage of the §8.5 contract using the real arctic-m-long
embedder + LanceDB repository:

* AC 5: a 50K-char prompt is ingested successfully — the embedding is
  generated from <= max_seq_length tokens, and ``capture_cap_hit_total``
  appears in the JSONL metrics journal.

The unit suite (``tests/unit/test_ingest_pipeline.py``) already pins the
fine-grained truncation arithmetic on a mocked embedder; here we verify the
production wiring (real tokenizer, real registry contract, real journal
writer) actually fires.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.metrics_journal import set_journal_dir
from spatial_memory.services.ingest_pipeline import IngestConfig, IngestPipeline


def _make_token_dense_prompt(target_chars: int = 50_000, seed: int = 42) -> str:
    """Build a ``target_chars`` prompt of random alphanumeric tokens.

    Uniform-random short words defeat BPE merging so the resulting prompt
    decodes to many more tokens than typical English would — guaranteeing
    we exceed arctic-m-long's 8K-token cap with a single ingest call.
    """
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    pieces: list[str] = []
    total = 0
    while total < target_chars:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 8)))
        pieces.append(word)
        total += len(word) + 1
    return " ".join(pieces)[:target_chars]


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


@pytest.mark.integration
class TestIngestTokenCapEndToEnd:
    """AC 5: 50K-char prompt round-trip with real embedder + journal."""

    def test_50k_char_prompt_truncated_and_capture_cap_hit_journaled(
        self,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
        journal_dir: Path,
    ) -> None:
        """A 50K-char prompt is capped to <= max_seq_length tokens AND
        the truncation event is recorded in the journal."""
        pipeline = IngestPipeline(
            repository=module_repository, embeddings=session_embedding_service
        )

        # 50K chars of token-dense word soup — well past arctic-m-long's 8K-token cap.
        prompt = _make_token_dense_prompt(50_000)
        assert len(prompt) == 50_000

        result = pipeline.ingest(
            content=prompt,
            namespace="default",
            metadata={"tool": "remember"},
            config=IngestConfig(cognitive_offloading_enabled=False),
        )

        assert result.status == "stored"
        assert result.id, "Memory must be persisted"

        # The stored embedding must match the embedder's dimensions
        # (NOT some degraded shape — confirms the capped text actually
        # round-tripped through detokenize → embed_passage).
        stored = module_repository.get(result.id)
        assert stored is not None
        # Vector retrieval lives on the result, not the Memory model — but
        # the embedding lifecycle is exercised end-to-end by ingest succeeding.
        # We verify the cap fired via the journal instead.

        observations = _read_journal_observations(journal_dir)
        assert observations, "Journal must contain at least one observation"

        cap_hits = [o for o in observations if o["metric"] == "capture_cap_hit_total"]
        sizes = [o for o in observations if o["metric"] == "capture_size_tokens"]

        assert cap_hits, (
            "capture_cap_hit_total must appear in journal — 50K-char prompt "
            "exceeds 8K-token cap so the cap-hit counter must fire"
        )
        assert cap_hits[0]["labels"] == {"tool": "remember"}, (
            f"cap-hit labels = {cap_hits[0]['labels']}; expected {{'tool': 'remember'}}"
        )

        # Pre-cap and post-cap sizes must both be journaled
        stages = {o["labels"].get("stage") for o in sizes}
        assert "pre_cap" in stages and "post_cap" in stages

        # post_cap value must be <= max_seq_length (the registry contract)
        post_cap = next(o for o in sizes if o["labels"].get("stage") == "post_cap")
        max_len = session_embedding_service.max_seq_length
        assert post_cap["value"] <= max_len, (
            f"post_cap tokens={post_cap['value']} exceeds max_seq_length={max_len}"
        )
        # And pre_cap must be strictly larger (truncation actually happened)
        pre_cap = next(o for o in sizes if o["labels"].get("stage") == "pre_cap")
        assert pre_cap["value"] > post_cap["value"], (
            f"pre_cap={pre_cap['value']} not > post_cap={post_cap['value']}; "
            "truncation did not fire"
        )

    def test_short_prompt_no_cap_hit(
        self,
        module_repository: LanceDBMemoryRepository,
        session_embedding_service: EmbeddingService,
        journal_dir: Path,
    ) -> None:
        """A short prompt below the cap journals pre/post sizes but no cap-hit."""
        pipeline = IngestPipeline(
            repository=module_repository, embeddings=session_embedding_service
        )
        result = pipeline.ingest(
            content="A short note about spatial navigation.",
            namespace="default",
            metadata={"tool": "remember"},
            config=IngestConfig(cognitive_offloading_enabled=False),
        )
        assert result.status == "stored"

        observations = _read_journal_observations(journal_dir)
        cap_hits = [o for o in observations if o["metric"] == "capture_cap_hit_total"]
        sizes = [o for o in observations if o["metric"] == "capture_size_tokens"]

        assert cap_hits == [], "Short prompt must not trigger capture_cap_hit_total"
        stages = {o["labels"].get("stage") for o in sizes}
        assert "pre_cap" in stages and "post_cap" in stages, (
            "Pre/post sizes must journal even when no truncation fires"
        )
        # And on a short prompt pre == post (no truncation)
        pre = next(o for o in sizes if o["labels"].get("stage") == "pre_cap")["value"]
        post = next(o for o in sizes if o["labels"].get("stage") == "post_cap")["value"]
        assert pre == post, f"pre={pre} != post={post}; truncation should not have fired"
