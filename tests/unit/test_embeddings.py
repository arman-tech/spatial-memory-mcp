"""Unit tests for EmbeddingService role-aware API (v2.0).

These tests mock the underlying model so they don't require downloading
real embedding weights; they verify the prefix/role/cache behavior
against the registry-of-one (arctic-m-long).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatial_memory.core.embedding_registry import resolve_model
from spatial_memory.core.embeddings import EmbeddingService

pytestmark = pytest.mark.unit


def _make_service(model_name: str = "arctic-m-long") -> EmbeddingService:
    """Construct an EmbeddingService and stub the underlying local model."""
    svc = EmbeddingService(model_name=model_name)
    fake_model = MagicMock()
    cfg = resolve_model(model_name)
    dim = cfg.dim or 768

    def encode(texts: list[str], **_: object) -> np.ndarray:
        # Distinct, deterministic vectors per text so cache hits round-trip
        # to the same value.
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**31 - 1)
            out[i] = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    fake_model.encode.side_effect = encode
    fake_model.get_sentence_embedding_dimension.return_value = dim
    fake_model.max_seq_length = cfg.max_seq_length
    svc._model = fake_model
    svc._dimensions = dim
    svc._active_backend = "pytorch"
    return svc


def test_config_property_resolves_registry_entry() -> None:
    svc = _make_service("arctic-m-long")
    assert svc.config.alias == "arctic-m-long"
    assert svc.config.dim == 768
    assert svc.config.max_seq_length == 8192


def test_embed_query_prepends_query_prefix() -> None:
    svc = _make_service("arctic-m-long")
    svc.embed_query("how do I reset my password?")
    args, _kwargs = svc._model.encode.call_args
    sent = args[0]
    assert sent == [
        "Represent this sentence for searching relevant passages: how do I reset my password?"
    ]


def test_embed_passage_uses_empty_prefix_for_arctic() -> None:
    """arctic-m-long is asymmetric: passage_prefix is empty, raw text is encoded."""
    svc = _make_service("arctic-m-long")
    svc.embed_passage("the quick brown fox")
    args, _kwargs = svc._model.encode.call_args
    sent = args[0]
    assert sent == ["the quick brown fox"]


def test_cache_key_distinguishes_role() -> None:
    svc = _make_service("arctic-m-long")
    text = "shared text"
    qk = svc._get_cache_key(text, role="query")
    pk = svc._get_cache_key(text, role="passage")
    assert qk != pk


def test_role_aware_cache_does_not_cross_pollute() -> None:
    svc = _make_service("arctic-m-long")
    text = "vector retrieval explained"
    q_vec = svc.embed_query(text)
    p_vec = svc.embed_passage(text)
    # The encode payloads differed (prefixed vs raw), so vectors differ —
    # this checks that we did NOT serve the query vector from the passage
    # cache slot.
    assert not np.array_equal(q_vec, p_vec)


def test_embed_passage_cache_hit_skips_model_call() -> None:
    svc = _make_service("arctic-m-long")
    svc.embed_passage("hello")
    call_count_after_first = svc._model.encode.call_count
    svc.embed_passage("hello")
    # Second call must come from cache.
    assert svc._model.encode.call_count == call_count_after_first


def test_legacy_embed_delegates_to_passage() -> None:
    svc = _make_service("arctic-m-long")
    a = svc.embed("foo")
    b = svc.embed_passage("foo")
    assert np.array_equal(a, b)


def test_legacy_embed_batch_delegates_to_passage_batch() -> None:
    svc = _make_service("arctic-m-long")
    a = svc.embed_batch(["foo", "bar"])
    b = svc.embed_passage_batch(["foo", "bar"])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_embed_batch_empty_returns_empty_list() -> None:
    svc = _make_service("arctic-m-long")
    assert svc.embed_query_batch([]) == []
    assert svc.embed_passage_batch([]) == []


def test_max_seq_length_set_after_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Loading an arctic-m-long model must set max_seq_length=8192."""
    from spatial_memory.core import embeddings as emb_mod

    fake_st_module = MagicMock()
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = 768
    fake_model.max_seq_length = 512  # default — will be overridden
    fake_st_module.SentenceTransformer.return_value = fake_model

    with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
        with patch.object(emb_mod, "_detect_backend", return_value="pytorch"):
            svc = EmbeddingService(model_name="arctic-m-long")
            svc._load_local_model()

    assert fake_model.max_seq_length == 8192


def test_openai_alias_rejected_at_construction() -> None:
    """Registry-of-one rejects every name except arctic-m-long."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        EmbeddingService(model_name="openai:text-embedding-3-small")


def test_max_seq_length_property_reads_registry() -> None:
    svc = _make_service("arctic-m-long")
    assert svc.max_seq_length == 8192


def test_tokenize_detokenize_round_trip() -> None:
    svc = _make_service("arctic-m-long")
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.return_value = [101, 7592, 2088, 102]
    fake_tokenizer.decode.return_value = "hello world"
    svc._model.tokenizer = fake_tokenizer  # type: ignore[union-attr]

    ids = svc.tokenize("hello world")
    assert ids == [101, 7592, 2088, 102]
    fake_tokenizer.encode.assert_called_once_with("hello world", add_special_tokens=False)

    text = svc.detokenize(ids)
    assert text == "hello world"
    fake_tokenizer.decode.assert_called_once_with(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# US-021 / §10.1 — every embed call records latency with model="arctic-m-long".
# The wire-up was always present at core/embeddings.py:_embed_local; US-021
# pins the contract under a unit test.
# ---------------------------------------------------------------------------


def test_embed_query_records_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.1: embed_query produces exactly one record_embedding_latency call."""
    captured: list[tuple[float, str]] = []
    monkeypatch.setattr(
        "spatial_memory.core.embeddings.record_embedding_latency",
        lambda duration, model: captured.append((duration, model)),
    )

    svc = _make_service("arctic-m-long")
    svc.embed_query("hello")

    assert len(captured) == 1
    duration, model = captured[0]
    assert duration >= 0
    assert model == "arctic-m-long"


def test_embed_passage_records_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.1: embed_passage also records latency."""
    captured: list[tuple[float, str]] = []
    monkeypatch.setattr(
        "spatial_memory.core.embeddings.record_embedding_latency",
        lambda duration, model: captured.append((duration, model)),
    )

    svc = _make_service("arctic-m-long")
    svc.embed_passage("hello")

    assert len(captured) == 1
    assert captured[0][1] == "arctic-m-long"


def test_embed_batch_records_one_latency_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.1: a batched embed records ONE latency observation per batch, not per text.

    Anything else would double-count and skew the percentiles in §10.4.
    """
    captured: list[tuple[float, str]] = []
    monkeypatch.setattr(
        "spatial_memory.core.embeddings.record_embedding_latency",
        lambda duration, model: captured.append((duration, model)),
    )

    svc = _make_service("arctic-m-long")
    svc.embed_passage_batch(["a", "b", "c"])

    assert len(captured) == 1


def test_embed_query_journals_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.1 / AC6: one journal line per embed call regardless of prometheus availability."""
    captured: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        "spatial_memory.core.metrics_journal.journal_observation",
        lambda metric, value, labels: captured.append((metric, value, labels)),
    )

    svc = _make_service("arctic-m-long")
    svc.embed_query("hello")

    latency_lines = [c for c in captured if c[0] == "embedding_latency_seconds"]
    assert len(latency_lines) == 1
    metric, value, labels = latency_lines[0]
    assert value >= 0
    assert labels == {"model": "arctic-m-long"}


def test_from_config_constructs_and_warms_up() -> None:
    """from_config builds from Settings and runs a warm-up embed_query."""
    from spatial_memory.core import embeddings as emb_mod

    fake_st_module = MagicMock()
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = 768
    fake_model.max_seq_length = 8192

    def fake_encode(texts: list[str], **_: object) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i in range(len(texts)):
            out[i, 0] = 1.0
        return out

    fake_model.encode.side_effect = fake_encode
    fake_st_module.SentenceTransformer.return_value = fake_model

    from spatial_memory.config import Settings

    settings = Settings(embedding_model="arctic-m-long", embedding_backend="pytorch")
    with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
        with patch.object(emb_mod, "_detect_backend", return_value="pytorch"):
            svc = EmbeddingService.from_config(settings)

    assert svc.model_name == "arctic-m-long"
    assert svc.dimensions == 768
    # Warm-up should have invoked encode at least once with the query prefix.
    encode_args, _ = fake_model.encode.call_args
    assert encode_args[0][0].startswith("Represent this sentence for searching relevant passages:")


# ---------------------------------------------------------------------------
# US-022 / spec §7.1 — from_config wires set_journal_dir + cold-start gauge
#
# The cold-start metric is written by the warm-up embed_query call inside
# from_config. For the journal to land under settings.memory_path / 'metrics'
# (and not the env-var fallback), set_journal_dir MUST be called BEFORE the
# warmup. After the warmup, record_embedder_cold_start sets the gauge and
# adds one journal line.
# ---------------------------------------------------------------------------


def _build_st_mocks(dim: int = 768) -> tuple[MagicMock, MagicMock]:
    """Construct a (fake_module, fake_model) pair for sentence_transformers."""
    fake_st_module = MagicMock()
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = dim
    fake_model.max_seq_length = 8192

    def fake_encode(texts: list[str], **_: object) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i in range(len(texts)):
            out[i, 0] = 1.0
        return out

    fake_model.encode.side_effect = fake_encode
    fake_st_module.SentenceTransformer.return_value = fake_model
    return fake_st_module, fake_model


def test_from_config_calls_set_journal_dir_before_warmup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """AC3: set_journal_dir must run BEFORE embed_query('warmup').

    Order matters: the warm-up triggers record_embedding_latency + cold-start
    journal writes; both must land in settings.memory_path / 'metrics', not in
    whatever the env var or lazy fallback resolves to.
    """
    from spatial_memory.config import Settings
    from spatial_memory.core import embeddings as emb_mod

    fake_st_module, _fake_model = _build_st_mocks()

    call_log: list[str] = []
    monkeypatch.setattr(
        emb_mod,
        "set_journal_dir",
        lambda p: call_log.append(f"set_journal_dir:{p}"),
    )
    real_embed_query = emb_mod.EmbeddingService.embed_query

    def spy_embed_query(self, text):  # type: ignore[no-untyped-def]
        call_log.append(f"embed_query:{text}")
        return real_embed_query(self, text)

    monkeypatch.setattr(emb_mod.EmbeddingService, "embed_query", spy_embed_query)

    settings = Settings(
        memory_path=tmp_path,
        embedding_model="arctic-m-long",
        embedding_backend="pytorch",
    )
    with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
        with patch.object(emb_mod, "_detect_backend", return_value="pytorch"):
            emb_mod.EmbeddingService.from_config(settings)

    # set_journal_dir must come BEFORE embed_query
    set_idx = next(i for i, c in enumerate(call_log) if c.startswith("set_journal_dir:"))
    embed_idx = next(i for i, c in enumerate(call_log) if c.startswith("embed_query:"))
    assert set_idx < embed_idx, f"set_journal_dir must precede warmup; got {call_log}"
    assert call_log[set_idx] == f"set_journal_dir:{tmp_path / 'metrics'}"


def test_from_config_records_cold_start_gauge(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """AC5: embedder_cold_start_seconds is set after from_config (>= 0)."""
    from spatial_memory.config import Settings
    from spatial_memory.core import embeddings as emb_mod

    fake_st_module, _fake_model = _build_st_mocks()

    captured: list[float] = []
    monkeypatch.setattr(
        emb_mod,
        "record_embedder_cold_start",
        lambda seconds: captured.append(seconds),
    )

    settings = Settings(
        memory_path=tmp_path,
        embedding_model="arctic-m-long",
        embedding_backend="pytorch",
    )
    with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
        with patch.object(emb_mod, "_detect_backend", return_value="pytorch"):
            emb_mod.EmbeddingService.from_config(settings)

    assert len(captured) == 1
    assert captured[0] >= 0


def test_from_config_journals_cold_start_under_memory_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """AC4 + AC6 (negative control): journal lands under settings.memory_path / 'metrics',
    NOT under the env-var directory, and contains an embedder_cold_start_seconds entry.
    """
    import json as _json

    from spatial_memory.config import Settings
    from spatial_memory.core import embeddings as emb_mod
    from spatial_memory.core import metrics_journal

    # Reset the module-level _JOURNAL_DIR to a known pristine state so the
    # production code path inside from_config drives the binding.
    metrics_journal._JOURNAL_DIR = None
    metrics_journal._CACHED_DAILY_FILE = None

    env_dir = tmp_path / "from-env"
    explicit_dir = tmp_path / "from-settings"
    monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(env_dir))

    fake_st_module, _fake_model = _build_st_mocks()
    settings = Settings(
        memory_path=explicit_dir,
        embedding_model="arctic-m-long",
        embedding_backend="pytorch",
    )
    with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
        with patch.object(emb_mod, "_detect_backend", return_value="pytorch"):
            emb_mod.EmbeddingService.from_config(settings)

    # Journal must NOT exist under env-var directory
    env_metrics = env_dir / "metrics"
    assert not env_metrics.exists() or not list(env_metrics.glob("*.jsonl")), (
        "Journal leaked to SPATIAL_MEMORY_MEMORY_PATH dir; must follow settings.memory_path"
    )

    # Journal must exist under explicit Settings.memory_path / 'metrics'
    settings_metrics = explicit_dir / "metrics"
    jsonls = list(settings_metrics.glob("*.jsonl"))
    assert jsonls, f"Expected journal under {settings_metrics}; found none"

    # And contain at least one embedder_cold_start_seconds entry
    found_cold_start = False
    for jl in jsonls:
        for line in jl.read_text().splitlines():
            entry = _json.loads(line)
            if entry.get("metric") == "embedder_cold_start_seconds":
                found_cold_start = True
                assert entry["value"] >= 0
                break
        if found_cold_start:
            break
    assert found_cold_start, "No embedder_cold_start_seconds entry in journal"
