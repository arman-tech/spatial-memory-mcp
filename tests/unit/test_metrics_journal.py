"""Unit tests for spatial_memory.core.metrics_journal (US-016 / spec §10.3).

The metrics journal is the always-on observability path: every helper in
core/metrics.py writes one JSON line to disk regardless of whether the
optional ``prometheus_client`` dep is installed. The module is deliberately
stdlib-only and ~50 LOC so it can survive in resource-constrained deployments
without dragging in third-party I/O libraries.

These tests pin the public contract before the implementation lands so the
acceptance criteria are encoded as RED → GREEN evidence.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Each test resets the module's cached journal_dir + daily file pin.

    The module is intentionally module-state-driven (a daemon-style facility),
    so every test that mutates state via ``set_journal_dir`` must restore the
    pre-test state — otherwise tests bleed config across each other.
    """
    from spatial_memory.core import metrics_journal as mj

    # Snapshot before
    pre_dir = getattr(mj, "_JOURNAL_DIR", None)
    pre_cached = getattr(mj, "_CACHED_DAILY_FILE", None)
    yield
    # Restore after
    mj._JOURNAL_DIR = pre_dir
    mj._CACHED_DAILY_FILE = pre_cached


def test_module_exports_three_public_functions() -> None:
    """Acceptance criterion 2: module exports the three public callables."""
    from spatial_memory.core import metrics_journal as mj

    assert callable(mj.set_journal_dir)
    assert callable(mj.journal_observation)
    assert callable(mj.prune_journal)


def test_journal_observation_writes_well_formed_json_line(tmp_path: Path) -> None:
    """Acceptance criterion 3: one call → one JSON line with the spec'd schema."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)
    mj.journal_observation("test_metric", 42.0, {"tool": "recall"})

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry.keys()) == {"ts", "request_id", "metric", "value", "labels"}
    assert entry["metric"] == "test_metric"
    assert entry["value"] == 42.0
    assert entry["labels"] == {"tool": "recall"}


def test_filename_matches_spec_pattern(tmp_path: Path) -> None:
    """Acceptance criterion 4: filename is <YYYY-MM-DD>-<pid>-<startepoch>.jsonl."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)
    mj.journal_observation("m", 1.0, {})

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    pid = os.getpid()
    pattern = rf"^\d{{4}}-\d{{2}}-\d{{2}}-{pid}-\d+\.jsonl$"
    assert re.match(pattern, files[0].name), f"Filename {files[0].name!r} does not match {pattern}"


def test_set_journal_dir_is_idempotent_and_resets_cache(tmp_path: Path) -> None:
    """Acceptance criterion 5: re-calling set_journal_dir clears the cached daily file.

    First write → file under dir A. Re-point at dir B → next write goes to B
    (proving the cache was reset, not leaking the old path).
    """
    from spatial_memory.core import metrics_journal as mj

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    mj.set_journal_dir(dir_a)
    mj.journal_observation("m", 1.0, {})
    assert list(dir_a.glob("*.jsonl"))

    mj.set_journal_dir(dir_b)
    mj.journal_observation("m", 2.0, {})
    assert list(dir_b.glob("*.jsonl"))


def test_prune_journal_removes_files_older_than_retention(tmp_path: Path) -> None:
    """Acceptance criterion 6: prune_journal(1) removes 2-day-old, keeps yesterday."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)

    pid = os.getpid()
    epoch = int(time.time())
    yesterday_file = tmp_path / f"{yesterday.isoformat()}-{pid}-{epoch}.jsonl"
    two_days_ago_file = tmp_path / f"{two_days_ago.isoformat()}-{pid}-{epoch}.jsonl"
    yesterday_file.write_text('{"m":"x"}\n', encoding="utf-8")
    two_days_ago_file.write_text('{"m":"x"}\n', encoding="utf-8")

    mj.prune_journal(retention_days=1)

    assert yesterday_file.exists(), "yesterday's file must survive retention=1"
    assert not two_days_ago_file.exists(), "2-day-old file must be pruned"


def test_self_creates_journal_directory_on_first_write(tmp_path: Path) -> None:
    """Acceptance criterion 7: directory is auto-created (no FileNotFoundError)."""
    from spatial_memory.core import metrics_journal as mj

    nonexistent = tmp_path / "does" / "not" / "exist"
    assert not nonexistent.exists()

    mj.set_journal_dir(nonexistent)
    mj.journal_observation("m", 1.0, {})

    assert nonexistent.exists()
    assert list(nonexistent.glob("*.jsonl"))


def test_journal_observation_swallows_oserror_silently(tmp_path: Path) -> None:
    """Acceptance criterion 8: fail-open under IOError/OSError (no propagation)."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)

    with patch("builtins.open", side_effect=OSError("disk full")):
        # Must not raise — the journal is best-effort.
        mj.journal_observation("m", 1.0, {})


def test_journal_observation_swallows_ioerror_silently(tmp_path: Path) -> None:
    """Acceptance criterion 8: same fail-open contract for IOError."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)

    with patch("builtins.open", side_effect=OSError("broken pipe")):
        mj.journal_observation("m", 1.0, {})


def test_long_label_values_truncated_to_64_chars(tmp_path: Path) -> None:
    """Acceptance criterion 9 / R16: keep line size <PIPE_BUF (4096) by capping labels."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)
    long_value = "x" * 200
    mj.journal_observation("m", 1.0, {"big": long_value})

    files = list(tmp_path.glob("*.jsonl"))
    entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert len(entry["labels"]["big"]) == 64
    assert entry["labels"]["big"] == "x" * 64


def test_lazy_fallback_uses_spatial_memory_memory_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 10: when set_journal_dir was never called, fall back to env."""
    from spatial_memory.core import metrics_journal as mj

    # Clear any prior set_journal_dir call from another test
    mj._JOURNAL_DIR = None
    mj._CACHED_DAILY_FILE = None

    monkeypatch.setenv("SPATIAL_MEMORY_MEMORY_PATH", str(tmp_path))
    mj.journal_observation("m", 1.0, {})

    expected_dir = tmp_path / "metrics"
    assert expected_dir.exists()
    assert list(expected_dir.glob("*.jsonl"))


def test_request_id_pulled_from_tracing_context(tmp_path: Path) -> None:
    """request_id wires through core/tracing.RequestContext.get_current_context()."""
    from spatial_memory.core import metrics_journal as mj
    from spatial_memory.core.tracing import RequestContext, set_context

    mj.set_journal_dir(tmp_path)

    ctx = RequestContext.create(tool_name="recall")

    def _run_in_ctx() -> None:
        set_context(ctx)
        mj.journal_observation("m", 1.0, {})

    cv_ctx = contextvars.copy_context()
    cv_ctx.run(_run_in_ctx)

    files = list(tmp_path.glob("*.jsonl"))
    entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert entry["request_id"] == ctx.request_id


def test_request_id_falls_back_to_startup_when_no_context(tmp_path: Path) -> None:
    """When no RequestContext is active, request_id is the literal 'startup'."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)
    mj.journal_observation("m", 1.0, {})

    files = list(tmp_path.glob("*.jsonl"))
    entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert entry["request_id"] == "startup"


def test_module_state_pid_and_startepoch_stable_for_session() -> None:
    """_PROCESS_START_EPOCH is captured once at import (not per-call)."""
    from spatial_memory.core import metrics_journal as mj

    epoch_a = mj._PROCESS_START_EPOCH
    time.sleep(0.05)
    epoch_b = mj._PROCESS_START_EPOCH
    assert epoch_a == epoch_b


def test_pipe_buf_safe_line_size(tmp_path: Path) -> None:
    """R16: each emitted line stays under PIPE_BUF (4096 bytes) for atomic-append."""
    from spatial_memory.core import metrics_journal as mj

    mj.set_journal_dir(tmp_path)
    # 10 labels each at the truncation cap → bounded by metric_name + value + labels
    big_labels = {f"k{i}": "v" * 200 for i in range(10)}
    mj.journal_observation("very_long_metric_name_for_size_check", 1.234, big_labels)

    files = list(tmp_path.glob("*.jsonl"))
    line = files[0].read_text(encoding="utf-8").splitlines()[0]
    line_bytes = len(line.encode("utf-8"))
    assert line_bytes < 4096, f"line size {line_bytes} exceeds PIPE_BUF"
