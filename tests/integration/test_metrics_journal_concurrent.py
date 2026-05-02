"""Integration test for per-process journal isolation (spec §10.3 L684).

Verifies the spec contract: ``parallel pytest workers each get their own file;
combined parse over all files yields the expected line count``. The journal
filename embeds ``<pid>-<startepoch>`` to avoid cross-platform locking, and
this test pins that contract under real concurrent processes (rather than
the single-process unit-test surrogate that just regex-matches the filename).

We use ``multiprocessing.get_context("spawn")`` for cross-platform
compatibility — fork would also work on Linux but Windows lacks fork, and
spawn forces each child to re-import ``metrics_journal`` so each gets its
own ``_PROCESS_START_EPOCH`` stamp (matching real production behavior).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
from pathlib import Path

import pytest

# Force spawn on all platforms — keeps each child a clean module-import
# (its own ``_PROCESS_START_EPOCH`` and PID-derived filename), and avoids
# fork-unsafe surprises if any pulled-in dependency starts a thread.
_mp_context = multiprocessing.get_context("spawn")

_NUM_WORKERS = 3
_OBSERVATIONS_PER_WORKER = 25


def _journal_worker(journal_dir: str, worker_id: int, num_observations: int) -> None:
    """Child-process entry point: write ``num_observations`` JSONL lines.

    Each observation embeds the worker_id and observation index in its labels
    so the parent can verify nothing was lost or duplicated across processes.
    """
    from spatial_memory.core.metrics_journal import journal_observation, set_journal_dir

    set_journal_dir(Path(journal_dir))
    for i in range(num_observations):
        journal_observation(
            "concurrent_test_metric",
            float(i),
            {"worker_id": str(worker_id), "obs_index": str(i)},
        )


@pytest.mark.integration
def test_parallel_processes_each_get_their_own_journal_file(tmp_path: Path) -> None:
    """§10.3 L684: each worker's PID produces its own JSONL file.

    Spawns ``_NUM_WORKERS`` child processes pointing at the same journal
    directory and asserts:
      1. exactly ``_NUM_WORKERS`` JSONL files exist (one per child PID);
      2. every filename matches the ``<YYYY-MM-DD>-<pid>-<startepoch>.jsonl``
         pattern from spec §10.3;
      3. each file's PID component is unique (no two children shared a writer);
      4. each child's PID matches one of the filenames (proof that filenames
         derive from ``os.getpid()`` rather than some shared parent identifier);
      5. combined line count across all files equals
         ``_NUM_WORKERS * _OBSERVATIONS_PER_WORKER``.
    """
    journal_dir = tmp_path / "metrics"
    journal_dir.mkdir()

    processes: list[multiprocessing.Process] = []
    for worker_id in range(_NUM_WORKERS):
        p = _mp_context.Process(
            target=_journal_worker,
            args=(str(journal_dir), worker_id, _OBSERVATIONS_PER_WORKER),
        )
        p.start()
        processes.append(p)

    child_pids = {p.pid for p in processes}
    # Every spawned child must have a real OS PID before we wait on it.
    assert None not in child_pids, "multiprocessing.Process.start() did not assign PIDs"

    for p in processes:
        p.join(timeout=30)
        assert not p.is_alive(), f"Worker {p.pid} did not finish within 30s"
        assert p.exitcode == 0, f"Worker {p.pid} exited with code {p.exitcode}"

    # Parent process did NOT call journal_observation, so it should not have
    # written a file under this dir. Children own all files here.
    files = sorted(journal_dir.glob("*.jsonl"))
    assert len(files) == _NUM_WORKERS, (
        f"Expected {_NUM_WORKERS} per-PID journal files, got {len(files)}: "
        f"{[f.name for f in files]}"
    )

    # Filename pattern from spec §10.3: <YYYY-MM-DD>-<pid>-<startepoch>.jsonl
    filename_re = re.compile(r"^\d{4}-\d{2}-\d{2}-(\d+)-\d+\.jsonl$")
    file_pids: set[int] = set()
    for f in files:
        m = filename_re.match(f.name)
        assert m, (
            f"Filename {f.name!r} does not match "
            f"<YYYY-MM-DD>-<pid>-<startepoch>.jsonl from spec §10.3"
        )
        file_pids.add(int(m.group(1)))

    # No two children shared a writer — file PIDs are pairwise distinct.
    assert len(file_pids) == _NUM_WORKERS, (
        f"Expected {_NUM_WORKERS} distinct PIDs across files, got {file_pids}"
    )

    # Every file PID matches a real spawned-child PID. Filename PIDs derive from
    # os.getpid() inside each child — not from the parent or some shared id —
    # so this proves cross-process isolation.
    parent_pid = os.getpid()
    assert parent_pid not in file_pids, (
        "Parent PID leaked into journal filenames — set_journal_dir was not "
        "isolated to children"
    )
    assert file_pids == {pid for pid in child_pids if pid is not None}, (
        f"File PIDs {file_pids} != child PIDs {child_pids}; cross-process "
        "isolation broken"
    )

    # Combined parse over all files yields the expected line count.
    total_lines = 0
    seen_obs: set[tuple[str, str]] = set()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            assert entry["metric"] == "concurrent_test_metric"
            seen_obs.add((entry["labels"]["worker_id"], entry["labels"]["obs_index"]))
            total_lines += 1

    expected_total = _NUM_WORKERS * _OBSERVATIONS_PER_WORKER
    assert total_lines == expected_total, (
        f"Combined line count {total_lines} != expected "
        f"{_NUM_WORKERS} × {_OBSERVATIONS_PER_WORKER} = {expected_total}; "
        "concurrent appends must not lose lines"
    )

    # Every (worker_id, obs_index) pair appears exactly once — no dupes, no
    # losses. Confirms POSIX atomic-append under PIPE_BUF held across workers.
    expected_obs = {
        (str(w), str(i))
        for w in range(_NUM_WORKERS)
        for i in range(_OBSERVATIONS_PER_WORKER)
    }
    assert seen_obs == expected_obs, (
        f"Observation set mismatch — missing {expected_obs - seen_obs}, "
        f"unexpected {seen_obs - expected_obs}"
    )
