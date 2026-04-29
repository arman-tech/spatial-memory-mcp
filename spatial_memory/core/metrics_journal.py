"""Always-on metrics journal — JSONL observability sink (US-016 / spec §10.3).

Every helper in :mod:`spatial_memory.core.metrics` writes one structured JSON
line per observation regardless of whether the optional ``prometheus_client``
dependency is installed. The journal is the *always-on* path; Prometheus is
optional bookkeeping on top.

Operational notes
-----------------
* Per-process filenames (``<YYYY-MM-DD>-<pid>-<startepoch>.jsonl``) avoid
  cross-platform locking — concurrent processes never share a writer.
* Each line is appended in a single ``write`` call sized below ``PIPE_BUF``
  (4096 bytes, R16). POSIX guarantees atomic-append for writes ≤ ``PIPE_BUF``,
  so multi-thread interleaving inside a single process stays line-clean.
* Fail-open contract: ``IOError`` and ``OSError`` are swallowed silently —
  the journal must never break a request path.
* Lazy fallback: if no caller invoked :func:`set_journal_dir`, the directory
  resolves to ``$SPATIAL_MEMORY_MEMORY_PATH/metrics`` (or
  ``./.spatial-memory/metrics`` if the env var is unset) so unit tests and
  ad-hoc imports still produce output without explicit wiring.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LABEL_VALUE_MAX = 64
_PROCESS_START_EPOCH: int = int(time.time())
_JOURNAL_DIR: Path | None = None
_CACHED_DAILY_FILE: Path | None = None


def set_journal_dir(path: Path) -> None:
    """Pin the journal directory; idempotent. Resets the cached daily file."""
    global _JOURNAL_DIR, _CACHED_DAILY_FILE
    _JOURNAL_DIR = Path(path)
    _CACHED_DAILY_FILE = None


def _resolve_journal_dir() -> Path:
    if _JOURNAL_DIR is not None:
        return _JOURNAL_DIR
    base = Path(os.environ.get("SPATIAL_MEMORY_MEMORY_PATH", ".spatial-memory"))
    return base / "metrics"


def _resolve_daily_file() -> Path:
    global _CACHED_DAILY_FILE
    today = datetime.now(timezone.utc).date().isoformat()
    journal_dir = _resolve_journal_dir()
    expected = journal_dir / f"{today}-{os.getpid()}-{_PROCESS_START_EPOCH}.jsonl"
    if _CACHED_DAILY_FILE is None or _CACHED_DAILY_FILE != expected:
        _CACHED_DAILY_FILE = expected
    return _CACHED_DAILY_FILE


def _truncate_labels(labels: dict[str, Any]) -> dict[str, Any]:
    truncated: dict[str, Any] = {}
    for k, v in labels.items():
        if isinstance(v, str) and len(v) > _LABEL_VALUE_MAX:
            truncated[k] = v[:_LABEL_VALUE_MAX]
        else:
            truncated[k] = v
    return truncated


def _resolve_request_id() -> str:
    from spatial_memory.core.tracing import get_current_context

    ctx = get_current_context()
    return ctx.request_id if ctx is not None else "startup"


def journal_observation(metric: str, value: float, labels: dict[str, Any]) -> None:
    """Append one JSON line to the daily journal file. Fail-open."""
    try:
        target = _resolve_daily_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": _resolve_request_id(),
            "metric": metric,
            "value": value,
            "labels": _truncate_labels(labels),
        }
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        return


def prune_journal(retention_days: int = 7) -> None:
    """Remove journal files older than ``retention_days``. Fail-open."""
    try:
        journal_dir = _resolve_journal_dir()
        if not journal_dir.exists():
            return
        cutoff_date = datetime.now(timezone.utc).date()
        for jsonl in journal_dir.glob("*.jsonl"):
            stem_parts = jsonl.stem.split("-")
            if len(stem_parts) < 3:
                continue
            try:
                file_date = datetime.strptime("-".join(stem_parts[:3]), "%Y-%m-%d").date()
            except ValueError:
                continue
            age_days = (cutoff_date - file_date).days
            if age_days > retention_days:
                try:
                    jsonl.unlink()
                except OSError:
                    continue
    except OSError:
        return
