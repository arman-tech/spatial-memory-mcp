"""Hook script core modules for cognitive offloading.

These modules are the **source of truth** for hook script logic — tested,
linted, and importable from the server.  However, hook scripts (Phase B/C)
will load them via direct file-level import to avoid pulling in heavy
dependencies through ``spatial_memory/__init__.py``.

Phase B — ``post_tool_use.py``: Captures tool inputs/responses.
Phase C — ``pre_compact.py``, ``stop.py``: Captures assistant text from
the session transcript (decisions, analyses, solutions that PostToolUse
misses between tool calls).

**STDLIB-ONLY CONSTRAINT**: Every module in this package must use only
Python standard library imports (no lancedb, numpy, sentence-transformers,
or any third-party package).  Hook scripts spawn a fresh Python process on
every invocation; importing heavy dependencies would add ~500ms of latency.
"""
