# Spatial Memory MCP - Contributor Guide

This document helps AI assistants (and human contributors) work effectively on this codebase.

## Project Overview

**spatial-memory-mcp** is a persistent semantic memory MCP server for LLMs. It provides vector-based memory storage with spatial navigation capabilities using LanceDB and sentence-transformers.

## Architecture

```
spatial_memory/
├── server.py           # MCP server, tool handlers, instructions injection
├── factory.py          # Dependency injection, service instantiation
├── config.py           # Settings (Pydantic), environment configuration
├── core/               # Core infrastructure
│   ├── database.py     # LanceDB wrapper, CRUD operations
│   ├── embeddings.py   # Sentence-transformer embedding service
│   ├── models.py       # Pydantic data models (Memory, etc.)
│   ├── errors.py       # Exception hierarchy
│   ├── validation.py   # Input validation, security checks
│   ├── security.py     # Security utilities
│   ├── db_*.py         # Database utilities (search, indexes, migrations)
│   └── spatial_ops.py  # SLERP, vector operations
├── services/           # Business logic layer
│   ├── memory.py       # Core memory operations (remember, recall, forget)
│   ├── spatial.py      # Spatial operations (journey, wander, regions)
│   ├── lifecycle.py    # Decay, reinforce, consolidate, extract
│   ├── export_import.py# Import/export functionality
│   └── utility.py      # Stats, namespaces, health
├── adapters/           # External service adapters
├── ports/              # Interface definitions
└── tools/              # MCP tool definitions
```

## Key Commands

```bash
# Run tests (unit tests only by default)
pytest tests/ -v

# Run all tests including integration
pytest tests/ -v -m ""

# Run integration tests only
pytest tests/ -v -m integration

# Type checking
mypy spatial_memory/

# Linting
ruff check spatial_memory/ tests/

# Format code
ruff format spatial_memory/ tests/

# Run the server directly
python -m spatial_memory
```

## Test Markers

- `@pytest.mark.unit` - Fast tests with mocked dependencies
- `@pytest.mark.integration` - Tests with real database/embeddings (slower)
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.requires_model` - Tests needing the embedding model loaded

## Configuration

Settings via environment variables or `.env` file:
- `SPATIAL_MEMORY_MEMORY_PATH` - Database storage location (default: `.spatial-memory/`)
- `SPATIAL_MEMORY_EMBEDDING_MODEL` - Model name (default: `arctic-m-long`; v2.0 registry-of-one per D1)
- `SPATIAL_MEMORY_AMBIENT_RECALL` - Ambient-recall hook toggle (default: `on`; D3)
- `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K` - Top-K memories injected per prompt (default: `5`)
- `SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE` - Optional namespace scope for ambient recall
- `SPATIAL_MEMORY_LOG_LEVEL` - Logging level
- See `config.py` and `.env.example` for full list

## Code Style

- Line length: 100 characters
- Python 3.10+ (use modern syntax: `|` unions, etc.)
- Type hints required (strict mypy)
- Ruff for linting/formatting

## Important Patterns

1. **Dependency Injection**: Services receive dependencies via constructor, not global imports
2. **Sync Services, Async MCP Handlers**: Service and database methods are synchronous; MCP handlers run them via `ThreadPoolExecutor`
3. **Pydantic Models**: Use for data validation and serialization
4. **Error Hierarchy**: Custom exceptions in `core/errors.py`

## MCP Tool Categories

The server exposes these tool groups:
- **Memory**: `remember`, `recall`, `hybrid_recall`, `nearby`, `forget`
- **Spatial**: `journey`, `wander`, `regions`, `visualize`
- **Lifecycle**: `decay`, `reinforce`, `extract`, `consolidate`
- **Admin**: `stats`, `namespaces`, `export_memories`, `import_memories`

## Notes

- MCP server instructions are auto-injected via `_get_server_instructions()` in `server.py`
- Database uses LanceDB (embedded vector database)
- Embeddings: `arctic-m-long` only (Snowflake/snowflake-arctic-embed-m-long, 768-dim, 8K context, PyTorch backend, `trust_remote_code=True` scoped to this model). Preloaded at MCP startup. See D1 in `internal-docs/v2-architectural-decisions.md`.

## Hook events

Two install paths register Claude Code hooks. The `setup-hooks` CLI registers five events by default — `SessionStart`, `PostToolUse`, `PreCompact`, `Stop`, `UserPromptSubmit` — with `--no-session-start` available to opt out. The `plugin/hooks/hooks.json` path registers four (`PostToolUse`, `PreCompact`, `Stop`, `UserPromptSubmit`); `SessionStart` is deliberately excluded because Claude Code v2.1.37 freezes terminal input when a plugin registers a `SessionStart` hook on Windows (workaround: install `SessionStart` via `.claude/settings.json` using `setup-hooks`). The `UserPromptSubmit` event drives v2.0's ambient-recall feature — the dispatcher invokes `spatial_memory.hooks.user_prompt_submit.handle`, which runs `hybrid_recall` against the user's prompt and emits a top-K context block to stdout (Claude Code reads stdout as the context-injection payload). Behavior is gated by three env vars: `SPATIAL_MEMORY_AMBIENT_RECALL` (`on`/`off`, default `on`), `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K` (default `5`), `SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE` (default unset → all namespaces). Fail-open: any exception or timeout becomes a no-op.

**Cursor gap (v2.0):** Cursor's hook surface (`postToolUse`, `preCompact`, `stop`) has no `UserPromptSubmit`-equivalent event. Ambient recall is therefore Claude-Code-only in v2.0; Cursor users continue to get post-tool-use ingestion and pre-compact distillation but no automatic prompt-time recall injection. This is tracked as a deliberate scope decision in spec §8.4.
