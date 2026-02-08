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
- `SPATIAL_MEMORY_EMBEDDING_MODEL` - Model name (default: `all-MiniLM-L6-v2`)
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
- Embeddings default to ONNX-optimized sentence-transformers for performance
