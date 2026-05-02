# Contributing to Spatial Memory MCP Server

Thank you for your interest in contributing! This document provides guidelines for development.

## Quick Start

1. Read the [README](README.md) for project overview
2. Set up your development environment (see [Installation](#installation) below)
3. Run tests to verify setup: `pytest tests/ -v`
4. Look for issues labeled "good first issue" on GitHub
5. If using Claude Code, check [CLAUDE.md](CLAUDE.md) for AI assistant instructions

## Supported Platforms

This project supports development on:
- **Windows 11**
- **macOS** (latest)
- **Linux** (Fedora, Ubuntu, Linux Mint)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy spatial_memory/ --ignore-missing-imports

# Linting
ruff check spatial_memory/ tests/
```

## Project Structure

```
spatial-memory-mcp/
├── spatial_memory/          # Main package
│   ├── __init__.py         # Public API exports
│   ├── __main__.py         # CLI entry point (serve, migrate, instructions)
│   ├── server.py           # MCP server, tool handlers
│   ├── factory.py          # Dependency injection, service wiring
│   ├── config.py           # Settings (Pydantic), environment configuration
│   ├── core/               # Core infrastructure
│   │   ├── database.py     # LanceDB wrapper, CRUD operations
│   │   ├── embeddings.py   # Embedding service (arctic-m-long, PyTorch — registry-of-one per D1)
│   │   ├── models.py       # Pydantic data models (Memory, etc.)
│   │   ├── errors.py       # Exception hierarchy
│   │   ├── validation.py   # Input validation, security checks
│   │   ├── security.py     # Security utilities
│   │   ├── db_*.py         # Database utilities (search, indexes, migrations)
│   │   └── spatial_ops.py  # SLERP, vector operations
│   ├── services/           # Business logic layer
│   │   ├── memory.py       # Core memory operations (remember, recall, forget)
│   │   ├── spatial.py      # Spatial operations (journey, wander, regions)
│   │   ├── lifecycle.py    # Decay, reinforce, consolidate, extract
│   │   ├── ingest_pipeline.py # Dedup + quality gate pipeline
│   │   ├── queue_processor.py # Async file-drop queue processing
│   │   ├── decay_manager.py # Automatic memory decay management
│   │   ├── export_import.py# Import/export functionality
│   │   └── utility.py      # Stats, namespaces, health
│   ├── adapters/           # External service adapters (LanceDB repo, filesystem I/O)
│   ├── ports/              # Protocol interfaces
│   └── tools/              # MCP tool definitions
├── tests/                   # Test suite (~61 modules, 1750+ tests)
│   ├── conftest.py         # Shared fixtures
│   ├── unit/               # Unit tests (mocked dependencies)
│   └── integration/        # Integration tests (real DB/embeddings)
├── .env.example            # Example configuration
├── pyproject.toml          # Package configuration
├── CLAUDE.md               # Instructions for Claude Code AI assistant
├── SECURITY.md             # Security policy and guidelines
├── CHANGELOG.md            # Version history
└── README.md               # Project overview
```

## Testing

### Running Tests

```bash
# Run unit tests only (default, fast)
pytest tests/ -v

# Run ALL tests (unit + integration)
pytest tests/ -v -m ""

# Run integration tests only
pytest tests/ -v -m integration

# Run specific test file
pytest tests/unit/test_ingest_pipeline.py -v

# Run tests matching a pattern
pytest tests/ -k "test_remember" -v

# Run with coverage
pytest tests/ --cov=spatial_memory --cov-report=html
```

### Test Markers

- `@pytest.mark.unit` — Fast tests with mocked dependencies (default)
- `@pytest.mark.integration` — Tests with real database/embeddings (slower)
- `@pytest.mark.slow` — Tests taking >1 second
- `@pytest.mark.requires_model` — Tests needing the embedding model loaded

### Embedding Model Cache (CI & Contributor Setup)

The integration suite loads the production-default embedding model
(`arctic-m-long` — Snowflake Arctic Embed M Long, ~768 MB on disk including
PyTorch weights, tokenizer, and the model's `trust_remote_code` files) the
first time it is requested. arctic-m-long forces the PyTorch backend per
`spatial_memory/core/embeddings.py:_load_local_model` because the ONNX path
in sentence-transformers does not reliably plumb `trust_remote_code`
through to `AutoConfig.from_pretrained`. Re-downloading that on every
`pytest` invocation is painful on metered connections and slow on
cold-cache CI runners.

**Persist the Hugging Face cache between runs:**

```bash
# Linux/macOS — add to ~/.bashrc, ~/.zshrc, or your shell init file
export HF_HOME="$HOME/.cache/huggingface"

# Windows (PowerShell, persistent)
[Environment]::SetEnvironmentVariable("HF_HOME", "$env:USERPROFILE\.cache\huggingface", "User")
```

After the first integration run downloads the model, subsequent runs reuse the
cached weights and start in seconds rather than minutes.

**One model load per pytest session:**

`tests/conftest.py` exposes a session-scoped `shared_embedding_service`
fixture that constructs `EmbeddingService.from_config(Settings())` exactly
once per `pytest` invocation. Integration fixtures (`session_embedding_service`,
`embedding_service`) delegate to it, so the entire suite reuses a single
warm model regardless of how many integration tests run. If you write a new
integration test that needs embeddings, depend on `shared_embedding_service`
(or one of its aliases) — never call `EmbeddingService.from_config` or
`EmbeddingService("arctic-m-long")` directly inside a test or module-scoped
fixture, or you'll trigger a second load.

**CI runners** should mount/restore `$HF_HOME` between jobs (e.g. via
`actions/cache` keyed on the embedding-model alias) to make the first
integration job pay the download cost and every subsequent job hit the cache.

### Writing Tests

#### Use Fixtures

```python
# tests/conftest.py provides these fixtures:

def test_with_temp_storage(temp_storage):
    """temp_storage provides an isolated database directory."""
    # Database operations here are isolated
    pass

def test_with_mock_embeddings(mock_embedding_service):
    """mock_embedding_service avoids real embedding computation."""
    vector = mock_embedding_service.embed("test")
    assert len(vector) == 768
```

#### Test Naming Convention

```python
def test_<function>_<scenario>_<expected_outcome>():
    """Tests should follow this naming pattern."""
    pass

# Examples:
def test_remember_valid_content_returns_memory_id():
    pass

def test_recall_empty_query_raises_validation_error():
    pass
```

#### Testing Errors

```python
import pytest
from spatial_memory.core.errors import ValidationError

def test_invalid_input_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        # Code that should raise
        pass
    assert "expected message" in str(exc_info.value)
```

## Code Style

### Python Version

- Use Python 3.10+ features (union types with `|`, match statements, etc.)
- Type hints are required for all public functions

### Formatting & Linting

```bash
# Check for issues
ruff check spatial_memory/ tests/

# Auto-fix issues
ruff check --fix spatial_memory/ tests/
```

### Type Checking

```bash
mypy spatial_memory/ --ignore-missing-imports
```

### Documentation

- All public functions need docstrings
- Use Google-style docstrings:

```python
def remember(content: str, namespace: str = "default") -> str:
    """Store a memory in vector space.

    Args:
        content: The text content to remember.
        namespace: Namespace for isolation.

    Returns:
        The UUID of the created memory.

    Raises:
        ValidationError: If content is empty or too long.
        StorageError: If database operation fails.
    """
```

## Pull Request Process

1. **Fork** the repository at https://github.com/arman-tech/spatial-memory-mcp
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Ensure all tests pass**: `pytest tests/ -v`
5. **Ensure no lint errors**: `ruff check spatial_memory/ tests/`
6. **Ensure type checking passes**: `mypy spatial_memory/`
7. **Commit** with clear messages
8. **Push** to your fork
9. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Test coverage for new code

## Error Handling

### Exception Hierarchy

```
SpatialMemoryError (base)
├── MemoryNotFoundError    # Memory ID doesn't exist
├── NamespaceNotFoundError # Namespace doesn't exist
├── EmbeddingError         # Embedding generation failed
├── StorageError           # Database operation failed
├── ValidationError        # Input validation failed
├── ConfigurationError     # Invalid configuration
├── ClusteringError        # Clustering operation failed
└── VisualizationError     # Visualization generation failed
```

### When to Use Each

```python
from spatial_memory.core.errors import (
    ValidationError,
    StorageError,
    MemoryNotFoundError,
)

def my_function(memory_id: str):
    # Input validation
    if not memory_id:
        raise ValidationError("memory_id cannot be empty")

    # Database operations
    try:
        result = db.get(memory_id)
    except Exception as e:
        raise StorageError(f"Database error: {e}")

    # Not found
    if result is None:
        raise MemoryNotFoundError(memory_id)
```

## Configuration

### Adding New Settings

1. Add the field to `Settings` class in `config.py`:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    my_new_setting: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Description of what this does",
    )
```

2. Add to `.env.example`:

```bash
# Description of what this does
# SPATIAL_MEMORY_MY_NEW_SETTING=10
```

3. Add tests in `test_config.py`

## Questions?

- Open a GitHub issue for bugs or feature requests
- Check existing issues before creating new ones
- See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities
