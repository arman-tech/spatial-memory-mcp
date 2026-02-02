# Spatial Memory MCP Server

[![PyPI version](https://badge.fury.io/py/spatial-memory-mcp.svg)](https://pypi.org/project/spatial-memory-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A vector-based spatial memory system that treats knowledge as a navigable landscape, not a filing cabinet.

> **Version 1.6.2** — Production-ready with 1,360 tests passing.

## Supported Platforms

- **Windows 11**
- **macOS** (latest)
- **Linux** (Fedora, Ubuntu, Linux Mint)

## Overview

Spatial Memory MCP Server provides persistent, semantic memory for LLMs through the Model Context Protocol (MCP). Unlike traditional keyword-based memory systems, it uses vector embeddings to enable:

- **Semantic Search**: Find memories by meaning, not just keywords
- **Spatial Navigation**: Discover connections through `journey` and `wander` operations
- **Auto-Clustering**: `regions` automatically groups related concepts
- **Cognitive Dynamics**: Memories consolidate, decay, and reinforce like human cognition
- **Visual Understanding**: Generate Mermaid/SVG/JSON visualizations of your knowledge space
- **Hybrid Search**: Combine vector similarity with full-text search

## Why Spatial Memory?

### Zero Cognitive Load
**You never think about memory. Claude handles everything automatically.**

- Auto-loads relevant context at session start
- Recognizes memory-worthy moments and asks "Save this? y/n"
- Synthesizes answers naturally—no raw JSON, no memory IDs
- MCP instructions inject automatically—zero configuration

### Cognitive Architecture, Not Storage
Other memory systems store embeddings. Spatial Memory implements how memory actually works:

- **Decay**: Memories fade over time (Ebbinghaus forgetting curve)
- **Reinforcement**: Frequently accessed memories grow stronger
- **Consolidation**: Similar memories merge intelligently
- **Extraction**: Auto-capture facts, decisions, patterns from conversations

### Spatial Navigation (Unique Innovation)
Navigate knowledge like a landscape, not a filing cabinet:

| Tool | What It Does |
|------|-------------|
| **Journey** | SLERP between two memories—discover what's conceptually in between |
| **Wander** | Random walk exploration—find unexpected connections |
| **Regions** | HDBSCAN clustering—see how your knowledge self-organizes |
| **Visualize** | UMAP projection—view your memory space in 2D/3D |

### 22 Tools vs. 3-6 in Competitors
Full lifecycle management: core operations, spatial navigation, memory lifecycle, hybrid search, namespace management, data import/export, and health/stats monitoring.

### Enterprise-Ready
Connection pooling, circuit breakers, per-agent rate limiting, request tracing, response caching, and defense-in-depth security (path traversal prevention, SQL injection detection, input validation).

> *See [MARKETING.md](MARKETING.md) for the complete comparison with competitors.*

## Features

- **22 MCP tools** across 4 categories (core, spatial, lifecycle, utility)
- **Clean Architecture** with ports/adapters pattern for testability
- **LanceDB** vector storage with automatic indexing
- **Dual embedding support**: Local (sentence-transformers) or OpenAI API
- **ONNX Runtime** by default for 2-3x faster embeddings
- **Enterprise features**: Connection pooling, retry logic, batch operations
- **Comprehensive security**: Path validation, SQL injection prevention, input sanitization
- **1360 tests** including security edge cases

## Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1: Foundation | Complete | Config, Database, Embeddings, Models, Errors |
| Phase 2: Core Operations | Complete | `remember`, `recall`, `nearby`, `forget` |
| Phase 3: Spatial Operations | Complete | `journey`, `wander`, `regions`, `visualize` |
| Phase 4: Lifecycle Operations | Complete | `consolidate`, `extract`, `decay`, `reinforce` |
| Phase 5: Utilities | Complete | `stats`, `namespaces`, `export`, `import`, `hybrid_recall` |
| Phase 6: Polish & Release | Complete | v1.6.0 on PyPI |

## Installation

### From PyPI (Recommended)

```bash
pip install spatial-memory-mcp

# Or with uv
uv pip install spatial-memory-mcp
```

### Development Setup

```bash
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp
pip install -e ".[dev]"
```

### With OpenAI Support

```bash
# From PyPI
pip install spatial-memory-mcp[openai]

# From source
pip install -e ".[dev,openai]"
```

### Verify Installation

After installation, verify that all dependencies are correctly installed:

```bash
python -m spatial_memory.verify
```

Or manually check:

```python
# Run in Python interpreter
from spatial_memory.core.embeddings import EmbeddingService, _is_onnx_available
print(f"ONNX available: {_is_onnx_available()}")
svc = EmbeddingService()
print(f"Backend: {svc.backend}, Dimensions: {svc.dimensions}")
```

Expected output with ONNX Runtime (default):
```
ONNX available: True
Backend: onnx, Dimensions: 384
```

If ONNX shows as unavailable, reinstall with:
```bash
pip install --force-reinstall "sentence-transformers[onnx]"
```

### CLI Commands

The `spatial-memory` CLI provides several commands:

```bash
# Start the MCP server (default)
spatial-memory serve

# View the MCP instructions injected into Claude's context
spatial-memory instructions

# Run database migrations
spatial-memory migrate --status    # Check migration status
spatial-memory migrate --dry-run   # Preview migrations
spatial-memory migrate             # Apply pending migrations

# Show version
spatial-memory --version
```

The `instructions` command is useful for understanding what behavioral guidelines are automatically injected when the MCP server connects to Claude.

### Optional Performance Dependencies

For best performance, install these optional dependencies:

```bash
# ONNX Runtime - 2-3x faster embeddings, 60% less memory
pip install onnxruntime

# Or install with sentence-transformers ONNX support
pip install "sentence-transformers[onnx]"

# scipy - Optimized pairwise similarity calculations
# Used by visualize, regions, and consolidate operations
pip install scipy
```

**Performance comparison:**

| Component | Without | With | Benefit |
|-----------|---------|------|---------|
| ONNX Runtime | PyTorch inference | ONNX inference | 2-3x faster embeddings |
| scipy | numpy matrix ops | scipy.cdist | Faster similarity calculations |

Both are optional - the system includes fallbacks that use numpy when these aren't available.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MEMORY_PATH` | `./.spatial-memory` | LanceDB storage directory |
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model (local or `openai:*`) |
| `SPATIAL_MEMORY_EMBEDDING_BACKEND` | `auto` | Backend: `auto`, `onnx`, or `pytorch` |
| `SPATIAL_MEMORY_OPENAI_API_KEY` | - | Required only for OpenAI embeddings |
| `SPATIAL_MEMORY_LOG_LEVEL` | `INFO` | Logging verbosity |
| `SPATIAL_MEMORY_AUTO_CREATE_INDEXES` | `true` | Auto-create vector indexes |

### Embedding Models

**Local models** (no API key required):
- `all-MiniLM-L6-v2` - Fast, good quality (384 dimensions)
- `all-mpnet-base-v2` - Slower, better quality (768 dimensions)

**OpenAI models** (requires API key):
- `openai:text-embedding-3-small` - Fast, cost-effective (1536 dimensions)
- `openai:text-embedding-3-large` - Best quality (3072 dimensions)

### Embedding Backend

By default, local models use **ONNX Runtime** for 2-3x faster inference and 60% less memory:

| Backend | Speed | Memory | Notes |
|---------|-------|--------|-------|
| ONNX Runtime (default) | 2-3x faster | 60% less | Optimized for CPU inference |
| PyTorch | Baseline | Baseline | Full flexibility |

Configure via environment variable:
```bash
# Auto-detect (default) - uses ONNX if available
SPATIAL_MEMORY_EMBEDDING_BACKEND=auto

# Force specific backend
SPATIAL_MEMORY_EMBEDDING_BACKEND=onnx
SPATIAL_MEMORY_EMBEDDING_BACKEND=pytorch
```

## Usage

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "python",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_MEMORY_PATH": "/path/to/memory/storage"
      }
    }
  }
}
```

## Available Tools (22 Total)

For complete API documentation including parameters, return types, and examples, see [docs/API.md](docs/API.md).

### Core Operations

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with optional tags, importance, and metadata |
| `remember_batch` | Store multiple memories efficiently in a single operation |
| `recall` | Find memories semantically similar to a query with optional filters |
| `nearby` | Find memories spatially close to a specific memory by ID |
| `forget` | Remove a memory by ID |
| `forget_batch` | Remove multiple memories by IDs |

### Spatial Operations

| Tool | Description |
|------|-------------|
| `journey` | Interpolate a path between two memories using SLERP, discovering concepts along the way |
| `wander` | Random walk through memory space for serendipitous discovery |
| `regions` | Discover conceptual regions via HDBSCAN clustering with auto-generated labels |
| `visualize` | Generate 2D visualization (JSON coordinates, Mermaid diagrams, or SVG) |

### Lifecycle Operations

| Tool | Description |
|------|-------------|
| `decay` | Reduce importance of stale/unused memories based on time or access patterns |
| `reinforce` | Boost importance of useful memories |
| `extract` | Auto-extract memorable facts, decisions, and insights from text |
| `consolidate` | Find and merge similar/duplicate memories with configurable strategies |

### Utility Operations

| Tool | Description |
|------|-------------|
| `stats` | Get comprehensive database statistics (counts, storage, indexes) |
| `namespaces` | List all namespaces with memory counts |
| `delete_namespace` | Delete a namespace and all its memories |
| `rename_namespace` | Rename a namespace |
| `export_memories` | Export memories to Parquet, JSON, or CSV format |
| `import_memories` | Import memories from exported files with validation |
| `hybrid_recall` | Combined vector + full-text search with configurable weighting |
| `health` | Check system health status |

## Tool Examples

### Remember a Memory

```
Store this: "Use repository pattern for database access in this project"
Tags: architecture, patterns
Importance: 0.8
```

### Semantic Recall

```
What do I know about database patterns?
```

### Journey Between Concepts

```
Show me a journey from "React components" to "database design"
```
This reveals intermediate concepts like state management, data flow, API design, etc.

### Discover Regions

```
What conceptual regions exist in my memories?
```
Returns auto-clustered groups with labels and representative memories.

### Apply Memory Decay

```
Decay unused memories (dry run first to preview)
```

### Export for Backup

```
Export all memories to parquet format
```

## Security Features

- **Path Traversal Prevention**: All file operations validate paths against allowed directories
- **Symlink Attack Protection**: Optional symlink blocking for sensitive environments
- **SQL Injection Prevention**: 13 patterns covering major attack vectors
- **Input Validation**: Pydantic models validate all inputs
- **Error Sanitization**: Internal errors return reference IDs, not stack traces
- **Secure Credential Handling**: API keys stored as SecretStr

## Deployment Considerations

### Network Filesystems (NFS/SMB/CIFS)

Spatial Memory uses file-based locking to prevent data corruption when multiple processes access the same storage. **File locking does not work reliably on network filesystems** such as NFS, SMB/CIFS, or SSHFS.

If the storage path is on a network filesystem, you will see a warning at startup:

```
WARNING: Storage path appears to be on a network filesystem (nfs).
File-based locking does not work reliably on network filesystems.
Running multiple instances against this storage may cause data corruption.
```

**Recommendations:**
1. **Use local storage** (default: `./.spatial-memory`) for reliable operation
2. **Single instance only**: If you must use network storage, ensure only one MCP server instance accesses it
3. **Acknowledge the risk**: Set `SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FS_RISK=true` to suppress the warning

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast unit tests only
pytest tests/ -m unit -v

# Security-specific tests
pytest tests/ -k "security or injection" -v
```

### Type Checking

```bash
mypy spatial_memory/ --ignore-missing-imports
```

### Linting

```bash
ruff check spatial_memory/ tests/
```

## Architecture

The project follows Clean Architecture principles:

```
spatial_memory/
├── core/           # Domain logic (database, embeddings, models, errors)
├── services/       # Application services (memory, spatial, lifecycle)
├── adapters/       # Infrastructure (LanceDB repository)
├── ports/          # Protocol interfaces
└── server.py       # MCP server entry point
```

See [SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md](SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md) for visual architecture documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete API reference for all 22 tools |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Performance benchmarks and test results |
| [docs/METRICS.md](docs/METRICS.md) | Prometheus metrics documentation |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Troubleshooting guide |

## Troubleshooting

### Common Issues

**Model download fails**: The first run downloads the embedding model (~80MB). Ensure internet connectivity.

**Permission errors**: Check that `SPATIAL_MEMORY_MEMORY_PATH` is writable.

**OpenAI errors**: Verify `SPATIAL_MEMORY_OPENAI_API_KEY` is set correctly.

**Import validation errors**: Use `dry_run=true` first to preview validation issues.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Security

For security vulnerabilities, please email directly rather than opening a public issue.

## For Claude Code Users

When the MCP server connects, behavioral instructions are **automatically injected** into Claude's context—no configuration needed. Run `spatial-memory instructions` to view what gets injected.

For contributors working on this codebase, [CLAUDE.md](CLAUDE.md) provides project-specific guidance for AI assistants.

## License

MIT - See [LICENSE](LICENSE)
