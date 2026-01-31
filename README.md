# Spatial Memory MCP Server

A vector-based spatial memory system that treats knowledge as a navigable landscape, not a filing cabinet.

> **Project Status**: All phases complete. Production-ready with 1094 tests passing.

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

## Features

- **21 MCP tools** across 4 categories (core, spatial, lifecycle, utility)
- **Clean Architecture** with ports/adapters pattern for testability
- **LanceDB** vector storage with automatic indexing
- **Dual embedding support**: Local (sentence-transformers) or OpenAI API
- **Enterprise features**: Connection pooling, retry logic, batch operations
- **Comprehensive security**: Path validation, SQL injection prevention, input sanitization
- **1094 tests** including security edge cases

## Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1: Foundation | Complete | Config, Database, Embeddings, Models, Errors |
| Phase 2: Core Operations | Complete | `remember`, `recall`, `nearby`, `forget` |
| Phase 3: Spatial Operations | Complete | `journey`, `wander`, `regions`, `visualize` |
| Phase 4: Lifecycle Operations | Complete | `consolidate`, `extract`, `decay`, `reinforce` |
| Phase 5: Utilities | Complete | `stats`, `namespaces`, `export`, `import`, `hybrid_recall` |
| Phase 6: Polish & Release | In Progress | PyPI release pending |

## Installation

### Development Setup

```bash
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp
pip install -e ".[dev]"
```

### With OpenAI Support

```bash
pip install -e ".[dev,openai]"
```

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

## Available Tools (21 Total)

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
- **SQL Injection Prevention**: 15+ dangerous patterns detected and blocked
- **Input Validation**: Pydantic models validate all inputs
- **Error Sanitization**: Internal errors return reference IDs, not stack traces
- **Secure Credential Handling**: API keys stored as SecretStr

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

This project includes [CLAUDE.md](CLAUDE.md) with instructions for the Claude Code AI assistant to interact with the memory system.

## License

MIT - See [LICENSE](LICENSE)
