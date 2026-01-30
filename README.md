# Spatial Memory MCP Server

A vector-based spatial memory system that treats knowledge as a navigable landscape, not a filing cabinet.

> **Project Status**: Phase 2 (Core Operations) complete. Phase 3 (Spatial Operations) planned.

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

## Current Capabilities (Phase 2)

Phase 2 core operations are complete with:

- **7 MCP tools**: remember, remember_batch, recall, nearby, forget, forget_batch, health
- Configuration system with environment variables and dependency injection
- LanceDB integration for vector storage with SQL injection prevention
- Embedding service supporting local models (sentence-transformers) and OpenAI API
- Pydantic data models with full validation
- Comprehensive error handling framework
- Enterprise features: connection pooling, auto-indexing, hybrid search, retry logic
- 111+ unit and integration tests passing

## Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1: Foundation | Complete | Config, Database, Embeddings, Models, Errors |
| Phase 2: Core Operations | Complete | `remember`, `recall`, `nearby`, `forget`, `health` |
| Phase 3: Spatial Operations | Planned | `journey`, `wander`, `regions`, `visualize` |
| Phase 4: Lifecycle Operations | Planned | `consolidate`, `extract`, `decay`, `reinforce` |
| Phase 5: Utilities | Planned | `stats`, `namespaces`, `export`, `import` |
| Phase 6: Polish & Release | Planned | Integration tests, docs, PyPI release |

## Installation

### Development Setup

```bash
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp
pip install -e ".[dev]"
```

### Future (after PyPI release)

```bash
pip install spatial-memory-mcp
```

Or with uvx:

```bash
uvx spatial-memory-mcp
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for all configuration options.

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
- `openai:text-embedding-3-small` - Fast, cheap (1536 dimensions)
- `openai:text-embedding-3-large` - Best quality (3072 dimensions)

## Usage

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "uvx",
      "args": ["spatial-memory-mcp"]
    }
  }
}
```

## Available Tools

### Core Operations (Phase 2 - Implemented)
| Tool | Description |
|------|-------------|
| `remember` | Store a memory in vector space |
| `remember_batch` | Store multiple memories efficiently |
| `recall` | Find memories semantically similar to a query |
| `nearby` | Find memories spatially close to a specific memory |
| `forget` | Remove a memory by ID |
| `forget_batch` | Remove multiple memories by IDs |
| `health` | Check system health status |

### Spatial Operations (Phase 3 - Planned)
| Tool | Description |
|------|-------------|
| `journey` | Interpolate a path between two memories using SLERP |
| `wander` | Random walk through memory space for serendipitous discovery |
| `regions` | Discover conceptual regions via HDBSCAN clustering |
| `visualize` | Generate visual representation (JSON/Mermaid/SVG) |

### Lifecycle Operations (Phase 4 - Planned)
| Tool | Description |
|------|-------------|
| `consolidate` | Merge similar memories |
| `extract` | Auto-extract memories from text |
| `decay` | Reduce importance of stale memories |
| `reinforce` | Boost importance of useful memories |

### Utility Operations (Phase 5 - Planned)
| Tool | Description |
|------|-------------|
| `stats` | Get memory statistics |
| `namespaces` | List, create, or delete namespaces |
| `export_memories` | Export memories to JSON |
| `import_memories` | Import memories from JSON |

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Type Checking

```bash
mypy spatial_memory/ --ignore-missing-imports
```

### Linting

```bash
ruff check spatial_memory/ tests/
```

## Troubleshooting

Having issues? See the [Troubleshooting Guide](docs/troubleshooting.md) for common problems and solutions.

## Architecture

See [SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md](SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md) for visual architecture documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Security

See [SECURITY.md](SECURITY.md) for security considerations and vulnerability reporting.

## For Claude Code Users

This project includes [CLAUDE.md](CLAUDE.md) with instructions for the Claude Code AI assistant to interact with the memory system.

## License

MIT - See [LICENSE](LICENSE)
