# Spatial Memory MCP Server

[![PyPI version](https://badge.fury.io/py/spatial-memory-mcp.svg)](https://pypi.org/project/spatial-memory-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A vector-based spatial memory system that treats knowledge as a navigable landscape, not a filing cabinet.

> **Version 1.10.1** — Production-ready with 2,300+ tests across Windows, macOS, and Linux.

## What It Does

Spatial Memory gives LLMs persistent, semantic memory through the [Model Context Protocol](https://modelcontextprotocol.io/). It goes beyond simple vector storage:

- **Semantic search** — find memories by meaning, not keywords
- **Spatial navigation** — `journey` between concepts, `wander` to discover unexpected connections
- **Cognitive dynamics** — memories `decay`, `reinforce`, `consolidate`, and `extract` like human cognition
- **Automatic capture** — hook scripts silently save decisions, fixes, and insights during coding sessions
- **Multi-client** — works with Claude Code, Cursor, Windsurf, Antigravity, and VS Code Copilot

## Quick Start

### Claude Code Plugin (Recommended)

Zero-config install — hooks, MCP server, and cognitive offloading all set up automatically:

```bash
# Add the marketplace
claude plugin marketplace add arman-tech/spatial-memory-mcp

# Install the plugin
claude plugin install spatial-memory@spatial-memory-marketplace
```

The plugin registers 3 hooks (PostToolUse, PreCompact, Stop) and starts the MCP server. Memories are captured automatically as you work.

> **Note (Windows):** Plugin SessionStart hooks freeze terminal input in Claude Code v2.1.37. After installing the plugin, also run `spatial-memory setup-hooks --client claude-code` and copy the SessionStart hook into `.claude/settings.json`.

### pip Install (Manual Setup)

```bash
pip install spatial-memory-mcp
```

Then add to your MCP client config (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "python",
      "args": ["-m", "spatial_memory"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

### Other Clients

Generate client-specific configuration with the `setup_hooks` MCP tool or CLI:

```bash
# Generate config for your client
spatial-memory setup-hooks --client cursor
spatial-memory setup-hooks --client windsurf
spatial-memory setup-hooks --client antigravity
spatial-memory setup-hooks --client vscode-copilot

# JSON output for scripting
spatial-memory setup-hooks --client claude-code --json
```

| Client | Tier | Hooks | MCP | Notes |
|--------|------|-------|-----|-------|
| Claude Code | 1 | Native | Yes | Full auto-capture via plugin or settings |
| Cursor | 2 | Via adapter | Yes | Adapter translates to native hook format |
| Windsurf | 3 | Rules only | Yes | MCP works; add rules file for best results |
| Antigravity | 3 | Rules only | Yes | MCP works; add GEMINI.md for best results |
| VS Code Copilot | 3 | Rules only | Yes | MCP works; add copilot-instructions.md |

## How It Works

### Cognitive Offloading (Auto-Capture)

With hooks enabled, memory capture happens silently in the background:

1. **PostToolUse** — after each tool call, analyzes the conversation for decisions, errors, and solutions
2. **PreCompact** — before context compaction, scans the transcript for insights that would be lost
3. **Stop** — at session end, captures any remaining valuable context

Content is classified into 3 tiers:
- **Tier 1** (auto-save): Decisions, bug fixes, error root causes, architecture choices
- **Tier 2** (ask first): Patterns, preferences, configuration discoveries, workarounds
- **Tier 3** (skip): Trivial observations, duplicates, speculative information

Secrets (API keys, tokens, passwords) are automatically redacted before storage.

### 23 MCP Tools

| Category | Tools |
|----------|-------|
| **Core** | `remember`, `remember_batch`, `recall`, `nearby`, `forget`, `forget_batch` |
| **Spatial** | `journey`, `wander`, `regions`, `visualize` |
| **Lifecycle** | `decay`, `reinforce`, `extract`, `consolidate` |
| **Utility** | `stats`, `namespaces`, `delete_namespace`, `rename_namespace`, `export_memories`, `import_memories`, `hybrid_recall`, `health` |
| **Setup** | `setup_hooks` |

See [docs/API.md](docs/API.md) for complete parameter and return type documentation.

### Spatial Navigation

Navigate knowledge like a landscape:

| Tool | What It Does |
|------|-------------|
| `journey` | SLERP interpolation between two memories — discover what's conceptually in between |
| `wander` | Temperature-based random walk — find unexpected connections |
| `regions` | HDBSCAN clustering — see how your knowledge self-organizes |
| `visualize` | UMAP projection — view your memory space in 2D/3D (JSON, Mermaid, SVG) |

## Configuration

Settings via environment variables or `.env` file. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MEMORY_PATH` | `./.spatial-memory` | LanceDB storage directory |
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model (or `openai:text-embedding-3-small`) |
| `SPATIAL_MEMORY_EMBEDDING_BACKEND` | `auto` | `auto` (ONNX if available), `onnx`, or `pytorch` |
| `SPATIAL_MEMORY_OPENAI_API_KEY` | — | Required only for OpenAI embeddings |
| `SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED` | `false` | Enable queue-based auto-capture pipeline |
| `SPATIAL_MEMORY_AUTO_DECAY_ENABLED` | `true` | Automatic importance decay over time |
| `SPATIAL_MEMORY_LOG_LEVEL` | `INFO` | Logging verbosity |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference including auto-decay tuning, rate limiting, and connection pool settings.

## CLI Commands

```bash
spatial-memory serve                     # Start the MCP server (default)
spatial-memory instructions              # View auto-injected MCP instructions
spatial-memory setup-hooks --client X    # Generate hook config for client X
spatial-memory migrate --status          # Check database migration status
spatial-memory --version                 # Show version
```

## Security

- **Path traversal prevention** on all file operations
- **SQL injection detection** (13 patterns)
- **Secret redaction** in cognitive offloading (AWS, GitHub, Stripe, OpenAI, SSH keys, JWTs, etc.)
- **Input validation** via Pydantic models on all tool inputs
- **Error sanitization** — internal errors return reference IDs, not stack traces
- **Secure credentials** — API keys stored as `SecretStr`

## Development

```bash
# Install from source
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp
pip install -e ".[dev]"

# Run tests
pytest tests/ -v              # Unit tests only
pytest tests/ -v -m ""        # All tests (unit + integration)

# Quality checks
ruff check spatial_memory/ tests/
ruff format --check spatial_memory/ tests/
mypy spatial_memory/
```

## Architecture

Clean Architecture with ports/adapters pattern:

```
spatial_memory/
├── server.py       # MCP server + tool handlers
├── factory.py      # Dependency injection container
├── config.py       # Pydantic settings
├── core/           # Database, embeddings, models, validation, security
├── services/       # Business logic (memory, spatial, lifecycle, utility)
├── adapters/       # LanceDB repository, project detection, git utils
├── ports/          # Protocol interfaces
├── hooks/          # Cognitive offloading hook scripts
└── tools/          # MCP tool definitions + setup_hooks generator
```

See [SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md](SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md) for visual documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete API reference for all 23 tools |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Full configuration reference |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step tutorial |
| [docs/TECHNICAL_HIGHLIGHTS.md](docs/TECHNICAL_HIGHLIGHTS.md) | Algorithm deep-dives (SLERP, HDBSCAN, UMAP) |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Performance benchmarks |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |

## Supported Platforms

- **Windows 11**, **macOS** (latest), **Linux** (Fedora, Ubuntu, Linux Mint)
- Python 3.10+
- CI tested across 3 OS x 4 Python versions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v -m ""`)
5. Submit a pull request

For contributors using AI assistants, see [CLAUDE.md](CLAUDE.md) for project-specific guidance.

## License

MIT — See [LICENSE](LICENSE)
