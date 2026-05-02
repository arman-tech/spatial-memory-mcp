# Spatial Memory MCP Server

[![PyPI version](https://badge.fury.io/py/spatial-memory-mcp.svg)](https://pypi.org/project/spatial-memory-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A persistent semantic memory system for LLMs via the [Model Context Protocol](https://modelcontextprotocol.io/) that treats knowledge as a navigable landscape, not a filing cabinet.

> **Version 2.0** — Drop-in semantic memory for Claude Code and Cursor. Local-only, fresh-install-only, registry-of-one (`arctic-m-long`, 768-dim, 8K context, PyTorch backend). See [internal-docs/v2-architectural-decisions.md](internal-docs/v2-architectural-decisions.md) for the locked decisions D1–D7 that shape this release.

**Your AI assistant forgets everything between sessions. Spatial Memory fixes that.** It gives Claude Code, Cursor, and any MCP client a persistent brain — memories that fade when stale, sharpen with use, and organize themselves into a navigable knowledge graph. Install in one command, capture knowledge automatically, and let your AI build on what it learned yesterday.

## Why Spatial Memory?

Most memory servers store and retrieve. Spatial Memory **thinks** about your knowledge.

### Memories That Fade Like Yours Do

Other memory tools treat every piece of information as equally important forever. Spatial Memory applies **time-based decay** — old, unused memories gradually lose importance while frequently accessed knowledge stays sharp. The result: your AI assistant surfaces what's relevant *now*, not what was relevant six months ago. Decay is automatic and configurable — adjust half-life, decay curves (exponential, linear, step), and minimum importance floors. Memories accessed frequently decay slower, just like human recall.

**Why this approach?** The cognitive memory model is inspired by established research:

- **[Ebbinghaus, H. (1885)](https://psychclassics.yorku.ca/Ebbinghaus/index.htm)** — *Memory: A Contribution to Experimental Psychology*. The foundational research on the forgetting curve showing how memory retention decays exponentially over time. Our exponential decay function directly models this curve.
- **[Settles, B. & Meeder, B. (2016)](https://aclanthology.org/P16-1174/)** — *A Trainable Spaced Repetition Model for Language Learning*. Duolingo's half-life regression (HLR) algorithm for optimizing memory retention. Our configurable half-life and access-count weighting draw from this work.
- **[FSRS Algorithm](https://github.com/open-spaced-repetition/fsrs4anki)** — Free Spaced Repetition Scheduler. A modern open-source algorithm for optimizing review intervals based on memory research. Informed our adaptive decay that slows for frequently accessed memories.

### Zero-Effort Memory Capture

You shouldn't have to stop coding to tell your AI "remember this." With **cognitive offloading**, hook scripts run silently in the background and capture decisions, bug fixes, error root causes, and architecture choices *as they happen* — no manual `remember` calls needed.

- **PostToolUse** — captures insights after each tool call
- **PreCompact** — saves knowledge before context window compaction would erase it
- **Stop** — grabs remaining valuable context at session end
- **UserPromptSubmit** — runs ambient recall on the user's prompt and injects the top-K matches as context (Claude Code only — see [Ambient recall](#ambient-recall))

Content is classified into tiers (auto-save, ask-first, skip) and secrets are automatically redacted before storage.

### Navigate and Search Like No Other Memory Server

Traditional memory is a search box. Spatial Memory is a **map with a search engine**. You get **hybrid search** — combined vector similarity *and* keyword matching with a tunable alpha — plus spatial tools that let you explore the space *between* and *around* your memories:

| Tool | What It Does |
|------|-------------|
| `hybrid_recall` | Combined vector + keyword search with tunable balance (alpha 0.0-1.0) — find memories that match both meaning *and* specific terms |
| `journey` | Walk the conceptual path between two memories using SLERP interpolation — discover what lies in between "authentication" and "performance" |
| `wander` | Take a temperature-controlled random walk — stumble into unexpected connections you'd never think to search for |
| `regions` | See how your knowledge self-organizes into clusters via HDBSCAN — find the natural shape of what you know |
| `visualize` | Project your memory space into 2D/3D via UMAP — render as JSON, Mermaid diagrams, or SVG |

### Local-First, Whole-Document Embeddings

No GPU. No hosted API. v2.0 ships a registry-of-one (D1): **[Snowflake Arctic Embed M Long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long)** (768-dim, 8K context, normalized, asymmetric prefixes), loaded via the **PyTorch** backend and preloaded at MCP startup so the first tool call doesn't pay model-load latency. The 8K context lets us embed whole documents — decisions, summaries, code excerpts, conversation snippets — without a chunking pipeline (D2).

- **CPU-only** — no CUDA, no GPU drivers, works everywhere Python runs
- **`trust_remote_code=True`** is scoped narrowly to the Snowflake/arctic-m-long modeling code (NomicBert custom code from a known publisher); see D1
- **Cold start ~5–15 s** at MCP launch, observable via `embedder_cold_start_seconds`; per-embed warm 5–30 ms

---

## How Is This Different?

Most MCP memory servers are vector stores with semantic recall — store text, search by similarity, retrieve results. Spatial Memory starts there but adds what they don't: **time-based decay** that fades stale knowledge automatically, **cognitive offloading hooks** that capture decisions and errors without manual calls, **spatial navigation** (SLERP interpolation, random walks, HDBSCAN clustering) for exploring the space *between* memories, and **hybrid search** that combines vector similarity with keyword matching. If you need a simple key-value memory, any of those will work. If you want memory that behaves more like human recall — fading, reinforcing, and organizing itself — this is the one.

## Quick Start

> **Windows users:** This plugin requires `uvx` ([uv](https://docs.astral.sh/uv/)). Install it first:
> ```
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

### Claude Code (Recommended)

**Option A — Plugin (zero-config):**

```bash
claude plugin marketplace add arman-tech/spatial-memory-mcp
claude plugin install spatial-memory@spatial-memory-marketplace
```

This installs the plugin globally (user scope) — available across all your projects. To install for the current project only:

```bash
cd /path/to/your/project
claude plugin install spatial-memory@spatial-memory-marketplace --scope project
```

That's it. The plugin registers 4 hooks (PostToolUse, PreCompact, Stop, UserPromptSubmit), starts the MCP server, and begins capturing knowledge automatically as you work. Cursor users get the first three; the prompt-time recall hook is Claude-Code-only in v2.0 (see [Ambient recall](#ambient-recall)).

**Option B — Manual MCP config:**

Add to your Claude Code settings (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "uvx",
      "args": ["--from", "spatial-memory-mcp", "spatial-memory", "serve"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

No `pip install` needed — `uvx` fetches the package from PyPI automatically.

### Cursor

From your project root, one command writes `.cursor/mcp.json`, `.cursor/hooks.json`, and `.cursor/rules/spatial-memory.mdc`:

```bash
pip install spatial-memory-mcp
cd /path/to/your/project
spatial-memory init --client cursor
```

### Claude Desktop / Other MCP Clients

Add to your MCP client config (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "uvx",
      "args": ["--from", "spatial-memory-mcp", "spatial-memory", "serve"],
      "env": {
        "SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED": "true"
      }
    }
  }
}
```

No `pip install` needed — `uvx` fetches the package from PyPI automatically.

| Client | Install | Hooks | Notes |
|--------|---------|-------|-------|
| Claude Code | Plugin or pip | Native (auto) | Full auto-capture via plugin or manual settings |
| Cursor | pip + `init` | Native (auto) | One-command setup via `spatial-memory init --client cursor` |
| Claude Desktop | pip | Manual | Add MCP config, hooks require manual setup |
| Other MCP clients | pip | Manual | Any client that speaks MCP works |

## How It Works

### Cognitive Offloading (Auto-Capture)

Four hooks (PostToolUse, PreCompact, Stop, UserPromptSubmit) run silently in the background to capture knowledge as you work — see [Why Spatial Memory?](#zero-effort-memory-capture) for the full description. Captured content is classified into tiers:

| Tier | Behavior | What's Captured |
|------|----------|-----------------|
| **1** | Auto-save | Decisions, bug fixes, error root causes, architecture choices |
| **2** | Ask first | Patterns, preferences, configuration discoveries, workarounds |
| **3** | Skip | Trivial observations, duplicates, speculative information |

Secrets (API keys, tokens, passwords) are automatically redacted before storage.

### 25 MCP Tools

| Category | Tools |
|----------|-------|
| **Core** | `remember`, `remember_batch`, `recall`, `nearby`, `forget`, `forget_batch` |
| **Spatial** | `journey`, `wander`, `regions`, `visualize` |
| **Lifecycle** | `decay`, `reinforce`, `extract`, `consolidate` |
| **Utility** | `stats`, `namespaces`, `delete_namespace`, `rename_namespace`, `export_memories`, `import_memories`, `hybrid_recall`, `health` |
| **Cross-corpus** | `discover_connections`, `corpus_bridges` |
| **Setup** | `setup_hooks` |

See [docs/API.md](docs/API.md) for complete parameter and return type documentation.

## Configuration

Settings via environment variables or `.env` file. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MEMORY_PATH` | `./.spatial-memory` | LanceDB storage directory |
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `arctic-m-long` | Embedding model — registry-of-one in v2.0 (D1) |
| `SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED` | `false` | Enable queue-based auto-capture pipeline |
| `SPATIAL_MEMORY_AUTO_DECAY_ENABLED` | `true` | Automatic importance decay over time |
| `SPATIAL_MEMORY_AMBIENT_RECALL` | `on` | Toggle the UserPromptSubmit ambient-recall hook (`off` to disable) |
| `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K` | `5` | Top-K memories injected as context per prompt |
| `SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE` | _(unset)_ | Restrict ambient recall to a single namespace; leave unset to search all namespaces |
| `SPATIAL_MEMORY_LOG_LEVEL` | `INFO` | Logging verbosity |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference including auto-decay tuning, rate limiting, and connection pool settings.

## Ambient recall

v2.0 adds a `UserPromptSubmit` hook (D3): every time you send a prompt to Claude Code, the dispatcher runs `hybrid_recall` against the prompt and injects the top-K matching memories as context before the model sees the message. No tool calls, no manual `recall` — your assistant just *knows* what you wrote down yesterday. The hook is fail-open: if the recall path raises or times out (5 s envelope), no output is emitted and your prompt continues unaffected.

Three env-var toggles control behavior:

| Variable | Default | Effect |
|----------|---------|--------|
| `SPATIAL_MEMORY_AMBIENT_RECALL` | `on` | Set to `off` to disable; the hook exits with no output |
| `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K` | `5` | Top-K passed to `hybrid_recall` |
| `SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE` | _(unset)_ | Scope recall to a specific namespace; leave unset to search all namespaces |

**Cursor:** Cursor's hook surface (`postToolUse` / `preCompact` / `stop`) has no `UserPromptSubmit`-equivalent event in v2.0; ambient recall is therefore Claude-Code-only. Cursor users still get post-tool-use ingestion and pre-compact distillation.

## Observability

Every metric is captured at runtime to a per-process daily JSONL journal at `<memory_path>/metrics/<YYYY-MM-DD>-<pid>-<startepoch>.jsonl` (D7). Lines stay under PIPE_BUF for atomic appends; retention is 7 days by default and auto-pruned on startup. Read the data with `stats(include_observations=True)` for a `RecentObservations` block summarizing the last 7 days (p50/p95 percentile pairs for the five §10.2 metrics, the capture cap-hit rate, and the total sample size over the window) — useful for users without a Prometheus stack. The deferred decisions in D1–D7 each link to a journal metric (`capture_size_tokens`, `capture_cap_hit_total`, `recall_result_count`, `reinforced_memory_rank`, `embedder_cold_start_seconds`) that informs whether to reopen them in v2.x+.

## Migration from v1.x

v2.0 is a **fresh-install release**: the schema bumps to `2.0.0` and v1 databases are not auto-migrated (D5). The supported recipe is `spatial-memory export` on v1.x → upgrade → `spatial-memory import` into v2.0 (the export format is provider-agnostic, so memories re-embed under arctic-m-long on import). Backups stay readable; no data loss; one short downtime window.

**OpenAI provider note (D1):** v2.0 removes the hosted `openai:*` embedding-provider path. The `SPATIAL_MEMORY_OPENAI_API_KEY` and `SPATIAL_MEMORY_OPENAI_EMBEDDING_MODEL` settings are gone, and the `[openai]` extra is dropped from `pyproject.toml`. If you were on a v1 OpenAI configuration you have two options: (a) export from v1.x and import into v2.0 to re-embed under the local arctic-m-long default, or (b) pin v1.x if a hosted embedder is required. This is a deliberate scope reduction (local-first, IDE-companion alignment under D4), not a regression — see D1 for the full rationale.

## CLI Commands

```bash
# Server
spatial-memory serve                     # Start the MCP server (default)

# Setup
spatial-memory init --client cursor      # Auto-configure Cursor (writes 3 files)
spatial-memory setup-hooks --client X    # Generate hook config for Claude Code or Cursor

# Database maintenance
spatial-memory namespaces                # List all namespaces with memory counts
spatial-memory consolidate <namespace>   # Merge duplicate memories (dry run by default)
spatial-memory consolidate <ns> --no-dry-run  # Actually apply merges
spatial-memory migrate --status          # Check database migration status

# Utilities
spatial-memory hook <event> --client X   # Run a hook event (used by hook configs)
spatial-memory instructions              # View auto-injected MCP instructions
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

```mermaid
graph TD
    Client["MCP Clients<br>Claude Code · Cursor"] --> Server["MCP Server<br>server.py · 25 tools"]
    Hooks["Hook Dispatcher<br>PostToolUse · PreCompact · Stop · UserPromptSubmit"] -.->|file queue / stdout| Server
    Server --> Services["Services<br>Memory · Spatial · Lifecycle · Utility"]
    Services --> DB["Database Facade<br>database.py · 8 managers"]
    Services --> Emb["Embeddings<br>embeddings.py"]
    DB --> Lance["LanceDB"]
    Emb --> ST["sentence-transformers<br>arctic-m-long · PyTorch · 768-dim · 8K context"]
```

```
spatial_memory/
├── server.py       # MCP server + tool handlers
├── factory.py      # Dependency injection container
├── config.py       # Pydantic settings
├── core/           # Database, embeddings, models, validation, security
├── services/       # Business logic (memory, spatial, lifecycle, utility)
├── adapters/       # LanceDB repository, project detection, git utils
├── ports/          # Protocol interfaces
├── hooks/          # Cognitive offloading dispatcher + pipeline
├── tools/          # MCP tool definitions + setup_hooks generator
└── migrations/     # Database schema migrations
```

See [SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md](SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md) for visual documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete API reference for all 25 tools |
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
