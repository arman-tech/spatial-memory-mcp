# Configuration Guide

Spatial Memory MCP Server is configured via environment variables with the `SPATIAL_MEMORY_` prefix.

## Configuration Methods

### 1. `.mcp.json` (Recommended for MCP Clients)

Create a `.mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "env": {
        "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true",
        "SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED": "false",
        "SPATIAL_MEMORY_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 2. Claude Desktop Configuration

Edit `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "env": {
        "SPATIAL_MEMORY_MEMORY_PATH": "/path/to/storage",
        "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true"
      }
    }
  }
}
```

### 3. Environment Variables

```bash
# Export before running
export SPATIAL_MEMORY_AUTO_DECAY_ENABLED=true
export SPATIAL_MEMORY_LOG_LEVEL=DEBUG
spatial-memory
```

Or inline:

```bash
SPATIAL_MEMORY_AUTO_DECAY_ENABLED=false spatial-memory
```

### 4. `.env` File

Create a `.env` file in your working directory:

```env
SPATIAL_MEMORY_MEMORY_PATH=./.spatial-memory
SPATIAL_MEMORY_LOG_LEVEL=INFO
SPATIAL_MEMORY_AUTO_DECAY_ENABLED=true
SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED=true
```

---

## Configuration Reference

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MEMORY_PATH` | `./.spatial-memory` | Path to LanceDB storage directory |
| `SPATIAL_MEMORY_ACKNOWLEDGE_NETWORK_FILESYSTEM_RISK` | `false` | Suppress network filesystem warnings |

### Embedding Model

v2.0 ships a **registry-of-one**: only `arctic-m-long` is supported. The model
runs on the PyTorch backend and produces 768-dimensional vectors over an
8192-token context window. Vector dimension is derived from the model — the
deprecated `SPATIAL_MEMORY_EMBEDDING_DIMENSIONS` override has been removed.
Hosted-API embedding providers from v1.x are no longer bundled; operators
who relied on them must pin `spatial-memory-mcp<2.0.0`.

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `arctic-m-long` | Canonical alias (only supported value under v2.0) |
| `SPATIAL_MEMORY_EMBEDDING_BACKEND` | `auto` | Backend: `auto` or `pytorch` (arctic runs on PyTorch) |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `SPATIAL_MEMORY_LOG_FORMAT` | `text` | Log format: `text` or `json` |

### Memory Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_DEFAULT_NAMESPACE` | `default` | Default namespace for memories |
| `SPATIAL_MEMORY_DEFAULT_IMPORTANCE` | `0.5` | Default importance (0.0-1.0) |

### Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_MAX_BATCH_SIZE` | `100` | Max memories per batch operation |
| `SPATIAL_MEMORY_MAX_RECALL_LIMIT` | `100` | Max results from recall |
| `SPATIAL_MEMORY_MAX_JOURNEY_STEPS` | `20` | Max steps in journey |
| `SPATIAL_MEMORY_MAX_WANDER_STEPS` | `20` | Max steps in wander |

---

## Auto-Decay Settings

Auto-decay automatically reduces the importance of memories over time, favoring recently accessed memories in search results.

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_AUTO_DECAY_ENABLED` | `true` | Enable decay calculation during recall |
| `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED` | `true` | Save decayed values to database |
| `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_BATCH_SIZE` | `100` | Batch size for persistence |
| `SPATIAL_MEMORY_AUTO_DECAY_PERSIST_FLUSH_INTERVAL_SECONDS` | `5.0` | How often to flush updates (seconds) |
| `SPATIAL_MEMORY_AUTO_DECAY_MIN_CHANGE_THRESHOLD` | `0.01` | Min change to trigger persist (1%) |
| `SPATIAL_MEMORY_AUTO_DECAY_MAX_QUEUE_SIZE` | `10000` | Max queued updates |
| `SPATIAL_MEMORY_AUTO_DECAY_FUNCTION` | `exponential` | Decay function: `exponential`, `linear`, or `step` |

### How Auto-Decay Works

1. **Configurable Decay Function**: Choose from `exponential`, `linear`, or `step`
   - `exponential`: Smooth decay following `2^(-t/half_life)`, best for most use cases
   - `linear`: Constant rate decay reaching 0 at 2x half_life
   - `step`: Discrete drops at half_life intervals (1.0 → 0.5 → 0.25)
2. **Default Half-Life**: 30 days (memory loses 50% importance after 30 days without access)
3. **Adaptive Half-Life**: Access count and importance affect decay rate
   - Frequently accessed memories decay slower (1.5× half-life per access, up to 20)
   - Higher importance memories also decay slower
4. **Minimum Floor**: Importance never drops below 10%

### Example Configurations

**Disable auto-decay entirely:**
```json
{
  "env": {
    "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "false"
  }
}
```

**Enable decay calculation but don't persist changes:**
```json
{
  "env": {
    "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true",
    "SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED": "false"
  }
}
```

**Aggressive persistence (frequent flushes):**
```json
{
  "env": {
    "SPATIAL_MEMORY_AUTO_DECAY_PERSIST_FLUSH_INTERVAL_SECONDS": "1.0",
    "SPATIAL_MEMORY_AUTO_DECAY_MIN_CHANGE_THRESHOLD": "0.001"
  }
}
```

---

## Full Example: `.mcp.json`

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "env": {
        "SPATIAL_MEMORY_MEMORY_PATH": "./my-project-memory",
        "SPATIAL_MEMORY_LOG_LEVEL": "INFO",
        "SPATIAL_MEMORY_DEFAULT_NAMESPACE": "project",
        "SPATIAL_MEMORY_DEFAULT_IMPORTANCE": "0.5",
        "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true",
        "SPATIAL_MEMORY_AUTO_DECAY_PERSIST_ENABLED": "true"
      }
    }
  }
}
```

## Full Example: Claude Desktop

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "env": {
        "SPATIAL_MEMORY_MEMORY_PATH": "/Users/me/memories",
        "SPATIAL_MEMORY_EMBEDDING_MODEL": "arctic-m-long",
        "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true",
        "SPATIAL_MEMORY_AMBIENT_RECALL": "on"
      }
    }
  }
}
```

---

## Ambient Recall (v2.0)

Ambient recall fires the `UserPromptSubmit` hook before each prompt reaches
the model and injects the top-K most relevant memories as additional context.
The hook fail-opens on timeout or exception (no stdout, no error).

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_AMBIENT_RECALL` | `on` | Master switch; set to `off` to disable injection. |
| `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K` | `5` | How many memories to inject. |
| `SPATIAL_MEMORY_AMBIENT_RECALL_NAMESPACE` | _(unset)_ | Restrict ambient recall to a single namespace. Leave unset to search all namespaces. |

The hook is registered for Claude Code via `setup-hooks --client claude-code`
or by installing the spatial-memory plugin (zero-config). Cursor has no
equivalent event and is not registered for ambient recall in v2.0.

---

## Advanced Settings

For power users, additional settings are available for:

- **Indexing**: `SPATIAL_MEMORY_VECTOR_INDEX_THRESHOLD`, `SPATIAL_MEMORY_AUTO_CREATE_INDEXES`, `SPATIAL_MEMORY_INDEX_TYPE`
- **Full-Text Search**: `SPATIAL_MEMORY_ENABLE_FTS_INDEX`, `SPATIAL_MEMORY_FTS_LANGUAGE`
- **Rate Limiting**: `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_ENABLED`, `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_RATE`
- **Caching**: `SPATIAL_MEMORY_RESPONSE_CACHE_ENABLED`, `SPATIAL_MEMORY_RESPONSE_CACHE_MAX_SIZE`
- **Manual Decay** (via `decay` tool): `SPATIAL_MEMORY_DECAY_DEFAULT_HALF_LIFE_DAYS`, `SPATIAL_MEMORY_DECAY_MIN_IMPORTANCE_FLOOR`
- **Export/Import**: `SPATIAL_MEMORY_EXPORT_ALLOWED_PATHS`, `SPATIAL_MEMORY_IMPORT_MAX_FILE_SIZE_MB`

See the full list in [`spatial_memory/config.py`](../spatial_memory/config.py).

---

## Verifying Configuration

Check that your settings are applied by enabling debug logging:

```json
{
  "env": {
    "SPATIAL_MEMORY_LOG_LEVEL": "DEBUG"
  }
}
```

Then check the server logs for configuration values at startup.
