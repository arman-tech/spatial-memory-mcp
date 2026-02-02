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

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name or `openai:model-name` |
| `SPATIAL_MEMORY_EMBEDDING_DIMENSIONS` | `384` | Vector dimensions (auto-detected) |
| `SPATIAL_MEMORY_EMBEDDING_BACKEND` | `auto` | Backend: `auto`, `onnx`, or `pytorch` |

### OpenAI (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_MEMORY_OPENAI_API_KEY` | `null` | OpenAI API key for embeddings |
| `SPATIAL_MEMORY_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model to use |

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

### How Auto-Decay Works

1. **Exponential Decay**: `effective_importance = importance × 2^(-days_since_access / half_life)`
2. **Default Half-Life**: 30 days (memory loses 50% importance after 30 days without access)
3. **Access Count Bonus**: Frequently accessed memories decay slower
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
        "SPATIAL_MEMORY_EMBEDDING_MODEL": "openai:text-embedding-3-small",
        "SPATIAL_MEMORY_OPENAI_API_KEY": "sk-...",
        "SPATIAL_MEMORY_AUTO_DECAY_ENABLED": "true"
      }
    }
  }
}
```

---

## Advanced Settings

For power users, additional settings are available for:

- **Indexing**: `SPATIAL_MEMORY_VECTOR_INDEX_THRESHOLD`, `SPATIAL_MEMORY_AUTO_CREATE_INDEXES`, `SPATIAL_MEMORY_INDEX_TYPE`
- **Full-Text Search**: `SPATIAL_MEMORY_ENABLE_FTS_INDEX`, `SPATIAL_MEMORY_FTS_LANGUAGE`
- **Rate Limiting**: `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_ENABLED`, `SPATIAL_MEMORY_RATE_LIMIT_PER_AGENT_RATE`
- **Circuit Breaker**: `SPATIAL_MEMORY_CIRCUIT_BREAKER_ENABLED`, `SPATIAL_MEMORY_CIRCUIT_BREAKER_FAILURE_THRESHOLD`
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
