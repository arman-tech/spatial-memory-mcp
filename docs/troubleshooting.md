# Troubleshooting Guide

## Common Issues

### 1. Legacy `openai:*` Embedding Configuration Detected

**Symptom:** Server logs surface a `legacy-openai` configuration error or
`spatial-memory doctor` reports `embedding.match=openai-legacy`.

**Cause:** v2.0 removed the hosted OpenAI embedding provider (decision D1 —
registry-of-one). `SPATIAL_MEMORY_EMBEDDING_MODEL=openai:*`,
`SPATIAL_MEMORY_OPENAI_API_KEY`, and `SPATIAL_MEMORY_OPENAI_EMBEDDING_MODEL`
are no longer accepted.

**Solution:** Either migrate to the v2.0 default (`arctic-m-long`) or pin
the v1.x line if a hosted embedder is required.

```bash
# Recommended: drop the openai:* alias and re-embed under arctic-m-long
SPATIAL_MEMORY_EMBEDDING_MODEL=arctic-m-long
```

To re-embed an existing v1 corpus, export from v1.x and import into v2.0
(see [Configuration](CONFIGURATION.md) and the v2.0 release notes in
`CHANGELOG.md`). To stay on v1.x, pin `spatial-memory-mcp<2.0.0`.

### 2. "Storage path not writable" Error

**Symptom:** Server fails to start with permission error.

**Cause:** The storage directory cannot be created or written to.

**Solution:**
- Check directory permissions
- Ensure parent directory exists
- Use a different path

```bash
# Check permissions
ls -la ~/.spatial-memory

# Create directory with correct permissions
mkdir -p ~/.spatial-memory
chmod 755 ~/.spatial-memory

# Or use a different path in .env
SPATIAL_MEMORY_MEMORY_PATH=/path/to/writable/directory
```

### 3. Slow Search Performance

**Symptom:** Recall operations take a long time.

**Causes and Solutions:**

1. **Large dataset without index**: Create vector index manually or enable auto-indexing
   ```bash
   SPATIAL_MEMORY_AUTO_CREATE_INDEXES=true
   SPATIAL_MEMORY_VECTOR_INDEX_THRESHOLD=10000
   ```

2. **Low nprobes**: Increase `SPATIAL_MEMORY_INDEX_NPROBES`
   ```bash
   SPATIAL_MEMORY_INDEX_NPROBES=30
   ```

3. **Many small fragments**: Run optimization via the health tool

### 4. High Memory Usage

**Symptom:** Server uses excessive memory.

**Solutions:**
- Reduce `SPATIAL_MEMORY_CONNECTION_POOL_MAX_SIZE`
- Run optimization to compact data
- Use smaller batch sizes

```bash
SPATIAL_MEMORY_CONNECTION_POOL_MAX_SIZE=5
SPATIAL_MEMORY_MAX_BATCH_SIZE=50
```

### 5. Embedding Model Download Fails

**Symptom:** First run fails to download sentence-transformers model.

**Causes and Solutions:**
1. **Network issues**: Check internet connectivity
2. **Proxy required**: Configure HTTP_PROXY/HTTPS_PROXY environment variables
3. **Disk space**: Ensure sufficient space for the `arctic-m-long` model download (~600 MB; `model.safetensors` ≈ 547 MB plus tokenizer/config files)

### 6. "Browser executable not found" Error (Playwright)

**Symptom:** If using Playwright tools alongside, browser not installed.

**Solution:** Not related to Spatial Memory MCP. This is a Playwright issue.

## Debugging

### Enable Debug Logging

```bash
export SPATIAL_MEMORY_LOG_LEVEL=DEBUG
```

Or in `.env`:
```bash
SPATIAL_MEMORY_LOG_LEVEL=DEBUG
```

### Check Health

Use the `health` tool to check system status:

```json
{
  "tool": "health",
  "arguments": {
    "verbose": true
  }
}
```

This returns:
- Overall health status (healthy, degraded, unhealthy)
- Individual component status (database, embeddings, storage)
- Latency metrics
- Ready/alive indicators

### View Index Status

Check health metrics for index information. The health tool reports:
- Whether indexes are created
- Index type and configuration
- Storage statistics

### Common Log Messages

| Log Message | Meaning |
|-------------|---------|
| `Auto-detected embedding dimensions: N` | Model loaded successfully |
| `Configuration warning: ...` | Non-fatal configuration issue |
| `Received SIGTERM, initiating graceful shutdown` | Server shutting down cleanly |
| `Retry attempt N for storage operation` | Transient storage error, retrying |

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/arman-tech/spatial-memory-mcp/issues)
2. Enable debug logging and review the output
3. Open a new issue with:
   - Error message and stack trace
   - Configuration (sanitize API keys)
   - Steps to reproduce
