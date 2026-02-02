# Getting Started with Spatial Memory MCP

This guide walks you through setting up and using Spatial Memory MCP for the first time.

## Prerequisites

- Python 3.10 or higher
- An MCP-compatible client (Claude Desktop, Claude Code, etc.)

## Step 1: Installation

### Option A: Install from PyPI (Recommended)

```bash
pip install spatial-memory-mcp
```

### Option B: Install from Source

```bash
git clone https://github.com/arman-tech/spatial-memory-mcp.git
cd spatial-memory-mcp
pip install -e .
```

### Verify Installation

```bash
spatial-memory --version
```

You should see output like:
```
spatial-memory 1.8.0
```

## Step 2: Configuration

### For Claude Desktop

Add to your Claude Desktop configuration file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "args": ["serve"]
    }
  }
}
```

### For Claude Code

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "spatial-memory": {
      "command": "spatial-memory",
      "args": ["serve"]
    }
  }
}
```

### Custom Storage Location (Optional)

Set `SPATIAL_MEMORY_PATH` to store memories in a specific location:

```bash
export SPATIAL_MEMORY_PATH=~/.my-memories
```

## Step 3: Your First Memory

Once configured, restart your MCP client. You can now interact with Spatial Memory through natural conversation.

### Store a Memory

Ask Claude to remember something:

> "Remember that the API uses JWT tokens with a 24-hour expiration, and refresh tokens last 7 days."

Behind the scenes, this calls the `remember` tool:
```json
{
  "content": "The API uses JWT tokens with a 24-hour expiration, and refresh tokens last 7 days.",
  "namespace": "default",
  "importance": 0.7
}
```

### Recall Memories

Ask Claude to recall related information:

> "What do you remember about authentication?"

This performs a semantic search that finds memories related to "authentication" even if that exact word wasn't used.

### Organize with Namespaces

Use namespaces to organize memories by project or topic:

> "Remember in the 'ProjectAlpha' namespace: The database uses PostgreSQL 15 with read replicas."

### Add Tags for Filtering

Tags help categorize memories:

> "Remember with tags 'architecture' and 'database': We chose event sourcing for the order service."

## Step 4: Spatial Navigation

Spatial Memory treats your memories as points in semantic space. You can navigate between them.

### Journey Between Concepts

Discover the conceptual path between two memories:

> "Show me the journey from 'authentication' to 'database design'"

This uses SLERP (Spherical Linear Interpolation) to find memories along the semantic path.

### Wander Through Memory Space

Explore related concepts with controlled randomness:

> "Wander through my memories starting from the authentication topic"

Temperature controls exploration:
- Low (0.1-0.3): Stay close to related concepts
- Medium (0.4-0.6): Balanced exploration
- High (0.7-1.0): More random, serendipitous discoveries

### Discover Regions

Find natural clusters in your memories:

> "What regions or clusters exist in my memories?"

This uses HDBSCAN clustering to identify topic groupings.

## Step 5: Memory Lifecycle

### Reinforce Important Memories

Boost the importance of frequently-used memories:

> "Reinforce the memory about JWT tokens"

### Apply Decay

Reduce importance of old, unused memories:

> "Apply decay to memories in the 'old-project' namespace"

### Consolidate Duplicates

Merge similar memories to reduce redundancy:

> "Consolidate memories in the default namespace"

## Step 6: Export and Backup

### Export Memories

Save memories to a file for backup:

> "Export all memories to memories-backup.parquet"

Supported formats: `.parquet` (recommended), `.json`, `.csv`

### Import Memories

Restore from a backup:

> "Import memories from memories-backup.parquet"

## Common Use Cases

### Project Context

```
Remember in 'MyProject': The frontend uses React 18 with TypeScript and Zustand for state.
Remember in 'MyProject': API endpoints follow REST conventions at /api/v1/*.
Remember in 'MyProject': Deployment uses Docker with GitHub Actions CI/CD.
```

### Decision Log

```
Remember with tags 'decision', 'architecture': We chose PostgreSQL over MongoDB because of complex relational queries.
Remember with tags 'decision', 'frontend': Selected Tailwind CSS for consistent styling with utility classes.
```

### Debugging Notes

```
Remember with tags 'bug', 'resolved': Memory leak in useEffect was caused by missing cleanup function.
Remember with tags 'workaround': Safari requires explicit font-display: swap for web fonts.
```

## Troubleshooting

### Server Not Starting

Check if the embedding model downloaded successfully:
```bash
spatial-memory serve
```

First run downloads ~80MB model. Allow time for completion.

### Memories Not Found

Ensure you're searching in the correct namespace:
```
Recall memories about "authentication" in namespace "MyProject"
```

### Performance Issues

For large memory sets (>10,000), consider:
- Using namespaces to partition memories
- Running consolidation to merge duplicates
- Applying decay to reduce low-importance memories

## Next Steps

- Read [API.md](API.md) for complete tool reference
- See [CONFIGURATION.md](CONFIGURATION.md) for all settings
- Check [BENCHMARKS.md](BENCHMARKS.md) for performance data
- Review [TECHNICAL_HIGHLIGHTS.md](TECHNICAL_HIGHLIGHTS.md) for architecture details

## Quick Reference

| Tool | Purpose |
|------|---------|
| `remember` | Store a new memory |
| `recall` | Semantic search for memories |
| `hybrid_recall` | Combined keyword + semantic search |
| `nearby` | Find memories similar to a specific one |
| `forget` | Delete a memory |
| `journey` | Navigate between two memories |
| `wander` | Random walk through memory space |
| `regions` | Discover topic clusters |
| `decay` | Age-out old memories |
| `reinforce` | Boost memory importance |
| `consolidate` | Merge similar memories |
| `export_memories` | Backup to file |
| `import_memories` | Restore from file |
