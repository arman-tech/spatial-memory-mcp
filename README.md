# Spatial Memory MCP Server

A vector-based spatial memory system that treats knowledge as a navigable landscape, not a filing cabinet.

## Features

- **Spatial Navigation**: `journey` and `wander` let LLMs discover connections they didn't know to ask for
- **Visual Understanding**: Real-time visualization (Mermaid/SVG/JSON) shows how knowledge is organized
- **Cognitive Dynamics**: Memories consolidate, decay, and reinforce like human cognition
- **Semantic Discovery**: `regions` auto-clusters related concepts without manual tagging

## Installation

```bash
pip install spatial-memory-mcp
```

Or with uvx:

```bash
uvx spatial-memory-mcp
```

## Quick Start

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

## Tools

### Core Operations
- `remember` - Store a memory in vector space
- `recall` - Find memories semantically similar to a query
- `nearby` - Find memories spatially close to a specific memory
- `forget` - Remove a memory

### Spatial Operations
- `journey` - Interpolate a path between two memories
- `wander` - Random walk through memory space
- `regions` - Discover conceptual regions via clustering
- `visualize` - Generate visual representation

### Lifecycle Operations
- `consolidate` - Merge similar memories
- `extract` - Auto-extract memories from text
- `decay` - Reduce importance of stale memories
- `reinforce` - Boost importance of useful memories

## License

MIT
