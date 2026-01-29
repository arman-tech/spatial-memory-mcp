"""Entry point for running the Spatial Memory MCP Server."""

import sys


def main() -> None:
    """Run the Spatial Memory MCP Server."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      Spatial Memory MCP Server v0.1.0                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Status: Phase 1 (Foundation) Complete                                        ║
║                                                                               ║
║  The MCP server is not yet implemented. Phase 2 (Core Operations) is in       ║
║  development and will add the following tools:                                ║
║                                                                               ║
║    • remember / recall / nearby / forget                                      ║
║    • journey / wander / regions / visualize                                   ║
║    • consolidate / extract / decay / reinforce                                ║
║    • stats / namespaces / export / import                                     ║
║                                                                               ║
║  Current Phase 1 capabilities (available for development):                    ║
║    • Configuration system (spatial_memory.config)                             ║
║    • LanceDB database wrapper (spatial_memory.core.database)                  ║
║    • Embedding service (spatial_memory.core.embeddings)                       ║
║    • Data models (spatial_memory.core.models)                                 ║
║    • Error handling (spatial_memory.core.errors)                              ║
║                                                                               ║
║  For more information:                                                        ║
║    • README: https://github.com/arman-tech/spatial-memory-mcp                 ║
║    • Run tests: pytest tests/ -v                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    sys.exit(0)


if __name__ == "__main__":
    main()
