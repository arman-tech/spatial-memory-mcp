"""Entry point for running the Spatial Memory MCP Server."""

import asyncio


def main() -> None:
    """Run the Spatial Memory MCP Server."""
    from spatial_memory.server import main as server_main

    asyncio.run(server_main())


if __name__ == "__main__":
    main()
