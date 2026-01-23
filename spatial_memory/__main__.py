"""Entry point for running the Spatial Memory MCP Server."""

import logging


def main() -> None:
    """Run the Spatial Memory MCP Server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Spatial Memory MCP Server - Not yet implemented")
    logger.info("Run 'python -m spatial_memory' to start the server")


if __name__ == "__main__":
    main()
