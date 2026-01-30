"""Port interfaces for Spatial Memory MCP Server."""

from spatial_memory.ports.repositories import (
    EmbeddingServiceProtocol,
    MemoryRepositoryProtocol,
)

__all__ = [
    "EmbeddingServiceProtocol",
    "MemoryRepositoryProtocol",
]
