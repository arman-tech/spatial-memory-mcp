"""MCP Server for Spatial Memory.

This module provides the MCP (Model Context Protocol) server implementation
that exposes memory operations as tools for LLM assistants.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import signal
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.config import ConfigurationError, get_settings, validate_startup
from spatial_memory.core.database import (
    Database,
    clear_connection_cache,
    set_connection_pool_max_size,
)
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.errors import (
    MemoryNotFoundError,
    SpatialMemoryError,
    ValidationError,
)
from spatial_memory.core.health import HealthChecker
from spatial_memory.core.logging import configure_logging
from spatial_memory.core.metrics import is_available as metrics_available
from spatial_memory.core.metrics import record_request
from spatial_memory.services.memory import MemoryService

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )

logger = logging.getLogger(__name__)


# Tool definitions for MCP
TOOLS = [
    Tool(
        name="remember",
        description="Store a new memory in the spatial memory system.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content to remember",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace for organizing memories (default: 'default')",
                    "default": "default",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score from 0.0 to 1.0 (default: 0.5)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to the memory",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="remember_batch",
        description="Store multiple memories efficiently in a single operation.",
        inputSchema={
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "namespace": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "importance": {"type": "number"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["content"],
                    },
                    "description": "Array of memories to store",
                },
            },
            "required": ["memories"],
        },
    ),
    Tool(
        name="recall",
        description="Search for similar memories using semantic similarity.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                },
                "namespace": {
                    "type": "string",
                    "description": "Filter to specific namespace",
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum similarity threshold (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nearby",
        description="Find memories similar to a specific memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the reference memory",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of neighbors (default: 5)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                },
                "namespace": {
                    "type": "string",
                    "description": "Filter neighbors to specific namespace",
                },
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="forget",
        description="Delete a memory by its ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to delete",
                },
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="forget_batch",
        description="Delete multiple memories by their IDs.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of memory IDs to delete",
                },
            },
            "required": ["memory_ids"],
        },
    ),
    Tool(
        name="health",
        description="Check system health status.",
        inputSchema={
            "type": "object",
            "properties": {
                "verbose": {
                    "type": "boolean",
                    "description": "Include detailed check results",
                    "default": False,
                },
            },
        },
    ),
]


class SpatialMemoryServer:
    """MCP Server for Spatial Memory operations.

    Uses dependency injection for testability.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol | None = None,
        embeddings: EmbeddingServiceProtocol | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            repository: Optional repository (uses LanceDB if not provided).
            embeddings: Optional embedding service (uses local model if not provided).
        """
        self._settings = get_settings()
        self._db: Database | None = None

        # Configure connection pool size from settings
        set_connection_pool_max_size(self._settings.connection_pool_max_size)

        # Set up dependencies
        if repository is None or embeddings is None:
            # Create embedding service FIRST to auto-detect dimensions
            if embeddings is None:
                embeddings = EmbeddingService(
                    model_name=self._settings.embedding_model,
                    openai_api_key=self._settings.openai_api_key,
                )

            # Auto-detect embedding dimensions from the model
            embedding_dim = embeddings.dimensions
            logger.info(f"Auto-detected embedding dimensions: {embedding_dim}")

            # Create database with all config values wired
            self._db = Database(
                storage_path=self._settings.memory_path,
                embedding_dim=embedding_dim,
                auto_create_indexes=self._settings.auto_create_indexes,
                vector_index_threshold=self._settings.vector_index_threshold,
                enable_fts=self._settings.enable_fts_index,
                index_nprobes=self._settings.index_nprobes,
                index_refine_factor=self._settings.index_refine_factor,
                max_retry_attempts=self._settings.max_retry_attempts,
                retry_backoff_seconds=self._settings.retry_backoff_seconds,
                read_consistency_interval_ms=self._settings.read_consistency_interval_ms,
                index_wait_timeout_seconds=self._settings.index_wait_timeout_seconds,
                fts_stem=self._settings.fts_stem,
                fts_remove_stop_words=self._settings.fts_remove_stop_words,
                fts_language=self._settings.fts_language,
                index_type=self._settings.index_type,
                hnsw_m=self._settings.hnsw_m,
                hnsw_ef_construction=self._settings.hnsw_ef_construction,
                enable_memory_expiration=self._settings.enable_memory_expiration,
                default_memory_ttl_days=self._settings.default_memory_ttl_days,
            )
            self._db.connect()

            if repository is None:
                repository = LanceDBMemoryRepository(self._db)

        self._memory_service = MemoryService(
            repository=repository,
            embeddings=embeddings,
        )

        # Store embeddings and database for health checks
        self._embeddings = embeddings

        # Log metrics availability
        if metrics_available():
            logger.info("Prometheus metrics enabled")
        else:
            logger.info("Prometheus metrics disabled (prometheus_client not installed)")

        # Create MCP server
        self._server = Server("spatial-memory")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return the list of available tools."""
            return TOOLS

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            try:
                result = self._handle_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, default=str))]
            except MemoryNotFoundError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "MemoryNotFound", "message": str(e)}),
                )]
            except ValidationError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "ValidationError", "message": str(e)}),
                )]
            except SpatialMemoryError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "SpatialMemoryError", "message": str(e)}),
                )]
            except Exception as e:
                logger.exception(f"Unexpected error in tool {name}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "InternalError", "message": str(e)}),
                )]

    def _handle_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Route tool call to appropriate handler.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as dictionary.

        Raises:
            ValidationError: If tool name is unknown.
        """
        # Record metrics for this tool call
        with record_request(name, "success"):
            return self._handle_tool_impl(name, arguments)

    def _handle_tool_impl(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Implementation of tool routing.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as dictionary.

        Raises:
            ValidationError: If tool name is unknown.
        """
        if name == "remember":
            remember_result = self._memory_service.remember(
                content=arguments["content"],
                namespace=arguments.get("namespace", "default"),
                tags=arguments.get("tags"),
                importance=arguments.get("importance", 0.5),
                metadata=arguments.get("metadata"),
            )
            return asdict(remember_result)

        elif name == "remember_batch":
            batch_result = self._memory_service.remember_batch(
                memories=arguments["memories"],
            )
            return asdict(batch_result)

        elif name == "recall":
            recall_result = self._memory_service.recall(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                namespace=arguments.get("namespace"),
                min_similarity=arguments.get("min_similarity", 0.0),
            )
            # Convert MemoryResult objects to dicts
            return {
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "similarity": m.similarity,
                        "namespace": m.namespace,
                        "tags": m.tags,
                        "importance": m.importance,
                        "created_at": m.created_at.isoformat(),
                        "metadata": m.metadata,
                    }
                    for m in recall_result.memories
                ],
                "total": recall_result.total,
            }

        elif name == "nearby":
            nearby_result = self._memory_service.nearby(
                memory_id=arguments["memory_id"],
                limit=arguments.get("limit", 5),
                namespace=arguments.get("namespace"),
            )
            return {
                "reference": {
                    "id": nearby_result.reference.id,
                    "content": nearby_result.reference.content,
                    "namespace": nearby_result.reference.namespace,
                },
                "neighbors": [
                    {
                        "id": n.id,
                        "content": n.content,
                        "similarity": n.similarity,
                        "namespace": n.namespace,
                    }
                    for n in nearby_result.neighbors
                ],
            }

        elif name == "forget":
            forget_result = self._memory_service.forget(
                memory_id=arguments["memory_id"],
            )
            return asdict(forget_result)

        elif name == "forget_batch":
            forget_batch_result = self._memory_service.forget_batch(
                memory_ids=arguments["memory_ids"],
            )
            return asdict(forget_batch_result)

        elif name == "health":
            verbose = arguments.get("verbose", False)

            # Create health checker
            health_checker = HealthChecker(
                database=self._db,
                embeddings=self._embeddings,
                storage_path=self._settings.memory_path,
            )

            # Get health report
            report = health_checker.get_health_report()

            # Build response
            result: dict[str, Any] = {
                "status": report.status.value,
                "timestamp": report.timestamp.isoformat(),
                "ready": health_checker.is_ready(),
                "alive": health_checker.is_alive(),
            }

            # Add detailed checks if verbose
            if verbose:
                result["checks"] = [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "latency_ms": check.latency_ms,
                    }
                    for check in report.checks
                ]

            return result

        else:
            raise ValidationError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    def close(self) -> None:
        """Clean up resources."""
        if self._db is not None:
            self._db.close()


def create_server(
    repository: MemoryRepositoryProtocol | None = None,
    embeddings: EmbeddingServiceProtocol | None = None,
) -> SpatialMemoryServer:
    """Create a new SpatialMemoryServer instance.

    This factory function allows dependency injection for testing.

    Args:
        repository: Optional repository implementation.
        embeddings: Optional embedding service implementation.

    Returns:
        Configured SpatialMemoryServer instance.
    """
    return SpatialMemoryServer(repository=repository, embeddings=embeddings)


async def main() -> None:
    """Main entry point for the MCP server."""
    # Get settings
    settings = get_settings()

    # Validate configuration
    try:
        warnings = validate_startup(settings)
        # Use basic logging temporarily for startup validation
        logging.basicConfig(level=settings.log_level)
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    except ConfigurationError as e:
        # Use basic logging for error
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Configure logging properly
    configure_logging(
        level=settings.log_level,
        json_format=settings.log_format == "json",
    )

    server = create_server()
    cleanup_done = False

    def cleanup() -> None:
        """Cleanup function for server resources."""
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        logger.info("Cleaning up server resources...")
        server.close()
        clear_connection_cache()
        logger.info("Server shutdown complete")

    def handle_shutdown(signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

    # Register signal handlers for logging (both platforms use same code)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Register atexit as a safety net for cleanup
    atexit.register(cleanup)

    try:
        await server.run()
    except asyncio.CancelledError:
        logger.info("Server task cancelled")
    finally:
        cleanup()
        atexit.unregister(cleanup)  # Prevent double cleanup
