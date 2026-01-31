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
    ConsolidationError,
    DecayError,
    ExportError,
    ExtractionError,
    FileSizeLimitError,
    ImportRecordLimitError,
    MemoryImportError,
    MemoryNotFoundError,
    NamespaceNotFoundError,
    NamespaceOperationError,
    PathSecurityError,
    ReinforcementError,
    SpatialMemoryError,
    ValidationError,
)
from spatial_memory.core.health import HealthChecker
from spatial_memory.core.logging import configure_logging
from spatial_memory.core.metrics import is_available as metrics_available
from spatial_memory.core.metrics import record_request
from spatial_memory.services.export_import import ExportImportConfig, ExportImportService
from spatial_memory.services.lifecycle import LifecycleConfig, LifecycleService
from spatial_memory.services.memory import MemoryService
from spatial_memory.services.spatial import SpatialConfig, SpatialService
from spatial_memory.services.utility import UtilityConfig, UtilityService

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
    Tool(
        name="journey",
        description=(
            "Navigate semantic space between two memories using spherical "
            "interpolation (SLERP). Discovers memories along the conceptual path."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "start_id": {
                    "type": "string",
                    "description": "Starting memory UUID",
                },
                "end_id": {
                    "type": "string",
                    "description": "Ending memory UUID",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 20,
                    "default": 10,
                    "description": "Number of interpolation steps",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace filter for nearby search",
                },
            },
            "required": ["start_id", "end_id"],
        },
    ),
    Tool(
        name="wander",
        description=(
            "Explore memory space through random walk. Uses temperature-based "
            "selection to balance exploration and exploitation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "start_id": {
                    "type": "string",
                    "description": "Starting memory UUID (random if not provided)",
                },
                "steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                    "description": "Number of exploration steps",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Randomness (0.0=focused, 1.0=very random)",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace filter",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="regions",
        description=(
            "Discover semantic clusters in memory space using HDBSCAN. "
            "Returns cluster info with representative memories and keywords."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace filter",
                },
                "min_cluster_size": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 50,
                    "default": 3,
                    "description": "Minimum memories per cluster",
                },
                "max_clusters": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum clusters to return",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="visualize",
        description=(
            "Project memories to 2D/3D for visualization using UMAP. "
            "Returns coordinates and optional similarity edges."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific memory UUIDs to visualize",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace filter (if memory_ids not specified)",
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "mermaid", "svg"],
                    "default": "json",
                    "description": "Output format",
                },
                "dimensions": {
                    "type": "integer",
                    "enum": [2, 3],
                    "default": 2,
                    "description": "Projection dimensionality",
                },
                "include_edges": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include similarity edges",
                },
            },
            "required": [],
        },
    ),
    # Lifecycle tools
    Tool(
        name="decay",
        description=(
            "Apply time and access-based decay to memory importance scores. "
            "Memories not accessed recently will have reduced importance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to decay (all if not specified)",
                },
                "decay_function": {
                    "type": "string",
                    "enum": ["exponential", "linear", "step"],
                    "default": "exponential",
                    "description": "Decay curve shape",
                },
                "half_life_days": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                    "description": "Days until importance halves (exponential)",
                },
                "min_importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.5,
                    "default": 0.1,
                    "description": "Minimum importance floor",
                },
                "access_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "Weight of access count in decay calculation",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview changes without applying",
                },
            },
        },
    ),
    Tool(
        name="reinforce",
        description=(
            "Boost memory importance based on usage or explicit feedback. "
            "Reinforcement increases importance and can reset decay timer."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs to reinforce",
                },
                "boost_type": {
                    "type": "string",
                    "enum": ["additive", "multiplicative", "set_value"],
                    "default": "additive",
                    "description": "Type of boost to apply",
                },
                "boost_amount": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                    "description": "Amount to boost importance",
                },
                "update_access": {
                    "type": "boolean",
                    "default": True,
                    "description": "Update last_accessed timestamp",
                },
            },
            "required": ["memory_ids"],
        },
    ),
    Tool(
        name="extract",
        description=(
            "Automatically extract memories from conversation text. "
            "Uses pattern matching to identify facts, decisions, and key information."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to extract memories from",
                },
                "namespace": {
                    "type": "string",
                    "default": "extracted",
                    "description": "Namespace for extracted memories",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Minimum confidence to extract",
                },
                "deduplicate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip if similar memory exists",
                },
                "dedup_threshold": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 0.99,
                    "default": 0.9,
                    "description": "Similarity threshold for deduplication",
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="consolidate",
        description=(
            "Merge similar or duplicate memories to reduce redundancy. "
            "Finds memories above similarity threshold and merges them."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to consolidate (required)",
                },
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 0.99,
                    "default": 0.85,
                    "description": "Minimum similarity for duplicates",
                },
                "strategy": {
                    "type": "string",
                    "enum": [
                        "keep_newest",
                        "keep_oldest",
                        "keep_highest_importance",
                        "merge_content",
                    ],
                    "default": "keep_highest_importance",
                    "description": "Strategy for merging duplicates",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without changes",
                },
                "max_groups": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum groups to process",
                },
            },
            "required": ["namespace"],
        },
    ),
    # Phase 5: Utility Tools
    Tool(
        name="stats",
        description="Get database statistics and health metrics.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Filter stats to specific namespace",
                },
                "include_index_details": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include detailed index statistics",
                },
            },
        },
    ),
    Tool(
        name="namespaces",
        description="List all namespaces with memory counts and date ranges.",
        inputSchema={
            "type": "object",
            "properties": {
                "include_stats": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include memory counts and date ranges per namespace",
                },
            },
        },
    ),
    Tool(
        name="delete_namespace",
        description="Delete all memories in a namespace. DESTRUCTIVE - use dry_run first.",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to delete",
                },
                "confirm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set to true to confirm deletion (required when dry_run=false)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview deletion without executing",
                },
            },
            "required": ["namespace"],
        },
    ),
    Tool(
        name="rename_namespace",
        description="Rename a namespace, moving all its memories to the new name.",
        inputSchema={
            "type": "object",
            "properties": {
                "old_namespace": {
                    "type": "string",
                    "description": "Current namespace name",
                },
                "new_namespace": {
                    "type": "string",
                    "description": "New namespace name",
                },
            },
            "required": ["old_namespace", "new_namespace"],
        },
    ),
    Tool(
        name="export_memories",
        description="Export memories to file (Parquet, JSON, or CSV format).",
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Path for output file (extension determines format)",
                },
                "format": {
                    "type": "string",
                    "enum": ["parquet", "json", "csv"],
                    "description": "Export format (auto-detected from extension if not specified)",
                },
                "namespace": {
                    "type": "string",
                    "description": "Export only this namespace (all if not specified)",
                },
                "include_vectors": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include embedding vectors in export",
                },
            },
            "required": ["output_path"],
        },
    ),
    Tool(
        name="import_memories",
        description="Import memories from file with validation. Use dry_run=true first.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to source file",
                },
                "format": {
                    "type": "string",
                    "enum": ["parquet", "json", "csv"],
                    "description": "Import format (auto-detected from extension if not specified)",
                },
                "namespace_override": {
                    "type": "string",
                    "description": "Override namespace for all imported memories",
                },
                "deduplicate": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip records similar to existing memories",
                },
                "dedup_threshold": {
                    "type": "number",
                    "minimum": 0.7,
                    "maximum": 0.99,
                    "default": 0.95,
                    "description": "Similarity threshold for deduplication",
                },
                "validate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate records before import",
                },
                "regenerate_embeddings": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate new embeddings (required if vectors missing)",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate without importing",
                },
            },
            "required": ["source_path"],
        },
    ),
    Tool(
        name="hybrid_recall",
        description="Search memories using combined vector and keyword (full-text) search.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "alpha": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Balance: 1.0=pure vector, 0.0=pure keyword, 0.5=balanced",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                    "description": "Maximum number of results",
                },
                "namespace": {
                    "type": "string",
                    "description": "Filter to specific namespace",
                },
                "min_similarity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum similarity threshold",
                },
            },
            "required": ["query"],
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
                    backend=self._settings.embedding_backend,  # type: ignore[arg-type]
                )

            # Auto-detect embedding dimensions from the model
            embedding_dim = embeddings.dimensions
            logger.info(f"Auto-detected embedding dimensions: {embedding_dim}")
            logger.info(f"Embedding backend: {embeddings.backend}")

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

        # Create spatial service for exploration operations
        self._spatial_service = SpatialService(
            repository=repository,
            embeddings=embeddings,
            config=SpatialConfig(
                journey_max_steps=self._settings.max_journey_steps,
                wander_max_steps=self._settings.max_wander_steps,
                regions_max_memories=self._settings.regions_max_memories,
                visualize_max_memories=self._settings.max_visualize_memories,
                visualize_n_neighbors=self._settings.umap_n_neighbors,
                visualize_min_dist=self._settings.umap_min_dist,
                visualize_similarity_threshold=self._settings.visualize_similarity_threshold,
            ),
        )

        # Create lifecycle service for memory lifecycle management
        self._lifecycle_service = LifecycleService(
            repository=repository,
            embeddings=embeddings,
            config=LifecycleConfig(
                decay_default_half_life_days=self._settings.decay_default_half_life_days,
                decay_default_function=self._settings.decay_default_function,
                decay_min_importance_floor=self._settings.decay_min_importance_floor,
                decay_batch_size=self._settings.decay_batch_size,
                reinforce_default_boost=self._settings.reinforce_default_boost,
                reinforce_max_importance=self._settings.reinforce_max_importance,
                extract_max_text_length=self._settings.extract_max_text_length,
                extract_max_candidates=self._settings.extract_max_candidates,
                extract_default_importance=self._settings.extract_default_importance,
                extract_default_namespace=self._settings.extract_default_namespace,
                consolidate_min_threshold=self._settings.consolidate_min_threshold,
                consolidate_content_weight=self._settings.consolidate_content_weight,
                consolidate_max_batch=self._settings.consolidate_max_batch,
            ),
        )

        # Create utility service for stats, namespaces, and hybrid search
        self._utility_service = UtilityService(
            repository=repository,
            embeddings=embeddings,
            config=UtilityConfig(
                hybrid_default_alpha=self._settings.hybrid_default_alpha,
                hybrid_min_alpha=self._settings.hybrid_min_alpha,
                hybrid_max_alpha=self._settings.hybrid_max_alpha,
                stats_include_index_details=True,
                namespace_batch_size=self._settings.namespace_batch_size,
                delete_namespace_require_confirmation=self._settings.destructive_require_namespace_confirmation,
            ),
        )

        # Create export/import service for data portability
        self._export_import_service = ExportImportService(
            repository=repository,
            embeddings=embeddings,
            config=ExportImportConfig(
                default_export_format=self._settings.export_default_format,
                export_batch_size=self._settings.export_batch_size,
                import_batch_size=self._settings.import_batch_size,
                import_deduplicate=self._settings.import_deduplicate_default,
                import_dedup_threshold=self._settings.import_dedup_threshold,
                validate_on_import=self._settings.import_validate_vectors,
                parquet_compression="zstd",
                max_import_records=self._settings.import_max_records,
                csv_include_vectors=self._settings.csv_include_vectors,
                max_export_records=self._settings.max_export_records,
            ),
            allowed_export_paths=self._settings.export_allowed_paths,
            allowed_import_paths=self._settings.import_allowed_paths,
            allow_symlinks=self._settings.export_allow_symlinks,
            max_import_size_bytes=int(self._settings.import_max_file_size_mb * 1024 * 1024),
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
                    text=json.dumps({
                        "error": "MemoryNotFound", "message": str(e), "isError": True
                    }),
                )]
            except ValidationError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ValidationError", "message": str(e), "isError": True
                    }),
                )]
            except DecayError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "DecayError", "message": str(e), "isError": True
                    }),
                )]
            except ReinforcementError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ReinforcementError", "message": str(e), "isError": True
                    }),
                )]
            except ExtractionError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ExtractionError", "message": str(e), "isError": True
                    }),
                )]
            except ConsolidationError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ConsolidationError", "message": str(e), "isError": True
                    }),
                )]
            # Phase 5 error handlers
            except ExportError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ExportError", "message": str(e), "isError": True
                    }),
                )]
            except MemoryImportError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "MemoryImportError", "message": str(e), "isError": True
                    }),
                )]
            except PathSecurityError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "PathSecurityError", "message": str(e), "isError": True
                    }),
                )]
            except FileSizeLimitError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "FileSizeLimitError", "message": str(e), "isError": True
                    }),
                )]
            except ImportRecordLimitError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "ImportRecordLimitError", "message": str(e), "isError": True
                    }),
                )]
            except NamespaceNotFoundError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "NamespaceNotFoundError", "message": str(e), "isError": True
                    }),
                )]
            except NamespaceOperationError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "NamespaceOperationError", "message": str(e), "isError": True
                    }),
                )]
            except SpatialMemoryError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "SpatialMemoryError", "message": str(e), "isError": True
                    }),
                )]
            except Exception as e:
                import uuid as uuid_module
                error_id = str(uuid_module.uuid4())[:8]
                logger.exception(f"Error {error_id} in tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "InternalError",
                        "message": f"An internal error occurred. Reference: {error_id}",
                        "isError": True
                    }),
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

        elif name == "journey":
            journey_result = self._spatial_service.journey(
                start_id=arguments["start_id"],
                end_id=arguments["end_id"],
                steps=arguments.get("steps", 10),
                namespace=arguments.get("namespace"),
            )
            return {
                "start_id": journey_result.start_id,
                "end_id": journey_result.end_id,
                "steps": [
                    {
                        "step": s.step,
                        "t": s.t,
                        "position": s.position,
                        "nearby_memories": [
                            {
                                "id": m.id,
                                "content": m.content,
                                "similarity": m.similarity,
                            }
                            for m in s.nearby_memories
                        ],
                        "distance_to_path": s.distance_to_path,
                    }
                    for s in journey_result.steps
                ],
                "path_coverage": journey_result.path_coverage,
            }

        elif name == "wander":
            # Get start_id - use random memory if not provided
            start_id = arguments.get("start_id")
            if start_id is None:
                # Get a random memory to start from
                all_memories = self._memory_service.recall(
                    query="",
                    limit=1,
                    namespace=arguments.get("namespace"),
                )
                if not all_memories.memories:
                    raise ValidationError("No memories available for wander")
                start_id = all_memories.memories[0].id

            wander_result = self._spatial_service.wander(
                start_id=start_id,
                steps=arguments.get("steps", 10),
                temperature=arguments.get("temperature", 0.5),
                namespace=arguments.get("namespace"),
            )
            return {
                "start_id": wander_result.start_id,
                "steps": [
                    {
                        "step": s.step,
                        "memory": {
                            "id": s.memory.id,
                            "content": s.memory.content,
                            "namespace": s.memory.namespace,
                            "tags": s.memory.tags,
                            "similarity": s.memory.similarity,
                        },
                        "similarity_to_previous": s.similarity_to_previous,
                        "selection_probability": s.selection_probability,
                    }
                    for s in wander_result.steps
                ],
                "total_distance": wander_result.total_distance,
            }

        elif name == "regions":
            regions_result = self._spatial_service.regions(
                namespace=arguments.get("namespace"),
                min_cluster_size=arguments.get("min_cluster_size", 3),
                max_clusters=arguments.get("max_clusters"),
            )
            return {
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "keywords": c.keywords,
                        "representative_memory": {
                            "id": c.representative_memory.id,
                            "content": c.representative_memory.content,
                        },
                        "sample_memories": [
                            {
                                "id": m.id,
                                "content": m.content,
                                "similarity": m.similarity,
                            }
                            for m in c.sample_memories
                        ],
                        "coherence": c.coherence,
                    }
                    for c in regions_result.clusters
                ],
                "total_memories": regions_result.total_memories,
                "noise_count": regions_result.noise_count,
                "clustering_quality": regions_result.clustering_quality,
            }

        elif name == "visualize":
            visualize_result = self._spatial_service.visualize(
                memory_ids=arguments.get("memory_ids"),
                namespace=arguments.get("namespace"),
                format=arguments.get("format", "json"),
                dimensions=arguments.get("dimensions", 2),
                include_edges=arguments.get("include_edges", True),
            )
            # If format is not JSON, return the formatted output directly
            output_format = arguments.get("format", "json")
            if output_format in ("mermaid", "svg"):
                return {
                    "format": output_format,
                    "output": visualize_result.output,
                    "node_count": len(visualize_result.nodes),
                }
            # JSON format - return full structured data
            return {
                "nodes": [
                    {
                        "id": n.id,
                        "x": n.x,
                        "y": n.y,
                        "label": n.label,
                        "cluster": n.cluster,
                        "importance": n.importance,
                    }
                    for n in visualize_result.nodes
                ],
                "edges": [
                    {
                        "from_id": e.from_id,
                        "to_id": e.to_id,
                        "weight": e.weight,
                    }
                    for e in visualize_result.edges
                ] if visualize_result.edges else [],
                "bounds": visualize_result.bounds,
                "format": visualize_result.format,
            }

        elif name == "decay":
            decay_result = self._lifecycle_service.decay(
                namespace=arguments.get("namespace"),
                decay_function=arguments.get("decay_function", "exponential"),
                half_life_days=arguments.get("half_life_days", 30.0),
                min_importance=arguments.get("min_importance", 0.1),
                access_weight=arguments.get("access_weight", 0.3),
                dry_run=arguments.get("dry_run", True),
            )
            return {
                "memories_analyzed": decay_result.memories_analyzed,
                "memories_decayed": decay_result.memories_decayed,
                "avg_decay_factor": decay_result.avg_decay_factor,
                "decayed_memories": [
                    {
                        "id": m.id,
                        "content_preview": m.content_preview,
                        "old_importance": m.old_importance,
                        "new_importance": m.new_importance,
                        "decay_factor": m.decay_factor,
                        "days_since_access": m.days_since_access,
                        "access_count": m.access_count,
                    }
                    for m in decay_result.decayed_memories
                ],
                "dry_run": decay_result.dry_run,
            }

        elif name == "reinforce":
            reinforce_result = self._lifecycle_service.reinforce(
                memory_ids=arguments["memory_ids"],
                boost_type=arguments.get("boost_type", "additive"),
                boost_amount=arguments.get("boost_amount", 0.1),
                update_access=arguments.get("update_access", True),
            )
            return {
                "memories_reinforced": reinforce_result.memories_reinforced,
                "avg_boost": reinforce_result.avg_boost,
                "reinforced": [
                    {
                        "id": m.id,
                        "content_preview": m.content_preview,
                        "old_importance": m.old_importance,
                        "new_importance": m.new_importance,
                        "boost_applied": m.boost_applied,
                    }
                    for m in reinforce_result.reinforced
                ],
                "not_found": reinforce_result.not_found,
            }

        elif name == "extract":
            extract_result = self._lifecycle_service.extract(
                text=arguments["text"],
                namespace=arguments.get("namespace", "extracted"),
                min_confidence=arguments.get("min_confidence", 0.5),
                deduplicate=arguments.get("deduplicate", True),
                dedup_threshold=arguments.get("dedup_threshold", 0.9),
            )
            return {
                "candidates_found": extract_result.candidates_found,
                "memories_created": extract_result.memories_created,
                "deduplicated_count": extract_result.deduplicated_count,
                "extractions": [
                    {
                        "content": e.content,
                        "confidence": e.confidence,
                        "pattern_matched": e.pattern_matched,
                        "start_pos": e.start_pos,
                        "end_pos": e.end_pos,
                        "stored": e.stored,
                        "memory_id": e.memory_id,
                    }
                    for e in extract_result.extractions
                ],
            }

        elif name == "consolidate":
            consolidate_result = self._lifecycle_service.consolidate(
                namespace=arguments["namespace"],
                similarity_threshold=arguments.get("similarity_threshold", 0.85),
                strategy=arguments.get("strategy", "keep_highest_importance"),
                dry_run=arguments.get("dry_run", True),
                max_groups=arguments.get("max_groups", 50),
            )
            return {
                "groups_found": consolidate_result.groups_found,
                "memories_merged": consolidate_result.memories_merged,
                "memories_deleted": consolidate_result.memories_deleted,
                "groups": [
                    {
                        "representative_id": g.representative_id,
                        "member_ids": g.member_ids,
                        "avg_similarity": g.avg_similarity,
                        "action_taken": g.action_taken,
                    }
                    for g in consolidate_result.groups
                ],
                "dry_run": consolidate_result.dry_run,
            }

        # Phase 5: Utility Tools
        elif name == "stats":
            stats_result = self._utility_service.stats(
                namespace=arguments.get("namespace"),
                include_index_details=arguments.get("include_index_details", True),
            )
            return {
                "total_memories": stats_result.total_memories,
                "memories_by_namespace": stats_result.memories_by_namespace,
                "storage_bytes": stats_result.storage_bytes,
                "storage_mb": stats_result.storage_mb,
                "estimated_vector_bytes": stats_result.estimated_vector_bytes,
                "has_vector_index": stats_result.has_vector_index,
                "has_fts_index": stats_result.has_fts_index,
                "indices": [
                    {
                        "name": idx.name,
                        "index_type": idx.index_type,
                        "column": idx.column,
                        "num_indexed_rows": idx.num_indexed_rows,
                        "status": idx.status,
                    }
                    for idx in stats_result.indices
                ] if stats_result.indices else [],
                "num_fragments": stats_result.num_fragments,
                "needs_compaction": stats_result.needs_compaction,
                "table_version": stats_result.table_version,
                "oldest_memory_date": (
                    stats_result.oldest_memory_date.isoformat()
                    if stats_result.oldest_memory_date else None
                ),
                "newest_memory_date": (
                    stats_result.newest_memory_date.isoformat()
                    if stats_result.newest_memory_date else None
                ),
                "avg_content_length": stats_result.avg_content_length,
            }

        elif name == "namespaces":
            namespaces_result = self._utility_service.namespaces(
                include_stats=arguments.get("include_stats", True),
            )
            return {
                "namespaces": [
                    {
                        "name": ns.name,
                        "memory_count": ns.memory_count,
                        "oldest_memory": (
                            ns.oldest_memory.isoformat() if ns.oldest_memory else None
                        ),
                        "newest_memory": (
                            ns.newest_memory.isoformat() if ns.newest_memory else None
                        ),
                    }
                    for ns in namespaces_result.namespaces
                ],
                "total_namespaces": namespaces_result.total_namespaces,
                "total_memories": namespaces_result.total_memories,
            }

        elif name == "delete_namespace":
            delete_result = self._utility_service.delete_namespace(
                namespace=arguments["namespace"],
                confirm=arguments.get("confirm", False),
                dry_run=arguments.get("dry_run", True),
            )
            return {
                "namespace": delete_result.namespace,
                "memories_deleted": delete_result.memories_deleted,
                "success": delete_result.success,
                "message": delete_result.message,
                "dry_run": delete_result.dry_run,
            }

        elif name == "rename_namespace":
            rename_result = self._utility_service.rename_namespace(
                old_namespace=arguments["old_namespace"],
                new_namespace=arguments["new_namespace"],
            )
            return {
                "old_namespace": rename_result.old_namespace,
                "new_namespace": rename_result.new_namespace,
                "memories_renamed": rename_result.memories_renamed,
                "success": rename_result.success,
                "message": rename_result.message,
            }

        elif name == "export_memories":
            export_result = self._export_import_service.export_memories(
                output_path=arguments["output_path"],
                format=arguments.get("format"),
                namespace=arguments.get("namespace"),
                include_vectors=arguments.get("include_vectors", True),
            )
            return {
                "format": export_result.format,
                "output_path": export_result.output_path,
                "memories_exported": export_result.memories_exported,
                "file_size_bytes": export_result.file_size_bytes,
                "file_size_mb": export_result.file_size_mb,
                "namespaces_included": export_result.namespaces_included,
                "duration_seconds": export_result.duration_seconds,
                "compression": export_result.compression,
            }

        elif name == "import_memories":
            dry_run = arguments.get("dry_run", True)
            import_result = self._export_import_service.import_memories(
                source_path=arguments["source_path"],
                format=arguments.get("format"),
                namespace_override=arguments.get("namespace_override"),
                deduplicate=arguments.get("deduplicate", False),
                dedup_threshold=arguments.get("dedup_threshold", 0.95),
                validate=arguments.get("validate", True),
                regenerate_embeddings=arguments.get("regenerate_embeddings", False),
                dry_run=dry_run,
            )
            return {
                "source_path": import_result.source_path,
                "format": import_result.format,
                "total_records_in_file": import_result.total_records_in_file,
                "memories_imported": import_result.memories_imported,
                "memories_skipped": import_result.memories_skipped,
                "memories_failed": import_result.memories_failed,
                "validation_errors": [
                    {
                        "row_number": err.row_number,
                        "field": err.field,
                        "error": err.error,
                        "value": str(err.value) if err.value is not None else None,
                    }
                    for err in import_result.validation_errors
                ] if import_result.validation_errors else [],
                "namespace_override": import_result.namespace_override,
                "duration_seconds": import_result.duration_seconds,
                "dry_run": dry_run,
                "imported_memories": [
                    {
                        "id": m.id,
                        "content_preview": m.content_preview,
                        "namespace": m.namespace,
                    }
                    for m in import_result.imported_memories[:10]
                ] if import_result.imported_memories else [],
            }

        elif name == "hybrid_recall":
            hybrid_result = self._utility_service.hybrid_recall(
                query=arguments["query"],
                alpha=arguments.get("alpha", 0.5),
                limit=arguments.get("limit", 5),
                namespace=arguments.get("namespace"),
                min_similarity=arguments.get("min_similarity", 0.0),
            )
            return {
                "query": hybrid_result.query,
                "alpha": hybrid_result.alpha,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "similarity": m.similarity,
                        "namespace": m.namespace,
                        "tags": m.tags,
                        "importance": m.importance,
                        "created_at": (
                            m.created_at.isoformat() if m.created_at else None
                        ),
                        "metadata": m.metadata,
                        "vector_score": m.vector_score,
                        "fts_score": m.fts_score,
                    }
                    for m in hybrid_result.memories
                ],
                "total": hybrid_result.total,
                "search_type": hybrid_result.search_type,
            }

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
