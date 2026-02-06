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
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from spatial_memory import __version__
from spatial_memory.config import ConfigurationError, get_settings, validate_startup
from spatial_memory.core.database import (
    clear_connection_cache,
    set_connection_pool_max_size,
)
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
from spatial_memory.core.response_types import (
    ConsolidateResponse,
    DecayResponse,
    DeleteNamespaceResponse,
    ExportResponse,
    ExtractResponse,
    ForgetBatchResponse,
    ForgetResponse,
    HandlerResponse,
    HealthResponse,
    HybridRecallResponse,
    ImportResponse,
    JourneyResponse,
    NamespacesResponse,
    NearbyResponse,
    RecallResponse,
    RegionsResponse,
    ReinforceResponse,
    RememberBatchResponse,
    RememberResponse,
    RenameNamespaceResponse,
    StatsResponse,
    VisualizeResponse,
    WanderResponse,
)
from spatial_memory.core.tracing import (
    RequestContext,
    TimingContext,
    request_context,
)
from spatial_memory.factory import ServiceFactory
from spatial_memory.tools import TOOLS

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )

logger = logging.getLogger(__name__)

# Tools that can be cached (read-only operations)
CACHEABLE_TOOLS = frozenset({"recall", "nearby", "hybrid_recall", "regions"})

# Tools that invalidate cache by namespace
NAMESPACE_INVALIDATING_TOOLS = frozenset({"remember", "forget", "forget_batch"})

# Tools that invalidate entire cache
FULL_INVALIDATING_TOOLS = frozenset({"decay", "reinforce", "consolidate"})


def _generate_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Generate a cache key from tool name and arguments.

    Args:
        tool_name: Name of the tool.
        arguments: Tool arguments (excluding _agent_id).

    Returns:
        A string cache key suitable for response caching.
    """
    # Remove _agent_id from cache key computation (same query from different agents = same result)
    cache_args = {k: v for k, v in sorted(arguments.items()) if k != "_agent_id"}
    # Create a stable string representation
    args_str = json.dumps(cache_args, sort_keys=True, default=str)
    return f"{tool_name}:{hash(args_str)}"


# Error type to response name mapping for standardized error responses
ERROR_MAPPINGS: dict[type[Exception], str] = {
    MemoryNotFoundError: "MemoryNotFound",
    ValidationError: "ValidationError",
    DecayError: "DecayError",
    ReinforcementError: "ReinforcementError",
    ExtractionError: "ExtractionError",
    ConsolidationError: "ConsolidationError",
    ExportError: "ExportError",
    MemoryImportError: "ImportError",
    PathSecurityError: "PathSecurityError",
    FileSizeLimitError: "FileSizeLimitError",
    ImportRecordLimitError: "ImportRecordLimitError",
    NamespaceNotFoundError: "NamespaceNotFound",
    NamespaceOperationError: "NamespaceOperationError",
    SpatialMemoryError: "SpatialMemoryError",
}


def _create_error_response(error: Exception, error_id: str | None = None) -> list[TextContent]:
    """Create standardized error response for tool handlers."""
    error_type = ERROR_MAPPINGS.get(type(error), "UnknownError")
    response: dict[str, Any] = {
        "error": error_type,
        "message": str(error),
        "isError": True,
    }
    if error_id:
        response["error_id"] = error_id
    return [TextContent(type="text", text=json.dumps(response))]


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

        # Configure connection pool size from settings
        set_connection_pool_max_size(self._settings.connection_pool_max_size)

        # Use ServiceFactory for dependency injection
        factory = ServiceFactory(
            settings=self._settings,
            repository=repository,
            embeddings=embeddings,
        )
        services = factory.create_all()

        # Store service references
        self._db = services.database
        self._embeddings = services.embeddings
        self._memory_service = services.memory
        self._spatial_service = services.spatial
        self._lifecycle_service = services.lifecycle
        self._utility_service = services.utility
        self._export_import_service = services.export_import

        # Rate limiting
        self._per_agent_rate_limiting = services.per_agent_rate_limiting
        self._rate_limiter = services.rate_limiter
        self._agent_rate_limiter = services.agent_rate_limiter

        # Response cache
        self._cache_enabled = services.cache_enabled
        self._cache = services.cache
        self._regions_cache_ttl = services.regions_cache_ttl

        # Auto-decay manager
        self._decay_manager = services.decay_manager
        if self._decay_manager is not None:
            self._decay_manager.start()
            logger.info("Auto-decay manager started")

        # ThreadPoolExecutor for non-blocking embedding operations
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="embed-",
        )

        # Tool handler registry for dispatch pattern
        self._tool_handlers: dict[str, Callable[[dict[str, Any]], HandlerResponse]] = {
            "remember": self._handle_remember,
            "remember_batch": self._handle_remember_batch,
            "recall": self._handle_recall,
            "nearby": self._handle_nearby,
            "forget": self._handle_forget,
            "forget_batch": self._handle_forget_batch,
            "health": self._handle_health,
            "journey": self._handle_journey,
            "wander": self._handle_wander,
            "regions": self._handle_regions,
            "visualize": self._handle_visualize,
            "decay": self._handle_decay,
            "reinforce": self._handle_reinforce,
            "extract": self._handle_extract,
            "consolidate": self._handle_consolidate,
            "stats": self._handle_stats,
            "namespaces": self._handle_namespaces,
            "delete_namespace": self._handle_delete_namespace,
            "rename_namespace": self._handle_rename_namespace,
            "export_memories": self._handle_export_memories,
            "import_memories": self._handle_import_memories,
            "hybrid_recall": self._handle_hybrid_recall,
        }

        # Log metrics availability
        if metrics_available():
            logger.info("Prometheus metrics enabled")
        else:
            logger.info("Prometheus metrics disabled (prometheus_client not installed)")

        # Create MCP server with behavioral instructions
        self._server = Server(
            name="spatial-memory",
            version=__version__,
            instructions=self._get_server_instructions(),
        )
        self._setup_handlers()

    async def _run_in_executor(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run a synchronous function in the thread pool executor.

        This allows CPU-bound or blocking operations (like embedding generation)
        to run without blocking the asyncio event loop.

        Args:
            func: The synchronous function to run.
            *args: Arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args))

    async def _handle_tool_async(self, name: str, arguments: dict[str, Any]) -> HandlerResponse:
        """Handle tool call asynchronously by running handler in executor.

        This wraps synchronous handlers to run in a thread pool, preventing
        blocking operations from stalling the event loop.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as typed dictionary.

        Raises:
            ValidationError: If tool name is unknown.
        """
        result: HandlerResponse = await self._run_in_executor(self._handle_tool, name, arguments)
        return result

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def list_tools() -> list[Tool]:
            """Return the list of available tools."""
            return TOOLS

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls with tracing, caching, and rate limiting."""
            # Extract _agent_id for tracing and rate limiting (don't pass to handler)
            agent_id = arguments.pop("_agent_id", None)

            # Apply rate limiting
            if self._per_agent_rate_limiting and self._agent_rate_limiter is not None:
                if not self._agent_rate_limiter.wait(agent_id=agent_id, timeout=30.0):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": "RateLimitExceeded",
                                    "message": "Too many requests. Please wait and try again.",
                                    "isError": True,
                                }
                            ),
                        )
                    ]
            elif self._rate_limiter is not None:
                if not self._rate_limiter.wait(timeout=30.0):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": "RateLimitExceeded",
                                    "message": "Too many requests. Please wait and try again.",
                                    "isError": True,
                                }
                            ),
                        )
                    ]

            # Use request context for tracing
            namespace = arguments.get("namespace")
            with request_context(tool_name=name, agent_id=agent_id, namespace=namespace) as ctx:
                timing = TimingContext()
                cache_hit = False

                try:
                    # Check cache for cacheable tools
                    if self._cache_enabled and self._cache is not None and name in CACHEABLE_TOOLS:
                        cache_key = _generate_cache_key(name, arguments)
                        with timing.measure("cache_lookup"):
                            cached_result = self._cache.get(cache_key)
                        if cached_result is not None:
                            cache_hit = True
                            result = cached_result
                        else:
                            with timing.measure("handler"):
                                # Run handler in executor to avoid blocking event loop
                                result = await self._handle_tool_async(name, arguments)
                            # Cache the result with appropriate TTL
                            ttl = self._regions_cache_ttl if name == "regions" else None
                            self._cache.set(cache_key, result, ttl=ttl)
                    else:
                        with timing.measure("handler"):
                            # Run handler in executor to avoid blocking event loop
                            result = await self._handle_tool_async(name, arguments)

                        # Invalidate cache on mutations
                        if self._cache_enabled and self._cache is not None:
                            self._invalidate_cache_for_tool(name, arguments)

                    # Add _meta to response if enabled
                    if self._settings.include_request_meta:
                        result["_meta"] = self._build_meta(ctx, timing, cache_hit)

                    return [TextContent(type="text", text=json.dumps(result, default=str))]
                except tuple(ERROR_MAPPINGS.keys()) as e:
                    return _create_error_response(e)
                except Exception as e:
                    error_id = str(uuid.uuid4())[:8]
                    logger.error(f"Unexpected error [{error_id}] in {name}: {e}", exc_info=True)
                    return _create_error_response(e, error_id)

    def _build_meta(
        self,
        ctx: RequestContext,
        timing: TimingContext,
        cache_hit: bool,
    ) -> dict[str, Any]:
        """Build the _meta object for response.

        Args:
            ctx: The request context.
            timing: The timing context.
            cache_hit: Whether this was a cache hit.

        Returns:
            Dictionary with request metadata.
        """
        meta: dict[str, Any] = {
            "request_id": ctx.request_id,
            "agent_id": ctx.agent_id,
            "cache_hit": cache_hit,
        }
        if self._settings.include_timing_breakdown:
            meta["timing_ms"] = timing.summary()
        return meta

    def _invalidate_cache_for_tool(self, name: str, arguments: dict[str, Any]) -> None:
        """Invalidate cache entries based on the tool that was called.

        Args:
            name: Tool name.
            arguments: Tool arguments.
        """
        if self._cache is None:
            return

        if name in FULL_INVALIDATING_TOOLS:
            self._cache.invalidate_all()
        elif name in NAMESPACE_INVALIDATING_TOOLS:
            namespace = arguments.get("namespace", "default")
            self._cache.invalidate_namespace(namespace)

    # =========================================================================
    # Tool Handler Methods
    # =========================================================================

    def _handle_remember(self, arguments: dict[str, Any]) -> RememberResponse:
        """Handle remember tool call."""
        remember_result = self._memory_service.remember(
            content=arguments["content"],
            namespace=arguments.get("namespace", "default"),
            tags=arguments.get("tags"),
            importance=arguments.get("importance", 0.5),
            metadata=arguments.get("metadata"),
        )
        return asdict(remember_result)  # type: ignore[return-value]

    def _handle_remember_batch(self, arguments: dict[str, Any]) -> RememberBatchResponse:
        """Handle remember_batch tool call."""
        batch_result = self._memory_service.remember_batch(
            memories=arguments["memories"],
        )
        return asdict(batch_result)  # type: ignore[return-value]

    def _handle_recall(self, arguments: dict[str, Any]) -> RecallResponse:
        """Handle recall tool call."""
        recall_result = self._memory_service.recall(
            query=arguments["query"],
            limit=arguments.get("limit", 5),
            namespace=arguments.get("namespace"),
            min_similarity=arguments.get("min_similarity", 0.0),
        )

        # Convert to dict list for potential decay processing
        memories_list = [
            {
                "id": m.id,
                "content": m.content,
                "similarity": m.similarity,
                "namespace": m.namespace,
                "tags": m.tags,
                "importance": m.importance,
                "created_at": m.created_at.isoformat(),
                "metadata": m.metadata,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
            }
            for m in recall_result.memories
        ]

        # Apply auto-decay if enabled (adds effective_importance, re-ranks)
        if self._decay_manager is not None and self._decay_manager.enabled:
            memories_list = self._decay_manager.apply_decay_to_results(memories_list, rerank=True)

        # Build response - include effective_importance if present
        response_memories = []
        for m in memories_list:
            mem_dict: dict[str, Any] = {
                "id": m["id"],
                "content": m["content"],
                "similarity": m["similarity"],
                "namespace": m["namespace"],
                "tags": m["tags"],
                "importance": m["importance"],
                "created_at": m["created_at"],
                "metadata": m["metadata"],
            }
            if "effective_importance" in m:
                mem_dict["effective_importance"] = m["effective_importance"]
            response_memories.append(mem_dict)

        return {
            "memories": response_memories,  # type: ignore[typeddict-item]
            "total": len(response_memories),
        }

    def _handle_nearby(self, arguments: dict[str, Any]) -> NearbyResponse:
        """Handle nearby tool call."""
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

    def _handle_forget(self, arguments: dict[str, Any]) -> ForgetResponse:
        """Handle forget tool call."""
        forget_result = self._memory_service.forget(
            memory_id=arguments["memory_id"],
        )
        return asdict(forget_result)  # type: ignore[return-value]

    def _handle_forget_batch(self, arguments: dict[str, Any]) -> ForgetBatchResponse:
        """Handle forget_batch tool call."""
        forget_batch_result = self._memory_service.forget_batch(
            memory_ids=arguments["memory_ids"],
        )
        return asdict(forget_batch_result)  # type: ignore[return-value]

    def _handle_health(self, arguments: dict[str, Any]) -> HealthResponse:
        """Handle health tool call."""
        verbose = arguments.get("verbose", False)

        health_checker = HealthChecker(
            database=self._db,
            embeddings=self._embeddings,
            storage_path=self._settings.memory_path,
        )

        report = health_checker.get_health_report()

        result: HealthResponse = {
            "version": __version__,
            "status": report.status.value,
            "timestamp": report.timestamp.isoformat(),
            "ready": health_checker.is_ready(),
            "alive": health_checker.is_alive(),
        }

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

    def _handle_journey(self, arguments: dict[str, Any]) -> JourneyResponse:
        """Handle journey tool call."""
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

    def _handle_wander(self, arguments: dict[str, Any]) -> WanderResponse:
        """Handle wander tool call."""
        start_id = arguments.get("start_id")
        if start_id is None:
            all_memories = self._memory_service.recall(
                query="any topic",
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

    def _handle_regions(self, arguments: dict[str, Any]) -> RegionsResponse:
        """Handle regions tool call."""
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

    def _handle_visualize(self, arguments: dict[str, Any]) -> VisualizeResponse:
        """Handle visualize tool call."""
        visualize_result = self._spatial_service.visualize(
            memory_ids=arguments.get("memory_ids"),
            namespace=arguments.get("namespace"),
            format=arguments.get("format", "json"),
            dimensions=arguments.get("dimensions", 2),
            include_edges=arguments.get("include_edges", True),
        )
        output_format = arguments.get("format", "json")
        if output_format in ("mermaid", "svg"):
            return {
                "format": output_format,
                "output": visualize_result.output,
                "node_count": len(visualize_result.nodes),
            }
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
            ]
            if visualize_result.edges
            else [],
            "bounds": visualize_result.bounds,
            "format": visualize_result.format,
        }

    def _handle_decay(self, arguments: dict[str, Any]) -> DecayResponse:
        """Handle decay tool call."""
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

    def _handle_reinforce(self, arguments: dict[str, Any]) -> ReinforceResponse:
        """Handle reinforce tool call."""
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

    def _handle_extract(self, arguments: dict[str, Any]) -> ExtractResponse:
        """Handle extract tool call."""
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

    def _handle_consolidate(self, arguments: dict[str, Any]) -> ConsolidateResponse:
        """Handle consolidate tool call."""
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

    def _handle_stats(self, arguments: dict[str, Any]) -> StatsResponse:
        """Handle stats tool call."""
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
            ]
            if stats_result.indices
            else [],
            "num_fragments": stats_result.num_fragments,
            "needs_compaction": stats_result.needs_compaction,
            "table_version": stats_result.table_version,
            "oldest_memory_date": (
                stats_result.oldest_memory_date.isoformat()
                if stats_result.oldest_memory_date
                else None
            ),
            "newest_memory_date": (
                stats_result.newest_memory_date.isoformat()
                if stats_result.newest_memory_date
                else None
            ),
            "avg_content_length": stats_result.avg_content_length,
        }

    def _handle_namespaces(self, arguments: dict[str, Any]) -> NamespacesResponse:
        """Handle namespaces tool call."""
        namespaces_result = self._utility_service.namespaces(
            include_stats=arguments.get("include_stats", True),
        )
        return {
            "namespaces": [
                {
                    "name": ns.name,
                    "memory_count": ns.memory_count,
                    "oldest_memory": (ns.oldest_memory.isoformat() if ns.oldest_memory else None),
                    "newest_memory": (ns.newest_memory.isoformat() if ns.newest_memory else None),
                }
                for ns in namespaces_result.namespaces
            ],
            "total_namespaces": namespaces_result.total_namespaces,
            "total_memories": namespaces_result.total_memories,
        }

    def _handle_delete_namespace(self, arguments: dict[str, Any]) -> DeleteNamespaceResponse:
        """Handle delete_namespace tool call."""
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

    def _handle_rename_namespace(self, arguments: dict[str, Any]) -> RenameNamespaceResponse:
        """Handle rename_namespace tool call."""
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

    def _handle_export_memories(self, arguments: dict[str, Any]) -> ExportResponse:
        """Handle export_memories tool call."""
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

    def _handle_import_memories(self, arguments: dict[str, Any]) -> ImportResponse:
        """Handle import_memories tool call."""
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
            ]
            if import_result.validation_errors
            else [],
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
            ]
            if import_result.imported_memories
            else [],
        }

    def _handle_hybrid_recall(self, arguments: dict[str, Any]) -> HybridRecallResponse:
        """Handle hybrid_recall tool call."""
        hybrid_result = self._utility_service.hybrid_recall(
            query=arguments["query"],
            alpha=arguments.get("alpha", 0.5),
            limit=arguments.get("limit", 5),
            namespace=arguments.get("namespace"),
            min_similarity=arguments.get("min_similarity", 0.0),
        )

        # Convert to dict list for potential decay processing
        memories_list = [
            {
                "id": m.id,
                "content": m.content,
                "similarity": m.similarity,
                "namespace": m.namespace,
                "tags": m.tags,
                "importance": m.importance,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "metadata": m.metadata,
                "vector_score": m.vector_score,
                "fts_score": m.fts_score,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
            }
            for m in hybrid_result.memories
        ]

        # Apply auto-decay if enabled (adds effective_importance, re-ranks)
        if self._decay_manager is not None and self._decay_manager.enabled:
            memories_list = self._decay_manager.apply_decay_to_results(memories_list, rerank=True)

        # Build response - include effective_importance if present
        response_memories = []
        for m in memories_list:
            mem_dict: dict[str, Any] = {
                "id": m["id"],
                "content": m["content"],
                "similarity": m["similarity"],
                "namespace": m["namespace"],
                "tags": m["tags"],
                "importance": m["importance"],
                "created_at": m["created_at"],
                "metadata": m["metadata"],
                "vector_score": m.get("vector_score"),
                "fts_score": m.get("fts_score"),
            }
            if "effective_importance" in m:
                mem_dict["effective_importance"] = m["effective_importance"]
            response_memories.append(mem_dict)

        return {
            "query": hybrid_result.query,
            "alpha": hybrid_result.alpha,
            "memories": response_memories,  # type: ignore[typeddict-item]
            "total": len(response_memories),
            "search_type": hybrid_result.search_type,
        }

    # =========================================================================
    # Tool Routing
    # =========================================================================

    def _handle_tool(self, name: str, arguments: dict[str, Any]) -> HandlerResponse:
        """Route tool call to appropriate handler.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as typed dictionary.

        Raises:
            ValidationError: If tool name is unknown.
        """
        # Record metrics for this tool call
        with record_request(name, "success"):
            return self._handle_tool_impl(name, arguments)

    def _handle_tool_impl(self, name: str, arguments: dict[str, Any]) -> HandlerResponse:
        """Implementation of tool routing using dispatch pattern.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as dictionary.

        Raises:
            ValidationError: If tool name is unknown.
        """
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise ValidationError(f"Unknown tool: {name}")
        return handler(arguments)

    @staticmethod
    def _get_server_instructions() -> str:
        """Return behavioral instructions for Claude when using spatial-memory.

        These instructions are automatically injected into Claude's system prompt
        when the MCP server connects, enabling proactive memory management without
        requiring user configuration.
        """
        return """## Spatial Memory System

You have access to a persistent semantic memory system. Use it proactively to
build cumulative knowledge across sessions.

### Session Start
At conversation start, call `recall` with the user's apparent task/context to
load relevant memories. Present insights naturally:
- Good: "Based on previous work, you decided to use PostgreSQL because..."
- Bad: "The database returned: [{id: '...', content: '...'}]"

### Auto-Save Behavior

Memories are saved using a 3-tier system. Before saving anything, call `recall`
with a brief summary to check for duplicates. Skip saving if a similar memory
already exists.

#### Tier 1 — Auto-save (save immediately, notify the user)
Save these automatically without asking. After saving, display a brief note:
`> Memorized: [one-line summary of what was saved]`

Signal phrases and situations:
- **Decisions with reasoning**: "decided to use X because Y", "we chose...",
  "the approach is...", "going with X over Y because..."
- **Bug fixes and solutions**: "the fix was...", "resolved by...",
  "the solution is...", "fixed it by..."
- **Error root causes**: "the issue was caused by...", "failed because...",
  "the error was due to...", "it broke because..."
- **Architecture choices**: "we'll structure it as...", "the design is...",
  "the architecture will be..."

Save with: importance 0.8-1.0, namespace by project or "decisions"/"errors",
descriptive tags for technologies and concepts involved. Include full context
and reasoning so future agents can understand without prior conversation.

#### Tier 2 — Ask first ("Save this? y/n")
Ask briefly before saving these:
- **General patterns and learnings**: "the trick is...", "pattern:",
  "this pattern works...", "always do X when..."
- **Preferences and conventions**: "we prefer...", "the team standard is...",
  "convention here is..."
- **Configuration discoveries**: "you need to set X to Y",
  "the config requires...", "important setting:..."
- **Workarounds and gotchas**: "watch out for...", "the workaround is...",
  "gotcha:...", "caveat:..."

Save with: importance 0.5-0.7, namespace "patterns" or by project,
descriptive tags.

#### Tier 3 — Never save
Do NOT save:
- Trivial observations, greetings, status updates
- Information that already exists in memory (always check with `recall` first)
- Speculative or unconfirmed information
- Temporary debugging steps or intermediate exploration

### Synthesizing Answers
When using `recall` or `hybrid_recall`, present results as natural knowledge:
- Integrate memories into your response conversationally
- Reference prior decisions: "You previously decided X because Y"
- Don't expose raw JSON or tool mechanics to the user

### Auto-Extract for Long Sessions
For significant problem-solving conversations (debugging sessions, architecture
discussions), offer:
"This session had good learnings. Extract key memories? y/n"
Then use `extract` to automatically capture important information.

### Tool Selection Guide
- `remember`: Store a single memory with full context
- `recall`: Semantic search for relevant memories
- `hybrid_recall`: Combined keyword + semantic search (better for specific terms)
- `extract`: Auto-extract memories from conversation text
- `nearby`: Find memories similar to a known memory
- `regions`: Discover topic clusters in memory space
- `journey`: Navigate conceptual path between two memories"""

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
        # Stop the decay manager (flushes pending updates)
        if self._decay_manager is not None:
            self._decay_manager.stop()

        # Shutdown the thread pool executor
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

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
