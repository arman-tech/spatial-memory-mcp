"""MCP Tool definitions for Spatial Memory Server.

This module contains all the tool definitions that are exposed
to MCP clients for interacting with the spatial memory system.
"""

from typing import Any

from mcp.types import Tool

# Common parameter: _agent_id for request tracing and per-agent rate limiting
_AGENT_ID_PARAM: dict[str, Any] = {
    "_agent_id": {
        "type": "string",
        "description": "Optional agent identifier for request tracing and per-agent rate limiting.",
    },
}

# Common parameter: project for scoping memories to a specific project
_PROJECT_PARAM: dict[str, Any] = {
    "project": {
        "type": "string",
        "description": (
            "Project scope for this operation. "
            "Omit to auto-detect from environment. "
            'Use "*" to search across all projects.'
        ),
    },
}


def _add_agent_id(properties: dict[str, Any]) -> dict[str, Any]:
    """Add _agent_id parameter to tool properties."""
    return {**properties, **_AGENT_ID_PARAM}


def _add_common_params(properties: dict[str, Any], *, project: bool = False) -> dict[str, Any]:
    """Add common parameters (_agent_id, and optionally project) to tool properties."""
    result = {**properties, **_AGENT_ID_PARAM}
    if project:
        result = {**result, **_PROJECT_PARAM}
    return result


# Tool definitions for MCP
TOOLS = [
    Tool(
        name="remember",
        description="Store a new memory in the spatial memory system.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                    "idempotency_key": {
                        "type": "string",
                        "description": (
                            "Optional unique key for idempotent writes. "
                            "If the same key is used again, returns the cached result "
                            "instead of creating a duplicate."
                        ),
                    },
                },
                project=True,
            ),
            "required": ["content"],
        },
    ),
    Tool(
        name="remember_batch",
        description="Store multiple memories efficiently in a single operation.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                project=True,
            ),
            "required": ["memories"],
        },
    ),
    Tool(
        name="recall",
        description="Search for similar memories using semantic similarity.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                project=True,
            ),
            "required": ["query"],
        },
    ),
    Tool(
        name="nearby",
        description="Find memories similar to a specific memory.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                project=True,
            ),
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="forget",
        description="Delete a memory by its ID.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to delete",
                    },
                }
            ),
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="forget_batch",
        description="Delete multiple memories by their IDs.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of memory IDs to delete",
                    },
                }
            ),
            "required": ["memory_ids"],
        },
    ),
    Tool(
        name="health",
        description="Check system health status.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "verbose": {
                        "type": "boolean",
                        "description": "Include detailed check results",
                        "default": False,
                    },
                }
            ),
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
            "properties": _add_agent_id(
                {
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
                }
            ),
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
            "properties": _add_agent_id(
                {
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
                }
            ),
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
            "properties": _add_common_params(
                {
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
                project=True,
            ),
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
            "properties": _add_common_params(
                {
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
                project=True,
            ),
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
            "properties": _add_common_params(
                {
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
                project=True,
            ),
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
            "properties": _add_agent_id(
                {
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
                }
            ),
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
            "properties": _add_common_params(
                {
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
                project=True,
            ),
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
            "properties": _add_common_params(
                {
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
                project=True,
            ),
            "required": ["namespace"],
        },
    ),
    # Phase 5: Utility Tools
    Tool(
        name="stats",
        description="Get database statistics and health metrics.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                project=True,
            ),
        },
    ),
    Tool(
        name="namespaces",
        description="List all namespaces with memory counts and date ranges.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "include_stats": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include memory counts and date ranges per namespace",
                    },
                },
            ),
        },
    ),
    Tool(
        name="delete_namespace",
        description="Delete all memories in a namespace. DESTRUCTIVE - use dry_run first.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to delete",
                    },
                    "confirm": {
                        "type": "boolean",
                        "default": False,
                        "description": "Confirm deletion (required when dry_run=false)",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "Preview deletion without executing",
                    },
                }
            ),
            "required": ["namespace"],
        },
    ),
    Tool(
        name="rename_namespace",
        description="Rename a namespace, moving all its memories to the new name.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "old_namespace": {
                        "type": "string",
                        "description": "Current namespace name",
                    },
                    "new_namespace": {
                        "type": "string",
                        "description": "New namespace name",
                    },
                }
            ),
            "required": ["old_namespace", "new_namespace"],
        },
    ),
    Tool(
        name="export_memories",
        description="Export memories to file (Parquet, JSON, or CSV format).",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
                    "output_path": {
                        "type": "string",
                        "description": "Path for output file (extension determines format)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["parquet", "json", "csv"],
                        "description": "Export format (auto-detected from extension)",
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
                project=True,
            ),
            "required": ["output_path"],
        },
    ),
    Tool(
        name="import_memories",
        description="Import memories from file with validation. Use dry_run=true first.",
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "source_path": {
                        "type": "string",
                        "description": "Path to source file",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["parquet", "json", "csv"],
                        "description": "Import format (auto-detected from extension)",
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
            ),
            "required": ["source_path"],
        },
    ),
    Tool(
        name="hybrid_recall",
        description="Search memories using combined vector and keyword (full-text) search.",
        inputSchema={
            "type": "object",
            "properties": _add_common_params(
                {
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
                project=True,
            ),
            "required": ["query"],
        },
    ),
    Tool(
        name="setup_hooks",
        description=(
            "Generate hook configuration for cognitive offloading. "
            "Returns ready-to-use hooks JSON for Claude Code or Cursor."
        ),
        inputSchema={
            "type": "object",
            "properties": _add_agent_id(
                {
                    "client": {
                        "type": "string",
                        "enum": [
                            "claude-code",
                            "cursor",
                        ],
                        "default": "claude-code",
                        "description": "Target client for hook configuration",
                    },
                    "python_path": {
                        "type": "string",
                        "description": (
                            "Python interpreter path. "
                            "Defaults to the interpreter running the server."
                        ),
                    },
                    "include_session_start": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include the SessionStart recall nudge hook",
                    },
                    "include_mcp_config": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include MCP server configuration in output",
                    },
                }
            ),
        },
    ),
]
