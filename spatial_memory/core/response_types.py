"""TypedDict response types for MCP handler responses.

This module provides compile-time type checking for all 22 handler responses
in the Spatial Memory MCP server. Using TypedDicts enables mypy to catch
type mismatches in handler implementations.

Usage in server.py:
    def _handle_recall(self, arguments: dict[str, Any]) -> RecallResponse:
        ...
"""

from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired

# =============================================================================
# Nested TypedDicts (shared across multiple responses)
# =============================================================================


class MemoryResultDict(TypedDict):
    """Memory with similarity score from search operations."""

    id: str
    content: str
    similarity: float
    namespace: str
    tags: list[str]
    importance: float
    created_at: str  # ISO 8601 format
    metadata: dict[str, Any]
    project: NotRequired[str]
    effective_importance: NotRequired[float]  # Time-decayed importance (auto-decay)


class MemoryReferenceDict(TypedDict):
    """Minimal memory reference for nearby operations."""

    id: str
    content: str
    namespace: str


class NeighborDict(TypedDict):
    """Neighbor memory with similarity for nearby operations."""

    id: str
    content: str
    similarity: float
    namespace: str


class JourneyMemoryDict(TypedDict):
    """Memory found along a journey path."""

    id: str
    content: str
    similarity: float


class JourneyStepDict(TypedDict):
    """A step along the journey path."""

    step: int
    t: float
    nearby_memories: list[JourneyMemoryDict]
    distance_to_path: float


class WanderMemoryDict(TypedDict):
    """Memory at a wander step."""

    id: str
    content: str
    namespace: str
    tags: list[str]
    similarity: float


class WanderStepDict(TypedDict):
    """A step in a random walk."""

    step: int
    memory: WanderMemoryDict
    similarity_to_previous: float
    selection_probability: float


class RepresentativeMemoryDict(TypedDict):
    """Representative memory for a cluster."""

    id: str
    content: str


class SampleMemoryDict(TypedDict):
    """Sample memory from a cluster."""

    id: str
    content: str
    similarity: float


class ClusterDict(TypedDict):
    """A discovered cluster in regions analysis."""

    cluster_id: int
    size: int
    keywords: list[str]
    representative_memory: RepresentativeMemoryDict
    sample_memories: list[SampleMemoryDict]
    coherence: float


class VisualizationNodeDict(TypedDict):
    """A node in the visualization."""

    id: str
    x: float
    y: float
    label: str
    cluster: int
    importance: float


class VisualizationEdgeDict(TypedDict):
    """An edge in the visualization."""

    from_id: str
    to_id: str
    weight: float


class DecayedMemoryDict(TypedDict):
    """A memory with calculated decay."""

    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    decay_factor: float
    days_since_access: int
    access_count: int


class ReinforcedMemoryDict(TypedDict):
    """A memory that was reinforced."""

    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    boost_applied: float


class ExtractionDict(TypedDict):
    """An extracted memory from text."""

    content: str
    confidence: float
    pattern_matched: str
    start_pos: int
    end_pos: int
    stored: bool
    memory_id: str | None


class ConsolidationGroupDict(TypedDict):
    """A group of similar memories for consolidation."""

    representative_id: str
    member_ids: list[str]
    avg_similarity: float
    action_taken: str


class IndexInfoDict(TypedDict):
    """Information about a database index."""

    name: str
    index_type: str
    column: str
    num_indexed_rows: int
    status: str


class NamespaceInfoDict(TypedDict):
    """Information about a namespace."""

    name: str
    memory_count: int
    oldest_memory: str | None  # ISO 8601 format
    newest_memory: str | None  # ISO 8601 format


class HealthCheckDict(TypedDict):
    """A single health check result."""

    name: str
    status: str
    message: str | None
    latency_ms: float | None


class ImportValidationErrorDict(TypedDict):
    """A validation error during import."""

    row_number: int
    field: str
    error: str
    value: str | None


class ImportedMemoryDict(TypedDict):
    """Information about an imported memory."""

    id: str
    content_preview: str
    namespace: str


class HybridMemoryDict(TypedDict):
    """A memory matched by hybrid search."""

    id: str
    content: str
    similarity: float
    namespace: str
    tags: list[str]
    importance: float
    created_at: str | None  # ISO 8601 format
    metadata: dict[str, Any]
    vector_score: float | None
    fts_score: float | None
    project: NotRequired[str]
    effective_importance: NotRequired[float]  # Time-decayed importance (auto-decay)


# =============================================================================
# Handler Response TypedDicts (22 total)
# =============================================================================


class RememberResponse(TypedDict):
    """Response for remember handler."""

    id: str
    content: str
    namespace: str
    deduplicated: bool
    status: NotRequired[str]  # "stored", "rejected_quality", "rejected_exact", etc.
    quality_score: NotRequired[float]
    existing_memory: NotRequired[dict[str, Any]]


class RememberBatchResponse(TypedDict):
    """Response for remember_batch handler."""

    ids: list[str]
    count: int


class RecallResponse(TypedDict):
    """Response for recall handler."""

    memories: list[MemoryResultDict]
    total: int


class NearbyResponse(TypedDict):
    """Response for nearby handler."""

    reference: MemoryReferenceDict
    neighbors: list[NeighborDict]


class ForgetResponse(TypedDict):
    """Response for forget handler."""

    deleted: int
    ids: list[str]


class ForgetBatchResponse(TypedDict):
    """Response for forget_batch handler."""

    deleted: int
    ids: list[str]


class HealthResponse(TypedDict, total=False):
    """Response for health handler.

    Uses total=False for optional 'checks' field.
    """

    version: str
    status: str
    timestamp: str  # ISO 8601 format
    ready: bool
    alive: bool
    checks: list[HealthCheckDict]  # Optional, only with verbose=True


class JourneyResponse(TypedDict):
    """Response for journey handler."""

    start_id: str
    end_id: str
    steps: list[JourneyStepDict]
    path_coverage: float


class WanderResponse(TypedDict):
    """Response for wander handler."""

    start_id: str
    steps: list[WanderStepDict]
    total_distance: float


class RegionsResponse(TypedDict):
    """Response for regions handler."""

    clusters: list[ClusterDict]
    total_memories: int
    noise_count: int
    clustering_quality: float


class VisualizeJsonResponse(TypedDict):
    """Response for visualize handler with JSON format."""

    nodes: list[VisualizationNodeDict]
    edges: list[VisualizationEdgeDict]
    bounds: dict[str, float]
    format: str


class VisualizeTextResponse(TypedDict):
    """Response for visualize handler with mermaid/svg format."""

    format: str
    output: str
    node_count: int


# Union type for visualize response
VisualizeResponse = VisualizeJsonResponse | VisualizeTextResponse


class DecayResponse(TypedDict):
    """Response for decay handler."""

    memories_analyzed: int
    memories_decayed: int
    avg_decay_factor: float
    decayed_memories: list[DecayedMemoryDict]
    dry_run: bool


class ReinforceResponse(TypedDict):
    """Response for reinforce handler."""

    memories_reinforced: int
    avg_boost: float
    reinforced: list[ReinforcedMemoryDict]
    not_found: list[str]


class ExtractResponse(TypedDict):
    """Response for extract handler."""

    candidates_found: int
    memories_created: int
    deduplicated_count: int
    extractions: list[ExtractionDict]


class ConsolidateResponse(TypedDict):
    """Response for consolidate handler."""

    groups_found: int
    memories_merged: int
    memories_deleted: int
    groups: list[ConsolidationGroupDict]
    dry_run: bool


class StatsResponse(TypedDict):
    """Response for stats handler."""

    total_memories: int
    memories_by_namespace: dict[str, int]
    storage_bytes: int
    storage_mb: float
    estimated_vector_bytes: int
    has_vector_index: bool
    has_fts_index: bool
    indices: list[IndexInfoDict]
    num_fragments: int
    needs_compaction: bool
    table_version: int
    oldest_memory_date: str | None  # ISO 8601 format
    newest_memory_date: str | None  # ISO 8601 format
    avg_content_length: float | None


class NamespacesResponse(TypedDict):
    """Response for namespaces handler."""

    namespaces: list[NamespaceInfoDict]
    total_namespaces: int
    total_memories: int


class DeleteNamespaceResponse(TypedDict):
    """Response for delete_namespace handler."""

    namespace: str
    memories_deleted: int
    success: bool
    message: str
    dry_run: bool


class RenameNamespaceResponse(TypedDict):
    """Response for rename_namespace handler."""

    old_namespace: str
    new_namespace: str
    memories_renamed: int
    success: bool
    message: str


class ExportResponse(TypedDict):
    """Response for export_memories handler."""

    format: str
    output_path: str
    memories_exported: int
    file_size_bytes: int
    file_size_mb: float
    namespaces_included: list[str]
    duration_seconds: float
    compression: str | None


class ImportResponse(TypedDict):
    """Response for import_memories handler."""

    source_path: str
    format: str
    total_records_in_file: int
    memories_imported: int
    memories_skipped: int
    memories_failed: int
    validation_errors: list[ImportValidationErrorDict]
    namespace_override: str | None
    duration_seconds: float
    dry_run: bool
    imported_memories: list[ImportedMemoryDict]


class HybridRecallResponse(TypedDict):
    """Response for hybrid_recall handler."""

    query: str
    alpha: float
    memories: list[HybridMemoryDict]
    total: int
    search_type: str


# =============================================================================
# Type alias for any handler response
# =============================================================================

HandlerResponse = (
    RememberResponse
    | RememberBatchResponse
    | RecallResponse
    | NearbyResponse
    | ForgetResponse
    | ForgetBatchResponse
    | HealthResponse
    | JourneyResponse
    | WanderResponse
    | RegionsResponse
    | VisualizeResponse
    | DecayResponse
    | ReinforceResponse
    | ExtractResponse
    | ConsolidateResponse
    | StatsResponse
    | NamespacesResponse
    | DeleteNamespaceResponse
    | RenameNamespaceResponse
    | ExportResponse
    | ImportResponse
    | HybridRecallResponse
)
