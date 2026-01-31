"""Data models for Spatial Memory MCP Server."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from spatial_memory.core.utils import utc_now

# Type alias for filter values - covers all expected filter value types
FilterValue = str | int | float | bool | datetime | list[str] | list[int] | list[float]


class MemorySource(str, Enum):
    """Source of a memory."""

    MANUAL = "manual"  # Explicitly stored via remember()
    EXTRACTED = "extracted"  # Auto-extracted from conversation
    CONSOLIDATED = "consolidated"  # Result of consolidation


class Memory(BaseModel):
    """A single memory in the spatial memory system."""

    id: str = Field(..., description="Unique identifier (UUID)")
    content: str = Field(..., description="Text content of the memory", max_length=100000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed: datetime = Field(default_factory=utc_now)
    access_count: int = Field(default=0, ge=0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    namespace: str = Field(default="default")
    tags: list[str] = Field(default_factory=list)
    source: MemorySource = Field(default=MemorySource.MANUAL)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResult(BaseModel):
    """A memory with similarity score from search."""

    id: str
    content: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    namespace: str
    tags: list[str] = Field(default_factory=list)
    importance: float
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClusterInfo(BaseModel):
    """Information about a discovered cluster/region."""

    cluster_id: int
    label: str  # Auto-generated or centroid-based
    size: int
    centroid_memory_id: str  # Memory closest to centroid
    sample_memories: list[str]  # Sample content from cluster
    coherence: float = Field(ge=0.0, le=1.0)  # How tight the cluster is


class JourneyStep(BaseModel):
    """A step in a journey between two memories.

    Represents a point along the interpolated path between two memories,
    with nearby memories discovered at that position.
    """

    step: int
    t: float = Field(..., ge=0.0, le=1.0, description="Interpolation parameter [0, 1]")
    position: list[float] = Field(..., description="Interpolated vector position")
    nearby_memories: list[MemoryResult] = Field(
        default_factory=list, description="Memories near this path position"
    )
    distance_to_path: float = Field(
        default=0.0, ge=0.0, description="Distance from nearest memory to ideal path"
    )


class JourneyResult(BaseModel):
    """Result of a journey operation between two memories.

    Contains the full path with steps and discovered memories along the way.
    """

    start_id: str = Field(..., description="Starting memory ID")
    end_id: str = Field(..., description="Ending memory ID")
    steps: list[JourneyStep] = Field(default_factory=list, description="Journey steps")
    path_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of path with nearby memories",
    )


class WanderStep(BaseModel):
    """A single step in a random walk through memory space.

    Represents transitioning from one memory to another based on
    similarity-weighted random selection.
    """

    step: int = Field(..., ge=0, description="Step number in the walk")
    memory: MemoryResult = Field(..., description="Memory at this step")
    similarity_to_previous: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to the previous step's memory",
    )
    selection_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability this memory was selected",
    )


class WanderResult(BaseModel):
    """Result of a wander (random walk) operation.

    Contains the path taken during the random walk through memory space.
    """

    start_id: str = Field(..., description="Starting memory ID")
    steps: list[WanderStep] = Field(default_factory=list, description="Walk steps")
    total_distance: float = Field(
        default=0.0, ge=0.0, description="Total distance traveled in embedding space"
    )


class RegionCluster(BaseModel):
    """A cluster discovered during regions analysis.

    Represents a semantic region in memory space with coherent memories.
    """

    cluster_id: int = Field(..., description="Cluster identifier (-1 for noise)")
    size: int = Field(..., ge=0, description="Number of memories in cluster")
    representative_memory: MemoryResult = Field(
        ..., description="Memory closest to cluster centroid"
    )
    sample_memories: list[MemoryResult] = Field(
        default_factory=list, description="Sample memories from the cluster"
    )
    coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Internal cluster coherence (tightness)",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords describing the cluster"
    )


class RegionsResult(BaseModel):
    """Result of a regions (clustering) operation.

    Contains discovered clusters and clustering quality metrics.
    """

    clusters: list[RegionCluster] = Field(
        default_factory=list, description="Discovered clusters"
    )
    noise_count: int = Field(
        default=0, ge=0, description="Number of memories not in any cluster"
    )
    total_memories: int = Field(
        default=0, ge=0, description="Total memories analyzed"
    )
    clustering_quality: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Overall clustering quality (silhouette score)",
    )


class VisualizationNode(BaseModel):
    """A node in the visualization."""

    id: str
    x: float
    y: float
    label: str
    cluster: int = -1  # -1 for noise/unclustered
    importance: float = 0.5
    highlighted: bool = False


class VisualizationEdge(BaseModel):
    """An edge connecting two nodes in visualization."""

    from_id: str
    to_id: str
    weight: float = Field(ge=0.0, le=1.0)


class VisualizationCluster(BaseModel):
    """Cluster metadata for visualization."""

    id: int
    label: str
    color: str
    center_x: float
    center_y: float


class VisualizationData(BaseModel):
    """Data for visualizing the memory space."""

    nodes: list[VisualizationNode]
    edges: list[VisualizationEdge] = Field(default_factory=list)
    clusters: list[VisualizationCluster] = Field(default_factory=list)
    bounds: dict[str, float] = Field(
        default_factory=lambda: {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}
    )


class VisualizationResult(BaseModel):
    """Result of a visualization operation.

    Contains the complete visualization output including nodes, edges,
    and the formatted output string.
    """

    nodes: list[VisualizationNode] = Field(
        default_factory=list, description="Visualization nodes"
    )
    edges: list[VisualizationEdge] = Field(
        default_factory=list, description="Connections between nodes"
    )
    bounds: dict[str, float] = Field(
        default_factory=lambda: {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
        },
        description="Coordinate bounds of the visualization",
    )
    format: str = Field(
        default="json",
        description="Output format (json, mermaid, svg)",
    )
    output: str = Field(
        default="", description="Formatted output string in the specified format"
    )


class FilterOperator(str, Enum):
    """Filter operators for querying memories."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NIN = "nin"  # Not in list
    CONTAINS = "contains"  # String/list contains


class Filter(BaseModel):
    """A single filter condition."""

    field: str
    operator: FilterOperator
    value: FilterValue


class FilterGroup(BaseModel):
    """A group of filters with logical operator."""

    operator: Literal["and", "or"] = "and"
    filters: list[Filter | FilterGroup] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_filters_not_empty(self) -> FilterGroup:
        """Validate that filters list is not empty."""
        if not self.filters:
            raise ValueError("FilterGroup must contain at least one filter")
        return self


# Update forward references
FilterGroup.model_rebuild()


# =============================================================================
# Lifecycle Enums
# =============================================================================


class DecayFunction(str, Enum):
    """Decay function types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


class BoostType(str, Enum):
    """Reinforcement boost types."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    SET_VALUE = "set_value"


class ConsolidationStrategy(str, Enum):
    """Consolidation strategies."""

    KEEP_NEWEST = "keep_newest"
    KEEP_OLDEST = "keep_oldest"
    KEEP_HIGHEST_IMPORTANCE = "keep_highest_importance"
    MERGE_CONTENT = "merge_content"


class ExtractionPatternType(str, Enum):
    """Types of extracted content."""

    DECISION = "decision"
    DEFINITION = "definition"
    SOLUTION = "solution"
    ERROR = "error"
    PATTERN = "pattern"
    EXPLICIT = "explicit"
    IMPORTANT = "important"


# =============================================================================
# Lifecycle Result Dataclasses
# =============================================================================


@dataclass
class DecayedMemory:
    """A memory with calculated decay."""

    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    decay_factor: float
    days_since_access: int
    access_count: int


@dataclass
class DecayResult:
    """Result of decay operation."""

    memories_analyzed: int
    memories_decayed: int
    avg_decay_factor: float
    decayed_memories: list[DecayedMemory] = field(default_factory=list)
    dry_run: bool = True


@dataclass
class ReinforcedMemory:
    """A memory that was reinforced."""

    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    boost_applied: float


@dataclass
class ReinforceResult:
    """Result of reinforcement operation."""

    memories_reinforced: int
    avg_boost: float
    reinforced: list[ReinforcedMemory] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)


@dataclass
class ExtractedMemory:
    """A memory candidate extracted from text."""

    content: str
    confidence: float
    pattern_matched: str
    start_pos: int
    end_pos: int
    stored: bool  # False if deduplicated
    memory_id: str | None = None  # Set if stored


@dataclass
class ExtractResult:
    """Result of memory extraction."""

    candidates_found: int
    memories_created: int
    deduplicated_count: int
    extractions: list[ExtractedMemory] = field(default_factory=list)


@dataclass
class ConsolidationGroup:
    """A group of similar memories."""

    representative_id: str
    member_ids: list[str]
    avg_similarity: float
    action_taken: str  # "merged", "deleted", "preview"


@dataclass
class ConsolidateResult:
    """Result of consolidation."""

    groups_found: int
    memories_merged: int
    memories_deleted: int
    groups: list[ConsolidationGroup] = field(default_factory=list)
    dry_run: bool = True


# =============================================================================
# Phase 5 Utility Result Dataclasses
# =============================================================================


@dataclass
class IndexInfo:
    """Information about a single database index."""

    name: str
    index_type: str
    column: str
    num_indexed_rows: int
    status: str  # "ready", "building", "needs_update"


@dataclass
class StatsResult:
    """Result of database statistics query."""

    total_memories: int
    memories_by_namespace: dict[str, int]
    storage_bytes: int
    storage_mb: float
    estimated_vector_bytes: int
    has_vector_index: bool
    has_fts_index: bool
    indices: list[IndexInfo]
    num_fragments: int
    needs_compaction: bool
    table_version: int
    oldest_memory_date: datetime | None = None
    newest_memory_date: datetime | None = None
    avg_content_length: float | None = None


@dataclass
class NamespaceInfo:
    """Information about a single namespace."""

    name: str
    memory_count: int
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None


@dataclass
class NamespacesResult:
    """Result of namespace listing."""

    namespaces: list[NamespaceInfo]
    total_namespaces: int
    total_memories: int


@dataclass
class DeleteNamespaceResult:
    """Result of namespace deletion."""

    namespace: str
    memories_deleted: int
    success: bool
    message: str
    dry_run: bool = False


@dataclass
class RenameNamespaceResult:
    """Result of namespace rename."""

    old_namespace: str
    new_namespace: str
    memories_renamed: int
    success: bool
    message: str


@dataclass
class ExportResult:
    """Result of memory export."""

    format: str  # parquet, json, csv
    output_path: str
    memories_exported: int
    file_size_bytes: int
    file_size_mb: float
    namespaces_included: list[str]
    duration_seconds: float
    compression: str | None = None


@dataclass
class ImportedMemory:
    """Information about a single imported memory."""

    id: str
    content_preview: str
    namespace: str
    was_deduplicated: bool = False
    original_id: str | None = None


@dataclass
class ImportValidationError:
    """A validation error during import."""

    row_number: int
    field: str
    error: str
    value: str | None = None


@dataclass
class ImportResult:
    """Result of memory import."""

    source_path: str
    format: str
    total_records_in_file: int
    memories_imported: int
    memories_skipped: int
    memories_failed: int
    validation_errors: list[ImportValidationError]
    duration_seconds: float
    namespace_override: str | None = None
    imported_memories: list[ImportedMemory] | None = None


@dataclass
class HybridMemoryMatch:
    """A memory matched by hybrid search."""

    id: str
    content: str
    similarity: float
    namespace: str
    tags: list[str]
    importance: float
    created_at: datetime
    metadata: dict[str, Any]
    vector_score: float | None = None
    fts_score: float | None = None
    combined_score: float = 0.0


@dataclass
class HybridRecallResult:
    """Result of hybrid recall operation."""

    query: str
    alpha: float
    memories: list[HybridMemoryMatch]
    total: int
    search_type: str = "hybrid"


# =============================================================================
# Phase 5 Service Configuration Dataclasses
# =============================================================================


@dataclass
class UtilityConfig:
    """Configuration for utility operations."""

    hybrid_default_alpha: float = 0.5
    hybrid_min_alpha: float = 0.0
    hybrid_max_alpha: float = 1.0
    stats_include_index_details: bool = True
    namespace_batch_size: int = 1000
    delete_namespace_require_confirmation: bool = True


@dataclass
class ExportImportConfig:
    """Configuration for export/import operations."""

    default_export_format: str = "parquet"
    export_batch_size: int = 5000
    import_batch_size: int = 1000
    import_deduplicate: bool = False
    import_dedup_threshold: float = 0.95
    validate_on_import: bool = True
    parquet_compression: str = "zstd"
    csv_include_vectors: bool = False
    max_export_records: int = 0  # 0 = unlimited
    max_import_records: int = 100_000  # Maximum records per import
