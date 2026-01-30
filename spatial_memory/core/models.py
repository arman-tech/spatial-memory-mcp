"""Data models for Spatial Memory MCP Server."""

from __future__ import annotations

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
