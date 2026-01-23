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
    """A step in a journey between two memories."""

    step: int
    nearest_memory: MemoryResult | None = None  # Closest actual memory
    distance_to_ideal: float  # How far the nearest memory is from ideal path


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
