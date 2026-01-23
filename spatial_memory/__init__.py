"""Spatial Memory MCP Server - Vector-based semantic memory for LLMs."""

__version__ = "0.1.0"
__author__ = "Jon"

# Re-export core components for convenience
from spatial_memory.config import Settings, get_settings
from spatial_memory.core import (
    ClusterInfo,
    ClusteringError,
    ConfigurationError,
    # Core services
    Database,
    EmbeddingError,
    EmbeddingService,
    Filter,
    FilterGroup,
    FilterOperator,
    JourneyStep,
    # Models
    Memory,
    MemoryNotFoundError,
    MemoryResult,
    MemorySource,
    NamespaceNotFoundError,
    # Errors
    SpatialMemoryError,
    StorageError,
    ValidationError,
    VisualizationCluster,
    VisualizationData,
    VisualizationEdge,
    VisualizationError,
    VisualizationNode,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Configuration
    "Settings",
    "get_settings",
    # Errors
    "SpatialMemoryError",
    "MemoryNotFoundError",
    "NamespaceNotFoundError",
    "EmbeddingError",
    "StorageError",
    "ValidationError",
    "ConfigurationError",
    "ClusteringError",
    "VisualizationError",
    # Models
    "Memory",
    "MemorySource",
    "MemoryResult",
    "ClusterInfo",
    "JourneyStep",
    "VisualizationNode",
    "VisualizationEdge",
    "VisualizationCluster",
    "VisualizationData",
    "Filter",
    "FilterOperator",
    "FilterGroup",
    # Core services
    "Database",
    "EmbeddingService",
]
