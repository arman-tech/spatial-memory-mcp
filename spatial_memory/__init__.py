"""Spatial Memory MCP Server - Vector-based semantic memory for LLMs."""

__version__ = "0.1.0"
__author__ = "arman-tech"

# Re-export core components for convenience
# Adapters
from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
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

# Server
from spatial_memory.server import SpatialMemoryServer, create_server

# Services
from spatial_memory.services.memory import (
    ForgetResult,
    MemoryService,
    NearbyResult,
    RecallResult,
    RememberBatchResult,
    RememberResult,
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
    # Services
    "MemoryService",
    "RememberResult",
    "RememberBatchResult",
    "RecallResult",
    "NearbyResult",
    "ForgetResult",
    # Adapters
    "LanceDBMemoryRepository",
    # Server
    "SpatialMemoryServer",
    "create_server",
]
