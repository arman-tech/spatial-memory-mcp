"""Service layer for Spatial Memory MCP Server."""

from spatial_memory.core.errors import (
    ConsolidationError,
    DecayError,
    ExtractionError,
    ReinforcementError,
)
from spatial_memory.services.lifecycle import (
    ConsolidateResult,
    ConsolidationGroupResult,
    DecayedMemory,
    DecayResult,
    ExtractedMemory,
    ExtractResult,
    LifecycleConfig,
    LifecycleService,
    ReinforcedMemory,
    ReinforceResult,
)
from spatial_memory.services.memory import (
    ForgetResult,
    MemoryService,
    NearbyResult,
    RecallResult,
    RememberResult,
)
from spatial_memory.services.spatial import (
    SpatialConfig,
    SpatialService,
)

__all__ = [
    # Lifecycle
    "ConsolidateResult",
    "ConsolidationError",
    "ConsolidationGroupResult",
    "DecayedMemory",
    "DecayError",
    "DecayResult",
    "ExtractedMemory",
    "ExtractionError",
    "ExtractResult",
    "LifecycleConfig",
    "LifecycleService",
    "ReinforcedMemory",
    "ReinforcementError",
    "ReinforceResult",
    # Memory
    "ForgetResult",
    "MemoryService",
    "NearbyResult",
    "RecallResult",
    "RememberResult",
    # Spatial
    "SpatialConfig",
    "SpatialService",
]
