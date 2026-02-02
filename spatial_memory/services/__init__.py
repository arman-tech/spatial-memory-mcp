"""Service layer for Spatial Memory MCP Server."""

from spatial_memory.core.errors import (
    ConsolidationError,
    DecayError,
    ExtractionError,
    NamespaceOperationError,
    ReinforcementError,
)
from spatial_memory.services.export_import import (
    ExportImportService,
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
from spatial_memory.services.utility import (
    UtilityService,
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
    # Utility
    "NamespaceOperationError",
    "UtilityService",
    # Export/Import
    "ExportImportService",
]
