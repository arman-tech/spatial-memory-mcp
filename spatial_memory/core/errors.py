"""Custom exceptions for Spatial Memory MCP Server."""


class SpatialMemoryError(Exception):
    """Base exception for all spatial memory errors."""

    pass


class MemoryNotFoundError(SpatialMemoryError):
    """Raised when a memory ID doesn't exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class NamespaceNotFoundError(SpatialMemoryError):
    """Raised when a namespace doesn't exist."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        super().__init__(f"Namespace not found: {namespace}")


class EmbeddingError(SpatialMemoryError):
    """Raised when embedding generation fails."""

    pass


class StorageError(SpatialMemoryError):
    """Raised when database operations fail."""

    pass


class ValidationError(SpatialMemoryError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(SpatialMemoryError):
    """Raised when configuration is invalid."""

    pass


class ClusteringError(SpatialMemoryError):
    """Raised when clustering fails (e.g., too few memories)."""

    pass


class VisualizationError(SpatialMemoryError):
    """Raised when visualization generation fails."""

    pass


class InsufficientMemoriesError(SpatialMemoryError):
    """Raised when operation requires more memories than available."""

    def __init__(self, required: int, available: int, operation: str) -> None:
        self.required = required
        self.available = available
        self.operation = operation
        super().__init__(
            f"{operation} requires at least {required} memories, but only {available} available"
        )


class JourneyError(SpatialMemoryError):
    """Raised when journey path cannot be computed."""

    pass


class WanderError(SpatialMemoryError):
    """Raised when wander cannot continue."""

    pass


class DecayError(SpatialMemoryError):
    """Raised when decay calculation or application fails."""

    pass


class ReinforcementError(SpatialMemoryError):
    """Raised when reinforcement fails."""

    pass


class ExtractionError(SpatialMemoryError):
    """Raised when memory extraction fails."""

    pass


class ConsolidationError(SpatialMemoryError):
    """Raised when consolidation fails."""

    pass
