"""Configuration system for Spatial Memory MCP Server."""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Spatial Memory Server Configuration."""

    # Storage
    memory_path: Path = Field(
        default=Path("./.spatial-memory"),
        description="Path to LanceDB storage directory",
    )

    # Embedding Model
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name or 'openai:model-name'",
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Embedding vector dimensions (auto-detected if not set)",
    )

    # OpenAI (optional)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for API-based embeddings",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use",
    )

    # Server
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Memory Defaults
    default_namespace: str = Field(
        default="default",
        description="Default namespace for memories",
    )
    default_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default importance for new memories",
    )

    # Limits
    max_batch_size: int = Field(
        default=100,
        description="Maximum memories per batch operation",
    )
    max_recall_limit: int = Field(
        default=100,
        description="Maximum results from recall",
    )
    max_journey_steps: int = Field(
        default=20,
        description="Maximum steps in journey",
    )
    max_wander_steps: int = Field(
        default=20,
        description="Maximum steps in wander",
    )
    max_visualize_memories: int = Field(
        default=500,
        description="Maximum memories in visualization",
    )

    # Decay Settings
    decay_time_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for time-based decay (0-1)",
    )
    decay_access_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for access-based decay (0-1)",
    )
    decay_days_threshold: int = Field(
        default=30,
        description="Days without access before decay starts",
    )

    # Clustering
    min_cluster_size: int = Field(
        default=3,
        ge=2,
        description="Minimum memories for a cluster",
    )

    # Indexing
    vector_index_threshold: int = Field(
        default=10_000,
        ge=1000,
        description="Create vector index when dataset exceeds this size",
    )
    auto_create_indexes: bool = Field(
        default=True,
        description="Automatically create indexes when thresholds are met",
    )
    index_nprobes: int = Field(
        default=20,
        ge=1,
        description="Number of partitions to search (higher = better recall, slower)",
    )
    index_refine_factor: int = Field(
        default=5,
        ge=1,
        description="Re-rank top (refine_factor * limit) candidates for accuracy",
    )

    # Hybrid Search
    enable_fts_index: bool = Field(
        default=True,
        description="Enable full-text search index for hybrid search",
    )
    default_hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default balance between vector (1.0) and keyword (0.0) search",
    )

    # Performance
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts for transient errors",
    )
    retry_backoff_seconds: float = Field(
        default=0.5,
        ge=0.1,
        description="Initial backoff time for retries (doubles each attempt)",
    )
    batch_size: int = Field(
        default=1000,
        ge=100,
        description="Batch size for large operations",
    )
    compaction_threshold: int = Field(
        default=10,
        ge=1,
        description="Number of small fragments before auto-compaction",
    )

    # Connection Pool
    connection_pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connections in the pool (LRU eviction)",
    )

    # Read Consistency
    read_consistency_interval_ms: int = Field(
        default=0,
        ge=0,
        description="Interval for read consistency checks (0 = strong consistency)",
    )

    # Index Management
    index_wait_timeout_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout for waiting on index creation",
    )

    # UMAP
    umap_n_neighbors: int = Field(
        default=15,
        ge=2,
        description="UMAP neighborhood size",
    )
    umap_min_dist: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="UMAP minimum distance",
    )

    model_config = {
        "env_prefix": "SPATIAL_MEMORY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Settings singleton with dependency injection support
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the settings instance (lazy-loaded singleton).

    Returns:
        The Settings instance.

    Example:
        from spatial_memory.config import get_settings
        settings = get_settings()
        print(settings.memory_path)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(new_settings: Settings) -> None:
    """Override the settings instance (for testing).

    Args:
        new_settings: The new Settings instance to use.

    Example:
        from spatial_memory.config import override_settings, Settings
        test_settings = Settings(memory_path="/tmp/test")
        override_settings(test_settings)
    """
    global _settings
    _settings = new_settings


def reset_settings() -> None:
    """Reset settings to None (forces reload on next get_settings call)."""
    global _settings
    _settings = None


# Backwards compatibility - lazy property that calls get_settings()
class _SettingsProxy:
    """Proxy object for backwards compatibility with `settings` global."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return repr(get_settings())


settings = _SettingsProxy()
