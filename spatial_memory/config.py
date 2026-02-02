"""Configuration system for Spatial Memory MCP Server."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from spatial_memory.core.errors import ConfigurationError

# Re-export for backward compatibility
__all__ = [
    "Settings",
    "ConfigurationError",
    "get_settings",
    "override_settings",
    "reset_settings",
    "validate_startup",
]


class Settings(BaseSettings):
    """Spatial Memory Server Configuration."""

    # Storage
    memory_path: Path = Field(
        default=Path("./.spatial-memory"),
        description="Path to LanceDB storage directory",
    )
    acknowledge_network_filesystem_risk: bool = Field(
        default=False,
        description=(
            "Set to True to suppress warnings about network filesystem usage. "
            "File-based locking does not work reliably on NFS/SMB/CIFS. "
            "Only set this if you are certain only one instance will access the storage."
        ),
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
    embedding_backend: str = Field(
        default="auto",
        description="Embedding backend: 'auto', 'onnx', or 'pytorch'",
    )

    # OpenAI (optional)
    openai_api_key: SecretStr | None = Field(
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
    log_format: str = Field(
        default="text",
        description="Log format: 'text' or 'json'",
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
    regions_max_memories: int = Field(
        default=1000,
        description="Maximum memories to consider for region clustering",
    )
    visualize_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity to show edges in visualization",
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
    index_type: Literal["IVF_PQ", "IVF_FLAT", "HNSW_SQ"] = Field(
        default="IVF_PQ",
        description="Vector index type: IVF_PQ, IVF_FLAT, or HNSW_SQ",
    )
    hnsw_m: int = Field(
        default=20,
        ge=4,
        le=64,
        description="HNSW connections per node",
    )
    hnsw_ef_construction: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="HNSW build-time search width",
    )

    # Hybrid Search
    enable_fts_index: bool = Field(
        default=True,
        description="Enable full-text search index for hybrid search",
    )

    # FTS Configuration
    fts_stem: bool = Field(
        default=True,
        description="Enable stemming in FTS (running -> run)",
    )
    fts_remove_stop_words: bool = Field(
        default=True,
        description="Remove stop words in FTS (the, is, etc.)",
    )
    fts_language: str = Field(
        default="English",
        description="Language for FTS stemming",
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

    # Cross-Process Locking
    filelock_enabled: bool = Field(
        default=True,
        description="Enable cross-process file locking for multi-instance safety",
    )
    filelock_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for acquiring filelock",
    )
    filelock_poll_interval: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Interval between lock acquisition attempts",
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

    # TTL Configuration
    enable_memory_expiration: bool = Field(
        default=False,
        description="Enable automatic memory expiration",
    )
    default_memory_ttl_days: int | None = Field(
        default=None,
        description="Default TTL for memories in days (None = no expiration)",
    )

    # Rate Limiting
    embedding_rate_limit: float = Field(
        default=100.0,
        ge=1.0,
        description="Maximum embedding operations per second",
    )
    batch_rate_limit: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum batch operations per second",
    )

    # =========================================================================
    # Lifecycle Settings
    # =========================================================================

    # Decay Settings
    decay_default_half_life_days: float = Field(
        default=30.0,
        ge=1.0,
        le=365.0,
        description="Default half-life for exponential decay",
    )
    decay_default_function: str = Field(
        default="exponential",
        description="Default decay function (exponential, linear, step)",
    )
    decay_min_importance_floor: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Minimum importance after decay",
    )
    decay_batch_size: int = Field(
        default=500,
        ge=100,
        description="Batch size for decay updates",
    )

    # Reinforcement Settings
    reinforce_default_boost: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Default boost amount for reinforcement",
    )
    reinforce_max_importance: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum importance after reinforcement",
    )

    # Extraction Settings
    extract_max_text_length: int = Field(
        default=50000,
        ge=1000,
        description="Maximum text length for extraction",
    )
    extract_max_candidates: int = Field(
        default=20,
        ge=1,
        description="Maximum candidates per extraction",
    )
    extract_default_importance: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Default importance for extracted memories",
    )
    extract_default_namespace: str = Field(
        default="extracted",
        description="Default namespace for extracted memories",
    )

    # Consolidation Settings
    consolidate_min_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=0.99,
        description="Minimum similarity threshold for consolidation",
    )
    consolidate_content_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of content overlap vs vector similarity",
    )
    consolidate_max_batch: int = Field(
        default=1000,
        ge=100,
        description="Maximum memories per consolidation pass",
    )

    # =========================================================================
    # Phase 5: Utility Settings
    # =========================================================================

    hybrid_default_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default alpha for hybrid search (1.0=vector, 0.0=keyword)",
    )
    namespace_batch_size: int = Field(
        default=1000,
        ge=100,
        description="Batch size for namespace operations",
    )

    # =========================================================================
    # Phase 5: File Security Settings
    # =========================================================================

    # Export Settings
    export_allowed_paths: list[str] = Field(
        default_factory=lambda: ["./exports", "./backups"],
        description="Directories where exports are allowed (relative to memory_path)",
    )
    export_allow_symlinks: bool = Field(
        default=False,
        description="Allow following symlinks in export paths",
    )

    # Import Settings
    import_allowed_paths: list[str] = Field(
        default_factory=lambda: ["./imports", "./backups"],
        description="Directories where imports are allowed (relative to memory_path)",
    )
    import_allow_symlinks: bool = Field(
        default=False,
        description="Allow following symlinks in import paths",
    )
    import_max_file_size_mb: float = Field(
        default=100.0,
        ge=1.0,
        le=1000.0,
        description="Maximum import file size in megabytes",
    )
    import_max_records: int = Field(
        default=100_000,
        ge=1000,
        le=10_000_000,
        description="Maximum records per import operation",
    )
    import_fail_fast: bool = Field(
        default=False,
        description="Stop import on first validation error",
    )
    import_validate_vectors: bool = Field(
        default=True,
        description="Validate vector dimensions match embedding model",
    )

    # =========================================================================
    # Phase 5: Destructive Operation Settings
    # =========================================================================

    destructive_confirm_threshold: int = Field(
        default=100,
        ge=1,
        description="Require confirmation for operations affecting more than N records",
    )
    destructive_require_namespace_confirmation: bool = Field(
        default=True,
        description="Require explicit namespace confirmation for delete_namespace",
    )

    # =========================================================================
    # Phase 5: Export/Import Operational Settings
    # =========================================================================

    export_default_format: str = Field(
        default="parquet",
        description="Default export format (parquet, json, csv)",
    )
    export_batch_size: int = Field(
        default=5000,
        ge=100,
        description="Records per batch during export",
    )
    import_batch_size: int = Field(
        default=1000,
        ge=100,
        description="Records per batch during import",
    )
    import_deduplicate_default: bool = Field(
        default=False,
        description="Deduplicate imports by default",
    )
    import_dedup_threshold: float = Field(
        default=0.95,
        ge=0.7,
        le=0.99,
        description="Similarity threshold for import deduplication",
    )

    # CSV Export
    csv_include_vectors: bool = Field(
        default=False,
        description="Include vector embeddings in CSV exports (large files)",
    )

    # Export Limits
    max_export_records: int = Field(
        default=1_000_000,
        ge=1000,
        le=10_000_000,
        description="Maximum records per export operation",
    )

    # Hybrid Search Bounds
    hybrid_min_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum alpha for hybrid search (0.0=pure keyword)",
    )
    hybrid_max_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum alpha for hybrid search (1.0=pure vector)",
    )

    # =========================================================================
    # v1.5.3 Phase 1: Observability Settings
    # =========================================================================

    include_request_meta: bool = Field(
        default=False,
        description="Include _meta object in responses (request_id, timing, etc.)",
    )
    log_include_trace_context: bool = Field(
        default=True,
        description="Add [req=][agent=] trace context to log messages",
    )
    include_timing_breakdown: bool = Field(
        default=False,
        description="Include timing_ms breakdown in _meta (requires include_request_meta)",
    )

    # =========================================================================
    # v1.5.3 Phase 2: Efficiency Settings
    # =========================================================================

    warm_up_on_start: bool = Field(
        default=True,
        description="Pre-load embedding model on startup for faster first request",
    )
    response_cache_enabled: bool = Field(
        default=True,
        description="Enable response caching for idempotent operations",
    )
    response_cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum number of cached responses (LRU eviction)",
    )
    response_cache_default_ttl: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Default TTL in seconds for cached responses",
    )
    response_cache_regions_ttl: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="TTL in seconds for regions() responses (expensive operation)",
    )
    idempotency_enabled: bool = Field(
        default=True,
        description="Enable idempotency key support for write operations",
    )
    idempotency_key_ttl_hours: float = Field(
        default=24.0,
        ge=1.0,
        le=168.0,
        description="Hours to remember idempotency keys (max 7 days)",
    )

    # =========================================================================
    # v1.5.3 Phase 3: Resilience Settings
    # =========================================================================

    rate_limit_per_agent_enabled: bool = Field(
        default=True,
        description="Enable per-agent rate limiting",
    )
    rate_limit_per_agent_rate: float = Field(
        default=25.0,
        ge=1.0,
        le=1000.0,
        description="Maximum operations per second per agent",
    )
    rate_limit_max_tracked_agents: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum number of agents to track for rate limiting (LRU eviction)",
    )
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for external dependencies",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of consecutive failures before circuit opens",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Seconds to wait before attempting half-open state",
    )
    backpressure_queue_enabled: bool = Field(
        default=False,
        description="Enable backpressure queue for overload protection (future)",
    )
    backpressure_queue_max_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Maximum queue depth when backpressure is enabled",
    )

    # =========================================================================
    # v1.6.3: Auto-Decay Settings
    # =========================================================================

    auto_decay_enabled: bool = Field(
        default=True,
        description="Enable automatic decay calculation during recall operations",
    )
    auto_decay_persist_enabled: bool = Field(
        default=True,
        description="Persist decay updates to database (disable for read-only scenarios)",
    )
    auto_decay_persist_batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Batch size for persisting decay updates to database",
    )
    auto_decay_persist_flush_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Interval between background flush operations for decay updates",
    )
    auto_decay_min_change_threshold: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Minimum importance change to trigger database persistence (1% default)",
    )
    auto_decay_max_queue_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum queue size for pending decay updates (backpressure control)",
    )
    auto_decay_function: str = Field(
        default="exponential",
        description="Decay function for auto-decay: exponential, linear, or step",
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


def validate_startup(settings: Settings) -> list[str]:
    """Validate settings at startup.

    Args:
        settings: The settings to validate.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        ConfigurationError: For fatal configuration issues.
    """
    warnings = []

    # 1. Validate OpenAI key when using OpenAI embeddings
    has_openai_key = (
        settings.openai_api_key is not None
        and settings.openai_api_key.get_secret_value() != ""
    )
    if settings.embedding_model.startswith("openai:") and not has_openai_key:
        raise ConfigurationError(
            "OpenAI API key required when using OpenAI embeddings. "
            "Set SPATIAL_MEMORY_OPENAI_API_KEY environment variable."
        )

    # 2. Validate storage path exists or can be created
    try:
        settings.memory_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ConfigurationError(f"Cannot create storage path: {settings.memory_path}: {e}")

    # 3. Check storage path is writable
    test_file = settings.memory_path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        raise ConfigurationError(f"Storage path not writable: {settings.memory_path}: {e}")

    # 4. Validate embedding_backend setting
    valid_backends = ("auto", "onnx", "pytorch")
    if settings.embedding_backend not in valid_backends:
        raise ConfigurationError(
            f"Invalid embedding_backend: '{settings.embedding_backend}'. "
            f"Must be one of: {', '.join(valid_backends)}"
        )

    # 5. Check ONNX availability if explicitly requested
    if settings.embedding_backend == "onnx":
        try:
            import onnxruntime  # noqa: F401
            import optimum.onnxruntime  # noqa: F401
        except ImportError:
            raise ConfigurationError(
                "ONNX Runtime requested but not fully installed. "
                "Install with: pip install sentence-transformers[onnx]"
            )

    # 6. Warn on suboptimal settings
    if settings.index_nprobes < 10:
        warnings.append(
            f"index_nprobes={settings.index_nprobes} is low; consider 20+ for better recall"
        )

    if settings.max_retry_attempts < 2:
        warnings.append(
            "max_retry_attempts < 2 may cause failures on transient errors"
        )

    return warnings
