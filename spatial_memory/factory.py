"""Service factory for dependency injection and initialization.

This module provides a factory pattern for creating and wiring all services
used by the SpatialMemoryServer. It centralizes configuration and dependency
injection, making the server initialization cleaner and services more testable.

Usage:
    from spatial_memory.factory import ServiceFactory

    factory = ServiceFactory(settings)
    services = factory.create_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.config import Settings
from spatial_memory.core.cache import ResponseCache
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.models import AutoDecayConfig
from spatial_memory.core.rate_limiter import AgentAwareRateLimiter, RateLimiter
from spatial_memory.services.decay_manager import DecayManager
from spatial_memory.services.export_import import ExportImportConfig, ExportImportService
from spatial_memory.services.lifecycle import LifecycleConfig, LifecycleService
from spatial_memory.services.memory import MemoryService
from spatial_memory.services.spatial import SpatialConfig, SpatialService
from spatial_memory.services.utility import UtilityConfig, UtilityService

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """Container for all initialized services.

    Provides access to all service instances created by the factory.
    This allows the server to access services through a single container
    rather than managing individual references.

    Attributes:
        embeddings: Embedding service for vector generation.
        database: Database connection and operations.
        repository: Memory repository for CRUD operations.
        memory: Memory service for remember/recall operations.
        spatial: Spatial service for exploration operations.
        lifecycle: Lifecycle service for decay/reinforce/consolidate.
        utility: Utility service for stats/namespaces/hybrid search.
        export_import: Export/import service for data portability.
        decay_manager: Automatic decay manager for real-time importance decay.
        rate_limiter: Simple rate limiter (if per-agent disabled).
        agent_rate_limiter: Per-agent rate limiter (if enabled).
        cache: Response cache for read operations.
        per_agent_rate_limiting: Whether per-agent rate limiting is enabled.
        cache_enabled: Whether response caching is enabled.
        regions_cache_ttl: TTL for regions cache entries.
    """

    embeddings: EmbeddingServiceProtocol
    database: Database | None
    repository: MemoryRepositoryProtocol
    memory: MemoryService
    spatial: SpatialService
    lifecycle: LifecycleService
    utility: UtilityService
    export_import: ExportImportService
    decay_manager: DecayManager | None
    rate_limiter: RateLimiter | None
    agent_rate_limiter: AgentAwareRateLimiter | None
    cache: ResponseCache | None
    per_agent_rate_limiting: bool
    cache_enabled: bool
    regions_cache_ttl: float


class ServiceFactory:
    """Factory for creating and wiring all services.

    Centralizes service creation with proper dependency injection.
    This simplifies server initialization and improves testability.

    Example:
        factory = ServiceFactory(settings)
        services = factory.create_all()
        # Use services.memory, services.spatial, etc.
    """

    def __init__(
        self,
        settings: Settings,
        repository: MemoryRepositoryProtocol | None = None,
        embeddings: EmbeddingServiceProtocol | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            settings: Application settings.
            repository: Optional repository override for testing.
            embeddings: Optional embeddings override for testing.
        """
        self._settings = settings
        self._injected_repository = repository
        self._injected_embeddings = embeddings

    def create_embedding_service(self) -> EmbeddingService:
        """Create the embedding service.

        Returns:
            Configured EmbeddingService instance.
        """
        return EmbeddingService(
            model_name=self._settings.embedding_model,
            openai_api_key=self._settings.openai_api_key,
            backend=self._settings.embedding_backend,  # type: ignore[arg-type]
        )

    def create_database(self, embedding_dim: int) -> Database:
        """Create the database connection.

        Args:
            embedding_dim: Dimension of embedding vectors.

        Returns:
            Configured Database instance.
        """
        db = Database(
            storage_path=self._settings.memory_path,
            embedding_dim=embedding_dim,
            auto_create_indexes=self._settings.auto_create_indexes,
            vector_index_threshold=self._settings.vector_index_threshold,
            enable_fts=self._settings.enable_fts_index,
            index_nprobes=self._settings.index_nprobes,
            index_refine_factor=self._settings.index_refine_factor,
            max_retry_attempts=self._settings.max_retry_attempts,
            retry_backoff_seconds=self._settings.retry_backoff_seconds,
            read_consistency_interval_ms=self._settings.read_consistency_interval_ms,
            index_wait_timeout_seconds=self._settings.index_wait_timeout_seconds,
            fts_stem=self._settings.fts_stem,
            fts_remove_stop_words=self._settings.fts_remove_stop_words,
            fts_language=self._settings.fts_language,
            index_type=self._settings.index_type,
            hnsw_m=self._settings.hnsw_m,
            hnsw_ef_construction=self._settings.hnsw_ef_construction,
            enable_memory_expiration=self._settings.enable_memory_expiration,
            default_memory_ttl_days=self._settings.default_memory_ttl_days,
            acknowledge_network_filesystem_risk=self._settings.acknowledge_network_filesystem_risk,
        )
        db.connect()
        return db

    def create_repository(self, database: Database) -> LanceDBMemoryRepository:
        """Create the memory repository.

        Args:
            database: Database instance.

        Returns:
            LanceDBMemoryRepository instance.
        """
        return LanceDBMemoryRepository(database)

    def create_memory_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> MemoryService:
        """Create the memory service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured MemoryService instance.
        """
        return MemoryService(
            repository=repository,
            embeddings=embeddings,
        )

    def create_spatial_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> SpatialService:
        """Create the spatial service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured SpatialService instance.
        """
        return SpatialService(
            repository=repository,
            embeddings=embeddings,
            config=SpatialConfig(
                journey_max_steps=self._settings.max_journey_steps,
                wander_max_steps=self._settings.max_wander_steps,
                regions_max_memories=self._settings.regions_max_memories,
                visualize_max_memories=self._settings.max_visualize_memories,
                visualize_n_neighbors=self._settings.umap_n_neighbors,
                visualize_min_dist=self._settings.umap_min_dist,
                visualize_similarity_threshold=self._settings.visualize_similarity_threshold,
            ),
        )

    def create_lifecycle_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> LifecycleService:
        """Create the lifecycle service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured LifecycleService instance.
        """
        return LifecycleService(
            repository=repository,
            embeddings=embeddings,
            config=LifecycleConfig(
                decay_default_half_life_days=self._settings.decay_default_half_life_days,
                decay_default_function=self._settings.decay_default_function,
                decay_min_importance_floor=self._settings.decay_min_importance_floor,
                decay_batch_size=self._settings.decay_batch_size,
                reinforce_default_boost=self._settings.reinforce_default_boost,
                reinforce_max_importance=self._settings.reinforce_max_importance,
                extract_max_text_length=self._settings.extract_max_text_length,
                extract_max_candidates=self._settings.extract_max_candidates,
                extract_default_importance=self._settings.extract_default_importance,
                extract_default_namespace=self._settings.extract_default_namespace,
                consolidate_min_threshold=self._settings.consolidate_min_threshold,
                consolidate_content_weight=self._settings.consolidate_content_weight,
                consolidate_max_batch=self._settings.consolidate_max_batch,
            ),
        )

    def create_utility_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> UtilityService:
        """Create the utility service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured UtilityService instance.
        """
        return UtilityService(
            repository=repository,
            embeddings=embeddings,
            config=UtilityConfig(
                hybrid_default_alpha=self._settings.hybrid_default_alpha,
                hybrid_min_alpha=self._settings.hybrid_min_alpha,
                hybrid_max_alpha=self._settings.hybrid_max_alpha,
                stats_include_index_details=True,
                namespace_batch_size=self._settings.namespace_batch_size,
                delete_namespace_require_confirmation=self._settings.destructive_require_namespace_confirmation,
            ),
        )

    def create_export_import_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> ExportImportService:
        """Create the export/import service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured ExportImportService instance.
        """
        return ExportImportService(
            repository=repository,
            embeddings=embeddings,
            config=ExportImportConfig(
                default_export_format=self._settings.export_default_format,
                export_batch_size=self._settings.export_batch_size,
                import_batch_size=self._settings.import_batch_size,
                import_deduplicate=self._settings.import_deduplicate_default,
                import_dedup_threshold=self._settings.import_dedup_threshold,
                validate_on_import=self._settings.import_validate_vectors,
                parquet_compression="zstd",
                max_import_records=self._settings.import_max_records,
                csv_include_vectors=self._settings.csv_include_vectors,
                max_export_records=self._settings.max_export_records,
            ),
            allowed_export_paths=self._settings.export_allowed_paths,
            allowed_import_paths=self._settings.import_allowed_paths,
            allow_symlinks=self._settings.export_allow_symlinks,
            max_import_size_bytes=int(self._settings.import_max_file_size_mb * 1024 * 1024),
        )

    def create_rate_limiter(self) -> tuple[RateLimiter | None, AgentAwareRateLimiter | None, bool]:
        """Create rate limiter based on settings.

        Returns:
            Tuple of (simple_limiter, agent_limiter, per_agent_enabled).
        """
        per_agent = self._settings.rate_limit_per_agent_enabled
        if per_agent:
            return (
                None,
                AgentAwareRateLimiter(
                    global_rate=self._settings.embedding_rate_limit,
                    per_agent_rate=self._settings.rate_limit_per_agent_rate,
                    max_agents=self._settings.rate_limit_max_tracked_agents,
                ),
                True,
            )
        return (
            RateLimiter(
                rate=self._settings.embedding_rate_limit,
                capacity=int(self._settings.embedding_rate_limit * 2),
            ),
            None,
            False,
        )

    def create_cache(self) -> tuple[ResponseCache | None, bool, float]:
        """Create response cache based on settings.

        Returns:
            Tuple of (cache, enabled, regions_ttl).
        """
        if not self._settings.response_cache_enabled:
            return None, False, 0.0
        return (
            ResponseCache(
                max_size=self._settings.response_cache_max_size,
                default_ttl=self._settings.response_cache_default_ttl,
            ),
            True,
            self._settings.response_cache_regions_ttl,
        )

    def create_decay_manager(
        self,
        repository: MemoryRepositoryProtocol,
    ) -> DecayManager | None:
        """Create the decay manager based on settings.

        Args:
            repository: Repository for persisting decay updates.

        Returns:
            DecayManager if auto-decay is enabled, None otherwise.
        """
        if not self._settings.auto_decay_enabled:
            return None

        config = AutoDecayConfig(
            enabled=self._settings.auto_decay_enabled,
            persist_enabled=self._settings.auto_decay_persist_enabled,
            persist_batch_size=self._settings.auto_decay_persist_batch_size,
            persist_flush_interval_seconds=self._settings.auto_decay_persist_flush_interval_seconds,
            min_change_threshold=self._settings.auto_decay_min_change_threshold,
            max_queue_size=self._settings.auto_decay_max_queue_size,
            half_life_days=self._settings.decay_default_half_life_days,
            min_importance_floor=self._settings.decay_min_importance_floor,
            access_weight=0.3,  # Default access weight
        )

        return DecayManager(repository=repository, config=config)

    def create_all(self) -> ServiceContainer:
        """Create all services with proper dependency wiring.

        Returns:
            ServiceContainer with all services initialized.
        """
        # Use injected dependencies or create new ones
        if self._injected_embeddings is None:
            embeddings = self.create_embedding_service()
        else:
            embeddings = self._injected_embeddings

        # Auto-detect embedding dimensions
        embedding_dim = embeddings.dimensions
        logger.info(f"Auto-detected embedding dimensions: {embedding_dim}")
        logger.info(f"Embedding backend: {embeddings.backend}")

        # Create database and repository
        database: Database | None = None
        if self._injected_repository is None:
            database = self.create_database(embedding_dim)
            repository = self.create_repository(database)
        else:
            repository = self._injected_repository

        # Create services with shared dependencies
        memory = self.create_memory_service(repository, embeddings)
        spatial = self.create_spatial_service(repository, embeddings)
        lifecycle = self.create_lifecycle_service(repository, embeddings)
        utility = self.create_utility_service(repository, embeddings)
        export_import = self.create_export_import_service(repository, embeddings)

        # Create decay manager
        decay_manager = self.create_decay_manager(repository)

        # Create rate limiter
        rate_limiter, agent_rate_limiter, per_agent_enabled = self.create_rate_limiter()

        # Create cache
        cache, cache_enabled, regions_cache_ttl = self.create_cache()

        return ServiceContainer(
            embeddings=embeddings,
            database=database,
            repository=repository,
            memory=memory,
            spatial=spatial,
            lifecycle=lifecycle,
            utility=utility,
            export_import=export_import,
            decay_manager=decay_manager,
            rate_limiter=rate_limiter,
            agent_rate_limiter=agent_rate_limiter,
            cache=cache,
            per_agent_rate_limiting=per_agent_enabled,
            cache_enabled=cache_enabled,
            regions_cache_ttl=regions_cache_ttl,
        )
