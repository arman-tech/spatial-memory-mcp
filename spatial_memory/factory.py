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

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.adapters.project_detection import (
    ProjectDetectionConfig,
    ProjectDetector,
)
from spatial_memory.config import Settings
from spatial_memory.core.cache import ResponseCache
from spatial_memory.core.database import Database
from spatial_memory.core.embedding_registry import (
    EmbeddingModelConfig,
    resolve_model,
)
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.errors import ConfigurationError
from spatial_memory.core.metrics_journal import prune_journal, set_journal_dir
from spatial_memory.core.models import AutoDecayConfig
from spatial_memory.core.queue_constants import QUEUE_DIR_NAME
from spatial_memory.core.rate_limiter import AgentAwareRateLimiter, RateLimiter
from spatial_memory.ports.rate_limiting import (
    AsyncAgentRateLimiterPort,
    AsyncRateLimiterPort,
)
from spatial_memory.ports.repositories import (
    EmbeddingServiceProtocol,
    MemoryRepositoryProtocol,
)
from spatial_memory.services.decay_manager import DecayManager
from spatial_memory.services.export_import import ExportImportConfig, ExportImportService
from spatial_memory.services.ingest_pipeline import IngestPipeline
from spatial_memory.services.lifecycle import LifecycleConfig, LifecycleService
from spatial_memory.services.memory import MemoryService
from spatial_memory.services.queue_processor import QueueProcessor
from spatial_memory.services.similarity import CrossCorpusSimilarityService
from spatial_memory.services.spatial import SpatialConfig, SpatialService
from spatial_memory.services.utility import UtilityConfig, UtilityService

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
    project_detector: ProjectDetector
    queue_processor: QueueProcessor | None
    decay_manager: DecayManager | None
    rate_limiter: AsyncRateLimiterPort | None
    agent_rate_limiter: AsyncAgentRateLimiterPort | None
    similarity: CrossCorpusSimilarityService
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

        # Pin the metrics-journal directory to settings.memory_path / 'metrics'
        # so observability writes follow Settings, not the env-var fallback.
        # Idempotent — safe even if EmbeddingService.from_config also calls it.
        set_journal_dir(self._settings.memory_path / "metrics")
        prune_journal()

    def create_embedding_service(self) -> EmbeddingService:
        """Create the embedding service.

        Returns:
            Configured EmbeddingService instance.
        """
        return EmbeddingService(
            model_name=self._settings.embedding_model,
            backend=self._settings.embedding_backend,  # type: ignore[arg-type]
        )

    def _validate_embedding_model(
        self,
        database: Database,
        configured: EmbeddingModelConfig,
    ) -> None:
        """Hard-fail mismatch guard: configured model vs stored DB metadata.

        Implements 4 of the 5 cases from spec §6.3 (case 2 — v1.x rejection —
        is handled at connect time by ``Database._check_schema_refusal`` per
        D5 clean break, so the factory only worries about cross-version v2
        mismatches):

        1. Fresh DB (no stored alias) -> return.
        3. v2.x match (stored alias == configured.alias) -> return.
        4. Same-dim mismatch -> ConfigurationError §8.2.
        5. Cross-dim mismatch -> ConfigurationError §8.3.

        When the stored alias is unknown to the v2.0 registry-of-one, the
        stored dim is read from the table schema via ``detect_vector_dim``.
        """
        stored_alias = database.get_system_metadata("embedding_model")

        if stored_alias is None:
            return

        if stored_alias == configured.alias:
            return

        try:
            stored_dim = resolve_model(stored_alias).dim
        except ValueError:
            stored_dim = database.detect_vector_dim()

        memory_path = self._settings.memory_path

        if stored_dim == configured.dim:
            raise ConfigurationError(
                f"Embedding model mismatch: configured={configured.alias}, "
                f"stored={stored_alias} (both {configured.dim}-dim).\n"
                f"Either set SPATIAL_MEMORY_EMBEDDING_MODEL back to the stored model, "
                f"or (in v2.1+) run:\n"
                f"  spatial-memory reembed --to={configured.alias}"
            )

        raise ConfigurationError(
            f"Embedding model mismatch: configured={configured.alias} "
            f"(dim={configured.dim}), stored={stored_alias} (dim={stored_dim}).\n"
            f"Dimension change requires export/import:\n"
            f"  1. Set SPATIAL_MEMORY_EMBEDDING_MODEL={stored_alias}\n"
            f"  2. spatial-memory export -o backup.json\n"
            f"  3. Set SPATIAL_MEMORY_EMBEDDING_MODEL={configured.alias}\n"
            f"  4. rm -rf {memory_path}\n"
            f"  5. Restart server (creates fresh DB)\n"
            f"  6. spatial-memory import -i backup.json"
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
            filelock_enabled=self._settings.filelock_enabled,
            filelock_timeout=self._settings.filelock_timeout,
            filelock_poll_interval=self._settings.filelock_poll_interval,
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

    def create_ingest_pipeline(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
    ) -> IngestPipeline:
        """Create the ingest pipeline.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.

        Returns:
            Configured IngestPipeline instance.
        """
        return IngestPipeline(
            repository=repository,
            embeddings=embeddings,
        )

    def create_memory_service(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        ingest_pipeline: IngestPipeline | None = None,
    ) -> MemoryService:
        """Create the memory service.

        Args:
            repository: Memory repository.
            embeddings: Embedding service.
            ingest_pipeline: Optional ingest pipeline for dedup/quality/store.

        Returns:
            Configured MemoryService instance.
        """
        return MemoryService(
            repository=repository,
            embeddings=embeddings,
            ingest_pipeline=ingest_pipeline,
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
            decay_function=self._settings.auto_decay_function,  # type: ignore[arg-type]
        )

        return DecayManager(repository=repository, config=config)

    def create_project_detector(self) -> ProjectDetector:
        """Create the project detector.

        Returns:
            Configured ProjectDetector instance.
        """
        config = ProjectDetectionConfig(
            explicit_project=self._settings.project,
        )
        return ProjectDetector(config=config)

    def create_queue_processor(
        self,
        memory_service: MemoryService,
        project_detector: ProjectDetector,
    ) -> QueueProcessor | None:
        """Create the queue processor if cognitive offloading is enabled.

        Args:
            memory_service: Memory service for storing memories.
            project_detector: Detector for resolving project identity.

        Returns:
            QueueProcessor if enabled, None otherwise.
        """
        if not self._settings.cognitive_offloading_enabled:
            return None

        queue_dir = self._settings.memory_path / QUEUE_DIR_NAME
        return QueueProcessor(
            memory_service=memory_service,
            project_detector=project_detector,
            queue_dir=queue_dir,
            poll_interval=self._settings.queue_poll_interval_seconds,
            dedup_threshold=self._settings.dedup_vector_threshold,
            signal_threshold=self._settings.signal_threshold,
            cognitive_offloading_enabled=True,
        )

    def create_similarity_service(
        self,
        repository: MemoryRepositoryProtocol,
    ) -> CrossCorpusSimilarityService:
        """Create the cross-corpus similarity service.

        Args:
            repository: Memory repository (provides both search and CRUD).

        Returns:
            Configured CrossCorpusSimilarityService instance.
        """
        from spatial_memory.core.models import SimilarityConfig
        from spatial_memory.core.scoring import get_scoring_strategy

        config = SimilarityConfig(
            default_min_similarity=self._settings.cross_corpus_min_similarity,
            max_batch_size=self._settings.cross_corpus_max_batch,
            scoring_strategy=self._settings.cross_corpus_scoring,
            content_weight=self._settings.cross_corpus_content_weight,
        )
        scoring = get_scoring_strategy(
            config.scoring_strategy,
            **(
                {"content_weight": config.content_weight}
                if config.scoring_strategy == "vector_content"
                else {}
            ),
        )
        return CrossCorpusSimilarityService(
            repository=repository,
            namespace_provider=repository,
            config=config,
            scoring_strategy=scoring,
            memory_repository=repository,
        )

    def create_all(self) -> ServiceContainer:
        """Create all services with proper dependency wiring.

        v2.0 ordering (registry-of-one — dim is always known up front):
          1. Resolve the configured model from the registry (no model load).
          2. Create + connect the database using the configured dim.
          3. Run ``_validate_embedding_model`` BEFORE building the rest of
             the services so dim/alias mismatches fail in <1s.
          4. Instantiate the EmbeddingService if not injected.
          5. Wire the rest of the services.
        """
        configured_cfg = resolve_model(self._settings.embedding_model)

        embeddings: EmbeddingServiceProtocol
        database: Database | None = None
        repository: MemoryRepositoryProtocol

        if self._injected_embeddings is not None:
            db_dim = self._injected_embeddings.dimensions
        else:
            db_dim = configured_cfg.dim

        if self._injected_repository is None:
            database = self.create_database(db_dim)
            self._validate_embedding_model(database, configured_cfg)
            repository = self.create_repository(database)
        else:
            repository = self._injected_repository

        if self._injected_embeddings is not None:
            embeddings = self._injected_embeddings
        else:
            embeddings = self.create_embedding_service()

        embedding_dim = embeddings.dimensions
        logger.info(f"Auto-detected embedding dimensions: {embedding_dim}")
        logger.info(f"Embedding backend: {embeddings.backend}")

        # Create services with shared dependencies
        ingest_pipeline = self.create_ingest_pipeline(repository, embeddings)
        ingest_pipeline.seed_from_repository(repository)
        memory = self.create_memory_service(repository, embeddings, ingest_pipeline)
        spatial = self.create_spatial_service(repository, embeddings)
        lifecycle = self.create_lifecycle_service(repository, embeddings)
        utility = self.create_utility_service(repository, embeddings)
        export_import = self.create_export_import_service(repository, embeddings)

        # Create cross-corpus similarity service
        similarity = self.create_similarity_service(repository)

        # Create project detector
        project_detector = self.create_project_detector()

        # Create queue processor
        queue_processor = self.create_queue_processor(memory, project_detector)

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
            similarity=similarity,
            project_detector=project_detector,
            queue_processor=queue_processor,
            decay_manager=decay_manager,
            rate_limiter=rate_limiter,
            agent_rate_limiter=agent_rate_limiter,
            cache=cache,
            per_agent_rate_limiting=per_agent_enabled,
            cache_enabled=cache_enabled,
            regions_cache_ttl=regions_cache_ttl,
        )
