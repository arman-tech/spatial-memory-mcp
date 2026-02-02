# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Medium severity architectural improvements (MED-ARCH-001 through MED-ARCH-004)
- Migration system (MED-DB-005)

## [1.6.1] - 2026-02-02

### Added
- `spatial-memory instructions` CLI command to view the MCP instructions injected into Claude's context

### Changed
- Updated CLAUDE.md to be contributor documentation (removed stale claude-memory references)

## [1.6.0] - 2026-02-02

### Fixed

#### Critical Issues
- **CRIT-001**: Fixed consolidation data loss window - implemented add-before-delete pattern with pending status marker
- **CRIT-002**: Fixed journey N+1 query pattern - plumbed `include_vector` parameter through search pipeline

#### High Severity Issues
- **HIGH-001**: Fixed sequential DB calls in wander (resolved via CRIT-002)
- **HIGH-002**: Fixed O(n²) similarity calculation in visualize - replaced with `scipy.spatial.distance.cdist` vectorized operation
- **HIGH-003**: Fixed inefficient batch search - implemented native LanceDB batch search
- **HIGH-004**: Fixed duplicate embedding generation in extract - batch embed upfront with `embed_batch()`
- **HIGH-005**: Fixed sequential updates in decay - added `update_batch()` method
- **HIGH-006**: Fixed sequential fetches in reinforce - added `get_batch()` method
- **HIGH-007**: Fixed non-atomic batch insert - added `PartialBatchInsertError` with rollback capability
- **HIGH-008**: Fixed rename_namespace lacking rollback - track renamed IDs with `_rollback_namespace_rename()`
- **HIGH-009**: Fixed consolidation loading entire namespace - added streaming `_consolidate_chunked()`
- **HIGH-010**: Fixed file lock on NFS/SMB - added detection and startup warning

#### Medium Severity Issues
- **MED-SEC-002**: Fixed memory exhaustion in JSON/CSV export - implemented streaming export
- **MED-CQ-001**: Fixed `forget_batch` returning incorrect IDs
- **MED-CQ-002**: Added missing `backend` to `EmbeddingServiceProtocol`
- **MED-CQ-003**: Fixed rate limiter token consumption issue
- **MED-CQ-004**: Fixed embedding cache leak in batch operations
- **MED-DB-001**: Added missing index to idempotency table
- **MED-DB-003**: Fixed `get_namespace_stats()` loading all records
- **MED-DB-004**: Added vector dimension validation at insert

#### Low Severity Issues
- **LOW-001**: Made embedding cache size configurable (was hardcoded 1000)
- **LOW-002**: Standardized datetime handling in lifecycle.py
- **LOW-003**: Extracted magic numbers in retry logic to constants
- **LOW-004**: Added Literal type validation for `index_type`
- **LOW-006**: Fixed `validate_namespace` docstring mismatch
- **LOW-007**: Normalized mock embedding vectors in conftest.py
- **LOW-008**: Documented content validation security approach
- **LOW-009**: Added metadata depth validation
- **LOW-010**: Sanitized paths in error messages
- **LOW-011**: Added proactive connection health check
- **LOW-012**: Fixed `close()` to remove from connection pool
- **LOW-013**: Implemented auto-compaction scheduling
- **LOW-014**: Cached stop words set (was recreated per call)
- **LOW-015**: Improved cache key hashing

### Added
- `get_batch()` method for bulk memory retrieval
- `update_batch()` method returning `(success_count, failed_ids)` tuple
- Native LanceDB batch vector search
- `PartialBatchInsertError` exception with `succeeded_ids` for recovery
- `atomic=True` parameter to `insert_batch()` with rollback capability
- Streaming consolidation with configurable `consolidate_chunk_size`
- NFS/SMB filesystem detection in `core/filesystem.py`
- Config option `acknowledge_network_filesystem_risk` to suppress warnings

### Changed
- Refactored `database.py` into separate manager modules for maintainability
- Improved type safety across async operations

### Closed (By Design)
- **MED-SEC-001**: No namespace-level authorization - single-developer local use is intended
- **MED-DB-002**: `get_namespaces()` loads all values - TTL caching is sufficient
- **LOW-005**: Duplicate local imports - intentional circular import avoidance

## [0.1.0] - 2026-01-20

### Added

#### Configuration System
- Pydantic-based settings with environment variable support
- Dependency injection pattern for testability
- Full configuration validation with bounds checking
- Support for `.env` files

#### Database Layer
- LanceDB integration for vector storage
- SQL injection prevention with pattern detection
- UUID validation for memory IDs
- Namespace format validation
- Atomic updates with rollback support

#### Embedding Service
- Local embedding support via sentence-transformers
- OpenAI API embedding support
- Dual-backend architecture with automatic routing
- Model: `all-MiniLM-L6-v2` (384 dimensions) as default

#### Data Models
- `Memory` - Core memory representation
- `MemoryResult` - Search result with similarity score
- `Filter` / `FilterGroup` - Query filtering system
- `ClusterInfo` - Cluster metadata for regions
- `JourneyStep` - Step in journey interpolation
- `VisualizationData` - Visualization output format

#### Error Handling
- Custom exception hierarchy
- `SpatialMemoryError` base class
- Specific errors: `MemoryNotFoundError`, `NamespaceNotFoundError`, `EmbeddingError`, `StorageError`, `ValidationError`, `ConfigurationError`, `ClusteringError`, `VisualizationError`

#### Testing
- 71 unit tests covering all Phase 1 components
- Pytest fixtures for isolated testing
- Mock embedding service for fast tests

#### Documentation
- README with project overview and roadmap
- Architecture diagrams (Mermaid)
- Security documentation
- Contributing guidelines
- Configuration reference (`.env.example`)

### Security
- Input validation on all user-provided data
- SQL injection prevention
- Namespace isolation
- Sanitized error messages
