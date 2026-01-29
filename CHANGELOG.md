# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: Core operations (`remember`, `recall`, `nearby`, `forget`)
- Phase 3: Spatial operations (`journey`, `wander`, `regions`, `visualize`)
- Phase 4: Lifecycle operations (`consolidate`, `extract`, `decay`, `reinforce`)
- Phase 5: Utility operations (`stats`, `namespaces`, `export`, `import`)
- Phase 6: Integration tests, documentation, PyPI release

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
