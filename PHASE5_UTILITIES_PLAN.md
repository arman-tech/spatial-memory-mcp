# Phase 5 Implementation Plan: Utility Operations

## Executive Summary

This plan details the implementation of Phase 5 Utility Operations for the Spatial Memory MCP Server. Based on comprehensive analysis by specialized agents (LanceDB Architect, Threat Modeler, Tech Lead Advisor, Pytest Master), this plan provides complete specifications for 7 new MCP tools.

**Current State:** Phases 1-4 + Lifecycle complete with 541 tests passing, 13 of 17 tools implemented
**Phase 5 Goal:** Add utility operations for database management, data portability, and enhanced search

**Architecture Review (2026-01-30):** Plan updated based on Clean Architecture and MCP standards review to ensure proper separation of concerns and protocol compliance.

---

## Clean Architecture Compliance

This implementation follows the established Clean Architecture (Ports & Adapters) pattern:

```
MCP Server (server.py)           ← Presentation Layer (tool handlers)
       │
       ▼
Services (services/)             ← Application Layer (business logic)
  - UtilityService               - stats, namespaces, delete/rename, hybrid_recall
  - ExportImportService          - export_memories, import_memories
       │
       ▼
Protocols (ports/)               ← Ports Layer (interfaces)
  - MemoryRepositoryProtocol     - data access contract
       │
       ▼
Adapters (adapters/)             ← Infrastructure Layer
  - LanceDBMemoryRepository      - implements protocol
       │
       ▼
Database (core/database.py)      ← Infrastructure Implementation
```

### Key Architectural Decisions

1. **Services depend on Protocols, not implementations** - Dependency injection via constructor
2. **File I/O is service-layer concern** - ExportImportService handles file operations; repository provides data streaming
3. **Security validation at service layer** - PathValidator invoked by services before any file operations
4. **Protocol defined BEFORE services** - Implementation order ensures contracts exist before use

### Boundary Responsibilities

| Layer | Responsibility | Phase 5 Examples |
|-------|---------------|------------------|
| **Service** | Business logic, orchestration, security validation | Format conversion, path validation, streaming coordination |
| **Repository** | Data access only | `get_all_for_export()` yields data, `bulk_import()` accepts records |
| **Database** | LanceDB operations | Arrow queries, merge_insert, index operations |

---

## MCP Standards Compliance

Following MCP specification (November 2025) requirements:

### Error Response Format

All tool errors MUST include `isError: true` flag per MCP standard:

```python
# In server.py call_tool handler
except SpatialMemoryError as e:
    return [TextContent(
        type="text",
        text=json.dumps({
            "error": type(e).__name__,
            "message": str(e),
            "isError": True,  # MCP REQUIRED
        }),
    )]
```

### Tool Output Schemas

Phase 5 tools include `outputSchema` for structured results:

```python
# Example: stats tool output schema
"outputSchema": {
    "type": "object",
    "properties": {
        "total_memories": {"type": "integer"},
        "memories_by_namespace": {"type": "object"},
        "storage_mb": {"type": "number"},
        "has_vector_index": {"type": "boolean"},
        "has_fts_index": {"type": "boolean"},
    },
    "required": ["total_memories", "storage_mb"],
}
```

### Tool Naming Convention

All Phase 5 tools follow MCP naming requirements:
- Length: 1-128 characters
- Characters: A-Z, a-z, 0-9, underscore, hyphen
- Unique within server

---

## Tool Summary

| Tool | Category | Description | Complexity |
|------|----------|-------------|------------|
| `stats` | Utility | Database statistics and health metrics | Medium |
| `namespaces` | Utility | List namespaces with memory counts | Low |
| `delete_namespace` | Utility | Delete all memories in a namespace | Medium |
| `rename_namespace` | Utility | Rename namespace (move all memories) | Medium |
| `export_memories` | Export/Import | Export to JSON/CSV/Parquet | High |
| `import_memories` | Export/Import | Import with validation | High |
| `hybrid_recall` | Search | Combined vector + FTS search | Medium |

---

## Architecture Decision: Two New Services

Based on Single Responsibility Principle and cohesion analysis:

| Service | Responsibility | Tools |
|---------|---------------|-------|
| **UtilityService** | Database stats, namespace management | stats, namespaces, delete_namespace, rename_namespace, hybrid_recall |
| **ExportImportService** | Data portability and backup | export_memories, import_memories |

### Service-Layer vs Repository-Layer Responsibilities

**ExportImportService (Service Layer):**
- Path security validation (PathValidator)
- Format detection and conversion (Parquet/JSON/CSV)
- File I/O operations (read/write files)
- Progress tracking and error aggregation
- Streaming coordination

**Repository (Data Layer):**
- `get_all_for_export()` - yields memory data in batches (no file I/O)
- `bulk_import()` - accepts record iterator (no file parsing)
- Pure data access operations

This separation ensures:
1. Repository remains testable without filesystem dependencies
2. Multiple export formats can be added without modifying repository
3. Security validation happens at appropriate layer (before data access)

### File Structure

```
spatial_memory/
  services/
    __init__.py              # Add exports
    utility.py               # NEW: UtilityService
    export_import.py         # NEW: ExportImportService
  core/
    errors.py                # Add new error types
    file_security.py         # NEW: Path validation module
    import_security.py       # NEW: Import data validation
```

---

## 1. New Error Types

Add to `spatial_memory/core/errors.py`:

```python
class ExportError(SpatialMemoryError):
    """Raised when memory export fails."""
    pass


class ImportError(SpatialMemoryError):
    """Raised when memory import fails."""
    pass


class NamespaceOperationError(SpatialMemoryError):
    """Raised when namespace operation fails."""
    pass


class PathSecurityError(SpatialMemoryError):
    """Raised when a file path violates security constraints.

    Examples:
        - Path traversal attempt (../)
        - Path outside allowed directories
        - Symlink to disallowed location
        - Invalid file extension
    """

    def __init__(
        self,
        path: str,
        violation_type: str,
        message: str | None = None,
    ) -> None:
        self.path = path
        self.violation_type = violation_type
        self.message = message or f"Path security violation ({violation_type}): {path}"
        super().__init__(self.message)


class FileSizeLimitError(SpatialMemoryError):
    """Raised when a file exceeds size limits."""

    def __init__(
        self,
        path: str,
        actual_size_bytes: int,
        max_size_bytes: int,
    ) -> None:
        self.path = path
        self.actual_size_bytes = actual_size_bytes
        self.max_size_bytes = max_size_bytes
        actual_mb = actual_size_bytes / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        super().__init__(
            f"File exceeds size limit: {path} is {actual_mb:.2f}MB "
            f"(max: {max_mb:.2f}MB)"
        )


class DimensionMismatchError(ValidationError):
    """Raised when imported vectors have wrong dimensions."""

    def __init__(
        self,
        expected_dim: int,
        actual_dim: int,
        record_index: int | None = None,
    ) -> None:
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.record_index = record_index
        location = f" at record {record_index}" if record_index is not None else ""
        super().__init__(
            f"Vector dimension mismatch{location}: expected {expected_dim}, "
            f"got {actual_dim}"
        )


class SchemaValidationError(ValidationError):
    """Raised when import data fails schema validation."""

    def __init__(
        self,
        field: str,
        error: str,
        record_index: int | None = None,
    ) -> None:
        self.field = field
        self.error = error
        self.record_index = record_index
        location = f" at record {record_index}" if record_index is not None else ""
        super().__init__(f"Schema validation failed for '{field}'{location}: {error}")


class ImportRecordLimitError(SpatialMemoryError):
    """Raised when import file contains too many records."""

    def __init__(
        self,
        actual_count: int,
        max_count: int,
    ) -> None:
        self.actual_count = actual_count
        self.max_count = max_count
        super().__init__(
            f"Import file contains {actual_count} records "
            f"(max: {max_count})"
        )
```

---

## 2. Configuration Additions

Add to `spatial_memory/config.py`:

```python
    # =========================================================================
    # File Security Settings (Phase 5)
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

    # Destructive Operation Settings
    destructive_confirm_threshold: int = Field(
        default=100,
        ge=1,
        description="Require confirmation for operations affecting more than N records",
    )
    destructive_require_namespace_confirmation: bool = Field(
        default=True,
        description="Require explicit namespace confirmation for delete_namespace",
    )

    # Export/Import Operational Settings
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
```

---

## 3. Result Dataclasses

### StatsResult

```python
@dataclass
class IndexInfo:
    """Information about a single index."""
    name: str
    index_type: str
    column: str
    num_indexed_rows: int
    status: str  # "ready", "building", "needs_update"


@dataclass
class StatsResult:
    """Result of database statistics query."""
    total_memories: int
    memories_by_namespace: dict[str, int]
    storage_bytes: int
    storage_mb: float
    estimated_vector_bytes: int
    has_vector_index: bool
    has_fts_index: bool
    indices: list[IndexInfo]
    num_fragments: int
    needs_compaction: bool
    table_version: int
    oldest_memory_date: datetime | None = None
    newest_memory_date: datetime | None = None
    avg_content_length: float | None = None
```

### NamespacesResult

```python
@dataclass
class NamespaceInfo:
    """Information about a single namespace."""
    name: str
    memory_count: int
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None


@dataclass
class NamespacesResult:
    """Result of namespace listing."""
    namespaces: list[NamespaceInfo]
    total_namespaces: int
    total_memories: int
```

### DeleteNamespaceResult

```python
@dataclass
class DeleteNamespaceResult:
    """Result of namespace deletion."""
    namespace: str
    memories_deleted: int
    success: bool
    message: str
    dry_run: bool = False
```

### RenameNamespaceResult

```python
@dataclass
class RenameNamespaceResult:
    """Result of namespace rename."""
    old_namespace: str
    new_namespace: str
    memories_renamed: int
    success: bool
    message: str
```

### ExportResult

```python
@dataclass
class ExportResult:
    """Result of memory export."""
    format: str  # parquet, json, csv
    output_path: str
    memories_exported: int
    file_size_bytes: int
    file_size_mb: float
    namespaces_included: list[str]
    duration_seconds: float
    compression: str | None = None
```

### ImportResult

```python
@dataclass
class ImportedMemory:
    """Information about a single imported memory."""
    id: str
    content_preview: str
    namespace: str
    was_deduplicated: bool = False
    original_id: str | None = None


@dataclass
class ImportValidationError:
    """A validation error during import."""
    row_number: int
    field: str
    error: str
    value: str | None = None


@dataclass
class ImportResult:
    """Result of memory import."""
    source_path: str
    format: str
    total_records_in_file: int
    memories_imported: int
    memories_skipped: int
    memories_failed: int
    validation_errors: list[ImportValidationError]
    namespace_override: str | None = None
    duration_seconds: float
    imported_memories: list[ImportedMemory] | None = None
```

### HybridRecallResult

```python
@dataclass
class HybridMemoryMatch:
    """A memory matched by hybrid search."""
    id: str
    content: str
    similarity: float
    namespace: str
    tags: list[str]
    importance: float
    created_at: datetime
    metadata: dict[str, Any]
    vector_score: float | None = None
    fts_score: float | None = None
    combined_score: float = 0.0


@dataclass
class HybridRecallResult:
    """Result of hybrid recall operation."""
    query: str
    alpha: float
    memories: list[HybridMemoryMatch]
    total: int
    search_type: str = "hybrid"
```

---

## 4. Service Configurations

### UtilityConfig

```python
@dataclass
class UtilityConfig:
    """Configuration for utility operations."""
    hybrid_default_alpha: float = 0.5
    hybrid_min_alpha: float = 0.0
    hybrid_max_alpha: float = 1.0
    stats_include_index_details: bool = True
    namespace_batch_size: int = 1000
    delete_namespace_require_confirmation: bool = True
```

### ExportImportConfig

```python
@dataclass
class ExportImportConfig:
    """Configuration for export/import operations."""
    default_export_format: str = "parquet"
    export_batch_size: int = 5000
    import_batch_size: int = 1000
    import_deduplicate: bool = False
    import_dedup_threshold: float = 0.95
    validate_on_import: bool = True
    parquet_compression: str = "zstd"
    csv_include_vectors: bool = False
    max_export_records: int = 0  # 0 = unlimited
```

---

## 5. MCP Tool Schemas

### stats Tool

```python
Tool(
    name="stats",
    description=(
        "Get comprehensive database statistics including memory counts, "
        "storage size, index information, and health metrics."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Filter stats to specific namespace",
            },
            "include_index_details": {
                "type": "boolean",
                "default": True,
                "description": "Include detailed index statistics",
            },
        },
    },
    outputSchema={
        "type": "object",
        "properties": {
            "total_memories": {"type": "integer"},
            "memories_by_namespace": {"type": "object"},
            "storage_bytes": {"type": "integer"},
            "storage_mb": {"type": "number"},
            "has_vector_index": {"type": "boolean"},
            "has_fts_index": {"type": "boolean"},
            "num_fragments": {"type": "integer"},
            "needs_compaction": {"type": "boolean"},
            "indices": {"type": "array"},
        },
        "required": ["total_memories", "storage_mb"],
    },
),
```

### namespaces Tool

```python
Tool(
    name="namespaces",
    description="List all namespaces with memory counts and date ranges.",
    inputSchema={
        "type": "object",
        "properties": {
            "include_stats": {
                "type": "boolean",
                "default": True,
                "description": "Include memory counts and date ranges",
            },
        },
    },
    outputSchema={
        "type": "object",
        "properties": {
            "namespaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "memory_count": {"type": "integer"},
                        "oldest_memory": {"type": "string", "format": "date-time"},
                        "newest_memory": {"type": "string", "format": "date-time"},
                    },
                },
            },
            "total_namespaces": {"type": "integer"},
            "total_memories": {"type": "integer"},
        },
        "required": ["namespaces", "total_namespaces"],
    },
),
```

### delete_namespace Tool

```python
Tool(
    name="delete_namespace",
    description=(
        "Delete all memories in a namespace. DESTRUCTIVE - requires confirmation. "
        "Use dry_run=true first to preview."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace to delete",
            },
            "confirm": {
                "type": "boolean",
                "default": False,
                "description": "Set to true to confirm deletion",
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "Preview deletion without executing",
            },
        },
        "required": ["namespace"],
    },
    outputSchema={
        "type": "object",
        "properties": {
            "namespace": {"type": "string"},
            "memories_deleted": {"type": "integer"},
            "success": {"type": "boolean"},
            "message": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["namespace", "memories_deleted", "success", "dry_run"],
    },
),
```

### rename_namespace Tool

```python
Tool(
    name="rename_namespace",
    description="Rename all memories from one namespace to another.",
    inputSchema={
        "type": "object",
        "properties": {
            "old_namespace": {
                "type": "string",
                "description": "Current namespace name",
            },
            "new_namespace": {
                "type": "string",
                "description": "New namespace name",
            },
        },
        "required": ["old_namespace", "new_namespace"],
    },
    outputSchema={
        "type": "object",
        "properties": {
            "old_namespace": {"type": "string"},
            "new_namespace": {"type": "string"},
            "memories_renamed": {"type": "integer"},
            "success": {"type": "boolean"},
            "message": {"type": "string"},
        },
        "required": ["old_namespace", "new_namespace", "memories_renamed", "success"],
    },
),
```

### export_memories Tool

```python
Tool(
    name="export_memories",
    description=(
        "Export memories to file. Supports parquet (recommended), json, or csv formats."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "Path for output file (extension determines format)",
            },
            "format": {
                "type": "string",
                "enum": ["parquet", "json", "csv"],
                "description": "Export format (auto-detected from extension if not specified)",
            },
            "namespace": {
                "type": "string",
                "description": "Export only this namespace (all if not specified)",
            },
            "include_vectors": {
                "type": "boolean",
                "default": True,
                "description": "Include embedding vectors",
            },
        },
        "required": ["output_path"],
    },
    outputSchema={
        "type": "object",
        "properties": {
            "format": {"type": "string"},
            "output_path": {"type": "string"},
            "memories_exported": {"type": "integer"},
            "file_size_bytes": {"type": "integer"},
            "file_size_mb": {"type": "number"},
            "namespaces_included": {"type": "array", "items": {"type": "string"}},
            "duration_seconds": {"type": "number"},
        },
        "required": ["output_path", "memories_exported", "file_size_mb"],
    },
),
```

### import_memories Tool

```python
Tool(
    name="import_memories",
    description=(
        "Import memories from backup file. Supports parquet, json, or csv formats."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "Path to source file",
            },
            "format": {
                "type": "string",
                "enum": ["parquet", "json", "csv"],
                "description": "Import format (auto-detected from extension if not specified)",
            },
            "namespace_override": {
                "type": "string",
                "description": "Override namespace for all imported memories",
            },
            "deduplicate": {
                "type": "boolean",
                "default": False,
                "description": "Skip records similar to existing memories",
            },
            "dedup_threshold": {
                "type": "number",
                "minimum": 0.7,
                "maximum": 0.99,
                "default": 0.95,
                "description": "Similarity threshold for deduplication",
            },
            "validate": {
                "type": "boolean",
                "default": True,
                "description": "Validate records before import",
            },
            "regenerate_embeddings": {
                "type": "boolean",
                "default": False,
                "description": "Generate new embeddings (required if vectors missing)",
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "Validate without importing",
            },
        },
        "required": ["source_path"],
    },
    outputSchema={
        "type": "object",
        "properties": {
            "source_path": {"type": "string"},
            "format": {"type": "string"},
            "total_records_in_file": {"type": "integer"},
            "memories_imported": {"type": "integer"},
            "memories_skipped": {"type": "integer"},
            "memories_failed": {"type": "integer"},
            "validation_errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_number": {"type": "integer"},
                        "field": {"type": "string"},
                        "error": {"type": "string"},
                    },
                },
            },
            "dry_run": {"type": "boolean"},
            "duration_seconds": {"type": "number"},
        },
        "required": ["source_path", "memories_imported", "dry_run"],
    },
),
```

### hybrid_recall Tool

```python
Tool(
    name="hybrid_recall",
    description=(
        "Search using combined vector similarity and keyword matching. "
        "Alpha parameter controls balance: 1.0=pure vector, 0.0=pure keyword, 0.5=balanced."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "alpha": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
                "description": "Balance between vector (1.0) and keyword (0.0) search",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 5,
                "description": "Maximum number of results",
            },
            "namespace": {
                "type": "string",
                "description": "Filter to specific namespace",
            },
            "min_similarity": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.0,
                "description": "Minimum similarity threshold",
            },
        },
        "required": ["query"],
    },
    outputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "alpha": {"type": "number"},
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "similarity": {"type": "number"},
                        "namespace": {"type": "string"},
                        "vector_score": {"type": "number"},
                        "fts_score": {"type": "number"},
                    },
                },
            },
            "total": {"type": "integer"},
            "search_type": {"type": "string"},
        },
        "required": ["query", "memories", "total"],
    },
),
```

---

## 6. Database Layer Methods

### get_stats

```python
def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
    """Get comprehensive database statistics.

    Uses efficient LanceDB queries:
    - count_rows() for totals (O(1) with cache)
    - SQL aggregations via search().to_arrow() for distributions
    - Index stats via list_indices() and index_stats()

    Args:
        namespace: Filter stats to specific namespace (None = all).

    Returns:
        Dictionary with statistics.

    Raises:
        StorageError: If database operation fails.
    """
    try:
        metrics = self.get_health_metrics()

        # Get memory counts by namespace using efficient Arrow aggregation
        ns_arrow = self.table.search().select(["namespace"]).to_arrow()
        ns_counts = ns_arrow.to_pandas()["namespace"].value_counts().to_dict()

        return {
            "total_memories": metrics.total_rows,
            "namespaces": ns_counts,
            "storage_bytes": metrics.total_bytes,
            "storage_mb": metrics.total_bytes_mb,
            "num_fragments": metrics.num_fragments,
            "needs_compaction": metrics.needs_compaction,
            "has_vector_index": metrics.has_vector_index,
            "has_fts_index": metrics.has_fts_index,
            "table_version": metrics.version,
            "indices": [
                {
                    "name": idx.name,
                    "index_type": idx.index_type,
                    "num_indexed_rows": idx.num_indexed_rows,
                    "status": "ready" if not idx.needs_update else "needs_update",
                }
                for idx in metrics.indices
            ],
        }
    except Exception as e:
        raise StorageError(f"Failed to get stats: {e}") from e
```

### rename_namespace

```python
def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
    """Rename all memories from one namespace to another.

    Uses atomic batch update via merge_insert for data integrity.

    Args:
        old_namespace: Source namespace name.
        new_namespace: Target namespace name.

    Returns:
        Number of memories renamed.

    Raises:
        ValidationError: If namespace names are invalid.
        NamespaceNotFoundError: If old_namespace doesn't exist.
        StorageError: If database operation fails.
    """
    old_namespace = _validate_namespace(old_namespace)
    new_namespace = _validate_namespace(new_namespace)
    safe_old = _sanitize_string(old_namespace)

    try:
        # Check if source namespace exists
        existing = self.get_namespaces()
        if old_namespace not in existing:
            raise NamespaceNotFoundError(old_namespace)

        # Fetch all records in batches
        BATCH_SIZE = 1000
        updated = 0

        while True:
            records = (
                self.table.search()
                .where(f"namespace = '{safe_old}'")
                .limit(BATCH_SIZE)
                .to_list()
            )

            if not records:
                break

            # Update namespace field
            for r in records:
                r["namespace"] = new_namespace
                r["updated_at"] = utc_now()
                if isinstance(r.get("metadata"), dict):
                    r["metadata"] = json.dumps(r["metadata"])
                if isinstance(r.get("vector"), np.ndarray):
                    r["vector"] = r["vector"].tolist()

            # Atomic upsert
            (
                self.table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )

            updated += len(records)

        self._invalidate_namespace_cache()
        return updated

    except (ValidationError, NamespaceNotFoundError):
        raise
    except Exception as e:
        raise StorageError(f"Failed to rename namespace: {e}") from e
```

### get_all_for_export (Streaming)

```python
def get_all_for_export(
    self,
    namespace: str | None = None,
    batch_size: int = 1000,
) -> Iterator[list[dict[str, Any]]]:
    """Stream all memories for export in batches.

    Memory-efficient export using generator pattern.

    Args:
        namespace: Optional namespace filter.
        batch_size: Records per batch.

    Yields:
        Batches of memory dictionaries.

    Raises:
        StorageError: If database operation fails.
    """
    try:
        search = self.table.search()

        if namespace:
            namespace = _validate_namespace(namespace)
            safe_ns = _sanitize_string(namespace)
            search = search.where(f"namespace = '{safe_ns}'")

        # Use Arrow for efficient streaming
        arrow_table = search.to_arrow()
        records = arrow_table.to_pylist()

        # Yield in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            # Process metadata
            for record in batch:
                if isinstance(record.get("metadata"), str):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}

            yield batch

    except Exception as e:
        raise StorageError(f"Failed to stream export: {e}") from e
```

### bulk_import

```python
def bulk_import(
    self,
    records: Iterator[dict[str, Any]],
    batch_size: int = 1000,
    namespace_override: str | None = None,
) -> tuple[int, list[str]]:
    """Import memories from an iterator of records.

    Supports streaming import for large datasets.

    Args:
        records: Iterator of memory dictionaries.
        batch_size: Records per database insert batch.
        namespace_override: Override namespace for all records.

    Returns:
        Tuple of (records_imported, list_of_new_ids).

    Raises:
        ValidationError: If records contain invalid data.
        StorageError: If database operation fails.
    """
    if namespace_override:
        namespace_override = _validate_namespace(namespace_override)

    imported = 0
    all_ids: list[str] = []
    batch: list[dict[str, Any]] = []

    try:
        for record in records:
            prepared = self._prepare_import_record(record, namespace_override)
            batch.append(prepared)

            if len(batch) >= batch_size:
                ids = self.insert_batch(batch, batch_size=batch_size)
                all_ids.extend(ids)
                imported += len(ids)
                batch = []

        # Import remaining
        if batch:
            ids = self.insert_batch(batch, batch_size=batch_size)
            all_ids.extend(ids)
            imported += len(ids)

        return imported, all_ids

    except (ValidationError, StorageError):
        raise
    except Exception as e:
        raise StorageError(f"Bulk import failed: {e}") from e
```

### hybrid_search Enhancement

```python
def hybrid_search(
    self,
    query: str,
    query_vector: np.ndarray,
    limit: int = 5,
    namespace: str | None = None,
    alpha: float = 0.5,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """Hybrid search combining vector similarity and keyword matching.

    Enhanced to handle different score column variants from LanceDB:
    - _distance: Vector search distance (lower = better)
    - _score: FTS BM25 score (higher = better)
    - _relevance_score: Reranker combined score (higher = better)

    Args:
        query: Text query for full-text search.
        query_vector: Embedding vector for semantic search.
        limit: Number of results.
        namespace: Filter to namespace.
        alpha: Balance between vector (1.0) and keyword (0.0).
        min_similarity: Minimum similarity threshold (0-1).

    Returns:
        List of memory records with similarity scores.
    """
    # ... implementation handles score column variants

    for record in results:
        # Handle different score columns
        similarity = None

        if "_relevance_score" in record:
            similarity = float(record["_relevance_score"])
            del record["_relevance_score"]
        elif "_score" in record:
            score = float(record["_score"])
            similarity = score / (1 + score)  # Normalize BM25
            del record["_score"]
        elif "_distance" in record:
            distance = float(record["_distance"])
            similarity = max(0.0, min(1.0, 1 - distance))
            del record["_distance"]
        else:
            similarity = 0.5

        record["similarity"] = similarity
        record["search_type"] = "hybrid"
        record["alpha"] = alpha

        if similarity >= min_similarity:
            filtered.append(record)
```

---

## 7. Protocol Extensions

### Architectural Decision: What Belongs on Repository Protocol

Following Clean Architecture principles, the repository protocol should contain:
- **Data access operations** (CRUD, search, queries)
- **Data mutation operations** (namespace rename is a data update)
- **Aggregation queries** (stats are efficient database operations)

The repository protocol should NOT contain:
- **File I/O operations** (moved to service layer)
- **Format conversion** (service layer concern)
- **Security validation** (service layer concern)

### Methods to Add to `MemoryRepositoryProtocol`

Add to `spatial_memory/ports/repositories.py`:

```python
def delete_by_namespace(self, namespace: str) -> int:
    """Delete all memories in a namespace.

    Note: Already exists in Database class, needs protocol method.
    """
    ...

def rename_namespace(self, old_namespace: str, new_namespace: str) -> int:
    """Rename all memories from one namespace to another.

    This is a data mutation operation, appropriate for repository.
    """
    ...

def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
    """Get database statistics.

    Aggregation query - efficient at database layer.
    """
    ...

def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
    """Get statistics for a specific namespace."""
    ...

def get_all_for_export(
    self,
    namespace: str | None = None,
    batch_size: int = 1000,
) -> Iterator[list[dict[str, Any]]]:
    """Stream all memories for export in batches.

    Note: Returns data only - no file I/O. Service layer handles
    file writing and format conversion.
    """
    ...

def bulk_import(
    self,
    records: Iterator[dict[str, Any]],
    batch_size: int = 1000,
    namespace_override: str | None = None,
) -> tuple[int, list[str]]:
    """Import memories from an iterator of records.

    Note: Accepts pre-parsed records - no file I/O. Service layer
    handles file reading, parsing, and validation.
    """
    ...
```

### Existing Methods to Keep (Already on Protocol)

These methods already exist and support Phase 5:
- `export_to_parquet()` - Keep for backward compatibility (low-level Parquet only)
- `import_from_parquet()` - Keep for backward compatibility (low-level Parquet only)
- `hybrid_search()` - Already exists, will be enhanced
- `get_health_metrics()` - Already exists, used by stats

---

## 8. Security Specifications

### Path Traversal Prevention

```python
# Path traversal patterns to detect
PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\."),              # Parent directory
    re.compile(r"%2e%2e", re.I),      # URL-encoded ..
    re.compile(r"%252e%252e", re.I),  # Double URL-encoded
    re.compile(r"\\\\"),              # UNC path (Windows)
]

# Sensitive directories
SENSITIVE_DIRECTORIES = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/var/log", "/root",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
})
```

### PathValidator Class

```python
class PathValidator:
    """Validates file paths for security constraints."""

    def validate_export_path(self, path: str | Path) -> Path:
        """Validate path for export operations.

        Checks:
        1. Path traversal detection
        2. Canonicalization
        3. Symlink detection
        4. Allowlist validation
        5. Extension validation
        """
        ...

    def validate_import_path(self, path: str | Path, max_size_bytes: int) -> Path:
        """Validate path for import operations.

        Additional checks:
        6. File existence
        7. Size limit enforcement
        """
        ...
```

### Destructive Operation Confirmation

```python
def delete_namespace(
    namespace: str,
    confirm: bool = False,
    dry_run: bool = True,
) -> DeleteNamespaceResult:
    """Delete all memories in a namespace.

    DESTRUCTIVE OPERATION - requires explicit confirmation unless dry_run.

    Pattern:
    1. First call with dry_run=True to preview
    2. Second call with dry_run=False, confirm=True to execute
    """
    if not dry_run and not confirm:
        raise ValidationError(
            f"Deletion of {memory_count} memories requires confirm=True"
        )
```

---

## 9. Testing Strategy

### Existing Test Coverage (DO NOT DUPLICATE)

The following existing tests provide coverage that Phase 5 tests should NOT duplicate:

| Existing File | Coverage | Phase 5 Implication |
|---------------|----------|---------------------|
| `test_backup_restore.py` | Parquet export/import at database layer | Phase 5 tests focus on service layer, multi-format, security |
| `test_enterprise_features.py` | `TestHybridSearchAlpha` - alpha parameter behavior | Phase 5 hybrid tests focus on service layer, score normalization |

### Test File Overview

| File | Location | Tests | Description |
|------|----------|-------|-------------|
| `test_stats_ops.py` | `tests/unit/` | 18 | Stats calculation unit tests |
| `test_namespace_ops.py` | `tests/unit/` | 24 | Namespace operation unit tests |
| `test_export_ops.py` | `tests/unit/` | 22 | Export service-layer tests (format conversion, security) |
| `test_import_ops.py` | `tests/unit/` | 28 | Import service-layer tests (validation, security) |
| `test_hybrid_search_ops.py` | `tests/unit/` | 12 | Hybrid search service tests (score normalization, column variants) |
| `test_utility_service.py` | `tests/unit/` | 20 | UtilityService unit tests |
| `test_utility_integration.py` | `tests/integration/` | 18 | Integration tests |
| `test_security.py` | `tests/unit/` | 6 | Security-focused tests |

**Total: 148 new tests** (reduced from 178 to avoid duplicating existing coverage)

### conftest.py Updates Required

Add mock repository methods for Phase 5:

```python
# In tests/conftest.py - add to mock_repository fixture
repo.get_stats.return_value = {"total_memories": 0, "namespaces": {}}
repo.rename_namespace.return_value = 0
repo.delete_by_namespace.return_value = 0
repo.get_all_for_export.return_value = iter([])
repo.bulk_import.return_value = (0, [])
repo.hybrid_search.return_value = []
```

### Critical Security Tests

```python
class TestPathTraversal:
    """Test path traversal attack prevention."""

    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "..\\..\\Windows\\System32\\config",
        "exports/../../sensitive",
        "exports/%2e%2e/secret",
        "exports/..%252f../admin",
    ])
    def test_traversal_blocked(self, validator, malicious_path):
        with pytest.raises(PathSecurityError) as exc_info:
            validator.validate_export_path(malicious_path)
        assert exc_info.value.violation_type == "traversal_attempt"


class TestFileSizeLimits:
    """Test file size limit enforcement."""

    def test_oversized_file_rejected(self, tmp_path):
        large_file = tmp_path / "large.json"
        large_file.write_bytes(b"x" * (101 * 1024 * 1024))  # 101MB

        with pytest.raises(FileSizeLimitError):
            validate_import_file(large_file, max_size_mb=100)


class TestDimensionValidation:
    """Test vector dimension validation."""

    def test_dimension_mismatch_rejected(self):
        config = ImportValidationConfig(expected_vector_dim=384)
        record = {"content": "test", "vector": [0.1] * 512}  # Wrong dim

        result = validate_import_record(record, config, 0)
        assert not result.valid
        assert any("dimension mismatch" in e for e in result.errors)
```

---

## 10. Implementation Order

**CRITICAL: Protocol methods MUST be defined before services that use them.**

### Phase 5.1: Foundation (Day 1-2)
1. Add error types to `spatial_memory/core/errors.py`
2. Add config options to `spatial_memory/config.py`
3. Create `spatial_memory/core/file_security.py` with PathValidator
4. Add result dataclasses to `spatial_memory/core/models.py`
5. **Add protocol extensions to `spatial_memory/ports/repositories.py`** (BEFORE services)
6. Update `tests/conftest.py` with mock repository methods

### Phase 5.2: Database Layer (Day 3-4)
1. Add `get_stats()`, `get_namespace_stats()` to database.py
2. Add `rename_namespace()` to database.py
3. Add `get_all_for_export()`, `bulk_import()` to database.py
4. Add `delete_by_namespace()` protocol method to LanceDBMemoryRepository
5. Enhance `hybrid_search()` in database.py (score column handling)

### Phase 5.3: UtilityService (Day 5-6)
1. Create `spatial_memory/services/utility.py` with UtilityService
2. Implement: stats(), namespaces(), delete_namespace(), rename_namespace(), hybrid_recall()
3. Register tools: stats, namespaces, delete_namespace, rename_namespace, hybrid_recall
4. Write tests: test_stats_ops.py, test_namespace_ops.py, test_hybrid_search_ops.py
5. Write tests: test_utility_service.py

### Phase 5.4: ExportImportService (Day 7-8)
1. Create `spatial_memory/core/import_security.py`
2. Create `spatial_memory/services/export_import.py` with ExportImportService
3. Implement: export_memories() with format handlers (Parquet/JSON/CSV)
4. Implement: import_memories() with validation and streaming
5. Register tools: export_memories, import_memories
6. Write tests: test_export_ops.py, test_import_ops.py, test_security.py

### Phase 5.5: Integration & Polish (Day 9-10)
1. Write integration tests: test_utility_integration.py
2. Add `isError: true` to all error responses in server.py
3. Update README.md with new tools
4. Run full test suite, fix any issues
5. Update CHANGELOG.md

---

## 11. Verification Commands

```bash
# 1. Type checking
mypy spatial_memory --strict

# 2. Linting
ruff check spatial_memory tests

# 3. Unit tests
pytest tests/unit/test_stats_ops.py tests/unit/test_namespace_ops.py -v
pytest tests/unit/test_export_ops.py tests/unit/test_import_ops.py -v
pytest tests/unit/test_hybrid_search_ops.py tests/unit/test_utility_service.py -v
pytest tests/unit/test_security.py -v

# 4. Integration tests
pytest tests/integration/test_utility_integration.py -v

# 5. All tests with coverage
pytest tests/ -v --cov=spatial_memory --cov-report=term-missing

# 6. Verify tool count
python -c "from spatial_memory.server import SpatialMemoryServer; s = SpatialMemoryServer(); print(len(s._tools))"
# Expected: 20 (13 existing + 7 new)
```

---

## 12. Success Criteria

### Functional
- [ ] All 7 MCP tools implemented and tested
- [ ] stats returns comprehensive database information
- [ ] namespaces lists all namespaces with counts
- [ ] delete_namespace requires confirmation, supports dry_run
- [ ] rename_namespace atomically updates all records
- [ ] export_memories supports JSON/CSV/Parquet with streaming
- [ ] import_memories validates data, handles conflicts
- [ ] hybrid_recall balances vector and keyword search

### Quality
- [ ] 148+ new tests passing (adjusted to avoid duplicating existing coverage)
- [ ] Total test count ~690
- [ ] mypy strict: 0 errors
- [ ] ruff: 0 issues
- [ ] No security vulnerabilities in path handling

### Security
- [ ] Path traversal attacks blocked
- [ ] Symlink attacks detected and blocked
- [ ] File size limits enforced
- [ ] Vector dimensions validated on import
- [ ] Destructive operations require confirmation

### Performance
- [ ] Export 100K memories in <60 seconds
- [ ] Import 10K memories in <30 seconds
- [ ] Stats query <1 second for 100K memories
- [ ] Namespace rename <10 seconds for 10K memories

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Path traversal attacks | Comprehensive PathValidator with pattern detection |
| Large file DoS | File size limits (configurable, default 100MB) |
| Import data corruption | Schema validation, dimension checking |
| Accidental deletion | Require explicit confirm=True, dry_run preview |
| Memory exhaustion on export | Streaming batches, configurable batch size |
| Breaking LanceDB API changes | Pin version >=0.17.0, add compatibility layer |

---

## References

### Project Plans
- Phase 4 Plan: `PHASE4_PLAN.md`
- Phase 4B Plan: `PHASE4B_PLAN.md`
- Lifecycle Phase Plan: `LIFECYCLE_PHASE_PLAN.md`

### Technical Documentation
- LanceDB Documentation: https://lancedb.github.io/lancedb/
- MCP Specification (November 2025): https://modelcontextprotocol.io/specification/2025-11-25
- MCP Tools Specification: https://modelcontextprotocol.io/specification/2025-11-25/server/tools

### Architecture References
- Clean Architecture: Ports & Adapters pattern
- Interface Segregation Principle: Protocol design
- Single Responsibility: Service layer separation

### Memory Decisions
- `.claude-memory/memories/decisions/`
- Architecture review decision: 2026-01-30
