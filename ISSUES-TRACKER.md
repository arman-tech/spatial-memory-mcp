# Spatial Memory MCP - Issues Tracker

> Generated: 2026-02-01
> Status: Planning Complete
> Last Updated: 2026-02-01

---

## Overview

This document tracks all identified issues from the comprehensive codebase review and their planned resolutions.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 2 | Solutions Planned |
| High | 10 | Solutions Planned |
| Medium | 18 | Triaged |
| Low | 15+ | Backlog |

### Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CRIT-001 Data Safety | Add-before-delete pattern | Simpler than soft-delete, no schema change required |
| HIGH-010 Deployment | Single-machine only | Document limitation, detect and warn at startup |
| MED-SEC-001 Multi-tenant | Not a priority | Current behavior is intended for single-developer local use |

---

## Critical Issues

### CRIT-001: Consolidation Data Loss Window

- **Location:** `spatial_memory/services/lifecycle.py:690-720`
- **Description:** "merge_content" strategy deletes originals before adding merged memory. If add fails (e.g., embedding timeout), originals are permanently lost.
- **Status:** ✅ COMPLETED (2026-02-01)
- **Priority:** P0

#### Root Cause
The code uses delete-before-add pattern:
```
delete(id_1) -> delete(id_2) -> add(merged) [FAILS] -> DATA LOST
```

#### Solution: Add-Before-Delete with Status Pattern

**Implementation Steps:**
1. **Prepare merged memory** - Generate content, metadata, embedding (no DB write)
2. **Create pending merged memory** - Add with `_consolidation_status: "pending"` in metadata
3. **Verify merged memory exists** - Confirm write succeeded
4. **Delete original memories** - Use `delete_batch()` for atomic deletion
5. **Activate merged memory** - Remove pending marker from metadata

**Files to Modify:**
- `spatial_memory/services/lifecycle.py` - Lines 668-724
- `spatial_memory/core/models.py` - Extend `ConsolidateResult`

**Interface Changes:**
- Add `pending_count` and `recovery_info` fields to `ConsolidateResult`
- Add `recover_pending_consolidations()` method to `LifecycleService`

**Test Strategy:**
- `test_consolidate_merge_content_adds_before_delete`
- `test_consolidate_rollback_on_add_failure`
- `test_consolidate_pending_status_marker`

**Breaking Changes:** None - internal implementation change only

---

### CRIT-002: Journey N+1 Query Pattern

- **Location:** `spatial_memory/services/spatial.py:231-237`
- **Description:** For each neighbor in journey results, `_get_vector_for_memory()` hits the database. 10-step journey with 3 neighbors = 30 extra DB calls.
- **Status:** ✅ COMPLETED (2026-02-01)
- **Priority:** P0

#### Root Cause
Loop calls `_get_vector_for_memory(neighbor.id)` for distance calculation, triggering individual DB queries.

#### Key Finding
**Database already supports this!** `database.py:2233` has `include_vector: bool = False` parameter - just needs plumbing through layers.

#### Solution: Plumb include_vector Through Search Pipeline

**Implementation Steps:**
1. **Extend MemoryResult** - Add `vector: list[float] | None = None` field
2. **Update MemoryRepositoryProtocol** - Add `include_vector` param to `search()` and `batch_vector_search()`
3. **Update LanceDBMemoryRepository** - Pass `include_vector` to database layer
4. **Update _record_to_memory_result** - Handle vector in result conversion
5. **Update journey()** - Request vectors with `include_vector=True`, use returned vectors

**Files to Modify:**
- `spatial_memory/core/models.py` - Add vector field to `MemoryResult`
- `spatial_memory/ports/repositories.py` - Update protocol signatures
- `spatial_memory/adapters/lancedb_repository.py` - Pass through parameter
- `spatial_memory/services/spatial.py` - Use vectors from search results

**Test Strategy:**
- `test_journey_uses_batch_search_with_vectors`
- `test_journey_no_individual_vector_lookups`
- Performance benchmark: 20-step journey before/after

**Breaking Changes:** None - new parameters have defaults preserving existing behavior

---

## High Severity Issues

### HIGH-001: Sequential DB Calls in Wander
- **Location:** `spatial_memory/services/spatial.py:331-383`
- **Description:** 2 DB calls per step (search + get_with_vector)
- **Status:** ✅ COMPLETED (2026-02-01) - Fixed as part of CRIT-002
- **Priority:** P1
- **Category:** Performance

#### Solution
Same as CRIT-002 - use `include_vector=True` in search, use returned vectors instead of follow-up `get_with_vector()` call.

---

### HIGH-002: O(n²) Similarity in Visualize
- **Location:** `spatial_memory/services/spatial.py:668-679`
- **Description:** 124,750 pairwise similarity calculations for 500 memories
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Performance

#### Solution
Use vectorized numpy operations or `scipy.spatial.distance.cdist` with cosine metric. Replace nested loop with single matrix operation.

**Implementation:**
```python
# Before: O(n²) loop
for i in range(len(vectors)):
    for j in range(i + 1, len(vectors)):
        similarity = 1.0 - self._cosine_distance(vectors[i], vectors[j])

# After: Vectorized
from scipy.spatial.distance import cdist
similarity_matrix = 1.0 - cdist(vectors, vectors, metric='cosine')
```

---

### HIGH-003: Inefficient Batch Search (Sequential)
- **Location:** `spatial_memory/services/spatial.py:781-805`
- **Description:** `_batch_vector_search` iterates and calls search individually
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Performance

#### Key Finding
**LanceDB supports native batch search:**
```python
batch_results = table.search([vec1, vec2, vec3]).limit(5).to_pandas()
# Results include 'query_index' to map back to originating query
```

#### Solution
Implement true batch search at database layer, expose through repository.

**Files to Modify:**
- `spatial_memory/core/database.py` - Add `batch_vector_search()` using native LanceDB batch
- `spatial_memory/adapters/lancedb_repository.py` - Implement `batch_vector_search()`
- `spatial_memory/services/spatial.py` - Use batch method instead of loop

---

### HIGH-004: Duplicate Embedding Generation in Extract
- **Location:** `spatial_memory/services/lifecycle.py:481-485, 781`
- **Description:** Embedding generated twice - once for dedup check, once for storage
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Performance

#### Solution
1. Batch embed all candidates upfront
2. Pass vectors through dedup check and store operations
3. Add `_check_duplicates_batch()` method for efficiency

---

### HIGH-005: Sequential Updates in Decay
- **Location:** `spatial_memory/services/lifecycle.py:276-283`
- **Description:** 500 individual writes instead of batch update
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Performance

#### Solution
Add `update_batch()` method following `update_access_batch()` pattern (uses `merge_insert`).

**Implementation:**
```python
# Before: 500 individual updates
for memory_id, new_importance in memories_to_update:
    self._repo.update(memory_id, {"importance": new_importance})

# After: Single batch update
updates = [(mid, {"importance": imp}) for mid, imp in memories_to_update]
success_count, failed_ids = self._repo.update_batch(updates)
```

**Files to Modify:**
- `spatial_memory/core/database.py` - Add `update_batch()` method
- `spatial_memory/ports/repositories.py` - Add to protocol
- `spatial_memory/adapters/lancedb_repository.py` - Implement
- `spatial_memory/services/lifecycle.py` - Use batch method

---

### HIGH-006: Sequential Fetches in Reinforce
- **Location:** `spatial_memory/services/lifecycle.py:348-387`
- **Description:** 2 DB calls per memory (get + update)
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Performance

#### Solution
Add `get_batch()` method, use with `update_batch()`.

**Implementation:**
```python
# Before: 2N calls
for memory_id in memory_ids:
    memory = self._repo.get(memory_id)  # N calls
    self._repo.update(memory_id, updates)  # N calls

# After: 2 calls total
memory_map = self._repo.get_batch(memory_ids)  # 1 call
updates_to_apply = [(mid, calc_updates(mem)) for mid, mem in memory_map.items()]
self._repo.update_batch(updates_to_apply)  # 1 call
```

---

### HIGH-007: Batch Insert NOT Atomic
- **Location:** `spatial_memory/core/database.py:1350`
- **Description:** Partial failures leave orphaned records
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Data Integrity

#### Solution: Compensating Transaction Pattern

**Implementation Steps:**
1. Create `PartialBatchInsertError` with `succeeded_ids` for recovery
2. Create `BatchInsertContext` to track inserted IDs
3. Add `insert_batch_atomic()` method that rolls back on failure
4. Track inserted IDs, delete them if subsequent batch fails

**New Types:**
```python
class PartialBatchInsertError(StorageError):
    succeeded_ids: list[str]
    failed_batch_index: int
    original_error: Exception

@dataclass
class BatchInsertResult:
    ids: list[str]
    total_inserted: int
    was_atomic: bool
```

**Breaking Changes:** Return type change mitigated with `simple: bool = True` parameter

---

### HIGH-008: rename_namespace Lacks Rollback
- **Location:** `spatial_memory/core/database.py:1684-1736`
- **Description:** Partial renames leave namespace in mixed state
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Data Integrity

#### Solution: In-Place Update with ID Tracking

**Implementation Steps:**
1. Get all IDs in old namespace upfront
2. Track renamed IDs as we go
3. On failure, rollback renamed IDs to old namespace
4. Optional: Add operation log table for crash recovery

**New Types:**
```python
class NamespaceOperationState(Enum):
    PENDING, RENAMING, COMPLETED, FAILED, ROLLED_BACK

@dataclass
class NamespaceOperationResult:
    renamed_count: int
    operation_id: str
    duration_ms: float
```

---

### HIGH-009: Consolidation Loads Entire Namespace
- **Location:** `spatial_memory/services/lifecycle.py:573-577`
- **Description:** Memory exhaustion risk for large namespaces
- **Status:** Solution Planned
- **Priority:** P1
- **Category:** Data Integrity

#### Solution: Streaming Mini-Batch Consolidation

**Implementation Steps:**
1. Add `get_all_paginated()` for memory-efficient iteration
2. Process in chunks with overlap to catch cross-chunk duplicates
3. Add `memory_budget_mb` parameter to limit memory usage
4. Optional: LSH pre-filtering for O(n * bucket_size) vs O(n²)

**New Parameters:**
```python
def consolidate(
    ...,
    max_memories_per_pass: int | None = None,
    memory_budget_mb: float = 100.0,
) -> ConsolidateResult
```

---

### HIGH-010: File Lock Doesn't Work on NFS/SMB
- **Location:** `spatial_memory/core/database.py:239-388`
- **Description:** Data corruption risk in shared filesystem
- **Status:** ✅ COMPLETED (2026-02-01)
- **Priority:** P1
- **Category:** Distributed Systems

#### Decision: Single-Machine Only (Document + Detect + Warn)

**Implementation Steps:**
1. Create `core/filesystem.py` with `detect_filesystem_type()`
2. Add startup check in `Database.connect()`
3. Log warning if NFS/SMB detected
4. Add config `acknowledge_network_filesystem_risk: bool = False`
5. Update README.md with "Deployment Considerations" section

**Detection Logic:**
- Unix: Use `df -T` to check filesystem type
- Windows: Use `GetDriveTypeW()` to detect `DRIVE_REMOTE`

**New Config:**
```python
acknowledge_network_filesystem_risk: bool = False
filelock_backend: str = "filelock"  # or "redis://host:port" for distributed
```

---

## Medium Severity Issues

### Security

| ID | Issue | Location | Status | Notes |
|----|-------|----------|--------|-------|
| MED-SEC-001 | No namespace-level authorization | server.py | Closed - By Design | Single-developer local use is intended |
| MED-SEC-002 | Memory exhaustion in JSON/CSV export | export_import.py:616-695 | Planning | Stream instead of accumulate |

### Code Quality

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-CQ-001 | forget_batch returns incorrect IDs | memory.py:407-412 | Planning |
| MED-CQ-002 | Missing backend in EmbeddingServiceProtocol | ports/repositories.py:543-581 | Planning |
| MED-CQ-003 | Rate limiter token consumption issue | rate_limiter.py:213-227 | Planning |
| MED-CQ-004 | Embedding cache leak in batch operations | embeddings.py:211-214 | Planning |

### Database

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-DB-001 | Idempotency table missing index | database.py:3073-3081 | Planning |
| MED-DB-002 | get_namespaces() loads all values | database.py:2676 | Planning |
| MED-DB-003 | get_namespace_stats() loads all records | database.py:1839-1845 | Planning |
| MED-DB-004 | Vector dimension not validated at insert | database.py:1267-1336 | Planning |
| MED-DB-005 | No migration system | Project-wide | Backlog |

### Architecture

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-ARCH-001 | LifecycleService.consolidate complexity | lifecycle.py:520-752 | Planning |
| MED-ARCH-002 | SpatialMemoryServer.__init__ complexity | server.py:140-345 | Backlog |
| MED-ARCH-003 | Tool handlers return untyped dicts | server.py:469-1012 | Backlog |
| MED-ARCH-004 | Synchronous embedding in async context | server.py:355-423 | Planning |

---

## Low Severity Issues (Backlog)

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| LOW-001 | Hardcoded embedding cache size (1000) | embeddings.py | Backlog |
| LOW-002 | Inconsistent datetime handling | lifecycle.py | Backlog |
| LOW-003 | Magic numbers in retry logic | database.py | Backlog |
| LOW-004 | Missing validation for index_type | config.py | Backlog |
| LOW-005 | Duplicate local imports | lancedb_repository.py | Backlog |
| LOW-006 | validate_namespace docstring mismatch | validation.py | Backlog |
| LOW-007 | Test mocks return non-normalized vectors | conftest.py | Backlog |
| LOW-008 | Memory content not validated for injection | validation.py | Backlog |
| LOW-009 | Metadata values not deeply validated | validation.py | Backlog |
| LOW-010 | Exception messages may leak paths | Various | Backlog |
| LOW-011 | Connection pool never validates health | connection_pool.py | Backlog |
| LOW-012 | Database.close() doesn't remove from pool | database.py | Backlog |
| LOW-013 | No automatic compaction scheduling | database.py | Backlog |
| LOW-014 | Stop words set recreated per call | spatial.py | Backlog |
| LOW-015 | MD5 for cache keys | embeddings.py | Backlog |

---

## Implementation Phases

### Phase 1: Critical Issues + Quick Wins
**Target:** Immediate
**Issues:** CRIT-001, CRIT-002, HIGH-010

| Task | Complexity | Risk |
|------|------------|------|
| CRIT-001: Add-before-delete consolidation | Medium | High value - prevents data loss |
| CRIT-002: Plumb include_vector | Low | Existing infra, just wiring |
| HIGH-010: NFS/SMB detection | Low | Quick win, prevents corruption |

### Phase 2: Batch Operations Infrastructure
**Target:** Next Sprint
**Issues:** HIGH-001, HIGH-003, HIGH-005, HIGH-006

| Task | Complexity | Dependency |
|------|------------|------------|
| Add `get_batch()` to database | Medium | None |
| Add `update_batch()` to database | Medium | None |
| Add native `batch_vector_search()` | Medium | LanceDB batch API |
| Update decay/reinforce to use batches | Low | Above methods |

### Phase 3: Remaining High Severity
**Target:** Following Sprint
**Issues:** HIGH-002, HIGH-004, HIGH-007, HIGH-008, HIGH-009

| Task | Complexity | Notes |
|------|------------|-------|
| Vectorize similarity (HIGH-002) | Low | scipy.cdist |
| Batch embedding in extract (HIGH-004) | Medium | Requires embedding cache refactor |
| Atomic batch insert (HIGH-007) | High | Compensating transaction pattern |
| Namespace rename rollback (HIGH-008) | High | Operation log table |
| Streaming consolidation (HIGH-009) | Medium | Chunked processing |

### Phase 4: Medium Severity
**Target:** Future
**Issues:** All MED-* issues

### Phase 5: Low Severity
**Target:** Opportunistic
**Issues:** All LOW-* issues

---

## Key Files by Issue

| File | Issues |
|------|--------|
| `spatial_memory/services/lifecycle.py` | CRIT-001, HIGH-004, HIGH-005, HIGH-006, HIGH-009 |
| `spatial_memory/services/spatial.py` | CRIT-002, HIGH-001, HIGH-002, HIGH-003 |
| `spatial_memory/core/database.py` | HIGH-003, HIGH-005, HIGH-006, HIGH-007, HIGH-008, HIGH-010 |
| `spatial_memory/core/models.py` | CRIT-001, CRIT-002 |
| `spatial_memory/ports/repositories.py` | CRIT-002, HIGH-003, HIGH-005, HIGH-006 |
| `spatial_memory/adapters/lancedb_repository.py` | CRIT-002, HIGH-003, HIGH-005, HIGH-006 |
| `spatial_memory/config.py` | HIGH-010 |

---

## Change Log

| Date | Issue | Change | Author |
|------|-------|--------|--------|
| 2026-02-01 | All | Initial tracking document created | Claude |
| 2026-02-01 | All | Solution plans added for Critical and High severity | Claude |
| 2026-02-01 | MED-SEC-001 | Closed as "By Design" - single-developer local use | Claude |
| 2026-02-01 | HIGH-010 | Decision: Single-machine only with detection/warning | User |
| 2026-02-01 | CRIT-001 | Decision: Add-before-delete pattern | User |
| 2026-02-01 | CRIT-001 | ✅ Implemented add-before-delete pattern in consolidation | Claude |
| 2026-02-01 | CRIT-002 | ✅ Plumbed include_vector through search pipeline (also fixes HIGH-001) | Claude |
| 2026-02-01 | HIGH-010 | ✅ Added NFS/SMB detection and warning at startup | Claude |

---

## References

- [LanceDB Batch Search Documentation](https://docs.lancedb.com/search/vector-search)
- [LanceDB merge_insert API](https://lancedb.github.io/lancedb/python/python/)
- Code Review Report: Saved to spatial-memory namespace (memory ID: 55cd072b-8bc6-4426-a80b-fa07edaee7e7)
