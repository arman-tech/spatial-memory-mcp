# Spatial Memory MCP - Issues Tracker

> Generated: 2026-02-01
> Status: **All Phases Complete - Released v1.6.0**
> Last Updated: 2026-02-02

---

## Overview

This document tracks all identified issues from the comprehensive codebase review and their resolutions.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 2 | ✅ All Complete |
| High | 10 | ✅ All Complete |
| Medium | 18 | ✅ 8 Complete, 10 Backlog |
| Low | 15+ | ✅ 4 Complete, 11+ Backlog |

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

#### Solution Implemented
Add-before-delete pattern with pending status marker in metadata.

---

### CRIT-002: Journey N+1 Query Pattern

- **Location:** `spatial_memory/services/spatial.py:231-237`
- **Description:** For each neighbor in journey results, `_get_vector_for_memory()` hits the database. 10-step journey with 3 neighbors = 30 extra DB calls.
- **Status:** ✅ COMPLETED (2026-02-01)
- **Priority:** P0

#### Solution Implemented
Plumbed `include_vector` parameter through search pipeline.

---

## High Severity Issues

### HIGH-001: Sequential DB Calls in Wander
- **Location:** `spatial_memory/services/spatial.py:331-383`
- **Description:** 2 DB calls per step (search + get_with_vector)
- **Status:** ✅ COMPLETED (2026-02-01) - Fixed as part of CRIT-002
- **Priority:** P1

---

### HIGH-002: O(n²) Similarity in Visualize
- **Location:** `spatial_memory/services/spatial.py:668-679`
- **Description:** 124,750 pairwise similarity calculations for 500 memories
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Replaced O(n²) loop with `scipy.spatial.distance.cdist` vectorized operation. Falls back to numpy if scipy unavailable.

---

### HIGH-003: Inefficient Batch Search (Sequential)
- **Location:** `spatial_memory/services/spatial.py:781-805`
- **Description:** `_batch_vector_search` iterates and calls search individually
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Implemented native LanceDB batch search at database layer using `table.search([vec1, vec2, ...])`.

---

### HIGH-004: Duplicate Embedding Generation in Extract
- **Location:** `spatial_memory/services/lifecycle.py:481-485, 781`
- **Description:** Embedding generated twice - once for dedup check, once for storage
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Batch embed all candidates upfront with `embed_batch()`, pass vectors through dedup check and store operations.

---

### HIGH-005: Sequential Updates in Decay
- **Location:** `spatial_memory/services/lifecycle.py:276-283`
- **Description:** 500 individual writes instead of batch update
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Added `update_batch()` method returning `(success_count, failed_ids)` tuple.

---

### HIGH-006: Sequential Fetches in Reinforce
- **Location:** `spatial_memory/services/lifecycle.py:348-387`
- **Description:** 2 DB calls per memory (get + update)
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Added `get_batch()` method returning `{id: Memory}` dict. Combined with `update_batch()` for 2 calls total.

---

### HIGH-007: Batch Insert NOT Atomic
- **Location:** `spatial_memory/core/database.py:1350`
- **Description:** Partial failures leave orphaned records
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Added `PartialBatchInsertError` with `succeeded_ids` for recovery. Added `atomic=True` parameter to `insert_batch()` with rollback capability.

---

### HIGH-008: rename_namespace Lacks Rollback
- **Location:** `spatial_memory/core/database.py:1684-1736`
- **Description:** Partial renames leave namespace in mixed state
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Track renamed IDs during operation. On failure, `_rollback_namespace_rename()` reverts records to original namespace.

---

### HIGH-009: Consolidation Loads Entire Namespace
- **Location:** `spatial_memory/services/lifecycle.py:573-577`
- **Description:** Memory exhaustion risk for large namespaces
- **Status:** ✅ COMPLETED (2026-02-02)
- **Priority:** P1

#### Solution Implemented
Added `_consolidate_chunked()` for streaming consolidation with configurable `consolidate_chunk_size` (default: 200).

---

### HIGH-010: File Lock Doesn't Work on NFS/SMB
- **Location:** `spatial_memory/core/database.py:239-388`
- **Description:** Data corruption risk in shared filesystem
- **Status:** ✅ COMPLETED (2026-02-01)
- **Priority:** P1

#### Solution Implemented
Added `core/filesystem.py` with detection. Startup warning if NFS/SMB detected. Config `acknowledge_network_filesystem_risk` to suppress.

---

## Medium Severity Issues

### Security

| ID | Issue | Location | Status | Notes |
|----|-------|----------|--------|-------|
| MED-SEC-001 | No namespace-level authorization | server.py | Closed - By Design | Single-developer local use is intended |
| MED-SEC-002 | Memory exhaustion in JSON/CSV export | export_import.py:616-695 | ✅ COMPLETED (2026-02-01) | Streaming export implemented |

### Code Quality

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-CQ-001 | forget_batch returns incorrect IDs | memory.py:407-412 | ✅ COMPLETED (2026-02-01) |
| MED-CQ-002 | Missing backend in EmbeddingServiceProtocol | ports/repositories.py:543-581 | ✅ COMPLETED (2026-02-01) |
| MED-CQ-003 | Rate limiter token consumption issue | rate_limiter.py:213-227 | ✅ COMPLETED (2026-02-01) |
| MED-CQ-004 | Embedding cache leak in batch operations | embeddings.py:211-214 | ✅ COMPLETED (2026-02-01) |

### Database

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-DB-001 | Idempotency table missing index | database.py:3073-3081 | ✅ COMPLETED (2026-02-01) |
| MED-DB-002 | get_namespaces() loads all values | database.py:2676 | ✅ CLOSED (Acceptable - TTL caching sufficient) |
| MED-DB-003 | get_namespace_stats() loads all records | database.py:1839-1845 | ✅ COMPLETED (2026-02-01) |
| MED-DB-004 | Vector dimension not validated at insert | database.py:1267-1336 | ✅ COMPLETED (2026-02-01) |
| MED-DB-005 | No migration system | Project-wide | Backlog |

### Architecture

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| MED-ARCH-001 | LifecycleService.consolidate complexity | lifecycle.py:520-752 | Backlog |
| MED-ARCH-002 | SpatialMemoryServer.__init__ complexity | server.py:140-345 | Backlog |
| MED-ARCH-003 | Tool handlers return untyped dicts | server.py:469-1012 | Backlog |
| MED-ARCH-004 | Synchronous embedding in async context | server.py:355-423 | Backlog |

---

## Low Severity Issues

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| LOW-001 | Hardcoded embedding cache size (1000) | embeddings.py | ✅ COMPLETED (2026-02-01) |
| LOW-002 | Inconsistent datetime handling | lifecycle.py | ✅ COMPLETED |
| LOW-003 | Magic numbers in retry logic | database.py | ✅ COMPLETED (2026-02-01) |
| LOW-004 | Missing validation for index_type | config.py | ✅ COMPLETED |
| LOW-005 | Duplicate local imports | lancedb_repository.py | ✅ CLOSED (False Positive) |
| LOW-006 | validate_namespace docstring mismatch | validation.py | ✅ COMPLETED |
| LOW-007 | Test mocks return non-normalized vectors | conftest.py | ✅ COMPLETED |
| LOW-008 | Memory content not validated for injection | validation.py | ✅ COMPLETED (Documented as by-design) |
| LOW-009 | Metadata values not deeply validated | validation.py | ✅ COMPLETED |
| LOW-010 | Exception messages may leak paths | Various | ✅ COMPLETED |
| LOW-011 | Connection pool never validates health | connection_pool.py | ✅ COMPLETED |
| LOW-012 | Database.close() doesn't remove from pool | database.py | ✅ COMPLETED |
| LOW-013 | No automatic compaction scheduling | database.py | ✅ COMPLETED |
| LOW-014 | Stop words set recreated per call | spatial.py | ✅ COMPLETED (2026-02-01) |
| LOW-015 | MD5 for cache keys | embeddings.py | ✅ COMPLETED (2026-02-01) |

---

## Implementation Phases

### Phase 1: Critical Issues + Quick Wins ✅ COMPLETE
**Completed:** 2026-02-01
**Issues:** CRIT-001, CRIT-002, HIGH-010

### Phase 2: Batch Operations Infrastructure ✅ COMPLETE
**Completed:** 2026-02-02
**Issues:** HIGH-001, HIGH-003, HIGH-005, HIGH-006

### Phase 3: Remaining High Severity ✅ COMPLETE
**Completed:** 2026-02-02
**Issues:** HIGH-002, HIGH-004, HIGH-007, HIGH-008, HIGH-009

### Phase 4: Medium Severity ✅ COMPLETE
**Completed:** 2026-02-01
**Issues:** MED-SEC-002, MED-CQ-001-004, MED-DB-001, MED-DB-003, MED-DB-004

### Phase 5: Low Severity ✅ COMPLETE
**Completed:** 2026-02-01
**Issues:** LOW-001, LOW-003, LOW-014, LOW-015

---

## Summary

**All Critical, High, and most Low severity issues have been resolved.**

| Category | Resolved | Backlog |
|----------|----------|---------|
| Critical | 2/2 | 0 |
| High | 10/10 | 0 |
| Medium | 8/18 | 8 (architecture items) |
| Low | 15/15 | 0 |

Remaining backlog items are Medium severity architectural improvements (refactoring, migration system) that don't affect functionality or data integrity.

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
| 2026-02-02 | Phase 2 | ✅ Implemented batch operations: get_batch, update_batch, batch_vector_search | Claude |
| 2026-02-02 | Phase 3 | ✅ Implemented scipy.cdist, batch embedding, atomic insert, namespace rollback, streaming consolidation | Claude |
| 2026-02-01 | Phase 4 | ✅ Implemented MED-SEC-002, MED-CQ-001-004, MED-DB-001, MED-DB-003, MED-DB-004 | Claude |
| 2026-02-01 | Phase 5 | ✅ Implemented LOW-001, LOW-003, LOW-014, LOW-015 | Claude |
| 2026-02-01 | All | **All phases complete** - 1360 tests passing | Claude |
| 2026-02-02 | All | **Released v1.6.0** - Pushed to repository | User |
| 2026-02-01 | Phase A-C Backlog | ✅ Implemented remaining Low severity issues: | Claude |
|  | LOW-002 | Standardized datetime handling in lifecycle.py |  |
|  | LOW-004 | Added Literal type validation for index_type |  |
|  | LOW-005 | Closed as FALSE POSITIVE (intentional circular import avoidance) |  |
|  | LOW-006 | Fixed validate_namespace docstring |  |
|  | LOW-007 | Normalized mock embedding vectors in conftest.py |  |
|  | LOW-008 | Documented content validation security approach |  |
|  | LOW-009 | Added metadata depth validation |  |
|  | LOW-010 | Sanitized paths in error messages |  |
|  | LOW-011 | Added proactive connection health check |  |
|  | LOW-012 | Fixed close() to remove from pool |  |
|  | LOW-013 | Implemented auto-compaction |  |
|  | MED-DB-002 | Closed as ACCEPTABLE (TTL caching sufficient) |  |

---

## References

- [LanceDB Batch Search Documentation](https://docs.lancedb.com/search/vector-search)
- [LanceDB merge_insert API](https://lancedb.github.io/lancedb/python/python/)
- Code Review Report: Saved to spatial-memory namespace (memory ID: 55cd072b-8bc6-4426-a80b-fa07edaee7e7)
