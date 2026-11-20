# Stream A: `sm-store`

## Goal

Build `crates/sm-store`: the SQLite system of record for version 3.0. It holds the schema and its migrations, exact vector search (project-scoped in Rust, cross-project through `vec1`), the FTS5 keyword index, anchors and links, the map cache tables, injection records and backups. It implements `sm_core::traits::VectorIndex`. `sm-daemon` (stream E) and `sm-share` (stream F) call it directly; there is no store trait (CONTRACTS.md section 2).

**SPEC 14 done when:** "Schema, migrations, exact vector search, FTS5, links; storage tests pass."

## You own

- `crates/sm-store/` (new crate; depends on `sm-core` only, with `sm-core` `test-support` in `[dev-dependencies]`).
- Nothing else. `vendor/vec1/` is read-only: compile it from where it is.

## Read

- **SPEC:** section 4 (data model: every table, every rule), section 6 (recall pipeline steps 2 to 6, the FTS quoted phrase, `related`'s two-hop walk), section 8 (what each tool needs from storage), section 9 "Scope", section 10 (what migration writes), section 11 (AT-5, AT-11, AT-12 and the `recall` budgets), section 13 (the vec1 risk and its `sqlite-vec` fallback).
- **CONTRACTS.md:** sections 2, 9 (`VectorIndex`: best chunk per memory, sorted by score descending then id; empty `filter.statuses` means active only) and 10.
- **DECISIONS:** B-4 (data layout), B-7 (vec1 vendored, two objects), B-12 (ids), B-14 (`content_hash`), B-18 (timestamps), B-19 (provenance extensions), B-20 (project identity), B-28 to B-32 (tool semantics the store must support: batch remember with content-hash dedupe, forget by filter, consolidate, region labels, reinforce), B-36, B-37 (you decide `LinkView.stale`).
- **SPIKE.md** section b, and the spike program at `%USERPROFILE%\Documents\code-repo\spatial-memory-spikes\spike-vec1\` (`build.rs`, `src/main.rs`): how vec1 was compiled with `cc` against the bundled SQLite, the defines (`NDEBUG`, `VEC1_STATIC`, `VEC1SIMD=SCALAR` and `AVX2` with `/arch:AVX2` or `-mavx2 -mfma`), how it was registered, and the measured exact search (160 ms median at 100,000 chunks on Windows). The spike is a reference; write the crate fresh. Read `vendor/vec1/README.md`.
- **sm-core:** `types` (`Memory`, `NewMemory`, `Provenance`, `Project`, `AnchorRef`, `Link`, `LinkEnd`, `IndexStamp` and the meta keys, the limits), `ids`, `time`, `hash`, `traits::VectorIndex` and its query types, `error` (codes such as `not_found`, `index_mismatch`, `invalid_input`), `paths::ProfilePaths::{database, backups_dir}`.

## Build

1. **Schema and migrations** (SPEC 4). Every table in SPEC 4's table, with its key columns, foreign keys, and the indexes the queries below need: at least `chunks(project_id)`, `memories(project_id, status)`, `memories(project_id, content_hash)`, `memory_anchors(anchor_id)`, `links` by each end, `injections(session_id)`.
   - `chunks_fts` is FTS5 with tokenizer `porter unicode61`, columns title and text, one row per chunk, written in the same transaction as `chunks`.
   - Vectors are 768 little-endian `f32` in a BLOB.
   - A schema version in `meta`. Migrations run forward in order, in one transaction each, and copy the database file to `backups_dir()` first (SPEC 4 "Housekeeping").
   - WAL mode, one writer (SPEC 3). Set a busy timeout so that no lock error reaches a caller under AT-5's load; choose how readers and the writer get connections and record it.
2. **vec1** compiled from `vendor/vec1/vec1.c` by `build.rs` with `cc` (pinned), linked statically, and registered on every connection the store opens. Use `rusqlite` and `libsqlite3-sys` with the `bundled` feature at their pinned versions. `unsafe` is allowed in this crate for the registration call; keep it in one small module with a safety comment.
3. **Writes.**
   - Insert a memory with its provenance, anchors and chunks (and their FTS rows) in one transaction. Dedupe by `content_hash` within a project (B-29) and report which inputs were duplicates.
   - `supersedes_id` marks the older memory `superseded` in the same transaction (SPEC 4 "Types and status").
   - Update content (replacing chunks), title, type, status, tags, anchors and importance; each revise is one transaction.
   - Replace all chunks of a memory (for `reindex` and for revised content).
   - **Forget cascades** (SPEC 4): deleting a memory removes its chunks, keyword rows, anchor mappings, links, map points and injection records in one transaction. Refuse to forget a memory whose origin is `shared` (SPEC 4), with the matching `sm-core` error.
   - Forget by filter (B-30): support the filter fields in `sm_core::protocol::ForgetFilter`, and return the count before deleting so the daemon can require `confirm`.
   - Projects: upsert by identity (SPEC 4 `projects`, B-20), rename, delete (cascading its memories).
4. **Reads.**
   - Get memories by id with provenance and anchors; list and filter by project, type, status, tags and anchor.
   - **Vector search** implementing `VectorIndex::nearest_memories`:
     - Project scope: read the vectors `WHERE project_id = ?` through the index, score by dot product (the vectors are unit length) in Rust, collapse to the best chunk per memory.
     - Scope `all`: `vec1` in exact mode across all chunks, then the same collapse. Never build or use vec1's approximate index (out of scope in 3.0).
     - Apply the status filter (active only by default), type and anchor filters, sort by score descending then id, then `limit`.
   - **Keyword search** (SPEC 6 step 6): FTS5 with the query always passed as a quoted phrase, internal double quotes doubled, so text with `.`, `:` or `-` never raises a syntax error. Returns memory ids with the matching chunk.
   - **Anchors and links** (D-8): add, remove and list links between anchors or memories, with the closed relation set and `source` manual or extracted; list memories for an anchor; the two-hop walk over `links` and `memory_anchors` for `related` (SPEC 6 "Related").
   - **`LinkView.stale`** (B-37): decide how, or whether, the store can tell that `verified_commit` is no longer an ancestor of the project head without running git. If it cannot be decided, `stale` stays absent. Record the decision.
   - Counts per project and per status, and index state, for `status`.
5. **Meta and the index stamp** (AT-11): read and write `IndexStamp` through the `sm-core` meta keys. When the configured stamp differs from the stored one, every search returns `index_mismatch` until a reindex finishes and writes the new stamp. A reindex copies the database file as a backup first (SPEC 4 "Housekeeping").
6. **Map cache** (SPEC 7 step 4): store and load a run (`map_runs`, `map_clusters`, `map_points`) keyed by `input_hash`, with cluster labels per run (B-32). Cluster numbers are never keys outside their run.
7. **Injections** (SPEC 6 "No repeats"): record injected memory ids per session, read them back for a session, prune rows older than 30 days.
8. **Iteration** for `reindex`, `export` and `migrate`: stream every memory with its provenance and anchors, in a stable order, without loading the whole store into memory.

## Acceptance tests and other tests

- **AT-5**: two reader threads and one writer thread against one database file for 60 seconds; no lock error reaches any of them. The test runs its full 60 seconds.
- **AT-11** (store part): a stamp mismatch blocks vector and keyword search with `index_mismatch`; after the reindex path writes the new stamp, search works.
- **AT-12** (store part): a memory stored in project A is not returned by a project-B search; it is returned with scope `all`. Two profiles are two database files and never see each other.
- **D-2 and D-3**: exact vector search matches a brute-force scan on random unit vectors (both scopes, at least 1,000 chunks, the top 10 identical); FTS5 finds identifiers containing `.`, `::`, `_` and `-`.
- **SPEC 4 rules**: forget cascade (every dependent table emptied), shared memories refused, supersede in one transaction, migrations from an older schema version with a backup file written, injection pruning, map-cache round trip.
- **Budget** (SPEC 11): measure, in a release-profile test or example, project-scoped and cross-project search at 100,000 chunks of 768 dimensions with random unit vectors, 95th percentile. Report the numbers against 250 ms and 350 ms (the budgets include embedding, which is not yours; report the search part).

Use the fake embedder or random unit vectors in every test; no test here needs the real model.
