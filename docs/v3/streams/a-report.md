# Stream A report: `sm-store`

Branch `arman-tech/stream-a-store`, from `v3` at `ea82480`. Task `task_c71067fb0f26`.

## 1. Done when

**Yes.** Schema, migrations, exact vector search, FTS5 (full-text search, version 5) and links are built.

- All 50 storage tests pass on Windows and on Linux in Docker, including AT-5, AT-11, AT-12, D-2 and D-3.
- `cargo fmt --check` and `cargo clippy --workspace --all-targets --all-features -- -D warnings` are clean on both.
- Both SPEC 11 search budgets are met for the search part.

Two findings need the lead (section 9):

- **Project scope:** in a project that holds all 100,000 chunks, the store alone takes 243.7 ms at the 95th percentile (p95) against 250 ms. That leaves no room for embedding the query.
- **Bulk inserts:** they slow linearly with table size inside vec1, so bulk loads take quadratic time. `remember` is not at risk.

## 2. Files built

All under `crates/sm-store/` (about 8,300 lines of Rust).

- `Cargo.toml`: depends on `sm-core`, `rusqlite`, `libsqlite3-sys` (for the bundled SQLite header) and `serde_json`; `cc` to build; `tempfile` and `sm-core/test-support` for tests. Every version is the workspace pin.
- `build.rs`: compiles `vendor/vec1/vec1.c` from where it is, as a scalar object plus, on x86_64, an AVX2 (Advanced Vector Extensions 2) object chosen at run time (spike b).
- `src/lib.rs`: crate documentation (connections, vectors) and the public API.
- `src/vec1.rs`: the only `unsafe` code, which registers vec1 on each connection, with a safety comment.
- `src/conn.rs`: opening connections: WAL (write-ahead log) mode, 30-second busy timeout, foreign keys, `synchronous=FULL`, vec1 check.
- `src/schema.rs`: schema version 1 (every SPEC 4 table, indexes, triggers), forward migrations, `VACUUM INTO` backups.
- `src/store.rs`: `Store`: one writer connection, a reader pool, read and write transactions, backups, `iter_memories`.
- `src/memories.rs`: insert (batch, content-hash dedupe, supersede), revise, merge, forget (by id and by filter), get, find, list, memories for an anchor, the `Access` rule for shared memories.
- `src/projects.rs`: upsert by identity, rename, delete (cascading), per-project counts.
- `src/search.rs`: exact vector search in both scopes (`VectorIndex`), FTS5 keyword search, memory vectors for the map.
- `src/index.rs`: index stamp, pending chunks, `set_embeddings`, `reindex`.
- `src/links.rs`: links, the two-hop `related` walk, `LinkView.stale` (B-37).
- `src/maps.rs`: map cache: runs, clusters, points, region labels, pruning.
- `src/injections.rs`: injection records per session, 30-day pruning.
- `src/codec.rs`: vector and timestamp encoding, unit-length check.
- `src/error.rs`: SQLite errors mapped to `sm_core::error` codes (lock waits past the timeout become `timeout`).
- `src/tests/`: `mod.rs` (fixtures, seeded generator, exact vectors), `concurrency.rs`, `index.rs`, `injections.rs`, `keyword.rs`, `links.rs`, `maps.rs`, `memories.rs`, `migrations.rs`, `projects.rs`, `scope.rs`, `search_exact.rs`.
- `examples/search_budget.rs`: the SPEC 11 search budget measurement (release profile).
- `Cargo.lock` (root): the crates in section 10.

## 3. Test commands and exact results

**Windows** (Windows 11 Pro 10.0.26300, PowerShell, Rust 1.95.0 from `rust-toolchain.toml`), run from the worktree root:

| Command | Result |
| --- | --- |
| `cargo fmt --check` | exit 0, no diff |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | exit 0, no warnings |
| `cargo test -p sm-store --all-features -- --nocapture` | see below |
| `cargo test --workspace --all-features` | see below |

`cargo test -p sm-store --all-features -- --nocapture`:

```
AT-5: 60.0 s; writer 24978 operations, readers 59768 and 60248 operations
test result: ok. 50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.76s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test --workspace --all-features` (`sm_core` unit tests, `sm_store` unit tests, then the two crates' doc tests):

```
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
test result: ok. 50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.64s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**Linux** (Docker image `rust:1.95-trixie` on WSL2 kernel 6.6.87.2, x86_64, 32 logical cores), source mounted read-only. Run from Git Bash with `MSYS_NO_PATHCONV=1`, where `<worktree>` is this branch's worktree:

```
docker run --rm -v "<worktree>:/src:ro" -v sm-v3-cargo:/usr/local/cargo/registry -v sm-v3-target-a:/target -e CARGO_TARGET_DIR=/target -w /src rust:1.95-trixie bash -c "rustup component add clippy rustfmt && cargo test -p sm-store --all-features --locked -- --nocapture && cargo clippy -p sm-store --all-targets --all-features --locked -- -D warnings && echo CLIPPY_OK && cargo fmt --check && echo FMT_OK"
```

```
AT-5: 60.0 s; writer 10923 operations, readers 81640 and 81608 operations
test result: ok. 50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.51s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
CLIPPY_OK
FMT_OK
```

Clippy printed no warnings on Linux. The container exited 0.

## 4. Acceptance tests

All 50 `sm-store` tests pass on Windows and on Linux. None is ignored, and none needs the real model. The stream's required tests:

| Requirement | Test | Result |
| --- | --- | --- |
| AT-5: 2 readers and 1 writer for 60 s | `tests::concurrency::at5_two_readers_and_a_writer_for_sixty_seconds` | pass |
| AT-11 (store part): a stamp mismatch blocks both searches; after reindex they work | `tests::index::at11_stamp_mismatch_blocks_search_until_reindex` | pass |
| AT-11: a store with no stamp refuses to search | `tests::index::at11_unconfigured_store_refuses_to_search` | pass |
| AT-12 (store part): project scope versus scope `all` | `tests::scope::at12_project_scope_hides_other_projects` | pass |
| AT-12: two profiles are two database files | `tests::scope::at12_two_profiles_are_two_database_files` | pass |
| D-2: exact search equals brute force, ≥ 1,000 chunks, top 10 identical | `tests::search_exact::d2_exact_search_matches_brute_force_in_both_scopes` | pass |
| D-2: the exact test vectors are unit length in `f32` | `tests::d2_exact_vectors_are_unit_length_in_f32` | pass |
| D-2: pending chunks are not vector-searched | `tests::search_exact::d2_pending_chunks_are_not_vector_searched` | pass |
| D-2: query vectors are checked | `tests::search_exact::d2_query_vectors_are_checked` | pass |
| D-3: FTS finds `.`, `::`, `_` and `-` | `tests::keyword::d3_identifiers_with_punctuation_are_found` | pass |
| D-3: keyword text is one quoted phrase | `search::tests::d3_keyword_text_is_one_quoted_phrase` | pass |
| D-3: query syntax is searched as text | `tests::keyword::d3_query_syntax_is_searched_as_text` | pass |
| D-3: pending chunks are keyword-searched | `tests::keyword::d3_pending_chunks_are_keyword_searched` | pass |
| D-3: status filter, one hit per memory | `tests::keyword::d3_status_filter_and_one_hit_per_memory` | pass |
| D-3: titles are searched and follow revisions | `tests::keyword::d3_titles_are_searched_and_follow_revisions` | pass |
| SPEC 7: memory vectors are normalized chunk means | `tests::search_exact::s7_memory_vectors_are_normalized_chunk_means` | pass |

SPEC 4 rules, all pass:

- Vectors and schema:
  - `codec::tests::s4_vectors_are_768_little_endian_f32`
  - `codec::tests::s4_vectors_must_be_unit_length_and_finite`
  - `schema::tests::s4_check_constraints_list_every_enum_value`
- Index stamp:
  - `tests::index::s4_first_configuration_writes_the_stamp`
  - `tests::index::s4_invalid_stamps_are_refused`
- Injections:
  - `tests::injections::s4_injections_older_than_30_days_are_pruned`
  - `tests::injections::s4_injections_record_and_keep_the_latest_time`
- Links:
  - `tests::links::s4_head_only_settles_only_the_head`
  - `tests::links::s4_link_ends_are_checked`
  - `tests::links::s4_links_are_added_once_and_removed`
  - `tests::links::s4_links_of_reports_staleness_from_the_oracle`
  - `tests::links::s4_related_walks_links_and_anchors_two_hops`
- Map cache:
  - `tests::maps::s4_map_cache_round_trip`
  - `tests::maps::s4_map_runs_are_validated`
  - `tests::maps::s4_prune_keeps_the_newest_runs`
- Memories:
  - `tests::memories::s4_duplicates_return_the_existing_memory`
  - `tests::memories::s4_forget_by_filter_counts_what_it_deletes`
  - `tests::memories::s4_forget_cascades_to_every_dependent_table`
  - `tests::memories::s4_insert_stores_every_field`
  - `tests::memories::s4_invalid_items_fail_the_whole_batch`
  - `tests::memories::s4_iter_memories_streams_every_memory`
  - `tests::memories::s4_list_memories_filters_and_pages`
  - `tests::memories::s4_merge_supersedes_into_the_kept_memory`
  - `tests::memories::s4_revise_replaces_chunks_and_hash`
  - `tests::memories::s4_shared_memories_are_read_only_locally`
  - `tests::memories::s4_supersede_marks_the_old_memory_in_one_transaction`
- Migrations:
  - `tests::migrations::s4_failed_migration_leaves_the_previous_version`
  - `tests::migrations::s4_migration_writes_a_backup_first`
  - `tests::migrations::s4_new_database_is_created_without_a_backup`
  - `tests::migrations::s4_unreadable_schema_versions_are_refused`
- Projects:
  - `tests::projects::s4_delete_project_cascades`
  - `tests::projects::s4_project_status_counts_types_statuses_and_shared`
  - `tests::projects::s4_rename_project_checks_names`
  - `tests::projects::s4_upsert_keeps_the_name_and_refreshes_the_root`

The SPEC 11 search budgets are measured by `examples/search_budget.rs`, not by a test (section 5).

## 5. Measured budgets

**Machine:** AMD Ryzen 9 9950X3D (`AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`), 16 cores and 32 logical processors, 93.6 GB of memory. Windows 11 Pro 10.0.26300.

**Build:** release profile (`lto = "thin"`, `codegen-units = 1`). vec1 is compiled at C optimisation level 3, with its AVX2 object chosen at run time.

**SPEC 11 search budgets, search part only.** Command: `cargo run --release -p sm-store --example search_budget`.

- **What is timed:** `Store::nearest` (exact vector search) alone. Embedding the query, which the budgets also cover, is not the store's part.
- **Data:** 100,000 chunks × 768 dimensions of seeded random unit vectors, two chunks per memory.
- **Queries:** the default `recall` top 5, with 300 timed queries after 10 warm-up queries, each a fresh random unit vector. p50 is the median, and p95 the 95th percentile.

| Store | Scope | p50 | p95 | max | Budget (p95) | Result |
| --- | --- | --- | --- | --- | --- | --- |
| spread: 50,000 memories in 60 projects, rows interleaved | one project (about 1,667 chunks) | 8.4 ms | 9.9 ms | 11.5 ms | 250 ms | within |
| spread | all projects | 147.6 ms | 152.9 ms | 205.5 ms | 350 ms | within |
| single: 50,000 memories in 1 project | that project (all 100,000 chunks) | 235.2 ms | 243.7 ms | 247.8 ms | 250 ms | within, by 6.3 ms |
| single | all projects | 156.1 ms | 236.4 ms | 310.2 ms | 350 ms | within |

- **The single project is the worst case for project scope, and it leaves almost nothing for embedding.**
  - Project scope reads and scores every chunk of the project in Rust. With all 100,000 chunks in one project, the store alone takes 243.7 ms at p95.
  - SPEC 11 sets the project-scope budget "as for the hook". The hook row's measured basis on this machine is "45 to 95 ms to embed". End to end, `recall` in such a project would take about 290 to 340 ms against 250 ms.
  - The search part, which is what this stream was asked to measure, is within the budget. Section 9, item 2 lists the options.
- **The single store's all-projects tail is heavier than the spread store's** (p95 236.4 ms against 152.9 ms; max 310.2 ms). Both stores hold the same kind of data, and their medians agree (156.1 ms and 147.6 ms).
  - No timed query fell back to the full Rust scan. That is inferred from timing, not instrumented: a fallback costs a vec1 search plus a scan of all 100,000 chunks, about 150 + 235 ms, which is longer than the 310.2 ms maximum.
  - The tail is therefore variation in the fast path itself: the vec1 search and the re-scoring of its candidates. The machine is shared with other streams' workers, and outside load at that time cannot be ruled out. It was measured once.
- **Build time:**
  - The spread store took 1,557.3 s to build, with nothing else from this stream running.
  - The single store took 1,605.0 s, while the insert probes and the test gates ran alongside it (section 9, item 1).
  - Nothing else from this stream ran during either store's timed queries.
- **Concurrency (AT-5)**, 60 s with one writer and two readers, each with its own `Store` on one file:
  - Windows, debug test profile: writer 24,978 operations, readers 59,768 and 60,248, no error.
  - Linux: writer 10,923, readers 81,640 and 81,608, no error.

## 6. Decisions

Each entry gives what was decided, why, the evidence and the SPEC section it serves.

1. **Connections** (SPEC 3, SPEC 4; AT-5).
   - **What:** a `Store` holds one writer connection behind a mutex, plus a pool of reader connections opened on demand, with up to 4 kept idle.
     - Every write is one `BEGIN IMMEDIATE` transaction. Every read is one deferred transaction, so a search checks the index stamp and reads its vectors from one snapshot.
     - Every connection uses WAL mode, a 30-second busy timeout, `foreign_keys=ON` and `synchronous=FULL`, and checks that vec1 is registered (`SELECT vec1_info()`).
     - A lock still held after the timeout becomes `timeout`, with a next action. Any other SQLite error becomes `storage`. No raw lock error reaches a caller.
   - **Why:**
     - `BEGIN IMMEDIATE` takes the write lock at the start, so two writers never deadlock upgrading a read lock.
     - In WAL mode, readers do not wait for the writer.
     - The 30-second timeout is far longer than any single write transaction here.
   - **Evidence:** `at5_two_readers_and_a_writer_for_sixty_seconds` gives each thread its own `Store` on one file, as separate processes would, and fails on any error at all, not only a lock error. Figures are in section 5.
2. **Vectors stored twice** (SPEC 4 `chunks`, Cross-project search).
   - **What:** the embedding is kept in two places:
     - `chunks.embedding`: 768 little-endian `f32`, with `CHECK (length = 3072)`. `model_id` is set exactly when the embedding is.
     - The vec1 table `chunk_vectors`: flat index, cosine distance, kept in step by insert, update and delete triggers on `chunks`.
   - **Why:** project scope scans `chunks` through its `project_id` index without touching vec1. The vec1 table serves scope `all`. The triggers mean no code path can write one copy and forget the other.
   - **Evidence:** `s4_vectors_are_768_little_endian_f32`, `s4_forget_cascades_to_every_dependent_table` (counts `chunk_vectors` rows), `d2_exact_search_matches_brute_force_in_both_scopes`.
3. **Exact search in both scopes** (SPEC 4; D-2).
   - **Project scope:** reads `chunks WHERE project_id = ?` through its index and takes the dot product in Rust.
   - **Scope `all`:**
     - Asks vec1 for the nearest k = max(8 × (limit + excluded ids), 256) chunks, then re-scores, filters and collapses them in Rust.
     - The answer is returned only when it is provably exact: vec1 returned fewer than k chunks, or the `limit`-th memory's score exceeds (1 − the k-th distance) + 2.01 × 10⁻³ (twice the unit-length tolerance, plus 10⁻⁵ for float error).
     - Otherwise it scores every embedded chunk in Rust. An anchor filter skips vec1 and scores the anchor's memories directly.
   - **Ranking and scores:** score descending, then id. A memory's best chunk wins, and on a tied score the lower `seq`. Scores always come from the Rust dot product, never from vec1's distance, so both scopes report identical scores for the same chunk.
   - **Why:** status, type and exclusion filters can remove vec1's top chunks, and vec1's `f32` cosine can differ from the dot product in the last bits. The proof margin and the fallback keep scope `all` equal to brute force whatever the filters remove.
   - **Evidence:** `d2_exact_search_matches_brute_force_in_both_scopes`. It uses at least 1,000 chunks of exactly representable unit vectors (integers over 1024; see `tests/mod.rs`), so the store's `f32` scores and the integer brute force agree bit for bit. The top 10 are identical, with filters and exclusions, in both scopes.
4. **Unit-length check** (SPEC 4, SPEC 5).
   - **What:** every stored or query vector must be finite, with squared length within 10⁻³ of 1. A vector that fails is rejected with `invalid_input`.
   - **Why:** dot product equals cosine only for unit vectors, and the scope-`all` proof margin depends on the bound.
   - **Evidence:** `s4_vectors_must_be_unit_length_and_finite`, `d2_query_vectors_are_checked`.
5. **Keyword index** (SPEC 4 `chunks_fts`, SPEC 6 step 6; D-3).
   - **What:** a contentless FTS5 table (`content=''`, `contentless_delete=1`) with tokenizer `porter unicode61`, columns `title` and `text`, one row per chunk, its rowid equal to the chunk id. Triggers keep it in step, including on a title revision.
   - **Query:** always one quoted phrase, with internal `"` doubled. Each memory appears once, with its best-ranked chunk, ordered by bm25 rank, then id.
   - **Pending chunks:** chunks waiting for an embedding are keyword-searchable at once (B-27).
   - **Why:** a contentless table does not store the text a second time. `contentless_delete=1` lets forget and revise delete rows.
   - **Known limit:** a phrase that spans a chunk boundary is not found.
   - **Evidence:** `d3_identifiers_with_punctuation_are_found` (`.`, `::`, `_`, `-`), `d3_query_syntax_is_searched_as_text`, `d3_titles_are_searched_and_follow_revisions`, `d3_pending_chunks_are_keyword_searched`, `d3_status_filter_and_one_hit_per_memory`.
6. **Index stamp and reindex** (SPEC 4 "Model stamp"; AT-11; B-27).
   - **Stamp:** `configure_index` records the stamp on a store that has none. When the stored stamp differs, or none is configured, every vector search, keyword search and `memory_vectors` call returns `index_mismatch`.
   - **Reindex:**
     - Writes a backup first.
     - Re-chunks memories in pages of 256 by id, one transaction per page, with the caller's chunker running outside any transaction.
     - Rebuilt chunks are stored as pending, and `set_embeddings` fills them later through the bulk queue.
     - A memory forgotten or revised (its `content_hash` changed) between reading a page and writing it is skipped.
     - The new stamp is written last, so an interrupted reindex leaves searches blocked rather than wrong.
   - **`set_embeddings`:** refuses a model other than the configured one and a mismatched index. It updates only chunks still pending, so a chunk replaced meanwhile is skipped.
   - **Evidence:** `at11_stamp_mismatch_blocks_search_until_reindex`, `at11_unconfigured_store_refuses_to_search`, `s4_first_configuration_writes_the_stamp`, `s4_invalid_stamps_are_refused`.
7. **Deduplication** (B-29, B-14).
   - **What:** an input is a duplicate when:
     - its id already exists; or
     - a memory in the same project has the same `content_hash`, whatever its status. The earliest by `created_at`, then id, is returned.
   - Within a batch, later items see earlier ones. The reply gives outcome `Duplicate` and the existing id.
   - Any invalid item fails the whole batch, and nothing is written.
   - **Why:** matching across statuses means a retracted or superseded copy is not silently stored again.
   - **Evidence:** `s4_duplicates_return_the_existing_memory`, `s4_invalid_items_fail_the_whole_batch`.
8. **Chunks:** a stored memory has at least one chunk, numbered 0, 1, 2 … with no gaps.
   - **Why:** keyword hits, vector hits and `memory_vector` report a chunk by `seq`, and the walk and the map assume a memory has a vector once embedded.
   - **Evidence:** `s4_invalid_items_fail_the_whole_batch` (a chunk with the wrong `seq`, a memory with no chunks).
9. **Supersede and merge** (SPEC 4 "Types and status"; B-31).
   - **Supersede:** inserting with `supersedes_id` marks the older memory `superseded` in the same transaction.
   - **Merge:** each merged memory becomes `superseded`, with its `supersedes_id` set to the kept memory, as B-31 says. Section 8 notes how this conflicts with SPEC 4.
   - **Evidence:** `s4_supersede_marks_the_old_memory_in_one_transaction`, `s4_merge_supersedes_into_the_kept_memory`.
10. **Shared memories are read-only locally** (SPEC 4 "origin").
    - **What:** every write takes an `Access`:
      - `Local` for the daemon and the tools.
      - `Sync` for `sm-share`, the only writer that may change shared memories.
    - **Under `Local`:**
      - Revise, merge, forget by id and supersede refuse a shared memory with `shared_read_only`, and write nothing.
      - Forget by filter skips shared memories and counts them in `ForgetCount.shared_skipped`.
    - **Why:** forget by id names a memory, so refusing it is the honest answer. A filter only describes memories, and failing a whole bulk forget because one shared memory matches would make bulk forget unusable in a project that syncs.
    - **Evidence:** `s4_shared_memories_are_read_only_locally`, `s4_forget_by_filter_counts_what_it_deletes`.
11. **Forget** (SPEC 4; B-30).
    - **Forget by id:** one transaction, all or nothing. It removes provenance, chunks, keyword rows, vectors, anchor mappings, links, map points and injections, through foreign keys and triggers.
    - **Forget by filter:**
      - `count_forget` gives the count the daemon needs before it asks for `confirm`.
      - `forget_by_filter` counts again and deletes in one transaction, and fails if the two disagree.
    - **Evidence:** `s4_forget_cascades_to_every_dependent_table` empties every dependent table, virtual tables included. `s4_forget_by_filter_counts_what_it_deletes`.
12. **Links** (SPEC 4 `links`, SPEC 6 "Related"; D-8).
    - **Adding:** an identical link (same ends and relation) returns the stored one with `created` false.
    - **Ends:**
      - A memory end must exist (`not_found`) and be in the link's project (`invalid_input`).
      - The two ends must differ.
      - Links go when either memory end is forgotten or the project is deleted.
    - **`related`:** walks two hops in either direction over `links` and over shared anchors (`memory_anchors`). An edge through a shared anchor is reported as a synthetic link with extractor `memory_anchors`.
    - **Evidence:** `s4_links_are_added_once_and_removed`, `s4_link_ends_are_checked`, `s4_related_walks_links_and_anchors_two_hops`.
13. **`LinkView.stale`** (B-37).
    - **What:** the store does not walk git ancestry. `links_of` takes an ancestry oracle, `Fn(commit) -> Option<bool>`, and sets `stale = oracle(commit).map(|is_ancestor| !is_ancestor)`.
    - The shipped oracle, `head_only(head)`, answers only what the head settles:
      - A commit equal to the head, ignoring case, or a hexadecimal prefix of it of at least 7 characters, is not stale.
      - Anything else, or an unknown head, leaves `stale` absent.
    - **Why:** B-37 forbids running git. Reading git's object store needs a pack-file reader, which is out of scope. The oracle lets a later stream supply a real ancestry check without changing the store.
    - **Evidence:** `s4_head_only_settles_only_the_head`, `s4_links_of_reports_staleness_from_the_oracle`.
14. **Map cache** (SPEC 7 step 4; B-32).
    - **Runs:** keyed by a unique `input_hash`. `put_map_run` returns the existing run with `created` false for a known hash.
    - **Validation:** a run needs an input hash and starts without region labels. Its regions are listed once each. Each memory has one point, at a finite position, and in a region the run lists.
    - **Labels:** stored against (run, region), so a new run starts unlabelled. `prune_map_runs` keeps the newest N runs per project, or across all projects.
    - **Evidence:** `s4_map_cache_round_trip`, `s4_map_runs_are_validated`, `s4_prune_keeps_the_newest_runs`.
15. **Memory vectors for the map** (SPEC 7 step 1).
    - **What:** `memory_vectors` and `memory_vector` return the mean of a memory's embedded chunk vectors, re-normalized. If the chunks cancel out, the first chunk stands for the memory.
    - **Why:** SPEC 7 projects one point per memory.
    - **Evidence:** `s7_memory_vectors_are_normalized_chunk_means` (pending chunks left out, cancelling chunks, scope and exclusion). The AT-11 and AT-12 tests show both calls blocked by a stamp mismatch and confined to the project scope.
16. **Injections** (SPEC 6 "No repeats"): one row per (session, memory), keeping the latest time. Rows older than 30 days are pruned.
    - **Evidence:** `s4_injections_record_and_keep_the_latest_time`, `s4_injections_older_than_30_days_are_pruned`.
17. **Migrations and backups** (SPEC 4 "Housekeeping").
    - **Migrations:** the schema version lives in `meta`, and each migration runs forward in its own `BEGIN IMMEDIATE` transaction. A newer, unknown version is refused.
    - **Backups:** an existing database is backed up before its first migration. A new database is created without one.
      - Backups are written with `VACUUM INTO`, named `memory-<UTC time>-<reason>.db`, and never overwritten (a `-1`, `-2` suffix is added).
      - Nothing prunes old backups.
    - **Evidence:** `s4_new_database_is_created_without_a_backup`, `s4_migration_writes_a_backup_first`, `s4_failed_migration_leaves_the_previous_version`, `s4_unreadable_schema_versions_are_refused`.
18. **Iteration** for `reindex`, `export` and `migrate`: `iter_memories` streams memories in id order, in pages read in their own transactions, so the whole store is never in memory.
    - **Evidence:** `s4_iter_memories_streams_every_memory`.

## 7. Deviations

1. **Backups use `VACUUM INTO`, not a file copy.** SPEC 4 "Housekeeping" and the stream file say a migration or reindex "copies the database file".
   - **Why:** in WAL mode a file copy taken while another process writes can miss committed pages that are still in the `-wal` file. `VACUUM INTO` writes a complete, consistent database from one read transaction.
   - **Result:** the same as the SPEC intends, one restorable database file in `backups_dir()`, only compacted.
2. **Scope `all` does not rely on vec1 alone.** The stream file says "vec1 in exact mode across all chunks, then the same collapse". vec1 supplies the first candidates, but:
   - scores come from the Rust dot product;
   - a full Rust scan takes over whenever the filters leave the vec1 candidates unable to prove the top `limit` (decision 3);
   - an anchor filter skips vec1.

   The result is still exactly the brute-force answer, which is what D-2 requires. vec1's approximate index is never built.
3. **Forget by filter skips shared memories instead of refusing.** The stream file says to "refuse to forget a memory whose origin is `shared`". Forget by id does refuse. Forget by filter skips shared memories and reports how many it skipped (decision 10).

No conflict between RULES.md or CONTRACTS.md and the stream file arose.

## 8. Contract changes requested

None is required: `sm-core` was not changed and nothing in it blocked this stream. Points for the lead, which may want a B-n entry:

1. **`supersedes_id` has two directions.**
   - SPEC 4 (line 378) defines `supersedes` as "Identifier of the memory this one replaces", so a new memory points at the old one. Insert follows it.
   - B-31 has each merged (old) memory's `supersedes_id` point at the kept memory, the opposite direction. Merge follows it, as the newer decision.
   - So one column means "replaces" or "replaced by" depending on which path wrote it.
   - Suggested fix: a B-n entry that either redefines the column as "the memory this one is linked to by supersession" plus status, or records merges differently, for example a `supersedes` link.
2. **`Access::{Local, Sync}`.** Stream E's daemon passes `Local`. Stream F's `sync` must pass `Sync` to update or delete shared memories.
3. **Store-defined input types.** `MemoryInput`, `NewChunk` (embedded or pending), `Revision`, `ContentChange`, `MemoryListFilter`, `NewMapRun` and the results `Inserted`, `ForgetCount`, `LinkAdded`, `WalkHit` and `PendingChunk` are `sm-store` types, as CONTRACTS.md section 2 allows (there is no store trait).
4. **Behaviour the daemon and `sm-share` should know:**
   - Link memory ends must be in the link's project.
   - Superseding a shared memory is refused under `Local`.
   - Every search needs `configure_index` first.
   - `links_of` takes an ancestry oracle (`head_only` is provided).
   - `memory_vectors` and `memory_vector` exist for the map.

## 9. Unfinished or unverified

1. **Bulk loading slows as the store grows, and the cause is inside vec1.** SPEC 11 has no bulk-insert budget. The lead asked for one measurement of insert cost against table size; this is it.
   - **Batched inserts:** a temporary probe (not committed), release profile, one store, one project, random unit vectors, two chunks per memory. Each `insert_memories` call took 500 memories (1,000 chunks) in one transaction. The probe ran three times, changing only the `chunks_ai` trigger. Mean milliseconds per chunk over each 5,000 chunks:

     | `chunks_ai` trigger | 0–5k | 5–10k | 10–15k | 15–20k | 20–25k | 25–30k | 30,000 chunks |
     | --- | --- | --- | --- | --- | --- | --- | --- |
     | as shipped (vec1 and FTS inserts) | 0.66 | 2.20 | 3.83 | 5.41 | 7.17 | 8.91 | 140.9 s |
     | without the vec1 insert | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 1.5 s |
     | without the FTS insert | 0.64 | 2.18 | 3.71 | 5.27 | 7.08 | 8.95 | 139.2 s |

   - **Single-chunk inserts** (one memory of one chunk per transaction, the shape of `remember`), 20 at each size:

     | Chunks in the store | Median | Max |
     | --- | --- | --- |
     | 10,020 | 4.46 ms | 7.10 ms |
     | 20,040 | 7.83 ms | 9.45 ms |

     Both probes ran on the machine in section 5 while the budget example was building its second store, so their absolute times carry some noise. The comparison between variants does not depend on it: the gap is a factor of about 100.

   - **Cause:** the insert into the vec1 table `chunk_vectors`, which is configured with a flat index. Without it, the rest of the store costs a flat 0.04 to 0.05 ms per chunk: the `chunks` row, its indexes, FTS5 and the other triggers. The FTS insert costs nothing measurable.
     - Each vec1 insert costs about 0.33 ms more for every 1,000 chunks already in the table, so a bulk load takes time quadratic in its size.
     - The single-chunk slope (0.34 ms per 1,000 chunks) matches the batched one, so the cost is per insert, not per transaction.
     - Reading the insert path in `vendor/vec1/vec1.c` did not locate the responsible code. The search for a block to append to (`VEC1_SQL_SEARCH_IDX_FOR_INSERT`) runs only when a bucket's builder is empty, not on every insert, so it does not explain the slope. Finding the line needs a profiler inside vec1.
   - **Effect:**
     - `remember` (SPEC 11: at most 2 s for one memory of up to one chunk) is not at risk. Extrapolated linearly to 100,000 chunks, the store's part is about 35 ms. That figure is extrapolated, not measured.
     - Bulk paths are quadratic: `import`, `migrate`, and `set_embeddings` after a `reindex`. `set_embeddings` reaches vec1 through the `chunks_au_embedding` trigger, which is not measured separately. Building the 100,000-chunk benchmark store took 1,557 s (section 5).
   - **Left for the lead:** `vendor/vec1` stays unchanged (B-7), and sm-store has no fix:
     - Batching is already 1,000 chunks per transaction, and the cost is per insert.
     - No pragma was tried: the growth sits in the vec1 insert, and the store's other writes on the same connection stay flat.
     - `synchronous=FULL` is unchanged.
     - One option for the lead: SPIKE b inserted 100,000 vectors into a vec1 table with no index in one transaction in 0.60 s, then rebuilt it to flat cosine in 2.22 s. A bulk path could load first and rebuild after. That changes how `chunk_vectors` is maintained, so it needs a decision first.
2. **Project scope in a project of 100,000 chunks nearly fills the budget on its own** (section 5). The store's part is 243.7 ms at p95 against 250 ms. Adding the 45 to 95 ms that SPEC 11 gives for embedding the query puts `recall` over the budget end to end.
   - The spread store, where the project holds 1,667 chunks, takes 9.9 ms, so only a project holding most of a large store is affected.
   - Two options for the lead, neither tried nor measured, since the done-when asks for the search part only:
     - **Faster reads for the scan.** The reader connections use SQLite's defaults: about 2 MB of page cache and no memory-mapped reads. Every scan reads about 300 MB of chunk rows through that cache. `PRAGMA mmap_size` on reader connections might cut the copying. It would be a measured decision in sm-store's own code.
     - **Use vec1 when a project holds most of the store**, with the proof margin and fallback that scope `all` already uses. vec1 measured 147.6 to 156.1 ms at the median here, which still leaves little room for embedding.
3. **The budget was measured on Windows only.** The Linux check (section 3) runs the tests and clippy, not the 100,000-chunk example.
4. **Temporary folders leaked by earlier test runs.** Before the `Fixture` drop-order fix, a test's database was still open when its temporary folder was removed. Windows refused the delete and `tempfile` ignores that failure. 79 leaked `.tmp*` folders, 63.3 MB in all, remain in `%LOCALAPPDATA%\Temp`, outside the repository. They were left in place because deleting files needs the owner's permission under this machine's rules. Current runs remove their folders: the three test runs at 14:48, 14:49 and 14:53 on 16 November left none.
5. **No backup retention.** Backups accumulate in `backups_dir()`. Nothing in SPEC 4 says how many to keep, and pruning is left to a later decision.

## 10. Licences

Crates added to `Cargo.lock` by this stream, from `cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked`. All are on the permissive list; none is flagged.

| Crate | Version | Licence |
| --- | --- | --- |
| cc | 1.11.2 | MIT OR Apache-2.0 |
| fallible-iterator | 0.3.0 | MIT/Apache-2.0 |
| fallible-streaming-iterator | 0.1.9 | MIT/Apache-2.0 |
| find-msvc-tools | 0.1.15 | MIT OR Apache-2.0 |
| foldhash | 0.2.0 | Zlib |
| hashlink | 0.12.2 | MIT OR Apache-2.0 |
| libsqlite3-sys | 0.40.0 | MIT |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 |
| rsqlite-vfs | 0.2.0 | MIT |
| rusqlite | 0.43.0 | MIT |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| smallvec | 1.16.5 | MIT OR Apache-2.0 |
| sqlite-wasm-rs | 0.6.3 | MIT |
| vcpkg | 0.2.15 | MIT/Apache-2.0 |
| sm-store | 3.0.0 | MIT (this crate) |

`sqlite-wasm-rs` (and its `rsqlite-vfs`) is in the lock file because `rusqlite` 0.43 declares it for WebAssembly targets only. `cargo tree -i sqlite-wasm-rs` prints nothing for the host target, so it is not built on Windows or Linux. `foldhash` is new because `hashlink` turns on `hashbrown`'s default hasher. The bundled SQLite and vec1 are public domain.
