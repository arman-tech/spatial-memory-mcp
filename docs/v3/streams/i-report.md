# Package I report: `sm-store` and `sm-share` follow-ups (phase 3)

Branch `arman-tech/stream-i-store`, from `v3` at 93044bb.

Every figure below was measured on an AMD Ryzen 9 9950X3D central processing unit (CPU): 16 cores, 32 logical processors, 93.6 GB of memory. Windows figures are from Windows 11 Pro 10.0.26300. Linux figures are from Docker Desktop, running the `rust:1.95-trixie` image (Debian 13). The build profile is given with each figure. Packages H (`sm-daemon`) and J (`sm-cli`) were building on the same machine throughout, so a timing that looked disturbed was repeated, and the repeat is given too.

## 1. Done when

Yes. All eight Build items are fixed and tested, and B-74 and B-75 are fixed in `sm-store` and measured before and after. Both gates pass on Windows and on Linux, for the full workspace. This report is committed.

## 2. Files built

- `crates/sm-store/src/memories.rs`: the following additions.
  - The liveness rule (B-87) on `forget`, `forget_by_filter` and local `revise`.
  - The refusal to make a held memory `active`.
  - `InsertOptions` with `insert_memories_with`.
  - `ReviseOptions` with `revise_memory_with`, which takes a sync mark and a keep-`updated_at` option.
  - `MarkRow` with `list_marks`.
  - The constant `UNRESOLVED_SUPERSEDES`.
- `crates/sm-store/src/lib.rs`: re-exports `InsertOptions`, `ReviseOptions`, `MarkRow` and `UNRESOLVED_SUPERSEDES`.
- `crates/sm-store/src/schema.rs`: `VEC1_OPEN_BLOCKS_SQL`, a partial index of the vec1 blocks that have room (B-74), created in schema version 1.
- `crates/sm-store/src/conn.rs`: `READER_MMAP_BYTES` and `open_reader`, so that reader connections read the file through a memory map (B-75).
- `crates/sm-store/src/store.rs`: the reader pool opens connections with `open_reader`.
- `crates/sm-store/src/search.rs`: two changes (B-75).
  - Project scope tests each chunk against the list of its project's memories that pass the filters, built once.
  - Ranking partitions out the first `limit` hits before sorting them.
- `crates/sm-store/src/tests/liveness.rs`: new, 10 tests.
- `crates/sm-store/src/tests/vec1_blocks.rs`: new, 2 tests.
- `crates/sm-store/src/tests/concurrency.rs`: 1 new test.
- `crates/sm-store/src/tests/mod.rs`: registers the two new modules.
- `crates/sm-store/examples/bulk_insert.rs`: new. Measures embedding and inserting 2,000, 10,000 and 50,000 chunks, the median single insert at each size, and all-projects search.
- `crates/sm-share/src/share.rs`: refuses a memory that was merged into another (B-89).
- `crates/sm-share/src/sync.rs`: two changes.
  - It lists marks with `list_marks`, and loads whole memories only for the known files it reads.
  - It revises a changed file's memory with its mark in one transaction (`revise_memory_with`).
- `crates/sm-share/src/export.rs`: import keeps `updated_at` on supersession targets and at the status restore.
- `crates/sm-share/src/tests/share.rs`, `tests/sync.rs`: 3 new tests.
- `crates/sm-share/src/tests/export_import.rs`: the B-77 exemption for `updated_at` is removed from the round-trip comparisons.
- `crates/sm-share/src/tests/budgets.rs`: the sync budget also times `list_marks`.
- `docs/v3/streams/i-report.md`: this report.

Commits:

- e63b7ad (B-87, B-90, B-92, B-93)
- c13fdd9 (B-89)
- 7ccd5c2 (B-90, B-92, B-93)
- 1008eb5 (B-74)
- 9f26330 (B-75)
- the report commit.

## 3. Test commands and exact results

### Windows 11 (PowerShell)

`cargo fmt --check`: exit 0.

`cargo clippy --workspace --all-targets --all-features -- -D warnings`: exit 0.

`cargo test --workspace --all-features --no-fail-fast`: exit 0. Each line is labelled with its test binary: a crate's name for its unit tests, the file for an integration test, `doc-tests` and the crate for its documentation tests. The binaries run in this order: `sm-capture`, `sm-cli` (unit tests, `main.rs`, then `autostart` to `setup`), `sm-core`, `sm-daemon` (unit tests, then `acceptance`, `hooks` and `socket`), `sm-embed` (unit tests, then `real_model`), `sm-share`, `sm-spatial` (unit tests, then `at6_determinism`, `at7_quality` and `s7_pipeline`), `sm-store`.

```
sm_capture (src\lib.rs): test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
sm_cli (src\lib.rs): test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
spatial_memory (src\main.rs): test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart (tests\autostart.rs): test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 23.25s
hook_budgets (tests\hook_budgets.rs): test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 65.24s
hook_fixtures (tests\hook_fixtures.rs): test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.23s
mcp_proxy (tests\mcp_proxy.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
ops (tests\ops.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
plugin (tests\plugin.rs): test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.54s
setup (tests\setup.rs): test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
sm_core (src\lib.rs): test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon (src\lib.rs): test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
acceptance (tests\acceptance.rs): test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.82s
hooks (tests\hooks.rs): test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.22s
socket (tests\socket.rs): test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.58s
sm_embed (src\lib.rs): test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
real_model (tests\real_model.rs): test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.22s
sm_share (src\lib.rs): test result: ok. 92 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 26.29s
sm_spatial (src\lib.rs): test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
at6_determinism (tests\at6_determinism.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.03s
at7_quality (tests\at7_quality.rs): test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
s7_pipeline (tests\s7_pipeline.rs): test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.72s
sm_store (src\lib.rs): test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.85s
doc-tests sm_capture: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_cli: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_core: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_daemon: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_embed: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_share: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_spatial: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_store: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### Linux (Docker, `rust:1.95-trixie`, source mounted read-only, target in volume `sm-v3-target-i`)

The container reports `nproc` 32 and kernel `Linux 6.6.87.2-microsoft-standard-WSL2`. It ran `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` and `cargo test --workspace --all-features --locked --no-fail-fast`. The results were `FMT_EXIT=0`, `CLIPPY_EXIT=0` and `TEST_EXIT=0`. The binaries are in the same order as on Windows.

Some counts differ from Windows because some tests are compiled for one platform only: `sm_cli` 36 against 38, `autostart` 12 against 11, `sm_core` 97 against 96, and `sm_daemon` 11 against 8. `real_model` passes in 0.00 s because the container has no model folder, so the test returns at once, as CLAUDE.md "Tests and the model" says it should.

```
sm_capture (src/lib.rs): test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
sm_cli (src/lib.rs): test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
spatial_memory (src/main.rs): test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart (tests/autostart.rs): test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.04s
hook_budgets (tests/hook_budgets.rs): test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.56s
hook_fixtures (tests/hook_fixtures.rs): test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.46s
mcp_proxy (tests/mcp_proxy.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
ops (tests/ops.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
plugin (tests/plugin.rs): test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
setup (tests/setup.rs): test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_core (src/lib.rs): test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon (src/lib.rs): test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
acceptance (tests/acceptance.rs): test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.52s
hooks (tests/hooks.rs): test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.12s
socket (tests/socket.rs): test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.50s
sm_embed (src/lib.rs): test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
real_model (tests/real_model.rs): test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_share (src/lib.rs): test result: ok. 92 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 56.01s
sm_spatial (src/lib.rs): test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.66s
at6_determinism (tests/at6_determinism.rs): test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.17s
at7_quality (tests/at7_quality.rs): test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.53s
s7_pipeline (tests/s7_pipeline.rs): test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.03s
sm_store (src/lib.rs): test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.59s
doc-tests sm_capture: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_cli: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_core: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_daemon: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_embed: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_share: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_spatial: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests sm_store: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

## 4. Acceptance tests

Package I owns no acceptance test. These acceptance tests run over code this package changed, and each passes on both Windows and Linux:

- **AT-5:** `at5_two_readers_and_a_writer_for_sixty_seconds`. It now runs with memory-mapped readers.
- **AT-8:** the `at8_*` tests in `sm-share/src/tests/migrate.rs`. Migrate's inserts now go through the B-74 index.
- **AT-9:** `at9_share_writes_fixture_credentials_redacted_with_kinds` and `at9_share_refuses_credential_in_tag_or_anchor`. Share has a new refusal.
- **AT-11:** `at11_stamp_mismatch_blocks_search_until_reindex` and `at11_unconfigured_store_refuses_to_search`. Search was changed.
- **AT-12:** `at12_project_scope_hides_other_projects` and `at12_two_profiles_are_two_database_files`. The project-scope query was changed.

Tests added, by rule:

- **B-87 (SPEC 4):**
  - `s4_forget_restores_a_target_left_with_no_live_superseder`
  - `s4_forget_never_restores_a_retracted_merged_or_shared_target`
  - `s4_only_a_superseder_of_the_same_project_holds_a_target`
  - `s4_forget_by_filter_restores_targets_too`
  - `s4_retracting_a_superseder_locally_restores_its_target`
  - `s4_revise_refuses_to_make_a_held_memory_active`
- **`updated_at` (B-92, B-93):**
  - `s4_insert_can_keep_the_target_updated_at`
  - `s4_revise_can_keep_updated_at`
  - `s15_export_import_export_round_trips`, now byte-identical with no exemption
  - `s15_an_interrupted_import_resumes` and `s15_an_import_stopped_before_restoring_statuses_resumes`, with the same exemption removed
- **Sync (B-90, B-93):**
  - `s9_revise_records_a_sync_mark_in_the_same_transaction`
  - `s9_list_marks_matches_the_whole_memories`
  - `s9_sync_adopts_and_revises_the_authors_copy_with_its_mark`
  - `s9_sync_reports_a_known_file_whose_memory_was_forgotten_while_it_ran`
- **Share (B-89):** `s9_share_refuses_a_locally_merged_memory`
- **B-74:**
  - `s4_vec1_finds_a_block_with_room_in_one_index_lookup`, which checks the vendored statement, the default `blocksize-min` and the query plan
  - `s4_vec1_blocks_stay_consistent_with_the_index_of_blocks_with_room`, which runs inserts, forgets and revisions, then `PRAGMA integrity_check`
- **B-75:** `s4_readers_map_the_database_and_the_writer_does_not`. The existing `d2_exact_search_matches_brute_force_in_both_scopes` covers exactness after the search change: ties, limits of 0, 1, 50 and 2,000, every filter, the anchor, exclusion, and both scopes.

## 5. Measured budgets

All figures are from release builds.

### B-75: vector search at 100,000 chunks

Measured with `cargo run --release -p sm-store --example search_budget`: 100,000 chunks of 768 dimensions, top 5, 300 timed queries after 10 warm-up. p50 and p95 are the 50th and 95th percentiles.

On Windows 11:

| Store | Scope | Before: p50 / p95 / max | After: p50 / p95 / max | Repeat after: p50 / p95 / max |
| --- | --- | --- | --- | --- |
| Spread, 60 projects | Project (about 1,667 chunks) | 9.3 / 9.7 / 15.2 ms | 0.9 / 1.8 / 2.8 ms | 0.9 / 0.9 / 1.1 ms |
| Spread, 60 projects | All | 154.4 / 160.3 / 168.6 ms | 30.7 / 37.7 / 49.7 ms | 29.8 / 30.4 / 30.8 ms |
| Single project | Project (all 100,000 chunks) | 244.7 / **257.5** / 345.1 ms (over the 250 ms budget) | 67.9 / **71.1** / 86.2 ms | 67.9 / 70.2 / 73.7 ms |
| Single project | All | 150.2 / 155.0 / 159.6 ms | 28.8 / 29.7 / 30.3 ms | 28.5 / 28.9 / 29.6 ms |

The "before" figures are from the current code at the start of this item, after B-74. The figures in DECISIONS B-75 are older: 243.7 ms at p95 for the single project.

Which change does what, for the single project at project scope, on Windows:

| Configuration | p50 / p95 |
| --- | --- |
| Before (join, page cache, full sort) | 244.7 / 257.5 ms |
| Readers memory-mapped, nothing else | 79.0 / 80.6 ms |
| Memory-mapped and top-`limit` selection, join kept | 75.3 / 77.4 ms; repeat 72.9 / 75.0 ms |
| Memory-mapped, top-`limit` and memories listed once (committed) | 67.9 / 71.1 ms; repeat 67.9 / 70.2 ms |

The memory map gives most of the gain on every scope. All-projects search also falls from about 155 ms to about 30 ms, because vec1's scan reads its blocks through the map.

A probe isolated the database query from the rest of the search: the same database, the query alone, with each query shape and connection setting. It was a temporary example and is not committed. These are p50 / p95 for reading the rows and scoring them:

| Query shape | Page cache (default) | `mmap_size` 2 GiB | `cache_size` 512 MiB |
| --- | --- | --- | --- |
| Join chunk to memory (before) | 218.8 / 222.1 ms | 105.3 / 131.9 ms | 66.7 / 67.7 ms |
| Chunks alone, no filter (a floor) | 195.3 / 201.3 ms | 33.0 / 41.5 ms | 44.2 / 45.6 ms |
| Memories outer, chunks by `(memory_id, seq)` | 220.4 / 304.8 ms | 57.1 / 59.5 ms | 60.6 / 61.4 ms |
| Chunks with `memory_id IN (memories that pass)` (committed) | 253.5 / 278.5 ms | 54.6 / 56.0 ms | 69.4 / 72.0 ms |
| Table scan, then join | 255.1 / 504.0 ms | 58.7 / 59.7 ms | 64.3 / 65.4 ms |

Dot products alone over 100,000 vectors already in memory take 16.6 ms at p50.

Linux in Docker, after the change, release build. It is the same example, run in the gate's container (kernel `6.6.87.2-microsoft-standard-WSL2`, 32 logical processors). No "before" was taken on Linux.

| Store | Scope | p50 / p95 / max | Budget |
| --- | --- | --- | --- |
| Spread, 60 projects | Project (about 1,667 chunks) | 0.9 / 0.9 / 1.0 ms | 250 ms: within |
| Spread, 60 projects | All | 26.5 / 27.3 / 27.9 ms | 350 ms: within |
| Single project | Project (all 100,000 chunks) | 65.1 / **67.1** / 69.2 ms | 250 ms: within |
| Single project | All | 27.1 / 27.5 / 27.8 ms | 350 ms: within |

Docker's file share is slower than a native disk. Here only the source sits on it, mounted read-only. Each example writes its database to the system temporary directory, which is the container's own `/tmp`, so the timings above are not taken through the file share.

### B-74: bulk inserts into vec1

Measured with `cargo run --release -p sm-store --example bulk_insert`, on Windows 11. "Before" is the base commit 93044bb, built from a `git archive` copy. "After" is commit 1008eb5.

| Measure | Chunks | Before | After |
| --- | --- | --- | --- |
| Embed: `set_embeddings`, 4 per call, onto pending chunks | 2,000 | 0.94 s (0.472 ms per chunk) | 0.79 s (0.396 ms per chunk) |
| | 10,000 | 23.83 s (2.383 ms per chunk) | 3.97 s (0.397 ms per chunk) |
| | 50,000 | 467.39 s (9.348 ms per chunk) | 20.16 s (0.403 ms per chunk) |
| Insert: `insert_memories`, 1,000 chunks per transaction | 2,000 | 0.49 s | 0.11 s |
| | 10,000 | 14.36 s | 0.57 s |
| | 50,000 | 424.33 s | 3.76 s |
| Single one-chunk insert, median, at the size shown | 2,000 | 2.67 ms | 1.18 ms |
| | 10,000 | 5.59 ms | 1.28 ms |
| | 50,000 | 16.32 ms | 1.31 ms |
| All-projects search at 50,000 chunks, p95: after embedding / after inserts | 50,000 | 83.5 / 89.1 ms | 76.9 / 76.4 ms |

After the change, the cost per chunk is flat: 0.396 to 0.403 ms from 2,000 to 50,000 chunks. Blocks average 6.0 vectors, which is vec1's 16 KiB block size.

### Sync of 10,000 shared files (B-90, B-93)

Measured with `cargo test -p sm-share --all-features --release s9_budget -- --nocapture`, on Windows 11. "Before" is the base commit 93044bb, built from a `git archive` copy. "After items 4 and 5" is commit 7ccd5c2, measured before B-74. "After B-74 and B-75" is commit 9f26330.

| Measure | Before | After items 4 and 5 | After B-74 and B-75 |
| --- | --- | --- | --- |
| Second sync, nothing changed, 3 runs | 282 / 286 / 286 ms | 122 / 126 / 120 ms | 89 / 87 / 87 ms |
| Third sync, one file changed | 329 ms | 145 ms | 112 ms |
| Fourth sync, all 10,000 files changed | 34,948 ms | 19,391 ms | 18,483 ms |
| Fifth sync, nothing changed | 294 ms | 127 ms | 102 ms |
| Part: `git ls-tree -l` of 10,000 files | 38 ms | 33 ms | 34 ms |
| Part: loading the project's 10,000 whole memories, which sync no longer does | 203 ms | 215 ms | 177 ms |
| Part: listing the project's 10,000 marks, which sync now does | none | 55 ms | 22 ms |

The rest of that run, after B-74 and B-75:

- committing 10,000 shared files took 833 ms;
- a dry-run sync of 10,000 new files took 491 ms;
- the first sync of 10,000 new files took 1,062 ms.

## 6. Decisions

Each entry below gives what was decided, why, the evidence, and the SPEC section it touches.

1. **`forget` and `forget_by_filter` apply the liveness rule under either access.**
   - **What.** After the forget, in the same transaction, each target of a forgotten memory is restored to `active` if all of the following hold:
     - it is local and `superseded`;
     - it is not merged (`extra.merged_into`);
     - it was not forgotten in the same call;
     - no memory of its project that is not retracted still supersedes it, through `supersedes_id` or `extra.unresolved_supersedes`.

     A retracted target is never touched.
   - **Why.** A forget leaves no tombstone that a later pass could repair. Restricting the restore to local targets keeps it off memories whose status comes from a file.
   - **Evidence.** `s4_forget_*` and `s4_only_a_superseder_of_the_same_project_holds_a_target`.
   - **Touches.** SPEC 4, "Forgetting cascades" and status; B-87.
2. **A shared target is left alone.**
   - **What.** Neither forget nor a revision restores a target whose origin is `shared`.
   - **Why.** Its status comes from its shared file, which only `sync` reads, and `sync` settles supersessions itself.
   - **Evidence.** `s4_forget_never_restores_a_retracted_merged_or_shared_target`.
   - **Touches.** SPEC 9, "Supersedes".
3. **The liveness rule in `revise` applies under local access only.**
   - **What.** Under `Access::Local`, three things happen:
     - retracting a superseder restores its unheld targets, as in decision 1;
     - making a retracted superseder live again marks its local `active` targets of the same project `superseded` again;
     - the refusal in decision 4 applies.

     Under `Access::Sync` none of this happens.
   - **Why.** `sync` applies a file's status as written and settles supersessions itself. Second-guessing it in the store would fight the shared folder. Re-superseding on un-retract is the rule read in reverse.
   - **Evidence.** `s4_retracting_a_superseder_locally_restores_its_target`.
   - **Touches.** SPEC 4, SPEC 9 "Supersedes"; B-87.
4. **The refusal to make a held memory `active` is `invalid_input`.**
   - **What.** It applies only when the status changes to `active` from another status.
     - **Message:** "memory X cannot be made active: memory Y supersedes it and is not retracted, and a memory stays superseded while a live memory supersedes it".
     - **Next action:** "Retract or forget memory Y first, or revise memory Y instead."

     A merged memory with no live superseder can still be made `active` by hand.
   - **Why.** The code must come from the frozen set, and the request breaks a rule of the input, so `invalid_input` fits. `confirm_required` would suggest that confirming makes it allowed, which it does not. Merging uses `extra.merged_into`, not `supersedes_id`, so a person can undo a merge.
   - **Evidence.** `s4_revise_refuses_to_make_a_held_memory_active`.
   - **Touches.** SPEC 8, codes; B-87.
5. **The options are new functions, not changed signatures.**
   - **What.** `insert_memories_with(inputs, access, InsertOptions { keep_target_updated_at })` and `revise_memory_with(id, rev, access, &ReviseOptions { sync_mark, keep_updated_at })` sit beside the existing functions, which now call them with the default options.
   - **Why.** Packages H and J build against the existing signatures.
   - **Touches.** CONTRACTS section 1; B-92, B-93.
6. **A sync mark is accepted only under sync access.**
   - **What.** `revise_memory_with` with a sync mark under `Access::Local` returns `invalid_input`, with the next action "Revise the memory without a sync mark, or run sync."
   - **Why.** A mark turns the memory `shared` and names its file. Only `sync` may claim that.
   - **Evidence.** `s9_revise_records_a_sync_mark_in_the_same_transaction`.
   - **Touches.** SPEC 9; B-90.
7. **`MarkRow` carries more than the list in the brief.**
   - **What.** It has `id`, `status`, `origin`, `content_hash`, `supersedes_id`, `unresolved_supersedes`, `merged`, `file_path` and `mark` (the recorded `SyncMark`: blob and text hash).
   - **Why.** `merged` and `content_hash` let sync make every decision without loading the whole memory.
   - **Evidence.** `s9_list_marks_matches_the_whole_memories` checks each field against `get_memory`.
   - **Touches.** SPEC 9, "Change detection before reading"; B-93.
8. **Sync loads whole memories only for changed known files.**
   - **What.** If such a memory is forgotten between the listing and the load, that file becomes a file issue in the reply, and the sync carries on.
   - **Why.** Listing and loading are two reads, and a `forget` may run between them.
   - **Evidence.** `s9_sync_reports_a_known_file_whose_memory_was_forgotten_while_it_ran`.
   - **Touches.** SPEC 9.
9. **Import keeps `updated_at`.**
   - **What.** Import inserts with `keep_target_updated_at` and restores statuses with `keep_updated_at`. An export, import and export again is now byte-identical apart from the header, which removes B-77's one known difference.
   - **Evidence.** `s15_export_import_export_round_trips`.
   - **Touches.** SPEC 10; B-77, B-92.
10. **Share refuses a merged memory as `invalid_input`.**
    - **What.** The refusal names the memory it was merged into, with the next action "Share <that memory> instead."
    - **Why.** It goes into the reply's refusals like the other refusals.
    - **Evidence.** `s9_share_refuses_a_locally_merged_memory`.
    - **Touches.** SPEC 9 "Team sharing rules"; B-89, B-73.
11. **B-74: the cause, and the fix in `sm-store`.**
    - **Cause.** vec1 looks for a block with room before each statement's vectors, with `SELECT id, first, last, val FROM chunk_vectors_idx WHERE bucket = ? AND length(val) < ? LIMIT 1`. It does this at every statement because it flushes at each statement's savepoint. Its only index is on `(bucket, first, last)`, so the search walked every full block before it found the one with room. That made each insert cost more the more vectors were stored, and a bulk load quadratic.
    - **Fix.** The store adds `CREATE INDEX chunk_vectors_idx_open ON chunk_vectors_idx(bucket) WHERE length(val) < 16384`. It holds only the blocks with room, usually one. SQLite re-prepares the statement with the bound `blocksize-min`, which equals the index's literal, and uses the index.
      - `vendor/vec1` is unchanged (B-7).
      - vec1 has no shadow-table guard.
      - vec1's rebuild empties the table with `DELETE`, so the index survives it.
      - The fix sits behind the existing functions and makes every insert path linear, single inserts included.
    - **Evidence.** The bulk insert table in section 5, and both `s4_vec1_*` tests.
    - **Touches.** SPEC 4; B-74, B-7.
12. **B-74: alternatives rejected.**
    - **An expression index.** vec1 writes delete tombstones into a block in place with `sqlite3_blob_write`, and SQLite refuses to open a blob for writing in a table with an expression index. A prototype failed exactly there.
    - **Loading without the vector index, then rebuilding it once.** It needs a separate load path for `import`, `migrate` and `reindex`. It costs a 2.22 s rebuild per 100,000 vectors every time (SPIKE b). It leaves `remember` quadratic. The partial index needs none of this.
    - **A larger `blocksize-min`.** It changes vec1's storage layout. By reasoning, not measurement, it only divides the walk by a constant, so it stays quadratic.
13. **B-74: schema version 1 is edited in place.**
    - **What.** The index is created in `v1()`, and `SCHEMA_VERSION` stays 1.
    - **Why.** B-94 set the precedent, and no 3.0 database has shipped.
    - **Cost.** A database created by an earlier development build lacks the index. It works, as slowly as before.
    - **Touches.** SPEC 4, migrations; B-94.
14. **B-75: reader connections are memory-mapped.**
    - **What.** Readers set `PRAGMA mmap_size` to SQLite's ceiling, 0x7fff0000 bytes (just under 2 GiB), which covers about 250,000 chunks. Pages past it are read as before. The writer is not mapped.
    - **Why.** Project scope reads one 4 KiB page per chunk. Through the page cache, each page costs a read call and a copy. Mapped, each is a pointer into the operating system's file cache, which every reader shares.
    - **Safety.**
      - The store runs no in-place `VACUUM` and no auto-vacuum, only `VACUUM INTO`. A checkpoint therefore never shrinks the file under a mapping.
      - On Windows, SQLite treats a length change refused because the file is mapped (`ERROR_USER_MAPPED_FILE`) as done.
      - AT-5 (60 seconds, two readers and a writer) passes with mapped readers.
    - **Evidence.** Section 5, and `s4_readers_map_the_database_and_the_writer_does_not`.
    - **Touches.** SPEC 4 "Project-scoped search", SPEC 11 `recall` budget; B-75.
15. **B-75: project scope lists the memories that pass once, and ranking selects the top `limit`.**
    - **What.** The query is `chunks WHERE project_id = ? AND memory_id IN (SELECT id FROM memories WHERE project_id = ? AND <filters>)`. Before, each chunk was joined to its memory row. Ranking uses `select_nth_unstable_by` on the total order (score, then id) before sorting, which gives exactly the same first `limit` as a full sort.
    - **Why.** It is 5 ms faster than the join with top-`limit`, in two runs each: 67.9 against 72.9 to 75.3 ms at p50.
    - **Invariant relied on.** A chunk's `project_id` is its memory's. `insert_chunks` copies it from the memory, and no statement in `sm-store` changes either afterwards. It is documented at `search_project`.
    - **Trade-off.** With no map, the probe shows this shape slower than the join (253.5 against 218.8 ms at p50). That applies only past the map's ceiling of about 250,000 chunks, where project scope over a whole store is over budget either way.
    - **Touches.** SPEC 4 "Project-scoped search"; D-2; B-75.
16. **The measurements are examples, and the "before" figures ran on a copy of the base commit.**
    - **What.** The B-74 and sync "before" figures were taken from a `git archive` copy of 93044bb, with the same example or test, in a release build.
    - **Why.** The worktree never switched branch, and the before and after runs used one measuring program.

## 7. Deviations

- Schema version 1 is changed in place, without a migration (decision 13). This follows B-94. A V2 migration that only creates the index is an option for the lead if development databases matter.
- Nothing else differs from the SPEC or CONTRACTS.md. Every existing public `sm-store` and `sm-share` signature is unchanged.

## 8. Contract changes requested

None to `sm-core`. For the lead:

- **DECISIONS B-74.** Record the cause and the partial index (decisions 11 to 13). Strike "the cause … was not located" and the deferral to 3.1. Add a B-n entry for editing schema version 1 in place.
- **DECISIONS B-75.** Record the reader memory map, the `IN` shape and top-`limit` selection, with the figures (decisions 14 and 15). Single-project project scope is now 71.1 ms at p95, from 257.5 ms. The `recall` estimate adds the query embedding (45 to 95 ms, SPEC 11) and comes to about 115 to 165 ms. That is an estimate; `recall` end to end belongs to package H.
- **DECISIONS B-87, B-89, B-90, B-92 and B-93.** Mark them closed, using decisions 1 to 10.
- **CONTRACTS.md.** Optionally list the new `sm-store` functions beside the existing ones: `insert_memories_with`, `revise_memory_with`, `list_marks`, `MarkRow`, `InsertOptions`, `ReviseOptions` and `UNRESOLVED_SUPERSEDES`.

What `sm-daemon` must change: nothing.

- The new refusals arrive as an ordinary `CoreError` with code `invalid_input`, a message and a next action, which the daemon already maps (SPEC 8):
  - making a held memory `active`;
  - a sync mark under local access;
  - a merged memory in `share`.
- The daemon may use `insert_memories_with` or `revise_memory_with` if it ever needs to keep `updated_at`. Nothing it does today needs that.
- With mapped readers, the daemon's file-backed memory (the shared operating-system cache of the database file) grows with the store, up to 2 GiB. Its private memory does not. A memory figure in RESULTS should say which of the two it reports.

## 9. Unfinished or unverified

- **`recall` end to end** at 100,000 chunks, with mapped readers, is not measured here. It belongs to package H and RESULTS.
- **Inserting a superseder that is already retracted** still marks its target `superseded` (`insert_memories`, from stream A). The same is true of a `supersedes_id` pointing into another project. Local tools insert new memories as `active`, and `sync` and import settle statuses themselves afterwards, so no current path leaves a wrong status. The store does not enforce it at insert.
- **Development databases** created before 1008eb5 lack the B-74 index and stay slow until they are recreated (decision 13).
- **The all-changed sync** of 10,000 files still takes 18.5 s. It commits one transaction per file, each with a full sync of the write-ahead log (WAL) to disk (`synchronous=FULL`). Batching revisions into fewer transactions would be an `sm-share` and `sm-store` change for a later package. SPEC 15 sets no budget for it.
- **The Linux figures** come from Docker Desktop, not native Linux.

## 10. Licences

No crate was added to `Cargo.lock`, and no version changed. `Cargo.lock` is unchanged on this branch.
