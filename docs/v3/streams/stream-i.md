# Package I (phase 3): `sm-store` and `sm-share` follow-ups

## Goal

Close the store and sharing items streams A and F left for phase 3: the liveness rule on every local path, the `updated_at` and sync-mark gaps, `share` refusing a merged memory, and the two store measurements, with a fix where one fits inside `sm-store`.

**Done when:** every item under "Build" is fixed and tested, or measured and reported, as it says; both gates pass (RULES.md sections 5 and 6, with the full workspace on Linux too); and `i-report.md` is committed.

## You own

- `crates/sm-store/` and `crates/sm-share/`.
- Package H (`sm-daemon`) and package J (`sm-cli`, the daemon's sharing adapter) run at the same time. **Keep every public `sm-store` and `sm-share` function's signature as it is**; add new functions and options beside them. If a change is unavoidable, list it in the report so the lead can fix the callers.
- You may not edit `sm-core` (frozen), `sm-daemon` or `vendor/vec1`. If `sm-daemon` must change to use something you add (for example, mapping a new refusal to an error code), say exactly what in the report.

## Read

- **SPEC:** sections 4 (the data model, status and `supersedes_id`, "Forgetting cascades"), 6 (recall and the vector search), 9 ("Team sharing", "Shared file format"), 10, 15 (budgets); always 11, 13 and 14.
- **CONTRACTS.md:** all of it.
- **DECISIONS:** B-69 to B-76 (stream A; B-74 the quadratic bulk insert, B-75 project scope at 100,000 chunks), B-87 (the liveness rule: "A memory is superseded while at least one memory of its project that is not retracted supersedes it."), B-88 to B-93 (stream F; B-89 `share`, B-90 sync commits, B-92 the import restore, B-93 the three store requests), B-94.
- **Reports:** `docs/v3/streams/a-report.md` (sections 8 and 9), `f-report.md`; `docs/v3/PROGRESS.md`, "Stream A follow-ups", "Stream F follow-ups" and "Left for the lead in phase 3".
- **Code:** `sm-store/src/memories.rs` (`insert_memories`, `revise_memory`, `record_sync`, `SyncMark`, `forget`, `count_forget`, `forget_by_filter`), `search.rs`, `vec1.rs`, `index.rs`, `conn.rs`; `sm-share/src/share.rs`, `sync.rs`, `export.rs`, `migrate.rs`.

## Build

1. **B-87 on every local path.** When a local superseder is forgotten (`forget`, `forget_by_filter`) or revised to `retracted` outside `sync`, restore, in the same transaction, each target of its project that is left with no live superseder to `active`. Never restore a target that is itself retracted or merged (`extra.merged_into`). Projects that never sync have no other repair.
2. **`revise` refuses to make a memory `active` while a live memory supersedes it** (B-87). Return a `CoreError` with a code from the frozen set, a message and a next action.
3. **Keep `updated_at`** when a superseder is inserted (the target's status changes, its content does not) and at the import restore, so an export and import round trip is byte-identical (B-92, B-93). Add this as an option or a new function; do not change the existing signatures.
4. **`revise_memory` that records the sync mark in the same transaction** (B-90, B-93), and use it in `sync`. Measure the no-change and the all-changed sync before and after on Windows, release build.
5. **A marks-only listing for one project** (`id`, `file_path`, blob, text hash, status, `supersedes_id`, `extra.unresolved_supersedes`, origin) and use it in `sync` (B-93). About 200 ms of the 317 ms no-change sync goes on loading whole memories; measure it after.
6. **`share` refuses a locally merged memory** (`extra.merged_into`), which it would otherwise write as a bare `superseded` (B-89). It goes into the reply's refusals with a reason, like the other refusals.
7. **B-74, bulk inserts into `vec1` are quadratic** (about 0.33 ms more per chunk for every 1,000 already stored). Find where the time goes in `sm-store`'s use of `vec1` (the extension is vendored unchanged and you may not edit it). If a fix fits inside `sm-store`, such as loading `import`, `migrate` and the embeddings after `reindex` with no vector index and building it once, build it behind the existing API and measure 2,000, 10,000 and 50,000 chunks before and after. If not, report the measurements and the options.
8. **B-75, project scope at 100,000 chunks** is 243.7 ms at the 95th percentile for the store alone against `recall`'s 250 ms. Try memory-mapped reads on reader connections (`PRAGMA mmap_size`), and any other change inside `sm-store`. Measure before and after, release build, on Windows; on Linux in Docker if the time allows, and say that Docker's file share is slower.

## Tests

- Names cite their rule (B-40): `s4_` for the store, `s9_` for sharing, `s15_` for budgets, `at8_` or `at9_` where an acceptance test is touched.
- Measurements are tests or examples that print their figures, release build. Never tune a default to make a test pass.

## Report

`docs/v3/streams/i-report.md`, in RULES.md section 9's format. Every figure with the CPU (AMD Ryzen 9 9950X3D, 16 cores, 32 logical processors), the operating system and the build profile.
