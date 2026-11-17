# Stream F report: `sm-share`

Branch `arman-tech/stream-f-share`, from `v3` at `9387a08`. Commits:

- `d85b4c8`: a cherry-pick of the lead's B-86 scanner fix (`89d2a16` on `v3`). It is identical to that commit.
- `b9f90b4`: the crate.
- `7263f00`: the sync mark is stored with the inserted row, and the 10,000-item budget tests are added.
- `27bd52f`: the first version of this report.
- `068e32e`: the fixes for the independent review (task `task_ccc13fda2ff3`) and the second version of this report.

The fixes for the lead's second review (task `task_66be243b1a99`), and this revision of the report, are the commit after them. Section 6, "Review fixes", holds the two lead decisions (suggested B-87 and B-88), the choices made within them, and decision 37 from the second review.

## 1. Done when

Yes. The sharing and migration tests pass on Windows and on Linux, using the real `RegexSecretScanner` from stream G: all 89 tests in `sm-share`, including AT-8 and the `share` side of AT-9.

The second review's four items:

| Item | What | Where |
|---|---|---|
| 1, must fix | The liveness repair did not survive a sync stopped after its writes and before settling. Settling now sweeps every recorded supersession, from what is already in memory, so the next sync puts it right. This also closes decision 14's choice 1 gap, except for a superseder that was forgotten (deleted). | Decision 14 (choices 1, 4 and 5, and choice 6, new) |
| 2, must fix | A memory that consolidation merged into another could be made active again. Neither `sync`'s settling nor `import`'s restore now changes the status of a memory with `extra.merged_into`, and `sync` keeps it when it adopts such a local copy. | Decision 37 (new) |
| 3, should fix | A target the store marks `superseded` when its superseder's file is inserted is now listed in `updated` and not counted in `unchanged`. Done in code. | Decisions 12 and 14 |
| 4, record only | Two gaps in the `key: value` scan. | Decision 32 |

The tests that cover them are listed in section 4, "Second review fixes".

The first review's seven items:

| Item | What | Where |
|---|---|---|
| 1, must fix | Import kept a target's status only when its superseder's line came first. Now it keeps it in either order, and a stopped run is put right by the next one. | Decision 27 (corrected) and decision 33 |
| 2, must fix, DECISION | The liveness rule: a supersession ends when no memory that is not retracted holds it. | Decision 14 (replaced), suggested B-87 |
| 3, must fix | `resolve()` refuses a target stored `retracted` earlier in the same sync. | Decision 34 |
| 4, must fix, DECISION | `migrate` and `import` redact every string they store. | Decision 32, suggested B-88 |
| 5, should fix | A known file whose `supersedes` changed is also listed in `updated` when the rest of it changed its memory. | Decision 15 |
| 6, should fix | The duplicate-text message says when the existing copy is retracted and gives an action. | Decision 35 |
| 7 | Mark spoofing in a crafted import file: accepted by the lead. | Decision 36 |

The tests that cover them are listed in section 4, "Review fixes". Decision 14 has since been changed by the second review's item 1.

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-share/Cargo.toml` | The library crate. Dependencies: `sm-core`, `sm-store`, `serde` and `serde_json`, plus `windows-sys` on Windows only, for `CREATE_NO_WINDOW`. Dev-dependencies: `sm-core` with `test-support`, `sm-capture` (the real scanner, B-8) and `tempfile`. All are workspace pins. `[lints] workspace = true`. |
| `crates/sm-share/src/lib.rs` | Crate documentation and re-exports, with `#![forbid(unsafe_code)]`, plus two crate-private helpers: `is_item_error`, and `is_merged` (decision 37). |
| `crates/sm-share/src/format.rs` | The shared-file format: a strict writer and parser, body normalisation, the text hash, the path and slug, and the `.gitattributes` rule. |
| `crates/sm-share/src/share.rs` | `share`: redacts, rescans and writes files into the working tree. |
| `crates/sm-share/src/git.rs` | The git runner, with no shell. It wraps `rev-parse`, `ls-tree -r -z -l --full-tree` and a single `cat-file --batch` process, plus the error mapping. |
| `crates/sm-share/src/sync.rs` | `sync`: applies the committed shared folder at `HEAD`, with change detection by blob id. |
| `crates/sm-share/src/migrate.rs` | `migrate`: imports a version 1 or 2 JSON (JavaScript Object Notation) export. It streams the file twice and never builds a vector. |
| `crates/sm-share/src/export.rs` | `export` and `import`: the whole store as JSON Lines, using this crate's own line structures. |
| `crates/sm-share/src/redact.rs` | Redaction of the strings `migrate` and `import` store besides content and title (decision 32). Crate-private. |
| `crates/sm-share/src/tests/` | `mod.rs` (temporary stores and isolated git repositories), plus `format.rs`, `share.rs`, `sync.rs`, `migrate.rs`, `export_import.rs` and `budgets.rs`. |
| `Cargo.lock` | Gains `sm-share` only (section 10). |
| `docs/v3/streams/f-report.md` | This report. |

### Public API (application programming interface)

```rust
pub use export::{EXPORT_FORMAT, EXPORT_VERSION, export, import};
pub use migrate::migrate;
pub use share::share;
pub use sync::{MAX_SHARED_FILE_BYTES, sync};
pub mod format;   // SharedDoc, ParseError, SHARED_DIR, SCHEMA, GITATTRIBUTES, SLUG_MAX_CHARS,
                  // text_hash, memory_text_hash, shared_path, slug, normalize_body, clean_title, …

pub fn share(store: &Store, scanner: &dyn SecretScanner, project: &Project,
             repo_root: &Path, params: &ShareParams) -> Result<ShareReply>;
pub fn sync(store: &Store, chunker: &dyn Chunker, chunk_params: &ChunkParams,
            project: &Project, repo_root: &Path, params: &SyncParams) -> Result<SyncReply>;
pub fn migrate(store: &Store, chunker: &dyn Chunker, scanner: &dyn SecretScanner,
               chunk_params: &ChunkParams, params: &MigrateParams,
               cwd_project: Option<&ProjectIdentity>) -> Result<ImportReport>;
pub fn export(store: &Store, params: &ExportParams) -> Result<ExportReport>;
pub fn import(store: &Store, chunker: &dyn Chunker, scanner: &dyn SecretScanner,
              chunk_params: &ChunkParams, params: &ImportParams) -> Result<ImportReport>;
```

The functions are synchronous and keep no process-global state. Every error is an `sm_core::error` code with a message and a next action. No daemon type appears in the API.

## 3. Test commands and exact results

### Windows

Machine: Windows 11 Pro 10.0.26300, AMD Ryzen 9 9950X3D, 16 cores and 32 logical processors, git 2.50.1.windows.1. Cargo ran through PowerShell.

**`cargo fmt --check`:** exit 0.

**`cargo clippy --workspace --all-targets --all-features -- -D warnings`:** exit 0.

These runs are on the final code of this revision, with the second review's fixes.

**`cargo test -p sm-share --all-features`:** exit 0.

```
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**`cargo test --workspace --all-features --no-fail-fast`:** exit 0, 374 tests. The lines are in Cargo's order: `sm_capture`, `sm_core`, `sm_embed`, `real_model`, `sm_share`, `sm_spatial`, `at6_determinism`, `at7_quality`, `s7_pipeline`, `sm_store`, then the doc-tests.

```
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.64s
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.97s
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.00s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.01s
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.63s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (×6 doc-test lines)
```

**`cargo test -p sm-share --all-features --release budget -- --nocapture --test-threads=1`:** exit 0.

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 87 filtered out; finished in 5.36s
```

This run gave a first sync of 10,000 new files in 1,223 ms, a second sync with nothing changed in 393 ms, and `migrate` of 10,000 records in 1,005 ms. The second sync is slower than section 5's 317 ms, so it was compared with the previous commit, `068e32e`, built from `git archive` and run alternately with this code three times each on the same machine:

| Second sync, nothing changed (ms) | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| `068e32e` | 415 | 413 | 325 |
| This revision | 413 | 333 | 328 |

The two are the same within the machine's spread, and most of the spread is in loading the project's memories (207 to 293 ms), which is `sm-store` code this revision does not change. The budget test records no supersession, so the sweep has nothing to settle there. Section 5 is left as measured.

### Linux (Docker)

Machine: image `rust:1.95-trixie` on the WSL2 (Windows Subsystem for Linux 2) kernel 6.6.87.2-microsoft-standard-WSL2. The same CPU (central processing unit), with 32 processors visible. The source was mounted read-only and the target directory was a volume.

The whole workspace was checked, in one container run, in this order.

**`cargo fmt --check`:** exit 0.

**`cargo test --workspace --all-features --locked --no-fail-fast`:** exit 0, 375 tests. Same order as on Windows. `sm_core` has 97 tests here against 96 on Windows, because `s16_spool_dirs_are_user_only` is built on Unix only. `real_model` passes in 0.00 s because the container has no model directory: the real-model test returns at once without loading it, as CLAUDE.md describes.

```
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s
test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.04s
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.66s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.09s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.52s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.03s
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.72s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (×6 doc-test lines)
```

**`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`:** exit 0.

**`cargo test -p sm-share --all-features --locked --release budget -- --nocapture --test-threads=1`:** exit 0. The timings are close to section 5 (first sync 1,625 ms, second sync 139 ms, `migrate` 932 ms; peak 12,800 kB after `migrate`).

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 87 filtered out; finished in 4.73s
```

During this container run one comment in `lib.rs` (the doc of `is_merged`) was corrected, with no code change. Afterwards `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings` (with `sm-share` checked again) and `cargo test -p sm-share --all-features` (89 passed) were run again on Windows on the final text, each with exit 0.

## 4. Acceptance tests

| Test | Test names | Windows | Linux |
|---|---|---|---|
| AT-8, migration | `at8_sample_fixture_migrates`, `at8_full_fixture_migrates`, `at8_full_fixture_record_4_secret_is_redacted`, `at8_full_fixture_report_lists_projects_and_redactions`, `at8_sample_fixture_resumes_after_a_stopped_batch`, `at8_full_fixture_resumes_after_a_stopped_batch` | pass | pass |
| AT-9, redaction (`share` side) | `at9_share_writes_fixture_credentials_redacted_with_kinds`, `at9_share_refuses_credential_in_tag_or_anchor` | pass | pass |

**What AT-8 checks.** Both fixtures are used.

- `read` is 10 for one fixture and 6 for the other, and every record is stored.
- The ids are kept: UUIDs (universally unique identifiers) stay as they are, and `mem-0001` and similar become B-12 ids.
- Timestamps are unchanged to the microsecond.
- Record 5's hash is kept. Record 4's hash is recomputed after redaction, with the version 1 value kept in `imported`.
- Record 2 adds no tag, and record 3 goes to the default project.
- Record 4's key and secret are gone from both the content and keyword search.
- Record 6's Markdown is byte-for-byte unchanged.
- Keyword search finds a migrated memory while its chunks are still pending.
- Every pending chunk is then embedded with `FakeEmbedder` through `pending_chunks` and `set_embeddings`. Every memory ends with embedded chunks and nothing is left pending.
- A second run inserts nothing and counts every record in `skipped_existing`.
- A run stopped after its first batch resumes.

**What the `share` side of AT-9 checks.**

- Each fake credential from `redaction_cases.json`, stored directly in the memory, is written as `[REDACTED:<kind>]` with its kinds listed.
- A credential in a tag or an anchor is refused with `secret_detected`, and no file is written.
- The `remember` side of AT-9 belongs to the daemon.

**The other 81 tests.** The `s9_`, `s10_` and `s15_` tests cover everything else in the brief's test list:

- the format, with every malformed case refused at its line;
- `share`'s path and slug, the `.gitattributes` file, replacing a file after a type change, the refusals, and `not_git_repository`;
- every `sync` case in the brief, including the full share, commit and sync loop between two stores;
- `migrate`'s B-35 and B-79 cases;
- `export` and `import`: the round trip, refusals, project selection, format and version checks, resuming, links with a missing end, and shared memories with their supersessions.

**Review fixes.** These 14 tests are new; the round trip test gained cases. Each one fails when its fix is reverted (the evidence is under each decision).

| Item | Tests |
|---|---|
| 1 | `s15_import_restores_a_target_whose_line_comes_first`, `s15_an_import_stopped_before_restoring_statuses_resumes`, `s15_a_repeated_import_keeps_a_supersession_made_since`. `s15_export_import_export_round_trips` now has two targets whose lines come first: `retired`, superseded and then retracted, and `revived`, whose superseder was retracted. They come back `retracted` and `active`. |
| 2 | `s9_sync_restores_a_target_when_its_superseding_file_is_deleted` (the revert), `s9_sync_restores_a_target_when_its_superseding_file_says_retracted` (S's file changed to `status: retracted`), `s9_sync_keeps_a_target_superseded_while_one_superseder_is_live` (two superseders, one retracted), `s9_sync_makes_a_local_target_active_when_its_superseder_is_retracted`, `s9_sync_leaves_a_target_active_when_its_new_superseder_is_retracted` (decision 14, choice 3) |
| 3 | `s9_sync_does_not_resolve_a_target_added_retracted_in_the_same_sync` |
| 4 | `s10_migrate_redacts_every_string_it_stores`, `s15_import_redacts_every_string_it_stores` |
| 5 | `s9_sync_reports_a_changed_supersedes_and_lists_the_rest_as_updated` |
| 6 | `s9_sync_names_a_retracted_local_copy_of_a_new_files_text`, `s9_sync_names_a_shared_copy_this_sync_retracts` |

Each item 2 test checks the reply's lists, the statuses stored and `blobs_read`. The shared-target tests go on to further syncs and check each one: a sync with nothing changed, bringing the superseder back, or retracting the last live superseder. The revert test and the choice 3 test also check that the dry run writes nothing and gives the reply the real run gives.

**Second review fixes.** These 9 tests are new, and one earlier test gained assertions. Each new test fails when its fix is reverted; the evidence is under decisions 12, 14 and 37. The choice 5 test also pins a cost: how many blobs a stuck shared target costs on each sync.

| Item | Tests |
|---|---|
| 1 | `s9_sync_restores_a_shared_target_after_a_sync_stopped_before_settling`, `s9_sync_restores_a_local_target_after_a_sync_stopped_before_settling`, `s9_sync_restores_a_target_after_a_sync_stopped_once_its_retracted_superseder_was_added` (decision 14, choice 3, interrupted), `s9_sync_supersedes_a_target_again_after_a_sync_stopped_once_its_superseder_came_back` (the other direction, choice 6). `s9_sync_reads_again_on_every_sync_a_target_whose_file_says_superseded` (choice 5: one blob per sync while the target stays so) |
| 2 | `s9_sync_keeps_a_merged_target_superseded_when_its_superseder_is_retracted` (settling), `s15_a_repeated_import_keeps_a_merge_made_since` (the import restore), `s9_sync_adopts_a_merged_local_copy_and_keeps_it_superseded` (a known file, an extension of the item) |
| 3 | `s9_sync_lists_a_target_the_store_marks_as_updated_not_unchanged`. `s9_sync_makes_a_local_target_active_when_its_superseder_is_retracted` now also checks its first sync: `updated` is the target and `unchanged` is 0. |

How the item 1 tests stop a sync: a crate-private test hook, `SyncOptions::stop_before_settle`, ends the sync with an `internal` error after every file is applied and the missing ones are retracted, before settling. The hook is not public API, and `sync` always runs with it off, as with import's `ImportOptions::stop_before_restore`.

- The stopped sync leaves the store as a crash at that point would. Each test checks this: the superseder's retraction (or its revision back to `active`) has committed, and the target has not been settled.
- The next sync, with nothing else changed, settles the target and lists it in `updated`. Each test checks `unchanged` and `blobs_read` too:
  - the shared target restored: 1 blob, its own file read again because its mark skipped it;
  - the local target restored: 0 blobs;
  - the target of a superseder whose file says `retracted`: 2 blobs, that file and the target's own;
  - the target superseded again: 0 blobs.
- The sync after that changes nothing. It reads 0 blobs, or 1 in the test whose superseder's file says `retracted`, because a retracted memory's file is never skipped by a mark (decision 14, choice 3).

## 5. Measured budgets

SPEC 11 sets no target for `sync` or `migrate`. These figures show they are not the bottleneck: SPEC 10 says embedding is, at about 67 minutes for 10,000 chunks.

**Setup.**

- The tests are `s9_budget_sync_reads_only_changed_files_among_10000` and `s10_budget_migrate_streams_a_10000_record_export_with_vectors`.
- Profile: release.
- Run with `--test-threads=1`.
- The sync test commits 10,000 shared files with `git fast-import`.
- The migrate test writes a version 1 export of 10,000 records with 384-float vectors: 81.0 MB, 80,956,527 bytes.

**Machines.**

- Windows: Windows 11 Pro 10.0.26300, AMD Ryzen 9 9950X3D, 16 cores and 32 logical processors.
- Linux: Docker `rust:1.95-trixie` on the same machine, with 32 processors visible.

**Timings, in milliseconds.** The figures come from the final run on `7263f00`, before the review fixes. Section 3 gives runs with both reviews' fixes. On Windows it compares the second sync with the previous commit's on the same machine at the same time: they are the same within the machine's spread. Three earlier Windows runs of the same code gave 1,127 to 1,192 ms for the first sync, 306 to 324 ms for the second and 922 ms for `migrate`.

| Operation | Windows | Linux |
|---|---:|---:|
| Dry run, 10,000 new files (reads, parses and hashes every blob, writes nothing) | 526 | 837 |
| First sync, 10,000 new files (inserted with pending chunks) | 1,329 | 1,583 |
| Second sync, nothing changed (0 blobs read) | 317 | 137 |
| — of which `git ls-tree -l` lists 10,000 files | 52 | 14 |
| — of which the project's 10,000 memories are loaded | 204 | 97 |
| Third sync, one file changed (1 blob read) | 365 | 142 |
| `migrate`, 10,000 records with vectors | 946 | 828 |
| `migrate` again (all 10,000 counted in `skipped_existing`) | 278 | 226 |

**Peak memory.**

- **Linux:** `VmHWM` from `/proc/self/status`, for the whole test process.
  - Before migrate: 6,656 kB.
  - After migrate: 12,288 kB, for an 81 MB file. This shows the file is streamed and vectors are never built.
  - After the second migrate: 12,544 kB.
  - After the syncs: 87,848 kB.
- **Windows:** `PeakWorkingSet64` of the test process, polled every 20 ms from PowerShell, with each test in its own process.
  - The migrate test process: 14.8 MB.
  - The sync test process: 62.6 MB.

**For comparison.** SPEC 9 says: "Listing 10,000 files took 9 ms; reading, parsing and hashing all 10,000 took 115 ms."

- Listing is 14 ms on Linux and 52 ms on Windows.
- When nothing has changed, most of the time goes on loading the project's memories to find their sync marks. Section 9 has a possible follow-up.

## 6. Decisions

Each item gives what was chosen, why, and the SPEC section it touches. The evidence for each is the named test, or the module documentation in the file named.

### Format (SPEC 9, "Shared file format"; `format.rs`)

1. **Absent values.** An absent optional value, and an empty list, are written as the key and a colon with nothing after it (`supersedes:`). This is the form SPEC 9's example uses. Lists are written `[a, b]` on one line. Test: `s9_render_writes_the_spec_example_layout`.
2. **A strict parser that accepts a few hand-written forms.** The parser takes the nine keys and `schema: 1` only, with literal values and no quoting, escapes or comments. For people editing files by hand, it also accepts:
   - keys in another order;
   - an omitted optional key;
   - `[]` for an empty list;
   - a byte-order mark;
   - CRLF (carriage return and line feed) or lone CR line endings;
   - blank lines before the title.

   Rendering such a file gives the canonical bytes. Tests: `s9_omitted_optional_keys_and_empty_lists_read_as_empty`, `s9_crlf_input_reads_as_lf_and_renders_canonically` and the six `s9_malformed_*` tests.
3. **Body normalisation.**
   - CRLF and lone CR become LF.
   - Blank lines before the first text are dropped.
   - Trailing whitespace at the end of the body is removed.
   - Inner whitespace is kept.
   - A title is one line: CR and LF in it become spaces, and it is trimmed. A memory with no title uses `derive_title` (B-18).

   Tests: `s9_round_trip_body_with_headings_and_fenced_code`, `s9_crlf_input_reads_as_lf_and_renders_canonically`.
4. **Text hash.** The hash is the SHA-256 (Secure Hash Algorithm, 256-bit) of `# <title>\n\n<body>\n`, exactly as `render` writes it. One function serves both `share` and `sync`. Test: `s9_text_hash_is_of_the_markdown_part`.
5. **Slug.**
   - The title is lower-cased to ASCII.
   - Each run of characters other than letters and digits becomes one `-`, and leading and trailing `-` are removed.
   - It is cut to 40 characters (`SLUG_MAX_CHARS`), and any `-` the cut leaves at the end is removed.
   - An empty result becomes `memory`.

   With the type folder (`convention` is the longest) and the `--<8>.md` suffix, the path under the repository is at most 87 characters, well inside Windows path limits. Test: `s9_path_and_slug`.
6. **`.gitattributes`.** `.spatial-memory/shared/.gitattributes` holds `* text eol=lf`. It is written when missing and left alone when present. Test: `s9_share_writes_gitattributes_once_and_keeps_an_existing_one`.

### `share` (SPEC 9, rules 1 and 2, and "Security"; `share.rs`)

7. **Redact, then rescan.** The title and body are redacted, and then the whole rendered file is scanned again. A match in the front matter (a tag or an anchor) refuses the memory with `secret_detected`. Redacting it would change what the memory is attached to. Test: `at9_share_refuses_credential_in_tag_or_anchor`.
8. **Finding the earlier file.** It is found by its `--<last 8>.md` suffix and confirmed by the id inside it. A file with that suffix that holds another id, or does not parse, is left alone.
9. **Writing safely.** A file at the target path that holds another id is refused with `invalid_input`. Writes go to a temporary file in the same folder and are then renamed. Symbolic links in the shared folder are never followed, and a shared folder that is itself a link fails the call. Tests: `s9_share_replaces_the_file_after_a_type_change`, `s9_share_refuses_a_path_held_by_another_id`.

### `sync` (SPEC 9, rules 3 to 5, and "Change detection before reading"; B-78; `sync.rs`, `git.rs`)

10. **Git failures.** Git missing from `PATH`, and git failing, both use `not_git_repository`, carrying git's message. There is no closer code in `sm_core::error` (see section 8). The next action is specific:
    - "dubious ownership" names `safe.directory`;
    - "not a git repository" says to run sync from inside one;
    - anything else names the git command to run by hand;
    - git missing from `PATH` says to install git.
11. **File size limit.** One file may be at most 1 MiB (`MAX_SHARED_FILE_BYTES`). A larger file is reported in `errors` and not read. This bounds what one sync holds in memory.
12. **`.gitattributes` is not reported.** The folder's own `.gitattributes` is written by `share` and is not a memory, so sync skips it without a report. Every other entry lands in exactly one list of the reply, with two exceptions:
    - a known file whose `supersedes` changed is in `errors`, and also in `updated` when the rest of the file changed its memory (decision 15);
    - a memory whose status sync changed for a supersession is in `updated`, and its file is then not counted in `unchanged` (decision 14). That is a memory that settling changed, and also a stored memory that the store marked `superseded` when a new file that supersedes it was inserted (second review, item 3; the first version listed only the settled ones). A local memory with no file can be listed this way. A memory added in the same sync is listed in `added` only.

    Test: `s9_sync_lists_a_target_the_store_marks_as_updated_not_unchanged`, which also checks that the dry run gives the same reply. Mutation evidence: without the line that lists such a target, 2 tests fail ("87 passed; 2 failed"). They are this test and `s9_sync_makes_a_local_target_active_when_its_superseder_is_retracted`, whose first sync now checks `updated`.
13. **Missing `supersedes` targets.** A file whose `supersedes` target is in neither the store nor the files is still added. Its `supersedes_id` is left empty and the target id is kept in provenance `extra.unresolved_supersedes`. When the target arrives in a later sync, it is stored as superseded. Test: `s9_sync_keeps_an_unresolved_supersedes_until_the_target_arrives`.
14. **Supersession liveness.** Lead DECISION (review item 2), suggested B-87. It replaces this stream's first rule, that a supersession is never undone. The rule: "a memory is superseded while at least one memory of its project that is not retracted supersedes it." In `sync`:
    - **Before any write**, the targets held live are worked out from the statuses this sync will leave. A known file gives its memory's status, and a memory whose file is gone counts as retracted. A memory that is already retracted, or that this sync retracts, holds nothing.
    - **A target's own file does not undo a live supersession.** A file that says `active` is stored as `superseded` while a live superseder holds it.
    - **After every file is applied, every recorded supersession of the project is settled** (second review, item 1). The first version settled only the targets this sync touched, so a sync that stopped after its writes and before settling left its targets wrong for good. Now every target that a memory of the project records, in `supersedes_id` or in `extra.unresolved_supersedes`, is checked against the statuses this sync has left, using what sync has already loaded:
      - A target that is `active` and has a live superseder becomes `superseded`.
      - A target that is `superseded` and has none is restored. A shared target takes the status its file gives; when its mark had skipped the file, the blob is read again and counted in `blobs_read`. A local target becomes `active`.
      - A retracted target is never changed, and neither is a memory that consolidation merged into another (decision 37).
    - **A changed target is listed in `updated`** (decision 12).
    - **The cost.** Only targets whose status breaks the rule are acted on. Normally there are none, so settling reads and writes nothing, and a sync with nothing changed still reads 0 blobs (the 10,000-file budget test). A local target that needs settling costs no I/O. A shared target that needs it costs one blob read when its mark had skipped its file, and then holds.
    - **A stopped sync.** The writes before settling each commit on their own: a batch of new files, a revision, a retraction. A crash or a store error after them leaves the targets unsettled. The next sync settles them, even when nothing else changed.

    Tests: the five first-review item 2 tests and the four second-review item 1 tests (section 4).

    Choices made within the decision:
    1. **Every recorded supersession is settled, from what sync has already loaded** (replaces "only touched targets are settled"). This also settles a retraction made outside sync, such as a local superseder retracted with `revise_memory`. The next sync finds the same state a stopped sync leaves, which `s9_sync_restores_a_local_target_after_a_sync_stopped_before_settling` covers.
       - Remaining gap: a superseder that was forgotten. `forget` deletes the superseder's row (`DELETE FROM memories`), so nothing records that it superseded the target, and the target stays `superseded`. Only sync can retract a shared memory (`sm-store` refuses local writes to shared memories), so the gap concerns local superseders only. The place to apply the rule there is the daemon's `forget`, which is the lead's.
    2. **A local target is made `active` with `Access::Sync`**, which `sm-store` allows. It is listed in `updated` although it has no file.
    3. **A new file that is already `retracted` still stores its `supersedes_id`**, the record of what it superseded. The store marks the target `superseded` when the file is inserted, and settling restores it from its file. So the target is written twice in that sync. It is listed in `updated`, although its status ends where it began, because it was written, and its file is not counted in `unchanged`. Test: `s9_sync_leaves_a_target_active_when_its_new_superseder_is_retracted`. A sync that stops between the insert and settling leaves the target `superseded`, and the next sync restores it. Test: `s9_sync_restores_a_target_after_a_sync_stopped_once_its_retracted_superseder_was_added`.
    4. **Some shared targets keep their status.** These are a target whose file was reported in `errors`, one whose id is in two files, and one whose file could not be read again. The file's error is reported and nothing is changed. Settling now checks every supersession, so a file whose mark skipped it but that cannot be read again is retried and reported on every sync while that lasts. Before, that happened only when something touched its target. A file in `errors` costs nothing more: it has no mark, so it is read on every sync anyway.
    5. **A target whose own file says `superseded` keeps that status** when no live superseder is left. The rule takes the file's status, and sync never rewrites a file.
       - The cost: settling finds such a target on every sync, and its file, skipped by its mark, is read again. That is one blob read per sync while it stays so, which the lead accepted for a stuck shared target.
       - It arises when a target is shared while `superseded`, since `share` writes the stored status into the file, and all its superseders are then retracted.
       - Test: `s9_sync_reads_again_on_every_sync_a_target_whose_file_says_superseded`.
    6. **Both directions are settled, not only the restore the lead named.** A target left `active` while a live memory supersedes it is made `superseded` too. That is the state a sync leaves when it stops after bringing a retracted superseder back. It costs no read.
       - A consequence: a local target that someone made `active` by hand, while a live memory still supersedes it, is made `superseded` again by the next sync, as the rule says.
       - Test: `s9_sync_supersedes_a_target_again_after_a_sync_stopped_once_its_superseder_came_back`.

    Mutation evidence:
    - First review, before the sweep: with settling turned off, five tests failed, the four item 2 tests and the choice 3 test.
    - Second review: with the old scope put back (settle only the targets this sync touched, the first version's rule), 5 tests fail ("84 passed; 5 failed"):
      - the four item 1 tests, each at the sync after the stopped one, which leaves the target unsettled;
      - the choice 5 test, `s9_sync_reads_again_on_every_sync_a_target_whose_file_says_superseded`, at "read again on every sync". Nothing touches the target after the first sync, so the old scope never reaches it and reads 0 blobs.
15. **A changed `supersedes` on a known memory.** The change is reported, the rest of the file is applied, and no mark is recorded. The report therefore repeats until the file is fixed, and the file is never silently treated as unchanged (B-73). The memory's id is also listed in `updated` when the rest of the file changed something (review item 5). An unchanged rest is reported in `errors` only. Test: `s9_sync_reports_a_changed_supersedes_and_lists_the_rest_as_updated`. Removing the line that lists it makes the test fail.
16. **Adoption with a mismatch.** Adopting the author's local copy is refused when the file's `supersedes` differs from the copy's.
17. **Unreadable files.** A file that is still listed but cannot be read protects its memory from retraction. The memory's id is unknown, and the reported error is what needs fixing.
18. **Batches and commit format.** New files are inserted in batches of 128, one transaction each, with targets first. The commit is 4 to 64 hexadecimal digits.
19. **The mark is stored with the inserted row.** `extra.sync = {blob, text_hash}` is the layout `SyncMark::of` reads, written inside the batch's own transaction rather than by a later `record_sync` per file. The store commits with `synchronous=FULL`, so the per-file call cost one extra fsync per file.

    The evidence is the first sync of 10,000 files on Windows, in release: 16,118 ms before the change and 1,127 to 1,329 ms after it. Revisions and adoptions still use `record_sync`. Tests: `s9_sync_adds_a_new_file_as_shared_pending_and_marked`, `s9_budget_sync_reads_only_changed_files_among_10000`. See also section 7.

### `migrate` (SPEC 10; B-12, B-14, B-18, B-35, B-79; `migrate.rs`)

20. **The version 1 and 2 exporters write the same format.** `_export_json` is the same on `master` and on `origin/feature/v2-implementations`: the two branches differ only in the embedding role and type-checker comments. Both write one JSON array of records.

    `_prepare_record_for_export` decodes a string `metadata` into an object, or `{}` when it cannot. So a JSON export carries `metadata` as an object, and only CSV (comma-separated values) and Parquet exports carry it as a string. `migrate` keeps `metadata` exactly as given, object or string, in `provenance.extra.metadata`. It also accepts a `tags` value that is a JSON-encoded string list.
21. **Project matching gets a third step.** Between B-79's name match and creating a project, a value equal to the working directory's project name uses that project, created by its own identity. This is what B-79's gap note expects. A value that looks like a remote (`host/path`) creates a `git:` project named after its last path segment; any other value creates a `name:` project. Tests: `s10_project_values_match_existing_projects_by_remote_then_name`, `s10_a_remote_value_creates_a_git_project_and_other_values_a_name_project`.
22. **Fields kept in `imported`.** Unknown fields go to `imported` under their own names. So does a timestamp that does not parse, which is not stored as a time. A missing timestamp is left to the store.
23. **Records that cannot be stored are issues.** Each of these is an `ImportIssue`, so `read = inserted + skipped_existing + issues.len()` always holds:
    - an element that is not an object;
    - a missing id;
    - an id that appears earlier in the file;
    - empty content;
    - content whose hash another memory of the project holds (the issue names that memory).

    A numeric id is accepted. Issues are sorted by position in the file, with the id as the file writes it. Test: `s10_unstorable_records_are_issues_and_the_count_balances`.
24. **Counting and batches.** `redacted` counts inserted records only. Batches are 256 records per transaction.

### `export` and `import` (B-77; `export.rs`)

25. **The file.** The format is named `spatial-memory-export`, version 1, and uses this crate's own line structures. Export batches are 256.
26. **Import access and redaction.** Import uses `Access::Sync`, because it restores shared memories and supersessions that local access may not write. It redacts every string it stores, not only the title and content (decision 32).
27. **Supersessions on import** (corrected for review item 1). A memory waits until its `supersedes_id` target has been inserted. The first version of this decision said the target's status from the file is then restored. That was true only when the superseder's line came first; in the normal order, target first, the target was left `superseded`. Now the first pass collects every `supersedes_id` in the file, and the status each target's line gives is kept whichever line comes first. Decision 33 has the details.
28. **Counting.** Valid project lines are not records: they count only in `projects_created`. Memory lines, link lines and unusable lines each land in exactly one of `inserted`, `skipped_existing` and `issues`. An issue's `index` is the 0-based line number, with the header at line 0. Tests: `s15_import_reports_unusable_lines_and_redacts_secrets`, `s15_an_interrupted_import_resumes`.

### Tests

29. **AT-9 uses a hand-written mapper.** `redaction_cases.json` gives the expected text with version 1's `[REDACTED_<KIND>]` placeholders. The test rewrites them as version 3's `[REDACTED:<kind>]` with a small hand-written scanner, because `regex` is not a dev-dependency of this crate. The kinds are compared as the fixture lists them.
30. **The budget tests build their commit with `git fast-import`.** `git add` of 10,000 files took 39.8 s on Windows; `fast-import` takes 0.9 s. `sync` reads only the committed tree, so the work tree is not needed.
31. **Every test git process is isolated.** It gets `GIT_CONFIG_GLOBAL` pointing at an empty file, `GIT_CONFIG_NOSYSTEM=1`, and a fixed author and committer, so the user's own git settings cannot change a result.

### Review fixes (the independent review, items 1 to 7)

The liveness rule, lead DECISION and suggested B-87, is decision 14 above.

32. **Redaction of every stored string.** Lead DECISION (review item 4), suggested B-88. `migrate` and `import` redact every string they store. Values are redacted; keys are not. The code is in `redact.rs`.
    - **What `import` scans:**
      - content and title;
      - the provenance author, session, commit, file path and tool;
      - every string value in `provenance.extra` and in `imported`, at any depth;
      - tags and anchors;
      - a link's extractor and verified commit.
    - **What `migrate` scans:**
      - content;
      - the namespace and the tags;
      - every string value in `provenance.extra` (`source`, `metadata`) and in `imported`, at any depth, unknown fields included.
    - **How.**
      - Every string value of a JSON tree is scanned.
      - A value under an object key is scanned again as `key: value`. The scanner's keyword tier needs `keyword` followed by `=` or `:`, so this finds `{"password": "hunter2"}` the way it finds `password: hunter2` in text.
      - A string value that holds a JSON object or array, such as version 1's string `metadata`, is parsed and scanned the same way. It is written back, compactly, only when a value inside it changed.
      - A value that already holds a `[REDACTED:<kind>]` marker is not changed again (B-64).
      - The content hash is recomputed from the redacted content.
    - **A tag, namespace or anchor that holds a secret is dropped**, because it cannot hold a marker. The record counts once in `redacted`.
      - No `ImportIssue` names the dropped kinds. The lead allowed one only if it could not read as "not stored". An issue here means a record that was not stored, and adding one for a stored record would break AT-8's count, `read = inserted + skipped_existing + issues.len()`.
      - `migrate` keeps the dropped value, redacted, in `imported.tags` or `imported.namespace`, where the original list already goes when a tag is cleaned. `import` drops it with no copy.
    - **A link with an anchor end that holds a secret is an issue and is not stored.** Dropping the end would change what the link joins. The id the issue shows is redacted. A redacted extractor or verified commit counts the link once in `redacted`.
    - Gaps:
      - project values (`migrate`) and project lines (`import`) are not scanned, because they are matched and created as identities;
      - a record's raw id is not scanned;
      - the `key: value` scan has no left word boundary, because the scanner's keyword patterns have none. So a string under a key that ends in `pwd`, `password`, `secret` or `token` is redacted: `{"pwd": "/home/u/proj"}` is stored as `{"pwd": "[REDACTED:password]"}`, because `pwd: /home/u/proj` matches the `password` pattern and the path is not a placeholder that the false positive filter drops. Accepted by the lead: version 1 uses `cwd`, and redacting too much is the conservative error (second review, item 4);
      - only strings are scanned. So `{"password": 12345678}` is stored unchanged, although the same text in content would be redacted (second review, item 4).

    Tests:
    - `s10_migrate_redacts_every_string_it_stores` covers a secret in `metadata` in object form, in string form holding JSON and in plain string form, in `source`, in unknown fields at depth, in a tag (both in a list and in a JSON-encoded string list), and in the namespace.
    - `s15_import_redacts_every_string_it_stores` covers a secret in the provenance author, `provenance.extra`, `imported`, a tag, an anchor, a link's extractor, and a link's anchor end. For the anchor end, the issue must not show the secret.
    - A second import of the same file changes nothing (`redacted` 0).
    - AT-8's full-fixture tests still pass with their counts.
33. **The import status restore** (review item 1; decision 27 corrected).
    - **What changed.** The first pass collects every `supersedes_id` in the file. The status each target's line gives is recorded whichever line comes first. After the superseder's batch is stored, a target the store marked `superseded` gets its line's status back.
    - **Resuming.** The restore is its own write after the batch, so a run can stop between the two. The next run over the same file finds the target already stored (`skipped_existing`) and restores it the same way, so the restore can safely run twice.
      - Test: `s15_an_import_stopped_before_restoring_statuses_resumes`. It stops the run with a crate-private test hook, `ImportOptions::stop_before_restore`. The hook is not public API, and `import` always runs with it off.
    - **Narrowed from the lead's wording.** The lead said to restore when the stored status differs from the exported one. The restore acts only when the stored status is `superseded` and the line's status is not.
      - An interrupted run can only leave a target wrongly `superseded`. Any other difference is a change made after the import, and a repeated import must not undo it.
      - A target stays `superseded` when a stored memory outside the file that is not retracted supersedes it (the liveness rule). The exception is a target this run inserted, whose line then rules.
      - Test: `s15_a_repeated_import_keeps_a_supersession_made_since`.
      - A target that consolidation merged into another always stays `superseded` (decision 37). Test: `s15_a_repeated_import_keeps_a_merge_made_since`.
    - **Gap.** Take a target that is already stored and was changed locally since, for example retracted. It gets its line's status back when a superseder in the file is newly inserted in this run. The store's mark overwrote the local status before the restore could see it, and `sm-store` gives no way to tell the two apart.
    - `updated_at` is stamped again by the restore (deviation 2).
    - Tests: `s15_import_restores_a_target_whose_line_comes_first` and the round trip's new target-first cases. Mutation evidence: before the fix, only targets whose superseder's line was already waiting were recorded. Putting that check back makes 6 tests fail.
34. **`resolve()` refuses a target retracted in the same sync** (review item 3). It looks up the status that a target added earlier in the same batch is being stored with. For any other target it uses the status now. A `retracted` target is not stored as `supersedes_id`, because the store would mark it `superseded` and so undo its retraction.
    - The target's id is kept in `extra.unresolved_supersedes`, as for a missing target (decision 13). Sync's view of who supersedes whom (`recorded_supersedes`) reads that field as well as `supersedes_id`, so the liveness rule still sees the supersession.
    - Test: `s9_sync_does_not_resolve_a_target_added_retracted_in_the_same_sync`. Mutation evidence: with `resolve` ignoring the status, it fails.
35. **The duplicate-text message names a retracted copy** (review item 6, message only). The store holds one copy of a text per project. So a new file whose text is a retracted memory's is still skipped, and the message now says why.
    - The message: "its text is already stored as memory `<id>`, which is retracted (or: which this sync retracts); the file was skipped and the retracted copy is kept."
    - For a local copy it adds: "Forget memory `<id>` and run sync again to add this file, or change this file's text."
    - Otherwise it adds: "To bring the text back, restore memory `<id>`'s own file, or change this file's text."
    - A live copy keeps the old message, "Remove one of the two copies."
    - Known gap, left as the lead asked: the store is not changed, so a retracted copy blocks the same text under a new id until it is forgotten or its file restored.
    - Tests: `s9_sync_names_a_retracted_local_copy_of_a_new_files_text`, `s9_sync_names_a_shared_copy_this_sync_retracts`.
36. **Mark spoofing in a crafted import file: accepted by the lead and left as is** (review item 7). `import` keeps `provenance.extra`, including a `sync` mark, as an export wrote it. A crafted file can carry a mark naming the path and blob of a committed shared file. A later `sync` then counts that file as unchanged and keeps the imported text instead of the file's. Nothing was changed.

### Second review fixes (the lead's second review, items 1 to 4)

Item 1 is decision 14 (choices 1 and 3 to 6), item 3 is decision 12, and item 4 is two gaps added to decision 32.

37. **A merged memory is never made active again** (second review, item 2). `sm-store`'s `merge_memories` (B-73) sets the merged memory `superseded` and records provenance `extra.merged_into`. No memory's `supersedes_id` points at it. So no superseder holds it, and the liveness rule alone would make it active. The check is a crate-private `is_merged` in `lib.rs`: `extra.merged_into` is set and not null.
    - **`sync`'s settling** leaves a merged target out, so it never changes the target's status.
      - Test: `s9_sync_keeps_a_merged_target_superseded_when_its_superseder_is_retracted`. A local memory that a shared file supersedes is merged locally, and the shared file is then deleted.
      - The sync retracts the shared memory. The merged memory stays `superseded`, with its `merged_into`. Nothing is in `updated`, and 0 blobs are read.
    - **`import`'s status restore** never changes a merged target.
      - Test: `s15_a_repeated_import_keeps_a_merge_made_since`. A target exported `active` (its superseder was retracted) is imported, merged locally, and the same file is imported again.
      - The second import has no issues and the counts balance. The memory stays `superseded`, and every stored memory is as it was before the second import.
    - **Extension: a known file in `sync`.** The lead named settling and the import restore. A third path could also make a merged memory active: the author's local copy, shared and then merged, is adopted from its file, which says `active`.
      - Here the merge counts as a live superseder: the file's `active` is taken as `superseded`.
      - A file that says `retracted` still retracts the memory. This matches a deleted file: sync retracts a memory whose file is gone whether or not it is merged. A file brought back to `active` returns the memory to `superseded`, not `active`.
      - The first draft of this extension kept every status of a merged memory. That would have ignored a retraction by file while obeying one by deletion, so it was narrowed.
      - Test: `s9_sync_adopts_a_merged_local_copy_and_keeps_it_superseded`. Adoption keeps the memory `superseded`, and a sync with nothing changed then reads 0 blobs. The file changed to `retracted` retracts it, and the file brought back gives `superseded`.
    - **Gap.** `import` never changes a merged target's status, even when the target's line says `retracted`. This follows the lead's wording ("never change"). A repeated import must not undo a local change, and a merge is a local change (decision 33).
    - Mutation evidence. Each change below was made alone and the 89 tests run; each time "88 passed; 1 failed", and the failing test was the one named:
      - settling without the merged filter: `s9_sync_keeps_a_merged_target_superseded_when_its_superseder_is_retracted`;
      - the import restore without `is_merged`: `s15_a_repeated_import_keeps_a_merge_made_since`, whose merged memory comes back `active`;
      - the known-file mapping turned off: `s9_sync_adopts_a_merged_local_copy_and_keeps_it_superseded`, whose adopted copy is left `active`;
      - the first draft's guard put back (every status of a merged memory kept): the same test, at the step where the file says `retracted`, because nothing is updated.

## 7. Deviations

1. **New files are marked when they are inserted, not "then marked with `record_sync`".** That quote is from the brief, step 3. The mark has the same layout (`SyncMark::of` reads it), and it is written in the same transaction as the insert. The reason and the evidence are in decision 19: 16.1 s down to 1.3 s for 10,000 files. `s9_sync_adds_a_new_file_as_shared_pending_and_marked` checks the mark.
2. **The export and import round trip differs in one field** (corrected for review item 1). It is identical apart from the header, except for the `updated_at` of a memory that another memory supersedes. That field becomes the time of the import. `sm-store` stamps it when the superseder is inserted, and stamps it again when the import gives the target its own status back (decision 33). The store offers no way to set it back.
   - The first version implied that the round trip kept every target's status. That was true only when the superseder's line came first, which the test data then covered. In the normal order the target's line comes first, and such a target came back `superseded`.
   - `s15_export_import_export_round_trips` now has two targets whose lines come first, one retracted and one active, and compares every other byte.
   - The brief allows the difference when it is stated. Keeping `updated_at` is a phase 3 lead item (section 8).

## 8. Contract changes requested, with the lead's answers

1. **A `git_failed` error code**, or an agreed reuse. Today git missing from `PATH`, or git refusing a repository, is reported as `not_git_repository`, which is misleading for "dubious ownership". The message and next action carry the real cause (decision 10).
   **Lead's answer: declined for 3.0, because the code set is frozen.** Decision 10 stands as it is.
2. **An `sm-store` option to keep `updated_at` when a superseder is inserted.** With it, the export and import round trip would be byte-identical (deviation 2). It would also need to cover the status restore (decision 33).
   **Lead's answer: a phase 3 lead item.**
3. **An `sm-store` `revise_memory` that records the sync mark in the same transaction.** A changed file now costs two commits: `revise_memory`, then `record_sync`. Each commit is about 1.5 ms on this machine, so a mass update of 10,000 files would spend about 30 s on commits. That figure is estimated, not measured.
   **Lead's answer: a phase 3 lead item.**
4. **Optional: a marks-only listing in `sm-store`**, for example `(id, file_path, blob, text_hash, status, supersedes_id)` for one project. When nothing has changed, about 200 ms of the 317 ms on Windows goes on loading whole memories.
   **Lead's answer: a phase 3 lead item.** The liveness rule also reads `extra.unresolved_supersedes` and `origin`, so such a listing would need those two as well.

## 9. Unfinished or unverified

- **Not measured:** a sync in which all 10,000 files change. The estimate is in section 8, item 3.
- **The lead's, not done here:** the daemon adapter (`SharingOps`, B-79), `MIGRATION.md` and the command-line subcommands.
- **SPEC 12's exit condition is not run:** "the migration report on a real v1 store shows no lost memories". It needs the owner's store. The fixtures and the 10,000-record synthetic export pass.
- **Re-embedding was tested with `FakeEmbedder` only.** The real model runs in the daemon's queue.
- **Windows-only code:** the `CREATE_NO_WINDOW` path. Linux does not run it.
- **Peak memory was measured differently on the two systems.** On Linux it is `VmHWM` for one process that ran both tests. On Windows it is `PeakWorkingSet64` of a separate process per test, polled every 20 ms, so a peak shorter than 20 ms could be missed.
- **Known gaps left by the review fixes**, each described under its decision:
  - Liveness (decision 14, choice 1): a local superseder that is forgotten leaves its target `superseded`, because `forget` deletes the row that recorded the supersession. A retraction made outside sync is now settled by the next sync. The daemon's `forget` is the place to apply the rule, and that is the lead's.
  - Liveness (decision 14, choice 3): a target whose new superseder arrives already retracted is written twice in that sync and listed in `updated`.
  - Liveness (decision 14, choice 4): a shared target whose file was skipped by its mark and cannot be read again is retried, and its error reported, on every sync while that lasts.
  - Liveness (decision 14, choice 5): a target whose own file says `superseded`, with no live superseder left, costs one blob read on every sync while it stays so.
  - Liveness (decision 14, choice 6): a local target made `active` by hand while a live memory still supersedes it is made `superseded` again by the next sync.
  - Merged memories (decision 37): `import` never changes a merged target's status, even when its line says `retracted`.
  - Redaction (decision 32): project values and project lines are not scanned, and a raw id is not scanned. The `key: value` scan has no left word boundary, so a key that ends in `pwd`, `password`, `secret` or `token` has its string value redacted. Only strings are scanned, so a number under a `password` key is stored unchanged.
  - Import restore (decision 33): a stored target changed locally since the export gets its line's status back when a superseder in the file is newly inserted.
  - Duplicate text (decision 35): a retracted copy still blocks the same text under a new id. The store is unchanged, as the lead asked.
  - Mark spoofing in a crafted import file (decision 36): accepted by the lead, not changed.

## 10. Licences

`Cargo.lock` gains one package: `sm-share 3.0.0`, MIT. It adds no third-party crates. Everything it uses (`serde`, `serde_json`, `windows-sys 0.61.2`, `sm-capture`, `tempfile`) was already in the lock file. Nothing to flag.
