# Stream F report: `sm-share`

Branch `arman-tech/stream-f-share`, from `v3` at `9387a08`. Commits:

- `d85b4c8`: a cherry-pick of the lead's B-86 scanner fix (`89d2a16` on `v3`). It is identical to that commit.
- `b9f90b4`: the crate.
- `7263f00`: the sync mark is stored with the inserted row, and the 10,000-item budget tests are added.

This report is the commit after them.

## 1. Done when

Yes. The sharing and migration tests pass on Windows and on Linux, using the real `RegexSecretScanner` from stream G: all 66 tests in `sm-share`, including AT-8 and the `share` side of AT-9.

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-share/Cargo.toml` | The library crate. Dependencies: `sm-core`, `sm-store`, `serde` and `serde_json`, plus `windows-sys` on Windows only, for `CREATE_NO_WINDOW`. Dev-dependencies: `sm-core` with `test-support`, `sm-capture` (the real scanner, B-8) and `tempfile`. All are workspace pins. `[lints] workspace = true`. |
| `crates/sm-share/src/lib.rs` | Crate documentation and re-exports, with `#![forbid(unsafe_code)]`. |
| `crates/sm-share/src/format.rs` | The shared-file format: a strict writer and parser, body normalisation, the text hash, the path and slug, and the `.gitattributes` rule. |
| `crates/sm-share/src/share.rs` | `share`: redacts, rescans and writes files into the working tree. |
| `crates/sm-share/src/git.rs` | The git runner, with no shell. It wraps `rev-parse`, `ls-tree -r -z -l --full-tree` and a single `cat-file --batch` process, plus the error mapping. |
| `crates/sm-share/src/sync.rs` | `sync`: applies the committed shared folder at `HEAD`, with change detection by blob id. |
| `crates/sm-share/src/migrate.rs` | `migrate`: imports a version 1 or 2 JSON (JavaScript Object Notation) export. It streams the file twice and never builds a vector. |
| `crates/sm-share/src/export.rs` | `export` and `import`: the whole store as JSON Lines, using this crate's own line structures. |
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

**`cargo test -p sm-share --all-features`:**

```
test result: ok. 66 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.69s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**`cargo test --workspace --all-features --no-fail-fast`:** exit 0. The lines are in Cargo's order: `sm_capture`, `sm_core`, `sm_embed`, `real_model`, `sm_share`, `sm_spatial`, `at6_determinism`, `at7_quality`, `s7_pipeline`, `sm_store`, then the doc-tests.

```
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.66s
test result: ok. 66 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.54s
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.63s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.00s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.94s
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.70s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (×6 doc-test lines)
```

**`cargo test -p sm-share --all-features --release budget -- --nocapture --test-threads=1`:**

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 64 filtered out; finished in 5.35s
```

### Linux (Docker)

Machine: image `rust:1.95-trixie` on the WSL2 (Windows Subsystem for Linux 2) kernel 6.6.87.2-microsoft-standard-WSL2. The same CPU (central processing unit), with 32 processors visible. The source was mounted read-only and the target directory was a volume.

**`cargo test -p sm-share --all-features --locked --no-fail-fast`:** exit 0.

```
test result: ok. 66 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.36s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**`cargo clippy -p sm-share --all-targets --all-features --locked -- -D warnings`:** exit 0.

**`cargo test -p sm-share --all-features --locked --release budget -- --nocapture --test-threads=1`:** exit 0.

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 64 filtered out; finished in 4.51s
```

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

**The other 58 tests.** The `s9_`, `s10_` and `s15_` tests cover everything else in the brief's test list:

- the format, with every malformed case refused at its line;
- `share`'s path and slug, the `.gitattributes` file, replacing a file after a type change, the refusals, and `not_git_repository`;
- every `sync` case in the brief, including the full share, commit and sync loop between two stores;
- `migrate`'s B-35 and B-79 cases;
- `export` and `import`: the round trip, refusals, project selection, format and version checks, resuming, links with a missing end, and shared memories with their supersessions.

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

**Timings, in milliseconds.** The Windows figures come from the final run on `HEAD`. Three earlier runs of the same code gave 1,127 to 1,192 ms for the first sync, 306 to 324 ms for the second and 922 ms for `migrate`.

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
12. **`.gitattributes` is not reported.** The folder's own `.gitattributes` is written by `share` and is not a memory, so sync skips it without a report. Every other entry lands in exactly one list of the reply.
13. **Missing `supersedes` targets.** A file whose `supersedes` target is in neither the store nor the files is still added. Its `supersedes_id` is left empty and the target id is kept in provenance `extra.unresolved_supersedes`. When the target arrives in a later sync, it is stored as superseded. Test: `s9_sync_keeps_an_unresolved_supersedes_until_the_target_arrives`.
14. **Supersession is not undone by the target's file.** A target whose own file still says `active` stays `superseded` while a memory of this project supersedes it. Retracting the superseding memory does not undo the supersession either.
15. **A changed `supersedes` on a known memory.** The change is reported, the rest of the file is applied, and no mark is recorded. The report therefore repeats until the file is fixed, and the file is never silently treated as unchanged (B-73).
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
26. **Import access and redaction.** Import uses `Access::Sync`, because it restores shared memories and supersessions that local access may not write. It redacts the title as well as the content.
27. **Supersessions on import.** A memory waits until its `supersedes_id` target has been inserted. After that, the target's status from the file is restored.
28. **Counting.** Valid project lines are not records: they count only in `projects_created`. Memory lines, link lines and unusable lines each land in exactly one of `inserted`, `skipped_existing` and `issues`. An issue's `index` is the 0-based line number, with the header at line 0. Tests: `s15_import_reports_unusable_lines_and_redacts_secrets`, `s15_an_interrupted_import_resumes`.

### Tests

29. **AT-9 uses a hand-written mapper.** `redaction_cases.json` gives the expected text with version 1's `[REDACTED_<KIND>]` placeholders. The test rewrites them as version 3's `[REDACTED:<kind>]` with a small hand-written scanner, because `regex` is not a dev-dependency of this crate. The kinds are compared as the fixture lists them.
30. **The budget tests build their commit with `git fast-import`.** `git add` of 10,000 files took 39.8 s on Windows; `fast-import` takes 0.9 s. `sync` reads only the committed tree, so the work tree is not needed.
31. **Every test git process is isolated.** It gets `GIT_CONFIG_GLOBAL` pointing at an empty file, `GIT_CONFIG_NOSYSTEM=1`, and a fixed author and committer, so the user's own git settings cannot change a result.

## 7. Deviations

1. **New files are marked when they are inserted, not "then marked with `record_sync`".** That quote is from the brief, step 3. The mark has the same layout (`SyncMark::of` reads it), and it is written in the same transaction as the insert. The reason and the evidence are in decision 19: 16.1 s down to 1.3 s for 10,000 files. `s9_sync_adds_a_new_file_as_shared_pending_and_marked` checks the mark.
2. **The export and import round trip differs in one field.** It is identical apart from the header, except for the `updated_at` of a memory that another memory supersedes. That field becomes the time of the import, because `sm-store` stamps it when the superseder is inserted and offers no way to set it back. `s15_export_import_export_round_trips` compares every other byte. The brief allows this when it is stated.

## 8. Contract changes requested

1. **A `git_failed` error code**, or an agreed reuse. Today git missing from `PATH`, or git refusing a repository, is reported as `not_git_repository`, which is misleading for "dubious ownership". The message and next action carry the real cause (decision 10).
2. **An `sm-store` option to keep `updated_at` when a superseder is inserted.** With it, the export and import round trip would be byte-identical (deviation 2).
3. **An `sm-store` `revise_memory` that records the sync mark in the same transaction.** A changed file now costs two commits: `revise_memory`, then `record_sync`. Each commit is about 1.5 ms on this machine, so a mass update of 10,000 files would spend about 30 s on commits. That figure is estimated, not measured.
4. **Optional: a marks-only listing in `sm-store`**, for example `(id, file_path, blob, text_hash, status, supersedes_id)` for one project. When nothing has changed, about 200 ms of the 317 ms on Windows goes on loading whole memories.

## 9. Unfinished or unverified

- **Not measured:** a sync in which all 10,000 files change. The estimate is in section 8, item 3.
- **The lead's, not done here:** the daemon adapter (`SharingOps`, B-79), `MIGRATION.md` and the command-line subcommands.
- **SPEC 12's exit condition is not run:** "the migration report on a real v1 store shows no lost memories". It needs the owner's store. The fixtures and the 10,000-record synthetic export pass.
- **Re-embedding was tested with `FakeEmbedder` only.** The real model runs in the daemon's queue.
- **Windows-only code:** the `CREATE_NO_WINDOW` path. Linux does not run it.
- **Peak memory was measured differently on the two systems.** On Linux it is `VmHWM` for one process that ran both tests. On Windows it is `PeakWorkingSet64` of a separate process per test, polled every 20 ms, so a peak shorter than 20 ms could be missed.

## 10. Licences

`Cargo.lock` gains one package: `sm-share 3.0.0`, MIT. It adds no third-party crates. Everything it uses (`serde`, `serde_json`, `windows-sys 0.61.2`, `sm-capture`, `tempfile`) was already in the lock file. Nothing to flag.
