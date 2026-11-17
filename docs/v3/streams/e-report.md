# Stream E report: `sm-daemon`

Branch `arman-tech/stream-e-daemon`, from `v3` at `9387a08`.

- **First delivery:** `903e05b` (the crate and `Cargo.lock`), `d08aac4` (socket tests), `1d3ed55` (hook tests and the sessionless-capture fix), `1be4769` (acceptance tests) and `dac5744` (the first report and two test changes).
- **Second delivery, after the review:** `e40820a` merges `v3` into the branch (B-94 in `360fe02`, and streams D and F). The commit after it holds the review's fixes and this revised report.

The layout follows `RULES.md` section 9. Sections 1, 2, 3, 4, 6, 8, 9 and 10 changed in the second delivery. Each correction of a claim the first report made says so where it is made.

Abbreviations, written out once:
- AT-n is acceptance test n in SPEC section 11.
- API is application programming interface, and JSON is JavaScript Object Notation.
- LRU is least recently used.
- DACL is discretionary access control list, SDDL is security descriptor definition language, and SID is security identifier.
- ONNX is Open Neural Network Exchange.
- SHA-256 is the 256-bit Secure Hash Algorithm 2.
- p95 is the 95th percentile.
- TCP is the Transmission Control Protocol.
- uid is a Unix user id.

## 1. Done when

**Yes.** Every review item (0 to 7) is fixed, recorded, or both; the table below gives the fix, its tests and its decision.

- **Over the real socket:** each of the 24 operations of CONTRACTS 5 is tested for its success and, where it has one, its main error. `capture` stores what each applicable hook payload should, and `ambient_recall` runs every applicable fixture case.
- **Tests:** the crate has 91 tests: 11 unit, 47 in `socket`, 20 in `hooks` and 13 in `acceptance`. Windows runs 86 of them and Linux 89: three unit tests and two socket tests are Unix-only, and two socket tests are Windows-only. All pass on Windows and on Linux (Docker). `cargo fmt --check` and clippy with `-D warnings` are clean (section 3).

**Corrected: "works end to end."** The first report claimed this, and it did not hold for late vectors. Before B-94, a chunk id could be reused while a bulk batch ran, by any of:
- a revise;
- a forget followed by a remember;
- a reindex that re-chunked.

`set_embeddings` then stored the old text's vectors on the new chunks. B-94 (`360fe02`, merged in `e40820a`) makes the store never give out a chunk id twice. No daemon change was needed, and no daemon-level test was added (the brief made it optional). The store's `s5_a_late_embedding_never_lands_on_a_chunk_that_took_its_place` covers it.

| Review item | Fix | Tests | Decision |
|---|---|---|---|
| 0. Merge `v3` (B-94) | Merged in `e40820a`; no daemon change needed. | `sm-store`'s `s5_a_late_embedding_never_lands_on_a_chunk_that_took_its_place` | (B-94) |
| 1. A failed capture was lost (must fix) | The pipeline runs on a copy of the session, which is committed only after the store succeeds. Failures that may pass are retried with a doubling wait, up to 5 attempts. At the bound, `spool/failed/` gets the file and the log names the request id and the cause. | `s16_a_failed_capture_is_retried_and_its_session_moves_on_only_once_stored`, `s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound`, `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once`, `at15_spool_is_drained` | 27 |
| 2. A capture in process was lost at shutdown | A capture is a file in `spool/new/` from the moment it is taken until it is stored; the drain skips the files a job holds. The lock is released after the threads stop, or after the grace with every held capture on disk. | `at15_a_capture_held_at_shutdown_is_stored_after_a_restart`, `at15_a_drained_file_stays_in_new_until_its_capture_is_stored` | 28, 29 |
| 3. The project cache was never refreshed | The identity is resolved on every use. The cache keeps only the stored project for an identity. | `s16_capture_disabled_in_project_toml_counts_without_a_restart`, `s9_a_folder_that_becomes_a_repository_is_one_without_a_restart` | 30 |
| 4. Binding on macOS | When interprocess reports `Unsupported`, the socket is bound in a folder only this user can enter, set to 0600, then renamed into place. | `s3_a_socket_bound_in_a_private_folder_is_user_only_and_moved_into_place` (Linux, called directly; unverified on macOS) | 31 |
| 5. A fallback socket in `/tmp` was followed without an owner check | `connect` refuses a socket another user owns, checked with `lstat` before connecting and again after. `bind` refuses to replace one. | `s3_a_socket_another_user_owns_is_refused_by_connect_and_bind`, `s3_the_owner_check_does_not_follow_a_link` | 32 |
| 6a. A `record_injections` error was dropped | Logged with the session, the count and the cause. | none: the store offers no way to make that call fail from outside | 21 |
| 6b. Replies could exceed their budget | Every reply fits. The first item sheds fields before it may be left out, the second list gets only the room left, and a skeleton over the limit is refused. | `s8_no_reply_exceeds_its_limit_in_the_worst_case`, `s8_a_reply_fits_its_limit_and_counts_what_it_left_out`, and the unit tests `s8_a_first_memory_that_cannot_fit_whole_is_cut_to_fit`, `s8_a_whole_item_over_the_budget_is_left_out` and `s8_a_limit_below_the_fixed_fields_is_refused` | 24 |
| 6c. A map run computed before a drop stayed indexed | A generation counter: a run is indexed only under the generation read before its vectors. | unit `s7_a_run_computed_before_a_drop_is_never_indexed` | 34 |
| 6d. `doctor_eval` opened a second bulk session | `doctor_eval` embeds through the queue's one bulk session, a batch at a time. | `at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget` | 35 |
| 6e. A failing batch blocked the queue, and a panic ended it | The batch is retried one chunk at a time. A chunk that fails alone 3 times is skipped and logged, and stays pending. Panics are caught. | `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on` | 33 |
| 6f. Relative file paths were accepted | Refused in dispatch with `invalid_input` and a next action. | `s9_a_relative_file_path_is_refused_before_the_op_runs` | 36 |
| 6g. Ranks were numbered before vanished memories were dropped | `recall` and `related` number ranks after the drop. | none: the window is a race between the search and the read of the memories; the order of the code now rules it out | (none) |
| 6h. No read timeout or connection cap | Recorded, not changed. | (none) | 37 |
| 7. Claims the code did not support | Corrected: section 1 (late vectors), section 2, section 4 AT-3, decisions 7, 11, 23 and 24, and section 10. | (none) | (none) |

**Mutation evidence.** Each fix of items 1, 2 and 6e, and of 6b and 6d, was removed in turn. The test named failed each time, and passed again with the fix restored (Windows, `cargo test -p sm-daemon --all-features --test <binary> <name>`). The fix-removed source was restored byte for byte (checked with `cmp`) before the gates in section 3 ran.

| Item | Fix removed | Test | Result |
|---|---|---|---|
| 1 | The session is committed before `store_candidates` (the old order) | `s16_a_failed_capture_is_retried_and_its_session_moves_on_only_once_stored` | Failed at `hooks.rs:1106`: "the capture was stored on its second attempt" |
| 1 | Every error counts as permanent (every failure quarantined, the old behaviour) | the same | Failed at the same line |
| 1 | No attempt bound | `s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound` | Failed at `hooks.rs:1168`: "both moved to spool/failed: []" |
| 2 | A socket capture is not spooled before it is processed | `at15_a_capture_held_at_shutdown_is_stored_after_a_restart` | Failed at `hooks.rs:1319`: "a socket capture is written to spool/new before it is processed", 0 files where 1 was expected |
| 2 | The drain may take a file a job holds | `at15_a_drained_file_stays_in_new_until_its_capture_is_stored` | Failed at `hooks.rs:1279`: "the drain never queued the held file a second time", 6 calls where 1 was expected |
| 2 | A drained file is removed once queued (CONTRACTS 7 as written) | the same | Failed at `hooks.rs:1264`: "the file stays in spool/new while its capture is processed" |
| 6e | No `catch_unwind` in `with_bulk` | `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on` | Failed at `socket.rs:1419` with 5 chunks pending; `sm-bulk` panicked and stopped |
| 6e | The one-chunk-at-a-time retry never runs | the same | Failed at `socket.rs:1419` with 5 chunks pending |
| 6e | No skip (`SKIP_AFTER` at `u32::MAX`) | the same | Failed at `socket.rs:1419`: 2 chunks pending, but no skip line in the log |
| 6b | A cut first item is kept even when still over its room | `s8_no_reply_exceeds_its_limit_in_the_worst_case`; unit `s8_a_first_memory_that_cannot_fit_whole_is_cut_to_fit` | Failed at `socket.rs:761`: "recall at limit 100: 103 tokens"; the unit test failed at `budget.rs:306` |
| 6b | A cut memory never sheds a field | the same two | Failed at `socket.rs:879`: "a cut first result is kept at some limit"; the unit test failed at `budget.rs:318` |
| 6d | `doctor_eval` opens its own bulk session (the old behaviour) | `at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget` | Failed at `acceptance.rs:501`: "12 bulk threads alive at once, budget 6; sessions opened [(Query, 4), (Bulk, 6), (Bulk, 6)]" |

The line numbers are those of the tree at the time of each run; the socket test added for 6b moved some later lines of `socket.rs`.

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-daemon/Cargo.toml` | The new library crate. It depends on `sm-core`, `sm-store`, `sm-embed`, `sm-spatial`, `sm-capture`, `interprocess`, `serde` and `serde_json`; `windows-sys` on Windows and `libc` on Unix; `sm-core` with `test-support` and `tempfile` for development only. Every version comes from the root pins; no feature was added. It does not depend on `sm-share` (B-79). Unchanged in the second delivery. |
| `src/lib.rs` | Crate documentation (why threads, no process-global state), `#![deny(unsafe_code)]`, and the public API below. |
| `src/server.rs` | `start` and `run`: the lock, the store and its index stamp, the endpoint, then the loader thread; the accept loop with one thread per connection; the frame loop; the worker threads; `embedder_cold_start_seconds`; the shutdown order of decision 29, which `DaemonHandle::drop` follows too; the logged `record_injections` error. |
| `src/endpoint.rs` | `bind` with retry, `connect` and `accept`. Unix: the owner check (`lstat`) in `connect`, before and after connecting, and in `bind`; this user's leftover socket removed; mode 0600, with the private-folder fallback of decision 31. Windows: the first pipe instance, remote clients rejected, and the user-only descriptor, plus `current_user_sid` and `pipe_dacl_sddl` to read the DACL back. The crate's only unsafe code is in this file's two `security` modules: the Win32 calls, and `geteuid` on Unix. |
| `src/lock.rs` | `DaemonLock`: `daemon.lock` opened without truncation and never written, `try_lock` every 25 ms for up to 2 s, held for the daemon's life, released last. |
| `src/loader.rs` | The `ModelLoader` trait (`info` before `load`), `LoadedModel` (provider and chunker), and `GemmaLoader`: `verify` on the model directory, then `GemmaProvider` and its tokenizer-based chunker. |
| `src/dispatch.rs` | One request: `Envelope::peek`, `version_mismatch`, unknown keys, the profile check, `Op::validate` and the absolute-path check (decision 36), the model-state check, the deadline, the op table, and the work done after the reply is written. |
| `src/state.rs` | `Shared`, one per daemon: the model state (`Loading`, `Ready`, `Failed`), the caches, the capture queue and the names of the spool files its jobs hold, the one bulk session, the map index with its generation (decision 34), the wake signals and an LRU map. |
| `src/options.rs` | `DaemonOptions`: the endpoint and every timing and size, injectable for tests, with the defaults in section 6 (new: `capture_retry_backoff`, 2 s). |
| `src/log.rs` | `DaemonLog`: one write call per line, messages cut at 2,000 characters, rotation to `daemon.log.1` at 1 MiB. |
| `src/project.rs` | `resolve_project` on each request's `cwd`, every time. An LRU of 256 per folder keeps the stored project for the identity last seen there, so the database is written only when the identity changes (decision 30). |
| `src/queue.rs` | The bulk embedding worker: batches of `embed.batch_size` and one `set_embeddings` per batch. It uses the daemon's one bulk session (`with_bulk`, shared with `doctor_eval`), opened on demand, closed after the grace period, and with panics caught. The maps are dropped after each batch. A storage failure or refused index backs off; any other failure is retried one chunk at a time, and a chunk that fails alone 3 times is skipped (decision 33). |
| `src/capture.rs` | `capture`: the reply once queued, and one worker running `sm_capture::capture` on a copy of the session. Captures stay on disk until stored (decision 28), and are retried with a bound (decision 27). Also: sessions and transcript offsets, storing through `remember`'s path, the capture metrics, and spooling when it cannot process. |
| `src/ambient.rs` | `ambient_recall`: the four-word minimum, the project, the model state, the prompt cut, embedding, search, `ambient::apply_gates`, and the injections recorded after the reply. |
| `src/maintenance.rs` | The maintenance thread: the spool drain at start and every 60 s (an accepted file stays in `spool/new/`, held by its job; a held file is skipped), the daily pruning of injections and journal files, and the idle exit. |
| `src/admin.rs` | The command-line and sharing ops: `share`, `sync`, `import`, `export`, `migrate` through `SharingOps`; `reindex`; `project_rename`; `project_delete`; `doctor_eval`, which embeds a scratch store's chunks on the queue's bulk session. |
| `src/sharing.rs` | The `SharingOps` trait, `SharingContext`, `ResolvedProject`, and `UnwiredSharing`. |
| `src/budget.rs` | Output budgets (decision 24): the skeleton and its refusal, in-order fitting, the first item's cut with its shedding order, and the nested cuts of regions, duplicate groups, linked memories, journey steps and duplicates. |
| `src/view.rs` | `MemoryView` construction, with project names. |
| `src/tools/mod.rs` | The tool modules. |
| `src/tools/remember.rs` | `remember`, and the store path every new memory takes: redact, title, chunk, insert with pending chunks, wake the queue, journal `chunks_per_memory`. |
| `src/tools/recall.rs` | `recall` (vector, keyword, auto with appended keyword hits), weak marking, the recall metrics, the per-session ranks (numbered after vanished memories are dropped), and `reinforce`. |
| `src/tools/related.rs` | `related`: vector neighbours and the two-hop walk over links and anchors; ranks numbered after the drop; the link list fitted in the room the vector list leaves. |
| `src/tools/edit.rs` | `revise` (in place or `supersede_with`) and `forget` (by id or filter, with the confirm rule). |
| `src/tools/link.rs` | `link`: add, remove and list, with staleness from the head commit. |
| `src/tools/maps.rs` | `regions` and `journey` through `sm-spatial`, the map cache and its generation, coordinates when asked, labels; regions fitted before points. |
| `src/tools/consolidate.rs` | `consolidate`: propose near-duplicate groups, or apply exactly the listed groups. |
| `src/tools/status.rs` | `status`: state and `state_error`, model, index stamp, pending embeddings, projects, spool, and the journal summary when asked. |
| `tests/common/mod.rs` | The test rig: a temporary profile, the endpoint passed in, a client over `interprocess`, test loaders (fake, gated, failing, closure), a recording provider, a recording `SharingOps`, and helpers. Unchanged in the second delivery. |
| `tests/socket.rs` | 47 tests (45 run on each platform; 2 are Unix-only, 2 Windows-only): the lock, the endpoint, frames and envelopes, every op with its success and main error, the budget's worst case, the skipped chunk, relative paths, and a folder that becomes a repository. |
| `tests/hooks.rs` | 20 tests: every hook event, daemon side, with the capture and ambient fixtures (AT-13 to AT-17), plus retries, held files, shutdown with a capture held, and `capture.enabled` read afresh. The faults come from `FaultChunker`, a test chunker the loader hands the daemon (decision 27). |
| `tests/acceptance.rs` | 13 tests: AT-3 (daemon side, two tests), AT-4, AT-5, AT-9 (`remember` side), AT-10, AT-11, AT-12, the SPEC 6 recall rules, the SPEC 7 map cache, and B-79's dropped maps. |
| Unit tests in `src/` | 11: budget (3), map generation (1), endpoint (3, Unix-only), and the four of the first delivery. |
| `Cargo.lock` | 5 packages added by the first delivery (section 10); the merge of `v3` brought the lock up to date with `v3`. No version was changed by this stream. |
| `docs/v3/streams/e-report.md` | This report. |

### Public API

Unchanged in the second delivery.

```rust
pub mod endpoint;   // connect; on Windows also current_user_sid, pipe_dacl_sddl
pub use endpoint::connect;
pub use loader::{GemmaLoader, LoadedModel, ModelLoader};
pub use lock::DaemonLock;
pub use log::DaemonLog;
pub use options::DaemonOptions;
pub use server::{DaemonHandle, Exit, Started, run, start};
pub use sharing::{ResolvedProject, SharingContext, SharingOps, UnwiredSharing};
```

- `run(paths, settings, loader, sharing, opts) -> Exit` is what the `daemon` subcommand calls. It blocks until shutdown, the idle exit, or another daemon holding the lock (`Exit::LockHeld`, which is exit 0).
- `start(...) -> Result<Started>` is the same, without blocking: `Started::LockHeld` or `Started::Running(DaemonHandle)`. The handle has `endpoint`, `log_path`, `stopping`, `shutdown(reason)` and `wait`. Dropping it stops the daemon in the same order as `wait`.
- `ModelLoader` has `info() -> Result<ModelInfo>`, read before loading so the index stamp is configured at start, and `load() -> Result<LoadedModel>`. `GemmaLoader::new(model_dir, &settings)` is the real one.
- `SharingOps: Send + Sync` has `share`, `sync`, `migrate`, `import` and `export`. Each takes a `SharingContext` and the op's parameters, and returns the op's reply type from `sm_core::protocol`. The context holds:
  - the `Store` and the chunker, if loaded;
  - the secret scanner and `remember`'s `ChunkParams`;
  - the resolved project with its identity.
- `endpoint::connect` refuses, on Unix, a socket another user owns. The `sm-cli` hook client needs the same check (section 8, request 4).

## 3. Test commands and exact results

**Windows.** Windows 11 Pro 10.0.26300, x86_64, AMD Ryzen 9 9950X3D (16 cores, 32 logical), 93.6 GB, rustc 1.95.0, PowerShell, `test` profile (debug). On the final tree:

- `cargo fmt --check` exited 0.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` exited 0.
- `cargo test --workspace --all-features -- --nocapture` exited 0. Every `test result:` line, in order:

```
sm_capture (unit)        test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
sm_cli (unit)            test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_cli (binary)          test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_cli autostart         test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.07s
sm_cli hook_budgets      test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 65.43s
sm_cli hook_fixtures     test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.19s
sm_cli mcp_proxy         test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s
sm_cli ops               test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
sm_cli plugin            test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.72s
sm_cli setup             test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
sm_core (unit)           test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon (unit)         test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon acceptance     test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.62s
sm_daemon hooks          test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.18s
sm_daemon socket         test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.59s
sm_embed (unit)          test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
sm_embed real_model      test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.98s
sm_share (unit)          test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.43s
sm_spatial (unit)        test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
sm_spatial at6           test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.97s
sm_spatial at7           test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
sm_spatial s7_pipeline   test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.83s
sm_store (unit)          test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.75s
doc-tests (8 crates)     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s  (each)
```

The crate and binary names on the left are added here; each line after them is copied from the log.
- The `real_model` binary belongs to `sm-embed` and ran because the model is in the per-user cache; no `sm-daemon` test loads the model.
- The "panicked" lines in the `hooks` and `socket` output are the panics those tests inject: "injected chunker panic" (`s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound`) and "injected embedder panic" (`s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on`). The daemon catches both and keeps running.

**Linux (Docker).** `rust:1.95-trixie` on the same machine, the source mounted read-only, as in `CLAUDE.md`, with `cargo fmt --check` added:

```
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --check
```

All three exited 0, and the container exited 0. Clippy printed no warning, and `cargo fmt --check` printed no difference. Every `test result:` line, in order:

```
sm_capture (unit)        test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
sm_cli (unit)            test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_cli (binary)          test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_cli autostart         test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.92s
sm_cli hook_budgets      test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.37s
sm_cli hook_fixtures     test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.44s
sm_cli mcp_proxy         test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
sm_cli ops               test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
sm_cli plugin            test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
sm_cli setup             test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_core (unit)           test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
sm_daemon (unit)         test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon acceptance     test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.58s
sm_daemon hooks          test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.19s
sm_daemon socket         test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.41s
sm_embed (unit)          test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
sm_embed real_model      test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_share (unit)          test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.08s
sm_spatial (unit)        test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
sm_spatial at6           test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.08s
sm_spatial at7           test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
sm_spatial s7_pipeline   test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.68s
sm_store (unit)          test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.60s
doc-tests (8 crates)     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s  (each)
```

- The names on the left are added here, as for Windows.
- `real_model` passes by skipping there: the log says "SKIPPED: the model folder ... is absent", because the container has no model.
- The counts that differ from Windows outside `sm-daemon` (`sm_core`, `sm_cli`, `autostart`) are those crates' own platform-only tests.

**Platform-only tests.**
- Unix-only, run on Linux:
  - the socket tests `s3_a_leftover_socket_file_is_removed` and `s3_the_socket_is_user_only`;
  - the unit tests `s3_a_socket_bound_in_a_private_folder_is_user_only_and_moved_into_place`, `s3_a_socket_another_user_owns_is_refused_by_connect_and_bind` and `s3_the_owner_check_does_not_follow_a_link`.
- Windows-only, run on Windows: `spike1_a_second_first_instance_of_the_pipe_is_refused` and `spike1_the_pipe_dacl_admits_only_the_current_user`.

**Test changes in the second delivery.**
- **`s8_a_reply_fits_its_limit_and_counts_what_it_left_out` asserted a defect.** At limit 100 it expected a cut first result, but that result was over the limit; this was the defect of review item 6b.
  - It now asserts what a correct reply does at 100: no result, one omitted, `truncated` set, within the limit. It checks the cut first result at 150.
  - Its three-results case computes the limit from the reply's skeleton instead of the old fixed 40 tokens.
  - The assertions are no weaker. The test checks the true behaviour at both limits, and `s8_no_reply_exceeds_its_limit_in_the_worst_case` adds the worst case.
- **The other tests of the first delivery are unchanged.** Every other new test is listed in section 1.

## 4. Acceptance tests

All pass on Windows and Linux.

| Test | Test names | What is checked |
|---|---|---|
| AT-3, daemon side | `at3_warm_ambient_latency_idle_and_during_a_bulk_import_within_the_thread_budget`, `at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget` | 100 warm `ambient_recall` calls idle and 100 during a bulk import of 2,000 chunks, each p95 ≤ 300 ms. The bulk session never has more live threads than `bulk_threads`, and each session opens with its budget (the provider records it, standing in for `ThreadReport`). **Corrected:** the first report missed that `doctor_eval` opened a second bulk session while the queue's could be open, so bulk threads could reach twice the budget. `doctor_eval` now shares the queue's session (decision 35). The second test runs a `doctor_eval` with its own memories during an import and sees at most `bulk_threads` alive and one bulk session opened. |
| AT-4 | `at4_a_50000_character_memory_is_stored_in_several_chunks_and_embedded` | A 50,000-character `remember` succeeds with at least 10 chunks; the queue embeds all of them; the memory vector exists; a vector recall of a phrase from its middle finds it. |
| AT-5 | `at5_two_readers_and_one_writer_for_60_seconds_see_no_lock_error` | 60 seconds of two readers (vector, auto and keyword `recall`; `status` and `ambient_recall` over all projects) and one writer (`remember`), each on its own connection. No error reaches a client, and every write is stored. |
| AT-9, `remember` side | `at9_remember_redacts_the_fixture_credentials_and_lists_the_kinds` | Every case of `fixtures/capture/redaction_cases.json` goes through `remember`. The stored content equals the fixture's redacted text in v3 markers, and no fake credential is in the content or the title. The reply's `redactions` equal the fixture's kinds, and the empty input gets `invalid_input`. The capture side is in `at13_hook_payloads_over_the_socket`. |
| AT-10 | `at10_one_changed_byte_fails_the_model_and_the_rest_is_served` | The real `sm_embed::model::verify_files` runs in the loader, over a small file set: one 4 KiB file, with its size and SHA-256 pinned. Intact, the daemon is `ready`. With one byte changed: `hello` and `status` say `failed` with `model_integrity`, model requests fail with that error, and `forget`, `status` and keyword `recall` are served (B-36). See section 7. |
| AT-11 | `at11_a_changed_chunk_size_or_model_blocks_search_until_reindex` | A changed `chunk.max_tokens`, and a loader with another model id, each give exactly one stamp difference. Vector, auto and keyword `recall` then fail with `index_mismatch`. `reindex` re-chunks every memory and queues the embedding, and search works again by vector and keyword. |
| AT-12 | `at12_a_memory_of_project_a_is_not_returned_from_b_without_scope_all`, `at12_two_profiles_never_see_each_others_memories` | Project B's recall never returns project A's memory unless `scope: all`. Two daemons on two profiles have separate databases and never return each other's memories. |
| AT-13, daemon side | `at13_hook_payloads_over_the_socket`, `at13_default_settings_store_strong_text_and_nothing_for_weak` | Every applicable payload of `fixtures/capture/hook_payloads/` is sent as `capture`. The stored memories match the fixture in content, type, status, importance, provenance `capture` and redactions; weak text stores nothing. Includes the Stop loop guard (`cc_stop_loop_guard_active`, `cc_stop_handler_loop_guard`). |
| AT-14, daemon side | `at14_transcript_stores_only_what_post_tool_use_missed`, `s16_transcript_offsets_per_session`, `s16_a_failed_capture_is_retried_and_its_session_moves_on_only_once_stored` | With `fixtures/capture/rust/at14_session.jsonl`, PreCompact and Stop store the assistant text PostToolUse missed and not what it captured. Offsets are kept per session, a capture with no session keeps no state, and a session's offset moves only once its capture is stored. Printing the Stop continue response and PreCompact's empty output are the hook's (stream D). |
| AT-15, daemon side | `at15_spool_is_drained`, `at15_a_drained_file_stays_in_new_until_its_capture_is_stored`, `at15_a_capture_held_at_shutdown_is_stored_after_a_restart` | A capture spooled while no daemon runs is processed at start, and its file removed once stored. A drained file stays in `spool/new/` while its capture is processed and is not queued twice. A capture being processed at shutdown is still in `spool/new/` after the daemon stops, and the next daemon stores it once. A corrupt file moves to `spool/failed/`. A stale `tmp/` file is removed and a fresh one kept. A file written while the daemon runs is taken within one drain interval (500 ms in the test). The 200 ms hook exit with the daemon stopped is stream D's. |
| AT-16, daemon side | `at16_tiers_are_stored_active` | Medium-signal text is stored active with importance ≥ 0.4 and strong-signal text with ≥ 0.7, from PostToolUse, PreCompact and Stop. |
| AT-17, daemon side | `at17_ambient_block_is_the_reply_text_and_never_repeats`, `at17_capture_over_the_index_cap_is_stored_whole` | `AmbientReply.text` is exactly the block `sm_core::ambient` builds, for Claude Code and Cursor, and no memory repeats in a session. A capture over `capture.max_index_tokens` is stored whole, indexed up to the cap, and journalled as `capture_cap_hit_total` with `capture_size_tokens`. `SPATIAL_MEMORY_AMBIENT_RECALL=off` is the hook's (stream D). |

**The other rules the brief lists**, all passing:

- **Every op** (CONTRACTS 5, B-28):
  - in `socket.rs`, the `s8_*`, `s9_*` and `s16_capture_*` tests, and `s5_reindex_requeues_every_chunk`, `s11_doctor_eval_runs_the_retrieval_set_and_its_error` and `s6_ambient_recall_answers_and_refuses_a_blank_prompt`;
  - unknown fields: `s3_unknown_fields_are_refused_for_every_op`;
  - relative paths: `s9_a_relative_file_path_is_refused_before_the_op_runs`;
  - the budget: `s8_a_reply_fits_its_limit_and_counts_what_it_left_out` and `s8_no_reply_exceeds_its_limit_in_the_worst_case`.
- **CONTRACTS 3.1 and 4:**
  - lock and start: `s3_two_daemons_on_one_profile_give_one_server_and_lock_held`;
  - shutdown: `s3_shutdown_releases_the_lock_and_a_new_daemon_starts`, `s3_shutdown_over_the_socket_stops_the_daemon_with_its_reason` and `at15_a_capture_held_at_shutdown_is_stored_after_a_restart`;
  - the Unix socket: `s3_a_leftover_socket_file_is_removed`, `s3_the_socket_is_user_only`, and the three Unix unit tests of section 3;
  - versions and profiles: `s3_another_build_gets_version_mismatch`, `s3_shutdown_from_another_build_works_for_this_profile_only` and `s3_a_request_for_another_profile_is_refused`;
  - frames: `s3_an_empty_frame_is_refused_and_the_connection_stays_usable`, `s3_a_malformed_body_is_refused` and `s3_an_oversized_frame_closes_the_connection_and_the_daemon_still_serves`;
  - serving: `s3_requests_on_many_connections_at_once_are_served`, `s3_a_passed_deadline_gives_timeout`, `s3_hello_says_loading_and_writes_wait_while_reads_are_served` and `s3_hello_reports_identity`;
  - idle exit: `s3_the_daemon_exits_when_idle` and `s3_idle_exit_waits_while_requests_keep_coming`;
  - the log: `s3_the_daemon_log_stays_bounded`.
- **Windows endpoint:** `spike1_a_second_first_instance_of_the_pipe_is_refused`, `spike1_the_pipe_dacl_admits_only_the_current_user` (reads the DACL back and compares it with `D:P(A;;GA;;;<SID>)`).
- **SPEC 5, bulk queue:** `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on`.
- **SPEC 6:**
  - `s6_identifier_query_appends_marked_keyword_hits_after_vector_hits` and `s6_weak_results_are_marked_not_dropped`;
  - `s6_ambient_gates`, `s6_ambient_prompt_cases`, `s6_ambient_errors_and_loading`, `s6_ambient_deadline_records_nothing` and `s6_ambient_fixture_integration_data`;
  - the unit tests `s6_identifier_queries_run_keyword_search` and `s6_anchor_start_vector_is_the_normalized_mean`.
- **SPEC 7:** `s7_regions_cached_until_an_edit_and_coordinates_only_when_asked`, and the unit tests `s7_each_run_variant_knows_its_sibling` and `s7_a_run_computed_before_a_drop_is_never_indexed`.
- **SPEC 9 and SPEC 16, project read afresh:** `s9_a_folder_that_becomes_a_repository_is_one_without_a_restart`, `s16_capture_disabled_in_project_toml_counts_without_a_restart`.
- **SPEC 16, never loses:** `s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound`, `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once`, and the AT-14 and AT-15 tests above.
- **B-79:**
  - `s9_sharing_ops_reach_the_implementation_with_the_resolved_project`: each op reaches the fake with the resolved project and identity, then the queue wakes;
  - `s9_import_drops_the_maps_and_its_memory_is_embedded`: the cached run is dropped and the imported memory embedded;
  - `s9_share_and_sync_need_a_repository_and_the_wired_implementation`;
  - `s9_import_export_and_migrate_without_the_implementation_say_so`.

### Hook payloads that cannot apply

`fixtures/capture/hook_payloads/INDEX.json` has 75 entries: 45 are sent as `capture` and compared, and 30 cannot apply to the daemon.

- **Raw standard input (4):** `cc_post_tool_use_empty_stdin`, `cc_post_tool_use_invalid_json`, `cc_pre_compact_empty_stdin`, `cc_stop_empty_stdin`. The payload is the hook's raw input text, which never reaches the daemon as JSON. Parsing standard input is stream D's.
- **Not a capture event (26):** the daemon refuses a SessionStart or UserPromptSubmit `capture` with `accepted: false` (`s16_capture_refusals`), and these cases check the hook's printed output (stream D).
  - SessionStart: `cc_session_start_startup`, `cc_session_start_resume`, `cc_session_start_clear`, `cc_session_start_compact`, `cc_session_start_unknown_source`, `cc_session_start_missing_source`, `cc_session_start_empty_stdin`, `cc_session_start_whitespace_stdin`, `cc_session_start_not_json`, `cc_session_start_json_list`, `cc_session_start_output_format`, `cc_session_start_handler_empty_source`.
  - UserPromptSubmit: `cc_user_prompt_submit_recall_off`, `cc_user_prompt_submit_off_unit_payload`, `cc_user_prompt_submit_formatted_block`, `cc_user_prompt_submit_empty_prompt`, `cc_user_prompt_submit_blank_prompt`, `cc_user_prompt_submit_missing_prompt`, `cc_user_prompt_submit_recall_raises`, `cc_user_prompt_submit_empty_results`, `cc_user_prompt_submit_namespace_env`, `cc_user_prompt_submit_default_top_k`. The daemon's side of the prompt hook is `ambient_recall`, tested from `ambient_recall_cases.json`.
  - Cursor: `cursor_sessionStart_synthesized_source`, `cursor_sessionStart_clear`, `cursor_sessionStart_detected_by_version`, `cursor_userPromptSubmit_block`.
- **`transcript_cases.json`:** two cases do apply: `load_empty_session_id` and `save_empty_session_id_noop`, because a sessionless capture keeps no state (decision 8). The others cannot:
  - `load_corrupted_json_returns_defaults`: v3 keeps offsets in the daemon's memory, so there is no state file to corrupt;
  - the state probes: v3 builds no path from a session id.

### Ambient cases that cannot apply

From `fixtures/capture/ambient_recall_cases.json`, these are covered:
- the empty, blank and missing prompt (`invalid_input`);
- `recall_returns_formatted_block` (AT-17);
- `recall_raises_returns_none` (`s6_ambient_errors_and_loading`);
- `recall_empty_results_returns_none` (the `empty` gate in `s6_ambient_gates`);
- all 9 prompt probes (`s6_ambient_prompt_cases`): non-strings get `invalid_input` and short strings `short_prompt`;
- the misc probes `client_cursor_same_output`, `payload_without_session_or_cwd`, `hybrid_recall_raises` and `build_factory_raises`.

These cannot apply:

- **The hook's switches:** `ambient_recall_off_returns_none`, the `ambient_recall_env` probes (13) and the `top_k_env` probes (14). The hook resolves them (`config::resolve_ambient`, tested in sm-core `at17_version_2_ambient_switches_keep_their_rules`) and sends the resolved `top_k`.
- **Namespaces:** `namespace_env_passed_through` and the `namespace_env` probes (4). v3 has projects, not namespaces.
- **Python object shapes:** the misc probes `client_other_string`, `result_without_memories_attribute`, `memories_null` and `similarity_null_escapes`. v3's replies are typed; a reply without memories or with a null similarity cannot be built.
- **The block format:** the `format` probes (15). They test `sm_core::ambient::format_block` (sm-core `at17_ambient_block_keeps_version_2_format`); the daemon returns its output unchanged (AT-17 above).
- **Environment round trip:** `env_round_trip` (8 entries), the hook's handling of its environment.

## 5. Measured budgets

The setup:
- the machine of section 3, Windows 11, `test` profile (debug);
- the fake embedder;
- `physical_cores` fixed at 8 in `DaemonOptions`, so the budget is the same on every machine.

These are daemon-path figures only; the real-model figures are phase 3.

The figures come from the second delivery's run in section 3 (printed with `--nocapture`) and, to show the spread, from the first delivery's final run on the same machine.

| What | Second delivery | First delivery | Budget the test asserts |
|---|---|---|---|
| AT-3, warm `ambient_recall`, idle (100 calls) | median 888.6 µs, p95 3.899 ms | median 547.6 µs, p95 961.6 µs | p95 ≤ 300 ms |
| AT-3, the same during a bulk import of 2,000 chunks (100 calls) | median 3.285 ms, p95 4.132 ms; 1,764 chunks still pending after | median 4.097 ms, p95 5.402 ms; 1,728 pending after | p95 ≤ 300 ms; the import still running after the calls |
| AT-3, threads | bulk budget 6 of 8, at most 6 alive; query session 4; sessions opened `[(Query, 4), (Bulk, 6)]` | the same | live bulk threads ≤ the bulk budget; each session opened with its budget |
| AT-3, `doctor_eval` during the import (new test) | bulk budget 6, at most 6 alive; 540 chunks pending before the evaluation, 492 after | (no such test) | live bulk threads ≤ the bulk budget; one bulk session |
| AT-4, a 50,000-character `remember` | 26 chunks, all embedded | 26 chunks, all embedded | at least 10 chunks; nothing pending |
| AT-5, 60 seconds, two readers and one writer | reader a 1,943 requests, reader b 5,942, writer 8,976; no error | 2,380, 7,064, 8,102; no error | no error; every write stored |
| AT-9, `redaction_cases.json` through `remember` | 33 cases redacted, 7 left unchanged | the same | every case equal to the fixture |
| AT-15, a spool file written while the daemon runs | taken in 518.7 ms (interval 500 ms) | 452.3 ms | gone within the interval plus 1 s |

**Reading the figures.**
- *During the import* the median rises from under 1 ms to 3 to 4 ms, far inside the budget, so a query is not held behind the bulk queue.
- *The idle p95* differs between the two runs (0.96 ms and 3.9 ms). Both are far inside the budget, and the whole workspace's tests run at the same time.
- *What carries over:* only that shape, not the numbers. This is a debug build with a fake embedder that sleeps 3 ms per document.
- *The `doctor_eval` row:* the pending count falls while the evaluation runs, because the queue and the evaluation take turns on the one session (decision 35).

For scale, `sm-embed`'s real-model test in the same run printed:
- a median query embedding of 47.1 ms at 128 tokens on 4 threads;
- a peak of 1,416 MB with both sessions loaded.

Those are `sm-embed`'s figures, not measured through the daemon.

## 6. Decisions

Each needs a B-n entry from the lead. Decisions 6, 7, 11, 21, 23 and 24 changed in the second delivery; 27 to 37 are new.

1. **Threads, not tokio (SPEC 14 stream E build 2; SPEC 3).**
   - *What:* one accept thread, one thread per connection, a loader thread, the bulk worker, the capture worker and the maintenance thread, over `interprocess`'s blocking listener. No async runtime and no extra crate feature.
   - *Why:* every request is short and blocks on SQLite or ONNX Runtime, so a runtime would add a dependency and nothing else; a slow op holds only its own connection's thread.
   - *Evidence:* `s3_requests_on_many_connections_at_once_are_served`; AT-3's p95 while the bulk queue runs (section 5); AT-5 with three busy connections.
2. **The Windows pipe descriptor (SPIKE c; spike decisions 1 and 5).**
   - *What:* SDDL `D:P(A;;GA;;;<current user SID>)`, protected from inheritance. It is made with `ConvertStringSecurityDescriptorToSecurityDescriptorW`, copied into an owned `interprocess::os::windows::security_descriptor::SecurityDescriptor`, and passed with `ListenerOptionsExt::security_descriptor`.
   - *Verified in the `interprocess` 2.4.8 source:* it adds `FILE_FLAG_FIRST_PIPE_INSTANCE` itself when it creates the first instance, and `PIPE_REJECT_REMOTE_CLIENTS` is its default.
   - *Evidence:* `spike1_a_second_first_instance_of_the_pipe_is_refused`, `spike1_the_pipe_dacl_admits_only_the_current_user`.
3. **Requests that need the model are refused with `daemon_loading` while it loads (stream E build 3).**
   - *What:* these need the loaded model and get `daemon_loading` until it is ready:
     - `remember`, and `revise` with new content or `supersede_with`;
     - vector and auto `recall`, and `ambient_recall`;
     - `reindex` and `doctor_eval`.
   - *Served while loading:* keyword `recall`, `forget`, `link`, `related` (stored vectors), `status`, `regions`, `journey` and `consolidate`.
   - *Capture:* a `capture` is accepted and queued while loading; the worker waits for the model, and spools the queue if loading fails.
   - *Why:* the chunker needs the tokenizer, which loads with the model, and a memory is never stored without its chunks. A vector search needs the query embedded. Everything else reads rows or stored vectors.
   - *Evidence:* `s3_hello_says_loading_and_writes_wait_while_reads_are_served`.
4. **Bulk session grace: 10 seconds (SPEC 5, query and bulk sessions; B-27).**
   - *What:* the bulk session stays open for `bulk_grace` (10 s) after its last batch, then is dropped. Nothing new is journalled.
   - *Why:* a burst of writes (a capture burst, an import) reuses one session instead of opening one per `remember`; memory returns to the idle level 10 s after the last batch.
   - *Evidence:* AT-3 opened one bulk session for the whole 2,000-chunk import (section 5). The real cost of opening a session is not measured (section 9).
5. **The log is bounded by rotation (stream E build 3).**
   - *What:* `daemon.log` rotates to `daemon.log.1` at 1 MiB (`log_max_bytes`), so at most about 2 MiB is kept. Each message is cut at 2,000 characters and written in one call.
   - *Evidence:* `s3_the_daemon_log_stays_bounded`.
6. **Bounded caches (SPEC 9; SPEC 16; B-32).**
   - *What:* three LRU caches of 256 entries each:
     - the stored project per folder (the identity itself is resolved on every use, decision 30);
     - capture sessions, holding each one's state and transcript offset (`capture_sessions`);
     - recall ranks for `reinforce` (`rank_sessions`).
   - *Why:* a daemon lives for days. A forgotten capture session starts again at offset 0 with fresh hashes, and the store's dedupe by content hash (B-29) stops a text being stored twice.
   - *Evidence:* `s8_remember_creates_then_finds_the_duplicate`, `s16_transcript_offsets_per_session`.
7. **The capture queue holds 1,024 jobs (SPEC 16, never loses; B-33).**
   - *What:* a capture stays in `spool/new/`, or is written there, for a later pass or a later daemon, in three cases:
     - the queue is full;
     - the model has failed;
     - the daemon stops with jobs still queued.
   - *What goes to `spool/failed/`:*
     - a file the daemon cannot parse, a request for another profile, or a capture it does not accept (B-33);
     - a capture that can never be stored (`invalid_input`, `secret_detected`, `project_unresolved`);
     - a capture that failed 5 times (decision 27).
   - **Corrected.** The first report said every processing failure went to `spool/failed/`, was logged and was never retried. That lost a capture for good on a storage error, and B-33 does not cover it.
   - *Evidence:* `at15_spool_is_drained`, `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once`, `s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound`. The 1,024-job spill is covered by code, not by its own test (section 9).
8. **A capture with no session keeps no state (SPEC 16).**
   - *What:* when neither the request nor the payload has a `session_id`, the transcript is read from offset 0 with fresh hashes. Version 1 did the same (`load_empty_session_id`, `save_empty_session_id_noop`).
   - *Why:* two unrelated sessionless transcripts never share an offset.
   - *Evidence:* `s16_transcript_offsets_per_session`.
9. **`capture_size_tokens` (stream E build 8; B-65).**
   - *What:* the value is the chunker's token count of the whole content: `ChunkPlan.total_tokens`, the same tokenizer as `count_tokens`. It is labelled with `project`, `tool` and `indexed_tokens`, and `capture_cap_hit_total` is recorded when `ChunkPlan.capped`. The 200,000-character extraction cap (`provenance.extra.truncated`) is not counted.
   - *Evidence:* `at17_capture_over_the_index_cap_is_stored_whole`.
10. **The spool drain is skipped while the model is `failed` (B-33, B-36).**
    - *What and why:* those captures cannot be chunked, so the files stay for a later daemon instead of being read and spooled again every minute.
    - *Evidence:* code only (`maintenance.rs`).
11. **The map cache key covers vectors (SPEC 7; B-46, B-47).**
    - *What:*
      - The key is the SHA-256 of `sm_spatial::input_hash`, the project, and each memory's id, `updated_at` and vector bytes.
      - Runs live in `map_runs`, so they survive a restart; a per-project index in memory skips reading vectors on a hit.
      - A run without coordinates is stored under the suffix `:regions`, with every point at (0, 0), which is never returned. A run with them is stored under `:coords`. A plain request is served by either.
      - Labels are copied between the two variants of the same data.
    - *Why:* `input_hash` covers ids, not vectors (c-report decision 6), and any edit, forget or re-embed must drop the run.
    - **Corrected.** The first report did not cover a re-embed while `regions` computes. A batch stored, with the maps dropped, during the computation left the run built from older vectors in the index until the next write. Decision 34 closes this.
    - *Evidence:* `s7_regions_cached_until_an_edit_and_coordinates_only_when_asked`, `s7_each_run_variant_knows_its_sibling`, `s7_a_run_computed_before_a_drop_is_never_indexed`. The (0, 0) points are a contract change request (section 8).
12. **Sharing calls wake the queue and drop the maps after every call, successful or not (B-79).**
    - *Why:* a failed import may already have stored part of a file.
    - *Evidence:* `s9_sharing_ops_reach_the_implementation_with_the_resolved_project`, `s9_import_drops_the_maps_and_its_memory_is_embedded`.
13. **`UnwiredSharing` (B-79).**
    - *What:* `start` and `run` take `Arc<dyn SharingOps>` with no default, so a caller must choose an implementation. `UnwiredSharing` is a public one that fails every call with `internal`, with the next action "Start the daemon with `spatial-memory daemon`, which wires in sharing". It never pretends to succeed.
    - *Why:* the `daemon` subcommand can start before phase 3 wires `sm-share` in, and tests that do not use sharing need no fake. The lead may delete it once the real implementation exists.
    - *Evidence:* `s9_import_export_and_migrate_without_the_implementation_say_so`.
14. **Unknown keys refused in dispatch (B-28).**
    - *What:* checked against their keys before decoding: the request, the `call` wrapper, and the arguments of `shutdown`, `ambient_recall`, `capture`, `import`, `export`, `migrate`, `reindex`, `project_rename`, `project_delete` and `doctor_eval`.
    - *Why:* those structs do not deny unknown fields (section 8).
    - *Evidence:* `s3_unknown_fields_are_refused_for_every_op`.
15. **Link staleness (SPEC 4, links; B-37).**
    - *What:*
      - Links made through the tool are `manual` and carry no commit.
      - `list` uses `head_only(head)`. A `verified_commit` equal to the head, or a prefix of it, is current. Anything else is undecided rather than stale, because the daemon has no ancestry walk.
      - When the head cannot be read from the repository's files, `stale` is absent.
    - *Evidence:* `s8_link_add_list_remove_and_its_errors`.
16. **`forget`'s confirm rule is literal (B-30).**
    - *What:* "more than one memory" needs `confirm: true`; one memory does not.
    - *Evidence:* `s8_forget_needs_confirm_for_more_than_one_memory`.
17. **`doctor_eval` uses a scratch store for a set with its own memories (SPEC 11).**
    - *What:* the scratch store is built in the profile folder from the set's memories and removed afterwards. Its chunks are embedded on the queue's bulk session (decision 35). A set without memories runs against the live store: over the current project if `cwd` resolves, over every project otherwise. Each query takes recall's vector path.
    - *Evidence:* `s11_doctor_eval_runs_the_retrieval_set_and_its_error`, `at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget`.
18. **Ambient search asks for `top_k` plus the session's injected memories (SPEC 6, ambient recall).**
    - *What:* memories already shown are replaced by the next best, instead of turning a fresh result into `repeat`.
    - *Evidence:* `at17_ambient_block_is_the_reply_text_and_never_repeats`.
19. **The ambient switches are the hook's (SPEC 6; B-34).**
    - *What:* the daemon never answers `disabled` and uses the `top_k` it is sent. It does not journal `hook_gate_total`, `hook_top_score` or `hook_latency_ms`.
    - *Evidence:* `s6_ambient_gates`.
20. **The daemon checks the four-word minimum again (SPEC 6).**
    - *What:* a direct caller gets `short_prompt`, as the hook would. Words are counted as Python's `str.split()` counts them (`ambient::prompt_word_count`), so a prompt of only control whitespace passes Rust's `trim` and still gets `short_prompt`. SPEC 6's minimum supersedes version 2's behaviour.
    - *Evidence:* `s6_ambient_prompt_cases`.
21. **Injections are recorded after the reply is written, and only before the deadline (SPEC 6, step 6; B-37).**
    - *What:* a reply built after the deadline is `timeout` and records nothing.
    - *Why:* B-37 accepts that the daemon records an injection when it builds the block. Recording after the write narrows that race, because a reply that is never written records nothing. B-37's gap remains when the hook's own deadline passes after the reply is written.
    - *Deadlines* are checked after embedding, after each search, and inside the long loops (`regions`, `consolidate`, `doctor_eval`).
    - **Changed (review item 6a):** a `record_injections` failure is now logged. The line names the session, the number of memories and the cause, because those memories may be injected again in the session. Before, the error was dropped.
    - *Evidence:* `s6_ambient_deadline_records_nothing`, `s3_a_passed_deadline_gives_timeout`. The log line has no test: the store gives no way to make that call fail from outside.
22. **Idle exit (CONTRACTS 4).**
    - *What:* with `daemon.idle_exit_minutes` above 0, the daemon stops after that long with no request, nothing in flight, no capture queued and no bulk session open.
    - *Evidence:* `s3_the_daemon_exits_when_idle`, `s3_idle_exit_waits_while_requests_keep_coming`.
23. **Shutdown grace: 5 seconds (CONTRACTS 3.1).**
    - *What:* the workers and the requests in flight get `shutdown_grace` (5 s) after the endpoint closes. The full order is in decision 29.
    - **Corrected.** The first report said the lock was released last. It was released after the other steps, but:
      - the capture being processed was neither stored nor spooled;
      - connection threads could still be running;
      - `DaemonHandle::drop` released the lock straight away.

      Now the lock is released only after the threads have stopped, or after the grace with every held capture on disk. `drop` follows the same order.
    - *Evidence:* `s3_shutdown_releases_the_lock_and_a_new_daemon_starts`, `at15_a_capture_held_at_shutdown_is_stored_after_a_restart`.
24. **Output budgets (SPEC 8, output; B-28).**
    - **Rewritten (review item 6b).** The first rule always kept the first item. It cut that item's content to no fewer than 16 characters and left every other field whole. `related` and the `regions` points then fitted their second list against the whole budget again. So a reply could exceed its limit.
    - *Estimate:*
      - Tokens are estimated as for the hook block, four characters a token (`ambient::estimate_tokens`), over JSON.
      - The reply's own fields are measured on its skeleton: the reply with every budgeted list empty, `truncated` set and `omitted` at `u32::MAX`. Eight tokens of slack are added for the keys of lists an empty list leaves out.
      - Each item costs its JSON plus one token for its comma, measured after any cut.
      - Since ceil(a + b) ≤ ceil(a) + ceil(b), the whole reply never exceeds the sum of its parts.
    - *Fitting:*
      - Items are kept in order while they fit. The first item that does not fit whole is cut, and kept only if it then fits; any later item that does not fit ends the list.
      - A memory is cut by its content first. When fewer than 16 characters of content would remain, it sheds fields in this order: (1) provenance `extra`, tags and anchors; (2) its title; (3) its provenance beyond `source`, and its `supersedes_id`. Then it keeps whatever content still fits, possibly none, with `content_truncated` set.
      - Never cut: the id, project, type, status, score and provenance source, which SPEC 8 asks every result to carry.
      - Nested items:
        - a region cuts its list of central memories;
        - a duplicate group gives each of its views an equal share of its room;
        - a linked memory, a journey step and a duplicate each cut the memory view inside them.
    - *Two lists:*
      - `related` fits `by_vector` first. When the vector list does not fit whole, `by_links` gets none and is counted in `omitted`.
      - `regions` fits the regions first, then the points, whole, in the room left.
    - *Refusal:* when the skeleton alone is over the limit, the request gets `invalid_input`: "limit {n} is too small for {tool}: its fixed fields take about {used} tokens". The next action is "Give a limit of at least {used}, or none." Above the minimum limit of 100, only `status` (with its journal summary) can reach this.
    - *Found while fixing:* at the minimum limit of 100, no memory view fits. A view's fixed fields take about 85 to 90 tokens and recall's skeleton 26, so `recall` at 100 returns no result, with every hit counted in `omitted` and `truncated` set (section 8, request 5).
    - *Evidence:*
      - `s8_no_reply_exceeds_its_limit_in_the_worst_case`. Its four memories have every field at its longest:
        - about 2,700 characters of content, which JSON escapes to more than its length;
        - a 1,500-character title;
        - 30 tags and 30 anchors;
        - a long author and file path.

        The test runs `recall`, `related`, `regions`, `journey`, `consolidate`, `link` list and `status` at limits 100, 120, 150, 250, 400, 1,000 and 4,000.
      - `s8_a_reply_fits_its_limit_and_counts_what_it_left_out` (changed, section 3).
      - The three budget unit tests.
25. **`consolidate` compares every pair (B-31).**
    - *What:* proposals compare the mean vectors of every pair of active, embedded memories in the project. That is O(n²), about 2 million dot products at 2,000 memories.
    - Not timed here.
26. **Test conventions.**
    - Each test project folder holds its own `.git`, so project resolution never climbs into whatever encloses the temporary folder.
    - `physical_cores` is fixed at 8 in the tests.
    - The ambient fixture integration test lowers the similarity floor to 0.30 for that test only, because the fake embedder's cosine for the fixture's texts is about 0.43. The default is unchanged.
    - AT-3's bulk import is 20 `remember` batches of 100 one-chunk memories, embedded by a recording provider that takes 3 ms per document, so the import runs during all 100 calls.
    - AT-4 reads vectors through a second `Store` handle configured with the daemon's stamp, because the store refuses vector reads before `configure_index`.
    - The faults of the capture tests come from `FaultChunker`, the chunker the test loader hands the daemon. By a marker in the content, it can:
      - fail once with a `storage` error;
      - fail every time;
      - panic;
      - block until the test opens it.

      `remember`'s store path calls the chunker inside the store step, so the error arrives where a storage error would. No crate-private hook was needed.
27. **Capture retries (SPEC 16, never loses; B-33; review item 1).**
    - *What:*
      - `sm_capture::capture` runs on a copy of the session's `SessionState` and transcript offset. The copy replaces the session only after the store succeeds. Until then, the session's offset and hashes stay where its last stored capture left them.
      - A failure that may pass later is tried again: every error code but `invalid_input`, `secret_detected` and `project_unresolved`, and a panic, which is caught and treated as `internal`.
      - The wait is `capture_retry_backoff` (2 s), doubled after each failure: 2, 4, 8 and 16 s. The worker takes other jobs meanwhile.
      - After `MAX_ATTEMPTS` (5) failures in one daemon, or at once for the three codes, the file moves to `spool/failed/`. The log line names the request id, the code and message, and the reason ("failed 5 times" or "it can never be stored").
      - Each failed attempt is logged too ("attempt n of 5 failed, tried again in …").
    - *Why:*
      - B-33 keeps `spool/failed/` for files the daemon cannot parse. A storage error, such as the database busy past its timeout or a full disk, usually passes.
      - Five attempts over about 30 s ride out a long write or a brief full disk, and the bound stops one bad job from looping.
      - The count is per daemon: a file still in `spool/new/` when the daemon stops starts again at 0 in the next one, which is what a restart after freeing disk space should do.
    - *Evidence:*
      - `s16_a_failed_capture_is_retried_and_its_session_moves_on_only_once_stored`: a storage error on the first attempt, then success. The capture is stored once, and the session's next capture reads only the text appended after it.
      - `s16_a_capture_that_keeps_failing_moves_to_failed_after_the_bound`: a store that always fails and a chunker that panics each reach `spool/failed/` after 5 attempts, with the log line, and the worker keeps running.
      - `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once`.
      - `at15_spool_is_drained`: a file that cannot be parsed still goes to `spool/failed/`.
28. **A capture is on disk until it is stored (SPEC 16, never loses; AT-15; CONTRACTS 7; review item 2).**
    - *What:*
      - A capture is a file in `spool/new/` from the moment the worker takes it until it is stored. A drained file stays where it is, and a socket capture is written there before it is processed.
      - The file is removed only after the store succeeds.
      - `Shared::held` names the files that jobs hold, and the drain skips those names, so one daemon never queues a file twice. The frozen layout (`tmp`, `new`, `failed`) is unchanged.
    - *At least once:*
      - After a crash, or when the next daemon starts while an old worker is still busy, the file is processed again.
      - The store's duplicate rule (B-29, `sm-store`'s `s4_duplicates_return_the_existing_memory`) returns the memory already stored for the same content in the same project. Checked: `at15_a_capture_held_at_shutdown_is_stored_after_a_restart` lets both daemons' workers finish the same capture, and one memory is stored.
      - A capture processed twice journals its capture metrics twice: that test sees two `capture_size_tokens` rows for the request.
    - *Contract change:* CONTRACTS 7's "removes each file after its capture is queued" becomes "removes each file after its capture is stored". The coordinator accepted it and edits CONTRACTS at the merge.
    - *Evidence:* `at15_a_drained_file_stays_in_new_until_its_capture_is_stored`, `at15_a_capture_held_at_shutdown_is_stored_after_a_restart`, `at15_spool_is_drained`.
29. **The shutdown order (CONTRACTS 3.1; review item 2).**
    - *What:* `finish`, which both `wait` and `DaemonHandle::drop` run, goes in this order:
      1. The `shutting_down` reply and the stop flag.
      2. `closed` is set, so a request still arriving on an open connection gets `daemon_unavailable`.
      3. Every worker is woken, and the accept loop is woken by a connection to itself; it stops, which closes the endpoint.
      4. The worker threads are joined, and requests in flight are waited for, until `shutdown_grace` (5 s) has passed.
      5. Every capture still queued is written to `spool/new/`.
      6. "stopped" is logged.
      7. The lock is released, last.
    - *After the grace:* a thread still running holds nothing that is not on disk. A capture request in flight sees the stop flag and spools itself, and the worker's job is a file in `spool/new/` that the next daemon processes (decision 28). A connection thread blocked on a read holds no request.
    - *Evidence:* `at15_a_capture_held_at_shutdown_is_stored_after_a_restart` (the worker is held in the chunker past a 300 ms grace), `s3_shutdown_releases_the_lock_and_a_new_daemon_starts`, `s3_shutdown_over_the_socket_stops_the_daemon_with_its_reason`.
30. **The project is resolved on every use (SPEC 9; SPEC 16; review item 3).**
    - *What:* `resolve_project` runs for every request and every capture. It reads a few files and never runs git: the folder walk to `.git`, `project.toml` and the git configuration. The cache, keyed by the working folder, keeps the stored project for an identity, so the database is written only when the identity changes. So these count from the next request, with no restart:
      - a `capture.enabled = false` added to `project.toml`;
      - a `git init` in a folder that was not a repository.
    - *Why:* the cache was keyed by the raw folder and dropped only on a rename or delete. With `idle_exit_minutes = 0` the daemon never restarts.
    - *Cost:* not timed on its own. AT-3's `ambient_recall` latencies (section 5) include it.
    - *Evidence:* `s16_capture_disabled_in_project_toml_counts_without_a_restart`, `s9_a_folder_that_becomes_a_repository_is_one_without_a_restart`.
31. **The Unix bind where `fchmod` before `bind` is unsupported (SPEC 3 rule 1; review item 4).**
    - *What:*
      - interprocess 2.4.8 sets the mode with `fchmod` on the socket before `bind`. It documents this for Linux, FreeBSD 14.3 and later, and OpenBSD, and maps `EINVAL` to `Unsupported`; macOS is expected to give `Unsupported`.
      - On `Unsupported`, the daemon creates a folder next to the socket with mode 0700 (`.<pid in hex><n>`), binds `s` inside it with no mode, and sets the socket to 0600. It then renames the socket into place and removes the folder.
      - Connecting needs search permission on every folder of the path, so no other user can connect before the mode is set.
      - The folder's path is no longer than the final one, so it stays within the `sun_path` limit.
    - **Unverified on macOS:** there is no macOS machine. On Linux, interprocess's own path works, so the fallback runs only in the unit test that calls it directly.
    - *Evidence:* `s3_a_socket_bound_in_a_private_folder_is_user_only_and_moved_into_place` (Linux): mode 0600, the socket in place, no folder left over, and the socket removed with the endpoint.
32. **The socket's owner is checked (SPEC 3 rule 1; review item 5).**
    - *What:* on Unix the socket must belong to the effective user (`geteuid`, in the endpoint's `security` module), checked with `lstat`, which does not follow a link.
      - `connect` refuses a socket another user owns. It checks before connecting, and again once connected, before anything is sent, because an entry missing at the first check may have been created since. The error is `daemon_unavailable`, with a next action: set `XDG_RUNTIME_DIR` to a folder only you can enter, or use a shorter profile path.
      - `bind` refuses to replace an entry another user owns, with a next action that also names the file for an administrator to remove. A leftover of this user is still replaced.
      - A missing entry passes.
    - *Why:* when the profile path is too long for `sun_path` and there is no absolute `XDG_RUNTIME_DIR`, `sm_core::paths::unix_socket_path` gives `/tmp/spatial-memory-<hash>.sock`. That name is predictable and the folder is shared. `/tmp` is sticky, so only its owner can remove an entry: once an entry passes the check, nobody else can swap it.
    - *What it does not cover:*
      - Another user who binds the name first still stops the daemon from starting, now with a clear error. A per-user folder in `sm-core` would remove that (section 8, request 4).
      - The second check in `connect` cannot be staged in a test; every Unix test that connects runs through it.
    - *Evidence:* `s3_a_socket_another_user_owns_is_refused_by_connect_and_bind`, `s3_the_owner_check_does_not_follow_a_link` (Linux). Another user is stood in for by passing a different uid to the private `connect_as` and `bind_as`.
33. **A chunk that always fails is skipped (SPEC 5, Bulk; review item 6e).**
    - *What:*
      - The queue embeds a batch of pending chunks in id order.
      - A storage failure (the database busy past its timeout, a full disk) or a refused index is retried after `bulk_error_backoff` (5 s).
      - Any other failure of a batch, a panic in the embedder included, is retried one chunk at a time, so the chunks that embed alone are stored. The panic is caught, and the thread keeps running.
      - When no chunk embeds alone, the session is dropped (the next batch opens a fresh one) and the queue waits `bulk_error_backoff`.
    - *A skipped chunk:*
      - A chunk that fails alone `SKIP_AFTER` (3) times is skipped, with a log line naming it.
      - It stays pending in the store: nothing is lost, and `status` counts it. This daemon no longer offers it to the embedder, and a later daemon tries it again.
      - A revise or a reindex gives the memory new chunks, which are not skipped.
    - *Evidence:* `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on`. Two chunks fail every time, one by an error and one by a panic. The three good chunks are stored, the two bad ones stay pending, the skip is logged, and a later `remember` is still embedded.
34. **A map run computed before a drop is never indexed (SPEC 7; B-46; review item 6c).**
    - *What:* every drop of the maps moves a generation counter on. `regions` and `journey` read the generation before they read the vectors, and index their run only under that generation.
    - *Why:* a run built from vectors that a batch replaced meanwhile is stored in `map_runs` under its own key, which covers those vectors, so it is never served for the new vectors. But it is never put in the in-memory index either.
    - *Evidence:* `s7_a_run_computed_before_a_drop_is_never_indexed`.
35. **One bulk session per daemon (SPEC 3 rule 5; AT-3; review item 6d).**
    - *What:* the daemon has one bulk session, in `Shared::bulk`. It is opened on first use with `Settings::bulk_threads` threads and dropped once idle for `bulk_grace`. The queue and `doctor_eval` both embed through `with_bulk`, one batch at a time, so the bulk threads alive never exceed the budget. A `doctor_eval` during an import takes turns with the queue.
    - *Evidence:* `at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget`: at most 6 bulk threads alive with a budget of 6, and the sessions opened `[(Query, 4), (Bulk, 6)]`.
36. **File paths must be absolute (CONTRACTS 5, Requests; review item 6f).**
    - *What:* these fields of `import`, `export`, `migrate` and `doctor_eval` must be absolute: `path`, `from`, `set_path` and `baseline_path`. A relative path is refused in dispatch, before the op runs, with `invalid_input`: "`<field>` needs an absolute path, not `<path>`". The next action is "Resolve the path against your own working folder before sending `<op>`; the daemon's working folder is not yours."
    - *Why:* a relative path would resolve against the daemon's working folder, not the caller's.
    - *Evidence:* `s9_a_relative_file_path_is_refused_before_the_op_runs`.
37. **Connections have no read timeout and no cap (CONTRACTS 3.1; review item 6h; recorded, not changed).**
    - *What:* a connection thread blocks on its next frame for as long as the client keeps the connection open, and the number of connections is not limited.
    - *Why:* only the same user can connect (the pipe's DACL on Windows, mode 0600 and the owner check on Unix). The MCP process holds one connection for its life, and each hook opens one and closes it.
    - *Risk:* a buggy client of the same user could hold many threads.

## 7. Deviations

- **AT-10.** SPEC 11 says a changed byte "stops the daemon from starting, with a clear error". Following B-36, the daemon instead starts, binds, and stays up in `failed` with `model_integrity` as `state_error`.
  - `hello` and `status` give the clear error.
  - Every model request fails with it; the test checks vector `recall`, `remember` and `reindex`.
  - The rest is served; the test checks keyword `recall`, `forget` and `status`.
  - *Evidence:* `at10_one_changed_byte_fails_the_model_and_the_rest_is_served`.
- **AT-4, peak memory.** SPEC 11 also asks for peak memory under 2 GB. With the fake embedder the figure means nothing, so it is not measured; it is phase 3 with the real model.

## 8. Contract changes requested

1. **`deny_unknown_fields` on the remaining protocol structs (B-28).**
   - *The gap:* `Request` and the parameters of `shutdown`, `ambient_recall`, `capture`, `import`, `export`, `migrate`, `reindex`, `project_rename`, `project_delete` and `doctor_eval` in `sm_core::protocol` accept unknown keys. The tool parameters deny them.
   - *Impact:* the daemon refuses unknown keys itself (decision 14), so nothing is broken. With the attribute on those structs, dispatch's key lists could go.
   - *Evidence:* `crates/sm-core/src/protocol.rs` (`Request` and `ShutdownParams`, and the others after them); `s3_unknown_fields_are_refused_for_every_op`.
2. **Optional map positions in `sm-store` (SPEC 7; B-47).**
   - *The gap:* `sm_store::MapPointRow` needs finite `x` and `y` (`crates/sm-store/src/maps.rs`), but a run without coordinates has none. So the daemon stores (0, 0) and records `"coords": false` in the run's parameters (decision 11).
   - *The change:* nullable positions would let a run say what it holds.
3. **CONTRACTS 7, the spool drain (decision 28).** "Removes each file after its capture is queued" becomes "removes each file after its capture is stored". The coordinator accepted this and edits CONTRACTS at the merge.
4. **The socket owner check outside this crate (decision 32; review item 5).** Both are the lead's in phase 3.
   - **`sm-cli`, the hook client.** It must either connect through `sm_daemon::connect`, or make the same check: `lstat` the socket and refuse it unless its owner is the effective user, before connecting and again after, before sending anything. Otherwise a hook sends prompts and memory to whoever bound the `/tmp` name, and takes forged ambient text back.
   - **`sm-core`, `paths::unix_socket_path`.** The `/tmp` fallback could name a per-user folder such as `/tmp/spatial-memory-<uid>/`, with the socket inside. The daemon would create that folder with mode 0700 and check its owner. That would also stop another user from blocking the daemon's start by binding the name first, which the daemon can now only refuse.
5. **`MIN_OUTPUT_TOKEN_BUDGET` is too small for one result (SPEC 8; decision 24).**
   - *The gap:* its documentation says the minimum limit of 100 leaves "room for one result line and its provenance". As JSON, a `MemoryView`'s fixed fields alone take about 85 to 90 tokens: id, project, type, status, origin, score, rank, `matched_by`, importance, provenance, `created_at` and `updated_at`. Recall's skeleton takes 26 more, so `recall` at 100 returns no result.
   - *The change:* either raise the minimum (about 120 holds one cut memory with a short project name), or slim the view, which needs `sm-core`.

## 9. Unfinished or unverified

- **The real `SharingOps`.** `share`, `sync`, `import`, `export` and `migrate` run only through the test fake and `UnwiredSharing`. The lead writes the implementation over `sm-share` in phase 3.
- **The real model in the daemon.** `GemmaLoader` is built, but no test here loads it, because the one real-model test binary lives in `sm-embed`. Phase 3 has:
  - the real-model AT-3 figure;
  - the real `ThreadReport` of the query and bulk sessions;
  - the cost of opening a bulk session (decision 4).
- **Peak and idle memory** (AT-4, SPEC 11 budgets) are not measured: phase 3.
- **The hook side** of these tests belongs to stream D:
  - AT-3: with the daemon killed, exit 0 with empty output inside 1,500 ms;
  - AT-14: the printed output;
  - AT-15: exit within 200 ms with the daemon stopped;
  - AT-17: the off switch.
- **macOS.** The private-folder bind (decision 31) and the owner check (decision 32) are tested on Linux only; nothing ran on macOS.
- **No test of their own:**
  - decision 10 (the drain skipped while `failed`) and the 1,024-job spill of decision 7, covered by code only;
  - the `record_injections` log line (decision 21);
  - review item 6g, rank numbering, a race that holds by the order of the code;
  - the second owner check in `connect` (decision 32).

## 10. Licences

Packages added to `Cargo.lock` by this stream since `9387a08`:

| Crate | Version | Licence |
|---|---|---|
| `interprocess` | 2.4.8 | 0BSD OR Apache-2.0 |
| `doctest-file` | 1.1.1 | 0BSD |
| `recvmsg` | 1.0.0 | 0BSD |
| `widestring` | 1.2.1 | MIT OR Apache-2.0 |
| `sm-daemon` | 3.0.0 | MIT |

The second delivery adds no package. The merge of `v3` brought in the other streams' lock entries, and no version was changed here.

**Flag:** `doctest-file` and `recvmsg` are 0BSD only; both come in through `interprocess`. 0BSD (Zero-Clause BSD) is a BSD-family licence with no conditions, equivalent to public domain, but it is not written on the permissive list. `LICENSES.md` already records `adler2` with 0BSD as one option. The lead decides whether to add 0BSD to the list.

The crate's other dependencies were already in the lock: `windows-sys` 0.52.0 and 0.61.2, `libc` 0.2.191, `serde`, `serde_json` and `tempfile`.

**SPEC 13 risk, retired for Windows and Linux only.** SPEC 13 lists `interprocess` as "Chosen from documentation, not tested", with loopback TCP and a per-user token file as the fallback.
- **Tested:** the named pipe on Windows and the Unix socket on Linux. The fallback was not needed there.
- **Corrected.** The first report called the risk retired without saying so for macOS, which SPEC 3 rule 1 names. On macOS, the socket mode is expected to take the untested path of decision 31, so the risk stays open there until a macOS run.
