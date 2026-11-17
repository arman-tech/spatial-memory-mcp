# Stream E report: `sm-daemon`

Branch `arman-tech/stream-e-daemon`, from `v3` at `9387a08`. Commits: `903e05b` (the crate and `Cargo.lock`), `d08aac4` (socket tests), `1d3ed55` (hook tests and the sessionless-capture fix), `1be4769` (acceptance tests), and a last commit with this report and two test changes (section 3). The layout follows `RULES.md` section 9.

Abbreviations, written out once: AT-n is acceptance test n in SPEC section 11. API is application programming interface. JSON is JavaScript Object Notation. LRU is least recently used. DACL is discretionary access control list, SDDL is security descriptor definition language, and SID is security identifier. ONNX is Open Neural Network Exchange. SHA-256 is the 256-bit Secure Hash Algorithm 2. p95 is the 95th percentile. TCP is the Transmission Control Protocol.

## 1. Done when

**Yes.** Every tool and every hook event works end to end over the real socket: each of the 24 operations of CONTRACTS 5 is tested for its success and, where it has one, its main error; `capture` stores what each applicable hook payload should, and `ambient_recall` runs every applicable fixture case. The crate has 73 tests. 71 run on each platform, because two are Unix-only and two Windows-only, and all pass on Windows and on Linux (Docker).

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-daemon/Cargo.toml` | The new library crate. It depends on `sm-core`, `sm-store`, `sm-embed`, `sm-spatial`, `sm-capture`, `interprocess`, `serde` and `serde_json`; `windows-sys` on Windows and `libc` on Unix; `sm-core` with `test-support` and `tempfile` for development only. Every version comes from the root pins; no feature was added. It does not depend on `sm-share` (B-79). |
| `src/lib.rs` | Crate documentation (why threads, no process-global state), `#![deny(unsafe_code)]`, and the public API below. |
| `src/server.rs` | `start` and `run`: the lock, the store and its index stamp, the endpoint, then the loader thread; the accept loop with one thread per connection; the frame loop; the worker threads; `embedder_cold_start_seconds`; the shutdown order. |
| `src/endpoint.rs` | `bind` with retry, `connect` and `accept`. Unix: the leftover socket file removed, mode 0600. Windows: the first pipe instance, remote clients rejected, and the user-only descriptor (`#[allow(unsafe_code)]`, the crate's only unsafe code), plus `current_user_sid` and `pipe_dacl_sddl` to read the DACL back. |
| `src/lock.rs` | `DaemonLock`: `daemon.lock` opened without truncation and never written, `try_lock` every 25 ms for up to 2 s, held for the daemon's life, released last. |
| `src/loader.rs` | The `ModelLoader` trait (`info` before `load`), `LoadedModel` (provider and chunker), and `GemmaLoader`: `verify` on the model directory, then `GemmaProvider` and its tokenizer-based chunker. |
| `src/dispatch.rs` | One request: `Envelope::peek`, `version_mismatch`, unknown keys, the profile check, `Op::validate`, the model-state check, the deadline, the op table, and the work done after the reply is written. |
| `src/state.rs` | `Shared`, one per daemon: the model state (`Loading`, `Ready`, `Failed`), the caches, the queues, the wake signals and an LRU map. |
| `src/options.rs` | `DaemonOptions`: the endpoint and every timing and size, injectable for tests, with the defaults in section 6. |
| `src/log.rs` | `DaemonLog`: one write call per line, messages cut at 2,000 characters, rotation to `daemon.log.1` at 1 MiB. |
| `src/project.rs` | `resolve_project` on each request's `cwd`, cached per folder (LRU, 256), then upserted. |
| `src/queue.rs` | The bulk embedding worker: batches of `embed.batch_size`, one `set_embeddings` per batch, the bulk session opened on demand and closed after the grace period, the maps dropped per batch, a back-off after a failed batch. |
| `src/capture.rs` | `capture`: the reply once queued, one worker running `sm_capture::capture`, sessions and transcript offsets, storing through `remember`'s path, the capture metrics, spooling when it cannot process. |
| `src/ambient.rs` | `ambient_recall`: the four-word minimum, the project, the model state, the prompt cut, embedding, search, `ambient::apply_gates`, and the injections recorded after the reply. |
| `src/maintenance.rs` | The maintenance thread: the spool drain at start and every 60 s, the daily pruning of injections and journal files, and the idle exit. |
| `src/admin.rs` | The command-line and sharing ops: `share`, `sync`, `import`, `export`, `migrate` through `SharingOps`; `reindex`; `project_rename`; `project_delete`; `doctor_eval`. |
| `src/sharing.rs` | The `SharingOps` trait, `SharingContext`, `ResolvedProject`, and `UnwiredSharing`. |
| `src/budget.rs` | Output budgets: each reply fits `limit` or `output.token_budget`, with `truncated` and `omitted`. |
| `src/view.rs` | `MemoryView` construction, with project names. |
| `src/tools/mod.rs` | The tool modules. |
| `src/tools/remember.rs` | `remember`, and the store path every new memory takes: redact, title, chunk, insert with pending chunks, wake the queue, journal `chunks_per_memory`. |
| `src/tools/recall.rs` | `recall` (vector, keyword, auto with appended keyword hits), weak marking, the recall metrics, the per-session ranks, and `reinforce`. |
| `src/tools/related.rs` | `related`: vector neighbours and the two-hop walk over links and anchors. |
| `src/tools/edit.rs` | `revise` (in place or `supersede_with`) and `forget` (by id or filter, with the confirm rule). |
| `src/tools/link.rs` | `link`: add, remove and list, with staleness from the head commit. |
| `src/tools/maps.rs` | `regions` and `journey` through `sm-spatial`, the map cache, coordinates when asked, labels. |
| `src/tools/consolidate.rs` | `consolidate`: propose near-duplicate groups, or apply exactly the listed groups. |
| `src/tools/status.rs` | `status`: state and `state_error`, model, index stamp, pending embeddings, projects, spool, and the journal summary when asked. |
| `tests/common/mod.rs` | The test rig: a temporary profile, the endpoint passed in, a client over `interprocess`, test loaders (fake, gated, failing, closure), a recording provider, a recording `SharingOps`, and helpers. |
| `tests/socket.rs` | 43 tests: the lock, the endpoint, frames and envelopes, every op with its success and main error. 41 run on each platform (2 are Unix-only, 2 Windows-only). |
| `tests/hooks.rs` | 14 tests: every hook event, daemon side, with the capture and ambient fixtures (AT-13 to AT-17). |
| `tests/acceptance.rs` | 12 tests: AT-3 (daemon side), AT-4, AT-5, AT-9 (`remember` side), AT-10, AT-11, AT-12, the SPEC 6 recall rules, the SPEC 7 map cache, and B-79's dropped maps. |
| `Cargo.lock` | 5 packages added (section 10). No existing version changed. |
| `docs/v3/streams/e-report.md` | This report. |

### Public API

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
- `start(...) -> Result<Started>` is the same, without blocking: `Started::LockHeld` or `Started::Running(DaemonHandle)`. The handle has `endpoint`, `log_path`, `stopping`, `shutdown(reason)` and `wait`.
- `ModelLoader` has `info() -> Result<ModelInfo>`, read before loading so the index stamp is configured at start, and `load() -> Result<LoadedModel>`. `GemmaLoader::new(model_dir, &settings)` is the real one.
- `SharingOps: Send + Sync` has `share`, `sync`, `migrate`, `import` and `export`, each taking a `SharingContext` (the `Store`, the chunker if loaded, the secret scanner, `remember`'s `ChunkParams`, and the resolved project with its identity) and the op's parameters, and returning the op's reply type from `sm_core::protocol`.

## 3. Test commands and exact results

**Windows.** Windows 11 Pro 10.0.26300, x86_64, AMD Ryzen 9 9950X3D (16 cores, 32 logical), 93.6 GB, rustc 1.95.0, PowerShell, `test` profile (debug).

- `cargo fmt --check` exited 0.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` exited 0.
- `cargo test --workspace --all-features -- --nocapture` exited 0. Every `test result:` line, in order:

```
sm_capture (unit)        test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
sm_core (unit)           test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_daemon (unit)         test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_daemon acceptance     test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.62s
sm_daemon hooks          test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.76s
sm_daemon socket         test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.56s
sm_embed (unit)          test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.33s
sm_embed real_model      test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.68s
sm_spatial (unit)        test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.64s
sm_spatial at6           test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.17s
sm_spatial at7           test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.53s
sm_spatial s7_pipeline   test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.20s
sm_store (unit)          test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.81s
doc-tests (6 crates)     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s  (each)
```

The crate names on the left are added here; each line after them is copied from the log. The `real_model` binary belongs to `sm-embed` and ran because the model is in the per-user cache; no `sm-daemon` test loads the model.

**Linux (Docker).** `rust:1.95-trixie` on the same machine, the source mounted read-only, as in `CLAUDE.md`:

```
cargo test -p sm-daemon --all-features --locked
cargo clippy -p sm-daemon --all-targets --all-features --locked -- -D warnings
```

Both exited 0. The `test result:` lines, in order (unit, `acceptance`, `hooks`, `socket`, doc-tests):

```
sm_daemon (unit)         test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm_daemon acceptance     test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.52s
sm_daemon hooks          test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.71s
sm_daemon socket         test result: ok. 41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.43s
Doc-tests sm_daemon      test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Clippy printed no warning, and the container exited 0.

The Unix-only tests `s3_a_leftover_socket_file_is_removed` and `s3_the_socket_is_user_only` ran and passed there. The two Windows-only tests, `spike1_a_second_first_instance_of_the_pipe_is_refused` and `spike1_the_pipe_dacl_admits_only_the_current_user`, ran and passed on Windows.

**Two test changes in the report commit.** `at16_capture_over_the_index_cap_is_stored_whole` is renamed `at17_capture_over_the_index_cap_is_stored_whole`, because SPEC 11 lists the index cap under AT-17 and `TRACE.md` is built from the prefix (B-40). `s8_a_reply_fits_its_limit_and_counts_what_it_left_out` is new: it covers the budget rule (SPEC 8, B-28), which no earlier test checked with a small `limit`. The same commit corrects one citation in the module comment of `src/capture.rs` (B-67 to B-65, which is where the extraction cap's "must not count" rule is). `cargo fmt --check` and `cargo clippy -p sm-daemon --all-targets --all-features -- -D warnings` were run again on Windows after that change, and both exited 0.

## 4. Acceptance tests

All pass on Windows and Linux.

| Test | Test names | What is checked |
|---|---|---|
| AT-3, daemon side | `at3_warm_ambient_latency_idle_and_during_a_bulk_import_within_the_thread_budget` | 100 warm `ambient_recall` calls idle and 100 during a bulk import of 2,000 chunks, each p95 ≤ 300 ms. The bulk session never has more live threads than `bulk_threads`, and each session opens with its budget (the provider records it, standing in for `ThreadReport`). |
| AT-4 | `at4_a_50000_character_memory_is_stored_in_several_chunks_and_embedded` | A 50,000-character `remember` succeeds with at least 10 chunks; the queue embeds all of them; the memory vector exists; a vector recall of a phrase from its middle finds it. |
| AT-5 | `at5_two_readers_and_one_writer_for_60_seconds_see_no_lock_error` | 60 seconds of two readers (vector, auto and keyword `recall`; `status` and `ambient_recall` over all projects) and one writer (`remember`), each on its own connection. No error reaches a client, and every write is stored. |
| AT-9, `remember` side | `at9_remember_redacts_the_fixture_credentials_and_lists_the_kinds` | Every case of `fixtures/capture/redaction_cases.json` through `remember`: the stored content equals the fixture's redacted text in v3 markers; no fake credential is in the content or the title; the reply's `redactions` equal the fixture's kinds; the empty input gets `invalid_input`. The capture side is in `at13_hook_payloads_over_the_socket`. |
| AT-10 | `at10_one_changed_byte_fails_the_model_and_the_rest_is_served` | The real `sm_embed::model::verify_files` over a small file set (one 4 KiB file, size and SHA-256 pinned) runs in the loader. Intact, the daemon is `ready`. With one byte changed, `hello` and `status` say `failed` with `model_integrity`; model requests fail with that error; `forget`, `status` and keyword `recall` are served (B-36). See section 7. |
| AT-11 | `at11_a_changed_chunk_size_or_model_blocks_search_until_reindex` | A changed `chunk.max_tokens`, and a loader with another model id, each give exactly one stamp difference; vector, auto and keyword `recall` fail with `index_mismatch`; `reindex` re-chunks every memory and queues the embedding; then search works again by vector and keyword. |
| AT-12 | `at12_a_memory_of_project_a_is_not_returned_from_b_without_scope_all`, `at12_two_profiles_never_see_each_others_memories` | Project B's recall never returns project A's memory unless `scope: all`. Two daemons on two profiles have separate databases and never return each other's memories. |
| AT-13, daemon side | `at13_hook_payloads_over_the_socket`, `at13_default_settings_store_strong_text_and_nothing_for_weak` | Every applicable payload of `fixtures/capture/hook_payloads/` sent as `capture`: the stored memories match the fixture (content, type, status, importance, provenance `capture`, redactions), nothing for weak text. Includes the Stop loop guard (`cc_stop_loop_guard_active`, `cc_stop_handler_loop_guard`). |
| AT-14, daemon side | `at14_transcript_stores_only_what_post_tool_use_missed`, `s16_transcript_offsets_per_session` | With `fixtures/capture/rust/at14_session.jsonl`, PreCompact and Stop store the assistant text PostToolUse missed and not what it captured; offsets are kept per session, and a capture with no session keeps no state. Printing the Stop continue response and PreCompact's empty output are the hook's (stream D). |
| AT-15, daemon side | `at15_spool_is_drained` | A capture spooled while no daemon runs is processed at start and its file removed; a corrupt file moves to `spool/failed/`; a stale `tmp/` file is removed and a fresh one kept; a file written while the daemon runs is taken within one drain interval (500 ms in the test). The 200 ms hook exit with the daemon stopped is stream D's. |
| AT-16, daemon side | `at16_tiers_are_stored_active` | Medium-signal text is stored active with importance ≥ 0.4 and strong-signal text with ≥ 0.7, from PostToolUse, PreCompact and Stop. |
| AT-17, daemon side | `at17_ambient_block_is_the_reply_text_and_never_repeats`, `at17_capture_over_the_index_cap_is_stored_whole` | `AmbientReply.text` is exactly the block `sm_core::ambient` builds, for Claude Code and Cursor, and no memory repeats in a session. A capture over `capture.max_index_tokens` is stored whole, indexed up to the cap, and journalled as `capture_cap_hit_total` with `capture_size_tokens`. `SPATIAL_MEMORY_AMBIENT_RECALL=off` is the hook's (stream D). |

**The other rules the brief lists**, all passing:

- **Every op** (CONTRACTS 5, B-28): the `s8_*`, `s9_*`, `s5_reindex_requeues_every_chunk`, `s11_doctor_eval_runs_the_retrieval_set_and_its_error`, `s6_ambient_recall_answers_and_refuses_a_blank_prompt` and `s16_capture_*` tests in `socket.rs`; unknown fields in `s3_unknown_fields_are_refused_for_every_op`; the budget in `s8_a_reply_fits_its_limit_and_counts_what_it_left_out`.
- **CONTRACTS 3.1 and 4:** `s3_two_daemons_on_one_profile_give_one_server_and_lock_held`, `s3_shutdown_releases_the_lock_and_a_new_daemon_starts`, `s3_shutdown_over_the_socket_stops_the_daemon_with_its_reason`, `s3_a_leftover_socket_file_is_removed`, `s3_the_socket_is_user_only`, `s3_another_build_gets_version_mismatch`, `s3_shutdown_from_another_build_works_for_this_profile_only`, `s3_a_request_for_another_profile_is_refused`, `s3_an_empty_frame_is_refused_and_the_connection_stays_usable`, `s3_a_malformed_body_is_refused`, `s3_an_oversized_frame_closes_the_connection_and_the_daemon_still_serves`, `s3_requests_on_many_connections_at_once_are_served`, `s3_a_passed_deadline_gives_timeout`, `s3_hello_says_loading_and_writes_wait_while_reads_are_served`, `s3_the_daemon_exits_when_idle`, `s3_idle_exit_waits_while_requests_keep_coming`, `s3_the_daemon_log_stays_bounded`, `s3_hello_reports_identity`.
- **Windows endpoint:** `spike1_a_second_first_instance_of_the_pipe_is_refused`, `spike1_the_pipe_dacl_admits_only_the_current_user` (reads the DACL back and compares it with `D:P(A;;GA;;;<SID>)`).
- **SPEC 6:** `s6_identifier_query_appends_marked_keyword_hits_after_vector_hits`, `s6_weak_results_are_marked_not_dropped`, `s6_ambient_gates`, `s6_ambient_prompt_cases`, `s6_ambient_errors_and_loading`, `s6_ambient_deadline_records_nothing`, `s6_ambient_fixture_integration_data`, and the unit tests `s6_identifier_queries_run_keyword_search` and `s6_anchor_start_vector_is_the_normalized_mean`.
- **SPEC 7:** `s7_regions_cached_until_an_edit_and_coordinates_only_when_asked`, and the unit test `s7_each_run_variant_knows_its_sibling`.
- **B-79:** `s9_sharing_ops_reach_the_implementation_with_the_resolved_project` (each op reaches the fake with the resolved project and identity, then the queue wakes), `s9_import_drops_the_maps_and_its_memory_is_embedded` (the cached run is dropped and the imported memory embedded), `s9_share_and_sync_need_a_repository_and_the_wired_implementation`, `s9_import_export_and_migrate_without_the_implementation_say_so`.

### Hook payloads that cannot apply

`fixtures/capture/hook_payloads/INDEX.json` has 75 entries: 45 are sent as `capture` and compared, and 30 cannot apply to the daemon.

- **Raw standard input (4):** `cc_post_tool_use_empty_stdin`, `cc_post_tool_use_invalid_json`, `cc_pre_compact_empty_stdin`, `cc_stop_empty_stdin`. The payload is the hook's raw input text, which never reaches the daemon as JSON. Parsing standard input is stream D's.
- **Not a capture event (26):** the daemon refuses a SessionStart or UserPromptSubmit `capture` with `accepted: false` (`s16_capture_refusals`), and these cases check the hook's printed output (stream D).
  - SessionStart: `cc_session_start_startup`, `cc_session_start_resume`, `cc_session_start_clear`, `cc_session_start_compact`, `cc_session_start_unknown_source`, `cc_session_start_missing_source`, `cc_session_start_empty_stdin`, `cc_session_start_whitespace_stdin`, `cc_session_start_not_json`, `cc_session_start_json_list`, `cc_session_start_output_format`, `cc_session_start_handler_empty_source`.
  - UserPromptSubmit: `cc_user_prompt_submit_recall_off`, `cc_user_prompt_submit_off_unit_payload`, `cc_user_prompt_submit_formatted_block`, `cc_user_prompt_submit_empty_prompt`, `cc_user_prompt_submit_blank_prompt`, `cc_user_prompt_submit_missing_prompt`, `cc_user_prompt_submit_recall_raises`, `cc_user_prompt_submit_empty_results`, `cc_user_prompt_submit_namespace_env`, `cc_user_prompt_submit_default_top_k`. The daemon's side of the prompt hook is `ambient_recall`, tested from `ambient_recall_cases.json`.
  - Cursor: `cursor_sessionStart_synthesized_source`, `cursor_sessionStart_clear`, `cursor_sessionStart_detected_by_version`, `cursor_userPromptSubmit_block`.
- **`transcript_cases.json`:** `load_corrupted_json_returns_defaults` and the state probes cannot apply. v3 keeps offsets in the daemon's memory, so there is no state file to corrupt and no path built from a session id. `load_empty_session_id` and `save_empty_session_id_noop` do apply: a sessionless capture keeps no state (decision 8).

### Ambient cases that cannot apply

From `fixtures/capture/ambient_recall_cases.json`, covered: the empty, blank and missing prompt (`invalid_input`); `recall_returns_formatted_block` (AT-17); `recall_raises_returns_none` (`s6_ambient_errors_and_loading`); `recall_empty_results_returns_none` (the `empty` gate in `s6_ambient_gates`); all 9 prompt probes (`s6_ambient_prompt_cases`: non-strings get `invalid_input`, short strings `short_prompt`); and the misc probes `client_cursor_same_output`, `payload_without_session_or_cwd`, `hybrid_recall_raises` and `build_factory_raises`. These cannot apply:

- **The hook's switches:** `ambient_recall_off_returns_none`, the `ambient_recall_env` probes (13) and the `top_k_env` probes (14). The hook resolves them (`config::resolve_ambient`, tested in sm-core `at17_version_2_ambient_switches_keep_their_rules`) and sends the resolved `top_k`.
- **Namespaces:** `namespace_env_passed_through` and the `namespace_env` probes (4). v3 has projects, not namespaces.
- **Python object shapes:** the misc probes `client_other_string`, `result_without_memories_attribute`, `memories_null` and `similarity_null_escapes`. v3's replies are typed; a reply without memories or with a null similarity cannot be built.
- **The block format:** the `format` probes (15). They test `sm_core::ambient::format_block` (sm-core `at17_ambient_block_keeps_version_2_format`); the daemon returns its output unchanged (AT-17 above).
- **Environment round trip:** `env_round_trip` (8 entries), the hook's handling of its environment.

## 5. Measured budgets

Machine as in section 3, Windows 11, `test` profile (debug), fake embedder, `physical_cores` fixed at 8 in `DaemonOptions` so the budget is the same on every machine. These are daemon-path figures only; the real-model figures are phase 3.

Figures from the final run in section 3 (printed with `--nocapture`), and from an earlier full run of the same tests on the same machine to show the spread.

| What | Final run | Earlier run | Budget the test asserts |
|---|---|---|---|
| AT-3, warm `ambient_recall`, idle (100 calls) | median 547.6 µs, p95 961.6 µs | median 548.6 µs, p95 1.002 ms | p95 ≤ 300 ms |
| AT-3, the same during a bulk import of 2,000 chunks (100 calls) | median 4.097 ms, p95 5.402 ms; 1,728 chunks still pending after | median 3.319 ms, p95 4.089 ms; 1,760 pending after | p95 ≤ 300 ms; the import still running after the calls |
| AT-3, threads | bulk budget 6 of 8, at most 6 alive; query session 4; sessions opened `[(Query, 4), (Bulk, 6)]` | the same | live bulk threads ≤ the bulk budget; each session opened with its budget |
| AT-4, a 50,000-character `remember` | 26 chunks, all embedded | 26 chunks, all embedded | at least 10 chunks; nothing pending |
| AT-5, 60 seconds, two readers and one writer | reader a 2,380 requests, reader b 7,064, writer 8,102; no error | 2,485, 6,859, 8,702; no error | no error; every write stored |
| AT-9, `redaction_cases.json` through `remember` | 33 cases redacted, 7 left unchanged | the same | every case equal to the fixture |
| AT-15, a spool file written while the daemon runs | taken in 452.3 ms (interval 500 ms) | 501.8 ms | gone within the interval plus 1 s |

During the import the median rises from about 0.5 ms to about 4 ms, far inside the budget, so a query is not held behind the bulk queue. This is a debug build with a fake embedder that sleeps 3 ms per document, so only that shape carries over, not the numbers. For scale, `sm-embed`'s real-model test in the same run printed a median query embedding of 47.3 ms at 128 tokens on 4 threads, and a peak of 1,416 MB with both sessions loaded. Those are `sm-embed`'s figures, not measured through the daemon.

## 6. Decisions

Each needs a B-n entry from the lead.

1. **Threads, not tokio (SPEC 14 stream E build 2; SPEC 3).**
   - *What:* one accept thread, one thread per connection, a loader thread, the bulk worker, the capture worker and the maintenance thread, over `interprocess`'s blocking listener. No async runtime and no extra crate feature.
   - *Why:* every request is short and blocks on SQLite or ONNX Runtime, so a runtime would add a dependency and nothing else; a slow op holds only its own connection's thread.
   - *Evidence:* `s3_requests_on_many_connections_at_once_are_served`; AT-3's p95 while the bulk queue runs (section 5); AT-5 with three busy connections.
2. **The Windows pipe descriptor (SPIKE c; spike decisions 1 and 5).**
   - *What:* SDDL `D:P(A;;GA;;;<current user SID>)`, protected from inheritance, made with `ConvertStringSecurityDescriptorToSecurityDescriptorW`, copied into an owned `interprocess::os::windows::security_descriptor::SecurityDescriptor`, and passed with `ListenerOptionsExt::security_descriptor`. Verified in the `interprocess` 2.4.8 source: it adds `FILE_FLAG_FIRST_PIPE_INSTANCE` itself when it creates the first instance, and `PIPE_REJECT_REMOTE_CLIENTS` is its default.
   - *Evidence:* `spike1_a_second_first_instance_of_the_pipe_is_refused`, `spike1_the_pipe_dacl_admits_only_the_current_user`.
3. **Requests that need the model are refused with `daemon_loading` while it loads (stream E build 3).**
   - *What:* `remember`, `revise` with new content or `supersede_with`, vector and auto `recall`, `ambient_recall`, `reindex` and `doctor_eval` need the loaded model and get `daemon_loading` until it is ready; keyword `recall`, `forget`, `link`, `related` (stored vectors), `status`, `regions`, `journey` and `consolidate` are served. A `capture` is accepted and queued while loading; the worker waits for the model, and spools the queue if loading fails.
   - *Why:* the chunker needs the tokenizer, which loads with the model, and a memory is never stored without its chunks. A vector search needs the query embedded. Everything else reads rows or stored vectors.
   - *Evidence:* `s3_hello_says_loading_and_writes_wait_while_reads_are_served`.
4. **Bulk session grace: 10 seconds (SPEC 5, query and bulk sessions; B-27).**
   - *What:* the bulk session stays open for `bulk_grace` (10 s) after the queue empties, then is dropped. Nothing new is journalled.
   - *Why:* a burst of writes (a capture burst, an import) reuses one session instead of opening one per `remember`; memory returns to the idle level 10 s after the last batch.
   - *Evidence:* AT-3 opened one bulk session for the whole 2,000-chunk import (section 5). The real cost of opening a session is not measured (section 9).
5. **The log is bounded by rotation (stream E build 3).** `daemon.log` rotates to `daemon.log.1` at 1 MiB (`log_max_bytes`), so at most about 2 MiB is kept; each message is cut at 2,000 characters and written in one call. *Evidence:* `s3_the_daemon_log_stays_bounded`.
6. **Bounded caches (SPEC 9; SPEC 16; B-32).** Project resolutions per folder: 256. Capture sessions (state and transcript offset): 256 (`capture_sessions`). Recall ranks for `reinforce`: 256 sessions (`rank_sessions`). Each is an LRU. *Why:* a daemon lives for days. A forgotten capture session starts again at offset 0 with fresh hashes, and the store's dedupe by content hash (B-29) stops a text being stored twice. *Evidence:* `s8_remember_creates_then_finds_the_duplicate`, `s16_transcript_offsets_per_session`.
7. **The capture queue holds 1,024 jobs (SPEC 16, never loses).** More, or a capture while the model has failed, or one arriving while the daemon stops, is written to `spool/new/` for a later daemon. A job that fails for another reason goes to `spool/failed/`, is logged, and is never retried (B-33). *Evidence:* `at15_spool_is_drained`; the code path in `capture.rs`.
8. **A capture with no session keeps no state (SPEC 16).** With neither the request's nor the payload's `session_id`, the transcript is read from offset 0 with fresh hashes, as version 1 did (`load_empty_session_id`, `save_empty_session_id_noop`), so two unrelated sessionless transcripts never share an offset. *Evidence:* `s16_transcript_offsets_per_session`.
9. **`capture_size_tokens` (stream E build 8; B-65).** The value is the chunker's token count of the whole content (`ChunkPlan.total_tokens`, the same tokenizer as `count_tokens`), labelled with `project`, `tool` and `indexed_tokens`; `capture_cap_hit_total` is recorded when `ChunkPlan.capped`. The 200,000-character extraction cap (`provenance.extra.truncated`) is not counted. *Evidence:* `at17_capture_over_the_index_cap_is_stored_whole`.
10. **The spool drain is skipped while the model is `failed` (B-33, B-36).** Those captures cannot be chunked, so the files stay for a later daemon instead of being read and spooled again every minute. *Evidence:* code only (`maintenance.rs`).
11. **The map cache key covers vectors (SPEC 7; B-46, B-47).**
    - *What:* the key is the SHA-256 of `sm_spatial::input_hash`, the project, and each memory's id, `updated_at` and vector bytes. Runs live in `map_runs`, so they survive a restart; a per-project index in memory skips reading vectors on a hit. A run without coordinates is stored under the suffix `:regions` with every point at (0, 0), which is never returned; a run with them under `:coords`. A plain request is served by either. Labels are copied between the two variants of the same data.
    - *Why:* `input_hash` covers ids, not vectors (c-report decision 6), and any edit, forget or re-embed must drop the run.
    - *Evidence:* `s7_regions_cached_until_an_edit_and_coordinates_only_when_asked`, `s7_each_run_variant_knows_its_sibling`. The (0, 0) points are a contract change request (section 8).
12. **Sharing calls wake the queue and drop the maps after every call, successful or not (B-79).** A failed import may already have stored part of a file. *Evidence:* `s9_sharing_ops_reach_the_implementation_with_the_resolved_project`, `s9_import_drops_the_maps_and_its_memory_is_embedded`.
13. **`UnwiredSharing` (B-79).** `start` and `run` take `Arc<dyn SharingOps>` with no default, so a caller must choose an implementation. `UnwiredSharing` is a public one that fails every call with `internal` and the next action "Start the daemon with `spatial-memory daemon`, which wires in sharing"; it never pretends to succeed. *Why:* the `daemon` subcommand can start before phase 3 wires `sm-share` in, and tests that do not use sharing need no fake. The lead may delete it once the real implementation exists. *Evidence:* `s9_import_export_and_migrate_without_the_implementation_say_so`.
14. **Unknown keys refused in dispatch (B-28).** The request, the `call` wrapper and the arguments of `shutdown`, `ambient_recall`, `capture`, `import`, `export`, `migrate`, `reindex`, `project_rename`, `project_delete` and `doctor_eval` are checked against their keys before decoding, because those structs do not deny unknown fields (section 8). *Evidence:* `s3_unknown_fields_are_refused_for_every_op`.
15. **Link staleness (SPEC 4, links; B-37).** Links made through the tool are `manual` and carry no commit. `list` uses `head_only(head)`: a `verified_commit` equal to the head (or a prefix of it) is current, anything else is undecided rather than stale, because the daemon has no ancestry walk. When the head cannot be read from the repository's files, `stale` is absent. *Evidence:* `s8_link_add_list_remove_and_its_errors`.
16. **`forget`'s confirm rule is literal (B-30).** "More than one memory" needs `confirm: true`; one memory does not. *Evidence:* `s8_forget_needs_confirm_for_more_than_one_memory`.
17. **`doctor_eval` uses a scratch store for a set with its own memories (SPEC 11).** It is built in the profile folder from the set's memories and removed afterwards; a set without memories runs against the live store, over the current project if `cwd` resolves and over every project otherwise. Each query takes recall's vector path. *Evidence:* `s11_doctor_eval_runs_the_retrieval_set_and_its_error`.
18. **Ambient search asks for `top_k` plus the session's injected memories (SPEC 6, ambient recall).** Memories already shown are replaced by the next best instead of turning a fresh result into `repeat`. *Evidence:* `at17_ambient_block_is_the_reply_text_and_never_repeats`.
19. **The ambient switches are the hook's (SPEC 6; B-34).** The daemon never answers `disabled`, uses the `top_k` it is sent, and does not journal `hook_gate_total`, `hook_top_score` or `hook_latency_ms`. *Evidence:* `s6_ambient_gates`.
20. **The daemon checks the four-word minimum again (SPEC 6).** A direct caller gets `short_prompt` as the hook would. Words are counted as Python's `str.split()` does (`ambient::prompt_word_count`), so a prompt of only control whitespace passes Rust's `trim` and still gets `short_prompt`. SPEC 6's minimum supersedes version 2's behaviour. *Evidence:* `s6_ambient_prompt_cases`.
21. **Injections are recorded after the reply is written, and only before the deadline (SPEC 6, step 6; B-37).** A reply built after the deadline is `timeout` and records nothing. B-37 accepts that the daemon records an injection when it builds the block; recording after the write narrows that race, because a reply that is never written records nothing. B-37's gap remains when the hook's own deadline passes after the reply is written. Deadlines are checked after embedding, after each search, and inside the long loops (`regions`, `consolidate`, `doctor_eval`). *Evidence:* `s6_ambient_deadline_records_nothing`, `s3_a_passed_deadline_gives_timeout`.
22. **Idle exit (CONTRACTS 4).** With `daemon.idle_exit_minutes` above 0, the daemon stops after that long with no request, nothing in flight, no capture queued and no bulk session open. *Evidence:* `s3_the_daemon_exits_when_idle`, `s3_idle_exit_waits_while_requests_keep_coming`.
23. **Shutdown grace: 5 seconds (CONTRACTS 3.1).** After the endpoint closes, in-flight requests and the workers get `shutdown_grace`; queued captures are spooled; the lock is released last. *Evidence:* `s3_shutdown_releases_the_lock_and_a_new_daemon_starts`.
24. **Budget estimate (SPEC 8, output; B-28).** Tokens are estimated as for the hook block (four characters a token) over each item's JSON, plus one per item and 40 for the reply's own fields. Items are kept in order until the next does not fit; a first item too large alone is kept with its content cut and `content_truncated` set. *Evidence:* `s8_a_reply_fits_its_limit_and_counts_what_it_left_out`.
25. **`consolidate` compares every pair (B-31).** Proposals compare the mean vectors of every pair of active, embedded memories in the project: O(n²), about 2 million dot products at 2,000 memories. Not timed here.
26. **Test conventions.** Each test project folder holds its own `.git`, so project resolution never climbs into whatever encloses the temporary folder. `physical_cores` is fixed at 8 in the tests. The ambient fixture integration test lowers the similarity floor to 0.30 for that test only, because the fake embedder's cosine for the fixture's texts is about 0.43; the default is unchanged. AT-3's bulk import is 20 `remember` batches of 100 one-chunk memories, embedded by a recording provider that takes 3 ms per document, so the import runs during all 100 calls. AT-4 reads vectors through a second `Store` handle configured with the daemon's stamp, because the store refuses vector reads before `configure_index`.

## 7. Deviations

- **AT-10.** SPEC 11 says a changed byte "stops the daemon from starting, with a clear error". Following B-36, the daemon starts, binds, and stays up in `failed` with `model_integrity` as `state_error`: `hello` and `status` give the clear error, every model request fails with it (the test checks vector `recall`, `remember` and `reindex`), and the rest is served (the test checks keyword `recall`, `forget` and `status`). *Evidence:* `at10_one_changed_byte_fails_the_model_and_the_rest_is_served`.
- **AT-4, peak memory.** SPEC 11 also asks for peak memory under 2 GB. With the fake embedder the figure means nothing, so it is not measured; it is phase 3 with the real model.

## 8. Contract changes requested

1. **`deny_unknown_fields` on the remaining protocol structs (B-28).** `Request` and the parameters of `shutdown`, `ambient_recall`, `capture`, `import`, `export`, `migrate`, `reindex`, `project_rename`, `project_delete` and `doctor_eval` in `sm_core::protocol` accept unknown keys; the tool parameters deny them. The daemon refuses unknown keys itself (decision 14), so nothing is broken. With the attribute on those structs, dispatch's key lists could go. *Evidence:* `crates/sm-core/src/protocol.rs` (`Request` at line 219, `ShutdownParams` at line 1172, and the others after it); `s3_unknown_fields_are_refused_for_every_op`.
2. **Optional map positions in `sm-store` (SPEC 7; B-47).** `sm_store::MapPointRow` needs finite `x` and `y` (`crates/sm-store/src/maps.rs`, line 122), but a run without coordinates has none, so the daemon stores (0, 0) and records `"coords": false` in the run's parameters (decision 11). Nullable positions would let a run say what it holds.

## 9. Unfinished or unverified

- **The real `SharingOps`.** `share`, `sync`, `import`, `export` and `migrate` run only through the test fake and `UnwiredSharing`. The lead writes the implementation over `sm-share` in phase 3.
- **The real model in the daemon.** `GemmaLoader` is built but never loaded by a test here, because the one real-model test binary lives in `sm-embed`. The real-model AT-3 figure, the real `ThreadReport` of the query and bulk sessions, and the cost of opening a bulk session (decision 4) are phase 3.
- **Peak and idle memory** (AT-4, SPEC 11 budgets) are not measured: phase 3.
- **The hook side** of AT-3 (with the daemon killed, exit 0 with empty output inside 1,500 ms), AT-14 (printed output), AT-15 (exit within 200 ms with the daemon stopped) and AT-17 (the off switch) belongs to stream D.
- **Decision 10** (the drain skipped while `failed`) and the 1,024-job spill in decision 7 are covered by code, not by their own tests.

## 10. Licences

Packages added to `Cargo.lock` since `9387a08`:

| Crate | Version | Licence |
|---|---|---|
| `interprocess` | 2.4.8 | 0BSD OR Apache-2.0 |
| `doctest-file` | 1.1.1 | 0BSD |
| `recvmsg` | 1.0.0 | 0BSD |
| `widestring` | 1.2.1 | MIT OR Apache-2.0 |
| `sm-daemon` | 3.0.0 | MIT |

**Flag:** `doctest-file` and `recvmsg` are 0BSD only (both come in through `interprocess`). 0BSD (Zero-Clause BSD) is a BSD-family licence with no conditions, equivalent to public domain, but it is not written on the permissive list. `LICENSES.md` already records `adler2` with 0BSD as one option. The lead decides whether to add 0BSD to the list.

The crate's other dependencies were already in the lock: `windows-sys` 0.52.0 and 0.61.2, `libc` 0.2.191, `serde`, `serde_json` and `tempfile`.

**SPEC 13 risk retired.** SPEC 13 lists `interprocess` as "Chosen from documentation, not tested", with loopback TCP and a per-user token file as the fallback. It is now tested: the named pipe on Windows and the Unix socket on Linux. The fallback was not needed.
