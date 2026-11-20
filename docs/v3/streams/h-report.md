# Package H report (phase 3): `sm-daemon` hardening

Branch `arman-tech/stream-h-hardening`, from `v3` at `93044bb`. Commits, in order:

| Commit | Items |
|---|---|
| `6c0aa70` | 2, 3, 4, 11 and 17 (the endpoint and the profile folder), and the documentation of item 9 |
| `d08d60f` | 1, 5, 6 and 7 (capture and the spool) |
| `887a875` | 8 (the bulk queue's strikes) |
| `7c29de4` | 10 (a view that shed a field says it was cut) |
| `6f2a94b` | 12 (the drift guard of the ten ops) |
| `c7af5bb` | 13 (spike decision 5) |
| `79c5c9d` | 14 (the drain while `failed`, the spill at 1,024) |
| `5e4d3aa` | 15 (a title-only revise re-embeds) |
| `f17acdf` | 16 (AT-16 through the daemon) |
| `106d114` | 12 (the drift guard of the request envelope, the eleventh struct of B-102) |
| this commit | this report |

The layout follows `RULES.md` section 9. Item 17 is not in `stream-h.md`: the coordinator added it during the package (the profile folder is user-only on Unix).

Abbreviations, written out once:
- AT-n is acceptance test n in SPEC section 11.
- API is application programming interface, and JSON is JavaScript Object Notation.
- ACL is access control list, DACL is discretionary access control list, and SID is security identifier.
- uid is a Unix user id; `SO_PEERCRED` and `LOCAL_PEERCRED` are the Linux and macOS socket options that give a connected peer's credentials.
- SipHash is the keyed hash behind Rust's `RandomState`.

## 1. Done when

**Yes.** Every Build item (1 to 16) and the added item 17 is fixed and tested, documented (item 9), or tested (items 12, 13, 14 and 16) as `stream-h.md` asks. Both gates pass on Windows, and the full workspace passes on Linux (Docker); section 3 has every `test result:` line.

- **Tests.** `sm-daemon` has 124 test functions: 25 unit, 55 in `socket`, 30 in `hooks` and 14 in `acceptance`; 33 are new. Windows runs 108 (13 unit, 52, 29, 14) and Linux 118 (23 unit, 52, 29, 14). Twelve unit tests, three `socket` tests and one `hooks` test are Unix-only; two unit tests, three `socket` tests and one `hooks` test are Windows-only. At `93044bb` there were 91 (Windows 86, Linux 89). All pass on both.
- **Public API.** Unchanged: `connect`, `endpoint::*`, `DaemonHandle`, `start`, `run`, `Started`, `Exit` and `DaemonOptions` keep their names and signatures. `connect` and `bind` refuse more (items 2, 3 and 4), always with `daemon_unavailable` and a next action, as before. Nothing public was added.
- **Package J's files** (`src/sharing.rs`, `src/share_adapter.rs`, their `lib.rs` and `Cargo.toml` lines) are untouched. The helpers this package added to `tests/common/mod.rs` are appended at its end, under `// --- stream H helpers`. No new test names `sm_daemon::UnwiredSharing`.
- **Nothing outside `crates/sm-daemon/`** changed except this report. `Cargo.lock` and every `Cargo.toml` are unchanged.

| Item | Fix | Tests | Decision |
|---|---|---|---|
| 1. A capture candidate that always fails blocks its session (should fix) | Failures are per candidate. A candidate that can never be stored (`invalid_input`, `secret_detected`), or that still fails at attempt `MAX_ATTEMPTS` (5), is dropped with one log line; the job's other candidates are stored and the session moves on. The request moves to `spool/failed/` as the record. A candidate that could still be stored is never dropped before the last attempt. | `s16_a_candidate_that_can_never_be_stored_is_dropped_and_its_session_moves_on`, `s16_a_candidate_that_keeps_failing_is_dropped_at_the_bound_and_its_session_moves_on`, `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once` (changed, section 7) | 1, 2 |
| 2. The owner check after `connect` passes when the socket is gone (should fix) | After connecting, before anything is sent, the client refuses a socket entry that is gone or is another user's, and a peer whose effective uid is not this user's (`SO_PEERCRED` on Linux, `LOCAL_PEERCRED` on macOS, through `interprocess`'s `peer_creds`). The daemon checks each client's uid the same way and drops another user's before reading. | unit (Linux): `s3_a_socket_gone_after_connecting_is_refused`, `s3_a_peer_of_another_user_is_refused_by_client_and_daemon` | 3 |
| 3. Windows: the client does not check who serves the pipe | Once connected, the client reads the server's process id (`GetNamedPipeServerProcessId`), opens the process and its token, and refuses the pipe unless the token's user SID equals the current user's (`EqualSid`). The first instance has `FILE_FLAG_FIRST_PIPE_INSTANCE`: `interprocess` 2.4.8 sets it (`create_instance.rs` line 86). | unit (Windows): `spike1_connect_accepts_a_pipe_this_user_serves`, `spike1_connect_refuses_a_pipe_another_user_serves`; stream E's `spike1_a_second_first_instance_of_the_pipe_is_refused` | 4 |
| 4. The fallback socket folder (B-112) | `bind` creates a missing socket folder with mode 0700. `bind` and `connect` refuse a fallback folder (`spatial-memory-<16 hex>`) that is a link, that another user owns, or that group or others can write to, naming the folder, with a next action. Any socket folder another user owns is refused. | unit (Linux): `s3_the_fallback_socket_folder_is_created_user_only`, `s3_a_fallback_folder_that_is_not_private_is_refused_by_bind_and_connect`, `s3_a_group_writable_profile_folder_still_binds`, `s3_only_the_hashed_folder_is_the_fallback_folder` | 5 |
| 5. `project_unresolved` fails at once | Transient: tried again within `MAX_ATTEMPTS`, then `spool/failed/`. | `s16_an_unresolved_project_is_tried_again_within_the_bound`, `s16_a_project_that_never_resolves_moves_to_failed_after_the_bound` | 6 |
| 6. An unremovable spool file is processed again on every drain | Its name goes to `Shared::unremovable` (under the `held` lock, before the name is released) with one log line. The drain never claims it again; it retries the removal quietly at each pass and forgets the name once the file is gone. | `s16_a_stored_capture_whose_file_cannot_be_removed_is_not_processed_again` (a Unix and a Windows variant) | 7 |
| 7. A failed `hold_new` processes the capture from memory | Kept, and made explicit: logged, processed from memory, the spool write tried again before each retry, and a capture that fails for good written to `spool/failed/` as usual. | `s16_a_capture_that_cannot_be_spooled_is_processed_from_memory` | 8 |
| 8. An embedder-wide error strikes every chunk | A chunk is struck only in a round in which another chunk was stored; a batch of one is never struck. The wait after a round with nothing stored doubles, at most 6 times (5 s to 320 s). | `s5_an_embedder_wide_failure_strikes_no_chunk_and_every_chunk_embeds_after_it`, `s5_a_lone_chunk_that_fails_is_never_struck`, `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on` (changed, section 7) | 9 |
| 9. Threads can outlive `DaemonHandle::wait` | Documented on `DaemonHandle::wait`: the caller must exit the process once it returns; the same holds for `shutdown` and for dropping the handle. | none (documentation) | 10 |
| 10. A short memory can lose fields while `content_truncated` stays false | `content_truncated` is also set when a shed group removed a field the memory had. A cut that removes nothing is not marked. | unit `s8_a_short_memory_that_sheds_fields_is_marked_cut`; `s8_a_short_memory_cut_to_fit_says_it_was_cut` | 11 |
| 11. The private bind folder's name is predictable | `.` and 8 hex characters from a freshly keyed SipHash over the process id, the time and the attempt; created with `DirBuilder::create` (fails on any existing entry); up to 16 names tried. | unit (Linux): `s3_the_private_bind_folder_name_is_unpredictable`; stream E's `s3_a_socket_bound_in_a_private_folder_is_user_only_and_moved_into_place` | 12 |
| 12. Drift guard (B-102) | Test only. | unit: `s3_allowed_args_match_every_key_the_unchecked_params_serialise`, `s3_request_keys_match_every_key_a_request_serialises` | 13 |
| 13. Spike decision 5 | Test only. | `spike5_the_query_session_is_opened_first_and_kept_for_life` | 14 |
| 14. The drain while `failed` (B-96), the spill at 1,024 (B-98) | Tests only; both close a "covered by code only" gap. | `s16_the_spool_is_not_drained_while_the_model_has_failed`, `s16_a_capture_past_1024_queued_is_spilled_to_the_spool` | 15 |
| 15. A title-only `revise` keeps the old title's vectors | It did: the bulk queue embeds the title with each chunk's text, and a title change rebuilt nothing. Now a title change rebuilds the chunks from the stored content, in the same transaction, and they wait for the bulk queue; a captured memory keeps its capture index cap. | `s8_a_title_only_revise_re_embeds_the_chunks_with_the_new_title`, `s8_a_new_title_waits_for_the_model_and_other_fields_do_not`, `s8_a_title_revise_of_a_captured_memory_keeps_its_index_cap` | 16 |
| 16. AT-16 through the daemon | Test only. | `at16_importance_does_not_change_recall_ranking` | 17 |
| 17. The profile folder (added by the coordinator) | Unix: a missing profile folder is created 0700; an existing one of this user open to group or others is set to 0700 with one log line; one another user owns is refused with `daemon_unavailable`. Windows is left as it is. | unit (Linux): `s3_the_profile_folder_is_user_only`, `s3_a_profile_folder_another_user_owns_is_refused`; `s3_the_daemon_makes_the_profile_folder_user_only` (Unix), `s3_the_daemon_leaves_the_profile_folder_alone_on_windows` (Windows) | 18 |

**Item 15, how it was found.** `queue.rs` builds each document from the memory's title and the chunk's text, and `revise` passed a title change to the store without a `ContentChange`, so the chunks kept their vectors. `s8_a_title_only_revise_re_embeds_the_chunks_with_the_new_title` records every document the fake embedder receives (`DocsProvider`) and asserts that the chunks are embedded again with the new title.

## 2. Files built

All under `crates/sm-daemon/`, except this report. From `93044bb` to `106d114`: 13 files changed, 2,888 lines added and 257 removed.

- `src/endpoint.rs`: the Unix folder checks (B-112), the socket-gone and peer uid checks on both ends, the unpredictable private folder, the Windows pipe server's user check (`mod security`: `token_user`, `runs_as`), `current_user`, and `Accepted` (a client of this user, or one refused). 10 new unit tests (8 Linux, 2 Windows).
- `src/server.rs`: `private_profile` (item 17), called in `start` before the lock file is created; refused clients are logged; the `DaemonHandle::wait` documentation (item 9). 2 new unit tests (Unix).
- `src/capture.rs`: permanent and transient codes, per-candidate drops, `project_unresolved` transient, `Shared::unremovable` and `retry_removal`, processing from memory when the spool write fails, and the retry of that write.
- `src/maintenance.rs`: the drain skips, and retries the removal of, a stored capture's unremovable file.
- `src/queue.rs`: strikes only in a round in which another chunk was stored; the stall backoff and `MAX_STALL_DOUBLINGS`.
- `src/budget.rs`: `content_truncated` set when a shed group removed a field. 1 new unit test.
- `src/dispatch.rs`: the B-102 drift guards. 2 new unit tests.
- `src/tools/edit.rs`: a title change rebuilds the chunks; every `revise` holds `Shared::revising`.
- `src/state.rs`: `Shared::unremovable` and `Shared::revising`.
- `tests/socket.rs`: 8 new tests (7 per platform).
- `tests/hooks.rs`: 10 new tests (9 per platform); 1 changed.
- `tests/acceptance.rs`: `at16_importance_does_not_change_recall_ranking` and its helpers.
- `tests/common/mod.rs`: appended at the end: `Rig::start_with_provider`, `Rig::start_loader`, `failing_loader`, `DownProvider` (every batch fails while down) and `DocsProvider` (records every document embedded).
- `docs/v3/streams/h-report.md`: this report.

## 3. Test commands and exact results

Commands run through PowerShell from the worktree, at `106d114` (the source this report describes; the report commit changes no code). Output folder: `%USERPROFILE%/AppData/Local/Temp/claude/409aeca82b3b64a0/120806bb-ad38-440a-a1f2-1d2c0b5ad9bb/scratchpad/gate2/`.

### Windows 11 Pro (10.0.26300), Rust 1.95.0, debug profile

Exit codes: `cargo fmt --check` 0; `cargo clippy --workspace --all-targets --all-features -- -D warnings` 0; `cargo test -p sm-daemon --all-features` 0; `cargo test --workspace --all-features` 0.

`cargo test -p sm-daemon --all-features`:

```
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s      (unit)
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.51s     (acceptance)
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.55s      (hooks)
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.83s      (socket)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s       (doc-tests)
```

`cargo test --workspace --all-features`, every line in order:

```
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s   (sm_capture unit)
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s   (sm_cli unit)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (spatial_memory unit)
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.95s   (autostart)
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 66.27s   (hook_budgets)
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.39s   (hook_fixtures)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s   (mcp_proxy)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s   (ops)
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.71s   (plugin)
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s   (setup)
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm_core unit)
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm_daemon unit)
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.66s   (acceptance)
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.69s   (hooks)
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.94s   (socket)
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s   (sm_embed unit)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.41s   (real_model)
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.27s   (sm_share unit)
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.64s   (sm_spatial unit)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.05s   (at6_determinism)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s   (at7_quality)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.80s   (s7_pipeline)
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.66s   (sm_store unit)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_capture doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_cli doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_core doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_daemon doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_embed doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_share doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_spatial doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_store doc-tests)
```

The real-model binary (`real_model`) ran: the model is in this machine's per-user cache.

### Linux (Docker `rust:1.95-trixie`, a `git archive` snapshot of `106d114`, debug profile)

```
MSYS_NO_PATHCONV=1 docker run --rm -v "<snapshot>:/src:ro" -v sm-v3-cargo:/usr/local/cargo/registry -v sm-v3-target-h:/target -e CARGO_TARGET_DIR=/target -w /src rust:1.95-trixie bash -c "rustup component add clippy rustfmt >/dev/null 2>&1; cargo fmt --check; echo FMT_EXIT=\$?; cargo clippy --workspace --all-targets --all-features --locked -- -D warnings; echo CLIPPY_EXIT=\$?; cargo test --workspace --all-features --locked --no-fail-fast; echo TEST_EXIT=\$?"
```

Exit codes: `FMT_EXIT=0`, `CLIPPY_EXIT=0`, `TEST_EXIT=0`; `docker run` exited 0. Output: `%USERPROFILE%/AppData/Local/Temp/claude/409aeca82b3b64a0/120806bb-ad38-440a-a1f2-1d2c0b5ad9bb/scratchpad/linux-2.txt`.

`cargo test --workspace --all-features --locked`, every line in order:

```
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s   (sm_capture unit)
test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s   (sm_cli unit)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (spatial_memory unit)
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.89s   (autostart)
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.45s   (hook_budgets)
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.45s   (hook_fixtures)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s   (mcp_proxy)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s   (ops)
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s   (plugin)
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s   (setup)
test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm_core unit)
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm_daemon unit)
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.57s   (acceptance)
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.32s   (hooks)
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.85s   (socket)
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s   (sm_embed unit)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (real_model)
test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.18s   (sm_share unit)
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.64s   (sm_spatial unit)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.15s   (at6_determinism)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s   (at7_quality)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.86s   (s7_pipeline)
test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.63s   (sm_store unit)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_capture doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_cli doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_core doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_daemon doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_embed doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_share doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_spatial doc-tests)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (sm_store doc-tests)
```

On Linux the `real_model` binary skipped its body, as it is built to do: it printed `real_model: SKIPPED: the model folder /root/.cache/spatial-memory/_models/embeddinggemma-2/... is absent`, because the container has no model cache. The counts that differ from Windows (`sm_cli` 36 against 38, `autostart` 12 against 11, `sm_core` 97 against 96, `sm_daemon` unit 23 against 13) come from platform-specific tests. Only the `sm_daemon` difference involves tests this package wrote (section 1).

## 4. Acceptance tests

This package owns one acceptance test and closes three recorded gaps.

- **AT-16, importance does not change ranking, through the daemon:** `at16_importance_does_not_change_recall_ranking` (`tests/acceptance.rs`). Passes on Windows and Linux.
  - Seven memories; two of them have the same words in another order, so they tie on score.
  - The same calls are made three times: with every memory at the default importance, with importances the reverse of the ranking, and with importances in its order. Each time: vector `recall` (k = 7), auto `recall` (k = 2), keyword `recall` of a phrase three memories hold, and the prompt hook's `ambient_recall` (top 3, a new session each time, gate `Injected`). All three answers are equal, and the vector order is score descending, then id ascending (B-70), so the tied pair keeps its id order whichever has the higher importance.
  - `min_importance` 0.5 keeps exactly the memories at or above it, in ranking order, under both importance orders: it is a filter only.
- **B-96 gap, the drain while `failed`:** `s16_the_spool_is_not_drained_while_the_model_has_failed`. With a loader that always fails, a capture file, a corrupt file and a stale `tmp/` write stay as they are through 20 drain intervals (a drain would store, quarantine and remove them), and a socket capture is accepted and written to `spool/new/` and stays there. A later daemon with a model processes all of it.
- **B-98 gap, the spill at 1,024:** `s16_a_capture_past_1024_queued_is_spilled_to_the_spool`. While the model loads, 1,024 captures are accepted and queued with no file; the 1,025th is accepted and written to `spool/new/` under its request id, and the drain leaves it while the queue is full. Once the model is ready, the last queued capture and the spilled one are stored, every file is removed, and nothing moves to `spool/failed/`.
- **B-95 gap, spike decision 5:** `spike5_the_query_session_is_opened_first_and_kept_for_life`. A provider numbers every session it opens. When the daemon is ready, the query session is the only one open. That first session serves requests before, during and after a bulk import of 100 memories, and after an idle period long enough for the bulk session to be dropped (past `bulk_grace`) but short of idle exit. No other query session is opened, and it is never dropped while the daemon runs.

Stream E's acceptance tests (AT-3, AT-4, AT-5, AT-9 to AT-17) still pass unchanged.

## 5. Measured budgets

No budget of SPEC 15 was re-measured; the AT-3 tests still pass within their budgets.

- **Queueing 1,024 captures** over the socket while the model loads (`s16_a_capture_past_1024_queued_is_spilled_to_the_spool`, which prints it): 198 ms at `106d114` (`1024 captures queued in 198.3531ms`, run alone with `--nocapture`); an earlier run while writing the test gave about 226 ms. AMD Ryzen 9 9950X3D, 16 cores and 32 logical processors, Windows 11 Pro, debug profile, with up to three other streams building on the machine. It is evidence that the test is cheap, not a budget.

## 6. Decisions

1. **Which capture failures are permanent (item 1).** `invalid_input` and `secret_detected` are permanent: the candidate is dropped at once. Every other code, and a panic, is transient until attempt `MAX_ATTEMPTS` (5), at which a candidate that still fails is dropped. When a candidate fails transiently, the whole attempt fails and the job is tried again after its backoff: nothing is dropped yet and the session stays where it was, so a candidate that could still succeed is never lost; the store's dedupe (B-29) keeps the candidates already stored from being stored twice. Any drop logs one line (the request id, the cause, how many of how many candidates) and moves the request to `spool/failed/` as the record. Trade-off: the job's good candidates wait for the bound too (about 30 seconds with the default `capture_retry_backoff` of 2 s: 2, 4, 8 and 16 s). Evidence: the two item 1 tests; the second asserts the good candidate is stored once, at the last attempt, and that the next capture of the session reads only what was appended. SPEC 16 "never loses", B-33, B-108.
2. **A failure of the job itself does not move the session on (item 1).** When the project cannot be resolved or the pipeline panics before candidates exist, there is no new session state to commit, so the session stays where it was and the request goes to `spool/failed/` at the bound. The next capture of that session reads the same text again. Evidence: `s16_a_project_that_never_resolves_moves_to_failed_after_the_bound`. SPEC 16.
3. **The peer's user on Unix (item 2).** Read with `interprocess`'s `peer_creds`, which uses `SO_PEERCRED` on Linux and `LOCAL_PEERCRED` on macOS and the BSDs; `getpeereid` is built on `LOCAL_PEERCRED` there, so this is the same source with no new `unsafe`. Order on the client: the folder, the socket entry (it must still exist and be this user's), then the peer's effective uid, which no file operation can change. An unreadable uid is a refusal. The daemon checks each client's uid and drops another user's before reading, with a log line. Evidence: the two item 2 unit tests, run against a user id the peer is not. SPEC 3, B-95.
4. **The pipe server's user on Windows (item 3).** `GetNamedPipeServerProcessId` (through `peer_creds`), `OpenProcess` with `PROCESS_QUERY_LIMITED_INFORMATION`, `OpenProcessToken` with `TOKEN_QUERY`, `GetTokenInformation(TokenUser)`, `EqualSid` against the current user's SID. Every handle is closed and every system buffer freed on every path. The daemon does not open its clients' tokens: the pipe's DACL already admits only the current user (stream E). The "another user" test is a client that expects the daemon to run as `SYSTEM` (`S-1-5-18`). Spike decision 1, B-95.
5. **Which socket folders must be private (item 4).** The mode rule (no group or other write, not a link) is for the fallback folder only, recognised by its name, `spatial-memory-` and 16 hex characters. The owner rule holds for any socket folder. A profile folder its group can write to is accepted: under the user-private-group umask 002 (Debian, Ubuntu) every folder is created that way, and the peer check still holds. Folders are read with `lstat`. Evidence: the four item 4 unit tests. SPEC 3, B-112.
6. **`project_unresolved` is transient (item 5).** A working folder that is gone may come back (a drive not yet mounted), so it is tried again within `MAX_ATTEMPTS` like any transient failure. Evidence: the two item 5 tests. SPEC 16, B-33.
7. **An unremovable spool file (item 6).** Its name is kept in memory for the life of the process, logged once, and the removal is retried quietly at each drain. A later daemon processes the file again; the store's dedupe (B-29) keeps its memories from being stored twice. Evidence: the two item 6 tests. Windows makes the removal fail with a handle opened without delete sharing; Unix replaces `spool/new` with a plain file while the capture is held (`ENOTDIR`, which holds for root too). SPEC 16, B-108.
8. **A capture that cannot be spooled (item 7).** Processed from memory, with a log line: the capture is in memory either way, and processing gives it the one durable place left, the store. Only a crash before the store loses it, as it would lose a capture still queued. The spool write is tried again before each retry, and a capture that fails for good is written to `spool/failed/` (that write may fail as well, and is logged). Evidence: `s16_a_capture_that_cannot_be_spooled_is_processed_from_memory`. SPEC 16, B-108.
9. **Strikes in the bulk queue (item 8).** A chunk that fails alone is struck only in a round in which another chunk was stored, so an embedder-wide failure strikes no chunk. A batch of one chunk is never struck: nothing shows whose fault it is. After a round with nothing stored the session is dropped and the wait (`bulk_error_backoff`, 5 s) doubles for each such round in a row, at most 6 times (320 s); a stored chunk resets it and any write wakes the queue. Trade-off: a poison chunk that is alone is never skipped and is tried at most every 320 s; once it shares a round with a chunk that is stored, it is struck. Evidence: the three item 8 tests. SPEC 5 "Runtime", e-report decision 33.
10. **Threads after `wait` (item 9).** Documented, not changed: a thread still running after `shutdown_grace` is not joined. The `daemon` subcommand exits the process. SPEC 3.
11. **What `content_truncated` covers (item 10).** The frozen `MemoryView` has no other flag, so it is also set when the view shed a field it had (title, tags, anchors). A cut that removes nothing (the memory had no such field) is not marked. Evidence: the two item 10 tests. SPEC 8 "Output", e-report decision 24. The `sm-core` documentation is asked to say so (section 8).
12. **The private bind folder's name (item 11).** Taken from std's `RandomState` (fresh SipHash keys seeded from the operating system) over the process id, the time and the attempt: `.` and 8 hex characters, which keep `/.<name>/s` no longer than `/daemon.sock`, so the `sun_path` limit holds. No new crate. Created with `DirBuilder::create`, which fails on any existing entry; a collision takes another name, up to 16. SPEC 3, B-95.
13. **The drift guard (item 12).** B-102 lists ten ops and says "11 structs": the ten parameter structs and the request envelope `Request`. Both are guarded: each test builds the struct with every field set, by a literal that names every field (a field added in `sm-core` stops it compiling), and checks that the keys serialised equal `allowed_args` or `REQUEST_KEYS`, and that dispatch's key check lets them through. SPEC 3, B-102.
14. **How spike decision 5 is observed (item 13).** Through a fake provider that numbers the sessions it opens and records each one's kind, its documents and whether it was dropped. The idle period is longer than `bulk_grace` (so the bulk session is dropped) and shorter than idle exit. Spike decision 5, B-95.
15. **How the B-96 and B-98 behaviour is observed (item 14).** B-96: a drain would store the capture, quarantine the corrupt file and remove the stale `tmp/` write; none of the three happens through 20 drain intervals. B-98: no file exists for the 1,024 queued captures, then exactly one, named with the 1,025th request id. SPEC 16, B-96, B-98.
16. **A title change re-embeds (item 15).**
    - A revise that changes the title rebuilds the memory's chunks from its stored content, in the same transaction as the title, and the chunks wait for the bulk queue.
    - A captured memory is chunked again with `capture.max_index_tokens`, the cap it was stored with, so what is indexed stays the same.
    - Like a content change, it needs the tokenizer, so it is refused with `daemon_loading` while the model loads and with the model's error once it has failed; a revise to the title the memory has, or of other fields only, is served. This adds "`revise` with a new title" to B-96's list (section 8).
    - `reembedded` is true for a title rebuild.
    - Every `revise` holds `Shared::revising`, so no other revise changes the content between the read and the rebuild.
    - Evidence: the three item 15 tests. SPEC 5 "Bulk", SPEC 8 `revise`, B-96.
17. **AT-16's reach (item 16).** All three recall modes and the prompt hook's `ambient_recall`, and `min_importance` as the only place importance acts. SPEC 11 AT-16, SPEC 16 "Importance is stored, not used", B-70.
18. **The profile folder (item 17).** Unix: created 0700 if missing (missing parents get the default mode); an existing folder of this user open to group or others is set to 0700, with one log line naming it and its old mode; a folder another user owns, or a path that is not a folder, is refused with `daemon_unavailable` before anything is written in it, with a next action. The folder is read through a link, so a profile moved elsewhere and linked keeps working. This runs in `start` before the lock file is created. Windows is left as it is: the folder is under `%LOCALAPPDATA%`, whose ACL admits only the user, SYSTEM and administrators, and is inherited. SPEC 3 rule 1, SPEC 9 "Security".

## 7. Deviations

1. **Two stream E tests changed, neither weakened.**
   - `s16_a_capture_that_can_never_be_stored_moves_to_failed_at_once` had two captures: a corrupt file, and one whose working folder is gone. Item 5 makes the second transient, so it moved to the new `s16_a_project_that_never_resolves_moves_to_failed_after_the_bound`, which asserts it reaches `spool/failed/` after the bound. In its place is a capture whose one candidate can never be stored (`invalid_input`). The test still asserts both move to `spool/failed/` at once.
   - `s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on` failed its chunk in rounds in which nothing else was stored. Item 8 counts no strike there, so each round now also carries a good chunk (six good chunks, nine memories). The test still asserts the chunk is skipped with a log line, stays pending, and that the thread survives the panic and embeds later chunks.
2. **`revise` with a new title is refused while the model loads.** B-96 lists only new content and `supersede_with`. The tokenizer is needed to rebuild the chunks (decision 16).
3. **`ReviseReply.reembedded` is true for a title-only change**, and **`MemoryView.content_truncated` is true for a view that shed a field.** Each `sm-core` documentation line says less (section 8). No type changed.
4. **macOS uses `LOCAL_PEERCRED`, not `getpeereid` by name** (decision 3). The value is the same.

## 8. Contract changes requested

1. **`sm-core`, `MemoryView.content_truncated`** (`protocol.rs` line 1439). Now: "`content` was cut to fit the token budget." Asked: "The view was cut to fit the token budget: `content` was shortened, or a field (title, tags, anchors) was shed." Reason: item 10; the frozen view has no other way to say it (decision 11).
2. **`sm-core`, `ReviseReply.reembedded`** (`protocol.rs` line 1541). Now: "Content changed, so chunks were rebuilt." Asked: "Content or title changed, so chunks were rebuilt." Reason: item 15 (decision 16).
3. **DECISIONS, B-96.** Add "`revise` with a new title" to the requests refused while the model loads, and replace its "Gap" line with `s16_the_spool_is_not_drained_while_the_model_has_failed` and `s8_a_new_title_waits_for_the_model_and_other_fields_do_not`.
4. **DECISIONS, B-98, B-95 and B-102.** Replace B-98's gap with `s16_a_capture_past_1024_queued_is_spilled_to_the_spool`, B-95's spike decision 5 gap with `spike5_the_query_session_is_opened_first_and_kept_for_life`, and mark B-102's phase 3 drift-guard item done (`s3_allowed_args_match_every_key_the_unchecked_params_serialise`, `s3_request_keys_match_every_key_a_request_serialises`).
5. **To package I, optional (`sm-store`).** A way to set a memory's chunks back to pending without re-chunking. A title change would then not need the tokenizer, and could be served while the model loads.
6. **To package J and the lead (`sm-cli`).** `client.rs` line 170 creates the profile folder with `create_dir_all` (the default mode) before it starts the daemon. The daemon then sets it to 0700 and logs one line on every new profile's first start. Creating it with mode 0700 in `sm-cli` would avoid the line and the moment in which it is open. `sm-daemon`'s own `lock.rs` and `log.rs` also call `create_dir_all`, but only after `private_profile` has run.
7. **Relayed from package J's message to this package:** a structured failure kind from `connect`, and the model state in `SharingContext`. Both would change the public API this package kept stable, so neither was done here; J reports them itself.

## 9. Unfinished or unverified

- **macOS.** The peer credential check (`LOCAL_PEERCRED`) and the private bind folder are unverified there: no macOS machine. The Linux tests call the private bind path directly.
- **Windows pipe check, edge cases.** Not tested: an elevated daemon with an unelevated client (the token user is the same, so it should pass), a server process that exits after connecting (the process id could be reused; `OpenProcess` then opens another process), and a client denied `PROCESS_QUERY_LIMITED_INFORMATION` (a refusal).
- **Job-level failures.** A capture whose pipeline panics for the same text every time sends each later capture of that session to `spool/failed/` at the bound too, because the session never moves on (decision 2). No test makes the pipeline panic for a fixed text.
- **A lone poison chunk** is never skipped while it is alone (decision 9). This is by design; it is covered by `s5_a_lone_chunk_that_fails_is_never_struck`, but its cost over a long run (one failed embedding at most every 320 s) was not measured.
- **`reindex` re-chunks without the capture cap.** Seen while fixing item 15, not changed: a captured memory over `capture.max_index_tokens` is indexed whole after a `reindex`. It is outside this package's items; it is for the lead.
- **No mutation checks.** Unlike stream E's report, no fix was removed to show its test then fails.

## 10. Licences

No crate was added: `Cargo.lock` and every `Cargo.toml` are unchanged against `93044bb`.
