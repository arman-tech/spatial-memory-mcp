# Package K report: end to end, with the real model, and RESULTS.md

Branch `arman-tech/stream-k-results`, from `v3` at `25c627e`. The commits, in order:

| Commit | What |
|---|---|
| `8746bec` | `sm-daemon`: item 9, the idle-exit line says "1 minute" for one minute. |
| `6a4fd3c` | `sm-daemon`: item 7, `reindex` keeps a captured memory's index cap (B-118). |
| `aa64d8d` | `sm-core`, `sm-embed`, `sm-daemon`: item 8, the `doctor_numbers` operation (B-125). This is the one `sm-core` edit. |
| `7e64021` | `sm-cli`: item 8, `doctor` prints both embedder numbers from the daemon. |
| `d16f5b9` | `sm-cli`: items 1 to 5, the hooks end to end through the binary (AT-3, AT-13, AT-14, AT-15, AT-17). |
| `177a104` | `sm-cli`: items 6 and 10, `remember`, `recall` and `share` end to end through `spatial-memory mcp` (AT-16, AT-9). |
| `97a2742` | `sm-cli`: Part 2, `examples/budgets`, the real-model measurements. |
| `b2060fa` | `docs/v3/RESULTS.md` (Part 4, with Part 3's CI times). |

This report is the commit after them. Every gate result in section 3 is for `97a2742`, the last code commit. The two commits after it change documents only.

## 1. Done when

**Yes.** Every condition is met:

| Condition | Met | Evidence |
|---|---|---|
| Every item under "Build" is tested or measured as it says | Yes | Part 1, items 1 to 10: section 4. Part 2, cases 1 to 16: `RESULTS.md` sections 2 to 4. Part 3: `RESULTS.md` section 6. Part 4: `RESULTS.md`. |
| `RESULTS.md` has a row for every SPEC 11 budget and every acceptance test, AT-1 to AT-17 | Yes | `RESULTS.md` section 2 (budgets, one row per budget and condition) and section 3 (AT-1 to AT-17). |
| Both gates pass, with the full workspace on Linux | Yes | Section 3, on `97a2742`. Windows: fmt, clippy and test all exit 0. Linux in Docker: `FMT_EXIT=0`, `CLIPPY_EXIT=0`, `TEST_EXIT=0`. |
| `k-report.md` is committed | Yes | The commit after the `RESULTS.md` commit. |

Two budgets are missed, and they are results, not blockers (section 5): idle memory after `recall` at 100,000 chunks, and the raw thread count on Windows during the import.

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-core/src/protocol.rs` | B-125: `Op::DoctorNumbers`, `Reply::DoctorNumbers(DoctorNumbersReply)`, and the reply struct. `s3_op_tags_are_stable` covers the tag and a round trip. The exact diff is in section 8. |
| `crates/sm-embed/src/doctor.rs` | `time_sessions` times the fixed sample on a bulk and a query session that are already open. `doctor_numbers` opens the two sessions and calls it, so its tests pass unchanged. |
| `crates/sm-daemon/src/admin.rs` | `admin::doctor_numbers`: runs `time_sessions` on the queue's own bulk session through `queue::with_bulk` and on the ready daemon's query session. `reindex` takes each memory's index cap from `Shared::stored_index_cap`. |
| `crates/sm-daemon/src/dispatch.rs` | `doctor_numbers` takes no arguments (B-102 guard), and its two B-102 tests cover the new op. |
| `crates/sm-daemon/src/state.rs` | `Shared::stored_index_cap`: `capture.max_index_tokens` for a memory whose source is `Capture`, none for any other. |
| `crates/sm-daemon/src/tools/edit.rs` | The title-only `revise` arm calls `stored_index_cap` instead of its inline rule. |
| `crates/sm-daemon/src/maintenance.rs` | `idle_exit_line`: "no request for 1 minute; exiting", or "N minutes" for any other N. |
| `crates/sm-daemon/tests/socket.rs` | The four `s5_rule7_doctor_numbers_` tests and `s3_the_idle_exit_line_says_minute_for_one_and_minutes_otherwise`. |
| `crates/sm-daemon/tests/hooks.rs` | `s16_reindex_keeps_a_captured_memorys_index_cap`. |
| `crates/sm-cli/src/doctor.rs` | The `embedder` line. It asks the daemon for `doctor_numbers` after the status checks, and polls every 50 ms while the daemon loads, inside the connection's 10 s window. "Not measured" is information, not a failure. `doctor::exit_code` gives the command's exit code. Two unit tests. |
| `crates/sm-cli/src/cli.rs` | `doctor` exits with `doctor::exit_code`. |
| `crates/sm-cli/tests/doctor.rs` | Three `s5_rule7_` tests of `doctor::checks`: against the production daemon in this process with the fake model, and against two mocks that stay loading. |
| `crates/sm-cli/tests/common/e2e.rs` | The end-to-end harness. It runs the real daemon with the fake model, in this process or in a child (this test binary run again on `at3_child_daemon_serves_only_when_asked`). A model that records its queries, or blocks one query until the process is killed. Direct requests that read what the binary's runs left in the store. |
| `crates/sm-cli/tests/e2e_hooks.rs` | Nine tests: AT-3, AT-13, AT-14, AT-15 and AT-17 through `spatial-memory hook`. |
| `crates/sm-cli/tests/e2e_mcp.rs` | Two tests: AT-16 and AT-9 through `spatial-memory mcp`. |
| `crates/sm-cli/examples/budgets.rs` | The measurement program: `cargo run --release -p sm-cli --example budgets -- <case>... | all`, 15 cases. Never run by `cargo test`. |
| `crates/sm-cli/examples/budgets/cases.rs` | The cases. |
| `crates/sm-cli/examples/budgets/data.rs` | What the cases store and ask: the spatial fixture's 320 texts, 128-token prompts made from them, generated version 1 exports, and seeded unit vectors. |
| `crates/sm-cli/examples/budgets/harness.rs` | The release binary in a private temporary home, its real daemon, hook and command runs, and the figures. |
| `crates/sm-cli/examples/budgets/mcp.rs` | `spatial-memory mcp` with an `rmcp` client attached, as an agent runs it. |
| `crates/sm-cli/examples/budgets/procs.rs` | Readings of the daemon from outside: memory and threads (B-58, B-114). |
| `crates/sm-cli/Cargo.toml` | Dev-dependencies: `sm-store`, and a `cfg(windows)` section that adds three `windows-sys` features for `procs.rs`. |
| `Cargo.lock` | Gains `sm-store` under `sm-cli`, an internal crate. |
| `docs/v3/RESULTS.md` | Part 4. |
| `docs/v3/streams/k-report.md` | This report. |

## 3. Test commands and exact results

Both gates ran on `97a2742`, the commands of RULES sections 5 and 6 on the full workspace. After each, the end-to-end hook tests ran again with `--nocapture` to print their timing lines.

### Windows 11 (PowerShell, debug)

`cargo fmt --check`: exit 0. `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`: exit 0. `cargo test --workspace --all-features --locked --no-fail-fast`: exit 0. The test binaries, in order:

```
sm-capture         test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
sm-cli             test result: ok. 56 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
sm-cli main.rs     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart          test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 23.14s
daemon             test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.68s
doctor             test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.99s
e2e_hooks          test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.61s
e2e_mcp            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.58s
hook_budgets       test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 70.31s
hook_fixtures      test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.36s
mcp_proxy          test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
ops                test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.14s
plugin             test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.98s
setup              test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
sharing            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.53s
sm-core            test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm-daemon          test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.38s
acceptance         test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.75s
hooks              test result: ok. 30 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.38s
socket             test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.96s
sm-embed           test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
real_model         test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.92s
sm-share           test result: ok. 92 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 27.53s
sm-spatial         test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
at6_determinism    test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.04s
at7_quality        test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s
s7_pipeline        test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.84s
sm-store           test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.88s
doc-tests (8)      test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s  (each)
```

641 tests. Per crate: `sm-capture` 34, `sm-cli` 131, `sm-core` 96, `sm-daemon` 119, `sm-embed` 55, `sm-share` 92, `sm-spatial` 47, `sm-store` 67. Against package J's run, `sm-cli` gains 2 unit tests and the `doctor` (3), `e2e_hooks` (9) and `e2e_mcp` (2) binaries; `sm-daemon` gains 1 in `hooks` and 5 in `socket`. On Windows `real_model` found the model in the per-user cache and ran it. Afterwards no `spatial-memory.exe` was running.

`cargo test -p sm-cli --all-features --locked --test e2e_hooks -- --nocapture`: exit 0, `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.46s`.

### Linux (Docker, `rust:1.95-trixie`, debug, as root)

The brief's command, with the target in `sm-v3-target-k`: `FMT_EXIT=0`, `CLIPPY_EXIT=0`, and `cargo test --workspace --all-features --locked --no-fail-fast` gave `TEST_EXIT=0`. The test binaries, in order:

```
sm-capture         test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
sm-cli             test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm-cli main.rs     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart          test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.07s
daemon             test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.34s
doctor             test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.99s
e2e_hooks          test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.55s
e2e_mcp            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.57s
hook_budgets       test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 61.28s
hook_fixtures      test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.58s
mcp_proxy          test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
ops                test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
plugin             test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
setup              test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sharing            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.28s
sm-core            test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
sm-daemon          test result: ok. 28 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.19s
acceptance         test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.60s
hooks              test result: ok. 30 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.27s
socket             test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.68s
sm-embed           test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
real_model         test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
sm-share           test result: ok. 92 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 44.38s
sm-spatial         test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.62s
at6_determinism    test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.01s
at7_quality        test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s
s7_pipeline        test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.71s
sm-store           test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.66s
doc-tests (8)      test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s  (each)
```

651 tests. Per crate: `sm-capture` 34, `sm-cli` 130, `sm-core` 97, `sm-daemon` 129, `sm-embed` 55, `sm-share` 92, `sm-spatial` 47, `sm-store` 67. The counts differ from Windows only where a test is for one platform, as in package J's report. `real_model` printed ``real_model: SKIPPED: the model folder /root/.cache/spatial-memory/_models/embeddinggemma-2/daa72c51243991dfcaf9f9137d2c573d8f7790c0 is absent; set SPATIAL_MEMORY_MODEL_DIR or run `spatial-memory setup` ``, because the gate container has no model.

`cargo test -p sm-cli --all-features --locked --test e2e_hooks -- --nocapture`: `E2E_EXIT=0`, `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.22s`.

### The real-model test (release, once at the end)

`cargo test --release -p sm-embed --test real_model`:
- Windows: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.28s`. Token fingerprint `434a9df05b5b5fc5`, equal to the reference.
- Linux in Docker, with the model in the `sm-v3-model` volume and `SPATIAL_MEMORY_MODEL_DIR=/model`: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.32s`. The same fingerprint.

### The hook process's own budget (release, case 15)

`cargo test --release -p sm-cli --test hook_budgets s11_hook_process_warm_100_runs_inside_the_hook_budget -- --nocapture`:
- Windows: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 7.55s`. All 800 runs: median 8.6, p95 10.2, worst 15.5 ms.
- Linux in Docker, all processors, two runs: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out; finished in 1.34s` and `... finished in 1.35s`. Both runs: median 1.5, p95 1.7, worst 2.2 ms.
- Linux in Docker, two processors: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out; finished in 1.25s`. Median 1.4, p95 1.5, worst 2.6 ms.

## 4. Acceptance tests

All pass on Windows and on Linux in Docker, in both gates.

| Item | Tests | Figures |
|---|---|---|
| 1. AT-3, the daemon killed | `at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_is_killed_mid_request` (20 runs), `at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_died_before_it_connected` (5 runs), and the child entry `at3_child_daemon_serves_only_when_asked` | Killed mid-request, the hook's wall time: Windows median 23.4, p95 24.5, **worst 26.5 ms**; Linux 21.1, 21.3, **worst 21.5 ms**. From the kill to the hook's exit: worst 6.0 ms on Windows and 1.3 ms on Linux. Dead before connect: worst 13.4 ms on Windows and 3.0 ms on Linux. Every run exits 0, prints nothing and journals `gate=error`. With the real daemon (`kill` case): worst 74.8 ms on Windows and 59.5 / 67.1 ms on Linux. |
| 2. AT-13 | `at13_capture_hooks_through_the_binary_store_strong_text_redacted_and_weak_text_not_at_all` | The Edit, Write and Bash fixture payloads: strong text stored active with capture provenance, the trivial output stores nothing, and the fake AWS key is stored as `[REDACTED:aws_key]`. |
| 3. AT-14 | `at14_pre_compact_and_stop_through_the_binary_store_only_what_post_tool_use_missed` | PreCompact and Stop store only what PostToolUse missed. PreCompact prints nothing. Stop prints its continue response. `stop_hook_active: true` stores and journals nothing. |
| 4. AT-15 | `at15_capture_hooks_spool_inside_200_ms_with_no_daemon_and_a_started_daemon_stores_the_same` | With no daemon, every capture hook of both clients exits 0 with its payload in `spool/new/`. A daemon started afterwards stores the same memories and empties the spool. Worst of 5 runs per hook, Windows / Linux, in ms: `cc_post_tool_use_edit_decision` 10.2 / 2.0; `cc_pre_compact_runs_pipeline` 9.9 / 1.9; `cc_stop_runs_pipeline` 14.9 / 2.1; `cursor_postToolUse_edit_decision` 17.1 / 1.9; `cursor_preCompact_status_trigger` 10.8 / 1.8; `cursor_stop_status_trigger` 10.8 / 2.0. The budget is 200 ms. |
| 5. AT-17 | `at17_ambient_recall_off_makes_no_contact_with_the_daemon`, `at17_the_block_through_the_binary_is_version_2s_format_with_ids_and_the_footer`, `at17_a_capture_over_the_index_cap_is_stored_whole_indexed_to_the_cap_and_journaled` | "Off" is shown from the daemon's side: a relay in front of the daemon counts no connection. The block matches version 2's format, with ids and the footer, and its scores match the journal's. The over-cap capture is stored whole, indexed to `capture.max_index_tokens`, and journaled as a cap hit with the same request id. |
| 6. AT-16 | `at16_importance_does_not_change_recall_order_through_mcp` | Two memories with the same words, importance 0.2 and 0.9, and in another folder 0.9 and 0.2. `recall` (auto and vector) returns them in the same order, the order of their ids, both ways round. |
| 7. `reindex` cap (B-118) | `s16_reindex_keeps_a_captured_memorys_index_cap` | A capture and a remembered memory of the same length, both longer than the cap. After `reindex`, the chunk count, the documents embedded and the indexed words are unchanged; the capture is capped and the other is not. |
| 8. `doctor_numbers` (B-125) | `sm-daemon`: `s5_rule7_doctor_numbers_counts_match_the_settings`, `s5_rule7_doctor_numbers_uses_the_queues_bulk_session_and_opens_no_third`, `s5_rule7_doctor_numbers_is_daemon_loading_while_the_model_loads`, `s5_rule7_doctor_numbers_gives_the_models_error_after_a_failed_load`; `dispatch.rs`'s `s3_allowed_args_match_every_key_the_unchecked_params_serialise` and `s3_request_keys_match_every_key_a_request_serialises` cover the op. `sm-cli`: `s5_rule7_numbers_line_gives_both_numbers_with_threads_and_batch`, `s5_rule7_not_measured_is_not_a_failure_and_says_run_doctor_again` (unit); `s5_rule7_doctor_prints_both_numbers_from_an_in_process_daemon`, `s5_rule7_doctor_says_not_measured_while_the_daemon_stays_loading`, `s5_rule7_doctor_exits_0_when_only_the_numbers_were_not_measured`. `sm-core`: `s3_op_tags_are_stable`. `sm-embed`'s `s5_rule7_` tests pass unchanged. | With the real model: `RESULTS.md` section 4, "`doctor`'s numbers". |
| 9. The idle line | `s3_the_idle_exit_line_says_minute_for_one_and_minutes_otherwise` | An idle exit at 1 and at 2 minutes, with a 150 ms idle unit; each writes one line. |
| 10. AT-9 | `at9_remember_redacts_and_share_writes_no_credential_through_mcp` | The 33 credential cases of `redaction_cases.json`, remembered through the binary, list the fixture's kinds. The empty input is refused with `invalid_input`. Inputs that redact to the same text are one memory, so there are 29 memories (K-6). The shared file, read from disk, holds version 3's markers and no credential. A credential in a tag or an anchor makes `share` refuse with `secret_detected`, "front matter" in the message and a next action, and no file is written under `.spatial-memory/shared`. |

The real-model acceptance results (AT-1, AT-2, AT-4, AT-7, AT-8 and AT-10) are in `RESULTS.md` section 3.

## 5. Measured budgets

`RESULTS.md` has every figure: the machine (section 1), the budgets (section 2), the acceptance tests (section 3), the other figures (section 4), what was not measured (section 5) and the CI times (section 6).

**Every time budget is met**, on Windows and in both Docker runs. The p95 of the warm prompt hook is 91.6 to 170.4 ms against 300 ms in every case. The hard deadline is met on every run, with a worst of 74.8 ms against 1,500 ms. The p95 of `recall` at 100,000 chunks is at most 109.0 ms at project scope (budget 250 ms) and at most 66.7 ms over all projects (budget 350 ms).

**Missed budgets.**

1. **Daemon memory, idle, after `recall` at 100,000 chunks: budget 1 GB (953.7 MiB) of working set.** Missed in 11 of 12 readings. Working set / private memory in MiB:

   | Reading | Windows | Linux in Docker, all processors | Linux in Docker, two processors |
   |---|---|---|---|
   | After 300 project-scope queries, 60 projects | 686.2 / 649.9 (met) | **1,052.9** / 640.3 | **1,055.7** / 641.4 |
   | After 300 more all-projects queries, 60 projects | **1,058.5** / 651.8 | **1,490.6** / 640.3 | **1,493.0** / 641.4 |
   | After 300 project-scope queries, one project | **1,077.4** / 656.9 | **1,497.2** / 647.0 | **1,499.5** / 647.9 |
   | After 300 more all-projects queries, one project | **1,400.2** / 653.1 | **1,497.4** / 647.3 | **1,499.7** / 647.8 |

   Private memory stays between 640 and 657 MiB. The rest is the mapped database file (B-114). On Linux, `RssFile` rose from 27 MiB at ready to 268 MiB once the daemon had opened the 750 MiB database, then 413 MiB and 850 MiB after the two blocks of queries over 60 projects. As B-114 says, this is recorded as missed. `mmap_size` and the search were not changed. Section 8 asks the lead to decide.

2. **Threads during the import (AT-3), Windows, raw count: 27 against a budget of 24, over by 3.** The daemon's own threads (`sm-*` and its connection threads) peaked at 20 against 20. The bulk session added exactly 12 (11 `sm-embed-pool` threads and the `sm-embed-bulk` calling thread). The 3 extra threads are two Windows thread-pool workers that start in `ntdll.dll` and one unnamed thread whose start module the snapshot could not name. On Linux the raw count stayed within (22 against 23).

**Part 3, CI build and test times** (`RESULTS.md` section 6, on `97a2742`, every step exit 0 and all 36 test binaries ok). Each job's wall time in seconds:

| Job | Cold, 4 processors | Cold, 2 processors | Warm, 4 processors | Warm, 2 processors |
|---|---|---|---|---|
| `lint` (Linux in Docker) | 75.6 | 134.0 | 15.6 | 7.5 |
| `test-linux` (Linux in Docker) | 410.3 | 539.5 | 293.8 | 386.6 |
| `test-windows` (native) | 502.7 | 788.3 | 333.7 | 347.0 |
| Wait for a push's result | 8.4 min | 13.1 min | 5.6 min | 6.4 min |

A warm push spends 13 to 22 s building and 276 to 362 s running the tests. `hook_budgets`, `acceptance`, `sm-store` and `sm-share` make up about four fifths of the running. These figures are lower bounds; CI_PROPOSAL section 4 doubles them.

## 6. Decisions

Numbered K-n here so the lead can turn each into a B-n entry.

**K-1. `doctor` asks for the numbers after its status checks, and "not measured" is not a failure.**
- **What.** `doctor` sends `doctor_numbers` once the `state` check has run. While the daemon answers `daemon_loading`, it asks again every 50 ms until the connection's 10 s window closes. Then it prints "embedder: not measured" with the next action "Run `spatial-memory doctor` again in a few seconds." The exit code is what the other checks give (`doctor::exit_code`). After a failed load there is no embedder line, because `state` and `model` already fail.
- **Why.** Stream K item 8. SPEC 8 gives every error a next action, and a daemon that is still loading is not broken.
- **Evidence.** `s5_rule7_doctor_says_not_measured_while_the_daemon_stays_loading`, `s5_rule7_doctor_exits_0_when_only_the_numbers_were_not_measured`.
- **Touches.** SPEC 5 rule 7, SPEC 8, CONTRACTS 3.1, B-125.

**K-2. The end-to-end daemon runs with test options.**
- **What.** `e2e::options` gives the daemon in `tests/common/e2e.rs`, in this process or in a child, these options: lock and bind waits of 300 ms, a retry every 10 ms, 8 physical cores, a 5 s shutdown grace, and the spool drained every second instead of every 60 s. They are `DaemonOptions` fields, as the other daemon tests use them. No default changes.
- **Why.** Short waits keep each test's waits bounded. The drain at start (`maintenance.rs`) is unchanged, so AT-15's "a daemon started afterwards" is the production path. Build rule "never tune a default".
- **Touches.** SPEC 16 (the spool), SPEC 15.

**K-3. AT-3's killed daemon is this test binary run again.**
- **What.** The child runs `at3_child_daemon_serves_only_when_asked` with `SM_E2E_CHILD_HOME` set. Its fake model blocks a query that holds a probe word, after writing a marker file (`SM_E2E_CHILD_MARKER`). The test kills the child once the marker exists, so the hook is waiting on that request. Without the variable the child test returns at once. Every child is killed and reaped, and the test waits for its lock to be free.
- **Why.** Stream K Part 1: the daemon must be a separate process that can be killed. The marker makes "mid-request" exact instead of timed.
- **Touches.** SPEC 11 AT-3, B-80.

**K-4. AT-15's "no daemon" holds the profile lock.**
- **What.** A child daemon runs and is killed first, so the profile is one where a daemon ran. The test then holds the daemon lock while the capture hooks run. A hook's auto-start finds the lock held and spawns nothing. The lock is released before the daemon that drains the spool starts.
- **Why.** Without the lock, each capture hook's auto-start would spawn a real daemon of the binary, which would load the (empty) model folder, and a test could leave it running (J-11).
- **Touches.** SPEC 3 (auto-start), SPEC 16, J-11.

**K-5. AT-17's "no contact" is counted by a relay.**
- **What.** A relay sits on the endpoint the hook uses and forwards to the daemon. The test checks that its connection count does not move during the hook.
- **Why.** Item 5 asks for the daemon's side, not only the hook's output.
- **Touches.** SPEC 11 AT-17.

**K-6. AT-9 through the binary: 33 cases give 29 memories.**
- **What.** `redaction_cases.json` has 41 cases: 33 with a fake credential, 7 without one, and the empty input. The test remembers each credential case. Inputs that redact to the same text are one memory: `remember` returns the outcome `duplicate` with the first one's id. The test checks each duplicate names an id it already has, and that memories plus duplicates equal the cases. The run gives 29 memories and 4 duplicates. The empty input is refused with `invalid_input` and a next action. The 7 cases without a credential have nothing to redact and are not remembered.
- **Why.** This is SPEC 4's duplicate rule applied after redaction. The test accepts it rather than expecting 33 memories.
- **Touches.** SPEC 4, SPEC 9 "Security", SPEC 11 AT-9.

**K-7. The bulk session's opening time is a difference of medians.**
- **What.** The time to open the bulk session is the median of 3 `doctor_numbers` calls with the session closed, less the median of 3 with it open.
- **Why.** No log line marks the opening, and `doctor_numbers` uses the queue's own session (B-125), so it opens the session when it is closed.
- **Evidence.** `RESULTS.md` section 4: 365.9 ms on Windows, 241.2 ms on Linux in Docker.
- **Touches.** B-97, B-125.

**K-8. The throttling diagnostic is applied from outside, and not shipped.**
- **What.** On Windows, the `hook` case measures the daemon a second time after it calls `SetProcessInformation(ProcessPowerThrottling)` on the daemon's process handle to turn EcoQoS off. The figures are labelled as a diagnostic. The shipped figures are the throttled ones.
- **Why.** The query embedding took 85 ms in the daemon and 47 ms in B-56's test process. The diagnostic tests whether throttling explains the difference (it gives 47.6 ms). Turning it off in the daemon would change shipped behaviour, which is the lead's decision (section 8).
- **Touches.** SPEC 5, SPEC 11, B-56.

**K-9. Thread readings and what counts as the daemon's own.**
- **What.** Threads are sampled every 50 ms from outside. On Windows, a thread's name comes from `GetThreadDescription`, and an unnamed thread's start module comes from its start address. The daemon's own threads are those named `sm-*`, which include the connection threads. The raw count is reported beside them.
- **Why.** Item 4 asks for each counted thread to be explained, and Windows adds pool threads the daemon did not create.
- **Touches.** SPEC 5 (the thread budget), B-58.

**K-10. Two processors means `--cpuset-cpus 0-1`, and three more cases ran with them.**
- **What.** Case 15 lists cases 1 to 4, 7 and 8. `kill`, `recall100k` and `doctor` also ran with two processors.
- **Why.** Item 16 asks for `doctor` "in case 15's container both ways". `recall100k` and `kill` are budget cases a two-core runner would also face.
- **Touches.** B-52.

**K-11. How Part 3 runs `ci.yml`'s jobs.**
- **What.**
  - Linux: each job (`lint`: `cargo fmt --check` then `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`; `test`: `cargo test --workspace --all-features --locked`) runs in its own `docker run --cpus N`, with its own new registry volume and target volume, both kept for the warm run. Cargo's job count and the test harness's thread count come from the container's processor quota (`cpu.max`). A one-line program in the same image printed `available_parallelism()` as `Ok(4)` under `--cpus 4` and `Ok(2)` under `--cpus 2`, while `nproc` said 32.
  - Windows: `cargo test -j N --workspace --all-features --locked`, with `RUST_TEST_THREADS=N`, and a new `CARGO_HOME`, `CARGO_TARGET_DIR` and ONNX Runtime cache outside the repository. `SPATIAL_MEMORY_MODEL_DIR` points at an empty folder, so `real_model` skips itself, as it would on a runner.
  - The warm change is one comment line appended to `crates/sm-cli/src/lib.rs`, which rebuilds the `sm-cli` library, the binary and all its test binaries. It was removed afterwards.
- **Why.** CI_PROPOSAL section 4's method. A comment is the smallest change that still rebuilds `sm-cli` and everything that depends on it.
- **Touches.** CI_PROPOSAL section 4.

## 7. Deviations

1. **The `sm-core` edit (B-125, commit `aa64d8d`).** It is the one edit the brief allows, made as B-125 gives it, with one line B-125 does not name: `Op::validate` gains `| Op::DoctorNumbers` in its `Ok(())` arm. The match is exhaustive, so the crate does not compile without it. Section 8 has the exact diff.
2. **The `reindex` fix (B-118, commit `6a4fd3c`)**, in `sm-daemon`'s `state.rs`, `admin.rs` and `tools/edit.rs`, as item 7 gives it. A revise with new content stays uncapped.
3. **The idle-line fix (commit `8746bec`)**, in `sm-daemon/src/maintenance.rs`. No test or user document quoted the old line. Package J's report quotes it, as a log from a manual run, and is left as history.
4. **`sm-cli`'s dev-dependencies.** `sm-store` is one the brief allows. The `cfg(windows)` dev-dependencies section adds three `windows-sys` features (`Wdk_System_Threading`, `Win32_System_Diagnostics_ToolHelp`, `Win32_System_ProcessStatus`) for `procs.rs`. The root pin is unchanged and no crate is added; the lock file gains only `sm-store` under `sm-cli`.
5. **AT-10 through the `daemon` subcommand.** Item 14 says the daemon "does not become ready". It does not: it answers `failed` with `model_integrity` after 0.5 s, and `daemon.log` and `doctor` both say so with the next action. The daemon stays up to report the cause rather than exiting, as B-36 and B-96 decide.
6. **The Windows thread sampler is slower than 50 ms.** A Toolhelp snapshot takes longer than the interval, so the Windows sampler took 322 samples in the 50.2 s import, against 1,955 in 98.1 s on Linux. A shorter peak could have been missed on Windows.
7. **Part 3's processor limits are not dedicated cores.** `--cpus N` is a quota of processor time spread over the virtual machine's 32 processors. On Windows, `-j N` and `RUST_TEST_THREADS=N` limit Cargo's jobs and the test threads but not the processors a process may use, so the Windows warm runs at `-j 4` and `-j 2` differ by only 5 s of running. Both are lower bounds for a hosted runner (`RESULTS.md` section 6).

## 8. Contract changes requested

To the lead:

1. **CONTRACTS sections 5 and 6: `doctor_numbers` (B-125).**
   - Section 5 (the operations): `doctor_numbers`, no arguments, `{"op":"doctor_numbers"}`, always valid. `daemon_loading` while the model loads, and the model's error after a failed load. It times the queue's own bulk session (the queue waits meanwhile) and the query session, and opens no third session.
   - Section 6 (the replies): `Reply::DoctorNumbers`, kind `"doctor_numbers"`, with `DoctorNumbersReply`.
   - The exact `sm-core` diff is at the end of this section.
2. **Windows power throttling (EcoQoS).** Windows throttles the background daemon within 2 to 4 s of its start. Its query embedding then takes about 85 ms instead of 47.6 ms, and the prompt hook's p50 is 100.1 ms instead of 61.4 ms. The budget is met either way. Decide whether the daemon opts out with `SetProcessInformation(ProcessPowerThrottling)` (K-8; `RESULTS.md` section 2). Opting out would trade battery for latency on laptops.
3. **`num_cpus::get_physical()` ignores the processor set.** Under `--cpuset-cpus 0-1` the daemon saw 16 physical cores and chose 4 query and 12 bulk threads on 2 processors, which are the two hardware threads of one core. SPEC 5 says "Never run more threads than cores". Decide whether the thread budget should also be capped by the processors the process may use (`std::thread::available_parallelism`).
4. **The idle memory budget and mapped file pages (B-114).** Decide whether the 1 GB idle budget counts the database's mapped pages, which reach 850 MiB after `recall` at 100,000 chunks while private memory stays near 650 MiB (section 5). Until then it is recorded as missed.
5. **The raw thread count on Windows.** Decide whether the thread budget counts Windows thread-pool threads the daemon did not create (section 5).
6. **`TRACE.md`** needs the new tests: the `at3_`, `at9_`, `at13_`, `at14_`, `at15_`, `at16_` and `at17_` tests of section 4; the `s5_rule7_` tests in `sm-daemon` and `sm-cli`; `s16_reindex_keeps_a_captured_memorys_index_cap`; and `s3_the_idle_exit_line_says_minute_for_one_and_minutes_otherwise`.
7. **`DECISIONS.md`**: K-1 to K-11 as B-n entries. **`PROGRESS.md`**: package K done. **SPEC.md's spike notes**: the hook process with ONNX Runtime linked is 1.7 ms at p95 on Linux in Docker (B-124 gives 10.4 ms on Windows only).
8. **CI_PROPOSAL section 4**: the figures in `RESULTS.md` section 6.

**The B-125 diff.** `git show aa64d8d -- crates/sm-core/src/protocol.rs`, without its two test hunks:

```diff
@@ -314,6 +314,8 @@ pub enum Op {
     ProjectRename(ProjectRenameParams),
     ProjectDelete(ProjectDeleteParams),
     DoctorEval(DoctorEvalParams),
+    /// The embedder's numbers for `doctor` (SPEC 5 rule 7; B-125).
+    DoctorNumbers,
 }
 
 impl Op {
@@ -344,13 +346,14 @@ impl Op {
             Op::ProjectRename(_) => "project_rename",
             Op::ProjectDelete(_) => "project_delete",
             Op::DoctorEval(_) => "doctor_eval",
+            Op::DoctorNumbers => "doctor_numbers",
         }
     }
 
     /// Checks the parameters that do not depend on the store.
     pub fn validate(&self) -> Result<()> {
         match self {
-            Op::Hello | Op::Status(_) | Op::Sync(_) | Op::Reindex(_) => Ok(()),
+            Op::Hello | Op::Status(_) | Op::Sync(_) | Op::Reindex(_) | Op::DoctorNumbers => Ok(()),
             Op::Shutdown(_) => Ok(()),
             Op::Remember(p) => p.validate(),
             Op::Recall(p) => p.validate(),
@@ -1380,6 +1383,7 @@ pub enum Reply {
     /// `project_rename` and `project_delete`.
     Project(ProjectReply),
     Eval(Box<EvalReport>),
+    DoctorNumbers(DoctorNumbersReply),
 }
 
 fn is_false(b: &bool) -> bool {
@@ -1839,6 +1843,26 @@ pub struct ExportReport {
     pub links: u64,
 }
 
+/// The embedder's numbers, timed by the daemon on its own sessions (SPEC 5
+/// rule 7; B-56, B-125): bulk throughput on the bulk session and the
+/// median query latency on the query session.
+#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
+pub struct DoctorNumbersReply {
+    pub physical_cores: u32,
+    pub query_threads: u32,
+    pub bulk_threads: u32,
+    pub batch_size: u32,
+    /// Chunks in the timed bulk sample.
+    pub sample_chunks: u32,
+    /// Tokens in each timed query.
+    pub query_tokens: u32,
+    /// Timed query runs the median is taken over.
+    pub query_runs: u32,
+    pub bulk_elapsed_ms: f64,
+    pub bulk_chunks_per_sec: f64,
+    pub query_median_ms: f64,
+}
+
 #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
 pub struct ReindexReport {
     /// The database copy made first (SPEC section 4, Housekeeping).
```

The two test hunks extend `s3_op_tags_are_stable`. It adds `Op::DoctorNumbers` to its list of ops, asserts that the op serialises as `{"op": "doctor_numbers"}`, and round-trips a `Reply::DoctorNumbers` through `Response::ok`, checking the kind `"doctor_numbers"` and `data.query_tokens`.

## 9. Unfinished or unverified

**Unfinished.** Nothing in the brief.

**Unverified.**
- **The fallback model, Arctic Embed M (D-5, B-57).** Never loaded. Package K downloads no model.
- **Native Linux.** Every Linux figure is Linux in Docker in the WSL2 virtual machine.
- **macOS.** No machine.
- **A cold file cache.** Every cold start ran with the model files in the file cache. Clearing it on Windows needs administrator rights.
- **A real version 1 store (SPEC 12)** and **the evaluation on real memories**. Both need the owner's data.
- **Bulk throughput on Windows with throttling off.** Only the query side was measured that way.
- **A two-core runner's thread counts.** See section 8, request 3. The two-processor figures measure 16 threads on 2 processors.
- **The three unexplained Windows threads.** The snapshot names two as Windows thread-pool workers and cannot name the start module of the third.
- **Whether throttling explains the real-model test's split** on Windows (46.6 ms early, 85.0 ms late). It fits, but the test does not read the throttling state.
- **Idle exit with the real model.** It is tested with the fake embedder only.
- **Claude Code and Cursor themselves.** The binary was run with their hook payloads and JSON-RPC, not by the clients.

**Docker volumes and folders.**

| Created by package K | Kept or removed |
|---|---|
| `sm-v3-model`, the model for case 15 (filled once from the per-user cache, as case 15 says) | Kept, for reruns of `RESULTS.md` in Docker. The lead may remove it. |
| `sm-v3-target-k`, the target for the Linux gates and measurements | Kept, like `sm-v3-target-a` to `-j`. |
| `sm-v3-ci-reg-4l`, `-4t`, `-2l`, `-2t` and `sm-v3-ci-tgt-4l`, `-4t`, `-2l`, `-2t` (Part 3) | Removed. |
| The Windows Part 3 folders `ci/j4` and `ci/j2` in the session scratchpad | Removed. |
| Each `budgets` case's temporary home | Removed by the case at its end. |

No other volume or folder was removed. `sm-v3-cargo` was used, read and written, as the gates use it. Nothing touched the owner's profile, `~/.claude` or `~/.cursor`. Every daemon started was stopped.

## 10. Licences

No third-party crate was added. `Cargo.lock` gains only the internal `sm-store` under `sm-cli`. The `windows-sys` features are of a crate already pinned and covered by `docs/v3/LICENSES.md`.
