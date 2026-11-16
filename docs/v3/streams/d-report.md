# Stream D report: `sm-cli` without the daemon, and the plugin package

Branch `arman-tech/stream-d-clients`. Code commits `573415a` (`sm-cli`) and `6ea692c` (plugin), on top of `61fd6d5`.

## 1. Done when

Yes: all five hook events and the plugin package run against a mock daemon that speaks the frozen message schema, on Windows and on Linux; the `daemon` subcommand is left for phase 3.

## 2. Files built

- `crates/sm-cli/Cargo.toml`: library `sm_cli` and binary `spatial-memory`; dependencies taken from the workspace pins, no extra features enabled.
- `crates/sm-cli/src/main.rs`: the binary; calls `sm_cli::main_entry`.
- `crates/sm-cli/src/lib.rs`: `main_entry`; dispatches `hook` before the argument parser so a hook never fails on parsing.
- `crates/sm-cli/src/cli.rs`: the clap command line (`mcp`, `setup`, `doctor`, `import`, `export`, `migrate`, `reindex`, `project`); `hook` and `daemon` are not clap subcommands.
- `crates/sm-cli/src/client.rs`: the daemon client: one request per connection, deadlines, auto-start under the lock, other-build retirement, Windows handle disinheritance.
- `crates/sm-cli/src/hook.rs`: the hook client for the five events and both clients; always exits 0.
- `crates/sm-cli/src/hook_input.rs`: version 2's hook input rules (event aliases, client detection, standard input decoding, session, working directory and transcript rules).
- `crates/sm-cli/src/mcp.rs`: the Model Context Protocol (MCP) proxy over rmcp, 13 tools with input and output schemas.
- `crates/sm-cli/src/setup.rs`: `setup` and `setup --remove` for Claude Code and Cursor.
- `crates/sm-cli/src/doctor.rs`: `doctor` checks and `doctor --eval`.
- `crates/sm-cli/src/ops.rs`: import, export, migrate, reindex, project rename and project delete.
- `crates/sm-cli/src/profile.rs`: profile resolution for the command line.
- `crates/sm-cli/src/locate.rs`: `PATH` lookup.
- `crates/sm-cli/src/model_step.rs`: the B-43 model step trait and its unwired implementation.
- `crates/sm-cli/tests/common/mod.rs`: `MockDaemon`, `MockSpawn`, `Sandbox` and the binary runner.
- `crates/sm-cli/tests/autostart.rs`: auto-start, lock, concurrent start, other-build and hang tests.
- `crates/sm-cli/tests/hook_budgets.rs`: AT-3, AT-14, AT-15, AT-17 and the SPEC 11 hook budget.
- `crates/sm-cli/tests/hook_fixtures.rs`: every `hook_payloads/` file and the applicable `event_aliases.json` cases through the binary.
- `crates/sm-cli/tests/mcp_proxy.rs`: the proxy driven by an rmcp client over a child process.
- `crates/sm-cli/tests/ops.rs`: operation subcommands against the mock daemon.
- `crates/sm-cli/tests/plugin.rs`: the plugin files, the plugin hooks against the mock daemon, and `claude plugin validate`.
- `crates/sm-cli/tests/setup.rs`: registration for Claude Code and Cursor.
- `plugin/.claude-plugin/plugin.json`: manifest 3.0.0 with the `profile`, `ambient_recall` and `binary_path` options.
- `plugin/.mcp.json`: the `spatial-memory mcp` server, passing the profile option in `env`.
- `plugin/hooks/hooks.json`: the five hooks in exec form.
- `plugin/README.md`: user documentation.
- `plugin/icon.png`: copy of `logo/spatial-memory-icon-512.png`.
- `.claude-plugin/marketplace.json`: the marketplace entry for 3.0.0.
- `Cargo.lock`: 63 packages added, none removed or changed in version.

## 3. Test commands and exact results

### Windows (PowerShell)

`cargo fmt --check`: exit 0, no output.

`cargo clippy --workspace --all-targets --all-features -- -D warnings`: exit 0, no warnings.

`cargo test -p sm-cli --all-features -- --nocapture`: exit 0.

```
lib            test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
main           test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart      test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 21.04s
hook_budgets   test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 65.39s
hook_fixtures  test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.26s
mcp_proxy      test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s
ops            test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
plugin         test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.49s
setup          test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
doc-tests      test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test --workspace --all-features`: exit 0.

```
sm_cli lib     test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_cli main    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart      test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.06s
hook_budgets   test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 65.34s
hook_fixtures  test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.21s
mcp_proxy      test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
ops            test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
plugin         test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.55s
setup          test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
sm_core lib    test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
doc sm_cli     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc sm_core    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test --release -p sm-cli --all-features --test hook_budgets s11_ -- --nocapture`:

```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 3.75s
```

The Windows library has 2 more tests than Linux (`s3_windows_flags_detach_without_breakaway`, `spike2_detached_daemon_does_not_inherit_the_callers_pipes`); `autostart` has 1 fewer (`spike2_unix_daemon_leads_a_new_session` is Unix only).

### Linux (Docker, `rust:1.95-trixie`, target volume `sm-v3-target-d`)

Run 3: `cargo fmt --check && cargo test --workspace --all-features --locked && cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`. Output `fmt-ok`, `clippy-ok`, docker exit 0.

```
sm_cli lib     test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_cli main    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
autostart      test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.98s
hook_budgets   test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 60.43s
hook_fixtures  test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.44s
mcp_proxy      test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
ops            test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
plugin         test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
setup          test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
sm_core lib    test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
doc sm_cli     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc sm_core    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Run 2 (`cargo test -p sm-cli ... -- --nocapture`, `cargo clippy -p sm-cli ...`, then the release budget) also exited 0; its release budget line:

```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.86s
```

In Docker the plugin test prints `claude is not on PATH: plugin validation skipped` and passes, as the stream file asks.

## 4. Acceptance tests

Every test below passes on Windows and on Linux unless marked otherwise.

**All five hook events, both clients** (stream D file):
- `hook_fixtures::s16_every_hook_payload_against_an_answering_daemon`: all 75 `hook_payloads/` files (42 send a capture, 6 a recall) against an answering mock daemon.
- `hook_fixtures::at15_every_hook_payload_with_no_daemon`: the same 75 with no daemon.
- `hook_fixtures::s16_stdout_contract_for_both_clients` (14 cases), `s16_cli_cases` (17), `s16_dispatch_cases` (7), `s16_session_start_handler_cases` (9): each case run twice, with an answering daemon and with none.
- The 11 `hook_input::tests::s16_*` unit tests: `normalize_event_cases` (26), `parse_client_flag_cases` (7), `detect_client_cases` (9), `normalize_stdin_cases` (23) and `normalize_stdin_rules` (8), `validate_common_cases` (13), plus session, working directory and transcript rules.
- `plugin::s16_every_plugin_hook_runs_against_the_mock_daemon`: each entry of `plugin/hooks/hooks.json` run as written.

Cases that do not apply to version 3, with the reason:
- `translate_output_for_cursor_cases` (7), `legacy_entrypoint_cases` (5), `handler_map`, `legacy_trigger_sources`: version 2's Python entrypoints and translation layer; version 3 has one binary and writes each client's format directly.
- `canonical_events_master`, `master_vs_v2`: comparisons with the Python `master` branch.
- `queue_files`, `state_files`, `hook_errors_log`: the daemon's job in version 3.
- The `dispatcher.py` usage text: replaced by the version 3 usage.
- Cursor `sessionStart` output: dropped by B-24; 3 cases overridden in `hook_fixtures.rs` (`cursor_sessionStart_detected_by_version`, `cursor_sessionStart_synthesized_source`, `contract_cursor_SessionStart_startup`).
- The namespace environment variable: version 3 has no namespaces.
- `post_tool_use_skip_tool`: version 3's hook sends the capture and the daemon filters by tool, so the expectation changes from "nothing sent" to "capture sent".
- `registration` data: checked by the setup and plugin tests instead of by the hook.

**AT-3 (safety part)**: `hook_budgets::at3_prompt_hook_with_no_daemon_prints_nothing_inside_the_deadline`, `hook_budgets::at3_prompt_hook_with_a_hung_daemon_prints_nothing_inside_the_deadline`. 20 runs each, both clients. Worst: no daemon 9.8 ms (Windows), 2.1 ms (Linux); hung daemon 1,422.8 ms (Windows), 1,401.3 ms (Linux). Limit 1,500 ms. Passes.

**AT-14 (hook side)**: `at14_stop_with_the_loop_guard_prints_continue_and_sends_nothing`, `at14_stop_without_the_loop_guard_captures_and_prints_continue`, `at14_pre_compact_prints_nothing_with_or_without_a_daemon`. Passes.

**AT-15 (hook side)**: `at15_capture_hooks_with_no_daemon_spool_inside_200_ms` (20 runs per event and client; each spool file read back as a capture `Request`), `at15_the_hook_never_sends_a_spooled_request_again`, `at15_a_late_or_refusing_daemon_gets_the_capture_spooled_once`, and `hook_fixtures::at15_every_hook_payload_with_no_daemon`. Worst of all runs: 17.4 ms (Windows), 2.1 ms (Linux). Limit 200 ms. Passes.

**AT-17 (hook side)**: `at17_switched_off_makes_no_connection_takes_no_lock_and_spawns_nothing` (environment variable off), `at17_switched_off_in_process_counts_no_spawn`, `at17_the_printed_block_is_exactly_the_reply_text_through_prompt_hook_stdout`, and `plugin::s16_the_plugin_option_off_silences_the_prompt_hook` (plugin option off). Passes.

**CONTRACTS 3.1 and spike decision 2**:
- `autostart::spike2_hooks_started_together_with_no_daemon_start_exactly_one_daemon`: 10 rounds of 4 hooks; one daemon running every round.
- `autostart::s3_waiting_clients_started_together_start_exactly_one_daemon`.
- `client::tests::s3_windows_flags_detach_without_breakaway` (Windows): the creation flags never include `CREATE_BREAKAWAY_FROM_JOB`.
- `client::tests::spike2_detached_daemon_does_not_inherit_the_callers_pipes` (Windows).
- `autostart::spike2_unix_daemon_leads_a_new_session` (Linux only).
- Version mismatch sends `shutdown` and restarts: `s3_the_hook_retires_another_build_and_the_next_prompt_reaches_its_own`, `s3_the_waiting_client_retires_another_build_and_starts_its_own`, `s3_a_capture_hook_meeting_another_build_spools_then_retires_it`.
- Also `s3_a_held_lock_means_the_client_does_not_spawn`, `s3_the_waiting_client_starts_a_missing_daemon_once_and_waits_for_it`, `s3_a_daemon_that_never_comes_gives_daemon_unavailable_after_the_window`, `s3_a_hung_daemon_gives_daemon_unavailable`, `s3_loading_counts_as_connected_and_a_model_op_gets_daemon_loading`, `s3_the_hook_binary_leaves_the_lock_free_after_starting_a_daemon`.

All pass.

**SPEC 8 and CONTRACTS 11**: `mcp_proxy::s8_the_proxy_lists_the_thirteen_tools_with_the_parameter_type_schemas`, `s8_a_call_with_an_unknown_field_is_refused_before_the_daemon`, `s8_a_daemon_error_reaches_the_client_with_its_code_and_next_action`, `s8_a_loading_daemon_reaches_the_client_as_daemon_loading`. Driven by an rmcp client over a child process. Passes.

**SPEC 16 registration**: `setup::s16_setup_twice_gives_the_same_settings_file_and_one_backup_per_change`, `s16_setup_remove_restores_the_original`, `s16_setup_pins_an_explicit_profile_in_every_registration`, `s16_setup_refuses_while_the_plugin_is_enabled`, `s16_cursor_files_hold_plain_command_strings`, `s16_cursor_setup_keeps_the_users_own_hooks`, `s16_the_binary_registers_with_cursor_in_the_users_home_only`, `s16_the_binary_refuses_claude_code_setup_while_the_plugin_is_enabled`. Passes. No test touches the real `~/.claude`, `~/.cursor` or the real `claude mcp` commands.

**The plugin**: `plugin::s16_plugin_hooks_are_exec_form_with_the_five_events_timeouts_and_async`, `s16_plugin_manifest_mcp_file_and_marketplace_agree`, `s16_every_plugin_hook_runs_against_the_mock_daemon`, `s16_the_plugin_option_off_silences_the_prompt_hook`, `s16_claude_plugin_validate_passes_on_the_plugin_and_the_marketplace`. Passes.

Validator output (Claude Code 2.1.325, run on Windows). `claude plugin validate plugin`, with `--strict`, with `--profile directory`, and with `--strict --profile directory`, each printed this and exited 0:

```
Validating plugin manifest: <worktree>\plugin\.claude-plugin\plugin.json

Plugin directory checks for <worktree>\plugin. The plugin directory is the catalog that plugins are submitted to; this is a subset of what it runs.

✔ These checks found nothing to report

The directory also runs a safety scan and a policy review that this command does not.

✔ Validation passed
```

`claude plugin validate .claude-plugin/marketplace.json`, with and without `--strict`, exited 0:

```
Validating marketplace manifest: <worktree>\.claude-plugin\marketplace.json

✔ Validation passed
```

`claude plugin validate .claude-plugin/marketplace.json --profile directory`, with and without `--strict`, exited 1. The directory profile applies only to a plugin folder:

```
Validating marketplace manifest: <worktree>\.claude-plugin\marketplace.json

--profile directory needs a plugin folder (one with .claude-plugin/plugin.json, or one with no plugin.json and a skills/<name>/SKILL.md; a symbolic link in place of the folder, of skills, of <name> or of SKILL.md does not count, and SKILL.md must be a file), and this target was validated as a marketplace. To check a plugin in it, pass that plugin's folder or its .claude-plugin/plugin.json. None of the plugin directory's rules were checked.

✘ Validation failed (--profile directory needs a plugin folder)
```

**Budget**: `hook_budgets::s11_hook_process_warm_100_runs_inside_the_hook_budget`. Figures in section 5. Passes.

## 5. Measured budgets

Machine: AMD Ryzen 9 9950X3D, 16 cores, 32 logical processors, 93.6 GB, shared with up to three other streams' builds. Windows 11 Pro 10.0.26300. Linux: Docker `rust:1.95-trixie` on WSL2 (Windows Subsystem for Linux) kernel 6.6.87.2-microsoft-standard-WSL2, `nproc` 32. Each figure is the hook process from start to exit against the mock daemon.

**Hook budget, release build, 100 warm runs per case** (SPEC 11 hook budget: 95th percentile (p95) at most 300 ms, which also covers embedding and search in the daemon; spike e measured 5.4 ms and 6.0 ms for the program alone):

| Case | Windows median / p95 / worst (ms) | Linux median / p95 / worst (ms) |
|---|---|---|
| prompt, Claude Code | 4.7 / 6.0 / 8.1 | 1.0 / 1.1 / 1.4 |
| prompt, Cursor | 4.8 / 5.4 / 7.2 | 1.0 / 1.1 / 1.4 |
| post-tool-use, Claude Code | 3.8 / 4.1 / 7.8 | 0.9 / 1.0 / 1.4 |
| pre-compact, Claude Code | 3.9 / 4.4 / 6.4 | 0.9 / 1.0 / 1.3 |
| stop, Claude Code | 3.9 / 4.3 / 6.4 | 0.9 / 1.1 / 1.2 |
| post-tool-use, Cursor | 3.9 / 4.3 / 7.0 | 0.9 / 1.0 / 1.2 |
| pre-compact, Cursor | 3.9 / 4.3 / 6.7 | 0.9 / 1.0 / 1.2 |
| stop, Cursor | 3.9 / 4.1 / 6.5 | 0.9 / 1.0 / 1.2 |
| all events, n=800 | 3.9 / 5.1 / 8.1 | 0.9 / 1.1 / 1.4 |

The same test in the debug test profile: all events, n=800, Windows 4.5 / 6.4 / 9.3 ms, Linux 1.2 / 1.5 / 2.1 ms.

**AT-3 and AT-15, debug test profile, 20 runs each** (median / p95 / worst, ms):

| Case | Windows | Linux |
|---|---|---|
| AT-3 no daemon, Claude Code | 7.9 / 9.0 / 9.8 | 1.5 / 2.0 / 2.1 |
| AT-3 no daemon, Cursor | 7.7 / 8.5 / 9.4 | 1.5 / 1.8 / 1.9 |
| AT-3 hung daemon, Claude Code | 1413.4 / 1421.0 / 1422.8 | 1401.0 / 1401.2 / 1401.2 |
| AT-3 hung daemon, Cursor | 1414.6 / 1422.5 / 1422.5 | 1401.0 / 1401.2 / 1401.3 |
| AT-15 Claude Code post-tool-use | 7.4 / 7.7 / 11.5 | 1.5 / 1.7 / 2.1 |
| AT-15 Claude Code pre-compact | 7.5 / 9.8 / 17.4 | 1.5 / 1.7 / 2.0 |
| AT-15 Claude Code stop | 7.4 / 7.9 / 10.4 | 1.5 / 1.7 / 1.9 |
| AT-15 Cursor postToolUse | 7.6 / 9.6 / 10.2 | 1.5 / 1.7 / 2.1 |
| AT-15 Cursor preCompact | 7.5 / 8.2 / 9.4 | 1.5 / 1.8 / 2.0 |
| AT-15 Cursor stop | 7.6 / 10.0 / 11.4 | 1.5 / 1.7 / 1.9 |

The hung-daemon prompt hook gives up at 1,400 ms by design (PRINT_RESERVE, section 6); the Windows worst leaves 77 ms of the 1,500 ms limit.

**Single runs, debug** (Windows / Linux): capture with a late daemon 182.5 / 162.0 ms; capture with a hung daemon 181.2 / 161.8 ms; capture meeting another build 9.5 / 2.5 ms; prompt with a late daemon 1412.1 / 1401.4 ms; prompt meeting another build 9.2 / 2.3 ms; waiting client from no daemon to connected 51.0 / 50.9 ms.

**Concurrent start, debug**: 4 waiting clients started together made 3 spawn calls on Windows and 1 on Linux, one daemon each time. 10 rounds of 4 hooks: spawn calls per round `[1, 2, 1, 2, 2, 2, 2, 2, 2, 2]` (Windows) and `[1, 1, 3, 2, 1, 1, 3, 1, 2, 1]` (Linux), one daemon running every round.

## 6. Decisions

1. **Deadlines run on a thread.** A request with a deadline runs on its own thread and the caller waits with `recv_timeout`. Why: interprocess 2.4.8 has no receive timeout on Windows named pipes. Evidence: `interprocess-2.4.8/src/os/windows/named_pipe/local_socket/stream.rs:55` returns `no_timeouts()`; the Unix form maps to `set_read_timeout` (`src/os/unix/uds_local_socket/stream.rs:62`). SPEC 3, CONTRACTS 3.
2. **Client constants.** Poll interval 50 ms, connect window 10 s, probe timeout 1 s, connect wait 2 s (`client.rs:35-43`). The SPEC gives the window; the rest are not specified. SPEC 3.
3. **Hook reserves.** The hook keeps 40 ms to write a spool file (SPOOL_RESERVE, `hook.rs:34`) and 100 ms to print and exit (PRINT_RESERVE, `hook.rs:39`). Why: on Windows, start, journal line and exit measured 13 to 21 ms past the give-up point; 100 ms keeps the hung-daemon case under 1,500 ms (worst 1,422.8 ms). SPEC 11 AT-3, SPEC 16.
4. **The prompt hook makes one exchange and never waits.** A reply of `daemon_loading`, a hang, or another build gives empty output. After a version mismatch the hook retires the other build and returns at once; the next prompt reaches the new daemon (measured 9.2 ms on Windows). SPEC 16, CONTRACTS 3.1.
5. **The capture hook spools on `daemon_loading`.** Loading, failure and timeout all spool the capture, so nothing is lost while the model loads (`hook.rs:381`). SPEC 16, AT-15.
6. **The connector tries the lock again after retiring another build.** Normally the waiting client makes one `try_lock` and never retries. After it has retired another build, each poll that finds nothing listening tries the lock once more, until this client has spawned a daemon (`client.rs:436-471`). Why: the retiring daemon releases the lock last (CONTRACTS 3.1, daemon rule 5), so the first attempt can find it still held. CONTRACTS 3.1.
7. **Concurrent spawns are allowed; one daemon runs.** Between a client dropping the lock and the new daemon taking it, another client can spawn too. The extra daemon finds the lock held and exits (CONTRACTS 3.1, daemon rule 1). The test checks one daemon running, spawn calls = running + exited, and the lock held. The 4-client test saw 3 spawn calls on Windows. SPEC 3, spike 2.
8. **The Windows daemon does not inherit the caller's standard handles.** Before spawning, the client clears `HANDLE_FLAG_INHERIT` on its standard handles with `SetHandleInformation` (`client.rs:121-150`). Why: `Command::spawn` passes `bInheritHandles=TRUE`, so the daemon held the hook's or proxy's stdout pipe and Claude Code would see no end of file until the daemon exited. Test: `spike2_detached_daemon_does_not_inherit_the_callers_pipes`. Creation flags `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`, never breakaway; Unix uses `setsid`. Spike 2.
9. **`unsafe` in `sm-cli`.** Two blocks in delivered code: `setsid` in `pre_exec` (`client.rs:101`) and `SetHandleInformation` (`client.rs:131`), plus two in Windows unit tests (`client.rs:538`, `:545`). `sm-core` stays `forbid(unsafe_code)`. Spike 2.
10. **The MCP proxy connects lazily.** The first tool call starts the daemon, so a server health check (`claude mcp get`, `claude mcp list`) does not. It runs `parse_args` then `Op::validate` before sending, so an unknown field never reaches the daemon. Errors return `isError: true` with `structuredContent` holding code, message and next action. SPEC 8, CONTRACTS 11.
11. **`setup` idempotency reads `~/.claude.json`.** It reads `mcpServers.spatial-memory` from `~/.claude.json`, where `claude mcp add --scope user` stores it (measured with Claude Code 2.1.325, `setup.rs:215-218, 524`). Same command and arguments prints `MCP server: already registered at user scope`; otherwise `claude mcp remove --scope user`, then `add`. Why: `claude mcp get` and `list` health-check every server (slow, and they would start our `mcp` process) and print text meant for people. SPEC 16.
12. **Detecting the enabled plugin.** `enabledPlugins` keys are `plugin@marketplace`, read from local, then project, then user settings. If none of them has the key, the fallback is `claude plugin list --json`. Managed settings are not read. Evidence: Claude Code settings reference, line 99: "enabledPlugins | Turn individual plugins on or off per scope | ... | Any file". SPEC 16.
13. **Backups and writes.** Backups are named `<file>.spatial-memory-<millis>[-n].bak` (`setup.rs:301`). Every write goes to a temporary file and is renamed into place. `--remove` with no original file leaves `{}` (`tests/setup.rs:319`). The Cursor `mcp.json` keeps every other key. SPEC 16.
14. **`setup --remove` uses `Settings::default()`** (`cli.rs:264`). Removal needs no profile, so a broken configuration file cannot block it. SPEC 16.
15. **`doctor` sends `status{metrics: true}`** when the daemon is ready (`doctor.rs:227`). The registration check passes when neither the plugin nor `setup` is registered, and fails only when both are. SPEC 16.
16. **SessionStart** is registered with matcher `startup|resume` and a 5 s timeout. It prints the short note for Claude Code and nothing for Cursor. B-5, B-24.
17. **Plugin `profile` option defaults to `""`.** A non-empty option overrides `SPATIAL_MEMORY_PROFILE` and the `[[profile_rule]]` entries; empty counts as unset. Order: `--profile`, `CLAUDE_PLUGIN_OPTION_PROFILE`, `SPATIAL_MEMORY_PROFILE`, rule, default (sm-core `config.rs:553`). SPEC 9, SPEC 16.
18. **The MCP entry passes the profile in `env`.** `.mcp.json` sets `CLAUDE_PLUGIN_OPTION_PROFILE: "${user_config.profile}"`. Why: Claude Code exports `CLAUDE_PLUGIN_OPTION_<KEY>` to hook processes only; MCP stdio servers get `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA` and `CLAUDE_PROJECT_DIR` (plugins reference, lines 493 and 567). SPEC 16.
19. **Plugin `ambient_recall` option: `on` or `off`, default `on`.** An option value must be a 1 to 64 character label, so an empty "unset" value is impossible. SPEC 6 says on by default, and sm-core's `AmbientSettings` default is `enabled: true`. sm-core's `resolve_ambient` (`config.rs:734`) gives the plugin option precedence over `SPATIAL_MEMORY_AMBIENT_RECALL` and the configuration (SPEC 16, line 624). Consequence: for a plugin user whose option Claude Code exports, the environment variable has no effect; the plugin option is the switch. SPEC 6, SPEC 16.
20. **The README install line targets the default branch** of `arman-tech/spatial-memory-mcp`. It gives the one-step `/plugin install` and the two-step form for older Claude Code. SPEC 16.
21. **spike2 test tolerance (found on Linux).** In Linux run 1, a prompt hook in the concurrent test printed recall: a sibling hook's mock daemon was already listening. The test now accepts no output or exactly `prompt_hook_stdout(BLOCK)` from the sibling's daemon and counts it; any other output still fails. The single-daemon, lock and spawn-count assertions were added, not removed. Spike 2.

## 7. Deviations

- **`binary_path` is declared but not applied** (SPEC 16, line 652: "The plugin offers an optional `binary_path` setting"). Claude Code substitutes `${user_config.KEY}` in MCP server configuration, language server configuration, exec-form hook `args`, and skill and agent content, but not in a hook's `command` (plugins reference, line 492). An empty default command would also break every hook. The plugin always runs `spatial-memory` from `PATH`. The option's description and `plugin/README.md` say it is not applied in 3.0 and point to `spatial-memory setup`, which registers the absolute path. The lead may prefer to drop the option instead.

## 8. Contract changes requested

None.

## 9. Unfinished or unverified

- **The model step is unwired** (B-43). `setup` and `doctor` reach it and fail with `model_missing`, a message citing B-43, and the next action "Use a phase 3 build of spatial-memory to download and verify the model."; both exit nonzero.
- **The `daemon` subcommand is not registered**; phase 3 wires it. The tests start a mock daemon through `MockSpawn`.
- **`binary_path` is not applied** (section 7).
- **`CLAUDE_CONFIG_DIR` is not honoured.** `setup`, `doctor` and the plugin check read `~/.claude` and `~/.claude.json` only.
- **Claude Code version.** `userConfig` options need Claude Code 2.1.271 or later (plugins reference, lines 433 and 480); earlier versions cannot load the plugin.
- **Unverified: whether Claude Code exports an option's default** when the user never saved a value. If it does not, `ambient_recall` falls back to `SPATIAL_MEMORY_AMBIENT_RECALL`, then the configuration; recall stays on by default either way.
- **`claude plugin validate .claude-plugin/marketplace.json --profile directory` fails by design**; the directory profile applies only to a plugin folder.
- **The plugin validation test is skipped in Docker**: `claude` is not on `PATH` there.
- **The real `claude mcp add` and `claude mcp remove` were never run.** The setup tests use an in-process fake `ClaudeRunner` (`setup.rs:113`), and the binary-level setup tests stop before `claude` is called. Only `claude plugin validate` (read-only) ran for real. No round trip of a real `claude mcp add` followed by `spatial-memory setup` was run, so decision 11's reading of `~/.claude.json` is untested against a live Claude Code.

## 10. Licences

63 packages added to `Cargo.lock` (`cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked`). No extra crate features were enabled. All are permissive. `doctest-file` and `recvmsg` are 0BSD (Zero-Clause BSD, a public-domain-equivalent BSD licence); `interprocess` offers 0BSD or Apache-2.0.

| Crate | Version | Licence |
|---|---|---|
| android_system_properties | 0.1.6 | MIT OR Apache-2.0 |
| anstyle | 1.0.14 | MIT OR Apache-2.0 |
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| base64 | 0.23.1 | MIT OR Apache-2.0 |
| bytes | 1.12.1 | MIT |
| cc | 1.11.2 | MIT OR Apache-2.0 |
| cfg_aliases | 0.2.2 | MIT |
| chrono | 0.4.45 | MIT OR Apache-2.0 |
| clap | 4.6.7 | MIT OR Apache-2.0 |
| clap_builder | 4.6.7 | MIT OR Apache-2.0 |
| clap_derive | 4.6.7 | MIT OR Apache-2.0 |
| clap_lex | 1.1.1 | MIT OR Apache-2.0 |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 |
| darling | 0.24.1 | MIT |
| darling_core | 0.24.1 | MIT |
| darling_macro | 0.24.1 | MIT |
| doctest-file | 1.1.1 | 0BSD |
| find-msvc-tools | 0.1.15 | MIT OR Apache-2.0 |
| futures | 0.3.34 | MIT OR Apache-2.0 |
| futures-channel | 0.3.34 | MIT OR Apache-2.0 |
| futures-executor | 0.3.34 | MIT OR Apache-2.0 |
| futures-io | 0.3.34 | MIT OR Apache-2.0 |
| futures-macro | 0.3.34 | MIT OR Apache-2.0 |
| futures-sink | 0.3.34 | MIT OR Apache-2.0 |
| heck | 0.5.0 | MIT OR Apache-2.0 |
| iana-time-zone | 0.1.65 | MIT OR Apache-2.0 |
| iana-time-zone-haiku | 0.1.2 | MIT OR Apache-2.0 |
| ident_case | 1.0.1 | MIT/Apache-2.0 |
| interprocess | 2.4.8 | 0BSD OR Apache-2.0 |
| log | 0.4.35 | MIT OR Apache-2.0 |
| mio | 1.2.5 | MIT |
| nix | 0.31.3 | MIT |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| pastey | 0.2.3 | MIT OR Apache-2.0 |
| process-wrap | 10.0.1 | Apache-2.0 OR MIT |
| recvmsg | 1.0.0 | 0BSD |
| rmcp | 3.10.0 | Apache-2.0 |
| rmcp-macros | 3.10.0 | Apache-2.0 |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| signal-hook-registry | 1.4.8 | MIT OR Apache-2.0 |
| sm-cli | 3.0.0 | MIT |
| socket2 | 0.6.5 | MIT OR Apache-2.0 |
| strsim | 0.11.1 | MIT |
| syn | 2.0.119 | MIT OR Apache-2.0 |
| tokio | 1.54.1 | MIT |
| tokio-macros | 2.7.2 | MIT |
| tokio-stream | 0.1.19 | MIT |
| tokio-util | 0.7.20 | MIT |
| tracing | 0.1.44 | MIT |
| tracing-attributes | 0.1.31 | MIT |
| tracing-core | 0.1.36 | MIT |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| widestring | 1.2.1 | MIT OR Apache-2.0 |
| windows | 0.62.2 | MIT OR Apache-2.0 |
| windows-collections | 0.3.2 | MIT OR Apache-2.0 |
| windows-core | 0.62.2 | MIT OR Apache-2.0 |
| windows-future | 0.3.2 | MIT OR Apache-2.0 |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 |
| windows-numerics | 0.3.1 | MIT OR Apache-2.0 |
| windows-result | 0.4.1 | MIT OR Apache-2.0 |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 |
| windows-threading | 0.2.1 | MIT OR Apache-2.0 |
