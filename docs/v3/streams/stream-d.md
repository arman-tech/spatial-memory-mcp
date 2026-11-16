# Stream D: `sm-cli` and the plugin

## Goal

Build `crates/sm-cli`, the `spatial-memory` binary, without the daemon itself: the hook client for all five events, the MCP proxy, `setup`, `doctor`, the command-line operations, and the client side of auto-start. Replace the version 1 plugin package with version 3's. Everything runs against a mock daemon that speaks the frozen message schema; the lead wires the real daemon in phase 3.

**SPEC 14 done when:** "All five hook events and the plugin package run against a mock daemon that speaks the frozen message schema; the daemon subcommand is wired in phase 3."

## You own

- `crates/sm-cli/` (new crate; binary name `spatial-memory`; depends on `sm-core` only in phase 1, with `sm-core` `test-support` in `[dev-dependencies]`).
- `plugin/` (replace the version 1 files: `.claude-plugin/plugin.json`, `.mcp.json`, `hooks/hooks.json`; add the README and the icon).
- `.claude-plugin/marketplace.json` at the repository root.

## Read

- **SPEC:** section 3 (the subcommand table, rules 1 to 3 and 7), section 6 (ambient recall: the gates, the four-word minimum, the 1,500 ms deadline, the block), section 8 (the 13 tools, the conventions, what moved to the command line), section 9 "Local only" and "Profiles", section 11 (AT-3, AT-14, AT-15, AT-17, the hook budget, test conventions), section 16 (all of it: the event table, the rules, "Carried over from version 2", "How the server and hooks are registered", the verified hooks reference, "Shell-agnostic rules", the spike results).
- **CONTRACTS.md:** all of it, closely sections 3, 3.1 (auto-start and the lock: you build the client side), 4 (clients peek and handle `version_mismatch`), 5, 6, 7 ("Stream D, the hook client") and 11 (the generated tool schemas).
- **DECISIONS:** B-5 (the plugin registers SessionStart), B-6 (exec form for every hook), B-11 and B-16 (configuration layering and profile resolution), B-15 (ambient block), B-17 (endpoint naming), B-24 (Cursor and session-start output), B-25 and B-33 (spool), B-26 (framing and handshake), B-28 (strict tool parameters), B-30 (forget confirms through an error), B-34 (the hook journals its own gates and latency), B-37 (the injection race), B-42 (clap 4.6.7), B-43 (setup and doctor take `sm-embed` in phase 3), B-44 (the lock).
- **SPIKE.md** sections c, d, e and f, and `docs/v3/spikes/` (`rmcp-stdio` for an rmcp 3.10.0 stdio server and client; `plugin-exec` for a plugin with exec-form hooks and the three `userConfig` options that passed `claude plugin validate`). References; write the crate fresh.
- **sm-core:** `protocol` (`Request`, `Response`, `Envelope`, `Op`, `Tool::ALL`, `Tool::input_schema`, `Tool::output_schema`, `Tool::parse_args`, `HookEvent`, `AmbientParams`, `AmbientReply`, `CaptureParams`, `CaptureReply`, `DaemonInfo`, `encode_frame`, `read_message`, `new_request_id`, `PROTOCOL_VERSION`), `ambient` (every constant and the stdout helpers), `config` (`Settings`, `resolve_profile`, `resolve_ambient`, `PLUGIN_PROFILE_ENV`, `PLUGIN_AMBIENT_ENV`), `paths` (`ProfilePaths`, `pipe_name`, `unix_socket_path`, `lock_file`), `spool`, `journal`, `types::Client`, `error`.
- **Fixtures:** `fixtures/capture/event_aliases.json` (140 cases: aliases, client detection, Cursor stdin normalisation, validation, stdout per event and client, the command line, dispatch, the version 2 registration data), `fixtures/capture/hook_payloads/` (75 complete payloads, `INDEX.json`), `fixtures/capture/ambient_recall_cases.json`, `fixtures/capture/hook_helpers_cases.json` (stdin parsing, stdout response). Read `fixtures/capture/README.md` first: it says how to read `expect` against `computed_*`.
- **Version 2's command line and setup:** `git show origin/feature/v2-implementations:<path>` for the hooks CLI, `setup_hooks` and the Cursor registration. Port behaviour, not structure.
- **Crate sources:** `rmcp-3.10.0/`, `interprocess-2.4.8/`, `clap-4.6.7/` under `~/.cargo/registry/src/`.
- **Claude Code:** `claude --version` is 2.1.325 here. `claude mcp add --help`, `claude plugin validate --help`, and the official plugins and hooks references for `userConfig`, the icon, the marketplace file and `enabledPlugins`. Verify each field you write against them.

## Build

1. **The command line** (SPEC 3, CONTRACTS 5). clap 4.6.7 with the pinned features. Subcommands `mcp`, `hook <event> --client <claude-code|cursor>`, `setup`, `doctor` (with `--eval <file>` and an optional baseline), `import`, `export`, `migrate`, `reindex`, and `project rename` and `project delete`. Every subcommand takes `--profile`, resolved with `config::resolve_profile`. Paths sent to the daemon are made absolute by the client. Do not register the `daemon` subcommand; the lead adds it in phase 3.
2. **The daemon client** (CONTRACTS 3, 3.1, 4).
   - Connect to the profile's endpoint with `interprocess` 2.4.8: a named pipe on Windows, a Unix domain socket elsewhere. Send `Request` frames with `encode_frame` and read replies with `read_message`.
   - Every request carries `v`, `build`, `id`, `cwd`, `session_id` when there is one, `client`, `profile` and `deadline_ms`.
   - Peek every reply's envelope (`Response::check_version`) before decoding the body. On `version_mismatch`, send `shutdown`, wait for the endpoint to close, then auto-start.
   - **Deadlines hold even when the daemon accepts the connection and never answers.** The hook's 1,500 ms and 200 ms limits are wall-clock limits on the whole process. Verify in the `interprocess` source how a blocking read can time out on each platform, and choose a design that cannot hang (for example a reader thread with a channel and `recv_timeout`). Record it.
   - **Auto-start, client side** (CONTRACTS 3.1): one `try_lock` without retry; spawn `<current_exe> daemon --profile <name>` detached with null standard streams; Windows flags `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` and never `CREATE_BREAKAWAY_FROM_JOB`; Unix `setsid()` in `pre_exec`, no `process_group(0)`. The spawn is injectable, so tests start the mock daemon instead.
   - The MCP proxy and the command line retry connect plus `hello` every 50 ms for up to 10 s; the hook never waits.
3. **The hook client** (CONTRACTS 7, SPEC 16, every case in `event_aliases.json` that applies to version 3).
   - Read standard input as UTF-8, map event aliases, detect the client, normalise Cursor's input to Claude Code's field names, and apply version 2's validation.
   - `session-start`: `ambient::session_start_output(source)` for Claude Code, nothing for Cursor (B-24). No daemon contact.
   - `user-prompt-submit`: resolve the switches with `config::resolve_ambient` (plugin option, then version 2's variables, then configuration). When ambient recall is off, print nothing and contact nothing: no connect, no lock, no spawn (AT-17). Apply the four-word minimum. Send `ambient_recall` with `deadline_ms` set to the time left of 1,500 ms. Print `ambient::prompt_hook_stdout(text)` only when `gate` is `injected`. Journal `hook_gate_total`, `hook_top_score` and `hook_latency_ms` (B-34).
   - `post-tool-use`, `pre-compact`, `stop`: normalise the payload and send `capture` within 200 ms. If the daemon cannot be reached or does not accept in time, write the same request with `spool::write_spool`. `stop` with `stop_hook_active` true sends nothing. `stop` prints `ambient::CONTINUE_JSON` on Claude Code; `pre-compact` prints nothing (AT-14). Use each fixture's stdout rule for Cursor.
   - A capture spooled after its 200 ms ran out may still reach the daemon from the abandoned connection; the daemon's content-hash dedupe covers that (SPEC 16 pipeline step 4). Say so in a code comment, and test that the hook never sends a spooled request a second time itself.
   - **Never blocks** (SPEC 3 rule 3): on any error, timeout, a daemon still loading, a malformed payload or empty input, print the event's empty output and exit 0. Never exit with a blocking code. UTF-8 output with no byte-order mark.
4. **The MCP proxy** (SPEC 8, CONTRACTS 11). An rmcp 3.10.0 server over standard input and output that lists exactly the 13 `Tool::ALL` tools with their generated input schemas (and output schemas, if rmcp 3.10.0 supports them), with descriptions written from SPEC 8's table. Each call goes through `Tool::parse_args`, then to the daemon; a daemon error becomes a structured MCP tool error carrying `code`, `message` and `next_action`. Unknown fields are refused (B-28).
5. **`setup`** (SPEC 16 "How the server and hooks are registered"; spike s2).
   - Register the MCP server with `claude mcp add --scope user spatial-memory -- <absolute path of this binary> mcp`, adding `--profile` when it is not the default. Start `claude` directly, never through a shell. Make it safe to run twice (check with `claude mcp get` first, or remove and add; choose and record).
   - Merge the five hooks into `~/.claude/settings.json` in exec form with the absolute binary path, after writing a backup. Running twice changes nothing. Keep every hook and setting that is not ours.
   - `--remove` undoes both.
   - Refuse to run while the plugin is enabled, so nothing is registered twice. Find out from the official reference how Claude Code records an enabled plugin, and record the evidence.
   - `--client cursor` registers Cursor's capture hooks and its MCP server, as version 2's setup did, with command strings that are the program name and plain arguments only (SPEC 16 "Shell-agnostic rules").
   - The model step (download and checksums) goes through a trait private to `sm-cli` (B-43), tested with a fake. Its real implementation calls `sm-embed` and is written by the lead in phase 3. Do not write a placeholder that pretends to succeed: until phase 3 the binary reports the model step as "not wired in this build" and `setup` exits non-zero after doing everything else. List this under "Unfinished" in your report.
   - Every file and command the tests touch is under a temporary folder. Calls to `claude` go through an injectable runner; no test runs the real `claude mcp add`, `get` or `remove`.
6. **`doctor`** (SPEC 5 rule 7, SPEC 16).
   - Checks, each reported as passed or failed with a next action: the daemon answers `hello`, its build matches, and its state; the profile and its folders; `spatial-memory` resolves on `PATH` and its version matches the plugin's `plugin.json`; the plugin and `setup` are not both registered; spool files waiting; the model step through the same private trait.
   - `doctor --eval <file>` sends `doctor_eval` and prints nDCG@10, hit-at-5, mean reciprocal rank and the score distributions from `EvalReport`.
   - No `doctor --bench`.
7. **The other command-line operations**: `import`, `export`, `migrate`, `reindex`, `project rename` and `project delete` send their ops (CONTRACTS 5) and print the report. `project delete` without `--confirm` prints what would be removed (B-30). Exit codes: 0 on success, non-zero on an error, with the error's message and next action on standard error.
8. **The plugin package** (SPEC 16; B-5; B-6; SPIKE f).
   - `plugin.json`: version 3.0.0, the MCP entry, and `userConfig` with `profile`, `ambient_recall` (with its options) and `binary_path`. Verify how a `userConfig` value can change the command a hook or MCP entry runs; if `binary_path` cannot be applied that way, record what you found and how the plugin falls back to `PATH`.
   - `.mcp.json`: `spatial-memory mcp` in exec form.
   - `hooks/hooks.json`: all five hooks in exec form, `"command": "spatial-memory"` with `"args": ["hook", "<event>", "--client", "claude-code"]`. SessionStart with matcher `startup|resume` and 5 s; UserPromptSubmit 5 s; PostToolUse 10 s with `"async": true`; PreCompact and Stop 15 s. Nothing sets `onFailure` to block. Matchers follow version 2's registration data in `event_aliases.json`.
   - A README of at least 40 words and an icon from `logo/`, so validation shows neither warning.
   - `.claude-plugin/marketplace.json` updated for 3.0.

## Acceptance tests and other tests

Run the binary from tests through `env!("CARGO_BIN_EXE_spatial-memory")` with `SPATIAL_MEMORY_HOME` set to a temporary folder, so each test has its own endpoint. The mock daemon listens on that endpoint, speaks the frozen schema with `sm_core::protocol`, records every request, and can be told to answer, answer late, hang, reply `loading`, reply `version_mismatch`, or be absent.

- **All five hook events** against the mock daemon, for Claude Code and for Cursor: every applicable case in `event_aliases.json` and every payload in `hook_payloads/`. A case that cannot apply to version 3 (for example a Python entrypoint) is listed in your report with the reason, not silently dropped.
- **AT-3** (safety part): with no daemon, and with a daemon that hangs, the prompt hook exits 0 with empty output inside 1,500 ms. Run it 20 times each and report the worst time.
- **AT-14** (hook side): Stop prints its continue response and sends nothing when `stop_hook_active` is true; PreCompact prints nothing.
- **AT-15** (hook side): with no daemon, each capture hook exits 0 within 200 ms and its payload is in `spool/new/` as a readable capture `Request`. Report the worst of 20 runs per event.
- **AT-17** (hook side): with `SPATIAL_MEMORY_AMBIENT_RECALL=off`, and with the plugin option off, the prompt hook prints nothing and the mock daemon records no connection, no lock is taken and nothing is spawned. The printed block is exactly `AmbientReply.text` through `prompt_hook_stdout`.
- **CONTRACTS 3.1 and spike decision 2**: two hooks started together with no daemon spawn exactly one daemon; the Windows creation flags never include `CREATE_BREAKAWAY_FROM_JOB`; a version mismatch sends `shutdown` and restarts.
- **SPEC 8 and CONTRACTS 11**: the proxy lists exactly 13 tools whose schemas equal `Tool::input_schema`; a call with an unknown field is refused; a daemon error reaches the MCP client with its code and next action. Drive the proxy with an rmcp client over a child process, as spike d did.
- **SPEC 16 registration**: `setup` run twice gives the same settings file and one backup per change; `--remove` restores the original; it refuses while the plugin is enabled; the Cursor files hold plain command strings.
- **The plugin**: a test parses `hooks/hooks.json` and checks exec form, the five events, the timeouts, `async` and the absence of `onFailure`. When `claude` is on `PATH`, the test also runs `claude plugin validate` on `plugin/` and on `.claude-plugin/marketplace.json` and requires success; when it is not (Docker), the test prints one line and passes. Report the validator's full output, and the result of `--strict` and `--profile directory`.
- **Budget**: measure the hook process from start to exit against the mock daemon, 100 runs warm, median and 95th percentile, release build (spike e measured 5.4 ms and 6.0 ms for the program alone). Report it against SPEC 11's 300 ms hook budget, which also includes embedding and search in the daemon.

Run the hook and auto-start tests on Linux in Docker as well; the Unix socket and `setsid` paths are only tested there.
