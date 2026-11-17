# Package K (phase 3): end to end, with the real model, and RESULTS.md

## Goal

Prove the finished binary against SPEC 11. Package K runs after H, I and J have merged:

- end-to-end tests that run the `spatial-memory` binary as Claude Code would, against a real daemon;
- measurements through the `daemon` subcommand with the real model, for every SPEC 11 budget and for the acceptance tests that earlier streams could check only with the fake embedder;
- `docs/v3/RESULTS.md`, which records each budget and acceptance test with its result and the machine it ran on.

K finds out whether the build meets its budgets. It does not tune anything to make it meet them (build rule: never tune a default to make a test pass).

**Done when:**

- every item under "Build" is tested or measured as it says;
- `RESULTS.md` has a row for every SPEC 11 budget and every acceptance test, AT-1 to AT-17;
- both gates pass (RULES.md sections 5 and 6, with the full workspace on Linux);
- `k-report.md` is committed.

A budget that is missed is a result, not a blocker. Record it with the evidence and carry on.

## You own

- **New test files** under `crates/sm-cli/tests/`. Append any helpers to `crates/sm-cli/tests/common/`.
- **A measurement program**, `crates/sm-cli/examples/budgets.rs`, and any small modules it needs under `crates/sm-cli/examples/budgets/`. It is a development tool. It is never shipped and never run by `cargo test`.
- **`docs/v3/RESULTS.md`** (new) and `docs/v3/streams/k-report.md`.
- **Fixes.** If a test or measurement finds a defect, you may fix it in any crate except `sm-core`. Keep the fix small, add a regression test named for its rule, and list it in the report under "Deviations". A fix that needs `sm-core`, `CONTRACTS.md` or a new third-party crate is a request in the report (RULES.md section 2).
- **One `sm-core` edit, and only this one.** The lead has fixed the `doctor_numbers` contract in B-125 (item 8 below). You make that edit in `crates/sm-core/src/protocol.rs`, exactly as B-125 gives it. Make no other change to `sm-core`. `CONTRACTS.md` stays the lead's: report the sections it needs (5 and 6), and do not edit it.
- No new third-party crate. The `sm-cli` dev-dependencies may add workspace crates (`sm-daemon`, `sm-store`, `sm-embed`), with `sm-core`'s `test-support` feature.

## Read

- **SPEC:** sections 3 (the daemon, auto-start, threads), 5 ("Bulk", the thread budget), 10 (`migrate`), 16 (capture, the spool, "Never blocks"); always 11, 13 and 14. Section 11 is the checklist for this package.
- **CONTRACTS.md:** sections 3, 7 and 8.
- **DECISIONS:**
  - B-49: the AT-7 fixture.
  - B-52: model integrity, and the Docker cold start at 8.95 s.
  - B-58: the real-model test binary and how it takes its readings.
  - B-75 and B-114: project scope at 100,000 chunks. It was 243.7 ms at the 95th percentile in the store alone; after mapped readers it is 71.1 ms on Windows and 67.1 ms on Linux in Docker. With query embedding, `recall` is estimated at 115 to 165 ms. B-114 also sets how memory figures are judged.
  - B-80: client deadlines.
  - B-97: one bulk session, and the 10-second grace.
  - B-103: `consolidate`, which compares every pair of memories and is not yet timed.
  - B-109: the project is resolved on every request.
  - B-106: stream E's test conventions.
  - B-113 onwards: packages H, I and J. B-113 makes bulk vec1 inserts linear. B-116 covers capture failures and the bulk queue's strikes. B-118 has the open `reindex` defect below.
  - B-119 to B-123: J's binary. The daemon's exit codes and log lines, clients through `sm_daemon::connect`, the detached spawn, the production sharing adapter, the model step and `status`.
  - B-124: the hook process after linking, 10.4 ms at the 95th percentile on Windows, measured before Linux.
  - B-125: the `doctor_numbers` operation, which you build (item 8).
- **Reports:** `h-report.md`, `i-report.md`, `j-report.md`, `e-report.md`, `d-report.md`, `b-report.md` section 3. Also `docs/v3/PROGRESS.md`, "Left for the lead in phase 3".
- **Code:**
  - `crates/sm-cli/tests/hook_budgets.rs` and `hook_fixtures.rs` (how the hook binary is driven today);
  - `crates/sm-daemon/tests/common/mod.rs` and `acceptance.rs` (the test loader and the fault chunker);
  - `crates/sm-daemon/src/loader.rs`: `ModelLoader` is public, so an `sm-cli` test can give the daemon a fake embedder;
  - `crates/sm-embed/tests/real_model.rs` (how readings are taken on each OS);
  - `crates/sm-store/examples/search_budget.rs`;
  - `crates/sm-cli/src/cli.rs`, the subcommands J delivered: `mcp`, `daemon`, `status`, `hook`, `setup`, `doctor` (with `--eval FILE [BASELINE]`), `import`, `export`, `migrate --from FILE [--default-project NAME]`, `reindex`, and `project rename` and `project delete`. There is no `recall`, `remember`, `share` or `sync` subcommand. Those are MCP tools: reach them by running `spatial-memory mcp` and sending JSON-RPC `tools/call`, as `crates/sm-cli/tests/sharing.rs` does (`mcp_in`, `call_ok`);
  - `crates/sm-cli/tests/common/mod.rs` (J's `Sandbox`, `MockDaemon` and helpers) and `crates/sm-cli/tests/sharing.rs` (an in-process daemon with a fake `ModelLoader` and the production `ShareAdapter`).

## Build

### Part 1: end-to-end tests with the fake embedder (in `cargo test`)

These run in both gates on every machine. Each test runs the binary at the path Cargo gives integration tests (`common::BIN`, which is `env!("CARGO_BIN_EXE_spatial-memory")`).

**Isolation: use J's `Sandbox` (B-120).** It already gives each test its own `SPATIAL_MEMORY_HOME` and user home in a temporary folder, points `SPATIAL_MEMORY_MODEL_DIR` at an empty folder, and points `SPATIAL_MEMORY_MODEL_URL` at a local server that answers 404 (`model_url`, counted by `model_requests()`). So the real model is never loaded and never downloaded.
- `Sandbox::new()` blocks the store, so a daemon the binary auto-starts exits 1 at once. Use it where no daemon must answer.
- `Sandbox::with_daemon()` allows a daemon and shuts it down when dropped.
- Never leave a daemon running after a test, on any path.

**The daemon under test** is the real `sm-daemon`, given a fake `ModelLoader` built on `sm-core`'s fake embedder.
- In process, as `sharing.rs` does: `sm_daemon::start` with the fake loader and the production `ShareAdapter`, on the sandbox's paths, then wait for `hello` to report ready. Use this wherever the daemon need not be killed.
- AT-3 and AT-15 need it in a separate process that can be killed. Run it in a child: the test binary starts itself again (`std::env::current_exe()`, with `--exact <child test name> --nocapture`), with an environment variable that makes that one test serve as the daemon. Without the variable, the child test returns at once. Kill and reap every child before the test ends, on every path.

1. **AT-3, the daemon killed.**
   - A prompt hook is waiting on a request when the daemon process is killed. The hook exits 0 with empty output inside 1,500 ms.
   - Repeat 20 times and report the worst time.
   - Also run the case where the daemon dies before the hook connects.
2. **AT-13 through the binary.** Feed the fixture PostToolUse payloads for Edit, Write and Bash (`fixtures/capture/hook_payloads/`) to `spatial-memory hook`. Then check, through the daemon:
   - strong-signal text is stored as an active memory with capture provenance;
   - weak-signal text stores nothing;
   - the fake credential is redacted.
3. **AT-14 through the binary.** With the fixture transcript (`fixtures/capture/transcript/`), check:
   - PreCompact and Stop store the assistant text that PostToolUse missed, and do not store what it already captured;
   - Stop with `stop_hook_active: true` stores nothing and prints its continue response;
   - PreCompact prints nothing.
4. **AT-15 through the binary.**
   - With no daemon, each capture hook exits 0 within 200 ms, and its payload is in `spool/new/`.
   - Then start the daemon: the spooled item is stored, and the spool is empty.
   - Report the worst time for each hook.
5. **AT-17 through the binary.**
   - With `SPATIAL_MEMORY_AMBIENT_RECALL=off`, the prompt hook prints nothing and does not contact the daemon. Show this from the daemon's side, for example by its connection count, not only by the hook's output.
   - The injected block matches version 2's format (the fixture in `fixtures/capture/ambient_recall_cases.json`).
   - A capture longer than `capture.max_index_tokens` is stored whole, indexed up to the cap, and recorded as a cap hit in the journal.
6. **AT-16 through the binary.** Importance does not change `recall` ranking. Package H delivered this through the daemon (`at16_importance_does_not_change_recall_ranking`, B-118), so add only the binary-level case: two memories that differ only in importance, stored and recalled through `spatial-memory mcp` (`tools/call` for `remember` and `recall`), come back in the same order either way round.
7. **Fix: `reindex` keeps a capture's index cap (B-118).**
   - `reindex` (`crates/sm-daemon/src/admin.rs`) re-chunks every memory with `sh.chunk_params(None)`. A memory whose provenance source is `Capture` can therefore be indexed past `capture.max_index_tokens`.
   - `revise` with a new title already re-chunks with the cap (`crates/sm-daemon/src/tools/edit.rs`, the `(None, Some(t))` arm). Move that provenance-to-cap rule into one helper on `Shared` in `state.rs`. Call it from both places, and in `reindex` call it per memory inside the closure that `Store::reindex` passes each `Memory` to.
   - A revise with new content stays uncapped: it is an explicit edit, and SPEC 16 says "Explicit `remember` is not capped". Do not change it.
   - Regression test, in `crates/sm-daemon/tests/`: `s16_reindex_keeps_a_captured_memorys_index_cap`. Store a capture longer than the cap, run `reindex`, and check that the chunk count and indexed tokens are the same as before. Also check that a `remember`ed memory of the same length is not capped.
   - This is a fix in `sm-daemon`, which you may make (see "Fixes" above). List it under "Deviations".
8. **Build: `doctor` reports both embedder numbers through the daemon (SPEC 5 rule 7; B-125).** SPEC 5 rule 7 says "In version 3.0, doctor reports chunks per second and query latency at the configured budget." `sm-embed` measures both (`crates/sm-embed/src/doctor.rs`, `doctor_numbers`), but only the daemon holds the model, and no operation asks it. Build it as B-125 gives it:
   - **`sm-core`, `protocol.rs`.**
     - Add `Op::DoctorNumbers`, a unit variant like `Hello`, serialised as `{"op":"doctor_numbers"}`.
     - Add `Reply::DoctorNumbers(DoctorNumbersReply)`. It has `physical_cores`, `query_threads`, `bulk_threads`, `batch_size`, `sample_chunks`, `query_tokens` and `query_runs` as `u32`, and `bulk_elapsed_ms`, `bulk_chunks_per_sec` and `query_median_ms` as `f64`. Give it the same derives and serde attributes as the other reply structs.
     - `Op::name` gives `"doctor_numbers"`. Add it to `s3_op_tags_are_stable` and to the round-trip tests beside it.
   - **`sm-embed`.** Move the timing in `doctor_numbers` into a public function that takes an open bulk `&dyn Embedder` and an open query `&dyn Embedder`, with the chunker and settings. `doctor_numbers(provider, ...)` opens the two sessions and calls it, so its existing tests pass unchanged.
   - **`sm-daemon`.**
     - The daemon times its own sessions: the bulk one through `queue::with_bulk` (`crates/sm-daemon/src/queue.rs`), and the query one through `Ready.query` (`state.rs`). It never opens a third session, so AT-4's peak is not raised. The bulk queue waits for the few seconds the timing takes.
     - While the model loads, the operation returns `daemon_loading` (`state::loading()`). After a failed load it returns the model's error.
     - Dispatch accepts no arguments for it (`dispatch.rs`, beside `Op::DoctorEval`). Add it to `s3_request_keys_match_every_key_a_request_serialises`.
   - **`sm-cli`, `doctor` (`crates/sm-cli/src/doctor.rs`).**
     - When the daemon is ready, print the bulk rate (chunks per second, with the thread count and batch size) and the median query latency at 128 tokens (with the thread count). Nothing is tuned from them.
     - While the daemon is loading, `doctor` polls every 50 ms, inside the same 10-second window the waiting client uses (`Connector`, CONTRACTS 3.1). If it is still loading at the end, it prints that the numbers were not measured, with "Run `spatial-memory doctor` again in a few seconds." as the next action. That line is not a failure. Leave J's existing `state` check as it is.
     - A failed model is already a failure in `doctor`'s `state` and `model` checks. Do not add a second failure for it.
   - **Tests**, named `s5_rule7_...`:
     - In `sm-daemon`: the reply's counts match the settings; the bulk session the timing used is the queue's own, and no third session is opened (count opens on a fake provider); `daemon_loading` while loading; the model's error after a failed load.
     - In `sm-cli`: `doctor` against an in-process daemon with the fake loader prints both numbers; against a `MockDaemon` that stays loading, it prints "not measured" with the next action, and exits as it would without that line.
   - List it under "Deviations" as the lead's contract change (B-125). In the report, give the exact `sm-core` diff and name CONTRACTS sections 5 and 6 for the lead.
9. **Fix: the idle line's plural.** `crates/sm-daemon/src/maintenance.rs` logs "no request for {} minutes; exiting", which reads "1 minutes" when `daemon.idle_exit_minutes` is 1. Write "1 minute" for one and "N minutes" otherwise. Regression test, `s3_the_idle_exit_line_says_minute_for_one_and_minutes_otherwise`. Check that no other test or document quotes the old line, and update any that does. List it under "Deviations".
10. **AT-9 through the binary.** SPEC 11 AT-9: "Fixtures with fake credentials are redacted on `remember`, and `share` refuses a file that still matches." The daemon and `sm-share` cases pass already (TRACE, AT-9). Add the binary-level case, in `crates/sm-cli/tests/`, named `at9_`:
    - In a temporary git repository, as `sharing.rs` sets one up, run `spatial-memory mcp` against an in-process daemon with the production `ShareAdapter`.
    - `remember` a memory whose content holds a fixture credential from `fixtures/capture/redaction_cases.json`, picked as `at9_remember_redacts_the_fixture_credentials_and_lists_the_kinds` (`crates/sm-daemon/tests/acceptance.rs`) picks them. Then `share` it.
    - The shared file exists, contains the redaction marker, and contains no credential text. Read the file from disk; do not trust the tool's reply alone.
    - Also run the refusal case through the binary: a memory whose tag or anchor still matches a credential is refused by `share`, with the error code and next action that `at9_share_refuses_credential_in_tag_or_anchor` expects, and no file is written.

### Part 2: measurements with the real model (`examples/budgets.rs`, run by hand)

**Isolation.**
- Each case gets a fresh `SPATIAL_MEMORY_HOME` in the system temporary folder, which it removes at the end. It never touches the owner's profile, the repository, or a worktree.
- The model is read from the per-user cache (RULES.md section 3). Never download it.
- If the model folder is missing, the program prints one skip line and exits 0.

**How it runs.**
- Build first with `cargo build --release -p sm-cli --examples`.
- Run with `cargo run --release -p sm-cli --example budgets -- <case>`. The program runs the release `spatial-memory` binary from the same target folder, and finds it from its own path.
- It prints one line per figure: the case, the count, the 50th and 95th percentiles, the maximum, and the budget. Time is printed, never asserted (B-58). Memory budgets are checked and printed as pass or fail.

**Readings.** Read the daemon process from outside: its peak and current working set, its private memory, and its threads. The methods are as in `real_model.rs` (B-58), applied to the daemon's process id:
- on Windows, `GetProcessMemoryInfo` with `PROCESS_MEMORY_COUNTERS_EX`: the working set, the peak working set, and `PrivateUsage`. Threads from a Toolhelp snapshot.
- on Linux, `/proc/<pid>/status`: `VmRSS` and `VmHWM` for the resident set, `RssAnon` for private memory, and `RssFile` for file-backed pages. Threads from `/proc/<pid>/task`.

**Memory figures (B-114).** Reader connections map the database, so file-backed pages count towards the working set as searches touch them, up to 2 GiB.
- Report the working set and private memory side by side for every memory figure.
- Judge each budget on the working set.
- A budget missed on the working set and met on private memory is recorded as **missed**, with both figures. Do not change `mmap_size` or any other setting to meet it.

**Cases.**

1. **Cold start.**
   - Measure from spawning `spatial-memory daemon` to the first answer that reports the model as ready. Run it 5 times.
   - The Windows file cache cannot be cleared without administrator rights, so record the runs as "warm file cache".
2. **Idle memory.** The daemon's working set 10 s after it is ready, and again after the bulk grace (B-97) has passed once. Budget: 1 GB.
3. **Hook, warm (AT-3, first part).**
   - Run 100 prompt hooks through the binary against the real daemon, with a query of about 128 tokens and a store of a few hundred memories.
   - Report the 95th percentile against 300 ms.
   - Also time the parts the daemon logs or returns: query embedding, search and project resolution (B-109).
4. **Hook during a bulk import (AT-3, second part).**
   - Start an import big enough to keep the bulk session busy for the whole run. For example, `spatial-memory migrate` of a generated version 1 export of 2,000 one-chunk records (`fixtures/migration/make_v1_export_full.py` shows the format).
   - Run 100 hooks during it, and report their 95th percentile.
   - Sample the daemon's thread count every 50 ms. Its peak must not exceed the budget the daemon reports for itself: the query threads, the bulk threads, and the daemon's own threads, which you count before the import starts. Explain each thread you count.
5. **Opening the bulk session (B-97).** Time how long the first batch of an import takes to open its session. Also measure how long hooks take while a session is opening, compared with hooks after it has opened.
6. **The daemon killed (AT-3, third part).** Run Part 1's first item against the real `daemon` subcommand: kill its process during a prompt hook, 20 runs, and report the worst time.
7. **`remember` of one chunk.**
   - Time 20 calls to `remember` with a memory under 256 tokens, until the memory is found by vector `recall`. Budget: 2 s.
   - Say whether `remember` embeds inline or through the queue.
8. **AT-4, peak memory through the daemon.**
   - Store a 50,000-character memory through `spatial-memory mcp` (`remember` has no subcommand). It must succeed and produce multiple chunks, and the daemon's peak working set must stay under 2 GB.
   - Also report the peak during case 4's import. Budget: 2 GB.
9. **`recall` at 100,000 chunks (B-75).**
   - Fill a store through `sm-store`'s public API with 100,000 chunks of synthetic unit vectors. Stamp it with the real model's identity, so the daemon accepts it.
   - Build two layouts, as `search_budget.rs` does: spread over 60 projects, and one project holding all of them.
   - Then run 300 `recall` queries through the daemon with real query embedding, at project scope and at all-projects scope. Budgets: 250 ms and 350 ms at the 95th percentile.
   - Record that the stored vectors are synthetic. Exact search costs the same whatever the vectors hold.
   - Package I already maps the readers (B-114). The store alone is 71.1 ms at the 95th percentile for project scope in the single-project layout, so `recall` is estimated at 115 to 165 ms.
   - If a budget is missed, record it with the figures and report it. Do not change the store's search to meet it: that is a lead decision.
   - Also report the daemon's working set and private memory after the 300 queries, for both layouts.
10. **`consolidate` at 2,000 memories.** Time it on a project of 2,000 embedded memories. There is no budget. Record the figure (B-103).
11. **AT-2 end to end.**
    - Store the retrieval eval set's corpus through the daemon, and let the real model embed it.
    - Run `spatial-memory doctor --eval fixtures/retrieval/retrieval_eval_set.json fixtures/retrieval/retrieval_quality_baseline.json`.
    - nDCG@10, mean reciprocal rank and hit-at-5 must stay within the fixture's tolerances of the baseline.
12. **AT-7 with live embedding (B-49).**
    - Store the 320 texts of `fixtures/spatial/texts.json` through the daemon, with the real model.
    - Run `regions`: at most 25% unclustered and no cluster over 40%.
    - Report the cluster count and how the result compares with the committed-vector result.
13. **AT-8 with the real model.**
    - `spatial-memory migrate --from fixtures/migration/v1_export_full.json`.
    - Wait until `spatial-memory status` prints "0 waiting for embedding" on its `Index:` line (the `status` tool's `pending_embeddings`).
    - Every memory has vectors, and the report's counts match the row count.
    - Report how long embedding took and the chunks per second.
14. **AT-10 through the `daemon` subcommand.**
    - Copy the three model files into a temporary folder outside the repository, and change one byte of one file.
    - Point `SPATIAL_MEMORY_MODEL_DIR` at the copy and start the daemon. It does not become ready. The error names the file, and the log and `doctor` both say `model_integrity` with the next action.
    - Remove the copy afterwards.
15. **Linux (B-52).**
    - Run cases 1 to 4, 7 and 8 in Docker (`rust:1.95-trixie`), with the model in a named Docker volume (`sm-v3-model`), not the Windows file share. Fill the volume once from the per-user cache.
    - Label the figures "Linux in Docker (WSL2 virtual machine, ext4)": they are approximate, not native.
    - Run them twice: with every processor, and with `--cpuset-cpus 0-1` as a stand-in for a two-core runner. Report which thread counts the daemon chose each time.
    - Also run the hook process's own budget test in the same container, both ways: `cargo test --release -p sm-cli --test hook_budgets s11_hook_process_warm_100_runs_inside_the_hook_budget -- --nocapture`. Compare it with stream D's 1.1 ms from before the binary linked the daemon and the embedder, and with B-124's 10.4 ms on Windows.
16. **`doctor`'s numbers with the real model (SPEC 5 rule 7; B-56, B-125).**
    - After item 8, run `spatial-memory doctor` against the real daemon once it is ready, 3 times, on Windows and in case 15's container both ways.
    - Record the bulk chunks per second, the median query latency, and the thread counts and batch size it printed.
    - Compare them with B-56's figures from `sm-embed` alone, and with case 13's migrate throughput.
    - Check that the hooks of a concurrent case are not slowed past their budget while `doctor` times the sessions. Run 20 prompt hooks during one `doctor` run and report their worst time.

### Part 3: build and test times for the CI proposal

Run this part on a quiet machine, after Part 2, with nothing else running. `docs/v3/CI_PROPOSAL.md` section 4 gives the method:

- **Linux:** Docker with `--cpus 4` and with `--cpus 2`, cold (a new target volume and a new registry volume) and warm (the target volume kept, one change in `sm-cli`);
- **Windows:** natively, with `cargo -j 4` and `-j 2`, in a fresh `CARGO_TARGET_DIR` outside the repository, cold and warm.

Time `cargo fmt --check`, `cargo clippy` and `cargo test` as `ci.yml` would run them. Put the figures in `RESULTS.md`; the lead copies them into CI_PROPOSAL. Remove the temporary target folders and volumes you created for this part.

### Part 4: `docs/v3/RESULTS.md`

1. **The machine.** The CPU, the physical and logical core counts, the memory, the OS and its version, the Docker version, and the build profile. Give the date and the commit.
2. **Budgets.** One row per SPEC 11 budget:
   - the target;
   - Windows: 50th percentile, 95th percentile and maximum;
   - Linux in Docker, with every processor and with two;
   - met or missed;
   - the case that produced it, and the command that reruns it.
3. **Acceptance tests.** One row per AT-1 to AT-17:
   - the test names, or the measurement case;
   - the result on Windows and on Linux;
   - for the real-model part, the figure.
4. **Other measured figures.** Opening the bulk session, project resolution, `consolidate` at 2,000 memories, migrate embedding throughput, and `doctor`'s numbers (case 16).
5. **Not measured.** Each item, with the reason. These include:
   - native Linux: the build machine has none, so every Linux figure is Linux in Docker;
   - a cold file cache on Windows: clearing it needs administrator rights;
   - macOS;
   - anything else not measured.
6. **CI build and test times** from Part 3.

## Tests

- Names cite their rule (B-40): `at3_`, `at9_`, `at13_`, `at14_`, `at15_`, `at16_` and `at17_` for Part 1; `s5_rule7_` for item 8; `s3_` or `s16_` for a fix outside an acceptance test.
- Part 1 tests run in full on both systems. Platform-only cases carry `#[cfg(unix)]` or `#[cfg(windows)]`, with a counterpart for the other system.
- The real-model test binary stays `crates/sm-embed/tests/real_model.rs`. Do not add another test that loads the model. The measurements in Part 2 are a program, not tests.
- Run the real-model test once at the end (`cargo test --release -p sm-embed --test real_model`) and copy its `test result:` line into the report.

## Report

`docs/v3/streams/k-report.md`, in RULES.md section 9's format. Section 5, "Measured budgets", may point to `RESULTS.md` instead of repeating it. List every missed budget there, with its figure.
