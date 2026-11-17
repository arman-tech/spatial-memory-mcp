# Results: version 3.0 against SPEC 11

Package K measured the finished `spatial-memory` binary against every SPEC 11 budget and acceptance test. This file gives each result, the machine it ran on, and the command that reruns it. A missed budget is recorded as a result. Nothing was tuned to meet one (SPEC 11; build rule "never tune a default to make a test pass").

**How the figures were taken.**

- `crates/sm-cli/examples/budgets.rs` runs the release binary as Claude Code would:
  - `spatial-memory daemon` with the real model;
  - `spatial-memory hook <event>` for each hook, with the hook's JSON on standard input;
  - `spatial-memory mcp` for `remember`, `recall` and the other tools, as JSON-RPC `tools/call`.
- Each case runs in a fresh `SPATIAL_MEMORY_HOME` in the system temporary folder, and removes it at its end. The model is read from the per-user cache and is never downloaded. Without the model the program prints one line and exits 0.
- Memory and threads are read from outside the daemon process, with B-58's methods:
  - Windows: `GetProcessMemoryInfo` (working set, peak working set, `PrivateUsage`) and a Toolhelp snapshot;
  - Linux: `/proc/<pid>/status` (`VmRSS`, `VmHWM`, `RssAnon`, `RssFile`) and `/proc/<pid>/task`.
- Times are printed, never asserted (B-58). p50, p95 and max are the 50th percentile, the 95th percentile and the maximum, by nearest rank.
- **Memory is judged on the working set (B-114).** Every memory figure gives the working set (the resident set on Linux) and private memory side by side. A budget missed on the working set and met on private memory is recorded as missed. 1 GB is 10⁹ bytes (953.7 MiB) and 2 GB is 2×10⁹ bytes (1,907.3 MiB).
- **Rerun.** Build with `cargo build --release -p sm-cli --bins --examples`, then run `cargo run --release -p sm-cli --example budgets -- <case>...` or `-- all`. The cases are `cold`, `idle`, `hook`, `import`, `open`, `kill`, `remember`, `at4`, `recall100k`, `consolidate`, `at2`, `at7`, `at8`, `at10` and `doctor`.
- **Rerun in Docker.** Fill the model volume once from the per-user cache, then build and run in the container (add `--cpuset-cpus 0-1` for two processors):

  ```
  docker run --rm -v "<model cache>:/m:ro" -v sm-v3-model:/model rust:1.95-trixie cp -r /m/. /model/
  docker run --rm -v "<repo>:/src:ro" -v sm-v3-cargo:/usr/local/cargo/registry -v <target volume>:/target -v sm-v3-model:/model:ro -e CARGO_TARGET_DIR=/target -e SPATIAL_MEMORY_MODEL_DIR=/model -w /src rust:1.95-trixie bash -c "cargo build --release -p sm-cli --bins --examples --locked && /target/release/examples/budgets <case>..."
  ```

  `<model cache>` is the folder that holds the pinned revision, `%LOCALAPPDATA%\spatial-memory\_models\embeddinggemma-2\daa72c51243991dfcaf9f9137d2c573d8f7790c0` on Windows.

## 1. The machine

| | |
| --- | --- |
| Processor | AMD Ryzen 9 9950X3D, 16 physical cores, 32 logical processors |
| Memory | 93.6 GiB (100,516,663,296 bytes), DDR5-6000 |
| Windows | Windows 11 Pro, version 10.0.26300 |
| Linux | Linux in Docker: Docker Desktop 29.9.3 (client and server), image `rust:1.95-trixie`, WSL2 (Windows Subsystem for Linux 2) kernel 6.6.87.2-microsoft-standard-WSL2. The virtual machine has 32 processors and 45.9 GiB (`MemTotal` 48,095,328 kB). The database and the model are on ext4 inside it, in the container's own folder and in the named volume `sm-v3-model` |
| Toolchain | Rust 1.95.0 (`rust-toolchain.toml`) |
| Build | `release` profile: `lto = "thin"`, `codegen-units = 1` (root `Cargo.toml`). The gates in section 3 of `k-report.md` are debug builds |
| Model | `embeddinggemma-2` at revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`, 768 dimensions, three files of 1,116,762,171 bytes in all |
| Settings | SPEC 15 defaults: 4 query threads and 12 bulk threads on 16 physical cores, batch 4, chunks of 256 tokens, bulk grace 10 s |
| Date | 2026-11-17 |
| Commit | Measured on `97a2742` (branch `arman-tech/stream-k-results`, from `v3` at `25c627e`) |

**Labels.**

- **Linux in Docker** figures are labelled "Linux in Docker (WSL2 virtual machine, ext4)". They are approximate, not native.
- **Two processors** means `docker run --cpuset-cpus 0-1`, a stand-in for a two-core runner. The container sees 2 processors in `nproc` and `Cpus_allowed_list`. `/proc/cpuinfo` still reports 16 cores, and `num_cpus::get_physical()` reads it, so the daemon chose 4 query and 12 bulk threads there too (section 5). Processors 0 and 1 are the two hardware threads of one physical core (`core id` 0 for both in `/proc/cpuinfo`). After the merge, B-133 caps the core count by `available_parallelism()`, so a daemon with two processors now runs 2 query threads and 1 bulk thread. The two-processor figures in this file were measured before that change.
- **Warm file cache.** Every cold start ran with the model files already in the file cache (section 5).

## 2. Budgets

One row per SPEC 11 budget, and one per condition of a budget. "Met" and "missed" judge the 95th percentile for time and the working set for memory.

Times are p50 / p95 / max in milliseconds. Memory is the working set / private memory in MiB. "Two processors" is `--cpuset-cpus 0-1`. Each row reruns with `cargo run --release -p sm-cli --example budgets -- <case>`, and in Docker as section 1 gives it.

| Budget | Target | Windows | Linux in Docker, all processors | Linux in Docker, two processors | Result | Case |
| --- | --- | --- | --- | --- | --- | --- |
| Hook, warm, end to end: 100 prompt hooks, 128-token prompt, 320 memories | p95 ≤ 300 ms | 100.1 / 105.0 / 110.3 | 81.2 / 91.6 / 93.7 | 98.0 / 101.0 / 103.4 | Met | `hook` |
| Hook, warm, during a bulk import of 2,000 chunks (AT-3): 100 prompt hooks | p95 ≤ 300 ms | 107.7 / 113.1 / 116.0 | 93.6 / 96.0 / 106.5 | 119.7 / 151.1 / 181.6 | Met | `import` |
| Hook while the bulk session opens | p95 ≤ 300 ms | 103.6 / 112.5 / 112.5 (17 hooks) | 90.3 / 94.0 / 94.0 (15 hooks) | not run | Met | `open` |
| Hook while `doctor` times the sessions: 20 prompt hooks, all overlapping a `doctor` run | p95 ≤ 300 ms | 125.9 / 170.4 / 170.9 | 107.7 / 155.6 / 155.8 | 119.0 / 164.9 / 166.3 | Met | `doctor` |
| Hook hard deadline: the daemon killed during a prompt hook, 20 runs, wall time | every run ≤ 1,500 ms, exit 0, empty output | 69.5 / 73.0 / 74.8 | 58.6 / 59.2 / 59.5 | 66.2 / 66.9 / 67.1 | Met, every run | `kill` |
| Daemon ready from cold: spawn to ready, 5 runs, warm file cache | ≤ 5,000 ms | 1,466.5 / 1,478.1 / 1,478.1 | 1,186.8 / 1,276.3 / 1,276.3 | 1,183.9 / 1,232.2 / 1,232.2 | Met | `cold` |
| `recall`, project scope, 100,000 chunks over 60 projects (about 1,667 in the project), 300 queries | p95 ≤ 250 ms | 27.1 / 31.9 / 36.6 | 33.9 / 39.2 / 47.8 | 35.5 / 40.0 / 45.7 | Met | `recall100k` |
| `recall`, project scope, 100,000 chunks in one project, 300 queries | p95 ≤ 250 ms | 99.7 / 107.8 / 117.3 | 101.3 / 106.0 / 113.8 | 104.4 / 109.0 / 112.5 | Met | `recall100k` |
| `recall`, all projects, 100,000 chunks over 60 projects, 300 queries | p95 ≤ 350 ms | 58.3 / 66.7 / 73.7 | 57.1 / 60.8 / 64.7 | 58.7 / 64.3 / 68.6 | Met | `recall100k` |
| `recall`, all projects, 100,000 chunks in one project, 300 queries | p95 ≤ 350 ms | 56.0 / 61.4 / 65.5 | 57.4 / 61.6 / 67.1 | 58.8 / 64.4 / 70.4 | Met | `recall100k` |
| `remember` of one chunk until vector `recall` finds it: 20 in a row, the first opening the bulk session | each ≤ 2,000 ms | 85.5 / 106.1 / 435.1 | 150.5 / 178.9 / 624.0 | 142.8 / 164.9 / 620.4 | Met, every run | `remember` |
| `remember` of one chunk until found, each after the bulk grace (session closed), 3 runs | each ≤ 2,000 ms | 474.8 / 555.9 / 555.9 | 408.6 / 413.1 / 413.1 | 382.6 / 388.0 / 388.0 | Met, every run | `remember` |
| Daemon memory, idle, 10 s after ready | ≤ 1 GB (953.7 MiB) | 626.9 / 615.3 | 650.1 / 623.3 | 651.0 / 623.7 | Met | `idle` |
| Daemon memory, idle, after the bulk grace has passed once | ≤ 1 GB | 633.5 / 618.7 | 710.1 / 681.9 | 712.5 / 683.1 | Met | `idle` |
| Daemon memory, idle, after 100 hooks | ≤ 1 GB | 695.4 / 680.3 | 778.7 / 675.8 | 783.3 / 676.5 | Met | `hook` |
| Daemon memory, idle, after 300 project-scope `recall`s at 100,000 chunks over 60 projects | ≤ 1 GB | 686.2 / 649.9 | **1,052.9** / 640.3 | **1,055.7** / 641.4 | Met on Windows; **missed** on Linux in Docker | `recall100k` |
| Daemon memory, idle, after 300 more all-projects `recall`s, over 60 projects | ≤ 1 GB | **1,058.5** / 651.8 | **1,490.6** / 640.3 | **1,493.0** / 641.4 | **Missed** | `recall100k` |
| Daemon memory, idle, after 300 project-scope `recall`s, one project of 100,000 chunks | ≤ 1 GB | **1,077.4** / 656.9 | **1,497.2** / 647.0 | **1,499.5** / 647.9 | **Missed** | `recall100k` |
| Daemon memory, idle, after 300 more all-projects `recall`s, one project | ≤ 1 GB | **1,400.2** / 653.1 | **1,497.4** / 647.3 | **1,499.7** / 647.8 | **Missed** | `recall100k` |
| Daemon memory, peak while indexing: AT-4, 50,000 characters | ≤ 2 GB (1,907.3 MiB) | 1,411.5 / 1,685.7 | 1,436.7 / 1,399.3 | 1,426.8 / 1,388.5 | Met | `at4` |
| Daemon memory, peak while importing 2,000 chunks with 100 hooks | ≤ 2 GB | 1,397.3 / 1,491.5 | 1,541.6 / 1,360.5 | 1,571.3 / 1,387.1 | Met | `import` |
| Threads during the import (AT-3): the peak of the threads the daemon itself creates, against its threads before the import plus 12 bulk threads plus connections in flight. The budget counts only threads the daemon creates (SPEC 5; B-131, B-141); the raw count of all threads is given for information | never more than the budget | The daemon's own (`sm-*`): 20 against 20, within. All threads, for information: 27 against 24 | `sm-*`: 21 against 22. All threads: 22 against 23. Within | `sm-*`: 21 against 22. All threads: 22 against 23. Within | Met (below) | `import` |

**Missed: idle memory after `recall` at 100,000 chunks.** The working set passes 1 GB after the 100,000-chunk queries in 11 of the 12 readings (four on Windows and four in each Linux run). Private memory stays between 640 and 657 MiB in every one. The rest is the mapped database file: reader connections map it (B-114), and the pages a search touches stay in the working set. On Linux, `RssFile` shows it directly: file-backed pages rose from 27 MiB at ready to 268 MiB once the daemon had opened the 750 MiB database, 413 MiB after the project-scope queries over 60 projects, and 850 MiB after the all-projects queries. Windows gives no file-backed figure, but the working set less private memory follows the same pattern. As B-114 says, a budget missed on the working set and met on private memory is recorded as missed. `mmap_size` and the search were not changed. The lead's decision is B-132. SPEC 11's budget is "idle after load", and that is met. These readings after heavy `recall` stay recorded as a missed condition, judged on the working set as B-114 says, for the owner to decide whether idle memory after use needs its own bound.

**For information: the raw thread count during the import on Windows.** The thread budget counts the threads the daemon itself creates (SPEC 5). The owner accepted that reading of B-131 on 17 November 2026 (B-141). The raw count is kept here because it exceeds the figure the daemon reports for itself: the 12 threads before the import (7 of its own, the main thread, the "linger pool" thread and three unnamed threads that start in `ntdll.dll`) plus 12 bulk threads, which is 24. At the peak, the daemon had 27:

- added by the bulk session: 11 `sm-embed-pool` threads and the `sm-embed-bulk` calling thread, which is 12, as configured;
- added by Windows: two more unnamed threads that start in `ntdll.dll`, which are workers of the Windows thread pool, and one unnamed thread whose start module the snapshot could not name.

The daemon's own threads (`sm-*`, with the connection threads) peaked at 20, exactly the budget of 7 + 12 + 1 connection. The three extra threads are not threads the daemon created, and the snapshot cannot say what they ran. AT-3 is judged on the threads the daemon creates, which stayed within (B-131, B-141); the raw count is reported here for information. On Linux the raw count stayed within. The Windows sampler took 322 samples in the 50.2 s import (a Toolhelp snapshot takes longer than the 50 ms interval); the Linux one took 1,955 in 98.1 s.

### What the hook's time is made of (Windows)

The hook budget is met with a wide margin. Of the hook's p50 of 100.1 ms, the query embedding takes about 85 ms:

| Part | p50 | p95 | max | Source |
| --- | --- | --- | --- | --- |
| Prompt hook, end to end | 100.1 ms | 105.0 ms | 110.3 ms | `hook`, 100 runs |
| The hook's own journal, `hook_latency_ms` | 87.2 ms | 91.0 ms | 95.6 ms | the hook's journal, 105 entries |
| `ambient_recall` through the socket, no hook process (embedding, project resolution, search, gate) | 86.5 ms | 88.4 ms | 92.4 ms | `hook`, 100 calls |
| The hook process and its pipe, by difference of the medians | 13.6 ms | | | |
| Query embedding at 128 tokens on 4 threads, median of 5 | 85.3 ms | | | `doctor_numbers` |
| Search, `Store::nearest`, project scope, 320 memories, top 5 | 0.2 ms | 0.2 ms | 0.2 ms | in process, 100 runs |
| Project resolution, `resolve_project` (B-109) | 0.2 ms | 0.2 ms | 0.4 ms | in process, 100 runs, in a plain folder and in this repository's work tree |

**Windows power throttling.** Windows applies power throttling (EcoQoS) to the background daemon within 2 to 4 seconds of its start: it has no window and its standard streams are closed. Throttled threads run at lower processor performance, and the query embedding then takes about 85 ms instead of 47 ms. The `hook` case also measures the same daemon with throttling turned off for its process (`SetProcessInformation(ProcessPowerThrottling)`), as a diagnostic, not as shipped:

| Diagnostic, throttling off | p50 | p95 | max |
| --- | --- | --- | --- |
| Prompt hook, end to end | 61.4 ms | 67.6 ms | 79.0 ms |
| `ambient_recall` round trip | 49.2 ms | 50.4 ms | 52.0 ms |
| Query embedding, median of 5 | 47.6 ms | | |

The throttled figures are the shipped ones, and they meet the budget. The throttling-off query median matches B-56's 47.2 ms, which `sm-embed` measured in a test process. The real-model test of section 3 shows the same split inside one process: its query median was 46.6 ms about 2 s after it started, and `doctor`'s query median was 85.0 ms about 12 s after it started. That test was started from a shell with no window too. This fits throttling, but the test cannot show it, because it does not read the process's throttling state. Linux has no such throttling. The daemon does not opt out in 3.0 (B-130; the owner's decision, B-142): the budget is met while throttled, and opting out trades battery for latency.

## 3. Acceptance tests

"Tests" are the automated tests, which run in both gates on the fake embedder unless marked as the real-model binary. "Case" is a `budgets` case with the real model. Every test named here passed on Windows and on Linux in Docker (`k-report.md`, section 3).

| AT | Tests added by package K (earlier tests in `TRACE.md`) | Real model: case or binary, and the figure | Windows | Linux in Docker |
| --- | --- | --- | --- | --- |
| AT-1 Embedding correctness | none; `at1_scores_match_the_reference` in the real-model binary | `cargo test --release -p sm-embed --test real_model`, once at the end. Windows: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.28s`; token fingerprint `434a9df05b5b5fc5` as the reference; scores Venus 0.675, Mars 0.853, Jupiter 0.747, Saturn 0.778, each equal to the reference; lowest cosine of a text alone against the padded batch 0.999999 | Pass | Pass, with the model in `sm-v3-model` and the same fingerprint, scores and cosine: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.32s` |
| AT-2 Retrieval regression | none | `at2`: the eval set's corpus stored through the daemon and embedded by it, then `spatial-memory doctor --eval`. nDCG@10 0.9865, hit@5 1.0000, MRR 1.0000 against a baseline of 0.9742, 1.0000 and 0.9667: "No regressions against the baseline", exit 0, on both systems | Pass | Pass |
| AT-3 Hook latency and safety | `at3_child_daemon_serves_only_when_asked`, `at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_is_killed_mid_request` (20 runs, wall time worst 26.5 ms on Windows and 21.5 ms on Linux, debug build, fake embedder), `at3_prompt_hook_exits_inside_the_deadline_when_its_daemon_died_before_it_connected` | `hook`, `import`, `open`, `doctor`: the prompt hook's p95 is between 91.6 and 170.4 ms in every case and run, against 300 ms. `kill`: worst 74.8 ms on Windows, 59.5 and 67.1 ms on Linux, against 1,500 ms. Threads in section 2 | Pass; the daemon's own threads stayed within the budget (section 2; B-141) | Pass |
| AT-4 Long input | none | `at4`: a 50,000-character `remember` through `spatial-memory mcp`: created, 47 chunks, all embedded in 3.6 s; peak working set 1,411.5 MiB on Windows and 1,436.7 MiB on Linux, against 1,907.3 MiB | Pass | Pass |
| AT-5 Concurrency | none | not needed: the store and daemon tests run 60 s with two readers and a writer | Pass | Pass |
| AT-6 Determinism | none | not needed: the tests use golden digests | Pass | Pass |
| AT-7 Clustering quality | none | `at7`: the 320 texts of `fixtures/spatial/texts.json` stored and embedded through the daemon, then `regions`. Windows: 5 regions, 16.2% unclustered, largest 16.9%. Linux: 5 regions, 13.4% unclustered, largest 18.4%. The committed vectors gave 9 regions, 7.8% and 17.8% (B-49) | Pass | Pass |
| AT-8 Migration | none | `at8`: `spatial-memory migrate --from fixtures/migration/v1_export_full.json`: 6 read, 6 inserted, 1 redacted, 6 pending; `status` said "0 waiting for embedding" after 539 ms on Windows and 614 ms on Linux; 6 memories in 3 projects, matching the report. `import`: 2,000 generated records, all embedded | Pass | Pass |
| AT-9 Redaction | `at9_remember_redacts_and_share_writes_no_credential_through_mcp` (through `spatial-memory mcp`: the shared file read from disk holds the marker and no credential; the refusal case writes no file) | not needed | Pass | Pass |
| AT-10 Model integrity | none | `at10`: the three model files copied to a temporary folder and the byte at offset 542,085,120 of `onnx/model.onnx_data` changed. The daemon never became ready: it answered `failed` after 0.5 s with `model_integrity`, naming the file, its SHA-256 and the pinned value, with the next action. `daemon.log` has the line, and `doctor` exits 1 with `state` and `model` failed, each with the next action. As SPEC 11 says (B-36, accepted by the owner as B-140), the file is never loaded: the daemon stays up in a failed state, reports the cause and embeds nothing | Pass | Pass |
| AT-11 Index mismatch | none | not needed | Pass | Pass |
| AT-12 Scope | none | not needed | Pass | Pass |
| AT-13 Capture from tools | `at13_capture_hooks_through_the_binary_store_strong_text_redacted_and_weak_text_not_at_all` | not needed: capture is the same with either embedder | Pass | Pass |
| AT-14 Capture from the transcript | `at14_pre_compact_and_stop_through_the_binary_store_only_what_post_tool_use_missed` | not needed | Pass | Pass |
| AT-15 Capture never blocks | `at15_capture_hooks_spool_inside_200_ms_with_no_daemon_and_a_started_daemon_stores_the_same` (with no daemon, the worst of 5 runs per hook is 9.9 to 17.1 ms on Windows and 1.8 to 2.1 ms on Linux, against 200 ms; debug build) | not needed | Pass | Pass |
| AT-16 Both tiers are saved | `at16_importance_does_not_change_recall_order_through_mcp` | not needed | Pass | Pass |
| AT-17 Version 2's hooks | `at17_ambient_recall_off_makes_no_contact_with_the_daemon` (the daemon's connection count does not move), `at17_the_block_through_the_binary_is_version_2s_format_with_ids_and_the_footer`, `at17_a_capture_over_the_index_cap_is_stored_whole_indexed_to_the_cap_and_journaled` | the `hook` case's 100 hooks each injected a block through the real daemon | Pass | Pass |

The `reindex` fix (B-118) adds `s16_reindex_keeps_a_captured_memorys_index_cap`, which AT-17's index cap depends on.

## 4. Other measured figures

No budget applies to these. Times are p50 / p95 / max in milliseconds unless a row says otherwise.

### Opening the bulk session (B-97), case `open`

| | Windows | Linux in Docker, all processors |
| --- | --- | --- |
| Time to open the bulk session: the median of 3 `doctor_numbers` calls with the session closed, less the median of 3 with it open | 365.9 ms (2,340.8 less 1,974.9) | 241.2 ms (2,322.5 less 2,081.2) |
| From `remember` of 16 chunks with the session closed to its first chunk embedded (opening plus one batch), 3 runs | 601, 614 and 626 ms | 453, 447 and 458 ms |
| Prompt hooks before, with the session closed | 99.2 / 106.7 / 108.9 (32 hooks) | 80.1 / 84.2 / 84.5 (39) |
| Prompt hooks started while the session was opening | 103.6 / 112.5 / 112.5 (17) | 90.3 / 94.0 / 94.0 (15) |
| Prompt hooks started while its chunks were embedding | 109.5 / 113.7 / 113.7 (8) | 91.7 / 101.9 / 101.9 (18) |
| Prompt hooks after, with the session open and idle | 98.4 / 101.5 / 103.8 (60) | 79.5 / 81.0 / 81.6 (75) |

Opening the session and embedding beside a hook adds about 5 to 11 ms to its median.

### The hook's parts, and project resolution (B-109), case `hook`

| | Windows | Linux in Docker, all processors |
| --- | --- | --- |
| `ambient_recall` through the socket, no hook process | 86.5 / 88.4 / 92.4 | 78.3 / 79.7 / 81.2 |
| The hook process and its pipe, by difference of the medians | 13.6 ms | 2.9 ms |
| Query embedding at 128 tokens on 4 threads, median of 5 | 85.3 ms | 76.7 ms |
| Search, `Store::nearest`, project scope, 320 memories, top 5 | 0.2 / 0.2 / 0.2 | 0.2 / 0.2 / 0.3 |
| `resolve_project` in a plain folder | 0.2 / 0.2 / 0.4 | 0.0 / 0.0 / 0.0 |
| `resolve_project` in this repository's work tree | 0.2 / 0.2 / 0.4 | 2.1 / 2.2 / 8.2 |

In Docker the repository is the Windows folder mounted into the container, so reading `.git` there crosses the file share.

### The hook process alone, `s11_hook_process_warm_100_runs_inside_the_hook_budget`

The test runs the release `spatial-memory hook` against a mock daemon, 100 runs for each of 8 events and clients.

| | All events, 800 runs: median / p95 / worst |
| --- | --- |
| Windows | 8.6 / 10.2 / 15.5 ms |
| Linux in Docker, all processors (two runs) | 1.5 / 1.7 / 2.2 ms; 1.5 / 1.7 / 2.2 ms |
| Linux in Docker, two processors | 1.4 / 1.5 / 2.6 ms |
| Stream D, before the binary linked the daemon and the embedder | 1.1 ms on Linux and 5.1 ms on Windows at p95 |
| B-124, after linking | 10.4 ms on Windows at p95 |

### `consolidate` at 2,000 memories (B-103), case `consolidate`

`consolidate` proposes merges only. One project of 2,000 embedded memories, 3 runs:

| | Windows | Linux in Docker, all processors |
| --- | --- | --- |
| Time | 153.9 / 180.8 / 180.8 (154, 145 and 181 ms) | 129.9 / 137.6 / 137.6 (138, 128 and 130 ms) |
| Proposed | 2 groups, 10 duplicates; truncated, 313 omitted | the same |
| Daemon memory after, working set / private (the bulk session from the seeding was still inside its grace) | 1,314.7 / 1,309.1 MiB | 1,409.8 / 1,291.3 MiB |

### Embedding throughput

| | Windows | Linux in Docker, all processors |
| --- | --- | --- |
| `migrate` of 2,000 generated version 1 records, one chunk each (two fixture texts, roughly 20 to 80 tokens), 100 hooks during the first 11 s | 50.2 s, 39.8 chunks/s | 98.1 s, 20.4 chunks/s |
| `migrate` of `v1_export_full.json` (AT-8, 6 chunks), from its start to "0 waiting", opening the session included | 539 ms, 11.1 chunks/s | 614 ms, 9.8 chunks/s |
| 2,000 `remember`s through `spatial-memory mcp`, one short chunk each, until all are embedded | 43.6 s, 45.8 chunks/s | 84.8 s, 23.6 chunks/s |
| AT-4's 50,000 characters, 47 chunks of up to 256 tokens | 3.6 s, 13.0 chunks/s | 4.2 s, 11.2 chunks/s |
| `doctor_numbers`, 16 chunks of 256 tokens | 13.7 to 13.9 chunks/s | 12.6 to 12.7 chunks/s |

Short chunks embed faster than full 256-token chunks, so migrate's rate is higher than `doctor`'s.

### `doctor`'s numbers (SPEC 5 rule 7; B-56, B-125), case `doctor`

`spatial-memory doctor` against the ready daemon, 3 runs, each exit 0 with 11 checks passed. The `embedder` line reads, for example, "bulk 13.9 chunks/s on 12 threads, batch 4 (16 chunks in 1153 ms); query median 85.3 ms at 128 tokens on 4 threads (5 runs); 16 physical cores".

| | Windows | Linux in Docker, all processors | Linux in Docker, two processors |
| --- | --- | --- | --- |
| `doctor`, end to end | 2,827 to 2,893 ms | 2,521 to 2,571 ms | 4,367 to 4,423 ms |
| Bulk, as printed: chunks/s on 12 threads, batch 4 | 13.9, 13.8, 13.7 | 12.6, 12.7, 12.7 | 6.0, 6.0, 6.0 |
| Query median at 128 tokens on 4 threads, as printed | 85.3, 84.8, 84.5 ms | 76.6, 76.9, 76.6 ms | 92.7, 93.7, 93.6 ms |
| `doctor_numbers` called directly, 3 times: bulk; query | 13.7, 13.8, 13.9; 85.3, 85.2, 84.7 ms | 12.7, 12.7, 12.7; 76.2, 76.2, 76.2 ms | 6.0, 6.1, 6.0; 92.3, 94.4, 94.2 ms |
| B-56, `sm-embed` alone in a test process | 17.26 chunks/s; 47.2 ms | 12.10 chunks/s; 79.8 ms | |
| The real-model test of section 3, `sm-embed` alone, `doctor_numbers` at the end of its run | 13.25 chunks/s; 85.0 ms | 12.22 chunks/s; 80.4 ms | |

On Linux the daemon's numbers match B-56. On Windows both are slower than B-56, the query median by 80%, and the real-model test gives the daemon's figures, not B-56's, when it times `doctor_numbers` 12 s after it started. Section 2's power throttling fits this: the same test's query median was 46.6 ms 2 s after it started, and the daemon's query median with throttling off was 47.6 ms. On Linux the same test gave 77.7 ms early and 80.4 ms late, with no such split. The bulk rate was not measured with throttling off.

**Two processors after B-133.** The lead reran the `doctor` case under `--cpuset-cpus 0-1` on the tree with B-133's cap, in the same image and with the same model volume. The daemon now reports "2 physical cores" and runs 2 query threads and 1 bulk thread, batch 4.

| Linux in Docker, two processors | Before B-133: 4 query and 12 bulk threads (the table above) | After B-133: 2 query and 1 bulk thread |
| --- | --- | --- |
| `doctor`, end to end, 3 runs | 4,367 to 4,423 ms | 7,048 to 7,228 ms |
| Bulk, as printed, 3 runs | 6.0, 6.0, 6.0 chunks/s | 3.3, 3.4, 3.4 chunks/s |
| Query median at 128 tokens, as printed, 3 runs | 92.7, 93.7, 93.6 ms | 102.6, 96.6, 98.7 ms |
| `doctor_numbers` called directly, 3 times: bulk; query | 6.0, 6.1, 6.0; 92.3, 94.4, 94.2 ms | 3.4, 3.4, 3.4; 96.0, 94.6, 94.6 ms |
| 20 prompt hooks while `doctor` times the sessions: p50 / p95 / max | 119.0 / 164.9 / 166.3 ms | 91.7 / 121.6 / 162.5 ms |

The cap obeys SPEC 5 rule 2, and the hooks during a bulk session are faster. The bulk rate on two processors falls from 6.0 to 3.4 chunks per second, because SPEC 15's bulk share, three quarters of the cores rounded down, is one thread on two cores. That default is not tuned here (build rule); the journal's bulk metric records it.

### Other

| | Windows | Linux in Docker, all processors |
| --- | --- | --- |
| Spawn to the daemon's first answer, while loading, 5 runs | 33.2 / 36.7 / 36.7 | 31.4 / 31.5 / 31.5 |
| The daemon's own `ready_seconds`, 5 runs | 1.43 to 1.45 s | 1.09 to 1.25 s |
| B-52, the Docker cold start with the model on the Windows file share | | 8.95 s; here 1.19 s at the median, with the model in a Docker volume |
| `remember` reply through `spatial-memory mcp`: it returns before embedding, with the chunk queued (B-27) | 4.1 / 6.7 / 7.7 (23 calls) | 6.1 / 10.3 / 15.0 (23) |
| The daemon's threads when idle | 12: 7 of its own, the main thread, the "linger pool" thread and 3 Windows thread-pool threads | 8: 7 of its own and the main thread |
| The 100,000-chunk store: database file, build time through `sm-store` | 750.2 MiB; 10.4 s spread, 9.0 s single | 750.2 MiB; 6.9 s spread, 6.2 s single |

## 5. Not measured

| Not measured | Why |
| --- | --- |
| Native Linux | The build machine has none. Every Linux figure is Linux in Docker in the WSL2 virtual machine, and is approximate |
| A cold file cache on Windows | Clearing the Windows file cache needs administrator rights. Every cold start, on Windows and in Docker, ran with the model files already in the cache |
| macOS | No macOS machine |
| A two-core runner's thread counts | `--cpuset-cpus 0-1` gives the container 2 processors, but `/proc/cpuinfo` still lists 16 cores. `num_cpus::get_physical()` reads that file and ignores the processor set, so the daemon chose 4 query and 12 bulk threads on 2 processors. The two-processor figures therefore measure 16 threads on 2 processors, not the counts a real two-core machine would get. SPEC 5 says "Never run more threads than cores". B-133 now caps the count by `available_parallelism()`. Only the `doctor` case was rerun under the cap: 2 query threads and 1 bulk thread (section 4, "`doctor`'s numbers"). The other two-processor cases were not rerun |
| The cases outside case 15 with two processors | Case 15 names cases 1 to 4, 7 and 8; `kill`, `recall100k` and `doctor` (case 16) were added. `open`, `consolidate`, `at2`, `at7`, `at8` and `at10` ran with all processors only |
| Bulk throughput on Windows with power throttling off | Only the query side was measured with throttling off, as a diagnostic |
| The fallback model, Arctic Embed M (D-5, B-57) | It has never been loaded. Package K does not download models |
| A run on a real version 1 store | Not required: version 3 is a clean break, and SPEC 12 no longer asks for it (B-139). The fixture and 2,000 generated records were migrated |
| The evaluation on real memories (SPEC 11) | The owner's task. `doctor --eval` was run on the fixture set |
| `recall` over real vectors at 100,000 chunks | The stored vectors are synthetic unit vectors. Exact search costs the same whatever the vectors hold; the query was embedded by the real model |
| The model step of `setup` and a first download | Package K never downloads the model |
| Claude Code and Cursor themselves | The binary was run as they run it, with their hook payloads and JSON-RPC, but not by the clients |
| Idle exit with the real model | The idle timeout is tested with the fake embedder; no case waited for it with the real model |

## 6. CI build and test times

`ci.yml`'s three jobs as CI_PROPOSAL section 3 gives them, timed on the machine in section 1 while nothing else ran, on `97a2742`. The lead copies these figures into CI_PROPOSAL section 4.

**Method.**

- **Linux in Docker.** Each job runs in its own `docker run --rm --cpus N rust:1.95-trixie`, with the source mounted read-only and its own registry and target volumes. `--cpus` sets the container's processor quota (`cpu.max` 400000 100000 for 4, 200000 100000 for 2). `nproc` still says 32, but `std::thread::available_parallelism()` returns 4 and 2, so Cargo runs 4 or 2 jobs and the test harness runs 4 or 2 test threads. The quota covers every process in the container, including the daemons the tests start.
- **Windows, natively.** `cargo test -j N --workspace --all-features --locked` with `RUST_TEST_THREADS=N`, and a new `CARGO_HOME`, `CARGO_TARGET_DIR` and `ORT_CACHE_DIR` in a temporary folder outside the repository. `-j` and `RUST_TEST_THREADS` limit Cargo's jobs and the test threads, but not the processors each process may use. The Windows figures are therefore a looser lower bound than the Linux ones.
- **No model.** `SPATIAL_MEMORY_MODEL_DIR` points at an empty folder on Windows, and the containers have none, so `real_model` skips itself, as in CI.
- **Cold.** New, empty registry and target folders. This is what a cache miss costs. It includes downloading the crates and the ONNX Runtime archive.
- **Warm.** The same folders, kept, and one comment line appended to `crates/sm-cli/src/lib.rs` (removed afterwards). This rebuilds the `sm-cli` library, the binary and every `sm-cli` test binary. It is what a typical push costs with a cache hit.
- **Steps.** "Toolchain" is `rustup component add clippy rustfmt`. The image already holds Rust 1.95.0, so this step does not show what a hosted runner's `rustup toolchain install` would take. "Fetch" is `cargo fetch --locked`. "Test" is `cargo test --workspace --all-features --locked`: building and running. "Job" is the whole job's wall time, including the container start on Linux.
- Every step exited 0, and every test step reported `test result: ok` for all 36 test binaries, the doc tests included.

**Times in seconds.**

| Job | Step | Cold, 4 processors | Cold, 2 processors | Warm, 4 processors | Warm, 2 processors |
| --- | --- | --- | --- | --- | --- |
| `lint` (Linux) | toolchain | 1.2 | 1.1 | 1.5 | 1.3 |
| | fetch | 5.3 | 5.5 | 0.8 | 0.6 |
| | `cargo fmt --check` | 2.1 | 2.2 | 2.7 | 2.2 |
| | `cargo clippy` | 66.1 | 124.3 | 2.9 | 2.6 |
| | **job** | **75.6** | **134.0** | **15.6** | **7.5** |
| `test-linux` | toolchain | 1.2 | 1.1 | 1.1 | 1.1 |
| | fetch | 5.9 | 6.3 | 0.6 | 0.6 |
| | `cargo test` | 402.7 | 531.3 | 291.3 | 384.1 |
| | of which building | not split | not split | 13.3 | 20.6 |
| | of which running (sum of the binaries' times) | not split | not split | 276.4 | 361.7 |
| | **job** | **410.3** | **539.5** | **293.8** | **386.6** |
| `test-windows` | fetch | 11.0 | 16.4 | 0.3 | 0.3 |
| | `cargo test` | 491.4 | 771.6 | 333.0 | 346.4 |
| | of which building | 209 | 474 | 13.3 | 21.7 |
| | of which running (sum of the binaries' times) | 279.4 | 293.9 | 317.8 | 322.7 |
| | **job** | **502.7** | **788.3** | **333.7** | **347.0** |

"Building" is the time on Cargo's `Finished` line. "Running" adds the `finished in` times of the 36 binaries. The cold Linux runs printed neither line. If their tests ran as long as they did warm, building took about 126 s with 4 processors and about 170 s with 2.

**Per push.** `lint` and `test-linux` run in parallel, and `test-windows` beside them. Runner time is the sum of each operating system's jobs; the wait is the longest job.

| | Cold, 4 processors | Cold, 2 processors | Warm, 4 processors | Warm, 2 processors |
| --- | --- | --- | --- | --- |
| Linux runner time (`lint` + `test-linux`) | 485.9 s (8.1 min) | 673.5 s (11.2 min) | 309.4 s (5.2 min) | 394.1 s (6.6 min) |
| Windows runner time (`test-windows`) | 502.7 s (8.4 min) | 788.3 s (13.1 min) | 333.7 s (5.6 min) | 347.0 s (5.8 min) |
| Wait for the push's result | 502.7 s (8.4 min) | 788.3 s (13.1 min) | 333.7 s (5.6 min) | 386.6 s (6.4 min) |

**What the figures show.**

- **A warm push is dominated by running the tests, not by building them.** Building took 13 to 22 s. Running took 276 to 362 s. Four binaries account for about four fifths of it: `hook_budgets` (61 to 74 s), `acceptance` (about 61 s), `sm-store` (about 60 s) and `sm-share` (28 to 44 s). `cargo test` runs the binaries one after another, so most of that time is these binaries' own timed waits, not processor time.
- **Going from 4 to 2 processors** roughly doubles a cold build (clippy 66 to 124 s; the Windows build 209 to 474 s). On Linux it adds about 85 s to a warm test run. On Windows it adds 5 s, because `-j` does not confine the test processes.
- **These are lower bounds.** This processor is faster per core than a hosted runner's, a Docker quota is time on 32 processors rather than dedicated cores, and the hosted toolchain install and cache restore are not included. CI_PROPOSAL section 4 multiplies them by 2 until real runs give actual minutes.
