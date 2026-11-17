# Version 3.0 build progress

Read this file and `DECISIONS.md` first after a restart or a context compaction.

## Current phase

Phase 3. Phase 0 is committed on `v3` as `97426c2`, and the contracts are frozen (CONTRACTS.md section 1). The stream specifications are in `docs/v3/streams/` (`stream-a.md` to `stream-d.md`, `stream-g.md`, and `RULES.md`, which is appended to each). Phase 1 runs through Orca with at most four workers: A, B, C and G first, then D when a slot frees. Streams C, B, G, A and D are merged, so phase 1 is complete. Phase 2's specifications are written (`stream-e.md`, `stream-f.md`). Streams F and E are merged, so phase 2 is complete. Stream E was sent back once after review; the lead fixed the review's `sm-store` defect on `v3` (B-94). Phase 3, the lead's integration, has started; its list is under "Left for the lead in phase 3" below. Packages I and H are merged (`237af2c`, `b4053ff`) and both gates pass on the merged tree. Package J is merged (`31bc7ce`), and both gates pass on the merged tree: 619 tests on Windows and 629 on Linux in Docker, with format and clippy clean. Package K is merged (`7b89635`). The lead's fix after it caps the thread budget's core count by the processors the process may use (B-133). K's worktree target folder is removed. Two independent reviews read the code merged after stream E: packages H to K and the lead's commits. The lead fixed what they found (B-134 to B-138) and refuted one finding (B-136). After those fixes both gates pass: 648 tests on Windows and 658 on Linux in Docker, with format and clippy clean. The lead's final report followed. The owner then removed the migration of a real version 1 store from SPEC 12's exit condition. Version 3 is a clean break, and no memories are carried over (B-139). Every acceptance test passes on the build machine, AT-10 as SPEC 11 now states it (B-140), and Linux is verified in Docker, so the exit condition is met. The owner's decisions of 17 November 2026 are B-140 to B-143, and the documentation they touch is B-144. What still stands before a release is under "Follow-ups before the release" and "Open problems" below.

## Phase 0 checklist

| Step | State | Evidence |
| --- | --- | --- |
| Trivial `cargo build` through PowerShell finds the MSVC linker | Done | rustc 1.95.0, `Hello, world!` (SPIKE.md, P0-0; B-2) |
| Model copied into the per-user cache, SHA-256 verified | Done | All three hashes match SPEC 5 (SPIKE.md, P0-1; B-3) |
| `rust-toolchain.toml` pinned | Done | Channel 1.95.0, rustfmt and clippy, minimal profile (B-1) |
| Spike a: ort loads EmbeddingGemma 2, self-test scores | Done | SPIKE.md a |
| Spike b: vec1 via cc, rusqlite, exact search, vendored with SHA-256 | Done | SPIKE.md b; `vendor/vec1/` (B-7) |
| Spike c: local socket between processes | Done | SPIKE.md c |
| Spike d: rmcp stdio one-tool server called by an MCP client | Done | SPIKE.md d |
| Spike e: exec-form hook by bare name, 50 runs start to exit | Done | 5.4 ms median, 6.0 ms 95th percentile (SPIKE.md e; B-6) |
| Spike f: `claude plugin validate`; plugin hooks on Windows; mcp_tool against exec form | Done | SPIKE.md f (B-5, B-6) |
| Python tree removed, docs archived, new CLAUDE.md | Done | Python sources deleted; docs moved to `docs/archive/python/`; root `CLAUDE.md` rewritten for v3 |
| Workspace, `sm-core`, socket schema, CONTRACTS.md, fixtures | Done | `crates/sm-core`; `docs/v3/CONTRACTS.md`; `fixtures/` (B-10, B-13) |
| `cargo test --workspace` green on Windows; Linux build in Docker | Done | Windows: 96 pass (one test is Unix-only), clippy and fmt clean. Linux (Docker `rust:1.95-trixie`): 97 pass, clippy clean |
| DECISIONS, TRACE, LICENSES | Done | DECISIONS B-1 to B-41; TRACE.md; LICENSES.md |
| Phase 0 commit on `v3` | Done | `97426c2` |

## Phase 1 preparation

| Step | State | Evidence |
| --- | --- | --- |
| Pins for the model download and the command line | Done | ureq 3.5.2 with rustls, clap 4.6.7 (B-42); LICENSES.md updated |
| `sm-cli` takes `sm-embed` in phase 3 | Done | B-43 |
| Auto-start lock held by the daemon for life | Done | CONTRACTS.md 3.1 (B-44) |
| Stream specifications A, B, C, D, G and RULES.md | Done | `docs/v3/streams/` |

## Phase 2 preparation

| Step | State | Evidence |
| --- | --- | --- |
| Whole-store `export` and `import` given to stream F, as JSON Lines | Done | B-77; CONTRACTS.md section 2 |
| How `sync` reads the repository and records what it read | Done | B-78; `Store::record_sync`, `SyncMark` |
| The daemon reaches `sm-share` through E's `SharingOps` trait; projects by identity key | Done | B-79; `Store::ensure_project`, `ProjectRecord`, `project_records`, `project_links`; CONTRACTS.md section 2 |
| Store additions tested | Done | `s9_record_sync_makes_a_memory_shared_with_its_file`, `s10_ensure_project_finds_or_creates_by_identity`, `s4_project_links_lists_every_stored_link` |
| Stream specifications E and F | Done | `docs/v3/streams/stream-e.md`, `stream-f.md` |

## Streams

| Stream | Crate | Phase | Depends on | Orca task | Worker | State |
| --- | --- | --- | --- | --- | --- | --- |
| A store | `sm-store` | 1 | | `task_c71067fb0f26` | `ctx_5dba5f9a9d4d`, worktree `stream-a-store` | **Merged.** Store parts of AT-5, AT-11 and AT-12 pass; D-2, D-3 and D-8 covered; 50 tests on Windows and Linux (Docker); search budgets measured on Windows by `examples/search_budget.rs`; B-69 to B-76, with a lead change to merge (B-73); report `docs/v3/streams/a-report.md` |
| B embed | `sm-embed` | 1 | | `task_8922c54475a3` | `ctx_eac8a4efaf42`, worktree `stream-b-embed` | **Merged.** AT-1 passes; AT-2 (embedding path), AT-4 (embedding path) and AT-10 (load and download) pass; D-4, D-6 and D-10 covered; 54 unit tests and the real-model binary on Windows, 54 on Linux (Docker); B-50 to B-58; report `docs/v3/streams/b-report.md` |
| C spatial | `sm-spatial` | 1 | | `task_f3585000cbb0` | `ctx_613e314199cd`, worktree `stream-c-spatial` | **Merged.** AT-6 and AT-7 pass on Windows and Linux (Docker); 47 tests; B-45 to B-49; report `docs/v3/streams/c-report.md` |
| D clients | `sm-cli` | 1 | | `task_2e39969f24a8` | `ctx_37824ef4196e`, worktree `stream-d-clients` | **Merged.** The hook parts of AT-3, AT-14, AT-15 and AT-17 pass; spike decision 2 covered; the hook client, auto-start, MCP proxy, `setup`, `doctor`, the operation subcommands and the plugin package; 89 tests on Windows, 88 on Linux (Docker); B-80 to B-85, with a lead change to drop `binary_path` (`b59e3c7`, B-85); report `docs/v3/streams/d-report.md` |
| G capture | `sm-capture` | 1 | | `task_24c15023b6ca` | `ctx_e485d1bf5b49`, worktree `stream-g-capture` | **Merged.** Library parts of AT-9, AT-13, AT-14, AT-16 and AT-17 pass; 726 of 726 ported fixture cases pass, 278 not applicable with reasons; 33 tests on Windows and Linux (Docker); B-59 to B-68; report `docs/v3/streams/g-report.md`; the lead added the AWS secret access key pattern after stream F found the gap (`89d2a16`, B-86) |
| E daemon | `sm-daemon` | 2 | A, B, C, G (the capture pipeline) | `task_287f5c859fe6`; follow-up `task_b3fe611ba8f6` | `ctx_8fa989b4ae90`, then `ctx_063deecf2119`, worktree `stream-e-daemon` | **Merged.** Sent back once: the first review found a capture lost for good when its processing failed, a capture lost at shutdown, a project cache never refreshed, a macOS bind, an unchecked `/tmp` socket owner and smaller defects; the lead fixed the review's `sm-store` defect (B-94). The second delivery fixed the rest. The daemon sides of AT-3, AT-4, AT-5 and AT-9 to AT-17 pass; AT-5, AT-11 and AT-12 are complete through the daemon; spike decision 1 is covered. 91 tests: 86 pass on Windows and 89 on Linux (Docker) on the merged tree. The second review found no defect that must be fixed before the merge; its two should-fix defects (a capture session blocked by a candidate that always fails, and the owner check after `connect` passing a missing socket), an earlier gap (the Windows client does not check who serves the pipe) and its notes are the first items under "Left for the lead in phase 3". The lead added a barrier to one test, which read the journal before the worker had written it and failed once on Linux under load. B-95 to B-111, with the lead's rulings on E's contract requests (B-95, B-100, B-102, B-108, B-111) and on 0BSD (B-107); report `docs/v3/streams/e-report.md` |
| F sharing and migration | `sm-share` (B-8, B-77) | 2 | A; G's scanner as a dev-dependency | `task_b42a480547e0`, follow-ups `task_ccc13fda2ff3` and `task_66be243b1a99` | `ctx_d0c5203bb20c`, follow-up dispatches `ctx_38bc9fe3f74e` and `ctx_d1d0d55f523e`, worktree `stream-f-share` | **Merged.** The library parts of AT-8 and of AT-9's `share` side pass; `share`, `sync`, `migrate`, `export` and `import`; 89 tests on Windows and Linux (Docker). Sent back twice after review. First: import lost a supersession target's status in the normal file order, and a retracted superseder kept its target superseded for good (fixed in `068e32e`). Second: a sync stopped before its settle step left a target superseded with no live superseder, and a merged memory could be made active again (fixed in `a72135d`). Each round was checked by the same reviewer. B-87 to B-93, two of them lead decisions (B-87 liveness, B-88 redaction); report `docs/v3/streams/f-report.md` |
| H daemon hardening | `sm-daemon` | 3 | E | Agent tool (not Orca) | worktree `stream-h-hardening` | **Merged** (`b4053ff`). The second E review's two should-fix defects are fixed. Capture failures are permanent or bounded, and the bulk queue strikes a chunk only in a round in which another was stored (B-116). Both ends check each other's user, the fallback folder is private and the bind folder's name random, and the profile folder is user-only on Unix (B-117). A title change re-embeds, and a view that shed a field says it was cut (B-118). AT-16 passes. Spike decision 5 is covered, and so is the drift guard over every unchecked struct. 108 sm-daemon tests on Windows, 118 on Linux (Docker). Two stream E tests changed, neither weakened (B-116). Open: `reindex` re-chunks a captured memory without its capture cap (B-118, given to K). Report `docs/v3/streams/h-report.md` |
| I store and sharing follow-ups | `sm-store`, `sm-share` | 3 | A, F | Agent tool (not Orca) | worktree `stream-i-store` | **Merged** (`237af2c`). B-87's rule holds on every local path; `share` refuses a merged memory; `sync` lists marks only and records each mark in its revision's transaction; import keeps `updated_at` (B-115). B-74 is fixed by an index of vec1 blocks with room (B-113). B-75 is fixed by mapped readers and a project-scope subquery, 71.1 ms at p95 on Windows (B-114). sm-store 67 and sm-share 92 tests. Report `docs/v3/streams/i-report.md` |
| J wiring | `sm-cli`; `sm-daemon`'s sharing adapter | 3 | D, E, F | Agent tool (not Orca) | worktree `stream-j-wiring` | **Merged** (`31bc7ce`). The `daemon` subcommand, with an exit code and a log line for each way out (B-119). Every client goes through `sm_daemon::connect` (B-119). On Windows the detached daemon inherits no handle (B-120). The model step runs over `sm-embed`, with `SPATIAL_MEMORY_MODEL_URL` for a mirror (B-121). The production `ShareAdapter` over `sm-share` is wired (B-122). `status` exits 1 on a failed model, an index mismatch or an error reply. The client makes the profile folder user-only on Unix (B-123). D-1 is covered. sm-cli has 115 tests on Windows and 114 on Linux (Docker), and sm-daemon 113 and 123. ONNX Runtime is now in every hook process, which adds about 4 ms (B-124). Open: `doctor` does not yet print the embedder's numbers (SPEC 5 rule 7; B-125, given to K). Report `docs/v3/streams/j-report.md` |
| K end to end and real-model measurements | tests in `sm-cli` and `sm-daemon`; `RESULTS.md`; `doctor_numbers` (B-125) | 3 | H, I, J | Agent tool (not Orca) | worktree `stream-k-results` (branch `arman-tech/stream-k-results`, from `25c627e`) | **Merged** (`7b89635`). The hooks run end to end through the binary against the `daemon` subcommand: AT-3 with the daemon killed mid-request, AT-13, AT-14, AT-15 (spooled while stopped, then drained) and AT-17. `remember`, `recall` and `share` run end to end through `spatial-memory mcp` (AT-9, AT-16). `reindex` keeps a captured memory's index cap (B-118). The `doctor_numbers` operation, printed by `doctor` (SPEC 5 rule 7; B-125, B-126). The idle-exit line says "1 minute". Every SPEC 11 budget and AT-1 to AT-17 measured with the real model on Windows and on Linux in Docker by `examples/budgets`, and the CI build and test times (`RESULTS.md`; B-129). Two results missed: idle memory after `recall` at 100,000 chunks, on the working set (B-132), and the raw thread count on Windows during an import, over by 3 (B-131). Report `docs/v3/streams/k-report.md`. After the merge the lead capped the thread budget's core count by the processors the process may use (B-133) |

Phase 3 packages H, I and J run as Agent-tool subagents in parallel, each in its own worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at the commit that adds their specifications. Each spec is the package file followed by `RULES.md`. Package K (end-to-end hook runs, real-model measurements, RESULTS.md) follows J.

The lead's phase 3 documents, drafted while H, I and J run:

- **`README.md`, the 3.0 README.** The 1.x README moved to `docs/archive/python/README.md`. Its `status` and `daemon` rows were checked against J's merge, and it lists `SPATIAL_MEMORY_MODEL_URL` (B-121). Done after K: the `doctor` row names the embedder's numbers (B-125), and the memory sentence gives K's readings, including the working set after `recall` at 100,000 chunks (B-132).
- **`docs/v3/MIGRATION.md`, step 3,** reads `status` as J's binary prints it, and says when `status` exits 1.
- **`docs/v3/MIGRATION.md`.** It is checked against version 1's `export_memories`. An export path resolves against the Python server's working directory and must be inside `exports/` or `backups/`. `project: "*"` is needed to export every project.
- **`docs/v3/CI_PROPOSAL.md`.** Runner prices and hardware were read on 17 November 2026. The repository is public, so standard runners are free. The replacements for the three Python workflows are proposed. Done after K: section 4 gives package K's measured build and test minutes, cold and warm, at 4 and 2 processors, doubled for planning, with the cost per push if the repository becomes private (B-129).

- **`CONTRIBUTING.md`, rewritten for 3.0.** It points to `CLAUDE.md` and the `docs/v3` documents and gives the cargo commands and the rules for a change. The 1.x guide moved to `docs/archive/python/CONTRIBUTING.md`.

- **`SECURITY.md`, rewritten for 3.0** and committed after package H, whose peer checks and user-only folders it describes. The 1.x policy moved to `docs/archive/python/SECURITY.md`.

Follow-ups before the release:
- Done: `CHANGELOG.md` has its 3.0.0 entry (B-9). It has no entries for 1.11.x or 2.0.0. The entry describes a clean break, with migration optional and tested on the export fixtures only, and says where the history before 3.0 is kept (B-144).
- Done: `MARKETING.md`, which describes 1.x, is moved unchanged to `docs/archive/python/` (B-144).
- The three Python workflows (`tests.yml`, `lint.yml`, `publish.yml`) are replaced, in one change, before v3 is merged or released. 3.0 is not announced until the owner has reviewed the runner cost (CI_PROPOSAL.md sections 1 and 4). No workflow is touched on `v3`.

Orca run: `run_991d66c5f298`. At most four workers run at once. Workers use Claude Opus 5.5 (`claude-opus-5-5`) at effort `xhigh`, each in a child worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at `ea82480`. Each task's spec is its stream file followed by `RULES.md`. Streams E and F branch from `v3` at `9387a08`, the commit that adds their specifications.

## Open problems

- **Arctic feature licence.** The `arctic-embed-m` feature (D-5) brings `option-ext` 0.2.0, which is MPL-2.0, through `fastembed` → `hf-hub` → `dirs`. The default build does not contain it. Flagged for the owner in LICENSES.md and B-41.
- **Permissive licence not on the list.** Done: `webpki-roots` (CDLA-Permissive-2.0), linked into the default binary for the model download, is accepted by the owner for version 3.0 (B-143; LICENSES.md). Using the operating system's certificate store instead was reported, not done. `webpki-root-certs` and `libfuzzer-sys` left the tree with B-50, and so did the system OpenSSL crates.
- **Stream B follow-ups (B-52 to B-58).** Model load in Docker took 8.95 s against the 5-second cold-start budget, read over the Docker Desktop file share at about 230 MB/s. Done by package K: with the model in a Docker volume, spawn to ready takes 1,186.8 ms at the median and 1,276.3 ms at worst on Linux in Docker, and 1,466.5 and 1,478.1 ms on Windows (RESULTS section 2), so B-52's verify stamp stays unadopted. Native Linux is not available on the build machine. Verify on a cold file cache was not measured on Windows: clearing the file cache needs administrator rights. A live download from Hugging Face ran once, on Windows on 17 November 2026: 3 files, 1,116,762,171 bytes, 29.1 s, every SHA-256 passing (B-144); the tests use a local server. The Arctic fallback is built and unit-tested but never loaded or run (D-5 Partial). The daemon's refusal on a bad model is stream E's AT-10 test.
- **Stream G follow-ups (B-59 to B-68).** The capture pipeline has not run end to end; streams D and E own the rest of AT-9, AT-13, AT-14, AT-16 and AT-17, and stream D must cover the fixture cases G marked as stream D's (stdin, stdout, the working-directory check, ambient recall). Stream E counts `capture_cap_hit_total` on the index cap, not on G's extraction cap (B-65). Python's `\s` matches U+001C to U+001F and the `regex` crate's does not (B-59). Linux capture times are debug only (B-68). SPEC 16 (Stop and the transcript, B-61) and SPEC 3 and 14 (the spool's crate, B-66) are out of date.
- **Stream A follow-ups (B-69 to B-76).**
  - Done: bulk vec1 inserts are linear (B-113, closes B-74). A database made by a development build before `1008eb5` lacks the index and stays slow until it is recreated.
  - Done for the store: project scope in a single project of 100,000 chunks is 71.1 ms at p95 on Windows and 67.1 ms on Linux in Docker (B-114, closes B-75). Package K measures `recall` end to end on Windows and on Linux in Docker, and reports private memory and the working set separately, because mapped readers grow the working set up to 2 GiB (B-114).
  - Two memory-vector functions disagree when a memory's chunks cancel out: the store's falls back to the first chunk, `sm_spatial::memory_vector` returns an error. Stream E uses the store's (B-76).
  - No backup retention: backups accumulate in `backups_dir()` (B-71).
  - 79 temporary folders (63.3 MB) leaked by early stream A test runs remain in `%LOCALAPPDATA%\Temp`, outside the repository. Deleting them needs the owner's permission; current runs leave none.
- **Protocol gaps accepted for 3.0 (B-37).** The injection race (narrowed by B-104: the daemon records an injection only after it writes the reply), and `LinkView.stale` left absent when ancestry cannot be decided: stream A's `head_only` oracle settles only a commit equal to the head or a prefix of it (B-76).
- **Model step of `setup` and `doctor`.** Done by package J: the `ModelStep` in `crates/sm-cli/src/model_step.rs` runs over `sm-embed`, and `Unwired` is removed (B-121). The tests use a local server; one live download from Hugging Face has been run (B-144).
- **Package J follow-ups (B-119 to B-125).**
  - Done by package K: `doctor` prints bulk chunks per second and the median query time from the daemon's own sessions (SPEC 5 rule 7; B-125). "Not measured" while the daemon loads is not a failure (B-126).
  - On Unix, a descriptor the hook inherited without close-on-exec would still reach the daemon. It is not tested (B-120).
  - Every hook process is about 4 ms slower on Windows since ONNX Runtime was linked: p95 10.2 ms in K's run. Package K measured it on Linux in Docker: p95 1.7 ms. The cause on Windows is not measured, and the budget holds (B-124).
  - Done by package K: the idle-exit log line says "1 minute" for one minute.
  - J's test runs keep a sandbox whose daemon lock was not yet free, under `%TEMP%\spatial-memory-cli-tests\sb-*`; later runs delete them after an hour (B-120). 92 were kept after J's last run.
- **Stream D follow-ups (B-80 to B-85).**
  - `CLAUDE_CONFIG_DIR` is not honoured: `setup`, `doctor` and the plugin check read `~/.claude` and `~/.claude.json` only (B-84).
  - The real `claude mcp add` and `claude mcp remove` were never run; the tests use a fake runner (B-84).
  - Whether Claude Code exports a plugin option's default when the user never saved a value is unverified; recall stays on either way (B-85).
  - `claude plugin validate` runs only where `claude` is on `PATH`, so it is skipped in Docker (B-85).
  - The AT-3 hung-daemon case is 1,422.8 ms at worst on Windows against 1,500 ms; a loaded machine could push it over (B-80).
- **Stream F follow-ups (B-87 to B-93).**
  - Done: a forgotten local superseder, and one retracted outside `sync`, restore their targets in the same transaction (B-115).
  - A shared target whose own file says `superseded`, with no live superseder, has its file read again on every sync while that lasts (B-87).
  - Import restore: a target already stored and changed locally since gets its export line's status back when a superseder in the file is newly inserted (B-92; still open after B-115).
  - Redaction does not scan project values, project lines or raw ids (B-88).
  - A retracted copy blocks the same text under a new id in `sync` until it is forgotten or its file restored (B-90).
  - Measured: a sync in which all 10,000 files change takes 18.5 s on Windows, still one transaction per file (B-115). SPEC 15 sets no budget.
  - Done: the export and import round trip is byte-identical apart from the header (B-115).
  - Inserting a superseder that is already retracted, or one in another project, still marks its target `superseded`. No current path leaves a wrong status, but the store does not enforce it (B-115).
  - Withdrawn: SPEC 12 no longer asks for a migration of a real version 1 store. `migrate` is kept, and it has been tested on the export fixtures only (B-139).
- **Stream E follow-ups (B-95 to B-111).**
  - Done: spike decision 5 is tested (B-117).
  - macOS: the private-folder bind, the socket owner check and the `LOCAL_PEERCRED` peer check are tested on Linux only, and SPEC 13's `interprocess` risk stays open there (B-95, B-117).
  - Done: AT-10 follows B-36, which the owner accepted: with a changed byte the daemon starts and stays `failed`, and SPEC 11 now says so (B-96, B-140).
  - Covered by code only: rank numbering after a memory vanishes (B-103) and the log line for a failed injection record (B-104). The drain skipped while `failed`, the spill at 1,024 and the socket checks in `connect` are now tested (B-116, B-117).
  - On Windows, an elevated daemon, a reused process id and a server process the client may not open are not tested (B-117).
  - A capture pipeline that panics on the same text every time, and the cost of a lone poison chunk over a long run, are not measured (B-116).
  - A capture processed twice after a crash is stored once but journals its capture metrics twice (B-99, B-108).
  - Measured by package K: opening the bulk session costs 365.9 ms on Windows and 241.2 ms on Linux in Docker. The 10-second grace stays provisional, with its journal metric (B-97). Resolving the project takes 0.2 ms at the median on Windows, and 2.2 ms in a work tree on Linux in Docker (B-109).
  - `consolidate` is O(n²). Measured by package K at 2,000 memories: 153.9 ms at the median on Windows and 129.9 ms on Linux in Docker (B-103).
  - Connections have no read timeout and no cap; only the same user can connect (B-105).
  - Done by package K: AT-3 during an import and AT-4's peak memory through the daemon with the real model (B-106; RESULTS section 2).
- **Stream C follow-ups.** Done by package K: AT-7 rerun with live embedding through the daemon passes, with 5 regions, 16.2% unclustered and the largest 16.9% on Windows (B-49; RESULTS section 3). `regions` takes about 6.7 s at 2,000 memories and the coordinates as much again, cold, on the reference machine; there is no SPEC 11 budget, so stream E caches runs under `input_hash`, computes coordinates only when asked, and drops a cached run when a member is edited or re-embedded (B-46, B-47). `journey` is not bit-identical across operating systems and is outside AT-6 (B-48). Linux timings for `sm-spatial` are not measured.
- **Left for the lead in phase 3, from the stream reports.**
  - Done by package I: bulk vec1 inserts (B-113) and project scope at 100,000 chunks (B-114).
  - Done: `journey`'s wording is B-48 (one memory per step, no repeats; SPEC 7's "as in version 1" is read as the interpolation).
  - Done: the verify stamp is B-52 (not adopted in 3.0; reconsidered only if Linux misses the cold-start budget; package K measures it in Docker, as native Linux is not available here).
  - Done by package J: the `SharingOps` adapter over `sm-share` and the `daemon` subcommand (B-79, B-119, B-122); `MIGRATION.md` says to run `migrate` from inside the main repository (B-79's gap).
  - Done by package J: `sm-embed` wired into `setup` and `doctor` in place of `Unwired` (B-43, B-121).
  - Done by package J: `migrate`, `export` and `import` on the command line, and `share` and `sync` through `spatial-memory mcp`, reach the daemon's adapter (B-79, B-122). `share` and `sync` are MCP tools, not subcommands; SPEC 3's command table has none for them.
  - Done by package I (B-115): B-87's liveness rule on every local path; stream F's `sm-store` items (B-93); `share` refusing a locally merged memory (B-89).
  - Exec-form hooks by bare name on Linux inside Claude Code; if they fail, SPEC 13's `claude-hook` fallback applies (B-85).
  - Done by package H: the second review of stream E's three items and its smaller notes (B-116 to B-118); the drift guard (B-102); the query session kept for life (spike decision 5); the daemon's folder checks (B-112, B-117); the title-only `revise` (B-118); the tests for the drain while `failed` and the spill at 1,024 (B-116); the AT-16 ranking test.
  - Done by package J: the `daemon` subcommand exits as soon as `run` returns (B-119), and `sm-cli`'s client creates the profile folder with mode 0700 on Unix (B-123).
  - Done by package J: `UnwiredSharing` lives only in the tests, and its test is kept (B-101, B-122).
  - Done: `MIN_OUTPUT_TOKEN_BUDGET` is 200, and a limit below it is refused before any tool runs (B-111, B-112).
  - Done by package K: `reindex` keeps a captured memory's index cap, with a regression test (B-118).
  - Done by package K: the `doctor_numbers` operation and `doctor`'s printing of it (SPEC 5 rule 7; B-125), the idle line's plural, the hook process on Linux in Docker (B-124), and a fixture credential shared through the binary (AT-9).
  - Done by package K: the hook binary end to end against the `daemon` subcommand: AT-3 with the daemon killed mid-request, AT-13, AT-14, AT-15 (spooled while stopped, then drained) and AT-17.
  - Done by package K: through the daemon with the real model, AT-3 during an import, AT-4's peak memory, the cost of opening a bulk session, project resolution per request, and `consolidate` at 2,000 memories (B-97, B-103, B-106, B-109).
- **Package K follow-ups (B-124 to B-133).**
  - Idle memory after `recall` at 100,000 chunks is missed on the working set in 11 of 12 readings, up to 1,499.7 MiB. Private memory stays between 640 and 657 MiB; the rest is the mapped database file. SPEC 11's "idle after load" is met. Whether idle memory after use needs its own bound is the owner's call (B-132).
  - Done: on Windows the raw thread count during an import peaks at 27 against 24, and the daemon's own threads are 20 against 20 (B-131). The owner accepted that the budget counts the threads the daemon itself creates; SPEC 5 and RESULTS say so (B-141).
  - Done: Windows applies power throttling (EcoQoS) to the windowless daemon within seconds of its start. The query embedding then takes about 85 ms instead of 47.6 ms. The hook budget holds. The owner decided the daemon does not opt out in 3.0, and the README's troubleshooting table names the effect (B-130, B-142).
  - K's two-processor figures were taken before the lead's fix: the daemon read 16 cores from `/proc/cpuinfo` and chose 4 query and 12 bulk threads on 2 processors. The thread budget now takes the smaller of the physical cores and the processors the process may use (B-133).
  - Not run with the real model: the idle exit, the fallback model Arctic Embed M (D-5, B-57), bulk throughput on Windows with throttling off, and `open`, `consolidate`, `at2`, `at7`, `at8` and `at10` on two processors (RESULTS section 5).
- **The two reviews after package K (B-134 to B-138).** Two independent reviewers read the code merged after stream E: packages H to K and the lead's commits.
  - Fixed: `import` now keeps a captured memory's index cap, as `reindex` does (B-134; B-118 is closed).
  - Fixed: a relative `SPATIAL_MEMORY_HOME` or `SPATIAL_MEMORY_MODEL_DIR` is refused with `config_invalid`. On Windows, `\data` and `C:data` count as relative. This is the lead's `sm-core` change (B-135).
  - Fixed: a refusal that a daemon exiting at that moment can cause is polled again, and a daemon is started. Every other refusal is still reported at once (B-136).
  - Fixed: the AWS secret access key is now caught with a hyphenated keyword, `=>`, a space, an XML element, and Terraform's `secret_key` (B-137).
  - Fixed: `status` no longer prints a total that covers only some of the projects it counts. The "no daemon answered" error now names `daemon.log` (B-138).
  - Refuted: an unbounded `doctor_numbers` wait. The bulk lock is held one batch at a time (B-136).
  - Report only, unchanged in 3.0 (B-136):
    - Done: SPEC 16 no longer describes `binary_path` (B-85, B-144).
    - On Windows, `physical_cores()` ignores an affinity mask (B-133).
    - `SyncOptions.forget_before_loading` is in production code for a test.
- **Not verified in phase 0.** Exec-form hooks by bare name on Linux inside Claude Code; native Linux timings; macOS (SPIKE.md, "Not verified in phase 0"). What a user could meet is listed in README.md, "Not verified" (B-144).
