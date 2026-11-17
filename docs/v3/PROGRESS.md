# Version 3.0 build progress

Read this file and `DECISIONS.md` first after a restart or a context compaction.

## Current phase

Phase 3. Phase 0 is committed on `v3` as `97426c2`, and the contracts are frozen (CONTRACTS.md section 1). The stream specifications are in `docs/v3/streams/` (`stream-a.md` to `stream-d.md`, `stream-g.md`, and `RULES.md`, which is appended to each). Phase 1 runs through Orca with at most four workers: A, B, C and G first, then D when a slot frees. Streams C, B, G, A and D are merged, so phase 1 is complete. Phase 2's specifications are written (`stream-e.md`, `stream-f.md`). Streams F and E are merged, so phase 2 is complete. Stream E was sent back once after review; the lead fixed the review's `sm-store` defect on `v3` (B-94). Phase 3, the lead's integration, has started; its list is under "Left for the lead in phase 3" below.

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
| H daemon hardening | `sm-daemon` | 3 | E | Agent tool (not Orca) | worktree `stream-h-hardening` | **Running.** The second E review's should-fix defects, the Windows pipe identity, the B-112 socket folder checks, notes C to I, the missing tests; spec `docs/v3/streams/stream-h.md` |
| I store and sharing follow-ups | `sm-store`, `sm-share` | 3 | A, F | Agent tool (not Orca) | worktree `stream-i-store` | **Running.** B-87 on every local path, `updated_at`, the sync mark and marks-only listing, `share` refusing a merged memory, B-74 and B-75 measured; spec `docs/v3/streams/stream-i.md` |
| J wiring | `sm-cli`; `sm-daemon`'s sharing adapter | 3 | D, E, F | Agent tool (not Orca) | worktree `stream-j-wiring` | **Running.** The `daemon` subcommand, the `SharingOps` adapter over `sm-share`, the model step over `sm-embed`, clients through `sm_daemon::connect`; spec `docs/v3/streams/stream-j.md` |

Phase 3 packages H, I and J run as Agent-tool subagents in parallel, each in its own worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at the commit that adds their specifications. Each spec is the package file followed by `RULES.md`. Package K (end-to-end hook runs, real-model measurements, RESULTS.md) follows J.

The lead's phase 3 documents, drafted while H, I and J run:

- **`README.md`, the 3.0 README.** The 1.x README moved to `docs/archive/python/README.md`. It describes `status` and `daemon`, which J adds; check both against J's merge.
- **`docs/v3/MIGRATION.md`.** It is checked against version 1's `export_memories`. An export path resolves against the Python server's working directory and must be inside `exports/` or `backups/`. `project: "*"` is needed to export every project.
- **`docs/v3/CI_PROPOSAL.md`.** Runner prices and hardware were read on 17 November 2026. The repository is public, so standard runners are free. The replacements for the three Python workflows are proposed. Section 4's build and test minutes are measured once H, I and J have merged, on a quiet machine.

Follow-up: `CONTRIBUTING.md` still describes the Python workflow (ruff, pytest). Point it to `CLAUDE.md` before the release.

Orca run: `run_991d66c5f298`. At most four workers run at once. Workers use Claude Opus 5.5 (`claude-opus-5-5`) at effort `xhigh`, each in a child worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at `ea82480`. Each task's spec is its stream file followed by `RULES.md`. Streams E and F branch from `v3` at `9387a08`, the commit that adds their specifications.

## Open problems

- **Arctic feature licence.** The `arctic-embed-m` feature (D-5) brings `option-ext` 0.2.0, which is MPL-2.0, through `fastembed` → `hf-hub` → `dirs`. The default build does not contain it. Flagged for the owner in LICENSES.md and B-41.
- **Permissive licence not on the list.** `webpki-roots` (CDLA-Permissive-2.0), linked into the default binary for the model download. Flagged in LICENSES.md. `webpki-root-certs` and `libfuzzer-sys` left the tree with B-50, and so did the system OpenSSL crates.
- **Stream B follow-ups (B-52 to B-58).** Model load in Docker took 8.95 s against the 5-second cold-start budget, read over the Docker Desktop file share at about 230 MB/s; native Linux load and verify are measured in phase 3. Verify on a cold file cache was not measured on Windows. No live download was run; the download is tested against a local server. The Arctic fallback is built and unit-tested but never loaded or run (D-5 Partial). The daemon's refusal on a bad model is stream E's AT-10 test.
- **Stream G follow-ups (B-59 to B-68).** The capture pipeline has not run end to end; streams D and E own the rest of AT-9, AT-13, AT-14, AT-16 and AT-17, and stream D must cover the fixture cases G marked as stream D's (stdin, stdout, the working-directory check, ambient recall). Stream E counts `capture_cap_hit_total` on the index cap, not on G's extraction cap (B-65). Python's `\s` matches U+001C to U+001F and the `regex` crate's does not (B-59). Linux capture times are debug only (B-68). SPEC 16 (Stop and the transcript, B-61) and SPEC 3 and 14 (the spool's crate, B-66) are out of date.
- **Stream A follow-ups (B-69 to B-76).**
  - Bulk inserts into vec1 slow by about 0.33 ms per chunk for every 1,000 chunks already stored, so `import`, `migrate` and `set_embeddings` after a `reindex` take time quadratic in their size (B-74). Stream E keeps each `set_embeddings` transaction to one embedding batch and never makes the hook reply wait for the writer; phase 3 measures import and migrate.
  - Project scope in a single project of 100,000 chunks: the store alone is 243.7 ms at p95 against `recall`'s 250 ms (B-75). Phase 3 measures `recall` end to end on Windows and native Linux; the search budgets have not been measured on Linux.
  - Two memory-vector functions disagree when a memory's chunks cancel out: the store's falls back to the first chunk, `sm_spatial::memory_vector` returns an error. Stream E uses the store's (B-76).
  - No backup retention: backups accumulate in `backups_dir()` (B-71).
  - 79 temporary folders (63.3 MB) leaked by early stream A test runs remain in `%LOCALAPPDATA%\Temp`, outside the repository. Deleting them needs the owner's permission; current runs leave none.
- **Protocol gaps accepted for 3.0 (B-37).** The injection race (narrowed by B-104: the daemon records an injection only after it writes the reply), and `LinkView.stale` left absent when ancestry cannot be decided: stream A's `head_only` oracle settles only a commit equal to the head or a prefix of it (B-76).
- **Model step of `setup` and `doctor`.** Stream D built it behind the `ModelStep` trait in `crates/sm-cli/src/model_step.rs`; the delivered `Unwired` step reports `model_missing` with a next action until the lead wires `sm-embed` in phase 3 (B-43).
- **Stream D follow-ups (B-80 to B-85).**
  - `CLAUDE_CONFIG_DIR` is not honoured: `setup`, `doctor` and the plugin check read `~/.claude` and `~/.claude.json` only (B-84).
  - The real `claude mcp add` and `claude mcp remove` were never run; the tests use a fake runner (B-84).
  - Whether Claude Code exports a plugin option's default when the user never saved a value is unverified; recall stays on either way (B-85).
  - `claude plugin validate` runs only where `claude` is on `PATH`, so it is skipped in Docker (B-85).
  - The AT-3 hung-daemon case is 1,422.8 ms at worst on Windows against 1,500 ms; a loaded machine could push it over (B-80).
- **Stream F follow-ups (B-87 to B-93).**
  - A forgotten local superseder leaves its target superseded for good, because `forget` deletes the row and with it the record of the supersession; `forget` must apply B-87's rule (the lead item below). A superseder retracted outside `sync` is settled by the next sync (B-87).
  - A shared target whose own file says `superseded`, with no live superseder, has its file read again on every sync while that lasts (B-87).
  - Import restore: a target already stored and changed locally since gets its export line's status back when a superseder in the file is newly inserted (B-92).
  - Redaction does not scan project values, project lines or raw ids (B-88).
  - A retracted copy blocks the same text under a new id in `sync` until it is forgotten or its file restored (B-90).
  - A sync in which all 10,000 files change was not measured; it is estimated at about 30 s of commits (B-90, B-93).
  - The export and import round trip changes the `updated_at` of superseded memories (B-92).
  - SPEC 12's exit condition, a migration report on a real version 1 store with no lost memories, needs the owner's store (B-91).
- **Stream E follow-ups (B-95 to B-111).**
  - Spike decision 5: no test checks that the query session stays open for life (B-95).
  - macOS: the private-folder bind and the socket owner check are tested on Linux only, and SPEC 13's `interprocess` risk stays open there (B-95).
  - AT-10 follows B-36: with a changed byte the daemon starts and stays `failed`, instead of not starting (B-96).
  - Covered by code only: the spool drain skipped while the model is `failed` (B-96), the spill to the spool at 1,024 queued captures (B-98), rank numbering after a memory vanishes (B-103), the log line for a failed injection record (B-104), and the second owner check in `connect` (B-95).
  - A capture processed twice after a crash is stored once but journals its capture metrics twice (B-99, B-108).
  - The cost of opening a bulk session is not measured, so the 10-second grace is provisional (B-97); resolving the project on every request is not timed on its own (B-109).
  - `consolidate` is O(n²) and not timed (B-103).
  - Connections have no read timeout and no cap; only the same user can connect (B-105).
  - Every daemon figure is a debug build with the fake embedder; the real-model AT-3 figure and AT-4's peak memory through the daemon are phase 3 (B-106).
- **Stream C follow-ups.** Rerun AT-7 with live embedding in phase 3 (B-49). `regions` takes about 6.7 s at 2,000 memories and the coordinates as much again, cold, on the reference machine; there is no SPEC 11 budget, so stream E caches runs under `input_hash`, computes coordinates only when asked, and drops a cached run when a member is edited or re-embedded (B-46, B-47). `journey` is not bit-identical across operating systems and is outside AT-6 (B-48). Linux timings for `sm-spatial` are not measured.
- **Left for the lead in phase 3, from the stream reports.**
  - Bulk vec1 inserts are quadratic (a-report 9.1, B-74). One option: load `import` and `migrate` with no vector index, then rebuild it.
  - Project scope at 100,000 chunks is 243.7 ms at p95 for the store alone (a-report 9.2, B-75). Options: memory-mapped reads on reader connections, or vec1.
  - Done: `journey`'s wording is B-48 (one memory per step, no repeats; SPEC 7's "as in version 1" is read as the interpolation).
  - Done: the verify stamp is B-52 (not adopted in 3.0; reconsidered only if native Linux misses the cold-start budget, which package K measures).
  - The `SharingOps` adapter over `sm-share` and the `daemon` subcommand (B-79); `MIGRATION.md` says to run `migrate` from inside the main repository (B-79's gap).
  - Wire `sm-embed` into `setup` and `doctor` in place of `Unwired` (B-43).
  - The command-line subcommands for `share`, `sync`, `migrate`, `export` and `import` through the daemon (B-79).
  - Apply B-87's liveness rule when a local superseder is forgotten or retracted outside `sync`: in `sm-store`'s `forget` and `forget_by_filter` (and a local revision to `retracted`), in the same transaction, restore each local target left with no live superseder to `active`, never a retracted or merged one. Projects that never sync have no other repair.
  - `sm-store` items from stream F (B-93): keep `updated_at` when a superseder is inserted and at the import restore; `revise_memory` that records the sync mark in the same transaction; a marks-only listing for `sync`; `revise` refuses to make a memory `active` while a live memory supersedes it (B-87).
  - `share` refuses a locally merged memory (`extra.merged_into`), which it would otherwise write as a bare `superseded` (B-89).
  - Exec-form hooks by bare name on Linux inside Claude Code; if they fail, SPEC 13's `claude-hook` fallback applies (B-85).
  - **First, from the second review of stream E.** Fix these before the `sm-cli` client copies the daemon's checks.
    - A transcript candidate that always fails, such as content that gives no tokens, blocks every later capture of its session, because the session moves on only after a store. Drop such a candidate and log it, and move the session on when the failure is permanent or the bound is reached (SPEC 16, B-108).
    - The owner check after `connect` passes when the socket is gone, so another user who binds the `/tmp` fallback and unlinks it after `accept` is trusted. Refuse a missing socket and check the peer's user id (`SO_PEERCRED`, `getpeereid`) (SPEC 3, B-95).
    - On Windows the client does not check who serves the pipe, and the pipe name is predictable. Check the user of the server process (`GetNamedPipeServerProcessId`), and create the first instance with `FILE_FLAG_FIRST_PIPE_INSTANCE` (spike decision 1, B-95).
  - Smaller notes from that review, to settle in phase 3 (B-99, B-105, B-108, B-110, B-111):
    - `project_unresolved` sends a capture to `spool/failed/` at once, even for a folder that is only absent for now, such as a drive not yet mounted.
    - A spool file that cannot be removed after its store is processed again on every drain.
    - A failed `hold_new` processes the capture from memory only.
    - The bulk queue counts an embedder-wide `internal` error as the chunk's fault, so after three rounds with nothing embedded every chunk of the batch is skipped until a restart. Count a strike only in a round where another chunk was stored.
    - Threads can outlive `DaemonHandle::wait`, so the `daemon` subcommand must exit as soon as it returns, and the handle must say so.
    - A memory with under 16 characters of content can lose its tags, anchors and title while `content_truncated` stays false.
    - The private bind folder's name is predictable in `/tmp` (denial of service only).
  - When the `daemon` subcommand passes the `sm-share` adapter, move `UnwiredSharing` into `crates/sm-daemon/tests/common` and keep its test (B-101).
  - A drift-guard test that every key each unchecked `sm-core` parameter struct serialises is in dispatch's `allowed_args` (B-102).
  - A test that the query session stays open for life (spike decision 5, B-95).
  - The socket owner check in the `sm-cli` hook client. The fallback socket's path is done: `daemon.sock` in a `spatial-memory-<hash>` folder of its own per profile (B-112). Left: the daemon creates that folder with mode 0700, and both ends refuse a folder another user owns or can write to (B-95, B-112).
  - Done: `MIN_OUTPUT_TOKEN_BUDGET` is 200, and a limit below it is refused before any tool runs (B-111, B-112).
  - Check whether a title-only `revise` leaves chunks embedded with the old title: the embedded document includes the title, and a title change does not re-chunk.
  - Tests for the spool drain skipped while `failed` and for the spill at 1,024 queued captures (B-96, B-98).
  - One end-to-end run of the hook binary against the `daemon` subcommand: AT-3 with the daemon killed mid-request, AT-13, AT-14, AT-15 (spooled while stopped, then drained) and AT-17; and an AT-16 test that importance does not change `recall` ranking.
  - Measure through the daemon with the real model: AT-3 during an import, AT-4's peak memory, the cost of opening a bulk session, project resolution per request, and `consolidate` at 2,000 memories (B-97, B-103, B-106, B-109).
- **Not verified in phase 0.** Exec-form hooks by bare name on Linux inside Claude Code; native Linux timings; macOS (SPIKE.md, "Not verified in phase 0").
