# Version 3.0 build progress

Read this file and `DECISIONS.md` first after a restart or a context compaction.

## Current phase

Phases 1 and 2. Phase 0 is committed on `v3` as `97426c2`, and the contracts are frozen (CONTRACTS.md section 1). The stream specifications are in `docs/v3/streams/` (`stream-a.md` to `stream-d.md`, `stream-g.md`, and `RULES.md`, which is appended to each). Phase 1 runs through Orca with at most four workers: A, B, C and G first, then D when a slot frees. Streams C, B, G and A are merged; D is running. Phase 2's specifications are written (`stream-e.md`, `stream-f.md`), and streams E and F start in the two free slots.

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
| D clients | `sm-cli` | 1 | | `task_2e39969f24a8` | `ctx_37824ef4196e`, worktree `stream-d-clients` | Running |
| G capture | `sm-capture` | 1 | | `task_24c15023b6ca` | `ctx_e485d1bf5b49`, worktree `stream-g-capture` | **Merged.** Library parts of AT-9, AT-13, AT-14, AT-16 and AT-17 pass; 726 of 726 ported fixture cases pass, 278 not applicable with reasons; 33 tests on Windows and Linux (Docker); B-59 to B-68; report `docs/v3/streams/g-report.md` |
| E daemon | `sm-daemon` | 2 | A, B, C, G (the capture pipeline) | | | Specification written |
| F sharing and migration | `sm-share` (B-8, B-77) | 2 | A; G's scanner as a dev-dependency | | | Specification written |

Orca run: `run_991d66c5f298`. At most four workers run at once. Workers use Claude Opus 5.5 (`claude-opus-5-5`) at effort `xhigh`, each in a child worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at `ea82480`. Each task's spec is its stream file followed by `RULES.md`. Streams E and F branch from the `v3` commit that adds their specifications.

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
- **Protocol gaps accepted for 3.0 (B-37).** The injection race, and `LinkView.stale` left absent when ancestry cannot be decided: stream A's `head_only` oracle settles only a commit equal to the head or a prefix of it (B-76).
- **Model step of `setup` and `doctor`.** Stream D builds it behind a trait private to `sm-cli`; until the lead wires `sm-embed` in phase 3, the binary reports it as not wired (B-43).
- **Stream C follow-ups.** Rerun AT-7 with live embedding in phase 3 (B-49). `regions` takes about 6.7 s at 2,000 memories and the coordinates as much again, cold, on the reference machine; there is no SPEC 11 budget, so stream E caches runs under `input_hash`, computes coordinates only when asked, and drops a cached run when a member is edited or re-embedded (B-46, B-47). `journey` is not bit-identical across operating systems and is outside AT-6 (B-48). Linux timings for `sm-spatial` are not measured.
- **Left for the lead in phase 3, from the stream reports.**
  - Bulk vec1 inserts are quadratic (a-report 9.1, B-74). One option: load `import` and `migrate` with no vector index, then rebuild it.
  - Project scope at 100,000 chunks is 243.7 ms at p95 for the store alone (a-report 9.2, B-75). Options: memory-mapped reads on reader connections, or vec1.
  - A B-n entry on `journey`'s wording (c-report).
  - The verify-stamp decision (b-report).
  - The `SharingOps` adapter over `sm-share` and the `daemon` subcommand (B-79); `MIGRATION.md` says to run `migrate` from inside the main repository (B-79's gap).
- **Not verified in phase 0.** Exec-form hooks by bare name on Linux inside Claude Code; native Linux timings; macOS (SPIKE.md, "Not verified in phase 0").
