# Version 3.0 build progress

Read this file and `DECISIONS.md` first after a restart or a context compaction.

## Current phase

Phase 1. Phase 0 is committed on `v3` as `97426c2`, and the contracts are frozen (CONTRACTS.md section 1). The stream specifications are in `docs/v3/streams/` (`stream-a.md` to `stream-d.md`, `stream-g.md`, and `RULES.md`, which is appended to each). Phase 1 runs through Orca with at most four workers: A, B, C and G first, then D when a slot frees. Streams C and B are merged; A, D and G are running. E and F start when A, B and G (E also needs C) are merged.

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

## Streams

| Stream | Crate | Phase | Depends on | Orca task | Worker | State |
| --- | --- | --- | --- | --- | --- | --- |
| A store | `sm-store` | 1 | | `task_c71067fb0f26` | `ctx_5dba5f9a9d4d`, worktree `stream-a-store` | Running |
| B embed | `sm-embed` | 1 | | `task_8922c54475a3` | `ctx_eac8a4efaf42`, worktree `stream-b-embed` | **Merged.** AT-1 passes; AT-2 (embedding path), AT-4 (embedding path) and AT-10 (load and download) pass; D-4, D-6 and D-10 covered; 54 unit tests and the real-model binary on Windows, 54 on Linux (Docker); B-50 to B-58; report `docs/v3/streams/b-report.md` |
| C spatial | `sm-spatial` | 1 | | `task_f3585000cbb0` | `ctx_613e314199cd`, worktree `stream-c-spatial` | **Merged.** AT-6 and AT-7 pass on Windows and Linux (Docker); 47 tests; B-45 to B-49; report `docs/v3/streams/c-report.md` |
| D clients | `sm-cli` | 1 | | `task_2e39969f24a8` | `ctx_37824ef4196e`, worktree `stream-d-clients` | Running |
| G capture | `sm-capture` | 1 | | `task_24c15023b6ca` | `ctx_e485d1bf5b49`, worktree `stream-g-capture` | Running |
| E daemon | `sm-daemon` | 2 | A, B, C, G (the capture pipeline) | | | Not started |
| F sharing and migration | `sm-share` (B-8) | 2 | A, B; G's scanner as a dev-dependency | | | Not started |

Orca run: `run_991d66c5f298`. At most four workers run at once. Workers use Claude Opus 5.5 (`claude-opus-5-5`) at effort `xhigh`, each in a child worktree under `C:/Users/jon/orca/workspaces/spatial-memory-mcp/`, branched from `v3` at `ea82480`. Each task's spec is its stream file followed by `RULES.md`. Tasks for E and F are created when their specifications are written.

## Open problems

- **Arctic feature licence.** The `arctic-embed-m` feature (D-5) brings `option-ext` 0.2.0, which is MPL-2.0, through `fastembed` → `hf-hub` → `dirs`. The default build does not contain it. Flagged for the owner in LICENSES.md and B-41.
- **Permissive licence not on the list.** `webpki-roots` (CDLA-Permissive-2.0), linked into the default binary for the model download. Flagged in LICENSES.md. `webpki-root-certs` and `libfuzzer-sys` left the tree with B-50, and so did the system OpenSSL crates.
- **Stream B follow-ups (B-52 to B-58).** Model load in Docker took 8.95 s against the 5-second cold-start budget, read over the Docker Desktop file share at about 230 MB/s; native Linux load and verify are measured in phase 3. Verify on a cold file cache was not measured on Windows. No live download was run; the download is tested against a local server. The Arctic fallback is built and unit-tested but never loaded or run (D-5 Partial). The daemon's refusal on a bad model is stream E's AT-10 test.
- **Protocol gaps accepted for 3.0 (B-37).** The injection race, and `LinkView.stale` left absent when ancestry cannot be decided (stream A decides).
- **Model step of `setup` and `doctor`.** Stream D builds it behind a trait private to `sm-cli`; until the lead wires `sm-embed` in phase 3, the binary reports it as not wired (B-43).
- **Stream C follow-ups.** Rerun AT-7 with live embedding in phase 3 (B-49). `regions` takes about 6.7 s at 2,000 memories and the coordinates as much again, cold, on the reference machine; there is no SPEC 11 budget, so stream E caches runs under `input_hash`, computes coordinates only when asked, and drops a cached run when a member is edited or re-embedded (B-46, B-47). `journey` is not bit-identical across operating systems and is outside AT-6 (B-48). Linux timings for `sm-spatial` are not measured.
- **Not verified in phase 0.** Exec-form hooks by bare name on Linux inside Claude Code; native Linux timings; macOS (SPIKE.md, "Not verified in phase 0").
