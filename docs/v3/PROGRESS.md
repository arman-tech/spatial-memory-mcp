# Version 3.0 build progress

Read this file and `DECISIONS.md` first after a restart or a context compaction.

## Current phase

Phase 1. Phase 0 is committed on `v3` as `97426c2`, and the contracts are frozen (CONTRACTS.md section 1). The stream specifications are in `docs/v3/streams/` (`stream-a.md` to `stream-d.md`, `stream-g.md`, and `RULES.md`, which is appended to each). Phase 1 runs through Orca with at most four workers: A, B, C and G first, then D when a slot frees.

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
| A store | `sm-store` | 1 | | | | Not started |
| B embed | `sm-embed` | 1 | | | | Not started |
| C spatial | `sm-spatial` | 1 | | | | Not started |
| D clients | `sm-cli` | 1 | | | | Not started |
| G capture | `sm-capture` | 1 | | | | Not started |
| E daemon | `sm-daemon` | 2 | A, B, C, G (the capture pipeline) | | | Not started |
| F sharing and migration | `sm-share` (B-8) | 2 | A, B; G's scanner as a dev-dependency | | | Not started |

Orca run: not created yet. At most four workers run at once.

## Open problems

- **Arctic feature licence.** The `arctic-embed-m` feature (D-5) brings `option-ext` 0.2.0, which is MPL-2.0, through `fastembed` → `hf-hub` → `dirs`. The default build does not contain it. Flagged for the owner in LICENSES.md and B-41.
- **Permissive licences not on the list.** `webpki-roots` and `webpki-root-certs` (CDLA-Permissive-2.0) and `libfuzzer-sys` (NCSA, arctic only). Flagged in LICENSES.md.
- **Protocol gaps accepted for 3.0 (B-37).** The injection race, and `LinkView.stale` left absent when ancestry cannot be decided (stream A decides).
- **Model step of `setup` and `doctor`.** Stream D builds it behind a trait private to `sm-cli`; until the lead wires `sm-embed` in phase 3, the binary reports it as not wired (B-43).
- **Not verified in phase 0.** Exec-form hooks by bare name on Linux inside Claude Code; native Linux timings; macOS (SPIKE.md, "Not verified in phase 0").
