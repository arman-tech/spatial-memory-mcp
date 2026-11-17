# Package J (phase 3): wiring the binary

## Goal

Make `spatial-memory` a complete program: the `daemon` subcommand, the `SharingOps` adapter over `sm-share`, the real model step in `setup` and `doctor`, and every client connecting through the daemon's owner-checked `connect`. After this package, the binary runs end to end with the real model, and package K tests it that way.

**Done when:** every item under "Build" works and is tested, the binary built in release starts a daemon with the real model and answers `recall` through `mcp` and `hook` (a manual run, written up in the report), both gates pass (RULES.md sections 5 and 6, with the full workspace on Linux too), and `j-report.md` is committed.

## You own

- `crates/sm-cli/`.
- In `crates/sm-daemon/`: `src/sharing.rs`; a new `src/share_adapter.rs`; the lines in `src/lib.rs` and `Cargo.toml` that add them (`sm-share.workspace = true`); and a new file under `tests/common/` for `UnwiredSharing`. Package H owns the rest of `sm-daemon` and is changing it at the same time, so touch nothing else there. If you need a change in H's files, describe it in the report.
- `sm-cli` may now depend on `sm-daemon` and `sm-embed` (B-43; CONTRACTS section 2, phase 3). `sm-daemon` may depend on `sm-share`. No new third-party crate.
- You may not edit `sm-core` (frozen), `sm-store` or `sm-share` (package I is changing them; keep to their current public functions).

## Read

- **SPEC:** sections 3 (the command table, the daemon, auto-start), 5 (the model download and checksums), 8, 9 (`share`, `sync`), 10 (`migrate`), 16 ("Finding the binary", the hook client); always 11, 13 and 14.
- **CONTRACTS.md:** all of it; sections 2 (who depends on whom), 3 (the endpoint, the auto-start lock), 7.
- **DECISIONS:** B-43 (the model step), B-44 (the lock), B-79 (the `SharingOps` trait and the functions `sm-share` exposes), B-80 to B-85 (stream D: client deadlines, profiles, setup, the plugin), B-95 (the endpoint and owner checks), B-101 (`UnwiredSharing` leaves the delivered path), B-112.
- **Reports:** `d-report.md`, `e-report.md`, `f-report.md`, `b-report.md` (the download and verify); `docs/v3/PROGRESS.md`, "Left for the lead in phase 3".
- **Code:** `sm-cli/src/cli.rs` (the command enum; `daemon` is missing), `client.rs` (its own `connect` at about line 259, the detached spawn that runs `daemon --profile <name>`), `model_step.rs` (`ModelStep`, `Unwired`), `setup.rs`, `doctor.rs`, `hook.rs`, `mcp.rs`; `sm-daemon/src/lib.rs`, `server.rs` (`start`, `run`, `DaemonHandle`), `sharing.rs` (`SharingOps`, `SharingContext`, `ResolvedProject`, `UnwiredSharing`), `loader.rs` (`GemmaLoader`), `options.rs`, `endpoint.rs` (`connect`); `sm-share/src/lib.rs`; `sm-embed/src/download.rs`, `model.rs` (`verify`, `MODEL_FILES`), `doctor.rs`.

## Build

1. **The `daemon` subcommand.** `spatial-memory daemon [--profile <name>]` loads the profile's settings, builds `GemmaLoader`, the sharing adapter and `DaemonOptions`, and calls `sm_daemon::run`. It exits the process as soon as `run` returns, because daemon threads can outlive it (review note; H documents it on `DaemonHandle`). Exit codes and the log line for each `Exit` variant, including `LockHeld`. It must work when spawned detached by the client's auto-start on Windows and Linux (spike decision 2: never `CREATE_BREAKAWAY_FROM_JOB`).
2. **The `SharingOps` adapter** in `sm-daemon/src/share_adapter.rs`: each of `share`, `sync`, `migrate`, `import` and `export` calls the `sm-share` function B-79 names, with the context's store, chunker, scanner, chunk parameters and resolved project. No placeholder path. Test each op through the daemon with the fake embedder (the existing `s9_` tests in `sm-daemon/tests` show how).
3. **`UnwiredSharing` leaves the delivered path** (B-101): move it into `sm-daemon/tests/common/`, drop it from `lib.rs`'s exports, and keep `s9_import_export_and_migrate_without_the_implementation_say_so` running against the moved type.
4. **The model step** (B-43): an implementation of `ModelStep` over `sm-embed`. `setup` downloads the model into the per-user cache when it is missing and verifies the SHA-256 of every file; `doctor` verifies and reports, with a next action on failure. Replace `Unwired` in the binary; keep `Unwired` only if a test needs it, and then only under `#[cfg(test)]` or in `tests/`. Test against a local server as stream B did; **never download the real model in a test**, and never into the repository or a worktree. On this machine the model is already in the cache (RULES.md section 3).
5. **Clients connect through `sm_daemon::connect`.** The hook client, the MCP proxy and the command-line operations replace `sm-cli`'s own `connect` with the daemon crate's, so every client gets the owner and peer checks package H is hardening (B-95, B-112). Keep the client's deadlines (B-80) and the never-blocks rule (CONTRACTS section 7). The existing mock-daemon tests must still pass.
6. **`share` and `sync` are MCP tools** (SPEC 8), already forwarded by `mcp`; `import`, `export` and `migrate` are command-line subcommands that already exist. Check each reaches the adapter end to end through the binary.
7. **The hook budget after linking.** `sm-cli` now links `sm-embed` and ONNX Runtime. Rerun `s11_hook_process_warm_100_runs_inside_the_hook_budget` in release and report the 95th percentile before and after, and the binary size.

## Tests

- Names cite their rule (B-40): `s3_` for the daemon subcommand and clients, `s5_` for the model step, `s9_` and `s10_` for sharing and migration, `s11_` for budgets, `d1_` for the one-binary decision.
- Unit tests use the fake embedder. The real-model test binary stays `crates/sm-embed/tests/real_model.rs`; do not add another. The manual release run in "Done when" is written up, not a test.

## Report

`docs/v3/streams/j-report.md`, in RULES.md section 9's format.
