# Package H (phase 3): `sm-daemon` hardening

## Goal

Fix the defects the second review of stream E left for phase 3, and add the tests it found missing, in `crates/sm-daemon`. The daemon is merged and every test passes; this package makes it safe before the `sm-cli` client starts calling `sm_daemon::connect` (package J, running at the same time).

**Done when:** every item under "Build" is fixed or tested as it says, both gates pass (RULES.md sections 5 and 6, with the full workspace on Linux too), and `h-report.md` is committed.

## You own

- `crates/sm-daemon/`, except the files package J owns: `src/sharing.rs`, a new `src/share_adapter.rs`, and the lines in `src/lib.rs` and `Cargo.toml` that add them. J also moves `UnwiredSharing` into a new file under `tests/common/`. If you add helpers to `tests/common/mod.rs`, append them at the end of the file so the two branches merge cleanly.
- **Keep the public API stable.** `connect`, `endpoint::*`, `DaemonHandle`, `start`, `run`, `Started`, `Exit` and `DaemonOptions` keep their names and signatures; package J writes against them. Add, do not change. If a change is unavoidable, list it in your report.
- You may not edit `sm-core` (frozen). If a fix needs an `sm-core` change, request it in the report with the evidence (RULES.md section 2).

## Read

- **SPEC:** sections 3 (architecture, the endpoint), 5 ("Bulk"), 8 ("Conventions"), 16 (capture, retries, the spool); always 11, 13 and 14.
- **CONTRACTS.md:** all of it; sections 3 (the endpoint and socket path), 7 and 8 (capture and the spool).
- **DECISIONS:** B-95 to B-112. B-95 (the endpoint, the private-folder bind, owner checks), B-108 (captures stay in `spool/new/` until stored), B-111 and B-112 (the reply floor of 200, the fallback socket folder).
- **Reports:** `docs/v3/streams/e-report.md`; `docs/v3/PROGRESS.md`, "Stream E follow-ups" and "Left for the lead in phase 3".
- **Code:** `src/endpoint.rs` (`bind_as` about line 112, `check_owner` about 245, `connect` and `connect_as` about 277 and 286, `mod security` about 312, the Windows part about 356 to 450), `src/capture.rs` (about 407 to 466), `src/tools/remember.rs` (about 32 to 36), `src/queue.rs`, `src/dispatch.rs` (`allowed_args`), `src/budget.rs`, `src/server.rs`.
- **Third-party:** `interprocess` 2.4.8, `libc`, `windows-sys` under `~/.cargo/registry/src/`. Read the real signatures; do not write against an API from memory.

## Build

### First: the two should-fix defects and the Windows gap

1. **A capture candidate that always fails blocks its session.** A transcript candidate whose processing can never succeed (for example, content that gives no tokens) holds back every later capture of its session, because the session moves on only after a store. Drop such a candidate, log one line, and move the session on when the failure is permanent or the attempt bound (`MAX_ATTEMPTS`) is reached. Never lose a capture that could still succeed (SPEC 16, B-108).
2. **The owner check after `connect` passes when the socket is gone.** Another user who binds the `/tmp` fallback and unlinks it after `accept` is trusted. Refuse a missing socket. Check the peer's user id on the connected stream: `SO_PEERCRED` on Linux, `getpeereid` on macOS (behind `cfg`; macOS cannot be tested here, so say so). The server checks its clients the same way (SPEC 3, B-95).
3. **Windows: the client does not check who serves the pipe.** Check the pipe server's user: `GetNamedPipeServerProcessId`, open the process token, compare its user SID with the current user's. The server creates its first pipe instance with `FILE_FLAG_FIRST_PIPE_INSTANCE`, so a pipe someone else already created is refused (spike decision 1, B-95).
4. **The fallback socket folder (B-112).** `sm_core::paths::unix_socket_path` now returns `<runtime>/spatial-memory-<hash>/daemon.sock` when the profile path is too long. The daemon creates that folder with mode 0700. Both `bind` and `connect` refuse a folder that another user owns or that group or others can write to, with an error that names the folder and a next action. `bind_as` already calls `create_dir_all`; that leaves the default mode.

### Then: the review's smaller notes

5. `project_unresolved` sends a capture to `spool/failed/` at once, even for a folder that is only absent for now (a drive not yet mounted). Treat it as transient, within the attempt bound.
6. A spool file that cannot be removed after its store is processed again on every drain. Stop that (for example, remember its name for the life of the process and log once).
7. A failed `hold_new` processes the capture from memory only. Decide what is safe, record it.
8. The bulk queue counts an embedder-wide `internal` error as the chunk's fault, so after three rounds with nothing embedded every chunk of the batch is skipped until a restart. Count a strike only in a round where another chunk was stored.
9. Threads can outlive `DaemonHandle::wait`. Say so in its documentation: the caller must exit the process when it returns. (Package J's `daemon` subcommand does.)
10. A memory with under 16 characters of content can lose its tags, anchors and title in the budget cut while `content_truncated` stays false. A caller must be able to tell that the view was cut. Use only the frozen `sm-core` view types.
11. The private bind folder's name is predictable in `/tmp`. Make it unpredictable, created with `create_dir` (fails if it exists), retried on collision.

### Then: missing tests and one check

12. A drift-guard test: every key each unchecked `sm-core` parameter struct serialises is in dispatch's `allowed_args` (B-102).
13. Spike decision 5: a test that the daemon creates the query session first and keeps it for life (the same session object across requests, a bulk import, and an idle period short of idle exit) (B-95).
14. Tests for the spool drain skipped while the model is `failed` (B-96) and the spill to the spool at 1,024 queued captures (B-98).
15. Check whether a title-only `revise` leaves chunks embedded with the old title: the embedded document includes the title, and a title change does not re-chunk. If it does, fix it so the memory is re-chunked and re-embedded, and test it.
16. An AT-16 test through the daemon: importance does not change `recall` ranking.

## Tests

- Names cite their rule (B-40): `s3_` for the endpoint, `s16_` for capture and the spool, `s5_` for the bulk queue, `s8_` for tools and budgets, `spike1_` and `spike5_` for the spike decisions, `at16_` for AT-16.
- Unix-only tests carry `#[cfg(unix)]` and their Windows counterpart `#[cfg(windows)]`.
- Never weaken, skip or delete a test.

## Report

`docs/v3/streams/h-report.md`, in RULES.md section 9's format. Under "Decisions", one entry per choice above, so the lead can write the B-n entries.
