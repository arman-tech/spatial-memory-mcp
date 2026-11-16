# Stream E: `sm-daemon`

## Goal

Build `crates/sm-daemon`, the library behind `spatial-memory daemon`: the endpoint and the daemon side of the lock, the envelope check, dispatch of every op, the query session and the bulk embedding queue, recall and the ambient gates, the capture pipeline, the spool drain, the map cache, the journal metrics, and the `SharingOps` trait through which the lead wires `sm-share` in phase 3. Everything is tested over the real socket with the fake embedder; the lead adds the `daemon` subcommand to `sm-cli` and the real-model checks in phase 3.

**SPEC 14 done when:** "Every tool and every hook event works end to end over the socket."

## You own

- `crates/sm-daemon/` (new library crate; depends on `sm-core`, `sm-store`, `sm-embed`, `sm-spatial` and `sm-capture`; `sm-core` with `test-support` in `[dev-dependencies]` only). It does not depend on `sm-share`: stream F builds that crate at the same time (B-79).
- Nothing else. `sm-cli` and its binary belong to stream D and then the lead.

## Read

- **SPEC:** section 3 (all rules, especially 1 to 6), section 4 (all rules: status, chunks are derived, the model stamp, both search scopes, links and stale links, forgetting cascades, housekeeping), section 5 (the configuration table, "Using the machine's cores" rules 1 to 6, query and bulk sessions), section 6 (the recall pipeline, the similarity floor, ambient recall and its gates, the metrics journal, `related`), section 7 (the pipeline, labels, `journey`), section 8 (the 13 tools and the conventions), section 9 "Scope" and "Local only", section 11 (budgets, AT-3, AT-4, AT-5, AT-10 to AT-17, test conventions), section 15 (defaults), section 16 (the pipeline, "Never loses", the spool, the capture rules).
- **CONTRACTS.md:** all of it, closely sections 3 and 3.1 (you build the daemon side of the lock, steps 1 to 5), 4 (the peek and `version_mismatch`), 5 (every op and its reply kind), 6 (errors and daemon states), 7 ("Stream E, the daemon"), 8 (spool), 9 (the seven traits) and 10.
- **DECISIONS:**
  - Endpoint, protocol and states: B-17 (endpoint naming), B-26 (framing and handshake), B-27 (synchronous requests, queued embedding, no job model), B-28 (strict parameters, `limit` is a token budget), B-36 (the failed state), B-37 (the injection race, stale-link ancestry), B-44 (the lock).
  - Tools: B-29 (`remember` batch and dedupe), B-30 (`forget` and `project delete` confirm through an error), B-31 and B-73 (`consolidate` applies only the listed groups; `merged_into`), B-32 (region labels, `reinforce` ranks).
  - Ambient and journal: B-15 (the ambient block), B-34 (the hook journals its own gates; the daemon does not repeat them).
  - Spool: B-25, B-33 (drain at start and every 60 s, quarantine, stale `tmp/`), B-66.
  - Map: B-45 to B-48 (`sm-spatial` threads, the region pipeline and cache key, coordinates, `journey`).
  - Embedding: B-51 to B-55 (threads, integrity at every load, session settings, bulk priority, inputs and the chunker), B-56 (doctor numbers, for `status` if useful).
  - Capture: B-59 to B-68, closely B-61 (Stop), B-62 (session dedupe), B-63 (the daemon supplies `commit_sha`), B-65 (the transcript cap, `capture_cap_hit_total`), B-67 (transcript fixtures).
  - Store: B-69 to B-76, closely B-71 (backups, migrations, the index stamp, `reindex`), B-72 (shared memories are read-only locally: pass `Access::Local`), B-74 and B-75 (known store costs) and B-76 (what the daemon relies on).
  - Sharing: B-8, B-77, B-78 and B-79 (the `SharingOps` trait you define).
- **Stream reports** (they describe the crates you call):
  - `a-report.md` section 2 and section 8 (store types, `Access`, `configure_index` before any search, the `links_of` oracle, `memory_vectors`).
  - `b-report.md` sections 2, 6 and 8 (`GemmaProvider`, `StructuredChunker`, `ModelTokenizer`, `verify`, `ThreadReport`, `physical_cores`).
  - `c-report.md` section 2 and section 8 (the region pipeline and `journey`; map `PathStep` to `JourneyStep`, `RegionResult::central(k)` to `Region.central`, `RegionRun.unclustered` to the `u32` count; an edit or re-embed must invalidate a cached run).
  - `g-report.md` section 2, "The entry point stream E calls" and "How stream E uses it" (the `CaptureParams` to `CaptureInput` mapping, one `SessionState` and transcript offset per session, `capture_size_tokens` from characters).
- **SPIKE.md** section c (local socket, the user-only pipe descriptor, `FILE_FLAG_FIRST_PIPE_INSTANCE`) and SPIKE-WINDOWS spike decisions 1 to 5. References; write the crate fresh.
- **sm-core:** `protocol` (all of it), `ambient` (`apply_gates`, the block format and every constant), `config` (`Settings`, every section, `idle_exit_minutes`), `paths` (`ProfilePaths`, `pipe_name`, `unix_socket_path`, `lock_file`), `project` (`resolve_project`), `journal` (`Journal`, the metric names, `summarize`), `spool`, `eval`, `traits`, `types`, `error`, `testing` (`FakeEmbedder`, `FakeProvider`, `SimpleChunker`).
- **Fixtures:** `fixtures/capture/hook_payloads/` (`INDEX.json`), `fixtures/capture/ambient_recall_cases.json`, `fixtures/capture/transcript/transcript_cases.json` and the transcripts beside it, `fixtures/capture/rust/` (`at14_session.jsonl`, `cursor_normalized_payloads.json`, written by stream G), `fixtures/retrieval/` (for `doctor_eval`). Read `fixtures/capture/README.md` first.
- **Version 2's daemon-side behaviour:** `git show origin/feature/v2-implementations:<path>` for ambient recall, the injection records and the journal. Port behaviour, not structure.
- **Crate sources:** `interprocess-2.4.8/`, `tokio-1.54.1/` and `windows-sys-0.61.2/` under `~/.cargo/registry/src/`.

## Build

1. **The entry point.** A public function the `daemon` subcommand calls in phase 3, taking the profile's `ProfilePaths` and `Settings`, a model loader, and a `SharingOps` implementation. It returns when the daemon shuts down or exits idle.
   - The model loader is injectable. The real one uses `sm-embed`: `verify` on the model directory (B-52), then `GemmaProvider` and `ModelTokenizer`. Tests pass `FakeProvider` and `SimpleChunker`, and loaders that are slow, fail with `model_missing`, or fail with `model_integrity`.
   - No process-global state that would stop two daemons for two profiles running in one test process.
2. **The endpoint and the lock** (CONTRACTS 3, 3.1 steps 1 to 5; B-44; spike decisions 1 and 5).
   - `daemon.lock`: open without truncating, `try_lock` every 25 ms for up to 2 s, otherwise log one line and return success. Never write to the file. Hold it for the daemon's life.
   - With the lock held, remove a leftover Unix socket, then bind. Windows: the first pipe instance with `FILE_FLAG_FIRST_PIPE_INSTANCE` and a security descriptor that admits only the current user. Verify in the `interprocess` 2.4.8 source how a descriptor is passed to the listener, and record it. A bind that fails is retried every 25 ms for up to 2 s, then the daemon fails with a log line.
   - Answer `hello` with `loading` as soon as the endpoint is bound. Create the query session first, keep it for life, then mark the daemon `ready`. A loader error moves it to `failed` with `state_error` (B-36).
   - Choose between `tokio` and threads, and record why. If you need a feature of a pinned crate that the root does not enable (for example `interprocess`'s tokio support), add it in your own `Cargo.toml` and record it; a new crate is a lead decision.
3. **Requests** (CONTRACTS 4 to 6).
   - Read each frame, then `Envelope::peek` before a full decode. Another `v` or `build` gets `Response::version_mismatch` for every op except `shutdown`. A request for another profile gets `invalid_input`. A malformed frame gets `invalid_input` and the connection stays usable when the frame boundary is intact.
   - `Op::validate()` before dispatch. Every error is a `CoreError` with its code, message and next action.
   - Connections are served concurrently. A slow op (`regions` with coordinates took about 13 s at 2,000 memories, c-report section 9) never delays `hello`, `ambient_recall` or `capture` on another connection.
   - `deadline_ms`: work still running past it is abandoned with `timeout`. Check it at least after embedding and after search.
   - States: in `loading` or `failed`, requests that need the model get `daemon_loading` or the load error; the rest are served (B-36). Writes need the chunker, which needs the tokenizer file: decide whether writes are accepted while the model loads, and record it. Never store a memory without its chunks.
   - `shutdown`: reply `shutting_down`, stop accepting, close the endpoint, return. The lock is released last.
   - **Idle exit:** with `daemon.idle_exit_minutes` above 0, exit after that long with no request and an empty queue. 0 means never.
   - `daemon.log` in the profile folder: one write call per line (spike decision 3). Decide and record how it is bounded.
4. **Project resolution.** Resolve each request's `cwd` with `project::resolve_project`, cached per `cwd`, and upsert the project. `project_unresolved` for the home folder, a filesystem root, or an unreadable folder. Reads default to the current project; `scope: all` must be explicit (SPEC 9).
5. **The 13 tools** (SPEC 8, CONTRACTS 5). Every write passes `Access::Local` (B-72). Every reply fits `limit` or `output.token_budget` (B-28) and uses `MemoryView`.
   - `remember`: 1 to 100 items (B-29). Redact with `sm-capture`'s `RegexSecretScanner` before storing and report the kinds (AT-9). Chunk with `chunk.max_tokens` and `chunk.overlap`, and no index cap: `ChunkParams.index_cap` is `None` for an explicit `remember` and `capture.max_index_tokens` only for captures (SPEC 16). A duplicate returns outcome `duplicate` with the existing id. Journal `chunks_per_memory`.
   - `recall` (SPEC 6 steps 1 to 6): cut the query to the chunk limit, embed it with the query session, exact search in the scope, best chunk per memory, filters, top `k` (default `recall.k`), results under `recall.similarity_floor` marked weak and still returned. In `auto` mode, a query that looks like an identifier also runs FTS5 as a quoted phrase, and exact matches not already present are appended as keyword hits, never score-fused. `keyword` mode needs no model. Journal `recall_result_count` and `recall_weak_result_total`. Remember each session's latest ranks for `reinforce` (B-32).
   - `related`: vector neighbours and the two-hop walk over links and anchors (`related_to_memory`, `related_to_anchor`).
   - `revise`, `forget` (B-30), `link` (with stale detection: read the head commit from the repository's files, never by running git on a request path, and pass `head_only(head)`; when the head cannot be read, `stale` is absent, B-37), `consolidate` (B-31, B-73), `reinforce` (B-32, journal `reinforced_memory_rank`).
   - `regions` and `journey` through `sm-spatial`, with the run cached by `input_hash` (B-46). Coordinates only when asked (B-47). Any edit, forget or re-embed in a project drops its cached runs. Labels stored against `(run, region)` (B-32).
   - `status`: daemon state and `state_error`, model, index stamp and `pending_embeddings`, counts per project, and, when asked, the 7-day medians and 95th percentiles from `journal::summarize`.
   - `share` and `sync`: through `SharingOps` (item 10).
6. **Embedding** (SPEC 3 rules 4 and 5; SPEC 5; B-27; B-51 to B-54).
   - **The query session:** created first, kept for life, `embed.query_threads` threads, one inference at a time. Nothing else uses it.
   - **The bulk queue:** writes commit their chunks as pending and wake the queue. One bulk session with `cpu.budget` × physical cores threads (`sm-embed` lowers its priority), batches of `embed.batch_size`, one `set_embeddings` transaction per batch. Opened when there is work and released when the queue is empty, so idle memory stays under 1 GB. If opening a session per `remember` would break the 2-second budget, keep the session for a short grace period after the queue empties; choose it, record it, and journal nothing new.
   - Chunks left pending by an earlier run are queued at start.
   - `embedder_cold_start_seconds` once per start.
   - **Index stamp** (AT-11): call `configure_index` at start with the configured model and chunk settings. On a mismatch, search fails with `index_mismatch` until `reindex` has run.
7. **Ambient recall** (CONTRACTS 7, SPEC 6, B-15, B-37).
   - Cut the prompt to `ambient::QUERY_MAX_TOKENS`, embed it, search the current project with `MemoryFilter::default()` (active only), then `ambient::apply_gates` with `ambient.top_k` and the floor.
   - Record the injected ids in `injections` only when the reply is built before the deadline. Write them off the reply path, so the reply is not delayed by the write.
   - Reply `gate`, `text`, `injected`, `top_scores` and `project`. The block is exactly what `ambient` builds (AT-17).
   - Do not journal `hook_gate_total`, `hook_top_score` or `hook_latency_ms`: the hook does (B-34).
   - Prune `injections` older than 30 days at start and once a day (SPEC 4 "Housekeeping").
8. **Capture** (CONTRACTS 7; SPEC 16; g-report section 2).
   - Reply `accepted` once the request is queued, then run `sm_capture::capture` off the reply path.
   - One `SessionState` and one transcript offset per session; store `new_offset` when it is `Some`. Bound the number of sessions held, and record how.
   - `commit_sha` from the repository's files, as for links (B-63).
   - Store each candidate with `Candidate::to_new_memory` through the same path as `remember`, provenance source `capture`. The store's dedupe applies; a capture the hook spooled after its budget ran out may also arrive on the abandoned connection, and the dedupe covers it.
   - Journal `capture_cap_hit_total` and `capture_size_tokens` (convert characters to tokens with the chunker's `count_tokens`, and record how).
9. **Spool drain** (CONTRACTS 8; B-25, B-33, B-66). At start and every 60 s: `remove_stale_tmp`, then `list_spool` in name order; dispatch each capture through the same path as a socket request; remove each file after its capture is queued; move a file that cannot be read or parsed to `spool/failed/` with `spool::quarantine`.
10. **`SharingOps`** (B-79). Define `pub trait SharingOps: Send + Sync` in `sm-daemon` with `share`, `sync`, `migrate`, `import` and `export`. Each takes a context and the op's parameters and returns the op's reply type from `sm_core::protocol`. The context holds the `Store`, the `Chunker`, the `SecretScanner`, `remember`'s `ChunkParams`, and the request's resolved project and `ProjectIdentity` when there is one. `share` and `sync` outside a repository fail with `not_git_repository` before the trait is called. After each call, wake the embedding queue and drop the cached maps of the affected projects. The daemon cannot be built without an implementation; your tests use a fake that records its calls. List the real implementation under "Unfinished": the lead writes it over `sm-share` in phase 3.
11. **The other ops.**
    - `reindex`: back up first (B-71), then `Store::reindex`, queue the embedding, reply `ReindexReport`.
    - `project_rename`, `project_delete` (`confirm_required` lists what would go, B-30).
    - `doctor_eval`: load the set and the optional baseline with `sm_core::eval`, run each query through the recall path, reply `EvalReport` with nDCG@10, hit-at-5, mean reciprocal rank and the score distributions of right and wrong results.
    - `import`, `export` and `migrate`: through `SharingOps`.

## Acceptance tests and other tests

Start the daemon in the test process on its own temporary profile folder (`SPATIAL_MEMORY_HOME` is process-wide, so pass `ProfilePaths` directly), and talk to it over the real endpoint with a small client built from `sm_core::protocol` and `interprocess`. Use `FakeProvider` and `SimpleChunker` unless a test needs the structured chunker with `WhitespaceTokenizer`.

- **Every op** over the socket: one success and the main error for each of the 25 ops in CONTRACTS 5, with the reply `kind` checked. Unknown fields are refused (B-28).
- **Every hook event, daemon side:** `ambient_recall` with each applicable case in `ambient_recall_cases.json`; `capture` with every applicable payload in `hook_payloads/` for `post-tool-use`, `pre-compact` and `stop`; `session-start` needs no daemon, so state that in a test comment. A case that cannot apply is listed in your report with the reason.
- **CONTRACTS 3.1 and 4:** two daemons started together on one profile give one server and one clean exit; a leftover Unix socket is removed; a `v` or `build` mismatch gets `version_mismatch` and `shutdown` still works; a request for another profile gets `invalid_input`; `hello` answers `loading` while a slow loader runs; `shutdown` releases the lock so a new daemon can start.
- **Windows endpoint:** the pipe refuses a second first instance; its descriptor admits only the current user (read it back and check it).
- **AT-3** (daemon side): 100 warm `ambient_recall` calls over the socket with the fake embedder, median and 95th percentile, and again while a bulk import of at least 2,000 chunks runs; the bulk session never runs more threads than the budget (`ThreadReport`). The real-model figure is phase 3.
- **AT-4:** a 50,000-character `remember` succeeds, produces several chunks, and its chunks are embedded by the queue.
- **AT-5:** two readers and one writer over the socket for 60 seconds; no lock error reaches a client.
- **AT-9** (`remember` side): fake credentials from the capture fixtures are redacted and the reply lists the kinds.
- **AT-10:** with a model directory that has one changed byte, `sm-embed`'s `verify` fails, the daemon is `failed` with `model_integrity` in `hello` and `status`, model requests fail with that error, and `forget`, `status` and keyword `recall` are still served (B-36). Use a small file set against the real `verify_files` if `sm-embed` allows it, otherwise an injected loader error; say which.
- **AT-11:** changing the model id or the chunk size blocks search with `index_mismatch` until `reindex` completes.
- **AT-12:** a memory stored from project A's folder is not returned from project B without `scope: all`, and two daemons on two profiles never see each other's memories.
- **AT-13, AT-14, AT-16, AT-17** (daemon side, end to end over the socket): the tool payloads store strong-signal text with capture provenance and nothing for weak text, with credentials redacted; the transcript cases store what PostToolUse missed and not what it already captured; the two tiers are stored as active with importance at least 0.4 and 0.7; a capture over the index cap is stored whole, indexed up to the cap, and journalled as a cap hit; the ambient block is exactly `AmbientReply.text`.
- **AT-15** (daemon side): a capture written with `spool::write_spool` while no daemon runs is processed when the daemon starts and its file is gone; a corrupt file moves to `spool/failed/`; a stale `tmp/` file is removed; a file written while the daemon runs is processed within one drain interval (make the interval injectable for the test).
- **SPEC 6:** an identifier query returns keyword hits appended after vector hits and marked; weak results are marked, not dropped; no repeats within a session for ambient recall; injections are not recorded after the deadline.
- **SPEC 7:** `regions` twice on the same data returns the cached run; an edit drops it; coordinates only when asked.
- **B-79:** each sharing op reaches the fake `SharingOps` with the resolved project, then the queue wakes and the maps are dropped.

Run the endpoint, lock and spool tests on Linux in Docker as well; the Unix socket path is only tested there.
