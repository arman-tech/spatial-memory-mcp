# Stream B: `sm-embed`

## Goal

Build `crates/sm-embed`: EmbeddingGemma 2 through ONNX Runtime behind `sm_core::traits::{EmbedderProvider, Embedder}`, the structure-aware `Chunker`, the model download with SHA-256 checks, the integrity check the daemon runs at start, and the thread budget and priority rules of SPEC 5. You also own the one integration test binary that loads the real model.

**SPEC 14 done when:** "Embedding self-test and chunker tests pass; model download verifies checksums; the thread budget and priority rules of section 5 work."

## You own

- `crates/sm-embed/` (new crate; depends on `sm-core` only, with `sm-core` `test-support` in `[dev-dependencies]`).
- Its integration test binary, `crates/sm-embed/tests/real_model.rs` (or one file of your naming): the only test in the workspace that loads the real model.

## Read

- **SPEC:** section 5 (all of it: the pinned model table, chunking rules 1 to 6, configuration, "Using the machine's cores" rules 1 to 7, the measurements), section 6 step 1 (query cut to 128 tokens for the hook, to the chunk limit for recall), section 11 (AT-1, AT-4, AT-10, the "Daemon ready from cold", memory and `remember` budgets, test conventions), section 13 (the rows on Windows ONNX Runtime, the community export, below-normal priority, long prompts), section 16 "Results on the owner's machine" and "Follow-up spikes s7b, s9 and s10".
- **CONTRACTS.md:** sections 2 and 9 (`Embedder`, `EmbedderProvider`, `Chunker`; titles cut to `MAX_TITLE_TOKENS` 32; every method returns `sm_core::error::Result`).
- **DECISIONS:** B-3 (model cache), B-22 (ort features), B-23 (arctic revision not pinned), B-39 (seven traits), B-41 (the arctic feature carries MPL-2.0 through `option-ext`; flagged, keep it behind the feature), B-42 (ureq 3.5.2 with rustls for the download), B-43 (`sm-cli` takes `sm-embed` in phase 3 for `setup` and `doctor`).
- **SPIKE.md** section a, and the spike program `%USERPROFILE%\Documents\code-repo\spatial-memory-spikes\spike-embed\src\main.rs`: how ort rc.15 loaded the model on Windows (inputs, the three empty feature tensors, padding, output name), the self-test scores, 0.35 s load, 662 MB peak. A reference; write the crate fresh.
- **sm-core:** `traits` (the three traits and `SessionKind`, `DocInput`, `ChunkParams`, `ChunkPlan`, `EmbedderInfo`), `config` (`EmbedModel`, `EMBEDDINGGEMMA_REVISION`, `Settings::query_threads`, `Settings::bulk_threads`, `embed.spinning`, `embed.batch_size`, `chunk.*`, `capture.max_index_tokens`), `paths::model_dir`, `error` (`model_missing`, `model_integrity`), `testing::SimpleChunker` (the contract your chunker must also meet).
- **Crate sources:** `~/.cargo/registry/src/*/ort-2.0.0-rc.15/`, `tokenizers-0.23.2/`, `fastembed-7.7.0/`, `ureq-3.5.2/`. Verify every builder method (threads, spinning, optimisation level, thread manager) there before using it.

## Build

1. **Model files and integrity** (SPEC 5 "Model, pinned"; AT-10).
   - The three files, their sizes and SHA-256 values from SPEC 5, as constants in this crate.
   - `verify` checks size, then SHA-256, of every file in a model folder and returns `model_missing` or `model_integrity` with the file name and a next action (SPEC 8 error shape).
   - The daemon (stream E) calls it before loading. Measure how long the full check takes on this machine. If a full SHA-256 of 1.08 GB every start breaks the 5-second cold-start budget, record the measurement and propose a scheme that still catches AT-10's one changed byte; until the lead decides, check in full.
2. **Download** (B-42, B-43). With ureq 3.5.2 and rustls, from `https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/<revision>/<path>` at the pinned revision.
   - Stream each file into a temporary name inside the target model folder, hashing as it writes. Rename into place only when size and SHA-256 match. On a mismatch, delete the partial file and return `model_integrity`.
   - The target is the per-user cache (`sm_core::paths::model_dir`), never the repository or a worktree.
   - Skip a file that is already present and verified.
   - Expose it as a library function. `setup` and `doctor` in `sm-cli` call it in phase 3.
   - Tests serve good and corrupted bytes from a local `std::net::TcpListener` in the test itself. No test touches the network.
3. **Embedder** (SPEC 5 table).
   - `ort` with graph optimisation level 3, intra-op threads set explicitly, inputs `input_ids` and `attention_mask` as `i64`, and `image_features`, `video_features`, `audio_features` as `f32` of shape 0 × 512.
   - Output `sentence_embedding`, batch × 768, already unit length. Check that it is, and report if not.
   - Query prompt `task: search result | query: {text}`; document prompt `title: {title or "none"} | text: {chunk}`. Pad on the right with token 0.
   - `embed_query(text, max_tokens)` cuts the query to `max_tokens` tokens before the prompt is added.
   - `embed_documents` cuts each title to 32 tokens, embeds in batches of `embed.batch_size`, and never sends an input longer than the configured chunk limit plus the prompt and title (SPEC 5 rule 6).
   - The tokenizer is loaded once per provider and shared by its sessions.
4. **Provider and threads** (SPEC 5 "Using the machine's cores", spike decision 5).
   - `open(SessionKind::Query, threads)` and `open(SessionKind::Bulk, threads)` each create their own ONNX Runtime session with that many intra-op threads.
   - The daemon passes `Settings::query_threads(physical)` and `Settings::bulk_threads(physical)`, with physical cores from `num_cpus::get_physical()`. Never more threads than physical cores (rules 1 and 2).
   - Spinning off unless `embed.spinning` is on (rule 5), for intra-op and inter-op threads.
   - **Bulk sessions run below normal priority** (rule 4): on Windows `THREAD_PRIORITY_BELOW_NORMAL`, on Linux a positive nice value per thread. Apply it to every thread the bulk session runs, for example through ort's thread-manager hook if rc.15 has one; verify in the source. You may enable extra `windows-sys` features in your own `Cargo.toml`.
   - If priority cannot be set on a platform, apply SPEC 13's fallback: halve the processor budget for bulk work. Report it with evidence.
   - Expose a way to count the threads a session is running, so the daemon's AT-3 test can prove it never exceeds the budget.
5. **Chunker** (SPEC 5 "Chunking rules" 1 to 6).
   - Count tokens with the model's tokenizer.
   - A memory at or under the limit is one chunk.
   - Split on Markdown headings, then blank-line paragraphs, then sentences. Fenced code blocks stay whole. Pack adjacent pieces up to the limit, never across a heading.
   - A single piece over the limit splits into token windows with `chunk.overlap` (one eighth) overlap.
   - Stop at `index_cap` tokens of content (`capture.max_index_tokens`, 8,192): `ChunkPlan` reports `total_tokens`, `indexed_tokens` and `capped` (AT-17's cap hit).
   - Keep the splitting logic testable with a whitespace token counter, so unit tests do not need `tokenizer.json`. The integration binary tests it with the real tokenizer.
6. **Fallback model** (D-5, B-23, B-41). Feature `arctic-embed-m`: Snowflake Arctic Embed M through `fastembed` 7.7.0 behind the same traits, with fastembed's cache folder set under the per-user model cache. The feature is off by default. It must compile and pass clippy under `--all-features`.
7. **Numbers for `doctor`** (SPEC 5 rule 7). A function that reports chunks per second for the bulk session and query latency for the query session at the configured budget, on a fixed sample. It reports; it tunes nothing. Do not build `doctor --bench` or a worker pool.

## Acceptance tests and other tests

**Unit tests** use `FakeEmbedder`, `FakeProvider` and a whitespace token counter; none loads the model. They cover the chunking rules, `verify` on synthetic files (one changed byte gives `model_integrity`), the download against the local server, the thread-count arithmetic, and the batch and title cut rules.

**The real-model integration binary** loads the model once and runs every case in sequence. It is skipped, with one printed line, when the model folder is absent. Cases:

- **AT-1**: Venus 0.675, Mars 0.853, Jupiter 0.747, Saturn 0.778, each within 0.01, for "Which planet is known as the Red Planet?" against the four documents of README.md's first example at the pinned revision, each with the document prompt. A text embedded alone and inside a padded batch gives cosine ≥ 0.9999.
- **AT-4** (embedding part): a 50,000-character memory is chunked with the real tokenizer into several chunks, every chunk is embedded at the default batch size, and the process's peak memory stays under 2 GB. Read the peak from the operating system (`PeakWorkingSetSize` on Windows, `VmHWM` in `/proc/self/status` on Linux).
- **AT-2** (embedding part): embed `fixtures/retrieval/retrieval_eval_set.json` with the real chunker, rank by cosine with a document's best chunk as its score, and gate with `sm_core::eval` against `retrieval_quality_baseline.json`. Report nDCG@10, mean reciprocal rank and hit-at-5. Phase 3 reruns AT-2 through the daemon.
- **AT-10** (real files): the installed model passes `verify`.
- **D-10, SPEC 5 rules 1, 2 and 4**: a bulk session opened with n threads runs no more than n, and its threads run below normal priority. Read the priority back from the operating system.
- **Budgets**: model load time, idle memory after load (1 GB budget), 128-token query latency on the query session, and bulk chunks per second at the default budget. Report each with CPU, core count and build profile.

Run the integration binary on Windows. In Docker without the model it must report itself skipped and pass. Then run it on Linux with the model mounted read-only, which is not a copy: add `-v "%USERPROFILE%/AppData/Local/spatial-memory/_models/embeddinggemma-2/daa72c51243991dfcaf9f9137d2c573d8f7790c0:/model:ro" -e SPATIAL_MEMORY_MODEL_DIR=/model` to the Docker command in RULES.md section 6, and run the integration binary with `--release` so the timings mean something. Report both runs.
