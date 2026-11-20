# Stream B report: `sm-embed`

Branch `arman-tech/stream-b-embed`, from `v3` at `ea82480`. All runs below are on the committed sources.

## 1. Done when

**Yes.** The embedding self-test (acceptance test AT-1) and the chunker tests pass, the model download verifies Secure Hash Algorithm 256-bit (SHA-256) checksums, and the SPEC 5 thread budget and priority rules (decision D-10) read back correctly from the operating system on Windows and Linux.

## 2. Files built

- `crates/sm-embed/Cargo.toml`: the crate manifest. Feature `arctic-embed-m` is off by default. Extra `windows-sys` features: `Win32_System_Diagnostics_ToolHelp` and `Win32_System_ProcessStatus`. `libc` on Unix.
- `crates/sm-embed/src/lib.rs`: module list and re-exports.
- `crates/sm-embed/src/model.rs`: the three pinned files with their sizes and SHA-256 values, plus `verify`, `verify_files`, `file_is_verified`, `hash_file` and the `model_missing` and `model_integrity` errors (AT-10).
- `crates/sm-embed/src/download.rs`: `download_model` and `download_files`. Built on ureq with rustls; streams into a temporary file, hashes as it writes, renames only on a match, and skips files that are already verified (B-42, B-43).
- `crates/sm-embed/src/gemma.rs`: `GemmaProvider` and `GemmaEmbedder`, which implement `EmbedderProvider` and `Embedder` over Open Neural Network Exchange (ONNX) Runtime through `ort`. Also contains the bulk runner thread and `embed_in_batches`.
- `crates/sm-embed/src/input.rs`: prompts, the query and title cuts, the batch plan, right-padding with token 0, and the output split with its unit-length check.
- `crates/sm-embed/src/tokenize.rs`: the `TextTokenizer` trait, `ModelTokenizer` (backed by `tokenizers`), `WhitespaceTokenizer` for unit tests, and character-boundary-safe cuts.
- `crates/sm-embed/src/chunker.rs`: `StructuredChunker`, covering SPEC 5 chunking rules 1 to 6.
- `crates/sm-embed/src/threads.rs`: the clamp to physical cores, the priority probe, the per-session thread plan, the SPEC 13 halving fallback, `CountingThreads` (ort's thread-manager hook) and `ThreadReport`.
- `crates/sm-embed/src/os.rs`: lowers a thread and reads back thread priority, process thread count, current and peak memory, and the CPU name. Windows and Linux.
- `crates/sm-embed/src/runtime.rs`: the one ONNX Runtime environment per process, with telemetry off.
- `crates/sm-embed/src/doctor.rs`: `doctor_numbers` (SPEC 5 rule 7), on a fixed sample.
- `crates/sm-embed/src/arctic.rs`: Snowflake Arctic Embed M through `fastembed` 7.7.0, behind feature `arctic-embed-m` (D-5, B-23).
- `crates/sm-embed/tests/real_model.rs`: the one test binary that loads the real model. It runs every case in sequence and skips with one line when the model is absent.
- `Cargo.lock`: 293 package versions added, none removed, no existing version changed. Twelve existing entries gain a version qualifier on a dependency name, because a second version of that crate now exists. `futures-util` 0.3.34 lists four more dependencies, from features turned on by the new crates.
- `docs/v3/streams/b-report.md`: this report.

## 3. Test commands and exact results

### Windows (PowerShell; Ryzen 9 9950X3D; Windows 11 Pro 10.0.26300)

`cargo fmt --check`: exit 0.

`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`: exit 0.

`cargo test --workspace --all-features --locked` (debug). The model is installed, so `real_model` ran in full:

```
sm_core      test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_embed     test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
real_model   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.75s
doc-tests    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test -p sm-embed --all-features --locked --release --test real_model -- --nocapture`:

```
real_model   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.02s
```

### Linux (Docker `rust:1.95-trixie`, without the model)

The command is RULES section 6, with the source mounted read-only and this stream's own target volume `sm-v3-target-b`.

`cargo test -p sm-embed --all-features --locked`:

```
sm_embed     test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
real_model: SKIPPED: the model folder /root/.cache/spatial-memory/_models/embeddinggemma-2/daa72c51243991dfcaf9f9137d2c573d8f7790c0 is absent; set SPATIAL_MEMORY_MODEL_DIR or run `spatial-memory setup`
real_model   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test --workspace --all-features --locked`. Linux runs 97 `sm_core` tests against 96 on Windows; the extra one is a platform-specific test in `sm-core`.

```
sm_core      test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
sm_embed     test result: ok. 54 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
real_model   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s   (skipped, same line as above)
doc-tests    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
doc-tests    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`: exit 0.

### Linux (Docker, model mounted read-only, release)

Added flags: `-v "%USERPROFILE%/AppData/Local/spatial-memory/_models/embeddinggemma-2/daa72c51243991dfcaf9f9137d2c573d8f7790c0:/model:ro" -e SPATIAL_MEMORY_MODEL_DIR=/model`. Command: `cargo test -p sm-embed --release --test real_model --locked -- --nocapture`.

```
real_model   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 25.99s
```

## 4. Acceptance tests

Unit tests use the whitespace tokenizer and the `sm-core` fakes; none loads the model. The real-model cases are functions called in sequence by the single test `s11_real_model_cases_in_sequence` in `tests/real_model.rs`. Each prints its figures and a PASS line, and any failed assertion fails the test.

| Rule | Test names | Result |
|---|---|---|
| AT-1 (self-test scores, alone and in a padded batch) | `at1_scores_match_the_reference` (real) | Pass on Windows and Linux. Venus 0.675, Mars 0.853, Jupiter 0.747 and Saturn 0.778 equal the reference to 3 places. The lowest cosine between a text embedded alone and inside a padded batch is 0.999999. The token fingerprint `434a9df05b5b5fc5` matches. |
| AT-2, embedding part (retrieval quality) | `at2_retrieval_quality_holds` (real) | Pass on both. nDCG@10 (normalised discounted cumulative gain at 10) 0.9865 against a baseline of 0.9742; MRR (mean reciprocal rank) 1.0000 against 0.9667; hit@5 1.0000 against 1.0000. 15 queries. |
| AT-4, embedding part (50,000-character memory, peak under 2 GB) | `at4_long_memory_stays_under_2_gb` (real) | Pass on both. 43 chunks; 9,097 tokens in all, 8,192 indexed, capped. Peak 1,416 MB on Windows and 1,438 MB on Linux, with the query and bulk sessions both loaded. |
| AT-10 (integrity: one changed byte, missing file, download) | `at10_one_changed_byte_is_model_integrity`, `at10_wrong_size_is_model_integrity_before_any_hash`, `at10_missing_file_is_model_missing`, `at10_verify_accepts_matching_files`, `at10_load_refuses_a_missing_model`, `at10_download_with_one_changed_byte_is_model_integrity_and_leaves_nothing`, `at10_download_of_wrong_length_is_model_integrity`, `at10_installed_model_verifies` (real) | Pass on both. |
| D-10, SPEC 5 rules 1, 2 and 4 (a bulk session with n threads runs no more than n, below normal priority) | `d10_counting_threads_count_and_lower_pool_threads`, `d10_bulk_sessions_are_lowered_within_the_budget`, `d10_a_lowered_thread_reads_back_below_normal`, `d10_lowering_is_one_step_however_often_it_runs`, `d10_bulk_threads_and_priority` (real), `d10_small_bulk_session_stays_within_n` (real) | Pass on both. Default budget: 12 asked, at most 12 running, 12 lowered, 0 refused. Every thread reads back at -1 (`THREAD_PRIORITY_BELOW_NORMAL`) on Windows and at nice 10 on Linux. With n = 2: at most 2 running, both lowered. |
| AT-3 (thread count the daemon reads) | `s5_thread_report_counts_the_calling_thread`, `s5_threads_never_exceed_physical_cores` | The counting this stream owes passes. AT-3 itself is the daemon's (stream E). |
| SPEC 13 (priority refused: halve the bulk budget) | `s13_bulk_budget_halves_when_priority_is_refused` | Pass. |
| SPEC 5 rule 7 (doctor numbers) | `s5_rule7_doctor_reports_both_numbers_at_the_budget`, `s5_rule7_sample_is_fixed`, `s11_doctor_numbers` (real) | Pass. |
| SPEC 11 budgets | `s11_load_and_idle_memory`, `s11_query_latency_at_128_tokens` (real) | Measured (section 5). Memory budgets are asserted; time budgets are reported. |

Other unit tests by rule:

- **Chunking, SPEC 5 rules 1 to 6:**
  - `s5_rule1_counts_with_the_given_tokenizer`, `s5_rule1_title_is_cut_to_32_tokens`
  - `s5_rule2_content_at_or_under_the_limit_is_one_chunk`
  - `s5_rule3_packs_adjacent_paragraphs_up_to_the_limit`, `s5_rule3_never_packs_across_a_heading`, `s5_rule3_code_fences_stay_whole`, `s5_rule3_an_oversized_paragraph_splits_into_sentences`
  - `s5_rule4_an_oversized_piece_becomes_overlapping_windows`
  - `s5_rule6_no_input_exceeds_limit_plus_prompt_and_title`
  - `s5_index_cap_stops_chunking_and_reports_tokens`, `s5_every_chunk_is_part_of_the_content_within_the_limit`, `s5_empty_and_whitespace_content_has_no_chunks`, `s5_invalid_parameters_are_invalid_input`, `s5_boundaries_respect_characters`
- **Embedder inputs and outputs:**
  - `s5_prompts_match_the_spec_table`, `s5_padding_is_on_the_right_with_zero`
  - `s5_batches_follow_batch_size_shortest_first`, `s5_batches_never_exceed_batch_size_and_keep_input_order`
  - `s5_output_must_be_unit_length_and_shaped`, `s5_a_short_batch_result_is_an_internal_error`
  - `s5_truncate_content_keeps_the_end_token`, `s5_whitespace_tokenizer_spans_and_encoding`, `s5_config_follows_the_settings`, `s5_the_environment_is_built_once_and_shared`
- **Query cut, SPEC 6 step 1:** `s6_query_is_cut_before_the_prompt_is_added`, `s6_cut_to_tokens_keeps_a_prefix`
- **Download:**
  - `s5_download_writes_verified_files_and_skips_them_next_time`, `s5_download_replaces_a_corrupt_file`
  - `s5_download_cut_short_is_an_error_and_leaves_nothing`, `s5_download_http_error_is_model_missing`
  - `s5_pinned_files_match_the_spec_table`, `s5_pinned_url_names_repository_and_revision`
  - All of these run against a local `TcpListener`; none touches the network.
- **Priority:** `s5_priority_direction_matches_the_os`, `s5_query_sessions_run_at_normal_priority`, `s5_os_readings_are_available`
- **Arctic (feature build), D-5:** `d5_query_prompt_and_bulk_fallback`, `d5_chunk_limit_leaves_room_for_the_prompt`, `d5_fallback_cache_is_under_the_model_cache_not_the_override`

## 5. Measured budgets

Machine: AMD Ryzen 9 9950X3D (AMD64 Family 26 Model 68), 16 physical cores and 32 logical. Defaults: query session 4 threads, bulk session 12 threads, batch size 4, chunk limit from `Settings::default()`.

| Figure | Budget | Windows 11 Pro, release | Linux (Docker Desktop, Debian trixie), release | Windows, debug |
|---|---|---|---|---|
| Full size and SHA-256 check of the 3 files (1,065 MB) | part of "ready from cold ≤ 5 s" | 0.51 s (2,095 MB/s) | 4.55 s (234 MB/s) | 0.52 s |
| Load: verify, tokenizer and runtime, then the query session | daemon ready from cold ≤ 5 s | 1.43 s (provider 1.07 s, query session 0.35 s) | **8.95 s, over budget** (provider 5.26 s, query session 3.69 s) | 1.92 s |
| Idle memory after load | 1 GB (1,024 MB) | 625 MB | 644 MB | 627 MB |
| Query latency at 128 tokens, 4 threads (median of 10, slowest) | SPEC test machine measured 200 ms | 47.3 ms, 49.6 ms | 78.1 ms, 81.1 ms | (debug, not meaningful) |
| AT-4 bulk: 43 chunks, 12 threads, batch 4, first batch included | — | 2.05 s (20.98 chunks/s) | 3.19 s (13.46 chunks/s) | 2.10 s |
| AT-4 peak memory, both sessions loaded | under 2 GB | 1,416 MB | 1,438 MB | 1,415 MB |
| Doctor: bulk chunks per second (16 chunks, 12 threads) | reported only | 17.26 chunks/s | 12.10 chunks/s | 15.34 chunks/s |
| Doctor: query median at 128 tokens (5 runs) | reported only | 47.2 ms | 79.8 ms | 48.2 ms |
| Peak memory over the whole binary (several sessions opened in turn) | — | 1,624 MB | 1,837 MB | — |

The Linux load is over budget because of where the model is read from, not because of the hashing:

- Docker Desktop shares the Windows folder into its Linux virtual machine over a 9p file share (`/proc/mounts`: `C:\134 /model 9p ro,...,aname=drvfs;path=C:\;...`).
- A plain `cat` of the 1,034 MB `model.onnx_data` through that mount took 4.653 s and 4.626 s on two runs, so caching did not help.
- `sha256sum` of 1,034 MB from `/dev/zero` in the same container took 0.755 s.
- Session creation reads the weights through the same mount, hence the 3.69 s query session against 0.35 s on Windows.

On Windows the measurements had a warm file cache. A cold-cache figure was not measured (section 9).

## 6. Decisions

Each item below states what was chosen, then the reasoning and evidence, with the SPEC section it touches.

### Model integrity and download

1. **Full verify at every load, inside `GemmaProvider::load`.**
   - **What:** `load` runs `model::verify` before anything else, so no caller can load an unchecked model.
   - **Why and evidence:** SPEC 9 says model files are "verified by SHA-256 before loading". A full check costs 0.51 s natively (section 5), well inside 5 s.
   - **Proposal for the lead, only if slow disks break the budget:** a "verified" stamp next to the files holding size, modification time and file id per file, with a full re-hash whenever any of them changes. A one-byte edit (AT-10) changes the modification time. Until the lead decides, the full check stays.
   - **SPEC:** 5 "Model, pinned"; 9; 11 AT-10.
2. **Error mapping.**
   - **What:**
     - Size is checked before hashing, so a truncated file fails without being read.
     - A missing or unreadable file is `model_missing`. Next action: run `spatial-memory setup` or set `SPATIAL_MEMORY_MODEL_DIR`.
     - A size mismatch, a hash mismatch, or a tokenizer whose `<pad>` is not id 0 is `model_integrity`. Next action: delete the model folder and run `spatial-memory setup` again. Every message names the file.
   - **SPEC:** 8.
3. **Download details.**
   - **What:**
     - The temporary file is `<name>.download-<pid>` in the target's own folder, so the rename never crosses a volume. A drop guard deletes it on every failure path.
     - An HTTP or network error, or a failure to write, is `model_missing`. A short body, wrong length or wrong hash is `model_integrity`, and nothing is left behind.
     - "Already verified" means a full size and SHA-256 check, the same as `verify`.
     - Connect and first-byte timeouts are set. The body has no timeout, because the weights file is 1 GB.
   - **SPEC:** B-42, B-43; SPEC 5.

### Sessions, threads and priority

4. **Optimisation level 3 is `GraphOptimizationLevel::Level3`.**
   - **What:** this maps to `ORT_ENABLE_LAYOUT` in ort rc.15 (`ort-2.0.0-rc.15/src/session/builder/impl_options.rs`). `All` maps to `ORT_ENABLE_ALL`.
   - **SPEC:** 5 table ("optimisation level 3").
5. **Session threads.**
   - **What:**
     - Each session has its own thread pool (`with_independent_thread_pool`).
     - Inter-op threads are 1, with sequential execution.
     - Intra-op and inter-op spinning are on only when `embed.spinning` is on.
     - Intra-op threads are clamped to `num_cpus::get_physical()`.
   - **SPEC:** 5 "Using the machine's cores", rules 1, 2 and 5.
6. **How bulk threads are lowered.**
   - **What:**
     - A bulk session gets its own runner thread. The runner lowers itself, then builds the session, so every pool thread ONNX Runtime starts goes through the `CountingThreads` thread-manager hook, which lowers and counts it. The runner then runs every batch.
     - The runner counts as one of the n intra-op threads, so the pool holds n − 1.
     - Query sessions run on the caller's thread at normal priority.
   - **Evidence:** "running at most 12" and "lowered 12" with 12 asked, on both systems.
   - **SPEC:** 5 rule 4; D-10.
7. **Priority values.**
   - **What:**
     - Windows uses `THREAD_PRIORITY_BELOW_NORMAL`.
     - Linux sets the thread's nice value to the process's own nice plus 10, capped at 19. Both are absolute targets, so lowering twice, or lowering a thread that inherited a lowered value from the thread that started it, still lands one step below normal.
   - **Evidence:** the first Linux run read back `[10, 19, 19, …]`, because pool threads inherit nice from the lowered runner and were then lowered again. After the fix every thread reads 10. Test: `d10_lowering_is_one_step_however_often_it_runs`.
   - **SPEC:** 5 rule 4.
8. **Priority probe.**
   - **What:** once per process, a short-lived thread is lowered and read back. If that is refused, every bulk session takes the SPEC 13 fallback (half the budget, at least 1) and reports why in `ThreadPlan`.
   - **SPEC:** 13 "below-normal priority".

### Embedder inputs and outputs

9. **Batches.**
   - **What:** inputs are sorted shortest first into batches of `embed.batch_size`, which limits padding. Results come back in input order. Padding is token 0 on the right, and the tokenizer's `<pad>` id is checked to be 0 at load.
   - **SPEC:** 5 table.
10. **Unit length.**
    - **What:** every output vector must have a norm within 1e-3 of 1, and the batch shape must match. Otherwise the result is an internal error, not a silently wrong vector.
    - **SPEC:** 5 ("already unit length; check").
11. **Over-limit inputs.**
    - **What:**
      - A chunk over the chunk limit passed to `embed_documents` is `invalid_input`. The chunker never produces one.
      - Should the prompt and title still push an input past the cap, the content is cut, keeping the end token, and counted in `truncated_inputs()`. That count was 0 in AT-4.
      - Queries are cut to the smaller of `max_tokens` and the chunk limit, before the prompt is added.
    - **SPEC:** 5 rule 6; 6 step 1.

### Chunker

12. **Chunker details.**
    - **What:**
      - Only ATX headings (`#` to `######`) split.
      - Sentences also end at a line break.
      - A fenced code block (three or more backticks or tildes; an unclosed fence runs to the end) is one piece. It is never split at its blank lines or `#` lines. If it alone is over the limit it becomes token windows, as SPEC 5 rule 4 requires for any oversized piece.
      - Pieces are never packed across a heading.
      - Token windows are never packed with neighbours.
      - `overlap` must be in [0, 1) and `max_tokens` at least 1, otherwise `invalid_input`.
      - `indexed_tokens` never exceeds the cap.
      - Every chunk is a trimmed, contiguous part of the content, and its token count is measured on its own text.
    - **SPEC:** 5 rules 1 to 6.

### Doctor

13. **Doctor sample.**
    - **What:** a fixed sample of 62 engineering words chosen by a 64-bit linear congruential generator, with a sentence break every 17 words. It is chunked at the configured limit and the first 16 chunks are timed, after one warm-up batch. The query figure is the median of 5 runs at 128 tokens, after one warm-up. Nothing is tuned from the numbers.
    - **SPEC:** 5 rule 7.

### Arctic fallback

14. **Arctic Embed M (feature `arctic-embed-m`).**
    - **What:**
      - fastembed's cache is `sm_core::paths::model_dir` for the Arctic model id and the revision `fastembed-7.7.0`, so it sits under the per-user model cache. It ignores `SPATIAL_MEMORY_MODEL_DIR`, which names the EmbeddingGemma folder.
      - The query prefix comes from the model card. There is no document prompt, so the title is unused.
      - The chunk limit must leave 32 tokens of the 512 for the prompt.
      - Spinning is set through session configuration.
      - A load failure is `model_missing`, with a network next action.
      - fastembed has no thread-creation hook, so bulk always takes the SPEC 13 fallback and reports it.
    - **SPEC:** D-5, B-23, B-41.

### Process-wide and testing

15. **One ONNX Runtime environment per process, telemetry off.**
    - **SPEC:** 9 "Local only".
16. **Real-model binary design.**
    - **What:**
      - One test loads the model once and runs every case in order.
      - Lines go straight to stdout, so they show under `--nocapture`.
      - Memory budgets are asserted.
      - Time budgets are printed but not asserted (build to the budgets; never tune to a test).
      - AT-4 runs with both the query and bulk sessions alive, as the daemon would.
    - **SPEC:** 11 "test conventions".
17. **Operating-system readings.**
    - **What:**
      - Physical cores come from `num_cpus::get_physical()`.
      - On Windows, peak and current memory come from `GetProcessMemoryInfo`, and thread counts and priorities from the Toolhelp snapshot and `GetThreadPriority`.
      - On Linux, memory comes from `/proc/self/status` (`VmHWM`, `VmRSS`). The thread count is the number of entries in `/proc/self/task`. A thread's nice value is field 19 of `/proc/self/task/<tid>/stat`, and it is set with `setpriority`.
    - **SPEC:** 11 AT-4.

## 7. Deviations

1. **The thread count is "at most", not "exactly".**
   - `EmbedderProvider::open` is documented in `sm-core` as opening "a session with exactly `threads` intra-op threads".
   - This crate clamps to physical cores (SPEC 5 rules 1 and 2), and halves a bulk budget when priority cannot be lowered (SPEC 13). So it can run fewer threads than asked.
   - `ThreadReport` shows what was asked and what was used.
   - Evidence: `s5_threads_never_exceed_physical_cores` and `s13_bulk_budget_halves_when_priority_is_refused`.
2. **Cold start over 5 s in Linux Docker with a mounted model.**
   - 8.95 s against the 5 s budget.
   - The cause is the 9p mount's read speed, measured in section 5. Natively on Windows the same load takes 1.43 s.
   - The full check was kept, as the stream instructions require until the lead decides.
3. **The arctic feature build contains native-tls and OpenSSL.**
   - The root pin of `fastembed` 7.7.0 takes its default features `onnx`, `ort-download-binaries-native-tls`, `hf-hub-native-tls` and `image-models`.
   - The feature build therefore links `native-tls` (on Linux, `openssl-sys` against the system libssl) and also contains `reqwest`, `tokio` and `image`. B-42 chose rustls everywhere else.
   - The default build is not affected. The Docker image has libssl, so `--all-features` builds there.

## 8. Contract changes requested

1. **Pin `fastembed` without its default features.**
   - Requested pin: `fastembed = { version = "=7.7.0", default-features = false, features = ["onnx", "hf-hub-rustls-tls", "ort-download-binaries-rustls-tls"] }`.
   - Reason: one TLS stack (B-42), no system OpenSSL on Linux, and no `image`, `rav1e` or `libfuzzer-sys` (the NCSA flag in section 10).
   - Evidence: fastembed 7.7.0's `[features]` defines `hf-hub-rustls-tls = ["hf-hub", "hf-hub?/rustls-tls"]` and `ort-download-binaries-rustls-tls = ["onnx", "ort/download-binaries", "ort/tls-rustls"]`.
   - Not tried, because the root pins are frozen. `arctic.rs` uses only `TextEmbedding`, which is under `onnx`.
2. **Change the wording of `EmbedderProvider::open`.**
   - From "exactly `threads` intra-op threads" to "at most `threads` intra-op threads, never more than the physical cores, halved for bulk work when priority cannot be lowered (SPEC 13)".
   - Reason: deviation 1.
3. **A SPEC decision on the verify stamp (not blocking).**
   - Keep the full SHA-256 at every start, or adopt the stamp in decision 1 for slow disks.
   - Evidence: 0.51 s natively and 4.55 s on Docker's 9p mount.

## 9. Unfinished or unverified

- **Arctic Embed M** compiles, passes clippy and passes its three unit tests. It was never loaded or run: no model download and no real embedding.
- **The download** was tested only against a local `TcpListener`, with good, corrupted, short and HTTP-error responses. It was never run against huggingface.co.
- **Cold-cache verify** on Windows was not measured; every Windows figure had a warm file cache. Native Linux (no Docker) and macOS were neither built nor run.
- **The daemon's own tests are not covered here.** AT-3's latency test is stream E's. AT-2 through the daemon is phase 3. This crate supplies the thread counts and the embedding part only.
- **Two defects were found and fixed in this stream, with tests:**
  - Linux pool threads were lowered twice (nice 19) (decision 7).
  - A memory-reading test raced with parallel tests and once read current memory above peak. It now reads current first; the peak only grows.

## 10. Licences

`Cargo.lock` gains 293 package versions:

- **107** are in the default `sm-embed` build.
- **173** come only with feature `arctic-embed-m`.
- **13** are in `Cargo.lock`, but `cargo tree --all-features --target all` does not show them anywhere in the workspace. They are cookie 0.18.2, cookie_store 0.22.1, deranged 0.5.9, document-features 0.2.12, hyper-rustls 0.27.10, litrs 1.0.1, num-conv 0.2.2, powerfmt 0.2.1, time 0.3.55, time-core 0.1.9, time-macros 0.2.32, tokio-rustls 0.26.6 and zlib-rs 0.6.8. All are MIT/Apache-2.0, Apache-2.0/ISC/MIT or Zlib.

The list was made with `cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked` and the same command without `--all-features`.

**Flagged.** All four are already in `docs/v3/LICENSES.md`:

| Crate | Version | Licence | Where | Note |
|---|---|---|---|---|
| `webpki-roots` | 1.0.12 | CDLA-Permissive-2.0 | **Default build**: `sm-embed` → `ureq` (rustls) for the model download, and the `ort-sys` build script | Permissive, but not on the list (it covers data: Mozilla's root certificates). |
| `webpki-root-certs` | 1.0.12 | CDLA-Permissive-2.0 | Feature only | As above. |
| `option-ext` | 0.2.0 | MPL-2.0 | Feature only: `fastembed` → `hf-hub` → `dirs` → `dirs-sys` | Weak copyleft; B-41 keeps it behind the feature. |
| `libfuzzer-sys` | 0.4.13 | (MIT OR Apache-2.0) AND NCSA | Feature only: `fastembed` → `image` → `ravif` → `rav1e` | NCSA is BSD-style and permissive, but not on the list. It goes away with contract change 1. |

**Not flagged, but worth a note:**

- `encoding_rs` 0.8.44 is (Apache-2.0 OR MIT) AND BSD-3-Clause; all three are permissive. Feature only.
- `r-efi` 5.3.0 offers LGPL-2.1-or-later only as one choice among MIT and Apache-2.0.
- `ryu` 1.0.23 is Apache-2.0 OR BSL-1.0.
- `ring` 0.17.14 is Apache-2.0 AND ISC.
- `onig_sys` 69.9.3 (MIT) compiles the Oniguruma C library into the default build, through `tokenizers`' `onig` feature (B-22).
- `ort-sys` downloads the ONNX Runtime binary (MIT) at build time.
- With the feature, `openssl-sys` links the system OpenSSL on Linux.

Full list. "Default build" is `yes` for the default `sm-embed` build, `feature only` for `arctic-embed-m` only, and `no` for entries locked but not resolved.

| Crate | Version | Licence | Default build |
|---|---|---|---|
| `adler2` | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | feature only |
| `ahash` | 0.8.12 | MIT OR Apache-2.0 | yes |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | yes |
| `aligned` | 0.4.3 | MIT OR Apache-2.0 | feature only |
| `aligned-vec` | 0.6.4 | MIT | feature only |
| `allocator-api2` | 0.2.21 | MIT OR Apache-2.0 | feature only |
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 | feature only |
| `arbitrary` | 1.5.0 | MIT OR Apache-2.0 | feature only |
| `arg_enum_proc_macro` | 0.3.4 | MIT | feature only |
| `arrayvec` | 0.7.8 | MIT OR Apache-2.0 | feature only |
| `as-slice` | 0.2.1 | MIT OR Apache-2.0 | feature only |
| `atomic-waker` | 1.1.2 | Apache-2.0 OR MIT | feature only |
| `autocfg` | 1.5.1 | Apache-2.0 OR MIT | yes |
| `av-scenechange` | 0.14.1 | MIT | feature only |
| `av1-grain` | 0.2.5 | BSD-2-Clause | feature only |
| `avif-serialize` | 0.8.9 | BSD-3-Clause | feature only |
| `base64` | 0.13.1 | MIT/Apache-2.0 | yes |
| `base64` | 0.22.1 | MIT OR Apache-2.0 | feature only |
| `base64` | 0.23.1 | MIT OR Apache-2.0 | yes |
| `base64ct` | 1.8.3 | Apache-2.0 OR MIT | feature only |
| `bitstream-io` | 4.10.0 | MIT/Apache-2.0 | feature only |
| `bit_field` | 0.10.4 | MIT OR Apache-2.0 | feature only |
| `built` | 0.8.1 | MIT | feature only |
| `bytemuck` | 1.25.2 | Zlib OR Apache-2.0 OR MIT | feature only |
| `byteorder-lite` | 0.1.0 | Unlicense OR MIT | feature only |
| `bytes` | 1.12.1 | MIT | yes |
| `castaway` | 0.2.4 | MIT | yes |
| `cc` | 1.11.2 | MIT OR Apache-2.0 | yes |
| `color_quant` | 1.1.0 | MIT | feature only |
| `compact_str` | 0.9.1 | MIT | yes |
| `console` | 0.16.6 | MIT | feature only |
| `cookie` | 0.18.2 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `cookie_store` | 0.22.1 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `core-foundation` | 0.10.1 | MIT OR Apache-2.0 | feature only |
| `core-foundation` | 0.9.4 | MIT OR Apache-2.0 | feature only |
| `core-foundation-sys` | 0.8.7 | MIT OR Apache-2.0 | feature only |
| `core_detect` | 1.0.0 | MIT/Apache-2.0 | feature only |
| `crc32fast` | 1.5.5 | MIT OR Apache-2.0 | feature only |
| `crossbeam-deque` | 0.8.8 | MIT OR Apache-2.0 | yes |
| `crossbeam-epoch` | 0.9.21 | MIT OR Apache-2.0 | yes |
| `crossbeam-utils` | 0.8.23 | MIT OR Apache-2.0 | yes |
| `crunchy` | 0.2.4 | MIT | feature only |
| `daachorse` | 3.0.3 | MIT OR Apache-2.0 | yes |
| `darling` | 0.20.11 | MIT | yes |
| `darling_core` | 0.20.11 | MIT | yes |
| `darling_macro` | 0.20.11 | MIT | yes |
| `dary_heap` | 0.3.9 | MIT OR Apache-2.0 | yes |
| `der` | 0.8.3 | Apache-2.0 OR MIT | feature only |
| `deranged` | 0.5.9 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `derive_builder` | 0.20.2 | MIT OR Apache-2.0 | yes |
| `derive_builder_core` | 0.20.2 | MIT OR Apache-2.0 | yes |
| `derive_builder_macro` | 0.20.2 | MIT OR Apache-2.0 | yes |
| `dirs` | 6.0.0 | MIT OR Apache-2.0 | feature only |
| `dirs-sys` | 0.5.0 | MIT OR Apache-2.0 | feature only |
| `displaydoc` | 0.2.7 | MIT OR Apache-2.0 | feature only |
| `document-features` | 0.2.12 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `either` | 1.19.0 | MIT OR Apache-2.0 | yes |
| `encode_unicode` | 1.0.0 | Apache-2.0 OR MIT | feature only |
| `encoding_rs` | 0.8.44 | (Apache-2.0 OR MIT) AND BSD-3-Clause | feature only |
| `equator` | 0.4.2 | MIT | feature only |
| `equator-macro` | 0.4.2 | MIT | feature only |
| `esaxx-rs` | 0.1.10 | Apache-2.0 | yes |
| `exr` | 1.74.3 | BSD-3-Clause | feature only |
| `fastembed` | 7.7.0 | Apache-2.0 | feature only |
| `fax` | 0.2.7 | MIT | feature only |
| `fdeflate` | 0.3.7 | MIT OR Apache-2.0 | feature only |
| `find-msvc-tools` | 0.1.15 | MIT OR Apache-2.0 | yes |
| `flate2` | 1.1.11 | MIT OR Apache-2.0 | feature only |
| `fnv` | 1.0.7 | Apache-2.0 / MIT | yes |
| `foldhash` | 0.2.0 | Zlib | feature only |
| `foreign-types` | 0.3.2 | MIT/Apache-2.0 | feature only |
| `foreign-types-shared` | 0.1.1 | MIT/Apache-2.0 | feature only |
| `form_urlencoded` | 1.2.2 | MIT OR Apache-2.0 | feature only |
| `futures-channel` | 0.3.34 | MIT OR Apache-2.0 | feature only |
| `futures-io` | 0.3.34 | MIT OR Apache-2.0 | feature only |
| `futures-macro` | 0.3.34 | MIT OR Apache-2.0 | feature only |
| `futures-sink` | 0.3.34 | MIT OR Apache-2.0 | feature only |
| `getrandom` | 0.2.17 | MIT OR Apache-2.0 | yes |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | yes |
| `gif` | 0.14.2 | MIT OR Apache-2.0 | feature only |
| `h2` | 0.4.20 | MIT | feature only |
| `half` | 2.7.1 | MIT OR Apache-2.0 | feature only |
| `hashbrown` | 0.16.1 | MIT OR Apache-2.0 | feature only |
| `hermit-abi` | 0.5.3 | MIT OR Apache-2.0 | yes |
| `hf-hub` | 0.5.0 | Apache-2.0 | feature only |
| `hmac-sha256` | 1.1.15 | ISC | yes |
| `http` | 1.5.0 | MIT OR Apache-2.0 | yes |
| `http-body` | 1.1.0 | MIT | feature only |
| `http-body-util` | 0.1.5 | MIT | feature only |
| `httparse` | 1.10.1 | MIT OR Apache-2.0 | yes |
| `hyper` | 1.12.0 | MIT | feature only |
| `hyper-rustls` | 0.27.10 | Apache-2.0 OR ISC OR MIT | no (in Cargo.lock, not in the resolved tree) |
| `hyper-tls` | 0.6.0 | MIT/Apache-2.0 | feature only |
| `hyper-util` | 0.1.21 | MIT | feature only |
| `icu_collections` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_locale_core` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_normalizer` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_normalizer_data` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_properties` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_properties_data` | 2.4.0 | Unicode-3.0 | feature only |
| `icu_provider` | 2.4.0 | Unicode-3.0 | feature only |
| `ident_case` | 1.0.1 | MIT/Apache-2.0 | yes |
| `idna` | 1.1.0 | MIT OR Apache-2.0 | feature only |
| `idna_adapter` | 1.2.2 | Apache-2.0 OR MIT | feature only |
| `image` | 0.25.10 | MIT OR Apache-2.0 | feature only |
| `image-webp` | 0.2.4 | MIT OR Apache-2.0 | feature only |
| `imgref` | 1.12.3 | CC0-1.0 OR Apache-2.0 | feature only |
| `indicatif` | 0.18.6 | MIT | feature only |
| `interpolate_name` | 0.2.4 | MIT | feature only |
| `ipnet` | 2.12.2 | MIT OR Apache-2.0 | feature only |
| `itertools` | 0.14.0 | MIT OR Apache-2.0 | yes |
| `jobserver` | 0.1.35 | MIT OR Apache-2.0 | feature only |
| `lebe` | 0.5.3 | BSD-3-Clause | feature only |
| `libfuzzer-sys` | 0.4.13 | (MIT OR Apache-2.0) AND NCSA | feature only |
| `libredox` | 0.1.25 | MIT | feature only |
| `litemap` | 0.8.3 | Unicode-3.0 | feature only |
| `litrs` | 1.0.1 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `log` | 0.4.35 | MIT OR Apache-2.0 | yes |
| `loop9` | 0.1.5 | MIT | feature only |
| `lzma-rust2` | 0.15.8 | Apache-2.0 | yes |
| `macro_rules_attribute` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | yes |
| `macro_rules_attribute-proc_macro` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | yes |
| `matrixmultiply` | 0.3.11 | MIT/Apache-2.0 | yes |
| `maybe-rayon` | 0.1.1 | MIT | feature only |
| `mime` | 0.3.18 | MIT OR Apache-2.0 | feature only |
| `minimal-lexical` | 0.2.1 | MIT/Apache-2.0 | yes |
| `miniz_oxide` | 0.8.9 | MIT OR Zlib OR Apache-2.0 | feature only |
| `miniz_oxide` | 0.9.1 | MIT OR Zlib OR Apache-2.0 | feature only |
| `mio` | 1.2.5 | MIT | feature only |
| `monostate` | 0.1.18 | MIT OR Apache-2.0 | yes |
| `monostate-impl` | 0.1.18 | MIT OR Apache-2.0 | yes |
| `moxcms` | 0.8.1 | BSD-3-Clause OR Apache-2.0 | feature only |
| `multiversion` | 0.8.1 | MIT OR Apache-2.0 | feature only |
| `multiversion-macros` | 0.8.1 | MIT OR Apache-2.0 | feature only |
| `multiversion_no_op` | 1.0.0 | Apache-2.0 OR MIT | feature only |
| `native-tls` | 0.2.18 | MIT OR Apache-2.0 | feature only |
| `ndarray` | 0.17.2 | MIT OR Apache-2.0 | yes |
| `new_debug_unreachable` | 1.0.6 | MIT | feature only |
| `nom` | 7.1.3 | MIT | yes |
| `nom` | 8.0.0 | MIT | feature only |
| `noop_proc_macro` | 0.3.0 | MIT | feature only |
| `no_std_io2` | 0.9.4 | Apache-2.0 OR MIT | feature only |
| `num-bigint` | 0.4.8 | MIT OR Apache-2.0 | feature only |
| `num-complex` | 0.4.6 | MIT OR Apache-2.0 | yes |
| `num-conv` | 0.2.2 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `num-derive` | 0.4.2 | MIT OR Apache-2.0 | feature only |
| `num-integer` | 0.1.47 | MIT OR Apache-2.0 | yes |
| `num-rational` | 0.4.2 | MIT OR Apache-2.0 | feature only |
| `num-traits` | 0.2.19 | MIT OR Apache-2.0 | yes |
| `num_cpus` | 1.17.0 | MIT OR Apache-2.0 | yes |
| `onig` | 6.5.3 | MIT | yes |
| `onig_sys` | 69.9.3 | MIT | yes |
| `openssl` | 0.10.82 | Apache-2.0 | feature only |
| `openssl-macros` | 0.1.1 | MIT/Apache-2.0 | feature only |
| `openssl-probe` | 0.2.2 | MIT OR Apache-2.0 | feature only |
| `openssl-sys` | 0.9.118 | MIT | feature only |
| `option-ext` | 0.2.0 | MPL-2.0 | feature only |
| `ort` | 2.0.0-rc.15 | MIT OR Apache-2.0 | yes |
| `ort-sys` | 2.0.0-rc.15 | MIT OR Apache-2.0 | yes |
| `paste` | 1.0.15 | MIT OR Apache-2.0 | yes |
| `pastey` | 0.1.1 | MIT OR Apache-2.0 | feature only |
| `pastey` | 0.2.3 | MIT OR Apache-2.0 | yes |
| `pem-rfc7468` | 1.0.0 | Apache-2.0 OR MIT | feature only |
| `percent-encoding` | 2.3.2 | MIT OR Apache-2.0 | yes |
| `pkg-config` | 0.3.34 | MIT OR Apache-2.0 | yes |
| `png` | 0.18.1 | MIT OR Apache-2.0 | feature only |
| `portable-atomic` | 1.15.0 | Apache-2.0 OR MIT | yes |
| `portable-atomic-util` | 0.2.8 | Apache-2.0 OR MIT | yes |
| `potential_utf` | 0.1.6 | Unicode-3.0 | feature only |
| `powerfmt` | 0.2.1 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `ppv-lite86` | 0.2.21 | MIT OR Apache-2.0 | yes |
| `profiling` | 1.0.18 | MIT OR Apache-2.0 | feature only |
| `profiling-procmacros` | 1.0.18 | MIT OR Apache-2.0 | feature only |
| `pxfm` | 0.1.30 | BSD-3-Clause OR Apache-2.0 | feature only |
| `qoi` | 0.4.1 | MIT/Apache-2.0 | feature only |
| `quick-error` | 2.0.1 | MIT/Apache-2.0 | feature only |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | yes |
| `rand` | 0.9.5 | MIT OR Apache-2.0 | yes |
| `rand_chacha` | 0.9.0 | MIT OR Apache-2.0 | yes |
| `rand_core` | 0.9.5 | MIT OR Apache-2.0 | yes |
| `rav1e` | 0.8.1 | BSD-2-Clause | feature only |
| `ravif` | 0.13.0 | BSD-3-Clause | feature only |
| `rawpointer` | 0.2.1 | MIT/Apache-2.0 | yes |
| `rayon` | 1.12.0 | MIT OR Apache-2.0 | yes |
| `rayon-cond` | 0.4.0 | Apache-2.0/MIT | yes |
| `rayon-core` | 1.13.0 | MIT OR Apache-2.0 | yes |
| `redox_users` | 0.5.3 | MIT | feature only |
| `regex` | 1.13.1 | MIT OR Apache-2.0 | yes |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | yes |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | yes |
| `reqwest` | 0.12.28 | MIT OR Apache-2.0 | feature only |
| `rgb` | 0.8.53 | MIT | feature only |
| `ring` | 0.17.14 | Apache-2.0 AND ISC | yes |
| `rustls` | 0.23.46 | Apache-2.0 OR ISC OR MIT | yes |
| `rustls-pki-types` | 1.16.0 | MIT OR Apache-2.0 | yes |
| `rustls-webpki` | 0.103.17 | ISC | yes |
| `ryu` | 1.0.23 | Apache-2.0 OR BSL-1.0 | yes |
| `safetensors` | 0.8.0 | Apache-2.0 | feature only |
| `schannel` | 0.1.30 | MIT | feature only |
| `scopeguard` | 1.2.0 | MIT OR Apache-2.0 | feature only |
| `security-framework` | 3.7.0 | MIT OR Apache-2.0 | feature only |
| `security-framework-sys` | 2.17.0 | MIT OR Apache-2.0 | feature only |
| `serde_urlencoded` | 0.7.1 | MIT/Apache-2.0 | feature only |
| `shlex` | 2.0.1 | MIT OR Apache-2.0 | yes |
| `simd-adler32` | 0.3.10 | MIT | feature only |
| `simdutf8` | 0.1.5 | MIT OR Apache-2.0 | feature only |
| `simd_helpers` | 0.1.0 | MIT | feature only |
| `sm-embed` | 3.0.0 | MIT | yes |
| `smallvec` | 1.16.5 | MIT OR Apache-2.0 | yes |
| `socket2` | 0.6.5 | MIT OR Apache-2.0 | feature only |
| `spm_precompiled` | 0.1.4 | Apache-2.0 | yes |
| `stable_deref_trait` | 1.2.1 | MIT OR Apache-2.0 | feature only |
| `static_assertions` | 1.1.0 | MIT OR Apache-2.0 | yes |
| `strsim` | 0.11.1 | MIT | yes |
| `subtle` | 2.6.1 | BSD-3-Clause | yes |
| `syn` | 2.0.119 | MIT OR Apache-2.0 | yes |
| `sync_wrapper` | 1.0.2 | Apache-2.0 | feature only |
| `synstructure` | 0.14.0 | MIT | feature only |
| `system-configuration` | 0.7.0 | MIT OR Apache-2.0 | feature only |
| `system-configuration-sys` | 0.6.0 | MIT OR Apache-2.0 | feature only |
| `target-features` | 0.1.6 | MIT OR Apache-2.0 | feature only |
| `tiff` | 0.11.3 | MIT | feature only |
| `time` | 0.3.55 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `time-core` | 0.1.9 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `time-macros` | 0.2.32 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `tinystr` | 0.8.5 | Unicode-3.0 | feature only |
| `tokenizers` | 0.23.2 | Apache-2.0 | yes |
| `tokio` | 1.54.1 | MIT | feature only |
| `tokio-native-tls` | 0.3.1 | MIT | feature only |
| `tokio-rustls` | 0.26.6 | MIT OR Apache-2.0 | no (in Cargo.lock, not in the resolved tree) |
| `tokio-util` | 0.7.20 | MIT | feature only |
| `tower` | 0.5.3 | MIT | feature only |
| `tower-http` | 0.6.11 | MIT | feature only |
| `tower-layer` | 0.3.3 | MIT | feature only |
| `tower-service` | 0.3.3 | MIT | feature only |
| `tracing` | 0.1.44 | MIT | feature only |
| `tracing-core` | 0.1.36 | MIT | feature only |
| `try-lock` | 0.2.5 | MIT | feature only |
| `unicode-normalization-alignments` | 0.1.12 | MIT/Apache-2.0 | yes |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | yes |
| `unicode-width` | 0.2.3 | MIT OR Apache-2.0 | feature only |
| `unicode_categories` | 0.1.1 | MIT OR Apache-2.0 | yes |
| `unit-prefix` | 0.5.2 | MIT | feature only |
| `untrusted` | 0.9.0 | ISC | yes |
| `ureq` | 3.5.2 | MIT OR Apache-2.0 | yes |
| `ureq-proto` | 0.7.1 | MIT OR Apache-2.0 | yes |
| `url` | 2.5.8 | MIT OR Apache-2.0 | feature only |
| `utf8-zero` | 0.8.1 | MIT OR Apache-2.0 | yes |
| `utf8_iter` | 1.1.0 | Apache-2.0 OR MIT | feature only |
| `utf_types` | 1.0.0 | Apache-2.0 OR MIT | feature only |
| `vcpkg` | 0.2.15 | MIT/Apache-2.0 | feature only |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | yes |
| `v_frame` | 0.3.9 | BSD-2-Clause | feature only |
| `want` | 0.3.2 | MIT | feature only |
| `wasi` | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | yes |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | yes |
| `wasm-bindgen-futures` | 0.4.80 | MIT OR Apache-2.0 | feature only |
| `wasm-streams` | 0.4.2 | MIT OR Apache-2.0 | feature only |
| `web-sys` | 0.3.107 | MIT OR Apache-2.0 | feature only |
| `web-time` | 1.1.0 | MIT OR Apache-2.0 | feature only |
| `webpki-root-certs` | 1.0.12 | CDLA-Permissive-2.0 | feature only |
| `webpki-roots` | 1.0.12 | CDLA-Permissive-2.0 | yes |
| `weezl` | 0.1.12 | MIT OR Apache-2.0 | feature only |
| `windows-registry` | 0.6.1 | MIT OR Apache-2.0 | feature only |
| `windows-result` | 0.4.1 | MIT OR Apache-2.0 | feature only |
| `windows-strings` | 0.5.1 | MIT OR Apache-2.0 | feature only |
| `windows-sys` | 0.52.0 | MIT OR Apache-2.0 | yes |
| `windows-targets` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_aarch64_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_aarch64_msvc` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_i686_gnu` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_i686_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_i686_msvc` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_x86_64_gnu` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_x86_64_gnullvm` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `windows_x86_64_msvc` | 0.52.6 | MIT OR Apache-2.0 | yes |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | yes |
| `writeable` | 0.6.4 | Unicode-3.0 | feature only |
| `y4m` | 0.8.0 | MIT | feature only |
| `yoke` | 0.8.4 | Unicode-3.0 | feature only |
| `yoke-derive` | 0.8.4 | Unicode-3.0 | feature only |
| `zerocopy` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | yes |
| `zerocopy-derive` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | yes |
| `zerofrom` | 0.1.8 | Unicode-3.0 | feature only |
| `zerofrom-derive` | 0.1.8 | Unicode-3.0 | feature only |
| `zeroize` | 1.9.1 | Apache-2.0 OR MIT | yes |
| `zerotrie` | 0.2.5 | Unicode-3.0 | feature only |
| `zerovec` | 0.11.9 | Unicode-3.0 | feature only |
| `zerovec-derive` | 0.11.7 | Unicode-3.0 | feature only |
| `zlib-rs` | 0.6.8 | Zlib | no (in Cargo.lock, not in the resolved tree) |
| `zune-core` | 0.5.3 | MIT OR Apache-2.0 OR Zlib | feature only |
| `zune-inflate` | 0.2.54 | MIT OR Apache-2.0 OR Zlib | feature only |
| `zune-jpeg` | 0.5.15 | MIT OR Apache-2.0 OR Zlib | feature only |
