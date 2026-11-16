# Decisions taken during the version 3.0 build

Each entry: identifier, date, the decision, the evidence, and the SPEC rule it touches. The SPEC (`docs/v3/SPEC.md`) stays the source of truth for what to build; this file records where the build had to choose, deviate or apply a section 13 fallback.

## B-1 Rust toolchain pinned to 1.95.0 (2026-11-16)

- **Decision.** `rust-toolchain.toml` pins channel `1.95.0` with the `rustfmt` and `clippy` components.
- **Evidence.** The spikes ran on rustc 1.92.0. `cargo info holomap@0.3.0` reports `rust-version: 1.93`, so 1.92 cannot build the projection crate the SPEC names (section 7). `ort@2.0.0-rc.15` needs 1.92, `rusqlite@0.43.0` needs 1.89, `rmcp@3.10.0` needs 1.88. 1.95.0 is installed on the build machine and is the newest stable installed there.
- **Touches.** SPEC 15 (tested versions). The tested crate versions are unchanged.

## B-2 Cargo runs through PowerShell on the build machine (2026-11-16)

- **Decision.** Every `cargo` command on this Windows machine runs through `powershell.exe -NoProfile -Command "cargo ..."`, never through Git Bash. Every worker's task specification carries this rule.
- **Evidence.** SPIKE-WINDOWS.md (s7b, s9, s10 follow-up): Git's `link.exe` shadows the MSVC linker in Git Bash. Confirmed in phase 0: `where.exe link.exe` in PowerShell lists only Git's copy, yet `cargo +1.95 build` of a fresh binary crate through PowerShell links and runs (`Hello, world!`), because rustc locates the MSVC linker itself.
- **Touches.** Build prompt, phase 0, spike result (6).

## B-3 Model cache location and the phase 0 copy (2026-11-16)

- **Decision.** The per-user model cache is `<platform cache dir>/spatial-memory/_models/<model id>/<revision>/`, where the platform cache dir is `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache` on Linux, and `~/Library/Caches` on macOS (the rules of `directories::BaseDirs::cache_dir`, now resolved in `sm-core` without that crate, B-41). `SPATIAL_MEMORY_MODEL_DIR` overrides the whole path (for air-gapped machines, SPEC 9 "Local only"). Profile names must match `^[a-z0-9][a-z0-9-]{0,31}$`, so no profile can be called `_models`.
- **Phase 0 copy.** The three pinned files were already downloaded and verified by the spike repository (`spatial-memory-spikes/model`, outside this repository). Phase 0 copied them into `C:\Users\jon\AppData\Local\spatial-memory\_models\embeddinggemma-2\daa72c51243991dfcaf9f9137d2c573d8f7790c0\` and re-verified all three SHA-256 values against SPEC 5 instead of downloading 1.1 GB again. The download code path is still built and tested by stream B against a local file server, without re-downloading the model.
- **Touches.** SPEC 4 "Locations", SPEC 5 "Model, pinned", build prompt "Download the model once".

## B-4 Data directory layout (2026-11-16)

- **Decision.** The profile directory is `<platform data-local dir>/spatial-memory/<profile>/` (`%LOCALAPPDATA%` on Windows, `$XDG_DATA_HOME` or `~/.local/share` on Linux, `~/Library/Application Support` on macOS). When `SPATIAL_MEMORY_HOME` is set, it replaces `<platform data-local dir>/spatial-memory`, so the profile directory is `$SPATIAL_MEMORY_HOME/<profile>/`. It does not move the model cache.
- **Why.** The SPEC says the variable "overrides" the location without saying at which level. Replacing the `spatial-memory` root keeps profiles separate under the override, which AT-12 requires.
- **Touches.** SPEC 4 "Locations", SPEC 3 rule 7.

## B-5 The plugin registers SessionStart (2026-11-16)

- **Decision.** The plugin's `hooks/hooks.json` registers all five hooks, SessionStart included (matcher `startup|resume`, exec form, 5-second timeout). `setup`, for people without the plugin, registers the same five in `~/.claude/settings.json`.
- **Why.** SPEC 16 "Registration" keeps version 2's rule that SessionStart goes through settings and not through the plugin. That rule worked around a Claude Code bug on Windows at 2.1.37 (SPEC 16 "Not verified"), and it contradicts two other SPEC 16 rules: the plugin registers "every hook", and `setup` refuses to run while the plugin is enabled, so a plugin user could never get SessionStart through settings. Phase 0 spike f ran a plugin SessionStart hook on Windows with Claude Code 2.1.325 for both `startup` and `resume` (SPIKE.md, f).
- **Touches.** SPEC 16 "Registration", "How the server and hooks are registered", "Not verified".

## B-6 The prompt hook stays in exec form (2026-11-16)

- **Decision.** Every hook, the prompt hook included, is an exec-form command hook (`"command": "spatial-memory"`, `"args": ["hook", "<event>", "--client", "claude-code"]`). No `mcp_tool` hook is registered.
- **Evidence.** SPIKE.md, e and f. The program starts and exits by bare name in 5.4 ms median and 6.0 ms at the 95th percentile, inside SPEC 16's 25 ms and 60 ms rule. Over 50 prompts in Claude Code, an `mcp_tool` prompt hook took 14 ms median against 31 ms for exec form, a saving of about 17 ms (6% of the 300 ms budget). But its plain-text result was not injected into the model's context on Claude Code 2.1.325, and it is skipped when the MCP server is not connected. Something is lost, so the build prompt's rule keeps exec form.
- **Touches.** SPEC 16 "Considered and not the default", build prompt phase 0 step 1 f.

## B-7 vec1 vendored at a recorded check-in (2026-11-16)

- **Decision.** `vendor/vec1/vec1.c` is SQLite check-in `60cf9357e8a60bc1de92068ed10e090cf432552f8040ba9f0294155051416347` (trunk, 2026-11-12), file `ext/vec1/vec1.c`, version 0.8, SHA-256 `93b8e1848ea3c11faaa9b533338a7e0a2404edb0acee47507b4fbee60c7a4b62`, unchanged. A `.gitattributes` rule turns off line-ending conversion for `vendor/`. `sm-store` compiles it with `cc` as two objects (scalar, and AVX2 with `/arch:AVX2` or `-mavx2 -mfma`), as spike b did.
- **Evidence.** SPIKE.md, b: the spike copy's SHA3-256 resolves in SQLite's Fossil repository to that check-in. Correctness 20 of 20 against brute force on Rust 1.95.
- **Touches.** SPEC 13 "vec1 compiled into the binary" (fallback not needed), SPEC 15 tested versions.

## B-8 Stream F owns a new crate, `sm-share` (2026-11-16)

- **Decision.** Stream F builds `crates/sm-share`: the shared-file format (strict front-matter writer and parser, path naming, `.gitattributes` rule), `share`, `sync` change detection and application, and `migrate` from a version 1 or 2 JSON export, as library functions over `sm-store` and the `sm-core` traits (`Chunker`, `SecretScanner`). It depends on `sm-core` and `sm-store`, with `sm-capture` as a development dependency for the real scanner in tests. `sm-daemon` depends on it like the other crates. The lead wires the `share`, `sync` and `migrate` requests into `sm-daemon` in phase 3. Whole-store `export` and `import` were added to the crate by B-77, and the wiring goes through B-79's `SharingOps` trait.
- **Why.** SPEC 3's crate table has no crate for sharing and migration, and SPEC 14 runs streams E and F at the same time. If both edited `sm-daemon`, they would conflict, and stream E would otherwise need a placeholder for handlers it cannot implement, which the build prompt forbids. A separate crate keeps the streams apart; the extra edge (`sm-share` → `sm-store`) stays inside "sm-daemon depends on all of the above".
- **Touches.** SPEC 3 "Crates", SPEC 14 phase 2.

## B-9 What the restructure removes, moves and keeps (2026-11-16)

- **Removed from the `v3` working tree** (they remain on `master` and in history): `spatial_memory/`, `tests/`, `pyproject.toml`, `requirements.lock`, `scripts/` and `benchmarks/` (all Python), `.env.example` (Python settings names), and the Python files under `plugin/` (`plugin/hooks/*.py`, `plugin/scripts/find_python.*`).
- **Moved unchanged** with `git mv`: every file under `docs/` except `docs/v3/` into `docs/archive/python/`, plus `SPATIAL-MEMORY-ARCHITECTURE-DIAGRAMS.md` from the root, which also describes the Python architecture only.
- **Kept**: `LICENSE`, `logo/`, `.github/` (workflows untouched, as the build prompt requires), `CHANGELOG.md` (history; a 3.0.0 entry is added in phase 3), `CODE_OF_CONDUCT.md`, `SECURITY.md`, `MARKETING.md`, `README.md` and `CONTRIBUTING.md` (rewritten in phase 3), `.claude-plugin/marketplace.json`, `plugin/.claude-plugin/plugin.json`, `plugin/.mcp.json` and `plugin/hooks/hooks.json` (version 1 content until stream D replaces them).
- **Rewritten**: `.gitignore`, for a Rust workspace. The project-specific entries are kept, except `.spatial-memory/`, which is narrowed so test fixtures can contain a `.spatial-memory/` folder.
- **Touches.** Build prompt phase 0 step 3, SPEC 14 "Work lands on a v3 branch".

## B-10 Fixtures live in `fixtures/` at the repository root (2026-11-16)

- **Decision.** Ported fixtures sit under `fixtures/` (`fixtures/migration/`, `fixtures/retrieval/`, `fixtures/capture/`), outside any crate, so several crates' tests can read them through `env!("CARGO_MANIFEST_DIR")/../../fixtures`. Each folder's `README.md` records the git ref and path every file came from.
- **Why.** `tests/` held the Python tests being removed, and the capture fixtures serve stream G, stream E and the phase 3 acceptance tests.

## B-11 Configuration layering and limits (2026-11-16)

- **Decision.**
  - Two TOML files: `<data root>/config.toml` (shared settings plus `[[profile_rule]]` entries) and `<data root>/<profile>/config.toml`. The profile file is merged over the root file table by table, key by key (`config::merge_tables`); a profile file may not hold `[[profile_rule]]`.
  - Unknown keys and out-of-range values are `config_invalid`, never ignored.
  - `output.token_budget` has a floor of 100 (`MIN_OUTPUT_TOKEN_BUDGET`), and a tool's `limit` uses the same floor.
  - `k` and `ambient.top_k` are capped at 50 (`MAX_K`).
  - The ambient constants are fixed in code (`ambient.rs`): the four-word prompt minimum, 200-character lines, the 800-token block ceiling, the 1,500 ms hook deadline, the 128-token query cut and the 200 ms capture budget.
- **Why.** SPEC 9 "Profiles" and SPEC 15 name both files but not how they combine. Merging by key lets a profile change one value without copying the rest. Refusing unknown keys catches v1 settings names. The floors and caps stop a request that would return nothing useful or flood the context.
- **Evidence.** Tests `s9_profile_config_overrides_root_key_by_key`, `s5_out_of_range_values_are_config_invalid`, `s5_defaults_match_the_spec`.
- **Touches.** SPEC 6, 8 "Conventions", 9 "Profiles", 15 "Defaults".

## B-12 Migrated memory ids (2026-11-16)

- **Decision.** A v1 or v2 id that is a UUID is kept as is. Any other id becomes a UUID version 8 built from the first 16 bytes of `sha256("spatial-memory:v1-id:" + id)`, so a migration run twice gives the same ids and resumes rather than duplicating (`ids::MemoryId::from_v1`).
- **Why.** SPEC 4 says ids are UUID version 7. SPEC 10 says migration keeps the memories but does not say what happens to ids that are not UUIDs. A deterministic mapping keeps AT-8's "report matches the row count" true across re-runs.
- **Evidence.** Tests `s10_v1_uuid_ids_are_kept`, `s10_non_uuid_v1_ids_map_to_stable_v8`.
- **Touches.** SPEC 4, SPEC 10, AT-8.

## B-13 Evaluation fixtures, the gate and doctor --eval sets (2026-11-16)

- **Decision.**
  - AT-2 gates on v2's tolerances (`EPS_NDCG` 0.02, `EPS_MRR` 0.02, `EPS_HIT_AT_K` 0.07). A metric fails only when it falls below the baseline by more than its tolerance.
  - A `doctor --eval` set may omit `memories`; it then runs against the live store. A set must still have `queries`.
  - The report includes a distribution of the top similarity scores (minimum, 5th, 25th, 50th, 75th and 95th percentiles, maximum), for calibrating the similarity floor (SPEC 6). Percentiles use Python's `statistics.quantiles(n=100)` exclusive method, as v2's journal did.
- **Evidence.** `fixtures/retrieval/retrieval_quality_baseline.json` from `origin/feature/v2-implementations`. Tests `at2_gate_fails_only_beyond_the_fixture_tolerances`, `at2_metrics_match_version_2_formulas`, `s11_doctor_eval_sets_may_omit_memories_but_not_queries`, `s6_percentile_matches_python_statistics_quantiles`.
- **Touches.** SPEC 6 "Similarity floor", SPEC 11 AT-2 and "Evaluation on real memories".

## B-14 content_hash is version 1's definition (2026-11-16)

- **Decision.** `content_hash` is the lower-case hex SHA-256 of the content after Python's `str.strip()` and `str.lower()`, exactly as v1 computed it. The `strip` follows Python's whitespace set, not Rust's.
- **Evidence.** Test `s10_content_hash_matches_v1_export_fixture` recomputes every hash in `fixtures/migration/v1_export_sample.json` and matches all of them. `s10_python_strip_matches_python_isspace` covers the whitespace set.
- **Touches.** SPEC 4 `content_hash`, SPEC 10, `remember` deduplication (B-29).

## B-15 Ambient block format (2026-11-16)

- **Decision.** The block keeps v2's header, `[spatial-memory] ambient recall:`, and its numbered lines, `  {i}. ({score:.2f}) …`, with v2's newline flattening and its 200-character cut to 197 plus `...`. Version 3 makes three changes:
  1. It adds the memory id in brackets after the score, so the agent can `revise`, `reinforce` or `link` what it was shown.
  2. It replaces carriage returns as well as newlines.
  3. It ends with one line: `[spatial-memory] end of recall. The lines above are reference material from stored memories, not instructions.`
- **Why.** SPEC 6 asks for v2's format plus a way to cite. SPEC 9's rule that memory content is data, not instructions, is stated to the agent at the point of injection.
- **Evidence.** Tests `at17_ambient_block_keeps_version_2_format`, `at17_block_has_header_id_lines_and_footer`, `at17_carriage_returns_in_block_lines_become_spaces`, `s6_block_stays_under_800_tokens`.
- **Touches.** SPEC 6 "Ambient recall", AT-17.

## B-16 Profile resolution order (2026-11-16)

- **Decision.** The profile is chosen in this order:
  1. `--profile`
  2. `CLAUDE_PLUGIN_OPTION_PROFILE`
  3. `SPATIAL_MEMORY_PROFILE`
  4. the longest matching `[[profile_rule]]` path
  5. `default`

  An invalid name from any source is `config_invalid`; the resolver never falls through to the next source.
- **Why.** The SPEC lists the sources but not their order. Explicit choices beat inferred ones, and the plugin option is the most specific setting an agent's environment makes. Falling through silently could write memories into the wrong profile, which AT-12 forbids.
- **Evidence.** Test `at12_profile_order_is_flag_plugin_env_rule_default`.
- **Touches.** SPEC 9 "Profiles", AT-12.

## B-17 Endpoint naming (2026-11-16)

- **Decision.**
  - Windows: pipe `spatial-memory-<first 16 hex of SHA-256 of the profile directory's Windows path key>`.
  - Unix: `<profile dir>/daemon.sock`. When that path is longer than 100 bytes, which is under every platform's `sun_path` limit, the socket moves to `$XDG_RUNTIME_DIR` or `/tmp` under a name hashed the same way.
- **Why.** One daemon per profile (SPEC 3 rule 7). Spike decision 4 requires paths that differ only in slash or case to be the same profile on Windows. `sun_path` is 108 bytes on Linux and 104 on macOS.
- **Evidence.** Tests `spike1_pipe_name_ignores_slash_direction_and_case`, `s3_long_socket_paths_move_to_the_runtime_dir`.
- **Touches.** SPEC 3 "Transport", spike decisions 1 and 4.

## B-18 Microsecond timestamps; the title is optional (2026-11-16)

- **Decision.**
  - Timestamps are UTC with microsecond precision, written as RFC 3339 with `Z`. Input without an offset is read as UTC, and the journal writes v2's `+00:00` form.
  - `title` is optional. When a memory has none, the document prompt uses `none` (SPEC 5). Displays derive one from the first text line, cut to 80 characters (`types::derive_title`), without storing it.
- **Why.** v1 exports carry microsecond `isoformat()` values, some naive. SPEC 4 lists `title` as a column but `remember` does not require it, and SPEC 5's prompt already has a `none` case.
- **Evidence.** Tests `s10_v1_fixture_timestamps_round_trip_unchanged`, `s10_parses_v1_naive_timestamps_as_utc`, `s6_journal_form_matches_python_isoformat`, `s4_derive_title_takes_first_text_line`.
- **Touches.** SPEC 4, 5, 10.

## B-19 Provenance extensions and list-item rules (2026-11-16)

- **Decision.**
  - `Provenance` adds `tool` (the capturing tool, SPEC 16 "Type and provenance") and `extra`, which keeps v1's `source` and `metadata` on migration so nothing is lost (SPEC 10), to SPEC 4's key columns.
  - Anchor names (at most 256 characters) and tags (at most 64) may not be empty, have surrounding whitespace, or contain control characters, `,`, `[` or `]`. Tags are deduplicated.
- **Why.** Shared files write anchors and tags as one-line YAML lists (SPEC 9). Those characters would break the strict parser.
- **Evidence.** Tests `s4_anchor_refs_parse_and_print`, `s9_tags_are_validated_and_deduplicated`.
- **Touches.** SPEC 4, 9 "Shared file format", 10, 16.

## B-20 Project identity details (2026-11-16)

- **Decision.**
  - **Identity key.** A project's identity key is `git:<normalised remote>`, or `path:<root path key>` when there is no remote.
  - **Remote.** It comes from the repository's own config files, without running git. `upstream` is preferred, then `origin`, then the first remote. Linked worktrees (`.git` is a file) are followed to the common directory. The remote is normalised like v1, quirks included.
  - **Not a project.** Outside a repository, the nearest folder holding `.spatial-memory/project.toml` is the project, else the folder itself; the home folder and a filesystem root are `project_unresolved`.
  - **project.toml.** A `project.toml` that cannot be parsed turns capture off for that repository and adds a warning, rather than failing the request.
- **Why.**
  - SPEC 15 "What a project is": projects are repositories.
  - SPEC 16 "Shell-agnostic rules": no shelling out on the hook path.
  - A broken opt-out file is read as an opt-out, the conservative choice.
  - The v1 quirks are kept so that migrated v1 namespaces map to the same project.
- **Evidence.** Tests `s9_remote_forms_normalise_like_version_1`, `s9_remote_quirks_are_kept_from_version_1`, `s9_git_config_remote_priority_is_upstream_origin_first`, `s16_worktree_git_file_is_followed_to_the_common_dir`, `s16_project_toml_opts_out_of_capture_and_broken_file_is_conservative`, `s9_filesystem_root_is_not_a_project`.
- **Touches.** SPEC 9 "Scope", 15, 16.

## B-21 capture.ignore defaults (2026-11-16)

- **Decision.** By default, capture skips `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.keystore`, `*.jks`, `id_rsa*`, `id_dsa*`, `id_ecdsa*`, `id_ed25519*` and `.netrc`. The default tool list is `Edit`, `Write`, `Bash` and `NotebookEdit`.
- **Why.** SPEC 16 asks for an ignore list but does not give one. These files hold secrets by convention, and the redactor is a second line of defence, not the first.
- **Evidence.** Tests `s16_capture_ignore_patterns_cover_env_and_key_files`, `s16_glob_rules`.
- **Touches.** SPEC 16 "Configuration", SPEC 9 "Secrets".

## B-22 Workspace pins, extra crates and ort features (2026-11-16)

- **Decision.**
  - Every crate SPEC 15 tested is pinned at that version in the root `[workspace.dependencies]`: ort 2.0.0-rc.15, tokenizers 0.23.2, fastembed 7.7.0, rmcp 3.10.0, rusqlite 0.43.0, hdbscan 0.12.0, holomap 0.3.0 and interprocess 2.4.8. `directories` 6.0.0, which SPEC 15 lists as "not tested; version checked only", is not used (B-41). vec1 0.8 is vendored (B-7). sqlite-vec 0.1.9 is SPEC 13's fallback and is not used.
  - Extra crates, at the newest stable version on the local registry on 2026-11-16: serde, serde_json, schemars (uuid1), thiserror, uuid (v7, v8, serde), sha2, toml, regex, tokio, windows-sys 0.61.2, libc, ndarray 0.17.2, num_cpus, cc 1.11.2, libsqlite3-sys 0.40.0 (bundled, the version rusqlite 0.43.0 requires) and tempfile.
  - ort is built with default features off and `std`, `ndarray`, `download-binaries`, `tls-rustls` and `api-24`, as spike a ran.
  - A new third-party crate is a lead decision, recorded here.
- **Why.** The build prompt says to verify versions against the registry and record any difference. None of SPEC 15's tested versions differs. The extras are the smallest set the SPEC's design needs.
- **Touches.** SPEC 15 "Tested versions", SPEC 13.

## B-23 The fallback model's revision is not pinned (2026-11-16)

- **Decision.** `embed.model = "arctic-embed-m"` loads Snowflake Arctic Embed M through fastembed 7.7.0, which downloads its own ONNX export and does not expose a Hugging Face revision. The index stamp records the model id `arctic-embed-m` and the fastembed version as the revision. EmbeddingGemma stays pinned to revision `daa72c51…` with SHA-256 checks (SPEC 5).
- **Why.** SPEC 5 pins the default model only. Pinning the fallback would mean bypassing fastembed's loader, which is outside the SPEC's fallback design (D-5).
- **Touches.** SPEC 5, D-5.

## B-24 Cursor and session-start output; probes that do not apply (2026-11-16)

- **Decision.**
  - For Cursor, `session-start` prints nothing, as v2's `translate_output_for_cursor` did for that event.
  - Two v2 ambient-switch probes, `content_none` and `content_int`, test Python-only coercions of `None` and integer values. The configuration types reject these inputs, so the probes are recorded in the fixture as not applicable rather than ported.
- **Evidence.** `fixtures/capture/event_aliases.json`; test `s16_session_start_output_matches_version_2`.
- **Touches.** SPEC 16 "Events", AT-17.

## B-25 The spool format lives in sm-core (2026-11-16)

- **Decision.** The spool's layout, file naming and write, list, read, quarantine and stale-file functions are in `sm_core::spool`, not in `sm-capture` as SPEC 3's crate table says. `sm-capture` keeps extraction, transcripts, signals, the scanner and overlap detection.
- **Why.** Two other crates use the spool: the hook client in `sm-cli` writes it and the daemon reads it. Neither should depend on the capture pipeline for a file format, and putting the format in `sm-core` freezes it with the other contracts.
- **Evidence.** Tests `s16_spool_write_is_tmp_then_rename`, `s16_spool_lists_in_order`, `s16_spool_removes_only_stale_tmp_files`, `s16_spool_dirs_are_user_only` (Unix).
- **Touches.** SPEC 3 "Crates", SPEC 16 "Never loses".

## B-26 Protocol framing, envelope and version handshake (2026-11-16)

- **Decision.**
  - Frames are a 4-byte little-endian length plus UTF-8 JSON, limited to 64 MiB.
  - A connection carries any number of request and response pairs, strictly in turn.
  - Every message carries `v` and `build`. A daemon of another protocol or build answers `version_mismatch` to everything except `shutdown`.
  - Frozen forever: the frame format, `v`, `build`, `id`, `call.op`, `shutdown` and `version_mismatch`.
  - CONTRACTS.md sections 3 and 4 describe the rest.
- **Why.** SPEC 3 says length-prefixed JSON with a protocol version and the restart-on-mismatch rule, without the frame details. Freezing the handshake fields means any future build can still replace an old daemon.
- **Evidence.** Tests `s3_frame_round_trip_and_size_limit`, `s3_version_mismatch_reply_shape`, `s3_op_tags_are_stable`, `s3_request_carries_version_cwd_and_session`.
- **Touches.** SPEC 3 "Transport", rules 1 and 5.

## B-27 Requests map synchronously; embedding is queued, with no job model (2026-11-16)

- **Decision.**
  - Every op is one request and one reply; there are no job ids or polling ops.
  - Writes return once the rows and chunks are committed. Their embedding goes onto the bulk queue, and keyword search finds them meanwhile.
  - `status` reports `index.pending_embeddings`. `import`, `migrate` and `reindex` report how many chunks they queued.
- **Why.** SPEC 5 says v3.0 uses one bulk embedding session. A job model would add ops, state and failure modes for no visible gain, because progress is already visible through `status`.
- **Touches.** SPEC 5 "Bulk", SPEC 8, SPEC 10.

## B-28 Strict tool parameters; limit is a token budget (2026-11-16)

- **Decision.**
  - Every tool parameter type has `deny_unknown_fields`, so an agent sending a v1 name such as `namespace` gets an error that names the valid fields.
  - List filters accept one value or an array.
  - `limit` is the reply's token budget, at least 100; result counts are `k`.
- **Why.** SPEC 8 "Conventions": schemas come from the Rust types, and every reply fits a token budget. Accepting one value or an array matches how agents send filters.
- **Evidence.** Test `s8_tool_args_parse_and_validate`.
- **Touches.** SPEC 8.

## B-29 remember takes a batch of 1 to 100 and deduplicates by content hash (2026-11-16)

- **Decision.** `remember` takes `memories: [1..100]`. An item whose `content_hash` already exists in the project is not stored again; its reply entry has outcome `duplicate` and the existing id.
- **Why.** SPEC 8 merges v1's `remember` and `remember_batch`. v1 refused exact duplicates by hash, and silently storing a second copy would defeat consolidation.
- **Touches.** SPEC 8, B-14.

## B-30 forget confirms through an error; a filter needs a criterion (2026-11-16)

- **Decision.**
  - `forget` without `confirm: true` deletes nothing. It fails with `confirm_required`, and the message says how many memories would be deleted.
  - A filter must name at least one criterion other than scope, so `forget {filter: {}}` cannot empty a project.
  - `project delete` follows the same rule.
- **Why.** SPEC 8 requires confirmation for bulk deletion. An error cannot be mistaken for success by an agent that skips reading the reply.
- **Touches.** SPEC 8, SPEC 9.

## B-31 consolidate applies only the listed groups (2026-11-16)

- **Decision.**
  - `consolidate` without `apply` proposes groups and changes nothing.
  - With `apply: true`, it acts only on the `groups` the caller passes. In each `{keep, merge}`, the `merge` memories become `superseded`; nothing is deleted. **Amended by B-73:** the merged memory's `supersedes_id` is not changed; its `provenance.extra.merged_into` records `keep`.
- **Why.** SPEC 8 says consolidation proposes and the agent confirms. Applying exactly what was reviewed avoids a race with memories added in between. Marking memories superseded keeps history, as SPEC 4's status model intends.
- **Touches.** SPEC 4 "status", SPEC 8.

## B-32 Region labels and reinforce ranks (2026-11-16)

- **Decision.**
  - A `regions` label is stored against `(run, region)`. A new clustering run starts without labels, because cluster numbers do not carry across runs.
  - `reinforce` records the rank the caller gives. Without one, it uses the rank the memory had in the session's latest `recall`, or `unknown`. The reply says which (`rank_source`).
- **Why.** SPEC 7 caches runs and SPEC 8 lets the agent label regions. A label moved onto a different cluster would mislead. v2's `reinforced_memory_rank` metric needs a rank, and the latest recall is the honest source when none is given.
- **Touches.** SPEC 7, SPEC 8, SPEC 11 journal metrics.

## B-33 Spool behaviour (2026-11-16)

- **Decision.**
  - **No cap and no fsync.** The spool has no size cap and does not fsync.
  - **Failed files.** A file the daemon cannot parse moves to `spool/failed/` and is never retried automatically.
  - **Stale tmp files.** `tmp/` files older than one hour are removed.
  - **Drain schedule.** The daemon drains `new/` when it starts, as SPEC 16 says. It also drains it every 60 seconds while running, so a capture spooled because the daemon was busy past the 200 ms budget is not stranded until the next restart. With `idle_exit_minutes = 0`, that restart may never come.
- **Why.**
  - The hook has 200 ms and must never lose a capture (AT-15). A cap would lose data. An fsync on Windows can exceed the budget.
  - A file that fails to parse once will fail again.
- **Touches.** SPEC 16 "Never loses", AT-15.

## B-34 The hook client journals its own gates and latency (2026-11-16)

- **Decision.**
  - The hook process, not the daemon, writes `hook_gate_total`, `hook_top_score` and `hook_latency_ms` for each prompt. It is the only process that knows whether its 1,500 ms deadline passed.
  - `hook_latency_ms` is a v3 addition, used to measure AT-3.
  - Each journal line is under 4,096 bytes and goes out in one write call. Labels are cut to 64 characters.
- **Evidence.** Spike decision 3; tests `s6_journal_lines_stay_under_4096_bytes`, `s6_hook_gates_have_stable_names`.
- **Touches.** SPEC 6, SPEC 11 "Journal", AT-3.

## B-35 migrate's default project (2026-11-16)

- **Decision.** In a v1 or v2 export, a record with no `project` field goes to `--default-project` when given. Otherwise it goes to the project of the working directory `migrate` runs in. If neither resolves, `migrate` fails with `project_unresolved` before writing anything.
- **Why.** SPEC 10 maps namespaces to projects but v1 records may lack one. Failing before writing is the conservative choice.
- **Touches.** SPEC 10, AT-8.

## B-36 The daemon's failed state (2026-11-16)

- **Decision.** `DaemonState` has `loading`, `ready` and `failed`. In `failed`, the model could not be loaded. `hello` and `status` carry the error in `state_error`. Requests that need the model fail with that error, such as `model_missing` or `model_integrity`. Requests that do not need it are still served, such as keyword-mode `recall`, `forget`, `link`, `status` and `export`.
- **Why.** SPEC 3 has the daemon load the model at start but does not say what happens if loading fails. A daemon that exits would be restarted by every hook and fail again in a loop. One that serves what it can and reports the cause is diagnosable through `status` and `doctor`.
- **Touches.** SPEC 3, SPEC 5 "Integrity", AT-10.

## B-37 Known protocol gaps accepted for 3.0 (2026-11-16)

- **Injection race.** The daemon records an injection when it builds the block. If the hook's deadline passes before the reply arrives, the block is not shown, but the memories still count as injected for the no-repeat rule in that session. The cost is that a memory may be held back once in a session. Fixing it would need a second round trip on the 300 ms path.
- **Stale-link ancestry.** `LinkView.stale` needs an ancestry check of `verified_commit` against the head without running git. Walking ancestry through git's object store is unspecified. Until stream A decides it, `stale` is absent when it cannot be decided, which the type allows.
- **Touches.** SPEC 6 "No repeats", SPEC 4 `links`.

## B-38 The Python queue-processor tests are not ported (2026-11-16)

- **Decision.** `tests/test_wp6_queue_processor.py` from master is not ported to `fixtures/capture/`. It tests v1's queue-processor module, whose job the v3 spool and the daemon's drain take over (B-25, B-33). The spool's behaviour is tested in `sm_core::spool` and, end to end, by AT-15.
- **Touches.** SPEC 16, SPEC 14 stream G.

## B-39 Seven traits, not six (2026-11-16)

- **Decision.** `sm_core::traits` has seven traits. `EmbedderProvider`, which opens query and bulk sessions with a thread count, is added to SPEC 3's six (`Embedder`, `Chunker`, `VectorIndex`, `Projector`, `Clusterer`, `SecretScanner`).
- **Why.** Spike decision 5 keeps one query session for the daemon's life, and SPEC 5 runs one bulk session at lower priority. The daemon needs a way to open each with its own thread count, and tests need a way to count sessions (`FakeProvider::opened`).
- **Touches.** SPEC 3 "Crates", SPEC 5 "Runtime".

## B-40 Test names cite their rule (2026-11-16)

- **Decision.** Test names start with the rule they test: `atN_` for an acceptance test, `dN_` for a SPEC 2 decision, `sN_` for a SPEC section, or `spikeN_` for a spike decision. Phase 0 renamed six `sm-core` tests that had no prefix.
- **Why.** The build prompt requires every test name to cite an AT, a D or a section, and `docs/v3/TRACE.md` is built from these prefixes.
- **Touches.** Build prompt section 2.

## B-41 `directories` dropped for a licence; the arctic feature carries MPL-2.0 (2026-11-16)

- **Finding.** `directories` 6.0.0 depends on `dirs-sys` 0.5.0, which depends on `option-ext` 0.2.0, licensed MPL-2.0. MPL-2.0 is weak copyleft, not on the build prompt's permissive list. It was the only non-permissive crate in the phase 0 dependency tree (`cargo metadata`).
- **Decision.**
  - `sm-core` no longer uses `directories`. `paths::base_dir` resolves the same two folders by the same rules:
    - Linux and the BSDs: `$XDG_DATA_HOME` or `$XDG_CACHE_HOME` when absolute, else `~/.local/share` or `~/.cache`.
    - macOS: `~/Library/Application Support` and `~/Library/Caches`.
    - Windows: `LOCALAPPDATA` when absolute, else `<home>\AppData\Local`.
  - The home folder is `std::env::home_dir`, which Rust 1.95 no longer deprecates and which follows the same `HOME`-then-password-database order as `dirs-sys` on Unix.
  - On Windows, `directories` called `SHGetKnownFolderPath(FOLDERID_LocalAppData)`. Windows sets `LOCALAPPDATA` from that known folder at logon, so the two agree on a normal session; reading the variable keeps `sm-core` under `#![forbid(unsafe_code)]`. Test `s4_platform_dirs_resolve_on_this_machine` checks the folder against `LOCALAPPDATA` on the build machine.
  - The default build is now all permissive. The `arctic-embed-m` feature (D-5) brings `option-ext` back through `fastembed` 7.7.0 → `hf-hub` 0.5.0 → `dirs` 6.0.0. That feature is off by default; `docs/v3/LICENSES.md` flags it for the owner.
- **Why.** The build prompt says permissive licences only. SPEC 15 lists `directories` as "not tested; version checked only", so dropping it loses no tested result, and the replacement is about 60 lines with no unsafe code.
- **Evidence.** `cargo metadata` before and after; `Cargo.lock` no longer contains `directories`, `dirs-sys` or `option-ext`. A scratch crate with the planned stream dependencies (`ort`, `tokenizers`, `rusqlite`, `hdbscan`, `holomap`, `interprocess`, `tokio`, `rmcp`, `regex`) resolves no `option-ext` without `fastembed` (`cargo tree -i option-ext`: no match) and resolves it through `hf-hub` with it. Tests `s4_platform_dirs_follow_the_xdg_rules`, `s4_platform_dirs_resolve_on_this_machine`.
- **Touches.** SPEC 4 "Locations", SPEC 15 "Tested versions", B-3, B-4, B-22, build prompt section 2 "Licences".

## B-42 Pins for the model download and the command line: ureq 3.5.2, clap 4.6.7 (2026-11-16)

- **Decision.** The root `[workspace.dependencies]` gains two pins:
  - `ureq = { version = "=3.5.2", default-features = false, features = ["rustls"] }` for `sm-embed`'s model download (SPEC 3 command table: `setup` downloads and verifies the model; SPEC 5 pins the hashes).
  - `clap = { version = "=4.6.7", default-features = false, features = ["std", "help", "usage", "error-context", "derive"] }` for the `spatial-memory` binary's subcommands and flags (SPEC 3 "Architecture", command table).
- **Why.**
  - `ort-sys`'s build script already pulls `ureq` 3.5.2 with `rustls` 0.23.46 and `ring` 0.17.14 to fetch ONNX Runtime (`download-binaries`, `tls-rustls`). Pinning the same version keeps one HTTP client and one TLS stack in the tree instead of adding a second (for example `reqwest` and `hyper`).
  - SPEC 15 names no HTTP or argument-parsing crate. `clap` with `derive` is the conventional choice; default features are off so no colour or terminal-size crates are pulled in.
- **Licence consequence.** Through `ureq`'s `rustls` feature, `webpki-roots` 1.0.12 (CDLA-Permissive-2.0) is now linked into the binary, not only used by a build script. Every other crate the two pins add is MIT, Apache-2.0, ISC or BSD-3-Clause (`subtle`). `docs/v3/LICENSES.md` is updated.
- **Evidence.** The phase 0 scratch crate resolves both pins together with the other planned dependencies: `cargo tree -e normal` lists `ureq` 3.5.2, `ureq-proto` 0.7.1, `rustls` 0.23.46, `ring` 0.17.14 (Apache-2.0 AND ISC), `rustls-webpki` 0.103.17 (ISC), `webpki-roots` 1.0.12, `clap` 4.6.7, `clap_builder` 4.6.7, `clap_derive` 4.6.7, `clap_lex` 1.1.1, `anstyle` 1.0.14, `strsim` 0.11.1, `heck` 0.5.0.
- **Touches.** SPEC 5, SPEC 15 "Tested versions", CONTRACTS.md section 1 (a new third-party crate is a lead decision), build prompt "Licences".

## B-43 `sm-cli` takes `sm-embed` in phase 3 (2026-11-16)

- **Decision.** `sm-cli` depends on `sm-embed` from phase 3, for `setup`'s model download and checksum step and for `doctor`'s model checks. In phase 1, stream D writes those steps against a trait private to `sm-cli` and tests them with a fake. The lead wires `sm-embed` behind that trait in phase 3, as for the `daemon` subcommand. CONTRACTS.md section 2 records it.
- **Why.** SPEC 3's command table says `setup` and `doctor` "download and verify the model", and SPEC 9 "Local only" makes that the one network access, and only `sm-embed` holds the download code and the pinned hashes. Duplicating them in `sm-cli` would let the two drift. Taking `sm-embed` in phase 1 would make stream D wait on stream B.
- **Touches.** SPEC 3 command table and "Crates", SPEC 9 "Local only", CONTRACTS.md section 2.

## B-44 Auto-start lock: the daemon holds an operating-system lock for life (2026-11-16)

- **Decision.** The lock in SPEC 3 rule 2 is an operating-system file lock on `<profile dir>/daemon.lock`, taken with `std::fs::File::try_lock`. CONTRACTS.md section 3.1 gives the full protocol. In short:
  - The daemon takes the lock (retrying every 25 ms for up to 2 s), holds it for life, and exits 0 if another daemon has it. Only then does it remove a stale Unix socket file and bind the endpoint.
  - A client that cannot connect tries the lock once. If it gets it, it spawns `<current_exe> daemon --profile <name>` detached and drops the lock at once; if not, a daemon is running or starting and it does not spawn.
  - The hook never waits. The MCP proxy and the command line poll connect plus `hello` every 50 ms for up to 10 s, then return `daemon_unavailable`.
  - The spawn is detached without `CREATE_BREAKAWAY_FROM_JOB` on Windows (spike decision 2) and with `setsid` on Unix.
  - The lock file stays empty, because a Windows lock also blocks reads.
- **Why.**
  - An operating-system lock is released when its holder dies, so a crashed daemon never leaves a stale lock to clean up. A create-new lock file with a staleness age would need a guess at the age and a race-free takeover.
  - Holding it for life also answers "is a daemon running?" without connecting, and makes a second daemon (from a race, or a person running `spatial-memory daemon` by hand) exit cleanly.
  - The client's single `try_lock` keeps two clients from spawning two daemons, as SPEC 3 rule 2 says. If both still spawn (one released the lock before the other tried), the second daemon fails the lock and exits 0, which is harmless.
  - Spike s8 measured a cold auto-start of 21 ms with a lock that refused a second daemon (SPEC, spike results), so the design is known to work.
- **Evidence.** `File::try_lock` and `TryLockError::WouldBlock` compile and behave on Rust 1.95.0 on Windows: a scratch program locked a file through one handle, saw `WouldBlock` through a second, then acquired it through the second after dropping the first. It uses `LockFileEx` on Windows and `flock` on Unix.
- **Touches.** SPEC 3 rules 2 and 3, spike decisions 1 and 2, CONTRACTS.md section 3.1, streams D and E.

## B-45 `sm-spatial` runs single-threaded; no thread count to pin (2026-11-16)

- **Decision.** `hdbscan` 0.12.0 is used with its default serial build; its `parallel` (rayon) feature stays off. `holomap` 0.3.0 has no threads. A `regions` run is therefore single-threaded wherever it is called from, and AT-6's "second thread count" is tested as the same run made concurrently from four threads.
- **Why.** SPEC 11 AT-6 asks for identical assignments on every run. With no thread pool there is no scheduling order to vary, and nothing to pin.
- **Evidence.** `holomap` uses exact nearest neighbours and draws every random number from one seeded PCG64 (permuted congruential generator) stream through `libm`. `at6_fixture_runs_are_identical_on_every_run_and_thread` matches two golden SHA-256 digests computed on Windows and met unchanged on Linux in Docker (stream C report, section 4).
- **Touches.** SPEC 7 step 2, SPEC 11 AT-6, `crates/sm-spatial/Cargo.toml`.

## B-46 The region pipeline: centrality, sample, nearest-centre assignment, numbering and cache key (2026-11-16)

- **Decision.**
  - **Memory vector.** The re-normalised mean of a memory's chunk vectors, summed in a fixed order. Memories are put in identifier order first, so a run depends on the set of memories, not the order passed in.
  - **Centrality** (SPEC 7 "Labels"). A member's centrality is its cosine to its region's centre, the re-normalised mean of all members' vectors in the original 768 dimensions; ties go by memory identifier. `RegionResult.members` is the whole region, ranked; `central(k)` gives the top k. The text is the `CENTRALITY` constant.
  - **Sample** (above 2,000 memories). Each identifier is keyed by `fnv1a64` of the seed (`ProjectParams.seed`, 42, as 8 little-endian bytes) followed by the identifier's 16 bytes; identifiers are sorted on that key, ties by identifier, and the first 2,000 are taken. Adding one memory changes at most one sample member.
  - **Assignment of the rest.** Each cluster's centre comes from its sample members only. Every memory outside the sample joins the cluster with the highest cosine in 768 dimensions, ties to the lower cluster, with no distance threshold (SPEC 7 pitfall). Sample members marked as noise stay unclustered; a sample with no cluster leaves every memory unclustered.
  - **Cluster parameters for a sampled run.** `ClusterParams::for_n(sample size)`, that is `for_n(2000)` (22 and 500), not `for_n(total)`. SPEC 7's formula is on n memories and does not say which n applies to a sample; `for_n(5000)` would allow a cluster of 1,250 of the 2,000 projected points (62.5%), past AT-7's 40% limit.
  - **Numbering.** Regions are numbered by size, largest first, ties to the region holding the smallest identifier. The numbers are local to a run (SPEC 4: cluster numbers are never keys). Under 30 memories the whole project is region 0 and `clustered` is false.
  - **`input_hash`.** SHA-256 (hex) of the `ALGORITHM` text (`sm-spatial regions v1; holomap 0.3.0; hdbscan 0.12.0`), every projection field, `max_points`, `min_memories`, the cluster formula and method texts, n and the sorted identifiers. It covers identifiers, not vectors, so the daemon must drop a cached run when a member is edited or re-embedded.
  - **Input checks.** Duplicate identifiers, a memory with no chunks, non-finite values, zero vectors and mismatched lengths are `invalid_input`. The clusterer checks its parameters before calling `hdbscan`, whose builder would otherwise print a warning to standard output, the MCP channel, and clamp them.
- **Why.** Each is the simplest deterministic rule where SPEC 7 is open. No random-number crate is pinned, so the sample uses the hash already in `sm-core`.
- **Evidence.** `s7_central_members_are_ranked_by_cosine_to_the_centre`, `s7_cap_samples_2000_and_assigns_the_rest_to_the_nearest_centre`, `s7_sample_changes_by_at_most_one_member_when_a_memory_is_added`, `s7_regions_are_numbered_largest_first_with_ranked_members`, `s7_input_hash_changes_with_any_id_or_parameter`. At 5,000 fake memories the sampled run gives 36 clusters, 1.5% unclustered and a largest share of 3.8%.
- **Touches.** SPEC 4, SPEC 7 steps 1 to 4 and "Labels", SPEC 11 AT-7, stream E (caching and invalidation).

## B-47 Map coordinates: a separate 2-dimension projection (2026-11-16)

- **Decision.** `regions` with `coords` runs a second `holomap` projection with the same parameters at 2 dimensions, over the same sample, with neighbours capped at the number projected minus 1. A memory outside the sample sits at the mean position of its 15 nearest sample memories by cosine in 768 dimensions. One memory sits at (0, 0); two sit at (0, 0) and (1 - cos, 0). `coordinates(memories, params, run)` returns `sm_core::protocol::MapPoint` with each point's region from `run`, and refuses a run made from other memories or parameters.
- **Caveat.** The 2-D map is not the 5-D projection that was clustered, so a region need not look contiguous on it.
- **Cost.** About 6.5 s at 2,000 memories on the reference machine, on top of about 6.7 s for `regions` itself; `holomap`'s dense spectral initialisation dominates both. SPEC 11 sets no budget for `regions`. Stream E computes coordinates only when asked and caches runs under `input_hash`. `manifolds-rs` 0.9.1 (SPEC 7, SPEC 13) was not evaluated.
- **Touches.** SPEC 7, SPEC 8 (`regions` with `coords`), SPEC 13.

## B-48 `journey` returns no repeats and refuses opposite endpoints (2026-11-16)

- **Decision.**
  - Step 0 is `from` and the last step is `to`. Each step between is the nearest memory that is neither endpoint nor already on the path; steps fill in order from `from`, so an earlier step wins a memory two steps would both pick. A step with no candidate left is empty (`memory: None`). The caller's exclusions are kept.
  - The path is slerp (spherical linear interpolation) in `f64` at `t = i / (steps - 1)`, falling back to normalised linear interpolation below 1e-6 rad, version 1's threshold.
  - Opposite endpoints (cosine below -1 + 1e-6) and the same memory at both ends are `invalid_input`.
  - The score is the index's `MemoryHit` score for intermediate steps and the cosine to the endpoint vector at the ends.
- **Deviation from SPEC 7.** SPEC 7 says `journey` "stays as in version 1". Version 1 searched each step on its own with no visited set, so one memory could fill several steps, and its slerp divided by `sin(omega)`, which is about 0 for opposite endpoints. The frozen contract, `sm_core::protocol::JourneyStep.memory`, reads "the nearest memory not already on the path, if any", and the code wins over prose (CONTRACTS.md section 1). Repeats also make a path that says nothing new at those steps. SPEC 7's "as in version 1" is read as the interpolation, not the repeats.
- **Not under AT-6.** `journey` uses the platform's `sin` and `acos`, so bit-for-bit equality across operating systems is not claimed; its tests compare with tolerances. AT-6 covers `regions` only.
- **Evidence.** `s7_journey_picks_the_nearest_memory_not_already_on_the_path`, `s7_journey_rejects_one_memory_twice_and_opposite_vectors`, `s7_journey_steps_are_unit_length_and_evenly_spaced`.
- **Touches.** SPEC 7 "Journey", SPEC 11 AT-6, CONTRACTS.md (`JourneyStep`).

## B-49 The AT-7 clustering fixture and its measure (2026-11-16)

- **Decision.**
  - `fixtures/spatial/texts.json` holds 320 developer-memory texts written for the test, in 10 topics of 32, with close topic pairs on purpose (postgres and sqlite, ci and testing, ci and kubernetes). The nearest neighbour of a text shares its topic for 79.7% of texts.
  - `fixtures/spatial/vectors.f32le` holds their 320 x 768 little-endian `f32` vectors from EmbeddingGemma 2 at the SPEC 5 revision, SHA-256 `d7845752b6dafaeb4f9976b3bf8238766bc18571618aa2a0421c8815b859ead8`. The committed vectors are the reference, because ONNX Runtime may differ in the last bits between machines. `fixtures/spatial/generator/` is a standalone Cargo project with its own lock file that reads the model in place from the per-user cache; it is a development tool, never shipped.
  - The adjusted Rand index (ARI) is computed in-crate in scikit-learn's pair-confusion form, with unclustered memories as one more label, as SPEC 7's measurement treated HDBSCAN's `-1`. It is reported, not gated; AT-7's gates are the unclustered and largest-cluster shares.
- **Result.** 9 clusters, 7.8% unclustered (gate 25%), largest cluster 17.8% (gate 40%), ARI 0.734; ci and git merged into one region. Equal on Windows and Linux.
- **Open.** A rerun of AT-7 with live embedding is a phase 3 step.
- **Evidence.** `at7_clustering_quality_on_the_fixture`, `at7_fixture_hash_matches_readme`, `at7_fixture_has_ten_topics_of_at_least_30`, `at7_ari_matches_scikit_learn`.
- **Touches.** SPEC 11 AT-7, SPEC 5 (model revision), fixtures.

## B-50 `fastembed` is pinned without its default features (2026-11-16)

- **Decision.** The root pin is `fastembed = { version = "=7.7.0", default-features = false, features = ["onnx", "hf-hub-rustls-tls", "ort-download-binaries-rustls-tls"] }`. This is a lead change to a frozen pin (CONTRACTS.md section 1), made at stream B's request (stream B report, section 8 item 1).
- **Why.** fastembed's default features are `onnx`, `ort-download-binaries-native-tls`, `hf-hub-native-tls` and `image-models`. They put a second TLS stack into the `arctic-embed-m` build, linking the system OpenSSL on Linux, where B-42 chose rustls. They also add an image stack the text model never uses. `sm-embed/src/arctic.rs` uses only `TextEmbedding`, which is under `onnx`. Both rustls features exist in fastembed 7.7.0's `[features]`.
- **Evidence.**
  - The lock loses 92 packages, among them `openssl`, `openssl-sys`, `native-tls`, `image`, `rav1e`, `libfuzzer-sys` (the NCSA flag) and `webpki-root-certs` (one of the two CDLA-Permissive-2.0 flags).
  - It gains 8: `quinn`, `quinn-proto`, `quinn-udp`, `lru-slab`, `rustc-hash`, `tinyvec`, `tinyvec_macros` and `cfg_aliases`, all MIT, Apache-2.0 or Zlib. None is in the resolved tree, `--all-features --target all` included; they are lock-only.
  - `cargo test`, `clippy -D warnings` and `fmt --check` pass on the merged tree with `--all-features`, which builds the feature.
- **Still open.** `option-ext` (MPL-2.0) stays in the feature build through `hf-hub` → `dirs` (B-41). Arctic was compiled and unit-tested but never loaded or run (B-57).
- **Touches.** Root `Cargo.toml`, `Cargo.lock`, LICENSES.md, D-5, B-41, B-42.

## B-51 `EmbedderProvider::open` gives at most the threads asked for (2026-11-16)

- **Decision.** The `sm-core` doc for `EmbedderProvider::open` and the CONTRACTS.md section 9 row now read "at most `threads` intra-op threads, never more than the physical cores, halved for bulk work when priority cannot be lowered". They used to say "exactly `threads`". This is a lead change to frozen `sm-core`, made at stream B's request (section 8 item 2). Only the documentation changed, not the message schema, so `PROTOCOL_VERSION` is unchanged.
- **Why.** SPEC 5 rules 1 and 2 cap a session at the physical cores, and SPEC 13 halves the bulk budget when priority cannot be lowered. "Exactly" contradicted both. `GemmaEmbedder::threads()` reports what was asked for and what was used.
- **Evidence.** `s5_threads_never_exceed_physical_cores`, `s13_bulk_budget_halves_when_priority_is_refused`.
- **Touches.** `crates/sm-core/src/traits.rs`, CONTRACTS.md section 9, SPEC 5, SPEC 13, stream E (it reads the thread report for AT-3).

## B-52 Model integrity: a full check at every load; download and error rules (2026-11-16)

- **Decision.**
  - `GemmaProvider::load` runs `model::verify` before anything else, so no caller can load an unchecked model. The check compares sizes first, so a truncated file fails without being read, then the SHA-256 of all three pinned files.
  - **Errors.** A missing or unreadable file is `model_missing`, with the next action `spatial-memory setup` or `SPATIAL_MEMORY_MODEL_DIR`. A wrong size, a wrong hash, a tokenizer whose `<pad>` is not id 0, or a model whose inputs and output are not the pinned ones is `model_integrity`; its next action is to delete the folder and run `setup` again. Every message names the file.
  - **Download.** Each file goes to `<name>.download-<pid>` in the target's own folder, so the rename never crosses a volume, and a drop guard deletes it on every failure path. A network or write error is `model_missing`. A short body, wrong length or wrong hash is `model_integrity`, and nothing is left behind. A file that is already present counts as done only after the same size and SHA-256 check. Connect and first-byte timeouts are set. The body has none, because the weights are about 1 GB.
  - **No verify stamp in 3.0.** Stream B proposed one for slow disks: size, modification time and file id per file, with a full re-hash on any change. It is not adopted, and the full check stays.
- **Why.** SPEC 9 says the files are "verified by SHA-256 before loading". The full check costs 0.51 s on the reference machine (about 2.1 GB/s), well inside the 5 s cold-start budget. A stamp is a second way to be wrong about integrity, and nothing yet shows it is needed.
- **Over budget in Docker.** Load took 8.95 s on Linux in Docker against the 5 s budget. The model was read over Docker Desktop's 9p share at about 230 MB/s: a plain `cat` of the 1,034 MB weights file took 4.65 s, while `sha256sum` of the same size from `/dev/zero` took 0.76 s. This figure measures the share, not the code. Native Linux is unmeasured; the cold-start budget on Linux is measured natively in phase 3, and the stamp is reconsidered only if that run misses the budget. Cold-cache Windows is also unmeasured.
- **Evidence.** The eight `at10_*` tests (TRACE AT-10); `s5_download_*`; `s5_pinned_files_match_the_spec_table`; `s5_pinned_url_names_repository_and_revision`. All download tests run against a local `TcpListener`. A live download from huggingface.co was not run.
- **Touches.** SPEC 5 "Model, pinned", SPEC 8, SPEC 9, SPEC 11 AT-10 and "ready from cold", B-42, B-43.

## B-53 ONNX Runtime session settings (2026-11-16)

- **Decision.**
  - **Optimisation level.** SPEC 5's "graph optimization level 3" is `GraphOptimizationLevel::Level3`, which ort 2.0.0-rc.15 maps to `ORT_ENABLE_LAYOUT`. `GraphOptimizationLevel::All` maps to `ORT_ENABLE_ALL`, which would also include any level ONNX Runtime adds later. ort's own comment still calls `Level3` "all optimizations", from before the mapping changed. The literal level 3 is kept, and AT-1, AT-2 and every budget were measured with it.
  - **Threads.** Each session has its own thread pool (`with_independent_thread_pool`) with one inter-op thread and sequential execution. Its intra-op threads follow the plan in B-54. Intra-op and inter-op spinning are on only when `embed.spinning` is on.
  - **One environment.** There is one ONNX Runtime environment per process, with telemetry off (SPEC 9 "Local only").
  - **Signature check.** At load, the session must take the five pinned inputs (`input_ids`, `attention_mask`, and empty `image_features`, `video_features` and `audio_features` of width 512) and give `sentence_embedding`.
- **Touches.** SPEC 5 table and "Using the machine's cores" rules 1, 2 and 5; SPEC 9.

## B-54 Bulk threads below normal priority, counted (2026-11-16)

- **Decision.**
  - **Bulk runner.** A bulk session owns a runner thread. The runner lowers itself, then builds the session, so every pool thread ONNX Runtime starts goes through the `CountingThreads` thread-manager hook, which lowers and counts it. The runner runs every batch, and it counts as one of the n intra-op threads.
  - **Query sessions** run inference on the caller's thread at normal priority.
  - **Priority values are absolute targets.** Windows uses `THREAD_PRIORITY_BELOW_NORMAL`. On Linux a thread's nice value is set to the process's own plus 10, capped at 19, and a thread already at or past that is left alone. Linux threads inherit nice from the thread that starts them, so a relative step would take pool threads to 19; the first Linux run did exactly that.
  - **Probe.** Once per process, a short-lived thread is lowered and read back. If that fails, every bulk session takes the SPEC 13 fallback: half the budget, at least 1, not lowered, with the reason in `ThreadPlan`.
- **Evidence.** With the defaults (12 asked), at most 12 threads ran and all 12 were lowered, with none refused. Every thread read back -1 on Windows and nice 10 on Linux. Tests: `d10_*` (six), `s5_query_sessions_run_at_normal_priority`, `s5_thread_report_counts_the_calling_thread`, `s5_threads_never_exceed_physical_cores`, `s13_bulk_budget_halves_when_priority_is_refused`.
- **Touches.** SPEC 5 rule 4, SPEC 13, D-10, AT-3 (stream E reads `ThreadReport`).

## B-55 Embedder inputs and outputs; chunker details (2026-11-16)

- **Decision.**
  - **Batches.** Inputs are sorted shortest first into batches of `embed.batch_size`, and results come back in input order. Padding is token 0 on the right, and the tokenizer's `<pad>` id is checked to be 0 at load.
  - **Outputs.** Every vector's norm must be within 1e-3 of 1, and the batch shape must match. Otherwise the result is `internal`, never a silently wrong vector.
  - **Over-limit inputs.** A chunk over the limit passed to `embed_documents` is `invalid_input`; the chunker never produces one. If the prompt and title still push an input past the cap, the content is cut, keeping the end token, and counted in `truncated_inputs()`, which was 0 in AT-4. A query is cut to the smaller of `max_tokens` and the chunk limit before the prompt is added.
  - **Chunker.**
    - Only ATX headings (`#` to `######`) split, and pieces are never packed across one.
    - Sentences also end at a line break.
    - A fenced code block (three or more backticks or tildes; an unclosed fence runs to the end) is one piece. If it alone is over the limit, it becomes overlapping token windows, as SPEC 5 rule 4 requires. Windows are never packed with neighbours.
    - `overlap` must be in [0, 1) and `max_tokens` at least 1.
    - `indexed_tokens` never exceeds the cap.
    - Every chunk is a trimmed, contiguous part of the content, and its token count is measured on its own text.
- **Evidence.** `s5_rule1_*` to `s5_rule6_*`, `s5_batches_*`, `s5_padding_is_on_the_right_with_zero`, `s5_output_must_be_unit_length_and_shaped`, `s5_truncate_content_keeps_the_end_token`, `s6_query_is_cut_before_the_prompt_is_added`.
- **Touches.** SPEC 5 table and chunking rules 1 to 6, SPEC 6 step 1, AT-4.

## B-56 `doctor` numbers: a fixed sample (2026-11-16)

- **Decision.** The sample is 62 engineering words drawn by a 64-bit linear congruential generator, with a sentence break every 17 words. It is chunked at the configured limit, and the first 16 chunks are timed after one warm-up batch. The query figure is the median of 5 runs at 128 tokens, after one warm-up. Nothing is tuned from the numbers. `sm-embed` provides `doctor_numbers`; the command-line `doctor` wires it in phase 3 (B-43).
- **Evidence.** `s5_rule7_doctor_reports_both_numbers_at_the_budget`, `s5_rule7_sample_is_fixed`, `s11_doctor_numbers`. Windows: 17.26 chunks/s and 47.2 ms. Linux in Docker: 12.10 chunks/s and 79.8 ms.
- **Touches.** SPEC 5 rule 7, SPEC 11, B-43.

## B-57 Arctic Embed M fallback (feature `arctic-embed-m`) (2026-11-16)

- **Decision.**
  - fastembed's cache is `sm_core::paths::model_dir` for the Arctic model id, with the revision `fastembed-7.7.0`, so it sits under the per-user model cache. It ignores `SPATIAL_MEMORY_MODEL_DIR`, which names the EmbeddingGemma folder.
  - The query prefix comes from the model card. There is no document prompt, so the title is unused. The chunk limit must leave 32 of the 512 tokens for the prompt.
  - A load failure is `model_missing`, with a network next action.
  - fastembed has no thread-creation hook, so a bulk session always takes the SPEC 13 fallback (half the budget) and reports it.
- **Not verified.** Arctic is compiled, passes clippy and passes its three unit tests. It was never downloaded, loaded or run. Phase 3 does not need it for any AT; if it is not run there, RESULTS lists it as unverified.
- **Evidence.** `d5_query_prompt_and_bulk_fallback`, `d5_chunk_limit_leaves_room_for_the_prompt`, `d5_fallback_cache_is_under_the_model_cache_not_the_override`.
- **Touches.** D-5, B-23, B-41, B-50.

## B-58 The real-model test binary and its operating-system readings (2026-11-16)

- **Decision.**
  - `crates/sm-embed/tests/real_model.rs` holds one test, `s11_real_model_cases_in_sequence`, which loads the model once and runs every case in order: AT-1, AT-2's embedding part, AT-4's embedding part, AT-10 on the installed files, D-10 and the SPEC 11 figures. Without the model folder it prints a skip line and passes, as the build rules require.
  - Memory budgets are asserted. Time budgets are printed but not asserted, so a slow machine is reported rather than tuned around.
  - AT-4 runs with both the query and the bulk session alive, as the daemon would.
  - **Readings.** Physical cores come from `num_cpus::get_physical()`. On Windows, memory comes from `GetProcessMemoryInfo`, and threads and their priorities from a Toolhelp snapshot and `GetThreadPriority`. On Linux they come from `/proc/self/status` (`VmHWM`, `VmRSS`), `/proc/self/task`, and field 19 of each task's `stat`.
- **Evidence.** On Windows: 1 passed in 10.2 s with the model present, both in the worktree and after the merge. On Linux in Docker: skipped without the model, and passed with it mounted (stream B report, section 3).
- **Touches.** SPEC 11 "test conventions", AT-1, AT-2, AT-4, AT-10, D-10.

## B-59 Capture signals: version 1's patterns, two active tiers, clamped importance (2026-11-16)

- **Decision.** (Stream G, G-1 to G-3.)
  - **Patterns.** Version 1's 22 signal patterns are ported as regular expressions, compiled once per process. Python's `$` also matches before a final newline, so `(?:\.|$)` is written `(?:\.|\n?$)` in the 15 patterns that use it.
  - **Tiers.** A score of 0.8 or more is strong, 0.5 to 0.79 is medium, and below 0.5 is weak and gives no candidate. There is no tier 0.
  - **Importance.** Strong is `max(0.7, score)` and medium is `max(0.4, 0.8 × score)`, both clamped to 1.0. A score that is not finite gives no candidate. Version 1 left importance unclamped, treated tier 0 like tier 2 and turned NaN into 0.0.
  - **Type.** The memory type comes from the pattern family through version 1's `derive_namespace`: decisions → `decision`, troubleshooting → `pitfall`, patterns → `convention`, anything else → `note`.
- **Not matched.** Python's `\s` includes U+001C to U+001F, and the `regex` crate's `\s` (Unicode White_Space) does not. A pattern that relies on `\s` therefore does not match across those four control characters. No fixture covers it. Stripping is unaffected, because every strip goes through `sm_core::hash::python_strip`. Accepted for 3.0: these characters do not occur in normal tool output or prose.
- **Evidence.** `s16_signal_patterns_table`, `s16_signal_cases`, `s16_types_come_from_pattern_families`, `at16_medium_and_strong_signals_give_active_candidates`.
- **Touches.** SPEC 16 steps 2 and 5 and "Type and provenance", SPEC 4, AT-16.

## B-60 Which tool calls are captured, and the gate order (2026-11-16)

- **Decision.** (Stream G, G-4 to G-7.)
  - **Own tools are never captured.** In Claude Code, `mcp__spatial-memory__<tool>` and the plugin form `mcp__plugin_<plugin>_spatial-memory__<tool>` are recognised for any `<tool>`. Cursor's `MCP:<name>` does not name the server, so it is recognised when `<name>` is one of version 3's 13 tool names or version 1's 23. Another server's tool with one of those names is skipped too, which is the conservative side.
  - **The default `capture.tools` has no MCP tools** (B-21). MCP extraction is ported but only reached when the user adds a tool. The default was not changed to make a fixture pass: `at13_hook_payloads_default_settings` checks that the default gives `ToolNotCaptured` for the version 1 MCP payloads.
  - **Ignore patterns** are checked on both the raw path and the path after `normalize_file_anchor`. Either match skips the call.
  - **Gate order.**
    1. `capture.enabled`, then the project opt-out, then the event.
    2. For PostToolUse: the tool name, `captures_tool`, the skip rules, the ignore check, then extraction.
    3. Then, for each text: empty, classify, importance, redact (with the skip rule), session dedupe, record.
- **Evidence.** `s16_gate_order`, `s16_own_tools_are_never_captured`, `s16_a_tool_not_in_capture_tools_gives_nothing`, `s16_each_default_ignore_pattern_skips_edit_and_write`, `s16_ignore_patterns_fold_case_windows` and `_unix`, `s16_queue_and_pipeline_notes`.
- **Touches.** SPEC 15 `[capture]`, SPEC 16 step 1, B-21.

## B-61 Stop prefers the final message and falls back to the transcript (2026-11-16)

- **Decision.** (Stream G, G-8 and its deviation D-1.)
  - The loop guard follows Python truthiness, so `stop_hook_active: "false"` counts as set and Stop does nothing. When the guard is unclear, skip.
  - When the payload has `last_assistant_message`, Stop captures that text and does not read the transcript.
  - When it does not, Stop reads the transcript since the session's offset, as version 1 did. The default trigger is `session_end`.
- **Why this departs from SPEC 16.** The SPEC 16 event table says Stop "does not read the transcript". Every Stop payload in the fixtures comes from version 1's tests and lacks `last_assistant_message`, and a Stop that drops the turn would lose the most likely place a decision is stated. The stream G specification asked for the fallback. SPEC 16 should say: "Stop sends `last_assistant_message`; when it is absent, the daemon reads the transcript since the offset."
- **Evidence.** `at14_stop_prefers_last_assistant_message` (no transcript read, `new_offset` is `None`), `at14_stop_reads_the_transcript_when_there_is_no_final_message`, `s16_stop_loop_guard_and_missing_transcript`.
- **Touches.** SPEC 16 event table, AT-14.

## B-62 Session dedupe compares redacted text and keeps 4,096 hashes (2026-11-16)

- **Decision.** (Stream G, G-9 and G-10.)
  - `SessionState` holds the content hash (B-14) of each candidate the session produced. A new text is hashed after redaction, so both sides of the comparison are the text a memory stores. Version 1 compared raw transcript text against redacted queue files, so a text holding a secret never matched its own capture.
  - Blank text is never a duplicate.
  - PostToolUse dedupes as well as PreCompact and Stop.
  - The state keeps at most 4,096 hashes (`SESSION_HASH_CAP`) and drops the oldest first, which bounds it to about 600 KB a session. A text that falls out of the window can be captured again by the same session; the store's content-hash dedupe (B-29) then catches it.
  - The daemon owns one `SessionState` and one transcript offset per session id, and stores `CaptureOutcome.new_offset` whenever it is `Some`.
- **Evidence.** `at14_pre_compact_returns_only_what_post_tool_use_missed`, `at14_overlap_cases`, `at14_session_state_bound_is_fifo`.
- **Touches.** SPEC 16 step 4, AT-14, B-14, B-29.

## B-63 Capture provenance, tags and anchors (2026-11-16)

- **Decision.** (Stream G, G-11 to G-13.)
  - **Anchors.** Edit and Write get a `file` anchor. NotebookEdit records the notebook in `provenance.file_path` only, because SPEC 16 anchors Edit and Write and a cell edit is not a file revision that `related` can follow.
  - **Provenance.**
    - `source` is `capture`; `session_id` comes from the payload; `tool` is set for PostToolUse only.
    - `file_path` goes through `normalize_file_anchor`, so it is relative to the project root when inside it.
    - `commit_sha` is whatever the daemon passes in: `sm-capture` reads no repository.
    - `extra` holds `hook`, `tier`, `score`, `patterns`, `v1_namespace`, and, when they apply, `redactions`, `truncated` and `trigger`.
  - **Tags** come from version 1's `derive_tags` and `derive_transcript_tags`. A tag that fails `normalize_tags`, or an anchor that fails B-19, is left out and listed in the candidate's `dropped_tags` or `dropped_anchor`, never silently lost.
- **Evidence.** `s16_edit_and_write_are_anchored_to_their_file`, `s16_provenance_records_capture_with_session_tool_file_and_commit`, `s16_to_new_memory_leaves_the_store_fields_empty`.
- **Touches.** SPEC 4, SPEC 16 "Type and provenance", B-19, CONTRACTS 7.

## B-64 The secret scanner: version 1's 22 patterns, stable markers, version 1's skip rule (2026-11-16)

- **Decision.** (Stream G, G-14.)
  - `RegexSecretScanner` implements `sm_core::traits::SecretScanner`. It has 14 tier 1 key patterns, 5 tier 2 structural patterns (PEM and PGP private keys, JWT, `Authorization` headers, passwords in URLs) and 3 tier 3 value patterns (`password`, `secret`, `token`) behind version 1's false-positive filter.
  - Each match is replaced with `[REDACTED:<kind>]` through `redact_matches`. An existing `[REDACTED:...]` or version 1 `[REDACTED_...]` marker counts as already redacted, so redacting twice changes nothing (AT-9).
  - Only the secret part is replaced: the password in a URL, and the value (not its closing quote) in a tier 3 match.
  - **Skip rule.** A text is dropped as `Secret` when it holds a private key or when more than half of its characters were redacted (version 1's `should_skip`).
  - Stream F uses the same scanner for `share` (SPEC 9).
- **Evidence.** `at9_redaction_cases` (43 of 43), `at9_redaction_patterns_table` (26 of 26).
- **Touches.** SPEC 9 "Scope, team sharing and security", AT-9, CONTRACTS 9.

## B-65 Transcript reading, the safety cap and tool responses (2026-11-16)

- **Decision.** (Stream G, G-15 to G-19.)
  - **Reader.** `read_transcript(path, offset)` is stateless and returns the entries and the new offset. CRLF and LF lines are read. A line over 1 MiB, a blank line, a line that is not UTF-8, a line without the `"assistant"` marker, a non-object and a sidechain entry are skipped. An offset past the end of the file restarts from 0, and a re-read after a daemon restart is caught by the store's dedupe (B-29).
  - **Path check.** The transcript path comes from a hook payload, so it is checked before it is opened: not empty, absolute, no `..`, a regular file (links followed), and not a Windows device name (`CON`, `NUL`, `COM1` and so on, with or without an extension) on any platform. Session ids with a trailing newline are rejected.
  - **One cap.** A single 200,000-character cap applies at extraction, with version 2's `...` marker: to the PostToolUse text, to `last_assistant_message`, and to the transcript texts together (the text that crosses the cap is cut and ends the list). Master's per-tool, per-entry and combined caps are gone; version 2 wins where the SPEC is silent.
  - **Two caps, not one.** This 200,000-character cap is SPEC 16's safety cap at extraction; the text after it is stored whole. The index cap (`capture.max_index_tokens`, 8,192 tokens) is a different cap, applied by the chunker (`s5_index_cap_stops_chunking_and_reports_tokens`, stream B).
  - **Metrics.** SPEC 6 defines `capture_size_tokens` (size before and after the cap) and `capture_cap_hit_total` (hits by tool) on the index cap. Stream E records them from the chunker's report when it stores a capture. `CaptureOutcome.cap_hit` and `extracted_chars` describe the safety cap. The candidate's `provenance.extra.truncated` records that cap; stream E may also log it, but must not count it as `capture_cap_hit_total`.
  - **Non-string tool responses** are written as JSON with sorted keys and Python's separators, because `serde_json` keeps no insertion order without `preserve_order`, which is not turned on.
  - **Routing.** The event comes from the request (`CaptureParams.event`), never from the payload's `hook_event_name`.
- **Evidence.** `s16_transcript_cases`, `s16_hook_helpers_cases_windows` and `_unix`, `s16_content_extractor_cases`, `at17_long_capture_is_cut_at_extraction_with_the_marker`, `s16_pipeline_time_on_a_1mb_transcript_delta`.
- **Touches.** SPEC 9, SPEC 16 step 1 and "Index cap", AT-14, AT-17, B-29.

## B-66 The spool belongs to `sm-core` and the daemon, not `sm-capture` (2026-11-16)

- **Decision.** `sm-capture` does not touch the spool. The spool format is `sm_core::spool` (CONTRACTS 8) and the spool drain is the daemon's (CONTRACTS 7, stream E).
- **Why.** The SPEC 3 crate table ("spool format") and the SPEC 14 phase 1 table ("sm-capture: … spool") place it in `sm-capture`. CONTRACTS, which follows the code, already placed it in `sm-core` in phase 0. The SPEC rows are out of date.
- **Touches.** SPEC 3, SPEC 14, CONTRACTS 7 and 8.

## B-67 Capture fixtures: what differs from Python, and one misleading case (2026-11-16)

- **Decision.** (Stream G, G-20 to G-23.)
  - **Results.** 726 of 726 ported fixture cases pass. 278 are not applicable to `sm-capture`, and each prints its reason. About 90 are version 1's queue files and their rate limit (replaced by the spool, B-25 and B-38) and 22 its hook error log (replaced by the journal). Most of the rest belong to another stream: reading stdin, writing stdout, the working-directory check, the project root, Cursor normalisation and ambient recall are stream D's, and the per-session transcript offset is stream E's. Streams D and E must cover those cases from the same fixture files; it is listed in their specifications. 34 cases satisfy their `expect` but differ from Python's `computed_*` value. Each difference follows from B-59 to B-65 and is listed in `docs/v3/streams/g-report.md` (G-22).
  - **Generated input.** `fixtures/capture/rust/cursor_normalized_payloads.json` was produced by running version 2's own `_normalize_stdin`, because `normalize_stdin_cases` covers only small field sets. `at14_session.jsonl` is a hand-written transcript for AT-14.
  - **Checker operator.** The fixture checker supports `one_of` (the value at `path` equals one of the listed values), which `cc_post_tool_use_edit_fix_round_trip.json` uses. The lead added it to the operator list in `fixtures/capture/README.md` at the merge; no case changed.
  - **Misleading case, kept as it is.** `hook_payloads/cc_post_tool_use_bash_private_key_skipped.json` (from version 2's `test_private_key_causes_skip`) has no signal, so it is dropped as weak before the secret rule runs and does not test what it is named for. The fixture is data and is not edited. `at13_hook_payloads_default_settings` adds a strong variant ("The fix is to rotate this key."), which is dropped as `Secret`.
- **Touches.** SPEC 16, AT-9, AT-13, B-13.

## B-68 Capture budgets: Windows release, Linux debug (2026-11-16)

- **Decision.**
  - The two classifier budgets in `signal_cases.json` are asserted in every build by `s16_signal_cases`: under 1 ms per `classify` call on average, and under 100 ms for the definition pattern on a ReDoS (regular-expression denial of service) input.
  - SPEC 11 sets no budget for the whole pipeline, which the daemon runs after it has replied `accepted`. The two pipeline tests check the outcome and print the time.
  - Windows release figures: classify 0.0007 ms; the ReDoS input 0.147 ms; the pipeline on a 200,000-character Bash output 0.600 ms; a 1 MB transcript delta 4.564 ms. In a debug build on Windows at the merge: 0.0158 ms, 33.2 ms, 64.4 ms and 103.8 ms.
  - Linux was measured in a debug build only, in Docker. Both classifier budgets passed with a wide margin: 0.0093 ms against 1 ms, and 16.9 ms against 100 ms.
- **Open.** Native Linux release figures are measured in phase 3 and go in RESULTS. No end-to-end capture run exists until streams D and E are merged.
- **Evidence.** `s16_signal_cases`, `s16_pipeline_time_on_a_200k_bash_output`, `s16_pipeline_time_on_a_1mb_transcript_delta`.
- **Touches.** SPEC 11 budgets, SPEC 16.

## B-69 The store: connections, transactions and two copies of each vector (2026-11-16)

- **Decision.** (Stream A, report decisions 1, 2, 4 and 8.)
  - **Connections.** A `Store` holds one writer connection behind a mutex and a pool of readers (up to 4 kept idle). Every write is one `BEGIN IMMEDIATE` transaction; every read is one deferred transaction, so a search checks the index stamp and reads its vectors from one snapshot. Every connection uses WAL mode, a 30-second busy timeout, `synchronous=FULL` and `foreign_keys=ON`, and checks that vec1 is registered.
  - **Errors.** A lock still held after the timeout becomes `timeout`, with a next action. Any other SQLite error becomes `storage`. No raw lock error reaches a caller.
  - **Two copies of each vector.** `chunks.embedding` holds 768 little-endian `f32` (`CHECK (length = 3072)`), and the vec1 table `chunk_vectors` (flat index, cosine) holds the same vector. Triggers on `chunks` and `memories` (`chunks_ai`, `chunks_ad`, `chunks_au_embedding`, `chunks_au_text`, `memories_au_title`, `memories_ad`) keep both copies and the keyword index in step, so no code path can write one and forget the other.
  - **Unit length.** Every stored or query vector must be finite, with squared length within 10⁻³ of 1, or it is `invalid_input`.
  - **Chunks.** A stored memory has at least one chunk, numbered 0, 1, 2 … with no gaps.
  - **vec1.** `build.rs` compiles `vendor/vec1` unchanged (B-7) as a scalar object and an AVX2 object at `-O3`. `vec1.rs` holds the crate's only `unsafe` code.
- **Why.** `BEGIN IMMEDIATE` takes the write lock at the start, so two writers never deadlock upgrading a read lock. In WAL mode readers do not wait for the writer. Project scope reads `chunks` through its `project_id` index without touching vec1; vec1 serves scope `all`. Dot product equals cosine only for unit vectors, and the scope-`all` proof margin (B-70) depends on the bound.
- **Evidence.** `at5_two_readers_and_a_writer_for_sixty_seconds` (Windows: writer 24,978 operations, readers 59,768 and 60,248; Linux in Docker: 10,923, 81,640 and 81,608; no error of any kind), `s4_vectors_are_768_little_endian_f32`, `s4_vectors_must_be_unit_length_and_finite`, `s4_forget_cascades_to_every_dependent_table`, `s4_invalid_items_fail_the_whole_batch`.
- **Touches.** SPEC 3, SPEC 4, AT-5, B-7.

## B-70 Exact search in both scopes, and the keyword index (2026-11-16)

- **Decision.** (Stream A, report decisions 3 and 5, deviation 2.)
  - **Project scope** reads `chunks WHERE project_id = ?` through its index and takes the dot product in Rust.
  - **Scope `all`** asks vec1 for the nearest k = max(8 × (limit + excluded ids), 256) chunks, then re-scores, filters and collapses them in Rust. The answer stands only when it is provably exact: vec1 returned fewer than k chunks, or the `limit`-th memory's score exceeds (1 − the k-th distance) + 2.01 × 10⁻³. Otherwise every embedded chunk is scored in Rust. An anchor filter skips vec1 and scores the anchor's memories directly.
  - **Scores and ranking.** Scores always come from the Rust dot product, never from vec1's distance, so both scopes give identical scores for the same chunk. A memory's best chunk wins (the lower `seq` on a tie). Results are ordered by score, then id. Importance is not used in the store's ranking; SPEC 6's ranking is the daemon's (stream E).
  - **Keyword index.** A contentless FTS5 table (`content=''`, `contentless_delete=1`, tokenizer `porter unicode61`) with columns `title` and `text`, one row per chunk, its rowid the chunk id. The query is always one quoted phrase with internal `"` doubled, so query syntax is searched as text. Each memory appears once, with its best-ranked chunk, by bm25 then id. Pending chunks are keyword-searchable at once (B-27).
  - **Known limit.** A phrase that spans a chunk boundary is not found.
- **Why.** The stream file asked for vec1 in exact mode then the collapse. Status, type and exclusion filters can remove vec1's top chunks, and vec1's `f32` cosine can differ from the dot product in the last bits; the proof margin and the fallback keep scope `all` equal to brute force whatever the filters remove, which is what D-2 requires. vec1's approximate index is never built. A contentless table does not store the text twice, and `contentless_delete=1` lets forget and revise delete rows.
- **Evidence.** `d2_exact_search_matches_brute_force_in_both_scopes` (at least 1,000 chunks of exactly representable unit vectors; top 10 identical with filters and exclusions in both scopes), `d2_query_vectors_are_checked`, `d2_pending_chunks_are_not_vector_searched`, `d2_exact_vectors_are_unit_length_in_f32`, and the six `d3_` tests.
- **Touches.** SPEC 4, SPEC 6 step 6, D-2, D-3, AT-16.

## B-71 Backups by `VACUUM INTO`; migrations; the index stamp and reindex (2026-11-16)

- **Decision.** (Stream A, report decisions 6 and 17, deviation 1.)
  - **Backups.** A migration or reindex backs up an existing database first, with `VACUUM INTO`, as `memory-<UTC time>-<reason>.db` in `backups_dir()`, never overwriting (a `-1`, `-2` suffix is added). A new database is created without a backup.
  - **Migrations.** The schema version lives in `meta` (`SCHEMA_VERSION = 1`). Each migration runs forward in its own `BEGIN IMMEDIATE` transaction. A newer, unknown version is refused.
  - **Stamp.** `configure_index` records the stamp on a store that has none. When the stored stamp differs, or none is configured, every vector search, keyword search and `memory_vectors` call returns `index_mismatch`.
  - **Reindex.** Writes a backup, then re-chunks memories in pages of 256 by id, one transaction per page, with the caller's chunker outside any transaction. Rebuilt chunks are stored pending and `set_embeddings` fills them later. A memory forgotten or revised in between is skipped. The new stamp is written last, so an interrupted reindex leaves searches blocked rather than wrong. `set_embeddings` refuses another model or a mismatched index and updates only chunks still pending.
- **Why.** SPEC 4 "Housekeeping" says to copy the database file. In WAL mode a file copy taken while another process writes can miss committed pages still in the `-wal` file; `VACUUM INTO` writes a complete, consistent database from one read transaction. The result is what the SPEC intends: one restorable file.
- **Open.** Nothing prunes old backups; SPEC 4 sets no retention. Recorded in PROGRESS.
- **Evidence.** `at11_stamp_mismatch_blocks_search_until_reindex`, `at11_unconfigured_store_refuses_to_search`, `s4_first_configuration_writes_the_stamp`, `s4_invalid_stamps_are_refused`, `s4_new_database_is_created_without_a_backup`, `s4_migration_writes_a_backup_first`, `s4_failed_migration_leaves_the_previous_version`, `s4_unreadable_schema_versions_are_refused`.
- **Touches.** SPEC 4 "Model stamp" and "Housekeeping", AT-11, B-27.

## B-72 Shared memories are read-only locally (2026-11-16)

- **Decision.** (Stream A, report decisions 10 and 11, deviation 3.)
  - Every write takes an `Access`: `Local` for the daemon and the tools, `Sync` for `sm-share`, the only writer that may change shared memories.
  - Under `Local`, revise, merge, forget by id and supersede refuse a shared memory with `shared_read_only` and write nothing.
  - Forget by filter skips shared memories and reports them in `ForgetCount.shared_skipped`. `count_forget` gives the count the daemon shows before asking for `confirm`; `forget_by_filter` counts again and deletes in one transaction, and fails if the two counts disagree.
  - Forget by id is one transaction, all or nothing, and removes provenance, chunks, keyword rows, vectors, anchor mappings, links, map points and injections.
- **Why.** The stream file said to refuse forgetting any shared memory. Forget by id names a memory, so refusing is the honest answer. A filter only describes memories; failing a whole bulk forget because one shared memory matches would make bulk forget unusable in a project that syncs.
- **Evidence.** `s4_shared_memories_are_read_only_locally`, `s4_forget_by_filter_counts_what_it_deletes`, `s4_forget_cascades_to_every_dependent_table`.
- **Touches.** SPEC 4 "origin", SPEC 9, B-30; streams E (passes `Local`) and F (passes `Sync`).

## B-73 `supersedes_id` keeps SPEC 4's one meaning; a merge records `merged_into` (amends B-31) (2026-11-16)

- **Decision.**
  - `supersedes_id` always means "the memory this one replaces" (SPEC 4, shared-file field `supersedes`). Only insert writes it.
  - Applying a consolidation group `{keep, merge}` marks each merged memory `superseded`, leaves its `supersedes_id` as it was, and records `merged_into = <keep id>` in its `provenance.extra`, in the same transaction. An `extra` that is not a JSON object is kept under `extra.value`.
  - "What replaced memory X" is the memories whose `supersedes_id` is X, plus X's `provenance.extra.merged_into` when X was merged.
- **Why.** B-31 pointed the merged memory's `supersedes_id` at the kept memory, the opposite direction to SPEC 4, so one column meant "replaces" or "replaced by" depending on the path that wrote it (stream A report, section 8). A merge also overwrote the merged memory's own record of what it had replaced. And sync (stream F) applies SPEC 4's rule "setting `supersedes_id` marks the older memory superseded" to shared files, so a merged memory exported with `supersedes: <keep>` would mark the kept memory superseded on every other machine.
- **Lead change at the merge.** `merge_memories` and `s4_merge_supersedes_into_the_kept_memory` were changed by the lead; the test now also checks a merged memory that had replaced another, an existing `extra` object, a non-object `extra` and that a failed group writes no provenance.
- **Touches.** SPEC 4 "Types and status", SPEC 8 `consolidate`, B-31; stream E's `consolidate` reply and stream F's shared files (a merged memory is exported as `status: superseded` with its own `supersedes`, if any).

## B-74 Bulk inserts into vec1 slow as the table grows; accepted for 3.0 (2026-11-16)

- **Measured.** (Stream A report, section 9.1; release build, Windows.) Each insert into the vec1 table `chunk_vectors` costs about 0.33 ms more for every 1,000 chunks already in it: 0.66 ms per chunk at 0–5k, 8.91 ms at 25–30k, 140.9 s for 30,000 chunks. Without the vec1 insert the store costs a flat 0.04–0.05 ms per chunk (1.5 s). A single one-chunk `remember` measured a median 4.46 ms at 10,020 chunks and 7.83 ms at 20,040. The FTS insert costs nothing measurable. The cause is inside vec1's insert path and was not located.
- **Decision.**
  - `vendor/vec1` stays unchanged (B-7) and the triggers stay, so the two vector copies cannot drift (B-69).
  - Stream E keeps each `set_embeddings` transaction to one embedding batch, so the writer lock is held for one batch at a time. The hook reply never waits for the writer: injections are recorded off the reply path. AT-3 is measured while an import runs.
  - Phase 3 measures `import` and `migrate` against the store size and reports them in RESULTS.
  - The load-without-index-then-rebuild path (SPIKE b: 100,000 vectors in 0.60 s, rebuilt to flat cosine in 2.22 s) is deferred to 3.1; it changes how `chunk_vectors` is maintained.
- **Why.** SPEC 11 sets no bulk-insert budget. `remember` (at most 2 s) is not at risk: the store's part extrapolates to about 35 ms at 100,000 chunks (extrapolated, not measured). One bulk embedding session runs at 13.2 chunks per second on the default 12 threads (SPEC 5, measured 16 November), about 76 ms per chunk, against at most about 33 ms per chunk of vec1 insert at 100,000 chunks, so the bulk queue stays embedding-bound.
- **Touches.** SPEC 4, SPEC 5 bulk queue, SPEC 11, AT-3, B-7.

## B-75 Project scope in one 100,000-chunk project: the store alone is 243.7 ms at p95 (a risk) (2026-11-16)

- **Measured.** (`cargo run --release -p sm-store --example search_budget`; Windows, Ryzen 9 9950X3D, 100,000 chunks of 768 dimensions, top 5, 300 queries.)

  | Store | Scope | p50 | p95 | max |
  | --- | --- | --- | --- | --- |
  | Spread, 60 projects | Project | 8.4 ms | 9.9 ms | 11.5 ms |
  | Spread, 60 projects | All | 147.6 ms | 152.9 ms | 205.5 ms |
  | Single project | Project | 235.2 ms | 243.7 ms | 247.8 ms |
  | Single project | All | 156.1 ms | 236.4 ms | 310.2 ms |

- **Decision.** No change in 3.0's store now. SPEC 4's figure (3–5 ms at 100,000 chunks) matches the spread store, where the project holds 1,667 chunks. A project holding most of a large store puts `recall` at an estimated 290–340 ms end to end once the query embedding (45–95 ms, SPEC 11) is added, over the 250 ms budget. Phase 3 measures `recall` end to end on Windows and native Linux. If it is over budget, try `PRAGMA mmap_size` on reader connections first, then the vec1 path restricted to the project with the B-70 proof margin, and keep a change only if it is measured to help.
- **Open.** The budgets were measured on Windows only; Linux ran the tests, not the example.
- **Touches.** SPEC 4 "Project-scoped search", SPEC 11 `recall` budget.

## B-76 Store behaviour the daemon and `sm-share` rely on (2026-11-16)

- **Decision.** (Stream A, report decisions 7, 12 to 16 and 18, and section 8.)
  - **Dedupe.** An input is a duplicate when its id exists, or a memory in the same project has the same `content_hash` whatever its status; the earliest by `created_at`, then id, is returned with outcome `Duplicate`. Later items in a batch see earlier ones. Any invalid item fails the whole batch (B-29, B-14).
  - **Links.** An identical link (same ends and relation) returns the stored one with `created` false. A memory end must exist (`not_found`) and be in the link's project (`invalid_input`); the two ends must differ. Links go when a memory end is forgotten or the project deleted. `related_to_memory` and `related_to_anchor` walk two hops in either direction over `links` and shared anchors; an edge through a shared anchor is a synthetic link with extractor `memory_anchors` (`ANCHOR_EDGE_EXTRACTOR`).
  - **Staleness (B-37).** The store does not walk git ancestry. `links_of` takes an oracle `Fn(commit) -> Option<bool>`; the shipped `head_only(head)` settles only a commit equal to the head, or a hexadecimal prefix of it of at least 7 characters, ignoring case. Anything else leaves `stale` absent.
  - **Map cache (B-32).** Runs are keyed by a unique `input_hash`; `put_map_run` returns the existing run for a known hash. Labels are stored against (run, region). `prune_map_runs` keeps the newest N runs.
  - **Memory vectors.** `memory_vectors` and `memory_vector` return the mean of a memory's embedded chunk vectors, re-normalised; if the chunks cancel out, the first chunk stands for the memory. `sm_spatial::memory_vector` instead returns an error on a zero sum. Stream E uses the store's functions for every memory vector, so the map and the search agree.
  - **Injections.** One row per (session, memory), keeping the latest time; rows older than 30 days are pruned (`INJECTION_RETENTION_DAYS`).
  - **Iteration.** `iter_memories` streams memories in id order, in pages read in their own transactions, for `reindex`, `export` and `migrate`.
  - **Types.** `MemoryInput`, `NewChunk`, `Revision`, `ContentChange`, `MemoryListFilter`, `NewMapRun`, `Inserted`, `ForgetCount`, `LinkAdded`, `WalkHit` and `PendingChunk` are `sm-store` types. CONTRACTS section 2 allows this (there is no store trait); no contract change.
- **Evidence.** `s4_duplicates_return_the_existing_memory`, `s4_links_are_added_once_and_removed`, `s4_link_ends_are_checked`, `s4_related_walks_links_and_anchors_two_hops`, `s4_head_only_settles_only_the_head`, `s4_links_of_reports_staleness_from_the_oracle`, `s4_map_cache_round_trip`, `s4_map_runs_are_validated`, `s4_prune_keeps_the_newest_runs`, `s7_memory_vectors_are_normalized_chunk_means`, `s4_injections_record_and_keep_the_latest_time`, `s4_injections_older_than_30_days_are_pruned`, `s4_iter_memories_streams_every_memory`.
- **Touches.** SPEC 4, SPEC 6 "Related" and "No repeats", SPEC 7 steps 1 and 4, D-8, B-14, B-29, B-32, B-37, CONTRACTS 2.

## B-77 Whole-store `export` and `import` belong to stream F, as JSON Lines (amends B-8) (2026-11-16)

- **Decision.**
  - **Owner.** Stream F builds whole-store `export` and `import` in `sm-share`, beside `migrate`, whose project and redaction code they share. The lead wires them into `sm-daemon` in phase 3 with `share`, `sync` and `migrate` (B-79).
  - **Format.** JSON Lines: UTF-8, LF line endings, one JSON object per line.
    - The first line is a header: `{"format":"spatial-memory-export","version":1,"build":"<binary version>","exported_at":"<RFC 3339>","schema":<SCHEMA_VERSION>}`.
    - Then one `{"project":{…}}` line per project in id order, with its identity key, name, remote and root path (`Store::project_records`).
    - Then one `{"memory":{…}}` line per memory in id order, streamed with `iter_memories`: every stored field, including provenance, anchors, origin, `supersedes_id` and the `imported` JSON.
    - Then one `{"link":{…}}` line per stored link (`Store::project_links`).
  - **Nothing derived.** No chunks, vectors, keyword rows, map runs or injections. Import chunks the content again and queues the embedding.
  - **Export.**
    - Refuses a file that exists, with `invalid_input`.
    - Writes a temporary file in the same folder, then renames it, so a failed export leaves no partial file under the requested name.
    - `project` selects one project by name. A name that matches no project is `not_found`, and one that matches two is `invalid_input`, as for `project rename`.
    - The same store gives the same bytes, apart from the header.
  - **Import.**
    - Refuses an unknown `format` or a newer `version` with `invalid_input`, before writing anything.
    - Finds or creates each project by identity key (`Store::ensure_project`).
    - Keeps each memory's id. An id already present counts in `skipped_existing`, so an interrupted import resumes where it stopped.
    - Scans and redacts content like `migrate`. Redacting twice changes nothing (B-64), so a store exported from version 3 comes back unchanged.
    - A link whose ends are missing is reported as an `ImportIssue` and skipped.
- **Why.**
  - SPEC 8 moves whole-store `export` and `import` to the command line, and SPEC 15 "One user, several machines" names them as the way to move a store by hand. No stream owned them, and CONTRACTS 2 listed only `share`, `sync` and `migrate` for `sm-share`.
  - JSON Lines streams a large store without holding it in memory, and gives each error a line number.
  - Scanning on import is the conservative choice for a file that can be edited by hand.
- **Evidence.** `Store::project_records`, `Store::project_links` and `Store::ensure_project` (lead change, this entry and B-79). Tests `s10_ensure_project_finds_or_creates_by_identity`, `s4_project_links_lists_every_stored_link`.
- **Touches.** SPEC 8 "Moved to the command line", SPEC 15 "One user, several machines", CONTRACTS 2 and 5, B-8.

## B-78 How `sync` reads the repository and records what it read (2026-11-16)

- **Decision.**
  - **Reading.**
    - `sync` reads the committed tree at `HEAD`, not the working tree. `git ls-tree -r -z HEAD -- .spatial-memory/shared/` lists each file with its blob id; `git cat-file --batch` reads only the blobs that changed; `git rev-parse HEAD` gives the commit.
    - Uncommitted edits are not synced: review is the gate (SPEC 9, step 2). A repository with no commits has nothing to sync.
    - git is started directly, never through a shell, with `CREATE_NO_WINDOW` on Windows. Stream F chooses the error code for a missing git and records it; the next action says to install git.
  - **What is recorded.**
    - Each synced memory records its file in provenance, through `Store::record_sync` and `SyncMark`: `file_path` (relative to the repository root, `/` separators), `commit_sha`, and `extra.sync = {"blob": …, "text_hash": …}`.
    - `commit_sha` is the file's `commit` field when it has one, otherwise the `HEAD` commit at which the file was read.
    - A file whose blob id equals the recorded one is not read again.
  - **Text hash.** The SHA-256 of the title and body exactly as `share` writes them, so a memory and the file shared from it hash the same.
    - When the hash changes, the content is chunked again and the embedding queued.
    - A change to type, status, tags or anchors only updates the row; nothing is embedded (SPEC 9).
    - `supersedes` is applied when the memory is first inserted. The store writes `supersedes_id` only at insert (B-73), so a later change to a file's `supersedes` is not applied: the rest of the file is, and a `FileIssue` says so.
  - **Each file and memory.**
    - **A new id.** Inserted with origin `shared`, provenance source `sync`, `Access::Sync`, and pending chunks.
    - **An id held by a local memory of this project.** This is the author's own copy, after the pull request merges. It is adopted with `record_sync` when the text hashes are equal. Otherwise nothing changes, and a `FileIssue` tells the user to share the memory again or forget the local copy.
    - **An id that belongs to another project, or that appears in two files.** A `FileIssue`; neither file is applied.
    - **A shared memory whose file is gone at `HEAD`.** Retracted (status `retracted`, not deleted) and listed in `removed`. If the file comes back, a later `sync` applies it again.
  - **`dry_run`** reads and reports, and writes nothing.
  - **No sync on project open.** SPEC 9 step 3 lets the daemon sync when a project opens, "if enabled", but SPEC 15 has no setting for it. Version 3.0 syncs only when asked.
- **Why.**
  - SPEC 9 says what `sync` compares (git blob ids) and what it does (retract, revise, refuse local edits). It does not say where the stored blob ids live, which tree is read, or what happens to the author's own copy.
  - SPEC 4's provenance already has `file_path` and `commit_sha`, so no schema change is needed.
  - Reading `HEAD` keeps unreviewed edits out.
  - Adopting the author's copy only when the hashes match never overwrites local content with text the author has not seen.
- **Evidence.** `Store::record_sync` and `SyncMark` (lead change); test `s9_record_sync_makes_a_memory_shared_with_its_file`. SPEC 9 measured 9 ms to list 10,000 files.
- **Touches.** SPEC 4 `provenance`, SPEC 9 "Team sharing" and "Shared file format", B-72.

## B-79 How the daemon reaches `sm-share`, and projects created by identity (amends B-8) (2026-11-16)

- **Decision.**
  - **A trait in `sm-daemon`.**
    - Stream E defines `pub trait SharingOps: Send + Sync` with five methods: `share`, `sync`, `migrate`, `import` and `export`.
    - Each method takes a context and the op's parameters, and returns the op's reply. The context holds the `Store`, the `Chunker`, the `SecretScanner`, `remember`'s `ChunkParams`, and the request's resolved project and identity when there is one.
    - The daemon cannot be built without an implementation. E's tests pass a fake, so nothing in `sm-daemon` answers these ops with a placeholder.
    - In phase 3 the lead implements the trait over `sm-share`, and the `daemon` subcommand passes it in.
    - After each call, the daemon wakes the embedding queue and drops the project's cached maps.
  - **Functions stream F exposes**, so the phase 3 adapter is mechanical. F may reorder arguments, but needs no daemon type.
    - `share(store, scanner, project, repo_root, &ShareParams) -> Result<ShareReply>`
    - `sync(store, chunker, chunk_params, project, repo_root, &SyncParams) -> Result<SyncReply>`
    - `migrate(store, chunker, scanner, chunk_params, &MigrateParams, cwd_project) -> Result<ImportReport>`
    - `export(store, &ExportParams) -> Result<ExportReport>`
    - `import(store, chunker, scanner, chunk_params, &ImportParams) -> Result<ImportReport>`
  - **Projects by identity.**
    - `Store::ensure_project(&ProjectRecord)` finds a project by identity key or creates it, and never changes an existing one.
    - `Store::project_records()` lists every project with its key.
    - Keys are `git:<remote>` or `path:<root key>` (B-20), or `name:<name>` for a project known only by name, which no working directory resolves to.
  - **`migrate`'s project matching** (SPEC 10). For a record's `project` value:
    1. An existing project with key `git:<value>`.
    2. Else the one project with that name.
    3. Else a new project. When the value is a normalised remote (`host/path`, as version 1 stored them), the project gets key `git:<value>` and that remote, so opening a clone of the repository finds it. Otherwise it gets key `name:<value>` and no remote, as SPEC 10 says.
    - Two projects with the name fail the migration with `invalid_input` before anything is written.
    - Records without a `project` field follow B-35.
- **Why.**
  - SPEC 14 runs streams E and F at the same time, so E cannot call F's functions, and the build prompt forbids placeholder handlers (B-8). A trait keeps the daemon complete and testable now and makes the phase 3 wiring a small adapter.
  - SPEC 10 says unmatched values create a project with no remote. But version 1 stored normalised remotes as project values (B-20 keeps version 1's normalisation quirks for this reason), and a project with no remote would never be found from its repository. The `name:` key covers values that are only names, such as the fixtures' `spatial-memory-mcp`.
- **Gap.** A `name:` project is found only with `scope: all`, unless `migrate` runs inside a repository whose project has the same name. MIGRATION.md says to run `migrate` from inside the main repository.
- **Evidence.** Test `s10_ensure_project_finds_or_creates_by_identity`.
- **Touches.** SPEC 10, SPEC 14, CONTRACTS 2, B-8, B-20, B-35.
