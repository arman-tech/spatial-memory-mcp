# Requirement trace

Maps every requirement identifier to the tests that cover it and the commits that delivered it. A requirement with no test is a gap and is listed as one.

- **Identifiers.** `AT-n` is acceptance test n in SPEC 11. `D-n` is the n-th decision in SPEC 2's table. Other rules are cited as a SPEC section plus rule name. Spike decisions are cited as `spikeN` (SPIKE-WINDOWS.md, decisions 1 to 6).
- **Test names.** Every test name starts with its identifier: `atN_`, `dN_`, `sN_` or `spikeN_` (DECISIONS B-40). This file is checked against the test names with `rg -o "fn ((at|d|s|spike)[0-9]+_[a-z0-9_]+)" -r '$1' crates`.
- **States.** *Covered*: the requirement's own test exists and passes. *Partial*: tests cover the shared types or formulas it rests on; the end-to-end test is still owed by the named stream. *Gap*: no test yet.
- **Last updated.** Phase 0, 16 November 2026. All tests below are in `crates/sm-core`. Windows: 96 pass (one test is Unix-only). Linux (Docker, `rust:1.95-trixie`): 97 pass.

## Acceptance tests (SPEC 11)

| Id | Requirement | State | Tests | Owner of the remaining test |
| --- | --- | --- | --- | --- |
| AT-1 | Embedding correctness | Gap | Spike a passed outside the workspace (SPIKE.md) | Stream B; real-model integration binary |
| AT-2 | Retrieval regression | Partial | `at2_eval_fixtures_load`, `at2_metrics_match_version_2_formulas`, `at2_gate_fails_only_beyond_the_fixture_tolerances`, `at2_perfect_run_scores_one_and_reports_distributions` | Phase 3, real-model integration binary |
| AT-3 | Hook latency and safety | Gap | | Streams D and E; phase 3 |
| AT-4 | Long input, 50,000 characters, peak under 2 GB | Gap | | Stream B (chunker); phase 3 |
| AT-5 | Concurrency, two readers and one writer | Gap | | Stream A; stream E end to end |
| AT-6 | Determinism of `regions` | Gap | | Stream C |
| AT-7 | Clustering quality | Partial | `s7_cluster_params_follow_the_spec_formula` | Stream C |
| AT-8 | Migration | Partial | `s10_content_hash_matches_v1_export_fixture`, `s10_v1_uuid_ids_are_kept`, `s10_non_uuid_v1_ids_map_to_stable_v8`, `s10_v1_fixture_timestamps_round_trip_unchanged`, `s10_parses_v1_naive_timestamps_as_utc` | Stream F |
| AT-9 | Redaction and share refusal | Partial | `at9_redaction_replaces_and_merges_matches` | Stream G (scanner), stream F (share refusal) |
| AT-10 | Model integrity | Gap | | Stream B; stream E for the daemon's refusal |
| AT-11 | Index mismatch | Partial | `at11_changed_model_or_chunk_size_is_a_mismatch`, `at11_index_stamp_round_trips_through_meta` | Stream A; stream E for blocking search |
| AT-12 | Scope | Partial | `at12_profile_order_is_flag_plugin_env_rule_default` | Stream A (project scope); phase 3 (profiles) |
| AT-13 | Capture from tools | Gap | | Stream G; stream E end to end |
| AT-14 | Capture from the transcript | Gap | | Stream G; stream D for the printed responses |
| AT-15 | Capture never blocks and never loses | Partial | `s16_spool_write_is_tmp_then_rename`, `s16_spool_lists_in_order`, `s16_spool_removes_only_stale_tmp_files`, `s16_spool_dirs_are_user_only` (Unix) | Stream D (hook spools), stream E (drain) |
| AT-16 | Both tiers are saved | Gap | | Stream G |
| AT-17 | Compatibility with version 2's hooks | Partial | `at17_ambient_block_keeps_version_2_format`, `at17_block_has_header_id_lines_and_footer`, `at17_carriage_returns_in_block_lines_become_spaces`, `at17_version_2_ambient_switches_keep_their_rules` | Stream D (`off` never contacts the daemon); stream G and A (index cap) |

## Decisions (SPEC 2)

| Id | Decision | State | Tests | Owner of the remaining test |
| --- | --- | --- | --- | --- |
| D-1 | Language and shape: one Rust binary, daemon holds the model | Partial | `s3_*` protocol tests | Phase 3 |
| D-2 | Storage: SQLite with exact vector search | Gap | Spike b passed outside the workspace | Stream A |
| D-3 | Keyword search: FTS5 | Gap | | Stream A |
| D-4 | Embedding model: EmbeddingGemma 2 | Partial | `s4_model_dir_override_and_cache_layout` | Stream B |
| D-5 | Fallback model: Arctic Embed M behind a Cargo feature | Gap | | Stream B (B-23) |
| D-6 | Chunk limit | Partial | `s14_simple_chunker_windows_and_cap` (the fake chunker's contract) | Stream B |
| D-7 | Clustering | Partial | `s7_cluster_params_follow_the_spec_formula` | Stream C |
| D-8 | Cross-project structure: tables now, map in 3.1 | Partial | `s4_anchor_refs_parse_and_print`, `s4_link_ends_distinguish_memories_and_anchors` | Stream A (anchor and link tables) |
| D-9 | GraphRAG: not built in 3.0 | Not applicable | Nothing to test; the link table is the only part in 3.0 | |
| D-10 | Runtime threads | Partial | `s5_thread_counts_follow_physical_cores` | Stream B (budget and priority); AT-3 |

## SPEC sections

| Section and rule | Tests |
| --- | --- |
| 3 Transport, framing and version handshake | `s3_frame_round_trip_and_size_limit`, `s3_version_mismatch_reply_shape`, `s3_op_tags_are_stable`, `s3_request_carries_version_cwd_and_session`, `s3_long_socket_paths_move_to_the_runtime_dir` |
| 4 Data model | `s4_anchor_refs_parse_and_print`, `s4_derive_title_takes_first_text_line`, `s4_enums_use_spec_strings`, `s4_importance_must_be_in_range`, `s4_link_ends_distinguish_memories_and_anchors`, `s4_malformed_memory_ids_are_rejected`, `s4_new_memory_ids_are_uuid_v7`, `s4_timestamp_civil_conversion_round_trips`, `s4_timestamps_serialise_as_rfc3339`, `s4_home_override_replaces_the_root_and_keeps_profiles_apart`, `s4_model_dir_override_and_cache_layout`, `s4_platform_dirs_follow_the_xdg_rules`, `s4_platform_dirs_resolve_on_this_machine` |
| 5 Configuration defaults and threads | `s5_defaults_match_the_spec`, `s5_out_of_range_values_are_config_invalid`, `s5_thread_counts_follow_physical_cores` |
| 6 Ambient recall, gates and journal | `s6_block_stays_under_800_tokens`, `s6_default_filter_keeps_active_only`, `s6_gates_floor_repeat_cap_in_order`, `s6_hook_gates_have_stable_names`, `s6_journal_date_string_and_days_since`, `s6_journal_files_are_per_process_and_pruned_after_7_days`, `s6_journal_form_matches_python_isoformat`, `s6_journal_line_matches_version_2_shape`, `s6_journal_lines_stay_under_4096_bytes`, `s6_minimum_prompt_is_four_words`, `s6_percentile_matches_python_statistics_quantiles`, `s6_python_int_matches_python` |
| 7 Clustering parameters | `s7_cluster_params_follow_the_spec_formula` |
| 8 Tools, errors and conventions | `s8_enum_schema_lists_values`, `s8_error_codes_round_trip_as_snake_case_strings`, `s8_error_serialises_code_message_and_next_action`, `s8_memory_view_omits_empty_fields`, `s8_revise_supersede_excludes_other_edits`, `s8_thirteen_tools_with_generated_schemas`, `s8_tool_args_parse_and_validate`, `s8_unknown_error_code_is_rejected` |
| 9 Scope, profiles and projects | `s9_default_name_prefers_remote_then_folder`, `s9_filesystem_root_is_not_a_project`, `s9_folder_without_repository_is_identified_by_path`, `s9_git_config_remote_priority_is_upstream_origin_first`, `s9_last8_is_the_id_suffix`, `s9_profile_config_overrides_root_key_by_key`, `s9_profile_names_are_restricted`, `s9_remote_forms_normalise_like_version_1`, `s9_remote_quirks_are_kept_from_version_1`, `s9_settings_load_from_files`, `s9_tags_are_validated_and_deduplicated` |
| 10 Migration: hashes, ids and timestamps | `s10_content_hash_ignores_case_and_surrounding_space`, `s10_content_hash_matches_v1_export_fixture`, `s10_invalid_timestamps_are_rejected`, `s10_non_uuid_v1_ids_map_to_stable_v8`, `s10_parses_v1_naive_timestamps_as_utc`, `s10_python_strip_matches_python_isspace`, `s10_sha256_hex_matches_known_vector`, `s10_timestamps_parse_offsets_and_separators`, `s10_v1_fixture_timestamps_round_trip_unchanged`, `s10_v1_uuid_ids_are_kept` |
| 11 Evaluation on real memories (`doctor --eval`) | `s11_doctor_eval_sets_may_omit_memories_but_not_queries` |
| 14 Phase 0 deliverable: the fake embedder, provider and chunker that SPEC 11 "Test conventions" requires | `s14_fake_embedder_hash_fnv1a64_matches_known_vectors`, `s14_fake_embedder_is_deterministic_and_unit_length`, `s14_fake_provider_opens_sessions`, `s14_simple_chunker_windows_and_cap` |
| 16 Capture: events, ignore list, project opt-out, spool, stdout | `s16_capture_ignore_patterns_cover_env_and_key_files`, `s16_file_anchors_are_repository_relative`, `s16_glob_rules`, `s16_head_commit_reads_packed_refs_and_detached_heads`, `s16_hook_events_use_version_2_names`, `s16_plugin_option_is_read_before_version_2_variables`, `s16_project_toml_opts_out_of_capture_and_broken_file_is_conservative`, `s16_session_start_output_matches_version_2`, `s16_spool_dirs_are_user_only`, `s16_spool_lists_in_order`, `s16_spool_removes_only_stale_tmp_files`, `s16_spool_write_is_tmp_then_rename`, `s16_stdout_contract_for_continue_and_prompt`, `s16_worktree_git_file_is_followed_to_the_common_dir` |

## Spike decisions (SPIKE-WINDOWS.md)

| Decision | Tests |
| --- | --- |
| 1 Named pipe for the current user only | `spike1_pipe_name_ignores_slash_direction_and_case`; the security descriptor is stream E's test |
| 2 No dependence on `CREATE_BREAKAWAY_FROM_JOB` | Gap; stream D's auto-start test (CONTRACTS.md 3.1, B-44) |
| 3 One write call per log, journal and spool line | `s6_journal_lines_stay_under_4096_bytes`, `s16_spool_write_is_tmp_then_rename` |
| 4 Paths differing only in slash or case are one project on Windows | `spike4_windows_paths_differing_in_slash_or_case_share_a_key`, `spike4_strip_verbatim_removes_only_drive_and_unc_prefixes` |
| 5 The query session is created first and kept for life | Gap; stream E |
| 6 Cargo runs through PowerShell | Process rule (B-2); not testable |

## Commits

| Commit | Identifiers |
| --- | --- |
| `97426c2` | SPEC 14 phase 0; B-1 to B-41; the `sm-core` tests above |
| Phase 1 preparation (stream specifications) | SPEC 14 phase 1; B-42 to B-44; no new tests |
