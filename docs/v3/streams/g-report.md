# Stream G report: `sm-capture`

Branch `arman-tech/stream-g-capture`, from `v3` at `ea82480`. Code commit `a310453`; this report is the commit after it.

## 1. Done when

Yes. All 726 ported cases from version 1's and version 2's hook tests pass on Windows and on Linux. The other 278 cases are not applicable, and each one has a stated reason (section 4).

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-capture/Cargo.toml` | The crate. Dependencies: `sm-core`, `regex`, `serde_json`. Dev-dependencies: `sm-core` with `test-support`, `tempfile`. All come from the workspace pins, with no extra features. `[lints] workspace = true`. |
| `crates/sm-capture/src/lib.rs` | Crate documentation and re-exports. |
| `crates/sm-capture/src/pipeline.rs` | `capture`: the gates and the steps in order, for PostToolUse, PreCompact and Stop. |
| `crates/sm-capture/src/extract.rs` | Version 2's per-tool extraction with one 200,000-character cap, version 1's tool skip rules, and recognition of this server's own tools. |
| `crates/sm-capture/src/signal.rs` | The 22 signal patterns, tiers, importance, and the mapping from pattern family to memory type. |
| `crates/sm-capture/src/secrets.rs` | `RegexSecretScanner`, which implements `sm_core::traits::SecretScanner`: 19 key patterns, 3 value patterns, the false-positive filter, and version 1's skip rule. |
| `crates/sm-capture/src/session.rs` | `SessionState`: this session's content hashes, at most 4,096, oldest dropped first. |
| `crates/sm-capture/src/transcript.rs` | The stateless transcript delta reader, the path and session-identifier checks, and the assistant-text extractor. |
| `crates/sm-capture/src/candidate.rs` | `Candidate` and `Candidate::to_new_memory`, tags and the B-19 tag check, and the `file` anchor. |
| `crates/sm-capture/src/tests/` | `mod.rs` (fixture loading and per-file coverage counts), `checker.rs` (the README's `{path, op, value}` checker and the `$concat`, `$repeat` and `$float` generators), and one test module per fixture file. Also `acceptance.rs`, `rules.rs` and `budgets.rs`. |
| `fixtures/capture/rust/README.md` | Where the two files below came from and how to regenerate them. |
| `fixtures/capture/rust/cursor_normalized_payloads.json` | The ten Cursor payloads for PostToolUse, PreCompact and Stop, before and after normalisation. The outputs come from running version 2's own `_normalize_stdin` (commit `4e51df0a5388ebb117f4fa29019424bc4c2602db`), not a reimplementation of it. |
| `fixtures/capture/rust/at14_session.jsonl` | A hand-written five-line transcript for AT-14. |
| `Cargo.lock` | Gains `sm-capture` and the `regex` family (section 10). |
| `docs/v3/streams/g-report.md` | This report. |

### Public API (application programming interface)

These are re-exported at the crate root:

```rust
pub use candidate::{Candidate, DroppedAnchor, DroppedTag};
pub use pipeline::{CaptureInput, CaptureOutcome, Skip, TextDrop, TextSource, capture};
pub use secrets::{RegexSecretScanner, SecretReport};
pub use session::{SESSION_HASH_CAP, SessionState};
pub use signal::{Signal, Tier, classify};
```

**The entry point stream E calls**

```rust
pub fn capture(input: &CaptureInput<'_>, session: &mut SessionState) -> CaptureOutcome;

pub struct CaptureInput<'a> {
    pub event: HookEvent,                 // from the CLI subcommand, via CaptureParams.event
    pub payload: &'a Value,               // already normalised to Claude Code's names
    pub settings: &'a CaptureSettings,    // the profile's [capture]
    pub project: &'a ProjectIdentity,     // capture_enabled = the project.toml opt-out; root anchors files
    pub commit_sha: Option<&'a str>,      // the daemon's repository lookup
    pub transcript_offset: u64,           // this session's offset; 0 at first
}

pub struct CaptureOutcome {
    pub candidates: Vec<Candidate>,
    pub skipped: Option<Skip>,            // why the whole request gave nothing
    pub source: Option<TextSource>,       // ToolCall | LastAssistantMessage | Transcript
    pub drops: Vec<TextDrop>,             // Empty | Weak | NonFiniteScore | Secret | Duplicate, one per text
    pub new_offset: Option<u64>,          // store it when Some; None = keep the old one
    pub cap_hit: bool,                    // for capture_cap_hit_total
    pub extracted_chars: usize,           // the size before the cap, in characters
}

pub enum Skip { CaptureDisabled, ProjectOptedOut, NotACaptureEvent(HookEvent), NoToolName,
                ToolNotCaptured(String), ToolSkipped(ToolSkip), PathIgnored(String),
                StopHookActive, NoTranscript }
```

**The candidate**

`Candidate` has these fields:
- `source_hook`
- `content`: redacted
- `content_hash`: B-14
- `memory_type`
- `status`: always `Active`
- `importance`, `score`, `tier`
- `patterns`: the families
- `redaction_kinds`
- `tags`, `dropped_tags`
- `provenance`
- `anchors`, `dropped_anchor`
- `v1_namespace`
- `cap_hit`

`Candidate::to_new_memory(&self, project_id: ProjectId) -> NewMemory` sets the content, the hash, the type, the status, `Origin::Local`, the importance, the tags, the provenance and the anchors. The title, identifier and times are left for the store to fill.

**Modules, for direct use and tests**

| Module | Public items |
|---|---|
| `signal` | `SignalPattern`, `SIGNAL_PATTERNS: [_; 22]`, `STRONG_THRESHOLD`, `MEDIUM_THRESHOLD`, `Tier::{from_score, number, from_number}`, `Signal::none`, `classify`, `matching_pattern_ids`, `importance(tier, score) -> Option<f64>`, `NAMESPACE_MAP`, `v1_namespace`, `memory_type_for` |
| `secrets` | `KeyPattern`, `ValuePattern`, `KEY_PATTERNS: [_; 19]`, `VALUE_PATTERNS: [_; 3]`, `FALSE_POSITIVE_REGEX`, `is_false_positive`, `python_placeholder`, `SecretReport{redaction, matches, has_private_key, should_skip}` with `count()`, `RegexSecretScanner::{new, report}`, and `impl SecretScanner` |
| `extract` | `MAX_CAPTURE_CHARS` (200,000), `TRUNCATION_MARKER` (`...`), `truncate`, `safe_get`, `python_str`, `python_truthy`, `python_float`, `python_string_repr`, `python_json_dumps`, `tool_response_text`, `extract_edit`, `extract_write`, `extract_bash`, `extract_notebook_edit`, `extract_mcp_tool`, `Extracted{text, cap_hit, original_chars}`, `extract_content`, `SKIP_TOOLS`, `SPATIAL_MEMORY_PREFIX`, `V1_TOOL_NAMES`, `ToolSkip::reason`, `is_spatial_memory_tool`, `skip_rule` |
| `transcript` | `MAX_LINE_BYTES`, `MAX_SESSION_ID_CHARS`, `is_windows_device_name`, `sanitize_session_id`, `check_transcript_path`, `validate_transcript_path`, `TranscriptEntry`, `parse_entry`, `TranscriptRead{entries, new_offset}`, `read_transcript(path, last_offset)`, `AssistantTexts{texts, cap_hit}`, `extract_assistant_texts` |
| `session` | `SessionState::{new, contains, record, record_content, is_duplicate, len, is_empty, hashes}` |
| `candidate` | `TRANSCRIPT_TAG`, `DroppedTag{tag, reason}`, `DroppedAnchor{name, reason}`, `validate_tags`, `derive_tags`, `derive_transcript_tags`, `file_anchor` |

**How stream E uses it**
1. Map `CaptureParams{event, payload, received_at}` to a `CaptureInput`.
2. Keep one `SessionState` and one transcript offset for each session.
3. Store `new_offset` when it is `Some`.
4. Turn each candidate into a `NewMemory` with `to_new_memory`, then store and embed it. The store's own dedupe applies (B-29).
5. Journal two metrics:
   - `capture_cap_hit_total`, from `cap_hit`.
   - `capture_size_tokens`, from the text. This crate reports `extracted_chars` in characters, not tokens, because it has no tokenizer.

## 3. Test commands and exact results

Every command was run from the worktree root on the committed tree (`a310453`).

**Windows 11 Pro 10.0.26300, PowerShell, debug**

`cargo fmt --check`: exit 0.

`cargo clippy --workspace --all-targets --all-features -- -D warnings`: exit 0, no warnings.

`cargo test -p sm-capture --all-features -- --nocapture`:
```
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo test --workspace --all-features`:
```
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s   (sm-capture)
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm-core)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests, sm-capture)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests, sm-core)
```

**Windows, release:** `cargo test -p sm-capture --all-features --release -- --nocapture`
```
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**Linux in Docker** (`rust:1.95-trixie`, the source mounted read-only, target volume `sm-v3-target-g`)

The stream's own crate, with the RULES 6 command plus `-- --nocapture` on the test step, which prints the coverage counts and budgets: `cargo test -p sm-capture --all-features --locked -- --nocapture && cargo clippy -p sm-capture --all-targets --all-features --locked -- -D warnings`
```
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```
Clippy: no warnings.

The documented workspace check (CLAUDE.md), with `cargo fmt --check` added: `cargo test --workspace --all-features --locked && cargo clippy --workspace --all-targets --all-features --locked -- -D warnings && cargo fmt --check && echo LINUX_ALL_GATES_OK`
```
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s   (sm-capture)
test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s   (sm-core)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests, sm-capture)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests, sm-core)
LINUX_ALL_GATES_OK
```

Platform pairs:
- The 33 tests on Linux include `s16_hook_helpers_cases_unix` and `s16_ignore_patterns_fold_case_unix`. On Windows, `..._windows` takes their place (RULES 7).
- sm-core runs 96 tests on Windows and 97 on Linux. That difference comes from sm-core's own platform-specific test, not from this stream.

Stub check: `rg "todo!|unimplemented!|#\[ignore\]|unsafe" crates/sm-capture` finds nothing.

## 4. Acceptance tests

| Test | Library part checked here | Test names | Windows | Linux |
|---|---|---|---|---|
| AT-9 | Every case in `redaction_cases.json`. Over every input, and over all inputs joined: each match lies on `char` boundaries, each kind is in `v3_mapping`, `scan(redact(x).text)` is empty, and redacting twice changes nothing. The same checks run over redacted text placed next to raw text. The 19 + 3 patterns match `redaction_patterns.json` in order, tier, kind, private-key flag, expression and placeholder. | `at9_redaction_cases`, `at9_redaction_patterns_table` | pass | pass |
| AT-13 | Every `hook_payloads/` case for PostToolUse, PreCompact and Stop, in both Claude Code and Cursor form. Under the default settings: Edit, Write and the Bash AWS (Amazon Web Services) payload each give one strong, active candidate with importance ≥ 0.7. The Edit and Write candidates are anchored to `/src/db.py` and `/docs/fix.md`. `AKIA1234567890ABCDEF` becomes `[REDACTED:aws_key]`, with kind `aws_key`. The trivial Bash payload is dropped as weak. A strong text holding a private key is dropped as `Secret`. An MCP (Model Context Protocol) tool not in `capture.tools` gives `ToolNotCaptured`. | `at13_hook_payloads`, `at13_hook_payloads_default_settings` | pass | pass |
| AT-14 | PostToolUse captures two texts. PreCompact then reads the same transcript with the same `SessionState` and returns only the texts PostToolUse missed. Stop prefers `last_assistant_message`, and falls back to the transcript when that field is absent. All cases in `overlap_cases.json`. The session bound drops the oldest hash first. | `at14_pre_compact_returns_only_what_post_tool_use_missed`, `at14_stop_prefers_last_assistant_message`, `at14_stop_reads_the_transcript_when_there_is_no_final_message`, `at14_overlap_cases`, `at14_session_state_bound_is_fifo` | pass | pass |
| AT-16 | A medium-signal text gives an `Active` candidate with importance ≥ 0.4, and a strong one ≥ 0.7. | `at16_medium_and_strong_signals_give_active_candidates` | pass | pass |
| AT-17 (cap clause) | Text over 200,000 characters is cut at extraction, the marker is added, and `cap_hit` is set. This covers PostToolUse, `last_assistant_message` and the transcript. | `at17_long_capture_is_cut_at_extraction_with_the_marker` | pass | pass |

**SPEC 16 rules** (stream file, "SPEC 16 rules"); all pass on both platforms:

| Test | What it checks |
|---|---|
| `s16_capture_disabled_gives_nothing` | `capture.enabled` off gives nothing. |
| `s16_project_opt_out_gives_nothing` | The `project.toml` opt-out, through `ProjectIdentity.capture_enabled`, gives nothing. |
| `s16_a_tool_not_in_capture_tools_gives_nothing` | A tool not in `capture.tools` gives nothing. |
| `s16_each_default_ignore_pattern_skips_edit_and_write` | Each of the 13 B-21 ignore patterns skips Edit and Write paths. |
| `s16_ignore_patterns_fold_case_windows` / `_unix` | Ignore patterns fold case on Windows and not on Unix. |
| `s16_gate_order` | The gates run in order. |
| `s16_own_tools_are_never_captured` | This server's own tools are never captured. |
| `s16_edit_and_write_are_anchored_to_their_file` | Edit and Write get a `file` anchor. |
| `s16_provenance_records_capture_with_session_tool_file_and_commit` | Provenance has source `capture` with the session, tool, file and commit. |
| `s16_types_come_from_pattern_families` | The type comes from the pattern families. |
| `s16_to_new_memory_leaves_the_store_fields_empty` | `to_new_memory` leaves the store's fields empty. |
| `s16_stop_loop_guard_and_missing_transcript` | Stop's loop guard, and Stop with no transcript. |
| `s16_hook_payloads_not_a_capture_event` | SessionStart and UserPromptSubmit give `NotACaptureEvent`. |
| `s16_cursor_normalised_payloads_are_read_by_claude_code_names` | The 23 `normalize_stdin_cases` outputs are read by Claude Code's field names. |

### Fixture coverage (the done-when)

Each fixture file has its own test. The test runs every case and lists every failing case id together. It also prints the counts below.

| File | Test | Ported | Passing | Not applicable |
|---|---|---|---|---|
| `redaction_patterns.json` | `at9_redaction_patterns_table` | 26 | 26 | 0 |
| `redaction_cases.json` | `at9_redaction_cases` | 43 | 43 | 0 |
| `signal_patterns.json` | `s16_signal_patterns_table` | 23 | 23 | 0 |
| `signal_cases.json` | `s16_signal_cases` | 64 | 64 | 0 |
| `content_extractor_cases.json` | `s16_content_extractor_cases` | 125 | 125 | 0 |
| `transcript/transcript_cases.json` (and its `.jsonl` files) | `s16_transcript_cases` | 59 | 59 | 11 |
| `overlap_cases.json` | `at14_overlap_cases` | 37 | 37 | 23 |
| `hook_helpers_cases.json` | `s16_hook_helpers_cases_windows` (against `computed_win32`) / `_unix` (against `computed_posix_simulated`) | 54 | 54 | 84 |
| `queue_and_pipeline_notes.json` | `s16_queue_and_pipeline_notes` | 227 | 227 | 97 |
| `hook_payloads/` (with `INDEX.json`) | `at13_hook_payloads` | 45 | 45 | 63 |
| `event_aliases.json` `normalize_stdin_cases`, as capture input | `s16_cursor_normalised_payloads_are_read_by_claude_code_names` | 23 | 23 | 0 |
| **Total** | | **726** | **726** | **278** |

`ambient_recall_cases.json` and the rest of `event_aliases.json` (aliases, registration, stdout) belong to stream D or to the daemon's ambient recall. They are not ported here.

**Not applicable, with reasons and counts.** The test prints each case id; these are the totals.

- `transcript_cases.json` (11): the daemon keeps each session's offset (stream E). This crate's reader is stateless, and the B-29 content-hash dedupe catches a re-read.
- `overlap_cases.json` (23): version 1's queue files are replaced by the spool (B-25, B-38). This session's hashes are held in `SessionState`, which the candidates it produced feed.
- `hook_helpers_cases.json` (84):

  | Count | Reason |
  |---|---|
  | 23 | stdin is stream D's |
  | 21 | The hook error log is replaced by the journal. |
  | 17 | The working-directory check is stream D's. |
  | 7 | The working-directory check is stream D's. The daemon resolves the project from `cwd` with sm-core's `resolve_project` (CONTRACTS 2). |
  | 4 | The project root is stream D's. |
  | 3 | The project root is stream D's and sm-core's `resolve_project`. |
  | 2 | `load_state` and `save_state` keep the per-session offset, which is the daemon's. |
  | 2 | stdout is stream D's. |
  | 5 | Constants that are stream D's or the journal's: `4096`, `1048576`, `524288`, `"hook-errors.log"`, `"hook-errors.log.1"`. |

- `queue_and_pipeline_notes.json` (97):

  | Count | Reason |
  |---|---|
  | 58 | Queue files are replaced by the spool (B-25, B-38). |
  | 8 | The per-session transcript offset is the daemon's. This crate takes `transcript_offset` and returns `new_offset`. |
  | 7 | Every expectation is the spool's, the daemon's or stream D's. |
  | 6 | This crate records no client. Stream D normalises Cursor payloads first. |
  | 4 | `CaptureOutcome` replaces `ProcessingResult`. The mapping is exercised by the pipeline cases. |
  | 3 | The queue writer's rate limit is replaced by the spool. |
  | 2 | The candidate belongs to `CaptureInput.project`, and the daemon stores it under that project's `ProjectId` (CONTRACTS 7). |
  | 2 | The queue file path (spool). |
  | 2 | The stored offset is the daemon's. |
  | 1 | The payload has no `hook_event_name`. The event comes from the CLI subcommand as `CaptureInput.event`. |
  | 1 | The event can never be empty, for the same reason. |
  | 1 | Version 3's `Tier` has no tier 0. |
  | 1 | The stored offset and the queued hashes are replaced by the daemon and by `SessionState`. |
  | 1 | Python only (`$py_literal`): a lone surrogate cannot be held in a Rust `String`, and the queue writer it tested is replaced by the spool. |

- `hook_payloads/` (63):

  | Count | Reason |
  |---|---|
  | 26 | SessionStart and UserPromptSubmit are ambient recall, which belongs to stream D and the daemon. `capture` returns `Skip::NotACaptureEvent`, and `s16_hook_payloads_not_a_capture_event` checks it. |
  | 23 | The hook's stdout is stream D's. |
  | 5 | The session's transcript state is the daemon's. |
  | 4 | The payload is raw stdin text or bytes. |
  | 1 | The client is stream D's. |
  | 1 | Queue files (spool). |
  | 1 | The hook error log (journal). |
  | 1 | Files a hook run creates (stream D and the spool). |
  | 1 | The legacy per-event scripts and master's dispatcher. `expected` is version 2's dispatcher, which version 3 ports. |

Where a result satisfies `expect` but differs from a `computed_*` value, the case is listed under Decisions (G-22).

## 5. Measured budgets

**Machine.** AMD Ryzen 9 9950X3D, 16 cores and 32 logical processors, 93.6 GB of memory.
- Windows: Windows 11 Pro 10.0.26300.
- Linux: the Docker container `rust:1.95-trixie` (Debian 13) on the same machine.
- Builds:
  - Release is the workspace `[profile.release]`: thin LTO (link-time optimisation), `codegen-units = 1`.
  - Debug is the workspace `dev` profile, as `cargo test` builds it.
- How each figure was taken:
  - The pipeline figures are the median of 11 runs in release and of 3 in debug.
  - `classify_under_1ms` is the mean over the fixture's iterations and texts.
  - `definition_pattern_no_redos` is one timed call.

| Budget | Limit | Windows release | Windows debug | Linux debug |
|---|---|---|---|---|
| `signal_cases.json` `classify_under_1ms`: classify, per call | < 1 ms | **0.0007 ms** | 0.0115 ms | 0.0093 ms |
| `signal_cases.json` `definition_pattern_no_redos`: the ReDoS (regular-expression denial of service) input for `definition` | < 100 ms | **0.147 ms** | 18.047 ms | 16.859 ms |
| Whole pipeline, 200,000-character Bash output (200,041 extracted, cut to 200,000; one strong signal, one AWS key) | none in SPEC 11 | **0.600 ms**, median of 11 | 36.656 ms, median of 3 | 35.088 ms, median of 3 |
| Whole pipeline, 1 MB transcript delta, PreCompact (1,049,017 bytes, 345,105 assistant characters, cut at 200,000, 316 candidates) | none in SPEC 11 | **4.564 ms**, median of 11 | 58.369 ms, median of 3 | 52.697 ms, median of 3 |

- The two classifier budgets are asserted in `s16_signal_cases` in every profile.
- The two pipeline times are printed by `s16_pipeline_time_on_a_200k_bash_output` and `s16_pipeline_time_on_a_1mb_transcript_delta`, which also check the outcome.
- Linux was measured in debug only (section 9).

## 6. Decisions

Each entry gives what was chosen, why, the evidence, and the SPEC section it touches. The lead may turn them into B-n entries.

**G-1. Signal patterns are ported with the README's rule.** (SPEC 16 step 2)
- What changed:
  - The 15 patterns that ended in `(?:\.|$)` now end in `(?:\.|\n?$)`. This keeps Python's "`$` before a final newline" behaviour.
  - Every pattern is compiled once with `(?i)`, in a `LazyLock`.
- The 7 patterns left unchanged: `testing`, `performance`, `environment`, `dependency`, `api`, `procedure`, `workflow`.
- Each pattern's keyword prefilter runs on the lower-cased text, as version 1's did.
- Evidence: `s16_signal_patterns_table` (23/23) and `s16_signal_cases` (64/64).

**G-2. Tiers and importance.** (SPEC 16 step 5; AT-16)
- Tiers: a score ≥ 0.8 is strong, 0.5 to 0.79 is medium, and anything lower is weak and gives no candidate.
- Importance: strong is `max(0.7, score)` and medium is `max(0.4, 0.8 × score)`, clamped to [0, 1].
- A non-finite score gives no candidate (`TextDrop::NonFiniteScore`).
- Version 3 has no tier 0. Python treated tier 0 like tier 2.
- The cases where this differs from Python are listed in G-22.

**G-3. The memory type comes from the pattern family, through version 1's `derive_namespace`.** (SPEC 16 "Type and provenance"; SPEC 4)
- With several families, the first `NAMESPACE_MAP` entry, in map order, that holds any of them decides. The map order:
  1. decisions (decision)
  2. troubleshooting (error, solution, testing)
  3. patterns (pattern, convention, workaround, configuration, performance)
  4. context (environment, dependency)
  5. procedures (procedure, workflow)
  6. references (definition, api)
  7. notes (important, explicit)
- The fallback is `notes`.
- Namespaces map to types: decisions → `decision`, troubleshooting → `pitfall`, patterns → `convention`, everything else → `note`.
- Version 1's namespace is kept on the candidate (`v1_namespace`) and in `provenance.extra`, for comparison and migration.

The 22 patterns, in evaluation order. The importance column is for a text that matches that pattern alone.

| # | Pattern | Family | Confidence | Tier | Importance | Version 1 namespace | Type |
|---|---|---|---|---|---|---|---|
| 1 | `decision` | decision | 0.80 | strong | 0.80 | decisions | decision |
| 2 | `important` | important | 0.90 | strong | 0.90 | notes | note |
| 3 | `solution_fix` | solution | 0.85 | strong | 0.85 | troubleshooting | pitfall |
| 4 | `error_diagnosis` | error | 0.80 | strong | 0.80 | troubleshooting | pitfall |
| 5 | `explicit_save` | explicit | 0.95 | strong | 0.95 | notes | note |
| 6 | `pattern_learning` | pattern | 0.85 | strong | 0.85 | patterns | convention |
| 7 | `resolved_by` | solution | 0.80 | strong | 0.80 | troubleshooting | pitfall |
| 8 | `broke_failed_because` | error | 0.80 | strong | 0.80 | troubleshooting | pitfall |
| 9 | `error_due_to` | error | 0.80 | strong | 0.80 | troubleshooting | pitfall |
| 10 | `architecture` | decision | 0.80 | strong | 0.80 | decisions | decision |
| 11 | `definition` | definition | 0.60 | medium | 0.48 | references | note |
| 12 | `convention` | convention | 0.70 | medium | 0.56 | patterns | convention |
| 13 | `watch_out_for` | workaround | 0.70 | medium | 0.56 | patterns | convention |
| 14 | `workaround_is` | workaround | 0.75 | medium | 0.60 | patterns | convention |
| 15 | `configuration` | configuration | 0.65 | medium | 0.52 | patterns | convention |
| 16 | `testing` | testing | 0.70 | medium | 0.56 | troubleshooting | pitfall |
| 17 | `performance` | performance | 0.70 | medium | 0.56 | patterns | convention |
| 18 | `environment` | environment | 0.60 | medium | 0.48 | context | note |
| 19 | `dependency` | dependency | 0.60 | medium | 0.48 | context | note |
| 20 | `api` | api | 0.60 | medium | 0.48 | references | note |
| 21 | `procedure` | procedure | 0.60 | medium | 0.48 | procedures | note |
| 22 | `workflow` | workflow | 0.60 | medium | 0.48 | procedures | note |

Evidence: `s16_types_come_from_pattern_families`, and the `derive_namespace_cases` in `s16_queue_and_pipeline_notes`.

**G-4. How this crate's own tools are recognised and skipped.** (SPEC 16 step 1; version 1's `should_skip_tool`)

`extract::is_spatial_memory_tool` decides. A match is skipped with `ToolSkip::SpatialMemoryTool`, after the `capture.tools` check.

| Client | Form | Recognised as this server's tool when |
|---|---|---|
| Claude Code | `mcp__spatial-memory__<tool>` | always, for any `<tool>`: the server name is in the prefix, so all 13 version 3 names (`remember`, `recall`, `related`, `revise`, `forget`, `link`, `regions`, `journey`, `consolidate`, `share`, `sync`, `status`, `reinforce`) and any later ones are covered |
| Claude Code plugin | `mcp__plugin_<plugin>_spatial-memory__<tool>` | always, for any `<tool>` (the plugin package installs the server under this form) |
| Cursor | `MCP:<name>` | `<name>` is one of version 3's 13 names (`sm_core::protocol::Tool::ALL`) or version 1's 23 names (`V1_TOOL_NAMES`) |

- Cursor's form does not name the server. Another server's tool with one of these names is therefore skipped too. That is the conservative side: it avoids capturing our own output.
- `MCP:corpus_bridges` and `MCP:discover_connections` are not skipped, as in version 1, whose list of 23 names lacked them.
- Evidence: `s16_own_tools_are_never_captured`, and the `skip_rules` cases in `s16_content_extractor_cases`.

**G-5. The default `capture.tools` does not include MCP tools** (B-21: `Edit`, `Write`, `Bash`, `NotebookEdit`). (SPEC 15; SPEC 16 step 1)
- Version 2's MCP extraction is ported (`extract_mcp_tool`), but it is reached only when the user adds a tool to `capture.tools`.
- The `hook_payloads/` cases that captured an MCP tool in version 1 run with that payload's tool added to the settings. That ports version 1's behaviour.
- `at13_hook_payloads_default_settings` checks that the default settings give `ToolNotCaptured` for the same payload.
- The default was not changed to make a test pass.

**G-6. The ignore check runs on the raw path and on the normalised path.** (SPEC 16; B-21)
- A path is skipped when either form matches a `capture.ignore` pattern.
- The normalised path is the result of `sm_core::project::normalize_file_anchor`.
- Evidence: `s16_each_default_ignore_pattern_skips_edit_and_write` and `s16_ignore_patterns_fold_case_*`.

**G-7. Gate order.** (SPEC 16)
1. `capture.enabled`, then the project opt-out.
2. Then the event: PostToolUse, PreCompact or Stop. Anything else gives `NotACaptureEvent`.
3. For PostToolUse:
   1. The tool name.
   2. `captures_tool`.
   3. The skip rules.
   4. The ignore check.
   5. Extraction.
   6. The anchor, for Edit and Write only.
4. Each text then goes through these steps; each `TextDrop` reason names the step that dropped it:
   1. Empty.
   2. Classify (weak is dropped).
   3. Importance (a non-finite score is dropped).
   4. Redact (the skip rule drops it as `Secret`).
   5. Session dedupe, on the redacted text.
   6. Record the hash.

Evidence: `s16_gate_order`, and the pipeline cases in `s16_queue_and_pipeline_notes`, which replay version 1's call-order fakes.

**G-8. Stop.** (SPEC 16 event table; AT-14)
- The loop guard follows Python truthiness, so `stop_hook_active: "false"` counts as active. This is version 1's behaviour, recorded in the README's "must decide" list, and it is the conservative choice: when the guard is unclear, skip.
- The text is `last_assistant_message`. It is stripped with sm-core's `python_strip` and then cut at the 200,000-character cap.
- When the field is absent, Stop falls back to the transcript since the offset (section 7, D-1).
- The default trigger is `"session_end"`, as in version 1's Stop hook.
- Evidence:
  - `at14_stop_prefers_last_assistant_message`
  - `at14_stop_reads_the_transcript_when_there_is_no_final_message`
  - `s16_stop_loop_guard_and_missing_transcript`

**G-9. Which text the session dedupe compares.** (SPEC 16 step 4; AT-14; B-14)
- Both sides are the **redacted** text. The hash recorded is `content_hash` of the content that is stored, and the check hashes the same redacted form.
- This fixes version 1's mismatch, which compared pre-redaction transcript text with post-redaction queued text. Under that mismatch, a text holding a secret never matched its own copy.
- PostToolUse dedupes and records too. A tool call repeated within a session is captured once.
- Blank text is never a duplicate. Python's "whitespace matches the empty hash" is not kept (G-22).
- Evidence:
  - `at14_overlap_cases`
  - `probe.dedup_and_redact_get_raw_text` and the order cases (G-22)
  - `at14_pre_compact_returns_only_what_post_tool_use_missed`

**G-10. The session bound.** (SPEC 16 step 4)
- `SESSION_HASH_CAP = 4096` hashes for each session.
- When the state is full, the oldest hash is forgotten first.
- At 64 hexadecimal characters a hash, the state stays under about 600 KB a session.
- A text that falls out of the window can be captured again by this session; the store's B-29 dedupe then catches it.
- Evidence: `at14_session_state_bound_is_fifo`.

**G-11. NotebookEdit has no anchor.** (SPEC 16 "Type and provenance")
- NotebookEdit gets `provenance.file_path` (from `notebook_path`, normalised) but no `file` anchor.
- SPEC 16 anchors Edit and Write only.
- A notebook cell edit is not a file revision that `related` can follow.
- Version 1's queue omitted the notebook path from its context (G-22, `notebook_path_not_used`). Keeping it in provenance loses nothing.

**G-12. Provenance.** (SPEC 4; B-19; CONTRACTS 7)
- `ProvenanceSource::Capture`.
- `session_id`: the payload's session id.
- `tool`: for PostToolUse only.
- `file_path`: through `normalize_file_anchor`, relative to the project root when the path is inside it.
- `commit_sha`: the caller's `CaptureInput.commit_sha`. This crate reads no repository; the daemon looks the commit up.
- `extra`:
  - `hook`: sm-core's kebab-case event name, for example `"post-tool-use"`.
  - `tier`: 1 or 2.
  - `score`.
  - `patterns`.
  - `v1_namespace`.
  - `redactions`: when any.
  - `truncated`: true when the cap was hit.
  - `trigger`: PreCompact and Stop.
- Evidence: `s16_provenance_records_capture_with_session_tool_file_and_commit`.

**G-13. Tags and the anchor follow B-19, and every drop is recorded.** (B-19)
- Tags come from version 1's `derive_tags` (the tool name in lower case, then the families) and `derive_transcript_tags` (`transcript`, then the families).
- A tag that fails `sm_core::types::normalize_tags` is left out and listed in `dropped_tags` with the reason.
- A `file` anchor that fails B-19 is left out and recorded in `dropped_anchor`.
- Evidence: the `derive_tags_cases` and `derive_transcript_tags_cases` in `s16_queue_and_pipeline_notes`.

**G-14. Secret scanner.** (SPEC 9 "Security"; AT-9; CONTRACTS 9)

The 22 patterns and the kind each reports:

| Tier | Pattern → kind |
|---|---|
| 1 | `aws_access_key_id` → `aws_key`, `github_token` → `github_token`, `github_pat` → `github_token`, `stripe_key` → `stripe_key`, `slack_token` → `slack_token`, `slack_webhook` → `slack_webhook`, `openai_project_key` → `openai_key`, `openai_legacy_key` → `openai_key`, `google_api_key` → `google_key`, `sendgrid_key` → `sendgrid_key`, `pypi_token` → `pypi_token`, `npm_token` → `npm_token`, `anthropic_key` → `anthropic_key`, `huggingface_token` → `hf_token` |
| 2 | `private_key_pem` → `private_key`, `pgp_private_key_block` → `private_key`, `jwt` → `jwt`, `auth_header` → `auth_header`, `password_in_url` → `password_url` |
| 3 (value) | `password` → `password`, `secret` → `secret`, `token` → `token` (value group 1, after the false-positive filter) |

- **Markers.** Each match is replaced with `[REDACTED:<kind>]` through `sm_core::traits::redact_matches`.
- **Stability.** An existing `[REDACTED:...]` or Python `[REDACTED_...]` marker counts as already redacted, so `scan` of redacted text is empty and redacting twice changes nothing (AT-9).
- **Partial matches.**
  - `password_in_url` returns only the password's offsets; the scheme and user stay.
  - In a tier 3 match, only the value is replaced and the closing quote stays. Python dropped it (`password="[REDACTED_PASSWORD]` with no closing quote).
- **Bearer tokens.** `Authorization: Bearer <JWT>` (JSON Web Token) gives `Authorization: Bearer [REDACTED:jwt]`, as Python does.
- **Skip rule.** `should_skip` is version 1's rule: a private key was found, or more than half of the original characters were redacted. The pipeline drops such a text as `TextDrop::Secret`.
- Evidence: `at9_redaction_cases` (43/43) and `at9_redaction_patterns_table` (26/26).

**G-15. The transcript reader.** (SPEC 16 step 1; AT-14)
- The reader is stateless: `read_transcript(path, last_offset)` returns the entries and `new_offset`.
- Its doc comment records the B-29 note: after a daemon restart a re-read from 0 is caught by the store's content-hash dedupe.
- An offset past the end of the file (a truncated or replaced transcript) restarts from 0.
- Line handling:
  - CRLF (carriage return and line feed) and LF lines are both read.
  - A line over 1 MiB, a blank line, a line that is not UTF-8, and a line without the `"assistant"` marker are skipped. So are non-objects and sidechain entries.
  - The new offset is the end of the file as read, even when the last line was skipped.
- An empty entry list returns early with `new_offset`, so the daemon still advances.

**G-16. Transcript path and session identifier.** (SPEC 16; stream file "Transcript path")
- `validate_transcript_path` is version 2's string check: not empty, no `..`, absolute.
- It then adds two checks:
  - The file name is not a Windows device name (`CON`, `NUL`, `PRN`, `AUX`, `COM1-9`, `LPT1-9`, with or without an extension).
  - The path is a regular file (`metadata().is_file()`, symbolic links followed).
- The device-name check runs on every platform, not only Windows. It is the conservative side. A Claude Code transcript is named by its session UUID (universally unique identifier), so no real transcript is affected. On Linux, `is_file` already excludes device files.
- `sanitize_session_id` rejects a trailing newline, which Python's `$` accepted (G-22).
- Evidence: `s16_hook_helpers_cases_windows` and `_unix`.

**G-17. Where the cap applies.** (SPEC 16 "Index cap"; AT-17)
- One 200,000-character (code point) cap applies at extraction, with version 2's `...` marker.
- It applies to:
  - PostToolUse text.
  - `last_assistant_message`, after stripping.
  - The transcript texts together: a text that would pass the cap is cut and ends the list.
- `cap_hit` and `extracted_chars` (the size before the cap) are returned for the journal. `capture_size_tokens` needs the daemon's tokenizer.

**G-18. Non-string tool responses are written as JSON (JavaScript Object Notation).**
- When `tool_response` is an object or a list, it is written as JSON with `serde_json`, keys sorted, with Python's `", "` and `": "` separators.
- Python wrote it two ways. Version 2's content extractor used `str()`, giving `{'success': True}`. Version 2's dispatcher first re-serialised it with `json.dumps`, keeping insertion order.
- Version 3 has no Python dictionaries. `serde_json` keeps no insertion order without the `preserve_order` feature, which this stream did not turn on.
- Evidence: `non_string_tool_response_coerced` and `cc_post_tool_use_tool_response_object` (G-22).

**G-19. Inputs.**
- `CaptureInput` carries the project identity, the commit and the transcript offset besides the payload and settings, as the stream file asks.
- The event comes from `CaptureParams.event`, never from the payload's `hook_event_name`. This is version 2's routing (README "must decide": routing by the CLI argument).

**G-20. Two test-support additions in `fixtures/capture/rust/`.**
- `cursor_normalized_payloads.json` was generated by running version 2's own `_normalize_stdin`. The stream file says to use `normalize_stdin_cases` as the Cursor input, and those cases are used directly too (23/23). They cover small field sets only, so the full payloads needed the generated file.
- `at14_session.jsonl` is a hand-written transcript for AT-14.

**G-21. The fixture checker adds the operator `one_of`.**
- `hook_payloads/cc_post_tool_use_edit_fix_round_trip.json` uses it, but `fixtures/capture/README.md` does not list it.
- The checker implements it as "the value at `path` equals one of the listed values".
- The README should list it. That file is read-only to this stream, so it is recorded here.

**G-22. Results that satisfy `expect` but differ from `computed_*`.**

| File | Case | Version 3 | Python `computed_*` | Why |
|---|---|---|---|---|
| `overlap_cases.json` | `probe.whitespace_content_matches_empty_hash` | `false` | `true` | Blank text is never a duplicate (G-9). |
| `hook_helpers_cases.json` | `probe.trailing_newline_accepted` | `""` (rejected) | `"abc\n"` | A trailing newline is not part of a session id (G-16). |
| `content_extractor_cases.json` | `non_string_tool_response_coerced` | `Output: {"success": true}` | `Output: {'success': True}` | JSON, not Python `repr` (G-18). |
| `content_extractor_cases.json` | `master_edit_truncation`, `master_write_truncation`, `master_bash_large_output_truncated`, `master_total_cap_enforced` | the text whole, up to 200,000 characters | master's smaller per-tool and total caps | Version 2's single cap replaces master's (stream file; RULES 1.5: version 2 wins). |
| `transcript_cases.json` | `extractor_superseded.master_individual_entry_truncated` | lengths `[6000]` | `[5000]` | Version 2's single 200,000 cap; master's 5,000 cap for each entry is gone. |
| `transcript_cases.json` | `extractor_superseded.master_combined_budget_enforced` | `[3433, 3433, 3433, 3433]` | `[3433, 3433, 3134]` | Same: master's combined budget is gone. |
| `transcript_cases.json` | `extractor_superseded.master_partial_entry_at_budget_boundary` | `[5000, 6000]` | `[5000, 5000]` | Same. |
| `hook_payloads/` | `cc_post_tool_use_notebook_edit_decision` | the context has `file_path: /nb/analysis.ipynb` | no `file_path` | G-11. |
| `hook_payloads/` | `cc_post_tool_use_tool_response_object` | `Output: {"interrupted": false, "stderr": "", "stdout": ...}` (keys sorted) | `Output: {"stdout": ..., "stderr": "", "interrupted": false}` (`json.dumps`, insertion order) | G-18. |
| `queue_and_pipeline_notes.json` | `probe.tier_1_above_one_not_clamped` | 1.0 | 1.5 | Importance is clamped (G-2). |
| `queue_and_pipeline_notes.json` | `probe.tier_2_above_one` | 1.0 | 1.2000000000000002 | Clamped (G-2). |
| `queue_and_pipeline_notes.json` | `probe.score_above_one_passed_unclamped` | importance 1.0 in the write call | unclamped | Clamped (G-2). |
| `queue_and_pipeline_notes.json` | `probe.tier_3_fallback` | `null` (no candidate) | 0.7200000000000001 | Weak gives no candidate (G-2). |
| `queue_and_pipeline_notes.json` | `probe.tier_0_fallback` | `null` | 0.7200000000000001 | No tier 0 (G-2). |
| `queue_and_pipeline_notes.json` | `probe.tier_1_nan` | `null` | 0.7 | A non-finite score gives no candidate (G-2). |
| `queue_and_pipeline_notes.json` | `probe.tier_2_nan` | `null` | 0.4 | Same. |
| `queue_and_pipeline_notes.json` | `probe.notebook_path_not_used` | `file_path: "/n.ipynb"` | no `file_path` | G-11. |
| `queue_and_pipeline_notes.json` | `probe.file_path_not_validated` | `"/x"` | `"../x"` | Provenance paths go through `normalize_file_anchor` (G-12). |
| `queue_and_pipeline_notes.json` | `probe.metadata_full` | `file_path: "a.py"` | `"/p/a.py"` | Same: relative to the project root `/p`, on both platforms. |
| `queue_and_pipeline_notes.json` | `duplicate_skipped` | `redact` is called before the duplicate check | no `redact` call | Redact first, then dedupe on the redacted text (G-7, G-9). |
| `queue_and_pipeline_notes.json` | `should_skip_stops_processing` | no `is_duplicate` call | `is_duplicate` on the raw text | The skip rule runs before the dedupe (G-7). |
| `queue_and_pipeline_notes.json` | `clean_content_proceeds`, `client_default_is_claude_code`, `client_custom_passed` | `is_duplicate("clean…")`, the redacted text | the raw text | G-9. |
| `queue_and_pipeline_notes.json` | `multiple_texts_processed`, `empty_text_in_list_skipped`, `probe.python_strip_control_chars`, `probe.hashes_loaded_once_with_limit_50` | `is_duplicate` gets the redaction fake's output | the raw texts | G-9: the fakes' redactor returns a fixed text, so the dedupe sees it. |
| `queue_and_pipeline_notes.json` | `probe.dedup_and_redact_get_raw_text` | `is_duplicate("key [REDACTED] decided")` | `"key AKIA0000 decided"` | G-9: both sides compare redacted text. |
| `queue_and_pipeline_notes.json` | `probe.counters_mixed` | `redact` on t2 to t5; `is_duplicate` on t2, t4, t5 | `is_duplicate` on t2 to t5 | G-7: redaction runs before the dedupe, and t3, which the skip rule drops, is never deduped. |

**G-23. The README's "Python behaviours v3 must decide on".** (`fixtures/capture/README.md`)

| Behaviour | Version 3 |
|---|---|
| Invalid JSON on stdin | Stream D's; this crate takes a parsed `Value`. |
| Routing by the CLI event argument | Kept (G-19). |
| Cursor SessionStart `source` | Stream D's. |
| Stop loop guard truthiness | Kept (G-8). |
| Stop reads the transcript; `last_assistant_message` never read | `last_assistant_message` is preferred, with the transcript as the fallback (G-8, section 7 D-1). |
| Legacy entry points skip `validate_cwd` | Stream D's. |
| Cursor drive paths | Stream D's. |
| Dedupe compares raw text with redacted text | Fixed: both sides redacted (G-9). |
| `get_queued_hashes` raises on a non-object | Not applicable: queue files are replaced by the spool, and hashes by `SessionState`. |
| Queue writer cap, rate limit, directories, temporary files | Not applicable (B-25, B-38). |
| Importance unclamped, tier 0 like tier 2, NaN (not a number) becomes 0.0 | Clamped, no tier 0, NaN gives no candidate (G-2). |
| Fallback namespace `notes`; the fake uses `patterns or ["decision"]` | `notes` → `note` (G-3). The fake is reproduced only inside the pipeline-order tests. |
| 23 tool names, without `corpus_bridges` and `discover_connections` | Kept, and extended with version 3's 13 names (G-4). |
| Ambient recall parsing | Stream D's and the daemon's. |
| Lengths in code points | Kept everywhere (`chars()`). |
| `strip()` and `isspace()` include U+001C to U+001F | Kept for stripping: `sm_core::hash::python_strip` is used throughout, and `probe.python_strip_control_chars` passes. Not matched inside the regexes (section 9). |
| Error log rotation | Replaced by the journal. |

**A fixture case worth correcting.** This is recorded rather than edited, since the file is read-only.
- `hook_payloads/cc_post_tool_use_bash_private_key_skipped.json` comes from version 2's `test_private_key_causes_skip`.
- Its text carries no signal: both engines drop it as weak before the secret rule runs. So the case does not exercise the private-key skip it is named for.
- `at13_hook_payloads_default_settings` adds a strong variant ("The fix is to rotate this key."), and that variant is dropped as `Secret`.

## 7. Deviations

**D-1. Stop reads the transcript when `last_assistant_message` is absent.**
- The SPEC 16 event table says Stop "does not read the transcript".
- The stream file overrides it: fall back to the transcript, as version 1 did. Every Stop payload in the fixtures comes from version 1's tests and lacks the field.
- When the field is present, the transcript is not read.
- Evidence: `at14_stop_prefers_last_assistant_message` (the transcript is not read, and `new_offset` is `None`) and `at14_stop_reads_the_transcript_when_there_is_no_final_message`.

**D-2. The spool is not built here.**
- SPEC 14 (the phase 1 table: "sm-capture: extraction, transcript reader, signal classifier, secret scanner, spool") and the SPEC 3 crate table ("spool format") list the spool for this crate.
- CONTRACTS 8 makes the spool `sm_core::spool`, and CONTRACTS 7 makes the spool drain the daemon's (stream E). The stream file agrees with CONTRACTS.
- This crate does not touch the spool. The SPEC rows should be corrected to match CONTRACTS.

**Where RULES or CONTRACTS govern the stream file.** No rule in RULES or CONTRACTS contradicts the stream file. Where they add to it or decide a point, this stream followed them:
- **CONTRACTS 8 and 7**, over SPEC 14, for the spool (D-2).
- **RULES 1.5**, "where the SPEC is silent, version 2 wins over version 1". It decided:
  - the transcript caps, where master's caps are superseded (G-22);
  - the content-extractor caps;
  - the base of `validate_transcript_path`.
- **RULES 3** routes "record it in DECISIONS.md" to this report's Decisions. `DECISIONS.md` was not edited.
- **RULES 6** gives a crate-only Docker command. CLAUDE.md gives a workspace one. Both were run (section 3).
- **RULES 7** sets the fixture path and the `#[cfg(windows)]` / `#[cfg(unix)]` test pairs. Both are followed.

No other differences from SPEC or CONTRACTS.md.

## 8. Contract changes requested

None. Every type used comes from sm-core unchanged:
- `HookEvent`, `CaptureSettings` with `captures_tool` and `is_ignored`
- `ProjectIdentity`, `normalize_file_anchor`
- `content_hash`, `python_strip`
- `Provenance`, `NewMemory`, `AnchorRef`, `normalize_tags`
- `SecretScanner`, `SecretMatch`, `Redaction`, `redact_matches`
- `Tool::ALL`

## 9. Unfinished or unverified

- **Whitespace inside the regexes.** Python's `\s` includes U+001C to U+001F; the `regex` crate's `\s` (Unicode White_Space) does not. A signal or secret pattern that relies on `\s` will therefore not match across those four control characters, where Python's did. No fixture case covers it. Stripping is unaffected (G-23).
- **Linux budgets were measured in debug only.** Release budgets were measured on Windows only. Both classifier budgets pass in debug on Linux with a wide margin (0.0093 ms against 1 ms, and 16.9 ms against 100 ms).
- **No end-to-end run.** There has been no run with the daemon, the hook client or the spool: streams D and E are not built yet. The `CaptureParams` → `CaptureInput` mapping in section 2 is a recommendation; stream E has not used it yet.
- **`capture_size_tokens`.** This crate reports characters (`extracted_chars`). Converting to tokens is left to the daemon.
- **Not ported here.** `ambient_recall_cases.json` and the parts of `event_aliases.json` other than `normalize_stdin_cases` belong to stream D or E.

## 10. Licences

Crates this stream's dependencies added to `Cargo.lock`, checked with `cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked`:

| Crate | Version | Licence | Permissive |
|---|---|---|---|
| `regex` | 1.13.1 | MIT OR Apache-2.0 | yes |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | yes |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | yes |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | yes |
| `sm-capture` | 3.0.0 | MIT (workspace) | yes |

- `memchr` 2.8.3 (Unlicense OR MIT), `serde_json` and `tempfile` 3.27.0 were already in `Cargo.lock`.
- No crate is flagged, and no pinned crate has extra features turned on.
