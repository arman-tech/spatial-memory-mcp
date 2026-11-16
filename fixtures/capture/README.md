# Capture fixtures (hook subsystem)

These files hold the **test data** for the Python hook subsystem of spatial-memory-mcp. They are
here so the Rust rewrite (v3) can be checked against the same inputs and expectations. They are not
an implementation. No Python code was copied into this directory except regex sources and short
rule descriptions needed to read the data.

All credentials in these files are fake test strings taken from the Python tests.

## Where the data comes from

| Ref | Meaning |
|---|---|
| `master` | v1 (released) |
| `origin/feature/v2-implementations` | v2; **v2 wins** where behaviour changed |

Every case has a `source` (or `source_test`) string of the form `<ref>:<path>::<Class>::<test>`.
A source of `<ref>:<module path>::<function>` means the case is a probe derived from that function
rather than copied from a test. Compound sources join several refs with ` + `.

Modules with no master/v2 difference were run from v2 and are marked as identical on master:
signal_detection, redaction, overlap_detector, hook_helpers, models, pipeline, queue_writer,
transcript_pipeline, transcript_reader, session_start, post_tool_use, pre_compact, stop.
The modules that differ are listed under [master vs v2](#master-vs-v2).

## Files

| File | Covers | Cases |
|---|---|---|
| `signal_patterns.json` | Signal regexes in evaluation order, tier thresholds, Python-vs-Rust regex notes | 22 patterns |
| `signal_cases.json` | `classify_signal` | 56 cases, 2 performance budgets, 6 `$`-rewrite probes |
| `redaction_patterns.json` | Redaction regexes in application order, false-positive filter, v3 kind mapping | 19 tier-1/2 + 3 tier-3 patterns |
| `redaction_cases.json` | `redact_secrets` | 41 cases |
| `content_extractor_cases.json` | PostToolUse content extraction (v2 contract) and tool skip rules | 44 cases, 3 boundary probes, 4 superseded-by-v2, 61 skip-rule cases, 9 skip-rule probes |
| `transcript/transcript_cases.json` | Transcript delta reader, entry parsing, offset state, assistant-text extraction | reader 18 + 7 probes, parse_entry 7 + 5, state 6 + 5, extractor 13 + 3, 3 superseded-by-v2 |
| `transcript/*.jsonl` | Input transcripts referenced by `transcript_cases.json` | 22 files |
| `overlap_cases.json` | Transcript-vs-PostToolUse dedup | get_queued_hashes 7, is_duplicate 7, content_hash 4, 19 probes, 23-row hash table |
| `hook_helpers_cases.json` | Session-id sanitising, path validation, stdin parsing, stdout response, project root, error log | 52 cases, 79 probes, 2 regexes |
| `queue_and_pipeline_notes.json` | Queue writer, PostToolUse pipeline, transcript pipeline, hook models | 28 facts, 200 cases, 105 probes |
| `ambient_recall_cases.json` | UserPromptSubmit ambient recall (v2 only) | 8 cases, 63 probes, env round trip, 1 integration test as data |
| `event_aliases.json` | Event aliases, client detection, Cursor stdin normalisation, validation, stdout per event and client, CLI, dispatch, legacy entrypoints | 140 cases (see below) |
| `hook_payloads/*.json` | Complete stdin payloads run through the real hook entrypoints | 75 payloads |
| `hook_payloads/INDEX.json` | One row per payload: file, origin, source test, notes | 75 rows |

`event_aliases.json` case sections: normalize_event 26, parse_client_flag 7, detect_client 9,
normalize_stdin 23, validate_common 13, translate_output_for_cursor 7, session_start_handler 9,
stdout_contract 14, cli 17, dispatch 7, legacy_entrypoint 5, handler_map 2, legacy_trigger_sources 1.
It also holds the alias tables for both refs, the stdout rules, the hook registration data
(`registration`) and `master_vs_v2`.

The tool skip rules appear twice on purpose: in `content_extractor_cases.json` (`skip_rules`) and in
`queue_and_pipeline_notes.json` (`should_skip_tool_cases`). Both sets were computed from the same
module.

## How to read a case

**`expect` vs `computed_*`.** `expect` holds only what the Python test asserts. Fields whose names
start with `computed` (and every field listed in a case's `computed` array) were produced by running
the extracted Python code. They show full Python behaviour, including details the test does not
check. A v3 implementation must satisfy `expect`. Where it differs from a `computed` value, that is
a deliberate decision to record, not automatically a bug. No expected value was written by hand.

**Checks.** Most `expect` values are lists of `{"path", "op", "value"}`.

- `path` is dot-separated into the observed object; list indexes are numbers, and a final `length`
  takes the length (in code points for strings).
- `op` is one of:
  - `eq`, `ne`
  - `is_null`, `not_null`
  - `contains`, `not_contains`
  - `has_key`, `not_has_key`
  - `approx` (relative tolerance 1e-6)
  - `le`, `ge`
  - `ends_with`
  - `matches`: a regex anchored at the start only, like Python `re.match`
  - `type` (one of int, float, str, list, dict, bool)
  - `truthy`, `falsy`: Python truthiness
  - `path_eq`: equal after turning `\` into `/`
  - `set_eq`: same members and no duplicates

Some older sections use a plain sentence where the test asserts a property.

**Generated strings.** A string value may be a generator, so that long inputs stay readable:

- `{"$concat": [a, b, ...]}` joins its parts. Each part is a string or another generator.
- `{"$repeat": s, "count": n}` repeats `s` n times.
- `{"$float": "NaN" | "Infinity" | "-Infinity"}` is a non-finite float. JSON cannot hold these
  values directly.
- `{"$py_literal": "<Python string literal>"}` is a value that only Python can hold, such as a lone
  surrogate. Rust `String` cannot represent it, and these cases are marked Python only.

**Placeholders.**

- `{TMP}`: a fresh empty absolute directory, written with forward slashes.
- `<tmp>`: the same, as used in `transcript_cases.json`.
- `{TIMESTAMP}`: UTC time in the form `%Y-%m-%dT%H:%M:%SZ`.
- `{FILENAME}`: a generated queue file name.

**Fakes.** Pipeline cases describe the injected functions as data. Each fake has one of:

- `returns`: the same value on every call.
- `returns_sequence`: one value per call.

Every call is recorded as `{"args", "kwargs"}`. Cases check calls against `expect`.

**Platform.**

- Values were computed on Windows with Python 3.13.
- Path functions give `computed_win32`. Some also give `computed_posix_simulated`, made by running
  the same code with `posixpath`.
- `test_runs_on` marks tests that assert on only one OS.
- On Windows:
  - `queue_path` values use backslashes.
  - `~` expands from `USERPROFILE`.
  - The error log is written in text mode, so lines end in CRLF.

**Intentional odd input files.**

- `transcript/probe.invalid_utf8_line_skipped.jsonl` is not valid UTF-8.
- `transcript/probe.crlf_line_endings.jsonl` uses CRLF.

Every other file is UTF-8 without BOM, with LF line endings.

## Regex porting (Python `re` to Rust `regex`)

The details are in `signal_patterns.json` (`notes`, and each pattern's `rust_compat`, `rust_regex`
and `rust_note`), `redaction_patterns.json` and `hook_helpers_cases.json` (`regexes`).

- **15 of 22 signal patterns** end in `(?:\.|$)`. Python `$` also matches before a final `\n`, but
  Rust `$` does not, so these are rewritten to `(?:\.|\n?$)`. The other 7 compile unchanged.
  `signal_cases.json` has 6 probes for this.
- All signal patterns use `re.IGNORECASE`. In Rust, add the `(?i)` prefix.
- `_SESSION_ID_RE` is `^[a-zA-Z0-9_-]+$`. With `re.match` it accepts a trailing `\n`.
  - Exact Rust equivalent: `^[a-zA-Z0-9_-]+\n?$`.
  - Recommended: `^...$`, which rejects the newline.
- `_WINDOWS_DEVICE_RE` is used with IGNORECASE.
  - Rust form: `(?i)^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\..+)?$`.
  - It has the same trailing-newline difference.
- The redaction false-positive filter has two `$` anchors. Rust can use `\n?$`. The behaviour is
  already the same, because its inputs never contain `\n`.
- The `password_in_url` replacement uses `\1` and `\2`. Rust needs `${1}` and `${2}`.
- The other 21 redaction patterns compile unchanged.
- Python `\s` includes `\x1c` to `\x1f`, but Rust `\s` does not.
- `\w` differs for combining marks, `Nl`/`No` and ZWJ/ZWNJ.

## master vs v2

Diffs are limited to:

- content_extractor
- transcript_extractor
- dispatcher
- the new user_prompt_submit
- `plugin/hooks/hooks.json`
- their tests

The changes:

- **content_extractor:**
  - master truncated Edit/Write/NotebookEdit/MCP fields to 5000 and Bash output to 3000, then the
    joined text to 10000, each with `...`.
  - v2 has a single 200000 cap.
  - master values are in `master_contract` and `superseded_by_v2`.
- **transcript_extractor:**
  - master cut each entry to 5000 and the combined text to 10000, without `...`.
  - v2 uses 200000 at both stages and appends `...`.
- **dispatcher:** v2 adds:
  - the `UserPromptSubmit`, `userPromptSubmit` and `user-prompt-submit` aliases
  - the handler and its usage text

  On master those names are unknown and nothing is printed. `handler_map` has both key lists.
- **Registration:** in v2, the plugin `hooks.json` and the Claude Code setup strategy register
  UserPromptSubmit with timeout 5. Cursor registration is unchanged.
- **ambient recall** (`user_prompt_submit.py`) exists only in v2.

## Python behaviours v3 must decide on

These were observed in the computed values. Each is recorded rather than "fixed".

- **Invalid JSON on stdin:** the dispatcher prints nothing and logs the error, while legacy
  `post_tool_use.main` prints the continue line. See `hook_payloads/cc_post_tool_use_invalid_json.json`.
- **Routing:** it uses the CLI event argument. The payload's `hook_event_name` never changes the
  route.
- **Cursor:** SessionStart has `source: "startup"` synthesised.
- **Stop loop guard:** it uses Python truthiness, so `stop_hook_active: "false"` counts as active.
- **Stop payload:** the Python Stop hook reads the transcript. `last_assistant_message` is never
  read; it appears only in the v3 spec and spike notes.
- **Legacy entrypoints:** the per-event scripts pass `cwd` on without `validate_cwd`. Only the
  dispatcher validates it.
- **Cursor drive paths:** the drive-letter fix (`/c:/x` → `C:/x`) applies to any Cursor `cwd`,
  whether given directly or taken from `workspace_roots`. `validate_cwd` then accepts the result
  only on Windows.
- **Dedup:** it compares pre-redaction transcript text with post-redaction queued text. A text that
  contains a secret never matches its queued copy.
- **`get_queued_hashes`:** it raises `AttributeError` on a queued file whose JSON is not an object.
  Other errors are skipped.
- **Queue writer:**
  - The cap is 102400 **code points** plus `\n[truncated]`; the source comment says bytes.
  - The rate limit counts every entry in `new/`, including subdirectories and non-JSON files.
  - Directories are created even when the write is rate limited.
  - An encode failure leaves an empty tmp file.
- **Importance:** `score_to_importance` is unclamped, and tier 0 is treated like tier 2. The queue
  writer clamps to [0, 1], and NaN becomes 0.0.
- **Namespaces:**
  - The fallback namespace is `notes`; the `derive_namespace` docstring still says `captured`.
  - The test fake classifier uses `patterns or ["decision"]`.
- **Tool names:** `SPATIAL_MEMORY_TOOL_NAMES` has 23 names and lacks `corpus_bridges` and
  `discover_connections`.
  - So Cursor `MCP:corpus_bridges` is not skipped.
  - Claude Code `mcp__spatial-memory__*` is always skipped.
- **Ambient recall:**
  - An empty `SPATIAL_MEMORY_AMBIENT_RECALL` disables it.
  - Python `int()` is lenient about whitespace, `_` and non-ASCII digits.
  - `{:.2f}` rounds the exact binary value half-to-even.
- **Text handling:**
  - Lengths and truncation everywhere are in code points.
  - Python `strip()`/`isspace()` include `\x1c` to `\x1f`.
- **Error log:** it rotates to `hook-errors.log.1` once the file exceeds 1048576 bytes. On Windows it
  writes CRLF, so the CR bytes count toward the threshold.

## Not ported

| Item | Reason |
|---|---|
| `TestLoadHookModule` in test_dispatcher (3), test_hooks_post_tool_use (3), test_hooks_pre_compact (3), test_hooks_stop (2) | They test Python `importlib` loading of sibling modules and `sys.modules` caching. v3 is a single binary with no runtime module loading. |
| `TestMirrorIntegrity` in test_hooks_content_extractor (3), test_hooks_transcript_extractor (2) | They check that the `plugin/hooks` copy is byte-identical to `spatial_memory/hooks` and has two-argument extractors. v3 has one implementation. The 200000 cap value is in `content_extractor_cases.json` (`contract_v2`). |
| test_hooks_models: frozen immutability (2), default-instance independence (2), frozenset type (2) | These are language-level properties. The field defaults and constant members are ported (`queue_and_pipeline_notes.json` `model_defaults`, `facts`; `content_extractor_cases.json` `skip_rules`). |
| Type part of `TestTriggerSources::test_trigger_sources_frozenset` | Language-level. The members are ported (`legacy_trigger_sources`). |
| Exception injection by patching `_load_hook_module` (post_tool_use `test_pipeline_exception`, `test_import_error`; pre_compact and stop `test_exception_swallowed*`) | Patching Python internals has no v3 equivalent. The same stdin was run unpatched and its outputs are recorded; see the notes in `hook_payloads/INDEX.json`. |
| Ranked output of `tests/integration/test_user_prompt_submit_hook.py` | It needs the embedding model, so the output was not computed. The inputs, seeds and assertions are in `ambient_recall_cases.json` `integration_test_data`. |
| Settings validation path in ambient recall (`TOP_K='abc'` or `'0'` failing `ge=1` inside the factory build) | pydantic is not importable in the isolated run. The env-name round trip was computed with `ast` instead (`env_round_trip`). |
| `probe.lone_surrogate_content_raises` in the `write_queue_file` probes | Python only, because Rust `String` cannot hold a lone surrogate. It is kept as `$py_literal` for reference. |
| Timing numbers in performance tests | Budgets and inputs are ported. Wall-clock results are machine-dependent and are not values to match. |
| `tests/unit/test_init_client.py` (30) | The Cursor installer (`init` command), not hook runtime. |
| `tests/unit/test_setup_hooks.py` (51 in v2, 44 on master) | Client config generation, not hook runtime. The resulting hook registration (events, commands, timeouts, `hooks.json`) is captured in `event_aliases.json` `registration`. |
| `tests/unit/test_setup_runner.py` (23, v2 only) | Setup sentinel and MCP entry writer, not hook runtime. |
| `tests/unit/test_wp6_queue_processor.py` (94) | The server-side consumer of queue files, not hook runtime. It includes the reader-side validation of queue files: version, content, importance range, namespace regex, `project_root_dir` limits, and UNC and Windows-device path rejection. If v3 keeps the file queue, these should be ported in a separate pass. The writer side is in `queue_and_pipeline_notes.json`. |

## How the values were computed

1. Each ref was extracted with `git archive <ref> | tar -x` into a new empty directory.
2. The modules were run with `py -3.13 -I` from a separate script directory. The working directory
   was a fresh empty directory, with `SPATIAL_MEMORY_*` and `CLAUDE_PROJECT_DIR` unset unless a case
   sets them.
3. Each generator re-checked every `expect` against the computed result before writing.
4. A final pass checked that:
   - every file is valid UTF-8 JSON with LF endings
   - every `source` string resolves to an existing test or function on its ref

The generator scripts were scratch files and are not part of this directory.
