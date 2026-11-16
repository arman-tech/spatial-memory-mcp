# Stream G: `sm-capture`

## Goal

Build `crates/sm-capture`: the capture pipeline that the daemon runs on every capture hook. It extracts memory-worthy text from a tool call, from the payload's final assistant message, or from the transcript since a given offset; classifies it with version 1's signal patterns; redacts secrets; drops what this session already captured; and returns candidate memories with their type, importance, anchors, tags and provenance fields. It also implements `sm_core::traits::SecretScanner`, which `remember` (stream E), `share` and `migrate` (stream F) use too.

It is a library with no I/O beyond reading the transcript file a payload names (CONTRACTS 7). The daemon (stream E) owns the queue, the store, embedding, the spool drain and the journal.

**SPEC 14 done when:** "Version 1's hook test cases, ported as fixtures, pass against the new crate."

## You own

- `crates/sm-capture/` (new crate; depends on `sm-core` only, with `sm-core` `test-support` in `[dev-dependencies]`).
- `fixtures/capture/rust/` (new, optional): Rust forms of regexes or case annotations you add, with a README. The existing files in `fixtures/capture/` are read-only; record any case you believe is wrong in your report instead of editing it.

## Read

- **SPEC:** section 16 (all of it: the event table, "Pipeline in the daemon" steps 1 to 5, the rules, "Carried over from version 2"), section 9 "Security" (secret redaction on write; `share` refuses a file that still matches), section 4 (types, status, importance, provenance, anchors, tags), section 11 (AT-9, AT-13, AT-14, AT-16, AT-17's cap clause, test conventions), section 15 (the `capture.*` defaults).
- **CONTRACTS.md:** sections 2, 7 ("Stream G, `sm-capture`"), 8 (the spool belongs to `sm-core`, not to you), 9 (`SecretScanner`, `SecretMatch`, `Redaction`, `redact_matches`) and 12.
- **DECISIONS:** B-14 (`content_hash`), B-19 (provenance `tool`; tag and anchor validation), B-20 (project identity, `project.toml` opt-out, `normalize_file_anchor`), B-21 (`capture.ignore` and `capture.tools` defaults), B-24, B-25 and B-38 (the queue writer and queue processor are replaced by the spool; their cases do not apply to you), B-33, B-40.
- **Fixtures, read the README first:** `fixtures/capture/README.md` says how to read `expect` against `computed_*`, the generators, placeholders, the regex porting notes, and "Python behaviours v3 must decide on". Then `signal_patterns.json`, `signal_cases.json`, `redaction_patterns.json`, `redaction_cases.json`, `content_extractor_cases.json`, `transcript/transcript_cases.json` and the `.jsonl` files, `overlap_cases.json`, `hook_helpers_cases.json` (`sanitize_session_id_cases` and `validate_transcript_path_cases` are yours; `validate_cwd`, stdin, stdout, project root and the error log belong to stream D or are replaced by the journal), `queue_and_pipeline_notes.json` (the pipeline parts), and `hook_payloads/` with `INDEX.json`.
- **sm-core:** `traits` (`SecretScanner`, `SecretMatch`, `Redaction`, `redact_matches`), `types` (`MemoryType`, `Status`, `Origin`, `ProvenanceSource`, `Provenance`, `NewMemory`, `AnchorRef`, `AnchorKind`, the tag and anchor limits), `config` (`CaptureSettings`, `captures_tool`, `is_ignored`, `glob_match`), `project` (`normalize_file_anchor`, `path_key`), `hash::content_hash`, `protocol` (`HookEvent`, `CaptureParams`), `journal` (the capture metric names, for your report only; the daemon writes them).
- **Version 1 and 2 sources:** `git show master:spatial_memory/hooks/<module>.py` for signal detection, content extraction, redaction, overlap detection, the transcript reader and pipeline, and the per-event modules. `git show origin/feature/v2-implementations:<path>` where version 2 changed them (content extractor, transcript extractor). Port the behaviour and the patterns, not the structure.
- **Crate source:** `~/.cargo/registry/src/*/regex-1.13.1/`, for the syntax differences the fixture README lists.

## Build

1. **Signal classification** (SPEC 16 step 2; `signal_patterns.json`).
   - All 22 patterns, in evaluation order, case-insensitive with `(?i)`. Apply the README's porting rule: the 15 patterns that end in `(?:\.|$)` become `(?:\.|\n?$)`. Compile each once (for example in a `std::sync::LazyLock`).
   - A score of 0.8 or more is strong, 0.5 to 0.79 medium, anything lower is dropped.
   - Importance from the tier (SPEC 16 step 5): strong is the higher of 0.7 and the score; medium is the higher of 0.4 and 0.8 times the score. Clamp to 0 to 1; a non-finite score gives no candidate. Record where this differs from the Python `computed` values (the README notes `score_to_importance` is unclamped).
   - **Type from the pattern family** (SPEC 16 "Type and provenance"): each pattern maps to `decision`, `pitfall` or `convention`, and anything else is `note`. Start from version 1's `derive_namespace` (`queue_and_pipeline_notes.json` `derive_namespace_cases`), whose namespaces become types. Write the full table of 22 patterns to types in your report.
2. **Extraction** (SPEC 16 step 1; "Carried over from version 2").
   - **PostToolUse:** version 2's content extractor per tool (`content_extractor_cases.json` `contract_v2`), one safety cap of 200,000 characters (code points) at extraction with version 2's marker, and the text stored whole after that. The tool must pass `CaptureSettings::captures_tool`, and the skip rules (`skip_rules`) apply after that. A path that matches `CaptureSettings::is_ignored` is not captured. Decide and record how this crate's own tools are recognised and skipped under version 3's 13 tool names, in both the Claude Code (`mcp__spatial-memory__*`) and Cursor forms.
   - **Stop:** the text is the payload's `last_assistant_message` (SPEC 16 event table). When the field is absent, which is true of every Stop payload in the fixtures because they come from version 1's tests, fall back to the transcript since the offset, as version 1 did. Record the fallback.
   - **PreCompact:** assistant text from the transcript since the offset (`transcript_cases.json`): JSON lines, a line that is not valid UTF-8 skipped, CRLF accepted, version 2's 200,000-character caps with its marker.
   - **Transcript offsets.** The reader takes a starting byte offset and returns the new one; it holds no state. The daemon keeps the offset per session. If the daemon restarts and reads from 0, step 4's content-hash dedupe against existing memories catches what was already stored; say so in a code comment.
   - **Transcript path.** Validate it as version 2 did (`hook_helpers_cases.json` path validation): a regular file, not a device name on Windows. Reading it is the only I/O this crate does.
   - Cursor payloads reach you already normalised to Claude Code's field names by the hook client. In tests, use the normalised outputs recorded in `event_aliases.json` `normalize_stdin` as your Cursor inputs; do not reimplement the normalisation.
3. **Secret scanner** (SPEC 9 "Security"; `redaction_patterns.json`).
   - Implement `SecretScanner::scan` with every tier-1, tier-2 and tier-3 pattern and the false-positive filter, returning byte offsets on `char` boundaries. Use the file's `v3_mapping` for `kind`.
   - Where a Python pattern replaces only part of its match (for example `password_in_url`, which keeps the scheme and user), return the offsets of the secret part only, so `redact_matches` keeps the rest. Record each pattern where the redacted text differs from Python's `computed` text.
   - Redacted text must be stable: `scan` finds nothing in `redact(text).text`, and redacting twice gives the same text. Stream F's `share` refuses a file that still matches, so a scanner that matches its own `[REDACTED:<kind>]` marker would block every share.
4. **Overlap and session dedupe** (SPEC 16 step 4; `overlap_cases.json`).
   - A value type the daemon keeps per session that remembers the `content_hash` (B-14) of every item captured in that session, so transcript text that PostToolUse already captured is not captured again (AT-14).
   - Compare like with like. Version 1 compared pre-redaction transcript text with post-redaction queued text, so a text holding a secret never matched (README, "Dedup"). Choose one side, apply it to both, and record it.
   - Dedupe against memories already in the store is the store's job (B-29); you only report each candidate's `content_hash`.
   - Bound the per-session state, for example with a cap on the number of hashes kept; record the bound.
5. **Candidates** (CONTRACTS 7). For one normalised payload, settings and the session state, return zero or more candidates. Each carries:
   - the redacted content and its `content_hash`;
   - the type, the importance, the signal score and tier, and the matched pattern names;
   - the redaction kinds;
   - tags from version 1's `derive_tags` and `derive_transcript_tags` (`derive_tags_cases`, `derive_transcript_tags_cases`), dropping any that fail B-19's validation, and recording each drop;
   - provenance fields: source `capture`, the session id, the tool, and the file path. The daemon adds the commit and the project, because it owns the repository lookup;
   - for Edit and Write, a `file` anchor made with `sm_core::project::normalize_file_anchor` against the project root the caller passes (SPEC 16). Decide whether NotebookEdit is anchored and record it.
   Return candidates in a form the daemon can turn into `NewMemory` without guessing. Stream E builds against your public API after you merge, so document every public type and function, and list the API in your report.
6. **Controls.** `capture.enabled` off returns nothing. The repository opt-out in `.spatial-memory/project.toml` is resolved by `sm_core::project` (B-20); take it as an input, do not read the file yourself.

## Acceptance tests and other tests

Load each fixture file with `serde_json` and run every case through one small checker that implements the README's `{path, op, value}` operations and the `$concat`, `$repeat` and `$float` generators. Give each fixture file its own test, named for its rule, that runs all its cases and reports every failing case id together, not only the first.

- **Fixture coverage, the done-when.** For each file: cases ported, passing, and not applicable, each not-applicable case with its reason. Not applicable means Python-only (`$py_literal`), a module replaced by the spool (B-25, B-38: the queue writer's file naming, rate limit and caps), or stream D's (stdin, stdout, aliases). Where your result satisfies `expect` but differs from a `computed_*` value, list it under Decisions. Nothing is dropped silently.
- **AT-13** (library part): the Edit, Write and Bash payloads in `hook_payloads/` give a candidate when the text carries a strong signal, nothing when the signal is weak, and the fake credential in a payload is redacted with its kind reported.
- **AT-14** (library part): with a fixture transcript and session state that already holds PostToolUse's captures, PreCompact and Stop return the assistant text PostToolUse missed and not what it already captured. Stop prefers `last_assistant_message` when the payload has it.
- **AT-16** (library part): a medium-signal text gives an active candidate with importance at least 0.4; a strong one at least 0.7. The candidate's status is `active`.
- **AT-9** (library part): every case in `redaction_cases.json`; plus, over all of them, `scan(redact(x).text)` is empty and redacting twice changes nothing.
- **AT-17** (library part): a capture longer than 200,000 characters is cut at extraction with the marker. The index cap itself is the chunker's (stream B); do not duplicate it.
- **SPEC 16 rules**: `capture.enabled` off; a tool not in `capture.tools`; each default ignore pattern of B-21 against Edit and Write paths; anchors for Edit and Write; provenance source `capture`; types from pattern families.
- **Budget**: `signal_cases.json` has two performance budgets; meet them on a release build. Also measure and report the time to run the whole pipeline on a 200,000-character Bash output and on a 1 MB transcript delta, release build. There is no SPEC 11 budget for this path; the daemon runs it after it has replied `accepted`.

Run the tests on Linux in Docker as well: path validation and the Windows device-name rule differ by platform.
