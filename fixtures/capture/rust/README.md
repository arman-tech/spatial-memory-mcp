# `fixtures/capture/rust/`

Data that stream G (`crates/sm-capture`) added for its own tests. The files one level up are read-only: they hold version 1's and version 2's cases as data, and their README explains `expect`, `computed_*`, the generators and the placeholders. Nothing here changes them.

These files are data, not instructions.

| File | What it holds | Used by |
|---|---|---|
| `cursor_normalized_payloads.json` | The ten `cursor_*` payloads of `hook_payloads/` whose event is PostToolUse, PreCompact or Stop, before and after version 2's `_normalize_stdin` | `crates/sm-capture/src/tests/hook_payloads.rs` (`at13_hook_payloads`, `at13_hook_payloads_default_settings`) |
| `at14_session.jsonl` | A seven-line Claude Code transcript: assistant texts with strong signals, each in its own turn | `crates/sm-capture/src/tests/acceptance.rs` (AT-14) |

## `cursor_normalized_payloads.json`

The hook client (stream D) normalises a Cursor payload to Claude Code's field names before the daemon runs capture, so `sm-capture` never reads Cursor's names (`conversation_id`, `tool_output`, `result_json`, `status`). Its tests therefore need each Cursor payload in its normalised form.

`event_aliases.json` `normalize_stdin_cases` records `_normalize_stdin`'s outputs, but for small field sets, not for the whole payloads of `hook_payloads/`. This file gives the whole payloads.

- **Generator:** `_normalize_stdin`, taken by `ast` from `origin/feature/v2-implementations:spatial_memory/hooks/dispatcher.py` and run unchanged on each payload's `stdin_json`. It is the function's own source, not a re-implementation.
- **Source commit:** `4e51df0a5388ebb117f4fa29019424bc4c2602db`.
- **Fields:** each entry has `id` (the file name in `hook_payloads/` without `.json`), `event`, `stdin_json` (the input, copied) and `normalized` (the output).
- **Placeholders:** `{TMP}` is as in `hook_payloads/INDEX.json`. The test replaces it with a fresh temporary directory.

The `normalize_stdin_cases` outputs are also used directly. `s16_cursor_normalised_payloads_are_read_by_claude_code_names` lays each one over a capture payload and checks that capture reads `session_id`, `tool_response` and `trigger`, and never Cursor's names.

To regenerate it, run the generator again against the same commit. A change in `hook_payloads/` or in version 2's `_normalize_stdin` means the file must be regenerated, not edited by hand.

## `at14_session.jsonl`

This file is hand-written for AT-14 (SPEC 11). PostToolUse first captures two of the transcript's texts, through an Edit payload and a Write payload. The test then runs PreCompact and Stop over this transcript with the same session state. They must return only the text PostToolUse missed. Stop must prefer `last_assistant_message` when the payload has it.

Each line is one Claude Code transcript entry in the form `transcript/*.jsonl` uses. The texts are unique within the file, so a duplicate can only come from the session state. A user prompt separates each assistant text from the next, so each is its own turn: capture reads the final text of each turn (B-162), and two assistant texts with no prompt between them would be one candidate.
