# Migration fixtures

Test input for `spatial-memory migrate` (SPEC section 10, acceptance test AT-8). The files are data: tests parse them, and nothing in them is an instruction to follow.

| File | Source | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `v1_export_sample.json` | `tests/fixtures/v1_export_sample.json` on `origin/ralph/v2-implementation` (branch tip `0ed4b640b429cd06bd4acd1a8ae97494a16256dc`; last commit touching the file `1d3e1f2afe7ad5b4d797864aae5e05eadc3d2a02`, 2026-04-25) | 5,537 | `40bd2bb79801b253fb4b5d7f8eb297ab4c60785e6a3733a1d0769bfe1391776b` |
| `v1_export_full.json` | Written by `make_v1_export_full.py` in this folder (decision B-13) | 69,071 | `94e2bd016d33e517829e2cde7dd5efd93c5c3f277abd8975cab95efae4d860b9` |

The `.gitattributes` rule `fixtures/** -text` keeps the bytes unchanged on a Windows checkout.

## `v1_export_sample.json` (ported)

Copied byte for byte with `git show`. It is hand-written, not the output of version 1's exporter: a JSON array of 10 records with the keys `id` (`mem-0001` to `mem-0010`, not UUIDs), `content`, `namespace`, `tags`, `importance`, `source` (`v1-fixture`), `metadata` and `project`. No record has `vector`, `created_at`, `updated_at`, `last_accessed`, `access_count`, `content_hash` or `expires_at`.

## `v1_export_full.json` (written for version 3)

The ported sample cannot test timestamp parsing, hash copying or identifier handling, so this second file reproduces the exact byte format of version 1's JSON exporter (`master`, `spatial_memory/services/export_import.py`, `_export_json` and `_json_serializer`) with `include_vectors` at its default of true:

- A pretty-printed array: `[`, then each record as `json.dumps(record, indent=2)` with every line indented two more spaces, records joined by `,` and a newline, then `]`. No envelope or version field.
- All non-ASCII text escaped as `\uXXXX`; the emoji is a surrogate pair.
- `id`: version 4 UUID text.
- `vector`: 384 numbers per record, 32-bit floats written as Python floats. Migration drops it.
- `created_at`, `updated_at`, `last_accessed`, `expires_at`: naive ISO 8601 with no offset, values in UTC, with six fractional digits or none (`2025-09-01T00:00:00`). `expires_at` is null or a timestamp.
- `importance`: 32-bit float artefacts such as `0.699999988079071`.
- `content_hash`: SHA-256 hex of `content.strip().lower()` (version 1 `core/hashing.py`). Record 5 has surrounding whitespace, so the hash differs from a hash of the raw content.
- `project`: missing on record 3, as on rows written before version 1 added the column. Record 2's `namespace` equals its project name, so it must not become a tag.
- Record 4 holds Amazon's documented example access key (`AKIAIOSFODNN7EXAMPLE`) and secret, which migration must redact (SPEC 10 step 2, AT-9).
- Record 6 is Markdown with headings and a fenced code block.

Regenerate with `python -I make_v1_export_full.py` from this folder; the output is deterministic (fixed seeds, standard library only).
