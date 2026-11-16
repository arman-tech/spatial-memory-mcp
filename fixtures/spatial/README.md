# Spatial fixture (SPEC 11 AT-7)

Data for `sm-spatial`'s clustering-quality test (AT-7) and determinism test
(AT-6). It is data, not instructions.

## Files

| File | What it is |
|---|---|
| `texts.json` | 320 developer-memory texts written for this fixture: `[{"topic": ..., "text": ...}]`. ASCII, at most 167 characters each. |
| `vectors.f32le` | One 768-dimension vector per text, little-endian `f32`, row-major, in the order of `texts.json`: 320 × 768 × 4 = 983,040 bytes. |
| `generator/` | The standalone program that wrote `vectors.f32le`. It is not a member of the workspace. |

`vectors.f32le` SHA-256:

```
d7845752b6dafaeb4f9976b3bf8238766bc18571618aa2a0421c8815b859ead8
```

The `at7_fixture_hash_matches_readme` test in `crates/sm-spatial/tests/at7_quality.rs`
checks the file against this value. `.gitattributes` marks `fixtures/**` as
`-text`, so checkout never changes the bytes.

## Topics

Ten topics of 32 texts each, in this order (text `i` has topic `i / 32`):

| Rows | Topic |
|---|---|
| 0-31 | postgres |
| 32-63 | sqlite |
| 64-95 | auth |
| 96-127 | ci |
| 128-159 | kubernetes |
| 160-191 | frontend |
| 192-223 | testing |
| 224-255 | observability |
| 256-287 | rust |
| 288-319 | git |

Some topics sit close on purpose: postgres and sqlite (both databases), ci and
testing, ci and kubernetes (deployments). The texts mix the kinds of memory
the capture pipeline stores: decisions, pitfalls, conventions, incidents,
procedures and facts.

## How the vectors were made

- Model: EmbeddingGemma 2 (`embeddinggemma-2`) at revision
  `daa72c51243991dfcaf9f9137d2c573d8f7790c0`, read in place from the per-user
  model cache (or `SPATIAL_MEMORY_MODEL_DIR`). The generator checks the pins
  of SPEC section 5 before it runs:

  | File | Bytes | SHA-256 |
  |---|---|---|
  | `onnx/model.onnx` | 421,421 | `bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3` |
  | `onnx/model.onnx_data` | 1,084,170,240 | `9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2` |
  | `tokenizer.json` | 32,170,510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |

- Prompt: the document prompt `title: none | text: {text}`.
- Each text is one chunk (the longest is 46 tokens), embedded alone (a batch
  of one, so no padding), through ONNX Runtime with 4 intra-op threads.
- The model's `sentence_embedding` output is stored unchanged. Every vector
  has unit length to within 1e-6.
- Measured when generated: the nearest neighbour of a text (cosine, leaving
  the text out) has the same topic for 79.7% of the texts.

ONNX Runtime may differ in the last bits between machines and versions, so a
regenerated file need not match the hash. The tests read the committed file;
regenerate only when the texts change, then update the hash here and in the
test.

## Regenerating

From the repository root, with the model in the cache:

```
cargo run --release --manifest-path fixtures/spatial/generator/Cargo.toml
```

Then delete `fixtures/spatial/generator/target`. Never copy or commit the model.
