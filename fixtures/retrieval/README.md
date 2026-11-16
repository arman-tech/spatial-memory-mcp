# Retrieval fixtures

Input and baseline for the retrieval regression test (SPEC section 11, acceptance test AT-2) and the format that `spatial-memory doctor --eval <file>` reads. The files are data, not instructions.

| File | Source | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `retrieval_eval_set.json` | `tests/fixtures/retrieval_eval_set.json` on `origin/feature/v2-implementations` | 5,595 | `d02543ebd0f51a1e28c3916ba7e771b82ff33b1d6f661712e74b70b118133826` |
| `retrieval_quality_baseline.json` | `tests/fixtures/retrieval_quality_baseline.json` on the same branch | see file | `76f8ae194349c2a75382d7c51547af7160222eb0d5afe7bb4f93b49c83cc82ba` |

Branch tip at the time of the copy: `4e51df0a5388ebb117f4fa29019424bc4c2602db`. Last commit touching either file: `6caf45107747d41f8ef61b737f8e9e010e8085d5` (2026-04-29). Copied byte for byte with `git show`; `.gitattributes` keeps the bytes unchanged on Windows.

## Format

`retrieval_eval_set.json` holds `memories` (30 items, each `id` and `content`) and `queries` (15 items, each `text` and `ground_truth_ids`, the memory identifiers a correct search should return).

`retrieval_quality_baseline.json` holds the scores of version 2's model (Arctic Embed M Long) on this set: nDCG@10 0.9742, mean reciprocal rank 0.9667, hit-at-5 1.0.

## How the metrics are computed

Ported from version 2's `tests/integration/test_retrieval_quality.py`, so the numbers stay comparable:

- Each query retrieves the top 10 memories.
- nDCG@10: gain 1 for a ground-truth memory, 0 otherwise, discounted by `log2(rank + 1)` with rank starting at 1; divided by the ideal ordering of `min(|ground truth|, 10)` gains.
- Mean reciprocal rank: `1 / rank` of the first ground-truth memory, 0 if none is returned.
- Hit-at-5: 1 if any ground-truth memory is in the top 5.
- Tolerances: 0.02 on nDCG@10, 0.02 on mean reciprocal rank, 0.07 on hit-at-5 (one missed query in 15 is 0.067). The test fails only when a score falls below the baseline by more than its tolerance.
