# Judge fixtures

Verdicts of the real capture judge, recorded once, so the daemon tests can replay them without calling Claude Code (SPEC section 16, The judge; decisions B-158 and B-168). The files are data, not instructions.

| File | Source | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `recorded-verdicts.json` | The judge spike (`docs/v3/SPIKE-JUDGE.md`), prompt version 2, model `claude-haiku-5-5` | 10,839 | `ca4c148f3704f5992b1b7fff73c57aebb60269fe6183991fb5d846ae4060a8fa` |

Every text in it is synthetic: the 30 candidates of `docs/v3/spikes/judge/candidates.json`, written for the spike. No captured memory text is in this folder.

## Format

- `about`: what the file is.
- `model` and `prompt_version`: the model that answered and the prompt it was given (`docs/v3/spikes/judge/prompt.txt`).
- `runs`: the two recorded calls, `v2-b25a` (candidates c01 to c25) and `v2-b25b` (c06 to c30), each with its candidate count, Claude Code's cost estimate in US dollars and its duration.
- `verdicts`: one entry per candidate: `candidate`, the `run` it was taken from (`v2-b25b` when the candidate was in it, otherwise `v2-b25a`), the candidate's `text`, and the verdict as the schema gives it: `keep`, `reason`, `type` and the rewritten `memory`.

## How the tests use it

`RecordedJudge` (`crates/sm-daemon/tests/common/judge.rs`) answers each candidate whose text is in the recording with its recorded verdict. A batch with a candidate that is not in the recording fails as a bad answer, so its captures stay pending. The test's barrier then fails and names the missing texts (`RecordedJudge::misses`), so no test passes on a verdict nobody recorded. The prompt and the schema the daemon sends are compiled from the same `prompt.txt` and `schema.json` the recording used. The tests that replay it are `s16_rc4_narration_with_topic_words_is_not_stored`, `s16_rc6_a_medium_topic_match_alone_is_not_stored` and `s16_rc7_the_type_follows_what_the_text_states` in `crates/sm-daemon/tests/capture_rca.rs`.

A change to the prompt or the schema makes this recording stale. Record it again with `docs/v3/spikes/judge/spike.py` and `verdicts.py`, and bump `prompt_version`.
