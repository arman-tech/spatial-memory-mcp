# Judge fixtures

Verdicts of the real capture judge, recorded once, so the daemon tests can replay them without calling Claude Code (SPEC section 16, The judge; decisions B-158, B-168, B-173 and B-182). The files are data, not instructions.

| File | Source | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `recorded-verdicts.json` | The judge spike (`docs/v3/SPIKE-JUDGE.md`), prompt version 6, model `claude-haiku-5-5`, prompt caching off | 10,947 | `29e9bf9406375a8d8e3a54abcd15d35fd365e175a1d215026bd8f99eca292c3c` |

Every text in it is synthetic: the 30 candidates of `docs/v3/spikes/judge/candidates.json`, written for the spike. No captured memory text is in this folder.

## Format

- `about`: what the file is.
- `model` and `prompt_version`: the model that answered and the prompt it was given (`docs/v3/spikes/judge/prompt.txt`).
- `runs`: the two recorded calls, `v6-b25a` (candidates c01 to c25) and `v6-b25b` (c06 to c30), each with its candidate count, Claude Code's cost estimate in US dollars and its duration.
- `verdicts`: one entry per candidate: `candidate`, the `run` it was taken from (`v6-b25b` when the candidate was in it, otherwise `v6-b25a`), the candidate's `text`, and the verdict as the schema gives it: `keep`, `reason`, `type` and `quotes`, the sentences of the text the judge chose to keep. The daemon builds the memory from the sentences of the text that the quotes locate (B-173).

## How the tests use it

`RecordedJudge` (`crates/sm-daemon/tests/common/judge.rs`) answers each candidate whose text is in the recording with its recorded verdict. A batch with a candidate that is not in the recording fails as a bad answer, so its captures stay pending. The test's barrier then fails and names the missing texts (`RecordedJudge::misses`), so no test passes on a verdict nobody recorded. The prompt and the schema the daemon sends are compiled from the same `prompt.txt` and `schema.json` the recording used. The tests that replay it are `s16_rc4_narration_with_topic_words_is_not_stored`, `s16_rc6_a_medium_topic_match_alone_is_not_stored` and `s16_rc7_the_type_follows_what_the_text_states` in `crates/sm-daemon/tests/capture_rca.rs`.

A change to the prompt or the schema makes this recording stale. Record it again from `docs/v3/spikes/judge/`, with the new prompt version `N`:

```
py -3.13 -I spike.py vN-b25a 0 25 --extra-env DISABLE_PROMPT_CACHING=1
py -3.13 -I spike.py vN-b25b 5 25 --extra-env DISABLE_PROMPT_CACHING=1
py -3.13 -I verdicts.py vN-b25a vN-b25b
py -3.13 -I record.py ../../../../fixtures/judge/recorded-verdicts.json N vN-b25a vN-b25b
```

Then update the bytes and the SHA-256 above. `record.py` rebuilds the committed file byte for byte from the two version 6 summaries. The daemon runs the judge with prompt caching off (B-182), so the recording is made the same way. Version 6 was recorded on 2026-11-18: of the 30 synthetic candidates, the two calls agree with the spike's expected verdicts on 24 of 25 each. Call a typed c03 (a build failure, its cause and fix) an incident where the spike expected a pitfall; call b rejected c22 (a key-rotation procedure) where the spike expected a runbook. Call a also rejected c12 as a duplicate, but the recording takes c12 from call b, which kept it as a decision. The summaries and raw outputs are not committed; they are kept outside the repository.
