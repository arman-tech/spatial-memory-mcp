# A statement judge for 3.0: a proposal

**This is a proposal only.** Nothing is built. No product code, judge prompt, setting or test has changed, and nothing here is a decision; the owner accepts, changes or rejects it. The figures are B-239's: harness only, the design set (172 must keep, 639 groups) and the first hold-out (182 must keep, 631 groups), every configuration run twice. B-238 holds the ladder they start from.

## 1. What it proposes

Build R2 (B-238): haiku reads session batches of today's captures with the labellers' task and returns every statement worth keeping. Each statement it keeps becomes one memory, and every grade becomes active. A statement-level duplicate step sits between that call and the store. Today's judge stays available behind `capture.judge.task = "verdicts"`.

## 2. The figures it rests on

The two values in each cell are runs a and b. For the duplicate step they are its runs 1 and 2; both start from R2's run a. Junk is given two ways: rule M (B-177), and direct labelling of a fixed sample (two `claude-opus-5-5` labellers and an arbiter, narration or other, B-238). For "R2 with the step", the direct sample is the part of run a's 400-memory sample that the step kept. It is not a new sample.

| Configuration | Set | Active | Must keep | All groups | Stored twice | Junk, rule M | Junk, direct (items) | Cost a run (US dollars) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0, today's judge | design | 120, 135 | 66, 80 | 151, 193 | 44, 44 | 8.3%, 5.9% | 3.3% (all 120 of run a) | 0.21, 0.22 |
| R0 | hold-out | 170, 165 | 66, 63 | 173, 172 | 72, 68 | 14.1%, 15.8% | 4.7% (all 170 of run a) | 0.27, 0.26 |
| R2 as run | design | 1,094, 1,130 | 127, 131 | 398, 412 | 355, 371 | 38.9%, 38.5% | 8.2% (400) | 0.52, 0.55 |
| R2 | hold-out | 1,217, 1,191 | 124, 122 | 421, 400 | 426, 399 | 38.8%, 40.6% | 11.5% (400) | 0.57, 0.59 |
| R2 with the step (proposed) | design | 649, 637 | 106, 105 | 325, 323 | 79, 72 | 45.5%, 45.8% | 11.4%, 11.8% (236, 237) | 0.64, 0.65 |
| R2 with the step (proposed) | hold-out | 722, 720 | 105, 106 | 355, 357 | 102, 100 | 43.6%, 43.6% | 14.7%, 15.2% (238, 237) | 0.70, 0.69 |

The proposal holds more must-keep statements than today's judge: 105 to 106 against 66 to 80 on the design set (25 to 40 more), and 105 to 106 against 63 to 66 on the hold-out (about 40 more). It stores 4 to 5 times as many memories, and by direct labelling about 11% to 15% of them are junk, against 3% to 5% today.

## 3. A memory is one statement, not one capture

| R2 run a, b | Set | Active | Must keep | Stored twice | Type right | Junk, direct (400, run a) |
| --- | --- | --- | --- | --- | --- | --- |
| One memory a statement | design | 1,094, 1,130 | 127, 131 | 355, 371 | 73.4%, 69.4% | 8.2% |
| One memory a capture | design | 578, 595 | 130, 134 | 212, 214 | 66.2%, 60.6% | 11.5% |
| One memory a statement | hold-out | 1,217, 1,191 | 124, 122 | 426, 399 | 75.5%, 75.2% | 11.5% |
| One memory a capture | hold-out | 630, 619 | 126, 122 | 227, 212 | 64.5%, 67.2% | 14.2% |

- Must keep is the same either way, within the runs' noise.
- The type is right more often for statements: by 7 to 9 points on the design set and 8 to 11 on the hold-out.
- Junk by direct labelling is lower for statements, by about 3 points on both sets.
- A duplicate step on statements drops only the statement that repeats. A step on captures keeps or drops the whole capture.

**The alternative is to join the step's survivors per capture.** That gives 417 to 420 memories (design) and 469 to 472 (hold-out). Must keep stays the same (106 to 108), and the stored-twice share is about the same: 12% to 14%, against 11% to 14%. Its type and direct junk were not measured. I would still store statements, because that is the unit the labels, the type figures and the duplicate step measure.

## 4. Every grade becomes active; the grade is recorded, not used

- Storing only the statements haiku grades `must_keep` (policy b) halves must keep: 60 and 65 against 127 and 131 (design), and 65 and 62 against 124 and 122 (hold-out).
- Haiku's grade against the settled grade, for the statements that capture a labelled group:
  - agreement 75.4% and 75.3%, kappa 0.43 and 0.43 (design);
  - agreement 74.0% and 72.0%, kappa 0.34 and 0.30 (hold-out).

  Haiku grades `must_keep` for 111 of the 257 statements whose settled grade is must keep (design, run a).
- Haiku graded `must_keep` for 78 and 74 of the statements rule M calls junk (design), and for 94 and 83 (hold-out). In the direct samples it gave that grade to 3 of 33 junk statements (design) and 6 of 46 (hold-out).

So every statement the judge keeps is stored `active`. Its grade goes in the provenance's `extra.judge.grade`, and `importance` stays as it is today. Nothing here supports using the grade for recall ranking, pruning or a stricter policy.

## 5. The duplicate step

**Where it sits.** After the statements call of a batch, and before the batch is settled. Under the statement task it replaces B-192's repeats check, which works per capture at 0.70. The capture-time check at 0.93 (`CAPTURE_NEAR_DUPLICATE_COSINE`) stays.

**How it works, as measured:**
- Each kept statement is embedded. The same vectors become its chunks.
- Its candidates are the project's active memories and the batch's earlier kept statements, at cosine 0.80 or more, nearest first, at most 5.
- A statement with no candidate is stored without a call.
- One haiku call per batch, split at 40 statements or 60,000 characters, asks one question of each statement: does one of these state the same thing, and which. A statement with a match is not stored.
- **The answer schema is per position.** Statement *i*'s `match` is an enum of its own candidates' ids, or none.
  - The harness used one enum, the union of the call's ids, and haiku named the statement itself as its match.
  - After 3 tries, whole calls were left unchecked and their statements stored: 65 and 40 statements on the design set, 34 and 25 on the hold-out. This was not repaired.
- **A bad answer halves the call (B-206).** After `JUDGE_MAX_ATTEMPTS` its statements are stored unchecked, and they are counted.
- **B-190's automatic reject at 0.90 is off under the statement task.** Haiku is asked about every candidate at 0.80 or more.

| Policy, set | Statements asked | Dropped | Stored twice, before and after (rule M) | Must keep lost (rule M) | Drop pairs labelled directly: repeat, distinct, not worth keeping | Cost a statement (US dollars) |
| --- | --- | --- | --- | --- | --- | --- |
| (a) every statement, design | 717, 713 of 1,094 | 445, 457 | 355 to 79, 72 | 21, 22 | 88.9%, 91.7%; 9.9%, 7.1%; 1.2%, 1.2% | 0.000109, 0.000112 |
| (a), hold-out | 772, 773 of 1,217 | 495, 497 | 426 to 102, 100 | 19, 18 | 80.0%, 81.1%; 13.3%, 14.4%; 6.7%, 4.4% | 0.000102, 0.000096 |
| (b) must keep only, design | 120, 120 of 207 | 95, 94 | 66 to 2, 2 | 12, 12 | 90.5%, 90.4%; 8.4%, 8.5%; 1.1%, 1.1% | 0.000089, 0.000083 |
| (b), hold-out | 89, 89 of 210 | 67, 65 | 51 to 9, 11 | 8, 8 | 92.5%, 93.8%; 7.5%, 6.2%; 0%, 0% | 0.000073, 0.000074 |

- **By the nearest candidate's cosine.** At 0.90 or more, haiku dropped 92% to 100% of the statements it was asked about.
  - The design samples' drops there were 94 repeats, 0 distinct and 1 not worth keeping. The hold-out's were 53 repeats and 3 distinct.
  - An automatic reject at 0.90 would save about 30% of the questions answered: 204 of 652 in design run 1, 222 of 738 in hold-out run 1. That is a few cents a day.
  - It would also drop the statements at 0.90 or more that haiku kept: 11 and 9 in those runs. They were not labelled.

## 6. The queue

- **Session batches.** A batch takes the oldest waiting capture's session and that session's waiting captures in transcript order, up to 25 captures or 100,000 characters (today's `JUDGE_BATCH_MAX` and `JUDGE_BATCH_CHARS`). A capture with an attempt goes alone, as today (B-206).
- **The queue needs only the session.** `Store::judge_queue` joins the provenance for `session_id`, and `JudgeItem` gains `session`. The R2 prompt sends each capture's id, text and context, nothing more.
- No new table or column. Whether an index on the provenance's `session_id` is needed was not measured.

## 7. The store

- **No migration.** `SCHEMA_VERSION` stays 7.
- **A new `Settle::Statements`.** It holds the capture's id, the judge model, and per statement its content, title, type, chunks, grade and reason.
- **The rows.** The first stored statement takes the capture's row, from pending to active, as `Settle::Keep` does today. Each further statement is a new memory of the same project.
  - Its provenance is copied from the capture's.
  - Its `extra.judge` is `{model, task: "statements", capture, statement, grade}`.
- **Each statement** is redacted by the secret scanner and chunked, as `settle` does for a rewrite today. It has its own content-hash check (`Settled::Duplicate`).
- **A capture with nothing stored is rejected:** `nothing_kept` when the judge kept no statement, `repeats` when the step dropped every one. B-190's retention applies as today.
- `importance` is unchanged. Each capture is one transaction, as today.

## 8. The journal (sm-core: the lead's change, with a B-n entry)

- **A new `capture_statements_total`.** Its labels:
  - `outcome`: `stored`, `repeat` (the step), `duplicate` (haiku's own label), `unquoted` (span not found), `hash` or `unchecked`;
  - `grade`, `type` and `project`.
- **`capture_judged_total`** keeps counting captures, as `keep` or `reject`.
- **`capture_judge_cost_usd`** gains `call` values `statements` and `same`, so these calls can be told apart from today's verdict calls and B-192's `repeats`.
- **B-200's quality summary** counts statements under the statement task.

## 9. `status` and `doctor` (sm-core protocol: the lead's change)

- `JudgeStatus` gains `task`, `statements_today`, `repeats_today` and `unchecked_today`.
- `doctor` names the task in its judge line.

## 10. The switch

- **`capture.judge.task`** is `"statements"` (the default) or `"verdicts"`. `verdicts` is today's judge, unchanged:
  - prompt v10 and one verdict a capture;
  - B-190's reject at 0.90 and B-192's repeats check;
  - the queue's batches across sessions.
- **It is an sm-core change.** `JudgeSettings` is `deny_unknown_fields`, so the new field belongs to sm-core, and an older binary refuses a config that names it.
- **The daily caps stay at 600 calls and 3 dollars.**
  - Scaled to a typical day (B-193), the proposal costs 0.72 to 0.85 dollars (design) and 0.67 to 0.78 (hold-out), in about 130 to 160 calls.
  - The harness's cost per capture is about half B-203's live rate, so a live day may cost up to about 1.7 dollars.

## 11. The tests that would fail first

**New tests, named by B-40 (SPEC 16):**
- `s16_the_judge_task_is_statements_by_default` (sm-core config).
- `s16_the_queue_takes_one_session_in_transcript_order` (sm-store).
- `s16_settle_statements_stores_each_the_first_in_the_captures_row` (sm-store).
- `s16_a_capture_with_no_statement_stored_is_rejected` (sm-store): `nothing_kept` and `repeats`.
- `s16_a_statement_whose_span_is_not_found_is_not_stored` (sm-daemon).
- `s16_each_statement_is_redacted_and_chunked` (sm-daemon).
- `s16_the_same_schema_offers_each_statement_only_its_own_candidates` (sm-daemon).
- `s16_a_statement_repeating_an_active_memory_is_not_stored` (sm-daemon).
- `s16_a_statement_repeating_an_earlier_statement_of_its_batch_is_not_stored` (sm-daemon).
- `s16_a_statement_with_no_candidate_at_0_80_makes_no_call` (sm-daemon).
- `s16_a_bad_same_answer_halves_the_call_then_stores_unchecked` (sm-daemon).
- `s16_capture_statements_total_counts_each_outcome` (sm-core journal).
- `s16_status_names_the_task_and_counts_statements` (sm-cli ops).
- `s16_doctor_names_the_judge_task` (sm-cli doctor).

**Existing tests.** The verdict tests assume today's judge:
- sm-daemon `tests/judge.rs` (27 tests);
- the prompt and repeats tests in `src/judge/tests.rs` (39 tests);
- `capture_rca.rs` rc4, rc6 and rc7, the RCA cases that read a verdict.

They are set to `task = "verdicts"` and are otherwise unchanged; none is weakened. Statement versions of the RCA cases are new tests, not edits.

## 12. What the figures do not support

- **Retrieval.** Nothing here measures recall over 4 to 5 times as many, shorter memories.
- **A live store.** Each set starts from an empty store. Live candidates include the existing active memories, so the step will see more neighbours.
- **Live batching.** The 5-minute wait, sessions still open, and captures arriving between calls.
- **`tool_result` captures.** The sets include them, but they are off by default since B-236.
- **Using the grade** for anything.
- **Other settings for the step.** Floors other than 0.80, more or fewer than 5 candidates, and the per-position schema, which neither the harness nor the product has run.
- **Live cost and calls.**
- **Memory growth.** Scaled to a typical day's candidates, the proposal stores about 700 to 820 memories, against about 130 to 190 for today's judge.
- **The step adds no junk, but removes none.** By direct labelling about 11% to 15% of what is stored is junk. On a typical day's candidates that is about 80 to 120 junk memories, against about 4 to 9 for today's judge. The junk share rises after the step by both measures, because the step removes repeats, not junk.
- **Rule M's must keep lost.** Rule M overstates the must-keep statements the step loses. Of the sampled drops it counts as losing a must-keep group, direct pair labels call between a third and three quarters repeats. These samples are 5 to 14 drops a run.
- **Direct labelling is a sample:** 400 memories of run a, and at most 100 drop pairs a set and policy.
- **The other hold-outs.** h2 and h3 were not used.
