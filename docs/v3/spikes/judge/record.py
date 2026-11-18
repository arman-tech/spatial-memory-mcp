"""Builds fixtures/judge/recorded-verdicts.json from two spike calls on the
synthetic candidates (B-168, B-173). Every text is synthetic.

usage: py -3.13 -I record.py <out.json> <prompt version> <run a> <run b>

<run a> is a spike.py label for candidates c01 to c25 and <run b> one for
c06 to c30, both saved next to this script. Where a candidate is in both
calls, the verdict of <run b> is kept. The run fails unless both calls
succeeded, every candidate has a verdict and one model answered.
"""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
out, version, run_a, run_b = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cands = {c["id"]: c for c in json.load(open(os.path.join(H, "candidates.json"), encoding="utf-8"))}
runs, verdicts, models = {}, {}, set()
for label in (run_a, run_b):
    s = json.load(open(os.path.join(H, label + ".summary.json"), encoding="utf-8"))
    r = s["result"]
    assert s["exit"] == 0 and not r.get("is_error"), label
    vs = s["structured_output"]["verdicts"]
    runs[label] = {"candidates": len(vs), "cost_usd": r["total_cost_usd"], "duration_ms": r["duration_ms"]}
    models.update((r.get("modelUsage") or {}).keys())
    for v in vs:
        verdicts[v["id"]] = {"candidate": v["id"], "run": label, "text": cands[v["id"]]["text"],
                             "keep": v["keep"], "reason": v["reason"], "type": v["type"], "quotes": v["quotes"]}
assert sorted(verdicts) == sorted(cands), "every candidate has a verdict"
assert len(models) == 1, models
doc = {
    "about": "Verdicts of the real judge on the synthetic candidates of docs/v3/spikes/judge/candidates.json, "
             f"recorded with prompt version {version} (docs/v3/SPIKE-JUDGE.md, B-173): the judge quotes the "
             f"candidate's sentences and never writes the memory. Each entry is the verdict of one recorded call: "
             f"{run_b} (c06 to c30) where the candidate was in it, else {run_a} (c01 to c25). "
             "Replayed by the daemon tests; every text is synthetic.",
    "model": models.pop(),
    "prompt_version": version,
    "runs": runs,
    "verdicts": [verdicts[k] for k in sorted(verdicts)],
}
open(out, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
print("verdicts", len(doc["verdicts"]), "model", doc["model"], "runs", runs)
