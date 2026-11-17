"""Compares judge verdicts with the expect field of the synthetic candidates (all synthetic text)."""
import json, sys, os
H = os.path.dirname(os.path.abspath(__file__))
cands = {c["id"]: c for c in json.load(open(os.path.join(H, "candidates.json"), encoding="utf-8"))}
for label in sys.argv[1:]:
    s = json.load(open(os.path.join(H, label + ".summary.json"), encoding="utf-8"))
    vs = (s.get("structured_output") or {}).get("verdicts", [])
    ok_keep = ok_detail = 0
    rows = []
    for v in vs:
        want_keep, want_detail = cands[v["id"]]["expect"].split(":")
        k = v["keep"] == (want_keep == "keep")
        d = (v["type"] if v["keep"] else v["reason"]) == want_detail
        ok_keep += k; ok_detail += k and d
        if not (k and d):
            rows.append(f"  {v['id']}: got keep={v['keep']} reason={v['reason']} type={v['type']}; expected {want_keep}:{want_detail}")
    ids = [v["id"] for v in vs]
    print(f"{label}: n={len(vs)} keep_ok={ok_keep} detail_ok={ok_detail} unique_ids={len(set(ids))==len(ids)}")
    for r in rows: print(r)
