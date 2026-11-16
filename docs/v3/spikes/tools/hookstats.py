import json, sys, re, statistics
def pct(v, p):
    v = sorted(v); k = max(0, min(len(v)-1, int(round(p/100*len(v)+0.5))-1)); return v[k]
for path in sys.argv[1:]:
    durs = {}; contents = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: o = json.loads(line)
            except Exception: continue
            a = o.get("attachment") or {}
            if a.get("hookEvent") == "UserPromptSubmit" or a.get("hookName","").startswith("UserPromptSubmit"):
                cmd = a.get("command") or a.get("hookName") or ""
                key = ("orca" if "orca" in str(cmd) else "sm") + ":" + str(a.get("type"))
                if "durationMs" in a: durs.setdefault(key, []).append(a["durationMs"])
                contents.setdefault(key, set()).add(str(a.get("content"))[:60])
    print(path.rsplit("/",1)[-1])
    for k, v in durs.items():
        print(f"  {k}: n={len(v)} median={statistics.median(v)} p95={pct(v,95)} max={max(v)} min={min(v)}")
        print("    content sample:", list(contents[k])[:2])
