import json
import os

here = os.path.dirname(os.path.abspath(__file__))
canary = os.path.join(here, "canary.py")


def hook(event):
    return [{"hooks": [{"type": "command", "command": r"C:\Windows\py.exe",
                        "args": ["-3.13", "-I", canary, event]}]}]


settings = {"hooks": {e: hook(e) for e in ("SessionStart", "UserPromptSubmit", "Stop", "SessionEnd")}}
open(os.path.join(here, "canary-settings.json"), "w", encoding="utf-8").write(json.dumps(settings))
print(json.dumps(settings)[:300])
