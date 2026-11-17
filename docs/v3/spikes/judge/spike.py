"""Judge spike: one non-interactive Claude Code call on a batch of synthetic candidates.

usage: py -3.13 -I spike.py <label> <first> <count> [--json] [--extra-env K=V ...] [--config-dir DIR]
       [--settings-file F] [--claude PATH] [--no-safe-mode] [--effort LEVEL] [--ids c01,c02]

Writes <label>.out.jsonl (the raw stdout), <label>.debug.log (claude's debug file)
and <label>.summary.json next to this script. Candidates are synthetic
(candidates.json), written for this spike.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def find_claude():
    """claude.exe on PATH; else the npm package's binary next to a claude.cmd on PATH;
    else the native installer's ~/.local/bin/claude.exe."""
    exe = shutil.which("claude.exe")
    if exe:
        return exe
    cmd = shutil.which("claude.cmd") or shutil.which("claude")
    if cmd:
        cand = os.path.join(os.path.dirname(cmd), "node_modules", "@anthropic-ai",
                            "claude-code", "bin", "claude.exe")
        if os.path.exists(cand):
            return cand
    return os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe")


CLAUDE = find_claude()
WORK = os.path.join(os.environ["LOCALAPPDATA"], "spatial-memory", "judge", "work")

PASS_ENV = ["PATH", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "SystemRoot",
            "SYSTEMROOT", "TEMP", "TMP", "ComSpec", "PATHEXT", "windir", "ProgramData",
            "USERNAME", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy",
            "https_proxy", "no_proxy", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_GIT_BASH_PATH",
            "SystemDrive", "HOMEDRIVE", "HOMEPATH", "ProgramFiles", "ProgramFiles(x86)",
            "CommonProgramFiles", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "OS"]


def main():
    args = sys.argv[1:]
    label, first, count = args[0], int(args[1]), int(args[2])
    use_json = "--json" in args
    extra = {}
    config_dir = None
    settings = None
    claude = CLAUDE
    effort = None
    ids = None
    i = 3
    while i < len(args):
        if args[i] == "--extra-env":
            k, v = args[i + 1].split("=", 1)
            extra[k] = v
            i += 2
        elif args[i] == "--config-dir":
            config_dir = args[i + 1]
            i += 2
        elif args[i] == "--settings-file":
            settings = open(args[i + 1], encoding="utf-8").read()
            i += 2
        elif args[i] == "--claude":
            claude = args[i + 1]
            i += 2
        elif args[i] == "--effort":
            effort = args[i + 1]
            i += 2
        elif args[i] == "--ids":
            ids = args[i + 1].split(",")
            i += 2
        else:
            i += 1

    cands = json.load(open(os.path.join(HERE, "candidates.json"), encoding="utf-8"))
    batch = [{"id": c["id"], "text": c["text"], "context": c["context"]}
             for c in (cands[first:first + count] if ids is None else [c for c in cands if c["id"] in ids])]
    prompt = open(os.path.join(HERE, "prompt.txt"), encoding="utf-8").read()
    schema = open(os.path.join(HERE, "schema.json"), encoding="utf-8").read().strip()

    os.makedirs(WORK, exist_ok=True)
    assert not os.listdir(WORK), f"working folder not empty: {os.listdir(WORK)}"

    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env["SPATIAL_MEMORY_JUDGE"] = "1"
    if config_dir:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    env.update(extra)

    debug = os.path.join(HERE, f"{label}.debug.log")
    if os.path.exists(debug):
        os.remove(debug)
    cmd = [claude, "-p",
           "--model", "haiku",
           *([] if "--no-safe-mode" in args else ["--safe-mode"]),
           "--setting-sources", "",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--tools", "",
           "--disable-slash-commands",
           "--no-session-persistence",
           "--permission-prompts", "none",
           "--system-prompt", prompt,
           "--json-schema", schema,
           "--output-format", "json" if use_json else "stream-json",
           "--debug-file", debug]
    if not use_json:
        cmd.append("--verbose")
    if settings:
        cmd += ["--settings", settings]
    if effort:
        cmd += ["--effort", effort]

    projects = os.path.join(config_dir or os.path.join(os.environ["USERPROFILE"], ".claude"), "projects")
    before = snapshot(projects)
    t0 = time.monotonic()
    r = subprocess.run(cmd, input=json.dumps(batch, ensure_ascii=False), capture_output=True,
                       text=True, encoding="utf-8", cwd=WORK, env=env, timeout=600)
    wall = time.monotonic() - t0
    after = snapshot(projects)
    leftovers = os.listdir(WORK)

    open(os.path.join(HERE, f"{label}.out.jsonl"), "w", encoding="utf-8").write(r.stdout)
    events = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    summary = {
        "label": label, "count": len(batch), "exit": r.returncode, "wall_s": round(wall, 2),
        "stderr": r.stderr[-2000:],
        "new_files_under_projects": sorted(set(after) - set(before)),
        "changed_files_under_projects": sorted(k for k in after if k in before and after[k] != before[k]),
        "work_folder_after": leftovers,
    }
    if init:
        summary["init"] = {k: init.get(k) for k in ("cwd", "model", "apiKeySource", "tools",
                                                    "mcp_servers", "slash_commands", "agents",
                                                    "skills", "plugins", "permissionMode",
                                                    "output_style", "claude_code_version")}
    if result:
        summary["result"] = {k: result.get(k) for k in ("subtype", "is_error", "duration_ms",
                                                        "duration_api_ms", "num_turns",
                                                        "total_cost_usd", "usage", "modelUsage",
                                                        "permission_denials", "api_error_status")}
        summary["structured_output"] = result.get("structured_output")
        summary["result_text"] = (result.get("result") or "")[:500]
    if os.path.exists(debug):
        lines = open(debug, encoding="utf-8", errors="replace").read().splitlines()
        summary["debug_lines"] = len(lines)
        summary["debug_hook_lines"] = [l[:300] for l in lines if "hook" in l.lower()][:40]
        summary["debug_mcp_lines"] = [l[:300] for l in lines if "mcp" in l.lower()][:20]
        summary["debug_plugin_lines"] = [l[:300] for l in lines if "plugin" in l.lower()][:20]
    json.dump(summary, open(os.path.join(HERE, f"{label}.summary.json"), "w", encoding="utf-8"), indent=1)
    res = summary.get("result") or {}
    print(json.dumps({"label": label, "n": len(batch), "exit": r.returncode, "wall_s": summary["wall_s"],
                      "cost": res.get("total_cost_usd"), "turns": res.get("num_turns"),
                      "api_ms": res.get("duration_api_ms"), "subtype": res.get("subtype"),
                      "is_error": res.get("is_error"),
                      "models": list((res.get("modelUsage") or {}).keys()),
                      "new_project_files": summary["new_files_under_projects"],
                      "stderr": r.stderr[-300:]}))


def snapshot(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return out


if __name__ == "__main__":
    main()
