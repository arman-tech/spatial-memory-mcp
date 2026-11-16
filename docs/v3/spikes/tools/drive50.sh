#!/usr/bin/env bash
# Spike f driver: one Claude Code session, N prompts sent one at a time
# (each waits for the previous turn's result), so every prompt fires
# its own UserPromptSubmit hook. Usage: drive50.sh <variant exec|mcp> <N>
set -u
S="C:/Users/jon/AppData/Local/Temp/claude/409aeca82b3b64a0/120806bb-ad38-440a-a1f2-1d2c0b5ad9bb/scratchpad/spikes"
V="$1"; N="$2"
OUT="$S/run50/$V"; mkdir -p "$OUT"; rm -f "$OUT"/*
export PATH="$(cygpath -u "$S/rmcp-stdio/bin"):$PATH"
export SPIKE_HOOK_LOG="$(cygpath -w "$OUT/hook.log")"
cd "$OUT"
EXTRA=()
if [ "$V" = exec ]; then EXTRA=(--strict-mcp-config); fi
coproc CL { claude -p --input-format stream-json --output-format stream-json --verbose \
  --model haiku "${EXTRA[@]}" --plugin-dir "$S/plugin-$V" \
  --debug=hooks --debug-file "$OUT/debug.log" 2> "$OUT/stderr.log"; }
for i in $(seq 1 "$N"); do
  printf '{"type":"user","message":{"role":"user","content":"Message %d of the hook timing run: reply with only the number %d."}}\n' "$i" "$i" >&"${CL[1]}"
  while IFS= read -r line <&"${CL[0]}"; do
    printf '%s\n' "$line" >> "$OUT/out.jsonl"
    case "$line" in *'"type":"result"'*) break ;; esac
  done
done
exec {CL[1]}>&-
wait
echo "done $V results=$(rg -c '"type":"result"' "$OUT/out.jsonl")"
