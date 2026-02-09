#!/bin/bash
# Resolve Python 3 interpreter for hook scripts and MCP server.
# Uses exec to replace the shell process — correct for both hook scripts
# (stdin/stdout JSON) and MCP server (stdio protocol).
# Fail-open: if no Python found, emit a safe hook response and exit 0.

if command -v python3 &>/dev/null; then
    exec python3 "$@"
elif command -v python &>/dev/null; then
    exec python "$@"
else
    echo '{"continue": true, "suppressOutput": true}'
    exit 0
fi
