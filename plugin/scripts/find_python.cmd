@echo off
REM Resolve Python 3 interpreter for hook scripts and MCP server (Windows).
REM Fail-open: if no Python found, emit a safe hook response and exit 0.

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python %*
    exit /b %ERRORLEVEL%
)

where python3 >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python3 %*
    exit /b %ERRORLEVEL%
)

echo {"continue": true, "suppressOutput": true}
exit /b 0
