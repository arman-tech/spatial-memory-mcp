"""Sync hook modules from spatial_memory/hooks/ to plugin/hooks/.

Copies all .py files (excluding __init__.py) from the source-of-truth
location to the plugin directory.  The plugin directory is committed to
the repo so that Claude Code can install it without building from source.

Usage:
    python scripts/sync_plugin_hooks.py           # Copy files
    python scripts/sync_plugin_hooks.py --check   # Verify (CI mode, no writes)
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "spatial_memory" / "hooks"
DST_DIR = REPO_ROOT / "plugin" / "hooks"

# Standard library top-level module names (Python 3.10+).
# We only check for obviously non-stdlib imports; a full check would
# require importlib.metadata which varies across Python versions.
_KNOWN_THIRD_PARTY = frozenset(
    {
        "lancedb",
        "numpy",
        "sentence_transformers",
        "openai",
        "pydantic",
        "mcp",
        "pyarrow",
        "requests",
        "httpx",
        "torch",
        "transformers",
        "onnxruntime",
    }
)


def _check_stdlib_only(path: Path) -> list[str]:
    """Check that a Python file uses only stdlib imports.

    Returns a list of violation messages (empty = clean).
    """
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return [f"{path.name}: syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _KNOWN_THIRD_PARTY:
                    violations.append(
                        f"{path.name}:{node.lineno}: imports third-party '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _KNOWN_THIRD_PARTY:
                    violations.append(
                        f"{path.name}:{node.lineno}: imports third-party '{node.module}'"
                    )

    return violations


def get_hook_files() -> list[Path]:
    """Get all .py files from source dir, excluding __init__.py."""
    return sorted(p for p in SRC_DIR.glob("*.py") if p.name != "__init__.py")


def sync(*, check_only: bool = False) -> int:
    """Sync or check hook files. Returns 0 on success, 1 on failure."""
    if not SRC_DIR.is_dir():
        print(f"ERROR: Source directory not found: {SRC_DIR}")
        return 1

    hook_files = get_hook_files()
    if not hook_files:
        print(f"ERROR: No .py files found in {SRC_DIR}")
        return 1

    print(f"Source: {SRC_DIR}")
    print(f"Target: {DST_DIR}")
    print(f"Files:  {len(hook_files)}")
    print()

    # Check stdlib-only constraint
    all_violations: list[str] = []
    for path in hook_files:
        all_violations.extend(_check_stdlib_only(path))

    if all_violations:
        print("STDLIB VIOLATIONS:")
        for v in all_violations:
            print(f"  {v}")
        print()
        return 1

    if check_only:
        # Compare files
        mismatches: list[str] = []
        missing: list[str] = []

        for src_path in hook_files:
            dst_path = DST_DIR / src_path.name
            if not dst_path.exists():
                missing.append(src_path.name)
                continue
            src_content = src_path.read_bytes()
            dst_content = dst_path.read_bytes()
            if src_content != dst_content:
                mismatches.append(src_path.name)

        # Check for extra files in destination
        if DST_DIR.is_dir():
            src_names = {p.name for p in hook_files}
            for dst_path in DST_DIR.glob("*.py"):
                if dst_path.name not in src_names:
                    mismatches.append(f"{dst_path.name} (extra in plugin/)")

        if missing or mismatches:
            if missing:
                print("MISSING in plugin/hooks/:")
                for name in missing:
                    print(f"  {name}")
            if mismatches:
                print("OUT OF SYNC:")
                for name in mismatches:
                    print(f"  {name}")
            print()
            print("Run 'python scripts/sync_plugin_hooks.py' to fix.")
            return 1

        print("OK: All files in sync.")
        return 0

    # Copy files
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for src_path in hook_files:
        dst_path = DST_DIR / src_path.name
        shutil.copy2(src_path, dst_path)
        print(f"  {src_path.name}")

    print(f"\nSynced {len(hook_files)} files.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync hook modules to plugin directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify files are in sync (CI mode, no writes)",
    )
    args = parser.parse_args()
    sys.exit(sync(check_only=args.check))


if __name__ == "__main__":
    main()
