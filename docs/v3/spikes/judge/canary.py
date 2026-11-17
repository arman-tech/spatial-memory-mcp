"""Canary hook for the judge isolation check: appends one line per call."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "canary.log"), "a", encoding="utf-8") as f:
    f.write(f"{sys.argv[1]} marker={os.environ.get('SPATIAL_MEMORY_JUDGE', '<unset>')}\n")
