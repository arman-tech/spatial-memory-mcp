"""Build a v1.x-shaped LanceDB sample database fixture.

Run from repo root:

    python -m tests.fixtures.scripts.build_v1_database_sample [output_path]

Default output path: tests/fixtures/v1_database_sample/

The fixture mimics a v1.11.4 database:
  * `memories` table with 5 rows of 384-dim vectors
  * NO `system_metadata` table (the field that distinguishes v1 from v2)
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "v1_database_sample"
EMBEDDING_DIM = 384
ROW_COUNT = 5


def build(output_path: Path) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(output_path))

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("last_accessed", pa.timestamp("us")),
            pa.field("access_count", pa.int32()),
            pa.field("importance", pa.float32()),
            pa.field("namespace", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("source", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("project", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("expires_at", pa.timestamp("us")),
        ]
    )

    rng = np.random.default_rng(seed=42)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for i in range(ROW_COUNT):
        vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        rows.append(
            {
                "id": f"v1-mem-{i:03d}",
                "content": f"Sample v1 memory #{i}",
                "vector": vec.tolist(),
                "created_at": now,
                "updated_at": now,
                "last_accessed": now,
                "access_count": 0,
                "importance": 0.5,
                "namespace": "default",
                "tags": [],
                "source": "v1-fixture",
                "metadata": "{}",
                "project": "",
                "content_hash": f"hash-{i:03d}",
                "expires_at": None,
            }
        )

    db.create_table("memories", data=rows, schema=schema)
    # Intentionally NOT creating system_metadata — that absence is what
    # distinguishes a v1.x DB from a v2.0 DB.

    return output_path


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    path = build(out)
    print(f"Built v1 fixture at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
