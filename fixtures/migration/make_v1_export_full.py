"""Writes v1_export_full.json: a version 1 export in the exact byte format of
version 1's JSON exporter (master, spatial_memory/services/export_import.py,
_export_json L606-650 and _json_serializer L749), with include_vectors left at
its default (true).

The records are invented test data. The secret in record 4 is the example key
from Amazon's own documentation, not a real credential.

Run with: python -I make_v1_export_full.py   (writes next to this file)
Only the standard library is used, so the output is reproducible.
"""

import hashlib
import json
import random
import struct
from datetime import datetime
from pathlib import Path


def f32(value: float) -> float:
    """Round a Python float through 32-bit storage, as LanceDB does."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
    norm = sum(x * x for x in raw) ** 0.5
    return [f32(x / norm) for x in raw]


def content_hash(content: str) -> str:
    """version 1 core/hashing.py: sha256 hex of content.strip().lower()."""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


def iso(dt: datetime) -> str:
    """version 1 _json_serializer: naive datetime.isoformat(), values in UTC."""
    return dt.isoformat()


RECORDS = [
    {
        "id": "3f6c2a1e-8b7d-4c59-9e0a-5d1f2b3c4a6e",
        "content": (
            "Decision: use optimistic concurrency on dbo.Orders. Two writers updating "
            "the same order lost updates under the old last-write-wins rule; a rowversion "
            "column now rejects the stale write and the API retries once."
        ),
        "created_at": datetime(2025, 8, 14, 9, 21, 7, 512340),
        "updated_at": datetime(2025, 8, 14, 9, 21, 7),
        "last_accessed": datetime(2025, 10, 2, 16, 3, 44, 90001),
        "access_count": 3,
        "importance": f32(0.7),
        "namespace": "decisions",
        "tags": ["database", "concurrency", "decision"],
        "source": "manual",
        "metadata": {"author": "dev", "review_state": "approved"},
        "project": "spatial-memory-mcp",
        "expires_at": None,
        "seed": 1,
    },
    {
        "id": "a8e5b0c4-1d2f-4e6a-b7c8-9d0e1f2a3b4c",
        "content": (
            "订单服务在缓存未命中时会返回"
            "空值 — always check the cache result before dereferencing. "
            "\U0001f680 Fixed in billing-service 2.4."
        ),
        "created_at": datetime(2025, 9, 1, 0, 0, 0),
        "updated_at": datetime(2025, 9, 3, 12, 30, 0),
        "last_accessed": datetime(2025, 9, 3, 12, 30, 0),
        "access_count": 0,
        "importance": f32(0.85),
        "namespace": "billing-service",
        "tags": ["i18n", "cache"],
        "source": "extracted",
        "metadata": {"locale": "zh-CN"},
        "project": "billing-service",
        "expires_at": datetime(2027, 1, 1, 0, 0, 0),
        "seed": 2,
    },
    {
        "id": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f",
        "content": "Convention: integration tests run against a throwaway SQLite file, never the shared dev database.",
        "created_at": datetime(2024, 12, 31, 23, 59, 59, 999999),
        "updated_at": datetime(2025, 1, 1, 0, 0, 0, 1),
        "last_accessed": datetime(2025, 1, 1, 0, 0, 0, 1),
        "access_count": 12,
        "importance": f32(0.5),
        "namespace": "default",
        "tags": [],
        "source": "consolidated",
        "metadata": {},
        "project": None,
        "expires_at": None,
        "seed": 3,
    },
    {
        "id": "0b9a8c7d-6e5f-4a3b-9c2d-1e0f9a8b7c6d",
        "content": (
            "Runbook: the staging deploy needs AWS credentials in the environment. "
            "Example from the AWS docs: AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE "
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        ),
        "created_at": datetime(2025, 6, 30, 8, 0, 0, 250000),
        "updated_at": datetime(2025, 6, 30, 8, 0, 0, 250000),
        "last_accessed": datetime(2025, 6, 30, 8, 0, 0, 250000),
        "access_count": 1,
        "importance": f32(0.6),
        "namespace": "runbooks",
        "tags": ["deploy", "aws"],
        "source": "import",
        "metadata": {"imported_from": "wiki"},
        "project": "spatial-memory-mcp",
        "expires_at": None,
        "seed": 4,
    },
    {
        "id": "f0e1d2c3-b4a5-4697-8879-6a5b4c3d2e1f",
        "content": "  Pitfall: CS8602 null dereference in OrderService.GetOrderAsync when the cache misses.  \n",
        "created_at": datetime(2025, 11, 5, 14, 45, 10, 300),
        "updated_at": datetime(2025, 11, 5, 14, 45, 10, 300),
        "last_accessed": datetime(2025, 11, 6, 9, 0, 0),
        "access_count": 2,
        "importance": f32(0.9),
        "namespace": "troubleshooting",
        "tags": ["csharp", "nullability"],
        "source": "manual",
        "metadata": {"severity": "high"},
        "project": "billing-service",
        "expires_at": None,
        "seed": 5,
    },
    {
        "id": "5e4d3c2b-1a09-4f8e-8d7c-6b5a49382716",
        "content": (
            "# Release checklist\n\n"
            "## Before tagging\n\n"
            "Run the full test suite on Windows and Linux. Check that the changelog names "
            "every user-visible change and that the version in the manifest matches the tag.\n\n"
            "## After tagging\n\n"
            "Install from the tag on a clean machine, run setup, and confirm that recall "
            "returns a memory stored before the upgrade.\n\n"
            "```text\nspatial-memory doctor\nspatial-memory status\n```\n"
        ),
        "created_at": datetime(2025, 3, 3, 3, 3, 3),
        "updated_at": datetime(2025, 3, 4, 4, 4, 4, 404040),
        "last_accessed": datetime(2025, 3, 4, 4, 4, 4, 404040),
        "access_count": 7,
        "importance": f32(0.4),
        "namespace": "procedures",
        "tags": ["release"],
        "source": "manual",
        "metadata": {},
        "project": "spatial-memory-mcp",
        "expires_at": None,
        "seed": 6,
    },
]


def export_record(spec: dict) -> dict:
    rec = {
        "id": spec["id"],
        "content": spec["content"],
        "vector": vector(spec["seed"]),
        "created_at": iso(spec["created_at"]),
        "updated_at": iso(spec["updated_at"]),
        "last_accessed": iso(spec["last_accessed"]),
        "access_count": spec["access_count"],
        "importance": spec["importance"],
        "namespace": spec["namespace"],
        "tags": spec["tags"],
        "source": spec["source"],
        "metadata": spec["metadata"],
    }
    # project was added to version 1 by a later migration; old rows lack it.
    if spec["project"] is not None:
        rec["project"] = spec["project"]
    rec["content_hash"] = content_hash(spec["content"])
    rec["expires_at"] = iso(spec["expires_at"]) if spec["expires_at"] else None
    return rec


def main() -> None:
    parts = []
    for spec in RECORDS:
        text = json.dumps(export_record(spec), indent=2)
        parts.append("\n".join("  " + line for line in text.split("\n")))
    out = Path(__file__).with_name("v1_export_full.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("[\n")
        f.write(",\n".join(parts))
        f.write("\n]")


if __name__ == "__main__":
    main()
