"""Common utility functions for Spatial Memory MCP."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def deserialize_metadata(record: dict[str, Any]) -> None:
    """Deserialize metadata JSON string to dict in-place.

    Args:
        record: Database record dict to modify in-place
    """
    if record.get("metadata"):
        if isinstance(record["metadata"], str):
            record["metadata"] = json.loads(record["metadata"])
    else:
        record["metadata"] = {}


def serialize_metadata(metadata: dict[str, Any] | None) -> str:
    """Serialize metadata dict to JSON string.

    Args:
        metadata: Metadata dict or None

    Returns:
        JSON string
    """
    return json.dumps(metadata) if metadata else "{}"


def convert_distance_to_similarity(record: dict[str, Any]) -> None:
    """Convert _distance field to similarity score in-place.

    Similarity is calculated as 1 - distance and clamped to [0, 1].

    Args:
        record: Database record dict to modify in-place
    """
    if "_distance" in record:
        record["similarity"] = max(0.0, min(1.0, 1 - record["_distance"]))
        del record["_distance"]


def deserialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fully deserialize a database record.

    Applies:
    - Metadata JSON deserialization
    - Distance to similarity conversion

    Args:
        record: Database record dict

    Returns:
        Deserialized record
    """
    deserialize_metadata(record)
    convert_distance_to_similarity(record)
    return record


def serialize_vector(vector: np.ndarray | list[float]) -> list[float]:
    """Convert vector to list format for storage.

    Args:
        vector: numpy array or list

    Returns:
        List of floats
    """
    if isinstance(vector, np.ndarray):
        result: list[float] = vector.tolist()
        return result
    return vector
