"""Tests for spatial_memory.core.helpers module."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from spatial_memory.core.helpers import (
    convert_distance_to_similarity,
    deserialize_metadata,
    deserialize_record,
    serialize_metadata,
    serialize_vector,
)


class TestDeserializeMetadata:
    """Tests for deserialize_metadata function."""

    def test_deserialize_json_string(self) -> None:
        """Test deserializing valid JSON string."""
        record: dict[str, Any] = {"metadata": '{"key": "value", "count": 42}'}
        deserialize_metadata(record)
        assert record["metadata"] == {"key": "value", "count": 42}

    def test_deserialize_empty_string(self) -> None:
        """Test deserializing empty string."""
        record: dict[str, Any] = {"metadata": ""}
        deserialize_metadata(record)
        assert record["metadata"] == {}

    def test_deserialize_null(self) -> None:
        """Test deserializing None value."""
        record: dict[str, Any] = {"metadata": None}
        deserialize_metadata(record)
        assert record["metadata"] == {}

    def test_deserialize_missing_field(self) -> None:
        """Test deserializing when metadata field is missing."""
        record: dict[str, Any] = {"id": "123"}
        deserialize_metadata(record)
        assert record["metadata"] == {}

    def test_already_dict(self) -> None:
        """Test when metadata is already a dict (no-op)."""
        record = {"metadata": {"already": "dict"}}
        deserialize_metadata(record)
        assert record["metadata"] == {"already": "dict"}

    def test_complex_nested_json(self) -> None:
        """Test deserializing complex nested JSON."""
        record: dict[str, Any] = {"metadata": '{"nested": {"array": [1, 2, 3], "bool": true}}'}
        deserialize_metadata(record)
        assert record["metadata"] == {"nested": {"array": [1, 2, 3], "bool": True}}

    def test_modifies_in_place(self) -> None:
        """Test that function modifies record in-place."""
        record = {"metadata": '{"test": "value"}'}
        deserialize_metadata(record)
        assert isinstance(record["metadata"], dict)


class TestSerializeMetadata:
    """Tests for serialize_metadata function."""

    def test_serialize_dict(self) -> None:
        """Test serializing a dictionary."""
        metadata = {"key": "value", "count": 42}
        result = serialize_metadata(metadata)
        assert json.loads(result) == metadata

    def test_serialize_empty_dict(self) -> None:
        """Test serializing empty dictionary."""
        result = serialize_metadata({})
        assert result == "{}"

    def test_serialize_none(self) -> None:
        """Test serializing None."""
        result = serialize_metadata(None)
        assert result == "{}"

    def test_serialize_nested(self) -> None:
        """Test serializing nested structure."""
        metadata = {"nested": {"array": [1, 2, 3], "bool": True}}
        result = serialize_metadata(metadata)
        assert json.loads(result) == metadata

    def test_roundtrip(self) -> None:
        """Test serialize -> deserialize roundtrip."""
        original = {"test": "value", "number": 123}
        serialized = serialize_metadata(original)
        record: dict[str, Any] = {"metadata": serialized}
        deserialize_metadata(record)
        assert record["metadata"] == original


class TestConvertDistanceToSimilarity:
    """Tests for convert_distance_to_similarity function."""

    def test_convert_zero_distance(self) -> None:
        """Test converting zero distance (perfect match)."""
        record = {"_distance": 0.0}
        convert_distance_to_similarity(record)
        assert record["similarity"] == 1.0
        assert "_distance" not in record

    def test_convert_small_distance(self) -> None:
        """Test converting small distance."""
        record = {"_distance": 0.2}
        convert_distance_to_similarity(record)
        assert record["similarity"] == 0.8
        assert "_distance" not in record

    def test_convert_large_distance(self) -> None:
        """Test converting distance > 1 (clamped to 0)."""
        record = {"_distance": 1.5}
        convert_distance_to_similarity(record)
        assert record["similarity"] == 0.0
        assert "_distance" not in record

    def test_convert_negative_distance(self) -> None:
        """Test converting negative distance (clamped to 1)."""
        record = {"_distance": -0.5}
        convert_distance_to_similarity(record)
        assert record["similarity"] == 1.0
        assert "_distance" not in record

    def test_no_distance_field(self) -> None:
        """Test when _distance field is missing (no-op)."""
        record = {"id": "123"}
        convert_distance_to_similarity(record)
        assert "similarity" not in record
        assert "_distance" not in record

    def test_preserves_other_fields(self) -> None:
        """Test that other fields are preserved."""
        record = {"_distance": 0.3, "id": "123", "content": "test"}
        convert_distance_to_similarity(record)
        assert record["similarity"] == 0.7
        assert record["id"] == "123"
        assert record["content"] == "test"

    def test_modifies_in_place(self) -> None:
        """Test that function modifies record in-place."""
        record = {"_distance": 0.5}
        convert_distance_to_similarity(record)
        assert "_distance" not in record
        assert "similarity" in record


class TestDeserializeRecord:
    """Tests for deserialize_record function."""

    def test_full_deserialization(self) -> None:
        """Test complete record deserialization."""
        record = {
            "id": "123",
            "metadata": '{"key": "value"}',
            "_distance": 0.3,
            "content": "test",
        }
        result = deserialize_record(record)
        assert result is record  # Returns same object
        assert result["metadata"] == {"key": "value"}
        assert result["similarity"] == 0.7
        assert "_distance" not in result

    def test_metadata_only(self) -> None:
        """Test record with only metadata to deserialize."""
        record = {"metadata": '{"test": "data"}'}
        result = deserialize_record(record)
        assert result["metadata"] == {"test": "data"}
        assert "similarity" not in result

    def test_distance_only(self) -> None:
        """Test record with only distance to convert."""
        record = {"_distance": 0.1}
        result = deserialize_record(record)
        assert result["similarity"] == 0.9
        assert "_distance" not in result
        assert result["metadata"] == {}

    def test_empty_record(self) -> None:
        """Test empty record."""
        record: dict[str, Any] = {}
        result = deserialize_record(record)
        assert result["metadata"] == {}
        assert "similarity" not in result

    def test_already_deserialized(self) -> None:
        """Test record that is already deserialized."""
        record = {"metadata": {"already": "dict"}, "similarity": 0.95}
        result = deserialize_record(record)
        assert result["metadata"] == {"already": "dict"}
        assert result["similarity"] == 0.95


class TestSerializeVector:
    """Tests for serialize_vector function."""

    def test_numpy_array(self) -> None:
        """Test converting numpy array to list."""
        vector = np.array([1.0, 2.0, 3.0])
        result = serialize_vector(vector)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_list_passthrough(self) -> None:
        """Test that lists are returned as-is."""
        vector = [1.0, 2.0, 3.0]
        result = serialize_vector(vector)
        assert result is vector
        assert result == [1.0, 2.0, 3.0]

    def test_multidimensional_array(self) -> None:
        """Test converting multidimensional numpy array."""
        vector = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Note: This test uses a 2D array, which is not the intended type
        # but the function handles it gracefully at runtime
        result: Any = serialize_vector(vector)
        assert isinstance(result, list)
        assert len(result) == 2  # Just verify structure without exact comparison

    def test_empty_array(self) -> None:
        """Test converting empty numpy array."""
        vector = np.array([])
        result = serialize_vector(vector)
        assert isinstance(result, list)
        assert result == []

    def test_integer_array(self) -> None:
        """Test converting integer numpy array."""
        vector = np.array([1, 2, 3])
        result = serialize_vector(vector)
        assert isinstance(result, list)
        # NumPy integers are preserved in the list
        assert len(result) == 3

    def test_large_vector(self) -> None:
        """Test converting large numpy array."""
        vector = np.random.randn(384)  # Common embedding dimension
        result = serialize_vector(vector)
        assert isinstance(result, list)
        assert len(result) == 384
