"""Unit tests for Database.insert(_skip_field_validation=True).

Tests verify that the _skip_field_validation flag correctly bypasses
content and importance checks while still enforcing vector dimension
validation and providing defaults for tags/metadata.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial_memory.core.database import Database
from spatial_memory.core.errors import DimensionMismatchError, ValidationError

# ===========================================================================
# Helpers
# ===========================================================================

EMBEDDING_DIM = 384


@pytest.fixture
def db() -> Database:
    """Create a Database instance without calling connect().

    We mock the `table` property so that insert() can call
    self.table.add() without a real LanceDB connection.
    """
    database = Database.__new__(Database)
    database.embedding_dim = EMBEDDING_DIM
    database.default_memory_ttl_days = None
    # Provide a mock table so insert() can call self.table.add(...)
    mock_table = MagicMock()
    database._table = mock_table
    # _to_arrow_table is called internally; stub it out
    database._to_arrow_table = MagicMock()
    # Stub cache/tracking helpers that insert() calls after table.add
    database._invalidate_count_cache = MagicMock()
    database._track_modification = MagicMock()
    database._invalidate_namespace_cache = MagicMock()
    # Provide lock attributes required by the write decorators
    database._process_lock = None  # None = no cross-process lock (skip decorator)
    database._write_lock = threading.RLock()
    return database


def make_vector(dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ===========================================================================
# TestSkipFieldValidation
# ===========================================================================


@pytest.mark.unit
class TestSkipFieldValidation:
    """Tests for the _skip_field_validation flag on Database.insert()."""

    def test_skip_validation_bypasses_content_check(self, db: Database) -> None:
        """With _skip_field_validation=True, empty content should NOT raise."""
        vector = make_vector()

        # Should NOT raise ValidationError for empty content
        memory_id = db.insert(
            content="",
            vector=vector,
            _skip_field_validation=True,
        )

        assert isinstance(memory_id, str)
        assert len(memory_id) == 36  # UUID format

    def test_skip_validation_bypasses_importance_check(self, db: Database) -> None:
        """With _skip_field_validation=True, importance=-1 should NOT raise."""
        vector = make_vector()

        # Should NOT raise ValidationError for out-of-range importance
        memory_id = db.insert(
            content="test",
            vector=vector,
            importance=-1,
            _skip_field_validation=True,
        )

        assert isinstance(memory_id, str)

    def test_skip_validation_still_checks_vector_dimension(self, db: Database) -> None:
        """With _skip_field_validation=True, wrong-dimension vector should STILL raise."""
        wrong_dim_vector = make_vector(dim=128)  # Expect 384

        with pytest.raises(DimensionMismatchError) as exc_info:
            db.insert(
                content="test",
                vector=wrong_dim_vector,
                _skip_field_validation=True,
            )

        assert exc_info.value.expected_dim == EMBEDDING_DIM
        assert exc_info.value.actual_dim == 128

    def test_skip_validation_defaults_none_tags(self, db: Database) -> None:
        """With _skip_field_validation=True, tags=None should default to []."""
        vector = make_vector()

        db.insert(
            content="test",
            vector=vector,
            tags=None,
            _skip_field_validation=True,
        )

        # Inspect the record that was passed to _to_arrow_table
        call_args = db._to_arrow_table.call_args
        records = call_args[0][0]
        assert records[0]["tags"] == []

    def test_skip_validation_defaults_none_metadata(self, db: Database) -> None:
        """With _skip_field_validation=True, metadata=None should default to {}."""
        vector = make_vector()

        db.insert(
            content="test",
            vector=vector,
            metadata=None,
            _skip_field_validation=True,
        )

        # Inspect the record that was passed to _to_arrow_table
        # metadata is json.dumps()'d in insert(), so {} becomes '{}'
        call_args = db._to_arrow_table.call_args
        records = call_args[0][0]
        assert records[0]["metadata"] == "{}"

    # --- Contrast tests: without flag, validation IS enforced ---

    def test_without_flag_empty_content_raises(self, db: Database) -> None:
        """Without _skip_field_validation, empty content SHOULD raise."""
        vector = make_vector()

        with pytest.raises(ValidationError, match="Content must be between"):
            db.insert(content="", vector=vector, _skip_field_validation=False)

    def test_without_flag_bad_importance_raises(self, db: Database) -> None:
        """Without _skip_field_validation, importance=-1 SHOULD raise."""
        vector = make_vector()

        with pytest.raises(ValidationError, match="Importance must be between"):
            db.insert(content="test", vector=vector, importance=-1, _skip_field_validation=False)
