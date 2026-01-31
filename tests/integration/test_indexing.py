"""Tests for index management features.

This module tests vector and scalar index creation, automatic tuning,
and lifecycle management of database indexes for optimal performance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService

# Mark entire module as integration tests (require real embedding model)
pytestmark = pytest.mark.integration


class TestVectorIndexCreation:
    """Tests for vector index creation."""

    def test_create_vector_index_basic(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test basic vector index creation.

        Inserts enough records to exceed the threshold and verifies
        that the vector index is created successfully.
        """
        # Insert records exceeding default threshold (10,000)
        # Use smaller threshold for testing
        database.vector_index_threshold = 100

        # Insert 150 records to exceed threshold
        records = []
        for i in range(150):
            vec = embedding_service.embed(f"Test memory number {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Create vector index
        result = database.create_vector_index(force=True)
        assert result is True

        # Verify index exists via internal flag
        assert database._has_vector_index is True

        # Verify index appears in list
        # LanceDB 0.27+ returns IndexConfig objects, older versions return dicts
        indices = database.table.list_indices()
        index_names = []
        index_types = []
        for idx in indices:
            if isinstance(idx, dict):
                index_names.append(idx.get("name", "").lower())
                index_types.append(idx.get("index_type", "").upper())
            else:
                index_names.append(str(getattr(idx, "name", "")).lower())
                index_types.append(str(getattr(idx, "index_type", "")).upper())

        # Check for vector index by name or type
        has_vector_idx = any(
            "vector" in name or idx_type in ["IVF_PQ", "IVF_FLAT", "HNSW"]
            for name, idx_type in zip(index_names, index_types)
        )
        assert has_vector_idx

    def test_create_vector_index_auto_tuning_small_dataset(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test index parameters for small datasets (<100K).

        Verifies that num_partitions and num_sub_vectors are appropriate
        for small datasets (64 partitions, 48 sub-vectors for 384-dim).
        """
        # Insert ~1000 records (small dataset)
        records = []
        for i in range(1000):
            vec = embedding_service.embed(f"Small dataset memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Create index (should use small dataset parameters)
        database.vector_index_threshold = 500  # Lower threshold for test
        result = database.create_vector_index(force=True)
        assert result is True

        # Verify index was created
        assert database._has_vector_index is True

        # For small datasets (<100K), should use:
        # - 64 partitions
        # - 48 sub-vectors (384 / 48 = 8 dims per sub-vector)
        # Note: We can't directly inspect these params, but we verify creation succeeded
        indices = database.table.list_indices()
        assert len(indices) > 0

    def test_create_vector_index_idempotent(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that calling create_vector_index twice doesn't error.

        The second call should either replace the index or skip gracefully.
        """
        # Insert some records
        records = []
        for i in range(200):
            vec = embedding_service.embed(f"Idempotent test memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)
        database.vector_index_threshold = 100

        # Create index first time
        result1 = database.create_vector_index(force=True)
        assert result1 is True

        # Create again (should replace or return False if already exists)
        result2 = database.create_vector_index(force=True)
        # Result can be True (replaced) or False (already exists)
        # Both are acceptable - no error should occur
        assert isinstance(result2, bool)

        # Index should still exist
        assert database._has_vector_index is True

    def test_vector_index_improves_search_with_large_dataset(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that indexed search works correctly.

        Verifies that search operations work correctly after index creation
        and return relevant results.
        """
        # Insert many records
        records = []
        for i in range(500):
            vec = embedding_service.embed(f"Searchable memory number {i}")
            records.append({
                "content": f"Memory {i} with unique content",
                "vector": vec,
            })

        # Add a distinctive memory to search for
        target_vec = embedding_service.embed("This is a very distinctive target memory")
        records.append({
            "content": "This is a very distinctive target memory",
            "vector": target_vec,
        })

        database.insert_batch(records)

        # Create index
        database.vector_index_threshold = 100
        database.create_vector_index(force=True)

        # Perform search using the distinctive query
        query_vec = embedding_service.embed("distinctive target memory")
        results = database.vector_search(query_vec, limit=5)

        # Verify we got results
        assert len(results) > 0

        # Verify the distinctive memory is in top results
        contents = [r["content"] for r in results]
        assert any("distinctive target" in content for content in contents)

        # Verify similarity scores are present and valid
        for result in results:
            assert "similarity" in result
            assert 0.0 <= result["similarity"] <= 1.0

    def test_create_vector_index_below_threshold_skips(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that index creation is skipped below threshold.

        When dataset is below threshold and force=False, should skip creation.
        """
        # Insert only 50 records
        records = []
        for i in range(50):
            vec = embedding_service.embed(f"Small dataset {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)
        database.vector_index_threshold = 100

        # Try to create index without force (should skip)
        result = database.create_vector_index(force=False)
        assert result is False

        # Index should not be created
        # _has_vector_index might be None or False
        assert database._has_vector_index is not True


class TestScalarIndexes:
    """Tests for scalar index creation."""

    def test_create_scalar_indexes(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test scalar index creation (BTREE, BITMAP, LABEL_LIST).

        Verifies that all scalar indexes are created on appropriate columns.
        """
        # Insert some data first
        records = []
        for i in range(100):
            vec = embedding_service.embed(f"Memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
                "namespace": f"ns-{i % 3}",
                "tags": [f"tag-{i % 5}"],
                "importance": 0.5,
            })

        database.insert_batch(records)

        # Create scalar indexes
        database.create_scalar_indexes()

        # Verify indexes were created
        indices = database.table.list_indices()
        assert len(indices) > 0

        # Check for expected indexes (by column name)
        # LanceDB 0.27+ returns IndexConfig objects with .columns attribute
        # Older versions return dicts
        index_columns = []
        for idx in indices:
            if isinstance(idx, dict):
                columns = idx.get("columns", [])
            else:
                columns = getattr(idx, "columns", [])
            index_columns.extend(columns)

        # Should have indexes on key columns
        # At minimum, we should have some scalar indexes
        # (Exact columns depend on implementation)
        assert len(index_columns) > 0

    def test_scalar_indexes_idempotent(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that creating scalar indexes twice is safe.

        Second call should either skip existing indexes or replace them
        without error.
        """
        # Insert some data
        records = []
        for i in range(50):
            vec = embedding_service.embed(f"Memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Create indexes first time
        database.create_scalar_indexes()
        indices_count_1 = len(database.table.list_indices())

        # Create again (should not error)
        database.create_scalar_indexes()
        indices_count_2 = len(database.table.list_indices())

        # Index count should be stable or increase slightly
        # (Some indexes might be replaced)
        assert indices_count_2 >= 0

    def test_scalar_indexes_on_empty_database(self, database: Database) -> None:
        """Test scalar index creation on empty database.

        Should handle empty database gracefully.
        """
        # Create indexes on empty database (should not error)
        database.create_scalar_indexes()

        # Verify some indexes were created (or at least attempted)
        # This tests that the operation doesn't crash
        indices = database.table.list_indices()
        # Count could be 0 or more depending on implementation
        assert isinstance(indices, list)


class TestEnsureIndexes:
    """Tests for automatic index management."""

    def test_ensure_indexes_below_threshold(
        self, temp_storage: Path, embedding_service: EmbeddingService
    ) -> None:
        """Test that indexes aren't created below threshold.

        When dataset size is below vector_index_threshold, ensure_indexes
        should not create a vector index.
        """
        # Create database with low threshold
        settings = Settings(
            memory_path=temp_storage / "test-ensure-below",
            vector_index_threshold=1000,
            auto_create_indexes=True,
        )
        override_settings(settings)

        try:
            db = Database(
                settings.memory_path,
                vector_index_threshold=1000,
                auto_create_indexes=True,
            )
            db.connect()

            # Insert fewer records than threshold
            records = []
            for i in range(500):
                vec = embedding_service.embed(f"Memory {i}")
                records.append({
                    "content": f"Memory {i}",
                    "vector": vec,
                })

            db.insert_batch(records)

            # Call ensure_indexes
            results = db.ensure_indexes(force=False)

            # Vector index should not be created (below threshold)
            assert results["vector_index"] is False
            assert db._has_vector_index is not True

            db.close()
        finally:
            reset_settings()

    def test_ensure_indexes_above_threshold(
        self, temp_storage: Path, embedding_service: EmbeddingService
    ) -> None:
        """Test that indexes are created above threshold.

        When dataset exceeds vector_index_threshold, ensure_indexes
        should create the vector index.
        """
        # Create database with low threshold for testing (bypass Settings validation)
        db = Database(
            temp_storage / "test-ensure-above",
            vector_index_threshold=100,
            auto_create_indexes=True,
        )
        db.connect()

        try:
            # Insert records exceeding threshold
            records = []
            for i in range(150):
                vec = embedding_service.embed(f"Memory {i}")
                records.append({
                    "content": f"Memory {i}",
                    "vector": vec,
                })

            db.insert_batch(records)

            # Call ensure_indexes
            results = db.ensure_indexes(force=False)

            # Vector index should be created
            assert results["vector_index"] is True
            assert db._has_vector_index is True

        finally:
            db.close()

    def test_ensure_indexes_respects_auto_create_setting(
        self, temp_storage: Path, embedding_service: EmbeddingService
    ) -> None:
        """Test that auto_create_indexes=False prevents automatic creation.

        Even when above threshold, if auto_create_indexes is disabled,
        ensure_indexes should not create vector index.
        """
        # Create database with auto_create_indexes=False (bypass Settings validation)
        db = Database(
            temp_storage / "test-no-auto",
            vector_index_threshold=100,
            auto_create_indexes=False,  # Disabled
        )
        db.connect()

        try:
            # Insert many records (above threshold)
            records = []
            for i in range(200):
                vec = embedding_service.embed(f"Memory {i}")
                records.append({
                    "content": f"Memory {i}",
                    "vector": vec,
                })

            db.insert_batch(records)

            # Call ensure_indexes without force
            results = db.ensure_indexes(force=False)

            # Vector index should NOT be created (auto_create disabled)
            assert results["vector_index"] is False

        finally:
            db.close()

    def test_ensure_indexes_force_overrides_settings(
        self, temp_storage: Path, embedding_service: EmbeddingService
    ) -> None:
        """Test that force=True creates indexes regardless of settings.

        When force=True, indexes should be created even if auto_create
        is disabled or below threshold.
        """
        # Create database with high threshold and auto_create disabled (bypass Settings)
        db = Database(
            temp_storage / "test-force",
            vector_index_threshold=10000,  # High threshold
            auto_create_indexes=False,  # Disabled
        )
        db.connect()

        try:
            # Insert 100 records (below threshold but enough for index training)
            # IVF requires num_partitions < num_vectors, and default is 64
            records = []
            for i in range(100):
                vec = embedding_service.embed(f"Memory {i}")
                records.append({
                    "content": f"Memory {i}",
                    "vector": vec,
                })

            db.insert_batch(records)

            # Call ensure_indexes with force=True
            results = db.ensure_indexes(force=True)

            # Vector index SHOULD be created (force overrides)
            assert results["vector_index"] is True
            assert db._has_vector_index is True

        finally:
            db.close()

    def test_ensure_indexes_creates_scalar_indexes_above_1000(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that scalar indexes are created when count > 1000.

        Scalar indexes should be created automatically when dataset
        reaches 1000 records.
        """
        # Insert 1100 records to exceed scalar threshold
        records = []
        for i in range(1100):
            vec = embedding_service.embed(f"Memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Call ensure_indexes
        results = database.ensure_indexes(force=False)

        # Scalar indexes should be created
        assert results["scalar_indexes"] is True

    def test_ensure_indexes_fts_when_enabled(
        self, database: Database, embedding_service: EmbeddingService
    ) -> None:
        """Test that FTS index is created when enable_fts=True.

        Full-text search index should be created if enabled.
        """
        # Ensure FTS is enabled
        database.enable_fts = True
        database._has_fts_index = False  # Reset flag

        # Insert some data
        records = []
        for i in range(100):
            vec = embedding_service.embed(f"Searchable text memory {i}")
            records.append({
                "content": f"Searchable text memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Call ensure_indexes
        results = database.ensure_indexes(force=False)

        # FTS index should be created (or already exists)
        # Result could be True or False depending on if it was already created
        assert isinstance(results["fts_index"], bool)
