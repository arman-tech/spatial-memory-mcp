"""Tests for backup and restore features.

This module tests Parquet export/import functionality for database
backup and restoration, including namespace override and data integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

# Mark entire module as integration tests (require real embedding model)
pytestmark = pytest.mark.integration

from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.errors import StorageError


class TestParquetExport:
    """Tests for Parquet export."""

    def test_export_to_parquet_creates_file(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that export creates a valid Parquet file.

        Verifies file is created, has correct format, and contains data.
        """
        # Insert test memories
        records = []
        for i in range(50):
            vec = embedding_service.embed(f"Export test memory {i}")
            records.append({
                "content": f"Memory {i} for export",
                "vector": vec,
                "namespace": "export-test",
                "tags": [f"tag-{i % 5}"],
                "importance": 0.5 + (i % 10) * 0.05,
            })

        database.insert_batch(records)

        # Export to Parquet
        output_path = temp_storage / "export_test.parquet"
        result = database.export_to_parquet(output_path, namespace=None)

        # Verify file was created
        assert output_path.exists()
        assert output_path.is_file()

        # Verify result metadata
        assert result["rows_exported"] == 50
        assert result["output_path"] == str(output_path)
        assert result["size_bytes"] > 0
        assert result["size_mb"] > 0

        # Verify file is valid Parquet format
        table = pq.read_table(output_path)
        assert table.num_rows == 50
        assert "content" in table.column_names
        assert "vector" in table.column_names
        assert "namespace" in table.column_names

    def test_export_to_parquet_returns_correct_count(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that export returns correct record count.

        Count returned should match number of records exported.
        """
        # Insert N records
        n_records = 25
        records = []
        for i in range(n_records):
            vec = embedding_service.embed(f"Count test memory {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        # Export
        output_path = temp_storage / "count_test.parquet"
        result = database.export_to_parquet(output_path)

        # Verify returned count
        assert result["rows_exported"] == n_records

        # Verify actual file content
        table = pq.read_table(output_path)
        assert table.num_rows == n_records

    def test_export_empty_database(
        self, database: Database, temp_storage: Path
    ) -> None:
        """Test exporting empty database.

        Should create file with 0 records or handle gracefully.
        """
        # Export empty database
        output_path = temp_storage / "empty_export.parquet"
        result = database.export_to_parquet(output_path)

        # Should succeed with 0 rows
        assert result["rows_exported"] == 0
        assert output_path.exists()

        # Verify file is valid but empty
        table = pq.read_table(output_path)
        assert table.num_rows == 0

    def test_export_filtered_by_namespace(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test exporting only specific namespace.

        When namespace is specified, only those records should be exported.
        """
        # Insert records in different namespaces
        for ns in ["ns-a", "ns-b", "ns-c"]:
            for i in range(10):
                vec = embedding_service.embed(f"Memory {i} in {ns}")
                database.insert(
                    f"Memory {i} in {ns}",
                    vec,
                    namespace=ns,
                )

        # Export only ns-b
        output_path = temp_storage / "filtered_export.parquet"
        result = database.export_to_parquet(output_path, namespace="ns-b")

        # Verify only ns-b records exported
        assert result["rows_exported"] == 10

        # Verify namespace in file
        table = pq.read_table(output_path)
        assert table.num_rows == 10
        namespaces = table.column("namespace").to_pylist()
        assert all(ns == "ns-b" for ns in namespaces)

    def test_export_creates_parent_directory(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that export creates parent directories if needed.

        Should handle nested paths gracefully.
        """
        # Insert some data
        vec = embedding_service.embed("Test memory")
        database.insert("Test memory", vec)

        # Export to nested path
        nested_path = temp_storage / "backups" / "2024" / "export.parquet"
        result = database.export_to_parquet(nested_path)

        # Verify parent directories were created
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert result["rows_exported"] == 1


class TestParquetImport:
    """Tests for Parquet import."""

    def test_import_from_parquet_basic(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test basic import from Parquet file.

        Export, clear database, then import and verify data restored.
        """
        # Insert test data
        original_ids = []
        records = []
        for i in range(30):
            vec = embedding_service.embed(f"Import test memory {i}")
            records.append({
                "content": f"Memory {i} to restore",
                "vector": vec,
                "namespace": "import-test",
                "tags": [f"tag-{i}"],
                "importance": 0.7,
            })

        database.insert_batch(records)

        # Get original data for comparison
        original_data = database.get_all(namespace="import-test")
        original_contents = {r["content"] for r in original_data}

        # Export
        export_path = temp_storage / "import_basic.parquet"
        database.export_to_parquet(export_path, namespace="import-test")

        # Clear database by deleting all in namespace
        database.delete_by_namespace("import-test")
        assert database.count(namespace="import-test") == 0

        # Import
        result = database.import_from_parquet(export_path)

        # Verify import result
        assert result["rows_imported"] == 30
        assert result["source"] == str(export_path)

        # Verify records restored (note: IDs will be different)
        restored_data = database.get_all(namespace="import-test")
        assert len(restored_data) == 30

        # Verify content matches (IDs will be regenerated)
        restored_contents = {r["content"] for r in restored_data}
        assert restored_contents == original_contents

    def test_import_from_parquet_namespace_override(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test namespace override during import.

        All imported records should be placed in the override namespace.
        """
        # Insert to namespace "original"
        records = []
        for i in range(20):
            vec = embedding_service.embed(f"Override test {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
                "namespace": "original",
            })

        database.insert_batch(records)

        # Export
        export_path = temp_storage / "override_test.parquet"
        database.export_to_parquet(export_path, namespace="original")

        # Import with namespace override to "imported"
        result = database.import_from_parquet(
            export_path,
            namespace_override="imported",
        )

        # Verify import succeeded
        assert result["rows_imported"] == 20

        # Verify all records in "imported" namespace
        imported_data = database.get_all(namespace="imported")
        assert len(imported_data) == 20
        assert all(r["namespace"] == "imported" for r in imported_data)

        # Verify "original" namespace is empty (we only imported, didn't re-insert)
        # Actually, original data is still there from the insert
        original_count = database.count(namespace="original")
        assert original_count == 20  # Original data still exists

    def test_import_from_parquet_returns_correct_count(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that import returns correct record count.

        Count returned should match number of records imported.
        """
        # Insert and export 15 records
        records = []
        for i in range(15):
            vec = embedding_service.embed(f"Count import {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        database.insert_batch(records)

        export_path = temp_storage / "count_import.parquet"
        database.export_to_parquet(export_path)

        # Clear and import
        database.delete_by_namespace("default")
        result = database.import_from_parquet(export_path)

        # Verify count
        assert result["rows_imported"] == 15
        assert database.count(namespace="default") == 15

    def test_import_nonexistent_file_raises_error(
        self, database: Database, temp_storage: Path
    ) -> None:
        """Test importing from nonexistent file raises StorageError.

        Should provide clear error message about missing file.
        """
        nonexistent_path = temp_storage / "does_not_exist.parquet"

        with pytest.raises(StorageError) as exc_info:
            database.import_from_parquet(nonexistent_path)

        assert "not found" in str(exc_info.value).lower()

    def test_import_regenerates_ids(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that import regenerates IDs to avoid conflicts.

        Imported records should have new IDs, not the exported ones.
        """
        # Insert original data
        records = []
        for i in range(10):
            vec = embedding_service.embed(f"ID test {i}")
            records.append({
                "content": f"Memory {i}",
                "vector": vec,
            })

        original_ids = database.insert_batch(records)

        # Export
        export_path = temp_storage / "id_test.parquet"
        database.export_to_parquet(export_path)

        # Import (without clearing - tests ID conflict avoidance)
        database.import_from_parquet(
            export_path,
            namespace_override="imported",  # Different namespace to avoid confusion
        )

        # Get imported data
        imported_data = database.get_all(namespace="imported")
        imported_ids = [r["id"] for r in imported_data]

        # Verify IDs are different
        assert len(set(imported_ids) & set(original_ids)) == 0


class TestBackupRestoreRoundtrip:
    """Tests for complete backup/restore cycle."""

    def test_roundtrip_preserves_all_fields(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test that export/import preserves all memory fields.

        All fields (content, namespace, tags, importance, metadata, etc.)
        should be preserved through export/import cycle.
        """
        # Insert memory with all fields populated
        vec = embedding_service.embed("Comprehensive field test")
        original_memory = {
            "content": "Comprehensive field test memory",
            "vector": vec,
            "namespace": "field-test",
            "tags": ["tag1", "tag2", "tag3"],
            "importance": 0.85,
            "source": "test-source",
            "metadata": {
                "key1": "value1",
                "key2": 42,
                "nested": {"foo": "bar"},
            },
        }

        memory_id = database.insert(**original_memory)

        # Get full record for comparison
        original_record = database.get(memory_id)

        # Export
        export_path = temp_storage / "roundtrip_fields.parquet"
        database.export_to_parquet(export_path, namespace="field-test")

        # Create new database for clean import
        new_db_path = temp_storage / "new-db"
        new_db = Database(new_db_path)
        new_db.connect()

        try:
            # Import
            new_db.import_from_parquet(
                export_path,
                namespace_override="field-test",  # Maintain namespace
            )

            # Get imported records
            imported_records = new_db.get_all(namespace="field-test")
            assert len(imported_records) == 1

            imported_record = imported_records[0]

            # Verify all fields match (except ID which is regenerated)
            assert imported_record["content"] == original_record["content"]
            assert imported_record["namespace"] == original_record["namespace"]
            assert set(imported_record["tags"]) == set(original_record["tags"])
            assert imported_record["importance"] == original_record["importance"]
            assert imported_record["source"] == original_record["source"]

            # Verify metadata (handle potential JSON serialization)
            imported_meta = imported_record["metadata"]
            original_meta = original_record["metadata"]
            if isinstance(imported_meta, str):
                imported_meta = json.loads(imported_meta)
            if isinstance(original_meta, str):
                original_meta = json.loads(original_meta)
            assert imported_meta == original_meta

            # Verify vector is preserved (compare a few elements)
            # Note: Vectors are lists in the database
            original_vec = original_record["vector"]
            imported_vec = imported_record["vector"]
            assert len(original_vec) == len(imported_vec)
            # Check first and last elements
            assert abs(original_vec[0] - imported_vec[0]) < 1e-6
            assert abs(original_vec[-1] - imported_vec[-1]) < 1e-6

        finally:
            new_db.close()

    def test_roundtrip_multiple_namespaces(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test roundtrip with multiple namespaces.

        Export all data (multiple namespaces), import to new database,
        and verify all namespaces are preserved.
        """
        # Insert data in multiple namespaces
        namespace_data = {
            "default": 10,
            "project-a": 15,
            "project-b": 20,
        }

        for ns, count in namespace_data.items():
            records = []
            for i in range(count):
                vec = embedding_service.embed(f"Memory {i} in {ns}")
                records.append({
                    "content": f"Memory {i} in {ns}",
                    "vector": vec,
                    "namespace": ns,
                })
            database.insert_batch(records)

        # Export all data
        export_path = temp_storage / "multi_namespace.parquet"
        database.export_to_parquet(export_path, namespace=None)  # Export all

        # Create new database
        new_db_path = temp_storage / "new-db-multi"
        new_db = Database(new_db_path)
        new_db.connect()

        try:
            # Import
            new_db.import_from_parquet(export_path)

            # Verify all namespaces present
            namespaces = new_db.get_namespaces()
            assert set(namespaces) == set(namespace_data.keys())

            # Verify counts per namespace
            for ns, expected_count in namespace_data.items():
                actual_count = new_db.count(namespace=ns)
                assert actual_count == expected_count

        finally:
            new_db.close()

    def test_roundtrip_large_dataset_performance(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test roundtrip with larger dataset for performance verification.

        Ensures export/import works efficiently with more realistic data size.
        """
        # Insert 500 records
        records = []
        for i in range(500):
            vec = embedding_service.embed(f"Large dataset memory {i}")
            records.append({
                "content": f"Memory {i} with some longer content for realism",
                "vector": vec,
                "tags": [f"tag-{i % 10}", f"category-{i % 5}"],
            })

        database.insert_batch(records)

        # Export
        export_path = temp_storage / "large_dataset.parquet"
        export_result = database.export_to_parquet(export_path)
        assert export_result["rows_exported"] == 500

        # Create new database
        new_db_path = temp_storage / "new-db-large"
        new_db = Database(new_db_path)
        new_db.connect()

        try:
            # Import
            import_result = new_db.import_from_parquet(export_path)
            assert import_result["rows_imported"] == 500

            # Verify count
            assert new_db.count() == 500

        finally:
            new_db.close()

    def test_roundtrip_empty_metadata(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test roundtrip with empty metadata field.

        Verifies that empty metadata is handled correctly.
        """
        # Insert memory with empty metadata
        vec = embedding_service.embed("Empty metadata test")
        database.insert(
            "Empty metadata test",
            vec,
            metadata={},  # Empty dict
        )

        # Export
        export_path = temp_storage / "empty_metadata.parquet"
        database.export_to_parquet(export_path)

        # Create new database and import
        new_db_path = temp_storage / "new-db-empty-meta"
        new_db = Database(new_db_path)
        new_db.connect()

        try:
            new_db.import_from_parquet(export_path)

            # Verify import
            records = new_db.get_all()
            assert len(records) == 1
            assert records[0]["metadata"] == {} or records[0]["metadata"] == "{}"

        finally:
            new_db.close()

    def test_roundtrip_special_characters_in_content(
        self, database: Database, embedding_service: EmbeddingService, temp_storage: Path
    ) -> None:
        """Test roundtrip with special characters in content.

        Ensures proper encoding/decoding of special characters.
        """
        # Insert memories with special characters
        special_contents = [
            "Content with 'single quotes' and \"double quotes\"",
            "Unicode characters: 日本語, emoji 🚀, symbols ±∞",
            "Newlines\nand\ttabs",
            "SQL-like: SELECT * FROM memories WHERE id = '123'",
        ]

        for content in special_contents:
            vec = embedding_service.embed(content)
            database.insert(content, vec)

        # Export
        export_path = temp_storage / "special_chars.parquet"
        database.export_to_parquet(export_path)

        # Create new database and import
        new_db_path = temp_storage / "new-db-special"
        new_db = Database(new_db_path)
        new_db.connect()

        try:
            new_db.import_from_parquet(export_path)

            # Verify all content preserved
            records = new_db.get_all()
            restored_contents = {r["content"] for r in records}
            assert restored_contents == set(special_contents)

        finally:
            new_db.close()
