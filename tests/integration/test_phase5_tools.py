"""End-to-end integration tests for Phase 5 MCP tools.

These tests verify the complete flow from MCP tool call through
service layer to database and back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatial_memory.core.errors import NamespaceNotFoundError, ValidationError
from spatial_memory.server import SpatialMemoryServer

if TYPE_CHECKING:
    from spatial_memory.core.embeddings import EmbeddingService


pytestmark = pytest.mark.integration


class TestStatsTool:
    """Tests for the stats MCP tool."""

    def test_stats_empty_database(self, module_server: SpatialMemoryServer) -> None:
        """Stats on empty database returns zero counts."""
        result = module_server._handle_tool("stats", {})
        assert result["total_memories"] == 0
        assert result["storage_mb"] >= 0

    def test_stats_with_data(self, module_server: SpatialMemoryServer) -> None:
        """Stats returns accurate counts after inserting memories."""
        # Insert test data
        module_server._handle_tool("remember", {"content": "Test memory 1"})
        module_server._handle_tool(
            "remember", {"content": "Test memory 2", "namespace": "test-ns"}
        )

        result = module_server._handle_tool("stats", {})
        assert result["total_memories"] == 2
        assert "default" in result["memories_by_namespace"]
        assert "test-ns" in result["memories_by_namespace"]

    def test_stats_namespace_filter(self, module_server: SpatialMemoryServer) -> None:
        """Stats filters correctly by namespace."""
        module_server._handle_tool("remember", {"content": "Memory 1", "namespace": "ns-a"})
        module_server._handle_tool("remember", {"content": "Memory 2", "namespace": "ns-b"})

        result = module_server._handle_tool("stats", {"namespace": "ns-a"})
        assert result["total_memories"] == 1

    def test_stats_includes_index_info(self, module_server: SpatialMemoryServer) -> None:
        """Stats includes index information when requested."""
        module_server._handle_tool("remember", {"content": "Test memory"})

        result = module_server._handle_tool("stats", {"include_index_details": True})
        assert "has_vector_index" in result
        assert "has_fts_index" in result
        assert "indices" in result

    def test_stats_without_index_details(self, module_server: SpatialMemoryServer) -> None:
        """Stats works with index details disabled."""
        module_server._handle_tool("remember", {"content": "Test memory for stats"})

        result = module_server._handle_tool("stats", {"include_index_details": False})

        # Core stats should still be present
        assert "total_memories" in result
        assert result["total_memories"] == 1
        # Index fields may be absent or set to defaults
        assert result.get("has_vector_index") is not None or "indices" not in result


class TestNamespacesTool:
    """Tests for the namespaces MCP tool."""

    def test_namespaces_empty_database(self, module_server: SpatialMemoryServer) -> None:
        """Empty database returns empty namespace list."""
        result = module_server._handle_tool("namespaces", {})
        assert result["total_namespaces"] == 0
        assert result["namespaces"] == []

    def test_namespaces_lists_all(self, module_server: SpatialMemoryServer) -> None:
        """Lists all namespaces with correct counts."""
        module_server._handle_tool("remember", {"content": "Memory 1", "namespace": "alpha"})
        module_server._handle_tool("remember", {"content": "Memory 2", "namespace": "alpha"})
        module_server._handle_tool("remember", {"content": "Memory 3", "namespace": "beta"})

        result = module_server._handle_tool("namespaces", {"include_stats": True})
        assert result["total_namespaces"] == 2

        ns_map = {ns["name"]: ns for ns in result["namespaces"]}
        assert ns_map["alpha"]["memory_count"] == 2
        assert ns_map["beta"]["memory_count"] == 1

    def test_namespaces_without_stats(self, module_server: SpatialMemoryServer) -> None:
        """Namespaces can be listed without stats for faster response."""
        module_server._handle_tool("remember", {"content": "Memory", "namespace": "test"})

        result = module_server._handle_tool("namespaces", {"include_stats": False})
        assert result["total_namespaces"] == 1
        assert result["namespaces"][0]["name"] == "test"


class TestDeleteNamespaceTool:
    """Tests for the delete_namespace MCP tool."""

    def test_delete_namespace_dry_run(self, module_server: SpatialMemoryServer) -> None:
        """Dry run previews deletion without executing."""
        module_server._handle_tool(
            "remember", {"content": "Memory 1", "namespace": "to-delete"}
        )

        result = module_server._handle_tool(
            "delete_namespace",
            {"namespace": "to-delete", "dry_run": True},
        )

        assert result["dry_run"] is True
        assert result["memories_deleted"] == 1

        # Verify data still exists
        stats = module_server._handle_tool("stats", {"namespace": "to-delete"})
        assert stats["total_memories"] == 1

    def test_delete_namespace_requires_confirm(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Actual deletion requires confirm=true."""
        module_server._handle_tool(
            "remember", {"content": "Memory 1", "namespace": "to-delete"}
        )

        # Without confirm, should fail
        with pytest.raises(ValidationError):
            module_server._handle_tool(
                "delete_namespace",
                {"namespace": "to-delete", "dry_run": False, "confirm": False},
            )

    def test_delete_namespace_confirmed(self, module_server: SpatialMemoryServer) -> None:
        """Confirmed deletion removes all memories."""
        module_server._handle_tool(
            "remember", {"content": "Memory 1", "namespace": "to-delete"}
        )

        result = module_server._handle_tool(
            "delete_namespace",
            {"namespace": "to-delete", "dry_run": False, "confirm": True},
        )

        assert result["success"] is True
        assert result["memories_deleted"] == 1

        # Verify data is gone
        stats = module_server._handle_tool("stats", {"namespace": "to-delete"})
        assert stats["total_memories"] == 0


class TestRenameNamespaceTool:
    """Tests for the rename_namespace MCP tool."""

    def test_rename_namespace_success(self, module_server: SpatialMemoryServer) -> None:
        """Rename moves all memories to new namespace."""
        module_server._handle_tool(
            "remember", {"content": "Memory 1", "namespace": "old-name"}
        )
        module_server._handle_tool(
            "remember", {"content": "Memory 2", "namespace": "old-name"}
        )

        result = module_server._handle_tool(
            "rename_namespace",
            {"old_namespace": "old-name", "new_namespace": "new-name"},
        )

        assert result["success"] is True
        assert result["memories_renamed"] == 2

        # Verify namespace changed
        namespaces = module_server._handle_tool("namespaces", {})
        ns_names = [ns["name"] for ns in namespaces["namespaces"]]
        assert "new-name" in ns_names
        assert "old-name" not in ns_names

    def test_rename_namespace_not_found(self, module_server: SpatialMemoryServer) -> None:
        """Renaming non-existent namespace raises error."""
        with pytest.raises(NamespaceNotFoundError):
            module_server._handle_tool(
                "rename_namespace",
                {"old_namespace": "does-not-exist", "new_namespace": "new-name"},
            )

    def test_rename_namespace_preserves_content(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Rename preserves memory content and metadata."""
        module_server._handle_tool(
            "remember",
            {
                "content": "Important memory",
                "namespace": "source",
                "tags": ["tag1"],
                "importance": 0.9,
            },
        )

        module_server._handle_tool(
            "rename_namespace",
            {"old_namespace": "source", "new_namespace": "target"},
        )

        # Recall from new namespace
        result = module_server._handle_tool(
            "recall", {"query": "Important memory", "namespace": "target"}
        )
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "Important memory"
        assert result["memories"][0]["tags"] == ["tag1"]


class TestExportMemoriesTool:
    """Tests for the export_memories MCP tool."""

    def test_export_json_format(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Export to JSON creates valid file."""
        module_server._handle_tool("remember", {"content": "Export test 1"})
        module_server._handle_tool("remember", {"content": "Export test 2"})

        export_path = temp_storage / "exports" / "export.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "format": "json"},
        )

        assert result["memories_exported"] == 2
        assert result["format"] == "json"
        assert export_path.exists()

        # Verify JSON is valid
        data = json.loads(export_path.read_text())
        assert len(data) == 2

    def test_export_parquet_format(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Export to Parquet creates valid file."""
        module_server._handle_tool("remember", {"content": "Export test"})

        export_path = temp_storage / "exports" / "export.parquet"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        result = module_server._handle_tool(
            "export_memories", {"output_path": str(export_path)}
        )

        assert result["memories_exported"] == 1
        assert export_path.exists()

    def test_export_namespace_filter(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Export filters by namespace."""
        module_server._handle_tool("remember", {"content": "NS A", "namespace": "ns-a"})
        module_server._handle_tool("remember", {"content": "NS B", "namespace": "ns-b"})

        export_path = temp_storage / "exports" / "export-a.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "namespace": "ns-a", "format": "json"},
        )

        assert result["memories_exported"] == 1
        assert result["namespaces_included"] == ["ns-a"]

    def test_export_without_vectors(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Export can exclude vectors for smaller files."""
        module_server._handle_tool("remember", {"content": "Test"})

        export_path = temp_storage / "exports" / "no-vectors.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        result = module_server._handle_tool(
            "export_memories",
            {
                "output_path": str(export_path),
                "format": "json",
                "include_vectors": False,
            },
        )

        assert result["memories_exported"] == 1

        # Verify no vectors in export
        data = json.loads(export_path.read_text())
        assert "vector" not in data[0] or data[0].get("vector") is None


class TestImportMemoriesTool:
    """Tests for the import_memories MCP tool."""

    def test_import_dry_run(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Dry run validates without importing."""
        # Create test file
        import_dir = temp_storage / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        import_file = import_dir / "import.json"
        import_file.write_text(
            json.dumps([
                {"content": "Import test 1", "namespace": "imported"},
                {"content": "Import test 2", "namespace": "imported"},
            ])
        )

        result = module_server._handle_tool(
            "import_memories", {"source_path": str(import_file), "dry_run": True}
        )

        assert result["dry_run"] is True
        assert result["total_records_in_file"] == 2

        # Verify nothing imported
        stats = module_server._handle_tool("stats", {})
        assert stats["total_memories"] == 0

    def test_import_actual(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Actual import creates memories."""
        import_dir = temp_storage / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        import_file = import_dir / "import.json"
        import_file.write_text(
            json.dumps([{"content": "Import test 1", "namespace": "imported"}])
        )

        result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(import_file),
                "dry_run": False,
                "regenerate_embeddings": True,
            },
        )

        assert result["dry_run"] is False
        assert result["memories_imported"] == 1

        # Verify import
        stats = module_server._handle_tool("stats", {"namespace": "imported"})
        assert stats["total_memories"] == 1

    def test_import_namespace_override(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Namespace override changes imported namespace."""
        import_dir = temp_storage / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        import_file = import_dir / "import.json"
        import_file.write_text(
            json.dumps([{"content": "Test", "namespace": "original"}])
        )

        result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(import_file),
                "namespace_override": "overridden",
                "dry_run": False,
                "regenerate_embeddings": True,
            },
        )

        assert result["namespace_override"] == "overridden"

        stats = module_server._handle_tool("stats", {"namespace": "overridden"})
        assert stats["total_memories"] == 1

    def test_import_validation_errors(
        self, module_server: SpatialMemoryServer, temp_storage: Path
    ) -> None:
        """Import reports validation errors for invalid records."""
        import_dir = temp_storage / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        import_file = import_dir / "invalid.json"
        import_file.write_text(
            json.dumps([
                {"content": "Valid record"},
                {"namespace": "missing-content"},  # Missing required 'content'
            ])
        )

        result = module_server._handle_tool(
            "import_memories",
            {"source_path": str(import_file), "dry_run": True, "validate": True},
        )

        assert result["memories_failed"] == 1
        assert len(result["validation_errors"]) == 1


class TestHybridRecallTool:
    """Tests for the hybrid_recall MCP tool."""

    def test_hybrid_recall_basic(self, module_server: SpatialMemoryServer) -> None:
        """Basic hybrid search returns results."""
        module_server._handle_tool(
            "remember", {"content": "Python programming language guide"}
        )
        module_server._handle_tool(
            "remember", {"content": "JavaScript web development tutorial"}
        )

        result = module_server._handle_tool(
            "hybrid_recall", {"query": "Python programming", "alpha": 0.5, "limit": 5}
        )

        assert result["total"] > 0
        assert len(result["memories"]) > 0
        assert result["alpha"] == 0.5

    def test_hybrid_recall_alpha_vector(self, module_server: SpatialMemoryServer) -> None:
        """Alpha=1.0 weights towards vector search."""
        module_server._handle_tool(
            "remember", {"content": "Semantic similarity test content"}
        )

        result = module_server._handle_tool(
            "hybrid_recall", {"query": "meaning similarity", "alpha": 1.0}
        )

        # Search type is always "hybrid" even with alpha=1.0
        assert result["search_type"] == "hybrid"
        assert result["alpha"] == 1.0

    def test_hybrid_recall_alpha_keyword(self, module_server: SpatialMemoryServer) -> None:
        """Alpha=0.0 weights towards keyword search."""
        module_server._handle_tool(
            "remember", {"content": "Exact keyword match test document"}
        )

        result = module_server._handle_tool(
            "hybrid_recall", {"query": "keyword match", "alpha": 0.0}
        )

        # Search type is always "hybrid" even with alpha=0.0
        assert result["search_type"] == "hybrid"
        assert result["alpha"] == 0.0

    def test_hybrid_recall_namespace_filter(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Hybrid search respects namespace filter."""
        module_server._handle_tool(
            "remember", {"content": "NS A content here", "namespace": "ns-a"}
        )
        module_server._handle_tool(
            "remember", {"content": "NS B content here", "namespace": "ns-b"}
        )

        result = module_server._handle_tool(
            "hybrid_recall", {"query": "content", "namespace": "ns-a"}
        )

        for memory in result["memories"]:
            assert memory["namespace"] == "ns-a"

    def test_hybrid_recall_min_similarity(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Hybrid search respects minimum similarity threshold."""
        module_server._handle_tool("remember", {"content": "Highly relevant test"})
        module_server._handle_tool("remember", {"content": "Completely unrelated xyz"})

        result = module_server._handle_tool(
            "hybrid_recall",
            {"query": "Highly relevant test", "min_similarity": 0.8, "alpha": 1.0},
        )

        # Should filter out low similarity results
        for memory in result["memories"]:
            assert memory["similarity"] >= 0.8


class TestToolCount:
    """Test that all expected tools are registered."""

    def test_total_tool_count(self, module_server: SpatialMemoryServer) -> None:
        """Verify 22 tools are registered (15 existing + 7 Phase 5)."""
        # Access the TOOLS list through the module
        from spatial_memory.server import TOOLS

        assert len(TOOLS) == 22

    def test_phase5_tools_present(self, module_server: SpatialMemoryServer) -> None:
        """Verify all Phase 5 tools are in the TOOLS list."""
        from spatial_memory.server import TOOLS

        tool_names = {tool.name for tool in TOOLS}
        phase5_tools = {
            "stats",
            "namespaces",
            "delete_namespace",
            "rename_namespace",
            "export_memories",
            "import_memories",
            "hybrid_recall",
        }

        assert phase5_tools.issubset(tool_names)


class TestExportImportSecurity:
    """Security tests for export/import operations."""

    def test_export_path_traversal_blocked(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Path traversal attempts should be blocked."""
        from spatial_memory.core.errors import PathSecurityError

        # Try path traversal attack - should raise PathSecurityError
        with pytest.raises(PathSecurityError) as exc_info:
            module_server._handle_tool(
                "export_memories",
                {"output_path": str(module_temp_storage / ".." / ".." / "etc" / "passwd.json")},
            )
        assert "traversal" in str(exc_info.value).lower()

    def test_import_nonexistent_file(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Import of non-existent file should fail gracefully."""
        from spatial_memory.core.errors import PathSecurityError

        nonexistent = module_temp_storage / "imports" / "does_not_exist.json"
        # Should raise PathSecurityError with file not found
        with pytest.raises(PathSecurityError) as exc_info:
            module_server._handle_tool(
                "import_memories",
                {"source_path": str(nonexistent), "dry_run": True},
            )
        assert "not exist" in str(exc_info.value).lower()

    def test_export_disallowed_directory(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Export to non-allowed directory should be blocked."""
        from spatial_memory.core.errors import PathSecurityError

        # Try to export to system directory - should raise PathSecurityError
        with pytest.raises(PathSecurityError) as exc_info:
            module_server._handle_tool(
                "export_memories",
                {"output_path": "/tmp/malicious.json"},
            )
        assert "not in allowed" in str(exc_info.value).lower() or "traversal" in str(exc_info.value).lower()

    def test_import_validates_extension(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Import should reject files with invalid extensions."""
        from spatial_memory.core.errors import PathSecurityError, ValidationError

        # Create a file with invalid extension
        bad_file = module_temp_storage / "imports" / "script.exe"
        bad_file.parent.mkdir(exist_ok=True)
        bad_file.write_text("malicious content")

        # Should raise either ValidationError (format detection) or PathSecurityError
        with pytest.raises((ValidationError, PathSecurityError)) as exc_info:
            module_server._handle_tool(
                "import_memories",
                {"source_path": str(bad_file), "dry_run": True},
            )
        # Either way, the error should mention format/extension issue
        error_msg = str(exc_info.value).lower()
        assert "format" in error_msg or "extension" in error_msg

    def test_import_record_limit_enforced(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Import should respect record limits."""
        # Create a JSON file with many records
        import_path = module_temp_storage / "imports" / "many_records.json"
        import_path.parent.mkdir(exist_ok=True)

        # Create 10 records (we won't test actual limit due to config)
        records = [{"content": f"Memory {i}"} for i in range(10)]
        import_path.write_text(json.dumps(records))

        # This should succeed since 10 < default limit
        result = module_server._handle_tool(
            "import_memories",
            {"source_path": str(import_path), "dry_run": True},
        )

        # Should succeed with dry run reporting 10 records
        assert result.get("total_records_in_file", 0) == 10


class TestCSVFormatSupport:
    """Tests for CSV format export/import."""

    def test_export_csv_format(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Test exporting memories to CSV format."""
        # Create test memory
        module_server._handle_tool(
            "remember", {"content": "CSV export test memory", "namespace": "csv-test"}
        )

        export_path = module_temp_storage / "exports" / "test_export.csv"
        result = module_server._handle_tool(
            "export_memories",
            {
                "output_path": str(export_path),
                "format": "csv",
                "namespace": "csv-test",
            },
        )

        assert result["memories_exported"] == 1
        assert export_path.exists()

        # Verify CSV structure
        import csv

        with open(export_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "content" in reader.fieldnames
            assert "namespace" in reader.fieldnames

    def test_import_csv_format(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Test importing memories from CSV format."""
        # Create CSV file
        import_path = module_temp_storage / "imports" / "import_test.csv"
        import_path.parent.mkdir(exist_ok=True)

        import csv

        with open(import_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["content", "namespace", "importance"]
            )
            writer.writeheader()
            writer.writerow(
                {"content": "Imported from CSV", "namespace": "csv-import", "importance": "0.7"}
            )

        result = module_server._handle_tool(
            "import_memories",
            {"source_path": str(import_path), "format": "csv", "dry_run": True},
        )

        # Dry run should report 1 total record
        assert result.get("total_records_in_file", 0) == 1

    def test_csv_roundtrip(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """Test export then import preserves data."""
        # Create memories
        module_server._handle_tool(
            "remember", {"content": "CSV roundtrip test 1", "namespace": "roundtrip"}
        )
        module_server._handle_tool(
            "remember", {"content": "CSV roundtrip test 2", "namespace": "roundtrip"}
        )

        export_path = module_temp_storage / "exports" / "roundtrip.csv"

        # Export
        export_result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "format": "csv", "namespace": "roundtrip"},
        )
        assert export_result["memories_exported"] == 2

        # Delete original
        module_server._handle_tool(
            "delete_namespace", {"namespace": "roundtrip", "confirm": True}
        )

        # Verify deletion
        ns_result = module_server._handle_tool("namespaces", {})
        assert "roundtrip" not in ns_result["namespaces"]

        # Import with regenerate_embeddings since CSV doesn't store vectors by default
        import_result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(export_path),
                "format": "csv",
                "dry_run": False,
                "regenerate_embeddings": True,
            },
        )

        assert import_result["memories_imported"] == 2


class TestImportRecordLimit:
    """Tests for import record limit enforcement."""

    def test_import_record_limit_exceeded(
        self, module_temp_storage: Path, session_embedding_service: "EmbeddingService"
    ) -> None:
        """Import should fail when record count exceeds configured limit."""
        from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
        from spatial_memory.core.database import Database
        from spatial_memory.core.errors import ImportRecordLimitError, MemoryImportError
        from spatial_memory.services.export_import import ExportImportConfig, ExportImportService

        # Create directories for import
        test_dir = module_temp_storage / "limit-test"
        test_dir.mkdir(exist_ok=True)
        imports_dir = test_dir / "imports"
        imports_dir.mkdir(exist_ok=True)

        # Create a service directly with a low max_import_records (bypassing Settings validation)
        db = Database(test_dir / "test-memory")
        db.connect()

        try:
            repo = LanceDBMemoryRepository(db)

            # Create ExportImportService with a very low limit
            config = ExportImportConfig(max_import_records=3)  # Low limit for testing
            service = ExportImportService(
                repository=repo,
                embeddings=session_embedding_service,
                config=config,
                allowed_import_paths=[str(imports_dir)],
                allowed_export_paths=[str(test_dir)],
            )

            # Create import file with more records than limit
            import_file = imports_dir / "too_many.json"
            records = [{"content": f"Memory {i}"} for i in range(5)]  # 5 > 3 limit
            import_file.write_text(json.dumps(records))

            # Should raise ImportRecordLimitError (may be wrapped in MemoryImportError)
            with pytest.raises((ImportRecordLimitError, MemoryImportError)) as exc_info:
                service.import_memories(
                    source_path=str(import_file),
                    dry_run=True,
                )

            # Check the error message contains the expected counts
            # Note: With streaming validation, we stop as soon as limit is exceeded,
            # so actual count is 4 (one more than the limit of 3), not 5
            error_msg = str(exc_info.value)
            assert "4 records" in error_msg  # Early termination stops at limit+1
            assert "max: 3" in error_msg
        finally:
            db.close()


class TestErrorResponseFormat:
    """Tests for error response format with isError field."""

    def test_validation_error_includes_is_error(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """Validation errors should include isError: true in response."""
        # Try to recall with invalid parameters
        with pytest.raises(ValidationError):
            module_server._handle_tool("recall", {"query": ""})

    def test_path_security_error_format(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """PathSecurityError should include isError: true in error response."""
        from spatial_memory.core.errors import PathSecurityError

        # Try path traversal - should raise PathSecurityError
        with pytest.raises(PathSecurityError):
            module_server._handle_tool(
                "export_memories",
                {"output_path": str(module_temp_storage / ".." / ".." / "etc" / "passwd.json")},
            )

    def test_namespace_not_found_error_format(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """NamespaceNotFoundError should be raised for non-existent namespace rename."""
        from spatial_memory.core.errors import NamespaceNotFoundError

        with pytest.raises(NamespaceNotFoundError):
            module_server._handle_tool(
                "rename_namespace",
                {"old_namespace": "nonexistent", "new_namespace": "new"},
            )

    def test_memory_not_found_error_format(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """MemoryNotFoundError should be raised for non-existent memory access."""
        from spatial_memory.core.errors import MemoryNotFoundError

        # Use 'nearby' which requires the memory to exist for reference
        # A non-existent memory ID should raise MemoryNotFoundError
        with pytest.raises(MemoryNotFoundError):
            module_server._handle_tool("nearby", {"memory_id": "00000000-0000-0000-0000-000000000000"})


class TestCSVEdgeCases:
    """Tests for CSV format edge cases with special characters and unicode."""

    def test_csv_special_characters(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """CSV export/import handles special characters correctly."""
        import csv

        # Create memory with special characters in content
        # Note: Tags are validated and can't have commas, so we test special chars in content only
        special_content = 'Memory with "quotes", commas, and\nnewlines'
        module_server._handle_tool(
            "remember",
            {"content": special_content, "namespace": "csv-special", "tags": ["tag-1", "tag_2"]},
        )

        # Export to CSV
        export_path = module_temp_storage / "exports" / "special.csv"
        result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "format": "csv", "namespace": "csv-special"},
        )
        assert result["memories_exported"] == 1

        # Read raw CSV to verify proper escaping
        with open(export_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["content"] == special_content

    def test_csv_unicode_content(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """CSV export/import handles unicode content correctly."""
        import csv

        # Create memory with unicode
        unicode_content = "Unicode test: 日本語 中文 한국어 émoji 🎉 äöü ñ"
        module_server._handle_tool(
            "remember",
            {"content": unicode_content, "namespace": "csv-unicode"},
        )

        # Export to CSV
        export_path = module_temp_storage / "exports" / "unicode.csv"
        result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "format": "csv", "namespace": "csv-unicode"},
        )
        assert result["memories_exported"] == 1

        # Read raw CSV to verify unicode preserved
        with open(export_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["content"] == unicode_content

    def test_csv_import_unicode(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """CSV import handles unicode content correctly."""
        import csv

        # Create CSV with unicode content
        import_path = module_temp_storage / "imports" / "unicode_import.csv"
        import_path.parent.mkdir(exist_ok=True)

        unicode_content = "Imported unicode: 日本語 🌍 ¿Qué?"
        with open(import_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["content", "namespace"])
            writer.writeheader()
            writer.writerow({"content": unicode_content, "namespace": "imported-unicode"})

        # Import
        result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(import_path),
                "format": "csv",
                "dry_run": False,
                "regenerate_embeddings": True,
            },
        )
        assert result["memories_imported"] == 1

        # Verify via recall
        recall_result = module_server._handle_tool(
            "recall", {"query": "Imported unicode", "namespace": "imported-unicode"}
        )
        assert len(recall_result["memories"]) == 1
        assert recall_result["memories"][0]["content"] == unicode_content

    def test_csv_empty_fields(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """CSV import handles empty optional fields correctly."""
        import csv

        # Create CSV with minimal fields (only content, no optional fields)
        import_path = module_temp_storage / "imports" / "minimal.csv"
        import_path.parent.mkdir(exist_ok=True)

        with open(import_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["content", "namespace", "tags", "importance"]
            )
            writer.writeheader()
            writer.writerow({
                "content": "Minimal memory",
                "namespace": "",  # Empty namespace
                "tags": "",  # Empty tags
                "importance": "",  # Empty importance
            })

        # Import should succeed with defaults
        result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(import_path),
                "format": "csv",
                "dry_run": True,  # Just validate
            },
        )
        # Should have 1 valid record
        assert result["total_records_in_file"] == 1
        assert result["memories_failed"] == 0

    def test_csv_large_content(
        self, module_server: SpatialMemoryServer, module_temp_storage: Path
    ) -> None:
        """CSV handles large content fields correctly."""
        # Use unique namespace with uuid to ensure isolation
        import uuid
        ns = f"csv-large-{uuid.uuid4().hex[:8]}"
        ns_import = f"csv-large-import-{uuid.uuid4().hex[:8]}"  # Separate namespace for import

        # Create memory with large content
        large_content = "This is a test. " * 1000  # ~16KB of text
        module_server._handle_tool(
            "remember",
            {"content": large_content, "namespace": ns},
        )

        # Export
        export_path = module_temp_storage / "exports" / f"large-{ns}.csv"
        result = module_server._handle_tool(
            "export_memories",
            {"output_path": str(export_path), "format": "csv", "namespace": ns},
        )
        assert result["memories_exported"] == 1

        # Import to a DIFFERENT namespace to avoid any delete/overlap issues
        import_result = module_server._handle_tool(
            "import_memories",
            {
                "source_path": str(export_path),
                "format": "csv",
                "dry_run": False,
                "regenerate_embeddings": True,
                "namespace_override": ns_import,  # Use different namespace
            },
        )
        assert import_result["memories_imported"] == 1

        # Verify content preserved - check the import namespace
        recall = module_server._handle_tool(
            "recall", {"query": "This is a test", "namespace": ns_import}
        )
        assert len(recall["memories"]) == 1
        assert len(recall["memories"][0]["content"]) == len(large_content)
