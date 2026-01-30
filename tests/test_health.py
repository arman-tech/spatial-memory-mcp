"""Tests for health check infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from spatial_memory.core.health import HealthChecker, HealthStatus


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_health_check_all_passing(self) -> None:
        """Test health check when all components are healthy."""
        # Mock database
        mock_db = Mock()
        mock_db.count.return_value = 100

        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        # Create temp directory for storage
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create health checker
            checker = HealthChecker(
                database=mock_db,
                embeddings=mock_embeddings,
                storage_path=storage_path,
            )

            # Get health report
            report = checker.get_health_report()

            # Verify overall status is healthy
            assert report.status == HealthStatus.HEALTHY
            assert len(report.checks) == 3

            # Verify all individual checks are healthy
            for check in report.checks:
                assert check.status == HealthStatus.HEALTHY

            # Verify readiness and liveness
            assert checker.is_ready() is True
            assert checker.is_alive() is True

    def test_health_check_database_degraded(self) -> None:
        """Test health check when database is slow but operational."""
        # Mock slow database
        mock_db = Mock()

        def slow_count() -> int:
            import time
            time.sleep(6)  # Simulate slow operation (> 5s threshold)
            return 100

        mock_db.count = slow_count

        # Mock healthy embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        # Create health checker
        checker = HealthChecker(
            database=mock_db,
            embeddings=mock_embeddings,
        )

        # Get health report
        report = checker.get_health_report()

        # Overall status should be degraded
        assert report.status == HealthStatus.DEGRADED

        # Database check should be degraded
        db_check = next(c for c in report.checks if c.name == "database")
        assert db_check.status == HealthStatus.DEGRADED
        assert db_check.latency_ms is not None
        assert db_check.latency_ms > 5000

        # System should still be ready (degraded is acceptable)
        assert checker.is_ready() is True

    def test_health_check_embeddings_failed(self) -> None:
        """Test health check when embeddings service fails."""
        # Mock healthy database
        mock_db = Mock()
        mock_db.count.return_value = 100

        # Mock failing embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed.side_effect = RuntimeError("Model failed to load")

        # Create health checker
        checker = HealthChecker(
            database=mock_db,
            embeddings=mock_embeddings,
        )

        # Get health report
        report = checker.get_health_report()

        # Overall status should be unhealthy
        assert report.status == HealthStatus.UNHEALTHY

        # Embeddings check should be unhealthy
        embeddings_check = next(c for c in report.checks if c.name == "embeddings")
        assert embeddings_check.status == HealthStatus.UNHEALTHY
        assert "Model failed to load" in embeddings_check.message

        # System should not be ready
        assert checker.is_ready() is False

    def test_is_ready_requires_all_checks(self) -> None:
        """Test that readiness requires all critical checks to pass."""
        # Mock database failure
        mock_db = Mock()
        mock_db.count.side_effect = Exception("Database connection failed")

        # Mock healthy embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        # Create health checker
        checker = HealthChecker(
            database=mock_db,
            embeddings=mock_embeddings,
        )

        # System should not be ready when database is unhealthy
        assert checker.is_ready() is False

    def test_is_alive_basic(self) -> None:
        """Test that is_alive returns true when process is running."""
        # Create checker with no dependencies
        checker = HealthChecker()

        # Should always return true for basic liveness
        assert checker.is_alive() is True

    def test_database_check_not_configured(self) -> None:
        """Test database check when database is not configured."""
        checker = HealthChecker(database=None)

        check = checker.check_database()

        assert check.status == HealthStatus.UNHEALTHY
        assert check.name == "database"
        assert "not configured" in check.message.lower()

    def test_embeddings_check_not_configured(self) -> None:
        """Test embeddings check when service is not configured."""
        checker = HealthChecker(embeddings=None)

        check = checker.check_embeddings()

        assert check.status == HealthStatus.UNHEALTHY
        assert check.name == "embeddings"
        assert "not configured" in check.message.lower()

    def test_storage_check_path_does_not_exist(self) -> None:
        """Test storage check when path does not exist."""
        non_existent_path = Path("/tmp/non_existent_path_12345")

        checker = HealthChecker(storage_path=non_existent_path)

        check = checker.check_storage()

        assert check.status == HealthStatus.UNHEALTHY
        assert check.name == "storage"
        assert "does not exist" in check.message.lower()

    def test_storage_check_not_writable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test storage check when path is not writable."""
        # Create a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Mock write_text to raise PermissionError
            original_write_text = Path.write_text

            def mock_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
                if ".health_check" in str(self):
                    raise PermissionError("Permission denied")
                return original_write_text(self, *args, **kwargs)

            monkeypatch.setattr(Path, "write_text", mock_write_text)

            checker = HealthChecker(storage_path=storage_path)
            check = checker.check_storage()

            assert check.status == HealthStatus.UNHEALTHY
            assert check.name == "storage"
            assert "not writable" in check.message.lower() or "permission" in check.message.lower()

    def test_storage_check_skipped_when_not_configured(self) -> None:
        """Test storage check is skipped when no path configured."""
        checker = HealthChecker(storage_path=None)

        check = checker.check_storage()

        assert check.status == HealthStatus.HEALTHY
        assert check.name == "storage"
        assert "skipped" in check.message.lower()

    def test_embeddings_dimension_mismatch(self) -> None:
        """Test embeddings check when vector dimension is wrong."""
        mock_embeddings = Mock()
        # Return wrong dimension
        mock_embeddings.embed.return_value = np.array([0.1] * 100, dtype=np.float32)
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        checker = HealthChecker(embeddings=mock_embeddings)

        check = checker.check_embeddings()

        assert check.status == HealthStatus.UNHEALTHY
        assert "dimension mismatch" in check.message.lower()

    def test_embeddings_check_degraded_slow(self) -> None:
        """Test embeddings check when service is slow."""
        mock_embeddings = Mock()

        def slow_embed(text: str) -> np.ndarray:
            import time
            time.sleep(11)  # Simulate slow operation (> 10s threshold)
            return np.array([0.1] * 384, dtype=np.float32)

        mock_embeddings.embed = slow_embed
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        checker = HealthChecker(embeddings=mock_embeddings)

        check = checker.check_embeddings()

        assert check.status == HealthStatus.DEGRADED
        assert check.latency_ms is not None
        assert check.latency_ms > 10000

    def test_health_report_includes_timestamp(self) -> None:
        """Test that health report includes timestamp."""
        checker = HealthChecker()

        report = checker.get_health_report()

        assert report.timestamp is not None
        assert isinstance(report.timestamp, object)  # datetime object

    def test_storage_check_successful_write_and_delete(self) -> None:
        """Test storage check successfully writes and deletes test file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            checker = HealthChecker(storage_path=storage_path)

            check = checker.check_storage()

            assert check.status == HealthStatus.HEALTHY

            # Verify test file was cleaned up
            test_file = storage_path / ".health_check"
            assert not test_file.exists()

    def test_is_ready_ignores_storage_check(self) -> None:
        """Test that is_ready ignores storage check failures."""
        # Mock healthy database
        mock_db = Mock()
        mock_db.count.return_value = 100

        # Mock healthy embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_embeddings.dimensions = 384
        mock_embeddings.model_name = "test-model"

        # Use non-existent storage path
        bad_storage = Path("/tmp/non_existent_path_12345")

        checker = HealthChecker(
            database=mock_db,
            embeddings=mock_embeddings,
            storage_path=bad_storage,
        )

        # Should still be ready even though storage check fails
        # because storage is not critical for readiness
        assert checker.is_ready() is True
