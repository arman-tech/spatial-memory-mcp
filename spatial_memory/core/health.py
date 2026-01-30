"""Health check infrastructure for Spatial Memory MCP Server."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatial_memory.core.database import Database

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str
    latency_ms: float | None = None


@dataclass
class HealthReport:
    """Aggregate health report for system components."""

    status: HealthStatus
    checks: list[CheckResult]
    timestamp: datetime


class HealthChecker:
    """Health checker for system components.

    This class performs health checks on the database, embeddings service,
    and storage system to ensure the server is ready to accept traffic.
    """

    def __init__(
        self,
        database: Database | None = None,
        embeddings: Any = None,
        storage_path: Path | None = None,
    ) -> None:
        """Initialize health checker.

        Args:
            database: Database instance to check (optional).
            embeddings: Embedding service to check (optional).
            storage_path: Storage path to check for writability (optional).
        """
        self.database = database
        self.embeddings = embeddings
        self.storage_path = storage_path

    def check_database(self) -> CheckResult:
        """Check database connectivity and basic operations.

        Returns:
            CheckResult with database health status.
        """
        if self.database is None:
            return CheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database not configured",
            )

        try:
            start_time = time.perf_counter()

            # Try to get row count to verify database is operational
            count = self.database.count()
            latency = (time.perf_counter() - start_time) * 1000

            # Check if database is degraded based on latency
            if latency > 5000:  # 5 seconds
                return CheckResult(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message=f"Database operational but slow ({count} rows)",
                    latency_ms=latency,
                )

            return CheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message=f"Database operational ({count} rows)",
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return CheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)[:100]}",
            )

    def check_embeddings(self) -> CheckResult:
        """Check embedding service is functional.

        Returns:
            CheckResult with embeddings service health status.
        """
        if self.embeddings is None:
            return CheckResult(
                name="embeddings",
                status=HealthStatus.UNHEALTHY,
                message="Embeddings service not configured",
            )

        try:
            start_time = time.perf_counter()

            # Try a test embedding
            test_vector = self.embeddings.embed("health check test")
            latency = (time.perf_counter() - start_time) * 1000

            # Verify vector is correct shape
            if len(test_vector) != self.embeddings.dimensions:
                return CheckResult(
                    name="embeddings",
                    status=HealthStatus.UNHEALTHY,
                    message=(
                        f"Embedding dimension mismatch: "
                        f"got {len(test_vector)}, expected {self.embeddings.dimensions}"
                    ),
                    latency_ms=latency,
                )

            # Check if service is degraded based on latency
            if latency > 10000:  # 10 seconds
                return CheckResult(
                    name="embeddings",
                    status=HealthStatus.DEGRADED,
                    message=f"Embeddings operational but slow ({self.embeddings.model_name})",
                    latency_ms=latency,
                )

            return CheckResult(
                name="embeddings",
                status=HealthStatus.HEALTHY,
                message=(
                    f"Embeddings operational "
                    f"({self.embeddings.model_name}, {self.embeddings.dimensions}d)"
                ),
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"Embeddings health check failed: {e}")
            return CheckResult(
                name="embeddings",
                status=HealthStatus.UNHEALTHY,
                message=f"Embeddings error: {str(e)[:100]}",
            )

    def check_storage(self) -> CheckResult:
        """Check storage path is writable.

        Returns:
            CheckResult with storage health status.
        """
        if self.storage_path is None:
            return CheckResult(
                name="storage",
                status=HealthStatus.HEALTHY,
                message="Storage check skipped (no path configured)",
            )

        try:
            # Check if directory exists
            if not self.storage_path.exists():
                return CheckResult(
                    name="storage",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Storage path does not exist: {self.storage_path}",
                )

            # Try to write and delete a test file
            test_file = self.storage_path / ".health_check"
            try:
                test_file.write_text("health check", encoding="utf-8")
                test_file.unlink()
            except Exception as e:
                return CheckResult(
                    name="storage",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Storage path not writable: {e}",
                )

            return CheckResult(
                name="storage",
                status=HealthStatus.HEALTHY,
                message=f"Storage writable ({self.storage_path})",
            )

        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return CheckResult(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Storage error: {str(e)[:100]}",
            )

    def get_health_report(self) -> HealthReport:
        """Run all checks and return aggregate report.

        Returns:
            HealthReport with all check results.
        """
        checks: list[CheckResult] = []

        # Run all configured checks
        if self.database is not None:
            checks.append(self.check_database())

        if self.embeddings is not None:
            checks.append(self.check_embeddings())

        if self.storage_path is not None:
            checks.append(self.check_storage())

        # Determine overall status
        # If any check is unhealthy, overall is unhealthy
        # If any check is degraded, overall is degraded
        # Otherwise, healthy
        overall_status = HealthStatus.HEALTHY
        for check in checks:
            if check.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif check.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        return HealthReport(
            status=overall_status,
            checks=checks,
            timestamp=datetime.now(timezone.utc),
        )

    def is_ready(self) -> bool:
        """Check if system can accept traffic (readiness probe).

        All critical checks (database and embeddings) must pass.

        Returns:
            True if system is ready, False otherwise.
        """
        report = self.get_health_report()

        # All checks must be at least degraded (not unhealthy)
        for check in report.checks:
            # Storage is optional, skip it
            if check.name == "storage":
                continue

            if check.status == HealthStatus.UNHEALTHY:
                return False

        return True

    def is_alive(self) -> bool:
        """Check if process is alive (liveness probe).

        Basic process health check.

        Returns:
            True if process is alive, False otherwise.
        """
        # Basic check - if we can run this code, we're alive
        # In a more sophisticated implementation, this could check
        # for deadlocks, stuck threads, etc.
        return True
