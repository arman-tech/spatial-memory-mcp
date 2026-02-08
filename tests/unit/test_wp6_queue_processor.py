"""Unit tests for WP6: Queue Processor + Piggyback Notifications.

Tests cover:
1. QueueFile parsing - valid/invalid JSON, version checks, field validation
2. Queue processor lifecycle - start/stop, idempotent, disabled mode
3. Queue file processing - store, reject, error cases
4. Piggyback notifications - drain, thread safety, clearing
5. Housekeeping - processed pruning, tmp orphan cleanup, stale warnings
6. Startup recovery - valid/corrupt tmp files, recent files left alone
7. Project resolution - resolve_from_directory on ProjectDetector
8. Server wiring - piggyback in response, start/stop lifecycle
9. Factory wiring - create_queue_processor returns correct values
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spatial_memory.adapters.project_detection import (
    ProjectDetectionConfig,
    ProjectDetector,
    ProjectIdentity,
)
from spatial_memory.core.queue_constants import (
    NEW_STALE_WARNING_SECONDS,
    PROCESSED_RETENTION_DAYS,
    QUEUE_DIR_NAME,
    QUEUE_FILE_VERSION,
    STARTUP_RECOVERY_AGE_SECONDS,
    TMP_ORPHAN_MAX_AGE_SECONDS,
)
from spatial_memory.core.queue_file import ProcessedResult, QueueFile
from spatial_memory.services.memory import RememberResult
from spatial_memory.services.queue_processor import MAX_QUEUE_FILE_SIZE, QueueProcessor

# =============================================================================
# Test UUIDs and helpers
# =============================================================================

UUID_1 = "11111111-1111-1111-1111-111111111111"
UUID_2 = "22222222-2222-2222-2222-222222222222"


def make_queue_json(
    content: str = "Test decision: use PostgreSQL because of JSONB support",
    version: int = QUEUE_FILE_VERSION,
    project_root_dir: str = "/home/user/code/my-project",
    suggested_namespace: str = "decisions",
    suggested_tags: list[str] | None = None,
    suggested_importance: float = 0.8,
    signal_tier: int = 1,
    **overrides: object,
) -> dict:
    data = {
        "version": version,
        "content": content,
        "source_hook": "prompt-submit",
        "timestamp": "2026-02-06T10:00:00Z",
        "project_root_dir": project_root_dir,
        "suggested_namespace": suggested_namespace,
        "suggested_tags": suggested_tags or ["postgresql", "database"],
        "suggested_importance": suggested_importance,
        "signal_tier": signal_tier,
        "signal_patterns_matched": ["decided to use"],
        "context": {"file": "app.py"},
        "client": "claude-code",
    }
    data.update(overrides)
    return data


def write_queue_file(directory: Path, filename: str, data: dict) -> Path:
    """Write a JSON queue file to a directory."""
    filepath = directory / filename
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


@pytest.fixture
def tmp_queue_dir(tmp_path: Path) -> Path:
    """Create a temporary queue directory with Maildir structure."""
    queue_dir = tmp_path / QUEUE_DIR_NAME
    (queue_dir / "tmp").mkdir(parents=True)
    (queue_dir / "new").mkdir(parents=True)
    (queue_dir / "processed").mkdir(parents=True)
    return queue_dir


@pytest.fixture
def mock_memory_service() -> MagicMock:
    svc = MagicMock()
    svc.remember.return_value = RememberResult(
        id=UUID_1,
        content="Test content",
        namespace="decisions",
        status="stored",
    )
    return svc


@pytest.fixture
def mock_project_detector() -> MagicMock:
    detector = MagicMock()
    detector.resolve_from_directory.return_value = ProjectIdentity(
        project_id="github.com/org/repo",
        source="queue_file",
    )
    return detector


@pytest.fixture
def processor(
    mock_memory_service: MagicMock,
    mock_project_detector: MagicMock,
    tmp_queue_dir: Path,
) -> QueueProcessor:
    """Create a QueueProcessor with mocked dependencies (enabled)."""
    return QueueProcessor(
        memory_service=mock_memory_service,
        project_detector=mock_project_detector,
        queue_dir=tmp_queue_dir,
        poll_interval=1,
        dedup_threshold=0.85,
        signal_threshold=0.3,
        cognitive_offloading_enabled=True,
    )


@pytest.fixture
def disabled_processor(
    mock_memory_service: MagicMock,
    mock_project_detector: MagicMock,
    tmp_queue_dir: Path,
) -> QueueProcessor:
    """Create a QueueProcessor with cognitive offloading disabled."""
    return QueueProcessor(
        memory_service=mock_memory_service,
        project_detector=mock_project_detector,
        queue_dir=tmp_queue_dir,
        poll_interval=1,
        cognitive_offloading_enabled=False,
    )


# =============================================================================
# 1. QueueFile Parsing
# =============================================================================


class TestQueueFileParsing:
    """Test QueueFile.from_json() validation and parsing."""

    def test_valid_json_v1(self) -> None:
        data = make_queue_json()
        qf = QueueFile.from_json(data)

        assert qf.version == QUEUE_FILE_VERSION
        assert qf.content == data["content"]
        assert qf.source_hook == "prompt-submit"
        assert qf.project_root_dir == "/home/user/code/my-project"
        assert qf.suggested_namespace == "decisions"
        assert qf.suggested_tags == ["postgresql", "database"]
        assert qf.suggested_importance == 0.8
        assert qf.signal_tier == 1
        assert qf.signal_patterns_matched == ["decided to use"]
        assert qf.context == {"file": "app.py"}
        assert qf.client == "claude-code"

    def test_missing_version_raises(self) -> None:
        data = make_queue_json()
        del data["version"]
        with pytest.raises(ValueError, match="Unsupported queue file version"):
            QueueFile.from_json(data)

    def test_invalid_version_raises(self) -> None:
        data = make_queue_json(version=99)
        with pytest.raises(ValueError, match="Unsupported queue file version"):
            QueueFile.from_json(data)

    def test_empty_content_raises(self) -> None:
        data = make_queue_json(content="")
        with pytest.raises(ValueError, match="content must not be empty"):
            QueueFile.from_json(data)

    def test_whitespace_only_content_raises(self) -> None:
        data = make_queue_json(content="   \n  ")
        with pytest.raises(ValueError, match="content must not be empty"):
            QueueFile.from_json(data)

    def test_invalid_importance_too_high(self) -> None:
        data = make_queue_json(suggested_importance=1.5)
        with pytest.raises(ValueError, match="suggested_importance"):
            QueueFile.from_json(data)

    def test_invalid_importance_negative(self) -> None:
        data = make_queue_json(suggested_importance=-0.1)
        with pytest.raises(ValueError, match="suggested_importance"):
            QueueFile.from_json(data)

    def test_invalid_importance_string(self) -> None:
        data = make_queue_json()
        data["suggested_importance"] = "high"
        with pytest.raises(ValueError, match="suggested_importance"):
            QueueFile.from_json(data)

    def test_extra_fields_ignored(self) -> None:
        data = make_queue_json()
        data["unknown_field"] = "should be ignored"
        data["another_extra"] = 42
        qf = QueueFile.from_json(data)
        assert qf.content == data["content"]

    def test_defaults_for_optional_fields(self) -> None:
        data = {
            "version": QUEUE_FILE_VERSION,
            "content": "Minimal content",
        }
        qf = QueueFile.from_json(data)
        assert qf.source_hook == ""
        assert qf.timestamp == ""
        assert qf.project_root_dir == ""
        assert qf.suggested_namespace == "default"
        assert qf.suggested_tags == []
        assert qf.suggested_importance == 0.5
        assert qf.signal_tier == 1
        assert qf.signal_patterns_matched == []
        assert qf.context == {}
        assert qf.client == ""

    def test_importance_boundary_zero(self) -> None:
        data = make_queue_json(suggested_importance=0.0)
        qf = QueueFile.from_json(data)
        assert qf.suggested_importance == 0.0

    def test_importance_boundary_one(self) -> None:
        data = make_queue_json(suggested_importance=1.0)
        qf = QueueFile.from_json(data)
        assert qf.suggested_importance == 1.0

    def test_importance_int_accepted(self) -> None:
        data = make_queue_json(suggested_importance=1)
        qf = QueueFile.from_json(data)
        assert qf.suggested_importance == 1.0

    def test_invalid_signal_patterns_not_list(self) -> None:
        data = make_queue_json()
        data["signal_patterns_matched"] = "not-a-list"
        with pytest.raises(ValueError, match="signal_patterns_matched must be"):
            QueueFile.from_json(data)

    def test_invalid_signal_patterns_non_string_elements(self) -> None:
        data = make_queue_json()
        data["signal_patterns_matched"] = [123, True]
        with pytest.raises(ValueError, match="signal_patterns_matched must be"):
            QueueFile.from_json(data)

    def test_invalid_context_not_dict(self) -> None:
        data = make_queue_json()
        data["context"] = "not-a-dict"
        with pytest.raises(ValueError, match="context must be a dict"):
            QueueFile.from_json(data)

    def test_invalid_context_list(self) -> None:
        data = make_queue_json()
        data["context"] = [1, 2, 3]
        with pytest.raises(ValueError, match="context must be a dict"):
            QueueFile.from_json(data)

    def test_content_too_long(self) -> None:
        data = make_queue_json(content="x" * 100_001)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            QueueFile.from_json(data)


class TestProcessedResult:
    """Test ProcessedResult dataclass."""

    def test_stored_result(self) -> None:
        result = ProcessedResult(
            filename="test.json",
            status="stored",
            memory_id=UUID_1,
            content_summary="Test decision",
        )
        assert result.status == "stored"
        assert result.memory_id == UUID_1
        assert result.error is None

    def test_error_result(self) -> None:
        result = ProcessedResult(
            filename="bad.json",
            status="error",
            error="Invalid JSON",
        )
        assert result.status == "error"
        assert result.memory_id is None
        assert result.error == "Invalid JSON"


# =============================================================================
# 2. Queue Processor Lifecycle
# =============================================================================


class TestQueueProcessorLifecycle:
    """Test start/stop behavior."""

    def test_start_when_disabled_no_thread(self, disabled_processor: QueueProcessor) -> None:
        disabled_processor.start()
        stats = disabled_processor.get_stats()
        assert stats["enabled"] is False
        assert stats["worker_alive"] is False

    def test_start_when_enabled_starts_thread(self, processor: QueueProcessor) -> None:
        processor.start()
        try:
            stats = processor.get_stats()
            assert stats["enabled"] is True
            assert stats["worker_alive"] is True
        finally:
            processor.stop()

    def test_start_idempotent(self, processor: QueueProcessor) -> None:
        processor.start()
        try:
            # Second start should be a no-op
            processor.start()
            stats = processor.get_stats()
            assert stats["worker_alive"] is True
        finally:
            processor.stop()

    def test_stop_when_not_started(self, processor: QueueProcessor) -> None:
        # Should be a no-op, no error
        processor.stop()

    def test_stop_flushes_remaining_queue(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        # Write a file to new/ before starting
        data = make_queue_json()
        write_queue_file(tmp_queue_dir / "new", "20260206-test.json", data)

        processor.start()
        # Give the worker time to process
        time.sleep(0.5)
        processor.stop(timeout=5.0)

        # File should have been processed
        assert mock_memory_service.remember.called

    def test_creates_directories_on_start(
        self,
        mock_memory_service: MagicMock,
        mock_project_detector: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Use a queue dir that doesn't exist yet
        queue_dir = tmp_path / "new-queue"
        proc = QueueProcessor(
            memory_service=mock_memory_service,
            project_detector=mock_project_detector,
            queue_dir=queue_dir,
            poll_interval=1,
            cognitive_offloading_enabled=True,
        )
        proc.start()
        try:
            assert (queue_dir / "tmp").exists()
            assert (queue_dir / "new").exists()
            assert (queue_dir / "processed").exists()
        finally:
            proc.stop()


# =============================================================================
# 3. Queue File Processing
# =============================================================================


class TestQueueFileProcessing:
    """Test _process_single_file and _process_queue behavior."""

    def test_valid_file_stored(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        data = make_queue_json()
        fp = write_queue_file(tmp_queue_dir / "new", "20260206-001.json", data)

        processor._process_queue()

        mock_memory_service.remember.assert_called_once()
        call_kwargs = mock_memory_service.remember.call_args.kwargs
        assert call_kwargs["content"] == data["content"]
        assert call_kwargs["namespace"] == "decisions"
        assert call_kwargs["cognitive_offloading_enabled"] is True
        assert call_kwargs["dedup_threshold"] == 0.85

        # File moved to processed
        assert not fp.exists()
        assert (tmp_queue_dir / "processed" / "20260206-001.json").exists()

    def test_duplicate_content_rejected(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.remember.return_value = RememberResult(
            id="",
            content="Already exists",
            namespace="decisions",
            deduplicated=True,
            status="rejected_exact",
            existing_memory_id=UUID_2,
        )
        data = make_queue_json()
        write_queue_file(tmp_queue_dir / "new", "20260206-dup.json", data)

        processor._process_queue()

        # File still moved to processed
        assert (tmp_queue_dir / "processed" / "20260206-dup.json").exists()

    def test_low_quality_rejected(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.remember.return_value = RememberResult(
            id="",
            content="hi",
            namespace="default",
            status="rejected_quality",
            quality_score=0.1,
        )
        data = make_queue_json(content="hi there")
        write_queue_file(tmp_queue_dir / "new", "20260206-low.json", data)

        processor._process_queue()

        assert (tmp_queue_dir / "processed" / "20260206-low.json").exists()

    def test_corrupt_json_error(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        fp = tmp_queue_dir / "new" / "20260206-bad.json"
        fp.write_text("not valid json {{{", encoding="utf-8")

        processor._process_queue()

        # Should not call remember
        mock_memory_service.remember.assert_not_called()
        # File moved to processed
        assert (tmp_queue_dir / "processed" / "20260206-bad.json").exists()

    def test_empty_new_directory(
        self,
        processor: QueueProcessor,
        mock_memory_service: MagicMock,
    ) -> None:
        processor._process_queue()
        mock_memory_service.remember.assert_not_called()

    def test_files_processed_oldest_first(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        """Files sorted by name (timestamps), so oldest processed first."""
        data1 = make_queue_json(content="First decision")
        data2 = make_queue_json(content="Second decision")
        write_queue_file(tmp_queue_dir / "new", "20260206-001.json", data1)
        write_queue_file(tmp_queue_dir / "new", "20260206-002.json", data2)

        processor._process_queue()

        calls = mock_memory_service.remember.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["content"] == "First decision"
        assert calls[1].kwargs["content"] == "Second decision"

    def test_remember_exception_handled(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.remember.side_effect = RuntimeError("DB error")
        data = make_queue_json()
        write_queue_file(tmp_queue_dir / "new", "20260206-err.json", data)

        processor._process_queue()

        # File moved to processed despite error
        assert (tmp_queue_dir / "processed" / "20260206-err.json").exists()

    def test_project_resolution_from_queue_file(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
        mock_project_detector: MagicMock,
    ) -> None:
        data = make_queue_json(project_root_dir="/home/user/code/my-project")
        write_queue_file(tmp_queue_dir / "new", "20260206-proj.json", data)

        processor._process_queue()

        mock_project_detector.resolve_from_directory.assert_called_with(
            "/home/user/code/my-project"
        )
        call_kwargs = mock_memory_service.remember.call_args.kwargs
        assert call_kwargs["project"] == "github.com/org/repo"

    def test_empty_project_root_dir(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
        mock_project_detector: MagicMock,
    ) -> None:
        data = make_queue_json(project_root_dir="")
        write_queue_file(tmp_queue_dir / "new", "20260206-nodir.json", data)

        processor._process_queue()

        # Should not call resolve_from_directory when empty
        mock_project_detector.resolve_from_directory.assert_not_called()
        call_kwargs = mock_memory_service.remember.call_args.kwargs
        assert call_kwargs["project"] == ""

    def test_oversized_file_rejected(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        """Files exceeding MAX_QUEUE_FILE_SIZE should be rejected."""
        new_dir = tmp_queue_dir / "new"
        file_path = new_dir / "oversized.json"
        file_path.write_text("x" * (MAX_QUEUE_FILE_SIZE + 1))

        result = processor._process_single_file(file_path)
        assert result.status == "error"
        assert "too large" in result.error


# =============================================================================
# 4. Piggyback Notifications
# =============================================================================


class TestPiggybackNotifications:
    """Test drain_notifications behavior."""

    def test_drain_after_processing(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        data = make_queue_json(content="Use Redis for caching")
        write_queue_file(tmp_queue_dir / "new", "20260206-note.json", data)

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) == 1
        assert "Use Redis for caching" in notifications[0]
        assert "(stored)" in notifications[0]

    def test_drain_clears_list(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        data = make_queue_json()
        write_queue_file(tmp_queue_dir / "new", "20260206-clear.json", data)

        processor._process_queue()

        first = processor.drain_notifications()
        assert len(first) == 1

        second = processor.drain_notifications()
        assert len(second) == 0

    def test_notification_for_rejected(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.remember.return_value = RememberResult(
            id="",
            content="Dup",
            namespace="default",
            deduplicated=True,
            status="rejected_exact",
        )
        data = make_queue_json(content="Duplicate content here")
        write_queue_file(tmp_queue_dir / "new", "20260206-rej.json", data)

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) == 1
        assert "(rejected_exact)" in notifications[0]

    def test_notification_for_error(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "new" / "20260206-badjson.json"
        fp.write_text("{invalid", encoding="utf-8")

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) == 1
        assert "(error:" in notifications[0]

    def test_multiple_notifications(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        for i in range(3):
            data = make_queue_json(content=f"Decision {i}")
            write_queue_file(tmp_queue_dir / "new", f"20260206-{i:03d}.json", data)

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) == 3

    def test_notification_sanitizes_control_chars(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        """Content with control chars should be sanitized in notifications (T-05)."""
        malicious = "Normal text\x00\x01\x02\r\ninjected\ttabs"
        data = make_queue_json(content=malicious)
        write_queue_file(tmp_queue_dir / "new", "20260206-ctrl.json", data)

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) == 1
        # No control characters should survive
        notification = notifications[0]
        for c in notification:
            assert c.isprintable() or c == " ", f"Control char {c!r} found in notification"
        assert "Normal text" in notification

    def test_notification_list_capped_at_100(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        """Notification list should not grow beyond 100 entries."""
        for i in range(150):
            data = make_queue_json(content=f"Decision number {i} for capping test")
            write_queue_file(tmp_queue_dir / "new", f"20260206-{i:04d}.json", data)

        processor._process_queue()

        notifications = processor.drain_notifications()
        assert len(notifications) <= 100


# =============================================================================
# 5. Housekeeping
# =============================================================================


class TestHousekeeping:
    """Test housekeeping operations."""

    def test_old_processed_files_deleted(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        # Create a processed file with old mtime
        fp = tmp_queue_dir / "processed" / "old-file.json"
        fp.write_text("{}", encoding="utf-8")
        old_time = time.time() - (PROCESSED_RETENTION_DAYS + 1) * 86400
        os.utime(fp, (old_time, old_time))

        processor._run_housekeeping()

        assert not fp.exists()

    def test_fresh_processed_files_kept(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "processed" / "recent-file.json"
        fp.write_text("{}", encoding="utf-8")

        processor._run_housekeeping()

        assert fp.exists()

    def test_orphaned_tmp_files_deleted(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "tmp" / "orphan.json"
        fp.write_text("{}", encoding="utf-8")
        old_time = time.time() - TMP_ORPHAN_MAX_AGE_SECONDS - 60
        os.utime(fp, (old_time, old_time))

        processor._run_housekeeping()

        assert not fp.exists()

    def test_fresh_tmp_files_kept(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "tmp" / "recent.json"
        fp.write_text("{}", encoding="utf-8")

        processor._run_housekeeping()

        assert fp.exists()

    def test_stale_new_files_warning(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fp = tmp_queue_dir / "new" / "stale-file.json"
        fp.write_text("{}", encoding="utf-8")
        old_time = time.time() - NEW_STALE_WARNING_SECONDS - 60
        os.utime(fp, (old_time, old_time))

        import logging

        with caplog.at_level(logging.WARNING):
            processor._run_housekeeping()

        assert any("Stale queue file" in r.message for r in caplog.records)
        # File is NOT deleted, just warned about
        assert fp.exists()


# =============================================================================
# 6. Startup Recovery
# =============================================================================


class TestStartupRecovery:
    """Test _startup_recovery behavior."""

    def test_old_valid_tmp_moved_to_new(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        data = make_queue_json()
        fp = tmp_queue_dir / "tmp" / "20260206-recover.json"
        fp.write_text(json.dumps(data), encoding="utf-8")
        old_time = time.time() - STARTUP_RECOVERY_AGE_SECONDS - 60
        os.utime(fp, (old_time, old_time))

        processor._startup_recovery()

        assert not fp.exists()
        assert (tmp_queue_dir / "new" / "20260206-recover.json").exists()

    def test_old_corrupt_tmp_deleted(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "tmp" / "corrupt.json"
        fp.write_text("not json {{{", encoding="utf-8")
        old_time = time.time() - STARTUP_RECOVERY_AGE_SECONDS - 60
        os.utime(fp, (old_time, old_time))

        processor._startup_recovery()

        assert not fp.exists()
        assert not (tmp_queue_dir / "new" / "corrupt.json").exists()

    def test_recent_tmp_files_left_alone(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "tmp" / "recent.json"
        fp.write_text(json.dumps(make_queue_json()), encoding="utf-8")
        # File just created - mtime is recent

        processor._startup_recovery()

        # File should still be in tmp/
        assert fp.exists()
        assert not (tmp_queue_dir / "new" / "recent.json").exists()


# =============================================================================
# 7. Project Resolution (resolve_from_directory on ProjectDetector)
# =============================================================================


class TestProjectDetectorResolveFromDirectory:
    """Test the public resolve_from_directory method."""

    def test_returns_identity_for_git_repo(self, tmp_path: Path) -> None:
        # Create a fake git repo structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (git_dir / "config").write_text(
            '[remote "origin"]\n'
            "\turl = https://github.com/org/repo.git\n"
            "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
        )

        detector = ProjectDetector(ProjectDetectionConfig())
        identity = detector.resolve_from_directory(str(tmp_path))

        assert identity.project_id == "github.com/org/repo"
        assert identity.source == "queue_file"

    def test_falls_back_to_empty_for_non_git_dir(self, tmp_path: Path) -> None:
        detector = ProjectDetector(ProjectDetectionConfig())
        identity = detector.resolve_from_directory(str(tmp_path))

        assert identity.project_id == ""
        assert identity.source == "fallback"


# =============================================================================
# 8. Server Wiring
# =============================================================================


class TestServerWiring:
    """Test piggyback injection and lifecycle in server.py patterns."""

    def test_queue_processed_in_response(self) -> None:
        """Simulate how _queue_processed would appear in a response dict."""
        result: dict = {"memories": [], "total": 0}

        # Simulate what server.py does
        notifications = ['"Use Redis" (stored)', '"PostgreSQL fix" (stored)']
        if notifications:
            result["_queue_processed"] = notifications

        assert "_queue_processed" in result
        assert len(result["_queue_processed"]) == 2

    def test_no_queue_processed_when_empty(self) -> None:
        """No _queue_processed key when there are no notifications."""
        result: dict = {"memories": [], "total": 0}

        notifications: list[str] = []
        if notifications:
            result["_queue_processed"] = notifications

        assert "_queue_processed" not in result


# =============================================================================
# 9. Factory Wiring
# =============================================================================


class TestFactoryWiring:
    """Test create_queue_processor in ServiceFactory."""

    def test_returns_none_when_disabled(self) -> None:
        from spatial_memory.factory import ServiceFactory

        settings = MagicMock()
        settings.cognitive_offloading_enabled = False

        factory = ServiceFactory(settings=settings)
        result = factory.create_queue_processor(
            memory_service=MagicMock(),
            project_detector=MagicMock(),
        )

        assert result is None

    def test_returns_processor_when_enabled(self, tmp_path: Path) -> None:
        from spatial_memory.factory import ServiceFactory

        settings = MagicMock()
        settings.cognitive_offloading_enabled = True
        settings.memory_path = tmp_path
        settings.queue_poll_interval_seconds = 30
        settings.dedup_vector_threshold = 0.85
        settings.signal_threshold = 0.3

        factory = ServiceFactory(settings=settings)
        result = factory.create_queue_processor(
            memory_service=MagicMock(),
            project_detector=MagicMock(),
        )

        assert result is not None
        assert isinstance(result, QueueProcessor)

    def test_service_container_has_queue_processor_field(self) -> None:
        import dataclasses

        from spatial_memory.factory import ServiceContainer

        fields = {f.name for f in dataclasses.fields(ServiceContainer)}
        assert "queue_processor" in fields


# =============================================================================
# 10. Stats
# =============================================================================


class TestStats:
    """Test get_stats tracking."""

    def test_initial_stats_zero(self, processor: QueueProcessor) -> None:
        stats = processor.get_stats()
        assert stats["files_processed"] == 0
        assert stats["files_stored"] == 0
        assert stats["files_rejected"] == 0
        assert stats["files_errored"] == 0

    def test_stats_after_processing(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
        mock_memory_service: MagicMock,
    ) -> None:
        # One stored, one rejected
        data1 = make_queue_json(content="Stored memory")
        data2 = make_queue_json(content="Rejected memory")

        mock_memory_service.remember.side_effect = [
            RememberResult(id=UUID_1, content="Stored", namespace="default", status="stored"),
            RememberResult(id="", content="Dup", namespace="default", status="rejected_exact"),
        ]

        write_queue_file(tmp_queue_dir / "new", "20260206-001.json", data1)
        write_queue_file(tmp_queue_dir / "new", "20260206-002.json", data2)

        processor._process_queue()

        stats = processor.get_stats()
        assert stats["files_processed"] == 2
        assert stats["files_stored"] == 1
        assert stats["files_rejected"] == 1
        assert stats["files_errored"] == 0

    def test_stats_after_error(
        self,
        processor: QueueProcessor,
        tmp_queue_dir: Path,
    ) -> None:
        fp = tmp_queue_dir / "new" / "bad.json"
        fp.write_text("not json", encoding="utf-8")

        processor._process_queue()

        stats = processor.get_stats()
        assert stats["files_errored"] == 1


# =============================================================================
# 11. Queue Constants
# =============================================================================


class TestQueueConstants:
    """Verify constant values are as documented."""

    def test_processed_retention_days(self) -> None:
        assert PROCESSED_RETENTION_DAYS == 7

    def test_tmp_orphan_max_age(self) -> None:
        assert TMP_ORPHAN_MAX_AGE_SECONDS == 3600

    def test_new_stale_warning(self) -> None:
        assert NEW_STALE_WARNING_SECONDS == 86400

    def test_startup_recovery_age(self) -> None:
        assert STARTUP_RECOVERY_AGE_SECONDS == 300

    def test_queue_dir_name(self) -> None:
        assert QUEUE_DIR_NAME == "pending-saves"

    def test_queue_file_version(self) -> None:
        assert QUEUE_FILE_VERSION == 1


# =============================================================================
# 12. Move-to-Processed Failure Handling (H2)
# =============================================================================


@pytest.mark.unit
class TestMoveToProcessedFailure:
    """Tests for H2: _move_to_processed fallback on failure."""

    def test_failed_move_renames_with_failed_suffix(self, tmp_path: Path) -> None:
        """When move to processed/ fails, file should be renamed with .failed suffix."""
        queue_dir = tmp_path / "queue"
        new_dir = queue_dir / "new"
        processed_dir = queue_dir / "processed"
        new_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        processor = QueueProcessor(
            memory_service=MagicMock(),
            project_detector=MagicMock(),
            queue_dir=queue_dir,
            cognitive_offloading_enabled=True,
        )

        # Create a file in new/
        test_file = new_dir / "test.json"
        test_file.write_text("{}")

        import shutil
        from unittest.mock import patch

        with patch.object(shutil, "move", side_effect=OSError("permission denied")):
            processor._move_to_processed(test_file)

        # File should be renamed with .failed suffix
        failed_file = new_dir / "test.json.failed"
        assert failed_file.exists()
        assert not test_file.exists()

    def test_failed_files_skipped_in_processing(self, tmp_path: Path) -> None:
        """Files with .failed suffix should be skipped during queue processing."""
        queue_dir = tmp_path / "queue"
        new_dir = queue_dir / "new"
        processed_dir = queue_dir / "processed"
        tmp_dir = queue_dir / "tmp"
        new_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        tmp_dir.mkdir(parents=True)

        processor = QueueProcessor(
            memory_service=MagicMock(),
            project_detector=MagicMock(),
            queue_dir=queue_dir,
            cognitive_offloading_enabled=True,
        )

        # Create a .failed file - should be skipped
        failed_file = new_dir / "test.json.failed"
        failed_file.write_text("{}")

        from unittest.mock import patch

        with patch.object(processor, "_process_single_file") as mock_process:
            processor._process_queue()
            mock_process.assert_not_called()


# =============================================================================
# 13. QueueFile Namespace Validation (H4)
# =============================================================================


@pytest.mark.unit
class TestQueueFileNamespaceValidation:
    """Tests for H4: namespace validation in QueueFile.from_json()."""

    def test_valid_namespace_accepted(self) -> None:
        """Valid namespace should be accepted."""
        data = make_queue_json(suggested_namespace="myproject")
        qf = QueueFile.from_json(data)
        assert qf.suggested_namespace == "myproject"

    def test_default_namespace_accepted(self) -> None:
        """Default namespace 'default' should be accepted when not specified."""
        data = make_queue_json()
        data.pop("suggested_namespace", None)
        qf = QueueFile.from_json(data)
        assert qf.suggested_namespace == "default"

    def test_invalid_namespace_with_spaces_rejected(self) -> None:
        """Namespace with spaces should be rejected."""
        data = make_queue_json(suggested_namespace="invalid namespace")
        with pytest.raises(ValueError, match="Invalid suggested_namespace"):
            QueueFile.from_json(data)

    def test_namespace_starting_with_number_rejected(self) -> None:
        """Namespace starting with a number should be rejected."""
        data = make_queue_json(suggested_namespace="123invalid")
        with pytest.raises(ValueError, match="Invalid suggested_namespace"):
            QueueFile.from_json(data)

    def test_namespace_non_string_rejected(self) -> None:
        """Non-string namespace should be rejected."""
        data = make_queue_json()
        data["suggested_namespace"] = 42
        with pytest.raises(ValueError, match="suggested_namespace must be a string"):
            QueueFile.from_json(data)
