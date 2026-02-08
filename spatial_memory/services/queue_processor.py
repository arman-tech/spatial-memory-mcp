"""Background queue processor for pending memory saves.

Follows the same daemon-thread pattern as DecayManager:
- __init__: threading primitives, config
- start(): idempotent, creates daemon thread
- stop(timeout): sets shutdown event, joins thread, logs stats
- _background_worker(): poll loop with event.wait(timeout)

Queue files are written by client-side hooks into a Maildir-style directory:
    pending-saves/
        tmp/       # Partial writes (crash-safe)
        new/       # Ready for processing
        processed/ # Completed (kept for debugging, pruned after 7 days)
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatial_memory.core.queue_constants import (
    NEW_STALE_WARNING_SECONDS,
    PROCESSED_RETENTION_DAYS,
    STARTUP_RECOVERY_AGE_SECONDS,
    TMP_ORPHAN_MAX_AGE_SECONDS,
)
from spatial_memory.core.queue_file import ProcessedResult, QueueFile

MAX_QUEUE_FILE_SIZE = 1_048_576  # 1MB - generous limit given 100KB content max

if TYPE_CHECKING:
    from spatial_memory.adapters.project_detection import ProjectDetector
    from spatial_memory.services.memory import MemoryService

logger = logging.getLogger(__name__)


class QueueProcessor:
    """Background queue processor for pending memory saves.

    Polls the Maildir-style queue directory for new files, processes them
    through dedup + quality gate via MemoryService.remember(), and moves
    processed files to the processed/ directory.

    Thread-safe piggyback notifications allow the server to append brief
    save summaries to the next MCP tool response.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        project_detector: ProjectDetector,
        queue_dir: Path,
        poll_interval: int = 30,
        dedup_threshold: float = 0.85,
        signal_threshold: float = 0.3,
        cognitive_offloading_enabled: bool = False,
    ) -> None:
        """Initialize the queue processor.

        Args:
            memory_service: Service for storing memories.
            project_detector: Detector for resolving project identity.
            queue_dir: Path to the Maildir-style queue directory.
            poll_interval: Seconds between queue polls.
            dedup_threshold: Vector similarity threshold for dedup.
            signal_threshold: Quality gate minimum score.
            cognitive_offloading_enabled: Feature flag - if False, no-op.
        """
        self._memory_service = memory_service
        self._project_detector = project_detector
        self._queue_dir = queue_dir
        self._poll_interval = poll_interval
        self._dedup_threshold = dedup_threshold
        self._signal_threshold = signal_threshold
        self._enabled = cognitive_offloading_enabled

        # Maildir subdirectories
        self._tmp_dir = queue_dir / "tmp"
        self._new_dir = queue_dir / "new"
        self._processed_dir = queue_dir / "processed"

        # Threading primitives
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Piggyback notifications (thread-safe via _lock)
        self._notifications: list[str] = []

        # Statistics
        self._stats_lock = threading.Lock()
        self._files_processed = 0
        self._files_stored = 0
        self._files_rejected = 0
        self._files_errored = 0

        self._housekeeping_counter = 0

    def start(self) -> None:
        """Start the background queue processor.

        If cognitive offloading is disabled, logs a debug message and returns.
        Safe to call multiple times - will only start if not already running.
        Thread-safe: uses self._lock to prevent duplicate worker threads from
        concurrent calls.
        """
        if not self._enabled:
            logger.debug("Cognitive offloading disabled, skipping queue processor")
            return

        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.debug("Queue processor already running")
                return

            # Create Maildir directory structure
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            self._new_dir.mkdir(parents=True, exist_ok=True)
            self._processed_dir.mkdir(parents=True, exist_ok=True)

            # Run startup recovery
            self._startup_recovery()

            # Start daemon thread
            self._shutdown_event.clear()
            self._worker_thread = threading.Thread(
                target=self._background_worker,
                name="queue-processor",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Queue processor started (poll_interval=%ds)", self._poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker gracefully.

        Signals shutdown and waits for the worker to finish. The worker
        performs a final queue drain before exiting.

        Args:
            timeout: Maximum seconds to wait for worker shutdown.
        """
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return

        logger.info("Stopping queue processor...")
        self._shutdown_event.set()

        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning("Queue processor did not stop within timeout")
        else:
            with self._stats_lock:
                logger.info(
                    "Queue processor stopped. Processed: %d, Stored: %d, Rejected: %d, Errors: %d",
                    self._files_processed,
                    self._files_stored,
                    self._files_rejected,
                    self._files_errored,
                )

    def get_stats(self) -> dict[str, Any]:
        """Get queue processor statistics.

        Returns:
            Dictionary with processing stats.
        """
        with self._stats_lock:
            stats = {
                "enabled": self._enabled,
                "files_processed": self._files_processed,
                "files_stored": self._files_stored,
                "files_rejected": self._files_rejected,
                "files_errored": self._files_errored,
                "worker_alive": (
                    self._worker_thread is not None and self._worker_thread.is_alive()
                ),
            }
        with self._lock:
            stats["pending_notifications"] = len(self._notifications)
        return stats

    def drain_notifications(self) -> list[str]:
        """Drain and return all pending piggyback notifications.

        Thread-safe: acquires lock, swaps pending notifications with
        empty list, returns old list.

        Returns:
            List of notification strings (e.g. '"Queue pattern decision" (stored)').
        """
        with self._lock:
            notifications = self._notifications
            self._notifications = []
            return notifications

    # =========================================================================
    # Background Worker
    # =========================================================================

    def _background_worker(self) -> None:
        """Background worker that polls the queue directory."""
        logger.debug("Queue processor worker started")

        while not self._shutdown_event.is_set():
            try:
                self._process_queue()
                self._housekeeping_counter += 1
                if self._housekeeping_counter >= 60:  # ~30min at 30s poll interval
                    self._run_housekeeping()
                    self._housekeeping_counter = 0
            except Exception:
                logger.error("Error in queue processor worker", exc_info=True)
                time.sleep(1.0)

            self._shutdown_event.wait(timeout=self._poll_interval)

        # Final drain on shutdown
        try:
            self._process_queue()
        except Exception:
            logger.error("Error in final queue drain", exc_info=True)

        logger.debug("Queue processor worker stopped")

    # =========================================================================
    # Queue Processing
    # =========================================================================

    def _process_queue(self) -> None:
        """Process all files in the new/ directory.

        Files are sorted by name (oldest first, since filenames start with
        timestamps) and processed sequentially.
        """
        if not self._new_dir.exists():
            return

        files = sorted(self._new_dir.iterdir())
        if not files:
            return

        for file_path in files:
            if self._shutdown_event.is_set():
                break
            if file_path.is_file() and not file_path.name.endswith(".failed"):
                result = self._process_single_file(file_path)
                self._record_result(result)

    def _process_single_file(self, file_path: Path) -> ProcessedResult:
        """Process a single queue file.

        1. Read and parse JSON
        2. Resolve project from project_root_dir
        3. Call memory_service.remember() with cognitive offloading params
        4. Move file to processed/

        Args:
            file_path: Path to the queue file in new/.

        Returns:
            ProcessedResult with outcome details.
        """
        filename = file_path.name
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_QUEUE_FILE_SIZE:
                logger.warning(
                    "Queue file %s too large (%d bytes, limit %d), skipping",
                    filename,
                    file_size,
                    MAX_QUEUE_FILE_SIZE,
                )
                self._move_to_processed(file_path)
                return ProcessedResult(
                    filename=filename,
                    status="error",
                    error=f"File too large: {file_size} bytes (limit {MAX_QUEUE_FILE_SIZE})",
                )
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            queue_file = QueueFile.from_json(data)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("Failed to parse queue file %s: %s", filename, e)
            self._move_to_processed(file_path)
            return ProcessedResult(
                filename=filename,
                status="error",
                content_summary="",
                error=str(e),
            )

        # Resolve project from directory
        project = ""
        if queue_file.project_root_dir:
            identity = self._project_detector.resolve_from_directory(queue_file.project_root_dir)
            project = identity.project_id

        # Store via memory service
        # Sanitize content summary: strip control chars to prevent prompt injection
        content_summary = "".join(
            c if c.isprintable() or c == " " else "" for c in queue_file.content[:50]
        )
        try:
            remember_result = self._memory_service.remember(
                content=queue_file.content,
                namespace=queue_file.suggested_namespace,
                tags=queue_file.suggested_tags or None,
                importance=queue_file.suggested_importance,
                metadata=queue_file.context if queue_file.context else None,
                project=project,
                cognitive_offloading_enabled=True,
                dedup_threshold=self._dedup_threshold,
                signal_threshold=self._signal_threshold,
            )
        except Exception as e:
            logger.error("Error storing queue file %s: %s", filename, e, exc_info=True)
            self._move_to_processed(file_path)
            return ProcessedResult(
                filename=filename,
                status="error",
                content_summary=content_summary,
                error=str(e),
            )

        # Map remember_result.status to our ProcessedResult status
        status = remember_result.status  # "stored", "rejected_exact", etc.

        self._move_to_processed(file_path)

        return ProcessedResult(
            filename=filename,
            status=status,
            memory_id=remember_result.id if status == "stored" else None,
            content_summary=content_summary,
        )

    @staticmethod
    def _safe_move(src: Path, dest_dir: Path) -> None:
        """Move a file to a directory, adding a UUID suffix on collision.

        Args:
            src: Source file path.
            dest_dir: Destination directory.
        """
        dest = dest_dir / src.name
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            dest = dest_dir / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"
        shutil.move(str(src), str(dest))

    def _move_to_processed(self, file_path: Path) -> None:
        """Move a file from new/ to processed/.

        If the move fails, renames the file with a .failed suffix to prevent
        infinite reprocessing. As a last resort, deletes the file.

        Args:
            file_path: Path to the file to move.
        """
        try:
            self._safe_move(file_path, self._processed_dir)
        except OSError as e:
            logger.error("Failed to move %s to processed/: %s", file_path.name, e)
            # Fallback: rename with .failed suffix to prevent infinite reprocessing
            try:
                failed_path = file_path.with_suffix(file_path.suffix + ".failed")
                file_path.rename(failed_path)
                logger.warning(
                    "Renamed %s to %s to prevent reprocessing",
                    file_path.name,
                    failed_path.name,
                )
            except OSError as rename_err:
                logger.error(
                    "Failed to rename %s: %s, attempting delete",
                    file_path.name,
                    rename_err,
                )
                try:
                    file_path.unlink()
                    logger.warning("Deleted %s to prevent infinite reprocessing", file_path.name)
                except OSError as del_err:
                    logger.critical(
                        "Cannot move, rename, or delete queue file %s: %s. "
                        "Manual intervention required.",
                        file_path.name,
                        del_err,
                    )

    def _record_result(self, result: ProcessedResult) -> None:
        """Record processing result in stats and notifications.

        Args:
            result: The processing outcome.
        """
        with self._stats_lock:
            self._files_processed += 1
            if result.status == "stored":
                self._files_stored += 1
            elif result.status == "error":
                self._files_errored += 1
            else:
                self._files_rejected += 1

        # Build piggyback notification
        if result.status == "stored":
            notification = f'"{result.content_summary}" (stored)'
        elif result.status == "error":
            notification = f'"{result.content_summary}" (error: {result.error})'
        else:
            notification = f'"{result.content_summary}" ({result.status})'

        with self._lock:
            if len(self._notifications) < 100:
                self._notifications.append(notification)

        logger.debug("Queue file %s: %s", result.filename, result.status)

    # =========================================================================
    # Housekeeping
    # =========================================================================

    def _run_housekeeping(self) -> None:
        """Run periodic housekeeping tasks.

        - Delete old processed/ files (> PROCESSED_RETENTION_DAYS)
        - Delete orphaned tmp/ files (> TMP_ORPHAN_MAX_AGE_SECONDS)
        - Warn on stale new/ files (> NEW_STALE_WARNING_SECONDS)
        """
        now = time.time()

        # Prune old processed files
        if self._processed_dir.exists():
            retention_cutoff = now - (PROCESSED_RETENTION_DAYS * 86400)
            for f in self._processed_dir.iterdir():
                if f.is_file():
                    try:
                        if f.stat().st_mtime < retention_cutoff:
                            f.unlink()
                            logger.debug("Pruned old processed file: %s", f.name)
                    except OSError:
                        pass

        # Clean orphaned tmp files
        if self._tmp_dir.exists():
            orphan_cutoff = now - TMP_ORPHAN_MAX_AGE_SECONDS
            for f in self._tmp_dir.iterdir():
                if f.is_file():
                    try:
                        if f.stat().st_mtime < orphan_cutoff:
                            f.unlink()
                            logger.debug("Deleted orphaned tmp file: %s", f.name)
                    except OSError:
                        pass

        # Warn on stale new files
        if self._new_dir.exists():
            stale_cutoff = now - NEW_STALE_WARNING_SECONDS
            for f in self._new_dir.iterdir():
                if f.is_file():
                    try:
                        if f.stat().st_mtime < stale_cutoff:
                            logger.warning("Stale queue file in new/: %s", f.name)
                    except OSError:
                        pass

    def _startup_recovery(self) -> None:
        """Recover orphaned tmp/ files on startup.

        Files in tmp/ older than STARTUP_RECOVERY_AGE_SECONDS are either:
        - Valid JSON: moved to new/ for reprocessing
        - Invalid/corrupt: deleted with a warning

        Recent tmp/ files (< STARTUP_RECOVERY_AGE_SECONDS) are left alone
        as they may be in-progress writes.
        """
        if not self._tmp_dir.exists():
            return

        now = time.time()
        recovery_cutoff = now - STARTUP_RECOVERY_AGE_SECONDS
        recovered = 0
        deleted = 0

        for f in self._tmp_dir.iterdir():
            if not f.is_file():
                continue

            try:
                file_stat = f.stat()
                if file_stat.st_mtime >= recovery_cutoff:
                    continue  # Too recent, might be in-progress write
            except OSError:
                continue

            # Check file size before reading
            try:
                if f.stat().st_size > MAX_QUEUE_FILE_SIZE:
                    logger.warning("Oversized tmp file during recovery: %s, deleting", f.name)
                    f.unlink()
                    deleted += 1
                    continue
            except OSError:
                continue

            # Old file - try to recover
            try:
                raw = f.read_text(encoding="utf-8")
                json.loads(raw)  # Validate JSON
                # Valid JSON - move to new/
                self._safe_move(f, self._new_dir)
                recovered += 1
            except (json.JSONDecodeError, OSError):
                # Corrupt - delete
                try:
                    f.unlink()
                    deleted += 1
                    logger.warning("Deleted corrupt tmp file during recovery: %s", f.name)
                except OSError:
                    logger.warning("Failed to delete corrupt tmp file during recovery: %s", f.name)

        if recovered or deleted:
            logger.info(
                "Startup recovery: %d files recovered, %d corrupt files deleted",
                recovered,
                deleted,
            )
