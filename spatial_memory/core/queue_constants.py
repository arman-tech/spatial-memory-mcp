"""Queue processor housekeeping constants (hardcoded, not configurable)."""

from __future__ import annotations

PROCESSED_RETENTION_DAYS = 7
TMP_ORPHAN_MAX_AGE_SECONDS = 3600  # 1 hour
NEW_STALE_WARNING_SECONDS = 86400  # 24 hours
STARTUP_RECOVERY_AGE_SECONDS = 300  # 5 minutes

QUEUE_DIR_NAME = "pending-saves"
QUEUE_FILE_VERSION = 1
