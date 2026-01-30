"""Secure structured logging for Spatial Memory MCP Server."""

import json
import logging
import re
from datetime import datetime, timezone

# Patterns to mask in logs
SENSITIVE_PATTERNS = [
    (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', re.I), 'api_key=***MASKED***'),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), '***OPENAI_KEY***'),
    (re.compile(r'password["\']?\s*[:=]\s*["\']?[^\s"\']+', re.I), 'password=***MASKED***'),
]


class SecureFormatter(logging.Formatter):
    """Formatter that masks sensitive data."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record and mask sensitive data.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message with sensitive data masked.
        """
        message = super().format(record)
        for pattern, replacement in SENSITIVE_PATTERNS:
            message = pattern.sub(replacement, message)
        return message


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with sensitive data masked.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log message with sensitive data masked.
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Mask sensitive data
        json_str = json.dumps(log_data)
        for pattern, replacement in SENSITIVE_PATTERNS:
            json_str = pattern.sub(replacement, json_str)

        return json_str


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    mask_sensitive: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: Use JSON format for structured logging.
        mask_sensitive: Mask sensitive data in logs.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Choose formatter
    if json_format:
        formatter: logging.Formatter = JSONFormatter()
    elif mask_sensitive:
        formatter = SecureFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
