"""Secure structured logging for Spatial Memory MCP Server.

This module provides secure logging with sensitive data masking and
optional request context tracking for observability.

Features:
    - Sensitive data masking (API keys, passwords)
    - JSON structured logging format
    - Request context integration ([req=xxx][agent=yyy] prefixes)
    - Configurable log levels and formats
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Patterns to mask in logs
SENSITIVE_PATTERNS = [
    (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', re.I), "api_key=***MASKED***"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "***OPENAI_KEY***"),
    (re.compile(r'password["\']?\s*[:=]\s*["\']?[^\s"\']+', re.I), "password=***MASKED***"),
]


def _get_trace_context() -> tuple[str | None, str | None]:
    """Get request context without importing at module level.

    Returns:
        Tuple of (request_id, agent_id) or (None, None) if no context.
    """
    try:
        # Import here to avoid circular imports
        from spatial_memory.core.tracing import get_current_context

        ctx = get_current_context()
        if ctx:
            return ctx.request_id, ctx.agent_id
    except ImportError:
        pass
    return None, None


class SecureFormatter(logging.Formatter):
    """Formatter that masks sensitive data and includes trace context."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        include_trace_context: bool = True,
    ) -> None:
        """Initialize the secure formatter.

        Args:
            fmt: Format string for log messages.
            datefmt: Date format string.
            include_trace_context: Whether to include [req=xxx][agent=yyy] prefix.
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.include_trace_context = include_trace_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record and mask sensitive data.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message with sensitive data masked and trace context.
        """
        message = super().format(record)

        # Add trace context prefix if available
        if self.include_trace_context:
            request_id, agent_id = _get_trace_context()
            if request_id:
                prefix_parts = [f"[req={request_id}]"]
                if agent_id:
                    prefix_parts.append(f"[agent={agent_id}]")
                prefix = "".join(prefix_parts) + " "
                # Insert after timestamp and logger name
                # Format: "2024-01-15 10:30:00 - logger - LEVEL - message"
                # We want: "2024-01-15 10:30:00 - logger - LEVEL - [req=xxx] message"
                parts = message.split(" - ", 3)
                if len(parts) == 4:
                    message = f"{parts[0]} - {parts[1]} - {parts[2]} - {prefix}{parts[3]}"
                else:
                    # Fallback: just prepend
                    message = prefix + message

        for pattern, replacement in SENSITIVE_PATTERNS:
            message = pattern.sub(replacement, message)
        return message


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with trace context."""

    def __init__(self, include_trace_context: bool = True) -> None:
        """Initialize the JSON formatter.

        Args:
            include_trace_context: Whether to include request_id and agent_id fields.
        """
        super().__init__()
        self.include_trace_context = include_trace_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with sensitive data masked.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log message with sensitive data masked.
        """
        log_data: dict[str, str | None] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace context if available
        if self.include_trace_context:
            request_id, agent_id = _get_trace_context()
            if request_id:
                log_data["request_id"] = request_id
            if agent_id:
                log_data["agent_id"] = agent_id

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
    include_trace_context: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: Use JSON format for structured logging.
        mask_sensitive: Mask sensitive data in logs.
        include_trace_context: Include [req=xxx][agent=yyy] in log messages.
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
        formatter: logging.Formatter = JSONFormatter(include_trace_context=include_trace_context)
    elif mask_sensitive:
        formatter = SecureFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            include_trace_context=include_trace_context,
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
