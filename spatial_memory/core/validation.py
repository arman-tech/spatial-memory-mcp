"""Centralized input validation for Spatial Memory MCP.

This module consolidates all validation logic from database.py and memory.py
to provide a single source of truth for input validation.

Security features:
- SQL injection prevention through pattern matching and escaping
- UUID format validation
- Content length validation
- Tag format and count validation
- Metadata size and serializability validation
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from spatial_memory.core.errors import ValidationError

# Content validation constants
MAX_CONTENT_LENGTH = 100_000  # 100KB of text

# Tag validation constants
MAX_TAGS = 100  # Maximum number of tags per memory
MAX_TAG_LENGTH = 50  # Maximum length of a single tag

# Metadata validation constants
MAX_METADATA_SIZE = 65536  # 64KB serialized JSON

# Namespace validation pattern
# Must start with letter, followed by letters/numbers/dash/underscore, max 63 chars
NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,62}$")

# Tag validation pattern
# Must start with letter or number, followed by letters/numbers/dash/underscore, max 50 chars
TAG_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$")

# Dangerous SQL patterns for injection prevention
DANGEROUS_PATTERNS = [
    r";\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
    r"--\s*$",
    r"/\*.*\*/",
    r"'\s*OR\s*'",
    r"'\s*AND\s*'",
    r"'\s*UNION\s+(?:ALL\s+)?SELECT",
    # Additional patterns for stored procedures and timing attacks
    r";\s*EXEC(?:UTE)?\s",                # EXEC/EXECUTE stored procedures
    r"WAITFOR\s+DELAY",                   # Time-based SQL injection
    r"(?:xp_|sp_)\w+",                    # SQL Server stored procedures
    r"0x[0-9a-fA-F]+",                    # Hex-encoded strings
    r"BENCHMARK\s*\(",                    # MySQL timing attack
    r"SLEEP\s*\(",                        # MySQL/PostgreSQL sleep
    r"PG_SLEEP\s*\(",                     # PostgreSQL specific
]


def validate_uuid(value: str) -> str:
    """Validate and return a UUID string.

    Args:
        value: The value to validate as a UUID.

    Returns:
        The validated UUID string.

    Raises:
        ValidationError: If the value is not a valid UUID format.

    Examples:
        >>> validate_uuid("550e8400-e29b-41d4-a716-446655440000")
        '550e8400-e29b-41d4-a716-446655440000'
        >>> validate_uuid("not-a-uuid")
        Traceback (most recent call last):
            ...
        ValidationError: Invalid UUID format: not-a-uuid
    """
    try:
        # Attempt to parse as UUID to validate format
        uuid.UUID(value)
        return value
    except (ValueError, AttributeError) as e:
        raise ValidationError(f"Invalid UUID format: {value}") from e


def validate_namespace(namespace: str) -> str:
    """Validate namespace format.

    Namespaces must:
    - Start with a letter
    - Contain only letters, numbers, dash, underscore, or dot
    - Be between 1-256 characters
    - Not be empty

    Args:
        namespace: The namespace to validate.

    Returns:
        The validated namespace string.

    Raises:
        ValidationError: If the namespace is invalid.

    Examples:
        >>> validate_namespace("default")
        'default'
        >>> validate_namespace("my-namespace_v1.0")
        'my-namespace_v1.0'
        >>> validate_namespace("")
        Traceback (most recent call last):
            ...
        ValidationError: Namespace cannot be empty
    """
    if not namespace:
        raise ValidationError("Namespace cannot be empty")

    if len(namespace) > 256:
        raise ValidationError("Namespace too long (max 256 characters)")

    # Allow alphanumeric, dash, underscore, dot
    if not re.match(r"^[\w\-\.]+$", namespace):
        raise ValidationError(f"Invalid namespace format: {namespace}")

    return namespace


def validate_content(content: str) -> None:
    """Validate memory content.

    Content must:
    - Not be empty or whitespace-only
    - Not exceed MAX_CONTENT_LENGTH characters

    Args:
        content: Content to validate.

    Raises:
        ValidationError: If content is empty, whitespace-only, or too long.

    Examples:
        >>> validate_content("This is valid content")
        >>> validate_content("")
        Traceback (most recent call last):
            ...
        ValidationError: Content cannot be empty
        >>> validate_content("x" * 100001)
        Traceback (most recent call last):
            ...
        ValidationError: Content exceeds maximum length...
    """
    if not content or not content.strip():
        raise ValidationError("Content cannot be empty")

    if len(content) > MAX_CONTENT_LENGTH:
        raise ValidationError(
            f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters "
            f"(got {len(content)} characters)"
        )


def validate_importance(importance: float) -> None:
    """Validate importance value (0.0-1.0).

    Args:
        importance: Importance to validate.

    Raises:
        ValidationError: If importance is out of range.

    Examples:
        >>> validate_importance(0.5)
        >>> validate_importance(1.5)
        Traceback (most recent call last):
            ...
        ValidationError: Importance must be between 0.0 and 1.0
    """
    if not 0.0 <= importance <= 1.0:
        raise ValidationError("Importance must be between 0.0 and 1.0")


def validate_tags(tags: list[str] | None) -> list[str]:
    """Validate and return tags list.

    Tags must:
    - Start with a letter or number
    - Contain only letters, numbers, dash, or underscore
    - Be between 1-50 characters each
    - Have at most MAX_TAGS total tags

    Args:
        tags: List of tags to validate (None is treated as empty list).

    Returns:
        Validated tags list (empty list if None was provided).

    Raises:
        ValidationError: If tags are invalid.

    Examples:
        >>> validate_tags(["tag1", "tag2"])
        ['tag1', 'tag2']
        >>> validate_tags(None)
        []
        >>> validate_tags(["invalid tag"])
        Traceback (most recent call last):
            ...
        ValidationError: Invalid tag format...
    """
    if tags is None:
        return []

    if len(tags) > MAX_TAGS:
        raise ValidationError(f"Maximum {MAX_TAGS} tags allowed, got {len(tags)}")

    validated = []
    for tag in tags:
        # Must be a string
        if not isinstance(tag, str):
            raise ValidationError(f"Tag must be a string, got {type(tag).__name__}")

        # Must match pattern: start with letter/number, alphanumeric with dash/underscore
        if not TAG_PATTERN.match(tag):
            raise ValidationError(
                f"Invalid tag format: '{tag}'. Tags must be 1-{MAX_TAG_LENGTH} characters, "
                "start with letter or number, and contain only letters, numbers, dash, "
                "or underscore."
            )

        validated.append(tag)

    return validated


def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and return metadata dict.

    Metadata must:
    - Be a dictionary
    - Be JSON-serializable
    - Not exceed MAX_METADATA_SIZE bytes when serialized

    Args:
        metadata: Metadata dictionary to validate (None is treated as empty dict).

    Returns:
        Validated metadata dictionary (empty dict if None was provided).

    Raises:
        ValidationError: If metadata is invalid.

    Examples:
        >>> validate_metadata({"key": "value"})
        {'key': 'value'}
        >>> validate_metadata(None)
        {}
        >>> validate_metadata("not a dict")
        Traceback (most recent call last):
            ...
        ValidationError: Metadata must be a dictionary...
    """
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise ValidationError(f"Metadata must be a dictionary, got {type(metadata).__name__}")

    # Check serialized size (max 64KB)
    try:
        serialized = json.dumps(metadata)
        if len(serialized) > MAX_METADATA_SIZE:
            raise ValidationError(
                f"Metadata exceeds 64KB limit ({len(serialized)} bytes)"
            )
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Metadata must be JSON-serializable: {e}") from e

    return metadata


def sanitize_string(value: str) -> str:
    """Sanitize string for safe SQL usage.

    Prevents SQL injection by:
    1. Validating input type
    2. Detecting dangerous SQL patterns
    3. Escaping single quotes

    Args:
        value: The string value to sanitize.

    Returns:
        Sanitized string safe for use in filter expressions.

    Raises:
        ValidationError: If the value contains invalid characters or SQL injection patterns.

    Examples:
        >>> sanitize_string("hello")
        'hello'
        >>> sanitize_string("it's")
        "it''s"
        >>> sanitize_string("'; DROP TABLE users--")
        Traceback (most recent call last):
            ...
        ValidationError: Invalid characters in value...
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value).__name__}")

    # Check for dangerous SQL injection patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            # Only show first 50 chars in error to prevent log flooding
            raise ValidationError(f"Invalid characters in value: {value[:50]}")

    # Escape single quotes by doubling them (standard SQL escaping)
    return value.replace("'", "''")
