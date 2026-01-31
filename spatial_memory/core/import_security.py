"""Import validation module for secure memory import operations.

Provides validation for import records to ensure data integrity and
security during bulk import operations.

Classes:
    ImportValidationConfig: Configuration for import validation.
    ImportValidator: Validates import records against schema and constraints.
    BatchValidationResult: Result of batch validation operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Validation Patterns
# =============================================================================

# Namespace must be alphanumeric with hyphens and underscores only
# No path traversal characters: /, \, .., ;
NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Invalid namespace patterns for security
INVALID_NAMESPACE_PATTERNS = [
    re.compile(r"\.\."),  # Path traversal
    re.compile(r"[/\\]"),  # Directory separators
    re.compile(r"[;]"),  # SQL injection risk
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ImportValidationConfig:
    """Configuration for import validation.

    Attributes:
        expected_vector_dim: Expected vector dimensions for embeddings.
        fail_fast: If True, stop validation on first error.
        max_errors: Maximum number of errors to collect per record.
    """

    expected_vector_dim: int
    fail_fast: bool = False
    max_errors: int = 100


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class BatchValidationResult:
    """Result of batch validation operation.

    Attributes:
        valid_count: Number of valid records.
        invalid_count: Number of invalid records.
        errors: List of (record_index, error_messages) tuples.
        is_valid: True if all records are valid.
    """

    valid_count: int = 0
    invalid_count: int = 0
    errors: list[tuple[int, list[str]]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if all records are valid."""
        return self.invalid_count == 0


# =============================================================================
# ImportValidator
# =============================================================================


class ImportValidator:
    """Validates import records for schema and constraint compliance.

    Validates individual records or batches of records against:
    - Required fields (content)
    - Optional field types (namespace, tags, importance, metadata, vector)
    - Vector dimension matching
    - Value constraints (importance range, etc.)

    Example:
        >>> config = ImportValidationConfig(expected_vector_dim=384)
        >>> validator = ImportValidator(config)
        >>> is_valid, errors = validator.validate_record(record, 384, 0)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """

    def __init__(self, config: ImportValidationConfig) -> None:
        """Initialize validator with configuration.

        Args:
            config: Validation configuration.
        """
        self.config = config

    def validate_record(
        self,
        record: dict[str, Any],
        expected_vector_dim: int,
        record_index: int,
    ) -> tuple[bool, list[str]]:
        """Validate a single import record.

        Args:
            record: The record dictionary to validate.
            expected_vector_dim: Expected vector dimensions.
            record_index: Index of record in batch (for error messages).

        Returns:
            Tuple of (is_valid, list_of_errors).
            is_valid is True if record passes all validation.
            list_of_errors contains error messages for failures.
        """
        errors: list[str] = []

        # Validate required fields first
        self._validate_content(record, record_index, errors)

        # Early exit if fail_fast and already have errors
        if self.config.fail_fast and errors:
            return False, errors

        # Validate optional fields
        if self._should_continue(errors):
            self._validate_namespace(record, record_index, errors)

        if self._should_continue(errors):
            self._validate_tags(record, record_index, errors)

        if self._should_continue(errors):
            self._validate_importance(record, record_index, errors)

        if self._should_continue(errors):
            self._validate_metadata(record, record_index, errors)

        if self._should_continue(errors):
            self._validate_vector(record, expected_vector_dim, record_index, errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_batch(
        self,
        records: list[dict[str, Any]],
        expected_vector_dim: int,
    ) -> BatchValidationResult:
        """Validate a batch of import records.

        Args:
            records: List of records to validate.
            expected_vector_dim: Expected vector dimensions.

        Returns:
            BatchValidationResult with counts and errors.
        """
        result = BatchValidationResult()

        for index, record in enumerate(records):
            is_valid, errors = self.validate_record(
                record, expected_vector_dim, index
            )

            if is_valid:
                result.valid_count += 1
            else:
                result.invalid_count += 1
                result.errors.append((index, errors))

                # Fail fast for batch
                if self.config.fail_fast:
                    break

        return result

    def _should_continue(self, errors: list[str]) -> bool:
        """Check if validation should continue.

        Args:
            errors: Current error list.

        Returns:
            True if validation should continue.
        """
        if self.config.fail_fast and errors:
            return False
        if len(errors) >= self.config.max_errors:
            return False
        return True

    def _validate_content(
        self,
        record: dict[str, Any],
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the content field (required).

        Args:
            record: Record to validate.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        content = record.get("content")

        # Check if content is missing
        if content is None or "content" not in record:
            errors.append(
                f"Record {record_index}: 'content' field is required"
            )
            return

        # Check if content is a string
        if not isinstance(content, str):
            errors.append(
                f"Record {record_index}: 'content' must be a string, "
                f"got {type(content).__name__}"
            )
            return

        # Check if content is non-empty (after stripping whitespace)
        if not content.strip():
            errors.append(
                f"Record {record_index}: 'content' must be non-empty"
            )

    def _validate_namespace(
        self,
        record: dict[str, Any],
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the namespace field (optional).

        Args:
            record: Record to validate.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        namespace = record.get("namespace")

        # Namespace is optional
        if namespace is None:
            return

        # Check type
        if not isinstance(namespace, str):
            errors.append(
                f"Record {record_index}: 'namespace' must be a string, "
                f"got {type(namespace).__name__}"
            )
            return

        # Check empty
        if not namespace:
            errors.append(
                f"Record {record_index}: 'namespace' cannot be empty string"
            )
            return

        # Check for invalid patterns (security)
        for pattern in INVALID_NAMESPACE_PATTERNS:
            if pattern.search(namespace):
                errors.append(
                    f"Record {record_index}: 'namespace' contains invalid characters: "
                    f"'{namespace}'"
                )
                return

        # Check valid format
        if not NAMESPACE_PATTERN.match(namespace):
            errors.append(
                f"Record {record_index}: 'namespace' has invalid format: "
                f"'{namespace}' (must be alphanumeric with hyphens/underscores)"
            )

    def _validate_tags(
        self,
        record: dict[str, Any],
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the tags field (optional).

        Args:
            record: Record to validate.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        tags = record.get("tags")

        # Tags are optional
        if tags is None:
            return

        # Check type is list
        if not isinstance(tags, list):
            errors.append(
                f"Record {record_index}: 'tags' must be a list, "
                f"got {type(tags).__name__}"
            )
            return

        # Check all items are strings
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                errors.append(
                    f"Record {record_index}: 'tags[{i}]' must be a string, "
                    f"got {type(tag).__name__}"
                )
                return  # One error for tags is enough

    def _validate_importance(
        self,
        record: dict[str, Any],
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the importance field (optional).

        Args:
            record: Record to validate.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        importance = record.get("importance")

        # Importance is optional
        if importance is None:
            return

        # Check type (allow int or float)
        if not isinstance(importance, (int, float)):
            errors.append(
                f"Record {record_index}: 'importance' must be a number, "
                f"got {type(importance).__name__}"
            )
            return

        # Check range [0, 1]
        if importance < 0 or importance > 1:
            errors.append(
                f"Record {record_index}: 'importance' must be between 0 and 1, "
                f"got {importance}"
            )

    def _validate_metadata(
        self,
        record: dict[str, Any],
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the metadata field (optional).

        Args:
            record: Record to validate.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        metadata = record.get("metadata")

        # Metadata is optional
        if metadata is None:
            return

        # Check type is dict
        if not isinstance(metadata, dict):
            errors.append(
                f"Record {record_index}: 'metadata' must be a dict, "
                f"got {type(metadata).__name__}"
            )

    def _validate_vector(
        self,
        record: dict[str, Any],
        expected_dim: int,
        record_index: int,
        errors: list[str],
    ) -> None:
        """Validate the vector field (optional).

        Args:
            record: Record to validate.
            expected_dim: Expected vector dimensions.
            record_index: Index for error messages.
            errors: List to append errors to.
        """
        vector = record.get("vector")

        # Vector is optional (will be generated during import)
        if vector is None:
            return

        # Check type is list or tuple
        if not isinstance(vector, (list, tuple)):
            errors.append(
                f"Record {record_index}: 'vector' must be a list, "
                f"got {type(vector).__name__}"
            )
            return

        # Check dimensions
        actual_dim = len(vector)
        if actual_dim != expected_dim:
            errors.append(
                f"Record {record_index}: vector dimension mismatch - "
                f"expected {expected_dim}, got {actual_dim}"
            )
            return

        # Check all values are numeric
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                errors.append(
                    f"Record {record_index}: 'vector[{i}]' must be numeric, "
                    f"got {type(val).__name__}"
                )
                return  # One error for vector values is enough
