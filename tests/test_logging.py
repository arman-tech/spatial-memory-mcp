"""Tests for secure structured logging."""

import json
import logging

from spatial_memory.core.logging import (
    JSONFormatter,
    SecureFormatter,
    configure_logging,
)


class TestSecureFormatter:
    """Tests for SecureFormatter."""

    def test_masks_api_key(self) -> None:
        """Test that API keys are masked."""
        formatter = SecureFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API_KEY=sk-abc123xyz456",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "sk-abc123xyz456" not in formatted
        assert "***MASKED***" in formatted

    def test_masks_openai_key_pattern(self) -> None:
        """Test that OpenAI key pattern is masked."""
        formatter = SecureFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Using key: sk-abcdefghij1234567890",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "sk-abcdefghij1234567890" not in formatted
        assert "***OPENAI_KEY***" in formatted

    def test_masks_api_key_variations(self) -> None:
        """Test masking of various API key formats."""
        formatter = SecureFormatter()

        test_cases = [
            'api_key="secret123"',
            "apikey=secret456",
            "api-key: secret789",
        ]

        for msg in test_cases:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )

            formatted = formatter.format(record)
            assert "secret" not in formatted.lower() or "***MASKED***" in formatted

    def test_masks_password(self) -> None:
        """Test that passwords are masked."""
        formatter = SecureFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='password="mySecretPass123"',
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "mySecretPass123" not in formatted
        assert "***MASKED***" in formatted

    def test_does_not_mask_normal_content(self) -> None:
        """Test that normal content is not masked."""
        formatter = SecureFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing request for user data",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "Processing request for user data" in formatted
        assert "***MASKED***" not in formatted


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_formats_as_json(self) -> None:
        """Test that output is valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_masks_sensitive_data_in_json(self) -> None:
        """Test that sensitive data is masked in JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API_KEY=sk-abc123xyz456",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "sk-abc123xyz456" not in formatted
        assert "***MASKED***" in formatted

        # Verify it's still valid JSON
        parsed = json.loads(formatted)
        assert "***MASKED***" in parsed["message"]

    def test_includes_exception_info(self) -> None:
        """Test that exception info is included."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "exception" in parsed
        assert "ValueError: Test error" in parsed["exception"]


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_configures_text_logging(self) -> None:
        """Test text logging configuration."""
        configure_logging(level="INFO", json_format=False)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_configures_json_logging(self) -> None:
        """Test JSON logging configuration."""
        configure_logging(level="DEBUG", json_format=True)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Check that JSONFormatter is used
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_configures_secure_logging(self) -> None:
        """Test secure logging with masking."""
        configure_logging(level="INFO", json_format=False, mask_sensitive=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        # Check that SecureFormatter is used
        assert isinstance(handler.formatter, SecureFormatter)

    def test_removes_existing_handlers(self) -> None:
        """Test that existing handlers are removed."""
        # Add a dummy handler
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)

        # Configure logging
        configure_logging(level="INFO")

        # Should only have the new handler
        assert len(root_logger.handlers) == 1
        assert dummy_handler not in root_logger.handlers
