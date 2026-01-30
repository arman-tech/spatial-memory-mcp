"""Demonstration of configuration validation and secure logging."""

import logging
import sys
from pathlib import Path

from spatial_memory.config import (
    ConfigurationError,
    Settings,
    validate_startup,
)
from spatial_memory.core.logging import configure_logging

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def demo_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Demo ===\n")

    # Valid configuration
    print("1. Valid configuration:")
    valid_settings = Settings(
        memory_path=Path("./.demo-memory"),
        index_nprobes=20,
        max_retry_attempts=3,
    )
    warnings = validate_startup(valid_settings)
    print(f"   ✓ Validation passed with {len(warnings)} warnings")

    # Configuration with warnings
    print("\n2. Configuration with warnings:")
    warning_settings = Settings(
        memory_path=Path("./.demo-memory"),
        index_nprobes=5,  # Low value
        max_retry_attempts=1,  # Low value
    )
    warnings = validate_startup(warning_settings)
    print(f"   ⚠ Validation passed with {len(warnings)} warnings:")
    for warning in warnings:
        print(f"     - {warning}")

    # Invalid configuration (OpenAI without key)
    print("\n3. Invalid configuration (OpenAI without API key):")
    try:
        invalid_settings = Settings(
            memory_path=Path("./.demo-memory"),
            embedding_model="openai:text-embedding-3-small",
            openai_api_key=None,
        )
        validate_startup(invalid_settings)
        print("   ✗ Should have failed!")
    except ConfigurationError as e:
        print(f"   ✓ Caught ConfigurationError: {e}")


def demo_secure_logging():
    """Demonstrate secure logging with sensitive data masking."""
    print("\n\n=== Secure Logging Demo ===\n")

    # Text format with masking
    print("1. Text format with sensitive data masking:")
    configure_logging(level="INFO", json_format=False, mask_sensitive=True)
    logger = logging.getLogger("demo")

    logger.info("Normal log message")
    logger.info("API key detected: api_key=sk-abc123def456ghi789")
    logger.warning("Password found: password=secretPass123")

    # JSON format with masking
    print("\n2. JSON format with sensitive data masking:")
    configure_logging(level="INFO", json_format=True, mask_sensitive=True)

    logger.info("Normal log message in JSON")
    logger.info("API key detected: api_key=sk-abc123def456ghi789")


def demo_log_format_setting():
    """Demonstrate log_format configuration setting."""
    print("\n\n=== Log Format Configuration ===\n")

    print("1. Settings with text format:")
    text_settings = Settings(log_format="text")
    print(f"   log_format = {text_settings.log_format}")

    print("\n2. Settings with JSON format:")
    json_settings = Settings(log_format="json")
    print(f"   log_format = {json_settings.log_format}")


if __name__ == "__main__":
    demo_validation()
    demo_secure_logging()
    demo_log_format_setting()

    print("\n✓ Demo complete!\n")
