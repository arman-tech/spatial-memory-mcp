"""Tests for configuration system."""

from pathlib import Path

import pytest

from spatial_memory.config import (
    Settings,
    get_settings,
    override_settings,
    reset_settings,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = Settings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.embedding_dimensions == 384
        assert settings.default_namespace == "default"
        assert settings.default_importance == 0.5
        assert settings.log_level == "INFO"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        settings = Settings(
            memory_path="/custom/path",
            embedding_model="custom-model",
            default_importance=0.8,
        )
        # Use Path comparison to handle platform differences
        assert settings.memory_path == Path("/custom/path")
        assert settings.embedding_model == "custom-model"
        assert settings.default_importance == 0.8

    def test_importance_bounds(self) -> None:
        """Test importance value bounds."""
        # Valid bounds
        settings = Settings(default_importance=0.0)
        assert settings.default_importance == 0.0

        settings = Settings(default_importance=1.0)
        assert settings.default_importance == 1.0

        # Invalid bounds should raise
        with pytest.raises(ValueError):
            Settings(default_importance=-0.1)

        with pytest.raises(ValueError):
            Settings(default_importance=1.1)


class TestSettingsInjection:
    """Tests for settings dependency injection."""

    def test_get_settings_returns_singleton(self) -> None:
        """Test that get_settings returns same instance."""
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_override_settings(self) -> None:
        """Test settings override for testing."""
        reset_settings()
        original = get_settings()

        custom = Settings(memory_path="/override/path")
        override_settings(custom)

        current = get_settings()
        # Use Path comparison to handle platform differences
        assert current.memory_path == Path("/override/path")
        assert current is custom
        assert current is not original

        reset_settings()

    def test_reset_settings(self) -> None:
        """Test settings reset."""
        reset_settings()
        s1 = get_settings()

        reset_settings()
        s2 = get_settings()

        # Should be new instance after reset
        assert s1 is not s2
