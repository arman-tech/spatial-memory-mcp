"""Tests for configuration system."""

from pathlib import Path

import pytest

from spatial_memory.config import (
    ConfigurationError,
    Settings,
    get_settings,
    override_settings,
    reset_settings,
    validate_startup,
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


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_startup_openai_requires_key(self, tmp_path: Path) -> None:
        """Test that OpenAI embeddings require API key."""
        settings = Settings(
            memory_path=tmp_path,
            embedding_model="openai:text-embedding-3-small",
            openai_api_key=None,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup(settings)

        assert "OpenAI API key required" in str(exc_info.value)
        assert "SPATIAL_MEMORY_OPENAI_API_KEY" in str(exc_info.value)

    def test_validate_startup_openai_with_key(self, tmp_path: Path) -> None:
        """Test that OpenAI embeddings work with API key."""
        settings = Settings(
            memory_path=tmp_path,
            embedding_model="openai:text-embedding-3-small",
            openai_api_key="sk-test123456789012345678901234567890",
        )

        warnings = validate_startup(settings)
        # Should not raise, warnings are fine
        assert isinstance(warnings, list)

    def test_validate_startup_storage_path_created(self, tmp_path: Path) -> None:
        """Test that storage path is created if it doesn't exist."""
        storage_path = tmp_path / "new" / "nested" / "path"
        assert not storage_path.exists()

        settings = Settings(memory_path=storage_path)
        warnings = validate_startup(settings)

        assert storage_path.exists()
        assert isinstance(warnings, list)

    def test_validate_startup_storage_path_writable(self, tmp_path: Path) -> None:
        """Test that storage path writability is checked."""
        settings = Settings(memory_path=tmp_path)
        warnings = validate_startup(settings)

        # Should not raise
        assert isinstance(warnings, list)

    def test_validate_startup_warns_low_nprobes(self, tmp_path: Path) -> None:
        """Test warning for low index_nprobes."""
        settings = Settings(
            memory_path=tmp_path,
            index_nprobes=5,
        )

        warnings = validate_startup(settings)

        assert len(warnings) > 0
        assert any("index_nprobes" in w and "low" in w for w in warnings)

    def test_validate_startup_warns_low_retry_attempts(self, tmp_path: Path) -> None:
        """Test warning for low max_retry_attempts."""
        settings = Settings(
            memory_path=tmp_path,
            max_retry_attempts=1,
        )

        warnings = validate_startup(settings)

        assert len(warnings) > 0
        assert any("max_retry_attempts" in w for w in warnings)

    def test_validate_startup_no_warnings_for_good_config(self, tmp_path: Path) -> None:
        """Test that good configuration produces no warnings."""
        settings = Settings(
            memory_path=tmp_path,
            index_nprobes=20,
            max_retry_attempts=3,
        )

        warnings = validate_startup(settings)

        assert len(warnings) == 0
