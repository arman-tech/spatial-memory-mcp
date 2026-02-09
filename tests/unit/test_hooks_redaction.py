"""Unit tests for spatial_memory.hooks.redaction.

Tests cover:
1. Tier 1 secrets — each prefix-based pattern
2. Tier 2 secrets — structural patterns including SSH key should_skip
3. Tier 3 secrets — keyword patterns
4. False positive resistance — UUID, templates, placeholders, stopwords
5. should_skip — threshold behavior
6. Clean text — no secrets passes through unchanged
7. Multiple secrets — multiple types in one text
"""

from __future__ import annotations

import pytest

from spatial_memory.hooks.redaction import RedactionResult, redact_secrets

# =============================================================================
# Tier 1 Secrets (prefix-based)
# =============================================================================


@pytest.mark.unit
class TestTier1Secrets:
    """Each prefix-based pattern detects and redacts correctly."""

    def test_aws_key(self) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = redact_secrets(text)
        assert "[REDACTED_AWS_KEY]" in result.redacted_text
        assert "AKIAIOSFODNN7EXAMPLE" not in result.redacted_text
        assert result.redaction_count >= 1

    def test_github_token_ghp(self) -> None:
        token = "ghp_" + "A" * 40
        text = f"GITHUB_TOKEN={token}"
        result = redact_secrets(text)
        assert "[REDACTED_GITHUB_TOKEN]" in result.redacted_text
        assert token not in result.redacted_text

    def test_github_token_ghs(self) -> None:
        token = "ghs_" + "B" * 40
        result = redact_secrets(f"Token: {token}")
        assert "[REDACTED_GITHUB_TOKEN]" in result.redacted_text

    def test_github_pat(self) -> None:
        token = "github_pat_" + "C" * 30
        result = redact_secrets(f"PAT={token}")
        assert "[REDACTED_GITHUB_TOKEN]" in result.redacted_text

    def test_stripe_live_key(self) -> None:
        key = "sk_live_" + "D" * 24
        result = redact_secrets(f"STRIPE_KEY={key}")
        assert "[REDACTED_STRIPE_KEY]" in result.redacted_text
        assert key not in result.redacted_text

    def test_stripe_test_key(self) -> None:
        key = "pk_test_" + "E" * 24
        result = redact_secrets(f"Key: {key}")
        assert "[REDACTED_STRIPE_KEY]" in result.redacted_text

    def test_slack_token_xoxb(self) -> None:
        token = "xoxb-1234567890-abcdefghij"
        result = redact_secrets(f"SLACK_TOKEN={token}")
        assert "[REDACTED_SLACK_TOKEN]" in result.redacted_text

    def test_slack_webhook(self) -> None:
        url = "https://hooks.slack.com/services/T01234567/B01234567/abcdefghijklmn"
        result = redact_secrets(f"Webhook: {url}")
        assert "[REDACTED_SLACK_WEBHOOK]" in result.redacted_text

    def test_openai_key_proj(self) -> None:
        key = "sk-proj-" + "F" * 40
        result = redact_secrets(f"OPENAI_API_KEY={key}")
        assert "[REDACTED_OPENAI_KEY]" in result.redacted_text

    def test_google_api_key(self) -> None:
        key = "AIza" + "G" * 35
        result = redact_secrets(f"GOOGLE_KEY={key}")
        assert "[REDACTED_GOOGLE_KEY]" in result.redacted_text

    def test_sendgrid_key(self) -> None:
        key = "SG." + "H" * 22 + "." + "I" * 43
        result = redact_secrets(f"SENDGRID_API_KEY={key}")
        assert "[REDACTED_SENDGRID_KEY]" in result.redacted_text

    def test_pypi_token(self) -> None:
        token = "pypi-" + "J" * 20
        result = redact_secrets(f"PYPI_TOKEN={token}")
        assert "[REDACTED_PYPI_TOKEN]" in result.redacted_text

    def test_npm_token(self) -> None:
        token = "npm_" + "K" * 40
        result = redact_secrets(f"NPM_TOKEN={token}")
        assert "[REDACTED_NPM_TOKEN]" in result.redacted_text

    def test_anthropic_key(self) -> None:
        key = "sk-ant-" + "L" * 30
        result = redact_secrets(f"ANTHROPIC_API_KEY={key}")
        assert "[REDACTED_ANTHROPIC_KEY]" in result.redacted_text

    def test_huggingface_token(self) -> None:
        token = "hf_" + "M" * 36
        result = redact_secrets(f"HF_TOKEN={token}")
        assert "[REDACTED_HF_TOKEN]" in result.redacted_text


# =============================================================================
# Tier 2 Secrets (structural)
# =============================================================================


@pytest.mark.unit
class TestTier2Secrets:
    """Structural patterns detect and redact correctly."""

    def test_ssh_private_key(self) -> None:
        text = (
            "Here is my key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF...\n"
            "-----END RSA PRIVATE KEY-----\n"
            "Don't share this!"
        )
        result = redact_secrets(text)
        assert "[REDACTED_PRIVATE_KEY]" in result.redacted_text
        assert "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn" not in result.redacted_text
        assert result.should_skip is True  # private key triggers skip

    def test_ec_private_key(self) -> None:
        text = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHQCAQEEIBkg4LVWM9nuwNSK...\n"
            "-----END EC PRIVATE KEY-----"
        )
        result = redact_secrets(text)
        assert "[REDACTED_PRIVATE_KEY]" in result.redacted_text
        assert result.should_skip is True

    def test_openssh_private_key(self) -> None:
        text = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdjEAAAAA...\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        result = redact_secrets(text)
        assert "[REDACTED_PRIVATE_KEY]" in result.redacted_text
        assert result.should_skip is True

    def test_pgp_private_key(self) -> None:
        text = (
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF5...\n-----END PGP PRIVATE KEY BLOCK-----"
        )
        result = redact_secrets(text)
        assert "[REDACTED_PRIVATE_KEY]" in result.redacted_text
        assert result.should_skip is True

    def test_jwt_token(self) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = f"Authorization: Bearer {jwt}"
        result = redact_secrets(text)
        redacted = result.redacted_text
        assert "[REDACTED_JWT]" in redacted or "[REDACTED_AUTH_HEADER]" in redacted
        assert result.redaction_count >= 1

    def test_bearer_auth_header(self) -> None:
        text = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert result.redaction_count >= 1

    def test_password_in_url(self) -> None:
        text = "Connect to postgres://admin:s3cretP4ss@db.example.com:5432/mydb"
        result = redact_secrets(text)
        assert "[REDACTED_PASSWORD_URL]" in result.redacted_text
        assert result.redaction_count >= 1

    def test_connection_string_password(self) -> None:
        text = "Server=db;Password=MyS3cret!;Database=mydb"
        result = redact_secrets(text)
        assert "[REDACTED_PASSWORD]" in result.redacted_text
        assert "MyS3cret!" not in result.redacted_text


# =============================================================================
# Tier 3 Secrets (keyword-based)
# =============================================================================


@pytest.mark.unit
class TestTier3Secrets:
    """Keyword-based patterns detect real secrets."""

    def test_password_assignment(self) -> None:
        text = "password = SuperSecret123!"
        result = redact_secrets(text)
        assert "[REDACTED" in result.redacted_text
        assert "SuperSecret123!" not in result.redacted_text
        assert result.redaction_count >= 1

    def test_api_key_assignment(self) -> None:
        text = "api_key = abc123def456ghi789"
        result = redact_secrets(text)
        assert "[REDACTED" in result.redacted_text
        assert result.redaction_count >= 1

    def test_token_assignment(self) -> None:
        text = "token = mytokenvalue12345678"
        result = redact_secrets(text)
        assert "[REDACTED" in result.redacted_text
        assert result.redaction_count >= 1


# =============================================================================
# False Positive Resistance
# =============================================================================


@pytest.mark.unit
class TestFalsePositiveResistance:
    """Templates, placeholders, and stopwords should NOT be redacted."""

    def test_template_expression(self) -> None:
        text = "password = ${DB_PASSWORD}"
        result = redact_secrets(text)
        # The template should be recognized as false positive (Tier 3)
        assert "${DB_PASSWORD}" in result.redacted_text

    def test_jinja_template(self) -> None:
        text = "api_key = {{ secrets.API_KEY }}"
        result = redact_secrets(text)
        assert "{{ secrets.API_KEY }}" in result.redacted_text

    def test_placeholder_changeme(self) -> None:
        text = "password = changeme"
        result = redact_secrets(text)
        # "changeme" is a stopword — should not be redacted
        assert "changeme" in result.redacted_text

    def test_placeholder_example(self) -> None:
        text = "secret = example_value_here"
        result = redact_secrets(text)
        assert "example_value_here" in result.redacted_text

    def test_uuid_not_redacted(self) -> None:
        """UUIDs should not be matched as secrets."""
        text = "id = 550e8400-e29b-41d4-a716-446655440000"
        result = redact_secrets(text)
        # UUID should pass through (not a secret pattern)
        assert "550e8400" in result.redacted_text

    def test_asterisk_placeholder(self) -> None:
        text = "password = ****"
        result = redact_secrets(text)
        assert "****" in result.redacted_text


# =============================================================================
# should_skip Behavior
# =============================================================================


@pytest.mark.unit
class TestShouldSkip:
    """Test the should_skip flag behavior."""

    def test_private_key_triggers_skip(self) -> None:
        text = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg...\n-----END PRIVATE KEY-----"
        result = redact_secrets(text)
        assert result.should_skip is True

    def test_mostly_secrets_triggers_skip(self) -> None:
        """Text that is mostly secrets should trigger skip."""
        # Create text where secrets dominate
        key = "AKIA" + "A" * 16
        text = key  # entire text is a secret
        result = redact_secrets(text)
        assert result.should_skip is True

    def test_small_redaction_no_skip(self) -> None:
        """Text with a small secret in a large body should not skip."""
        text = (
            "This is a long paragraph about database configuration. "
            "We need to set up connection pooling properly. "
            "The max pool size should be 20 connections. "
            "AWS key for reference: AKIAIOSFODNN7EXAMPLE "
            "Make sure to configure the timeout as well."
        )
        result = redact_secrets(text)
        assert result.redaction_count >= 1
        assert result.should_skip is False


# =============================================================================
# Clean Text
# =============================================================================


@pytest.mark.unit
class TestCleanText:
    """Text without secrets passes through unchanged."""

    def test_no_secrets(self) -> None:
        text = "This is a normal paragraph about software architecture."
        result = redact_secrets(text)
        assert result.redacted_text == text
        assert result.redaction_count == 0
        assert result.should_skip is False

    def test_empty_string(self) -> None:
        result = redact_secrets("")
        assert result == RedactionResult(redacted_text="", redaction_count=0, should_skip=False)


# =============================================================================
# Multiple Secrets
# =============================================================================


@pytest.mark.unit
class TestMultipleSecrets:
    """Text with multiple secret types."""

    def test_multiple_types(self) -> None:
        aws_key = "AKIA" + "A" * 16
        gh_token = "ghp_" + "B" * 40
        text = f"Here are the credentials:\nAWS: {aws_key}\nGitHub: {gh_token}\nAll set!"
        result = redact_secrets(text)
        assert "[REDACTED_AWS_KEY]" in result.redacted_text
        assert "[REDACTED_GITHUB_TOKEN]" in result.redacted_text
        assert aws_key not in result.redacted_text
        assert gh_token not in result.redacted_text
        assert result.redaction_count >= 2

    def test_same_type_multiple(self) -> None:
        """Multiple AWS keys in same text."""
        key1 = "AKIA" + "X" * 16
        key2 = "AKIA" + "Y" * 16
        text = f"Key1: {key1}, Key2: {key2}"
        result = redact_secrets(text)
        assert key1 not in result.redacted_text
        assert key2 not in result.redacted_text
        assert result.redaction_count >= 2


# =============================================================================
# H-1: Tier 3 value-is-substring-of-key regression
# =============================================================================


@pytest.mark.unit
class TestTier3SubstringRegression:
    """H-1: str.index() found value at wrong offset when value is substring of key."""

    def test_tier3_password_value_substring_of_key(self) -> None:
        """'passwd = pass' should redact 'pass', not corrupt the key prefix."""
        result = redact_secrets("passwd = pass1234")
        assert "passwd" in result.redacted_text
        assert "[REDACTED_PASSWORD]" in result.redacted_text
        assert "pass1234" not in result.redacted_text

    def test_tier3_secret_value_substring_of_key(self) -> None:
        """'api_key = api' should preserve the key name 'api_key'."""
        result = redact_secrets("api_key = api_secret_value123")
        assert "api_key" in result.redacted_text
        assert "[REDACTED_SECRET]" in result.redacted_text
        assert "api_secret_value123" not in result.redacted_text
