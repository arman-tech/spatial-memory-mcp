"""Stdlib-only secret/PII redaction pipeline for hook scripts.

Detects and replaces secrets (API keys, tokens, private keys, passwords)
with typed placeholders before content is queued for memory storage.

Pattern sources: Gitleaks/detect-secrets (MIT licensed), adapted for
pre-compiled stdlib ``re`` usage.

**STDLIB-ONLY**: Only ``re`` imports allowed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RedactionResult(NamedTuple):
    """Result of secret redaction."""

    redacted_text: str
    redaction_count: int
    should_skip: bool  # True if >50% redacted or private key found


# ---------------------------------------------------------------------------
# Internal: pattern to strip placeholders for length calculation
# ---------------------------------------------------------------------------

_PLACEHOLDER_STRIP_RE = re.compile(r"\[REDACTED_[A-Z_]+\]")

# ---------------------------------------------------------------------------
# Tier 1: Prefix-based patterns (high confidence, specific prefixes)
# Each entry: (compiled_regex, placeholder_string, is_private_key)
# ---------------------------------------------------------------------------

_TIER1_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    # AWS Access Key ID (starts with AKIA, 20 chars)
    (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "[REDACTED_AWS_KEY]",
        False,
    ),
    # GitHub tokens (ghp_, gho_, ghs_, ghr_, github_pat_)
    (
        re.compile(r"(?:ghp|gho|ghs|ghr)_[A-Za-z0-9_]{36,255}"),
        "[REDACTED_GITHUB_TOKEN]",
        False,
    ),
    (
        re.compile(r"github_pat_[A-Za-z0-9_]{22,255}"),
        "[REDACTED_GITHUB_TOKEN]",
        False,
    ),
    # Stripe keys (sk_live_, pk_live_, sk_test_, pk_test_)
    (
        re.compile(r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{20,99}"),
        "[REDACTED_STRIPE_KEY]",
        False,
    ),
    # Slack tokens (xoxb-, xoxp-) and webhook URLs
    (
        re.compile(r"xox[bp]-[0-9]{10,13}-[0-9A-Za-z-]+"),
        "[REDACTED_SLACK_TOKEN]",
        False,
    ),
    (
        re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"),
        "[REDACTED_SLACK_WEBHOOK]",
        False,
    ),
    # OpenAI keys (sk-proj-...  or  sk-<org>-...T3BlbkFJ)
    (
        re.compile(r"sk-proj-[A-Za-z0-9_-]{20,255}"),
        "[REDACTED_OPENAI_KEY]",
        False,
    ),
    (
        re.compile(r"sk-[A-Za-z0-9]{20,48}T3BlbkFJ[A-Za-z0-9]*"),
        "[REDACTED_OPENAI_KEY]",
        False,
    ),
    # Google API key (AIza...)
    (
        re.compile(r"AIza[0-9A-Za-z_-]{35}"),
        "[REDACTED_GOOGLE_KEY]",
        False,
    ),
    # SendGrid API key
    (
        re.compile(r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
        "[REDACTED_SENDGRID_KEY]",
        False,
    ),
    # PyPI token
    (
        re.compile(r"pypi-[A-Za-z0-9_-]{16,255}"),
        "[REDACTED_PYPI_TOKEN]",
        False,
    ),
    # npm token
    (
        re.compile(r"npm_[A-Za-z0-9]{36,64}"),
        "[REDACTED_NPM_TOKEN]",
        False,
    ),
    # Anthropic API key
    (
        re.compile(r"sk-ant-[A-Za-z0-9_-]{20,255}"),
        "[REDACTED_ANTHROPIC_KEY]",
        False,
    ),
    # HuggingFace token
    (
        re.compile(r"hf_[A-Za-z0-9]{34,}"),
        "[REDACTED_HF_TOKEN]",
        False,
    ),
]

# ---------------------------------------------------------------------------
# Tier 2: Structural patterns (multi-line, JWT, auth headers, URLs)
# ---------------------------------------------------------------------------

_TIER2_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    # SSH / PGP private keys (multi-line block)
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
        True,  # is_private_key -> should_skip
    ),
    # PGP private key block
    (
        re.compile(
            r"-----BEGIN PGP PRIVATE KEY BLOCK-----"
            r"[\s\S]*?"
            r"-----END PGP PRIVATE KEY BLOCK-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
        True,
    ),
    # JWT (header.payload.signature, each base64url)
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "[REDACTED_JWT]",
        False,
    ),
    # Basic/Bearer auth headers
    (
        re.compile(
            r"(?:Authorization|authorization)\s*[:=]\s*"
            r"(?:Basic|Bearer)\s+[A-Za-z0-9+/=_-]{20,}",
        ),
        "[REDACTED_AUTH_HEADER]",
        False,
    ),
    # Password in URL  (scheme://user:pass@host)
    (
        re.compile(r"://[^:\s]+:([^@\s]{3,})@"),
        "[REDACTED_PASSWORD_URL]",
        False,
    ),
]

# ---------------------------------------------------------------------------
# Tier 3: Keyword-based patterns (generic, with false-positive filters)
# ---------------------------------------------------------------------------

# Stopwords / template patterns that indicate non-real secrets
_FALSE_POSITIVE_RE = re.compile(
    r"^\$\{|^\{\{|^\$\(|"  # template expressions: ${...}, {{...}}, $(...)
    r"^<[^>]+>$|"  # XML/angle-bracket placeholders: <SECRET>
    r"^\*{2,}$|"  # asterisk placeholders: ***, ****
    r"^\[REDACTED_|"  # already-redacted values from earlier tiers
    r"^(changeme|example|replace|your[_-]|my[_-]|test|dummy|fake|xxx|placeholder)",
    re.IGNORECASE,
)


def _is_false_positive(value: str) -> bool:
    """Check if a matched 'secret' value is actually a placeholder/template."""
    stripped = value.strip().strip("'\"")
    if not stripped or len(stripped) < 3:
        return True
    return bool(_FALSE_POSITIVE_RE.search(stripped))


_TIER3_PASSWORD_RE = re.compile(
    r"""(?:password|passwd|pwd)\s*[=:]\s*['"]?([^\s'";\n]{3,})['"]?""",
    re.IGNORECASE,
)

_TIER3_SECRET_RE = re.compile(
    r"""(?:secret|api_key|apikey|api_secret|access_token)\s*[=:]\s*['"]?([^\s'";\n]{3,})['"]?""",
    re.IGNORECASE,
)

_TIER3_TOKEN_RE = re.compile(
    r"""(?:token|auth_token)\s*[=:]\s*['"]?([^\s'";\n]{8,})['"]?""",
    re.IGNORECASE,
)


def _apply_tier3(text: str) -> tuple[str, int]:
    """Apply Tier 3 keyword patterns with false-positive filtering.

    Returns:
        (redacted_text, match_count)
    """
    count = 0

    for pattern, placeholder in (
        (_TIER3_PASSWORD_RE, "[REDACTED_PASSWORD]"),
        (_TIER3_SECRET_RE, "[REDACTED_SECRET]"),
        (_TIER3_TOKEN_RE, "[REDACTED_TOKEN]"),
    ):

        def _make_replacer(ph: str) -> Callable[[re.Match[str]], str]:
            def _replacer(m: re.Match[str]) -> str:
                nonlocal count
                value = m.group(1)
                if _is_false_positive(value):
                    return m.group(0)  # leave as-is
                count += 1
                # Replace just the value portion, keeping the key prefix
                full = m.group(0)
                return full[: full.index(value)] + ph

            return _replacer

        text = pattern.sub(_make_replacer(placeholder), text)

    return text, count


# ---------------------------------------------------------------------------
# Combined pattern list (Tier 1 + Tier 2; Tier 3 handled separately)
# ---------------------------------------------------------------------------

_ALL_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = _TIER1_PATTERNS + _TIER2_PATTERNS


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def redact_secrets(text: str) -> RedactionResult:
    """Detect and redact secrets/PII from text.

    Applies patterns in three tiers:
    - Tier 1: Prefix-based (AWS, GitHub, Stripe, etc.)
    - Tier 2: Structural (private keys, JWT, auth headers, URLs)
    - Tier 3: Keyword-based (password=, secret=, etc.) with false-positive filter

    Args:
        text: The text to scan for secrets.

    Returns:
        RedactionResult with redacted text, count, and skip recommendation.
    """
    if not text:
        return RedactionResult(redacted_text="", redaction_count=0, should_skip=False)

    redacted = text
    count = 0
    has_private_key = False

    # Tier 1 + Tier 2 patterns
    for pattern, placeholder, is_private_key in _ALL_PATTERNS:
        matches = pattern.findall(redacted)
        if matches:
            count += len(matches)
            redacted = pattern.sub(placeholder, redacted)
            if is_private_key:
                has_private_key = True

    # Tier 3 patterns (with false-positive filtering)
    redacted, tier3_count = _apply_tier3(redacted)
    count += tier3_count

    # should_skip: private key found OR >50% of original chars redacted
    if has_private_key:
        should_skip = True
    elif count > 0:
        orig_len = len(text)
        if orig_len > 0:
            non_placeholder_len = len(_PLACEHOLDER_STRIP_RE.sub("", redacted))
            redacted_fraction = 1.0 - (non_placeholder_len / orig_len)
            should_skip = redacted_fraction > 0.5
        else:
            should_skip = False
    else:
        should_skip = False

    return RedactionResult(
        redacted_text=redacted,
        redaction_count=count,
        should_skip=should_skip,
    )
