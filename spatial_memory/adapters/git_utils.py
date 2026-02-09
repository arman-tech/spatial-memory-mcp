"""Git utilities for project detection.

Provides git repository discovery, remote URL parsing, and project ID
normalization for the project detection cascade.
"""

from __future__ import annotations

import configparser
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from spatial_memory.adapters.path_utils import is_blocklisted, normalize_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedRemoteURL:
    """Parsed components of a git remote URL."""

    host: str
    path: str  # org/repo (without .git suffix)
    scheme: str  # https, ssh, git, etc.
    user: str  # git, user, etc.


def find_git_root(start_path: str | Path, blocklist_check: bool = True) -> Path | None:
    """Walk up from start_path to find the nearest .git directory.

    Args:
        start_path: Directory to start searching from.
        blocklist_check: If True, stop at blocklisted directories.

    Returns:
        Path to the git repository root, or None if not found.
    """
    current = normalize_path(start_path)

    # If start_path is a file, start from its parent directory
    if current.is_file():
        current = current.parent

    while True:
        git_path = current / ".git"
        if git_path.exists():
            return current

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            return None

        if blocklist_check and is_blocklisted(parent):
            return None

        current = parent


def parse_git_entry(git_path: Path) -> Path:
    """Parse a .git entry which may be a directory or a file.

    Git worktrees and submodules use a .git file containing:
        gitdir: <relative-or-absolute-path>

    Args:
        git_path: Path to .git entry.

    Returns:
        Path to the actual git directory.
    """
    if git_path.is_dir():
        return git_path

    # .git is a file (worktree or submodule)
    try:
        content = git_path.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            gitdir_value = content[len("gitdir:") :].strip()
            resolved = (git_path.parent / gitdir_value).resolve()

            # Validate the resolved gitdir path
            if not resolved.is_dir():
                logger.debug("gitdir target is not a directory: %s", resolved)
                return git_path
            if is_blocklisted(resolved):
                logger.warning("gitdir target is a blocklisted root: %s", resolved)
                return git_path

            return resolved
    except OSError:
        pass

    return git_path


def get_remote_url(git_dir: Path) -> str | None:
    """Get the preferred remote URL from a git directory.

    Priority: upstream > origin > first remote found.

    Args:
        git_dir: Path to .git directory.

    Returns:
        Remote URL string, or None if no remotes found.
    """
    config_path = git_dir / "config"
    if not config_path.exists():
        return None

    try:
        config = configparser.RawConfigParser()
        config.read(str(config_path), encoding="utf-8")

        remotes: dict[str, str] = {}
        for section in config.sections():
            if section.startswith('remote "') and section.endswith('"'):
                remote_name = section[8:-1]  # Extract name from 'remote "name"'
                url = config.get(section, "url", fallback=None)
                if url:
                    remotes[remote_name] = url

        if not remotes:
            return None

        # Priority: upstream > origin > first
        for preferred in ("upstream", "origin"):
            if preferred in remotes:
                return remotes[preferred]

        return next(iter(remotes.values()))
    except (configparser.Error, OSError) as e:
        logger.debug("Failed to parse git config: %s", e)
        return None


# Regex patterns for various git URL formats
# HTTPS: https://github.com/org/repo.git
_HTTPS_RE = re.compile(
    r"^https?://(?:(?P<user>[^@]+)@)?(?P<host>[^/:]+)"
    r"(?::(?P<port>\d+))?/(?P<path>.+?)(?:\.git)?/?$"
)

# SSH URL: ssh://git@github.com/org/repo.git or ssh://git@github.com:22/org/repo.git
_SSH_URL_RE = re.compile(
    r"^ssh://(?:(?P<user>[^@]+)@)?(?P<host>[^/:]+)"
    r"(?::(?P<port>\d+))?/(?P<path>.+?)(?:\.git)?/?$"
)

# SCP-like: git@github.com:org/repo.git
_SCP_RE = re.compile(r"^(?:(?P<user>[^@]+)@)?(?P<host>[^:]+):(?!/)(?P<path>.+?)(?:\.git)?/?$")

# Git protocol: git://github.com/org/repo.git
_GIT_RE = re.compile(r"^git://(?P<host>[^/:]+)(?::(?P<port>\d+))?/(?P<path>.+?)(?:\.git)?/?$")

# Azure DevOps: https://dev.azure.com/org/project/_git/repo
_AZURE_RE = re.compile(
    r"^https?://(?P<host>dev\.azure\.com|[^.]+\.visualstudio\.com)/(?P<path>.+?)/?$"
)


def parse_remote_url(url: str) -> ParsedRemoteURL | None:
    """Parse a git remote URL into its components.

    Supports: HTTPS, SSH (URL and SCP-like), git://, Azure DevOps.

    Args:
        url: Git remote URL string.

    Returns:
        ParsedRemoteURL or None if the URL cannot be parsed.
    """
    url = url.strip()

    # Try Azure DevOps first (more specific)
    m = _AZURE_RE.match(url)
    if m:
        path = m.group("path")
        # Normalize Azure DevOps path: org/project/_git/repo -> org/project/repo
        path = re.sub(r"/_git/", "/", path)
        return ParsedRemoteURL(
            host=m.group("host").lower(),
            path=path,
            scheme="https",
            user="",
        )

    # HTTPS
    m = _HTTPS_RE.match(url)
    if m:
        return ParsedRemoteURL(
            host=m.group("host").lower(),
            path=m.group("path"),
            scheme="https",
            user=m.group("user") or "",
        )

    # SSH URL format
    m = _SSH_URL_RE.match(url)
    if m:
        return ParsedRemoteURL(
            host=m.group("host").lower(),
            path=m.group("path"),
            scheme="ssh",
            user=m.group("user") or "git",
        )

    # SCP-like format
    m = _SCP_RE.match(url)
    if m:
        return ParsedRemoteURL(
            host=m.group("host").lower(),
            path=m.group("path"),
            scheme="ssh",
            user=m.group("user") or "git",
        )

    # Git protocol
    m = _GIT_RE.match(url)
    if m:
        return ParsedRemoteURL(
            host=m.group("host").lower(),
            path=m.group("path"),
            scheme="git",
            user="",
        )

    logger.debug("Could not parse git remote URL: %s", url)
    return None


def normalize_project_id(parsed: ParsedRemoteURL) -> str:
    """Normalize a parsed remote URL into a canonical project ID.

    Format: host/org/repo (lowercase host, original case for path).

    Args:
        parsed: ParsedRemoteURL object.

    Returns:
        Normalized project ID string.
    """
    # Strip .git suffix if somehow still present
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]

    return f"{parsed.host}/{path}"
