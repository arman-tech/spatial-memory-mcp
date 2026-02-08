"""Tests for git_utils module."""

import configparser
from pathlib import Path

import pytest

from spatial_memory.adapters.git_utils import (
    ParsedRemoteURL,
    find_git_root,
    get_remote_url,
    normalize_project_id,
    parse_git_entry,
    parse_remote_url,
)


@pytest.mark.unit
class TestFindGitRoot:
    """Tests for find_git_root."""

    def test_finds_git_root(self, tmp_path: Path) -> None:
        """Test finding .git directory."""
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "deep"
        subdir.mkdir(parents=True)
        assert find_git_root(subdir, blocklist_check=False) == tmp_path

    def test_no_git_root(self, tmp_path: Path) -> None:
        """Test when no .git directory exists."""
        subdir = tmp_path / "no-git"
        subdir.mkdir()
        assert find_git_root(subdir, blocklist_check=False) is None

    def test_finds_at_start_path(self, tmp_path: Path) -> None:
        """Test finding .git at the start path itself."""
        (tmp_path / ".git").mkdir()
        assert find_git_root(tmp_path, blocklist_check=False) == tmp_path

    def test_file_input(self, tmp_path: Path) -> None:
        """Test with file input (should use parent)."""
        (tmp_path / ".git").mkdir()
        test_file = tmp_path / "src" / "main.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()
        assert find_git_root(test_file, blocklist_check=False) == tmp_path


@pytest.mark.unit
class TestParseGitEntry:
    """Tests for parse_git_entry."""

    def test_directory_git(self, tmp_path: Path) -> None:
        """Test .git as a directory."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        assert parse_git_entry(git_dir) == git_dir

    def test_worktree_git_file(self, tmp_path: Path) -> None:
        """Test .git as a file (worktree)."""
        actual_git = tmp_path / "actual-git-dir"
        actual_git.mkdir()

        worktree = tmp_path / "worktree"
        worktree.mkdir()
        git_file = worktree / ".git"
        git_file.write_text(f"gitdir: {actual_git}")

        result = parse_git_entry(git_file)
        assert result.resolve() == actual_git.resolve()

    def test_submodule_git_file(self, tmp_path: Path) -> None:
        """Test .git file with relative path (submodule)."""
        main_git = tmp_path / ".git" / "modules" / "sub"
        main_git.mkdir(parents=True)

        sub = tmp_path / "sub"
        sub.mkdir()
        git_file = sub / ".git"
        git_file.write_text("gitdir: ../.git/modules/sub")

        result = parse_git_entry(git_file)
        assert result.resolve() == main_git.resolve()


@pytest.mark.unit
class TestGetRemoteUrl:
    """Tests for get_remote_url."""

    def _write_git_config(self, git_dir: Path, remotes: dict[str, str]) -> None:
        """Helper to write a git config file."""
        config = configparser.ConfigParser()
        for name, url in remotes.items():
            config[f'remote "{name}"'] = {"url": url}
        config_path = git_dir / "config"
        with open(config_path, "w") as f:
            config.write(f)

    def test_origin_remote(self, tmp_path: Path) -> None:
        """Test getting origin remote."""
        self._write_git_config(tmp_path, {"origin": "https://github.com/org/repo.git"})
        assert get_remote_url(tmp_path) == "https://github.com/org/repo.git"

    def test_upstream_preferred(self, tmp_path: Path) -> None:
        """Test that upstream is preferred over origin."""
        self._write_git_config(
            tmp_path,
            {
                "origin": "https://github.com/fork/repo.git",
                "upstream": "https://github.com/org/repo.git",
            },
        )
        assert get_remote_url(tmp_path) == "https://github.com/org/repo.git"

    def test_no_remotes(self, tmp_path: Path) -> None:
        """Test when no remotes configured."""
        config = configparser.ConfigParser()
        config_path = tmp_path / "config"
        with open(config_path, "w") as f:
            config.write(f)
        assert get_remote_url(tmp_path) is None

    def test_no_config_file(self, tmp_path: Path) -> None:
        """Test when config file doesn't exist."""
        assert get_remote_url(tmp_path) is None

    def test_first_remote_fallback(self, tmp_path: Path) -> None:
        """Test falling back to first remote."""
        self._write_git_config(tmp_path, {"custom": "https://github.com/org/repo.git"})
        assert get_remote_url(tmp_path) == "https://github.com/org/repo.git"


@pytest.mark.unit
class TestParseRemoteUrl:
    """Tests for parse_remote_url."""

    def test_https_github(self) -> None:
        """Test HTTPS GitHub URL."""
        result = parse_remote_url("https://github.com/org/repo.git")
        assert result is not None
        assert result.host == "github.com"
        assert result.path == "org/repo"
        assert result.scheme == "https"

    def test_https_no_git_suffix(self) -> None:
        """Test HTTPS URL without .git suffix."""
        result = parse_remote_url("https://github.com/org/repo")
        assert result is not None
        assert result.path == "org/repo"

    def test_ssh_scp_format(self) -> None:
        """Test SCP-like SSH URL."""
        result = parse_remote_url("git@github.com:org/repo.git")
        assert result is not None
        assert result.host == "github.com"
        assert result.path == "org/repo"
        assert result.scheme == "ssh"
        assert result.user == "git"

    def test_ssh_url_format(self) -> None:
        """Test SSH URL format."""
        result = parse_remote_url("ssh://git@github.com/org/repo.git")
        assert result is not None
        assert result.host == "github.com"
        assert result.path == "org/repo"
        assert result.scheme == "ssh"

    def test_git_protocol(self) -> None:
        """Test git:// protocol."""
        result = parse_remote_url("git://github.com/org/repo.git")
        assert result is not None
        assert result.host == "github.com"
        assert result.path == "org/repo"
        assert result.scheme == "git"

    def test_azure_devops(self) -> None:
        """Test Azure DevOps URL."""
        result = parse_remote_url("https://dev.azure.com/org/project/_git/repo")
        assert result is not None
        assert result.host == "dev.azure.com"
        assert result.path == "org/project/repo"

    def test_gitlab_nested_groups(self) -> None:
        """Test GitLab nested group URL."""
        result = parse_remote_url("https://gitlab.com/group/subgroup/repo.git")
        assert result is not None
        assert result.host == "gitlab.com"
        assert result.path == "group/subgroup/repo"

    def test_https_with_credentials(self) -> None:
        """Test HTTPS URL with embedded credentials."""
        result = parse_remote_url("https://user@github.com/org/repo.git")
        assert result is not None
        assert result.user == "user"
        assert result.path == "org/repo"

    def test_ssh_with_port(self) -> None:
        """Test SSH URL with port."""
        result = parse_remote_url("ssh://git@github.com:22/org/repo.git")
        assert result is not None
        assert result.host == "github.com"
        assert result.path == "org/repo"

    def test_trailing_slash(self) -> None:
        """Test URL with trailing slash."""
        result = parse_remote_url("https://github.com/org/repo/")
        assert result is not None
        assert result.path == "org/repo"

    def test_invalid_url(self) -> None:
        """Test invalid URL returns None."""
        assert parse_remote_url("not a url") is None

    def test_empty_url(self) -> None:
        """Test empty URL returns None."""
        assert parse_remote_url("") is None


@pytest.mark.unit
class TestNormalizeProjectId:
    """Tests for normalize_project_id."""

    def test_basic_normalization(self) -> None:
        """Test basic project ID normalization."""
        parsed = ParsedRemoteURL(host="github.com", path="org/repo", scheme="https", user="")
        assert normalize_project_id(parsed) == "github.com/org/repo"

    def test_host_lowercase(self) -> None:
        """Test that host is already lowercase."""
        parsed = ParsedRemoteURL(host="github.com", path="Org/Repo", scheme="https", user="")
        result = normalize_project_id(parsed)
        assert result == "github.com/Org/Repo"

    def test_strips_trailing_slash(self) -> None:
        """Test stripping trailing slash from path."""
        parsed = ParsedRemoteURL(host="github.com", path="org/repo/", scheme="https", user="")
        result = normalize_project_id(parsed)
        assert result == "github.com/org/repo"
