"""Project detection cascade for memory scoping.

Implements a 7-level cascade to automatically determine which project
a memory operation belongs to. This enables project-scoped memories
where different codebases maintain separate memory spaces.

Detection cascade (stops at first match):
    1. Explicit project parameter (user override)
    2. File path (walk up to .git)
    3. Environment variables ($CLAUDE_PROJECT_DIR, $CURSOR_PROJECT_DIR)
    4. MCP roots (stub for Phase 2)
    5. Config setting (SPATIAL_MEMORY_PROJECT)
    6. Single-project heuristic (DB has only 1 project)
    7. Fallback (empty string = unscoped)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatial_memory.core.git_utils import (
    find_git_root,
    get_remote_url,
    normalize_project_id,
    parse_git_entry,
    parse_remote_url,
)
from spatial_memory.core.monorepo_detection import detect_sub_project, is_workspace_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectIdentity:
    """Resolved project identity.

    Attributes:
        project_id: Canonical project identifier (e.g., "github.com/org/repo").
        source: Which cascade level resolved this identity.
        git_root: Path to git repository root (if found).
        remote_url: Original remote URL (if resolved from git).
        sub_project: Sub-project path within monorepo (if detected).
    """

    project_id: str
    source: str
    git_root: Path | None = None
    remote_url: str | None = None
    sub_project: str | None = None


@dataclass
class ProjectDetectionConfig:
    """Configuration for project detection.

    Attributes:
        explicit_project: Explicit project override from config.
        blocklisted_roots: Additional paths to blocklist.
        env_var_names: Environment variable names to check for project dir.
    """

    explicit_project: str = ""
    blocklisted_roots: list[Path] = field(default_factory=list)
    env_var_names: list[str] = field(
        default_factory=lambda: [
            "CLAUDE_PROJECT_DIR",
            "CURSOR_PROJECT_DIR",
            "WINDSURF_PROJECT_DIR",
        ]
    )


class ProjectDetector:
    """Detects project identity using a 7-level cascade.

    Example:
        detector = ProjectDetector(config)
        identity = detector.detect()
        print(identity.project_id)  # "github.com/org/repo"
    """

    def __init__(
        self,
        config: ProjectDetectionConfig | None = None,
        project_counter: Any | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            config: Detection configuration.
            project_counter: Optional callable returning dict of project -> count
                for single-project heuristic. If None, level 6 is skipped.
        """
        self._config = config or ProjectDetectionConfig()
        self._project_counter = project_counter

    def detect(
        self,
        explicit_project: str | None = None,
        file_path: str | Path | None = None,
    ) -> ProjectIdentity:
        """Run the detection cascade.

        Args:
            explicit_project: Explicit project from tool argument.
            file_path: Optional file path to detect project from.

        Returns:
            ProjectIdentity with the resolved project.
        """
        # Level 1: Explicit project parameter
        result = self._from_explicit(explicit_project)
        if result:
            return result

        # Level 2: File path detection
        if file_path:
            result = self._from_file_path(file_path)
            if result:
                return result

        # Level 3: Environment variables
        result = self._from_env_var()
        if result:
            return result

        # Level 4: MCP roots (stub for Phase 2)
        result = self._from_mcp_roots()
        if result:
            return result

        # Level 5: Config setting
        result = self._from_config()
        if result:
            return result

        # Level 6: Single-project heuristic
        result = self._from_single_project()
        if result:
            return result

        # Level 7: Fallback
        return self._fallback()

    def _from_explicit(self, project: str | None) -> ProjectIdentity | None:
        """Level 1: Use explicit project parameter."""
        if not project:
            return None

        # Wildcard passes through as-is
        if project == "*":
            return ProjectIdentity(project_id="*", source="explicit")

        return ProjectIdentity(project_id=project, source="explicit")

    def _from_file_path(self, file_path: str | Path) -> ProjectIdentity | None:
        """Level 2: Detect project from file path by finding .git."""
        path = Path(file_path)
        return self._resolve_from_directory(path, source="file_path")

    def _from_env_var(self) -> ProjectIdentity | None:
        """Level 3: Detect project from environment variables."""
        for var_name in self._config.env_var_names:
            value = os.environ.get(var_name)
            if value:
                path = Path(value)
                if path.is_dir():
                    result = self._resolve_from_directory(path, source="env_var")
                    if result:
                        return result
        return None

    def _from_mcp_roots(self) -> ProjectIdentity | None:
        """Level 4: Detect project from MCP roots (stub for Phase 2)."""
        return None

    def _from_config(self) -> ProjectIdentity | None:
        """Level 5: Use project from config setting."""
        if self._config.explicit_project:
            return ProjectIdentity(
                project_id=self._config.explicit_project,
                source="config",
            )
        return None

    def _from_single_project(self) -> ProjectIdentity | None:
        """Level 6: Use single-project heuristic."""
        if self._project_counter is None:
            return None

        try:
            projects = self._project_counter()
            # Only use if exactly one non-empty project exists
            non_empty = {k: v for k, v in projects.items() if k}
            if len(non_empty) == 1:
                project_id = next(iter(non_empty.keys()))
                return ProjectIdentity(
                    project_id=project_id,
                    source="single_project",
                )
        except Exception as e:
            logger.debug("Single-project heuristic failed: %s", e)

        return None

    def _fallback(self) -> ProjectIdentity:
        """Level 7: Return unscoped fallback."""
        return ProjectIdentity(project_id="", source="fallback")

    def resolve_from_directory(self, directory: str) -> ProjectIdentity:
        """Resolve project identity from a directory path.

        Used by QueueProcessor for queue files that include project_root_dir.
        Falls back to empty project if git resolution fails.

        Args:
            directory: Filesystem path to resolve from.

        Returns:
            ProjectIdentity (never None — falls back to unscoped).
        """
        path = Path(directory)
        result = self._resolve_from_directory(path, source="queue_file")
        if result is not None:
            return result
        return self._fallback()

    def _resolve_from_directory(self, path: Path, source: str) -> ProjectIdentity | None:
        """Resolve project identity from a directory path.

        Finds the git root, parses the remote URL, and detects sub-projects.

        Args:
            path: Directory path to resolve from.
            source: Cascade source label.

        Returns:
            ProjectIdentity or None if resolution fails.
        """
        git_root = find_git_root(path)
        if git_root is None:
            return None

        git_dir = parse_git_entry(git_root / ".git")
        remote_url = get_remote_url(git_dir)

        if remote_url is None:
            # No remote - use directory name as project ID
            project_id = git_root.name
            return ProjectIdentity(
                project_id=project_id,
                source=source,
                git_root=git_root,
            )

        parsed = parse_remote_url(remote_url)
        if parsed is None:
            # Could not parse remote URL - use directory name
            project_id = git_root.name
            return ProjectIdentity(
                project_id=project_id,
                source=source,
                git_root=git_root,
                remote_url=remote_url,
            )

        project_id = normalize_project_id(parsed)

        # Detect sub-project in monorepo
        sub_project = None
        if is_workspace_root(git_root):
            sub_project = detect_sub_project(path, git_root)

        return ProjectIdentity(
            project_id=project_id,
            source=source,
            git_root=git_root,
            remote_url=remote_url,
            sub_project=sub_project,
        )
