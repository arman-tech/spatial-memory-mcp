"""Strategy pattern for consolidation operations.

This module implements the Strategy design pattern for memory consolidation,
allowing different approaches to be used when merging duplicate memories.

Each strategy determines:
1. Which memory becomes the representative (for non-merge strategies)
2. How to handle the group of duplicates
3. What action to record in the result

Usage in lifecycle.py:
    strategy_impl = get_strategy(strategy_name)
    action = strategy_impl.apply(group_member_dicts, ...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from spatial_memory.core.lifecycle_ops import (
    merge_memory_content,
    merge_memory_metadata,
)
from spatial_memory.core.models import Memory, MemorySource

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


# =============================================================================
# Strategy Result
# =============================================================================


@dataclass
class ConsolidationAction:
    """Result of applying a consolidation strategy to a group.

    Attributes:
        representative_id: ID of the memory kept/created as representative.
        deleted_ids: IDs of memories that were deleted.
        action: Description of what was done.
        memories_merged: Count of memories merged into one.
        memories_deleted: Count of memories deleted.
    """

    representative_id: str
    deleted_ids: list[str]
    action: str
    memories_merged: int = 0
    memories_deleted: int = 0


# =============================================================================
# Strategy Protocol/Base Class
# =============================================================================


class ConsolidationStrategy(ABC):
    """Abstract base class for consolidation strategies.

    Each strategy defines how to select a representative memory
    and how to process a group of duplicate memories.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def select_representative(self, members: list[dict[str, Any]]) -> int:
        """Select the index of the representative memory.

        Args:
            members: List of memory dictionaries with 'created_at',
                    'importance', and 'content' keys.

        Returns:
            Index of the representative memory within the list.
        """
        ...

    @abstractmethod
    def apply(
        self,
        members: list[dict[str, Any]],
        member_ids: list[str],
        namespace: str,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        dry_run: bool = True,
    ) -> ConsolidationAction:
        """Apply the consolidation strategy to a group of memories.

        Args:
            members: List of memory dictionaries.
            member_ids: List of memory IDs.
            namespace: Namespace of the memories.
            repository: Repository for database operations.
            embeddings: Embedding service for generating vectors.
            dry_run: If True, preview without making changes.

        Returns:
            ConsolidationAction describing what was done.
        """
        ...


# =============================================================================
# Keep Representative Strategies
# =============================================================================


class KeepRepresentativeStrategy(ConsolidationStrategy):
    """Base class for strategies that keep one memory and delete others."""

    def apply(
        self,
        members: list[dict[str, Any]],
        member_ids: list[str],
        namespace: str,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        dry_run: bool = True,
    ) -> ConsolidationAction:
        """Keep the representative and delete other members."""
        rep_idx = self.select_representative(members)
        rep_id = member_ids[rep_idx]

        if dry_run:
            return ConsolidationAction(
                representative_id=rep_id,
                deleted_ids=[],
                action="preview",
                memories_merged=0,
                memories_deleted=0,
            )

        # Delete non-representative members
        deleted_ids = []
        for mid in member_ids:
            if mid != rep_id:
                try:
                    repository.delete(mid)
                    deleted_ids.append(mid)
                except Exception as e:
                    # Log but continue with other deletions
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to delete memory {mid}: {e}")

        return ConsolidationAction(
            representative_id=rep_id,
            deleted_ids=deleted_ids,
            action="kept_representative",
            memories_merged=1,
            memories_deleted=len(deleted_ids),
        )


class KeepNewestStrategy(KeepRepresentativeStrategy):
    """Keep the most recently created memory."""

    @property
    def name(self) -> str:
        return "keep_newest"

    def select_representative(self, members: list[dict[str, Any]]) -> int:
        """Select the newest memory by created_at."""
        if not members:
            return 0
        return max(range(len(members)), key=lambda i: members[i].get("created_at", 0))


class KeepOldestStrategy(KeepRepresentativeStrategy):
    """Keep the oldest (original) memory."""

    @property
    def name(self) -> str:
        return "keep_oldest"

    def select_representative(self, members: list[dict[str, Any]]) -> int:
        """Select the oldest memory by created_at."""
        if not members:
            return 0
        return min(
            range(len(members)),
            key=lambda i: members[i].get("created_at", float("inf")),
        )


class KeepHighestImportanceStrategy(KeepRepresentativeStrategy):
    """Keep the memory with the highest importance score."""

    @property
    def name(self) -> str:
        return "keep_highest_importance"

    def select_representative(self, members: list[dict[str, Any]]) -> int:
        """Select the memory with highest importance."""
        if not members:
            return 0
        return max(range(len(members)), key=lambda i: members[i].get("importance", 0.5))


# =============================================================================
# Merge Content Strategy
# =============================================================================


class MergeContentStrategy(ConsolidationStrategy):
    """Merge all memories into a new combined memory."""

    @property
    def name(self) -> str:
        return "merge_content"

    def select_representative(self, members: list[dict[str, Any]]) -> int:
        """Select the memory with longest content as base.

        For merge strategy, this determines which memory's structure
        is used as the base for the merged content.
        """
        if not members:
            return 0
        return max(
            range(len(members)),
            key=lambda i: len(members[i].get("content", "")),
        )

    def apply(
        self,
        members: list[dict[str, Any]],
        member_ids: list[str],
        namespace: str,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        dry_run: bool = True,
    ) -> ConsolidationAction:
        """Merge all memories into a new combined memory."""
        rep_idx = self.select_representative(members)
        rep_id = member_ids[rep_idx]

        if dry_run:
            return ConsolidationAction(
                representative_id=rep_id,
                deleted_ids=[],
                action="preview",
                memories_merged=0,
                memories_deleted=0,
            )

        import logging

        logger = logging.getLogger(__name__)

        # Create merged content
        group_contents = [str(m["content"]) for m in members]
        merged_content = merge_memory_content(group_contents)
        merged_meta = merge_memory_metadata(members)

        # Generate new embedding
        new_vector = embeddings.embed(merged_content)

        # Prepare merged memory with pending status marker
        pending_metadata = merged_meta.get("metadata", {}).copy()
        pending_metadata["_consolidation_status"] = "pending"
        pending_metadata["_consolidation_source_ids"] = member_ids

        merged_memory = Memory(
            id="",  # Will be assigned
            content=merged_content,
            namespace=namespace,
            tags=merged_meta.get("tags", []),
            importance=merged_meta.get("importance", 0.5),
            source=MemorySource.CONSOLIDATED,
            metadata=pending_metadata,
        )

        # ADD FIRST pattern: add merged memory before deleting originals
        try:
            new_id = repository.add(merged_memory, new_vector)
        except Exception as add_err:
            logger.error(
                f"Consolidation add failed, originals preserved. "
                f"Group IDs: {member_ids}. Error: {add_err}"
            )
            return ConsolidationAction(
                representative_id=rep_id,
                deleted_ids=[],
                action="failed",
                memories_merged=0,
                memories_deleted=0,
            )

        # Delete originals using batch operation
        deleted_ids: list[str] = []
        try:
            deleted_count, deleted_ids = repository.delete_batch(member_ids)
        except Exception as del_err:
            logger.warning(
                f"Consolidation delete failed after add. "
                f"Merged memory {new_id} has pending status. "
                f"Original IDs: {member_ids}. Error: {del_err}"
            )
            return ConsolidationAction(
                representative_id=new_id,
                deleted_ids=[],
                action="failed",
                memories_merged=0,
                memories_deleted=0,
            )

        # Activate merged memory by removing pending status
        try:
            final_metadata = merged_meta.get("metadata", {}).copy()
            repository.update(new_id, {"metadata": final_metadata})
        except Exception as update_err:
            # Minor issue - memory works, just has pending marker
            logger.warning(f"Failed to remove pending status from {new_id}: {update_err}")
            # Don't fail - consolidation succeeded

        return ConsolidationAction(
            representative_id=new_id,
            deleted_ids=deleted_ids,
            action="merged",
            memories_merged=1,
            memories_deleted=len(deleted_ids),
        )


# =============================================================================
# Strategy Registry
# =============================================================================

# Type for valid strategy names
StrategyName = Literal["keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"]

# Registry of available strategies
_STRATEGIES: dict[str, ConsolidationStrategy] = {
    "keep_newest": KeepNewestStrategy(),
    "keep_oldest": KeepOldestStrategy(),
    "keep_highest_importance": KeepHighestImportanceStrategy(),
    "merge_content": MergeContentStrategy(),
}


def get_strategy(name: str) -> ConsolidationStrategy:
    """Get a consolidation strategy by name.

    Args:
        name: Strategy name (keep_newest, keep_oldest,
              keep_highest_importance, merge_content).

    Returns:
        The corresponding ConsolidationStrategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    strategy = _STRATEGIES.get(name)
    if strategy is None:
        valid_names = ", ".join(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Valid strategies: {valid_names}")
    return strategy


def register_strategy(name: str, strategy: ConsolidationStrategy) -> None:
    """Register a custom consolidation strategy.

    This allows extending the consolidation system with custom strategies
    without modifying the core code.

    Args:
        name: Unique name for the strategy.
        strategy: ConsolidationStrategy instance.
    """
    _STRATEGIES[name] = strategy


def list_strategies() -> list[str]:
    """List all registered strategy names.

    Returns:
        List of registered strategy names.
    """
    return list(_STRATEGIES.keys())
