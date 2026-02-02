"""Lifecycle service for memory management operations.

This service provides the application layer for memory lifecycle operations:
- decay: Apply time/access-based importance reduction
- reinforce: Boost memory importance
- extract: Auto-extract memories from text
- consolidate: Merge similar/duplicate memories

These operations address the "Context Window Pollution" problem by providing
utility-based memory management with cognitive-like dynamics.

The service uses dependency injection for repository and embedding services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from spatial_memory.core.consolidation_strategies import (
    ConsolidationAction,
    get_strategy,
)
from spatial_memory.core.errors import (
    ConsolidationError,
    DecayError,
    ExtractionError,
    ReinforcementError,
    ValidationError,
)
from spatial_memory.core.lifecycle_ops import (
    apply_decay,
    calculate_decay_factor,
    calculate_reinforcement,
    combined_similarity,
    extract_candidates,
    find_duplicate_groups,
    jaccard_similarity,
)
from spatial_memory.core.models import (
    ConsolidateResult,
    ConsolidationGroup,
    DecayedMemory,
    DecayResult,
    ExtractedMemory,
    ExtractResult,
    Memory,
    MemorySource,
    ReinforcedMemory,
    ReinforceResult,
)
from spatial_memory.core.utils import to_naive_utc, utc_now, utc_now_naive
from spatial_memory.core.validation import validate_namespace

# Alias for backward compatibility
ConsolidationGroupResult = ConsolidationGroup

logger = logging.getLogger(__name__)

# Explicit exports for mypy
__all__ = [
    # Result types (re-exported from models)
    "ConsolidateResult",
    "ConsolidationGroupResult",
    "DecayedMemory",
    "DecayResult",
    "ExtractedMemory",
    "ExtractResult",
    "ReinforcedMemory",
    "ReinforceResult",
    # Config and service
    "LifecycleConfig",
    "LifecycleService",
]

if TYPE_CHECKING:
    from spatial_memory.ports.repositories import (
        EmbeddingServiceProtocol,
        MemoryRepositoryProtocol,
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle operations.

    Attributes:
        decay_default_half_life_days: Default half-life for exponential decay.
        decay_default_function: Default decay function type.
        decay_min_importance_floor: Minimum importance after decay.
        decay_batch_size: Batch size for decay updates.
        reinforce_default_boost: Default boost amount.
        reinforce_max_importance: Maximum importance after reinforcement.
        extract_max_text_length: Maximum text length for extraction.
        extract_max_candidates: Maximum candidates per extraction.
        extract_default_importance: Default importance for extracted memories.
        extract_default_namespace: Default namespace for extracted memories.
        consolidate_min_threshold: Minimum similarity threshold.
        consolidate_content_weight: Weight of content overlap vs vector similarity.
        consolidate_max_batch: Maximum memories per consolidation pass.
    """

    # Decay settings
    decay_default_half_life_days: float = 30.0
    decay_default_function: str = "exponential"  # exponential, linear, step
    decay_min_importance_floor: float = 0.1
    decay_batch_size: int = 500

    # Reinforce settings
    reinforce_default_boost: float = 0.1
    reinforce_max_importance: float = 1.0

    # Extract settings
    extract_max_text_length: int = 50000
    extract_max_candidates: int = 20
    extract_default_importance: float = 0.4
    extract_default_namespace: str = "extracted"

    # Consolidate settings
    consolidate_min_threshold: float = 0.7
    consolidate_content_weight: float = 0.3
    consolidate_max_batch: int = 1000
    consolidate_chunk_size: int = 200  # Process in smaller chunks for memory efficiency


# =============================================================================
# Service Implementation
# =============================================================================


class LifecycleService:
    """Service for memory lifecycle management.

    Uses Clean Architecture - depends on protocol interfaces, not implementations.
    Implements cognitive-like memory dynamics: decay, reinforcement, extraction,
    and consolidation.
    """

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: LifecycleConfig | None = None,
    ) -> None:
        """Initialize the lifecycle service.

        Args:
            repository: Repository for memory storage.
            embeddings: Service for generating embeddings.
            config: Optional configuration (uses defaults if not provided).
        """
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or LifecycleConfig()

    def decay(
        self,
        namespace: str | None = None,
        decay_function: Literal["exponential", "linear", "step"] = "exponential",
        half_life_days: float | None = None,
        min_importance: float | None = None,
        access_weight: float = 0.3,
        dry_run: bool = True,
    ) -> DecayResult:
        """Apply time and access-based decay to memory importance scores.

        Implements the "forgetting curve" - memories not accessed become less
        important over time. More accesses and higher base importance slow decay.

        Args:
            namespace: Namespace to decay (all if not specified).
            decay_function: Decay curve shape ("exponential", "linear", "step").
            half_life_days: Days until importance halves (default from config).
            min_importance: Minimum importance floor (default from config).
            access_weight: Weight of access count in decay calculation (0-1).
            dry_run: Preview changes without applying (default True).

        Returns:
            DecayResult with decay statistics and affected memories.

        Raises:
            DecayError: If decay calculation or application fails.
            ValidationError: If input validation fails.
        """
        # Validate inputs
        if namespace is not None:
            namespace = validate_namespace(namespace)

        if decay_function not in ("exponential", "linear", "step"):
            raise ValidationError(
                f"Invalid decay function: {decay_function}. "
                "Must be 'exponential', 'linear', or 'step'."
            )

        if not 0.0 <= access_weight <= 1.0:
            raise ValidationError("access_weight must be between 0.0 and 1.0")

        # Use config defaults
        effective_half_life = (
            half_life_days
            if half_life_days is not None
            else self._config.decay_default_half_life_days
        )
        effective_min_importance = (
            min_importance
            if min_importance is not None
            else self._config.decay_min_importance_floor
        )

        if effective_half_life < 1.0:
            raise ValidationError("half_life_days must be at least 1.0")
        if not 0.0 <= effective_min_importance <= 0.5:
            raise ValidationError("min_importance must be between 0.0 and 0.5")

        try:
            # Fetch all memories for decay calculation
            all_memories = self._repo.get_all(
                namespace=namespace,
                limit=self._config.decay_batch_size * 10,  # Allow multiple batches
            )

            if not all_memories:
                logger.info("No memories found for decay")
                return DecayResult(
                    memories_analyzed=0,
                    memories_decayed=0,
                    avg_decay_factor=1.0,
                    decayed_memories=[],
                    dry_run=dry_run,
                )

            # Use naive UTC to match LanceDB storage format (avoids timezone mismatch errors)
            now = utc_now_naive()
            decayed_memories: list[DecayedMemory] = []
            total_decay_factor = 0.0
            memories_to_update: list[tuple[str, float]] = []

            for memory, _ in all_memories:
                # Normalize last_accessed to naive UTC (handle both aware and naive timestamps)
                last_accessed = to_naive_utc(memory.last_accessed)

                # Calculate days since last access
                days_since_access = (now - last_accessed).total_seconds() / 86400

                # Calculate decay factor
                decay_factor = calculate_decay_factor(
                    days_since_access=days_since_access,
                    access_count=memory.access_count,
                    base_importance=memory.importance,
                    decay_function=decay_function,
                    half_life_days=effective_half_life,
                    access_weight=access_weight,
                )

                # Apply decay to get new importance
                new_importance = apply_decay(
                    current_importance=memory.importance,
                    decay_factor=decay_factor,
                    min_importance=effective_min_importance,
                )

                # Track if importance actually changed
                if abs(new_importance - memory.importance) > 0.001:
                    decayed_memories.append(
                        DecayedMemory(
                            id=memory.id,
                            content_preview=memory.content[:100] + "..."
                            if len(memory.content) > 100
                            else memory.content,
                            old_importance=memory.importance,
                            new_importance=new_importance,
                            decay_factor=decay_factor,
                            days_since_access=int(days_since_access),
                            access_count=memory.access_count,
                        )
                    )
                    memories_to_update.append((memory.id, new_importance))

                total_decay_factor += decay_factor

            avg_decay = (
                total_decay_factor / len(all_memories) if all_memories else 1.0
            )

            # Apply updates if not dry run - use batch update for efficiency
            failed_updates: list[str] = []
            if not dry_run and memories_to_update:
                logger.info(f"Applying decay to {len(memories_to_update)} memories")
                # Convert to batch update format: list of (memory_id, updates_dict)
                batch_updates = [
                    (memory_id, {"importance": new_importance})
                    for memory_id, new_importance in memories_to_update
                ]
                try:
                    success_count, failed_ids = self._repo.update_batch(batch_updates)
                    failed_updates = failed_ids
                    logger.debug(
                        f"Batch decay update: {success_count} succeeded, "
                        f"{len(failed_ids)} failed"
                    )
                except Exception as e:
                    logger.warning(f"Batch decay update failed: {e}")
                    # Fall back to individual updates on batch failure
                    for memory_id, new_importance in memories_to_update:
                        try:
                            self._repo.update(memory_id, {"importance": new_importance})
                        except Exception as update_err:
                            logger.warning(f"Failed to update {memory_id}: {update_err}")
                            failed_updates.append(memory_id)

            return DecayResult(
                memories_analyzed=len(all_memories),
                memories_decayed=len(decayed_memories),
                avg_decay_factor=avg_decay,
                decayed_memories=decayed_memories,
                dry_run=dry_run,
                failed_updates=failed_updates,
            )

        except (ValidationError, DecayError):
            raise
        except Exception as e:
            raise DecayError(f"Decay operation failed: {e}") from e

    def reinforce(
        self,
        memory_ids: list[str],
        boost_type: Literal["additive", "multiplicative", "set_value"] = "additive",
        boost_amount: float | None = None,
        update_access: bool = True,
    ) -> ReinforceResult:
        """Boost memory importance based on usage or explicit feedback.

        Reinforcement increases importance and can reset decay timer by
        updating the access timestamp.

        Args:
            memory_ids: Memory IDs to reinforce.
            boost_type: Type of boost ("additive", "multiplicative", "set_value").
            boost_amount: Amount for boost (default from config).
            update_access: Also update last_accessed timestamp (default True).

        Returns:
            ReinforceResult with reinforcement statistics.

        Raises:
            ReinforcementError: If reinforcement fails.
            ValidationError: If input validation fails.
        """
        if not memory_ids:
            raise ValidationError("memory_ids cannot be empty")

        if boost_type not in ("additive", "multiplicative", "set_value"):
            raise ValidationError(
                f"Invalid boost_type: {boost_type}. "
                "Must be 'additive', 'multiplicative', or 'set_value'."
            )

        effective_boost = (
            boost_amount
            if boost_amount is not None
            else self._config.reinforce_default_boost
        )

        if effective_boost < 0.0 or effective_boost > 1.0:
            raise ValidationError("boost_amount must be between 0.0 and 1.0")

        try:
            reinforced: list[ReinforcedMemory] = []
            not_found: list[str] = []
            failed_updates: list[str] = []
            total_boost = 0.0

            # Batch fetch all memories in a single query
            memory_map = self._repo.get_batch(memory_ids)

            # Track which memories were not found
            for memory_id in memory_ids:
                if memory_id not in memory_map:
                    not_found.append(memory_id)
                    logger.warning(f"Memory not found for reinforcement: {memory_id}")

            if not memory_map:
                return ReinforceResult(
                    memories_reinforced=0,
                    avg_boost=0.0,
                    reinforced=[],
                    not_found=not_found,
                    failed_updates=[],
                )

            # Calculate reinforcement for all found memories
            now = utc_now()
            batch_updates: list[tuple[str, dict[str, Any]]] = []
            # Tuple: (id, memory, new_importance, boost_applied)
            reinforcement_info: list[tuple[str, Memory, float, float]] = []

            for memory_id, memory in memory_map.items():
                # Calculate new importance
                new_importance, actual_boost = calculate_reinforcement(
                    current_importance=memory.importance,
                    boost_type=boost_type,
                    boost_amount=effective_boost,
                    max_importance=self._config.reinforce_max_importance,
                )

                # Prepare update
                updates: dict[str, Any] = {"importance": new_importance}
                if update_access:
                    updates["last_accessed"] = now
                    updates["access_count"] = memory.access_count + 1

                batch_updates.append((memory_id, updates))
                reinforcement_info.append((memory_id, memory, new_importance, actual_boost))

            # Apply all updates in a single batch operation
            try:
                success_count, batch_failed_ids = self._repo.update_batch(batch_updates)
                failed_updates = batch_failed_ids
                logger.debug(
                    f"Batch reinforce update: {success_count} succeeded, "
                    f"{len(batch_failed_ids)} failed"
                )
            except Exception as e:
                logger.warning(
                    f"Batch reinforce update failed: {e}, falling back to individual updates"
                )
                # Fall back to individual updates on batch failure
                batch_failed_ids = []
                for memory_id, updates in batch_updates:
                    try:
                        self._repo.update(memory_id, updates)
                    except Exception as update_err:
                        logger.warning(f"Failed to reinforce {memory_id}: {update_err}")
                        batch_failed_ids.append(memory_id)
                failed_updates = batch_failed_ids

            # Build reinforced results for successful updates
            failed_set = set(failed_updates)
            for memory_id, memory, new_importance, actual_boost in reinforcement_info:
                if memory_id not in failed_set:
                    reinforced.append(
                        ReinforcedMemory(
                            id=memory_id,
                            content_preview=memory.content[:100] + "..."
                            if len(memory.content) > 100
                            else memory.content,
                            old_importance=memory.importance,
                            new_importance=new_importance,
                            boost_applied=actual_boost,
                        )
                    )
                    total_boost += actual_boost

            avg_boost = total_boost / len(reinforced) if reinforced else 0.0

            return ReinforceResult(
                memories_reinforced=len(reinforced),
                avg_boost=avg_boost,
                reinforced=reinforced,
                not_found=not_found,
                failed_updates=failed_updates,
            )

        except (ValidationError, ReinforcementError):
            raise
        except Exception as e:
            raise ReinforcementError(f"Reinforcement operation failed: {e}") from e

    def extract(
        self,
        text: str,
        namespace: str | None = None,
        min_confidence: float = 0.5,
        deduplicate: bool = True,
        dedup_threshold: float = 0.9,
    ) -> ExtractResult:
        """Automatically extract memories from conversation text.

        Uses pattern matching to identify facts, decisions, and key information
        from unstructured text.

        Args:
            text: Text to extract memories from.
            namespace: Namespace for extracted memories (default from config).
            min_confidence: Minimum confidence to extract (0-1).
            deduplicate: Skip if similar memory exists (default True).
            dedup_threshold: Similarity threshold for deduplication (0.7-0.99).

        Returns:
            ExtractResult with extraction statistics and created memories.

        Raises:
            ExtractionError: If extraction fails.
            ValidationError: If input validation fails.
        """
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")

        if len(text) > self._config.extract_max_text_length:
            raise ValidationError(
                f"Text exceeds maximum length of {self._config.extract_max_text_length}"
            )

        if not 0.0 <= min_confidence <= 1.0:
            raise ValidationError("min_confidence must be between 0.0 and 1.0")

        if not 0.7 <= dedup_threshold <= 0.99:
            raise ValidationError("dedup_threshold must be between 0.7 and 0.99")

        effective_namespace = namespace or self._config.extract_default_namespace
        effective_namespace = validate_namespace(effective_namespace)

        try:
            # Extract candidates using pattern matching
            candidates = extract_candidates(
                text=text,
                min_confidence=min_confidence,
                max_candidates=self._config.extract_max_candidates,
            )

            if not candidates:
                logger.info("No extraction candidates found")
                return ExtractResult(
                    candidates_found=0,
                    memories_created=0,
                    deduplicated_count=0,
                    extractions=[],
                )

            # Generate embeddings for all candidates in a single batch
            # This is much more efficient than generating one at a time
            candidate_texts = [c.content for c in candidates]
            candidate_vectors = self._embeddings.embed_batch(candidate_texts)
            logger.debug(f"Generated {len(candidate_vectors)} embeddings in batch")

            extractions: list[ExtractedMemory] = []
            memories_created = 0
            deduplicated_count = 0

            for candidate, vector in zip(candidates, candidate_vectors):
                extraction = ExtractedMemory(
                    content=candidate.content,
                    confidence=candidate.confidence,
                    pattern_matched=candidate.pattern_type,
                    start_pos=candidate.start_pos,
                    end_pos=candidate.end_pos,
                    stored=False,
                    memory_id=None,
                )

                # Check for duplicates if requested (use pre-computed vector)
                if deduplicate:
                    is_duplicate = self._check_duplicate_with_vector(
                        content=candidate.content,
                        vector=vector,
                        namespace=effective_namespace,
                        threshold=dedup_threshold,
                    )
                    if is_duplicate:
                        deduplicated_count += 1
                        extractions.append(extraction)
                        continue

                # Store the extracted memory (use pre-computed vector)
                try:
                    memory_id = self._store_extracted_memory_with_vector(
                        content=candidate.content,
                        vector=vector,
                        namespace=effective_namespace,
                        confidence=candidate.confidence,
                        pattern_type=candidate.pattern_type,
                    )
                    extraction.stored = True
                    extraction.memory_id = memory_id
                    memories_created += 1
                except Exception as e:
                    logger.warning(f"Failed to store extraction: {e}")

                extractions.append(extraction)

            return ExtractResult(
                candidates_found=len(candidates),
                memories_created=memories_created,
                deduplicated_count=deduplicated_count,
                extractions=extractions,
            )

        except (ValidationError, ExtractionError):
            raise
        except Exception as e:
            raise ExtractionError(f"Extraction operation failed: {e}") from e

    def consolidate(
        self,
        namespace: str,
        similarity_threshold: float = 0.85,
        strategy: Literal[
            "keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"
        ] = "keep_highest_importance",
        dry_run: bool = True,
        max_groups: int = 50,
    ) -> ConsolidateResult:
        """Merge similar or duplicate memories to reduce redundancy.

        Finds memories above similarity threshold and merges them according
        to the specified strategy.

        Args:
            namespace: Namespace to consolidate (required).
            similarity_threshold: Minimum similarity for duplicates (0.7-0.99).
            strategy: How to handle duplicates:
                - "keep_newest": Keep most recent memory
                - "keep_oldest": Keep oldest memory
                - "keep_highest_importance": Keep highest importance
                - "merge_content": Combine content and re-embed
            dry_run: Preview without changes (default True).
            max_groups: Maximum groups to process.

        Returns:
            ConsolidateResult with consolidation statistics.

        Raises:
            ConsolidationError: If consolidation fails.
            ValidationError: If input validation fails.
        """
        namespace = validate_namespace(namespace)

        if not 0.7 <= similarity_threshold <= 0.99:
            raise ValidationError(
                "similarity_threshold must be between 0.7 and 0.99"
            )

        # Validate strategy using the strategy registry
        try:
            strategy_impl = get_strategy(strategy)
        except ValueError as e:
            raise ValidationError(str(e)) from e

        if max_groups < 1:
            raise ValidationError("max_groups must be at least 1")

        try:
            # Get total count to decide processing strategy
            total_count = self._repo.count(namespace=namespace)

            if total_count < 2:
                logger.info("Not enough memories for consolidation")
                return ConsolidateResult(
                    groups_found=0,
                    memories_merged=0,
                    memories_deleted=0,
                    groups=[],
                    dry_run=dry_run,
                )

            # Use chunked processing for large namespaces to reduce memory usage
            chunk_size = min(
                self._config.consolidate_chunk_size,
                self._config.consolidate_max_batch,
            )
            use_chunked = total_count > chunk_size

            if use_chunked:
                logger.info(
                    f"Using chunked consolidation: {total_count} memories in "
                    f"chunks of {chunk_size}"
                )
                return self._consolidate_chunked(
                    namespace=namespace,
                    similarity_threshold=similarity_threshold,
                    strategy=strategy,
                    dry_run=dry_run,
                    max_groups=max_groups,
                    chunk_size=chunk_size,
                )

            # Standard single-pass processing for smaller namespaces
            all_memories = self._repo.get_all(
                namespace=namespace,
                limit=self._config.consolidate_max_batch,
            )

            # Build lookup structures
            memories = [m for m, _ in all_memories]
            vectors_list = [v for _, v in all_memories]
            vectors_array = np.array(vectors_list, dtype=np.float32)
            memory_ids = [m.id for m in memories]
            contents = [m.content for m in memories]
            memory_dicts: list[dict[str, Any]] = [
                {
                    "id": m.id,
                    "content": m.content,
                    "created_at": m.created_at,
                    "last_accessed": m.last_accessed,
                    "access_count": m.access_count,
                    "importance": m.importance,
                    "tags": list(m.tags),
                }
                for m in memories
            ]

            # Find duplicate groups using array-based API
            group_indices = find_duplicate_groups(
                memory_ids=memory_ids,
                vectors=vectors_array,
                contents=contents,
                threshold=similarity_threshold,
                content_weight=self._config.consolidate_content_weight,
            )

            # Limit groups
            group_indices = group_indices[:max_groups]

            if not group_indices:
                logger.info("No duplicate groups found")
                return ConsolidateResult(
                    groups_found=0,
                    memories_merged=0,
                    memories_deleted=0,
                    groups=[],
                    dry_run=dry_run,
                )

            result_groups: list[ConsolidationGroupResult] = []
            memories_merged = 0
            memories_deleted = 0

            for member_indices in group_indices:
                group_member_dicts = [memory_dicts[i] for i in member_indices]
                group_member_ids = [str(d["id"]) for d in group_member_dicts]

                # Calculate average similarity for the group
                total_sim = 0.0
                pair_count = 0
                for i_idx, i in enumerate(member_indices):
                    for j in member_indices[i_idx + 1 :]:
                        # Vector similarity
                        v1, v2 = vectors_array[i], vectors_array[j]
                        dot = float(np.dot(v1, v2))
                        norm1 = float(np.linalg.norm(v1))
                        norm2 = float(np.linalg.norm(v2))
                        if norm1 > 1e-10 and norm2 > 1e-10:
                            v_sim = dot / (norm1 * norm2)
                        else:
                            v_sim = 0.0
                        # Content similarity
                        c_sim = jaccard_similarity(contents[i], contents[j])
                        combined = combined_similarity(
                            v_sim, c_sim, self._config.consolidate_content_weight
                        )
                        total_sim += combined
                        pair_count += 1
                avg_similarity = total_sim / pair_count if pair_count > 0 else 0.0

                # Apply consolidation strategy
                try:
                    action_result: ConsolidationAction = strategy_impl.apply(
                        members=group_member_dicts,
                        member_ids=group_member_ids,
                        namespace=namespace,
                        repository=self._repo,
                        embeddings=self._embeddings,
                        dry_run=dry_run,
                    )
                    memories_merged += action_result.memories_merged
                    memories_deleted += action_result.memories_deleted
                except Exception as e:
                    logger.warning(f"Failed to consolidate group: {e}")
                    action_result = ConsolidationAction(
                        representative_id=group_member_ids[0],
                        deleted_ids=[],
                        action="failed",
                    )

                result_groups.append(
                    ConsolidationGroupResult(
                        representative_id=action_result.representative_id,
                        member_ids=group_member_ids,
                        avg_similarity=avg_similarity,
                        action_taken=action_result.action,
                    )
                )

            return ConsolidateResult(
                groups_found=len(group_indices),
                memories_merged=memories_merged,
                memories_deleted=memories_deleted,
                groups=result_groups,
                dry_run=dry_run,
            )

        except (ValidationError, ConsolidationError):
            raise
        except Exception as e:
            raise ConsolidationError(f"Consolidation operation failed: {e}") from e

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _consolidate_chunked(
        self,
        namespace: str,
        similarity_threshold: float,
        strategy: Literal[
            "keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"
        ],
        dry_run: bool,
        max_groups: int,
        chunk_size: int,
    ) -> ConsolidateResult:
        """Process consolidation in memory-efficient chunks.

        Processes memories in smaller chunks to reduce peak memory usage.
        Note: This may miss duplicates that span chunk boundaries.

        Args:
            namespace: Namespace to consolidate.
            similarity_threshold: Minimum similarity for duplicates.
            strategy: How to handle duplicates.
            dry_run: Preview without changes.
            max_groups: Maximum groups to process total.
            chunk_size: Memories per chunk.

        Returns:
            Aggregated ConsolidateResult from all chunks.
        """
        all_groups: list[ConsolidationGroupResult] = []
        total_merged = 0
        total_deleted = 0
        offset = 0
        groups_remaining = max_groups

        while groups_remaining > 0:
            # Fetch chunk of memories
            chunk_memories = self._repo.get_all(
                namespace=namespace,
                limit=chunk_size,
            )

            # Skip already processed memories by filtering by offset
            # Note: This is a simplified approach - in production, you'd want
            # to track processed IDs or use cursor-based pagination
            if offset > 0:
                # Re-fetch with offset simulation (get more and skip)
                all_chunk = self._repo.get_all(
                    namespace=namespace,
                    limit=offset + chunk_size,
                )
                if len(all_chunk) <= offset:
                    # No more memories to process
                    break
                chunk_memories = all_chunk[offset:offset + chunk_size]

            if len(chunk_memories) < 2:
                break

            # Build lookup structures for this chunk
            memories = [m for m, _ in chunk_memories]
            vectors_list = [v for _, v in chunk_memories]
            vectors_array = np.array(vectors_list, dtype=np.float32)
            memory_ids = [m.id for m in memories]
            contents = [m.content for m in memories]
            memory_dicts: list[dict[str, Any]] = [
                {
                    "id": m.id,
                    "content": m.content,
                    "created_at": m.created_at,
                    "last_accessed": m.last_accessed,
                    "access_count": m.access_count,
                    "importance": m.importance,
                    "tags": list(m.tags),
                }
                for m in memories
            ]

            # Find duplicate groups in this chunk
            group_indices = find_duplicate_groups(
                memory_ids=memory_ids,
                vectors=vectors_array,
                contents=contents,
                threshold=similarity_threshold,
                content_weight=self._config.consolidate_content_weight,
            )

            # Limit groups for this chunk
            group_indices = group_indices[:groups_remaining]

            if not group_indices:
                offset += len(chunk_memories)
                continue

            # Get strategy implementation
            strategy_impl = get_strategy(strategy)

            # Process groups in this chunk
            for member_indices in group_indices:
                group_member_dicts = [memory_dicts[i] for i in member_indices]
                group_member_ids = [str(d["id"]) for d in group_member_dicts]

                # Calculate average similarity
                total_sim = 0.0
                pair_count = 0
                for i_idx, i in enumerate(member_indices):
                    for j in member_indices[i_idx + 1:]:
                        v1, v2 = vectors_array[i], vectors_array[j]
                        dot = float(np.dot(v1, v2))
                        norm1 = float(np.linalg.norm(v1))
                        norm2 = float(np.linalg.norm(v2))
                        if norm1 > 1e-10 and norm2 > 1e-10:
                            v_sim = dot / (norm1 * norm2)
                        else:
                            v_sim = 0.0
                        c_sim = jaccard_similarity(contents[i], contents[j])
                        combined = combined_similarity(
                            v_sim, c_sim, self._config.consolidate_content_weight
                        )
                        total_sim += combined
                        pair_count += 1
                avg_similarity = total_sim / pair_count if pair_count > 0 else 0.0

                # Apply consolidation strategy
                try:
                    action_result: ConsolidationAction = strategy_impl.apply(
                        members=group_member_dicts,
                        member_ids=group_member_ids,
                        namespace=namespace,
                        repository=self._repo,
                        embeddings=self._embeddings,
                        dry_run=dry_run,
                    )
                    total_merged += action_result.memories_merged
                    total_deleted += action_result.memories_deleted
                except Exception as e:
                    logger.warning(f"Failed to consolidate group: {e}")
                    action_result = ConsolidationAction(
                        representative_id=group_member_ids[0],
                        deleted_ids=[],
                        action="failed",
                    )

                all_groups.append(
                    ConsolidationGroupResult(
                        representative_id=action_result.representative_id,
                        member_ids=group_member_ids,
                        avg_similarity=avg_similarity,
                        action_taken=action_result.action,
                    )
                )
                groups_remaining -= 1

            offset += len(chunk_memories)
            logger.debug(
                f"Processed chunk at offset {offset - len(chunk_memories)}, "
                f"found {len(group_indices)} groups"
            )

        return ConsolidateResult(
            groups_found=len(all_groups),
            memories_merged=total_merged,
            memories_deleted=total_deleted,
            groups=all_groups,
            dry_run=dry_run,
        )

    def _check_duplicate(
        self,
        content: str,
        namespace: str,
        threshold: float,
    ) -> bool:
        """Check if similar content already exists.

        Args:
            content: Content to check.
            namespace: Namespace to search.
            threshold: Similarity threshold.

        Returns:
            True if a similar memory exists.
        """
        try:
            # Generate embedding for content
            vector = self._embeddings.embed(content)

            # Search for similar memories
            results = self._repo.search(vector, limit=5, namespace=namespace)

            for result in results:
                # Check vector similarity
                if result.similarity >= threshold:
                    return True

                # Also check content overlap
                content_sim = jaccard_similarity(content, result.content)
                combined = combined_similarity(
                    result.similarity,
                    content_sim,
                    self._config.consolidate_content_weight,
                )
                if combined >= threshold:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return False

    def _store_extracted_memory(
        self,
        content: str,
        namespace: str,
        confidence: float,
        pattern_type: str,
    ) -> str:
        """Store an extracted memory.

        Args:
            content: Memory content.
            namespace: Target namespace.
            confidence: Extraction confidence.
            pattern_type: Type of pattern matched.

        Returns:
            The new memory's ID.
        """
        # Generate embedding
        vector = self._embeddings.embed(content)

        # Scale importance by confidence but keep lower than manual memories
        importance = self._config.extract_default_importance * confidence

        # Create memory
        memory = Memory(
            id="",  # Will be assigned
            content=content,
            namespace=namespace,
            tags=[f"extracted-{pattern_type}"],
            importance=importance,
            source=MemorySource.EXTRACTED,
            metadata={
                "extraction_confidence": confidence,
                "extraction_pattern": pattern_type,
            },
        )

        return self._repo.add(memory, vector)

    def _check_duplicate_with_vector(
        self,
        content: str,
        vector: np.ndarray,
        namespace: str,
        threshold: float,
    ) -> bool:
        """Check if similar content already exists using pre-computed vector.

        Args:
            content: Content to check.
            vector: Pre-computed embedding vector.
            namespace: Namespace to search.
            threshold: Similarity threshold.

        Returns:
            True if a similar memory exists.
        """
        try:
            # Search for similar memories using pre-computed vector
            results = self._repo.search(vector, limit=5, namespace=namespace)

            for result in results:
                # Check vector similarity
                if result.similarity >= threshold:
                    return True

                # Also check content overlap
                content_sim = jaccard_similarity(content, result.content)
                combined = combined_similarity(
                    result.similarity,
                    content_sim,
                    self._config.consolidate_content_weight,
                )
                if combined >= threshold:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return False

    def _store_extracted_memory_with_vector(
        self,
        content: str,
        vector: np.ndarray,
        namespace: str,
        confidence: float,
        pattern_type: str,
    ) -> str:
        """Store an extracted memory using pre-computed vector.

        Args:
            content: Memory content.
            vector: Pre-computed embedding vector.
            namespace: Target namespace.
            confidence: Extraction confidence.
            pattern_type: Type of pattern matched.

        Returns:
            The new memory's ID.
        """
        # Scale importance by confidence but keep lower than manual memories
        importance = self._config.extract_default_importance * confidence

        # Create memory
        memory = Memory(
            id="",  # Will be assigned
            content=content,
            namespace=namespace,
            tags=[f"extracted-{pattern_type}"],
            importance=importance,
            source=MemorySource.EXTRACTED,
            metadata={
                "extraction_confidence": confidence,
                "extraction_pattern": pattern_type,
            },
        )

        return self._repo.add(memory, vector)
