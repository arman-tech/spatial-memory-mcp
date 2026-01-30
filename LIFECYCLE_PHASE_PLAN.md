# Lifecycle Phase Implementation Plan

## Executive Summary

This plan details the implementation of 4 lifecycle management tools for the Spatial Memory MCP Server:

| Tool | Purpose | Complexity |
|------|---------|------------|
| **decay** | Apply time/access-based importance decay | Medium |
| **reinforce** | Boost memory importance on access | Medium |
| **extract** | Auto-extract memories from text | High |
| **consolidate** | Merge similar/duplicate memories | High |

These tools address the "Context Window Pollution" problem by providing utility-based memory management.

---

## Architecture Overview

### New Files to Create

```
spatial_memory/
  services/
    lifecycle.py           # LifecycleService with all 4 operations
  core/
    lifecycle_ops.py       # Core algorithms (decay formulas, similarity, extraction patterns)

tests/
  unit/
    test_lifecycle_ops.py      # Algorithm unit tests
    test_lifecycle_service.py  # Service unit tests
```

### Files to Modify

```
spatial_memory/
  server.py              # Add 4 new tools + handlers
  config.py              # Add lifecycle configuration
  core/errors.py         # Add DecayError, ConsolidationError, ExtractionError, ReinforcementError
  core/models.py         # Add result dataclasses and enums
```

### Service Architecture (Clean Architecture)

```python
class LifecycleService:
    """Service for memory lifecycle management."""

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        embeddings: EmbeddingServiceProtocol,
        config: LifecycleConfig | None = None,
    ) -> None:
        self._repo = repository
        self._embeddings = embeddings
        self._config = config or LifecycleConfig()
```

---

## 1. DECAY Tool

### 1.1 Purpose
Apply time and access-based decay to memory importance scores. Implements the "forgetting curve" - memories not accessed become less important over time.

### 1.2 Algorithm: Modified Half-Life Regression

Based on research (Ebbinghaus, Duolingo HLR, FSRS), we use a configurable decay formula:

```python
# Exponential decay with adaptive half-life
half_life = base_half_life * (1 + access_bonus)^access_count * importance_factor
decay_factor = 2^(-days_since_access / half_life)
decayed_importance = max(min_floor, base_importance * decay_factor)
```

**Key Features:**
- More accesses = slower decay (memory strengthens)
- Higher base importance = slower decay
- Configurable minimum floor (memories never fully forgotten)
- Three decay functions: exponential (default), linear, step

### 1.3 Tool Schema

```python
Tool(
    name="decay",
    description=(
        "Apply time and access-based decay to memory importance scores. "
        "Memories not accessed recently will have reduced importance."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace to decay (all if not specified)",
            },
            "decay_function": {
                "type": "string",
                "enum": ["exponential", "linear", "step"],
                "default": "exponential",
                "description": "Decay curve shape",
            },
            "half_life_days": {
                "type": "number",
                "minimum": 1,
                "maximum": 365,
                "default": 30,
                "description": "Days until importance halves (exponential)",
            },
            "min_importance": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.5,
                "default": 0.1,
                "description": "Minimum importance floor",
            },
            "access_weight": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.3,
                "description": "Weight of access count in decay calculation",
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "Preview changes without applying",
            },
        },
    },
)
```

### 1.4 Service Method

```python
@dataclass
class DecayedMemory:
    """A memory with calculated decay."""
    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    decay_factor: float
    days_since_access: int
    access_count: int

@dataclass
class DecayResult:
    """Result of decay operation."""
    memories_analyzed: int
    memories_decayed: int
    avg_decay_factor: float
    decayed_memories: list[DecayedMemory]
    dry_run: bool

def decay(
    self,
    namespace: str | None = None,
    decay_function: Literal["exponential", "linear", "step"] = "exponential",
    half_life_days: float = 30.0,
    min_importance: float = 0.1,
    access_weight: float = 0.3,
    dry_run: bool = True,
) -> DecayResult:
```

### 1.5 Core Algorithm

```python
def calculate_decay_factor(
    days_since_access: float,
    access_count: int,
    base_importance: float,
    decay_function: str,
    half_life_days: float,
    access_weight: float,
) -> float:
    """Calculate decay factor for a memory."""
    # Adaptive half-life: more accesses = longer half-life
    access_bonus = 0.5  # Each access adds 50% to half-life
    access_factor = (1 + access_bonus) ** min(access_count, 20)  # Cap at 20

    # Higher importance also slows decay
    importance_factor = 1 + base_importance

    effective_half_life = half_life_days * access_factor * importance_factor

    if decay_function == "exponential":
        time_decay = 2.0 ** (-days_since_access / effective_half_life)
    elif decay_function == "linear":
        time_decay = max(0.0, 1.0 - days_since_access / (2 * effective_half_life))
    elif decay_function == "step":
        if days_since_access < effective_half_life:
            time_decay = 1.0
        elif days_since_access < 2 * effective_half_life:
            time_decay = 0.5
        else:
            time_decay = 0.25

    # Blend time decay with access-based stability
    access_stability = min(1.0, math.log1p(access_count) / math.log(100))

    return (1 - access_weight) * time_decay + access_weight * access_stability
```

### 1.6 Configuration

```python
# Decay Settings
decay_default_half_life_days: float = Field(
    default=30.0,
    ge=1.0,
    le=365.0,
    description="Default half-life for exponential decay",
)
decay_default_function: str = Field(
    default="exponential",
    description="Default decay function",
)
decay_min_importance_floor: float = Field(
    default=0.1,
    ge=0.0,
    le=0.5,
    description="Minimum importance after decay",
)
decay_batch_size: int = Field(
    default=500,
    ge=100,
    description="Batch size for decay updates",
)
```

### 1.7 Test Cases

```python
# Unit Tests
- test_exponential_decay_halves_at_half_life
- test_linear_decay_reaches_zero_at_double_half_life
- test_step_decay_transitions_at_thresholds
- test_access_count_slows_decay
- test_high_importance_slows_decay
- test_min_floor_enforced
- test_dry_run_no_changes

# Integration Tests
- test_decay_updates_database
- test_decay_respects_namespace_filter
- test_decay_batch_processing
```

---

## 2. REINFORCE Tool

### 2.1 Purpose
Boost memory importance based on access or explicit feedback. Counterbalances decay - important memories get reinforced.

### 2.2 Algorithm: Access-Based Reinforcement with Caps

```python
# Reinforcement with diminishing returns
boost = base_boost * (diminishing_factor ** (boosts_today / 0.1))
boost = min(boost, daily_cap - boosts_today)
new_importance = min(max_importance, current_importance + boost)
```

**Key Features:**
- Configurable boost types: additive, multiplicative, set_value
- Daily caps to prevent runaway inflation
- Diminishing returns for repeated reinforcement
- Optional access timestamp update

### 2.3 Tool Schema

```python
Tool(
    name="reinforce",
    description=(
        "Boost memory importance based on usage or explicit feedback. "
        "Reinforcement increases importance and can reset decay timer."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Memory IDs to reinforce",
            },
            "boost_type": {
                "type": "string",
                "enum": ["additive", "multiplicative", "set_value"],
                "default": "additive",
            },
            "boost_amount": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.1,
            },
            "update_access": {
                "type": "boolean",
                "default": True,
                "description": "Update last_accessed timestamp",
            },
        },
        "required": ["memory_ids"],
    },
)
```

### 2.4 Service Method

```python
@dataclass
class ReinforcedMemory:
    """A memory that was reinforced."""
    id: str
    content_preview: str
    old_importance: float
    new_importance: float
    boost_applied: float

@dataclass
class ReinforceResult:
    """Result of reinforcement operation."""
    memories_reinforced: int
    avg_boost: float
    reinforced: list[ReinforcedMemory]
    not_found: list[str]

def reinforce(
    self,
    memory_ids: list[str],
    boost_type: Literal["additive", "multiplicative", "set_value"] = "additive",
    boost_amount: float = 0.1,
    update_access: bool = True,
) -> ReinforceResult:
```

### 2.5 Core Algorithm

```python
def calculate_reinforcement(
    current_importance: float,
    boost_type: str,
    boost_amount: float,
    max_importance: float = 1.0,
) -> tuple[float, float]:
    """Calculate new importance after reinforcement.

    Returns (new_importance, actual_boost_applied).
    """
    if boost_type == "additive":
        new_importance = current_importance + boost_amount
    elif boost_type == "multiplicative":
        new_importance = current_importance * (1.0 + boost_amount)
    elif boost_type == "set_value":
        new_importance = boost_amount
    else:
        new_importance = current_importance

    # Cap at maximum
    new_importance = min(max_importance, new_importance)
    actual_boost = new_importance - current_importance

    return new_importance, actual_boost
```

### 2.6 Configuration

```python
# Reinforcement Settings
reinforce_default_boost: float = Field(
    default=0.1,
    ge=0.01,
    le=0.5,
    description="Default boost amount",
)
reinforce_max_importance: float = Field(
    default=1.0,
    ge=0.5,
    le=1.0,
    description="Maximum importance after reinforcement",
)
reinforce_on_access: bool = Field(
    default=True,
    description="Auto-reinforce when memory is accessed via recall",
)
reinforce_access_boost: float = Field(
    default=0.02,
    ge=0.0,
    le=0.1,
    description="Automatic boost on access",
)
```

### 2.7 Test Cases

```python
# Unit Tests
- test_additive_boost_calculation
- test_multiplicative_boost_calculation
- test_set_value_override
- test_max_importance_cap
- test_update_access_flag

# Integration Tests
- test_reinforce_persists_to_database
- test_reinforce_multiple_memories
- test_not_found_handling
```

---

## 3. EXTRACT Tool

### 3.1 Purpose
Automatically extract memory-worthy content from conversation text using pattern matching and heuristics.

### 3.2 Algorithm: Tiered Pattern Extraction

**Tier 1 (Fast):** Regex patterns for common knowledge structures
**Tier 2 (Standard):** TF-IDF keyword scoring + pattern matching
**Tier 3 (Thorough):** LLM verification (optional, future)

### 3.3 Default Extraction Patterns

```python
EXTRACTION_PATTERNS = [
    # Decisions
    (r"(?:decided|chose|going with|selected|will use)\s+(.+?)(?:\.|$)", 0.8, "decision"),

    # Facts/Definitions
    (r"(.+?)\s+(?:is|are|means|refers to)\s+(.+?)(?:\.|$)", 0.6, "definition"),

    # Important points
    (r"(?:important|note|remember|key point):\s*(.+?)(?:\.|$)", 0.9, "important"),

    # Solutions/Fixes
    (r"(?:the (?:fix|solution|approach) (?:is|was))\s+(.+?)(?:\.|$)", 0.85, "solution"),

    # Error diagnoses
    (r"(?:the (?:issue|problem|bug) was)\s+(.+?)(?:\.|$)", 0.8, "error"),

    # Explicit save requests
    (r"(?:save|remember|note|store)(?:\s+that)?\s+(.+?)(?:\.|$)", 0.95, "explicit"),

    # Patterns/Learnings
    (r"(?:the trick is|the key is|pattern:)\s+(.+?)(?:\.|$)", 0.85, "pattern"),
]
```

### 3.4 Tool Schema

```python
Tool(
    name="extract",
    description=(
        "Automatically extract memories from conversation text. "
        "Uses pattern matching to identify facts, decisions, and key information."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract memories from",
            },
            "namespace": {
                "type": "string",
                "default": "extracted",
                "description": "Namespace for extracted memories",
            },
            "min_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
                "description": "Minimum confidence to extract",
            },
            "deduplicate": {
                "type": "boolean",
                "default": True,
                "description": "Skip if similar memory exists",
            },
            "dedup_threshold": {
                "type": "number",
                "minimum": 0.7,
                "maximum": 0.99,
                "default": 0.9,
                "description": "Similarity threshold for deduplication",
            },
        },
        "required": ["text"],
    },
)
```

### 3.5 Service Method

```python
@dataclass
class ExtractedMemory:
    """A memory candidate extracted from text."""
    content: str
    confidence: float
    pattern_matched: str
    start_pos: int
    end_pos: int
    stored: bool  # False if deduplicated
    memory_id: str | None  # Set if stored

@dataclass
class ExtractResult:
    """Result of memory extraction."""
    candidates_found: int
    memories_created: int
    deduplicated_count: int
    extractions: list[ExtractedMemory]

def extract(
    self,
    text: str,
    namespace: str = "extracted",
    min_confidence: float = 0.5,
    deduplicate: bool = True,
    dedup_threshold: float = 0.9,
) -> ExtractResult:
```

### 3.6 Core Algorithm

```python
def extract_candidates(
    text: str,
    patterns: list[tuple[str, float, str]],
    min_confidence: float,
) -> list[dict]:
    """Extract memory candidates using pattern matching."""
    candidates = []

    for pattern, base_confidence, pattern_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            content = match.group(1).strip() if match.groups() else match.group(0).strip()

            # Skip too short/long
            if len(content) < 10 or len(content) > 5000:
                continue

            # Adjust confidence based on content quality
            confidence = score_extraction_confidence(content, base_confidence)

            if confidence >= min_confidence:
                candidates.append({
                    "content": content,
                    "confidence": confidence,
                    "pattern_type": pattern_type,
                    "start": match.start(),
                    "end": match.end(),
                })

    # Deduplicate overlapping extractions (keep highest confidence)
    return dedupe_overlapping(candidates)


def score_extraction_confidence(content: str, base: float) -> float:
    """Adjust confidence based on content quality."""
    score = base

    # Longer content = more informative
    word_count = len(content.split())
    if word_count >= 10:
        score += 0.1
    elif word_count < 5:
        score -= 0.1

    # Technical terms boost
    tech_terms = {"api", "database", "function", "class", "config", "error"}
    if any(term in content.lower() for term in tech_terms):
        score += 0.05

    # Code presence boost
    if "```" in content or re.search(r"^\s{4,}", content, re.MULTILINE):
        score += 0.1

    return min(1.0, max(0.0, score))
```

### 3.7 Configuration

```python
# Extraction Settings
extract_max_text_length: int = Field(
    default=50000,
    ge=1000,
    description="Maximum text length for extraction",
)
extract_max_candidates: int = Field(
    default=20,
    ge=1,
    description="Maximum candidates per extraction",
)
extract_default_importance: float = Field(
    default=0.4,
    ge=0.0,
    le=1.0,
    description="Default importance for extracted memories",
)
extract_default_namespace: str = Field(
    default="extracted",
    description="Default namespace for extracted memories",
)
```

### 3.8 Test Cases

```python
# Unit Tests
- test_decision_pattern_extraction
- test_solution_pattern_extraction
- test_explicit_save_pattern
- test_confidence_scoring
- test_overlapping_deduplication
- test_min_confidence_filtering

# Integration Tests
- test_extract_stores_memories
- test_deduplication_against_existing
- test_extracted_memories_searchable
```

---

## 4. CONSOLIDATE Tool

### 4.1 Purpose
Merge similar or duplicate memories to reduce redundancy. Prevents context pollution from near-duplicate information.

### 4.2 Algorithm: Two-Pass Similarity Detection

**Pass 1:** Vector cosine similarity (fast filter)
**Pass 2:** Content overlap (Jaccard similarity)

```python
# Combined similarity score
combined = (1 - content_weight) * vector_similarity + content_weight * content_overlap

# Decision thresholds
if combined >= 0.90 and content_overlap >= 0.50:
    decision = MERGE
elif combined >= 0.80:
    decision = REVIEW
elif combined >= 0.75:
    decision = LINK
else:
    decision = KEEP_SEPARATE
```

### 4.3 Consolidation Strategies

| Strategy | Description |
|----------|-------------|
| `keep_newest` | Keep the most recent memory |
| `keep_oldest` | Keep the oldest (canonical) memory |
| `keep_highest_importance` | Keep highest importance |
| `merge_content` | Combine content, re-embed |

### 4.4 Tool Schema

```python
Tool(
    name="consolidate",
    description=(
        "Merge similar or duplicate memories to reduce redundancy. "
        "Finds memories above similarity threshold and merges them."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace to consolidate (required)",
            },
            "similarity_threshold": {
                "type": "number",
                "minimum": 0.7,
                "maximum": 0.99,
                "default": 0.85,
                "description": "Minimum similarity for duplicates",
            },
            "strategy": {
                "type": "string",
                "enum": ["keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"],
                "default": "keep_highest_importance",
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "Preview without changes",
            },
            "max_groups": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
            },
        },
        "required": ["namespace"],
    },
)
```

### 4.5 Service Method

```python
@dataclass
class ConsolidationGroup:
    """A group of similar memories."""
    representative_id: str
    member_ids: list[str]
    avg_similarity: float
    action_taken: str  # "merged", "deleted", "preview"

@dataclass
class ConsolidateResult:
    """Result of consolidation."""
    groups_found: int
    memories_merged: int
    memories_deleted: int
    groups: list[ConsolidationGroup]
    dry_run: bool

def consolidate(
    self,
    namespace: str,
    similarity_threshold: float = 0.85,
    strategy: Literal["keep_newest", "keep_oldest", "keep_highest_importance", "merge_content"] = "keep_highest_importance",
    dry_run: bool = True,
    max_groups: int = 50,
) -> ConsolidateResult:
```

### 4.6 Core Algorithm

```python
def find_duplicate_groups(
    memory_ids: list[str],
    vectors: np.ndarray,
    contents: list[str],
    threshold: float,
    content_weight: float = 0.3,
) -> list[list[int]]:
    """Find groups of duplicate memories using Union-Find."""
    n = len(memory_ids)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Calculate pairwise similarities
    vector_sim = cosine_similarity(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            # Vector similarity
            v_sim = vector_sim[i, j]

            # Content overlap (Jaccard)
            c_sim = jaccard_similarity(contents[i], contents[j])

            # Combined score
            combined = (1 - content_weight) * v_sim + content_weight * c_sim

            if combined >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Return only groups with duplicates
    return [g for g in groups.values() if len(g) > 1]


def merge_memories(
    memories: list[dict],
    vectors: list[np.ndarray],
    strategy: str,
) -> tuple[dict, np.ndarray]:
    """Merge multiple memories into one."""
    if strategy == "keep_newest":
        primary = max(memories, key=lambda m: m["created_at"])
    elif strategy == "keep_oldest":
        primary = min(memories, key=lambda m: m["created_at"])
    elif strategy == "keep_highest_importance":
        primary = max(memories, key=lambda m: m["importance"])
    elif strategy == "merge_content":
        # Combine unique content
        contents = [m["content"] for m in memories]
        merged_content = "\n\n---\n\n".join(contents)
        primary = memories[0].copy()
        primary["content"] = merged_content

    # Merge metadata
    merged = {
        "content": primary["content"],
        "created_at": min(m["created_at"] for m in memories),
        "last_accessed": max(m["last_accessed"] for m in memories),
        "access_count": sum(m["access_count"] for m in memories),
        "importance": max(m["importance"] for m in memories),
        "tags": list(set(tag for m in memories for tag in m.get("tags", []))),
        "source": "consolidated",
        "metadata": {
            "consolidated_from": [m["id"] for m in memories],
        },
    }

    # Weighted average vector (by content length)
    total_len = sum(len(m["content"]) for m in memories)
    merged_vector = sum(
        (len(m["content"]) / total_len) * v
        for m, v in zip(memories, vectors)
    )
    merged_vector = merged_vector / np.linalg.norm(merged_vector)

    return merged, merged_vector
```

### 4.7 Configuration

```python
# Consolidation Settings
consolidate_min_threshold: float = Field(
    default=0.7,
    ge=0.5,
    le=0.99,
    description="Minimum similarity threshold",
)
consolidate_content_weight: float = Field(
    default=0.3,
    ge=0.0,
    le=1.0,
    description="Weight of content overlap vs vector similarity",
)
consolidate_max_batch: int = Field(
    default=1000,
    ge=100,
    description="Maximum memories per consolidation pass",
)
consolidate_require_dry_run_first: bool = Field(
    default=True,
    description="Require dry_run before actual consolidation",
)
```

### 4.8 Test Cases

```python
# Unit Tests
- test_find_duplicate_groups_basic
- test_union_find_correctness
- test_strategy_keep_newest
- test_strategy_keep_oldest
- test_strategy_keep_highest_importance
- test_strategy_merge_content
- test_metadata_merging

# Integration Tests
- test_consolidate_dry_run
- test_consolidate_actual_merge
- test_consolidated_memory_searchable
- test_source_marked_consolidated
```

---

## 5. Error Types

```python
# spatial_memory/core/errors.py additions

class DecayError(SpatialMemoryError):
    """Raised when decay calculation or application fails."""
    pass

class ReinforcementError(SpatialMemoryError):
    """Raised when reinforcement fails."""
    pass

class ExtractionError(SpatialMemoryError):
    """Raised when memory extraction fails."""
    pass

class ConsolidationError(SpatialMemoryError):
    """Raised when consolidation fails."""
    pass
```

---

## 6. Data Models

```python
# spatial_memory/core/models.py additions

class DecayFunction(str, Enum):
    """Decay function types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"

class BoostType(str, Enum):
    """Reinforcement boost types."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    SET_VALUE = "set_value"

class ConsolidationStrategy(str, Enum):
    """Consolidation strategies."""
    KEEP_NEWEST = "keep_newest"
    KEEP_OLDEST = "keep_oldest"
    KEEP_HIGHEST_IMPORTANCE = "keep_highest_importance"
    MERGE_CONTENT = "merge_content"

class ExtractionPattern(str, Enum):
    """Types of extracted content."""
    DECISION = "decision"
    DEFINITION = "definition"
    SOLUTION = "solution"
    ERROR = "error"
    PATTERN = "pattern"
    EXPLICIT = "explicit"
    IMPORTANT = "important"
```

---

## 7. Implementation Order

### Phase 1: Foundation
1. Create `LifecycleService` skeleton
2. Add error types to `errors.py`
3. Add enums to `models.py`
4. Add config options to `config.py`

### Phase 2: Decay (Simplest)
1. Implement decay algorithm in `lifecycle_ops.py`
2. Implement `decay()` in `LifecycleService`
3. Add tool to server
4. Write tests

### Phase 3: Reinforce (Pairs with Decay)
1. Implement reinforcement algorithm
2. Implement `reinforce()` in `LifecycleService`
3. Add tool to server
4. Write tests

### Phase 4: Extract (Independent)
1. Implement pattern matching in `lifecycle_ops.py`
2. Implement `extract()` in `LifecycleService`
3. Add tool to server
4. Write tests

### Phase 5: Consolidate (Most Complex)
1. Implement similarity detection
2. Implement merge strategies
3. Implement `consolidate()` in `LifecycleService`
4. Add tool to server
5. Write tests

---

## 8. Feature Interaction Matrix

```
             decay     reinforce   extract    consolidate
            +--------+-----------+----------+-------------+
decay       |   -    | Counter-  | Extracted| Low imp     |
            |        | balance   | decay too| = merge?    |
            +--------+-----------+----------+-------------+
reinforce   | Slows  |     -     | Initial  | Boost       |
            | decay  |           | importance| sources    |
            +--------+-----------+----------+-------------+
extract     | Start  | Low init  |     -    | Dedup check |
            | fresh  | importance|          | before store|
            +--------+-----------+----------+-------------+
consolidate | Use    | Boost     | Creates  |      -      |
            | base   | sources   | targets  |             |
            +--------+-----------+----------+-------------+
```

### Interaction Rules
1. **Decay + Reinforce:** Must be balanced (equilibrium model)
2. **Extract + Consolidate:** Dedup before storing extracted memories
3. **Consolidate + Reinforce:** Boost source memories before archiving
4. **Extract + Decay:** Extracted memories start with lower importance (0.4 vs 0.5)

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Consolidation deletes data | High | Require dry_run first; log all deletions |
| Decay makes memories unfindable | Medium | Enforce min_importance floor; decay affects ranking only |
| Extract creates too many | Medium | Deduplication; confidence threshold |
| Performance with large datasets | Medium | Batch processing; efficient similarity |

---

## 10. Verification Plan

```bash
# After implementation:
pytest tests/unit/test_lifecycle_ops.py -v
pytest tests/unit/test_lifecycle_service.py -v
pytest tests/integration/test_lifecycle_tools.py -v

# Type checking
mypy spatial_memory/services/lifecycle.py
mypy spatial_memory/core/lifecycle_ops.py

# Linting
ruff check spatial_memory/services/lifecycle.py
ruff check spatial_memory/core/lifecycle_ops.py
```

---

## References

- Ebbinghaus Forgetting Curve (1885)
- Duolingo Half-Life Regression (2016)
- FSRS Algorithm v4.5
- TF-IDF Keyword Extraction
- Union-Find for Clustering
