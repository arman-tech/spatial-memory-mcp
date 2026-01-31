# Spatial Memory MCP Server - API Reference

Complete reference documentation for all 21 MCP tools provided by the Spatial Memory server.

## Table of Contents

- [Overview](#overview)
  - [Tool Categories](#tool-categories)
  - [Common Response Format](#common-response-format)
  - [Error Handling](#error-handling)
- [Core Operations](#core-operations)
  - [remember](#remember)
  - [remember_batch](#remember_batch)
  - [recall](#recall)
  - [nearby](#nearby)
  - [forget](#forget)
  - [forget_batch](#forget_batch)
- [Spatial Operations](#spatial-operations)
  - [journey](#journey)
  - [wander](#wander)
  - [regions](#regions)
  - [visualize](#visualize)
- [Lifecycle Operations](#lifecycle-operations)
  - [decay](#decay)
  - [reinforce](#reinforce)
  - [extract](#extract)
  - [consolidate](#consolidate)
- [Utility Operations](#utility-operations)
  - [stats](#stats)
  - [namespaces](#namespaces)
  - [delete_namespace](#delete_namespace)
  - [rename_namespace](#rename_namespace)
  - [export_memories](#export_memories)
  - [import_memories](#import_memories)
  - [hybrid_recall](#hybrid_recall)
- [Error Reference](#error-reference)

---

## Overview

### Tool Categories

The 21 tools are organized into four categories:

| Category | Tools | Purpose |
|----------|-------|---------|
| **Core** | 6 | Basic CRUD operations for memories |
| **Spatial** | 4 | Navigation and exploration of memory space |
| **Lifecycle** | 4 | Memory maintenance and evolution |
| **Utility** | 7 | Administration, export/import, and hybrid search |

### Common Response Format

All tool responses are JSON objects. Successful responses contain the operation results directly:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "content": "Use repository pattern for data access",
  "namespace": "architecture"
}
```

Error responses include an `isError` flag:

```json
{
  "error": "ValidationError",
  "message": "Content cannot be empty",
  "isError": true
}
```

### Error Handling

When an operation fails, the response includes:

| Field | Type | Description |
|-------|------|-------------|
| `error` | string | Error type name |
| `message` | string | Human-readable error description |
| `isError` | boolean | Always `true` for errors |

For internal errors, a reference ID is provided instead of stack traces for security.

---

## Core Operations

### remember

Store a new memory in the spatial memory system.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The text content to remember (max 100,000 characters) |
| `namespace` | string | No | `"default"` | Namespace for organizing memories |
| `tags` | string[] | No | `[]` | Optional tags for categorization |
| `importance` | number | No | `0.5` | Importance score from 0.0 to 1.0 |
| `metadata` | object | No | `{}` | Optional metadata to attach to the memory |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Generated UUID for the memory |
| `content` | string | The stored content |
| `namespace` | string | Storage namespace |

#### Example

```json
// Request
{
  "content": "Use repository pattern for data access",
  "namespace": "architecture",
  "tags": ["patterns", "design"],
  "importance": 0.8,
  "metadata": {"source": "code-review"}
}

// Response
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "content": "Use repository pattern for data access",
  "namespace": "architecture"
}
```

#### Errors

- `ValidationError` - Content is empty or exceeds 100,000 characters
- `ValidationError` - Importance is outside 0.0-1.0 range
- `EmbeddingError` - Failed to generate embedding for content
- `StorageError` - Database write failed

---

### remember_batch

Store multiple memories efficiently in a single operation.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memories` | object[] | Yes | - | Array of memories to store |

Each memory object in the array accepts:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | - | Text content to remember |
| `namespace` | string | No | `"default"` | Namespace for organization |
| `tags` | string[] | No | `[]` | Optional tags |
| `importance` | number | No | `0.5` | Importance score (0.0-1.0) |
| `metadata` | object | No | `{}` | Optional metadata |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `ids` | string[] | Generated UUIDs for each memory |
| `count` | number | Number of memories stored |

#### Example

```json
// Request
{
  "memories": [
    {
      "content": "React uses virtual DOM for efficient updates",
      "namespace": "frontend",
      "tags": ["react", "performance"]
    },
    {
      "content": "TypeScript adds static typing to JavaScript",
      "namespace": "frontend",
      "importance": 0.7
    }
  ]
}

// Response
{
  "ids": [
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "b2c3d4e5-f6a7-8901-bcde-f23456789012"
  ],
  "count": 2
}
```

#### Errors

- `ValidationError` - Memory list is empty
- `ValidationError` - Any memory has empty or invalid content
- `EmbeddingError` - Failed to generate embeddings
- `StorageError` - Database write failed

---

### recall

Search for similar memories using semantic similarity.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The search query text |
| `limit` | integer | No | `5` | Maximum number of results (1-100) |
| `namespace` | string | No | - | Filter to specific namespace |
| `min_similarity` | number | No | `0.0` | Minimum similarity threshold (0.0-1.0) |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `memories` | object[] | Array of matching memories |
| `total` | number | Number of results returned |

Each memory object contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Memory UUID |
| `content` | string | Memory content |
| `similarity` | number | Similarity score (0.0-1.0) |
| `namespace` | string | Memory namespace |
| `tags` | string[] | Memory tags |
| `importance` | number | Importance score |
| `created_at` | string | ISO 8601 timestamp |
| `metadata` | object | Memory metadata |

#### Example

```json
// Request
{
  "query": "database design patterns",
  "limit": 3,
  "namespace": "architecture",
  "min_similarity": 0.5
}

// Response
{
  "memories": [
    {
      "id": "a1b2c3d4-...",
      "content": "Use repository pattern for data access",
      "similarity": 0.82,
      "namespace": "architecture",
      "tags": ["patterns", "design"],
      "importance": 0.8,
      "created_at": "2026-01-31T12:00:00Z",
      "metadata": {}
    }
  ],
  "total": 1
}
```

#### Errors

- `ValidationError` - Query is empty
- `ValidationError` - Limit is less than 1 or greater than 100
- `EmbeddingError` - Failed to generate query embedding

---

### nearby

Find memories similar to a specific memory.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | - | The UUID of the reference memory |
| `limit` | integer | No | `5` | Maximum number of neighbors (1-100) |
| `namespace` | string | No | - | Filter neighbors to specific namespace |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `reference` | object | The reference memory |
| `neighbors` | object[] | Array of neighboring memories |

Reference object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Memory UUID |
| `content` | string | Memory content |
| `namespace` | string | Memory namespace |

Neighbor objects include `similarity` score in addition to standard memory fields.

#### Example

```json
// Request
{
  "memory_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "limit": 3
}

// Response
{
  "reference": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "content": "Use repository pattern for data access",
    "namespace": "architecture"
  },
  "neighbors": [
    {
      "id": "b2c3d4e5-...",
      "content": "Repository abstracts data layer from business logic",
      "similarity": 0.91,
      "namespace": "architecture"
    }
  ]
}
```

#### Errors

- `MemoryNotFoundError` - Memory with given ID does not exist

---

### forget

Delete a memory by its ID.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | - | The UUID of the memory to delete |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `deleted` | number | Number of memories deleted (0 or 1) |
| `ids` | string[] | IDs of deleted memories |

#### Example

```json
// Request
{
  "memory_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}

// Response
{
  "deleted": 1,
  "ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890"]
}
```

#### Errors

- None - Returns `deleted: 0` if memory not found

---

### forget_batch

Delete multiple memories by their IDs.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_ids` | string[] | Yes | - | Array of memory UUIDs to delete |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `deleted` | number | Number of memories deleted |
| `ids` | string[] | IDs of deleted memories |

#### Example

```json
// Request
{
  "memory_ids": [
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "b2c3d4e5-f6a7-8901-bcde-f23456789012"
  ]
}

// Response
{
  "deleted": 2,
  "ids": [
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "b2c3d4e5-f6a7-8901-bcde-f23456789012"
  ]
}
```

#### Errors

- `ValidationError` - Memory ID list is empty

---

## Spatial Operations

### journey

Navigate semantic space between two memories using spherical interpolation (SLERP). Discovers memories along the conceptual path between start and end points.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_id` | string | Yes | - | Starting memory UUID |
| `end_id` | string | Yes | - | Ending memory UUID |
| `steps` | integer | No | `10` | Number of interpolation steps (2-20) |
| `namespace` | string | No | - | Optional namespace filter for nearby search |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `start_id` | string | Starting memory UUID |
| `end_id` | string | Ending memory UUID |
| `steps` | object[] | Journey steps with discovered memories |
| `path_coverage` | number | Fraction of path with nearby memories (0.0-1.0) |

Each step object:

| Field | Type | Description |
|-------|------|-------------|
| `step` | number | Step number |
| `t` | number | Interpolation parameter (0.0-1.0) |
| `position` | number[] | Interpolated vector position |
| `nearby_memories` | object[] | Memories near this path position |
| `distance_to_path` | number | Distance from nearest memory to ideal path |

#### Example

```json
// Request
{
  "start_id": "a1b2c3d4-...",
  "end_id": "x9y8z7w6-...",
  "steps": 5
}

// Response
{
  "start_id": "a1b2c3d4-...",
  "end_id": "x9y8z7w6-...",
  "steps": [
    {
      "step": 0,
      "t": 0.0,
      "position": [0.12, 0.34, ...],
      "nearby_memories": [
        {
          "id": "a1b2c3d4-...",
          "content": "React components...",
          "similarity": 1.0
        }
      ],
      "distance_to_path": 0.0
    },
    {
      "step": 2,
      "t": 0.5,
      "position": [0.23, 0.45, ...],
      "nearby_memories": [
        {
          "id": "m3n4o5p6-...",
          "content": "State management connects UI to data...",
          "similarity": 0.78
        }
      ],
      "distance_to_path": 0.15
    }
  ],
  "path_coverage": 0.8
}
```

#### Errors

- `MemoryNotFoundError` - Start or end memory does not exist
- `JourneyError` - Path cannot be computed

---

### wander

Explore memory space through random walk. Uses temperature-based selection to balance exploration and exploitation.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_id` | string | No | random | Starting memory UUID (random if not provided) |
| `steps` | integer | No | `10` | Number of exploration steps (1-20) |
| `temperature` | number | No | `0.5` | Randomness: 0.0=focused, 1.0=very random |
| `namespace` | string | No | - | Optional namespace filter |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `start_id` | string | Starting memory UUID |
| `steps` | object[] | Walk steps |
| `total_distance` | number | Total distance traveled in embedding space |

Each step object:

| Field | Type | Description |
|-------|------|-------------|
| `step` | number | Step number |
| `memory` | object | Memory at this step |
| `similarity_to_previous` | number | Similarity to previous step's memory |
| `selection_probability` | number | Probability this memory was selected |

#### Example

```json
// Request
{
  "start_id": "a1b2c3d4-...",
  "steps": 3,
  "temperature": 0.3
}

// Response
{
  "start_id": "a1b2c3d4-...",
  "steps": [
    {
      "step": 0,
      "memory": {
        "id": "a1b2c3d4-...",
        "content": "React uses virtual DOM...",
        "namespace": "frontend",
        "tags": ["react"],
        "similarity": 1.0
      },
      "similarity_to_previous": 1.0,
      "selection_probability": 1.0
    },
    {
      "step": 1,
      "memory": {
        "id": "c3d4e5f6-...",
        "content": "Vue also uses virtual DOM...",
        "namespace": "frontend",
        "tags": ["vue"],
        "similarity": 0.85
      },
      "similarity_to_previous": 0.85,
      "selection_probability": 0.42
    }
  ],
  "total_distance": 0.32
}
```

#### Errors

- `ValidationError` - No memories available for wander
- `MemoryNotFoundError` - Start memory does not exist
- `WanderError` - Walk cannot continue

---

### regions

Discover semantic clusters in memory space using HDBSCAN clustering. Returns cluster info with representative memories and keywords.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | No | - | Optional namespace filter |
| `min_cluster_size` | integer | No | `3` | Minimum memories per cluster (2-50) |
| `max_clusters` | integer | No | - | Maximum clusters to return |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `clusters` | object[] | Discovered clusters |
| `total_memories` | number | Total memories analyzed |
| `noise_count` | number | Memories not in any cluster |
| `clustering_quality` | number | Overall quality (silhouette score, -1.0 to 1.0) |

Each cluster object:

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | number | Cluster identifier |
| `size` | number | Number of memories in cluster |
| `keywords` | string[] | Keywords describing the cluster |
| `representative_memory` | object | Memory closest to cluster centroid |
| `sample_memories` | object[] | Sample memories from the cluster |
| `coherence` | number | Internal cluster tightness (0.0-1.0) |

#### Example

```json
// Request
{
  "namespace": "architecture",
  "min_cluster_size": 3
}

// Response
{
  "clusters": [
    {
      "cluster_id": 0,
      "size": 5,
      "keywords": ["database", "repository", "data access"],
      "representative_memory": {
        "id": "a1b2c3d4-...",
        "content": "Use repository pattern for data access"
      },
      "sample_memories": [
        {
          "id": "b2c3d4e5-...",
          "content": "Repository abstracts data layer...",
          "similarity": 0.91
        }
      ],
      "coherence": 0.85
    }
  ],
  "total_memories": 20,
  "noise_count": 3,
  "clustering_quality": 0.72
}
```

#### Errors

- `ClusteringError` - Too few memories for clustering
- `InsufficientMemoriesError` - Not enough memories available

---

### visualize

Project memories to 2D/3D for visualization using UMAP. Returns coordinates and optional similarity edges.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_ids` | string[] | No | - | Specific memory UUIDs to visualize |
| `namespace` | string | No | - | Namespace filter (if memory_ids not specified) |
| `format` | string | No | `"json"` | Output format: `json`, `mermaid`, or `svg` |
| `dimensions` | integer | No | `2` | Projection dimensionality: `2` or `3` |
| `include_edges` | boolean | No | `true` | Include similarity edges |

#### Returns

For `json` format:

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | object[] | Visualization nodes |
| `edges` | object[] | Connections between nodes |
| `bounds` | object | Coordinate bounds (`x_min`, `x_max`, `y_min`, `y_max`) |
| `format` | string | Output format used |

For `mermaid` or `svg` format:

| Field | Type | Description |
|-------|------|-------------|
| `format` | string | Output format |
| `output` | string | Formatted output string |
| `node_count` | number | Number of nodes visualized |

Node object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Memory UUID |
| `x` | number | X coordinate |
| `y` | number | Y coordinate |
| `label` | string | Display label (truncated content) |
| `cluster` | number | Cluster ID (-1 for unclustered) |
| `importance` | number | Memory importance |

Edge object:

| Field | Type | Description |
|-------|------|-------------|
| `from_id` | string | Source node ID |
| `to_id` | string | Target node ID |
| `weight` | number | Similarity weight (0.0-1.0) |

#### Example

```json
// Request
{
  "namespace": "architecture",
  "format": "json",
  "dimensions": 2
}

// Response
{
  "nodes": [
    {
      "id": "a1b2c3d4-...",
      "x": 0.52,
      "y": -0.31,
      "label": "Use repository pattern...",
      "cluster": 0,
      "importance": 0.8
    }
  ],
  "edges": [
    {
      "from_id": "a1b2c3d4-...",
      "to_id": "b2c3d4e5-...",
      "weight": 0.91
    }
  ],
  "bounds": {
    "x_min": -1.0,
    "x_max": 1.0,
    "y_min": -1.0,
    "y_max": 1.0
  },
  "format": "json"
}
```

#### Errors

- `VisualizationError` - Failed to generate visualization
- `InsufficientMemoriesError` - Not enough memories to visualize

---

## Lifecycle Operations

### decay

Apply time and access-based decay to memory importance scores. Memories not accessed recently will have reduced importance.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | No | - | Namespace to decay (all if not specified) |
| `decay_function` | string | No | `"exponential"` | Decay curve: `exponential`, `linear`, or `step` |
| `half_life_days` | number | No | `30` | Days until importance halves (1-365) |
| `min_importance` | number | No | `0.1` | Minimum importance floor (0.0-0.5) |
| `access_weight` | number | No | `0.3` | Weight of access count in decay (0.0-1.0) |
| `dry_run` | boolean | No | `true` | Preview changes without applying |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `memories_analyzed` | number | Total memories analyzed |
| `memories_decayed` | number | Memories that would be/were decayed |
| `avg_decay_factor` | number | Average decay factor applied |
| `decayed_memories` | object[] | Details of affected memories |
| `dry_run` | boolean | Whether this was a preview |

Each decayed memory object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Memory UUID |
| `content_preview` | string | Truncated content preview |
| `old_importance` | number | Previous importance |
| `new_importance` | number | New/proposed importance |
| `decay_factor` | number | Decay multiplier applied |
| `days_since_access` | number | Days since last access |
| `access_count` | number | Total access count |

#### Example

```json
// Request
{
  "namespace": "temporary",
  "decay_function": "exponential",
  "half_life_days": 14,
  "dry_run": true
}

// Response
{
  "memories_analyzed": 50,
  "memories_decayed": 12,
  "avg_decay_factor": 0.72,
  "decayed_memories": [
    {
      "id": "a1b2c3d4-...",
      "content_preview": "Old meeting notes from...",
      "old_importance": 0.6,
      "new_importance": 0.43,
      "decay_factor": 0.72,
      "days_since_access": 21,
      "access_count": 2
    }
  ],
  "dry_run": true
}
```

#### Errors

- `DecayError` - Decay calculation or application failed
- `ValidationError` - Invalid parameter values

---

### reinforce

Boost memory importance based on usage or explicit feedback. Reinforcement increases importance and updates access timestamp.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_ids` | string[] | Yes | - | Memory IDs to reinforce |
| `boost_type` | string | No | `"additive"` | Type: `additive`, `multiplicative`, or `set_value` |
| `boost_amount` | number | No | `0.1` | Amount to boost (0.0-1.0) |
| `update_access` | boolean | No | `true` | Update last_accessed timestamp |

**Boost Types:**
- `additive`: new_importance = old_importance + boost_amount
- `multiplicative`: new_importance = old_importance * (1 + boost_amount)
- `set_value`: new_importance = boost_amount

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `memories_reinforced` | number | Number of memories reinforced |
| `avg_boost` | number | Average boost applied |
| `reinforced` | object[] | Details of reinforced memories |
| `not_found` | string[] | IDs that were not found |

Each reinforced memory object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Memory UUID |
| `content_preview` | string | Truncated content |
| `old_importance` | number | Previous importance |
| `new_importance` | number | New importance |
| `boost_applied` | number | Actual boost applied |

#### Example

```json
// Request
{
  "memory_ids": ["a1b2c3d4-...", "b2c3d4e5-..."],
  "boost_type": "additive",
  "boost_amount": 0.2
}

// Response
{
  "memories_reinforced": 2,
  "avg_boost": 0.2,
  "reinforced": [
    {
      "id": "a1b2c3d4-...",
      "content_preview": "Important architecture decision...",
      "old_importance": 0.5,
      "new_importance": 0.7,
      "boost_applied": 0.2
    }
  ],
  "not_found": []
}
```

#### Errors

- `ReinforcementError` - Reinforcement operation failed
- `ValidationError` - Invalid parameter values

---

### extract

Automatically extract memories from conversation text. Uses pattern matching to identify facts, decisions, and key information.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to extract memories from |
| `namespace` | string | No | `"extracted"` | Namespace for extracted memories |
| `min_confidence` | number | No | `0.5` | Minimum confidence to extract (0.0-1.0) |
| `deduplicate` | boolean | No | `true` | Skip if similar memory exists |
| `dedup_threshold` | number | No | `0.9` | Similarity threshold for deduplication (0.7-0.99) |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `candidates_found` | number | Number of extraction candidates |
| `memories_created` | number | Number of memories stored |
| `deduplicated_count` | number | Number skipped due to duplicates |
| `extractions` | object[] | Details of each extraction |

Each extraction object:

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Extracted content |
| `confidence` | number | Extraction confidence (0.0-1.0) |
| `pattern_matched` | string | Pattern type: `decision`, `definition`, `solution`, `error`, `pattern`, `explicit`, `important` |
| `start_pos` | number | Start position in source text |
| `end_pos` | number | End position in source text |
| `stored` | boolean | Whether memory was stored |
| `memory_id` | string | Memory ID if stored |

#### Example

```json
// Request
{
  "text": "We decided to use PostgreSQL for the main database. The solution to the performance issue was adding an index on user_id.",
  "namespace": "meeting-notes",
  "min_confidence": 0.6
}

// Response
{
  "candidates_found": 2,
  "memories_created": 2,
  "deduplicated_count": 0,
  "extractions": [
    {
      "content": "We decided to use PostgreSQL for the main database",
      "confidence": 0.85,
      "pattern_matched": "decision",
      "start_pos": 0,
      "end_pos": 51,
      "stored": true,
      "memory_id": "a1b2c3d4-..."
    },
    {
      "content": "The solution to the performance issue was adding an index on user_id",
      "confidence": 0.80,
      "pattern_matched": "solution",
      "start_pos": 53,
      "end_pos": 121,
      "stored": true,
      "memory_id": "b2c3d4e5-..."
    }
  ]
}
```

#### Errors

- `ExtractionError` - Extraction process failed
- `ValidationError` - Text is empty or invalid

---

### consolidate

Merge similar or duplicate memories to reduce redundancy. Finds memories above similarity threshold and merges them using the specified strategy.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | Yes | - | Namespace to consolidate |
| `similarity_threshold` | number | No | `0.85` | Minimum similarity for duplicates (0.7-0.99) |
| `strategy` | string | No | `"keep_highest_importance"` | Merge strategy (see below) |
| `dry_run` | boolean | No | `true` | Preview without changes |
| `max_groups` | integer | No | `50` | Maximum groups to process (1-100) |

**Consolidation Strategies:**
- `keep_newest`: Keep the most recently created memory
- `keep_oldest`: Keep the oldest memory
- `keep_highest_importance`: Keep the memory with highest importance
- `merge_content`: Combine content from all similar memories

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `groups_found` | number | Number of duplicate groups found |
| `memories_merged` | number | Memories merged (kept) |
| `memories_deleted` | number | Memories deleted |
| `groups` | object[] | Details of each group |
| `dry_run` | boolean | Whether this was a preview |

Each group object:

| Field | Type | Description |
|-------|------|-------------|
| `representative_id` | string | ID of the kept memory |
| `member_ids` | string[] | IDs of all group members |
| `avg_similarity` | number | Average similarity within group |
| `action_taken` | string | Action: `merged`, `deleted`, or `preview` |

#### Example

```json
// Request
{
  "namespace": "notes",
  "similarity_threshold": 0.9,
  "strategy": "keep_highest_importance",
  "dry_run": true
}

// Response
{
  "groups_found": 3,
  "memories_merged": 3,
  "memories_deleted": 5,
  "groups": [
    {
      "representative_id": "a1b2c3d4-...",
      "member_ids": ["a1b2c3d4-...", "b2c3d4e5-...", "c3d4e5f6-..."],
      "avg_similarity": 0.94,
      "action_taken": "preview"
    }
  ],
  "dry_run": true
}
```

#### Errors

- `ConsolidationError` - Consolidation process failed
- `NamespaceNotFoundError` - Namespace does not exist
- `ValidationError` - Invalid parameter values

---

## Utility Operations

### stats

Get database statistics and health metrics.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | No | - | Filter stats to specific namespace |
| `include_index_details` | boolean | No | `true` | Include detailed index statistics |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `total_memories` | number | Total memory count |
| `memories_by_namespace` | object | Memory counts per namespace |
| `storage_bytes` | number | Total storage in bytes |
| `storage_mb` | number | Total storage in megabytes |
| `estimated_vector_bytes` | number | Estimated vector storage size |
| `has_vector_index` | boolean | Whether vector index exists |
| `has_fts_index` | boolean | Whether full-text search index exists |
| `indices` | object[] | Index details (if requested) |
| `num_fragments` | number | Number of storage fragments |
| `needs_compaction` | boolean | Whether compaction is recommended |
| `table_version` | number | Current table version |
| `oldest_memory_date` | string | ISO 8601 timestamp of oldest memory |
| `newest_memory_date` | string | ISO 8601 timestamp of newest memory |
| `avg_content_length` | number | Average content length in characters |

Each index object:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Index name |
| `index_type` | string | Index type (IVF_PQ, HNSW, etc.) |
| `column` | string | Indexed column |
| `num_indexed_rows` | number | Rows in index |
| `status` | string | `ready`, `building`, or `needs_update` |

#### Example

```json
// Request
{
  "include_index_details": true
}

// Response
{
  "total_memories": 1250,
  "memories_by_namespace": {
    "default": 500,
    "architecture": 250,
    "notes": 500
  },
  "storage_bytes": 52428800,
  "storage_mb": 50.0,
  "estimated_vector_bytes": 1920000,
  "has_vector_index": true,
  "has_fts_index": true,
  "indices": [
    {
      "name": "vector_idx",
      "index_type": "IVF_PQ",
      "column": "vector",
      "num_indexed_rows": 1250,
      "status": "ready"
    }
  ],
  "num_fragments": 3,
  "needs_compaction": false,
  "table_version": 42,
  "oldest_memory_date": "2025-01-01T00:00:00Z",
  "newest_memory_date": "2026-01-31T12:00:00Z",
  "avg_content_length": 150.5
}
```

#### Errors

- `StorageError` - Failed to retrieve statistics

---

### namespaces

List all namespaces with memory counts and date ranges.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_stats` | boolean | No | `true` | Include memory counts and date ranges |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `namespaces` | object[] | List of namespace info |
| `total_namespaces` | number | Total namespace count |
| `total_memories` | number | Total memories across all namespaces |

Each namespace object:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Namespace name |
| `memory_count` | number | Number of memories |
| `oldest_memory` | string | ISO 8601 timestamp |
| `newest_memory` | string | ISO 8601 timestamp |

#### Example

```json
// Request
{
  "include_stats": true
}

// Response
{
  "namespaces": [
    {
      "name": "default",
      "memory_count": 500,
      "oldest_memory": "2025-01-01T00:00:00Z",
      "newest_memory": "2026-01-31T12:00:00Z"
    },
    {
      "name": "architecture",
      "memory_count": 250,
      "oldest_memory": "2025-03-15T00:00:00Z",
      "newest_memory": "2026-01-30T10:00:00Z"
    }
  ],
  "total_namespaces": 2,
  "total_memories": 750
}
```

#### Errors

- `StorageError` - Failed to retrieve namespace list

---

### delete_namespace

Delete all memories in a namespace. **DESTRUCTIVE** - use `dry_run` first to preview.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | Yes | - | Namespace to delete |
| `confirm` | boolean | No | `false` | Must be `true` when `dry_run=false` |
| `dry_run` | boolean | No | `true` | Preview deletion without executing |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `namespace` | string | Deleted namespace name |
| `memories_deleted` | number | Number of memories deleted |
| `success` | boolean | Whether operation succeeded |
| `message` | string | Status message |
| `dry_run` | boolean | Whether this was a preview |

#### Example

```json
// Request (preview)
{
  "namespace": "old-notes",
  "dry_run": true
}

// Response
{
  "namespace": "old-notes",
  "memories_deleted": 150,
  "success": true,
  "message": "Would delete 150 memories from namespace 'old-notes'",
  "dry_run": true
}

// Request (actual deletion)
{
  "namespace": "old-notes",
  "confirm": true,
  "dry_run": false
}

// Response
{
  "namespace": "old-notes",
  "memories_deleted": 150,
  "success": true,
  "message": "Deleted 150 memories from namespace 'old-notes'",
  "dry_run": false
}
```

#### Errors

- `NamespaceNotFoundError` - Namespace does not exist
- `NamespaceOperationError` - Deletion failed
- `ValidationError` - `confirm=true` required when `dry_run=false`

---

### rename_namespace

Rename a namespace, moving all its memories to the new name.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `old_namespace` | string | Yes | - | Current namespace name |
| `new_namespace` | string | Yes | - | New namespace name |

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `old_namespace` | string | Original namespace name |
| `new_namespace` | string | New namespace name |
| `memories_renamed` | number | Number of memories renamed |
| `success` | boolean | Whether operation succeeded |
| `message` | string | Status message |

#### Example

```json
// Request
{
  "old_namespace": "temp-notes",
  "new_namespace": "archived-notes"
}

// Response
{
  "old_namespace": "temp-notes",
  "new_namespace": "archived-notes",
  "memories_renamed": 75,
  "success": true,
  "message": "Renamed 75 memories from 'temp-notes' to 'archived-notes'"
}
```

#### Errors

- `NamespaceNotFoundError` - Source namespace does not exist
- `NamespaceOperationError` - Target namespace already exists
- `ValidationError` - Invalid namespace names

---

### export_memories

Export memories to file (Parquet, JSON, or CSV format).

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `output_path` | string | Yes | - | Path for output file |
| `format` | string | No | auto | Export format: `parquet`, `json`, or `csv` (auto-detected from extension) |
| `namespace` | string | No | - | Export only this namespace (all if not specified) |
| `include_vectors` | boolean | No | `true` | Include embedding vectors in export |

**Notes:**
- Parquet format uses zstd compression
- CSV format may not include vectors by default (configurable)
- File paths are validated against allowed directories

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `format` | string | Export format used |
| `output_path` | string | Actual output path |
| `memories_exported` | number | Number of memories exported |
| `file_size_bytes` | number | File size in bytes |
| `file_size_mb` | number | File size in megabytes |
| `namespaces_included` | string[] | Namespaces in export |
| `duration_seconds` | number | Export duration |
| `compression` | string | Compression used (if any) |

#### Example

```json
// Request
{
  "output_path": "./exports/memories-backup.parquet",
  "namespace": "architecture",
  "include_vectors": true
}

// Response
{
  "format": "parquet",
  "output_path": "./exports/memories-backup.parquet",
  "memories_exported": 250,
  "file_size_bytes": 1048576,
  "file_size_mb": 1.0,
  "namespaces_included": ["architecture"],
  "duration_seconds": 0.5,
  "compression": "zstd"
}
```

#### Errors

- `ExportError` - Export operation failed
- `PathSecurityError` - Path outside allowed directories or traversal attempt
- `ValidationError` - Invalid format or path

---

### import_memories

Import memories from file with validation. Always use `dry_run=true` first to preview.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_path` | string | Yes | - | Path to source file |
| `format` | string | No | auto | Import format: `parquet`, `json`, or `csv` |
| `namespace_override` | string | No | - | Override namespace for all imported memories |
| `deduplicate` | boolean | No | `false` | Skip records similar to existing memories |
| `dedup_threshold` | number | No | `0.95` | Similarity threshold for deduplication (0.7-0.99) |
| `validate` | boolean | No | `true` | Validate records before import |
| `regenerate_embeddings` | boolean | No | `false` | Generate new embeddings (required if vectors missing) |
| `dry_run` | boolean | No | `true` | Validate without importing |

**Notes:**
- Maximum import records is configurable (default: 100,000)
- File size limits apply (configurable)
- Vector dimensions must match if importing with vectors

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | string | Source file path |
| `format` | string | Detected/specified format |
| `total_records_in_file` | number | Records in source file |
| `memories_imported` | number | Memories successfully imported |
| `memories_skipped` | number | Records skipped (dedup/validation) |
| `memories_failed` | number | Records that failed |
| `validation_errors` | object[] | Validation error details |
| `namespace_override` | string | Namespace override if used |
| `duration_seconds` | number | Import duration |
| `dry_run` | boolean | Whether this was a validation-only run |
| `imported_memories` | object[] | First 10 imported memories (preview) |

Validation error object:

| Field | Type | Description |
|-------|------|-------------|
| `row_number` | number | Row with error |
| `field` | string | Field with error |
| `error` | string | Error description |
| `value` | string | Invalid value |

#### Example

```json
// Request (validation)
{
  "source_path": "./exports/memories-backup.parquet",
  "dry_run": true
}

// Response
{
  "source_path": "./exports/memories-backup.parquet",
  "format": "parquet",
  "total_records_in_file": 250,
  "memories_imported": 0,
  "memories_skipped": 0,
  "memories_failed": 0,
  "validation_errors": [],
  "duration_seconds": 0.2,
  "dry_run": true,
  "imported_memories": []
}

// Request (actual import)
{
  "source_path": "./exports/memories-backup.parquet",
  "namespace_override": "restored",
  "dry_run": false
}

// Response
{
  "source_path": "./exports/memories-backup.parquet",
  "format": "parquet",
  "total_records_in_file": 250,
  "memories_imported": 250,
  "memories_skipped": 0,
  "memories_failed": 0,
  "validation_errors": [],
  "namespace_override": "restored",
  "duration_seconds": 2.5,
  "dry_run": false,
  "imported_memories": [
    {
      "id": "new-uuid-1-...",
      "content_preview": "Use repository pattern...",
      "namespace": "restored"
    }
  ]
}
```

#### Errors

- `MemoryImportError` - Import operation failed
- `PathSecurityError` - Path outside allowed directories
- `FileSizeLimitError` - File exceeds size limit
- `ImportRecordLimitError` - Too many records in file
- `DimensionMismatchError` - Vector dimensions don't match
- `ValidationError` - Invalid format or parameters

---

### hybrid_recall

Search memories using combined vector and keyword (full-text) search.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `alpha` | number | No | `0.5` | Balance: `1.0`=pure vector, `0.0`=pure keyword, `0.5`=balanced |
| `limit` | integer | No | `5` | Maximum number of results (1-100) |
| `namespace` | string | No | - | Filter to specific namespace |
| `min_similarity` | number | No | `0.0` | Minimum similarity threshold (0.0-1.0) |

**Alpha Values:**
- `1.0`: Pure semantic/vector search
- `0.0`: Pure keyword/full-text search
- `0.5`: Equal balance of both
- `0.7`: Favor semantic similarity
- `0.3`: Favor keyword matching

#### Returns

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query |
| `alpha` | number | Alpha value used |
| `memories` | object[] | Matching memories |
| `total` | number | Number of results |
| `search_type` | string | Search type used |

Each memory object includes standard fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `vector_score` | number | Score from vector search (if applicable) |
| `fts_score` | number | Score from full-text search (if applicable) |

#### Example

```json
// Request
{
  "query": "database repository pattern",
  "alpha": 0.7,
  "limit": 5
}

// Response
{
  "query": "database repository pattern",
  "alpha": 0.7,
  "memories": [
    {
      "id": "a1b2c3d4-...",
      "content": "Use repository pattern for data access",
      "similarity": 0.88,
      "namespace": "architecture",
      "tags": ["patterns", "design"],
      "importance": 0.8,
      "created_at": "2026-01-15T10:00:00Z",
      "metadata": {},
      "vector_score": 0.92,
      "fts_score": 0.82
    }
  ],
  "total": 1,
  "search_type": "hybrid"
}
```

#### Errors

- `ValidationError` - Query is empty or invalid alpha
- `EmbeddingError` - Failed to generate query embedding

---

## Error Reference

### Error Types

| Error Type | When Raised |
|------------|-------------|
| `ValidationError` | Invalid input parameters (empty content, out-of-range values) |
| `MemoryNotFoundError` | Memory ID doesn't exist |
| `NamespaceNotFoundError` | Namespace doesn't exist |
| `NamespaceOperationError` | Namespace operation failed (rename, delete) |
| `StorageError` | Database operation failed |
| `EmbeddingError` | Embedding generation failed |
| `ClusteringError` | Clustering failed (too few memories) |
| `VisualizationError` | Visualization generation failed |
| `InsufficientMemoriesError` | Operation needs more memories than available |
| `JourneyError` | Journey path cannot be computed |
| `WanderError` | Wander cannot continue |
| `DecayError` | Decay calculation or application failed |
| `ReinforcementError` | Reinforcement operation failed |
| `ExtractionError` | Memory extraction failed |
| `ConsolidationError` | Consolidation operation failed |
| `ExportError` | Memory export failed |
| `MemoryImportError` | Memory import failed |
| `PathSecurityError` | File path violates security constraints |
| `FileSizeLimitError` | File exceeds size limit |
| `ImportRecordLimitError` | Import file has too many records |
| `DimensionMismatchError` | Imported vector dimensions don't match |

### Error Response Format

All errors follow this format:

```json
{
  "error": "ErrorTypeName",
  "message": "Human-readable description of what went wrong",
  "isError": true
}
```

### Security Errors

Security-related errors (PathSecurityError, etc.) are intentionally vague to prevent information disclosure:

```json
{
  "error": "PathSecurityError",
  "message": "Path security violation (traversal): /path/to/file",
  "isError": true
}
```

### Internal Errors

For unexpected internal errors, a reference ID is provided instead of stack traces:

```json
{
  "error": "InternalError",
  "message": "An internal error occurred. Reference: a1b2c3d4",
  "isError": true
}
```

Report internal errors with the reference ID for troubleshooting.
