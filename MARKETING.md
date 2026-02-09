# Spatial Memory MCP

## Knowledge as a Navigable Landscape, Not a Filing Cabinet

**The first semantic memory system designed for how LLMs actually think.**

---

## The Problem with Memory Today

Every AI memory solution asks the same question: "How do we store things so we can find them later?"

Wrong question.

The right question is: "How do we give AI systems memory that works like memory—with forgetting, reinforcement, association, and discovery?"

Spatial Memory MCP answers that question.

---

## Zero Cognitive Load: Memory That Just Works

Most memory systems require developers to explicitly manage storage, retrieval, and organization. Spatial Memory MCP takes a radically different approach.

**You never think about memory. Claude handles everything automatically.**

Here's how it works:

1. **Automatic Context Loading** — Claude loads relevant memories at session start. No initialization code. No explicit queries.

2. **Intelligent Recognition** — Claude recognizes memory-worthy moments in conversation and asks simply: "Save this decision? y/n"

3. **Natural Synthesis** — When you ask about past context, Claude synthesizes answers naturally. No raw JSON. No memory IDs. Just knowledge.

4. **MCP Instructions Injection** — Memory behaviors are injected directly into Claude's system prompt. Zero configuration required.

The result? Developers interact with an AI that genuinely remembers—without writing a single line of memory management code.

---

## This Is Not Storage. This Is Cognitive Architecture.

Traditional vector databases store embeddings. Spatial Memory MCP implements a cognitive memory model inspired by established memory research (see [References](#references)).

### Memory Decay (Ebbinghaus Forgetting Curve)
Memories fade over time if not accessed—just like human memory. Choose exponential, linear, or step decay functions. Set half-life periods. Define importance floors. Memories that matter persist. Memories that don't, gracefully fade.

### Reinforcement Learning
Memories grow stronger through use. Every access, every retrieval, every reference boosts importance scores. Frequently needed knowledge rises to the surface. Additive, multiplicative, or explicit value boosts—you control the reinforcement dynamics.

### Consolidation
Similar memories merge intelligently. Four strategies available:
- **keep_newest** — Preserve recent knowledge
- **keep_oldest** — Maintain original context
- **keep_highest_importance** — Prioritize what matters
- **merge_content** — Synthesize into unified memories

### Auto-Extraction
Point the system at a conversation transcript. It extracts facts, decisions, and key information automatically. Pattern matching identifies what's worth remembering. Deduplication prevents redundancy.

---

## Spatial Navigation: The Innovation That Changes Everything

Here's what makes Spatial Memory MCP fundamentally different: **knowledge exists in semantic space, and you can navigate it.**

### Journey (SLERP Interpolation)
Navigate between two memories using spherical interpolation. Discover what's conceptually *between* them. Start with "machine learning basics" and end with "production deployment"—Journey reveals the learning path: feature engineering, model validation, containerization, monitoring.

### Wander (Temperature-Based Random Walk)
Serendipitous exploration through memory space. Low temperature: focused, related concepts. High temperature: unexpected connections. Set a starting point or start anywhere. Let the system surprise you with what it finds.

### Regions (HDBSCAN Clustering)
Automatic topic discovery without predefined labels. The system identifies natural clusters in your knowledge base. Returns representative memories and auto-generated keywords for each region. See how your knowledge self-organizes.

### Visualize (UMAP Projection)
Project your memories into 2D or 3D space. Export as JSON, Mermaid diagrams, or SVG. See your knowledge landscape. Identify gaps. Understand relationships.

**No other memory system offers this.** Competitors give you search. We give you exploration.

---

## 22 Tools. Complete Lifecycle Management.

| Category | Spatial Memory MCP | Typical Competitor |
|----------|-------------------|-------------------|
| Core Operations | remember, recall, forget, nearby | 3-4 tools |
| Batch Operations | remember_batch, forget_batch | None |
| Spatial Navigation | journey, wander, regions, visualize | None |
| Lifecycle Management | decay, reinforce, consolidate, extract | None |
| Hybrid Search | hybrid_recall with tunable alpha | Single mode |
| Namespace Management | namespaces, delete_namespace, rename_namespace | Basic or none |
| Data Portability | export_memories, import_memories | None |
| Administration | stats, health | Minimal |

**22 production-ready tools** covering the complete memory lifecycle—from ingestion to navigation to maintenance to export.

---

## Hybrid Search: The Best of Both Worlds

Semantic search finds conceptually similar content. Keyword search finds exact matches. Why choose?

**Spatial Memory MCP's hybrid_recall** combines both with a tunable alpha parameter:

- `alpha = 1.0` — Pure semantic search
- `alpha = 0.5` — Balanced hybrid
- `alpha = 0.0` — Pure keyword search

Search for "authentication error handling" and find both semantically related security discussions AND exact keyword matches in error logs. One query. Complete results.

---

## Enterprise-Grade Infrastructure

Production deployment demands production infrastructure.

### Connection Management
- **Connection pooling** with LRU eviction and health checks
- **Cross-process file locking** for multi-instance safety
- **Circuit breaker pattern** prevents cascade failures
- **NFS/SMB detection** warns about shared filesystem risks

### Data Integrity
- **Atomic batch operations** with rollback on failure
- **Add-before-delete pattern** prevents data loss during consolidation
- **Namespace rename rollback** recovers from partial failures
- **Streaming consolidation** handles large namespaces safely

### Observability
- **Request tracing** with correlation IDs
- **Per-agent rate limiting** using token bucket algorithm
- **Response caching** with configurable TTL
- **Health endpoints** with detailed diagnostics

### Security
- **Path traversal prevention** — No filesystem escapes
- **SQL injection detection** — 13 patterns covering major attack vectors
- **Input validation** — Pydantic models throughout
- **Credential masking** — Sensitive data never logged
- **Dry-run modes** — Preview destructive operations safely

---

## Performance by Default

### ONNX Runtime
2.75x faster than PyTorch. 60% less memory. No GPU required. CPU inference that actually performs.

### Intelligent Indexing
Auto-indexing triggers at 10K+ memories using IVF_PQ. Sub-linear search complexity at scale. Configurable thresholds for your workload.

### Efficient Defaults
- **all-MiniLM-L6-v2** embedding model (~80MB)
- Fast inference, quality embeddings
- Batch operations for bulk ingestion
- Connection pooling eliminates overhead

---

## Competitor Landscape

| Feature | Spatial Memory MCP | mcp-mem0 | mcp-memory-service | Memory Bank MCP |
|---------|-------------------|----------|-------------------|-----------------|
| Total Tools | 22 | 3 | ~6 | ~5 |
| Spatial Navigation | Yes | No | No | No |
| Memory Decay | Yes | No | No | No |
| Reinforcement | Yes | No | No | No |
| Hybrid Search | Yes | No | Partial | No |
| Batch Operations | Yes | No | No | No |
| HDBSCAN Clustering | Yes | No | No | No |
| SLERP Interpolation | Yes | No | No | No |
| Visualization | Yes | No | No | No |
| Enterprise Features | Full Suite | Minimal | Basic | Basic |
| External Dependencies | None (embedded) | PostgreSQL | Varies | File-based |
| Test Coverage | 1,750+ tests | Minimal | Unknown | Unknown |

**mcp-mem0**: 3 tools, 4 commits, requires PostgreSQL infrastructure.

**mcp-memory-service**: No spatial navigation, no lifecycle management, limited cognitive features.

**Memory Bank MCP**: Go-based, no cognitive memory model, basic CRUD operations.

**basic-memory**: No vector search, file-based limitations, no semantic understanding.

---

## Use Cases

### Development Teams
Maintain architectural decisions across sessions. Remember why you chose that database, that API design, that authentication pattern. Never repeat the same discussion. Never lose hard-won context.

### AI Agents
Long-running agents that learn and adapt. Remember user preferences. Track conversation history. Build genuine relationships through persistent, intelligent memory.

### Knowledge Management
Transform organizational knowledge into a navigable landscape. Let teams explore connections they didn't know existed. Surface relevant context automatically.

### Research and Analysis
Navigate between concepts. Discover intermediate ideas. Visualize knowledge structures. Enable serendipitous discovery that filing systems can never provide.

---

## Technical Specifications

- **Language**: Python 3.10+
- **Database**: LanceDB (embedded vector database)
- **Embedding**: ONNX Runtime with all-MiniLM-L6-v2
- **Clustering**: HDBSCAN
- **Projection**: UMAP
- **Interpolation**: SLERP (Spherical Linear Interpolation)
- **Architecture**: Clean architecture with dependency injection
- **Type Safety**: Full type hints, mypy strict mode
- **Test Suite**: 1,750+ passing tests
- **Version**: 1.7.0

---

## Getting Started

```bash
# Install
pip install spatial-memory-mcp

# Or with UV
uv pip install spatial-memory-mcp
```

Configure in Claude Desktop's MCP settings. The system injects its instructions automatically. Start a conversation. Claude remembers.

No initialization code. No explicit memory calls. No cognitive overhead.

**Just memory that works like memory should.**

---

## The Bottom Line

Most AI memory systems are databases with an API. Spatial Memory MCP is a cognitive architecture.

It decays. It reinforces. It consolidates. It discovers.

It treats knowledge as a navigable landscape—not a filing cabinet.

And it does all of this with zero cognitive load on developers. Claude handles memory. You handle what matters.

---

## References

The cognitive memory model is inspired by established research:

1. **Ebbinghaus, H. (1885)**. *Memory: A Contribution to Experimental Psychology*. The foundational research on the forgetting curve showing how memory retention decays exponentially over time. [Read the translation](https://psychclassics.yorku.ca/Ebbinghaus/index.htm)

2. **Settles, B. & Meeder, B. (2016)**. *A Trainable Spaced Repetition Model for Language Learning*. Duolingo's half-life regression (HLR) algorithm for optimizing memory retention. [ACL Anthology](https://aclanthology.org/P16-1174/)

3. **FSRS Algorithm**. Free Spaced Repetition Scheduler - a modern open-source algorithm for optimizing review intervals based on memory research. [GitHub](https://github.com/open-spaced-repetition/fsrs4anki)

---

**Spatial Memory MCP**

*Knowledge as a navigable landscape, not a filing cabinet.*

[GitHub Repository](https://github.com/arman-tech/spatial-memory-mcp) | [Documentation](https://github.com/arman-tech/spatial-memory-mcp#readme) | [PyPI](https://pypi.org/project/spatial-memory-mcp/)
