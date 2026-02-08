# Cognitive Offloading Design Document

**Status**: Draft (Research Complete, Pending Approval)
**Author**: arman-tech + Claude
**Date**: 2026-02-05
**Version**: spatial-memory-mcp v2.0 (proposed)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Design Goals](#design-goals)
- [Architecture Overview](#architecture-overview)
- [Layer 3: Client Hooks (Capture Layer)](#layer-3-client-hooks-capture-layer)
- [Layer 1: Smart Gating (Server-Side)](#layer-1-smart-gating-server-side)
- [Layer 2: Session Extraction (Server-Side)](#layer-2-session-extraction-server-side)
- [Reinforcement Loop](#reinforcement-loop)
- [Queue-First Communication](#queue-first-communication)
- [Signal Detection](#signal-detection)
- [Deduplication Pipeline](#deduplication-pipeline)
- [Project as First-Class Field](#project-as-first-class-field)
- [Project Detection Edge Cases](#project-detection-edge-cases)
- [Client Detection and Hook Distribution](#client-detection-and-hook-distribution)
- [Privacy, Security, and Performance](#privacy-security-and-performance)
- [Token Cost Analysis](#token-cost-analysis)
- [Competitive Landscape](#competitive-landscape)
- [Configuration Audit](#configuration-audit)
- [Implementation Phases](#implementation-phases)
- [Open Questions](#open-questions)
- [Research References](#research-references)

---

## Problem Statement

Today, memory capture in spatial-memory-mcp depends on the LLM deciding to call `remember`. This creates three problems:

1. **Cognitive burden**: The LLM must continuously evaluate whether information is worth saving, consuming attention and tokens that could go toward the user's actual task.
2. **Inconsistent capture**: Different LLM models, temperatures, and prompt contexts produce wildly different save rates. Important decisions, bug fixes, and architecture choices slip through.
3. **Cold start**: Each session starts from scratch unless the LLM proactively calls `recall` — which the MCP server instructions encourage but cannot enforce.

The v1.9.3 tiered auto-save instructions (Part A) improved the LLM's guidance, but the LLM remains the sole decision-maker. Part B moves the intelligence server-side and into client hooks, making capture **deterministic and reliable**.

---

## Design Goals

| Priority | Goal | Metric |
|----------|------|--------|
| **P0** | No loss of critical information (decisions, fixes, errors, architecture) | <5% miss rate on Tier 1 signals |
| **P1** | Minimize token overhead per session | <7,000 tokens/session for memory operations |
| **P2** | Work across MCP clients (Claude Code, Cursor, others) | Layers 1+2 portable to any client |
| **P3** | Zero configuration for end users | Plugin install or `setup_hooks` tool |
| **P4** | Respect privacy — never capture secrets or PII | Redaction pipeline before storage |

---

## Architecture Overview

The system uses three capture layers plus a reinforcement feedback loop. Each layer is independent — if hooks aren't available (Layer 3), server-side layers (1+2) still function.

```
                    ┌─────────────────────────────────────┐
                    │         MCP Client (LLM)            │
                    │  Claude Code / Cursor / Other       │
                    └──────┬────────────────┬─────────────┘
                           │                │
                    ┌──────▼──────┐  ┌──────▼──────────┐
                    │  Layer 3    │  │  LLM calls       │
                    │  Client     │  │  remember()      │
                    │  Hooks      │  │  (existing flow) │
                    └──────┬──────┘  └──────┬───────────┘
                           │                │
                    ┌──────▼──────┐         │
                    │  Queue File │         │
                    │  (Maildir)  │         │
                    └──────┬──────┘         │
                           │                │
                    ┌──────▼────────────────▼───────────┐
                    │         MCP Server                 │
                    │  ┌─────────────────────────────┐   │
                    │  │  Layer 1: Smart Gating       │   │
                    │  │  (quality + dedup on save)   │   │
                    │  └─────────────┬───────────────┘   │
                    │                │                    │
                    │  ┌─────────────▼───────────────┐   │
                    │  │  Layer 2: Session Extraction  │   │
                    │  │  (passive log + periodic      │   │
                    │  │   extract on flush/timer)     │   │
                    │  └─────────────┬───────────────┘   │
                    │                │                    │
                    │  ┌─────────────▼───────────────┐   │
                    │  │  Reinforcement Loop           │   │
                    │  │  (recall frequency → boost)   │   │
                    │  └─────────────────────────────┘   │
                    │                                     │
                    │         LanceDB Storage              │
                    └─────────────────────────────────────┘
```

### Layer Summary

| Layer | Location | Trigger | Depends on LLM? | Portable? |
|-------|----------|---------|------------------|-----------|
| Layer 1: Smart Gating | Server | `remember()` call | Yes (LLM initiates) | All clients |
| Layer 2: Session Extraction | Server | Timer / flush / session-end | No | All clients |
| Layer 3: Client Hooks | Client | PostToolUse / PreCompact / Stop | No | Claude Code + Cursor |
| Reinforcement Loop | Server | Every `recall()` call | No | All clients |

---

## Layer 3: Client Hooks (Capture Layer)

Client hooks are the **first line of capture**. They intercept tool calls and session events to identify memory-worthy content before it can be lost.

### Hook Events Used

| Hook Event | When It Fires | What It Captures |
|------------|---------------|------------------|
| **PostToolUse** | After every tool call | Decisions, fixes, errors from tool output |
| **PreCompact** | Before context compaction | Full transcript about to be compressed |
| **Stop** | Session ends | Remaining uncaptured content |

### Hook Script Behavior

Each hook runs the same core logic:

1. **Receive** JSON on stdin (tool name, input, output, transcript path)
2. **Analyze** content against signal detection patterns (see [Signal Detection](#signal-detection))
3. **Redact** any detected secrets/PII (see [Privacy](#privacy-security-and-performance))
4. **Write** qualifying memories to queue directory (see [Queue-First Communication](#queue-first-communication))
5. **Return** immediately (target: <100ms synchronous, or use `async: true`)

### PostToolUse Hook

Fires after every tool call. The hook examines the tool output for Tier 1 and Tier 2 signals.

**Input** (Claude Code):
```json
{
  "session_id": "abc-123",
  "tool_name": "Edit",
  "tool_input": { "file_path": "/src/auth.py", "old_string": "...", "new_string": "..." },
  "tool_response": "File edited successfully",
  "transcript_path": "/tmp/transcript.jsonl"
}
```

**Input** (Cursor):
```json
{
  "tool_name": "Edit",
  "tool_input": { "file_path": "/src/auth.py", "old_string": "...", "new_string": "..." },
  "tool_output": "File edited successfully",
  "tool_use_id": "xyz-789",
  "duration": 42,
  "model": "claude-sonnet-4-5-20250929"
}
```

**Hook logic**:
- Skip read-only tools (Read, Glob, Grep) — they rarely produce memory-worthy output
- For write tools (Edit, Write, Bash): scan tool input/output for signal phrases
- For MCP tools: check if it's a spatial-memory tool (skip to avoid recursion)
- If signals found: construct memory JSON, write to queue

### PreCompact Hook

Fires before context window compaction. This is the **critical safety net** — content about to be lost forever.

**Hook logic**:
1. Read transcript from `transcript_path`
2. Run full extraction (regex + heuristic patterns) over the transcript
3. Filter out content already captured by PostToolUse (positional overlap check)
4. Write candidate memories to queue
5. Must complete quickly — compaction is time-sensitive

**Claude Code only.** Cursor has `preCompact` as well (added in later versions). For clients without this hook, the Stop hook serves as the fallback.

### Stop Hook

Fires when the session ends. Last chance to capture anything missed.

**Hook logic**:
1. Read full transcript
2. Run extraction over content not yet captured
3. Write to queue
4. Server processes queue on next session's first tool call

---

## Layer 1: Smart Gating (Server-Side)

Layer 1 runs on every `remember()` call — whether initiated by the LLM directly, or by the server processing queued memories from hooks.

### Quality Gate

Before storing a memory, evaluate its quality:

```
Score = signal_score × (0.3)
      + content_length_score × (0.2)
      + structure_score × (0.2)
      + context_richness × (0.3)

If score < threshold (default 0.3): reject
If score 0.3–0.5: store with reduced importance
If score > 0.5: store at requested importance
```

**Signal score**: Presence and count of Tier 1/2 signal phrases (see [Signal Detection](#signal-detection)).

**Content length score**: Normalized length — too short (<20 chars) scores 0, optimal (100-500 chars) scores 1.0, very long (>2000 chars) scores 0.7 (may need splitting).

**Structure score**: Has tags? Has reasoning ("because", "so that")? Has specific names/paths?

**Context richness**: References files, functions, versions, or other concrete artifacts?

### Deduplication (for Direct Saves)

1. **Content hash** (SHA-256): Exact match check. Free, catches copy-paste duplicates.
2. **Vector similarity** (0.85–0.90 threshold): Catches paraphrased duplicates. Uses existing embedding infrastructure.
3. **LLM arbitration via tool response** (borderline 0.80–0.85 only): Instead of making a separate LLM API call, return the similar existing memory in the `remember()` tool response with `status: "potential_duplicate"`. The LLM (already in conversation) decides whether to proceed — zero extra token cost.

For **queue-processed saves** (background, no LLM in loop): use layers 1-2 only with strict 0.85 threshold. No LLM arbitration available — near-duplicates in 0.80-0.85 range are acceptable; `consolidate` can merge later.

No positional overlap check needed here (that's a Layer 2 concern for batch extraction).

---

## Layer 2: Session Extraction (Server-Side)

Layer 2 passively observes the session and periodically extracts memories without LLM involvement.

### Session Log

The server maintains an in-memory log of tool interactions:

```python
@dataclass
class ToolInteraction:
    timestamp: datetime
    tool_name: str
    tool_input: dict
    tool_output_summary: str  # Truncated to 500 chars
    memory_candidates: list[str]  # Signal phrases detected
```

**Not the full transcript** — just structured summaries to control memory usage.

### Extraction Triggers

| Trigger | When | Scope |
|---------|------|-------|
| **Timer** | Every N minutes (configurable, default 10) | Recent interactions since last extraction |
| **Interaction count** | Every N tool calls (default 20) | Recent interactions since last extraction |
| **Queue flush** | Server processes queued memories from hooks | Merge with hook candidates |
| **Session end** | Server shutdown / client disconnect | Full session log |

### Extraction Process

1. Collect unprocessed interactions from session log
2. Run signal detection over collected text
3. Run **all 4 dedup layers**:
   - Content hash (exact match)
   - Positional overlap (same-text across extraction windows)
   - Vector similarity (against existing memories)
   - LLM arbitration (optional, borderline cases)
4. Assign importance based on signal tier (Tier 1: 0.8–1.0, Tier 2: 0.5–0.7)
5. Auto-detect project (see [Project as First-Class Field](#project-as-first-class-field))
6. Store via Layer 1 (smart gating applies)

---

## Reinforcement Loop

The reinforcement loop is **already partially implemented** via the existing auto-decay and reinforce tools. This design formalizes the feedback cycle:

```
recall() called
    → memory returned to LLM
    → access_count incremented (existing)
    → importance boosted proportionally (existing reinforce logic)
    → if memory recalled 3+ times: mark as "core knowledge"
    → decay slows for frequently-accessed memories (existing adaptive half-life)
    → project_last_active updated for the queried project (NEW)
```

### Per-Project Decay

Auto-decay is scoped per project rather than globally:
- Each project tracks a `project_last_active` timestamp (updated on every `recall`/`remember` for that project)
- Decay is calculated relative to each project's last activity, not a global clock
- A project not accessed for 30 days has its memories decayed, even if other projects are active daily
- This prevents penalizing projects worked on cyclically (e.g., 2 weeks on A, 2 weeks on B)
- The `decay` tool (manual) can still operate globally or per-project/namespace
- Config: `SPATIAL_MEMORY_AUTO_DECAY_PER_PROJECT=true` (default when cognitive offloading is enabled)

### New: Signal Pattern Feedback

Track which signal patterns lead to memories that get recalled frequently:

```python
# When a memory created by signal detection gets recalled:
signal_effectiveness[pattern] += 1

# Periodically adjust signal weights:
# Patterns that produce useful memories → increase weight
# Patterns that produce never-recalled memories → decrease weight
```

This makes the system **self-tuning** over time — signal detection improves with usage.

---

## Queue-First Communication

### Why Not HTTP?

We evaluated 4 callback mechanisms for hook → server communication:

| Approach | Reliability | Complexity | Token Cost |
|----------|-------------|------------|------------|
| HTTP sidecar (claude-mem approach) | High | High (port mgmt, security) | Low |
| Context injection (additionalContext) | **60-70%** | Low | High (490-1,130/memory) |
| Queue file (Maildir pattern) | **~100%** | Low | Low (10-20/notification) |
| **Queue-first (selected)** | **~100%** | **Low** | **Low** |

Context injection was **rejected** after deep-dive research revealed:
- 60-70% LLM compliance rate (LLM frequently ignores injected context)
- Documented failures: Claude Code GitHub issues #10373, #14281, #13650, #16538
- Silent truncation when context window is near capacity
- No delivery guarantee or acknowledgment mechanism

### Maildir Queue Pattern

The Maildir pattern is a proven, crash-safe filesystem queue:

```
.spatial-memory/
└── pending-saves/
    ├── tmp/           # Hook writes here first (incomplete files)
    ├── new/           # Atomic rename from tmp/ (ready for processing)
    └── processed/     # Server moves here after processing (audit trail)
```

**Write flow** (hook side):
```
1. Generate unique filename: {timestamp}-{pid}-{random}.json
2. Write memory JSON to tmp/{filename}
3. Atomic rename: tmp/{filename} → new/{filename}
```

`os.rename()` is atomic on all platforms (Windows, macOS, Linux) when source and destination are on the same filesystem. This guarantees no partial reads.

**Read flow** (server side — background timer):
```
1. QueueProcessor thread polls new/ every 30 seconds
2. For each file in new/:
   a. Read and parse JSON
   b. Run through Layer 1 (smart gating + dedup)
   c. If accepted: store memory, move file to processed/
   d. If rejected: move file to processed/ with rejection reason
3. Record pending piggyback notifications
4. On next MCP tool response, append notification summary
```

### Queue File Format

```json
{
  "version": 1,
  "source_hook": "PostToolUse",
  "timestamp": "2026-02-05T10:30:00Z",
  "client": "claude-code",
  "project_root_dir": "/home/user/code/spatial-memory-mcp",
  "content": "Decided to use queue-first pattern instead of HTTP sidecar because...",
  "suggested_namespace": "decisions",
  "suggested_tags": ["architecture", "hooks", "queue"],
  "suggested_importance": 0.9,
  "signal_tier": 1,
  "signal_patterns_matched": ["decided to use X because Y"],
  "context": {
    "tool_name": "Edit",
    "file_path": "/home/user/code/spatial-memory-mcp/spatial_memory/server.py",
    "session_id": "abc-123"
  }
}
```

**Field notes:**
- `project_root_dir`: From `$CLAUDE_PROJECT_DIR` or `$CURSOR_PROJECT_DIR` env var (hook reads this, always available). The server resolves this to a full project identity (git remote URL, monorepo sub-project) during background processing. Cached per directory.
- `context.file_path`: From hook `tool_input.file_path` — only present for Read/Write/Edit tools. For other tools (Bash, MCP tools), this field is absent. The server uses `project_root_dir` as the primary project signal.
```

### Piggyback Notifications

When the server processes queued saves, it appends a brief note to the next tool response (10-20 tokens):

```
[Auto-saved 2 memories: "Queue-first pattern decision", "Hook script performance target"]
```

This keeps the LLM aware of what was captured without depending on it to act. The LLM can reference these in conversation naturally.

### Queue Housekeeping

- **Startup scan**: On server init, recover orphaned `tmp/` files older than 5 minutes (move to `new/` if valid JSON, delete if corrupt)
- `processed/` files older than 7 days: auto-delete
- `tmp/` files older than 1 hour: auto-delete (orphaned writes from crashed hooks)
- `new/` files older than 24 hours: log warning (server may not be processing)

---

## Signal Detection

Signal detection identifies memory-worthy content using regex patterns organized by tier.

### Tier 1 Signals (Auto-Save)

These indicate high-value information that should always be captured:

| Category | Signal Phrases |
|----------|---------------|
| **Decisions** | "decided to", "we chose", "going with X over Y", "the approach is", "selected X because" |
| **Bug fixes** | "the fix was", "resolved by", "the solution is", "fixed it by", "root cause was" |
| **Error causes** | "the issue was caused by", "failed because", "the error was due to", "broke because" |
| **Architecture** | "we'll structure it as", "the design is", "the architecture will be", "the pattern is" |

### Tier 2 Signals (Lower Importance)

| Category | Signal Phrases |
|----------|---------------|
| **Patterns** | "the trick is", "this pattern works", "always do X when", "the key insight" |
| **Preferences** | "I prefer", "the team standard is", "convention here is", "we always use" |
| **Configuration** | "you need to set", "the config requires", "important setting", "env var" |
| **Workarounds** | "watch out for", "the workaround is", "gotcha:", "caveat:" |

### Tier 3 (Never Save)

| Category | Description |
|----------|-------------|
| Greetings | "hello", "hi", "thanks", status updates |
| Speculative | "maybe we should", "I wonder if", "possibly" |
| Debugging noise | "let me check", "trying X", intermediate exploration |

### Additional Signal Categories (Future)

| Category | Signal Phrases |
|----------|---------------|
| **Constraints** | "must not", "never", "always ensure", "requirement:" |
| **Relationships** | "X depends on Y", "X calls Y", "X is part of Y" |
| **Versions/Config** | File paths, env variables, version numbers, ports |

### Signal Scoring

```python
def calculate_signal_score(text: str) -> tuple[float, int]:
    """Returns (score, tier). Score 0.0-1.0, tier 1-3."""
    tier1_matches = count_matches(text, TIER_1_PATTERNS)
    tier2_matches = count_matches(text, TIER_2_PATTERNS)
    tier3_matches = count_matches(text, TIER_3_PATTERNS)

    if tier3_matches > tier1_matches + tier2_matches:
        return (0.0, 3)  # Mostly noise
    if tier1_matches > 0:
        return (min(1.0, 0.6 + tier1_matches * 0.2), 1)
    if tier2_matches > 0:
        return (min(0.7, 0.3 + tier2_matches * 0.15), 2)
    return (0.1, 3)  # No clear signals
```

---

## Deduplication Pipeline

The system uses a 4-layer deduplication pipeline. Not all layers are needed in every context.

### Layer Definitions

| Layer | Method | Cost | Catches | Used In |
|-------|--------|------|---------|---------|
| 1 | Content hash (SHA-256) | Free | Exact duplicates, copy-paste | All contexts |
| 2 | Positional overlap | Cheap | Same-text across extraction windows | Layer 2 only |
| 3 | Vector similarity (0.85-0.90) | 1 embedding + search | Paraphrased duplicates | All contexts |
| 4 | LLM arbitration (via tool response) | Free (zero extra tokens) | Semantic nuance, borderline 0.80-0.85 | Direct `remember()` only |

### When Each Layer Applies

**Layer 1 (Smart Gating)** — direct `remember()` calls (LLM in conversation):
- Dedup Layer 1: Content hash
- Dedup Layer 3: Vector similarity
- Dedup Layer 4: LLM arbitration via tool response (borderline 0.80-0.85 only — returns candidate in response, LLM decides; zero extra token cost)
- No positional overlap needed (single saves, not batch extraction)

**Queue-processed saves** (background, no LLM in loop):
- Dedup Layer 1: Content hash
- Dedup Layer 3: Vector similarity (strict threshold: >0.85 = reject, <0.85 = accept)
- No LLM arbitration — threshold is sufficient for background processing
- Near-duplicates in the 0.80-0.85 range are acceptable; `consolidate` can merge later

**Layer 2 (Session Extraction)** — batch extraction from transcript:
- Dedup Layer 2: Positional overlap (same text extracted from overlapping windows)
- Dedup Layer 1: Content hash (before storing)
- Dedup Layer 3: Vector similarity (against existing memories)
- No LLM arbitration (extraction is server-side, no LLM in loop)
- All 3 layers needed because batch extraction is prone to duplicates

### Vector Similarity Thresholds

| Similarity | Action |
|-----------|--------|
| > 0.90 | Definite duplicate — reject silently |
| 0.85 - 0.90 | Likely duplicate — reject, log for audit |
| 0.80 - 0.85 | Borderline — send to LLM arbitration if enabled |
| < 0.80 | Distinct memory — accept |

---

## Project as First-Class Field

### Motivation

Currently, project context is either encoded in the namespace (e.g., `spatial-memory-mcp/decisions`) or in metadata. This overloads namespace semantics and prevents clean cross-project queries.

Industry validation: Mem0, cldmemory, memory-mcp, enhanced-mcp-memory, and Microsoft Kernel Memory all treat project/scope as a **first-class pre-filter**, not metadata. No existing MCP memory server has robust automatic project detection — we would be the first.

### Proposed Model Change

```python
class Memory(BaseModel):
    id: str
    content: str
    project: str = Field(default="", description="Project identifier, auto-detected")  # NEW
    namespace: str = Field(default="default", description="Content category")
    tags: list[str]
    importance: float
    metadata: dict
    # ... existing fields unchanged
```

### Project Auto-Detection Priority Chain

Deep-dive research revealed that several commonly-assumed detection signals are **unreliable** in MCP environments:

| Signal | Assumption | Reality |
|--------|-----------|---------|
| `os.getcwd()` | Returns workspace dir | Returns uvx cache dir, Cursor install dir, or static config path in ~80% of MCP deployments |
| MCP `roots/list` | Clients implement it | Claude Code declares but times out (issue #3315); Cursor broken per community reports |
| Hook `cwd` field | Available in hook data | Missing from Claude Code PostToolUse data (bug #16541, broke claude-mem) |

The **revised detection chain** accounts for these realities:

| Priority | Source | How It Works | Reliability |
|----------|--------|-------------|-------------|
| 1 | Explicit `project` param on tool call | User/LLM passes project name directly | 100% — user decides |
| 2 | File path from hook `tool_input` → walk up to git root | PostToolUse hook receives file paths (e.g., Edit `file_path`); walk up to find `.git`, extract remote URL | High — file paths are always accurate |
| 3 | `$CLAUDE_PROJECT_DIR` env var | Available to Claude Code hook scripts; points to project root | High — Claude Code only |
| 4 | MCP `roots/list` (3s timeout) | Request workspace folders from client; gracefully handle timeout | Medium — works for VS Code; broken in Claude Code + Cursor |
| 5 | Server config env var | `SPATIAL_MEMORY_PROJECT=my-project` set at install time | Medium — static, single-project only |
| 6 | Single project in DB | If only one project exists, use it | 100% — no ambiguity |
| 7 | Ask user | Return project list with memory counts, ask user to specify | 100% — last resort |

**Key change**: `os.getcwd()` is **not in the chain** — it is unreliable in MCP server environments.

### Project Identity Format

```
# Standard (single-project repo, with remote)
github.com/arman-tech/spatial-memory-mcp

# Monorepo sub-project
github.com/acme/platform:web

# Non-GitHub host
gitlab.com/org/group/subgroup/repo

# Azure DevOps (nested URL structure)
dev.azure.com/org/project/repo

# Local-only repo (no remote configured)
local/spatial-memory-mcp

# No git, identified from manifest
manifest/spatial-memory-mcp

# No git, no manifest
unaffiliated
```

**Normalization rules:**
- Strip `.git` suffix from remote URLs
- Lowercase host and org (case-insensitive on GitHub/GitLab/Bitbucket)
- Strip credentials from URL before storing (`https://user:token@host/...` → `host/...`)
- Resolve `insteadOf` aliases if raw URL is unparseable (e.g., `gh:org/repo`)
- Prefer `upstream` remote over `origin` for forks (store both in metadata)
- On Windows/macOS: normalize path case for identity comparisons

### Git Remote URL Parsing

Remote URLs come in many formats, all of which must be parsed:

| Format | Example | Notes |
|--------|---------|-------|
| HTTPS | `https://github.com/org/repo.git` | Most common |
| HTTPS (no .git) | `https://github.com/org/repo` | GitHub accepts both |
| SSH (SCP-like) | `git@github.com:org/repo.git` | Note the colon, not slash |
| SSH (URL form) | `ssh://git@github.com/org/repo.git` | Less common |
| SSH with port | `ssh://git@host:2222/org/repo.git` | SCP syntax can't specify port |
| Git protocol | `git://github.com/org/repo.git` | Read-only, rare |
| Local path | `/path/to/repo.git` | Local clones |
| With credentials | `https://user:token@github.com/org/repo.git` | Must strip before storing |

**Remote selection priority**: `upstream` (canonical for forks) → `origin` (default) → first available remote.

### Monorepo Sub-Project Detection

When the git root is found but the file is deep inside a monorepo, a second walk identifies the sub-project:

```
Given file: /monorepo/apps/web/src/index.tsx
  → Walk up, find .git at /monorepo/ (git root)
  → Continue walking from file toward git root, checking for manifest files:
    - /monorepo/apps/web/package.json exists (no "workspaces" field) → sub-project = "web"
  → Project identity = "github.com/acme/platform:web"
```

**Manifest files to detect sub-projects** (by ecosystem):

| Ecosystem | Workspace Root Marker | Sub-Project Marker |
|-----------|----------------------|-------------------|
| Node.js/npm | `package.json` with `"workspaces"` field | `package.json` without `"workspaces"` |
| pnpm | `pnpm-workspace.yaml` | `package.json` in member dirs |
| Rust | `Cargo.toml` with `[workspace]` section | `Cargo.toml` with `[package]` section |
| Python (uv) | `pyproject.toml` with `[tool.uv.workspace]` | `pyproject.toml` per package |
| Python (general) | `pyproject.toml` or `setup.py` at root | `pyproject.toml` in subdirs |
| Go | `go.work` (Go 1.18+) | `go.mod` per module |
| Java/Maven | `pom.xml` with `<modules>` | `pom.xml` per module |
| Java/Gradle | `settings.gradle` with `include()` | `build.gradle` per project |
| .NET | `.sln` / `.slnx` file | `.csproj` / `.fsproj` |
| C/C++ | `CMakeLists.txt` with `add_subdirectory()` | `CMakeLists.txt` per component |
| Bazel | `WORKSPACE` / `MODULE.bazel` | `BUILD` per package |
| Nx | `nx.json` at root | `project.json` per project |

**Shared library ambiguity**: In monorepos, `packages/shared-utils/` may be used by multiple apps. We attribute the memory to the sub-project containing the file being edited. If the user is editing the shared lib itself, the memory belongs to that shared lib sub-project.

### `.git` File Handling (Worktrees and Submodules)

The `.git` entry can be either a directory (normal clone) or a **file** (worktree or submodule):

```
# Git worktree .git file:
gitdir: /path/to/main/repo/.git/worktrees/feature-branch

# Git submodule .git file:
gitdir: ../../.git/modules/my-submodule
```

**Detection algorithm:**
1. Find `.git` entry (file or directory)
2. If it's a **file**: read and parse the `gitdir:` line, resolve the path (may be relative)
3. If it's a **directory**: verify it contains `HEAD`
4. Extract remote URL from the resolved git directory's `config`
5. For submodules: the submodule is treated as its own project (it has its own remote URL)

### Query Patterns

```python
# Default: pre-filter by auto-detected project + namespace
search.where("project = 'github.com/arman-tech/spatial-memory-mcp' AND namespace = 'decisions'",
             prefilter=True)

# All memories for current project (default recall behavior)
search.where("project = 'github.com/arman-tech/spatial-memory-mcp'", prefilter=True)

# Monorepo sub-project
search.where("project = 'github.com/acme/platform:web'", prefilter=True)

# Explicit cross-project search (user must opt-in via project="*")
# No project filter applied — searches all memories
search.where("namespace = 'decisions'", prefilter=True)
```

**Dedup scope**: Per-project. A memory in Project A is not considered a duplicate of a similar memory in Project B, since each may carry project-specific context.

### Recall Project Resolution

When `recall` is called, the `project` parameter is resolved using the detection chain cascade:

| Scenario | Behavior |
|----------|----------|
| `project` explicitly passed | Search that project |
| `project="*"` passed | Cross-project search (no project filter) |
| `project` omitted, auto-detect succeeds (file path / env var / roots) | Use auto-detected project silently |
| `project` omitted, auto-detect fails, 1 project in DB | Use that project |
| `project` omitted, auto-detect fails, multiple projects | Return project list, ask user to specify |

When the server cannot determine the project and multiple projects exist, it returns an informational response instead of results:

```
Multiple projects found. Please specify which project to search:
- github.com/arman-tech/spatial-memory-mcp (847 memories)
- github.com/acme/my-web-app (234 memories)
- local/data-pipeline (56 memories)
Or use project="*" to search all.
```

This avoids both silent wrong-project searches and unnecessary friction in the common case.

### Database Migration

Adding `project` field requires a LanceDB schema migration:
- Add `project` column with default `""` (empty string)
- Create index on `project` column for pre-filtering
- **Auto-backfill on first startup**: Scan existing memories and assign project from content analysis (file paths, project names, metadata hints)
- **CLI override** for manual correction:
  ```bash
  spatial-memory backfill-project --namespace decisions --project github.com/arman-tech/spatial-memory-mcp
  spatial-memory backfill-project --dry-run  # Preview assignments
  ```

---

## Project Detection Edge Cases

Deep-dive research identified 30 edge cases across 8 categories. This section documents all cases and how the system handles them.

### Critical Cases (Must Handle)

#### MCP Server Working Directory is Wrong

`os.getcwd()` returns the wrong path in ~80% of MCP deployments:
- **`uvx` sandbox**: Returns `~/.cache/uv/...`
- **Cursor**: Returns MCP server installation path
- **Claude Desktop**: Static config path, no workspace concept

**Mitigation**: `os.getcwd()` is excluded from the detection chain entirely. Use file paths from hook data, `$CLAUDE_PROJECT_DIR`, or MCP roots instead.

#### MCP `roots/list` Not Implemented

Claude Code declares `roots` capability but the handler isn't implemented — calls time out after 5+ seconds (issue #3315, closed NOT_PLANNED). Cursor's implementation is also broken per community reports.

**Mitigation**: Attempt `roots/list` with a 3-second timeout. On timeout or error, fall through to the next detection priority. Never block on this.

#### Hook `cwd` Field Missing

Claude Code PostToolUse hook data is missing the `cwd` field despite being documented (bug #16541). This is what broke `claude-mem`.

**Mitigation**: Use `$CLAUDE_PROJECT_DIR` environment variable instead (available to hook scripts, documented, works). For Cursor hooks, use the `cwd` field if present but don't depend on it.

#### Monorepo: Git Root at Wrong Level

In monorepos (~30-40% of professional projects), `.git` is at the repository root, not the sub-project level. Walking up to `.git` gives you `monorepo/` when you want `monorepo/apps/web/`.

**Mitigation**: After finding the git root, perform a second walk from the file path toward the git root, checking for manifest files (see [Monorepo Sub-Project Detection](#monorepo-sub-project-detection)). Distinguish workspace root manifests (e.g., `package.json` with `"workspaces"` field) from sub-project manifests.

#### Multi-Root Workspace: Which Folder?

VS Code/Cursor multi-root workspaces can contain folders from completely different disk locations. The MCP protocol has no concept of "active project" — only static root boundaries.

**Mitigation**: Priority 2 in the detection chain (file path from hook `tool_input`) naturally resolves this — the file path tells us exactly which workspace folder the user is working in. Each tool call may target a different project, and that's correct.

#### Git Submodule / Worktree: `.git` is a File

Since Git 1.7.8, submodules use a `.git` **file** (not directory) containing `gitdir: path/to/actual/git/dir`. Git worktrees use the same format.

**Mitigation**: When walking up and finding `.git`, check if it's a file or directory. If it's a file, read the `gitdir:` line and resolve the path (may be relative). Verify the resolved path exists. If it doesn't, treat as degraded detection and continue walking up.

#### Devcontainer / Remote Development: Path Mismatch

In devcontainers, the workspace is typically bind-mounted at `/workspaces/<folder-name>`. The container path is completely different from the host path. SSH remote development has the same issue.

**Mitigation**: This is generally acceptable — the MCP server runs inside the container and all paths are container paths. The project identity from git remote URL is path-independent and survives across environments. The only issue is if memories are created on the host and later accessed from the container (or vice versa) — the git remote URL identity ensures they match.

### High-Severity Cases (Should Handle)

#### Symlinks: Logical vs Physical Path

`os.getcwd()` returns the logical path (preserving symlinks), while `os.path.realpath()` resolves to the physical path. The same project directory accessed via different paths gets different identities.

**Mitigation**: Always canonicalize paths with `os.path.realpath()` before walking up for detection. This ensures consistent identity regardless of how the path was accessed. On Windows, this also resolves junction points (Python 3.8+).

#### Case Insensitivity (Windows + macOS)

On Windows (NTFS) and macOS (default APFS/HFS+), `C:\MyProject` and `c:\myproject` refer to the same directory. However, `os.path.normcase()` is a no-op on macOS despite case-insensitive filesystem.

**Mitigation**:
- Windows: Use `os.path.normcase()` (lowercases the path)
- macOS: Use `os.path.realpath()` (returns the canonical case) or `os.path.samefile()` for comparisons
- Project identity based on git remote URL sidesteps this entirely (URLs are case-normalized)

#### WSL Path Translation

Same project accessed as `C:\Users\dev\project` (Windows) and `/mnt/c/Users/dev/project` (WSL) gets different path-based identities.

**Mitigation**: Project identity from git remote URL is path-independent and identical in both environments. Only path-based fallbacks (no git) would see different identities — this is acceptable for the rare case.

#### No Remote Configured (Local-Only Repo)

~15-20% of experimental/learning projects have no git remote. No URL-based identity is possible.

**Mitigation**: Fall back to `local/{repo-directory-name}`. Extract the name from the git root directory. If the directory is later pushed to a remote, the project identity changes — this is a known limitation; the backfill CLI can reassign memories.

#### Multiple Remotes (Origin + Upstream)

~20-30% of open-source contributors have both `origin` (their fork) and `upstream` (canonical repo).

**Mitigation**: Prefer `upstream` remote (if exists) for canonical identity, then `origin`, then first available. Store the full remote list in metadata for debugging.

#### Same Repo Name, Different Org

`user/my-app` and `org/my-app` are different projects. Using repo name alone causes collisions.

**Mitigation**: Project identity includes the full `{host}/{org}/{repo}` path, not just the repo name. This is already handled by the identity format.

#### `.git` in Home Directory (Accidental `git init ~`)

If a user accidentally runs `git init` in their home directory, every file under `~` resolves to that repo.

**Mitigation**: Maintain a **blocklist** of directories that should never be treated as project roots:
- `$HOME` / `~`
- Filesystem root (`/`, `C:\`, `D:\`, etc.)
- Temp directories (`/tmp`, `%TEMP%`, `/var/tmp`)
- WSL mount roots (`/mnt`, `/mnt/c`)
- Common parent directories (`/Users`, `/home`)

If `.git` is found in a blocklisted directory, skip it and continue walking up (which will terminate at root with no match).

#### Monorepo: Workspace Root vs Sub-Project Manifest

A monorepo root often has a `package.json` with a `"workspaces"` field. Naive detection would treat this as a sub-project.

**Mitigation**: When a manifest file is found during the sub-project walk, check for workspace markers:
- `package.json`: Check for `"workspaces"` field → skip, this is the workspace root
- `Cargo.toml`: Check for `[workspace]` section without `[package]` → skip
- `pyproject.toml`: Check for `[tool.uv.workspace]` → skip

#### Concurrent Multi-Project Editing

A developer may have multiple editor windows connected to the same MCP server, each working on different projects.

**Mitigation**: Each tool call carries its own context (file paths in `tool_input`), so each `remember` or `recall` resolves its project independently per request. The server does not maintain a single "current project" — it resolves per call.

### Medium-Severity Cases (Nice to Handle)

#### No Version Control System

~6% of developers don't use git. Some use SVN (`.svn/`), Mercurial (`.hg/`), or no VCS at all.

**Mitigation**: After checking for `.git`, also check for `.hg/` and `.svn/` as project root markers. For non-VCS projects, fall back to manifest files (`package.json`, `pyproject.toml`, `Cargo.toml`, etc.) as the project root indicator, with `manifest/{name}` as the identity format.

#### Temporary / Scratch Files

Files in `/tmp/`, `~/Desktop/`, or other non-project locations have no project context.

**Mitigation**: If no project root is found after walking to the filesystem root, assign `project = "unaffiliated"`. These memories are still searchable via `project="*"` cross-project search.

#### Files Inside `node_modules/` / `.venv/`

A user debugging a dependency might edit files inside `node_modules/some-package/index.js`. This file has its own `package.json` but should be attributed to the parent project.

**Mitigation**: When walking up from a file, skip manifest files found inside known dependency/build directories:
- `node_modules/`, `.venv/`, `venv/`, `vendor/`
- `dist/`, `build/`, `target/`, `.next/`, `out/`
- `__pycache__/`, `.tox/`, `bin/`, `obj/`

Continue walking up past these directories to find the actual project root.

#### Credentials in Remote URL

~1-2% of configurations embed credentials: `https://user:token@github.com/org/repo.git`.

**Mitigation**: Strip the `user:pass@` portion from URLs before storing as project identity. Detect via `@` preceded by `://` in the URL.

#### Azure DevOps / GitLab Nested URL Structures

Azure DevOps: `https://dev.azure.com/{org}/{project}/_git/{repo}` (extra nesting level).
GitLab nested groups: `https://gitlab.com/org/group/subgroup/repo.git`.

**Mitigation**: Use a URL parsing library like `giturlparse` (PyPI) that handles all major hosting platforms, or implement platform-specific parsing rules. Normalize to `{host}/{full-path}` format.

#### `insteadOf` URL Rewriting

Git config can rewrite URLs: `[url "ssh://git@github.com/"] insteadOf = https://github.com/`. The stored remote URL might be a short alias like `gh:org/repo`.

**Mitigation**: Use `git remote get-url origin` which returns the raw URL. If the raw URL is unparseable (e.g., `gh:org/repo`), attempt to expand by reading git config `insteadOf` rules. If still unparseable, fall back to directory name.

#### Project Directory Renamed/Moved

If a user renames `~/projects/old-name` to `~/projects/new-name`, path-based identities break.

**Mitigation**: Git remote URL-based identity is path-independent and survives renames. Only `local/{dirname}` identities (no-remote repos) are affected. The backfill CLI can reassign.

### Low-Severity Cases (Defensive)

#### Bare Repo with `GIT_DIR`/`GIT_WORK_TREE` Env Vars

Some users (especially for dotfile management) use bare repos with environment variables to set the working tree. No `.git` exists in the working tree.

**Mitigation**: Do not attempt to detect this. These are rare (<5%) and the user can set `SPATIAL_MEMORY_PROJECT` explicitly or pass `project` in tool calls.

#### Circular Symlinks

Symlink `A → B → A` creates a loop. `os.path.realpath()` handles this correctly, but a naive walk could loop.

**Mitigation**: `os.path.realpath()` resolves cycles. Additionally, track visited directories during the upward walk and stop if a cycle is detected (by inode on Unix or canonical path).

#### Docker Volumes (Not Bind Mounts)

Docker volumes have no host path mapping at all. Files exist only inside the container.

**Mitigation**: Same as devcontainer handling — git remote URL identity is path-independent. No special handling needed.

#### Cursor Duplicate MCP Server Init in Multi-Root Workspaces

Cursor creates 4x `CreateClient` actions in multi-root workspaces.

**Mitigation**: Ensure server initialization is idempotent. The `QueueProcessor` and other background threads should handle multiple init/shutdown cycles gracefully.

### Complete Edge Case Summary

| # | Category | Edge Case | Severity | Handled By |
|---|----------|-----------|----------|------------|
| 1 | MCP env | `os.getcwd()` wrong | Critical | Excluded from chain |
| 2 | MCP env | `roots/list` broken | Critical | 3s timeout fallback |
| 3 | MCP env | Hook `cwd` missing | Critical | `$CLAUDE_PROJECT_DIR` env var |
| 4 | Monorepo | `.git` at wrong level | Critical | Manifest file sub-walk |
| 5 | Workspace | Multi-root: which folder? | Critical | File path from tool_input |
| 6 | Git | `.git` file (submodule/worktree) | Critical | Parse `gitdir:` line |
| 7 | Remote dev | Devcontainer path mismatch | Critical | Git remote URL identity (path-free) |
| 8 | Filesystem | Symlink logical vs physical | High | `os.path.realpath()` |
| 9 | Filesystem | Case insensitivity (Win/Mac) | High | `normcase()` + URL-based identity |
| 10 | Filesystem | WSL path translation | High | Git remote URL identity (path-free) |
| 11 | Git | No remote configured | High | `local/{dirname}` fallback |
| 12 | Git | Multiple remotes | High | Prefer upstream → origin → first |
| 13 | Git | Same name, different org | High | Full `{host}/{org}/{repo}` identity |
| 14 | Git | `.git` in home directory | High | Blocklist check |
| 15 | Monorepo | Workspace root vs sub-project | High | Check for `workspaces` field |
| 16 | MCP env | Concurrent multi-project | High | Per-request resolution |
| 17 | VCS | No git (SVN, Hg, none) | Medium | Check `.hg/`, `.svn/`, manifest fallback |
| 18 | Filesystem | Temp/scratch files | Medium | `project = "unaffiliated"` |
| 19 | Filesystem | Files in node_modules/.venv | Medium | Skip dependency dirs |
| 20 | Git | Credentials in URL | Medium | Strip before storing |
| 21 | Git | Azure DevOps nested URL | Medium | Platform-aware URL parsing |
| 22 | Git | GitLab nested groups | Medium | Full path normalization |
| 23 | Git | `insteadOf` rewriting | Medium | Read raw URL + expand aliases |
| 24 | Filesystem | Project renamed/moved | Medium | URL identity survives; CLI backfill |
| 25 | Git | Remote URL formats (10+) | Medium | Comprehensive URL parser |
| 26 | Filesystem | Path separator `\` vs `/` | Medium | `os.path.normpath()` |
| 27 | Git | Bare repo + GIT_DIR env | Low | User sets project explicitly |
| 28 | Filesystem | Circular symlinks | Low | `os.path.realpath()` + cycle check |
| 29 | Remote dev | Docker volume (no host path) | Low | Git remote URL identity |
| 30 | MCP env | Cursor duplicate init | Low | Idempotent initialization |

---

## Client Detection and Hook Distribution

### Client Detection via MCP Protocol

The MCP `initialize` handshake includes `clientInfo.name`. The server reads this to determine which client is connected:

```python
session = ctx.request_context.session
client_name = session.client_params.clientInfo.name  # e.g., "claude-code"
```

| `clientInfo.name` | Hooks Available |
|-------------------|-----------------|
| `claude-code` | PostToolUse, PreCompact, Stop |
| `cursor-vscode` | postToolUse, preCompact, stop, afterMCPExecution |
| Others | None (server-side only) |

### Hook Distribution Strategy

**Claude Code**: Package as a Claude Code plugin with bundled hooks.
```
.claude-plugin/
├── plugin.json          # Plugin manifest
├── hooks/
│   └── hooks.json       # Hook configuration
├── scripts/
│   ├── post_tool_use.py # PostToolUse handler
│   ├── pre_compact.py   # PreCompact handler
│   └── stop.py          # Stop handler
└── .mcp.json            # MCP server config
```

Install: `claude plugin install spatial-memory` (one command)

**Cursor**: Cursor's "Third-party skills" feature auto-reads Claude Code hook configs from `.claude/settings.json`. This means one hook config works for both clients.

**Other clients**: Fall back to Layers 1+2 only. Offer a `setup_hooks` MCP tool that returns the appropriate hook config as text for manual installation.

### Hook Script Language

Hook scripts are **Python** (pending cold-start benchmark vs Node.js). Python is preferred because:
- Project is already Python — shared code for signal detection and redaction
- Import `spatial_memory.hooks.signal_detection` for signal pattern matching
- Import `spatial_memory.hooks.redaction` for secret filtering
- Use the Maildir queue writer from `spatial_memory.hooks.queue`
- Target: <100ms synchronous execution, or use `async: true` for PostToolUse
- If Python cold-start exceeds Node.js by >50ms, consider Node.js with pattern files

---

## Privacy, Security, and Performance

### Privacy & Security

**Hooks run with full user permissions** — no sandboxing. The transcript contains everything: prompts, responses, tool outputs, potentially API keys and PII.

**Mandatory redaction before queue write:**

| Pattern | Example | Action |
|---------|---------|--------|
| API keys | `sk-...`, `AKIA...`, `ghp_...` | Replace with `[REDACTED_API_KEY]` |
| Passwords | `password=`, `secret=` | Replace with `[REDACTED]` |
| .env file contents | Tool output from reading .env | Skip entire memory |
| SSH keys | `-----BEGIN RSA PRIVATE KEY-----` | Skip entire memory |
| JWT tokens | `eyJ...` (base64 header) | Replace with `[REDACTED_TOKEN]` |
| Email addresses | `user@domain.com` | Optionally redact (configurable) |

**Additional protections:**
- Hook stdout/stderr: set `suppressOutput: true` to prevent leaking to LLM
- Cursor `afterMCPExecution`: fail-closed (hook crash = MCP call blocked) — handle errors gracefully
- Claude Code hooks: fail-open (hook crash = action proceeds) — memory loss but not user-blocking

### Performance

| Constraint | Target | Rationale |
|------------|--------|-----------|
| PostToolUse hook latency | <100ms sync, or async | Runs on every tool call |
| PreCompact hook latency | <200ms | Compaction is time-sensitive |
| Stop hook latency | <2s | Session ending, user waiting |
| Queue file write | ~50us | Filesystem atomic rename |
| Server queue processing | <50ms per file | Background thread, every 30s |
| Memory footprint (session log) | <10MB | Truncated tool outputs |

**Async hooks**: Claude Code supports `async: true` for non-blocking hooks. Use this for PostToolUse to avoid adding latency to every tool call. PreCompact and Stop should be synchronous (must complete before the event proceeds).

### Configuration (Opt-In)

Cognitive offloading should be **opt-in by default**. After a configuration audit of the existing 127 env vars, the new parameters were deliberately kept minimal — 6 new vars rather than the 11 originally proposed.

**Removed from original proposal** (configuration audit decision):
- 3 per-hook toggles (`HOOK_POST_TOOL_USE_ENABLED`, `HOOK_PRE_COMPACT_ENABLED`, `HOOK_STOP_ENABLED`) — the master toggle is sufficient; per-hook toggles add complexity without benefit
- `EXTRACTION_INTERACTION_COUNT` — a single timer trigger is simpler than timer + interaction count
- `AUTO_DECAY_PER_PROJECT` — always on when cognitive offloading is enabled; no reason to disable

**Final new parameters:**

```bash
# Master toggle — enables server-side intelligence + client hook processing
SPATIAL_MEMORY_COGNITIVE_OFFLOADING_ENABLED=false  # default: false

# Layer 2 extraction timer interval (minutes)
SPATIAL_MEMORY_EXTRACTION_INTERVAL_MINUTES=10

# Signal detection sensitivity (0.0-1.0, lower = capture more)
SPATIAL_MEMORY_SIGNAL_THRESHOLD=0.3

# Queue processor polling interval (seconds)
SPATIAL_MEMORY_QUEUE_POLL_INTERVAL_SECONDS=30

# Ingest-time dedup: reject if similarity exceeds this threshold
SPATIAL_MEMORY_DEDUP_VECTOR_THRESHOLD=0.85

# Explicit project override (priority 5 in the 7-level detection chain)
SPATIAL_MEMORY_PROJECT=my-project
```

See [Configuration Audit](#configuration-audit) for the full analysis of existing parameters and the `docs/CONFIGURATION_ALMANAC.md` for the comprehensive reference of all ~124 parameters.

---

## Token Cost Analysis

### Per-Session Token Budget

| Component | Tokens | Notes |
|-----------|--------|-------|
| Queue file processing (10 memories) | 100-200 | Piggyback notifications only |
| Stop hook extraction (safety net) | 980-6,510 | One-time, end of session |
| Server-side extraction (Layer 2) | 0 | No LLM involvement |
| LLM arbitration (dedup via tool response) | 0 | Candidates returned in existing response — zero extra cost |
| **Total (typical session)** | **1,080-6,710** | |

### Comparison with Alternatives

| Approach | Tokens/Session | Reliability |
|----------|---------------|-------------|
| Current (LLM-only, v1.9.3 instructions) | 2,000-15,000 | ~70% (LLM discretion) |
| Context injection (rejected) | 4,900-11,300 | ~65% (LLM compliance) |
| HTTP sidecar (claude-mem) | 500-1,000 | ~95% |
| **Queue-first (proposed)** | **1,080-6,710** | **~99%** |

Queue-first achieves the **highest reliability at the lowest token cost** — the optimal balance given the P0 goal of no information loss. The revised LLM arbitration approach (returning dedup candidates in tool responses instead of MCP sampling) eliminates the only variable token cost component.

---

## Competitive Landscape

| System | Capture Method | Dedup | Token Efficiency | Auto-Detect Project |
|--------|---------------|-------|------------------|-------------------|
| **Mem0** | 2-phase LLM pipeline | LLM-based (ADD/UPDATE/DELETE/NOOP) | 90% reduction vs baseline | Via user_id/agent_id |
| **claude-mem** | 5 lifecycle hooks + HTTP sidecar | Compression pipeline | ~500 tokens/observation | Via cwd |
| **mcp-memory-service** | Dual-layer regex + semantic | 85%+ semantic | 900 tokens at session start | No |
| **A-MEM** | Zettelkasten auto-linking | LLM-generated links | Up to 10x reduction | No |
| **Zep/Graphiti** | Temporal knowledge graph | Bi-temporal model | N/A (cloud service) | Via project_id |
| **spatial-memory (proposed)** | 3-layer hybrid + queue-first | 4-layer pipeline | 1,130-6,810/session | Git remote + 7-level fallback chain |

### Competitor Project Detection (Deep-Dive Findings)

| System | Project Detection | Method | Limitation |
|--------|------------------|--------|------------|
| **mem0-mcp** | None | Explicit `user_id` param | No auto-detection |
| **claude-mem** | Via `cwd` | Hook `cwd` field | **Currently broken** (Claude Code bug #16541) |
| **mcp-memory-service** | None | Tag-based organization | No auto-detection |
| **ContextStream** | Via `folder_path` | Explicit param in `session_init` | Requires LLM to pass path |
| **MCP official memory** | None | Flat global namespace | No project concept |
| **spatial-memory (proposed)** | **Robust auto-detect** | 7-level cascade + monorepo support | First MCP memory server with this |

### Key Differentiators

1. **Queue-first** (no HTTP): Simpler than claude-mem's sidecar, more reliable than context injection
2. **4-layer dedup**: More thorough than any competitor's single-method approach
3. **Signal pattern feedback**: Self-tuning detection that improves with usage
4. **Cross-client hooks**: Single config works for Claude Code + Cursor
5. **Graceful degradation**: Works without hooks (server-side only), just captures less
6. **Robust project auto-detection**: 7-level cascade with monorepo support, 30 edge cases handled — no competitor has this

---

## Configuration Audit

A full audit of `spatial_memory/config.py` was performed to understand the configuration surface area before adding cognitive offloading parameters. The codebase has **127 existing env vars**. Adding cognitive offloading responsibly required trimming dead code and avoiding parameter sprawl.

### Dead Code — 9 parameters to remove

These are defined in `config.py` but **never referenced** anywhere in the codebase:

| Parameter | Reason |
|-----------|--------|
| `SPATIAL_MEMORY_DEFAULT_NAMESPACE` | Tool schemas already default to `"default"` |
| `SPATIAL_MEMORY_DEFAULT_IMPORTANCE` | Tool schemas already default to `0.5` |
| `SPATIAL_MEMORY_MAX_BATCH_SIZE` | Never enforced; batch tools have their own validation |
| `SPATIAL_MEMORY_MAX_RECALL_LIMIT` | Never enforced; recall `limit` param already capped in tool schema |
| `SPATIAL_MEMORY_MIN_CLUSTER_SIZE` | Config value ignored; hardcoded in `SpatialConfig` |
| `SPATIAL_MEMORY_WARM_UP_ON_START` | Feature never implemented |
| `SPATIAL_MEMORY_BATCH_RATE_LIMIT` | Never referenced in code |
| `SPATIAL_MEMORY_BACKPRESSURE_QUEUE_ENABLED` | Marked "(future)", never implemented; replaced by cognitive offloading queue |
| `SPATIAL_MEMORY_BACKPRESSURE_QUEUE_MAX_SIZE` | Related to above |

### Partially Wired — 3 parameters need factory.py fix

These are defined in `config.py` but **not passed** from `factory.py` to `Database.__init__()`. The database uses hardcoded defaults instead:

| Parameter | Default | Issue |
|-----------|---------|-------|
| `SPATIAL_MEMORY_FILELOCK_ENABLED` | `true` | Not passed from factory → Database |
| `SPATIAL_MEMORY_FILELOCK_TIMEOUT` | `30.0` | Database hardcodes `30.0` instead of reading config |
| `SPATIAL_MEMORY_FILELOCK_POLL_INTERVAL` | `0.1` | Database hardcodes `0.1` instead of reading config |

**Fix**: Wire these through `factory.py:create_database()`.

### Stale `.env.example` — 3 legacy entries to remove

These entries exist in `.env.example` but have **no corresponding field** in `config.py`:

- `SPATIAL_MEMORY_DECAY_TIME_WEIGHT` — removed in v1.6.0
- `SPATIAL_MEMORY_DECAY_ACCESS_WEIGHT` — removed in v1.6.0
- `SPATIAL_MEMORY_DECAY_DAYS_THRESHOLD` — removed in v1.6.0

### Merge Candidate — Duplicate decay function settings

Two parameters control the same thing for different code paths:

| Parameter | Used By |
|-----------|---------|
| `SPATIAL_MEMORY_DECAY_DEFAULT_FUNCTION` | Manual `decay` tool |
| `SPATIAL_MEMORY_AUTO_DECAY_FUNCTION` | Auto-decay during `recall` |

Both accept the same values (`exponential`, `linear`, `step`). In practice, nobody sets different functions for manual vs. auto decay. Recommend merging into a single `SPATIAL_MEMORY_DECAY_FUNCTION` in a future version.

### Parameter Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| Existing active | 118 | 127 minus 9 dead code |
| New (cognitive offloading) | 6 | Trimmed from 11 originally proposed |
| **Total after v2.0** | **124** | |

Full reference for all parameters: [`docs/CONFIGURATION_ALMANAC.md`](CONFIGURATION_ALMANAC.md)

---

## Implementation Phases

### Phase 1: Foundation (Server-Side)

**Scope**: Layer 1 (smart gating) + `project` field + queue processor

| Task | Description |
|------|-------------|
| Add `project` field to Memory model | Schema change + migration + auto-backfill |
| Add `project` param to query tool schemas | Add optional `project` to recall, hybrid_recall, remember, remember_batch, nearby, regions, visualize, consolidate, extract, stats, namespaces, export, import |
| Project auto-detection (7-level cascade) | Explicit param → file path walk → `$CLAUDE_PROJECT_DIR` → MCP roots (3s timeout) → env var → single-project shortcut → ask user |
| Hook project detection (env var approach) | Hook reads `$CLAUDE_PROJECT_DIR` / `$CURSOR_PROJECT_DIR`, includes `project_root_dir` in queue file |
| Server project identity resolution | Resolve `project_root_dir` → git remote URL → monorepo sub-project; cache per directory |
| Git remote URL parser | Handle 10+ URL formats (HTTPS, SSH, SCP-like, with credentials, Azure DevOps, GitLab nested groups) |
| Monorepo sub-project detection | Manifest file walk with workspace root vs sub-project distinction (10 ecosystems) |
| `.git` file handling | Parse `gitdir:` for worktrees and submodules |
| Path normalization | `os.path.realpath()`, case normalization, blocklist for home/root/tmp dirs |
| `backfill-project` CLI command | Manual project assignment with `--dry-run` support |
| Content hash dedup | SHA-256 check before vector similarity |
| LLM arbitration via tool response | Return `status: "potential_duplicate"` with candidate in `remember()` response for borderline 0.80-0.85 cases |
| Quality gate on `remember()` | Signal scoring + threshold |
| Queue processor (background thread) | 30s timer, polls `pending-saves/new/`, Maildir pattern |
| Startup queue scan | Recover orphaned `tmp/` files on server init |
| Piggyback notifications | Append save summary to tool responses |
| Per-project decay | Track `project_last_active`, scope auto-decay per project |
| Remove 9 dead config parameters | Delete unused fields from `config.py` (see [Configuration Audit](#configuration-audit)) |
| Wire filelock settings through factory.py | Pass `filelock_enabled`, `filelock_timeout`, `filelock_poll_interval` from config to `Database.__init__()` |
| Clean up `.env.example` | Remove 3 stale legacy entries (`DECAY_TIME_WEIGHT`, `DECAY_ACCESS_WEIGHT`, `DECAY_DAYS_THRESHOLD`) |
| Configuration | 6 new env vars for cognitive offloading settings |
| Benchmark: Python vs Node.js hook cold-start | Measure and document latency difference |

### Phase 2: Extraction Engine (Server-Side)

**Scope**: Layer 2 (session extraction) + signal detection library

| Task | Description |
|------|-------------|
| Session log | In-memory tool interaction log |
| Signal detection module | Regex patterns for Tier 1/2/3 |
| Extraction engine | Timer/count/flush triggers |
| Positional overlap dedup | For batch extraction |
| Redaction pipeline | Secret/PII filtering |

### Phase 3: Client Hooks

**Scope**: Layer 3 (hooks) + distribution

| Task | Description |
|------|-------------|
| Hook scripts | PostToolUse, PreCompact, Stop handlers |
| Queue writer (Maildir) | Atomic tmp/ → new/ write |
| Claude Code plugin | Plugin manifest + bundled hooks |
| Cursor compatibility | Verify .claude/ auto-read works |
| `setup_hooks` MCP tool | Fallback for manual installation |

### Phase 4: Reinforcement & Tuning

**Scope**: Reinforcement loop + self-tuning signal detection

| Task | Description |
|------|-------------|
| Signal pattern tracking | Log which patterns produce recalled memories |
| Adaptive signal weights | Increase/decrease pattern weights based on recall frequency |
| LLM arbitration (optional) | Borderline dedup cases |
| Metrics & dashboards | Capture rates, dedup rates, token costs |

---

## Resolved Design Decisions

Decisions made during design review (2026-02-05):

### 1. Hook Script Language — Python (benchmark first)

**Decision**: Use Python for hook scripts, but benchmark Python vs Node.js cold-start latency first. Leaning toward Python since the project is already Python and the shared signal detection / redaction code can be imported directly. If the cold-start difference is minimal (<50ms), Python wins on code reuse.

### 2. Queue Processing Frequency — Background timer (30s)

**Decision**: Process queue on a **30-second background timer thread**, not on every tool call. This avoids adding latency to MCP tool responses. Follows the same pattern as the existing `DecayManager` background flush thread.

Implementation:
- `QueueProcessor` thread starts on server init
- Polls `pending-saves/new/` every 30 seconds
- Also processes on server shutdown (drain remaining files)
- Piggyback notification appended to next tool response after processing

### 3. LLM Arbitration — Via tool response (zero extra cost)

**Decision**: Keep LLM arbitration (dedup Layer 4) for borderline cases (vector similarity 0.80–0.85), but implement via tool response pattern instead of MCP sampling (which is not supported by any major MCP client). When `remember()` detects a borderline duplicate, return the existing memory in the response with `status: "potential_duplicate"` — the LLM already in conversation decides. For background queue-processed saves, use strict 0.85 threshold only (no LLM in loop). This has **zero additional token cost**.

### 4. Existing Memory Backfill — Auto-backfill + CLI override

**Decision**: When the `project` field is added:
- **Auto-backfill**: Scan existing memories and attempt to assign a project based on content analysis (file paths, project names in text, metadata hints)
- **CLI override**: Provide a CLI command for manual backfill/correction:
  ```bash
  spatial-memory backfill-project --namespace decisions --project spatial-memory-mcp
  spatial-memory backfill-project --dry-run  # Preview assignments
  ```
- Migration runs auto-backfill on first startup after upgrade

### 5. Cross-Project Memory Sharing — Per-project isolation with smart resolution

**Decision**: Deduplication is **scoped per project**. Each project maintains its own memory space.

- A "use UTC" memory in Project A and Project B are independent memories
- Each may have different project-specific context, so they're not true duplicates
- `recall` resolves project via 7-level cascade: explicit param → file path from hooks → `$CLAUDE_PROJECT_DIR` → MCP roots (3s timeout) → env var → single-project shortcut → ask user
- `recall` with `project: "*"` enables explicit cross-project search
- No automatic sharing/promotion between projects
- When auto-detect fails and multiple projects exist, the server returns the project list and asks the user to specify — never silently searches the wrong project
- `os.getcwd()` is **excluded** from the detection chain (unreliable in MCP environments)

This keeps the architecture simple and avoids false positive dedup across projects. Universal patterns naturally get captured per-project when they come up in context.

### 6. Hook Crash Recovery — Startup scan

**Decision**: Add a startup scan in addition to the 1-hour orphan cleanup:
- On server init, scan `pending-saves/tmp/` for orphaned files
- Files older than 5 minutes in `tmp/`: attempt to recover (move to `new/` if valid JSON, delete if corrupt)
- Log recovered/deleted counts at startup
- The 1-hour periodic cleanup remains as ongoing safety net

### 7. Project Detection — Hybrid hook + server approach

**Decision**: Split project detection between the hook (fast, instant) and the server (full resolution, background).

**Hook side** (runs on every PostToolUse/PreCompact/Stop):
- Read `$CLAUDE_PROJECT_DIR` (Claude Code) or `$CURSOR_PROJECT_DIR` (Cursor) environment variable — these are always available to hook scripts and point to the project root directory
- If available, include in the queue file as `project_root_dir`
- Extract `file_path` from `tool_input` (only available for Read/Write/Edit tools — other tools like `remember`, `Bash` etc. may not have a file path)
- Include both in the queue file — the hook does NOT resolve git remote URLs (too slow for hook latency targets)

**Server side** (background queue processing, no latency constraint):
- Receives `project_root_dir` and/or `file_path` from the queue file
- Resolves full project identity: walk up to `.git`, parse remote URL, detect monorepo sub-project
- Caches identity per `project_root_dir` (subsequent saves from same dir skip git resolution)
- For direct `remember()` calls (not from queue): use the 7-level cascade as designed

This gives hooks sub-millisecond overhead (just reading an env var) while the server does the heavy lifting asynchronously.

### 8. LLM Arbitration — Tool response pattern (no MCP sampling)

**Decision**: MCP `sampling/createMessage` is **not viable** — Claude Code, Cursor, and Claude Desktop all lack support (Claude Code issue #1785, 8+ months open, only ~5-10% of MCP clients implement it).

**For direct `remember()` calls** (LLM is in conversation):
- Run dedup layers 1-3 (content hash, vector similarity)
- If vector similarity is in the borderline range (0.80-0.85), return the similar memory in the tool response with `status: "potential_duplicate"` and the existing memory's content
- The LLM (already in conversation) decides whether to proceed or skip — this is "free" since the LLM is already processing
- No extra API call needed

**For queue-processed saves** (background, no LLM in loop):
- Run dedup layers 1-3 only (content hash, vector similarity)
- Use the vector similarity threshold strictly: >0.85 = reject, <0.85 = accept
- **No LLM arbitration** for background saves — the 0.85 threshold is sufficient
- Edge cases (0.80-0.85 borderline) may occasionally produce near-duplicates, but this is acceptable — the consolidation tool can merge them later

This approach has **zero additional token cost** for LLM arbitration — the dedup candidates are returned in an existing tool response.

### 9. `project` Parameter on Tool Schemas — Query tools only

**Decision**: Add an optional `project` parameter to **query and scoping tools** but NOT to ID-based tools:

**Tools that get `project` parameter** (search/scope context matters):
- `recall`, `hybrid_recall` — search within project
- `remember`, `remember_batch` — assign project on save
- `nearby` — scope neighbor search
- `regions`, `visualize` — scope clustering/visualization
- `consolidate`, `extract` — scope lifecycle operations
- `stats`, `namespaces` — scope reporting
- `export_memories`, `import_memories` — scope data transfer

**Tools that do NOT get `project` parameter** (operate on specific IDs or global):
- `forget`, `forget_batch` — operate by memory ID (already scoped)
- `journey` — operates between two specific memory IDs
- `wander` — starts from a specific memory ID
- `decay`, `reinforce` — operate by memory ID or namespace
- `health` — system-wide check

**Special values:**
- `project` omitted → auto-detect via 7-level cascade
- `project="*"` → cross-project search (no project filter)
- `project="github.com/org/repo"` → explicit project

### 10. Per-Project Decay — Scoped auto-decay

**Decision**: Auto-decay is scoped per project, not global. Always enabled when cognitive offloading is active — no separate configuration toggle (removed `SPATIAL_MEMORY_AUTO_DECAY_PER_PROJECT` per config audit; a toggle for this adds no value).

**Behavior:**
- When `recall()` or `hybrid_recall()` is called, auto-decay applies only to memories in the queried project
- Track `project_last_active` timestamp per project — updated on every recall/remember for that project
- Decay is relative to each project's last activity, not a global clock
- A project not accessed for 30 days has its memories decayed, even if other projects are active daily
- The `decay` tool (manual) can still operate globally or per-project/namespace

**Rationale**: Global decay would unfairly penalize projects that are worked on cyclically (e.g., 2 weeks on Project A, 2 weeks on Project B). Per-project decay ensures that returning to Project B after 2 weeks doesn't find all its memories degraded just because Project A was active.

---

## Open Questions

(No remaining open questions — all resolved above.)

---

## Research References

All research findings are stored in the `cognitive-offloading-research` namespace in spatial memory (15+ memories). Key references:

### Architecture & Approach
- **Option 6 Hybrid Architecture**: Evaluated 6 approaches, selected 3-layer hybrid
- **Context Injection Reliability**: GitHub issues #10373, #14281, #13650, #16538
- **Maildir Pattern**: Proven filesystem queue, used by email servers for 30+ years

### Client & Hook Research
- **MCP Client Detection**: `clientInfo.name` in initialize handshake
- **Cursor Hooks**: v1.7 (Oct 2025), `.cursor/hooks.json`, Third-party skills compatibility
- **Claude Code Hook Bugs**: `cwd` missing (issue #16541), `roots/list` not implemented (issue #3315)

### Project Detection
- **MCP `roots` Specification**: Protocol revision 2025-06-18, `file://` URIs, `listChanged` notifications
- **MCP Roots Client Support**: Claude Code (broken), Cursor (broken), VS Code (works) — [apify/mcp-client-capabilities](https://github.com/apify/mcp-client-capabilities)
- **MCP CWD Problem**: `os.getcwd()` unreliable — uvx sandbox, Cursor install path, Claude Desktop static ([python-sdk #1520](https://github.com/modelcontextprotocol/python-sdk/issues/1520))
- **Git Remote URL Parsing**: 10+ formats, `insteadOf` rewriting, credential stripping
- **Monorepo Detection**: 10 ecosystem manifest formats, workspace root vs sub-project distinction
- **Git Worktree/Submodule**: `.git` file format, `gitdir:` parsing

### MCP Protocol Capabilities
- **MCP `sampling/createMessage`**: Not supported by Claude Code (issue #1785, 8+ months open), Cursor, or Claude Desktop. Only ~5-10% of MCP clients implement it. Not viable for LLM arbitration.
- **Hook `tool_input` data shapes**: Only Read/Write/Edit always have `file_path`. Bash has `command`. MCP tools vary. `$CLAUDE_PROJECT_DIR` and `$CURSOR_PROJECT_DIR` env vars are always available to hook scripts.
- **Hybrid project detection approach**: Hooks read env var (instant, sub-ms), server resolves full identity in background (git remote URL, monorepo sub-project detection).

### Configuration Audit
- **127 existing env vars**: Full audit of `config.py`, cross-referenced with actual usage in factory.py, server.py, database.py
- **9 dead code params**: Defined but never read (`default_namespace`, `default_importance`, `max_batch_size`, `max_recall_limit`, `min_cluster_size`, `warm_up_on_start`, `batch_rate_limit`, `backpressure_queue_*`)
- **3 partially wired params**: `filelock_*` not passed through factory.py
- **3 stale `.env.example` entries**: Legacy decay settings removed in v1.6.0
- **Comprehensive reference**: `docs/CONFIGURATION_ALMANAC.md` — Java Almanac-style entry for every parameter

### Competitive Analysis
- **Mem0, claude-mem, mcp-memory-service, A-MEM, Zep**: Capture methods, dedup, project awareness
- **A-MEM (NeurIPS 2025)**: Zettelkasten auto-linking approach
- **KuzuMemory**: Type-dependent decay (semantic/procedural = no decay, episodic = 30 days)
- **No competitor has robust project auto-detection**: claude-mem relies on broken `cwd` field; others require explicit params
