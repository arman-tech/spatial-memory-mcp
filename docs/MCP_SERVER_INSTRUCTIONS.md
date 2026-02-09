# MCP Server Instructions: Auto-Injected AI Behavioral Guidelines

## Discovery

When users install an MCP server, the server can provide **instructions** that are automatically injected into Claude's system prompt. This enables MCP servers to define behavioral guidelines that Claude follows without requiring users to manually configure anything.

## Technical Implementation

The MCP Python SDK's `Server` class accepts an `instructions` parameter:

```python
from mcp.server import Server

server = Server(
    name="spatial-memory",
    version="1.10.0",
    instructions="""
Your behavioral instructions here...
"""
)
```

These instructions are:
- Sent during the MCP initialize handshake
- Automatically concatenated into Claude's system prompt
- Displayed under "# MCP Server Instructions" in Claude's context
- Require **zero user action** - fully automatic

## Why This Matters

### The Problem
- Users install MCP servers but don't know optimal usage patterns
- AI assistants use tools reactively rather than proactively
- Knowledge capture requires manual effort ("save this to memory")
- Each session starts from scratch without prior context

### The Solution
Server-provided instructions enable:
1. **Zero cognitive load** - Claude handles memory mechanics automatically
2. **Proactive behavior** - Claude recognizes memory-worthy moments
3. **Consistent UX** - All users get the same optimized experience
4. **Seamless context** - Previous learnings auto-load at session start

## Comparison of Approaches

| Approach | Auto-loaded? | User Action Required | Scope |
|----------|--------------|---------------------|-------|
| `~/.claude/CLAUDE.md` | Yes | User must create/edit | Global |
| `<project>/CLAUDE.md` | Yes | User must create/edit | Project |
| MCP `instructions` param | Yes | **None** | Per-server |

## spatial-memory-mcp Instructions

The following instructions are injected when spatial-memory-mcp connects:

```
## Spatial Memory System

You have access to a persistent semantic memory system. Use it proactively to
build cumulative knowledge across sessions.

### Session Start
At conversation start, call `recall` with the user's apparent task/context to
load relevant memories. Present insights naturally:
- Good: "Based on previous work, you decided to use PostgreSQL because..."
- Bad: "The database returned: [{id: '...', content: '...'}]"

### Auto-Save Behavior

Memories are saved using a 3-tier system. Before saving anything, call `recall`
with a brief summary to check for duplicates. Skip saving if a similar memory
already exists.

#### Tier 1 — Auto-save (save immediately, notify the user)
Save these automatically without asking. After saving, display a brief note:
`> Memorized: [one-line summary of what was saved]`

Signal phrases and situations:
- **Decisions with reasoning**: "decided to use X because Y", "we chose...",
  "the approach is...", "going with X over Y because..."
- **Bug fixes and solutions**: "the fix was...", "resolved by...",
  "the solution is...", "fixed it by..."
- **Error root causes**: "the issue was caused by...", "failed because...",
  "the error was due to...", "it broke because..."
- **Architecture choices**: "we'll structure it as...", "the design is...",
  "the architecture will be..."

Save with: importance 0.8-1.0, namespace by project or "decisions"/"errors",
descriptive tags for technologies and concepts involved. Include full context
and reasoning so future agents can understand without prior conversation.

#### Tier 2 — Ask first ("Save this? y/n")
Ask briefly before saving these:
- **General patterns and learnings**: "the trick is...", "pattern:",
  "this pattern works...", "always do X when..."
- **Preferences and conventions**: "we prefer...", "the team standard is...",
  "convention here is..."
- **Configuration discoveries**: "you need to set X to Y",
  "the config requires...", "important setting:..."
- **Workarounds and gotchas**: "watch out for...", "the workaround is...",
  "gotcha:...", "caveat:..."

Save with: importance 0.5-0.7, namespace "patterns" or by project,
descriptive tags.

#### Tier 3 — Never save
Do NOT save:
- Trivial observations, greetings, status updates
- Information that already exists in memory (always check with `recall` first)
- Speculative or unconfirmed information
- Temporary debugging steps or intermediate exploration

### Synthesizing Answers
When using `recall` or `hybrid_recall`, present results as natural knowledge:
- Integrate memories into your response conversationally
- Reference prior decisions: "You previously decided X because Y"
- Don't expose raw JSON or tool mechanics to the user

### Auto-Extract for Long Sessions
For significant problem-solving conversations (debugging sessions, architecture
discussions), offer:
"This session had good learnings. Extract key memories? y/n"
Then use `extract` to automatically capture important information.

### Tool Selection Guide
- `remember`: Store a single memory with full context
- `recall`: Semantic search for relevant memories
- `hybrid_recall`: Combined keyword + semantic search (better for specific terms)
- `extract`: Auto-extract memories from conversation text
- `nearby`: Find memories similar to a known memory
- `regions`: Discover topic clusters in memory space
- `journey`: Navigate conceptual path between two memories
```

## Implementation Location

- **File**: `spatial_memory/server.py`
- **Method**: `SpatialMemoryServer._get_server_instructions()` (~line 1066)
- **Parameter**: `instructions`

## Related Files

- `spatial_memory/server.py` - Server implementation
- `CLAUDE.md` - Project-level instructions (for contributors)
- `docs/MCP_SERVER_INSTRUCTIONS.md` - This document
