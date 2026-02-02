# MCP Server Instructions: Auto-Injected AI Behavioral Guidelines

## Discovery

When users install an MCP server, the server can provide **instructions** that are automatically injected into Claude's system prompt. This enables MCP servers to define behavioral guidelines that Claude follows without requiring users to manually configure anything.

## Technical Implementation

The MCP Python SDK's `Server` class accepts an `instructions` parameter:

```python
from mcp.server import Server

server = Server(
    name="spatial-memory",
    version="1.5.4",
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

You have access to a persistent semantic memory system. Use it proactively.

### Session Start
At conversation start, call `recall` with the user's apparent task/context to load relevant memories. Present insights naturally: "Based on previous work..." not "The database returned...".

### Recognizing Memory-Worthy Moments
After these events, ask "Save this? y/n" (minimal friction):
- Decisions: "Let's use X approach...", "We decided..."
- Solutions: "The fix was...", "It failed because..."
- Patterns: "This pattern works...", "The trick is..."
- Discoveries: "I found that...", "Important:..."

### Saving Memories
When confirmed, save with:
- **Detailed content**: Future agents need full context
- **Contextual namespace**: Project name, "decisions", "errors", etc.
- **Descriptive tags**: Technologies, concepts, patterns involved

### Synthesizing Answers
When recalling memories, present as natural knowledge:
- Good: "In previous sessions, you decided to use PostgreSQL for..."
- Bad: "Here are the query results: [{id: '...', content: '...'}]"

### Auto-Extract
For significant problem-solving conversations, offer to use `extract` to automatically capture key learnings.
```

## Implementation Location

- **File**: `spatial_memory/server.py`
- **Line**: Server initialization (~340)
- **Parameter**: `instructions`

## Related Files

- `spatial_memory/server.py` - Server implementation
- `CLAUDE.md` - Project-level instructions (for contributors)
- `docs/MCP_SERVER_INSTRUCTIONS.md` - This document
