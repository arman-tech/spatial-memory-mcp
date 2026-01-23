# Claude Memory System

This project has a persistent memory system. On EVERY conversation start, call `start_session()` to load context.

## Automatic Behaviors

1. **Session Start**: Always call `start_session()` first. This loads relevant context.

2. **Memory Recognition**: When you recognize memory-worthy moments, ask briefly:
   - "Save this decision?" (for architectural choices)
   - "Save this fix?" (for error solutions)
   - "Note this pattern?" (for reusable approaches)

   Wait for "y" or similar confirmation, then save.

3. **Memory Queries**: When asked about past decisions or context, use `search_memories` or `get_context`. Don't guess or hallucinate.

4. **Session End**: If significant learnings occurred, offer: "Save session summary?"

## Memory-Worthy Triggers

Recognize these in conversation:
- "Let's go with...", "We decided...", "The approach is..." → Decision
- "The fix was...", "It was failing because..." → Error
- "This pattern works...", "The trick is..." → Pattern
- "Remember that...", "Important:..." → Explicit save request
- "?" in a note-to-self context → Question/TODO

## Tool Availability

If memory tools fail (MCP server unavailable), continue helping but note:
"Memory system is temporarily unavailable. I'll help without historical context."

Don't pretend to remember things you can't access.
