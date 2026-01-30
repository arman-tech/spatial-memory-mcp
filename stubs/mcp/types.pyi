# Stub for mcp.types module
from typing import Any

class Tool:
    name: str
    description: str
    inputSchema: dict[str, Any]

    def __init__(
        self,
        name: str,
        description: str,
        inputSchema: dict[str, Any],
    ) -> None: ...

class TextContent:
    type: str
    text: str

    def __init__(self, type: str, text: str) -> None: ...

class ListToolsRequest: ...

class ListToolsResult:
    tools: list[Tool]

class CallToolResult: ...
