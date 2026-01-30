# Stub for mcp.server.stdio module
from typing import Any

from anyio.abc import AsyncResource

class StdioServerContextManager(AsyncResource):
    async def __aenter__(self) -> tuple[Any, Any]: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None: ...

def stdio_server(
    stdin: Any = None,
    stdout: Any = None,
) -> StdioServerContextManager:
    """Context manager that provides (read_stream, write_stream) for stdio transport."""
    ...
