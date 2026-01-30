# Stub for mcp.server module
from collections.abc import Callable
from typing import Any, TypeVar

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

F = TypeVar("F", bound=Callable[..., Any])

class InitializationOptions: ...

class Server:
    def __init__(self, name: str) -> None: ...

    def list_tools(self) -> Callable[[F], F]:
        """Decorator for registering a list_tools handler."""
        ...

    def call_tool(
        self, *, validate_input: bool = True
    ) -> Callable[[F], F]:
        """Decorator for registering a call_tool handler."""
        ...

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[Any],
        write_stream: MemoryObjectSendStream[Any],
        initialization_options: InitializationOptions,
        raise_exceptions: bool = False,
        stateless: bool = False,
    ) -> None: ...

    def create_initialization_options(self) -> InitializationOptions: ...
