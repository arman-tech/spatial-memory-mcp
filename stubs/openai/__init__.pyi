"""Type stubs for openai."""

from typing import Any

class EmbeddingData:
    embedding: list[float]

class EmbeddingResponse:
    data: list[EmbeddingData]

class Embeddings:
    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse: ...

class OpenAI:
    embeddings: Embeddings

    def __init__(
        self,
        api_key: str | None = ...,
        **kwargs: Any,
    ) -> None: ...
