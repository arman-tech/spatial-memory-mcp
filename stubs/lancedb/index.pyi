"""Type stubs for lancedb.index."""

from typing import Any

class IvfPq:
    def __init__(
        self,
        num_partitions: int = ...,
        num_sub_vectors: int = ...,
        distance_type: str = ...,
        **kwargs: Any,
    ) -> None: ...

class BTree:
    def __init__(self, **kwargs: Any) -> None: ...

class Bitmap:
    def __init__(self, **kwargs: Any) -> None: ...

class LabelList:
    def __init__(self, **kwargs: Any) -> None: ...
