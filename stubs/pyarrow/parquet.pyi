"""Type stubs for pyarrow.parquet."""

from pathlib import Path
from typing import Any

import pyarrow

def write_table(
    table: pyarrow.Table,
    where: str | Path,
    compression: str = ...,
    **kwargs: Any,
) -> None: ...

def read_table(
    source: str | Path,
    **kwargs: Any,
) -> pyarrow.Table: ...
