"""Type stubs for lancedb.table."""

from datetime import timedelta
from typing import Any, Iterable, Union

import lancedb


class MergeInsertResult:
    """Result of a merge insert operation."""

    version: int
    num_updated_rows: int
    num_inserted_rows: int
    num_deleted_rows: int
    num_attempts: int


class LanceMergeInsertBuilder:
    """Builder for merge insert operations."""

    def when_matched_update_all(
        self, *, where: str | None = ...
    ) -> LanceMergeInsertBuilder: ...
    def when_not_matched_insert_all(self) -> LanceMergeInsertBuilder: ...
    def when_not_matched_by_source_delete(
        self, condition: str | None = ...
    ) -> LanceMergeInsertBuilder: ...
    def execute(
        self,
        new_data: Any,
        on_bad_vectors: str = ...,
        fill_value: float = ...,
        timeout: timedelta | None = ...,
    ) -> MergeInsertResult: ...


class Table:
    # Properties
    @property
    def version(self) -> int: ...

    def add(self, data: list[dict[str, Any]]) -> None: ...
    def search(self, query: Any = ..., query_type: str = ...) -> lancedb.LanceQueryBuilder: ...
    def delete(self, predicate: str) -> None: ...
    def count_rows(self, predicate: str = ...) -> int: ...
    # Merge insert for atomic upsert operations
    def merge_insert(self, on: Union[str, Iterable[str]]) -> LanceMergeInsertBuilder: ...
    # create_index accepts many different parameter combinations
    # Using permissive signature to allow both positional column and kwargs-only patterns
    def create_index(
        self,
        column: str | None = ...,
        config: Any = ...,
        index_type: str = ...,
        metric: str = ...,
        num_partitions: int = ...,
        num_sub_vectors: int = ...,
        vector_column_name: str = ...,
        replace: bool = ...,
        sample_rate: int = ...,
        m: int = ...,
        ef_construction: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def create_scalar_index(
        self,
        column: str,
        index_type: str = ...,
        replace: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def create_fts_index(
        self,
        column: str,
        use_tantivy: bool = ...,
        with_position: bool = ...,
        language: str = ...,
        stem: bool = ...,
        remove_stop_words: bool = ...,
        lower_case: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def list_indices(self) -> list[dict[str, Any]]: ...
    def compact_files(self) -> None: ...
    def optimize(self) -> None: ...
    def stats(self) -> dict[str, Any] | Any: ...
    # Versioning / Snapshots
    def restore(self, version: int) -> None: ...
    def list_versions(self) -> list[Any]: ...
