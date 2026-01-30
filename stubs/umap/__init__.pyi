"""Type stubs for umap-learn."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class UMAP:
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int | None = None,
        **kwargs: object,
    ) -> None: ...
    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def fit(self, X: NDArray[np.floating]) -> UMAP: ...
    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]: ...
    @property
    def embedding_(self) -> NDArray[np.floating]: ...
