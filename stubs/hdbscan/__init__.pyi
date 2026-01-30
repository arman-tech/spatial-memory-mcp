"""Type stubs for hdbscan."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class HDBSCAN:
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        **kwargs: object,
    ) -> None: ...
    def fit_predict(self, X: NDArray[np.floating]) -> NDArray[np.intp]: ...
    def fit(self, X: NDArray[np.floating]) -> HDBSCAN: ...
    @property
    def labels_(self) -> NDArray[np.intp]: ...
    @property
    def probabilities_(self) -> NDArray[np.floating]: ...
