"""Type stubs for sklearn.metrics."""

import numpy as np
from numpy.typing import NDArray

def silhouette_score(
    X: NDArray[np.floating],
    labels: NDArray[np.intp],
    metric: str = "euclidean",
    sample_size: int | None = None,
    random_state: int | None = None,
) -> float: ...
