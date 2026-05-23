from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_moving_average(series: pd.Series, length: int) -> pd.Series:
    """Return a linearly weighted moving average aligned to the input index."""
    if length <= 1:
        raise ValueError("length must be greater than 1")
    values = pd.Series(series, copy=False).astype(float)
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = float(weights.sum())
    return values.rolling(length, min_periods=length).apply(lambda window: float(np.dot(window, weights) / weight_sum), raw=True)
