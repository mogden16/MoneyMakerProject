from __future__ import annotations

import math

import pandas as pd

from trading_lab.indicators.moving_average import weighted_moving_average


def hull_moving_average(series: pd.Series, length: int) -> pd.Series:
    """Return the Hull Moving Average for a price series."""
    if length <= 1:
        raise ValueError("length must be greater than 1")
    values = pd.Series(series, copy=False).astype(float)
    half_length = max(length // 2, 1)
    sqrt_length = max(int(math.sqrt(length)), 1)
    wma_half = values if half_length == 1 else weighted_moving_average(values, half_length)
    wma_full = weighted_moving_average(values, length)
    raw_hma = (2.0 * wma_half) - wma_full
    return raw_hma if sqrt_length == 1 else weighted_moving_average(raw_hma, sqrt_length)
