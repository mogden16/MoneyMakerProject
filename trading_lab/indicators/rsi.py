from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength_index(series: pd.Series, length: int = 14) -> pd.Series:
    """Return Wilder-style RSI for a price series."""
    if length <= 1:
        raise ValueError("length must be greater than 1")
    values = pd.Series(series, copy=False).astype(float)
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    flat_mask = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    loss_zero_mask = avg_loss.eq(0.0) & avg_gain.gt(0.0)
    rsi = rsi.mask(flat_mask, 50.0)
    rsi = rsi.mask(loss_zero_mask, 100.0)
    return rsi
