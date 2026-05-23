from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.indicators.rsi import relative_strength_index


def qqe_indicator(
    series: pd.Series,
    *,
    rsi_length: int = 14,
    rsi_smoothing: int = 5,
    qqe_factor: float = 4.236,
    atr_smoothing: int = 5,
) -> pd.DataFrame:
    """Return a transparent QQE-style indicator frame for research use.

    This is inspired by the legacy repo's QQE usage, but it is intentionally rebuilt
    as a documented research indicator rather than a literal port of old intraday logic.
    """
    if rsi_length <= 1:
        raise ValueError("rsi_length must be greater than 1")
    if rsi_smoothing <= 0:
        raise ValueError("rsi_smoothing must be positive")
    if atr_smoothing <= 0:
        raise ValueError("atr_smoothing must be positive")
    if qqe_factor <= 0:
        raise ValueError("qqe_factor must be positive")

    values = pd.Series(series, copy=False).astype(float)
    rsi = relative_strength_index(values, length=rsi_length)
    rsi_smoothed = rsi.ewm(alpha=1 / rsi_smoothing, adjust=False, min_periods=rsi_smoothing).mean()
    rsi_delta = rsi_smoothed.diff().abs()
    rsi_atr = rsi_delta.ewm(alpha=1 / atr_smoothing, adjust=False, min_periods=atr_smoothing).mean()
    rsi_atr_smoothed = rsi_atr.ewm(alpha=1 / atr_smoothing, adjust=False, min_periods=atr_smoothing).mean()
    band_distance = rsi_atr_smoothed * qqe_factor
    upper_band = rsi_smoothed + band_distance
    lower_band = rsi_smoothed - band_distance

    trailing_band = pd.Series(np.nan, index=values.index, dtype=float)
    trend = pd.Series(0, index=values.index, dtype=int)

    for idx in range(len(values)):
        if idx == 0 or pd.isna(rsi_smoothed.iloc[idx]) or pd.isna(band_distance.iloc[idx]):
            continue
        prev_trailing = trailing_band.iloc[idx - 1]
        prev_trend = trend.iloc[idx - 1]
        current_rsi = rsi_smoothed.iloc[idx]
        candidate_upper = upper_band.iloc[idx]
        candidate_lower = lower_band.iloc[idx]

        if prev_trend >= 0:
            next_band = candidate_lower if pd.isna(prev_trailing) else max(candidate_lower, prev_trailing)
            if current_rsi < next_band:
                trend.iloc[idx] = -1
                trailing_band.iloc[idx] = candidate_upper
            else:
                trend.iloc[idx] = 1
                trailing_band.iloc[idx] = next_band
        else:
            next_band = candidate_upper if pd.isna(prev_trailing) else min(candidate_upper, prev_trailing)
            if current_rsi > next_band:
                trend.iloc[idx] = 1
                trailing_band.iloc[idx] = candidate_lower
            else:
                trend.iloc[idx] = -1
                trailing_band.iloc[idx] = next_band

    trend = trend.where(rsi_smoothed.notna(), 0)
    signal = trend.diff().fillna(0).clip(-1, 1).astype(int)
    frame = pd.DataFrame(
        {
            "rsi": rsi,
            "rsi_smoothed": rsi_smoothed,
            "rsi_atr": rsi_atr_smoothed,
            "qqe_fast": rsi_smoothed,
            "qqe_slow": trailing_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "trend": trend.astype(int),
            "signal": signal,
        },
        index=values.index,
    )
    return frame
