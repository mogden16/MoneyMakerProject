from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from trading_lab.indicators.hma import hull_moving_average
from trading_lab.indicators.qqe import qqe_indicator


def build_qqe_state_frame(
    close: pd.Series,
    *,
    rsi_length: int = 14,
    rsi_smoothing: int = 5,
    qqe_factor: float = 4.236,
    atr_smoothing: int = 14,
    neutral_band: float = 2.5,
    qqe_distance_band: float | None = None,
) -> pd.DataFrame:
    frame = qqe_indicator(
        close,
        rsi_length=rsi_length,
        rsi_smoothing=rsi_smoothing,
        qqe_factor=qqe_factor,
        atr_smoothing=atr_smoothing,
    ).copy()
    distance_band = neutral_band / 2 if qqe_distance_band is None else qqe_distance_band
    frame["qqe_distance"] = (frame["qqe_fast"] - frame["qqe_slow"]).abs()
    neutral = (
        (frame["rsi_smoothed"] - 50.0).abs() < neutral_band
    ) | (frame["qqe_distance"] < distance_band) | (frame["trend"] == 0)
    long_state = (frame["qqe_fast"] > frame["qqe_slow"]) & (frame["rsi_smoothed"] > 50.0) & (frame["trend"] >= 0)
    short_state = (frame["qqe_fast"] < frame["qqe_slow"]) & (frame["rsi_smoothed"] < 50.0) & (frame["trend"] <= 0)
    frame["qqe_long_state"] = long_state & ~neutral
    frame["qqe_short_state"] = short_state & ~neutral
    frame["qqe_neutral_state"] = neutral | (~frame["qqe_long_state"] & ~frame["qqe_short_state"])
    neutral_groups = (~frame["qqe_neutral_state"]).cumsum()
    frame["qqe_neutral_consecutive"] = frame["qqe_neutral_state"].groupby(neutral_groups).cumsum().astype(int)
    return frame


def build_volume_pressure_frame(
    bars: pd.DataFrame,
    *,
    pressure_length: int = 20,
) -> pd.DataFrame:
    frame = bars.copy()
    min_periods = max(2, min(pressure_length, max(5, pressure_length // 2)))
    range_size = (frame["high"] - frame["low"]).replace(0, np.nan)
    clv = ((2.0 * frame["close"] - frame["high"] - frame["low"]) / range_size).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volume_pressure = clv * frame["volume"].astype(float)
    rolling_mean = volume_pressure.rolling(pressure_length, min_periods=min_periods).mean()
    rolling_std = volume_pressure.rolling(pressure_length, min_periods=min_periods).std().replace(0, np.nan)
    pressure_z = ((volume_pressure - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score = pd.Series(50, index=frame.index, dtype=float)
    score = score.mask(pressure_z >= 1.5, 100.0)
    score = score.mask((pressure_z >= 1.0) & (pressure_z < 1.5), 75.0)
    score = score.mask((pressure_z < 0.0) & (pressure_z >= -1.0), 25.0)
    score = score.mask(pressure_z < -1.0, 0.0)
    return pd.DataFrame(
        {
            "clv": clv,
            "volume_pressure": volume_pressure,
            "pressure_z": pressure_z,
            "bullish_pressure": pressure_z > 1.0,
            "bearish_pressure": pressure_z < -1.0,
            "neutral_pressure": pressure_z.between(-1.0, 1.0, inclusive="both"),
            "pressure_score": score,
        },
        index=frame.index,
    )


def build_swingarm_atr_frame(
    bars: pd.DataFrame,
    *,
    atr_length: int = 14,
    swing_lookback: int = 10,
    atr_multiplier: float = 2.5,
    use_high_low_extremes: bool = False,
) -> pd.DataFrame:
    frame = bars.copy()
    prior_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prior_close).abs(),
            (frame["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_length, adjust=False, min_periods=atr_length).mean()
    high_source = frame["high"] if use_high_low_extremes else frame["close"]
    low_source = frame["low"] if use_high_low_extremes else frame["close"]
    highest = high_source.shift(1).rolling(swing_lookback, min_periods=swing_lookback).max()
    lowest = low_source.shift(1).rolling(swing_lookback, min_periods=swing_lookback).min()
    long_arm = highest - atr_multiplier * atr
    short_arm = lowest + atr_multiplier * atr
    return pd.DataFrame(
        {
            "atr": atr,
            "swing_highest": highest,
            "swing_lowest": lowest,
            "long_arm": long_arm,
            "short_arm": short_arm,
            "long_arm_rising": long_arm.diff().fillna(0.0) >= 0.0,
            "short_arm_falling": short_arm.diff().fillna(0.0) <= 0.0,
            "swingarm_bullish": (frame["close"] > long_arm) & (long_arm.diff().fillna(0.0) >= 0.0),
            "swingarm_bearish": (frame["close"] < short_arm) & (short_arm.diff().fillna(0.0) <= 0.0),
            "swingarm_neutral": (frame["close"] <= long_arm) & (frame["close"] >= short_arm),
        },
        index=frame.index,
    )


def build_opening_range_frame(
    bars: pd.DataFrame,
    *,
    opening_range_end: str = "10:00",
    entry_after: str = "10:00",
    max_entry_time: str = "11:30",
    breakout_buffer_pct: float = 0.0005,
    max_or_width_pct: float = 0.01,
) -> pd.DataFrame:
    frame = bars.copy()
    timestamps = pd.to_datetime(frame["timestamp"])
    session_dates = pd.to_datetime(frame["session_date"]).dt.date
    opening_range_end_time = _parse_time(opening_range_end)
    entry_after_time = _parse_time(entry_after)
    max_entry_clock = _parse_time(max_entry_time)
    rows: list[dict[str, float | bool]] = []
    for session_date, session_frame in frame.groupby(session_dates):
        session_ts = pd.to_datetime(session_frame["timestamp"])
        session_times = session_ts.dt.time
        or_mask = (session_times >= time(9, 30)) & (session_times < opening_range_end_time)
        if not or_mask.any():
            session_rows = pd.DataFrame(index=session_frame.index)
            session_rows["or_high"] = np.nan
            session_rows["or_low"] = np.nan
            session_rows["or_mid"] = np.nan
            session_rows["or_width"] = np.nan
            session_rows["or_width_pct"] = np.nan
            session_rows["or_ready"] = False
            session_rows["or_breakout"] = False
            session_rows["avoid_long_after_or_breakdown"] = False
            rows.append(session_rows)
            continue
        or_high = float(session_frame.loc[or_mask, "high"].max())
        or_low = float(session_frame.loc[or_mask, "low"].min())
        or_mid = (or_high + or_low) / 2.0
        or_width = or_high - or_low
        or_width_pct = (or_width / or_mid) if or_mid else np.nan
        prior_close = session_frame["close"].shift(1)
        breakout_level = or_high * (1.0 + breakout_buffer_pct)
        breakdown_seen = (
            (session_frame["close"] < or_low) & (session_times >= entry_after_time)
        ).shift(1, fill_value=False).cummax()
        session_rows = pd.DataFrame(index=session_frame.index)
        session_rows["or_high"] = or_high
        session_rows["or_low"] = or_low
        session_rows["or_mid"] = or_mid
        session_rows["or_width"] = or_width
        session_rows["or_width_pct"] = or_width_pct
        session_rows["or_ready"] = session_times >= entry_after_time
        session_rows["or_breakout"] = (
            (session_times >= entry_after_time)
            & (session_times < max_entry_clock)
            & (session_frame["close"] > breakout_level)
            & (prior_close <= or_high)
            & pd.Series(or_width_pct, index=session_frame.index).le(max_or_width_pct)
        )
        session_rows["avoid_long_after_or_breakdown"] = breakdown_seen.astype(bool)
        rows.append(session_rows)
    result = pd.concat(rows).reindex(frame.index)
    return result.sort_index()


def add_intraday_common_features(
    bars: pd.DataFrame,
    *,
    hma_length: int = 21,
    pressure_length: int = 20,
    atr_length: int = 14,
    swing_lookback: int = 10,
    atr_multiplier: float = 2.5,
    rsi_length: int = 14,
    rsi_smoothing: int = 5,
    qqe_factor: float = 4.236,
    qqe_atr_smoothing: int = 14,
    neutral_band: float = 2.5,
) -> pd.DataFrame:
    frame = bars.copy()
    frame["hma"] = hull_moving_average(frame["close"], hma_length)
    frame["hma_slope_positive"] = frame["hma"].diff().fillna(0.0) > 0.0
    volume_pressure = build_volume_pressure_frame(frame, pressure_length=pressure_length)
    swingarm = build_swingarm_atr_frame(
        frame,
        atr_length=atr_length,
        swing_lookback=swing_lookback,
        atr_multiplier=atr_multiplier,
    )
    qqe_states = build_qqe_state_frame(
        frame["close"],
        rsi_length=rsi_length,
        rsi_smoothing=rsi_smoothing,
        qqe_factor=qqe_factor,
        atr_smoothing=qqe_atr_smoothing,
        neutral_band=neutral_band,
    )
    return pd.concat([frame, volume_pressure, swingarm, qqe_states], axis=1)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))
