from __future__ import annotations

from typing import Any

import pandas as pd

from trading_lab.indicators.rsi import relative_strength_index


def average_true_range(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length, min_periods=length).mean()


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable, time-safe features from one SPY bar frame."""
    frame = bars.copy().sort_values("timestamp").reset_index(drop=True)
    close = frame["close"]
    frame["return_1d"] = close.pct_change(1)
    frame["return_5d"] = close.pct_change(5)
    frame["return_10d"] = close.pct_change(10)
    frame["return_20d"] = close.pct_change(20)
    frame["realized_vol_10d"] = frame["return_1d"].rolling(10, min_periods=10).std()
    frame["realized_vol_20d"] = frame["return_1d"].rolling(20, min_periods=20).std()
    frame["rsi_14"] = relative_strength_index(close, 14)
    frame["sma_20"] = close.rolling(20, min_periods=20).mean()
    frame["sma_50"] = close.rolling(50, min_periods=50).mean()
    frame["sma_200"] = close.rolling(200, min_periods=200).mean()
    frame["sma_distance_20"] = (close / frame["sma_20"]) - 1
    frame["sma_distance_50"] = (close / frame["sma_50"]) - 1
    frame["sma_distance_200"] = (close / frame["sma_200"]) - 1
    frame["close_above_200_sma"] = (close > frame["sma_200"]).astype(int)
    frame["atr_14"] = average_true_range(frame, 14)
    frame["atr_pct"] = frame["atr_14"] / close
    atr_rank = frame["atr_pct"].rolling(60, min_periods=20)
    frame["atr_percentile"] = atr_rank.apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)
    frame["drawdown_from_20d_high"] = close / close.rolling(20, min_periods=20).max() - 1
    frame["drawdown_from_50d_high"] = close / close.rolling(50, min_periods=50).max() - 1
    timestamp = pd.to_datetime(frame["timestamp"])
    frame["day_of_week"] = timestamp.dt.dayofweek
    frame["month"] = timestamp.dt.month
    frame["session_year"] = timestamp.dt.year
    return frame


def extract_signal_feature_row(
    feature_frame: pd.DataFrame,
    signal_index: int,
    *,
    strategy_name: str,
    exit_structure_name: str,
    timeframe: str,
    entry_parameters: dict[str, Any],
    exit_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Extract one signal-row feature payload with only information known at signal time."""
    row = feature_frame.iloc[signal_index]
    payload: dict[str, Any] = {
        "timestamp": pd.Timestamp(row["timestamp"]),
        "session_date": row["session_date"],
        "symbol": row["symbol"],
        "timeframe": timeframe,
        "entry_strategy_name": strategy_name,
        "entry_parameters_json": entry_parameters,
        "exit_structure_name": exit_structure_name,
        "exit_parameters_json": exit_parameters,
        "signal_price": float(row["close"]),
        "close": float(row["close"]),
        "return_1d": float(row["return_1d"]) if pd.notna(row["return_1d"]) else 0.0,
        "return_5d": float(row["return_5d"]) if pd.notna(row["return_5d"]) else 0.0,
        "return_10d": float(row["return_10d"]) if pd.notna(row["return_10d"]) else 0.0,
        "return_20d": float(row["return_20d"]) if pd.notna(row["return_20d"]) else 0.0,
        "realized_vol_10d": float(row["realized_vol_10d"]) if pd.notna(row["realized_vol_10d"]) else 0.0,
        "realized_vol_20d": float(row["realized_vol_20d"]) if pd.notna(row["realized_vol_20d"]) else 0.0,
        "rsi_14": float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else 50.0,
        "sma_distance_20": float(row["sma_distance_20"]) if pd.notna(row["sma_distance_20"]) else 0.0,
        "sma_distance_50": float(row["sma_distance_50"]) if pd.notna(row["sma_distance_50"]) else 0.0,
        "sma_distance_200": float(row["sma_distance_200"]) if pd.notna(row["sma_distance_200"]) else 0.0,
        "close_above_200_sma": int(row["close_above_200_sma"]) if pd.notna(row["close_above_200_sma"]) else 0,
        "atr_14": float(row["atr_14"]) if pd.notna(row["atr_14"]) else 0.0,
        "atr_pct": float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else 0.0,
        "atr_percentile": float(row["atr_percentile"]) if pd.notna(row["atr_percentile"]) else 0.0,
        "drawdown_from_20d_high": float(row["drawdown_from_20d_high"]) if pd.notna(row["drawdown_from_20d_high"]) else 0.0,
        "drawdown_from_50d_high": float(row["drawdown_from_50d_high"]) if pd.notna(row["drawdown_from_50d_high"]) else 0.0,
        "day_of_week": int(row["day_of_week"]),
        "month": int(row["month"]),
    }
    for column in ["daily_regime_bull", "intraday_ma", "trend_sma", "prior_high", "prior_low", "hma", "qqe_fast", "qqe_slow", "rsi"]:
        if column in feature_frame.columns:
            value = row[column]
            payload[column] = float(value) if pd.notna(value) and isinstance(value, (int, float)) else (int(value) if isinstance(value, bool) else value)
    return payload
