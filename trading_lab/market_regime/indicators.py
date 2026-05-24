from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_lab.indicators.rsi import relative_strength_index


def extract_close_series(frame: pd.DataFrame) -> pd.Series:
    price_column = "adj_close" if "adj_close" in frame.columns and frame["adj_close"].notna().any() else "close"
    series = pd.Series(frame[price_column], index=pd.to_datetime(frame["timestamp"]), copy=False).astype(float)
    return series.sort_index()


def compute_period_return(series: pd.Series, periods: int) -> float | None:
    values = pd.Series(series, copy=False).dropna()
    if len(values) <= periods:
        return None
    start_value = float(values.iloc[-periods - 1])
    if start_value == 0.0:
        return None
    return float(values.iloc[-1] / start_value - 1.0)


def compute_moving_average_distance(series: pd.Series, window: int) -> float | None:
    values = pd.Series(series, copy=False).dropna()
    if len(values) < window:
        return None
    moving_average = values.rolling(window).mean().iloc[-1]
    if pd.isna(moving_average) or moving_average == 0.0:
        return None
    return float(values.iloc[-1] / moving_average - 1.0)


def compute_latest_rsi(series: pd.Series, length: int = 14) -> float | None:
    values = pd.Series(series, copy=False).dropna()
    if len(values) <= length:
        return None
    rsi = relative_strength_index(values, length=length).dropna()
    if rsi.empty:
        return None
    return float(rsi.iloc[-1])


def compute_realized_volatility(series: pd.Series, window: int = 20) -> float | None:
    values = pd.Series(series, copy=False).dropna()
    returns = values.pct_change().dropna()
    if len(returns) < window:
        return None
    trailing = returns.tail(window)
    return float(trailing.std(ddof=0) * np.sqrt(252))


def build_index_momentum_rows(price_history_by_symbol: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in ["SPY", "QQQ", "IWM", "SOXX"]:
        history = price_history_by_symbol.get(symbol)
        if history is None or history.empty:
            continue
        close_series = extract_close_series(history)
        distance_200 = compute_moving_average_distance(close_series, 200)
        row = {
            "symbol": symbol,
            "last_price": float(close_series.iloc[-1]),
            "return_5d": compute_period_return(close_series, 5),
            "return_20d": compute_period_return(close_series, 20),
            "distance_50dma": compute_moving_average_distance(close_series, 50),
            "distance_200dma": distance_200,
            "rsi_14": compute_latest_rsi(close_series, 14),
            "realized_vol_20d": compute_realized_volatility(close_series, 20),
            "overextended_flag": bool(symbol == "SOXX" and distance_200 is not None and distance_200 > 0.32),
        }
        rows.append(row)
    return rows


def build_sector_leadership_rows(price_history_by_symbol: dict[str, pd.DataFrame], benchmark_symbol: str = "SPY") -> list[dict[str, Any]]:
    benchmark_history = price_history_by_symbol.get(benchmark_symbol)
    benchmark_return_20 = None
    if benchmark_history is not None and not benchmark_history.empty:
        benchmark_return_20 = compute_period_return(extract_close_series(benchmark_history), 20)

    rows: list[dict[str, Any]] = []
    for symbol in ["XLK", "XLF", "XLE", "XLU", "XLV", "XLP", "XLY", "XLI", "XLC"]:
        history = price_history_by_symbol.get(symbol)
        if history is None or history.empty:
            continue
        close_series = extract_close_series(history)
        return_20 = compute_period_return(close_series, 20)
        rows.append(
            {
                "symbol": symbol,
                "return_1w": compute_period_return(close_series, 5),
                "return_1m": return_20,
                "trend_vs_50dma": "Above" if (compute_moving_average_distance(close_series, 50) or 0.0) >= 0 else "Below",
                "distance_50dma": compute_moving_average_distance(close_series, 50),
                "relative_strength_vs_spy": None if return_20 is None or benchmark_return_20 is None else float(return_20 - benchmark_return_20),
            }
        )
    return rows
