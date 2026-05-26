from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pybroker import YFinance

from trading_lab.pybroker_lab.config import PyBrokerLabConfig


def normalize_pybroker_data(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    data = frame.copy()
    if "timestamp" in data.columns and "date" not in data.columns:
        data["date"] = pd.to_datetime(data["timestamp"])
    elif "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    else:
        raise ValueError("Price data must include either a 'date' or 'timestamp' column.")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in data.columns:
            raise ValueError(f"Price data is missing required column {column!r}.")
    data["symbol"] = data["symbol"].astype(str)
    normalized = data[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    normalized = normalized.sort_values(["symbol", "date"]).reset_index(drop=True)
    return normalized


def load_market_data(
    config: PyBrokerLabConfig,
    *,
    data_frame: pd.DataFrame | None = None,
    extra_symbols: Iterable[str] = (),
) -> pd.DataFrame:
    symbols = tuple(dict.fromkeys([*config.symbols, config.benchmark_symbol, *extra_symbols]))
    if data_frame is not None:
        normalized = normalize_pybroker_data(data_frame)
        filtered = normalized[
            normalized["symbol"].isin(symbols)
            & (normalized["date"] >= pd.Timestamp(config.start_date))
            & (normalized["date"] <= pd.Timestamp(config.end_date))
        ].copy()
        if filtered.empty:
            raise ValueError("No rows remained after filtering the provided price data.")
        return filtered.reset_index(drop=True)
    source = YFinance(auto_adjust=True)
    queried = source.query(symbols, config.start_date, config.end_date, _timeframe=config.timeframe)
    return normalize_pybroker_data(queried)


def symbol_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    bars = frame[frame["symbol"] == symbol].copy()
    return bars.sort_values("date").reset_index(drop=True)
