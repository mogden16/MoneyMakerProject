from __future__ import annotations

import pandas as pd

from trading_lab.strategies.base import StrategyBase


class TrendFilterStrategy(StrategyBase):
    """Long-only trend filter that stays invested while price remains above its SMA."""

    name = "trend_filter_strategy"

    def __init__(self, sma_length: int = 200) -> None:
        if sma_length <= 1:
            raise ValueError("sma_length must be greater than 1")
        self.sma_length = sma_length

    def parameters(self) -> dict:
        return {"sma_length": self.sma_length}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["trend_sma"] = data["close"].rolling(self.sma_length).mean()
        data["regime"] = (data["close"] > data["trend_sma"]).fillna(False)
        data["regime_prev"] = data["regime"].shift(1).fillna(False).astype(bool)
        data["entry_signal"] = data["regime"].astype(bool) & (~data["regime_prev"])
        data["exit_signal"] = (~data["regime"].astype(bool)) & data["regime_prev"]
        return data
