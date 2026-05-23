from __future__ import annotations

import pandas as pd

from trading_lab.strategies.base import StrategyBase


class MovingAverageCrossStrategy(StrategyBase):
    name = "moving_average_crossover"

    def __init__(self, fast_window: int = 20, slow_window: int = 50) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def parameters(self) -> dict:
        return {"fast_window": self.fast_window, "slow_window": self.slow_window}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["fast_ma"] = data["close"].rolling(self.fast_window).mean()
        data["slow_ma"] = data["close"].rolling(self.slow_window).mean()
        data["regime"] = (data["fast_ma"] > data["slow_ma"]).fillna(False)
        data["regime_prev"] = data["regime"].shift(1).fillna(False).astype(bool)
        data["entry_signal"] = data["regime"].astype(bool) & (~data["regime_prev"])
        data["exit_signal"] = (~data["regime"].astype(bool)) & data["regime_prev"]
        return data
