from __future__ import annotations

import pandas as pd

from trading_lab.strategies.base import StrategyBase


class BreakoutStrategy(StrategyBase):
    name = "daily_breakout"

    def __init__(self, lookback_window: int = 20, exit_lookback_window: int | None = None) -> None:
        self.lookback_window = lookback_window
        self.exit_lookback_window = exit_lookback_window or lookback_window

    def parameters(self) -> dict:
        return {"lookback_window": self.lookback_window, "exit_lookback_window": self.exit_lookback_window}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["prior_high"] = data["high"].shift(1).rolling(self.lookback_window).max()
        data["prior_low"] = data["low"].shift(1).rolling(self.exit_lookback_window).min()
        data["entry_signal"] = data["close"] > data["prior_high"]
        data["exit_signal"] = data["close"] < data["prior_low"]
        return data
