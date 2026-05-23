from __future__ import annotations

import pandas as pd

from trading_lab.indicators.rsi import relative_strength_index
from trading_lab.strategies.base import StrategyBase


class RSIMeanReversionStrategy(StrategyBase):
    name = "rsi_mean_reversion"

    def __init__(
        self,
        rsi_length: int = 14,
        buy_threshold: float = 30.0,
        sell_threshold: float = 55.0,
        max_holding_days: int = 10,
        trend_sma_window: int = 200,
    ) -> None:
        self.rsi_length = rsi_length
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_holding_days = max_holding_days
        self.trend_sma_window = trend_sma_window

    def parameters(self) -> dict:
        return {
            "rsi_length": self.rsi_length,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "max_holding_days": self.max_holding_days,
            "trend_sma_window": self.trend_sma_window,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["rsi"] = relative_strength_index(data["close"], self.rsi_length)
        data["trend_sma"] = data["close"].rolling(self.trend_sma_window).mean()
        data["sma_200"] = data["trend_sma"]
        data["entry_signal"] = (data["rsi"] < self.buy_threshold) & (data["close"] > data["trend_sma"])
        data["exit_signal"] = data["rsi"] > self.sell_threshold
        return data
