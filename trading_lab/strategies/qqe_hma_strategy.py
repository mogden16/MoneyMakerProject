from __future__ import annotations

import pandas as pd

from trading_lab.indicators.hma import hull_moving_average
from trading_lab.indicators.qqe import qqe_indicator
from trading_lab.strategies.base import StrategyBase


class QQEHMAStrategy(StrategyBase):
    """Daily research-only strategy inspired by legacy QQE/HMA ideas."""

    name = "qqe_hma_strategy"

    def __init__(
        self,
        hma_length: int = 21,
        rsi_length: int = 14,
        rsi_smoothing: int = 5,
        qqe_factor: float = 4.236,
        atr_smoothing: int = 5,
        require_hma_slope: bool = True,
        exit_on_hma_break: bool = True,
        exit_on_qqe_bearish: bool = True,
    ) -> None:
        if hma_length <= 1:
            raise ValueError("hma_length must be greater than 1")
        self.hma_length = hma_length
        self.rsi_length = rsi_length
        self.rsi_smoothing = rsi_smoothing
        self.qqe_factor = qqe_factor
        self.atr_smoothing = atr_smoothing
        self.require_hma_slope = require_hma_slope
        self.exit_on_hma_break = exit_on_hma_break
        self.exit_on_qqe_bearish = exit_on_qqe_bearish

    def parameters(self) -> dict:
        return {
            "hma_length": self.hma_length,
            "rsi_length": self.rsi_length,
            "rsi_smoothing": self.rsi_smoothing,
            "qqe_factor": self.qqe_factor,
            "atr_smoothing": self.atr_smoothing,
            "require_hma_slope": self.require_hma_slope,
            "exit_on_hma_break": self.exit_on_hma_break,
            "exit_on_qqe_bearish": self.exit_on_qqe_bearish,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        qqe = qqe_indicator(
            data["close"],
            rsi_length=self.rsi_length,
            rsi_smoothing=self.rsi_smoothing,
            qqe_factor=self.qqe_factor,
            atr_smoothing=self.atr_smoothing,
        )
        data["hma"] = hull_moving_average(data["close"], self.hma_length)
        data["hma_slope"] = data["hma"].diff()
        data = pd.concat([data, qqe.reset_index(drop=True)], axis=1)

        bullish_turn = (data["trend"] == 1) & (data["trend"].shift(1).fillna(0) <= 0)
        bearish_turn = (data["trend"] == -1) & (data["trend"].shift(1).fillna(0) >= 0)
        price_above_hma = data["close"] > data["hma"]
        hma_slope_positive = data["hma_slope"] > 0

        entry_mask = bullish_turn & price_above_hma
        if self.require_hma_slope:
            entry_mask &= hma_slope_positive

        exit_mask = pd.Series(False, index=data.index)
        if self.exit_on_qqe_bearish:
            exit_mask |= bearish_turn
        if self.exit_on_hma_break:
            exit_mask |= data["close"] < data["hma"]

        valid_indicator_window = data[["hma", "trend"]].notna().all(axis=1)
        data["entry_signal"] = (entry_mask & valid_indicator_window).fillna(False)
        data["exit_signal"] = (exit_mask & valid_indicator_window).fillna(False)
        return data
