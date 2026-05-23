from __future__ import annotations

import pandas as pd

from trading_lab.indicators.rsi import relative_strength_index
from trading_lab.strategies.base import StrategyBase


class IntradayPullbackStrategy(StrategyBase):
    """Intraday SPY pullback entry gated by a daily trend regime."""

    name = "intraday_pullback"

    def __init__(
        self,
        rsi_length: int = 14,
        oversold_threshold: float = 35.0,
        recovery_threshold: float = 45.0,
        moving_average_length: int = 8,
        pullback_lookback_bars: int = 4,
        require_daily_regime: bool = True,
        end_of_day_exit: bool = True,
        allow_overnight: bool = False,
    ) -> None:
        self.rsi_length = rsi_length
        self.oversold_threshold = oversold_threshold
        self.recovery_threshold = recovery_threshold
        self.moving_average_length = moving_average_length
        self.pullback_lookback_bars = pullback_lookback_bars
        self.require_daily_regime = require_daily_regime
        self.end_of_day_exit = end_of_day_exit
        self.allow_overnight = allow_overnight

    def parameters(self) -> dict:
        return {
            "rsi_length": self.rsi_length,
            "oversold_threshold": self.oversold_threshold,
            "recovery_threshold": self.recovery_threshold,
            "moving_average_length": self.moving_average_length,
            "pullback_lookback_bars": self.pullback_lookback_bars,
            "require_daily_regime": self.require_daily_regime,
            "end_of_day_exit": self.end_of_day_exit,
            "allow_overnight": self.allow_overnight,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["rsi"] = relative_strength_index(data["close"], self.rsi_length)
        data["intraday_ma"] = data["close"].rolling(self.moving_average_length).mean()
        recent_oversold = data["rsi"].rolling(self.pullback_lookback_bars).min() < self.oversold_threshold
        regime_ok = data.get("daily_regime_bull", pd.Series(True, index=data.index)).fillna(False)
        recovery_cross = (data["rsi"] >= self.recovery_threshold) & (data["rsi"].shift(1) < self.recovery_threshold)
        ma_reclaim = (data["close"] > data["intraday_ma"]) & (data["close"].shift(1) <= data["intraday_ma"].shift(1))
        data["entry_signal"] = recent_oversold & recovery_cross & ma_reclaim & regime_ok
        if not self.require_daily_regime:
            data["entry_signal"] = recent_oversold & recovery_cross & ma_reclaim
        data["exit_signal"] = (data["close"] < data["intraday_ma"]) | (~regime_ok)
        return data
