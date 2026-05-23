from __future__ import annotations

import pandas as pd

from trading_lab.strategies.base import StrategyBase


class IntradayBreakoutStrategy(StrategyBase):
    """Intraday breakout entry gated by a completed daily trend filter."""

    name = "intraday_breakout"

    def __init__(
        self,
        breakout_lookback_bars: int = 12,
        exit_lookback_bars: int = 6,
        require_daily_regime: bool = True,
        end_of_day_exit: bool = True,
        allow_overnight: bool = False,
    ) -> None:
        self.breakout_lookback_bars = breakout_lookback_bars
        self.exit_lookback_bars = exit_lookback_bars
        self.require_daily_regime = require_daily_regime
        self.end_of_day_exit = end_of_day_exit
        self.allow_overnight = allow_overnight

    def parameters(self) -> dict:
        return {
            "breakout_lookback_bars": self.breakout_lookback_bars,
            "exit_lookback_bars": self.exit_lookback_bars,
            "require_daily_regime": self.require_daily_regime,
            "end_of_day_exit": self.end_of_day_exit,
            "allow_overnight": self.allow_overnight,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data["prior_high"] = data["high"].shift(1).rolling(self.breakout_lookback_bars).max()
        data["prior_low"] = data["low"].shift(1).rolling(self.exit_lookback_bars).min()
        regime_ok = data.get("daily_regime_bull", pd.Series(True, index=data.index)).fillna(False)
        data["entry_signal"] = (data["close"] > data["prior_high"]) & regime_ok
        if not self.require_daily_regime:
            data["entry_signal"] = data["close"] > data["prior_high"]
        data["exit_signal"] = (data["close"] < data["prior_low"]) | (~regime_ok)
        return data
