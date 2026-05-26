from __future__ import annotations

import pandas as pd

from trading_lab.indicators.intraday_signals import build_swingarm_atr_frame
from trading_lab.strategies.base import StrategyBase


class SwingArmTrendStrategy(StrategyBase):
    """Long-only SPY trend entry using a volatility-adjusted SwingArm line."""

    name = "swingarm_trend"

    def __init__(
        self,
        atr_length: int = 14,
        swing_lookback: int = 10,
        atr_multiplier: float = 2.5,
        require_daily_regime: bool = True,
        end_of_day_exit: bool = True,
        allow_overnight: bool = False,
    ) -> None:
        self.atr_length = atr_length
        self.swing_lookback = swing_lookback
        self.atr_multiplier = atr_multiplier
        self.require_daily_regime = require_daily_regime
        self.end_of_day_exit = end_of_day_exit
        self.allow_overnight = allow_overnight

    def parameters(self) -> dict:
        return {
            "atr_length": self.atr_length,
            "swing_lookback": self.swing_lookback,
            "atr_multiplier": self.atr_multiplier,
            "require_daily_regime": self.require_daily_regime,
            "end_of_day_exit": self.end_of_day_exit,
            "allow_overnight": self.allow_overnight,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        swing = build_swingarm_atr_frame(
            data,
            atr_length=self.atr_length,
            swing_lookback=self.swing_lookback,
            atr_multiplier=self.atr_multiplier,
        )
        data = pd.concat([data, swing], axis=1)
        regime_ok = data.get("daily_regime_bull", pd.Series(True, index=data.index)).fillna(False)
        if not self.require_daily_regime:
            regime_ok = pd.Series(True, index=data.index)
        cross_above = (data["close"] > data["long_arm"]) & (data["close"].shift(1) <= data["long_arm"].shift(1))
        cross_below = (data["close"] < data["long_arm"]) & (data["close"].shift(1) >= data["long_arm"].shift(1))
        data["entry_signal"] = cross_above & regime_ok & data["long_arm_rising"].fillna(False)
        data["exit_signal"] = (cross_below | (~regime_ok)).fillna(False)
        return data
