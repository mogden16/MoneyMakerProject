from __future__ import annotations

import pandas as pd

from trading_lab.indicators.intraday_signals import add_intraday_common_features, build_opening_range_frame
from trading_lab.strategies.base import StrategyBase


class OpeningRangeBreakoutStrategy(StrategyBase):
    """Long-only SPY opening-range breakout with optional confirmation and SwingArm exits."""

    name = "opening_range_breakout"

    def __init__(
        self,
        breakout_buffer_pct: float = 0.0005,
        max_or_width_pct: float = 0.01,
        max_entry_time: str = "11:30",
        require_daily_regime: bool = True,
        use_volume_pressure: bool = True,
        volume_pressure_threshold: float = 0.0,
        qqe_state_mode: str = "off",
        use_swingarm_exit: bool = False,
        swing_lookback: int = 10,
        atr_length: int = 14,
        atr_multiplier: float = 2.5,
        neutral_exit_bars: int = 3,
        exit_on_or_failure: bool = True,
        end_of_day_exit: bool = True,
        allow_overnight: bool = False,
    ) -> None:
        self.breakout_buffer_pct = breakout_buffer_pct
        self.max_or_width_pct = max_or_width_pct
        self.max_entry_time = max_entry_time
        self.require_daily_regime = require_daily_regime
        self.use_volume_pressure = use_volume_pressure
        self.volume_pressure_threshold = volume_pressure_threshold
        self.qqe_state_mode = qqe_state_mode
        self.use_swingarm_exit = use_swingarm_exit
        self.swing_lookback = swing_lookback
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.neutral_exit_bars = neutral_exit_bars
        self.exit_on_or_failure = exit_on_or_failure
        self.end_of_day_exit = end_of_day_exit
        self.allow_overnight = allow_overnight

    def parameters(self) -> dict:
        return {
            "breakout_buffer_pct": self.breakout_buffer_pct,
            "max_or_width_pct": self.max_or_width_pct,
            "max_entry_time": self.max_entry_time,
            "require_daily_regime": self.require_daily_regime,
            "use_volume_pressure": self.use_volume_pressure,
            "volume_pressure_threshold": self.volume_pressure_threshold,
            "qqe_state_mode": self.qqe_state_mode,
            "use_swingarm_exit": self.use_swingarm_exit,
            "swing_lookback": self.swing_lookback,
            "atr_length": self.atr_length,
            "atr_multiplier": self.atr_multiplier,
            "neutral_exit_bars": self.neutral_exit_bars,
            "exit_on_or_failure": self.exit_on_or_failure,
            "end_of_day_exit": self.end_of_day_exit,
            "allow_overnight": self.allow_overnight,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data = add_intraday_common_features(
            data,
            atr_length=self.atr_length,
            swing_lookback=self.swing_lookback,
            atr_multiplier=self.atr_multiplier,
        )
        opening_range = build_opening_range_frame(
            data,
            breakout_buffer_pct=self.breakout_buffer_pct,
            max_or_width_pct=self.max_or_width_pct,
            max_entry_time=self.max_entry_time,
        )
        data = pd.concat([data, opening_range], axis=1)
        regime_ok = data.get("daily_regime_bull", pd.Series(True, index=data.index)).fillna(False)
        if not self.require_daily_regime:
            regime_ok = pd.Series(True, index=data.index)
        pressure_ok = data["pressure_z"] > self.volume_pressure_threshold if self.use_volume_pressure else pd.Series(True, index=data.index)
        qqe_ok = pd.Series(True, index=data.index)
        if self.qqe_state_mode == "long_only":
            qqe_ok = data["qqe_long_state"].fillna(False)
        elif self.qqe_state_mode == "long_or_neutral_positive":
            qqe_ok = data["qqe_long_state"].fillna(False) | (
                data["qqe_neutral_state"].fillna(False) & (data["rsi_smoothed"] >= 50.0)
            )
        data["entry_signal"] = (
            data["or_breakout"].fillna(False)
            & ~data["avoid_long_after_or_breakdown"].fillna(False)
            & regime_ok
            & pressure_ok
            & qqe_ok
        )
        exit_signal = ~regime_ok
        if self.exit_on_or_failure:
            exit_signal = exit_signal | (data["or_ready"].fillna(False) & (data["close"] < data["or_high"]))
        if self.use_swingarm_exit:
            exit_signal = exit_signal | (data["close"] < data["long_arm"])
        if self.qqe_state_mode != "off":
            exit_signal = exit_signal | data["qqe_short_state"].fillna(False)
            exit_signal = exit_signal | (data["qqe_neutral_consecutive"].fillna(0) >= self.neutral_exit_bars)
        data["exit_signal"] = exit_signal.fillna(False)
        return data

