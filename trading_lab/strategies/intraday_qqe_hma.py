from __future__ import annotations

import pandas as pd

from trading_lab.indicators.intraday_signals import add_intraday_common_features
from trading_lab.strategies.base import StrategyBase


class IntradayQQEHMAStateStrategy(StrategyBase):
    """Long-only intraday QQE/HMA state strategy with optional SwingArm exit."""

    name = "intraday_qqe_hma"

    def __init__(
        self,
        hma_length: int = 21,
        rsi_length: int = 14,
        rsi_smoothing: int = 5,
        qqe_factor: float = 4.236,
        qqe_atr_smoothing: int = 14,
        neutral_band: float = 2.5,
        require_daily_regime: bool = True,
        require_hma_slope: bool = True,
        volume_pressure_threshold: float = 0.0,
        use_swingarm_exit: bool = True,
        swing_lookback: int = 10,
        atr_length: int = 14,
        atr_multiplier: float = 2.5,
        neutral_exit_bars: int = 3,
        exit_on_hma_break: bool = True,
        end_of_day_exit: bool = True,
        allow_overnight: bool = False,
    ) -> None:
        self.hma_length = hma_length
        self.rsi_length = rsi_length
        self.rsi_smoothing = rsi_smoothing
        self.qqe_factor = qqe_factor
        self.qqe_atr_smoothing = qqe_atr_smoothing
        self.neutral_band = neutral_band
        self.require_daily_regime = require_daily_regime
        self.require_hma_slope = require_hma_slope
        self.volume_pressure_threshold = volume_pressure_threshold
        self.use_swingarm_exit = use_swingarm_exit
        self.swing_lookback = swing_lookback
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.neutral_exit_bars = neutral_exit_bars
        self.exit_on_hma_break = exit_on_hma_break
        self.end_of_day_exit = end_of_day_exit
        self.allow_overnight = allow_overnight

    def parameters(self) -> dict:
        return {
            "hma_length": self.hma_length,
            "rsi_length": self.rsi_length,
            "rsi_smoothing": self.rsi_smoothing,
            "qqe_factor": self.qqe_factor,
            "qqe_atr_smoothing": self.qqe_atr_smoothing,
            "neutral_band": self.neutral_band,
            "require_daily_regime": self.require_daily_regime,
            "require_hma_slope": self.require_hma_slope,
            "volume_pressure_threshold": self.volume_pressure_threshold,
            "use_swingarm_exit": self.use_swingarm_exit,
            "swing_lookback": self.swing_lookback,
            "atr_length": self.atr_length,
            "atr_multiplier": self.atr_multiplier,
            "neutral_exit_bars": self.neutral_exit_bars,
            "exit_on_hma_break": self.exit_on_hma_break,
            "end_of_day_exit": self.end_of_day_exit,
            "allow_overnight": self.allow_overnight,
        }

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy().sort_values("timestamp").reset_index(drop=True)
        data = add_intraday_common_features(
            data,
            hma_length=self.hma_length,
            pressure_length=20,
            atr_length=self.atr_length,
            swing_lookback=self.swing_lookback,
            atr_multiplier=self.atr_multiplier,
            rsi_length=self.rsi_length,
            rsi_smoothing=self.rsi_smoothing,
            qqe_factor=self.qqe_factor,
            qqe_atr_smoothing=self.qqe_atr_smoothing,
            neutral_band=self.neutral_band,
        )
        regime_ok = data.get("daily_regime_bull", pd.Series(True, index=data.index)).fillna(False)
        if not self.require_daily_regime:
            regime_ok = pd.Series(True, index=data.index)
        hma_ok = data["close"] > data["hma"]
        if self.require_hma_slope:
            hma_ok = hma_ok & data["hma_slope_positive"].fillna(False)
        pressure_ok = data["pressure_z"] > self.volume_pressure_threshold
        data["entry_signal"] = regime_ok & data["qqe_long_state"].fillna(False) & hma_ok & pressure_ok
        exit_signal = (~regime_ok) | data["qqe_short_state"].fillna(False) | (data["qqe_neutral_consecutive"].fillna(0) >= self.neutral_exit_bars)
        if self.exit_on_hma_break:
            exit_signal = exit_signal | (data["close"] < data["hma"])
        if self.use_swingarm_exit:
            exit_signal = exit_signal | (data["close"] < data["long_arm"])
        data["exit_signal"] = exit_signal.fillna(False)
        return data

