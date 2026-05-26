from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybroker.common import PositionMode

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.fixed_strategy_utils import (
    bars_to_frame,
    combine_weighted_value,
    compute_qqe_frame,
    cross_above,
    cross_below,
    merge_higher_timeframe_provenance,
    preclose_signal,
    resolve_position_size_shares,
    resample_ohlcv,
    timeframe_minutes,
)
from trading_lab.pybroker_lab.strategies import ensure_indicator


LEGACY_MTF_QQE_RSI_SETTINGS: dict[str, Any] = {
    "trade_type": "both",
    "session": "marketHours",
    "exit_type": "exitSignal",
    "source": "hl2",
    "rsi_period": 6,
    "slow_factor": 3,
    "qqe": 2.621,
    "additional_timeframe": "30min",
    "rsi_super_oversold": 20,
    "rsi_oversold": 35,
    "rsi_low_neutral": 45,
    "rsi_high_neutral": 55,
    "rsi_overbought": 75,
    "rsi_super_overbought": 80,
    "rsi_paint_type": "weighted",
    "time_span": "multi",
    "risk_mode": "risky",
    "chop_length": 15,
    "chop_signal": 3,
    "choppy_level": 62,
    "midline": 50,
    "trending_level": 38,
    "legacy_weighting_mode": "corrected_weighted",
}


def _source_series(frame: pd.DataFrame, source: str) -> pd.Series:
    if source == "hl2":
        return (frame["high"] + frame["low"]) / 2.0
    if source == "close":
        return frame["close"].astype(float)
    raise ValueError(f"Unsupported source: {source}")


def generate_signal_frame(frame: pd.DataFrame, *, timeframe: str, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(LEGACY_MTF_QQE_RSI_SETTINGS if settings is None else settings)
    bars = bars_to_frame(frame)
    current_source = _source_series(bars, str(cfg["source"]))
    current = compute_qqe_frame(
        current_source,
        rsi_period=int(cfg["rsi_period"]),
        slow_factor=int(cfg["slow_factor"]),
        qqe=float(cfg["qqe"]),
    )
    higher_bars = resample_ohlcv(bars, str(cfg["additional_timeframe"]))
    higher_source = _source_series(higher_bars, str(cfg["source"]))
    higher = compute_qqe_frame(
        higher_source,
        rsi_period=int(cfg["rsi_period"]),
        slow_factor=int(cfg["slow_factor"]),
        qqe=float(cfg["qqe"]),
    )
    merged_higher = merge_higher_timeframe_provenance(
        bars,
        pd.concat([higher_bars[["timestamp"]], higher], axis=1).sort_values("timestamp"),
        timeframe=str(cfg["additional_timeframe"]),
        value_columns=("rsi_ma", "trailing_line", "rsi_ma_dot"),
        prefix="higher_qqe",
    )
    multiplier = max(float(timeframe_minutes(str(cfg["additional_timeframe"])) / max(timeframe_minutes(timeframe), 1)), 1.0)
    combined_higher = combine_weighted_value(
        merged_higher["rsi_ma_dot"].fillna(0.0),
        merged_higher["rsi_ma"].fillna(50.0),
        multiplier,
        str(cfg["legacy_weighting_mode"]),
    )
    combined_momentum = (current["rsi_ma"].fillna(50.0) + combined_higher) / 2.0
    current_bull = current["rsi_ma"].gt(current["trailing_line"]) & current["rsi_ma"].gt(50)
    current_bear = current["rsi_ma"].lt(current["trailing_line"]) & current["rsi_ma"].lt(50)
    higher_bull = merged_higher["rsi_ma"].gt(merged_higher["trailing_line"]) & merged_higher["rsi_ma"].gt(50)
    higher_bear = merged_higher["rsi_ma"].lt(merged_higher["trailing_line"]) & merged_higher["rsi_ma"].lt(50)
    if str(cfg["risk_mode"]) == "risky":
        long_cross = cross_above(combined_momentum, float(cfg["rsi_oversold"])) | cross_above(combined_momentum, float(cfg["rsi_low_neutral"]))
        short_cross = cross_below(combined_momentum, float(cfg["rsi_overbought"])) | cross_below(combined_momentum, float(cfg["rsi_high_neutral"]))
    else:
        long_cross = cross_above(combined_momentum, float(cfg["rsi_oversold"]))
        short_cross = cross_below(combined_momentum, float(cfg["rsi_overbought"]))
    long_entry_signal = long_cross & current_bull.fillna(False) & higher_bull.fillna(False)
    short_entry_signal = short_cross & current_bear.fillna(False) & higher_bear.fillna(False)
    long_exit_signal = cross_below(combined_momentum, float(cfg["rsi_high_neutral"])) | current_bear.fillna(False)
    short_exit_signal = cross_above(combined_momentum, float(cfg["rsi_low_neutral"])) | current_bull.fillna(False)
    eod_exit = preclose_signal(bars)
    return pd.DataFrame(
        {
            "source_used": str(cfg["source"]),
            "rsi": current["rsi"],
            "rsi_ma": current["rsi_ma"],
            "qqe_trailing_line": current["trailing_line"],
            "higher_rsi_ma": merged_higher["rsi_ma"],
            "higher_trailing_line": merged_higher["trailing_line"],
            "higher_qqe_source_timestamp": merged_higher["higher_qqe_source_timestamp"],
            "higher_qqe_source_close_timestamp": merged_higher["higher_qqe_source_close_timestamp"],
            "higher_qqe_higher_close_timestamp": merged_higher["higher_qqe_higher_close_timestamp"],
            "higher_qqe_bar_complete": merged_higher["higher_qqe_bar_complete"],
            "higher_qqe_lookahead_flag": merged_higher["higher_qqe_lookahead_flag"],
            "higher_qqe_lookahead_result": merged_higher["higher_qqe_lookahead_result"],
            "combined_momentum": combined_momentum,
            "current_bull": current_bull.fillna(False),
            "current_bear": current_bear.fillna(False),
            "higher_bull": higher_bull.fillna(False),
            "higher_bear": higher_bear.fillna(False),
            "long_entry_signal": long_entry_signal.fillna(False),
            "short_entry_signal": short_entry_signal.fillna(False),
            "long_exit_signal": long_exit_signal.fillna(False) | eod_exit,
            "short_exit_signal": short_exit_signal.fillna(False) | eod_exit,
            "long_entry_reason": pd.Series(np.where(long_entry_signal.fillna(False), "qqe_momentum_cross_up", None), index=bars.index),
            "short_entry_reason": pd.Series(np.where(short_entry_signal.fillna(False), "qqe_momentum_cross_down", None), index=bars.index),
            "long_exit_reason": pd.Series(np.where(eod_exit, "end_of_day_exit", np.where(long_exit_signal.fillna(False), "bearish_qqe_exit", None)), index=bars.index),
            "short_exit_reason": pd.Series(np.where(eod_exit, "end_of_day_exit", np.where(short_exit_signal.fillna(False), "bullish_qqe_exit", None)), index=bars.index),
        }
    )


def _make_indicator(column: str, *, timeframe: str):
    name = f"pbl_legacy_mtf_qqe_{column}_{timeframe}"

    def _indicator(data):
        return generate_signal_frame(bars_to_frame(data), timeframe=timeframe)[column].to_numpy()

    return ensure_indicator(name, _indicator)


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    indicators = tuple(
        _make_indicator(column, timeframe=config.timeframe)
        for column in [
            "long_entry_signal",
            "short_entry_signal",
            "long_exit_signal",
            "short_exit_signal",
            "combined_momentum",
        ]
    )
    indicator_names = {indicator.name: indicator for indicator in indicators}

    def exec_fn(ctx) -> None:
        target_shares = resolve_position_size_shares(ctx, sizing_method=config.sizing_method, sizing_value=config.sizing_value)
        long_entry = pd.Series(ctx.indicator(indicator_names[f"pbl_legacy_mtf_qqe_long_entry_signal_{config.timeframe}"].name)).fillna(False).astype(bool)
        short_entry = pd.Series(ctx.indicator(indicator_names[f"pbl_legacy_mtf_qqe_short_entry_signal_{config.timeframe}"].name)).fillna(False).astype(bool)
        long_exit = pd.Series(ctx.indicator(indicator_names[f"pbl_legacy_mtf_qqe_long_exit_signal_{config.timeframe}"].name)).fillna(False).astype(bool)
        short_exit = pd.Series(ctx.indicator(indicator_names[f"pbl_legacy_mtf_qqe_short_exit_signal_{config.timeframe}"].name)).fillna(False).astype(bool)
        if long_entry.empty:
            return
        if ctx.long_pos() is not None and bool(long_exit.iloc[-1] or short_entry.iloc[-1]):
            ctx.sell_all_shares()
            if bool(short_entry.iloc[-1]):
                ctx.sell_shares = target_shares
        elif ctx.short_pos() is not None and bool(short_exit.iloc[-1] or long_entry.iloc[-1]):
            ctx.cover_all_shares()
            if bool(long_entry.iloc[-1]):
                ctx.buy_shares = target_shares
        elif ctx.long_pos() is None and ctx.short_pos() is None:
            if bool(long_entry.iloc[-1]):
                ctx.buy_shares = target_shares
            elif bool(short_entry.iloc[-1]):
                ctx.sell_shares = target_shares

    return PyBrokerStrategyDefinition(
        name="legacy_mtf_qqe_rsi_momentum",
        symbols=tuple(config.symbols),
        indicators=indicators,
        execution=exec_fn,
        description="Legacy multi-timeframe QQE and RSI momentum strategy with a corrected weighted higher-timeframe blend.",
        assumptions=(
            "The higher-timeframe weighting multiplier is the ratio of additional timeframe minutes to base timeframe minutes.",
            "The corrected weighted mode is the default internal blend for the legacy suspicious formula.",
            "End-of-day exits are scheduled on the bar before the session close so fills remain next-bar and same-session.",
        ),
        max_long_positions=1,
        max_short_positions=1,
        position_mode=PositionMode.DEFAULT,
    )
