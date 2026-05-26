from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybroker.common import PositionMode

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.fixed_strategy_utils import (
    bars_to_frame,
    compute_higher_timeframe_hma,
    modified_true_range,
    resolve_position_size_shares,
    standard_true_range,
    wilder_moving_average,
)
from trading_lab.pybroker_lab.strategies import ensure_indicator


BLACKFLAG_FTS_HMA_SETTINGS: dict[str, Any] = {
    "trail_type": "modified",
    "atr_period": 28,
    "atr_factor": 5,
    "first_trade": "long",
    "average_type": "wilders",
    "fib1_level": 61.8,
    "fib2_level": 78.6,
    "fib3_level": 88.6,
    "hma_length": 21,
    "hma_lookback": 2,
    "hma_timeframe": "60min",
    "hma_source": "close",
}


def compute_blackflag_fts(frame: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**BLACKFLAG_FTS_HMA_SETTINGS, **(settings or {})}
    bars = bars_to_frame(frame)
    tr_frame = modified_true_range(bars, int(cfg["atr_period"]))
    true_range = tr_frame["true_range"] if cfg["trail_type"] == "modified" else standard_true_range(bars)
    loss = float(cfg["atr_factor"]) * wilder_moving_average(true_range, int(cfg["atr_period"]))
    state = pd.Series(0, index=bars.index, dtype=int)
    trail = pd.Series(np.nan, index=bars.index, dtype=float)
    extremum = pd.Series(np.nan, index=bars.index, dtype=float)
    buy_signal = pd.Series(False, index=bars.index)
    sell_signal = pd.Series(False, index=bars.index)

    first_state = 1 if str(cfg["first_trade"]).lower() == "long" else -1
    for idx in range(len(bars)):
        if pd.isna(loss.iloc[idx]):
            continue
        close = float(bars["close"].iloc[idx])
        high = float(bars["high"].iloc[idx])
        low = float(bars["low"].iloc[idx])
        if idx == 0 or state.iloc[idx - 1] == 0 or pd.isna(trail.iloc[idx - 1]):
            state.iloc[idx] = first_state
            trail.iloc[idx] = close - loss.iloc[idx] if first_state == 1 else close + loss.iloc[idx]
            extremum.iloc[idx] = high if first_state == 1 else low
            continue

        prev_state = int(state.iloc[idx - 1])
        prev_trail = float(trail.iloc[idx - 1])
        prev_extremum = float(extremum.iloc[idx - 1])
        if prev_state == 1:
            if close > prev_trail:
                state.iloc[idx] = 1
                trail.iloc[idx] = max(prev_trail, close - loss.iloc[idx])
                extremum.iloc[idx] = max(prev_extremum, high)
            else:
                state.iloc[idx] = -1
                trail.iloc[idx] = close + loss.iloc[idx]
                extremum.iloc[idx] = low
                sell_signal.iloc[idx] = True
        else:
            if close < prev_trail:
                state.iloc[idx] = -1
                trail.iloc[idx] = min(prev_trail, close + loss.iloc[idx])
                extremum.iloc[idx] = min(prev_extremum, low)
            else:
                state.iloc[idx] = 1
                trail.iloc[idx] = close - loss.iloc[idx]
                extremum.iloc[idx] = high
                buy_signal.iloc[idx] = True

    fib_levels = {}
    for fib_name, fib_level in [("fib1", cfg["fib1_level"]), ("fib2", cfg["fib2_level"]), ("fib3", cfg["fib3_level"])]:
        fib_ratio = float(fib_level) / 100.0
        fib_levels[fib_name] = np.where(
            state.eq(1),
            trail + (extremum - trail) * fib_ratio,
            trail - (trail - extremum) * fib_ratio,
        )

    return pd.DataFrame(
        {
            "HiLo": tr_frame["HiLo"],
            "HRef": tr_frame["HRef"],
            "LRef": tr_frame["LRef"],
            "true_range": true_range,
            "loss": loss,
            "blackflag_state": state,
            "trail": trail,
            "extremum": extremum,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            **fib_levels,
        }
    )


def classify_hull_value(hma: pd.Series, lookback: int) -> pd.DataFrame:
    delta = hma.shift(1) - hma.shift(lookback + 1)
    delta_per_bar = delta / float(lookback)
    next_bar = hma.shift(1) + delta_per_bar
    concavity = pd.Series(np.where(hma > next_bar, 1, -1), index=hma.index, dtype=int)
    prior_hma = hma.shift(1)
    hull_value = pd.Series(np.nan, index=hma.index, dtype=float)
    hull_value = hull_value.mask(concavity.eq(-1) & hma.gt(prior_hma), 3)
    hull_value = hull_value.mask(concavity.eq(-1) & hma.le(prior_hma), 4)
    hull_value = hull_value.mask(concavity.eq(1) & hma.lt(prior_hma), 1)
    hull_value = hull_value.mask(concavity.eq(1) & hma.ge(prior_hma), 2)
    return pd.DataFrame({"higher_hma": hma, "delta": delta, "next_bar": next_bar, "concavity": concavity, "hull_value": hull_value})


def generate_signal_frame(frame: pd.DataFrame, *, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**BLACKFLAG_FTS_HMA_SETTINGS, **(settings or {})}
    bars = bars_to_frame(frame)
    blackflag = compute_blackflag_fts(bars, cfg)
    higher_hma = compute_higher_timeframe_hma(
        bars,
        timeframe=str(cfg["hma_timeframe"]),
        source=str(cfg["hma_source"]),
        length=int(cfg["hma_length"]),
    )
    hull = classify_hull_value(higher_hma["higher_hma"], int(cfg["hma_lookback"]))
    signals = pd.concat([bars[["timestamp"]], blackflag, hull], axis=1)
    signals["close"] = bars["close"].astype(float)
    for column in [
        "higher_hma_source_timestamp",
        "higher_hma_source_close_timestamp",
        "higher_hma_higher_close_timestamp",
        "higher_hma_bar_complete",
        "higher_hma_lookahead_flag",
        "higher_hma_lookahead_result",
    ]:
        if column in higher_hma.columns:
            signals[column] = higher_hma[column]
    signals["long_entry_signal"] = signals["buy_signal"] & signals["hull_value"].le(2)
    signals["long_exit_signal"] = signals["hull_value"].gt(2)
    signals["short_entry_signal"] = signals["sell_signal"] & signals["hull_value"].ge(3)
    signals["short_exit_signal"] = signals["hull_value"].lt(3)
    signals["long_entry_reason"] = np.where(signals["long_entry_signal"], "blackflag_buy_plus_hma_confirm", None)
    signals["short_entry_reason"] = np.where(signals["short_entry_signal"], "blackflag_sell_plus_hma_confirm", None)
    signals["long_exit_reason"] = np.where(signals["long_exit_signal"], "hma_filter_exit", None)
    signals["short_exit_reason"] = np.where(signals["short_exit_signal"], "hma_filter_exit", None)
    return signals


def _make_indicator(column: str):
    name = f"pbl_blackflag_fts_hma_{column}"

    def _indicator(data):
        return generate_signal_frame(bars_to_frame(data))[column].to_numpy()

    return ensure_indicator(name, _indicator)


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    indicators = tuple(
        _make_indicator(column)
        for column in [
            "long_entry_signal",
            "long_exit_signal",
            "short_entry_signal",
            "short_exit_signal",
            "blackflag_state",
            "hull_value",
        ]
    )
    indicator_names = {indicator.name: indicator for indicator in indicators}

    def exec_fn(ctx) -> None:
        target_shares = resolve_position_size_shares(ctx, sizing_method=config.sizing_method, sizing_value=config.sizing_value)
        long_entry = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_fts_hma_long_entry_signal"].name)).fillna(False).astype(bool)
        long_exit = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_fts_hma_long_exit_signal"].name)).fillna(False).astype(bool)
        short_entry = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_fts_hma_short_entry_signal"].name)).fillna(False).astype(bool)
        short_exit = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_fts_hma_short_exit_signal"].name)).fillna(False).astype(bool)
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
        name="blackflag_fts_hma",
        symbols=tuple(config.symbols),
        indicators=indicators,
        execution=exec_fn,
        description="Blackflag FTS trend flips filtered by a higher-timeframe Hull Moving Average state.",
        assumptions=(
            "Signals are calculated on completed bars and orders execute on the next bar open.",
            "Higher-timeframe Hull values are merged with backward-only asof joins to avoid lookahead.",
            "Fib levels are computed for research output but are not used for entries or exits.",
        ),
        max_long_positions=1,
        max_short_positions=1,
        position_mode=PositionMode.DEFAULT,
    )
