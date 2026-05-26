from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybroker.common import PositionMode

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.fixed_strategy_utils import bars_to_frame, ema, preclose_signal, resolve_position_size_shares, standard_true_range, time_at_or_after, time_between
from trading_lab.pybroker_lab.strategies import ensure_indicator


EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS: dict[str, Any] = {
    "trade_type": "both",
    "session": "lunchBreak",
    "lunch_start": "12:30",
    "lunch_stop": "13:40",
    "exit_type": "reversal",
    "oco": "TPSL",
    "take_profit_pct": 0.35,
    "stop_loss_pct": 0.35,
    "take_profit_after_lunch_pct": 0.14,
    "trail_stop_pct": 0.04,
    "fast_ema_period": 30,
    "slow_ema_period": 250,
    "spread_under_percent": 0.10,
    "spread_over_percent": 0.21,
    "chop_length": 15,
    "chop_signal": 3,
    "choppy_level": 62,
    "midline": 50,
    "trending_level": 38,
}


def compute_choppiness_index(frame: pd.DataFrame, chop_length: int, chop_signal: int) -> pd.DataFrame:
    prev_close = frame["close"].shift(1)
    tr = standard_true_range(frame)
    range_high = pd.concat([frame["high"], prev_close], axis=1).max(axis=1).rolling(chop_length).max()
    range_low = pd.concat([frame["low"], prev_close], axis=1).min(axis=1).rolling(chop_length).min()
    range_span = (range_high - range_low).replace(0.0, np.nan)
    ci = (np.log10(tr.rolling(chop_length).sum() / range_span) / np.log10(chop_length)) * 100.0
    ci_avg = ci.rolling(chop_signal).mean()
    return pd.DataFrame({"true_range": tr, "ci": ci, "ci_avg": ci_avg})


def generate_signal_frame(frame: pd.DataFrame, *, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS if settings is None else settings)
    bars = bars_to_frame(frame)
    chop = compute_choppiness_index(bars, int(cfg["chop_length"]), int(cfg["chop_signal"]))
    fast_ema = ema(bars["close"], int(cfg["fast_ema_period"]))
    slow_ema = ema(bars["close"], int(cfg["slow_ema_period"]))
    ema_spread = fast_ema - slow_ema
    ema_spread_percent = ema_spread.abs() / bars["close"].replace(0.0, np.nan)
    candle_range = bars["high"] - bars["low"]
    lunch_mask = time_between(bars, str(cfg["lunch_start"]), str(cfg["lunch_stop"]))
    active_session = ~lunch_mask if str(cfg["session"]) == "lunchBreak" else pd.Series(True, index=bars.index)
    volume_confirm = bars["volume"].gt(bars["volume"].shift(1)) & bars["volume"].gt(bars["volume"].shift(2)) & bars["volume"].gt(bars["volume"].shift(3))
    range_expand = candle_range.gt(candle_range.shift(1)) & candle_range.gt(candle_range.shift(2))
    ci_confirm = chop["ci_avg"] > chop["ci"]
    long_spread = (ema_spread_percent * 100.0 < float(cfg["spread_under_percent"])) | (
        (ema_spread_percent * 100.0 > float(cfg["spread_over_percent"])) & fast_ema.lt(slow_ema)
    )
    short_spread = (ema_spread_percent * 100.0 < float(cfg["spread_under_percent"])) | (
        (ema_spread_percent * 100.0 > float(cfg["spread_over_percent"])) & fast_ema.gt(slow_ema)
    )
    break_up = bars["close"].gt(bars["high"].shift(1))
    break_down = bars["close"].lt(bars["low"].shift(1))
    long_signal = (
        active_session
        & ci_confirm
        & long_spread
        & bars["close"].gt(bars["close"].shift(1))
        & bars["close"].gt(bars["open"])
        & break_up
        & range_expand
        & volume_confirm
    )
    short_signal = (
        active_session
        & ci_confirm
        & short_spread
        & bars["close"].lt(bars["close"].shift(1))
        & bars["close"].lt(bars["open"])
        & break_down
        & range_expand
        & volume_confirm
    )
    eod_exit = preclose_signal(bars)
    long_exit_signal = eod_exit | short_signal
    short_exit_signal = eod_exit | long_signal
    take_profit_pct = np.where(time_at_or_after(bars, str(cfg["lunch_stop"])), float(cfg["take_profit_after_lunch_pct"]), float(cfg["take_profit_pct"]))
    return pd.DataFrame(
        {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "ema_spread": ema_spread,
            "ema_spread_percent": ema_spread_percent,
            "ci": chop["ci"],
            "ci_avg": chop["ci_avg"],
            "candle_range": candle_range,
            "prior_candle_range_1": candle_range.shift(1),
            "prior_candle_range_2": candle_range.shift(2),
            "volume": bars["volume"],
            "prior_volume_1": bars["volume"].shift(1),
            "prior_volume_2": bars["volume"].shift(2),
            "prior_volume_3": bars["volume"].shift(3),
            "break_up": break_up,
            "break_down": break_down,
            "active_session": active_session,
            "volume_confirm": volume_confirm,
            "range_expand": range_expand,
            "long_entry_signal": long_signal,
            "short_entry_signal": short_signal,
            "long_exit_signal": long_exit_signal,
            "short_exit_signal": short_exit_signal,
            "eod_exit_signal": eod_exit,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": float(cfg["stop_loss_pct"]),
            "long_entry_reason": np.where(long_signal, "ema_compression_breakout_long", None),
            "short_entry_reason": np.where(short_signal, "ema_compression_breakout_short", None),
            "long_exit_reason": np.where(eod_exit, "end_of_day_exit", np.where(short_signal, "reverse_to_short", None)),
            "short_exit_reason": np.where(eod_exit, "end_of_day_exit", np.where(long_signal, "reverse_to_long", None)),
        }
    )


def _make_indicator(column: str):
    name = f"pbl_ema_compression_breakout_{column}"

    def _indicator(data):
        return generate_signal_frame(bars_to_frame(data))[column].to_numpy()

    return ensure_indicator(name, _indicator)


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    indicators = tuple(
        _make_indicator(column)
        for column in [
            "long_entry_signal",
            "short_entry_signal",
            "eod_exit_signal",
            "take_profit_pct",
            "stop_loss_pct",
        ]
    )
    indicator_names = {indicator.name: indicator for indicator in indicators}

    def exec_fn(ctx) -> None:
        target_shares = resolve_position_size_shares(ctx, sizing_method=config.sizing_method, sizing_value=config.sizing_value)
        long_entry = pd.Series(ctx.indicator(indicator_names["pbl_ema_compression_breakout_long_entry_signal"].name)).fillna(False).astype(bool)
        short_entry = pd.Series(ctx.indicator(indicator_names["pbl_ema_compression_breakout_short_entry_signal"].name)).fillna(False).astype(bool)
        eod_exit = pd.Series(ctx.indicator(indicator_names["pbl_ema_compression_breakout_eod_exit_signal"].name)).fillna(False).astype(bool)
        take_profit = pd.Series(ctx.indicator(indicator_names["pbl_ema_compression_breakout_take_profit_pct"].name)).fillna(float(EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["take_profit_pct"]))
        stop_loss = pd.Series(ctx.indicator(indicator_names["pbl_ema_compression_breakout_stop_loss_pct"].name)).fillna(float(EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["stop_loss_pct"]))
        if long_entry.empty:
            return
        if ctx.long_pos() is not None and bool(eod_exit.iloc[-1] or short_entry.iloc[-1]):
            ctx.sell_all_shares()
            if bool(short_entry.iloc[-1]):
                ctx.sell_shares = target_shares
                ctx.stop_profit_pct = float(take_profit.iloc[-1])
                ctx.stop_loss_pct = float(stop_loss.iloc[-1])
        elif ctx.short_pos() is not None and bool(eod_exit.iloc[-1] or long_entry.iloc[-1]):
            ctx.cover_all_shares()
            if bool(long_entry.iloc[-1]):
                ctx.buy_shares = target_shares
                ctx.stop_profit_pct = float(take_profit.iloc[-1])
                ctx.stop_loss_pct = float(stop_loss.iloc[-1])
        elif ctx.long_pos() is None and ctx.short_pos() is None:
            if bool(long_entry.iloc[-1]):
                ctx.buy_shares = target_shares
                ctx.stop_profit_pct = float(take_profit.iloc[-1])
                ctx.stop_loss_pct = float(stop_loss.iloc[-1])
            elif bool(short_entry.iloc[-1]):
                ctx.sell_shares = target_shares
                ctx.stop_profit_pct = float(take_profit.iloc[-1])
                ctx.stop_loss_pct = float(stop_loss.iloc[-1])

    return PyBrokerStrategyDefinition(
        name="ema_compression_volume_breakout",
        symbols=tuple(config.symbols),
        indicators=indicators,
        execution=exec_fn,
        description="EMA compression breakout with volume and candle expansion filters plus fixed TP/SL brackets.",
        assumptions=(
            "TPSL exits use fixed in-code percentages and still execute with pybroker's next-bar order delay.",
            "Lunch-break filtering blocks new entries between 12:30 and 13:40 Eastern.",
            "End-of-day exits are signaled on the bar before the session close so the exit fills on the final bar open.",
        ),
        max_long_positions=1,
        max_short_positions=1,
        position_mode=PositionMode.DEFAULT,
    )
