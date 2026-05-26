from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybroker.common import PositionMode

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.fixed_strategy_utils import bars_to_frame, compute_qqe_frame, preclose_signal, resolve_position_size_shares
from trading_lab.pybroker_lab.strategies import ensure_indicator
from trading_lab.pybroker_lab.strategies.blackflag_fts_hma import compute_blackflag_fts


BLACKFLAG_FTS_QQE_MOMO_SETTINGS: dict[str, Any] = {
    "blackflag": {
        "trail_type": "modified",
        "atr_period": 28,
        "atr_factor": 5,
        "first_trade": "long",
        "average_type": "wilders",
    },
    "qqe_momo": {
        "source": "close",
        "rsi_period": 6,
        "slow_factor": 3,
        "qqe": 2.621,
        "show_momo_cloud": False,
        "show_qqe_label": True,
        "label_style": "None",
    },
}


def generate_signal_frame(frame: pd.DataFrame, *, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = BLACKFLAG_FTS_QQE_MOMO_SETTINGS if settings is None else settings
    bars = bars_to_frame(frame)
    blackflag = compute_blackflag_fts(bars, cfg["blackflag"])
    qqe = compute_qqe_frame(
        bars[str(cfg["qqe_momo"]["source"])].astype(float),
        rsi_period=int(cfg["qqe_momo"]["rsi_period"]),
        slow_factor=int(cfg["qqe_momo"]["slow_factor"]),
        qqe=float(cfg["qqe_momo"]["qqe"]),
    )
    qqe_bullish = (qqe["rsi_ma"].gt(qqe["trailing_line"]) & qqe["rsi_ma"].gt(50.0)).fillna(False).astype(bool)
    qqe_bearish = (qqe["rsi_ma"].lt(qqe["trailing_line"]) & qqe["rsi_ma"].lt(50.0)).fillna(False).astype(bool)
    blackflag_bull = blackflag["blackflag_state"].eq(1).fillna(False).astype(bool)
    blackflag_bear = blackflag["blackflag_state"].eq(-1).fillna(False).astype(bool)
    prior_blackflag_bull = blackflag_bull.shift(1, fill_value=False).astype(bool)
    prior_blackflag_bear = blackflag_bear.shift(1, fill_value=False).astype(bool)
    prior_qqe_bullish = qqe_bullish.shift(1, fill_value=False).astype(bool)
    prior_qqe_bearish = qqe_bearish.shift(1, fill_value=False).astype(bool)
    long_entry = (blackflag["buy_signal"] | (blackflag_bull & ~prior_blackflag_bull) | (qqe_bullish & ~prior_qqe_bullish)) & qqe_bullish & blackflag_bull
    short_entry = (blackflag["sell_signal"] | (blackflag_bear & ~prior_blackflag_bear) | (qqe_bearish & ~prior_qqe_bearish)) & qqe_bearish & blackflag_bear
    long_exit = blackflag["sell_signal"] | qqe_bearish | preclose_signal(bars)
    short_exit = blackflag["buy_signal"] | qqe_bullish | preclose_signal(bars)
    return pd.DataFrame(
        {
            "close": bars["close"].astype(float),
            "blackflag_state": blackflag["blackflag_state"],
            "true_range": blackflag["true_range"],
            "loss": blackflag["loss"],
            "trail": blackflag["trail"],
            "buy_signal": blackflag["buy_signal"],
            "sell_signal": blackflag["sell_signal"],
            "fib1": blackflag["fib1"],
            "fib2": blackflag["fib2"],
            "fib3": blackflag["fib3"],
            "source_used": str(cfg["qqe_momo"]["source"]),
            "rsi": qqe["rsi"],
            "qqe_rsi_ma": qqe["rsi_ma"],
            "qqe_trailing_line": qqe["trailing_line"],
            "qqe_bullish": qqe_bullish.fillna(False),
            "qqe_bearish": qqe_bearish.fillna(False),
            "long_entry_signal": long_entry.fillna(False),
            "short_entry_signal": short_entry.fillna(False),
            "long_exit_signal": long_exit.fillna(False),
            "short_exit_signal": short_exit.fillna(False),
            "long_entry_reason": np.where(long_entry.fillna(False), "blackflag_plus_qqe_bullish", None),
            "short_entry_reason": np.where(short_entry.fillna(False), "blackflag_plus_qqe_bearish", None),
            "long_exit_reason": np.where(long_exit.fillna(False), "blackflag_or_qqe_bearish_exit", None),
            "short_exit_reason": np.where(short_exit.fillna(False), "blackflag_or_qqe_bullish_exit", None),
        }
    )


def _make_indicator(column: str):
    name = f"pbl_blackflag_qqe_momo_{column}"

    def _indicator(data):
        return generate_signal_frame(bars_to_frame(data))[column].to_numpy()

    return ensure_indicator(name, _indicator)


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    indicators = tuple(
        _make_indicator(column)
        for column in [
            "long_entry_signal",
            "short_entry_signal",
            "long_exit_signal",
            "short_exit_signal",
            "blackflag_state",
            "qqe_bullish",
            "qqe_bearish",
        ]
    )
    indicator_names = {indicator.name: indicator for indicator in indicators}

    def exec_fn(ctx) -> None:
        target_shares = resolve_position_size_shares(ctx, sizing_method=config.sizing_method, sizing_value=config.sizing_value)
        long_entry = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_qqe_momo_long_entry_signal"].name)).fillna(False).astype(bool)
        short_entry = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_qqe_momo_short_entry_signal"].name)).fillna(False).astype(bool)
        long_exit = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_qqe_momo_long_exit_signal"].name)).fillna(False).astype(bool)
        short_exit = pd.Series(ctx.indicator(indicator_names["pbl_blackflag_qqe_momo_short_exit_signal"].name)).fillna(False).astype(bool)
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
        name="blackflag_fts_qqe_momo",
        symbols=tuple(config.symbols),
        indicators=indicators,
        execution=exec_fn,
        description="Blackflag FTS state machine confirmed by a fixed QQE momentum regime.",
        assumptions=(
            "QQE bullish confirmation is smoothed RSI above the QQE trailing line and above 50.",
            "QQE bearish confirmation is smoothed RSI below the QQE trailing line and below 50.",
            "Visual TradingView-style label and cloud settings remain hard-coded metadata and do not affect the backtest logic.",
        ),
        max_long_positions=1,
        max_short_positions=1,
        position_mode=PositionMode.DEFAULT,
    )
