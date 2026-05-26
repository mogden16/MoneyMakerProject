from __future__ import annotations

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import make_bollinger_lower_indicator, make_rsi_indicator


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    rsi_length = int(config.strategy_params.get("rsi_length", 14))
    rsi_entry = float(config.strategy_params.get("rsi_entry", 30.0))
    rsi_exit = float(config.strategy_params.get("rsi_exit", 50.0))
    bollinger_length = int(config.strategy_params.get("bollinger_length", 20))
    bollinger_std = float(config.strategy_params.get("bollinger_std", 2.0))
    hold_bars = int(config.strategy_params.get("hold_bars", 5))
    rsi_indicator = make_rsi_indicator(f"pbl_mr_rsi_{rsi_length}", rsi_length)
    lower_band = make_bollinger_lower_indicator(f"pbl_mr_lower_{bollinger_length}_{bollinger_std}", bollinger_length, bollinger_std)

    def exec_fn(ctx) -> None:
        rsi = ctx.indicator(rsi_indicator.name)
        lower = ctx.indicator(lower_band.name)
        if len(rsi) < 1 or len(lower) < 1:
            return
        close = float(ctx.close[-1])
        enter_signal = float(rsi[-1]) < rsi_entry or close < float(lower[-1])
        if ctx.long_pos() is None and enter_signal:
            ctx.buy_shares = ctx.calc_target_shares(1.0)
            ctx.hold_bars = hold_bars
        elif ctx.long_pos() is not None and float(rsi[-1]) >= rsi_exit:
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="spy_mean_reversion",
        symbols=("SPY",),
        indicators=(rsi_indicator, lower_band),
        execution=exec_fn,
        description="Buys SPY on short-term oversold conditions.",
        assumptions=(
            f"RSI entry/exit = {rsi_entry:.1f}/{rsi_exit:.1f}.",
            f"Bollinger length/std = {bollinger_length}/{bollinger_std}.",
            f"Max hold = {hold_bars} bars.",
        ),
    )
