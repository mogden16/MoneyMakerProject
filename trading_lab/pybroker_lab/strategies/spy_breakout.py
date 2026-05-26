from __future__ import annotations

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import make_high_indicator, make_sma_indicator


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    lookback = int(config.strategy_params.get("breakout_lookback", 50))
    hold_bars = int(config.strategy_params.get("hold_bars", 20))
    exit_sma_length = int(config.strategy_params.get("exit_sma_length", 20))
    stop_loss_pct = float(config.strategy_params.get("stop_loss_pct", 0.08))
    high_indicator = make_high_indicator(f"pbl_breakout_high_{lookback}", lookback)
    exit_sma = make_sma_indicator(f"pbl_breakout_exit_sma_{exit_sma_length}", exit_sma_length)

    def exec_fn(ctx) -> None:
        highs = ctx.indicator(high_indicator.name)
        sma = ctx.indicator(exit_sma.name)
        if len(highs) < 2 or len(sma) < 1:
            return
        close = float(ctx.close[-1])
        breakout = float(highs[-1]) > float(highs[-2])
        if ctx.long_pos() is None and breakout:
            ctx.buy_shares = ctx.calc_target_shares(1.0)
            ctx.hold_bars = hold_bars
            ctx.stop_loss_pct = stop_loss_pct * 100.0
        elif ctx.long_pos() is not None and close < float(sma[-1]):
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="spy_breakout",
        symbols=("SPY",),
        indicators=(high_indicator, exit_sma),
        execution=exec_fn,
        description="Buys SPY on new N-day highs and exits on time or SMA failure.",
        assumptions=(
            f"Breakout lookback = {lookback}.",
            f"Exit SMA length = {exit_sma_length}.",
            f"Stop loss = {stop_loss_pct:.1%}.",
        ),
    )
