from __future__ import annotations

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import make_sma_indicator


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    sma_length = int(config.strategy_params.get("sma_length", 200))
    sma_indicator = make_sma_indicator(f"pbl_sma_{sma_length}", sma_length)

    def exec_fn(ctx) -> None:
        sma = ctx.indicator(sma_indicator.name)
        if len(sma) < 2:
            return
        price = float(ctx.close[-1])
        if ctx.long_pos() is None and price > float(sma[-1]):
            ctx.buy_shares = ctx.calc_target_shares(1.0)
        elif ctx.long_pos() is not None and price <= float(sma[-1]):
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="spy_sma_regime",
        symbols=("SPY",),
        indicators=(sma_indicator,),
        execution=exec_fn,
        description="Long SPY when price is above the configured long-term SMA.",
        assumptions=(f"SMA length = {sma_length}.", "Long-only, otherwise in cash."),
    )
