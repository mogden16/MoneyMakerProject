from __future__ import annotations

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    def exec_fn(ctx) -> None:
        if ctx.long_pos() is None:
            ctx.buy_shares = ctx.calc_target_shares(1.0)

    return PyBrokerStrategyDefinition(
        name="spy_buy_hold",
        symbols=("SPY",),
        indicators=(),
        execution=exec_fn,
        description="Buys SPY once and holds it for the duration of the out-of-sample test.",
        assumptions=("Long-only.", "Fully invested after the first eligible buy signal."),
    )
