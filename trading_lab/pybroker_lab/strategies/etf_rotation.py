from __future__ import annotations

import pandas as pd

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import make_return_indicator

DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "TLT", "GLD")


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    symbols = tuple(config.symbols) if len(config.symbols) > 1 else DEFAULT_UNIVERSE
    momentum_period = int(config.strategy_params.get("momentum_period", 63))
    top_n = int(config.strategy_params.get("top_n", 2))
    rebalance_frequency = str(config.strategy_params.get("rebalance_frequency", "monthly")).lower()
    momentum = make_return_indicator(f"pbl_rotation_mom_{momentum_period}", momentum_period)

    def exec_fn(ctx) -> None:
        current_date = pd.Timestamp(ctx.dt)
        rebalance_key = current_date.to_period("W-FRI") if rebalance_frequency == "weekly" else current_date.to_period("M")
        session_key = f"rotation_selection_{rebalance_key}"
        if session_key not in ctx.session:
            scores: list[tuple[str, float]] = []
            for symbol in symbols:
                values = ctx.indicator(momentum.name, symbol)
                if len(values) == 0 or pd.isna(values[-1]):
                    continue
                scores.append((symbol, float(values[-1])))
            scores.sort(key=lambda item: item[1], reverse=True)
            ctx.session[session_key] = {symbol for symbol, _ in scores[:top_n]}
        selected = ctx.session[session_key]
        if ctx.symbol in selected:
            if ctx.long_pos() is None:
                ctx.score = float(ctx.indicator(momentum.name)[-1])
                ctx.buy_shares = ctx.calc_target_shares(1.0 / max(top_n, 1))
        elif ctx.long_pos() is not None:
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="etf_rotation",
        symbols=symbols,
        indicators=(momentum,),
        execution=exec_fn,
        description="Ranks a small ETF universe by trailing momentum and holds the leaders.",
        assumptions=(
            f"Universe = {', '.join(symbols)}.",
            f"Top N = {top_n}.",
            f"Rebalance frequency = {rebalance_frequency}.",
        ),
        max_long_positions=top_n,
    )
