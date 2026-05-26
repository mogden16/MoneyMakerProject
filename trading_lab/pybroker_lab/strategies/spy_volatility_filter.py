from __future__ import annotations

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.strategies import make_sma_indicator, make_vol_indicator


def build_strategy(config: PyBrokerLabConfig) -> PyBrokerStrategyDefinition:
    trend_length = int(config.strategy_params.get("trend_sma_length", 200))
    volatility_length = int(config.strategy_params.get("volatility_length", 20))
    volatility_threshold = float(config.strategy_params.get("volatility_threshold", 0.2))
    trend_sma = make_sma_indicator(f"pbl_vol_trend_sma_{trend_length}", trend_length)
    vol_indicator = make_vol_indicator(f"pbl_realized_vol_{volatility_length}", volatility_length)

    def exec_fn(ctx) -> None:
        sma = ctx.indicator(trend_sma.name)
        vol = ctx.indicator(vol_indicator.name)
        if len(sma) < 1 or len(vol) < 1:
            return
        close = float(ctx.close[-1])
        uptrend = close > float(sma[-1])
        low_vol = float(vol[-1]) <= volatility_threshold
        if ctx.long_pos() is None and uptrend and low_vol:
            ctx.buy_shares = ctx.calc_target_shares(1.0)
        elif ctx.long_pos() is not None and (not uptrend or not low_vol):
            ctx.sell_all_shares()

    return PyBrokerStrategyDefinition(
        name="spy_volatility_filter",
        symbols=("SPY",),
        indicators=(trend_sma, vol_indicator),
        execution=exec_fn,
        description="Owns SPY only when long-term trend is positive and realized volatility is muted.",
        assumptions=(
            f"Trend SMA length = {trend_length}.",
            f"Volatility threshold = {volatility_threshold:.2f}.",
        ),
    )
