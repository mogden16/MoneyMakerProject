from __future__ import annotations

import pandas as pd

from trading_lab.pybroker_lab.metrics import compute_equity_metrics


def build_buy_and_hold_curve(
    bars: pd.DataFrame,
    *,
    initial_cash: float,
    strategy_name: str = "SPY Buy and Hold Benchmark",
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["date", "equity", "cash", "market_value", "exposure", "curve_type", "strategy_name"])
    ordered = bars.sort_values("date").reset_index(drop=True)
    entry_price = float(ordered["close"].iloc[0])
    shares = initial_cash / entry_price if entry_price else 0.0
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(ordered["date"]),
            "equity": ordered["close"].astype(float) * shares,
            "cash": 0.0,
            "market_value": ordered["close"].astype(float) * shares,
            "exposure": 1.0 if shares else 0.0,
            "curve_type": "benchmark",
            "strategy_name": strategy_name,
        }
    )
    curve.loc[curve.index[0], "equity"] = initial_cash
    curve.loc[curve.index[0], "market_value"] = initial_cash
    return curve


def benchmark_metrics(bars: pd.DataFrame, *, initial_cash: float) -> dict[str, float | int | str]:
    curve = build_buy_and_hold_curve(bars, initial_cash=initial_cash)
    return compute_equity_metrics(
        equity_curve=curve,
        trades=pd.DataFrame(),
        initial_cash=initial_cash,
        strategy_name="SPY Buy and Hold Benchmark",
    )
