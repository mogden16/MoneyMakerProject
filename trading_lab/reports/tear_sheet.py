from __future__ import annotations

import pandas as pd

from trading_lab.backtest.metrics import compute_summary_metrics, monthly_returns_table


def build_tear_sheet(equity_curve: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict:
    return {
        "metrics": compute_summary_metrics(equity_curve, trades, initial_capital),
        "monthly_returns": monthly_returns_table(equity_curve),
    }

