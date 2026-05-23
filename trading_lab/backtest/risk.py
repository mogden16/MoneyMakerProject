from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskExit:
    exit_price: float
    exit_reason: str


def evaluate_risk_exit(
    bar: pd.Series,
    entry_price: float,
    highest_close: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    trailing_stop_pct: float | None,
) -> RiskExit | None:
    if stop_loss_pct:
        stop_price = entry_price * (1 - stop_loss_pct)
        if float(bar["low"]) <= stop_price:
            return RiskExit(exit_price=stop_price, exit_reason="stop_loss")

    if take_profit_pct:
        take_profit_price = entry_price * (1 + take_profit_pct)
        if float(bar["high"]) >= take_profit_price:
            return RiskExit(exit_price=take_profit_price, exit_reason="take_profit")

    if trailing_stop_pct:
        trailing_stop_price = highest_close * (1 - trailing_stop_pct)
        if float(bar["low"]) <= trailing_stop_price:
            return RiskExit(exit_price=trailing_stop_price, exit_reason="trailing_stop")

    return None

