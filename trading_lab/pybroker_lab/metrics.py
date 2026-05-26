from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from trading_lab.backtest.metrics import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
)
from trading_lab.pybroker_lab.fixed_strategy_utils import strategy_metric_from_trades


@dataclass(frozen=True)
class StrategyDecision:
    status: str
    reason: str


def normalize_portfolio_curve(portfolio: pd.DataFrame, *, strategy_name: str, curve_type: str = "strategy") -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame(columns=["date", "equity", "cash", "market_value", "exposure", "curve_type", "strategy_name"])
    curve = portfolio.reset_index().rename(columns={"index": "date"}).copy()
    curve["date"] = pd.to_datetime(curve["date"])
    market_value = curve.get("market_value", pd.Series(0.0, index=curve.index)).astype(float)
    normalized = pd.DataFrame(
        {
            "date": curve["date"],
            "equity": curve["equity"].astype(float),
            "cash": curve.get("cash", pd.Series(0.0, index=curve.index)).astype(float),
            "market_value": market_value,
            "exposure": market_value.gt(0).astype(float),
            "curve_type": curve_type,
            "strategy_name": strategy_name,
        }
    )
    return normalized.reset_index(drop=True)


def normalize_trade_log(trades: pd.DataFrame, *, strategy_name: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["strategy_name"])
    frame = trades.reset_index().rename(columns={"index": "trade_id"}).copy()
    frame["strategy_name"] = strategy_name
    return frame


def compute_equity_metrics(
    *,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    strategy_name: str,
) -> dict[str, float | int | str]:
    if equity_curve.empty:
        return {
            "strategy_name": strategy_name,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "trade_count": 0,
            "exposure": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
        }
    trade_metrics = strategy_metric_from_trades(trades)
    curve = equity_curve[["date", "equity"]].rename(columns={"date": "timestamp"}).copy()
    metrics = {
        "strategy_name": strategy_name,
        "total_return": calculate_total_return(curve, initial_cash),
        "cagr": calculate_cagr(curve, initial_cash),
        "max_drawdown": calculate_max_drawdown(curve),
        "sharpe": calculate_sharpe_ratio(curve),
        "sortino": calculate_sortino_ratio(curve),
        "calmar": calculate_calmar_ratio(curve, initial_cash),
        "trade_count": int(len(trades)),
        "exposure": float(equity_curve["exposure"].mean()) if "exposure" in equity_curve.columns else 0.0,
        "win_rate": trade_metrics["win_rate"],
        "profit_factor": trade_metrics["profit_factor"],
        "average_win": trade_metrics["average_win"],
        "average_loss": trade_metrics["average_loss"],
    }
    return metrics


def build_delta_metrics(strategy_metrics: dict[str, Any], benchmark_metrics: dict[str, Any]) -> dict[str, Any]:
    fields = ["total_return", "cagr", "max_drawdown", "sharpe", "sortino", "calmar", "exposure", "win_rate", "profit_factor", "average_win", "average_loss"]
    delta = {"strategy_name": strategy_metrics["strategy_name"]}
    for field in fields:
        delta[f"{field}_delta"] = float(strategy_metrics.get(field, 0.0)) - float(benchmark_metrics.get(field, 0.0))
    return delta


def evaluate_strategy(
    *,
    strategy_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    cagr_tolerance: float,
) -> StrategyDecision:
    beats_sharpe = float(strategy_metrics.get("sharpe", 0.0)) > float(benchmark_metrics.get("sharpe", 0.0))
    better_drawdown = float(strategy_metrics.get("max_drawdown", 0.0)) > float(benchmark_metrics.get("max_drawdown", 0.0))
    cagr_gap = float(strategy_metrics.get("cagr", 0.0)) - float(benchmark_metrics.get("cagr", 0.0))
    if beats_sharpe and better_drawdown and cagr_gap >= -cagr_tolerance:
        return StrategyDecision(status="PASS", reason="Sharpe and drawdown beat SPY buy-and-hold without a materially worse CAGR.")
    return StrategyDecision(
        status="FAIL",
        reason="Did not clear the Sharpe/drawdown screen against SPY buy-and-hold while staying within the CAGR tolerance.",
    )


def bootstrap_to_frame(bootstrap: Any, *, strategy_name: str) -> pd.DataFrame:
    if bootstrap is None:
        return pd.DataFrame(columns=["strategy_name", "metric", "confidence", "lower", "upper"])
    intervals = bootstrap.conf_intervals.reset_index().copy()
    intervals.columns = ["metric", "confidence", "lower", "upper"]
    intervals["strategy_name"] = strategy_name
    return intervals[["strategy_name", "metric", "confidence", "lower", "upper"]]


def bootstrap_reliability_note(bootstrap: Any) -> str:
    if bootstrap is None:
        return "Bootstrap metrics were not available."
    try:
        sharpe_95 = float(bootstrap.sharpe.low_5)
        profit_factor_95 = float(bootstrap.profit_factor.low_5)
        drawdown_95 = float(bootstrap.drawdown.pct_confs.q_05)
    except Exception:
        return "Bootstrap metrics were present but could not be normalized."
    if sharpe_95 > 0 and profit_factor_95 > 1:
        return f"Bootstrap lower bounds stay constructive at 95% confidence (Sharpe >= {sharpe_95:.2f}, Profit Factor >= {profit_factor_95:.2f}, drawdown <= {drawdown_95:.2%})."
    return f"Bootstrap lower bounds are weak at 95% confidence (Sharpe >= {sharpe_95:.2f}, Profit Factor >= {profit_factor_95:.2f}, drawdown <= {drawdown_95:.2%})."


def decision_to_row(decision: StrategyDecision, *, strategy_name: str) -> dict[str, str]:
    return {
        "strategy_name": strategy_name,
        "status": decision.status,
        "decision_reason": decision.reason,
    }
