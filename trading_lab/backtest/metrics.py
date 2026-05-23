from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.DataFrame, initial_capital: float) -> float:
    if equity_curve.empty:
        return 0.0
    return float(equity_curve["equity"].iloc[-1] / initial_capital - 1)


def calculate_cagr(equity_curve: pd.DataFrame, initial_capital: float) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    start = pd.to_datetime(equity_curve["timestamp"].iloc[0])
    end = pd.to_datetime(equity_curve["timestamp"].iloc[-1])
    years = max((end - start).days / 365.25, 1 / 365.25)
    ending_value = equity_curve["equity"].iloc[-1]
    return float((ending_value / initial_capital) ** (1 / years) - 1)


def calculate_max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve["equity"].cummax()
    drawdowns = equity_curve["equity"] / running_max - 1
    return float(drawdowns.min())


def _equity_returns(equity_curve: pd.DataFrame) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float)
    return equity_curve["equity"].pct_change().dropna()


def calculate_sharpe_ratio(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    returns = _equity_returns(equity_curve)
    if returns.std() == 0 or returns.empty:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return float((excess.mean() / excess.std()) * math.sqrt(252))


def calculate_sortino_ratio(equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    returns = _equity_returns(equity_curve)
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * math.sqrt(252))


def calculate_calmar_ratio(equity_curve: pd.DataFrame, initial_capital: float) -> float:
    cagr = calculate_cagr(equity_curve, initial_capital)
    max_drawdown = abs(calculate_max_drawdown(equity_curve))
    if max_drawdown == 0:
        return 0.0
    return float(cagr / max_drawdown)


def calculate_benchmark_metrics(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if equity_curve.empty or benchmark_curve.empty:
        return {
            "benchmark_total_return": 0.0,
            "benchmark_cagr": 0.0,
            "benchmark_max_drawdown": 0.0,
            "excess_cagr": 0.0,
            "beta": 0.0,
            "correlation": 0.0,
        }
    benchmark_equity = benchmark_curve.rename(columns={"benchmark_equity": "equity"})
    benchmark_total = calculate_total_return(benchmark_equity, initial_capital)
    benchmark_cagr = calculate_cagr(benchmark_equity, initial_capital)
    benchmark_dd = calculate_max_drawdown(benchmark_equity)

    strategy_returns = _equity_returns(equity_curve)
    benchmark_returns = benchmark_equity["equity"].pct_change().dropna()
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0 or aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        beta = 0.0
        correlation = 0.0
    else:
        beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
        correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return {
        "benchmark_total_return": benchmark_total,
        "benchmark_cagr": benchmark_cagr,
        "benchmark_max_drawdown": benchmark_dd,
        "excess_cagr": float(calculate_cagr(equity_curve, initial_capital) - benchmark_cagr),
        "beta": beta,
        "correlation": correlation,
    }


def compute_summary_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    *,
    benchmark_curve: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    total_return = calculate_total_return(equity_curve, initial_capital)
    cagr = calculate_cagr(equity_curve, initial_capital)
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    calmar = calculate_calmar_ratio(equity_curve, initial_capital)
    benchmark_metrics = calculate_benchmark_metrics(equity_curve, benchmark_curve if benchmark_curve is not None else pd.DataFrame(), initial_capital)

    base = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Benchmark Total Return": benchmark_metrics["benchmark_total_return"],
        "Benchmark CAGR": benchmark_metrics["benchmark_cagr"],
        "Benchmark Max Drawdown": benchmark_metrics["benchmark_max_drawdown"],
        "Excess CAGR": benchmark_metrics["excess_cagr"],
        "Beta": benchmark_metrics["beta"],
        "Correlation": benchmark_metrics["correlation"],
    }

    if trades.empty:
        return {
            **base,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Average Trade Return": 0.0,
            "Average Holding Period": 0.0,
            "Number of Trades": 0,
            "Exposure %": 0.0,
            "Best Trade": 0.0,
            "Worst Trade": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    invested_days = equity_curve.get("positions_value", pd.Series(dtype=float)).gt(0).mean() if "positions_value" in equity_curve.columns else 0.0
    return {
        **base,
        "Win Rate": float((trades["pnl"] > 0).mean()),
        "Profit Factor": float(gross_profit / gross_loss) if gross_loss else 0.0,
        "Average Trade Return": float(trades["return_pct"].mean()),
        "Average Holding Period": float(trades["holding_days"].mean()),
        "Number of Trades": int(len(trades)),
        "Exposure %": float(invested_days),
        "Best Trade": float(trades["return_pct"].max()),
        "Worst Trade": float(trades["return_pct"].min()),
    }


def monthly_returns_table(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame()
    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    monthly = frame.set_index("timestamp")["equity"].resample("ME").last().pct_change()
    if monthly.empty:
        return pd.DataFrame()
    result = monthly.to_frame("return")
    result["year"] = result.index.year
    result["month"] = result.index.month_name().str[:3]
    pivot = result.pivot(index="year", columns="month", values="return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=month_order)
