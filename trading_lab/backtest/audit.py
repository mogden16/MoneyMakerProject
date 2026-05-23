from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_lab.backtest.robustness import profit_concentration_analysis


@dataclass
class AuditFinding:
    severity: str
    message: str


def generate_audit_findings(
    metrics: dict[str, float | int],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    benchmark_metrics: dict[str, float] | None = None,
    strategy_parameters: dict | None = None,
    regime_comments: list[str] | None = None,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    num_trades = int(metrics.get("Number of Trades", 0) or 0)
    win_rate = float(metrics.get("Win Rate", 0.0) or 0.0)
    cagr = float(metrics.get("CAGR", 0.0) or 0.0)
    max_drawdown = abs(float(metrics.get("Max Drawdown", 0.0) or 0.0))
    exposure = float(metrics.get("Exposure %", 0.0) or 0.0)

    if num_trades < 10:
        findings.append(AuditFinding("warning", f"This strategy only generated {num_trades} trades. The sample is thin, so treat the result cautiously."))
    if num_trades < 15 and win_rate > 0.8:
        findings.append(AuditFinding("warning", "The win rate is very high, but the trade count is low. This can look better than it really is out of sample."))

    concentration_report = profit_concentration_analysis(trades)
    if concentration_report["best_trade_profit_share"] > 0.5:
        findings.append(AuditFinding("critical", f"This strategy produced a strong result, but {concentration_report['best_trade_profit_share']:.0%} of total profit came from one trade."))
    if concentration_report["top_5_profit_share"] > 0.75:
        findings.append(AuditFinding("warning", f"The top 5 trades generated {concentration_report['top_5_profit_share']:.0%} of total profit."))
    if concentration_report["ticker_contribution"]:
        top_symbol, top_symbol_share = next(iter(concentration_report["ticker_contribution"].items()))
        if top_symbol_share > 0.7:
            findings.append(AuditFinding("warning", f"Performance is highly concentrated in {top_symbol}, which contributed {top_symbol_share:.0%} of total PnL."))
    if concentration_report["year_contribution"]:
        top_year, top_year_share = next(iter(concentration_report["year_contribution"].items()))
        if top_year_share > 0.7:
            findings.append(AuditFinding("warning", f"Results are heavily concentrated in {top_year}, which contributed {top_year_share:.0%} of total profit."))

    if not trades.empty:
        if metrics.get("Profit Factor", 0.0) and float(metrics.get("Profit Factor", 0.0)) > 3 and num_trades < 20:
            findings.append(AuditFinding("warning", "Profit factor is unusually high for a small sample. That is often a sign of fragile or overfit behavior."))

    if cagr > 0 and max_drawdown > cagr * 1.5:
        findings.append(AuditFinding("warning", "Drawdown is large relative to CAGR. The return stream may not be as efficient as the headline result suggests."))
    if benchmark_metrics is not None and cagr < float(benchmark_metrics.get("benchmark_cagr", 0.0)):
        findings.append(AuditFinding("warning", "The strategy underperformed the benchmark on a CAGR basis. The added complexity may not be justified."))
    if exposure < 0.2:
        findings.append(AuditFinding("info", "The strategy spends long periods out of the market. Make sure the apparent edge is not just one short favorable regime."))
    if equity_curve.shape[0] > 10:
        daily_returns = equity_curve["equity"].pct_change().dropna()
        if not daily_returns.empty and daily_returns.std() < 0.002 and cagr > 0.1:
            findings.append(AuditFinding("critical", "The equity curve looks unusually smooth for the return level. Recheck data handling, execution assumptions, and leakage risk."))

    if strategy_parameters:
        parameter_count = len(strategy_parameters)
        if num_trades and parameter_count / max(num_trades, 1) > 0.3:
            findings.append(AuditFinding("warning", "There are many tunable parameters relative to the number of trades. Overfit risk is elevated."))
        if {"hma_length", "qqe_factor"}.issubset(set(strategy_parameters)):
            if num_trades < 20:
                findings.append(
                    AuditFinding(
                        "warning",
                        "QQE/HMA generated too few trades for a higher-parameter strategy. The added complexity needs a larger sample before it is credible.",
                    )
                )
            findings.append(
                AuditFinding(
                    "info",
                    "This strategy adapts legacy intraday QQE/HMA ideas to daily bars. Interpret positive results cautiously until intraday confirmation logic is tested separately.",
                )
            )

    if regime_comments:
        for comment in regime_comments:
            severity = "warning" if "worse" in comment.lower() or "most returns" in comment.lower() else "info"
            findings.append(AuditFinding(severity, comment))

    if not findings:
        findings.append(AuditFinding("info", "No major statistical red flags were detected, but this remains a backtest and still needs out-of-sample validation."))
    return findings


def generate_strategy_audit(
    metrics: dict[str, float | int],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    benchmark_metrics: dict[str, float] | None = None,
    strategy_parameters: dict | None = None,
    symbols: list[str] | None = None,
    regime_comments: list[str] | None = None,
) -> list[str]:
    """Return plain-English research warnings for a backtest."""
    return [
        finding.message
        for finding in generate_audit_findings(
            metrics,
            trades,
            equity_curve,
            benchmark_metrics=benchmark_metrics,
            strategy_parameters=strategy_parameters,
            regime_comments=regime_comments,
        )
    ]
