from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.robustness import parameter_stability_summary


@dataclass
class OptionsOverlayCandidateAssessment:
    flag: bool
    label: str
    passed_checks: list[str]
    failed_checks: list[str]
    manual_review: list[str]

    @property
    def explanation_bullets(self) -> list[str]:
        bullets = [f"Candidate label: {self.label}."]
        bullets.extend(self.passed_checks)
        bullets.extend(self.failed_checks)
        bullets.extend(self.manual_review)
        return bullets


def evaluate_options_overlay_candidate(
    metrics: dict[str, float | int],
    *,
    robustness_score: int,
    concentration: dict[str, object] | None = None,
    train_test_summary: dict[str, object] | None = None,
    walk_forward_summary: dict[str, object] | None = None,
    parameter_stability: dict[str, object] | None = None,
) -> OptionsOverlayCandidateAssessment:
    """Rate whether a stock strategy is worth future options-overlay research."""
    concentration = concentration or {}
    passed: list[str] = []
    failed: list[str] = []
    review: list[str] = []

    num_trades = int(metrics.get("Number of Trades", 0) or 0)
    cagr = float(metrics.get("CAGR", 0.0) or 0.0)
    excess_cagr = float(metrics.get("Excess CAGR", 0.0) or 0.0)
    max_drawdown = abs(float(metrics.get("Max Drawdown", 0.0) or 0.0))
    benchmark_drawdown = abs(float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0))
    best_trade_share = float(concentration.get("best_trade_profit_share", 0.0) or 0.0)
    top_5_share = float(concentration.get("top_5_profit_share", 0.0) or 0.0)

    checks = {
        "trade_count": num_trades >= 30,
        "positive_cagr": cagr > 0,
        "positive_excess_cagr": excess_cagr > 0,
        "drawdown_vs_benchmark": benchmark_drawdown == 0.0 or max_drawdown <= benchmark_drawdown * 1.1,
        "robustness": robustness_score >= 60,
        "profit_concentration": best_trade_share <= 0.5 and top_5_share <= 0.75,
    }

    for passed_check, message in [
        (checks["trade_count"], f"Trade count reached {num_trades}, which is a more usable sample for later options overlay work."),
        (checks["positive_cagr"], f"CAGR remained positive at {cagr:.1%}."),
        (checks["positive_excess_cagr"], f"Excess CAGR versus benchmark remained positive at {excess_cagr:.1%}."),
        (checks["drawdown_vs_benchmark"], "Drawdown stayed at or below a benchmark-comparable range."),
        (checks["robustness"], f"Robustness Score stayed in the candidate range at {robustness_score}/100."),
        (checks["profit_concentration"], "Profit concentration was not dominated by one trade or a tiny cluster of trades."),
    ]:
        (passed if passed_check else failed).append(message)

    if train_test_summary:
        degradation = float(train_test_summary.get("degradation", {}).get("CAGR", 0.0) or 0.0)
        if degradation < -0.05:
            failed.append("Train/test degradation was severe enough to raise out-of-sample concerns.")
            checks["train_test"] = False
        else:
            passed.append("Train/test degradation was not severe.")
            checks["train_test"] = True

    if walk_forward_summary:
        profitable_pct = float(walk_forward_summary.get("profitable_test_fold_pct", 0.0) or 0.0)
        consistency = float(walk_forward_summary.get("consistency_score", 0.0) or 0.0)
        if profitable_pct >= 0.5 and consistency >= 0.5:
            passed.append("Walk-forward results were not obviously fragile.")
            checks["walk_forward"] = True
        else:
            failed.append("Walk-forward consistency was poor or too few test folds were profitable.")
            checks["walk_forward"] = False

    if parameter_stability:
        positive_pct = float(parameter_stability.get("positive_return_pct", 0.0) or 0.0)
        conclusion = str(parameter_stability.get("conclusion", ""))
        if positive_pct >= 0.5 and "stable" in conclusion.lower():
            passed.append("Parameter sweep stability was broad enough to avoid a narrow sweet spot warning.")
            checks["parameter_stability"] = True
        else:
            failed.append("Parameter sweep stability looked narrow enough to raise overfit risk.")
            checks["parameter_stability"] = False

    if num_trades < 50:
        review.append("Review whether later options expression would leave enough trades after stricter filters and option liquidity constraints.")
    if abs(float(metrics.get("Beta", 0.0) or 0.0)) > 1.5:
        review.append("Benchmark beta is high. Recheck whether the edge is mostly benchmark exposure.")
    if float(metrics.get("Exposure %", 0.0) or 0.0) < 0.2:
        review.append("Exposure is low. Make sure the strategy is not mostly one brief market regime.")

    hard_fail = not all(checks[key] for key in ["trade_count", "positive_cagr", "positive_excess_cagr", "robustness", "profit_concentration"])
    pass_count = sum(1 for value in checks.values() if value)
    if not hard_fail and pass_count >= 8:
        label = "Strong candidate"
        flag = True
    elif not hard_fail and pass_count >= 6:
        label = "Possible candidate"
        flag = True
    else:
        label = "Not ready"
        flag = False
    return OptionsOverlayCandidateAssessment(flag=flag, label=label, passed_checks=passed, failed_checks=failed, manual_review=review)


def run_slippage_sensitivity(
    engine: BacktestEngine,
    strategy_builders: dict[str, Any],
    data_by_symbol: dict[str, pd.DataFrame],
    base_config: BacktestConfig,
    benchmark_symbol: str,
    slippage_levels: list[float],
) -> pd.DataFrame:
    """Run a strategy set across multiple slippage assumptions."""
    rows: list[dict[str, object]] = []
    for strategy_name, builder in strategy_builders.items():
        for slippage_pct in slippage_levels:
            config = base_config.model_copy(update={"slippage_pct": float(slippage_pct)})
            result = engine.run(data_by_symbol=data_by_symbol, strategy=builder(), config=config, benchmark_symbol=benchmark_symbol)
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "slippage_pct": float(slippage_pct),
                    "CAGR": float(result.metrics.get("CAGR", 0.0) or 0.0),
                    "Max Drawdown": float(result.metrics.get("Max Drawdown", 0.0) or 0.0),
                    "Profit Factor": float(result.metrics.get("Profit Factor", 0.0) or 0.0),
                    "Number of Trades": int(result.metrics.get("Number of Trades", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_slippage_warnings(slippage_results: pd.DataFrame) -> list[str]:
    """Return plain-English slippage fragility warnings."""
    warnings: list[str] = []
    if slippage_results.empty:
        return warnings
    for strategy_name, group in slippage_results.groupby("strategy_name"):
        ordered = group.sort_values("slippage_pct")
        base_cagr = float(ordered["CAGR"].iloc[0] or 0.0)
        stressed = ordered[ordered["slippage_pct"] >= 0.0005]
        if stressed.empty or base_cagr == 0:
            continue
        stressed_cagr = float(stressed["CAGR"].iloc[0] or 0.0)
        if stressed_cagr < base_cagr * 0.5:
            warnings.append(f"{strategy_name} loses more than half of its CAGR under modest slippage. Treat the edge cautiously.")
    return warnings


def summarize_saved_sweep_stability(results_by_strategy: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Compare sweep stability across strategies using saved sweep results."""
    rows: list[dict[str, object]] = []
    for strategy_name, result_frames in results_by_strategy.items():
        combined = pd.concat([frame for frame in result_frames if not frame.empty], ignore_index=True) if result_frames else pd.DataFrame()
        if combined.empty:
            continue
        normalized = combined.rename(
            columns={
                "cagr": "CAGR",
                "max_drawdown": "Max Drawdown",
                "total_return": "Total Return",
            }
        )
        if "CAGR" not in normalized.columns:
            continue
        summary = parameter_stability_summary(normalized)
        best_cagr = float(summary.get("best_cagr", 0.0) or 0.0)
        median_cagr = float(summary.get("median_cagr", 0.0) or 0.0)
        worst_cagr = float(summary.get("worst_cagr", 0.0) or 0.0)
        rows.append(
            {
                "strategy_name": strategy_name,
                "percent_profitable": float(summary.get("positive_return_pct", 0.0) or 0.0),
                "percent_beating_benchmark": float(summary.get("beating_benchmark_pct", 0.0) or 0.0),
                "median_cagr": median_cagr,
                "median_max_drawdown": float(summary.get("median_max_drawdown", 0.0) or 0.0),
                "best_to_median_gap": best_cagr - median_cagr,
                "best_to_worst_gap": best_cagr - worst_cagr,
                "stability_label": "Stable" if "stable" in str(summary.get("conclusion", "")).lower() else "Fragile",
                "warning": "Best result is much better than the median result." if best_cagr - median_cagr > 0.1 else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_label", "median_cagr"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame()
