from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def profit_concentration_analysis(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty or trades["pnl"].sum() <= 0:
        return {
            "best_trade_profit_share": 0.0,
            "top_3_profit_share": 0.0,
            "top_5_profit_share": 0.0,
            "ticker_contribution": {},
            "year_contribution": {},
        }
    pnl_sum = float(trades["pnl"].sum())
    sorted_pnl = trades.sort_values("pnl", ascending=False)["pnl"]
    result = {
        "best_trade_profit_share": float(sorted_pnl.head(1).sum() / pnl_sum),
        "top_3_profit_share": float(sorted_pnl.head(3).sum() / pnl_sum),
        "top_5_profit_share": float(sorted_pnl.head(5).sum() / pnl_sum),
    }
    ticker_contribution = (trades.groupby("symbol")["pnl"].sum() / pnl_sum).sort_values(ascending=False)
    if "exit_timestamp" in trades.columns:
        trade_years = trades.copy()
        trade_years["year"] = pd.to_datetime(trade_years["exit_timestamp"]).dt.year
        year_contribution = (trade_years.groupby("year")["pnl"].sum() / pnl_sum).sort_values(ascending=False)
    else:
        year_contribution = pd.Series(dtype=float)
    result["ticker_contribution"] = {str(key): float(value) for key, value in ticker_contribution.items()}
    result["year_contribution"] = {str(key): float(value) for key, value in year_contribution.items()}
    return result


def parameter_stability_summary(sweep_results: pd.DataFrame, benchmark_metric_name: str = "Benchmark Total Return", drawdown_threshold: float = -0.25) -> dict[str, object]:
    if sweep_results.empty:
        return {
            "top_parameter_set": None,
            "median_result": {},
            "positive_return_pct": 0.0,
            "beating_benchmark_pct": 0.0,
            "drawdown_below_threshold_pct": 0.0,
            "best_cagr": 0.0,
            "median_cagr": 0.0,
            "worst_cagr": 0.0,
            "best_max_drawdown": 0.0,
            "median_max_drawdown": 0.0,
            "worst_max_drawdown": 0.0,
            "conclusion": "No sweep results are available.",
        }
    frame = sweep_results.copy()
    top_row = frame.iloc[0].to_dict()
    positive_return_pct = float((frame["Total Return"] > 0).mean())
    beating_benchmark_pct = float((frame.get("Excess CAGR", pd.Series(dtype=float)) > 0).mean()) if "Excess CAGR" in frame.columns else 0.0
    drawdown_pct = float((frame["Max Drawdown"] >= drawdown_threshold).mean())
    median_result = frame.median(numeric_only=True).to_dict()
    stable = positive_return_pct >= 0.6 and drawdown_pct >= 0.6 and frame["CAGR"].quantile(0.75) - frame["CAGR"].quantile(0.25) < 0.2
    conclusion = "This strategy appears stable across nearby parameters." if stable else "This strategy only works for a narrow parameter range and may be overfit."
    return {
        "top_parameter_set": top_row,
        "median_result": median_result,
        "positive_return_pct": positive_return_pct,
        "beating_benchmark_pct": beating_benchmark_pct,
        "drawdown_below_threshold_pct": drawdown_pct,
        "best_cagr": float(frame["CAGR"].max()),
        "median_cagr": float(frame["CAGR"].median()),
        "worst_cagr": float(frame["CAGR"].min()),
        "best_max_drawdown": float(frame["Max Drawdown"].max()),
        "median_max_drawdown": float(frame["Max Drawdown"].median()),
        "worst_max_drawdown": float(frame["Max Drawdown"].min()),
        "conclusion": conclusion,
    }


@dataclass
class RobustnessScore:
    score: int
    label: str
    strengths: list[str]
    red_flags: list[str]
    explanation_bullets: list[str]


def _score_label(score: int) -> str:
    if score >= 80:
        return "Strong but still needs review"
    if score >= 60:
        return "Promising"
    if score >= 40:
        return "Unproven"
    if score >= 20:
        return "Weak"
    return "Likely unreliable"


def compute_robustness_score(
    metrics: dict[str, float | int],
    *,
    train_test_summary: dict[str, object] | None = None,
    walk_forward_summary: dict[str, object] | None = None,
    concentration: dict[str, object] | None = None,
    regime_metrics: pd.DataFrame | None = None,
    parameter_stability: dict[str, object] | None = None,
) -> RobustnessScore:
    score = 50
    strengths: list[str] = []
    red_flags: list[str] = []
    bullets: list[str] = []

    num_trades = int(metrics.get("Number of Trades", 0) or 0)
    cagr = float(metrics.get("CAGR", 0.0) or 0.0)
    drawdown = abs(float(metrics.get("Max Drawdown", 0.0) or 0.0))
    excess_cagr = float(metrics.get("Excess CAGR", 0.0) or 0.0)

    if num_trades >= 30:
        score += 10
        strengths.append("Trade count is reasonably sized for a first-pass study.")
    else:
        score -= 12
        red_flags.append("Trade count is low, which makes the edge harder to trust.")

    if drawdown <= 0.2:
        score += 8
        strengths.append("Drawdown stayed contained.")
    elif drawdown > 0.4:
        score -= 12
        red_flags.append("Drawdown is severe relative to the return stream.")

    if excess_cagr > 0.03:
        score += 8
        strengths.append("The strategy beat the benchmark on a CAGR basis.")
    elif excess_cagr < 0:
        score -= 10
        red_flags.append("The strategy lagged the benchmark.")

    if concentration:
        top_trade_share = float(concentration.get("best_trade_profit_share", 0.0))
        top_5_share = float(concentration.get("top_5_profit_share", 0.0))
        if top_trade_share > 0.5:
            score -= 10
            red_flags.append("One trade contributed too much of total profit.")
        if top_5_share > 0.75:
            score -= 8
            red_flags.append("A small cluster of trades drove most profits.")

    if train_test_summary:
        degradation = float(train_test_summary.get("degradation", {}).get("CAGR", 0.0))
        if degradation < -0.05:
            score -= 12
            red_flags.append("Out-of-sample CAGR degraded materially from the train period.")
        else:
            score += 6
            strengths.append("Train/test degradation was not severe.")

    if walk_forward_summary:
        profitable_pct = float(walk_forward_summary.get("profitable_test_fold_pct", 0.0))
        consistency = float(walk_forward_summary.get("consistency_score", 0.0))
        if profitable_pct >= 0.6:
            score += 8
            strengths.append("A majority of walk-forward test folds were profitable.")
        else:
            score -= 10
            red_flags.append("Walk-forward test folds were not consistently profitable.")
        score += int((consistency - 0.5) * 20)

    if parameter_stability:
        pos_pct = float(parameter_stability.get("positive_return_pct", 0.0))
        if pos_pct >= 0.6:
            score += 6
            strengths.append("Nearby parameter sets produced positive results often enough to suggest some stability.")
        else:
            score -= 8
            red_flags.append("The parameter sweep looks fragile rather than broad-based.")

    if regime_metrics is not None and not regime_metrics.empty:
        trend = regime_metrics[regime_metrics["regime_type"] == "trend"]
        if trend.shape[0] >= 2:
            cagr_spread = float(trend["cagr"].max() - trend["cagr"].min())
            if cagr_spread > 0.2:
                score -= 8
                red_flags.append("Performance varied sharply across market regimes.")

    score = max(0, min(100, int(round(score))))
    bullets.append(f"Robustness Score is {score}/100. This is a research heuristic, not a prediction.")
    if cagr > 0:
        bullets.append(f"CAGR was {cagr:.1%} with drawdown {drawdown:.1%}.")
    if excess_cagr:
        bullets.append(f"Excess CAGR versus benchmark was {excess_cagr:.1%}.")
    return RobustnessScore(score=score, label=_score_label(score), strengths=strengths, red_flags=red_flags, explanation_bullets=bullets)
