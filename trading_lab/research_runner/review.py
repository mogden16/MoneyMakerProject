from __future__ import annotations

import json
from typing import Any

import pandas as pd


def build_manual_review_shortlist(
    *,
    candidates: pd.DataFrame,
    model_comparison: pd.DataFrame,
    approved_breakdown: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or model_comparison.empty or approved_breakdown.empty:
        return pd.DataFrame()

    improving_models = model_comparison[
        (model_comparison["forward_return_edge_10d"] > 0)
        & (model_comparison["win_rate_edge"] > 0)
    ][["model_name", "forward_return_edge_10d", "win_rate_edge", "approval_rate"]]
    if improving_models.empty:
        return pd.DataFrame()

    shortlist = approved_breakdown.merge(improving_models, on="model_name", how="inner")
    candidates_for_merge = candidates.copy()
    for column in ["entry_parameters_json", "exit_parameters_json"]:
        if column in candidates_for_merge.columns:
            candidates_for_merge[column] = candidates_for_merge[column].apply(
                lambda value: json.dumps(value, default=str) if isinstance(value, dict) else value
            )
    shortlist = shortlist.merge(
        candidates_for_merge,
        on=[
            "timeframe",
            "entry_strategy_name",
            "entry_parameters_json",
            "exit_structure_name",
            "exit_parameters_json",
        ],
        how="left",
        suffixes=("", "_candidate"),
    )
    shortlist["manual_review_priority"] = shortlist.apply(_priority_score, axis=1)
    shortlist["manual_review_comment"] = shortlist.apply(_priority_comment, axis=1)
    shortlist = shortlist[
        (shortlist["approved_signal_count"] >= 20)
        & (shortlist["candidate_label"].isin(["Strong candidate", "Possible candidate"]))
    ].copy()
    if shortlist.empty:
        return pd.DataFrame()
    return shortlist.sort_values(
        [
            "manual_review_priority",
            "avg_forward_return_10d",
            "approved_signal_count",
            "profit_factor",
            "calmar",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def _priority_score(row: pd.Series) -> float:
    approved_count = float(row.get("approved_signal_count", 0.0) or 0.0)
    avg_forward_return = float(row.get("avg_forward_return_10d", 0.0) or 0.0)
    win_rate = float(row.get("win_rate", 0.0) or 0.0)
    forward_edge = float(row.get("forward_return_edge_10d", 0.0) or 0.0)
    win_edge = float(row.get("win_rate_edge", 0.0) or 0.0)
    calmar = float(row.get("calmar", 0.0) or 0.0)
    profit_factor = float(row.get("profit_factor", 0.0) or 0.0)
    red_flags = float(row.get("red_flag_count", 0.0) or 0.0)
    complexity = float(row.get("complexity_score", 0.0) or 0.0)
    return (
        (avg_forward_return * 1000.0)
        + (forward_edge * 800.0)
        + (win_rate * 20.0)
        + (win_edge * 10.0)
        + (min(approved_count, 250.0) * 0.02)
        + (calmar * 2.0)
        + (profit_factor * 1.5)
        - (red_flags * 1.5)
        - (complexity * 0.25)
    )


def _priority_comment(row: pd.Series) -> str:
    parts: list[str] = []
    if float(row.get("forward_return_edge_10d", 0.0) or 0.0) > 0:
        parts.append("model improved forward return")
    if float(row.get("win_rate_edge", 0.0) or 0.0) > 0:
        parts.append("model improved win rate")
    if float(row.get("approved_signal_count", 0.0) or 0.0) >= 100:
        parts.append("approval count is large enough to inspect")
    if str(row.get("candidate_label", "")) == "Strong candidate":
        parts.append("backtest candidate quality is strong")
    elif str(row.get("candidate_label", "")) == "Possible candidate":
        parts.append("backtest candidate quality is plausible")
    if bool(row.get("experimental", False)):
        parts.append("strategy remains experimental")
    return "; ".join(parts) if parts else "manual review needed"


def summarize_manual_review_shortlist(shortlist: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if shortlist.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in shortlist.head(limit).iterrows():
        rows.append(
            {
                "model_name": row["model_name"],
                "entry_strategy_name": row["entry_strategy_name"],
                "exit_structure_name": row["exit_structure_name"],
                "approved_signal_count": int(row["approved_signal_count"]),
                "avg_forward_return_10d": float(row["avg_forward_return_10d"]),
                "candidate_label": row["candidate_label"],
                "manual_review_comment": row["manual_review_comment"],
            }
        )
    return rows


def recommend_candidate_for_implementation(
    *,
    candidates: pd.DataFrame,
    model_comparison: pd.DataFrame,
    manual_review_shortlist: pd.DataFrame,
) -> dict[str, Any]:
    """Pick one practical implementation recommendation from the current research artifacts."""
    if not manual_review_shortlist.empty:
        top = manual_review_shortlist.iloc[0].to_dict()
        return {
            "source": "model_backed_shortlist",
            "timeframe": top.get("timeframe"),
            "entry_strategy_name": top.get("entry_strategy_name"),
            "strategy_archetype": top.get("strategy_archetype", top.get("entry_strategy_name")),
            "exit_structure_name": top.get("exit_structure_name"),
            "exit_archetype": top.get("exit_archetype", top.get("exit_structure_name")),
            "candidate_label": top.get("candidate_label"),
            "approved_signal_count": int(top.get("approved_signal_count", 0) or 0),
            "avg_forward_return_10d": float(top.get("avg_forward_return_10d", 0.0) or 0.0),
            "model_name": top.get("model_name"),
            "manual_review_comment": top.get("manual_review_comment", ""),
            "recommendation_reason": _implementation_reason(top, prefer_model_backed=True),
        }

    if candidates.empty:
        return {}

    frame = candidates.copy()
    frame = frame[frame["candidate_label"].isin(["Strong candidate", "Possible candidate"])].copy()
    if frame.empty:
        return {}
    frame["implementation_score"] = (
        frame["calmar"].fillna(0.0) * 20
        + frame["profit_factor"].fillna(0.0) * 8
        + frame["excess_cagr"].fillna(0.0) * 100
        + frame["drawdown_improvement"].fillna(0.0) * 30
        - frame["red_flag_count"].fillna(0.0) * 6
        - frame["complexity_score"].fillna(0.0) * 2
        - frame["experimental"].astype(int) * 8
    )
    top = frame.sort_values(
        ["implementation_score", "number_of_trades", "calmar"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    return {
        "source": "candidate_table",
        "timeframe": top.get("timeframe"),
        "entry_strategy_name": top.get("entry_strategy_name"),
        "strategy_archetype": top.get("strategy_archetype", top.get("entry_strategy_name")),
        "exit_structure_name": top.get("exit_structure_name"),
        "exit_archetype": top.get("exit_archetype", top.get("exit_structure_name")),
        "candidate_label": top.get("candidate_label"),
        "approved_signal_count": None,
        "avg_forward_return_10d": None,
        "model_name": None,
        "manual_review_comment": top.get("summary_comment", ""),
        "recommendation_reason": _implementation_reason(top, prefer_model_backed=False),
    }


def _implementation_reason(row: dict[str, Any] | pd.Series, *, prefer_model_backed: bool) -> str:
    timeframe = str(row.get("timeframe", "1d"))
    archetype = str(row.get("strategy_archetype", row.get("entry_strategy_name", "strategy")))
    exit_archetype = str(row.get("exit_archetype", row.get("exit_structure_name", "exit")))
    label = str(row.get("candidate_label", ""))
    if prefer_model_backed:
        return (
            f"Start with {archetype} using {exit_archetype} on {timeframe} because it survived both the "
            f"candidate screen and the model-backed shortlist."
        )
    if label == "Strong candidate":
        return f"Start with {archetype} using {exit_archetype} on {timeframe} because it is the strongest non-experimental candidate."
    return f"Watch {archetype} using {exit_archetype} on {timeframe} first; it is interesting but still needs caution."
