from __future__ import annotations

import pandas as pd


def calculate_profit_factor(closed_trades: pd.DataFrame) -> float:
    """Calculate profit factor from closed paper trades."""
    if closed_trades.empty or "realized_pnl" not in closed_trades.columns:
        return 0.0
    wins = closed_trades.loc[closed_trades["realized_pnl"] > 0, "realized_pnl"].sum()
    losses = abs(closed_trades.loc[closed_trades["realized_pnl"] < 0, "realized_pnl"].sum())
    return float(wins / losses) if losses else 0.0


def calculate_expectancy(closed_trades: pd.DataFrame) -> float:
    """Calculate expectancy per closed trade."""
    if closed_trades.empty or "realized_pnl" not in closed_trades.columns:
        return 0.0
    return float(closed_trades["realized_pnl"].mean())


def calculate_r_multiple(trade: pd.Series | dict[str, object]) -> float:
    """Calculate realized R multiple from planned dollar risk."""
    planned_entry = float((trade.get("planned_entry") if isinstance(trade, dict) else trade.get("planned_entry")) or 0.0)
    stop_loss = float((trade.get("stop_loss") if isinstance(trade, dict) else trade.get("stop_loss")) or 0.0)
    shares = float((trade.get("shares") if isinstance(trade, dict) else trade.get("shares")) or 0.0)
    realized_pnl = float((trade.get("realized_pnl") if isinstance(trade, dict) else trade.get("realized_pnl")) or 0.0)
    planned_risk = max((planned_entry - stop_loss) * shares, 0.0)
    if planned_risk <= 0:
        return 0.0
    return float(realized_pnl / planned_risk)


def planned_vs_actual_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Build a planned-vs-actual comparison view for paper trades."""
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    for column in ["actual_entry", "exit_price", "planned_entry", "stop_loss", "take_profit", "shares", "realized_pnl"]:
        if column not in frame.columns:
            frame[column] = 0.0
    frame["planned_dollar_risk"] = (frame["planned_entry"].fillna(0.0) - frame["stop_loss"].fillna(0.0)).clip(lower=0.0) * frame["shares"].fillna(0)
    frame["planned_reward_risk"] = (
        (frame["take_profit"].fillna(frame["planned_entry"]) - frame["planned_entry"].fillna(0.0)).clip(lower=0.0)
        / (frame["planned_entry"].fillna(0.0) - frame["stop_loss"].fillna(0.0)).replace(0, pd.NA)
    ).fillna(0.0)
    frame["realized_r_multiple"] = frame.apply(calculate_r_multiple, axis=1)
    frame["planned_vs_actual_entry_diff"] = frame["actual_entry"].fillna(frame["planned_entry"]) - frame["planned_entry"].fillna(0.0)
    frame["planned_stop_vs_actual_exit_diff"] = frame["exit_price"].fillna(0.0) - frame["stop_loss"].fillna(0.0)
    frame["planned_target_vs_actual_exit_diff"] = frame["exit_price"].fillna(0.0) - frame["take_profit"].fillna(0.0)
    return frame


def closed_trade_analytics(trades: pd.DataFrame) -> dict[str, object]:
    """Summarize closed paper-trade performance and grouping views."""
    if trades.empty:
        return {
            "summary": {
                "total_realized_pnl": 0.0,
                "realized_return_pct": 0.0,
                "win_rate": 0.0,
                "average_winning_trade": 0.0,
                "average_losing_trade": 0.0,
                "profit_factor": 0.0,
                "expectancy_per_trade": 0.0,
                "average_holding_period": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "average_reward_risk_planned": 0.0,
                "actual_result_vs_planned_rr": 0.0,
            },
            "by_strategy": pd.DataFrame(),
            "by_ticker": pd.DataFrame(),
            "by_universe": pd.DataFrame(),
            "by_tag": pd.DataFrame(),
            "by_signal_quality": pd.DataFrame(),
            "by_qualification_status": pd.DataFrame(),
            "by_robustness_bucket": pd.DataFrame(),
            "mistake_tags": pd.DataFrame(),
            "planned_vs_actual": pd.DataFrame(),
        }

    frame = planned_vs_actual_frame(trades.copy())
    closed = frame[frame["status"] == "closed"].copy()
    if closed.empty:
        return {
            "summary": {
                "total_realized_pnl": 0.0,
                "realized_return_pct": 0.0,
                "win_rate": 0.0,
                "average_winning_trade": 0.0,
                "average_losing_trade": 0.0,
                "profit_factor": 0.0,
                "expectancy_per_trade": 0.0,
                "average_holding_period": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "average_reward_risk_planned": 0.0,
                "actual_result_vs_planned_rr": 0.0,
            },
            "by_strategy": pd.DataFrame(),
            "by_ticker": pd.DataFrame(),
            "by_universe": pd.DataFrame(),
            "by_tag": pd.DataFrame(),
            "by_signal_quality": pd.DataFrame(),
            "by_qualification_status": pd.DataFrame(),
            "by_robustness_bucket": pd.DataFrame(),
            "mistake_tags": pd.DataFrame(),
            "planned_vs_actual": closed,
        }

    closed["holding_days"] = (pd.to_datetime(closed["exit_date"]) - pd.to_datetime(closed["entry_date"])).dt.days.fillna(0)
    closed["robustness_bucket"] = pd.cut(
        closed["linked_robustness_score"].fillna(-1),
        bins=[-1, 39, 59, 79, 100],
        labels=["weak", "unproven", "promising", "strong"],
    ).astype(str)

    def group_mean(column: str) -> pd.DataFrame:
        return closed.groupby(column, dropna=False)["realized_pnl"].agg(["count", "sum", "mean"]).reset_index()

    tag_rows: list[dict[str, object]] = []
    for _, row in closed.iterrows():
        for tag in [item.strip() for item in str(row.get("tags") or "").split(",") if item.strip()]:
            tag_rows.append({"tag": tag, "realized_pnl": row["realized_pnl"]})
    mistake_rows: list[dict[str, object]] = []
    for _, row in closed.iterrows():
        for tag in [item.strip() for item in str(row.get("mistake_tags") or "").split(",") if item.strip()]:
            mistake_rows.append({"mistake_tag": tag, "count": 1})

    summary = {
        "total_realized_pnl": float(closed["realized_pnl"].sum()),
        "realized_return_pct": float(closed["realized_return_pct"].mean()) if "realized_return_pct" in closed.columns else 0.0,
        "win_rate": float((closed["realized_pnl"] > 0).mean()),
        "average_winning_trade": float(closed.loc[closed["realized_pnl"] > 0, "realized_pnl"].mean() or 0.0),
        "average_losing_trade": float(closed.loc[closed["realized_pnl"] < 0, "realized_pnl"].mean() or 0.0),
        "profit_factor": calculate_profit_factor(closed),
        "expectancy_per_trade": calculate_expectancy(closed),
        "average_holding_period": float(closed["holding_days"].mean()),
        "best_trade": float(closed["realized_pnl"].max()),
        "worst_trade": float(closed["realized_pnl"].min()),
        "average_reward_risk_planned": float(closed["planned_reward_risk"].mean()),
        "actual_result_vs_planned_rr": float(closed["realized_r_multiple"].mean()),
    }
    return {
        "summary": summary,
        "by_strategy": group_mean("strategy_name"),
        "by_ticker": group_mean("ticker"),
        "by_universe": group_mean("universe_name"),
        "by_tag": pd.DataFrame(tag_rows).groupby("tag")["realized_pnl"].agg(["count", "sum", "mean"]).reset_index() if tag_rows else pd.DataFrame(),
        "by_signal_quality": group_mean("signal_quality_label"),
        "by_qualification_status": group_mean("qualification_status"),
        "by_robustness_bucket": group_mean("robustness_bucket"),
        "mistake_tags": pd.DataFrame(mistake_rows).groupby("mistake_tag")["count"].sum().reset_index().sort_values("count", ascending=False) if mistake_rows else pd.DataFrame(),
        "planned_vs_actual": closed,
    }
