from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def create_paper_trade_payload(plan: dict[str, object]) -> dict[str, object]:
    """Create a new planned paper trade payload from a trade plan."""
    now = datetime.now(UTC).replace(tzinfo=None)
    return {
        "paper_trade_id": str(uuid4()),
        "created_at": now,
        "updated_at": now,
        "ticker": plan["ticker"],
        "strategy_name": plan["strategy"],
        "signal_date": plan["setup_date"],
        "planned_entry": plan["planned_entry"],
        "actual_entry": None,
        "stop_loss": plan["stop_loss"],
        "take_profit": plan["take_profit"],
        "shares": int(plan["position_size"]),
        "status": "planned",
        "entry_date": None,
        "exit_date": None,
        "exit_price": None,
        "exit_reason": None,
        "realized_pnl": 0.0,
        "realized_return_pct": 0.0,
        "notes": str(plan.get("notes", "")),
        "tags": str(plan.get("tags", "")),
        "linked_backtest_run_id": plan.get("linked_backtest_run_id"),
        "linked_qualification_id": plan.get("linked_qualification_id"),
        "scanner_snapshot_id": plan.get("scanner_snapshot_id"),
        "scanner_result_id": plan.get("scanner_result_id"),
        "signal_quality_score": plan.get("quality_score"),
        "qualification_status": plan.get("qualification_status"),
        "signal_explanation": plan.get("signal_explanation"),
        "signal_warnings_json": plan.get("signal_warnings_json", "[]"),
        "thesis_review": "",
        "execution_review": "",
        "what_went_well": "",
        "what_went_wrong": "",
        "lesson_learned": "",
        "mistake_tags": "",
        "followed_plan_flag": None,
        "entry_quality_rating": None,
        "exit_quality_rating": None,
        "emotional_discipline_rating": None,
        "universe_name": plan.get("universe_name"),
    }


def calculate_realized_pnl(actual_entry: float, exit_price: float, shares: int) -> tuple[float, float]:
    """Calculate realized paper-trade PnL and percentage return."""
    if shares <= 0 or actual_entry <= 0:
        return 0.0, 0.0
    pnl = (exit_price - actual_entry) * shares
    return_pct = exit_price / actual_entry - 1
    return float(pnl), float(return_pct)


def open_paper_trade_payload(existing: dict[str, object], *, actual_entry: float, entry_date) -> dict[str, object]:
    """Transition a planned trade to open."""
    now = datetime.now(UTC).replace(tzinfo=None)
    updated = dict(existing)
    updated["updated_at"] = now
    updated["actual_entry"] = float(actual_entry)
    updated["entry_date"] = entry_date
    updated["status"] = "open"
    return updated


def close_paper_trade_payload(existing: dict[str, object], *, exit_price: float, exit_date, exit_reason: str) -> dict[str, object]:
    """Transition an open trade to closed and calculate realized PnL."""
    now = datetime.now(UTC).replace(tzinfo=None)
    actual_entry = float(existing.get("actual_entry") or existing.get("planned_entry") or 0.0)
    shares = int(existing.get("shares") or 0)
    pnl, return_pct = calculate_realized_pnl(actual_entry, float(exit_price), shares)
    updated = dict(existing)
    updated["updated_at"] = now
    updated["exit_date"] = exit_date
    updated["exit_price"] = float(exit_price)
    updated["exit_reason"] = exit_reason
    updated["status"] = "closed"
    updated["realized_pnl"] = pnl
    updated["realized_return_pct"] = return_pct
    return updated


def update_post_trade_review(existing: dict[str, object], review: dict[str, object]) -> dict[str, object]:
    """Apply a structured post-trade review to an existing paper trade payload."""
    updated = dict(existing)
    updated["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
    for key in [
        "thesis_review",
        "execution_review",
        "what_went_well",
        "what_went_wrong",
        "lesson_learned",
        "mistake_tags",
        "followed_plan_flag",
        "entry_quality_rating",
        "exit_quality_rating",
        "emotional_discipline_rating",
    ]:
        if key in review:
            updated[key] = review[key]
    return updated
