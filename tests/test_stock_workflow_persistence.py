from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_lab.data.database import TradingLabDatabase
from trading_lab.paper.analytics import calculate_expectancy, calculate_profit_factor, calculate_r_multiple, closed_trade_analytics, planned_vs_actual_frame
from trading_lab.paper.journal import create_paper_trade_payload, update_post_trade_review


def sample_scanner_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scanner_result_id": "scan-1",
                "snapshot_id": "snapshot-1",
                "ticker": "AAPL",
                "strategy_name": "Moving Average Crossover",
                "signal_type": "new_buy_signal",
                "signal_date": pd.Timestamp("2024-01-10"),
                "latest_close": 100.0,
                "suggested_entry": 100.0,
                "suggested_stop": 95.0,
                "suggested_target": 110.0,
                "risk_per_share": 5.0,
                "reward_per_share": 10.0,
                "reward_risk_ratio": 2.0,
                "robustness_score": 72,
                "qualification_status": "Strong candidate",
                "signal_quality_score": 81,
                "signal_quality_label": "High quality",
                "explanation": "AAPL triggered a crossover entry.",
                "warnings_json": json.dumps(["None"]),
                "linked_paper_trade_id": None,
            },
            {
                "scanner_result_id": "scan-2",
                "snapshot_id": "snapshot-1",
                "ticker": "MSFT",
                "strategy_name": "Daily Breakout",
                "signal_type": "exit_signal",
                "signal_date": pd.Timestamp("2024-01-10"),
                "latest_close": 200.0,
                "suggested_entry": 200.0,
                "suggested_stop": 190.0,
                "suggested_target": 220.0,
                "risk_per_share": 10.0,
                "reward_per_share": 20.0,
                "reward_risk_ratio": 2.0,
                "robustness_score": 40,
                "qualification_status": "Not ready",
                "signal_quality_score": 30,
                "signal_quality_label": "Ignore",
                "explanation": "MSFT triggered an exit.",
                "warnings_json": json.dumps(["Weak setup"]),
                "linked_paper_trade_id": None,
            },
        ]
    )


def sample_paper_trade() -> dict[str, object]:
    return create_paper_trade_payload(
        {
            "ticker": "AAPL",
            "strategy": "Moving Average Crossover",
            "setup_date": pd.Timestamp("2024-01-10"),
            "planned_entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "position_size": 10,
            "notes": "test",
            "tags": "quality,tech",
            "linked_backtest_run_id": "run-1",
            "linked_qualification_id": "qual-1",
            "scanner_snapshot_id": "snapshot-1",
            "scanner_result_id": "scan-1",
            "quality_score": 81,
            "qualification_status": "Strong candidate",
            "signal_explanation": "AAPL triggered a crossover entry.",
            "signal_warnings_json": json.dumps(["None"]),
            "universe_name": "Large-cap tech",
        }
    )


def test_scanner_snapshot_persistence_and_saved_retrieval(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "scanner.duckdb"))
    db.replace_scanner_snapshot(
        {
            "snapshot_id": "snapshot-1",
            "created_at": pd.Timestamp("2024-01-10"),
            "universe_name": "Large-cap tech",
            "tickers": "AAPL,MSFT",
            "strategies": "Moving Average Crossover,Daily Breakout",
            "benchmark_symbol": "SPY",
            "price_mode": "adjusted_price_mode",
            "scanner_config_json": json.dumps({"refresh": False}),
            "notes": "daily open scan",
            "tags": "scanner,tech",
        },
        sample_scanner_results(),
    )
    listed = db.list_scanner_snapshots(tag="scanner")
    saved = db.get_scanner_snapshot("snapshot-1")
    results = db.read_scanner_snapshot_results("snapshot-1")
    assert len(listed) == 1
    assert saved is not None
    assert saved["universe_name"] == "Large-cap tech"
    assert len(results) == 2


def test_scanner_to_paper_trade_linking_and_action_status(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "linking.duckdb"))
    db.replace_scanner_snapshot(
        {
            "snapshot_id": "snapshot-1",
            "created_at": pd.Timestamp("2024-01-10"),
            "universe_name": "Large-cap tech",
            "tickers": "AAPL,MSFT",
            "strategies": "Moving Average Crossover,Daily Breakout",
            "benchmark_symbol": "SPY",
            "price_mode": "adjusted_price_mode",
            "scanner_config_json": "{}",
            "notes": "",
            "tags": "",
        },
        sample_scanner_results(),
    )
    trade = sample_paper_trade()
    db.insert_paper_trade(trade)
    db.update_scanner_snapshot_result_link("scan-1", trade["paper_trade_id"])
    linked = db.read_scanner_snapshot_results("snapshot-1", "planned")
    assert len(linked) == 1
    assert linked.iloc[0]["action_status"] == "planned"
    assert linked.iloc[0]["linked_paper_trade_id"] == trade["paper_trade_id"]


def test_paper_trade_analytics_profit_factor_expectancy_and_planned_vs_actual():
    trades = pd.DataFrame(
        [
            {
                "paper_trade_id": "1",
                "status": "closed",
                "planned_entry": 100.0,
                "stop_loss": 95.0,
                "take_profit": 110.0,
                "shares": 10,
                "actual_entry": 101.0,
                "exit_price": 108.0,
                "realized_pnl": 70.0,
                "realized_return_pct": 0.0693,
                "entry_date": pd.Timestamp("2024-01-11"),
                "exit_date": pd.Timestamp("2024-01-15"),
                "strategy_name": "Moving Average Crossover",
                "ticker": "AAPL",
                "universe_name": "Large-cap tech",
                "tags": "quality,tech",
                "signal_quality_label": "High quality",
                "qualification_status": "Strong candidate",
                "linked_robustness_score": 80,
                "mistake_tags": "exited early",
            },
            {
                "paper_trade_id": "2",
                "status": "closed",
                "planned_entry": 50.0,
                "stop_loss": 48.0,
                "take_profit": 56.0,
                "shares": 20,
                "actual_entry": 50.0,
                "exit_price": 47.0,
                "realized_pnl": -60.0,
                "realized_return_pct": -0.06,
                "entry_date": pd.Timestamp("2024-01-12"),
                "exit_date": pd.Timestamp("2024-01-13"),
                "strategy_name": "Daily Breakout",
                "ticker": "MSFT",
                "universe_name": "Large-cap tech",
                "tags": "breakout",
                "signal_quality_label": "Watch",
                "qualification_status": "Possible candidate",
                "linked_robustness_score": 62,
                "mistake_tags": "ignored warning",
            },
        ]
    )
    planned_actual = planned_vs_actual_frame(trades)
    analytics = closed_trade_analytics(trades)
    assert calculate_profit_factor(trades) > 1.0
    assert calculate_expectancy(trades) == 5.0
    assert calculate_r_multiple(trades.iloc[0]) > 1.0
    assert "realized_r_multiple" in planned_actual.columns
    assert analytics["summary"]["profit_factor"] > 1.0
    assert not analytics["by_strategy"].empty
    assert not analytics["mistake_tags"].empty


def test_post_trade_review_persistence_and_mistake_tags(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "review.duckdb"))
    trade = sample_paper_trade()
    trade["status"] = "closed"
    trade["actual_entry"] = 100.0
    trade["entry_date"] = pd.Timestamp("2024-01-11")
    trade["exit_date"] = pd.Timestamp("2024-01-12")
    trade["exit_price"] = 107.0
    trade["realized_pnl"] = 70.0
    trade["realized_return_pct"] = 0.07
    db.insert_paper_trade(trade)
    reviewed = update_post_trade_review(
        trade,
        {
            "thesis_review": "trend was intact",
            "execution_review": "entry was slightly late",
            "what_went_well": "held to target zone",
            "what_went_wrong": "chased the entry",
            "lesson_learned": "wait for pullback",
            "mistake_tags": "chased entry,poor reward risk",
            "followed_plan_flag": False,
            "entry_quality_rating": 2,
            "exit_quality_rating": 4,
            "emotional_discipline_rating": 3,
        },
    )
    db.update_paper_trade(reviewed)
    saved = db.get_paper_trade(trade["paper_trade_id"])
    assert saved is not None
    assert saved["thesis_review"] == "trend was intact"
    assert "chased entry" in (saved["mistake_tags"] or "")


def test_watchlist_category_persistence(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "watchlist_category.duckdb"))
    db.upsert_watchlist_item(
        {
            "watchlist_id": "AAPL",
            "ticker": "AAPL",
            "created_at": pd.Timestamp("2024-01-10"),
            "updated_at": pd.Timestamp("2024-01-10"),
            "category": "high priority",
            "notes": "watch breakout",
            "tags": "tech,watch",
        }
    )
    watchlist = db.list_watchlist(tag="tech")
    assert watchlist.iloc[0]["category"] == "high priority"


def test_scanner_history_query_helpers(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "scanner_history.duckdb"))
    db.replace_scanner_snapshot(
        {
            "snapshot_id": "snapshot-1",
            "created_at": pd.Timestamp("2024-01-10"),
            "universe_name": "Large-cap tech",
            "tickers": "AAPL,MSFT",
            "strategies": "Moving Average Crossover,Daily Breakout",
            "benchmark_symbol": "SPY",
            "price_mode": "adjusted_price_mode",
            "scanner_config_json": "{}",
            "notes": "",
            "tags": "",
        },
        sample_scanner_results(),
    )
    assert not db.scanner_history_summary().empty
    assert not db.scanner_history_by_ticker().empty
    assert not db.scanner_history_by_strategy_quality().empty


def test_no_scanner_snapshots_no_paper_trades_and_incomplete_trade_edge_cases(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "empty_edges.duckdb"))
    assert db.list_scanner_snapshots().empty
    assert db.read_scanner_snapshot_results("missing").empty
    assert db.list_paper_trades().empty
    empty_analytics = closed_trade_analytics(pd.DataFrame())
    assert empty_analytics["summary"]["profit_factor"] == 0.0
    incomplete = pd.DataFrame(
        [
            {
                "paper_trade_id": "1",
                "status": "closed",
                "planned_entry": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "shares": 0,
                "realized_pnl": 0.0,
                "realized_return_pct": 0.0,
                "strategy_name": "Moving Average Crossover",
                "ticker": "AAPL",
            }
        ]
    )
    planned_actual = planned_vs_actual_frame(incomplete)
    assert planned_actual.iloc[0]["realized_r_multiple"] == 0.0
