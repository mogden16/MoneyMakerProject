from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_lab.backtest.engine import BacktestConfig
from trading_lab.data.database import TradingLabDatabase
from trading_lab.paper.journal import calculate_realized_pnl, close_paper_trade_payload, create_paper_trade_payload, open_paper_trade_payload
from trading_lab.signals.scanner import evaluate_signal_quality, plan_trade_from_signal, scan_symbol_strategy
from trading_lab.strategies.breakout import BreakoutStrategy
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy
from trading_lab.strategies.rsi_mean_reversion import RSIMeanReversionStrategy


def make_bars(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="B")
    series = pd.Series(closes, index=timestamps, dtype=float)
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(closes),
            "symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": series.values,
            "high": series.values + 1,
            "low": series.values - 1,
            "close": series.values,
            "adj_close": series.values,
            "volume": [1000.0] * len(closes),
            "dividends": [0.0] * len(closes),
            "stock_splits": [0.0] * len(closes),
            "adjusted_flag": [True] * len(closes),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(closes),
        }
    )


def test_signal_scanner_output_shape_and_buy_detection():
    bars = make_bars("AAA", [10, 10, 9, 9, 12])
    result = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="Moving Average Crossover",
        strategy=MovingAverageCrossStrategy(fast_window=2, slow_window=3),
        config=BacktestConfig(initial_capital=10000.0),
        robustness_score=65,
        qualification_status="Possible candidate",
    )
    record = result.to_record()
    assert result.signal_type == "new_buy_signal"
    assert set(record) >= {
        "ticker",
        "strategy",
        "signal_type",
        "signal_date",
        "latest_close",
        "suggested_entry_reference",
        "suggested_stop",
        "suggested_target",
        "reward_risk_ratio",
        "explanation",
    }


def test_exit_signal_detection():
    bars = make_bars("AAA", [12, 12, 13, 13, 10])
    result = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="Moving Average Crossover",
        strategy=MovingAverageCrossStrategy(fast_window=2, slow_window=3),
        config=BacktestConfig(initial_capital=10000.0),
    )
    assert result.signal_type == "exit_signal"


def test_no_signal_behavior_and_empty_bars_edge_case():
    flat_bars = make_bars("AAA", [10, 10, 10, 10, 10, 10])
    result = scan_symbol_strategy(
        ticker="AAA",
        bars=flat_bars,
        strategy_name="Daily Breakout",
        strategy=BreakoutStrategy(lookback_window=2),
        config=BacktestConfig(initial_capital=10000.0),
    )
    assert result.signal_type == "no_signal"

    empty = scan_symbol_strategy(
        ticker="AAA",
        bars=pd.DataFrame(),
        strategy_name="QQE/HMA Daily",
        strategy=QQEHMAStrategy(),
        config=BacktestConfig(initial_capital=10000.0),
    )
    assert empty.signal_type == "no_signal"
    assert empty.signal_quality_label == "Ignore"


def test_signal_explanation_generation():
    bars = make_bars("AAA", [10, 11, 12, 13, 14, 15])
    result = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="Daily Breakout",
        strategy=BreakoutStrategy(lookback_window=2),
        config=BacktestConfig(initial_capital=10000.0),
    )
    assert "breakout" in result.explanation.lower()


def test_trade_plan_generation_and_sizing_modes():
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13])
    signal = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="Moving Average Crossover",
        strategy=MovingAverageCrossStrategy(fast_window=2, slow_window=3),
        config=BacktestConfig(initial_capital=10000.0),
    )
    fixed = plan_trade_from_signal(signal, portfolio_value=100000.0, sizing_method="fixed_dollar_allocation", sizing_value=5000.0)
    pct = plan_trade_from_signal(signal, portfolio_value=100000.0, sizing_method="percent_of_portfolio", sizing_value=0.1)
    risk = plan_trade_from_signal(signal, portfolio_value=100000.0, sizing_method="fixed_dollar_risk", sizing_value=500.0)
    assert fixed["position_size"] > 0
    assert pct["estimated_capital_required"] <= 100000.0 * 0.1 + signal.suggested_entry_reference
    assert risk["max_dollar_risk"] <= 500.0 + signal.risk_per_share


def test_signal_quality_score_behavior():
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13])
    strong = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="Moving Average Crossover",
        strategy=MovingAverageCrossStrategy(fast_window=2, slow_window=3),
        config=BacktestConfig(initial_capital=10000.0),
        robustness_score=80,
        qualification_status="Strong candidate",
    )
    weak_quality = evaluate_signal_quality(
        strong,
        {
            "trade_count": 5,
            "data_quality_warnings": ["gap warning"],
            "parameter_stability_poor": True,
        },
    )
    assert strong.signal_quality_score is not None
    assert weak_quality.label in {"Low quality", "Ignore", "Watch"}


def test_paper_trade_persistence_status_transitions_and_events(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "paper.duckdb"))
    plan = {
        "ticker": "AAA",
        "strategy": "Moving Average Crossover",
        "setup_date": pd.Timestamp("2024-01-10"),
        "planned_entry": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "risk_per_share": 5.0,
        "position_size": 10,
        "notes": "test plan",
        "tags": "journal",
        "linked_backtest_run_id": "run-1",
        "linked_qualification_id": "qual-1",
    }
    created = create_paper_trade_payload(plan)
    db.insert_paper_trade(created)
    db.insert_paper_trade_event(
        {
            "event_id": "e1",
            "paper_trade_id": created["paper_trade_id"],
            "created_at": pd.Timestamp("2024-01-10"),
            "event_type": "planned",
            "event_note": "created",
            "price": 100.0,
            "quantity": 10,
        }
    )
    opened = open_paper_trade_payload(created, actual_entry=101.0, entry_date=pd.Timestamp("2024-01-11"))
    db.update_paper_trade(opened)
    closed = close_paper_trade_payload(opened, exit_price=108.0, exit_date=pd.Timestamp("2024-01-15"), exit_reason="manual_close")
    db.update_paper_trade(closed)
    db.insert_paper_trade_event(
        {
            "event_id": "e2",
            "paper_trade_id": created["paper_trade_id"],
            "created_at": pd.Timestamp("2024-01-15"),
            "event_type": "closed",
            "event_note": "closed",
            "price": 108.0,
            "quantity": 10,
        }
    )
    saved = db.get_paper_trade(created["paper_trade_id"])
    events = db.read_paper_trade_events(created["paper_trade_id"])
    assert saved is not None
    assert saved["status"] == "closed"
    assert saved["realized_pnl"] == 70.0
    assert len(events) == 2


def test_realized_pnl_calculation():
    pnl, return_pct = calculate_realized_pnl(100.0, 110.0, 5)
    assert pnl == 50.0
    assert return_pct == pytest.approx(0.1)


def test_watchlist_persistence(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "watchlist.duckdb"))
    payload = {
        "watchlist_id": "AAPL",
        "ticker": "AAPL",
        "created_at": pd.Timestamp("2024-01-10"),
        "updated_at": pd.Timestamp("2024-01-10"),
        "notes": "watch earnings breakout",
        "tags": "tech,watch",
    }
    db.upsert_watchlist_item(payload)
    saved = db.list_watchlist(tag="tech")
    assert len(saved) == 1
    assert saved.iloc[0]["ticker"] == "AAPL"


def test_rsi_and_qqe_scanner_paths_do_not_crash():
    bars = make_bars("AAA", [100 + (index % 3) for index in range(260)])
    rsi_result = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="RSI Mean Reversion",
        strategy=RSIMeanReversionStrategy(),
        config=BacktestConfig(initial_capital=10000.0),
    )
    qqe_result = scan_symbol_strategy(
        ticker="AAA",
        bars=bars,
        strategy_name="QQE/HMA Daily",
        strategy=QQEHMAStrategy(),
        config=BacktestConfig(initial_capital=10000.0),
    )
    assert rsi_result.signal_type in {"new_buy_signal", "active_long_signal", "exit_signal", "no_signal"}
    assert qqe_result.signal_type in {"new_buy_signal", "active_long_signal", "exit_signal", "no_signal"}
