from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_lab.backtest.benchmark import evaluate_benchmark_diagnostics
from trading_lab.backtest.corporate_actions import summarize_corporate_action_warnings
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.sweep import run_parameter_sweep
from trading_lab.data.database import TradingLabDatabase
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.base import StrategyBase


class SignalStrategy(StrategyBase):
    name = "signal_strategy"

    def __init__(self, entries: list[bool], exits: list[bool]) -> None:
        self.entries = entries
        self.exits = exits

    def parameters(self) -> dict:
        return {"entries": self.entries, "exits": self.exits}

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = self.entries[: len(frame)]
        frame["exit_signal"] = self.exits[: len(frame)]
        return frame


def make_bars(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="B")
    closes_series = pd.Series(closes, index=timestamps, dtype=float)
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(closes),
            "symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": closes_series.values,
            "high": closes_series.values + 1.0,
            "low": closes_series.values - 1.0,
            "close": closes_series.values,
            "adj_close": closes_series.values,
            "volume": [1000.0] * len(closes),
            "dividends": [0.0] * len(closes),
            "stock_splits": [0.0] * len(closes),
            "adjusted_flag": [False] * len(closes),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(closes),
        }
    )


def test_sweep_tables_persist_and_link_runs(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "sweeps.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13, 14, 15])

    sweep_id, results = run_parameter_sweep(
        engine,
        lambda params: MovingAverageCrossStrategy(**params),
        {"AAA": bars},
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        {"fast_window": [2], "slow_window": [3, 4]},
        benchmark_symbol="SPY",
        strategy_name="Moving Average Crossover",
        notes="sweep note",
        tags="momentum,promising",
    )

    sweep_run = db.get_sweep_run(sweep_id)
    sweep_results = db.read_sweep_results(sweep_id)
    sweep_parameters = db.read_sweep_parameters(sweep_id)

    assert sweep_run is not None
    assert sweep_run["notes"] == "sweep note"
    assert "promising" in (sweep_run["tags"] or "")
    assert len(sweep_results) == len(results)
    assert set(sweep_parameters["parameter_name"]) == {"fast_window", "slow_window"}
    linked_run_id = sweep_results.iloc[0]["backtest_run_id"]
    assert db.get_backtest_run(linked_run_id)["sweep_id"] == sweep_id


def test_saved_sweep_retrieval_and_tag_filtering(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "saved_sweeps.duckdb"))
    payload = {
        "sweep_id": "sweep-1",
        "created_at": pd.Timestamp("2024-01-10"),
        "strategy_name": "Daily Breakout",
        "tickers": "AAA,BBB",
        "start_date": pd.Timestamp("2024-01-01").date(),
        "end_date": pd.Timestamp("2024-02-01").date(),
        "benchmark_symbol": "SPY",
        "initial_capital": 100000.0,
        "price_mode": "adjusted_price_mode",
        "position_sizing_method": "fixed_dollar",
        "risk_settings_json": json.dumps({"stop_loss_pct": 0.08}),
        "sweep_config_json": json.dumps({"param_grid": {"lookback_window": [10, 20]}}),
        "notes": "candidate for later options overlay",
        "tags": "options-candidate,breakout",
    }
    results = pd.DataFrame(
        [
            {
                "sweep_result_id": "result-1",
                "sweep_id": "sweep-1",
                "backtest_run_id": "run-1",
                "parameter_json": json.dumps({"lookback_window": 10}),
                "total_return": 0.12,
                "cagr": 0.11,
                "max_drawdown": -0.08,
                "sharpe": 1.0,
                "sortino": 1.2,
                "calmar": 1.3,
                "win_rate": 0.5,
                "profit_factor": 1.4,
                "number_of_trades": 12,
                "exposure_pct": 0.4,
                "robustness_score": 67,
                "beats_benchmark_flag": True,
                "created_at": pd.Timestamp("2024-01-10"),
            }
        ]
    )
    params = pd.DataFrame([{"sweep_id": "sweep-1", "parameter_name": "lookback_window", "parameter_values_json": "[10,20]"}])
    db.replace_sweep_run(payload, results, params)

    filtered = db.list_sweep_runs(tag="options-candidate")
    assert len(filtered) == 1
    assert filtered.iloc[0]["sweep_id"] == "sweep-1"
    assert not db.read_sweep_results("sweep-1").empty


def test_research_dashboard_rows_and_tag_filter(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "dashboard.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13])
    strategy = SignalStrategy(entries=[False, True, False, False, False, False], exits=[False, False, False, True, False, False])
    result = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0))
    db.update_backtest_run_annotations(result.run_id, "Testing RSI strategy on large-cap tech", "promising,tech")
    db.replace_robustness_score(
        {
            "run_id": result.run_id,
            "score": 72,
            "label": "Promising",
            "strengths_json": "[]",
            "red_flags_json": "[]",
            "explanation_bullets_json": "[]",
            "created_at": pd.Timestamp("2024-01-10"),
        }
    )
    db.replace_benchmark_diagnostics(
        result.run_id,
        {
            "run_id": result.run_id,
            "benchmark_symbol": "SPY",
            "coverage_ratio": 1.0,
            "missing_session_count": 0,
            "dropped_strategy_dates": 0,
            "zero_return_days": 0,
            "status": "fresh",
            "warnings_json": "[]",
            "created_at": pd.Timestamp("2024-01-10"),
        },
    )

    tagged_runs = db.list_backtest_runs(tag="tech")
    dashboard = db.get_research_dashboard_rows()

    assert len(tagged_runs) == 1
    assert tagged_runs.iloc[0]["run_id"] == result.run_id
    assert not dashboard.empty
    assert dashboard.iloc[0]["robustness_score"] == 72


def test_benchmark_diagnostics_warn_on_missing_data_and_date_mismatch():
    strategy_equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=5, freq="B"),
            "equity": [100000, 100500, 101000, 101500, 102000],
        }
    )
    benchmark_bars = make_bars("SPY", [100, 100, 100, 100], start="2024-01-02").iloc[[0, 2, 3]].reset_index(drop=True)
    benchmark_curve = pd.DataFrame(
        {
            "run_id": ["run"] * 3,
            "benchmark_symbol": ["SPY"] * 3,
            "timestamp": benchmark_bars["timestamp"],
            "benchmark_equity": [100000, 100000, 100000],
        }
    )

    diagnostics = evaluate_benchmark_diagnostics(strategy_equity, benchmark_bars, benchmark_curve, "SPY")

    assert diagnostics.status == "warning"
    assert diagnostics.coverage_ratio < 1.0
    assert diagnostics.missing_session_count >= 1
    assert diagnostics.warnings


def test_benchmark_diagnostics_warn_on_missing_benchmark():
    strategy_equity = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=3, freq="B"), "equity": [100000, 100500, 101000]})
    diagnostics = evaluate_benchmark_diagnostics(strategy_equity, pd.DataFrame(), pd.DataFrame(), "SPY")
    assert diagnostics.status == "critical"
    assert diagnostics.warnings


def test_corporate_action_warning_modes():
    actions = pd.DataFrame(
        [
            {"symbol": "AAA", "action_type": "split", "cash_amount": None},
            {"symbol": "AAA", "action_type": "dividend", "cash_amount": 1.25},
        ]
    )
    raw_warnings = summarize_corporate_action_warnings(actions, price_mode="raw_price_mode", adjusted_available=True)
    adjusted_warnings = summarize_corporate_action_warnings(actions, price_mode="adjusted_price_mode", adjusted_available=False)

    assert any("split" in warning.lower() for warning in raw_warnings)
    assert any("adjusted close" in warning.lower() for warning in adjusted_warnings)


def test_empty_saved_runs_and_sweeps_queries(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "empty.duckdb"))
    assert db.list_backtest_runs().empty
    assert db.list_sweep_runs().empty
    assert db.read_sweep_results("missing").empty
