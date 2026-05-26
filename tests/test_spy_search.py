from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.data.database import TradingLabDatabase
from trading_lab.paper.forward_engine import build_active_paper_strategy_payload
from trading_lab.spy_lab import (
    SpySearchCombination,
    SpySearchEntryPreset,
    SpySearchExitPreset,
    build_spy_search_summary_comment,
    describe_spy_search_archetype,
    generate_approved_spy_entry_presets,
    generate_approved_spy_exit_presets,
    generate_spy_search_combinations,
    grade_spy_search_candidate,
    rank_spy_search_results,
    run_automated_spy_search,
)


def make_bars(symbol: str, closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="B")
    close_series = pd.Series(closes, index=timestamps, dtype=float)
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(closes),
            "symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": close_series.values,
            "high": close_series.values + 1.0,
            "low": close_series.values - 1.0,
            "close": close_series.values,
            "adj_close": close_series.values,
            "volume": [1000.0] * len(closes),
            "dividends": [0.0] * len(closes),
            "stock_splits": [0.0] * len(closes),
            "adjusted_flag": [True] * len(closes),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(closes),
        }
    )


def test_approved_spy_entry_and_exit_preset_generation():
    entries = generate_approved_spy_entry_presets()
    intraday_entries = generate_approved_spy_entry_presets("15m")
    experimental_entries = generate_approved_spy_entry_presets("5m")
    exits = generate_approved_spy_exit_presets()
    assert len(entries) == 14
    assert len(intraday_entries) == 5
    assert len(experimental_entries) == 2
    assert len(exits) == 22
    assert all(entry.entry_strategy_name for entry in entries)
    assert all(entry.timeframe == "15m" for entry in intraday_entries)
    assert "Opening Range Breakout" in {entry.entry_strategy_name for entry in intraday_entries}
    assert "SwingArm Trend" in {entry.entry_strategy_name for entry in intraday_entries}
    assert "partial_take_profit_plus_trailing_stop" not in {exit_preset.exit_structure_key for exit_preset in exits}


def test_spy_search_combination_generation():
    entries = generate_approved_spy_entry_presets()
    exits = generate_approved_spy_exit_presets()
    combinations = generate_spy_search_combinations()
    assert len(combinations) == len(entries) * len(exits)
    assert combinations[0].combination_id


def test_spy_search_candidate_grading_and_suspicious_detection():
    strong_label, strong_flags = grade_spy_search_candidate(
        {
            "number_of_trades": 30,
            "cagr": 0.12,
            "excess_cagr": 0.04,
            "max_drawdown": -0.12,
            "spy_max_drawdown": -0.22,
            "calmar": 0.8,
            "profit_factor": 1.3,
            "avg_r_multiple": 0.4,
            "exposure_pct": 0.6,
            "experimental": False,
        }
    )
    reject_label, reject_flags = grade_spy_search_candidate(
        {
            "number_of_trades": 0,
            "cagr": -0.02,
            "excess_cagr": -0.04,
            "max_drawdown": -0.30,
            "spy_max_drawdown": -0.22,
            "calmar": 0.0,
            "profit_factor": 0.8,
            "avg_r_multiple": -0.2,
            "exposure_pct": 1.0,
            "experimental": True,
        }
    )
    assert strong_label == "Strong candidate"
    assert strong_flags <= 2
    assert reject_label == "Reject"
    assert reject_flags >= 5


def test_spy_search_summary_comment_mentions_intraday_archetype():
    comment = build_spy_search_summary_comment(
        {
            "timeframe": "15m",
            "entry_strategy_name": "Opening Range Breakout",
            "entry_preset_label": "15m opening range + pressure",
            "exit_structure_name": "OCO bracket",
            "exit_preset_label": "OCO 3% / 6%",
            "number_of_trades": 12,
            "excess_cagr": 0.03,
            "drawdown_improvement": 0.05,
            "experimental": False,
        }
    )
    assert "opening-range breakout with volume-pressure confirmation" in comment
    assert "OCO bracket" in comment


def test_spy_search_archetype_description_handles_intraday_variants():
    archetype = describe_spy_search_archetype(
        {
            "entry_strategy_name": "Intraday QQE/HMA State",
            "entry_preset_label": "15m QQE/HMA + SwingArm",
        }
    )
    assert archetype == "QQE/HMA state filter"


def test_spy_search_ranking_categories():
    frame = pd.DataFrame(
        [
            {
                "result_id": "best",
                "entry_preset_label": "Trend filter 200",
                "exit_preset_label": "Signal exit only",
                "candidate_label": "Strong candidate",
                "cagr": 0.12,
                "excess_cagr": 0.04,
                "max_drawdown": -0.12,
                "drawdown_improvement": 0.10,
                "sharpe": 1.0,
                "sortino": 1.2,
                "calmar": 0.9,
                "profit_factor": 1.4,
                "number_of_trades": 25,
                "red_flag_count": 1,
                "robustness_score": 72,
                "complexity_score": 1,
                "experimental": False,
                "summary_comment": "Best",
            },
            {
                "result_id": "sus",
                "entry_preset_label": "QQE/HMA tight",
                "exit_preset_label": "Trailing 3%",
                "candidate_label": "Not ready",
                "cagr": 0.30,
                "excess_cagr": 0.18,
                "max_drawdown": -0.35,
                "drawdown_improvement": -0.12,
                "sharpe": 0.7,
                "sortino": 0.8,
                "calmar": 0.3,
                "profit_factor": 1.1,
                "number_of_trades": 4,
                "red_flag_count": 6,
                "robustness_score": 35,
                "complexity_score": 5,
                "experimental": True,
                "summary_comment": "Suspicious",
            },
        ]
    )
    highlights = rank_spy_search_results(frame)
    assert highlights["Best Overall"]["result_id"] == "best"
    assert highlights["Most Suspicious High Return"]["result_id"] == "sus"


def test_spy_search_result_output_shape_and_no_trade_handling(monkeypatch):
    entry = SpySearchEntryPreset(
        "trend_200",
        "trend_filter_200",
        "SPY 200-Day Trend Filter",
        "Trend filter 200",
        {"sma_length": 200},
        "Trend filter with a 200-day SMA.",
        1,
    )
    exit_preset = SpySearchExitPreset("signal_only", "signal_exit_only", "Signal exit only", "Signal exit only", {}, "Signal exit only.")
    monkeypatch.setattr(
        "trading_lab.spy_lab.generate_spy_search_combinations",
        lambda timeframe="1d": [SpySearchCombination("only", entry, exit_preset)],
    )
    engine = BacktestEngine(database=None)
    payload, results, highlights = run_automated_spy_search(
        engine=engine,
        data_by_symbol={"SPY": make_bars("SPY", [100.0] * 30)},
        timeframe="1d",
        start_date="2020-01-01",
        end_date="2020-03-01",
        price_mode="adjusted_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_sizing_value=1000.0,
        slippage_pct=0.0,
        commission_per_trade=0.0,
    )
    assert payload["total_combinations_tested"] == 1
    assert payload["timeframe"] == "1d"
    assert set(results.columns) >= {
        "result_id",
        "entry_strategy_name",
        "strategy_archetype",
        "exit_structure_name",
        "exit_archetype",
        "candidate_label",
        "summary_comment",
    }
    assert results.iloc[0]["candidate_label"] in {"Reject", "Not ready"}
    assert "Best Overall" in highlights


def test_spy_search_persistence_and_saved_retrieval(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "spy_search.duckdb"))
    payload = {
        "search_run_id": "search-1",
        "created_at": datetime.now(UTC).replace(tzinfo=None),
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "timeframe": "1d",
        "price_mode": "adjusted_price_mode",
        "initial_capital": 100000.0,
        "slippage_pct": 0.0005,
        "commission_per_trade": 1.0,
        "position_sizing_method": "percent_of_portfolio",
        "position_sizing_value": 1.0,
        "benchmark_symbol": "SPY",
        "total_combinations_tested": 2,
        "notes": "test search",
        "tags": "spy-only",
    }
    results = pd.DataFrame(
        [
            {
                "result_id": "result-1",
                "search_run_id": "search-1",
                "timeframe": "1d",
                "entry_strategy_name": "SPY 200-Day Trend Filter",
                "entry_parameters_json": {"sma_length": 200},
                "entry_preset_id": "trend_200",
                "entry_preset_label": "Trend filter 200",
                "exit_structure_key": "signal_exit_only",
                "exit_structure_name": "Signal exit only",
                "exit_parameters_json": {},
                "exit_preset_id": "signal_only",
                "exit_preset_label": "Signal exit only",
                "backtest_run_id": "run-1",
                "total_return": 0.2,
                "cagr": 0.1,
                "spy_cagr": 0.08,
                "excess_cagr": 0.02,
                "max_drawdown": -0.15,
                "spy_max_drawdown": -0.22,
                "drawdown_improvement": 0.07,
                "sharpe": 0.9,
                "sortino": 1.1,
                "calmar": 0.66,
                "number_of_trades": 20,
                "win_rate": 0.55,
                "profit_factor": 1.2,
                "avg_trade_return": 0.01,
                "avg_r_multiple": 0.2,
                "exposure_pct": 0.6,
                "robustness_score": 70,
                "candidate_label": "Strong candidate",
                "ranking_category": "Best Overall",
                "red_flag_count": 1,
                "summary_comment": "Good",
                "promoted_active_strategy_id": None,
                "experimental": False,
                "complexity_score": 1,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            }
        ]
    )
    db.replace_spy_strategy_search_run(payload, results)
    saved_runs = db.list_spy_strategy_search_runs(limit=10)
    assert not saved_runs.empty
    saved_payload = db.get_spy_strategy_search_run("search-1")
    assert saved_payload is not None
    saved_results = db.read_spy_strategy_search_results("search-1")
    assert not saved_results.empty
    assert saved_results.iloc[0]["entry_parameters_json"]["sma_length"] == 200


def test_spy_search_promotion_linkage(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "spy_search_promote.duckdb"))
    strategy_payload = build_active_paper_strategy_payload(
        strategy_name="SPY 200-Day Trend Filter",
        strategy_parameters={"sma_length": 200},
        universe_name="SPY Trading Workbench Automated Search",
        tickers=["SPY"],
        benchmark_symbol="SPY",
        price_mode="adjusted_price_mode",
        initial_capital=100000.0,
        position_sizing_method="percent_of_portfolio",
        position_sizing_value=1.0,
        max_positions=1,
        risk_settings={"fill_rule": "next_open"},
        slippage_pct=0.0,
        commission_per_trade=0.0,
        linked_search_run_id="search-1",
        linked_search_result_id="result-1",
        status="active",
    )
    db.insert_active_paper_strategy(strategy_payload)
    payload = {
        "search_run_id": "search-1",
        "created_at": datetime.now(UTC).replace(tzinfo=None),
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "timeframe": "1d",
        "price_mode": "adjusted_price_mode",
        "initial_capital": 100000.0,
        "slippage_pct": 0.0,
        "commission_per_trade": 0.0,
        "position_sizing_method": "percent_of_portfolio",
        "position_sizing_value": 1.0,
        "benchmark_symbol": "SPY",
        "total_combinations_tested": 1,
        "notes": "",
        "tags": "",
    }
    results = pd.DataFrame(
        [
            {
                "result_id": "result-1",
                "search_run_id": "search-1",
                "timeframe": "1d",
                "entry_strategy_name": "SPY 200-Day Trend Filter",
                "entry_parameters_json": {"sma_length": 200},
                "entry_preset_id": "trend_200",
                "entry_preset_label": "Trend filter 200",
                "exit_structure_key": "signal_exit_only",
                "exit_structure_name": "Signal exit only",
                "exit_parameters_json": {},
                "exit_preset_id": "signal_only",
                "exit_preset_label": "Signal exit only",
                "backtest_run_id": "run-1",
                "total_return": 0.2,
                "cagr": 0.1,
                "spy_cagr": 0.08,
                "excess_cagr": 0.02,
                "max_drawdown": -0.15,
                "spy_max_drawdown": -0.22,
                "drawdown_improvement": 0.07,
                "sharpe": 0.9,
                "sortino": 1.1,
                "calmar": 0.66,
                "number_of_trades": 20,
                "win_rate": 0.55,
                "profit_factor": 1.2,
                "avg_trade_return": 0.01,
                "avg_r_multiple": 0.2,
                "exposure_pct": 0.6,
                "robustness_score": 70,
                "candidate_label": "Strong candidate",
                "ranking_category": "Best Overall",
                "red_flag_count": 1,
                "summary_comment": "Good",
                "promoted_active_strategy_id": None,
                "experimental": False,
                "complexity_score": 1,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            }
        ]
    )
    db.replace_spy_strategy_search_run(payload, results)
    db.update_spy_strategy_search_result_promotion("result-1", strategy_payload["active_strategy_id"])
    saved_results = db.read_spy_strategy_search_results("search-1")
    assert saved_results.iloc[0]["promoted_active_strategy_id"] == strategy_payload["active_strategy_id"]
    active_strategy = db.get_active_paper_strategy(strategy_payload["active_strategy_id"])
    assert active_strategy is not None
    assert active_strategy["linked_search_run_id"] == "search-1"
    assert active_strategy["linked_search_result_id"] == "result-1"
