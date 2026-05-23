from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_lab.backtest.audit import generate_audit_findings
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.qualification import evaluate_options_overlay_candidate, run_slippage_sensitivity, summarize_saved_sweep_stability
from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.universes import get_universe_tickers, list_universe_names, normalize_ticker_list
from trading_lab.strategies.base import StrategyBase
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy


class FlatStrategy(StrategyBase):
    name = "flat"

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = False
        frame["exit_signal"] = False
        return frame


class SingleTradeStrategy(StrategyBase):
    name = "single_trade"

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = False
        frame["exit_signal"] = False
        if len(frame) > 3:
            frame.loc[1, "entry_signal"] = True
            frame.loc[3, "exit_signal"] = True
        return frame


def make_bars(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
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


def test_predefined_universes_and_custom_normalization():
    names = list_universe_names()
    assert "Large-cap tech" in names
    assert get_universe_tickers("Broad ETFs") == ["SPY", "QQQ", "IWM", "DIA"]
    assert normalize_ticker_list(" spy, qqq ,SPY , aapl ") == ["SPY", "QQQ", "AAPL"]


def test_strategy_qualification_persistence_and_saved_retrieval(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "qualification.duckdb"))
    payload = {
        "qualification_id": "qual-1",
        "created_at": pd.Timestamp("2024-01-10"),
        "universe_name": "Large-cap tech",
        "tickers": "AAPL,MSFT,NVDA",
        "benchmark_symbol": "SPY",
        "start_date": pd.Timestamp("2020-01-01").date(),
        "end_date": pd.Timestamp("2024-01-01").date(),
        "price_mode": "adjusted_price_mode",
        "initial_capital": 100000.0,
        "risk_settings_json": json.dumps({"stop_loss_pct": 0.08}),
        "notes": "qualification note",
        "tags": "options-candidate,tech",
    }
    results = pd.DataFrame(
        [
            {
                "qualification_result_id": "res-1",
                "qualification_id": "qual-1",
                "strategy_name": "Moving Average Crossover",
                "backtest_run_id": "run-1",
                "total_return": 0.15,
                "cagr": 0.12,
                "max_drawdown": -0.1,
                "sharpe": 1.1,
                "sortino": 1.3,
                "calmar": 1.2,
                "win_rate": 0.55,
                "profit_factor": 1.4,
                "number_of_trades": 40,
                "exposure_pct": 0.5,
                "excess_cagr": 0.03,
                "robustness_score": 68,
                "red_flag_count": 1,
                "options_candidate_flag": True,
                "candidate_label": "Possible candidate",
                "candidate_explanation_json": json.dumps(["Candidate label: Possible candidate."]),
                "created_at": pd.Timestamp("2024-01-10"),
            }
        ]
    )
    db.replace_strategy_qualification_run(payload, results)

    listed = db.list_strategy_qualification_runs(tag="tech")
    saved = db.get_strategy_qualification_run("qual-1")
    detail = db.read_strategy_qualification_results("qual-1")

    assert len(listed) == 1
    assert saved is not None
    assert saved["notes"] == "qualification note"
    assert detail.iloc[0]["strategy_name"] == "Moving Average Crossover"


def test_options_overlay_candidate_flag_logic():
    assessment = evaluate_options_overlay_candidate(
        {
            "Number of Trades": 50,
            "CAGR": 0.18,
            "Excess CAGR": 0.05,
            "Max Drawdown": -0.15,
            "Benchmark Max Drawdown": -0.2,
            "Beta": 0.8,
            "Exposure %": 0.6,
        },
        robustness_score=72,
        concentration={"best_trade_profit_share": 0.2, "top_5_profit_share": 0.45},
        train_test_summary={"degradation": {"CAGR": -0.01}},
        walk_forward_summary={"profitable_test_fold_pct": 0.7, "consistency_score": 0.7},
        parameter_stability={"positive_return_pct": 0.7, "conclusion": "This strategy appears stable across nearby parameters."},
    )
    assert assessment.flag is True
    assert assessment.label in {"Strong candidate", "Possible candidate"}


def test_no_trade_candidate_edge_case_is_not_ready():
    assessment = evaluate_options_overlay_candidate(
        {
            "Number of Trades": 0,
            "CAGR": 0.0,
            "Excess CAGR": -0.02,
            "Max Drawdown": 0.0,
            "Benchmark Max Drawdown": -0.1,
        },
        robustness_score=25,
        concentration={},
    )
    assert assessment.flag is False
    assert assessment.label == "Not ready"


def test_slippage_sensitivity_calculations():
    engine = BacktestEngine(database=None)
    bars = make_bars("AAA", [10, 11, 12, 13, 14, 15, 16, 17])
    results = run_slippage_sensitivity(
        engine,
        {"single_trade": lambda: SingleTradeStrategy()},
        {"AAA": bars},
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        "SPY",
        [0.0, 0.001, 0.0025],
    )
    assert list(results["slippage_pct"]) == [0.0, 0.001, 0.0025]
    assert set(results.columns) >= {"strategy_name", "CAGR", "Max Drawdown", "Profit Factor", "Number of Trades"}


def test_parameter_stability_comparison_helper():
    summary = summarize_saved_sweep_stability(
        {
            "Moving Average Crossover": [
                pd.DataFrame(
                    {
                        "cagr": [0.12, 0.1, -0.02],
                        "max_drawdown": [-0.1, -0.12, -0.3],
                        "total_return": [0.15, 0.12, -0.05],
                        "Excess CAGR": [0.03, 0.01, -0.07],
                    }
                )
            ]
        }
    )
    assert not summary.empty
    assert summary.iloc[0]["strategy_name"] == "Moving Average Crossover"
    assert "stability_label" in summary.columns


def test_qqe_hma_audit_warning_for_low_trade_count():
    bars = make_bars("AAA", [100 + index for index in range(12)])
    strategy = QQEHMAStrategy(hma_length=5, rsi_length=5, rsi_smoothing=3, qqe_factor=4.236, atr_smoothing=3)
    signals = strategy.generate_signals(bars)
    findings = generate_audit_findings(
        {"Number of Trades": 5, "Win Rate": 0.6, "CAGR": 0.1, "Max Drawdown": -0.05, "Exposure %": 0.2},
        pd.DataFrame(columns=["symbol", "pnl"]),
        pd.DataFrame({"equity": [100000, 101000], "timestamp": pd.date_range("2024-01-01", periods=2, freq="B")}),
        strategy_parameters=strategy.parameters(),
    )
    assert any("QQE/HMA generated too few trades" in finding.message for finding in findings)
    assert signals.shape[0] == bars.shape[0]


def test_research_dashboard_rows_include_candidate_flag(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "dashboard.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    benchmark = make_bars("SPY", [100 + index for index in range(len(bars))])
    result = engine.run(
        {"AAA": bars, "SPY": benchmark},
        MovingAverageCrossStrategy(fast_window=2, slow_window=3),
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        benchmark_symbol="SPY",
    )
    db.replace_robustness_score(
        {
            "run_id": result.run_id,
            "score": 70,
            "label": "Promising",
            "strengths_json": "[]",
            "red_flags_json": "[]",
            "explanation_bullets_json": "[]",
            "created_at": pd.Timestamp("2024-01-10"),
        }
    )
    dashboard = db.get_research_dashboard_rows()
    assert "options_candidate_flag" in dashboard.columns
    assert "options_candidate_label" in dashboard.columns


def test_empty_saved_qualification_queries(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "empty_qualification.duckdb"))
    assert db.list_strategy_qualification_runs().empty
    assert db.read_strategy_qualification_results("missing").empty
