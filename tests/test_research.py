from pathlib import Path

import pandas as pd

from trading_lab.backtest.audit import generate_audit_findings
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.regime import classify_market_regimes, compute_regime_metrics
from trading_lab.backtest.robustness import compute_robustness_score, parameter_stability_summary, profit_concentration_analysis
from trading_lab.backtest.walk_forward import generate_walk_forward_folds, run_walk_forward_analysis
from trading_lab.data.database import TradingLabDatabase
from trading_lab.strategies.base import StrategyBase


class AlwaysFlatStrategy(StrategyBase):
    name = "always_flat"

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = False
        frame["exit_signal"] = False
        return frame


class EveryMonthStrategy(StrategyBase):
    name = "every_month"

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = False
        frame["exit_signal"] = False
        if len(frame) > 1:
            frame.loc[1::20, "entry_signal"] = True
            frame.loc[10::20, "exit_signal"] = True
        return frame


def make_long_bars(symbol: str, periods: int = 320, start: str = "2020-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="B")
    closes = pd.Series(range(periods), index=timestamps, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * periods,
            "symbol": [symbol] * periods,
            "timeframe": ["1d"] * periods,
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": closes.values,
            "high": closes.values + 1,
            "low": closes.values - 1,
            "close": closes.values,
            "adj_close": closes.values,
            "volume": [1000.0] * periods,
            "dividends": [0.0] * periods,
            "stock_splits": [0.0] * periods,
            "adjusted_flag": [False] * periods,
            "retrieved_at": [pd.Timestamp("2024-01-01")] * periods,
        }
    )


def test_generate_walk_forward_folds():
    folds = generate_walk_forward_folds("2020-01-01", "2021-12-31", train_window_months=6, test_window_months=3, step_months=3)
    assert len(folds) >= 4
    assert folds[0].train_start == pd.Timestamp("2020-01-01")


def test_run_walk_forward_analysis():
    engine = BacktestEngine(database=None)
    bars = make_long_bars("AAA", periods=420)
    walk_id, folds, summary = run_walk_forward_analysis(
        engine,
        EveryMonthStrategy(),
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        {"AAA": bars},
        "SPY",
        train_window_months=6,
        test_window_months=3,
        step_months=3,
    )
    assert walk_id
    assert not folds.empty
    assert "average_test_cagr" in summary


def test_regime_classification_and_metrics():
    benchmark = make_long_bars("SPY", periods=260)
    benchmark.loc[220:, "close"] = benchmark.loc[220:, "close"] - 80
    benchmark["adj_close"] = benchmark["close"]
    classification = classify_market_regimes(benchmark)
    assert {"bull", "bear"}.issubset(set(classification.frame["trend_regime"]))
    equity = pd.DataFrame({"timestamp": benchmark["timestamp"], "equity": 100000 + benchmark["close"].cumsum(), "positions_value": 1})
    trades = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "entry_timestamp": [benchmark["timestamp"].iloc[50], benchmark["timestamp"].iloc[230]],
            "exit_timestamp": [benchmark["timestamp"].iloc[60], benchmark["timestamp"].iloc[240]],
            "entry_price": [100.0, 120.0],
            "exit_price": [110.0, 90.0],
            "shares": [10, 10],
            "pnl": [100.0, -300.0],
            "return_pct": [0.1, -0.25],
            "holding_days": [10, 10],
            "exit_reason": ["signal_exit", "signal_exit"],
        }
    )
    regime_metrics = compute_regime_metrics(equity, trades, benchmark, 100000.0)
    assert not regime_metrics.empty


def test_profit_concentration_analysis():
    trades = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "CCC", "DDD"],
            "exit_timestamp": pd.to_datetime(["2021-01-01", "2021-02-01", "2022-01-01", "2022-02-01", "2023-01-01"]),
            "pnl": [1000.0, 100.0, 50.0, 25.0, 10.0],
        }
    )
    report = profit_concentration_analysis(trades)
    assert report["best_trade_profit_share"] > 0.5
    assert "AAA" in report["ticker_contribution"]


def test_parameter_stability_summary():
    sweep = pd.DataFrame(
        {
            "Total Return": [0.2, 0.18, -0.05],
            "Max Drawdown": [-0.1, -0.12, -0.4],
            "CAGR": [0.15, 0.14, -0.02],
            "Excess CAGR": [0.05, 0.03, -0.08],
            "parameters_json": [{"a": 1}, {"a": 2}, {"a": 3}],
        }
    )
    summary = parameter_stability_summary(sweep, drawdown_threshold=-0.2)
    assert summary["positive_return_pct"] == 2 / 3
    assert "conclusion" in summary


def test_robustness_score_behavior():
    metrics = {"Number of Trades": 40, "CAGR": 0.18, "Max Drawdown": -0.15, "Excess CAGR": 0.06}
    concentration = {"best_trade_profit_share": 0.2, "top_5_profit_share": 0.4}
    score = compute_robustness_score(metrics, concentration=concentration, walk_forward_summary={"profitable_test_fold_pct": 0.7, "consistency_score": 0.8})
    assert score.score >= 60
    assert score.label in {"Promising", "Strong but still needs review"}


def test_audit_persistence_and_custom_benchmark_selection(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "audit.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_long_bars("AAA", periods=40)
    benchmark = make_long_bars("QQQ", periods=40)
    result = engine.run(
        {"AAA": bars, "QQQ": benchmark},
        AlwaysFlatStrategy(),
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, price_mode="adjusted_price_mode"),
        benchmark_symbol="QQQ",
    )
    findings = generate_audit_findings(result.metrics, result.trade_log, result.equity_curve)
    audit_frame = pd.DataFrame(
        [{"run_id": result.run_id, "severity": finding.severity, "message": finding.message, "created_at": pd.Timestamp("2024-01-01")} for finding in findings]
    )
    db.replace_audit_results(result.run_id, audit_frame)
    saved = db.get_backtest_run(result.run_id)
    audit = db.read_audit_results(result.run_id)
    assert saved["benchmark_symbol"] == "QQQ"
    assert not audit.empty


def test_no_trade_and_missing_benchmark_edge_cases(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "edge.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_long_bars("AAA", periods=30)
    result = engine.run(
        {"AAA": bars},
        AlwaysFlatStrategy(),
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        benchmark_symbol="SPY",
    )
    assert result.trade_log.empty
    assert result.metrics["Number of Trades"] == 0
    assert result.metrics["Benchmark Total Return"] == 0.0
