from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.paper.forward_engine import build_active_paper_strategy_payload
from trading_lab.spy_lab import (
    build_spy_backtest_config,
    build_spy_robustness_checklist,
    build_spy_strategy,
    build_spy_workbench_config,
    get_spy_strategy_preset,
    get_spy_exit_structure,
    list_spy_exit_structures,
    list_spy_strategy_presets,
    run_spy_exit_comparison,
    run_spy_parameter_stability,
    spy_daily_signal_status,
    spy_strategy_summary,
    spy_summary_commentary,
    summarize_profit_concentration,
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


def test_spy_strategy_preset_creation_and_fixed_ticker_behavior():
    presets = list_spy_strategy_presets()
    assert any(preset.label == "SPY 200-Day Trend Filter" for preset in presets)
    assert any(exit_structure.label == "Signal exit only" for exit_structure in list_spy_exit_structures())
    payload = build_active_paper_strategy_payload(
        strategy_name="SPY 200-Day Trend Filter",
        strategy_parameters={"sma_length": 200},
        universe_name="SPY Trading Workbench",
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
        status="active",
    )
    assert payload["tickers"] == "SPY"
    assert payload["benchmark_symbol"] == "SPY"


def test_spy_200_day_trend_filter_strategy():
    strategy = build_spy_strategy("trend_filter_200")
    bars = make_bars("SPY", [100] * 205 + [110, 112, 90, 89])
    signals = strategy.generate_signals(bars)
    assert "trend_sma" in signals.columns
    assert signals["entry_signal"].any()
    assert signals["exit_signal"].any()


def test_spy_moving_average_rsi_and_breakout_presets():
    ma_strategy = build_spy_strategy("moving_average_50_200")
    rsi_strategy = build_spy_strategy("rsi_pullback_uptrend")
    breakout_strategy = build_spy_strategy("breakout_50_20")
    assert ma_strategy.parameters()["fast_window"] == 50
    assert rsi_strategy.parameters()["trend_sma_window"] == 200
    assert breakout_strategy.parameters()["exit_lookback_window"] == 20


def test_spy_summary_metrics_and_buy_hold_comparison():
    summary = spy_strategy_summary(
        {
            "Total Return": 0.45,
            "Benchmark Total Return": 0.30,
            "CAGR": 0.12,
            "Benchmark CAGR": 0.08,
            "Max Drawdown": -0.12,
            "Benchmark Max Drawdown": -0.22,
            "Sharpe Ratio": 1.0,
            "Calmar Ratio": 1.0,
            "Number of Trades": 25,
            "Win Rate": 0.55,
            "Profit Factor": 1.4,
            "Exposure %": 0.6,
            "Excess CAGR": 0.04,
        },
        benchmark_sharpe=0.75,
    )
    assert summary["Strategy CAGR"] == 0.12
    assert summary["Buy-and-Hold SPY CAGR"] == 0.08
    assert summary["Drawdown Improvement vs SPY"] > 0
    assert "beat buy-and-hold SPY" in spy_summary_commentary(summary)


def test_spy_robustness_checklist_and_parameter_stability():
    checklist, label = build_spy_robustness_checklist(
        metrics={
            "Excess CAGR": 0.03,
            "Max Drawdown": -0.12,
            "Benchmark Max Drawdown": -0.20,
            "Calmar Ratio": 1.1,
            "Benchmark CAGR": 0.08,
            "Number of Trades": 30,
        },
        concentration={"best_trade_profit_share": 0.2, "top_5_profit_share": 0.45},
        robustness_payload={
            "train_test": {"degradation": {"CAGR": -0.02}},
            "walk_summary": {"profitable_test_fold_pct": 0.6, "consistency_score": 0.7},
            "parameter_stability": {"positive_return_pct": 0.6, "percent_beating_spy": 0.5},
            "slippage_warnings": [],
        },
    )
    assert not checklist.empty
    assert label in {"Strong SPY candidate", "Possible SPY candidate"}


def test_spy_parameter_stability(tmp_path: Path):
    engine = BacktestEngine(database=None)
    bars = make_bars("SPY", [100 + index for index in range(260)])
    sweep_id, results, summary = run_spy_parameter_stability(
        engine=engine,
        config=BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        data_by_symbol={"SPY": bars},
        preset_key="trend_filter_200",
        benchmark_symbol="SPY",
    )
    assert sweep_id
    assert not results.empty
    assert "percent_beating_spy" in summary


def test_spy_workbench_config_and_backtest_config_object():
    config = build_spy_workbench_config(
        preset_key="trend_filter_200",
        entry_parameters={"sma_length": 200},
        exit_structure_key="fixed_stop_loss",
        exit_parameters={"stop_loss_pct": 0.08},
        start_date="2020-01-01",
        end_date="2024-01-01",
        price_mode="adjusted_price_mode",
        initial_capital=100000.0,
        position_sizing_method="percent_of_portfolio",
        position_size_value=1.0,
        max_positions=1,
        slippage_pct=0.0005,
        commission_per_trade=1.0,
    )
    engine_config = build_spy_backtest_config(config)
    assert config.exit_structure_label == get_spy_exit_structure("fixed_stop_loss").label
    assert engine_config.stop_loss_pct == 0.08
    assert engine_config.take_profit_pct is None


def test_spy_exit_comparison_results():
    engine = BacktestEngine(database=None)
    bars = make_bars("SPY", [100 + index for index in range(260)])
    workbench = build_spy_workbench_config(
        preset_key="trend_filter_200",
        entry_parameters={"sma_length": 200},
        exit_structure_key="signal_exit_only",
        exit_parameters={},
        start_date="2020-01-01",
        end_date="2024-01-01",
        price_mode="adjusted_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_size_value=1000.0,
        max_positions=1,
        slippage_pct=0.0,
        commission_per_trade=0.0,
    )
    comparison = run_spy_exit_comparison(
        engine=engine,
        data_by_symbol={"SPY": bars},
        workbench=workbench,
        exit_structure_keys=["signal_exit_only", "fixed_stop_loss", "partial_take_profit_plus_trailing_stop"],
        benchmark_symbol="SPY",
    )
    assert set(comparison.columns) >= {"Exit Structure", "Status", "CAGR", "Candidate Label"}
    assert "Planned" in set(comparison["Status"])


def test_spy_forward_promotion_and_daily_signal_status():
    payload = build_active_paper_strategy_payload(
        strategy_name="SPY 200-Day Trend Filter",
        strategy_parameters={"sma_length": 200},
        universe_name="SPY Trading Workbench",
        tickers=["SPY"],
        benchmark_symbol="SPY",
        price_mode="adjusted_price_mode",
        initial_capital=50000.0,
        position_sizing_method="percent_of_portfolio",
        position_sizing_value=1.0,
        max_positions=1,
        risk_settings={"fill_rule": "next_open"},
        slippage_pct=0.0,
        commission_per_trade=0.0,
        status="active",
    )
    assert payload["universe_name"] == "SPY Trading Workbench"
    strategy = build_spy_strategy("trend_filter_200")
    bars = make_bars("SPY", [100] * 205 + [110, 112])
    pending_orders = pd.DataFrame([{"status": "pending"}])
    open_positions = pd.DataFrame()
    status = spy_daily_signal_status(
        bars=bars,
        strategy=strategy,
        latest_close=float(bars["close"].iloc[-1]),
        data_freshness_status="fresh",
        pending_orders=pending_orders,
        open_positions=open_positions,
    )
    assert status["pending_order"] is True
    assert status["next_expected_action"] == "pending entry next open"


def test_spy_no_data_and_insufficient_data_edge_cases():
    strategy = build_spy_strategy("trend_filter_200")
    empty_status = spy_daily_signal_status(
        bars=pd.DataFrame(),
        strategy=strategy,
        latest_close=0.0,
        data_freshness_status="unknown",
        pending_orders=pd.DataFrame(),
        open_positions=pd.DataFrame(),
    )
    assert empty_status["current_signal"] == "no_signal"
    short_bars = make_bars("SPY", [100, 101, 102])
    short_status = spy_daily_signal_status(
        bars=short_bars,
        strategy=strategy,
        latest_close=102.0,
        data_freshness_status="fresh",
        pending_orders=pd.DataFrame(),
        open_positions=pd.DataFrame(),
    )
    assert short_status["position_state"] == "flat"


def test_spy_profit_concentration_helper():
    trades = pd.DataFrame(
        [
            {"pnl": 100.0, "symbol": "SPY", "exit_timestamp": pd.Timestamp("2024-01-05")},
            {"pnl": 50.0, "symbol": "SPY", "exit_timestamp": pd.Timestamp("2024-02-05")},
            {"pnl": -20.0, "symbol": "SPY", "exit_timestamp": pd.Timestamp("2024-03-05")},
        ]
    )
    concentration = summarize_profit_concentration(trades)
    assert concentration["best_trade_profit_share"] > 0
