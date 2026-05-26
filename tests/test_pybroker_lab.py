from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_lab.app import build_candle_comparison_chart, build_pybroker_debug_chart
from trading_lab.pybroker_lab.audit import build_chart_metadata, build_raw_bars_export_name
from trading_lab.pybroker_lab.parity import compare_candle_frames, parse_tradingview_csv
from trading_lab.pybroker_lab.benchmarks import benchmark_metrics
from trading_lab.pybroker_lab.config import PyBrokerLabConfig
from trading_lab.pybroker_lab.runner import run_pybroker_lab, strategy_registry


def make_fixture_data(days: int = 25, timeframe: str = "15m") -> pd.DataFrame:
    symbols = ["SPY"]
    step_minutes = 15 if timeframe == "15m" else 5
    bars_per_day = 26 if timeframe == "15m" else 78
    business_days = pd.date_range("2024-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    base_prices = {"SPY": 100.0}
    drifts = {"SPY": 0.03}
    for symbol in symbols:
        price = base_prices[symbol]
        for day_index, day in enumerate(business_days):
            for bar_index in range(bars_per_day):
                timestamp = day + pd.Timedelta(hours=9, minutes=30 + step_minutes * bar_index)
                close = price + drifts[symbol] + ((day_index + bar_index) % 6 - 3) * 0.08
                rows.append(
                    {
                        "symbol": symbol,
                        "date": timestamp,
                        "open": price,
                        "high": max(price, close) + 0.4,
                        "low": min(price, close) - 0.4,
                        "close": close,
                        "volume": 1000000 + day_index * 100 + bar_index * 1000,
                    }
                )
                price = close
    return pd.DataFrame(rows)


def test_config_defaults():
    config = PyBrokerLabConfig()
    assert config.benchmark_symbol == "SPY"
    assert config.timeframe == "1d"
    assert config.walkforward_windows == 3


def test_benchmark_metrics_are_computed():
    spy_bars = make_fixture_data()[lambda frame: frame["symbol"] == "SPY"]
    metrics = benchmark_metrics(spy_bars, initial_cash=100000.0)
    assert "total_return" in metrics
    assert "sharpe" in metrics


def test_each_strategy_can_instantiate():
    config = PyBrokerLabConfig(output_dir=Path("outputs/pybroker_lab_test"))
    definitions = [builder(config) for builder in strategy_registry().values()]
    assert {definition.name for definition in definitions} == set(strategy_registry())


def test_runner_writes_expected_files(tmp_path: Path):
    fixture = make_fixture_data()
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=50,
        output_dir=tmp_path,
        sizing_method="percent_equity",
        sizing_value=1.0,
    )
    result = run_pybroker_lab(config, strategy_name="all", data_frame=fixture)
    assert not result.summary.empty
    for filename in [
        "summary.csv",
        "strategy_metrics.csv",
        "benchmark_metrics.csv",
        "actual_data_used.csv",
        "benchmark_summary.csv",
        "trade_audit.csv",
        "trades.csv",
        "equity_curve.csv",
        "report.md",
    ]:
        assert (tmp_path / filename).exists()


def test_actual_data_range_and_trade_audit_are_recorded(tmp_path: Path):
    fixture = make_fixture_data()
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=20,
        output_dir=tmp_path,
        sizing_method="fixed_dollar",
        sizing_value=25000.0,
    )
    result = run_pybroker_lab(config, strategy_name="blackflag_fts_hma", data_frame=fixture)
    assert not result.actual_data_used.empty
    actual_row = result.actual_data_used.iloc[0]
    assert actual_row["requested_start_date"] == "2023-01-01"
    assert pd.to_datetime(actual_row["actual_first_bar_timestamp"]) == pd.to_datetime(fixture["date"].min())
    assert pd.to_datetime(actual_row["actual_first_bar_timestamp"]) > pd.Timestamp("2023-01-01")
    assert actual_row["timezone"] == "America/New_York"
    assert bool(actual_row["extended_hours_included"]) is False
    assert actual_row["timestamp_basis"] == "bar_start"
    assert not result.trade_audit.empty
    assert {"strategy_id", "signal_timestamp", "entry_timestamp", "exit_reason", "entry_indicator_values", "exit_indicator_values"} <= set(result.trade_audit.columns)
    assert (tmp_path / "trade_audit.csv").exists()


def test_debug_chart_and_sizing_metadata_render(tmp_path: Path):
    fixture = make_fixture_data()
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=20,
        output_dir=tmp_path,
        sizing_method="fixed_shares",
        sizing_value=75.0,
    )
    result = run_pybroker_lab(config, strategy_name="ema_compression_volume_breakout", data_frame=fixture)
    assert "ema_compression_volume_breakout" in result.debug_frames
    figure = build_pybroker_debug_chart(
        result.debug_frames["ema_compression_volume_breakout"],
        result.trade_audit[result.trade_audit["strategy_id"] == "ema_compression_volume_breakout"],
        "ema_compression_volume_breakout",
    )
    assert any(trace.type == "candlestick" for trace in figure.data)
    assert any(trace.type == "bar" for trace in figure.data)
    assert figure.layout.showlegend is False
    assert not result.benchmark_summary.empty
    summary_row = result.benchmark_summary.iloc[0]
    assert summary_row["sizing_method"] == "fixed_shares"
    assert pd.to_datetime(summary_row["actual_first_bar_timestamp"]) == pd.to_datetime(fixture["date"].min())
    assert "ema_compression_volume_breakout" in result.data_quality_audits
    assert not result.data_quality_audits["ema_compression_volume_breakout"].empty
    assert "ema_compression_volume_breakout" in result.actual_bars
    export_name = build_raw_bars_export_name("SPY", "15m", result.actual_bars["ema_compression_volume_breakout"])
    assert export_name.endswith("_yf_bars.csv")
    metadata = build_chart_metadata(
        actual_data_row=result.actual_data_used.iloc[0].to_dict(),
        chart_frame=result.debug_frames["ema_compression_volume_breakout"],
        symbol="SPY",
    )
    assert "chart_display_range" in metadata.columns


def test_qqe_debug_chart_renders_oscillator_panel(tmp_path: Path):
    fixture = make_fixture_data()
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=20,
        output_dir=tmp_path,
    )
    result = run_pybroker_lab(config, strategy_name="legacy_mtf_qqe_rsi_momentum", data_frame=fixture)
    figure = build_pybroker_debug_chart(
        result.debug_frames["legacy_mtf_qqe_rsi_momentum"],
        result.trade_audit[result.trade_audit["strategy_id"] == "legacy_mtf_qqe_rsi_momentum"],
        "legacy_mtf_qqe_rsi_momentum",
    )
    assert any(trace.name == "Combined Momentum" for trace in figure.data)
    assert "yaxis3" in figure.layout
    assert "legacy_mtf_qqe_rsi_momentum" in result.indicator_debug_tables
    assert figure.layout.height == 980


def test_higher_timeframe_lookahead_checks_are_recorded(tmp_path: Path):
    fixture = make_fixture_data(days=25, timeframe="5m")
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="5m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=20,
        output_dir=tmp_path,
    )
    hma_result = run_pybroker_lab(config, strategy_name="blackflag_fts_hma", data_frame=fixture)
    hma_check = hma_result.higher_timeframe_checks["blackflag_fts_hma"]
    assert not hma_check.empty
    assert set(["higher_timeframe_source_timestamp", "higher_timeframe_close_timestamp", "lookahead_check_result"]) <= set(hma_check.columns)
    assert not hma_check["lookahead_check_result"].eq("FAIL").any()

    qqe_result = run_pybroker_lab(config, strategy_name="legacy_mtf_qqe_rsi_momentum", data_frame=fixture)
    qqe_check = qqe_result.higher_timeframe_checks["legacy_mtf_qqe_rsi_momentum"]
    assert not qqe_check.empty
    assert not qqe_check["lookahead_check_result"].eq("FAIL").any()


def test_non_higher_timeframe_strategy_has_no_lookahead_check(tmp_path: Path):
    fixture = make_fixture_data(days=25, timeframe="15m")
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date="2023-01-01",
        end_date=str(fixture["date"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=20,
        output_dir=tmp_path,
    )
    result = run_pybroker_lab(config, strategy_name="ema_compression_volume_breakout", data_frame=fixture)
    assert result.higher_timeframe_checks["ema_compression_volume_breakout"].empty


def test_tradingview_csv_parse_and_exact_match_comparison():
    fixture = make_fixture_data(days=2, timeframe="5m")
    csv_frame = fixture.rename(
        columns={
            "date": "Time",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Time", "Open", "High", "Low", "Close", "Volume"]]
    payload = csv_frame.to_csv(index=False).encode("utf-8")
    tradingview = parse_tradingview_csv(payload, timeframe="5m", timezone="America/New_York", symbol="SPY")
    assert not tradingview.empty
    comparison = compare_candle_frames(
        fixture,
        tradingview,
        symbol="SPY",
        timeframe="5m",
        regular_hours_only=True,
    )
    summary_row = comparison.summary.iloc[0]
    assert summary_row["matched_bars_count"] == len(tradingview)
    assert summary_row["bars_with_ohlc_difference_over_tolerance"] == 0
    figure = build_candle_comparison_chart(comparison)
    assert any(trace.type == "candlestick" for trace in figure.data)


def test_candle_comparison_detects_one_bar_shift():
    fixture = make_fixture_data(days=1, timeframe="5m")
    shifted = fixture.copy()
    shifted["date"] = pd.to_datetime(shifted["date"]) + pd.Timedelta(minutes=5)
    comparison = compare_candle_frames(
        fixture,
        shifted,
        symbol="SPY",
        timeframe="5m",
        regular_hours_only=True,
    )
    assert comparison.summary.iloc[0]["first_mismatched_timestamp"] == pd.to_datetime(fixture["date"].min())
    aligned = compare_candle_frames(
        fixture,
        shifted,
        symbol="SPY",
        timeframe="5m",
        regular_hours_only=False,
        shift_dataset="shift_yfinance_forward_1_bar",
    )
    assert aligned.summary.iloc[0]["matched_bars_count"] == len(fixture)


def test_candle_comparison_regular_hours_filter_works():
    fixture = make_fixture_data(days=1, timeframe="5m")
    with_extra = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "symbol": "SPY",
                        "date": pd.Timestamp("2024-01-02 09:25"),
                        "open": 99.0,
                        "high": 99.5,
                        "low": 98.5,
                        "close": 99.25,
                        "volume": 5000.0,
                    }
                ]
            ),
            fixture,
        ],
        ignore_index=True,
    ).sort_values("date")
    comparison = compare_candle_frames(
        with_extra,
        with_extra,
        symbol="SPY",
        timeframe="5m",
        regular_hours_only=True,
    )
    assert comparison.summary.iloc[0]["matched_bars_count"] == len(fixture)
