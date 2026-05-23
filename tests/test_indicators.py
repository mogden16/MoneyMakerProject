from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.indicators.hma import hull_moving_average
from trading_lab.indicators.moving_average import weighted_moving_average
from trading_lab.indicators.qqe import qqe_indicator
from trading_lab.indicators.rsi import relative_strength_index
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy
from trading_lab.data.database import TradingLabDatabase


def make_series(length: int = 20, start: float = 100.0) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="B")
    values = pd.Series(range(length), index=index, dtype=float) + start
    return values


def make_bars(symbol: str, closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(closes),
            "symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "timestamp": closes.index,
            "session_date": closes.index.date,
            "open": closes.values,
            "high": closes.values + 1.0,
            "low": closes.values - 1.0,
            "close": closes.values,
            "adj_close": closes.values,
            "volume": [1000.0] * len(closes),
            "dividends": [0.0] * len(closes),
            "stock_splits": [0.0] * len(closes),
            "adjusted_flag": [False] * len(closes),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(closes),
        }
    )


def test_weighted_moving_average_preserves_shape_and_index():
    series = make_series(10)
    result = weighted_moving_average(series, 3)
    assert len(result) == len(series)
    assert result.index.equals(series.index)


def test_weighted_moving_average_monotonic_sanity():
    series = make_series(10)
    result = weighted_moving_average(series, 3).dropna()
    assert result.is_monotonic_increasing


def test_weighted_moving_average_insufficient_data_and_invalid_length():
    series = make_series(3)
    result = weighted_moving_average(series, 3)
    assert result.iloc[:2].isna().all()
    with pytest.raises(ValueError):
        weighted_moving_average(series, 1)


def test_hma_preserves_shape_index_and_handles_short_history():
    series = make_series(12)
    result = hull_moving_average(series, 5)
    assert len(result) == len(series)
    assert result.index.equals(series.index)
    assert result.iloc[:4].isna().any()


def test_hma_monotonic_input_and_invalid_length():
    series = make_series(20)
    result = hull_moving_average(series, 8).dropna()
    assert result.is_monotonic_increasing
    with pytest.raises(ValueError):
        hull_moving_average(series, 1)


def test_rsi_shape_range_and_flat_series_behavior():
    series = make_series(30)
    rsi = relative_strength_index(series, 14)
    assert len(rsi) == len(series)
    valid = rsi.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()

    flat = pd.Series([100.0] * 30, index=series.index)
    flat_rsi = relative_strength_index(flat, 14)
    assert flat_rsi.dropna().eq(50.0).all()


def test_rsi_invalid_length_and_no_divide_by_zero():
    series = pd.Series([100.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="B"))
    with pytest.raises(ValueError):
        relative_strength_index(series, 1)
    result = relative_strength_index(series, 14)
    assert result.dropna().notna().all()


def test_qqe_expected_columns_shape_and_signal_domain():
    series = make_series(80)
    result = qqe_indicator(series)
    expected = {"rsi", "rsi_smoothed", "rsi_atr", "qqe_fast", "qqe_slow", "upper_band", "lower_band", "trend", "signal"}
    assert expected.issubset(result.columns)
    assert len(result) == len(series)
    assert result.index.equals(series.index)
    assert set(result["signal"].dropna().unique()).issubset({-1, 0, 1})


def test_qqe_handles_flat_series_missing_data_and_invalid_params():
    series = pd.Series([100.0] * 50, index=pd.date_range("2024-01-01", periods=50, freq="B"))
    result = qqe_indicator(series)
    assert result["trend"].isin([-1, 0, 1]).all()
    assert result.iloc[:10].isna().any().any()
    with pytest.raises(ValueError):
        qqe_indicator(series, rsi_length=1)


def test_qqe_no_lookahead_behavior():
    series = make_series(60)
    baseline = qqe_indicator(series)
    mutated = series.copy()
    mutated.iloc[-1] = mutated.iloc[-1] + 100.0
    changed = qqe_indicator(mutated)
    pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1])


def test_qqe_hma_strategy_signal_shape_and_insufficient_data():
    strategy = QQEHMAStrategy()
    short_bars = make_bars("AAA", make_series(10))
    signals = strategy.generate_signals(short_bars)
    assert {"entry_signal", "exit_signal", "hma", "trend"}.issubset(signals.columns)
    assert not signals["entry_signal"].any()


def test_qqe_hma_strategy_engine_integration_and_next_bar_behavior(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "qqe_hma.duckdb"))
    engine = BacktestEngine(database=db)
    closes = pd.Series(
        [100, 99, 98, 99, 100, 102, 104, 106, 108, 110, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132],
        index=pd.date_range("2024-01-01", periods=30, freq="B"),
        dtype=float,
    )
    bars = make_bars("AAA", closes)
    strategy = QQEHMAStrategy(hma_length=5, rsi_length=5, rsi_smoothing=3, atr_smoothing=3, qqe_factor=2.5, require_hma_slope=False)
    signal_frame = strategy.generate_signals(bars)
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
    )
    assert len(signal_frame) == len(bars)
    if signal_frame["entry_signal"].any():
        first_signal_ts = signal_frame.loc[signal_frame["entry_signal"], "timestamp"].iloc[0]
        assert not result.trade_log.empty
        assert result.trade_log.iloc[0]["entry_timestamp"] > first_signal_ts
