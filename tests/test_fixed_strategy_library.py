from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.pybroker_lab.config import PyBrokerLabConfig
from trading_lab.pybroker_lab.runner import run_pybroker_lab
from trading_lab.pybroker_lab.strategy_registry import fixed_strategy_library
from trading_lab.pybroker_lab.strategies.blackflag_fts_hma import (
    BLACKFLAG_FTS_HMA_SETTINGS,
    classify_hull_value,
    compute_blackflag_fts,
    generate_signal_frame as generate_blackflag_hma_signals,
)
from trading_lab.pybroker_lab.strategies.blackflag_fts_qqe_momo import generate_signal_frame as generate_blackflag_qqe_signals
from trading_lab.pybroker_lab.strategies.ema_compression_volume_breakout import (
    EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS,
    compute_choppiness_index,
    generate_signal_frame as generate_ema_breakout_signals,
)
from trading_lab.pybroker_lab.strategies.legacy_mtf_qqe_rsi_momentum import (
    LEGACY_MTF_QQE_RSI_SETTINGS,
    generate_signal_frame as generate_legacy_qqe_signals,
)


def make_intraday_frame(*, timeframe: str = "15m", days: int = 20, symbol: str = "SPY") -> pd.DataFrame:
    step_minutes = 15 if timeframe == "15m" else 5
    bars_per_day = 26 if timeframe == "15m" else 78
    business_days = pd.date_range("2024-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    price = 100.0
    for day_index, day in enumerate(business_days):
        for bar_index in range(bars_per_day):
            timestamp = day + pd.Timedelta(hours=9, minutes=30 + step_minutes * bar_index)
            swing = np.sin((day_index * bars_per_day + bar_index) / 8.0) * 0.8
            drift = 0.04 if day_index % 2 == 0 else -0.02
            close = price + drift + swing
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": price,
                    "high": max(price, close) + 0.35,
                    "low": min(price, close) - 0.35,
                    "close": close,
                    "volume": 100000 + day_index * 100 + bar_index * 50 + (bar_index % 5) * 5000,
                }
            )
            price = close
    frame = pd.DataFrame(rows)
    frame["date"] = frame["timestamp"]
    return frame


def test_blackflag_modified_true_range_and_wilder_loss():
    frame = make_intraday_frame(days=5).iloc[:40].copy()
    settings = dict(BLACKFLAG_FTS_HMA_SETTINGS)
    settings["atr_period"] = 5
    blackflag = compute_blackflag_fts(frame, settings)
    assert {"HiLo", "HRef", "LRef", "true_range", "loss"} <= set(blackflag.columns)
    valid = blackflag[["true_range", "HiLo"]].dropna()
    assert valid["true_range"].ge(valid["HiLo"]).all()
    assert blackflag["loss"].dropna().iloc[-1] > 0


def test_blackflag_state_transitions_long_and_short():
    timestamps = pd.date_range("2024-01-02 09:30", periods=10, freq="15min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10, 11, 12, 13, 12, 11, 10, 9, 10, 11],
            "high": [11, 12, 13, 14, 13, 12, 11, 10, 11, 12],
            "low": [9, 10, 11, 12, 11, 10, 9, 8, 9, 10],
            "close": [10.5, 11.5, 12.5, 13.5, 11.0, 10.0, 9.0, 8.5, 10.5, 11.5],
            "volume": [1000] * 10,
        }
    )
    settings = dict(BLACKFLAG_FTS_HMA_SETTINGS)
    settings["atr_period"] = 2
    settings["atr_factor"] = 1
    blackflag = compute_blackflag_fts(frame, settings)
    assert blackflag["sell_signal"].any()
    assert blackflag["buy_signal"].iloc[blackflag["sell_signal"].idxmax() + 1 :].any()


def test_blackflag_hma_merge_without_lookahead_and_hull_classification():
    frame = make_intraday_frame(days=4).iloc[:60].copy()
    signals = generate_blackflag_hma_signals(frame, settings={**BLACKFLAG_FTS_HMA_SETTINGS, "hma_length": 4, "hma_lookback": 2, "atr_period": 3})
    first_valid = signals["higher_hma"].first_valid_index()
    assert first_valid is not None
    assert signals["higher_hma"].iloc[:first_valid].isna().all()
    hull = classify_hull_value(pd.Series([1.0, 2.0, 3.0, 2.8, 2.6, 2.9, 3.1]), 2)
    assert set(hull["hull_value"].dropna().astype(int).unique()) <= {1, 2, 3, 4}


def test_blackflag_entries_require_flip_and_hull_filter():
    frame = make_intraday_frame(days=6).iloc[:80].copy()
    signals = generate_blackflag_hma_signals(frame, settings={**BLACKFLAG_FTS_HMA_SETTINGS, "hma_length": 4, "hma_lookback": 2, "atr_period": 3})
    if signals["long_entry_signal"].any():
        rows = signals.loc[signals["long_entry_signal"]]
        assert rows["buy_signal"].all()
        assert rows["hull_value"].le(2).all()
    if signals["short_entry_signal"].any():
        rows = signals.loc[signals["short_entry_signal"]]
        assert rows["sell_signal"].all()
        assert rows["hull_value"].ge(3).all()


def test_ema_spread_choppiness_and_lunch_filter():
    frame = make_intraday_frame(days=15).copy()
    signals = generate_ema_breakout_signals(frame)
    chop = compute_choppiness_index(frame, EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["chop_length"], EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["chop_signal"])
    assert "ci" in chop.columns and "ci_avg" in chop.columns
    assert {"ema_spread_percent", "active_session", "volume_confirm", "range_expand"} <= set(signals.columns)
    lunch_rows = signals[signals.index.to_series().map(lambda idx: frame.loc[idx, "timestamp"].strftime("%H:%M")).between("12:30", "13:39")]
    if not lunch_rows.empty:
        assert (~lunch_rows["active_session"]).all()


def test_ema_long_short_signal_components_and_eod_flag():
    frame = make_intraday_frame(days=20).copy()
    signals = generate_ema_breakout_signals(frame)
    assert signals["eod_exit_signal"].any()
    assert set(signals["take_profit_pct"].dropna().unique()) <= {
        EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["take_profit_pct"],
        EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS["take_profit_after_lunch_pct"],
    }
    if signals["long_entry_signal"].any():
        row = signals.loc[signals["long_entry_signal"]].iloc[0]
        assert bool(row["volume_confirm"])
        assert bool(row["range_expand"])
    if signals["short_entry_signal"].any():
        row = signals.loc[signals["short_entry_signal"]].iloc[0]
        assert bool(row["volume_confirm"])
        assert bool(row["range_expand"])


def test_legacy_qqe_outputs_merge_and_weighted_formula():
    frame = make_intraday_frame(days=20).copy()
    corrected = generate_legacy_qqe_signals(frame, timeframe="15m")
    legacy_exact = generate_legacy_qqe_signals(
        frame,
        timeframe="15m",
        settings={**LEGACY_MTF_QQE_RSI_SETTINGS, "legacy_weighting_mode": "legacy_exact"},
    )
    assert {"combined_momentum", "higher_rsi_ma", "long_entry_signal", "short_entry_signal"} <= set(corrected.columns)
    assert corrected["higher_rsi_ma"].isna().sum() > 0
    assert not corrected["combined_momentum"].equals(legacy_exact["combined_momentum"])


def test_legacy_qqe_cross_detection_and_next_bar_signal_presence():
    frame = make_intraday_frame(days=20).copy()
    signals = generate_legacy_qqe_signals(frame, timeframe="15m")
    assert signals["long_exit_signal"].any() or signals["short_exit_signal"].any()
    if signals["long_entry_signal"].any():
        first_idx = signals.index[signals["long_entry_signal"]][0]
        assert first_idx + 1 < len(signals)
    if signals["short_entry_signal"].any():
        first_idx = signals.index[signals["short_entry_signal"]][0]
        assert first_idx + 1 < len(signals)


def test_blackflag_qqe_confirmation_requires_both_states():
    frame = make_intraday_frame(days=10).copy()
    signals = generate_blackflag_qqe_signals(frame)
    assert {"blackflag_state", "qqe_bullish", "qqe_bearish", "long_entry_signal", "short_entry_signal"} <= set(signals.columns)
    if signals["long_entry_signal"].any():
        rows = signals.loc[signals["long_entry_signal"]]
        assert rows["qqe_bullish"].all()
        assert rows["blackflag_state"].eq(1).all()
    if signals["short_entry_signal"].any():
        rows = signals.loc[signals["short_entry_signal"]]
        assert rows["qqe_bearish"].all()
        assert rows["blackflag_state"].eq(-1).all()


def test_fixed_strategy_smoke_runs(tmp_path: Path):
    frame = make_intraday_frame(days=25).copy()
    config = PyBrokerLabConfig(
        symbols=("SPY",),
        benchmark_symbol="SPY",
        start_date=str(frame["timestamp"].min().date()),
        end_date=str(frame["timestamp"].max().date()),
        timeframe="15m",
        warmup_bars=20,
        walkforward_windows=2,
        train_size=0.6,
        bootstrap_sample_size=25,
        output_dir=tmp_path,
    )
    for strategy_id in fixed_strategy_library():
        output_dir = tmp_path / strategy_id
        result = run_pybroker_lab(config.__class__(**{**config.__dict__, "output_dir": output_dir}), strategy_name=strategy_id, data_frame=frame)
        assert not result.summary.empty
        assert not result.strategy_metrics.empty
        assert not result.benchmark_metrics.empty
        assert isinstance(result.trades, pd.DataFrame)
        assert (output_dir / "summary.csv").exists()
