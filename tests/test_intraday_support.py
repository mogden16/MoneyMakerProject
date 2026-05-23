from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.data.database import TradingLabDatabase
from trading_lab.paper.forward_engine import ForwardPaperEngine, build_active_paper_strategy_payload
from trading_lab.spy_lab import build_spy_backtest_config, build_spy_strategy, build_spy_workbench_config, prepare_spy_timeframe_bars


def make_intraday_bars(timeframe: str = "15m") -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-05-20 09:30",
            "2024-05-20 09:45",
            "2024-05-20 10:00",
            "2024-05-20 10:15",
            "2024-05-21 09:30",
            "2024-05-21 09:45",
            "2024-05-21 10:00",
            "2024-05-21 10:15",
        ]
    )
    closes = [100.0, 99.0, 100.5, 101.0, 101.2, 101.8, 102.5, 102.0]
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(timestamps),
            "symbol": ["SPY"] * len(timestamps),
            "timeframe": [timeframe] * len(timestamps),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": closes,
            "high": [value + 0.6 for value in closes],
            "low": [value - 0.6 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [1000.0] * len(timestamps),
            "dividends": [0.0] * len(timestamps),
            "stock_splits": [0.0] * len(timestamps),
            "adjusted_flag": [True] * len(timestamps),
            "retrieved_at": [pd.Timestamp("2024-05-21")] * len(timestamps),
        }
    )


def make_daily_regime_bars() -> pd.DataFrame:
    index = pd.date_range("2023-07-01", periods=240, freq="B")
    closes = [100.0] * 220 + [130.0] * 20
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(index),
            "symbol": ["SPY"] * len(index),
            "timeframe": ["1d"] * len(index),
            "timestamp": index,
            "session_date": index.date,
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [1000.0] * len(index),
            "dividends": [0.0] * len(index),
            "stock_splits": [0.0] * len(index),
            "adjusted_flag": [True] * len(index),
            "retrieved_at": [pd.Timestamp("2024-05-21")] * len(index),
        }
    )


class IntradayProvider:
    def __init__(self, intraday: pd.DataFrame, daily: pd.DataFrame, status=None) -> None:
        self.intraday = intraday
        self.daily = daily
        self.status = status

    def get_stock_bars(self, *, symbol: str, start_date: str, end_date: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame:
        return self.daily.copy() if timeframe == "1d" else self.intraday.copy()

    def get_last_fetch_status(self, symbol: str):
        return self.status


def test_intraday_regime_alignment_and_next_bar_execution():
    intraday = make_intraday_bars("15m")
    daily = make_daily_regime_bars()
    prepared = prepare_spy_timeframe_bars(primary_bars=intraday, timeframe="15m", daily_bars=daily)
    strategy = build_spy_strategy("intraday_pullback")
    workbench = build_spy_workbench_config(
        preset_key="intraday_pullback",
        entry_parameters=strategy.parameters(),
        timeframe="15m",
        exit_structure_key="signal_exit_only",
        exit_parameters={},
        start_date="2024-05-20",
        end_date="2024-05-21",
        price_mode="adjusted_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_size_value=1000.0,
        max_positions=1,
        slippage_pct=0.0,
        commission_per_trade=0.0,
    )
    result = BacktestEngine(database=None).run(
        data_by_symbol={"SPY": prepared},
        strategy=strategy,
        config=BacktestConfig(**build_spy_backtest_config(workbench).model_dump()),
        benchmark_symbol="SPY",
    )
    if not result.trade_log.empty:
        assert (pd.to_datetime(result.trade_log["exit_timestamp"]) >= pd.to_datetime(result.trade_log["entry_timestamp"])).all()


def test_intraday_forward_paper_eod_exit_and_timeframe_persistence(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "intraday_forward.duckdb"))
    payload = build_active_paper_strategy_payload(
        strategy_name="Daily Trend + Intraday Breakout",
        strategy_parameters={"breakout_lookback_bars": 2, "exit_lookback_bars": 1, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False},
        universe_name="SPY Workbench",
        tickers=["SPY"],
        timeframe="15m",
        benchmark_symbol="SPY",
        price_mode="raw_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_sizing_value=1000.0,
        max_positions=1,
        risk_settings={"fill_rule": "next_open", "same_bar_stop_target_rule": "conservative_stop_first", "end_of_day_exit": True, "allow_overnight": False},
        slippage_pct=0.0,
        commission_per_trade=0.0,
        status="active",
    )
    payload["created_at"] = pd.Timestamp("2024-05-20 09:30")
    payload["updated_at"] = pd.Timestamp("2024-05-20 09:30")
    db.insert_active_paper_strategy(payload)
    saved = db.get_active_paper_strategy(payload["active_strategy_id"])
    assert saved is not None
    assert saved["timeframe"] == "15m"

    provider = IntradayProvider(make_intraday_bars("15m"), make_daily_regime_bars())
    result = ForwardPaperEngine().run_update(active_strategy=payload, provider=provider)
    assert result.skipped is False
    if not result.trades.empty:
        assert result.trades["timeframe"].eq("15m").all()
        assert "end_of_day_exit" in set(result.trades["exit_reason"]) or "signal_exit" in set(result.trades["exit_reason"])


def test_intraday_forward_skip_on_missing_data_warning():
    provider = IntradayProvider(
        make_intraday_bars("5m"),
        make_daily_regime_bars(),
        status=SimpleNamespace(cache_status="fresh", validation_warnings=["Missing intraday bars detected."]),
    )
    payload = build_active_paper_strategy_payload(
        strategy_name="Daily Trend + Intraday Pullback",
        strategy_parameters={"rsi_length": 10, "oversold_threshold": 30.0, "recovery_threshold": 40.0, "moving_average_length": 8, "pullback_lookback_bars": 5, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False},
        universe_name="SPY Workbench",
        tickers=["SPY"],
        timeframe="5m",
        benchmark_symbol="SPY",
        price_mode="raw_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_sizing_value=1000.0,
        max_positions=1,
        risk_settings={"fill_rule": "next_open", "same_bar_stop_target_rule": "conservative_stop_first", "end_of_day_exit": True, "allow_overnight": False},
        slippage_pct=0.0,
        commission_per_trade=0.0,
        status="active",
    )
    result = ForwardPaperEngine().run_update(active_strategy=payload, provider=provider)
    assert result.skipped is True
