from pathlib import Path

import pandas as pd

from trading_lab.backtest.audit import generate_strategy_audit
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.sweep import run_parameter_sweep
from trading_lab.backtest.train_test import run_train_test_analysis, split_data_by_date
from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.market_calendar import MarketCalendar
from trading_lab.strategies.base import StrategyBase
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy


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


def make_bars(
    symbol: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(opens), freq="D")
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(opens),
            "symbol": [symbol] * len(opens),
            "timeframe": ["1d"] * len(opens),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": [1000.0] * len(opens),
            "dividends": [0.0] * len(opens),
            "stock_splits": [0.0] * len(opens),
            "adjusted_flag": [False] * len(opens),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(opens),
        }
    )


def test_moving_average_signal_generation():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "open": [10, 10, 10, 10, 10, 10, 10, 10],
            "high": [10, 10, 10, 10, 10, 10, 10, 10],
            "low": [10, 10, 10, 10, 10, 10, 10, 10],
            "close": [10, 9, 8, 9, 10, 11, 12, 13],
        }
    )
    strategy = MovingAverageCrossStrategy(fast_window=2, slow_window=3)
    signals = strategy.generate_signals(bars)
    entry_rows = signals.index[signals["entry_signal"]].tolist()
    assert entry_rows == [4]


def test_backtest_respects_next_bar_execution(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "bt.duckdb"))
    engine = BacktestEngine(database=db)
    strategy = MovingAverageCrossStrategy(fast_window=2, slow_window=3)
    bars = make_bars(
        "AAA",
        opens=[10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        highs=[10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        lows=[10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        closes=[10.0, 9.0, 8.0, 15.0, 16.0, 17.0],
    )
    result = engine.run(
        data_by_symbol={"AAA": bars},
        strategy=strategy,
        config=BacktestConfig(
            initial_capital=10000.0,
            slippage_pct=0.0,
            commission_per_trade=0.0,
            position_sizing_method="fixed_dollar",
            position_size_value=1000.0,
            max_positions=1,
        ),
    )
    first_trade = result.trade_log.iloc[0]
    assert first_trade["entry_timestamp"] == pd.Timestamp("2024-01-05")
    assert first_trade["entry_price"] == 20.0


def test_stop_loss_triggers_correctly(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "stop.duckdb")))
    bars = make_bars(
        "AAA",
        opens=[10.0, 10.0, 10.0, 9.0],
        highs=[10.0, 10.2, 10.1, 9.2],
        lows=[10.0, 9.9, 9.8, 8.5],
        closes=[10.0, 10.0, 10.0, 9.0],
    )
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, stop_loss_pct=0.1),
    )
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["entry_timestamp"] == pd.Timestamp("2024-01-03")
    assert trade["exit_timestamp"] == pd.Timestamp("2024-01-04")
    assert trade["exit_price"] == 9.0


def test_take_profit_triggers_correctly(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "take.duckdb")))
    bars = make_bars(
        "AAA",
        opens=[10.0, 10.0, 10.0, 10.4],
        highs=[10.0, 10.3, 10.5, 11.5],
        lows=[10.0, 9.9, 9.8, 10.0],
        closes=[10.0, 10.0, 10.0, 11.2],
    )
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, take_profit_pct=0.1),
    )
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == 11.0


def test_max_positions_constraint_limits_concurrent_symbols(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "maxpos.duckdb")))
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    bars_a = make_bars("AAA", [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10])
    bars_b = make_bars("BBB", [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20])
    result = engine.run(
        {"AAA": bars_a, "BBB": bars_b},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, max_positions=1),
    )
    assert len(result.trade_log) == 1


def test_no_same_bar_close_to_close_execution(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "samebar.duckdb")))
    bars = make_bars(
        "AAA",
        opens=[10.0, 12.0, 13.0, 14.0],
        highs=[10.0, 12.0, 13.0, 14.0],
        lows=[10.0, 12.0, 13.0, 14.0],
        closes=[10.0, 11.5, 12.5, 13.5],
    )
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, True, False])
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
    )
    trade = result.trade_log.iloc[0]
    assert trade["entry_timestamp"] == pd.Timestamp("2024-01-03")
    assert trade["entry_price"] == 13.0
    assert trade["exit_timestamp"] == pd.Timestamp("2024-01-04")
    assert trade["exit_price"] == 14.0


def test_multi_symbol_backtest_does_not_overallocate_capital(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "capital.duckdb")))
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    bars_a = make_bars("AAA", [50, 50, 50, 50], [50, 50, 50, 50], [50, 50, 50, 50], [50, 50, 50, 50])
    bars_b = make_bars("BBB", [60, 60, 60, 60], [60, 60, 60, 60], [60, 60, 60, 60], [60, 60, 60, 60])
    result = engine.run(
        {"AAA": bars_a, "BBB": bars_b},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=8000.0, max_positions=2),
    )
    total_entry_notional = float((result.trade_log["entry_price"] * result.trade_log["shares"]).sum())
    assert total_entry_notional <= 10000.0
    assert result.equity_curve["cash"].min() >= 0


def test_trade_log_records_expected_fields(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "trades.duckdb")))
    bars = make_bars(
        "AAA",
        opens=[10.0, 10.0, 10.0, 11.0],
        highs=[10.0, 10.0, 10.0, 11.0],
        lows=[10.0, 10.0, 10.0, 11.0],
        closes=[10.0, 10.0, 10.0, 11.0],
    )
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, True, False])
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
    )
    trade = result.trade_log.iloc[0]
    assert trade["entry_timestamp"] == pd.Timestamp("2024-01-03")
    assert trade["exit_timestamp"] == pd.Timestamp("2024-01-04")
    assert trade["entry_price"] == 10.0
    assert trade["exit_price"] == 11.0
    assert trade["exit_reason"] == "signal_exit"


def test_benchmark_alignment_handles_missing_dates(tmp_path: Path):
    engine = BacktestEngine(database=TradingLabDatabase(str(tmp_path / "benchmark.duckdb")))
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    bars_a = make_bars("AAA", [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 11, 12, 13])
    benchmark = make_bars("SPY", [100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100], [100, 101, 102, 103])
    benchmark = benchmark.iloc[[0, 2, 3]].reset_index(drop=True)
    result = engine.run(
        {"AAA": bars_a, "SPY": benchmark},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        benchmark_symbol="SPY",
    )
    assert len(result.benchmark_curve) == 4
    jan_2_value = result.benchmark_curve.loc[result.benchmark_curve["timestamp"] == pd.Timestamp("2024-01-02"), "benchmark_equity"].iloc[0]
    assert jan_2_value == 10000.0


def test_backtest_run_summary_is_persisted(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "saved.duckdb"))
    engine = BacktestEngine(database=db)
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, True, False])
    bars = make_bars("AAA", [10, 10, 10, 11], [10, 10, 10, 11], [10, 10, 10, 11], [10, 10, 10, 11])
    result = engine.run(
        {"AAA": bars},
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
    )
    saved = db.get_backtest_run(result.run_id)
    assert saved is not None
    assert saved["strategy_name"] == "signal_strategy"
    assert saved["number_of_trades"] == 1
    assert saved["benchmark_symbol"] == "SPY"


def test_compare_backtest_runs_query(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "compare.duckdb"))
    engine = BacktestEngine(database=db)
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, True, False])
    bars = make_bars("AAA", [10, 10, 10, 11], [10, 10, 10, 11], [10, 10, 10, 11], [10, 10, 10, 11])
    first = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0))
    second = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0))
    comparison = db.compare_backtest_runs([first.run_id, second.run_id])
    assert len(comparison) == 2


def test_parameter_sweep_result_persistence(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "sweep.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13])
    sweep_id, results = run_parameter_sweep(
        engine,
        lambda params: MovingAverageCrossStrategy(**params),
        {"AAA": bars},
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        {"fast_window": [2], "slow_window": [3, 4]},
        benchmark_symbol="SPY",
    )
    assert not results.empty
    saved = db.get_backtest_run(results.iloc[0]["run_id"])
    assert saved["sweep_id"] == sweep_id


def test_train_test_split_metrics(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "train_test.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13])
    strategy = SignalStrategy(entries=[False, True, False, False, True, False], exits=[False, False, True, False, False, True])
    train_data, test_data = split_data_by_date({"AAA": bars}, "2024-01-04")
    analysis = run_train_test_analysis(
        engine,
        strategy,
        BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0),
        train_data,
        test_data,
        benchmark_symbol="SPY",
    )
    assert "train_metrics" in analysis and "test_metrics" in analysis


def test_strategy_audit_warnings():
    metrics = {
        "Number of Trades": 3,
        "Win Rate": 1.0,
        "CAGR": 0.2,
        "Max Drawdown": -0.4,
        "Exposure %": 0.1,
        "Profit Factor": 4.0,
    }
    trades = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB"], "pnl": [1000.0, 50.0, 10.0], "return_pct": [0.5, 0.01, 0.01]})
    equity_curve = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=20, freq="D"), "equity": [100 + i for i in range(20)]})
    warnings = generate_strategy_audit(metrics, trades, equity_curve, benchmark_metrics={"benchmark_cagr": 0.25}, strategy_parameters={"a": 1, "b": 2})
    assert len(warnings) >= 2


def test_dividend_mode_does_not_double_count(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "dividends.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10])
    bars["dividends"] = [0.0, 0.0, 0.0, 1.0]
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    price_only = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, return_mode="price_return_only"))
    total_return = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, return_mode="total_return_with_dividends"))
    dividend_cash = (total_return.equity_curve["equity"].iloc[-1] - price_only.equity_curve["equity"].iloc[-1])
    assert dividend_cash == 100.0


def test_adjusted_price_mode_handles_split_smoother_than_raw_mode(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "split.duckdb"))
    engine = BacktestEngine(database=db)
    bars = make_bars("AAA", [100, 100, 50, 50], [100, 100, 50, 50], [100, 100, 50, 50], [100, 100, 50, 50])
    bars["adj_close"] = [100, 100, 100, 100]
    bars["stock_splits"] = [0.0, 0.0, 2.0, 0.0]
    strategy = SignalStrategy(entries=[False, True, False, False], exits=[False, False, False, False])
    raw_result = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, price_mode="raw_price_mode"))
    adj_result = engine.run({"AAA": bars}, strategy, BacktestConfig(initial_capital=10000.0, position_sizing_method="fixed_dollar", position_size_value=1000.0, price_mode="adjusted_price_mode"))
    assert adj_result.equity_curve["equity"].iloc[-1] >= raw_result.equity_curve["equity"].iloc[-1]


def test_market_calendar_latest_completed_session_holiday_weekend():
    calendar = MarketCalendar("NYSE")
    weekend_session = calendar.latest_completed_session("2024-07-06", now=pd.Timestamp("2024-07-06 12:00", tz="America/New_York").to_pydatetime())
    holiday_session = calendar.latest_completed_session("2024-07-04", now=pd.Timestamp("2024-07-04 12:00", tz="America/New_York").to_pydatetime())
    assert weekend_session == pd.Timestamp("2024-07-05").date()
    assert holiday_session == pd.Timestamp("2024-07-03").date()
