from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trading_lab.data.database import TradingLabDatabase
from trading_lab.paper.forward_engine import (
    ForwardPaperEngine,
    build_active_paper_strategy_payload,
    build_promotion_checklist,
    compare_forward_to_backtest,
)
from trading_lab.strategies.base import StrategyBase


class ScheduledSignalStrategy(StrategyBase):
    name = "scheduled"

    def __init__(self, entry_indices: list[int] | None = None, exit_indices: list[int] | None = None, max_holding_days: int | None = None) -> None:
        self.entry_indices = entry_indices or []
        self.exit_indices = exit_indices or []
        self.max_holding_days = max_holding_days

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["entry_signal"] = False
        frame["exit_signal"] = False
        for idx in self.entry_indices:
            if 0 <= idx < len(frame):
                frame.loc[idx, "entry_signal"] = True
        for idx in self.exit_indices:
            if 0 <= idx < len(frame):
                frame.loc[idx, "exit_signal"] = True
        return frame


class FakeProvider:
    def __init__(self, data_map: dict[str, pd.DataFrame], statuses: dict[str, object] | None = None) -> None:
        self.data_map = data_map
        self.statuses = statuses or {}

    def get_stock_bars(self, *, symbol: str, start_date: str, end_date: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame:
        return self.data_map[symbol].copy()

    def get_last_fetch_status(self, symbol: str):
        return self.statuses.get(symbol)


def make_bars(
    symbol: str,
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "source_vendor": ["test"] * len(closes),
            "symbol": [symbol] * len(closes),
            "timeframe": ["1d"] * len(closes),
            "timestamp": timestamps,
            "session_date": timestamps.date,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": [1000.0] * len(closes),
            "dividends": [0.0] * len(closes),
            "stock_splits": [0.0] * len(closes),
            "adjusted_flag": [True] * len(closes),
            "retrieved_at": [pd.Timestamp("2024-01-10")] * len(closes),
        }
    )


def make_active_payload(**overrides):
    payload = build_active_paper_strategy_payload(
        strategy_name="scheduled",
        strategy_parameters={"entry_indices": [1], "exit_indices": []},
        universe_name="Custom",
        tickers=["AAA"],
        benchmark_symbol="SPY",
        price_mode="raw_price_mode",
        initial_capital=10000.0,
        position_sizing_method="fixed_dollar",
        position_sizing_value=2000.0,
        max_positions=2,
        risk_settings={"stop_loss_pct": 0.05, "take_profit_pct": 0.05, "fill_rule": "next_open", "same_bar_stop_target_rule": "conservative_stop_first"},
        slippage_pct=0.0,
        commission_per_trade=0.0,
        status="active",
    )
    payload["created_at"] = pd.Timestamp("2024-01-01")
    payload["updated_at"] = pd.Timestamp("2024-01-01")
    payload.update(overrides)
    return payload


def install_scheduled_strategy(monkeypatch):
    monkeypatch.setattr(
        "trading_lab.paper.forward_engine.build_strategy_instance",
        lambda strategy_name, parameters: ScheduledSignalStrategy(
            entry_indices=list(parameters.get("entry_indices", [])),
            exit_indices=list(parameters.get("exit_indices", [])),
            max_holding_days=parameters.get("max_holding_days"),
        ),
    )


def test_active_paper_strategy_persistence_and_status_transitions(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "forward_active.duckdb"))
    payload = make_active_payload()
    db.insert_active_paper_strategy(payload)
    saved = db.get_active_paper_strategy(payload["active_strategy_id"])
    assert saved is not None
    assert saved["status"] == "active"
    saved["status"] = "paused"
    db.update_active_paper_strategy(saved)
    paused = db.get_active_paper_strategy(payload["active_strategy_id"])
    assert paused is not None
    assert paused["status"] == "paused"
    paused["status"] = "retired"
    db.update_active_paper_strategy(paused)
    retired = db.get_active_paper_strategy(payload["active_strategy_id"])
    assert retired is not None
    assert retired["status"] == "retired"


def test_forward_paper_order_creation_and_next_open_fill(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100, 100, 101, 102, 103],
        highs=[101, 101, 102, 103, 104],
        lows=[99, 99, 100, 101, 102],
        closes=[100, 100, 101, 102, 103],
    )
    benchmark = make_bars("SPY", opens=[300, 301, 302, 303, 304], highs=[301, 302, 303, 304, 305], lows=[299, 300, 301, 302, 303], closes=[300, 301, 302, 303, 304])
    provider = FakeProvider({"AAA": bars, "SPY": benchmark})
    result = ForwardPaperEngine().run_update(active_strategy=make_active_payload(), provider=provider)
    assert result.skipped is False
    assert len(result.orders) == 1
    order = result.orders.iloc[0]
    assert order["status"] == "filled"
    assert pd.Timestamp(order["actual_fill_date"]) > pd.Timestamp(order["signal_date"])
    assert result.positions["status"].eq("open").any()
    assert {"order_created", "order_filled", "update_summary"} <= set(result.events["event_type"])


def test_forward_engine_no_same_bar_execution(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100, 100, 110],
        highs=[101, 111, 112],
        lows=[99, 99, 109],
        closes=[100, 110, 111],
    )
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")})
    result = ForwardPaperEngine().run_update(active_strategy=make_active_payload(strategy_parameters_json='{"entry_indices":[1],"exit_indices":[]}'), provider=provider)
    order = result.orders.iloc[0]
    assert pd.Timestamp(order["signal_date"]) == pd.Timestamp("2024-01-02")
    assert pd.Timestamp(order["actual_fill_date"]) == pd.Timestamp("2024-01-03")


def test_forward_engine_stop_loss_and_ambiguity_stop_first(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 106],
        lows=[99, 99, 99, 94],
        closes=[100, 100, 100, 100],
    )
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")})
    result = ForwardPaperEngine().run_update(active_strategy=make_active_payload(), provider=provider)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == 95.0


def test_forward_engine_take_profit(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 106],
        lows=[99, 99, 99, 99],
        closes=[100, 100, 100, 100],
    )
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")})
    payload = make_active_payload()
    payload["risk_settings_json"] = '{"stop_loss_pct": 0.02, "take_profit_pct": 0.05, "fill_rule": "next_open", "same_bar_stop_target_rule": "conservative_stop_first"}'
    result = ForwardPaperEngine().run_update(active_strategy=payload, provider=provider)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == 105.0


def test_forward_state_persistence_and_equity_curve(monkeypatch, tmp_path: Path):
    install_scheduled_strategy(monkeypatch)
    db = TradingLabDatabase(str(tmp_path / "forward_state.duckdb"))
    bars = make_bars(
        "AAA",
        opens=[100, 100, 101, 103, 104],
        highs=[101, 101, 103, 104, 105],
        lows=[99, 99, 100, 102, 103],
        closes=[100, 100, 102, 103, 104],
    )
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")})
    payload = make_active_payload()
    db.insert_active_paper_strategy(payload)
    result = ForwardPaperEngine().run_update(active_strategy=payload, provider=provider)
    db.replace_forward_paper_state(payload["active_strategy_id"], result.orders, result.positions, result.trades, result.equity_curve)
    db.replace_forward_engine_events(payload["active_strategy_id"], result.events)
    assert not db.read_forward_paper_orders(payload["active_strategy_id"]).empty
    assert not db.read_forward_paper_positions(payload["active_strategy_id"]).empty
    assert not db.read_forward_paper_equity_curve(payload["active_strategy_id"]).empty
    assert not db.read_active_paper_strategy_events(payload["active_strategy_id"]).empty


def test_forward_engine_stale_data_skip_behavior(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100, 100, 101],
        highs=[101, 101, 102],
        lows=[99, 99, 100],
        closes=[100, 100, 101],
    )
    stale_status = SimpleNamespace(cache_status="stale", validation_warnings=["missing trading sessions"])
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")}, statuses={"AAA": stale_status})
    result = ForwardPaperEngine().run_update(active_strategy=make_active_payload(), provider=provider)
    assert result.skipped is True
    assert result.orders.empty
    assert result.events.iloc[0]["event_type"] == "data_skip"


def test_promotion_checklist_and_forward_vs_backtest_warnings():
    checklist = build_promotion_checklist(
        run_record={"number_of_trades": 40, "cagr": 0.12, "excess_cagr": 0.03, "max_drawdown": -0.15},
        robustness_score=68,
        train_test_summary={"degradation": {"CAGR": -0.01}},
        walk_forward_summary={"profitable_test_fold_pct": 0.6, "consistency_score": 0.7},
        parameter_stability={"positive_return_pct": 0.6, "conclusion": "This strategy appears stable across nearby parameters."},
        benchmark_warning_count=0,
    )
    assert checklist["passed"].all()
    warnings = compare_forward_to_backtest(
        backtest_run={"cagr": 0.20, "max_drawdown": -0.10},
        forward_metrics={"CAGR": 0.02, "Max Drawdown": -0.18, "Number of Trades": 1},
        days_since_activation=10,
    )
    assert any("materially worse" in warning for warning in warnings)
    assert any("Too few forward trades" in warning for warning in warnings)


def test_forward_engine_no_signal_and_insufficient_data_edge_cases(monkeypatch):
    install_scheduled_strategy(monkeypatch)
    bars = make_bars(
        "AAA",
        opens=[100],
        highs=[101],
        lows=[99],
        closes=[100],
    )
    provider = FakeProvider({"AAA": bars, "SPY": bars.assign(symbol="SPY")})
    payload = make_active_payload(strategy_parameters_json='{"entry_indices":[],"exit_indices":[]}')
    result = ForwardPaperEngine().run_update(active_strategy=payload, provider=provider)
    assert result.orders.empty
    assert result.trades.empty
    assert result.positions.empty
    assert not result.equity_curve.empty
