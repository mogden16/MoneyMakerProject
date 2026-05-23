from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.execution import apply_slippage
from trading_lab.backtest.metrics import compute_summary_metrics
from trading_lab.data.providers.yfinance_provider import CacheStatus, YFinanceDataProvider
from trading_lab.spy_lab import prepare_spy_timeframe_bars
from trading_lab.strategies.breakout import BreakoutStrategy
from trading_lab.strategies.intraday_breakout import IntradayBreakoutStrategy
from trading_lab.strategies.intraday_pullback import IntradayPullbackStrategy
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy
from trading_lab.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading_lab.strategies.trend_filter import TrendFilterStrategy


STRATEGY_NAME_MAP = {
    "SPY 200-Day Trend Filter": "trend_filter_strategy",
    "trend_filter_strategy": "SPY 200-Day Trend Filter",
    "Moving Average Crossover": "moving_average_crossover",
    "moving_average_crossover": "Moving Average Crossover",
    "RSI Mean Reversion": "rsi_mean_reversion",
    "rsi_mean_reversion": "RSI Mean Reversion",
    "Daily Breakout": "daily_breakout",
    "daily_breakout": "Daily Breakout",
    "QQE/HMA Daily": "qqe_hma_strategy",
    "qqe_hma_strategy": "QQE/HMA Daily",
    "Daily Trend + Intraday Pullback": "intraday_pullback",
    "intraday_pullback": "Daily Trend + Intraday Pullback",
    "Daily Trend + Intraday Breakout": "intraday_breakout",
    "intraday_breakout": "Daily Trend + Intraday Breakout",
}


@dataclass
class ForwardUpdateResult:
    active_strategy_id: str
    skipped: bool
    skip_reason: str | None
    orders: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    events: pd.DataFrame
    metrics: dict[str, float | int]
    warnings: list[str]
    current_equity: float


def display_strategy_name(name: str) -> str:
    """Normalize stored strategy names into the Streamlit display names."""
    return STRATEGY_NAME_MAP.get(name, name)


def build_strategy_instance(strategy_name: str, parameters: dict[str, Any]):
    """Instantiate a supported research strategy from stored metadata."""
    normalized = display_strategy_name(strategy_name)
    if normalized == "SPY 200-Day Trend Filter":
        return TrendFilterStrategy(**parameters)
    if normalized == "Moving Average Crossover":
        return MovingAverageCrossStrategy(**parameters)
    if normalized == "RSI Mean Reversion":
        return RSIMeanReversionStrategy(**parameters)
    if normalized == "Daily Breakout":
        return BreakoutStrategy(**parameters)
    if normalized == "QQE/HMA Daily":
        return QQEHMAStrategy(**parameters)
    if normalized == "Daily Trend + Intraday Pullback":
        return IntradayPullbackStrategy(**parameters)
    if normalized == "Daily Trend + Intraday Breakout":
        return IntradayBreakoutStrategy(**parameters)
    raise ValueError(f"Unsupported active paper strategy: {strategy_name}")


def parse_strategy_parameters(raw: str | None) -> dict[str, Any]:
    """Parse a stored strategy-parameter JSON payload safely."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_active_paper_strategy_payload(
    *,
    strategy_name: str,
    strategy_parameters: dict[str, Any],
    universe_name: str,
    tickers: list[str],
    benchmark_symbol: str,
    timeframe: str = "1d",
    price_mode: str,
    initial_capital: float,
    position_sizing_method: str,
    position_sizing_value: float,
    max_positions: int,
    risk_settings: dict[str, Any],
    slippage_pct: float,
    commission_per_trade: float,
    linked_qualification_id: str | None = None,
    linked_sweep_id: str | None = None,
    linked_backtest_run_id: str | None = None,
    linked_search_run_id: str | None = None,
    linked_search_result_id: str | None = None,
    activation_reason: str = "",
    notes: str = "",
    tags: str = "",
    status: str = "draft",
) -> dict[str, Any]:
    """Create a forward-paper strategy payload with a frozen research configuration."""
    now = datetime.now(UTC).replace(tzinfo=None)
    return {
        "active_strategy_id": str(uuid4()),
        "created_at": now,
        "updated_at": now,
        "status": status,
        "strategy_name": display_strategy_name(strategy_name),
        "strategy_parameters_json": json.dumps(strategy_parameters, default=str),
        "universe_name": universe_name,
        "tickers": ",".join(tickers),
        "timeframe": timeframe,
        "benchmark_symbol": benchmark_symbol,
        "price_mode": price_mode,
        "initial_capital": float(initial_capital),
        "current_paper_equity": float(initial_capital),
        "position_sizing_method": position_sizing_method,
        "position_sizing_value": float(position_sizing_value),
        "max_positions": int(max_positions),
        "risk_settings_json": json.dumps(risk_settings, default=str),
        "slippage_pct": float(slippage_pct),
        "commission_per_trade": float(commission_per_trade),
        "linked_qualification_id": linked_qualification_id,
        "linked_sweep_id": linked_sweep_id,
        "linked_backtest_run_id": linked_backtest_run_id,
        "linked_search_run_id": linked_search_run_id,
        "linked_search_result_id": linked_search_result_id,
        "activation_reason": activation_reason,
        "notes": notes,
        "tags": tags,
    }


def build_promotion_checklist(
    *,
    run_record: dict[str, Any] | None,
    robustness_score: int | None = None,
    train_test_summary: dict[str, Any] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
    parameter_stability: dict[str, Any] | None = None,
    benchmark_warning_count: int = 0,
) -> pd.DataFrame:
    """Summarize whether a saved research result is strong enough for forward paper promotion."""
    record = run_record or {}
    number_of_trades = int(record.get("number_of_trades", 0) or 0)
    cagr = float(record.get("cagr", 0.0) or 0.0)
    excess_cagr = float(record.get("excess_cagr", 0.0) or 0.0)
    max_drawdown = abs(float(record.get("max_drawdown", 0.0) or 0.0))
    train_test_ok = True
    if train_test_summary:
        train_test_ok = float(train_test_summary.get("degradation", {}).get("CAGR", 0.0) or 0.0) >= -0.05
    walk_forward_ok = True
    if walk_forward_summary:
        walk_forward_ok = (
            float(walk_forward_summary.get("profitable_test_fold_pct", 0.0) or 0.0) >= 0.5
            and float(walk_forward_summary.get("consistency_score", 0.0) or 0.0) >= 0.5
        )
    stability_ok = True
    if parameter_stability:
        stability_ok = (
            float(parameter_stability.get("positive_return_pct", 0.0) or 0.0) >= 0.5
            and "narrow" not in str(parameter_stability.get("conclusion", "")).lower()
        )
    rows = [
        {"check": "Sufficient trade count", "passed": number_of_trades >= 30, "details": f"{number_of_trades} historical trades."},
        {"check": "Positive CAGR", "passed": cagr > 0, "details": f"Backtest CAGR: {cagr:.1%}."},
        {"check": "Positive excess CAGR", "passed": excess_cagr > 0, "details": f"Benchmark excess CAGR: {excess_cagr:.1%}."},
        {"check": "Acceptable max drawdown", "passed": max_drawdown <= 0.25, "details": f"Max drawdown: {max_drawdown:.1%}."},
        {"check": "Robustness score", "passed": int(robustness_score or 0) >= 60, "details": f"Robustness Score: {int(robustness_score or 0)}/100."},
        {"check": "Train/test status", "passed": train_test_ok, "details": "Train/test degradation is acceptable." if train_test_ok else "Train/test degradation is materially worse than desired."},
        {"check": "Walk-forward status", "passed": walk_forward_ok, "details": "Walk-forward consistency is acceptable." if walk_forward_ok else "Walk-forward evidence is weak or inconsistent."},
        {"check": "Parameter stability", "passed": stability_ok, "details": "Saved sweep stability looks usable." if stability_ok else "Saved sweep stability looks narrow or fragile."},
        {"check": "Benchmark/data diagnostics", "passed": benchmark_warning_count == 0, "details": "No stored benchmark warnings." if benchmark_warning_count == 0 else f"{benchmark_warning_count} benchmark/data warnings were saved."},
    ]
    return pd.DataFrame(rows)


def compare_forward_to_backtest(
    *,
    backtest_run: dict[str, Any] | None,
    forward_metrics: dict[str, float | int],
    days_since_activation: int,
) -> list[str]:
    """Generate plain-English forward validation warnings versus the original research result."""
    warnings: list[str] = []
    if backtest_run is None:
        warnings.append("No linked backtest run is available, so forward-paper performance cannot be benchmarked against original research yet.")
        return warnings
    if days_since_activation < 20:
        warnings.append("The forward-paper period is still short. Treat early performance cautiously.")
    if int(forward_metrics.get("Number of Trades", 0) or 0) < 5:
        warnings.append("Too few forward trades have closed to judge the strategy yet.")
    backtest_cagr = float(backtest_run.get("cagr", 0.0) or 0.0)
    backtest_drawdown = abs(float(backtest_run.get("max_drawdown", 0.0) or 0.0))
    if float(forward_metrics.get("CAGR", 0.0) or 0.0) < backtest_cagr - 0.10:
        warnings.append("Forward-paper CAGR is materially worse than the original backtest expectation.")
    if abs(float(forward_metrics.get("Max Drawdown", 0.0) or 0.0)) > max(backtest_drawdown * 1.25, 0.05):
        warnings.append("Forward-paper drawdown has exceeded the expected drawdown range from the backtest.")
    if int(forward_metrics.get("Number of Trades", 0) or 0) == 0:
        warnings.append("The strategy has not generated any closed forward trades yet.")
    return warnings


class ForwardPaperEngine:
    """Replay daily-bar signals from an activation date into a deterministic forward-paper state."""

    def __init__(self, warmup_days: int = 400) -> None:
        self.warmup_days = warmup_days
        self.backtest_engine = BacktestEngine(database=None)

    def run_update(
        self,
        *,
        active_strategy: dict[str, Any],
        provider: YFinanceDataProvider,
        end_date: str | None = None,
    ) -> ForwardUpdateResult:
        tickers = [item.strip().upper() for item in str(active_strategy.get("tickers") or "").split(",") if item.strip()]
        benchmark_symbol = str(active_strategy.get("benchmark_symbol") or "SPY").upper()
        timeframe = str(active_strategy.get("timeframe") or "1d")
        if not tickers:
            return self._skip_result(str(active_strategy["active_strategy_id"]), "No tickers were configured for this active paper strategy.")

        created_at = pd.Timestamp(active_strategy["created_at"]).tz_localize(None) if pd.Timestamp(active_strategy["created_at"]).tzinfo else pd.Timestamp(active_strategy["created_at"])
        activation_date = created_at.normalize()
        end_ts = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.today().normalize()
        warmup_start = (activation_date - pd.Timedelta(days=self.warmup_days)).date()
        fetch_end = end_ts.date()

        data_by_symbol: dict[str, pd.DataFrame] = {}
        statuses: dict[str, CacheStatus] = {}
        warnings: list[str] = []
        for symbol in list(dict.fromkeys(tickers + [benchmark_symbol])):
            bars = provider.get_stock_bars(
                symbol=symbol,
                start_date=str(warmup_start),
                end_date=str(fetch_end),
                timeframe=timeframe,
                force_refresh=False,
            )
            data_by_symbol[symbol] = bars
            status = provider.get_last_fetch_status(symbol)
            if status is not None:
                statuses[symbol] = status

        severe_symbols = [
            symbol
            for symbol in tickers
            if symbol in statuses and (statuses[symbol].cache_status == "stale" or self._has_severe_warning(statuses[symbol].validation_warnings))
        ]
        if severe_symbols:
            reason = f"Skipped forward-paper update because data was stale or invalid for: {', '.join(severe_symbols)}."
            return self._skip_result(str(active_strategy["active_strategy_id"]), reason)

        strategy = build_strategy_instance(
            str(active_strategy["strategy_name"]),
            parse_strategy_parameters(active_strategy.get("strategy_parameters_json")),
        )
        price_mode = str(active_strategy.get("price_mode") or "raw_price_mode")
        risk_settings = parse_strategy_parameters(active_strategy.get("risk_settings_json"))
        fill_rule = str(risk_settings.get("fill_rule", "next_open"))
        ambiguity_rule = str(risk_settings.get("same_bar_stop_target_rule", "conservative_stop_first"))
        end_of_day_exit = bool(risk_settings.get("end_of_day_exit", timeframe != "1d"))
        allow_overnight = bool(risk_settings.get("allow_overnight", timeframe == "1d"))

        prepared_input = {symbol: data_by_symbol[symbol] for symbol in tickers}
        if timeframe != "1d" and benchmark_symbol in data_by_symbol:
            daily_start = str((activation_date - pd.Timedelta(days=self.warmup_days)).date())
            daily_bars = provider.get_stock_bars(symbol=benchmark_symbol, start_date=daily_start, end_date=str(fetch_end), timeframe="1d", force_refresh=False)
            prepared_input = {
                symbol: prepare_spy_timeframe_bars(primary_bars=prepared_input[symbol], timeframe=timeframe, daily_bars=daily_bars)
                for symbol in prepared_input
            }
        prepared = self._prepare_frames(prepared_input, strategy, price_mode)
        benchmark_curve = self.backtest_engine._build_benchmark_curve(  # noqa: SLF001
            data_by_symbol,
            benchmark_symbol,
            float(active_strategy.get("initial_capital", 0.0) or 0.0),
            sorted(set().union(*[set(pd.to_datetime(frame["timestamp"])) for frame in prepared.values()])) if prepared else [],
            BacktestConfig(
                initial_capital=float(active_strategy.get("initial_capital", 1.0) or 1.0),
                price_mode=price_mode,
                timeframe=timeframe,
            ),
        )

        order_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        current_positions: dict[str, dict[str, Any]] = {}
        cash = float(active_strategy.get("initial_capital", 0.0) or 0.0)
        all_timestamps = sorted(set().union(*[set(pd.to_datetime(frame["timestamp"])) for frame in prepared.values()])) if prepared else []

        if not all_timestamps:
            return self._skip_result(str(active_strategy["active_strategy_id"]), "No bars were available for the selected strategy universe.")

        for current_ts in [ts for ts in all_timestamps if ts >= activation_date]:
            current_prices: dict[str, float] = {}
            for symbol, frame in prepared.items():
                indexed = frame.set_index(pd.to_datetime(frame["timestamp"]))
                if current_ts not in indexed.index:
                    continue
                local_idx = indexed.index.get_loc(current_ts)
                bar = indexed.loc[current_ts]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]
                current_prices[symbol] = float(self.backtest_engine._mark_price(bar, price_mode))  # noqa: SLF001
                prev_bar = indexed.iloc[local_idx - 1] if local_idx > 0 else None
                prev_ts = indexed.index[local_idx - 1] if local_idx > 0 else None

                if symbol in current_positions:
                    position = current_positions[symbol]
                    position["latest_price"] = current_prices[symbol]
                    max_holding_days = getattr(strategy, "max_holding_days", None)
                    signal_exit_due = bool(prev_bar.get("exit_signal", False)) if prev_bar is not None and prev_ts is not None and prev_ts >= activation_date else False
                    holding_exit_due = bool(max_holding_days and position["holding_days"] >= int(max_holding_days))
                    if signal_exit_due or holding_exit_due:
                        exit_reason = "signal_exit" if signal_exit_due else "max_holding_days"
                        exit_price = self._entry_fill_price(fill_rule, bar, side="sell", slippage_pct=float(active_strategy.get("slippage_pct", 0.0) or 0.0))
                        self._close_position(
                            active_strategy_id=str(active_strategy["active_strategy_id"]),
                            position=position,
                            exit_timestamp=current_ts,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            commission=float(active_strategy.get("commission_per_trade", 0.0) or 0.0),
                            cash_ref={"cash": cash},
                            position_rows=position_rows,
                            trade_rows=trade_rows,
                            event_rows=event_rows,
                        )
                        cash = float(position.pop("_cash_after_close"))
                        current_positions.pop(symbol, None)
                        continue

                    risk_exit = self._evaluate_exit_on_bar(position, bar, ambiguity_rule)
                    if risk_exit is not None:
                        self._close_position(
                            active_strategy_id=str(active_strategy["active_strategy_id"]),
                            position=position,
                            exit_timestamp=current_ts,
                            exit_price=float(risk_exit["exit_price"]),
                            exit_reason=str(risk_exit["exit_reason"]),
                            commission=float(active_strategy.get("commission_per_trade", 0.0) or 0.0),
                            cash_ref={"cash": cash},
                            position_rows=position_rows,
                            trade_rows=trade_rows,
                            event_rows=event_rows,
                        )
                        cash = float(position.pop("_cash_after_close"))
                        current_positions.pop(symbol, None)
                        continue

                    if timeframe != "1d" and end_of_day_exit and not allow_overnight and self._is_session_end(current_ts, all_timestamps):
                        exit_price = apply_slippage(float(bar["close"]), float(active_strategy.get("slippage_pct", 0.0) or 0.0), "sell")
                        self._close_position(
                            active_strategy_id=str(active_strategy["active_strategy_id"]),
                            position=position,
                            exit_timestamp=current_ts,
                            exit_price=exit_price,
                            exit_reason="end_of_day_exit",
                            commission=float(active_strategy.get("commission_per_trade", 0.0) or 0.0),
                            cash_ref={"cash": cash},
                            position_rows=position_rows,
                            trade_rows=trade_rows,
                            event_rows=event_rows,
                        )
                        cash = float(position.pop("_cash_after_close"))
                        current_positions.pop(symbol, None)
                        continue

                    position["holding_days"] += 1
                    position["highest_close"] = max(float(position["highest_close"]), current_prices[symbol])
                    if position.get("trailing_stop_pct"):
                        position["current_stop"] = float(position["highest_close"]) * (1 - float(position["trailing_stop_pct"]))

                if symbol in current_positions or prev_bar is None or prev_ts is None or prev_ts < activation_date:
                    continue
                if not bool(prev_bar.get("entry_signal", False)):
                    continue

                order_id = str(uuid4())
                planned_entry_reference = self._entry_reference(fill_rule, bar)
                shares = self._calculate_shares(
                    strategy_payload=active_strategy,
                    current_cash=cash,
                    current_prices=current_prices,
                    open_positions=current_positions,
                    entry_price=planned_entry_reference,
                    stop_loss_pct=float(risk_settings.get("stop_loss_pct") or 0.0),
                )
                order_row = {
                    "order_id": order_id,
                    "active_strategy_id": str(active_strategy["active_strategy_id"]),
                    "created_at": prev_ts,
                    "ticker": symbol,
                    "timeframe": timeframe,
                    "order_type": "market",
                    "side": "buy",
                    "status": "pending",
                    "signal_date": prev_ts,
                    "planned_fill_date": current_ts,
                    "planned_fill_rule": fill_rule,
                    "planned_entry_reference": planned_entry_reference,
                    "stop_loss": planned_entry_reference * (1 - float(risk_settings.get("stop_loss_pct") or 0.0)) if risk_settings.get("stop_loss_pct") else None,
                    "take_profit": planned_entry_reference * (1 + float(risk_settings.get("take_profit_pct") or 0.0)) if risk_settings.get("take_profit_pct") else None,
                    "trailing_stop": float(risk_settings.get("trailing_stop_pct") or 0.0) or None,
                    "shares": shares,
                    "estimated_price": planned_entry_reference,
                    "actual_fill_date": None,
                    "actual_fill_price": None,
                    "cancel_reason": None,
                    "notes": "",
                }
                order_rows.append(order_row.copy())
                event_rows.append(self._event(str(active_strategy["active_strategy_id"]), "order_created", f"{symbol} buy order queued for the next daily fill.", {"ticker": symbol, "signal_date": str(prev_ts.date())}))

                if len(current_positions) >= int(active_strategy.get("max_positions", 0) or 0):
                    order_rows[-1]["status"] = "canceled"
                    order_rows[-1]["cancel_reason"] = "max_positions"
                    continue
                if shares <= 0:
                    order_rows[-1]["status"] = "canceled"
                    order_rows[-1]["cancel_reason"] = "insufficient_position_size"
                    continue
                fill_price = self._entry_fill_price(fill_rule, bar, side="buy", slippage_pct=float(active_strategy.get("slippage_pct", 0.0) or 0.0))
                total_cost = shares * fill_price + float(active_strategy.get("commission_per_trade", 0.0) or 0.0)
                if total_cost > cash:
                    order_rows[-1]["status"] = "canceled"
                    order_rows[-1]["cancel_reason"] = "insufficient_cash"
                    continue

                cash -= total_cost
                order_rows[-1]["status"] = "filled"
                order_rows[-1]["actual_fill_date"] = current_ts
                order_rows[-1]["actual_fill_price"] = fill_price
                current_positions[symbol] = {
                    "position_id": str(uuid4()),
                    "active_strategy_id": str(active_strategy["active_strategy_id"]),
                    "ticker": symbol,
                    "timeframe": timeframe,
                    "strategy_name": str(active_strategy["strategy_name"]),
                    "entry_signal_date": prev_ts,
                    "entry_date": current_ts,
                    "entry_price": fill_price,
                    "shares": shares,
                    "stop_loss": order_rows[-1]["stop_loss"],
                    "take_profit": order_rows[-1]["take_profit"],
                    "trailing_stop": order_rows[-1]["trailing_stop"],
                    "trailing_stop_pct": float(risk_settings.get("trailing_stop_pct") or 0.0),
                    "current_stop": order_rows[-1]["stop_loss"],
                    "status": "open",
                    "exit_signal_date": None,
                    "exit_date": None,
                    "exit_price": None,
                    "exit_reason": None,
                    "realized_pnl": 0.0,
                    "realized_return_pct": 0.0,
                    "realized_r_multiple": 0.0,
                    "notes": "",
                    "highest_close": current_prices[symbol],
                    "holding_days": 0,
                }
                event_rows.append(self._event(str(active_strategy["active_strategy_id"]), "order_filled", f"{symbol} buy order filled in forward paper trading.", {"ticker": symbol, "fill_price": fill_price, "shares": shares}))

            positions_value = self._positions_value(current_positions, current_prices)
            equity_rows.append(
                {
                    "active_strategy_id": str(active_strategy["active_strategy_id"]),
                    "timestamp": current_ts,
                    "equity": cash + positions_value,
                    "cash": cash,
                    "positions_value": positions_value,
                    "drawdown": 0.0,
                }
            )

        last_timestamp = max(all_timestamps)
        for symbol, frame in prepared.items():
            indexed = frame.set_index(pd.to_datetime(frame["timestamp"]))
            if symbol in current_positions:
                continue
            if last_timestamp not in indexed.index:
                continue
            bar = indexed.loc[last_timestamp]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            local_idx = indexed.index.get_loc(last_timestamp)
            if local_idx < 0 or not bool(bar.get("entry_signal", False)):
                continue
            planned_fill_date = last_timestamp + pd.offsets.BDay(1)
            order_rows.append(
                {
                    "order_id": str(uuid4()),
                    "active_strategy_id": str(active_strategy["active_strategy_id"]),
                    "created_at": last_timestamp,
                    "ticker": symbol,
                    "timeframe": timeframe,
                    "order_type": "market",
                    "side": "buy",
                    "status": "pending",
                    "signal_date": last_timestamp,
                    "planned_fill_date": planned_fill_date,
                    "planned_fill_rule": fill_rule,
                    "planned_entry_reference": self._entry_reference(fill_rule, bar),
                    "stop_loss": None,
                    "take_profit": None,
                    "trailing_stop": float(risk_settings.get("trailing_stop_pct") or 0.0) or None,
                    "shares": 0,
                    "estimated_price": self._entry_reference(fill_rule, bar),
                    "actual_fill_date": None,
                    "actual_fill_price": None,
                    "cancel_reason": None,
                    "notes": "Signal is waiting for the next daily bar to fill.",
                }
            )

        for position in current_positions.values():
            position_rows.append({key: value for key, value in position.items() if key not in {"highest_close", "holding_days", "latest_price", "trailing_stop_pct"}})

        equity_curve = pd.DataFrame(equity_rows)
        if not equity_curve.empty:
            equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["equity"].cummax() - 1

        forward_trades = pd.DataFrame(trade_rows)
        metric_trades = self._metric_trade_frame(forward_trades)
        benchmark_curve = benchmark_curve[benchmark_curve["timestamp"].isin(equity_curve["timestamp"])] if not benchmark_curve.empty and not equity_curve.empty else pd.DataFrame()
        metrics = compute_summary_metrics(
            equity_curve,
            metric_trades,
            float(active_strategy.get("initial_capital", 0.0) or 0.0),
            benchmark_curve=benchmark_curve,
        )
        event_rows.append(
            self._event(
                str(active_strategy["active_strategy_id"]),
                "update_summary",
                "Forward-paper update completed.",
                {
                    "closed_trades": int(metrics.get("Number of Trades", 0) or 0),
                    "open_positions": len(current_positions),
                    "pending_orders": int((pd.DataFrame(order_rows)["status"] == "pending").sum()) if order_rows else 0,
                    "ending_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(active_strategy.get("initial_capital", 0.0) or 0.0),
                },
            )
        )

        return ForwardUpdateResult(
            active_strategy_id=str(active_strategy["active_strategy_id"]),
            skipped=False,
            skip_reason=None,
            orders=pd.DataFrame(order_rows),
            positions=pd.DataFrame(position_rows),
            trades=forward_trades,
            equity_curve=equity_curve,
            events=pd.DataFrame(event_rows),
            metrics=metrics,
            warnings=warnings,
            current_equity=float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(active_strategy.get("initial_capital", 0.0) or 0.0),
        )

    def _prepare_frames(self, data_by_symbol: dict[str, pd.DataFrame], strategy, price_mode: str) -> dict[str, pd.DataFrame]:
        return self.backtest_engine._prepare_frames(data_by_symbol, strategy, price_mode)  # noqa: SLF001

    def _is_session_end(self, current_ts: pd.Timestamp, all_timestamps: list[pd.Timestamp]) -> bool:
        current_idx = all_timestamps.index(current_ts)
        if current_idx >= len(all_timestamps) - 1:
            return True
        return pd.Timestamp(all_timestamps[current_idx + 1]).date() != pd.Timestamp(current_ts).date()

    def _has_severe_warning(self, warnings: list[str]) -> bool:
        severe_tokens = ["required columns", "high/low", "volume", "negative", "missing trading sessions", "missing intraday bars"]
        lowered = " ".join(str(item).lower() for item in warnings)
        return any(token in lowered for token in severe_tokens)

    def _entry_reference(self, fill_rule: str, bar: pd.Series) -> float:
        if fill_rule == "next_close":
            return float(bar["close"])
        return float(bar["open"])

    def _entry_fill_price(self, fill_rule: str, bar: pd.Series, *, side: str, slippage_pct: float) -> float:
        if fill_rule == "next_close":
            return apply_slippage(float(bar["close"]), slippage_pct, side)
        return apply_slippage(float(bar["open"]), slippage_pct, side)

    def _calculate_shares(
        self,
        *,
        strategy_payload: dict[str, Any],
        current_cash: float,
        current_prices: dict[str, float],
        open_positions: dict[str, dict[str, Any]],
        entry_price: float,
        stop_loss_pct: float,
    ) -> int:
        method = str(strategy_payload.get("position_sizing_method") or "percent_of_portfolio")
        value = float(strategy_payload.get("position_sizing_value", 0.0) or 0.0)
        commission = float(strategy_payload.get("commission_per_trade", 0.0) or 0.0)
        current_equity = current_cash + self._positions_value(open_positions, current_prices)
        if method == "fixed_dollar":
            allocation = min(value, current_cash)
        elif method == "percent_of_portfolio":
            allocation = min(current_equity * value, current_cash)
        elif method == "fixed_dollar_risk":
            risk_per_share = entry_price * stop_loss_pct if stop_loss_pct > 0 else 0.0
            return int(value / risk_per_share) if risk_per_share > 0 else 0
        else:
            allocation = min(value, current_cash)
        if allocation <= commission or entry_price <= 0:
            return 0
        return max(int((allocation - commission) / entry_price), 0)

    def _evaluate_exit_on_bar(self, position: dict[str, Any], bar: pd.Series, ambiguity_rule: str) -> dict[str, Any] | None:
        stop_price = float(position.get("current_stop") or position.get("stop_loss") or 0.0)
        target_price = float(position.get("take_profit") or 0.0)
        trailing_stop = float(position.get("current_stop") or 0.0) if position.get("trailing_stop") else 0.0
        low = float(bar["low"])
        high = float(bar["high"])
        stop_hit = stop_price > 0 and low <= stop_price
        target_hit = target_price > 0 and high >= target_price
        trailing_hit = trailing_stop > 0 and low <= trailing_stop
        if stop_hit and target_hit:
            if ambiguity_rule == "target_first":
                return {"exit_price": target_price, "exit_reason": "take_profit"}
            if ambiguity_rule == "skip_ambiguous":
                return None
            return {"exit_price": stop_price, "exit_reason": "stop_loss"}
        if stop_hit:
            return {"exit_price": stop_price, "exit_reason": "stop_loss"}
        if target_hit:
            return {"exit_price": target_price, "exit_reason": "take_profit"}
        if trailing_hit:
            return {"exit_price": trailing_stop, "exit_reason": "trailing_stop"}
        return None

    def _close_position(
        self,
        *,
        active_strategy_id: str,
        position: dict[str, Any],
        exit_timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        commission: float,
        cash_ref: dict[str, float],
        position_rows: list[dict[str, Any]],
        trade_rows: list[dict[str, Any]],
        event_rows: list[dict[str, Any]],
    ) -> None:
        proceeds = float(position["shares"]) * exit_price - commission
        cash_ref["cash"] += proceeds
        planned_risk = max((float(position.get("entry_price") or 0.0) - float(position.get("stop_loss") or 0.0)) * float(position["shares"]), 0.0)
        pnl = proceeds - (float(position["shares"]) * float(position["entry_price"]) + commission)
        closed = {
            **{key: value for key, value in position.items() if key not in {"highest_close", "holding_days", "latest_price", "trailing_stop_pct"}},
            "status": "closed",
            "exit_signal_date": exit_timestamp if exit_reason == "signal_exit" else position.get("exit_signal_date"),
            "exit_date": exit_timestamp,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "realized_pnl": pnl,
            "realized_return_pct": exit_price / float(position["entry_price"]) - 1 if float(position["entry_price"]) else 0.0,
            "realized_r_multiple": pnl / planned_risk if planned_risk > 0 else 0.0,
            "notes": "",
        }
        position_rows.append(closed)
        trade_rows.append(
            {
                "trade_id": str(uuid4()),
                "active_strategy_id": active_strategy_id,
                "ticker": position["ticker"],
                "timeframe": position.get("timeframe", "1d"),
                "strategy_name": position["strategy_name"],
                "entry_signal_date": position["entry_signal_date"],
                "entry_date": position["entry_date"],
                "entry_price": position["entry_price"],
                "exit_signal_date": closed["exit_signal_date"],
                "exit_date": exit_timestamp,
                "exit_price": exit_price,
                "shares": position["shares"],
                "exit_reason": exit_reason,
                "realized_pnl": pnl,
                "realized_return_pct": closed["realized_return_pct"],
                "realized_r_multiple": closed["realized_r_multiple"],
                "notes": "",
            }
        )
        event_rows.append(self._event(active_strategy_id, "position_closed", f"{position['ticker']} position closed in forward paper trading.", {"ticker": position["ticker"], "exit_reason": exit_reason, "exit_price": exit_price}))
        position["_cash_after_close"] = cash_ref["cash"]

    def _positions_value(self, positions: dict[str, dict[str, Any]], current_prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in positions.items():
            mark = current_prices.get(symbol, float(position.get("latest_price") or position["entry_price"]))
            total += float(position["shares"]) * mark
        return total

    def _metric_trade_frame(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(columns=["pnl", "return_pct", "holding_days"])
        metric = trades.rename(columns={"realized_pnl": "pnl", "realized_return_pct": "return_pct"}).copy()
        metric["holding_days"] = (
            pd.to_datetime(metric["exit_date"]) - pd.to_datetime(metric["entry_date"])
        ).dt.days.clip(lower=0)
        return metric

    def _event(self, active_strategy_id: str, event_type: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": str(uuid4()),
            "active_strategy_id": active_strategy_id,
            "created_at": datetime.now(UTC).replace(tzinfo=None),
            "event_type": event_type,
            "message": message,
            "details_json": json.dumps(details, default=str),
        }

    def _skip_result(self, active_strategy_id: str, reason: str) -> ForwardUpdateResult:
        return ForwardUpdateResult(
            active_strategy_id=active_strategy_id,
            skipped=True,
            skip_reason=reason,
            orders=pd.DataFrame(),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            events=pd.DataFrame([self._event(active_strategy_id, "data_skip", reason, {"reason": reason})]),
            metrics={},
            warnings=[reason],
            current_equity=0.0,
        )
