from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from trading_lab.backtest.execution import apply_slippage
from trading_lab.backtest.metrics import compute_summary_metrics
from trading_lab.backtest.portfolio import PortfolioState, Position
from trading_lab.backtest.risk import evaluate_risk_exit
from trading_lab.data.database import TradingLabDatabase
from trading_lab.strategies.base import StrategyBase


class BacktestConfig(BaseModel):
    initial_capital: float = Field(gt=0)
    slippage_pct: float = 0.0
    commission_per_trade: float = 0.0
    position_sizing_method: str = "percent_of_portfolio"
    position_size_value: float = 0.1
    max_positions: int = 5
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    return_mode: str = "price_return_only"
    price_mode: str = "raw_price_mode"
    sweep_id: str | None = None
    timeframe: str = "1d"
    end_of_day_exit: bool = False
    allow_overnight: bool = True


@dataclass
class BacktestResult:
    run_id: str
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    benchmark_curve: pd.DataFrame
    benchmark_symbol: str
    metrics: dict[str, float | int]


class BacktestEngine:
    def __init__(self, database: TradingLabDatabase | None = None) -> None:
        self.database = database

    def run(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        strategy: StrategyBase,
        config: BacktestConfig,
        benchmark_symbol: str = "SPY",
    ) -> BacktestResult:
        prepared = self._prepare_frames(data_by_symbol, strategy, config.price_mode)
        all_timestamps = sorted(set().union(*[set(pd.to_datetime(df["timestamp"])) for df in prepared.values()]))
        portfolio = PortfolioState(cash=config.initial_capital)
        trades: list[dict] = []
        equity_rows: list[dict] = []
        run_id = str(uuid4())
        signal_lookup = {symbol: frame.set_index(pd.to_datetime(frame["timestamp"])) for symbol, frame in prepared.items()}

        for date_idx, current_ts in enumerate(all_timestamps):
            current_prices: dict[str, float] = {}
            for symbol, frame in signal_lookup.items():
                if current_ts not in frame.index:
                    continue
                bar = frame.loc[current_ts]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]

                current_prices[symbol] = float(self._mark_price(bar, config.price_mode))
                if symbol in portfolio.positions:
                    self._update_position_state(portfolio.positions[symbol], bar, config.price_mode)
                    if config.return_mode == "total_return_with_dividends" and config.price_mode == "raw_price_mode":
                        dividend_cash = float(bar.get("dividends", 0.0)) * portfolio.positions[symbol].shares
                        portfolio.cash += dividend_cash
                    risk_exit = evaluate_risk_exit(
                        bar=bar,
                        entry_price=portfolio.positions[symbol].entry_price,
                        highest_close=portfolio.positions[symbol].highest_close,
                        stop_loss_pct=config.stop_loss_pct,
                        take_profit_pct=config.take_profit_pct,
                        trailing_stop_pct=config.trailing_stop_pct,
                    )
                    if risk_exit is not None:
                        self._close_position(
                            portfolio=portfolio,
                            symbol=symbol,
                            exit_timestamp=current_ts,
                            exit_price=apply_slippage(risk_exit.exit_price, config.slippage_pct, "sell"),
                            commission=config.commission_per_trade,
                            trades=trades,
                            exit_reason=risk_exit.exit_reason,
                            run_id=run_id,
                        )
                        continue
                    if config.timeframe != "1d" and config.end_of_day_exit and not config.allow_overnight and self._is_session_end(current_ts, all_timestamps, date_idx):
                        self._close_position(
                            portfolio=portfolio,
                            symbol=symbol,
                            exit_timestamp=current_ts,
                            exit_price=apply_slippage(float(bar["close"]), config.slippage_pct, "sell"),
                            commission=config.commission_per_trade,
                            trades=trades,
                            exit_reason="end_of_day_exit",
                            run_id=run_id,
                        )
                        continue

                if date_idx == 0:
                    continue
                prev_ts = all_timestamps[date_idx - 1]
                if prev_ts not in frame.index:
                    continue
                prev_bar = frame.loc[prev_ts]
                if isinstance(prev_bar, pd.DataFrame):
                    prev_bar = prev_bar.iloc[-1]

                if symbol in portfolio.positions:
                    max_holding_days = getattr(strategy, "max_holding_days", None)
                    if bool(prev_bar.get("exit_signal", False)) or (
                        max_holding_days and portfolio.positions[symbol].holding_days >= max_holding_days
                    ):
                        self._close_position(
                            portfolio=portfolio,
                            symbol=symbol,
                            exit_timestamp=current_ts,
                            exit_price=apply_slippage(float(bar["open"]), config.slippage_pct, "sell"),
                            commission=config.commission_per_trade,
                            trades=trades,
                            exit_reason="signal_exit" if bool(prev_bar.get("exit_signal", False)) else "max_holding_days",
                            run_id=run_id,
                        )
                        continue

                if symbol not in portfolio.positions and bool(prev_bar.get("entry_signal", False)):
                    if len(portfolio.positions) >= config.max_positions:
                        continue
                    shares = self._calculate_shares(
                        equity=portfolio.equity(current_prices),
                        cash=portfolio.cash,
                        open_price=float(bar["open"]),
                        config=config,
                    )
                    if shares <= 0:
                        continue
                    fill_price = apply_slippage(float(bar["open"]), config.slippage_pct, "buy")
                    total_cost = shares * fill_price + config.commission_per_trade
                    if total_cost > portfolio.cash:
                        continue
                    portfolio.cash -= total_cost
                    portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        shares=shares,
                        entry_price=fill_price,
                        entry_timestamp=current_ts,
                        highest_close=float(self._mark_price(bar, config.price_mode)),
                        entry_signal_timestamp=prev_ts,
                    )

            positions_value = portfolio.positions_value(current_prices)
            equity_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": current_ts,
                    "equity": portfolio.cash + positions_value,
                    "cash": portfolio.cash,
                    "positions_value": positions_value,
                    "drawdown": 0.0,
                }
            )

        self._liquidate_remaining_positions(portfolio, signal_lookup, config, trades, run_id, all_timestamps)
        equity_curve = pd.DataFrame(equity_rows)
        if not equity_curve.empty:
            equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["equity"].cummax() - 1
        trade_log = pd.DataFrame(trades)
        benchmark_curve = self._build_benchmark_curve(data_by_symbol, benchmark_symbol, config.initial_capital, all_timestamps, config)
        metrics = compute_summary_metrics(equity_curve, trade_log, config.initial_capital, benchmark_curve=benchmark_curve)
        self._persist_run(run_id, strategy, config, data_by_symbol, equity_curve, trade_log, benchmark_curve, metrics, benchmark_symbol)
        return BacktestResult(
            run_id=run_id,
            equity_curve=equity_curve,
            trade_log=trade_log,
            benchmark_curve=benchmark_curve,
            benchmark_symbol=benchmark_symbol,
            metrics=metrics,
        )

    def _prepare_frames(self, data_by_symbol: dict[str, pd.DataFrame], strategy: StrategyBase, price_mode: str) -> dict[str, pd.DataFrame]:
        prepared: dict[str, pd.DataFrame] = {}
        for symbol, bars in data_by_symbol.items():
            frame = bars.copy().sort_values("timestamp").reset_index(drop=True)
            frame["raw_close"] = frame["close"]
            if price_mode == "adjusted_price_mode" and "adj_close" in frame.columns:
                frame["close"] = frame["adj_close"]
            signal_frame = strategy.generate_signals(frame)
            signal_frame["symbol"] = symbol
            prepared[symbol] = signal_frame
        return prepared

    def _update_position_state(self, position: Position, bar: pd.Series, price_mode: str) -> None:
        position.holding_days += 1
        position.highest_close = max(position.highest_close, float(self._mark_price(bar, price_mode)))

    def _mark_price(self, bar: pd.Series, price_mode: str) -> float:
        if price_mode == "adjusted_price_mode" and pd.notna(bar.get("adj_close")):
            return float(bar["adj_close"])
        return float(bar["close"])

    def _calculate_shares(self, equity: float, cash: float, open_price: float, config: BacktestConfig) -> float:
        allocation = config.position_size_value if config.position_sizing_method == "fixed_dollar" else equity * config.position_size_value
        allocation = min(allocation, cash)
        if allocation <= config.commission_per_trade:
            return 0.0
        return max(int((allocation - config.commission_per_trade) / open_price), 0)

    def _close_position(
        self,
        portfolio: PortfolioState,
        symbol: str,
        exit_timestamp,
        exit_price: float,
        commission: float,
        trades: list[dict],
        exit_reason: str,
        run_id: str,
    ) -> None:
        position = portfolio.positions.pop(symbol)
        proceeds = position.shares * exit_price - commission
        portfolio.cash += proceeds
        pnl = proceeds - (position.shares * position.entry_price + commission)
        trades.append(
            {
                "run_id": run_id,
                "symbol": symbol,
                "entry_timestamp": position.entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "shares": position.shares,
                "pnl": pnl,
                "return_pct": exit_price / position.entry_price - 1,
                "holding_days": position.holding_days,
                "exit_reason": exit_reason,
            }
        )

    def _liquidate_remaining_positions(
        self,
        portfolio: PortfolioState,
        signal_lookup: dict[str, pd.DataFrame],
        config: BacktestConfig,
        trades: list[dict],
        run_id: str,
        all_timestamps: list[pd.Timestamp],
    ) -> None:
        if not portfolio.positions:
            return
        last_prices = {symbol: float(self._mark_price(frame.iloc[-1], config.price_mode)) for symbol, frame in signal_lookup.items()}
        final_ts = all_timestamps[-1] if all_timestamps else pd.Timestamp.utcnow().tz_localize(None)
        for symbol in list(portfolio.positions.keys()):
            self._close_position(
                portfolio=portfolio,
                symbol=symbol,
                exit_timestamp=final_ts,
                exit_price=apply_slippage(last_prices[symbol], config.slippage_pct, "sell"),
                commission=config.commission_per_trade,
                trades=trades,
                exit_reason="end_of_test",
                run_id=run_id,
            )

    def _build_benchmark_curve(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        benchmark_symbol: str,
        initial_capital: float,
        all_timestamps: list[pd.Timestamp],
        config: BacktestConfig,
    ) -> pd.DataFrame:
        if benchmark_symbol not in data_by_symbol:
            return pd.DataFrame()
        benchmark = data_by_symbol[benchmark_symbol].copy().sort_values("timestamp")
        benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"])
        price_column = "adj_close" if config.price_mode == "adjusted_price_mode" and "adj_close" in benchmark.columns else "close"
        aligned = benchmark.set_index("timestamp")[price_column].reindex(all_timestamps).ffill().dropna()
        if aligned.empty:
            return pd.DataFrame()
        result = aligned.to_frame("price").reset_index().rename(columns={"index": "timestamp"})
        result["benchmark_equity"] = initial_capital * (result["price"] / result["price"].iloc[0])
        result["run_id"] = None
        result["benchmark_symbol"] = benchmark_symbol
        return result[["run_id", "benchmark_symbol", "timestamp", "benchmark_equity"]]

    def _persist_run(
        self,
        run_id: str,
        strategy: StrategyBase,
        config: BacktestConfig,
        data_by_symbol: dict[str, pd.DataFrame],
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        benchmark_curve: pd.DataFrame,
        metrics: dict[str, float | int],
        benchmark_symbol: str,
    ) -> None:
        if self.database is None:
            return
        summary = {
            "run_id": run_id,
            "strategy_name": strategy.name,
            "parameters_json": json.dumps({**strategy.parameters(), **config.model_dump()}, default=str),
            "symbols_csv": ",".join(data_by_symbol.keys()),
            "start_date": min(pd.to_datetime(frame["session_date"]).min().date() for frame in data_by_symbol.values()),
            "end_date": max(pd.to_datetime(frame["session_date"]).max().date() for frame in data_by_symbol.values()),
            "created_at": pd.Timestamp.now("UTC").tz_localize(None),
            "initial_capital": config.initial_capital,
            "timeframe": config.timeframe,
            "total_return": metrics.get("Total Return", 0.0),
            "cagr": metrics.get("CAGR", 0.0),
            "max_drawdown": metrics.get("Max Drawdown", 0.0),
            "sharpe_ratio": metrics.get("Sharpe Ratio", 0.0),
            "sortino_ratio": metrics.get("Sortino Ratio", 0.0),
            "calmar_ratio": metrics.get("Calmar Ratio", 0.0),
            "win_rate": metrics.get("Win Rate", 0.0),
            "profit_factor": metrics.get("Profit Factor", 0.0),
            "exposure_pct": metrics.get("Exposure %", 0.0),
            "number_of_trades": int(metrics.get("Number of Trades", 0)),
            "benchmark_symbol": benchmark_symbol,
            "benchmark_total_return": metrics.get("Benchmark Total Return", 0.0),
            "benchmark_cagr": metrics.get("Benchmark CAGR", 0.0),
            "benchmark_max_drawdown": metrics.get("Benchmark Max Drawdown", 0.0),
            "excess_cagr": metrics.get("Excess CAGR", 0.0),
            "beta": metrics.get("Beta", 0.0),
            "correlation": metrics.get("Correlation", 0.0),
            "return_mode": config.return_mode,
            "price_mode": config.price_mode,
            "sweep_id": config.sweep_id,
            "notes": "",
            "tags": "",
        }
        self.database.insert_backtest_run(summary)
        self.database.insert_backtest_trades(trade_log)
        self.database.insert_backtest_equity_curve(equity_curve)
        if not benchmark_curve.empty:
            stored = benchmark_curve.copy()
            stored["run_id"] = run_id
            self.database.insert_backtest_benchmark_curve(stored)

    def _is_session_end(self, current_ts: pd.Timestamp, all_timestamps: list[pd.Timestamp], date_idx: int) -> bool:
        if date_idx >= len(all_timestamps) - 1:
            return True
        return pd.Timestamp(all_timestamps[date_idx + 1]).date() != pd.Timestamp(current_ts).date()
